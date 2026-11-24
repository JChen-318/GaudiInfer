// g2e_server: OpenAI-compatible HTTP server (the request/response format of vLLM's OpenAI entrypoints) for
// Qwen3.8-27B on one Gaudi2, with DSpine speculative decoding (greedy, and sampled via rejection sampling).
//
//   g2e_server --model-dir DIR --target-pack T.g2w [--draft-pack D.g2w] [--vision-pack V.g2w] [--served-model-name NAME]
//              [--host 0.0.0.0] [--port 8000] [--max-model-len 32768] [--slots 8] [--kv-cache-gib G]
//              [--reasoning-parser qwen3] [--max-image-pixels N] [--api-key KEY]
//
// Endpoints: GET /health, /v1/models, /version, /stats; POST /v1/chat/completions, /v1/completions (stream or not).
// Up to --slots requests are resident at once (dynamic KV slots within one budget); the scheduler thread runs every
// decoding request's speculative round (batched across requests), then one prefill segment of the oldest prompt.
#include "g2e/dspine.hpp"
#include "g2e/generator.hpp"
#include "g2e/inkling2.hpp"
#include "g2e/inkling2_dspark.hpp"
#include "g2e/inkling_parser.hpp"
#include "g2e/prefix_cache.hpp"
#include "g2e/vision.hpp"
#include "g2e/inkling_vision.hpp"
#include "g2e/qwen3_parser.hpp"
#include "g2e/qwen35.hpp"
#include "g2e/tokenizer.hpp"
#include "g2e/weights.hpp"

#include "httplib/httplib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <deque>
#include <condition_variable>
#include <random>
#include <sstream>

using namespace g2e;
using ojson = nlohmann::ordered_json;

namespace {

struct Options {
    std::string model_dir, target_pack, draft_pack, served_name, host = "0.0.0.0", api_key, reasoning_parser;
    int port = 8000;
    uint32_t max_model_len = 32768;
    std::string vision_pack;        // image input (vision encoder pack); none: text only
    uint64_t max_image_pixels = 0;  // 0: the processor's limit
    uint32_t slots = 1;             // requests resident at once (model slots), interleaved step by step
    double kv_gib = 0;              // KV cache budget of all slots; 0: the device memory left after loading, less 4 GiB
};

struct HttpError : std::runtime_error {
    int status;
    std::string type;
    HttpError(int s, const std::string& msg, std::string t = "BadRequestError") : std::runtime_error(msg), status(s), type(std::move(t)) {}
};

std::string error_json(const std::string& msg, const std::string& type, int code) {
    ojson e;
    e["error"] = {{"message", msg}, {"type", type}, {"param", nullptr}, {"code", code}};
    return e.dump();
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

// ---------------------------------------------------------------------------------------------------------------
struct Server {
    Options opt;
    std::unique_ptr<WeightPack> target, draft;
    std::unique_ptr<Device> dev;
    std::unique_ptr<Stream> stream;
    std::unique_ptr<Qwen35> model;
    std::unique_ptr<DSpine> drafter;
    std::unique_ptr<Generator> gen;
    std::unique_ptr<PrefixCache> prefix;
    int32_t im_start = -1;  // the chat template's message start token
    std::unique_ptr<WeightPack> vpack;
    std::unique_ptr<Vision> vision;
    std::unique_ptr<DeviceBuffer> image_feats;  // features of the current request's images
    std::unique_ptr<Tokenizer> tok;
    std::unique_ptr<ChatTemplate> tpl;
    ojson gen_defaults = ojson::object();  // generation_config.json sampling defaults (vLLM's override-generation-config)
    std::vector<int32_t> eos;
    // Inkling-2 (model_type inkling2_loop packs): up to 8 concurrent dynamic-KV slots (1M ctx), Bp in {2,4,8} batched Direct decode
    bool inkling = false;
    std::unique_ptr<Inkling2> ink;
    std::unique_ptr<InklingVision> ink_vision;  // Inkling image input (--vision-pack with an inkling2_vision pack)
    std::unique_ptr<InkDSpark> ink_drafter;     // DSpark speculative decoding for solo greedy & sampled requests (--draft-pack)
    int ink_taps_owner = -1;                    // slot whose latest taps currently live in ink->taps()
    // adaptive speculation: separate decayed acceptance counts & round costs (ms) for greedy vs sampled requests
    double spec_succ_g = 2.6, spec_tri_g = 4.0, spec_cost_g[5] = {0, 8.6, 13.5, 15.7, 17.3};
    double spec_succ_s = 2.4, spec_tri_s = 4.0, spec_cost_s[5] = {0, 9.2, 14.5, 16.5, 18.0};
    uint32_t ink_max_image_tokens = 4096;
    std::string ink_bf16_mode;
    uint32_t ink_bf16_cached = 0, ink_bf16_total = 0;
    bool ink_bf16_staging = false;
    uint64_t ink_bf16_bytes = 0, ink_kv_budget_tokens = 0, ink_dspark_kv_reserve = 0;
    struct Job;
    struct InkSlot {
        enum Phase : uint8_t { Idle = 0, Prefill = 1, Decode = 2 };
        Phase phase = Idle;
        std::shared_ptr<Job> job;
        std::vector<int32_t> history;
        uint64_t salt = 0;
        bool history_draft = false;
        uint64_t used = 0;
        GenerationResult r;
        std::chrono::steady_clock::time_point t0{}, t1{};
        size_t at = 0, pi = 0;
        std::vector<size_t> points;
        bool spec = false;
        uint32_t last_rows = 0;
        uint64_t tap_row_off = 0;
        uint32_t plain_left = 0;
        bool probe = false;
        int32_t pending_tok = 0;
        std::mt19937_64 rng;
        std::unique_ptr<DeviceBuffer> image_feats;
        struct PendingImg { uint32_t pos, rows; uint64_t off; };
        std::vector<PendingImg> pending_imgs;
    };
    std::vector<InkSlot> ink_slots;
    // state snapshots in pinned host memory (zero HBM pressure, fast PCIe DMA): LRU within G2E_INK_SNAPSHOT_GB (default 32)
    struct InkSnap {
        std::vector<int32_t> tokens;
        uint64_t salt = 0, used = 0, bytes = 0, draft_off = 0;  // draft_off: the drafter's context rows after the state
        bool has_draft = false;
        std::vector<float> last_logits;     // populated at prompt end for 100% exact-prompt reuse
        std::vector<uint16_t> last_taps;    // [4 * H] bf16 taps of row n-1 (for exact-prompt reuse with DSpark)
        std::unique_ptr<HostBuffer> buf;
    };
    std::vector<InkSnap> ink_snaps;
    uint64_t ink_snap_budget = 0, ink_snap_tick = 0;
    std::mutex engine;
    int64_t started = std::time(nullptr);
    // ---- scheduler: one engine thread, requests on model slots, one step (a prefill segment or a decoding
    // round) per running request in turn
    struct Job {
        std::vector<int32_t> prompt;
        SamplingParams sp;
        std::vector<VisionImage> images;
        std::mutex mu;
        std::condition_variable cv;
        std::vector<int32_t> tokens;  // delivered, not taken by the request thread yet
        bool done = false, cancel = false;
        GenerationResult gen;
        std::string error;
    };
    std::vector<std::unique_ptr<Generator>> gens;    // one per model slot
    std::vector<std::shared_ptr<Job>> running;      // per slot
    std::vector<uint64_t> slot_used;                // last use (slot choice among equals)
    std::deque<std::shared_ptr<Job>> queue;
    std::mutex sched_mu;
    std::condition_variable sched_cv;
    std::thread sched_thread;
    uint64_t sched_tick = 0;
    uint64_t kv_budget = 0;                      // device bytes for the KV caches of all slots
    std::atomic<uint64_t> kv_used{0}, kv_evictions{0}, device_free_bytes{0}, device_total_bytes{0};
    std::atomic<uint32_t> slots_active{0}, slots_decoding{0}, slots_prefilling{0};
    std::atomic<uint64_t> live_decode_tokens{0}, live_decode_rounds{0}, live_prefill_tokens{0};
    static constexpr uint32_t kPrefillStep = 2048;      // concurrent prefill step: caps decode inter-token stall
    static constexpr uint32_t kPrefillStepSolo = 4096;  // solo prefill step: unlocks full T=4096 prefill throughput
    std::mutex stats_mu;
    PrefixCache::Stats prefix_stats;

    static constexpr uint64_t kMarginBytes = 4ull << 30;  // device memory kept free for workspace growth
    size_t queued() {
        std::lock_guard lk(sched_mu);
        return queue.size();
    }
    void submit(std::shared_ptr<Job> job) {
        {
            std::lock_guard lk(sched_mu);
            queue.push_back(std::move(job));
        }
        sched_cv.notify_one();
    }
    // a free slot for the job: the one whose history it extends the most, else one without KV caches, else the least
    // recently used (its state is in the prefix cache); -1: none free
    int pick_slot(const Job& job) const {
        int best = -1;
        size_t best_match = 0;
        bool best_empty = false;
        for (size_t s = 0; s < gens.size(); ++s) {
            if (running[s]) continue;
            const auto& h = gens[s]->history();
            const size_t match = job.images.empty() && h.size() < job.prompt.size() &&
                                 std::equal(h.begin(), h.end(), job.prompt.begin()) ? h.size() : 0;
            const bool empty = model->slot_kv(uint32_t(s)) == 0;
            const bool better = best < 0 || match > best_match ||
                                (match == best_match && (empty > best_empty ||
                                                         (empty == best_empty && slot_used[s] < slot_used[size_t(best)])));
            if (better) {
                best = int(s);
                best_match = match;
                best_empty = empty;
            }
        }
        return best;
    }
    // dynamic KV: make room for `extra` more bytes on `slot` by freeing idle slots, least recently used first
    bool reserve_kv(uint32_t slot, uint64_t extra) {
        while (model->kv_total() + extra > kv_budget) {
            int v = -1;
            for (size_t s = 0; s < gens.size(); ++s)
                if (s != slot && !running[s] && model->slot_kv(uint32_t(s)) &&
                    (v < 0 || slot_used[s] < slot_used[size_t(v)]))
                    v = int(s);
            if (v < 0) return false;
            gens[size_t(v)]->release();
            ++kv_evictions;
        }
        return true;
    }
    // the KV bytes a job takes at its start (Generator::begin)
    uint64_t kv_estimate(const Job& job) const {
        const uint32_t P = uint32_t(job.prompt.size());
        const uint32_t want = std::min<uint32_t>(opt.max_model_len, P + std::min<uint32_t>(job.sp.max_tokens, 2048) + 17);
        return model->kv_bytes_for(model->round_capacity(want));
    }
    void begin_job(size_t s, const std::shared_ptr<Job>& job) {
        Generator& g = *gens[s];
        // images: features of every image, then one span per placeholder run (in prompt order); the image prefill
        // runs in the generator's first step, right below (the feature buffer is shared)
        std::vector<Qwen35::ImageSpan> spans;
        const auto& prompt = job->prompt;
        if (!job->images.empty()) {
            const auto& images = job->images;
            uint64_t rows = 0;
            for (const auto& im : images) rows += im.tokens();
            const uint64_t bytes = rows * vision->out_hidden() * 2;
            if (!image_feats || image_feats->size() < bytes) image_feats = std::make_unique<DeviceBuffer>(*dev, bytes);
            uint64_t at = 0;
            size_t k = 0;
            for (size_t i = 0; i < prompt.size() && k < images.size();) {
                if (prompt[i] != vision->image_token()) { ++i; continue; }
                const auto& im = images[k++];
                vision->encode(im, *image_feats, at * vision->out_hidden() * 2);
                spans.push_back({uint32_t(i), im.grid_h / 2, im.grid_w / 2, image_feats.get(), at * vision->out_hidden() * 2});
                at += im.tokens();
                i += im.tokens();
            }
        }
        if (prefix && im_start >= 0) {  // snapshots at the last two message starts (last user message, assistant header)
            std::vector<uint32_t> points;
            for (size_t i = prompt.size(); i-- > 0 && points.size() < 2;)
                if (prompt[i] == im_start) points.push_back(uint32_t(i));
            g.set_snapshot_points(std::move(points));
        }
        Job* jp = job.get();
        g.start(prompt, job->sp, [this, jp](std::span<const int32_t> toks) {
            if (!toks.empty()) {
                live_decode_tokens.fetch_add(toks.size(), std::memory_order_relaxed);
                live_decode_rounds.fetch_add(1, std::memory_order_relaxed);
            }
            std::lock_guard lk(jp->mu);
            jp->tokens.insert(jp->tokens.end(), toks.begin(), toks.end());
            jp->cv.notify_one();
            return !jp->cancel;
        }, spans);
        running[s] = job;
        slot_used[s] = ++sched_tick;
        if (!spans.empty()) step_job(s);  // the image prefill while the features are in place
    }
    void finish_job(size_t s, GenerationResult gen, std::string error) {
        auto job = std::move(running[s]);
        running[s].reset();
        {
            std::lock_guard lk(job->mu);
            job->gen = std::move(gen);
            job->error = std::move(error);
            job->done = true;
            job->cv.notify_one();
        }
    }
    void save_slot_prefix(size_t s) {
        if (!prefix) return;
        try {
            gens[s]->save_prefix();
            std::lock_guard lk(stats_mu);
            prefix_stats = prefix->stats();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[g2e] slot %zu: prefix cache save failed: %s\n", s, e.what());
        }
    }
    void step_job(size_t s, uint32_t prefill_step = kPrefillStep) {
        Generator& g = *gens[s];
        try {
            const size_t p0 = g.prefill_done();
            const bool done = g.step(prefill_step);
            const size_t p1 = g.prefill_done();
            if (p1 > p0) live_prefill_tokens.fetch_add(p1 - p0, std::memory_order_relaxed);
            if (done) {
                finish_job(s, g.take_result(), "");
                save_slot_prefix(s);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[g2e] slot %zu: %s\n", s, e.what());
            try { g.abort(); } catch (...) {}
            finish_job(s, {}, e.what());
        }
    }
    void step_batch_jobs(const std::vector<size_t>& group) {
        std::vector<Generator*> gp;
        for (size_t s : group) gp.push_back(gens[s].get());
        try {
            const auto done = Generator::step_batch(gp);
            std::vector<size_t> to_save;
            for (size_t k = 0; k < group.size(); ++k)
                if (done[k]) {
                    finish_job(group[k], gens[group[k]]->take_result(), "");
                    to_save.push_back(group[k]);
                }
            for (size_t s : to_save) save_slot_prefix(s);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[g2e] batched round: %s\n", e.what());
            for (size_t s : group) {
                if (!running[s]) continue;
                try { gens[s]->abort(); } catch (...) {}
                finish_job(s, {}, e.what());
            }
        }
    }
    void scheduler() {
        while (true) {
            std::vector<std::shared_ptr<Job>> starting;
            std::vector<size_t> where;
            {
                std::unique_lock lk(sched_mu);
                sched_cv.wait(lk, [&] {
                    if (!queue.empty()) return true;
                    for (const auto& r : running) if (r) return true;
                    return false;
                });
                uint64_t held = 0;  // KV of the running requests (and of those admitted now)
                for (size_t s = 0; s < gens.size(); ++s)
                    if (running[s]) held += model->slot_kv(uint32_t(s));
                while (!queue.empty()) {
                    auto& job = queue.front();
                    const uint64_t need = kv_estimate(*job);
                    if (need > kv_budget) {  // never fits
                        std::lock_guard jl(job->mu);
                        job->error = "not enough device memory for the KV cache of this request";
                        job->done = true;
                        job->cv.notify_one();
                        queue.pop_front();
                        continue;
                    }
                    if (held + need > kv_budget) break;  // wait for running requests to finish
                    const int s = pick_slot(*job);
                    if (s < 0) break;
                    held += need;
                    running[size_t(s)] = job;  // reserved (begin_job below, outside the lock)
                    starting.push_back(job);
                    where.push_back(size_t(s));
                    queue.pop_front();
                }
            }
            for (size_t k = 0; k < starting.size(); ++k) {
                try {
                    begin_job(where[k], starting[k]);
                } catch (const std::exception& e) {
                    try { gens[where[k]]->abort(); } catch (...) {}
                    finish_job(where[k], {}, e.what());
                }
            }
            // decoding first (one round each), then one prefill segment of the oldest prompt (first come first
            // served: its first token comes as early as possible, later prompts wait their turn instead of all
            // finishing together; the decoding requests stall for at most one segment per pass)
            // the decoding rounds of greedy text requests are batched: one verification forward for up to
            // kMaxBatch slots (the dense layers read their weights once); the others step one by one
            static const bool no_batch = std::getenv("G2E_NO_BATCH") != nullptr;
            int oldest = -1;
            bool has_decoding = false;
            std::vector<size_t> batch;
            for (size_t s = 0; s < gens.size(); ++s) {
                if (!running[s] || !gens[s]->busy()) continue;
                if (gens[s]->prefilling()) {
                    if (oldest < 0 || slot_used[s] < slot_used[size_t(oldest)]) oldest = int(s);
                } else if (!no_batch && gens[s]->batchable()) {
                    has_decoding = true;
                    batch.push_back(s);
                } else {
                    has_decoding = true;
                    step_job(s);
                }
            }
            for (size_t k = 0; k < batch.size(); k += Qwen35::kMaxBatch) {
                const std::vector<size_t> group(batch.begin() + long(k),
                                                batch.begin() + long(std::min<size_t>(batch.size(), k + Qwen35::kMaxBatch)));
                if (group.size() == 1) step_job(group[0]);
                else step_batch_jobs(group);
            }
            if (oldest >= 0) step_job(size_t(oldest), has_decoding ? kPrefillStep : kPrefillStepSolo);
            uint32_t active = 0, decoding = 0, prefilling = 0;
            for (size_t s = 0; s < running.size(); ++s) {
                if (!running[s]) continue;
                ++active;
                if (gens[s]->busy() && gens[s]->prefilling()) ++prefilling;
                else ++decoding;
            }
            kv_used = model->kv_total();
            slots_active = active;
            slots_decoding = decoding;
            slots_prefilling = prefilling;
        }
    }

    struct Totals {  // since start (updated under stats_mu)
        uint64_t requests = 0, cached_requests = 0, prompt_tokens = 0, cached_tokens = 0;
    } totals;

    void load() {
        if (inkling) return load_inkling();
        tok = std::make_unique<Tokenizer>(opt.model_dir + "/tokenizer.json");
        tpl = std::make_unique<ChatTemplate>(read_file(opt.model_dir + "/chat_template.jinja"));
        const auto gc = ojson::parse(read_file(opt.model_dir + "/generation_config.json"));
        for (const char* k : {"temperature", "top_p", "top_k", "min_p"})
            if (gc.contains(k)) gen_defaults[k] = gc[k];
        if (gc.contains("eos_token_id")) {
            if (gc["eos_token_id"].is_array()) for (auto& e : gc["eos_token_id"]) eos.push_back(e.get<int32_t>());
            else eos.push_back(gc["eos_token_id"].get<int32_t>());
        }
        target = std::make_unique<WeightPack>(opt.target_pack);
        dev = std::make_unique<Device>();
        stream = std::make_unique<Stream>(*dev);
        const auto up = target->upload(*dev, *stream);
        std::fprintf(stderr, "[g2e] target weights %.2f GiB in %.1f s\n", up.gib, up.seconds);
        model = std::make_unique<Qwen35>(*dev, *stream, *target, opt.max_model_len, 1);
        if (!opt.draft_pack.empty()) {
            draft = std::make_unique<WeightPack>(opt.draft_pack);
            const auto du = draft->upload(*dev, *stream);
            drafter = std::make_unique<DSpine>(*dev, *stream, *draft, *target, opt.max_model_len);
            model->set_capture(drafter->target_layers());
            std::fprintf(stderr, "[g2e] drafter weights %.2f GiB in %.1f s\n", du.gib, du.seconds);
        }
        if (!opt.vision_pack.empty()) {
            vpack = std::make_unique<WeightPack>(opt.vision_pack);
            const auto vu = vpack->upload(*dev, *stream);
            vision = std::make_unique<Vision>(*dev, *stream, *vpack);
            if (!opt.max_image_pixels) opt.max_image_pixels = vision->max_pixels();
            std::fprintf(stderr, "[g2e] vision weights %.2f GiB in %.1f s (images up to %llu pixels)\n", vu.gib, vu.seconds,
                         (unsigned long long)opt.max_image_pixels);
        }
        gen = std::make_unique<Generator>(*model, drafter.get(), eos, opt.max_model_len);
        {   // prefix cache: G2E_PREFIX_CACHE_GB of host memory (default 64, 0: off), G2E_PREFIX_DISK_DIR +
            // G2E_PREFIX_DISK_GB for a disk tier
            const char* gb = std::getenv("G2E_PREFIX_CACHE_GB");
            const double host_gb = gb ? std::atof(gb) : 64.0;
            const char* dir = std::getenv("G2E_PREFIX_DISK_DIR");
            const char* dgb = std::getenv("G2E_PREFIX_DISK_GB");
            if (host_gb > 0) {
                prefix = std::make_unique<PrefixCache>(*dev, *model, drafter.get(), uint64_t(host_gb * (1ull << 30)),
                                                       dir ? dir : "", uint64_t((dgb ? std::atof(dgb) : 0.0) * (1ull << 30)));
                gen->set_prefix_cache(prefix.get());
                const auto ims = tok->encode("<|im_start|>");
                if (ims.size() == 1) im_start = ims[0];
                std::fprintf(stderr, "[g2e] prefix cache: %.0f GiB host%s%s\n", host_gb, dir ? ", disk " : "", dir ? dir : "");
            }
        }
        // every recipe a request can need (compiled once, then loaded from the recipe cache)
        const auto p0 = std::chrono::steady_clock::now();
        auto report = [&](const char* what) {
            return [what, p0](size_t done, size_t total) {
                if (done == total || done % 20 == 0)
                    std::fprintf(stderr, "[g2e] %s recipes %zu/%zu (%.0f s)\n", what, done, total,
                                 std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count());
            };
        };
        model->precompile(report("target"));
        if (drafter) drafter->precompile(report("drafter"));
        // warm-up: load (or compile) the recipes of a short request, greedy and sampled
        const auto w0 = std::chrono::steady_clock::now();
        const auto ids = tok->encode("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
        SamplingParams sp;
        sp.temperature = 0;
        sp.max_tokens = 40;
        gen->generate(ids, sp, [](std::span<const int32_t>) { return true; });
        sp.temperature = 1;
        sp.top_k = 20;
        sp.max_tokens = 4;
        gen->generate(ids, sp, [](std::span<const int32_t>) { return true; });
        gen->reset();
        std::fprintf(stderr, "[g2e] warm-up %.1f s\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
        // slots: up to opt.slots requests at once, each up to max_model_len; the KV caches are allocated as the
        // requests need them, within one budget (the device memory left, less a margin for workspace growth)
        gens.push_back(std::move(gen));
        for (uint32_t k = 1; k < opt.slots; ++k) {
            const uint32_t s = model->add_slot(0);
            if (drafter) drafter->add_slot();
            auto g = std::make_unique<Generator>(*model, drafter.get(), eos, opt.max_model_len);
            g->set_slot(s);
            if (prefix) g->set_prefix_cache(prefix.get());
            gens.push_back(std::move(g));
        }
        running.resize(gens.size());
        slot_used.assign(gens.size(), 0);
        const uint64_t held = model->kv_total(), free_now = dev->info().free_bytes;
        kv_budget = opt.kv_gib > 0 ? uint64_t(opt.kv_gib * double(1ull << 30))
                                   : held + (free_now > kMarginBytes ? free_now - kMarginBytes : 0);
        gens[0]->release();  // slot 0's warm-up caches go back to the budget
        for (auto& g : gens) {
            g->set_kv_reserve([this](uint32_t slot, uint64_t extra) { return reserve_kv(slot, extra); });
            g->set_defer_prefix_save(true);
        }
        std::fprintf(stderr, "[g2e] %u slots, dynamic KV caches: budget %.1f GiB = %.0fK context rows in all (device free %.1f GiB)\n",
                     opt.slots, double(kv_budget) / (1ull << 30),
                     double(kv_budget) / double(model->kv_bytes_for(1)) / 1024.0, double(free_now) / (1ull << 30));
        sched_thread = std::thread([this] { scheduler(); });
        sched_thread.detach();
    }


    // ---- Inkling-2 engine --------------------------------------------------------------------------------------
    void load_inkling() {
        opt.max_model_len = (opt.max_model_len + 1023u) / 1024u * 1024u;
        opt.slots = std::clamp(opt.slots, 1u, 8u);
        tok = std::make_unique<Tokenizer>(opt.model_dir + "/tokenizer.json");
        tpl = std::make_unique<ChatTemplate>(read_file(opt.model_dir + "/chat_template.jinja"));
        const auto gc = ojson::parse(read_file(opt.model_dir + "/generation_config.json"));
        for (const char* k : {"temperature", "top_p", "top_k", "min_p"})
            if (gc.contains(k)) gen_defaults[k] = gc[k];
        target = std::make_unique<WeightPack>(opt.target_pack);
        dev = std::make_unique<Device>();
        stream = std::make_unique<Stream>(*dev);
        const auto up = target->upload(*dev, *stream);
        std::fprintf(stderr, "[g2e] inkling2 weights %.2f GiB in %.1f s\n", up.gib, up.seconds);
        const uint32_t init_cap = std::min<uint32_t>(opt.max_model_len, 16384u);
        ink = std::make_unique<Inkling2>(*dev, *stream, *target, opt.max_model_len, opt.slots, init_cap);
        if (!opt.vision_pack.empty()) {
            vpack = std::make_unique<WeightPack>(opt.vision_pack);
            const auto vu = vpack->upload(*dev, *stream);
            ink_vision = std::make_unique<InklingVision>(*dev, *stream, *vpack);
            // one image token per 40x40 cell; larger images are shrunk to fit (default 4096 tokens)
            if (opt.max_image_pixels) ink_max_image_tokens = uint32_t(std::max<uint64_t>(1, opt.max_image_pixels / 1600));
            std::fprintf(stderr, "[g2e] inkling vision weights %.2f GiB in %.1f s (images up to %u tokens)\n", vu.gib, vu.seconds,
                         ink_max_image_tokens);
        }
        eos = {ink->config().eos};
        if (gc.contains("eos_token_id")) {
            eos.clear();
            if (gc["eos_token_id"].is_array()) for (auto& e : gc["eos_token_id"]) eos.push_back(e.get<int32_t>());
            else eos.push_back(gc["eos_token_id"].get<int32_t>());
        }
        const std::string eng_summary = ink->engine_summary();
        std::fprintf(stderr, "[g2e] inkling2 engine: %s\n", eng_summary.c_str());
        // the drafts are verified through the Direct engine: without it the drafter would only take device memory
        const bool can_spec = ink_spec_on();
        if (!opt.draft_pack.empty() && !can_spec)
            std::fprintf(stderr, "[g2e] inkling2: --draft-pack ignored (speculation needs the Direct engine)\n");
        if (!opt.draft_pack.empty() && can_spec && !std::getenv("G2E_INK_NO_SPEC")) {
            draft = std::make_unique<WeightPack>(opt.draft_pack);
            const auto du = draft->upload(*dev, *stream);
            ink_drafter = std::make_unique<InkDSpark>(*dev, *stream, *draft, *target, opt.max_model_len);
            std::fprintf(stderr, "[g2e] inkling2 DSpark drafter %.2f GiB in %.1f s (max ctx %u)\n", du.gib, du.seconds,
                         ink_drafter->max_context());
        }
        // warm-up: Direct decode + DSpark first (allocates Synapse workspace before BF16 cache sizing), then
        // explicit BF16 cache sizing once all packs are loaded, then prefill warm-up
        const auto w0 = std::chrono::steady_clock::now();
        std::vector<int32_t> ids = tok->encode(tpl->render({{"messages", ojson::array({{{"role", "user"}, {"content", "Hello"}}})},
                                                           {"add_generation_prompt", true}}));
        while (ids.size() < 511) ids.push_back(ids.back());
        const bool direct_active = ink->direct_engine();
        const bool direct_full_dec = ink->direct_full_decode();
        if (direct_full_dec) {
            if (ink_drafter) {
                ink->set_aux_capture(true);
                ink->forward_argmax(std::span<const int32_t>(&ids.back(), 1));
                ink_drafter->precompile();
                ink_drafter->append(ink->taps(), 0, ink->tap_stride(), 1, 0);
                ink->invalidate_sram();
                const auto d = ink_drafter->draft(ids.back(), ink->position(), ink->taps(), 0, ink->tap_stride(), 3);
                ink->invalidate_sram();
                {
                    std::mt19937_64 warm_rng(42);
                    (void)ink_drafter->draft_sampled(ids.back(), ink->position(), ink->taps(), 0, ink->tap_stride(),
                                                     0.7f, 50, 0.95f, 0.0f, warm_rng, 3);
                    ink->invalidate_sram();
                }
                for (uint32_t T = 2; T <= 4; ++T) {
                    if (!ink->can_verify(T)) continue;
                    std::vector<int32_t> blk = {ids.back()};
                    blk.insert(blk.end(), d.begin(), d.begin() + (T - 1));
                    ink->verify(blk);
                    ink->accept(1);
                }
                ink->forward_argmax(std::span<const int32_t>(&ids.back(), 1));
                ink->set_aux_capture(false);
                ink_drafter->reset();
            } else {
                for (int i = 0; i < 3; ++i) ink->forward_argmax(std::span<const int32_t>(&ids.back(), 1));
            }
            ink->reset();
        }
        // Size the static BF16 expert cache once all packs and non-prefill workspaces are resident.
        // InkDirect::init_bf16_cache reserves kMarginBytes (4 GiB) + max(0, kv_need_bytes - ink->kv_total());
        // pass target KV need plus any unallocated lazy DSpark KV slots. No-op in Synapse-only mode (24/24 expanded at load).
        const uint64_t dspark_kv_total = ink_drafter ? uint64_t(opt.slots) * ink_drafter->slot_bytes() : 0;
        const uint64_t dspark_kv_held = (ink_drafter && ink_drafter->slot_allocated(0)) ? ink_drafter->slot_bytes() : 0;
        const uint64_t dspark_kv_unalloc = dspark_kv_total - dspark_kv_held;
        const uint64_t kv_rows_need = std::min<uint64_t>(uint64_t(opt.slots) * opt.max_model_len, 1048576ull);
        const uint64_t target_kv_need = opt.kv_gib > 0 ? uint64_t(opt.kv_gib * double(1ull << 30))
                                                       : ink->kv_bytes_for(uint32_t(kv_rows_need));
        ink->init_bf16_cache(target_kv_need + dspark_kv_unalloc);
        // Prefill warm-up: masked short/tail prefill, full 1024-token chunk, and (when Direct decode was off) decode
        ink->forward(ids, true);
        if (!direct_full_dec) {
            for (int i = 0; i < 3; ++i) ink->forward_argmax(std::span<const int32_t>(&ids.back(), 1));
        }
        {
            std::vector<int32_t> big = ids;
            while (big.size() < 1024) big.push_back(ids.back());
            for (const size_t n : {size_t(17), size_t(1024)}) {
                ink->reset();
                ink->forward(std::span<const int32_t>(big.data(), n), true);
            }
            ink->reset();
        }
        if (ink_drafter) {
            ink->reset();
            ink->set_aux_capture(true);
            ink->set_tap_hook([&](uint32_t p0, uint32_t n) {
                ink_drafter->append(ink->taps(), 0, ink->tap_stride(), n, p0);
                ink->invalidate_sram();
            });
            ink->forward(ids, true);
            ink->set_tap_hook({});
            ink->set_aux_capture(false);
            ink_drafter->reset();
            ink_drafter->set_slot_allocated(0, false);
        }
        ink->reset();
        ink->set_capacity(0);
        // Create remaining slots lazily (0 KV bytes) and fix the shared dynamic KV budget from what is left
        for (uint32_t k = 1; k < opt.slots; ++k) {
            ink->add_slot(0);
            if (ink_drafter) ink_drafter->add_slot(false);
        }
        ink->select_slot(0);
        const auto dinfo = dev->info();
        const uint64_t free_now = dinfo.free_bytes;
        device_free_bytes = free_now;
        device_total_bytes = dinfo.total_bytes;
        const uint64_t reserve = kMarginBytes + dspark_kv_total;
        const uint64_t max_kv = free_now > reserve ? free_now - reserve : 0;
        if (opt.kv_gib > 0) {
            const uint64_t req_kv = uint64_t(opt.kv_gib * double(1ull << 30));
            if (req_kv > max_kv) {
                std::fprintf(stderr, "[g2e] inkling2: --kv-cache-gib %.2f exceeds available %.2f GiB (after %.0f GiB margin%s); clamping\n",
                             opt.kv_gib, double(max_kv) / double(1ull << 30), double(kMarginBytes) / double(1ull << 30),
                             dspark_kv_total ? " + drafter KV" : "");
                kv_budget = max_kv;
            } else {
                kv_budget = req_kv;
            }
        } else {
            kv_budget = max_kv;
        }
        ink_slots.resize(opt.slots);
        kv_used = ink->kv_total();
        const auto& icfg = ink->config();
        ink_bf16_cached = 0;
        ink_bf16_total = icfg.layers;
        for (uint32_t L = 0; L < icfg.layers; ++L)
            if (ink->bf16_layer(L).cached) ++ink_bf16_cached;
        ink_bf16_staging = ink->bf16_staging_w13() != 0;
        const uint64_t layer_bf16_bytes = ((ink->bf16_w13_bytes() + 255ull) & ~255ull) +
                                          ((ink->bf16_w2_bytes() + 255ull) & ~255ull);
        ink_bf16_bytes = uint64_t(ink_bf16_cached + (ink_bf16_staging ? 1u : 0u)) * layer_bf16_bytes;
        if (!direct_active) {
            ink_bf16_mode = "synapse";
        } else {
            const char* env = std::getenv("G2E_INK_BF16_CACHE");
            ink_bf16_mode = (env && *env) ? std::string(env) : std::string("auto");
        }
        ink_dspark_kv_reserve = dspark_kv_total;
        ink_kv_budget_tokens = kv_budget / ink->kv_bytes_for(1);
        std::fprintf(stderr, "[g2e] inkling2 warm-up %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
        const std::string dspark_note = dspark_kv_total
            ? ", drafter KV reserve " + std::to_string(uint64_t(std::llround(double(dspark_kv_total) / double(1ull << 20)))) + " MiB"
            : std::string();
        std::fprintf(stderr, "[g2e] inkling2 %u slots (max ctx %u, batched Direct=%d): bf16_cache mode=%s %u/%u layers%s (%.2f GiB), "
                             "KV budget %.2f GiB (%llu tokens = %.0fK)%s (device free %.2f GiB)\n",
                     opt.slots, opt.max_model_len, int(ink->can_decode_batch()),
                     ink_bf16_mode.c_str(), ink_bf16_cached, ink_bf16_total, ink_bf16_staging ? "+staging" : "",
                     double(ink_bf16_bytes) / double(1ull << 30),
                     double(kv_budget) / double(1ull << 30), (unsigned long long)ink_kv_budget_tokens,
                     double(ink_kv_budget_tokens) / 1024.0, dspark_note.c_str(), double(free_now) / double(1ull << 30));
        {
            const char* e = std::getenv("G2E_INK_SNAPSHOT_GB");
            ink_snap_budget = uint64_t((e ? std::atof(e) : 32.0) * double(1ull << 30));
            std::fprintf(stderr, "[g2e] inkling2 host-RAM prefix cache budget %.1f GiB\n", double(ink_snap_budget) / double(1ull << 30));
        }
        sched_thread = std::thread([this] { ink_scheduler(); });
        sched_thread.detach();
    }

    // Exact filtered target probability distribution (vLLM order: temperature, top_k, top_p, min_p).
    static std::vector<std::pair<int32_t, double>> build_target_dist(const float* lg, size_t V, const SamplingParams& p) {
        if (p.greedy()) {
            const int32_t best = int32_t(std::max_element(lg, lg + V) - lg);
            return {{best, 1.0}};
        }
        const float inv_t = 1.0f / p.temperature;
        std::vector<std::pair<float, int32_t>> c;
        auto cmp = [](const auto& a, const auto& b) {
            return a.first > b.first || (a.first == b.first && a.second < b.second);
        };
        if (p.top_k > 0 && size_t(p.top_k) <= 512 && size_t(p.top_k) < V) {
            // Fast 1-pass exact min-heap for top_k (skips >99.7% of vocab with a single float compare).
            const size_t K = size_t(p.top_k);
            std::vector<std::pair<float, int32_t>> h(K);
            for (size_t i = 0; i < K; ++i) h[i] = {lg[i], int32_t(i)};
            auto worse = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                return a.first < b.first || (a.first == b.first && a.second > b.second);
            };
            auto sift_down = [&](size_t pos) {
                const auto val = h[pos];
                while (true) {
                    const size_t l = 2 * pos + 1;
                    if (l >= K) break;
                    const size_t r = l + 1;
                    const size_t sm = (r < K && worse(h[r], h[l])) ? r : l;
                    if (!worse(h[sm], val)) break;
                    h[pos] = h[sm];
                    pos = sm;
                }
                h[pos] = val;
            };
            for (size_t i = K / 2; i-- > 0;) sift_down(i);
            float thresh = h[0].first;
            for (size_t i = K; i < V; ++i) {
                const float v = lg[i];
                if (__builtin_expect(v > thresh, 0)) {
                    h[0] = {v, int32_t(i)};
                    sift_down(0);
                    thresh = h[0].first;
                }
            }
            std::sort(h.begin(), h.end(), cmp);
            const float mx = h[0].first;
            c.reserve(K);
            for (const auto& [v, id] : h) {
                const float z = (v - mx) * inv_t;
                if (z > -30.f) c.push_back({z, id});
                else break;
            }
        } else {
            const float mx = *std::max_element(lg, lg + V);
            float cut_z = -30.f;
            if (p.top_k <= 0 && p.top_p >= 1.0f && p.min_p > 0.0f)
                cut_z = std::max(cut_z, std::log(p.min_p) - 1e-4f);
            const float cut_lg = mx + cut_z * p.temperature;
            // Fast upper-tail scan for top_k <= 0 && top_p < 1.0: avoid collecting ~100K negligible tail tokens
            bool fast_ok = false;
            if (p.top_k <= 0 && p.top_p < 1.0f) {
                const float fast_z = -20.f;
                const float fast_lg = mx + fast_z * p.temperature;
                c.reserve(512);
                uint32_t n_lo = 0;
                float max_lo = -1e30f;
                for (size_t i = 0; i < V; ++i) {
                    const float v = lg[i];
                    if (__builtin_expect(v > fast_lg, 0)) {
                        c.push_back({(v - mx) * inv_t, int32_t(i)});
                    } else if (v > cut_lg) {
                        ++n_lo;
                        if (v > max_lo) max_lo = v;
                    }
                }
                if (!c.empty()) {
                    std::sort(c.begin(), c.end(), cmp);
                    double sum_hi = 0.0;
                    for (const auto& [z, _] : c) sum_hi += std::exp(double(z));
                    const double lo_bound = n_lo ? double(n_lo) * std::exp(double((max_lo - mx) * inv_t)) : 0.0;
                    const bool minp_safe = p.min_p <= 0.0f || double(p.min_p) > std::exp(double(fast_z));
                    if (minp_safe && lo_bound < 1e-5 * sum_hi) {
                        double acc = 0.0;
                        size_t n1 = c.size(), n2 = c.size();
                        for (size_t i = 0; i < c.size(); ++i) {
                            acc += std::exp(double(c[i].first));
                            if (n1 == c.size() && acc >= double(p.top_p) * sum_hi) n1 = i + 1;
                            if (n2 == c.size() && acc >= double(p.top_p) * (sum_hi + lo_bound)) { n2 = i + 1; break; }
                        }
                        if (n1 == n2 && n1 <= c.size()) fast_ok = true;
                    }
                }
                if (!fast_ok) c.clear();
            }
            if (!fast_ok) {
                c.reserve(4096);
                for (size_t i = 0; i < V; ++i) {
                    if (lg[i] > cut_lg) c.push_back({(lg[i] - mx) * inv_t, int32_t(i)});
                }
                if (c.empty()) return {{0, 1.0}};
                if (p.top_k > 0 && size_t(p.top_k) < c.size()) {
                    std::nth_element(c.begin(), c.begin() + long(p.top_k), c.end(), cmp);
                    c.resize(size_t(p.top_k));
                }
                std::sort(c.begin(), c.end(), cmp);
            }
        }
        if (c.empty()) return {{0, 1.0}};
        double sum = 0;
        std::vector<double> pr(c.size());
        for (size_t i = 0; i < c.size(); ++i) sum += pr[i] = std::exp(double(c[i].first));
        size_t n = c.size();
        if (p.top_p < 1.0f) {
            double acc = 0;
            for (size_t i = 0; i < c.size(); ++i) {
                acc += pr[i] / sum;
                if (acc >= p.top_p) { n = i + 1; break; }
            }
        }
        if (p.min_p > 0.0f) {
            const double thr = p.min_p * pr[0];
            size_t m = 0;
            while (m < n && pr[m] >= thr) ++m;
            n = std::max<size_t>(m, 1);
        }
        double tot = 0;
        for (size_t i = 0; i < n; ++i) tot += pr[i];
        std::vector<std::pair<int32_t, double>> dist;
        dist.reserve(n);
        const double inv_tot = 1.0 / tot;
        for (size_t i = 0; i < n; ++i) dist.push_back({c[i].second, pr[i] * inv_tot});
        return dist;
    }

    static int32_t draw_dist(const std::vector<std::pair<int32_t, double>>& dist, std::mt19937_64& rng) {
        double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
        for (const auto& [id, pr] : dist) {
            u -= pr;
            if (u <= 0.0) return id;
        }
        return dist.back().first;
    }

    // temperature / top-k / top-p / min-p sampling over the logits (vLLM order: temperature, then the filters)
    static int32_t sample_logits(const std::vector<float>& lg, const SamplingParams& p, std::mt19937_64& rng) {
        if (p.greedy()) return int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
        return draw_dist(build_target_dist(lg.data(), lg.size(), p), rng);
    }

    uint64_t ink_job_salt(const Job& job) const {
        uint64_t salt = 0;
        for (const auto& im : job.images) {
            uint64_t hh = 0x9e3779b97f4a7c15ull ^ (uint64_t(im.grid_h) << 32 | im.grid_w);
            const auto* wds = reinterpret_cast<const uint32_t*>(im.patches.data());
            for (size_t i = 0; i < im.patches.size(); ++i) hh = (hh ^ wds[i]) * 0x100000001b3ull;
            salt = (salt ^ hh) * 0xff51afd7ed558ccdull;
        }
        return salt;
    }

    // speculation: on when the Direct engine verifies 4-token blocks (G2E_INK_SPEC=1 forces it on, 0 off)
    bool ink_spec_on() const {
        static const int spec_env = std::getenv("G2E_INK_SPEC") ? std::atoi(std::getenv("G2E_INK_SPEC")) : -1;
        return spec_env > 0 || (spec_env < 0 && ink->can_verify(4));
    }
    bool ink_job_spec(const Job& job) const {
        return ink_drafter && ink_spec_on() && job.prompt.size() + 16 < ink_drafter->max_context();
    }

    uint64_t ink_kv_estimate(const Job& job) const {
        const uint32_t P = uint32_t(job.prompt.size());
        const uint32_t want = std::min<uint32_t>(opt.max_model_len, P + std::min<uint32_t>(job.sp.max_tokens, 2048u) + 17u);
        return ink->kv_bytes_for(ink->round_capacity(want));
    }

    int pick_ink_slot(const Job& job, uint64_t salt, bool spec) const {
        int best = -1;
        size_t best_match = 0;
        int best_fit = -1;
        bool best_same = false;
        const uint32_t need_rows = uint32_t(job.prompt.size() + 16);
        const uint32_t cur_slot = ink->slot();
        for (uint32_t s = 0; s < ink_slots.size(); ++s) {
            const auto& st = ink_slots[s];
            if (st.phase != InkSlot::Idle) continue;
            const size_t match = job.images.empty() && st.salt == salt && !st.history.empty() &&
                                 (!spec || st.history_draft) && st.history.size() < job.prompt.size() &&
                                 ink->slot_position(s) == st.history.size() &&
                                 std::equal(st.history.begin(), st.history.end(), job.prompt.begin())
                                     ? st.history.size()
                                     : 0;
            const uint32_t cap = ink->slot_capacity(s);
            const int fit = cap >= need_rows ? 2 : (cap > 0 ? 1 : 0);
            // Among equally-fitting uncached slots, prefer the currently active slot whose prefill plans are already hot.
            const bool same = (s == cur_slot) && (fit == 2);
            const bool better = best < 0 || match > best_match ||
                                (match == best_match &&
                                 (fit > best_fit ||
                                  (fit == best_fit &&
                                   (same > best_same ||
                                    (same == best_same && st.used < ink_slots[size_t(best)].used)))));
            if (better) {
                best = int(s);
                best_match = match;
                best_fit = fit;
                best_same = same;
            }
        }
        return best;
    }

    bool reserve_ink_kv(uint32_t slot, uint32_t need_rows) {
        const uint32_t min_cap = ink->round_capacity(std::min<uint32_t>(need_rows, opt.max_model_len));
        const uint32_t cur_cap = ink->slot_capacity(slot);
        if (min_cap <= cur_cap) return true;
        const uint32_t grow_rows = std::max<uint32_t>(need_rows, cur_cap ? std::min<uint32_t>(cur_cap * 2u, opt.max_model_len) : 16384u);
        uint32_t want_cap = ink->round_capacity(std::min<uint32_t>(grow_rows, opt.max_model_len));
        if (want_cap > min_cap && ink->kv_total() + (ink->kv_bytes_for(want_cap) - ink->kv_bytes_for(cur_cap)) > kv_budget)
            want_cap = min_cap;
        const uint64_t delta = ink->kv_bytes_for(want_cap) - ink->kv_bytes_for(cur_cap);
        while (ink->kv_total() + delta > kv_budget) {
            int v = -1;
            for (uint32_t s = 0; s < ink_slots.size(); ++s) {
                if (s != slot && ink_slots[s].phase == InkSlot::Idle && ink->slot_capacity(s) > 0 &&
                    (v < 0 || ink_slots[s].used < ink_slots[size_t(v)].used))
                    v = int(s);
            }
            if (v < 0) return false;
            ink->select_slot(uint32_t(v));
            ink->set_capacity(0);
            ink_slots[size_t(v)].history.clear();
            ink_slots[size_t(v)].history_draft = false;
            if (ink_drafter) ink_drafter->set_slot_allocated(uint32_t(v), false);
            ++kv_evictions;
        }
        ink->select_slot(slot);
        ink->set_capacity(want_cap);
        kv_used = ink->kv_total();
        return true;
    }

    void ink_snapshot(uint32_t s, const std::vector<int32_t>& prompt, size_t n, uint64_t salt, bool with_draft,
                      uint32_t last_rows, const std::vector<float>* end_logits = nullptr) {
        if (!ink_snap_budget || n < 16) return;
        const bool want_draft = with_draft && ink_drafter != nullptr && ink_drafter->slot_allocated(s) &&
                                n <= ink_drafter->max_context();
        int existing = -1;
        for (size_t i = 0; i < ink_snaps.size(); ++i) {
            auto& sn = ink_snaps[i];
            if (sn.salt == salt && sn.tokens.size() == n && std::equal(sn.tokens.begin(), sn.tokens.end(), prompt.begin())) {
                sn.used = ++ink_snap_tick;
                const bool need_draft = want_draft && !sn.has_draft;
                const bool need_logits = end_logits && !end_logits->empty() && sn.last_logits.empty();
                const bool need_taps = want_draft && end_logits && sn.last_taps.empty();
                if (!need_draft && !need_logits && !need_taps) return;
                existing = int(i);
                break;
            }
        }
        ink->select_slot(s);
        const uint64_t tbytes = (ink->state_bytes(uint32_t(n)) + 4095) / 4096 * 4096;
        const uint64_t bytes = tbytes + (want_draft ? ink_drafter->rows_bytes(uint32_t(n)) : 0);
        if (bytes > ink_snap_budget) return;
        try {
            if (existing >= 0) ink_snaps.erase(ink_snaps.begin() + existing);
            uint64_t held = 0;
            for (auto& sn : ink_snaps) held += sn.bytes;
            while (held + bytes > ink_snap_budget && !ink_snaps.empty()) {
                size_t v = 0;
                for (size_t i = 1; i < ink_snaps.size(); ++i) if (ink_snaps[i].used < ink_snaps[v].used) v = i;
                held -= ink_snaps[v].bytes;
                ink_snaps.erase(ink_snaps.begin() + long(v));
            }
            InkSnap sn;
            sn.tokens.assign(prompt.begin(), prompt.begin() + long(n));
            sn.salt = salt;
            sn.bytes = bytes;
            sn.used = ++ink_snap_tick;
            sn.has_draft = want_draft;
            sn.buf = std::make_unique<HostBuffer>(*dev, bytes);
            ink->save_state(*sn.buf, 0);
            sn.draft_off = want_draft ? tbytes : 0;
            if (want_draft) {
                ink_drafter->select_slot(s);
                ink_drafter->save_rows(*sn.buf, tbytes, uint32_t(n));
            }
            if (end_logits && !end_logits->empty()) {
                sn.last_logits = *end_logits;
                if (want_draft && last_rows > 0 && ink_taps_owner == int(s)) {
                    const uint64_t H = ink->config().hidden, rowb = H * 2;
                    const uint64_t row_off = uint64_t(last_rows - 1) * rowb;
                    HostBuffer th(*dev, 4 * rowb);
                    for (uint64_t k = 0; k < 4; ++k)
                        stream->download(ink->taps(), k * ink->tap_stride() + row_off, th, k * rowb, rowb);
                    stream->synchronize();
                    sn.last_taps.assign(th.as<uint16_t>(), th.as<uint16_t>() + 4 * H);
                } else {
                    stream->synchronize();
                }
            } else {
                stream->synchronize();
            }
            ink_snaps.push_back(std::move(sn));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[g2e] inkling2: host snapshot skipped (%s)\n", e.what());
        }
    }

    bool ink_emit(uint32_t s, int32_t t) {
        auto& st = ink_slots[s];
        auto& job = *st.job;
        const auto& sp = job.sp;
        st.r.tokens.push_back(t);
        live_decode_tokens.fetch_add(1, std::memory_order_relaxed);
        bool cancel = false;
        {
            std::lock_guard lk(job.mu);
            job.tokens.push_back(t);
            job.cv.notify_one();
            cancel = job.cancel;
        }
        if (!sp.ignore_eos && std::find(eos.begin(), eos.end(), t) != eos.end()) { st.r.finish = "stop"; return false; }
        if (std::find(sp.stop_token_ids.begin(), sp.stop_token_ids.end(), t) != sp.stop_token_ids.end()) {
            st.r.finish = "stop";
            st.r.stop_token = t;
            return false;
        }
        if (cancel) { st.r.finish = "abort"; return false; }
        if (st.r.tokens.size() >= sp.max_tokens) { st.r.finish = "length"; return false; }
        return true;
    }

    void ink_finish_slot(uint32_t s, const std::string& error = "", bool drop_kv = false) {
        using clk = std::chrono::steady_clock;
        auto& st = ink_slots[s];
        auto job = std::move(st.job);
        st.phase = InkSlot::Idle;
        st.pending_imgs.clear();
        st.used = ++sched_tick;
        if (!error.empty()) {
            try {
                ink->select_slot(s);
                if (drop_kv) ink->set_capacity(0);
                else ink->reset();
            } catch (...) {}
            st.history.clear();
            st.history_draft = false;
            if (ink_drafter && drop_kv) {
                try { ink_drafter->set_slot_allocated(s, false); } catch (...) {}
            }
            if (ink_taps_owner == int(s)) ink_taps_owner = -1;
        }
        if (!job) return;
        st.r.decode_ms = st.t1.time_since_epoch().count()
                             ? std::chrono::duration<double, std::milli>(clk::now() - st.t1).count()
                             : 0.0;
        st.r.drafts_accepted = uint32_t(st.r.tokens.size()) > st.r.rounds
                                   ? uint32_t(st.r.tokens.size()) - st.r.rounds
                                   : 0u;
        const bool want_snap = error.empty() && st.r.finish != "abort" && job->images.empty() &&
                               !st.history.empty() && ink->slot_position(s) == st.history.size();
        {
            std::lock_guard lk(job->mu);
            job->gen = std::move(st.r);
            job->error = error;
            job->done = true;
            job->cv.notify_one();
        }
        if (want_snap) {
            const bool cap_draft = st.spec && st.history_draft && ink_drafter != nullptr &&
                                   ink_drafter->slot_allocated(s) && st.history.size() <= ink_drafter->max_context();
            ink_snapshot(s, st.history, st.history.size(), st.salt, cap_draft, st.last_rows, nullptr);
        }
    }

    void ink_begin_slot(uint32_t s, std::shared_ptr<Job> job) {
        using clk = std::chrono::steady_clock;
        auto& st = ink_slots[s];
        st.job = std::move(job);
        st.r = GenerationResult{};
        st.r.prompt_tokens = uint32_t(st.job->prompt.size());
        st.r.finish = "length";
        st.t0 = clk::now();
        st.t1 = {};
        st.at = 0;
        st.pi = 0;
        st.points.clear();
        st.pending_imgs.clear();
        st.last_rows = 0;
        st.tap_row_off = 0;
        st.plain_left = 0;
        st.probe = false;
        st.rng.seed(st.job->sp.seed >= 0 ? uint64_t(st.job->sp.seed) : std::random_device{}());
        st.used = ++sched_tick;
        st.phase = InkSlot::Prefill;

        const auto& prompt = st.job->prompt;
        const auto& sp = st.job->sp;
        const uint64_t salt = ink_job_salt(*st.job);
        const bool spec = ink_job_spec(*st.job);
        st.spec = spec;

        if (!reserve_ink_kv(s, uint32_t(prompt.size() + 16))) {
            ink_finish_slot(s, "not enough device memory for the KV cache of this request", true);
            return;
        }
        ink->select_slot(s);
        ink->set_aux_capture(false);
        const bool live = st.salt == salt && !st.history.empty() && (!spec || st.history_draft) &&
                          st.history.size() < prompt.size() &&
                          std::equal(st.history.begin(), st.history.end(), prompt.begin()) &&
                          ink->position() == st.history.size();
        int best = -1;
        for (size_t i = 0; i < ink_snaps.size(); ++i) {
            const auto& sn = ink_snaps[i];
            if (sn.salt != salt || sn.tokens.size() > prompt.size()) continue;
            if (spec && !sn.has_draft) continue;
            if (sn.tokens.size() == prompt.size()) {
                const bool can_exact = !sn.last_logits.empty() && (!spec || !sn.last_taps.empty());
                if (!can_exact) continue;
            }
            if ((live && sn.tokens.size() <= st.history.size()) ||
                (best >= 0 && sn.tokens.size() <= ink_snaps[size_t(best)].tokens.size()))
                continue;
            if (std::equal(sn.tokens.begin(), sn.tokens.end(), prompt.begin())) best = int(i);
        }
        std::vector<float> exact_logits;
        if (best >= 0) {
            auto& sn = ink_snaps[size_t(best)];
            ink->load_state(*sn.buf, 0, uint32_t(sn.tokens.size()));
            if (spec && ink_drafter && sn.has_draft) {
                ink_drafter->select_slot(s);
                ink_drafter->set_slot_allocated(s, true);
                ink_drafter->load_rows(*sn.buf, sn.draft_off, uint32_t(sn.tokens.size()));
            }
            if (sn.tokens.size() == prompt.size()) {
                exact_logits = sn.last_logits;
                if (spec && !sn.last_taps.empty()) {
                    const uint64_t H = ink->config().hidden, rowb = H * 2;
                    HostBuffer th(*dev, 4 * rowb);
                    std::memcpy(th.data(), sn.last_taps.data(), 4 * rowb);
                    for (uint64_t k = 0; k < 4; ++k)
                        stream->upload(th, k * rowb, ink->taps(), k * ink->tap_stride(), rowb);
                    stream->synchronize();
                    st.last_rows = 1;
                    st.tap_row_off = 0;
                    ink_taps_owner = int(s);
                } else {
                    stream->synchronize();
                }
            } else {
                stream->synchronize();
            }
            st.history = sn.tokens;
            st.history_draft = spec && sn.has_draft;
            sn.used = ++ink_snap_tick;
            st.at = st.history.size();
        } else if (live) {
            st.at = st.history.size();
            if (!spec) st.history_draft = false;
        } else {
            // Reclaim stale giant KV buffers (>32K and >2x needed) when global KV exceeds 75% of budget
            const uint32_t need_cap = ink->round_capacity(std::min<uint32_t>(uint32_t(prompt.size() + 16), opt.max_model_len));
            const uint32_t shrink_to = std::max<uint32_t>(32768u, need_cap);
            if (ink->slot_capacity(s) > shrink_to * 2u && ink->kv_total() > (kv_budget * 3u) / 4u) {
                ink->set_capacity(shrink_to);
                kv_used = ink->kv_total();
            }
            ink->reset();
            st.history.clear();
            st.history_draft = spec;
            if (spec && ink_drafter) {
                ink_drafter->select_slot(s);
                ink_drafter->set_slot_allocated(s, true);
                ink_drafter->reset();
            }
        }
        st.salt = salt;
        st.r.reused_tokens = uint32_t(st.at);

        // Encode any images not yet in the reused prefix into this slot's private buffer
        if (!st.job->images.empty()) {
            const uint64_t rowb = uint64_t(ink_vision->out_hidden()) * 2;
            uint64_t rows = 0;
            for (const auto& im : st.job->images) rows += im.tokens();
            if (!st.image_feats || st.image_feats->size() < rows * rowb)
                st.image_feats = std::make_unique<DeviceBuffer>(*dev, rows * rowb);
            size_t k = 0;
            uint64_t off = 0;
            for (size_t i = 0; i < prompt.size() && k < st.job->images.size();) {
                if (prompt[i] != ink_vision->image_token()) { ++i; continue; }
                const auto& im = st.job->images[k++];
                const uint32_t n = im.tokens();
                if (i + n > st.at) {
                    ink_vision->encode(im, *st.image_feats, off);
                    ink->invalidate_sram();
                    st.pending_imgs.push_back({uint32_t(i), n, off});
                }
                off += n * rowb;
                i += n;
            }
        }

        if (prompt.size() > st.at + 2048) {
            for (size_t i = prompt.size() - 32; i-- > st.at + 1024;)
                if (prompt[i] >= 200000 && prompt[i] <= 200003) {
                    const size_t pt = (i / 1024u) * 1024u;
                    if (pt >= st.at + 1024u && pt + 1024u <= prompt.size()) st.points.push_back(pt);
                    break;
                }
        }
        if (st.at < prompt.size()) st.points.push_back(prompt.size());

        // 100% exact prefix cache hit: emit first token immediately and enter Decode!
        if (st.at == prompt.size()) {
            st.t1 = clk::now();
            st.r.prompt_ms = std::chrono::duration<double, std::milli>(st.t1 - st.t0).count();
            const int32_t first = sample_logits(exact_logits, sp, st.rng);
            st.pending_tok = first;
            st.phase = InkSlot::Decode;
            if (!ink_emit(s, first) || ink->slot_position(s) + 1 >= opt.max_model_len)
                ink_finish_slot(s);
        }
    }

    void ink_step_prefill(uint32_t s, bool solo) {
        using clk = std::chrono::steady_clock;
        auto& st = ink_slots[s];
        auto& job = *st.job;
        const auto& prompt = job.prompt;
        const auto& sp = job.sp;

        ink->select_slot(s);
        const bool capture = st.spec && st.history_draft && ink_drafter != nullptr &&
                             ink_drafter->slot_allocated(s) && prompt.size() <= ink_drafter->max_context();
        if (capture) {
            ink_drafter->select_slot(s);
            ink->set_aux_capture(true);
            ink->set_tap_hook([&](uint32_t p0, uint32_t n) {
                ink_drafter->append(ink->taps(), 0, ink->tap_stride(), n, p0);
                ink->invalidate_sram();
                st.last_rows = n;
                ink_taps_owner = int(s);
            });
        } else {
            ink->set_aux_capture(false);
            ink->set_tap_hook({});
        }
        ink->clear_injections();
        for (const auto& im : st.pending_imgs)
            if (im.pos + im.rows > st.at)
                ink->inject_rows(st.image_feats.get(), im.off, im.pos, im.rows);

        constexpr size_t kSeg = 2048;
        std::vector<float> logits;
        bool aborted = false;
        while (st.at < prompt.size()) {
            while (st.pi < st.points.size() && st.points[st.pi] <= st.at) ++st.pi;
            size_t n = std::min(kSeg, prompt.size() - st.at);
            if (st.pi < st.points.size()) n = std::min(n, st.points[st.pi] - st.at);
            const bool last = st.at + n == prompt.size();
            logits = ink->forward(std::span<const int32_t>(prompt.data() + st.at, n), last);
            st.history.insert(st.history.end(), prompt.begin() + long(st.at), prompt.begin() + long(st.at + n));
            st.at += n;
            live_prefill_tokens.fetch_add(n, std::memory_order_relaxed);
            if (!last && st.pi < st.points.size() && st.points[st.pi] == st.at)
                ink_snapshot(s, prompt, st.at, st.salt, capture, st.last_rows, nullptr);
            bool cancel = false;
            {
                std::lock_guard lk(job.mu);
                cancel = job.cancel;
            }
            if (cancel) { st.r.finish = "abort"; aborted = true; break; }
            if (!solo) break;  // yield back to decoding requests after 1 segment!
            // If solo, yield immediately if a new request just arrived in queue so it can be admitted
            if (queued() > 0) break;
        }
        ink->clear_injections();
        ink->set_tap_hook({});
        ink->set_aux_capture(false);

        if (aborted) {
            st.t1 = clk::now();
            st.r.prompt_ms = std::chrono::duration<double, std::milli>(st.t1 - st.t0).count();
            ink_finish_slot(s);
            return;
        }
        if (st.at == prompt.size()) {
            st.t1 = clk::now();
            st.r.prompt_ms = std::chrono::duration<double, std::milli>(st.t1 - st.t0).count();
            st.pending_imgs.clear();
            st.tap_row_off = st.last_rows > 0 ? uint64_t(st.last_rows - 1) * ink->config().hidden * 2 : 0;
            const int32_t first = sample_logits(logits, sp, st.rng);
            st.pending_tok = first;
            st.phase = InkSlot::Decode;
            const bool keep = ink_emit(s, first) && ink->slot_position(s) + 1 < opt.max_model_len;
            if (!keep) ink_finish_slot(s);
        }
    }

    void ink_step_decode_solo(uint32_t s) {
        using clk = std::chrono::steady_clock;
        auto& st = ink_slots[s];
        auto& job = *st.job;
        const auto& sp = job.sp;
        const uint32_t apos = ink->slot_position(s);
        if (apos + 1 >= opt.max_model_len) { ink_finish_slot(s); return; }

        const bool can_spec = st.spec && st.history_draft && ink_drafter != nullptr &&
                              ink_drafter->slot_allocated(s) && apos + 9 < ink_drafter->max_context();
        if (!reserve_ink_kv(s, apos + (can_spec ? 10u : 1u))) {
            ink_finish_slot(s, "KV cache budget exceeded during decode", true);
            return;
        }
        ink->select_slot(s);
        if (can_spec) {
            ink_drafter->select_slot(s);
            ink->set_aux_capture(true);
            const auto c0 = clk::now();
            const bool greedy = sp.greedy();
            double& spec_succ = greedy ? spec_succ_g : spec_succ_s;
            double& spec_tri = greedy ? spec_tri_g : spec_tri_s;
            double* spec_cost = greedy ? spec_cost_g : spec_cost_s;
            const double q = std::clamp(spec_succ / std::max(spec_tri, 1e-9), 0.01, 0.99);
            uint32_t T = 1, Ts = 0;
            double best_s = 1e30;
            if (ink_taps_owner == int(s) && st.plain_left == 0 && apos + 9 < opt.max_model_len) {
                for (uint32_t t = 2; t <= 4; ++t) {
                    if (!ink->can_verify(t)) continue;
                    const double e = (1.0 - std::pow(q, double(t))) / (1.0 - q);
                    if (spec_cost[t] / e < best_s) { best_s = spec_cost[t] / e; Ts = t; }
                }
            }
            if (Ts && (best_s < spec_cost[1] || st.probe)) T = Ts;
            st.probe = false;
            if (T == 1 && st.plain_left == 0 && ink_taps_owner == int(s)) st.plain_left = 8;
            if (T == 1) {
                int32_t nt = 0;
                if (greedy) {
                    nt = ink->forward_argmax(std::span<const int32_t>(&st.pending_tok, 1));
                } else {
                    const auto lg = ink->forward(std::span<const int32_t>(&st.pending_tok, 1), true);
                    nt = sample_logits(lg, sp, st.rng);
                }
                ink_drafter->append(ink->taps(), 0, ink->tap_stride(), 1, apos);
                ink->invalidate_sram();
                ink_taps_owner = int(s);
                st.history.push_back(st.pending_tok);
                st.tap_row_off = 0;
                st.pending_tok = nt;
                spec_cost[1] = 0.9 * spec_cost[1] + 0.1 * std::chrono::duration<double, std::milli>(clk::now() - c0).count();
                if (st.plain_left > 0 && --st.plain_left == 0) st.probe = true;
                st.r.rounds++;
                ink->set_aux_capture(false);
                if (!ink_emit(s, nt) || ink->position() + 1 >= opt.max_model_len) ink_finish_slot(s);
                return;
            }
            const uint64_t V = ink->config().vocab;
            std::vector<int32_t> d;
            uint32_t j = 0;
            int32_t bonus = 0;
            if (greedy) {
                d = ink_drafter->draft(st.pending_tok, apos, ink->taps(), st.tap_row_off, ink->tap_stride(), T - 1);
                ink->invalidate_sram();
                std::vector<int32_t> blk = {st.pending_tok};
                blk.insert(blk.end(), d.begin(), d.end());
                const auto vl = ink->verify(blk);
                auto tgt = [&](uint32_t row) {
                    return int32_t(std::max_element(vl.begin() + long(row * V), vl.begin() + long((row + 1) * V)) -
                                   (vl.begin() + long(row * V)));
                };
                while (j + 1 < T && d[j] == tgt(j)) ++j;
                bonus = tgt(j);
            } else {
                const auto sd = ink_drafter->draft_sampled(st.pending_tok, apos, ink->taps(), st.tap_row_off, ink->tap_stride(),
                                                           sp.temperature, sp.top_k, sp.top_p, sp.min_p, st.rng, T - 1);
                ink->invalidate_sram();
                d = sd.drafts;
                std::vector<int32_t> blk = {st.pending_tok};
                blk.insert(blk.end(), d.begin(), d.end());
                const auto vl = ink->verify(blk);
                const uint32_t K = sd.k;
                auto q_of = [&](uint32_t step, int32_t id) {
                    double qp = 0.0;
                    const size_t base = size_t(step) * K;
                    for (uint32_t k = 0; k < K; ++k)
                        if (sd.cand_ids[base + k] == id) qp += double(sd.cand_q[base + k]);
                    return qp;
                };
                std::uniform_real_distribution<double> u01(0.0, 1.0);
                std::vector<std::pair<int32_t, double>> rej_p;
                while (j + 1 < T) {
                    auto pa = build_target_dist(vl.data() + uint64_t(j) * V, V, sp);
                    double pd = 0.0;
                    for (const auto& [id, pr] : pa)
                        if (id == d[j]) { pd = pr; break; }
                    const double qd = q_of(j, d[j]);
                    if (u01(st.rng) < std::min(1.0, pd / std::max(qd, 1e-30))) {
                        ++j;
                    } else {
                        rej_p = std::move(pa);
                        break;
                    }
                }
                if (j + 1 == T) {
                    bonus = draw_dist(build_target_dist(vl.data() + uint64_t(j) * V, V, sp), st.rng);
                } else {
                    std::vector<double> w;
                    w.reserve(rej_p.size());
                    double total = 0.0;
                    for (const auto& [id, pr] : rej_p) {
                        const double rw = std::max(0.0, pr - q_of(j, id));
                        w.push_back(rw);
                        total += rw;
                    }
                    if (total <= 0.0) {
                        bonus = draw_dist(rej_p, st.rng);
                    } else {
                        double u = std::uniform_real_distribution<double>(0.0, total)(st.rng);
                        bonus = rej_p.back().first;
                        for (size_t idx = 0; idx < rej_p.size(); ++idx) {
                            u -= w[idx];
                            if (u <= 0.0) { bonus = rej_p[idx].first; break; }
                        }
                    }
                }
            }
            uint32_t m = 0;
            int32_t last = st.pending_tok;
            bool stop = false;
            for (uint32_t i = 0; i <= j; ++i) {
                last = i < j ? d[i] : bonus;
                ++m;
                if (!ink_emit(s, last)) { stop = true; break; }
            }
            ink->accept(m);
            ink_drafter->append(ink->taps(), 0, ink->tap_stride(), m, apos);
            ink->invalidate_sram();
            ink_taps_owner = int(s);
            st.history.push_back(st.pending_tok);
            for (uint32_t i = 0; i + 1 < m; ++i) st.history.push_back(d[i]);
            st.pending_tok = last;
            st.tap_row_off = uint64_t(m - 1) * ink->config().hidden * 2;
            spec_succ = 0.95 * spec_succ + j;
            spec_tri = 0.95 * spec_tri + (j + 1 < T ? j + 1 : T - 1);
            spec_cost[T] = 0.9 * spec_cost[T] + 0.1 * std::chrono::duration<double, std::milli>(clk::now() - c0).count();
            st.r.rounds++;
            ink->set_aux_capture(false);
            if (stop || ink->position() + 1 >= opt.max_model_len) ink_finish_slot(s);
            return;
        }
        // Plain Zero-Synapse Direct decode (greedy argmax or sampled logits)
        ink->set_aux_capture(false);
        int32_t nt = 0;
        if (sp.greedy()) {
            nt = ink->forward_argmax(std::span<const int32_t>(&st.pending_tok, 1));
        } else {
            const auto lg = ink->forward(std::span<const int32_t>(&st.pending_tok, 1), true);
            nt = sample_logits(lg, sp, st.rng);
        }
        st.history.push_back(st.pending_tok);
        st.history_draft = false;
        st.pending_tok = nt;
        st.r.rounds++;
        if (!ink_emit(s, nt) || ink->position() + 1 >= opt.max_model_len) ink_finish_slot(s);
    }

    void ink_step_decode_batch(const std::vector<uint32_t>& grp) {
        ink->set_aux_capture(false);
        std::vector<uint32_t> active;
        active.reserve(grp.size());
        for (uint32_t s : grp) {
            const uint32_t pos = ink->slot_position(s);
            if (pos + 1 >= opt.max_model_len) { ink_finish_slot(s); continue; }
            if (!reserve_ink_kv(s, pos + 1)) {
                ink_finish_slot(s, "KV cache budget exceeded during batched decode", true);
                continue;
            }
            active.push_back(s);
        }
        if (active.empty()) return;
        if (active.size() == 1) { ink_step_decode_solo(active[0]); return; }
        std::vector<int32_t> toks(active.size());
        std::vector<uint8_t> want_lg(active.size());
        for (size_t i = 0; i < active.size(); ++i) {
            auto& st = ink_slots[active[i]];
            toks[i] = st.pending_tok;
            want_lg[i] = st.job->sp.greedy() ? 0u : 1u;
            st.history_draft = false;
        }
        ink_taps_owner = -1;
        auto out = ink->decode_batch(active, toks, want_lg);
        for (size_t i = 0; i < active.size(); ++i) {
            const uint32_t s = active[i];
            auto& st = ink_slots[s];
            st.history.push_back(st.pending_tok);
            st.r.rounds++;
            const int32_t nt = want_lg[i] ? sample_logits(out.logits[i], st.job->sp, st.rng) : out.argmax[i];
            st.pending_tok = nt;
            if (!ink_emit(s, nt) || ink->slot_position(s) + 1 >= opt.max_model_len) ink_finish_slot(s);
        }
    }

    void ink_scheduler() {
        static const bool no_batch = std::getenv("G2E_NO_BATCH") != nullptr;
        while (true) {
            std::vector<std::pair<uint32_t, std::shared_ptr<Job>>> starting;
            {
                std::unique_lock lk(sched_mu);
                sched_cv.wait(lk, [&] {
                    if (!queue.empty()) return true;
                    for (const auto& st : ink_slots) if (st.phase != InkSlot::Idle) return true;
                    return false;
                });
                uint64_t held = 0;
                for (uint32_t s = 0; s < ink_slots.size(); ++s)
                    if (ink_slots[s].phase != InkSlot::Idle) held += ink->slot_kv(s);
                while (!queue.empty()) {
                    auto& job = queue.front();
                    const uint64_t need = ink_kv_estimate(*job);
                    if (need > kv_budget) {
                        std::lock_guard jl(job->mu);
                        job->error = "not enough device memory for the KV cache of this request";
                        job->done = true;
                        job->cv.notify_one();
                        queue.pop_front();
                        continue;
                    }
                    if (held + need > kv_budget) break;
                    const uint64_t salt = ink_job_salt(*job);
                    const bool spec = ink_job_spec(*job);
                    const int s = pick_ink_slot(*job, salt, spec);
                    if (s < 0) break;
                    held += need;
                    ink_slots[size_t(s)].phase = InkSlot::Prefill;  // reserve before unlocking
                    starting.push_back({uint32_t(s), job});
                    queue.pop_front();
                }
            }
            for (auto& [s, job] : starting) {
                try {
                    ink_begin_slot(s, std::move(job));
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[g2e] inkling2 slot %u begin: %s\n", s, e.what());
                    ink_finish_slot(s, e.what(), true);
                }
            }
            // 1: collect active decoding slots & oldest prefilling slot
            std::vector<uint32_t> decoding;
            int oldest_pf = -1;
            for (uint32_t s = 0; s < ink_slots.size(); ++s) {
                if (ink_slots[s].phase == InkSlot::Decode) {
                    decoding.push_back(s);
                } else if (ink_slots[s].phase == InkSlot::Prefill) {
                    if (oldest_pf < 0 || ink_slots[s].used < ink_slots[size_t(oldest_pf)].used) oldest_pf = int(s);
                }
            }
            // 2: run decode step (batched Bp in {2,4,8} when >= 2 decoding, else solo Direct / DSpark)
            if (decoding.size() >= 2 && !no_batch && ink->can_decode_batch()) {
                try {
                    live_decode_rounds.fetch_add(decoding.size(), std::memory_order_relaxed);
                    ink_step_decode_batch(decoding);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[g2e] inkling2 batched decode: %s\n", e.what());
                    for (uint32_t s : decoding) ink_finish_slot(s, e.what(), true);
                }
            } else {
                for (uint32_t s : decoding) {
                    try {
                        live_decode_rounds.fetch_add(1, std::memory_order_relaxed);
                        ink_step_decode_solo(s);
                    } catch (const std::exception& e) {
                        std::fprintf(stderr, "[g2e] inkling2 slot %u decode: %s\n", s, e.what());
                        ink_finish_slot(s, e.what(), true);
                    }
                }
            }
            // 3: advance oldest prefill slot (1 segment of 2048 if decoding is active, else continuous)
            if (oldest_pf >= 0) {
                const uint32_t s = uint32_t(oldest_pf);
                try {
                    ink_step_prefill(s, decoding.empty());
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "[g2e] inkling2 slot %u prefill: %s\n", s, e.what());
                    ink_finish_slot(s, e.what(), true);
                }
            }
            uint32_t active = 0, ndec = 0, npf = 0;
            for (const auto& st : ink_slots) {
                active += (st.phase != InkSlot::Idle);
                ndec += (st.phase == InkSlot::Decode);
                npf += (st.phase == InkSlot::Prefill);
            }
            slots_active = active;
            slots_decoding = ndec;
            slots_prefilling = npf;
            const uint64_t kvu = ink->kv_total();
            if (kvu != kv_used.load(std::memory_order_relaxed) || !starting.empty() || active == 0) {
                kv_used = kvu;
                device_free_bytes = dev->info().free_bytes;
            }
        }
    }

    // ---- request decoding (vLLM semantics) ----------------------------------------------------------------------
    SamplingParams sampling(const ojson& req, uint32_t prompt_len) const {
        SamplingParams p;
        auto num = [&](const char* k, double dflt) {
            if (req.contains(k) && !req[k].is_null()) return req[k].get<double>();
            if (gen_defaults.contains(k)) return gen_defaults[k].get<double>();
            return dflt;
        };
        p.temperature = float(num("temperature", 1.0));
        p.top_p = float(num("top_p", 1.0));
        p.top_k = int(num("top_k", 0));
        if (p.top_k < 0) p.top_k = 0;  // vLLM: -1 disables
        p.min_p = float(num("min_p", 0.0));
        if (req.contains("seed") && !req["seed"].is_null()) p.seed = req["seed"].get<int64_t>();
        if (req.contains("stop_token_ids") && req["stop_token_ids"].is_array())
            for (auto& t : req["stop_token_ids"]) p.stop_token_ids.push_back(t.get<int32_t>());
        p.ignore_eos = req.value("ignore_eos", false);
        const uint32_t room = opt.max_model_len > prompt_len ? opt.max_model_len - prompt_len : 0;
        int64_t mt = -1;
        for (const char* k : {"max_completion_tokens", "max_tokens"})
            if (req.contains(k) && !req[k].is_null()) { mt = req[k].get<int64_t>(); break; }
        if (prompt_len >= opt.max_model_len)
            throw HttpError(400, "This model's maximum context length is " + std::to_string(opt.max_model_len) +
                                     " tokens. However, your request has " + std::to_string(prompt_len) +
                                     " input tokens. Please reduce the length of the input messages.");
        if (mt == -1) {
            p.max_tokens = room;
        } else {
            if (mt < 1) throw HttpError(400, "max_tokens must be at least 1, got " + std::to_string(mt) + ".");
            if (uint64_t(mt) > room)
                throw HttpError(400, "This model's maximum context length is " + std::to_string(opt.max_model_len) +
                                         " tokens. However, you requested " + std::to_string(prompt_len + mt) + " tokens (" +
                                         std::to_string(prompt_len) + " in the messages, " + std::to_string(mt) +
                                         " in the completion). Please reduce the length of the messages or completion.");
            p.max_tokens = uint32_t(mt);
        }
        return p;
    }

    // An image from a data URL (data:image/...;base64,...) -> processor patches.
    VisionImage load_image(const std::string& url) const {
        static const std::string b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        if (url.rfind("data:", 0) != 0)
            throw HttpError(400, "only data: image URLs (base64) are supported by this server");
        const size_t comma = url.find(',');
        if (comma == std::string::npos || url.substr(0, comma).find(";base64") == std::string::npos)
            throw HttpError(400, "image data URL must be base64-encoded");
        std::vector<uint8_t> bytes;
        bytes.reserve((url.size() - comma) * 3 / 4);
        uint32_t acc = 0;
        int bits = 0;
        for (size_t i = comma + 1; i < url.size(); ++i) {
            const char c = url[i];
            if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
            const auto v = b64.find(c == '-' ? '+' : c == '_' ? '/' : c);
            if (v == std::string::npos) throw HttpError(400, "invalid base64 in the image data URL");
            acc = (acc << 6) | uint32_t(v);
            bits += 6;
            if (bits >= 8) {
                bits -= 8;
                bytes.push_back(uint8_t((acc >> bits) & 0xFF));
            }
        }
        int w = 0, h = 0;
        std::vector<uint8_t> rgb;
        try {
            rgb = decode_image(bytes.data(), bytes.size(), w, h);
        } catch (const std::exception& e) {
            throw HttpError(400, e.what());
        }
        if (ink_vision) return preprocess_inkling_image(rgb.data(), w, h, ink_max_image_tokens, ink_vision->patch(), ink_vision->merge());
        return preprocess_image(rgb.data(), w, h, vision->min_pixels(), opt.max_image_pixels);
    }
    int32_t image_token() const { return ink_vision ? ink_vision->image_token() : vision->image_token(); }

    static std::vector<std::string> stop_strings(const ojson& req) {
        std::vector<std::string> s;
        if (req.contains("stop")) {
            if (req["stop"].is_string()) s.push_back(req["stop"]);
            else if (req["stop"].is_array()) for (auto& x : req["stop"]) s.push_back(x);
        }
        s.erase(std::remove(s.begin(), s.end(), std::string()), s.end());
        return s;
    }

    // Template variables from a chat request (assistant tool-call arguments become objects, as vLLM does).
    // `images` receives the image sources (URLs) in the order the template renders their placeholders.
    ojson chat_vars(const ojson& req, std::vector<std::string>* images = nullptr) const {
        if (!req.contains("messages") || !req["messages"].is_array() || req["messages"].empty())
            throw HttpError(400, "messages must be a non-empty list");
        ojson msgs = ojson::array();
        for (auto m : req["messages"]) {
            if (m.contains("content") && m["content"].is_array()) {
                for (const auto& part : m["content"]) {
                    const std::string type = part.value("type", "text");
                    if (type == "image_url" || type == "image") {
                        if (!vision && !ink_vision) throw HttpError(400, "this server was started without image input (--vision-pack)");
                        std::string url;
                        if (part.contains("image_url")) {
                            const auto& iu = part["image_url"];
                            url = iu.is_string() ? iu.get<std::string>() : iu.value("url", "");
                        } else if (part.contains("image") && part["image"].is_string()) {
                            url = part["image"].get<std::string>();
                        }
                        if (url.empty()) throw HttpError(400, "image content without a url");
                        if (images) images->push_back(url);
                    } else if (type != "text") {
                        throw HttpError(400, "content type '" + type + "' is not supported by this server");
                    }
                }
            }
            if (m.contains("tool_calls") && m["tool_calls"].is_array())
                for (auto& tc : m["tool_calls"])
                    if (tc.contains("function") && tc["function"].contains("arguments") && tc["function"]["arguments"].is_string()) {
                        auto parsed = ojson::parse(tc["function"]["arguments"].get<std::string>(), nullptr, false);
                        if (!parsed.is_discarded()) tc["function"]["arguments"] = parsed;
                    }
            if ((!m.contains("reasoning_content") || m["reasoning_content"].is_null()) &&
                m.contains("reasoning") && m["reasoning"].is_string()) {
                m["reasoning_content"] = m["reasoning"];
            }
            msgs.push_back(m);
        }
        ojson v;
        v["messages"] = msgs;
        if (req.contains("tools") && req["tools"].is_array() && !req["tools"].empty()) v["tools"] = req["tools"];
        const bool cont = req.value("continue_final_message", false);
        v["add_generation_prompt"] = req.value("add_generation_prompt", !cont);
        // OpenAI's reasoning_effort: the Inkling template takes it as is (none, minimal, low, medium, high, xhigh, max
        // or a number in [0, 0.99]); explicit chat_template_kwargs below win
        if (inkling && req.contains("reasoning_effort") && !req["reasoning_effort"].is_null()) {
            auto e = req["reasoning_effort"];
            if (e.is_string()) {
                std::string x = e.get<std::string>();
                std::transform(x.begin(), x.end(), x.begin(), [](unsigned char c) { return char(std::tolower(c)); });
                if (x == "off") x = "none";
                e = x;
            }
            v["reasoning_effort"] = e;
        }
        if (req.contains("chat_template_kwargs") && req["chat_template_kwargs"].is_object())
            for (auto& [k, val] : req["chat_template_kwargs"].items()) v[k] = val;
        return v;
    }

    // ---- one generation with stop strings, streaming decoded text ------------------------------------------
    struct TextResult {
        GenerationResult gen;
        std::string text;
        ojson stop_reason = nullptr;
        std::string finish;
    };
    // on_text(delta, tokens generated so far) returns false when the client is gone.
    TextResult run(const std::vector<int32_t>& prompt, const SamplingParams& sp, const std::vector<std::string>& stops,
                   bool skip_special, bool include_stop, const std::function<bool(const std::string&, uint32_t)>& on_text,
                   const std::vector<VisionImage>& images = {}) {
        TextResult out;
        Detokenizer detok(*tok, skip_special);
        size_t sent = 0, hold = 0;
        for (const auto& s : stops) hold = std::max(hold, s.size() - 1);
        bool stop_hit = false, client_gone = false;
        auto check_stop = [&]() {
            for (const auto& s : stops) {
                const size_t from = sent > s.size() ? sent - s.size() : 0;
                const size_t at = out.text.find(s, from);
                if (at != std::string::npos) {
                    out.text.resize(include_stop ? at + s.size() : at);
                    out.stop_reason = s;
                    return true;
                }
            }
            return false;
        };
        uint32_t generated = 0;
        auto emit = [&](size_t upto) {
            if (upto > sent) {
                if (!on_text(out.text.substr(sent, upto - sent), generated)) client_gone = true;
                sent = upto;
            }
        };
        // the request goes to the scheduler (one engine thread, requests interleaved over the model slots); this
        // thread turns its tokens into text
        auto job = std::make_shared<Job>();
        job->prompt = prompt;
        job->sp = sp;
        job->images = images;
        submit(job);
        bool cancelled = false;
        while (true) {
            std::vector<int32_t> toks;
            bool fin;
            {
                std::unique_lock lk(job->mu);
                job->cv.wait(lk, [&] { return !job->tokens.empty() || job->done; });
                toks.swap(job->tokens);
                fin = job->done;
            }
            if (!stop_hit && !client_gone) {
                generated += uint32_t(toks.size());
                for (auto t : toks) out.text += detok.push(t);
                if (!stops.empty() && check_stop()) {
                    stop_hit = true;
                    emit(out.text.size());
                } else {
                    emit(out.text.size() > hold ? out.text.size() - hold : 0);
                }
            }
            if ((stop_hit || client_gone) && !cancelled) {
                std::lock_guard lk(job->mu);
                job->cancel = true;
                cancelled = true;
            }
            if (fin) break;
        }
        if (!job->error.empty()) throw std::runtime_error(job->error);
        out.gen = std::move(job->gen);
        if (!stop_hit) {
            out.text += detok.flush();
            if (!stops.empty() && check_stop()) stop_hit = true;
            emit(out.text.size());
        }
        out.finish = stop_hit ? "stop" : out.gen.finish;
        if (!stop_hit && out.gen.stop_token >= 0) out.stop_reason = out.gen.stop_token;
        const double tps = out.gen.decode_ms > 0 ? out.gen.tokens.size() / out.gen.decode_ms * 1e3 : 0;
        std::lock_guard slk(stats_mu);
        ++totals.requests;
        totals.prompt_tokens += out.gen.prompt_tokens;
        totals.cached_tokens += out.gen.reused_tokens;
        totals.cached_requests += out.gen.reused_tokens > 0;
        std::fprintf(stderr, "[g2e] prompt %u (cached %u, %.0f%%; all requests %.1f%%) %.0f ms | generated %zu in %.0f ms (%.1f tok/s) | rounds %u, %.2f tokens/round | %s\n",
                     out.gen.prompt_tokens, out.gen.reused_tokens,
                     out.gen.prompt_tokens ? 100.0 * out.gen.reused_tokens / out.gen.prompt_tokens : 0.0,
                     totals.prompt_tokens ? 100.0 * double(totals.cached_tokens) / double(totals.prompt_tokens) : 0.0,
                     out.gen.prompt_ms, out.gen.tokens.size(), out.gen.decode_ms,
                     tps, out.gen.rounds, out.gen.rounds ? double(out.gen.drafts_accepted + out.gen.rounds) / out.gen.rounds : 0.0,
                     out.finish.c_str());
        return out;
    }

    ojson usage(const TextResult& r) const {
        const uint32_t c = uint32_t(r.gen.tokens.size());
        return {{"prompt_tokens", r.gen.prompt_tokens}, {"total_tokens", r.gen.prompt_tokens + c}, {"completion_tokens", c},
                {"prompt_tokens_details", {{"cached_tokens", r.gen.reused_tokens}}}};
    }
    ojson timings(const TextResult& r) const {
        const uint32_t p_fresh = r.gen.prompt_tokens > r.gen.reused_tokens ? r.gen.prompt_tokens - r.gen.reused_tokens : 0u;
        const uint32_t c = uint32_t(r.gen.tokens.size());
        return {{"prompt_n", p_fresh}, {"cache_n", r.gen.reused_tokens}, {"prompt_ms", r.gen.prompt_ms},
                {"prompt_per_second", r.gen.prompt_ms > 0.0 ? double(p_fresh) / (r.gen.prompt_ms * 1e-3) : 0.0},
                {"predicted_n", c}, {"predicted_ms", r.gen.decode_ms},
                {"predicted_per_second", r.gen.decode_ms > 0.0 ? double(c) / (r.gen.decode_ms * 1e-3) : 0.0}};
    }
};

std::string sse(const ojson& j) { return "data: " + j.dump() + "\n\n"; }

// the output parser of the served model: Qwen3 (<think>, XML tool calls) or Inkling-2 (content blocks)
struct OutputParser {
    std::unique_ptr<Qwen3Parser> qwen;
    std::unique_ptr<InklingParser> ink;
    OutputParser(bool inkling, bool reasoning, bool thinking, const ojson& tools) {
        if (inkling) ink = std::make_unique<InklingParser>();
        else qwen = std::make_unique<Qwen3Parser>(reasoning, thinking, tools);
    }
    ParserDelta feed(const std::string& text, bool final) { return ink ? ink->feed(text, final) : qwen->feed(text, final); }
    int tool_calls_seen() const { return ink ? ink->tool_calls_seen() : qwen->tool_calls_seen(); }
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--model-dir") opt.model_dir = next();
        else if (a == "--target-pack") opt.target_pack = next();
        else if (a == "--draft-pack") opt.draft_pack = next();
        else if (a == "--served-model-name") opt.served_name = next();
        else if (a == "--host") opt.host = next();
        else if (a == "--port") opt.port = std::stoi(next());
        else if (a == "--max-model-len") opt.max_model_len = uint32_t(std::stoul(next()));
        else if (a == "--slots") opt.slots = std::max<uint32_t>(1, uint32_t(std::stoul(next())));
        else if (a == "--kv-cache-gib") opt.kv_gib = std::stod(next());
        else if (a == "--api-key") opt.api_key = next();
        else if (a == "--reasoning-parser") opt.reasoning_parser = next();
        else if (a == "--vision-pack") opt.vision_pack = next();
        else if (a == "--max-image-pixels") opt.max_image_pixels = std::stoull(next());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (opt.model_dir.empty() || opt.target_pack.empty()) {
        std::fprintf(stderr, "usage: %s --model-dir DIR --target-pack T.g2w [--draft-pack D.g2w] ...\n", argv[0]);
        return 2;
    }
    opt.max_model_len = (opt.max_model_len + 127) / 128 * 128;
    if (opt.served_name.empty()) opt.served_name = opt.model_dir;
    setvbuf(stderr, nullptr, _IOLBF, 0);
    Server S;
    S.opt = opt;
    try {
        {
            WeightPack probe(opt.target_pack);
            S.inkling = probe.header_json().find("\"inkling2_loop\"") != std::string::npos;
        }
        S.load();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[g2e] startup failed: %s\n", e.what());
        return 1;
    }
    httplib::Server http;
    http.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        if (!opt.api_key.empty() && req.path.rfind("/v1/", 0) == 0 && req.get_header_value("Authorization") != "Bearer " + opt.api_key) {
            res.status = 401;
            res.set_content(R"({"error":"Unauthorized"})", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.status = 200; });
    http.Get("/version", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"version":"g2e-0.1"})", "application/json");
    });
    http.Get("/stats", [&](const httplib::Request&, httplib::Response& res) {
        ojson j;
        {
            std::lock_guard lock(S.stats_mu);
            const auto& t = S.totals;
            j["requests"] = t.requests;
            j["prompt_tokens"] = t.prompt_tokens;
            j["cached_tokens"] = t.cached_tokens;
            j["token_hit_rate"] = t.prompt_tokens ? double(t.cached_tokens) / double(t.prompt_tokens) : 0.0;
            j["request_hit_rate"] = t.requests ? double(t.cached_requests) / double(t.requests) : 0.0;
            j["slots"] = {{"total", S.inkling ? S.opt.slots : S.gens.size()}, {"active", S.slots_active.load()},
                          {"decoding", S.slots_decoding.load(std::memory_order_relaxed)},
                          {"prefilling", S.slots_prefilling.load(std::memory_order_relaxed)},
                          {"queued", S.queued()},
                          {"kv_used_gib", double(S.kv_used.load()) / (1ull << 30)},
                          {"kv_budget_gib", double(S.kv_budget) / (1ull << 30)}, {"kv_evictions", S.kv_evictions.load()}};
            if (S.inkling) {
                j["slots"]["kv_budget_tokens"] = S.ink_kv_budget_tokens;
                j["slots"]["device_free_gib"] = double(S.device_free_bytes.load(std::memory_order_relaxed)) / double(1ull << 30);
                j["slots"]["device_total_gib"] = double(S.device_total_bytes.load(std::memory_order_relaxed)) / double(1ull << 30);
                j["engine"] = S.ink->engine_summary();
                j["bf16_cache"] = {{"mode", S.ink_bf16_mode},
                                   {"cached_layers", S.ink_bf16_cached},
                                   {"total_layers", S.ink_bf16_total},
                                   {"staging", S.ink_bf16_staging},
                                   {"bytes_gib", double(S.ink_bf16_bytes) / double(1ull << 30)},
                                   {"kv_budget_gib", double(S.kv_budget) / double(1ull << 30)},
                                   {"kv_budget_tokens", S.ink_kv_budget_tokens},
                                   {"drafter_kv_reserve_gib", double(S.ink_dspark_kv_reserve) / double(1ull << 30)}};
            }
            j["live"] = {{"decode_tokens", S.live_decode_tokens.load(std::memory_order_relaxed)},
                         {"decode_rounds", S.live_decode_rounds.load(std::memory_order_relaxed)},
                         {"prefill_tokens", S.live_prefill_tokens.load(std::memory_order_relaxed)},
                         {"decoding_slots", S.slots_decoding.load(std::memory_order_relaxed)},
                         {"prefilling_slots", S.slots_prefilling.load(std::memory_order_relaxed)}};
            if (S.prefix) {
                const auto& c = S.prefix_stats;
                j["prefix_cache"] = {{"host_gib", double(c.host_bytes) / (1ull << 30)},
                                     {"disk_gib", double(c.disk_bytes) / (1ull << 30)}, {"spare_pinned_gib", double(c.spare_bytes) / (1ull << 30)},
                                     {"chunks", c.chunks}, {"snapshots", c.snapshots}, {"hits", c.hits},
                                     {"misses", c.misses}, {"restored_tokens", c.restored_tokens}};
            }
        }
        res.set_content(j.dump(), "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        ojson perm = {{"id", random_id("modelperm-")}, {"object", "model_permission"}, {"created", S.started},
                      {"allow_create_engine", false}, {"allow_sampling", true}, {"allow_logprobs", true},
                      {"allow_search_indices", false}, {"allow_view", true}, {"allow_fine_tuning", false},
                      {"organization", "*"}, {"group", nullptr}, {"is_blocking", false}};
        ojson m = {{"id", opt.served_name}, {"object", "model"}, {"created", S.started}, {"owned_by", "g2e"},
                   {"root", opt.model_dir}, {"parent", nullptr}, {"max_model_len", opt.max_model_len},
                   {"permission", ojson::array({perm})}};
        res.set_content(ojson({{"object", "list"}, {"data", ojson::array({m})}}).dump(), "application/json");
    });

    // ---- shared handler for both endpoints ----------------------------------------------------------------------
    auto handle = [&](const httplib::Request& hreq, httplib::Response& res, bool chat) {
        try {
            auto req = ojson::parse(hreq.body, nullptr, false);
            if (req.is_discarded() || !req.is_object()) throw HttpError(400, "the request body is not a JSON object");
            if (req.contains("n") && !req["n"].is_null() && req["n"].get<int>() != 1) throw HttpError(400, "only n=1 is supported");
            const bool reasoning = chat && (S.opt.reasoning_parser == "qwen3" || S.inkling);
            std::vector<int32_t> prompt;
            std::vector<VisionImage> images;
            ojson tools = nullptr;
            bool thinking = true;
            if (chat) {
                std::vector<std::string> urls;
                const auto vars = S.chat_vars(req, &urls);
                if (vars.contains("enable_thinking") && vars["enable_thinking"].is_boolean()) thinking = vars["enable_thinking"];
                const std::string tool_choice = req.contains("tool_choice") && req["tool_choice"].is_string() ? req["tool_choice"].get<std::string>() : "auto";
                if (vars.contains("tools") && tool_choice != "none") tools = vars["tools"];
                prompt = S.tok->encode(S.tpl->render(vars));
                if (!urls.empty()) {  // decode + preprocess, then one placeholder token per merged patch
                    for (const auto& u : urls) images.push_back(S.load_image(u));
                    std::vector<int32_t> expanded;
                    size_t k = 0;
                    for (int32_t t : prompt) {
                        if (t == S.image_token() && k < images.size())
                            expanded.insert(expanded.end(), images[k++].tokens(), t);
                        else
                            expanded.push_back(t);
                    }
                    if (k != images.size()) throw HttpError(400, "the chat template did not place every image");
                    prompt = std::move(expanded);
                }
            } else {
                if (!req.contains("prompt")) throw HttpError(400, "prompt is required");
                const auto& p = req["prompt"];
                if (p.is_string()) prompt = S.tok->encode(p.get<std::string>());
                else if (p.is_array() && !p.empty() && p[0].is_number_integer()) for (auto& t : p) prompt.push_back(t.get<int32_t>());
                else if (p.is_array() && p.size() == 1 && p[0].is_string()) prompt = S.tok->encode(p[0].get<std::string>());
                else throw HttpError(400, "prompt must be a string, a list with one string, or a list of token ids");
            }
            if (prompt.empty()) throw HttpError(400, "the prompt is empty");
            const auto sp = S.sampling(req, uint32_t(prompt.size()));
            const auto stops = Server::stop_strings(req);
            // Inkling chat: the parser needs the block markers (special tokens)
            const bool skip_special = (chat && S.inkling) ? false : req.value("skip_special_tokens", true);
            const bool include_stop = req.value("include_stop_str_in_output", false);
            const bool stream = req.value("stream", false);
            const bool include_usage = stream && req.contains("stream_options") && req["stream_options"].is_object() &&
                                       req["stream_options"].value("include_usage", false);
            const bool continuous_usage = include_usage && req["stream_options"].value("continuous_usage_stats", false);
            const std::string id = random_id(chat ? "chatcmpl-" : "cmpl-");
            const int64_t created = std::time(nullptr);
            const std::string object = chat ? "chat.completion" : "text_completion";
            const std::string model_name = req.value("model", S.opt.served_name);

            if (!stream) {
                auto r = S.run(prompt, sp, stops, skip_special, include_stop, [](const std::string&, uint32_t) { return true; }, images);
                ojson choice = {{"index", 0}};
                std::string finish = r.finish == "abort" ? "stop" : r.finish;
                if (chat) {
                    OutputParser parser(S.inkling, reasoning, thinking, tools);
                    const auto d = parser.feed(r.text, true);
                    ojson msg = {{"role", "assistant"}};
                    const bool called = !d.tool_calls.empty();
                    const std::string content = called && d.content.find_first_not_of(" \t\r\n") == std::string::npos ? "" : d.content;
                    msg["content"] = content.empty() && called ? ojson(nullptr) : ojson(content);
                    msg["refusal"] = nullptr;
                    msg["annotations"] = nullptr;
                    msg["audio"] = nullptr;
                    msg["function_call"] = nullptr;
                    if (called) {
                        ojson calls = ojson::array();
                        for (const auto& c : d.tool_calls)
                            calls.push_back({{"id", c.id}, {"type", "function"}, {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
                        msg["tool_calls"] = calls;
                        finish = "tool_calls";
                    }
                    const ojson rval = reasoning && !d.reasoning.empty() ? ojson(d.reasoning) : ojson(nullptr);
                    msg["reasoning"] = rval;
                    msg["reasoning_content"] = rval;
                    choice["message"] = msg;
                } else {
                    choice["text"] = r.text;
                }
                choice["logprobs"] = nullptr;
                choice["finish_reason"] = finish;
                choice["stop_reason"] = r.stop_reason;
                if (!chat) choice["prompt_logprobs"] = nullptr;
                ojson resp = {{"id", id}, {"object", object}, {"created", created}, {"model", model_name},
                              {"choices", ojson::array({choice})}};
                if (chat) {
                    resp["service_tier"] = nullptr;
                    resp["system_fingerprint"] = nullptr;
                }
                resp["usage"] = S.usage(r);
                resp["timings"] = S.timings(r);
                resp["prompt_logprobs"] = nullptr;
                resp["prompt_token_ids"] = nullptr;
                resp["kv_transfer_params"] = nullptr;
                res.set_content(resp.dump(), "application/json");
                return;
            }
            // ---- streaming
            res.set_header("Cache-Control", "no-cache");
            res.set_chunked_content_provider("text/event-stream", [&S, prompt, sp, stops, skip_special, include_stop, include_usage, continuous_usage, id,
                                                                    created, model_name, chat, reasoning, thinking, tools, images](size_t, httplib::DataSink& sink) {
                const std::string obj = chat ? "chat.completion.chunk" : "text_completion";
                const uint32_t prompt_tokens = uint32_t(prompt.size());
                uint32_t done_tokens = 0;  // generated so far (continuous usage)
                auto chunk = [&](ojson choice) {
                    ojson c = {{"id", id}, {"object", obj}, {"created", created}, {"model", model_name}, {"choices", ojson::array({choice})}};
                    if (continuous_usage)
                        c["usage"] = {{"prompt_tokens", prompt_tokens}, {"total_tokens", prompt_tokens + done_tokens},
                                      {"completion_tokens", done_tokens}};
                    return c;
                };
                auto write = [&](const std::string& s) { return sink.write(s.data(), s.size()); };
                bool alive = true;
                if (chat) alive = write(sse(chunk({{"index", 0}, {"delta", {{"role", "assistant"}, {"content", ""}}}, {"logprobs", nullptr}, {"finish_reason", nullptr}})));
                OutputParser parser(S.inkling, reasoning, thinking, tools);
                int tool_index = 0;
                auto delta_json = [&](const ParserDelta& d) {
                    ojson delta = ojson::object();
                    if (!d.reasoning.empty()) {
                        delta["reasoning"] = d.reasoning;
                        delta["reasoning_content"] = d.reasoning;
                    }
                    if (!d.content.empty()) delta["content"] = d.content;
                    if (!d.tool_calls.empty()) {
                        ojson calls = ojson::array();
                        for (const auto& c : d.tool_calls)
                            calls.push_back({{"index", tool_index++}, {"id", c.id}, {"type", "function"},
                                             {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
                        delta["tool_calls"] = calls;
                    }
                    return delta;
                };
                auto r = S.run(prompt, sp, stops, skip_special, include_stop, [&](const std::string& text, uint32_t generated) {
                    if (!alive) return false;
                    done_tokens = generated;
                    if (chat) {
                        const auto d = parser.feed(text, false);
                        if (!d.empty())
                            alive = write(sse(chunk({{"index", 0}, {"delta", delta_json(d)}, {"logprobs", nullptr}, {"finish_reason", nullptr}})));
                    } else {
                        alive = write(sse(chunk({{"index", 0}, {"text", text}, {"logprobs", nullptr}, {"finish_reason", nullptr}, {"stop_reason", nullptr}})));
                    }
                    return alive;
                }, images);
                if (alive) {
                    done_tokens = uint32_t(r.gen.tokens.size());
                    std::string finish = r.finish == "abort" ? "stop" : r.finish;
                    if (chat) {
                        const auto d = parser.feed("", true);
                        if (parser.tool_calls_seen() > 0) finish = "tool_calls";
                        write(sse(chunk({{"index", 0}, {"delta", delta_json(d)}, {"logprobs", nullptr}, {"finish_reason", finish},
                                         {"stop_reason", r.stop_reason}})));
                    } else {
                        write(sse(chunk({{"index", 0}, {"text", ""}, {"logprobs", nullptr}, {"finish_reason", finish}, {"stop_reason", r.stop_reason}})));
                    }
                    if (include_usage)
                        write(sse({{"id", id}, {"object", obj}, {"created", created}, {"model", model_name}, {"choices", ojson::array()},
                                   {"usage", S.usage(r)}, {"timings", S.timings(r)}}));
                    write("data: [DONE]\n\n");
                }
                sink.done();
                return true;
            });
        } catch (const HttpError& e) {
            res.status = e.status;
            res.set_content(error_json(e.what(), e.type, e.status), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_json(e.what(), "InternalServerError", 500), "application/json");
        }
    };
    http.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& r) { handle(q, r, true); });
    http.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& r) { handle(q, r, false); });
    http.Post("/v1/tokenize", [&](const httplib::Request& hreq, httplib::Response& res) {
        try {
            auto req = ojson::parse(hreq.body, nullptr, false);
            if (req.is_discarded() || !req.is_object()) throw HttpError(400, "the request body is not a JSON object");
            std::vector<int32_t> toks;
            if (req.contains("messages")) {
                std::vector<std::string> urls;
                const auto vars = S.chat_vars(req, &urls);
                toks = S.tok->encode(S.tpl->render(vars));
                if (!urls.empty()) {
                    size_t extra = 0;
                    for (const auto& u : urls) {
                        const auto im = S.load_image(u);
                        if (im.tokens() > 1) extra += im.tokens() - 1;
                    }
                    ojson out = {{"count", toks.size() + extra}, {"max_model_len", opt.max_model_len}};
                    res.set_content(out.dump(), "application/json");
                    return;
                }
            } else if (req.contains("prompt") && req["prompt"].is_string()) {
                toks = S.tok->encode(req["prompt"].get<std::string>());
            } else {
                throw HttpError(400, "messages or prompt is required");
            }
            ojson out = {{"count", toks.size()}, {"max_model_len", opt.max_model_len}};
            res.set_content(out.dump(), "application/json");
        } catch (const HttpError& e) {
            res.status = e.status;
            res.set_content(error_json(e.what(), e.type, e.status), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(error_json(e.what(), "InternalServerError", 500), "application/json");
        }
    });
    std::fprintf(stderr, "[g2e] serving %s on %s:%d (max_model_len %u%s)\n", opt.served_name.c_str(), opt.host.c_str(), opt.port,
                 opt.max_model_len, S.inkling ? ", Inkling-2 looped MoE" : S.drafter ? ", DSpine speculative decoding for greedy requests" : "");
    if (!http.listen(opt.host, opt.port)) {
        std::fprintf(stderr, "[g2e] cannot listen on %s:%d\n", opt.host.c_str(), opt.port);
        return 1;
    }
    return 0;
}
