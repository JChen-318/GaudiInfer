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
#include "g2e/prefix_cache.hpp"
#include "g2e/vision.hpp"
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
    std::atomic<uint64_t> kv_used{0}, kv_evictions{0};
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
        return preprocess_image(rgb.data(), w, h, vision->min_pixels(), opt.max_image_pixels);
    }

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
                        if (!vision) throw HttpError(400, "this server was started without image input (--vision-pack)");
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
};

std::string sse(const ojson& j) { return "data: " + j.dump() + "\n\n"; }

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
            j["slots"] = {{"total", S.gens.size()}, {"active", S.slots_active.load()},
                          {"decoding", S.slots_decoding.load(std::memory_order_relaxed)},
                          {"prefilling", S.slots_prefilling.load(std::memory_order_relaxed)},
                          {"queued", S.queued()},
                          {"kv_used_gib", double(S.kv_used.load()) / (1ull << 30)},
                          {"kv_budget_gib", double(S.kv_budget) / (1ull << 30)}, {"kv_evictions", S.kv_evictions.load()}};
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
            const bool reasoning = chat && S.opt.reasoning_parser == "qwen3";
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
                        if (t == S.vision->image_token() && k < images.size())
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
            const bool skip_special = req.value("skip_special_tokens", true);
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
                    Qwen3Parser parser(reasoning, thinking, tools);
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
                Qwen3Parser parser(reasoning, thinking, tools);
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
                                   {"usage", S.usage(r)}}));
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
    std::fprintf(stderr, "[g2e] serving %s on %s:%d (max_model_len %u%s)\n", opt.served_name.c_str(), opt.host.c_str(), opt.port,
                 opt.max_model_len, S.drafter ? ", DSpine speculative decoding for greedy requests" : "");
    if (!http.listen(opt.host, opt.port)) {
        std::fprintf(stderr, "[g2e] cannot listen on %s:%d\n", opt.host.c_str(), opt.port);
        return 1;
    }
    return 0;
}
