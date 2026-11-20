#include "g2e/dspine.hpp"
#include "g2e/numeric.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace g2e {

namespace {
constexpr float kF32Tiny = 1.1754943508222875e-38f;
void output(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const DType dt = d.dtype;
    const auto shape = d.shape;
    const auto out = g.input(name, dt, shape);
    g.op("identity", {src}, {out});
}
std::string Lw(uint32_t L, const std::string& part) { return "layers." + std::to_string(L) + "." + part; }

// Persistent inputs created once per recipe build.
struct Inputs {
    GraphBuilder& g;
    std::map<std::string, TensorId> ids;
    TensorId operator()(const std::string& name, DType dt, Shape shape) {
        if (auto it = ids.find(name); it != ids.end()) return it->second;
        return ids[name] = g.input(name, dt, std::move(shape));
    }
};

// NeoX RoPE over the full head: x [T, heads, D] (any dtype) with cos/sin [T, 1, D/2] F32 -> BF16
TensorId rope_full(Ops& o, TensorId x, TensorId cosv, TensorId sinv, uint64_t D, const std::string& n) {
    const auto f = o.cast(x, DType::F32, n + "_f");
    const auto s = o.split(f, 2, {D / 2, D / 2}, n + "_s");
    const auto o1 = o.sub(o.mul(s[0], cosv, n + "_1c"), o.mul(s[1], sinv, n + "_2s"), n + "_o1");
    const auto o2 = o.add(o.mul(s[1], cosv, n + "_2c"), o.mul(s[0], sinv, n + "_1s"), n + "_o2");
    return o.cast(o.concat({o1, o2}, 2, n + "_c"), DType::BF16, n + "_b");
}
}  // namespace

DSpine::DSpine(const Device& device, const Stream& stream, const WeightPack& draft, const WeightPack& target,
               uint32_t max_context)
    : dev_(device), stream_(stream), w_(draft), tw_(target), max_ctx_(max_context), staging_(device, 8u << 20) {
    const auto j = nlohmann::json::parse(draft.header_json());
    const auto& c = j.at("config");
    layers_ = c.at("num_hidden_layers");
    hidden_ = c.at("hidden_size");
    heads_ = c.at("num_attention_heads");
    kv_heads_ = c.at("num_key_value_heads");
    head_dim_ = c.at("head_dim");
    inter_ = c.at("intermediate_size");
    vocab_ = c.at("vocab_size");
    window_ = c.value("sliding_window", 0u);
    eps_ = c.at("rms_norm_eps");
    theta_ = c.at("rope_theta");
    const auto& dc = c.at("dflash_config");
    mask_token_ = dc.at("mask_token_id");
    target_layers_ = dc.at("target_layer_ids").get<std::vector<uint32_t>>();
    const auto& br = dc.at("selected_prefix_code_bridge");
    code_ = br.at("code_size");
    rank_ = br.at("rank");
    bridge_eps_ = br.at("eps");
    if (target_layers_.size() != kFeatures) throw std::runtime_error("dspine: expected 5 target layers");
    if (max_ctx_ % 128) throw std::invalid_argument("dspine: max_context must be a multiple of 128");
    // the drafter sees the last `window` positions only: a ring of window + kBlock rows (rounded to 128), so a
    // full-block append (append_context) only overwrites slots that are already outside every query's window
    ring_ = window_ ? std::min<uint32_t>((window_ + kBlock + 127) / 128 * 128, max_ctx_) : max_ctx_;
    slot_pos_.assign(ring_, -1);
    const uint64_t kv_bytes = uint64_t(kv_heads_) * ring_ * head_dim_ * 2;
    for (uint32_t L = 0; L < layers_; ++L) {
        k_cache_.emplace_back(dev_, kv_bytes);
        v_cache_.emplace_back(dev_, kv_bytes);
    }
    tg_ = DeviceBuffer(dev_, uint64_t(vocab_) * rank_ * 2);
    tsv_ = DeviceBuffer(dev_, uint64_t(vocab_) * 2 * rank_ * 2);
    {   // RoPE tables for every position (a launch binds the rows of its positions) and the block's causal mask
        const uint64_t R2 = head_dim_ / 2, rows = uint64_t(max_ctx_) + kBlock;
        if ((R2 * 4) % 128) throw std::runtime_error("dspine: head_dim/2 * 4 bytes must be a multiple of 128");
        cos_tab_ = DeviceBuffer(dev_, rows * R2 * 4);
        sin_tab_ = DeviceBuffer(dev_, rows * R2 * 4);
        const uint64_t per = staging_.size() / (R2 * 4) / 2;
        for (uint64_t r0 = 0; r0 < rows; r0 += per) {
            const uint64_t n = std::min(per, rows - r0);
            auto* c = staging_.as<float>();
            auto* s = c + per * R2;
            for (uint64_t r = 0; r < n; ++r)
                for (uint64_t i = 0; i < R2; ++i) {
                    const double a = double(r0 + r) * std::pow(theta_, -double(2 * i) / double(head_dim_));
                    c[r * R2 + i] = float(std::cos(a));
                    s[r * R2 + i] = float(std::sin(a));
                }
            stream_.upload(staging_, 0, cos_tab_, r0 * R2 * 4, n * R2 * 4);
            stream_.upload(staging_, per * R2 * 4, sin_tab_, r0 * R2 * 4, n * R2 * 4);
            stream_.synchronize();
        }
        causal_ = DeviceBuffer(dev_, uint64_t(kBlock) * kBlock * 4);
        auto* m = staging_.as<float>();
        for (uint32_t t = 0; t < kBlock; ++t)
            for (uint32_t u = 0; u < kBlock; ++u) m[t * kBlock + u] = u <= t ? 0.f : -1e30f;
        stream_.upload(staging_, 0, causal_, 0, uint64_t(kBlock) * kBlock * 4);
        stream_.synchronize();
    }
    ctl_ = DeviceBuffer(dev_, 4096 + (uint64_t(ring_) * 4 + 127) / 128 * 128);  // controls, then slot positions
    out_idx_ = DeviceBuffer(dev_, 262144);  // draft_batch_full: cand at 0, idx at 8192, tables at 16384
    {
        const auto c = Ops::topk2_constants(kBlock - 1, vocab_, kBlock, kTopkChunk);
        topk_aux_ = DeviceBuffer(dev_, c.size() * 4);
        std::memcpy(staging_.data(), c.data(), c.size() * 4);
        stream_.upload(staging_, 0, topk_aux_, 0, c.size() * 4);
        stream_.synchronize();
    }
    build_tables();
    quantize_lm_head();
    quantize_layers();
    reset();
}
DSpine::~DSpine() = default;

void DSpine::precompile(const std::function<void(size_t, size_t)>& progress) {
    std::vector<uint32_t> pieces;
    for (uint32_t n = 1; n <= 512; n *= 2) pieces.push_back(n);
    const size_t total = 2 + pieces.size() + (kMaxBatch - 1);
    size_t done = 0;
    block_recipe(false);
    if (progress) progress(++done, total);
    block_recipe(true);
    if (progress) progress(++done, total);
    for (uint32_t n : pieces) {
        append_recipe(n);
        if (progress) progress(++done, total);
    }
    for (uint32_t B = 2; B <= kMaxBatch; ++B) {  // batched drafting (draft_batch, draft_batch_full)
        block_batch_recipe(B, false);
        block_batch_recipe(B, true);
        if (progress) progress(++done, total);
    }
}

uint64_t DSpine::state_bytes() const {
    uint64_t b = 0;
    for (const auto& c : k_cache_) b += c.size();
    for (const auto& c : v_cache_) b += c.size();
    return b;
}

void DSpine::save_state(const HostBuffer& dst, uint64_t off) const {
    for (const auto* v : {&k_cache_, &v_cache_})
        for (const auto& c : *v) {
            stream_.download(c, 0, dst, off, c.size());
            off += c.size();
        }
}

void DSpine::load_state(const HostBuffer& src, uint64_t off, std::vector<int64_t> slot_pos) {
    if (slot_pos.size() != slot_pos_.size()) throw std::invalid_argument("dspine: slot positions of another ring");
    for (auto* v : {&k_cache_, &v_cache_})
        for (auto& c : *v) {
            stream_.upload(src, off, c, 0, c.size());
            off += c.size();
        }
    slot_pos_ = std::move(slot_pos);
    zeroed_ = true;
}

uint32_t DSpine::add_slot() {
    Seq q;
    for (const auto* v : {&k_cache_, &v_cache_})
        for (const auto& c : *v) {
            auto& dst = v == &k_cache_ ? q.k : q.v;
            dst.emplace_back(dev_, c.size());
            stream_.zero(dst.back(), 0, c.size());
        }
    q.slot_pos.assign(slot_pos_.size(), -1);
    stream_.synchronize();
    seqs_.push_back(std::move(q));
    return uint32_t(seqs_.size() - 1);
}

void DSpine::select_slot(uint32_t slot) {
    if (slot >= seqs_.size()) throw std::out_of_range("dspine: no such slot");
    if (slot == active_) return;
    auto swap = [&](Seq& q) {
        std::swap(k_cache_, q.k);
        std::swap(v_cache_, q.v);
        std::swap(slot_pos_, q.slot_pos);
    };
    swap(seqs_[active_]);
    swap(seqs_[slot]);
    active_ = slot;
    block_bind_[0].clear();  // they point at the ring caches of the slot that was selected
    block_bind_[1].clear();
}

void DSpine::reset() {
    // stale slots are masked: the caches only need zeroing once (no NaN patterns)
    std::fill(slot_pos_.begin(), slot_pos_.end(), -1);
    if (zeroed_) return;
    for (auto* v : {&k_cache_, &v_cache_})
        for (auto& b : *v) stream_.zero(b, 0, b.size());
    stream_.synchronize();
    zeroed_ = true;
}

uint32_t DSpine::bucket_for(uint32_t rows) const {
    uint32_t b = 128;
    while (b < rows) {
        const uint32_t mid = b + b / 2;
        b = (mid >= rows && mid % 128 == 0) ? mid : b * 2;
    }
    return std::min(b, max_ctx_);
}

// TG [V,R] = bf16(bf16(lm_head * norm_w) @ write_last);  TSV [V,2R] = bf16(sqrt(C) codebook) @ [pred_gate; prev]^T
void DSpine::build_tables() {
    const uint64_t H = hidden_, C = code_, R = rank_, V = vocab_;
    uint64_t rows = V;
    for (uint64_t parts = 1; parts <= 256; ++parts)
        if (V % parts == 0 && V / parts <= 8192) { rows = V / parts; break; }
    GraphBuilder g(dev_);
    Ops o(g);
    const auto head = g.input("head", DType::BF16, {rows, H});
    const auto cb = g.input("cb", DType::BF16, {rows, C});
    const auto nw = g.input("nw", DType::BF16, {H});
    const auto ww = g.input("ww", DType::BF16, {H, R});
    const auto pv = g.input("pv", DType::BF16, {2 * R, C});
    const auto hn = o.cast(o.mul(o.cast(head, DType::F32, "hf"), o.cast(nw, DType::F32, "nwf"), "hn"), DType::BF16, "hnb");
    const auto tg = o.temp("tg_v", DType::BF16, {rows, R});
    g.gemm(hn, ww, tg);
    output(g, "tg", tg);
    const auto codes = o.cast(o.mul(o.cast(cb, DType::F32, "cbf"), o.constant(float(std::sqrt(double(C))), {1}, "sqc"), "cbs"),
                              DType::BF16, "cbb");
    output(g, "tsv", o.linear_bf16(codes, pv, "tsv_v", DType::BF16));
    auto r = g.compile("dspine_tables_" + std::to_string(rows));
    const auto& lm = tw_.at("lm_head");
    const auto& cbw = w_.at("bridge.codebook");
    for (uint64_t start = 0; start < V; start += rows) {
        std::vector<Binding> b = {{"head", &tw_.arena(), lm.offset + start * H * 2},
                                  {"cb", &w_.arena(), cbw.offset + start * C * 2},
                                  w_.bind("nw", "norm.weight"),
                                  w_.bind("ww", "bridge.write." + std::to_string(layers_ - 1)),
                                  {"pv", &w_.arena(), w_.at("bridge.packed").offset + R * C * 2},
                                  {"tg", &tg_, start * R * 2},
                                  {"tsv", &tsv_, start * 2 * R * 2}};
        r->enqueue(stream_, b);
    }
    stream_.synchronize();
}

Recipe& DSpine::append_recipe(uint32_t n) {
    const std::string key = "dspine_append_n" + std::to_string(n) + (w8_on_ ? "_w8" : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = hidden_, KV = kv_heads_, D = head_dim_;
    GraphBuilder g(dev_);
    Ops o(g);
    Inputs in{g, {}};
    const auto cosv = in("cos", DType::F32, {n, 1, D / 2}), sinv = in("sin", DType::F32, {n, 1, D / 2});
    TensorId acc = kNoTensor;
    for (uint32_t k = 0; k < kFeatures; ++k) {
        const auto part = o.linear_bf16(in("feat" + std::to_string(k), DType::BF16, {n, H}),
                                        in("fc." + std::to_string(k), DType::BF16, {H, H}), "fc" + std::to_string(k), DType::F32);
        acc = acc == kNoTensor ? part : o.add(acc, part, "fcs" + std::to_string(k));
    }
    const auto ctx = o.rms_norm(o.cast(acc, DType::BF16, "fcb"), in("hidden_norm.weight", DType::BF16, {H}), eps_, false, "ctx");
    auto LIN = [&](TensorId x, const std::string& wn, uint64_t N, uint64_t K, const std::string& nm) {
        if (w8_on_) return o.fp8_linear(x, in(wn + ".f8", DType::F8, {N, K}), in(wn + ".s8", DType::F32, {N}), nm);
        return o.linear_bf16(x, in(wn, DType::BF16, {N, K}), nm);
    };
    for (uint32_t L = 0; L < layers_; ++L) {
        const std::string p = "L" + std::to_string(L) + ".";
        const auto kraw = o.reshape(LIN(ctx, Lw(L, "self_attn.k_proj.weight"), KV * D, H, p + "k"), {n, KV, D}, p + "k3");
        const auto k = rope_full(o, o.rms_norm(kraw, in(Lw(L, "self_attn.k_norm.weight"), DType::BF16, {D}), eps_, false, p + "kn"),
                                 cosv, sinv, D, p + "kr");
        const auto v = o.reshape(LIN(ctx, Lw(L, "self_attn.v_proj.weight"), KV * D, H, p + "v"), {n, KV, D}, p + "v3");
        auto heads = [&](TensorId t3, const std::string& nm) {
            auto parts = o.split(o.transpose(t3, {1, 0, 2}, nm + "_t"), 0, std::vector<uint64_t>(KV, 1), nm + "_h");
            for (uint64_t h = 0; h < KV; ++h) parts[h] = o.reshape(parts[h], {n, D}, nm + std::to_string(h));
            return parts;
        };
        const auto kh = heads(k, p + "kh"), vh = heads(v, p + "vh");
        for (uint64_t h = 0; h < KV; ++h) {
            output(g, p + "kout" + std::to_string(h), kh[h]);
            output(g, p + "vout" + std::to_string(h), vh[h]);
        }
    }
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

void DSpine::mark(uint32_t pos, uint32_t n) {
    for (uint32_t t = 0; t < n; ++t) slot_pos_[(pos + t) % ring_] = int64_t(pos + t);
}

void DSpine::append_context(const DeviceBuffer& feats, uint64_t slot_stride, uint32_t n, uint32_t pos,
                            uint32_t marked, uint32_t src_row) {
    if (!n) return;
    const uint64_t H = hidden_, D = head_dim_, KV = kv_heads_, R2 = D / 2;
    // a verification's rows (n <= kBlock, features rows 0..kBlock-1 exist): one kBlock-row launch; rows past n are
    // written into slots that hold positions outside every window (ring = window + kBlock) or nothing yet, and stay
    // unmarked until a later append writes them properly
    const bool whole = n <= kBlock && (pos % ring_) + kBlock <= ring_;
    for (uint32_t done = 0, i = 0; done < n; ++i) {
        const uint32_t slot = (pos + done) % ring_;
        const uint32_t m = std::min<uint32_t>({512u, n - done, ring_ - slot});
        uint32_t piece = 1;  // power-of-two pieces keep the number of recipes small
        while (piece * 2 <= m) piece *= 2;
        if (whole) piece = kBlock;
        (void)i;
        const uint64_t rope_at = uint64_t(pos + done) * R2 * 4;  // RoPE rows of these positions
        std::vector<Binding> b = {{"cos", &cos_tab_, rope_at}, {"sin", &sin_tab_, rope_at},
                                  w_.bind("hidden_norm.weight", "hidden_norm.weight")};
        for (uint32_t k = 0; k < kFeatures; ++k) {
            b.push_back({"feat" + std::to_string(k), &feats, k * slot_stride + uint64_t(src_row + done) * H * 2});
            b.push_back(w_.bind("fc." + std::to_string(k), "fc." + std::to_string(k)));
        }
        for (uint32_t L = 0; L < layers_; ++L) {
            const std::string p = "L" + std::to_string(L) + ".";
            bind_proj(b, Lw(L, "self_attn.k_proj.weight"));
            bind_proj(b, Lw(L, "self_attn.v_proj.weight"));
            b.push_back(w_.bind(Lw(L, "self_attn.k_norm.weight"), Lw(L, "self_attn.k_norm.weight")));
            for (uint64_t h = 0; h < KV; ++h) {
                const uint64_t at = (h * uint64_t(ring_) + slot) * D * 2;
                b.push_back({p + "kout" + std::to_string(h), &k_cache_[L], at});
                b.push_back({p + "vout" + std::to_string(h), &v_cache_[L], at});
            }
        }
        append_recipe(piece).enqueue(stream_, b);
        const uint32_t valid = std::min(piece, n - done);
        for (uint32_t t = 0; t < valid; ++t)
            if (done + t < marked) slot_pos_[slot + t] = int64_t(pos + done + t);
        done += valid;
    }
}

namespace {
const char* const kProjections[] = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                                    "self_attn.o_proj.weight", "mlp.gate_proj.weight",    "mlp.up_proj.weight",
                                    "mlp.down_proj.weight"};
}

void DSpine::quantize_layers() {
    // opt-in: saved only 0.3 ms per draft (activation quantisation chains) and lowered acceptance on one prompt
    w8_on_ = std::getenv("G2E_DRAFT_FP8") != nullptr;
    if (!w8_on_) return;
    struct Item { std::string name; uint64_t N, K; };
    std::vector<Item> items;
    uint64_t total = 0;
    for (uint32_t L = 0; L < layers_; ++L)
        for (const char* pr : kProjections) {
            const std::string nm = Lw(L, pr);
            const auto& sh = w_.at(nm).shape;
            items.push_back({nm, sh.at(0), sh.at(1)});
            w8_off_[nm + ".f8"] = total;
            total = (total + sh[0] * sh[1] + 127) / 128 * 128;
            w8_off_[nm + ".s8"] = total;
            total = (total + sh[0] * 4 + 127) / 128 * 128;
        }
    w8_ = DeviceBuffer(dev_, total);
    for (const auto& it : items) {
        const std::string key = "dspine_wq_" + std::to_string(it.N) + "x" + std::to_string(it.K);
        auto r = recipes_.find(key);
        if (r == recipes_.end()) {
            GraphBuilder g(dev_);
            Ops o(g);
            const auto [codes, scale] = o.quantize(g.input("w", DType::BF16, {it.N, it.K}), "q");
            output(g, "codes", codes);
            output(g, "scale", scale);
            r = recipes_.emplace(key, g.compile(key)).first;
        }
        const auto wb = w_.bind("w", it.name);
        const Binding b[] = {wb, {"codes", &w8_, w8_off_.at(it.name + ".f8")}, {"scale", &w8_, w8_off_.at(it.name + ".s8")}};
        r->second->enqueue(stream_, b);
    }
    stream_.synchronize();
}

void DSpine::bind_proj(std::vector<Binding>& b, const std::string& weight) const {
    if (w8_on_) {
        b.push_back({weight + ".f8", &w8_, w8_off_.at(weight + ".f8")});
        b.push_back({weight + ".s8", &w8_, w8_off_.at(weight + ".s8")});
    } else {
        b.push_back(w_.bind(weight, weight));
    }
}

void DSpine::quantize_lm_head() {
    lm8_on_ = std::getenv("G2E_DRAFT_LM_BF16") == nullptr;
    if (!lm8_on_) return;
    const uint64_t V = vocab_, H = hidden_, piece = 15520;  // 16 pieces of the 248320-row vocabulary (piece * 4 bytes stays 128-aligned)
    if (V % piece) throw std::runtime_error("dspine: vocabulary size not a multiple of the quantisation piece");
    lm8_ = DeviceBuffer(dev_, V * H);
    lm8s_ = DeviceBuffer(dev_, V * 4);
    const std::string key = "dspine_lmq_" + std::to_string(piece) + "x" + std::to_string(H);
    auto it = recipes_.find(key);
    if (it == recipes_.end()) {
        GraphBuilder g(dev_);
        Ops o(g);
        const auto [codes, scale] = o.quantize(g.input("w", DType::BF16, {piece, H}), "q");
        output(g, "codes", codes);
        output(g, "scale", scale);
        it = recipes_.emplace(key, g.compile(key)).first;
    }
    const auto lm = tw_.bind("w", "lm_head");
    for (uint64_t r = 0; r < V; r += piece) {
        const Binding b[] = {{"w", lm.buffer, lm.offset + r * H * 2}, {"codes", &lm8_, r * H}, {"scale", &lm8s_, r * 4}};
        it->second->enqueue(stream_, b);
    }
    stream_.synchronize();
}

Recipe& DSpine::block_recipe(bool sampled) {
    const std::string key = "dspine_block_ring" + std::to_string(ring_) + (sampled ? "_sampled" : "") + "_tk3" + (lm8_on_ ? "_l8" : "") + (w8_on_ ? "_w8" : "") + "_eg_mg_sm2";
    const uint64_t Sb = ring_;
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t T = kBlock, H = hidden_, NH = heads_, KV = kv_heads_, D = head_dim_, G = NH / KV, F = inter_,
                   C = code_, R = rank_, V = vocab_, K = kBlock, N = kBlock - 1;
    GraphBuilder g(dev_);
    Ops o(g);
    Inputs in{g, {}};
    auto W = [&](const std::string& name, Shape s) { return in(name, DType::BF16, std::move(s)); };
    const auto cosv = in("cos", DType::F32, {T, 1, D / 2}), sinv = in("sin", DType::F32, {T, 1, D / 2});
    // additive mask [T, Sb + T]: slot c (holding position sp[c], -1 if empty) is visible to row t iff
    // lo[t] <= sp[c] <= pos - 1 (lo[t] = max(0, pos + t - window + 1)); the block's rows see each other causally
    const auto sp = in("sp", DType::F32, {1, Sb});
    const auto mk_hi = o.binary("min", o.sub(in("posm1", DType::F32, {1, 1}), sp, "mk_h"), o.constant(0.f, {1, 1}, "mk_z"), "mk_hm");
    const auto mk_lo = o.binary("min", o.sub(sp, in("lo", DType::F32, {T, 1}), "mk_l"), o.constant(0.f, {1, 1}, "mk_z2"), "mk_lm");
    // the ring part (BF16, as its scores) and the block's causal part are kept apart: the two parts of the softmax are
    // merged by their maxima and sums (as block_batch_recipe) instead of concatenating the F32 scores
    const auto ring_mask = o.reshape(o.cast(o.mul(o.add(mk_lo, mk_hi, "mk_s"), o.constant(1e30f, {1, 1}, "mk_big"), "mk_c"),
                                            DType::BF16, "mk_b"), {1, 1, T, Sb}, "mk_4");
    const auto causal4 = o.reshape(in("causal", DType::F32, {T, T}), {1, 1, T, T}, "causal4");
    const auto notanchor = in("notanchor", DType::F32, {T, 1});
    const auto anchor = in("anchor", DType::I32, {1});
    const auto anchor16 = in("anchor16", DType::I32, {1, K});
    const auto codebook = W("bridge.codebook", {V, C});
    const auto packed_w = W("bridge.packed", {3 * R, C});
    const auto gate_b = o.cast(W("bridge.gate.bias", {R}), DType::F32, "gateb");
    const float sqrt_c = float(std::sqrt(double(C)));
    const auto anchor_codes = o.cast(o.mul(o.cast(o.gather_rows(codebook, anchor, "acb"), DType::F32, "acf"),
                                           o.constant(sqrt_c, {1}, "sqc"), "acs"), DType::BF16, "acodes");  // [1, C]
    // block embeddings (anchor, then mask tokens) gathered from the target's table
    const auto x0 = o.gather_rows(in("embed", DType::BF16, {V, H}), in("ids", DType::I32, {T}), "x0");
    TensorId residual = x0;
    TensorId x_in = o.rms_norm(x0, W(Lw(0, "input_layernorm.weight"), {H}), eps_, false, "in0");
    TensorId st_raw = kNoTensor, st_rms = kNoTensor, st_packed = kNoTensor;
    const auto scale = o.constant(float(1.0 / std::sqrt(double(D))), {1}, "scale");
    auto LIN = [&](TensorId x, const std::string& wn, uint64_t Nw, uint64_t Kw, const std::string& nm) {
        if (w8_on_) return o.fp8_linear(x, in(wn + ".f8", DType::F8, {Nw, Kw}), in(wn + ".s8", DType::F32, {Nw}), nm);
        return o.linear_bf16(x, W(wn, {Nw, Kw}), nm);
    };
    for (uint32_t L = 0; L < layers_; ++L) {
        const std::string p = "L" + std::to_string(L) + ".";
        // ---- attention: context cache [Sb] + the block (causal), per kv head
        const auto q = rope_full(o, o.rms_norm(o.reshape(LIN(x_in, Lw(L, "self_attn.q_proj.weight"), NH * D, H, p + "q"),
                                                         {T, NH, D}, p + "q3"),
                                               W(Lw(L, "self_attn.q_norm.weight"), {D}), eps_, false, p + "qn"),
                                 cosv, sinv, D, p + "qr");
        const auto k = rope_full(o, o.rms_norm(o.reshape(LIN(x_in, Lw(L, "self_attn.k_proj.weight"), KV * D, H, p + "k"),
                                                         {T, KV, D}, p + "k3"),
                                               W(Lw(L, "self_attn.k_norm.weight"), {D}), eps_, false, p + "kn"),
                                 cosv, sinv, D, p + "kr");
        const auto v = o.reshape(LIN(x_in, Lw(L, "self_attn.v_proj.weight"), KV * D, H, p + "v"), {T, KV, D}, p + "v3");
        // all kv heads at once against the whole ring (stale and future slots are masked)
        const auto q4 = o.reshape(o.transpose(q, {1, 0, 2}, p + "qt"), {KV, G * T, D}, p + "q4");  // heads h*G.. -> kv h
        const auto kb = o.transpose(k, {1, 0, 2}, p + "kt");                                      // [KV, T, D]
        const auto vb = o.transpose(v, {1, 0, 2}, p + "vt");
        const auto kc = in(p + "kc", DType::BF16, {KV, Sb, D});
        const auto vc = in(p + "vc", DType::BF16, {KV, Sb, D});
        const auto qsc = o.cast(o.mul(o.cast(q4, DType::F32, p + "qf"), scale, p + "qm"), DType::BF16, p + "qsb");
        const auto s1 = o.add(o.reshape(o.batch_gemm(qsc, kc, p + "sc", false, true, DType::BF16), {KV, G, T, Sb}, p + "s14"),
                              ring_mask, p + "s1m");
        const auto s2 = o.add(o.reshape(o.batch_gemm(qsc, kb, p + "sn", false, true, DType::F32), {KV, G, T, T}, p + "s24"),
                              causal4, p + "s2m");
        const auto mx = o.binary("max", o.cast(o.reduce("max", s1, 3, p + "m1"), DType::F32, p + "m1f"),
                                 o.reduce("max", s2, 3, p + "m2"), p + "mm");                         // [KV, G, T, 1]
        const auto p1 = o.unary("exp", o.sub(s1, o.cast(mx, DType::BF16, p + "mb"), p + "d1"), p + "p1");  // BF16
        const auto p2 = o.unary("exp", o.sub(s2, mx, p + "d2"), p + "p2");
        const auto lsum = o.add(o.cast(o.reduce("sum", p1, 3, p + "l1"), DType::F32, p + "l1f"), o.reduce("sum", p2, 3, p + "l2"),
                                p + "l");
        const auto o1 = o.batch_gemm(o.reshape(p1, {KV, G * T, Sb}, p + "p13"), vc, p + "oc", false, false, DType::F32);
        const auto o2 = o.batch_gemm(o.cast(o.reshape(p2, {KV, G * T, T}, p + "p23"), DType::BF16, p + "p2b"), vb, p + "on", false,
                                     false, DType::F32);
        const auto att3 = o.div(o.add(o1, o2, p + "ao"), o.reshape(lsum, {KV, G * T, 1}, p + "l3"), p + "an");  // [KV, G*T, D]
        const auto att = o.reshape(o.transpose(o.reshape(o.cast(att3, DType::BF16, p + "attb"), {NH, T, D}, p + "a3"), {1, 0, 2},
                                               p + "at"), {T, NH * D}, p + "a2");
        const auto attn = LIN(att, Lw(L, "self_attn.o_proj.weight"), H, NH * D, p + "o");
        const auto hsum = o.add(residual, attn, p + "h");
        // ---- MLP
        const auto x2 = o.rms_norm(hsum, W(Lw(L, "post_attention_layernorm.weight"), {H}), eps_, false, p + "x2");
        const auto gt = LIN(x2, Lw(L, "mlp.gate_proj.weight"), F, H, p + "gt");
        const auto up = LIN(x2, Lw(L, "mlp.up_proj.weight"), F, H, p + "up");
        const auto gs = o.temp(p + "gs", DType::BF16, {T, F});
        g.op("silu_fwd_bf16", {gt}, {gs});
        const auto mlp = LIN(o.mul(gs, up, p + "act"), Lw(L, "mlp.down_proj.weight"), H, F, p + "dn");
        const auto out = o.add(hsum, mlp, p + "out");  // BF16
        // ---- code bridge (dspine_mlx bridge.py, rounding point by rounding point)
        const auto f = o.cast(out, DType::F32, p + "bf");
        const auto ms = o.mul(o.reduce("sum", o.mul(f, f, p + "f2"), 1, p + "fs"), o.constant(1.0f / float(H), {1}, p + "invh"), p + "ms");
        const auto rms = o.unary("sqrt", o.binary("max", ms, o.constant(kF32Tiny, {1}, p + "tiny"), p + "msc"), p + "rms");  // [T,1]
        const auto beps = o.constant(bridge_eps_, {1}, p + "beps");
        const auto normalized = o.cast(o.div(f, o.add(rms, beps, p + "re"), p + "nf"), DType::BF16, p + "nb");
        const auto projected = o.cast(o.linear_bf16(normalized, W("bridge.read." + std::to_string(L), {C, H}), p + "pj"),
                                      DType::F32, p + "pjf");
        const auto pn = o.unary("sqrt", o.reduce("sum", o.mul(projected, projected, p + "pj2"), 1, p + "pjs"), p + "pn");
        const auto codes_all = o.cast(o.div(o.mul(projected, o.constant(sqrt_c, {1}, p + "sqc"), p + "pjc"),
                                            o.add(pn, beps, p + "pne"), p + "cd"), DType::BF16, p + "cdb");
        const auto codes = o.concat({anchor_codes, o.split(codes_all, 0, {1, T - 1}, p + "cs")[1]}, 0, p + "codes");
        const auto packed = o.linear_bf16(codes, packed_w, p + "pk", DType::BF16);  // [T, 3R]
        if (L + 1 == layers_) {
            st_raw = out;
            st_rms = rms;
            st_packed = packed;
        }
        const auto qsv = o.split(o.cast(packed, DType::F32, p + "pkf"), 1, {R, R, R}, p + "qsv");
        const auto q_next = o.split(qsv[0], 0, {1, T - 1}, p + "qn1")[1];
        const auto s_prev = o.split(qsv[1], 0, {T - 1, 1}, p + "sp1")[0];
        const auto v_prev = o.split(qsv[2], 0, {T - 1, 1}, p + "vp1")[0];
        const auto gate = o.unary("sigmoid", o.add(o.add(q_next, s_prev, p + "gqs"), gate_b, p + "gqb"), p + "gate");
        const auto message = o.cast(o.concat({o.constant(0.0f, {1, R}, p + "z"), o.mul(gate, v_prev, p + "gv")}, 0, p + "msg"),
                                    DType::BF16, p + "msgb");
        const auto delta = o.cast(o.linear_bf16(message, W("bridge.write." + std::to_string(L), {H, R}), p + "dl"), DType::F32, p + "dlf");
        const auto out2 = o.cast(o.add(f, o.mul(o.mul(delta, rms, p + "dr"), notanchor, p + "dm"), p + "o2f"), DType::BF16, p + "o2");
        const auto next_w = L + 1 < layers_ ? W(Lw(L + 1, "input_layernorm.weight"), {H}) : W("norm.weight", {H});
        x_in = o.rms_norm(out2, next_w, eps_, false, p + "next");
        residual = out2;
    }
    // ---- refinement tail (greedy): rows 1..15
    const auto rows = o.split(x_in, 0, {1, N}, "rows")[1];                                   // [N, H] final-normed
    const auto lm = in("lm_head", DType::BF16, {V, H});
    TensorId logits;
    if (lm8_on_) {
        const auto [rc, rs] = o.quantize(rows, "rq");
        logits = o.fp8_gemm(rc, rs, in("lm8", DType::F8, {V, H}), in("lm8s", DType::F32, {V}), "logits");
    } else {
        logits = o.linear_bf16(rows, lm, "logits", DType::F32);
    }
    const auto cand = o.topk2(logits, uint32_t(K), kTopkChunk,
                              in("tk_consts", DType::I32, {V + 2 * N}), "tk").second;  // [N, K]
    const auto cand_flat = o.reshape(cand, {N * K}, "cflat");
    const auto head_rows = o.reshape(o.cast(o.gather_rows(lm, cand_flat, "hr"), DType::F32, "hrf"), {N, K, H}, "hr3");
    const auto raw_n = o.cast(o.split(st_raw, 0, {1, N}, "rawn")[1], DType::F32, "rawf");
    const auto rms_n = o.reshape(o.split(st_rms, 0, {1, N}, "rmsn")[1], {N, 1, 1}, "rms3");
    const auto rnw = o.reshape(o.mul(raw_n, o.cast(W("norm.weight", {H}), DType::F32, "nwf"), "rnw"), {N, 1, H}, "rnw3");
    const auto base = o.batch_gemm(rnw, head_rows, "base", false, true, DType::F32);           // [N, 1, K]
    const auto sid = o.reshape(o.concat({anchor16, o.split(cand, 0, {N - 1, 1}, "cprev")[0]}, 0, "sid"), {N * K}, "sidf");
    const auto tsv = o.reshape(o.cast(o.gather_rows(in("tsv", DType::BF16, {V, 2 * R}), sid, "tsvg"), DType::F32, "tsvf"),
                               {N, K, 2 * R}, "tsv3");
    const auto tv = o.split(tsv, 2, {R, R}, "tv");
    const auto qn = o.reshape(o.split(o.split(o.cast(st_packed, DType::F32, "stpf"), 1, {R, 2 * R}, "stq")[0], 0, {1, N}, "stqn")[1],
                              {N, 1, R}, "qn3");
    const auto m = o.mul(o.unary("sigmoid", o.add(o.add(qn, tv[0], "mq"), gate_b, "mqb"), "msig"), tv[1], "m");  // [N, K, R]
    const auto tgc = o.reshape(o.cast(o.gather_rows(in("tg", DType::BF16, {V, R}), cand_flat, "tgg"), DType::F32, "tgf"), {N, K, R}, "tg3");
    output(g, "cand", cand);
    if (sampled) {
        // refined[n,k] = raw[n] + rms[n] * (m[n,k] @ write^T), final-normed, against the candidates' head rows
        const auto wf = o.cast(W("bridge.write." + std::to_string(layers_ - 1), {H, R}), DType::F32, "wlf");
        const auto upd = o.temp("upd", DType::F32, {N * K, H});
        g.gemm(o.reshape(m, {N * K, R}, "mflat"), wf, upd, false, true);
        const auto refined = o.add(o.reshape(raw_n, {N, 1, H}, "raw3"), o.mul(rms_n, o.reshape(upd, {N, K, H}, "upd3"), "rupd"), "refined");
        const auto ms = o.mul(o.reduce("sum", o.mul(refined, refined, "rf2"), 2, "rfs"), o.constant(1.0f / float(H), {1}, "invh"), "rms_");
        const auto norm = o.unary("sqrt", o.add(ms, o.constant(eps_, {1}, "neps"), "msn"), "nrm");
        const auto normalized = o.mul(o.div(refined, norm, "rn"), o.cast(W("norm.weight", {H}), DType::F32, "nwf2"), "rnw2");
        output(g, "table", o.batch_gemm(normalized, head_rows, "table_v", false, true, DType::F32));  // [N, K, K]
    } else {
        const auto s = o.batch_gemm(m, tgc, "s", false, true, DType::F32);                     // [N, K(sender), K(cand)]
        const auto score = o.add(base, o.mul(rms_n, s, "rs"), "score");
        output(g, "idx", o.argmax(score, "best"));                                             // [N, K, 1]
    }
    (void)T;
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

// draft_batch controls in ctlb_ (byte offsets): ids [B*T] i32 at 0, rope positions [B*T] i32 at 512, anchors [B]
// i32 at 1024, anchor16 [B, K] i32 at 1152, pos - 1 [B] f32 at 1664, lo [B*T] f32 at 1792, notanchor [B*T] f32 at
// 2304, slot positions [B, ring] f32 at 4096 (every slot's row 128-byte aligned)
namespace {
constexpr uint64_t kBIds = 0, kBRope = 512, kBAnchor = 1024, kBAnchor16 = 1152, kBPosm1 = 1664, kBLo = 1792,
                   kBNotAnchor = 2304, kBSp = 4096;
}

Recipe& DSpine::block_batch_recipe(uint32_t B, bool sampled) {
    const std::string key = "dspine_blockb_B" + std::to_string(B) + "_ring" + std::to_string(ring_) + (lm8_on_ ? "_l8" : "") +
                            (w8_on_ ? "_w8" : "") + (sampled ? "_sampled" : "") + "_sm2";
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t T = kBlock, H = hidden_, NH = heads_, KV = kv_heads_, D = head_dim_, G = NH / KV, F = inter_,
                   C = code_, R = rank_, V = vocab_, K = kBlock, N = kBlock - 1, Sb = ring_, BT = B * T, BN = B * N;
    const uint64_t spr = (Sb * 4 + 127) / 128 * 128 / 4;  // slot-position row stride (f32)
    GraphBuilder g(dev_);
    Ops o(g);
    Inputs in{g, {}};
    auto W = [&](const std::string& name, Shape s) { return in(name, DType::BF16, std::move(s)); };
    auto rows_of = [&](TensorId x, uint64_t per, const std::string& nm) {  // split rows per slot
        return o.split(x, 0, std::vector<uint64_t>(B, per), nm);
    };
    auto cat = [&](const std::vector<TensorId>& v, const std::string& nm) { return v.size() == 1 ? v[0] : o.concat(v, 0, nm); };
    // RoPE rows gathered by position (per-slot bindings of the tables could overlap)
    const uint64_t tab = uint64_t(max_ctx_) + kBlock;
    const auto rpos = in("rope_pos", DType::I32, {BT});
    const auto cosv = o.reshape(o.gather_rows(in("cos_tab", DType::F32, {tab, D / 2}), rpos, "cosg"), {BT, 1, D / 2}, "cos3");
    const auto sinv = o.reshape(o.gather_rows(in("sin_tab", DType::F32, {tab, D / 2}), rpos, "sing"), {BT, 1, D / 2}, "sin3");
    // per slot: the additive mask [T, Sb + T] as block_recipe (its slot positions, pos - 1, lowest visible positions)
    const auto sp_in = in("sp", DType::F32, {B, spr});
    const auto sp_all = rows_of(spr == Sb ? sp_in : o.split(sp_in, 1, {Sb, spr - Sb}, "spcut")[0], 1, "sps");
    const auto posm1_all = rows_of(in("posm1", DType::F32, {B, 1}), 1, "pms");
    const auto lo_all = rows_of(in("lo", DType::F32, {BT, 1}), T, "los");
    const auto causal = in("causal", DType::F32, {T, T});
    // the ring part of each slot's mask (BF16, as its scores) and the block's causal part, kept apart: the two
    // parts of the softmax are merged by their maxima and sums instead of concatenating the scores (a copy of
    // [KV, G*T, ring] per slot and layer)
    std::vector<TensorId> ring_masks;
    for (uint32_t b = 0; b < B; ++b) {
        const auto bs = std::to_string(b);
        const auto mk_hi = o.binary("min", o.sub(posm1_all[b], sp_all[b], "mk_h" + bs), o.constant(0.f, {1, 1}, "mk_z" + bs), "mk_hm" + bs);
        const auto mk_lo = o.binary("min", o.sub(sp_all[b], lo_all[b], "mk_l" + bs), o.constant(0.f, {1, 1}, "mk_z2" + bs), "mk_lm" + bs);
        ring_masks.push_back(o.reshape(o.cast(o.mul(o.add(mk_lo, mk_hi, "mk_s" + bs), o.constant(1e30f, {1, 1}, "mk_big" + bs), "mk_c" + bs),
                                              DType::BF16, "mk_b" + bs), {1, 1, T, Sb}, "mk_4" + bs));
    }
    const auto causal4 = o.reshape(causal, {1, 1, T, T}, "causal4");
    const auto notanchor = in("notanchor", DType::F32, {BT, 1});
    const auto anchors = in("anchor", DType::I32, {B});
    const auto anchor16_all = rows_of(in("anchor16", DType::I32, {B, K}), 1, "a16s");
    const auto codebook = W("bridge.codebook", {V, C});
    const auto packed_w = W("bridge.packed", {3 * R, C});
    const auto gate_b = o.cast(W("bridge.gate.bias", {R}), DType::F32, "gateb");
    const float sqrt_c = float(std::sqrt(double(C)));
    const auto anchor_codes = rows_of(o.cast(o.mul(o.cast(o.gather_rows(codebook, anchors, "acb"), DType::F32, "acf"),
                                                   o.constant(sqrt_c, {1}, "sqc"), "acs"), DType::BF16, "acodes"), 1, "acsl");  // B x [1, C]
    const auto x0 = o.gather_rows(in("embed", DType::BF16, {V, H}), in("ids", DType::I32, {BT}), "x0");
    TensorId residual = x0;
    TensorId x_in = o.rms_norm(x0, W(Lw(0, "input_layernorm.weight"), {H}), eps_, false, "in0");
    TensorId st_raw = kNoTensor, st_rms = kNoTensor, st_packed = kNoTensor;
    const auto scale = o.constant(float(1.0 / std::sqrt(double(D))), {1}, "scale");
    auto LIN = [&](TensorId x, const std::string& wn, uint64_t Nw, uint64_t Kw, const std::string& nm) {
        if (w8_on_) return o.fp8_linear(x, in(wn + ".f8", DType::F8, {Nw, Kw}), in(wn + ".s8", DType::F32, {Nw}), nm);
        return o.linear_bf16(x, W(wn, {Nw, Kw}), nm);
    };
    for (uint32_t L = 0; L < layers_; ++L) {
        const std::string p = "L" + std::to_string(L) + ".";
        const auto q = rope_full(o, o.rms_norm(o.reshape(LIN(x_in, Lw(L, "self_attn.q_proj.weight"), NH * D, H, p + "q"),
                                                         {BT, NH, D}, p + "q3"),
                                               W(Lw(L, "self_attn.q_norm.weight"), {D}), eps_, false, p + "qn"),
                                 cosv, sinv, D, p + "qr");
        const auto k = rope_full(o, o.rms_norm(o.reshape(LIN(x_in, Lw(L, "self_attn.k_proj.weight"), KV * D, H, p + "k"),
                                                         {BT, KV, D}, p + "k3"),
                                               W(Lw(L, "self_attn.k_norm.weight"), {D}), eps_, false, p + "kn"),
                                 cosv, sinv, D, p + "kr");
        const auto v = o.reshape(LIN(x_in, Lw(L, "self_attn.v_proj.weight"), KV * D, H, p + "v"), {BT, KV, D}, p + "v3");
        const auto qs = rows_of(q, T, p + "qsl"), ks = rows_of(k, T, p + "ksl"), vs = rows_of(v, T, p + "vsl");
        std::vector<TensorId> atts;
        for (uint32_t b = 0; b < B; ++b) {  // as block_recipe, against slot b's ring
            const std::string pb = p + "b" + std::to_string(b) + ".";
            const auto q4 = o.reshape(o.transpose(qs[b], {1, 0, 2}, pb + "qt"), {KV, G * T, D}, pb + "q4");
            const auto kb = o.transpose(ks[b], {1, 0, 2}, pb + "kt");
            const auto vb = o.transpose(vs[b], {1, 0, 2}, pb + "vt");
            const auto kc = in(p + "kc" + std::to_string(b), DType::BF16, {KV, Sb, D});
            const auto vc = in(p + "vc" + std::to_string(b), DType::BF16, {KV, Sb, D});
            const auto qsc = o.cast(o.mul(o.cast(q4, DType::F32, pb + "qf"), scale, pb + "qm"), DType::BF16, pb + "qsb");
            // ring part in BF16 (scores, mask, exp), the block's 16 columns in F32
            const auto s1 = o.add(o.reshape(o.batch_gemm(qsc, kc, pb + "sc", false, true, DType::BF16), {KV, G, T, Sb}, pb + "s14"),
                                  ring_masks[b], pb + "s1m");
            const auto s2 = o.add(o.reshape(o.batch_gemm(qsc, kb, pb + "sn", false, true, DType::F32), {KV, G, T, T}, pb + "s24"),
                                  causal4, pb + "s2m");
            const auto m = o.binary("max", o.cast(o.reduce("max", s1, 3, pb + "m1"), DType::F32, pb + "m1f"),
                                    o.reduce("max", s2, 3, pb + "m2"), pb + "mm");                       // [KV, G, T, 1]
            const auto p1 = o.unary("exp", o.sub(s1, o.cast(m, DType::BF16, pb + "mb"), pb + "d1"), pb + "p1");  // BF16
            const auto p2 = o.unary("exp", o.sub(s2, m, pb + "d2"), pb + "p2");
            const auto l = o.add(o.cast(o.reduce("sum", p1, 3, pb + "l1"), DType::F32, pb + "l1f"), o.reduce("sum", p2, 3, pb + "l2"),
                                 pb + "l");
            const auto o1 = o.batch_gemm(o.reshape(p1, {KV, G * T, Sb}, pb + "p13"), vc, pb + "oc", false, false, DType::F32);
            const auto o2 = o.batch_gemm(o.cast(o.reshape(p2, {KV, G * T, T}, pb + "p23"), DType::BF16, pb + "p2b"), vb, pb + "on", false,
                                         false, DType::F32);
            const auto att3 = o.div(o.add(o1, o2, pb + "ao"), o.reshape(l, {KV, G * T, 1}, pb + "l3"), pb + "an");
            atts.push_back(o.reshape(o.transpose(o.reshape(o.cast(att3, DType::BF16, pb + "attb"), {NH, T, D}, pb + "a3"), {1, 0, 2},
                                                 pb + "at"), {T, NH * D}, pb + "a2"));
        }
        const auto attn = LIN(cat(atts, p + "atc"), Lw(L, "self_attn.o_proj.weight"), H, NH * D, p + "o");
        const auto hsum = o.add(residual, attn, p + "h");
        const auto x2 = o.rms_norm(hsum, W(Lw(L, "post_attention_layernorm.weight"), {H}), eps_, false, p + "x2");
        const auto gt = LIN(x2, Lw(L, "mlp.gate_proj.weight"), F, H, p + "gt");
        const auto up = LIN(x2, Lw(L, "mlp.up_proj.weight"), F, H, p + "up");
        const auto gs = o.temp(p + "gs", DType::BF16, {BT, F});
        g.op("silu_fwd_bf16", {gt}, {gs});
        const auto mlp = LIN(o.mul(gs, up, p + "act"), Lw(L, "mlp.down_proj.weight"), H, F, p + "dn");
        const auto out = o.add(hsum, mlp, p + "out");
        // ---- code bridge (per row; the anchor / previous-row structure per slot)
        const auto f = o.cast(out, DType::F32, p + "bf");
        const auto ms = o.mul(o.reduce("sum", o.mul(f, f, p + "f2"), 1, p + "fs"), o.constant(1.0f / float(H), {1}, p + "invh"), p + "ms");
        const auto rms = o.unary("sqrt", o.binary("max", ms, o.constant(kF32Tiny, {1}, p + "tiny"), p + "msc"), p + "rms");
        const auto beps = o.constant(bridge_eps_, {1}, p + "beps");
        const auto normalized = o.cast(o.div(f, o.add(rms, beps, p + "re"), p + "nf"), DType::BF16, p + "nb");
        const auto projected = o.cast(o.linear_bf16(normalized, W("bridge.read." + std::to_string(L), {C, H}), p + "pj"),
                                      DType::F32, p + "pjf");
        const auto pn = o.unary("sqrt", o.reduce("sum", o.mul(projected, projected, p + "pj2"), 1, p + "pjs"), p + "pn");
        const auto codes_all = rows_of(o.cast(o.div(o.mul(projected, o.constant(sqrt_c, {1}, p + "sqc"), p + "pjc"),
                                                    o.add(pn, beps, p + "pne"), p + "cd"), DType::BF16, p + "cdb"), T, p + "cdsl");
        std::vector<TensorId> codes;
        for (uint32_t b = 0; b < B; ++b)
            codes.push_back(o.concat({anchor_codes[b], o.split(codes_all[b], 0, {1, T - 1}, p + "cs" + std::to_string(b))[1]}, 0,
                                     p + "codes" + std::to_string(b)));
        const auto packed = o.linear_bf16(cat(codes, p + "codesc"), packed_w, p + "pk", DType::BF16);  // [BT, 3R]
        if (L + 1 == layers_) {
            st_raw = out;
            st_rms = rms;
            st_packed = packed;
        }
        const auto qsv = o.split(o.cast(packed, DType::F32, p + "pkf"), 1, {R, R, R}, p + "qsv");
        const auto q_all = rows_of(qsv[0], T, p + "qall"), s_all = rows_of(qsv[1], T, p + "sall"), v_all = rows_of(qsv[2], T, p + "vall");
        std::vector<TensorId> msgs;
        for (uint32_t b = 0; b < B; ++b) {
            const std::string pb = p + "m" + std::to_string(b);
            const auto q_next = o.split(q_all[b], 0, {1, T - 1}, pb + "qn1")[1];
            const auto s_prev = o.split(s_all[b], 0, {T - 1, 1}, pb + "sp1")[0];
            const auto v_prev = o.split(v_all[b], 0, {T - 1, 1}, pb + "vp1")[0];
            const auto gate = o.unary("sigmoid", o.add(o.add(q_next, s_prev, pb + "gqs"), gate_b, pb + "gqb"), pb + "gate");
            msgs.push_back(o.concat({o.constant(0.0f, {1, R}, pb + "z"), o.mul(gate, v_prev, pb + "gv")}, 0, pb + "msg"));
        }
        const auto message = o.cast(cat(msgs, p + "msgc"), DType::BF16, p + "msgb");
        const auto delta = o.cast(o.linear_bf16(message, W("bridge.write." + std::to_string(L), {H, R}), p + "dl"), DType::F32, p + "dlf");
        const auto out2 = o.cast(o.add(f, o.mul(o.mul(delta, rms, p + "dr"), notanchor, p + "dm"), p + "o2f"), DType::BF16, p + "o2");
        const auto next_w = L + 1 < layers_ ? W(Lw(L + 1, "input_layernorm.weight"), {H}) : W("norm.weight", {H});
        x_in = o.rms_norm(out2, next_w, eps_, false, p + "next");
        residual = out2;
    }
    // ---- refinement tail (greedy): rows 1..15 of every slot
    auto tail_rows = [&](TensorId x, const std::string& nm) {
        std::vector<TensorId> v;
        const auto parts = rows_of(x, T, nm + "sl");
        for (uint32_t b = 0; b < B; ++b) v.push_back(o.split(parts[b], 0, {1, N}, nm + std::to_string(b))[1]);
        return cat(v, nm + "c");
    };
    const auto rows = tail_rows(x_in, "rows");  // [BN, H]
    const auto lm = in("lm_head", DType::BF16, {V, H});
    TensorId logits;
    if (lm8_on_) {
        const auto [rc, rs] = o.quantize(rows, "rq");
        logits = o.fp8_gemm(rc, rs, in("lm8", DType::F8, {V, H}), in("lm8s", DType::F32, {V}), "logits");
    } else {
        logits = o.linear_bf16(rows, lm, "logits", DType::F32);
    }
    const auto cand = o.topk2(logits, uint32_t(K), kTopkChunk, in("tk_consts", DType::I32, {V + 2 * BN}), "tk").second;  // [BN, K]
    const auto cand_flat = o.reshape(cand, {BN * K}, "cflat");
    const auto head_rows = o.reshape(o.cast(o.gather_rows(lm, cand_flat, "hr"), DType::F32, "hrf"), {BN, K, H}, "hr3");
    const auto raw_n = o.cast(tail_rows(st_raw, "rawn"), DType::F32, "rawf");
    const auto rms_n = o.reshape(tail_rows(st_rms, "rmsn"), {BN, 1, 1}, "rms3");
    const auto rnw = o.reshape(o.mul(raw_n, o.cast(W("norm.weight", {H}), DType::F32, "nwf"), "rnw"), {BN, 1, H}, "rnw3");
    const auto base = o.batch_gemm(rnw, head_rows, "base", false, true, DType::F32);  // [BN, 1, K]
    std::vector<TensorId> sids;
    {
        const auto cparts = rows_of(cand, N, "cps");
        for (uint32_t b = 0; b < B; ++b)
            sids.push_back(o.concat({anchor16_all[b], o.split(cparts[b], 0, {N - 1, 1}, "cprev" + std::to_string(b))[0]}, 0,
                                    "sid" + std::to_string(b)));
    }
    const auto sid = o.reshape(cat(sids, "sidc"), {BN * K}, "sidf");
    const auto tsv = o.reshape(o.cast(o.gather_rows(in("tsv", DType::BF16, {V, 2 * R}), sid, "tsvg"), DType::F32, "tsvf"),
                               {BN, K, 2 * R}, "tsv3");
    const auto tv = o.split(tsv, 2, {R, R}, "tv");
    const auto qn = o.reshape(tail_rows(o.split(o.cast(st_packed, DType::F32, "stpf"), 1, {R, 2 * R}, "stq")[0], "stqn"),
                              {BN, 1, R}, "qn3");
    const auto m = o.mul(o.unary("sigmoid", o.add(o.add(qn, tv[0], "mq"), gate_b, "mqb"), "msig"), tv[1], "m");  // [BN, K, R]
    const auto tgc = o.reshape(o.cast(o.gather_rows(in("tg", DType::BF16, {V, R}), cand_flat, "tgg"), DType::F32, "tgf"), {BN, K, R}, "tg3");
    output(g, "cand", cand);
    const auto s = o.batch_gemm(m, tgc, "s", false, true, DType::F32);
    const auto score = o.add(base, o.mul(rms_n, s, "rs"), "score");
    output(g, "idx", o.argmax(score, "best"));  // [BN, K, 1]
    if (sampled) {  // as block_recipe's sampled tail, for every slot's rows
        const auto wf = o.cast(W("bridge.write." + std::to_string(layers_ - 1), {H, R}), DType::F32, "wlf");
        const auto upd = o.temp("upd", DType::F32, {BN * K, H});
        g.gemm(o.reshape(m, {BN * K, R}, "mflat"), wf, upd, false, true);
        const auto refined = o.add(o.reshape(raw_n, {BN, 1, H}, "raw3"), o.mul(rms_n, o.reshape(upd, {BN, K, H}, "upd3"), "rupd"),
                                   "refined");
        const auto ms = o.mul(o.reduce("sum", o.mul(refined, refined, "rf2"), 2, "rfs"), o.constant(1.0f / float(H), {1}, "invh"),
                              "rms_");
        const auto norm = o.unary("sqrt", o.add(ms, o.constant(eps_, {1}, "neps"), "msn"), "nrm");
        const auto normalized = o.mul(o.div(refined, norm, "rn"), o.cast(W("norm.weight", {H}), DType::F32, "nwf2"), "rnw2");
        output(g, "table", o.batch_gemm(normalized, head_rows, "table_v", false, true, DType::F32));  // [BN, K, K]
    }
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

std::vector<std::vector<int32_t>> DSpine::draft_batch(std::span<const uint32_t> slots, std::span<const int32_t> anchors,
                                                      std::span<const uint32_t> pos) {
    auto full = draft_batch_full(slots, anchors, pos, false);
    std::vector<std::vector<int32_t>> out;
    for (auto& f : full) out.push_back(std::move(f.drafts));
    return out;
}

std::vector<DSpine::BatchDraft> DSpine::draft_batch_full(std::span<const uint32_t> slots, std::span<const int32_t> anchors,
                                                         std::span<const uint32_t> pos, bool tables) {
    const uint32_t B = uint32_t(slots.size());
    const uint64_t T = kBlock, N = kBlock - 1, K = kBlock, Sb = ring_, BN = uint64_t(B) * N;
    if (B < 2 || B > kMaxBatch || anchors.size() != B || pos.size() != B) throw std::invalid_argument("dspine: draft_batch takes 2..8 slots");
    for (uint32_t b = 0; b < B; ++b) {
        if (slots[b] >= seqs_.size() || pos[b] + T > max_ctx_) throw std::invalid_argument("dspine: draft_batch slot / position");
        for (uint32_t c = 0; c < b; ++c)
            if (slots[c] == slots[b]) throw std::invalid_argument("dspine: draft_batch slots must differ");
    }
    // park the selected slot: every slot's ring is in seqs_ while this runs
    auto swap = [&](Seq& q) {
        std::swap(k_cache_, q.k);
        std::swap(v_cache_, q.v);
        std::swap(slot_pos_, q.slot_pos);
    };
    swap(seqs_[active_]);
    struct Unpark {
        std::function<void()> f;
        ~Unpark() { f(); }
    } unpark{[&] { swap(seqs_[active_]); }};
    const uint64_t spr = (Sb * 4 + 127) / 128 * 128;  // bytes per slot-position row
    if (!ctlb_.size()) ctlb_ = DeviceBuffer(dev_, kBSp + kMaxBatch * spr);
    auto& tk = topkb_[B];
    if (!tk.size()) {
        const auto c = Ops::topk2_constants(uint32_t(BN), vocab_, kBlock, kTopkChunk);
        tk = DeviceBuffer(dev_, c.size() * 4);
        std::memcpy(staging_.data(), c.data(), c.size() * 4);
        stream_.upload(staging_, 0, tk, 0, c.size() * 4);
        stream_.synchronize();
    }
    auto* base = staging_.as<uint8_t>();
    std::memset(base, 0, kBSp);
    auto* ids = reinterpret_cast<int32_t*>(base + kBIds);
    auto* rp = reinterpret_cast<int32_t*>(base + kBRope);
    auto* an = reinterpret_cast<int32_t*>(base + kBAnchor);
    auto* a16 = reinterpret_cast<int32_t*>(base + kBAnchor16);
    auto* pm = reinterpret_cast<float*>(base + kBPosm1);
    auto* lo = reinterpret_cast<float*>(base + kBLo);
    auto* na = reinterpret_cast<float*>(base + kBNotAnchor);
    for (uint32_t b = 0; b < B; ++b) {
        an[b] = anchors[b];
        for (uint64_t k = 0; k < K; ++k) a16[b * K + k] = anchors[b];
        pm[b] = float(pos[b]) - 1.f;
        const auto& sp = seqs_[slots[b]].slot_pos;
        auto* spf = reinterpret_cast<float*>(base + kBSp + b * spr);
        for (uint64_t c = 0; c < Sb; ++c) spf[c] = float(sp[c]);
        for (uint64_t t = 0; t < T; ++t) {
            const uint64_t r = b * T + t;
            ids[r] = t ? int32_t(mask_token_) : anchors[b];
            rp[r] = int32_t(pos[b] + t);
            lo[r] = window_ ? float(std::max<int64_t>(0, int64_t(pos[b] + t) - int64_t(window_) + 1)) : 0.f;
            na[r] = t ? 1.f : 0.f;
        }
    }
    stream_.upload(staging_, 0, ctlb_, 0, kBSp + B * spr);
    auto& r = block_batch_recipe(B, tables);
    std::vector<Binding> bnd = {{"ids", &ctlb_, kBIds},     {"rope_pos", &ctlb_, kBRope}, {"anchor", &ctlb_, kBAnchor},
                                {"anchor16", &ctlb_, kBAnchor16}, {"posm1", &ctlb_, kBPosm1}, {"lo", &ctlb_, kBLo},
                                {"notanchor", &ctlb_, kBNotAnchor}, {"sp", &ctlb_, kBSp},
                                {"cos_tab", &cos_tab_, 0},  {"sin_tab", &sin_tab_, 0},  {"causal", &causal_, 0},
                                tw_.bind("embed", "embed"), {"tg", &tg_, 0},          {"tsv", &tsv_, 0},
                                tw_.bind("lm_head", "lm_head"), {"cand", &out_idx_, 0}, {"idx", &out_idx_, 8192},
                                {"tk_consts", &tk, 0}};
    if (tables) bnd.push_back({"table", &out_idx_, 16384});
    if (lm8_on_) {
        bnd.push_back({"lm8", &lm8_, 0});
        bnd.push_back({"lm8s", &lm8s_, 0});
    }
    for (const auto& t : r.tensors()) {  // every weight the recipe declared, by name
        if (w_.has(t.name)) bnd.push_back(w_.bind(t.name, t.name));
        else if (auto it = w8_off_.find(t.name); it != w8_off_.end()) bnd.push_back({t.name, &w8_, it->second});
    }
    for (uint32_t L = 0; L < layers_; ++L) {
        const std::string p = "L" + std::to_string(L) + ".";
        for (uint32_t b = 0; b < B; ++b) {
            const Seq& q = seqs_[slots[b]];
            bnd.push_back({p + "kc" + std::to_string(b), &q.k[L], 0});
            bnd.push_back({p + "vc" + std::to_string(b), &q.v[L], 0});
        }
    }
    if (!ev0_) {
        ev0_ = std::make_unique<Event>(dev_);
        ev1_ = std::make_unique<Event>(dev_);
    }
    ev0_->record(stream_);
    r.enqueue(stream_, bnd);
    ev1_->record(stream_);
    // cand [BN, K], idx [BN, K] at 8192, tables [BN, K, K] at 16384
    stream_.download(out_idx_, 0, staging_, 0, tables ? 16384 + BN * K * K * 4 : 8192 + BN * K * 4);
    stream_.synchronize();
    last_ms_ = Event::elapsed_ms(*ev0_, *ev1_);
    const auto* cand = staging_.as<int32_t>();
    const auto* idx = reinterpret_cast<const int32_t*>(staging_.as<uint8_t>() + 8192);
    const auto* tab = reinterpret_cast<const float*>(staging_.as<uint8_t>() + 16384);
    std::vector<BatchDraft> out(B);
    for (uint32_t b = 0; b < B; ++b) {  // as draft(): the chain of best candidates, slot b's rows b*N ..
        const uint64_t r0 = uint64_t(b) * N;
        int32_t j = idx[r0 * K];
        for (uint64_t i = 0; i < N; ++i) {
            out[b].drafts.push_back(cand[(r0 + i) * K + uint64_t(j)]);
            if (i + 1 < N) j = idx[(r0 + i + 1) * K + uint64_t(j)];
        }
        if (tables) {  // as draft_table()
            out[b].table.cand.assign(cand + r0 * K, cand + (r0 + N) * K);
            out[b].table.table.assign(tab + r0 * K * K, tab + (r0 + N) * K * K);
        }
    }
    return out;
}

std::vector<int32_t> DSpine::draft(int32_t anchor, uint32_t pos) {
    const uint64_t N = kBlock - 1, K = kBlock;
    run_block(anchor, pos, false);
    const auto* cand = staging_.as<int32_t>();
    const auto* idx = reinterpret_cast<const int32_t*>(staging_.as<uint8_t>() + 4096);
    std::vector<int32_t> drafts;
    int32_t j = idx[0];  // row 0: every sender is the anchor
    for (uint64_t i = 0; i < N; ++i) {
        drafts.push_back(cand[i * K + uint64_t(j)]);
        if (i + 1 < N) j = idx[(i + 1) * K + uint64_t(j)];
    }
    return drafts;
}

DSpine::SampledDraft DSpine::draft_table(int32_t anchor, uint32_t pos) {
    const uint64_t N = kBlock - 1, K = kBlock;
    run_block(anchor, pos, true);
    SampledDraft d;
    d.cand.assign(staging_.as<int32_t>(), staging_.as<int32_t>() + N * K);
    const auto* t = reinterpret_cast<const float*>(staging_.as<uint8_t>() + 4096);
    d.table.assign(t, t + N * K * K);
    return d;
}

void DSpine::run_block(int32_t anchor, uint32_t pos, bool sampled) {
    static const bool timing = std::getenv("G2E_HOST_TIMING") != nullptr;
    static int calls = 0;
    const auto h0 = std::chrono::steady_clock::now();
    auto since = [&] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count(); };
    const uint64_t T = kBlock, D = head_dim_, R2 = D / 2, N = kBlock - 1, K = kBlock;
    if (pos + T > max_ctx_) throw std::out_of_range("dspine: context full");
    const uint32_t Sb = ring_;
    // one upload: controls [0, 4K) and the slot positions (f32) after them; RoPE rows and the mask come from
    // device tables / the graph
    (void)R2;
    auto* ctl = staging_.as<int32_t>();
    std::memset(ctl, 0, 4096);
    ctl[0] = anchor;
    for (uint64_t k = 0; k < K; ++k) ctl[32 + k] = anchor;
    auto* na = reinterpret_cast<float*>(ctl + 64);
    for (uint64_t t = 0; t < T; ++t) na[t] = t ? 1.f : 0.f;
    auto* ids = ctl + 128;  // byte 512: the block's token ids (anchor, then mask tokens)
    for (uint64_t t = 0; t < T; ++t) ids[t] = t ? int32_t(mask_token_) : anchor;
    reinterpret_cast<float*>(ctl)[256] = float(pos) - 1.f;  // byte 1024: pos - 1
    auto* lo = reinterpret_cast<float*>(ctl) + 288;           // byte 1152: lowest visible position per row
    for (uint64_t t = 0; t < T; ++t)
        lo[t] = window_ ? float(std::max<int64_t>(0, int64_t(pos + t) - int64_t(window_) + 1)) : 0.f;
    auto* spf = reinterpret_cast<float*>(staging_.as<uint8_t>() + 4096);
    for (uint32_t c = 0; c < Sb; ++c) spf[c] = float(slot_pos_[c]);
    stream_.upload(staging_, 0, ctl_, 0, 4096 + uint64_t(Sb) * 4);

    auto& r = block_recipe(sampled);
    auto& b = block_bind_[sampled ? 1 : 0];
    if (b.empty()) {
        // [1], [2]: RoPE rows, rebound per launch
        b = {{"ids", &ctl_, 512},    {"cos", &cos_tab_, 0},   {"sin", &sin_tab_, 0},
             tw_.bind("embed", "embed"),
             {"sp", &ctl_, 4096},    {"posm1", &ctl_, 1024},  {"lo", &ctl_, 1152},  {"causal", &causal_, 0},
             {"anchor", &ctl_, 0},   {"anchor16", &ctl_, 128},
             {"notanchor", &ctl_, 256}, {"tg", &tg_, 0},       {"tsv", &tsv_, 0},
             tw_.bind("lm_head", "lm_head"), {"cand", &out_idx_, 0},
             {sampled ? "table" : "idx", &out_idx_, 4096},
             {"tk_consts", &topk_aux_, 0}};
        if (lm8_on_) {
            b.push_back({"lm8", &lm8_, 0});
            b.push_back({"lm8s", &lm8s_, 0});
        }
        for (const auto& t : r.tensors()) {  // every weight the recipe declared, by name
            if (w_.has(t.name)) b.push_back(w_.bind(t.name, t.name));
            else if (auto it = w8_off_.find(t.name); it != w8_off_.end()) b.push_back({t.name, &w8_, it->second});
        }
        for (uint32_t L = 0; L < layers_; ++L) {
            const std::string p = "L" + std::to_string(L) + ".";
            b.push_back({p + "kc", &k_cache_[L], 0});
            b.push_back({p + "vc", &v_cache_[L], 0});
        }
    }
    b[1].offset = b[2].offset = uint64_t(pos) * (D / 2) * 4;
    const double t_prep = since();
    if (!ev0_) {
        ev0_ = std::make_unique<Event>(dev_);
        ev1_ = std::make_unique<Event>(dev_);
    }
    ev0_->record(stream_);
    r.enqueue(stream_, b);
    ev1_->record(stream_);
    const double t_launch = since();
    stream_.download(out_idx_, 0, staging_, 0, 4096 + N * K * (sampled ? K : 1) * 4);  // cand, then idx / table
    stream_.synchronize();
    last_ms_ = Event::elapsed_ms(*ev0_, *ev1_);
    if (timing && ++calls >= 5 && calls < 9)
        std::fprintf(stderr, "run_block host: prep %.0f us, launched %.0f us, done %.0f us (device %.0f us)\n", t_prep,
                     t_launch, since(), last_ms_ * 1e3);
}

}  // namespace g2e
