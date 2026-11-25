#include "g2e/inkling2_dspark.hpp"

#include "g2e/numeric.hpp"

#include <perf_lib_layer_params.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace g2e {
namespace {

void output(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const auto out = g.input(name, d.dtype, d.shape);
    g.op("identity", {src}, {out});
}
std::string Lw(uint32_t L, const char* part) { return "L" + std::to_string(L) + "." + part; }

struct Inputs {
    GraphBuilder& g;
    std::map<std::string, TensorId> ids;
    TensorId operator()(const std::string& name, DType dt, Shape shape) {
        if (auto it = ids.find(name); it != ids.end()) return it->second;
        return ids[name] = g.input(name, dt, std::move(shape));
    }
};

// neox RoPE over the full head: x [T, heads, D] with cos / sin [T, 1, D / 2] F32 -> BF16
TensorId rope_full(Ops& o, TensorId x, TensorId cosv, TensorId sinv, uint64_t D, const std::string& n) {
    const auto f = o.cast(x, DType::F32, n + "_f");
    const auto s = o.split(f, 2, {D / 2, D / 2}, n + "_s");
    const auto o1 = o.sub(o.mul(s[0], cosv, n + "_1c"), o.mul(s[1], sinv, n + "_2s"), n + "_o1");
    const auto o2 = o.add(o.mul(s[1], cosv, n + "_2c"), o.mul(s[0], sinv, n + "_1s"), n + "_o2");
    return o.cast(o.concat({o1, o2}, 2, n + "_c"), DType::BF16, n + "_b");
}

// bf16(rmsnorm(sum_j alpha[j] f_j) * w) in F32: f_j [n, H] BF16
TensorId mix_norm(Ops& o, const TensorId* f, const float* alpha, TensorId w, uint32_t n, uint32_t H, float eps, const std::string& nm) {
    (void)n;
    (void)H;
    TensorId acc = kNoTensor;
    for (int j = 0; j < 4; ++j) {
        const auto t = o.mul(o.cast(f[j], DType::F32, nm + "_f" + std::to_string(j)),
                             o.constant(alpha[j], {1, 1}, nm + "_a" + std::to_string(j)), nm + "_m" + std::to_string(j));
        acc = acc == kNoTensor ? t : o.add(acc, t, nm + "_p" + std::to_string(j));
    }
    return o.cast(o.rms_norm(acc, o.cast(w, DType::F32, nm + "_wf"), eps, false, nm + "_n"), DType::BF16, nm + "_b");
}

}  // namespace

InkDSpark::InkDSpark(const Device& device, const Stream& stream, const WeightPack& draft, const WeightPack& target,
                     uint32_t max_context)
    : dev_(device), stream_(stream), w_(draft), tw_(target), staging_(device, 8u << 20) {
    // Cap DSpark KV at 32K by default (396 MiB/slot vs 12 GiB at 1M); requests past max_ctx_ run Direct decode.
    static const uint32_t kDefaultCap = [] {
        const char* e = std::getenv("G2E_INK_DSPARK_MAX_CTX");
        const uint32_t v = e ? uint32_t(std::strtoul(e, nullptr, 0)) : 32768u;
        return std::max<uint32_t>(1024u, (v / 1024u) * 1024u);
    }();
    max_ctx_ = std::min(max_context, kDefaultCap);
    {
        const auto raw = w_.read("alpha");
        for (int r = 0; r < 4; ++r)
            for (int j = 0; j < 4; ++j) {
                uint16_t b;
                std::memcpy(&b, raw.data() + (r * 4 + j) * 2, 2);
                alpha_[r][j] = from_bf16(b);
            }
    }
    V_ = uint32_t(tw_.at("lm_head").shape[0]);
    Vfull_ = uint32_t(w_.at("mk.w1").shape[0]);
    Vpad_ = ((V_ + kTopkChunk - 1) / kTopkChunk) * kTopkChunk;
    cap_ = max_ctx_ + window_ + kBlock;  // window slices [lo, lo + window) never leave the caches
    slots_.resize(1);
    alloc_kv(kc_, vc_);
    {   // RoPE tables for every position
        const uint64_t R2 = D_ / 2, rows = cap_;
        cos_ = DeviceBuffer(dev_, rows * R2 * 4);
        sin_ = DeviceBuffer(dev_, rows * R2 * 4);
        const uint64_t per = staging_.size() / (R2 * 4) / 2;
        for (uint64_t r0 = 0; r0 < rows; r0 += per) {
            const uint64_t n = std::min(per, rows - r0);
            auto* c = staging_.as<float>();
            auto* s = c + per * R2;
            for (uint64_t r = 0; r < n; ++r)
                for (uint64_t i = 0; i < R2; ++i) {
                    const double a = double(r0 + r) * std::pow(double(theta_), -double(2 * i) / double(D_));
                    c[r * R2 + i] = float(std::cos(a));
                    s[r * R2 + i] = float(std::sin(a));
                }
            stream_.upload(staging_, 0, cos_, r0 * R2 * 4, n * R2 * 4);
            stream_.upload(staging_, per * R2 * 4, sin_, r0 * R2 * 4, n * R2 * 4);
            stream_.synchronize();
        }
    }
    masks_ = DeviceBuffer(dev_, (uint64_t(window_) + cap_) * 4 + 4096);
    ctl_ = DeviceBuffer(dev_, 4096);
    out_ = DeviceBuffer(dev_, 8192);
    {   // topk2 constants for [1, Vpad_] -> top-64 with chunk=256
        const auto c = Ops::topk2_constants(1, Vpad_, kSampleK, kTopkChunk);
        topk2_consts_ = DeviceBuffer(dev_, c.size() * 4);
        std::memcpy(staging_.data(), c.data(), c.size() * 4);
        stream_.upload(staging_, 0, topk2_consts_, 0, c.size() * 4);
        stream_.synchronize();
        // sample_aux_: [0..256) rank64 = {0..63} F32, [256..512) rank0_mask = {1, 0, ..., 0} F32
        sample_aux_ = DeviceBuffer(dev_, 512);
        auto* aux = staging_.as<float>();
        std::memset(aux, 0, 512);
        for (uint32_t i = 0; i < kSampleK; ++i) aux[i] = float(i);
        aux[kSampleK] = 1.0f;  // offset 256 bytes = 64 floats
        stream_.upload(staging_, 0, sample_aux_, 0, 512);
        stream_.synchronize();
    }
    reset();
}
InkDSpark::~InkDSpark() = default;

void InkDSpark::alloc_kv(std::vector<DeviceBuffer>& kc, std::vector<DeviceBuffer>& vc) {
    const uint64_t kvb = uint64_t(cap_) * KV_ * D_ * 2;
    kc.clear();
    vc.clear();
    for (int l = 0; l < 3; ++l) {
        kc.emplace_back(dev_, kvb);
        vc.emplace_back(dev_, kvb);
        stream_.zero(kc.back(), 0, kvb);
        stream_.zero(vc.back(), 0, kvb);
    }
    stream_.synchronize();
}

uint32_t InkDSpark::add_slot(bool alloc) {
    SlotKv sk;
    if (alloc) alloc_kv(sk.kc, sk.vc);
    slots_.push_back(std::move(sk));
    return uint32_t(slots_.size() - 1);
}

void InkDSpark::select_slot(uint32_t slot) {
    if (slot >= slots_.size()) throw std::out_of_range("dspark: bad slot");
    if (slot == active_slot_) return;
    std::swap(kc_, slots_[active_slot_].kc);
    std::swap(vc_, slots_[active_slot_].vc);
    active_slot_ = slot;
    std::swap(kc_, slots_[active_slot_].kc);
    std::swap(vc_, slots_[active_slot_].vc);
}

bool InkDSpark::slot_allocated(uint32_t slot) const {
    if (slot >= slots_.size()) return false;
    return slot == active_slot_ ? !kc_.empty() : !slots_[slot].kc.empty();
}

void InkDSpark::set_slot_allocated(uint32_t slot, bool on) {
    if (slot >= slots_.size()) throw std::out_of_range("dspark: bad slot");
    auto& kc = (slot == active_slot_) ? kc_ : slots_[slot].kc;
    auto& vc = (slot == active_slot_) ? vc_ : slots_[slot].vc;
    if (on) {
        if (kc.empty()) alloc_kv(kc, vc);
    } else {
        if (!kc.empty()) {
            stream_.synchronize();
            kc.clear();
            vc.clear();
        }
    }
}

void InkDSpark::reset() {
    if (kc_.empty()) return;
    for (auto* v : {&kc_, &vc_})
        for (auto& b : *v) stream_.zero(b, 0, b.size());
    stream_.synchronize();
}

void InkDSpark::save_rows(const DeviceBuffer& dst, uint64_t off, uint32_t n) const {
    if (kc_.empty()) throw std::logic_error("dspark: active slot not allocated");
    const uint64_t b = uint64_t(n) * KV_ * D_ * 2;
    for (int l = 0; l < 3; ++l) {
        stream_.copy(kc_[l], 0, dst, off, b);
        stream_.copy(vc_[l], 0, dst, off + b, b);
        off += 2 * b;
    }
}

void InkDSpark::load_rows(const DeviceBuffer& src, uint64_t off, uint32_t n) {
    if (kc_.empty()) alloc_kv(kc_, vc_);
    const uint64_t b = uint64_t(n) * KV_ * D_ * 2;
    for (int l = 0; l < 3; ++l) {
        stream_.copy(src, off, kc_[l], 0, b);
        stream_.copy(src, off + b, vc_[l], 0, b);
        off += 2 * b;
    }
}

void InkDSpark::save_rows(const HostBuffer& dst, uint64_t off, uint32_t n) const {
    if (kc_.empty()) throw std::logic_error("dspark: active slot not allocated");
    const uint64_t b = uint64_t(n) * KV_ * D_ * 2;
    if (!b) return;
    for (int l = 0; l < 3; ++l) {
        stream_.download(kc_[l], 0, dst, off, b);
        stream_.download(vc_[l], 0, dst, off + b, b);
        off += 2 * b;
    }
}

void InkDSpark::load_rows(const HostBuffer& src, uint64_t off, uint32_t n) {
    if (kc_.empty()) alloc_kv(kc_, vc_);
    const uint64_t b = uint64_t(n) * KV_ * D_ * 2;
    if (!b) return;
    for (int l = 0; l < 3; ++l) {
        stream_.upload(src, off, kc_[l], 0, b);
        stream_.upload(src, off + b, vc_[l], 0, b);
        off += 2 * b;
    }
}

void InkDSpark::precompile() {
    for (uint32_t n = 1; n <= 1024; n *= 2) append_recipe(n);
    for (uint32_t s = 1024;; s += s < 4096 ? 1024 : s / 2) {
        const uint32_t sb = std::min(s, max_ctx_);
        block_recipe(sb, 3);
        block_sampled_recipe(sb, 3);
        if (s >= max_ctx_) break;
    }
}

uint32_t InkDSpark::bucket(uint32_t rows) const {
    uint32_t b = 1024;
    while (b < rows) b += b < 4096 ? 1024 : b / 2;
    return std::min(b, max_ctx_);
}

Recipe& InkDSpark::append_recipe(uint32_t n) {
    const std::string key = "dspark_append_n" + std::to_string(n);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = H_, KV = KV_, D = D_;
    GraphBuilder g(dev_);
    Ops o(g);
    Inputs in{g, {}};
    TensorId f[4];
    for (int k = 0; k < 4; ++k) f[k] = in("feat" + std::to_string(k), DType::BF16, {n, H});
    const auto cosv = in("cos", DType::F32, {n, 1, D / 2}), sinv = in("sin", DType::F32, {n, 1, D / 2});
    for (uint32_t l = 0; l < 3; ++l) {
        const std::string p = "L" + std::to_string(l) + ".";
        const auto ctx = mix_norm(o, f, alpha_[l], in(Lw(l, "dflare_norm"), DType::BF16, {H}), n, uint32_t(H), eps_, p + "ctx");
        const auto kv = o.linear_bf16(ctx, in(Lw(l, "kvc_w"), DType::BF16, {2 * KV * D, H}), p + "kv");
        const auto s = o.split(kv, 1, {KV * D, KV * D}, p + "kvs");
        const auto k = rope_full(o, o.rms_norm(o.reshape(s[0], {n, KV, D}, p + "k3"), in(Lw(l, "k_norm_ctx"), DType::BF16, {D}), eps_, false,
                                               p + "kn"),
                                 cosv, sinv, D, p + "kr");
        output(g, p + "kout", k);
        output(g, p + "vout", o.reshape(s[1], {n, KV, D}, p + "v3"));
    }
    return *(recipes_[key] = g.compile(key));
}

void InkDSpark::append(const DeviceBuffer& feats, uint64_t off, uint64_t tap_stride, uint32_t n, uint32_t pos) {
    if (pos + n > max_ctx_) throw std::out_of_range("dspark: context full");
    if (kc_.empty()) alloc_kv(kc_, vc_);
    const uint64_t R2 = D_ / 2, row = uint64_t(KV_) * D_ * 2;
    uint32_t done = 0;
    while (done < n) {
        uint32_t piece = 1;
        while (piece * 2 <= n - done && piece * 2 <= 1024) piece *= 2;
        auto& r = append_recipe(piece);
        std::vector<Binding> b = {{"cos", &cos_, uint64_t(pos + done) * R2 * 4},
                                  {"sin", &sin_, uint64_t(pos + done) * R2 * 4}};
        for (int k = 0; k < 4; ++k) b.push_back({"feat" + std::to_string(k), &feats, off + uint64_t(k) * tap_stride + uint64_t(done) * H_ * 2});
        for (uint32_t l = 0; l < 3; ++l) {
            for (const char* part : {"dflare_norm", "kvc_w", "k_norm_ctx"}) b.push_back(w_.bind(Lw(l, part), Lw(l, part)));
            b.push_back({"L" + std::to_string(l) + ".kout", &kc_[l], uint64_t(pos + done) * row});
            b.push_back({"L" + std::to_string(l) + ".vout", &vc_[l], uint64_t(pos + done) * row});
        }
        r.enqueue(stream_, b);
        done += piece;
    }
}

Recipe& InkDSpark::block_recipe(uint32_t s1, uint32_t ndrafts) {
    if (ndrafts == 0 || ndrafts >= kBlock) throw std::invalid_argument("dspark: bad ndrafts");
    const std::string key = "dspark_block_s" + std::to_string(s1) + "_n" + std::to_string(ndrafts);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t T = kBlock, H = H_, NH = NH_, KV = KV_, D = D_, G = NH / KV, F = F_, V = V_, R = 256, W = window_;
    GraphBuilder g(dev_);
    Ops o(g);
    Inputs in{g, {}};
    auto BW = [&](const std::string& name, Shape s) { return in(name, DType::BF16, std::move(s)); };
    const auto cosv = in("cos", DType::F32, {T, 1, D / 2}), sinv = in("sin", DType::F32, {T, 1, D / 2});
    const auto anchor = in("anchor", DType::I32, {1});
    // block input: the anchor's normed target embedding, then the mask embedding x 7 (non-causal over all kBlock rows)
    const auto x0 = o.rms_norm(o.gather_rows(BW("embed", {Vfull_, H}), anchor, "x0g"), BW("embed_norm", {H}), eps_, false, "x0");
    const auto me = o.reshape(BW("mask_embed", {H}), {1, H}, "me");
    TensorId x = o.concat({x0, me, me, me, me, me, me, me}, 0, "x");
    const auto scale = o.constant(float(1.0 / std::sqrt(double(D))), {1}, "scale");
    for (uint32_t l = 0; l < 3; ++l) {
        const std::string p = "L" + std::to_string(l) + ".";
        const uint64_t S = l == 1 ? s1 : W;
        const auto a = o.rms_norm(x, BW(Lw(l, "input_ln"), {H}), eps_, false, p + "a");
        const auto qkv = o.linear_bf16(a, BW(Lw(l, "qkv_w"), {(NH + 2 * KV) * D, H}), p + "qkv");
        const auto qs = o.split(qkv, 1, {NH * D, KV * D, KV * D}, p + "qkvs");
        const auto q = rope_full(o, o.rms_norm(o.reshape(qs[0], {T, NH, D}, p + "q3"), BW(Lw(l, "q_norm"), {D}), eps_, false, p + "qn"),
                                 cosv, sinv, D, p + "qr");
        const auto k = rope_full(o, o.rms_norm(o.reshape(qs[1], {T, KV, D}, p + "k3"), BW(Lw(l, "k_norm"), {D}), eps_, false, p + "kn"),
                                 cosv, sinv, D, p + "kr");
        const auto v = o.reshape(qs[2], {T, KV, D}, p + "v3");
        const auto q4 = o.reshape(o.transpose(q, {1, 0, 2}, p + "qt"), {KV, G * T, D}, p + "q4");
        const auto qsc = o.cast(o.mul(o.cast(q4, DType::F32, p + "qf"), scale, p + "qm"), DType::BF16, p + "qsb");
        const auto kb = o.transpose(k, {1, 0, 2}, p + "kb");
        const auto vb = o.transpose(v, {1, 0, 2}, p + "vb");
        const auto kc = o.transpose(BW(p + "kc", {S, KV, D}), {1, 0, 2}, p + "kct");
        const auto vc = o.transpose(BW(p + "vc", {S, KV, D}), {1, 0, 2}, p + "vct");
        const auto mask = o.reshape(in(l == 1 ? "fmask" : "wmask", DType::F32, {1, S}), {1, 1, S}, p + "m3");
        const auto s1v = o.add(o.batch_gemm(qsc, kc, p + "sc", false, true, DType::F32), mask, p + "s1");  // [KV, G T, S]
        const auto s2v = o.batch_gemm(qsc, kb, p + "sb", false, true, DType::F32);                       // [KV, G T, T]
        const auto pr = o.cast(o.softmax(o.concat({s1v, s2v}, 2, p + "sall"), p + "pr"), DType::BF16, p + "prb");
        const auto ps = o.split(pr, 2, {S, T}, p + "ps");
        const auto att3 = o.add(o.batch_gemm(ps[0], vc, p + "oc", false, false, DType::F32),
                                o.batch_gemm(ps[1], vb, p + "ob", false, false, DType::F32), p + "ao");  // [KV, G T, D]
        const auto att = o.reshape(o.transpose(o.reshape(o.cast(att3, DType::BF16, p + "attb"), {NH, T, D}, p + "a3"), {1, 0, 2}, p + "at"),
                                   {T, NH * D}, p + "a2");
        const auto h = o.add(x, o.linear_bf16(att, BW(Lw(l, "o_w"), {H, NH * D}), p + "o"), p + "h");
        const auto b = o.rms_norm(h, BW(Lw(l, "post_ln"), {H}), eps_, false, p + "b");
        const auto gu = o.split(o.linear_bf16(b, BW(Lw(l, "gu_w"), {2 * F, H}), p + "gu"), 1, {F, F}, p + "gus");
        const auto gf = o.cast(gu[0], DType::F32, p + "gf");
        const auto act = o.cast(o.mul(o.unary("silu", gf, p + "si"), o.cast(gu[1], DType::F32, p + "uf"), p + "su"), DType::BF16, p + "act");
        x = o.add(h, o.linear_bf16(act, BW(Lw(l, "down_w"), {H, F}), p + "dn"), p + "x");
    }
    const auto hf = o.rms_norm(x, BW("norm", {H}), eps_, false, "hf");
    // Slice only rows 1..ndrafts to avoid wasted lm_head + Markov fold_w work when ndrafts < 7
    const auto hfN = ndrafts == T - 1 ? o.split(hf, 0, {1, T - 1}, "hfs")[1]
                                      : o.split(hf, 0, {1, ndrafts, T - 1 - ndrafts}, "hfs")[1];
    const auto base = o.linear_bf16(hfN, BW("lm_head", {V, H}), "base");  // [ndrafts, V] BF16
    TensorId fl[4];
    for (int k = 0; k < 4; ++k) fl[k] = BW("flast" + std::to_string(k), {1, H});
    const auto frnn = mix_norm(o, fl, alpha_[3], BW("mk.dflare_norm", {H}), 1, uint32_t(H), eps_, "frnn");
    const auto C = o.add(o.add(o.linear_bf16(hfN, BW("mk.wh", {3 * R, H}), "ch", DType::F32),
                               o.linear_bf16(frnn, BW("mk.wf", {3 * R, H}), "cf", DType::F32), "chf"),
                         o.reshape(o.cast(BW("mk.rb", {3 * R}), DType::F32, "rbf"), {1, 3 * R}, "rb2"), "C");  // [ndrafts, 768]
    const std::vector<TensorId> crow = ndrafts == 1 ? std::vector<TensorId>{C}
                                                    : o.split(C, 0, std::vector<uint64_t>(ndrafts, 1), "crow");
    const std::vector<TensorId> brow = ndrafts == 1 ? std::vector<TensorId>{base}
                                                    : o.split(base, 0, std::vector<uint64_t>(ndrafts, 1), "brow");
    TensorId state = o.constant(0.f, {1, R}, "st0", DType::BF16);
    TensorId prev = anchor;
    const auto w1 = BW("mk.w1", {Vfull_, R});
    const auto foldb = o.reshape(BW("mk.fold_b", {V}), {1, V}, "fb");
    std::vector<TensorId> drafts;
    drafts.reserve(ndrafts);
    for (uint32_t i = 0; i < ndrafts; ++i) {
        const std::string p = "mk" + std::to_string(i) + ".";
        const auto raw = o.add(o.add(crow[i], o.linear_bf16(state, BW("mk.ws", {3 * R, R}), p + "rs", DType::F32), p + "r1"),
                               o.linear_bf16(o.gather_rows(w1, prev, p + "w1"), BW("mk.wp", {3 * R, R}), p + "rp", DType::F32), p + "raw");
        const auto gco = o.split(raw, 1, {R, R, R}, p + "gco");
        auto bfr = [&](TensorId t, const std::string& n) { return o.cast(o.cast(t, DType::BF16, n + "_b"), DType::F32, n + "_f"); };
        const auto gg = bfr(o.unary("sigmoid", bfr(gco[0], p + "g0"), p + "gs"), p + "g");
        const auto cc = bfr(o.unary("tanh", bfr(gco[1], p + "c0"), p + "ct"), p + "c");
        const auto oo = bfr(gco[2], p + "o");
        const auto stf = o.cast(state, DType::F32, p + "stf");
        const auto gs = bfr(o.mul(gg, stf, p + "gsm"), p + "gs2");
        const auto omg = bfr(o.sub(o.constant(1.f, {1, 1}, p + "one"), gg, p + "omg0"), p + "omg");
        const auto oc = bfr(o.mul(omg, cc, p + "ocm"), p + "oc");
        state = o.cast(o.add(gs, oc, p + "sta"), DType::BF16, p + "st");
        const auto lat = o.cast(o.unary("tanh", oo, p + "lt"), DType::BF16, p + "lat");
        const auto bias = o.add(o.linear_bf16(lat, BW("mk.fold_w", {V, R}), p + "fw"), foldb, p + "bias");  // BF16 [1, V]
        // Greedy argmax: skip redundant 30*tanh(z/30) softcap (tanh is strictly monotonic)
        const auto z = o.cast(o.add(brow[i], bias, p + "zb"), DType::F32, p + "z");
        prev = o.reshape(o.argmax(z, p + "am"), {1}, p + "prev");
        drafts.push_back(prev);
    }
    output(g, "drafts", ndrafts == 1 ? drafts[0] : o.concat(drafts, 0, "dr"));
    return *(recipes_[key] = g.compile(key));
}

Recipe& InkDSpark::block_sampled_recipe(uint32_t s1, uint32_t ndrafts) {
    if (ndrafts == 0 || ndrafts >= kBlock) throw std::invalid_argument("dspark: bad ndrafts");
    const std::string key = "dspark_block_samp_s" + std::to_string(s1) + "_n" + std::to_string(ndrafts);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t T = kBlock, H = H_, NH = NH_, KV = KV_, D = D_, G = NH / KV, F = F_, V = V_, Vpad = Vpad_, R = 256, W = window_;
    const uint64_t pad = Vpad - V;
    GraphBuilder g(dev_);
    Ops o(g);
    Inputs in{g, {}};
    auto BW = [&](const std::string& name, Shape s) { return in(name, DType::BF16, std::move(s)); };
    const auto cosv = in("cos", DType::F32, {T, 1, D / 2}), sinv = in("sin", DType::F32, {T, 1, D / 2});
    const auto anchor = in("anchor", DType::I32, {1});
    // block input: non-causal 3-layer transformer over all kBlock=8 rows
    const auto x0 = o.rms_norm(o.gather_rows(BW("embed", {Vfull_, H}), anchor, "x0g"), BW("embed_norm", {H}), eps_, false, "x0");
    const auto me = o.reshape(BW("mask_embed", {H}), {1, H}, "me");
    TensorId x = o.concat({x0, me, me, me, me, me, me, me}, 0, "x");
    const auto scale = o.constant(float(1.0 / std::sqrt(double(D))), {1}, "scale");
    for (uint32_t l = 0; l < 3; ++l) {
        const std::string p = "L" + std::to_string(l) + ".";
        const uint64_t S = l == 1 ? s1 : W;
        const auto a = o.rms_norm(x, BW(Lw(l, "input_ln"), {H}), eps_, false, p + "a");
        const auto qkv = o.linear_bf16(a, BW(Lw(l, "qkv_w"), {(NH + 2 * KV) * D, H}), p + "qkv");
        const auto qs = o.split(qkv, 1, {NH * D, KV * D, KV * D}, p + "qkvs");
        const auto q = rope_full(o, o.rms_norm(o.reshape(qs[0], {T, NH, D}, p + "q3"), BW(Lw(l, "q_norm"), {D}), eps_, false, p + "qn"),
                                 cosv, sinv, D, p + "qr");
        const auto k = rope_full(o, o.rms_norm(o.reshape(qs[1], {T, KV, D}, p + "k3"), BW(Lw(l, "k_norm"), {D}), eps_, false, p + "kn"),
                                 cosv, sinv, D, p + "kr");
        const auto v = o.reshape(qs[2], {T, KV, D}, p + "v3");
        const auto q4 = o.reshape(o.transpose(q, {1, 0, 2}, p + "qt"), {KV, G * T, D}, p + "q4");
        const auto qsc = o.cast(o.mul(o.cast(q4, DType::F32, p + "qf"), scale, p + "qm"), DType::BF16, p + "qsb");
        const auto kb = o.transpose(k, {1, 0, 2}, p + "kb");
        const auto vb = o.transpose(v, {1, 0, 2}, p + "vb");
        const auto kc = o.transpose(BW(p + "kc", {S, KV, D}), {1, 0, 2}, p + "kct");
        const auto vc = o.transpose(BW(p + "vc", {S, KV, D}), {1, 0, 2}, p + "vct");
        const auto mask = o.reshape(in(l == 1 ? "fmask" : "wmask", DType::F32, {1, S}), {1, 1, S}, p + "m3");
        const auto s1v = o.add(o.batch_gemm(qsc, kc, p + "sc", false, true, DType::F32), mask, p + "s1");
        const auto s2v = o.batch_gemm(qsc, kb, p + "sb", false, true, DType::F32);
        const auto pr = o.cast(o.softmax(o.concat({s1v, s2v}, 2, p + "sall"), p + "pr"), DType::BF16, p + "prb");
        const auto ps = o.split(pr, 2, {S, T}, p + "ps");
        const auto att3 = o.add(o.batch_gemm(ps[0], vc, p + "oc", false, false, DType::F32),
                                o.batch_gemm(ps[1], vb, p + "ob", false, false, DType::F32), p + "ao");
        const auto att = o.reshape(o.transpose(o.reshape(o.cast(att3, DType::BF16, p + "attb"), {NH, T, D}, p + "a3"), {1, 0, 2}, p + "at"),
                                   {T, NH * D}, p + "a2");
        const auto h = o.add(x, o.linear_bf16(att, BW(Lw(l, "o_w"), {H, NH * D}), p + "o"), p + "h");
        const auto b = o.rms_norm(h, BW(Lw(l, "post_ln"), {H}), eps_, false, p + "b");
        const auto gu = o.split(o.linear_bf16(b, BW(Lw(l, "gu_w"), {2 * F, H}), p + "gu"), 1, {F, F}, p + "gus");
        const auto gf = o.cast(gu[0], DType::F32, p + "gf");
        const auto act = o.cast(o.mul(o.unary("silu", gf, p + "si"), o.cast(gu[1], DType::F32, p + "uf"), p + "su"), DType::BF16, p + "act");
        x = o.add(h, o.linear_bf16(act, BW(Lw(l, "down_w"), {H, F}), p + "dn"), p + "x");
    }
    const auto hf = o.rms_norm(x, BW("norm", {H}), eps_, false, "hf");
    const auto hfN = ndrafts == T - 1 ? o.split(hf, 0, {1, T - 1}, "hfs")[1]
                                      : o.split(hf, 0, {1, ndrafts, T - 1 - ndrafts}, "hfs")[1];
    const auto base = o.linear_bf16(hfN, BW("lm_head", {V, H}), "base");  // [ndrafts, V] BF16
    TensorId fl[4];
    for (int k = 0; k < 4; ++k) fl[k] = BW("flast" + std::to_string(k), {1, H});
    const auto frnn = mix_norm(o, fl, alpha_[3], BW("mk.dflare_norm", {H}), 1, uint32_t(H), eps_, "frnn");
    const auto C = o.add(o.add(o.linear_bf16(hfN, BW("mk.wh", {3 * R, H}), "ch", DType::F32),
                               o.linear_bf16(frnn, BW("mk.wf", {3 * R, H}), "cf", DType::F32), "chf"),
                         o.reshape(o.cast(BW("mk.rb", {3 * R}), DType::F32, "rbf"), {1, 3 * R}, "rb2"), "C");  // [ndrafts, 768]
    const std::vector<TensorId> crow = ndrafts == 1 ? std::vector<TensorId>{C}
                                                    : o.split(C, 0, std::vector<uint64_t>(ndrafts, 1), "crow");
    const std::vector<TensorId> brow = ndrafts == 1 ? std::vector<TensorId>{base}
                                                    : o.split(base, 0, std::vector<uint64_t>(ndrafts, 1), "brow");
    TensorId state = o.constant(0.f, {1, R}, "st0", DType::BF16);
    TensorId prev = anchor;
    const auto w1 = BW("mk.w1", {Vfull_, R});
    const auto foldb = o.reshape(BW("mk.fold_b", {V}), {1, V}, "fb");

    // Sampling controls & constants
    const auto pad_const = pad ? o.constant(-1e30f, {1, pad}, "lg_pad_c") : kNoTensor;
    const auto tk_consts = in("tk_consts", DType::I32, {Vpad + 2});
    const auto rank64 = in("rank64", DType::F32, {1, kSampleK});
    const auto rank0_mask = in("rank0_mask", DType::F32, {1, kSampleK});
    const auto inv_t = in("inv_t", DType::F32, {1, 1});
    const auto topk_f = in("topk_f", DType::F32, {1, 1});
    const auto topp_f = in("topp_f", DType::F32, {1, 1});
    const auto minp_f = in("minp_f", DType::F32, {1, 1});
    const auto u_rand = in("u_rand", DType::F32, {ndrafts, 1});
    const std::vector<TensorId> urow = ndrafts == 1 ? std::vector<TensorId>{u_rand}
                                                    : o.split(u_rand, 0, std::vector<uint64_t>(ndrafts, 1), "urow");
    const auto zero64 = o.constant(0.f, {1, kSampleK}, "zero64");
    const auto c63 = o.constant(float(kSampleK - 1), {1, 1}, "c63");

    // Broadcast via sub_fwd_f32 into [1, 64], then compare elementwise against zero64 -> I8 (0/1) -> F32 (0.f/1.f)
    auto cmp_f32 = [&](const char* guid, TensorId a, TensorId b, const std::string& nm) {
        const auto diff = o.sub(a, b, nm + "_d");
        const auto m = o.temp(nm + "_i8", DType::I8, o.shape(diff));
        g.op(guid, {diff, zero64}, {m});
        return o.cast(m, DType::F32, nm);
    };
    auto cumsum_last = [&](TensorId t, bool exclusive, const std::string& nm) {
        const auto out = o.temp(nm, DType::F32, o.shape(t));
        ns_CumSumKernel::Params cp{};
        cp.axis = 0;  // innermost dimension in Synapse order (kSampleK for [1, kSampleK])
        cp.exclusive = exclusive ? 1 : 0;
        cp.reverse = 0;
        g.op("cumsum_fwd_f32", {t}, {out}, &cp, sizeof(cp));
        return out;
    };
    const auto m_topk = cmp_f32("less_fwd_f32", rank64, topk_f, "m_topk");

    std::vector<TensorId> drafts, cand_ids, cand_qs;
    drafts.reserve(ndrafts);
    cand_ids.reserve(ndrafts);
    cand_qs.reserve(ndrafts);
    for (uint32_t i = 0; i < ndrafts; ++i) {
        const std::string p = "mk" + std::to_string(i) + ".";
        const auto raw = o.add(o.add(crow[i], o.linear_bf16(state, BW("mk.ws", {3 * R, R}), p + "rs", DType::F32), p + "r1"),
                               o.linear_bf16(o.gather_rows(w1, prev, p + "w1"), BW("mk.wp", {3 * R, R}), p + "rp", DType::F32), p + "raw");
        const auto gco = o.split(raw, 1, {R, R, R}, p + "gco");
        auto bfr = [&](TensorId t, const std::string& n) { return o.cast(o.cast(t, DType::BF16, n + "_b"), DType::F32, n + "_f"); };
        const auto gg = bfr(o.unary("sigmoid", bfr(gco[0], p + "g0"), p + "gs"), p + "g");
        const auto cc = bfr(o.unary("tanh", bfr(gco[1], p + "c0"), p + "ct"), p + "c");
        const auto oo = bfr(gco[2], p + "o");
        const auto stf = o.cast(state, DType::F32, p + "stf");
        const auto gs = bfr(o.mul(gg, stf, p + "gsm"), p + "gs2");
        const auto omg = bfr(o.sub(o.constant(1.f, {1, 1}, p + "one"), gg, p + "omg0"), p + "omg");
        const auto oc = bfr(o.mul(omg, cc, p + "ocm"), p + "oc");
        state = o.cast(o.add(gs, oc, p + "sta"), DType::BF16, p + "st");
        const auto lat = o.cast(o.unary("tanh", oo, p + "lt"), DType::BF16, p + "lat");
        const auto bias = o.add(o.linear_bf16(lat, BW("mk.fold_w", {V, R}), p + "fw"), foldb, p + "bias");  // BF16 [1, V]
        // Keep exact 30*tanh(z/30) softcap for sampled decoding (temperature depends on softcapped magnitudes)
        const auto z = bfr(o.add(o.cast(brow[i], DType::F32, p + "bf"), o.cast(bias, DType::F32, p + "bsf"), p + "z0"), p + "z");
        const auto lg = bfr(o.mul(o.unary("tanh", o.mul(z, o.constant(1.f / 30.f, {1, 1}, p + "i30"), p + "zs"), p + "zt"),
                                  o.constant(30.f, {1, 1}, p + "c30"), p + "lgm"),
                            p + "lg");
        const auto lg_pad = pad ? o.concat({lg, pad_const}, 1, p + "lgp") : lg;
        const auto [top_v, top_id] = o.topk2(lg_pad, kSampleK, kTopkChunk, tk_consts, p + "tk");  // [1, 64] sorted desc
        const auto pr0 = o.softmax(o.mul(top_v, inv_t, p + "ts"), p + "pr0");
        const auto prk = o.mul(pr0, m_topk, p + "prk");
        const auto pr = o.mul(prk, o.unary("reciprocal", o.reduce("sum", prk, 1, p + "prks"), p + "prki"), p + "pr");
        const auto cdf_prev = cumsum_last(pr, true, p + "cdfp");  // exclusive prefix sum, cdf_prev[0] == 0
        const auto m_topp = cmp_f32("less_fwd_f32", cdf_prev, topp_f, p + "m_topp");
        const auto pr_max = o.split(pr, 1, {1, kSampleK - 1}, p + "pr0s")[0];
        const auto m_minp = cmp_f32("greater_equal_fwd_f32", pr, o.mul(pr_max, minp_f, p + "mthr"), p + "m_minp");
        const auto keep = o.binary("max", o.mul(o.mul(m_topk, m_topp, p + "k1"), m_minp, p + "k2"), rank0_mask, p + "keep");
        const auto w = o.mul(pr, keep, p + "w");  // w[0] > 0 always
        const auto wcdf = cumsum_last(w, false, p + "wcdf");  // monotone non-decreasing, flat where w[j] == 0
        const auto wsum = o.split(wcdf, 1, {kSampleK - 1, 1}, p + "ws")[1];
        const auto q = o.mul(w, o.unary("reciprocal", wsum, p + "wsi"), p + "q");  // exact normalized draft prob
        const auto u_thr = o.mul(urow[i], wsum, p + "uthr");  // strictly < wsum since u_rand[i] < 1.0f
        const auto lt = cmp_f32("less_fwd_f32", wcdf, u_thr, p + "wlt");
        const auto choice_f = o.binary("min", o.reduce("sum", lt, 1, p + "ch_s"), c63, p + "ch_c");
        const auto choice = o.reshape(o.cast(choice_f, DType::I32, p + "ch_i"), {1}, p + "ch");
        prev = o.reshape(o.gather_rows(o.reshape(top_id, {kSampleK, 1}, p + "ti2"), choice, p + "tg"), {1}, p + "prev");
        drafts.push_back(prev);
        cand_ids.push_back(top_id);
        cand_qs.push_back(q);
    }
    output(g, "drafts", ndrafts == 1 ? drafts[0] : o.concat(drafts, 0, "dr"));
    output(g, "cand_ids", ndrafts == 1 ? cand_ids[0] : o.concat(cand_ids, 0, "cids"));
    output(g, "cand_q", ndrafts == 1 ? cand_qs[0] : o.concat(cand_qs, 0, "cqs"));
    return *(recipes_[key] = g.compile(key));
}

std::vector<int32_t> InkDSpark::draft(int32_t anchor, uint32_t apos, const DeviceBuffer& feats, uint64_t off, uint64_t tap_stride,
                                      uint32_t ndrafts) {
    const auto t0 = std::chrono::steady_clock::now();
    if (ndrafts == 0 || ndrafts >= kBlock) throw std::invalid_argument("dspark: bad ndrafts");
    if (kc_.empty()) alloc_kv(kc_, vc_);
    const uint32_t s1 = bucket(apos);
    if (apos > s1 || apos + kBlock > cap_) throw std::out_of_range("dspark: position out of range");
    const uint32_t lo = apos > window_ ? apos - window_ : 0;
    // controls: anchor; masks: window rows lo .. lo + W - 1 (< apos visible), full-layer rows 0 .. s1 - 1
    auto* m = staging_.as<float>();
    for (uint32_t c = 0; c < window_; ++c) m[c] = lo + c < apos ? 0.f : -1e30f;
    for (uint32_t c = 0; c < s1; ++c) m[window_ + c] = c < apos ? 0.f : -1e30f;
    const uint64_t moff = 0, aoff = ((uint64_t(window_) + s1) * 4 + 127) / 128 * 128;  // (bindings: 128-byte aligned)
    std::memcpy(staging_.as<uint8_t>() + aoff, &anchor, 4);
    stream_.upload(staging_, 0, masks_, 0, aoff + 4);
    auto& r = block_recipe(s1, ndrafts);
    const uint64_t R2 = D_ / 2, row = uint64_t(KV_) * D_ * 2;
    std::vector<Binding> b = {{"cos", &cos_, uint64_t(apos) * R2 * 4}, {"sin", &sin_, uint64_t(apos) * R2 * 4},
                              {"anchor", &masks_, aoff}, tw_.bind("embed", "embed"), tw_.bind("embed_norm", "embed_norm"),
                              tw_.bind("lm_head", "lm_head"), w_.bind("mask_embed", "mask_embed"), w_.bind("norm", "norm"),
                              {"drafts", &out_, 0}};
    for (int k = 0; k < 4; ++k) b.push_back({"flast" + std::to_string(k), &feats, off + uint64_t(k) * tap_stride});
    for (const char* n : {"mk.dflare_norm", "mk.wh", "mk.wf", "mk.rb", "mk.w1", "mk.ws", "mk.wp", "mk.fold_w", "mk.fold_b"})
        b.push_back(w_.bind(n, n));
    for (uint32_t l = 0; l < 3; ++l) {
        for (const char* part : {"input_ln", "qkv_w", "q_norm", "k_norm", "o_w", "post_ln", "gu_w", "down_w"})
            b.push_back(w_.bind(Lw(l, part), Lw(l, part)));
        const std::string p = "L" + std::to_string(l) + ".";
        const uint64_t start = l == 1 ? 0 : lo;
        b.push_back({p + "kc", &kc_[l], start * row});
        b.push_back({p + "vc", &vc_[l], start * row});
    }
    b.push_back({"wmask", &masks_, moff});
    b.push_back({"fmask", &masks_, moff + uint64_t(window_) * 4});
    r.enqueue(stream_, b);
    stream_.download(out_, 0, staging_, 0, uint64_t(ndrafts) * 4);
    stream_.synchronize();
    std::vector<int32_t> d(ndrafts);
    std::memcpy(d.data(), staging_.data(), uint64_t(ndrafts) * 4);
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return d;
}

InkDSpark::SampledDraft InkDSpark::draft_sampled(int32_t anchor, uint32_t apos, const DeviceBuffer& feats, uint64_t off,
                                                 uint64_t tap_stride, float temperature, int top_k, float top_p, float min_p,
                                                 std::mt19937_64& rng, uint32_t ndrafts) {
    const auto t0 = std::chrono::steady_clock::now();
    if (ndrafts == 0 || ndrafts >= kBlock) throw std::invalid_argument("dspark: bad ndrafts");
    if (kc_.empty()) alloc_kv(kc_, vc_);
    const uint32_t s1 = bucket(apos);
    if (apos > s1 || apos + kBlock > cap_) throw std::out_of_range("dspark: position out of range");
    const uint32_t lo = apos > window_ ? apos - window_ : 0;

    auto* m = staging_.as<float>();
    for (uint32_t c = 0; c < window_; ++c) m[c] = lo + c < apos ? 0.f : -1e30f;
    for (uint32_t c = 0; c < s1; ++c) m[window_ + c] = c < apos ? 0.f : -1e30f;
    const uint64_t moff = 0;
    const uint64_t aoff = ((uint64_t(window_) + s1) * 4 + 127) / 128 * 128;
    const uint64_t inv_t_off = aoff + 128;
    const uint64_t topk_off = aoff + 256;
    const uint64_t topp_off = aoff + 384;
    const uint64_t minp_off = aoff + 512;
    const uint64_t urand_off = aoff + 640;

    const float inv_t_v = 1.0f / std::max(temperature, 1e-5f);
    const float topk_v = (top_k > 0 && top_k < int(kSampleK)) ? float(top_k) : float(kSampleK);
    const float topp_v = top_p >= 1.0f ? 2.0f : std::max(0.0f, top_p);
    const float minp_v = std::max(0.0f, min_p);
    float u_vals[kBlock];
    std::uniform_real_distribution<double> udist(0.0, 1.0);
    for (uint32_t i = 0; i < ndrafts; ++i) {
        float u = float(udist(rng));
        if (!(u < 1.0f)) u = std::nextafter(1.0f, 0.0f);  // strictly < 1.0f in F32 so u*wsum < wsum
        u_vals[i] = u;
    }
    auto* base_p = staging_.as<uint8_t>();
    std::memcpy(base_p + aoff, &anchor, 4);
    std::memcpy(base_p + inv_t_off, &inv_t_v, 4);
    std::memcpy(base_p + topk_off, &topk_v, 4);
    std::memcpy(base_p + topp_off, &topp_v, 4);
    std::memcpy(base_p + minp_off, &minp_v, 4);
    std::memcpy(base_p + urand_off, u_vals, uint64_t(ndrafts) * 4);
    stream_.upload(staging_, 0, masks_, 0, urand_off + uint64_t(ndrafts) * 4);

    auto& r = block_sampled_recipe(s1, ndrafts);
    const uint64_t off_drafts = 0;
    const uint64_t off_cids = 128;
    const uint64_t cids_bytes = uint64_t(ndrafts) * kSampleK * 4;
    const uint64_t off_cq = off_cids + ((cids_bytes + 127) / 128) * 128;
    const uint64_t total_out = off_cq + cids_bytes;

    const uint64_t R2 = D_ / 2, row = uint64_t(KV_) * D_ * 2;
    std::vector<Binding> b = {{"cos", &cos_, uint64_t(apos) * R2 * 4}, {"sin", &sin_, uint64_t(apos) * R2 * 4},
                              {"anchor", &masks_, aoff}, tw_.bind("embed", "embed"), tw_.bind("embed_norm", "embed_norm"),
                              tw_.bind("lm_head", "lm_head"), w_.bind("mask_embed", "mask_embed"), w_.bind("norm", "norm"),
                              {"tk_consts", &topk2_consts_, 0}, {"rank64", &sample_aux_, 0}, {"rank0_mask", &sample_aux_, 256},
                              {"inv_t", &masks_, inv_t_off}, {"topk_f", &masks_, topk_off}, {"topp_f", &masks_, topp_off},
                              {"minp_f", &masks_, minp_off}, {"u_rand", &masks_, urand_off},
                              {"drafts", &out_, off_drafts}, {"cand_ids", &out_, off_cids}, {"cand_q", &out_, off_cq}};
    for (int k = 0; k < 4; ++k) b.push_back({"flast" + std::to_string(k), &feats, off + uint64_t(k) * tap_stride});
    for (const char* n : {"mk.dflare_norm", "mk.wh", "mk.wf", "mk.rb", "mk.w1", "mk.ws", "mk.wp", "mk.fold_w", "mk.fold_b"})
        b.push_back(w_.bind(n, n));
    for (uint32_t l = 0; l < 3; ++l) {
        for (const char* part : {"input_ln", "qkv_w", "q_norm", "k_norm", "o_w", "post_ln", "gu_w", "down_w"})
            b.push_back(w_.bind(Lw(l, part), Lw(l, part)));
        const std::string p = "L" + std::to_string(l) + ".";
        const uint64_t start = l == 1 ? 0 : lo;
        b.push_back({p + "kc", &kc_[l], start * row});
        b.push_back({p + "vc", &vc_[l], start * row});
    }
    b.push_back({"wmask", &masks_, moff});
    b.push_back({"fmask", &masks_, moff + uint64_t(window_) * 4});
    r.enqueue(stream_, b);
    stream_.download(out_, 0, staging_, 0, total_out);
    stream_.synchronize();

    SampledDraft res;
    res.k = kSampleK;
    res.drafts.resize(ndrafts);
    res.cand_ids.resize(size_t(ndrafts) * kSampleK);
    res.cand_q.resize(size_t(ndrafts) * kSampleK);
    std::memcpy(res.drafts.data(), base_p + off_drafts, uint64_t(ndrafts) * 4);
    std::memcpy(res.cand_ids.data(), base_p + off_cids, cids_bytes);
    std::memcpy(res.cand_q.data(), base_p + off_cq, cids_bytes);
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

}  // namespace g2e
