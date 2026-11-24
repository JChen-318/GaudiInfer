#include "g2e/inkling_vision.hpp"

#include "g2e/numeric.hpp"
#include "g2e/ops.hpp"

#include <perf_lib_layer_params.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace g2e {
namespace {

uint64_t align128(uint64_t x) { return (x + 127) & ~uint64_t(127); }

// bilinear taps (align_corners=False, border clamp) of target index i on an axis of n onto a `side` grid
void taps(uint32_t i, uint32_t n, uint32_t side, int t[2], float w[2]) {
    const float src = (float(i) + 0.5f) * float(side) / float(n) - 0.5f;
    const float fl = std::floor(src), fr = src - fl;
    t[0] = std::clamp(int(fl), 0, int(side) - 1);
    t[1] = std::clamp(int(fl) + 1, 0, int(side) - 1);
    w[0] = std::max(0.f, 1.f - fr);
    w[1] = std::max(0.f, 1.f - std::fabs(fr - 1.f));
}

}  // namespace

InklingVision::InklingVision(const Device& device, const Stream& stream, const WeightPack& pack)
    : dev_(device), stream_(stream), w_(pack), staging_(device, 96u << 20) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& c = j.at("config");
    if (c.value("model_type", std::string()) != "inkling2_vision") throw std::runtime_error("inkling vision: not an inkling2_vision pack");
    depth_ = c.at("num_hidden_layers");
    hidden_ = c.at("hidden_size");
    heads_ = c.at("num_attention_heads");
    inter_ = uint32_t(double(hidden_) * c.value("mlp_ratio", 4.0));
    patch_ = c.value("native_patch_size", 20);
    merge_ = c.value("adapter_merge_size", 2);
    out_hidden_ = c.at("text_hidden_size");
    proj_hidden_ = c.at("projection_hidden_dim");
    grid_side_ = c.value("posemb_grid_size", 32);
    eps_ = c.value("layer_norm_eps", 1e-5f);
    theta_ = c.value("rope_theta", 10000.f);
    image_token_ = c.value("image_token_id", 200054);
    if (c.value("projector_bias", false)) throw std::runtime_error("inkling vision: projector biases not supported");
}
InklingVision::~InklingVision() = default;

Recipe& InklingVision::recipe(uint32_t NP) {
    const std::string key = "inkvis_N" + std::to_string(NP);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t E = hidden_, K = uint64_t(patch_) * patch_ * 3, NH = heads_, HD = E / NH, F = inter_, G = grid_side_,
                   M = uint64_t(merge_) * merge_, S = uint64_t(NP) + 1, PH = proj_hidden_, O = out_hidden_;
    GraphBuilder g(dev_);
    Ops o(g);
    std::map<std::string, TensorId> ids;
    auto in = [&](const std::string& name, DType dt, Shape shape) {
        if (auto it = ids.find(name); it != ids.end()) return it->second;
        return ids[name] = g.input(name, dt, std::move(shape));
    };
    auto W = [&](const std::string& name, Shape s) { return in(name, DType::BF16, std::move(s)); };
    auto linear = [&](TensorId x, const std::string& w, uint64_t n, uint64_t k, const std::string& nm) {
        return o.add(o.linear_bf16(x, W(w + ".weight", {n, k}), nm), W(w + ".bias", {n}), nm + "_b");
    };
    // patch embedding (no bias) + the resampled position grid; the class token in front
    TensorId hp = o.linear_bf16(in("x", DType::BF16, {NP, K}), W("patch_embed", {E, K}), "pe");
    {
        const auto corners = o.cast(o.gather_rows(W("pos_embed", {G * G, E}), in("pidx", DType::I32, {4ull * NP}), "pg"),
                                    DType::F32, "pgf");
        const auto pos = o.reduce("sum", o.mul(o.reshape(corners, {4, NP, E}, "pg3"), in("pw", DType::F32, {4, NP, 1}), "pgw"),
                                  0, "pgs");
        hp = o.add(hp, o.cast(o.reshape(pos, {NP, E}, "pos2"), DType::BF16, "posb"), "h0");
    }
    TensorId h = o.concat({o.reshape(W("cls", {E}), {1, E}, "cls2"), hp}, 0, "hcat");
    h = o.layer_norm(h, W("ln_pre.weight", {E}), W("ln_pre.bias", {E}), eps_, "lnpre");
    // axial RoPE on interleaved pairs: (x1, x2) -> (x1 c - x2 s, x2 c + x1 s), one angle per pair
    const auto cosv = in("cos", DType::F32, {S, 1, HD / 2, 1}), sinv = in("sin", DType::F32, {S, 1, HD / 2, 1});
    auto rope = [&](TensorId t, const std::string& n) {
        const auto x = o.split(o.reshape(o.cast(t, DType::F32, n + "_f"), {S, NH, HD / 2, 2}, n + "_p"), 3, {1, 1}, n + "_x");
        const auto o1 = o.sub(o.mul(x[0], cosv, n + "_1c"), o.mul(x[1], sinv, n + "_2s"), n + "_o1");
        const auto o2 = o.add(o.mul(x[1], cosv, n + "_2c"), o.mul(x[0], sinv, n + "_1s"), n + "_o2");
        return o.cast(o.reshape(o.concat({o1, o2}, 3, n + "_cat"), {S, NH, HD}, n + "_r"), DType::BF16, n + "_b");
    };
    ns_Sdpa::ParamsV7 sp{};
    sp.scale = float(1.0 / std::sqrt(double(HD)));
    sp.is_causal = false;
    sp.is_inference = true;
    sp.softmax_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
    sp.wl = sp.wr = -1;
    for (uint32_t L = 0; L < depth_; ++L) {
        const std::string b = std::to_string(L) + ".", p = "b" + std::to_string(L) + ".";
        const auto n1 = o.layer_norm(h, W(b + "ln1.weight", {E}), W(b + "ln1.bias", {E}), eps_, p + "n1");
        const auto qkv = o.split(o.reshape(linear(n1, b + "qkv", 3 * E, E, p + "qkv"), {S, 3, E}, p + "qkv3"), 1, {1, 1, 1},
                                 p + "qkvs");
        auto heads4 = [&](TensorId t, const std::string& n) {  // [S, NH, HD] -> [1, NH, S, HD]
            return o.reshape(o.transpose(t, {1, 0, 2}, n + "_t"), {1, NH, S, HD}, n + "4");
        };
        const auto q = heads4(rope(o.reshape(qkv[0], {S, NH, HD}, p + "q3"), p + "qr"), p + "q");
        const auto k = heads4(rope(o.reshape(qkv[1], {S, NH, HD}, p + "k3"), p + "kr"), p + "k");
        const auto v = heads4(o.reshape(qkv[2], {S, NH, HD}, p + "v3"), p + "v");
        const auto att = o.temp(p + "att", DType::BF16, {1, NH, S, HD});
        std::vector<TensorId> ins = {q, k, v, kNoTensor, kNoTensor};
        for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
        g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(&att, 1), &sp, sizeof(sp));
        const auto a2 = o.reshape(o.transpose(o.reshape(att, {NH, S, HD}, p + "a3"), {1, 0, 2}, p + "at"), {S, E}, p + "a2");
        h = o.add(h, o.mul(linear(a2, b + "out", E, E, p + "proj"), W(b + "ls1", {E}), p + "ls1"), p + "h1");
        const auto n2 = o.layer_norm(h, W(b + "ln2.weight", {E}), W(b + "ln2.bias", {E}), eps_, p + "n2");
        const auto f1 = o.gelu(linear(n2, b + "fc1", F, E, p + "fc1"), false, p + "act");
        h = o.add(h, o.mul(linear(f1, b + "fc2", E, F, p + "fc2"), W(b + "ls2", {E}), p + "ls2"), p + "h2");
    }
    // drop the class token, merge each cell's patches, project
    const auto pst = o.split(h, 0, {1, NP}, "dropcls")[1];
    const auto m1 = o.gelu(o.linear_bf16(o.reshape(pst, {NP / M, M * E}, "cells"), W("proj_in", {PH, M * E}), "pi"), false,
                           "pact");
    const auto y = o.linear_bf16(m1, W("proj_out", {O, PH}), "po");
    const auto out = g.input("out", DType::BF16, {NP / M, O});
    g.op("identity", {y}, {out});
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

void InklingVision::encode(const VisionImage& img, const DeviceBuffer& out, uint64_t offset) {
    const uint64_t NP = uint64_t(img.grid_h) * img.grid_w, K = uint64_t(patch_) * patch_ * 3, S = NP + 1,
                   HD = hidden_ / heads_, R = HD / 2;
    if (!NP || img.grid_h % merge_ || img.grid_w % merge_ || img.patches.size() != NP * K)
        throw std::invalid_argument("inkling vision: bad image grid / patches");
    // device inputs: x bf16 [NP, K] | pidx i32 [4, NP] | pw f32 [4, NP] | cos f32 [S, R] | sin f32 [S, R]
    const uint64_t ox = 0, oi = align128(ox + NP * K * 2), ow = align128(oi + 4 * NP * 4), oc = align128(ow + 4 * NP * 4),
                   os = align128(oc + S * R * 4), total = align128(os + S * R * 4);
    if (total > staging_.size()) throw std::invalid_argument("inkling vision: image too large");
    if (total > in_bytes_) {
        in_ = DeviceBuffer(dev_, total);
        in_bytes_ = total;
    }
    auto* base = staging_.as<uint8_t>();
    auto* x = reinterpret_cast<uint16_t*>(base + ox);
    for (uint64_t i = 0; i < NP * K; ++i) x[i] = to_bf16(img.patches[i]);
    auto* pidx = reinterpret_cast<int32_t*>(base + oi);
    auto* pw = reinterpret_cast<float*>(base + ow);
    auto* cs = reinterpret_cast<float*>(base + oc);
    auto* sn = reinterpret_cast<float*>(base + os);
    // axial frequencies over HD/2 (one per pair, half for the column, half for the row)
    const uint64_t Q = R / 2;
    std::vector<float> inv(Q);
    for (uint64_t i = 0; i < Q; ++i) inv[i] = 1.0f / std::pow(theta_, float(2 * i) / float(R));
    auto angles = [&](uint64_t s, float row, float col) {
        for (uint64_t i = 0; i < R; ++i) {
            const float f = i < Q ? col * inv[i] : row * inv[i - Q];
            cs[s * R + i] = std::cos(f);
            sn[s * R + i] = std::sin(f);
        }
    };
    angles(0, 0.f, 0.f);  // the class token at (0, 0)
    const uint32_t cols = img.grid_w / merge_, side = grid_side_;
    for (uint64_t p = 0; p < NP; ++p) {
        const uint64_t blk = p / (merge_ * merge_), in = p % (merge_ * merge_);
        const uint32_t row = uint32_t((blk / cols) * merge_ + in / merge_), col = uint32_t((blk % cols) * merge_ + in % merge_);
        int th[2], tw[2];
        float wh[2], ww[2];
        taps(row, img.grid_h, side, th, wh);
        taps(col, img.grid_w, side, tw, ww);
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b) {
                pidx[uint64_t(a * 2 + b) * NP + p] = int32_t(th[a] * int(side) + tw[b]);
                pw[uint64_t(a * 2 + b) * NP + p] = wh[a] * ww[b];
            }
        angles(p + 1, float(row + 1), float(col + 1));  // patches 1-indexed
    }
    stream_.upload(staging_, 0, in_, 0, total);
    auto& r = recipe(uint32_t(NP));
    std::vector<Binding> b = {{"x", &in_, ox}, {"pidx", &in_, oi}, {"pw", &in_, ow}, {"cos", &in_, oc}, {"sin", &in_, os},
                              {"out", &out, offset}};
    for (const auto& t : r.tensors())
        if (w_.has(t.name)) b.push_back(w_.bind(t.name, t.name));
    Event e0(dev_), e1(dev_);
    e0.record(stream_);
    r.enqueue(stream_, b);
    e1.record(stream_);
    stream_.synchronize();
    last_ms_ = Event::elapsed_ms(e0, e1);
}

}  // namespace g2e
