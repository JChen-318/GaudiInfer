#include "g2e/vision.hpp"

#include "g2e/numeric.hpp"
#include "g2e/ops.hpp"

#include <perf_lib_layer_params.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace g2e {
namespace {

uint64_t align128(uint64_t x) { return (x + 127) & ~uint64_t(127); }

struct Inputs {
    GraphBuilder& g;
    std::map<std::string, TensorId> ids;
    TensorId operator()(const std::string& name, DType dt, Shape shape) {
        if (auto it = ids.find(name); it != ids.end()) return it->second;
        return ids[name] = g.input(name, dt, std::move(shape));
    }
};

// processor order: merge blocks row-major, the patches of a block row-major -> (row, col) of patch p
void patch_rc(uint32_t gh, uint32_t gw, uint32_t merge, uint32_t p, uint32_t& row, uint32_t& col) {
    (void)gh;
    const uint32_t per_block = merge * merge, blocks_w = gw / merge;
    const uint32_t blk = p / per_block, in = p % per_block;
    row = (blk / blocks_w) * merge + in / merge;
    col = (blk % blocks_w) * merge + in % merge;
}

}  // namespace

Vision::Vision(const Device& device, const Stream& stream, const WeightPack& pack)
    : dev_(device), stream_(stream), w_(pack), staging_(device, 64u << 20) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& c = j.at("config");
    depth_ = c.at("depth");
    hidden_ = c.at("hidden_size");
    heads_ = c.at("num_heads");
    inter_ = c.at("intermediate_size");
    patch_ = c.at("patch_size");
    merge_ = c.at("spatial_merge_size");
    out_hidden_ = c.at("out_hidden_size");
    grid_side_ = uint32_t(std::lround(std::sqrt(double(uint32_t(c.at("num_position_embeddings"))))));
    image_token_ = c.at("image_token_id");
    vision_start_ = c.at("vision_start_token_id");
    vision_end_ = c.at("vision_end_token_id");
    const auto& ip = c.at("image_processor");
    min_pixels_ = ip.at("size").at("shortest_edge");
    max_pixels_ = ip.at("size").at("longest_edge");
    if (c.at("temporal_patch_size") != 2 || c.at("in_channels") != 3) throw std::runtime_error("vision: unexpected patch shape");
}
Vision::~Vision() = default;

Recipe& Vision::recipe(uint32_t N) {
    const std::string key = "vision_N" + std::to_string(N);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t E = hidden_, K = 3ull * 2 * patch_ * patch_, HD = E / heads_, NH = heads_, F = inter_, S = grid_side_,
                   M = uint64_t(merge_) * merge_, O = out_hidden_;
    GraphBuilder g(dev_);
    Ops o(g);
    Inputs in{g, {}};
    auto W = [&](const std::string& name, Shape s) { return in(name, DType::BF16, std::move(s)); };
    auto linear = [&](TensorId x, const std::string& w, uint64_t n, uint64_t k, const std::string& nm) {
        return o.add(o.linear_bf16(x, W(w + ".weight", {n, k}), nm), W(w + ".bias", {n}), nm + "_b");
    };
    // patch embedding (a linear map of the flattened patch) + bilinearly interpolated position embedding
    TensorId h = linear(in("x", DType::BF16, {N, K}), "patch_embed.proj", E, K, "pe");
    {
        const auto corners = o.cast(o.gather_rows(W("pos_embed.weight", {S * S, E}), in("pidx", DType::I32, {4 * N}), "pg"),
                                    DType::F32, "pgf");
        const auto pos = o.reduce("sum", o.mul(o.reshape(corners, {4, N, E}, "pg3"), in("pw", DType::F32, {4, N, 1}), "pgw"),
                                  0, "pgs");
        h = o.add(h, o.cast(o.reshape(pos, {N, E}, "pos2"), DType::BF16, "posb"), "h0");
    }
    const auto cosv = in("cos", DType::F32, {N, 1, HD}), sinv = in("sin", DType::F32, {N, 1, HD});
    const auto cs = o.split(cosv, 2, {HD / 2, HD / 2}, "cs"), ss = o.split(sinv, 2, {HD / 2, HD / 2}, "ss");
    auto rope = [&](TensorId t, const std::string& n) {  // rotate-half over the whole head, FP32, back to BF16
        const auto x = o.split(o.cast(t, DType::F32, n + "_f"), 2, {HD / 2, HD / 2}, n + "_x");
        const auto o1 = o.sub(o.mul(x[0], cs[0], n + "_1c"), o.mul(x[1], ss[0], n + "_2s"), n + "_o1");
        const auto o2 = o.add(o.mul(x[1], cs[1], n + "_2c"), o.mul(x[0], ss[1], n + "_1s"), n + "_o2");
        return o.cast(o.concat({o1, o2}, 2, n + "_cat"), DType::BF16, n + "_b");
    };
    ns_Sdpa::ParamsV7 sp{};
    sp.scale = float(1.0 / std::sqrt(double(HD)));
    sp.is_causal = false;
    sp.is_inference = true;
    sp.softmax_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
    sp.wl = sp.wr = -1;
    for (uint32_t L = 0; L < depth_; ++L) {
        const std::string b = "blocks." + std::to_string(L) + ".", p = "b" + std::to_string(L) + ".";
        const auto n1 = o.layer_norm(h, W(b + "norm1.weight", {E}), W(b + "norm1.bias", {E}), 1e-6f, p + "n1");
        const auto qkv = o.split(o.reshape(linear(n1, b + "attn.qkv", 3 * E, E, p + "qkv"), {N, 3, NH * HD}, p + "qkv3"), 1,
                                 {1, 1, 1}, p + "qkvs");
        auto heads4 = [&](TensorId t, const std::string& n) {  // [N, NH, HD] -> [1, NH, N, HD]
            return o.reshape(o.transpose(t, {1, 0, 2}, n + "_t"), {1, NH, N, HD}, n + "4");
        };
        const auto q = heads4(rope(o.reshape(qkv[0], {N, NH, HD}, p + "q3"), p + "qr"), p + "q");
        const auto k = heads4(rope(o.reshape(qkv[1], {N, NH, HD}, p + "k3"), p + "kr"), p + "k");
        const auto v = heads4(o.reshape(qkv[2], {N, NH, HD}, p + "v3"), p + "v");
        const auto att = o.temp(p + "att", DType::BF16, {1, NH, N, HD});
        std::vector<TensorId> ins = {q, k, v, kNoTensor, kNoTensor};
        for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
        g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(&att, 1), &sp, sizeof(sp));
        const auto a2 = o.reshape(o.transpose(o.reshape(att, {NH, N, HD}, p + "a3"), {1, 0, 2}, p + "at"), {N, E}, p + "a2");
        h = o.add(h, linear(a2, b + "attn.proj", E, E, p + "proj"), p + "h1");
        const auto n2 = o.layer_norm(h, W(b + "norm2.weight", {E}), W(b + "norm2.bias", {E}), 1e-6f, p + "n2");
        const auto f1 = o.gelu(linear(n2, b + "mlp.linear_fc1", F, E, p + "fc1"), true, p + "act");
        h = o.add(h, linear(f1, b + "mlp.linear_fc2", E, F, p + "fc2"), p + "h2");
    }
    const auto mn = o.reshape(o.layer_norm(h, W("merger.norm.weight", {E}), W("merger.norm.bias", {E}), 1e-6f, "mn"),
                              {N / M, M * E}, "mn2");
    const auto m1 = o.gelu(linear(mn, "merger.linear_fc1", M * E, M * E, "mf1"), false, "mact");
    const auto y = linear(m1, "merger.linear_fc2", O, M * E, "mf2");
    const auto out = g.input("out", DType::BF16, {N / M, O});
    g.op("identity", {y}, {out});
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

void Vision::encode(const VisionImage& img, const DeviceBuffer& out, uint64_t offset) {
    const uint64_t N = uint64_t(img.grid_h) * img.grid_w, K = 3ull * 2 * patch_ * patch_, HD = hidden_ / heads_;
    if (!N || img.grid_h % merge_ || img.grid_w % merge_ || img.patches.size() != N * K)
        throw std::invalid_argument("vision: bad image grid / patches");
    // device inputs: x bf16 [N, K] | pidx i32 [4N] | pw f32 [4N] | cos f32 [N, HD] | sin f32 [N, HD]
    const uint64_t ox = 0, oi = align128(ox + N * K * 2), ow = align128(oi + 4 * N * 4), oc = align128(ow + 4 * N * 4),
                   os = align128(oc + N * HD * 4), total = align128(os + N * HD * 4);
    if (total > staging_.size()) throw std::invalid_argument("vision: image too large");
    if (total > in_bytes_) {
        in_ = DeviceBuffer(dev_, total);
        in_bytes_ = total;
    }
    auto* base = staging_.as<uint8_t>();
    auto* x = reinterpret_cast<uint16_t*>(base + ox);
    for (uint64_t i = 0; i < N * K; ++i) x[i] = to_bf16(img.patches[i]);
    auto* pidx = reinterpret_cast<int32_t*>(base + oi);
    auto* pw = reinterpret_cast<float*>(base + ow);
    auto* cs = reinterpret_cast<float*>(base + oc);
    auto* sn = reinterpret_cast<float*>(base + os);
    // bilinear corners on the learned grid (HF get_vision_bilinear_indices_and_weights: linspace(0, side-1, n))
    const uint32_t side = grid_side_;
    auto grid_coord = [&](uint32_t i, uint32_t n) {
        if (n == 1) return 0.f;
        const float step = float(side - 1) / float(n - 1);
        return i < n / 2 ? float(i) * step : float(side - 1) - float(n - 1 - i) * step;
    };
    // vision rotary: inv_freq over HD/2 dims, (row, col) halves, duplicated (cat(rpe, rpe))
    const uint64_t R = HD / 2, Q = R / 2;
    std::vector<float> inv(Q);
    for (uint64_t i = 0; i < Q; ++i) inv[i] = float(1.0 / std::pow(10000.0, double(2 * i) / double(R)));
    for (uint32_t p = 0; p < N; ++p) {
        uint32_t row, col;
        patch_rc(img.grid_h, img.grid_w, merge_, p, row, col);
        const float hg = grid_coord(row, img.grid_h), wg = grid_coord(col, img.grid_w);
        const int hf = int(hg), wf = int(wg);
        const int hc = std::min<int>(hf + 1, int(side) - 1), wc = std::min<int>(wf + 1, int(side) - 1);
        const float fh = hg - float(hf), fw = wg - float(wf);
        const int32_t id[4] = {int32_t(hf * side + wf), int32_t(hf * side + wc), int32_t(hc * side + wf), int32_t(hc * side + wc)};
        const float wt[4] = {(1 - fh) * (1 - fw), (1 - fh) * fw, fh * (1 - fw), fh * fw};
        for (int c = 0; c < 4; ++c) {
            pidx[uint64_t(c) * N + p] = id[c];
            pw[uint64_t(c) * N + p] = wt[c];
        }
        for (uint64_t i = 0; i < R; ++i) {
            const float f = i < Q ? float(row) * inv[i] : float(col) * inv[i - Q];
            for (uint64_t half = 0; half < 2; ++half) {
                cs[p * HD + half * R + i] = std::cos(f);
                sn[p * HD + half * R + i] = std::sin(f);
            }
        }
    }
    stream_.upload(staging_, 0, in_, 0, total);
    auto& r = recipe(uint32_t(N));
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
