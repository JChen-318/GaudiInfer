#include "g2e/h3_text_encoder.hpp"
#include "g2e/numeric.hpp"

#include <nlohmann/json.hpp>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace g2e {

namespace {

void set_out(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const DType dt = d.dtype;
    const auto sh = d.shape;
    const auto out = g.input(name, dt, sh);
    g.op("identity", {src}, {out});
}

// Half-split RoPE over full head_dim (128): x [S, NH, 128] BF16, cos/sin [S, 64] BF16.
TensorId apply_full_rope(Ops& o, TensorId x, TensorId cos_tab, TensorId sin_tab, const std::string& name) {
    const auto sh = o.shape(x);
    const uint64_t S = sh[0], half = sh[2] / 2;
    const auto parts = o.split(x, 2, {half, half}, name + "_sp");
    const auto c = o.reshape(cos_tab, {S, 1, half}, name + "_c");
    const auto s = o.reshape(sin_tab, {S, 1, half}, name + "_s");
    const auto r0 = o.sub(o.mul(parts[0], c, name + "_p0c"), o.mul(parts[1], s, name + "_p1s"), name + "_r0");
    const auto r1 = o.add(o.mul(parts[1], c, name + "_p1c"), o.mul(parts[0], s, name + "_p0s"), name + "_r1");
    return o.concat({r0, r1}, 2, name);
}

}  // namespace

H3TextEncoderConfig H3TextEncoderConfig::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& tc = j.at("config").at("text_config");
    H3TextEncoderConfig c;
    c.num_layers = j.at("layers").get<uint32_t>();
    c.hidden_size = tc.at("hidden_size");
    c.intermediate_size = tc.at("intermediate_size");
    c.num_heads = tc.at("num_attention_heads");
    c.num_kv_heads = tc.at("num_key_value_heads");
    c.head_dim = tc.value("head_dim", c.hidden_size / c.num_heads);
    c.vocab_size = tc.at("vocab_size");
    c.rms_norm_eps = tc.value("rms_norm_eps", 1e-6f);
    c.rope_theta = tc.value("rope_theta", 5000000.0);
    if (tc.contains("rope_scaling") && tc["rope_scaling"].contains("mrope_section")) {
        const auto& ms = tc["rope_scaling"]["mrope_section"];
        for (int i = 0; i < 3; ++i) c.mrope_section[i] = ms.at(i);
    }
    c.blocks_fp8 = (j.value("blocks_dtype", "f8") == "f8");
    c.has_vision = j.value("has_vision", false);
    if (j.at("config").contains("image_token_id")) c.image_token_id = j.at("config").at("image_token_id");
    if (j.at("config").contains("vision_start_token_id")) c.vision_start_token_id = j.at("config").at("vision_start_token_id");
    if (j.at("config").contains("vision_end_token_id")) c.vision_end_token_id = j.at("config").at("vision_end_token_id");
    if (j.at("config").contains("vision_config")) {
        const auto& vc = j.at("config").at("vision_config");
        c.vis_depth = vc.value("depth", 27u);
        c.vis_hidden = vc.value("hidden_size", 1152u);
        c.vis_heads = vc.value("num_heads", 16u);
        c.vis_inter = vc.value("intermediate_size", 4304u);
        c.vis_patch = vc.value("patch_size", 16u);
        c.vis_merge = vc.value("spatial_merge_size", 2u);
        c.vis_grid_side = uint32_t(std::lround(std::sqrt(double(vc.value("num_position_embeddings", 2304u)))));
        if (vc.contains("deepstack_visual_indexes")) {
            c.deepstack_indexes = vc.at("deepstack_visual_indexes").get<std::vector<uint32_t>>();
        }
    }
    return c;
}

H3TextEncoder::H3TextEncoder(const Device& device, const Stream& stream, std::string pack_path)
    : dev_(device), stream_(stream), pack_path_(std::move(pack_path)) {
    pack_ = std::make_unique<WeightPack>(pack_path_);
    cfg_ = H3TextEncoderConfig::from_pack(*pack_);
}

H3TextEncoder::~H3TextEncoder() = default;

UploadStats H3TextEncoder::load_to_hbm() {
    if (loaded_) return {};
    if (!pack_) pack_ = std::make_unique<WeightPack>(pack_path_);
    const auto st = pack_->upload(dev_, stream_);
    loaded_ = true;
    std::printf("[H3TextEnc] loaded %s (%u layers): %.2f GiB in %.2f s (%.1f GiB/s)\n",
                pack_path_.c_str(), cfg_.num_layers, st.gib, st.seconds, st.gib_per_s());
    return st;
}

void H3TextEncoder::release_hbm() {
    if (!loaded_) return;
    pack_.reset();
    loaded_ = false;
    std::printf("[H3TextEnc] released HBM weight arena\n");
}

Recipe& H3TextEncoder::get_embed_recipe(uint32_t seq_len) {
    if (auto it = r_embed_.find(seq_len); it != r_embed_.end()) return *it->second;
    GraphBuilder g(dev_);
    Ops o(g, "emb_");
    const auto ids = g.input("ids", DType::I32, {seq_len});
    const auto table = g.input("embed", DType::BF16, {cfg_.vocab_size, cfg_.hidden_size});
    set_out(g, "out", o.gather_rows(table, ids, "g"));
    auto r = g.compile("h3_q3vl_embed_" + std::to_string(seq_len));
    auto& ref = *r;
    r_embed_.emplace(seq_len, std::move(r));
    return ref;
}

Recipe& H3TextEncoder::get_layer_recipe(uint32_t seq_len) {
    if (auto it = r_layer_.find(seq_len); it != r_layer_.end()) return *it->second;
    const uint64_t S = seq_len, H = cfg_.hidden_size, F = cfg_.intermediate_size;
    const uint64_t NH = cfg_.num_heads, NKV = cfg_.num_kv_heads, HD = cfg_.head_dim, G = NH / NKV;
    const uint64_t QD = NH * HD, KVD = NKV * HD;

    GraphBuilder g(dev_);
    Ops o(g, "ql_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto cos_tab = g.input("cos", DType::BF16, {S, HD / 2});
    const auto sin_tab = g.input("sin", DType::BF16, {S, HD / 2});
    const auto in_norm = g.input("in_norm", DType::BF16, {H});
    const auto post_norm = g.input("post_norm", DType::BF16, {H});
    const auto q_norm = g.input("q_norm", DType::BF16, {HD});
    const auto k_norm = g.input("k_norm", DType::BF16, {HD});
    const DType wdt = cfg_.blocks_fp8 ? DType::F8 : DType::BF16;
    const auto qkv_w = g.input("qkv_w", wdt, {QD + 2 * KVD, H});
    const auto qkv_s = cfg_.blocks_fp8 ? g.input("qkv_s", DType::F32, {QD + 2 * KVD}) : kNoTensor;
    const auto o_w = g.input("o_w", wdt, {H, QD});
    const auto o_s = cfg_.blocks_fp8 ? g.input("o_s", DType::F32, {H}) : kNoTensor;
    const auto gu_w = g.input("gu_w", wdt, {2 * F, H});
    const auto gu_s = cfg_.blocks_fp8 ? g.input("gu_s", DType::F32, {2 * F}) : kNoTensor;
    const auto dn_w = g.input("dn_w", wdt, {H, F});
    const auto dn_s = cfg_.blocks_fp8 ? g.input("dn_s", DType::F32, {H}) : kNoTensor;

    auto lin = [&](TensorId a, TensorId w, TensorId ws, const std::string& nm) {
        return cfg_.blocks_fp8 ? o.fp8_linear(a, w, ws, nm) : o.linear_bf16(a, w, nm);
    };

    // 1. Input RMSNorm -> QKV projection
    const auto n1 = o.rms_norm(x, in_norm, cfg_.rms_norm_eps, false, "n1");
    const auto qkv = lin(n1, qkv_w, qkv_s, "qkv");
    const auto sp = o.split(qkv, 1, {QD, KVD, KVD}, "sp");
    auto q = o.rms_norm(o.reshape(sp[0], {S, NH, HD}, "qr"), q_norm, cfg_.rms_norm_eps, false, "qn");
    auto k = o.rms_norm(o.reshape(sp[1], {S, NKV, HD}, "kr"), k_norm, cfg_.rms_norm_eps, false, "kn");
    auto v = o.reshape(sp[2], {S, NKV, HD}, "vr");

    q = apply_full_rope(o, q, cos_tab, sin_tab, "qrope");
    k = apply_full_rope(o, k, cos_tab, sin_tab, "krope");

    // 2. Causal GQA SDPA (4D expanded KV [1, NH, S, HD] matching HF/vLLM-Omni Qwen3-VL)
    const auto qt = o.reshape(o.transpose(q, {1, 0, 2}, "qt"), {1, NH, S, HD}, "q4");
    auto repeat_kv = [&](TensorId kv, const std::string& rname) {
        // kv: [S, NKV, HD] -> [NKV, 1, S, HD] -> concat G times -> [1, NH, S, HD]
        const auto k4 = o.reshape(o.transpose(kv, {1, 0, 2}, rname + "_t"), {NKV, 1, S, HD}, rname + "_4");
        std::vector<TensorId> reps(G, k4);
        return o.reshape(o.concat(reps, 1, rname + "_rep"), {1, NH, S, HD}, rname);
    };
    const auto kt = repeat_kv(k, "k4");
    const auto vt = repeat_kv(v, "v4");
    const auto att = o.temp("att", DType::BF16, {1, NH, S, HD});
    ns_Sdpa::ParamsV7 p{};
    p.scale = float(1.0 / std::sqrt(double(HD)));
    p.is_causal = true;
    p.is_inference = true;
    p.softmax_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
    p.wl = p.wr = -1;
    std::vector<TensorId> ins = {qt, kt, vt, kNoTensor, kNoTensor};
    for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
    g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(&att, 1), &p, sizeof(p));

    const auto ctx = o.reshape(o.transpose(o.reshape(att, {NH, S, HD}, "a3"), {1, 0, 2}, "at"), {S, QD}, "af");
    const auto aout = lin(ctx, o_w, o_s, "oproj");
    const auto x1 = o.add(x, aout, "res1");

    // 3. Post-RMSNorm -> SwiGLU MLP
    const auto n2 = o.rms_norm(x1, post_norm, cfg_.rms_norm_eps, false, "n2");
    const auto gu = lin(n2, gu_w, gu_s, "gu");
    const auto gup = o.split(gu, 1, {F, F}, "gup");
    const auto act = o.mul(o.silu(gup[0], "gs"), gup[1], "act");
    const auto mout = lin(act, dn_w, dn_s, "dn");
    set_out(g, "out", o.add(x1, mout, "res2"));

    auto r = g.compile("h3_q3vl_layer_v2_" + std::string(cfg_.blocks_fp8 ? "f8_" : "bf16_") + std::to_string(S));
    auto& ref = *r;
    r_layer_.emplace(seq_len, std::move(r));
    return ref;
}

std::vector<uint16_t> H3TextEncoder::encode_text_ids(const std::vector<int32_t>& token_ids) {
    if (token_ids.empty()) throw std::invalid_argument("H3TextEncoder: empty token_ids");
    if (!loaded_) load_to_hbm();
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t S = uint32_t(token_ids.size());
    const uint32_t H = cfg_.hidden_size, R2 = cfg_.head_dim / 2;

    DeviceBuffer d_ids(dev_, uint64_t(S) * 4);
    DeviceBuffer d_cos(dev_, uint64_t(S) * R2 * 2);
    DeviceBuffer d_sin(dev_, uint64_t(S) * R2 * 2);
    DeviceBuffer d_h[2] = {DeviceBuffer(dev_, uint64_t(S) * H * 2), DeviceBuffer(dev_, uint64_t(S) * H * 2)};

    // Build 1D text RoPE table (for text-only tokens all 3 M-RoPE axes equal position index p)
    std::vector<uint16_t> h_cos(size_t(S) * R2), h_sin(size_t(S) * R2);
    for (uint32_t i = 0; i < R2; ++i) {
        const double inv = 1.0 / std::pow(cfg_.rope_theta, double(2 * i) / double(cfg_.head_dim));
        for (uint32_t p = 0; p < S; ++p) {
            const float a = float(double(p) * inv);
            h_cos[size_t(p) * R2 + i] = to_bf16(std::cos(a));
            h_sin[size_t(p) * R2 + i] = to_bf16(std::sin(a));
        }
    }

    {
        const size_t ib = size_t(S) * 4, rb = size_t(S) * R2 * 2;
        HostBuffer stg(dev_, ib + 2 * rb);
        std::memcpy(stg.data(), token_ids.data(), ib);
        std::memcpy(static_cast<uint8_t*>(stg.data()) + ib, h_cos.data(), rb);
        std::memcpy(static_cast<uint8_t*>(stg.data()) + ib + rb, h_sin.data(), rb);
        stream_.upload(stg, 0, d_ids, 0, ib);
        stream_.upload(stg, ib, d_cos, 0, rb);
        stream_.upload(stg, ib + rb, d_sin, 0, rb);
        stream_.synchronize();
    }

    // Embedding lookup
    {
        auto& re = get_embed_recipe(S);
        std::vector<Binding> b = {
            {"ids", &d_ids, 0},
            pack_->bind("embed", "embed"),
            {"out", &d_h[0], 0},
        };
        re.enqueue(stream_, b);
    }

    // 50 transformer layers (returns unnormalized hidden_states[50])
    int hp = 0;
    auto& rl = get_layer_recipe(S);
    for (uint32_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = "L" + std::to_string(i) + ".";
        std::vector<Binding> b = {
            {"x", &d_h[hp], 0},
            {"cos", &d_cos, 0},
            {"sin", &d_sin, 0},
            pack_->bind("in_norm", p + "in_norm"),
            pack_->bind("post_norm", p + "post_norm"),
            pack_->bind("q_norm", p + "q_norm"),
            pack_->bind("k_norm", p + "k_norm"),
            pack_->bind("qkv_w", p + "qkv.w"),
            pack_->bind("o_w", p + "o.w"),
            pack_->bind("gu_w", p + "gate_up.w"),
            pack_->bind("dn_w", p + "down.w"),
            {"out", &d_h[1 - hp], 0},
        };
        if (cfg_.blocks_fp8) {
            b.push_back(pack_->bind("qkv_s", p + "qkv.s"));
            b.push_back(pack_->bind("o_s", p + "o.s"));
            b.push_back(pack_->bind("gu_s", p + "gate_up.s"));
            b.push_back(pack_->bind("dn_s", p + "down.s"));
        }
        rl.enqueue(stream_, b);
        hp = 1 - hp;
    }

    HostBuffer stg_out(dev_, uint64_t(S) * H * 2);
    stream_.download(d_h[hp], 0, stg_out, 0, stg_out.size());
    stream_.synchronize();
    std::vector<uint16_t> out(size_t(S) * H);
    std::memcpy(out.data(), stg_out.data(), out.size() * 2);

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[H3TextEnc] encoded %u tokens (%u layers) in %.1f ms\n", S, cfg_.num_layers, ms);
    return out;
}

Recipe& H3TextEncoder::get_add_recipe(uint32_t seq_len) {
    if (auto it = r_add_.find(seq_len); it != r_add_.end()) return *it->second;
    GraphBuilder g(dev_);
    Ops o(g, "qadd_");
    const auto a = g.input("a", DType::BF16, {seq_len, cfg_.hidden_size});
    const auto b = g.input("b", DType::BF16, {seq_len, cfg_.hidden_size});
    set_out(g, "out", o.add(a, b, "sum"));
    auto r = g.compile("h3_q3vl_add_" + std::to_string(seq_len));
    auto& ref = *r;
    r_add_.emplace(seq_len, std::move(r));
    return ref;
}

Recipe& H3TextEncoder::get_vis_recipe(uint32_t n_patches) {
    if (auto it = r_vis_.find(n_patches); it != r_vis_.end()) return *it->second;
    const uint64_t N = n_patches, E = cfg_.vis_hidden, K = 3ull * 2 * cfg_.vis_patch * cfg_.vis_patch;
    const uint64_t NH = cfg_.vis_heads, HD = E / NH, F = cfg_.vis_inter, S = cfg_.vis_grid_side;
    const uint64_t M = uint64_t(cfg_.vis_merge) * cfg_.vis_merge, O = cfg_.hidden_size;

    GraphBuilder g(dev_);
    Ops o(g, "qv_");
    std::map<std::string, TensorId> ins_map;
    auto in = [&](const std::string& nm, DType dt, Shape sh) {
        if (auto it = ins_map.find(nm); it != ins_map.end()) return it->second;
        return ins_map[nm] = g.input(nm, dt, std::move(sh));
    };
    auto W = [&](const std::string& nm, Shape sh) { return in(nm, DType::BF16, std::move(sh)); };
    auto linear = [&](TensorId x, const std::string& w, uint64_t n, uint64_t k, const std::string& nm) {
        return o.add(o.linear_bf16(x, W(w + ".weight", {n, k}), nm), W(w + ".bias", {n}), nm + "_b");
    };

    TensorId h = linear(in("x", DType::BF16, {N, K}), "vis.patch_embed.proj", E, K, "pe");
    {
        const auto corners = o.cast(
            o.gather_rows(W("vis.pos_embed.weight", {S * S, E}), in("pidx", DType::I32, {4 * N}), "pg"),
            DType::F32, "pgf");
        const auto pos = o.reduce(
            "sum", o.mul(o.reshape(corners, {4, N, E}, "pg3"), in("pw", DType::F32, {4, N, 1}), "pgw"),
            0, "pgs");
        h = o.add(h, o.cast(o.reshape(pos, {N, E}, "pos2"), DType::BF16, "posb"), "h0");
    }
    const auto cosv = in("cos", DType::F32, {N, 1, HD}), sinv = in("sin", DType::F32, {N, 1, HD});
    const auto cs = o.split(cosv, 2, {HD / 2, HD / 2}, "cs"), ss = o.split(sinv, 2, {HD / 2, HD / 2}, "ss");
    auto rope = [&](TensorId t, const std::string& n) {
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

    std::vector<TensorId> ds_outs(cfg_.deepstack_indexes.size(), kNoTensor);
    for (uint32_t L = 0; L < cfg_.vis_depth; ++L) {
        const std::string b = "vis.blocks." + std::to_string(L) + ".";
        const std::string p = "b" + std::to_string(L) + "_";
        const auto n1 = o.layer_norm(h, W(b + "norm1.weight", {E}), W(b + "norm1.bias", {E}), 1e-6f, p + "n1");
        const auto qkv = o.split(o.reshape(linear(n1, b + "attn.qkv", 3 * E, E, p + "qkv"), {N, 3, NH * HD}, p + "qkv3"),
                                 1, {1, 1, 1}, p + "qkvs");
        auto heads4 = [&](TensorId t, const std::string& n) {
            return o.reshape(o.transpose(t, {1, 0, 2}, n + "_t"), {1, NH, N, HD}, n + "4");
        };
        const auto q = heads4(rope(o.reshape(qkv[0], {N, NH, HD}, p + "q3"), p + "qr"), p + "q");
        const auto k = heads4(rope(o.reshape(qkv[1], {N, NH, HD}, p + "k3"), p + "kr"), p + "k");
        const auto v = heads4(o.reshape(qkv[2], {N, NH, HD}, p + "v3"), p + "v");
        const auto att = o.temp(p + "att", DType::BF16, {1, NH, N, HD});
        std::vector<TensorId> sd_in = {q, k, v, kNoTensor, kNoTensor};
        for (int i = 0; i < 7; ++i) sd_in.push_back(kNoTensor);
        g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(sd_in), std::span<const TensorId>(&att, 1), &sp, sizeof(sp));
        const auto a2 = o.reshape(o.transpose(o.reshape(att, {NH, N, HD}, p + "a3"), {1, 0, 2}, p + "at"), {N, E}, p + "a2");
        h = o.add(h, linear(a2, b + "attn.proj", E, E, p + "proj"), p + "h1");
        const auto n2 = o.layer_norm(h, W(b + "norm2.weight", {E}), W(b + "norm2.bias", {E}), 1e-6f, p + "n2");
        const auto f1 = o.gelu(linear(n2, b + "mlp.linear_fc1", F, E, p + "fc1"), true, p + "act");
        h = o.add(h, linear(f1, b + "mlp.linear_fc2", E, F, p + "fc2"), p + "h2");

        for (size_t di = 0; di < cfg_.deepstack_indexes.size(); ++di) {
            if (cfg_.deepstack_indexes[di] == L) {
                const std::string dp = "vis.deepstack_merger_list." + std::to_string(di) + ".";
                const std::string dn = "ds" + std::to_string(di) + "_";
                // DeepStack merger uses post-shuffle LayerNorm over M*E = 4608!
                const auto dr = o.reshape(h, {N / M, M * E}, dn + "r");
                const auto dln = o.layer_norm(dr, W(dp + "norm.weight", {M * E}), W(dp + "norm.bias", {M * E}), 1e-6f, dn + "ln");
                const auto df1 = o.gelu(linear(dln, dp + "linear_fc1", M * E, M * E, dn + "f1"), false, dn + "a");
                ds_outs[di] = linear(df1, dp + "linear_fc2", O, M * E, dn + "f2");
            }
        }
    }

    const auto mn = o.reshape(o.layer_norm(h, W("vis.merger.norm.weight", {E}), W("vis.merger.norm.bias", {E}), 1e-6f, "mn"),
                              {N / M, M * E}, "mn2");
    const auto m1 = o.gelu(linear(mn, "vis.merger.linear_fc1", M * E, M * E, "mf1"), false, "mact");
    set_out(g, "out", linear(m1, "vis.merger.linear_fc2", O, M * E, "mf2"));
    for (size_t di = 0; di < ds_outs.size(); ++di) {
        if (ds_outs[di] != kNoTensor) set_out(g, "ds" + std::to_string(di), ds_outs[di]);
    }

    auto r = g.compile("h3_q3vl_vis_N" + std::to_string(N));
    auto& ref = *r;
    r_vis_.emplace(n_patches, std::move(r));
    return ref;
}

std::vector<uint16_t> H3TextEncoder::run_multimodal_forward(const std::vector<int32_t>& ids,
                                                            const std::vector<uint8_t>& mm_type,
                                                            const std::vector<ImgSpan>& spans,
                                                            const std::vector<VisionImage>& images) {
    if (images.empty()) return encode_text_ids(ids);
    if (!cfg_.has_vision) {
        throw std::runtime_error("H3TextEncoder: pack does not contain vision encoder weights");
    }
    if (!loaded_) load_to_hbm();
    const auto t0 = std::chrono::steady_clock::now();

    const uint32_t S = uint32_t(ids.size());
    const uint32_t H = cfg_.hidden_size, R2 = cfg_.head_dim / 2;
    const size_t n_ds = cfg_.deepstack_indexes.size();

    // 1. Build 3D M-RoPE position_ids [3, S] (matching _get_rope_index) & interleaved cos/sin [S, 64] BF16
    std::vector<int32_t> pos_t(S), pos_h(S), pos_w(S);
    int32_t cur_pos = 0;
    size_t img_cursor = 0;
    uint32_t p = 0;
    while (p < S) {
        if (mm_type[p] == 0) {
            uint32_t q = p;
            while (q < S && mm_type[q] == 0) {
                pos_t[q] = pos_h[q] = pos_w[q] = cur_pos++;
                ++q;
            }
            p = q;
        } else {
            const auto& sp = spans[img_cursor++];
            for (uint32_t hi = 0; hi < sp.llm_h; ++hi) {
                for (uint32_t wi = 0; wi < sp.llm_w; ++wi) {
                    const uint32_t idx = sp.start + hi * sp.llm_w + wi;
                    pos_t[idx] = cur_pos;
                    pos_h[idx] = cur_pos + int32_t(hi);
                    pos_w[idx] = cur_pos + int32_t(wi);
                }
            }
            cur_pos += int32_t(std::max(sp.llm_h, sp.llm_w));
            p += sp.count;
        }
    }

    std::vector<double> inv_freq(R2);
    for (uint32_t i = 0; i < R2; ++i)
        inv_freq[i] = 1.0 / std::pow(cfg_.rope_theta, double(2 * i) / double(cfg_.head_dim));
    const uint32_t len_h = cfg_.mrope_section[1] * 3;
    const uint32_t len_w = cfg_.mrope_section[2] * 3;
    std::vector<uint16_t> h_cos(size_t(S) * R2), h_sin(size_t(S) * R2);
    for (uint32_t k = 0; k < S; ++k) {
        for (uint32_t i = 0; i < R2; ++i) {
            int32_t axis_pos = pos_t[k];
            if (i < len_h && (i % 3 == 1)) axis_pos = pos_h[k];
            else if (i < len_w && (i % 3 == 2)) axis_pos = pos_w[k];
            const float a = float(double(axis_pos) * inv_freq[i]);
            h_cos[size_t(k) * R2 + i] = to_bf16(std::cos(a));
            h_sin[size_t(k) * R2 + i] = to_bf16(std::sin(a));
        }
    }

    // 2. Gather text embeddings into d_h[0] and zero DeepStack full additive buffers d_ds[0..2]
    DeviceBuffer d_ids(dev_, uint64_t(S) * 4);
    DeviceBuffer d_cos(dev_, uint64_t(S) * R2 * 2);
    DeviceBuffer d_sin(dev_, uint64_t(S) * R2 * 2);
    DeviceBuffer d_h[2] = {DeviceBuffer(dev_, uint64_t(S) * H * 2), DeviceBuffer(dev_, uint64_t(S) * H * 2)};
    std::vector<DeviceBuffer> d_ds;
    d_ds.reserve(n_ds);
    for (size_t di = 0; di < n_ds; ++di) {
        d_ds.emplace_back(dev_, uint64_t(S) * H * 2);
        stream_.zero(d_ds.back(), 0, uint64_t(S) * H * 2);
    }
    {
        const size_t ib = size_t(S) * 4, rb = size_t(S) * R2 * 2;
        HostBuffer stg(dev_, ib + 2 * rb);
        std::memcpy(stg.data(), ids.data(), ib);
        std::memcpy(static_cast<uint8_t*>(stg.data()) + ib, h_cos.data(), rb);
        std::memcpy(static_cast<uint8_t*>(stg.data()) + ib + rb, h_sin.data(), rb);
        stream_.upload(stg, 0, d_ids, 0, ib);
        stream_.upload(stg, ib, d_cos, 0, rb);
        stream_.upload(stg, ib + rb, d_sin, 0, rb);
        stream_.synchronize();
    }
    {
        auto& re = get_embed_recipe(S);
        std::vector<Binding> b = {{"ids", &d_ids, 0}, pack_->bind("embed", "embed"), {"out", &d_h[0], 0}};
        re.enqueue(stream_, b);
    }

    // 3. Encode each VisionImage on Gaudi2 -> copy merged features into d_h[0] & DeepStack features into d_ds[0..2]
    auto align128 = [](uint64_t v) { return (v + 127ull) & ~127ull; };
    const uint64_t K_patch = 3ull * 2 * cfg_.vis_patch * cfg_.vis_patch;
    const uint64_t HD_v = cfg_.vis_hidden / cfg_.vis_heads;
    const uint64_t R_v = HD_v / 2, Q_v = R_v / 2;
    std::vector<float> vis_inv(Q_v);
    for (uint64_t i = 0; i < Q_v; ++i) vis_inv[i] = float(1.0 / std::pow(10000.0, double(2 * i) / double(R_v)));

    for (size_t idx = 0; idx < images.size(); ++idx) {
        const auto& img = images[idx];
        const auto& sp = spans[idx];
        const uint64_t N = uint64_t(img.grid_h) * img.grid_w;
        const uint64_t ox = 0, oi = align128(ox + N * K_patch * 2), ow = align128(oi + 4 * N * 4);
        const uint64_t oc = align128(ow + 4 * N * 4), os = align128(oc + N * HD_v * 4), tot = align128(os + N * HD_v * 4);
        HostBuffer h_in(dev_, tot);
        DeviceBuffer d_in(dev_, tot);
        auto* base = h_in.as<uint8_t>();
        auto* xp = reinterpret_cast<uint16_t*>(base + ox);
        for (uint64_t i = 0; i < N * K_patch; ++i) xp[i] = to_bf16(img.patches[i]);
        auto* pidx = reinterpret_cast<int32_t*>(base + oi);
        auto* pw = reinterpret_cast<float*>(base + ow);
        auto* cs = reinterpret_cast<float*>(base + oc);
        auto* sn = reinterpret_cast<float*>(base + os);

        const uint32_t side = cfg_.vis_grid_side;
        const uint32_t per_blk = cfg_.vis_merge * cfg_.vis_merge, blocks_w = img.grid_w / cfg_.vis_merge;
        auto grid_coord = [&](uint32_t i, uint32_t n) {
            if (n == 1) return 0.0f;
            const float step = float(side - 1) / float(n - 1);
            return i < n / 2 ? float(i) * step : float(side - 1) - float(n - 1 - i) * step;
        };
        for (uint32_t pi = 0; pi < N; ++pi) {
            const uint32_t blk = pi / per_blk, in_b = pi % per_blk;
            const uint32_t row = (blk / blocks_w) * cfg_.vis_merge + in_b / cfg_.vis_merge;
            const uint32_t col = (blk % blocks_w) * cfg_.vis_merge + in_b % cfg_.vis_merge;
            const float hg = grid_coord(row, img.grid_h), wg = grid_coord(col, img.grid_w);
            const int hf = int(hg), wf = int(wg);
            const int hc = std::min<int>(hf + 1, int(side) - 1), wc = std::min<int>(wf + 1, int(side) - 1);
            const float fh = hg - float(hf), fw = wg - float(wf);
            const int32_t id4[4] = {int32_t(hf * side + wf), int32_t(hf * side + wc),
                                    int32_t(hc * side + wf), int32_t(hc * side + wc)};
            const float wt4[4] = {(1 - fh) * (1 - fw), (1 - fh) * fw, fh * (1 - fw), fh * fw};
            for (int c = 0; c < 4; ++c) {
                pidx[uint64_t(c) * N + pi] = id4[c];
                pw[uint64_t(c) * N + pi] = wt4[c];
            }
            for (uint64_t i = 0; i < R_v; ++i) {
                const float f = i < Q_v ? float(row) * vis_inv[i] : float(col) * vis_inv[i - Q_v];
                for (uint64_t half = 0; half < 2; ++half) {
                    cs[pi * HD_v + half * R_v + i] = std::cos(f);
                    sn[pi * HD_v + half * R_v + i] = std::sin(f);
                }
            }
        }
        stream_.upload(h_in, 0, d_in, 0, tot);

        const uint64_t tok_bytes = uint64_t(sp.count) * H * 2;
        DeviceBuffer d_vo(dev_, tok_bytes);
        std::vector<DeviceBuffer> d_vds;
        d_vds.reserve(n_ds);
        for (size_t di = 0; di < n_ds; ++di) d_vds.emplace_back(dev_, tok_bytes);

        auto& rv = get_vis_recipe(uint32_t(N));
        std::vector<Binding> bv = {{"x", &d_in, ox}, {"pidx", &d_in, oi}, {"pw", &d_in, ow},
                                   {"cos", &d_in, oc}, {"sin", &d_in, os}, {"out", &d_vo, 0}};
        for (size_t di = 0; di < n_ds; ++di) bv.push_back({"ds" + std::to_string(di), &d_vds[di], 0});
        for (const auto& td : rv.tensors())
            if (pack_->has(td.name)) bv.push_back(pack_->bind(td.name, td.name));
        rv.enqueue(stream_, bv);

        const uint64_t dst_off = uint64_t(sp.start) * H * 2;
        stream_.copy(d_vo, 0, d_h[0], dst_off, tok_bytes);
        for (size_t di = 0; di < n_ds; ++di) stream_.copy(d_vds[di], 0, d_ds[di], dst_off, tok_bytes);
        stream_.synchronize();
    }

    // 4. Run Qwen3-VL layers with DeepStack injection after layers 0, 1, 2
    int hp = 0;
    auto& rl = get_layer_recipe(S);
    auto& radd = get_add_recipe(S);
    for (uint32_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string pfx = "L" + std::to_string(i) + ".";
        std::vector<Binding> b = {
            {"x", &d_h[hp], 0},
            {"cos", &d_cos, 0},
            {"sin", &d_sin, 0},
            pack_->bind("in_norm", pfx + "in_norm"),
            pack_->bind("post_norm", pfx + "post_norm"),
            pack_->bind("q_norm", pfx + "q_norm"),
            pack_->bind("k_norm", pfx + "k_norm"),
            pack_->bind("qkv_w", pfx + "qkv.w"),
            pack_->bind("o_w", pfx + "o.w"),
            pack_->bind("gu_w", pfx + "gate_up.w"),
            pack_->bind("dn_w", pfx + "down.w"),
            {"out", &d_h[1 - hp], 0},
        };
        if (cfg_.blocks_fp8) {
            b.push_back(pack_->bind("qkv_s", pfx + "qkv.s"));
            b.push_back(pack_->bind("o_s", pfx + "o.s"));
            b.push_back(pack_->bind("gu_s", pfx + "gate_up.s"));
            b.push_back(pack_->bind("dn_s", pfx + "down.s"));
        }
        rl.enqueue(stream_, b);
        hp = 1 - hp;
        if (i < n_ds) {
            std::vector<Binding> ba = {{"a", &d_h[hp], 0}, {"b", &d_ds[i], 0}, {"out", &d_h[1 - hp], 0}};
            radd.enqueue(stream_, ba);
            hp = 1 - hp;
        }
    }

    HostBuffer stg_out(dev_, uint64_t(S) * H * 2);
    stream_.download(d_h[hp], 0, stg_out, 0, stg_out.size());
    stream_.synchronize();

    std::vector<uint16_t> out(size_t(S) * H);
    std::memcpy(out.data(), stg_out.data(), out.size() * 2);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[H3TextEnc] encoded multimodal %u tokens (%zu img, %u layers + DeepStack) in %.1f ms\n",
                S, images.size(), cfg_.num_layers, ms);
    return out;
}

H3MultimodalEncoded H3TextEncoder::encode_multimodal(const Tokenizer& tok, const std::string& prompt,
                                                     const std::vector<VisionImage>& images,
                                                     bool angle_brackets) {
    if (images.empty()) {
        const auto ids = tok.encode(prompt, false);
        H3MultimodalEncoded res;
        res.embeds = encode_text_ids(ids);
        res.seq_len = uint32_t(ids.size());
        res.token_tags.assign(res.seq_len, 1);
        return res;
    }
    std::vector<int32_t> ids;
    std::vector<int32_t> token_tags;
    std::vector<uint8_t> mm_type;
    std::vector<ImgSpan> spans;
    spans.reserve(images.size());

    for (size_t idx = 0; idx < images.size(); ++idx) {
        const std::string lbl_str = angle_brackets
            ? ("<Picture " + std::to_string(idx + 1) + ">: ")
            : ("Picture " + std::to_string(idx + 1) + ": ");
        const auto lbl = tok.encode(lbl_str, false);
        ids.insert(ids.end(), lbl.begin(), lbl.end());
        token_tags.insert(token_tags.end(), lbl.size(), 1);
        mm_type.insert(mm_type.end(), lbl.size(), 0);

        ids.push_back(cfg_.vision_start_token_id);
        token_tags.push_back(0);
        mm_type.push_back(0);

        const uint32_t n_tok = images[idx].tokens();
        spans.push_back({uint32_t(ids.size()), n_tok, images[idx].grid_h / cfg_.vis_merge,
                         images[idx].grid_w / cfg_.vis_merge});
        ids.insert(ids.end(), n_tok, cfg_.image_token_id);
        token_tags.insert(token_tags.end(), n_tok, 0);
        mm_type.insert(mm_type.end(), n_tok, 1);

        ids.push_back(cfg_.vision_end_token_id);
        token_tags.push_back(0);
        mm_type.push_back(0);
    }
    const auto p_ids = tok.encode(prompt, false);
    ids.insert(ids.end(), p_ids.begin(), p_ids.end());
    token_tags.insert(token_tags.end(), p_ids.size(), 1);
    mm_type.insert(mm_type.end(), p_ids.size(), 0);

    H3MultimodalEncoded res;
    res.seq_len = uint32_t(ids.size());
    res.token_tags = std::move(token_tags);
    res.embeds = run_multimodal_forward(ids, mm_type, spans, images);
    return res;
}

Qwen21Encoded H3TextEncoder::encode_qwen21(const Tokenizer& tok, const std::string& prompt,
                                           const std::vector<VisionImage>& images) {
    const std::string p_eff = prompt.empty() ? " " : prompt;
    const std::string sys_block =
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    const auto sys_ids = tok.encode(sys_block, true);
    const size_t drop_idx = sys_ids.size();

    std::vector<int32_t> ids = sys_ids;
    std::vector<uint8_t> mm_type(ids.size(), 0);
    std::vector<ImgSpan> spans;
    spans.reserve(images.size());

    auto append_text = [&](const std::string& s, bool parse_special) {
        const auto t = tok.encode(s, parse_special);
        ids.insert(ids.end(), t.begin(), t.end());
        mm_type.insert(mm_type.end(), t.size(), 0);
    };

    append_text("<|im_start|>user\n", true);
    for (size_t i = 0; i < images.size(); ++i) {
        append_text((i > 0 ? " " : "") + std::string("<image") + std::to_string(i + 1) + ">", false);
        ids.push_back(cfg_.vision_start_token_id);
        mm_type.push_back(0);
        const uint32_t n_tok = images[i].tokens();
        spans.push_back({uint32_t(ids.size()), n_tok, images[i].grid_h / cfg_.vis_merge,
                         images[i].grid_w / cfg_.vis_merge});
        ids.insert(ids.end(), n_tok, cfg_.image_token_id);
        mm_type.insert(mm_type.end(), n_tok, 1);
        ids.push_back(cfg_.vision_end_token_id);
        mm_type.push_back(0);
    }
    append_text(p_eff, false);
    append_text("<|im_end|>\n<|im_start|>assistant\n", true);

    const auto full_embeds = run_multimodal_forward(ids, mm_type, spans, images);
    const uint32_t H = cfg_.hidden_size;
    const uint32_t kept = uint32_t(ids.size() - drop_idx);

    Qwen21Encoded res;
    res.seq_len = kept;
    res.embeds.assign(full_embeds.begin() + ptrdiff_t(drop_idx) * H, full_embeds.end());
    res.image_pad_mask.resize(kept);
    for (uint32_t k = 0; k < kept; ++k) {
        res.image_pad_mask[k] = (ids[drop_idx + k] == cfg_.image_token_id) ? 1u : 0u;
    }
    return res;
}

}  // namespace g2e
