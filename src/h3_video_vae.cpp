#include "g2e/h3_video_vae.hpp"
#include "g2e/numeric.hpp"

#include <nlohmann/json.hpp>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>

namespace g2e {

namespace {

void set_out(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const DType dt = d.dtype;
    const auto sh = d.shape;
    const auto out = g.input(name, dt, sh);
    g.op("identity", {src}, {out});
}

TensorId add_bias2(Ops& o, TensorId x, TensorId b, const std::string& name) {
    return o.add(x, o.reshape(b, {1, o.shape(b)[0]}, name + "_b"), name);
}

// Non-affine RMSNorm (weight = all ones, eps = 1e-5) for Video VAE QK-Norm
TensorId rms_norm_no_affine(Ops& o, TensorId x, float eps, const std::string& name) {
    const auto sh = o.shape(x);
    const auto xf = o.cast(x, DType::F32, name + "_f");
    const auto sq = o.mul(xf, xf, name + "_sq");
    const auto mean = o.mul(o.reduce("sum", sq, unsigned(sh.size() - 1), name + "_sum"),
                            o.constant(1.0f / float(sh.back()), {1}, name + "_inv", DType::F32), name + "_m");
    const auto inv = o.unary("rsqrt", o.add(mean, o.constant(eps, {1}, name + "_e", DType::F32), name + "_me"), name + "_r");
    return o.cast(o.mul(xf, inv, name + "_y"), o.dtype(x), name);
}

// Partial half-split 3D RoPE: x [B, S, NH, 64] BF16, cos/sin [S, 24] BF16 -> rotates first 48 dims
TensorId apply_vae_rope(Ops& o, TensorId x, TensorId cos_tab, TensorId sin_tab, uint32_t rot_dim,
                        const std::string& name) {
    const auto sh = o.shape(x);
    const uint64_t S = sh[1], D = sh[3], half = rot_dim / 2, pass = D - rot_dim;
    const auto parts = o.split(x, 3, {half, half, pass}, name + "_sp");
    const auto c = o.reshape(cos_tab, {1, S, 1, half}, name + "_c");
    const auto s = o.reshape(sin_tab, {1, S, 1, half}, name + "_s");
    const auto r0 = o.sub(o.mul(parts[0], c, name + "_p0c"), o.mul(parts[1], s, name + "_p1s"), name + "_r0");
    const auto r1 = o.add(o.mul(parts[1], c, name + "_p1c"), o.mul(parts[0], s, name + "_p0s"), name + "_r1");
    return o.concat({r0, r1, parts[2]}, 3, name);
}

// Dynamic per-tensor FP8 E4M3 quantization for 4D [B, NH, S, HD] BF16 -> (codes F8, d_scale [1] F32)
std::pair<TensorId, TensorId> quantize_tensor_fp8_4d_vvae(Ops& o, TensorId x, const std::string& name) {
    const auto sh = o.shape(x);
    const uint64_t B = sh[0], NH = sh[1], S = sh[2], D = sh[3];
    const auto ax = o.unary("abs", o.reshape(x, {B * NH * S, D}, name + "_2d"), name + "_abs");
    const auto m1 = o.reshape(o.reduce("max", ax, 1, name + "_m1"), {B * NH, S}, name + "_m1r");
    const auto m2 = o.reduce("max", m1, 1, name + "_m2");
    const auto peak = o.cast(o.reshape(o.reduce("max", m2, 0, name + "_m3"), {1}, name + "_pk"), DType::F32, name + "_pkf");
    const auto scale = o.mul(o.add(peak, o.constant(1e-8f, {1}, name + "_eps", DType::F32), name + "_pe"),
                             o.constant(1.0f / 240.0f, {1}, name + "_imax", DType::F32), name + "_sc");
    const auto inv = o.unary("reciprocal", scale, name + "_inv");
    const auto codes = o.temp(name + "_f8", DType::F8, sh);
    ns_ConvertToFp8::ParamsV2 cp{};
    cp.round_mode = CAST_ROUND_HALF_NE;
    o.graph().op("convert_to_fp8_bf16", {x, inv}, {codes}, &cp, sizeof(cp));
    return {codes, scale};
}

// Split spatial length into overlapping tiles (matches AutoencoderKL.split_tiles)
struct TilePlan {
    std::vector<uint32_t> pos, len, overlap;
};

TilePlan split_tiles_plan(uint32_t input_len, uint32_t tile_size, uint32_t overlap_min, uint32_t vae_ratio) {
    TilePlan tp;
    if (input_len <= tile_size) {
        tp.pos = {0};
        tp.len = {input_len};
        return tp;
    }
    uint32_t n_tiles = (input_len + tile_size - 1) / tile_size;
    int64_t remaining = 0;
    while (true) {
        remaining = int64_t(tile_size) * n_tiles - int64_t(overlap_min) * (n_tiles - 1) - int64_t(input_len);
        if (remaining < 0) ++n_tiles;
        else break;
    }
    tp.overlap.assign(n_tiles - 1, overlap_min);
    const uint32_t rem_units = uint32_t(remaining) / vae_ratio;
    for (uint32_t i = 0; i < rem_units; ++i) tp.overlap[i % (n_tiles - 1)] += vae_ratio;
    tp.pos.push_back(0);
    for (uint32_t i = 0; i + 1 < n_tiles; ++i) tp.pos.push_back(tp.pos.back() + tile_size - tp.overlap[i]);
    tp.len.assign(n_tiles, tile_size);
    return tp;
}

}  // namespace

H3VideoVaeConfig H3VideoVaeConfig::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& w = j.at("config").at("wrapper");
    const auto& s = j.at("config").at("source");
    const auto& v = s.at("vit_decoder_kwargs");
    H3VideoVaeConfig c;
    c.num_layers = v.at("num_layers");
    c.num_heads = v.at("heads");
    c.dim_head = v.at("dim_head");
    c.hidden_size = c.num_heads * c.dim_head;
    c.ffn_hidden = c.hidden_size * 4;
    c.latent_channels = w.at("latent_channels");
    c.vit_decoder_2d = j.value("vit_decoder_2d", w.value("vit_decoder_2d", false));
    c.vae_ratio = s.value("vae_ratio", 16u);
    c.vae_ratio_t = c.vit_decoder_2d ? 1u : s.value("vae_ratio_t", 4u);
    c.clip_length = w.value("vae_clip_length", 17u);
    c.token_drop = c.vit_decoder_2d ? 0u : w.value("vae_token_drop", 3u);
    c.tile_size = w.value("vae_tile_size", 256u);
    c.tile_overlap_min = w.value("vae_tile_overlap_min", 64u);
    c.rope_theta = v.value("rope_theta", 100.0f);
    c.rope_dim_ratio = v.value("rope_dim_ratio", c.vit_decoder_2d ? 0.5f : 0.75f);
    c.blocks_fp8 = (j.value("blocks_dtype", "bf16") == "f8");
    c.has_encoder = j.value("has_encoder", false) && pack.has("encoder.conv_in.weight") && pack.has("quant_conv.weight");
    c.latents_mean = w.at("latents_mean").get<std::vector<float>>();
    c.latents_std = w.at("latents_std").get<std::vector<float>>();
    return c;
}

H3VideoVae::H3VideoVae(const Device& device, const Stream& stream, const WeightPack& weights)
    : dev_(device), stream_(stream), w_(weights), cfg_(H3VideoVaeConfig::from_pack(weights)) {}

H3VideoVae::~H3VideoVae() = default;

Recipe& H3VideoVae::get_pre_recipe(uint32_t batch, uint32_t n_patches) {
    auto key = std::make_pair(batch, n_patches);
    if (auto it = r_pre_.find(key); it != r_pre_.end()) return *it->second;
    const uint64_t B = batch, NP = n_patches, C = cfg_.latent_channels, H = cfg_.hidden_size, NR = cfg_.num_register_tokens;
    GraphBuilder g(dev_);
    Ops o(g, "vp_");
    const auto x = g.input("x", DType::F32, {B * NP, C});
    const auto pq_w = g.input("pq_w", DType::F32, {C, C});
    const auto pq_b = g.input("pq_b", DType::F32, {C});
    const auto xe_w = g.input("xe_w", DType::F32, {H, C});
    const auto xe_b = g.input("xe_b", DType::F32, {H});
    const auto reg = g.input("reg", DType::F32, {1, NR, H});

    const auto pq = o.temp("pq", DType::F32, {B * NP, C});
    g.gemm(x, pq_w, pq, false, true);
    const auto pqb = add_bias2(o, pq, pq_b, "pqb");

    const auto xe = o.temp("xe", DType::F32, {B * NP, H});
    g.gemm(pqb, xe_w, xe, false, true);
    const auto xeb = o.reshape(o.cast(add_bias2(o, xe, xe_b, "xeb"), DType::BF16, "xeb16"), {B, NP, H}, "xeb3");

    TensorId reg16 = o.cast(reg, DType::BF16, "reg16");
    if (B > 1) {
        std::vector<TensorId> reps(B, reg16);
        reg16 = o.concat(reps, 0, "regb");
    }
    const auto cls = o.constant(0.0f, {B, 1, H}, "cls", DType::BF16);
    set_out(g, "out", o.concat({xeb, reg16, cls}, 1, "cat"));

    auto r = g.compile("h3_vvae_pre_b" + std::to_string(B) + "_" + std::to_string(NP));
    auto& ref = *r;
    r_pre_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3VideoVae::get_block_recipe(uint32_t batch, uint32_t seq_len) {
    auto key = std::make_pair(batch, seq_len);
    if (auto it = r_block_.find(key); it != r_block_.end()) return *it->second;
    const uint64_t B = batch, S = seq_len, BS = B * S;
    const uint64_t H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.dim_head, F = cfg_.ffn_hidden;
    const uint64_t R2 = cfg_.rot_dim() / 2;
    const bool fused_vvae_mlp = (std::getenv("G2E_H3_VVAE_UNFUSED") == nullptr);
    const bool fused_vvae_rope = fused_vvae_mlp && (HD == 64) && (cfg_.rot_dim() == 48);
    const bool fp8_sdpa = (std::getenv("G2E_H3_VVAE_FP8_SDPA") != nullptr);
    const uint64_t rope_w = fused_vvae_rope ? HD : R2;

    GraphBuilder g(dev_);
    Ops o(g, "vb_");
    const auto x_in = g.input("x", DType::BF16, {B, S, H});
    const auto cos_tab = g.input("cos", DType::BF16, {S, rope_w});
    const auto sin_tab = g.input("sin", DType::BF16, {S, rope_w});
    const auto norm1 = g.input("norm1", DType::BF16, {H});
    const auto norm2 = g.input("norm2", DType::BF16, {H});
    const auto scale1 = g.input("scale1", DType::BF16, {H});
    const auto scale2 = g.input("scale2", DType::BF16, {H});
    const DType wdt = cfg_.blocks_fp8 ? DType::F8 : DType::BF16;
    const auto qkv_w = g.input("qkv_w", wdt, {3 * H, H});
    const auto qkv_s = cfg_.blocks_fp8 ? g.input("qkv_s", DType::F32, {3 * H}) : kNoTensor;
    const auto qkv_b = g.input("qkv_b", DType::BF16, {3 * H});
    const auto out_w = g.input("out_w", wdt, {H, H});
    const auto out_s = cfg_.blocks_fp8 ? g.input("out_s", DType::F32, {H}) : kNoTensor;
    const auto out_b = g.input("out_b", DType::BF16, {H});
    const auto w1_w = g.input("w1_w", wdt, {2 * F, H});
    const auto w1_s = cfg_.blocks_fp8 ? g.input("w1_s", DType::F32, {2 * F}) : kNoTensor;
    const auto w1_b = g.input("w1_b", DType::BF16, {2 * F});
    const auto w2_w = g.input("w2_w", wdt, {H, F});
    const auto w2_s = cfg_.blocks_fp8 ? g.input("w2_s", DType::F32, {H}) : kNoTensor;
    const auto w2_b = g.input("w2_b", DType::BF16, {H});

    auto lin = [&](TensorId a, TensorId w, TensorId ws, const std::string& nm) {
        return cfg_.blocks_fp8 ? o.fp8_linear(a, w, ws, nm) : o.linear_bf16(a, w, nm);
    };

    const auto x = o.reshape(x_in, {BS, H}, "x2");
    struct VvaeNormQuantParams { float eps, inv_w, inv_range; };
    const VvaeNormQuantParams nqp{cfg_.norm_eps, 1.0f / float(H), 0.004180908203125f};

    // 1. Attention: fused RMSNorm+FP8 quant -> to_qkv -> QKV bias + QK-RMSNorm + RoPE + transpose -> SDPA
    TensorId qkv_lin;
    if (fused_vvae_mlp && cfg_.blocks_fp8 && (H % 128 == 0) && H <= 8192) {
        const auto n1c = o.temp("n1_codes", DType::F8, {BS, H});
        const auto n1s = o.temp("n1_scale", DType::F32, {BS, 1});
        g.op("flashinfer_gaudi_h3_vvae_rmsnorm_quant_bf16_gaudi2", {x, norm1}, {n1c, n1s}, &nqp, sizeof(nqp));
        qkv_lin = o.fp8_gemm(n1c, n1s, qkv_w, qkv_s, "qkv");
    } else {
        const auto n1 = o.rms_norm(x, norm1, cfg_.norm_eps, false, "n1");
        qkv_lin = lin(n1, qkv_w, qkv_s, "qkv");
    }

    TensorId q4, k4, v4;
    if (fused_vvae_rope) {
        const auto qkv_4r = o.reshape(qkv_lin, {BS, NH, 3, HD}, "qkv4r");
        const auto qkv_b3 = o.reshape(qkv_b, {NH, 3, HD}, "qkvb3");
        q4 = o.temp("q4", DType::BF16, {B, NH, S, HD});
        k4 = o.temp("k4", DType::BF16, {B, NH, S, HD});
        v4 = o.temp("v4", DType::BF16, {B, NH, S, HD});
        const float qk_eps = cfg_.norm_eps;
        g.op("flashinfer_gaudi_h3_vvae_qk_norm_rope_bf16_gaudi2",
             {qkv_4r, qkv_b3, cos_tab, sin_tab}, {q4, k4, v4}, &qk_eps, sizeof(qk_eps));
    } else {
        const auto qkv = add_bias2(o, qkv_lin, qkv_b, "qkvb");
        const auto qkv4 = o.reshape(qkv, {B, S, NH, 3 * HD}, "qkv4");
        const auto sp = o.split(qkv4, 3, {HD, HD, HD}, "sp");
        auto q = apply_vae_rope(o, rms_norm_no_affine(o, sp[0], cfg_.norm_eps, "qn"), cos_tab, sin_tab, cfg_.rot_dim(), "qr");
        auto k = apply_vae_rope(o, rms_norm_no_affine(o, sp[1], cfg_.norm_eps, "kn"), cos_tab, sin_tab, cfg_.rot_dim(), "kr");
        auto v = sp[2];
        q4 = o.transpose(q, {0, 2, 1, 3}, "q4");  // [B, NH, S, HD]
        k4 = o.transpose(k, {0, 2, 1, 3}, "k4");
        v4 = o.transpose(v, {0, 2, 1, 3}, "v4");
    }

    const bool kv_split = (std::getenv("G2E_H3_VVAE_NO_KV_SPLIT") == nullptr) && (S >= 1024) && (S % 128 != 0);
    const uint64_t split_rem = S - (S / 128) * 128;
    const bool fast_tail = kv_split && !fp8_sdpa && (split_rem >= 1 && split_rem <= 32) &&
                           (std::getenv("G2E_H3_VVAE_NO_FAST_TAIL") == nullptr);
    TensorId att = kNoTensor;
    if (fp8_sdpa) {
        const auto [q8, dsq] = quantize_tensor_fp8_4d_vvae(o, q4, "sdq");
        const auto [k8, dsk] = quantize_tensor_fp8_4d_vvae(o, k4, "sdk");
        const auto [v8, dsv] = quantize_tensor_fp8_4d_vvae(o, v4, "sdv");
        const auto qs_s = o.constant(128.0f, {1}, "qs128", DType::F32);
        const auto ds_s = o.constant(1.0f / 128.0f, {1}, "ds128", DType::F32);
        if (!kv_split) {
            att = o.temp("att", DType::BF16, {B, NH, S, HD});
            ns_Sdpa::ParamsV7 p{};
            p.scale = float(1.0 / std::sqrt(double(HD)));
            p.is_causal = false;
            p.is_inference = true;
            p.softmax_mode = SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C;
            p.flags = 0x4002f00;
            p.wl = p.wr = -1;
            std::vector<TensorId> ins = {q8, k8, v8, kNoTensor, kNoTensor,
                                         dsq, dsk, dsv, qs_s, kNoTensor, ds_s, kNoTensor, kNoTensor};
            g.op("sdpa_recomp_fwd_hf8", std::span<const TensorId>(ins), std::span<const TensorId>(&att, 1), &p, sizeof(p));
        } else {
            const uint64_t split = (S / 128) * 128;
            const auto ks = o.split(k8, 2, {split, S - split}, "ksp");
            const auto vs = o.split(v8, 2, {split, S - split}, "vsp");
            auto run_stats = [&](TensorId kk, TensorId vv, const std::string& pfx) {
                const auto out = o.temp(pfx + "_o", DType::BF16, {B, NH, S, HD});
                const auto m = o.temp(pfx + "_m", DType::BF16, {B, NH, S, 1});
                const auto linv = o.temp(pfx + "_l", DType::BF16, {B, NH, S, 1});
                ns_Sdpa::ParamsV7 p{};
                p.scale = float(1.0 / std::sqrt(double(HD)));
                p.is_causal = false;
                p.is_inference = false;
                p.softmax_mode = SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C;
                p.flags = 0x4002f00;
                p.wl = p.wr = -1;
                std::vector<TensorId> ins = {q8, kk, vv, kNoTensor, kNoTensor,
                                             dsq, dsk, dsv, qs_s, kNoTensor, ds_s, kNoTensor, kNoTensor};
                const TensorId outs[3] = {out, m, linv};
                g.op("sdpa_recomp_fwd_hf8", std::span<const TensorId>(ins), std::span<const TensorId>(outs, 3), &p, sizeof(p));
                return std::tuple<TensorId, TensorId, TensorId>{out, m, linv};
            };
            const auto [o1, m1, l1] = run_stats(ks[0], vs[0], "sd1");
            const auto [o2, m2, l2] = run_stats(ks[1], vs[1], "sd2");
            const auto m1f = o.cast(m1, DType::F32, "m1f");
            const auto m2f = o.cast(m2, DType::F32, "m2f");
            const auto l1f = o.cast(l1, DType::F32, "l1f");
            const auto l2f = o.cast(l2, DType::F32, "l2f");
            const auto mm = o.binary("max", m1f, m2f, "ks_mm");
            const auto w1 = o.div(o.unary("exp", o.sub(m1f, mm, "ks_d1"), "ks_e1"), l1f, "ks_w1");
            const auto w2 = o.div(o.unary("exp", o.sub(m2f, mm, "ks_d2"), "ks_e2"), l2f, "ks_w2");
            const auto share = o.cast(o.div(w2, o.add(w1, w2, "ks_ws"), "ks_shf"), DType::BF16, "ks_sh");
            att = o.add(o1, o.mul(o.sub(o2, o1, "ks_diff"), share, "ks_upd"), "att");
        }
    } else if (!kv_split) {
        att = o.temp("att", DType::BF16, {B, NH, S, HD});
        ns_Sdpa::ParamsV7 p{};
        p.scale = float(1.0 / std::sqrt(double(HD)));
        p.is_causal = false;
        p.is_inference = true;
        p.softmax_mode = SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C;  // vllm-gaudi qualified 47 dB fast softmax
        p.wl = p.wr = -1;
        std::vector<TensorId> ins = {q4, k4, v4, kNoTensor, kNoTensor};
        for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
        g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(&att, 1), &p, sizeof(p));
    } else if (fast_tail) {
        // Fast aligned SDPA on [0..split) + MME batch_gemm tail-key merge in BF16 (eliminates slow 2nd SDPA on K=5!)
        const uint64_t split = (S / 128) * 128;
        const uint64_t rem = S - split;
        const uint64_t BH = B * NH;
        const auto ks = o.split(k4, 2, {split, rem}, "ksp");
        const auto vs = o.split(v4, 2, {split, rem}, "vsp");
        const auto o1 = o.temp("sd1_o", DType::BF16, {B, NH, S, HD});
        const auto m1 = o.temp("sd1_m", DType::BF16, {B, NH, S, 1});
        const auto l1 = o.temp("sd1_l", DType::BF16, {B, NH, S, 1});
        ns_Sdpa::ParamsV7 p{};
        p.scale = float(1.0 / std::sqrt(double(HD)));
        p.is_causal = false;
        p.is_inference = false;
        p.softmax_mode = SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C;
        p.wl = p.wr = -1;
        std::vector<TensorId> ins = {q4, ks[0], vs[0], kNoTensor, kNoTensor};
        for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
        const TensorId outs[3] = {o1, m1, l1};
        g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(outs, 3), &p, sizeof(p));

        const auto q3 = o.reshape(q4, {BH, S, HD}, "k5_q3");
        const auto k23 = o.reshape(ks[1], {BH, rem, HD}, "k5_k23");
        const auto v23 = o.reshape(vs[1], {BH, rem, HD}, "k5_v23");
        const auto s2 = o.mul(o.batch_gemm(q3, k23, "k5_s2m", false, true, DType::F32),
                              o.constant(p.scale, {1}, "k5_sc", DType::F32), "k5_s2");
        const auto m2 = o.reduce("max", s2, 2, "k5_m2");
        const auto m1f = o.cast(o.reshape(m1, {BH, S, 1}, "k5_m1r"), DType::F32, "k5_m1f");
        const auto l1f = o.cast(o.reshape(l1, {BH, S, 1}, "k5_l1r"), DType::F32, "k5_l1f");
        const auto mm = o.binary("max", m1f, m2, "k5_mm");
        const auto p2 = o.unary("exp", o.sub(s2, mm, "k5_d2"), "k5_p2");
        const auto l2 = o.reduce("sum", p2, 2, "k5_l2");
        // sdpa_recomp_fwd_bf16 in SDPA_SOFTMAX_HF8_1C mode scales internal exp weights by 128.0 (2^7),
        // so l1 = 1 / (128 * sum_exp) -> true sum_exp = 1 / (128 * l1)!
        const auto w1 = o.div(o.unary("exp", o.sub(m1f, mm, "k5_d1"), "k5_e1"),
                              o.mul(l1f, o.constant(128.0f, {1}, "k5_s128", DType::F32), "k5_l1s"), "k5_w1");
        const auto inv_den = o.unary("reciprocal", o.add(w1, l2, "k5_den"), "k5_id");
        const auto a1 = o.cast(o.mul(w1, inv_den, "k5_a1f"), DType::BF16, "k5_a1");
        const auto p2n = o.cast(o.mul(p2, inv_den, "k5_p2nf"), DType::BF16, "k5_p2n");
        const auto o2n = o.batch_gemm(p2n, v23, "k5_o2", false, false, DType::BF16);
        att = o.reshape(o.add(o.mul(o.reshape(o1, {BH, S, HD}, "k5_o1r"), a1, "k5_n1"), o2n, "k5_sum"),
                        {B, NH, S, HD}, "att");
    } else {
        const uint64_t split = (S / 128) * 128;
        const auto ks = o.split(k4, 2, {split, S - split}, "ksp");
        const auto vs = o.split(v4, 2, {split, S - split}, "vsp");
        auto run_stats = [&](TensorId kk, TensorId vv, const std::string& pfx) {
            const auto out = o.temp(pfx + "_o", DType::BF16, {B, NH, S, HD});
            const auto m = o.temp(pfx + "_m", DType::BF16, {B, NH, S, 1});
            const auto linv = o.temp(pfx + "_l", DType::BF16, {B, NH, S, 1});
            ns_Sdpa::ParamsV7 p{};
            p.scale = float(1.0 / std::sqrt(double(HD)));
            p.is_causal = false;
            p.is_inference = false;
            p.softmax_mode = SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C;
            p.wl = p.wr = -1;
            std::vector<TensorId> ins = {q4, kk, vv, kNoTensor, kNoTensor};
            for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
            const TensorId outs[3] = {out, m, linv};
            g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(outs, 3), &p, sizeof(p));
            return std::tuple<TensorId, TensorId, TensorId>{out, m, linv};
        };
        const auto [o1, m1, l1] = run_stats(ks[0], vs[0], "sd1");
        const auto [o2, m2, l2] = run_stats(ks[1], vs[1], "sd2");
        const auto m1f = o.cast(m1, DType::F32, "m1f");
        const auto m2f = o.cast(m2, DType::F32, "m2f");
        const auto l1f = o.cast(l1, DType::F32, "l1f");
        const auto l2f = o.cast(l2, DType::F32, "l2f");
        const auto mm = o.binary("max", m1f, m2f, "ks_mm");
        const auto w1 = o.div(o.unary("exp", o.sub(m1f, mm, "ks_d1"), "ks_e1"), l1f, "ks_w1");
        const auto w2 = o.div(o.unary("exp", o.sub(m2f, mm, "ks_d2"), "ks_e2"), l2f, "ks_w2");
        const auto share = o.cast(o.div(w2, o.add(w1, w2, "ks_ws"), "ks_shf"), DType::BF16, "ks_sh");
        att = o.add(o1, o.mul(o.sub(o2, o1, "ks_diff"), share, "ks_upd"), "att");
    }

    const auto ctx = o.reshape(o.transpose(att, {0, 2, 1, 3}, "at"), {BS, H}, "af");
    TensorId x1, n2c = kNoTensor, n2s = kNoTensor, n2 = kNoTensor;
    if (fused_vvae_mlp && cfg_.blocks_fp8 && (H % 256 == 0) && H <= 4096) {
        const auto ctxc = o.temp("ctx_codes", DType::F8, {BS, H});
        const auto ctxs = o.temp("ctx_scale", DType::F32, {BS, 1});
        g.op("flashinfer_gaudi_h3_row_quant_bf16_gaudi2", {ctx}, {ctxc, ctxs});
        const auto aout_raw = o.fp8_gemm(ctxc, ctxs, out_w, out_s, "out");
        n2c = o.temp("n2_codes", DType::F8, {BS, H});
        n2s = o.temp("n2_scale", DType::F32, {BS, 1});
        x1 = o.temp("x1", DType::BF16, {BS, H});
        g.op("flashinfer_gaudi_h3_vvae_scale_add_norm_quant_bf16_gaudi2",
             {x, aout_raw, out_b, scale1, norm2}, {n2c, n2s, x1}, &nqp, sizeof(nqp));
    } else {
        const auto aout = add_bias2(o, lin(ctx, out_w, out_s, "out"), out_b, "outb");
        x1 = o.add(x, o.mul(aout, o.reshape(scale1, {1, H}, "sc1"), "asc"), "res1");
        n2 = o.rms_norm(x1, norm2, cfg_.norm_eps, false, "n2");
    }
    TensorId fout;
    if (fused_vvae_mlp && cfg_.blocks_fp8 && (F % 128 == 0)) {
        if (n2c == kNoTensor) std::tie(n2c, n2s) = o.quantize(n2, "n2q");
        const auto w1_raw = o.temp("w1_raw", DType::BF16, {BS, 2 * F});
        ns_Fp8Gemm::ParamsV2 gp{};
        gp.transpose_a = false;
        gp.transpose_b = true;
        g.op("fp8_gemm_bf16", {n2c, w1_w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {w1_raw}, &gp, sizeof(gp));
        const auto ac = o.temp("act_codes", DType::F8, {BS, F});
        const auto as = o.temp("act_scale", DType::F32, {BS, 1});
        g.op("flashinfer_gaudi_h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2",
             {w1_raw, n2s, o.cast(w1_s, DType::BF16, "w1sb"), w1_b}, {ac, as});
        fout = add_bias2(o, o.fp8_gemm(ac, as, w2_w, w2_s, "w2"), w2_b, "w2b");
    } else {
        const auto gu = add_bias2(o, (n2c != kNoTensor) ? o.fp8_gemm(n2c, n2s, w1_w, w1_s, "w1")
                                                        : lin(n2, w1_w, w1_s, "w1"), w1_b, "w1b");
        const auto gup = o.split(gu, 1, {F, F}, "gup");
        const auto act = o.mul(o.silu(gup[0], "gs"), gup[1], "act");
        fout = add_bias2(o, lin(act, w2_w, w2_s, "w2"), w2_b, "w2b");
    }
    const auto x2 = o.add(x1, o.mul(fout, o.reshape(scale2, {1, H}, "sc2"), "fsc"), "res2");
    set_out(g, "out", o.reshape(x2, {B, S, H}, "out3"));

    auto r = g.compile("h3_vvae_blk_b" + std::to_string(B) + "_" + std::to_string(S) +
                       (cfg_.blocks_fp8 ? "_f8" : "") + (fast_tail ? "_ks4" : (kv_split ? "_ks" : "")) +
                       (fused_vvae_rope ? "_fused" : (fused_vvae_mlp ? "_fmlp" : "_unfused")) +
                       (fp8_sdpa ? "_f8sdpa" : ""));
    auto& ref = *r;
    r_block_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3VideoVae::get_post_recipe(uint32_t batch, uint32_t seq_len, uint32_t n_patches) {
    auto key = std::make_tuple(batch, seq_len, n_patches);
    if (auto it = r_post_.find(key); it != r_post_.end()) return *it->second;
    const uint64_t B = batch, S = seq_len, NP = n_patches, H = cfg_.hidden_size;
    const uint64_t P = 3ull * cfg_.vae_ratio_t * cfg_.vae_ratio * cfg_.vae_ratio;  // 3072
    GraphBuilder g(dev_);
    Ops o(g, "vpo_");
    const auto x = g.input("x", DType::BF16, {B, S, H});
    const auto nw = g.input("nw", DType::F32, {H});
    const auto nb = g.input("nb", DType::F32, {H});
    const auto pw = g.input("pw", DType::F32, {P, H});
    const auto pb = g.input("pb", DType::F32, {P});

    const auto trimmed = (NP < S) ? o.split(x, 1, {NP, S - NP}, "tr")[0] : x;
    const auto xf = o.cast(o.reshape(trimmed, {B * NP, H}, "fl"), DType::F32, "xf");
    const auto ln = o.layer_norm(xf, nw, nb, cfg_.norm_eps, "ln");
    const auto out = o.temp("po", DType::F32, {B * NP, P});
    g.gemm(ln, pw, out, false, true);
    set_out(g, "out", o.cast(add_bias2(o, out, pb, "pob"), DType::BF16, "pob16"));

    auto r = g.compile("h3_vvae_post_bf16_b" + std::to_string(B) + "_" + std::to_string(S) + "_" + std::to_string(NP));
    auto& ref = *r;
    r_post_.emplace(key, std::move(r));
    return ref;
}

std::vector<float> H3VideoVae::decode_tile(const float* tile_latent, uint32_t t, uint32_t h, uint32_t w) {
    return decode_tiles_batch(tile_latent, 1, t, h, w);
}

void H3VideoVae::enqueue_tiles_batch(const float* tiles_latent, uint32_t batch,
                                     uint32_t t, uint32_t h, uint32_t w, uint32_t slot) {
    const uint32_t C = cfg_.latent_channels, H = cfg_.hidden_size;
    const uint32_t n_patches = t * h * w;
    const uint32_t n_suffix = cfg_.num_register_tokens + 1;
    const uint32_t S = n_patches + n_suffix;
    const uint32_t R2 = cfg_.rot_dim() / 2;  // 24 (3D: 8/axis) or 16 (2D: 8/axis)
    const uint32_t n_axes = cfg_.vit_decoder_2d ? 2u : 3u;
    const uint32_t per_ax = R2 / n_axes;     // 8
    const bool fused_vvae = !cfg_.vit_decoder_2d && (std::getenv("G2E_H3_VVAE_UNFUSED") == nullptr) &&
                            (cfg_.dim_head == 64) && (cfg_.rot_dim() == 48);
    const uint32_t rope_w = fused_vvae ? cfg_.dim_head : R2;
    const size_t tile_in_elems = size_t(C) * n_patches;

    // Pack [B, C, t, h, w] -> [B * n_patches, C] directly into pinned host buffer
    const uint64_t in_bytes = uint64_t(batch) * n_patches * C * 4;
    const uint64_t rope_bytes = uint64_t(S) * rope_w * 2;
    const uint64_t h_bytes = uint64_t(batch) * S * H * 2;
    const uint64_t P = 3ull * cfg_.vae_ratio_t * cfg_.vae_ratio * cfg_.vae_ratio;
    const uint64_t out_bytes = uint64_t(batch) * n_patches * P * 2;  // BF16 D2H

    if (in_bytes > alloc_in_) {
        alloc_in_ = in_bytes;
        d_in_ = DeviceBuffer(dev_, in_bytes);
        const uint64_t stg_sz = alloc_in_ + std::max(rope_bytes * 2, alloc_rope_ * 2);
        h_in_stg_[0] = std::make_unique<HostBuffer>(dev_, stg_sz);
        h_in_stg_[1] = std::make_unique<HostBuffer>(dev_, stg_sz);
    }
    if (rope_bytes > alloc_rope_) {
        alloc_rope_ = rope_bytes;
        d_cos_ = DeviceBuffer(dev_, rope_bytes);
        d_sin_ = DeviceBuffer(dev_, rope_bytes);
        const uint64_t stg_sz = alloc_in_ + alloc_rope_ * 2;
        h_in_stg_[0] = std::make_unique<HostBuffer>(dev_, stg_sz);
        h_in_stg_[1] = std::make_unique<HostBuffer>(dev_, stg_sz);
        cached_t_ = cached_h_ = cached_w_ = 0;
    }
    if (h_bytes > alloc_h_) {
        alloc_h_ = h_bytes;
        d_h_[0] = DeviceBuffer(dev_, h_bytes);
        d_h_[1] = DeviceBuffer(dev_, h_bytes);
    }
    if (out_bytes > alloc_out_) {
        alloc_out_ = out_bytes;
        d_out_ = DeviceBuffer(dev_, out_bytes);
        h_out_stg_[0] = std::make_unique<HostBuffer>(dev_, out_bytes);
        h_out_stg_[1] = std::make_unique<HostBuffer>(dev_, out_bytes);
    }

    float* h_in_ptr = h_in_stg_[slot]->as<float>();
    for (uint32_t b = 0; b < batch; ++b) {
        const float* tsrc = tiles_latent + size_t(b) * tile_in_elems;
        float* tdst = h_in_ptr + size_t(b) * n_patches * C;
        for (uint32_t ti = 0; ti < t; ++ti)
            for (uint32_t hi = 0; hi < h; ++hi)
                for (uint32_t wi = 0; wi < w; ++wi) {
                    const uint32_t p = (ti * h + hi) * w + wi;
                    for (uint32_t ci = 0; ci < C; ++ci)
                        tdst[size_t(p) * C + ci] = tsrc[((size_t(ci) * t + ti) * h + hi) * w + wi];
                }
    }
    stream_.upload(*h_in_stg_[slot], 0, d_in_, 0, in_bytes);

    if (cached_t_ != t || cached_h_ != h || cached_w_ != w) {
        constexpr double kTwoPi = 6.28318530717958647692;
        std::vector<double> inv_freq(per_ax);
        for (uint32_t i = 0; i < per_ax; ++i)
            inv_freq[i] = 1.0 / std::pow(double(cfg_.rope_theta), double(i) / double(per_ax));
        auto* cos_p = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(h_in_stg_[slot]->data()) + alloc_in_);
        auto* sin_p = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(h_in_stg_[slot]->data()) + alloc_in_ + rope_bytes);
        std::fill(cos_p, cos_p + size_t(S) * rope_w, to_bf16(1.0f));
        std::fill(sin_p, sin_p + size_t(S) * rope_w, to_bf16(0.0f));
        for (uint32_t ti = 0; ti < t; ++ti)
            for (uint32_t hi = 0; hi < h; ++hi)
                for (uint32_t wi = 0; wi < w; ++wi) {
                    const uint32_t p = (ti * h + hi) * w + wi;
                    const double coords[3] = {
                        cfg_.vit_decoder_2d ? (2.0 * (double(hi) + 0.5) / double(h) - 1.0)
                                            : (2.0 * (double(ti) + 0.5) / double(t) - 1.0),
                        cfg_.vit_decoder_2d ? (2.0 * (double(wi) + 0.5) / double(w) - 1.0)
                                            : (2.0 * (double(hi) + 0.5) / double(h) - 1.0),
                        2.0 * (double(wi) + 0.5) / double(w) - 1.0,
                    };
                    for (uint32_t ax = 0; ax < n_axes; ++ax)
                        for (uint32_t i = 0; i < per_ax; ++i) {
                            const double ang = kTwoPi * coords[ax] * inv_freq[i];
                            const float cv = float(std::cos(ang)), sv = float(std::sin(ang));
                            const uint32_t k = ax * per_ax + i;
                            if (fused_vvae) {
                                cos_p[size_t(p) * rope_w + k] = to_bf16(cv);
                                cos_p[size_t(p) * rope_w + R2 + k] = to_bf16(cv);
                                sin_p[size_t(p) * rope_w + k] = to_bf16(-sv);
                                sin_p[size_t(p) * rope_w + R2 + k] = to_bf16(sv);
                            } else {
                                cos_p[size_t(p) * rope_w + k] = to_bf16(cv);
                                sin_p[size_t(p) * rope_w + k] = to_bf16(sv);
                            }
                        }
                }
        stream_.upload(*h_in_stg_[slot], alloc_in_, d_cos_, 0, rope_bytes);
        stream_.upload(*h_in_stg_[slot], alloc_in_ + rope_bytes, d_sin_, 0, rope_bytes);
        cached_t_ = t; cached_h_ = h; cached_w_ = w;
    }

    // Pre: post_quant_conv + x_embedder + concat(register_tokens, cls_token)
    {
        auto& rp = get_pre_recipe(batch, n_patches);
        std::vector<Binding> b = {
            {"x", &d_in_, 0},
            w_.bind("pq_w", "post_quant_conv.weight"),
            w_.bind("pq_b", "post_quant_conv.bias"),
            w_.bind("xe_w", "decoder.x_embedder.weight"),
            w_.bind("xe_b", "decoder.x_embedder.bias"),
            w_.bind("reg", "decoder.register_tokens"),
            {"out", &d_h_[0], 0},
        };
        rp.enqueue(stream_, b);
    }

    // 36 ViT blocks
    int hp = 0;
    auto& rb = get_block_recipe(batch, S);
    for (uint32_t i = 0; i < cfg_.num_layers; ++i) {
        const std::string p = "decoder.transformer_blocks." + std::to_string(i) + ".";
        std::vector<Binding> b = {
            {"x", &d_h_[hp], 0},
            {"cos", &d_cos_, 0},
            {"sin", &d_sin_, 0},
            w_.bind("norm1", p + "norm1.weight"),
            w_.bind("norm2", p + "norm2.weight"),
            w_.bind("scale1", p + "scale1"),
            w_.bind("scale2", p + "scale2"),
            w_.bind("qkv_w", p + "attn.to_qkv.weight"),
            w_.bind("qkv_b", p + "attn.to_qkv.bias"),
            w_.bind("out_w", p + "attn.to_out.weight"),
            w_.bind("out_b", p + "attn.to_out.bias"),
            w_.bind("w1_w", p + "ff.w1.weight"),
            w_.bind("w1_b", p + "ff.w1.bias"),
            w_.bind("w2_w", p + "ff.w2.weight"),
            w_.bind("w2_b", p + "ff.w2.bias"),
            {"out", &d_h_[1 - hp], 0},
        };
        if (cfg_.blocks_fp8) {
            b.push_back(w_.bind("qkv_s", p + "attn.to_qkv.scale"));
            b.push_back(w_.bind("out_s", p + "attn.to_out.scale"));
            b.push_back(w_.bind("w1_s", p + "ff.w1.scale"));
            b.push_back(w_.bind("w2_s", p + "ff.w2.scale"));
        }
        rb.enqueue(stream_, b);
        hp = 1 - hp;
    }

    // Post: norm_out + proj_out -> BF16
    {
        auto& rpo = get_post_recipe(batch, S, n_patches);
        std::vector<Binding> b = {
            {"x", &d_h_[hp], 0},
            w_.bind("nw", "decoder.norm_out.weight"),
            w_.bind("nb", "decoder.norm_out.bias"),
            w_.bind("pw", "decoder.proj_out.weight"),
            w_.bind("pb", "decoder.proj_out.bias"),
            {"out", &d_out_, 0},
        };
        rpo.enqueue(stream_, b);
    }

    stream_.download(d_out_, 0, *h_out_stg_[slot], 0, out_bytes);
}

std::vector<float> H3VideoVae::decode_tiles_batch(const float* tiles_latent, uint32_t batch,
                                                  uint32_t t, uint32_t h, uint32_t w) {
    const auto t_gpu0 = std::chrono::steady_clock::now();
    enqueue_tiles_batch(tiles_latent, batch, t, h, w, 0u);
    stream_.synchronize();
    const auto t_gpu1 = std::chrono::steady_clock::now();

    const uint32_t n_patches = t * h * w;
    const uint64_t P = 3ull * cfg_.vae_ratio_t * cfg_.vae_ratio * cfg_.vae_ratio;
    const uint32_t pt = cfg_.vae_ratio_t, ps = cfg_.vae_ratio;
    const uint32_t vt = t * pt, vh = h * ps, vw = w * ps;
    const size_t tile_out_elems = size_t(3) * vt * vh * vw;
    std::vector<float> vid(size_t(batch) * tile_out_elems);
    const uint16_t* src_all = h_out_stg_[0]->as<uint16_t>();
    auto unpack_one = [&](uint32_t b) {
        const uint16_t* src = src_all + size_t(b) * n_patches * P;
        float* dst = vid.data() + size_t(b) * tile_out_elems;
        for (uint32_t ti = 0; ti < t; ++ti)
            for (uint32_t hi = 0; hi < h; ++hi)
                for (uint32_t wi = 0; wi < w; ++wi) {
                    const uint16_t* row = src + size_t((ti * h + hi) * w + wi) * P;
                    for (uint32_t c = 0; c < 3; ++c)
                        for (uint32_t dt = 0; dt < pt; ++dt)
                            for (uint32_t dh = 0; dh < ps; ++dh) {
                                float* drow = dst + (((size_t(c) * vt + (ti * pt + dt)) * vh + (hi * ps + dh)) * vw + wi * ps);
                                const uint16_t* srow = row + (((size_t(c) * pt + dt) * ps + dh) * ps);
                                for (uint32_t dw = 0; dw < ps; ++dw) drow[dw] = from_bf16(srow[dw]);
                            }
                }
    };
    if (batch == 1) {
        unpack_one(0);
    } else {
        const uint32_t nt = std::min<uint32_t>(batch, 16u);
        std::vector<std::thread> ths;
        ths.reserve(nt);
        for (uint32_t tid = 0; tid < nt; ++tid) {
            ths.emplace_back([&, tid, nt]() {
                for (uint32_t b = tid; b < batch; b += nt) unpack_one(b);
            });
        }
        for (auto& th : ths) th.join();
    }
    const auto t_unp1 = std::chrono::steady_clock::now();
    if (std::getenv("G2E_H3_VVAE_PROF")) {
        std::printf("  [H3VideoVae] batch B=%u S=%u: gpu+d2h=%.1f ms, unpack=%.1f ms\n",
                    batch, n_patches + cfg_.num_register_tokens + 1,
                    std::chrono::duration<double, std::milli>(t_gpu1 - t_gpu0).count(),
                    std::chrono::duration<double, std::milli>(t_unp1 - t_gpu1).count());
    }
    return vid;
}

std::vector<uint8_t> H3VideoVae::decode(const float* latent, uint32_t latent_t, uint32_t latent_h, uint32_t latent_w,
                                        uint32_t target_frames, std::vector<float>* out_frames_f32,
                                        RgbChunkCallback on_rgb_frames) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t C = cfg_.latent_channels, ps = cfg_.vae_ratio, pt = cfg_.vae_ratio_t;
    const uint64_t P = 3ull * pt * ps * ps;

    // 1. Denormalize latent: z * std + mean
    std::vector<float> z(size_t(C) * latent_t * latent_h * latent_w);
    for (uint32_t c = 0; c < C; ++c) {
        const float m = cfg_.latents_mean[c], s = cfg_.latents_std[c];
        const size_t plane = size_t(latent_t) * latent_h * latent_w;
        for (size_t i = 0; i < plane; ++i) z[c * plane + i] = latent[c * plane + i] * s + m;
    }

    const uint32_t full_h = latent_h * ps, full_w = latent_w * ps;
    const size_t sp_sz = size_t(full_h) * full_w;
    const auto y_plan = split_tiles_plan(full_h, cfg_.tile_size, cfg_.tile_overlap_min, ps);
    const auto x_plan = split_tiles_plan(full_w, cfg_.tile_size, cfg_.tile_overlap_min, ps);

    // Precompute spatial tile descriptors, per-tile 2D feather masks, and 1/weight ONCE for all temporal chunks!
    struct TileDesc { uint32_t y0, x0, ly0, lh, lx0, lw; std::vector<float> w2_mat; };
    std::vector<TileDesc> tiles;
    tiles.reserve(y_plan.pos.size() * x_plan.pos.size());
    std::vector<float> weight(sp_sz, 0.0f);
    for (size_t iy = 0; iy < y_plan.pos.size(); ++iy) {
        const uint32_t y0 = y_plan.pos[iy], th = y_plan.len[iy];
        const uint32_t ly0 = y0 / ps, lh = th / ps;
        const uint32_t ov_top = (iy > 0) ? y_plan.overlap[iy - 1] : 0;
        const uint32_t ov_bot = (iy + 1 < y_plan.pos.size()) ? y_plan.overlap[iy] : 0;
        std::vector<float> wy_v(th, 1.0f);
        for (uint32_t y = 0; y < th; ++y) {
            float wy = 1.0f;
            if (ov_top && y < ov_top) wy = float(y) / float(ov_top);
            if (ov_bot && y >= th - ov_bot) wy = std::min(wy, float(th - 1 - y) / float(ov_bot));
            wy_v[y] = wy;
        }
        for (size_t ix = 0; ix < x_plan.pos.size(); ++ix) {
            const uint32_t x0 = x_plan.pos[ix], tw = x_plan.len[ix];
            const uint32_t lx0 = x0 / ps, lw = tw / ps;
            const uint32_t ov_left = (ix > 0) ? x_plan.overlap[ix - 1] : 0;
            const uint32_t ov_right = (ix + 1 < x_plan.pos.size()) ? x_plan.overlap[ix] : 0;
            std::vector<float> wx_v(tw, 1.0f);
            for (uint32_t x = 0; x < tw; ++x) {
                float wx = 1.0f;
                if (ov_left && x < ov_left) wx = float(x) / float(ov_left);
                if (ov_right && x >= tw - ov_right) wx = std::min(wx, float(tw - 1 - x) / float(ov_right));
                wx_v[x] = wx;
            }
            TileDesc td{y0, x0, ly0, lh, lx0, lw, std::vector<float>(size_t(th) * tw)};
            for (uint32_t y = 0; y < th; ++y) {
                const float wy = wy_v[y];
                float* wrow = weight.data() + size_t(y0 + y) * full_w + x0;
                float* mrow = td.w2_mat.data() + size_t(y) * tw;
                for (uint32_t x = 0; x < tw; ++x) {
                    const float w2 = std::max(1e-6f, wy * wx_v[x]);
                    mrow[x] = w2;
                    wrow[x] += w2;
                }
            }
            tiles.push_back(std::move(td));
        }
    }
    std::vector<float> inv_weight(sp_sz);
    for (size_t p = 0; p < sp_sz; ++p) inv_weight[p] = 1.0f / weight[p];

    const uint32_t tile_lh = tiles[0].lh, tile_lw = tiles[0].lw;
    const uint32_t tile_tw = tile_lw * ps;
    uint32_t max_batch = cfg_.vit_decoder_2d ? 64u : 28u;
    if (const char* e = std::getenv("G2E_H3_VVAE_TILE_BATCH")) {
        uint32_t v = uint32_t(std::stoul(e));
        if (v > 0) max_batch = v;
    }

    // 16-thread fused BF16-unpack + spatial feather-blend directly from pinned staging into clip [3, ct*pt, H, W]!
    auto parallel_run = [](uint32_t count, auto&& fn) {
        const uint32_t nt = std::min<uint32_t>(count, 16u);
        if (nt <= 1) {
            for (uint32_t i = 0; i < count; ++i) fn(i);
            return;
        }
        std::vector<std::thread> ths;
        ths.reserve(nt);
        for (uint32_t tid = 0; tid < nt; ++tid) {
            ths.emplace_back([&, tid, nt]() {
                for (uint32_t i = tid; i < count; i += nt) fn(i);
            });
        }
        for (auto& th : ths) th.join();
    };

    auto blend_bf16_batch = [&](const uint16_t* bf16_all, uint32_t t_off, uint32_t B, uint32_t ct, float* clip_out) {
        const uint32_t vt = ct * pt;
        const uint32_t n_patches = ct * tile_lh * tile_lw;
        parallel_run(3u * vt, [&](uint32_t cf) {
            const uint32_t c = cf / vt;
            const uint32_t f = cf % vt;
            const uint32_t ti = f / pt;
            const uint32_t dt = f % pt;
            float* out_plane = clip_out + size_t(cf) * sp_sz;
            for (uint32_t bi = 0; bi < B; ++bi) {
                const auto& td = tiles[t_off + bi];
                const uint16_t* tile_src = bf16_all + size_t(bi) * n_patches * P;
                const float* w2 = td.w2_mat.data();
                for (uint32_t hi = 0; hi < tile_lh; ++hi) {
                    for (uint32_t wi = 0; wi < tile_lw; ++wi) {
                        const uint16_t* row = tile_src + size_t((ti * tile_lh + hi) * tile_lw + wi) * P;
                        const uint16_t* sbase = row + ((size_t(c) * pt + dt) * ps) * ps;
                        const uint32_t y_base = hi * ps, x_base = wi * ps;
                        for (uint32_t dh = 0; dh < ps; ++dh) {
                            float* dst = out_plane + size_t(td.y0 + y_base + dh) * full_w + (td.x0 + x_base);
                            const float* mrow = w2 + size_t(y_base + dh) * tile_tw + x_base;
                            const uint16_t* srow = sbase + size_t(dh) * ps;
                            for (uint32_t dw = 0; dw < ps; ++dw) dst[dw] += from_bf16(srow[dw]) * mrow[dw];
                        }
                    }
                }
            }
        });
    };

    auto normalize_clip = [&](float* clip_out, uint32_t vt) {
        parallel_run(3u * vt, [&](uint32_t cf) {
            float* p = clip_out + size_t(cf) * sp_sz;
            const float* iw = inv_weight.data();
            for (size_t i = 0; i < sp_sz; ++i) p[i] *= iw[i];
        });
    };

    // 2. Temporal chunking + DOUBLE-BUFFERED ASYNC GPU||CPU PIPELINE across ALL (chunk, tile_batch) jobs!
    const uint32_t frame_pre_pad = (pt - (cfg_.clip_length % pt)) % pt;          // 3
    const uint32_t tokens_chunk = (cfg_.clip_length + pt - 1) / pt;              // 5
    const uint32_t token_ov = (tokens_chunk - (cfg_.token_drop % tokens_chunk)) % tokens_chunk;  // 2
    const uint32_t frame_ov = (token_ov * pt > frame_pre_pad) ? (token_ov * pt - frame_pre_pad) : 0;  // 5
    const uint32_t chunk_dec = tokens_chunk * pt;                                // 20

    struct BatchJob { uint32_t ci, ts, ct, t_off, B; bool is_first_in_chunk, is_last_in_chunk; };
    std::vector<BatchJob> jobs;
    std::vector<float> zp;
    uint32_t z_t = latent_t;
    uint32_t pad_tokens = 0;
    const bool single_clip = (latent_t < tokens_chunk);
    const size_t lat_frame_sz = size_t(latent_h) * latent_w;

    if (single_clip) {
        zp = std::move(z);
        for (size_t t_off = 0; t_off < tiles.size(); t_off += max_batch) {
            const uint32_t B = uint32_t(std::min<size_t>(max_batch, tiles.size() - t_off));
            jobs.push_back({0u, 0u, latent_t, uint32_t(t_off), B, t_off == 0, t_off + B >= tiles.size()});
        }
    } else {
        uint32_t pseudo_total = latent_t + cfg_.token_drop;
        const uint32_t rem = pseudo_total % tokens_chunk;
        pad_tokens = rem ? (tokens_chunk - rem) : 0u;
        pseudo_total += pad_tokens;
        const uint32_t num_chunks = pseudo_total / tokens_chunk - (cfg_.token_drop > 0 ? 1u : 0u);
        z_t = latent_t + pad_tokens;
        zp.resize(size_t(C) * z_t * lat_frame_sz);
        for (uint32_t c = 0; c < C; ++c) {
            std::memcpy(zp.data() + size_t(c) * z_t * lat_frame_sz,
                        z.data() + size_t(c) * latent_t * lat_frame_sz,
                        size_t(latent_t) * lat_frame_sz * sizeof(float));
            for (uint32_t pt_i = 0; pt_i < pad_tokens; ++pt_i) {
                std::memcpy(zp.data() + (size_t(c) * z_t + latent_t + pt_i) * lat_frame_sz,
                            z.data() + (size_t(c) * latent_t + latent_t - 1) * lat_frame_sz,
                            lat_frame_sz * sizeof(float));
            }
        }
        for (uint32_t ci = 0; ci < num_chunks; ++ci) {
            const uint32_t ts = ci * tokens_chunk;
            const uint32_t te = std::min(z_t, ts + tokens_chunk + token_ov);
            const uint32_t ct = te - ts;
            for (size_t t_off = 0; t_off < tiles.size(); t_off += max_batch) {
                const uint32_t B = uint32_t(std::min<size_t>(max_batch, tiles.size() - t_off));
                jobs.push_back({ci, ts, ct, uint32_t(t_off), B, t_off == 0, t_off + B >= tiles.size()});
            }
        }
    }

    auto extract_batch_latents = [&](const BatchJob& jb, std::vector<float>& tz_batch) {
        const size_t tile_in_elems = size_t(C) * jb.ct * tile_lh * tile_lw;
        tz_batch.resize(size_t(jb.B) * tile_in_elems);
        for (uint32_t bi = 0; bi < jb.B; ++bi) {
            const auto& td = tiles[jb.t_off + bi];
            float* tz = tz_batch.data() + size_t(bi) * tile_in_elems;
            for (uint32_t c = 0; c < C; ++c)
                for (uint32_t ti = 0; ti < jb.ct; ++ti)
                    for (uint32_t hi = 0; hi < tile_lh; ++hi)
                        for (uint32_t wi = 0; wi < tile_lw; ++wi)
                            tz[((size_t(c) * jb.ct + ti) * tile_lh + hi) * tile_lw + wi] =
                                zp[((size_t(c) * z_t + (jb.ts + ti)) * latent_h + (td.ly0 + hi)) * latent_w + (td.lx0 + wi)];
        }
    };

    const uint32_t split_count = (cfg_.token_drop > 0 ? 1u : 0u) + 1u;
    const bool streaming_rgb = static_cast<bool>(on_rgb_frames) && (out_frames_f32 == nullptr);
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

    // Precompute exact final output frame count so streaming mode can stop emitting at `expected_out_t` cleanly.
    uint32_t expected_out_t = 0;
    if (single_clip) {
        const uint32_t df = latent_t * pt;
        expected_out_t = (target_frames > 0 && target_frames < df) ? target_frames : df;
    } else {
        uint32_t sim_total = 0, sim_ov = 0;
        const uint32_t num_chunks = (latent_t + cfg_.token_drop + pad_tokens) / tokens_chunk - (cfg_.token_drop > 0 ? 1u : 0u);
        for (uint32_t ci = 0; ci < num_chunks; ++ci) {
            const uint32_t ts = ci * tokens_chunk;
            const uint32_t te = std::min(z_t, ts + tokens_chunk + token_ov);
            const uint32_t cf = (te - ts) * pt;
            for (uint32_t j = 0; j < split_count; ++j) {
                const uint32_t fs = j * chunk_dec;
                const uint32_t fe = std::min(fs + chunk_dec, cf);
                if (fe <= fs + frame_pre_pad) continue;
                const uint32_t nf = fe - (fs + frame_pre_pad);
                if (j == 0) { sim_total += nf; sim_ov = 0; }
                else { sim_ov = nf; }
            }
        }
        sim_total += sim_ov;
        const uint32_t pad_frames = pad_tokens * pt;
        const uint32_t df = (pad_frames < sim_total) ? (sim_total - pad_frames) : sim_total;
        expected_out_t = (target_frames > 0 && target_frames < df) ? target_frames : df;
    }

    std::vector<std::vector<float>> parts_c(3);
    if (!single_clip && !streaming_rgb) {
        const size_t est_frames = size_t(std::max<uint32_t>(target_frames, z_t * pt + 32u));
        for (uint32_t c = 0; c < 3; ++c) parts_c[c].reserve(est_frames * sp_sz);
    }
    std::vector<float> dec_overlap;
    uint32_t dec_overlap_f = 0;
    uint32_t total_out_f = 0;
    uint32_t emitted_out_f = 0;
    double stream_rgb_ms = 0.0;
    auto convert_planar_to_rgb = [&](const float* src, uint32_t src_t, uint32_t f0, uint32_t emit_nf) {
        std::vector<uint8_t> chunk(size_t(emit_nf) * sp_sz * 3);
        parallel_run(emit_nf, [&](uint32_t f) {
            const float* p0 = src + (size_t(0) * src_t + f0 + f) * sp_sz;
            const float* p1 = src + (size_t(1) * src_t + f0 + f) * sp_sz;
            const float* p2 = src + (size_t(2) * src_t + f0 + f) * sp_sz;
            uint8_t* dst = chunk.data() + size_t(f) * sp_sz * 3;
            for (size_t p = 0; p < sp_sz; ++p) {
                const float r = std::clamp(p0[p] * kStd[0] + kMean[0], 0.0f, 1.0f);
                const float g = std::clamp(p1[p] * kStd[1] + kMean[1], 0.0f, 1.0f);
                const float b = std::clamp(p2[p] * kStd[2] + kMean[2], 0.0f, 1.0f);
                dst[p * 3 + 0] = uint8_t(r * 255.0f + 0.5f);
                dst[p * 3 + 1] = uint8_t(g * 255.0f + 0.5f);
                dst[p * 3 + 2] = uint8_t(b * 255.0f + 0.5f);
            }
        });
        return chunk;
    };
    auto append_part = [&](const float* src, uint32_t src_t, uint32_t f0, uint32_t f1) {
        if (f1 <= f0) return;
        const uint32_t nf = f1 - f0;
        if (streaming_rgb) {
            const uint32_t rem = (emitted_out_f < expected_out_t) ? (expected_out_t - emitted_out_f) : 0u;
            const uint32_t emit_nf = std::min(nf, rem);
            if (emit_nf > 0) {
                const auto tr = std::chrono::steady_clock::now();
                auto chunk = convert_planar_to_rgb(src, src_t, f0, emit_nf);
                stream_rgb_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
                on_rgb_frames(std::move(chunk));
                emitted_out_f += emit_nf;
            }
            total_out_f += nf;
            return;
        }
        const size_t base_f = total_out_f;
        for (uint32_t c = 0; c < 3; ++c) parts_c[c].resize((base_f + nf) * sp_sz);
        parallel_run(3u * nf, [&](uint32_t idx) {
            const uint32_t c = idx / nf, f = idx % nf;
            std::memcpy(parts_c[c].data() + (base_f + f) * sp_sz,
                        src + (size_t(c) * src_t + f0 + f) * sp_sz,
                        sp_sz * sizeof(float));
        });
        total_out_f += nf;
    };

    // Launch job 0 and wait for its D2H into h_out_stg_[0], then double-buffer every subsequent job!
    std::vector<float> tz_batch;
    std::vector<float> cur_clip;
    std::vector<float> decoded;
    uint32_t dec_frames = 0;
    const bool prof = std::getenv("G2E_H3_VVAE_PROF") != nullptr;
    double gpu_wait_ms = 0.0, cpu_blend_ms = 0.0;

    if (!jobs.empty()) {
        extract_batch_latents(jobs[0], tz_batch);
        const auto tg0 = std::chrono::steady_clock::now();
        enqueue_tiles_batch(tz_batch.data(), jobs[0].B, jobs[0].ct, tile_lh, tile_lw, 0u);
        stream_.synchronize();
        gpu_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg0).count();
    }

    for (size_t ji = 0; ji < jobs.size(); ++ji) {
        const auto& jb = jobs[ji];
        const uint32_t cur_slot = uint32_t(ji & 1u);
        const bool has_next = (ji + 1 < jobs.size());
        if (has_next) {
            const auto& jn = jobs[ji + 1];
            extract_batch_latents(jn, tz_batch);
            enqueue_tiles_batch(tz_batch.data(), jn.B, jn.ct, tile_lh, tile_lw, cur_slot ^ 1u);
        }

        // Process cur_slot on 16 CPU threads IN PARALLEL while Gaudi2 crunches jobs[ji + 1]!
        const auto tc0 = std::chrono::steady_clock::now();
        const uint32_t cf = jb.ct * pt;
        if (jb.is_first_in_chunk) {
            cur_clip.resize(size_t(3) * cf * sp_sz);
            parallel_run(3u * cf, [&](uint32_t i) {
                std::memset(cur_clip.data() + size_t(i) * sp_sz, 0, sp_sz * sizeof(float));
            });
        }
        blend_bf16_batch(h_out_stg_[cur_slot]->as<uint16_t>(), jb.t_off, jb.B, jb.ct, cur_clip.data());
        if (jb.is_last_in_chunk) {
            normalize_clip(cur_clip.data(), cf);
            if (single_clip) {
                decoded = std::move(cur_clip);
                dec_frames = cf;
            } else {
                for (uint32_t j = 0; j < split_count; ++j) {
                    const uint32_t fs = j * chunk_dec;
                    const uint32_t fe = std::min(fs + chunk_dec, cf);
                    if (fe <= fs + frame_pre_pad) continue;
                    const uint32_t c0 = fs + frame_pre_pad;
                    const uint32_t nf = fe - c0;
                    if (j == 0) {
                        if (dec_overlap_f > 0) {
                            const uint32_t be = std::min({dec_overlap_f, nf, frame_ov});
                            std::vector<float> blended(size_t(3) * nf * sp_sz);
                            parallel_run(3u * nf, [&](uint32_t idx) {
                                const uint32_t c = idx / nf, f = idx % nf;
                                float* pd = blended.data() + (size_t(c) * nf + f) * sp_sz;
                                const float* pb = cur_clip.data() + (size_t(c) * cf + c0 + f) * sp_sz;
                                if (f < be) {
                                    const float wb = float(f) / float(be), wa = 1.0f - wb;
                                    const float* pa = dec_overlap.data() + (size_t(c) * dec_overlap_f + (dec_overlap_f - be + f)) * sp_sz;
                                    for (size_t p = 0; p < sp_sz; ++p) pd[p] = pa[p] * wa + pb[p] * wb;
                                } else {
                                    std::memcpy(pd, pb, sp_sz * sizeof(float));
                                }
                            });
                            append_part(blended.data(), nf, 0, nf);
                            dec_overlap.clear();
                            dec_overlap_f = 0;
                        } else {
                            append_part(cur_clip.data(), cf, c0, fe);
                        }
                    } else {
                        dec_overlap_f = nf;
                        dec_overlap.resize(size_t(3) * nf * sp_sz);
                        for (uint32_t c = 0; c < 3; ++c) {
                            std::memcpy(dec_overlap.data() + size_t(c) * nf * sp_sz,
                                        cur_clip.data() + (size_t(c) * cf + c0) * sp_sz,
                                        size_t(nf) * sp_sz * sizeof(float));
                        }
                    }
                }
            }
        }
        cpu_blend_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tc0).count();

        if (has_next) {
            const auto tw0 = std::chrono::steady_clock::now();
            stream_.synchronize();
            gpu_wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
        }
    }

    if (!single_clip) {
        if (dec_overlap_f > 0) append_part(dec_overlap.data(), dec_overlap_f, 0, dec_overlap_f);
        const uint32_t pad_frames = pad_tokens * pt;
        dec_frames = (pad_frames < total_out_f) ? (total_out_f - pad_frames) : total_out_f;
    }

    const uint32_t out_t = (target_frames > 0 && target_frames < dec_frames) ? target_frames : dec_frames;
    if (streaming_rgb) {
        if (single_clip && out_t > 0) {
            const auto tr = std::chrono::steady_clock::now();
            auto chunk = convert_planar_to_rgb(decoded.data(), dec_frames, 0, out_t);
            stream_rgb_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
            on_rgb_frames(std::move(chunk));
        }
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (prof) {
            std::printf("  [H3VideoVae] async stream pipeline: gpu_wait=%.1f ms, cpu_blend_overlapped=%.1f ms, rgb24=%.1f ms\n",
                        gpu_wait_ms, cpu_blend_ms, stream_rgb_ms);
        }
        std::printf("[H3VideoVae] decoded+streamed (%u,%u,%u) -> %u frames %ux%u in %.2f s\n",
                    latent_t, latent_h, latent_w, out_t, full_w, full_h, sec);
        return {};
    }
    // 3. 16-thread ImageNet denormalize -> [0, 1] clamp -> RGB uint8 [out_t, H, W, 3]
    const auto tr0 = std::chrono::steady_clock::now();
    if (out_frames_f32) out_frames_f32->resize(size_t(3) * out_t * sp_sz);
    std::vector<uint8_t> rgb(size_t(out_t) * sp_sz * 3);
    parallel_run(out_t, [&](uint32_t f) {
        const float* p0 = single_clip ? (decoded.data() + (size_t(0) * dec_frames + f) * sp_sz) : (parts_c[0].data() + size_t(f) * sp_sz);
        const float* p1 = single_clip ? (decoded.data() + (size_t(1) * dec_frames + f) * sp_sz) : (parts_c[1].data() + size_t(f) * sp_sz);
        const float* p2 = single_clip ? (decoded.data() + (size_t(2) * dec_frames + f) * sp_sz) : (parts_c[2].data() + size_t(f) * sp_sz);
        uint8_t* dst = rgb.data() + size_t(f) * sp_sz * 3;
        if (out_frames_f32) {
            float* o0 = out_frames_f32->data() + (size_t(0) * out_t + f) * sp_sz;
            float* o1 = out_frames_f32->data() + (size_t(1) * out_t + f) * sp_sz;
            float* o2 = out_frames_f32->data() + (size_t(2) * out_t + f) * sp_sz;
            for (size_t p = 0; p < sp_sz; ++p) {
                const float r = std::clamp(p0[p] * kStd[0] + kMean[0], 0.0f, 1.0f);
                const float g = std::clamp(p1[p] * kStd[1] + kMean[1], 0.0f, 1.0f);
                const float b = std::clamp(p2[p] * kStd[2] + kMean[2], 0.0f, 1.0f);
                o0[p] = r; o1[p] = g; o2[p] = b;
                dst[p * 3 + 0] = uint8_t(r * 255.0f + 0.5f);
                dst[p * 3 + 1] = uint8_t(g * 255.0f + 0.5f);
                dst[p * 3 + 2] = uint8_t(b * 255.0f + 0.5f);
            }
        } else {
            for (size_t p = 0; p < sp_sz; ++p) {
                const float r = std::clamp(p0[p] * kStd[0] + kMean[0], 0.0f, 1.0f);
                const float g = std::clamp(p1[p] * kStd[1] + kMean[1], 0.0f, 1.0f);
                const float b = std::clamp(p2[p] * kStd[2] + kMean[2], 0.0f, 1.0f);
                dst[p * 3 + 0] = uint8_t(r * 255.0f + 0.5f);
                dst[p * 3 + 1] = uint8_t(g * 255.0f + 0.5f);
                dst[p * 3 + 2] = uint8_t(b * 255.0f + 0.5f);
            }
        }
    });
    const double rgb_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (prof) {
        std::printf("  [H3VideoVae] async pipeline: gpu_wait=%.1f ms, cpu_blend_overlapped=%.1f ms, rgb24=%.1f ms\n",
                    gpu_wait_ms, cpu_blend_ms, rgb_ms);
    }
    std::printf("[H3VideoVae] decoded (%u,%u,%u) -> %u frames %ux%u in %.2f s\n",
                latent_t, latent_h, latent_w, out_t, full_w, full_h, sec);
    return rgb;
}

namespace {

// Exact IEEE-754 float32 -> float16 -> float32 round-to-nearest-even (matches PyTorch .to(torch.float16).float()).
float round_fp16(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int exp = int((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    uint16_t h = 0;
    if (exp <= 0) {
        if (exp < -10) {
            h = uint16_t(sign);
        } else {
            mant |= 0x800000u;
            const int shift = 14 - exp;
            const uint32_t half = 1u << (shift - 1);
            const uint32_t round_up = ((mant & ((1u << shift) - 1u)) > half ||
                                       ((mant & ((1u << shift) - 1u)) == half && ((mant >> shift) & 1u)))
                                          ? 1u
                                          : 0u;
            h = uint16_t(sign | ((mant >> shift) + round_up));
        }
    } else if (exp >= 31) {
        h = uint16_t(sign | 0x7BFFu);
    } else {
        const uint32_t rem = mant & 0x1FFFu;
        uint32_t m10 = mant >> 13;
        if (rem > 0x1000u || (rem == 0x1000u && (m10 & 1u))) {
            ++m10;
            if (m10 == 0x400u) {
                m10 = 0;
                ++exp;
            }
        }
        h = (exp >= 31) ? uint16_t(sign | 0x7BFFu) : uint16_t(sign | (uint32_t(exp) << 10) | m10);
    }
    const uint32_t hs = uint32_t(h & 0x8000u) << 16;
    uint32_t he = (h >> 10) & 0x1Fu;
    uint32_t hm = h & 0x3FFu;
    uint32_t f = 0;
    if (he == 0) {
        if (hm != 0) {
            he = 127 - 14;
            while ((hm & 0x400u) == 0) { hm <<= 1; --he; }
            hm &= 0x3FFu;
            f = hs | (he << 23) | (hm << 13);
        } else {
            f = hs;
        }
    } else if (he == 31) {
        f = hs | 0x7F800000u | (hm << 13);
    } else {
        f = hs | ((he + 112u) << 23) | (hm << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f, 4);
    return out;
}

TensorId slice1_axis(Ops& o, TensorId x, uint32_t axis, uint64_t idx, const std::string& nm) {
    const uint64_t len = o.shape(x)[axis];
    if (len <= 1) return x;
    if (idx == 0) return o.split(x, axis, {1, len - 1}, nm)[0];
    if (idx + 1 == len) return o.split(x, axis, {len - 1, 1}, nm)[1];
    return o.split(x, axis, {idx, 1, len - 1 - idx}, nm)[1];
}

// Exact PyTorch F.pad(x, (1, 1, 1, 1), mode="reflect") on NHWC [B, H, W, C].
TensorId reflect_pad_1(Ops& o, TensorId x, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t H = sh[1], W = sh[2];
    const auto top = slice1_axis(o, x, 1, H > 1 ? 1 : 0, nm + "_t");
    const auto bot = slice1_axis(o, x, 1, H > 1 ? H - 2 : 0, nm + "_b");
    const auto xh = o.concat({top, x, bot}, 1, nm + "_h");
    const auto lft = slice1_axis(o, xh, 2, W > 1 ? 1 : 0, nm + "_l");
    const auto rgt = slice1_axis(o, xh, 2, W > 1 ? W - 2 : 0, nm + "_r");
    return o.concat({lft, xh, rgt}, 2, nm);
}

// Exact PyTorch F.pad(x, (0, 1, 0, 1), mode="reflect") on NHWC [B, H, W, C] for stride-2 Downsample3D.
TensorId reflect_pad_rb1(Ops& o, TensorId x, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t H = sh[1], W = sh[2];
    const auto bot = slice1_axis(o, x, 1, H > 1 ? H - 2 : 0, nm + "_b");
    const auto xh = o.concat({x, bot}, 1, nm + "_h");
    const auto rgt = slice1_axis(o, xh, 2, W > 1 ? W - 2 : 0, nm + "_r");
    return o.concat({xh, rgt}, 2, nm);
}

// Pure FP32 2D GroupNorm(groups=32, eps=1e-6) on NHWC [B, H, W, C].
TensorId group_norm_nhwc_f32(Ops& o, TensorId x, TensorId w, TensorId b, uint32_t groups, float eps,
                             const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t B = sh[0], H = sh[1], W = sh[2], C = sh[3], G = groups, CpG = C / G;
    const auto r4 = o.reshape(x, {B, H * W, G, CpG}, nm + "_r4");
    const auto g3 = o.reshape(o.transpose(r4, {0, 2, 1, 3}, nm + "_t"), {B, G, H * W * CpG}, nm + "_g3");
    const auto inv_c = o.constant(1.0f / float(H * W * CpG), {1}, nm + "_in", DType::F32);
    const auto mean = o.mul(o.reduce("sum", g3, 2, nm + "_ms"), inv_c, nm + "_m");
    const auto diff = o.sub(g3, mean, nm + "_d");
    const auto var = o.mul(o.reduce("sum", o.mul(diff, diff, nm + "_d2"), 2, nm + "_vs"), inv_c, nm + "_v");
    const auto inv_std = o.unary("rsqrt", o.add(var, o.constant(eps, {1}, nm + "_ep", DType::F32), nm + "_ve"), nm + "_rs");
    const auto norm3 = o.mul(diff, inv_std, nm + "_n3");
    const auto norm4 = o.reshape(o.transpose(o.reshape(norm3, {B, G, H * W, CpG}, nm + "_u4"), {0, 2, 1, 3}, nm + "_ut"),
                                 {B, H, W, C}, nm + "_n");
    return o.add(o.mul(norm4, o.reshape(w, {1, 1, 1, C}, nm + "_w"), nm + "_nw"),
                 o.reshape(b, {1, 1, 1, C}, nm + "_b"), nm);
}

// 2D NHWC spatial_convolution with pre-transposed F32 weight [kH, kW, C_in, C_out] + bias [C_out].
TensorId conv2d_nhwc_f32(Ops& o, TensorId x, TensorId w, TensorId b, uint32_t stride, const std::string& nm) {
    const auto xs = o.shape(x);
    const auto ws = o.shape(w);
    const uint64_t B = xs[0], H = xs[1], W = xs[2], kH = ws[0], kW = ws[1], C_out = ws[3];
    const uint64_t H_out = (H - kH) / stride + 1;
    const uint64_t W_out = (W - kW) / stride + 1;
    const auto y = o.temp(nm + "_c", DType::F32, {B, H_out, W_out, C_out});
    synConvolutionParamsV2 p{};
    p.kH = uint32_t(kH);
    p.kW = uint32_t(kW);
    p.dH = stride;
    p.dW = stride;
    p.padL = p.padR = p.padT = p.padB = 0;
    p.dilH = p.dilW = 1;
    p.nGroups = 1;
    p.paddingType = PADDING_EXPLICIT;
    o.graph().op("spatial_convolution", {x, w}, {y}, &p, sizeof(p));
    return o.add(y, o.reshape(b, {1, 1, 1, C_out}, nm + "_b"), nm);
}

}  // namespace

Recipe& H3VideoVae::get_enc_recipe(uint32_t batch, uint32_t tile_h, uint32_t tile_w) {
    const auto key = std::make_tuple(batch, tile_h, tile_w);
    if (auto it = r_enc_.find(key); it != r_enc_.end()) return *it->second;

    GraphBuilder g(dev_);
    Ops o(g, "ve_");
    auto W = [&](const std::string& name) {
        const auto& e = w_.at(name);
        return g.input(name, e.dtype, e.shape);
    };

    // Exact 2D-sliced single-frame EncoderFCN3D (D=1 causal zero-pad -> exact last temporal slice!)
    TensorId h = g.input("x", DType::F32, {batch, tile_h, tile_w, 3});
    h = conv2d_nhwc_f32(o, reflect_pad_1(o, h, "ci_p"), W("encoder.conv_in.weight"), W("encoder.conv_in.bias"), 1, "ci");

    constexpr uint32_t kNumLevels = 6;
    constexpr uint32_t kChMult[kNumLevels] = {1, 2, 2, 4, 4, 8};
    constexpr uint32_t kSpaceDown[kNumLevels] = {2, 2, 2, 2, 1, 1};
    uint32_t cur_ch = 128;
    for (uint32_t lv = 0; lv < kNumLevels; ++lv) {
        const uint32_t out_ch = 128 * kChMult[lv];
        for (uint32_t bi = 0; bi < 2; ++bi) {
            const std::string bp = "encoder.down." + std::to_string(lv) + ".block." + std::to_string(bi) + ".";
            const std::string nm = "d" + std::to_string(lv) + "_b" + std::to_string(bi);
            TensorId r = o.silu(group_norm_nhwc_f32(o, h, W(bp + "norm1.weight"), W(bp + "norm1.bias"), 32, 1e-6f, nm + "_n1"),
                                nm + "_s1");
            r = conv2d_nhwc_f32(o, reflect_pad_1(o, r, nm + "_p1"), W(bp + "conv1.weight"), W(bp + "conv1.bias"), 1, nm + "_c1");
            r = o.silu(group_norm_nhwc_f32(o, r, W(bp + "norm2.weight"), W(bp + "norm2.bias"), 32, 1e-6f, nm + "_n2"),
                       nm + "_s2");
            r = conv2d_nhwc_f32(o, reflect_pad_1(o, r, nm + "_p2"), W(bp + "conv2.weight"), W(bp + "conv2.bias"), 1, nm + "_c2");
            TensorId sc = h;
            if (cur_ch != out_ch) {
                sc = conv2d_nhwc_f32(o, h, W(bp + "nin_shortcut.weight"), W(bp + "nin_shortcut.bias"), 1, nm + "_sc");
            }
            h = o.add(sc, r, nm + "_add");
            cur_ch = out_ch;
        }
        if (kSpaceDown[lv] == 2) {
            const std::string dp = "encoder.down." + std::to_string(lv) + ".downsample.conv.";
            const std::string nm = "d" + std::to_string(lv) + "_ds";
            h = conv2d_nhwc_f32(o, reflect_pad_rb1(o, h, nm + "_p"), W(dp + "weight"), W(dp + "bias"), 2, nm);
        }
    }

    h = o.silu(group_norm_nhwc_f32(o, h, W("encoder.norm_out.weight"), W("encoder.norm_out.bias"), 32, 1e-6f, "no"), "so");
    h = conv2d_nhwc_f32(o, reflect_pad_1(o, h, "co_p"), W("encoder.conv_out.weight"), W("encoder.conv_out.bias"), 1, "co");
    h = conv2d_nhwc_f32(o, h, W("quant_conv.weight"), W("quant_conv.bias"), 1, "qc");
    set_out(g, "moments", h);

    auto r = g.compile("h3_vvae_enc2d_b" + std::to_string(batch) + "_" + std::to_string(tile_h) + "x" + std::to_string(tile_w));
    auto& ref = *r;
    r_enc_.emplace(key, std::move(r));
    return ref;
}

std::vector<float> H3VideoVae::encode_image(const uint8_t* rgb, uint32_t width, uint32_t height,
                                            uint64_t seed, bool sample_posterior,
                                            std::vector<float>* out_moments) {
    if (!cfg_.has_encoder) {
        throw std::runtime_error("H3VideoVae::encode_image: loaded VideoVAE pack does not include encoder weights "
                                 "(re-run convert_h3_video_vae.py without --no-encoder)");
    }
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t ps = cfg_.vae_ratio;  // 16
    const uint32_t align = ps * 2;       // 32 (latent patch 2x2 * vae_ratio 16)
    const uint32_t crop_w = (width / align) * align;
    const uint32_t crop_h = (height / align) * align;
    if (!crop_w || !crop_h) throw std::invalid_argument("H3VideoVae::encode_image: image too small (<32px)");
    const uint32_t y_off = (height - crop_h) / 2;
    const uint32_t x_off = (width - crop_w) / 2;

    // 1. ImageNet normalize [crop_h, crop_w, 3] F32
    constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> img_norm(size_t(crop_h) * crop_w * 3);
    for (uint32_t y = 0; y < crop_h; ++y) {
        const uint8_t* srow = rgb + (size_t(y_off + y) * width + x_off) * 3;
        float* drow = img_norm.data() + size_t(y) * crop_w * 3;
        for (uint32_t x = 0; x < crop_w; ++x) {
            drow[x * 3 + 0] = (float(srow[x * 3 + 0]) * (1.0f / 255.0f) - kMean[0]) / kStd[0];
            drow[x * 3 + 1] = (float(srow[x * 3 + 1]) * (1.0f / 255.0f) - kMean[1]) / kStd[1];
            drow[x * 3 + 2] = (float(srow[x * 3 + 2]) * (1.0f / 255.0f) - kMean[2]) / kStd[2];
        }
    }

    // 2. Spatial tiling (matching AutoencoderKL.split_tiles & tiled_encode)
    const auto y_plan = split_tiles_plan(crop_h, cfg_.tile_size, cfg_.tile_overlap_min, ps);
    const auto x_plan = split_tiles_plan(crop_w, cfg_.tile_size, cfg_.tile_overlap_min, ps);
    const size_t ny = y_plan.pos.size(), nx = x_plan.pos.size();
    const size_t n_tiles = ny * nx;
    const uint32_t th = y_plan.len[0], tw = x_plan.len[0];
    const uint32_t lth = th / ps, ltw = tw / ps;
    const uint32_t C2 = cfg_.latent_channels * 2;  // 48

    const uint32_t max_b = 14u;
    std::vector<std::vector<float>> tile_out(n_tiles);  // each [C2, lth, ltw] planar F32
    for (size_t t0_idx = 0; t0_idx < n_tiles; t0_idx += max_b) {
        const uint32_t B = uint32_t(std::min<size_t>(max_b, n_tiles - t0_idx));
        const uint64_t in_bytes = uint64_t(B) * th * tw * 3 * 4;
        const uint64_t out_bytes = uint64_t(B) * lth * ltw * C2 * 4;
        DeviceBuffer d_x(dev_, in_bytes), d_m(dev_, out_bytes);
        HostBuffer h_x(dev_, in_bytes), h_m(dev_, out_bytes);
        float* hx = h_x.as<float>();
        for (uint32_t bi = 0; bi < B; ++bi) {
            const size_t idx = t0_idx + bi;
            const uint32_t iy = uint32_t(idx / nx), ix = uint32_t(idx % nx);
            const uint32_t py = y_plan.pos[iy], px = x_plan.pos[ix];
            float* dst = hx + size_t(bi) * th * tw * 3;
            for (uint32_t y = 0; y < th; ++y) {
                std::memcpy(dst + size_t(y) * tw * 3,
                            img_norm.data() + (size_t(py + y) * crop_w + px) * 3,
                            size_t(tw) * 3 * sizeof(float));
            }
        }
        stream_.upload(h_x, 0, d_x, 0, in_bytes);
        auto& r = get_enc_recipe(B, th, tw);
        std::vector<Binding> b;
        b.reserve(r.tensors().size());
        for (const auto& td : r.tensors()) {
            if (td.name == "x") b.push_back({"x", &d_x, 0});
            else if (td.name == "moments") b.push_back({"moments", &d_m, 0});
            else b.push_back(w_.bind(td.name, td.name));
        }
        r.enqueue(stream_, b);
        stream_.download(d_m, 0, h_m, 0, out_bytes);
        stream_.synchronize();

        const float* hm = h_m.as<float>();
        for (uint32_t bi = 0; bi < B; ++bi) {
            const size_t idx = t0_idx + bi;
            tile_out[idx].resize(size_t(C2) * lth * ltw);
            const float* src = hm + size_t(bi) * lth * ltw * C2;
            for (uint32_t y = 0; y < lth; ++y)
                for (uint32_t x = 0; x < ltw; ++x)
                    for (uint32_t c = 0; c < C2; ++c)
                        tile_out[idx][(size_t(c) * lth + y) * ltw + x] = src[(size_t(y) * ltw + x) * C2 + c];
        }
    }

    // 3. Exact klvae.py sequential tile blend in latent space -> [48, lh, lw]
    const uint32_t lh = crop_h / ps, lw = crop_w / ps;
    std::vector<float> moments(size_t(C2) * lh * lw, 0.0f);
    uint32_t dst_y = 0;
    for (size_t iy = 0; iy < ny; ++iy) {
        const uint32_t ov_y_prev = (iy > 0) ? (y_plan.overlap[iy - 1] / ps) : 0u;
        const uint32_t ov_y_next = (iy + 1 < ny) ? (y_plan.overlap[iy] / ps) : 0u;
        const uint32_t keep_h = lth - ov_y_next;
        uint32_t dst_x = 0;
        for (size_t ix = 0; ix < nx; ++ix) {
            const uint32_t ov_x_prev = (ix > 0) ? (x_plan.overlap[ix - 1] / ps) : 0u;
            const uint32_t ov_x_next = (ix + 1 < nx) ? (x_plan.overlap[ix] / ps) : 0u;
            const uint32_t keep_w = ltw - ov_x_next;
            // Start from current tile copy [C2, lth, ltw], blend vertical with rows[iy-1][ix], then horizontal with rows[iy][ix-1]
            std::vector<float> cur = tile_out[iy * nx + ix];
            if (ov_y_prev > 0) {
                const auto& prev = tile_out[(iy - 1) * nx + ix];
                const uint32_t be = std::min(lth, ov_y_prev);
                for (uint32_t c = 0; c < C2; ++c) {
                    for (uint32_t y = 0; y < be; ++y) {
                        const float wb = float(y) / float(be), wa = 1.0f - wb;
                        float* cd = cur.data() + (size_t(c) * lth + y) * ltw;
                        const float* ps_row = prev.data() + (size_t(c) * lth + (lth - be + y)) * ltw;
                        for (uint32_t x = 0; x < ltw; ++x) cd[x] = ps_row[x] * wa + cd[x] * wb;
                    }
                }
            }
            if (ov_x_prev > 0) {
                const auto& left = tile_out[iy * nx + (ix - 1)];
                const uint32_t be = std::min(ltw, ov_x_prev);
                for (uint32_t c = 0; c < C2; ++c) {
                    for (uint32_t y = 0; y < lth; ++y) {
                        float* cd = cur.data() + (size_t(c) * lth + y) * ltw;
                        const float* ls_row = left.data() + (size_t(c) * lth + y) * ltw + (ltw - be);
                        for (uint32_t x = 0; x < be; ++x) {
                            const float wb = float(x) / float(be), wa = 1.0f - wb;
                            cd[x] = ls_row[x] * wa + cd[x] * wb;
                        }
                    }
                }
            }
            // Write trimmed [0..keep_h, 0..keep_w) into global moments [C2, lh, lw]
            for (uint32_t c = 0; c < C2; ++c) {
                for (uint32_t y = 0; y < keep_h; ++y) {
                    std::memcpy(moments.data() + (size_t(c) * lh + (dst_y + y)) * lw + dst_x,
                                cur.data() + (size_t(c) * lth + y) * ltw,
                                size_t(keep_w) * sizeof(float));
                }
            }
            dst_x += keep_w;
        }
        dst_y += keep_h;
    }
    if (out_moments) *out_moments = moments;

    // 4. DiagonalGaussianDistribution sample (with deterministic seed) + FP16 round-trip + normalize + [1,2,2] patchify
    const uint32_t C = cfg_.latent_channels;  // 24
    std::vector<float> z_norm(size_t(C) * lh * lw);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (uint32_t c = 0; c < C; ++c) {
        const float mean_c = cfg_.latents_mean[c];
        const float inv_std_c = 1.0f / cfg_.latents_std[c];
        for (uint32_t y = 0; y < lh; ++y) {
            for (uint32_t x = 0; x < lw; ++x) {
                const size_t p = size_t(y) * lw + x;
                const float mu = moments[size_t(c) * lh * lw + p];
                float val = mu;
                if (sample_posterior) {
                    const float lv = std::clamp(moments[size_t(C + c) * lh * lw + p], -30.0f, 20.0f);
                    val = mu + std::exp(0.5f * lv) * nd(rng);
                }
                val = round_fp16(val);
                z_norm[size_t(c) * lh * lw + p] = (val - mean_c) * inv_std_c;
            }
        }
    }

    const uint32_t ph = lh / 2, pw = lw / 2;
    std::vector<float> rows(size_t(ph) * pw * 96);
    for (uint32_t hi = 0; hi < ph; ++hi) {
        for (uint32_t wi = 0; wi < pw; ++wi) {
            float* dst = rows.data() + size_t(hi * pw + wi) * 96;
            for (uint32_t c = 0; c < C; ++c)
                for (uint32_t dh = 0; dh < 2; ++dh)
                    for (uint32_t dw = 0; dw < 2; ++dw)
                        dst[(c * 2 + dh) * 2 + dw] = z_norm[(size_t(c) * lh + (hi * 2 + dh)) * lw + (wi * 2 + dw)];
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[H3VideoVae] encoded image %ux%u -> (%u,%u) -> %u cond rows in %.1f ms\n",
                crop_w, crop_h, lh, lw, ph * pw, ms);
    return rows;
}

}  // namespace g2e
