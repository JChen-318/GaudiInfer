#include "g2e/qwen21_dit.hpp"
#include "g2e/numeric.hpp"

#include <nlohmann/json.hpp>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace g2e {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void set_out(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const auto out = g.input(name, d.dtype, d.shape);
    g.op("identity", {src}, {out});
}

// Non-affine LayerNorm(eps) over innermost dim of [M, D] BF16.
TensorId layer_norm_no_affine(Ops& o, TensorId x, float eps, const std::string& nm) {
    const uint64_t D = o.shape(x).back();
    const auto ones = o.constant(1.0f, {D}, nm + "_w", DType::BF16);
    const auto zeros = o.constant(0.0f, {D}, nm + "_b", DType::BF16);
    return o.layer_norm(x, ones, zeros, eps, nm);
}

// Interleaved complex FP32 RoPE: x [S, NH, D] BF16, cos/sin [S, D/2] F32 -> [S, NH, D] BF16.
TensorId apply_interleaved_rope(Ops& o, TensorId x, TensorId cos_tab, TensorId sin_tab, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t S = sh[0], NH = sh[1], D = sh[2], D2 = D / 2;
    const auto xf = o.cast(x, DType::F32, nm + "_f");
    const auto xp = o.reshape(xf, {S, NH, D2, 2}, nm + "_p");
    const auto sp = o.split(xp, 3, {1, 1}, nm + "_sp");
    const auto c = o.reshape(cos_tab, {S, 1, D2, 1}, nm + "_c");
    const auto s = o.reshape(sin_tab, {S, 1, D2, 1}, nm + "_s");
    const auto r0 = o.sub(o.mul(sp[0], c, nm + "_0c"), o.mul(sp[1], s, nm + "_1s"), nm + "_r0");
    const auto r1 = o.add(o.mul(sp[0], s, nm + "_0s"), o.mul(sp[1], c, nm + "_1c"), nm + "_r1");
    const auto cat = o.reshape(o.concat({r0, r1}, 3, nm + "_cat"), {S, NH, D}, nm + "_r3");
    return o.cast(cat, DType::BF16, nm);
}

// Modulate prefix [0..P) with row 1 (t=0) and target [P..P+T) with row 0 (t=sigma).
TensorId modulate_split(Ops& o, TensorId x, TensorId p_row0, TensorId p_row1,
                        uint64_t P, uint64_t T, const std::string& nm) {
    if (P == 0) return o.mul(x, p_row0, nm);
    const auto sp = o.split(x, 0, {P, T}, nm + "_sp");
    const auto a = o.mul(sp[0], p_row1, nm + "_p");
    const auto b = o.mul(sp[1], p_row0, nm + "_t");
    return o.concat({a, b}, 0, nm);
}

// Compute QwenImage21TemporalTimesteps sinusoidal embedding for a scalar timestep in BF16.
void compute_timestep_sincos(float t_scalar, uint16_t* dst_bf16) {
    // Match PyTorch: timestep is cast to bf16 before time_text_embed, then 1000.0 * t.float()
    const float t_b = from_bf16(to_bf16(t_scalar));
    const double tf = 1000.0 * double(t_b);
    constexpr int kHalf = 128;
    const double log_max = std::log(10000.0);
    for (int i = 0; i < kHalf; ++i) {
        const double freq = std::exp(-log_max * double(i) / double(kHalf));
        const double a = tf * freq;
        dst_bf16[i] = to_bf16(float(std::cos(a)));
        dst_bf16[kHalf + i] = to_bf16(float(std::sin(a)));
    }
}

// Build QwenImage21Rope cos/sin [S, 64] F32 matching QwenImage21Rope.forward exactly.
void build_qwen21_rope(const std::vector<Qwen21ImgShape>& img_shapes, const std::vector<uint8_t>& exp_mask,
                       const std::array<uint32_t, 3>& axes_dims,
                       std::vector<float>& out_cos, std::vector<float>& out_sin) {
    const size_t S = exp_mask.size();
    const uint32_t D2 = (axes_dims[0] + axes_dims[1] + axes_dims[2]) / 2;  // 64
    std::vector<int64_t> f_idx(S), h_idx(S), w_idx(S);
    std::vector<int64_t> img_h, img_w;
    size_t cursor = 0;
    int64_t pos = 0;
    for (const auto& sh : img_shapes) {
        size_t bs = cursor;
        while (bs < S && exp_mask[bs] == 0) ++bs;
        if (bs >= S) throw std::runtime_error("Qwen21DiT: image_pad_mask has fewer blocks than img_shapes");
        for (size_t i = cursor; i < bs; ++i) f_idx[i] = pos++;
        const uint32_t nt = sh.tokens();
        cursor = bs + nt;
        for (size_t i = bs; i < cursor; ++i) f_idx[i] = pos;
        pos += int64_t(std::max(sh.h, sh.w));
        const int64_t h0 = -int64_t(sh.h - sh.h / 2), h1 = int64_t(sh.h / 2);
        const int64_t w0 = -int64_t(sh.w - sh.w / 2), w1 = int64_t(sh.w / 2);
        for (int64_t y = h0; y < h1; ++y)
            for (int64_t x = w0; x < w1; ++x) {
                img_h.push_back(y);
                img_w.push_back(x);
            }
    }
    for (size_t i = cursor; i < S; ++i) f_idx[i] = pos++;
    h_idx = f_idx;
    w_idx = f_idx;
    size_t k = 0;
    for (size_t i = 0; i < S; ++i) {
        if (exp_mask[i]) {
            h_idx[i] = img_h[k];
            w_idx[i] = img_w[k];
            ++k;
        }
    }
    if (k != img_h.size()) throw std::runtime_error("Qwen21DiT: image_pad_mask token count mismatch");

    std::array<std::vector<double>, 3> inv_freq;
    for (int a = 0; a < 3; ++a) {
        const uint32_t dim = axes_dims[a];
        inv_freq[a].resize(dim / 2);
        for (uint32_t i = 0; i < dim / 2; ++i) {
            inv_freq[a][i] = 1.0 / std::pow(10000.0, double(2 * i) / double(dim));
        }
    }
    out_cos.resize(S * D2);
    out_sin.resize(S * D2);
    for (size_t s = 0; s < S; ++s) {
        const int64_t coords[3] = {f_idx[s], h_idx[s], w_idx[s]};
        uint32_t off = 0;
        for (int a = 0; a < 3; ++a) {
            for (size_t i = 0; i < inv_freq[a].size(); ++i) {
                const float ang = float(double(coords[a]) * inv_freq[a][i]);
                out_cos[s * D2 + off + i] = std::cos(ang);
                out_sin[s * D2 + off + i] = std::sin(ang);
            }
            off += uint32_t(inv_freq[a].size());
        }
    }
}

}  // namespace

Qwen21DiTConfig Qwen21DiTConfig::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& c = j.at("config");
    Qwen21DiTConfig cfg;
    cfg.num_layers = j.at("layers").get<uint32_t>();
    cfg.num_heads = c.at("num_attention_heads");
    cfg.head_dim = c.at("attention_head_dim");
    cfg.hidden_size = cfg.num_heads * cfg.head_dim;
    cfg.mlp_ratio = c.value("mlp_ratio", 3u);
    cfg.ffn_hidden = cfg.hidden_size * cfg.mlp_ratio;
    cfg.in_channels = c.value("in_channels", 64u);
    cfg.out_channels = c.value("out_channels", 64u);
    if (c.contains("axes_dims_rope")) {
        for (int i = 0; i < 3; ++i) cfg.axes_dims_rope[i] = c["axes_dims_rope"].at(i);
    }
    cfg.eps = c.value("eps", 1e-6f);
    cfg.causal_condition = c.value("causal_condition", true);
    cfg.blocks_fp8 = (j.value("blocks_dtype", "f8") == "f8");
    return cfg;
}

Qwen21DiT::Qwen21DiT(const Device& device, const Stream& stream, const WeightPack& weights)
    : dev_(device), stream_(stream), w_(weights), cfg_(Qwen21DiTConfig::from_pack(weights)) {}

Qwen21DiT::~Qwen21DiT() = default;

Recipe& Qwen21DiT::get_mod_recipe() {
    if (r_mod_) return *r_mod_;
    const uint64_t H = cfg_.hidden_size;
    GraphBuilder g(dev_);
    Ops o(g, "qmod_");
    const auto t_in = g.input("t_in", DType::BF16, {2, 256});
    const auto l1 = g.input("l1", DType::BF16, {H, 256});
    const auto l2 = g.input("l2", DType::BF16, {H, H});
    const auto mod_w = g.input("mod_w", DType::BF16, {4 * H, H});
    const auto nout_w = g.input("nout_w", DType::BF16, {H, H});

    const auto temb = o.linear_bf16(o.silu(o.linear_bf16(t_in, l1, "t1"), "ts"), l2, "t2");  // [2, H]
    const auto temb_s = o.silu(temb, "ms");
    const auto mod = o.linear_bf16(temb_s, mod_w, "mod");  // [2, 4H]
    const auto sp = o.split(mod, 1, {H, H, H, H}, "sp");
    const auto one = o.constant(1.0f, {1}, "one", DType::BF16);
    const auto s1 = o.add(sp[0], one, "s1");
    const auto g1 = o.unary("tanh", sp[1], "g1");
    const auto s2 = o.add(sp[2], one, "s2");
    const auto g2 = o.unary("tanh", sp[3], "g2");

    // Split row 0 (t=sigma) and row 1 (t=0) for fast broadcast in blocks
    const auto s1_r = o.split(s1, 0, {1, 1}, "s1r");
    const auto g1_r = o.split(g1, 0, {1, 1}, "g1r");
    const auto s2_r = o.split(s2, 0, {1, 1}, "s2r");
    const auto g2_r = o.split(g2, 0, {1, 1}, "g2r");

    // Final norm_out scale for target tokens (row 0 = t=sigma)
    const auto temb0 = o.split(temb_s, 0, {1, 1}, "ts0")[0];
    const auto fsc = o.add(o.linear_bf16(temb0, nout_w, "ns"), one, "fsc");

    set_out(g, "s1_0", s1_r[0]); set_out(g, "s1_1", s1_r[1]);
    set_out(g, "g1_0", g1_r[0]); set_out(g, "g1_1", g1_r[1]);
    set_out(g, "s2_0", s2_r[0]); set_out(g, "s2_1", s2_r[1]);
    set_out(g, "g2_0", g2_r[0]); set_out(g, "g2_1", g2_r[1]);
    set_out(g, "fsc", fsc);

    r_mod_ = g.compile("qwen21_dit_mod");
    return *r_mod_;
}

Recipe& Qwen21DiT::get_entry_prefill_recipe(uint32_t vlm_len, uint32_t img_tokens, uint32_t total_len) {
    const auto key = std::make_tuple(vlm_len, img_tokens, total_len);
    if (auto it = r_entry_pre_.find(key); it != r_entry_pre_.end()) return *it->second;
    const uint64_t H = cfg_.hidden_size, C = cfg_.in_channels;
    GraphBuilder g(dev_);
    Ops o(g, "qep_");
    const auto enc_hs = g.input("enc_hs", DType::BF16, {vlm_len, H});
    const auto lat = g.input("lat", DType::BF16, {img_tokens, C});
    const auto gather_idx = g.input("gather_idx", DType::I32, {total_len});
    const auto txt_nw = g.input("txt_nw", DType::BF16, {H});
    const auto txt_in_w = g.input("txt_in_w", DType::BF16, {H, H});
    const auto txt_out_w = g.input("txt_out_w", DType::BF16, {H, H});
    const auto img_in_w = g.input("img_in_w", DType::BF16, {H, C});

    // txt_in: ZeroCenterRMSNorm (weight + 1) -> in_layer -> GELU(tanh) -> out_layer
    auto txt = o.rms_norm(enc_hs, txt_nw, cfg_.eps, true, "tn");
    txt = o.gelu(o.linear_bf16(txt, txt_in_w, "ti"), true, "tg");
    txt = o.linear_bf16(txt, txt_out_w, "to");

    const auto img = o.linear_bf16(lat, img_in_w, "ii");
    const auto pool = o.concat({txt, img}, 0, "pool");
    set_out(g, "joint", o.gather_rows(pool, gather_idx, "g"));

    auto r = g.compile("qwen21_entry_pre_" + std::to_string(vlm_len) + "_" + std::to_string(img_tokens) + "_" +
                       std::to_string(total_len));
    auto& ref = *r;
    r_entry_pre_.emplace(key, std::move(r));
    return ref;
}

Recipe& Qwen21DiT::get_entry_decode_recipe(uint32_t target_len) {
    if (auto it = r_entry_dec_.find(target_len); it != r_entry_dec_.end()) return *it->second;
    const uint64_t H = cfg_.hidden_size, C = cfg_.in_channels;
    GraphBuilder g(dev_);
    Ops o(g, "qed_");
    const auto lat = g.input("lat", DType::BF16, {target_len, C});
    const auto img_in_w = g.input("img_in_w", DType::BF16, {H, C});
    set_out(g, "out", o.linear_bf16(lat, img_in_w, "ii"));
    auto r = g.compile("qwen21_entry_dec_" + std::to_string(target_len));
    auto& ref = *r;
    r_entry_dec_.emplace(target_len, std::move(r));
    return ref;
}

Recipe& Qwen21DiT::get_blk_prefill_recipe(uint32_t prefix_len, uint32_t target_len) {
    const auto key = std::make_pair(prefix_len, target_len);
    if (auto it = r_blk_pre_.find(key); it != r_blk_pre_.end()) return *it->second;
    const uint64_t P = prefix_len, T = target_len, S = P + T;
    const uint64_t H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.head_dim, F = cfg_.ffn_hidden;

    GraphBuilder g(dev_);
    Ops o(g, "qbp_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto cos_tab = g.input("cos", DType::F32, {S, HD / 2});
    const auto sin_tab = g.input("sin", DType::F32, {S, HD / 2});
    const auto mask = g.input("mask", DType::BF16, {1, 1, S, S});
    const auto s1_0 = g.input("s1_0", DType::BF16, {1, H});
    const auto s1_1 = g.input("s1_1", DType::BF16, {1, H});
    const auto g1_0 = g.input("g1_0", DType::BF16, {1, H});
    const auto g1_1 = g.input("g1_1", DType::BF16, {1, H});
    const auto s2_0 = g.input("s2_0", DType::BF16, {1, H});
    const auto s2_1 = g.input("s2_1", DType::BF16, {1, H});
    const auto g2_0 = g.input("g2_0", DType::BF16, {1, H});
    const auto g2_1 = g.input("g2_1", DType::BF16, {1, H});

    const auto norm_q = g.input("norm_q", DType::BF16, {HD});
    const auto norm_k = g.input("norm_k", DType::BF16, {HD});
    const DType wdt = cfg_.blocks_fp8 ? DType::F8 : DType::BF16;
    const auto qkv_w = g.input("qkv_w", wdt, {3 * H, H});
    const auto qkv_s = cfg_.blocks_fp8 ? g.input("qkv_s", DType::F32, {3 * H}) : kNoTensor;
    const auto out_w = g.input("out_w", wdt, {H, H});
    const auto out_s = cfg_.blocks_fp8 ? g.input("out_s", DType::F32, {H}) : kNoTensor;
    const auto fc1_w = g.input("fc1_w", wdt, {2 * F, H});
    const auto fc1_s = cfg_.blocks_fp8 ? g.input("fc1_s", DType::F32, {2 * F}) : kNoTensor;
    const auto fc2_w = g.input("fc2_w", wdt, {H, F});
    const auto fc2_s = cfg_.blocks_fp8 ? g.input("fc2_s", DType::F32, {H}) : kNoTensor;

    auto lin = [&](TensorId a, TensorId w, TensorId ws, const std::string& nm) {
        return cfg_.blocks_fp8 ? o.fp8_linear(a, w, ws, nm) : o.linear_bf16(a, w, nm);
    };

    // 1. Attention
    const auto h1 = modulate_split(o, layer_norm_no_affine(o, x, cfg_.eps, "ln1"), s1_0, s1_1, P, T, "m1");
    const auto qkv = lin(h1, qkv_w, qkv_s, "qkv");
    const auto qsp = o.split(qkv, 1, {H, H, H}, "qsp");
    auto q = o.rms_norm(o.reshape(qsp[0], {S, NH, HD}, "qr"), norm_q, cfg_.eps, false, "qn");
    auto k = o.rms_norm(o.reshape(qsp[1], {S, NH, HD}, "kr"), norm_k, cfg_.eps, false, "kn");
    const auto v = o.reshape(qsp[2], {S, NH, HD}, "vr");
    q = apply_interleaved_rope(o, q, cos_tab, sin_tab, "qrope");
    k = apply_interleaved_rope(o, k, cos_tab, sin_tab, "krope");

    const auto q4 = o.reshape(o.transpose(q, {1, 0, 2}, "qt"), {1, NH, S, HD}, "q4");
    const auto k4 = o.reshape(o.transpose(k, {1, 0, 2}, "kt"), {1, NH, S, HD}, "k4");
    const auto v4 = o.reshape(o.transpose(v, {1, 0, 2}, "vt"), {1, NH, S, HD}, "v4");

    // Emit prefix K/V cache [1, NH, P, HD] BF16 for Steps 1..7 decode
    const auto k_pre = o.split(k4, 2, {P, T}, "kpre_sp")[0];
    const auto v_pre = o.split(v4, 2, {P, T}, "vpre_sp")[0];
    set_out(g, "k_pre", k_pre);
    set_out(g, "v_pre", v_pre);

    const auto att = o.temp("att", DType::BF16, {1, NH, S, HD});
    ns_Sdpa::ParamsV7 sp{};
    sp.scale = float(1.0 / std::sqrt(double(HD)));
    sp.is_causal = false;
    sp.is_inference = true;
    sp.softmax_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
    sp.wl = sp.wr = -1;
    std::vector<TensorId> sd_in = {q4, k4, v4, mask, kNoTensor};
    for (int i = 0; i < 7; ++i) sd_in.push_back(kNoTensor);
    g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(sd_in), std::span<const TensorId>(&att, 1), &sp, sizeof(sp));

    const auto ctx = o.reshape(o.transpose(o.reshape(att, {NH, S, HD}, "a3"), {1, 0, 2}, "at"), {S, H}, "af");
    const auto aout = lin(ctx, out_w, out_s, "out");
    const auto x1 = o.add(x, modulate_split(o, aout, g1_0, g1_1, P, T, "ag"), "res1");

    // 2. SwiGLU MLP (uses fused fp8_gemm_silu_mul_quant when FP8!)
    const auto h2 = modulate_split(o, layer_norm_no_affine(o, x1, cfg_.eps, "ln2"), s2_0, s2_1, P, T, "m2");
    TensorId mout;
    if (cfg_.blocks_fp8) {
        const auto [hc, hs] = o.quantize(h2, "hq");
        const auto [mc, ms] = o.fp8_gemm_silu_mul_quant(hc, hs, fc1_w, fc1_s, "fc1");
        mout = o.fp8_gemm(mc, ms, fc2_w, fc2_s, "fc2");
    } else {
        const auto gu = o.linear_bf16(h2, fc1_w, "fc1");
        const auto gup = o.split(gu, 1, {F, F}, "gup");
        mout = o.linear_bf16(o.mul(o.silu(gup[0], "gs"), gup[1], "act"), fc2_w, "fc2");
    }
    set_out(g, "out", o.add(x1, modulate_split(o, mout, g2_0, g2_1, P, T, "mg"), "res2"));

    auto r = g.compile("qwen21_blk_pre_" + std::to_string(P) + "_" + std::to_string(T));
    auto& ref = *r;
    r_blk_pre_.emplace(key, std::move(r));
    return ref;
}

Recipe& Qwen21DiT::get_blk_decode_recipe(uint32_t prefix_len, uint32_t target_len) {
    const auto key = std::make_pair(prefix_len, target_len);
    if (auto it = r_blk_dec_.find(key); it != r_blk_dec_.end()) return *it->second;
    const uint64_t P = prefix_len, T = target_len;
    const uint64_t H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.head_dim, F = cfg_.ffn_hidden;

    GraphBuilder g(dev_);
    Ops o(g, "qbd_");
    const auto x = g.input("x", DType::BF16, {T, H});
    const auto cos_tab = g.input("cos", DType::F32, {T, HD / 2});
    const auto sin_tab = g.input("sin", DType::F32, {T, HD / 2});
    const auto k_pre = g.input("k_pre", DType::BF16, {1, NH, P, HD});
    const auto v_pre = g.input("v_pre", DType::BF16, {1, NH, P, HD});
    const auto s1_0 = g.input("s1_0", DType::BF16, {1, H});
    const auto g1_0 = g.input("g1_0", DType::BF16, {1, H});
    const auto s2_0 = g.input("s2_0", DType::BF16, {1, H});
    const auto g2_0 = g.input("g2_0", DType::BF16, {1, H});

    const auto norm_q = g.input("norm_q", DType::BF16, {HD});
    const auto norm_k = g.input("norm_k", DType::BF16, {HD});
    const DType wdt = cfg_.blocks_fp8 ? DType::F8 : DType::BF16;
    const auto qkv_w = g.input("qkv_w", wdt, {3 * H, H});
    const auto qkv_s = cfg_.blocks_fp8 ? g.input("qkv_s", DType::F32, {3 * H}) : kNoTensor;
    const auto out_w = g.input("out_w", wdt, {H, H});
    const auto out_s = cfg_.blocks_fp8 ? g.input("out_s", DType::F32, {H}) : kNoTensor;
    const auto fc1_w = g.input("fc1_w", wdt, {2 * F, H});
    const auto fc1_s = cfg_.blocks_fp8 ? g.input("fc1_s", DType::F32, {2 * F}) : kNoTensor;
    const auto fc2_w = g.input("fc2_w", wdt, {H, F});
    const auto fc2_s = cfg_.blocks_fp8 ? g.input("fc2_s", DType::F32, {H}) : kNoTensor;

    auto lin = [&](TensorId a, TensorId w, TensorId ws, const std::string& nm) {
        return cfg_.blocks_fp8 ? o.fp8_linear(a, w, ws, nm) : o.linear_bf16(a, w, nm);
    };

    // 1. Unmasked Decode Attention over [cached_prefix_KV | target_KV]
    const auto h1 = o.mul(layer_norm_no_affine(o, x, cfg_.eps, "ln1"), s1_0, "m1");
    const auto qkv = lin(h1, qkv_w, qkv_s, "qkv");
    const auto qsp = o.split(qkv, 1, {H, H, H}, "qsp");
    auto q = o.rms_norm(o.reshape(qsp[0], {T, NH, HD}, "qr"), norm_q, cfg_.eps, false, "qn");
    auto k = o.rms_norm(o.reshape(qsp[1], {T, NH, HD}, "kr"), norm_k, cfg_.eps, false, "kn");
    const auto v = o.reshape(qsp[2], {T, NH, HD}, "vr");
    q = apply_interleaved_rope(o, q, cos_tab, sin_tab, "qrope");
    k = apply_interleaved_rope(o, k, cos_tab, sin_tab, "krope");

    const auto q4 = o.reshape(o.transpose(q, {1, 0, 2}, "qt"), {1, NH, T, HD}, "q4");
    const auto kt4 = o.reshape(o.transpose(k, {1, 0, 2}, "kt"), {1, NH, T, HD}, "kt4");
    const auto vt4 = o.reshape(o.transpose(v, {1, 0, 2}, "vt"), {1, NH, T, HD}, "vt4");
    const auto k_full = o.concat({k_pre, kt4}, 2, "kfull");
    const auto v_full = o.concat({v_pre, vt4}, 2, "vfull");

    const auto att = o.temp("att", DType::BF16, {1, NH, T, HD});
    ns_Sdpa::ParamsV7 sp{};
    sp.scale = float(1.0 / std::sqrt(double(HD)));
    sp.is_causal = false;
    sp.is_inference = true;
    const bool fast_sm = (std::getenv("G2E_QWEN21_EXACT_SOFTMAX") == nullptr);
    sp.softmax_mode = fast_sm ? SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C : SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
    sp.wl = sp.wr = -1;
    std::vector<TensorId> sd_in = {q4, k_full, v_full, kNoTensor, kNoTensor};
    for (int i = 0; i < 7; ++i) sd_in.push_back(kNoTensor);
    g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(sd_in), std::span<const TensorId>(&att, 1), &sp, sizeof(sp));

    const auto ctx = o.reshape(o.transpose(o.reshape(att, {NH, T, HD}, "a3"), {1, 0, 2}, "at"), {T, H}, "af");
    const auto aout = lin(ctx, out_w, out_s, "out");
    const auto x1 = o.add(x, o.mul(aout, g1_0, "ag"), "res1");

    // 2. SwiGLU MLP
    const auto h2 = o.mul(layer_norm_no_affine(o, x1, cfg_.eps, "ln2"), s2_0, "m2");
    TensorId mout;
    if (cfg_.blocks_fp8) {
        const auto [hc, hs] = o.quantize(h2, "hq");
        const auto [mc, ms] = o.fp8_gemm_silu_mul_quant(hc, hs, fc1_w, fc1_s, "fc1");
        mout = o.fp8_gemm(mc, ms, fc2_w, fc2_s, "fc2");
    } else {
        const auto gu = o.linear_bf16(h2, fc1_w, "fc1");
        const auto gup = o.split(gu, 1, {F, F}, "gup");
        mout = o.linear_bf16(o.mul(o.silu(gup[0], "gs"), gup[1], "act"), fc2_w, "fc2");
    }
    set_out(g, "out", o.add(x1, o.mul(mout, g2_0, "mg"), "res2"));

    auto r = g.compile("qwen21_blk_dec_" + std::to_string(P) + "_" + std::to_string(T));
    auto& ref = *r;
    r_blk_dec_.emplace(key, std::move(r));
    return ref;
}

Recipe& Qwen21DiT::get_final_recipe(uint32_t target_len) {
    if (auto it = r_final_.find(target_len); it != r_final_.end()) return *it->second;
    const uint64_t H = cfg_.hidden_size, C = cfg_.out_channels;
    GraphBuilder g(dev_);
    Ops o(g, "qfin_");
    const auto x = g.input("x", DType::BF16, {target_len, H});
    const auto fsc = g.input("fsc", DType::BF16, {1, H});
    const auto proj_w = g.input("proj_w", DType::BF16, {C, H});
    const auto h = o.mul(layer_norm_no_affine(o, x, cfg_.eps, "ln"), fsc, "sc");
    const auto out = o.linear_bf16(h, proj_w, "proj");
    set_out(g, "out", o.cast(out, DType::F32, "outf"));
    auto r = g.compile("qwen21_final_" + std::to_string(target_len));
    auto& ref = *r;
    r_final_.emplace(target_len, std::move(r));
    return ref;
}

double Qwen21DiT::denoise(const uint16_t* prompt_embeds, uint32_t vlm_len, const uint8_t* vlm_img_mask,
                          const float* cond_latents, uint32_t cond_tokens,
                          const std::vector<Qwen21ImgShape>& img_shapes,
                          std::vector<float>& target_latents,
                          const std::vector<float>& sigmas_in,
                          Qwen21StepDebug* debug_step0,
                          Qwen21StepDebug* debug_step1) {
    const auto t_start = Clock::now();
    if (img_shapes.empty()) throw std::invalid_argument("Qwen21DiT::denoise: empty img_shapes");
    const uint32_t T = img_shapes.back().tokens();
    const uint32_t total_img_tokens = cond_tokens + T;
    if (target_latents.size() != size_t(T) * cfg_.in_channels) {
        throw std::invalid_argument("Qwen21DiT::denoise: target_latents size mismatch");
    }

    // 1. Expand image_pad_mask (each VLM image slot -> 4 latent tokens) + append target slots (T/4)
    std::vector<uint8_t> exp_mask;
    std::vector<int32_t> gather_idx;
    exp_mask.reserve(vlm_len + total_img_tokens);
    gather_idx.reserve(vlm_len + total_img_tokens);
    uint32_t img_cursor = 0;
    for (uint32_t i = 0; i < vlm_len; ++i) {
        if (vlm_img_mask[i]) {
            for (int r = 0; r < 4; ++r) {
                exp_mask.push_back(1);
                gather_idx.push_back(int32_t(vlm_len + img_cursor++));
            }
        } else {
            exp_mask.push_back(0);
            gather_idx.push_back(int32_t(i));
        }
    }
    if (img_cursor != cond_tokens) {
        throw std::runtime_error("Qwen21DiT: cond_tokens (" + std::to_string(cond_tokens) +
                                 ") != 4 * count(vlm_img_mask) (" + std::to_string(img_cursor) + ")");
    }
    for (uint32_t i = 0; i < T; ++i) {
        exp_mask.push_back(1);
        gather_idx.push_back(int32_t(vlm_len + cond_tokens + i));
    }
    const uint32_t S = uint32_t(exp_mask.size());
    const uint32_t P = S - T;

    // 2. Build 3D RoPE cos/sin [S, 64] F32 & block-causal attention mask [1, 1, S, S] BF16
    std::vector<float> h_cos, h_sin;
    build_qwen21_rope(img_shapes, exp_mask, cfg_.axes_dims_rope, h_cos, h_sin);

    std::vector<int32_t> image_ids(S, -1);
    {
        size_t pos = 0;
        for (size_t b = 0; b < img_shapes.size(); ++b) {
            const uint32_t nt = img_shapes[b].tokens();
            uint32_t assigned = 0;
            while (pos < S && assigned < nt) {
                if (exp_mask[pos]) {
                    image_ids[pos] = int32_t(b);
                    ++assigned;
                }
                ++pos;
            }
        }
    }
    const uint16_t bf_zero = to_bf16(0.0f);
    const uint16_t bf_ninf = to_bf16(-1e30f);
    std::vector<uint16_t> h_mask(size_t(S) * S, bf_ninf);
    for (uint32_t qi = 0; qi < S; ++qi) {
        uint16_t* row = h_mask.data() + size_t(qi) * S;
        const int32_t qid = image_ids[qi];
        for (uint32_t ki = 0; ki <= qi; ++ki) row[ki] = bf_zero;
        if (qid >= 0) {
            for (uint32_t ki = qi + 1; ki < S; ++ki) {
                if (image_ids[ki] == qid) row[ki] = bf_zero;
            }
        }
    }

    // 3. Allocate device buffers + per-layer prefix KV cache
    const uint64_t H = cfg_.hidden_size, C = cfg_.in_channels, NH = cfg_.num_heads, HD = cfg_.head_dim, D2 = HD / 2;
    DeviceBuffer d_enc(dev_, uint64_t(vlm_len) * H * 2);
    DeviceBuffer d_gidx(dev_, uint64_t(S) * 4);
    DeviceBuffer d_cos(dev_, uint64_t(S) * D2 * 4);
    DeviceBuffer d_sin(dev_, uint64_t(S) * D2 * 4);
    DeviceBuffer d_mask(dev_, uint64_t(S) * S * 2);
    DeviceBuffer d_lat_all(dev_, uint64_t(total_img_tokens) * C * 2);
    DeviceBuffer d_lat_tgt(dev_, uint64_t(T) * C * 2);
    DeviceBuffer d_x[2] = {DeviceBuffer(dev_, uint64_t(S) * H * 2), DeviceBuffer(dev_, uint64_t(S) * H * 2)};
    DeviceBuffer d_xt[2] = {DeviceBuffer(dev_, uint64_t(T) * H * 2), DeviceBuffer(dev_, uint64_t(T) * H * 2)};
    DeviceBuffer d_pred(dev_, uint64_t(T) * C * 4);
    DeviceBuffer d_tin(dev_, 2ull * 256 * 2);
    DeviceBuffer d_s1_0(dev_, H * 2), d_s1_1(dev_, H * 2);
    DeviceBuffer d_g1_0(dev_, H * 2), d_g1_1(dev_, H * 2);
    DeviceBuffer d_s2_0(dev_, H * 2), d_s2_1(dev_, H * 2);
    DeviceBuffer d_g2_0(dev_, H * 2), d_g2_1(dev_, H * 2);
    DeviceBuffer d_fsc(dev_, H * 2);

    const uint64_t kv_bytes = uint64_t(NH) * P * HD * 2;
    std::vector<DeviceBuffer> d_k_cache, d_v_cache;
    d_k_cache.reserve(cfg_.num_layers);
    d_v_cache.reserve(cfg_.num_layers);
    for (uint32_t L = 0; L < cfg_.num_layers; ++L) {
        d_k_cache.emplace_back(dev_, kv_bytes);
        d_v_cache.emplace_back(dev_, kv_bytes);
    }

    // Upload static request inputs once
    {
        HostBuffer h_enc_buf(dev_, d_enc.size());
        std::memcpy(h_enc_buf.data(), prompt_embeds, d_enc.size());
        stream_.upload(h_enc_buf, 0, d_enc, 0, d_enc.size());

        HostBuffer h_gidx_buf(dev_, d_gidx.size());
        std::memcpy(h_gidx_buf.data(), gather_idx.data(), d_gidx.size());
        stream_.upload(h_gidx_buf, 0, d_gidx, 0, d_gidx.size());

        HostBuffer h_cs_buf(dev_, d_cos.size() * 2);
        std::memcpy(h_cs_buf.data(), h_cos.data(), d_cos.size());
        std::memcpy(h_cs_buf.as<uint8_t>() + d_cos.size(), h_sin.data(), d_sin.size());
        stream_.upload(h_cs_buf, 0, d_cos, 0, d_cos.size());
        stream_.upload(h_cs_buf, d_cos.size(), d_sin, 0, d_sin.size());

        HostBuffer h_m_buf(dev_, d_mask.size());
        std::memcpy(h_m_buf.data(), h_mask.data(), d_mask.size());
        stream_.upload(h_m_buf, 0, d_mask, 0, d_mask.size());
        stream_.synchronize();
    }

    // Build sigma schedule with terminal 0.0
    std::vector<float> sigmas = sigmas_in;
    if (sigmas.empty()) sigmas.assign(kTurboSigmas.begin(), kTurboSigmas.end());
    const uint32_t n_steps = uint32_t(sigmas.size());
    std::vector<float> sigmas_ext = sigmas;
    sigmas_ext.push_back(0.0f);

    HostBuffer h_tin(dev_, d_tin.size());
    HostBuffer h_lat_all(dev_, d_lat_all.size());
    HostBuffer h_lat_tgt(dev_, d_lat_tgt.size());
    HostBuffer h_pred(dev_, d_pred.size());

    auto& r_mod = get_mod_recipe();
    auto& r_ep = get_entry_prefill_recipe(vlm_len, total_img_tokens, S);
    auto& r_ed = get_entry_decode_recipe(T);
    auto& r_bp = get_blk_prefill_recipe(P, T);
    auto& r_bd = get_blk_decode_recipe(P, T);
    auto& r_fin = get_final_recipe(T);

    auto bind_blk_weights = [&](std::vector<Binding>& b, uint32_t L) {
        const std::string p = "B" + std::to_string(L) + ".";
        b.push_back(w_.bind("norm_q", p + "norm_q"));
        b.push_back(w_.bind("norm_k", p + "norm_k"));
        b.push_back(w_.bind("qkv_w", p + "qkv.w"));
        b.push_back(w_.bind("out_w", p + "out.w"));
        b.push_back(w_.bind("fc1_w", p + "fc1.w"));
        b.push_back(w_.bind("fc2_w", p + "fc2.w"));
        if (cfg_.blocks_fp8) {
            b.push_back(w_.bind("qkv_s", p + "qkv.s"));
            b.push_back(w_.bind("out_s", p + "out.s"));
            b.push_back(w_.bind("fc1_s", p + "fc1.s"));
            b.push_back(w_.bind("fc2_s", p + "fc2.s"));
        }
    };

    const uint64_t tgt_cos_off = uint64_t(P) * D2 * 4;
    const uint64_t tgt_x_off = uint64_t(P) * H * 2;
    std::vector<float> pred_f32(size_t(T) * C);

    for (uint32_t step = 0; step < n_steps; ++step) {
        const auto ts0 = Clock::now();
        const float sigma = sigmas_ext[step];
        const float dt = sigmas_ext[step + 1] - sigma;

        // Upload [sigma, 0.0] sinusoidal embedding -> shared modulation
        auto* tin_p = h_tin.as<uint16_t>();
        compute_timestep_sincos(sigma, tin_p);
        compute_timestep_sincos(0.0f, tin_p + 256);
        stream_.upload(h_tin, 0, d_tin, 0, d_tin.size());
        {
            std::vector<Binding> bm = {
                {"t_in", &d_tin, 0},
                w_.bind("l1", "time_embed.linear_1.w"),
                w_.bind("l2", "time_embed.linear_2.w"),
                w_.bind("mod_w", "modulation.w"),
                w_.bind("nout_w", "norm_out.linear.w"),
                {"s1_0", &d_s1_0, 0}, {"s1_1", &d_s1_1, 0},
                {"g1_0", &d_g1_0, 0}, {"g1_1", &d_g1_1, 0},
                {"s2_0", &d_s2_0, 0}, {"s2_1", &d_s2_1, 0},
                {"g2_0", &d_g2_0, 0}, {"g2_1", &d_g2_1, 0},
                {"fsc", &d_fsc, 0},
            };
            r_mod.enqueue(stream_, bm);
        }

        if (step == 0) {
            // Prefill: cond_latents + target_latents -> joint [S, H] -> 32 blocks (populate prefix KV cache)
            auto* lp = h_lat_all.as<uint16_t>();
            for (size_t i = 0; i < size_t(cond_tokens) * C; ++i) lp[i] = to_bf16(cond_latents[i]);
            for (size_t i = 0; i < size_t(T) * C; ++i) lp[size_t(cond_tokens) * C + i] = to_bf16(target_latents[i]);
            stream_.upload(h_lat_all, 0, d_lat_all, 0, d_lat_all.size());

            {
                std::vector<Binding> be = {
                    {"enc_hs", &d_enc, 0},
                    {"lat", &d_lat_all, 0},
                    {"gather_idx", &d_gidx, 0},
                    w_.bind("txt_nw", "txt_in.text_norm"),
                    w_.bind("txt_in_w", "txt_in.in_layer.w"),
                    w_.bind("txt_out_w", "txt_in.out_layer.w"),
                    w_.bind("img_in_w", "img_in.w"),
                    {"joint", &d_x[0], 0},
                };
                r_ep.enqueue(stream_, be);
            }
            int hp = 0;
            for (uint32_t L = 0; L < cfg_.num_layers; ++L) {
                std::vector<Binding> bb = {
                    {"x", &d_x[hp], 0},
                    {"cos", &d_cos, 0},
                    {"sin", &d_sin, 0},
                    {"mask", &d_mask, 0},
                    {"s1_0", &d_s1_0, 0}, {"s1_1", &d_s1_1, 0},
                    {"g1_0", &d_g1_0, 0}, {"g1_1", &d_g1_1, 0},
                    {"s2_0", &d_s2_0, 0}, {"s2_1", &d_s2_1, 0},
                    {"g2_0", &d_g2_0, 0}, {"g2_1", &d_g2_1, 0},
                    {"k_pre", &d_k_cache[L], 0},
                    {"v_pre", &d_v_cache[L], 0},
                    {"out", &d_x[1 - hp], 0},
                };
                bind_blk_weights(bb, L);
                r_bp.enqueue(stream_, bb);
                hp = 1 - hp;
                if (L == 0 && debug_step0) {
                    HostBuffer hb(dev_, uint64_t(T) * H * 2);
                    stream_.download(d_x[hp], tgt_x_off, hb, 0, hb.size());
                    stream_.synchronize();
                    debug_step0->b0_target.resize(size_t(T) * H);
                    const auto* src = hb.as<uint16_t>();
                    for (size_t i = 0; i < debug_step0->b0_target.size(); ++i) debug_step0->b0_target[i] = from_bf16(src[i]);
                }
            }
            // Copy target slice [P..S) to d_xt[0] (aligned at offset 0 for final recipe)
            stream_.copy(d_x[hp], tgt_x_off, d_xt[0], 0, uint64_t(T) * H * 2);
            std::vector<Binding> bf = {
                {"x", &d_xt[0], 0},
                {"fsc", &d_fsc, 0},
                w_.bind("proj_w", "proj_out.w"),
                {"out", &d_pred, 0},
            };
            r_fin.enqueue(stream_, bf);
        } else {
            // Decode: target tokens ONLY, attending to [cached_prefix_KV | target_KV] (unmasked!)
            auto* lp = h_lat_tgt.as<uint16_t>();
            for (size_t i = 0; i < size_t(T) * C; ++i) lp[i] = to_bf16(target_latents[i]);
            stream_.upload(h_lat_tgt, 0, d_lat_tgt, 0, d_lat_tgt.size());

            {
                std::vector<Binding> be = {
                    {"lat", &d_lat_tgt, 0},
                    w_.bind("img_in_w", "img_in.w"),
                    {"out", &d_xt[0], 0},
                };
                r_ed.enqueue(stream_, be);
            }
            int hp = 0;
            for (uint32_t L = 0; L < cfg_.num_layers; ++L) {
                std::vector<Binding> bb = {
                    {"x", &d_xt[hp], 0},
                    {"cos", &d_cos, tgt_cos_off},
                    {"sin", &d_sin, tgt_cos_off},
                    {"k_pre", &d_k_cache[L], 0},
                    {"v_pre", &d_v_cache[L], 0},
                    {"s1_0", &d_s1_0, 0},
                    {"g1_0", &d_g1_0, 0},
                    {"s2_0", &d_s2_0, 0},
                    {"g2_0", &d_g2_0, 0},
                    {"out", &d_xt[1 - hp], 0},
                };
                bind_blk_weights(bb, L);
                r_bd.enqueue(stream_, bb);
                hp = 1 - hp;
                if (L == 0 && step == 1 && debug_step1) {
                    HostBuffer hb(dev_, uint64_t(T) * H * 2);
                    stream_.download(d_xt[hp], 0, hb, 0, hb.size());
                    stream_.synchronize();
                    debug_step1->b0_target.resize(size_t(T) * H);
                    const auto* src = hb.as<uint16_t>();
                    for (size_t i = 0; i < debug_step1->b0_target.size(); ++i) debug_step1->b0_target[i] = from_bf16(src[i]);
                }
            }
            std::vector<Binding> bf = {
                {"x", &d_xt[hp], 0},
                {"fsc", &d_fsc, 0},
                w_.bind("proj_w", "proj_out.w"),
                {"out", &d_pred, 0},
            };
            r_fin.enqueue(stream_, bf);
        }

        stream_.download(d_pred, 0, h_pred, 0, d_pred.size());
        stream_.synchronize();
        std::memcpy(pred_f32.data(), h_pred.data(), d_pred.size());

        if (step == 0 && debug_step0) debug_step0->noise_pred = pred_f32;
        if (step == 1 && debug_step1) debug_step1->noise_pred = pred_f32;

        // FlowMatchEulerDiscreteScheduler step: x = x + (sigma_next - sigma) * pred (computed in F32)
        for (size_t i = 0; i < pred_f32.size(); ++i) {
            target_latents[i] += dt * pred_f32[i];
        }
        std::printf("  [Qwen21DiT] step %u/%u (%s, P=%u T=%u) sigma=%.4f->%.4f in %.1f ms\n",
                    step + 1, n_steps, step == 0 ? "prefill" : "kv-dec", P, T, sigma, sigmas_ext[step + 1], ms_since(ts0));
    }
    const double total_ms = ms_since(t_start);
    std::printf("[Qwen21DiT] denoised %u steps (P=%u, T=%u) in %.2f s\n", n_steps, P, T, total_ms / 1000.0);
    return total_ms / 1000.0;
}

}  // namespace g2e
