#include "g2e/qwen21_vae.hpp"
#include "g2e/numeric.hpp"

#include <nlohmann/json.hpp>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace g2e {

namespace {

void set_out(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const auto out = g.input(name, d.dtype, d.shape);
    g.op("identity", {src}, {out});
}

// QwenImage21RMS_norm on NHWC [1, H, W, C] BF16:
// F.normalize(x.float(), dim=C).to(bf16) * sqrt(C) * gamma
TensorId qwen_rms_norm_nhwc(Ops& o, TensorId x, TensorId gamma, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t C = sh[3];
    const auto xf = o.cast(x, DType::F32, nm + "_f");
    const auto sq_sum = o.reduce("sum", o.mul(xf, xf, nm + "_sq"), 3, nm + "_ss");
    const auto l2 = o.unary("sqrt", sq_sum, nm + "_l2");
    const auto denom = o.binary("max", l2, o.constant(1e-12f, {1}, nm + "_eps", DType::F32), nm + "_dn");
    const auto inv = o.unary("reciprocal", denom, nm + "_inv");
    const auto norm_b = o.cast(o.mul(xf, inv, nm + "_nf"), DType::BF16, nm + "_nb");
    const auto sc = o.constant(std::sqrt(float(C)), {1}, nm + "_sc", DType::BF16);
    const auto g4 = o.reshape(gamma, {1, 1, 1, C}, nm + "_g");
    return o.mul(o.mul(norm_b, sc, nm + "_ns"), g4, nm);
}

// 2D NHWC spatial_convolution in BF16: x [1, H, W, Cin], w [kH, kW, Cin, Cout], b [Cout].
TensorId conv2d_nhwc_bf16(Ops& o, TensorId x, TensorId w, TensorId b,
                          uint32_t stride, uint32_t pad_l, uint32_t pad_r, uint32_t pad_t, uint32_t pad_b,
                          const std::string& nm) {
    const auto xs = o.shape(x);
    const auto ws = o.shape(w);
    const uint64_t B = xs[0], H = xs[1], W = xs[2], kH = ws[0], kW = ws[1], Cout = ws[3];
    const uint64_t H_out = (H + pad_t + pad_b - kH) / stride + 1;
    const uint64_t W_out = (W + pad_l + pad_r - kW) / stride + 1;
    const auto y = o.temp(nm + "_c", DType::BF16, {B, H_out, W_out, Cout});
    synConvolutionParamsV2 p{};
    p.kH = uint32_t(kH);
    p.kW = uint32_t(kW);
    p.dH = stride;
    p.dW = stride;
    p.padL = int(pad_l);
    p.padR = int(pad_r);
    p.padT = int(pad_t);
    p.padB = int(pad_b);
    p.dilH = p.dilW = 1;
    p.nGroups = 1;
    p.paddingType = PADDING_EXPLICIT;
    o.graph().op("spatial_convolution", {x, w}, {y}, &p, sizeof(p));
    return o.add(y, o.reshape(b, {1, 1, 1, Cout}, nm + "_b"), nm);
}

// 1x1 conv on NHWC [1, H, W, Cin] via MME BF16 GEMM (w [Cout, Cin], b [Cout]).
TensorId conv1x1_nhwc_bf16(Ops& o, TensorId x, TensorId w, TensorId b, const std::string& nm) {
    const auto xs = o.shape(x);
    const uint64_t B = xs[0], H = xs[1], W = xs[2], Cin = xs[3], Cout = o.shape(w)[0];
    const auto x2 = o.reshape(x, {B * H * W, Cin}, nm + "_r2");
    const auto y2 = o.linear_bf16(x2, w, nm + "_mm");
    const auto y4 = o.reshape(y2, {B, H, W, Cout}, nm + "_r4");
    return o.add(y4, o.reshape(b, {1, 1, 1, Cout}, nm + "_b"), nm);
}

// Exact nearest-exact 2x spatial upsample on NHWC [B, H, W, C] (max tensor rank 5).
TensorId upsample2x_nhwc(Ops& o, TensorId x, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t B = sh[0], H = sh[1], W = sh[2], C = sh[3];
    const auto xw1 = o.reshape(x, {B, H, W, 1, C}, nm + "_w1");
    const auto xw2 = o.reshape(o.concat({xw1, xw1}, 3, nm + "_wc"), {B, H, 2 * W, C}, nm + "_w2");
    const auto xh1 = o.reshape(xw2, {B, H, 1, 2 * W, C}, nm + "_h1");
    return o.reshape(o.concat({xh1, xh1}, 2, nm + "_hc"), {B, 2 * H, 2 * W, C}, nm);
}

// Exact QwenImage21DupUp3D(..., first_chunk=True) for single-frame T=1 on NHWC [B, H, W, in_c].
TensorId dup_up_nhwc(Ops& o, TensorId x, uint32_t in_c, uint32_t out_c, bool temporal_up, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t B = sh[0], H = sh[1], W = sh[2];
    if (in_c == out_c) {
        // repeats = 8 (ft=2, fs=2): every (c, ft, dh, dw) maps directly to input channel c.
        return upsample2x_nhwc(o, x, nm);
    }
    if (in_c == 2 * out_c && temporal_up) {
        // repeats = 4 (ft=2, fs=2): ft=0 -> even channels, ft=1 -> odd channels; first_chunk keeps ft=1.
        const auto x5 = o.reshape(x, {B, H, W, out_c, 2}, nm + "_r5");
        const auto odd = o.reshape(o.split(x5, 4, {1, 1}, nm + "_sp")[1], {B, H, W, out_c}, nm + "_odd");
        return upsample2x_nhwc(o, odd, nm);
    }
    if (in_c == 2 * out_c && !temporal_up) {
        // repeats = 2 (ft=1, fs=2): dh=0 -> even channel 2c, dh=1 -> odd channel 2c+1; replicated across dw=0,1.
        const auto x5 = o.reshape(x, {B, H, W, out_c, 2}, nm + "_r5");
        const auto xt = o.transpose(x5, {0, 1, 4, 2, 3}, nm + "_tr");  // [B, H, 2, W, out_c]
        const auto xh = o.reshape(xt, {B, 2 * H, W, out_c}, nm + "_xh");
        const auto xw1 = o.reshape(xh, {B, 2 * H, W, 1, out_c}, nm + "_w1");
        return o.reshape(o.concat({xw1, xw1}, 3, nm + "_wc"), {B, 2 * H, 2 * W, out_c}, nm);
    }
    throw std::runtime_error("Qwen21Vae: unsupported DupUp3D config");
}

// Exact 2x2 spatial average pool on NHWC [B, H, W, C] (in F32 for precision, cast back to BF16).
TensorId avg_pool2x2_nhwc(Ops& o, TensorId x, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t B = sh[0], H = sh[1], W = sh[2], C = sh[3];
    const auto xf = o.cast(x, DType::F32, nm + "_f");
    const auto xh = o.reshape(xf, {B, H / 2, 2, W, C}, nm + "_rh");
    const auto sh2 = o.split(xh, 2, {1, 1}, nm + "_sh");
    const auto mh = o.add(sh2[0], sh2[1], nm + "_ah");
    const auto xw = o.reshape(mh, {B, H / 2, W / 2, 2, C}, nm + "_rw");
    const auto sw2 = o.split(xw, 3, {1, 1}, nm + "_sw");
    const auto s4 = o.add(sw2[0], sw2[1], nm + "_aw");
    const auto avg = o.mul(o.reshape(s4, {B, H / 2, W / 2, C}, nm + "_r4"),
                           o.constant(0.25f, {1}, nm + "_q", DType::F32), nm + "_avg");
    return o.cast(avg, DType::BF16, nm);
}

// Exact QwenImage21AvgDown3D for single-frame T=1 on NHWC [B, H, W, in_c].
TensorId avg_down_nhwc(Ops& o, TensorId x, uint32_t in_c, uint32_t out_c,
                       bool temporal_down, bool spatial_down, const std::string& nm) {
    if (!spatial_down && !temporal_down && in_c == out_c) return x;
    const auto avg = spatial_down ? avg_pool2x2_nhwc(o, x, nm + "_ap") : x;
    if (!temporal_down && in_c == out_c) return avg;
    if (temporal_down && out_c == 2 * in_c) {
        // pad_t = 1 prepends zero frame -> ft=0 (even out channels 2c) = 0, ft=1 (odd out channels 2c+1) = avg[c].
        const auto sh = o.shape(avg);
        const uint64_t B = sh[0], H2 = sh[1], W2 = sh[2], C = sh[3];
        const auto z = o.constant(0.0f, {B, H2, W2, C, 1}, nm + "_z", DType::BF16);
        const auto a5 = o.reshape(avg, {B, H2, W2, C, 1}, nm + "_a5");
        return o.reshape(o.concat({z, a5}, 4, nm + "_cat"), {B, H2, W2, 2 * C}, nm);
    }
    throw std::runtime_error("Qwen21Vae: unsupported AvgDown3D config");
}

// Single-head spatial self-attention block in mid_block (dim = 768 or 1152; uses explicit MME GEMM + softmax).
TensorId mid_attn_nhwc(Ops& o, TensorId x, const std::function<TensorId(const std::string&)>& W,
                       const std::string& prefix, const std::string& nm) {
    const auto sh = o.shape(x);
    const uint64_t B = sh[0], H = sh[1], Wd = sh[2], C = sh[3], S = H * Wd;
    const auto h = qwen_rms_norm_nhwc(o, x, W(prefix + "norm.gamma"), nm + "_n");
    const auto qkv = conv1x1_nhwc_bf16(o, h, W(prefix + "to_qkv.weight"), W(prefix + "to_qkv.bias"), nm + "_qkv");
    const auto qkv3 = o.reshape(qkv, {B, S, 3 * C}, nm + "_r3");
    const auto sp = o.split(qkv3, 2, {C, C, C}, nm + "_sp");
    const auto sc = o.constant(float(1.0 / std::sqrt(double(C))), {1}, nm + "_sc", DType::F32);
    const auto scores = o.mul(o.batch_gemm(sp[0], sp[1], nm + "_qk", false, true, DType::F32), sc, nm + "_qks");
    const auto probs = o.cast(o.softmax(scores, nm + "_sm"), DType::BF16, nm + "_pb");
    const auto ctx = o.reshape(o.batch_gemm(probs, sp[2], nm + "_pv", false, false, DType::BF16), {B, H, Wd, C}, nm + "_ctx");
    const auto out = conv1x1_nhwc_bf16(o, ctx, W(prefix + "proj.weight"), W(prefix + "proj.bias"), nm + "_proj");
    return o.add(x, out, nm);
}

TensorId resnet_block_nhwc(Ops& o, TensorId x, uint32_t in_c, uint32_t out_c,
                           const std::function<TensorId(const std::string&)>& W,
                           const std::string& prefix, const std::string& nm) {
    TensorId sc = x;
    if (in_c != out_c) {
        sc = conv1x1_nhwc_bf16(o, x, W(prefix + "conv_shortcut.weight"), W(prefix + "conv_shortcut.bias"), nm + "_sc");
    }
    TensorId h = o.silu(qwen_rms_norm_nhwc(o, x, W(prefix + "norm1.gamma"), nm + "_n1"), nm + "_s1");
    h = conv2d_nhwc_bf16(o, h, W(prefix + "conv1.weight"), W(prefix + "conv1.bias"), 1, 1, 1, 1, 1, nm + "_c1");
    h = o.silu(qwen_rms_norm_nhwc(o, h, W(prefix + "norm2.gamma"), nm + "_n2"), nm + "_s2");
    h = conv2d_nhwc_bf16(o, h, W(prefix + "conv2.weight"), W(prefix + "conv2.bias"), 1, 1, 1, 1, 1, nm + "_c2");
    return o.add(sc, h, nm);
}

// Evenly spread tile starts matching AutoencoderKLQwenImage21._tile_starts.
std::vector<uint32_t> tile_starts(uint32_t size, uint32_t tile_min, uint32_t stride) {
    if (size <= tile_min) return {0};
    const uint32_t span = size - tile_min;
    const uint32_t count = (span + stride - 1) / stride + 1;
    const uint32_t base = span / (count - 1);
    const uint32_t extra = span % (count - 1);
    std::vector<uint32_t> starts = {0};
    uint32_t off = 0;
    for (uint32_t n = 0; n + 1 < count; ++n) {
        off += base + (n < extra ? 1u : 0u);
        starts.push_back(off);
    }
    return starts;
}

void blend_v_nhwc(const std::vector<float>& above, std::vector<float>& cur,
                  uint32_t th, uint32_t tw, uint32_t C, uint32_t ov) {
    ov = std::min(ov, th);
    for (uint32_t y = 0; y < ov; ++y) {
        const float w = 0.5f - 0.5f * std::cos(float(M_PI) * (float(y) + 0.5f) / float(ov));
        const float wa = 1.0f - w;
        const float* pa = above.data() + size_t(th - ov + y) * tw * C;
        float* pb = cur.data() + size_t(y) * tw * C;
        for (size_t i = 0; i < size_t(tw) * C; ++i) pb[i] = pa[i] * wa + pb[i] * w;
    }
}

void blend_h_nhwc(const std::vector<float>& left, std::vector<float>& cur,
                  uint32_t th, uint32_t tw, uint32_t C, uint32_t ov) {
    ov = std::min(ov, tw);
    for (uint32_t y = 0; y < th; ++y) {
        const float* la = left.data() + size_t(y) * tw * C;
        float* cb = cur.data() + size_t(y) * tw * C;
        for (uint32_t x = 0; x < ov; ++x) {
            const float w = 0.5f - 0.5f * std::cos(float(M_PI) * (float(x) + 0.5f) / float(ov));
            const float wa = 1.0f - w;
            const float* pa = la + size_t(tw - ov + x) * C;
            float* pb = cb + size_t(x) * C;
            for (uint32_t c = 0; c < C; ++c) pb[c] = pa[c] * wa + pb[c] * w;
        }
    }
}

}  // namespace

Qwen21VaeConfig Qwen21VaeConfig::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& c = j.at("config");
    Qwen21VaeConfig cfg;
    cfg.base_dim = c.value("base_dim", 96u);
    cfg.decoder_base_dim = c.value("decoder_base_dim", 144u);
    cfg.z_dim = c.value("z_dim", 64u);
    cfg.in_channels = c.value("in_channels", 4u);
    cfg.out_channels = c.value("out_channels", 4u);
    cfg.num_res_blocks = c.value("num_res_blocks", 2u);
    cfg.dim_mult = c.at("dim_mult").get<std::vector<uint32_t>>();
    cfg.temporal_downsample = c.at("temperal_downsample").get<std::vector<bool>>();
    cfg.spatial_compression = 1u << cfg.temporal_downsample.size();
    cfg.latents_mean = c.at("latents_mean").get<std::vector<float>>();
    cfg.latents_std = c.at("latents_std").get<std::vector<float>>();
    return cfg;
}

Qwen21Vae::Qwen21Vae(const Device& device, const Stream& stream, const WeightPack& weights)
    : dev_(device), stream_(stream), w_(weights), cfg_(Qwen21VaeConfig::from_pack(weights)) {}

Qwen21Vae::~Qwen21Vae() = default;

Recipe& Qwen21Vae::get_dec_recipe(uint32_t lh, uint32_t lw) {
    const auto key = std::make_pair(lh, lw);
    if (auto it = r_dec_.find(key); it != r_dec_.end()) return *it->second;

    GraphBuilder g(dev_);
    Ops o(g, "qvd_");
    auto W = [&](const std::string& name) {
        const auto& e = w_.at(name);
        return g.input(name, e.dtype, e.shape);
    };

    const uint32_t n_stages = uint32_t(cfg_.dim_mult.size());
    std::vector<uint32_t> dims;
    dims.push_back(cfg_.decoder_base_dim * cfg_.dim_mult.back());
    for (int i = int(n_stages) - 1; i >= 0; --i) dims.push_back(cfg_.decoder_base_dim * cfg_.dim_mult[i]);
    std::vector<bool> temp_up(cfg_.temporal_downsample.rbegin(), cfg_.temporal_downsample.rend());

    TensorId h = g.input("z", DType::BF16, {1, lh, lw, cfg_.z_dim});
    h = conv1x1_nhwc_bf16(o, h, W("post_quant_conv.weight"), W("post_quant_conv.bias"), "pqc");
    h = conv2d_nhwc_bf16(o, h, W("decoder.conv_in.weight"), W("decoder.conv_in.bias"), 1, 1, 1, 1, 1, "cin");

    // mid_block
    h = resnet_block_nhwc(o, h, dims[0], dims[0], W, "decoder.mid_block.resnets.0.", "m_r0");
    h = mid_attn_nhwc(o, h, W, "decoder.mid_block.attentions.0.", "m_a0");
    h = resnet_block_nhwc(o, h, dims[0], dims[0], W, "decoder.mid_block.resnets.1.", "m_r1");

    // up_blocks
    for (uint32_t i = 0; i < n_stages; ++i) {
        const uint32_t in_d = dims[i], out_d = dims[i + 1];
        const bool up_flag = (i + 1 < n_stages);
        const bool t_up = up_flag ? temp_up[i] : false;
        const std::string up = "decoder.up_blocks." + std::to_string(i) + ".";
        const std::string nm = "u" + std::to_string(i);
        const TensorId x_copy = h;
        uint32_t cur_d = in_d;
        for (uint32_t r = 0; r < cfg_.num_res_blocks + 1; ++r) {
            h = resnet_block_nhwc(o, h, cur_d, out_d, W, up + "resnets." + std::to_string(r) + ".",
                                  nm + "_r" + std::to_string(r));
            cur_d = out_d;
        }
        if (up_flag) {
            h = upsample2x_nhwc(o, h, nm + "_up2");
            h = conv2d_nhwc_bf16(o, h, W(up + "upsampler.resample.1.weight"), W(up + "upsampler.resample.1.bias"),
                                 1, 1, 1, 1, 1, nm + "_uc");
            const TensorId sc = dup_up_nhwc(o, x_copy, in_d, out_d, t_up, nm + "_dup");
            h = o.add(h, sc, nm + "_add");
        }
    }

    h = o.silu(qwen_rms_norm_nhwc(o, h, W("decoder.norm_out.gamma"), "no"), "so");
    h = conv2d_nhwc_bf16(o, h, W("decoder.conv_out.weight"), W("decoder.conv_out.bias"), 1, 1, 1, 1, 1, "co");
    auto hf = o.cast(h, DType::F32, "cof");
    hf = o.binary("max", hf, o.constant(-1.0f, {1}, "cmin", DType::F32), "cl0");
    hf = o.binary("min", hf, o.constant(1.0f, {1}, "cmax", DType::F32), "cl1");
    set_out(g, "img", hf);

    auto r = g.compile("qwen21_vae_dec_" + std::to_string(lh) + "x" + std::to_string(lw));
    auto& ref = *r;
    r_dec_.emplace(key, std::move(r));
    return ref;
}

Recipe& Qwen21Vae::get_enc_recipe(uint32_t h_px, uint32_t w_px) {
    const auto key = std::make_pair(h_px, w_px);
    if (auto it = r_enc_.find(key); it != r_enc_.end()) return *it->second;

    GraphBuilder g(dev_);
    Ops o(g, "qve_");
    auto W = [&](const std::string& name) {
        const auto& e = w_.at(name);
        return g.input(name, e.dtype, e.shape);
    };

    const uint32_t n_stages = uint32_t(cfg_.dim_mult.size());
    std::vector<uint32_t> dims = {cfg_.base_dim};
    for (auto m : cfg_.dim_mult) dims.push_back(cfg_.base_dim * m);

    TensorId h = g.input("img", DType::BF16, {1, h_px, w_px, cfg_.in_channels});
    h = conv2d_nhwc_bf16(o, h, W("encoder.conv_in.weight"), W("encoder.conv_in.bias"), 1, 1, 1, 1, 1, "cin");

    for (uint32_t i = 0; i < n_stages; ++i) {
        const uint32_t in_d = dims[i], out_d = dims[i + 1];
        const bool down_flag = (i + 1 < n_stages);
        const bool t_down = down_flag ? cfg_.temporal_downsample[i] : false;
        const std::string dp = "encoder.down_blocks." + std::to_string(i) + ".";
        const std::string nm = "d" + std::to_string(i);
        const TensorId x_copy = h;
        uint32_t cur_d = in_d;
        for (uint32_t r = 0; r < cfg_.num_res_blocks; ++r) {
            h = resnet_block_nhwc(o, h, cur_d, out_d, W, dp + "resnets." + std::to_string(r) + ".",
                                  nm + "_r" + std::to_string(r));
            cur_d = out_d;
        }
        if (down_flag) {
            // ZeroPad2d((0, 1, 0, 1)) + stride-2 3x3 conv -> padL=0, padR=1, padT=0, padB=1
            h = conv2d_nhwc_bf16(o, h, W(dp + "downsampler.resample.1.weight"), W(dp + "downsampler.resample.1.bias"),
                                 2, 0, 1, 0, 1, nm + "_ds");
        }
        const TensorId sc = avg_down_nhwc(o, x_copy, in_d, out_d, t_down, down_flag, nm + "_avg");
        h = o.add(h, sc, nm + "_add");
    }

    const uint32_t mid_d = dims.back();
    h = resnet_block_nhwc(o, h, mid_d, mid_d, W, "encoder.mid_block.resnets.0.", "m_r0");
    h = mid_attn_nhwc(o, h, W, "encoder.mid_block.attentions.0.", "m_a0");
    h = resnet_block_nhwc(o, h, mid_d, mid_d, W, "encoder.mid_block.resnets.1.", "m_r1");

    h = o.silu(qwen_rms_norm_nhwc(o, h, W("encoder.norm_out.gamma"), "no"), "so");
    h = conv2d_nhwc_bf16(o, h, W("encoder.conv_out.weight"), W("encoder.conv_out.bias"), 1, 1, 1, 1, 1, "co");
    h = conv1x1_nhwc_bf16(o, h, W("quant_conv.weight"), W("quant_conv.bias"), "qc");
    // mode() = first z_dim channels of [1, lH, lW, 2*z_dim]
    const auto mode = o.split(h, 3, {cfg_.z_dim, cfg_.z_dim}, "m_sp")[0];
    set_out(g, "z", o.cast(mode, DType::F32, "zf"));

    auto r = g.compile("qwen21_vae_enc_" + std::to_string(h_px) + "x" + std::to_string(w_px));
    auto& ref = *r;
    r_enc_.emplace(key, std::move(r));
    return ref;
}

std::vector<float> Qwen21Vae::decode_tile_nhwc(const uint16_t* z_nhwc_bf16, uint32_t lh, uint32_t lw) {
    const uint32_t H = lh * cfg_.spatial_compression, W = lw * cfg_.spatial_compression;
    const uint64_t in_bytes = uint64_t(lh) * lw * cfg_.z_dim * 2;
    const uint64_t out_bytes = uint64_t(H) * W * cfg_.out_channels * 4;
    DeviceBuffer d_z(dev_, in_bytes), d_img(dev_, out_bytes);
    HostBuffer h_z(dev_, in_bytes), h_img(dev_, out_bytes);
    std::memcpy(h_z.data(), z_nhwc_bf16, in_bytes);
    stream_.upload(h_z, 0, d_z, 0, in_bytes);

    auto& r = get_dec_recipe(lh, lw);
    std::vector<Binding> b;
    b.reserve(r.tensors().size());
    for (const auto& td : r.tensors()) {
        if (td.name == "z") b.push_back({"z", &d_z, 0});
        else if (td.name == "img") b.push_back({"img", &d_img, 0});
        else b.push_back(w_.bind(td.name, td.name));
    }
    r.enqueue(stream_, b);
    stream_.download(d_img, 0, h_img, 0, out_bytes);
    stream_.synchronize();

    std::vector<float> out(size_t(H) * W * cfg_.out_channels);
    std::memcpy(out.data(), h_img.data(), out_bytes);
    return out;
}

std::vector<float> Qwen21Vae::encode_tile_nhwc(const uint16_t* img_nhwc_bf16, uint32_t h, uint32_t w) {
    const uint32_t lh = h / cfg_.spatial_compression, lw = w / cfg_.spatial_compression;
    const uint64_t in_bytes = uint64_t(h) * w * cfg_.in_channels * 2;
    const uint64_t out_bytes = uint64_t(lh) * lw * cfg_.z_dim * 4;
    DeviceBuffer d_img(dev_, in_bytes), d_z(dev_, out_bytes);
    HostBuffer h_img(dev_, in_bytes), h_z(dev_, out_bytes);
    std::memcpy(h_img.data(), img_nhwc_bf16, in_bytes);
    stream_.upload(h_img, 0, d_img, 0, in_bytes);

    auto& r = get_enc_recipe(h, w);
    std::vector<Binding> b;
    b.reserve(r.tensors().size());
    for (const auto& td : r.tensors()) {
        if (td.name == "img") b.push_back({"img", &d_img, 0});
        else if (td.name == "z") b.push_back({"z", &d_z, 0});
        else b.push_back(w_.bind(td.name, td.name));
    }
    r.enqueue(stream_, b);
    stream_.download(d_z, 0, h_z, 0, out_bytes);
    stream_.synchronize();

    std::vector<float> out(size_t(lh) * lw * cfg_.z_dim);
    std::memcpy(out.data(), h_z.data(), out_bytes);
    return out;
}

std::vector<float> Qwen21Vae::encode_rgba(const uint8_t* rgba, uint32_t width, uint32_t height,
                                          bool use_tiling, std::vector<float>* out_raw_chw) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t sc = cfg_.spatial_compression, C = cfg_.z_dim;
    if (width % sc != 0 || height % sc != 0) throw std::invalid_argument("Qwen21Vae::encode_rgba: size not divisible by 16");
    const uint32_t lh = height / sc, lw = width / sc;

    // Convert RGBA uint8 [H, W, 4] -> [-1, 1] BF16 NHWC
    std::vector<uint16_t> img_bf16(size_t(height) * width * 4);
    for (size_t i = 0; i < img_bf16.size(); ++i) {
        img_bf16[i] = to_bf16(float(rgba[i]) * (1.0f / 127.5f) - 1.0f);
    }

    std::vector<float> raw_nhwc;
    const bool do_tile = use_tiling && (width > 1024 || height > 1024);
    if (!do_tile) {
        raw_nhwc = encode_tile_nhwc(img_bf16.data(), height, width);
    } else {
        // Raised-cosine 512x512 stride-384 tiled encode
        const uint32_t t_min = 512, t_str = 384;
        const uint32_t lt_min = t_min / sc, lt_str = t_str / sc;
        const uint32_t bh = lt_min - lt_str, bw = lt_min - lt_str;
        std::vector<std::vector<std::vector<float>>> rows;
        for (uint32_t y0 = 0; y0 < height; y0 += t_str) {
            std::vector<std::vector<float>> row;
            const uint32_t th = std::min(t_min, height - y0);
            for (uint32_t x0 = 0; x0 < width; x0 += t_str) {
                const uint32_t tw = std::min(t_min, width - x0);
                std::vector<uint16_t> tile(size_t(th) * tw * 4);
                for (uint32_t y = 0; y < th; ++y)
                    std::memcpy(tile.data() + size_t(y) * tw * 4,
                                img_bf16.data() + (size_t(y0 + y) * width + x0) * 4, size_t(tw) * 4 * 2);
                row.push_back(encode_tile_nhwc(tile.data(), th, tw));
            }
            rows.push_back(std::move(row));
        }
        raw_nhwc.assign(size_t(lh) * lw * C, 0.0f);
        for (size_t iy = 0; iy < rows.size(); ++iy) {
            const uint32_t y0 = uint32_t(iy) * lt_str;
            const uint32_t th = std::min(lt_min, lh - y0);
            for (size_t ix = 0; ix < rows[iy].size(); ++ix) {
                const uint32_t x0 = uint32_t(ix) * lt_str;
                const uint32_t tw = std::min(lt_min, lw - x0);
                auto cur = rows[iy][ix];
                if (iy > 0) blend_v_nhwc(rows[iy - 1][ix], cur, th, tw, C, bh);
                if (ix > 0) blend_h_nhwc(rows[iy][ix - 1], cur, th, tw, C, bw);
                const uint32_t kh = std::min(lt_str, lh - y0), kw = std::min(lt_str, lw - x0);
                for (uint32_t y = 0; y < kh; ++y)
                    std::memcpy(raw_nhwc.data() + (size_t(y0 + y) * lw + x0) * C,
                                cur.data() + size_t(y) * tw * C, size_t(kw) * C * 4);
            }
        }
    }

    if (out_raw_chw) {
        out_raw_chw->resize(size_t(C) * lh * lw);
        for (uint32_t y = 0; y < lh; ++y)
            for (uint32_t x = 0; x < lw; ++x)
                for (uint32_t c = 0; c < C; ++c)
                    (*out_raw_chw)[(size_t(c) * lh + y) * lw + x] = raw_nhwc[(size_t(y) * lw + x) * C + c];
    }

    // Normalize -> packed [lh*lw, 64] F32 (note: packed order matches transpose(1,2) of [C, lH*lW] == NHWC [lH, lW, C]!)
    std::vector<float> packed(size_t(lh) * lw * C);
    for (size_t p = 0; p < size_t(lh) * lw; ++p) {
        for (uint32_t c = 0; c < C; ++c) {
            packed[p * C + c] = (raw_nhwc[p * C + c] - cfg_.latents_mean[c]) / cfg_.latents_std[c];
        }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[Qwen21Vae] encoded %ux%u -> (%u,%u) in %.1f ms\n", width, height, lh, lw, ms);
    return packed;
}

std::vector<uint8_t> Qwen21Vae::decode_rgba(const float* packed_latents, uint32_t latent_h, uint32_t latent_w,
                                            bool use_tiling, std::vector<float>* out_raw_chw) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t sc = cfg_.spatial_compression, C = cfg_.z_dim;
    const uint32_t H = latent_h * sc, W = latent_w * sc;

    // Denormalize packed [lH*lW, 64] F32 -> NHWC BF16 [lH, lW, 64]
    std::vector<uint16_t> z_bf16(size_t(latent_h) * latent_w * C);
    for (size_t p = 0; p < size_t(latent_h) * latent_w; ++p) {
        for (uint32_t c = 0; c < C; ++c) {
            const float v = packed_latents[p * C + c] * cfg_.latents_std[c] + cfg_.latents_mean[c];
            z_bf16[p * C + c] = to_bf16(v);
        }
    }

    std::vector<float> img_nhwc;
    const bool do_tile = (use_tiling && (H > 512 || W > 512)) || (H > 1024 || W > 1024);
    if (!do_tile) {
        img_nhwc = decode_tile_nhwc(z_bf16.data(), latent_h, latent_w);
    } else {
        // Exact AutoencoderKLQwenImage21._tile_starts + raised-cosine 512x512 / stride-384 blend
        const uint32_t lt_min = 512u / sc;   // 32
        const uint32_t lt_str = 384u / sc;   // 24
        const uint32_t th_l = std::min(lt_min, latent_h), tw_l = std::min(lt_min, latent_w);
        const auto sy = tile_starts(latent_h, th_l, lt_str);
        const auto sx = tile_starts(latent_w, tw_l, lt_str);
        const uint32_t th_px = th_l * sc, tw_px = tw_l * sc;

        std::vector<std::vector<std::vector<float>>> rows(sy.size(), std::vector<std::vector<float>>(sx.size()));
        for (size_t iy = 0; iy < sy.size(); ++iy) {
            for (size_t ix = 0; ix < sx.size(); ++ix) {
                std::vector<uint16_t> tz(size_t(th_l) * tw_l * C);
                for (uint32_t y = 0; y < th_l; ++y)
                    std::memcpy(tz.data() + size_t(y) * tw_l * C,
                                z_bf16.data() + (size_t(sy[iy] + y) * latent_w + sx[ix]) * C, size_t(tw_l) * C * 2);
                rows[iy][ix] = decode_tile_nhwc(tz.data(), th_l, tw_l);
            }
        }
        img_nhwc.assign(size_t(H) * W * 4, 0.0f);
        uint32_t dst_y = 0;
        for (size_t iy = 0; iy < sy.size(); ++iy) {
            const uint32_t keep_h = (iy + 1 < sy.size()) ? (sy[iy + 1] - sy[iy]) * sc : (H - dst_y);
            uint32_t dst_x = 0;
            for (size_t ix = 0; ix < sx.size(); ++ix) {
                const uint32_t keep_w = (ix + 1 < sx.size()) ? (sx[ix + 1] - sx[ix]) * sc : (W - dst_x);
                auto cur = rows[iy][ix];
                if (iy > 0) blend_v_nhwc(rows[iy - 1][ix], cur, th_px, tw_px, 4, (sy[iy - 1] + th_l - sy[iy]) * sc);
                if (ix > 0) blend_h_nhwc(rows[iy][ix - 1], cur, th_px, tw_px, 4, (sx[ix - 1] + tw_l - sx[ix]) * sc);
                for (uint32_t y = 0; y < keep_h; ++y)
                    std::memcpy(img_nhwc.data() + (size_t(dst_y + y) * W + dst_x) * 4,
                                cur.data() + size_t(y) * tw_px * 4, size_t(keep_w) * 4 * 4);
                dst_x += keep_w;
            }
            dst_y += keep_h;
        }
    }

    if (out_raw_chw) {
        out_raw_chw->resize(size_t(4) * H * W);
        for (uint32_t y = 0; y < H; ++y)
            for (uint32_t x = 0; x < W; ++x)
                for (uint32_t c = 0; c < 4; ++c)
                    (*out_raw_chw)[(size_t(c) * H + y) * W + x] = img_nhwc[(size_t(y) * W + x) * 4 + c];
    }

    // VaeImageProcessor.postprocess: (x / 2 + 0.5).clamp(0, 1) * 255 -> uint8 RGBA
    std::vector<uint8_t> rgba(size_t(H) * W * 4);
    for (size_t i = 0; i < rgba.size(); ++i) {
        const float v = std::clamp(img_nhwc[i] * 0.5f + 0.5f, 0.0f, 1.0f);
        rgba[i] = uint8_t(std::round(v * 255.0f));
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[Qwen21Vae] decoded (%u,%u) -> %ux%u RGBA (%s) in %.1f ms\n",
                latent_h, latent_w, W, H, do_tile ? "tiled" : "untiled", ms);
    return rgba;
}

}  // namespace g2e
