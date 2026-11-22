#include "g2e/h3_audio_vae.hpp"

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

void set_out(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const DType dt = d.dtype;
    const auto sh = d.shape;
    const auto out = g.input(name, dt, sh);
    g.op("identity", {src}, {out});
}

// Replicate-pad 4D NHWC [B, 1, T, C] along W (axis 2) by (pad_l, pad_r)
TensorId replicate_pad_w(Ops& o, TensorId x, uint64_t pad_l, uint64_t pad_r, const std::string& name) {
    const auto sh = o.shape(x);
    const uint64_t T = sh[2];
    const auto first = (T > 1) ? o.split(x, 2, {1, T - 1}, name + "_f")[0] : x;
    const auto last = (T > 1) ? o.split(x, 2, {T - 1, 1}, name + "_l")[1] : x;
    std::vector<TensorId> parts;
    for (uint64_t i = 0; i < pad_l; ++i) parts.push_back(first);
    parts.push_back(x);
    for (uint64_t i = 0; i < pad_r; ++i) parts.push_back(last);
    return o.concat(parts, 2, name);
}

// 1D convolution on NHWC [B, 1, T, C_in] with PyTorch weight [C_out, C_in, K] and optional bias [C_out]
TensorId conv1d_nhwc(Ops& o, TensorId x, TensorId w_oik, TensorId bias, uint32_t stride, uint32_t pad_l,
                     uint32_t pad_r, uint32_t dilation, uint32_t groups, const std::string& name) {
    const auto xs = o.shape(x);
    const auto ws = o.shape(w_oik);
    const uint64_t B = xs[0], T = xs[2], C_out = ws[0], C_in_g = ws[1], K = ws[2];
    const uint64_t T_out = (T + pad_l + pad_r - dilation * (K - 1) - 1) / stride + 1;
    // Transpose [C_out, C_in/g, K] -> [1, K, C_in/g, C_out]
    const auto w4 = o.reshape(o.transpose(w_oik, {2, 1, 0}, name + "_wt"), {1, K, C_in_g, C_out}, name + "_w4");
    const auto y = o.temp(name + "_c", DType::F32, {B, 1, T_out, C_out});
    synConvolutionParamsV2 p{};
    p.kW = uint32_t(K);
    p.kH = 1;
    p.dW = stride;
    p.dH = 1;
    p.padL = int(pad_l);
    p.padR = int(pad_r);
    p.padT = 0;
    p.padB = 0;
    p.dilW = dilation;
    p.dilH = 1;
    p.nGroups = groups;
    p.paddingType = PADDING_EXPLICIT;
    o.graph().op("spatial_convolution", {x, w4}, {y}, &p, sizeof(p));
    if (bias == kNoTensor) return y;
    return o.add(y, o.reshape(bias, {1, 1, 1, C_out}, name + "_b"), name);
}

// 1D transposed convolution on NHWC [B, 1, T, C_in] with PyTorch weight [C_in, C_out, K] and optional bias [C_out]
TensorId conv_transpose1d_nhwc(Ops& o, TensorId x, TensorId w_iok, TensorId bias, uint32_t stride, uint32_t pad,
                               uint32_t groups, const std::string& name) {
    const auto xs = o.shape(x);
    const auto ws = o.shape(w_iok);
    const uint64_t B = xs[0], T = xs[2], C_in = ws[0], C_out_g = ws[1], K = ws[2];
    const uint64_t C_out = C_out_g * groups;
    const uint64_t T_out = (T - 1) * stride - 2 * pad + K;
    // dedx expects forward conv weight layout [kH=1, kW=K, C_out/g, C_in]
    const auto w4 = o.reshape(o.transpose(w_iok, {2, 1, 0}, name + "_wt"), {1, K, C_out_g, C_in}, name + "_w4");
    const auto y = o.temp(name + "_ct", DType::F32, {B, 1, T_out, C_out});
    synConvolutionParamsV2 p{};
    p.kW = uint32_t(K);
    p.kH = 1;
    p.dW = stride;
    p.dH = 1;
    p.padL = int(pad);
    p.padR = int(pad);
    p.padT = 0;
    p.padB = 0;
    p.dilW = 1;
    p.dilH = 1;
    p.nGroups = groups;
    p.paddingType = PADDING_EXPLICIT;
    o.graph().op("dedx", {x, w4}, {y}, &p, sizeof(p));
    if (bias == kNoTensor) return y;
    return o.add(y, o.reshape(bias, {1, 1, 1, C_out}, name + "_b"), name);
}

// Depthwise Kaiser-sinc Activation1d(SnakeBeta): UpSample1d(2x, k=12) -> SnakeBeta -> DownSample1d(2x, k=12)
// x: [B, 1, T, C] F32 -> [B, 1, T, C] F32
TensorId activation1d_snakebeta(Ops& o, TensorId x, TensorId alpha_raw, TensorId beta_raw,
                                TensorId up_filt, TensorId dn_filt, const std::string& name) {
    const auto xs = o.shape(x);
    const uint64_t C = xs[3];
    // Expand [1, 1, 12] filter to [C, 1, 12] via concat/tile or broadcast
    // UpSample1d(ratio=2, kernel_size=12): pad=5, pad_left=15, pad_right=15
    const auto x_up_pad = replicate_pad_w(o, x, 5, 5, name + "_upad");  // T + 10
    // Broadcast filter [1, 1, 12] -> [C, 1, 12] by adding to zero [C, 1, 12]
    const auto zc = o.constant(0.0f, {C, 1, 12}, name + "_zc", DType::F32);
    const auto uf_c = o.add(zc, up_filt, name + "_ufc");
    const auto up_raw = conv_transpose1d_nhwc(o, x_up_pad, uf_c, kNoTensor, 2, 0, uint32_t(C), name + "_uct");
    // up_raw has width = (T + 10 - 1)*2 + 12 = 2*T + 30; slice [15 .. 15 + 2*T) and multiply by ratio=2.0
    const uint64_t up_T = 2 * xs[2];
    const auto up_mid = o.split(up_raw, 2, {15, up_T, 15}, name + "_usl")[1];
    const auto xu = o.mul(up_mid, o.constant(2.0f, {1}, name + "_r2", DType::F32), name + "_xu");

    // SnakeBeta (logscale=True): x + sin^2(exp(alpha) * x) / (exp(beta) + 1e-9)
    const auto a = o.reshape(o.unary("exp", alpha_raw, name + "_ea"), {1, 1, 1, C}, name + "_a");
    const auto b = o.reshape(o.add(o.unary("exp", beta_raw, name + "_eb"),
                                   o.constant(1e-9f, {1}, name + "_eps", DType::F32), name + "_be"),
                             {1, 1, 1, C}, name + "_b");
    const auto s = o.unary("sin", o.mul(xu, a, name + "_xa"), name + "_sin");
    const auto act = o.add(xu, o.div(o.mul(s, s, name + "_s2"), b, name + "_div"), name + "_sb");

    // DownSample1d(ratio=2, kernel_size=12): pad_left=5, pad_right=6, stride=2
    const auto x_dn_pad = replicate_pad_w(o, act, 5, 6, name + "_dpad");
    const auto df_c = o.add(zc, dn_filt, name + "_dfc");
    return conv1d_nhwc(o, x_dn_pad, df_c, kNoTensor, 2, 0, 0, 1, uint32_t(C), name + "_dn");
}

}  // namespace

H3AudioVaeConfig H3AudioVaeConfig::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& cfg = j.at("config");
    H3AudioVaeConfig c;
    c.latent_channels = cfg.value("latent_channels", 32u);
    c.output_channels = cfg.value("output_channel", 2u);
    c.sample_rate = cfg.value("sample_rate", 32000u);
    c.latents_mean = cfg.at("latents_mean").get<std::vector<float>>();
    c.latents_std = cfg.at("latents_std").get<std::vector<float>>();
    return c;
}

H3AudioVae::H3AudioVae(const Device& device, const Stream& stream, const WeightPack& weights)
    : dev_(device), stream_(stream), w_(weights), cfg_(H3AudioVaeConfig::from_pack(weights)) {}

H3AudioVae::~H3AudioVae() = default;

Recipe& H3AudioVae::get_decode_recipe(uint32_t audio_t) {
    if (auto it = r_decode_.find(audio_t); it != r_decode_.end()) return *it->second;
    const uint64_t B = cfg_.output_channels;  // 2 (stereo)

    GraphBuilder g(dev_);
    Ops o(g, "av_");
    auto W = [&](const std::string& name) {
        const auto& e = w_.at(name);
        return g.input(name, e.dtype, e.shape);
    };

    // Input: [2, 1, audio_t, 32] F32 (already denormalized on host)
    TensorId h = g.input("z", DType::F32, {B, 1, audio_t, cfg_.latent_channels});
    h = conv1d_nhwc(o, h, W("dec_in_proj.weight"), W("dec_in_proj.bias"), 1, 0, 0, 1, 1, "in_proj");
    h = conv1d_nhwc(o, h, W("decoder.conv_pre.weight"), W("decoder.conv_pre.bias"), 1, 3, 3, 1, 1, "conv_pre");

    for (uint32_t i = 0; i < 7; ++i) {
        const uint32_t u = cfg_.upsample_rates[i];
        const uint32_t k = cfg_.upsample_kernels[i];
        const uint32_t pad = (k - u) / 2;
        const std::string up_p = "decoder.ups." + std::to_string(i) + ".0.";
        h = conv_transpose1d_nhwc(o, h, W(up_p + "weight"), W(up_p + "bias"), u, pad, 1, "up" + std::to_string(i));

        TensorId xs = kNoTensor;
        for (uint32_t j = 0; j < 3; ++j) {
            const uint32_t rb_idx = i * 3 + j;
            const uint32_t rk = cfg_.resblock_kernels[j];
            const std::string rp = "decoder.resblocks." + std::to_string(rb_idx) + ".";
            TensorId rx = h;
            for (uint32_t d_idx = 0; d_idx < 3; ++d_idx) {
                const uint32_t dil = cfg_.resblock_dilations[d_idx];
                const uint32_t pad1 = (rk * dil - dil) / 2;
                const uint32_t pad2 = (rk - 1) / 2;
                const std::string pfx = "rb" + std::to_string(rb_idx) + "_" + std::to_string(d_idx);
                const std::string a1p = rp + "activations." + std::to_string(2 * d_idx) + ".";
                const std::string a2p = rp + "activations." + std::to_string(2 * d_idx + 1) + ".";
                const std::string c1p = rp + "convs1." + std::to_string(d_idx) + ".";
                const std::string c2p = rp + "convs2." + std::to_string(d_idx) + ".";

                TensorId xt = activation1d_snakebeta(
                    o, rx, W(a1p + "act.alpha"), W(a1p + "act.beta"),
                    W(a1p + "upsample.filter"), W(a1p + "downsample.lowpass.filter"), pfx + "_a1");
                xt = conv1d_nhwc(o, xt, W(c1p + "weight"), W(c1p + "bias"), 1, pad1, pad1, dil, 1, pfx + "_c1");
                xt = activation1d_snakebeta(
                    o, xt, W(a2p + "act.alpha"), W(a2p + "act.beta"),
                    W(a2p + "upsample.filter"), W(a2p + "downsample.lowpass.filter"), pfx + "_a2");
                xt = conv1d_nhwc(o, xt, W(c2p + "weight"), W(c2p + "bias"), 1, pad2, pad2, 1, 1, pfx + "_c2");
                rx = o.add(rx, xt, pfx + "_res");
            }
            xs = (xs == kNoTensor) ? rx : o.add(xs, rx, "rbsum_" + std::to_string(rb_idx));
        }
        h = o.mul(xs, o.constant(1.0f / 3.0f, {1}, "inv3_" + std::to_string(i), DType::F32), "avg_" + std::to_string(i));
    }

    h = activation1d_snakebeta(
        o, h, W("decoder.activation_post.act.alpha"), W("decoder.activation_post.act.beta"),
        W("decoder.activation_post.upsample.filter"), W("decoder.activation_post.downsample.lowpass.filter"), "post_a");
    h = conv1d_nhwc(o, h, W("decoder.conv_post.weight"), kNoTensor, 1, 3, 3, 1, 1, "conv_post");
    // Clamp [-1, 1] (use_tanh_at_final = False)
    const auto lo = o.constant(-1.0f, {1}, "cl_lo", DType::F32);
    const auto hi = o.constant(1.0f, {1}, "cl_hi", DType::F32);
    h = o.binary("min", o.binary("max", h, lo, "cmn"), hi, "cmx");
    set_out(g, "wav", o.reshape(h, {B, uint64_t(audio_t) * cfg_.hop_length()}, "wav2"));

    auto r = g.compile("h3_avae_" + std::to_string(audio_t));
    auto& ref = *r;
    r_decode_.emplace(audio_t, std::move(r));
    return ref;
}

std::vector<float> H3AudioVae::decode(const float* latent, uint32_t audio_t) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t B = cfg_.output_channels, C = cfg_.latent_channels;
    // Denormalize [B, C, T] -> NHWC [B, 1, T, C]
    std::vector<float> h_in(size_t(B) * audio_t * C);
    for (uint32_t b = 0; b < B; ++b)
        for (uint32_t t = 0; t < audio_t; ++t)
            for (uint32_t c = 0; c < C; ++c) {
                const float v = latent[(size_t(b) * C + c) * audio_t + t];
                h_in[(size_t(b) * audio_t + t) * C + c] = v * cfg_.latents_std[c] + cfg_.latents_mean[c];
            }

    const uint64_t out_samples = uint64_t(audio_t) * cfg_.hop_length();
    DeviceBuffer d_in(dev_, h_in.size() * 4);
    DeviceBuffer d_out(dev_, uint64_t(B) * out_samples * 4);
    {
        HostBuffer stg(dev_, h_in.size() * 4);
        std::memcpy(stg.data(), h_in.data(), h_in.size() * 4);
        stream_.upload(stg, 0, d_in, 0, h_in.size() * 4);
        stream_.synchronize();
    }

    auto& r = get_decode_recipe(audio_t);
    std::vector<Binding> b;
    for (const auto& td : r.tensors()) {
        if (td.name == "z") b.push_back({"z", &d_in, 0});
        else if (td.name == "wav") b.push_back({"wav", &d_out, 0});
        else b.push_back(w_.bind(td.name, td.name));
    }
    r.enqueue(stream_, b);

    HostBuffer stg_out(dev_, uint64_t(B) * out_samples * 4);
    stream_.download(d_out, 0, stg_out, 0, stg_out.size());
    stream_.synchronize();
    std::vector<float> wav(size_t(B) * out_samples);
    std::memcpy(wav.data(), stg_out.data(), wav.size() * 4);

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("[H3AudioVae] decoded (%u,%u) -> %llu stereo samples @ %u Hz in %.1f ms\n",
                B, audio_t, (unsigned long long)out_samples, cfg_.sample_rate, ms);
    return wav;
}

}  // namespace g2e
