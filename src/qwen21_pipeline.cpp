#include "g2e/qwen21_pipeline.hpp"
#include "g2e/vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace g2e {

namespace {

using Clock = std::chrono::steady_clock;
double sec_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

uint32_t align32(uint32_t v) {
    v = std::max<uint32_t>(32u, v);
    return (v / 32u) * 32u;
}

}  // namespace

std::pair<uint32_t, uint32_t> Qwen21Pipeline::calculate_dimensions(double target_area, double aspect_ratio) {
    double w = std::sqrt(target_area * aspect_ratio);
    double h = w / aspect_ratio;
    uint32_t rw = uint32_t(std::max(1.0, std::nearbyint(w / 32.0))) * 32u;
    uint32_t rh = uint32_t(std::max(1.0, std::nearbyint(h / 32.0))) * 32u;
    return {rw, rh};
}

Qwen21Pipeline::Qwen21Pipeline(const Device& device, const Stream& stream, const Qwen21PipelinePaths& paths)
    : dev_(device), stream_(stream), tok_(paths.tokenizer_json) {
    const auto t0 = Clock::now();
    // All 3 Qwen-Image-2.1 packs total ~16.2 GiB -> keep all 3 resident in 96 GiB Gaudi2 HBM permanently!
    text_enc_ = std::make_unique<H3TextEncoder>(dev_, stream_, paths.text_encoder_g2w);
    text_enc_->load_to_hbm();

    dit_pack_ = std::make_unique<WeightPack>(paths.dit_g2w);
    auto st_dit = dit_pack_->upload(dev_, stream_);
    std::printf("[Qwen21Pipe] loaded DiT: %.2f GiB in %.2f s (%.1f GiB/s)\n", st_dit.gib, st_dit.seconds, st_dit.gib_per_s());
    dit_ = std::make_unique<Qwen21DiT>(dev_, stream_, *dit_pack_);

    vae_pack_ = std::make_unique<WeightPack>(paths.vae_g2w);
    auto st_vae = vae_pack_->upload(dev_, stream_);
    std::printf("[Qwen21Pipe] loaded VAE: %.2f GiB in %.2f s (%.1f GiB/s)\n", st_vae.gib, st_vae.seconds, st_vae.gib_per_s());
    vae_ = std::make_unique<Qwen21Vae>(dev_, stream_, *vae_pack_);

    std::printf("[Qwen21Pipe] ready in %.2f s\n", sec_since(t0));
}

Qwen21Pipeline::~Qwen21Pipeline() = default;

Qwen21GenerateResult Qwen21Pipeline::generate(const Qwen21GenerateRequest& req) {
    const auto t_total = Clock::now();
    Qwen21GenerateResult res;

    if (req.condition_images_bytes.size() > 4) {
        throw std::invalid_argument("Qwen21Pipeline: at most 4 condition images are supported");
    }

    // 1. Decode & resize condition images (once for both VLM and RGBA VAE)
    const auto t_pre = Clock::now();
    std::vector<VisionImage> vlm_imgs;
    std::vector<std::vector<uint8_t>> vae_rgbas;
    std::vector<std::pair<uint32_t, uint32_t>> cond_sizes;
    uint32_t calc_w = 1024, calc_h = 1024;

    for (const auto& bytes : req.condition_images_bytes) {
        int raw_w = 0, raw_h = 0;
        auto rgba = decode_image_rgba(bytes.data(), bytes.size(), raw_w, raw_h);
        auto [cw, ch] = calculate_dimensions(1024.0 * 1024.0, double(raw_w) / double(raw_h));
        calc_w = cw;
        calc_h = ch;
        auto rgba_rs = resize_rgba_bicubic(rgba.data(), raw_w, raw_h, int(cw), int(ch));
        auto rgb_white = rgba_over_white_to_rgb(rgba_rs.data(), int(cw), int(ch));
        vlm_imgs.push_back(preprocess_image_exact(rgb_white.data(), int(cw), int(ch), 16, 2));
        vae_rgbas.push_back(std::move(rgba_rs));
        cond_sizes.push_back({cw, ch});
    }
    const uint32_t out_w = align32(req.width > 0 ? req.width : calc_w);
    const uint32_t out_h = align32(req.height > 0 ? req.height : calc_h);
    res.width = out_w;
    res.height = out_h;
    res.preprocess_s = sec_since(t_pre);

    // 2. Encode prompt (+ optional condition images) with Qwen3-VL-8B
    const auto t_te = Clock::now();
    const auto encoded = text_enc_->encode_qwen21(tok_, req.prompt, vlm_imgs);
    res.text_encode_s = sec_since(t_te);

    // 3. VAE-encode condition images
    const auto t_ve = Clock::now();
    std::vector<float> cond_latents;
    std::vector<Qwen21ImgShape> img_shapes;
    uint32_t cond_tokens = 0;
    for (size_t i = 0; i < vae_rgbas.size(); ++i) {
        const uint32_t cw = cond_sizes[i].first, ch = cond_sizes[i].second;
        auto lat = vae_->encode_rgba(vae_rgbas[i].data(), cw, ch, req.vae_use_tiling);
        cond_latents.insert(cond_latents.end(), lat.begin(), lat.end());
        img_shapes.push_back({1u, ch / 16u, cw / 16u});
        cond_tokens += (ch / 16u) * (cw / 16u);
    }
    const uint32_t lh = out_h / 16u, lw = out_w / 16u;
    img_shapes.push_back({1u, lh, lw});
    res.vae_encode_s = sec_since(t_ve);

    // 4. Initialize target noise [lh*lw, 64] F32 (packed order matches NCHW -> transpose(1,2))
    const uint32_t T = lh * lw, C = dit_->config().in_channels;
    std::vector<float> target_latents(size_t(T) * C);
    {
        std::mt19937_64 rng(req.seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        // Generate in [C, lH, lW] order then pack to [lH*lW, C] matching torch randn((1,1,C,lH,lW))
        for (uint32_t c = 0; c < C; ++c) {
            for (uint32_t p = 0; p < T; ++p) {
                target_latents[size_t(p) * C + c] = nd(rng);
            }
        }
    }

    // 5. Build sigma schedule (Turbo 8-step default; or linear if custom step count != 8)
    std::vector<float> sigmas = req.sigmas;
    if (sigmas.empty()) {
        if (req.num_inference_steps == 8) {
            sigmas.assign(Qwen21DiT::kTurboSigmas.begin(), Qwen21DiT::kTurboSigmas.end());
        } else {
            const uint32_t ns = std::max<uint32_t>(1u, req.num_inference_steps);
            sigmas.resize(ns);
            for (uint32_t i = 0; i < ns; ++i) {
                sigmas[i] = 1.0f - (float(i) / float(ns)) * (1.0f - 1.0f / float(ns));
            }
        }
    }

    // 6. Denoise with 32L FP8 DiT + Prefix KV Cache
    const auto t_dit = Clock::now();
    dit_->denoise(encoded.embeds.data(), encoded.seq_len, encoded.image_pad_mask.data(),
                  cond_latents.empty() ? nullptr : cond_latents.data(), cond_tokens,
                  img_shapes, target_latents, sigmas);
    res.denoise_s = sec_since(t_dit);

    // 7. VAE decode -> RGBA -> PNG
    const auto t_vd = Clock::now();
    const auto rgba = vae_->decode_rgba(target_latents.data(), lh, lw, req.vae_use_tiling);
    res.vae_decode_s = sec_since(t_vd);

    const auto t_png = Clock::now();
    res.png_bytes = encode_png_rgba(rgba.data(), int(out_w), int(out_h));
    res.png_encode_s = sec_since(t_png);

    res.total_s = sec_since(t_total);
    std::printf("[Qwen21Pipe] %s %ux%u (%zu cond) total=%.2fs (te=%.2fs ve=%.2fs dit=%.2fs vd=%.2fs png=%.2fs)\n",
                cond_sizes.empty() ? "T2I" : "Edit", out_w, out_h, cond_sizes.size(),
                res.total_s, res.text_encode_s, res.vae_encode_s, res.denoise_s, res.vae_decode_s, res.png_encode_s);
    return res;
}

}  // namespace g2e
