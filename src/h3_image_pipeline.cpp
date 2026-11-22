#include "g2e/h3_image_pipeline.hpp"
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

// Match LightX2V _bound_short_edge(image, min_short_edge, max_short_edge) -> 32-aligned resize.
std::pair<uint32_t, uint32_t> bound_short_edge_size(uint32_t w, uint32_t h, uint32_t min_se, uint32_t max_se) {
    const uint32_t se = std::min(w, h);
    if (se >= min_se && se <= max_se) {
        return {std::max<uint32_t>(32u, (w / 32u) * 32u), std::max<uint32_t>(32u, (h / 32u) * 32u)};
    }
    const double scale = double(se < min_se ? min_se : max_se) / double(se);
    const uint32_t rw = std::max<uint32_t>(32u, uint32_t(std::nearbyint(double(w) * scale / 32.0)) * 32u);
    const uint32_t rh = std::max<uint32_t>(32u, uint32_t(std::nearbyint(double(h) * scale / 32.0)) * 32u);
    return {rw, rh};
}

// Unpatchify target rows [(lh/2)*(lw/2), 96] -> planar [24, 1, lh, lw] F32.
std::vector<float> unpatchify_image_rows(const float* rows, uint32_t lh, uint32_t lw) {
    constexpr uint32_t C = 24;
    const uint32_t ph = lh / 2, pw = lw / 2;
    std::vector<float> lat(size_t(C) * lh * lw);
    for (uint32_t hi = 0; hi < ph; ++hi) {
        for (uint32_t wi = 0; wi < pw; ++wi) {
            const float* src = rows + size_t(hi * pw + wi) * 96;
            for (uint32_t c = 0; c < C; ++c)
                for (uint32_t dh = 0; dh < 2; ++dh)
                    for (uint32_t dw = 0; dw < 2; ++dw)
                        lat[(size_t(c) * lh + (hi * 2 + dh)) * lw + (wi * 2 + dw)] = src[(c * 2 + dh) * 2 + dw];
        }
    }
    return lat;
}

}  // namespace

std::pair<uint32_t, uint32_t> H3ImagePipeline::resolve_canvas_size(uint32_t height, uint32_t width,
                                                                   uint64_t min_generation_area) {
    if (!height || !width) throw std::invalid_argument("H3ImagePipeline: height and width must be positive");
    const double area = double(height) * double(width);
    const double scale = std::max(1.0, std::sqrt(double(min_generation_area) / area));
    const uint32_t ch = std::max<uint32_t>(32u, uint32_t(std::nearbyint(double(height) * scale / 32.0)) * 32u);
    const uint32_t cw = std::max<uint32_t>(32u, uint32_t(std::nearbyint(double(width) * scale / 32.0)) * 32u);
    return {ch, cw};
}

H3ImagePipeline::H3ImagePipeline(const Device& device, const Stream& stream, const H3ImagePipelinePaths& paths)
    : dev_(device), stream_(stream), tok_(paths.tokenizer_json) {
    const auto t0 = Clock::now();
    // 31 GiB DiT + 1.2 GiB ImageVAE + 25.3 GiB TextEncoder = ~57.5 GiB -> ALL THREE fit resident in 96 GiB Gaudi2 HBM!
    // (Leaves ~38 GiB free HBM for 4K 16 MP activations + DirectH3Attn buffers — zero reload latency!)
    dit_pack_ = std::make_unique<WeightPack>(paths.dit_g2w);
    auto st_dit = dit_pack_->upload(dev_, stream_);
    std::printf("[H3ImagePipe] loaded DiT: %.2f GiB in %.2f s (%.1f GiB/s)\n", st_dit.gib, st_dit.seconds, st_dit.gib_per_s());
    dit_ = std::make_unique<H3DiT>(dev_, stream_, *dit_pack_);

    vae_pack_ = std::make_unique<WeightPack>(paths.vae_g2w);
    auto st_vae = vae_pack_->upload(dev_, stream_);
    std::printf("[H3ImagePipe] loaded ImageVAE: %.2f GiB in %.2f s (%.1f GiB/s)\n", st_vae.gib, st_vae.seconds, st_vae.gib_per_s());
    vae_ = std::make_unique<H3VideoVae>(dev_, stream_, *vae_pack_);

    text_enc_ = std::make_unique<H3TextEncoder>(dev_, stream_, paths.text_encoder_g2w);
    text_enc_->load_to_hbm();
    std::printf("[H3ImagePipe] ready in %.2f s\n", sec_since(t0));
}

H3ImagePipeline::~H3ImagePipeline() = default;

H3ImageGenerateResult H3ImagePipeline::generate(const H3ImageGenerateRequest& req) {
    const auto t_total = Clock::now();
    if (req.prompt.empty()) throw std::invalid_argument("H3ImagePipeline: prompt must be non-empty");
    if (req.reference_images_bytes.size() > 16) {
        throw std::invalid_argument("H3ImagePipeline: at most 16 reference images supported");
    }

    H3ImageGenerateResult res;
    res.width = req.width > 0 ? req.width : 2048u;
    res.height = req.height > 0 ? req.height : 2048u;
    auto [canvas_h, canvas_w] = resolve_canvas_size(res.height, res.width, req.min_generation_area);
    res.canvas_height = canvas_h;
    res.canvas_width = canvas_w;

    // 1. Encode prompt (+ up to 16 reference images) via Qwen3-VL-32B Layer-50 + ImageVAE encoder
    const auto t_enc = Clock::now();
    std::vector<VisionImage> vlm_imgs;
    std::vector<std::vector<uint8_t>> vae_rgbs;
    std::vector<std::pair<uint32_t, uint32_t>> vae_sizes;
    for (const auto& bytes : req.reference_images_bytes) {
        int rw = 0, rh = 0;
        auto rgb = decode_image(bytes.data(), bytes.size(), rw, rh);
        if (std::min(rw, rh) < 32) throw std::invalid_argument("H3ImagePipeline: reference image < 32px");
        auto [qw, qh] = bound_short_edge_size(uint32_t(rw), uint32_t(rh), 1024u, 1024u);
        auto q_rgb = resize_rgb_bicubic(rgb.data(), rw, rh, int(qw), int(qh));
        vlm_imgs.push_back(preprocess_image(q_rgb.data(), int(qw), int(qh), 3136ull, 1048576ull, 16, 2));

        const uint32_t se = uint32_t(std::min(rw, rh));
        if (se <= 1024u) {
            vae_rgbs.push_back(std::move(q_rgb));
            vae_sizes.push_back({qw, qh});
        } else {
            auto [vw, vh] = bound_short_edge_size(uint32_t(rw), uint32_t(rh), 1024u, 2048u);
            vae_rgbs.push_back(resize_rgb_bicubic(rgb.data(), rw, rh, int(vw), int(vh)));
            vae_sizes.push_back({vw, vh});
        }
    }

    // MiniMax-H3-Image uses "Picture i: " (no angle brackets)
    const auto te_out = text_enc_->encode_multimodal(tok_, req.prompt, vlm_imgs, /*angle_brackets=*/false);

    // Encode reference images with ImageVAE encoder + build noisy anchors (0.999 * clean + 0.001 * noise)
    std::mt19937_64 rng(req.seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<std::pair<uint32_t, uint32_t>> ref_latent_grids;
    std::vector<float> ref_anchors;
    for (size_t i = 0; i < vae_rgbs.size(); ++i) {
        const uint32_t vw = vae_sizes[i].first, vh = vae_sizes[i].second;
        auto clean_rows = vae_->encode_image(vae_rgbs[i].data(), vw, vh, req.seed + i + 1, /*sample_posterior=*/true);
        ref_latent_grids.push_back({vh / 16u, vw / 16u});
        constexpr float kRefT = 0.999f;
        for (float v : clean_rows) {
            ref_anchors.push_back(kRefT * v + (1.0f - kRefT) * nd(rng));
        }
    }
    res.encode_s = sec_since(t_enc);

    // 2. Build H3-Image packed layout + initialize target noise
    const uint32_t lh = canvas_h / 16u, lw = canvas_w / 16u;
    const auto layout = H3PackedLayout::build_image(te_out.seq_len, ref_latent_grids, lh, lw, te_out.token_tags);

    const uint32_t row_w = dit_->config().video_row_width();  // 96
    std::vector<float> img_rows(size_t(layout.img_rows) * row_w);
    if (!ref_anchors.empty()) {
        std::memcpy(img_rows.data(), ref_anchors.data(), ref_anchors.size() * sizeof(float));
    }
    // Generate target noise in [24, lh, lw] order then patchify [1,2,2] -> [ph*pw, 96]
    {
        std::vector<float> noise_chw(size_t(24) * lh * lw);
        for (float& v : noise_chw) v = nd(rng);
        const uint32_t ph = lh / 2, pw = lw / 2;
        float* tgt = img_rows.data() + size_t(layout.cond_rows) * row_w;
        for (uint32_t hi = 0; hi < ph; ++hi)
            for (uint32_t wi = 0; wi < pw; ++wi) {
                float* dst = tgt + size_t(hi * pw + wi) * 96;
                for (uint32_t c = 0; c < 24; ++c)
                    for (uint32_t dh = 0; dh < 2; ++dh)
                        for (uint32_t dw = 0; dw < 2; ++dw)
                            dst[(c * 2 + dh) * 2 + dw] = noise_chw[(size_t(c) * lh + (hi * 2 + dh)) * lw + (wi * 2 + dw)];
            }
    }

    // 3. 6-step PDD denoising on Gaudi2 (reuses DirectH3Attn PV256!)
    const auto t_dit = Clock::now();
    dit_->denoise_image(layout, img_rows, te_out.embeds.data(),
                        ref_anchors.empty() ? nullptr : ref_anchors.data(), 0.999f);
    res.denoise_s = sec_since(t_dit);

    // 4. Unpatchify target rows -> MiniMaxH3ImageVAE 2D ViT decode -> RGB24 -> optional resize -> PNG
    const auto t_vd = Clock::now();
    const auto target_lat = unpatchify_image_rows(img_rows.data() + size_t(layout.cond_rows) * row_w, lh, lw);
    auto rgb = vae_->decode(target_lat.data(), 1u, lh, lw, 1u);
    if (canvas_w != res.width || canvas_h != res.height) {
        rgb = resize_rgb_bicubic(rgb.data(), int(canvas_w), int(canvas_h), int(res.width), int(res.height));
    }
    res.vae_decode_s = sec_since(t_vd);

    const auto t_png = Clock::now();
    res.png_bytes = encode_png_rgb(rgb.data(), int(res.width), int(res.height));
    res.png_encode_s = sec_since(t_png);

    res.total_s = sec_since(t_total);
    std::printf("[H3ImagePipe] %s %ux%u (canvas %ux%u, %zu ref, U=%u) total=%.2fs (enc=%.2fs dit=%.2fs vae=%.2fs png=%.2fs)\n",
                ref_latent_grids.empty() ? "T2I" : "Edit", res.width, res.height, canvas_w, canvas_h,
                ref_latent_grids.size(), layout.used_len, res.total_s, res.encode_s, res.denoise_s,
                res.vae_decode_s, res.png_encode_s);
    return res;
}

}  // namespace g2e
