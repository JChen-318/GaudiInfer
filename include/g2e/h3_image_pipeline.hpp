// End-to-end native Gaudi2 MiniMax-H3-Image T2I & Multi-Reference Image Editing pipeline.
// Reuses Qwen3-VL-32B Layer-50 text/vision encoder (h3_text_encoder_fp8.g2w), 50L FP8 H3DiT
// with 6-step PDD fused heads + DirectH3Attn PV256, and 24L ViT2D MiniMaxH3ImageVAE.
#pragma once

#include "g2e/h3_dit.hpp"
#include "g2e/h3_text_encoder.hpp"
#include "g2e/h3_video_vae.hpp"
#include "g2e/runtime.hpp"
#include "g2e/tokenizer.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace g2e {

struct H3ImagePipelinePaths {
    std::string tokenizer_json;
    std::string text_encoder_g2w;
    std::string dit_g2w;
    std::string vae_g2w;
};

struct H3ImageGenerateRequest {
    std::string prompt;
    uint32_t width = 2048;
    uint32_t height = 2048;
    uint64_t seed = 42;
    uint64_t min_generation_area = 2048ull * 2048ull;  // upscale canvas if requested area < min_generation_area
    std::vector<std::vector<uint8_t>> reference_images_bytes;  // 0 for T2I, 1..16 for Edit
};

struct H3ImageGenerateResult {
    std::vector<uint8_t> png_bytes;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t canvas_width = 0;
    uint32_t canvas_height = 0;
    double encode_s = 0.0;
    double denoise_s = 0.0;
    double vae_decode_s = 0.0;
    double png_encode_s = 0.0;
    double total_s = 0.0;
};

class H3ImagePipeline {
public:
    H3ImagePipeline(const Device& device, const Stream& stream, const H3ImagePipelinePaths& paths);
    ~H3ImagePipeline();

    H3ImageGenerateResult generate(const H3ImageGenerateRequest& req);

    // Match LightX2V get_image_canvas_size(height, width, min_generation_area=2048^2).
    static std::pair<uint32_t, uint32_t> resolve_canvas_size(uint32_t height, uint32_t width,
                                                             uint64_t min_generation_area = 2048ull * 2048ull);

private:
    const Device& dev_;
    const Stream& stream_;
    Tokenizer tok_;
    std::unique_ptr<H3TextEncoder> text_enc_;
    std::unique_ptr<WeightPack> dit_pack_;
    std::unique_ptr<WeightPack> vae_pack_;
    std::unique_ptr<H3DiT> dit_;
    std::unique_ptr<H3VideoVae> vae_;
};

}  // namespace g2e
