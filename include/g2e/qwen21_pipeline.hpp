// End-to-end native Gaudi2 Qwen-Image-2.1-Turbo T2I + Multi-Image Editing pipeline.
#pragma once

#include "g2e/h3_text_encoder.hpp"
#include "g2e/qwen21_dit.hpp"
#include "g2e/qwen21_vae.hpp"
#include "g2e/runtime.hpp"
#include "g2e/tokenizer.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

struct Qwen21PipelinePaths {
    std::string tokenizer_json;
    std::string text_encoder_g2w;
    std::string dit_g2w;
    std::string vae_g2w;
};

struct Qwen21GenerateRequest {
    std::string prompt;
    std::string negative_prompt;
    uint32_t width = 0;   // 0 -> 1024 (T2I) or derived from last condition image (Edit)
    uint32_t height = 0;
    uint32_t num_inference_steps = 8;
    std::vector<float> sigmas;  // empty -> Turbo 8-step sample_sigmas
    uint64_t seed = 42;
    bool vae_use_tiling = false;
    std::vector<std::vector<uint8_t>> condition_images_bytes;  // 0 for T2I, 1..4 for Edit
};

struct Qwen21GenerateResult {
    std::vector<uint8_t> png_bytes;
    uint32_t width = 0;
    uint32_t height = 0;
    double preprocess_s = 0.0;
    double text_encode_s = 0.0;
    double vae_encode_s = 0.0;
    double denoise_s = 0.0;
    double vae_decode_s = 0.0;
    double png_encode_s = 0.0;
    double total_s = 0.0;
};

class Qwen21Pipeline {
public:
    Qwen21Pipeline(const Device& device, const Stream& stream, const Qwen21PipelinePaths& paths);
    ~Qwen21Pipeline();

    Qwen21GenerateResult generate(const Qwen21GenerateRequest& req);

    // Match diffusers calculate_dimensions(1024*1024, w/h), 32-aligned.
    static std::pair<uint32_t, uint32_t> calculate_dimensions(double target_area, double aspect_ratio);

private:
    const Device& dev_;
    const Stream& stream_;
    Tokenizer tok_;
    std::unique_ptr<H3TextEncoder> text_enc_;
    std::unique_ptr<WeightPack> dit_pack_;
    std::unique_ptr<WeightPack> vae_pack_;
    std::unique_ptr<Qwen21DiT> dit_;
    std::unique_ptr<Qwen21Vae> vae_;
};

}  // namespace g2e
