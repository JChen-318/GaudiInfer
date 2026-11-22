// Native Gaudi2 Qwen-Image-2.1-Turbo Single-Stream DiT (32L FP8 + Prefix KV Cache).
#pragma once

#include "g2e/graph.hpp"
#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace g2e {

struct Qwen21DiTConfig {
    uint32_t num_layers = 32;
    uint32_t num_heads = 32;
    uint32_t head_dim = 128;
    uint32_t hidden_size = 4096;
    uint32_t mlp_ratio = 3;
    uint32_t ffn_hidden = 12288;
    uint32_t in_channels = 64;
    uint32_t out_channels = 64;
    std::array<uint32_t, 3> axes_dims_rope = {16, 56, 56};
    float eps = 1e-6f;
    bool causal_condition = true;
    bool blocks_fp8 = true;

    static Qwen21DiTConfig from_pack(const WeightPack& pack);
};

struct Qwen21ImgShape {
    uint32_t f = 1;
    uint32_t h = 0;  // latent height = pixel_h / 16
    uint32_t w = 0;  // latent width  = pixel_w / 16
    uint32_t tokens() const { return f * h * w; }
};

struct Qwen21StepDebug {
    std::vector<float> b0_target;    // [target_len, 4096] F32
    std::vector<float> noise_pred;   // [target_len, 64] F32
};

class Qwen21DiT {
public:
    static constexpr std::array<float, 8> kTurboSigmas = {
        1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f};

    Qwen21DiT(const Device& device, const Stream& stream, const WeightPack& weights);
    ~Qwen21DiT();

    const Qwen21DiTConfig& config() const { return cfg_; }

    // Run full multi-step rectified-flow Euler denoising using Step-0 Prefill + Prefix KV Cache
    // on Steps 1..(N-1).
    // - prompt_embeds: [vlm_len, 4096] BF16 (from H3TextEncoder::encode_qwen21)
    // - vlm_img_mask:  [vlm_len] uint8 (1 at <|image_pad|> slots, 0 elsewhere)
    // - cond_latents:  concatenated [sum_cond_tokens, 64] F32 (empty for T2I)
    // - img_shapes:    condition image shapes first, target image shape last
    // - target_latents: [target_tokens, 64] F32, modified in-place from noise -> clean latent
    // - sigmas:        timestep schedule in [0, 1] (default kTurboSigmas; terminal 0.0 appended automatically)
    // Optional debug hooks capture Step 0 and Step 1 b0_target + noise_pred for golden verification.
    double denoise(const uint16_t* prompt_embeds, uint32_t vlm_len, const uint8_t* vlm_img_mask,
                   const float* cond_latents, uint32_t cond_tokens,
                   const std::vector<Qwen21ImgShape>& img_shapes,
                   std::vector<float>& target_latents,
                   const std::vector<float>& sigmas,
                   Qwen21StepDebug* debug_step0 = nullptr,
                   Qwen21StepDebug* debug_step1 = nullptr);

private:
    Recipe& get_mod_recipe();
    Recipe& get_entry_prefill_recipe(uint32_t vlm_len, uint32_t img_tokens, uint32_t total_len);
    Recipe& get_entry_decode_recipe(uint32_t target_len);
    Recipe& get_blk_prefill_recipe(uint32_t prefix_len, uint32_t target_len);
    Recipe& get_blk_decode_recipe(uint32_t prefix_len, uint32_t target_len);
    Recipe& get_final_recipe(uint32_t target_len);

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    Qwen21DiTConfig cfg_;

    std::unique_ptr<Recipe> r_mod_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_entry_pre_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_entry_dec_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_blk_pre_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_blk_dec_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_final_;
};

}  // namespace g2e
