// Native Gaudi2 Qwen-Image-2.1-Turbo 2D RGBA AutoencoderKL (16x spatial, 64 latent channels).
#pragma once

#include "g2e/graph.hpp"
#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace g2e {

struct Qwen21VaeConfig {
    uint32_t base_dim = 96;
    uint32_t decoder_base_dim = 144;
    uint32_t z_dim = 64;
    uint32_t in_channels = 4;
    uint32_t out_channels = 4;
    uint32_t spatial_compression = 16;
    uint32_t num_res_blocks = 2;
    std::vector<uint32_t> dim_mult = {1, 2, 4, 8, 8};
    std::vector<bool> temporal_downsample = {false, true, true, true};
    std::vector<float> latents_mean;
    std::vector<float> latents_std;

    static Qwen21VaeConfig from_pack(const WeightPack& pack);
};

class Qwen21Vae {
public:
    Qwen21Vae(const Device& device, const Stream& stream, const WeightPack& weights);
    ~Qwen21Vae();

    const Qwen21VaeConfig& config() const { return cfg_; }

    // Encode RGBA [H, W, 4] uint8 (H, W multiples of 32) -> normalized packed latents [(H/16)*(W/16), 64] F32.
    // Optionally outputs raw [64, H/16, W/16] F32 mode before normalization.
    std::vector<float> encode_rgba(const uint8_t* rgba, uint32_t width, uint32_t height,
                                   bool use_tiling = false, std::vector<float>* out_raw_chw = nullptr);

    // Decode normalized packed latents [lH*lW, 64] F32 -> RGBA [lH*16, lW*16, 4] uint8.
    // Automatically tiles with raised-cosine 512x512 / stride-384 windows when `use_tiling` is true
    // or pixel dimensions exceed 1024. Optionally outputs raw [-1, 1] [4, H, W] F32.
    std::vector<uint8_t> decode_rgba(const float* packed_latents, uint32_t latent_h, uint32_t latent_w,
                                     bool use_tiling = false, std::vector<float>* out_raw_chw = nullptr);

private:
    Recipe& get_dec_recipe(uint32_t lh, uint32_t lw);
    Recipe& get_enc_recipe(uint32_t h, uint32_t w);

    // Run a single untiled decode on denormalized NHWC [lh, lw, 64] BF16 -> [lh*16, lw*16, 4] F32 clamped [-1, 1].
    std::vector<float> decode_tile_nhwc(const uint16_t* z_nhwc_bf16, uint32_t lh, uint32_t lw);
    // Run a single untiled encode on NHWC [-1, 1] [h, w, 4] BF16 -> raw mode [h/16, w/16, 64] F32.
    std::vector<float> encode_tile_nhwc(const uint16_t* img_nhwc_bf16, uint32_t h, uint32_t w);

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    Qwen21VaeConfig cfg_;

    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_dec_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_enc_;
};

}  // namespace g2e
