// MiniMax-H3 Video VAE (36-layer ViT3D decoder) native Gaudi2 engine.
// Decodes [1, 24, latent_t, latent_h, latent_w] FP32 latents into RGB video frames
// using temporal chunking (17-frame clips, token_drop=3) + 256px spatial tiling with
// feather blending and fast non-causal SDPA.
#pragma once

#include "g2e/graph.hpp"
#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace g2e {

struct H3VideoVaeConfig {
    uint32_t num_layers = 36;
    uint32_t hidden_size = 2048;
    uint32_t num_heads = 32;
    uint32_t dim_head = 64;
    uint32_t ffn_hidden = 8192;
    uint32_t latent_channels = 24;
    uint32_t vae_ratio = 16;         // spatial upsampling per latent pixel
    uint32_t vae_ratio_t = 4;        // temporal upsampling per latent token
    uint32_t clip_length = 17;
    uint32_t token_drop = 3;
    uint32_t tile_size = 256;
    uint32_t tile_overlap_min = 64;
    uint32_t num_register_tokens = 4;
    float rope_theta = 100.0f;
    float rope_dim_ratio = 0.75f;    // rotates 48 of 64 head dims
    float norm_eps = 1e-5f;
    bool blocks_fp8 = false;
    bool has_encoder = false;
    bool vit_decoder_2d = false;
    std::vector<float> latents_mean; // [24]
    std::vector<float> latents_std;  // [24]

    uint32_t rot_dim() const { return uint32_t(dim_head * rope_dim_ratio); }  // 48
    static H3VideoVaeConfig from_pack(const WeightPack& pack);
};

class H3VideoVae {
public:
    H3VideoVae(const Device& device, const Stream& stream, const WeightPack& weights);
    ~H3VideoVae();

    const H3VideoVaeConfig& config() const { return cfg_; }
    bool has_encoder() const { return cfg_.has_encoder; }

    // Encode a single RGB uint8 keyframe [height, width, 3] (dimensions multiple of 32) into normalized
    // patchified DiT condition rows [(height/32)*(width/32), 96] FP32 (matching MiniMax-H3 encode_image).
    // If out_moments is non-null, writes blended pre-sampling moments [48, height/16, width/16] FP32 for verification.
    std::vector<float> encode_image(const uint8_t* rgb, uint32_t width, uint32_t height,
                                    uint64_t seed = 42, bool sample_posterior = true,
                                    std::vector<float>* out_moments = nullptr);

    // Decode full latent [latent_channels=24, latent_t, latent_h, latent_w] (CHW-major, C outer)
    // into RGB uint8 frames [num_frames, H, W, 3] where H = latent_h*16, W = latent_w*16.
    // If out_frames_f32 is non-null, also outputs raw float [0, 1] [3, num_frames, H, W].
    // If on_rgb_frames is provided (and out_frames_f32 is null), streams finalized RGB24 frame chunks
    // directly as each temporal chunk completes (enabling full overlap with ffmpeg encoding) and returns empty vector.
    using RgbChunkCallback = std::function<void(std::vector<uint8_t>&&)>;
    std::vector<uint8_t> decode(const float* latent, uint32_t latent_t, uint32_t latent_h, uint32_t latent_w,
                                uint32_t target_frames = 0, std::vector<float>* out_frames_f32 = nullptr,
                                RgbChunkCallback on_rgb_frames = nullptr);

    // Decode a single (t, h, w) latent tile [24, t, h, w] -> [3, t*4, h*16, w*16] float (pre-revert).
    std::vector<float> decode_tile(const float* tile_latent, uint32_t t, uint32_t h, uint32_t w);

    // Decode a batch of B identical-shape (t, h, w) tiles [B, 24, t, h, w] -> [B, 3, t*4, h*16, w*16] float (pre-revert).
    std::vector<float> decode_tiles_batch(const float* tiles_latent, uint32_t batch, uint32_t t, uint32_t h, uint32_t w);

private:
    Recipe& get_pre_recipe(uint32_t batch, uint32_t n_patches);
    Recipe& get_block_recipe(uint32_t batch, uint32_t seq_len);
    Recipe& get_post_recipe(uint32_t batch, uint32_t seq_len, uint32_t n_patches);
    Recipe& get_enc_recipe(uint32_t batch, uint32_t tile_h, uint32_t tile_w);
    void enqueue_tiles_batch(const float* tiles_latent, uint32_t batch, uint32_t t, uint32_t h, uint32_t w,
                             uint32_t slot);

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    H3VideoVaeConfig cfg_;

    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_pre_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_block_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_post_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_enc_;

    // Persistent reusable scratch buffers (double-buffered host staging for async GPU||CPU overlap)
    uint64_t alloc_in_ = 0, alloc_rope_ = 0, alloc_h_ = 0, alloc_out_ = 0;
    uint32_t cached_t_ = 0, cached_h_ = 0, cached_w_ = 0;
    DeviceBuffer d_in_, d_cos_, d_sin_, d_h_[2], d_out_;
    std::unique_ptr<HostBuffer> h_in_stg_[2], h_out_stg_[2];
};

}  // namespace g2e
