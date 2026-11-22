// MiniMax-H3 FL2VA/T2VA Audio-Video DiT native Gaudi2 engine.
// Supports base FP8 (.g2w) weights + hot-swappable runtime BF16 LoRA adapters (.safetensors)
// as well as offline-merged FP8 packs.
#pragma once

#include "g2e/graph.hpp"
#include "g2e/h3_direct_attn.hpp"
#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace g2e {

struct H3DiTConfig {
    uint32_t num_layers = 50;
    uint32_t refiner_layers = 2;
    uint32_t hidden_size = 5376;
    uint32_t num_heads = 56;
    uint32_t head_dim = 128;
    uint32_t ffn_hidden = 14336;
    uint32_t latents_dim = 24;
    uint32_t audio_latents_dim = 32;
    uint32_t patch_t = 1, patch_h = 2, patch_w = 2;
    uint32_t text_dim = 5120;
    uint32_t timestep_input_dim = 256;
    uint32_t time_embed_hidden = 5376;
    uint32_t time_embed_dim = 2688;
    uint32_t rope_inv_freq_len = 16;  // rotary_dim = 6 * 16 = 96
    float norm_eps = 1e-5f;
    float qk_norm_eps = 1e-5f;
    float final_norm_eps = 1e-5f;
    bool adaln_bf16 = false;
    bool audio_enabled = true;
    uint32_t pdd_steps = 0;

    uint32_t attn_dim() const { return num_heads * head_dim; }
    uint32_t rot_dim() const { return 6 * rope_inv_freq_len; }
    uint32_t video_row_width() const { return latents_dim * patch_t * patch_h * patch_w; }

    static H3DiTConfig from_pack(const WeightPack& pack);
};

// Hot-swappable runtime BF16 LoRA adapter loaded directly from any Diffusers/LightX2V .safetensors file.
// Automatically maps Q/K/V -> packed qkv_a [3*R, H] & qkv_b [3, 7168, R], swaps ff.net.0.proj.lora_B [up, gate] -> [gate, up],
// and scales B by (alpha / rank) * scale so swapping acceleration LoRAs takes <0.3 s with zero re-quantization.
class H3LoraAdapter {
public:
    H3LoraAdapter(const Device& device, const Stream& stream, const std::string& safetensors_path,
                  const H3DiTConfig& cfg, float scale = 1.0f);

    uint32_t rank() const { return rank_; }
    uint32_t adaln_rank() const { return adaln_rank_; }
    uint32_t final_adaln_rank() const { return final_adaln_rank_; }
    float alpha() const { return alpha_; }
    float scale() const { return scale_; }
    float mult() const { return scale_ * (alpha_ / float(rank_)); }
    const std::string& path() const { return path_; }
    uint64_t bytes() const { return arena_ ? arena_->size() : 0; }

    // Bindings for block `idx` (refiner=true -> R{idx}, refiner=false -> B{idx}).
    // Tensor names: lora_qkv_a, lora_qkv_b, lora_out_a, lora_out_b, lora_fc1_a, lora_fc1_b, lora_fc2_a, lora_fc2_b.
    void bind_block(std::vector<Binding>& out, bool refiner, uint32_t idx) const;
    void bind_adaln(std::vector<Binding>& out, uint32_t idx) const;
    void bind_final_adaln(std::vector<Binding>& out) const;
    Binding bind_one(const std::string& name, bool refiner, uint32_t idx, int which) const;

private:
    struct BlockOffsets {
        uint64_t qkv_a = 0, qkv_b = 0;
        uint64_t out_a = 0, out_b = 0;
        uint64_t fc1_a = 0, fc1_b = 0;
        uint64_t fc2_a = 0, fc2_b = 0;
        uint64_t adaln_a = 0, adaln_b = 0;
    };
    std::string path_;
    uint32_t rank_ = 0;
    uint32_t adaln_rank_ = 0;
    uint32_t final_adaln_rank_ = 0;
    float alpha_ = 8.0f;
    float scale_ = 1.0f;
    uint64_t final_adaln_a_ = 0, final_adaln_b_ = 0;
    std::vector<BlockOffsets> refiner_off_;
    std::vector<BlockOffsets> block_off_;
    std::unique_ptr<DeviceBuffer> arena_;
};

// Packed sequence layout for one MiniMax-H3 FL2VA/T2VA request.
// Layout order: [text | cond_video (optional) | audio | video_target | pad] with seq_len aligned to 64.
struct H3PackedLayout {
    uint32_t text_len = 0;
    uint32_t latent_t = 0, latent_h = 0, latent_w = 0;
    uint32_t audio_t = 0, audio_channel = 2;
    uint32_t cond_rows = 0;
    uint32_t audio_rows = 0;
    uint32_t video_rows = 0;       // target video rows
    uint32_t img_rows = 0;         // cond_rows + video_rows
    uint32_t used_len = 0;
    uint32_t seq_len = 0;          // aligned to 64

    std::vector<int32_t> token_tags;        // [seq_len]: 1=text, 0=video, 2=audio, -1=pad
    std::vector<uint8_t> update_mask;       // [img_rows]: 0 for cond, 1 for target
    std::vector<double> img_position_ids;   // [seq_len * 3] (t, h, w) in fp64

    static H3PackedLayout build(uint32_t text_len, uint32_t latent_t, uint32_t latent_h, uint32_t latent_w,
                                uint32_t audio_t, bool include_keyframe_cond = false,
                                const std::vector<int>& keyframe_frame_indices = {}, int frame_count = 0,
                                const std::vector<int32_t>& prefix_token_tags = {});

    // Build MiniMax-H3-Image packed sequence: [text | ref_image_0..K-1 | target_image | pad].
    // Matches LightX2V build_image_packed_sequence (audio_rows=0, t = text_len + image_index).
    static H3PackedLayout build_image(uint32_t text_len,
                                      const std::vector<std::pair<uint32_t, uint32_t>>& ref_latent_grids,
                                      uint32_t latent_h, uint32_t latent_w,
                                      const std::vector<int32_t>& prefix_token_tags = {});
};

// Optional stage-by-stage debug dump callbacks/outputs for numerical verification against PyTorch.
struct H3ForwardDebug {
    std::vector<float> t_emb;              // [M, 2688]
    std::vector<float> rope_table;         // [seq_len, 96]
    std::vector<float> token_refiner_out;  // [text_len, 5376]
    std::vector<float> b0_out;             // [seq_len, 5376]
    std::vector<float> b1_out;             // [seq_len, 5376]
    bool dump_all_layers = false;
    uint32_t layer_sample_rows = 256;      // sample rows per block when dump_all_layers=true
    std::vector<std::vector<float>> layer_samples;  // [num_layers][sample_rows * hidden_size]
};

struct H3StepInput {
    const H3PackedLayout* layout = nullptr;
    const float* video_rows = nullptr;      // [img_rows, 96] F32
    const float* audio_rows = nullptr;      // [audio_rows, 32] F32
    const uint16_t* prompt_embeds = nullptr;// [text_len, 5120] BF16
    float t_video = 0.0f;
    float t_audio = 0.0f;
    float imgvid_cond_t = 0.999f;
    const H3LoraAdapter* lora = nullptr;    // nullptr -> base model (or baked pack)
    uint32_t step_idx = 0;                  // denoiser step index (for optional step-0 dense guardrail in Route B)
    int32_t pdd_head_idx = -1;              // >=0 -> bind pdd.video_out.{idx}.{w,b} (MiniMax-H3-Image 6-step PDD)
};

struct H3StepOutput {
    std::vector<float> video_logits;        // [img_rows, 96] F32 (condition rows masked to 0)
    std::vector<float> audio_logits;        // [audio_rows, 32] F32
    double embed_ms = 0;
    double blocks_ms = 0;
    double final_ms = 0;
    double total_ms = 0;
};

class H3DiT {
public:
    static constexpr uint32_t kPadM = 4;    // fixed padded unique_timesteps slot count for recipe reuse

    H3DiT(const Device& device, const Stream& stream, const WeightPack& weights);
    ~H3DiT();

    const H3DiTConfig& config() const { return cfg_; }

    // Single denoiser forward step: returns predicted (video_logits, audio_logits).
    H3StepOutput forward(const H3StepInput& in, H3ForwardDebug* debug = nullptr);

    // Rectified-flow Euler-eta0 multi-step denoise loop.
    // Modifies video_rows [img_rows, 96] and audio_rows [audio_rows, 32] in place.
    void denoise(const H3PackedLayout& layout, std::vector<float>& video_rows, std::vector<float>& audio_rows,
                 const uint16_t* prompt_embeds, const std::vector<float>& sigmas_video,
                 const std::vector<float>& sigmas_audio, const H3LoraAdapter* lora = nullptr,
                 const float* cond_anchor = nullptr, float imgvid_cond_noise_aug = 0.999f);

    // MiniMax-H3-Image 6-step PDD data-ward Euler velocity denoise loop (no audio, per-step PDD head 0..5).
    static constexpr float kImagePddSigmas[7] = {
        1.0f, 0.9675236940383911f, 0.8875485062599182f, 0.75f,
        0.5327739119529724f, 0.23200887441635132f, 0.0f};
    void denoise_image(const H3PackedLayout& layout, std::vector<float>& img_rows,
                       const uint16_t* prompt_embeds, const float* ref_anchor = nullptr,
                       float ref_timestep = 0.999f);

    // Build shifted sigma schedule matching minimax_h3_time_shift_sigmas.
    static std::vector<float> time_shift_sigmas(uint32_t num_steps, float shift_scale);

    // Pre-merge (or unmerge when nullptr) a LoRA adapter in-place into FP8 block weights (0 GiB extra HBM).
    void prepare_lora(const H3LoraAdapter* lora);

private:
    struct ShapeKey {
        uint32_t seq_len = 0, used_len = 0, lora_rank = 0, adaln_rank = 0, flags = 0;
        bool operator<(const ShapeKey& o) const {
            return std::tie(seq_len, used_len, lora_rank, adaln_rank, flags) <
                   std::tie(o.seq_len, o.used_len, o.lora_rank, o.adaln_rank, o.flags);
        }
    };
    struct RefinerKey {
        uint32_t text_len = 0, lora_rank = 0;
        bool operator<(const RefinerKey& o) const {
            if (text_len != o.text_len) return text_len < o.text_len;
            return lora_rank < o.lora_rank;
        }
    };

    struct MfKey {
        uint32_t seq_len = 0, used_len = 0, lora_rank = 0, adaln_rank = 0, h_sub = 0, q_chunk = 0, flags = 0;
        bool operator<(const MfKey& o) const {
            return std::tie(seq_len, used_len, lora_rank, adaln_rank, h_sub, q_chunk, flags) <
                   std::tie(o.seq_len, o.used_len, o.lora_rank, o.adaln_rank, o.h_sub, o.q_chunk, o.flags);
        }
    };
    struct MfChunkKey {
        uint32_t h_sub = 0, q_chunk = 0, k_pad = 0, pad_k = 0;
        bool operator<(const MfChunkKey& o) const {
            return std::tie(h_sub, q_chunk, k_pad, pad_k) < std::tie(o.h_sub, o.q_chunk, o.k_pad, o.pad_k);
        }
    };

    Recipe& get_time_embed_recipe();
    Recipe& get_patch_proj_recipe(uint32_t n_rows, bool is_audio);
    Recipe& get_cond_proj_recipe(uint32_t text_len);
    Recipe& get_refiner_block_recipe(uint32_t text_len, uint32_t lora_rank);
    Recipe& get_refiner_norm_recipe(uint32_t text_len);
    Recipe& get_dit_block_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank, uint32_t adaln_rank = 0);
    Recipe& get_dit_pre_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank, uint32_t h_sub,
                               uint32_t q_chunk, uint32_t adaln_rank = 0);
    Recipe& get_maxfree_chunk_recipe(uint32_t h_sub, uint32_t q_chunk, uint32_t k_pad, uint32_t pad_k);
    Recipe& get_dit_pre_fp8_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank, uint32_t h_sub,
                                   uint32_t q_chunk, bool kv_use_upad = false, bool with_blk_scores = false,
                                   bool v_aug_256 = false, uint32_t adaln_rank = 0);
    Recipe& get_fast_fp8_chunk_recipe(uint32_t h_sub, uint32_t q_chunk, uint32_t k_pad, uint32_t valid_k);
    Recipe& get_dit_post_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank, uint32_t h_sub, uint32_t q_chunk);
    Recipe& get_final_recipe(uint32_t seq_len, uint32_t adaln_rank = 0);
    Recipe& get_merge_weight_recipe(uint64_t N, uint64_t K, uint64_t R, uint64_t G);
    void restore_base_block_weights();
    void ensure_merged_lora(const H3LoraAdapter* lora);

    // Helper: BF16 linear with optional bias, and FP8 linear + optional BF16 LoRA branch.
    TensorId add_bias(Ops& o, TensorId x, TensorId bias, const std::string& name);
    TensorId adaln_linear(Ops& o, TensorId x_bf16, TensorId w, TensorId ws, TensorId bias, const std::string& name,
                          TensorId lora_a = kNoTensor, TensorId lora_b = kNoTensor);
    TensorId proj_with_lora(Ops& o, TensorId x_bf16, TensorId x_codes, TensorId x_scale,
                            TensorId w, TensorId ws, TensorId lora_a, TensorId lora_b,
                            bool is_qkv, const std::string& name);

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    H3DiTConfig cfg_;
    std::vector<float> inv_freq_;  // [16] F32 loaded from pack

    // Recipe caches
    std::unique_ptr<Recipe> r_time_embed_;
    std::map<std::pair<uint32_t, bool>, std::unique_ptr<Recipe>> r_patch_proj_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_cond_proj_;
    std::map<RefinerKey, std::unique_ptr<Recipe>> r_refiner_block_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_refiner_norm_;
    std::map<ShapeKey, std::unique_ptr<Recipe>> r_dit_block_;
    std::map<MfKey, std::unique_ptr<Recipe>> r_dit_pre_;
    std::map<MfChunkKey, std::unique_ptr<Recipe>> r_mf_chunk_;
    std::map<MfKey, std::unique_ptr<Recipe>> r_dit_pre_fp8_;
    std::map<MfChunkKey, std::unique_ptr<Recipe>> r_fp8_chunk_;
    std::map<MfKey, std::unique_ptr<Recipe>> r_dit_post_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<Recipe>> r_final_;
    std::map<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>, std::unique_ptr<Recipe>> r_merge_w_;

    // In-place dynamically merged FP8 LoRA state (0 GiB extra persistent HBM; restores from .g2w only on LoRA swap)
    bool merged_active_ = false;
    std::string merged_lora_path_;
    float merged_lora_scale_ = 0.0f;

    // Reusable device buffers (resized on demand when seq_len grows)
    uint32_t alloc_seq_ = 0;
    uint32_t alloc_text_ = 0;
    uint32_t alloc_img_ = 0;
    uint32_t alloc_aud_ = 0;
    uint32_t alloc_mf_upad_ = 0;
    uint32_t alloc_mf_kpad_ = 0;
    uint32_t alloc_mf_v_width_ = 128;
    bool alloc_mf_fp8_ = false;
    DeviceBuffer d_t_freq_, d_t_emb_;
    DeviceBuffer d_hidden_[2];     // ping-pong [seq_len, 5376] BF16
    DeviceBuffer d_rope_cos_, d_rope_sin_;  // [seq_len, 128] BF16
    DeviceBuffer d_comb_idx_, d_inv_idx_;   // [seq_len] I32
    DeviceBuffer d_vid_in_, d_vid_emb_;
    DeviceBuffer d_aud_in_, d_aud_emb_;
    DeviceBuffer d_txt_in_, d_txt_buf_[2];
    DeviceBuffer d_vid_out_, d_aud_out_;    // [seq_len, 96] and [seq_len, 32] F32
    // Split max-free / fast-FP8 / Direct MME+TPC+SRAM attention scratch buffers (allocated lazily)
    DeviceBuffer d_mf_q_, d_mf_b_, d_mf_k_, d_mf_v_, d_mf_o_, d_mf_ad_;
    DeviceBuffer d_mf_sc_, d_mf_dv_;

    // Direct Gaudi2 MME + 24-TPC + 48 MiB SRAM FlashAttention runtime & per-(U, sparse_keep) executor cache
    std::unique_ptr<DirectRuntime> direct_rt_;
    std::unique_ptr<DirectH3AttnAssets> direct_assets_;
    std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<DirectH3Attn>> direct_attn_cache_;
    DeviceBuffer d_blk_scores_;
    std::unique_ptr<HostBuffer> h_blk_scores_;
    uint64_t alloc_blk_scores_bytes_ = 0;

    // Cached RoPE table for the last layout pointer/dims to avoid recomputing across denoise steps
    uint32_t cached_rope_seq_ = 0;
    std::vector<double> cached_rope_pos_;

    void ensure_buffers(uint32_t seq_len, uint32_t text_len, uint32_t img_rows, uint32_t audio_rows);
    void ensure_mf_buffers(uint32_t u_pad, uint32_t k_pad, bool fp8_qkv, uint32_t v_width = 128);
    void upload_rope(const H3PackedLayout& layout, std::vector<float>* debug_rope_table);
};

}  // namespace g2e
