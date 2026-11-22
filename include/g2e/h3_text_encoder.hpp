// MiniMax-H3 Qwen3-VL-32B Layer-50 Text/Vision Encoder (native Gaudi2 FP8 engine).
// Runs only the first 50 decoder layers (unnormalized hidden_states[50], [seq, 5120] BF16)
// and supports instant HBM upload/release so short videos keep it resident while long (10-15s)
// videos free ~24.8 GiB HBM before DiT denoising.
#pragma once

#include "g2e/graph.hpp"
#include "g2e/ops.hpp"
#include "g2e/tokenizer.hpp"
#include "g2e/vision.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

struct H3TextEncoderConfig {
    uint32_t num_layers = 50;       // retained layers (output = hidden_states[50])
    uint32_t hidden_size = 5120;
    uint32_t intermediate_size = 25600;
    uint32_t num_heads = 64;
    uint32_t num_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t vocab_size = 151936;
    float rms_norm_eps = 1e-6f;
    double rope_theta = 5000000.0;
    uint32_t mrope_section[3] = {24, 20, 20};
    bool blocks_fp8 = true;
    bool has_vision = false;
    // Qwen3-VL vision tower config (when has_vision == true)
    uint32_t vis_depth = 27;
    uint32_t vis_hidden = 1152;
    uint32_t vis_heads = 16;
    uint32_t vis_inter = 4304;
    uint32_t vis_patch = 16;
    uint32_t vis_merge = 2;
    uint32_t vis_grid_side = 48;
    int32_t image_token_id = 151655;
    int32_t vision_start_token_id = 151652;
    int32_t vision_end_token_id = 151653;
    std::vector<uint32_t> deepstack_indexes = {8, 16, 24};

    static H3TextEncoderConfig from_pack(const WeightPack& pack);
};

struct H3MultimodalEncoded {
    std::vector<uint16_t> embeds;       // [seq_len, hidden] BF16
    std::vector<int32_t> token_tags;    // [seq_len]: 0=VIDEO (vision block), 1=TEXT
    uint32_t seq_len = 0;
};

struct Qwen21Encoded {
    std::vector<uint16_t> embeds;        // [seq_len, 4096] BF16 (after dropping system prompt prefix)
    std::vector<uint8_t> image_pad_mask; // [seq_len]: 1 at <|image_pad|> slots, 0 elsewhere
    uint32_t seq_len = 0;
};

class H3TextEncoder {
public:
    H3TextEncoder(const Device& device, const Stream& stream, std::string pack_path);
    ~H3TextEncoder();

    const H3TextEncoderConfig& config() const { return cfg_; }
    bool is_loaded() const { return loaded_; }

    // Upload weights to HBM (if not already resident). Returns upload stats.
    UploadStats load_to_hbm();
    // Release HBM weight arena (e.g. before 15s DiT denoising to reclaim ~24.8 GiB HBM).
    void release_hbm();

    // Encode text-only token IDs [seq_len] -> [seq_len, 5120] BF16 (as uint16_t).
    // Automatically loads weights if not resident.
    std::vector<uint16_t> encode_text_ids(const std::vector<int32_t>& token_ids);

    // Encode multimodal presentation: "<Picture i>: <|vision_start|>...<|vision_end|>" (video FL2VA)
    // or "Picture i: <|vision_start|>...<|vision_end|>" (H3-Image, angle_brackets=false) + prompt
    // using Qwen3-VL Vision Tower + DeepStack (layers 8/16/24 -> LLM layers 0/1/2) + Interleaved M-RoPE.
    H3MultimodalEncoded encode_multimodal(const Tokenizer& tok, const std::string& prompt,
                                          const std::vector<VisionImage>& images,
                                          bool angle_brackets = true);

    // Encode Qwen-Image-2.1 T2I or TI2I (1..4 condition images) prompt with Qwen3-VL-8B:
    // Applies system/user/assistant raw template, runs Vision + DeepStack + M-RoPE + all retained LM layers,
    // drops leading system-role tokens (_drop_idx), and returns embeds + image_pad_mask.
    Qwen21Encoded encode_qwen21(const Tokenizer& tok, const std::string& prompt,
                                const std::vector<VisionImage>& images);

private:
    struct ImgSpan { uint32_t start = 0, count = 0, llm_h = 0, llm_w = 0; };
    std::vector<uint16_t> run_multimodal_forward(const std::vector<int32_t>& ids,
                                                 const std::vector<uint8_t>& mm_type,
                                                 const std::vector<ImgSpan>& spans,
                                                 const std::vector<VisionImage>& images);
    Recipe& get_embed_recipe(uint32_t seq_len);
    Recipe& get_layer_recipe(uint32_t seq_len);
    Recipe& get_add_recipe(uint32_t seq_len);
    Recipe& get_vis_recipe(uint32_t n_patches);

    const Device& dev_;
    const Stream& stream_;
    std::string pack_path_;
    std::unique_ptr<WeightPack> pack_;
    H3TextEncoderConfig cfg_;
    bool loaded_ = false;

    std::map<uint32_t, std::unique_ptr<Recipe>> r_embed_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_layer_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_add_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_vis_;
};

}  // namespace g2e
