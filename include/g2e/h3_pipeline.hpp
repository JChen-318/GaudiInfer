// End-to-end MiniMax-H3 T2VA/FL2VA Audio-Video Generation Pipeline on Gaudi2.
// Orchestrates Tokenizer -> H3TextEncoder (Qwen3-VL-32B L50 FP8) -> H3DiT (FP8 + swappable BF16 LoRA)
// -> H3VideoVae (ViT3D) + H3AudioVae (BigVGAN 32kHz stereo) -> FFmpeg MP4 muxing.
#pragma once

#include "g2e/h3_audio_vae.hpp"
#include "g2e/h3_dit.hpp"
#include "g2e/h3_text_encoder.hpp"
#include "g2e/h3_video_vae.hpp"
#include "g2e/tokenizer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

struct H3PipelinePaths {
    std::string tokenizer_json;
    std::string text_encoder_g2w;
    std::string dit_g2w;
    std::string video_vae_g2w;
    std::string audio_vae_g2w;
    std::string default_lora;       // optional default acceleration LoRA .safetensors
    float default_lora_scale = 1.0f;
    std::string ffmpeg_bin = "ffmpeg";
};

struct H3GenerateRequest {
    std::string prompt;
    uint32_t width = 1344;          // 0 -> auto-resolve from first keyframe image (FL2VA)
    uint32_t height = 768;          // 0 -> auto-resolve from first keyframe image (FL2VA)
    float duration_seconds = 5.167f;
    uint32_t fps = 24;
    uint32_t num_steps = 4;         // denoiser forward passes (4 for LightX2V 4-step Turbo)
    float flow_shift = 6.0f;        // 6.0 for 768p Turbo, 12.0 for base
    float audio_flow_shift = 3.0f;
    uint64_t seed = 2101;
    std::string lora_path;          // empty -> use default_lora; "none" -> base model without LoRA
    float lora_scale = 1.0f;
    std::string output_mp4;
    // Optional FL2VA keyframe conditioning (1 image -> [0] or [-1]; 2 images -> [0, -1]).
    // Raw encoded image bytes (PNG/JPEG/WebP/etc.) decoded via stb_image.
    std::vector<std::vector<uint8_t>> keyframe_images_bytes;
    std::vector<int> keyframe_frame_indices;
};

struct H3GenerateResult {
    std::string output_mp4;
    uint32_t width = 0, height = 0, num_frames = 0, sample_rate = 32000;
    double encode_s = 0;
    double denoise_s = 0;
    double vvae_s = 0;
    double avae_s = 0;
    double mux_s = 0;
    double total_s = 0;
};

class H3Pipeline {
public:
    H3Pipeline(const Device& device, const Stream& stream, H3PipelinePaths paths);
    ~H3Pipeline();

    // Hot-swap (or unload if path == "none") the active runtime BF16 LoRA adapter.
    void set_lora(const std::string& lora_path, float scale = 1.0f);

    H3GenerateResult generate(const H3GenerateRequest& req);

    // Shape helpers matching MiniMax-H3's 17n+5 frame contract.
    static uint32_t align_frame_count(uint32_t frames);
    static uint32_t video_latent_t(uint32_t frames);
    static uint32_t audio_latent_t(float duration_s);

private:
    const Device& dev_;
    const Stream& stream_;
    H3PipelinePaths paths_;

    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<H3TextEncoder> text_enc_;
    std::unique_ptr<WeightPack> dit_pack_;
    std::unique_ptr<H3DiT> dit_;
    std::unique_ptr<WeightPack> vvae_pack_;
    std::unique_ptr<H3VideoVae> vvae_;
    std::unique_ptr<WeightPack> avae_pack_;
    std::unique_ptr<H3AudioVae> avae_;

    std::unique_ptr<H3LoraAdapter> active_lora_;
    std::string active_lora_path_;
    float active_lora_scale_ = 1.0f;
};

}  // namespace g2e
