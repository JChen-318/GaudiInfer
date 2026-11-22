// Native Gaudi2 MiniMax-H3 Audio-Video generation CLI.
#include "g2e/direct_mme.hpp"
#include "g2e/h3_pipeline.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace g2e;

static std::vector<uint8_t> read_file_bytes(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open image file: " + p);
    const size_t n = size_t(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), std::streamsize(n));
    return buf;
}

int main(int argc, char** argv) {
    try {
        H3PipelinePaths paths;
        paths.tokenizer_json = "/mnt/sas/models/minimax-h3/FL2VA/tokenizer/tokenizer.json";
        paths.text_encoder_g2w = "/mnt/sas/g2e/h3_text_encoder_fp8.g2w";
        paths.dit_g2w = "/mnt/sas/g2e/h3_dit_base_fp8.g2w";
        paths.video_vae_g2w = std::filesystem::exists("/mnt/sas/g2e/h3_video_vae_dec_fp8.g2w")
            ? "/mnt/sas/g2e/h3_video_vae_dec_fp8.g2w"
            : "/mnt/sas/g2e/h3_video_vae_dec.g2w";
        paths.audio_vae_g2w = "/mnt/sas/g2e/h3_audio_vae_dec.g2w";
        paths.default_lora = "/mnt/sas/models/minimax-h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors";
        paths.ffmpeg_bin = "/home/a/h3-deploy/media-bin/ffmpeg";

        H3GenerateRequest req;
        req.prompt = "A small stream flows through a sunlit forest. The camera slowly moves toward the water. "
                     "Birds chirp in the trees and water gurgles over smooth stones. Natural continuous sound, no music.";
        req.output_mp4 = "/mnt/sas/g2e/h3_out.mp4";
        std::string first_frame_path, last_frame_path;
        bool explicit_size = false;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--prompt" && i + 1 < argc) req.prompt = argv[++i];
            else if (a == "--out" && i + 1 < argc) req.output_mp4 = argv[++i];
            else if (a == "--width" && i + 1 < argc) { req.width = uint32_t(std::stoul(argv[++i])); explicit_size = true; }
            else if (a == "--height" && i + 1 < argc) { req.height = uint32_t(std::stoul(argv[++i])); explicit_size = true; }
            else if (a == "--first-frame" && i + 1 < argc) first_frame_path = argv[++i];
            else if (a == "--last-frame" && i + 1 < argc) last_frame_path = argv[++i];
            else if (a == "--duration" && i + 1 < argc) req.duration_seconds = std::stof(argv[++i]);
            else if (a == "--steps" && i + 1 < argc) req.num_steps = uint32_t(std::stoul(argv[++i]));
            else if (a == "--flow-shift" && i + 1 < argc) req.flow_shift = std::stof(argv[++i]);
            else if (a == "--seed" && i + 1 < argc) req.seed = std::stoull(argv[++i]);
            else if (a == "--lora" && i + 1 < argc) { paths.default_lora = argv[++i]; req.lora_path = paths.default_lora; }
            else if (a == "--lora-scale" && i + 1 < argc) { paths.default_lora_scale = std::stof(argv[++i]); req.lora_scale = paths.default_lora_scale; }
            else if (a == "--dit" && i + 1 < argc) paths.dit_g2w = argv[++i];
            else if ((a == "--text-enc" || a == "--text") && i + 1 < argc) paths.text_encoder_g2w = argv[++i];
            else if (a == "--tokenizer" && i + 1 < argc) paths.tokenizer_json = argv[++i];
            else if (a == "--vvae" && i + 1 < argc) paths.video_vae_g2w = argv[++i];
            else if (a == "--avae" && i + 1 < argc) paths.audio_vae_g2w = argv[++i];
            else if (a == "--ffmpeg" && i + 1 < argc) paths.ffmpeg_bin = argv[++i];
        }
        if (!first_frame_path.empty()) {
            req.keyframe_images_bytes.push_back(read_file_bytes(first_frame_path));
            req.keyframe_frame_indices.push_back(0);
        }
        if (!last_frame_path.empty()) {
            req.keyframe_images_bytes.push_back(read_file_bytes(last_frame_path));
            req.keyframe_frame_indices.push_back(-1);
        }
        if (!req.keyframe_images_bytes.empty() && !explicit_size) {
            req.width = 0;
            req.height = 0;
        }

        DirectRuntime::ensure_scal_config();
        Device dev;
        Stream st(dev);
        H3Pipeline pipe(dev, st, std::move(paths));
        pipe.generate(req);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
