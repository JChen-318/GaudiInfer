// MiniMax-H3 Audio VAE (BigVGAN 32kHz stereo decoder) native Gaudi2 engine.
// Decodes [2, 32, audio_t] FP32 audio latents (40 Hz) into [2, audio_t * 800] FP32 stereo
// waveforms at 32,000 Hz using Synapse spatial_convolution / dedx + SnakeBeta + Kaiser anti-aliasing.
#pragma once

#include "g2e/graph.hpp"
#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

struct H3AudioVaeConfig {
    uint32_t latent_channels = 32;
    uint32_t output_channels = 2;
    uint32_t sample_rate = 32000;
    uint32_t upsample_initial_channel = 1024;
    uint32_t upsample_rates[7] = {5, 5, 2, 2, 2, 2, 2};
    uint32_t upsample_kernels[7] = {9, 9, 4, 4, 4, 4, 4};
    uint32_t resblock_kernels[3] = {3, 7, 11};
    uint32_t resblock_dilations[3] = {1, 3, 5};
    std::vector<float> latents_mean;  // [32]
    std::vector<float> latents_std;   // [32]

    uint32_t hop_length() const { return 800; }  // 5*5*2*2*2*2*2
    static H3AudioVaeConfig from_pack(const WeightPack& pack);
};

class H3AudioVae {
public:
    H3AudioVae(const Device& device, const Stream& stream, const WeightPack& weights);
    ~H3AudioVae();

    const H3AudioVaeConfig& config() const { return cfg_; }

    // Decode stereo audio latent [2, 32, audio_t] F32 -> stereo waveform [2, audio_t * 800] F32 in [-1, 1].
    std::vector<float> decode(const float* latent, uint32_t audio_t);

private:
    Recipe& get_decode_recipe(uint32_t audio_t);

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    H3AudioVaeConfig cfg_;
    std::map<uint32_t, std::unique_ptr<Recipe>> r_decode_;
};

}  // namespace g2e
