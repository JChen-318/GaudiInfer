// Inkling-2 vision tower on the device: ViT (class token, learned 32x32 position grid resampled bilinearly, axial 2D
// RoPE on interleaved pairs, layer-scaled residuals) + 2x2 cell merge + 2-layer projector into the text embedding space.
// Pack: tools/convert_inkling2_vision.py. Images: preprocess_inkling_image (vision.hpp).
#pragma once

#include "g2e/graph.hpp"
#include "g2e/runtime.hpp"
#include "g2e/vision.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace g2e {

class InklingVision {
public:
    InklingVision(const Device& device, const Stream& stream, const WeightPack& pack);
    ~InklingVision();
    // Features of one image, BF16 [tokens, out_hidden] at `offset` of `out` (the text model's embedding rows).
    // Synchronous.
    void encode(const VisionImage& img, const DeviceBuffer& out, uint64_t offset);
    uint32_t out_hidden() const { return out_hidden_; }
    uint32_t patch() const { return patch_; }
    uint32_t merge() const { return merge_; }
    int32_t image_token() const { return image_token_; }
    double last_ms() const { return last_ms_; }

private:
    Recipe& recipe(uint32_t patches);

    Device dev_;
    const Stream& stream_;
    const WeightPack& w_;
    uint32_t depth_, hidden_, heads_, inter_, patch_, merge_, out_hidden_, proj_hidden_, grid_side_;
    float eps_, theta_;
    int32_t image_token_;
    std::map<std::string, std::unique_ptr<Recipe>> recipes_;
    HostBuffer staging_;
    DeviceBuffer in_;
    uint64_t in_bytes_ = 0;
    double last_ms_ = 0;
};

}  // namespace g2e
