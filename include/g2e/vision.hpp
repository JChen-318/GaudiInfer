// Qwen3.5/3.8 vision encoder (ViT with 2D RoPE, learned position grid, 2x2 patch merger) on the device.
#pragma once

#include "g2e/graph.hpp"
#include "g2e/runtime.hpp"
#include "g2e/weights.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

// One image after the processor: patches in the processor's order (2x2 merge blocks row-major, each block's
// patches row-major), each patch [3, temporal 2, P, P] flattened.
struct VisionImage {
    uint32_t grid_h = 0, grid_w = 0;   // in patches (multiples of the merge size)
    std::vector<float> patches;        // [grid_h * grid_w, 3 * 2 * P * P]
    uint32_t tokens() const { return grid_h * grid_w / 4; }
};

// Image processor (Qwen2VLImageProcessor): smart resize to multiples of patch*merge within [min_pixels,
// max_pixels] (bicubic, antialiased when shrinking), rescale 1/255, normalise (mean 0.5, std 0.5), patchify.
// `rgb` is 8-bit interleaved RGB.
VisionImage preprocess_image(const uint8_t* rgb, int width, int height, uint64_t min_pixels, uint64_t max_pixels,
                             uint32_t patch = 16, uint32_t merge = 2);
// Decode a PNG/JPEG/... file in memory to 8-bit RGB (stb_image). Throws on failure.
std::vector<uint8_t> decode_image(const uint8_t* data, size_t size, int& width, int& height);

class Vision {
public:
    Vision(const Device& device, const Stream& stream, const WeightPack& pack);
    ~Vision();
    // Features of one image, BF16 [tokens, out_hidden] at `offset` of `out`. Synchronous.
    void encode(const VisionImage& img, const DeviceBuffer& out, uint64_t offset);
    uint32_t out_hidden() const { return out_hidden_; }
    uint64_t min_pixels() const { return min_pixels_; }
    uint64_t max_pixels() const { return max_pixels_; }
    int32_t image_token() const { return image_token_; }
    int32_t vision_start_token() const { return vision_start_; }
    int32_t vision_end_token() const { return vision_end_; }
    double last_ms() const { return last_ms_; }

private:
    Recipe& recipe(uint32_t patches);

    Device dev_;
    const Stream& stream_;
    const WeightPack& w_;
    uint32_t depth_, hidden_, heads_, inter_, patch_, merge_, out_hidden_, grid_side_;
    int32_t image_token_, vision_start_, vision_end_;
    uint64_t min_pixels_, max_pixels_;
    std::map<std::string, std::unique_ptr<Recipe>> recipes_;
    HostBuffer staging_;
    DeviceBuffer in_;      // patches, position-embedding corners and weights, rope tables
    uint64_t in_bytes_ = 0;
    double last_ms_ = 0;
};

}  // namespace g2e
