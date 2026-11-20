// g2e weight packs (.g2w, written by tools/convert_qwen35.py): a JSON header and one data region whose layout is
// the HBM layout. upload() streams the data region into a single device arena (read and DMA overlapped through
// two pinned staging buffers); every weight is then bound to a recipe as (arena, offset).
#pragma once

#include "g2e/graph.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

struct WeightEntry {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<uint64_t> shape;
    uint64_t offset = 0, bytes = 0;  // relative to the data region / arena start
};

struct UploadStats {
    double seconds = 0, gib = 0;
    double gib_per_s() const { return seconds > 0 ? gib / seconds : 0; }
};

class WeightPack {
public:
    // Reads and validates the header only.
    explicit WeightPack(std::string path);
    ~WeightPack();
    WeightPack(const WeightPack&) = delete;
    WeightPack& operator=(const WeightPack&) = delete;

    const std::string& path() const { return path_; }
    const std::string& header_json() const { return header_; }  // the raw JSON (config, layer kinds, ...)
    uint64_t data_bytes() const { return data_bytes_; }
    const std::map<std::string, WeightEntry>& entries() const { return entries_; }
    bool has(const std::string& name) const { return entries_.count(name) != 0; }
    const WeightEntry& at(const std::string& name) const;

    // Allocate the arena and copy the data region into it. `chunk_bytes` is the size of each of the two staging
    // buffers.
    UploadStats upload(const Device& device, const Stream& stream, uint64_t chunk_bytes = 256ull << 20);
    // Allocate the whole arena but copy only the selected weights (tests and bisection).
    UploadStats upload_subset(const Device& device, const Stream& stream,
                              const std::function<bool(const WeightEntry&)>& select);
    const DeviceBuffer& arena() const;
    // Bind recipe tensor `tensor` to weight `weight` (shape/dtype are checked against the recipe at enqueue time
    // only by size; use check() for an exact match).
    Binding bind(const std::string& tensor, const std::string& weight) const;
    // Read one weight's bytes from the file (host side).
    std::vector<uint8_t> read(const std::string& name) const;
    // Throws unless weight `weight` has exactly this dtype and shape.
    void check(const std::string& weight, DType dtype, const std::vector<uint64_t>& shape) const;

private:
    std::string path_, header_;
    uint64_t data_start_ = 0, data_bytes_ = 0;
    std::map<std::string, WeightEntry> entries_;
    std::unique_ptr<DeviceBuffer> arena_;
};

}  // namespace g2e
