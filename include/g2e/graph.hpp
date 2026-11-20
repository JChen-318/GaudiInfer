// g2e graphs: describe a Synapse graph, compile it to a recipe (or load the identical recipe from the on-disk
// cache), launch it with the device addresses of its launch-bound tensors.
//
// A graph is recorded first and only materialised as a Synapse graph when the recipe cache misses: the cache key
// is a hash of the full description (tensors, nodes, node parameters) plus the SDK version and the TPC kernel
// libraries on GC_KERNEL_PATH (path, size, mtime), so a restart loads every recipe it compiled before and a
// changed kernel library recompiles. Shapes are outermost-first (PyTorch order); Synapse sizes are the reverse.
#pragma once

#include "g2e/runtime.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace g2e {

using TensorId = uint32_t;
inline constexpr TensorId kNoTensor = ~TensorId{0};  // an absent optional node input

struct TensorDesc {
    std::string name;
    DType dtype = DType::F32;
    std::vector<uint64_t> shape;
    // Launch-bound (persistent) tensors get their device address at every enqueue; the others are graph-internal
    // and live in the recipe's workspace.
    bool persistent = false;
    uint64_t elements() const;
    uint64_t bytes() const { return elements() * dtype_bytes(dtype); }
};

struct Binding {
    std::string name;
    const DeviceBuffer* buffer = nullptr;
    uint64_t offset = 0;
};

// recipe launches so far in this process (the profiler's enqueue numbering: hl-prof-config -g a-b)
uint64_t launch_count();

class Recipe {
public:
    ~Recipe();
    Recipe(const Recipe&) = delete;
    Recipe& operator=(const Recipe&) = delete;
    // Every persistent tensor needs exactly one binding (any order). Launches of one recipe are serialised.
    void enqueue(const Stream& stream, std::span<const Binding> bindings);
    std::span<const TensorDesc> tensors() const;  // the persistent tensors, in declaration order
    const std::string& name() const;
    uint64_t workspace_bytes() const;
    bool from_cache() const;  // loaded from the recipe cache instead of compiled
    double compile_ms() const;  // host time of compile or load
private:
    struct Impl;
    explicit Recipe(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class GraphBuilder;
};

class GraphBuilder {
public:
    explicit GraphBuilder(const Device& device);
    ~GraphBuilder();
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;

    TensorId tensor(TensorDesc desc);
    TensorId input(std::string name, DType dtype, std::vector<uint64_t> shape) {
        return tensor({std::move(name), dtype, std::move(shape), true});
    }
    TensorId temp(std::string name, DType dtype, std::vector<uint64_t> shape) {
        return tensor({std::move(name), dtype, std::move(shape), false});
    }
    // Valid until the next tensor() call (descriptors live in a vector).
    const TensorDesc& desc(TensorId id) const;
    // FP8 (fp8_143) exponent bias of a tensor (Gaudi2 supports several; default is the SDK's).
    void fp8_bias(TensorId id, uint32_t exponent_bias, double scale = 1.0);

    // Any SDK or custom TPC kernel by guid; the caller owns shapes and the parameter struct.
    void op(const char* guid, std::span<const TensorId> inputs, std::span<const TensorId> outputs,
            const void* parameters = nullptr, uint32_t bytes = 0);
    void op(const char* guid, std::initializer_list<TensorId> inputs, std::initializer_list<TensorId> outputs,
            const void* parameters = nullptr, uint32_t bytes = 0) {
        op(guid, std::span(inputs.begin(), inputs.size()), std::span(outputs.begin(), outputs.size()), parameters,
           bytes);
    }
    // MME matmul: out = op(a) @ op(b) for rank-2 (gemm) or batched (batch_gemm) operands.
    void gemm(TensorId a, TensorId b, TensorId output, bool transpose_a = false, bool transpose_b = false);
    void batch_gemm(TensorId a, TensorId b, TensorId output, bool transpose_a = false, bool transpose_b = false);
    void reshape(TensorId input, TensorId output);

    // Compile (or load from the cache) and allocate the workspace. The builder is spent afterwards.
    std::unique_ptr<Recipe> compile(const std::string& name);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Recipe cache directory ("" disables). Default: $G2E_RECIPE_CACHE.
void set_recipe_cache_dir(std::string directory);
const std::string& recipe_cache_dir();

}  // namespace g2e
