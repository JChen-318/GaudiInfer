#include "g2e/graph.hpp"

#include <synapse_common_types.hpp>

#include <atomic>
#include <map>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace g2e {

uint64_t TensorDesc::elements() const {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

namespace {
std::string& cache_dir_storage() {
    static std::string dir = [] {
        const char* env = std::getenv("G2E_RECIPE_CACHE");
        return std::string(env ? env : "");
    }();
    return dir;
}

// Two independent 64-bit FNV-1a lanes (different offset bases): a 128-bit key for the recipe cache.
struct Hash128 {
    uint64_t a = 1469598103934665603ULL, b = 0x6c62272e07bb0142ULL;
    void add(const void* data, size_t n) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < n; ++i) {
            a = (a ^ p[i]) * 1099511628211ULL;
            b = (b ^ p[i]) * 0x100000001b3ULL;
            b ^= b >> 29;
        }
    }
    void add(const std::string& s) { add(s.data(), s.size()); add("\n", 1); }
    std::string hex() const {
        char out[33];
        std::snprintf(out, sizeof(out), "%016llx%016llx", (unsigned long long)a, (unsigned long long)b);
        return out;
    }
};

// SDK version + every TPC kernel library on GC_KERNEL_PATH (path, size, mtime): any change invalidates recipes.
const std::string& environment_salt() {
    static std::string salt = [] {
        std::ostringstream s;
        char driver[256] = {};
        (void)synDriverGetVersion(driver, sizeof(driver));
        s << "g2e-recipe-v1 driver=" << driver;
        const char* path = std::getenv("GC_KERNEL_PATH");
        std::istringstream libs(path ? path : "");
        std::string item;
        while (std::getline(libs, item, ':')) {
            if (item.empty()) continue;
            struct stat st {};
            if (stat(item.c_str(), &st) == 0) s << " " << item << ":" << st.st_size << ":" << st.st_mtime;
            else s << " " << item << ":missing";
        }
        return s.str();
    }();
    return salt;
}

void release(synStatus status, const char* what) noexcept {
    if (status != synSuccess) std::fprintf(stderr, "g2e: %s failed (status %d)\n", what, status);
}
}  // namespace

void set_recipe_cache_dir(std::string directory) { cache_dir_storage() = std::move(directory); }
const std::string& recipe_cache_dir() { return cache_dir_storage(); }

// ---------------------------------------------------------------------------------------------------------------
// G2E_NO_RECORD=1: skip the per-launch completion event (only valid while a single stream is used).
bool record_done() {
    static const bool skip = std::getenv("G2E_NO_RECORD") != nullptr;
    return !skip;
}

// One workspace shared by every recipe: launches go to one stream and never overlap, so the scratch memory of the
// largest recipe serves all of them (per-recipe workspaces of hundreds of recipes would not fit in HBM).
// G2E_PRIVATE_WORKSPACE=1 gives every recipe its own (for multi-stream use).
struct SharedWorkspace {
    std::mutex mu;
    std::unique_ptr<DeviceBuffer> buf;
    uint64_t address(const Device& dev, uint64_t bytes, const std::string& who) {
        std::lock_guard lock(mu);
        if (!buf || buf->size() < bytes) {
            buf.reset();  // frees after a device synchronize
            buf = std::make_unique<DeviceBuffer>(dev, std::max<uint64_t>(bytes, 1u << 20));
            static const bool log = std::getenv("G2E_MEMSTAT") != nullptr;
            if (log) std::fprintf(stderr, "[mem] shared workspace -> %.2f GiB (%s)\n", double(buf->size()) / (1 << 30), who.c_str());
        }
        return buf->address();
    }
};
SharedWorkspace& shared_workspace() {
    static SharedWorkspace ws;
    return ws;
}
bool private_workspace() {
    static const bool on = std::getenv("G2E_PRIVATE_WORKSPACE") != nullptr;
    return on;
}

struct Recipe::Impl {
    Device device;
    synRecipeHandle handle = nullptr;
    std::vector<TensorDesc> tensors;
    std::vector<uint64_t> ids;
    std::unordered_map<std::string, size_t> index;
    std::vector<synLaunchTensorInfo> launch;
    std::string name;
    DeviceBuffer workspace;
    uint64_t workspace_size = 0;
    Event done;
    bool cached = false;
    double ms = 0;
    std::mutex submit;
    synStreamHandle last_stream = nullptr;  // launches on one stream are ordered already
    Impl(const Device& d) : device(d), done(d) {}
    ~Impl() {
        try { device.synchronize(); } catch (const std::exception& e) { std::fprintf(stderr, "g2e: %s\n", e.what()); }
        if (handle) release(synRecipeDestroy(handle), "synRecipeDestroy");
    }
};

Recipe::Recipe(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Recipe::~Recipe() = default;
std::span<const TensorDesc> Recipe::tensors() const { return impl_->tensors; }
const std::string& Recipe::name() const { return impl_->name; }
uint64_t Recipe::workspace_bytes() const { return impl_->workspace_size; }
bool Recipe::from_cache() const { return impl_->cached; }
double Recipe::compile_ms() const { return impl_->ms; }

namespace {
std::atomic<uint64_t> g_launches{0};
}
uint64_t launch_count() { return g_launches.load(); }

void Recipe::enqueue(const Stream& stream, std::span<const Binding> bindings) {
    auto& p = *impl_;
    std::lock_guard lock(p.submit);
    ++g_launches;
    if (bindings.size() != p.tensors.size())
        throw std::invalid_argument("recipe " + p.name + ": " + std::to_string(bindings.size()) + " bindings for " +
                                    std::to_string(p.tensors.size()) + " launch tensors");
    std::vector<bool> seen(p.tensors.size(), false);
    for (const auto& b : bindings) {
        const auto it = p.index.find(b.name);
        if (it == p.index.end()) throw std::invalid_argument("recipe " + p.name + ": unknown tensor " + b.name);
        if (seen[it->second]) throw std::invalid_argument("recipe " + p.name + ": duplicate binding " + b.name);
        seen[it->second] = true;
        const auto& d = p.tensors[it->second];
        if (!b.buffer || b.buffer->state() != p.device.state())
            throw std::invalid_argument("recipe " + p.name + ": missing/foreign buffer for " + b.name);
        if (b.offset > b.buffer->size() || d.bytes() > b.buffer->size() - b.offset)
            throw std::out_of_range("recipe " + p.name + ": buffer too small for " + b.name);
        const uint64_t address = b.buffer->address() + b.offset;
        if (address % 128) throw std::invalid_argument("recipe " + p.name + ": unaligned address for " + b.name);
        p.launch[it->second].pTensorAddress = address;
    }
    // One workspace per recipe: launches must not overlap. Same-stream launches are ordered by the stream; only a
    // launch on another stream has to wait for the previous one.
    const bool other_stream = p.last_stream && p.last_stream != stream.handle();
    if (other_stream && p.done.recorded()) stream.wait(p.done);
    const uint64_t ws = private_workspace() || !p.workspace_size ? p.workspace.address()
                                                                    : shared_workspace().address(p.device, p.workspace_size, p.name);
    {   // G2E_LOG_LAUNCHES=1: one stderr line per launch (its 1-based index, as the profiler counts them, and recipe)
        static const bool log = std::getenv("G2E_LOG_LAUNCHES") != nullptr;
        static std::atomic<uint64_t> n{0};
        if (log) std::fprintf(stderr, "launch %llu %s\n", (unsigned long long)++n, p.name.c_str());
    }
    static const bool host_timing = std::getenv("G2E_HOST_TIMING") != nullptr;
    const auto l0 = std::chrono::steady_clock::now();
    check(synLaunch(stream.handle(), p.launch.data(), static_cast<uint32_t>(p.launch.size()), ws,
                    p.handle, 0), ("synLaunch " + p.name).c_str());
    if (host_timing) {  // G2E_HOST_TIMING=1: host time of synLaunch per recipe, summarised at exit
        struct Acc {
            std::mutex mu;
            std::map<std::string, std::pair<uint64_t, double>> by;
            ~Acc() {
                for (const auto& [n, v] : by)
                    std::fprintf(stderr, "synLaunch host %-44s x%6llu  avg %8.1f us\n", n.c_str(),
                                 (unsigned long long)v.first, v.second / double(v.first));
            }
        };
        static Acc acc;
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - l0).count();
        std::lock_guard lk(acc.mu);
        auto& v = acc.by[p.name];
        ++v.first;
        v.second += us;
    }
    if (record_done()) p.done.record(stream);
    p.last_stream = stream.handle();
}

// ---------------------------------------------------------------------------------------------------------------
struct GraphBuilder::Impl {
    struct Node {
        std::string guid;
        std::vector<TensorId> inputs, outputs;
        std::vector<unsigned char> params;
    };
    struct Fp8 { uint32_t bias = 0; double scale = 1.0; bool set = false; };
    Device device;
    std::vector<TensorDesc> tensors;
    std::vector<Fp8> fp8;
    std::vector<Node> nodes;
    bool spent = false;
    explicit Impl(const Device& d) : device(d) {}
    const TensorDesc& get(TensorId id) const {
        if (spent) throw std::logic_error("g2e: graph already compiled");
        if (id >= tensors.size()) throw std::out_of_range("g2e: unknown tensor id");
        return tensors[id];
    }
    std::string signature() const {
        std::ostringstream s;
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto& t = tensors[i];
            s << "T " << t.name << " " << dtype_name(t.dtype) << " " << t.persistent;
            for (auto d : t.shape) s << " " << d;
            if (fp8[i].set) s << " fp8bias=" << fp8[i].bias << " scale=" << fp8[i].scale;
            s << "\n";
        }
        for (const auto& n : nodes) {
            s << "N " << n.guid << " in";
            for (auto i : n.inputs) s << " " << (i == kNoTensor ? -1 : int64_t(i));
            s << " out";
            for (auto o : n.outputs) s << " " << o;
            s << " p";
            static const char* hexd = "0123456789abcdef";
            for (auto c : n.params) s << hexd[c >> 4] << hexd[c & 15];
            s << "\n";
        }
        return s.str();
    }
    // Build the Synapse graph from the description and compile it.
    synRecipeHandle materialize_and_compile(const std::string& name) const {
        synGraphHandle graph = nullptr;
        check(synGraphCreate(&graph, synDeviceGaudi2), "synGraphCreate");
        std::vector<synTensor> handles(tensors.size(), nullptr);
        std::vector<synSectionHandle> sections;
        std::vector<synFpQuantParam> fp8_params(tensors.size());
        synRecipeHandle recipe = nullptr;
        try {
            for (size_t i = 0; i < tensors.size(); ++i) {
                const auto& t = tensors[i];
                check(synTensorHandleCreate(&handles[i], graph, DATA_TENSOR, t.name.c_str()), "synTensorHandleCreate");
                synTensorGeometry geometry{};
                geometry.dims = static_cast<uint32_t>(t.shape.size());
                for (size_t d = 0; d < t.shape.size(); ++d) geometry.sizes[d] = t.shape[t.shape.size() - 1 - d];
                check(synTensorSetGeometry(handles[i], &geometry, synGeometrySizes), "synTensorSetGeometry");
                check(synTensorSetDeviceDataType(handles[i], sdk_type(t.dtype)), "synTensorSetDeviceDataType");
                if (fp8[i].set) {
                    fp8_params[i] = synFpQuantParam{fp8[i].scale, fp8[i].bias};
                    synFpQuantMetadata metadata{syn_type_fp8_143, &fp8_params[i], 1};
                    check(synTensorSetQuantizationData(handles[i], SYN_FP_QUANT_METADATA, &metadata, sizeof(metadata)),
                          "synTensorSetQuantizationData(fp8)");
                }
                if (t.persistent) {
                    synSectionHandle section = nullptr;
                    check(synSectionCreate(&section, 0, graph), "synSectionCreate");
                    sections.push_back(section);
                    check(synSectionSetPersistent(section, true), "synSectionSetPersistent");
                    check(synTensorAssignToSection(handles[i], section, 0), "synTensorAssignToSection");
                }
            }
            for (size_t k = 0; k < nodes.size(); ++k) {
                const auto& n = nodes[k];
                std::vector<synTensor> in, out;
                for (auto i : n.inputs) in.push_back(i == kNoTensor ? nullptr : handles[i]);
                for (auto o : n.outputs) out.push_back(handles[o]);
                const auto node_name = n.guid + "_" + std::to_string(k);
                check(synNodeCreate(graph, in.data(), out.data(), static_cast<uint32_t>(in.size()),
                                    static_cast<uint32_t>(out.size()), n.params.empty() ? nullptr : n.params.data(),
                                    static_cast<unsigned>(n.params.size()), n.guid.c_str(), node_name.c_str(),
                                    nullptr, nullptr),
                      ("synNodeCreate " + n.guid).c_str());
            }
            check(synGraphCompile(&recipe, graph, name.c_str(), nullptr), ("synGraphCompile " + name).c_str());
        } catch (...) {
            for (auto s : sections) release(synSectionDestroy(s), "synSectionDestroy");
            release(synGraphDestroy(graph), "synGraphDestroy");
            throw;
        }
        for (auto s : sections) release(synSectionDestroy(s), "synSectionDestroy");
        release(synGraphDestroy(graph), "synGraphDestroy");
        return recipe;
    }
};

GraphBuilder::GraphBuilder(const Device& device) : impl_(std::make_unique<Impl>(device)) {}
GraphBuilder::~GraphBuilder() = default;

TensorId GraphBuilder::tensor(TensorDesc desc) {
    auto& p = *impl_;
    if (p.spent) throw std::logic_error("g2e: graph already compiled");
    if (desc.name.empty() || desc.shape.empty() || desc.shape.size() > 5)
        throw std::invalid_argument("g2e: tensor needs a name and rank 1..5: " + desc.name);
    for (const auto& t : p.tensors)
        if (t.name == desc.name) throw std::invalid_argument("g2e: duplicate tensor name " + desc.name);
    p.tensors.push_back(std::move(desc));
    p.fp8.emplace_back();
    return static_cast<TensorId>(p.tensors.size() - 1);
}
const TensorDesc& GraphBuilder::desc(TensorId id) const { return impl_->get(id); }
void GraphBuilder::fp8_bias(TensorId id, uint32_t exponent_bias, double scale) {
    if (impl_->get(id).dtype != DType::F8) throw std::invalid_argument("g2e: fp8_bias on a non-FP8 tensor");
    impl_->fp8[id] = {exponent_bias, scale, true};
}

void GraphBuilder::op(const char* guid, std::span<const TensorId> inputs, std::span<const TensorId> outputs,
                      const void* parameters, uint32_t bytes) {
    auto& p = *impl_;
    Impl::Node n;
    n.guid = guid;
    for (auto i : inputs) { if (i != kNoTensor) (void)p.get(i); n.inputs.push_back(i); }
    for (auto o : outputs) { (void)p.get(o); n.outputs.push_back(o); }
    if (parameters && bytes) {
        const auto* c = static_cast<const unsigned char*>(parameters);
        n.params.assign(c, c + bytes);
    }
    p.nodes.push_back(std::move(n));
}

void GraphBuilder::gemm(TensorId a, TensorId b, TensorId output, bool transpose_a, bool transpose_b) {
    const auto& A = desc(a); const auto& B = desc(b); const auto& O = desc(output);
    if (A.shape.size() != 2 || B.shape.size() != 2 || O.shape.size() != 2)
        throw std::invalid_argument("g2e gemm: rank-2 operands");
    const uint64_t m = A.shape[transpose_a ? 1 : 0], k = A.shape[transpose_a ? 0 : 1];
    const uint64_t kb = B.shape[transpose_b ? 1 : 0], n = B.shape[transpose_b ? 0 : 1];
    if (k != kb || O.shape != std::vector<uint64_t>{m, n}) throw std::invalid_argument("g2e gemm: shape mismatch");
    synGEMMParams params(transpose_a, transpose_b);
    op("gemm", {a, b}, {output}, &params, sizeof(params));
}

void GraphBuilder::batch_gemm(TensorId a, TensorId b, TensorId output, bool transpose_a, bool transpose_b) {
    const auto& A = desc(a); const auto& B = desc(b); const auto& O = desc(output);
    if (A.shape.size() < 3 || B.shape.size() != A.shape.size() || O.shape.size() != A.shape.size())
        throw std::invalid_argument("g2e batch_gemm: matching rank >= 3");
    synGEMMParams params(transpose_a, transpose_b);
    op("batch_gemm", {a, b}, {output}, &params, sizeof(params));
}

void GraphBuilder::reshape(TensorId input, TensorId output) {
    const auto& A = desc(input); const auto& B = desc(output);
    if (A.dtype != B.dtype || A.elements() != B.elements()) throw std::invalid_argument("g2e reshape: size/dtype");
    op("reshape", {input}, {output});
}

std::unique_ptr<Recipe> GraphBuilder::compile(const std::string& name) {
    auto& p = *impl_;
    if (p.spent) throw std::logic_error("g2e: graph already compiled");
    if (p.nodes.empty() || name.empty()) throw std::invalid_argument("g2e: empty or unnamed graph");
    const auto t0 = std::chrono::steady_clock::now();
    Hash128 h;
    h.add(environment_salt());
    h.add(name);
    h.add(p.signature());
    const auto key = h.hex();
    std::string path;
    if (!recipe_cache_dir().empty()) {
        std::filesystem::create_directories(recipe_cache_dir());
        path = recipe_cache_dir() + "/" + name + "-" + key + ".recipe";
    }
    auto impl = std::make_unique<Recipe::Impl>(p.device);
    impl->name = name;
    if (!path.empty() && std::filesystem::exists(path)) {
        if (synRecipeDeSerialize(&impl->handle, path.c_str()) == synSuccess) impl->cached = true;
        else { impl->handle = nullptr; std::fprintf(stderr, "g2e: unreadable cached recipe %s, recompiling\n", path.c_str()); }
    }
    static const bool reuse_prefill = std::getenv("G2E_REUSE_PREFILL_CACHE") != nullptr;
    const bool is_batch_recipe = name.find("gdnb") != std::string::npos ||
                                 name.find("apreb") != std::string::npos ||
                                 name.find("apostb") != std::string::npos ||
                                 name.find("acore") != std::string::npos ||
                                 name.find("speccommit") != std::string::npos ||
                                 name.find("vtopb") != std::string::npos;
    if (!impl->handle && reuse_prefill && !is_batch_recipe && !recipe_cache_dir().empty()) {
        const std::string prefix = name + "-";
        const std::string suffix = ".recipe";
        std::string best_path;
        std::filesystem::file_time_type best_time{};
        for (const auto& entry : std::filesystem::directory_iterator(recipe_cache_dir())) {
            const std::string fn = entry.path().filename().string();
            if (fn.size() == prefix.size() + 32 + suffix.size() &&
                fn.compare(0, prefix.size(), prefix) == 0 &&
                fn.compare(fn.size() - suffix.size(), suffix.size(), suffix) == 0) {
                const auto wt = entry.last_write_time();
                if (best_path.empty() || wt > best_time) {
                    best_time = wt;
                    best_path = entry.path().string();
                }
            }
        }
        if (!best_path.empty() && synRecipeDeSerialize(&impl->handle, best_path.c_str()) == synSuccess) {
            impl->cached = true;
        } else {
            impl->handle = nullptr;
        }
    }
    if (!impl->handle) {
        impl->handle = p.materialize_and_compile(name);
        if (!path.empty()) {
            const auto tmp = path + ".tmp" + std::to_string(::getpid());
            if (synRecipeSerialize(impl->handle, tmp.c_str()) == synSuccess) std::filesystem::rename(tmp, path);
            else std::fprintf(stderr, "g2e: could not serialize recipe %s\n", name.c_str());
        }
    }
    p.spent = true;
    for (const auto& t : p.tensors) if (t.persistent) impl->tensors.push_back(t);
    std::vector<const char*> names;
    for (const auto& t : impl->tensors) names.push_back(t.name.c_str());
    impl->ids.resize(names.size());
    if (!names.empty())
        check(synTensorRetrieveIds(impl->handle, names.data(), impl->ids.data(), static_cast<uint32_t>(names.size())),
              "synTensorRetrieveIds");
    impl->launch.resize(impl->tensors.size());
    for (size_t i = 0; i < impl->tensors.size(); ++i) {
        const auto& t = impl->tensors[i];
        impl->index.emplace(t.name, i);
        auto& info = impl->launch[i];
        info.tensorName = impl->tensors[i].name.c_str();
        info.tensorType = DATA_TENSOR;
        info.tensorId = impl->ids[i];
        for (size_t d = 0; d < t.shape.size(); ++d) info.tensorSize[d] = t.shape[t.shape.size() - 1 - d];
    }
    uint64_t workspace = 0;
    check(synWorkspaceGetSize(&workspace, impl->handle), "synWorkspaceGetSize");
    impl->workspace_size = workspace;
    if (private_workspace()) impl->workspace = DeviceBuffer(p.device, workspace);
    impl->ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return std::unique_ptr<Recipe>(new Recipe(std::move(impl)));
}

}  // namespace g2e
