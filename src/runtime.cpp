#include <atomic>
#include "g2e/runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace g2e {

void check(synStatus status, const char* operation) {
    if (status == synSuccess) return;
    char description[256] = {};
    (void)synStatusGetBriefDescription(status, description, sizeof(description));
    throw std::runtime_error(std::string(operation) + ": " + description + " (Synapse status " +
                             std::to_string(status) + ")");
}

uint64_t dtype_bytes(DType type) {
    switch (type) {
        case DType::F32: case DType::I32: return 4;
        case DType::BF16: case DType::F16: case DType::I16: return 2;
        case DType::F8: case DType::I8: case DType::U8: return 1;
        case DType::I64: return 8;
    }
    throw std::invalid_argument("unknown dtype");
}

synDataType sdk_type(DType type) {
    switch (type) {
        case DType::F32: return syn_type_single;
        case DType::BF16: return syn_type_bf16;
        case DType::F16: return syn_type_fp16;
        case DType::F8: return syn_type_fp8_143;
        case DType::I32: return syn_type_int32;
        case DType::I64: return syn_type_int64;
        case DType::I16: return syn_type_int16;
        case DType::I8: return syn_type_int8;
        case DType::U8: return syn_type_uint8;
    }
    throw std::invalid_argument("unknown dtype");
}

const char* dtype_name(DType type) {
    switch (type) {
        case DType::F32: return "f32"; case DType::BF16: return "bf16"; case DType::F16: return "f16";
        case DType::F8: return "f8"; case DType::I32: return "i32"; case DType::I64: return "i64";
        case DType::I16: return "i16"; case DType::I8: return "i8"; case DType::U8: return "u8";
    }
    return "?";
}

namespace {
std::mutex lifecycle_mutex;
bool acquired = false;
void cleanup(synStatus status, const char* operation) noexcept {
    if (status != synSuccess) std::fprintf(stderr, "g2e: cleanup %s failed (status %d)\n", operation, status);
}
// GC_KERNEL_PATH: the SDK's TPC kernel library plus ours (built next to the engine; G2E_TPC_LIB overrides).
void default_kernel_path() {
    std::string path;
    const char* existing = std::getenv("GC_KERNEL_PATH");
    path = existing && *existing ? existing : "/usr/lib/habanalabs/libtpc_kernels.so";
#ifdef G2E_TPC_LIB
    const char* override_lib = std::getenv("G2E_TPC_LIB");
    const std::string ours = override_lib ? override_lib : G2E_TPC_LIB;
    if (!ours.empty() && path.find(ours) == std::string::npos) path += ":" + ours;
#endif
    setenv("GC_KERNEL_PATH", path.c_str(), 1);
}
}  // namespace

namespace detail {
struct DeviceState {
    synDeviceId id = 0;
    DeviceState() {
        std::lock_guard lock(lifecycle_mutex);
        if (acquired) throw std::logic_error("g2e: one Device per process; copy the existing Device");
        default_kernel_path();
        check(synInitialize(), "synInitialize");
        try {
            check(synDeviceAcquireByDeviceType(&id, synDeviceGaudi2), "synDeviceAcquireByDeviceType(Gaudi2)");
        } catch (...) {
            cleanup(synDestroy(), "synDestroy");
            throw;
        }
        acquired = true;
    }
    ~DeviceState() {
        std::lock_guard lock(lifecycle_mutex);
        cleanup(synDeviceSynchronize(id), "synDeviceSynchronize");
        cleanup(synDeviceRelease(id), "synDeviceRelease");
        cleanup(synDestroy(), "synDestroy");
        acquired = false;
    }
};
}  // namespace detail

Device::Device() : state_(std::make_shared<detail::DeviceState>()) {}
synDeviceId Device::id() const { return state_->id; }
DeviceInfo Device::info() const {
    DeviceInfo value;
    value.id = state_->id;
    char name[256] = {}, driver[256] = {};
    check(synDeviceGetName(name, sizeof(name), state_->id), "synDeviceGetName");
    check(synDriverGetVersion(driver, sizeof(driver)), "synDriverGetVersion");
    check(synDeviceGetMemoryInfo(state_->id, &value.free_bytes, &value.total_bytes), "synDeviceGetMemoryInfo");
    value.name = name;
    value.driver = driver;
    return value;
}
void Device::synchronize() const { check(synDeviceSynchronize(state_->id), "synDeviceSynchronize"); }

namespace {
std::atomic<uint64_t> g_live{0}, g_peak{0};
}
DeviceMemory device_memory() { return {g_live.load(), g_peak.load()}; }

DeviceBuffer::DeviceBuffer(const Device& device, uint64_t bytes) : state_(device.state()), bytes_(bytes) {
    if (bytes) check(synDeviceMalloc(state_->id, bytes, 0, 0, &address_), "synDeviceMalloc");
    const uint64_t now = g_live += bytes;
    for (uint64_t p = g_peak.load(); now > p && !g_peak.compare_exchange_weak(p, now);) {}
}
DeviceBuffer::~DeviceBuffer() { reset(); }
void DeviceBuffer::reset() noexcept {
    if (address_) {
        cleanup(synDeviceSynchronize(state_->id), "synchronize before free");
        cleanup(synDeviceFree(state_->id, address_, 0), "synDeviceFree");
        g_live -= bytes_;
    }
    address_ = 0;
    bytes_ = 0;
}
DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : state_(std::move(other.state_)), address_(std::exchange(other.address_, 0)),
      bytes_(std::exchange(other.bytes_, 0)) {}
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        address_ = std::exchange(other.address_, 0);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

HostBuffer::HostBuffer(const Device& device, uint64_t bytes) : state_(device.state()), bytes_(bytes) {
    check(synHostMalloc(state_->id, bytes ? bytes : 1, 0, &data_), "synHostMalloc");
}
HostBuffer::~HostBuffer() {
    if (data_) cleanup(synHostFree(state_->id, data_, 0), "synHostFree");
}

namespace {
void range(uint64_t size, uint64_t offset, uint64_t bytes, const char* what) {
    if (offset > size || bytes > size - offset) throw std::out_of_range(std::string("g2e copy out of range: ") + what);
}
}  // namespace

Stream::Stream(const Device& device) : state_(device.state()) {
    check(synStreamCreateGeneric(&handle_, state_->id, 0), "synStreamCreateGeneric");
}
Stream::~Stream() {
    if (handle_) {
        cleanup(synStreamSynchronize(handle_), "synStreamSynchronize");
        cleanup(synStreamDestroy(handle_), "synStreamDestroy");
    }
}
void Stream::synchronize() const { check(synStreamSynchronize(handle_), "synStreamSynchronize"); }
void Stream::wait(const Event& event) const {
    check(synStreamWaitEvent(handle_, event.handle(), 0), "synStreamWaitEvent");
}
void Stream::upload(const HostBuffer& src, uint64_t so, const DeviceBuffer& dst, uint64_t dof, uint64_t bytes) const {
    range(src.size(), so, bytes, "host source");
    range(dst.size(), dof, bytes, "device target");
    check(synMemCopyAsync(handle_, reinterpret_cast<uint64_t>(src.data()) + so, bytes, dst.address() + dof,
                          HOST_TO_DRAM), "synMemCopyAsync(H2D)");
}
void Stream::download(const DeviceBuffer& src, uint64_t so, const HostBuffer& dst, uint64_t dof, uint64_t bytes) const {
    range(src.size(), so, bytes, "device source");
    range(dst.size(), dof, bytes, "host target");
    check(synMemCopyAsync(handle_, src.address() + so, bytes, reinterpret_cast<uint64_t>(dst.data()) + dof,
                          DRAM_TO_HOST), "synMemCopyAsync(D2H)");
}
void Stream::copy(const DeviceBuffer& src, uint64_t so, const DeviceBuffer& dst, uint64_t dof, uint64_t bytes) const {
    range(src.size(), so, bytes, "device source");
    range(dst.size(), dof, bytes, "device target");
    check(synMemCopyAsync(handle_, src.address() + so, bytes, dst.address() + dof, DRAM_TO_DRAM),
          "synMemCopyAsync(D2D)");
}

void Stream::zero(const DeviceBuffer& dst, uint64_t offset, uint64_t bytes) const {
    range(dst.size(), offset, bytes, "device target");
    if (!bytes) return;
    if (offset % 4 == 0 && bytes % 4 == 0)
        check(synMemsetD32Async(dst.address() + offset, 0, bytes / 4, handle_), "synMemsetD32Async");
    else
        check(synMemsetD8Async(dst.address() + offset, 0, bytes, handle_), "synMemsetD8Async");
}

Event::Event(const Device& device) : state_(device.state()) {
    check(synEventCreate(&handle_, state_->id, EVENT_COLLECT_TIME), "synEventCreate");
}
Event::~Event() {
    if (handle_) cleanup(synEventDestroy(handle_), "synEventDestroy");
}
void Event::record(const Stream& stream) {
    check(synEventRecord(handle_, stream.handle()), "synEventRecord");
    recorded_ = true;
}
void Event::synchronize() const { check(synEventSynchronize(handle_), "synEventSynchronize"); }
bool Event::ready() const {
    const auto status = synEventQuery(handle_);
    if (status == synSuccess) return true;
    if (status == synBusy) return false;
    check(status, "synEventQuery");
    return false;
}
double Event::elapsed_ms(const Event& start, const Event& end) {
    uint64_t ns = 0;
    check(synEventElapsedTime(&ns, start.handle(), end.handle()), "synEventElapsedTime");
    return double(ns) / 1e6;
}

}  // namespace g2e
