// g2e runtime: one Gaudi2 device, HBM and pinned host buffers, streams, events, copies.
#pragma once

#include <synapse_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace g2e {

// Throws std::runtime_error with the SDK's description when `status` is not synSuccess.
void check(synStatus status, const char* operation);

enum class DType : uint8_t { F32, BF16, F16, F8, I32, I64, I16, I8, U8 };
uint64_t dtype_bytes(DType type);
// device memory held by DeviceBuffers (bytes): now and the peak since start
struct DeviceMemory { uint64_t live, peak; };
DeviceMemory device_memory();
synDataType sdk_type(DType type);
const char* dtype_name(DType type);

namespace detail { struct DeviceState; }

struct DeviceInfo {
    std::string name, driver;
    uint32_t id = 0;
    uint64_t free_bytes = 0, total_bytes = 0;
};

// The process's one acquired Gaudi2 (Synapse allows one acquisition per process). Copies share the device.
class Device {
public:
    Device();
    DeviceInfo info() const;
    void synchronize() const;
    synDeviceId id() const;
    const std::shared_ptr<detail::DeviceState>& state() const { return state_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
};

// Device (HBM) memory; addresses are 128-byte aligned by the SDK.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const Device& device, uint64_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
    uint64_t address() const { return address_; }
    uint64_t size() const { return bytes_; }
    const std::shared_ptr<detail::DeviceState>& state() const { return state_; }
private:
    void reset() noexcept;
    std::shared_ptr<detail::DeviceState> state_;
    uint64_t address_ = 0, bytes_ = 0;
};

// Pinned, device-mapped host memory (the source/target of DMA copies).
class HostBuffer {
public:
    HostBuffer(const Device& device, uint64_t bytes);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    void* data() const { return data_; }
    template <class T> T* as() const { return static_cast<T*>(data_); }
    uint64_t size() const { return bytes_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
    void* data_ = nullptr;
    uint64_t bytes_ = 0;
};

class Event;
class Stream {
public:
    explicit Stream(const Device& device);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    void synchronize() const;
    void wait(const Event& event) const;
    // Asynchronous DMA. Host side must be a HostBuffer (pinned); offsets/bytes are checked against both sizes.
    void upload(const HostBuffer& src, uint64_t src_offset, const DeviceBuffer& dst, uint64_t dst_offset,
                uint64_t bytes) const;
    void download(const DeviceBuffer& src, uint64_t src_offset, const HostBuffer& dst, uint64_t dst_offset,
                  uint64_t bytes) const;
    void copy(const DeviceBuffer& src, uint64_t src_offset, const DeviceBuffer& dst, uint64_t dst_offset,
              uint64_t bytes) const;
    // Asynchronous device-side fill with zero bytes.
    void zero(const DeviceBuffer& dst, uint64_t offset, uint64_t bytes) const;
    synStreamHandle handle() const { return handle_; }
    const std::shared_ptr<detail::DeviceState>& state() const { return state_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
    synStreamHandle handle_ = nullptr;
};

class Event {
public:
    explicit Event(const Device& device);
    ~Event();
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void record(const Stream& stream);
    void synchronize() const;
    bool ready() const;
    bool recorded() const { return recorded_; }
    // Milliseconds between two recorded events (both must have completed).
    static double elapsed_ms(const Event& start, const Event& end);
    synEventHandle handle() const { return handle_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
    synEventHandle handle_ = nullptr;
    bool recorded_ = false;
};

}  // namespace g2e
