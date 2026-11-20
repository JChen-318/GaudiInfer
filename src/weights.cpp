#include "g2e/weights.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>
#include <vector>

namespace g2e {

namespace {
constexpr uint64_t kReaders = 8;
DType parse_dtype(const std::string& s) {
    if (s == "f8") return DType::F8;
    if (s == "bf16") return DType::BF16;
    if (s == "f32") return DType::F32;
    if (s == "f16") return DType::F16;
    if (s == "i32") return DType::I32;
    throw std::runtime_error("g2w: unknown dtype " + s);
}
struct Fd {
    int fd = -1;
    explicit Fd(const std::string& path) : fd(::open(path.c_str(), O_RDONLY)) {
        if (fd < 0) throw std::runtime_error("g2w: cannot open " + path + ": " + std::strerror(errno));
    }
    ~Fd() { if (fd >= 0) ::close(fd); }
    void read_exact(void* dst, uint64_t bytes, uint64_t offset) const {
        auto* p = static_cast<char*>(dst);
        while (bytes) {
            const auto n = ::pread(fd, p, bytes, static_cast<off_t>(offset));
            if (n < 0) { if (errno == EINTR) continue; throw std::runtime_error(std::string("g2w: read: ") + std::strerror(errno)); }
            if (n == 0) throw std::runtime_error("g2w: unexpected end of file");
            p += n; bytes -= uint64_t(n); offset += uint64_t(n);
        }
    }
};
}  // namespace

WeightPack::WeightPack(std::string path) : path_(std::move(path)) {
    Fd f(path_);
    char magic[16];
    f.read_exact(magic, 16, 0);
    if (std::memcmp(magic, "G2EW0001", 8) != 0) throw std::runtime_error("g2w: bad magic in " + path_);
    uint64_t n = 0;
    std::memcpy(&n, magic + 8, 8);
    if (n > (64u << 20)) throw std::runtime_error("g2w: header too large");
    header_.resize(n);
    f.read_exact(header_.data(), n, 16);
    const auto j = nlohmann::json::parse(header_);
    if (j.at("format") != "g2e-weights-1") throw std::runtime_error("g2w: unsupported format");
    data_start_ = j.at("data_start").get<uint64_t>();
    data_bytes_ = j.at("data_bytes").get<uint64_t>();
    for (const auto& [name, t] : j.at("tensors").items()) {
        WeightEntry e;
        e.name = name;
        e.dtype = parse_dtype(t.at("dtype").get<std::string>());
        e.shape = t.at("shape").get<std::vector<uint64_t>>();
        e.offset = t.at("offset").get<uint64_t>();
        e.bytes = t.at("bytes").get<uint64_t>();
        uint64_t elems = 1;
        for (auto d : e.shape) elems *= d;
        if (elems * dtype_bytes(e.dtype) != e.bytes || e.offset % 128 || e.offset + e.bytes > data_bytes_)
            throw std::runtime_error("g2w: inconsistent entry " + name);
        entries_.emplace(name, std::move(e));
    }
}
WeightPack::~WeightPack() = default;

const WeightEntry& WeightPack::at(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) throw std::out_of_range("g2w: no weight " + name);
    return it->second;
}

const DeviceBuffer& WeightPack::arena() const {
    if (!arena_) throw std::logic_error("g2w: weights not uploaded");
    return *arena_;
}

UploadStats WeightPack::upload(const Device& device, const Stream& stream, uint64_t chunk) {
    const auto t0 = std::chrono::steady_clock::now();
    arena_ = std::make_unique<DeviceBuffer>(device, data_bytes_);
    Fd f(path_);
    (void)posix_fadvise(f.fd, static_cast<off_t>(data_start_), static_cast<off_t>(data_bytes_), POSIX_FADV_SEQUENTIAL);
    HostBuffer staging[2] = {HostBuffer(device, chunk), HostBuffer(device, chunk)};
    Event copied[2] = {Event(device), Event(device)};
    for (uint64_t done = 0, i = 0; done < data_bytes_; done += chunk, ++i) {
        const uint64_t n = std::min(chunk, data_bytes_ - done);
        auto& buf = staging[i & 1];
        auto& ev = copied[i & 1];
        if (ev.recorded()) ev.synchronize();  // the DMA from this staging buffer two chunks ago has finished
        // Several requests in flight: a network filesystem (the model share is SMB) is far faster that way.
        const uint64_t piece = (n + kReaders - 1) / kReaders;
        std::vector<std::thread> readers;
        std::exception_ptr failure;
        std::mutex failure_lock;
        for (uint64_t r = 0; r < kReaders && r * piece < n; ++r)
            readers.emplace_back([&, r] {
                try {
                    const uint64_t a = r * piece, len = std::min(piece, n - a);
                    f.read_exact(static_cast<char*>(buf.data()) + a, len, data_start_ + done + a);
                } catch (...) { std::lock_guard l(failure_lock); failure = std::current_exception(); }
            });
        for (auto& t : readers) t.join();
        if (failure) std::rethrow_exception(failure);
        stream.upload(buf, 0, *arena_, done, n);
        ev.record(stream);
    }
    stream.synchronize();
    UploadStats s;
    s.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    s.gib = double(data_bytes_) / double(1ull << 30);
    return s;
}

UploadStats WeightPack::upload_subset(const Device& device, const Stream& stream,
                                      const std::function<bool(const WeightEntry&)>& select) {
    const auto t0 = std::chrono::steady_clock::now();
    arena_ = std::make_unique<DeviceBuffer>(device, data_bytes_);
    Fd f(path_);
    const uint64_t chunk = 64ull << 20;
    HostBuffer staging(device, chunk);
    uint64_t copied = 0;
    for (const auto& [name, e] : entries_) {
        if (!select(e)) continue;
        for (uint64_t done = 0; done < e.bytes; done += chunk) {
            const uint64_t n = std::min(chunk, e.bytes - done);
            f.read_exact(staging.data(), n, data_start_ + e.offset + done);
            stream.upload(staging, 0, *arena_, e.offset + done, n);
            stream.synchronize();
        }
        copied += e.bytes;
    }
    UploadStats s;
    s.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    s.gib = double(copied) / double(1ull << 30);
    return s;
}

std::vector<uint8_t> WeightPack::read(const std::string& name) const {
    const auto& e = at(name);
    std::vector<uint8_t> out(e.bytes);
    Fd f(path_);
    f.read_exact(out.data(), e.bytes, data_start_ + e.offset);
    return out;
}

Binding WeightPack::bind(const std::string& tensor, const std::string& weight) const {
    return Binding{tensor, &arena(), at(weight).offset};
}

void WeightPack::check(const std::string& weight, DType dtype, const std::vector<uint64_t>& shape) const {
    const auto& e = at(weight);
    if (e.dtype != dtype || e.shape != shape) {
        std::string want, got;
        for (auto d : shape) want += std::to_string(d) + ",";
        for (auto d : e.shape) got += std::to_string(d) + ",";
        throw std::invalid_argument("g2w: weight " + weight + " is " + dtype_name(e.dtype) + "[" + got + "], expected " +
                                    dtype_name(dtype) + "[" + want + "]");
    }
}

}  // namespace g2e
