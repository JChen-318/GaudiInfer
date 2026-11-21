// Direct Gaudi2 MME + TPC + SRAM execution via Synapse's SCAL instance (zero graph compiler overhead).
// Requires SCAL_CFG_FILE_PATH pointing to tools/scal-g2e-direct.json BEFORE Device construction.
#pragma once

#include "g2e/runtime.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace g2e {

struct MmeDesc {
    uint32_t act = 0;
    uint32_t engine = 0;  // 0: DCORE0_MME_0, 1: DCORE2_MME_0
    std::array<uint32_t, 182> words{};
    bool signals() const { return ((words[(0x148 - 8) / 4] >> 6) & 1u) != 0; }
};

struct MmeTemplate {
    std::vector<MmeDesc> descs;
    uint32_t signal_count = 0;  // number of descriptors with signalEn0 == 1
    uint32_t engine_mask = 0;   // bit 0: engine 0 used, bit 1: engine 1 used

    // Parse all act_*_mme_descriptor_dump_*.txt in `dir` (captured with PRINT_MME_DESCRIPTORS=1).
    static MmeTemplate load_dir(const std::string& dir);
};

// Hardware TPC tensor descriptor parameters (dim 0 fastest, matching Gaudi2TpcDesc).
struct DirectTpcTensor {
    uint64_t addr = 0;
    DType dtype = DType::BF16;
    uint32_t dims = 2;  // 1..5
    uint32_t sizes[5] = {1, 1, 1, 1, 1};
    uint32_t strides[5] = {0, 0, 0, 0, 0};  // 0 => compact row-major from sizes[]
    uint32_t padding = 0;
};

class DirectPlan;

// Process-wide SCAL direct-dispatch context borrowing the existing Synapse device.
class DirectRuntime {
public:
    explicit DirectRuntime(const Device& dev);
    ~DirectRuntime();
    DirectRuntime(const DirectRuntime&) = delete;
    DirectRuntime& operator=(const DirectRuntime&) = delete;

    uint64_t sram_base() const;
    uint64_t sram_size() const;
    uint32_t sob_base() const;
    uint32_t sob_count() const;
    uint32_t monitor_base() const;
    uint32_t monitor_count() const;
    uint32_t host_sob() const;

    // Extract .text from a tpc-clang ELF64 .o file and upload into SCAL hbm_binary (cached).
    uint64_t load_tpc_elf(const std::string& obj_path);

    // Synchronous HBM <-> SRAM copy on a temporary Synapse stream (for setup / verification).
    void copy_sync(uint64_t src, uint64_t dst, uint64_t bytes) const;

    // Upload all dirty patched ECBs across multiple DirectPlans in ONE batched PDMA sync.
    void flush_patches(std::span<DirectPlan* const> plans);
    // Same as flush_patches, plus one extra HBM upload (e.g., ids_ control block) in the same PDMA sync.
    void flush_patches_and_upload(std::span<DirectPlan* const> plans, uint64_t dst_addr, const void* src, size_t bytes);

    // Execute a sequence of prepared DirectPlans back-to-back on-device via SM2 SOB -> Scheduler ARC FENCE_WAIT
    // chaining (eliminates inter-plan host submit/wait_fence round-trips). Returns total elapsed us.
    double run_chain(std::span<DirectPlan* const> plans);

    bool chain_enabled() const;

    // Pre-interleave an F32 weight scale vector [N] so each 128-element chunk holds [even 64, odd 64],
    // matching v_convert_bf16_to_f32_all_b lane order with zero TPC shuffle cost.
    static std::vector<float> interleave_f32_scales(std::span<const float> src);

    struct Impl;
    Impl& impl() { return *impl_; }
    const Impl& impl() const { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

// A prepared sequence of MME and 24-TPC stages synchronized entirely in hardware via SM2 SOBs + monitors.
class DirectPlan {
public:
    explicit DirectPlan(DirectRuntime& rt, uint32_t init_mon = 0);
    ~DirectPlan();
    DirectPlan(const DirectPlan&) = delete;
    DirectPlan& operator=(const DirectPlan&) = delete;

    // Append an MME stage (both engines). If wait_sob >= 0, active engines wait on fence3 until
    // sob_base + wait_sob >= wait_target. All signaling descriptors increment sob_base + sig_sob
    // (or host_sob if sig_sob < 0).
    // Optional patch_tags {a, b, c0, c1}: if >= 0, records dynamic 64-bit address patch locations
    // so patch_addr(tag, new_addr) can retarget HBM pointers in-place after prepare() with zero rebuild.
    void add_mme(const MmeTemplate& tmpl, uint64_t a, uint64_t b, uint64_t c0, uint64_t c1 = 0,
                 int wait_sob = -1, uint32_t wait_target = 0, int sig_sob = -1,
                 std::array<int, 4> patch_tags = {-1, -1, -1, -1});

    // Run two independent single-engine GEMMs in parallel (engine 0 uses (a0,b0,c0), engine 1 uses (a1,b1,c1)),
    // replicating tmpl's engine-0 descriptor(s) onto both MME engines. Signals sig_sob twice per engine-0 signal.
    void add_mme_pair(const MmeTemplate& tmpl,
                      uint64_t a0, uint64_t b0, uint64_t c0,
                      uint64_t a1, uint64_t b1, uint64_t c1,
                      int wait_sob = -1, uint32_t wait_target = 0, int sig_sob = -1,
                      std::array<int, 4> patch0 = {-1, -1, -1, -1},
                      std::array<int, 4> patch1 = {-1, -1, -1, -1});

    // Append a 24-TPC stage splitting `grid_sizes[split_dim]` (starting at `split_base`) across `tpcs` engines.
    // Optional `tensor_patch_tags` (size 0 or tensors.size()): tag >= 0 records patch points for that tensor's base addr.
    // Optional `srf_patch_tags` (size 0 or srfs.size()): tag >= 0 records patch points for that 32-bit SRF value.
    void add_tpc_grid(uint64_t isa, std::span<const DirectTpcTensor> tensors, std::span<const uint32_t> srfs,
                      std::array<uint32_t, 5> grid_sizes, uint32_t split_dim, uint32_t tpcs = 24,
                      int wait_sob = -1, uint32_t wait_target = 0, int sig_sob = -1, uint32_t tpc_cmd = 0xf,
                      std::span<const int> tensor_patch_tags = {}, uint32_t split_base = 0,
                      std::span<const int> srf_patch_tags = {});

    // Convenience wrapper splitting `rows` across `tpcs` engines along tid_dim_1 (tid_dim_0 = 1).
    void add_tpc(uint64_t isa, std::span<const DirectTpcTensor> tensors, std::span<const uint32_t> srfs,
                 uint32_t rows, uint32_t tpcs = 24, int wait_sob = -1, uint32_t wait_target = 0,
                 int sig_sob = -1, uint32_t tpc_cmd = 0xf, std::span<const int> tensor_patch_tags = {},
                 std::span<const int> srf_patch_tags = {});

    // Validate every QMAN packet, upload ECBs to hbm_shared with readback verification, and pre-configure monitors.
    void prepare();

    // Retarget all MME/TPC descriptor base addresses tagged with `tag` directly inside uploaded hbm_shared ECBs.
    // Must be called after prepare() and before run() (when no prior run() of this plan is in flight).
    void patch_addr(int tag, uint64_t addr);

    // Retarget all 32-bit TPC SRF values tagged with `tag` directly inside uploaded hbm_shared ECBs.
    void patch_srf(int tag, uint32_t val);

    // Execute the full prepared plan in ONE SCAL submission and wait for completion. Returns elapsed us.
    double run();

    struct Impl;
    Impl& impl() { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace g2e
