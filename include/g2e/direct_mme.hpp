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

// One MME tile job for a dynamic grouped-GEMM stage (DirectDynMmeSet).
struct DirectMmeJob {
    const MmeTemplate* tmpl = nullptr;
    uint64_t a = 0;
    uint64_t b = 0;
    uint64_t c0 = 0;
    uint64_t c1 = 0;
};

// Reusable per-slot dynamic MME ECB buffer set (shared across all layers of one chunk slot in hbm_shared + host_shared).
// Builds diff-encoded QMAN chunks on CPU for up to `num_stages` dynamic stages and uploads ALL code in ONE SCAL PDMA.
class DirectDynMmeSet {
public:
    DirectDynMmeSet(class DirectRuntime& rt, uint32_t num_stages, uint32_t max_ecbs = 48,
                    size_t max_code_bytes = 384u << 10);
    ~DirectDynMmeSet();
    DirectDynMmeSet(const DirectDynMmeSet&) = delete;
    DirectDynMmeSet& operator=(const DirectDynMmeSet&) = delete;

    void reset();
    uint32_t set_stage(uint32_t stage_idx, std::span<const DirectMmeJob> jobs,
                       bool has_wait, uint32_t wait_target, uint32_t fence_id, uint32_t sig_sob_abs);
    void upload();

    uint32_t stage_credits(uint32_t stage_idx) const;
    size_t last_code_bytes() const;
    uint32_t last_ecb_count() const;

    struct Impl;
    Impl& impl() { return *impl_; }
    const Impl& impl() const { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

// Process-wide SCAL direct-dispatch context borrowing the existing Synapse device.
class DirectRuntime {
public:
    // Ensure SCAL_CONFIG_FILE_PATH points to scal-g2e-direct.json BEFORE Device construction.
    // Returns true if a valid SCAL direct config is now active in the process environment.
    static bool ensure_scal_config();

    // Opt-in behaviours, all off by default: the plans, waits and uploads of the models that do not ask for them stay
    // as they are. A model's Direct engine sets them once, before it builds its first plan. The environment overrides
    // each one (0 / 1): G2E_DIRECT_SPIN, G2E_DIRECT_MME_DIFF, G2E_DIRECT_PDMA_PATCH.
    struct Options {
        // wait_fence polls the host fence counter in user space (up to 50 ms) before it blocks in SCAL: no wake-up
        // latency after a short plan, at the price of one busy host core while a plan runs.
        bool spin_wait = false;
        // After the first MME descriptor of an ECB chunk only the words that changed are written (smaller plans).
        bool mme_diff = false;
        // flush_patches / flush_patches_and_upload send the dirty ECBs as PDMA commands of the compute scheduler on
        // the runtime's own stream instead of a Synapse stream copy + synchronisation.
        bool pdma_patches = false;
    };

    explicit DirectRuntime(const Device& dev);
    ~DirectRuntime();
    void set_options(const Options& o);
    const Options& options() const;
    DirectRuntime(const DirectRuntime&) = delete;
    DirectRuntime& operator=(const DirectRuntime&) = delete;

    uint64_t sram_base() const;
    uint64_t sram_size() const;
    // sram_size() less what SCAL keeps for itself at the end (the TPC barrier kernel, when the config puts it in SRAM)
    uint64_t sram_usable() const;
    uint32_t sob_base() const;
    uint32_t sob_count() const;
    uint32_t monitor_base() const;
    uint32_t monitor_count() const;
    uint32_t host_sob() const;

    // Extract .text from a tpc-clang ELF64 .o file and upload into SCAL hbm_binary (cached).
    uint64_t load_tpc_elf(const std::string& obj_path);

    // Synchronous HBM <-> SRAM copy on a temporary Synapse stream (for setup / verification).
    void copy_sync(uint64_t src, uint64_t dst, uint64_t bytes) const;
    // Fast synchronous DRAM -> host download on the runtime's persistent DMA-only patch stream.
    void download_sync(uint64_t src_dev_addr, void* dst_host, size_t bytes);
    // Direct SCAL PDMA (compute_media_scheduler PDMA_BATCH_TRANSFER -> host_shared, zero Synapse stream/ISR).
    struct PdmaH2D { const void* src; uint64_t dst; uint32_t bytes; };
    struct PdmaD2H { uint64_t src; void* dst; uint32_t bytes; };
    void pdma_d2h_sync(uint64_t src_dev_addr, void* dst_host, size_t bytes);
    void pdma_h2d_sync(const void* src_host, uint64_t dst_dev_addr, size_t bytes);
    void pdma_h2d_scatter_sync(std::span<const PdmaH2D> xfers);
    void pdma_d2h_scatter_sync(std::span<const PdmaD2H> xfers);

    // Upload all dirty patched ECBs across multiple DirectPlans in ONE batched PDMA sync.
    void flush_patches(std::span<DirectPlan* const> plans);
    // Same as flush_patches, plus one extra HBM upload (e.g., ids_ control block) in the same PDMA sync.
    void flush_patches_and_upload(std::span<DirectPlan* const> plans, uint64_t dst_addr, const void* src, size_t bytes);
    // Async versions for double-buffered CPU-patch/PDMA || GPU-chain overlap!
    void flush_patches_and_upload_async(std::span<DirectPlan* const> plans, uint64_t dst_addr, const void* src, size_t bytes);
    void wait_patches();

    // Execute a sequence of prepared DirectPlans back-to-back on-device via SM2 SOB -> Scheduler ARC FENCE_WAIT
    // chaining (eliminates inter-plan host submit/wait_fence round-trips). Returns total elapsed us.
    double run_chain(std::span<DirectPlan* const> plans);
    // Non-blocking submit + wait for double-buffered chain pipelining (requires plans fit in <= 1 chunk when pipelining).
    uint32_t submit_chain(std::span<DirectPlan* const> plans);
    void wait_chain(uint32_t in_flight);

    // Steps of one or two plans (DirectWave) back-to-back; the two plans of a step (lanes 1 and 2, DirectPlan::set_lane)
    // run concurrently with their stages dispatched in the step's interleaving order. Returns total elapsed us.
    double run_waves(std::span<const struct DirectWave> steps);

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
                 std::array<int, 4> patch_tags = {-1, -1, -1, -1}, int override_fence = -1);

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

    // Before the first stage: lane 1 or 2 gives the plan one half of the private SOBs / monitors (at most 64 SOBs,
    // 24 waiting MME stages, 27 waiting TPC stages), so a lane-1 and a lane-2 plan can run at the same time.
    void set_lane(uint32_t lane);
    // Stages so far (consecutive MME stages without a wait form one stage).
    size_t stage_count() const;
    // MME descriptors per engine in one ECB chunk (default 9, the most that fit)
    void set_mme_chunk_limit(uint32_t n);
    // Start a new MME stage on the next add_mme (even when wait_sob < 0), so wave interleaving can schedule
    // the other plan's MME stages between large MME blocks with zero extra waits/SOBs.
    void end_mme_stage();

    // Tag subsequent MME/TPC stages with an optional bit index [0..63] (-1: always active). Forces a stage break
    // whenever the tag changes. At runtime, stages whose bit is 0 in opt_mask are skipped and dependent SOB wait
    // targets are reduced by their skipped signal counts automatically.
    void set_opt_id(int id);
    void set_opt_mask(uint64_t mask);

    // Append a dynamic MME stage backed by `dyn_set`'s `dyn_stage_idx` (0 bytes in plan_code/plan_meta; reserves 2 MME
    // monitors if wait_sob >= 0). At run time, skips if inactive, arms with dyn stage's wait_target, and queues its ECBs.
    void add_dyn_mme(const DirectDynMmeSet* dyn_set, uint32_t dyn_stage_idx, int wait_sob = -1, int sig_sob = -1);

    // Dynamic wait-target slots [0..15] for TPC stages waiting on dynamic MME SOBs.
    void set_next_dyn_wait(int slot);
    void set_dyn_wait(int slot, uint32_t target);

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

// One step of DirectRuntime::run_waves: plan a, optionally plan b running concurrently. order: the dispatch sequence
// of their stages (bit 15: plan b; low 15 bits: the stage index in its plan); stages it leaves out follow in plan
// order. Stages of one plan on one engine type must keep their plan order.
struct DirectWave {
    DirectPlan* a = nullptr;
    DirectPlan* b = nullptr;
    std::vector<uint16_t> order;
};

}  // namespace g2e
