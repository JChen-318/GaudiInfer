// Direct Gaudi2 MME + 24-TPC + 48 MiB SRAM FlashDecoding for Qwen3.8-27B Attention Core (B=8, T=16, KV=4, G=6, D=256).
#pragma once

#include "g2e/direct_mme.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

struct DirectAttnAssets {
    MmeTemplate qk_bf16_1024, pv_bf16_1024;
    MmeTemplate qk_bf16_2048, pv_bf16_2048;
    MmeTemplate qk_bf16_4096, pv_bf16_4096;
    MmeTemplate qk_f8_1024, pv_f8_1024;
    MmeTemplate qk_f8_2048, pv_f8_2048;
    MmeTemplate qk_f8_4096, pv_f8_4096;
    uint64_t isa_tile_step_bf16 = 0;
    uint64_t isa_tile_step_f8 = 0;
    uint64_t isa_q_pack_bf16 = 0;
    uint64_t isa_q_pack_f8 = 0;

    static DirectAttnAssets load(DirectRuntime& rt, const std::string& desc_dir, const std::string& tpc_obj_dir);
};

// Dynamic per-call arguments for DirectAttnCore (8 slots, T=16).
struct DirectAttnRunArgs {
    uint64_t q_in = 0;      // if pack_q_from_hbm: HBM BF16 [8, 16, 24, 256]; else SRAM/HBM [8, 4, 96, 256] (BF16 or F8)
    uint64_t att_out = 0;   // HBM or SRAM BF16 [8, 16, 6144]
    uint64_t last_tab = 0;  // HBM I32 [8, 32] (row b col 0 = pos[b]; ZERO per-round SRF patching!)
    std::array<std::array<uint64_t, 4>, 8> k_cache{};  // [slot][kv_head] base addr of [cap, 256] (BF16 or F8)
    std::array<std::array<uint64_t, 4>, 8> v_cache{};  // [slot][kv_head] base addr of [cap, 256] (BF16 or F8)
};

// Prepared Direct MME+TPC+SRAM FlashDecoding plan for 8-slot speculative attention (B=8, T=16, KV=4, G=6, D=256).
// Supports arbitrary per-slot context lengths (multiples of 1024 up to 262144+) with zero HBM score/P traffic and
// full double-buffered MME || 24-TPC ping-pong overlap.
class DirectAttnCore {
public:
    // `slot_rows[b]`: number of KV rows scanned for slot b (must be >= pos[b] + 16 and multiple of 1024).
    // `sram_offset`: byte offset from rt.sram_base() where this core's ~17.3 MiB SRAM arena begins.
    DirectAttnCore(DirectRuntime& rt, const DirectAttnAssets& assets, bool kv8, bool pack_q_from_hbm,
                   std::array<uint32_t, 8> slot_rows, const DirectAttnRunArgs& init_args,
                   uint64_t sram_offset = 0, uint32_t tpc_cmd = 0xa);
    ~DirectAttnCore();

    // Retarget dynamic HBM addresses and per-slot context positions in-place without rebuilding ECBs.
    void update_args(const DirectAttnRunArgs& args);

    // Execute all sub-plans and wait for completion. Returns total elapsed us.
    double run();

    // Append all internal DirectPlan pointers for batched DirectRuntime::flush_patches().
    void collect_plans(std::vector<DirectPlan*>& out) const;

    uint64_t sram_used_bytes() const { return sram_used_; }
    uint32_t total_tiles() const { return total_tiles_; }
    uint32_t num_subplans() const { return uint32_t(plans_.size()); }

private:
    struct TileDesc {
        uint32_t slot = 0;
        uint32_t col_start = 0;
        uint32_t width = 0;  // 1024, 2048, or 4096
        bool is_first = false;
        bool is_last = false;
        uint32_t plan_idx = 0;
    };

    bool kv8_ = false;
    bool pack_q_ = true;
    uint64_t sram_used_ = 0;
    uint32_t total_tiles_ = 0;
    std::array<uint32_t, 8> slot_rows_{};
    std::vector<TileDesc> tiles_;
    std::vector<std::unique_ptr<DirectPlan>> plans_;
    DirectAttnRunArgs cur_{};
};

}  // namespace g2e
