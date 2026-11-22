// Direct Gaudi2 MME + 24-TPC + 48 MiB SRAM FP8 Bidirectional FlashAttention for MiniMax-H3 DiT (NH=56, D=128).
// Eliminates Synapse SDPA's TPC softmax bottleneck and HBM score/prob traffic by executing double-buffered
// FP8 QK/PV MME tiles in full hardware overlap with a 256-wide direct bitwise BF16->FP8 online softmax TPC kernel.
#pragma once

#include "g2e/direct_mme.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

struct DirectH3AttnAssets {
    // Large-sequence templates: BQ = 1536 (64 rows/TPC = 100% 24-TPC balance, 768 rows/MME = 100% systolic balance)
    MmeTemplate qk_1536_4096, pv_1536_4096, pv256_1536_4096, psum_1536_4096;
    MmeTemplate qk_1536_2048, pv_1536_2048, pv256_1536_2048, psum_1536_2048;
    MmeTemplate qk_1536_1024, pv_1536_1024, pv256_1536_1024, psum_1536_1024;
    MmeTemplate qk_1536_512,  pv_1536_512,  pv256_1536_512,  psum_1536_512;
    MmeTemplate qk_1536_256,  pv_1536_256,  pv256_1536_256,  psum_1536_256;
    // Small-sequence templates (U <= 512, e.g. unit tests): BQ = 256
    MmeTemplate qk_256_512,   pv_256_512,   pv256_256_512,   psum_256_512;
    MmeTemplate qk_256_256,   pv_256_256,   pv256_256_256,   psum_256_256;
    bool has_pv256 = false;

    uint64_t isa_step_fp8 = 0;

    static DirectH3AttnAssets load(DirectRuntime& rt, const std::string& desc_dir, const std::string& tpc_obj_dir);
};

struct DirectH3AttnArgs {
    uint64_t q_addr = 0;       // HBM F8   [NH, U_pad, 128]
    uint64_t k_addr = 0;       // HBM F8   [NH, K_pad, 128]
    uint64_t v_addr = 0;       // HBM F8   [NH, K_pad, 256 (pv256) or 128] (256-key interleaved)
    uint64_t o_addr = 0;       // HBM BF16 [NH, U_pad, 128]
    uint64_t sc_addr = 0;      // HBM F32  [1] (attn_scale * d_scale_q * d_scale_k)
    uint64_t dv_addr = 0;      // HBM F32  [1] (d_scale_v)
};

// Prepared reusable Direct MME+24-TPC+SRAM FlashAttention executor for fixed (NH, U).
// Supports arbitrary U from 1 to 200,000+ with bounded SCAL ECB memory (~10-40 MiB) and ~39.8 MiB SRAM.
class DirectH3Attn {
public:
    static uint32_t choose_bq(uint32_t valid_u) { return valid_u <= 512u ? 256u : 1536u; }
    static uint32_t aligned_u_pad(uint32_t valid_u) {
        const uint32_t bq = choose_bq(valid_u);
        return ((valid_u + bq - 1u) / bq) * bq;
    }
    static uint32_t aligned_k_pad(uint32_t valid_u) { return ((valid_u + 255u) / 256u) * 256u; }

    DirectH3Attn(const Device& dev, DirectRuntime& rt, const DirectH3AttnAssets& assets,
                 uint32_t num_heads, uint32_t valid_u, const DirectH3AttnArgs& init_args,
                 uint64_t sram_offset = 0, uint32_t tpc_cmd = 0xa, uint32_t out_rows = 0,
                 uint32_t kv_stride_rows = 0, uint32_t sparse_keep_full = 0, bool use_pv256 = true);
    ~DirectH3Attn();

    // Execute full (dense) or dynamic block-sparse bidirectional attention over all `num_heads` heads x `q_blocks`.
    // When sparse_keep_full() > 0, `active_full_tiles` must point to [num_heads * q_blocks_per_head * sparse_keep_full]
    // full-4096 K-tile indices (< n_full_tiles()). Returns elapsed ms.
    double run(const DirectH3AttnArgs& args, const uint16_t* active_full_tiles = nullptr);

    uint32_t bq() const { return bq_; }
    uint32_t u_pad() const { return u_pad_; }
    uint32_t k_pad() const { return k_pad_; }
    uint32_t q_blocks_per_head() const { return q_blocks_per_head_; }
    uint32_t n_full_tiles() const { return n_full_tiles_; }
    uint32_t sparse_keep_full() const { return sparse_keep_full_; }
    bool uses_pv256() const { return use_pv256_; }
    uint64_t sram_used_bytes() const { return sram_used_; }

private:
    struct KTile {
        uint32_t col_start = 0;
        uint32_t width = 0;
    };

    DirectRuntime& rt_;
    uint32_t nh_ = 0;
    uint32_t valid_u_ = 0;
    uint32_t bq_ = 0;
    uint32_t u_pad_ = 0;
    uint32_t k_pad_ = 0;
    uint32_t kv_stride_rows_ = 0;
    uint32_t q_blocks_per_head_ = 0;
    uint32_t n_full_tiles_ = 0;
    uint32_t sparse_keep_full_ = 0;
    bool use_pv256_ = false;
    uint32_t q_per_plan_ = 0;
    uint32_t sub_plans_per_q_ = 0;
    uint32_t plans_per_batch_ = 0;
    uint32_t num_banks_ = 1;
    uint64_t sram_used_ = 0;

    std::vector<KTile> k_tiles_;
    std::vector<std::unique_ptr<DirectPlan>> plans_[2];
    DeviceBuffer d_ctrl_[2];  // Double-buffered HBM I32 [max_slots, 4] = {head_idx, q_base, 0, 0}
    DeviceBuffer d_ones_;     // HBM F8  [4096, 1] = all 1.0f (0x38) for MME P@1 row sum
    DirectH3AttnArgs bound_{};
};

}  // namespace g2e
