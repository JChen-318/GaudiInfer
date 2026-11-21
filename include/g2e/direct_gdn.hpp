// Direct Gaudi2 MME + 24-TPC + 48 MiB SRAM execution for 8-slot speculative GDN + Attention-Post blocks (B=8, T=16, R=128).
#pragma once

#include "g2e/direct_mme.hpp"
#include "g2e/weights.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace g2e {

// Shared 6 MiB persistent SRAM zone (at rt.sram_base() + 0) keeping ALL inter-stage activations
// resident on-chip across all 64 layers (GDN -> DirectApre -> DirectAttnCore -> DirectApost -> GDN).
constexpr uint64_t kDirectSramResidOff = 0ull;         // BF16 [5120, 128]     = 1.25 MiB
constexpr uint64_t kDirectSramGateOff  = 0x140000ull;  // BF16 [6144, 128]     = 1.50 MiB
constexpr uint64_t kDirectSramQOff     = 0x2c0000ull;  // BF16/F8 [256,96,4,8] = 1.50 MiB
constexpr uint64_t kDirectSramAttOff   = 0x440000ull;  // BF16 [6144, 128]     = 1.50 MiB
constexpr uint64_t kDirectSramWorkOff  = 0x600000ull;  // 6.00 MiB (working scratch begins here)

// Loaded MME templates + TPC ISAs shared across all DirectGdnBlock instances in a process.
struct DirectGdnAssets {
    MmeTemplate qkv, z, ba, gram, bwr, out, guh, dn, attn_qkv;
    uint64_t isa_embed_nq = 0;
    uint64_t isa_anq_dual = 0;
    uint64_t isa_anq = 0;
    uint64_t isa_finalize = 0;
    uint64_t isa_spec_prep = 0;
    uint64_t isa_ba_siluz = 0;
    uint64_t isa_wyod = 0;
    uint64_t isa_norm_mq = 0;
    uint64_t isa_dq_silu = 0;
    uint64_t isa_dq_mq = 0;
    uint64_t isa_sig_mq = 0;
    uint64_t isa_apre_prep_bf16 = 0;
    uint64_t isa_apre_prep_f8 = 0;
    uint64_t isa_kv_scatter_bf16 = 0;
    uint64_t isa_kv_scatter_f8 = 0;
    uint64_t isa_state_commit = 0;
    uint64_t isa_conv_commit = 0;

    static DirectGdnAssets load(DirectRuntime& rt, const std::string& desc_dir, const std::string& tpc_obj_dir,
                                uint32_t batch_size = 8);
};

// Optional appended Attention pre-projection (apre_batch: in_norm -> qkv GEMM -> Q/K norm + RoPE -> Q/gate SRAM + KV cache scatter).
struct DirectApreLayerWeights {
    uint64_t in_norm1p = 0;    // BF16 [H]
    uint64_t qkv_w = 0;        // F8   [14336, H]
    uint64_t qkv_s_int = 0;    // F32  [14336] pre-interleaved
    uint64_t q_norm = 0;       // BF16 [256]
    uint64_t k_norm = 0;       // BF16 [256]
    uint64_t rope_pos = 0;     // I32  [128] HBM (ids_ + kIdsRopePos)
    uint64_t last_tab = 0;     // I32  [8, 32] HBM (ids_ + kIdsLast; row b col 0 = pos[b])
    uint64_t cos_tab = 0;      // F32  [tab, 32] HBM
    uint64_t sin_tab = 0;      // F32  [tab, 32] HBM
    uint32_t rope_tab_rows = 0;
    bool kv8 = false;
};

// Optional prepended Attention post-projection + MLP layer (apost_batch: sigmoid(gate)*att -> o_proj -> post_norm -> MLP).
struct DirectApostLayerWeights {
    uint64_t post_norm1p = 0;  // BF16 [H]
    uint64_t o_w = 0;          // F8   [H, Z]  (Z = NH * D = 6144)
    uint64_t o_s_int = 0;      // F32  [H]     pre-interleaved
    uint64_t gate_w = 0;       // F8   [F, H]  (first F=17408 rows of gate_up.w)
    uint64_t up_w = 0;         // F8   [F, H]  (last  F=17408 rows of gate_up.w)
    uint64_t gate_s_int = 0;   // F32  [F]     pre-interleaved
    uint64_t up_s_int = 0;     // F32  [F]     pre-interleaved
    uint64_t dn_w = 0;         // F8   [H, F]
    uint64_t dn_s_int = 0;     // F32  [H]     pre-interleaved
    uint64_t feat_out = 0;     // optional BF16 [R, H] drafter feature capture in HBM (0 = disabled)
};

// Static HBM pointers for one GDN layer (pre-interleaved weight scales & folded norm weights).
struct DirectGdnLayerWeights {
    uint64_t in_norm1p = 0;    // BF16 [H]
    uint64_t post_norm1p = 0;  // BF16 [H]
    uint64_t qkv_w = 0;        // F8   [C, H]  (first C=10240 rows of qkvz.w)
    uint64_t z_w = 0;          // F8   [Z, H]  (last  Z=6144  rows of qkvz.w)
    uint64_t qkv_s_int = 0;    // F32  [C]     pre-interleaved
    uint64_t z_s_int = 0;      // F32  [Z]     pre-interleaved
    uint64_t ba_w = 0;         // BF16 [96, H]
    uint64_t conv_t = 0;       // BF16 [4, C]
    uint64_t dt_bias = 0;      // BF16 [48]
    uint64_t negexpA = 0;      // F32  [48]
    uint64_t gdn_norm = 0;     // BF16 [128]
    uint64_t out_w = 0;        // F8   [H, Z]
    uint64_t out_s_int = 0;    // F32  [H]     pre-interleaved
    uint64_t gate_w = 0;       // F8   [F, H]  (first F=17408 rows of gate_up.w)
    uint64_t up_w = 0;         // F8   [F, H]  (last  F=17408 rows of gate_up.w)
    uint64_t gate_s_int = 0;   // F32  [F]     pre-interleaved
    uint64_t up_s_int = 0;     // F32  [F]     pre-interleaved
    uint64_t dn_w = 0;         // F8   [H, F]
    uint64_t dn_s_int = 0;     // F32  [H]     pre-interleaved
    uint64_t feat_out = 0;     // optional BF16 [R, H] drafter feature capture in HBM (0 = disabled)
};

// Dynamic per-call HBM addresses for a multi-layer (optional apost + NL GDN) block (B=8, T=16).
struct DirectGdnRunArgs {
    uint64_t embed_ids = 0;    // optional Span-0 direct embed gather: I32 [R] (ids_ + 0)
    uint64_t embed_table = 0;  // optional Span-0 direct embed gather: BF16 [H, V] (non-zero enables embed_nq)
    uint32_t embed_vocab = 0;
    uint64_t res_in = 0;
    uint64_t delta_in = 0;
    uint64_t gate_in = 0;   // used when apost != nullptr
    uint64_t att_in = 0;    // used when apost != nullptr
    uint64_t resid_in = 0;  // used when apost != nullptr
    uint64_t res_out = 0;
    uint64_t delta_out = 0;
    // Per GDN layer j in [0..NL-1]:
    std::vector<std::array<uint64_t, 8>> conv_in;  // [NL][8] BF16 [3, 10240]
    std::vector<std::array<uint64_t, 8>> ssm_in;   // [NL][8] F32  [48, 128, 128]
    std::vector<uint64_t> st_x;                    // [NL]    F32  [8, 16, 32, 128] (strided dim 3)
    std::vector<uint64_t> st_v;                    // [NL]    F32  [8, 16, 48, 128] (strided dim 3)
    std::vector<uint64_t> st_g;                    // [NL]    F32  [8, 16, 48]      (strided dim 2)
    std::vector<uint64_t> st_seq;                  // [NL]    BF16 [8, 19, 10240]   (strided dim 2)
    // Optional DirectApre KV cache destinations (used when apre != nullptr):
    std::array<uint64_t, 8> apre_k_dst{};          // q.k[s] + q.pos * 256 * kv_es
    std::array<uint64_t, 8> apre_v_dst{};          // q.v[s] + q.pos * 256 * kv_es
    std::array<uint32_t, 8> apre_kv_cap{};         // q.cap per slot (head stride = cap * 256 elems)
};

// Prepared Direct MME+TPC+SRAM plan for optional apost + NL consecutive GDN layers + optional apre (B=8, T=16, R=128).
class DirectGdnBlock {
public:
    // stash_strides_elems = {st_x_dim3, st_v_dim3, st_g_dim2, st_seq_dim2} (0 => compact).
    DirectGdnBlock(DirectRuntime& rt, const DirectGdnAssets& assets, uint64_t ones_f32_addr,
                   const std::vector<DirectGdnLayerWeights>& layers,
                   const DirectGdnRunArgs& init_args,
                   std::array<uint32_t, 4> stash_strides_elems = {0, 0, 0, 0},
                   uint32_t tpc_cmd = 0xa,
                   const DirectApostLayerWeights* apost = nullptr,
                   const DirectApreLayerWeights* apre = nullptr,
                   uint64_t sram_offset = 0,
                   uint32_t batch_size = 8);
    ~DirectGdnBlock();

    // Retarget dynamic HBM addresses in-place (patches only modified ECBs before run()).
    void update_args(const DirectGdnRunArgs& args);

    // Execute all layers in ONE SCAL submission and wait for completion. Returns elapsed us.
    double run();

    DirectPlan* plan() { return plan_.get(); }
    uint32_t num_layers() const { return nl_; }
    uint32_t batch_size() const { return batch_; }
    bool has_apost() const { return has_apost_; }
    bool has_apre() const { return has_apre_; }
    uint64_t sram_used_bytes() const { return sram_used_; }

private:
    uint32_t nl_ = 0;
    uint32_t batch_ = 8;
    bool has_apost_ = false;
    bool has_apre_ = false;
    uint64_t sram_used_ = 0;
    std::unique_ptr<DirectPlan> plan_;
    DirectGdnRunArgs cur_{};
};

// Dynamic HBM pointers for in-place Direct SCAL speculative GDN commit (B=8, T=16, NG GDN layers).
struct DirectGdnCommitArgs {
    std::array<uint64_t, 8> ssm{};      // IN-PLACE ssm_in == ssm_out: F32 [NG*48, 128, 128]
    std::array<uint64_t, 8> conv_out{}; // BF16 [NG, 3, 10240]
    std::array<uint64_t, 8> x{};        // F32 [NG, 16, 2*T, 128]
    std::array<uint64_t, 8> v{};        // F32 [NG, T, 48, 128]
    std::array<uint64_t, 8> g{};        // F32 [NG, T, 48]
    std::array<uint64_t, 8> seq{};      // BF16 [NG, T+3, 10240]
    std::array<uint64_t, 8> acc{};      // I32 [1]
    std::array<uint64_t, 8> rows{};     // I32 [3]
};

// Prepared Direct SCAL plan executing in-place GDN state + conv commit across all 8 slots with zero inter-slot barriers.
class DirectGdnCommit {
public:
    DirectGdnCommit(DirectRuntime& rt, const DirectGdnAssets& assets, uint32_t ng, uint32_t T,
                    const DirectGdnCommitArgs& init_args, uint32_t tpc_cmd = 0xa, uint32_t batch_size = 8);
    ~DirectGdnCommit();

    void update_args(const DirectGdnCommitArgs& args);
    double run();

private:
    uint32_t batch_ = 8;
    std::unique_ptr<DirectPlan> plan_;
    DirectGdnCommitArgs cur_{};
};

}  // namespace g2e
