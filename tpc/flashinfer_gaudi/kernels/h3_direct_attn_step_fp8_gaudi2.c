// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 Direct MME+24-TPC+48MiB-SRAM FP8 Bidirectional FlashAttention fused tile step:
//   SM(k) + ACC(k-2) [+ Finalize to HBM]
//
// ULTRA-TIGHT 4-ROW INTERLEAVED ZERO-SPILL + MME-PSUM-OFFLOAD ARCHITECTURE:
//   - Pass 1: 12 VLIW cycles per 1024 elements (hoisted IRFs, VLM score cache, BF16 row max)
//   - Pass 2: ONLY 12 VPU ops per 1024 elements (8 single-FMA mac.bf16 + 4 saturating cvt_bf16_u8 + 4 st_tnsr!)
//     Row sum P@1 is offloaded to the idle MME systolic array -> ZERO cvt_f8_bf16, ZERO add.bf16,
//     ZERO chunked F32 flushes, ZERO SUM64_QUAD on TPC!
//   - Edge rem256 vectors stored in unused VLM slots [30..31] -> ZERO VRF spills in MAX64/RESCALE!
//   - ACC(k-2): 4-row interleaved F32 O_acc + L_acc recurrence using MME-computed l_tile_in!
//
// Tensors (dim 0 fastest):
//   0  s_in        BF16 [TILE_K, BQ]       raw FP8 Q@K^T scores from MME QK(k) in SRAM (TILE_K <= 4096, %256==0)
//   1  p_out       F8   [TILE_K, BQ]       256-interleaved FP8 probs for MME PV(k) & PSUM(k) in SRAM
//   2  m_sm        F32  [1, BQ]            running exact BF16 bias (118.9 - m*kl8) for SM slot in SRAM
//   3  l_tile_in   F32  [1, BQ]            tile row sum P(k-2)@1 from MME PSUM(k-2) in SRAM
//   4  r_sm_out    F32  [1, BQ]            rescale factor exp2((b_new - b_old)/8) for tile k in SRAM
//   5  pv_in       F32  [128, BQ]          partial PV output from MME PV(k-2) in SRAM
//   6  o_acc       F32  [128, BQ]          running F32 output accumulator in SRAM
//   7  r_acc_in    F32  [1, BQ]            rescale factor for tile k-2 in SRAM
//   8  l_acc       F32  [1, BQ]            running F32 row-sum accumulator in SRAM
//   9  att_out     BF16 [128, U_pad, NH]   final BF16 attention output in HBM
//   10 score_scale F32  [1]                attn_scale * d_scale_q * d_scale_k in HBM
//   11 d_scale_v   F32  [1]                V dequant scale in HBM
//   12 ctrl_tab    I32  [4, MAX_SLOTS]     row slot_acc = {head_idx, q_base, 0, 0} in HBM (head_idx < 0 skips store)
//
// Scalar SRFs:
//   0 int valid_k       total valid key length U (cols >= valid_k are masked to 0)
//   1 int col_start_sm  first key column of tile k (multiple of 256)
//   2 int flags_sm      bit 0: do_sm, bit 1: is_first_sm
//   3 int flags_acc     bit 0: do_acc, bit 1: is_first_acc, bit 2: is_last_acc
//   4 int slot_acc      slot index into ctrl_tab for ACC(k-2) finalize
//
// Index space: dim 1 = row [0..BQ).
#include "norm_math_gaudi2.h"

static inline float64 max64_masks(float64 x, uchar256 m4, uchar256 m2, uchar256 m1) {
    const int all = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                    SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, all, x));
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, all, x));
    x = v_f32_max_b(x, v_f32_mov_group_b(x, 0xffffffff, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |
                                         SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, m4, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, m2, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, m1, 0, x));
    return x;
}

// 4-row fully interleaved max64 (0 RAW stalls across all 6 butterfly stages!)
#define MAX64_QUAD(X0, X1, X2, X3) do {                                                                 \
    const int all_ = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |\
                     SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3; \
    const int grp_ = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |                                  \
                     SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;                         \
    const float64 a0_ = v_f32_mov_dual_group_all_b((X0), 0xffffffff, 1, 0, 3, 2, all_, (X0));           \
    const float64 a1_ = v_f32_mov_dual_group_all_b((X1), 0xffffffff, 1, 0, 3, 2, all_, (X1));           \
    const float64 a2_ = v_f32_mov_dual_group_all_b((X2), 0xffffffff, 1, 0, 3, 2, all_, (X2));           \
    const float64 a3_ = v_f32_mov_dual_group_all_b((X3), 0xffffffff, 1, 0, 3, 2, all_, (X3));           \
    (X0) = v_f32_max_b((X0), a0_); (X1) = v_f32_max_b((X1), a1_);                                       \
    (X2) = v_f32_max_b((X2), a2_); (X3) = v_f32_max_b((X3), a3_);                                       \
    const float64 b0_ = v_f32_mov_dual_group_all_b((X0), 0xffffffff, 2, 3, 0, 1, all_, (X0));           \
    const float64 b1_ = v_f32_mov_dual_group_all_b((X1), 0xffffffff, 2, 3, 0, 1, all_, (X1));           \
    const float64 b2_ = v_f32_mov_dual_group_all_b((X2), 0xffffffff, 2, 3, 0, 1, all_, (X2));           \
    const float64 b3_ = v_f32_mov_dual_group_all_b((X3), 0xffffffff, 2, 3, 0, 1, all_, (X3));           \
    (X0) = v_f32_max_b((X0), b0_); (X1) = v_f32_max_b((X1), b1_);                                       \
    (X2) = v_f32_max_b((X2), b2_); (X3) = v_f32_max_b((X3), b3_);                                       \
    const float64 c0_ = v_f32_mov_group_b((X0), 0xffffffff, grp_, (X0));                                \
    const float64 c1_ = v_f32_mov_group_b((X1), 0xffffffff, grp_, (X1));                                \
    const float64 c2_ = v_f32_mov_group_b((X2), 0xffffffff, grp_, (X2));                                \
    const float64 c3_ = v_f32_mov_group_b((X3), 0xffffffff, grp_, (X3));                                \
    (X0) = v_f32_max_b((X0), c0_); (X1) = v_f32_max_b((X1), c1_);                                       \
    (X2) = v_f32_max_b((X2), c2_); (X3) = v_f32_max_b((X3), c3_);                                       \
    const float64 s0_ = v_f32_shuffle_b((X0), shuf4, 0, (X0));                                          \
    const float64 s1_ = v_f32_shuffle_b((X1), shuf4, 0, (X1));                                          \
    const float64 s2_ = v_f32_shuffle_b((X2), shuf4, 0, (X2));                                          \
    const float64 s3_ = v_f32_shuffle_b((X3), shuf4, 0, (X3));                                          \
    (X0) = v_f32_max_b((X0), s0_); (X1) = v_f32_max_b((X1), s1_);                                       \
    (X2) = v_f32_max_b((X2), s2_); (X3) = v_f32_max_b((X3), s3_);                                       \
    const float64 t0_ = v_f32_shuffle_b((X0), shuf2, 0, (X0));                                          \
    const float64 t1_ = v_f32_shuffle_b((X1), shuf2, 0, (X1));                                          \
    const float64 t2_ = v_f32_shuffle_b((X2), shuf2, 0, (X2));                                          \
    const float64 t3_ = v_f32_shuffle_b((X3), shuf2, 0, (X3));                                          \
    (X0) = v_f32_max_b((X0), t0_); (X1) = v_f32_max_b((X1), t1_);                                       \
    (X2) = v_f32_max_b((X2), t2_); (X3) = v_f32_max_b((X3), t3_);                                       \
    const float64 u0_ = v_f32_shuffle_b((X0), shuf1, 0, (X0));                                          \
    const float64 u1_ = v_f32_shuffle_b((X1), shuf1, 0, (X1));                                          \
    const float64 u2_ = v_f32_shuffle_b((X2), shuf1, 0, (X2));                                          \
    const float64 u3_ = v_f32_shuffle_b((X3), shuf1, 0, (X3));                                          \
    (X0) = v_f32_max_b((X0), u0_); (X1) = v_f32_max_b((X1), u1_);                                       \
    (X2) = v_f32_max_b((X2), u2_); (X3) = v_f32_max_b((X3), u3_);                                       \
} while (0)

// 4-row interleaved exp2((bn - bo) * 0.125f) with exact 1.0 when diff==0 and 0.0 when t < -120
#define RESCALE_QUAD_BIAS(R0, R1, R2, R3, D0, D1, D2, D3) do {                                          \
    const float64 t0_ = (D0) * 0.125f, t1_ = (D1) * 0.125f, t2_ = (D2) * 0.125f, t3_ = (D3) * 0.125f;   \
    const float64 tt0_ = v_f32_max_b(t0_, -120.0f) + 127.0f;                                            \
    const float64 tt1_ = v_f32_max_b(t1_, -120.0f) + 127.0f;                                            \
    const float64 tt2_ = v_f32_max_b(t2_, -120.0f) + 127.0f;                                            \
    const float64 tt3_ = v_f32_max_b(t3_, -120.0f) + 127.0f;                                            \
    const int64 n0_ = v_convert_f32_to_i32_b(tt0_, SW_RHNE);                                            \
    const int64 n1_ = v_convert_f32_to_i32_b(tt1_, SW_RHNE);                                            \
    const int64 n2_ = v_convert_f32_to_i32_b(tt2_, SW_RHNE);                                            \
    const int64 n3_ = v_convert_f32_to_i32_b(tt3_, SW_RHNE);                                            \
    const float64 f0_ = tt0_ - v_convert_i32_to_f32_b(n0_, 0);                                          \
    const float64 f1_ = tt1_ - v_convert_i32_to_f32_b(n1_, 0);                                          \
    const float64 f2_ = tt2_ - v_convert_i32_to_f32_b(n2_, 0);                                          \
    const float64 f3_ = tt3_ - v_convert_i32_to_f32_b(n3_, 0);                                          \
    float64 p0_ = v_f32_mac_b((float64)0.23842697f, f0_, (float64)0.7034453f);                          \
    float64 p1_ = v_f32_mac_b((float64)0.23842697f, f1_, (float64)0.7034453f);                          \
    float64 p2_ = v_f32_mac_b((float64)0.23842697f, f2_, (float64)0.7034453f);                          \
    float64 p3_ = v_f32_mac_b((float64)0.23842697f, f3_, (float64)0.7034453f);                          \
    const int64 b0_ = n0_ << 23, b1_ = n1_ << 23, b2_ = n2_ << 23, b3_ = n3_ << 23;                     \
    p0_ = v_f32_mac_b(p0_, f0_, (float64)1.00044307f);                                                  \
    p1_ = v_f32_mac_b(p1_, f1_, (float64)1.00044307f);                                                  \
    p2_ = v_f32_mac_b(p2_, f2_, (float64)1.00044307f);                                                  \
    p3_ = v_f32_mac_b(p3_, f3_, (float64)1.00044307f);                                                  \
    const float64 e0_ = p0_ * *((float64*)&b0_), e1_ = p1_ * *((float64*)&b1_);                         \
    const float64 e2_ = p2_ * *((float64*)&b2_), e3_ = p3_ * *((float64*)&b3_);                         \
    (R0) = v_f32_sel_eq_f32_b((D0), (float64)0.0f, (float64)1.0f, v_f32_sel_less_f32_b(t0_, -120.0f, (float64)0.0f, e0_)); \
    (R1) = v_f32_sel_eq_f32_b((D1), (float64)0.0f, (float64)1.0f, v_f32_sel_less_f32_b(t1_, -120.0f, (float64)0.0f, e1_)); \
    (R2) = v_f32_sel_eq_f32_b((D2), (float64)0.0f, (float64)1.0f, v_f32_sel_less_f32_b(t2_, -120.0f, (float64)0.0f, e2_)); \
    (R3) = v_f32_sel_eq_f32_b((D3), (float64)0.0f, (float64)1.0f, v_f32_sel_less_f32_b(t3_, -120.0f, (float64)0.0f, e3_)); \
} while (0)

static inline float64 exp_biased(float64 x, float kl, float64 nb) {
    const float64 t = v_f32_mac_b(x, kl, nb);
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.23842697f;
    p = v_f32_mac_b(p, f, 0.7034453f);
    p = v_f32_mac_b(p, f, 1.00044307f);
    const int64 bits = n << 23;
    return p * *((float64*)&bits);
}

static inline float64 rcp3_f32(float64 v) {
    float64 est = as_float64((int64)0x7ef311c3 - as_int64(v));
    est = est * (2.0f - v * est);
    est = est * (2.0f - v * est);
    est = est * (2.0f - v * est);
    return est;
}

static inline bfloat128 f32_to_bf16_lo64(float64 x) {
    const bfloat128 sp = v_convert_f32_to_bf16_b(x, 0, SW_RHNE, (bfloat128)0);
    bfloat128 pk = v_bf16_pack_b(sp, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0);
    pk = v_bf16_pack_b(sp, SW_GROUP_1 | SW_STRIDE_2, pk);
    return v_bf16_mov_dual_group_pack_b(pk, SW_PACK21, (bfloat128)0);
}

static inline bfloat128 f32_pair_to_bf16_linear_fast(float64 lo, float64 hi) {
    bfloat128 p0 = f32_to_bf16_lo64(lo);
    const bfloat128 p1 = f32_to_bf16_lo64(hi);
    p0 = v_bf16_mov_dual_group_b(p1, 0xffffffff, 0, 2, MkWr(1, 1), p0);
    p0 = v_bf16_mov_dual_group_b(p1, 0xffffffff, 1, 3, MkWr(1, 1), p0);
    return p0;
}

// 2-row interleaved 128-wide F32 -> linear BF16 pack (fits cleanly in 12 VRFs with ZERO VLM spills!)
#define PACK_F32_TO_BF16_PAIR(OB0, OB1, LO0, HI0, LO1, HI1) do {                                        \
    const bfloat128 s0a_ = v_convert_f32_to_bf16_b((LO0), 0, SW_RHNE, (bfloat128)0);                    \
    const bfloat128 s0b_ = v_convert_f32_to_bf16_b((HI0), 0, SW_RHNE, (bfloat128)0);                    \
    const bfloat128 s1a_ = v_convert_f32_to_bf16_b((LO1), 0, SW_RHNE, (bfloat128)0);                    \
    const bfloat128 s1b_ = v_convert_f32_to_bf16_b((HI1), 0, SW_RHNE, (bfloat128)0);                    \
    bfloat128 k0a_ = v_bf16_pack_b(s0a_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0);                       \
    bfloat128 k0b_ = v_bf16_pack_b(s0b_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0);                       \
    bfloat128 k1a_ = v_bf16_pack_b(s1a_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0);                       \
    bfloat128 k1b_ = v_bf16_pack_b(s1b_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0);                       \
    k0a_ = v_bf16_pack_b(s0a_, SW_GROUP_1 | SW_STRIDE_2, k0a_);                                         \
    k0b_ = v_bf16_pack_b(s0b_, SW_GROUP_1 | SW_STRIDE_2, k0b_);                                         \
    k1a_ = v_bf16_pack_b(s1a_, SW_GROUP_1 | SW_STRIDE_2, k1a_);                                         \
    k1b_ = v_bf16_pack_b(s1b_, SW_GROUP_1 | SW_STRIDE_2, k1b_);                                         \
    bfloat128 p0a_ = v_bf16_mov_dual_group_pack_b(k0a_, SW_PACK21, (bfloat128)0);                       \
    const bfloat128 p0b_ = v_bf16_mov_dual_group_pack_b(k0b_, SW_PACK21, (bfloat128)0);                 \
    bfloat128 p1a_ = v_bf16_mov_dual_group_pack_b(k1a_, SW_PACK21, (bfloat128)0);                       \
    const bfloat128 p1b_ = v_bf16_mov_dual_group_pack_b(k1b_, SW_PACK21, (bfloat128)0);                 \
    p0a_ = v_bf16_mov_dual_group_b(p0b_, 0xffffffff, 0, 2, MkWr(1, 1), p0a_);                           \
    p1a_ = v_bf16_mov_dual_group_b(p1b_, 0xffffffff, 0, 2, MkWr(1, 1), p1a_);                           \
    (OB0) = v_bf16_mov_dual_group_b(p0b_, 0xffffffff, 1, 3, MkWr(1, 1), p0a_);                          \
    (OB1) = v_bf16_mov_dual_group_b(p1b_, 0xffffffff, 1, 3, MkWr(1, 1), p1a_);                          \
} while (0)

void main(tensor s_in, tensor p_out, tensor m_sm, tensor l_tile_in, tensor r_sm_out,
          tensor pv_in, tensor o_acc, tensor r_acc_in, tensor l_acc, tensor att_out,
          tensor score_scale, tensor d_scale_v, tensor ctrl_tab,
          int valid_k, int col_start_sm, int flags_sm, int flags_acc, int slot_acc)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int tile_k = get_dim_size(s_in, 0);
    const int pv_w = get_dim_size(pv_in, 0);
    const int do_sm = flags_sm & 1;
    const int is_first_sm = (flags_sm >> 1) & 1;
    const int do_acc = flags_acc & 1;
    const int is_first_acc = (flags_acc >> 1) & 1;
    const int is_last_acc = (flags_acc >> 2) & 1;

    const int5 zc = {0, 0, 0, 0, 0};
    const float sc_f = s_f32_ld_g((__global__ float*)gen_addr(zc, score_scale));
    const float dv_f = s_f32_ld_g((__global__ float*)gen_addr(zc, d_scale_v));
    const float kl8_f = sc_f * 11.541560327111707f;      // 8 * score_scale * log2(e)
    const float64_pair_t kl8p = {(float64)kl8_f, (float64)kl8_f};
    const bfloat128 kl8_b = v_convert_f32_to_bf16_all_b(kl8p);
    const float64 kl8_bf = v_convert_bf16_to_f32_all_b(kl8_b).v1;
    const bfloat128 zero_b = 0;
    const bfloat128 neg_b = -3.0e38f;
    const minifloat256 zero_f8 = 0;

    // Precompute butterfly shuffle masks ONCE outside the row loop!
    const uint64 lanes8 = read_lane_id_4b_b() & 7;
    const uint64 c4_u = ((lanes8 ^ 4) | 0x80) * 0x01010101;
    const uint64 c2_u = ((lanes8 ^ 2) | 0x80) * 0x01010101;
    const uint64 c1_u = ((lanes8 ^ 1) | 0x80) * 0x01010101;
    const uchar256 shuf4 = *((const uchar256*)&c4_u);
    const uchar256 shuf2 = *((const uchar256*)&c2_u);
    const uchar256 shuf1 = *((const uchar256*)&c1_u);

    int head_idx = -1;
    int q_base = 0;
    if (do_acc && is_last_acc) {
        int5 cc0 = {0, slot_acc, 0, 0, 0};
        int5 cc1 = {1, slot_acc, 0, 0, 0};
        head_idx = s_i32_ld_g((__global__ int*)gen_addr(cc0, ctrl_tab));
        q_base = s_i32_ld_g((__global__ int*)gen_addr(cc1, ctrl_tab));
    }

    int visible = valid_k - col_start_sm;
    if (visible < 0) visible = 0;
    if (visible > tile_k) visible = tile_k;
    const int full_pairs = visible >> 8;     // full 256-col pairs (0..16)
    const int rem256 = visible & 255;

    // 32 KiB VLM cache (4 rows x 4096 BF16 cols).
    // When rem256 > 0, full_pairs <= 15 so slots [30..31] store masked edge vectors with ZERO VRF spills!
    bfloat128 s_vlm0[32];
    bfloat128 s_vlm1[32];
    bfloat128 s_vlm2[32];
    bfloat128 s_vlm3[32];

    const int row_end = end[1];
    int row = start[1];

    // =========================================================================
    // FAST PATH: 4-ROW INTERLEAVED (64 rows/TPC on BQ=1536 -> 100% hits this!)
    // =========================================================================
    for (; row + 4 <= row_end; row += 4) {
        const int r0 = row, r1 = row + 1, r2 = row + 2, r3 = row + 3;
        int5 rc0 = {0, r0, 0, 0, 0}, rc1 = {0, r1, 0, 0, 0};
        int5 rc2 = {0, r2, 0, 0, 0}, rc3 = {0, r3, 0, 0, 0};

        // ---- Part A: 4-row interleaved online 256-wide bitwise BF16->FP8 softmax (MME computes P@1!) ----
        if (do_sm) {
            if (visible == 0) {
                for (int col = 0; col < tile_k; col += 256) {
                    v_f8_st_tnsr((int5){col, r0, 0, 0, 0}, p_out, zero_f8);
                    v_f8_st_tnsr((int5){col, r1, 0, 0, 0}, p_out, zero_f8);
                    v_f8_st_tnsr((int5){col, r2, 0, 0, 0}, p_out, zero_f8);
                    v_f8_st_tnsr((int5){col, r3, 0, 0, 0}, p_out, zero_f8);
                }
                if (is_first_sm) {
                    v_f32_st_tnsr(rc0, m_sm, (float64)(1.0e30f));
                    v_f32_st_tnsr(rc1, m_sm, (float64)(1.0e30f));
                    v_f32_st_tnsr(rc2, m_sm, (float64)(1.0e30f));
                    v_f32_st_tnsr(rc3, m_sm, (float64)(1.0e30f));
                    v_f32_st_tnsr(rc0, r_sm_out, (float64)0.0f);
                    v_f32_st_tnsr(rc1, r_sm_out, (float64)0.0f);
                    v_f32_st_tnsr(rc2, r_sm_out, (float64)0.0f);
                    v_f32_st_tnsr(rc3, r_sm_out, (float64)0.0f);
                } else {
                    v_f32_st_tnsr(rc0, r_sm_out, (float64)1.0f);
                    v_f32_st_tnsr(rc1, r_sm_out, (float64)1.0f);
                    v_f32_st_tnsr(rc2, r_sm_out, (float64)1.0f);
                    v_f32_st_tnsr(rc3, r_sm_out, (float64)1.0f);
                }
            } else {
                // Pass 1: 12 VLIW cycles per 1024 elements (hoisted IRFs -> 0 setup bubbles!)
                bfloat128 m0a = neg_b, m0b = neg_b, m1a = neg_b, m1b = neg_b;
                bfloat128 m2a = neg_b, m2b = neg_b, m3a = neg_b, m3b = neg_b;
                int5 s0a = {0, r0, 0, 0, 0}, s1a = {0, r1, 0, 0, 0}, s2a = {0, r2, 0, 0, 0}, s3a = {0, r3, 0, 0, 0};
                int5 s0b = {128, r0, 0, 0, 0}, s1b = {128, r1, 0, 0, 0}, s2b = {128, r2, 0, 0, 0}, s3b = {128, r3, 0, 0, 0};
                for (int p = 0; p < full_pairs; ++p) {
                    const int i0 = p * 2, i1 = p * 2 + 1;
                    const bfloat128 x0a = v_bf16_ld_tnsr_b(s0a, s_in); s0a[0] += 256;
                    const bfloat128 x1a = v_bf16_ld_tnsr_b(s1a, s_in); s1a[0] += 256;
                    const bfloat128 x2a = v_bf16_ld_tnsr_b(s2a, s_in); s2a[0] += 256;
                    const bfloat128 x3a = v_bf16_ld_tnsr_b(s3a, s_in); s3a[0] += 256;
                    s_vlm0[i0] = x0a; s_vlm1[i0] = x1a; s_vlm2[i0] = x2a; s_vlm3[i0] = x3a;
                    m0a = v_bf16_max_b(m0a, x0a);
                    m1a = v_bf16_max_b(m1a, x1a);
                    m2a = v_bf16_max_b(m2a, x2a);
                    m3a = v_bf16_max_b(m3a, x3a);
                    const bfloat128 x0b = v_bf16_ld_tnsr_b(s0b, s_in); s0b[0] += 256;
                    const bfloat128 x1b = v_bf16_ld_tnsr_b(s1b, s_in); s1b[0] += 256;
                    const bfloat128 x2b = v_bf16_ld_tnsr_b(s2b, s_in); s2b[0] += 256;
                    const bfloat128 x3b = v_bf16_ld_tnsr_b(s3b, s_in); s3b[0] += 256;
                    s_vlm0[i1] = x0b; s_vlm1[i1] = x1b; s_vlm2[i1] = x2b; s_vlm3[i1] = x3b;
                    m0b = v_bf16_max_b(m0b, x0b);
                    m1b = v_bf16_max_b(m1b, x1b);
                    m2b = v_bf16_max_b(m2b, x2b);
                    m3b = v_bf16_max_b(m3b, x3b);
                }
                if (rem256 > 0) {
                    const int base = full_pairs * 256;
                    const ushort128 lid = read_lane_id_2b_b();
                    bfloat128 e0a = neg_b, e0b = neg_b, e1a = neg_b, e1b = neg_b;
                    bfloat128 e2a = neg_b, e2b = neg_b, e3a = neg_b, e3b = neg_b;
                    if (rem256 >= 128) {
                        e0a = v_bf16_ld_tnsr_b((int5){base, r0, 0, 0, 0}, s_in);
                        e1a = v_bf16_ld_tnsr_b((int5){base, r1, 0, 0, 0}, s_in);
                        e2a = v_bf16_ld_tnsr_b((int5){base, r2, 0, 0, 0}, s_in);
                        e3a = v_bf16_ld_tnsr_b((int5){base, r3, 0, 0, 0}, s_in);
                        m0a = v_bf16_max_b(m0a, e0a); m1a = v_bf16_max_b(m1a, e1a);
                        m2a = v_bf16_max_b(m2a, e2a); m3a = v_bf16_max_b(m3a, e3a);
                        const int rr1 = rem256 - 128;
                        if (rr1 > 0) {
                            const unsigned short lim = (unsigned short)rr1;
                            e0b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base + 128, r0, 0, 0, 0}, s_in), neg_b);
                            e1b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base + 128, r1, 0, 0, 0}, s_in), neg_b);
                            e2b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base + 128, r2, 0, 0, 0}, s_in), neg_b);
                            e3b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base + 128, r3, 0, 0, 0}, s_in), neg_b);
                            m0b = v_bf16_max_b(m0b, e0b); m1b = v_bf16_max_b(m1b, e1b);
                            m2b = v_bf16_max_b(m2b, e2b); m3b = v_bf16_max_b(m3b, e3b);
                        }
                    } else {
                        const unsigned short lim = (unsigned short)rem256;
                        e0a = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base, r0, 0, 0, 0}, s_in), neg_b);
                        e1a = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base, r1, 0, 0, 0}, s_in), neg_b);
                        e2a = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base, r2, 0, 0, 0}, s_in), neg_b);
                        e3a = v_bf16_sel_less_u16_b(lid, lim, v_bf16_ld_tnsr_b((int5){base, r3, 0, 0, 0}, s_in), neg_b);
                        m0a = v_bf16_max_b(m0a, e0a); m1a = v_bf16_max_b(m1a, e1a);
                        m2a = v_bf16_max_b(m2a, e2a); m3a = v_bf16_max_b(m3a, e3a);
                    }
                    s_vlm0[30] = e0a; s_vlm0[31] = e0b;
                    s_vlm1[30] = e1a; s_vlm1[31] = e1b;
                    s_vlm2[30] = e2a; s_vlm2[31] = e2b;
                    s_vlm3[30] = e3a; s_vlm3[31] = e3b;
                }

                // 4-row interleaved max64 reduction (0 spills, 0 RAW stalls!)
                const float64_pair_t mf0 = v_convert_bf16_to_f32_all_b(v_bf16_max_b(m0a, m0b));
                const float64_pair_t mf1 = v_convert_bf16_to_f32_all_b(v_bf16_max_b(m1a, m1b));
                const float64_pair_t mf2 = v_convert_bf16_to_f32_all_b(v_bf16_max_b(m2a, m2b));
                const float64_pair_t mf3 = v_convert_bf16_to_f32_all_b(v_bf16_max_b(m3a, m3b));
                float64 mt0 = v_f32_max_b(mf0.v1, mf0.v2);
                float64 mt1 = v_f32_max_b(mf1.v1, mf1.v2);
                float64 mt2 = v_f32_max_b(mf2.v1, mf2.v2);
                float64 mt3 = v_f32_max_b(mf3.v1, mf3.v2);
                MAX64_QUAD(mt0, mt1, mt2, mt3);

                // Compute exact BF16 bias bb <= 118.9057 - mt * kl8_bf (round toward -inf by 0.5 ULP)
                const float64 br0 = (float64)118.9057f - mt0 * kl8_bf;
                const float64 br1 = (float64)118.9057f - mt1 * kl8_bf;
                const float64 br2 = (float64)118.9057f - mt2 * kl8_bf;
                const float64 br3 = (float64)118.9057f - mt3 * kl8_bf;
                const float64 bf0 = br0 - v_f32_abs_b(br0) * (float64)0.004f;
                const float64 bf1 = br1 - v_f32_abs_b(br1) * (float64)0.004f;
                const float64 bf2 = br2 - v_f32_abs_b(br2) * (float64)0.004f;
                const float64 bf3 = br3 - v_f32_abs_b(br3) * (float64)0.004f;
                const bfloat128 bbt0 = v_convert_f32_to_bf16_all_b((float64_pair_t){bf0, bf0});
                const bfloat128 bbt1 = v_convert_f32_to_bf16_all_b((float64_pair_t){bf1, bf1});
                const bfloat128 bbt2 = v_convert_f32_to_bf16_all_b((float64_pair_t){bf2, bf2});
                const bfloat128 bbt3 = v_convert_f32_to_bf16_all_b((float64_pair_t){bf3, bf3});
                float64 bn0 = v_convert_bf16_to_f32_all_b(bbt0).v1;
                float64 bn1 = v_convert_bf16_to_f32_all_b(bbt1).v1;
                float64 bn2 = v_convert_bf16_to_f32_all_b(bbt2).v1;
                float64 bn3 = v_convert_bf16_to_f32_all_b(bbt3).v1;

                float64 rs0 = 0.0f, rs1 = 0.0f, rs2 = 0.0f, rs3 = 0.0f;
                if (!is_first_sm) {
                    const float64 bo0 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc0, m_sm), 0);
                    const float64 bo1 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc1, m_sm), 0);
                    const float64 bo2 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc2, m_sm), 0);
                    const float64 bo3 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc3, m_sm), 0);
                    bn0 = v_f32_min_b(bo0, bn0); bn1 = v_f32_min_b(bo1, bn1);
                    bn2 = v_f32_min_b(bo2, bn2); bn3 = v_f32_min_b(bo3, bn3);
                    const float64 df0 = bn0 - bo0, df1 = bn1 - bo1, df2 = bn2 - bo2, df3 = bn3 - bo3;
                    RESCALE_QUAD_BIAS(rs0, rs1, rs2, rs3, df0, df1, df2, df3);
                }
                v_f32_st_tnsr(rc0, m_sm, bn0); v_f32_st_tnsr(rc1, m_sm, bn1);
                v_f32_st_tnsr(rc2, m_sm, bn2); v_f32_st_tnsr(rc3, m_sm, bn3);
                v_f32_st_tnsr(rc0, r_sm_out, rs0); v_f32_st_tnsr(rc1, r_sm_out, rs1);
                v_f32_st_tnsr(rc2, r_sm_out, rs2); v_f32_st_tnsr(rc3, r_sm_out, rs3);

                const bfloat128 bb0 = v_convert_f32_to_bf16_all_b((float64_pair_t){bn0, bn0});
                const bfloat128 bb1 = v_convert_f32_to_bf16_all_b((float64_pair_t){bn1, bn1});
                const bfloat128 bb2 = v_convert_f32_to_bf16_all_b((float64_pair_t){bn2, bn2});
                const bfloat128 bb3 = v_convert_f32_to_bf16_all_b((float64_pair_t){bn3, bn3});

                // Pass 2: 2x-UNROLLED (512 cols/iter) ONLY 12 VPU ops per 1024 elements!
                // Computing all pu0..pu3 before st_tnsr + 2x unrolling eliminates V3 serialization & NOP bubbles!
                int5 p0c = {0, r0, 0, 0, 0}, p1c = {0, r1, 0, 0, 0}, p2c = {0, r2, 0, 0, 0}, p3c = {0, r3, 0, 0, 0};
                int p = 0;
                for (; p + 2 <= full_pairs; p += 2) {
                    const int i0 = p * 2, i1 = p * 2 + 1, i2 = p * 2 + 2, i3 = p * 2 + 3;
                    const bfloat128 f0a = v_bf16_mac_b(s_vlm0[i0], kl8_b, bb0);
                    const bfloat128 f0b = v_bf16_mac_b(s_vlm0[i1], kl8_b, bb0);
                    const bfloat128 f1a = v_bf16_mac_b(s_vlm1[i0], kl8_b, bb1);
                    const bfloat128 f1b = v_bf16_mac_b(s_vlm1[i1], kl8_b, bb1);
                    const bfloat128 f2a = v_bf16_mac_b(s_vlm2[i0], kl8_b, bb2);
                    const bfloat128 f2b = v_bf16_mac_b(s_vlm2[i1], kl8_b, bb2);
                    const bfloat128 f3a = v_bf16_mac_b(s_vlm3[i0], kl8_b, bb3);
                    const bfloat128 f3b = v_bf16_mac_b(s_vlm3[i1], kl8_b, bb3);

                    const uchar256 pu0 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f0a, f0b}, SW_RHNE);
                    const uchar256 pu1 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f1a, f1b}, SW_RHNE);
                    const uchar256 pu2 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f2a, f2b}, SW_RHNE);
                    const uchar256 pu3 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f3a, f3b}, SW_RHNE);

                    const bfloat128 g0a = v_bf16_mac_b(s_vlm0[i2], kl8_b, bb0);
                    const bfloat128 g0b = v_bf16_mac_b(s_vlm0[i3], kl8_b, bb0);
                    v_f8_st_tnsr(p0c, p_out, *(const minifloat256*)&pu0); p0c[0] += 256;
                    const bfloat128 g1a = v_bf16_mac_b(s_vlm1[i2], kl8_b, bb1);
                    const bfloat128 g1b = v_bf16_mac_b(s_vlm1[i3], kl8_b, bb1);
                    v_f8_st_tnsr(p1c, p_out, *(const minifloat256*)&pu1); p1c[0] += 256;
                    const bfloat128 g2a = v_bf16_mac_b(s_vlm2[i2], kl8_b, bb2);
                    const bfloat128 g2b = v_bf16_mac_b(s_vlm2[i3], kl8_b, bb2);
                    v_f8_st_tnsr(p2c, p_out, *(const minifloat256*)&pu2); p2c[0] += 256;
                    const bfloat128 g3a = v_bf16_mac_b(s_vlm3[i2], kl8_b, bb3);
                    const bfloat128 g3b = v_bf16_mac_b(s_vlm3[i3], kl8_b, bb3);
                    v_f8_st_tnsr(p3c, p_out, *(const minifloat256*)&pu3); p3c[0] += 256;

                    const uchar256 qu0 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){g0a, g0b}, SW_RHNE);
                    const uchar256 qu1 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){g1a, g1b}, SW_RHNE);
                    const uchar256 qu2 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){g2a, g2b}, SW_RHNE);
                    const uchar256 qu3 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){g3a, g3b}, SW_RHNE);

                    v_f8_st_tnsr(p0c, p_out, *(const minifloat256*)&qu0); p0c[0] += 256;
                    v_f8_st_tnsr(p1c, p_out, *(const minifloat256*)&qu1); p1c[0] += 256;
                    v_f8_st_tnsr(p2c, p_out, *(const minifloat256*)&qu2); p2c[0] += 256;
                    v_f8_st_tnsr(p3c, p_out, *(const minifloat256*)&qu3); p3c[0] += 256;
                }
                if (p < full_pairs) {
                    const int i0 = p * 2, i1 = p * 2 + 1;
                    const bfloat128 f0a = v_bf16_mac_b(s_vlm0[i0], kl8_b, bb0);
                    const bfloat128 f1a = v_bf16_mac_b(s_vlm1[i0], kl8_b, bb1);
                    const bfloat128 f2a = v_bf16_mac_b(s_vlm2[i0], kl8_b, bb2);
                    const bfloat128 f3a = v_bf16_mac_b(s_vlm3[i0], kl8_b, bb3);
                    const bfloat128 f0b = v_bf16_mac_b(s_vlm0[i1], kl8_b, bb0);
                    const bfloat128 f1b = v_bf16_mac_b(s_vlm1[i1], kl8_b, bb1);
                    const bfloat128 f2b = v_bf16_mac_b(s_vlm2[i1], kl8_b, bb2);
                    const bfloat128 f3b = v_bf16_mac_b(s_vlm3[i1], kl8_b, bb3);
                    const uchar256 pu0 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f0a, f0b}, SW_RHNE);
                    const uchar256 pu1 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f1a, f1b}, SW_RHNE);
                    const uchar256 pu2 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f2a, f2b}, SW_RHNE);
                    const uchar256 pu3 = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){f3a, f3b}, SW_RHNE);
                    v_f8_st_tnsr(p0c, p_out, *(const minifloat256*)&pu0);
                    v_f8_st_tnsr(p1c, p_out, *(const minifloat256*)&pu1);
                    v_f8_st_tnsr(p2c, p_out, *(const minifloat256*)&pu2);
                    v_f8_st_tnsr(p3c, p_out, *(const minifloat256*)&pu3);
                }

                int next_col = full_pairs * 256;
                if (rem256 > 0) {
                    bfloat128 u0a = v_bf16_max_b(v_bf16_mac_b(s_vlm0[30], kl8_b, bb0), zero_b);
                    bfloat128 u1a = v_bf16_max_b(v_bf16_mac_b(s_vlm1[30], kl8_b, bb1), zero_b);
                    bfloat128 u2a = v_bf16_max_b(v_bf16_mac_b(s_vlm2[30], kl8_b, bb2), zero_b);
                    bfloat128 u3a = v_bf16_max_b(v_bf16_mac_b(s_vlm3[30], kl8_b, bb3), zero_b);
                    bfloat128 u0b = zero_b, u1b = zero_b, u2b = zero_b, u3b = zero_b;
                    const ushort128 lid = read_lane_id_2b_b();
                    if (rem256 < 128) {
                        const unsigned short lim = (unsigned short)rem256;
                        u0a = v_bf16_sel_less_u16_b(lid, lim, u0a, zero_b);
                        u1a = v_bf16_sel_less_u16_b(lid, lim, u1a, zero_b);
                        u2a = v_bf16_sel_less_u16_b(lid, lim, u2a, zero_b);
                        u3a = v_bf16_sel_less_u16_b(lid, lim, u3a, zero_b);
                    } else if (rem256 > 128) {
                        const unsigned short lim = (unsigned short)(rem256 - 128);
                        u0b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_max_b(v_bf16_mac_b(s_vlm0[31], kl8_b, bb0), zero_b), zero_b);
                        u1b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_max_b(v_bf16_mac_b(s_vlm1[31], kl8_b, bb1), zero_b), zero_b);
                        u2b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_max_b(v_bf16_mac_b(s_vlm2[31], kl8_b, bb2), zero_b), zero_b);
                        u3b = v_bf16_sel_less_u16_b(lid, lim, v_bf16_max_b(v_bf16_mac_b(s_vlm3[31], kl8_b, bb3), zero_b), zero_b);
                    }
                    const bfloat128_pair_t up0 = {u0a, u0b}, up1 = {u1a, u1b}, up2 = {u2a, u2b}, up3 = {u3a, u3b};
                    const uchar256 pu0 = v_convert_bf16_to_u8_all_b(up0, SW_RHNE);
                    const uchar256 pu1 = v_convert_bf16_to_u8_all_b(up1, SW_RHNE);
                    const uchar256 pu2 = v_convert_bf16_to_u8_all_b(up2, SW_RHNE);
                    const uchar256 pu3 = v_convert_bf16_to_u8_all_b(up3, SW_RHNE);
                    v_f8_st_tnsr((int5){next_col, r0, 0, 0, 0}, p_out, *(const minifloat256*)&pu0);
                    v_f8_st_tnsr((int5){next_col, r1, 0, 0, 0}, p_out, *(const minifloat256*)&pu1);
                    v_f8_st_tnsr((int5){next_col, r2, 0, 0, 0}, p_out, *(const minifloat256*)&pu2);
                    v_f8_st_tnsr((int5){next_col, r3, 0, 0, 0}, p_out, *(const minifloat256*)&pu3);
                    next_col += 256;
                }
                for (int col = next_col; col < tile_k; col += 256) {
                    v_f8_st_tnsr((int5){col, r0, 0, 0, 0}, p_out, zero_f8);
                    v_f8_st_tnsr((int5){col, r1, 0, 0, 0}, p_out, zero_f8);
                    v_f8_st_tnsr((int5){col, r2, 0, 0, 0}, p_out, zero_f8);
                    v_f8_st_tnsr((int5){col, r3, 0, 0, 0}, p_out, zero_f8);
                }
            }
        }

        // ---- Part B: 4-row interleaved ACC(k-2) (O_acc + L_acc from MME PSUM!) [+ Finalize to HBM] ----
        if (do_acc) {
            int5 d0a = {0, r0, 0, 0, 0}, d0b = {64, r0, 0, 0, 0};
            int5 d1a = {0, r1, 0, 0, 0}, d1b = {64, r1, 0, 0, 0};
            int5 d2a = {0, r2, 0, 0, 0}, d2b = {64, r2, 0, 0, 0};
            int5 d3a = {0, r3, 0, 0, 0}, d3b = {64, r3, 0, 0, 0};
            float64 o0a = v_f32_ld_tnsr_b(d0a, pv_in), o0b = v_f32_ld_tnsr_b(d0b, pv_in);
            float64 o1a = v_f32_ld_tnsr_b(d1a, pv_in), o1b = v_f32_ld_tnsr_b(d1b, pv_in);
            float64 o2a = v_f32_ld_tnsr_b(d2a, pv_in), o2b = v_f32_ld_tnsr_b(d2b, pv_in);
            float64 o3a = v_f32_ld_tnsr_b(d3a, pv_in), o3b = v_f32_ld_tnsr_b(d3b, pv_in);
            float64 l0, l1, l2, l3;
            if (pv_w >= 256) {
                l0 = v_broadcast_element_f32(v_f32_ld_tnsr_b((int5){128, r0, 0, 0, 0}, pv_in), 0);
                l1 = v_broadcast_element_f32(v_f32_ld_tnsr_b((int5){128, r1, 0, 0, 0}, pv_in), 0);
                l2 = v_broadcast_element_f32(v_f32_ld_tnsr_b((int5){128, r2, 0, 0, 0}, pv_in), 0);
                l3 = v_broadcast_element_f32(v_f32_ld_tnsr_b((int5){128, r3, 0, 0, 0}, pv_in), 0);
            } else {
                l0 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc0, l_tile_in), 0);
                l1 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc1, l_tile_in), 0);
                l2 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc2, l_tile_in), 0);
                l3 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc3, l_tile_in), 0);
            }
            if (!is_first_acc) {
                const float64 rr0 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc0, r_acc_in), 0);
                const float64 rr1 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc1, r_acc_in), 0);
                const float64 rr2 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc2, r_acc_in), 0);
                const float64 rr3 = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc3, r_acc_in), 0);
                o0a = v_f32_mac_b(v_f32_ld_tnsr_b(d0a, o_acc), rr0, o0a);
                o1a = v_f32_mac_b(v_f32_ld_tnsr_b(d1a, o_acc), rr1, o1a);
                o2a = v_f32_mac_b(v_f32_ld_tnsr_b(d2a, o_acc), rr2, o2a);
                o3a = v_f32_mac_b(v_f32_ld_tnsr_b(d3a, o_acc), rr3, o3a);
                o0b = v_f32_mac_b(v_f32_ld_tnsr_b(d0b, o_acc), rr0, o0b);
                o1b = v_f32_mac_b(v_f32_ld_tnsr_b(d1b, o_acc), rr1, o1b);
                o2b = v_f32_mac_b(v_f32_ld_tnsr_b(d2b, o_acc), rr2, o2b);
                o3b = v_f32_mac_b(v_f32_ld_tnsr_b(d3b, o_acc), rr3, o3b);
                l0 = v_f32_mac_b(v_broadcast_element_f32(v_f32_ld_tnsr_b(rc0, l_acc), 0), rr0, l0);
                l1 = v_f32_mac_b(v_broadcast_element_f32(v_f32_ld_tnsr_b(rc1, l_acc), 0), rr1, l1);
                l2 = v_f32_mac_b(v_broadcast_element_f32(v_f32_ld_tnsr_b(rc2, l_acc), 0), rr2, l2);
                l3 = v_f32_mac_b(v_broadcast_element_f32(v_f32_ld_tnsr_b(rc3, l_acc), 0), rr3, l3);
            }
            if (is_last_acc) {
                if (head_idx >= 0) {
                    const float64 lv0 = v_f32_max_b(l0, 1.0e-30f);
                    const float64 lv1 = v_f32_max_b(l1, 1.0e-30f);
                    const float64 lv2 = v_f32_max_b(l2, 1.0e-30f);
                    const float64 lv3 = v_f32_max_b(l3, 1.0e-30f);
                    float64 e0 = as_float64((int64)0x7ef311c3 - as_int64(lv0));
                    float64 e1 = as_float64((int64)0x7ef311c3 - as_int64(lv1));
                    float64 e2 = as_float64((int64)0x7ef311c3 - as_int64(lv2));
                    float64 e3 = as_float64((int64)0x7ef311c3 - as_int64(lv3));
                    e0 = e0 * (2.0f - lv0 * e0); e1 = e1 * (2.0f - lv1 * e1);
                    e2 = e2 * (2.0f - lv2 * e2); e3 = e3 * (2.0f - lv3 * e3);
                    e0 = e0 * (2.0f - lv0 * e0); e1 = e1 * (2.0f - lv1 * e1);
                    e2 = e2 * (2.0f - lv2 * e2); e3 = e3 * (2.0f - lv3 * e3);
                    e0 = e0 * (2.0f - lv0 * e0); e1 = e1 * (2.0f - lv1 * e1);
                    e2 = e2 * (2.0f - lv2 * e2); e3 = e3 * (2.0f - lv3 * e3);
                    const float64 sc0 = (float64)dv_f * e0, sc1 = (float64)dv_f * e1;
                    const float64 sc2 = (float64)dv_f * e2, sc3 = (float64)dv_f * e3;
                    o0a *= sc0; o1a *= sc1; o2a *= sc2; o3a *= sc3;
                    o0b *= sc0; o1b *= sc1; o2b *= sc2; o3b *= sc3;
                    bfloat128 ob0, ob1, ob2, ob3;
                    PACK_F32_TO_BF16_PAIR(ob0, ob1, o0a, o0b, o1a, o1b);
                    PACK_F32_TO_BF16_PAIR(ob2, ob3, o2a, o2b, o3a, o3b);
                    // Store directly into [128, NH, S] == row-major [S, NH*128] (eliminates 550MB transpose in r_dit_post!)
                    if (q_base + r3 < valid_k) {
                        v_bf16_st_tnsr((int5){0, head_idx, q_base + r0, 0, 0}, att_out, ob0);
                        v_bf16_st_tnsr((int5){0, head_idx, q_base + r1, 0, 0}, att_out, ob1);
                        v_bf16_st_tnsr((int5){0, head_idx, q_base + r2, 0, 0}, att_out, ob2);
                        v_bf16_st_tnsr((int5){0, head_idx, q_base + r3, 0, 0}, att_out, ob3);
                    } else {
                        if (q_base + r0 < valid_k) v_bf16_st_tnsr((int5){0, head_idx, q_base + r0, 0, 0}, att_out, ob0);
                        if (q_base + r1 < valid_k) v_bf16_st_tnsr((int5){0, head_idx, q_base + r1, 0, 0}, att_out, ob1);
                        if (q_base + r2 < valid_k) v_bf16_st_tnsr((int5){0, head_idx, q_base + r2, 0, 0}, att_out, ob2);
                        if (q_base + r3 < valid_k) v_bf16_st_tnsr((int5){0, head_idx, q_base + r3, 0, 0}, att_out, ob3);
                    }
                }
            } else {
                v_f32_st_tnsr(d0a, o_acc, o0a); v_f32_st_tnsr(d0b, o_acc, o0b);
                v_f32_st_tnsr(d1a, o_acc, o1a); v_f32_st_tnsr(d1b, o_acc, o1b);
                v_f32_st_tnsr(d2a, o_acc, o2a); v_f32_st_tnsr(d2b, o_acc, o2b);
                v_f32_st_tnsr(d3a, o_acc, o3a); v_f32_st_tnsr(d3b, o_acc, o3b);
                v_f32_st_tnsr(rc0, l_acc, l0);  v_f32_st_tnsr(rc1, l_acc, l1);
                v_f32_st_tnsr(rc2, l_acc, l2);  v_f32_st_tnsr(rc3, l_acc, l3);
            }
        }
    }

    // =========================================================================
    // TAIL PATH: remaining 0..3 rows (only used when BQ % 96 != 0, e.g. BQ=256)
    // =========================================================================
    for (; row < row_end; ++row) {
        int5 rc = {0, row, 0, 0, 0};
        if (do_sm) {
            if (visible == 0) {
                for (int col = 0; col < tile_k; col += 256)
                    v_f8_st_tnsr((int5){col, row, 0, 0, 0}, p_out, zero_f8);
                if (is_first_sm) {
                    v_f32_st_tnsr(rc, m_sm, (float64)(1.0e30f));
                    v_f32_st_tnsr(rc, r_sm_out, (float64)0.0f);
                } else {
                    v_f32_st_tnsr(rc, r_sm_out, (float64)1.0f);
                }
            } else {
                bfloat128 m0 = neg_b, m1 = neg_b;
                for (int p = 0; p < full_pairs; ++p) {
                    const int base = p * 256;
                    const bfloat128 x0 = v_bf16_ld_tnsr_b((int5){base,       row, 0, 0, 0}, s_in);
                    const bfloat128 x1 = v_bf16_ld_tnsr_b((int5){base + 128, row, 0, 0, 0}, s_in);
                    s_vlm0[p * 2 + 0] = x0;
                    s_vlm0[p * 2 + 1] = x1;
                    m0 = v_bf16_max_b(m0, x0);
                    m1 = v_bf16_max_b(m1, x1);
                }
                if (rem256 > 0) {
                    const int base = full_pairs * 256;
                    bfloat128 edge0 = neg_b, edge1 = neg_b;
                    if (rem256 >= 128) {
                        edge0 = v_bf16_ld_tnsr_b((int5){base, row, 0, 0, 0}, s_in);
                        m0 = v_bf16_max_b(m0, edge0);
                        const int r1 = rem256 - 128;
                        if (r1 > 0) {
                            edge1 = v_bf16_sel_less_u16_b(read_lane_id_2b_b(), (unsigned short)r1,
                                                          v_bf16_ld_tnsr_b((int5){base + 128, row, 0, 0, 0}, s_in), neg_b);
                            m1 = v_bf16_max_b(m1, edge1);
                        }
                    } else {
                        edge0 = v_bf16_sel_less_u16_b(read_lane_id_2b_b(), (unsigned short)rem256,
                                                      v_bf16_ld_tnsr_b((int5){base, row, 0, 0, 0}, s_in), neg_b);
                        m0 = v_bf16_max_b(m0, edge0);
                    }
                    s_vlm0[30] = edge0;
                    s_vlm0[31] = edge1;
                }
                const float64_pair_t mf = v_convert_bf16_to_f32_all_b(v_bf16_max_b(m0, m1));
                const float64 m_tile = max64_masks(v_f32_max_b(mf.v1, mf.v2), shuf4, shuf2, shuf1);
                const float64 br = (float64)118.9057f - m_tile * kl8_bf;
                const float64 bf = br - v_f32_abs_b(br) * (float64)0.004f;
                const bfloat128 bbt = v_convert_f32_to_bf16_all_b((float64_pair_t){bf, bf});
                float64 b_new = v_convert_bf16_to_f32_all_b(bbt).v1;
                float64 rescale = 0.0f;
                if (!is_first_sm) {
                    const float64 b_old = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc, m_sm), 0);
                    b_new = v_f32_min_b(b_old, b_new);
                    const float64 diff = b_new - b_old;
                    const float64 t_sc = diff * 0.125f;
                    const float64 t_cl = v_f32_max_b(t_sc, -120.0f);
                    rescale = v_f32_sel_eq_f32_b(diff, (float64)0.0f, (float64)1.0f,
                                v_f32_sel_less_f32_b(t_sc, -120.0f, (float64)0.0f,
                                                     exp_biased(t_cl, 1.0f, (float64)127.0f)));
                }
                v_f32_st_tnsr(rc, m_sm, b_new);
                v_f32_st_tnsr(rc, r_sm_out, rescale);
                const bfloat128 bb = v_convert_f32_to_bf16_all_b((float64_pair_t){b_new, b_new});

                for (int p = 0; p < full_pairs; ++p) {
                    const bfloat128 u0 = v_bf16_mac_b(s_vlm0[p * 2 + 0], kl8_b, bb);
                    const bfloat128 u1 = v_bf16_mac_b(s_vlm0[p * 2 + 1], kl8_b, bb);
                    const uchar256 pu = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){u0, u1}, SW_RHNE);
                    v_f8_st_tnsr((int5){p * 256, row, 0, 0, 0}, p_out, *(const minifloat256*)&pu);
                }
                int next_col = full_pairs * 256;
                if (rem256 > 0) {
                    bfloat128 u0 = v_bf16_max_b(v_bf16_mac_b(s_vlm0[30], kl8_b, bb), zero_b);
                    bfloat128 u1 = zero_b;
                    if (rem256 < 128) {
                        u0 = v_bf16_sel_less_u16_b(read_lane_id_2b_b(), (unsigned short)rem256, u0, zero_b);
                    } else if (rem256 > 128) {
                        u1 = v_bf16_max_b(v_bf16_mac_b(s_vlm0[31], kl8_b, bb), zero_b);
                        u1 = v_bf16_sel_less_u16_b(read_lane_id_2b_b(), (unsigned short)(rem256 - 128), u1, zero_b);
                    }
                    const uchar256 pu = v_convert_bf16_to_u8_all_b((bfloat128_pair_t){u0, u1}, SW_RHNE);
                    v_f8_st_tnsr((int5){next_col, row, 0, 0, 0}, p_out, *(const minifloat256*)&pu);
                    next_col += 256;
                }
                for (int col = next_col; col < tile_k; col += 256)
                    v_f8_st_tnsr((int5){col, row, 0, 0, 0}, p_out, zero_f8);
            }
        }
        if (do_acc) {
            int5 d0 = {0, row, 0, 0, 0}, d1 = {64, row, 0, 0, 0};
            float64 o0 = v_f32_ld_tnsr_b(d0, pv_in), o1 = v_f32_ld_tnsr_b(d1, pv_in);
            float64 lv = (pv_w >= 256)
                ? v_broadcast_element_f32(v_f32_ld_tnsr_b((int5){128, row, 0, 0, 0}, pv_in), 0)
                : v_broadcast_element_f32(v_f32_ld_tnsr_b(rc, l_tile_in), 0);
            if (!is_first_acc) {
                const float64 r = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc, r_acc_in), 0);
                o0 = v_f32_mac_b(v_f32_ld_tnsr_b(d0, o_acc), r, o0);
                o1 = v_f32_mac_b(v_f32_ld_tnsr_b(d1, o_acc), r, o1);
                lv = v_f32_mac_b(v_broadcast_element_f32(v_f32_ld_tnsr_b(rc, l_acc), 0), r, lv);
            }
            if (is_last_acc) {
                if (head_idx >= 0) {
                    const float64 l_safe = v_f32_max_b(lv, 1.0e-30f);
                    const float64 scale = (float64)dv_f * rcp3_f32(l_safe);
                    o0 *= scale; o1 *= scale;
                    if (q_base + row < valid_k) {
                        v_bf16_st_tnsr((int5){0, head_idx, q_base + row, 0, 0}, att_out, f32_pair_to_bf16_linear_fast(o0, o1));
                    }
                }
            } else {
                v_f32_st_tnsr(d0, o_acc, o0);
                v_f32_st_tnsr(d1, o_acc, o1);
                v_f32_st_tnsr(rc, l_acc, lv);
            }
        }
    }
}
