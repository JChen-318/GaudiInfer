// SPDX-License-Identifier: Apache-2.0
// Inkling-2 full-layer in-place 48 MiB SRAM FlashAttention fused tile step (SM(k) + ACC(k-2)) for Direct prefill:
//   - SM(k) (when do_sm != 0): processes 4 rows in lockstep using VLM sbuf[4][32] (32 KiB <= 64 KiB VLM, TW <= 4096):
//       1) Pass 1 (8-chunk HW loop, interleaved max): reads sf f16 [TW, GT] from SRAM ONCE into sbuf[u][c] + tile max.
//       2) 4-row lockstep max64_4 + m_old + mh (eliminates 6-cycle VPU reduction NOP stalls!).
//       3) Pass 2 (6-chunk PROB6 HW loop: fused mac(x, kh, -mh*kh), minimax quadratic BF16 exp (<=0.71 ULP), dual store
//          coords, zero spills!): writes BF16 probs pf [TW, GT] IN-PLACE (pf aliases sf!) + accumulates tile sums.
//       4) 4-row lockstep sum64_4 + vectorized 64-row exp_vec (only 2 exp_vec/core, ZERO st_tnsr_partial RMWs!).
//   - ACC(k-2) (when do_acc != 0): rescales SRAM o_acc[:, r] by SRAM r_acc[r] and adds SRAM of_in[:, r] (2-row unrolled).
//   100% LUT-free, ZERO HBM traffic in the entire kernel! Index space [GT] (3072 -> 128 rows/core = 2 x 64-row blocks).
// Tensors (dim 0 fastest, ALL in SRAM):
//   0 sf    f16  [TW, GT]   (TW in {1024, 2048, 3072, 4096})
//   1 pf    bf16 [TW, GT]   (may alias sf in-place!)
//   2 m_buf f32  [GT]       (current KV head running row max)
//   3 l_buf f32  [GT]       (current KV head running row sum)
//   4 r_sm  f32  [GT]       (rescale output for SM(k))
//   5 of_in f32  [128, GT]  (partial FPV(k-2) output)
//   6 o_acc f32  [128, GT]  (current KV head running F32 output)
//   7 r_acc f32  [GT]       (rescale input for ACC(k-2))
// Params: float scale, int nv, int do_sm, int do_acc.
#include "ink_math_gaudi2.h"

__local__ half128 sbuf[4][32];

static inline float64 exp_vec(float64 x) {
    const float64 xc = v_f32_max_b(x, -87.0f);
    const float64 t = xc * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.0558755f;
    p = v_f32_mac_b(p, f, 0.2401397f);
    p = v_f32_mac_b(p, f, 0.6931705f);
    p = v_f32_mac_b(p, f, 1.0f);
    const int64 bits = (n + 127) << 23;
    return p * as_float64(bits);
}

#define MAX64_4(x0, x1, x2, x3, c4, c2, c1) do {                                                              \
        const int all_ = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |  \
                         SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;   \
        const int grp_ = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |                \
                         SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;                                               \
        float64 t0_, t1_, t2_, t3_;                                                                           \
        t0_ = v_f32_mov_dual_group_all_b((x0), 0xffffffff, 1, 0, 3, 2, all_, (x0));                           \
        t1_ = v_f32_mov_dual_group_all_b((x1), 0xffffffff, 1, 0, 3, 2, all_, (x1));                           \
        t2_ = v_f32_mov_dual_group_all_b((x2), 0xffffffff, 1, 0, 3, 2, all_, (x2));                           \
        t3_ = v_f32_mov_dual_group_all_b((x3), 0xffffffff, 1, 0, 3, 2, all_, (x3));                           \
        (x0) = v_f32_max_b((x0), t0_); (x1) = v_f32_max_b((x1), t1_);                                         \
        (x2) = v_f32_max_b((x2), t2_); (x3) = v_f32_max_b((x3), t3_);                                         \
        t0_ = v_f32_mov_dual_group_all_b((x0), 0xffffffff, 2, 3, 0, 1, all_, (x0));                           \
        t1_ = v_f32_mov_dual_group_all_b((x1), 0xffffffff, 2, 3, 0, 1, all_, (x1));                           \
        t2_ = v_f32_mov_dual_group_all_b((x2), 0xffffffff, 2, 3, 0, 1, all_, (x2));                           \
        t3_ = v_f32_mov_dual_group_all_b((x3), 0xffffffff, 2, 3, 0, 1, all_, (x3));                           \
        (x0) = v_f32_max_b((x0), t0_); (x1) = v_f32_max_b((x1), t1_);                                         \
        (x2) = v_f32_max_b((x2), t2_); (x3) = v_f32_max_b((x3), t3_);                                         \
        t0_ = v_f32_mov_group_b((x0), 0xffffffff, grp_, (x0));                                                \
        t1_ = v_f32_mov_group_b((x1), 0xffffffff, grp_, (x1));                                                \
        t2_ = v_f32_mov_group_b((x2), 0xffffffff, grp_, (x2));                                                \
        t3_ = v_f32_mov_group_b((x3), 0xffffffff, grp_, (x3));                                                \
        (x0) = v_f32_max_b((x0), t0_); (x1) = v_f32_max_b((x1), t1_);                                         \
        (x2) = v_f32_max_b((x2), t2_); (x3) = v_f32_max_b((x3), t3_);                                         \
        t0_ = v_f32_shuffle_b((x0), (c4), 0, (x0)); t1_ = v_f32_shuffle_b((x1), (c4), 0, (x1));               \
        t2_ = v_f32_shuffle_b((x2), (c4), 0, (x2)); t3_ = v_f32_shuffle_b((x3), (c4), 0, (x3));               \
        (x0) = v_f32_max_b((x0), t0_); (x1) = v_f32_max_b((x1), t1_);                                         \
        (x2) = v_f32_max_b((x2), t2_); (x3) = v_f32_max_b((x3), t3_);                                         \
        t0_ = v_f32_shuffle_b((x0), (c2), 0, (x0)); t1_ = v_f32_shuffle_b((x1), (c2), 0, (x1));               \
        t2_ = v_f32_shuffle_b((x2), (c2), 0, (x2)); t3_ = v_f32_shuffle_b((x3), (c2), 0, (x3));               \
        (x0) = v_f32_max_b((x0), t0_); (x1) = v_f32_max_b((x1), t1_);                                         \
        (x2) = v_f32_max_b((x2), t2_); (x3) = v_f32_max_b((x3), t3_);                                         \
        t0_ = v_f32_shuffle_b((x0), (c1), 0, (x0)); t1_ = v_f32_shuffle_b((x1), (c1), 0, (x1));               \
        t2_ = v_f32_shuffle_b((x2), (c1), 0, (x2)); t3_ = v_f32_shuffle_b((x3), (c1), 0, (x3));               \
        (x0) = v_f32_max_b((x0), t0_); (x1) = v_f32_max_b((x1), t1_);                                         \
        (x2) = v_f32_max_b((x2), t2_); (x3) = v_f32_max_b((x3), t3_);                                         \
    } while (0)

#define SUM64_4(x0, x1, x2, x3, c4, c2, c1) do {                                                              \
        const int all_ = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |  \
                         SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;   \
        const int grp_ = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |                \
                         SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;                                               \
        float64 t0_, t1_, t2_, t3_;                                                                           \
        t0_ = v_f32_mov_dual_group_all_b((x0), 0xffffffff, 1, 0, 3, 2, all_, (x0));                           \
        t1_ = v_f32_mov_dual_group_all_b((x1), 0xffffffff, 1, 0, 3, 2, all_, (x1));                           \
        t2_ = v_f32_mov_dual_group_all_b((x2), 0xffffffff, 1, 0, 3, 2, all_, (x2));                           \
        t3_ = v_f32_mov_dual_group_all_b((x3), 0xffffffff, 1, 0, 3, 2, all_, (x3));                           \
        (x0) += t0_; (x1) += t1_; (x2) += t2_; (x3) += t3_;                                                   \
        t0_ = v_f32_mov_dual_group_all_b((x0), 0xffffffff, 2, 3, 0, 1, all_, (x0));                           \
        t1_ = v_f32_mov_dual_group_all_b((x1), 0xffffffff, 2, 3, 0, 1, all_, (x1));                           \
        t2_ = v_f32_mov_dual_group_all_b((x2), 0xffffffff, 2, 3, 0, 1, all_, (x2));                           \
        t3_ = v_f32_mov_dual_group_all_b((x3), 0xffffffff, 2, 3, 0, 1, all_, (x3));                           \
        (x0) += t0_; (x1) += t1_; (x2) += t2_; (x3) += t3_;                                                   \
        t0_ = v_f32_mov_group_b((x0), 0xffffffff, grp_, (x0));                                                \
        t1_ = v_f32_mov_group_b((x1), 0xffffffff, grp_, (x1));                                                \
        t2_ = v_f32_mov_group_b((x2), 0xffffffff, grp_, (x2));                                                \
        t3_ = v_f32_mov_group_b((x3), 0xffffffff, grp_, (x3));                                                \
        (x0) += t0_; (x1) += t1_; (x2) += t2_; (x3) += t3_;                                                   \
        t0_ = v_f32_shuffle_b((x0), (c4), 0, (x0)); t1_ = v_f32_shuffle_b((x1), (c4), 0, (x1));               \
        t2_ = v_f32_shuffle_b((x2), (c4), 0, (x2)); t3_ = v_f32_shuffle_b((x3), (c4), 0, (x3));               \
        (x0) += t0_; (x1) += t1_; (x2) += t2_; (x3) += t3_;                                                   \
        t0_ = v_f32_shuffle_b((x0), (c2), 0, (x0)); t1_ = v_f32_shuffle_b((x1), (c2), 0, (x1));               \
        t2_ = v_f32_shuffle_b((x2), (c2), 0, (x2)); t3_ = v_f32_shuffle_b((x3), (c2), 0, (x3));               \
        (x0) += t0_; (x1) += t1_; (x2) += t2_; (x3) += t3_;                                                   \
        t0_ = v_f32_shuffle_b((x0), (c1), 0, (x0)); t1_ = v_f32_shuffle_b((x1), (c1), 0, (x1));               \
        t2_ = v_f32_shuffle_b((x2), (c1), 0, (x2)); t3_ = v_f32_shuffle_b((x3), (c1), 0, (x3));               \
        (x0) += t0_; (x1) += t1_; (x2) += t2_; (x3) += t3_;                                                   \
    } while (0)

#define ACC(acc, y) ((acc) = v_bf16_mac_acc32_b((y), (bfloat128)1.0f, (acc), 0, 1, 0))

// Optimal FP16 minimax quadratic for 128*(2^(k/128) - 1 - k/128) on k in [0..127]:
//   max error 0.71 BF16 ULP (108/128 exact bits, max diff 1 BF16 LSB -> KL 0.00000), saves 1 MAC/chunk vs cubic!
// Fused mac(x, kh, neg_mh_kh) replaces sub(x, mh) + mul(..., kh) (safe for |x| <= 4000, Inkling |x| <= 200).
#define PROB(x, out) do {                                                                     \
        short128 q_ = v_convert_f16_to_i16_b(v_f16_mac_b((x), kh, nmhkh), SW_RHNE);           \
        q_ = v_i16_max_b(q_, (short128)-16128);                                               \
        const half128 k_ = v_convert_i16_to_f16_b(q_ & (short128)127, SW_RHNE);               \
        half128 c_ = (half128)0.002681732178f;                                                \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.3508300781f);                                    \
        c_ = v_f16_mac_b(c_, k_, (half128)0.4558105469f);                                     \
        short128 b_ = q_ + v_convert_f16_to_i16_b(c_, SW_RHNE) + (short128)16256;             \
        (out) = *(bfloat128*)&b_;                                                             \
    } while (0)

// 6-way lockstep PROB6 matching Gaudi2's 6-cycle F16/MAC latency: 11 VPU ops/chunk, ZERO NOPs, ZERO VRF spills!
#define PROB6(x0, x1, x2, x3, x4, x5, o0, o1, o2, o3, o4, o5) do {                                   \
        const half128 p0_ = v_f16_mac_b((x0), kh, nmhkh), p1_ = v_f16_mac_b((x1), kh, nmhkh);        \
        const half128 p2_ = v_f16_mac_b((x2), kh, nmhkh), p3_ = v_f16_mac_b((x3), kh, nmhkh);        \
        const half128 p4_ = v_f16_mac_b((x4), kh, nmhkh), p5_ = v_f16_mac_b((x5), kh, nmhkh);        \
        short128 q0_ = v_convert_f16_to_i16_b(p0_, SW_RHNE);                                         \
        short128 q1_ = v_convert_f16_to_i16_b(p1_, SW_RHNE);                                         \
        short128 q2_ = v_convert_f16_to_i16_b(p2_, SW_RHNE);                                         \
        short128 q3_ = v_convert_f16_to_i16_b(p3_, SW_RHNE);                                         \
        short128 q4_ = v_convert_f16_to_i16_b(p4_, SW_RHNE);                                         \
        short128 q5_ = v_convert_f16_to_i16_b(p5_, SW_RHNE);                                         \
        q0_ = v_i16_max_b(q0_, (short128)-16128); q1_ = v_i16_max_b(q1_, (short128)-16128);          \
        q2_ = v_i16_max_b(q2_, (short128)-16128); q3_ = v_i16_max_b(q3_, (short128)-16128);          \
        q4_ = v_i16_max_b(q4_, (short128)-16128); q5_ = v_i16_max_b(q5_, (short128)-16128);          \
        const half128 k0_ = v_convert_i16_to_f16_b(q0_ & (short128)127, SW_RHNE);                    \
        const half128 k1_ = v_convert_i16_to_f16_b(q1_ & (short128)127, SW_RHNE);                    \
        const half128 k2_ = v_convert_i16_to_f16_b(q2_ & (short128)127, SW_RHNE);                    \
        const half128 k3_ = v_convert_i16_to_f16_b(q3_ & (short128)127, SW_RHNE);                    \
        const half128 k4_ = v_convert_i16_to_f16_b(q4_ & (short128)127, SW_RHNE);                    \
        const half128 k5_ = v_convert_i16_to_f16_b(q5_ & (short128)127, SW_RHNE);                    \
        half128 c0_ = (half128)0.002681732178f, c1_ = c0_, c2_ = c0_;                                \
        half128 c3_ = c0_, c4_ = c0_, c5_ = c0_;                                                     \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)-0.3508300781f);                                        \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)-0.3508300781f);                                        \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)-0.3508300781f);                                        \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)-0.3508300781f);                                        \
        c4_ = v_f16_mac_b(c4_, k4_, (half128)-0.3508300781f);                                        \
        c5_ = v_f16_mac_b(c5_, k5_, (half128)-0.3508300781f);                                        \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)0.4558105469f);                                         \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)0.4558105469f);                                         \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)0.4558105469f);                                         \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)0.4558105469f);                                         \
        c4_ = v_f16_mac_b(c4_, k4_, (half128)0.4558105469f);                                         \
        c5_ = v_f16_mac_b(c5_, k5_, (half128)0.4558105469f);                                         \
        const short128 e0_ = v_convert_f16_to_i16_b(c0_, SW_RHNE);                                   \
        const short128 e1_ = v_convert_f16_to_i16_b(c1_, SW_RHNE);                                   \
        const short128 e2_ = v_convert_f16_to_i16_b(c2_, SW_RHNE);                                   \
        const short128 e3_ = v_convert_f16_to_i16_b(c3_, SW_RHNE);                                   \
        const short128 e4_ = v_convert_f16_to_i16_b(c4_, SW_RHNE);                                   \
        const short128 e5_ = v_convert_f16_to_i16_b(c5_, SW_RHNE);                                   \
        short128 b0_ = q0_ + e0_, b1_ = q1_ + e1_, b2_ = q2_ + e2_;                                  \
        short128 b3_ = q3_ + e3_, b4_ = q4_ + e4_, b5_ = q5_ + e5_;                                  \
        b0_ += (short128)16256; b1_ += (short128)16256; b2_ += (short128)16256;                      \
        b3_ += (short128)16256; b4_ += (short128)16256; b5_ += (short128)16256;                      \
        (o0) = *(bfloat128*)&b0_; (o1) = *(bfloat128*)&b1_; (o2) = *(bfloat128*)&b2_;                \
        (o3) = *(bfloat128*)&b3_; (o4) = *(bfloat128*)&b4_; (o5) = *(bfloat128*)&b5_;                \
    } while (0)

void main(tensor sf, tensor pf, tensor m_buf, tensor l_buf, tensor r_sm,
          tensor of_in, tensor o_acc, tensor r_acc,
          float scale, int nv, int do_sm, int do_acc)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();

    if (do_sm) {
        const int TW = get_dim_size(sf, 0);
        const int NC = TW >> 7;
        const int nfc = nv >> 7;
        const int nfp = nv & 127;
        const int n8 = nfc >> 3;
        const int n8_end = n8 << 3;
        const int n6 = (nfc * 43) >> 8;  // floor(nfc / 6) exact for 0 <= nfc <= 47
        const int n6_end = n6 * 6;
        float128 kf;
        kf.v1 = 128.0f * 1.4426950408889634f * scale;
        kf.v2 = kf.v1;
        const half128 kh = v_convert_f32_to_f16_all_b(kf);
        const half128 nkh = v_f16_sub_b((half128)0.0f, kh);
        const short128 lane = (short128)read_lane_id_2b_b();
        const uint64 lanes4 = read_lane_id_4b_b();
        const float64 lane64 = v_convert_i32_to_f32_b(as_int64(lanes4), 0);
        const uint64 l8 = lanes4 & 7;
        const uint64 sc4 = ((l8 ^ 4) | 0x80) * 0x01010101;
        const uint64 sc2 = ((l8 ^ 2) | 0x80) * 0x01010101;
        const uint64 sc1 = ((l8 ^ 1) | 0x80) * 0x01010101;
        const uchar256 c4 = *(const uchar256*)&sc4;
        const uchar256 c2 = *(const uchar256*)&sc2;
        const uchar256 c1 = *(const uchar256*)&sc1;
        const half128 negf = (half128)-65504.0f;
        const short128 negb = *(short128*)&negf;
        const bfloat128 zb = 0;

        for (int r_base = start[0]; r_base < end[0]; r_base += 64) {
            float64 mr_vec = 0.0f;
            float64 lt_vec = 0.0f;
            float fi = 0.0f;
            for (int g4 = 0; g4 < 16; ++g4, fi += 4.0f) {
                const int r0 = r_base + (g4 << 2);
                half128 rm[4];

                // ---- Pass 1: load 4 rows of sf into VLM sbuf[0..3][] with interleaved max (overlaps ld/st!)
                for (int u = 0; u < 4; ++u) {
                    const int r = r0 + u;
                    half128 m0 = negf, m1 = negf, m2 = negf, m3 = negf;
                    for (int k = 0; k < n8; ++k) {
                        const int c = k << 3;
                        int5 fc = {c << 7, r, 0, 0, 0};
                        const half128 f0 = v_f16_ld_tnsr_b(fc, sf); fc[0] += 128;
                        const half128 f1 = v_f16_ld_tnsr_b(fc, sf); fc[0] += 128;
                        const half128 f2 = v_f16_ld_tnsr_b(fc, sf); fc[0] += 128;
                        const half128 f3 = v_f16_ld_tnsr_b(fc, sf); fc[0] += 128;
                        sbuf[u][c] = f0;     m0 = v_f16_max_b(m0, f0);
                        sbuf[u][c + 1] = f1; m1 = v_f16_max_b(m1, f1);
                        sbuf[u][c + 2] = f2; m2 = v_f16_max_b(m2, f2);
                        sbuf[u][c + 3] = f3; m3 = v_f16_max_b(m3, f3);
                        const half128 f4 = v_f16_ld_tnsr_b(fc, sf); fc[0] += 128;
                        const half128 f5 = v_f16_ld_tnsr_b(fc, sf); fc[0] += 128;
                        const half128 f6 = v_f16_ld_tnsr_b(fc, sf); fc[0] += 128;
                        const half128 f7 = v_f16_ld_tnsr_b(fc, sf);
                        sbuf[u][c + 4] = f4; m0 = v_f16_max_b(m0, f4);
                        sbuf[u][c + 5] = f5; m1 = v_f16_max_b(m1, f5);
                        sbuf[u][c + 6] = f6; m2 = v_f16_max_b(m2, f6);
                        sbuf[u][c + 7] = f7; m3 = v_f16_max_b(m3, f7);
                    }
                    for (int c = n8_end; c < nfc; ++c) {
                        int5 fc = {c << 7, r, 0, 0, 0};
                        const half128 f0 = v_f16_ld_tnsr_b(fc, sf);
                        sbuf[u][c] = f0;
                        m0 = v_f16_max_b(m0, f0);
                    }
                    if (nfp) {
                        int5 fc = {nfc << 7, r, 0, 0, 0};
                        const half128 fv = v_f16_ld_tnsr_b(fc, sf);
                        const short128 fm = v_i16_sel_less_i16_b(lane, (short)nfp, *(short128*)&fv, negb);
                        const half128 fh = *(half128*)&fm;
                        sbuf[u][nfc] = fh;
                        m1 = v_f16_max_b(m1, fh);
                    }
                    rm[u] = v_f16_max_b(v_f16_max_b(m0, m1), v_f16_max_b(m2, m3));
                }

                // ---- 4-row lockstep max64 + m_old merge + FP16 round + precompute -mh*kh
                const float128 mf0 = v_convert_f16_to_f32_all_b(rm[0]);
                const float128 mf1 = v_convert_f16_to_f32_all_b(rm[1]);
                const float128 mf2 = v_convert_f16_to_f32_all_b(rm[2]);
                const float128 mf3 = v_convert_f16_to_f32_all_b(rm[3]);
                float64 mt0 = v_f32_max_b(mf0.v1, mf0.v2);
                float64 mt1 = v_f32_max_b(mf1.v1, mf1.v2);
                float64 mt2 = v_f32_max_b(mf2.v1, mf2.v2);
                float64 mt3 = v_f32_max_b(mf3.v1, mf3.v2);
                MAX64_4(mt0, mt1, mt2, mt3, c4, c2, c1);

                int5 mc0 = {r0, 0, 0, 0, 0}, mc1 = {r0 + 1, 0, 0, 0, 0};
                int5 mc2 = {r0 + 2, 0, 0, 0, 0}, mc3 = {r0 + 3, 0, 0, 0, 0};
                const float64 mo0 = (float64)s_f32_ld_g((__global__ float*)gen_addr(mc0, m_buf));
                const float64 mo1 = (float64)s_f32_ld_g((__global__ float*)gen_addr(mc1, m_buf));
                const float64 mo2 = (float64)s_f32_ld_g((__global__ float*)gen_addr(mc2, m_buf));
                const float64 mo3 = (float64)s_f32_ld_g((__global__ float*)gen_addr(mc3, m_buf));
                float128 mm0, mm1, mm2, mm3;
                mm0.v1 = v_f32_max_b(mo0, mt0); mm0.v2 = mm0.v1;
                mm1.v1 = v_f32_max_b(mo1, mt1); mm1.v2 = mm1.v1;
                mm2.v1 = v_f32_max_b(mo2, mt2); mm2.v2 = mm2.v1;
                mm3.v1 = v_f32_max_b(mo3, mt3); mm3.v2 = mm3.v1;
                const half128 mh0 = v_convert_f32_to_f16_all_b(mm0);
                const half128 mh1 = v_convert_f32_to_f16_all_b(mm1);
                const half128 mh2 = v_convert_f32_to_f16_all_b(mm2);
                const half128 mh3 = v_convert_f32_to_f16_all_b(mm3);
                half128 nmhv[4];
                nmhv[0] = v_f16_mul_b(mh0, nkh);
                nmhv[1] = v_f16_mul_b(mh1, nkh);
                nmhv[2] = v_f16_mul_b(mh2, nkh);
                nmhv[3] = v_f16_mul_b(mh3, nkh);
                const float128 mr0 = v_convert_f16_to_f32_all_b(mh0);
                const float128 mr1 = v_convert_f16_to_f32_all_b(mh1);
                const float128 mr2 = v_convert_f16_to_f32_all_b(mh2);
                const float128 mr3 = v_convert_f16_to_f32_all_b(mh3);
                mr_vec = v_f32_sel_eq_f32_b(lane64, fi, mr0.v1, mr_vec);
                mr_vec = v_f32_sel_eq_f32_b(lane64, fi + 1.0f, mr1.v1, mr_vec);
                mr_vec = v_f32_sel_eq_f32_b(lane64, fi + 2.0f, mr2.v1, mr_vec);
                mr_vec = v_f32_sel_eq_f32_b(lane64, fi + 3.0f, mr3.v1, mr_vec);

                // ---- Pass 2: 6-way PROB6 HW loop (3 accumulators + dual store coords -> ZERO spills, ZERO NOPs!)
                float64 sa[4];
                for (int u = 0; u < 4; ++u) {
                    const int r = r0 + u;
                    const half128 nmhkh = nmhv[u];
                    float128 a0, a1, a2;
                    a0.v1 = 0; a0.v2 = 0;
                    a1 = a0; a2 = a0;

                    for (int k = 0; k < n6; ++k) {
                        const int c = k * 6;
                        bfloat128 o0, o1, o2, o3, o4, o5;
                        PROB6(sbuf[u][c], sbuf[u][c + 1], sbuf[u][c + 2],
                              sbuf[u][c + 3], sbuf[u][c + 4], sbuf[u][c + 5],
                              o0, o1, o2, o3, o4, o5);
                        int5 oc0 = {c << 7, r, 0, 0, 0};
                        int5 oc1 = {(c + 1) << 7, r, 0, 0, 0};
                        ACC(a0, o0); v_bf16_st_tnsr(oc0, pf, o0); oc0[0] += 256;
                        ACC(a1, o1); v_bf16_st_tnsr(oc1, pf, o1); oc1[0] += 256;
                        ACC(a2, o2); v_bf16_st_tnsr(oc0, pf, o2); oc0[0] += 256;
                        ACC(a0, o3); v_bf16_st_tnsr(oc1, pf, o3); oc1[0] += 256;
                        ACC(a1, o4); v_bf16_st_tnsr(oc0, pf, o4);
                        ACC(a2, o5); v_bf16_st_tnsr(oc1, pf, o5);
                    }
                    int c = n6_end;
                    for (; c < nfc; ++c) {
                        bfloat128 o0;
                        PROB(sbuf[u][c], o0);
                        ACC(a0, o0);
                        int5 oc = {c << 7, r, 0, 0, 0};
                        v_bf16_st_tnsr(oc, pf, o0);
                    }
                    if (nfp) {
                        bfloat128 o0;
                        PROB(sbuf[u][nfc], o0);
                        const short128 om = v_i16_sel_less_i16_b(lane, (short)nfp, *(short128*)&o0, (short128)0);
                        o0 = *(bfloat128*)&om;
                        ACC(a1, o0);
                        int5 oc = {nfc << 7, r, 0, 0, 0};
                        v_bf16_st_tnsr(oc, pf, o0);
                        ++c;
                    }
                    for (; c < NC; ++c) {
                        int5 oc = {c << 7, r, 0, 0, 0};
                        v_bf16_st_tnsr(oc, pf, zb);
                    }
                    sa[u] = ((a0.v1 + a1.v1) + a2.v1) + ((a0.v2 + a1.v2) + a2.v2);
                }

                // ---- 4-row lockstep sum64 + lt_vec update
                SUM64_4(sa[0], sa[1], sa[2], sa[3], c4, c2, c1);
                lt_vec = v_f32_sel_eq_f32_b(lane64, fi, sa[0], lt_vec);
                lt_vec = v_f32_sel_eq_f32_b(lane64, fi + 1.0f, sa[1], lt_vec);
                lt_vec = v_f32_sel_eq_f32_b(lane64, fi + 2.0f, sa[2], lt_vec);
                lt_vec = v_f32_sel_eq_f32_b(lane64, fi + 3.0f, sa[3], lt_vec);
            }
            // Vectorized 64-row online-softmax update: ONE exp_vec + 3 aligned 256-B SRAM stores!
            int5 vc = {r_base, 0, 0, 0, 0};
            const float64 m_old_v = v_f32_ld_tnsr_b(vc, m_buf);
            const float64 l_old_v = v_f32_ld_tnsr_b(vc, l_buf);
            const float64 alpha_v = exp_vec((m_old_v - mr_vec) * scale);
            const float64 l_new_v = v_f32_mac_b(l_old_v, alpha_v, lt_vec);
            v_f32_st_tnsr(vc, m_buf, mr_vec);
            v_f32_st_tnsr(vc, l_buf, l_new_v);
            v_f32_st_tnsr(vc, r_sm, alpha_v);
        }
    }

    if (do_acc) {
        for (int r = start[0]; r < end[0]; r += 2) {
            int5 rc0 = {r, 0, 0, 0, 0}, rc1 = {r + 1, 0, 0, 0, 0};
            const float64 a0 = (float64)s_f32_ld_g((__global__ float*)gen_addr(rc0, r_acc));
            const float64 a1 = (float64)s_f32_ld_g((__global__ float*)gen_addr(rc1, r_acc));
            int5 c0 = {0, r, 0, 0, 0}, c1 = {64, r, 0, 0, 0};
            int5 c2 = {0, r + 1, 0, 0, 0}, c3 = {64, r + 1, 0, 0, 0};
            const float64 f0 = v_f32_ld_tnsr_b(c0, of_in);
            const float64 f1 = v_f32_ld_tnsr_b(c1, of_in);
            const float64 f2 = v_f32_ld_tnsr_b(c2, of_in);
            const float64 f3 = v_f32_ld_tnsr_b(c3, of_in);
            const float64 o0 = v_f32_ld_tnsr_b(c0, o_acc);
            const float64 o1 = v_f32_ld_tnsr_b(c1, o_acc);
            const float64 o2 = v_f32_ld_tnsr_b(c2, o_acc);
            const float64 o3 = v_f32_ld_tnsr_b(c3, o_acc);
            v_f32_st_tnsr(c0, o_acc, v_f32_mac_b(o0, a0, f0));
            v_f32_st_tnsr(c1, o_acc, v_f32_mac_b(o1, a0, f1));
            v_f32_st_tnsr(c2, o_acc, v_f32_mac_b(o2, a1, f2));
            v_f32_st_tnsr(c3, o_acc, v_f32_mac_b(o3, a1, f3));
        }
    }
}
