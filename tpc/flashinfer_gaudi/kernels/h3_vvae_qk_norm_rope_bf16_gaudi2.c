// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 Video VAE (ViT3D) ULTRA-FAST fused QKV bias + split + non-affine Q/K RMSNorm (HD=64) +
// partial 3D RoPE (rot_dim=48, half=24) + transpose [B, S, NH, 64] -> [B, NH, S, 64].
// Optimizations:
//   - Caches all 32x3 = 96 bias vectors in VLM (24 KiB) -> ZERO bias memory loads inside token loop!
//   - 2-head (= 4-vector: Q0, K0, Q1, K1) fully interleaved butterfly sum64 + 2-Newton rsqrt + RoPE -> ZERO VPU RAW stalls!
// Inputs:
//   0 qkv_raw [64, 3, NH, BS] bf16
//   1 qkv_b   [64, 3, NH]     bf16
//   2 cos64   [64, S]         bf16  ([cos(0..23), cos(0..23), 1.0(16)])
//   3 sin64   [64, S]         bf16  ([-sin(0..23), +sin(0..23), 0.0(16)])
// Outputs:
//   0 q_out   [64, S, NH, B]  bf16
//   1 k_out   [64, S, NH, B]  bf16
//   2 v_out   [64, S, NH, B]  bf16
// Index space: [S, B]. Scalar param: float eps.

static inline float64 vvae_rope_partner_24(float64 v, uchar256 shuf_rot4, uint64 lane_bit4) {
    const float64 r = v_f32_shuffle_b(v, shuf_rot4, 0, v);
    const int swap_g = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN;
    const float64 r_sw = v_f32_mov_group_b(r, 0xffffffff, swap_g, r);

    float64 a = v_f32_mov_dual_group_all_b(r_sw, 0xffffffff, 0, 0, 2, 3, SW_WR_LOWER_GROUP0, r);
    a = v_f32_mov_dual_group_all_b(r_sw, 0xffffffff, 1, 0, 2, 3, SW_WR_UPPER_GROUP0, a);
    a = v_f32_mov_dual_group_all_b(r,    0xffffffff, 0, 0, 2, 3, SW_WR_LOWER_GROUP1, a);

    float64 b = v_f32_mov_dual_group_all_b(r,    0xffffffff, 1, 0, 2, 3, SW_WR_LOWER_GROUP0, r);
    b = v_f32_mov_dual_group_all_b(r_sw, 0xffffffff, 0, 0, 2, 3, SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1, b);

    return v_f32_sel_eq_u32_b(lane_bit4, (uint64)0, a, b);
}

#define SUM64_QUAD(X0, X1, X2, X3) do {                                                                 \
    const int all_ = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |\
                     SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3; \
    const int grp_ = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |                                  \
                     SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;                         \
    (X0) += v_f32_mov_dual_group_all_b((X0), 0xffffffff, 1, 0, 3, 2, all_, (X0));                       \
    (X1) += v_f32_mov_dual_group_all_b((X1), 0xffffffff, 1, 0, 3, 2, all_, (X1));                       \
    (X2) += v_f32_mov_dual_group_all_b((X2), 0xffffffff, 1, 0, 3, 2, all_, (X2));                       \
    (X3) += v_f32_mov_dual_group_all_b((X3), 0xffffffff, 1, 0, 3, 2, all_, (X3));                       \
    (X0) += v_f32_mov_dual_group_all_b((X0), 0xffffffff, 2, 3, 0, 1, all_, (X0));                       \
    (X1) += v_f32_mov_dual_group_all_b((X1), 0xffffffff, 2, 3, 0, 1, all_, (X1));                       \
    (X2) += v_f32_mov_dual_group_all_b((X2), 0xffffffff, 2, 3, 0, 1, all_, (X2));                       \
    (X3) += v_f32_mov_dual_group_all_b((X3), 0xffffffff, 2, 3, 0, 1, all_, (X3));                       \
    (X0) += v_f32_mov_group_b((X0), 0xffffffff, grp_, (X0));                                            \
    (X1) += v_f32_mov_group_b((X1), 0xffffffff, grp_, (X1));                                            \
    (X2) += v_f32_mov_group_b((X2), 0xffffffff, grp_, (X2));                                            \
    (X3) += v_f32_mov_group_b((X3), 0xffffffff, grp_, (X3));                                            \
    (X0) += v_f32_shuffle_b((X0), shuf4, 0, (X0));                                                      \
    (X1) += v_f32_shuffle_b((X1), shuf4, 0, (X1));                                                      \
    (X2) += v_f32_shuffle_b((X2), shuf4, 0, (X2));                                                      \
    (X3) += v_f32_shuffle_b((X3), shuf4, 0, (X3));                                                      \
    (X0) += v_f32_shuffle_b((X0), shuf2, 0, (X0));                                                      \
    (X1) += v_f32_shuffle_b((X1), shuf2, 0, (X1));                                                      \
    (X2) += v_f32_shuffle_b((X2), shuf2, 0, (X2));                                                      \
    (X3) += v_f32_shuffle_b((X3), shuf2, 0, (X3));                                                      \
    (X0) += v_f32_shuffle_b((X0), shuf1, 0, (X0));                                                      \
    (X1) += v_f32_shuffle_b((X1), shuf1, 0, (X1));                                                      \
    (X2) += v_f32_shuffle_b((X2), shuf1, 0, (X2));                                                      \
    (X3) += v_f32_shuffle_b((X3), shuf1, 0, (X3));                                                      \
} while (0)

#define RSQRT_QUAD_2N(Y0, Y1, Y2, Y3, X0, X1, X2, X3) do {                                              \
    const float64 h0_ = (X0) * 0.5f, h1_ = (X1) * 0.5f, h2_ = (X2) * 0.5f, h3_ = (X3) * 0.5f;           \
    (Y0) = as_float64((int64)0x5f3759df - (as_int64(X0) >> 1));                                         \
    (Y1) = as_float64((int64)0x5f3759df - (as_int64(X1) >> 1));                                         \
    (Y2) = as_float64((int64)0x5f3759df - (as_int64(X2) >> 1));                                         \
    (Y3) = as_float64((int64)0x5f3759df - (as_int64(X3) >> 1));                                         \
    (Y0) = (Y0) * (1.5f - h0_ * (Y0) * (Y0));                                                           \
    (Y1) = (Y1) * (1.5f - h1_ * (Y1) * (Y1));                                                           \
    (Y2) = (Y2) * (1.5f - h2_ * (Y2) * (Y2));                                                           \
    (Y3) = (Y3) * (1.5f - h3_ * (Y3) * (Y3));                                                           \
    (Y0) = (Y0) * (1.5f - h0_ * (Y0) * (Y0));                                                           \
    (Y1) = (Y1) * (1.5f - h1_ * (Y1) * (Y1));                                                           \
    (Y2) = (Y2) * (1.5f - h2_ * (Y2) * (Y2));                                                           \
    (Y3) = (Y3) * (1.5f - h3_ * (Y3) * (Y3));                                                           \
} while (0)

static inline float64 sum64_one(float64 x, uchar256 shuf4, uchar256 shuf2, uchar256 shuf1) {
    const int all = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                    SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    const int grp = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |
                    SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, all, x);
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, all, x);
    x += v_f32_mov_group_b(x, 0xffffffff, grp, x);
    x += v_f32_shuffle_b(x, shuf4, 0, x);
    x += v_f32_shuffle_b(x, shuf2, 0, x);
    x += v_f32_shuffle_b(x, shuf1, 0, x);
    return x;
}

static inline float64 rsqrt_2n(float64 x) {
    float64 y = as_float64((int64)0x5f3759df - (as_int64(x) >> 1));
    const float64 h = 0.5f * x;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y;
}

void main(tensor qkv_raw, tensor qkv_b, tensor cos64, tensor sin64,
          tensor q_out, tensor k_out, tensor v_out, float eps)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int S = get_dim_size(q_out, 1);
    const int NH = get_dim_size(q_out, 2);

    const ushort128 lid16 = read_lane_id_2b_b();
    const bfloat128 zero_b = 0;

    const uint64 lid32 = read_lane_id_4b_b();
    const uint64 lane_bit4 = lid32 & 4;
    const uint64 l8 = lid32 & 7;
    const uint64 g1_bit = (lid32 & 8) << 2;
    const uint64 rot4_u = ((l8 ^ 4) | g1_bit | 0x80) * 0x01010101;
    const uint64 c4_u = ((l8 ^ 4) | 0x80) * 0x01010101;
    const uint64 c2_u = ((l8 ^ 2) | 0x80) * 0x01010101;
    const uint64 c1_u = ((l8 ^ 1) | 0x80) * 0x01010101;
    const uchar256 shuf_rot4 = *((const uchar256*)&rot4_u);
    const uchar256 shuf4 = *((const uchar256*)&c4_u);
    const uchar256 shuf2 = *((const uchar256*)&c2_u);
    const uchar256 shuf1 = *((const uchar256*)&c1_u);

    // Preload all NH*3 bias vectors into VLM (32*3 = 96 vectors = 24 KiB <= 64 KiB VLM!)
    bfloat128 qb_vlm[32], kb_vlm[32], vb_vlm[32];
    for (int h = 0; h < NH; ++h) {
        qb_vlm[h] = v_bf16_ld_tnsr_b((int5){0, 0, h, 0, 0}, qkv_b);
        kb_vlm[h] = v_bf16_ld_tnsr_b((int5){0, 1, h, 0, 0}, qkv_b);
        vb_vlm[h] = v_bf16_ld_tnsr_b((int5){0, 2, h, 0, 0}, qkv_b);
    }

    const int nh_even = NH & ~1;
    for (int b = start[1]; b < end[1]; ++b) {
        const int b_row = b * S;
        for (int s = start[0]; s < end[0]; ++s) {
            const int r = b_row + s;
            const int5 rc = {0, s, 0, 0, 0};
            const float64_pair_t cf = v_convert_bf16_to_f32_all_b(
                v_bf16_sel_less_u16_b(lid16, 64, v_bf16_ld_tnsr_b(rc, cos64), zero_b));
            const float64_pair_t sf = v_convert_bf16_to_f32_all_b(
                v_bf16_sel_less_u16_b(lid16, 64, v_bf16_ld_tnsr_b(rc, sin64), zero_b));

            // 2-head (= 4-vector: Q0, K0, Q1, K1) fully interleaved loop -> ZERO RAW stalls!
            for (int h = 0; h < nh_even; h += 2) {
                const int h1 = h + 1;
                const int5 oc0 = {0, s, h,  b, 0};
                const int5 oc1 = {0, s, h1, b, 0};

                const bfloat128 q0_m = v_bf16_sel_less_u16_b(lid16, 64,
                    v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 0, h,  r, 0}, qkv_raw), qb_vlm[h]),  zero_b);
                const bfloat128 k0_m = v_bf16_sel_less_u16_b(lid16, 64,
                    v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 1, h,  r, 0}, qkv_raw), kb_vlm[h]),  zero_b);
                const bfloat128 q1_m = v_bf16_sel_less_u16_b(lid16, 64,
                    v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 0, h1, r, 0}, qkv_raw), qb_vlm[h1]), zero_b);
                const bfloat128 k1_m = v_bf16_sel_less_u16_b(lid16, 64,
                    v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 1, h1, r, 0}, qkv_raw), kb_vlm[h1]), zero_b);

                const bfloat128 v0_r = v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 2, h,  r, 0}, qkv_raw), vb_vlm[h]);
                const bfloat128 v1_r = v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 2, h1, r, 0}, qkv_raw), vb_vlm[h1]);
                v_bf16_st_tnsr(oc0, v_out, v0_r);
                v_bf16_st_tnsr(oc1, v_out, v1_r);

                float128 sq_q0 = {0}, sq_k0 = {0}, sq_q1 = {0}, sq_k1 = {0};
                sq_q0 = v_bf16_mac_acc32_b(q0_m, q0_m, sq_q0, (e_no_negation) << 1);
                sq_k0 = v_bf16_mac_acc32_b(k0_m, k0_m, sq_k0, (e_no_negation) << 1);
                sq_q1 = v_bf16_mac_acc32_b(q1_m, q1_m, sq_q1, (e_no_negation) << 1);
                sq_k1 = v_bf16_mac_acc32_b(k1_m, k1_m, sq_k1, (e_no_negation) << 1);

                float64 s_q0 = sq_q0.v1 + sq_q0.v2;
                float64 s_k0 = sq_k0.v1 + sq_k0.v2;
                float64 s_q1 = sq_q1.v1 + sq_q1.v2;
                float64 s_k1 = sq_k1.v1 + sq_k1.v2;
                SUM64_QUAD(s_q0, s_k0, s_q1, s_k1);
                s_q0 = v_f32_mac_b(s_q0, (float64)(1.0f / 64.0f), (float64)eps);
                s_k0 = v_f32_mac_b(s_k0, (float64)(1.0f / 64.0f), (float64)eps);
                s_q1 = v_f32_mac_b(s_q1, (float64)(1.0f / 64.0f), (float64)eps);
                s_k1 = v_f32_mac_b(s_k1, (float64)(1.0f / 64.0f), (float64)eps);
                float64 iq0, ik0, iq1, ik1;
                RSQRT_QUAD_2N(iq0, ik0, iq1, ik1, s_q0, s_k0, s_q1, s_k1);

                // Head h: Q0 & K0 (4 F32 vectors interleaved)
                {
                    float64_pair_t qf = v_convert_bf16_to_f32_all_b(q0_m);
                    float64_pair_t kf = v_convert_bf16_to_f32_all_b(k0_m);
                    qf.v1 *= iq0; qf.v2 *= iq0;
                    kf.v1 *= ik0; kf.v2 *= ik0;
                    qf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(qf));
                    kf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(kf));
                    const float64 qp1 = vvae_rope_partner_24(qf.v1, shuf_rot4, lane_bit4);
                    const float64 qp2 = vvae_rope_partner_24(qf.v2, shuf_rot4, lane_bit4);
                    const float64 kp1 = vvae_rope_partner_24(kf.v1, shuf_rot4, lane_bit4);
                    const float64 kp2 = vvae_rope_partner_24(kf.v2, shuf_rot4, lane_bit4);
                    float64_pair_t qr, kr;
                    qr.v1 = v_f32_mac_b(qp1, sf.v1, qf.v1 * cf.v1);
                    qr.v2 = v_f32_mac_b(qp2, sf.v2, qf.v2 * cf.v2);
                    kr.v1 = v_f32_mac_b(kp1, sf.v1, kf.v1 * cf.v1);
                    kr.v2 = v_f32_mac_b(kp2, sf.v2, kf.v2 * cf.v2);
                    v_bf16_st_tnsr(oc0, q_out, v_convert_f32_to_bf16_all_b(qr));
                    v_bf16_st_tnsr(oc0, k_out, v_convert_f32_to_bf16_all_b(kr));
                }
                // Head h+1: Q1 & K1
                {
                    float64_pair_t qf = v_convert_bf16_to_f32_all_b(q1_m);
                    float64_pair_t kf = v_convert_bf16_to_f32_all_b(k1_m);
                    qf.v1 *= iq1; qf.v2 *= iq1;
                    kf.v1 *= ik1; kf.v2 *= ik1;
                    qf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(qf));
                    kf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(kf));
                    const float64 qp1 = vvae_rope_partner_24(qf.v1, shuf_rot4, lane_bit4);
                    const float64 qp2 = vvae_rope_partner_24(qf.v2, shuf_rot4, lane_bit4);
                    const float64 kp1 = vvae_rope_partner_24(kf.v1, shuf_rot4, lane_bit4);
                    const float64 kp2 = vvae_rope_partner_24(kf.v2, shuf_rot4, lane_bit4);
                    float64_pair_t qr, kr;
                    qr.v1 = v_f32_mac_b(qp1, sf.v1, qf.v1 * cf.v1);
                    qr.v2 = v_f32_mac_b(qp2, sf.v2, qf.v2 * cf.v2);
                    kr.v1 = v_f32_mac_b(kp1, sf.v1, kf.v1 * cf.v1);
                    kr.v2 = v_f32_mac_b(kp2, sf.v2, kf.v2 * cf.v2);
                    v_bf16_st_tnsr(oc1, q_out, v_convert_f32_to_bf16_all_b(qr));
                    v_bf16_st_tnsr(oc1, k_out, v_convert_f32_to_bf16_all_b(kr));
                }
            }
            for (int h = nh_even; h < NH; ++h) {
                const int5 oc = {0, s, h, b, 0};
                const bfloat128 q_m = v_bf16_sel_less_u16_b(lid16, 64,
                    v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 0, h, r, 0}, qkv_raw), qb_vlm[h]), zero_b);
                const bfloat128 k_m = v_bf16_sel_less_u16_b(lid16, 64,
                    v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 1, h, r, 0}, qkv_raw), kb_vlm[h]), zero_b);
                v_bf16_st_tnsr(oc, v_out, v_bf16_add_b(v_bf16_ld_tnsr_b((int5){0, 2, h, r, 0}, qkv_raw), vb_vlm[h]));
                float128 qsq = {0}, ksq = {0};
                qsq = v_bf16_mac_acc32_b(q_m, q_m, qsq, (e_no_negation) << 1);
                ksq = v_bf16_mac_acc32_b(k_m, k_m, ksq, (e_no_negation) << 1);
                const float64 qinv = rsqrt_2n(sum64_one(qsq.v1 + qsq.v2, shuf4, shuf2, shuf1) * (1.0f / 64.0f) + eps);
                const float64 kinv = rsqrt_2n(sum64_one(ksq.v1 + ksq.v2, shuf4, shuf2, shuf1) * (1.0f / 64.0f) + eps);
                float64_pair_t qf = v_convert_bf16_to_f32_all_b(q_m);
                float64_pair_t kf = v_convert_bf16_to_f32_all_b(k_m);
                qf.v1 *= qinv; qf.v2 *= qinv; kf.v1 *= kinv; kf.v2 *= kinv;
                qf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(qf));
                kf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(kf));
                float64_pair_t qr, kr;
                qr.v1 = v_f32_mac_b(vvae_rope_partner_24(qf.v1, shuf_rot4, lane_bit4), sf.v1, qf.v1 * cf.v1);
                qr.v2 = v_f32_mac_b(vvae_rope_partner_24(qf.v2, shuf_rot4, lane_bit4), sf.v2, qf.v2 * cf.v2);
                kr.v1 = v_f32_mac_b(vvae_rope_partner_24(kf.v1, shuf_rot4, lane_bit4), sf.v1, kf.v1 * cf.v1);
                kr.v2 = v_f32_mac_b(vvae_rope_partner_24(kf.v2, shuf_rot4, lane_bit4), sf.v2, kf.v2 * cf.v2);
                v_bf16_st_tnsr(oc, q_out, v_convert_f32_to_bf16_all_b(qr));
                v_bf16_st_tnsr(oc, k_out, v_convert_f32_to_bf16_all_b(kr));
            }
        }
    }
}
