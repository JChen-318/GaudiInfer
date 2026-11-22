// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 ULTRA-FAST fused raw-FP8-GEMM QKV dequant + 64-row strip VLM-cached contiguous write +
// 2-head interleaved Q/K RMSNorm (D=128) + partial 3D RoPE (rot=96) + zero-padding + 256-key V interleave + block max!
//
// Inputs:
//   0 qkv_raw [3*NH*128, S] bf16  (raw fp8_gemm_bf16 output)
//   1 row_s   [1, S]        f32   (activation row scale from h3_adaln_rmsnorm_quant)
//   2 col_s   [3*NH*128]    bf16  (per-output-channel weight scale cast to bf16)
//   3 q_norm  [128]         bf16
//   4 k_norm  [128]         bf16
//   5 cos128  [128, S]      bf16
//   6 sin128  [128, S]      bf16
// Outputs:
//   0 q_pad   [128, U_pad, NH] bf16
//   1 k_pad   [128, U_pad, NH] bf16
//   2 v_int   [128, U_pad, NH] bf16  (256-key interleaved: [blk, half, idx] -> [blk, idx, half])
//   3 max_qkv [3, U_pad/256]   f32   (per-256-row-block max(|Q|), max(|K|), max(|V|))
// Index space: [U_pad / 256]. Scalar params: float eps, int valid_u.

static inline float64 h3_rope_partner_64(float64 v) {
    const int swap_g = SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |
                       SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN;
    const float64 v_sw = v_f32_mov_group_b(v, 0xffffffff, swap_g, v);
    float64 p = v_f32_mov_dual_group_all_b(
        v_sw, 0xffffffff, 1, 2, 0, 3,
        SW_WR_LOWER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_LOWER_GROUP2, v);
    p = v_f32_mov_dual_group_all_b(
        v_sw, 0xffffffff, 2, 0, 1, 3,
        SW_WR_UPPER_GROUP0 | SW_WR_UPPER_GROUP1 | SW_WR_UPPER_GROUP2, p);
    return p;
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

static inline float64 row_max_bf128(bfloat128 m, uchar256 shuf4, uchar256 shuf2, uchar256 shuf1) {
    const float64_pair_t mf = v_convert_bf16_to_f32_all_b(m);
    float64 x = v_f32_max_b(mf.v1, mf.v2);
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_max_b(x, v_f32_mov_group_b(x, 0xffffffff, 63, 0));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, shuf4, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, shuf2, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, shuf1, 0, x));
    return x;
}

void main(tensor qkv_raw, tensor row_s, tensor col_s, tensor q_norm, tensor k_norm, tensor cos128, tensor sin128,
          tensor q_pad, tensor k_pad, tensor v_int, tensor max_qkv,
          float eps, int valid_u)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NH = get_dim_size(q_pad, 2);
    const int ad = NH * 128;
    const int k_base = ad;
    const int v_base = 2 * ad;
    const int nh_even = NH & ~1;

    const int5 wc = {0, 0, 0, 0, 0};
    const float64_pair_t qw = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, q_norm));
    const float64_pair_t kw = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, k_norm));

    const uint64 lid32 = read_lane_id_4b_b();
    const uint64 l8 = lid32 & 7;
    const uint64 c4_u = ((l8 ^ 4) | 0x80) * 0x01010101;
    const uint64 c2_u = ((l8 ^ 2) | 0x80) * 0x01010101;
    const uint64 c1_u = ((l8 ^ 1) | 0x80) * 0x01010101;
    const uchar256 shuf4 = *((const uchar256*)&c4_u);
    const uchar256 shuf2 = *((const uchar256*)&c2_u);
    const uchar256 shuf1 = *((const uchar256*)&c1_u);
    const bfloat128 zero_b = 0;

    // 32 KiB VLM strip caches for 64 tokens -> outer loop over heads writes contiguous 16 KiB HBM bursts!
    bfloat128 cos_vlm[64];
    bfloat128 sin_vlm[64];
    float rs_vlm[64];

    for (int blk = start[0]; blk < end[0]; ++blk) {
        const int r_base = blk * 256;
        bfloat128 mq = 0, mk = 0, mv = 0;

        for (int st = 0; st < 4; ++st) {
            const int st_off = st * 64;
            const int r0 = r_base + st_off;
            if (r0 >= valid_u) {
                for (int h = 0; h < NH; ++h) {
                    for (int k = 0; k < 64; ++k) {
                        const int r = r0 + k;
                        const int off = st_off + k;
                        const int r_vint = r_base + ((off & 127) << 1) + (off >> 7);
                        v_bf16_st_tnsr((int5){0, r, h, 0, 0}, q_pad, zero_b);
                        v_bf16_st_tnsr((int5){0, r, h, 0, 0}, k_pad, zero_b);
                        v_bf16_st_tnsr((int5){0, r_vint, h, 0, 0}, v_int, zero_b);
                    }
                }
                continue;
            }

            const int valid_k = (r0 + 64 <= valid_u) ? 64 : (valid_u - r0);
            for (int k = 0; k < valid_k; ++k) {
                const int r = r0 + k;
                cos_vlm[k] = v_bf16_ld_tnsr_b((int5){0, r, 0, 0, 0}, cos128);
                sin_vlm[k] = v_bf16_ld_tnsr_b((int5){0, r, 0, 0, 0}, sin128);
                rs_vlm[k] = s_f32_ld_g(gen_addr((int5){0, r, 0, 0, 0}, row_s));
            }

            for (int h = 0; h < nh_even; h += 2) {
                const int h1 = h + 1;
                const int h0_128 = h * 128, h1_128 = h1 * 128;
                const float64_pair_t qs0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){h0_128, 0, 0, 0, 0}, col_s));
                const float64_pair_t ks0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){k_base + h0_128, 0, 0, 0, 0}, col_s));
                const float64_pair_t vs0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){v_base + h0_128, 0, 0, 0, 0}, col_s));
                const float64_pair_t qs1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){h1_128, 0, 0, 0, 0}, col_s));
                const float64_pair_t ks1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){k_base + h1_128, 0, 0, 0, 0}, col_s));
                const float64_pair_t vs1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){v_base + h1_128, 0, 0, 0, 0}, col_s));

                for (int k = 0; k < valid_k; ++k) {
                    const int r = r0 + k;
                    const int off = st_off + k;
                    const int r_vint = r_base + ((off & 127) << 1) + (off >> 7);
                    const float64 rsf = (float64)rs_vlm[k];

                    // V0, V1: dequant + 256-key interleaved store + absmax
                    {
                        float64_pair_t v0f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){v_base + h0_128, r, 0, 0, 0}, qkv_raw));
                        float64_pair_t v1f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){v_base + h1_128, r, 0, 0, 0}, qkv_raw));
                        v0f.v1 = (v0f.v1 * rsf) * vs0.v1; v0f.v2 = (v0f.v2 * rsf) * vs0.v2;
                        v1f.v1 = (v1f.v1 * rsf) * vs1.v1; v1f.v2 = (v1f.v2 * rsf) * vs1.v2;
                        const bfloat128 v0 = v_convert_f32_to_bf16_all_b(v0f);
                        const bfloat128 v1 = v_convert_f32_to_bf16_all_b(v1f);
                        v_bf16_st_tnsr((int5){0, r_vint, h,  0, 0}, v_int, v0);
                        v_bf16_st_tnsr((int5){0, r_vint, h1, 0, 0}, v_int, v1);
                        mv = v_bf16_max_b(mv, v_bf16_max_b(v_bf16_abs_b(v0), v_bf16_abs_b(v1)));
                    }

                    // Q0, K0, Q1, K1: fused dequant -> 4-way butterfly RMSNorm -> partial 3D RoPE -> contiguous store
                    float64_pair_t q0f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){h0_128, r, 0, 0, 0}, qkv_raw));
                    float64_pair_t k0f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){k_base + h0_128, r, 0, 0, 0}, qkv_raw));
                    float64_pair_t q1f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){h1_128, r, 0, 0, 0}, qkv_raw));
                    float64_pair_t k1f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((int5){k_base + h1_128, r, 0, 0, 0}, qkv_raw));
                    q0f.v1 = (q0f.v1 * rsf) * qs0.v1; q0f.v2 = (q0f.v2 * rsf) * qs0.v2;
                    k0f.v1 = (k0f.v1 * rsf) * ks0.v1; k0f.v2 = (k0f.v2 * rsf) * ks0.v2;
                    q1f.v1 = (q1f.v1 * rsf) * qs1.v1; q1f.v2 = (q1f.v2 * rsf) * qs1.v2;
                    k1f.v1 = (k1f.v1 * rsf) * ks1.v1; k1f.v2 = (k1f.v2 * rsf) * ks1.v2;

                    float64 s_q0 = v_f32_mac_b(q0f.v2, q0f.v2, q0f.v1 * q0f.v1);
                    float64 s_k0 = v_f32_mac_b(k0f.v2, k0f.v2, k0f.v1 * k0f.v1);
                    float64 s_q1 = v_f32_mac_b(q1f.v2, q1f.v2, q1f.v1 * q1f.v1);
                    float64 s_k1 = v_f32_mac_b(k1f.v2, k1f.v2, k1f.v1 * k1f.v1);
                    SUM64_QUAD(s_q0, s_k0, s_q1, s_k1);
                    s_q0 = v_f32_mac_b(s_q0, (float64)(1.0f / 128.0f), (float64)eps);
                    s_k0 = v_f32_mac_b(s_k0, (float64)(1.0f / 128.0f), (float64)eps);
                    s_q1 = v_f32_mac_b(s_q1, (float64)(1.0f / 128.0f), (float64)eps);
                    s_k1 = v_f32_mac_b(s_k1, (float64)(1.0f / 128.0f), (float64)eps);
                    float64 iq0, ik0, iq1, ik1;
                    RSQRT_QUAD_2N(iq0, ik0, iq1, ik1, s_q0, s_k0, s_q1, s_k1);

                    const float64_pair_t cf = v_convert_bf16_to_f32_all_b(cos_vlm[k]);
                    const float64_pair_t sf = v_convert_bf16_to_f32_all_b(sin_vlm[k]);

                    // Head h
                    {
                        q0f.v1 = (q0f.v1 * iq0) * qw.v1; q0f.v2 = (q0f.v2 * iq0) * qw.v2;
                        k0f.v1 = (k0f.v1 * ik0) * kw.v1; k0f.v2 = (k0f.v2 * ik0) * kw.v2;
                        q0f = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(q0f));
                        k0f = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(k0f));
                        float64_pair_t qr, kr;
                        qr.v1 = v_f32_mac_b(h3_rope_partner_64(q0f.v1), sf.v1, q0f.v1 * cf.v1);
                        qr.v2 = v_f32_mac_b(h3_rope_partner_64(q0f.v2), sf.v2, q0f.v2 * cf.v2);
                        kr.v1 = v_f32_mac_b(h3_rope_partner_64(k0f.v1), sf.v1, k0f.v1 * cf.v1);
                        kr.v2 = v_f32_mac_b(h3_rope_partner_64(k0f.v2), sf.v2, k0f.v2 * cf.v2);
                        const bfloat128 qo = v_convert_f32_to_bf16_all_b(qr);
                        const bfloat128 ko = v_convert_f32_to_bf16_all_b(kr);
                        v_bf16_st_tnsr((int5){0, r, h, 0, 0}, q_pad, qo);
                        v_bf16_st_tnsr((int5){0, r, h, 0, 0}, k_pad, ko);
                        mq = v_bf16_max_b(mq, v_bf16_abs_b(qo));
                        mk = v_bf16_max_b(mk, v_bf16_abs_b(ko));
                    }
                    // Head h+1
                    {
                        q1f.v1 = (q1f.v1 * iq1) * qw.v1; q1f.v2 = (q1f.v2 * iq1) * qw.v2;
                        k1f.v1 = (k1f.v1 * ik1) * kw.v1; k1f.v2 = (k1f.v2 * ik1) * kw.v2;
                        q1f = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(q1f));
                        k1f = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(k1f));
                        float64_pair_t qr, kr;
                        qr.v1 = v_f32_mac_b(h3_rope_partner_64(q1f.v1), sf.v1, q1f.v1 * cf.v1);
                        qr.v2 = v_f32_mac_b(h3_rope_partner_64(q1f.v2), sf.v2, q1f.v2 * cf.v2);
                        kr.v1 = v_f32_mac_b(h3_rope_partner_64(k1f.v1), sf.v1, k1f.v1 * cf.v1);
                        kr.v2 = v_f32_mac_b(h3_rope_partner_64(k1f.v2), sf.v2, k1f.v2 * cf.v2);
                        const bfloat128 qo = v_convert_f32_to_bf16_all_b(qr);
                        const bfloat128 ko = v_convert_f32_to_bf16_all_b(kr);
                        v_bf16_st_tnsr((int5){0, r, h1, 0, 0}, q_pad, qo);
                        v_bf16_st_tnsr((int5){0, r, h1, 0, 0}, k_pad, ko);
                        mq = v_bf16_max_b(mq, v_bf16_abs_b(qo));
                        mk = v_bf16_max_b(mk, v_bf16_abs_b(ko));
                    }
                }
                for (int k = valid_k; k < 64; ++k) {
                    const int r = r0 + k;
                    const int off = st_off + k;
                    const int r_vint = r_base + ((off & 127) << 1) + (off >> 7);
                    v_bf16_st_tnsr((int5){0, r, h,  0, 0}, q_pad, zero_b);
                    v_bf16_st_tnsr((int5){0, r, h1, 0, 0}, q_pad, zero_b);
                    v_bf16_st_tnsr((int5){0, r, h,  0, 0}, k_pad, zero_b);
                    v_bf16_st_tnsr((int5){0, r, h1, 0, 0}, k_pad, zero_b);
                    v_bf16_st_tnsr((int5){0, r_vint, h,  0, 0}, v_int, zero_b);
                    v_bf16_st_tnsr((int5){0, r_vint, h1, 0, 0}, v_int, zero_b);
                }
            }
        }

        // Write 3 scalar block maxima [max_q, max_k, max_v] into max_qkv [3, U_pad/256]
        const float64 fq = row_max_bf128(mq, shuf4, shuf2, shuf1);
        const float64 fk = row_max_bf128(mk, shuf4, shuf2, shuf1);
        const float64 fv = row_max_bf128(mv, shuf4, shuf2, shuf1);
        float64 mv3 = fq;
        mv3 = v_f32_sel_eq_u32_b(lid32, 1u, fk, mv3);
        mv3 = v_f32_sel_eq_u32_b(lid32, 2u, fv, mv3);
        v_f32_st_tnsr((int5){0, blk, 0, 0, 0}, max_qkv, mv3);
    }
}
