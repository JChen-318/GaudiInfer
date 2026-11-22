// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 fused QKV split + per-head Q/K RMSNorm (D=128) + partial 3D RoPE (rot_dim=96) + transpose.
//
// Inputs:
//   qkv    [3*NH*128, S] bf16  (packed [Q(NH*128), K(NH*128), V(NH*128)] per token)
//   q_norm [128] bf16
//   k_norm [128] bf16
//   cos128 [128, S] bf16  ([cos(0..47), cos(0..47), 1.0(32)])
//   sin128 [128, S] bf16  ([-sin(0..47), +sin(0..47), 0.0(32)])
// Outputs (already sliced to valid prefix U <= S and transposed for SDPA [1, NH, U, 128]):
//   q_out  [128, U, NH] bf16
//   k_out  [128, U, NH] bf16
//   v_out  [128, U, NH] bf16
// Index space: [U] (one index point owns one valid token row r in [0..U)).
// Scalar param: float eps.

#include "norm_math_gaudi2.h"

static inline float64 h3_rope_partner_64(float64 v) {
    // Swap G0<->G1 within every dual-group:
    //   v    = [G0, G1 | G2, G3 | G4, G5 | G6, G7]
    //   v_sw = [G1, G0 | G3, G2 | G5, G4 | G7, G6]
    // Desired partner for k<24 -> k+24, 24<=k<48 -> k-24:
    //   G0<-G3, G1<-G4, G2<-G5, G3<-G0, G4<-G1, G5<-G2
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

void main(tensor qkv, tensor q_norm, tensor k_norm, tensor cos128, tensor sin128,
          tensor q_out, tensor k_out, tensor v_out, float eps)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NH = get_dim_size(q_out, 2);
    const int ad = NH * 128;
    const int k_base = ad;
    const int v_base = 2 * ad;

    const int5 wc = {0, 0, 0, 0, 0};
    const float64_pair_t qw = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, q_norm));
    const float64_pair_t kw = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, k_norm));

    for (int r = start[0]; r < end[0]; ++r) {
        const int5 rc = {0, r, 0, 0, 0};
        const float64_pair_t cf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, cos128));
        const float64_pair_t sf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, sin128));

        for (int h = 0; h < NH; ++h) {
            const int h128 = h * 128;
            const int5 qc_in = {h128, r, 0, 0, 0};
            const int5 kc_in = {k_base + h128, r, 0, 0, 0};
            const int5 vc_in = {v_base + h128, r, 0, 0, 0};
            const int5 out_c = {0, r, h, 0, 0};

            // Q: RMSNorm -> BF16 round -> 3D partial RoPE -> store [128, U, NH]
            const bfloat128 q_raw = v_bf16_ld_tnsr_b(qc_in, qkv);
            float128 qsq = {0};
            qsq = v_bf16_mac_acc32_b(q_raw, q_raw, qsq, (e_no_negation) << 1);
            const float64 qinv = positive_rsqrt(row_sum(qsq.v1 + qsq.v2) * (1.0f / 128.0f) + eps);
            float64_pair_t qf = v_convert_bf16_to_f32_all_b(q_raw);
            qf.v1 = (qf.v1 * qinv) * qw.v1;
            qf.v2 = (qf.v2 * qinv) * qw.v2;
            // Match reference BF16 rounding between RMSNorm and RoPE
            qf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(qf));
            const float64 qp1 = h3_rope_partner_64(qf.v1);
            const float64 qp2 = h3_rope_partner_64(qf.v2);
            float64_pair_t qr;
            qr.v1 = v_f32_mac_b(qp1, sf.v1, qf.v1 * cf.v1);
            qr.v2 = v_f32_mac_b(qp2, sf.v2, qf.v2 * cf.v2);
            v_bf16_st_tnsr(out_c, q_out, v_convert_f32_to_bf16_all_b(qr));

            // K: RMSNorm -> BF16 round -> 3D partial RoPE -> store [128, U, NH]
            const bfloat128 k_raw = v_bf16_ld_tnsr_b(kc_in, qkv);
            float128 ksq = {0};
            ksq = v_bf16_mac_acc32_b(k_raw, k_raw, ksq, (e_no_negation) << 1);
            const float64 kinv = positive_rsqrt(row_sum(ksq.v1 + ksq.v2) * (1.0f / 128.0f) + eps);
            float64_pair_t kf = v_convert_bf16_to_f32_all_b(k_raw);
            kf.v1 = (kf.v1 * kinv) * kw.v1;
            kf.v2 = (kf.v2 * kinv) * kw.v2;
            kf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(kf));
            const float64 kp1 = h3_rope_partner_64(kf.v1);
            const float64 kp2 = h3_rope_partner_64(kf.v2);
            float64_pair_t kr;
            kr.v1 = v_f32_mac_b(kp1, sf.v1, kf.v1 * cf.v1);
            kr.v2 = v_f32_mac_b(kp2, sf.v2, kf.v2 * cf.v2);
            v_bf16_st_tnsr(out_c, k_out, v_convert_f32_to_bf16_all_b(kr));

            // V: direct transpose store [128, U, NH]
            v_bf16_st_tnsr(out_c, v_out, v_bf16_ld_tnsr_b(vc_in, qkv));
        }
    }
}
