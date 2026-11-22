// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 fused max-free attention softmax kernel (one pass over scores):
//   scores [K_pad, R] bf16 (K_pad % 128 == 0), bound [1, R] bf16, scalar float pad_count.
//   For each row r:
//     b = f32(bound[0, r])
//     p[c, r] = bf16(exp(clamp(scores[c, r] - b, -80.0, 30.0)))
//     s = sum_c p_f32[c, r] - pad_count * exp(max(-b, -80.0))
//     inv_sum[0, r] = 1.0 / max(s, 1e-30)
//
// Uses LUT-free base-2 minimax quadratic exp (rel err 1.7e-3 < BF16 half-ULP 3.9e-3)
// with 4x128-column unrolling and 4 independent FP32 accumulators.
#include "norm_math_gaudi2.h"

static inline float64 positive_reciprocal(float64 value) {
    float64 estimate = as_float64((int64)0x7ef311c3 - as_int64(value));
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    return v_f32_sel_eq_f32_b(value, as_float64((int64)0x7f800000), 0.0f, estimate);
}

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

#define MF_STEP(COL, ACC) do { \
    const int5 c_ = {(COL), r, 0, 0, 0}; \
    const bfloat128 xb_ = v_bf16_min_b(v_bf16_max_b(v_bf16_ld_tnsr_b(c_, scores), lo_b), hi_b); \
    const float64_pair_t xf_ = v_convert_bf16_to_f32_all_b(xb_); \
    float64_pair_t yf_; \
    yf_.v1 = exp_biased(xf_.v1, kl, nb); \
    yf_.v2 = exp_biased(xf_.v2, kl, nb); \
    v_bf16_st_tnsr(c_, probs, v_convert_f32_to_bf16_all_b(yf_)); \
    (ACC) += yf_.v1 + yf_.v2; \
} while (0)

void main(tensor scores, tensor bound, tensor probs, tensor inv_sum, float pad_count)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int K = get_dim_size(scores, 0);
    const int bulk = K & ~511;
    const float kl = 1.4426950408889634f;  // log2(e)

    for (int r = start[0]; r < end[0]; ++r) {
        const int5 bc = {0, r, 0, 0, 0};
        // Broadcast single BF16 bound[0, r] to all 64 FP32 lanes
        const float64 b_f = v_broadcast_element_f32(
            v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(bc, bound)).v1, 0);
        const float64 nb = 127.0f - b_f * kl;

        float64_pair_t lo_p, hi_p;
        lo_p.v1 = b_f - 80.0f; lo_p.v2 = lo_p.v1;
        hi_p.v1 = b_f + 30.0f; hi_p.v2 = hi_p.v1;
        const bfloat128 lo_b = v_convert_f32_to_bf16_all_b(lo_p);
        const bfloat128 hi_b = v_convert_f32_to_bf16_all_b(hi_p);

        float64 a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int col = 0; col < bulk; col += 512) {
            MF_STEP(col,       a0);
            MF_STEP(col + 128, a1);
            MF_STEP(col + 256, a2);
            MF_STEP(col + 384, a3);
        }
        for (int col = bulk; col < K; col += 128) {
            MF_STEP(col, a0);
        }

        float64 total = row_sum((a0 + a1) + (a2 + a3));
        if (pad_count > 0.0f) {
            const float64 neg_b = v_f32_max_b(-b_f, -80.0f);
            const float64 pad_exp = exp_biased(neg_b, kl, (float64)127.0f);
            total = total - pad_count * pad_exp;
        }
        const float64 inv = positive_reciprocal(v_f32_max_b(total, 1.0e-30f));
        v_f32_st_tnsr(bc, inv_sum, inv);
    }
}
