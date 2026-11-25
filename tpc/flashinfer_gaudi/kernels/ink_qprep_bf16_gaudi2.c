// SPDX-License-Identifier: Apache-2.0
// Inkling-2 query prep: per head of 128, q = bf16(clip(q_lin * scale)) with clip = x * min(thr * (1 / (|x| + 1e-6)), 1);
// full layers (tau input): q = bf16(f32(q) * tau[t])  (the log length scale).
// Tensors (dim 0 fastest): qside bf16 [QS, T] (the q-side projection row; q in columns 0 .. NH*128), scale bf16
// [128, NH], (TAU: tau f32 [T]), q bf16 [128, NH, T]. Index space [NH].
#include "norm_math_gaudi2.h"

static inline float64 recip_pos(float64 value) {
    float64 estimate = as_float64((int64)0x7ef311c3 - as_int64(value));
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    return estimate;
}

void main(tensor qside, tensor scale,
#ifdef TAU
          tensor tau,
#endif
          tensor q, float thr)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int T = get_dim_size(qside, 1);
    // four rows per iteration: their dependency chains (row norm, reciprocal) overlap
#ifdef TAU
#define QTAU(TV) do { \
            float64_pair_t y_ = v_convert_bf16_to_f32_all_b(qb_); \
            y_.v1 = y_.v1 * (TV); \
            y_.v2 = y_.v2 * (TV); \
            qb_ = v_convert_f32_to_bf16_all_b(y_); \
        } while (0)
#else
#define QTAU(TV) do { (void)(TV); } while (0)
#endif
#define QROW(T_, TV) do { \
            int5 ic_ = {h * 128, (T_), 0, 0, 0}; \
            float64_pair_t x_ = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ic_, qside)); \
            x_.v1 = x_.v1 * sc.v1; \
            x_.v2 = x_.v2 * sc.v2; \
            const float64 ss_ = row_sum(x_.v1 * x_.v1 + x_.v2 * x_.v2); \
            const float64 norm_ = ss_ * positive_rsqrt(ss_); \
            const float64 f_ = v_f32_min_b(thr * recip_pos(norm_ + 1.0e-6f), 1.0f); \
            x_.v1 = x_.v1 * f_; \
            x_.v2 = x_.v2 * f_; \
            bfloat128 qb_ = v_convert_f32_to_bf16_all_b(x_); \
            QTAU(TV); \
            int5 oc_ = {0, h, (T_), 0, 0}; \
            v_bf16_st_tnsr(oc_, q, qb_); \
        } while (0)
    for (int h = start[0]; h < end[0]; ++h) {
        int5 scc = {0, h, 0, 0, 0};
        const float64_pair_t sc = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(scc, scale));
        int t = 0;
        for (; t + 4 <= T; t += 4) {
#ifdef TAU
            int5 tc = {t, 0, 0, 0, 0};
            const float64 tv = v_f32_ld_tnsr_b(tc, tau);  // tau[t .. t + 63]
            const float64 tv0 = v_broadcast_element_f32(tv, 0), tv1 = v_broadcast_element_f32(tv, 1);
            const float64 tv2 = v_broadcast_element_f32(tv, 2), tv3 = v_broadcast_element_f32(tv, 3);
#else
            const float tv0 = 0, tv1 = 0, tv2 = 0, tv3 = 0;
#endif
            QROW(t, tv0);
            QROW(t + 1, tv1);
            QROW(t + 2, tv2);
            QROW(t + 3, tv3);
        }
        for (; t < T; ++t) {
#ifdef TAU
            int5 tc = {t, 0, 0, 0, 0};
            const float64 tvr = v_broadcast_element_f32(v_f32_ld_tnsr_b(tc, tau), 0);
#else
            const float tvr = 0;
#endif
            QROW(t, tvr);
        }
    }
}
