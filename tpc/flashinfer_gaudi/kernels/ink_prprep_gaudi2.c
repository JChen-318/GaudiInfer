// SPDX-License-Identifier: Apache-2.0
// Inkling-2 prefill relative-bias query prep (Direct engine), from the q-side projection rows:
//   r[t, h, :] = bf16(bf16(bf16(clip(rel[t, h, :] * r_scale[h, :])) [* tau[t]]) * isc)
// clip(x) = x * min(thr * (1 / (|x| + 1e-6)), 1) per head of 16 (l2_clip), isc = 1 / softmax scale (the bias table
// GEMM then gives bias / scale). The 384 rel values of a row are three 128-column chunks of 8 heads: after the BF16
// -> F32 conversion lane l of v1 / v2 holds column 2l / 2l + 1, so each head is an 8-lane group of both halves.
// Tensors (dim 0 fastest): row bf16 [N, T] (rel at column off_rel), rs f32 [128, 3] (r_scale per 128 columns
// [even 64 | odd 64]), (TAU: tau f32 [T]), r bf16 [384, T]. Index space [T]. Params: thr, isc, off_rel.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

// the sum over each 8-lane block (one head), in every lane of the block: masked full-vector sums (row_sum), eight
// per vector (the shuffle patterns' lane groups do not match 8-lane blocks)
static inline float64 sum8(float64 x) {
    const float64 lane = v_convert_i32_to_f32_b(as_int64(read_lane_id_4b_b()), 0);
    float64 out = 0;
#define S8(J) do { \
        const float64 m_ = v_f32_sel_less_f32_b(lane, 8.0f * (J) + 8.0f, v_f32_sel_less_f32_b(lane, 8.0f * (J), 0.0f, x), 0.0f); \
        const float64 s_ = row_sum(m_); \
        out = v_f32_sel_less_f32_b(lane, 8.0f * (J) + 8.0f, v_f32_sel_less_f32_b(lane, 8.0f * (J), out, s_), out); \
    } while (0)
    S8(0); S8(1); S8(2); S8(3); S8(4); S8(5); S8(6); S8(7);
#undef S8
    return out;
}

void main(tensor row, tensor rs,
#ifdef TAU
          tensor tau,
#endif
          tensor r, float thr, float isc, int off_rel)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    float64 se[3], so[3];
    for (int k = 0; k < 3; ++k) {
        int5 sc = {0, k, 0, 0, 0};
        se[k] = v_f32_ld_tnsr_b(sc, rs);
        sc[0] = 64;
        so[k] = v_f32_ld_tnsr_b(sc, rs);
    }
#define RCHUNK(K, TV) do { \
        int5 ic_ = {off_rel + (K) * 128, t, 0, 0, 0}; \
        const float64_pair_t x_ = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ic_, row)); \
        const float64 xe_ = x_.v1 * se[K], xo_ = x_.v2 * so[K]; \
        const float64 n2_ = sum8(xe_ * xe_ + xo_ * xo_); \
        const float64 n_ = v_f32_sel_grt_f32_b(n2_, 0.0f, n2_ * positive_rsqrt(n2_), 0.0f); \
        const float64 f_ = v_f32_min_b(thr * ink_recip(n_ + 1.0e-6f), 1.0f); \
        float64_pair_t p_; \
        p_.v1 = xe_ * f_; \
        p_.v2 = xo_ * f_; \
        float64_pair_t b_ = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(p_)); \
        TAUMUL(b_, TV); \
        b_.v1 = b_.v1 * isc; \
        b_.v2 = b_.v2 * isc; \
        int5 oc_ = {(K) * 128, t, 0, 0, 0}; \
        v_bf16_st_tnsr(oc_, r, v_convert_f32_to_bf16_all_b(b_)); \
    } while (0)
#ifdef TAU
#define TAUMUL(B, TV) do { \
        (B).v1 = (B).v1 * (TV); \
        (B).v2 = (B).v2 * (TV); \
        (B) = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(B)); \
    } while (0)
#else
#define TAUMUL(B, TV) do { (void)(TV); } while (0)
#endif
    for (int t = start[0]; t < end[0]; ++t) {
#ifdef TAU
        int5 tc = {t, 0, 0, 0, 0};
        const float64 tv = v_broadcast_element_f32(v_f32_ld_tnsr_b(tc, tau), 0);
#else
        const float tv = 1.0f;
#endif
        RCHUNK(0, tv);
        RCHUNK(1, tv);
        RCHUNK(2, tv);
    }
#undef RCHUNK
#undef TAUMUL
}
