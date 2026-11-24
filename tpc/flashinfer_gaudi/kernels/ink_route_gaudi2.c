// SPDX-License-Identifier: Apache-2.0
// Inkling-2 router (128 experts, top 8), one token per index point:
//   s = sigmoid(logits); pick the 8 largest s + bias (ties: the lower expert); w = s[pick] / (sum + 1e-20) * route_scale
// Tensors (dim 0 fastest): logits f32 [128, T], bias f32 [128], iota f32 [128] (0..127), idx i32 [8, T], w f32 [8, T].
// Index space [T]. Params: route_scale.
static inline float64 row_max64(float64 v) {
    v = v_f32_max_b(v, v_f32_mov_dual_group_all_b(v, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    v = v_f32_max_b(v, v_f32_mov_dual_group_all_b(v, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    v = v_f32_max_b(v, v_f32_mov_group_b(v, 0xffffffff, 63, 0));
    float64 r = v_f32_shuffle_b(v, (uchar256)0x80, 0, v);
    #pragma loop_unroll(7)
    for (int lane = 1; lane < 8; ++lane) r = v_f32_max_b(r, v_f32_shuffle_b(v, (uchar256)(0x80 | lane), 0, v));
    return r;
}
static inline float64 row_min64(float64 v) {
    v = v_f32_min_b(v, v_f32_mov_dual_group_all_b(v, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    v = v_f32_min_b(v, v_f32_mov_dual_group_all_b(v, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    v = v_f32_min_b(v, v_f32_mov_group_b(v, 0xffffffff, 63, 0));
    float64 r = v_f32_shuffle_b(v, (uchar256)0x80, 0, v);
    #pragma loop_unroll(7)
    for (int lane = 1; lane < 8; ++lane) r = v_f32_min_b(r, v_f32_shuffle_b(v, (uchar256)(0x80 | lane), 0, v));
    return r;
}
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

void main(tensor logits, tensor bias, tensor iota, tensor idx, tensor w, float route_scale)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    int5 c0 = {0, 0, 0, 0, 0}, c1 = {64, 0, 0, 0, 0};
    const float64 b0 = v_f32_ld_tnsr_b(c0, bias), b1 = v_f32_ld_tnsr_b(c1, bias);
    const float64 i0 = v_f32_ld_tnsr_b(c0, iota), i1 = v_f32_ld_tnsr_b(c1, iota);
    // two tokens per iteration: their pick chains (row max / min reductions, 8 picks) overlap
#define RT_PICK(V0, V1, S0, S1, OE, OW, K) do { \
        const float64 m_ = v_f32_max_b(row_max64(V0), row_max64(V1)); \
        const float64 c0_ = v_f32_sel_eq_f32_b(V0, m_, i0, 1.0e9f); \
        const float64 c1_ = v_f32_sel_eq_f32_b(V1, m_, i1, 1.0e9f); \
        const float64 e_ = v_f32_min_b(row_min64(c0_), row_min64(c1_)); \
        const float64 h0_ = v_f32_sel_eq_f32_b(i0, e_, S0, 0.0f); \
        const float64 h1_ = v_f32_sel_eq_f32_b(i1, e_, S1, 0.0f); \
        const float64 sv_ = row_sum(h0_ + h1_); \
        OE = v_f32_sel_eq_f32_b(i0, (float)(K), e_, OE); \
        OW = v_f32_sel_eq_f32_b(i0, (float)(K), sv_, OW); \
        V0 = v_f32_sel_eq_f32_b(i0, e_, -1.0e30f, V0); \
        V1 = v_f32_sel_eq_f32_b(i1, e_, -1.0e30f, V1); \
    } while (0)
#define RT_OUT(T_, OE, OW) do { \
        const float64 wsum_ = row_sum(v_f32_sel_less_f32_b(i0, 8.0f, OW, 0.0f)) + 1.0e-20f; \
        const float64 wn_ = OW * (route_scale * ink_recip(wsum_)); \
        int5 oc_ = {0, (T_), 0, 0, 0}; \
        v_f32_st_tnsr_partial(oc_, w, wn_, 7, 0); \
        v_i32_st_tnsr_partial(oc_, idx, v_convert_f32_to_i32_b(OE, SW_RHNE), 7, 0); \
    } while (0)
    int t2 = start[0];
    for (; t2 + 2 <= end[0]; t2 += 2) {
        int5 lc = {0, t2, 0, 0, 0};
        const float64 la0 = v_f32_ld_tnsr_b(lc, logits);
        lc[0] = 64;
        const float64 la1 = v_f32_ld_tnsr_b(lc, logits);
        lc[0] = 0; lc[1] = t2 + 1;
        const float64 lb0 = v_f32_ld_tnsr_b(lc, logits);
        lc[0] = 64;
        const float64 lb1 = v_f32_ld_tnsr_b(lc, logits);
        const float64 sa0 = ink_sigmoid(la0), sa1 = ink_sigmoid(la1);
        const float64 sb0 = ink_sigmoid(lb0), sb1 = ink_sigmoid(lb1);
        float64 va0 = sa0 + b0, va1 = sa1 + b1, vb0 = sb0 + b0, vb1 = sb1 + b1;
        float64 oea = 0, owa = 0, oeb = 0, owb = 0;
        for (int k = 0; k < 8; ++k) {
            RT_PICK(va0, va1, sa0, sa1, oea, owa, k);
            RT_PICK(vb0, vb1, sb0, sb1, oeb, owb, k);
        }
        RT_OUT(t2, oea, owa);
        RT_OUT(t2 + 1, oeb, owb);
    }
    for (int t = t2; t < end[0]; ++t) {
        int5 lc = {0, t, 0, 0, 0};
        const float64 l0 = v_f32_ld_tnsr_b(lc, logits);
        lc[0] = 64;
        const float64 l1 = v_f32_ld_tnsr_b(lc, logits);
        const float64 s0 = ink_sigmoid(l0), s1 = ink_sigmoid(l1);  // LUT-free (Direct engine)
        float64 v0 = s0 + b0, v1 = s1 + b1;
        float64 out_e = 0, out_w = 0;  // lane k: the k-th pick (expert id, score)
        for (int k = 0; k < 8; ++k) {
            const float64 m = v_f32_max_b(row_max64(v0), row_max64(v1));
            // the lowest expert holding the maximum
            const float64 c0v = v_f32_sel_eq_f32_b(v0, m, i0, 1.0e9f);
            const float64 c1v = v_f32_sel_eq_f32_b(v1, m, i1, 1.0e9f);
            const float64 e = v_f32_min_b(row_min64(c0v), row_min64(c1v));
            // its score (without the bias)
            const float64 h0 = v_f32_sel_eq_f32_b(i0, e, s0, 0.0f);
            const float64 h1 = v_f32_sel_eq_f32_b(i1, e, s1, 0.0f);
            const float64 sv = row_sum(h0 + h1);
            out_e = v_f32_sel_eq_f32_b(i0, (float)k, e, out_e);
            out_w = v_f32_sel_eq_f32_b(i0, (float)k, sv, out_w);
            v0 = v_f32_sel_eq_f32_b(i0, e, -1.0e30f, v0);
            v1 = v_f32_sel_eq_f32_b(i1, e, -1.0e30f, v1);
        }
        const float64 wsum = row_sum(v_f32_sel_less_f32_b(i0, 8.0f, out_w, 0.0f)) + 1.0e-20f;
        const float64 wn = out_w * (route_scale * ink_recip(wsum));
        int5 oc = {0, t, 0, 0, 0};
        v_f32_st_tnsr_partial(oc, w, wn, 7, 0);
        v_i32_st_tnsr_partial(oc, idx, v_convert_f32_to_i32_b(out_e, SW_RHNE), 7, 0);
    }
}
