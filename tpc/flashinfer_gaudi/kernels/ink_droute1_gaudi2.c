// SPDX-License-Identifier: Apache-2.0
// Compact Inkling-2 decode / verify router (128 experts, top 8), I-cache friendly (< 3 KiB .text vs 23 KiB ink_route).
//   s = sigmoid(logits); pick 8 largest s + bias (ties: lower expert); w = s[pick] / (sum + 1e-20) * route_scale
// Tensors (dim 0 fastest): logits f32 [128, T], bias f32 [128], [BMETA: bmeta i32 [8, T],] idx i32 [8, T], w f32 [8, T].
// Index space [T]. Params: route_scale.
// BMETA (batched decode): skips dummy rows (bmeta[3, t] == 0) immediately, writing idx = -1, w = 0.0f.
#include "ink_math_gaudi2.h"

static inline float64 rt_max64(float64 x, uchar256 c1, uchar256 c2, uchar256 c4) {
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_max_b(x, v_f32_mov_group_b(x, 0xffffffff, 63, 0));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, c1, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, c2, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, c4, 0, x));
    return x;
}
static inline float64 rt_min64(float64 x, uchar256 c1, uchar256 c2, uchar256 c4) {
    x = v_f32_min_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_min_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_min_b(x, v_f32_mov_group_b(x, 0xffffffff, 63, 0));
    x = v_f32_min_b(x, v_f32_shuffle_b(x, c1, 0, x));
    x = v_f32_min_b(x, v_f32_shuffle_b(x, c2, 0, x));
    x = v_f32_min_b(x, v_f32_shuffle_b(x, c4, 0, x));
    return x;
}
static inline float64 rt_sum64(float64 x, uchar256 c1, uchar256 c2, uchar256 c4) {
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0);
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0);
    x += v_f32_mov_group_b(x, 0xffffffff, 63, 0);
    x += v_f32_shuffle_b(x, c1, 0, x);
    x += v_f32_shuffle_b(x, c2, 0, x);
    x += v_f32_shuffle_b(x, c4, 0, x);
    return x;
}

void main(tensor logits, tensor bias,
#ifdef BMETA
          tensor bmeta,
#endif
          tensor idx, tensor w, float route_scale)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const uint64 ulanes = read_lane_id_4b_b();
    const uint64 l = ulanes & 7;
    const uint64 u1 = ((l ^ 1) | 0x80) * 0x01010101;
    const uint64 u2 = ((l ^ 2) | 0x80) * 0x01010101;
    const uint64 u4 = ((l ^ 4) | 0x80) * 0x01010101;
    const uchar256 c1 = *((uchar256*)&u1);
    const uchar256 c2 = *((uchar256*)&u2);
    const uchar256 c4 = *((uchar256*)&u4);

    const float64 i0 = v_convert_i32_to_f32_b((int64)ulanes, 0);
    const float64 i1 = i0 + 64.0f;
    int5 bc = {0, 0, 0, 0, 0};
    const float64 b0 = v_f32_ld_tnsr_b(bc, bias);
    bc[0] = 64;
    const float64 b1 = v_f32_ld_tnsr_b(bc, bias);

    for (int t = start[0]; t < end[0]; ++t) {
#ifdef BMETA
        int5 mc = {3, t, 0, 0, 0};
        if (s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta)) == 0) {
            int5 oc = {0, t, 0, 0, 0};
            v_f32_st_tnsr_partial(oc, w, (float64)0.0f, 7, 0);
            v_i32_st_tnsr_partial(oc, idx, (int64)-1, 7, 0);
            continue;
        }
#endif
        int5 lc = {0, t, 0, 0, 0};
        const float64 l0 = v_f32_ld_tnsr_b(lc, logits);
        lc[0] = 64;
        const float64 l1 = v_f32_ld_tnsr_b(lc, logits);
        const float64 s0 = ink_sigmoid(l0), s1 = ink_sigmoid(l1);
        float64 v0 = s0 + b0, v1 = s1 + b1;
        float64 out_e = 0.0f, out_w = 0.0f;
#pragma loop_unroll(1)
        for (int k = 0; k < 8; ++k) {
            const float64 m = rt_max64(v_f32_max_b(v0, v1), c1, c2, c4);
            const float64 c0v = v_f32_sel_eq_f32_b(v0, m, i0, 1.0e9f);
            const float64 c1v = v_f32_sel_eq_f32_b(v1, m, i1, 1.0e9f);
            const float64 e = rt_min64(v_f32_min_b(c0v, c1v), c1, c2, c4);
            const float64 h0 = v_f32_sel_eq_f32_b(i0, e, s0, 0.0f);
            const float64 h1 = v_f32_sel_eq_f32_b(i1, e, s1, 0.0f);
            const float64 sv = rt_sum64(h0 + h1, c1, c2, c4);
            out_e = v_f32_sel_eq_f32_b(i0, (float)k, e, out_e);
            out_w = v_f32_sel_eq_f32_b(i0, (float)k, sv, out_w);
            v0 = v_f32_sel_eq_f32_b(i0, e, -1.0e30f, v0);
            v1 = v_f32_sel_eq_f32_b(i1, e, -1.0e30f, v1);
        }
        const float64 wsum = rt_sum64(v_f32_sel_less_f32_b(i0, 8.0f, out_w, 0.0f), c1, c2, c4) + 1.0e-20f;
        const float64 wn = out_w * (route_scale * ink_recip(wsum));
        int5 oc = {0, t, 0, 0, 0};
        v_f32_st_tnsr_partial(oc, w, wn, 7, 0);
        v_i32_st_tnsr_partial(oc, idx, v_convert_f32_to_i32_b(out_e, SW_RHNE), 7, 0);
    }
}
