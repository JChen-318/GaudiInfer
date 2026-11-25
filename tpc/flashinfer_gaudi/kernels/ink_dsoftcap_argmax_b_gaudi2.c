// SPDX-License-Identifier: Apache-2.0
// Inkling-2 batched Direct-engine fused logit softcap + 24-TPC argmax reduction for Bp rows (LUT-free [9/8] Pade):
//   out_lg[i, b] = cap * tanh(in_lg[i, b] * inv_cap)   for i in [0, V), b in [0, Bp)
//   max_val[0, p, b], max_idx[0, p, b] = argmax over TPC p's slice of row b (ties broken by lowest index).
// Tensors (dim 0 fastest): in_lg f32 [V, Bp], out_lg f32 [V, Bp] (may alias),
// max_val f32 [64, 24, Bp], max_idx i32 [64, 24, Bp].
// Index space: [24]. Params: cap, inv_cap, V, active_B, skip_softcap.
// Only rows in [0, active_B) (clamped to Bp when active_B <= 0 or active_B > Bp) are processed.
// When skip_softcap != 0 (or cap <= 0): argmax runs directly on raw in_lg (monotonic tanh preserves argmax
// bit-for-bit) and skips [9/8] Pade tanh + out_lg HBM stores.
#include "ink_math_gaudi2.h"

#define TANH_PADE_4(X0, X1, X2, X3, Y0, Y1, Y2, Y3) do { \
        const float64 c0_ = v_f32_min_b(v_f32_max_b((X0), -6.5f), 6.5f); \
        const float64 c1_ = v_f32_min_b(v_f32_max_b((X1), -6.5f), 6.5f); \
        const float64 c2_ = v_f32_min_b(v_f32_max_b((X2), -6.5f), 6.5f); \
        const float64 c3_ = v_f32_min_b(v_f32_max_b((X3), -6.5f), 6.5f); \
        const float64 q0_ = c0_ * c0_, q1_ = c1_ * c1_, q2_ = c2_ * c2_, q3_ = c3_ * c3_; \
        float64 n0_ = v_f32_mac_b(q0_, 2.901963686e-8f, 2.872944049e-5f); \
        float64 n1_ = v_f32_mac_b(q1_, 2.901963686e-8f, 2.872944049e-5f); \
        float64 n2_ = v_f32_mac_b(q2_, 2.901963686e-8f, 2.872944049e-5f); \
        float64 n3_ = v_f32_mac_b(q3_, 2.901963686e-8f, 2.872944049e-5f); \
        float64 d0_ = v_f32_mac_b(q0_, 1.305883659e-6f, 4.022121669e-4f); \
        float64 d1_ = v_f32_mac_b(q1_, 1.305883659e-6f, 4.022121669e-4f); \
        float64 d2_ = v_f32_mac_b(q2_, 1.305883659e-6f, 4.022121669e-4f); \
        float64 d3_ = v_f32_mac_b(q3_, 1.305883659e-6f, 4.022121669e-4f); \
        n0_ = v_f32_mac_b(n0_, q0_, 3.921568627e-3f); n1_ = v_f32_mac_b(n1_, q1_, 3.921568627e-3f); \
        n2_ = v_f32_mac_b(n2_, q2_, 3.921568627e-3f); n3_ = v_f32_mac_b(n3_, q3_, 3.921568627e-3f); \
        d0_ = v_f32_mac_b(d0_, q0_, 2.745098039e-2f); d1_ = v_f32_mac_b(d1_, q1_, 2.745098039e-2f); \
        d2_ = v_f32_mac_b(d2_, q2_, 2.745098039e-2f); d3_ = v_f32_mac_b(d3_, q3_, 2.745098039e-2f); \
        n0_ = v_f32_mac_b(n0_, q0_, 1.372549020e-1f); n1_ = v_f32_mac_b(n1_, q1_, 1.372549020e-1f); \
        n2_ = v_f32_mac_b(n2_, q2_, 1.372549020e-1f); n3_ = v_f32_mac_b(n3_, q3_, 1.372549020e-1f); \
        d0_ = v_f32_mac_b(d0_, q0_, 4.705882353e-1f); d1_ = v_f32_mac_b(d1_, q1_, 4.705882353e-1f); \
        d2_ = v_f32_mac_b(d2_, q2_, 4.705882353e-1f); d3_ = v_f32_mac_b(d3_, q3_, 4.705882353e-1f); \
        n0_ = v_f32_mac_b(n0_, q0_, 1.0f); n1_ = v_f32_mac_b(n1_, q1_, 1.0f); \
        n2_ = v_f32_mac_b(n2_, q2_, 1.0f); n3_ = v_f32_mac_b(n3_, q3_, 1.0f); \
        d0_ = v_f32_mac_b(d0_, q0_, 1.0f); d1_ = v_f32_mac_b(d1_, q1_, 1.0f); \
        d2_ = v_f32_mac_b(d2_, q2_, 1.0f); d3_ = v_f32_mac_b(d3_, q3_, 1.0f); \
        float64 r0_ = as_float64((int64)0x7ef311c3 - as_int64(d0_)); \
        float64 r1_ = as_float64((int64)0x7ef311c3 - as_int64(d1_)); \
        float64 r2_ = as_float64((int64)0x7ef311c3 - as_int64(d2_)); \
        float64 r3_ = as_float64((int64)0x7ef311c3 - as_int64(d3_)); \
        n0_ = n0_ * c0_; n1_ = n1_ * c1_; n2_ = n2_ * c2_; n3_ = n3_ * c3_; \
        r0_ = r0_ * (2.0f - d0_ * r0_); r1_ = r1_ * (2.0f - d1_ * r1_); \
        r2_ = r2_ * (2.0f - d2_ * r2_); r3_ = r3_ * (2.0f - d3_ * r3_); \
        r0_ = r0_ * (2.0f - d0_ * r0_); r1_ = r1_ * (2.0f - d1_ * r1_); \
        r2_ = r2_ * (2.0f - d2_ * r2_); r3_ = r3_ * (2.0f - d3_ * r3_); \
        r0_ = r0_ * (2.0f - d0_ * r0_); r1_ = r1_ * (2.0f - d1_ * r1_); \
        r2_ = r2_ * (2.0f - d2_ * r2_); r3_ = r3_ * (2.0f - d3_ * r3_); \
        (Y0) = v_f32_min_b(v_f32_max_b(n0_ * r0_, -1.0f), 1.0f); \
        (Y1) = v_f32_min_b(v_f32_max_b(n1_ * r1_, -1.0f), 1.0f); \
        (Y2) = v_f32_min_b(v_f32_max_b(n2_ * r2_, -1.0f), 1.0f); \
        (Y3) = v_f32_min_b(v_f32_max_b(n3_ * r3_, -1.0f), 1.0f); \
    } while (0)

static inline float64 vec_max_f32(float64 v) {
    v = v_f32_max_b(v, v_f32_mov_dual_group_all_b(v, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    v = v_f32_max_b(v, v_f32_mov_dual_group_all_b(v, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    v = v_f32_max_b(v, v_f32_mov_group_b(v, 0xffffffff, 63, 0));
    float64 res = -1.0e30f;
#pragma loop_unroll(8)
    for (int lane = 0; lane < 8; ++lane) {
        const uchar256 bc = 0x80 | lane;
        res = v_f32_max_b(res, v_f32_shuffle_b(v, bc, 0, v));
    }
    return res;
}

static inline int64 vec_min_i32(int64 v) {
    v = v_i32_min_b(v, v_i32_mov_dual_group_all_b(v, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    v = v_i32_min_b(v, v_i32_mov_dual_group_all_b(v, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    v = v_i32_min_b(v, v_i32_mov_group_b(v, 0xffffffff, 63, 0));
    int64 res = 0x7fffffff;
#pragma loop_unroll(8)
    for (int lane = 0; lane < 8; ++lane) {
        const uchar256 bc = 0x80 | lane;
        res = v_i32_min_b(res, v_i32_shuffle_b(v, bc, 0, v));
    }
    return res;
}

void main(tensor in_lg, tensor out_lg, tensor max_val, tensor max_idx,
          float cap, float inv_cap, int V, int active_B, int skip_softcap)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int Bp = get_dim_size(in_lg, 1);
    const int rows = (active_B > 0 && active_B < Bp) ? active_B : Bp;
    const int nch = (V + 63) >> 6;
    const int64 lane = (int64)read_lane_id_4b_b();

    for (int p = start[0]; p < end[0]; ++p) {
        const int ch0 = (nch * p) / 24;
        const int ch1 = (nch * (p + 1)) / 24;
        const int fast_end = ch1 - 4;
        for (int row = 0; row < rows; ++row) {
            float64 bv = -1.0e30f;
            int64 bi = 0x7fffffff;
            int ch = ch0;
            if (skip_softcap || cap <= 0.0f) {
                for (; ch <= fast_end && ((ch + 4) << 6) <= V; ch += 4) {
                    const int b = ch << 6;
                    int5 c0 = {b, row, 0, 0, 0};
                    int5 c1 = {b + 64, row, 0, 0, 0};
                    int5 c2 = {b + 128, row, 0, 0, 0};
                    int5 c3 = {b + 192, row, 0, 0, 0};
                    const float64 y0 = v_f32_ld_tnsr_b(c0, in_lg);
                    const float64 y1 = v_f32_ld_tnsr_b(c1, in_lg);
                    const float64 y2 = v_f32_ld_tnsr_b(c2, in_lg);
                    const float64 y3 = v_f32_ld_tnsr_b(c3, in_lg);
                    const int64 i0 = lane + b, i1 = lane + (b + 64), i2 = lane + (b + 128), i3 = lane + (b + 192);
                    const float64 ov0 = bv; bv = v_f32_sel_grt_f32_b(y0, ov0, y0, ov0); bi = v_i32_sel_grt_f32_b(y0, ov0, i0, bi);
                    const float64 ov1 = bv; bv = v_f32_sel_grt_f32_b(y1, ov1, y1, ov1); bi = v_i32_sel_grt_f32_b(y1, ov1, i1, bi);
                    const float64 ov2 = bv; bv = v_f32_sel_grt_f32_b(y2, ov2, y2, ov2); bi = v_i32_sel_grt_f32_b(y2, ov2, i2, bi);
                    const float64 ov3 = bv; bv = v_f32_sel_grt_f32_b(y3, ov3, y3, ov3); bi = v_i32_sel_grt_f32_b(y3, ov3, i3, bi);
                }
                for (; ch < ch1; ++ch) {
                    const int b = ch << 6;
                    int5 c0 = {b, row, 0, 0, 0};
                    const float64 y0 = v_f32_ld_tnsr_b(c0, in_lg);
                    const int64 idx = lane + b;
                    const float64 ym = v_f32_sel_less_i32_b(idx, V, y0, -1.0e30f);
                    const float64 ov = bv;
                    bv = v_f32_sel_grt_f32_b(ym, ov, ym, ov);
                    bi = v_i32_sel_grt_f32_b(ym, ov, idx, bi);
                }
            } else {
                for (; ch <= fast_end && ((ch + 4) << 6) <= V; ch += 4) {
                    const int b = ch << 6;
                    int5 c0 = {b, row, 0, 0, 0};
                    int5 c1 = {b + 64, row, 0, 0, 0};
                    int5 c2 = {b + 128, row, 0, 0, 0};
                    int5 c3 = {b + 192, row, 0, 0, 0};
                    const float64 x0 = v_f32_ld_tnsr_b(c0, in_lg) * inv_cap;
                    const float64 x1 = v_f32_ld_tnsr_b(c1, in_lg) * inv_cap;
                    const float64 x2 = v_f32_ld_tnsr_b(c2, in_lg) * inv_cap;
                    const float64 x3 = v_f32_ld_tnsr_b(c3, in_lg) * inv_cap;
                    float64 t0, t1, t2, t3;
                    TANH_PADE_4(x0, x1, x2, x3, t0, t1, t2, t3);
                    const float64 y0 = t0 * cap, y1 = t1 * cap, y2 = t2 * cap, y3 = t3 * cap;
                    v_f32_st_tnsr(c0, out_lg, y0);
                    v_f32_st_tnsr(c1, out_lg, y1);
                    v_f32_st_tnsr(c2, out_lg, y2);
                    v_f32_st_tnsr(c3, out_lg, y3);
                    const int64 i0 = lane + b, i1 = lane + (b + 64), i2 = lane + (b + 128), i3 = lane + (b + 192);
                    const float64 ov0 = bv; bv = v_f32_sel_grt_f32_b(y0, ov0, y0, ov0); bi = v_i32_sel_grt_f32_b(y0, ov0, i0, bi);
                    const float64 ov1 = bv; bv = v_f32_sel_grt_f32_b(y1, ov1, y1, ov1); bi = v_i32_sel_grt_f32_b(y1, ov1, i1, bi);
                    const float64 ov2 = bv; bv = v_f32_sel_grt_f32_b(y2, ov2, y2, ov2); bi = v_i32_sel_grt_f32_b(y2, ov2, i2, bi);
                    const float64 ov3 = bv; bv = v_f32_sel_grt_f32_b(y3, ov3, y3, ov3); bi = v_i32_sel_grt_f32_b(y3, ov3, i3, bi);
                }
                for (; ch < ch1; ++ch) {
                    const int b = ch << 6;
                    int5 c0 = {b, row, 0, 0, 0};
                    const float64 x0 = v_f32_ld_tnsr_b(c0, in_lg) * inv_cap;
                    float64 t0, t1, t2, t3;
                    TANH_PADE_4(x0, x0, x0, x0, t0, t1, t2, t3);
                    const float64 y0 = t0 * cap;
                    v_f32_st_tnsr(c0, out_lg, y0);
                    const int64 idx = lane + b;
                    const float64 ym = v_f32_sel_less_i32_b(idx, V, y0, -1.0e30f);
                    const float64 ov = bv;
                    bv = v_f32_sel_grt_f32_b(ym, ov, ym, ov);
                    bi = v_i32_sel_grt_f32_b(ym, ov, idx, bi);
                }
            }
            const float64 mv = vec_max_f32(bv);
            const int64 ci = v_i32_sel_eq_f32_b(bv, mv, bi, 0x7fffffff);
            const int64 mi = vec_min_i32(ci);
            int5 oc = {0, p, row, 0, 0};
            v_f32_st_tnsr(oc, max_val, mv);
            v_i32_st_tnsr(oc, max_idx, mi);
        }
    }
}
