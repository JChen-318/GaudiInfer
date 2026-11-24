// SPDX-License-Identifier: Apache-2.0
// Inkling-2 message gates (passes 2 and 3 of the loop), around the two small GEMMs:
//  CAT (default): cat[g, t] = [bf16(rms(h_g[t])), bf16(rms(m_g[t]))]                       (unweighted RMSNorm, eps)
//  OUT:           y = (1 - tanh(f)) * h + tanh(mg) * m with [f | mg] = fg[g, t];
//                 out[g, t] = bf16((y * rsqrt(mean(y^2) + eps)) * w_g)                   (the next RMSNorm)
// F32 inputs are in the interleaved order of the TPC kernels (every 128 values: [even 64 | odd 64]); h may hold one
// state for every gate (Gh = 1) or one per gate. m (the messages [T, 4, H]) is read at gate m0 + g.
// Tensors (dim 0 fastest):
//  CAT: h f32 [H, Gh, T], m f32 [H, 4, T] -> cat bf16 [2H, T, G]
//  OUT: h f32 [H, Gh, T], m f32 [H, 4, T], fg f32 [2H, T, G], w bf16 [H, G] -> out bf16 [H, T, G]
// Index space [G, T]. Params: eps, m0 (int).
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

static inline float64_pair_t ld_pair(tensor x, int col, int a, int b) {
    float64_pair_t p;
    int5 c = {col, a, b, 0, 0};
    p.v1 = v_f32_ld_tnsr_b(c, x);
    c[0] = col + 64;
    p.v2 = v_f32_ld_tnsr_b(c, x);
    return p;
}
#ifdef OUT
#ifdef MG_LUT
static inline float64 tanh_vec(float64 x) { return v_tanh_f32(x); }  // lookup table (Synapse launches)
#elif !defined(MG_PADE)
static inline float64 tanh_vec(float64 x) { return ink_tanh(x); }  // LUT-free (Direct engine)
#endif
__local__ float64 yv[2 * 32];  // one row's y (H <= 4096)
#endif

#ifndef OUT
void main(tensor h, tensor m, tensor cat, float eps, int m0)
#else
void main(tensor h, tensor m, tensor fg, tensor w, tensor out, float eps, int m0)
#endif
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(h, 0);
    const int Gh = get_dim_size(h, 1);
    const int tiles = H / 128;
    const float inv_h = ink_srecip((float)H);
    for (int t = start[1]; t < end[1]; ++t) {
        for (int g = start[0]; g < end[0]; ++g) {
            const int gh = Gh == 1 ? 0 : g;
            const int gm = m0 + g;
#ifndef OUT
            float64 sh = 0, sm = 0;
            for (int i = 0; i < tiles; ++i) {
                const float64_pair_t a = ld_pair(h, i * 128, gh, t);
                const float64_pair_t b = ld_pair(m, i * 128, gm, t);
                sh = v_f32_mac_b(a.v1, a.v1, sh);
                sh = v_f32_mac_b(a.v2, a.v2, sh);
                sm = v_f32_mac_b(b.v1, b.v1, sm);
                sm = v_f32_mac_b(b.v2, b.v2, sm);
            }
            const float64 rh = positive_rsqrt(row_sum(sh) * inv_h + eps);
            const float64 rm = positive_rsqrt(row_sum(sm) * inv_h + eps);
            for (int i = 0; i < tiles; ++i) {
                float64_pair_t a = ld_pair(h, i * 128, gh, t);
                float64_pair_t b = ld_pair(m, i * 128, gm, t);
                a.v1 = a.v1 * rh; a.v2 = a.v2 * rh;
                b.v1 = b.v1 * rm; b.v2 = b.v2 * rm;
                int5 oc = {i * 128, t, g, 0, 0};
                v_bf16_st_tnsr(oc, cat, v_convert_f32_to_bf16_all_b(a));
                oc[0] = H + i * 128;
                v_bf16_st_tnsr(oc, cat, v_convert_f32_to_bf16_all_b(b));
            }
#else
#ifdef MG_PADE
// 4-way lockstep normalized [9/8] Pade tanh, |x| <= 6.5, three Newton steps of the reciprocal (max relative error
// 7e-6, mean 8e-7; the [7/6] form before had 1e-4 / 1e-5)
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
#endif
            // y once (four tanh per pair) into local memory, then the norm
            float64 sy1 = 0, sy2 = 0;
            for (int i = 0; i < tiles; ++i) {
                const float64_pair_t a = ld_pair(h, i * 128, gh, t);
                const float64_pair_t b = ld_pair(m, i * 128, gm, t);
                const float64_pair_t f = ld_pair(fg, i * 128, t, g);
                const float64_pair_t q = ld_pair(fg, H + i * 128, t, g);
#ifdef MG_PADE
                float64 tf1, tq1, tf2, tq2;
                TANH_PADE_4(f.v1, q.v1, f.v2, q.v2, tf1, tq1, tf2, tq2);
                const float64 y1 = (1.0f - tf1) * a.v1 + tq1 * b.v1;
                const float64 y2 = (1.0f - tf2) * a.v2 + tq2 * b.v2;
#else
                const float64 y1 = (1.0f - tanh_vec(f.v1)) * a.v1 + tanh_vec(q.v1) * b.v1;
                const float64 y2 = (1.0f - tanh_vec(f.v2)) * a.v2 + tanh_vec(q.v2) * b.v2;
#endif
                yv[2 * i] = y1;
                yv[2 * i + 1] = y2;
                sy1 = v_f32_mac_b(y1, y1, sy1);
                sy2 = v_f32_mac_b(y2, y2, sy2);
            }
            const float64 ry = positive_rsqrt(row_sum(sy1 + sy2) * inv_h + eps);
            for (int i = 0; i < tiles; ++i) {
                int5 wc = {i * 128, g, 0, 0, 0};
                const float64_pair_t wv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
                float64_pair_t y;
                y.v1 = (yv[2 * i] * ry) * wv.v1;
                y.v2 = (yv[2 * i + 1] * ry) * wv.v2;
                int5 oc = {i * 128, t, g, 0, 0};
                v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(y));
            }
#endif
        }
    }
}
