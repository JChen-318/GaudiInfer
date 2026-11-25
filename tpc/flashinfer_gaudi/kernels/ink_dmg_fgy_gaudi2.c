// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode/verify/batch fused message-gate fg GEMV + [9/8] Pade tanh blend + sq partials.
// Replaces mgf3x/mgf1 MME + ink_dmgy TPC in ONE 18-core TPC stage (eliminates 48 MME stages + barriers per step).
// Reuses fg_t weights in registers across T rows (single HBM read of fg_t for T = 1..4 and T = 8!).
// Tensors (dim 0 fastest):
//   h        f32  [H, T, Gh]    (split order; Gh = 1 shares state across all G gates)
//   m        f32  [H, T, 4]     (split order)
//   gr       bf16 [R=64, T, G]  (also backward-compatible with legacy 2D [R=64, G] when T == 1)
//   fg_t     bf16 [2H, R=64, G] (natural order transposed: [0..H) forget, [H..2H) gain)
//   y        f32  [H, T, G]     (split order)
//   partials f32  [64, NT, T]   (lane g = tile partial sum of y_g^2; legacy [64, NT] when T == 1)
// Index space [H / 128, T] (split_dim = 0 -> NT = 18 TPCs). Params: m0.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

__local__ float64 fg_buf[8][4];   // per-row (f.v1, f.v2, g.v1, g.v2) for up to 8 rows
__local__ float64 sq_acc[8];      // per-row partial sum accumulator across gates g = 0..G-1

static inline float64_pair_t ld_pair(tensor x, int col, int r, int b) {
    float64_pair_t p;
    int5 c = {col, r, b, 0, 0};
    p.v1 = v_f32_ld_tnsr_b(c, x);
    c[0] = col + 64;
    p.v2 = v_f32_ld_tnsr_b(c, x);
    return p;
}

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

#define STORE_FG(M_IDX, FA, GA) do { \
        fg_buf[M_IDX][0] = (FA).v1; fg_buf[M_IDX][1] = (FA).v2; \
        fg_buf[M_IDX][2] = (GA).v1; fg_buf[M_IDX][3] = (GA).v2; \
    } while (0)

void main(tensor h, tensor m, tensor gr, tensor fg_t, tensor y, tensor partials, int m0)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(h, 0);
    const int T = get_dim_size(y, 1);
    const int Gh = get_dim_size(h, 2);
    const int G = get_dim_size(y, 2);
    const int R = get_dim_size(gr, 0);
    const int gr_2d = (T == 1 && get_dim_size(gr, 2) == 1 && G > 1);
    const float64 lanes = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), 0);

    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        int rb = start[1];
        while (rb < end[1]) {
            const int rem = end[1] - rb;
            const int nb = (rem >= 8) ? 8 : (rem >= 4 ? 4 : rem);
            for (int mi = 0; mi < nb; ++mi) sq_acc[mi] = 0.0f;

            for (int g = 0; g < G; ++g) {
                int5 wf = {c, 0, g, 0, 0};
                int5 wg = {H + c, 0, g, 0, 0};

                if (nb == 1) {
                    float128 fa0 = {0}, fa1 = {0}, ga0 = {0}, ga1 = {0};
                    int5 gc = gr_2d ? (int5){0, g, 0, 0, 0} : (int5){0, rb, g, 0, 0};
                    for (int k = 0; k < R; k += 4) {
                        gc[0] = k;     const bfloat128 r0 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[0] = k + 1; const bfloat128 r1 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[0] = k + 2; const bfloat128 r2 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[0] = k + 3; const bfloat128 r3 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        wf[1] = k;     wg[1] = k;
                        fa0 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS), r0, fa0, 0);
                        ga0 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS), r0, ga0, 0);
                        wf[1] = k + 1; wg[1] = k + 1;
                        fa1 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS), r1, fa1, 0);
                        ga1 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS), r1, ga1, 0);
                        wf[1] = k + 2; wg[1] = k + 2;
                        fa0 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS), r2, fa0, 0);
                        ga0 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS), r2, ga0, 0);
                        wf[1] = k + 3; wg[1] = k + 3;
                        fa1 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS), r3, fa1, 0);
                        ga1 = v_bf16_mac_acc32_b(v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS), r3, ga1, 0);
                    }
                    fg_buf[0][0] = fa0.v1 + fa1.v1; fg_buf[0][1] = fa0.v2 + fa1.v2;
                    fg_buf[0][2] = ga0.v1 + ga1.v1; fg_buf[0][3] = ga0.v2 + ga1.v2;
                } else if (nb == 2) {
                    float128 fa0 = {0}, ga0 = {0}, fa1 = {0}, ga1 = {0};
                    int5 gc0 = {0, rb,     g, 0, 0};
                    int5 gc1 = {0, rb + 1, g, 0, 0};
                    for (int k = 0; k < R; k += 2) {
                        gc0[0] = k;     const bfloat128 a0 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc0, gr));
                        gc1[0] = k;     const bfloat128 b0 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc1, gr));
                        gc0[0] = k + 1; const bfloat128 a1 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc0, gr));
                        gc1[0] = k + 1; const bfloat128 b1 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc1, gr));
                        wf[1] = k; wg[1] = k;
                        const bfloat128 wfv0 = v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS);
                        const bfloat128 wgv0 = v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS);
                        fa0 = v_bf16_mac_acc32_b(wfv0, a0, fa0, 0); ga0 = v_bf16_mac_acc32_b(wgv0, a0, ga0, 0);
                        fa1 = v_bf16_mac_acc32_b(wfv0, b0, fa1, 0); ga1 = v_bf16_mac_acc32_b(wgv0, b0, ga1, 0);
                        wf[1] = k + 1; wg[1] = k + 1;
                        const bfloat128 wfv1 = v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS);
                        const bfloat128 wgv1 = v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS);
                        fa0 = v_bf16_mac_acc32_b(wfv1, a1, fa0, 0); ga0 = v_bf16_mac_acc32_b(wgv1, a1, ga0, 0);
                        fa1 = v_bf16_mac_acc32_b(wfv1, b1, fa1, 0); ga1 = v_bf16_mac_acc32_b(wgv1, b1, ga1, 0);
                    }
                    STORE_FG(0, fa0, ga0);
                    STORE_FG(1, fa1, ga1);
                } else if (nb == 3) {
                    float128 fa0 = {0}, ga0 = {0}, fa1 = {0}, ga1 = {0}, fa2 = {0}, ga2 = {0};
                    int5 gc0 = {0, rb,     g, 0, 0};
                    int5 gc1 = {0, rb + 1, g, 0, 0};
                    int5 gc2 = {0, rb + 2, g, 0, 0};
                    for (int k = 0; k < R; ++k) {
                        gc0[0] = k; const bfloat128 r0 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc0, gr));
                        gc1[0] = k; const bfloat128 r1 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc1, gr));
                        gc2[0] = k; const bfloat128 r2 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc2, gr));
                        wf[1] = k; wg[1] = k;
                        const bfloat128 wfv = v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS);
                        const bfloat128 wgv = v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS);
                        fa0 = v_bf16_mac_acc32_b(wfv, r0, fa0, 0); ga0 = v_bf16_mac_acc32_b(wgv, r0, ga0, 0);
                        fa1 = v_bf16_mac_acc32_b(wfv, r1, fa1, 0); ga1 = v_bf16_mac_acc32_b(wgv, r1, ga1, 0);
                        fa2 = v_bf16_mac_acc32_b(wfv, r2, fa2, 0); ga2 = v_bf16_mac_acc32_b(wgv, r2, ga2, 0);
                    }
                    STORE_FG(0, fa0, ga0);
                    STORE_FG(1, fa1, ga1);
                    STORE_FG(2, fa2, ga2);
                } else if (nb == 4) {
                    float128 fa0 = {0}, ga0 = {0}, fa1 = {0}, ga1 = {0};
                    float128 fa2 = {0}, ga2 = {0}, fa3 = {0}, ga3 = {0};
                    int5 gc0 = {0, rb,     g, 0, 0};
                    int5 gc1 = {0, rb + 1, g, 0, 0};
                    int5 gc2 = {0, rb + 2, g, 0, 0};
                    int5 gc3 = {0, rb + 3, g, 0, 0};
                    for (int k = 0; k < R; ++k) {
                        gc0[0] = k; const bfloat128 r0 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc0, gr));
                        gc1[0] = k; const bfloat128 r1 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc1, gr));
                        gc2[0] = k; const bfloat128 r2 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc2, gr));
                        gc3[0] = k; const bfloat128 r3 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc3, gr));
                        wf[1] = k; wg[1] = k;
                        const bfloat128 wfv = v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS);
                        const bfloat128 wgv = v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS);
                        fa0 = v_bf16_mac_acc32_b(wfv, r0, fa0, 0); ga0 = v_bf16_mac_acc32_b(wgv, r0, ga0, 0);
                        fa1 = v_bf16_mac_acc32_b(wfv, r1, fa1, 0); ga1 = v_bf16_mac_acc32_b(wgv, r1, ga1, 0);
                        fa2 = v_bf16_mac_acc32_b(wfv, r2, fa2, 0); ga2 = v_bf16_mac_acc32_b(wgv, r2, ga2, 0);
                        fa3 = v_bf16_mac_acc32_b(wfv, r3, fa3, 0); ga3 = v_bf16_mac_acc32_b(wgv, r3, ga3, 0);
                    }
                    STORE_FG(0, fa0, ga0);
                    STORE_FG(1, fa1, ga1);
                    STORE_FG(2, fa2, ga2);
                    STORE_FG(3, fa3, ga3);
                } else {
                    float128 fa0 = {0}, ga0 = {0}, fa1 = {0}, ga1 = {0};
                    float128 fa2 = {0}, ga2 = {0}, fa3 = {0}, ga3 = {0};
                    float128 fa4 = {0}, ga4 = {0}, fa5 = {0}, ga5 = {0};
                    float128 fa6 = {0}, ga6 = {0}, fa7 = {0}, ga7 = {0};
                    int5 gc = {0, rb, g, 0, 0};
                    for (int k = 0; k < R; ++k) {
                        wf[1] = k; wg[1] = k;
                        const bfloat128 wfv = v_bf16_ld_tnsr_b(wf, fg_t, SW_L0CS);
                        const bfloat128 wgv = v_bf16_ld_tnsr_b(wg, fg_t, SW_L0CS);
                        gc[0] = k;
                        gc[1] = rb;     const bfloat128 r0 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[1] = rb + 1; const bfloat128 r1 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[1] = rb + 2; const bfloat128 r2 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[1] = rb + 3; const bfloat128 r3 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        fa0 = v_bf16_mac_acc32_b(wfv, r0, fa0, 0); ga0 = v_bf16_mac_acc32_b(wgv, r0, ga0, 0);
                        fa1 = v_bf16_mac_acc32_b(wfv, r1, fa1, 0); ga1 = v_bf16_mac_acc32_b(wgv, r1, ga1, 0);
                        fa2 = v_bf16_mac_acc32_b(wfv, r2, fa2, 0); ga2 = v_bf16_mac_acc32_b(wgv, r2, ga2, 0);
                        fa3 = v_bf16_mac_acc32_b(wfv, r3, fa3, 0); ga3 = v_bf16_mac_acc32_b(wgv, r3, ga3, 0);
                        gc[1] = rb + 4; const bfloat128 r4 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[1] = rb + 5; const bfloat128 r5 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[1] = rb + 6; const bfloat128 r6 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        gc[1] = rb + 7; const bfloat128 r7 = v_bf16_ld_g((__global__ bf16*)gen_addr(gc, gr));
                        fa4 = v_bf16_mac_acc32_b(wfv, r4, fa4, 0); ga4 = v_bf16_mac_acc32_b(wgv, r4, ga4, 0);
                        fa5 = v_bf16_mac_acc32_b(wfv, r5, fa5, 0); ga5 = v_bf16_mac_acc32_b(wgv, r5, ga5, 0);
                        fa6 = v_bf16_mac_acc32_b(wfv, r6, fa6, 0); ga6 = v_bf16_mac_acc32_b(wgv, r6, ga6, 0);
                        fa7 = v_bf16_mac_acc32_b(wfv, r7, fa7, 0); ga7 = v_bf16_mac_acc32_b(wgv, r7, ga7, 0);
                    }
                    STORE_FG(0, fa0, ga0); STORE_FG(1, fa1, ga1);
                    STORE_FG(2, fa2, ga2); STORE_FG(3, fa3, ga3);
                    STORE_FG(4, fa4, ga4); STORE_FG(5, fa5, ga5);
                    STORE_FG(6, fa6, ga6); STORE_FG(7, fa7, ga7);
                }

                const int gh = Gh == 1 ? 0 : g;
#pragma loop_unroll(1)
                for (int mi = 0; mi < nb; ++mi) {
                    const int r = rb + mi;
                    const float64 f1 = fg_buf[mi][0], f2 = fg_buf[mi][1];
                    const float64 g1 = fg_buf[mi][2], g2 = fg_buf[mi][3];
                    float64 tf1, tg1, tf2, tg2;
                    TANH_PADE_4(f1, g1, f2, g2, tf1, tg1, tf2, tg2);
                    const float64_pair_t a = ld_pair(h, c, r, gh);
                    const float64_pair_t b = ld_pair(m, c, r, m0 + g);
                    float64_pair_t yy;
                    yy.v1 = (1.0f - tf1) * a.v1 + tg1 * b.v1;
                    yy.v2 = (1.0f - tf2) * a.v2 + tg2 * b.v2;
                    int5 yc = {c, r, g, 0, 0};
                    v_f32_st_tnsr(yc, y, yy.v1);
                    yc[0] = c + 64;
                    v_f32_st_tnsr(yc, y, yy.v2);
                    const float64 sq = row_sum(yy.v1 * yy.v1 + yy.v2 * yy.v2);
                    sq_acc[mi] = v_f32_sel_eq_f32_b(lanes, (float)g, sq, sq_acc[mi]);
                }
            }

            for (int mi = 0; mi < nb; ++mi) {
                int5 pc = {0, tile, rb + mi, 0, 0};
                v_f32_st_tnsr(pc, partials, sq_acc[mi]);
            }
            rb += nb;
        }
    }
}
