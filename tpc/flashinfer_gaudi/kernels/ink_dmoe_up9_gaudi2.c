// SPDX-License-Identifier: Apache-2.0
// Inkling-2 B=1/multi-T routed-expert gate/up GEMV + exact [9/8] Pade SwiGLU4 (CH = 1, NG = 9, 256 rows/point).
// Shared template compiled with -DNVF4 (ink_dmoe_up9_nvf4_gaudi2.c) or -DMXF8 (ink_dmoe_up9_mxf8_gaudi2.c).
// Tensors (dim 0 fastest): x bf16 [K, T], idx i32 [J, T], w u8 [2F, K/2 or K, E], bs f8 [2F, K/16 or K/32, E],
// si f32 [2F, E], act bf16 [F, J, T]. Index space [NG * J * T] (NG = 2F / 256 = 9). Params: up softcap (gate = 4).
#include "ink_math_gaudi2.h"

static inline float64 bf16r_v(float64 v) {
    float64_pair_t p;
    p.v1 = v; p.v2 = v;
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(p)).v1;
}

#define PADE98_ND2(X0, X1, N0, N1, D0, D1) do { \
        const float64 c0_ = v_f32_min_b(v_f32_max_b((X0), -6.5f), 6.5f); \
        const float64 c1_ = v_f32_min_b(v_f32_max_b((X1), -6.5f), 6.5f); \
        const float64 q0_ = c0_ * c0_, q1_ = c1_ * c1_; \
        float64 n0_ = v_f32_mac_b(q0_, 2.901963686e-8f, 2.872944049e-5f); \
        float64 n1_ = v_f32_mac_b(q1_, 2.901963686e-8f, 2.872944049e-5f); \
        float64 d0_ = v_f32_mac_b(q0_, 1.305883659e-6f, 4.022121669e-4f); \
        float64 d1_ = v_f32_mac_b(q1_, 1.305883659e-6f, 4.022121669e-4f); \
        n0_ = v_f32_mac_b(n0_, q0_, 3.921568627e-3f); n1_ = v_f32_mac_b(n1_, q1_, 3.921568627e-3f); \
        d0_ = v_f32_mac_b(d0_, q0_, 2.745098039e-2f); d1_ = v_f32_mac_b(d1_, q1_, 2.745098039e-2f); \
        n0_ = v_f32_mac_b(n0_, q0_, 1.372549020e-1f); n1_ = v_f32_mac_b(n1_, q1_, 1.372549020e-1f); \
        d0_ = v_f32_mac_b(d0_, q0_, 4.705882353e-1f); d1_ = v_f32_mac_b(d1_, q1_, 4.705882353e-1f); \
        n0_ = v_f32_mac_b(n0_, q0_, 1.0f); n1_ = v_f32_mac_b(n1_, q1_, 1.0f); \
        (D0) = v_f32_mac_b(d0_, q0_, 1.0f); (D1) = v_f32_mac_b(d1_, q1_, 1.0f); \
        (N0) = n0_ * c0_; (N1) = n1_ * c1_; \
    } while (0)

#define SWIGLU4_2(G0, G1, U0, U1, VA0) do { \
        float64 ng0_, ng1_, dg0_, dg1_; \
        PADE98_ND2((G0) * 0.25f, (G1) * 0.25f, ng0_, ng1_, dg0_, dg1_); \
        const float64 s0_ = dg0_ + ng0_, s1_ = dg1_ + ng1_; \
        const float64 ss0_ = s0_ * s0_, ss1_ = s1_ * s1_; \
        const float64 dd0_ = v_f32_mac_b(ng0_, ng0_, dg0_ * dg0_); \
        const float64 dd1_ = v_f32_mac_b(ng1_, ng1_, dg1_ * dg1_); \
        const float64 gn0_ = (ng0_ * ss0_) * c2cu, gn1_ = (ng1_ * ss1_) * c2cu; \
        const float64 gd0_ = dg0_ * dd0_, gd1_ = dg1_ * dd1_; \
        float64 nu0_, nu1_, du0_, du1_; \
        PADE98_ND2((U0) * inv_cu, (U1) * inv_cu, nu0_, nu1_, du0_, du1_); \
        const float64 nm0_ = gn0_ * nu0_, nm1_ = gn1_ * nu1_; \
        const float64 dm0_ = gd0_ * du0_, dm1_ = gd1_ * du1_; \
        float64 r0_ = as_float64((int64)0x7ef311c3 - as_int64(dm0_)); \
        float64 r1_ = as_float64((int64)0x7ef311c3 - as_int64(dm1_)); \
        r0_ = r0_ * (2.0f - dm0_ * r0_); r1_ = r1_ * (2.0f - dm1_ * r1_); \
        r0_ = r0_ * (2.0f - dm0_ * r0_); r1_ = r1_ * (2.0f - dm1_ * r1_); \
        r0_ = r0_ * (2.0f - dm0_ * r0_); r1_ = r1_ * (2.0f - dm1_ * r1_); \
        float64_pair_t p0_; \
        p0_.v1 = nm0_ * r0_; p0_.v2 = nm1_ * r1_; \
        (VA0) = v_convert_f32_to_bf16_all_b(p0_); \
    } while (0)

#ifdef NVF4
// NVFP4 weights (exact): w u8 [2F, K / 2, E] holds two K rows per byte (low nibble: even k, high nibble: odd k), each
// an E2M1 code (sign, 2 exponent bits, 1 mantissa bit); bs f8 [2F, K / 16, E] holds fp8_143(scale / 8) in the same
// column order; si f32 [2F, E] holds g[e] * 512 (2^6 from E2M1 FP8 code s00eem00 * 2^3 from scale / 8). Block
// accumulation sums 16 k of bare E2M1 codes into F32 before scaling once by bs.
#define NVF4_UNPACK_ASH(WW, LO, HI) do { \
        const char256 ws_ = *((char256*)&(WW)); \
        const char256 w4_ = ws_ << 4; \
        const char256 ah_ = v_i8_ash_b(ws_, (char)-2, 0); \
        const char256 al_ = v_i8_ash_b(w4_, (char)-2, 0); \
        (LO) = *((uchar256*)&al_) & (uchar256)0x9c; \
        (HI) = *((uchar256*)&ah_) & (uchar256)0x9c; \
    } while (0)
void main(tensor x, tensor idx, tensor w, tensor bs, tensor si, tensor act, float cu)
#elif defined(MXF8)
// Native MXFP8 weights (exact): w u8 [2F, K, E] holds checkpoint E4M3FN codes (1 byte/weight, natural k, checkpoint
// column order gate_0, up_0, ...); bs f8 [2F, K / 32, E] holds fp8_143(2^(s - s_max)) <= 1 in the same column order;
// si f32 [2F, E] holds g[e] = 2^(s_max - 127) (constant per expert). Exact table-free E4M3FN -> BF16 conversion:
// codes with exponent field 15 are converted as (code - 0x08) and doubled via +0x0080 to the BF16 exponent (8 VPU ops,
// 0 constant VRFs, 0 VPRFs). Block accumulation sums 32 k into F32 before scaling once by bs.
void main(tensor x, tensor idx, tensor w, tensor bs, tensor si, tensor act, float cu)
#else
#error "Define NVF4 or MXF8"
#endif
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
#ifdef NVF4
    const int K = get_dim_size(w, 1) * 2;
    const int NB = K >> 4;
#elif defined(MXF8)
    const int K = get_dim_size(w, 1);
    const int NB = K >> 5;
#endif
    const int NG = get_dim_size(w, 0) >> 8;  // 2F / 256 = 9
    const int JJ = get_dim_size(idx, 0);
    const int T = get_dim_size(idx, 1);
    const int interleave24 = (NG * JJ == 72);
    const int wpt = 3 * T;
    const float inv_cu = ink_srecip(cu), c2cu = 2.0f * cu;

    for (int pp = start[0]; pp < end[0]; ++pp) {
        int wi = pp;
        if (interleave24) {
            const int tpc_id = pp / wpt;
            const int wave = pp - tpc_id * wpt;
            wi = wave * 24 + tpc_id;
        }
        const int u = wi / NG;
        const int g = wi - u * NG;
        const int t = u / JJ;
        const int j = u - t * JJ;

        int5 ic = {j, t, 0, 0, 0};
        const int e = s_i32_ld_g((__global__ int*)gen_addr(ic, idx));
        const int n0 = g << 8;
        if (e < 0) {
            int5 oc = {n0 >> 1, j, t, 0, 0};
            v_bf16_st_tnsr(oc, act, (bfloat128)0);
            continue;
        }

        float128 g0 = {0}, u0 = {0};
        int5 wc = {n0, 0, e, 0, 0};
        int5 xc = {0, t, 0, 0, 0};
#ifdef NVF4
        int5 bc = {n0, 0, e, 0, 0};
        wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
        xc[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
        xc[0] = 1; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
        xc[0] = 2; bfloat128 xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
        xc[0] = 3; bfloat128 xe = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
#define PIPE_NV(WW, XA, XB, R_NEXT, K_NEXT, DO_W, DO_X) do { \
            uchar256 lo_, hi_; \
            NVF4_UNPACK_ASH((WW), lo_, hi_); \
            if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
            const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
            const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
            bg = v_bf16_mac_acc32_b(pl_.v1, (XA), bg, 0); \
            bu = v_bf16_mac_acc32_b(pl_.v2, (XA), bu, 0); \
            if (DO_X) { xc[0] = (K_NEXT); (XA) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x)); } \
            bg = v_bf16_mac_acc32_b(ph_.v1, (XB), bg, 0); \
            bu = v_bf16_mac_acc32_b(ph_.v2, (XB), bu, 0); \
            if (DO_X) { xc[0] = (K_NEXT) + 1; (XB) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x)); } \
        } while (0)
#pragma loop_unroll(1)
        for (int kb = 0; kb < NB - 1; ++kb) {
            const int r = kb << 3, k = kb << 4;
            float128 bg = {0}, bu = {0};
            PIPE_NV(w0, xa, xb, r + 4,  k + 4,  1, 1);
            PIPE_NV(w1, xd, xe, r + 5,  k + 6,  1, 1);
            PIPE_NV(w2, xa, xb, r + 6,  k + 8,  1, 1);
            PIPE_NV(w3, xd, xe, r + 7,  k + 10, 1, 1);
            PIPE_NV(w0, xa, xb, r + 8,  k + 12, 1, 1);
            PIPE_NV(w1, xd, xe, r + 9,  k + 14, 1, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            PIPE_NV(w2, xa, xb, r + 10, k + 16, 1, 1);
            PIPE_NV(w3, xd, xe, r + 11, k + 18, 1, 1);
            g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
            u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
        }
        {
            const int kb = NB - 1, r = kb << 3, k = kb << 4;
            float128 bg = {0}, bu = {0};
            PIPE_NV(w0, xa, xb, r + 4, k + 4,  1, 1);
            PIPE_NV(w1, xd, xe, r + 5, k + 6,  1, 1);
            PIPE_NV(w2, xa, xb, r + 6, k + 8,  1, 1);
            PIPE_NV(w3, xd, xe, r + 7, k + 10, 1, 1);
            PIPE_NV(w0, xa, xb, 0,     k + 12, 0, 1);
            PIPE_NV(w1, xd, xe, 0,     k + 14, 0, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            PIPE_NV(w2, xa, xb, 0,     0,      0, 0);
            PIPE_NV(w3, xd, xe, 0,     0,      0, 0);
            g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
            u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
        }
#undef PIPE_NV
#elif defined(MXF8)
        int5 bc = {n0, 0, e, 0, 0};
        wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
        xc[0] = 0; bfloat128 x0 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
        xc[0] = 1; bfloat128 x1 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
        xc[0] = 2; bfloat128 x2 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
        xc[0] = 3; bfloat128 x3 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x));
#define PIPE_MX(WW, XX, K_NEXT, DO_W, DO_X) do { \
            const uchar256 e_ = (WW) & (uchar256)0x78; \
            const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
            const uchar256 wa_ = (WW) - d8_; \
            if (DO_W) { wc[1] = (K_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
            const uchar256 d80_ = d8_ << 4; \
            const bfloat128_pair_t p_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
            const ushort128_pair_t d_ = v_convert_u8_to_u16_all_b(d80_); \
            const ushort128 r1_ = *((ushort128*)&p_.v1) + d_.v1; \
            const ushort128 r2_ = *((ushort128*)&p_.v2) + d_.v2; \
            bg = v_bf16_mac_acc32_b(*((bfloat128*)&r1_), (XX), bg, 0); \
            bu = v_bf16_mac_acc32_b(*((bfloat128*)&r2_), (XX), bu, 0); \
            if (DO_X) { xc[0] = (K_NEXT); (XX) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, x)); } \
        } while (0)
#pragma loop_unroll(1)
        for (int kb = 0; kb < NB - 1; ++kb) {
            const int k0 = kb << 5;
            float128 bg = {0}, bu = {0};
            PIPE_MX(w0, x0, k0 + 4, 1, 1);  PIPE_MX(w1, x1, k0 + 5, 1, 1);
            PIPE_MX(w2, x2, k0 + 6, 1, 1);  PIPE_MX(w3, x3, k0 + 7, 1, 1);
            PIPE_MX(w0, x0, k0 + 8, 1, 1);  PIPE_MX(w1, x1, k0 + 9, 1, 1);
            PIPE_MX(w2, x2, k0 + 10, 1, 1); PIPE_MX(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
            for (int sub = 1; sub < 3; ++sub) {
                const int ks = k0 + (sub << 3);
                PIPE_MX(w0, x0, ks + 4, 1, 1);  PIPE_MX(w1, x1, ks + 5, 1, 1);
                PIPE_MX(w2, x2, ks + 6, 1, 1);  PIPE_MX(w3, x3, ks + 7, 1, 1);
                PIPE_MX(w0, x0, ks + 8, 1, 1);  PIPE_MX(w1, x1, ks + 9, 1, 1);
                PIPE_MX(w2, x2, ks + 10, 1, 1); PIPE_MX(w3, x3, ks + 11, 1, 1);
            }
            PIPE_MX(w0, x0, k0 + 28, 1, 1); PIPE_MX(w1, x1, k0 + 29, 1, 1);
            PIPE_MX(w2, x2, k0 + 30, 1, 1); PIPE_MX(w3, x3, k0 + 31, 1, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            PIPE_MX(w0, x0, k0 + 32, 1, 1); PIPE_MX(w1, x1, k0 + 33, 1, 1);
            PIPE_MX(w2, x2, k0 + 34, 1, 1); PIPE_MX(w3, x3, k0 + 35, 1, 1);
            g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
            u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
        }
        {
            const int k0 = (NB - 1) << 5;
            float128 bg = {0}, bu = {0};
            PIPE_MX(w0, x0, k0 + 4, 1, 1);  PIPE_MX(w1, x1, k0 + 5, 1, 1);
            PIPE_MX(w2, x2, k0 + 6, 1, 1);  PIPE_MX(w3, x3, k0 + 7, 1, 1);
            PIPE_MX(w0, x0, k0 + 8, 1, 1);  PIPE_MX(w1, x1, k0 + 9, 1, 1);
            PIPE_MX(w2, x2, k0 + 10, 1, 1); PIPE_MX(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
            for (int sub = 1; sub < 3; ++sub) {
                const int ks = k0 + (sub << 3);
                PIPE_MX(w0, x0, ks + 4, 1, 1);  PIPE_MX(w1, x1, ks + 5, 1, 1);
                PIPE_MX(w2, x2, ks + 6, 1, 1);  PIPE_MX(w3, x3, ks + 7, 1, 1);
                PIPE_MX(w0, x0, ks + 8, 1, 1);  PIPE_MX(w1, x1, ks + 9, 1, 1);
                PIPE_MX(w2, x2, ks + 10, 1, 1); PIPE_MX(w3, x3, ks + 11, 1, 1);
            }
            PIPE_MX(w0, x0, k0 + 28, 1, 1); PIPE_MX(w1, x1, k0 + 29, 1, 1);
            PIPE_MX(w2, x2, k0 + 30, 1, 1); PIPE_MX(w3, x3, k0 + 31, 1, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            PIPE_MX(w0, x0, 0, 0, 0); PIPE_MX(w1, x1, 0, 0, 0);
            PIPE_MX(w2, x2, 0, 0, 0); PIPE_MX(w3, x3, 0, 0, 0);
            g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
            u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
        }
#undef PIPE_MX
#endif

        int5 sc = {n0, e, 0, 0, 0};
        const float64 ge = bf16r_v(g0.v1 * v_f32_ld_tnsr_b(sc, si)); sc[0] = n0 + 64;
        const float64 go = bf16r_v(g0.v2 * v_f32_ld_tnsr_b(sc, si)); sc[0] = n0 + 128;
        const float64 ue = bf16r_v(u0.v1 * v_f32_ld_tnsr_b(sc, si)); sc[0] = n0 + 192;
        const float64 uo = bf16r_v(u0.v2 * v_f32_ld_tnsr_b(sc, si));
        bfloat128 va;
        SWIGLU4_2(ge, go, ue, uo, va);
        int5 oc = {n0 >> 1, j, t, 0, 0};
        v_bf16_st_tnsr(oc, act, va);
    }
}
