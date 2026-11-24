// SPDX-License-Identifier: Apache-2.0
// Inkling-2 verify / batched-decode grouped routed-expert gate/up GEMV + exact [9/8] Pade SwiGLU4 (CH = 1, NG = 9).
// Shared template compiled with -DNVF4 (ink_vmoe_up_nvf4_gaudi2.c) or -DMXF8 (ink_vmoe_up_mxf8_gaudi2.c).
// Uses Thue-Morse paired-snake 24-TPC mapping (when NG * J == 72) for 3x finer load balance & zero straggler stacking.
// Tensors (dim 0 fastest):
//   x    bf16 [K, T]
//   grp  i32  [64, 4]      (from ink_vmoe_plan: row0=e, row1=cnt, row2=ts0, row3=ts1)
//   w    u8   [2F, K/2 or K, E]
//   bs   f8   [2F, K/16 or K/32, E]
//   si   f32  [2F, E]
//   act  bf16 [F, J, T]
// Index space [NG * J * T] (NG = 2F / 256 = 9 -> 9 * J * T). Params: cu (up softcap; gate softcap = 4).
#if !defined(NVF4) && !defined(MXF8)
#error "Define NVF4 or MXF8"
#endif
#include "ink_math_gaudi2.h"

__local__ float64 acc_buf[4][4];

static inline float64 bf16r_v(float64 v) {
    float64_pair_t p;
    p.v1 = v; p.v2 = v;
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(p)).v1;
}

#define PADE98_ND4(X0, X1, X2, X3, N0, N1, N2, N3, D0, D1, D2, D3) do { \
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
        (D0) = v_f32_mac_b(d0_, q0_, 1.0f); (D1) = v_f32_mac_b(d1_, q1_, 1.0f); \
        (D2) = v_f32_mac_b(d2_, q2_, 1.0f); (D3) = v_f32_mac_b(d3_, q3_, 1.0f); \
        (N0) = n0_ * c0_; (N1) = n1_ * c1_; (N2) = n2_ * c2_; (N3) = n3_ * c3_; \
    } while (0)

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

#define SWIGLU4_4(G0, G1, G2, G3, U0, U1, U2, U3, VA0, VA1) do { \
        float64 ng0_, ng1_, ng2_, ng3_, dg0_, dg1_, dg2_, dg3_; \
        PADE98_ND4((G0) * 0.25f, (G1) * 0.25f, (G2) * 0.25f, (G3) * 0.25f, \
                   ng0_, ng1_, ng2_, ng3_, dg0_, dg1_, dg2_, dg3_); \
        const float64 s0_ = dg0_ + ng0_, s1_ = dg1_ + ng1_, s2_ = dg2_ + ng2_, s3_ = dg3_ + ng3_; \
        const float64 ss0_ = s0_ * s0_, ss1_ = s1_ * s1_, ss2_ = s2_ * s2_, ss3_ = s3_ * s3_; \
        const float64 dd0_ = v_f32_mac_b(ng0_, ng0_, dg0_ * dg0_); \
        const float64 dd1_ = v_f32_mac_b(ng1_, ng1_, dg1_ * dg1_); \
        const float64 dd2_ = v_f32_mac_b(ng2_, ng2_, dg2_ * dg2_); \
        const float64 dd3_ = v_f32_mac_b(ng3_, ng3_, dg3_ * dg3_); \
        const float64 gn0_ = (ng0_ * ss0_) * c2cu, gn1_ = (ng1_ * ss1_) * c2cu; \
        const float64 gn2_ = (ng2_ * ss2_) * c2cu, gn3_ = (ng3_ * ss3_) * c2cu; \
        const float64 gd0_ = dg0_ * dd0_, gd1_ = dg1_ * dd1_, gd2_ = dg2_ * dd2_, gd3_ = dg3_ * dd3_; \
        float64 nu0_, nu1_, nu2_, nu3_, du0_, du1_, du2_, du3_; \
        PADE98_ND4((U0) * inv_cu, (U1) * inv_cu, (U2) * inv_cu, (U3) * inv_cu, \
                   nu0_, nu1_, nu2_, nu3_, du0_, du1_, du2_, du3_); \
        const float64 nm0_ = gn0_ * nu0_, nm1_ = gn1_ * nu1_, nm2_ = gn2_ * nu2_, nm3_ = gn3_ * nu3_; \
        const float64 dm0_ = gd0_ * du0_, dm1_ = gd1_ * du1_, dm2_ = gd2_ * du2_, dm3_ = gd3_ * du3_; \
        float64 r0_ = as_float64((int64)0x7ef311c3 - as_int64(dm0_)); \
        float64 r1_ = as_float64((int64)0x7ef311c3 - as_int64(dm1_)); \
        float64 r2_ = as_float64((int64)0x7ef311c3 - as_int64(dm2_)); \
        float64 r3_ = as_float64((int64)0x7ef311c3 - as_int64(dm3_)); \
        r0_ = r0_ * (2.0f - dm0_ * r0_); r1_ = r1_ * (2.0f - dm1_ * r1_); \
        r2_ = r2_ * (2.0f - dm2_ * r2_); r3_ = r3_ * (2.0f - dm3_ * r3_); \
        r0_ = r0_ * (2.0f - dm0_ * r0_); r1_ = r1_ * (2.0f - dm1_ * r1_); \
        r2_ = r2_ * (2.0f - dm2_ * r2_); r3_ = r3_ * (2.0f - dm3_ * r3_); \
        r0_ = r0_ * (2.0f - dm0_ * r0_); r1_ = r1_ * (2.0f - dm1_ * r1_); \
        r2_ = r2_ * (2.0f - dm2_ * r2_); r3_ = r3_ * (2.0f - dm3_ * r3_); \
        float64_pair_t p0_, p1_; \
        p0_.v1 = nm0_ * r0_; p0_.v2 = nm1_ * r1_; \
        p1_.v1 = nm2_ * r2_; p1_.v2 = nm3_ * r3_; \
        (VA0) = v_convert_f32_to_bf16_all_b(p0_); \
        (VA1) = v_convert_f32_to_bf16_all_b(p1_); \
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

#define STORE_ACC1(M_IDX, G0, U0) do { \
        acc_buf[M_IDX][0] = (G0).v1; acc_buf[M_IDX][1] = (G0).v2; \
        acc_buf[M_IDX][2] = (U0).v1; acc_buf[M_IDX][3] = (U0).v2; \
    } while (0)

#ifdef NVF4
#define NVF4_UNPACK_ASH(WW, LO, HI) do { \
        const char256 ws_ = *((char256*)&(WW)); \
        const char256 w4_ = ws_ << 4; \
        const char256 ah_ = v_i8_ash_b(ws_, (char)-2, 0); \
        const char256 al_ = v_i8_ash_b(w4_, (char)-2, 0); \
        (LO) = *((uchar256*)&al_) & (uchar256)0x9c; \
        (HI) = *((uchar256*)&ah_) & (uchar256)0x9c; \
    } while (0)
#endif
#define ZERO_ACC_VLM(M_IDX) do { \
        acc_buf[M_IDX][0] = (float64)0.0f; acc_buf[M_IDX][1] = (float64)0.0f; \
        acc_buf[M_IDX][2] = (float64)0.0f; acc_buf[M_IDX][3] = (float64)0.0f; \
    } while (0)
#define REDUCE_ACC_VLM(M_IDX, BG, BU, SC) do { \
        float128 og_, ou_; \
        og_.v1 = acc_buf[M_IDX][0]; og_.v2 = acc_buf[M_IDX][1]; \
        ou_.v1 = acc_buf[M_IDX][2]; ou_.v2 = acc_buf[M_IDX][3]; \
        og_ = v_f32_mac_x2_b((BG), (SC).v1, (SC).v3, og_, 0); \
        ou_ = v_f32_mac_x2_b((BU), (SC).v2, (SC).v4, ou_, 0); \
        acc_buf[M_IDX][0] = og_.v1; acc_buf[M_IDX][1] = og_.v2; \
        acc_buf[M_IDX][2] = ou_.v1; acc_buf[M_IDX][3] = ou_.v2; \
    } while (0)

void main(tensor x, tensor grp, tensor w, tensor bs, tensor si, tensor act, float cu)
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
    const int J = get_dim_size(act, 1);
    const int T = get_dim_size(act, 2);
    const int interleave24 = (NG * J == 72);
    const int wpt = 3 * T;
    const int single_tpc_slice = interleave24 && ((end[0] - start[0]) == wpt);
    const float inv_cu = ink_srecip(cu), c2cu = 2.0f * cu;

    for (int pp = start[0]; pp < end[0]; ++pp) {
        int work_idx = pp;
        if (interleave24) {
            const int tpc_id = pp / wpt;
            const int wave = pp - tpc_id * wpt;
            const int rev = (wave ^ (wave >> 1)) & 1;
            const int lane24 = rev ? (23 - tpc_id) : tpc_id;
            work_idx = wave * 24 + lane24;
        }
        const int u = work_idx / NG;
        const int g = work_idx - u * NG;

        int5 gc = {u, 0, 0, 0, 0};
        const int e = s_i32_ld_g((__global__ int*)gen_addr(gc, grp));
        if (e < 0) {
            if (single_tpc_slice) break;
            continue;
        }
        gc[1] = 1;
        const int cnt = s_i32_ld_g((__global__ int*)gen_addr(gc, grp));
        if (cnt <= 0) continue;
        gc[1] = 2;
        const int tsp = s_i32_ld_g((__global__ int*)gen_addr(gc, grp));
        const int nm = cnt < 4 ? cnt : 4;

        const int n0 = g << 8;
        int5 wc = {n0, 0, e, 0, 0};

        const int c0 = tsp & 0xff;
        const int c1 = (tsp >> 8) & 0xff;
        const int c2 = (tsp >> 16) & 0xff;
        const int c3 = (tsp >> 24) & 0xff;
        const int t0 = c0 >> 4, s0 = c0 & 15;
        const int t1 = c1 >> 4, s1 = c1 & 15;
        const int t2 = c2 >> 4, s2 = c2 & 15;
        const int t3 = c3 >> 4, s3 = c3 & 15;

#ifdef NVF4
        int5 bc = {n0, 0, e, 0, 0};
        if (nm == 1) {
            float128 g0 = {0}, u0 = {0};
            int5 xc0 = {0, t0, 0, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc0[0] = 1; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc0[0] = 2; bfloat128 xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc0[0] = 3; bfloat128 xe = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
#define NVP1(WW, XA, XB, R_NEXT, K_NEXT, DO_W, DO_X) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                bg = v_bf16_mac_acc32_b(pl_.v1, (XA), bg, 0); \
                bu = v_bf16_mac_acc32_b(pl_.v2, (XA), bu, 0); \
                if (DO_X) { xc0[0] = (K_NEXT); (XA) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); } \
                bg = v_bf16_mac_acc32_b(ph_.v1, (XB), bg, 0); \
                bu = v_bf16_mac_acc32_b(ph_.v2, (XB), bu, 0); \
                if (DO_X) { xc0[0] = (K_NEXT) + 1; (XB) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); } \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 bg = {0}, bu = {0};
                NVP1(w0, xa, xb, r + 4,  k + 4,  1, 1);
                NVP1(w1, xd, xe, r + 5,  k + 6,  1, 1);
                NVP1(w2, xa, xb, r + 6,  k + 8,  1, 1);
                NVP1(w3, xd, xe, r + 7,  k + 10, 1, 1);
                NVP1(w0, xa, xb, r + 8,  k + 12, 1, 1);
                NVP1(w1, xd, xe, r + 9,  k + 14, 1, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                NVP1(w2, xa, xb, r + 10, k + 16, 1, 1);
                NVP1(w3, xd, xe, r + 11, k + 18, 1, 1);
                g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
                u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 bg = {0}, bu = {0};
                NVP1(w0, xa, xb, r + 4, k + 4,  1, 1);
                NVP1(w1, xd, xe, r + 5, k + 6,  1, 1);
                NVP1(w2, xa, xb, r + 6, k + 8,  1, 1);
                NVP1(w3, xd, xe, r + 7, k + 10, 1, 1);
                NVP1(w0, xa, xb, 0,     k + 12, 0, 1);
                NVP1(w1, xd, xe, 0,     k + 14, 0, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                NVP1(w2, xa, xb, 0,     0,      0, 0);
                NVP1(w3, xd, xe, 0,     0,      0, 0);
                g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
                u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
            }
#undef NVP1
            STORE_ACC1(0, g0, u0);
        } else if (nm == 2) {
            float128 g0 = {0}, u0 = {0}, g1 = {0}, u1 = {0};
            int5 xc0 = {0, t0, 0, 0, 0};
            int5 xc1 = {0, t1, 0, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define NVP2(WW, KK, R_NEXT, DO_W) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                xc0[0] = (KK); const bfloat128 xa0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); \
                xc1[0] = (KK); const bfloat128 xb0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                bg0 = v_bf16_mac_acc32_b(pl_.v1, xa0_, bg0, 0); bu0 = v_bf16_mac_acc32_b(pl_.v2, xa0_, bu0, 0); \
                bg1 = v_bf16_mac_acc32_b(pl_.v1, xb0_, bg1, 0); bu1 = v_bf16_mac_acc32_b(pl_.v2, xb0_, bu1, 0); \
                xc0[0] = (KK) + 1; const bfloat128 xa1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); \
                xc1[0] = (KK) + 1; const bfloat128 xb1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                bg0 = v_bf16_mac_acc32_b(ph_.v1, xa1_, bg0, 0); bu0 = v_bf16_mac_acc32_b(ph_.v2, xa1_, bu0, 0); \
                bg1 = v_bf16_mac_acc32_b(ph_.v1, xb1_, bg1, 0); bu1 = v_bf16_mac_acc32_b(ph_.v2, xb1_, bu1, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                NVP2(w0, k + 0,  r + 2, 1); NVP2(w1, k + 2,  r + 3, 1);
                NVP2(w0, k + 4,  r + 4, 1); NVP2(w1, k + 6,  r + 5, 1);
                NVP2(w0, k + 8,  r + 6, 1); NVP2(w1, k + 10, r + 7, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                NVP2(w0, k + 12, r + 8, 1); NVP2(w1, k + 14, r + 9, 1);
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                NVP2(w0, k + 0,  r + 2, 1); NVP2(w1, k + 2,  r + 3, 1);
                NVP2(w0, k + 4,  r + 4, 1); NVP2(w1, k + 6,  r + 5, 1);
                NVP2(w0, k + 8,  r + 6, 1); NVP2(w1, k + 10, r + 7, 1);
                NVP2(w0, k + 12, 0,     0); NVP2(w1, k + 14, 0,     0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
            }
#undef NVP2
            STORE_ACC1(0, g0, u0); STORE_ACC1(1, g1, u1);
        } else if (nm == 3) {
            int5 xc0 = {0, t0, 0, 0, 0};
            int5 xc1 = {0, t1, 0, 0, 0};
            int5 xc2 = {0, t2, 0, 0, 0};
            float128 g0 = {0}, u0 = {0}, g1 = {0}, u1 = {0};
            ZERO_ACC_VLM(2);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define NVP3B(WW, KK, R_NEXT, DO_W) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                xc0[0] = (KK); const bfloat128 xa0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); \
                xc1[0] = (KK); const bfloat128 xb0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); \
                xc2[0] = (KK); const bfloat128 xc0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x)); \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                bg0 = v_bf16_mac_acc32_b(pl_.v1, xa0_, bg0, 0); bu0 = v_bf16_mac_acc32_b(pl_.v2, xa0_, bu0, 0); \
                bg1 = v_bf16_mac_acc32_b(pl_.v1, xb0_, bg1, 0); bu1 = v_bf16_mac_acc32_b(pl_.v2, xb0_, bu1, 0); \
                bg2 = v_bf16_mac_acc32_b(pl_.v1, xc0_, bg2, 0); bu2 = v_bf16_mac_acc32_b(pl_.v2, xc0_, bu2, 0); \
                xc0[0] = (KK) + 1; const bfloat128 xa1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); \
                xc1[0] = (KK) + 1; const bfloat128 xb1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); \
                xc2[0] = (KK) + 1; const bfloat128 xc1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                bg0 = v_bf16_mac_acc32_b(ph_.v1, xa1_, bg0, 0); bu0 = v_bf16_mac_acc32_b(ph_.v2, xa1_, bu0, 0); \
                bg1 = v_bf16_mac_acc32_b(ph_.v1, xb1_, bg1, 0); bu1 = v_bf16_mac_acc32_b(ph_.v2, xb1_, bu1, 0); \
                bg2 = v_bf16_mac_acc32_b(ph_.v1, xc1_, bg2, 0); bu2 = v_bf16_mac_acc32_b(ph_.v2, xc1_, bu2, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0}, bg2 = {0}, bu2 = {0};
                NVP3B(w0, k + 0, r + 2, 1);
                NVP3B(w1, k + 2, r + 3, 1);
#pragma loop_unroll(1)
                for (int sb = 1; sb < 4; ++sb) {
                    const int rs = r + (sb << 1), ks = k + (sb << 2);
                    NVP3B(w0, ks + 0, rs + 2, 1);
                    NVP3B(w1, ks + 2, rs + 3, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 og2, ou2;
                og2.v1 = acc_buf[2][0]; og2.v2 = acc_buf[2][1];
                ou2.v1 = acc_buf[2][2]; ou2.v2 = acc_buf[2][3];
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
                og2 = v_f32_mac_x2_b(bg2, sc.v1, sc.v3, og2, 0); ou2 = v_f32_mac_x2_b(bu2, sc.v2, sc.v4, ou2, 0);
                acc_buf[2][0] = og2.v1; acc_buf[2][1] = og2.v2;
                acc_buf[2][2] = ou2.v1; acc_buf[2][3] = ou2.v2;
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0}, bg2 = {0}, bu2 = {0};
                NVP3B(w0, k + 0,  r + 2, 1); NVP3B(w1, k + 2,  r + 3, 1);
                NVP3B(w0, k + 4,  r + 4, 1); NVP3B(w1, k + 6,  r + 5, 1);
                NVP3B(w0, k + 8,  r + 6, 1); NVP3B(w1, k + 10, r + 7, 1);
                NVP3B(w0, k + 12, 0,     0); NVP3B(w1, k + 14, 0,     0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
                REDUCE_ACC_VLM(2, bg2, bu2, sc);
            }
#undef NVP3B
            STORE_ACC1(0, g0, u0); STORE_ACC1(1, g1, u1);
        } else {
            int5 xc0 = {0, t0, 0, 0, 0};
            int5 xc1 = {0, t1, 0, 0, 0};
            int5 xc2 = {0, t2, 0, 0, 0};
            int5 xc3 = {0, t3, 0, 0, 0};
            ZERO_ACC_VLM(0); ZERO_ACC_VLM(1); ZERO_ACC_VLM(2); ZERO_ACC_VLM(3);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define NVP4V(WW, KK, R_NEXT, DO_W) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                xc0[0] = (KK); const bfloat128 xa0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); \
                xc1[0] = (KK); const bfloat128 xb0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); \
                xc2[0] = (KK); const bfloat128 xc0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x)); \
                xc3[0] = (KK); const bfloat128 xd0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, x)); \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                bg0 = v_bf16_mac_acc32_b(pl_.v1, xa0_, bg0, 0); bu0 = v_bf16_mac_acc32_b(pl_.v2, xa0_, bu0, 0); \
                bg1 = v_bf16_mac_acc32_b(pl_.v1, xb0_, bg1, 0); bu1 = v_bf16_mac_acc32_b(pl_.v2, xb0_, bu1, 0); \
                bg2 = v_bf16_mac_acc32_b(pl_.v1, xc0_, bg2, 0); bu2 = v_bf16_mac_acc32_b(pl_.v2, xc0_, bu2, 0); \
                bg3 = v_bf16_mac_acc32_b(pl_.v1, xd0_, bg3, 0); bu3 = v_bf16_mac_acc32_b(pl_.v2, xd0_, bu3, 0); \
                xc0[0] = (KK) + 1; const bfloat128 xa1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); \
                xc1[0] = (KK) + 1; const bfloat128 xb1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); \
                xc2[0] = (KK) + 1; const bfloat128 xc1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x)); \
                xc3[0] = (KK) + 1; const bfloat128 xd1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, x)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                bg0 = v_bf16_mac_acc32_b(ph_.v1, xa1_, bg0, 0); bu0 = v_bf16_mac_acc32_b(ph_.v2, xa1_, bu0, 0); \
                bg1 = v_bf16_mac_acc32_b(ph_.v1, xb1_, bg1, 0); bu1 = v_bf16_mac_acc32_b(ph_.v2, xb1_, bu1, 0); \
                bg2 = v_bf16_mac_acc32_b(ph_.v1, xc1_, bg2, 0); bu2 = v_bf16_mac_acc32_b(ph_.v2, xc1_, bu2, 0); \
                bg3 = v_bf16_mac_acc32_b(ph_.v1, xd1_, bg3, 0); bu3 = v_bf16_mac_acc32_b(ph_.v2, xd1_, bu3, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                float128 bg2 = {0}, bu2 = {0}, bg3 = {0}, bu3 = {0};
                NVP4V(w0, k + 0, r + 2, 1);
                NVP4V(w1, k + 2, r + 3, 1);
#pragma loop_unroll(1)
                for (int sb2 = 1; sb2 < 4; ++sb2) {
                    const int rs = r + (sb2 << 1), ks = k + (sb2 << 2);
                    NVP4V(w0, ks + 0, rs + 2, 1); NVP4V(w1, ks + 2, rs + 3, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 og0, ou0, og1, ou1;
                og0.v1 = acc_buf[0][0]; og0.v2 = acc_buf[0][1]; ou0.v1 = acc_buf[0][2]; ou0.v2 = acc_buf[0][3];
                og1.v1 = acc_buf[1][0]; og1.v2 = acc_buf[1][1]; ou1.v1 = acc_buf[1][2]; ou1.v2 = acc_buf[1][3];
                og0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, og0, 0); ou0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, ou0, 0);
                og1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, og1, 0); ou1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, ou1, 0);
                float128 og2, ou2, og3, ou3;
                og2.v1 = acc_buf[2][0]; og2.v2 = acc_buf[2][1]; ou2.v1 = acc_buf[2][2]; ou2.v2 = acc_buf[2][3];
                og3.v1 = acc_buf[3][0]; og3.v2 = acc_buf[3][1]; ou3.v1 = acc_buf[3][2]; ou3.v2 = acc_buf[3][3];
                acc_buf[0][0] = og0.v1; acc_buf[0][1] = og0.v2; acc_buf[0][2] = ou0.v1; acc_buf[0][3] = ou0.v2;
                acc_buf[1][0] = og1.v1; acc_buf[1][1] = og1.v2; acc_buf[1][2] = ou1.v1; acc_buf[1][3] = ou1.v2;
                og2 = v_f32_mac_x2_b(bg2, sc.v1, sc.v3, og2, 0); ou2 = v_f32_mac_x2_b(bu2, sc.v2, sc.v4, ou2, 0);
                og3 = v_f32_mac_x2_b(bg3, sc.v1, sc.v3, og3, 0); ou3 = v_f32_mac_x2_b(bu3, sc.v2, sc.v4, ou3, 0);
                acc_buf[2][0] = og2.v1; acc_buf[2][1] = og2.v2; acc_buf[2][2] = ou2.v1; acc_buf[2][3] = ou2.v2;
                acc_buf[3][0] = og3.v1; acc_buf[3][1] = og3.v2; acc_buf[3][2] = ou3.v1; acc_buf[3][3] = ou3.v2;
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                float128 bg2 = {0}, bu2 = {0}, bg3 = {0}, bu3 = {0};
                NVP4V(w0, k + 0,  r + 2, 1); NVP4V(w1, k + 2,  r + 3, 1);
                NVP4V(w0, k + 4,  r + 4, 1); NVP4V(w1, k + 6,  r + 5, 1);
                NVP4V(w0, k + 8,  r + 6, 1); NVP4V(w1, k + 10, r + 7, 1);
                NVP4V(w0, k + 12, 0,     0); NVP4V(w1, k + 14, 0,     0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                REDUCE_ACC_VLM(0, bg0, bu0, sc); REDUCE_ACC_VLM(1, bg1, bu1, sc);
                REDUCE_ACC_VLM(2, bg2, bu2, sc); REDUCE_ACC_VLM(3, bg3, bu3, sc);
            }
#undef NVP4V
        }
#elif defined(MXF8)
        int5 bc = {n0, 0, e, 0, 0};
        if (nm == 1) {
            float128 g0 = {0}, u0 = {0};
            int5 xc0 = {0, t0, 0, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 x0 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc0[0] = 1; bfloat128 x1 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc0[0] = 2; bfloat128 x2 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc0[0] = 3; bfloat128 x3 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
#define MXP1(WW, XX, K_NEXT, DO_W, DO_X) do { \
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
                if (DO_X) { xc0[0] = (K_NEXT); (XX) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); } \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 bg = {0}, bu = {0};
                MXP1(w0, x0, k0 + 4, 1, 1);  MXP1(w1, x1, k0 + 5, 1, 1);
                MXP1(w2, x2, k0 + 6, 1, 1);  MXP1(w3, x3, k0 + 7, 1, 1);
                MXP1(w0, x0, k0 + 8, 1, 1);  MXP1(w1, x1, k0 + 9, 1, 1);
                MXP1(w2, x2, k0 + 10, 1, 1); MXP1(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 3; ++sub) {
                    const int ks = k0 + (sub << 3);
                    MXP1(w0, x0, ks + 4, 1, 1);  MXP1(w1, x1, ks + 5, 1, 1);
                    MXP1(w2, x2, ks + 6, 1, 1);  MXP1(w3, x3, ks + 7, 1, 1);
                    MXP1(w0, x0, ks + 8, 1, 1);  MXP1(w1, x1, ks + 9, 1, 1);
                    MXP1(w2, x2, ks + 10, 1, 1); MXP1(w3, x3, ks + 11, 1, 1);
                }
                MXP1(w0, x0, k0 + 28, 1, 1); MXP1(w1, x1, k0 + 29, 1, 1);
                MXP1(w2, x2, k0 + 30, 1, 1); MXP1(w3, x3, k0 + 31, 1, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                MXP1(w0, x0, k0 + 32, 1, 1); MXP1(w1, x1, k0 + 33, 1, 1);
                MXP1(w2, x2, k0 + 34, 1, 1); MXP1(w3, x3, k0 + 35, 1, 1);
                g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
                u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 bg = {0}, bu = {0};
                MXP1(w0, x0, k0 + 4, 1, 1);  MXP1(w1, x1, k0 + 5, 1, 1);
                MXP1(w2, x2, k0 + 6, 1, 1);  MXP1(w3, x3, k0 + 7, 1, 1);
                MXP1(w0, x0, k0 + 8, 1, 1);  MXP1(w1, x1, k0 + 9, 1, 1);
                MXP1(w2, x2, k0 + 10, 1, 1); MXP1(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 3; ++sub) {
                    const int ks = k0 + (sub << 3);
                    MXP1(w0, x0, ks + 4, 1, 1);  MXP1(w1, x1, ks + 5, 1, 1);
                    MXP1(w2, x2, ks + 6, 1, 1);  MXP1(w3, x3, ks + 7, 1, 1);
                    MXP1(w0, x0, ks + 8, 1, 1);  MXP1(w1, x1, ks + 9, 1, 1);
                    MXP1(w2, x2, ks + 10, 1, 1); MXP1(w3, x3, ks + 11, 1, 1);
                }
                MXP1(w0, x0, k0 + 28, 1, 1); MXP1(w1, x1, k0 + 29, 1, 1);
                MXP1(w2, x2, k0 + 30, 1, 1); MXP1(w3, x3, k0 + 31, 1, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                MXP1(w0, x0, 0, 0, 0); MXP1(w1, x1, 0, 0, 0);
                MXP1(w2, x2, 0, 0, 0); MXP1(w3, x3, 0, 0, 0);
                g0 = v_f32_mac_x2_b(bg, sc.v1, sc.v3, g0, 0);
                u0 = v_f32_mac_x2_b(bu, sc.v2, sc.v4, u0, 0);
            }
#undef MXP1
            STORE_ACC1(0, g0, u0);
        } else if (nm == 2) {
            float128 g0 = {0}, u0 = {0}, g1 = {0}, u1 = {0};
            int5 xc0 = {0, t0, 0, 0, 0};
            int5 xc1 = {0, t1, 0, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define MXP2(WW, KK, K_NEXT, DO_W) do { \
                const uchar256 e_ = (WW) & (uchar256)0x78; \
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
                const uchar256 wa_ = (WW) - d8_; \
                if (DO_W) { wc[1] = (K_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const uchar256 d80_ = d8_ << 4; \
                xc0[0] = (KK); const bfloat128 xa_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); \
                xc1[0] = (KK); const bfloat128 xb_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); \
                const bfloat128_pair_t p_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
                const ushort128_pair_t d_ = v_convert_u8_to_u16_all_b(d80_); \
                const ushort128 r1_ = *((ushort128*)&p_.v1) + d_.v1; \
                const ushort128 r2_ = *((ushort128*)&p_.v2) + d_.v2; \
                bg0 = v_bf16_mac_acc32_b(*((bfloat128*)&r1_), xa_, bg0, 0); \
                bu0 = v_bf16_mac_acc32_b(*((bfloat128*)&r2_), xa_, bu0, 0); \
                bg1 = v_bf16_mac_acc32_b(*((bfloat128*)&r1_), xb_, bg1, 0); \
                bu1 = v_bf16_mac_acc32_b(*((bfloat128*)&r2_), xb_, bu1, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                MXP2(w0, k0 + 0, k0 + 4, 1); MXP2(w1, k0 + 1, k0 + 5, 1);
                MXP2(w2, k0 + 2, k0 + 6, 1); MXP2(w3, k0 + 3, k0 + 7, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 8; ++sub) {
                    const int ks = k0 + (sub << 2);
                    MXP2(w0, ks + 0, ks + 4, 1); MXP2(w1, ks + 1, ks + 5, 1);
                    MXP2(w2, ks + 2, ks + 6, 1); MXP2(w3, ks + 3, ks + 7, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                MXP2(w0, k0 + 0, k0 + 4, 1); MXP2(w1, k0 + 1, k0 + 5, 1);
                MXP2(w2, k0 + 2, k0 + 6, 1); MXP2(w3, k0 + 3, k0 + 7, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 7; ++sub) {
                    const int ks = k0 + (sub << 2);
                    MXP2(w0, ks + 0, ks + 4, 1); MXP2(w1, ks + 1, ks + 5, 1);
                    MXP2(w2, ks + 2, ks + 6, 1); MXP2(w3, ks + 3, ks + 7, 1);
                }
                MXP2(w0, k0 + 28, 0, 0); MXP2(w1, k0 + 29, 0, 0);
                MXP2(w2, k0 + 30, 0, 0); MXP2(w3, k0 + 31, 0, 0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
            }
#undef MXP2
            STORE_ACC1(0, g0, u0); STORE_ACC1(1, g1, u1);
        } else if (nm == 3) {
            int5 xc0 = {0, t0, 0, 0, 0};
            int5 xc1 = {0, t1, 0, 0, 0};
            int5 xc2 = {0, t2, 0, 0, 0};
            float128 g0 = {0}, u0 = {0}, g1 = {0}, u1 = {0};
            ZERO_ACC_VLM(2);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc1[0] = 0; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x));
            xc2[0] = 0; bfloat128 xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x));
            bfloat128 p1, p2;
            {
                const uchar256 e_ = w0 & (uchar256)0x78;
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT);
                const uchar256 wa_ = w0 - d8_;
                wc[1] = 2; w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
                const uchar256 d80_ = d8_ << 4;
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_));
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_);
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1;
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2;
                p1 = *((bfloat128*)&r1_); p2 = *((bfloat128*)&r2_);
            }
#define MXP3_SWP(P1, P2, WW, Q1, Q2, KX_NEXT, KW_NEXT, DO_W, DO_X) do { \
                const uchar256 e_ = (WW) & (uchar256)0x78; \
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
                const uchar256 wa_ = (WW) - d8_; \
                if (DO_W) { wc[1] = (KW_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const uchar256 d80_ = d8_ << 4; \
                bg0 = v_bf16_mac_acc32_b((P1), xa, bg0, 0); bu0 = v_bf16_mac_acc32_b((P2), xa, bu0, 0); \
                if (DO_X) { xc0[0] = (KX_NEXT); xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); } \
                bg1 = v_bf16_mac_acc32_b((P1), xb, bg1, 0); bu1 = v_bf16_mac_acc32_b((P2), xb, bu1, 0); \
                if (DO_X) { xc1[0] = (KX_NEXT); xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); } \
                bg2 = v_bf16_mac_acc32_b((P1), xc, bg2, 0); bu2 = v_bf16_mac_acc32_b((P2), xc, bu2, 0); \
                if (DO_X) { xc2[0] = (KX_NEXT); xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x)); } \
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_); \
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1; \
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2; \
                (Q1) = *((bfloat128*)&r1_); (Q2) = *((bfloat128*)&r2_); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0}, bg2 = {0}, bu2 = {0};
                bfloat128 q1, q2;
                MXP3_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXP3_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 16; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXP3_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXP3_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 og2, ou2;
                og2.v1 = acc_buf[2][0]; og2.v2 = acc_buf[2][1];
                ou2.v1 = acc_buf[2][2]; ou2.v2 = acc_buf[2][3];
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
                og2 = v_f32_mac_x2_b(bg2, sc.v1, sc.v3, og2, 0); ou2 = v_f32_mac_x2_b(bu2, sc.v2, sc.v4, ou2, 0);
                acc_buf[2][0] = og2.v1; acc_buf[2][1] = og2.v2;
                acc_buf[2][2] = ou2.v1; acc_buf[2][3] = ou2.v2;
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0}, bg2 = {0}, bu2 = {0};
                bfloat128 q1, q2;
                MXP3_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXP3_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 14; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXP3_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXP3_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                MXP3_SWP(p1, p2, w1, q1, q2, k0 + 29, k0 + 31, 1, 1);
                MXP3_SWP(q1, q2, w0, p1, p2, k0 + 30, 0,       0, 1);
                MXP3_SWP(p1, p2, w1, q1, q2, k0 + 31, 0,       0, 1);
                {
                    bg0 = v_bf16_mac_acc32_b(q1, xa, bg0, 0); bu0 = v_bf16_mac_acc32_b(q2, xa, bu0, 0);
                    bg1 = v_bf16_mac_acc32_b(q1, xb, bg1, 0); bu1 = v_bf16_mac_acc32_b(q2, xb, bu1, 0);
                    bg2 = v_bf16_mac_acc32_b(q1, xc, bg2, 0); bu2 = v_bf16_mac_acc32_b(q2, xc, bu2, 0);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                g0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, g0, 0); u0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, u0, 0);
                g1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, g1, 0); u1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, u1, 0);
                REDUCE_ACC_VLM(2, bg2, bu2, sc);
            }
#undef MXP3_SWP
            STORE_ACC1(0, g0, u0); STORE_ACC1(1, g1, u1);
        } else {
            int5 xc0 = {0, t0, 0, 0, 0};
            int5 xc1 = {0, t1, 0, 0, 0};
            int5 xc2 = {0, t2, 0, 0, 0};
            int5 xc3 = {0, t3, 0, 0, 0};
            ZERO_ACC_VLM(0); ZERO_ACC_VLM(1); ZERO_ACC_VLM(2); ZERO_ACC_VLM(3);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x));
            xc1[0] = 0; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x));
            xc2[0] = 0; bfloat128 xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x));
            xc3[0] = 0; bfloat128 xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, x));
            bfloat128 p1, p2;
            {
                const uchar256 e_ = w0 & (uchar256)0x78;
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT);
                const uchar256 wa_ = w0 - d8_;
                wc[1] = 2; w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
                const uchar256 d80_ = d8_ << 4;
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_));
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_);
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1;
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2;
                p1 = *((bfloat128*)&r1_); p2 = *((bfloat128*)&r2_);
            }
#define MXP4_SWP(P1, P2, WW, Q1, Q2, KX_NEXT, KW_NEXT, DO_W, DO_X) do { \
                const uchar256 e_ = (WW) & (uchar256)0x78; \
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
                const uchar256 wa_ = (WW) - d8_; \
                if (DO_W) { wc[1] = (KW_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const uchar256 d80_ = d8_ << 4; \
                bg0 = v_bf16_mac_acc32_b((P1), xa, bg0, 0); bu0 = v_bf16_mac_acc32_b((P2), xa, bu0, 0); \
                if (DO_X) { xc0[0] = (KX_NEXT); xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, x)); } \
                bg1 = v_bf16_mac_acc32_b((P1), xb, bg1, 0); bu1 = v_bf16_mac_acc32_b((P2), xb, bu1, 0); \
                if (DO_X) { xc1[0] = (KX_NEXT); xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, x)); } \
                bg2 = v_bf16_mac_acc32_b((P1), xc, bg2, 0); bu2 = v_bf16_mac_acc32_b((P2), xc, bu2, 0); \
                if (DO_X) { xc2[0] = (KX_NEXT); xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, x)); } \
                bg3 = v_bf16_mac_acc32_b((P1), xd, bg3, 0); bu3 = v_bf16_mac_acc32_b((P2), xd, bu3, 0); \
                if (DO_X) { xc3[0] = (KX_NEXT); xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, x)); } \
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_); \
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1; \
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2; \
                (Q1) = *((bfloat128*)&r1_); (Q2) = *((bfloat128*)&r2_); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                float128 bg2 = {0}, bu2 = {0}, bg3 = {0}, bu3 = {0};
                bfloat128 q1, q2;
                MXP4_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXP4_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 16; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXP4_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXP4_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 og0, ou0, og1, ou1;
                og0.v1 = acc_buf[0][0]; og0.v2 = acc_buf[0][1]; ou0.v1 = acc_buf[0][2]; ou0.v2 = acc_buf[0][3];
                og1.v1 = acc_buf[1][0]; og1.v2 = acc_buf[1][1]; ou1.v1 = acc_buf[1][2]; ou1.v2 = acc_buf[1][3];
                og0 = v_f32_mac_x2_b(bg0, sc.v1, sc.v3, og0, 0); ou0 = v_f32_mac_x2_b(bu0, sc.v2, sc.v4, ou0, 0);
                og1 = v_f32_mac_x2_b(bg1, sc.v1, sc.v3, og1, 0); ou1 = v_f32_mac_x2_b(bu1, sc.v2, sc.v4, ou1, 0);
                float128 og2, ou2, og3, ou3;
                og2.v1 = acc_buf[2][0]; og2.v2 = acc_buf[2][1]; ou2.v1 = acc_buf[2][2]; ou2.v2 = acc_buf[2][3];
                og3.v1 = acc_buf[3][0]; og3.v2 = acc_buf[3][1]; ou3.v1 = acc_buf[3][2]; ou3.v2 = acc_buf[3][3];
                acc_buf[0][0] = og0.v1; acc_buf[0][1] = og0.v2; acc_buf[0][2] = ou0.v1; acc_buf[0][3] = ou0.v2;
                acc_buf[1][0] = og1.v1; acc_buf[1][1] = og1.v2; acc_buf[1][2] = ou1.v1; acc_buf[1][3] = ou1.v2;
                og2 = v_f32_mac_x2_b(bg2, sc.v1, sc.v3, og2, 0); ou2 = v_f32_mac_x2_b(bu2, sc.v2, sc.v4, ou2, 0);
                og3 = v_f32_mac_x2_b(bg3, sc.v1, sc.v3, og3, 0); ou3 = v_f32_mac_x2_b(bu3, sc.v2, sc.v4, ou3, 0);
                acc_buf[2][0] = og2.v1; acc_buf[2][1] = og2.v2; acc_buf[2][2] = ou2.v1; acc_buf[2][3] = ou2.v2;
                acc_buf[3][0] = og3.v1; acc_buf[3][1] = og3.v2; acc_buf[3][2] = ou3.v1; acc_buf[3][3] = ou3.v2;
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 bg0 = {0}, bu0 = {0}, bg1 = {0}, bu1 = {0};
                float128 bg2 = {0}, bu2 = {0}, bg3 = {0}, bu3 = {0};
                bfloat128 q1, q2;
                MXP4_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXP4_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 14; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXP4_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXP4_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                MXP4_SWP(p1, p2, w1, q1, q2, k0 + 29, k0 + 31, 1, 1);
                MXP4_SWP(q1, q2, w0, p1, p2, k0 + 30, 0,       0, 1);
                MXP4_SWP(p1, p2, w1, q1, q2, k0 + 31, 0,       0, 1);
                {
                    bg0 = v_bf16_mac_acc32_b(q1, xa, bg0, 0); bu0 = v_bf16_mac_acc32_b(q2, xa, bu0, 0);
                    bg1 = v_bf16_mac_acc32_b(q1, xb, bg1, 0); bu1 = v_bf16_mac_acc32_b(q2, xb, bu1, 0);
                    bg2 = v_bf16_mac_acc32_b(q1, xc, bg2, 0); bu2 = v_bf16_mac_acc32_b(q2, xc, bu2, 0);
                    bg3 = v_bf16_mac_acc32_b(q1, xd, bg3, 0); bu3 = v_bf16_mac_acc32_b(q2, xd, bu3, 0);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                REDUCE_ACC_VLM(0, bg0, bu0, sc); REDUCE_ACC_VLM(1, bg1, bu1, sc);
                REDUCE_ACC_VLM(2, bg2, bu2, sc); REDUCE_ACC_VLM(3, bg3, bu3, sc);
            }
#undef MXP4_SWP
        }
#endif

        // Shared epilogue: load 4 row-scale vectors once, run 4-way SwiGLU4 per token pair (+ 2-way for odd token)
        int5 sc = {n0, e, 0, 0, 0};
        const float64 si0 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 64;
        const float64 si1 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 128;
        const float64 si2 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 192;
        const float64 si3 = v_f32_ld_tnsr_b(sc, si);
        const int toks[4] = {t0, t1, t2, t3};
        const int slots[4] = {s0, s1, s2, s3};
        int m = 0;
#pragma loop_unroll(1)
        for (; m + 1 < nm; m += 2) {
            const float64 ge0 = bf16r_v(acc_buf[m][0]     * si0);
            const float64 go0 = bf16r_v(acc_buf[m][1]     * si1);
            const float64 ue0 = bf16r_v(acc_buf[m][2]     * si2);
            const float64 uo0 = bf16r_v(acc_buf[m][3]     * si3);
            const float64 ge1 = bf16r_v(acc_buf[m + 1][0] * si0);
            const float64 go1 = bf16r_v(acc_buf[m + 1][1] * si1);
            const float64 ue1 = bf16r_v(acc_buf[m + 1][2] * si2);
            const float64 uo1 = bf16r_v(acc_buf[m + 1][3] * si3);
            bfloat128 va0, va1;
            SWIGLU4_4(ge0, go0, ge1, go1, ue0, uo0, ue1, uo1, va0, va1);
            int5 oc0 = {n0 >> 1, slots[m],     toks[m],     0, 0};
            int5 oc1 = {n0 >> 1, slots[m + 1], toks[m + 1], 0, 0};
            v_bf16_st_tnsr(oc0, act, va0);
            v_bf16_st_tnsr(oc1, act, va1);
        }
        if (m < nm) {
            const float64 ge0 = bf16r_v(acc_buf[m][0] * si0);
            const float64 go0 = bf16r_v(acc_buf[m][1] * si1);
            const float64 ue0 = bf16r_v(acc_buf[m][2] * si2);
            const float64 uo0 = bf16r_v(acc_buf[m][3] * si3);
            bfloat128 va0;
            SWIGLU4_2(ge0, go0, ue0, uo0, va0);
            int5 oc0 = {n0 >> 1, slots[m], toks[m], 0, 0};
            v_bf16_st_tnsr(oc0, act, va0);
        }
    }
}
