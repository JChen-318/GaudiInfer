// SPDX-License-Identifier: Apache-2.0
// Inkling-2 prefill MoE BF16 activation (Direct engine, table-free):
// For every slot in a wave's list, reads F32 gate/up products `gu` (1 row per slot; per 256 columns
// [gate 0..127 | up 0..127] matching ink_nvf4_deq_bf16_gaudi2), multiplies by per-row F32 `ws` (nv_gvec13),
// BF16-rounds pre-activations, computes [9/8] Pade SwiGLU4 in 4-way lockstep, and writes natural-order BF16
// activations `ab` [F, R].
// Tensors (dim 0 fastest): gu f32 [2F, R], code i32 [S], idx i32 [S], ws f32 [2F, E], ab bf16 [F, R],
// list i32 [S], cnt i32 [24], tpart i32 [25]. Index space [24]. Params: cu, lk.
#include "ink_math_gaudi2.h"
__local__ int lstart[25];

// 4-way lockstep f32->bf16 pack into two natural-order bfloat128 vectors (same as ink_sact2)
#define F32_4_TO_BF16_2(Q0, Q1, Q2, Q3, B0, B1) do { \
        const bfloat128 sp0_ = v_convert_f32_to_bf16_b((Q0), 0, SW_RHNE, (bfloat128)0); \
        const bfloat128 sp1_ = v_convert_f32_to_bf16_b((Q1), 0, SW_RHNE, (bfloat128)0); \
        const bfloat128 sp2_ = v_convert_f32_to_bf16_b((Q2), 0, SW_RHNE, (bfloat128)0); \
        const bfloat128 sp3_ = v_convert_f32_to_bf16_b((Q3), 0, SW_RHNE, (bfloat128)0); \
        bfloat128 pk0_ = v_bf16_pack_b(sp0_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0); \
        bfloat128 pk1_ = v_bf16_pack_b(sp1_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0); \
        bfloat128 pk2_ = v_bf16_pack_b(sp2_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0); \
        bfloat128 pk3_ = v_bf16_pack_b(sp3_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0); \
        pk0_ = v_bf16_pack_b(sp0_, SW_GROUP_1 | SW_STRIDE_2, pk0_); \
        pk1_ = v_bf16_pack_b(sp1_, SW_GROUP_1 | SW_STRIDE_2, pk1_); \
        pk2_ = v_bf16_pack_b(sp2_, SW_GROUP_1 | SW_STRIDE_2, pk2_); \
        pk3_ = v_bf16_pack_b(sp3_, SW_GROUP_1 | SW_STRIDE_2, pk3_); \
        bfloat128 lo0_ = v_bf16_mov_dual_group_pack_b(pk0_, SW_PACK21, (bfloat128)0); \
        const bfloat128 lo1_ = v_bf16_mov_dual_group_pack_b(pk1_, SW_PACK21, (bfloat128)0); \
        bfloat128 lo2_ = v_bf16_mov_dual_group_pack_b(pk2_, SW_PACK21, (bfloat128)0); \
        const bfloat128 lo3_ = v_bf16_mov_dual_group_pack_b(pk3_, SW_PACK21, (bfloat128)0); \
        lo0_ = v_bf16_mov_dual_group_b(lo1_, 0xffffffff, 0, 2, MkWr(1, 1), lo0_); \
        lo2_ = v_bf16_mov_dual_group_b(lo3_, 0xffffffff, 0, 2, MkWr(1, 1), lo2_); \
        (B0) = v_bf16_mov_dual_group_b(lo1_, 0xffffffff, 1, 3, MkWr(1, 1), lo0_); \
        (B1) = v_bf16_mov_dual_group_b(lo3_, 0xffffffff, 1, 3, MkWr(1, 1), lo2_); \
    } while (0)

#define F32_2_TO_BF16_1(Q0, Q1, B0) do { \
        const bfloat128 sp0_ = v_convert_f32_to_bf16_b((Q0), 0, SW_RHNE, (bfloat128)0); \
        const bfloat128 sp1_ = v_convert_f32_to_bf16_b((Q1), 0, SW_RHNE, (bfloat128)0); \
        bfloat128 pk0_ = v_bf16_pack_b(sp0_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0); \
        bfloat128 pk1_ = v_bf16_pack_b(sp1_, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0); \
        pk0_ = v_bf16_pack_b(sp0_, SW_GROUP_1 | SW_STRIDE_2, pk0_); \
        pk1_ = v_bf16_pack_b(sp1_, SW_GROUP_1 | SW_STRIDE_2, pk1_); \
        bfloat128 lo0_ = v_bf16_mov_dual_group_pack_b(pk0_, SW_PACK21, (bfloat128)0); \
        const bfloat128 lo1_ = v_bf16_mov_dual_group_pack_b(pk1_, SW_PACK21, (bfloat128)0); \
        lo0_ = v_bf16_mov_dual_group_b(lo1_, 0xffffffff, 0, 2, MkWr(1, 1), lo0_); \
        (B0) = v_bf16_mov_dual_group_b(lo1_, 0xffffffff, 1, 3, MkWr(1, 1), lo0_); \
    } while (0)

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

#define SWIGLU4_4_F32(G0, G1, G2, G3, U0, U1, U2, U3, O0, O1, O2, O3) do { \
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
        (O0) = nm0_ * r0_; (O1) = nm1_ * r1_; (O2) = nm2_ * r2_; (O3) = nm3_ * r3_; \
    } while (0)

#define SWIGLU4_2_F32(G0, G1, U0, U1, O0, O1) do { \
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
        (O0) = nm0_ * r0_; (O1) = nm1_ * r1_; \
    } while (0)

#define LOAD_GU128(COL, GP, UP) do { \
        int5 cg_ = {(COL), row, 0, 0, 0}; \
        int5 cw_ = {(COL), e, 0, 0, 0}; \
        float64_pair_t gr_, ur_; \
        gr_.v1 = v_f32_ld_tnsr_b(cg_, gu) * v_f32_ld_tnsr_b(cw_, ws); cg_[0] += 64; cw_[0] += 64; \
        gr_.v2 = v_f32_ld_tnsr_b(cg_, gu) * v_f32_ld_tnsr_b(cw_, ws); cg_[0] += 64; cw_[0] += 64; \
        ur_.v1 = v_f32_ld_tnsr_b(cg_, gu) * v_f32_ld_tnsr_b(cw_, ws); cg_[0] += 64; cw_[0] += 64; \
        ur_.v2 = v_f32_ld_tnsr_b(cg_, gu) * v_f32_ld_tnsr_b(cw_, ws); \
        (GP) = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(gr_)); \
        (UP) = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(ur_)); \
    } while (0)

void main(tensor gu, tensor code, tensor idx, tensor ws, tensor ab, tensor list, tensor cnt, tensor tpart, float cu, int lk)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int tiles = get_dim_size(ab, 0) >> 7;  // F / 128 = 9
    const int pairs = tiles & ~1;                // 8
    const float inv_cu = ink_srecip(cu), c2cu = 2.0f * cu;

    int total = 0;
    for (int q = 0; q < 24; ++q) {
        int5 qc = {q, 0, 0, 0, 0};
        lstart[q] = total;
        total += s_i32_ld_g((__global__ int*)gen_addr(qc, cnt));
    }
    lstart[24] = total;

    for (int p = start[0]; p < end[0]; ++p) {
        const int n0 = (total * p) / 24, n1 = (total * (p + 1)) / 24;
        int q = 0;
        while (q < 23 && lstart[q + 1] <= n0) ++q;
        int5 tq = {q, 0, 0, 0, 0};
        int base = s_i32_ld_g((__global__ int*)gen_addr(tq, tpart)) << lk;
        for (int n = n0; n < n1; ++n) {
            while (n >= lstart[q + 1]) {
                ++q;
                tq[0] = q;
                base = s_i32_ld_g((__global__ int*)gen_addr(tq, tpart)) << lk;
            }
            int5 lc = {base + n - lstart[q], 0, 0, 0, 0};
            const int s = s_i32_ld_g((__global__ int*)gen_addr(lc, list));
            int5 sc = {s, 0, 0, 0, 0};
            const int row = s_i32_ld_g((__global__ int*)gen_addr(sc, code));
            const int e = s_i32_ld_g((__global__ int*)gen_addr(sc, idx));
            int5 oc = {0, row, 0, 0, 0};
            for (int tl = 0; tl < pairs; tl += 2) {
                float64_pair_t gp0, up0, gp1, up1;
                LOAD_GU128(tl * 256, gp0, up0);
                LOAD_GU128(tl * 256 + 256, gp1, up1);
                float64 o0, o1, o2, o3;
                SWIGLU4_4_F32(gp0.v1, gp0.v2, gp1.v1, gp1.v2, up0.v1, up0.v2, up1.v1, up1.v2, o0, o1, o2, o3);
                bfloat128 b0, b1;
                F32_4_TO_BF16_2(o0, o1, o2, o3, b0, b1);
                oc[0] = tl << 7;
                v_bf16_st_tnsr(oc, ab, b0);
                oc[0] = (tl + 1) << 7;
                v_bf16_st_tnsr(oc, ab, b1);
            }
            for (int tl = pairs; tl < tiles; ++tl) {
                float64_pair_t gp0, up0;
                LOAD_GU128(tl * 256, gp0, up0);
                float64 o0, o1;
                SWIGLU4_2_F32(gp0.v1, gp0.v2, up0.v1, up0.v2, o0, o1);
                bfloat128 b0;
                F32_2_TO_BF16_1(o0, o1, b0);
                oc[0] = tl << 7;
                v_bf16_st_tnsr(oc, ab, b0);
            }
        }
    }
}
