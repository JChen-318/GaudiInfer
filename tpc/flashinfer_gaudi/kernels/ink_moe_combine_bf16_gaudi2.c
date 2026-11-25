// SPDX-License-Identifier: Apache-2.0
// Inkling-2 prefill MoE BF16 slot combine (Direct engine, table-free):
//   y_j = bf16(cs[e_j] * Y[row_j])            (1 row per slot, natural column order)
//   out[t] = bf16(sum_{j=0..K-1} rw[t, j] * f32(y_j))
// Tensors (dim 0 fastest): y f32 [H, R], cs f32 [H, E], code i32 [K, T], idx i32 [K, T],
// rw f32 [K, T], out bf16 [H, T], vlen i32 [1]. Index space [T]. H % 256 == 0, K <= 8.
#define MAXK 8
__local__ float64 rwv[MAXK];
__local__ int sl_r[MAXK];
__local__ int sl_e[MAXK];

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

void main(tensor y, tensor cs, tensor code, tensor idx, tensor rw, tensor out, tensor vlen)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int K = get_dim_size(code, 0);
    const int NC = get_dim_size(out, 0) >> 8;  // H / 256 = 9
    int5 vc = {0, 0, 0, 0, 0};
    const int TV = s_i32_ld_g((__global__ int*)gen_addr(vc, vlen));
    const int t_end = end[0] < TV ? end[0] : TV;

    for (int t = start[0]; t < t_end; ++t) {
        for (int j = 0; j < K; ++j) {
            int5 cc = {j, t, 0, 0, 0};
            sl_r[j] = s_i32_ld_g((__global__ int*)gen_addr(cc, code));
            sl_e[j] = s_i32_ld_g((__global__ int*)gen_addr(cc, idx));
            rwv[j] = v_broadcast_element_f32(v_f32_ld_tnsr_b(cc, rw), 0);
        }
        for (int c = 0; c < NC; ++c) {
            const int col0 = c << 8;
            float64 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            for (int j = 0; j < K; ++j) {
                const float64 r = rwv[j];
                int5 cy = {col0, sl_r[j], 0, 0, 0};
                int5 ck = {col0, sl_e[j], 0, 0, 0};
                float64_pair_t p, q;
                p.v1 = v_f32_ld_tnsr_b(cy, y) * v_f32_ld_tnsr_b(ck, cs); cy[0] += 64; ck[0] += 64;
                p.v2 = v_f32_ld_tnsr_b(cy, y) * v_f32_ld_tnsr_b(ck, cs); cy[0] += 64; ck[0] += 64;
                q.v1 = v_f32_ld_tnsr_b(cy, y) * v_f32_ld_tnsr_b(ck, cs); cy[0] += 64; ck[0] += 64;
                q.v2 = v_f32_ld_tnsr_b(cy, y) * v_f32_ld_tnsr_b(ck, cs);
                const float64_pair_t pb = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(p));
                const float64_pair_t qb = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(q));
                a0 = v_f32_mac_b(pb.v1, r, a0);
                a1 = v_f32_mac_b(pb.v2, r, a1);
                a2 = v_f32_mac_b(qb.v1, r, a2);
                a3 = v_f32_mac_b(qb.v2, r, a3);
            }
            bfloat128 b0, b1;
            F32_4_TO_BF16_2(a0, a1, a2, a3, b0, b1);
            int5 oc = {col0, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, out, b0);
            oc[0] = col0 + 128;
            v_bf16_st_tnsr(oc, out, b1);
        }
    }
}
