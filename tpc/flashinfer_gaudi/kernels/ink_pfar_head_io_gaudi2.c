// SPDX-License-Identifier: Apache-2.0
// Inkling-2 full-layer SRAM FlashAttention per-KV-head HBM <-> SRAM staging (Direct prefill):
//   - dir == 0 (load): copies head h's near output o [128, GT] F32, query q [128, GT] BF16,
//     running max m [GT] F32, and running sum l [GT] F32 from HBM into fixed SRAM buffers.
//   - dir == 1 (store): writes updated o [128, GT] F32 and l [GT] F32 from SRAM back to HBM.
// 100% table-free, 256-byte aligned full-vector transfers. Index space [24] (128 rows/core for GT = 3072).
// Tensors (dim 0 fastest):
//   0 o_hbm  f32  [128, GT, KV]
//   1 o_sram f32  [128, GT]
//   2 q_hbm  bf16 [128, GT, KV]
//   3 q_sram bf16 [128, GT]
//   4 m_hbm  f32  [GT, KV]
//   5 m_sram f32  [GT]
//   6 l_hbm  f32  [GT, KV]
//   7 l_sram f32  [GT]
// Params: int h, int dir.
void main(tensor o_hbm, tensor o_sram, tensor q_hbm, tensor q_sram,
          tensor m_hbm, tensor m_sram, tensor l_hbm, tensor l_sram,
          int h, int dir)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int GT = get_dim_size(m_sram, 0);
    const int rows_per = GT / 24;

    for (int cid = start[0]; cid < end[0]; ++cid) {
        const int r0 = cid * rows_per;
        const int r1 = r0 + rows_per;
        if (dir == 0) {
            for (int r = r0; r < r1; r += 64) {
                int5 hc = {r, h, 0, 0, 0};
                int5 sc = {r, 0, 0, 0, 0};
                v_f32_st_tnsr(sc, m_sram, v_f32_ld_tnsr_b(hc, m_hbm));
                v_f32_st_tnsr(sc, l_sram, v_f32_ld_tnsr_b(hc, l_hbm));
            }
            for (int r = r0; r < r1; r += 2) {
                int5 hc0 = {0, r, h, 0, 0}, hc1 = {64, r, h, 0, 0};
                int5 hc2 = {0, r + 1, h, 0, 0}, hc3 = {64, r + 1, h, 0, 0};
                const float64 o0 = v_f32_ld_tnsr_b(hc0, o_hbm);
                const float64 o1 = v_f32_ld_tnsr_b(hc1, o_hbm);
                const float64 o2 = v_f32_ld_tnsr_b(hc2, o_hbm);
                const float64 o3 = v_f32_ld_tnsr_b(hc3, o_hbm);
                int5 sc0 = {0, r, 0, 0, 0}, sc1 = {64, r, 0, 0, 0};
                int5 sc2 = {0, r + 1, 0, 0, 0}, sc3 = {64, r + 1, 0, 0, 0};
                v_f32_st_tnsr(sc0, o_sram, o0);
                v_f32_st_tnsr(sc1, o_sram, o1);
                v_f32_st_tnsr(sc2, o_sram, o2);
                v_f32_st_tnsr(sc3, o_sram, o3);
            }
            for (int r = r0; r < r1; r += 4) {
                int5 hc0 = {0, r, h, 0, 0}, hc1 = {0, r + 1, h, 0, 0};
                int5 hc2 = {0, r + 2, h, 0, 0}, hc3 = {0, r + 3, h, 0, 0};
                const bfloat128 q0 = v_bf16_ld_tnsr_b(hc0, q_hbm);
                const bfloat128 q1 = v_bf16_ld_tnsr_b(hc1, q_hbm);
                const bfloat128 q2 = v_bf16_ld_tnsr_b(hc2, q_hbm);
                const bfloat128 q3 = v_bf16_ld_tnsr_b(hc3, q_hbm);
                int5 sc0 = {0, r, 0, 0, 0}, sc1 = {0, r + 1, 0, 0, 0};
                int5 sc2 = {0, r + 2, 0, 0, 0}, sc3 = {0, r + 3, 0, 0, 0};
                v_bf16_st_tnsr(sc0, q_sram, q0);
                v_bf16_st_tnsr(sc1, q_sram, q1);
                v_bf16_st_tnsr(sc2, q_sram, q2);
                v_bf16_st_tnsr(sc3, q_sram, q3);
            }
        } else {
            for (int r = r0; r < r1; r += 64) {
                int5 hc = {r, h, 0, 0, 0};
                int5 sc = {r, 0, 0, 0, 0};
                v_f32_st_tnsr(hc, l_hbm, v_f32_ld_tnsr_b(sc, l_sram));
            }
            for (int r = r0; r < r1; r += 2) {
                int5 sc0 = {0, r, 0, 0, 0}, sc1 = {64, r, 0, 0, 0};
                int5 sc2 = {0, r + 1, 0, 0, 0}, sc3 = {64, r + 1, 0, 0, 0};
                const float64 o0 = v_f32_ld_tnsr_b(sc0, o_sram);
                const float64 o1 = v_f32_ld_tnsr_b(sc1, o_sram);
                const float64 o2 = v_f32_ld_tnsr_b(sc2, o_sram);
                const float64 o3 = v_f32_ld_tnsr_b(sc3, o_sram);
                int5 hc0 = {0, r, h, 0, 0}, hc1 = {64, r, h, 0, 0};
                int5 hc2 = {0, r + 1, h, 0, 0}, hc3 = {64, r + 1, h, 0, 0};
                v_f32_st_tnsr(hc0, o_hbm, o0);
                v_f32_st_tnsr(hc1, o_hbm, o1);
                v_f32_st_tnsr(hc2, o_hbm, o2);
                v_f32_st_tnsr(hc3, o_hbm, o3);
            }
        }
    }
}
