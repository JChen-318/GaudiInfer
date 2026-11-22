// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 fused indexed gated residual:
//   out[row, :] = bf16(f32(residual[row, :]) + f32(gate[indices[row], :]) * f32(branch[row, :]))
void main(tensor residual, tensor branch, tensor gate, tensor indices, tensor out)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(residual, 0);
    const int tiles = width / 128;

    for (int row = start[0]; row < end[0]; ++row) {
        const int5 idx_coord = {row, 0, 0, 0, 0};
        const int idx = s_i32_ld_g(gen_addr(idx_coord, indices));

        #pragma loop_unroll(2)
        for (int tile = 0; tile < tiles; ++tile) {
            const int col = tile * 128;
            const int5 coord = {col, row, 0, 0, 0};
            const int5 gcoord = {col, idx, 0, 0, 0};
            const float64_pair_t rf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coord, residual));
            const float64_pair_t bf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coord, branch));
            const float64_pair_t gf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gcoord, gate));
            float64_pair_t of;
            of.v1 = v_f32_mac_b(gf.v1, bf.v1, rf.v1);
            of.v2 = v_f32_mac_b(gf.v2, bf.v2, rf.v2);
            v_bf16_st_tnsr(coord, out, v_convert_f32_to_bf16_all_b(of));
        }
    }
}
