// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode / verification (T rows, Direct engine), end of layer application with HF rounding order:
//   s_r = bf16(sum_j f32(ys[r, j]) * rw[r, j])   (route weight applied AFTER down-proj BF16 rounding!)
//   m2 = conv(s), h_out_r = bf16(res_r + f32(m2_r))
// Tensors (dim 0 fastest): ys bf16 [H, J, T], rw f32 [J, T], st bf16 [H, 3], taps bf16 [H, 4], res f32 [H, T],
// st_out bf16 [H, 3] (or [H, 3 + T]), h_out bf16 [H, T], h_out2 bf16 [H, T]. Index space [H / 128]. Params: cscale.
#include "ink_dmix_gaudi2.h"

void main(tensor ys, tensor rw, tensor st, tensor taps, tensor res, tensor st_out, tensor h_out, tensor h_out2, float cscale)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int J = get_dim_size(ys, 1);
    const int T = get_dim_size(ys, 2);
    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        dmix_conv_t cv = dmix_conv_init(st, taps, st_out, c);
        for (int r = 0; r < T; ++r) {
            float64_pair_t sm;
            sm.v1 = 0.0f; sm.v2 = 0.0f;
            for (int j = 0; j < J; ++j) {
                int5 rc = {j, r, 0, 0, 0};
                const float wj = s_f32_ld_g((__global__ float*)gen_addr(rc, rw));
                int5 jc = {c, j, r, 0, 0};
                const float64_pair_t a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys));
                sm.v1 = v_f32_mac_b(a.v1, wj, sm.v1);
                sm.v2 = v_f32_mac_b(a.v2, wj, sm.v2);
            }
            cv = dmix_conv_step(cv, v_convert_f32_to_bf16_all_b(sm), st_out, c, r, cscale);
            const float64_pair_t mv = v_convert_bf16_to_f32_all_b(cv.y);
            const float64_pair_t rv = dmix_f32_tile_r(res, c, r);
            float64_pair_t o;
            o.v1 = rv.v1 + mv.v1;
            o.v2 = rv.v2 + mv.v2;
            int5 oc = {c, r, 0, 0, 0};
            const bfloat128 ob = v_convert_f32_to_bf16_all_b(o);
            v_bf16_st_tnsr(oc, h_out, ob);
            v_bf16_st_tnsr(oc, h_out2, ob);
        }
        dmix_conv_finish(cv, st_out, c);
    }
}
