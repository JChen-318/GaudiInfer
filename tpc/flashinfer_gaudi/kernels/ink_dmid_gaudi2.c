// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode / verification (T rows, Direct engine), after the output projection, one tile per index point:
//   a2 = conv(a1) over the T rows (post-attention short conv, ink_conv mode 0), res = f32(h_in) + f32(a2) (split order;
//   boundary layers, has_h = 0: res = f32(a2)), then phase 1 of the MLP depth attention over [blocks; res] per row.
// Tensors (dim 0 fastest): a1 bf16 [H, T], st bf16 [H, 3], taps bf16 [H, 4], h_in bf16 [H, T], blocks bf16 [H, T, S0],
// query f32 [H, 1], st_out bf16 [H, 3] (or [H, 3 + T]: the input history), res f32 [H, T], partials f32
// [64, H / 128, T]. Index space [H / 128]. Params: conv scale, has_h.
#include "ink_dmix_gaudi2.h"

void main(tensor a1, tensor st, tensor taps, tensor h_in, tensor blocks, tensor query, tensor st_out, tensor res,
          tensor partials, float cscale, int has_h)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int S0 = get_dim_size(blocks, 2);
    const int T = get_dim_size(a1, 1);
    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        dmix_conv_t cv = dmix_conv_init(st, taps, st_out, c);
        for (int r = 0; r < T; ++r) {
            int5 hc = {c, r, 0, 0, 0};
            cv = dmix_conv_step(cv, v_bf16_ld_tnsr_b(hc, a1), st_out, c, r, cscale);
            float64_pair_t rv = v_convert_bf16_to_f32_all_b(cv.y);
            if (has_h) {
                const float64_pair_t hv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(hc, h_in));
                rv.v1 = hv.v1 + rv.v1;
                rv.v2 = hv.v2 + rv.v2;
            }
            v_f32_st_tnsr(hc, res, rv.v1);
            hc[0] = c + 64;
            v_f32_st_tnsr(hc, res, rv.v2);
            dmix_states_t xs;
            for (int s = 0; s < S0; ++s) {
                int5 bc = {c, r, s, 0, 0};
                xs.v[s] = dmix_bf16_tile(blocks, bc);
            }
            xs.v[S0] = rv;
            dmix_gram_store(xs, S0 + 1, query, 1, c, partials, tile, r);
        }
        dmix_conv_finish(cv, st_out, c);
    }
}
