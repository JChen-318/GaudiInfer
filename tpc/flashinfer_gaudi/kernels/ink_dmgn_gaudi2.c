// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode message gate output, phase B (see ink_dmgy_gaudi2.c): out_g = bf16((y_g * rsqrt(sum_g / H + eps)) * w_g)
// with sum_g the summed lane g of the row's partials. Tensors (dim 0 fastest): partials f32 [64, NT, T], y f32
// [H, T, G] (split order), w bf16 [H, G], out bf16 [H, T, G]. Index space [H / 128, T]. Params: eps.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

void main(tensor partials, tensor y, tensor w, tensor out, float eps)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(y, 0);
    const int G = get_dim_size(y, 2);
    const int NT = get_dim_size(partials, 1);
    for (int row = start[1]; row < end[1]; ++row) {
    float64 tot = 0.0f, tot2 = 0.0f;
    int i = 0;
    for (; i + 1 < NT; i += 2) {
        int5 pc = {0, i, row, 0, 0};
        tot = tot + v_f32_ld_tnsr_b(pc, partials);
        pc[1] = i + 1;
        tot2 = tot2 + v_f32_ld_tnsr_b(pc, partials);
    }
    if (i < NT) {
        int5 pc = {0, i, row, 0, 0};
        tot = tot + v_f32_ld_tnsr_b(pc, partials);
    }
    // rsqrt for every lane at once (lane g = gate g)
    const float64 r = positive_rsqrt((tot + tot2) * ink_srecip((float)H) + eps);
    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        for (int g = 0; g < G; ++g) {
            const float64 rg = v_broadcast_element_f32(r, g);
            int5 yc = {c, row, g, 0, 0};
            float64_pair_t yy;
            yy.v1 = v_f32_ld_tnsr_b(yc, y);
            yc[0] = c + 64;
            yy.v2 = v_f32_ld_tnsr_b(yc, y);
            int5 wc = {c, g, 0, 0, 0};
            const float64_pair_t wv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
            float64_pair_t o;
            o.v1 = (yy.v1 * rg) * wv.v1;
            o.v2 = (yy.v2 * rg) * wv.v2;
            int5 oc = {c, row, g, 0, 0};
            v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(o));
        }
    }
    }
}
