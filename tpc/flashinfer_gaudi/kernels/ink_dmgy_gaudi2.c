// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode message gate output (ink_mg_out OUT) in two tile-parallel phases, one 128-channel tile of every gate
// per index point:
//  phase A (this kernel): y_g = (1 - tanh(f_g)) * h_g + tanh(gn_g) * m_(m0+g)  -> y f32 [H, G] (split order) and the
//                         partial sums of y_g^2 of the tile -> partials f32 [64, H / 128] (lane g)
//  phase B (ink_dmgn):    out_g = bf16((y_g * rsqrt(sum / H + eps)) * w_g)
// Tensors (dim 0 fastest): h f32 [H, T, Gh] (split order; Gh = 1: one state for every gate), m f32 [H, T, 4], fg f32
// [2H, T, G] ([f | gn] of gate g, split order per 128), y f32 [H, T, G], partials f32 [64, NT, T].
// Index space [H / 128, T].
// Params: m0.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

static inline float64_pair_t ld_pair(tensor x, int col, int a, int b) {
    float64_pair_t p;
    int5 c = {col, a, b, 0, 0};
    p.v1 = v_f32_ld_tnsr_b(c, x);
    c[0] = col + 64;
    p.v2 = v_f32_ld_tnsr_b(c, x);
    return p;
}

void main(tensor h, tensor m, tensor fg, tensor y, tensor partials, int m0)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(h, 0);
    const int Gh = get_dim_size(h, 2);
    const int G = get_dim_size(y, 2);
    const float64 lanes = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), 0);
    for (int r = start[1]; r < end[1]; ++r)
    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        float64 acc = 0.0f;
        for (int g = 0; g < G; ++g) {
            const float64_pair_t a = ld_pair(h, c, r, Gh == 1 ? 0 : g);
            const float64_pair_t b = ld_pair(m, c, r, m0 + g);
            const float64_pair_t f = ld_pair(fg, c, r, g);
            const float64_pair_t q = ld_pair(fg, H + c, r, g);
            float64_pair_t yy;
            yy.v1 = (1.0f - ink_tanh(f.v1)) * a.v1 + ink_tanh(q.v1) * b.v1;
            yy.v2 = (1.0f - ink_tanh(f.v2)) * a.v2 + ink_tanh(q.v2) * b.v2;
            int5 yc = {c, r, g, 0, 0};
            v_f32_st_tnsr(yc, y, yy.v1);
            yc[0] = c + 64;
            v_f32_st_tnsr(yc, y, yy.v2);
            const float64 sq = row_sum(yy.v1 * yy.v1 + yy.v2 * yy.v2);
            acc = v_f32_sel_eq_f32_b(lanes, (float)g, sq, acc);
        }
        int5 pc = {0, tile, r, 0, 0};
        v_f32_st_tnsr(pc, partials, acc);
    }
}
