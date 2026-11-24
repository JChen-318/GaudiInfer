// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode / verification depth attention, phase 2 (ink_dmix_gaudi2.h): coefficients from the summed
// partials of the row, then the normalized mix of one tile per index point for each query.
// Tensors (dim 0 fastest): partials f32 [64, NT, T], blocks bf16 [H, T, S0], part bf16 [H, T] (F32P: f32 [H, T],
// split order), w bf16 [H, Q], normed bf16 [H, T, Q]. Index space [H / 128, T]. Params: scale (1 / sqrt(H)), eps,
// plain (1: no depth attention, an RMSNorm of the part with weight: normed = bf16(part * rsqrt(G[S0][S0] / H + eps)
// * w), layer 0).
#include "ink_dmix_gaudi2.h"

void main(tensor partials, tensor blocks, tensor part, tensor w, tensor normed, float scale, float eps, int plain)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int S0 = get_dim_size(blocks, 2);
    const int H = get_dim_size(blocks, 0);
    const int Q = get_dim_size(w, 1);
    const int S = S0 + 1;
    const int NT = get_dim_size(partials, 1);
    const float inv_h = ink_srecip((float)H);
    for (int r = start[1]; r < end[1]; ++r) {
        dmix_coef_t cf;
        if (plain) {
            float64 tot = 0.0f;
            for (int i = 0; i < NT; ++i) {
                int5 pc = {0, i, r, 0, 0};
                tot = tot + v_f32_ld_tnsr_b(pc, partials);
            }
            cf.rm[0] = positive_rsqrt(dmix_bcast_01(tot, 6 * S0) * inv_h + eps);
        } else {
            cf = dmix_coefs_compact(partials, S, Q, scale, eps, inv_h, r);
        }
        for (int tile = start[0]; tile < end[0]; ++tile) {
            const int c = tile * 128;
            dmix_states_t xs;
#define x xs.v
            for (int s = 0; s < S0; ++s) {
                int5 bc = {c, r, s, 0, 0};
                x[s] = dmix_bf16_tile(blocks, bc);
            }
#ifdef F32P
            x[S0] = dmix_f32_tile_r(part, c, r);
#else
            int5 pc = {c, r, 0, 0, 0};
            x[S0] = dmix_bf16_tile(part, pc);
#endif
            for (int q = 0; q < Q; ++q) {
                float64_pair_t acc;
                if (plain) {
                    acc = x[S0];
                } else {
                    acc.v1 = 0; acc.v2 = 0;
                    for (int s = 0; s < S; ++s) {
                        acc.v1 = v_f32_mac_b(x[s].v1, cf.c[q][s], acc.v1);
                        acc.v2 = v_f32_mac_b(x[s].v2, cf.c[q][s], acc.v2);
                    }
                }
                int5 wc = {c, q, 0, 0, 0};
                const float64_pair_t wv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
                float64_pair_t nv;
                nv.v1 = (acc.v1 * cf.rm[q]) * wv.v1;
                nv.v2 = (acc.v2 * cf.rm[q]) * wv.v2;
                int5 oc = {c, r, q, 0, 0};
                v_bf16_st_tnsr(oc, normed, v_convert_f32_to_bf16_all_b(nv));
            }
#undef x
        }
    }
}
