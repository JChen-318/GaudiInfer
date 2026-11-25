// SPDX-License-Identifier: Apache-2.0
// Inkling-2 batched decode post-attention conv + residual + MLP depth-attention phase 1 (Bp rows, Direct engine).
// Reads/writes per-slot conv states via conv_pool [H, 3, 2*max_slots] indexed by bmeta[b] = (slot, side, valid).
// Tensors (dim 0 fastest): a1 bf16 [H, Bp], conv_in bf16 [H, 3, 2*S], taps bf16 [H, 4], h_in bf16 [H, Bp],
// blocks bf16 [H, Bp, S0], query f32 [H, 1], conv_out bf16 [H, 3, 2*S], res f32 [H, Bp], partials f32 [64, NT, Bp],
// bmeta i32 [8, Bp]. Index space: [NT * Bp]. Params: cscale, has_h.
#include "ink_dmix_gaudi2.h"

void main(tensor a1, tensor conv_in, tensor taps, tensor h_in, tensor blocks, tensor query,
          tensor conv_out, tensor res, tensor partials, tensor bmeta, float cscale, int has_h)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int S0 = get_dim_size(blocks, 2);
    const int NT = get_dim_size(partials, 1);

    for (int idx = start[0]; idx < end[0]; ++idx) {
        const int r = idx / NT;
        const int tile = idx - r * NT;
        const int c = tile * 128;
        int5 mc = {3, r, 0, 0, 0};
        if (s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta)) == 0) continue;
        mc[0] = 0; const int slot = s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta));
        mc[0] = 1; const int side = s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta));

        int5 wc = {c, 0, 0, 0, 0};
        const float64_pair_t w0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
        wc[1] = 1; const float64_pair_t w1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
        wc[1] = 2; const float64_pair_t w2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
        wc[1] = 3; const float64_pair_t w3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));

        const int ss_in = slot * 2 + side;
        int5 sc = {c, 0, ss_in, 0, 0};
        const bfloat128 b0 = v_bf16_ld_tnsr_b(sc, conv_in);
        sc[1] = 1; const bfloat128 b1 = v_bf16_ld_tnsr_b(sc, conv_in);
        sc[1] = 2; const bfloat128 b2 = v_bf16_ld_tnsr_b(sc, conv_in);
        int5 hc = {c, r, 0, 0, 0};
        const bfloat128 xv = v_bf16_ld_tnsr_b(hc, a1);

        const float64_pair_t s0 = v_convert_bf16_to_f32_all_b(b0);
        const float64_pair_t s1 = v_convert_bf16_to_f32_all_b(b1);
        const float64_pair_t s2 = v_convert_bf16_to_f32_all_b(b2);
        const float64_pair_t s3 = v_convert_bf16_to_f32_all_b(xv);
        float64_pair_t o;
        o.v1 = (((s0.v1 * w0.v1 + s1.v1 * w1.v1) + s2.v1 * w2.v1) + s3.v1 * w3.v1 + s3.v1) * cscale;
        o.v2 = (((s0.v2 * w0.v2 + s1.v2 * w1.v2) + s2.v2 * w2.v2) + s3.v2 * w3.v2 + s3.v2) * cscale;
        const bfloat128 y = v_convert_f32_to_bf16_all_b(o);

        const int ss_out = slot * 2 + (1 - side);
        int5 so = {c, 0, ss_out, 0, 0};
        v_bf16_st_tnsr(so, conv_out, b1);
        so[1] = 1; v_bf16_st_tnsr(so, conv_out, b2);
        so[1] = 2; v_bf16_st_tnsr(so, conv_out, xv);

        float64_pair_t rv = v_convert_bf16_to_f32_all_b(y);
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
}
