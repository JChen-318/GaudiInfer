// SPDX-License-Identifier: Apache-2.0
// Inkling-2 batched decode end-of-layer (Bp rows, Direct engine): route-weighted slot sum + per-slot MLP short conv
// + residual add. Dummy rows (bmeta[3, r] == 0) never overwrite conv_out.
// Tensors (dim 0 fastest): ys bf16 [H, J, Bp], rw f32 [J, Bp], conv_in bf16 [H, 3, 2*S], taps bf16 [H, 4],
// res f32 [H, Bp], conv_out bf16 [H, 3, 2*S], h_out bf16 [H, Bp], h_out2 bf16 [H, Bp], bmeta i32 [8, Bp].
// Index space: [NT * Bp]. Params: cscale.
#include "ink_dmix_gaudi2.h"

void main(tensor ys, tensor rw, tensor conv_in, tensor taps, tensor res, tensor conv_out,
          tensor h_out, tensor h_out2, tensor bmeta, float cscale)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(ys, 0);
    const int J = get_dim_size(ys, 1);
    const int NT = H / 128;

    for (int idx = start[0]; idx < end[0]; ++idx) {
        const int r = idx / NT;
        const int tile = idx - r * NT;
        const int c = tile * 128;
        int5 mc = {3, r, 0, 0, 0};
        if (s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta)) == 0) continue;
        mc[0] = 0; const int slot = s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta));
        mc[0] = 1; const int side = s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta));

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
        const bfloat128 xv = v_convert_f32_to_bf16_all_b(sm);

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
        const float64_pair_t s0 = v_convert_bf16_to_f32_all_b(b0);
        const float64_pair_t s1 = v_convert_bf16_to_f32_all_b(b1);
        const float64_pair_t s2 = v_convert_bf16_to_f32_all_b(b2);
        const float64_pair_t s3 = v_convert_bf16_to_f32_all_b(xv);
        float64_pair_t mo;
        mo.v1 = (((s0.v1 * w0.v1 + s1.v1 * w1.v1) + s2.v1 * w2.v1) + s3.v1 * w3.v1 + s3.v1) * cscale;
        mo.v2 = (((s0.v2 * w0.v2 + s1.v2 * w1.v2) + s2.v2 * w2.v2) + s3.v2 * w3.v2 + s3.v2) * cscale;
        const float64_pair_t mv = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(mo));

        const int ss_out = slot * 2 + (1 - side);
        int5 so = {c, 0, ss_out, 0, 0};
        v_bf16_st_tnsr(so, conv_out, b1);
        so[1] = 1; v_bf16_st_tnsr(so, conv_out, b2);
        so[1] = 2; v_bf16_st_tnsr(so, conv_out, xv);

        const float64_pair_t rv = dmix_f32_tile_r(res, c, r);
        float64_pair_t o;
        o.v1 = rv.v1 + mv.v1;
        o.v2 = rv.v2 + mv.v2;
        int5 oc = {c, r, 0, 0, 0};
        const bfloat128 ob = v_convert_f32_to_bf16_all_b(o);
        v_bf16_st_tnsr(oc, h_out, ob);
        v_bf16_st_tnsr(oc, h_out2, ob);
    }
}
