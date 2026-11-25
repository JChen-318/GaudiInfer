// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode / verification depth attention, phase 2 for the message applications (ink_dmix_gaudi2.h): instead
// of the normalized mix, the inputs of the message gates (ink_mg CAT fused in) and the F32 mixes for the gate output:
//   mix[q]  = sum_s c_s x_s over [blocks; part] (Q queries), rm[q] its unweighted RMS factor
//   msg[q]  = the same over the states [prev1, prev2] with the 4 message queries (mpartials), rmm[q]
//   cat[g]  = [bf16(mix[gh] * rm[gh]) | bf16(msg[m0 + g] * rmm[m0 + g])], gh = Q == 1 ? 0 : g      (ink_mg CAT)
// per row r. Tensors (dim 0 fastest): partials f32 [64, NT, T], blocks bf16 [H, T, S0], part bf16 [H, T] (F32P: f32
// [H, T], split order), mpartials f32 [64, NT, T], prev1 bf16 [H, T, 1], prev2 bf16 [H, T], cat bf16 [2H, T, G],
// mix f32 [H, T, Q] (split order), msg f32 [H, T, 4] (split order, written when write_msg). Index space [H / 128, T].
// Params: scale (1 / sqrt(H)), eps, m0, write_msg.
#include "ink_dmix_gaudi2.h"

void main(tensor partials, tensor blocks, tensor part, tensor mpartials, tensor prev1, tensor prev2, tensor cat,
          tensor mix, tensor msg, float scale, float eps, int m0, int write_msg)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int S0 = get_dim_size(blocks, 2);
    const int H = get_dim_size(blocks, 0);
    const int Q = get_dim_size(mix, 2);
    const int G = get_dim_size(cat, 2);
    const int S = S0 + 1;
    const int NT = get_dim_size(partials, 1);
    const float inv_h = ink_srecip((float)H);
    for (int r = start[1]; r < end[1]; ++r) {
        float64 ta0 = 0.0f, ta1 = 0.0f, tb0 = 0.0f, tb1 = 0.0f;
        int i = 0;
        for (; i + 1 < NT; i += 2) {
            int5 pc = {0, i, r, 0, 0};
            ta0 = ta0 + v_f32_ld_tnsr_b(pc, partials);
            tb0 = tb0 + v_f32_ld_tnsr_b(pc, mpartials);
            pc[1] = i + 1;
            ta1 = ta1 + v_f32_ld_tnsr_b(pc, partials);
            tb1 = tb1 + v_f32_ld_tnsr_b(pc, mpartials);
        }
        if (i < NT) {
            int5 pc = {0, i, r, 0, 0};
            ta0 = ta0 + v_f32_ld_tnsr_b(pc, partials);
            tb0 = tb0 + v_f32_ld_tnsr_b(pc, mpartials);
        }
        const float64 tot_a = ta0 + ta1;
        const float64 tot_b = tb0 + tb1;
        dmix_coef_t cfs[2];
        const int mq0 = write_msg ? 0 : m0;
        const int mq1 = write_msg ? 4 : (m0 + G);
#pragma loop_unroll(1)
        for (int pass = 0; pass < 2; ++pass) {
            cfs[pass] = dmix_coefs_tot_range(pass == 0 ? tot_a : tot_b,
                                             pass == 0 ? S : 2,
                                             pass == 0 ? 0 : mq0,
                                             pass == 0 ? Q : mq1,
                                             scale, eps, inv_h);
        }
#define cf cfs[0]
#define cm cfs[1]
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
            int5 bc0 = {c, r, 0, 0, 0};
            const float64_pair_t p1 = dmix_bf16_tile(prev1, bc0);
            const float64_pair_t p2 = dmix_bf16_tile(prev2, bc0);
            float64_pair_t mx[DMIX_MAXQ];
            for (int q = 0; q < Q; ++q) {
                float64_pair_t acc;
                acc.v1 = 0; acc.v2 = 0;
                for (int s = 0; s < S; ++s) {
                    acc.v1 = v_f32_mac_b(x[s].v1, cf.c[q][s], acc.v1);
                    acc.v2 = v_f32_mac_b(x[s].v2, cf.c[q][s], acc.v2);
                }
                mx[q] = acc;
                int5 mc = {c, r, q, 0, 0};
                v_f32_st_tnsr(mc, mix, acc.v1);
                mc[0] = c + 64;
                v_f32_st_tnsr(mc, mix, acc.v2);
            }
#undef x
            float64_pair_t mm[4];
            if (write_msg) {
                for (int q = 0; q < 4; ++q) {
                    mm[q].v1 = p1.v1 * cm.c[q][0] + p2.v1 * cm.c[q][1];
                    mm[q].v2 = p1.v2 * cm.c[q][0] + p2.v2 * cm.c[q][1];
                    int5 mc = {c, r, q, 0, 0};
                    v_f32_st_tnsr(mc, msg, mm[q].v1);
                    mc[0] = c + 64;
                    v_f32_st_tnsr(mc, msg, mm[q].v2);
                }
            } else {
                for (int g = 0; g < G; ++g) {
                    const int q = m0 + g;
                    mm[q].v1 = p1.v1 * cm.c[q][0] + p2.v1 * cm.c[q][1];
                    mm[q].v2 = p1.v2 * cm.c[q][0] + p2.v2 * cm.c[q][1];
                }
            }
            for (int g = 0; g < G; ++g) {
                const int gh = Q == 1 ? 0 : g;
                float64_pair_t a, b;
                a.v1 = mx[gh].v1 * cf.rm[gh];
                a.v2 = mx[gh].v2 * cf.rm[gh];
                b.v1 = mm[m0 + g].v1 * cm.rm[m0 + g];
                b.v2 = mm[m0 + g].v2 * cm.rm[m0 + g];
                int5 oc = {c, r, g, 0, 0};
                v_bf16_st_tnsr(oc, cat, v_convert_f32_to_bf16_all_b(a));
                oc[0] = H + c;
                v_bf16_st_tnsr(oc, cat, v_convert_f32_to_bf16_all_b(b));
            }
        }
#undef cf
#undef cm
    }
}
