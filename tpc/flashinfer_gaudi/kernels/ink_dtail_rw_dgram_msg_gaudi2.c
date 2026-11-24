// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode fused end-of-layer (ink_dtail_rw) + next-app message depth-attention phase 1 (ink_dgram_msg).
// For all non-loop-wrap msg transitions (h_out == next_apart; next_prev [H, T, 2] holds contiguous [prev1, prev2]).
// Tensors (dim 0 fastest, 14 tensors):
//   0: ys bf16 [H, J, T], 1: rw f32 [J, T], 2: st bf16 [H, 3], 3: taps bf16 [H, 4],
//   4: res f32 [H, T], 5: st_out bf16 [H, 3] (or [H, 3 + T]), 6: h_out bf16 [H, T], 7: h_out2 bf16 [H, T],
//   8: next_blocks bf16 [H, T, next_s0], 9: next_q f32 [H, next_Qa], 10: next_p1 f32 [64, NT, T],
//   11: next_prev bf16 [H, T, 2] (slice 0 = prev1, slice 1 = prev2), 12: next_mq f32 [H, 4], 13: next_p3 f32 [64, NT, T].
// Index space [H / 128] (split_dim = 0, tpcs = 24 -> NT = 18 signals). Params: cscale.
#include "ink_dmix_gaudi2.h"

void main(tensor ys, tensor rw, tensor st, tensor taps, tensor res, tensor st_out, tensor h_out, tensor h_out2,
          tensor next_blocks, tensor next_q, tensor next_p1,
          tensor next_prev, tensor next_mq, tensor next_p3, float cscale)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int J = get_dim_size(ys, 1);
    const int T = get_dim_size(ys, 2);
    const int S0 = get_dim_size(next_blocks, 2);
    const int Q = get_dim_size(next_q, 1);

    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        dmix_conv_t cv = dmix_conv_init(st, taps, st_out, c);
        for (int r = 0; r < T; ++r) {
            float64_pair_t sm;
            sm.v1 = 0.0f; sm.v2 = 0.0f;
            if (J == 8) {
                int5 r0 = {0, r, 0, 0, 0}, r1 = {1, r, 0, 0, 0}, r2 = {2, r, 0, 0, 0}, r3 = {3, r, 0, 0, 0};
                int5 r4 = {4, r, 0, 0, 0}, r5 = {5, r, 0, 0, 0}, r6 = {6, r, 0, 0, 0}, r7 = {7, r, 0, 0, 0};
                const float w0 = s_f32_ld_g((__global__ float*)gen_addr(r0, rw));
                const float w1 = s_f32_ld_g((__global__ float*)gen_addr(r1, rw));
                const float w2 = s_f32_ld_g((__global__ float*)gen_addr(r2, rw));
                const float w3 = s_f32_ld_g((__global__ float*)gen_addr(r3, rw));
                const float w4 = s_f32_ld_g((__global__ float*)gen_addr(r4, rw));
                const float w5 = s_f32_ld_g((__global__ float*)gen_addr(r5, rw));
                const float w6 = s_f32_ld_g((__global__ float*)gen_addr(r6, rw));
                const float w7 = s_f32_ld_g((__global__ float*)gen_addr(r7, rw));
                int5 jc = {c, 0, r, 0, 0};
                float64_pair_t a;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w0, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w0, sm.v2); jc[1] = 1;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w1, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w1, sm.v2); jc[1] = 2;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w2, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w2, sm.v2); jc[1] = 3;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w3, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w3, sm.v2); jc[1] = 4;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w4, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w4, sm.v2); jc[1] = 5;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w5, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w5, sm.v2); jc[1] = 6;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w6, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w6, sm.v2); jc[1] = 7;
                a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys)); sm.v1 = v_f32_mac_b(a.v1, w7, sm.v1); sm.v2 = v_f32_mac_b(a.v2, w7, sm.v2);
            } else {
                for (int j = 0; j < J; ++j) {
                    int5 rc = {j, r, 0, 0, 0};
                    const float wj = s_f32_ld_g((__global__ float*)gen_addr(rc, rw));
                    int5 jc = {c, j, r, 0, 0};
                    const float64_pair_t a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(jc, ys));
                    sm.v1 = v_f32_mac_b(a.v1, wj, sm.v1);
                    sm.v2 = v_f32_mac_b(a.v2, wj, sm.v2);
                }
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

            dmix_states_t xs;
            for (int s = 0; s < S0; ++s) {
                int5 bc = {c, r, s, 0, 0};
                xs.v[s] = dmix_bf16_tile(next_blocks, bc);
            }
            xs.v[S0] = v_convert_bf16_to_f32_all_b(ob);
            dmix_gram_store(xs, S0 + 1, next_q, Q, c, next_p1, tile, r);

            dmix_states_t ms;
            int5 pc = {c, r, 0, 0, 0};
            ms.v[0] = dmix_bf16_tile(next_prev, pc);
            pc[2] = 1;
            ms.v[1] = dmix_bf16_tile(next_prev, pc);
            dmix_gram_store(ms, 2, next_mq, 4, c, next_p3, tile, r);
        }
        dmix_conv_finish(cv, st_out, c);
    }
}
