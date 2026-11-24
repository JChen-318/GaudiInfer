// SPDX-License-Identifier: Apache-2.0
// Inkling-2 batched decode attention output gate (Bp rows, Direct engine, 100% LUT-free).
// FULL_MIX: per row b, if bmeta[2, b] >= W reads F32 o_acc (merged near+far), else converts BF16 pv directly.
// Tensors (dim 0 fastest): pv bf16 [128, G, KV, Bp], [FULL_MIX: o_acc f32 [128, G, KV, Bp], bmeta i32 [8, Bp]],
// sums f32 [NH, Bp], row bf16 [N_total], out bf16 [128*NH, Bp].
// Index space: [NH * Bp]. Params: off_gate, G, nrow, W.
#include "ink_math_gaudi2.h"

void main(tensor pv,
#ifdef FULL_MIX
          tensor o_acc, tensor bmeta,
#endif
          tensor sums, tensor row, tensor out, int off_gate, int G, int nrow, int W)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NH = get_dim_size(sums, 0);
    (void)W;
    for (int idx = start[0]; idx < end[0]; ++idx) {
        const int b = idx / NH;
        const int h = idx - b * NH;
        const int kv = h / G, g = h - kv * G;
        int5 pc = {0, g, kv, b, 0};
        float64_pair_t a;
#ifdef FULL_MIX
        int5 mc = {3, b, 0, 0, 0};
        if (s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta)) == 0) continue;
        mc[0] = 2;
        const int pos = s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta));
        if (pos >= W) {
            a.v1 = v_f32_ld_tnsr_b(pc, o_acc);
            pc[0] = 64;
            a.v2 = v_f32_ld_tnsr_b(pc, o_acc);
        } else {
            a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pv));
        }
#else
        a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pv));
#endif
        int5 gc = {b * nrow + off_gate + h * 128, 0, 0, 0, 0};
        const float64_pair_t gt = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, row));
        int5 sc = {h, b, 0, 0, 0};
        const float64 inv = ink_recip(v_broadcast_element_f32(v_f32_ld_tnsr_b(sc, sums), 0));
        float64_pair_t o;
        o.v1 = (a.v1 * inv) * ink_sigmoid(gt.v1);
        o.v2 = (a.v2 * inv) * ink_sigmoid(gt.v2);
        int5 oc = {h * 128, b, 0, 0, 0};
        v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(o));
    }
}
