// SPDX-License-Identifier: Apache-2.0
// Inkling-2 speculative verification attention output gate (T rows, Direct engine), one (head, row) per index point:
//   out[t, h * D + d] = bf16(((pv + pvn (+ pvf))[h, t, d] / sum[h, t]) * sigmoid(gate[t, h * D + d]))
// pv: the ring part of P . V, pvn: the block's rows, pvf (FAR): the far keys (BF16 each, added in F32); the gate is
// the q-side projection row t (row t * nrow + off_gate).
// Tensors (dim 0 fastest): pv bf16 [128, T, G, KV], pvn bf16 [128, T, G, KV], (FAR: pvf bf16 [128, T, G, KV]),
// sums f32 [NH, T], row bf16 [M nrow], out bf16 [128 NH, T]. Index space [NH T]. Params: off_gate, G, nrow.
#include "ink_math_gaudi2.h"

void main(tensor pv, tensor pvn,
#ifdef FAR
          tensor pvf,
#endif
          tensor sums, tensor row, tensor out, int off_gate, int G, int nrow)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NH = get_dim_size(sums, 0);
    for (int ip = start[0]; ip < end[0]; ++ip) {
        const int h = ip % NH, t = ip / NH;
        const int kv = h / G, g = h - (h / G) * G;
        int5 pc = {0, t, g, kv, 0};
        float64_pair_t a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pv));
        const float64_pair_t an = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pvn));
        a.v1 = a.v1 + an.v1;
        a.v2 = a.v2 + an.v2;
#ifdef FAR
        const float64_pair_t af = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pvf));
        a.v1 = a.v1 + af.v1;
        a.v2 = a.v2 + af.v2;
#endif
        int5 gc = {t * nrow + off_gate + h * 128, 0, 0, 0, 0};
        const float64_pair_t gt = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, row));
        int5 sc = {h, t, 0, 0, 0};
        const float64 inv = ink_recip(v_broadcast_element_f32(v_f32_ld_tnsr_b(sc, sums), 0));
        float64_pair_t o;
        o.v1 = (a.v1 * inv) * ink_sigmoid(gt.v1);
        o.v2 = (a.v2 * inv) * ink_sigmoid(gt.v2);
        int5 oc = {h * 128, t, 0, 0, 0};
        v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(o));
    }
}
