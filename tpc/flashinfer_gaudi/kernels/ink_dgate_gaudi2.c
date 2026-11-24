// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode attention output gate (T = 1, Direct engine), one head per index point:
//   out[h * D + d] = bf16((pv[h, d] / sum[h]) * sigmoid(gate[h * D + d]))       (LUT-free sigmoid)
// pv is the unnormalised P . V (BF16), gate the raw gate slice of the q-side projection row.
// Tensors (dim 0 fastest): pv bf16 [128, G, KV], sums f32 [NH], row bf16 [N] (gate at off_gate), out bf16 [128 * NH].
// FAR  (full layers): + pvf bf16 [128, G, KV], the far keys' P . V, added to pv in F32.
// FAR3 (full layers, split-K TPC PV): + pvf f32 [128, G, KV, NS] (split F32 lane order [even 64 | odd 64]), summed across NS and added to pv in F32.
// Index space [NH]. Params: off_gate, G.
#include "ink_math_gaudi2.h"

void main(tensor pv,
#if defined(FAR) || defined(FAR3)
          tensor pvf,
#endif
          tensor sums, tensor row, tensor out, int off_gate, int G)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int h = start[0]; h < end[0]; ++h) {
        const int kv = h / G, g = h - (h / G) * G;
        int5 pc = {0, g, kv, 0, 0};
#ifdef F32_IN
        float64_pair_t a;
        a.v1 = v_f32_ld_tnsr_b(pc, pv);
        pc[0] = 64;
        a.v2 = v_f32_ld_tnsr_b(pc, pv);
#elif defined(FAR)
        float64_pair_t a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pv));
        const float64_pair_t af = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pvf));
        a.v1 = a.v1 + af.v1;
        a.v2 = a.v2 + af.v2;
#elif defined(FAR3)
        float64_pair_t a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pv));
        const int NS = get_dim_size(pvf, 3);
        float64 f1, f2;
        if (NS == 3) {
            int5 fc0 = {0, g, kv, 0, 0};
            int5 fc1 = {0, g, kv, 1, 0};
            int5 fc2 = {0, g, kv, 2, 0};
            const float64 f0e = v_f32_ld_tnsr_b(fc0, pvf); fc0[0] = 64;
            const float64 f0o = v_f32_ld_tnsr_b(fc0, pvf);
            const float64 f1e = v_f32_ld_tnsr_b(fc1, pvf); fc1[0] = 64;
            const float64 f1o = v_f32_ld_tnsr_b(fc1, pvf);
            const float64 f2e = v_f32_ld_tnsr_b(fc2, pvf); fc2[0] = 64;
            const float64 f2o = v_f32_ld_tnsr_b(fc2, pvf);
            f1 = (f0e + f1e) + f2e;
            f2 = (f0o + f1o) + f2o;
        } else {
            int5 fc = {0, g, kv, 0, 0};
            f1 = v_f32_ld_tnsr_b(fc, pvf); fc[0] = 64;
            f2 = v_f32_ld_tnsr_b(fc, pvf);
            for (int sp = 1; sp < NS; ++sp) {
                fc[0] = 0; fc[3] = sp;
                f1 = f1 + v_f32_ld_tnsr_b(fc, pvf);
                fc[0] = 64;
                f2 = f2 + v_f32_ld_tnsr_b(fc, pvf);
            }
        }
        // Round split-K sum to BF16 to match MME d_fpv output rounding bit-for-bit before F32 add.
        float64_pair_t fp;
        fp.v1 = f1;
        fp.v2 = f2;
        fp = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(fp));
        a.v1 = a.v1 + fp.v1;
        a.v2 = a.v2 + fp.v2;
#else
        const float64_pair_t a = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pv));
#endif
        int5 gc = {off_gate + h * 128, 0, 0, 0, 0};
        const float64_pair_t gt = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gc, row));
        int5 sc = {h, 0, 0, 0, 0};
        const float64 inv = ink_recip(v_broadcast_element_f32(v_f32_ld_tnsr_b(sc, sums), 0));
        float64_pair_t o;
        o.v1 = (a.v1 * inv) * ink_sigmoid(gt.v1);
        o.v2 = (a.v2 * inv) * ink_sigmoid(gt.v2);
        int5 oc = {h * 128, 0, 0, 0, 0};
        v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(o));
    }
}
