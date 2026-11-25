// SPDX-License-Identifier: Apache-2.0
// Inkling-2 prefill attention output gate (Direct engine), one query head per index point, all T rows:
//   out[t, h * D + d] = bf16((o[kv, g T + t, d] / sum[kv, g T + t]) * sigmoid(gate[t, h * D + d]))    (h = kv G + g)
// o is the F32 P . V of the unnormalised probabilities (FAR: + the far keys' P . V), gate the raw gate columns of
// the q-side projection rows. F32 math in the natural column order (the gate converted linearly), LUT-free sigmoid.
// Tensors (dim 0 fastest): o f32 [128, GT, KV], (FAR: of f32 [128, GT, KV]), sums f32 [GT, KV], row bf16 [N, T]
// (gate at column off_gate), out bf16 [NH * 128, T]. Index space [NH]. Params: off_gate, G.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"
#include "ink_lin_gaudi2.h"

void main(tensor o,
#ifdef FAR
          tensor of,
#endif
          tensor sums, tensor row, tensor out, int off_gate, int G)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int T = get_dim_size(row, 1);
#ifdef FAR
#define OFAR(A, OC) do { \
        const float64 f0_ = v_f32_ld_tnsr_b((OC), of); \
        (OC)[0] = 64; \
        const float64 f1_ = v_f32_ld_tnsr_b((OC), of); \
        (OC)[0] = 0; \
        (A).v1 = (A).v1 + f0_; \
        (A).v2 = (A).v2 + f1_; \
    } while (0)
#else
#define OFAR(A, OC) do { } while (0)
#endif
#define GROW(T_) do { \
        int5 oc_ = {0, gT + (T_), kv, 0, 0}; \
        float64_pair_t a_; \
        a_.v1 = v_f32_ld_tnsr_b(oc_, o); \
        oc_[0] = 64; \
        a_.v2 = v_f32_ld_tnsr_b(oc_, o); \
        oc_[0] = 0; \
        OFAR(a_, oc_); \
        int5 sc_ = {gT + (T_), kv, 0, 0, 0}; \
        const float64 inv_ = ink_recip(v_broadcast_element_f32(v_f32_ld_tnsr_b(sc_, sums), 0)); \
        int5 gc_ = {off_gate + h * 128, (T_), 0, 0, 0}; \
        const float64_pair_t gt_ = bf16_to_f32_linear(v_bf16_ld_tnsr_b(gc_, row)); \
        const float64 y0_ = (a_.v1 * inv_) * ink_sigmoid(gt_.v1); \
        const float64 y1_ = (a_.v2 * inv_) * ink_sigmoid(gt_.v2); \
        int5 wc_ = {h * 128, (T_), 0, 0, 0}; \
        v_bf16_st_tnsr(wc_, out, f32_pair_to_bf16_linear(y0_, y1_)); \
    } while (0)
    for (int h = start[0]; h < end[0]; ++h) {
        const int kv = h / G, g = h - kv * G;
        const int gT = g * T;
        for (int t = 0; t < T; t += 2) {
            GROW(t);
            GROW(t + 1);
        }
    }
#undef GROW
#undef OFAR
}
