// SPDX-License-Identifier: Apache-2.0
// Inkling-2 Direct-engine fused token embedding gather + RMSNorm (replaces Synapse embed_recipe for T = 1 and T = 1024):
//   out[t] = bf16((embed[ids[t]] * rsqrt(mean(embed[ids[t]]^2) + eps)) * w)
// Tensors (dim 0 fastest): ids i32 [T], embed bf16 [H, V], w bf16 [H], out bf16 [H, T].
// Index space: [T]. Params: eps. H <= 128 * MAX_T, H % 256 == 0.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"
#define MAX_T 20
__local__ bfloat128 xv[MAX_T];

void main(tensor ids, tensor embed, tensor w, tensor out, float eps)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(embed, 0);
    const int NT = H / 128;
    const float inv_h = ink_srecip((float)H);
    for (int t = start[0]; t < end[0]; ++t) {
        int5 ic = {t, 0, 0, 0, 0};
        const int tok = s_i32_ld_g((__global__ int*)gen_addr(ic, ids));
        float128 s0 = {0}, s1 = {0};
        for (int i = 0; i < NT; i += 2) {
            int5 ec = {i * 128, tok, 0, 0, 0};
            const bfloat128 a = v_bf16_ld_tnsr_b(ec, embed);
            ec[0] = i * 128 + 128;
            const bfloat128 b = v_bf16_ld_tnsr_b(ec, embed);
            xv[i] = a;
            xv[i + 1] = b;
            s0 = v_bf16_mac_acc32_b(a, a, s0, 0);
            s1 = v_bf16_mac_acc32_b(b, b, s1, 0);
        }
        const float64 r = positive_rsqrt(row_sum((s0.v1 + s0.v2) + (s1.v1 + s1.v2)) * inv_h + eps);
        for (int i = 0; i < NT; ++i) {
            int5 wc = {i * 128, 0, 0, 0, 0};
            const float64_pair_t wf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
            float64_pair_t y = v_convert_bf16_to_f32_all_b(xv[i]);
            y.v1 = (y.v1 * r) * wf.v1;
            y.v2 = (y.v2 * r) * wf.v2;
            int5 oc = {i * 128, t, 0, 0, 0};
            v_bf16_st_tnsr(oc, out, v_convert_f32_to_bf16_all_b(y));
        }
    }
}
