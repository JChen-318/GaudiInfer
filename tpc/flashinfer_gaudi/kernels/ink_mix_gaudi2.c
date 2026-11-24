// SPDX-License-Identifier: Apache-2.0
// Inkling-2 Block Attention Residual ("depth attention") + RMSNorm, one row per index point:
//   states s = 0..S-1: blocks[0..S0) (BF16) then the partial state (BF16, or F32 in the interleaved order, F32P)
//   rms_s = rsqrt(mean(x_s^2) + 1e-6); logit[q, s] = (x_s . query_q) * rms_s * scale; p = softmax_s(logit)
//   mix[q] = sum_s (p[q, s] * rms_s) * x_s          (the mix of the normalized states, F32)
//   normed[q] = bf16((mix[q] * rsqrt(mean(mix[q]^2) + eps)) * w[q])
// F32 tensors (query, mix, F32P partial) use the interleaved order of the BF16 -> FP32 conversion: every 128 values
// as [even 64 | odd 64] (the graph converts with one transpose).
// Tensors (dim 0 fastest): blocks bf16 [H, T, S0], part bf16 [H, T] (F32P: f32), query f32 [H, Q], w bf16 [H, Q],
// normed bf16 [H, Q, T], mix f32 [H, Q, T]. Index space [24] (rows split evenly). Params: scale (1/sqrt(H)), eps (norm), flags.
#define MAXQ 4
#define MAXS 6

static inline float64 exp_vec(float64 x) {
    const float64 xc = v_f32_max_b(x, -87.0f);
    const float64 t = xc * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.0013333558f;
    p = v_f32_mac_b(p, f, 0.0096181291f);
    p = v_f32_mac_b(p, f, 0.0555041087f);
    p = v_f32_mac_b(p, f, 0.2402265070f);
    p = v_f32_mac_b(p, f, 0.6931471806f);
    p = v_f32_mac_b(p, f, 1.0f);
    const int64 bits = (n + 127) << 23;
    return p * (*((float64*)&bits));
}

#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

static inline float64_pair_t load_state(tensor blocks, tensor part, int s, int S0, int col, int t) {
    float64_pair_t x;
    if (s < S0) {
        int5 c = {col, t, s, 0, 0};
        x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, blocks));
    } else {
#ifdef F32P
        int5 c = {col, t, 0, 0, 0};
        x.v1 = v_f32_ld_tnsr_b(c, part);
        c[0] = col + 64;
        x.v2 = v_f32_ld_tnsr_b(c, part);
#else
        int5 c = {col, t, 0, 0, 0};
        x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, part));
#endif
    }
    return x;
}

// Queries in local memory; every state row is read once for the logits and once per output pass. Params flags:
// bit 0 the mix output, bit 1 the normed output (an output not asked for is not written).
#define MAXT 18
#define INK_MIX_P 24
__local__ float64 qv[MAXQ * MAXT * 2];  // [q][tile][half]
__local__ float64 cfv[MAXQ * MAXS];     // p * rms (lane-uniform)

// rows with compile-time state / query counts (the shapes the model uses)
#define SC 2
#define QC 1
#define ROWFN mix_row_2_1
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 3
#define QC 1
#define ROWFN mix_row_3_1
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 4
#define QC 1
#define ROWFN mix_row_4_1
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 5
#define QC 1
#define ROWFN mix_row_5_1
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 2
#define QC 3
#define ROWFN mix_row_2_3
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 3
#define QC 3
#define ROWFN mix_row_3_3
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 4
#define QC 3
#define ROWFN mix_row_4_3
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 5
#define QC 3
#define ROWFN mix_row_5_3
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN
#define SC 2
#define QC 4
#define ROWFN mix_row_2_4
#include "ink_mix_row_gaudi2.h"
#undef SC
#undef QC
#undef ROWFN


void main(tensor blocks, tensor part, tensor query, tensor w, tensor normed, tensor mix, float scale, float eps, int flags)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int H = get_dim_size(query, 0);
    const int Q = get_dim_size(query, 1);
    const int S0 = get_dim_size(blocks, 2);
    const int S = S0 + 1;
    const int tiles = H / 128;
    const float inv_h = ink_srecip((float)H);
    for (int q = 0; q < Q; ++q)
        for (int i = 0; i < tiles; ++i) {
            int5 qc = {i * 128, q, 0, 0, 0};
            qv[(q * MAXT + i) * 2] = v_f32_ld_tnsr_b(qc, query);
            qc[0] = i * 128 + 64;
            qv[(q * MAXT + i) * 2 + 1] = v_f32_ld_tnsr_b(qc, query);
        }
    // index space [INK_MIX_P] (one point per core): point p takes the rows [T p / P, T (p + 1) / P)
    const int T = get_dim_size(blocks, 1);
    const int t0 = (T * start[0]) / INK_MIX_P, t1 = (T * end[0]) / INK_MIX_P;
    for (int t = t0; t < t1; ++t) {
#define DISPATCH(SCV, QCV) if (S == SCV && Q == QCV) { mix_row_##SCV##_##QCV(blocks, part, w, normed, mix, t, tiles, inv_h, scale, eps, flags); continue; }
        DISPATCH(2, 1) DISPATCH(3, 1) DISPATCH(4, 1) DISPATCH(5, 1) DISPATCH(2, 3) DISPATCH(3, 3) DISPATCH(4, 3) DISPATCH(5, 3) DISPATCH(2, 4)
        // (the graph only builds the shapes above)
    }
}
