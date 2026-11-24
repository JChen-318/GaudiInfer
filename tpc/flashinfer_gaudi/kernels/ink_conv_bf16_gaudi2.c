// SPDX-License-Identifier: Apache-2.0
// Inkling-2 short convolution (depthwise, kernel 4) with residual, scale and an optional per-head post-op, one launch
// per conv (k / v of the attention, post-attention, post-MLP):
//   xin = [state (3 rows); x (T rows)]
//   y[t] = bf16(((w0 * xin[t] + w1 * xin[t+1]) + w2 * xin[t+2]) + w3 * xin[t+3] + x[t]) * scale)     (FP32 math)
//   taps w [C, 4] (time order: w3 multiplies the current token); DYN: w_i += dyn[t, head * 4 + (3 - i)] per
//   128-channel head (the dynamic convs of the full layers: lag 0 = current token)
//   mode 1: L2 clip per 128-channel head of the BF16 y: y = bf16(y * min(thr * (1 / (|y| + 1e-6)), 1))
//   mode 2: unweighted RMSNorm per head of the BF16 y:  y = bf16(y * rsqrt(mean(y^2) + eps))
//   state_out = the last 3 rows of xin; with state_out [C, 3 + T] (speculative verification): every row of xin, so
//   the state after any accepted prefix of n rows is rows n .. n + 2
// Tensors (dim 0 fastest): x bf16 [C, T], state bf16 [C, 3], w bf16 [C, 4], (DYN: dyn f32 [KV * 4, T]),
// y bf16 [C, T], state_out bf16 [C, 3]. Index space [C / 128, min(T, 24)].
// Direct-engine prefill variants: HM: y bf16 [128, C / 128, T] (head-major rows through the descriptor strides); HM2: a
// second output y2 with the same rows (the V rows also behind the ring copy of the P . V product: no copy stage);
// INPLACE (with HM / RESA / RESM; T >= 3): state_out is written by the point that reads the state (after its rows),
// so both may be the same tensor. RESA: + res bf16 [C, T], y f32 [C, T] = f32(bf16 conv) + f32(res) (the attention residual, F32 in the interleaved
// order of ink_mix_f32p: every 128 as [even 64 | odd 64]); RESM: + res f32 [C, T] (interleaved), y bf16 [C, T] =
// bf16(f32(bf16 conv) + res) (the MLP residual: the layer output).
#include "norm_math_gaudi2.h"

static inline float64 recip_pos(float64 value) {
    float64 estimate = as_float64((int64)0x7ef311c3 - as_int64(value));
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    return estimate;
}

// The rows are split into NTC = min(T, 24) chunks (index space [C / 128, NTC]): a chunk starts from the three rows
// before it (state rows for t < 0). The loops are specialized per (mode, hist): no branches per row.
#define INK_CONV_NTC(T) ((T) < 24 ? (T) : 24)

static inline bfloat128 conv_row_in(tensor x, tensor state, int c, int r) {
    int5 xc = {c, r, 0, 0, 0};
    if (r < 0) {
        xc[1] = r + 3;
        return v_bf16_ld_tnsr_b(xc, state);
    }
    return v_bf16_ld_tnsr_b(xc, x);
}

#ifdef DYN
// DYNB (Direct-engine prefill): dyn bf16 [KV * 4, T], the tap columns of the projection rows as they are
#ifdef DYNB
#define DYN_LD(C) ((float)s_bf16_ld_g((__global__ bf16*)gen_addr((C), dyn)))
#else
#define DYN_LD(C) s_f32_ld_g((__global__ float*)gen_addr((C), dyn))
#endif
#define CONV_TAPS(T_) \
    float64 k0a, k0b, k1a, k1b, k2a, k2b, k3a, k3b; \
    { \
        int5 dc = {tile * 4, (T_), 0, 0, 0}; \
        const float d0 = DYN_LD(dc); \
        dc[0] = tile * 4 + 1; \
        const float d1 = DYN_LD(dc); \
        dc[0] = tile * 4 + 2; \
        const float d2 = DYN_LD(dc); \
        dc[0] = tile * 4 + 3; \
        const float d3 = DYN_LD(dc); \
        k3a = w3.v1 + d0; k3b = w3.v2 + d0; \
        k2a = w2.v1 + d1; k2b = w2.v2 + d1; \
        k1a = w1.v1 + d2; k1b = w1.v2 + d2; \
        k0a = w0.v1 + d3; k0b = w0.v2 + d3; \
    }
#else
#define CONV_TAPS(T_) \
    const float64 k0a = w0.v1, k0b = w0.v2, k1a = w1.v1, k1b = w1.v2, k2a = w2.v1, k2b = w2.v2, k3a = w3.v1, k3b = w3.v2;
#endif

#if defined(HM2)
#define CONV_STORE(YB, T_) do { int5 yc_ = {0, c >> 7, (T_), 0, 0}; v_bf16_st_tnsr(yc_, y, (YB)); v_bf16_st_tnsr(yc_, y2, (YB)); } while (0)
#elif defined(HM)
#define CONV_STORE(YB, T_) do { int5 yc_ = {0, c >> 7, (T_), 0, 0}; v_bf16_st_tnsr(yc_, y, (YB)); } while (0)
#elif defined(RESA)
#define CONV_STORE(YB, T_) do { \
        int5 yc_ = {c, (T_), 0, 0, 0}; \
        const float64_pair_t yl_ = v_convert_bf16_to_f32_all_b(YB); \
        const float64_pair_t rl_ = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(yc_, res)); \
        v_f32_st_tnsr(yc_, y, yl_.v1 + rl_.v1); \
        yc_[0] = c + 64; \
        v_f32_st_tnsr(yc_, y, yl_.v2 + rl_.v2); \
    } while (0)
#elif defined(RESM)
#define CONV_STORE(YB, T_) do { \
        int5 yc_ = {c, (T_), 0, 0, 0}; \
        float64_pair_t yl_ = v_convert_bf16_to_f32_all_b(YB); \
        yl_.v1 += v_f32_ld_tnsr_b(yc_, res); \
        yc_[0] = c + 64; \
        yl_.v2 += v_f32_ld_tnsr_b(yc_, res); \
        yc_[0] = c; \
        v_bf16_st_tnsr(yc_, y, v_convert_f32_to_bf16_all_b(yl_)); \
    } while (0)
#else
#define CONV_STORE(YB, T_) do { int5 yc_ = {c, (T_), 0, 0, 0}; v_bf16_st_tnsr(yc_, y, (YB)); } while (0)
#endif

// one output row from the window (s0, s1, s2) and s3 = x[t]; two partial sums per half (MAC latency)
#define CONV_ROW(MODE, HIST, T_, XV, S3) do { \
        CONV_TAPS(T_) \
        float64_pair_t o; \
        o.v1 = v_f32_mac_b(s1.v1, k1a, s0.v1 * k0a) + v_f32_mac_b(S3.v1, k3a, s2.v1 * k2a); \
        o.v2 = v_f32_mac_b(s1.v2, k1b, s0.v2 * k0b) + v_f32_mac_b(S3.v2, k3b, s2.v2 * k2b); \
        o.v1 = (o.v1 + S3.v1) * scale; \
        o.v2 = (o.v2 + S3.v2) * scale; \
        bfloat128 yb = v_convert_f32_to_bf16_all_b(o); \
        if (MODE != 0) { \
            float64_pair_t yf = v_convert_bf16_to_f32_all_b(yb); \
            const float64 ss = row_sum(yf.v1 * yf.v1 + yf.v2 * yf.v2); \
            float64 f; \
            if (MODE == 1) { \
                const float64 norm = ss * positive_rsqrt(ss); \
                f = v_f32_min_b(thr * recip_pos(norm + 1.0e-6f), 1.0f); \
            } else { \
                f = positive_rsqrt(ss * (1.0f / 128.0f) + eps); \
            } \
            yf.v1 = yf.v1 * f; \
            yf.v2 = yf.v2 * f; \
            yb = v_convert_f32_to_bf16_all_b(yf); \
        } \
        CONV_STORE(yb, (T_)); \
        if (HIST) { \
            int5 hc_ = {c, 3 + (T_), 0, 0, 0}; \
            v_bf16_st_tnsr(hc_, state_out, XV); \
        } \
    } while (0)

#define CONV_LOOP(MODE, HIST) do { \
        int t = t0; \
        for (; t + 2 <= t1; t += 2) { \
            int5 xc = {c, t, 0, 0, 0}; \
            const bfloat128 xa = v_bf16_ld_tnsr_b(xc, x); \
            xc[1] = t + 1; \
            const bfloat128 xb = v_bf16_ld_tnsr_b(xc, x); \
            const float64_pair_t sa = v_convert_bf16_to_f32_all_b(xa); \
            const float64_pair_t sb = v_convert_bf16_to_f32_all_b(xb); \
            CONV_ROW(MODE, HIST, t, xa, sa); \
            s0 = s1; s1 = s2; s2 = sa; \
            CONV_ROW(MODE, HIST, t + 1, xb, sb); \
            s0 = s1; s1 = s2; s2 = sb; \
        } \
        for (; t < t1; ++t) { \
            int5 xc = {c, t, 0, 0, 0}; \
            const bfloat128 xa = v_bf16_ld_tnsr_b(xc, x); \
            const float64_pair_t sa = v_convert_bf16_to_f32_all_b(xa); \
            CONV_ROW(MODE, HIST, t, xa, sa); \
            s0 = s1; s1 = s2; s2 = sa; \
        } \
    } while (0)

void main(tensor x, tensor state, tensor w,
#ifdef DYN
          tensor dyn,
#endif
#if defined(RESA) || defined(RESM)
          tensor res,
#endif
          tensor y,
#ifdef HM2
          tensor y2,
#endif
          tensor state_out,
#ifdef INPLACE
          tensor vlen,
#endif
          float scale, float thr, float eps, int mode)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int T = get_dim_size(x, 1);
    const int NTC = INK_CONV_NTC(T);
    const int hist = get_dim_size(state_out, 1) > 3;
    for (int j = start[1]; j < end[1]; ++j)
    for (int tile = start[0]; tile < end[0]; ++tile) {
        const int c = tile * 128;
        const int t0 = (T * j) / NTC, t1 = (T * (j + 1)) / NTC;
        int5 wc = {c, 0, 0, 0, 0};
        const float64_pair_t w0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
        wc[1] = 1;
        const float64_pair_t w1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
        wc[1] = 2;
        const float64_pair_t w2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
        wc[1] = 3;
        const float64_pair_t w3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
        // the window before the chunk
        float64_pair_t s0 = v_convert_bf16_to_f32_all_b(conv_row_in(x, state, c, t0 - 3));
        float64_pair_t s1 = v_convert_bf16_to_f32_all_b(conv_row_in(x, state, c, t0 - 2));
        float64_pair_t s2 = v_convert_bf16_to_f32_all_b(conv_row_in(x, state, c, t0 - 1));
        if (hist && j == 0) {  // state_out rows 0..2: the state
            for (int k = 0; k < 3; ++k) {
                int5 sc = {c, k, 0, 0, 0};
                v_bf16_st_tnsr(sc, state_out, v_bf16_ld_tnsr_b(sc, state));
            }
        }
        if (hist) {
            if (mode == 0) CONV_LOOP(0, 1); else if (mode == 1) CONV_LOOP(1, 1); else CONV_LOOP(2, 1);
        } else {
            if (mode == 0) CONV_LOOP(0, 0); else if (mode == 1) CONV_LOOP(1, 0); else CONV_LOOP(2, 0);
        }
#ifdef INPLACE
        // (Direct-engine prefill) state_out may be the state itself: the point that read the state writes the new one
        // (the three rows ending at TV - 1, where TV = vlen[0] <= T supports masked short/tail chunks), after its rows.
        // Load all three rows before storing so TV < 3 in-place never overwrites a needed state row.
        if (j == 0) {
            int5 vc = {0, 0, 0, 0, 0};
            const int TV = s_i32_ld_g((__global__ int*)gen_addr(vc, vlen));
            const bfloat128 r0 = conv_row_in(x, state, c, TV - 3);
            const bfloat128 r1 = conv_row_in(x, state, c, TV - 2);
            const bfloat128 r2 = conv_row_in(x, state, c, TV - 1);
            int5 sc = {c, 0, 0, 0, 0};
            v_bf16_st_tnsr(sc, state_out, r0);
            sc[1] = 1;
            v_bf16_st_tnsr(sc, state_out, r1);
            sc[1] = 2;
            v_bf16_st_tnsr(sc, state_out, r2);
        }
#else
        if (!hist && j == NTC - 1) {  // state_out: the last three rows of [state; x]
            for (int k = 0; k < 3; ++k) {
                int5 sc = {c, k, 0, 0, 0};
                v_bf16_st_tnsr(sc, state_out, conv_row_in(x, state, c, T - 3 + k));
            }
        }
#endif
    }
}
