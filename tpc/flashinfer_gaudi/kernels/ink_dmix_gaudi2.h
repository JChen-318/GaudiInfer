// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode / verification (T rows, Direct engine) building blocks, one 128-channel tile per index point:
//  - the short conv of one tile (ink_conv mode 0 arithmetic),
//  - the depth-attention ("mix") split in two parallel phases through the Gram matrix of the states. Phase 1 stores
//    per tile the partial sums G[s][t] = x_s . x_t (s <= t, lane 5 s + t) and D[s][q] = x_s . query_q
//    (lane 32 + 4 s + q) as one 64-lane F32 vector. Phase 2 sums the tiles' vectors, then (redundantly per TPC):
//      rms_s = rsqrt(G[s][s] / H + 1e-6); logit = D[s][q] * rms_s * scale; p = softmax_s; c_s = p_s * rms_s
//      mean(mix^2) = sum_s,t c_s c_t G[s][t] / H;  rm = rsqrt(mean + eps)
//      normed[q] = bf16((sum_s c_s x_s) * rm * w[q])   for its own tile (ink_mix's op order for the output)
// States: blocks[0 .. S0) (BF16) then the partial state. F32 vectors are in the split order of the BF16 -> F32
// conversion ([even 64 | odd 64] per 128), as everywhere in the Inkling kernels.
#pragma once
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

#define DMIX_MAXS 5
#define DMIX_MAXQ 4

typedef struct { float64_pair_t v[DMIX_MAXS]; } dmix_states_t;
typedef struct { float64 c[DMIX_MAXQ][DMIX_MAXS]; float64 rm[DMIX_MAXQ]; } dmix_coef_t;

static inline float64_pair_t dmix_bf16_tile(tensor t, int5 c) { return v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c, t)); }

static inline float64_pair_t dmix_f32_tile_r(tensor t, int col, int row) {
    float64_pair_t x;
    int5 c = {col, row, 0, 0, 0};
    x.v1 = v_f32_ld_tnsr_b(c, t);
    c[0] = col + 64;
    x.v2 = v_f32_ld_tnsr_b(c, t);
    return x;
}
static inline float64_pair_t dmix_f32_tile(tensor t, int col) { return dmix_f32_tile_r(t, col, 0); }

// the short conv over T rows of one tile (ink_conv mode 0 arithmetic), stepped row by row:
//   y_t = bf16(((((s0 w0 + s1 w1) + s2 w2) + x_t w3) + x_t) * scale), state <- (s1, s2, x_t)
// st_out [C, 3]: the final state; st_out [C, 3 + T] (verification): every input row (the state after any accepted
// prefix of n rows is rows n .. n + 2)
typedef struct {
    float64_pair_t w0, w1, w2, w3, s0, s1, s2;
    bfloat128 b0, b1, b2, y;
} dmix_conv_t;
static inline int dmix_hist(tensor st_out) { return get_dim_size(st_out, 1) > 3; }

static inline dmix_conv_t dmix_conv_init(tensor st, tensor taps, tensor st_out, int c) {
    dmix_conv_t v;
    int5 wc = {c, 0, 0, 0, 0};
    v.w0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
    wc[1] = 1;
    v.w1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
    wc[1] = 2;
    v.w2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
    wc[1] = 3;
    v.w3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
    int5 sc = {c, 0, 0, 0, 0};
    v.b0 = v_bf16_ld_tnsr_b(sc, st);
    sc[1] = 1;
    v.b1 = v_bf16_ld_tnsr_b(sc, st);
    sc[1] = 2;
    v.b2 = v_bf16_ld_tnsr_b(sc, st);
    v.s0 = v_convert_bf16_to_f32_all_b(v.b0);
    v.s1 = v_convert_bf16_to_f32_all_b(v.b1);
    v.s2 = v_convert_bf16_to_f32_all_b(v.b2);
    if (dmix_hist(st_out)) {
        sc[1] = 0;
        v_bf16_st_tnsr(sc, st_out, v.b0);
        sc[1] = 1;
        v_bf16_st_tnsr(sc, st_out, v.b1);
        sc[1] = 2;
        v_bf16_st_tnsr(sc, st_out, v.b2);
    }
    v.y = v.b0;
    return v;
}

static inline dmix_conv_t dmix_conv_step(dmix_conv_t v, bfloat128 xv, tensor st_out, int c, int t, float scale) {
    const float64_pair_t s3 = v_convert_bf16_to_f32_all_b(xv);
    float64_pair_t o;
    o.v1 = v.s0.v1 * v.w0.v1;
    o.v1 = o.v1 + v.s1.v1 * v.w1.v1;
    o.v1 = o.v1 + v.s2.v1 * v.w2.v1;
    o.v1 = o.v1 + s3.v1 * v.w3.v1;
    o.v2 = v.s0.v2 * v.w0.v2;
    o.v2 = o.v2 + v.s1.v2 * v.w1.v2;
    o.v2 = o.v2 + v.s2.v2 * v.w2.v2;
    o.v2 = o.v2 + s3.v2 * v.w3.v2;
    o.v1 = (o.v1 + s3.v1) * scale;
    o.v2 = (o.v2 + s3.v2) * scale;
    v.y = v_convert_f32_to_bf16_all_b(o);
    if (dmix_hist(st_out)) {
        int5 hc = {c, 3 + t, 0, 0, 0};
        v_bf16_st_tnsr(hc, st_out, xv);
    }
    v.s0 = v.s1; v.s1 = v.s2; v.s2 = s3;
    v.b0 = v.b1; v.b1 = v.b2; v.b2 = xv;
    return v;
}

static inline void dmix_conv_finish(dmix_conv_t v, tensor st_out, int c) {
    if (dmix_hist(st_out)) return;
    int5 sc = {c, 0, 0, 0, 0};
    v_bf16_st_tnsr(sc, st_out, v.b0);
    sc[1] = 1;
    v_bf16_st_tnsr(sc, st_out, v.b1);
    sc[1] = 2;
    v_bf16_st_tnsr(sc, st_out, v.b2);
}

// Scratch VLM/SLM buffers for branchless 4-way lockstep Gram/Dot reduction (max 15 Gram + 20 Dot = 35 <= 40).
static __local__ float64 dmix_pbuf_[40];
static __local__ int dmix_lbuf_[40];

// Fast 4-way lockstep 64-lane row sum (butterfly XOR matches row_sum's binary tree bit-for-bit with 2.5x fewer shuffles).
#define DMIX_ROW_SUM4(V0, V1, V2, V3, C1, C2, C4) do { \
        (V0) += v_f32_mov_dual_group_all_b((V0), 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0); \
        (V1) += v_f32_mov_dual_group_all_b((V1), 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0); \
        (V2) += v_f32_mov_dual_group_all_b((V2), 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0); \
        (V3) += v_f32_mov_dual_group_all_b((V3), 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0); \
        (V0) += v_f32_mov_dual_group_all_b((V0), 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0); \
        (V1) += v_f32_mov_dual_group_all_b((V1), 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0); \
        (V2) += v_f32_mov_dual_group_all_b((V2), 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0); \
        (V3) += v_f32_mov_dual_group_all_b((V3), 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0); \
        (V0) += v_f32_mov_group_b((V0), 0xffffffff, 63, 0); \
        (V1) += v_f32_mov_group_b((V1), 0xffffffff, 63, 0); \
        (V2) += v_f32_mov_group_b((V2), 0xffffffff, 63, 0); \
        (V3) += v_f32_mov_group_b((V3), 0xffffffff, 63, 0); \
        (V0) += v_f32_shuffle_b((V0), (C1), 0, (V0)); \
        (V1) += v_f32_shuffle_b((V1), (C1), 0, (V1)); \
        (V2) += v_f32_shuffle_b((V2), (C1), 0, (V2)); \
        (V3) += v_f32_shuffle_b((V3), (C1), 0, (V3)); \
        (V0) += v_f32_shuffle_b((V0), (C2), 0, (V0)); \
        (V1) += v_f32_shuffle_b((V1), (C2), 0, (V1)); \
        (V2) += v_f32_shuffle_b((V2), (C2), 0, (V2)); \
        (V3) += v_f32_shuffle_b((V3), (C2), 0, (V3)); \
        (V0) += v_f32_shuffle_b((V0), (C4), 0, (V0)); \
        (V1) += v_f32_shuffle_b((V1), (C4), 0, (V1)); \
        (V2) += v_f32_shuffle_b((V2), (C4), 0, (V2)); \
        (V3) += v_f32_shuffle_b((V3), (C4), 0, (V3)); \
    } while (0)

// phase 1 of one tile: x[0..S) the states' tiles (F32 split order), query f32 [H, Q] (split order) -> partials[.., tile]
// Collects products into VLM then runs ONE branchless 4-way lockstep reduction loop (zero stalls, tiny .text, 0 VRF spills).
static inline void dmix_gram_store(dmix_states_t xs, int S, tensor query, int Q, int col, tensor partials, int tile, int row) {
#define x xs.v
    int n = 0;
#pragma loop_unroll(1)
    for (int s = 0; s < S; ++s) {
#pragma loop_unroll(1)
        for (int t = s; t < S; ++t) {
            dmix_pbuf_[n] = x[s].v1 * x[t].v1 + x[s].v2 * x[t].v2;
            dmix_lbuf_[n] = 5 * s + t;
            ++n;
        }
    }
#pragma loop_unroll(1)
    for (int q = 0; q < Q; ++q) {
        int5 qc = {col, q, 0, 0, 0};
        const float64 qa = v_f32_ld_tnsr_b(qc, query);
        qc[0] = col + 64;
        const float64 qb = v_f32_ld_tnsr_b(qc, query);
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) {
            dmix_pbuf_[n] = x[s].v1 * qa + x[s].v2 * qb;
            dmix_lbuf_[n] = 32 + 4 * s + q;
            ++n;
        }
    }
#undef x
    dmix_pbuf_[n] = 0.0f;     dmix_lbuf_[n] = -1;
    dmix_pbuf_[n + 1] = 0.0f; dmix_lbuf_[n + 1] = -1;
    dmix_pbuf_[n + 2] = 0.0f; dmix_lbuf_[n + 2] = -1;

    const uint64 ulanes = read_lane_id_4b_b();
    const float64 lanes = v_convert_i32_to_f32_b((int64)ulanes, 0);
    const uint64 l = ulanes & 7;
    const uint64 u1 = ((l ^ 1) | 0x80) * 0x01010101;
    const uint64 u2 = ((l ^ 2) | 0x80) * 0x01010101;
    const uint64 u4 = ((l ^ 4) | 0x80) * 0x01010101;
    const uchar256 c1 = *((uchar256*)&u1);
    const uchar256 c2 = *((uchar256*)&u2);
    const uchar256 c4 = *((uchar256*)&u4);
    float64 acc = 0.0f;
#pragma loop_unroll(1)
    for (int i = 0; i < n; i += 4) {
        float64 v0 = dmix_pbuf_[i];
        float64 v1 = dmix_pbuf_[i + 1];
        float64 v2 = dmix_pbuf_[i + 2];
        float64 v3 = dmix_pbuf_[i + 3];
        DMIX_ROW_SUM4(v0, v1, v2, v3, c1, c2, c4);
        acc = v_f32_sel_eq_f32_b(lanes, (float)dmix_lbuf_[i],     v0, acc);
        acc = v_f32_sel_eq_f32_b(lanes, (float)dmix_lbuf_[i + 1], v1, acc);
        acc = v_f32_sel_eq_f32_b(lanes, (float)dmix_lbuf_[i + 2], v2, acc);
        acc = v_f32_sel_eq_f32_b(lanes, (float)dmix_lbuf_[i + 3], v3, acc);
    }
    int5 pc = {0, tile, row, 0, 0};
    v_f32_st_tnsr(pc, partials, acc);
}

static inline float64 dmix_exp(float64 x) { return ink_exp(x); }

// Fast range-specialized lane broadcasts (2..3x fewer instructions & cycles than generic v_broadcast_element_f32):
// - dmix_bcast_03: lane in [0..3] (always DG0 group 0) -> 1 shuffle + 1 mov_dg.all
// - dmix_bcast_01: lane in [0..31] (DG0 or DG1) -> 1 shuffle + 2 mov_dg.all
// - dmix_bcast_23: off in [0..19] for lane (32 + off) in [32..51] (DG2 or DG3) -> 1 shuffle + 2 mov_dg.all
static inline float64 dmix_bcast_03(float64 v, int lane) {
    const unsigned char b = (unsigned char)(0x80 | (lane & 7));
    v = v_f32_shuffle_b(v, (uchar256)b, 0, v);
    return v_f32_mov_dual_group_all_b(v, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), 0);
}
static inline float64 dmix_bcast_01(float64 v, int lane) {
    const unsigned char b = (unsigned char)(0x80 | ((lane & 8) << 2) | (lane & 7));
    v = v_f32_shuffle_b(v, (uchar256)b, 0, v);
    const float64 d0 = v_f32_mov_dual_group_all_b(v, 0xffffffff, 0, 0, 0, 0, MkWrA(3, 3, 3, 3), 0);
    const float64 d1 = v_f32_mov_dual_group_all_b(v, 0xffffffff, 1, 1, 1, 1, MkWrA(3, 3, 3, 3), 0);
    return (lane & 16) ? d1 : d0;
}
static inline float64 dmix_bcast_23(float64 v, int off) {
    const unsigned char b = (unsigned char)(0x80 | ((off & 8) << 2) | (off & 7));
    v = v_f32_shuffle_b(v, (uchar256)b, 0, v);
    const float64 d2 = v_f32_mov_dual_group_all_b(v, 0xffffffff, 2, 2, 2, 2, MkWrA(3, 3, 3, 3), 0);
    const float64 d3 = v_f32_mov_dual_group_all_b(v, 0xffffffff, 3, 3, 3, 3, MkWrA(3, 3, 3, 3), 0);
    return (off & 16) ? d3 : d2;
}

// phase 2 core (64-lane vectorized, bit-exact to scalar formula): computes ALL S rms, ALL Q*S softmax exps,
// ALL [q0, q1) recips, and ALL [q0, q1) output rsqrts using only 4 vector transcendental calls total.
static inline dmix_coef_t dmix_coefs_tot_range(float64 tot, int S, int q0, int q1, float scale, float eps, float inv_h) {
    dmix_coef_t out;
#define coef out.c
#define rm out.rm
    const uint64 ulanes = read_lane_id_4b_b();
    const int64 ilanes = (int64)ulanes;
    const int64 d_s = (ilanes - 32) >> 2;
    const uint64 ux4 = (((ulanes & 7) ^ 4) | 0x80) * 0x01010101;
    const uchar256 c_xor4 = *((uchar256*)&ux4);

    // 1) All S state rsqrts in ONE positive_rsqrt call (G[s][s] sits at lane 6*s in tot, lanes 0..24)
    const float64 rms_all = positive_rsqrt(v_f32_max_b(tot, 0.0f) * inv_h + 1.0e-6f);
    float64 g[DMIX_MAXS][DMIX_MAXS];
    float64 rms[DMIX_MAXS];
    float64 rscale_d = 0.0f;
#pragma loop_unroll(1)
    for (int s = 0; s < S; ++s) {
#pragma loop_unroll(1)
        for (int t = s; t < S; ++t) {
            g[s][t] = dmix_bcast_01(tot, 5 * s + t);
            g[t][s] = g[s][t];
        }
        rms[s] = dmix_bcast_01(rms_all, 6 * s);
        rscale_d = v_f32_sel_eq_i32_b(d_s, s, rms[s] * scale, rscale_d);
    }

    // 2) All Q*S logits (lane 32 + 4*s + q) bit-exact `tot * (rms[s] * scale)` -> 4-op parallel max -> ONE ink_exp!
    const float64 lg_all = tot * rscale_d;
    float64 m = v_f32_sel_geq_i32_b(d_s, S, -3.4e38f, lg_all);
    m = v_f32_max_b(m, v_f32_mov_dual_group_all_b(m, 0xffffffff, 0, 1, 3, 2, MkWrA(3, 3, 3, 3), 0));
    m = v_f32_max_b(m, v_f32_mov_group_b(m, 0xffffffff, 63, 0));
    m = v_f32_max_b(m, v_f32_shuffle_b(m, c_xor4, 0, m));
    const float64 m_all = v_f32_mov_dual_group_all_b(m, 0xffffffff, 2, 2, 2, 2, MkWrA(3, 3, 3, 3), 0);
    const float64 exp_all = dmix_exp(lg_all - m_all);

    // 3) Bit-exact sequential sum per query q in [q0, q1) -> ONE ink_recip call for all active queries!
    float64 ev[DMIX_MAXQ][DMIX_MAXS];
    float64 sum_vec = 1.0f;
#pragma loop_unroll(1)
    for (int q = q0; q < q1; ++q) {
        float64 sq = 0.0f;
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) {
            ev[q][s] = dmix_bcast_23(exp_all, 4 * s + q);
            sq = sq + ev[q][s];
        }
        sum_vec = v_f32_sel_eq_i32_b(ilanes, q, sq, sum_vec);
    }
    const float64 inv_vec = ink_recip(sum_vec);

    // 4) Mix coefficients & mean-square per active query -> ONE positive_rsqrt call!
    float64 msq_vec = 0.0f;
#pragma loop_unroll(1)
    for (int q = q0; q < q1; ++q) {
        const float64 invq = dmix_bcast_03(inv_vec, q);
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) coef[q][s] = (ev[q][s] * invq) * rms[s];
        float64 msq = 0.0f;
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) {
            float64 r = 0.0f;
#pragma loop_unroll(1)
            for (int t = 0; t < S; ++t) r = r + coef[q][t] * g[s][t];
            msq = msq + coef[q][s] * r;
        }
        msq_vec = v_f32_sel_eq_i32_b(ilanes, q, msq, msq_vec);
    }
    const float64 rm_vec = positive_rsqrt(v_f32_max_b(msq_vec, 0.0f) * inv_h + eps);
#pragma loop_unroll(1)
    for (int q = q0; q < q1; ++q) rm[q] = dmix_bcast_03(rm_vec, q);
#undef coef
#undef rm
    return out;
}

static inline dmix_coef_t dmix_coefs_tot(float64 tot, int S, int Q, float scale, float eps, float inv_h) {
    return dmix_coefs_tot_range(tot, S, 0, Q, scale, eps, inv_h);
}

// Compact bit-exact phase 2 for single-mix ink_dnorm (Q=1 dominant): 1 vectorized rms_all + fast dmix_bcast_01/23,
// zero c_xor4 / sum_vec / msq_vec overhead, smaller & faster than original 6805255!
static inline dmix_coef_t dmix_coefs_compact(tensor partials, int S, int Q, float scale, float eps, float inv_h, int row) {
    const int NT = get_dim_size(partials, 1);
    float64 tot = 0.0f, tot2 = 0.0f;
    int i = 0;
    for (; i + 1 < NT; i += 2) {
        int5 pc = {0, i, row, 0, 0};
        tot = tot + v_f32_ld_tnsr_b(pc, partials);
        pc[1] = i + 1;
        tot2 = tot2 + v_f32_ld_tnsr_b(pc, partials);
    }
    if (i < NT) {
        int5 pc = {0, i, row, 0, 0};
        tot = tot + v_f32_ld_tnsr_b(pc, partials);
    }
    tot = tot + tot2;
    dmix_coef_t out;
#define coef out.c
#define rm out.rm
    const float64 rms_all = positive_rsqrt(v_f32_max_b(tot, 0.0f) * inv_h + 1.0e-6f);
    float64 g[DMIX_MAXS][DMIX_MAXS];
    float64 rms[DMIX_MAXS];
#pragma loop_unroll(1)
    for (int s = 0; s < S; ++s) {
#pragma loop_unroll(1)
        for (int t = s; t < S; ++t) {
            g[s][t] = dmix_bcast_01(tot, 5 * s + t);
            g[t][s] = g[s][t];
        }
        rms[s] = dmix_bcast_01(rms_all, 6 * s);
    }
#pragma loop_unroll(1)
    for (int q = 0; q < Q; ++q) {
        float64 lg[DMIX_MAXS];
        float64 m = -1.0e30f;
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) {
            lg[s] = dmix_bcast_23(tot, 4 * s + q) * (rms[s] * scale);
            m = v_f32_max_b(m, lg[s]);
        }
        float64 e[DMIX_MAXS];
        float64 sum = 0.0f;
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) {
            e[s] = dmix_exp(lg[s] - m);
            sum = sum + e[s];
        }
        const float64 inv = ink_recip(sum);
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) coef[q][s] = e[s] * inv * rms[s];
        float64 msq = 0.0f;
#pragma loop_unroll(1)
        for (int s = 0; s < S; ++s) {
            float64 r = 0.0f;
#pragma loop_unroll(1)
            for (int t = 0; t < S; ++t) r = r + coef[q][t] * g[s][t];
            msq = msq + coef[q][s] * r;
        }
        rm[q] = positive_rsqrt(msq * inv_h + eps);
    }
#undef coef
#undef rm
    return out;
}

// phase 2: the coefficients c[q][s] and the norm factors rm[q] from the summed partials (all TPCs alike)
static inline dmix_coef_t dmix_coefs(tensor partials, int S, int Q, float scale, float eps, float inv_h, int row) {
    const int NT = get_dim_size(partials, 1);
    float64 tot = 0.0f, tot2 = 0.0f;
    int i = 0;
    for (; i + 1 < NT; i += 2) {
        int5 pc = {0, i, row, 0, 0};
        tot = tot + v_f32_ld_tnsr_b(pc, partials);
        pc[1] = i + 1;
        tot2 = tot2 + v_f32_ld_tnsr_b(pc, partials);
    }
    if (i < NT) {
        int5 pc = {0, i, row, 0, 0};
        tot = tot + v_f32_ld_tnsr_b(pc, partials);
    }
    return dmix_coefs_tot(tot + tot2, S, Q, scale, eps, inv_h);
}
