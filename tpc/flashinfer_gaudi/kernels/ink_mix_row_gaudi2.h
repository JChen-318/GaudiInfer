// SPDX-License-Identifier: Apache-2.0
// One row of ink_mix with compile-time state and query counts (SC, QC): the loops over the states and queries
// unroll (no runtime branches in the tile loops). Included by ink_mix_gaudi2.c once per (SC, QC) as ROWFN.
// The VPU MAC has a latency of four cycles: every sum is split over independent accumulators (the v1 / v2 halves,
// even / odd states) so that the dependency chains overlap.
#define UN ((SC) <= 2 ? 3 : (SC) <= 3 ? 2 : 1)  // tiles per group (18 % UN == 0)

// acc = sum_s x[s] * c[q][s] over the states (two partial chains per half)
#define MIXSUM(XU, Q_, ACC) do { \
        float64 a1_ = XU[0].v1 * cfv[(Q_) * MAXS], a2_ = XU[0].v2 * cfv[(Q_) * MAXS]; \
        float64 b1_ = 0, b2_ = 0; \
        if (SC > 1) { b1_ = XU[1].v1 * cfv[(Q_) * MAXS + 1]; b2_ = XU[1].v2 * cfv[(Q_) * MAXS + 1]; } \
        for (int s_ = 2; s_ < SC; s_ += 2) { \
            a1_ = v_f32_mac_b(XU[s_].v1, cfv[(Q_) * MAXS + s_], a1_); \
            a2_ = v_f32_mac_b(XU[s_].v2, cfv[(Q_) * MAXS + s_], a2_); \
            if (s_ + 1 < SC) { \
                b1_ = v_f32_mac_b(XU[s_ + 1].v1, cfv[(Q_) * MAXS + s_ + 1], b1_); \
                b2_ = v_f32_mac_b(XU[s_ + 1].v2, cfv[(Q_) * MAXS + s_ + 1], b2_); \
            } \
        } \
        ACC.v1 = a1_ + b1_; \
        ACC.v2 = a2_ + b2_; \
    } while (0)

static inline void ROWFN(tensor blocks, tensor part, tensor w, tensor normed, tensor mix, int t, int tiles, float inv_h,
                         float scale, float eps, int flags)
{
    const int S0 = SC - 1;
    float64 rms[SC];
    // logits: per state its rms and the QC dots (queries from local memory); three tiles per group
    for (int s = 0; s < SC; ++s) {
        float64 sq1 = 0, sq2 = 0;
        float64 d1[QC], d2[QC];
        for (int q = 0; q < QC; ++q) { d1[q] = 0; d2[q] = 0; }
        for (int i = 0; i < tiles; i += 3) {
            const float64_pair_t xa = load_state(blocks, part, s, S0, i * 128, t);
            const float64_pair_t xb = load_state(blocks, part, s, S0, i * 128 + 128, t);
            const float64_pair_t xc = load_state(blocks, part, s, S0, i * 128 + 256, t);
            sq1 = v_f32_mac_b(xa.v1, xa.v1, sq1);
            sq2 = v_f32_mac_b(xa.v2, xa.v2, sq2);
            for (int q = 0; q < QC; ++q) {
                const int qb = (q * MAXT + i) * 2;
                d1[q] = v_f32_mac_b(xa.v1, qv[qb], d1[q]);
                d2[q] = v_f32_mac_b(xa.v2, qv[qb + 1], d2[q]);
            }
            sq1 = v_f32_mac_b(xb.v1, xb.v1, sq1);
            sq2 = v_f32_mac_b(xb.v2, xb.v2, sq2);
            for (int q = 0; q < QC; ++q) {
                const int qb = (q * MAXT + i) * 2;
                d1[q] = v_f32_mac_b(xb.v1, qv[qb + 2], d1[q]);
                d2[q] = v_f32_mac_b(xb.v2, qv[qb + 3], d2[q]);
            }
            sq1 = v_f32_mac_b(xc.v1, xc.v1, sq1);
            sq2 = v_f32_mac_b(xc.v2, xc.v2, sq2);
            for (int q = 0; q < QC; ++q) {
                const int qb = (q * MAXT + i) * 2;
                d1[q] = v_f32_mac_b(xc.v1, qv[qb + 4], d1[q]);
                d2[q] = v_f32_mac_b(xc.v2, qv[qb + 5], d2[q]);
            }
        }
        const float64 r = positive_rsqrt(row_sum(sq1 + sq2) * inv_h + 1.0e-6f);
        rms[s] = r;
        const float64 rs = r * scale;
        for (int q = 0; q < QC; ++q) cfv[q * MAXS + s] = row_sum(d1[q] + d2[q]) * rs;  // the logits first
    }
    // softmax over the states -> coefficients p * rms (local memory: cfv[q * MAXS + s])
    for (int q = 0; q < QC; ++q) {
        float64 m = cfv[q * MAXS];
        for (int s = 1; s < SC; ++s) m = v_f32_max_b(m, cfv[q * MAXS + s]);
        float64 e[SC];
        float64 sum = 0;
        for (int s = 0; s < SC; ++s) {
            e[s] = exp_vec(cfv[q * MAXS + s] - m);
            sum = sum + e[s];
        }
        const float64 inv = ink_recip(sum);
        for (int s = 0; s < SC; ++s) cfv[q * MAXS + s] = e[s] * inv * rms[s];
    }
    if (flags & 1) {  // the mixes (F32)
        for (int i0 = 0; i0 < tiles; i0 += UN) {
            float64_pair_t xu[UN][SC];
            for (int u = 0; u < UN; ++u)
                for (int s = 0; s < SC; ++s) xu[u][s] = load_state(blocks, part, s, S0, (i0 + u) * 128, t);
            for (int u = 0; u < UN; ++u)
                for (int q = 0; q < QC; ++q) {
                    float64_pair_t acc;
                    MIXSUM(xu[u], q, acc);
                    int5 mc = {(i0 + u) * 128, q, t, 0, 0};
                    v_f32_st_tnsr(mc, mix, acc.v1);
                    mc[0] = (i0 + u) * 128 + 64;
                    v_f32_st_tnsr(mc, mix, acc.v2);
                }
        }
    }
    if (flags & 2) {  // the normed mixes (BF16): mean squares, then the scaled values
        float64 m1[QC], m2[QC];
        for (int q = 0; q < QC; ++q) { m1[q] = 0; m2[q] = 0; }
        for (int i0 = 0; i0 < tiles; i0 += UN) {
            float64_pair_t xu[UN][SC];
            for (int u = 0; u < UN; ++u)
                for (int s = 0; s < SC; ++s) xu[u][s] = load_state(blocks, part, s, S0, (i0 + u) * 128, t);
            for (int u = 0; u < UN; ++u)
                for (int q = 0; q < QC; ++q) {
                    float64_pair_t acc;
                    MIXSUM(xu[u], q, acc);
                    m1[q] = v_f32_mac_b(acc.v1, acc.v1, m1[q]);
                    m2[q] = v_f32_mac_b(acc.v2, acc.v2, m2[q]);
                }
        }
        float64 rm[QC];
        for (int q = 0; q < QC; ++q) rm[q] = positive_rsqrt(row_sum(m1[q] + m2[q]) * inv_h + eps);
        for (int i0 = 0; i0 < tiles; i0 += UN) {
            float64_pair_t xu[UN][SC];
            for (int u = 0; u < UN; ++u)
                for (int s = 0; s < SC; ++s) xu[u][s] = load_state(blocks, part, s, S0, (i0 + u) * 128, t);
            for (int u = 0; u < UN; ++u)
                for (int q = 0; q < QC; ++q) {
                    float64_pair_t acc;
                    MIXSUM(xu[u], q, acc);
                    int5 wc = {(i0 + u) * 128, q, 0, 0, 0};
                    const float64_pair_t wv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, w));
                    float64_pair_t nv;
                    nv.v1 = (acc.v1 * rm[q]) * wv.v1;
                    nv.v2 = (acc.v2 * rm[q]) * wv.v2;
                    int5 nc = {(i0 + u) * 128, q, t, 0, 0};
                    v_bf16_st_tnsr(nc, normed, v_convert_f32_to_bf16_all_b(nv));
                }
        }
    }
}
#undef MIXSUM
#undef UN
