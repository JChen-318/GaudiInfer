// SPDX-License-Identifier: Apache-2.0
// Inkling-2 speculative verification (T rows at positions pos .. pos + T - 1, Direct engine): everything between the
// projection GEMM and the attention scores, as ink_dpost per row, but nothing is committed: the k / v rows go to the
// K/V scratch (accept() commits the accepted rows) and the convs write their input history.
// Index points [0, NH T): query prep of (head, row); [NH T, 2 NH T): relative-bias query + sink of (head, row);
// [2 NH T, 2 NH T + 2 KV): the k / v convs of one KV head over the T rows.
// The projection rows: q side of row t at t * nrow; k input of row t at (three ? T + t : t) * nrow + off_k, v at
// (three ? 2 T + t : t) * nrow + off_v (three: the message / full applications' stacked [q; k; v] GEMM).
// Tensors (dim 0 fastest): row bf16 [M nrow], qs bf16 [128, NH], q bf16 [128, T, G, KV], rs f32 [16, NH],
// r bf16 [16, NH, T], sink f32 [NH, T], st bf16 [KVD, 3, 2], taps bf16 [KVD, 4, 2], hist bf16 [KVD, 3 + T, 2],
// kvs bf16 [128, T, KV, 2] (FULL: + tau f32 [1]).
// Params: thr_q, thr_r, isc, conv scale, eps, off_rel, off_sink, off_k, off_v, nrow, three.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

void main(tensor row, tensor qs, tensor q, tensor rs, tensor r_out, tensor sink, tensor st, tensor taps, tensor hist,
          tensor kvs_t,
#ifdef FULL
          tensor tau,
#endif
          float thr_q, float thr_r, float isc, float cscale, float eps, int off_rel, int off_sink, int off_k, int off_v,
          int nrow, int three)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NH = get_dim_size(qs, 1);
    const int T = get_dim_size(q, 1);
    const int G = get_dim_size(q, 2);
    const int KV = get_dim_size(q, 3);
    const int KVD = KV * 128;
    (void)KVD;
    const float64 lanes = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), 0);
#ifdef FULL
    int5 z0 = {0, 0, 0, 0, 0};
    const float64 tv = v_broadcast_element_f32(v_f32_ld_tnsr_b(z0, tau), 0);
#endif
    for (int p = start[0]; p < end[0]; ++p) {
        if (p < NH * T) {
            const int h = p % NH, t = p / NH;
            int5 scc = {0, h, 0, 0, 0};
            const float64_pair_t sc = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(scc, qs));
            int5 ic = {t * nrow + h * 128, 0, 0, 0, 0};
            float64_pair_t x = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(ic, row));
            x.v1 = x.v1 * sc.v1;
            x.v2 = x.v2 * sc.v2;
            const float64 ss = row_sum(x.v1 * x.v1 + x.v2 * x.v2);
            const float64 norm = ss * positive_rsqrt(ss);
            const float64 f = v_f32_min_b(thr_q * ink_recip(norm + 1.0e-6f), 1.0f);
            x.v1 = x.v1 * f;
            x.v2 = x.v2 * f;
            bfloat128 qb = v_convert_f32_to_bf16_all_b(x);
#ifdef FULL
            {
                float64_pair_t y = v_convert_bf16_to_f32_all_b(qb);
                y.v1 = y.v1 * tv;
                y.v2 = y.v2 * tv;
                qb = v_convert_f32_to_bf16_all_b(y);
            }
#endif
            int5 oc = {0, t, h % G, h / G, 0};
            v_bf16_st_tnsr(oc, q, qb);
        } else if (p < 2 * NH * T) {
            const int pp = p - NH * T;
            const int h = pp % NH, t = pp / NH;
            int5 rc = {t * nrow + off_rel + h * 16, 0, 0, 0, 0};
            const float64_pair_t r = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(rc, row));
            int5 sc = {0, h, 0, 0, 0};
            const float64 se = v_f32_ld_tnsr_b(sc, rs);
            sc[0] = 8;
            const float64 so = v_f32_ld_tnsr_b(sc, rs);
            float64 xe = r.v1 * se, xo = r.v2 * so;
            const float64 sq = v_f32_sel_less_f32_b(lanes, 8.0f, xe * xe + xo * xo, 0.0f);
            const float64 n2 = row_sum(sq);
            const float64 n = v_f32_sel_grt_f32_b(n2, 0.0f, n2 * positive_rsqrt(n2), 0.0f);
            const float64 f = v_f32_min_b(thr_r * ink_recip(n + 1.0e-6f), 1.0f);
            float64_pair_t pr;
            pr.v1 = xe * f;
            pr.v2 = xo * f;
            float64_pair_t rb = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pr));
#ifdef FULL
            rb.v1 = rb.v1 * tv;
            rb.v2 = rb.v2 * tv;
            rb = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(rb));
#endif
            rb.v1 = rb.v1 * isc;
            rb.v2 = rb.v2 * isc;
            int5 oc = {0, h, t, 0, 0};
            v_bf16_st_tnsr_partial(oc, r_out, v_convert_f32_to_bf16_all_b(rb), 15, 0);
            int5 kc0 = {t * nrow + off_sink + h, 0, 0, 0, 0};
            const float64_pair_t sv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(kc0, row));
            int5 so0 = {h, t, 0, 0, 0};
            v_f32_st_tnsr_partial(so0, sink, sv.v1, 0, 0);
        } else {
            const int pp = p - 2 * NH * T;
            const int i = pp % KV, kvs = pp / KV;
            const int c = i * 128;
            int5 wc = {c, 0, kvs, 0, 0};
            const float64_pair_t w0s = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 1;
            const float64_pair_t w1s = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 2;
            const float64_pair_t w2s = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 3;
            const float64_pair_t w3s = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            int5 sc = {c, 0, kvs, 0, 0};
            bfloat128 b0 = v_bf16_ld_tnsr_b(sc, st);
            sc[1] = 1;
            bfloat128 b1 = v_bf16_ld_tnsr_b(sc, st);
            sc[1] = 2;
            bfloat128 b2 = v_bf16_ld_tnsr_b(sc, st);
            int5 hc = {c, 0, kvs, 0, 0};
            v_bf16_st_tnsr(hc, hist, b0);
            hc[1] = 1;
            v_bf16_st_tnsr(hc, hist, b1);
            hc[1] = 2;
            v_bf16_st_tnsr(hc, hist, b2);
            float64_pair_t s0 = v_convert_bf16_to_f32_all_b(b0);
            float64_pair_t s1 = v_convert_bf16_to_f32_all_b(b1);
            float64_pair_t s2 = v_convert_bf16_to_f32_all_b(b2);
            for (int t = 0; t < T; ++t) {
                const int base = (three ? (kvs == 0 ? T + t : 2 * T + t) : t) * nrow + (kvs == 0 ? off_k : off_v);
                int5 xc = {base + c, 0, 0, 0, 0};
                const bfloat128 xv = v_bf16_ld_tnsr_b(xc, row);
                hc[1] = 3 + t;
                v_bf16_st_tnsr(hc, hist, xv);
                const float64_pair_t s3 = v_convert_bf16_to_f32_all_b(xv);
                float64_pair_t w0 = w0s, w1 = w1s, w2 = w2s, w3 = w3s;
#ifdef FULL
                {   // dynamic taps of this head and row: d_j = row[base + KVD + 4 i + j], lag j: w_(3-j) += d_j
                    int5 dc = {base + KVD + i * 4, 0, 0, 0, 0};
                    const float64_pair_t dv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(dc, row));
                    const float64 d0 = v_broadcast_element_f32(dv.v1, 0), d2 = v_broadcast_element_f32(dv.v1, 1);
                    const float64 d1 = v_broadcast_element_f32(dv.v2, 0), d3 = v_broadcast_element_f32(dv.v2, 1);
                    w3.v1 = w3.v1 + d0; w3.v2 = w3.v2 + d0;
                    w2.v1 = w2.v1 + d1; w2.v2 = w2.v2 + d1;
                    w1.v1 = w1.v1 + d2; w1.v2 = w1.v2 + d2;
                    w0.v1 = w0.v1 + d3; w0.v2 = w0.v2 + d3;
                }
#endif
                float64_pair_t o;
                o.v1 = s0.v1 * w0.v1;
                o.v1 = o.v1 + s1.v1 * w1.v1;
                o.v1 = o.v1 + s2.v1 * w2.v1;
                o.v1 = o.v1 + s3.v1 * w3.v1;
                o.v2 = s0.v2 * w0.v2;
                o.v2 = o.v2 + s1.v2 * w1.v2;
                o.v2 = o.v2 + s2.v2 * w2.v2;
                o.v2 = o.v2 + s3.v2 * w3.v2;
                o.v1 = (o.v1 + s3.v1) * cscale;
                o.v2 = (o.v2 + s3.v2) * cscale;
                float64_pair_t yf = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(o));
                const float64 ss = row_sum(yf.v1 * yf.v1 + yf.v2 * yf.v2);
                float64 f;
                if (kvs == 0) {
                    const float64 norm = ss * positive_rsqrt(ss);
                    f = v_f32_min_b(thr_q * ink_recip(norm + 1.0e-6f), 1.0f);
                } else {
                    f = positive_rsqrt(ss * (1.0f / 128.0f) + eps);
                }
                yf.v1 = yf.v1 * f;
                yf.v2 = yf.v2 * f;
                int5 kz = {0, t, i, kvs, 0};
                v_bf16_st_tnsr(kz, kvs_t, v_convert_f32_to_bf16_all_b(yf));
                s0 = s1; s1 = s2; s2 = s3;
            }
        }
    }
}
