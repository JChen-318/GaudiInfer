// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode (T = 1, Direct engine): everything between the projection GEMM and the attention scores in one
// launch. Index points [0, NH): query prep of head p; [NH, 2 NH): relative-bias query + sink of head p - NH;
// [2 NH, 2 NH + KV): k conv of KV head i (L2 clip); [2 NH + KV, 2 NH + 2 KV): v conv (RMS per head). The k / v rows go
// straight into the KV ring at row pos % W (skips unused kvs_t HBM store when dim 3 < 2 in full-Direct mode).
//   q    = bf16(l2clip(q_lin * q_scale, thr_q))                                         (ink_qprep)
//   r    = bf16(bf16(l2clip(rel * r_scale, thr_r)) / softmax_scale), sink = row[off_sink + h]   (ink_dprep)
//   k, v = conv4(state, x) as ink_conv mode 1 / 2, state_out = the last 3 inputs           (ink_conv)
// Tensors (dim 0 fastest): row bf16 [N] (q | gate | rel | sink | k | v), qs bf16 [128, NH], q bf16 [128, NH],
// rs f32 [16, NH] (r_scale, [even 8 | odd 8] per head), r bf16 [16, NH], sink f32 [NH], st bf16 [KVD, 3, 2] (k, v
// conv states), taps bf16 [KVD, 4, 2], st_out bf16 [KVD, 3, 2], pos_t i32 [1], kc bf16 [D, W, KV], vc bf16 [D, W, KV],
// kvs_t bf16 [D, 1, KV, 2] (or [D, 1, KV, 1] to skip kvs store).
// Params: thr_q, thr_r, isc (1 / softmax scale), conv scale, eps, off_rel, off_sink, off_k, off_v (row offsets).
// FULL (full-attention layers): + tau f32 [1], klin / vlin bf16 [D, CAP, KV].
// LUT-free.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

void main(tensor row, tensor qs, tensor q, tensor rs, tensor r_out, tensor sink, tensor st, tensor taps, tensor st_out,
          tensor pos_t, tensor kc, tensor vc, tensor kvs_t,
#ifdef FULL
          tensor tau, tensor klin, tensor vlin,
#endif
          float thr_q, float thr_r, float isc, float cscale, float eps, int off_rel, int off_sink, int off_k, int off_v)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NH = get_dim_size(q, 1);
    const int KV = get_dim_size(kc, 2);
    const int KVD = KV * 128;
    const int write_kvs = get_dim_size(kvs_t, 3) >= 2;
    (void)KVD;
    const float64 lanes = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), 0);
    int5 z0 = {0, 0, 0, 0, 0};
    const int pos = s_i32_ld_g((__global__ int*)gen_addr(z0, pos_t));
    const int W = get_dim_size(kc, 1);
#ifdef FULL
    const float64 tv = v_broadcast_element_f32(v_f32_ld_tnsr_b(z0, tau), 0);
#endif
    for (int p = start[0]; p < end[0]; ++p) {
        if (p < NH) {
            const int h = p;
            int5 scc = {0, h, 0, 0, 0};
            const float64_pair_t sc = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(scc, qs));
            int5 ic = {h * 128, 0, 0, 0, 0};
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
            int5 oc = {0, h, 0, 0, 0};
            v_bf16_st_tnsr(oc, q, qb);
        } else if (p < 2 * NH) {
            const int h = p - NH;
            int5 rc = {off_rel + h * 16, 0, 0, 0, 0};
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
            int5 oc = {0, h, 0, 0, 0};
            v_bf16_st_tnsr_partial(oc, r_out, v_convert_f32_to_bf16_all_b(rb), 15, 0);
            int5 kc0 = {off_sink + h, 0, 0, 0, 0};
            const float64_pair_t sv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(kc0, row));
            int5 so0 = {h, 0, 0, 0, 0};
            v_f32_st_tnsr_partial(so0, sink, sv.v1, 0, 0);
        } else {
            const int i = (p - 2 * NH) % KV;
            const int kvs = (p - 2 * NH) / KV;
            const int c = i * 128;
            int5 wc = {c, 0, kvs, 0, 0};
            float64_pair_t w0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 1;
            float64_pair_t w1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 2;
            float64_pair_t w2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 3;
            float64_pair_t w3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
#ifdef FULL
            {
                int5 dc = {(kvs == 0 ? off_k : off_v) + KVD + i * 4, 0, 0, 0, 0};
                const float64_pair_t dv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(dc, row));
                const float64 d0 = v_broadcast_element_f32(dv.v1, 0), d2 = v_broadcast_element_f32(dv.v1, 1);
                const float64 d1 = v_broadcast_element_f32(dv.v2, 0), d3 = v_broadcast_element_f32(dv.v2, 1);
                w3.v1 = w3.v1 + d0; w3.v2 = w3.v2 + d0;
                w2.v1 = w2.v1 + d1; w2.v2 = w2.v2 + d1;
                w1.v1 = w1.v1 + d2; w1.v2 = w1.v2 + d2;
                w0.v1 = w0.v1 + d3; w0.v2 = w0.v2 + d3;
            }
#endif
            int5 sc = {c, 0, kvs, 0, 0};
            const bfloat128 b0 = v_bf16_ld_tnsr_b(sc, st);
            sc[1] = 1;
            const bfloat128 b1 = v_bf16_ld_tnsr_b(sc, st);
            sc[1] = 2;
            const bfloat128 b2 = v_bf16_ld_tnsr_b(sc, st);
            int5 xc = {(kvs == 0 ? off_k : off_v) + c, 0, 0, 0, 0};
            const bfloat128 xv = v_bf16_ld_tnsr_b(xc, row);
            const float64_pair_t s0 = v_convert_bf16_to_f32_all_b(b0);
            const float64_pair_t s1 = v_convert_bf16_to_f32_all_b(b1);
            const float64_pair_t s2 = v_convert_bf16_to_f32_all_b(b2);
            const float64_pair_t s3 = v_convert_bf16_to_f32_all_b(xv);
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
            const bfloat128 yb = v_convert_f32_to_bf16_all_b(yf);
            if (write_kvs) {
                int5 kz = {0, 0, i, kvs, 0};
                v_bf16_st_tnsr(kz, kvs_t, yb);
            }
            int5 dst = {0, pos % W, i, 0, 0};
            if (kvs == 0) v_bf16_st_tnsr(dst, kc, yb);
            else v_bf16_st_tnsr(dst, vc, yb);
#ifdef FULL
            int5 dl = {0, pos, i, 0, 0};
            if (kvs == 0) v_bf16_st_tnsr(dl, klin, yb);
            else v_bf16_st_tnsr(dl, vlin, yb);
#endif
            sc[1] = 0;
            v_bf16_st_tnsr(sc, st_out, b1);
            sc[1] = 1;
            v_bf16_st_tnsr(sc, st_out, b2);
            sc[1] = 2;
            v_bf16_st_tnsr(sc, st_out, xv);
        }
    }
}
