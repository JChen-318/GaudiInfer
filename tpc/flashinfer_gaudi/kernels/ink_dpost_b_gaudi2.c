// SPDX-License-Identifier: Apache-2.0
// Inkling-2 batched decode (Bp rows, Direct engine): fused q prep, r/sink prep, and per-slot k/v short convs
// straight into contiguous [max_slots] KV rings and [max_slots][2] conv pools using bmeta[8, Bp].
// Dummy padded rows (bmeta[3, b] == 0) never modify ring or conv state.
// Tensors (dim 0 fastest):
//   0 row           bf16 [N_total]
//   1 qs            bf16 [128, NH]
//   2 q             bf16 [128, NH, Bp]
//   3 rs            f32  [16, NH]
//   4 r_out         bf16 [16, NH, Bp]
//   5 sink          f32  [NH, Bp]
//   6 conv_in       bf16 [KVD, 3, 2, 2*max_slots] (stride dim3 = conv_slot_bytes / 2)
//   7 taps          bf16 [KVD, 4, 2]
//   8 conv_out      bf16 [KVD, 3, 2, 2*max_slots]
//   9 bmeta         i32  [8, Bp] (0:slot, 1:side, 2:pos, 3:valid, 4:tau_bits)
//  10 kc            bf16 [D, W, KV, max_slots]
//  11 vc            bf16 [D, W, KV, max_slots]
//  12 b_kvs         bf16 [D, KV, 2, Bp]
// Index space: [(2*NH + 2*KV) * Bp]. LUT-free.
#include "norm_math_gaudi2.h"
#include "ink_math_gaudi2.h"

void main(tensor row, tensor qs, tensor q, tensor rs, tensor r_out, tensor sink,
          tensor conv_in, tensor taps, tensor conv_out, tensor bmeta, tensor kc, tensor vc, tensor b_kvs,
          float thr_q, float thr_r, float isc, float cscale, float eps,
          int off_rel, int off_sink, int off_k, int off_v, int nrow, int three)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NH = get_dim_size(q, 1);
    const int Bp = get_dim_size(q, 2);
    const int KV = get_dim_size(kc, 2);
    const int W = get_dim_size(kc, 1);
    const int KVD = KV * 128;
    (void)KVD;
    const int stride_p = 2 * NH + 2 * KV;
    const float64 lanes = v_convert_i32_to_f32_b((int64)read_lane_id_4b_b(), 0);

    for (int idx = start[0]; idx < end[0]; ++idx) {
        const int b = idx / stride_p;
        const int p = idx - b * stride_p;
        int5 m0 = {3, b, 0, 0, 0};
        if (s_i32_ld_g((__global__ int*)gen_addr(m0, bmeta)) == 0) continue;
        m0[0] = 0; const int slot = s_i32_ld_g((__global__ int*)gen_addr(m0, bmeta));
        m0[0] = 1; const int side = s_i32_ld_g((__global__ int*)gen_addr(m0, bmeta));
        m0[0] = 2; const int pos = s_i32_ld_g((__global__ int*)gen_addr(m0, bmeta));
#ifdef FULL
        m0[0] = 4; const float tau_s = s_f32_ld_g((__global__ float*)gen_addr(m0, bmeta));
        const float64 tv = tau_s;
#endif
        const int rbase_q = b * nrow;
        const int rbase_k = (three ? (Bp + b) * nrow : b * nrow) + off_k;
        const int rbase_v = (three ? (2 * Bp + b) * nrow : b * nrow) + off_v;

        if (p < NH) {
            const int h = p;
            int5 scc = {0, h, 0, 0, 0};
            const float64_pair_t sc = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(scc, qs));
            int5 ic = {rbase_q + h * 128, 0, 0, 0, 0};
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
            int5 oc = {0, h, b, 0, 0};
            v_bf16_st_tnsr(oc, q, qb);
        } else if (p < 2 * NH) {
            const int h = p - NH;
            int5 rc = {rbase_q + off_rel + h * 16, 0, 0, 0, 0};
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
            int5 oc = {0, h, b, 0, 0};
            v_bf16_st_tnsr_partial(oc, r_out, v_convert_f32_to_bf16_all_b(rb), 15, 0);
            int5 kc0 = {rbase_q + off_sink + h, 0, 0, 0, 0};
            const float64_pair_t sv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(kc0, row));
            int5 so0 = {h, b, 0, 0, 0};
            v_f32_st_tnsr_partial(so0, sink, sv.v1, 0, 0);
        } else {
            const int i = (p - 2 * NH) % KV;
            const int kvs = (p - 2 * NH) / KV;
            const int c = i * 128;
            const int rkv = kvs == 0 ? rbase_k : rbase_v;
            int5 wc = {c, 0, kvs, 0, 0};
            float64_pair_t w0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 1; float64_pair_t w1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 2; float64_pair_t w2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
            wc[1] = 3; float64_pair_t w3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, taps));
#ifdef FULL
            {
                int5 dc = {rkv + KVD + i * 4, 0, 0, 0, 0};
                const float64_pair_t dv = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(dc, row));
                const float64 d0 = v_broadcast_element_f32(dv.v1, 0), d2 = v_broadcast_element_f32(dv.v1, 1);
                const float64 d1 = v_broadcast_element_f32(dv.v2, 0), d3 = v_broadcast_element_f32(dv.v2, 1);
                w3.v1 = w3.v1 + d0; w3.v2 = w3.v2 + d0;
                w2.v1 = w2.v1 + d1; w2.v2 = w2.v2 + d1;
                w1.v1 = w1.v1 + d2; w1.v2 = w1.v2 + d2;
                w0.v1 = w0.v1 + d3; w0.v2 = w0.v2 + d3;
            }
#endif
            const int ss_in = slot * 2 + side;
            const int ss_out = slot * 2 + (1 - side);
            int5 sc = {c, 0, kvs, ss_in, 0};
            const bfloat128 b0 = v_bf16_ld_tnsr_b(sc, conv_in);
            sc[1] = 1; const bfloat128 b1 = v_bf16_ld_tnsr_b(sc, conv_in);
            sc[1] = 2; const bfloat128 b2 = v_bf16_ld_tnsr_b(sc, conv_in);
            int5 xc = {rkv + c, 0, 0, 0, 0};
            const bfloat128 xv = v_bf16_ld_tnsr_b(xc, row);
            const float64_pair_t s0 = v_convert_bf16_to_f32_all_b(b0);
            const float64_pair_t s1 = v_convert_bf16_to_f32_all_b(b1);
            const float64_pair_t s2 = v_convert_bf16_to_f32_all_b(b2);
            const float64_pair_t s3 = v_convert_bf16_to_f32_all_b(xv);
            float64_pair_t o;
            o.v1 = (((s0.v1 * w0.v1 + s1.v1 * w1.v1) + s2.v1 * w2.v1) + s3.v1 * w3.v1 + s3.v1) * cscale;
            o.v2 = (((s0.v2 * w0.v2 + s1.v2 * w1.v2) + s2.v2 * w2.v2) + s3.v2 * w3.v2 + s3.v2) * cscale;
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
            int5 dst = {0, pos % W, i, slot, 0};
            if (kvs == 0) v_bf16_st_tnsr(dst, kc, yb);
            else v_bf16_st_tnsr(dst, vc, yb);
#ifdef FULL
            int5 kz = {0, i, kvs, b, 0};
            v_bf16_st_tnsr(kz, b_kvs, yb);
#endif
            int5 so2 = {c, 0, kvs, ss_out, 0};
            v_bf16_st_tnsr(so2, conv_out, b1);
            so2[1] = 1; v_bf16_st_tnsr(so2, conv_out, b2);
            so2[1] = 2; v_bf16_st_tnsr(so2, conv_out, xv);
        }
    }
}
