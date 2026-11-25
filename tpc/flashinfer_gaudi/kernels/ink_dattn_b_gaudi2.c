// SPDX-License-Identifier: Apache-2.0
// Inkling-2 batched decode ring attention softmax (Bp rows, Direct engine, 100% LUT-free).
// Exact same arithmetic as ink_dattn_gaudi2.c, with per-row pos read from bmeta[2, b].
// Tensors (dim 0 fastest): sc f16 [W, G, KV, Bp], bt f16 [RE, NH, Bp], sink f32 [NH, Bp], bmeta i32 [8, Bp],
// p bf16 [W, G, KV, Bp], sums f32 [NH, Bp] (+ dmax f32 [NH, Bp] when WRITE_MAX).
// Index space [NH * Bp]. Params: scale, G.
#include "ink_math_gaudi2.h"

static inline float64 sum64(float64 x) {
    const int all = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                    SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, all, x);
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, all, x);
    x += v_f32_mov_group_b(x, 0xffffffff, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN | SW_DUAL_GROUP1_EN |
                           SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, x);
    uint64 lanes = read_lane_id_4b_b() & 7;
    uint64 c4 = ((lanes ^ 4) | 0x80) * 0x01010101;
    uint64 c2 = ((lanes ^ 2) | 0x80) * 0x01010101;
    uint64 c1 = ((lanes ^ 1) | 0x80) * 0x01010101;
    x += v_f32_shuffle_b(x, *((uchar256*)&c4), 0, x);
    x += v_f32_shuffle_b(x, *((uchar256*)&c2), 0, x);
    x += v_f32_shuffle_b(x, *((uchar256*)&c1), 0, x);
    return x;
}
static inline float64 max64(float64 x) {
    const int all = SW_WR_LOWER_GROUP0 | SW_WR_UPPER_GROUP0 | SW_WR_LOWER_GROUP1 | SW_WR_UPPER_GROUP1 |
                    SW_WR_LOWER_GROUP2 | SW_WR_UPPER_GROUP2 | SW_WR_LOWER_GROUP3 | SW_WR_UPPER_GROUP3;
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, all, x));
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, all, x));
    x = v_f32_max_b(x, v_f32_mov_group_b(x, 0xffffffff, SW_GROUP0_EN | SW_GROUP1_EN | SW_DUAL_GROUP0_EN |
                                         SW_DUAL_GROUP1_EN | SW_DUAL_GROUP2_EN | SW_DUAL_GROUP3_EN, x));
    uint64 lanes = read_lane_id_4b_b() & 7;
    uint64 c4 = ((lanes ^ 4) | 0x80) * 0x01010101;
    uint64 c2 = ((lanes ^ 2) | 0x80) * 0x01010101;
    uint64 c1 = ((lanes ^ 1) | 0x80) * 0x01010101;
    x = v_f32_max_b(x, v_f32_shuffle_b(x, *((uchar256*)&c4), 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, *((uchar256*)&c2), 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, *((uchar256*)&c1), 0, x));
    return x;
}

#define PROB(x, out) do {                                                                  \
        short128 q_ = v_convert_f16_to_i16_b(v_f16_mul_b(v_f16_sub_b((x), mh), kh), SW_RHNE); \
        q_ = v_i16_max_b(q_, (short128)-16128);                                            \
        const half128 k_ = v_convert_i16_to_f16_b(q_ & (short128)127, SW_RHNE);            \
        half128 c_ = (half128)4.827976226806641e-06f;                                      \
        c_ = v_f16_mac_b(c_, k_, (half128)0.0017518997192382812f);                         \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.30322265625f);                                \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.022216796875f);                               \
        short128 b_ = q_ + v_convert_f16_to_i16_b(c_, SW_RHNE) + (short128)16256;          \
        (out) = *(bfloat128*)&b_;                                                          \
    } while (0)

#define MAXC 16

void main(tensor sc, tensor bt, tensor sink, tensor bmeta, tensor p, tensor sums,
#ifdef WRITE_MAX
          tensor dmax,
#endif
          float scale, int G)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int W = get_dim_size(sc, 0);
    const int RE = get_dim_size(bt, 0);
    const int NH = get_dim_size(bt, 1);
    const int CR = W / 128;
    const short128 lane = (short128)read_lane_id_2b_b();
    const half128 neg = (half128)-65504.0f;
    const short128 negb = *(short128*)&neg;
    float128 kf;
    kf.v1 = 128.0f * 1.4426950408889634f * scale;
    kf.v2 = kf.v1;
    const half128 kh = v_convert_f32_to_f16_all_b(kf);
    const float isc = ink_srecip(scale);
    half128 yv[MAXC];

    for (int idx = start[0]; idx < end[0]; ++idx) {
        const int b = idx / NH;
        const int h = idx - b * NH;
        const int kv = h / G, g = h - kv * G;
        int5 mc = {3, b, 0, 0, 0};
        if (s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta)) == 0) continue;
        mc[0] = 2;
        const int pos = s_i32_ld_g((__global__ int*)gen_addr(mc, bmeta));
        const int jw = pos % W;
        const int amax = pos < W - 1 ? pos : W - 1;

        half128 m0 = neg, m1 = neg;
        for (int c = 0; c < CR; ++c) {
            const int r0 = jw - c * 128;
            int5 scc = {c * 128, g, kv, b, 0};
            const half128 s = v_f16_ld_tnsr_b(scc, sc);
            int5 bc = {RE - 1 - r0, h, b, 0, 0};
            const half128 ba = v_f16_ld_tnsr_b(bc, bt);
            bc[0] = RE - 1 - r0 - W;
            const half128 bb = v_f16_ld_tnsr_b(bc, bt);
            const short128 sel = v_i16_sel_less_i16_b(lane, (short)(r0 + 1), *(short128*)&ba, *(short128*)&bb);
            short128 age = (short128)r0 - lane;
            age = v_i16_sel_less_i16_b(lane, (short)(r0 + 1), age, age + (short128)W);
            const half128 y = v_f16_add_b(s, *(half128*)&sel);
            const short128 ym = v_i16_sel_grt_i16_b(age, (short)amax, negb, *(short128*)&y);
            yv[c] = *(half128*)&ym;
            if (c & 1) m1 = v_f16_max_b(m1, yv[c]);
            else m0 = v_f16_max_b(m0, yv[c]);
        }
        int5 kc = {h, b, 0, 0, 0};
        const float64 ysk = v_broadcast_element_f32(v_f32_ld_tnsr_b(kc, sink), 0) * isc;
        const float128 mf = v_convert_f16_to_f32_all_b(v_f16_max_b(m0, m1));
        float128 mm;
        mm.v1 = v_f32_max_b(max64(v_f32_max_b(mf.v1, mf.v2)), ysk);
        mm.v2 = mm.v1;
        const half128 mh = v_convert_f32_to_f16_all_b(mm);
        const float128 mr = v_convert_f16_to_f32_all_b(mh);
        float128 a0, a1;
        a0.v1 = 0; a0.v2 = 0;
        a1 = a0;
        for (int c = 0; c < CR; ++c) {
            bfloat128 o;
            PROB(yv[c], o);
            const int r0 = jw - c * 128;
            short128 age = (short128)r0 - lane;
            age = v_i16_sel_less_i16_b(lane, (short)(r0 + 1), age, age + (short128)W);
            const short128 ob = v_i16_sel_grt_i16_b(age, (short)amax, (short128)0, *(short128*)&o);
            o = *(bfloat128*)&ob;
            if (c & 1) a1 = v_bf16_mac_acc32_b(o, (bfloat128)1.0f, a1, 0, 1, 0);
            else a0 = v_bf16_mac_acc32_b(o, (bfloat128)1.0f, a0, 0, 1, 0);
            int5 oc = {c * 128, g, kv, b, 0};
            v_bf16_st_tnsr(oc, p, o);
        }
        const float64 a = (a0.v1 + a1.v1) + (a0.v2 + a1.v2);
        const float64 tot = sum64(a) + ink_exp((ysk - mr.v1) * scale);
        int5 so = {h, b, 0, 0, 0};
        v_f32_st_tnsr_partial(so, sums, tot, 0, 0);
#ifdef WRITE_MAX
        v_f32_st_tnsr_partial(so, dmax, mr.v1, 0, 0);
#endif
    }
}
