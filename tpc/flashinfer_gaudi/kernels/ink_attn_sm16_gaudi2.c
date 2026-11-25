// SPDX-License-Identifier: Apache-2.0
// Inkling-2 near attention for prefill chunks (T % 128 == 0, no far keys), one KV head, the 128-lane FP16 form of
// masked_softmax_f16 (Qwen path): one row per index point (query t of query head g of this KV head),
//   y[j] = s[j] + bias[j] / scale        (FP16; bias from the reversed-distance table, already divided by scale)
//   p[j] = 2^(round(128 log2(e) scale (y[j] - m)) / 128) as BF16 bits, 0 for keys the row cannot see
//   sum  = sum of the BF16 p + exp(scale (sink / scale - m))      (the caller divides PV by sum)
// with m = max(visible y, sink / scale). Keys j: the W ring slots, then the T new rows. Ring slot s holds position
// pos - 1 - ((pos - 1 - s) mod W); distance = query - key position; sliding layers see distances < W, full layers
// every written key with the bias only below RE. Masks and bias offsets come from positions (no tables).
// exp: q = int16((y - m) * 128 log2(e) scale) is 128 x the base-2 logarithm; bits = q + 16256 + cubic(q & 127)
// (masked_softmax_f16_gaudi2.c). ~14 VPU instructions per 128 scores.
// Tensors (dim 0 fastest): sr f16 [W, GT], sn f16 [T, GT], br f16 [RE, NH, T], sink f32 [NH, T], pos i32 [1],
// p bf16 [SN, GT], sum f32 [1, GT]. GT = G * T rows ordered (g, t). Index space [GT].
// Params: scale, G, full, kv (this KV head: query head h = kv * G + g).
#define MAXC 16
// the sink logits: f32 [NH, T], or (SINK_BF16, Direct engine) the BF16 sink columns of the q-side rows (the
// descriptor starts at the first sink column: [NH, T] with the row stride)
#ifdef SINK_BF16
#define SINK_LOAD(C) v_broadcast_element_f32(v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b((C), sink)).v1, 0)
#else
#define SINK_LOAD(C) v_broadcast_element_f32(v_f32_ld_tnsr_b((C), sink), 0)
#endif
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

static inline float64 exp_vec(float64 x) {
    const float64 xc = v_f32_max_b(x, -87.0f);
    const float64 t = xc * 1.4426950408889634f;
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.0558755f;
    p = v_f32_mac_b(p, f, 0.2401397f);
    p = v_f32_mac_b(p, f, 0.6931705f);
    p = v_f32_mac_b(p, f, 1.0f);
    const int64 bits = (n + 127) << 23;
    return p * as_float64(bits);
}

// BF16 bits of p for one 128-score vector
#define PROB(x, out) do {                                                                  \
        short128 q_ = v_convert_f16_to_i16_b(v_f16_mul_b(v_f16_sub_b((x), mh), kh), SW_RHNE); \
        q_ = v_i16_max_b(q_, (short128)-16128);                                            \
        const half128 k_ = v_convert_i16_to_f16_b(q_ & (short128)127, SW_RHNE);            \
        half128 c_ = (half128)4.827976226806641e-06f;                                      \
        c_ = v_f16_mac_b(c_, k_, (half128)0.0017518997192382812f);                          \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.30322265625f);                                 \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.022216796875f);                                \
        short128 b_ = q_ + v_convert_f16_to_i16_b(c_, SW_RHNE) + (short128)16256;          \
        (out) = *(bfloat128*)&b_;                                                          \
    } while (0)
#define ACC(acc, y) ((acc) = v_bf16_mac_acc32_b((y), (bfloat128)1.0f, (acc), 0, 1, 0))
// PROB with the exponent also clamped from above (2^126: far scores may exceed the max they are taken against)
#define PROBF(x, out) do {                                                                 \
        short128 q_ = v_convert_f16_to_i16_b(v_f16_mul_b(v_f16_sub_b((x), mh), kh), SW_RHNE); \
        q_ = v_i16_min_b(v_i16_max_b(q_, (short128)-16128), (short128)16128);              \
        const half128 k_ = v_convert_i16_to_f16_b(q_ & (short128)127, SW_RHNE);            \
        half128 c_ = (half128)4.827976226806641e-06f;                                      \
        c_ = v_f16_mac_b(c_, k_, (half128)0.0017518997192382812f);                          \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.30322265625f);                                 \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.022216796875f);                                \
        short128 b_ = q_ + v_convert_f16_to_i16_b(c_, SW_RHNE) + (short128)16256;          \
        (out) = *(bfloat128*)&b_;                                                          \
    } while (0)

// 4-way lockstep PROB (bit-identical to PROB, hides 4-cycle VPU latency!)
#define PROB4(x0, x1, x2, x3, o0, o1, o2, o3) do {                                            \
        const half128 d0_ = v_f16_sub_b((x0), mh), d1_ = v_f16_sub_b((x1), mh);               \
        const half128 d2_ = v_f16_sub_b((x2), mh), d3_ = v_f16_sub_b((x3), mh);               \
        short128 q0_ = v_convert_f16_to_i16_b(v_f16_mul_b(d0_, kh), SW_RHNE);                 \
        short128 q1_ = v_convert_f16_to_i16_b(v_f16_mul_b(d1_, kh), SW_RHNE);                 \
        short128 q2_ = v_convert_f16_to_i16_b(v_f16_mul_b(d2_, kh), SW_RHNE);                 \
        short128 q3_ = v_convert_f16_to_i16_b(v_f16_mul_b(d3_, kh), SW_RHNE);                 \
        q0_ = v_i16_max_b(q0_, (short128)-16128); q1_ = v_i16_max_b(q1_, (short128)-16128);   \
        q2_ = v_i16_max_b(q2_, (short128)-16128); q3_ = v_i16_max_b(q3_, (short128)-16128);   \
        const half128 k0_ = v_convert_i16_to_f16_b(q0_ & (short128)127, SW_RHNE);             \
        const half128 k1_ = v_convert_i16_to_f16_b(q1_ & (short128)127, SW_RHNE);             \
        const half128 k2_ = v_convert_i16_to_f16_b(q2_ & (short128)127, SW_RHNE);             \
        const half128 k3_ = v_convert_i16_to_f16_b(q3_ & (short128)127, SW_RHNE);             \
        half128 c0_ = (half128)4.827976226806641e-06f, c1_ = c0_, c2_ = c0_, c3_ = c0_;       \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)0.0017518997192382812f);                         \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)0.0017518997192382812f);                         \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)0.0017518997192382812f);                         \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)0.0017518997192382812f);                         \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)-0.30322265625f);                                \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)-0.30322265625f);                                \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)-0.30322265625f);                                \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)-0.30322265625f);                                \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)-0.022216796875f);                               \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)-0.022216796875f);                               \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)-0.022216796875f);                               \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)-0.022216796875f);                               \
        const short128 e0_ = v_convert_f16_to_i16_b(c0_, SW_RHNE);                            \
        const short128 e1_ = v_convert_f16_to_i16_b(c1_, SW_RHNE);                            \
        const short128 e2_ = v_convert_f16_to_i16_b(c2_, SW_RHNE);                            \
        const short128 e3_ = v_convert_f16_to_i16_b(c3_, SW_RHNE);                            \
        short128 b0_ = q0_ + e0_ + (short128)16256, b1_ = q1_ + e1_ + (short128)16256;        \
        short128 b2_ = q2_ + e2_ + (short128)16256, b3_ = q3_ + e3_ + (short128)16256;        \
        (o0) = *(bfloat128*)&b0_; (o1) = *(bfloat128*)&b1_;                                   \
        (o2) = *(bfloat128*)&b2_; (o3) = *(bfloat128*)&b3_;                                   \
    } while (0)
#define PROBF4(x0, x1, x2, x3, o0, o1, o2, o3) do {                                           \
        const half128 d0_ = v_f16_sub_b((x0), mh), d1_ = v_f16_sub_b((x1), mh);               \
        const half128 d2_ = v_f16_sub_b((x2), mh), d3_ = v_f16_sub_b((x3), mh);               \
        short128 q0_ = v_convert_f16_to_i16_b(v_f16_mul_b(d0_, kh), SW_RHNE);                 \
        short128 q1_ = v_convert_f16_to_i16_b(v_f16_mul_b(d1_, kh), SW_RHNE);                 \
        short128 q2_ = v_convert_f16_to_i16_b(v_f16_mul_b(d2_, kh), SW_RHNE);                 \
        short128 q3_ = v_convert_f16_to_i16_b(v_f16_mul_b(d3_, kh), SW_RHNE);                 \
        q0_ = v_i16_min_b(v_i16_max_b(q0_, (short128)-16128), (short128)16128);               \
        q1_ = v_i16_min_b(v_i16_max_b(q1_, (short128)-16128), (short128)16128);               \
        q2_ = v_i16_min_b(v_i16_max_b(q2_, (short128)-16128), (short128)16128);               \
        q3_ = v_i16_min_b(v_i16_max_b(q3_, (short128)-16128), (short128)16128);               \
        const half128 k0_ = v_convert_i16_to_f16_b(q0_ & (short128)127, SW_RHNE);             \
        const half128 k1_ = v_convert_i16_to_f16_b(q1_ & (short128)127, SW_RHNE);             \
        const half128 k2_ = v_convert_i16_to_f16_b(q2_ & (short128)127, SW_RHNE);             \
        const half128 k3_ = v_convert_i16_to_f16_b(q3_ & (short128)127, SW_RHNE);             \
        half128 c0_ = (half128)4.827976226806641e-06f, c1_ = c0_, c2_ = c0_, c3_ = c0_;       \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)0.0017518997192382812f);                         \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)0.0017518997192382812f);                         \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)0.0017518997192382812f);                         \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)0.0017518997192382812f);                         \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)-0.30322265625f);                                \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)-0.30322265625f);                                \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)-0.30322265625f);                                \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)-0.30322265625f);                                \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)-0.022216796875f);                               \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)-0.022216796875f);                               \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)-0.022216796875f);                               \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)-0.022216796875f);                               \
        const short128 e0_ = v_convert_f16_to_i16_b(c0_, SW_RHNE);                            \
        const short128 e1_ = v_convert_f16_to_i16_b(c1_, SW_RHNE);                            \
        const short128 e2_ = v_convert_f16_to_i16_b(c2_, SW_RHNE);                            \
        const short128 e3_ = v_convert_f16_to_i16_b(c3_, SW_RHNE);                            \
        short128 b0_ = q0_ + e0_ + (short128)16256, b1_ = q1_ + e1_ + (short128)16256;        \
        short128 b2_ = q2_ + e2_ + (short128)16256, b3_ = q3_ + e3_ + (short128)16256;        \
        (o0) = *(bfloat128*)&b0_; (o1) = *(bfloat128*)&b1_;                                   \
        (o2) = *(bfloat128*)&b2_; (o3) = *(bfloat128*)&b3_;                                   \
    } while (0)

#ifndef ROW_WINDOW  // (ink_attn_sm16w_gaudi2.c takes the helpers above and brings its own rows and main)
// rows with compile-time chunk counts (W = 1024: 8 ring chunks; T = 128 .. 1024)
#define CR 8
#define NORING 0
#define NOWRAP 0
#define CN 1
#define ROWFN attn_row_1
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#define CN 2
#define ROWFN attn_row_2
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#define CN 4
#define ROWFN attn_row_4
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#define CN 8
#define ROWFN attn_row_8
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#undef NOWRAP
#define NOWRAP 1
#define CN 8
#define ROWFN attn_row_nw_8
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#undef NOWRAP
#define NOWRAP 0
#undef NORING
#define NORING 1
#define CN 1
#define ROWFN attn_row0_1
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#define CN 2
#define ROWFN attn_row0_2
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#define CN 4
#define ROWFN attn_row0_4
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#define CN 8
#define ROWFN attn_row0_8
#include "ink_attn_row16_gaudi2.h"
#undef CN
#undef ROWFN
#undef NORING
#undef NOWRAP
#undef CR

#ifdef ROW_FAR
void main(tensor sf, tensor sr, tensor sn, tensor br, tensor sink, tensor pos_t, tensor pf, tensor p, tensor sums, float scale,
          int G, int full, int kv)
#elif defined(SAVE_MAX)
void main(tensor sr, tensor sn, tensor br, tensor sink, tensor pos_t, tensor p, tensor sums, tensor m_out, float scale, int G,
          int full, int kv)
#else
void main(tensor sr, tensor sn, tensor br, tensor sink, tensor pos_t, tensor p, tensor sums, float scale, int G,
          int full, int kv)
#endif
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int W = get_dim_size(sr, 0);
    const int T = get_dim_size(sn, 0);
    const int RE = get_dim_size(br, 0);
    const int CR = W / 128, CN = T / 128;
    int lt = 0;
    while ((1 << lt) < T) ++lt;                         // T is a power of two
    int5 z = {0, 0, 0, 0, 0};
    const int pos0 = s_i32_ld_g((__global__ int*)gen_addr(z, pos_t));  // the chunk's first position
    const int jw = pos0 > 0 ? (pos0 - 1) & (W - 1) : 0;  // ring slot of the newest ring key (W: a power of two)
#ifdef ROW_FAR
    const int nf = pos0 > W ? pos0 - W : 0;            // far keys: the cache rows before the ring's positions
    const int NFC = (get_dim_size(sf, 0) + 127) / 128;  // far chunks of the bucket (the last one may be partial)
#endif
    const int cw = jw >> 7;                            // its chunk
    const int age_cap = pos0 - 1;                      // ring keys exist up to this age
    const short128 lane = (short128)read_lane_id_2b_b();
    const half128 neg = (half128)-65504.0f;
    const short128 negb = *(short128*)&neg;
    const bfloat128 zero = 0;
    float128 kf;
    kf.v1 = 128.0f * 1.4426950408889634f * scale;
    kf.v2 = kf.v1;
    const half128 kh = v_convert_f32_to_f16_all_b(kf);
    const float isc = ink_srecip(scale);  // (no scalar division: Direct-safe)
    if (age_cap < 0 && W == 1024 && (CN == 1 || CN == 2 || CN == 4 || CN == 8)) {
        // the first chunk (no ring keys written): rows without the ring part
#ifdef ROW_FAR
#define FARGS0 sf, pf, nf, NFC,
#else
#define FARGS0
#endif
#ifdef SAVE_MAX
#define MARGS m_out,
#else
#define MARGS
#endif
        for (int r = start[0]; r < end[0]; ++r) {
            const int t = r & (T - 1);
            const int h = kv * G + (r >> lt);
            const int bmax_eff = full ? RE - 1 - t : 32767;
            if (CN == 8) attn_row0_8(FARGS0 sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, -1, bmax_eff, kh, isc, scale, lane, negb);
            else if (CN == 4) attn_row0_4(FARGS0 sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, -1, bmax_eff, kh, isc, scale, lane, negb);
            else if (CN == 2) attn_row0_2(FARGS0 sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, -1, bmax_eff, kh, isc, scale, lane, negb);
            else attn_row0_1(FARGS0 sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, -1, bmax_eff, kh, isc, scale, lane, negb);
        }
#undef FARGS0
        return;
    }
    half128 yv[MAXC];
    for (int r = start[0]; r < end[0]; ++r) {
        const int t = r & (T - 1);
        const int h = kv * G + (r >> lt);
        // visible ring ages: <= pos0 - 1 (written) and, sliding, distance t + 1 + age <= W - 1
        int max_age = full ? age_cap : (age_cap < W - 2 - t ? age_cap : W - 2 - t);
        if (max_age > W - 1) max_age = W - 1;
        const int bmax = RE - 1 - t;                    // full layers: bias for ages < bmax (distance < RE)
        if (W == 1024 && (CN == 1 || CN == 2 || CN == 4 || CN == 8)) {  // the branch-free rows
            const int bmax_eff = full ? bmax : 32767;
#ifdef ROW_FAR
#define FARGS sf, pf, nf, NFC,
#else
#define FARGS
#endif
            if (CN == 8 && jw == W - 1) attn_row_nw_8(FARGS sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, max_age, bmax_eff, kh, isc, scale, lane, negb);
            else if (CN == 8) attn_row_8(FARGS sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, max_age, bmax_eff, kh, isc, scale, lane, negb);
            else if (CN == 4) attn_row_4(FARGS sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, max_age, bmax_eff, kh, isc, scale, lane, negb);
            else if (CN == 2) attn_row_2(FARGS sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, max_age, bmax_eff, kh, isc, scale, lane, negb);
            else attn_row_1(FARGS sr, sn, br, sink, p, sums, MARGS r, t, h, W, RE, jw, max_age, bmax_eff, kh, isc, scale, lane, negb);
#undef FARGS
#undef MARGS
            continue;
        }
        half128 m0 = neg, m1 = neg, m2 = neg, m3 = neg;
        // ---- chunks: ring chunk c (< CR) holds slots 128c + l; new chunk CR + c holds rows 128c + l.
        // Visible ranges (contiguous chunk index ranges): ring [ra0, ra1) and [rb0, rb1), new [CR, ne). Inside them
        // "fast" chunks need no mask (one bias run); the edge chunks (wrap chunk cw, a partial oldest chunk, the
        // partial last new chunk) take the general path. Full layers take the general path for every ring chunk.
        int ra0 = 0, ra1 = 0, rb0 = 0, rb1 = 0;         // visible ring ranges
        int fa0 = 0, fa1 = 0, fb0 = 0, fb1 = 0;         // their fast parts
        int e0 = -1, e1 = -1;                           // ring edge chunks (besides cw)
        if (full) {
            ra0 = 0; ra1 = CR;                          // all through the general path
        } else if (max_age >= 0) {
            const int lo = jw - max_age;
            if (lo >= 0) {
                const int cl = lo >> 7;
                ra0 = cl; ra1 = cw + 1;
                fa0 = (lo & 127) ? cl + 1 : cl; fa1 = cw;
                if ((lo & 127) && cl != cw) e0 = cl;
            } else {
                const int lo2 = lo + W, c2 = lo2 >> 7;
                ra0 = 0; ra1 = cw + 1;
                fa0 = 0; fa1 = cw;
                if (c2 > cw) {
                    rb0 = c2; rb1 = CR;
                    fb0 = (lo2 & 127) ? c2 + 1 : c2; fb1 = CR;
                    if (lo2 & 127) e1 = c2;
                } else {
                    rb0 = cw + 1; rb1 = CR;
                    fb0 = cw + 1; fb1 = CR;
                }
            }
        }
        const int nfn = (t + 1) >> 7;                  // fully visible new chunks
        const int ne = CR + nfn + (((t + 1) & 127) ? 1 : 0);
        // fast chunks, four at a time (loads first, then the adds)
#define FAST_RING(C) do { \
            const int r0_ = jw - (C) * 128; \
            int5 sc_ = {(C) * 128, r, 0, 0, 0}; \
            int5 bc_ = {RE - 2 - t - (r0_ >= 0 ? r0_ : r0_ + W), h, t, 0, 0}; \
            yv[(C)] = v_f16_add_b(v_f16_ld_tnsr_b(sc_, sr), v_f16_ld_tnsr_b(bc_, br)); \
        } while (0)
#define FAST_NEW(C) do { \
            int5 sc_ = {((C) - CR) * 128, r, 0, 0, 0}; \
            int5 bc_ = {RE - 1 - t + ((C) - CR) * 128, h, t, 0, 0}; \
            yv[(C)] = v_f16_add_b(v_f16_ld_tnsr_b(sc_, sn), v_f16_ld_tnsr_b(bc_, br)); \
        } while (0)
#define FAST_RANGE(LO, HI, ONE) do { \
            int c_ = (LO); \
            for (; c_ + 4 <= (HI); c_ += 4) { \
                ONE(c_); ONE(c_ + 1); ONE(c_ + 2); ONE(c_ + 3); \
                m0 = v_f16_max_b(m0, yv[c_]); m1 = v_f16_max_b(m1, yv[c_ + 1]); \
                m2 = v_f16_max_b(m2, yv[c_ + 2]); m3 = v_f16_max_b(m3, yv[c_ + 3]); \
            } \
            for (; c_ < (HI); ++c_) { ONE(c_); m0 = v_f16_max_b(m0, yv[c_]); } \
        } while (0)
        FAST_RANGE(fa0, fa1, FAST_RING);
        FAST_RANGE(fb0, fb1, FAST_RING);
        FAST_RANGE(CR, CR + nfn, FAST_NEW);
        // edge chunks: general ring path (wrap selects, bias range, visibility mask) / masked last new chunk
#define EDGE_RING(C) do { \
            const int r0_ = jw - (C) * 128; \
            const int wrap_ = r0_ >= 0 && r0_ < 127; \
            const int base_ = r0_ >= 0 ? r0_ : r0_ + W; \
            int5 sc_ = {(C) * 128, r, 0, 0, 0}; \
            const half128 sv_ = v_f16_ld_tnsr_b(sc_, sr); \
            int5 bc_ = {RE - 2 - t - base_, h, t, 0, 0}; \
            half128 b_ = v_f16_ld_tnsr_b(bc_, br); \
            short128 age_ = (short128)base_ - lane; \
            if (wrap_) { \
                bc_[0] = RE - 2 - t - base_ - W; \
                const half128 b2_ = v_f16_ld_tnsr_b(bc_, br); \
                const short128 sel_ = v_i16_sel_less_i16_b(lane, (short)(r0_ + 1), *(short128*)&b_, *(short128*)&b2_); \
                b_ = *(half128*)&sel_; \
                age_ = v_i16_sel_less_i16_b(lane, (short)(r0_ + 1), age_, age_ + (short128)W); \
            } \
            if (full) { \
                const short128 bs_ = v_i16_sel_less_i16_b(age_, (short)bmax, *(short128*)&b_, (short128)0); \
                b_ = *(half128*)&bs_; \
            } \
            const half128 y_ = v_f16_add_b(sv_, b_); \
            const short128 ym_ = v_i16_sel_grt_i16_b(age_, (short)max_age, negb, *(short128*)&y_); \
            yv[(C)] = *(half128*)&ym_; \
            m1 = v_f16_max_b(m1, yv[(C)]); \
        } while (0)
        if (full) {
            if (max_age >= 0)
                for (int c = 0; c < CR; ++c) EDGE_RING(c);
            else
                ra1 = 0;
        } else if (max_age >= 0) {
            EDGE_RING(cw);
            if (e0 >= 0) EDGE_RING(e0);
            if (e1 >= 0) EDGE_RING(e1);
        }
        if (ne > CR + nfn) {
            const int c = CR + nfn;
            int5 sc = {nfn * 128, r, 0, 0, 0};
            int5 bc = {RE - 1 - t + nfn * 128, h, t, 0, 0};
            const half128 y = v_f16_add_b(v_f16_ld_tnsr_b(sc, sn), v_f16_ld_tnsr_b(bc, br));
            const short128 ym = v_i16_sel_grt_i16_b(lane, (short)(t - nfn * 128), negb, *(short128*)&y);
            yv[c] = *(half128*)&ym;
            m2 = v_f16_max_b(m2, yv[c]);
        }
#undef FAST_RING
#undef FAST_NEW
#undef FAST_RANGE
#undef EDGE_RING
        // the sink logit of head h in y units (a vector load at its own position, broadcast; no scalar load)
        int5 skc = {h, t, 0, 0, 0};
        const float64 ysk = SINK_LOAD(skc) * isc;
        const float128 mf = v_convert_f16_to_f32_all_b(v_f16_max_b(v_f16_max_b(m0, m1), v_f16_max_b(m2, m3)));
        float128 mm;
        mm.v1 = v_f32_max_b(max64(v_f32_max_b(mf.v1, mf.v2)), ysk);
        mm.v2 = mm.v1;
        const half128 mh = v_convert_f32_to_f16_all_b(mm);
        const float128 mr = v_convert_f16_to_f32_all_b(mh);  // the m every exponent uses
        float128 a0, a1, a2, a3;
        a0.v1 = 0; a0.v2 = 0;
        a1 = a0; a2 = a0; a3 = a0;
        // probabilities over the visible ranges, zeros elsewhere
#define PROB_RANGE(LO, HI) do { \
            int c_ = (LO); \
            for (; c_ + 4 <= (HI); c_ += 4) { \
                bfloat128 o0_, o1_, o2_, o3_; \
                PROB(yv[c_], o0_); PROB(yv[c_ + 1], o1_); PROB(yv[c_ + 2], o2_); PROB(yv[c_ + 3], o3_); \
                ACC(a0, o0_); ACC(a1, o1_); ACC(a2, o2_); ACC(a3, o3_); \
                int5 oc_ = {c_ * 128, r, 0, 0, 0}; \
                v_bf16_st_tnsr(oc_, p, o0_); oc_[0] += 128; \
                v_bf16_st_tnsr(oc_, p, o1_); oc_[0] += 128; \
                v_bf16_st_tnsr(oc_, p, o2_); oc_[0] += 128; \
                v_bf16_st_tnsr(oc_, p, o3_); \
            } \
            for (; c_ < (HI); ++c_) { \
                bfloat128 o0_; \
                PROB(yv[c_], o0_); \
                ACC(a0, o0_); \
                int5 oc_ = {c_ * 128, r, 0, 0, 0}; \
                v_bf16_st_tnsr(oc_, p, o0_); \
            } \
        } while (0)
#define ZERO_RANGE(LO, HI) do { \
            int5 oc_ = {(LO) * 128, r, 0, 0, 0}; \
            for (int c_ = (LO); c_ < (HI); ++c_) { v_bf16_st_tnsr(oc_, p, zero); oc_[0] += 128; } \
        } while (0)
        PROB_RANGE(ra0, ra1);
        PROB_RANGE(rb0, rb1);
        PROB_RANGE(CR, ne);
        if (ra1 <= ra0) {
            ZERO_RANGE(0, CR);
        } else if (rb1 > rb0) {
            ZERO_RANGE(ra1, rb0);
        } else {
            ZERO_RANGE(0, ra0);
            ZERO_RANGE(ra1, CR);
        }
        ZERO_RANGE(ne, CR + CN);
#undef PROB_RANGE
#undef ZERO_RANGE
        const float64 a = ((a0.v1 + a1.v1) + (a2.v1 + a3.v1)) + ((a0.v2 + a1.v2) + (a2.v2 + a3.v2));
        const float64 tot = sum64(a) + exp_vec((ysk - mr.v1) * scale);
        int5 oc = {0, r, 0, 0, 0};
        v_f32_st_tnsr(oc, sums, tot);
#ifdef SAVE_MAX
        v_f32_st_tnsr(oc, m_out, mr.v1);
#endif
    }
}
#endif  // ROW_WINDOW
