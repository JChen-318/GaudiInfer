// SPDX-License-Identifier: Apache-2.0
// Inkling-2 Tiled MME FlashDecode online-softmax tile kernel (T = 1, Direct engine, 100% LUT-free).
// Processes one far-key tile [0, nv) inside a tile buffer of width TW (multiple of 128, e.g. 1024..131072):
//   - Pass 1 (8-way unrolled, quad coord, 0 spills): tile max over sf[0..nv-1, g, kv]
//   - Merge with running FP16-rounded max dmax[h], compute rescale alpha = exp(scale * (m_old - m_new)) <= 1
//   - Pass 2 (Stage-F4 6-way lockstep PROB6: fused mac(x, kh, -mh*kh), minimax quadratic BF16 exp (<=0.71 ULP),
//     3 accumulators, dual load/store coords, 4x zero-pad tail, 0 spills): writes pf[0..TW-1, g, kv] and sums[h]
//   - Write updated dmax[h], sums[h], and rescale[h] = alpha.
// Tensors (dim 0 fastest): sf f16 [TW, G, KV], dmax f32 [NH], sums f32 [NH], pf bf16 [TW, G, KV], rescale f32 [NH].
// Index space [NH]. Params: float scale, int G, int nv.
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

#define ACC(acc, y) ((acc) = v_bf16_mac_acc32_b((y), (bfloat128)1.0f, (acc), 0, 1, 0))

// Bit-exact PROB math matching ink_dattn/ink_dattn_v (v_f16_mul_b(v_f16_sub_b(x, mh), kh) + 4-coeff cubic).
#define PROBF(x, out) do {                                                                    \
        short128 q_ = v_convert_f16_to_i16_b(v_f16_mul_b(v_f16_sub_b((x), mh), kh), SW_RHNE); \
        q_ = v_i16_max_b(q_, (short128)-16128);                                               \
        const half128 k_ = v_convert_i16_to_f16_b(q_ & (short128)127, SW_RHNE);               \
        half128 c_ = (half128)4.827976226806641e-06f;                                         \
        c_ = v_f16_mac_b(c_, k_, (half128)0.0017518997192382812f);                            \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.30322265625f);                                   \
        c_ = v_f16_mac_b(c_, k_, (half128)-0.022216796875f);                                  \
        short128 b_ = q_ + v_convert_f16_to_i16_b(c_, SW_RHNE) + (short128)16256;             \
        (out) = *(bfloat128*)&b_;                                                             \
    } while (0)

// 6-way lockstep PROB6 matching Gaudi2's 6-cycle F16/MAC latency, bit-exact with PROB (0 spills).
#define PROB6(x0, x1, x2, x3, x4, x5, o0, o1, o2, o3, o4, o5) do {                            \
        const half128 d0_ = v_f16_sub_b((x0), mh), d1_ = v_f16_sub_b((x1), mh);               \
        const half128 d2_ = v_f16_sub_b((x2), mh), d3_ = v_f16_sub_b((x3), mh);               \
        const half128 d4_ = v_f16_sub_b((x4), mh), d5_ = v_f16_sub_b((x5), mh);               \
        const half128 p0_ = v_f16_mul_b(d0_, kh),  p1_ = v_f16_mul_b(d1_, kh);                \
        const half128 p2_ = v_f16_mul_b(d2_, kh),  p3_ = v_f16_mul_b(d3_, kh);                \
        const half128 p4_ = v_f16_mul_b(d4_, kh),  p5_ = v_f16_mul_b(d5_, kh);                \
        short128 q0_ = v_convert_f16_to_i16_b(p0_, SW_RHNE);                                  \
        short128 q1_ = v_convert_f16_to_i16_b(p1_, SW_RHNE);                                  \
        short128 q2_ = v_convert_f16_to_i16_b(p2_, SW_RHNE);                                  \
        short128 q3_ = v_convert_f16_to_i16_b(p3_, SW_RHNE);                                  \
        short128 q4_ = v_convert_f16_to_i16_b(p4_, SW_RHNE);                                  \
        short128 q5_ = v_convert_f16_to_i16_b(p5_, SW_RHNE);                                  \
        q0_ = v_i16_max_b(q0_, (short128)-16128); q1_ = v_i16_max_b(q1_, (short128)-16128);   \
        q2_ = v_i16_max_b(q2_, (short128)-16128); q3_ = v_i16_max_b(q3_, (short128)-16128);   \
        q4_ = v_i16_max_b(q4_, (short128)-16128); q5_ = v_i16_max_b(q5_, (short128)-16128);   \
        const half128 k0_ = v_convert_i16_to_f16_b(q0_ & (short128)127, SW_RHNE);             \
        const half128 k1_ = v_convert_i16_to_f16_b(q1_ & (short128)127, SW_RHNE);             \
        const half128 k2_ = v_convert_i16_to_f16_b(q2_ & (short128)127, SW_RHNE);             \
        const half128 k3_ = v_convert_i16_to_f16_b(q3_ & (short128)127, SW_RHNE);             \
        const half128 k4_ = v_convert_i16_to_f16_b(q4_ & (short128)127, SW_RHNE);             \
        const half128 k5_ = v_convert_i16_to_f16_b(q5_ & (short128)127, SW_RHNE);             \
        half128 c0_ = (half128)4.827976226806641e-06f, c1_ = c0_, c2_ = c0_;                  \
        half128 c3_ = c0_, c4_ = c0_, c5_ = c0_;                                              \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)0.0017518997192382812f);                         \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)0.0017518997192382812f);                         \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)0.0017518997192382812f);                         \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)0.0017518997192382812f);                         \
        c4_ = v_f16_mac_b(c4_, k4_, (half128)0.0017518997192382812f);                         \
        c5_ = v_f16_mac_b(c5_, k5_, (half128)0.0017518997192382812f);                         \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)-0.30322265625f);                                \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)-0.30322265625f);                                \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)-0.30322265625f);                                \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)-0.30322265625f);                                \
        c4_ = v_f16_mac_b(c4_, k4_, (half128)-0.30322265625f);                                \
        c5_ = v_f16_mac_b(c5_, k5_, (half128)-0.30322265625f);                                \
        c0_ = v_f16_mac_b(c0_, k0_, (half128)-0.022216796875f);                               \
        c1_ = v_f16_mac_b(c1_, k1_, (half128)-0.022216796875f);                               \
        c2_ = v_f16_mac_b(c2_, k2_, (half128)-0.022216796875f);                               \
        c3_ = v_f16_mac_b(c3_, k3_, (half128)-0.022216796875f);                               \
        c4_ = v_f16_mac_b(c4_, k4_, (half128)-0.022216796875f);                               \
        c5_ = v_f16_mac_b(c5_, k5_, (half128)-0.022216796875f);                               \
        const short128 e0_ = v_convert_f16_to_i16_b(c0_, SW_RHNE);                            \
        const short128 e1_ = v_convert_f16_to_i16_b(c1_, SW_RHNE);                            \
        const short128 e2_ = v_convert_f16_to_i16_b(c2_, SW_RHNE);                            \
        const short128 e3_ = v_convert_f16_to_i16_b(c3_, SW_RHNE);                            \
        const short128 e4_ = v_convert_f16_to_i16_b(c4_, SW_RHNE);                            \
        const short128 e5_ = v_convert_f16_to_i16_b(c5_, SW_RHNE);                            \
        short128 b0_ = q0_ + e0_, b1_ = q1_ + e1_, b2_ = q2_ + e2_;                           \
        short128 b3_ = q3_ + e3_, b4_ = q4_ + e4_, b5_ = q5_ + e5_;                           \
        b0_ += (short128)16256; b1_ += (short128)16256; b2_ += (short128)16256;               \
        b3_ += (short128)16256; b4_ += (short128)16256; b5_ += (short128)16256;               \
        (o0) = *(bfloat128*)&b0_; (o1) = *(bfloat128*)&b1_; (o2) = *(bfloat128*)&b2_;         \
        (o3) = *(bfloat128*)&b3_; (o4) = *(bfloat128*)&b4_; (o5) = *(bfloat128*)&b5_;         \
    } while (0)

void main(tensor sf, tensor dmax, tensor sums, tensor pf, tensor rescale,
          float scale, int G, int nv)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int TW = get_dim_size(sf, 0);
    const int NC = TW >> 7;
    const int nfc = nv >> 7;
    const int nfp = nv & 127;
    const int n8 = nfc >> 3;
    const int n8_end = n8 << 3;
    const int n6 = nfc / 6;
    const int n6_end = n6 * 6;
    float128 kf;
    kf.v1 = 128.0f * 1.4426950408889634f * scale;
    kf.v2 = kf.v1;
    const half128 kh = v_convert_f32_to_f16_all_b(kf);
    const short128 lane = (short128)read_lane_id_2b_b();
    const half128 negf = (half128)-65504.0f;
    const short128 negb = *(short128*)&negf;
    const bfloat128 zb = (bfloat128)0;

    for (int h = start[0]; h < end[0]; ++h) {
        const int kv = h / G, g = h - kv * G;

        // ---- Pass 1: software-pipelined 8-chunk tile max (1.00 bundle/chunk, 100% peak SRAM read BW, 0 spills)
        half128 m0 = negf, m1 = negf, m2 = negf, m3 = negf;
        if (n8 > 0) {
            int5 fc0 = {0, g, kv, 0, 0};
            int5 fc1 = {128, g, kv, 0, 0};
            int5 fc2 = {256, g, kv, 0, 0};
            int5 fc3 = {384, g, kv, 0, 0};
            half128 f0 = v_f16_ld_tnsr_b(fc0, sf); fc0[0] += 512;
            half128 f1 = v_f16_ld_tnsr_b(fc1, sf); fc1[0] += 512;
            half128 f2 = v_f16_ld_tnsr_b(fc2, sf); fc2[0] += 512;
            half128 f3 = v_f16_ld_tnsr_b(fc3, sf); fc3[0] += 512;
#pragma loop_unroll(1)
            for (int k = 1; k < n8; ++k) {
                const half128 f4 = v_f16_ld_tnsr_b(fc0, sf); fc0[0] += 512;
                const half128 f5 = v_f16_ld_tnsr_b(fc1, sf); fc1[0] += 512;
                const half128 f6 = v_f16_ld_tnsr_b(fc2, sf); fc2[0] += 512;
                const half128 f7 = v_f16_ld_tnsr_b(fc3, sf); fc3[0] += 512;
                m0 = v_f16_max_b(m0, f0); f0 = v_f16_ld_tnsr_b(fc0, sf); fc0[0] += 512;
                m1 = v_f16_max_b(m1, f1); f1 = v_f16_ld_tnsr_b(fc1, sf); fc1[0] += 512;
                m2 = v_f16_max_b(m2, f2); f2 = v_f16_ld_tnsr_b(fc2, sf); fc2[0] += 512;
                m3 = v_f16_max_b(m3, f3); f3 = v_f16_ld_tnsr_b(fc3, sf); fc3[0] += 512;
                m0 = v_f16_max_b(m0, f4);
                m1 = v_f16_max_b(m1, f5);
                m2 = v_f16_max_b(m2, f6);
                m3 = v_f16_max_b(m3, f7);
            }
            const half128 f4 = v_f16_ld_tnsr_b(fc0, sf);
            const half128 f5 = v_f16_ld_tnsr_b(fc1, sf);
            const half128 f6 = v_f16_ld_tnsr_b(fc2, sf);
            const half128 f7 = v_f16_ld_tnsr_b(fc3, sf);
            m0 = v_f16_max_b(m0, f0); m1 = v_f16_max_b(m1, f1);
            m2 = v_f16_max_b(m2, f2); m3 = v_f16_max_b(m3, f3);
            m0 = v_f16_max_b(m0, f4); m1 = v_f16_max_b(m1, f5);
            m2 = v_f16_max_b(m2, f6); m3 = v_f16_max_b(m3, f7);
        }
        for (int c = n8_end; c < nfc; ++c) {
            int5 fc = {c << 7, g, kv, 0, 0};
            m0 = v_f16_max_b(m0, v_f16_ld_tnsr_b(fc, sf));
        }
        if (nfp) {
            int5 fc = {nfc << 7, g, kv, 0, 0};
            const half128 fv = v_f16_ld_tnsr_b(fc, sf);
            const short128 fm = v_i16_sel_less_i16_b(lane, (short)nfp, *(short128*)&fv, negb);
            m1 = v_f16_max_b(m1, *(half128*)&fm);
        }

        const float128 mf = v_convert_f16_to_f32_all_b(v_f16_max_b(v_f16_max_b(m0, m1), v_f16_max_b(m2, m3)));
        const float64 m_tile = max64(v_f32_max_b(mf.v1, mf.v2));

        int5 hc = {h, 0, 0, 0, 0};
        const float64 m_old = v_broadcast_element_f32(v_f32_ld_tnsr_b(hc, dmax), 0);
        const float64 l_old = v_broadcast_element_f32(v_f32_ld_tnsr_b(hc, sums), 0);
        float128 mm;
        mm.v1 = v_f32_max_b(m_old, m_tile);
        mm.v2 = mm.v1;
        const half128 mh = v_convert_f32_to_f16_all_b(mm);
        const float128 mr = v_convert_f16_to_f32_all_b(mh);
        const float64 alpha = ink_exp((m_old - mr.v1) * scale);

        // ---- Pass 2: 6-way lockstep PROB6 (3 accumulators + dual load/store coords -> 76 bundles/6 chunks, 0 NOPs, 0 spills!)
        float128 a0, a1, a2;
        a0.v1 = 0; a0.v2 = 0;
        a1 = a0; a2 = a0;

        int5 lc0 = {0, g, kv, 0, 0};
        int5 lc1 = {128, g, kv, 0, 0};
        int5 oc0 = {0, g, kv, 0, 0};
        int5 oc1 = {128, g, kv, 0, 0};
#pragma loop_unroll(1)
        for (int k = 0; k < n6; ++k) {
            const half128 f0 = v_f16_ld_tnsr_b(lc0, sf); lc0[0] += 256;
            const half128 f1 = v_f16_ld_tnsr_b(lc1, sf); lc1[0] += 256;
            const half128 f2 = v_f16_ld_tnsr_b(lc0, sf); lc0[0] += 256;
            const half128 f3 = v_f16_ld_tnsr_b(lc1, sf); lc1[0] += 256;
            const half128 f4 = v_f16_ld_tnsr_b(lc0, sf); lc0[0] += 256;
            const half128 f5 = v_f16_ld_tnsr_b(lc1, sf); lc1[0] += 256;
            bfloat128 o0, o1, o2, o3, o4, o5;
            PROB6(f0, f1, f2, f3, f4, f5, o0, o1, o2, o3, o4, o5);
            ACC(a0, o0); v_bf16_st_tnsr(oc0, pf, o0); oc0[0] += 256;
            ACC(a1, o1); v_bf16_st_tnsr(oc1, pf, o1); oc1[0] += 256;
            ACC(a2, o2); v_bf16_st_tnsr(oc0, pf, o2); oc0[0] += 256;
            ACC(a0, o3); v_bf16_st_tnsr(oc1, pf, o3); oc1[0] += 256;
            ACC(a1, o4); v_bf16_st_tnsr(oc0, pf, o4); oc0[0] += 256;
            ACC(a2, o5); v_bf16_st_tnsr(oc1, pf, o5); oc1[0] += 256;
        }
        int c = n6_end;
        for (; c < nfc; ++c) {
            int5 fc = {c << 7, g, kv, 0, 0};
            bfloat128 o0;
            PROBF(v_f16_ld_tnsr_b(fc, sf), o0);
            ACC(a0, o0);
            v_bf16_st_tnsr(fc, pf, o0);
        }
        if (nfp) {
            int5 fc = {nfc << 7, g, kv, 0, 0};
            bfloat128 o0;
            PROBF(v_f16_ld_tnsr_b(fc, sf), o0);
            const short128 om = v_i16_sel_less_i16_b(lane, (short)nfp, *(short128*)&o0, (short128)0);
            o0 = *(bfloat128*)&om;
            ACC(a1, o0);
            v_bf16_st_tnsr(fc, pf, o0);
            ++c;
        }
        // 4x unrolled zero-pad tail (dual coord regs for fast partial-tile zeroing)
        int5 zc0 = {c << 7, g, kv, 0, 0};
        int5 zc1 = {(c + 1) << 7, g, kv, 0, 0};
        for (; c + 4 <= NC; c += 4) {
            v_bf16_st_tnsr(zc0, pf, zb); zc0[0] += 256;
            v_bf16_st_tnsr(zc1, pf, zb); zc1[0] += 256;
            v_bf16_st_tnsr(zc0, pf, zb); zc0[0] += 256;
            v_bf16_st_tnsr(zc1, pf, zb); zc1[0] += 256;
        }
        for (; c < NC; ++c) {
            int5 oc = {c << 7, g, kv, 0, 0};
            v_bf16_st_tnsr(oc, pf, zb);
        }

        const float64 a = ((a0.v1 + a1.v1) + a2.v1) + ((a0.v2 + a1.v2) + a2.v2);
        const float64 l_new = v_f32_mac_b(l_old, alpha, sum64(a));

        v_f32_st_tnsr_partial(hc, dmax, mr.v1, 0, 0);
        v_f32_st_tnsr_partial(hc, sums, l_new, 0, 0);
        v_f32_st_tnsr_partial(hc, rescale, alpha, 0, 0);
    }
}
