// SPDX-License-Identifier: Apache-2.0
// One row of ink_attn_sm16w for the rows of band B (t in [128 B, 128 B + 128), compile-time): the nine 128-key chunks
// u = B .. B + 8 of the key axis [ring | new] that hold the row's window (keys x with t < x <= t + 1024). Only the
// first chunk (lanes <= t - 128 B hidden) and the last one (lanes > t - 128 B hidden) need a mask. NORING: the first
// chunk of a prompt (no ring keys): the new chunks 0 .. B, zeros into the window's ring chunks.
// Chunk u adds to accumulator u mod 4, in ascending u: the visible keys in ink_attn_sm16's order (same sums).
// Included by ink_attn_sm16w_gaudi2.c once per (B, NORING) as ROWFN.
static inline void ROWFN(tensor sr, tensor sn, tensor br, tensor sink, tensor p, tensor sums, int r, int t, int h, int RE,
                         half128 kh, float isc, float scale, short128 lane, short128 negb)
{
    const int tl = t - 128 * B;  // 0 .. 127
    // y of window chunk J (u = B + J): ring chunk u (u < 8) or new chunk u - 8; the bias run is contiguous over u
#define LOADY_(J, Y) half128 Y; do { \
        int5 bc_ = {RE - 1025 - t + 128 * (B + (J)), h, t, 0, 0}; \
        half128 sv_; \
        if (B + (J) < 8) { \
            int5 sc_ = {(B + (J)) * 128, r, 0, 0, 0}; \
            sv_ = v_f16_ld_tnsr_b(sc_, sr); \
        } else { \
            int5 sc_ = {(B + (J) - 8) * 128, r, 0, 0, 0}; \
            sv_ = v_f16_ld_tnsr_b(sc_, sn); \
        } \
        Y = v_f16_add_b(sv_, v_f16_ld_tnsr_b(bc_, br)); \
    } while (0)
#define ACCJ_(J, O) do { \
        if (((B + (J)) & 3) == 0) ACC(a0, O); \
        else if (((B + (J)) & 3) == 1) ACC(a1, O); \
        else if (((B + (J)) & 3) == 2) ACC(a2, O); \
        else ACC(a3, O); \
    } while (0)
#define STJ_(J, O) do { \
        int5 oc_ = {(B + (J)) * 128, r, 0, 0, 0}; \
        v_bf16_st_tnsr(oc_, p, (O)); \
    } while (0)
    half128 m0, m1, m2, m3;
#if !NORING
    LOADY_(0, e0);
    LOADY_(1, y1); LOADY_(2, y2); LOADY_(3, y3); LOADY_(4, y4);
    LOADY_(5, y5); LOADY_(6, y6); LOADY_(7, y7);
    LOADY_(8, e8);
    const short128 s0_ = v_i16_sel_grt_i16_b(lane, (short)tl, *(short128*)&e0, negb);
    const short128 s8_ = v_i16_sel_grt_i16_b(lane, (short)tl, negb, *(short128*)&e8);
    const half128 y0 = *(half128*)&s0_, y8 = *(half128*)&s8_;
    m0 = v_f16_max_b(v_f16_max_b(y0, y4), y8);
    m1 = v_f16_max_b(y1, y5);
    m2 = v_f16_max_b(y2, y6);
    m3 = v_f16_max_b(y3, y7);
#else
    // new chunks 0 .. B = window chunks 8 - B .. 8 (the last one masked)
    LOADY_(8, e8);
    const short128 s8_ = v_i16_sel_grt_i16_b(lane, (short)tl, negb, *(short128*)&e8);
    const half128 y8 = *(half128*)&s8_;
    m0 = y8;
    m1 = (half128)-65504.0f;
    m2 = m1;
    m3 = m1;
#if B >= 1
    LOADY_(7, y7);
    m1 = y7;
#endif
#if B >= 2
    LOADY_(6, y6);
    m2 = y6;
#endif
#if B >= 3
    LOADY_(5, y5);
    m3 = y5;
#endif
#if B >= 4
    LOADY_(4, y4);
    m0 = v_f16_max_b(m0, y4);
#endif
#if B >= 5
    LOADY_(3, y3);
    m1 = v_f16_max_b(m1, y3);
#endif
#if B >= 6
    LOADY_(2, y2);
    m2 = v_f16_max_b(m2, y2);
#endif
#if B >= 7
    LOADY_(1, y1);
    m3 = v_f16_max_b(m3, y1);
#endif
#endif
    // the sink logit of head h in y units (a vector load at its own position, broadcast)
    int5 skc = {h, t, 0, 0, 0};
    const float64 ysk = SINK_LOAD(skc) * isc;
    const float128 mf = v_convert_f16_to_f32_all_b(v_f16_max_b(v_f16_max_b(m0, m1), v_f16_max_b(m2, m3)));
    float128 mm;
    mm.v1 = v_f32_max_b(max64(v_f32_max_b(mf.v1, mf.v2)), ysk);
    mm.v2 = mm.v1;
    const half128 mh = v_convert_f32_to_f16_all_b(mm);
    const float128 mr = v_convert_f16_to_f32_all_b(mh);
    float128 a0, a1, a2, a3;
    a0.v1 = 0; a0.v2 = 0;
    a1 = a0; a2 = a0; a3 = a0;
#define OUT1_(J, Y) do { \
        bfloat128 o_; \
        PROB(Y, o_); \
        ACCJ_(J, o_); \
        STJ_(J, o_); \
    } while (0)
#define OUT4_(J, Y0, Y1, Y2, Y3) do { \
        bfloat128 o0_, o1_, o2_, o3_; \
        PROB4(Y0, Y1, Y2, Y3, o0_, o1_, o2_, o3_); \
        ACCJ_(J, o0_); ACCJ_((J) + 1, o1_); ACCJ_((J) + 2, o2_); ACCJ_((J) + 3, o3_); \
        STJ_(J, o0_); STJ_((J) + 1, o1_); STJ_((J) + 2, o2_); STJ_((J) + 3, o3_); \
    } while (0)
#if !NORING
    OUT4_(0, y0, y1, y2, y3);
    OUT4_(4, y4, y5, y6, y7);
    OUT1_(8, y8);
#else
    {   // no ring key is written yet: zeros into the window's ring chunks B .. 7
        const bfloat128 zb_ = 0;
        int5 oc_ = {B * 128, r, 0, 0, 0};
        for (int c_ = B; c_ < 8; ++c_) { v_bf16_st_tnsr(oc_, p, zb_); oc_[0] += 128; }
    }
    // ascending u: window chunks 8 - B .. 8
#if B == 7
    OUT4_(1, y1, y2, y3, y4);
    OUT4_(5, y5, y6, y7, y8);
#elif B == 6
    OUT1_(2, y2); OUT1_(3, y3); OUT1_(4, y4);
    OUT4_(5, y5, y6, y7, y8);
#elif B == 5
    OUT1_(3, y3); OUT1_(4, y4);
    OUT4_(5, y5, y6, y7, y8);
#elif B == 4
    OUT1_(4, y4);
    OUT4_(5, y5, y6, y7, y8);
#elif B == 3
    OUT4_(5, y5, y6, y7, y8);
#elif B == 2
    OUT1_(6, y6); OUT1_(7, y7); OUT1_(8, y8);
#elif B == 1
    OUT1_(7, y7); OUT1_(8, y8);
#else
    OUT1_(8, y8);
#endif
#endif
#undef OUT1_
#undef OUT4_
#undef LOADY_
#undef ACCJ_
#undef STJ_
    const float64 a = ((a0.v1 + a1.v1) + (a2.v1 + a3.v1)) + ((a0.v2 + a1.v2) + (a2.v2 + a3.v2));
    const float64 tot = sum64(a) + exp_vec((ysk - mr.v1) * scale);
    int5 oc = {0, r, 0, 0, 0};
    v_f32_st_tnsr(oc, sums, tot);
}
