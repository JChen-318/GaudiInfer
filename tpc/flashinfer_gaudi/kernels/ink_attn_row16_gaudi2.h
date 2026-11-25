// SPDX-License-Identifier: Apache-2.0
// One row of ink_attn_sm16 with compile-time chunk counts (CR ring chunks, CN new chunks): every chunk is processed
// without branches (visibility, the wrap chunk and the full-layer bias cut are lane selects), the scores stay in
// registers. Included by ink_attn_sm16_gaudi2.c once per CN (and NORING: the first chunk, no ring keys) as ROWFN.
static inline void ROWFN(
#ifdef ROW_FAR
                         tensor sf, tensor pf, int nf, int NFC,
#endif
                         tensor sr, tensor sn, tensor br, tensor sink, tensor p, tensor sums,
#ifdef SAVE_MAX
                         tensor m_out,
#endif
                         int r, int t, int h,
                         int W, int RE, int jw, int max_age, int bmax, half128 kh, float isc, float scale,
                         short128 lane, short128 negb)
{
    half128 m0 = (half128)-65504.0f, m1 = m0, m2 = m0, m3 = m0;
#if NOWRAP
// Fast ring load when jw == W - 1 (all T = 1024 prefill chunks with pos0 >= 1024): zero wrap, 1 bias load per chunk!
#define RING_(C, Y) half128 Y; do { \
        const int base_ = (W - 1) - (C) * 128; \
        int5 sc_ = {(C) * 128, r, 0, 0, 0}; \
        const half128 sv_ = v_f16_ld_tnsr_b(sc_, sr); \
        int5 bc_ = {RE - 2 - t - base_, h, t, 0, 0}; \
        const half128 b1_ = v_f16_ld_tnsr_b(bc_, br); \
        const short128 age_ = (short128)base_ - lane; \
        const short128 bcut_ = v_i16_sel_less_i16_b(age_, (short)bmax, *(short128*)&b1_, (short128)0); \
        const half128 yv_ = v_f16_add_b(sv_, *(half128*)&bcut_); \
        const short128 ym_ = v_i16_sel_grt_i16_b(age_, (short)max_age, negb, *(short128*)&yv_); \
        Y = *(half128*)&ym_; \
    } while (0)
#else
#define RING_(C, Y) half128 Y; do { \
        const int r0_ = jw - (C) * 128; \
        const int base_ = r0_ >= 0 ? r0_ : r0_ + W; \
        const int wl_ = (r0_ >= 0 && r0_ < 127) ? r0_ + 1 : 128; \
        int5 sc_ = {(C) * 128, r, 0, 0, 0}; \
        const half128 sv_ = v_f16_ld_tnsr_b(sc_, sr); \
        int5 bc_ = {RE - 2 - t - base_, h, t, 0, 0}; \
        const half128 b1_ = v_f16_ld_tnsr_b(bc_, br); \
        bc_[0] = RE - 2 - t - base_ - W; \
        const half128 b2_ = v_f16_ld_tnsr_b(bc_, br); \
        const short128 bs_ = v_i16_sel_less_i16_b(lane, (short)wl_, *(short128*)&b1_, *(short128*)&b2_); \
        short128 age_ = (short128)base_ - lane; \
        age_ = v_i16_sel_less_i16_b(lane, (short)wl_, age_, age_ + (short128)W); \
        const short128 bcut_ = v_i16_sel_less_i16_b(age_, (short)bmax, bs_, (short128)0); \
        const half128 yv_ = v_f16_add_b(sv_, *(half128*)&bcut_); \
        const short128 ym_ = v_i16_sel_grt_i16_b(age_, (short)max_age, negb, *(short128*)&yv_); \
        Y = *(half128*)&ym_; \
    } while (0)
#endif
#define NEW_(C, Y) half128 Y; do { \
        int5 sc_ = {(C) * 128, r, 0, 0, 0}; \
        int5 bc_ = {RE - 1 - t + (C) * 128, h, t, 0, 0}; \
        const half128 yv_ = v_f16_add_b(v_f16_ld_tnsr_b(sc_, sn), v_f16_ld_tnsr_b(bc_, br)); \
        const short128 ym_ = v_i16_sel_grt_i16_b(lane, (short)(t - (C) * 128), negb, *(short128*)&yv_); \
        Y = *(half128*)&ym_; \
    } while (0)
#if !NORING
    RING_(0, y0); RING_(1, y1); RING_(2, y2); RING_(3, y3); RING_(4, y4); RING_(5, y5); RING_(6, y6); RING_(7, y7);
    m0 = v_f16_max_b(v_f16_max_b(m0, y0), y4);
    m1 = v_f16_max_b(v_f16_max_b(m1, y1), y5);
    m2 = v_f16_max_b(v_f16_max_b(m2, y2), y6);
    m3 = v_f16_max_b(v_f16_max_b(m3, y3), y7);
#endif
    NEW_(0, n0);
    m0 = v_f16_max_b(m0, n0);
#if CN > 1
    NEW_(1, n1);
    m1 = v_f16_max_b(m1, n1);
#endif
#if CN > 2
    NEW_(2, n2); NEW_(3, n3);
    m2 = v_f16_max_b(m2, n2);
    m3 = v_f16_max_b(m3, n3);
#endif
#if CN > 4
    NEW_(4, n4); NEW_(5, n5); NEW_(6, n6); NEW_(7, n7);
    m0 = v_f16_max_b(m0, n4);
    m1 = v_f16_max_b(m1, n5);
    m2 = v_f16_max_b(m2, n6);
    m3 = v_f16_max_b(m3, n7);
#endif
#undef RING_
#undef NEW_
#ifdef ROW_FAR
    // far keys (cache rows j < nf, no bias): an exact max pass over the far chunks first. (Taking the exponents against
    // the near max alone breaks: the attention sink at position 0 is a far key at long contexts and its probability
    // relative to the near max overflows the F32 accumulation of P . V.)
    const int nfc = nf >> 7, nfp = nf & 127;
#ifndef FAR_ONEPASS
    {
        int c = 0;
        for (; c + 8 <= nfc; c += 8) {
            half128 f[8];
            for (int u = 0; u < 8; ++u) {
                int5 fc_ = {(c + u) * 128, r, 0, 0, 0};
                f[u] = v_f16_ld_tnsr_b(fc_, sf);
            }
            m0 = v_f16_max_b(m0, v_f16_max_b(f[0], f[4]));
            m1 = v_f16_max_b(m1, v_f16_max_b(f[1], f[5]));
            m2 = v_f16_max_b(m2, v_f16_max_b(f[2], f[6]));
            m3 = v_f16_max_b(m3, v_f16_max_b(f[3], f[7]));
        }
        for (; c < nfc; ++c) {
            int5 fc_ = {c * 128, r, 0, 0, 0};
            m0 = v_f16_max_b(m0, v_f16_ld_tnsr_b(fc_, sf));
        }
        if (nfp) {
            int5 fc_ = {nfc * 128, r, 0, 0, 0};
            const half128 fv_ = v_f16_ld_tnsr_b(fc_, sf);
            const short128 fm_ = v_i16_sel_less_i16_b(lane, (short)nfp, *(short128*)&fv_, negb);
            m1 = v_f16_max_b(m1, *(half128*)&fm_);
        }
    }
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
#define OUT_(Y, C, A) do { \
        bfloat128 o_; \
        PROB(Y, o_); \
        ACC(A, o_); \
        int5 oc_ = {(C) * 128, r, 0, 0, 0}; \
        v_bf16_st_tnsr(oc_, p, o_); \
    } while (0)
#define OUT4_(Y0, Y1, Y2, Y3, C0) do { \
        bfloat128 o0_, o1_, o2_, o3_; \
        PROB4(Y0, Y1, Y2, Y3, o0_, o1_, o2_, o3_); \
        ACC(a0, o0_); ACC(a1, o1_); ACC(a2, o2_); ACC(a3, o3_); \
        int5 oc_ = {(C0) * 128, r, 0, 0, 0}; \
        v_bf16_st_tnsr(oc_, p, o0_); oc_[0] += 128; \
        v_bf16_st_tnsr(oc_, p, o1_); oc_[0] += 128; \
        v_bf16_st_tnsr(oc_, p, o2_); oc_[0] += 128; \
        v_bf16_st_tnsr(oc_, p, o3_); \
    } while (0)
#if !NORING
    OUT4_(y0, y1, y2, y3, 0);
    OUT4_(y4, y5, y6, y7, 4);
#else
    {   // no ring key is written yet (the first chunk): zero probabilities
        const bfloat128 zb_ = 0;
        int5 oc_ = {0, r, 0, 0, 0};
        for (int c_ = 0; c_ < CR; ++c_) { v_bf16_st_tnsr(oc_, p, zb_); oc_[0] += 128; }
    }
#endif
#if CN >= 4
    OUT4_(n0, n1, n2, n3, CR);
#if CN > 4
    OUT4_(n4, n5, n6, n7, CR + 4);
#endif
#else
    OUT_(n0, CR, a0);
#if CN > 1
    OUT_(n1, CR + 1, a1);
#endif
#endif
#undef OUT_
#undef OUT4_
#ifdef ROW_FAR
    {   // far probabilities (zeros past the valid rows)
        int c = 0;
        for (; c + 4 <= nfc; c += 4) {
            half128 f0, f1, f2, f3;
            int5 fc_ = {c * 128, r, 0, 0, 0};
            f0 = v_f16_ld_tnsr_b(fc_, sf); fc_[0] += 128;
            f1 = v_f16_ld_tnsr_b(fc_, sf); fc_[0] += 128;
            f2 = v_f16_ld_tnsr_b(fc_, sf); fc_[0] += 128;
            f3 = v_f16_ld_tnsr_b(fc_, sf);
            bfloat128 o0, o1, o2, o3;
            PROBF4(f0, f1, f2, f3, o0, o1, o2, o3);
            ACC(a0, o0); ACC(a1, o1); ACC(a2, o2); ACC(a3, o3);
            int5 oc_ = {c * 128, r, 0, 0, 0};
            v_bf16_st_tnsr(oc_, pf, o0); oc_[0] += 128;
            v_bf16_st_tnsr(oc_, pf, o1); oc_[0] += 128;
            v_bf16_st_tnsr(oc_, pf, o2); oc_[0] += 128;
            v_bf16_st_tnsr(oc_, pf, o3);
        }
        for (; c < nfc; ++c) {
            int5 fc_ = {c * 128, r, 0, 0, 0};
            const half128 f0 = v_f16_ld_tnsr_b(fc_, sf);
            bfloat128 o0;
            PROBF(f0, o0);
            ACC(a0, o0);
            v_bf16_st_tnsr(fc_, pf, o0);
        }
        if (nfp) {
            int5 fc_ = {nfc * 128, r, 0, 0, 0};
            const half128 fv_ = v_f16_ld_tnsr_b(fc_, sf);
            bfloat128 o0;
            PROBF(fv_, o0);
            const short128 om_ = v_i16_sel_less_i16_b(lane, (short)nfp, *(short128*)&o0, (short128)0);
            o0 = *(bfloat128*)&om_;
            ACC(a1, o0);
            v_bf16_st_tnsr(fc_, pf, o0);
            ++c;
        }
        const bfloat128 zb = 0;
        for (; c < NFC; ++c) {
            int5 fc_ = {c * 128, r, 0, 0, 0};
            v_bf16_st_tnsr(fc_, pf, zb);
        }
    }
#endif
    const float64 a = ((a0.v1 + a1.v1) + (a2.v1 + a3.v1)) + ((a0.v2 + a1.v2) + (a2.v2 + a3.v2));
    const float64 tot = sum64(a) + exp_vec((ysk - mr.v1) * scale);
    int5 oc = {0, r, 0, 0, 0};
    v_f32_st_tnsr(oc, sums, tot);
#ifdef SAVE_MAX
    v_f32_st_tnsr(oc, m_out, mr.v1);
#endif
}
