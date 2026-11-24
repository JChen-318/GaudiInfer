// SPDX-License-Identifier: Apache-2.0
// Inkling-2 B=1/multi-T routed-expert down-proj GEMV (CH = 1, NG = 9, 256 rows/point, LUT-free).
// Shared template compiled with -DNVF4 (ink_dmoe_dn9_nvf4_gaudi2.c) or -DMXF8 (ink_dmoe_dn9_mxf8_gaudi2.c).
// Tensors (dim 0 fastest): act bf16 [F, J, T], idx i32 [J, T], w u8 [H, F/2 or F, E], bs f8 [H, F/16 or F/32, E],
// si f32 [H, E], ys bf16 [H, J, T]. Index space [NG * J * T] (NG = H / 256 = 9). Params: none.

#ifdef NVF4
// NVFP4 weights (exact, natural K order, 256-col interleaved): w u8 [H, F/2, E], bs f8 [H, F/16, E] (fp8_143(scale/8)),
// si f32 [H, E] (holds g[e] * 512). Block accumulation per 16 k + 5-op ash unpack + 4-step SWP.
#define NVF4_UNPACK_ASH(WW, LO, HI) do { \
        const char256 ws_ = *((char256*)&(WW)); \
        const char256 w4_ = ws_ << 4; \
        const char256 ah_ = v_i8_ash_b(ws_, (char)-2, 0); \
        const char256 al_ = v_i8_ash_b(w4_, (char)-2, 0); \
        (LO) = *((uchar256*)&al_) & (uchar256)0x9c; \
        (HI) = *((uchar256*)&ah_) & (uchar256)0x9c; \
    } while (0)
void main(tensor act, tensor idx, tensor w, tensor bs, tensor si, tensor ys)
#elif defined(MXF8)
// Native MXFP8 weights (exact, natural K order, 256-col interleaved): w u8 [H, F, E] (E4M3FN codes),
// bs f8 [H, F/32, E] (fp8_143(2^(s - s_max)) <= 1), si f32 [H, E] (g[e] = 2^(s_max - 127), constant per expert).
// Exact table-free E4M3FN -> BF16 conversion + block accumulation per 32 k + 4-step SWP.
void main(tensor act, tensor idx, tensor w, tensor bs, tensor si, tensor ys)
#else
#error "Define NVF4 or MXF8"
#endif
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
#ifdef NVF4
    const int K = get_dim_size(w, 1) * 2;
    const int NB = K >> 4;
#elif defined(MXF8)
    const int K = get_dim_size(w, 1);
    const int NB = K >> 5;
#endif
    const int NG = get_dim_size(w, 0) >> 8;  // H / 256 = 9
    const int JJ = get_dim_size(idx, 0);
    const int T = get_dim_size(idx, 1);
    const int interleave24 = (NG * JJ == 72);
    const int wpt = 3 * T;

    for (int pp = start[0]; pp < end[0]; ++pp) {
        int wi = pp;
        if (interleave24) {
            const int tpc_id = pp / wpt;
            const int wave = pp - tpc_id * wpt;
            wi = wave * 24 + tpc_id;
        }
        const int u = wi / NG;
        const int g = wi - u * NG;
        const int t = u / JJ;
        const int j = u - t * JJ;

        int5 ic = {j, t, 0, 0, 0};
        const int e = s_i32_ld_g((__global__ int*)gen_addr(ic, idx));
        const int n0 = g << 8;
        if (e < 0) {
            int5 oc = {n0, j, t, 0, 0};
            v_bf16_st_tnsr(oc, ys, (bfloat128)0);
            oc[0] = n0 + 128;
            v_bf16_st_tnsr(oc, ys, (bfloat128)0);
            continue;
        }

        float128 a0 = {0}, b0 = {0};
        int5 wc = {n0, 0, e, 0, 0};
        int5 xc = {0, j, t, 0, 0};
#ifdef NVF4
        int5 bc = {n0, 0, e, 0, 0};
        wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
        xc[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
        xc[0] = 1; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
        xc[0] = 2; bfloat128 xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
        xc[0] = 3; bfloat128 xe = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
#define PIPE_DN(WW, XA, XB, R_NEXT, K_NEXT, DO_W, DO_X) do { \
            uchar256 lo_, hi_; \
            NVF4_UNPACK_ASH((WW), lo_, hi_); \
            if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
            const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
            const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
            ba = v_bf16_mac_acc32_b(pl_.v1, (XA), ba, 0); \
            bb = v_bf16_mac_acc32_b(pl_.v2, (XA), bb, 0); \
            if (DO_X) { xc[0] = (K_NEXT); (XA) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act)); } \
            ba = v_bf16_mac_acc32_b(ph_.v1, (XB), ba, 0); \
            bb = v_bf16_mac_acc32_b(ph_.v2, (XB), bb, 0); \
            if (DO_X) { xc[0] = (K_NEXT) + 1; (XB) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act)); } \
        } while (0)
#pragma loop_unroll(1)
        for (int kb = 0; kb < NB - 1; ++kb) {
            const int r = kb << 3, k = kb << 4;
            float128 ba = {0}, bb = {0};
            PIPE_DN(w0, xa, xb, r + 4,  k + 4,  1, 1);
            PIPE_DN(w1, xd, xe, r + 5,  k + 6,  1, 1);
            PIPE_DN(w2, xa, xb, r + 6,  k + 8,  1, 1);
            PIPE_DN(w3, xd, xe, r + 7,  k + 10, 1, 1);
            PIPE_DN(w0, xa, xb, r + 8,  k + 12, 1, 1);
            PIPE_DN(w1, xd, xe, r + 9,  k + 14, 1, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            PIPE_DN(w2, xa, xb, r + 10, k + 16, 1, 1);
            PIPE_DN(w3, xd, xe, r + 11, k + 18, 1, 1);
            a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
            b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
        }
        {
            const int kb = NB - 1, r = kb << 3, k = kb << 4;
            float128 ba = {0}, bb = {0};
            PIPE_DN(w0, xa, xb, r + 4, k + 4,  1, 1);
            PIPE_DN(w1, xd, xe, r + 5, k + 6,  1, 1);
            PIPE_DN(w2, xa, xb, r + 6, k + 8,  1, 1);
            PIPE_DN(w3, xd, xe, r + 7, k + 10, 1, 1);
            PIPE_DN(w0, xa, xb, 0,     k + 12, 0, 1);
            PIPE_DN(w1, xd, xe, 0,     k + 14, 0, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            PIPE_DN(w2, xa, xb, 0,     0,      0, 0);
            PIPE_DN(w3, xd, xe, 0,     0,      0, 0);
            a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
            b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
        }
#undef PIPE_DN
#elif defined(MXF8)
        int5 bc = {n0, 0, e, 0, 0};
        wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
        bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
        xc[0] = 0; bfloat128 x0 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
        xc[0] = 1; bfloat128 x1 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
        xc[0] = 2; bfloat128 x2 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
        xc[0] = 3; bfloat128 x3 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act));
#define PIPE_DN_MX(WW, XX, K_NEXT, DO_W, DO_X) do { \
            const uchar256 e_ = (WW) & (uchar256)0x78; \
            const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
            const uchar256 wa_ = (WW) - d8_; \
            if (DO_W) { wc[1] = (K_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
            const uchar256 d80_ = d8_ << 4; \
            const bfloat128_pair_t p_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
            const ushort128_pair_t d_ = v_convert_u8_to_u16_all_b(d80_); \
            const ushort128 r1_ = *((ushort128*)&p_.v1) + d_.v1; \
            const ushort128 r2_ = *((ushort128*)&p_.v2) + d_.v2; \
            ba = v_bf16_mac_acc32_b(*((bfloat128*)&r1_), (XX), ba, 0); \
            bb = v_bf16_mac_acc32_b(*((bfloat128*)&r2_), (XX), bb, 0); \
            if (DO_X) { xc[0] = (K_NEXT); (XX) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc, act)); } \
        } while (0)
#pragma loop_unroll(1)
        for (int kb = 0; kb < NB - 1; ++kb) {
            const int k0 = kb << 5;
            float128 ba = {0}, bb = {0};
            PIPE_DN_MX(w0, x0, k0 + 4, 1, 1);  PIPE_DN_MX(w1, x1, k0 + 5, 1, 1);
            PIPE_DN_MX(w2, x2, k0 + 6, 1, 1);  PIPE_DN_MX(w3, x3, k0 + 7, 1, 1);
            PIPE_DN_MX(w0, x0, k0 + 8, 1, 1);  PIPE_DN_MX(w1, x1, k0 + 9, 1, 1);
            PIPE_DN_MX(w2, x2, k0 + 10, 1, 1); PIPE_DN_MX(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
            for (int sub = 1; sub < 3; ++sub) {
                const int ks = k0 + (sub << 3);
                PIPE_DN_MX(w0, x0, ks + 4, 1, 1);  PIPE_DN_MX(w1, x1, ks + 5, 1, 1);
                PIPE_DN_MX(w2, x2, ks + 6, 1, 1);  PIPE_DN_MX(w3, x3, ks + 7, 1, 1);
                PIPE_DN_MX(w0, x0, ks + 8, 1, 1);  PIPE_DN_MX(w1, x1, ks + 9, 1, 1);
                PIPE_DN_MX(w2, x2, ks + 10, 1, 1); PIPE_DN_MX(w3, x3, ks + 11, 1, 1);
            }
            PIPE_DN_MX(w0, x0, k0 + 28, 1, 1); PIPE_DN_MX(w1, x1, k0 + 29, 1, 1);
            PIPE_DN_MX(w2, x2, k0 + 30, 1, 1); PIPE_DN_MX(w3, x3, k0 + 31, 1, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            PIPE_DN_MX(w0, x0, k0 + 32, 1, 1); PIPE_DN_MX(w1, x1, k0 + 33, 1, 1);
            PIPE_DN_MX(w2, x2, k0 + 34, 1, 1); PIPE_DN_MX(w3, x3, k0 + 35, 1, 1);
            a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
            b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
        }
        {
            const int k0 = (NB - 1) << 5;
            float128 ba = {0}, bb = {0};
            PIPE_DN_MX(w0, x0, k0 + 4, 1, 1);  PIPE_DN_MX(w1, x1, k0 + 5, 1, 1);
            PIPE_DN_MX(w2, x2, k0 + 6, 1, 1);  PIPE_DN_MX(w3, x3, k0 + 7, 1, 1);
            PIPE_DN_MX(w0, x0, k0 + 8, 1, 1);  PIPE_DN_MX(w1, x1, k0 + 9, 1, 1);
            PIPE_DN_MX(w2, x2, k0 + 10, 1, 1); PIPE_DN_MX(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
            for (int sub = 1; sub < 3; ++sub) {
                const int ks = k0 + (sub << 3);
                PIPE_DN_MX(w0, x0, ks + 4, 1, 1);  PIPE_DN_MX(w1, x1, ks + 5, 1, 1);
                PIPE_DN_MX(w2, x2, ks + 6, 1, 1);  PIPE_DN_MX(w3, x3, ks + 7, 1, 1);
                PIPE_DN_MX(w0, x0, ks + 8, 1, 1);  PIPE_DN_MX(w1, x1, ks + 9, 1, 1);
                PIPE_DN_MX(w2, x2, ks + 10, 1, 1); PIPE_DN_MX(w3, x3, ks + 11, 1, 1);
            }
            PIPE_DN_MX(w0, x0, k0 + 28, 1, 1); PIPE_DN_MX(w1, x1, k0 + 29, 1, 1);
            PIPE_DN_MX(w2, x2, k0 + 30, 1, 1); PIPE_DN_MX(w3, x3, k0 + 31, 1, 1);
            const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
            PIPE_DN_MX(w0, x0, 0, 0, 0); PIPE_DN_MX(w1, x1, 0, 0, 0);
            PIPE_DN_MX(w2, x2, 0, 0, 0); PIPE_DN_MX(w3, x3, 0, 0, 0);
            a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
            b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
        }
#undef PIPE_DN_MX
#endif

        int5 sc = {n0, e, 0, 0, 0};
        const float64 sa0 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 64;
        const float64 sa1 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 128;
        const float64 sb0 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 192;
        const float64 sb1 = v_f32_ld_tnsr_b(sc, si);
        float64_pair_t ra, rb;
        ra.v1 = a0.v1 * sa0; ra.v2 = a0.v2 * sa1;
        rb.v1 = b0.v1 * sb0; rb.v2 = b0.v2 * sb1;
        int5 oc = {n0, j, t, 0, 0};
        v_bf16_st_tnsr(oc, ys, v_convert_f32_to_bf16_all_b(ra));
        oc[0] = n0 + 128;
        v_bf16_st_tnsr(oc, ys, v_convert_f32_to_bf16_all_b(rb));
    }
}
