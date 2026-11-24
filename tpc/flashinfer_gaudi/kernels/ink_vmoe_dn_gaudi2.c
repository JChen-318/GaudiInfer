// SPDX-License-Identifier: Apache-2.0
// Inkling-2 verify / batched-decode grouped routed-expert down-proj GEMV (CH = 1, NG = 9, LUT-free).
// Shared template compiled with -DNVF4 (ink_vmoe_dn_nvf4_gaudi2.c) or -DMXF8 (ink_vmoe_dn_mxf8_gaudi2.c).
// Uses Thue-Morse paired-snake 24-TPC mapping (when NG * J == 72) for 3x finer load balance & zero straggler stacking.
// Tensors (dim 0 fastest):
//   act  bf16 [F, J, T]
//   grp  i32  [64, 4]      (from ink_vmoe_plan: row0=e, row1=cnt, row2=ts0, row3=ts1)
//   w    u8   [H, F/2 or F, E]
//   bs   f8   [H, F/16 or F/32, E]
//   si   f32  [H, E]
//   ys   bf16 [H, J, T]
// Index space [NG * J * T] (NG = H / 256 = 9 -> 9 * J * T). Params: none.
#if !defined(NVF4) && !defined(MXF8)
#error "Define NVF4 or MXF8"
#endif

__local__ float64 acc_buf[4][4];

#define STORE_ACC1(M_IDX, A0, B0) do { \
        acc_buf[M_IDX][0] = (A0).v1; acc_buf[M_IDX][1] = (A0).v2; \
        acc_buf[M_IDX][2] = (B0).v1; acc_buf[M_IDX][3] = (B0).v2; \
    } while (0)

#ifdef NVF4
#define NVF4_UNPACK_ASH(WW, LO, HI) do { \
        const char256 ws_ = *((char256*)&(WW)); \
        const char256 w4_ = ws_ << 4; \
        const char256 ah_ = v_i8_ash_b(ws_, (char)-2, 0); \
        const char256 al_ = v_i8_ash_b(w4_, (char)-2, 0); \
        (LO) = *((uchar256*)&al_) & (uchar256)0x9c; \
        (HI) = *((uchar256*)&ah_) & (uchar256)0x9c; \
    } while (0)
#endif
#define ZERO_ACC_VLM(M_IDX) do { \
        acc_buf[M_IDX][0] = (float64)0.0f; acc_buf[M_IDX][1] = (float64)0.0f; \
        acc_buf[M_IDX][2] = (float64)0.0f; acc_buf[M_IDX][3] = (float64)0.0f; \
    } while (0)
#define REDUCE_ACC_VLM(M_IDX, BA, BB, SC) do { \
        float128 oa_, ob_; \
        oa_.v1 = acc_buf[M_IDX][0]; oa_.v2 = acc_buf[M_IDX][1]; \
        ob_.v1 = acc_buf[M_IDX][2]; ob_.v2 = acc_buf[M_IDX][3]; \
        oa_ = v_f32_mac_x2_b((BA), (SC).v1, (SC).v3, oa_, 0); \
        ob_ = v_f32_mac_x2_b((BB), (SC).v2, (SC).v4, ob_, 0); \
        acc_buf[M_IDX][0] = oa_.v1; acc_buf[M_IDX][1] = oa_.v2; \
        acc_buf[M_IDX][2] = ob_.v1; acc_buf[M_IDX][3] = ob_.v2; \
    } while (0)

void main(tensor act, tensor grp, tensor w, tensor bs, tensor si, tensor ys)
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
    const int J = get_dim_size(ys, 1);
    const int T = get_dim_size(ys, 2);
    const int interleave24 = (NG * J == 72);
    const int wpt = 3 * T;
    const int single_tpc_slice = interleave24 && ((end[0] - start[0]) == wpt);

    for (int pp = start[0]; pp < end[0]; ++pp) {
        int work_idx = pp;
        if (interleave24) {
            const int tpc_id = pp / wpt;
            const int wave = pp - tpc_id * wpt;
            const int rev = (wave ^ (wave >> 1)) & 1;
            const int lane24 = rev ? (23 - tpc_id) : tpc_id;
            work_idx = wave * 24 + lane24;
        }
        const int u = work_idx / NG;
        const int g = work_idx - u * NG;

        int5 gc = {u, 0, 0, 0, 0};
        const int e = s_i32_ld_g((__global__ int*)gen_addr(gc, grp));
        if (e < 0) {
            if (single_tpc_slice) break;
            continue;
        }
        gc[1] = 1;
        const int cnt = s_i32_ld_g((__global__ int*)gen_addr(gc, grp));
        if (cnt <= 0) continue;
        gc[1] = 2;
        const int tsp = s_i32_ld_g((__global__ int*)gen_addr(gc, grp));
        const int nm = cnt < 4 ? cnt : 4;

        const int n0 = g << 8;
        int5 wc = {n0, 0, e, 0, 0};

        const int c0 = tsp & 0xff;
        const int c1 = (tsp >> 8) & 0xff;
        const int c2 = (tsp >> 16) & 0xff;
        const int c3 = (tsp >> 24) & 0xff;
        const int t0 = c0 >> 4, s0 = c0 & 15;
        const int t1 = c1 >> 4, s1 = c1 & 15;
        const int t2 = c2 >> 4, s2 = c2 & 15;
        const int t3 = c3 >> 4, s3 = c3 & 15;
#ifdef NVF4
        int5 bc = {n0, 0, e, 0, 0};
        if (nm == 1) {
            float128 a0 = {0}, b0 = {0};
            int5 xc0 = {0, s0, t0, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc0[0] = 1; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc0[0] = 2; bfloat128 xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc0[0] = 3; bfloat128 xe = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
#define NVD1(WW, XA, XB, R_NEXT, K_NEXT, DO_W, DO_X) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                ba = v_bf16_mac_acc32_b(pl_.v1, (XA), ba, 0); \
                bb = v_bf16_mac_acc32_b(pl_.v2, (XA), bb, 0); \
                if (DO_X) { xc0[0] = (K_NEXT); (XA) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); } \
                ba = v_bf16_mac_acc32_b(ph_.v1, (XB), ba, 0); \
                bb = v_bf16_mac_acc32_b(ph_.v2, (XB), bb, 0); \
                if (DO_X) { xc0[0] = (K_NEXT) + 1; (XB) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); } \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 ba = {0}, bb = {0};
                NVD1(w0, xa, xb, r + 4,  k + 4,  1, 1);
                NVD1(w1, xd, xe, r + 5,  k + 6,  1, 1);
                NVD1(w2, xa, xb, r + 6,  k + 8,  1, 1);
                NVD1(w3, xd, xe, r + 7,  k + 10, 1, 1);
                NVD1(w0, xa, xb, r + 8,  k + 12, 1, 1);
                NVD1(w1, xd, xe, r + 9,  k + 14, 1, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                NVD1(w2, xa, xb, r + 10, k + 16, 1, 1);
                NVD1(w3, xd, xe, r + 11, k + 18, 1, 1);
                a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
                b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 ba = {0}, bb = {0};
                NVD1(w0, xa, xb, r + 4, k + 4,  1, 1);
                NVD1(w1, xd, xe, r + 5, k + 6,  1, 1);
                NVD1(w2, xa, xb, r + 6, k + 8,  1, 1);
                NVD1(w3, xd, xe, r + 7, k + 10, 1, 1);
                NVD1(w0, xa, xb, 0,     k + 12, 0, 1);
                NVD1(w1, xd, xe, 0,     k + 14, 0, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                NVD1(w2, xa, xb, 0,     0,      0, 0);
                NVD1(w3, xd, xe, 0,     0,      0, 0);
                a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
                b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
            }
#undef NVD1
            STORE_ACC1(0, a0, b0);
        } else if (nm == 2) {
            float128 a0 = {0}, b0 = {0}, a1 = {0}, b1 = {0};
            int5 xc0 = {0, s0, t0, 0, 0};
            int5 xc1 = {0, s1, t1, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define NVD2(WW, KK, R_NEXT, DO_W) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                xc0[0] = (KK); const bfloat128 xa0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); \
                xc1[0] = (KK); const bfloat128 xb0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                ba0 = v_bf16_mac_acc32_b(pl_.v1, xa0_, ba0, 0); bb0 = v_bf16_mac_acc32_b(pl_.v2, xa0_, bb0, 0); \
                ba1 = v_bf16_mac_acc32_b(pl_.v1, xb0_, ba1, 0); bb1 = v_bf16_mac_acc32_b(pl_.v2, xb0_, bb1, 0); \
                xc0[0] = (KK) + 1; const bfloat128 xa1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); \
                xc1[0] = (KK) + 1; const bfloat128 xb1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                ba0 = v_bf16_mac_acc32_b(ph_.v1, xa1_, ba0, 0); bb0 = v_bf16_mac_acc32_b(ph_.v2, xa1_, bb0, 0); \
                ba1 = v_bf16_mac_acc32_b(ph_.v1, xb1_, ba1, 0); bb1 = v_bf16_mac_acc32_b(ph_.v2, xb1_, bb1, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                NVD2(w0, k + 0,  r + 2, 1); NVD2(w1, k + 2,  r + 3, 1);
                NVD2(w0, k + 4,  r + 4, 1); NVD2(w1, k + 6,  r + 5, 1);
                NVD2(w0, k + 8,  r + 6, 1); NVD2(w1, k + 10, r + 7, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                NVD2(w0, k + 12, r + 8, 1); NVD2(w1, k + 14, r + 9, 1);
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                NVD2(w0, k + 0,  r + 2, 1); NVD2(w1, k + 2,  r + 3, 1);
                NVD2(w0, k + 4,  r + 4, 1); NVD2(w1, k + 6,  r + 5, 1);
                NVD2(w0, k + 8,  r + 6, 1); NVD2(w1, k + 10, r + 7, 1);
                NVD2(w0, k + 12, 0,     0); NVD2(w1, k + 14, 0,     0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
            }
#undef NVD2
            STORE_ACC1(0, a0, b0); STORE_ACC1(1, a1, b1);
        } else if (nm == 3) {
            int5 xc0 = {0, s0, t0, 0, 0};
            int5 xc1 = {0, s1, t1, 0, 0};
            int5 xc2 = {0, s2, t2, 0, 0};
            float128 a0 = {0}, b0 = {0}, a1 = {0}, b1 = {0};
            ZERO_ACC_VLM(2);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define NVD3B(WW, KK, R_NEXT, DO_W) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                xc0[0] = (KK); const bfloat128 xa0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); \
                xc1[0] = (KK); const bfloat128 xb0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); \
                xc2[0] = (KK); const bfloat128 xc0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act)); \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                ba0 = v_bf16_mac_acc32_b(pl_.v1, xa0_, ba0, 0); bb0 = v_bf16_mac_acc32_b(pl_.v2, xa0_, bb0, 0); \
                ba1 = v_bf16_mac_acc32_b(pl_.v1, xb0_, ba1, 0); bb1 = v_bf16_mac_acc32_b(pl_.v2, xb0_, bb1, 0); \
                ba2 = v_bf16_mac_acc32_b(pl_.v1, xc0_, ba2, 0); bb2 = v_bf16_mac_acc32_b(pl_.v2, xc0_, bb2, 0); \
                xc0[0] = (KK) + 1; const bfloat128 xa1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); \
                xc1[0] = (KK) + 1; const bfloat128 xb1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); \
                xc2[0] = (KK) + 1; const bfloat128 xc1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                ba0 = v_bf16_mac_acc32_b(ph_.v1, xa1_, ba0, 0); bb0 = v_bf16_mac_acc32_b(ph_.v2, xa1_, bb0, 0); \
                ba1 = v_bf16_mac_acc32_b(ph_.v1, xb1_, ba1, 0); bb1 = v_bf16_mac_acc32_b(ph_.v2, xb1_, bb1, 0); \
                ba2 = v_bf16_mac_acc32_b(ph_.v1, xc1_, ba2, 0); bb2 = v_bf16_mac_acc32_b(ph_.v2, xc1_, bb2, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0}, ba2 = {0}, bb2 = {0};
                NVD3B(w0, k + 0, r + 2, 1);
                NVD3B(w1, k + 2, r + 3, 1);
#pragma loop_unroll(1)
                for (int sb = 1; sb < 4; ++sb) {
                    const int rs = r + (sb << 1), ks = k + (sb << 2);
                    NVD3B(w0, ks + 0, rs + 2, 1);
                    NVD3B(w1, ks + 2, rs + 3, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 oa2, ob2;
                oa2.v1 = acc_buf[2][0]; oa2.v2 = acc_buf[2][1];
                ob2.v1 = acc_buf[2][2]; ob2.v2 = acc_buf[2][3];
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
                oa2 = v_f32_mac_x2_b(ba2, sc.v1, sc.v3, oa2, 0); ob2 = v_f32_mac_x2_b(bb2, sc.v2, sc.v4, ob2, 0);
                acc_buf[2][0] = oa2.v1; acc_buf[2][1] = oa2.v2;
                acc_buf[2][2] = ob2.v1; acc_buf[2][3] = ob2.v2;
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0}, ba2 = {0}, bb2 = {0};
                NVD3B(w0, k + 0,  r + 2, 1); NVD3B(w1, k + 2,  r + 3, 1);
                NVD3B(w0, k + 4,  r + 4, 1); NVD3B(w1, k + 6,  r + 5, 1);
                NVD3B(w0, k + 8,  r + 6, 1); NVD3B(w1, k + 10, r + 7, 1);
                NVD3B(w0, k + 12, 0,     0); NVD3B(w1, k + 14, 0,     0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
                REDUCE_ACC_VLM(2, ba2, bb2, sc);
            }
#undef NVD3B
            STORE_ACC1(0, a0, b0); STORE_ACC1(1, a1, b1);
        } else {
            int5 xc0 = {0, s0, t0, 0, 0};
            int5 xc1 = {0, s1, t1, 0, 0};
            int5 xc2 = {0, s2, t2, 0, 0};
            int5 xc3 = {0, s3, t3, 0, 0};
            ZERO_ACC_VLM(0); ZERO_ACC_VLM(1); ZERO_ACC_VLM(2); ZERO_ACC_VLM(3);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define NVD4V(WW, KK, R_NEXT, DO_W) do { \
                uchar256 lo_, hi_; \
                NVF4_UNPACK_ASH((WW), lo_, hi_); \
                if (DO_W) { wc[1] = (R_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                xc0[0] = (KK); const bfloat128 xa0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); \
                xc1[0] = (KK); const bfloat128 xb0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); \
                xc2[0] = (KK); const bfloat128 xc0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act)); \
                xc3[0] = (KK); const bfloat128 xd0_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, act)); \
                const bfloat128_pair_t pl_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo_)); \
                ba0 = v_bf16_mac_acc32_b(pl_.v1, xa0_, ba0, 0); bb0 = v_bf16_mac_acc32_b(pl_.v2, xa0_, bb0, 0); \
                ba1 = v_bf16_mac_acc32_b(pl_.v1, xb0_, ba1, 0); bb1 = v_bf16_mac_acc32_b(pl_.v2, xb0_, bb1, 0); \
                ba2 = v_bf16_mac_acc32_b(pl_.v1, xc0_, ba2, 0); bb2 = v_bf16_mac_acc32_b(pl_.v2, xc0_, bb2, 0); \
                ba3 = v_bf16_mac_acc32_b(pl_.v1, xd0_, ba3, 0); bb3 = v_bf16_mac_acc32_b(pl_.v2, xd0_, bb3, 0); \
                xc0[0] = (KK) + 1; const bfloat128 xa1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); \
                xc1[0] = (KK) + 1; const bfloat128 xb1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); \
                xc2[0] = (KK) + 1; const bfloat128 xc1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act)); \
                xc3[0] = (KK) + 1; const bfloat128 xd1_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, act)); \
                const bfloat128_pair_t ph_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi_)); \
                ba0 = v_bf16_mac_acc32_b(ph_.v1, xa1_, ba0, 0); bb0 = v_bf16_mac_acc32_b(ph_.v2, xa1_, bb0, 0); \
                ba1 = v_bf16_mac_acc32_b(ph_.v1, xb1_, ba1, 0); bb1 = v_bf16_mac_acc32_b(ph_.v2, xb1_, bb1, 0); \
                ba2 = v_bf16_mac_acc32_b(ph_.v1, xc1_, ba2, 0); bb2 = v_bf16_mac_acc32_b(ph_.v2, xc1_, bb2, 0); \
                ba3 = v_bf16_mac_acc32_b(ph_.v1, xd1_, ba3, 0); bb3 = v_bf16_mac_acc32_b(ph_.v2, xd1_, bb3, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int r = kb << 3, k = kb << 4;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                float128 ba2 = {0}, bb2 = {0}, ba3 = {0}, bb3 = {0};
                NVD4V(w0, k + 0, r + 2, 1);
                NVD4V(w1, k + 2, r + 3, 1);
#pragma loop_unroll(1)
                for (int sb2 = 1; sb2 < 4; ++sb2) {
                    const int rs = r + (sb2 << 1), ks = k + (sb2 << 2);
                    NVD4V(w0, ks + 0, rs + 2, 1); NVD4V(w1, ks + 2, rs + 3, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 oa0, ob0, oa1, ob1;
                oa0.v1 = acc_buf[0][0]; oa0.v2 = acc_buf[0][1]; ob0.v1 = acc_buf[0][2]; ob0.v2 = acc_buf[0][3];
                oa1.v1 = acc_buf[1][0]; oa1.v2 = acc_buf[1][1]; ob1.v1 = acc_buf[1][2]; ob1.v2 = acc_buf[1][3];
                oa0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, oa0, 0); ob0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, ob0, 0);
                oa1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, oa1, 0); ob1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, ob1, 0);
                float128 oa2, ob2, oa3, ob3;
                oa2.v1 = acc_buf[2][0]; oa2.v2 = acc_buf[2][1]; ob2.v1 = acc_buf[2][2]; ob2.v2 = acc_buf[2][3];
                oa3.v1 = acc_buf[3][0]; oa3.v2 = acc_buf[3][1]; ob3.v1 = acc_buf[3][2]; ob3.v2 = acc_buf[3][3];
                acc_buf[0][0] = oa0.v1; acc_buf[0][1] = oa0.v2; acc_buf[0][2] = ob0.v1; acc_buf[0][3] = ob0.v2;
                acc_buf[1][0] = oa1.v1; acc_buf[1][1] = oa1.v2; acc_buf[1][2] = ob1.v1; acc_buf[1][3] = ob1.v2;
                oa2 = v_f32_mac_x2_b(ba2, sc.v1, sc.v3, oa2, 0); ob2 = v_f32_mac_x2_b(bb2, sc.v2, sc.v4, ob2, 0);
                oa3 = v_f32_mac_x2_b(ba3, sc.v1, sc.v3, oa3, 0); ob3 = v_f32_mac_x2_b(bb3, sc.v2, sc.v4, ob3, 0);
                acc_buf[2][0] = oa2.v1; acc_buf[2][1] = oa2.v2; acc_buf[2][2] = ob2.v1; acc_buf[2][3] = ob2.v2;
                acc_buf[3][0] = oa3.v1; acc_buf[3][1] = oa3.v2; acc_buf[3][2] = ob3.v1; acc_buf[3][3] = ob3.v2;
            }
            {
                const int kb = NB - 1, r = kb << 3, k = kb << 4;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                float128 ba2 = {0}, bb2 = {0}, ba3 = {0}, bb3 = {0};
                NVD4V(w0, k + 0,  r + 2, 1); NVD4V(w1, k + 2,  r + 3, 1);
                NVD4V(w0, k + 4,  r + 4, 1); NVD4V(w1, k + 6,  r + 5, 1);
                NVD4V(w0, k + 8,  r + 6, 1); NVD4V(w1, k + 10, r + 7, 1);
                NVD4V(w0, k + 12, 0,     0); NVD4V(w1, k + 14, 0,     0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                REDUCE_ACC_VLM(0, ba0, bb0, sc); REDUCE_ACC_VLM(1, ba1, bb1, sc);
                REDUCE_ACC_VLM(2, ba2, bb2, sc); REDUCE_ACC_VLM(3, ba3, bb3, sc);
            }
#undef NVD4V
        }
#elif defined(MXF8)
        int5 bc = {n0, 0, e, 0, 0};
        if (nm == 1) {
            float128 a0 = {0}, b0 = {0};
            int5 xc0 = {0, s0, t0, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 x0 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc0[0] = 1; bfloat128 x1 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc0[0] = 2; bfloat128 x2 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc0[0] = 3; bfloat128 x3 = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
#define MXD1(WW, XX, K_NEXT, DO_W, DO_X) do { \
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
                if (DO_X) { xc0[0] = (K_NEXT); (XX) = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); } \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 ba = {0}, bb = {0};
                MXD1(w0, x0, k0 + 4, 1, 1);  MXD1(w1, x1, k0 + 5, 1, 1);
                MXD1(w2, x2, k0 + 6, 1, 1);  MXD1(w3, x3, k0 + 7, 1, 1);
                MXD1(w0, x0, k0 + 8, 1, 1);  MXD1(w1, x1, k0 + 9, 1, 1);
                MXD1(w2, x2, k0 + 10, 1, 1); MXD1(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 3; ++sub) {
                    const int ks = k0 + (sub << 3);
                    MXD1(w0, x0, ks + 4, 1, 1);  MXD1(w1, x1, ks + 5, 1, 1);
                    MXD1(w2, x2, ks + 6, 1, 1);  MXD1(w3, x3, ks + 7, 1, 1);
                    MXD1(w0, x0, ks + 8, 1, 1);  MXD1(w1, x1, ks + 9, 1, 1);
                    MXD1(w2, x2, ks + 10, 1, 1); MXD1(w3, x3, ks + 11, 1, 1);
                }
                MXD1(w0, x0, k0 + 28, 1, 1); MXD1(w1, x1, k0 + 29, 1, 1);
                MXD1(w2, x2, k0 + 30, 1, 1); MXD1(w3, x3, k0 + 31, 1, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                MXD1(w0, x0, k0 + 32, 1, 1); MXD1(w1, x1, k0 + 33, 1, 1);
                MXD1(w2, x2, k0 + 34, 1, 1); MXD1(w3, x3, k0 + 35, 1, 1);
                a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
                b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 ba = {0}, bb = {0};
                MXD1(w0, x0, k0 + 4, 1, 1);  MXD1(w1, x1, k0 + 5, 1, 1);
                MXD1(w2, x2, k0 + 6, 1, 1);  MXD1(w3, x3, k0 + 7, 1, 1);
                MXD1(w0, x0, k0 + 8, 1, 1);  MXD1(w1, x1, k0 + 9, 1, 1);
                MXD1(w2, x2, k0 + 10, 1, 1); MXD1(w3, x3, k0 + 11, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 3; ++sub) {
                    const int ks = k0 + (sub << 3);
                    MXD1(w0, x0, ks + 4, 1, 1);  MXD1(w1, x1, ks + 5, 1, 1);
                    MXD1(w2, x2, ks + 6, 1, 1);  MXD1(w3, x3, ks + 7, 1, 1);
                    MXD1(w0, x0, ks + 8, 1, 1);  MXD1(w1, x1, ks + 9, 1, 1);
                    MXD1(w2, x2, ks + 10, 1, 1); MXD1(w3, x3, ks + 11, 1, 1);
                }
                MXD1(w0, x0, k0 + 28, 1, 1); MXD1(w1, x1, k0 + 29, 1, 1);
                MXD1(w2, x2, k0 + 30, 1, 1); MXD1(w3, x3, k0 + 31, 1, 1);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                MXD1(w0, x0, 0, 0, 0); MXD1(w1, x1, 0, 0, 0);
                MXD1(w2, x2, 0, 0, 0); MXD1(w3, x3, 0, 0, 0);
                a0 = v_f32_mac_x2_b(ba, sc.v1, sc.v3, a0, 0);
                b0 = v_f32_mac_x2_b(bb, sc.v2, sc.v4, b0, 0);
            }
#undef MXD1
            STORE_ACC1(0, a0, b0);
        } else if (nm == 2) {
            float128 a0 = {0}, b0 = {0}, a1 = {0}, b1 = {0};
            int5 xc0 = {0, s0, t0, 0, 0};
            int5 xc1 = {0, s1, t1, 0, 0};
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 2; uchar256 w2 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 3; uchar256 w3 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
#define MXD2(WW, KK, K_NEXT, DO_W) do { \
                const uchar256 e_ = (WW) & (uchar256)0x78; \
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
                const uchar256 wa_ = (WW) - d8_; \
                if (DO_W) { wc[1] = (K_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const uchar256 d80_ = d8_ << 4; \
                xc0[0] = (KK); const bfloat128 xa_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); \
                xc1[0] = (KK); const bfloat128 xb_ = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); \
                const bfloat128_pair_t p_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
                const ushort128_pair_t d_ = v_convert_u8_to_u16_all_b(d80_); \
                const ushort128 r1_ = *((ushort128*)&p_.v1) + d_.v1; \
                const ushort128 r2_ = *((ushort128*)&p_.v2) + d_.v2; \
                ba0 = v_bf16_mac_acc32_b(*((bfloat128*)&r1_), xa_, ba0, 0); \
                bb0 = v_bf16_mac_acc32_b(*((bfloat128*)&r2_), xa_, bb0, 0); \
                ba1 = v_bf16_mac_acc32_b(*((bfloat128*)&r1_), xb_, ba1, 0); \
                bb1 = v_bf16_mac_acc32_b(*((bfloat128*)&r2_), xb_, bb1, 0); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                MXD2(w0, k0 + 0, k0 + 4, 1); MXD2(w1, k0 + 1, k0 + 5, 1);
                MXD2(w2, k0 + 2, k0 + 6, 1); MXD2(w3, k0 + 3, k0 + 7, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 8; ++sub) {
                    const int ks = k0 + (sub << 2);
                    MXD2(w0, ks + 0, ks + 4, 1); MXD2(w1, ks + 1, ks + 5, 1);
                    MXD2(w2, ks + 2, ks + 6, 1); MXD2(w3, ks + 3, ks + 7, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                MXD2(w0, k0 + 0, k0 + 4, 1); MXD2(w1, k0 + 1, k0 + 5, 1);
                MXD2(w2, k0 + 2, k0 + 6, 1); MXD2(w3, k0 + 3, k0 + 7, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 7; ++sub) {
                    const int ks = k0 + (sub << 2);
                    MXD2(w0, ks + 0, ks + 4, 1); MXD2(w1, ks + 1, ks + 5, 1);
                    MXD2(w2, ks + 2, ks + 6, 1); MXD2(w3, ks + 3, ks + 7, 1);
                }
                MXD2(w0, k0 + 28, 0, 0); MXD2(w1, k0 + 29, 0, 0);
                MXD2(w2, k0 + 30, 0, 0); MXD2(w3, k0 + 31, 0, 0);
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
            }
#undef MXD2
            STORE_ACC1(0, a0, b0); STORE_ACC1(1, a1, b1);
        } else if (nm == 3) {
            int5 xc0 = {0, s0, t0, 0, 0};
            int5 xc1 = {0, s1, t1, 0, 0};
            int5 xc2 = {0, s2, t2, 0, 0};
            float128 a0 = {0}, b0 = {0}, a1 = {0}, b1 = {0};
            ZERO_ACC_VLM(2);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc1[0] = 0; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act));
            xc2[0] = 0; bfloat128 xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act));
            bfloat128 p1, p2;
            {
                const uchar256 e_ = w0 & (uchar256)0x78;
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT);
                const uchar256 wa_ = w0 - d8_;
                wc[1] = 2; w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
                const uchar256 d80_ = d8_ << 4;
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_));
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_);
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1;
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2;
                p1 = *((bfloat128*)&r1_); p2 = *((bfloat128*)&r2_);
            }
#define MXD3_SWP(P1, P2, WW, Q1, Q2, KX_NEXT, KW_NEXT, DO_W, DO_X) do { \
                const uchar256 e_ = (WW) & (uchar256)0x78; \
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
                const uchar256 wa_ = (WW) - d8_; \
                if (DO_W) { wc[1] = (KW_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const uchar256 d80_ = d8_ << 4; \
                ba0 = v_bf16_mac_acc32_b((P1), xa, ba0, 0); bb0 = v_bf16_mac_acc32_b((P2), xa, bb0, 0); \
                if (DO_X) { xc0[0] = (KX_NEXT); xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); } \
                ba1 = v_bf16_mac_acc32_b((P1), xb, ba1, 0); bb1 = v_bf16_mac_acc32_b((P2), xb, bb1, 0); \
                if (DO_X) { xc1[0] = (KX_NEXT); xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); } \
                ba2 = v_bf16_mac_acc32_b((P1), xc, ba2, 0); bb2 = v_bf16_mac_acc32_b((P2), xc, bb2, 0); \
                if (DO_X) { xc2[0] = (KX_NEXT); xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act)); } \
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_); \
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1; \
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2; \
                (Q1) = *((bfloat128*)&r1_); (Q2) = *((bfloat128*)&r2_); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0}, ba2 = {0}, bb2 = {0};
                bfloat128 q1, q2;
                MXD3_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXD3_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 16; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXD3_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXD3_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 oa2, ob2;
                oa2.v1 = acc_buf[2][0]; oa2.v2 = acc_buf[2][1];
                ob2.v1 = acc_buf[2][2]; ob2.v2 = acc_buf[2][3];
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
                oa2 = v_f32_mac_x2_b(ba2, sc.v1, sc.v3, oa2, 0); ob2 = v_f32_mac_x2_b(bb2, sc.v2, sc.v4, ob2, 0);
                acc_buf[2][0] = oa2.v1; acc_buf[2][1] = oa2.v2;
                acc_buf[2][2] = ob2.v1; acc_buf[2][3] = ob2.v2;
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0}, ba2 = {0}, bb2 = {0};
                bfloat128 q1, q2;
                MXD3_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXD3_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 14; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXD3_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXD3_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                MXD3_SWP(p1, p2, w1, q1, q2, k0 + 29, k0 + 31, 1, 1);
                MXD3_SWP(q1, q2, w0, p1, p2, k0 + 30, 0,       0, 1);
                MXD3_SWP(p1, p2, w1, q1, q2, k0 + 31, 0,       0, 1);
                {
                    ba0 = v_bf16_mac_acc32_b(q1, xa, ba0, 0); bb0 = v_bf16_mac_acc32_b(q2, xa, bb0, 0);
                    ba1 = v_bf16_mac_acc32_b(q1, xb, ba1, 0); bb1 = v_bf16_mac_acc32_b(q2, xb, bb1, 0);
                    ba2 = v_bf16_mac_acc32_b(q1, xc, ba2, 0); bb2 = v_bf16_mac_acc32_b(q2, xc, bb2, 0);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                a0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, a0, 0); b0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, b0, 0);
                a1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, a1, 0); b1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, b1, 0);
                REDUCE_ACC_VLM(2, ba2, bb2, sc);
            }
#undef MXD3_SWP
            STORE_ACC1(0, a0, b0); STORE_ACC1(1, a1, b1);
        } else {
            int5 xc0 = {0, s0, t0, 0, 0};
            int5 xc1 = {0, s1, t1, 0, 0};
            int5 xc2 = {0, s2, t2, 0, 0};
            int5 xc3 = {0, s3, t3, 0, 0};
            ZERO_ACC_VLM(0); ZERO_ACC_VLM(1); ZERO_ACC_VLM(2); ZERO_ACC_VLM(3);
            wc[1] = 0; uchar256 w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            wc[1] = 1; uchar256 w1 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
            bc[1] = 0; minifloat256 bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
            xc0[0] = 0; bfloat128 xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act));
            xc1[0] = 0; bfloat128 xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act));
            xc2[0] = 0; bfloat128 xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act));
            xc3[0] = 0; bfloat128 xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, act));
            bfloat128 p1, p2;
            {
                const uchar256 e_ = w0 & (uchar256)0x78;
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT);
                const uchar256 wa_ = w0 - d8_;
                wc[1] = 2; w0 = v_u8_ld_tnsr_b(wc, w, SW_L0CS);
                const uchar256 d80_ = d8_ << 4;
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_));
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_);
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1;
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2;
                p1 = *((bfloat128*)&r1_); p2 = *((bfloat128*)&r2_);
            }
#define MXD4_SWP(P1, P2, WW, Q1, Q2, KX_NEXT, KW_NEXT, DO_W, DO_X) do { \
                const uchar256 e_ = (WW) & (uchar256)0x78; \
                const uchar256 d8_ = v_u8_sub_b(e_, (unsigned char)0x70, SW_SAT); \
                const uchar256 wa_ = (WW) - d8_; \
                if (DO_W) { wc[1] = (KW_NEXT); (WW) = v_u8_ld_tnsr_b(wc, w, SW_L0CS); } \
                const uchar256 d80_ = d8_ << 4; \
                ba0 = v_bf16_mac_acc32_b((P1), xa, ba0, 0); bb0 = v_bf16_mac_acc32_b((P2), xa, bb0, 0); \
                if (DO_X) { xc0[0] = (KX_NEXT); xa = v_bf16_ld_g((__global__ bf16*)gen_addr(xc0, act)); } \
                ba1 = v_bf16_mac_acc32_b((P1), xb, ba1, 0); bb1 = v_bf16_mac_acc32_b((P2), xb, bb1, 0); \
                if (DO_X) { xc1[0] = (KX_NEXT); xb = v_bf16_ld_g((__global__ bf16*)gen_addr(xc1, act)); } \
                ba2 = v_bf16_mac_acc32_b((P1), xc, ba2, 0); bb2 = v_bf16_mac_acc32_b((P2), xc, bb2, 0); \
                if (DO_X) { xc2[0] = (KX_NEXT); xc = v_bf16_ld_g((__global__ bf16*)gen_addr(xc2, act)); } \
                ba3 = v_bf16_mac_acc32_b((P1), xd, ba3, 0); bb3 = v_bf16_mac_acc32_b((P2), xd, bb3, 0); \
                if (DO_X) { xc3[0] = (KX_NEXT); xd = v_bf16_ld_g((__global__ bf16*)gen_addr(xc3, act)); } \
                const bfloat128_pair_t pv_ = v_convert_f8_to_bf16_all_b(*((minifloat256*)&wa_)); \
                const ushort128_pair_t dv_ = v_convert_u8_to_u16_all_b(d80_); \
                const ushort128 r1_ = *((ushort128*)&pv_.v1) + dv_.v1; \
                const ushort128 r2_ = *((ushort128*)&pv_.v2) + dv_.v2; \
                (Q1) = *((bfloat128*)&r1_); (Q2) = *((bfloat128*)&r2_); \
            } while (0)
#pragma loop_unroll(1)
            for (int kb = 0; kb < NB - 1; ++kb) {
                const int k0 = kb << 5;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                float128 ba2 = {0}, bb2 = {0}, ba3 = {0}, bb3 = {0};
                bfloat128 q1, q2;
                MXD4_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXD4_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 16; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXD4_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXD4_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                bc[1] = kb + 1; bs_cur = v_f8_ld_tnsr_b(bc, bs, SW_L0CS);
                float128 oa0, ob0, oa1, ob1;
                oa0.v1 = acc_buf[0][0]; oa0.v2 = acc_buf[0][1]; ob0.v1 = acc_buf[0][2]; ob0.v2 = acc_buf[0][3];
                oa1.v1 = acc_buf[1][0]; oa1.v2 = acc_buf[1][1]; ob1.v1 = acc_buf[1][2]; ob1.v2 = acc_buf[1][3];
                oa0 = v_f32_mac_x2_b(ba0, sc.v1, sc.v3, oa0, 0); ob0 = v_f32_mac_x2_b(bb0, sc.v2, sc.v4, ob0, 0);
                oa1 = v_f32_mac_x2_b(ba1, sc.v1, sc.v3, oa1, 0); ob1 = v_f32_mac_x2_b(bb1, sc.v2, sc.v4, ob1, 0);
                float128 oa2, ob2, oa3, ob3;
                oa2.v1 = acc_buf[2][0]; oa2.v2 = acc_buf[2][1]; ob2.v1 = acc_buf[2][2]; ob2.v2 = acc_buf[2][3];
                oa3.v1 = acc_buf[3][0]; oa3.v2 = acc_buf[3][1]; ob3.v1 = acc_buf[3][2]; ob3.v2 = acc_buf[3][3];
                acc_buf[0][0] = oa0.v1; acc_buf[0][1] = oa0.v2; acc_buf[0][2] = ob0.v1; acc_buf[0][3] = ob0.v2;
                acc_buf[1][0] = oa1.v1; acc_buf[1][1] = oa1.v2; acc_buf[1][2] = ob1.v1; acc_buf[1][3] = ob1.v2;
                oa2 = v_f32_mac_x2_b(ba2, sc.v1, sc.v3, oa2, 0); ob2 = v_f32_mac_x2_b(bb2, sc.v2, sc.v4, ob2, 0);
                oa3 = v_f32_mac_x2_b(ba3, sc.v1, sc.v3, oa3, 0); ob3 = v_f32_mac_x2_b(bb3, sc.v2, sc.v4, ob3, 0);
                acc_buf[2][0] = oa2.v1; acc_buf[2][1] = oa2.v2; acc_buf[2][2] = ob2.v1; acc_buf[2][3] = ob2.v2;
                acc_buf[3][0] = oa3.v1; acc_buf[3][1] = oa3.v2; acc_buf[3][2] = ob3.v1; acc_buf[3][3] = ob3.v2;
            }
            {
                const int k0 = (NB - 1) << 5;
                float128 ba0 = {0}, bb0 = {0}, ba1 = {0}, bb1 = {0};
                float128 ba2 = {0}, bb2 = {0}, ba3 = {0}, bb3 = {0};
                bfloat128 q1, q2;
                MXD4_SWP(p1, p2, w1, q1, q2, k0 + 1, k0 + 3, 1, 1);
                MXD4_SWP(q1, q2, w0, p1, p2, k0 + 2, k0 + 4, 1, 1);
#pragma loop_unroll(1)
                for (int sub = 1; sub < 14; ++sub) {
                    const int ks = k0 + (sub << 1);
                    MXD4_SWP(p1, p2, w1, q1, q2, ks + 1, ks + 3, 1, 1);
                    MXD4_SWP(q1, q2, w0, p1, p2, ks + 2, ks + 4, 1, 1);
                }
                MXD4_SWP(p1, p2, w1, q1, q2, k0 + 29, k0 + 31, 1, 1);
                MXD4_SWP(q1, q2, w0, p1, p2, k0 + 30, 0,       0, 1);
                MXD4_SWP(p1, p2, w1, q1, q2, k0 + 31, 0,       0, 1);
                {
                    ba0 = v_bf16_mac_acc32_b(q1, xa, ba0, 0); bb0 = v_bf16_mac_acc32_b(q2, xa, bb0, 0);
                    ba1 = v_bf16_mac_acc32_b(q1, xb, ba1, 0); bb1 = v_bf16_mac_acc32_b(q2, xb, bb1, 0);
                    ba2 = v_bf16_mac_acc32_b(q1, xc, ba2, 0); bb2 = v_bf16_mac_acc32_b(q2, xc, bb2, 0);
                    ba3 = v_bf16_mac_acc32_b(q1, xd, ba3, 0); bb3 = v_bf16_mac_acc32_b(q2, xd, bb3, 0);
                }
                const float256 sc = v_convert_f8_to_f32_all_b(bs_cur);
                REDUCE_ACC_VLM(0, ba0, bb0, sc); REDUCE_ACC_VLM(1, ba1, bb1, sc);
                REDUCE_ACC_VLM(2, ba2, bb2, sc); REDUCE_ACC_VLM(3, ba3, bb3, sc);
            }
#undef MXD4_SWP
        }
#endif

        int5 sc = {n0, e, 0, 0, 0};
        const float64 si0 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 64;
        const float64 si1 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 128;
        const float64 si2 = v_f32_ld_tnsr_b(sc, si); sc[0] = n0 + 192;
        const float64 si3 = v_f32_ld_tnsr_b(sc, si);
        const int toks[4] = {t0, t1, t2, t3};
        const int slots[4] = {s0, s1, s2, s3};
#pragma loop_unroll(1)
        for (int m = 0; m < nm; ++m) {
            float64_pair_t ra, rb;
            ra.v1 = acc_buf[m][0] * si0; ra.v2 = acc_buf[m][1] * si1;
            rb.v1 = acc_buf[m][2] * si2; rb.v2 = acc_buf[m][3] * si3;
            int5 oc = {n0, slots[m], toks[m], 0, 0};
            v_bf16_st_tnsr(oc, ys, v_convert_f32_to_bf16_all_b(ra));
            oc[0] = n0 + 128;
            v_bf16_st_tnsr(oc, ys, v_convert_f32_to_bf16_all_b(rb));
        }
    }
}
