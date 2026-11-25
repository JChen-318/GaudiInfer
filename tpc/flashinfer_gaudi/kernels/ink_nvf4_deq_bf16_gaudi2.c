// SPDX-License-Identifier: Apache-2.0
// Exact bulk NVFP4 / MXFP8 -> BF16 expert weight dequantization (table-free, Direct-engine safe, 0 spills).
// One source builds both ink_nvf4_deq_bf16_gaudi2 (default) and ink_mxfp8_deq_bf16_gaudi2 (-DMXFP8).
// Both handle w13 and w2 in natural k order with the same output ABI.
//
// Inputs (TPC dim 0 fastest; pack/host shapes are reversed [E, ...]):
//   NVFP4 (default):
//     w        u8   [cols, K/2, E]   two E2M1 nibbles per byte along K (low nibble = even k, high nibble = odd k)
//     bs       f8   [cols, K/16, E]  Gaudi2 fp8_143 block scale = scale_ckpt / 8 (= E4M3FN code - 0x18)
//     out      bf16 [cols, K, E]     exact value = E2M1(nibble) * (scale_ckpt / 8) * 2^-6 (factor 512 folded in g[e])
//   MXFP8 (#ifdef MXFP8):
//     w        u8   [cols, K, E]     checkpoint E4M3FN codes (1 byte/weight, natural k; w13 checkpoint order,
//                                    w2 256-col interleaved)
//     bs       f8   [cols, K/32, E]  Gaudi2 fp8_143 block scale = 2^(s - s_max) in [2^-6, 1.0] (same col order)
//     out      bf16 [cols, K, E]     exact value = E4M3FN(code) * bs (factor g[e] = 2^(s_max - 127) applied in F32).
//                                    E4M3FN exponent field 15 (0x78..0x7E / 0xF8..0xFE, |v| 256..448) is converted
//                                    as fp8_143(code ^ 0x08) * (2 * bs) using u16 lane-aligned predicates matching
//                                    v_convert_f8_to_bf16_all_b (even byte -> v1, odd byte -> v2); all other codes
//                                    (normals, subnormals, +-0.0) convert directly. Exact for all 254 finite E4M3FN
//                                    codes (NaN codes 0x7F/0xFF are rejected by the converter and map to +-480 * bs)
//                                    and all finite Gaudi2 fp8_143 scales.
//   mask       i32  [E]              optional per-expert mask; when use_mask != 0, experts with mask[e] == 0 are skipped
//
// Output column order per 256-col chunk is [even-byte 128 | odd-byte 128], matching v_convert_f8_to_bf16_all_b with
// zero shuffle:
//   - w13: per 256 cols [gate 128 | up 128]
//   - w2:  natural columns [0..cols-1] (undoes the pack's 256-col interleave)
//
// Scalar parameter (SRF 0):
//   use_mask i32                   0: dequantize all experts (mask is not read); nonzero: write only mask[e] != 0
//
// Index space: [(cols / 256) * E], split along dim 0 across 24 TPCs.

void main(tensor w, tensor bs, tensor mask, tensor out, int use_mask)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int KB = get_dim_size(bs, 1);
    const int E = get_dim_size(w, 2);

    for (int pp = start[0]; pp < end[0]; ++pp) {
        const int g = pp / E;
        const int e = pp - g * E;
        if (use_mask) {
            int5 mc = {e, 0, 0, 0, 0};
            if (s_i32_ld_g((__global__ int*)gen_addr(mc, mask)) == 0) continue;
        }

        const int n0 = g << 8;
        int5 bc = {n0, 0, e, 0, 0};
        int5 wc = {n0, 0, e, 0, 0};
        int5 o0 = {n0, 0, e, 0, 0};
        int5 o1 = {n0 + 128, 0, e, 0, 0};

        for (int kb = 0; kb < KB; ++kb) {
            bc[1] = kb;
            const bfloat128_pair_t sp = v_convert_f8_to_bf16_all_b(v_f8_ld_tnsr_b(bc, bs));
            const bfloat128 s0 = sp.v1;
            const bfloat128 s1 = sp.v2;
#ifdef MXFP8
            const bfloat128 s0_2 = s0 + s0;
            const bfloat128 s1_2 = s1 + s1;
            const int kbase = kb << 5;
            for (int h = 0; h < 32; h += 4) {
                wc[1] = kbase + h;     const uchar256 b0 = v_u8_ld_tnsr_b(wc, w);
                wc[1] = kbase + h + 1; const uchar256 b1 = v_u8_ld_tnsr_b(wc, w);
                wc[1] = kbase + h + 2; const uchar256 b2 = v_u8_ld_tnsr_b(wc, w);
                wc[1] = kbase + h + 3; const uchar256 b3 = v_u8_ld_tnsr_b(wc, w);

#define MX_ROW(B, OFF) do { \
                    const ushort128 u = *((ushort128*)&(B)); \
                    const bool128 m0 = v_u16_cmp_eq_b(u & (unsigned short)0x0078, (unsigned short)0x0078); \
                    const bool128 m1 = v_u16_cmp_eq_b(u & (unsigned short)0x7800, (unsigned short)0x7800); \
                    const ushort128 a0 = v_u16_xor_vb(u, (unsigned short)0x0008, 0, u, m0, 0); \
                    const ushort128 a1 = v_u16_xor_vb(a0, (unsigned short)0x0800, 0, a0, m1, 0); \
                    const bfloat128_pair_t vp = v_convert_f8_to_bf16_all_b(*((minifloat256*)&a1)); \
                    const bfloat128 sc0 = v_bf16_mov_vb(s0_2, 0, s0, m0, 0); \
                    const bfloat128 sc1 = v_bf16_mov_vb(s1_2, 0, s1, m1, 0); \
                    o0[1] = kbase + h + (OFF); o1[1] = kbase + h + (OFF); \
                    v_bf16_st_tnsr(o0, out, vp.v1 * sc0); \
                    v_bf16_st_tnsr(o1, out, vp.v2 * sc1); \
                } while (0)
                MX_ROW(b0, 0);
                MX_ROW(b1, 1);
                MX_ROW(b2, 2);
                MX_ROW(b3, 3);
#undef MX_ROW
            }
#else
            const int rbase = kb << 3;
            for (int h = 0; h < 8; h += 2) {
                wc[1] = rbase + h;
                const uchar256 b0 = v_u8_ld_tnsr_b(wc, w);
                wc[1] = rbase + h + 1;
                const uchar256 b1 = v_u8_ld_tnsr_b(wc, w);

                // Place each nibble s e1 e0 m into Gaudi2 fp8_143 s 0 0 e1 e0 m 0 0 (= E2M1 * 2^-6).
                const char256 ws0 = *((char256*)&b0);
                const char256 ws1 = *((char256*)&b1);
                const char256 ah0 = v_i8_ash_b(ws0, (char)-2, 0);
                const char256 al0 = v_i8_ash_b(ws0 << 4, (char)-2, 0);
                const char256 ah1 = v_i8_ash_b(ws1, (char)-2, 0);
                const char256 al1 = v_i8_ash_b(ws1 << 4, (char)-2, 0);

                const uchar256 lo0 = *((uchar256*)&al0) & (uchar256)0x9c;
                const uchar256 hi0 = *((uchar256*)&ah0) & (uchar256)0x9c;
                const uchar256 lo1 = *((uchar256*)&al1) & (uchar256)0x9c;
                const uchar256 hi1 = *((uchar256*)&ah1) & (uchar256)0x9c;

                const bfloat128_pair_t pl0 = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo0));
                const bfloat128_pair_t ph0 = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi0));
                const bfloat128_pair_t pl1 = v_convert_f8_to_bf16_all_b(*((minifloat256*)&lo1));
                const bfloat128_pair_t ph1 = v_convert_f8_to_bf16_all_b(*((minifloat256*)&hi1));

                const int k0 = (rbase + h) << 1;
                o0[1] = k0;     o1[1] = k0;
                v_bf16_st_tnsr(o0, out, pl0.v1 * s0);
                v_bf16_st_tnsr(o1, out, pl0.v2 * s1);
                o0[1] = k0 + 1; o1[1] = k0 + 1;
                v_bf16_st_tnsr(o0, out, ph0.v1 * s0);
                v_bf16_st_tnsr(o1, out, ph0.v2 * s1);
                o0[1] = k0 + 2; o1[1] = k0 + 2;
                v_bf16_st_tnsr(o0, out, pl1.v1 * s0);
                v_bf16_st_tnsr(o1, out, pl1.v2 * s1);
                o0[1] = k0 + 3; o1[1] = k0 + 3;
                v_bf16_st_tnsr(o0, out, ph1.v1 * s0);
                v_bf16_st_tnsr(o1, out, ph1.v2 * s1);
            }
#endif
        }
    }
}
