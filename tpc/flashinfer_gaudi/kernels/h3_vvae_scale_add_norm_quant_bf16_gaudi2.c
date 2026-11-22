// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 fused RMSNorm + FP8 row quant variants (4-way unrolled + butterfly sum64/max64 + 2-Newton rsqrt):
//   default:        res_out = res + (branch + bias) * scale; n = rms_norm(res_out, norm_w); quant(n)  (W <= 4096)
//   VVAE_NORM_ONLY: n = rms_norm(x, norm_w); quant(n)                                                 (W <= 8192)
//   H3_QUANT_ONLY:  (codes, scales) = vLLM FP8 row quant(x)                                           (W <= 8192)

static inline float64 round_bf16(float64 value) {
    float64_pair_t pair = {value, value};
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pair)).v1;
}

static inline float64 positive_reciprocal(float64 value) {
    float64 est = as_float64((int64)0x7ef311c3 - as_int64(value));
    est = est * (2.0f - value * est);
    est = est * (2.0f - value * est);
    est = est * (2.0f - value * est);
    return v_f32_sel_eq_f32_b(value, as_float64((int64)0x7f800000), 0.0f, est);
}

static inline float64 fast_rsqrt_2n(float64 x) {
    float64 y = as_float64((int64)0x5f3759df - (as_int64(x) >> 1));
    const float64 h = 0.5f * x;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y;
}

static inline float64 row_sum_fast(float64 x, uchar256 c4, uchar256 c2, uchar256 c1) {
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0);
    x += v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0);
    x += v_f32_mov_group_b(x, 0xffffffff, 63, 0);
    x += v_f32_shuffle_b(x, c4, 0, x);
    x += v_f32_shuffle_b(x, c2, 0, x);
    x += v_f32_shuffle_b(x, c1, 0, x);
    return x;
}

static inline float64 row_max_fast(float64 x, uchar256 c4, uchar256 c2, uchar256 c1) {
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_max_b(x, v_f32_mov_dual_group_all_b(x, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    x = v_f32_max_b(x, v_f32_mov_group_b(x, 0xffffffff, 63, 0));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, c4, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, c2, 0, x));
    x = v_f32_max_b(x, v_f32_shuffle_b(x, c1, 0, x));
    return x;
}

static inline minifloat256 bf16_128_to_f8_lo128(bfloat128 x) {
    const minifloat256 sp = v_convert_bf16_to_f8_b(x, 0, SW_CLIP_FP, (minifloat256)0);
    minifloat256 pk = v_f8_pack_b(sp, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
    pk = v_f8_pack_b(sp, SW_GROUP_1 | SW_STRIDE_2, pk);
    return v_f8_mov_dual_group_pack_b(pk, SW_PACK21, (minifloat256)0);
}

#define NQ_QUANT_QUAD(TIDX) do {                                                                \
    const bfloat128 s0_ = x_vlm[(TIDX)]     * inv_b;                                            \
    const bfloat128 s1_ = x_vlm[(TIDX) + 1] * inv_b;                                            \
    const bfloat128 s2_ = x_vlm[(TIDX) + 2] * inv_b;                                            \
    const bfloat128 s3_ = x_vlm[(TIDX) + 3] * inv_b;                                            \
    const minifloat256 sp0_ = v_convert_bf16_to_f8_b(s0_, 0, SW_CLIP_FP, (minifloat256)0);      \
    const minifloat256 sp1_ = v_convert_bf16_to_f8_b(s1_, 0, SW_CLIP_FP, (minifloat256)0);      \
    const minifloat256 sp2_ = v_convert_bf16_to_f8_b(s2_, 0, SW_CLIP_FP, (minifloat256)0);      \
    const minifloat256 sp3_ = v_convert_bf16_to_f8_b(s3_, 0, SW_CLIP_FP, (minifloat256)0);      \
    minifloat256 p0_ = v_f8_pack_b(sp0_, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);            \
    minifloat256 p1_ = v_f8_pack_b(sp1_, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);            \
    minifloat256 p2_ = v_f8_pack_b(sp2_, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);            \
    minifloat256 p3_ = v_f8_pack_b(sp3_, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);            \
    p0_ = v_f8_pack_b(sp0_, SW_GROUP_1 | SW_STRIDE_2, p0_);                                     \
    p1_ = v_f8_pack_b(sp1_, SW_GROUP_1 | SW_STRIDE_2, p1_);                                     \
    p2_ = v_f8_pack_b(sp2_, SW_GROUP_1 | SW_STRIDE_2, p2_);                                     \
    p3_ = v_f8_pack_b(sp3_, SW_GROUP_1 | SW_STRIDE_2, p3_);                                     \
    p0_ = v_f8_mov_dual_group_pack_b(p0_, SW_PACK21, (minifloat256)0);                          \
    p1_ = v_f8_mov_dual_group_pack_b(p1_, SW_PACK21, (minifloat256)0);                          \
    p2_ = v_f8_mov_dual_group_pack_b(p2_, SW_PACK21, (minifloat256)0);                          \
    p3_ = v_f8_mov_dual_group_pack_b(p3_, SW_PACK21, (minifloat256)0);                          \
    p0_ = v_f8_mov_dual_group_b(p1_, 0xffffffff, 0, 2, MkWr(1, 1), p0_);                        \
    p2_ = v_f8_mov_dual_group_b(p3_, 0xffffffff, 0, 2, MkWr(1, 1), p2_);                        \
    p0_ = v_f8_mov_dual_group_b(p1_, 0xffffffff, 1, 3, MkWr(1, 1), p0_);                        \
    p2_ = v_f8_mov_dual_group_b(p3_, 0xffffffff, 1, 3, MkWr(1, 1), p2_);                        \
    v_f8_st_tnsr((int5){(TIDX) * 128, row, 0, 0, 0}, codes, p0_);                               \
    v_f8_st_tnsr((int5){((TIDX) + 2) * 128, row, 0, 0, 0}, codes, p2_);                         \
} while (0)

#if defined(H3_QUANT_ONLY)
void main(tensor x_in, tensor codes, tensor scales) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(x_in, 0);
    const int tiles = width / 128;
    const int bulk = tiles & ~3;
    const int pairs = tiles & ~1;
    const uint64 l8 = read_lane_id_4b_b() & 7;
    const uint64 c4u = ((l8 ^ 4) | 0x80) * 0x01010101;
    const uint64 c2u = ((l8 ^ 2) | 0x80) * 0x01010101;
    const uint64 c1u = ((l8 ^ 1) | 0x80) * 0x01010101;
    const uchar256 c4 = *((const uchar256*)&c4u), c2 = *((const uchar256*)&c2u), c1 = *((const uchar256*)&c1u);
    bfloat128 x_vlm[64];
    for (int row = start[0]; row < end[0]; ++row) {
        bfloat128 m0 = 0, m1 = 0, m2 = 0, m3 = 0;
        for (int t = 0; t < bulk; t += 4) {
            const bfloat128 v0 = v_bf16_ld_tnsr_b((int5){t * 128, row, 0, 0, 0}, x_in);
            const bfloat128 v1 = v_bf16_ld_tnsr_b((int5){(t + 1) * 128, row, 0, 0, 0}, x_in);
            const bfloat128 v2 = v_bf16_ld_tnsr_b((int5){(t + 2) * 128, row, 0, 0, 0}, x_in);
            const bfloat128 v3 = v_bf16_ld_tnsr_b((int5){(t + 3) * 128, row, 0, 0, 0}, x_in);
            x_vlm[t] = v0; x_vlm[t + 1] = v1; x_vlm[t + 2] = v2; x_vlm[t + 3] = v3;
            m0 = v_bf16_max_b(m0, v_bf16_abs_b(v0));
            m1 = v_bf16_max_b(m1, v_bf16_abs_b(v1));
            m2 = v_bf16_max_b(m2, v_bf16_abs_b(v2));
            m3 = v_bf16_max_b(m3, v_bf16_abs_b(v3));
        }
        for (int t = bulk; t < tiles; ++t) {
            const bfloat128 v = v_bf16_ld_tnsr_b((int5){t * 128, row, 0, 0, 0}, x_in);
            x_vlm[t] = v;
            m0 = v_bf16_max_b(m0, v_bf16_abs_b(v));
        }
        const float64_pair_t mf = v_convert_bf16_to_f32_all_b(
            v_bf16_max_b(v_bf16_max_b(m0, m1), v_bf16_max_b(m2, m3)));
        const float64 maximum = row_max_fast(v_f32_max_b(mf.v1, mf.v2), c4, c2, c1);
        const float64 sc = round_bf16(round_bf16(maximum + 1.0e-8f) * (float)(bf16)(1.0f / 240.0f));
        const float64 inv_sc = round_bf16(positive_reciprocal(sc));
        const bfloat128 inv_b = v_convert_f32_to_bf16_all_b((float64_pair_t){inv_sc, inv_sc});
        v_f32_st_tnsr((int5){0, row, 0, 0, 0}, scales, sc);
        for (int t = 0; t < bulk; t += 4) NQ_QUANT_QUAD(t);
        for (int t = bulk; t < pairs; t += 2) {
            minifloat256 pk0 = bf16_128_to_f8_lo128(x_vlm[t] * inv_b);
            const minifloat256 pk1 = bf16_128_to_f8_lo128(x_vlm[t + 1] * inv_b);
            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 0, 2, MkWr(1, 1), pk0);
            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 1, 3, MkWr(1, 1), pk0);
            v_f8_st_tnsr((int5){t * 128, row, 0, 0, 0}, codes, pk0);
        }
        for (int t = pairs; t < tiles; ++t) {
            const minifloat256 pk = bf16_128_to_f8_lo128(x_vlm[t] * inv_b);
            v_f8_st_tnsr_partial((int5){t * 128, row, 0, 0, 0}, codes, pk, 127, 0);
        }
    }
}
#elif defined(VVAE_NORM_ONLY)
void main(tensor x_in, tensor norm_w, tensor codes, tensor scales,
          float epsilon, float inv_width, float inv_range) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(x_in, 0);
    const int tiles = width / 128;
    const int bulk = tiles & ~3;
    const int pairs = tiles & ~1;
    const uint64 l8 = read_lane_id_4b_b() & 7;
    const uint64 c4u = ((l8 ^ 4) | 0x80) * 0x01010101;
    const uint64 c2u = ((l8 ^ 2) | 0x80) * 0x01010101;
    const uint64 c1u = ((l8 ^ 1) | 0x80) * 0x01010101;
    const uchar256 c4 = *((const uchar256*)&c4u), c2 = *((const uchar256*)&c2u), c1 = *((const uchar256*)&c1u);
    bfloat128 w_vlm[64], x_vlm[64];
    for (int t = 0; t < tiles; ++t) w_vlm[t] = v_bf16_ld_tnsr_b((int5){t * 128, 0, 0, 0, 0}, norm_w);
    for (int row = start[0]; row < end[0]; ++row) {
        float128 sq0 = {0}, sq1 = {0}, sq2 = {0}, sq3 = {0};
        bfloat128 m0 = 0, m1 = 0, m2 = 0, m3 = 0;
        for (int t = 0; t < bulk; t += 4) {
            const bfloat128 y0 = v_bf16_ld_tnsr_b((int5){t * 128, row, 0, 0, 0}, x_in);
            const bfloat128 y1 = v_bf16_ld_tnsr_b((int5){(t + 1) * 128, row, 0, 0, 0}, x_in);
            const bfloat128 y2 = v_bf16_ld_tnsr_b((int5){(t + 2) * 128, row, 0, 0, 0}, x_in);
            const bfloat128 y3 = v_bf16_ld_tnsr_b((int5){(t + 3) * 128, row, 0, 0, 0}, x_in);
            const bfloat128 wy0 = y0 * w_vlm[t], wy1 = y1 * w_vlm[t + 1];
            const bfloat128 wy2 = y2 * w_vlm[t + 2], wy3 = y3 * w_vlm[t + 3];
            x_vlm[t] = wy0; x_vlm[t + 1] = wy1; x_vlm[t + 2] = wy2; x_vlm[t + 3] = wy3;
            sq0 = v_bf16_mac_acc32_b(y0, y0, sq0, (e_no_negation) << 1);
            sq1 = v_bf16_mac_acc32_b(y1, y1, sq1, (e_no_negation) << 1);
            sq2 = v_bf16_mac_acc32_b(y2, y2, sq2, (e_no_negation) << 1);
            sq3 = v_bf16_mac_acc32_b(y3, y3, sq3, (e_no_negation) << 1);
            m0 = v_bf16_max_b(m0, v_bf16_abs_b(wy0));
            m1 = v_bf16_max_b(m1, v_bf16_abs_b(wy1));
            m2 = v_bf16_max_b(m2, v_bf16_abs_b(wy2));
            m3 = v_bf16_max_b(m3, v_bf16_abs_b(wy3));
        }
        for (int t = bulk; t < tiles; ++t) {
            const bfloat128 y = v_bf16_ld_tnsr_b((int5){t * 128, row, 0, 0, 0}, x_in);
            const bfloat128 wy = y * w_vlm[t];
            x_vlm[t] = wy;
            sq0 = v_bf16_mac_acc32_b(y, y, sq0, (e_no_negation) << 1);
            m0 = v_bf16_max_b(m0, v_bf16_abs_b(wy));
        }
        const float64 s_lo = (sq0.v1 + sq1.v1) + (sq2.v1 + sq3.v1);
        const float64 s_hi = (sq0.v2 + sq1.v2) + (sq2.v2 + sq3.v2);
        const float64 rinv = fast_rsqrt_2n(row_sum_fast(s_lo + s_hi, c4, c2, c1) * inv_width + epsilon);
        const float64_pair_t mf = v_convert_bf16_to_f32_all_b(
            v_bf16_max_b(v_bf16_max_b(m0, m1), v_bf16_max_b(m2, m3)));
        const float64 maximum = round_bf16(row_max_fast(v_f32_max_b(mf.v1, mf.v2), c4, c2, c1) * rinv);
        const float64 sc = round_bf16(round_bf16(maximum + 1.0e-8f) * inv_range);
        const float64 inv_sc = round_bf16(positive_reciprocal(sc));
        const float64 fused_sc = rinv * inv_sc;
        const bfloat128 inv_b = v_convert_f32_to_bf16_all_b((float64_pair_t){fused_sc, fused_sc});
        v_f32_st_tnsr((int5){0, row, 0, 0, 0}, scales, sc);
        for (int t = 0; t < bulk; t += 4) NQ_QUANT_QUAD(t);
        for (int t = bulk; t < pairs; t += 2) {
            minifloat256 pk0 = bf16_128_to_f8_lo128(x_vlm[t] * inv_b);
            const minifloat256 pk1 = bf16_128_to_f8_lo128(x_vlm[t + 1] * inv_b);
            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 0, 2, MkWr(1, 1), pk0);
            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 1, 3, MkWr(1, 1), pk0);
            v_f8_st_tnsr((int5){t * 128, row, 0, 0, 0}, codes, pk0);
        }
        for (int t = pairs; t < tiles; ++t) {
            const minifloat256 pk = bf16_128_to_f8_lo128(x_vlm[t] * inv_b);
            v_f8_st_tnsr_partial((int5){t * 128, row, 0, 0, 0}, codes, pk, 127, 0);
        }
    }
}
#else
void main(tensor res, tensor branch, tensor bias, tensor scale, tensor norm_w,
          tensor codes, tensor scales, tensor res_out,
          float epsilon, float inv_width, float inv_range) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(res, 0);
    const int tiles = width / 128;
    const int bulk = tiles & ~3;
    const int pairs = tiles & ~1;
    const uint64 l8 = read_lane_id_4b_b() & 7;
    const uint64 c4u = ((l8 ^ 4) | 0x80) * 0x01010101;
    const uint64 c2u = ((l8 ^ 2) | 0x80) * 0x01010101;
    const uint64 c1u = ((l8 ^ 1) | 0x80) * 0x01010101;
    const uchar256 c4 = *((const uchar256*)&c4u), c2 = *((const uchar256*)&c2u), c1 = *((const uchar256*)&c1u);

    bfloat128 b_vlm[32], s_vlm[32], w_vlm[32], x_vlm[32];
    for (int t = 0; t < tiles; ++t) {
        int5 c = {t * 128, 0, 0, 0, 0};
        b_vlm[t] = v_bf16_ld_tnsr_b(c, bias);
        s_vlm[t] = v_bf16_ld_tnsr_b(c, scale);
        w_vlm[t] = v_bf16_ld_tnsr_b(c, norm_w);
    }

    for (int row = start[0]; row < end[0]; ++row) {
        float128 sq0 = {0}, sq1 = {0}, sq2 = {0}, sq3 = {0};
        bfloat128 m0 = 0, m1 = 0, m2 = 0, m3 = 0;
        for (int t = 0; t < bulk; t += 4) {
            int5 c0 = {t * 128, row, 0, 0, 0}, c1c = {(t + 1) * 128, row, 0, 0, 0};
            int5 c2c = {(t + 2) * 128, row, 0, 0, 0}, c3c = {(t + 3) * 128, row, 0, 0, 0};
            const bfloat128 y0 = v_bf16_mac_b(v_bf16_add_b(v_bf16_ld_tnsr_b(c0,  branch), b_vlm[t]),     s_vlm[t],     v_bf16_ld_tnsr_b(c0,  res));
            const bfloat128 y1 = v_bf16_mac_b(v_bf16_add_b(v_bf16_ld_tnsr_b(c1c, branch), b_vlm[t + 1]), s_vlm[t + 1], v_bf16_ld_tnsr_b(c1c, res));
            const bfloat128 y2 = v_bf16_mac_b(v_bf16_add_b(v_bf16_ld_tnsr_b(c2c, branch), b_vlm[t + 2]), s_vlm[t + 2], v_bf16_ld_tnsr_b(c2c, res));
            const bfloat128 y3 = v_bf16_mac_b(v_bf16_add_b(v_bf16_ld_tnsr_b(c3c, branch), b_vlm[t + 3]), s_vlm[t + 3], v_bf16_ld_tnsr_b(c3c, res));
            v_bf16_st_tnsr(c0, res_out, y0); v_bf16_st_tnsr(c1c, res_out, y1);
            v_bf16_st_tnsr(c2c, res_out, y2); v_bf16_st_tnsr(c3c, res_out, y3);
            const bfloat128 wy0 = y0 * w_vlm[t], wy1 = y1 * w_vlm[t + 1];
            const bfloat128 wy2 = y2 * w_vlm[t + 2], wy3 = y3 * w_vlm[t + 3];
            x_vlm[t] = wy0; x_vlm[t + 1] = wy1; x_vlm[t + 2] = wy2; x_vlm[t + 3] = wy3;
            sq0 = v_bf16_mac_acc32_b(y0, y0, sq0, (e_no_negation) << 1);
            sq1 = v_bf16_mac_acc32_b(y1, y1, sq1, (e_no_negation) << 1);
            sq2 = v_bf16_mac_acc32_b(y2, y2, sq2, (e_no_negation) << 1);
            sq3 = v_bf16_mac_acc32_b(y3, y3, sq3, (e_no_negation) << 1);
            m0 = v_bf16_max_b(m0, v_bf16_abs_b(wy0));
            m1 = v_bf16_max_b(m1, v_bf16_abs_b(wy1));
            m2 = v_bf16_max_b(m2, v_bf16_abs_b(wy2));
            m3 = v_bf16_max_b(m3, v_bf16_abs_b(wy3));
        }
        for (int t = bulk; t < tiles; ++t) {
            int5 c = {t * 128, row, 0, 0, 0};
            const bfloat128 y = v_bf16_mac_b(v_bf16_add_b(v_bf16_ld_tnsr_b(c, branch), b_vlm[t]), s_vlm[t], v_bf16_ld_tnsr_b(c, res));
            v_bf16_st_tnsr(c, res_out, y);
            const bfloat128 wy = y * w_vlm[t];
            x_vlm[t] = wy;
            sq0 = v_bf16_mac_acc32_b(y, y, sq0, (e_no_negation) << 1);
            m0 = v_bf16_max_b(m0, v_bf16_abs_b(wy));
        }
        const float64 s_lo = (sq0.v1 + sq1.v1) + (sq2.v1 + sq3.v1);
        const float64 s_hi = (sq0.v2 + sq1.v2) + (sq2.v2 + sq3.v2);
        const float64 rinv = fast_rsqrt_2n(row_sum_fast(s_lo + s_hi, c4, c2, c1) * inv_width + epsilon);
        const float64_pair_t mf = v_convert_bf16_to_f32_all_b(
            v_bf16_max_b(v_bf16_max_b(m0, m1), v_bf16_max_b(m2, m3)));
        const float64 maximum = round_bf16(row_max_fast(v_f32_max_b(mf.v1, mf.v2), c4, c2, c1) * rinv);
        const float64 sc = round_bf16(round_bf16(maximum + 1.0e-8f) * inv_range);
        const float64 inv_sc = round_bf16(positive_reciprocal(sc));
        const float64 fused_sc = rinv * inv_sc;
        const bfloat128 inv_b = v_convert_f32_to_bf16_all_b((float64_pair_t){fused_sc, fused_sc});
        v_f32_st_tnsr((int5){0, row, 0, 0, 0}, scales, sc);

        for (int t = 0; t < bulk; t += 4) NQ_QUANT_QUAD(t);
        for (int t = bulk; t < pairs; t += 2) {
            minifloat256 pk0 = bf16_128_to_f8_lo128(x_vlm[t] * inv_b);
            const minifloat256 pk1 = bf16_128_to_f8_lo128(x_vlm[t + 1] * inv_b);
            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 0, 2, MkWr(1, 1), pk0);
            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 1, 3, MkWr(1, 1), pk0);
            v_f8_st_tnsr((int5){t * 128, row, 0, 0, 0}, codes, pk0);
        }
        for (int t = pairs; t < tiles; ++t) {
            const minifloat256 pk = bf16_128_to_f8_lo128(x_vlm[t] * inv_b);
            v_f8_st_tnsr_partial((int5){t * 128, row, 0, 0, 0}, codes, pk, 127, 0);
        }
    }
}
#endif
