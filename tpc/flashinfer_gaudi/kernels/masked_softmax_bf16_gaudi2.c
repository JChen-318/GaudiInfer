// SPDX-License-Identifier: Apache-2.0
// g2e: masked, unnormalised softmax of attention scores, one index point per row (VLIW-friendly form):
//   row r (query t = r % T) sees columns c <= last[t];  m = max of the visible scores;
//   p[r, c] = exp(scale * (s[r, c] - m)) (BF16, 0 past the visible columns);  sum[r] = sum of p (FP32 before
//   rounding to BF16). The caller divides PV by sum.
// Steady loops have no masking and no branches: four independent 128-column vectors per iteration (four
// accumulators), the boundary vector is masked separately, columns past it are zero-filled. exp is a LUT-free
// base-2 quadratic (relative error 1.7e-3, below BF16's half ulp 3.9e-3) with the row bias folded into the
// exponent field (after the H3 attention kernels, gaudi2-mme-programming/direct/attention/kernels/exp_sum.c).
// The kernel is VPU-bound (one vector instruction per cycle), so instructions per element are what counts: the
// row maximum and the underflow clamp work on 128 BF16 lanes at once, the sum adds the unrounded FP32 values.
// Tensors (dim 0 fastest): s [S, R] bf16, last [T] i32 (host constants), p [S, R] bf16, sum [1, R] f32.
// S % 128 == 0. p may also be [C, R, S/C] (C % 512 == 0): columns in chunks, ready for a batched PV GEMM.

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

// 2^t for t = x * kl + nb (kl = scale * log2(e), nb = 127 - m * kl: the IEEE bias folded in); x is clamped (in
// BF16, by the caller) so that t >= ~6 keeps the exponent field valid. Relative-minimax quadratic for 2^f,
// f in [-0.5, 0.5].
static inline float64 exp_biased(float64 x, float kl, float64 nb) {
    const float64 t = v_f32_mac_b(x, kl, nb);
    const int64 n = v_convert_f32_to_i32_b(t, SW_RHNE);
    const float64 f = t - v_convert_i32_to_f32_b(n, 0);
    float64 p = 0.23842697f;
    p = v_f32_mac_b(p, f, 0.7034453f);
    p = v_f32_mac_b(p, f, 1.00044307f);
    const int64 bits = n << 23;
    return p * *((float64*)&bits);
}

// p coordinates of score column col of row r (chunked layout when C < S)
static inline int5 p_coord(int col, int r, int C) {
    int5 c = {col % C, r, col / C, 0, 0};
    return c;
}
#define PC(col) p_coord((col), r, C)

// exp of one 128-column BF16 vector (clamped from below at lo) -> BF16 out; the FP32 values added to acc
#define EXP128(xb, out, acc) do {                                                  \
        const float128 x_ = v_convert_bf16_to_f32_all_b(v_bf16_max_b((xb), lo));  \
        float128 y_;                                                               \
        y_.v1 = exp_biased(x_.v1, kl, nb); y_.v2 = exp_biased(x_.v2, kl, nb);      \
        (out) = v_convert_f32_to_bf16_all_b(y_);                                   \
        (acc) += y_.v1 + y_.v2;                                                    \
    } while (0)

#ifdef WITH_MAXES
void main(tensor s, tensor last, tensor p, tensor sums, tensor maxes, float scale)
#else
void main(tensor s, tensor last, tensor p, tensor sums, float scale)
#endif
{
    const float kl = scale * 1.4426950408889634f;  // softmax scale (> 0) folded into the exponent
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int S = get_dim_size(s, 0);
    const int T = get_dim_size(last, 0);
    const bfloat128 zero = 0;
    const bfloat128 neg = -3.0e38f;
    const int chunked = get_dim_size(p, 2) > 1 || get_dim_size(p, 0) != S;
    const int C = chunked ? get_dim_size(p, 0) : S;
    for (int r = start[0]; r < end[0]; ++r) {
        int5 lc = {r % T, 0, 0, 0, 0};
        int visible = s_i32_ld_g((__global__ int*)gen_addr(lc, last)) + 1;
        visible = visible < S ? visible : S;
        const int full = visible / 128;         // whole visible vectors
        const int rest = visible - full * 128;  // visible lanes of the boundary vector
        const int groups = full / 4;
        // ---- pass 1: row maximum (BF16, 128 lanes per instruction)
        bfloat128 m0 = neg, m1 = neg, m2 = neg, m3 = neg;
        int5 c = {0, r, 0, 0, 0};
        for (int g = 0; g < groups; ++g) {
            int5 c0 = {g * 512, r, 0, 0, 0}, c1 = {g * 512 + 128, r, 0, 0, 0};
            int5 c2 = {g * 512 + 256, r, 0, 0, 0}, c3 = {g * 512 + 384, r, 0, 0, 0};
            m0 = v_bf16_max_b(m0, v_bf16_ld_tnsr_b(c0, s));
            m1 = v_bf16_max_b(m1, v_bf16_ld_tnsr_b(c1, s));
            m2 = v_bf16_max_b(m2, v_bf16_ld_tnsr_b(c2, s));
            m3 = v_bf16_max_b(m3, v_bf16_ld_tnsr_b(c3, s));
        }
        for (int v = groups * 4; v < full; ++v) {
            c[0] = v * 128;
            m0 = v_bf16_max_b(m0, v_bf16_ld_tnsr_b(c, s));
        }
        bfloat128 edge = neg;
        if (rest) {  // boundary vector: lanes >= rest masked
            c[0] = full * 128;
            edge = v_bf16_sel_less_u16_b(read_lane_id_2b_b(), (unsigned short)rest, v_bf16_ld_tnsr_b(c, s), neg);
            m1 = v_bf16_max_b(m1, edge);
        }
        const float128 mf = v_convert_bf16_to_f32_all_b(v_bf16_max_b(v_bf16_max_b(m0, m1), v_bf16_max_b(m2, m3)));
        const float64 m = max64(v_f32_max_b(mf.v1, mf.v2));
        const float64 nb = 127.0f - m * kl;
        // underflow clamp: scores below m - 120 / kl give 2^(t >= ~7) ~ 0 instead of a broken exponent field
        float128 lof;
        lof.v1 = m - 120.0f / kl;
        lof.v2 = lof.v1;
        const bfloat128 lo = v_convert_f32_to_bf16_all_b(lof);
        // ---- pass 2: exponentials and their sum
        float64 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int g = 0; g < groups; ++g) {
            int5 c0 = {g * 512, r, 0, 0, 0}, c1 = {g * 512 + 128, r, 0, 0, 0};
            int5 c2 = {g * 512 + 256, r, 0, 0, 0}, c3 = {g * 512 + 384, r, 0, 0, 0};
            const bfloat128 x0 = v_bf16_ld_tnsr_b(c0, s);
            const bfloat128 x1 = v_bf16_ld_tnsr_b(c1, s);
            const bfloat128 x2 = v_bf16_ld_tnsr_b(c2, s);
            const bfloat128 x3 = v_bf16_ld_tnsr_b(c3, s);
            bfloat128 o0, o1, o2, o3;
            EXP128(x0, o0, a0);
            EXP128(x1, o1, a1);
            EXP128(x2, o2, a2);
            EXP128(x3, o3, a3);
            v_bf16_st_tnsr(PC(g * 512), p, o0);
            v_bf16_st_tnsr(PC(g * 512 + 128), p, o1);
            v_bf16_st_tnsr(PC(g * 512 + 256), p, o2);
            v_bf16_st_tnsr(PC(g * 512 + 384), p, o3);
        }
        for (int v = groups * 4; v < full; ++v) {
            c[0] = v * 128;
            const bfloat128 x = v_bf16_ld_tnsr_b(c, s);
            bfloat128 o;
            EXP128(x, o, a0);
            v_bf16_st_tnsr(PC(v * 128), p, o);
        }
        int next = full;
        if (rest) {  // masked lanes were clamped to lo: their ~2^-120 terms vanish in the sum
            bfloat128 o;
            EXP128(edge, o, a1);
            o = v_bf16_sel_less_u16_b(read_lane_id_2b_b(), (unsigned short)rest, o, zero);
            v_bf16_st_tnsr(PC(full * 128), p, o);
            next = full + 1;
        }
        for (int v = next; v < S / 128; ++v) {  // past the visible columns
            v_bf16_st_tnsr(PC(v * 128), p, zero);
        }
        int5 sc = {0, r, 0, 0, 0};
        v_f32_st_tnsr(sc, sums, sum64((a0 + a1) + (a2 + a3)));
#ifdef WITH_MAXES
        v_f32_st_tnsr(sc, maxes, m);
#endif
    }
}
