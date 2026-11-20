// SPDX-License-Identifier: Apache-2.0
// silu_mul_quant with the FP8 GEMM's dequantisation folded in: the gate_up GEMM's raw output (no scales applied)
// is scaled here by its row's activation scale and the columns' weight scales, then
//   act = silu(gate) * up  (FP32, rounded to BF16 as the unfused path's BF16 product), and quantised per row to FP8.
// Two passes per row (the row maximum, then the codes) instead of keeping the products in VLM: the sigmoid's lookup
// table and a large VLM array do not fit together.
// Tensors (dim 0 fastest): raw [2F, M] bf16, sa [1, M] f32 (activation row scales), swb [2F] bf16 (weight column
// scales), output [F, M] fp8, scales [1, M] f32. One index point owns a row.
static inline float64 round_bf16(float64 value) {
    float64_pair_t pair;
    pair.v1 = value;
    pair.v2 = value;
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pair)).v1;
}

static inline float64 reciprocal_without_lookup(float64 value) {
    float64 estimate = as_float64((int64)0x7ef311c3 - as_int64(value));
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    return v_f32_sel_eq_f32_b(value, as_float64((int64)0x7f800000), 0.0f, estimate);
}

static inline float64 row_max_without_lookup(float64 value) {
    value = v_f32_max_b(value, v_f32_mov_dual_group_all_b(value, 0xffffffff, 1, 0, 3, 2,
                                                      MkWrA(3, 3, 3, 3), 0));
    value = v_f32_max_b(value, v_f32_mov_dual_group_all_b(value, 0xffffffff, 2, 3, 0, 1,
                                                      MkWrA(3, 3, 3, 3), 0));
    value = v_f32_max_b(value, v_f32_mov_group_b(value, 0xffffffff, 63, 0));
    float64 result = 0;
    #pragma loop_unroll(8)
    for (int lane = 0; lane < 8; ++lane) {
        const uchar256 broadcast = 0x80 | lane;
        result = v_f32_max_b(result, v_f32_shuffle_b(value, broadcast, 0, value));
    }
    return result;
}

// silu(gate) * up for one 128-column tile (FP32, in the lane order of v_convert_bf16_to_f32_all_b), BF16-rounded
static inline float64_pair_t product(tensor raw, tensor swb, int width, int row, int tile, float a)
{
    int5 coord = {tile * 128, row, 0, 0, 0};
    const float64_pair_t g = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coord, raw));
    int5 wc = {tile * 128, 0, 0, 0, 0};
    const float64_pair_t wg = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, swb));
    coord[0] += width;
    wc[0] += width;
    const float64_pair_t u = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coord, raw));
    const float64_pair_t wu = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wc, swb));
    float64_pair_t r;
    const float64 g1 = g.v1 * wg.v1 * a, g2 = g.v2 * wg.v2 * a;
    const float64 u1 = u.v1 * wu.v1 * a, u2 = u.v2 * wu.v2 * a;
    r.v1 = round_bf16(g1 * v_sigmoid_f32(g1) * u1);
    r.v2 = round_bf16(g2 * v_sigmoid_f32(g2) * u2);
    return r;
}

void main(tensor raw, tensor sa, tensor swb, tensor output, tensor scales)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int width = get_dim_size(output, 0);
    for (int row = start[0]; row < end[0]; ++row) {
        int5 sc = {0, row, 0, 0, 0};
        const float a = s_f32_ld_g(gen_addr(sc, sa));
        float64 maximum = 0;
        for (int tile = 0; tile < width / 128; ++tile) {
            const float64_pair_t p = product(raw, swb, width, row, tile, a);
            maximum = v_f32_max_b(maximum, v_f32_abs_b(p.v1));
            maximum = v_f32_max_b(maximum, v_f32_abs_b(p.v2));
        }
        maximum = row_max_without_lookup(maximum);
        // as silu_mul_quant: every BF16 boundary of the row-quant path
        const float64 scale = round_bf16(round_bf16(maximum + 1.0e-8f) * (float)(bf16)(1.0f / 240.0f));
        const float64 inverse = round_bf16(reciprocal_without_lookup(scale));
        v_f32_st_tnsr(sc, scales, scale);
        for (int tile = 0; tile < width / 128; ++tile) {
            const float64_pair_t fp32 = product(raw, swb, width, row, tile, a);
            minifloat256 packed = 0;
            packed = v_convert_f32_to_f8_b(round_bf16(fp32.v1 * inverse), 0, SW_CLIP_FP, packed);
            packed = v_convert_f32_to_f8_b(round_bf16(fp32.v2 * inverse), 2, SW_CLIP_FP, packed);
            const minifloat256 sparse = packed;
            packed = v_f8_pack_b(sparse, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            packed = v_f8_pack_b(sparse, SW_GROUP_1 | SW_STRIDE_2, packed);
            packed = v_f8_mov_dual_group_pack_b(packed, SW_PACK21, (minifloat256)0);
            int5 coord = {tile * 128, row, 0, 0, 0};
            v_f8_st_tnsr_partial(coord, output, packed, 127, 0);
        }
    }
}
