// SPDX-License-Identifier: Apache-2.0
// g2e: GDN output gate, stage 2 (index space [T rows]): per-row FP8 quantisation of the gated values with the
// row maximum assembled from stage 1's group maxima (gated_norm_quant_bf16_gaudi2). Rounding steps exactly as
// the silu_mul_quant kernel.
// Tensors (dim 0 fastest): gated [W, T] bf16 (W % 128 == 0), pmax [64, NG, T] f32, codes [W, T] f8, scale [1, T] f32.

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

void main(tensor gated, tensor pmax, tensor codes, tensor scales) {
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int W = get_dim_size(gated, 0);
    const int NG = get_dim_size(pmax, 1);
    for (int row = start[0]; row < end[0]; ++row) {
        float64 maximum = 0;
        for (int g = 0; g < NG; ++g) {
            int5 pc = {0, g, row, 0, 0};
            maximum = v_f32_max_b(maximum, v_f32_ld_tnsr_b(pc, pmax));
        }
        const float64 scale = round_bf16(round_bf16(maximum + 1.0e-8f) * (float)(bf16)(1.0f / 240.0f));
        const float64 inverse = round_bf16(reciprocal_without_lookup(scale));
        int5 scale_coord = {0, row, 0, 0, 0};
        v_f32_st_tnsr(scale_coord, scales, scale);
        #pragma loop_unroll(4)
        for (int tile = 0; tile < W / 128; ++tile) {
            int5 coord = {tile * 128, row, 0, 0, 0};
            const float64_pair_t fp32 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coord, gated));
            minifloat256 packed = 0;
            packed = v_convert_f32_to_f8_b(round_bf16(fp32.v1 * inverse), 0, SW_CLIP_FP, packed);
            packed = v_convert_f32_to_f8_b(round_bf16(fp32.v2 * inverse), 2, SW_CLIP_FP, packed);
            const minifloat256 sparse = packed;
            packed = v_f8_pack_b(sparse, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            packed = v_f8_pack_b(sparse, SW_GROUP_1 | SW_STRIDE_2, packed);
            packed = v_f8_mov_dual_group_pack_b(packed, SW_PACK21, (minifloat256)0);
            v_f8_st_tnsr_partial(coord, codes, packed, 127, 0);
        }
    }
}
