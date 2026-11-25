// SPDX-License-Identifier: Apache-2.0
// Natural-order BF16 <-> F32 conversions (from the Qwen kernels): v_convert_bf16_to_f32_all_b splits a 128-lane BF16
// vector into even / odd columns; these give (columns 0..63, 64..127) instead, to meet F32 tensors stored naturally.
#pragma once

static inline float64_pair_t bf16_to_f32_linear(bfloat128 input) {
    bfloat128_pair_t unpacked;
    unpacked.v1 = v_bf16_unpack_b(
        input, ((e_group_0) << 8) | ((e_every_second_element) << 9) | ((e_lower_half_group) << 10), unpacked.v1);
    unpacked.v2 = v_bf16_unpack_b(
        input, ((e_group_1) << 8) | ((e_every_second_element) << 9) | ((e_lower_half_group) << 10), unpacked.v2);
    const bfloat128 first_groups = unpacked.v1;
    unpacked.v1 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 0, 1, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 1, 2, MkWr(1, 1), unpacked.v1);
    unpacked.v1 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 1, 3, MkWr(1, 1), unpacked.v1);
    unpacked.v2 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 2, 0, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(unpacked.v2, 0xFFFFFFFF, 2, 1, MkWr(1, 1), unpacked.v2);
    unpacked.v2 = v_bf16_mov_dual_group_b(first_groups, 0xFFFFFFFF, 3, 2, MkWr(1, 1), unpacked.v2);
    const float64_pair_t first = v_convert_bf16_to_f32_all_b(unpacked.v1);
    const float64_pair_t second = v_convert_bf16_to_f32_all_b(unpacked.v2);
    float64_pair_t result;
    result.v1 = first.v1;
    result.v2 = second.v1;
    return result;
}

static inline bfloat128 f32_to_bf16_lo64(float64 x) {
    const bfloat128 sp = v_convert_f32_to_bf16_b(x, 0, SW_RHNE, (bfloat128)0);
    bfloat128 pk = v_bf16_pack_b(sp, SW_GROUP_0 | SW_STRIDE_2, (bfloat128)0);
    pk = v_bf16_pack_b(sp, SW_GROUP_1 | SW_STRIDE_2, pk);
    return v_bf16_mov_dual_group_pack_b(pk, SW_PACK21, (bfloat128)0);
}

static inline bfloat128 f32_pair_to_bf16_linear(float64 lo, float64 hi) {
    bfloat128 p0 = f32_to_bf16_lo64(lo);
    const bfloat128 p1 = f32_to_bf16_lo64(hi);
    p0 = v_bf16_mov_dual_group_b(p1, 0xffffffff, 0, 2, MkWr(1, 1), p0);
    p0 = v_bf16_mov_dual_group_b(p1, 0xffffffff, 1, 3, MkWr(1, 1), p0);
    return p0;
}

