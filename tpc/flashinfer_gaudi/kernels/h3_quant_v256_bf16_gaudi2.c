// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 fused per-tensor FP8 E4M3 V quantization + 256-col augmentation (col 128 = 1.0f, 129..255 = 0.0f).
// Lets DirectH3Attn fuse PV [BQ, K] @ V [K, 128] and PSUM [BQ, K] @ 1 [K, 1] into ONE single
// 1-activation MME GEMM [BQ, K] @ V_aug [K, 256], eliminating separate PSUM MME + 78 GB/layer SRAM read!
//
// Inputs:
//   0 v_in   BF16 [128, K_pad, NH]  (256-key interleaved BF16 V)
//   1 inv_v  F32  [1]               (240.0f / max(|V|))
// Output:
//   0 v_aug  F8   [256, K_pad, NH]  (cols 0..127 = F8(v_in * inv_v), col 128 = 1.0f [0x38], cols 129..255 = 0.0f)
// Index space: [K_pad / 4, NH].

static inline minifloat256 bf16_128_to_f8_lo128(bfloat128 x) {
    const minifloat256 sp = v_convert_bf16_to_f8_b(x, 0, SW_CLIP_FP, (minifloat256)0);
    minifloat256 pk = v_f8_pack_b(sp, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
    pk = v_f8_pack_b(sp, SW_GROUP_1 | SW_STRIDE_2, pk);
    return v_f8_mov_dual_group_pack_b(pk, SW_PACK21, (minifloat256)0);
}

void main(tensor v_in, tensor inv_v, tensor v_aug)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int5 zc = {0, 0, 0, 0, 0};
    const float inv_f = s_f32_ld_g((__global__ float*)gen_addr(zc, inv_v));
    const float64 inv_v64 = (float64)inv_f;

    // Constant tail_lo: byte 0 = FP8 E4M3 1.0f (0x38), bytes 1..127 = 0.0f
    const bfloat128 one0 = v_bf16_sel_less_u16_b(read_lane_id_2b_b(), (unsigned short)1u,
                                                 (bfloat128)1.0f, (bfloat128)0.0f);
    const minifloat256 tail_lo = bf16_128_to_f8_lo128(one0);

    for (int h = start[1]; h < end[1]; ++h) {
        for (int rb = start[0]; rb < end[0]; ++rb) {
            const int r0 = rb * 4;
            const int5 c0 = {0, r0,     h, 0, 0};
            const int5 c1 = {0, r0 + 1, h, 0, 0};
            const int5 c2 = {0, r0 + 2, h, 0, 0};
            const int5 c3 = {0, r0 + 3, h, 0, 0};

            float64_pair_t f0 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c0, v_in));
            float64_pair_t f1 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c1, v_in));
            float64_pair_t f2 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c2, v_in));
            float64_pair_t f3 = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(c3, v_in));
            f0.v1 *= inv_v64; f0.v2 *= inv_v64;
            f1.v1 *= inv_v64; f1.v2 *= inv_v64;
            f2.v1 *= inv_v64; f2.v2 *= inv_v64;
            f3.v1 *= inv_v64; f3.v2 *= inv_v64;

            const bfloat128 s0 = v_convert_f32_to_bf16_all_b(f0);
            const bfloat128 s1 = v_convert_f32_to_bf16_all_b(f1);
            const bfloat128 s2 = v_convert_f32_to_bf16_all_b(f2);
            const bfloat128 s3 = v_convert_f32_to_bf16_all_b(f3);

            const minifloat256 sp0 = v_convert_bf16_to_f8_b(s0, 0, SW_CLIP_FP, (minifloat256)0);
            const minifloat256 sp1 = v_convert_bf16_to_f8_b(s1, 0, SW_CLIP_FP, (minifloat256)0);
            const minifloat256 sp2 = v_convert_bf16_to_f8_b(s2, 0, SW_CLIP_FP, (minifloat256)0);
            const minifloat256 sp3 = v_convert_bf16_to_f8_b(s3, 0, SW_CLIP_FP, (minifloat256)0);

            minifloat256 p0 = v_f8_pack_b(sp0, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            minifloat256 p1 = v_f8_pack_b(sp1, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            minifloat256 p2 = v_f8_pack_b(sp2, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            minifloat256 p3 = v_f8_pack_b(sp3, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            p0 = v_f8_pack_b(sp0, SW_GROUP_1 | SW_STRIDE_2, p0);
            p1 = v_f8_pack_b(sp1, SW_GROUP_1 | SW_STRIDE_2, p1);
            p2 = v_f8_pack_b(sp2, SW_GROUP_1 | SW_STRIDE_2, p2);
            p3 = v_f8_pack_b(sp3, SW_GROUP_1 | SW_STRIDE_2, p3);

            p0 = v_f8_mov_dual_group_pack_b(p0, SW_PACK21, (minifloat256)0);
            p1 = v_f8_mov_dual_group_pack_b(p1, SW_PACK21, (minifloat256)0);
            p2 = v_f8_mov_dual_group_pack_b(p2, SW_PACK21, (minifloat256)0);
            p3 = v_f8_mov_dual_group_pack_b(p3, SW_PACK21, (minifloat256)0);

            p0 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 0, 2, MkWr(1, 1), p0);
            p1 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 0, 2, MkWr(1, 1), p1);
            p2 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 0, 2, MkWr(1, 1), p2);
            p3 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 0, 2, MkWr(1, 1), p3);
            p0 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 1, 3, MkWr(1, 1), p0);
            p1 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 1, 3, MkWr(1, 1), p1);
            p2 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 1, 3, MkWr(1, 1), p2);
            p3 = v_f8_mov_dual_group_b(tail_lo, 0xffffffff, 1, 3, MkWr(1, 1), p3);

            v_f8_st_tnsr(c0, v_aug, p0);
            v_f8_st_tnsr(c1, v_aug, p1);
            v_f8_st_tnsr(c2, v_aug, p2);
            v_f8_st_tnsr(c3, v_aug, p3);
        }
    }
}
