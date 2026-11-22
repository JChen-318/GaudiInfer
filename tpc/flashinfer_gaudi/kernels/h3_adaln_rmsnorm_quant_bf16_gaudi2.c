// SPDX-License-Identifier: Apache-2.0
// MiniMax-H3 fused RMSNorm + Indexed AdaLN (scale/shift) + Dynamic Row FP8 Quantization.
// When WITH_GATE_RESIDUAL is defined, also fuses the preceding gated residual:
//   res_out = bf16(f32(residual) + f32(gate[idx]) * f32(branch))
//   norm    = bf16(f32(res_out) * rsqrt(mean(res_out^2) + eps) * f32(weight))
//   mod     = bf16(f32(norm) * (1 + f32(scale[idx])) + f32(shift[idx]))
//   quant   = fp8_143(mod * inv_scale), row_scale = bf16((max|mod| + 1e-8) * inv_range)
//
// Width must be a multiple of 128 and <= 8192 (H3 hidden_size = 5376 -> 42 tiles, fits in VLM).
#include "norm_math_gaudi2.h"

#define H3_ADALN_MAX_TILES 64

static inline float64 bf16_round(float64 value) {
    float64_pair_t pair;
    pair.v1 = value;
    pair.v2 = value;
    return v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(pair)).v1;
}

static inline float64 positive_reciprocal(float64 value) {
    float64 estimate = as_float64((int64)0x7ef311c3 - as_int64(value));
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    estimate = estimate * (2.0f - value * estimate);
    return v_f32_sel_eq_f32_b(value, as_float64((int64)0x7f800000), 0.0f, estimate);
}

static inline float64 row_max(float64 value) {
    value = v_f32_max_b(value, v_f32_mov_dual_group_all_b(value, 0xffffffff, 1, 0, 3, 2, MkWrA(3, 3, 3, 3), 0));
    value = v_f32_max_b(value, v_f32_mov_dual_group_all_b(value, 0xffffffff, 2, 3, 0, 1, MkWrA(3, 3, 3, 3), 0));
    value = v_f32_max_b(value, v_f32_mov_group_b(value, 0xffffffff, 63, 0));
    float64 result = 0;
    #pragma loop_unroll(8)
    for (int lane = 0; lane < 8; ++lane) {
        const uchar256 broadcast = 0x80 | lane;
        result = v_f32_max_b(result, v_f32_shuffle_b(value, broadcast, 0, value));
    }
    return result;
}

#ifdef WITH_GATE_RESIDUAL
void main(tensor residual, tensor branch, tensor gate, tensor weight, tensor shift, tensor scale_in,
          tensor indices, tensor quantized, tensor scales, tensor modulated, tensor residual_out,
          float epsilon, float inverse_width, float inverse_range)
#else
void main(tensor input, tensor weight, tensor shift, tensor scale_in, tensor indices,
          tensor quantized, tensor scales, tensor modulated,
          float epsilon, float inverse_width, float inverse_range)
#endif
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
#ifdef WITH_GATE_RESIDUAL
    const int width = get_dim_size(residual, 0);
#else
    const int width = get_dim_size(input, 0);
#endif
    const int tiles = width / 128;
    bfloat128 cached[H3_ADALN_MAX_TILES];

    for (int row = start[0]; row < end[0]; ++row) {
        const int5 idx_coord = {row, 0, 0, 0, 0};
        const int idx = s_i32_ld_g(gen_addr(idx_coord, indices));

        // Pass 1: load (or compute gated residual), accumulate sum of squares in FP32, cache x in VLM
        float128 sq0 = {0}, sq1 = {0};
        for (int tile = 0; tile < tiles; ++tile) {
            const int col = tile * 128;
            const int5 coord = {col, row, 0, 0, 0};
            bfloat128 xv;
#ifdef WITH_GATE_RESIDUAL
            const int5 gcoord = {col, idx, 0, 0, 0};
            const float64_pair_t rf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coord, residual));
            const float64_pair_t bf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(coord, branch));
            const float64_pair_t gf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(gcoord, gate));
            float64_pair_t sf;
            sf.v1 = v_f32_mac_b(gf.v1, bf.v1, rf.v1);
            sf.v2 = v_f32_mac_b(gf.v2, bf.v2, rf.v2);
            xv = v_convert_f32_to_bf16_all_b(sf);
            v_bf16_st_tnsr(coord, residual_out, xv);
#else
            xv = v_bf16_ld_tnsr_b(coord, input);
#endif
            cached[tile] = xv;
            if (tile & 1) {
                sq1 = v_bf16_mac_acc32_b(xv, xv, sq1, (e_no_negation) << 1);
            } else {
                sq0 = v_bf16_mac_acc32_b(xv, xv, sq0, (e_no_negation) << 1);
            }
        }

        const float64 sum_sq = (sq0.v1 + sq1.v1) + (sq0.v2 + sq1.v2);
        const float64 rrms = positive_rsqrt(row_sum(sum_sq) * inverse_width + epsilon);

        // Pass 2: RMSNorm + AdaLN modulation -> store BF16 modulated, cache in VLM, track abs max
        bfloat128 max_v = 0;
        for (int tile = 0; tile < tiles; ++tile) {
            const int col = tile * 128;
            const int5 wcoord = {col, 0, 0, 0, 0};
            const int5 acoord = {col, idx, 0, 0, 0};
            const int5 ocoord = {col, row, 0, 0, 0};

            float64_pair_t xf = v_convert_bf16_to_f32_all_b(cached[tile]);
            const float64_pair_t wf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(wcoord, weight));
            xf.v1 = (xf.v1 * rrms) * wf.v1;
            xf.v2 = (xf.v2 * rrms) * wf.v2;
            // Match HPU reference: RMSNorm rounds to BF16 before AdaLN FP32 addcmul
            const bfloat128 norm_b = v_convert_f32_to_bf16_all_b(xf);
            const float64_pair_t nf = v_convert_bf16_to_f32_all_b(norm_b);

            const float64_pair_t scf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(acoord, scale_in));
            const float64_pair_t shf = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(acoord, shift));
            float64_pair_t mf;
            mf.v1 = v_f32_mac_b(nf.v1, scf.v1 + 1.0f, shf.v1);
            mf.v2 = v_f32_mac_b(nf.v2, scf.v2 + 1.0f, shf.v2);
            const bfloat128 mod_b = v_convert_f32_to_bf16_all_b(mf);

            v_bf16_st_tnsr(ocoord, modulated, mod_b);
            cached[tile] = mod_b;
            max_v = v_bf16_max_b(max_v, v_bf16_abs_b(mod_b));
        }

        const float64_pair_t max_f = v_convert_bf16_to_f32_all_b(max_v);
        const float64 maximum = row_max(v_f32_max_b(max_f.v1, max_f.v2));
        const float64 row_sc = bf16_round(bf16_round(maximum + 1.0e-8f) * inverse_range);
        const float64 inv_sc = bf16_round(positive_reciprocal(row_sc));
        float64_pair_t inv_pair;
        inv_pair.v1 = inv_sc;
        inv_pair.v2 = inv_sc;
        const bfloat128 inv_bf16 = v_convert_f32_to_bf16_all_b(inv_pair);
        const int5 sc_coord = {0, row, 0, 0, 0};
        v_f32_st_tnsr(sc_coord, scales, row_sc);

        // Pass 3: quantize cached BF16 modulated rows to packed fp8_143 (256 bytes per pair of tiles)
        const int pairs = tiles & ~1;
        #pragma loop_unroll(2)
        for (int tile = 0; tile < pairs; tile += 2) {
            const int5 c0 = {tile * 128, row, 0, 0, 0};
            const minifloat256 sp0 = v_convert_bf16_to_f8_b(cached[tile] * inv_bf16, 0, SW_CLIP_FP, (minifloat256)0);
            minifloat256 pk0 = v_f8_pack_b(sp0, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            pk0 = v_f8_pack_b(sp0, SW_GROUP_1 | SW_STRIDE_2, pk0);
            pk0 = v_f8_mov_dual_group_pack_b(pk0, SW_PACK21, (minifloat256)0);

            const minifloat256 sp1 = v_convert_bf16_to_f8_b(cached[tile + 1] * inv_bf16, 0, SW_CLIP_FP, (minifloat256)0);
            minifloat256 pk1 = v_f8_pack_b(sp1, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            pk1 = v_f8_pack_b(sp1, SW_GROUP_1 | SW_STRIDE_2, pk1);
            pk1 = v_f8_mov_dual_group_pack_b(pk1, SW_PACK21, (minifloat256)0);

            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 0, 2, MkWr(1, 1), pk0);
            pk0 = v_f8_mov_dual_group_b(pk1, 0xffffffff, 1, 3, MkWr(1, 1), pk0);
            v_f8_st_tnsr(c0, quantized, pk0);
        }
        for (int tile = pairs; tile < tiles; ++tile) {
            const int5 coord = {tile * 128, row, 0, 0, 0};
            const minifloat256 sp = v_convert_bf16_to_f8_b(cached[tile] * inv_bf16, 0, SW_CLIP_FP, (minifloat256)0);
            minifloat256 pk = v_f8_pack_b(sp, SW_GROUP_0 | SW_STRIDE_2, (minifloat256)0);
            pk = v_f8_pack_b(sp, SW_GROUP_1 | SW_STRIDE_2, pk);
            pk = v_f8_mov_dual_group_pack_b(pk, SW_PACK21, (minifloat256)0);
            v_f8_st_tnsr_partial(coord, quantized, pk, 127, 0);
        }
    }
}
