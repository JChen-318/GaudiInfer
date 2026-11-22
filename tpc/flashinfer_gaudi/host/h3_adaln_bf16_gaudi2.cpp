// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstring>
#include <limits>
#include "h3_adaln_bf16_gaudi2.hpp"

extern unsigned char _binary_h3_adaln_rmsnorm_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_adaln_rmsnorm_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_gate_adaln_rmsnorm_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_gate_adaln_rmsnorm_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_gate_residual_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_gate_residual_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_qk_norm_rope_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_qk_norm_rope_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_maxfree_exp_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_maxfree_exp_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_fast_softmax_fp8_gaudi2_o_start;
extern unsigned char _binary_h3_fast_softmax_fp8_gaudi2_o_end;
extern unsigned char _binary_h3_silu_mul_quant_dual_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_silu_mul_quant_dual_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_silu_mul_quant_dual_dq_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_silu_mul_quant_dual_dq_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_vvae_qk_norm_rope_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_vvae_qk_norm_rope_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_vvae_scale_add_norm_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_vvae_scale_add_norm_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_vvae_rmsnorm_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_vvae_rmsnorm_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_row_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_row_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_qk_norm_rope_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_qk_norm_rope_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_h3_quant_v256_bf16_gaudi2_o_start;
extern unsigned char _binary_h3_quant_v256_bf16_gaudi2_o_end;

namespace {

using namespace tpc_lib_api;

void map_2d_row(TensorAccessPattern& p, uint64_t width) {
    std::memset(&p, 0, sizeof(p));
    p.mapping[0].indexSpaceDim = 0;
    p.mapping[0].start_b = 0;
    p.mapping[0].end_b = static_cast<int32_t>(width) - 1;
    p.mapping[1].indexSpaceDim = 0;
    p.mapping[1].a = 1;
    p.mapping[1].start_b = 0;
    p.mapping[1].end_b = 0;
}

void map_1d_full(TensorAccessPattern& p, uint64_t width) {
    std::memset(&p, 0, sizeof(p));
    p.mapping[0].indexSpaceDim = 0;
    p.mapping[0].start_b = 0;
    p.mapping[0].end_b = static_cast<int32_t>(width) - 1;
}

void map_2d_table(TensorAccessPattern& p, uint64_t width, uint64_t table_rows) {
    std::memset(&p, 0, sizeof(p));
    p.mapping[0].indexSpaceDim = 0;
    p.mapping[0].start_b = 0;
    p.mapping[0].end_b = static_cast<int32_t>(width) - 1;
    p.mapping[1].indexSpaceDim = 0;
    p.mapping[1].a = 0;
    p.mapping[1].start_b = 0;
    p.mapping[1].end_b = static_cast<int32_t>(table_rows) - 1;
}

void map_1d_row_idx(TensorAccessPattern& p) {
    std::memset(&p, 0, sizeof(p));
    p.mapping[0].indexSpaceDim = 0;
    p.mapping[0].a = 1;
    p.mapping[0].start_b = 0;
    p.mapping[0].end_b = 0;
}

GlueCodeReturn copy_elf(HabanaKernelInstantiation* instance, const unsigned char* begin, const unsigned char* end) {
    const unsigned size = static_cast<unsigned>(end - begin);
    const unsigned cap = instance->kernel.elfSize;
    instance->kernel.elfSize = size;
    if (cap < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, begin, size);
    return GLUE_SUCCESS;
}

}  // namespace

GlueCodeReturn H3AdalnRmsNormQuantBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(AddRmsNormQuantParams))
        return GLUE_FAILED;

    AddRmsNormQuantParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    const auto& g_in = params->inputTensors[0].geometry;
    const auto width = g_in.maxSizes[0], rows = g_in.maxSizes[1];
    if (g_in.dims != 2 || g_in.dataType != DATA_BF16 || !rows ||
        rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        width < 128 || width > 8192 || width % 128 || !std::isnormal(scalar.epsilon) || scalar.epsilon <= 0)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    // 0: input [W, R] bf16, 1: weight [W] bf16, 2: shift [W, M3] bf16, 3: scale [W, M3] bf16, 4: idx [R] i32
    const auto& g_w = params->inputTensors[1].geometry;
    const auto& g_sh = params->inputTensors[2].geometry;
    const auto& g_sc = params->inputTensors[3].geometry;
    const auto& g_ix = params->inputTensors[4].geometry;
    if (g_w.dataType != DATA_BF16 || g_sh.dataType != DATA_BF16 || g_sc.dataType != DATA_BF16 || g_ix.dataType != DATA_I32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto m3 = g_sh.maxSizes[1];
    if (g_w.dims != 1 || g_w.maxSizes[0] != width ||
        g_sh.dims != 2 || g_sh.maxSizes[0] != width || !m3 ||
        g_sc.dims != 2 || g_sc.maxSizes[0] != width || g_sc.maxSizes[1] != m3 ||
        g_ix.dims != 1 || g_ix.maxSizes[0] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    // Outputs: 0: quantized [W, R] f8_143, 1: scales [1, R] f32, 2: modulated [W, R] bf16
    const TensorDataType odt[3] = {DATA_F8_143, DATA_F32, DATA_BF16};
    for (int i = 0; i < 3; ++i) {
        const auto& go = params->outputTensors[i].geometry;
        if (go.dataType != odt[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (go.dims != 2 || go.maxSizes[0] != (i == 1 ? 1u : width) || go.maxSizes[1] != rows)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = rows;
    map_2d_row(instance->inputTensorAccessPattern[0], width);
    map_1d_full(instance->inputTensorAccessPattern[1], width);
    map_2d_table(instance->inputTensorAccessPattern[2], width, m3);
    map_2d_table(instance->inputTensorAccessPattern[3], width, m3);
    map_1d_row_idx(instance->inputTensorAccessPattern[4]);

    map_2d_row(instance->outputTensorAccessPattern[0], width);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);
    map_2d_row(instance->outputTensorAccessPattern[2], width);

    instance->kernel.paramsNr = sizeof(scalar) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &scalar, sizeof(scalar));
    return copy_elf(instance, &_binary_h3_adaln_rmsnorm_quant_bf16_gaudi2_o_start,
                    &_binary_h3_adaln_rmsnorm_quant_bf16_gaudi2_o_end);
}

GlueCodeReturn H3GateAdalnRmsNormQuantBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 7) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 4) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(AddRmsNormQuantParams))
        return GLUE_FAILED;

    AddRmsNormQuantParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    const auto& g_res = params->inputTensors[0].geometry;
    const auto width = g_res.maxSizes[0], rows = g_res.maxSizes[1];
    if (g_res.dims != 2 || g_res.dataType != DATA_BF16 || !rows ||
        rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        width < 128 || width > 8192 || width % 128 || !std::isnormal(scalar.epsilon) || scalar.epsilon <= 0)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    // 0: res [W,R], 1: branch [W,R], 2: gate [W,M3], 3: weight [W], 4: shift [W,M3], 5: scale [W,M3], 6: idx [R]
    const auto& g_br = params->inputTensors[1].geometry;
    const auto& g_gt = params->inputTensors[2].geometry;
    const auto& g_w = params->inputTensors[3].geometry;
    const auto& g_sh = params->inputTensors[4].geometry;
    const auto& g_sc = params->inputTensors[5].geometry;
    const auto& g_ix = params->inputTensors[6].geometry;
    if (g_br.dataType != DATA_BF16 || g_gt.dataType != DATA_BF16 || g_w.dataType != DATA_BF16 ||
        g_sh.dataType != DATA_BF16 || g_sc.dataType != DATA_BF16 || g_ix.dataType != DATA_I32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto m3 = g_gt.maxSizes[1];
    if (g_br.dims != 2 || g_br.maxSizes[0] != width || g_br.maxSizes[1] != rows ||
        g_gt.dims != 2 || g_gt.maxSizes[0] != width || !m3 ||
        g_w.dims != 1 || g_w.maxSizes[0] != width ||
        g_sh.dims != 2 || g_sh.maxSizes[0] != width || g_sh.maxSizes[1] != m3 ||
        g_sc.dims != 2 || g_sc.maxSizes[0] != width || g_sc.maxSizes[1] != m3 ||
        g_ix.dims != 1 || g_ix.maxSizes[0] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    // Outputs: 0: quantized [W,R] f8_143, 1: scales [1,R] f32, 2: modulated [W,R] bf16, 3: res_out [W,R] bf16
    const TensorDataType odt[4] = {DATA_F8_143, DATA_F32, DATA_BF16, DATA_BF16};
    for (int i = 0; i < 4; ++i) {
        const auto& go = params->outputTensors[i].geometry;
        if (go.dataType != odt[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (go.dims != 2 || go.maxSizes[0] != (i == 1 ? 1u : width) || go.maxSizes[1] != rows)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = rows;
    map_2d_row(instance->inputTensorAccessPattern[0], width);
    map_2d_row(instance->inputTensorAccessPattern[1], width);
    map_2d_table(instance->inputTensorAccessPattern[2], width, m3);
    map_1d_full(instance->inputTensorAccessPattern[3], width);
    map_2d_table(instance->inputTensorAccessPattern[4], width, m3);
    map_2d_table(instance->inputTensorAccessPattern[5], width, m3);
    map_1d_row_idx(instance->inputTensorAccessPattern[6]);

    map_2d_row(instance->outputTensorAccessPattern[0], width);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);
    map_2d_row(instance->outputTensorAccessPattern[2], width);
    map_2d_row(instance->outputTensorAccessPattern[3], width);

    instance->kernel.paramsNr = sizeof(scalar) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &scalar, sizeof(scalar));
    return copy_elf(instance, &_binary_h3_gate_adaln_rmsnorm_quant_bf16_gaudi2_o_start,
                    &_binary_h3_gate_adaln_rmsnorm_quant_bf16_gaudi2_o_end);
}

GlueCodeReturn H3GateResidualBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern)
        return GLUE_FAILED;

    const auto& g_res = params->inputTensors[0].geometry;
    const auto width = g_res.maxSizes[0], rows = g_res.maxSizes[1];
    if (g_res.dims != 2 || g_res.dataType != DATA_BF16 || !rows ||
        rows > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        width < 128 || width % 128)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    const auto& g_br = params->inputTensors[1].geometry;
    const auto& g_gt = params->inputTensors[2].geometry;
    const auto& g_ix = params->inputTensors[3].geometry;
    const auto& g_out = params->outputTensors[0].geometry;
    if (g_br.dataType != DATA_BF16 || g_gt.dataType != DATA_BF16 || g_ix.dataType != DATA_I32 ||
        g_out.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const auto m3 = g_gt.maxSizes[1];
    if (g_br.dims != 2 || g_br.maxSizes[0] != width || g_br.maxSizes[1] != rows ||
        g_gt.dims != 2 || g_gt.maxSizes[0] != width || !m3 ||
        g_ix.dims != 1 || g_ix.maxSizes[0] != rows ||
        g_out.dims != 2 || g_out.maxSizes[0] != width || g_out.maxSizes[1] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = rows;
    map_2d_row(instance->inputTensorAccessPattern[0], width);
    map_2d_row(instance->inputTensorAccessPattern[1], width);
    map_2d_table(instance->inputTensorAccessPattern[2], width, m3);
    map_1d_row_idx(instance->inputTensorAccessPattern[3]);
    map_2d_row(instance->outputTensorAccessPattern[0], width);

    instance->kernel.paramsNr = 0;
    return copy_elf(instance, &_binary_h3_gate_residual_bf16_gaudi2_o_start,
                    &_binary_h3_gate_residual_bf16_gaudi2_o_end);
}

GlueCodeReturn H3QkNormRopeBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(float))
        return GLUE_FAILED;

    float eps = 0.0f;
    std::memcpy(&eps, params->nodeParams.nodeParams, sizeof(eps));
    const auto& g_qkv = params->inputTensors[0].geometry;
    const auto& g_qn = params->inputTensors[1].geometry;
    const auto& g_kn = params->inputTensors[2].geometry;
    const auto& g_cos = params->inputTensors[3].geometry;
    const auto& g_sin = params->inputTensors[4].geometry;
    const auto& g_qo = params->outputTensors[0].geometry;
    const auto& g_ko = params->outputTensors[1].geometry;
    const auto& g_vo = params->outputTensors[2].geometry;

    if (g_qkv.dataType != DATA_BF16 || g_qn.dataType != DATA_BF16 || g_kn.dataType != DATA_BF16 ||
        g_cos.dataType != DATA_BF16 || g_sin.dataType != DATA_BF16 ||
        g_qo.dataType != DATA_BF16 || g_ko.dataType != DATA_BF16 || g_vo.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;

    // qkv: [3*NH*128, S]; q_out/k_out/v_out: [128, U, NH]
    if (g_qkv.dims != 2 || g_qo.dims != 3 || g_ko.dims != 3 || g_vo.dims != 3)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t S = g_qkv.maxSizes[1];
    const uint64_t U = g_qo.maxSizes[1];
    const uint64_t NH = g_qo.maxSizes[2];
    if (!U || U > S || !NH || g_qo.maxSizes[0] != 128 ||
        g_qkv.maxSizes[0] != 3 * NH * 128 ||
        g_qn.dims != 1 || g_qn.maxSizes[0] != 128 ||
        g_kn.dims != 1 || g_kn.maxSizes[0] != 128 ||
        g_cos.dims != 2 || g_cos.maxSizes[0] != 128 || g_cos.maxSizes[1] != S ||
        g_sin.dims != 2 || g_sin.maxSizes[0] != 128 || g_sin.maxSizes[1] != S ||
        g_ko.maxSizes[0] != 128 || g_ko.maxSizes[1] != U || g_ko.maxSizes[2] != NH ||
        g_vo.maxSizes[0] != 128 || g_vo.maxSizes[1] != U || g_vo.maxSizes[2] != NH)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = U;
    map_2d_row(instance->inputTensorAccessPattern[0], 3 * NH * 128);
    map_1d_full(instance->inputTensorAccessPattern[1], 128);
    map_1d_full(instance->inputTensorAccessPattern[2], 128);
    map_2d_row(instance->inputTensorAccessPattern[3], 128);
    map_2d_row(instance->inputTensorAccessPattern[4], 128);

    auto map_3d_out = [&](TensorAccessPattern& p) {
        std::memset(&p, 0, sizeof(p));
        p.mapping[0].indexSpaceDim = 0;
        p.mapping[0].end_b = 127;
        p.mapping[1].indexSpaceDim = 0;
        p.mapping[1].a = 1;
        p.mapping[2].indexSpaceDim = 0;
        p.mapping[2].end_b = static_cast<int32_t>(NH) - 1;
    };
    map_3d_out(instance->outputTensorAccessPattern[0]);
    map_3d_out(instance->outputTensorAccessPattern[1]);
    map_3d_out(instance->outputTensorAccessPattern[2]);

    instance->kernel.paramsNr = 1;
    std::memcpy(instance->kernel.scalarParams, &eps, sizeof(eps));
    return copy_elf(instance, &_binary_h3_qk_norm_rope_bf16_gaudi2_o_start,
                    &_binary_h3_qk_norm_rope_bf16_gaudi2_o_end);
}

GlueCodeReturn H3MaxFreeExpBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(float))
        return GLUE_FAILED;

    float pad_count = 0.0f;
    std::memcpy(&pad_count, params->nodeParams.nodeParams, sizeof(pad_count));
    const auto& g_sc = params->inputTensors[0].geometry;
    const auto& g_bd = params->inputTensors[1].geometry;
    const auto& g_pr = params->outputTensors[0].geometry;
    const auto& g_is = params->outputTensors[1].geometry;
    if (g_sc.dataType != DATA_BF16 || g_bd.dataType != DATA_BF16 ||
        g_pr.dataType != DATA_BF16 || g_is.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (g_sc.dims != 2 || g_bd.dims != 2 || g_pr.dims != 2 || g_is.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t K = g_sc.maxSizes[0], R = g_sc.maxSizes[1];
    if (!K || !R || (K % 128) ||
        g_bd.maxSizes[0] != 1 || g_bd.maxSizes[1] != R ||
        g_pr.maxSizes[0] != K || g_pr.maxSizes[1] != R ||
        g_is.maxSizes[0] != 1 || g_is.maxSizes[1] != R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = R;
    map_2d_row(instance->inputTensorAccessPattern[0], K);
    map_2d_row(instance->inputTensorAccessPattern[1], 1);
    map_2d_row(instance->outputTensorAccessPattern[0], K);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);

    instance->kernel.paramsNr = 1;
    std::memcpy(instance->kernel.scalarParams, &pad_count, sizeof(pad_count));
    return copy_elf(instance, &_binary_h3_maxfree_exp_bf16_gaudi2_o_start,
                    &_binary_h3_maxfree_exp_bf16_gaudi2_o_end);
}

GlueCodeReturn H3FastSoftmaxFp8Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 3) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(int32_t))
        return GLUE_FAILED;

    int32_t valid_k = 0;
    std::memcpy(&valid_k, params->nodeParams.nodeParams, sizeof(valid_k));
    const auto& g_sc = params->inputTensors[0].geometry;
    const auto& g_ss = params->inputTensors[1].geometry;
    const auto& g_dv = params->inputTensors[2].geometry;
    const auto& g_pr = params->outputTensors[0].geometry;
    const auto& g_is = params->outputTensors[1].geometry;
    if (g_sc.dataType != DATA_BF16 || g_ss.dataType != DATA_F32 || g_dv.dataType != DATA_F32 ||
        g_pr.dataType != DATA_F8_143 || g_is.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (g_sc.dims != 2 || g_ss.dims != 1 || g_dv.dims != 1 || g_pr.dims != 2 || g_is.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t K = g_sc.maxSizes[0], R = g_sc.maxSizes[1];
    if (!K || !R || (K % 256) || valid_k <= 0 || uint64_t(valid_k) > K ||
        g_ss.maxSizes[0] != 1 || g_dv.maxSizes[0] != 1 ||
        g_pr.maxSizes[0] != K || g_pr.maxSizes[1] != R ||
        g_is.maxSizes[0] != 1 || g_is.maxSizes[1] != R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = R;
    map_2d_row(instance->inputTensorAccessPattern[0], K);
    map_1d_full(instance->inputTensorAccessPattern[1], 1);
    map_1d_full(instance->inputTensorAccessPattern[2], 1);
    map_2d_row(instance->outputTensorAccessPattern[0], K);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);

    instance->kernel.paramsNr = 1;
    std::memcpy(instance->kernel.scalarParams, &valid_k, sizeof(valid_k));
    return copy_elf(instance, &_binary_h3_fast_softmax_fp8_gaudi2_o_start,
                    &_binary_h3_fast_softmax_fp8_gaudi2_o_end);
}

GlueCodeReturn H3SiluMulQuantDualBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 1 && params->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern)
        return GLUE_FAILED;

    const bool dq = (params->inputTensorNr == 4);
    const auto& g_gu = params->inputTensors[0].geometry;
    const auto& g_cd = params->outputTensors[0].geometry;
    const auto& g_sc = params->outputTensors[1].geometry;
    const auto& g_ab = params->outputTensors[2].geometry;
    if (g_gu.dataType != DATA_BF16 || g_cd.dataType != DATA_F8_143 ||
        g_sc.dataType != DATA_F32 || g_ab.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (g_gu.dims != 2 || g_cd.dims != 2 || g_sc.dims != 2 || g_ab.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t F = g_cd.maxSizes[0], R = g_cd.maxSizes[1];
    if (!F || !R || (F % 128) || F > 136 * 128 ||
        g_gu.maxSizes[0] != 2 * F || g_gu.maxSizes[1] != R ||
        g_sc.maxSizes[0] != 1 || g_sc.maxSizes[1] != R ||
        g_ab.maxSizes[0] != F || g_ab.maxSizes[1] != R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (dq) {
        const auto& g_sa = params->inputTensors[1].geometry;
        const auto& g_sw = params->inputTensors[2].geometry;
        const auto& g_dl = params->inputTensors[3].geometry;
        if (g_sa.dataType != DATA_F32 || g_sw.dataType != DATA_BF16 || g_dl.dataType != DATA_BF16)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (g_sa.dims != 2 || g_sa.maxSizes[0] != 1 || g_sa.maxSizes[1] != R ||
            g_sw.dims != 1 || g_sw.maxSizes[0] != 2 * F ||
            g_dl.dims != 2 || g_dl.maxSizes[0] != 2 * F || g_dl.maxSizes[1] != R)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = R;
    map_2d_row(instance->inputTensorAccessPattern[0], 2 * F);
    if (dq) {
        map_2d_row(instance->inputTensorAccessPattern[1], 1);
        map_1d_full(instance->inputTensorAccessPattern[2], 2 * F);
        map_2d_row(instance->inputTensorAccessPattern[3], 2 * F);
    }
    map_2d_row(instance->outputTensorAccessPattern[0], F);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);
    map_2d_row(instance->outputTensorAccessPattern[2], F);
    instance->kernel.paramsNr = 0;
    return dq ? copy_elf(instance, &_binary_h3_silu_mul_quant_dual_dq_bf16_gaudi2_o_start,
                         &_binary_h3_silu_mul_quant_dual_dq_bf16_gaudi2_o_end)
              : copy_elf(instance, &_binary_h3_silu_mul_quant_dual_bf16_gaudi2_o_start,
                         &_binary_h3_silu_mul_quant_dual_bf16_gaudi2_o_end);
}

GlueCodeReturn H3VvaeQkNormRopeBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(float))
        return GLUE_FAILED;

    float eps = 0.0f;
    std::memcpy(&eps, params->nodeParams.nodeParams, sizeof(eps));
    const auto& g_qkv = params->inputTensors[0].geometry;
    const auto& g_qb  = params->inputTensors[1].geometry;
    const auto& g_cos = params->inputTensors[2].geometry;
    const auto& g_sin = params->inputTensors[3].geometry;
    const auto& g_qo  = params->outputTensors[0].geometry;
    const auto& g_ko  = params->outputTensors[1].geometry;
    const auto& g_vo  = params->outputTensors[2].geometry;
    if (g_qkv.dataType != DATA_BF16 || g_qb.dataType != DATA_BF16 ||
        g_cos.dataType != DATA_BF16 || g_sin.dataType != DATA_BF16 ||
        g_qo.dataType != DATA_BF16 || g_ko.dataType != DATA_BF16 || g_vo.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    // qkv: [64, 3, NH, BS]; qkv_b: [64, 3, NH]; cos/sin: [64, S]; out: [64, S, NH, B]
    if (g_qkv.dims != 4 || g_qb.dims != 3 || g_cos.dims != 2 || g_sin.dims != 2 ||
        g_qo.dims != 4 || g_ko.dims != 4 || g_vo.dims != 4)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t S = g_qo.maxSizes[1], NH = g_qo.maxSizes[2], B = g_qo.maxSizes[3], BS = B * S;
    if (!S || !NH || !B || g_qo.maxSizes[0] != 64 ||
        g_qkv.maxSizes[0] != 64 || g_qkv.maxSizes[1] != 3 || g_qkv.maxSizes[2] != NH || g_qkv.maxSizes[3] != BS ||
        g_qb.maxSizes[0] != 64 || g_qb.maxSizes[1] != 3 || g_qb.maxSizes[2] != NH ||
        g_cos.maxSizes[0] != 64 || g_cos.maxSizes[1] != S ||
        g_sin.maxSizes[0] != 64 || g_sin.maxSizes[1] != S ||
        g_ko.maxSizes[0] != 64 || g_ko.maxSizes[1] != S || g_ko.maxSizes[2] != NH || g_ko.maxSizes[3] != B ||
        g_vo.maxSizes[0] != 64 || g_vo.maxSizes[1] != S || g_vo.maxSizes[2] != NH || g_vo.maxSizes[3] != B)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 2;
    instance->indexSpaceGeometry[0] = S;
    instance->indexSpaceGeometry[1] = B;

    // Input 0: [64, 3, NH, BS] -> dim3 covers full BS conservatively per batch slice
    {
        auto& p = instance->inputTensorAccessPattern[0];
        std::memset(&p, 0, sizeof(p));
        p.mapping[0].end_b = 63;
        p.mapping[1].end_b = 2;
        p.mapping[2].end_b = static_cast<int32_t>(NH) - 1;
        p.mapping[3].indexSpaceDim = 1;
        p.mapping[3].a = static_cast<int32_t>(S);
        p.mapping[3].start_b = 0;
        p.mapping[3].end_b = static_cast<int32_t>(S) - 1;
    }
    // Input 1: [64, 3, NH]
    {
        auto& p = instance->inputTensorAccessPattern[1];
        std::memset(&p, 0, sizeof(p));
        p.mapping[0].end_b = 63;
        p.mapping[1].end_b = 2;
        p.mapping[2].end_b = static_cast<int32_t>(NH) - 1;
    }
    map_2d_row(instance->inputTensorAccessPattern[2], 64);
    map_2d_row(instance->inputTensorAccessPattern[3], 64);

    auto map_4d_out = [&](TensorAccessPattern& p) {
        std::memset(&p, 0, sizeof(p));
        p.mapping[0].end_b = 63;
        p.mapping[1].indexSpaceDim = 0;
        p.mapping[1].a = 1;
        p.mapping[2].end_b = static_cast<int32_t>(NH) - 1;
        p.mapping[3].indexSpaceDim = 1;
        p.mapping[3].a = 1;
    };
    map_4d_out(instance->outputTensorAccessPattern[0]);
    map_4d_out(instance->outputTensorAccessPattern[1]);
    map_4d_out(instance->outputTensorAccessPattern[2]);

    instance->kernel.paramsNr = 1;
    std::memcpy(instance->kernel.scalarParams, &eps, sizeof(eps));
    return copy_elf(instance, &_binary_h3_vvae_qk_norm_rope_bf16_gaudi2_o_start,
                    &_binary_h3_vvae_qk_norm_rope_bf16_gaudi2_o_end);
}

GlueCodeReturn H3VvaeScaleAddNormQuantBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(AddRmsNormQuantParams))
        return GLUE_FAILED;

    AddRmsNormQuantParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    const auto& g_res = params->inputTensors[0].geometry;
    const auto& g_br  = params->inputTensors[1].geometry;
    const auto& g_bi  = params->inputTensors[2].geometry;
    const auto& g_sc  = params->inputTensors[3].geometry;
    const auto& g_nw  = params->inputTensors[4].geometry;
    const auto& g_cd  = params->outputTensors[0].geometry;
    const auto& g_qs  = params->outputTensors[1].geometry;
    const auto& g_ro  = params->outputTensors[2].geometry;
    if (g_res.dataType != DATA_BF16 || g_br.dataType != DATA_BF16 || g_bi.dataType != DATA_BF16 ||
        g_sc.dataType != DATA_BF16 || g_nw.dataType != DATA_BF16 ||
        g_cd.dataType != DATA_F8_143 || g_qs.dataType != DATA_F32 || g_ro.dataType != DATA_BF16)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t W = g_res.maxSizes[0], R = g_res.maxSizes[1];
    if (g_res.dims != 2 || g_br.dims != 2 || g_bi.dims != 1 || g_sc.dims != 1 || g_nw.dims != 1 ||
        g_cd.dims != 2 || g_qs.dims != 2 || g_ro.dims != 2 ||
        !W || !R || (W % 256) || W > 4096 ||
        g_br.maxSizes[0] != W || g_br.maxSizes[1] != R ||
        g_bi.maxSizes[0] != W || g_sc.maxSizes[0] != W || g_nw.maxSizes[0] != W ||
        g_cd.maxSizes[0] != W || g_cd.maxSizes[1] != R ||
        g_qs.maxSizes[0] != 1 || g_qs.maxSizes[1] != R ||
        g_ro.maxSizes[0] != W || g_ro.maxSizes[1] != R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = R;
    map_2d_row(instance->inputTensorAccessPattern[0], W);
    map_2d_row(instance->inputTensorAccessPattern[1], W);
    map_1d_full(instance->inputTensorAccessPattern[2], W);
    map_1d_full(instance->inputTensorAccessPattern[3], W);
    map_1d_full(instance->inputTensorAccessPattern[4], W);
    map_2d_row(instance->outputTensorAccessPattern[0], W);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);
    map_2d_row(instance->outputTensorAccessPattern[2], W);

    instance->kernel.paramsNr = sizeof(scalar) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &scalar, sizeof(scalar));
    return copy_elf(instance, &_binary_h3_vvae_scale_add_norm_quant_bf16_gaudi2_o_start,
                    &_binary_h3_vvae_scale_add_norm_quant_bf16_gaudi2_o_end);
}

GlueCodeReturn H3VvaeRmsNormQuantBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(AddRmsNormQuantParams))
        return GLUE_FAILED;

    AddRmsNormQuantParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    const auto& g_in = params->inputTensors[0].geometry;
    const auto& g_nw = params->inputTensors[1].geometry;
    const auto& g_cd = params->outputTensors[0].geometry;
    const auto& g_qs = params->outputTensors[1].geometry;
    if (g_in.dataType != DATA_BF16 || g_nw.dataType != DATA_BF16 ||
        g_cd.dataType != DATA_F8_143 || g_qs.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t W = g_in.maxSizes[0], R = g_in.maxSizes[1];
    if (g_in.dims != 2 || g_nw.dims != 1 || g_cd.dims != 2 || g_qs.dims != 2 ||
        !W || !R || (W % 128) || W > 8192 ||
        g_nw.maxSizes[0] != W || g_cd.maxSizes[0] != W || g_cd.maxSizes[1] != R ||
        g_qs.maxSizes[0] != 1 || g_qs.maxSizes[1] != R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = R;
    map_2d_row(instance->inputTensorAccessPattern[0], W);
    map_1d_full(instance->inputTensorAccessPattern[1], W);
    map_2d_row(instance->outputTensorAccessPattern[0], W);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);
    instance->kernel.paramsNr = sizeof(scalar) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &scalar, sizeof(scalar));
    return copy_elf(instance, &_binary_h3_vvae_rmsnorm_quant_bf16_gaudi2_o_start,
                    &_binary_h3_vvae_rmsnorm_quant_bf16_gaudi2_o_end);
}

GlueCodeReturn H3RowQuantBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 1) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern)
        return GLUE_FAILED;

    const auto& g_in = params->inputTensors[0].geometry;
    const auto& g_cd = params->outputTensors[0].geometry;
    const auto& g_qs = params->outputTensors[1].geometry;
    if (g_in.dataType != DATA_BF16 || g_cd.dataType != DATA_F8_143 || g_qs.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t W = g_in.maxSizes[0], R = g_in.maxSizes[1];
    if (g_in.dims != 2 || g_cd.dims != 2 || g_qs.dims != 2 ||
        !W || !R || (W % 128) || W > 8192 ||
        g_cd.maxSizes[0] != W || g_cd.maxSizes[1] != R ||
        g_qs.maxSizes[0] != 1 || g_qs.maxSizes[1] != R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = R;
    map_2d_row(instance->inputTensorAccessPattern[0], W);
    map_2d_row(instance->outputTensorAccessPattern[0], W);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);
    instance->kernel.paramsNr = 0;
    return copy_elf(instance, &_binary_h3_row_quant_bf16_gaudi2_o_start,
                    &_binary_h3_row_quant_bf16_gaudi2_o_end);
}

GlueCodeReturn H3VvaeBiasSiluMulQuantDqBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 4) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 2) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern)
        return GLUE_FAILED;

    const auto& g_raw = params->inputTensors[0].geometry;
    const auto& g_sa  = params->inputTensors[1].geometry;
    const auto& g_swb = params->inputTensors[2].geometry;
    const auto& g_bi  = params->inputTensors[3].geometry;
    const auto& g_cd  = params->outputTensors[0].geometry;
    const auto& g_qs  = params->outputTensors[1].geometry;
    if (g_raw.dataType != DATA_BF16 || g_sa.dataType != DATA_F32 ||
        g_swb.dataType != DATA_BF16 || g_bi.dataType != DATA_BF16 ||
        g_cd.dataType != DATA_F8_143 || g_qs.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t F = g_cd.maxSizes[0], R = g_raw.maxSizes[1];
    if (g_raw.dims != 2 || g_sa.dims != 2 || g_swb.dims != 1 || g_bi.dims != 1 ||
        g_cd.dims != 2 || g_qs.dims != 2 ||
        !F || !R || (F % 128) || F > 136 * 128 ||
        g_raw.maxSizes[0] != 2 * F || g_sa.maxSizes[0] != 1 || g_sa.maxSizes[1] != R ||
        g_swb.maxSizes[0] != 2 * F || g_bi.maxSizes[0] != 2 * F ||
        g_cd.maxSizes[1] != R || g_qs.maxSizes[0] != 1 || g_qs.maxSizes[1] != R)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = R;
    map_2d_row(instance->inputTensorAccessPattern[0], 2 * F);
    map_2d_row(instance->inputTensorAccessPattern[1], 1);
    map_1d_full(instance->inputTensorAccessPattern[2], 2 * F);
    map_1d_full(instance->inputTensorAccessPattern[3], 2 * F);
    map_2d_row(instance->outputTensorAccessPattern[0], F);
    map_2d_row(instance->outputTensorAccessPattern[1], 1);
    instance->kernel.paramsNr = 0;
    return copy_elf(instance, &_binary_h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2_o_start,
                    &_binary_h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2_o_end);
}

GlueCodeReturn H3QkNormRopePrepBf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 7) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 4) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    struct PrepParams { float eps; int32_t valid_u; };
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(PrepParams))
        return GLUE_FAILED;

    PrepParams sp{};
    std::memcpy(&sp, params->nodeParams.nodeParams, sizeof(sp));
    const auto& g_qkv = params->inputTensors[0].geometry;
    const auto& g_rs  = params->inputTensors[1].geometry;
    const auto& g_cs  = params->inputTensors[2].geometry;
    const auto& g_qn  = params->inputTensors[3].geometry;
    const auto& g_kn  = params->inputTensors[4].geometry;
    const auto& g_cos = params->inputTensors[5].geometry;
    const auto& g_sin = params->inputTensors[6].geometry;
    const auto& g_qp  = params->outputTensors[0].geometry;
    const auto& g_kp  = params->outputTensors[1].geometry;
    const auto& g_vi  = params->outputTensors[2].geometry;
    const auto& g_mx  = params->outputTensors[3].geometry;

    if (g_qkv.dataType != DATA_BF16 || g_rs.dataType != DATA_F32 || g_cs.dataType != DATA_BF16 ||
        g_qn.dataType != DATA_BF16 || g_kn.dataType != DATA_BF16 ||
        g_cos.dataType != DATA_BF16 || g_sin.dataType != DATA_BF16 ||
        g_qp.dataType != DATA_BF16 || g_kp.dataType != DATA_BF16 || g_vi.dataType != DATA_BF16 ||
        g_mx.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;

    if (g_qkv.dims != 2 || g_rs.dims != 2 || g_cs.dims != 1 || g_qn.dims != 1 || g_kn.dims != 1 ||
        g_cos.dims != 2 || g_sin.dims != 2 || g_qp.dims != 3 || g_kp.dims != 3 || g_vi.dims != 3 || g_mx.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    const uint64_t S = g_qkv.maxSizes[1];
    const uint64_t U_pad = g_qp.maxSizes[1];
    const uint64_t K_pad = g_kp.maxSizes[1];
    const uint64_t NH = g_qp.maxSizes[2];
    const uint64_t nblk = U_pad / 256;
    if (!NH || (NH & 1) || !U_pad || (U_pad % 256) || K_pad != U_pad ||
        sp.valid_u <= 0 || uint64_t(sp.valid_u) > S || uint64_t(sp.valid_u) > U_pad ||
        g_qp.maxSizes[0] != 128 || g_qkv.maxSizes[0] != 3 * NH * 128 ||
        g_rs.maxSizes[0] != 1 || g_rs.maxSizes[1] != S || g_cs.maxSizes[0] != 3 * NH * 128 ||
        g_qn.maxSizes[0] != 128 || g_kn.maxSizes[0] != 128 ||
        g_cos.maxSizes[0] != 128 || g_cos.maxSizes[1] != S ||
        g_sin.maxSizes[0] != 128 || g_sin.maxSizes[1] != S ||
        g_kp.maxSizes[0] != 128 || g_kp.maxSizes[2] != NH ||
        g_vi.maxSizes[0] != 128 || g_vi.maxSizes[1] != U_pad || g_vi.maxSizes[2] != NH ||
        g_mx.maxSizes[0] != 3 || g_mx.maxSizes[1] != nblk)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = nblk;
    map_2d_table(instance->inputTensorAccessPattern[0], 3 * NH * 128, S);
    map_2d_table(instance->inputTensorAccessPattern[1], 1, S);
    map_1d_full(instance->inputTensorAccessPattern[2], 3 * NH * 128);
    map_1d_full(instance->inputTensorAccessPattern[3], 128);
    map_1d_full(instance->inputTensorAccessPattern[4], 128);
    map_2d_table(instance->inputTensorAccessPattern[5], 128, S);
    map_2d_table(instance->inputTensorAccessPattern[6], 128, S);

    auto map_3d_blk = [&](TensorAccessPattern& p) {
        std::memset(&p, 0, sizeof(p));
        p.mapping[0].end_b = 127;
        p.mapping[1].indexSpaceDim = 0;
        p.mapping[1].a = 256;
        p.mapping[1].end_b = 255;
        p.mapping[2].end_b = static_cast<int32_t>(NH) - 1;
    };
    map_3d_blk(instance->outputTensorAccessPattern[0]);
    map_3d_blk(instance->outputTensorAccessPattern[1]);
    map_3d_blk(instance->outputTensorAccessPattern[2]);
    map_2d_row(instance->outputTensorAccessPattern[3], 3);

    instance->kernel.paramsNr = 2;
    std::memcpy(instance->kernel.scalarParams, &sp, sizeof(sp));
    return copy_elf(instance, &_binary_h3_qk_norm_rope_prep_bf16_gaudi2_o_start,
                    &_binary_h3_qk_norm_rope_prep_bf16_gaudi2_o_end);
}

GlueCodeReturn H3QuantV256Bf16Gaudi2::GetGcDefinitions(
    HabanaKernelParams* params, HabanaKernelInstantiation* instance) {
    if (!params || !instance) return GLUE_FAILED;
    if (params->inputTensorNr != 2) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (params->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern)
        return GLUE_FAILED;

    const auto& g_v = params->inputTensors[0].geometry;
    const auto& g_s = params->inputTensors[1].geometry;
    const auto& g_o = params->outputTensors[0].geometry;
    if (g_v.dataType != DATA_BF16 || g_s.dataType != DATA_F32 || g_o.dataType != DATA_F8_143)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (g_v.dims != 3 || g_s.dims != 1 || g_o.dims != 3)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t K = g_v.maxSizes[1], NH = g_v.maxSizes[2];
    if (!K || (K % 4) || !NH || g_v.maxSizes[0] != 128 ||
        g_s.maxSizes[0] != 1 ||
        g_o.maxSizes[0] != 256 || g_o.maxSizes[1] != K || g_o.maxSizes[2] != NH)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;

    instance->indexSpaceRank = 2;
    instance->indexSpaceGeometry[0] = K / 4;
    instance->indexSpaceGeometry[1] = NH;

    {
        auto& p = instance->inputTensorAccessPattern[0];
        std::memset(&p, 0, sizeof(p));
        p.mapping[0].end_b = 127;
        p.mapping[1].indexSpaceDim = 0;
        p.mapping[1].a = 4;
        p.mapping[1].end_b = 3;
        p.mapping[2].indexSpaceDim = 1;
        p.mapping[2].a = 1;
    }
    map_1d_full(instance->inputTensorAccessPattern[1], 1);
    {
        auto& p = instance->outputTensorAccessPattern[0];
        std::memset(&p, 0, sizeof(p));
        p.mapping[0].end_b = 255;
        p.mapping[1].indexSpaceDim = 0;
        p.mapping[1].a = 4;
        p.mapping[1].end_b = 3;
        p.mapping[2].indexSpaceDim = 1;
        p.mapping[2].a = 1;
    }

    instance->kernel.paramsNr = 0;
    return copy_elf(instance, &_binary_h3_quant_v256_bf16_gaudi2_o_start,
                    &_binary_h3_quant_v256_bf16_gaudi2_o_end);
}
