// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <cstring>
#include <limits>
#include "add_rmsnorm_quant_bf16_gaudi2.hpp"

extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_small_o_start;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_small_o_end;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_no_norm_o_start;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_no_norm_o_end;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_dq_o_start;
extern unsigned char _binary_add_rmsnorm_quant_bf16_gaudi2_dq_o_end;

tpc_lib_api::GlueCodeReturn AddRmsNormQuantBf16Gaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance) {
    using namespace tpc_lib_api;
    if (!params || !instance) return GLUE_FAILED;
    const unsigned nin = params->inputTensorNr;
    const unsigned nout = params->outputTensorNr;
    if (nin != 3 && nin != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (nout != 4 && nout != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (nin == 5 && nout != 3) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (!params->inputTensors || !params->outputTensors || !instance->inputTensorAccessPattern ||
        !instance->outputTensorAccessPattern || !params->nodeParams.nodeParams ||
        params->nodeParams.nodeParamsSize != sizeof(AddRmsNormQuantParams)) return GLUE_FAILED;
    AddRmsNormQuantParams scalar{};
    std::memcpy(&scalar, params->nodeParams.nodeParams, sizeof(scalar));
    const auto& input = params->inputTensors[0].geometry;
    const auto width = input.maxSizes[0], rows = input.maxSizes[1];
    if (input.dims != 2 || !rows || rows > std::numeric_limits<int32_t>::max() ||
        width < 256 || width > (nout == 3 ? 8192u : 17408u) || width % 128 ||
        !std::isnormal(scalar.epsilon) || scalar.epsilon <= 0 ||
        scalar.inverse_width != 1.0f / static_cast<float>(width) ||
        (scalar.inverse_range != 1.0f / 240.0f && scalar.inverse_range != 0.004180908203125f))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (nin == 3) {
        for (int index = 0; index < 3; ++index) {
            const auto& geometry = params->inputTensors[index].geometry;
            if (geometry.dataType != DATA_BF16) return GLUE_INCOMPATIBLE_DATA_TYPE;
            if (geometry.dims != (index == 2 ? 1u : 2u) || geometry.maxSizes[0] != width ||
                (index != 2 && geometry.maxSizes[1] != rows)) return GLUE_INCOMPATIBLE_INPUT_SIZE;
        }
    } else {
        // 5 inputs: 0 raw [W, M] bf16, 1 sa [1, M] f32, 2 swb [W] bf16, 3 residual [W, M] bf16, 4 weight [W] bf16
        const auto& g0 = params->inputTensors[0].geometry;
        const auto& g1 = params->inputTensors[1].geometry;
        const auto& g2 = params->inputTensors[2].geometry;
        const auto& g3 = params->inputTensors[3].geometry;
        const auto& g4 = params->inputTensors[4].geometry;
        if (g0.dataType != DATA_BF16 || g1.dataType != DATA_F32 || g2.dataType != DATA_BF16 ||
            g3.dataType != DATA_BF16 || g4.dataType != DATA_BF16)
            return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (g1.dims != 2 || g1.maxSizes[0] != 1 || g1.maxSizes[1] != rows ||
            g2.dims != 1 || g2.maxSizes[0] != width ||
            g3.dims != 2 || g3.maxSizes[0] != width || g3.maxSizes[1] != rows ||
            g4.dims != 1 || g4.maxSizes[0] != width)
            return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    for (unsigned index = 0; index < nout; ++index) {
        const auto& geometry = params->outputTensors[index].geometry;
        const auto dtype = index == 0 ? DATA_F8_143 : index == 1 ? DATA_F32 : DATA_BF16;
        if (geometry.dataType != dtype) return GLUE_INCOMPATIBLE_DATA_TYPE;
        if (geometry.dims != 2 || geometry.maxSizes[0] != (index == 1 ? 1 : width) ||
            geometry.maxSizes[1] != rows) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    }
    instance->indexSpaceRank = 1;
    instance->indexSpaceGeometry[0] = rows;
    auto map = [](TensorAccessPattern& pattern, int columns, bool row) {
        std::memset(&pattern, 0, sizeof(pattern));
        pattern.mapping[0].indexSpaceDim = 0;
        pattern.mapping[0].end_b = columns - 1;
        if (row) {
            pattern.mapping[1].indexSpaceDim = 0;
            pattern.mapping[1].a = 1;
        }
    };
    if (nin == 3) {
        for (int index = 0; index < 3; ++index) map(instance->inputTensorAccessPattern[index], width, index != 2);
    } else {
        map(instance->inputTensorAccessPattern[0], width, true);
        map(instance->inputTensorAccessPattern[1], 1, true);
        map(instance->inputTensorAccessPattern[2], width, false);
        map(instance->inputTensorAccessPattern[3], width, true);
        map(instance->inputTensorAccessPattern[4], width, false);
    }
    for (unsigned index = 0; index < nout; ++index)
        map(instance->outputTensorAccessPattern[index], index == 1 ? 1 : width, true);
    instance->kernel.paramsNr = sizeof(scalar) / sizeof(uint32_t);
    std::memcpy(instance->kernel.scalarParams, &scalar, sizeof(scalar));
    const unsigned char* begin = nullptr;
    const unsigned char* end = nullptr;
    if (nin == 5) {
        begin = &_binary_add_rmsnorm_quant_bf16_gaudi2_dq_o_start;
        end = &_binary_add_rmsnorm_quant_bf16_gaudi2_dq_o_end;
    } else if (nout == 3) {
        begin = &_binary_add_rmsnorm_quant_bf16_gaudi2_no_norm_o_start;
        end = &_binary_add_rmsnorm_quant_bf16_gaudi2_no_norm_o_end;
    } else if (width <= 8192) {
        begin = &_binary_add_rmsnorm_quant_bf16_gaudi2_small_o_start;
        end = &_binary_add_rmsnorm_quant_bf16_gaudi2_small_o_end;
    } else {
        begin = &_binary_add_rmsnorm_quant_bf16_gaudi2_o_start;
        end = &_binary_add_rmsnorm_quant_bf16_gaudi2_o_end;
    }
    const unsigned size = end - begin;
    const unsigned capacity = instance->kernel.elfSize;
    instance->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    if (!instance->kernel.kernelElf) return GLUE_FAILED;
    std::memcpy(instance->kernel.kernelElf, begin, size);
    return GLUE_SUCCESS;
}
