// SPDX-License-Identifier: Apache-2.0
// Glue for kernels/masked_softmax_bf16_gaudi2.c (masked, unnormalised softmax of attention scores).
// Inputs: 0 s BF16 [S, R]; 1 last I32 [T] (R % T == 0). Outputs: 0 p BF16 [S, R] or [C, R, S/C]; 1 sum F32 [1, R].
#include "masked_softmax_bf16_gaudi2.hpp"

#include <cstring>

extern unsigned char _binary_masked_softmax_bf16_gaudi2_o_start;
extern unsigned char _binary_masked_softmax_bf16_gaudi2_o_end;
extern unsigned char _binary_masked_softmax_f16_gaudi2_o_start;
extern unsigned char _binary_masked_softmax_f16_gaudi2_o_end;
extern unsigned char _binary_masked_softmax_bf16_f8_gaudi2_o_start;
extern unsigned char _binary_masked_softmax_bf16_f8_gaudi2_o_end;

namespace {
tpc_lib_api::GlueCodeReturn masked_softmax_glue(tpc_lib_api::HabanaKernelParams* in, tpc_lib_api::HabanaKernelInstantiation* out,
                                               tpc_lib_api::TensorDataType score_type, unsigned char* elf_start,
                                               unsigned char* elf_end, bool maxes = false,
                                               tpc_lib_api::TensorDataType p_type = tpc_lib_api::DATA_BF16)
{
    using namespace tpc_lib_api;
    const unsigned nout = maxes ? 3 : 2;
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != nout) { in->outputTensorNr = nout; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& s = in->inputTensors[0].geometry;
    const auto& last = in->inputTensors[1].geometry;
    const auto& p = in->outputTensors[0].geometry;
    const auto& sum = in->outputTensors[1].geometry;
    if (s.dataType != score_type || last.dataType != DATA_I32 || p.dataType != p_type || sum.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t S = s.maxSizes[0], R = s.maxSizes[1], T = last.maxSizes[0];
    if (s.dims != 2 || S % 128 || !R || last.dims != 1 || !T || R % T) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    // p [S, R], or chunked [C, R, S/C] (C % 512 == 0) for a batched PV
    const bool chunked = p.dims == 3;
    const uint64_t C = p.maxSizes[0];
    const bool p_ok = chunked ? (C % 512 == 0 && S % C == 0 && p.maxSizes[1] == R && p.maxSizes[2] == S / C)
                              : (p.dims == 2 && C == S && p.maxSizes[1] == R);
    if (!p_ok || sum.dims != 2 || sum.maxSizes[0] != 1 || sum.maxSizes[1] != R) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    if (maxes) {
        const auto& mx = in->outputTensors[2].geometry;
        if (mx.dataType != DATA_F32 || mx.dims != 2 || mx.maxSizes[0] != 1 || mx.maxSizes[1] != R) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    }
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = R;
    auto row = [](TensorAccessPattern& a, int cols) {
        std::memset(&a, 0, sizeof(a));
        a.mapping[0].indexSpaceDim = 0;
        a.mapping[0].a = 0;
        a.mapping[0].end_b = cols - 1;
        a.mapping[1].indexSpaceDim = 0;
        a.mapping[1].a = 1;
    };
    row(out->inputTensorAccessPattern[0], int(S));
    out->inputTensorAccessPattern[1].allRequired = true;
    row(out->outputTensorAccessPattern[0], int(C));
    if (chunked) {  // every chunk of the row
        out->outputTensorAccessPattern[0].mapping[2].indexSpaceDim = 0;
        out->outputTensorAccessPattern[0].mapping[2].a = 0;
        out->outputTensorAccessPattern[0].mapping[2].end_b = int(S / C) - 1;
    }
    row(out->outputTensorAccessPattern[1], 1);
    if (maxes) row(out->outputTensorAccessPattern[2], 1);
    // optional node parameter: the softmax scale (float, > 0; default 1)
    float scale = 1.0f;
    if (in->nodeParams.nodeParams && in->nodeParams.nodeParamsSize == sizeof(float))
        std::memcpy(&scale, in->nodeParams.nodeParams, sizeof(float));
    if (!(scale > 0.0f)) return GLUE_FAILED;
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &scale, sizeof(float));
    const unsigned size = unsigned(elf_end - elf_start);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, elf_start, size);
    return GLUE_SUCCESS;
}
}  // namespace

tpc_lib_api::GlueCodeReturn MaskedSoftmaxBf16Gaudi2::GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                                      tpc_lib_api::HabanaKernelInstantiation* out)
{
    return masked_softmax_glue(in, out, tpc_lib_api::DATA_BF16, &_binary_masked_softmax_bf16_gaudi2_o_start,
                               &_binary_masked_softmax_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn MaskedSoftmaxF16Gaudi2::GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                                     tpc_lib_api::HabanaKernelInstantiation* out)
{
    return masked_softmax_glue(in, out, tpc_lib_api::DATA_F16, &_binary_masked_softmax_f16_gaudi2_o_start,
                               &_binary_masked_softmax_f16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn MaskedSoftmaxBf16F8Gaudi2::GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                                        tpc_lib_api::HabanaKernelInstantiation* out)
{
    return masked_softmax_glue(in, out, tpc_lib_api::DATA_BF16, &_binary_masked_softmax_bf16_f8_gaudi2_o_start,
                               &_binary_masked_softmax_bf16_f8_gaudi2_o_end, false, tpc_lib_api::DATA_F8_143);
}
