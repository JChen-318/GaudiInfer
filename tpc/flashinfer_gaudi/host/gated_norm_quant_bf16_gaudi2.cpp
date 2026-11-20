// SPDX-License-Identifier: Apache-2.0
// Glues for the GDN output gate in two stages:
//   kernels/gated_norm_quant_bf16_gaudi2.c (per-head RMSNorm x silu(z), group maxima): inputs 0 core BF16 [128, HV, T],
//     1 z BF16 [128, HV, T], 2 w BF16 [128]; outputs 0 gated BF16 [128*HV, T], 1 pmax F32 [64, HV/8, T]; Params {eps}.
//   kernels/row_quant_bf16_gaudi2.c (row FP8): inputs 0 gated BF16 [W, T], 1 pmax F32 [64, NG, T]; outputs 0 codes
//     F8 [W, T], 1 scale F32 [1, T].
#include "gated_norm_quant_bf16_gaudi2.hpp"

#include <cstring>
#include <initializer_list>

extern unsigned char _binary_gated_norm_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_gated_norm_quant_bf16_gaudi2_o_end;
extern unsigned char _binary_row_quant_bf16_gaudi2_o_start;
extern unsigned char _binary_row_quant_bf16_gaudi2_o_end;

namespace {
using namespace tpc_lib_api;

void pattern(TensorAccessPattern& a, std::initializer_list<int> full, int row_dim, int group_dim = -1, int group = 1) {
    // dims listed in `full` are read whole; row_dim follows index-space dim 1 (or 0 when group_dim < 0); group_dim
    // follows index-space dim 0 in units of `group`
    std::memset(&a, 0, sizeof(a));
    int d = 0;
    for (int size : full) {
        a.mapping[d].indexSpaceDim = 0;
        a.mapping[d].a = 0;
        a.mapping[d].end_b = size - 1;
        ++d;
    }
    (void)d;
    if (group_dim >= 0) {
        a.mapping[group_dim].indexSpaceDim = 0;
        a.mapping[group_dim].a = group;
        a.mapping[group_dim].start_b = 0;
        a.mapping[group_dim].end_b = group - 1;
        a.mapping[row_dim].indexSpaceDim = 1;
        a.mapping[row_dim].a = 1;
    } else {
        a.mapping[row_dim].indexSpaceDim = 0;
        a.mapping[row_dim].a = 1;
    }
}

GlueCodeReturn set_elf(HabanaKernelInstantiation* out, unsigned char* start, unsigned char* end) {
    const unsigned size = unsigned(end - start);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, start, size);
    return GLUE_SUCCESS;
}
}  // namespace

tpc_lib_api::GlueCodeReturn GatedNormQuantBf16Gaudi2::GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                                       tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 3) { in->inputTensorNr = 3; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != sizeof(Params)) return GLUE_FAILED;
    Params prm{};
    std::memcpy(&prm, in->nodeParams.nodeParams, sizeof(prm));
    const auto& core = in->inputTensors[0].geometry;
    const auto& z = in->inputTensors[1].geometry;
    const auto& w = in->inputTensors[2].geometry;
    const auto& gated = in->outputTensors[0].geometry;
    const auto& pmax = in->outputTensors[1].geometry;
    if (core.dataType != DATA_BF16 || z.dataType != DATA_BF16 || w.dataType != DATA_BF16 || gated.dataType != DATA_BF16 ||
        pmax.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t D = core.maxSizes[0], HV = core.maxSizes[1], T = core.maxSizes[2], NG = HV / 8;
    if (core.dims != 3 || D != 128 || !HV || HV % 8 || !T || z.dims != 3 || z.maxSizes[0] != D || z.maxSizes[1] != HV ||
        z.maxSizes[2] != T || w.dims != 1 || w.maxSizes[0] != D)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (gated.dims != 2 || gated.maxSizes[0] != D * HV || gated.maxSizes[1] != T || pmax.dims != 3 || pmax.maxSizes[0] != 64 ||
        pmax.maxSizes[1] != NG || pmax.maxSizes[2] != T)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = NG;
    out->indexSpaceGeometry[1] = T;
    pattern(out->inputTensorAccessPattern[0], {int(D)}, 2, 1, 8);
    pattern(out->inputTensorAccessPattern[1], {int(D)}, 2, 1, 8);
    out->inputTensorAccessPattern[2].allRequired = true;
    pattern(out->outputTensorAccessPattern[0], {}, 1, 0, int(8 * D));
    pattern(out->outputTensorAccessPattern[1], {64}, 2, 1, 1);
    out->kernel.paramsNr = 1;
    std::memcpy(out->kernel.scalarParams, &prm.eps, sizeof(float));
    return set_elf(out, &_binary_gated_norm_quant_bf16_gaudi2_o_start, &_binary_gated_norm_quant_bf16_gaudi2_o_end);
}

tpc_lib_api::GlueCodeReturn RowQuantBf16Gaudi2::GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                                 tpc_lib_api::HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& gated = in->inputTensors[0].geometry;
    const auto& pmax = in->inputTensors[1].geometry;
    const auto& codes = in->outputTensors[0].geometry;
    const auto& scale = in->outputTensors[1].geometry;
    if (gated.dataType != DATA_BF16 || pmax.dataType != DATA_F32 || codes.dataType != DATA_F8_143 || scale.dataType != DATA_F32)
        return GLUE_INCOMPATIBLE_DATA_TYPE;
    const uint64_t W = gated.maxSizes[0], T = gated.maxSizes[1];
    if (gated.dims != 2 || W % 128 || !T || pmax.dims != 3 || pmax.maxSizes[0] != 64 || pmax.maxSizes[2] != T)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (codes.dims != 2 || codes.maxSizes[0] != W || codes.maxSizes[1] != T || scale.dims != 2 || scale.maxSizes[0] != 1 ||
        scale.maxSizes[1] != T)
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = T;
    pattern(out->inputTensorAccessPattern[0], {int(W)}, 1);
    pattern(out->inputTensorAccessPattern[1], {64, int(pmax.maxSizes[1])}, 2);
    pattern(out->outputTensorAccessPattern[0], {int(W)}, 1);
    pattern(out->outputTensorAccessPattern[1], {1}, 1);
    out->kernel.paramsNr = 0;
    return set_elf(out, &_binary_row_quant_bf16_gaudi2_o_start, &_binary_row_quant_bf16_gaudi2_o_end);
}
