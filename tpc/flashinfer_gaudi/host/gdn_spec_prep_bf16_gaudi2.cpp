// SPDX-License-Identifier: Apache-2.0
// Glue for kernels/gdn_spec_prep_bf16_gaudi2.c (conv + layout for the WY verification core) and for the output-only
// WY core kernels/gdn_mtp_wyo_f32_gaudi2.c.
#include "gdn_spec_prep_bf16_gaudi2.hpp"

#include <cstring>
#include <initializer_list>

extern unsigned char _binary_gdn_spec_prep_bf16_gaudi2_o_start;
extern unsigned char _binary_gdn_spec_prep_bf16_gaudi2_o_end;
extern unsigned char _binary_gdn_mtp_wyo_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wyo_f32_gaudi2_o_end;
extern unsigned char _binary_gdn_mtp_wyo8_f32_gaudi2_o_start;
extern unsigned char _binary_gdn_mtp_wyo8_f32_gaudi2_o_end;

namespace {
using namespace tpc_lib_api;
bool Is(const Tensor& t, TensorDataType type, std::initializer_list<uint64_t> sizes)
{
    if (t.geometry.dataType != type || t.geometry.dims != sizes.size()) return false;
    unsigned d = 0;
    for (uint64_t s : sizes)
        if (t.geometry.maxSizes[d++] != s) return false;
    return true;
}
void Map(TensorAccessPattern& pattern, unsigned dim, unsigned indexDim, int a, int first, int last)
{
    pattern.mapping[dim].indexSpaceDim = indexDim;
    pattern.mapping[dim].a = a;
    pattern.mapping[dim].start_b = first;
    pattern.mapping[dim].end_b = last;
}
GlueCodeReturn Elf(HabanaKernelInstantiation* out, unsigned char* begin, unsigned char* end)
{
    const unsigned size = static_cast<unsigned>(end - begin);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, size);
    return GLUE_SUCCESS;
}
}  // namespace

GlueCodeReturn GdnSpecPrepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    if (in->inputTensorNr != 2) { in->inputTensorNr = 2; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 2) { in->outputTensorNr = 2; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto& seq = in->inputTensors[0].geometry;
    const uint64_t C = seq.maxSizes[0], rows = seq.maxSizes[1];
    if (seq.dataType != DATA_BF16 || seq.dims != 2 || C % 128 || C <= 4096 || rows < 4)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t T = rows - 3;
    if (!Is(in->inputTensors[1], DATA_BF16, {C, 4})) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 2 * T, 16}) || !Is(in->outputTensors[1], DATA_F32, {C - 4096, T}))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 1;
    out->indexSpaceGeometry[0] = C / 128;
    for (unsigned i = 0; i < 2; ++i) {
        out->inputTensorAccessPattern[i].allRequired = true;
        out->outputTensorAccessPattern[i].allRequired = true;
    }
    out->kernel.paramsNr = 0;
    return Elf(out, &_binary_gdn_spec_prep_bf16_gaudi2_o_start, &_binary_gdn_spec_prep_bf16_gaudi2_o_end);
}

// Inputs: 0 log_decay F32 [48, T, B]; 1 beta F32 [48, T, B]; 2 rmat F32 [384, 2T, 16, B] (key head h: rows t:
// S0_v k_t, rows T + t: S0_v q_t for its 3 value heads side by side); 3 gram F32 [2T, 2T, 16, B]; 4 vals F32
// [128, 48, T, B]. Output: F32 [128, 48, T, B]. T = 16 (wyo) or 8 (wyo8).
static GlueCodeReturn WyoDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out, uint64_t T,
                                     unsigned char* elf, unsigned char* elf_end)
{
    if (in->inputTensorNr != 5) { in->inputTensorNr = 5; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != 1) { in->outputTensorNr = 1; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    const auto* t = in->inputTensors;
    const uint64_t B = t[0].geometry.maxSizes[2];
    if (B == 0 || B > 16 || !Is(t[0], DATA_F32, {48, T, B}) || !Is(t[1], DATA_F32, {48, T, B}) ||
        !Is(t[2], DATA_F32, {384, 2 * T, 16, B}) || !Is(t[3], DATA_F32, {2 * T, 2 * T, 16, B}) ||
        !Is(t[4], DATA_F32, {128, 48, T, B}))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!Is(in->outputTensors[0], DATA_F32, {128, 48, T, B})) return GLUE_INCOMPATIBLE_OUTPUT_SIZE;
    out->indexSpaceRank = 4;
    out->indexSpaceGeometry[0] = 1;
    out->indexSpaceGeometry[1] = 1;
    out->indexSpaceGeometry[2] = 48;
    out->indexSpaceGeometry[3] = B;
    for (unsigned i = 0; i < 5; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    Map(out->outputTensorAccessPattern[0], 0, 0, 0, 0, 127);
    Map(out->outputTensorAccessPattern[0], 1, 2, 1, 0, 0);
    Map(out->outputTensorAccessPattern[0], 2, 0, 0, 0, static_cast<int>(T) - 1);
    Map(out->outputTensorAccessPattern[0], 3, 3, 1, 0, 0);
    out->kernel.paramsNr = 0;
    return Elf(out, elf, elf_end);
}

GlueCodeReturn GdnMtpWyoF32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return WyoDefinitions(in, out, 16, &_binary_gdn_mtp_wyo_f32_gaudi2_o_start, &_binary_gdn_mtp_wyo_f32_gaudi2_o_end);
}

GlueCodeReturn GdnMtpWyo8F32Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out)
{
    return WyoDefinitions(in, out, 8, &_binary_gdn_mtp_wyo8_f32_gaudi2_o_start, &_binary_gdn_mtp_wyo8_f32_gaudi2_o_end);
}
