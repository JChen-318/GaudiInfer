// SPDX-License-Identifier: Apache-2.0
// Glue for kernels/ink_*.c (Inkling-2). The graph builder (src/inkling2.cpp) creates every output with its shape;
// the glue checks the tensor counts and data types, sets the index space and copies the scalar parameters.
#include <cstring>
#include "inkling_gaudi2.hpp"

#define INK_ELF(K) extern unsigned char _binary_##K##_o_start; extern unsigned char _binary_##K##_o_end;
INK_ELF(ink_conv_bf16_gaudi2)
INK_ELF(ink_conv_dyn_bf16_gaudi2)
INK_ELF(ink_qprep_bf16_gaudi2)
INK_ELF(ink_qprep_tau_bf16_gaudi2)
INK_ELF(ink_kv_commit_bf16_gaudi2)
INK_ELF(ink_mix_gaudi2)
INK_ELF(ink_mix_f32p_gaudi2)
INK_ELF(ink_route_gaudi2)
INK_ELF(ink_mg_gaudi2)
INK_ELF(ink_mg_out_gaudi2)
INK_ELF(ink_mg_out_lut_gaudi2)
INK_ELF(ink_attn_sm16_gaudi2)
INK_ELF(ink_attn_sm16f_gaudi2)
INK_ELF(ink_nvf4_deq_bf16_gaudi2)
INK_ELF(ink_mxfp8_deq_bf16_gaudi2)

namespace {
using namespace tpc_lib_api;

struct Spec {
    unsigned inputs, outputs;
    const TensorDataType* in_types;
    const TensorDataType* out_types;
    unsigned params_bytes;  // scalar parameter struct size
    unsigned char* elf_begin;
    unsigned char* elf_end;
};

GlueCodeReturn glue(HabanaKernelParams* in, HabanaKernelInstantiation* out, const Spec& s, unsigned rank,
                    const uint64_t* index_space) {
    if (!in || !out) return GLUE_FAILED;
    if (in->inputTensorNr != s.inputs) { in->inputTensorNr = s.inputs; return GLUE_INCOMPATIBLE_INPUT_COUNT; }
    if (in->outputTensorNr != s.outputs) { in->outputTensorNr = s.outputs; return GLUE_INCOMPATIBLE_OUTPUT_COUNT; }
    for (unsigned i = 0; i < s.inputs; ++i)
        if (in->inputTensors[i].geometry.dataType != s.in_types[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
    for (unsigned i = 0; i < s.outputs; ++i)
        if (in->outputTensors[i].geometry.dataType != s.out_types[i]) return GLUE_INCOMPATIBLE_DATA_TYPE;
    if (s.params_bytes) {
        if (!in->nodeParams.nodeParams || in->nodeParams.nodeParamsSize != s.params_bytes) return GLUE_FAILED;
        std::memcpy(out->kernel.scalarParams, in->nodeParams.nodeParams, s.params_bytes);
    }
    out->kernel.paramsNr = s.params_bytes / 4;
    out->indexSpaceRank = rank;
    for (unsigned d = 0; d < rank; ++d) out->indexSpaceGeometry[d] = index_space[d];
    for (unsigned i = 0; i < s.inputs; ++i) out->inputTensorAccessPattern[i].allRequired = true;
    for (unsigned i = 0; i < s.outputs; ++i) out->outputTensorAccessPattern[i].allRequired = true;
    const unsigned size = unsigned(s.elf_end - s.elf_begin);
    const unsigned capacity = out->kernel.elfSize;
    out->kernel.elfSize = size;
    if (capacity < size) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, s.elf_begin, size);
    return GLUE_SUCCESS;
}

constexpr TensorDataType B = DATA_BF16, F = DATA_F32;
}  // namespace

// x [C, T], state [C, 3], w [C, 4] (, dyn f32 [KV*4, T]) -> y [C, T], state_out [C, 3]; params scale, thr, eps, mode
GlueCodeReturn InkConvBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {B, B, B}, ot[] = {B, B};
    const Spec s{3, 2, it, ot, 16, &_binary_ink_conv_bf16_gaudi2_o_start, &_binary_ink_conv_bf16_gaudi2_o_end};
    if (!in || in->inputTensorNr < 1) return GLUE_FAILED;
    const uint64_t T = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t idx[2] = {in->inputTensors[0].geometry.maxSizes[0] / 128, T < 24 ? T : 24};  // INK_CONV_NTC
    return glue(in, out, s, 2, idx);
}
GlueCodeReturn InkConvDynBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {B, B, B, F}, ot[] = {B, B};
    const Spec s{4, 2, it, ot, 16, &_binary_ink_conv_dyn_bf16_gaudi2_o_start, &_binary_ink_conv_dyn_bf16_gaudi2_o_end};
    if (!in || in->inputTensorNr < 1) return GLUE_FAILED;
    const uint64_t T = in->inputTensors[0].geometry.maxSizes[1];
    const uint64_t idx[2] = {in->inputTensors[0].geometry.maxSizes[0] / 128, T < 24 ? T : 24};  // INK_CONV_NTC
    return glue(in, out, s, 2, idx);
}
// qside [QS, T], scale [128, NH] (, tau f32 [T]) -> q [128, NH, T]; params thr
GlueCodeReturn InkQprepBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {B, B}, ot[] = {B};
    const Spec s{2, 1, it, ot, 4, &_binary_ink_qprep_bf16_gaudi2_o_start, &_binary_ink_qprep_bf16_gaudi2_o_end};
    if (!in || in->inputTensorNr < 2) return GLUE_FAILED;
    const uint64_t idx[1] = {in->inputTensors[1].geometry.maxSizes[1]};
    return glue(in, out, s, 1, idx);
}
GlueCodeReturn InkQprepTauBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {B, B, F}, ot[] = {B};
    const Spec s{3, 1, it, ot, 4, &_binary_ink_qprep_tau_bf16_gaudi2_o_start, &_binary_ink_qprep_tau_bf16_gaudi2_o_end};
    if (!in || in->inputTensorNr < 2) return GLUE_FAILED;
    const uint64_t idx[1] = {in->inputTensors[1].geometry.maxSizes[1]};
    return glue(in, out, s, 1, idx);
}
// kv bf16 [D, T, KV, 2], rows i32 [T] -> kc, vc bf16 [D, R, KV]; index [KV, T]
GlueCodeReturn InkKvCommitBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {B, DATA_I32}, ot[] = {B, B};
    const Spec s{2, 2, it, ot, 0, &_binary_ink_kv_commit_bf16_gaudi2_o_start, &_binary_ink_kv_commit_bf16_gaudi2_o_end};
    if (!in || in->inputTensorNr < 1) return GLUE_FAILED;
    const auto& kv = in->inputTensors[0].geometry;
    const uint64_t idx[2] = {kv.maxSizes[2], kv.maxSizes[1]};
    return glue(in, out, s, 2, idx);
}
// blocks bf16 [H, T, S0], part bf16|f32 [H, T], query f32 [H, Q], w bf16 [H, Q] -> normed bf16 [H, Q, T], mix f32 [H, Q, T]
GlueCodeReturn InkMixGaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {B, B, F, B}, ot[] = {B, F};
    const Spec s{4, 2, it, ot, 12, &_binary_ink_mix_gaudi2_o_start, &_binary_ink_mix_gaudi2_o_end};
    const uint64_t idx[1] = {24};  // one point per core (rows split in the kernel)
    return glue(in, out, s, 1, idx);
}
GlueCodeReturn InkMixF32pGaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {B, F, F, B}, ot[] = {B, F};
    const Spec s{4, 2, it, ot, 12, &_binary_ink_mix_f32p_gaudi2_o_start, &_binary_ink_mix_f32p_gaudi2_o_end};
    const uint64_t idx[1] = {24};  // one point per core (rows split in the kernel)
    return glue(in, out, s, 1, idx);
}
// logits f32 [128, T], bias f32 [128], iota f32 [128] -> idx i32 [8, T], w f32 [8, T]; index [T]; params route_scale
GlueCodeReturn InkRouteGaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {F, F, F}, ot[] = {DATA_I32, F};
    const Spec s{3, 2, it, ot, 4, &_binary_ink_route_gaudi2_o_start, &_binary_ink_route_gaudi2_o_end};
    if (!in || in->inputTensorNr < 1) return GLUE_FAILED;
    const uint64_t idx[1] = {in->inputTensors[0].geometry.maxSizes[1]};
    return glue(in, out, s, 1, idx);
}
// h f32 [H, Gh, T], m f32 [H, 4, T] -> cat bf16 [2H, T, G]; index [G, T]; params eps, m0
GlueCodeReturn InkMgGaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {F, F}, ot[] = {B};
    const Spec s{2, 1, it, ot, 8, &_binary_ink_mg_gaudi2_o_start, &_binary_ink_mg_gaudi2_o_end};
    if (!in || in->outputTensorNr < 1) return GLUE_FAILED;
    const auto& o = in->outputTensors[0].geometry;
    const uint64_t idx[2] = {o.maxSizes[2], o.maxSizes[1]};
    return glue(in, out, s, 2, idx);
}
// h, m, fg f32 [2H, T, G], w bf16 [H, G] -> out bf16 [H, T, G]; index [G, T]; params eps, m0
GlueCodeReturn InkMgOutGaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {F, F, F, B}, ot[] = {B};
    const Spec s{4, 1, it, ot, 8, &_binary_ink_mg_out_gaudi2_o_start, &_binary_ink_mg_out_gaudi2_o_end};
    if (!in || in->outputTensorNr < 1) return GLUE_FAILED;
    const auto& o = in->outputTensors[0].geometry;
    const uint64_t idx[2] = {o.maxSizes[2], o.maxSizes[1]};
    return glue(in, out, s, 2, idx);
}
GlueCodeReturn InkMgOutLutGaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {F, F, F, B}, ot[] = {B};
    const Spec s{4, 1, it, ot, 8, &_binary_ink_mg_out_lut_gaudi2_o_start, &_binary_ink_mg_out_lut_gaudi2_o_end};
    if (!in || in->outputTensorNr < 1) return GLUE_FAILED;
    const auto& o = in->outputTensors[0].geometry;
    const uint64_t idx[2] = {o.maxSizes[2], o.maxSizes[1]};
    return glue(in, out, s, 2, idx);
}
// sr f16 [W, GT], sn f16 [T, GT], br f16 [RE, NH, T], sink f32 [NH, T], pos i32 [1] -> p bf16 [SN, GT], sum f32 [1, GT];
// index [GT]; params scale, G, full, kv
GlueCodeReturn InkAttnSm16fGaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {DATA_F16, DATA_F16, DATA_F16, DATA_F16, F, DATA_I32}, ot[] = {B, B, F};
    const Spec s{6, 3, it, ot, 16, &_binary_ink_attn_sm16f_gaudi2_o_start, &_binary_ink_attn_sm16f_gaudi2_o_end};
    if (!in || in->outputTensorNr < 3) return GLUE_FAILED;
    const uint64_t idx[1] = {in->outputTensors[1].geometry.maxSizes[1]};
    return glue(in, out, s, 1, idx);
}
GlueCodeReturn InkAttnSm16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {DATA_F16, DATA_F16, DATA_F16, F, DATA_I32}, ot[] = {B, F};
    const Spec s{5, 2, it, ot, 16, &_binary_ink_attn_sm16_gaudi2_o_start, &_binary_ink_attn_sm16_gaudi2_o_end};
    if (!in || in->outputTensorNr < 2) return GLUE_FAILED;
    const uint64_t idx[1] = {in->outputTensors[0].geometry.maxSizes[1]};
    return glue(in, out, s, 1, idx);
}
// w u8 [cols, K/2, E], bs f8 [cols, K/16, E], mask i32 [E] -> out bf16 [cols, K, E]; index [(cols/256)*E]; param use_mask
GlueCodeReturn InkNvf4DeqBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {DATA_U8, DATA_F8_143, DATA_I32}, ot[] = {B};
    const Spec s{3, 1, it, ot, 4, &_binary_ink_nvf4_deq_bf16_gaudi2_o_start, &_binary_ink_nvf4_deq_bf16_gaudi2_o_end};
    if (!in || in->inputTensorNr < 1) return GLUE_FAILED;
    const auto& w = in->inputTensors[0].geometry;
    if (w.maxSizes[0] % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t idx[1] = {(w.maxSizes[0] / 256) * w.maxSizes[2]};
    return glue(in, out, s, 1, idx);
}
// w u8 [cols, K, E], bs f8 [cols, K/32, E], mask i32 [E] -> out bf16 [cols, K, E]; index [(cols/256)*E]; param use_mask
GlueCodeReturn InkMxfp8DeqBf16Gaudi2::GetGcDefinitions(HabanaKernelParams* in, HabanaKernelInstantiation* out) {
    static const TensorDataType it[] = {DATA_U8, DATA_F8_143, DATA_I32}, ot[] = {B};
    const Spec s{3, 1, it, ot, 4, &_binary_ink_mxfp8_deq_bf16_gaudi2_o_start, &_binary_ink_mxfp8_deq_bf16_gaudi2_o_end};
    if (!in || in->inputTensorNr < 1) return GLUE_FAILED;
    const auto& w = in->inputTensors[0].geometry;
    if (w.maxSizes[0] % 256) return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const uint64_t idx[1] = {(w.maxSizes[0] / 256) * w.maxSizes[2]};
    return glue(in, out, s, 1, idx);
}
