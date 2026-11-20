// SPDX-License-Identifier: Apache-2.0
// Kernel database of the GaudiInfer TPC library: the GUIDs the engine launches and their glue.
#include <cstring>

#include "add_rmsnorm_quant_bf16_gaudi2.hpp"
#include "gated_norm_quant_bf16_gaudi2.hpp"
#include "gdn_packed_decode_f32_gaudi2.hpp"
#include "gdn_spec_prep_bf16_gaudi2.hpp"
#include "masked_softmax_bf16_gaudi2.hpp"
#include "silu_mul_quant_bf16_gaudi2.hpp"
#include "spec_conv_bf16_gaudi2.hpp"

namespace {
template <class K>
tpc_lib_api::GlueCodeReturn glue(tpc_lib_api::HabanaKernelParams* params, tpc_lib_api::HabanaKernelInstantiation* instance)
{
    return K{}.GetGcDefinitions(params, instance);
}
template <class K>
void static_name(char* out) { std::strcpy(out, K::name); }
template <class K>
void call_name(char* out) { K{}.GetKernelName(out); }

struct Entry {
    void (*name)(char*);
    tpc_lib_api::GlueCodeReturn (*glue)(tpc_lib_api::HabanaKernelParams*, tpc_lib_api::HabanaKernelInstantiation*);
};

constexpr Entry kKernels[] = {
    {&static_name<AddRmsNormQuantBf16Gaudi2>, &glue<AddRmsNormQuantBf16Gaudi2>},
    {&static_name<GatedNormQuantBf16Gaudi2>, &glue<GatedNormQuantBf16Gaudi2>},
    {&call_name<GdnPackedDecodeF32Gaudi2>, &glue<GdnPackedDecodeF32Gaudi2>},
    {&static_name<GdnMtpWyo8F32Gaudi2>, &glue<GdnMtpWyo8F32Gaudi2>},
    {&static_name<GdnMtpWyoF32Gaudi2>, &glue<GdnMtpWyoF32Gaudi2>},
    {&static_name<GdnSpecPrepBf16Gaudi2>, &glue<GdnSpecPrepBf16Gaudi2>},
    {&static_name<MaskedSoftmaxBf16Gaudi2>, &glue<MaskedSoftmaxBf16Gaudi2>},
    {&static_name<MaskedSoftmaxF16Gaudi2>, &glue<MaskedSoftmaxF16Gaudi2>},
    {&static_name<RowQuantBf16Gaudi2>, &glue<RowQuantBf16Gaudi2>},
    {&static_name<SiluMulQuantBf16Gaudi2>, &glue<SiluMulQuantBf16Gaudi2>},
    {&static_name<SiluMulQuantDqBf16Gaudi2>, &glue<SiluMulQuantDqBf16Gaudi2>},
    {&static_name<SpecConvBf16Gaudi2>, &glue<SpecConvBf16Gaudi2>},
};
constexpr uint32_t kCount = sizeof(kKernels) / sizeof(kKernels[0]);
}  // namespace

extern "C" {
tpc_lib_api::GlueCodeReturn GetKernelGuids(tpc_lib_api::DeviceId deviceId, uint32_t* kernelCount,
                                           tpc_lib_api::GuidInfo* guids)
{
    if (kernelCount == nullptr) return tpc_lib_api::GLUE_FAILED;
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
        *kernelCount = 0;
        return tpc_lib_api::GLUE_SUCCESS;
    }
    const uint32_t capacity = *kernelCount;
    *kernelCount = kCount;
    if (guids == nullptr || capacity == 0) return tpc_lib_api::GLUE_SUCCESS;
    if (capacity < kCount) return tpc_lib_api::GLUE_FAILED;
    std::memset(guids, 0, kCount * sizeof(tpc_lib_api::GuidInfo));
    for (uint32_t i = 0; i < kCount; ++i) kKernels[i].name(guids[i].name);
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn InstantiateTpcKernel(tpc_lib_api::HabanaKernelParams* params,
                                                 tpc_lib_api::HabanaKernelInstantiation* instance)
{
    if (!params || !instance) return tpc_lib_api::GLUE_FAILED;
    char name[tpc_lib_api::MAX_NODE_NAME];
    for (const auto& k : kKernels) {
        k.name(name);
        if (std::strcmp(params->guid.name, name) == 0) return k.glue(params, instance);
    }
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(tpc_lib_api::DeviceId, tpc_lib_api::ShapeInferenceParams*,
                                              tpc_lib_api::ShapeInferenceOutput*)
{
    return tpc_lib_api::GLUE_SUCCESS;
}
}  // extern "C"
