// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

struct MaskedSoftmaxBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_masked_softmax_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

// the same contract with FP16 scores (kernels/masked_softmax_f16_gaudi2.c)
struct MaskedSoftmaxF16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_masked_softmax_f16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};
