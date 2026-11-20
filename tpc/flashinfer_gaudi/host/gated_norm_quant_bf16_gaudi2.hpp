// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

struct GatedNormQuantBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gated_norm_quant_bf16_gaudi2";
    struct Params {
        float eps;
    };
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct RowQuantBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_row_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};
