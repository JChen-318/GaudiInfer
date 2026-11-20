// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

struct GdnSpecPrepBf16Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_spec_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

struct GdnMtpWyoF32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_mtp_wyo_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};

// the same core for T = 8 (batched verification at width 8)
struct GdnMtpWyo8F32Gaudi2 {
    static constexpr const char* name = "flashinfer_gaudi_gdn_mtp_wyo8_f32_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in,
                                                 tpc_lib_api::HabanaKernelInstantiation* out);
};
