// SPDX-License-Identifier: Apache-2.0
// Glue of the Inkling-2 kernels (kernels/ink_*.c). Every kernel declares its index space from its first input and
// marks all tensors fully accessed by every index point (no graph-compiler slicing).
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

#define INK_KERNEL(CLASS, GUID)                                                                    \
    class CLASS {                                                                                  \
    public:                                                                                        \
        static constexpr const char* name = GUID;                                                  \
        tpc_lib_api::GlueCodeReturn GetGcDefinitions(tpc_lib_api::HabanaKernelParams* params,      \
                                                     tpc_lib_api::HabanaKernelInstantiation* instance); \
    };

INK_KERNEL(InkConvBf16Gaudi2, "ink_conv_bf16_gaudi2")
INK_KERNEL(InkConvDynBf16Gaudi2, "ink_conv_dyn_bf16_gaudi2")
INK_KERNEL(InkQprepBf16Gaudi2, "ink_qprep_bf16_gaudi2")
INK_KERNEL(InkQprepTauBf16Gaudi2, "ink_qprep_tau_bf16_gaudi2")
INK_KERNEL(InkKvCommitBf16Gaudi2, "ink_kv_commit_bf16_gaudi2")
INK_KERNEL(InkMixGaudi2, "ink_mix_gaudi2")
INK_KERNEL(InkMixF32pGaudi2, "ink_mix_f32p_gaudi2")
INK_KERNEL(InkRouteGaudi2, "ink_route_gaudi2")
INK_KERNEL(InkMgGaudi2, "ink_mg_gaudi2")
INK_KERNEL(InkMgOutGaudi2, "ink_mg_out_gaudi2")
INK_KERNEL(InkMgOutLutGaudi2, "ink_mg_out_lut_gaudi2")
INK_KERNEL(InkAttnSm16Gaudi2, "ink_attn_sm16_gaudi2")
INK_KERNEL(InkAttnSm16fGaudi2, "ink_attn_sm16f_gaudi2")
INK_KERNEL(InkNvf4DeqBf16Gaudi2, "ink_nvf4_deq_bf16_gaudi2")
INK_KERNEL(InkMxfp8DeqBf16Gaudi2, "ink_mxfp8_deq_bf16_gaudi2")

#undef INK_KERNEL
