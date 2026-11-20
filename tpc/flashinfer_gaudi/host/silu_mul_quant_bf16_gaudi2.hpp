// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

// with the gate_up FP8 GEMM's dequantisation folded in (kernels/silu_mul_quant_dq_bf16_gaudi2.c)
class SiluMulQuantDqBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_silu_mul_quant_dq_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class SiluMulQuantBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_silu_mul_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};
