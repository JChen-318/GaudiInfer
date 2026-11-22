// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
#include "../norm_quant_params.h"

class H3AdalnRmsNormQuantBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_adaln_rmsnorm_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3GateAdalnRmsNormQuantBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_gate_adaln_rmsnorm_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3GateResidualBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_gate_residual_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3QkNormRopeBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_qk_norm_rope_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3MaxFreeExpBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_maxfree_exp_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3FastSoftmaxFp8Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_fast_softmax_fp8_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3SiluMulQuantDualBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_silu_mul_quant_dual_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3VvaeQkNormRopeBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_vvae_qk_norm_rope_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3VvaeScaleAddNormQuantBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_vvae_scale_add_norm_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3VvaeRmsNormQuantBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_vvae_rmsnorm_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3RowQuantBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_row_quant_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3VvaeBiasSiluMulQuantDqBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3QkNormRopePrepBf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_qk_norm_rope_prep_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};

class H3QuantV256Bf16Gaudi2 {
public:
    static constexpr const char* name = "flashinfer_gaudi_h3_quant_v256_bf16_gaudi2";
    tpc_lib_api::GlueCodeReturn GetGcDefinitions(
        tpc_lib_api::HabanaKernelParams* params,
        tpc_lib_api::HabanaKernelInstantiation* instance);
};
