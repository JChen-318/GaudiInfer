# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(flashinfer_gaudi_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(TPC_COMPILER tpc-clang REQUIRED)
find_program(ELF_OBJCOPY objcopy REQUIRED)
set(TPC_INCLUDE_DIR "/usr/lib/habanatools/include" CACHE PATH "TPC compiler include directory")
set(HABANA_INCLUDE_DIR "/usr/include/habanalabs" CACHE PATH "Gaudi host API include directory")

set(TPC_EMBEDDED_OBJECTS)
set(WY_EXTRA "" CACHE STRING "extra tpc-clang flags for experiments")
separate_arguments(WY_EXTRA)
foreach(KERNEL
        add_rmsnorm_quant_bf16_gaudi2
        add_rmsnorm_quant_bf16_gaudi2_dq
        add_rmsnorm_quant_bf16_gaudi2_no_norm
        add_rmsnorm_quant_bf16_gaudi2_small
        attn_qkv_prep_bf16_gaudi2
        attn_qkv_prep_f8_gaudi2
        direct_attn_q_pack_bf16_gaudi2
        direct_attn_q_pack_f8_gaudi2
        direct_attn_tile_step_bf16_gaudi2
        direct_attn_tile_step_f8_gaudi2
        direct_conv_commit_bf16_gaudi2
        direct_dq_add_rmsnorm_quant_bf16_gaudi2
        direct_dq_add_rmsnorm_quant_dual_bf16_gaudi2
        direct_dq_attn_qkv_prep_bf16_gaudi2
        direct_dq_attn_qkv_prep_f8_gaudi2
        direct_dq_finalize_bf16_gaudi2
        direct_dq_mul_quant_bf16_gaudi2
        direct_dq_silu_bf16_gaudi2
        direct_embed_norm_quant_bf16_gaudi2
        direct_gdn_ba_siluz_bf16_gaudi2
        direct_gdn_dq_spec_prep_bf16_gaudi2
        direct_gdn_norm_mul_quant_bf16_gaudi2
        direct_kv_scatter_bf16_gaudi2
        direct_kv_scatter_f8_gaudi2
        direct_sigmoid_mul_quant_bf16_gaudi2
        dspine_bridge_codes_bf16_gaudi2
        dspine_bridge_gate_bf16_gaudi2
        dspine_bridge_norm_bf16_gaudi2
        dspine_bridge_post_bf16_gaudi2
        dspine_qkv_prep_bf16_gaudi2
        gated_norm_quant_bf16_gaudi2
        gdn_mtp_wyo8_f32_gaudi2
        gdn_mtp_wyo_f32_gaudi2
        gdn_mtp_wyod8_f32_gaudi2
        gdn_mtp_wyod_f32_gaudi2
        gdn_packed_decode_f32_gaudi2
        gdn_prefill_conv_prep_bf16_gaudi2
        gdn_prefill_decay_prep_bf16_gaudi2
        gdn_prefill_gate_quant_bf16_gaudi2
        gdn_spec_prep_batch_bf16_gaudi2
        gdn_spec_prep_bf16_gaudi2
        gdn_state_commit_f32_gaudi2
        h3_adaln_rmsnorm_quant_bf16_gaudi2
        h3_direct_attn_step_fp8_gaudi2
        h3_fast_softmax_fp8_gaudi2
        h3_gate_adaln_rmsnorm_quant_bf16_gaudi2
        h3_gate_residual_bf16_gaudi2
        h3_maxfree_exp_bf16_gaudi2
        h3_qk_norm_rope_bf16_gaudi2
        h3_qk_norm_rope_prep_bf16_gaudi2
        h3_quant_v256_bf16_gaudi2
        h3_row_quant_bf16_gaudi2
        h3_silu_mul_quant_dual_bf16_gaudi2
        h3_silu_mul_quant_dual_dq_bf16_gaudi2
        h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2
        h3_vvae_qk_norm_rope_bf16_gaudi2
        h3_vvae_rmsnorm_quant_bf16_gaudi2
        h3_vvae_scale_add_norm_quant_bf16_gaudi2
        masked_softmax_bf16_f8_gaudi2
        masked_softmax_bf16_gaudi2
        masked_softmax_f16_gaudi2
        row_quant_bf16_gaudi2
        row_softmax_stats_cmax_f32_gaudi2
        row_softmax_stats_f32_gaudi2
        silu_and_mul_bf16_gaudi2
        silu_mul_quant_bf16_gaudi2
        silu_mul_quant_dq_bf16_gaudi2
        spec_conv_bf16_gaudi2
        ink_conv_bf16_gaudi2
        ink_conv_dyn_bf16_gaudi2
        ink_qprep_bf16_gaudi2
        ink_qprep_tau_bf16_gaudi2
        ink_kv_commit_bf16_gaudi2
        ink_mix_gaudi2
        ink_mix_f32p_gaudi2
        ink_route_gaudi2
        ink_mg_gaudi2
        ink_mg_out_gaudi2
        ink_mg_out_lut_gaudi2
        ink_attn_sm16_gaudi2
        ink_attn_sm16f_gaudi2
        ink_dattn_gaudi2
        ink_dgate_gaudi2
        ink_dpost_gaudi2
        ink_dpost_v_gaudi2
        ink_dpost_vf_gaudi2
        ink_dattn_v_gaudi2
        ink_dattn_vf_gaudi2
        ink_dgate_v_gaudi2
        ink_dgate_vf_gaudi2
        ink_dpost_full_gaudi2
        ink_dattn_far_gaudi2
        ink_dgate_far_gaudi2
        ink_dgram_gaudi2
        ink_dnorm_gaudi2
        ink_dnorm_f32p_gaudi2
        ink_dgram_msg_gaudi2
        ink_dnorm_msg_gaudi2
        ink_dnorm_msg_f32p_gaudi2
        ink_dmid_gaudi2
        ink_rollback_gaudi2
        ink_dmgy_gaudi2
        ink_dmgn_gaudi2
        ink_prprep_gaudi2
        ink_prprep_tau_gaudi2
        ink_pgate_gaudi2
        ink_pgate_far_gaudi2
        ink_conv_hm_bf16_gaudi2
        ink_conv_dyn_hm_bf16_gaudi2
        ink_conv_resa_bf16_gaudi2
        ink_conv_resm_bf16_gaudi2
        ink_pcopy_gaudi2
        ink_attn_sm16b_gaudi2
        ink_attn_sm16fb_gaudi2
        ink_rplan_hist_gaudi2
        ink_pnorm_gaudi2
        ink_mg_out_pade_gaudi2
        ink_conv_hm2_bf16_gaudi2
        ink_conv_dyn_hm2_bf16_gaudi2
        ink_sgather_bf16_gaudi2
        ink_act_bf16_gaudi2
        ink_rplan_dyn_tab_bf16_gaudi2
        ink_rplan_dyn_slots_bf16_gaudi2
        ink_attn_sm16w_gaudi2
        ink_moe_combine_bf16_gaudi2
        ink_droute1_gaudi2
        ink_dtail_rw_gaudi2
        ink_dmg_fgy_gaudi2
        ink_zero_gaudi2
        ink_attn_sm16_near_gaudi2
        ink_attn_far_step_gaudi2
        ink_pfar_head_io_gaudi2
        ink_dembed_gaudi2
        ink_dsoftcap_argmax_gaudi2
        ink_dattn_near_max_gaudi2
        ink_dfar_sm_gaudi2
        ink_dfar_acc_gaudi2
        ink_dgate_f32_gaudi2
        ink_dpost_b_gaudi2
        ink_dpost_b_full_gaudi2
        ink_kv_commit_d1_gaudi2
        ink_dattn_b_gaudi2
        ink_dattn_b_near_max_gaudi2
        ink_dgate_b_gaudi2
        ink_dgate_b_full_gaudi2
        ink_dmid_b_gaudi2
        ink_dtail_rw_b_gaudi2
        ink_dsoftcap_argmax_b_gaudi2
        ink_vmoe_plan_gaudi2
        ink_vmoe_up_nvf4_gaudi2
        ink_vmoe_up_mxf8_gaudi2
        ink_vmoe_dn_nvf4_gaudi2
        ink_vmoe_dn_mxf8_gaudi2
        ink_dtail_rw_dgram_gaudi2
        ink_dtail_rw_dgram_msg_gaudi2
        ink_dmoe_up9_nvf4_gaudi2
        ink_dmoe_up9_mxf8_gaudi2
        ink_dmoe_dn9_nvf4_gaudi2
        ink_dmoe_dn9_mxf8_gaudi2
        ink_dfar_pv_gaudi2
        ink_dfar_acc3_gaudi2
        ink_dgate_far3_gaudi2
        ink_droute1_b_gaudi2
        ink_kv_commit_b4_gaudi2
        ink_nvf4_deq_bf16_gaudi2
        ink_mxfp8_deq_bf16_gaudi2)
    set(TPC_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${KERNEL}.c")
    set(TPC_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}.o")
    set(TPC_EMBEDDED_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/${KERNEL}_embedded.o")
    set(TPC_EXTRA_DEPENDENCIES)
    set(TPC_EXTRA_FLAGS)
    if(KERNEL STREQUAL "ink_mg_out_lut_gaudi2" OR KERNEL STREQUAL "ink_mg_out_gaudi2" OR KERNEL STREQUAL "ink_mg_out_pade_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_mg_gaudi2.c")
    endif()
    if(KERNEL MATCHES "^ink_attn_sm16")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_attn_row16_gaudi2.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_attn_roww_gaudi2.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_attn_sm16_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_attn_sm16f_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "ink_mix_gaudi2" OR KERNEL STREQUAL "ink_mix_f32p_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_mix_row_gaudi2.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_mix_gaudi2.c")
    endif()
    if(KERNEL MATCHES "^add_rmsnorm_quant_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h")
    endif()
    if(KERNEL STREQUAL "gdn_mtp_wy8_f32_gaudi2" OR KERNEL STREQUAL "gdn_mtp_wyo_f32_gaudi2" OR
       KERNEL STREQUAL "gdn_mtp_wyo8_f32_gaudi2" OR KERNEL STREQUAL "gdn_mtp_wyod_f32_gaudi2" OR
       KERNEL STREQUAL "gdn_mtp_wyod8_f32_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_mtp_wy_f32_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "masked_softmax_bf16m_gaudi2" OR KERNEL STREQUAL "masked_softmax_bf16_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/masked_softmax_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "gdn_spec_prep_batch_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gdn_spec_prep_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "attn_qkv_prep_bf16_gaudi2" OR KERNEL STREQUAL "attn_qkv_prep_f8_gaudi2" OR
       KERNEL STREQUAL "direct_embed_norm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "direct_dq_add_rmsnorm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "direct_dq_add_rmsnorm_quant_dual_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_bridge_norm_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_bridge_codes_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_bridge_post_bf16_gaudi2" OR
       KERNEL STREQUAL "dspine_qkv_prep_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h")
    endif()
    if(KERNEL STREQUAL "attn_qkv_prep_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/attn_qkv_prep_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_attn_tile_step_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_attn_tile_step_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_attn_q_pack_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_attn_q_pack_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_dq_attn_qkv_prep_bf16_gaudi2" OR KERNEL STREQUAL "direct_dq_attn_qkv_prep_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h")
    endif()
    if(KERNEL STREQUAL "direct_dq_attn_qkv_prep_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_dq_attn_qkv_prep_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "direct_kv_scatter_f8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/direct_kv_scatter_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "add_rmsnorm_quant_bf16_gaudi2_small" OR
       KERNEL STREQUAL "add_rmsnorm_quant_bf16_gaudi2_no_norm" OR
       KERNEL STREQUAL "add_rmsnorm_quant_bf16_gaudi2_dq")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_rmsnorm_quant_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "h3_adaln_rmsnorm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_gate_adaln_rmsnorm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_qk_norm_rope_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_maxfree_exp_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_fast_softmax_fp8_gaudi2" OR
       KERNEL STREQUAL "h3_direct_attn_step_fp8_gaudi2" OR
       KERNEL STREQUAL "h3_vvae_qk_norm_rope_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_vvae_scale_add_norm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_vvae_rmsnorm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_row_quant_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h")
    endif()
    if(KERNEL STREQUAL "h3_vvae_rmsnorm_quant_bf16_gaudi2" OR
       KERNEL STREQUAL "h3_row_quant_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/h3_vvae_scale_add_norm_quant_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "h3_silu_mul_quant_dual_dq_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/h3_vvae_bias_silu_mul_quant_dq_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "h3_gate_adaln_rmsnorm_quant_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/h3_adaln_rmsnorm_quant_bf16_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "ink_dmoe_up9_nvf4_gaudi2" OR KERNEL STREQUAL "ink_dmoe_up9_mxf8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dmoe_up9_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "ink_dmoe_dn9_nvf4_gaudi2" OR KERNEL STREQUAL "ink_dmoe_dn9_mxf8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dmoe_dn9_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "ink_vmoe_up_nvf4_gaudi2" OR KERNEL STREQUAL "ink_vmoe_up_mxf8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_vmoe_up_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "ink_vmoe_dn_nvf4_gaudi2" OR KERNEL STREQUAL "ink_vmoe_dn_mxf8_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_vmoe_dn_gaudi2.c")
    endif()
    if(KERNEL STREQUAL "ink_mxfp8_deq_bf16_gaudi2")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_nvf4_deq_bf16_gaudi2.c")
    endif()
    if(KERNEL MATCHES "^ink_")
        list(APPEND TPC_EXTRA_DEPENDENCIES "${CMAKE_CURRENT_SOURCE_DIR}/kernels/norm_math_gaudi2.h"
             "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_conv_bf16_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_qprep_bf16_gaudi2.c"
             "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_mix_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_mg_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_math_gaudi2.h" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dmix_gaudi2.h" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dgram_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dnorm_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dpost_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dattn_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dgate_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dpost_v_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dattn_v_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dgate_v_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dpost_b_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dattn_b_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_dgate_b_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_droute1_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_lin_gaudi2.h" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_prprep_gaudi2.c" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ink_pgate_gaudi2.c")
    endif()
    add_custom_command(
        OUTPUT "${TPC_OBJECT}"
        COMMAND "${TPC_COMPILER}" -Wall -Werror -O2 -march=gaudi2 ${TPC_EXTRA_FLAGS} ${WY_EXTRA}
                -I"${TPC_INCLUDE_DIR}" "${TPC_SOURCE}" -c -o "${TPC_OBJECT}"
        DEPENDS "${TPC_SOURCE}" ${TPC_EXTRA_DEPENDENCIES}
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${TPC_EMBEDDED_OBJECT}"
        COMMAND "${ELF_OBJCOPY}" -I binary -O elf64-x86-64 -B i386:x86-64
                "${KERNEL}.o" "${KERNEL}_embedded.o"
        DEPENDS "${TPC_OBJECT}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        VERBATIM
    )
    set_source_files_properties("${TPC_EMBEDDED_OBJECT}" PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    list(APPEND TPC_EMBEDDED_OBJECTS "${TPC_EMBEDDED_OBJECT}")
endforeach()

add_library(
    flashinfer_gaudi_kernels
    SHARED
    host/entry_points.cpp
    host/add_rmsnorm_quant_bf16_gaudi2.cpp
    host/gated_norm_quant_bf16_gaudi2.cpp
    host/gdn_packed_decode_f32_gaudi2.cpp
    host/gdn_spec_prep_bf16_gaudi2.cpp
    host/h3_adaln_bf16_gaudi2.cpp
    host/masked_softmax_bf16_gaudi2.cpp
    host/silu_and_mul_bf16_gaudi2.cpp
    host/silu_mul_quant_bf16_gaudi2.cpp
    host/spec_conv_bf16_gaudi2.cpp
    host/inkling_gaudi2.cpp
    ${TPC_EMBEDDED_OBJECTS}
)
target_include_directories(
    flashinfer_gaudi_kernels
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/host"
    "${TPC_INCLUDE_DIR}"
    "${HABANA_INCLUDE_DIR}"
)
target_compile_options(flashinfer_gaudi_kernels PRIVATE -Wall -Wextra -Werror -fno-strict-aliasing)
target_link_options(flashinfer_gaudi_kernels PRIVATE -Wl,-z,noexecstack)
