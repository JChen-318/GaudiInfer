#include "g2e/direct_gdn.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace g2e {

namespace {

constexpr uint32_t kB = 8;
constexpr uint32_t kT = 16;
constexpr uint32_t kR = kB * kT;  // 128
constexpr uint32_t kH = 5120;
constexpr uint32_t kF = 17408;
constexpr uint32_t kHK = 16;
constexpr uint32_t kHV = 48;
constexpr uint32_t kDK = 128;
constexpr uint32_t kDV = 128;
constexpr uint32_t kC = 2 * kHK * kDK + kHV * kDV;  // 10240
constexpr uint32_t kZ = kHV * kDV;                  // 6144
constexpr uint32_t kTpcs = 24;
constexpr float kEps = 1e-6f;

uint64_t align256(uint64_t x) { return (x + 255ull) & ~255ull; }
uint32_t f32_bits(float v) { uint32_t u = 0; std::memcpy(&u, &v, 4); return u; }

enum PatchTags : int {
    kTagResIn = 1,
    kTagDeltaIn = 2,
    kTagResOut = 3,
    kTagDeltaOut = 4,
    kTagGateIn = 5,
    kTagAttIn = 6,
    kTagResidIn = 7,
};
int tag_conv_in(uint32_t j, uint32_t b) { return 100 + int(j) * 32 + int(b); }
int tag_ssm_in(uint32_t j, uint32_t b) { return 100 + int(j) * 32 + 8 + int(b); }
int tag_st_x(uint32_t j) { return 100 + int(j) * 32 + 16; }
int tag_st_v(uint32_t j) { return 100 + int(j) * 32 + 17; }
int tag_st_g(uint32_t j) { return 100 + int(j) * 32 + 18; }
int tag_st_seq(uint32_t j) { return 100 + int(j) * 32 + 19; }
int tag_apre_k(uint32_t b) { return 50 + int(b); }
int tag_apre_v(uint32_t b) { return 60 + int(b); }

}  // namespace

DirectGdnAssets DirectGdnAssets::load(DirectRuntime& rt, const std::string& desc_dir, const std::string& tpc_obj_dir,
                                      uint32_t batch_size) {
    if (batch_size != 1 && batch_size != 8) throw std::invalid_argument("direct_gdn: batch_size must be 1 or 8");
    const std::string rtag = (batch_size == 1) ? "_R16" : "_R128";
    DirectGdnAssets a;
    a.qkv = MmeTemplate::load_dir(desc_dir + "/qkv" + rtag + "/mme_debug");
    a.z = MmeTemplate::load_dir(desc_dir + "/z" + rtag + "/mme_debug");
    a.ba = MmeTemplate::load_dir(desc_dir + "/ba" + rtag + "/mme_debug");
    a.gram = MmeTemplate::load_dir(desc_dir + "/gram" + rtag + "/mme_debug");
    a.bwr = MmeTemplate::load_dir(desc_dir + "/bwr" + rtag + "/mme_debug");
    a.out = MmeTemplate::load_dir(desc_dir + "/out" + rtag + "/mme_debug");
    a.guh = MmeTemplate::load_dir(desc_dir + "/guh" + rtag + "/mme_debug");
    a.dn = MmeTemplate::load_dir(desc_dir + "/dn" + rtag + "/mme_debug");
    a.attn_qkv = MmeTemplate::load_dir(batch_size == 1 ? (desc_dir + "/attn_qkv_R16/mme_debug")
                                                       : (desc_dir + "/attn_qkv_R128"));

    a.isa_embed_nq = rt.load_tpc_elf(tpc_obj_dir + "/direct_embed_norm_quant_bf16_gaudi2.o");
    a.isa_anq_dual = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_add_rmsnorm_quant_dual_bf16_gaudi2.o");
    a.isa_anq = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_add_rmsnorm_quant_bf16_gaudi2.o");
    a.isa_finalize = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_finalize_bf16_gaudi2.o");
    a.isa_spec_prep = rt.load_tpc_elf(tpc_obj_dir + "/direct_gdn_dq_spec_prep_bf16_gaudi2.o");
    a.isa_ba_siluz = rt.load_tpc_elf(tpc_obj_dir + "/direct_gdn_ba_siluz_bf16_gaudi2.o");
    a.isa_wyod = rt.load_tpc_elf(tpc_obj_dir + "/gdn_mtp_wyod_f32_gaudi2.o");
    a.isa_norm_mq = rt.load_tpc_elf(tpc_obj_dir + "/direct_gdn_norm_mul_quant_bf16_gaudi2.o");
    a.isa_dq_silu = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_silu_bf16_gaudi2.o");
    a.isa_dq_mq = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_mul_quant_bf16_gaudi2.o");
    a.isa_sig_mq = rt.load_tpc_elf(tpc_obj_dir + "/direct_sigmoid_mul_quant_bf16_gaudi2.o");
    a.isa_apre_prep_bf16 = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_attn_qkv_prep_bf16_gaudi2.o");
    a.isa_apre_prep_f8 = rt.load_tpc_elf(tpc_obj_dir + "/direct_dq_attn_qkv_prep_f8_gaudi2.o");
    a.isa_kv_scatter_bf16 = rt.load_tpc_elf(tpc_obj_dir + "/direct_kv_scatter_bf16_gaudi2.o");
    a.isa_kv_scatter_f8 = rt.load_tpc_elf(tpc_obj_dir + "/direct_kv_scatter_f8_gaudi2.o");
    a.isa_state_commit = rt.load_tpc_elf(tpc_obj_dir + "/gdn_state_commit_f32_gaudi2.o");
    a.isa_conv_commit = rt.load_tpc_elf(tpc_obj_dir + "/direct_conv_commit_bf16_gaudi2.o");
    return a;
}

DirectGdnBlock::DirectGdnBlock(DirectRuntime& rt, const DirectGdnAssets& assets, uint64_t ones_f32_addr,
                               const std::vector<DirectGdnLayerWeights>& layers,
                               const DirectGdnRunArgs& init_args,
                               std::array<uint32_t, 4> stash_strides_elems,
                               uint32_t tpc_cmd,
                               const DirectApostLayerWeights* apost,
                               const DirectApreLayerWeights* apre,
                               uint64_t sram_offset,
                               uint32_t batch_size)
    : nl_(uint32_t(layers.size())), batch_(batch_size), has_apost_(apost != nullptr), has_apre_(apre != nullptr),
      plan_(std::make_unique<DirectPlan>(rt, 0)), cur_(init_args) {
    if (batch_ != 1 && batch_ != 8) throw std::invalid_argument("direct_gdn: batch_size must be 1 or 8");
    if (nl_ == 0 && !has_apost_) throw std::invalid_argument("direct_gdn: empty block");
    if (init_args.conv_in.size() != nl_ || init_args.ssm_in.size() != nl_ ||
        init_args.st_x.size() != nl_ || init_args.st_v.size() != nl_ ||
        init_args.st_g.size() != nl_ || init_args.st_seq.size() != nl_)
        throw std::invalid_argument("direct_gdn: init_args layer count mismatch");

    const uint32_t B = batch_;
    const uint32_t R = B * kT;
    const uint32_t sig_r = std::min(kTpcs, R);

    // ---- SRAM arena layout (starts at sram_base + sram_offset, working peak ~25.0 MiB for B=8, ~3.2 MiB for B=1) ----
    uint64_t sp = align256(rt.sram_base() + sram_offset);
    auto alloc = [&](uint64_t bytes) -> uint64_t {
        const uint64_t addr = sp;
        sp = align256(sp + bytes);
        return addr;
    };

    // Persistent across layers:
    const uint64_t s_res0 = alloc(uint64_t(R) * kH * 2);
    const uint64_t s_res1 = alloc(uint64_t(R) * kH * 2);
    const uint64_t s_xnorm = alloc(uint64_t(R) * kH * 2);
    const uint64_t s_x8 = alloc(uint64_t(R) * kH);
    const uint64_t s_xs = alloc(uint64_t(R) * 4);
    const uint64_t scratch_base = sp;

    // Phase A (GDN mixer / apost o_proj):
    const uint64_t s_qkv = alloc(uint64_t(R) * kC * 2);
    const uint64_t s_z = alloc(uint64_t(R) * kZ * 2);
    const uint64_t s_ba = alloc(uint64_t(R) * (2 * kHV) * 4);
    const uint64_t s_g = alloc(uint64_t(R) * kHV * 4);
    const uint64_t s_beta = alloc(uint64_t(R) * kHV * 4);
    const uint64_t s_X = alloc(uint64_t(B) * kHK * (2 * kT) * kDK * 4);
    const uint64_t s_V = alloc(uint64_t(B) * kT * kZ * 4);
    const uint64_t s_gram = alloc(uint64_t(B) * kHK * (2 * kT) * (2 * kT) * 4);
    const uint64_t s_rmat = alloc(uint64_t(B) * kHK * (2 * kT) * (3 * kDV) * 4);
    const uint64_t s_core = alloc(uint64_t(R) * kHV * kDV * 4);
    const uint64_t s_o8 = alloc(uint64_t(R) * kZ);
    const uint64_t s_os = alloc(uint64_t(R) * 4);
    const uint64_t s_out = alloc(uint64_t(R) * kH * 2);
    const uint64_t peak_a = sp;

    // Phase B (MLP, reuses scratch_base):
    sp = scratch_base;
    const uint64_t s_gate = alloc(uint64_t(R) * kF * 2);
    const uint64_t s_up = alloc(uint64_t(R) * kF * 2);
    const uint64_t s_m8 = alloc(uint64_t(R) * kF);
    const uint64_t s_ms = alloc(uint64_t(R) * 4);
    const uint64_t s_dn = alloc(uint64_t(R) * kH * 2);
    const uint64_t peak_b = sp;

    // Phase C (optional DirectApre, reuses scratch_base):
    sp = scratch_base;
    const uint64_t s_attn_qkv = alloc(uint64_t(R) * 14336ull * 2ull);
    const uint64_t s_k_sram = alloc(uint64_t(B) * 4ull * kT * 256ull * 2ull);
    const uint64_t s_v_sram = alloc(uint64_t(B) * 4ull * kT * 256ull * 2ull);
    const uint64_t peak_c = sp;

    sram_used_ = std::max({peak_a, peak_b, peak_c}) - rt.sram_base();
    if (sram_used_ > rt.sram_size())
        throw std::runtime_error("direct_gdn: SRAM overflow");

    const uint32_t anq_srfs[4] = {f32_bits(kEps), f32_bits(1.0f / float(kH)), f32_bits(1.0f / 240.0f), 0};
    const uint32_t eps_srf[1] = {f32_bits(kEps)};

    // Private SOB counters: 0 = MME-A/join, 1 = MME-B/join, 2 = TPC
    uint32_t s0 = 0, s1 = 0, s2 = 0;

    auto emit_mlp_and_transition = [&](uint64_t gate_w, uint64_t up_w, uint64_t gate_s_int, uint64_t up_s_int,
                                       uint64_t dn_w, uint64_t dn_s_int, uint64_t feat_out,
                                       bool is_final, uint64_t next_in_norm1p, uint64_t next_res_dst) {
        // Overlapped MLP: MME(gate) -> MME(up) || TPC(dq_silu(gate)) -> TPC(dq_mul_quant) -> MME(down)
        plan_->add_mme(assets.guh, s_x8, gate_w, s_gate, 0, 2, s2, 0);
        s0 += assets.guh.signal_count;

        plan_->add_mme(assets.guh, s_x8, up_w, s_up, 0, -1, 0, 1);
        s1 += assets.guh.signal_count;

        {
            const DirectTpcTensor t[3] = {
                {s_gate, DType::BF16, 2, {kF, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {gate_s_int, DType::F32, 1, {kF, 1, 1, 1, 1}},
            };
            const uint32_t silu_items = 3u * R;  // 48 items at B=1 (exact 2/TPC on all 24 TPCs!), 384 at B=8
            plan_->add_tpc_grid(assets.isa_dq_silu, t, {}, {silu_items, 1, 1, 1, 1}, 0, kTpcs, 0, s0, 1, tpc_cmd);
            s1 += std::min(kTpcs, silu_items);
        }

        {
            const DirectTpcTensor t[6] = {
                {s_gate, DType::BF16, 2, {kF, R, 1, 1, 1}},
                {s_up, DType::BF16, 2, {kF, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {up_s_int, DType::F32, 1, {kF, 1, 1, 1, 1}},
                {s_m8, DType::F8, 2, {kF, R, 1, 1, 1}},
                {s_ms, DType::F32, 2, {1, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_dq_mq, t, {}, R, kTpcs, 1, s1, 2, tpc_cmd);
            s2 += sig_r;
        }

        plan_->add_mme(assets.dn, s_m8, dn_w, s_dn, 0, 2, s2, 0);
        s0 += assets.dn.signal_count;

        if (!is_final) {
            const uint32_t srfs[4] = {anq_srfs[0], anq_srfs[1], anq_srfs[2], feat_out ? 1u : 0u};
            const DirectTpcTensor t[10] = {
                {s_dn, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_ms, DType::F32, 2, {1, R, 1, 1, 1}},
                {dn_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {next_in_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
                {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {next_res_dst, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_xnorm, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {feat_out ? feat_out : next_res_dst, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_anq_dual, t, srfs, R, kTpcs, 0, s0, 2, tpc_cmd);
            s2 += sig_r;
        } else {
            const uint32_t srfs[1] = {feat_out ? 1u : 0u};
            const DirectTpcTensor t[7] = {
                {s_dn, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_ms, DType::F32, 2, {1, R, 1, 1, 1}},
                {dn_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {init_args.res_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {init_args.delta_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {feat_out ? feat_out : init_args.res_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            const int tags[7] = {-1, -1, -1, -1, kTagResOut, kTagDeltaOut, -1};
            plan_->add_tpc(assets.isa_finalize, t, srfs, R, kTpcs, 0, s0, -1, tpc_cmd, tags);
        }
    };

    if (has_apost_) {
        // Optional prepended Attention post-projection + MLP (apost):
        {
            const DirectTpcTensor t[4] = {
                {init_args.gate_in, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {init_args.att_in, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {s_o8, DType::F8, 2, {kZ, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
            };
            const int tags[4] = {kTagGateIn, kTagAttIn, -1, -1};
            plan_->add_tpc(assets.isa_sig_mq, t, {}, R, kTpcs, -1, 0, 2, tpc_cmd, tags);
            s2 += sig_r;
        }
        plan_->add_mme(assets.out, s_o8, apost->o_w, s_out, 0, 2, s2, 0);
        s0 += assets.out.signal_count;
        {
            const DirectTpcTensor t[8] = {
                {s_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
                {apost->o_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {init_args.resid_in, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {apost->post_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
                {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            const int tags[8] = {-1, -1, -1, kTagResidIn, -1, -1, -1, -1};
            plan_->add_tpc(assets.isa_anq, t, std::span<const uint32_t>(anq_srfs, 3), R, kTpcs, 0, s0, 2, tpc_cmd, tags);
            s2 += sig_r;
        }
        const bool apost_to_apre = (nl_ == 0 && has_apre_);
        emit_mlp_and_transition(apost->gate_w, apost->up_w, apost->gate_s_int, apost->up_s_int,
                                apost->dn_w, apost->dn_s_int, apost->feat_out,
                                nl_ == 0 && !has_apre_,
                                nl_ ? layers[0].in_norm1p : (apost_to_apre ? apre->in_norm1p : 0),
                                apost_to_apre ? (rt.sram_base() + kDirectSramResidOff) : s_res0);
    } else if (init_args.embed_table != 0 && init_args.embed_ids != 0 && init_args.embed_vocab > 0) {
        // Direct Span-0 entry: fused embed gather + L0 RMSNorm + FP8 quant directly into SRAM (zero Synapse embed launch!)
        const DirectTpcTensor t[7] = {
            {init_args.embed_ids, DType::I32, 1, {R, 1, 1, 1, 1}},
            {init_args.embed_table, DType::BF16, 2, {kH, init_args.embed_vocab, 1, 1, 1}},
            {layers[0].in_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
            {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
            {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
            {s_res0, DType::BF16, 2, {kH, R, 1, 1, 1}},
            {s_xnorm, DType::BF16, 2, {kH, R, 1, 1, 1}},
        };
        plan_->add_tpc(assets.isa_embed_nq, t, std::span<const uint32_t>(anq_srfs, 3), R, kTpcs, -1, 0, 2, tpc_cmd);
        s2 += sig_r;
    } else {
        // Entry stage: HBM (res_in + delta_in) -> SRAM (s_res0, s_x8, s_xs, s_xnorm) using L0.in_norm1p
        const DirectTpcTensor t[10] = {
            {init_args.delta_in, DType::BF16, 2, {kH, R, 1, 1, 1}},
            {ones_f32_addr, DType::F32, 2, {1, R, 1, 1, 1}},
            {ones_f32_addr, DType::F32, 1, {kH, 1, 1, 1, 1}},
            {init_args.res_in, DType::BF16, 2, {kH, R, 1, 1, 1}},
            {layers[0].in_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
            {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
            {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
            {s_res0, DType::BF16, 2, {kH, R, 1, 1, 1}},
            {s_xnorm, DType::BF16, 2, {kH, R, 1, 1, 1}},
            {s_res0, DType::BF16, 2, {kH, R, 1, 1, 1}},
        };
        const int tags[10] = {kTagDeltaIn, -1, -1, kTagResIn, -1, -1, -1, -1, -1, -1};
        plan_->add_tpc(assets.isa_anq_dual, t, anq_srfs, R, kTpcs, -1, 0, 2, tpc_cmd, tags);
        s2 += sig_r;
    }

    for (uint32_t j = 0; j < nl_; ++j) {
        const auto& W = layers[j];
        const bool last = (j + 1 == nl_);

        // 1) MME(qkv) + MME(z) + MME(ba): coalesced into ONE MME ECB chunk!
        plan_->add_mme(assets.qkv, s_x8, W.qkv_w, s_qkv, 0, 2, s2, 0);
        s0 += assets.qkv.signal_count;

        plan_->add_mme(assets.z, s_x8, W.z_w, s_z, 0, -1, 0, 1);
        s1 += assets.z.signal_count;
        plan_->add_mme(assets.ba, s_xnorm, W.ba_w, s_ba, 0, -1, 0, 1);
        s1 += assets.ba.signal_count;

        // 2b) CONCURRENT 24-TPC(gdn_dq_spec_prep): split into QK (wakes gram+bwr early!) and V (overlaps gram+bwr!)
        {
            DirectTpcTensor t[16];
            t[0] = {s_qkv, DType::BF16, 2, {kC, R, 1, 1, 1}};
            t[1] = {s_xs, DType::F32, 2, {1, R, 1, 1, 1}};
            t[2] = {W.qkv_s_int, DType::F32, 1, {kC, 1, 1, 1, 1}};
            t[3] = {W.conv_t, DType::BF16, 2, {kC, 4, 1, 1, 1}};
            t[4] = {s_X, DType::F32, 4, {kDK, 2 * kT, kHK, B, 1}};
            t[5] = {s_V, DType::F32, 3, {kZ, kT, B, 1, 1}};
            t[6] = {init_args.st_x[j], DType::F32, 4, {kDK, 2 * kT, kHK, B, 1},
                    {0, 0, 0, stash_strides_elems[0], 0}};
            t[7] = {init_args.st_seq[j], DType::BF16, 3, {kC, kT + 3, B, 1, 1},
                    {0, 0, stash_strides_elems[3], 0, 0}};
            int tags[16] = {-1, -1, -1, -1, -1, -1, tag_st_x(j), tag_st_seq(j),
                            -1, -1, -1, -1, -1, -1, -1, -1};
            for (uint32_t b = 0; b < 8; ++b) {
                t[8 + b] = {init_args.conv_in[j][b < B ? b : 0], DType::BF16, 2, {kC, 3, 1, 1, 1}};
                tags[8 + b] = (b < B) ? tag_conv_in(j, b) : -1;
            }
            const uint32_t kQkItems = (2 * kHK) * B;              // 32 * B
            const uint32_t kVItems = ((kC / 128) - 2 * kHK) * B;  // 48 * B
            plan_->add_tpc_grid(assets.isa_spec_prep, t, {}, {kQkItems, 1, 1, 1, 1}, 0, kTpcs,
                                0, s0, 2, tpc_cmd, tags, 0);
            s2 += std::min(kTpcs, kQkItems);
            plan_->add_tpc_grid(assets.isa_spec_prep, t, {}, {kVItems, 1, 1, 1, 1}, 0, kTpcs,
                                -1, 0, 0, tpc_cmd, tags, kQkItems);
            s0 += std::min(kTpcs, kVItems);
        }

        // 3a) MME(gram) + Bx MME(bwr): coalesced into ONE MME ECB chunk!
        plan_->add_mme(assets.gram, s_X, s_X, s_gram, 0, 2, s2, 0);
        s0 += assets.gram.signal_count;
        constexpr uint64_t kXSlotBytes = uint64_t(kHK) * (2 * kT) * kDK * 4;
        constexpr uint64_t kRmatSlotBytes = uint64_t(kHK) * (2 * kT) * (3 * kDV) * 4;
        for (uint32_t b = 0; b < B; ++b) {
            plan_->add_mme(assets.bwr, s_X + b * kXSlotBytes, init_args.ssm_in[j][b],
                           s_rmat + b * kRmatSlotBytes, 0, -1, 0, 0, {-1, tag_ssm_in(j, b), -1, -1});
            s0 += assets.bwr.signal_count;
        }

        // 3b) CONCURRENT TPC(gdn_ba_siluz): waits s1 (z + ba done), runs in parallel with gram + bwr!
        {
            const DirectTpcTensor t[9] = {
                {s_ba, DType::F32, 2, {2 * kHV, R, 1, 1, 1}},
                {W.dt_bias, DType::BF16, 1, {kHV, 1, 1, 1, 1}},
                {W.negexpA, DType::F32, 1, {kHV, 1, 1, 1, 1}},
                {s_g, DType::F32, 3, {kHV, kT, B, 1, 1}},
                {s_beta, DType::F32, 3, {kHV, kT, B, 1, 1}},
                {init_args.st_g[j], DType::F32, 3, {kHV, kT, B, 1, 1},
                 {0, 0, stash_strides_elems[2], 0, 0}},
                {s_z, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {W.z_s_int, DType::F32, 1, {kZ, 1, 1, 1, 1}},
            };
            const int tags[9] = {-1, -1, -1, -1, -1, tag_st_g(j), -1, -1, -1};
            plan_->add_tpc(assets.isa_ba_siluz, t, {}, R, kTpcs, 1, s1, 0, tpc_cmd, tags);
            s0 += sig_r;
        }

        // 4) 24-TPC(gdn_mtp_wyod): waits s0 (gram + bwr + spec_prep_v + ba_siluz ALL done!)
        {
            const DirectTpcTensor t[7] = {
                {s_g, DType::F32, 3, {kHV, kT, B, 1, 1}},
                {s_beta, DType::F32, 3, {kHV, kT, B, 1, 1}},
                {s_rmat, DType::F32, 4, {3 * kDV, 2 * kT, kHK, B, 1}},
                {s_gram, DType::F32, 4, {2 * kT, 2 * kT, kHK, B, 1}},
                {s_V, DType::F32, 4, {kDV, kHV, kT, B, 1}},
                {s_core, DType::F32, 4, {kDV, kHV, kT, B, 1}},
                {init_args.st_v[j], DType::F32, 4, {kDV, kHV, kT, B, 1},
                 {0, 0, 0, stash_strides_elems[1], 0}},
            };
            const int tags[7] = {-1, -1, -1, -1, -1, -1, tag_st_v(j)};
            plan_->add_tpc_grid(assets.isa_wyod, t, {}, {1, 1, kHV, B, 1}, 2, kTpcs,
                                0, s0, 2, tpc_cmd, tags);
            s2 += std::min(kTpcs, kHV);
        }

        // 5) TPC(direct_gdn_norm_mul_quant): waits s2, RMSNorm(s_core) * s_z -> s_o8, s_os
        {
            const DirectTpcTensor t[5] = {
                {s_core, DType::F32, 3, {kDV, kHV, R, 1, 1}},
                {s_z, DType::BF16, 2, {kZ, R, 1, 1, 1}},
                {W.gdn_norm, DType::BF16, 1, {kDV, 1, 1, 1, 1}},
                {s_o8, DType::F8, 2, {kZ, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_norm_mq, t, eps_srf, R, kTpcs, 2, s2, 2, tpc_cmd);
            s2 += sig_r;
        }

        // 6) MME(out): waits s2, s_o8 @ W_out^T -> s_out (signals s0)
        plan_->add_mme(assets.out, s_o8, W.out_w, s_out, 0, 2, s2, 0);
        s0 += assets.out.signal_count;

        // 7) TPC(post_norm anq): waits s0, dequant(s_out) + s_res0 -> s_res1, s_x8, s_xs (signals s2)
        {
            const DirectTpcTensor t[8] = {
                {s_out, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {s_os, DType::F32, 2, {1, R, 1, 1, 1}},
                {W.out_s_int, DType::F32, 1, {kH, 1, 1, 1, 1}},
                {s_res0, DType::BF16, 2, {kH, R, 1, 1, 1}},
                {W.post_norm1p, DType::BF16, 1, {kH, 1, 1, 1, 1}},
                {s_x8, DType::F8, 2, {kH, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {s_res1, DType::BF16, 2, {kH, R, 1, 1, 1}},
            };
            plan_->add_tpc(assets.isa_anq, t, std::span<const uint32_t>(anq_srfs, 3), R, kTpcs, 0, s0, 2, tpc_cmd);
            s2 += sig_r;
        }

        // 8-9) Overlapped MLP & transition to next GDN layer, DirectApre, or finalize to HBM
        const bool to_apre = (last && has_apre_);
        emit_mlp_and_transition(W.gate_w, W.up_w, W.gate_s_int, W.up_s_int,
                                W.dn_w, W.dn_s_int, W.feat_out,
                                last && !has_apre_,
                                !last ? layers[j + 1].in_norm1p : (to_apre ? apre->in_norm1p : 0),
                                to_apre ? (rt.sram_base() + kDirectSramResidOff) : s_res0);
    }

    if (has_apre_) {
        // Optional appended DirectApre: MME(attn_qkv) -> 24-TPC(qkv_prep) -> kv_scatter
        const DType kv_dt = apre->kv8 ? DType::F8 : DType::BF16;
        const uint64_t s_persist_gate = rt.sram_base() + kDirectSramGateOff;
        const uint64_t s_persist_q = rt.sram_base() + kDirectSramQOff;

        plan_->add_mme(assets.attn_qkv, s_x8, apre->qkv_w, s_attn_qkv, 0, 2, s2, 0);
        s0 += assets.attn_qkv.signal_count;

        {
            const DirectTpcTensor t[12] = {
                {s_attn_qkv, DType::BF16, 2, {14336, R, 1, 1, 1}},
                {s_xs, DType::F32, 2, {1, R, 1, 1, 1}},
                {apre->qkv_s_int, DType::F32, 1, {14336, 1, 1, 1, 1}},
                {apre->q_norm, DType::BF16, 1, {256, 1, 1, 1, 1}},
                {apre->k_norm, DType::BF16, 1, {256, 1, 1, 1, 1}},
                {apre->rope_pos, DType::I32, 1, {R, 1, 1, 1, 1}},
                {apre->cos_tab, DType::F32, 2, {32, apre->rope_tab_rows, 1, 1, 1}},
                {apre->sin_tab, DType::F32, 2, {32, apre->rope_tab_rows, 1, 1, 1}},
                {s_persist_q, kv_dt, 4, {256, 96, 4, B, 1}},
                {s_persist_gate, DType::BF16, 2, {6144, R, 1, 1, 1}},
                {s_k_sram, kv_dt, 4, {256, kT, 4, B, 1}},
                {s_v_sram, kv_dt, 4, {256, kT, 4, B, 1}},
            };
            const uint64_t isa_prep = apre->kv8 ? assets.isa_apre_prep_f8 : assets.isa_apre_prep_bf16;
            const uint32_t prep_items = 3 * R;
            plan_->add_tpc(isa_prep, t, eps_srf, prep_items, kTpcs, 0, s0, 2, tpc_cmd);
            s2 += std::min(kTpcs, prep_items);
        }

        const uint64_t isa_scat = apre->kv8 ? assets.isa_kv_scatter_f8 : assets.isa_kv_scatter_bf16;
        const uint32_t halves = (B <= 4) ? 1u : 2u;
        for (uint32_t half = 0; half < halves; ++half) {
            const uint32_t sb = half * 4u;
            const uint32_t cnt = std::min<uint32_t>(4u, B - sb);
            const uint32_t items = cnt * 64u;
            DirectTpcTensor t[11];
            t[0] = {s_k_sram, kv_dt, 4, {256, kT, 4, B, 1}};
            t[1] = {s_v_sram, kv_dt, 4, {256, kT, 4, B, 1}};
            int tags[11] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
            for (uint32_t s = 0; s < 4; ++s) {
                const uint32_t b = (s < cnt) ? (sb + s) : 0u;
                const uint32_t cap = init_args.apre_kv_cap[b];
                const uint32_t h_stride = cap * 256u;
                t[2 + s] = {init_args.apre_k_dst[b], kv_dt, 3, {256, cap, 4, 1, 1}, {0, 0, h_stride, 0, 0}};
                t[6 + s] = {init_args.apre_v_dst[b], kv_dt, 3, {256, cap, 4, 1, 1}, {0, 0, h_stride, 0, 0}};
                if (s < cnt) {
                    tags[2 + s] = tag_apre_k(b);
                    tags[6 + s] = tag_apre_v(b);
                }
            }
            t[10] = {apre->last_tab, DType::I32, 2, {32, 8, 1, 1, 1}};
            const uint32_t sb_srf[1] = {sb};
            const int sig = (half + 1 == halves) ? -1 : 2;
            plan_->add_tpc(isa_scat, t, sb_srf, items, kTpcs, 2, s2, sig, tpc_cmd, tags);
            if (sig == 2) s2 += std::min(kTpcs, items);
        }
    }

    plan_->prepare();
}

DirectGdnBlock::~DirectGdnBlock() = default;

void DirectGdnBlock::update_args(const DirectGdnRunArgs& args) {
    if (args.res_in != cur_.res_in) { plan_->patch_addr(kTagResIn, args.res_in); cur_.res_in = args.res_in; }
    if (args.delta_in != cur_.delta_in) { plan_->patch_addr(kTagDeltaIn, args.delta_in); cur_.delta_in = args.delta_in; }
    if (args.gate_in != cur_.gate_in) { plan_->patch_addr(kTagGateIn, args.gate_in); cur_.gate_in = args.gate_in; }
    if (args.att_in != cur_.att_in) { plan_->patch_addr(kTagAttIn, args.att_in); cur_.att_in = args.att_in; }
    if (args.resid_in != cur_.resid_in) { plan_->patch_addr(kTagResidIn, args.resid_in); cur_.resid_in = args.resid_in; }
    if (args.res_out != cur_.res_out) { plan_->patch_addr(kTagResOut, args.res_out); cur_.res_out = args.res_out; }
    if (args.delta_out != cur_.delta_out) { plan_->patch_addr(kTagDeltaOut, args.delta_out); cur_.delta_out = args.delta_out; }
    for (uint32_t j = 0; j < nl_; ++j) {
        for (uint32_t b = 0; b < batch_; ++b) {
            if (args.conv_in[j][b] != cur_.conv_in[j][b]) {
                plan_->patch_addr(tag_conv_in(j, b), args.conv_in[j][b]);
                cur_.conv_in[j][b] = args.conv_in[j][b];
            }
            if (args.ssm_in[j][b] != cur_.ssm_in[j][b]) {
                plan_->patch_addr(tag_ssm_in(j, b), args.ssm_in[j][b]);
                cur_.ssm_in[j][b] = args.ssm_in[j][b];
            }
        }
        if (args.st_x[j] != cur_.st_x[j]) { plan_->patch_addr(tag_st_x(j), args.st_x[j]); cur_.st_x[j] = args.st_x[j]; }
        if (args.st_v[j] != cur_.st_v[j]) { plan_->patch_addr(tag_st_v(j), args.st_v[j]); cur_.st_v[j] = args.st_v[j]; }
        if (args.st_g[j] != cur_.st_g[j]) { plan_->patch_addr(tag_st_g(j), args.st_g[j]); cur_.st_g[j] = args.st_g[j]; }
        if (args.st_seq[j] != cur_.st_seq[j]) { plan_->patch_addr(tag_st_seq(j), args.st_seq[j]); cur_.st_seq[j] = args.st_seq[j]; }
    }
    if (has_apre_) {
        for (uint32_t b = 0; b < batch_; ++b) {
            if (args.apre_k_dst[b] != cur_.apre_k_dst[b]) {
                plan_->patch_addr(tag_apre_k(b), args.apre_k_dst[b]);
                cur_.apre_k_dst[b] = args.apre_k_dst[b];
            }
            if (args.apre_v_dst[b] != cur_.apre_v_dst[b]) {
                plan_->patch_addr(tag_apre_v(b), args.apre_v_dst[b]);
                cur_.apre_v_dst[b] = args.apre_v_dst[b];
            }
        }
    }
}

double DirectGdnBlock::run() { return plan_->run(); }

DirectGdnCommit::DirectGdnCommit(DirectRuntime& rt, const DirectGdnAssets& assets, uint32_t ng, uint32_t T,
                                 const DirectGdnCommitArgs& init_args, uint32_t tpc_cmd, uint32_t batch_size)
    : batch_(batch_size), plan_(std::make_unique<DirectPlan>(rt, 0)), cur_(init_args) {
    if (batch_ < 1 || batch_ > 8) throw std::invalid_argument("direct_gdn_commit: batch_size must be 1..8");
    // Emit all active slots' in-place state commit + conv commit back-to-back with ZERO inter-slot barriers:
    // every slot writes only to its own independent ssm/conv_out buffers and signals the host SOB directly!
    for (uint32_t b = 0; b < batch_; ++b) {
        const DirectTpcTensor st_t[6] = {
            {init_args.ssm[b], DType::F32, 3, {kDK, kDV, ng * kHV, 1, 1}},
            {init_args.x[b], DType::F32, 4, {kDK, 2 * T, kHK, ng, 1}},
            {init_args.v[b], DType::F32, 4, {kDV, kHV, T, ng, 1}},
            {init_args.g[b], DType::F32, 3, {kHV, T, ng, 1, 1}},
            {init_args.acc[b], DType::I32, 1, {1, 1, 1, 1, 1}},
            {init_args.ssm[b], DType::F32, 3, {kDK, kDV, ng * kHV, 1, 1}},
        };
        const int st_tags[6] = {int(b), int(16 + b), int(24 + b), int(32 + b), int(48 + b), int(b)};
        plan_->add_tpc_grid(assets.isa_state_commit, st_t, {}, {ng * kHV, 1, 1, 1, 1}, 0, kTpcs,
                            -1, 0, -1, tpc_cmd, st_tags);

        const DirectTpcTensor cv_t[3] = {
            {init_args.seq[b], DType::BF16, 3, {kC, T + 3, ng, 1, 1}},
            {init_args.rows[b], DType::I32, 1, {3, 1, 1, 1, 1}},
            {init_args.conv_out[b], DType::BF16, 3, {kC, 3, ng, 1, 1}},
        };
        const int cv_tags[3] = {int(40 + b), int(56 + b), int(8 + b)};
        plan_->add_tpc_grid(assets.isa_conv_commit, cv_t, {}, {1, ng, 1, 1, 1}, 1, kTpcs,
                            -1, 0, -1, tpc_cmd, cv_tags);
    }
    plan_->prepare();
}

DirectGdnCommit::~DirectGdnCommit() = default;

void DirectGdnCommit::update_args(const DirectGdnCommitArgs& args) {
    for (uint32_t b = 0; b < batch_; ++b) {
        if (args.ssm[b] != cur_.ssm[b]) { plan_->patch_addr(int(b), args.ssm[b]); cur_.ssm[b] = args.ssm[b]; }
        if (args.conv_out[b] != cur_.conv_out[b]) { plan_->patch_addr(int(8 + b), args.conv_out[b]); cur_.conv_out[b] = args.conv_out[b]; }
        if (args.x[b] != cur_.x[b]) { plan_->patch_addr(int(16 + b), args.x[b]); cur_.x[b] = args.x[b]; }
        if (args.v[b] != cur_.v[b]) { plan_->patch_addr(int(24 + b), args.v[b]); cur_.v[b] = args.v[b]; }
        if (args.g[b] != cur_.g[b]) { plan_->patch_addr(int(32 + b), args.g[b]); cur_.g[b] = args.g[b]; }
        if (args.seq[b] != cur_.seq[b]) { plan_->patch_addr(int(40 + b), args.seq[b]); cur_.seq[b] = args.seq[b]; }
        if (args.acc[b] != cur_.acc[b]) { plan_->patch_addr(int(48 + b), args.acc[b]); cur_.acc[b] = args.acc[b]; }
        if (args.rows[b] != cur_.rows[b]) { plan_->patch_addr(int(56 + b), args.rows[b]); cur_.rows[b] = args.rows[b]; }
    }
}

double DirectGdnCommit::run() { return plan_->run(); }

}  // namespace g2e
