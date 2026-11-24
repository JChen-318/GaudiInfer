#include "g2e/inkling2_direct.hpp"

#include <functional>

#include "g2e/inkling2.hpp"
#include "g2e/numeric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace g2e {
namespace {

std::string Lw(uint32_t i, const char* part) { return "L" + std::to_string(i) + "." + part; }
uint64_t align256(uint64_t x) { return (x + 255) & ~uint64_t(255); }

DirectTpcTensor tt(uint64_t addr, DType dt, std::initializer_list<uint32_t> sizes) {
    DirectTpcTensor t{};
    t.addr = addr;
    t.dtype = dt;
    t.dims = uint32_t(sizes.size());
    uint32_t i = 0;
    for (auto s : sizes) t.sizes[i++] = s;
    for (; i < 5; ++i) t.sizes[i] = 1;  // (the kernels read row counts from dims beyond the given ones)
    return t;
}
template <typename P>
std::vector<uint32_t> srf(const P& p) {
    std::vector<uint32_t> v(sizeof(P) / 4);
    std::memcpy(v.data(), &p, sizeof(P));
    return v;
}

}  // namespace

InkDirect::InkDirect(Inkling2& m) : m_(m) {
    const char* e = std::getenv("G2E_INK_DIRECT");
    mask_ = e ? uint32_t(std::strtoul(e, nullptr, 0)) : 0;
    if (!mask_ && !std::getenv("G2E_INK_DPREFILL")) return;
    if (!std::getenv("SCAL_CFG_FILE_PATH") && !std::getenv("SCAL_CONFIG_FILE_PATH"))
        throw std::runtime_error("inkling2 direct: set SCAL_CFG_FILE_PATH and SCAL_CONFIG_FILE_PATH to tools/scal-g2e-ink.json "
                                 "before starting");
    rt_ = std::make_unique<DirectRuntime>(m_.dev_);
    // Two lanes of 64 SOBs and the TPC monitors up to 1392, plus the chain's SOB and 4 monitors: more than the stock
    // Direct config of the other models has.
    if (rt_->sob_count() < 129 || rt_->monitor_count() < 1396)
        throw std::runtime_error("inkling2 direct: the SCAL config has " + std::to_string(rt_->sob_count()) + " SOBs and " +
                                 std::to_string(rt_->monitor_count()) + " monitors, too few for the two plan lanes: point "
                                 "SCAL_CFG_FILE_PATH and SCAL_CONFIG_FILE_PATH at tools/scal-g2e-ink.json");
    // This engine's plans count on the diff-encoded MME descriptors (the plan pool), on the scheduler PDMA for its
    // per-chunk patches, and on the polled fence wait (a decode token is a few short plans).
    rt_->set_options({.spin_wait = true, .mme_diff = true, .pdma_patches = true});
    const char* td = std::getenv("G2E_TPC_OBJ_DIR");
    const std::string tpc = (td && *td) ? td : std::string(G2E_TPC_OBJ_DIR);
    for (const char* k : {"ink_dgram_gaudi2", "ink_dgram_msg_gaudi2", "ink_dnorm_msg_gaudi2", "ink_dnorm_msg_f32p_gaudi2", "ink_dmgy_gaudi2", "ink_dmgn_gaudi2", "ink_dnorm_gaudi2", "ink_dnorm_f32p_gaudi2", "ink_dmid_gaudi2", "ink_rollback_gaudi2", "ink_dpost_gaudi2", "ink_dpost_full_gaudi2", "ink_dpost_v_gaudi2", "ink_dpost_vf_gaudi2", "ink_dattn_v_gaudi2",
                          "ink_dattn_vf_gaudi2", "ink_dgate_v_gaudi2", "ink_dgate_vf_gaudi2", "ink_dattn_far_gaudi2", "ink_dgate_far_gaudi2",
                          "ink_dattn_gaudi2", "ink_dgate_gaudi2", "ink_route_gaudi2", "ink_droute1_gaudi2", "ink_dmg_fgy_gaudi2",
                          "ink_dtail_rw_gaudi2"})
        isa_[k] = rt_->load_tpc_elf(tpc + "/" + k + ".o");
    for (const char* k : {"ink_dembed_gaudi2", "ink_dsoftcap_argmax_gaudi2", "ink_kv_commit_bf16_gaudi2",
                          "ink_dattn_near_max_gaudi2", "ink_dfar_sm_gaudi2", "ink_dfar_acc_gaudi2", "ink_dgate_f32_gaudi2",
                          "ink_dpost_b_gaudi2", "ink_dpost_b_full_gaudi2", "ink_kv_commit_d1_gaudi2",
                          "ink_dattn_b_gaudi2", "ink_dattn_b_near_max_gaudi2", "ink_dgate_b_gaudi2", "ink_dgate_b_full_gaudi2",
                          "ink_dmid_b_gaudi2", "ink_dtail_rw_b_gaudi2", "ink_dsoftcap_argmax_b_gaudi2",
                          "ink_vmoe_plan_gaudi2", "ink_vmoe_up_nvf4_gaudi2", "ink_vmoe_up_mxf8_gaudi2",
                          "ink_vmoe_dn_nvf4_gaudi2", "ink_vmoe_dn_mxf8_gaudi2",
                          "ink_dtail_rw_dgram_gaudi2", "ink_dtail_rw_dgram_msg_gaudi2",
                          "ink_dmoe_up9_nvf4_gaudi2", "ink_dmoe_up9_mxf8_gaudi2",
                          "ink_dmoe_dn9_nvf4_gaudi2", "ink_dmoe_dn9_mxf8_gaudi2",
                          "ink_pcopy_gaudi2",
                          "ink_dfar_pv_gaudi2", "ink_dfar_acc3_gaudi2", "ink_dgate_far3_gaudi2",
                          "ink_droute1_b_gaudi2", "ink_kv_commit_b4_gaudi2"})
        if (std::filesystem::exists(tpc + "/" + k + ".o")) isa_[k] = rt_->load_tpc_elf(tpc + "/" + k + ".o");
    const char* dd = std::getenv("G2E_INK_MME_DIR");
    const std::string mme = (dd && *dd) ? dd : std::string(G2E_SOURCE_DIR) + "/tools/mme_descs_ink";
    for (uint32_t T = 2; T <= kVerifyMaxT; ++T)
        for (const char* k : {"proj", "proj3s", "proj3f", "wo", "router", "bias", "qk", "qkn", "pv", "pvn", "mgd3", "mgf3x", "mgd1", "mgf1"}) {
            const std::string n = "v" + std::to_string(T) + "_" + k;
            if (std::filesystem::exists(mme + "/" + n + "/mme_debug")) mme_[n] = MmeTemplate::load_dir(mme + "/" + n + "/mme_debug");
        }
    for (const char* k : {"b8_proj", "b8_proj3s", "b8_proj3f", "b8_wo", "b8_router", "b8_bias",
                          "b8_mgd3", "b8_mgf3x", "b8_mgd1", "b8_mgf1",
                          "b2_lmhead", "b4_lmhead", "b8_lmhead"})
        if (std::filesystem::exists(mme + "/" + k + "/mme_debug")) mme_[k] = MmeTemplate::load_dir(mme + "/" + k + "/mme_debug");
    for (const char* k : {"d_proj_s", "d_bias16", "d_qk16", "d_pvb", "d_wo", "d_router", "d_mgd3", "d_mgd1", "d_mgf1", "d_proj3s",
                          "d_proj3f", "d_mgf3x"})
        mme_[k] = MmeTemplate::load_dir(mme + "/" + k + "/mme_debug");
    if (std::filesystem::exists(mme + "/d_lmhead/mme_debug"))
        mme_["d_lmhead"] = MmeTemplate::load_dir(mme + "/d_lmhead/mme_debug");
    vfar_cap_ = std::min<uint32_t>(m_.max_ctx_, 65536u);
    // Load all far-attention templates up to 131072 (verify capped at vfar_cap_ <= 65536).
    static constexpr uint32_t kFarBuckets[] = {1024, 2048, 3072, 4096, 6144, 8192, 9216, 13824, 16384,
                                               20736, 31104, 32768, 46656, 65536, 69984, 104976, 131072};
    auto load_far = [&](uint32_t b) {
        for (const char* kind : {"d_fqk_", "d_fpv_"}) {
            const std::string n = kind + std::to_string(b);
            if (!mme_.count(n) && std::filesystem::exists(mme + "/" + n + "/mme_debug"))
                mme_[n] = MmeTemplate::load_dir(mme + "/" + n + "/mme_debug");
        }
        if (b <= vfar_cap_) {
            for (const char* kind : {"v2_fqk_", "v2_fpv_", "v3_fqk_", "v3_fpv_", "v4_fqk_", "v4_fpv_"}) {
                const std::string n = kind + std::to_string(b);
                if (!mme_.count(n) && std::filesystem::exists(mme + "/" + n + "/mme_debug"))
                    mme_[n] = MmeTemplate::load_dir(mme + "/" + n + "/mme_debug");
            }
        }
    };
    for (uint32_t b : kFarBuckets) {
        if (b > m_.max_ctx_ && b != 1024u) continue;
        load_far(b);
    }
    if (m_.max_ctx_ <= 131072u) load_far(m_.max_ctx_);
    prepare_constants();
    // scratch (HBM / SRAM)
    const auto& c = m_.cfg_;
    const uint64_t H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim, W = c.window, RE = c.rel_extent;
    const uint64_t N = NH * D * 2 + NH * c.d_rel + NH + 2 * KV * D;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t o = at; at = align256(at + bytes); return o; };
    s_.xa = take(3 * H * 2); s_.row = take(3 * (N + 2 * KV * 4) * 2); s_.q = take(NH * D * 2); s_.r = take(NH * 16 * 2);
    s_.sink = take(NH * 4); s_.bt = take(NH * RE * 2); s_.sc = take(KV * 3 * W * 2); s_.p = take(KV * 3 * W * 2);
    s_.sums = take(NH * 4); s_.pv = take(NH * D * 2); s_.gated = take(NH * D * 2); s_.a1 = take(H * 2);
    s_.res = take(H * 4); s_.xn = take(H * 2); s_.logits = take(c.experts * 4); s_.idx = take(64);
    s_.rw = take(64); s_.act = take(c.topk * c.moe_inter * 2); s_.ys = take(c.topk * H * 2);
    s_.p1 = take(64 * 4 * (H / 128)); s_.p2 = take(64 * 4 * (H / 128)); s_.p3 = take(64 * 4 * (H / 128));
    s_.cat = take(3 * 2 * H * 2); s_.xil = take(3 * H * 4); s_.msg = take(4 * H * 4); s_.gr = take(3 * c.gate_rank * 2);
    s_.fgv = take(3 * 3 * 2 * H * 4); s_.fqkv = take(3 * H * 2); s_.mrmix = take(H * 4); s_.ymg = take(3 * H * 4); s_.p4 = take(64 * 4 * (H / 128));
    s_.pvf = take(NH * D * 2); s_.pvf2 = take(NH * D * 2); s_.pvf_part = take(3 * NH * D * 4);
    s_.dmax = take(NH * 4); s_.dres = take(NH * 4); s_.dres2 = take(NH * 4); s_.o_acc = take(NH * D * 4);
    s_.dblocks = take(4 * H * 2); s_.dh0 = take(H * 2); s_.dh1 = take(H * 2);
    s_.dprev0 = take(2 * H * 2); s_.dprev1 = take(2 * H * 2);
    scratch_ = DeviceBuffer(m_.dev_, at);
    uint64_t base = scratch_.address();
    dfar_cap_ = std::min<uint32_t>(m_.max_ctx_, 131072u);
    // G2E_INK_DIRECT_SRAM=1: the activations / scratch in SRAM (lower latency for the many small kernels). Only valid
    // while no Synapse recipe runs concurrently (Synapse uses SRAM as workspace): the plans run synchronously.
    if (const char* e2 = std::getenv("G2E_INK_DIRECT_SRAM"); e2 && *e2 == '1') {
        if (at > rt_->sram_usable()) throw std::runtime_error("inkling2 direct: scratch does not fit in SRAM");
        base = align256(rt_->sram_base());
        sram_scratch_ = true;
    }
    for (uint64_t* f : {&s_.xa, &s_.row, &s_.q, &s_.r, &s_.sink, &s_.bt, &s_.sc, &s_.p, &s_.sums, &s_.pv, &s_.gated,
                        &s_.a1, &s_.res, &s_.xn, &s_.logits, &s_.idx, &s_.rw, &s_.act, &s_.ys, &s_.p1, &s_.p2, &s_.p3, &s_.cat, &s_.xil,
                        &s_.msg, &s_.gr, &s_.fgv, &s_.fqkv, &s_.mrmix, &s_.pvf, &s_.pvf2, &s_.pvf_part, &s_.ymg, &s_.p4, &s_.dmax, &s_.dres, &s_.dres2, &s_.o_acc,
                        &s_.dblocks, &s_.dh0, &s_.dh1, &s_.dprev0, &s_.dprev1})
        *f += base;
    s_.pvf_buf[0] = s_.pvf;
    s_.pvf_buf[1] = s_.pvf2;
    s_.dres_buf[0] = s_.dres;
    s_.dres_buf[1] = s_.dres2;
    d_argmax_ = DeviceBuffer(m_.dev_, 2ull * 64 * 24 * 4);
    bmeta_ = DeviceBuffer(m_.dev_, 256);
    b_argmax_ = DeviceBuffer(m_.dev_, 2ull * 8 * 24 * 64 * 4);
    uint64_t sram_used = sram_scratch_ ? align256(at) : 0;
    {   // batched decode scratch (Bp <= 8): ~5.2 MiB, placed in SRAM right after s_ when sram_scratch_ is enabled
        const uint64_t Bp = 8, NT = H / 128, E = c.experts, F = c.moe_inter, J = c.topk;
        const uint64_t NROW = NH * D * 2 + NH * c.d_rel + NH + 2 * (KV * D + KV * 4);
        uint64_t bat = 0;
        auto btake = [&](uint64_t bytes) { const uint64_t o = bat; bat = align256(bat + std::max<uint64_t>(bytes, 256)); return o; };
        b_.xa = btake(3 * Bp * H * 2); b_.row = btake(3 * Bp * NROW * 2); b_.q = btake(Bp * NH * D * 2);
        b_.r = btake(Bp * NH * 16 * 2); b_.sink = btake(Bp * NH * 4); b_.bt = btake(Bp * NH * RE * 2);
        b_.sc = btake(Bp * KV * 3 * W * 2); b_.p = btake(Bp * KV * 3 * W * 2); b_.sums = btake(Bp * NH * 4);
        b_.dmax = btake(Bp * NH * 4); b_.pv = btake(Bp * NH * D * 2); b_.o_acc = btake(Bp * NH * D * 4);
        b_.b_kvs = btake(Bp * 2 * KV * D * 2); b_.gated = btake(Bp * NH * D * 2); b_.a1 = btake(Bp * H * 2);
        b_.res = btake(Bp * H * 4); b_.xn = btake(Bp * H * 2); b_.logits = btake(Bp * E * 4);
        b_.idx = btake(Bp * J * 4); b_.rw = btake(Bp * J * 4); b_.act = btake(Bp * J * F * 2);
        b_.ys = btake(Bp * J * H * 2); b_.p1 = btake(64 * 4 * NT * Bp); b_.p2 = btake(64 * 4 * NT * Bp);
        b_.p3 = btake(64 * 4 * NT * Bp); b_.p4 = btake(64 * 4 * NT * Bp); b_.cat = btake(3 * Bp * 2 * H * 2);
        b_.xil = btake(3 * Bp * H * 4); b_.msg = btake(4 * Bp * H * 4); b_.gr = btake(3 * Bp * c.gate_rank * 2);
        b_.fgv = btake(3 * Bp * 3 * 2 * H * 4); b_.fqkv = btake(3 * Bp * H * 2); b_.mrmix = btake(Bp * H * 4);
        b_.ymg = btake(3 * Bp * H * 4);
        b_.grp = btake(64 * 4 * 4);
        uint64_t bbase = 0;
        if (sram_scratch_ && sram_used + bat <= rt_->sram_usable()) {
            bbase = align256(rt_->sram_base() + sram_used);
            sram_used += align256(bat);
        } else {
            bscratch_ = DeviceBuffer(m_.dev_, bat);
            bbase = bscratch_.address();
        }
        for (uint64_t* f : {&b_.xa, &b_.row, &b_.q, &b_.r, &b_.sink, &b_.bt, &b_.sc, &b_.p, &b_.sums, &b_.dmax,
                            &b_.pv, &b_.o_acc, &b_.b_kvs, &b_.gated, &b_.a1, &b_.res, &b_.xn, &b_.logits, &b_.idx,
                            &b_.rw, &b_.act, &b_.ys, &b_.p1, &b_.p2, &b_.p3, &b_.p4, &b_.cat, &b_.xil, &b_.msg,
                            &b_.gr, &b_.fgv, &b_.fqkv, &b_.mrmix, &b_.ymg, &b_.grp})
            *f += bbase;
    }
    {   // verification scratch (T = kVerifyMaxT rows): ~2.4 MiB activations placed in SRAM right after b_ when enabled;
        // far buffers (sf, pf, capped at 65536 = 24 MiB) stay in HBM vscratch_.
        const uint64_t T = kVerifyMaxT, NT = H / 128, E = c.experts, F = c.moe_inter;
        const uint64_t NROW = NH * D * 2 + NH * c.d_rel + NH + 2 * (KV * D + KV * 4);
        uint64_t vat = 0;
        auto vtake = [&](uint64_t bytes) { const uint64_t o = vat; vat = align256(vat + std::max<uint64_t>(bytes, 256)); return o; };
        auto& v = vs_;
        v.xa = vtake(3 * T * H * 2); v.row = vtake(3 * T * NROW * 2); v.q = vtake(NH * T * D * 2); v.r = vtake(T * NH * 16 * 2);
        v.sink = vtake(T * NH * 4); v.bt = vtake(T * NH * RE * 2); v.sc = vtake(NH * T * W * 2); v.scn = vtake(NH * T * T * 2);
        v.p = vtake(NH * T * W * 2); v.pn = vtake(NH * T * T * 2); v.sums = vtake(T * NH * 4); v.pv = vtake(NH * T * D * 2);
        v.pvn = vtake(NH * T * D * 2); v.pvf = vtake(NH * T * D * 2); v.gated = vtake(T * NH * D * 2); v.a1 = vtake(T * H * 2);
        v.res = vtake(T * H * 4); v.xn = vtake(T * H * 2); v.logits = vtake(T * E * 4); v.idx = vtake(T * 8 * 4); v.rw = vtake(T * 8 * 4);
        v.act = vtake(T * 8 * F * 2); v.ys = vtake(T * 8 * H * 2); v.p1 = vtake(64 * 4 * NT * T); v.p2 = vtake(64 * 4 * NT * T);
        v.p3 = vtake(64 * 4 * NT * T); v.p4 = vtake(64 * 4 * NT * T); v.cat = vtake(3 * T * 2 * H * 2); v.xil = vtake(3 * T * H * 4);
        v.msg = vtake(4 * T * H * 4); v.gr = vtake(3 * T * 64 * 2); v.fgv = vtake(3 * T * 3 * 2 * H * 4); v.fqkv = vtake(3 * T * H * 2);
        v.mrmix = vtake(T * H * 4); v.ymg = vtake(3 * T * H * 4);
        v.grp = vtake(64 * 4 * 4);
        uint64_t vbase = 0;
        if (sram_scratch_ && sram_used + vat <= rt_->sram_usable()) {
            vbase = align256(rt_->sram_base() + sram_used);
            sram_used += align256(vat);
        } else {
            vscratch_act_ = DeviceBuffer(m_.dev_, vat);
            vbase = vscratch_act_.address();
        }
        for (uint64_t* f : {&v.xa, &v.row, &v.q, &v.r, &v.sink, &v.bt, &v.sc, &v.scn, &v.p, &v.pn, &v.sums, &v.pv, &v.pvn, &v.pvf, &v.gated,
                            &v.a1, &v.res, &v.xn, &v.logits, &v.idx, &v.rw, &v.act, &v.ys, &v.p1, &v.p2, &v.p3, &v.p4, &v.cat, &v.xil,
                            &v.msg, &v.gr, &v.fgv, &v.fqkv, &v.mrmix, &v.ymg, &v.grp})
            *f += vbase;
        const uint64_t vfar_half = align256(NH * T * uint64_t(vfar_cap_) * 2);
        vscratch_ = DeviceBuffer(m_.dev_, 2 * vfar_half);
        v.sf = vscratch_.address();
        v.pf = v.sf + vfar_half;
    }
    {   // SRAM cache for read-only small layer constants + interleaved depth-attention queries (~3.8 MiB total,
        // placed in SRAM right after s_ + b_ + vs_; refreshed via ink_pcopy_gaudi2 once after prefill/Synapse).
        static const bool sram_const_env = [] { const char* e = std::getenv("G2E_INK_SRAM_CONST"); return !e || *e != '0'; }();
        if (sram_scratch_ && sram_const_env && isa_.count("ink_pcopy_gaudi2") &&
            const_small_bytes_ > 0 && (const_small_bytes_ % 8192u) == 0 &&
            m_.qil_.size() > 0 && (m_.qil_.size() % 8192u) == 0 &&
            sram_used + const_small_bytes_ + m_.qil_.size() <= rt_->sram_usable()) {
            sram_const_base_ = align256(rt_->sram_base() + sram_used);
            sram_used += align256(const_small_bytes_);
            sram_qil_base_ = align256(rt_->sram_base() + sram_used);
            sram_used += align256(m_.qil_.size());
            sram_const_on_ = true;
        }
    }
    {   // Double-buffered far attention scores / probabilities (sf_buf[2], pf_buf[2]; 4 x 6 MiB = 24 MiB at 131K):
        // placed in 48 MiB SRAM right after s_ + b_ + vs_ + const_small + qil_ (~12.2 + 24.0 = 36.2 MiB <= 48 MiB).
        // Non-tiled 2K..131K callers use s_.sf = sf_buf[0], s_.pf = pf_buf[0] and get SRAM residency for free!
        static const bool dfar_sram_env = [] { const char* e = std::getenv("G2E_INK_DFAR_SRAM"); return !e || *e != '0'; }();
        const uint64_t far_half = align256(uint64_t(NH) * dfar_cap_ * 2);
        const uint64_t far_bytes = 4ull * far_half;
        uint64_t fbase = 0;
        if (sram_scratch_ && dfar_sram_env && sram_used + far_bytes <= rt_->sram_usable()) {
            fbase = align256(rt_->sram_base() + sram_used);
            sram_used += far_bytes;
        } else {
            far_scratch_ = DeviceBuffer(m_.dev_, far_bytes);
            fbase = far_scratch_.address();
        }
        s_.sf_buf[0] = fbase + 0 * far_half;
        s_.sf_buf[1] = fbase + 1 * far_half;
        s_.pf_buf[0] = fbase + 2 * far_half;
        s_.pf_buf[1] = fbase + 3 * far_half;
        s_.sf = s_.sf_buf[0];
        s_.pf = s_.pf_buf[0];
    }
    {
        const char* fe = std::getenv("G2E_INK_DIRECT_FULL");
        d_full_on_ = (!fe || *fe != '0') && isa_.count("ink_dembed_gaudi2") &&
                     isa_.count("ink_dsoftcap_argmax_gaudi2") && mme_.count("d_lmhead") &&
                     m_.qil_off_.count("final_q");
        const char* te = std::getenv("G2E_INK_DFAR_TILE");
        dfar_tile_thresh_ = te ? uint32_t(std::strtoul(te, nullptr, 0)) : 131072u;
        const char* pe = std::getenv("G2E_INK_DFAR_PIPE");
        dfar_pipe_on_ = !pe || *pe != '0';
        const char* tpve = std::getenv("G2E_INK_DFAR_TPC_PV");
        dfar_tpc_pv_on_ = (!tpve || *tpve != '0') &&
                          isa_.count("ink_dfar_pv_gaudi2") != 0 &&
                          isa_.count("ink_dfar_acc3_gaudi2") != 0 &&
                          isa_.count("ink_dgate_far3_gaudi2") != 0;
        dfar_tiled_ok_ = isa_.count("ink_dattn_near_max_gaudi2") && isa_.count("ink_dfar_sm_gaudi2") &&
                         (isa_.count("ink_dfar_acc_gaudi2") || dfar_tpc_pv_on_) && isa_.count("ink_dgate_f32_gaudi2") &&
                         mme_.count("d_fqk_1024") && (mme_.count("d_fpv_1024") || dfar_tpc_pv_on_);
        const bool nv_dmoe_ok = !m_.cfg_.has_nvfp4() ||
                                (isa_.count("ink_dmoe_up9_nvf4_gaudi2") != 0 &&
                                 isa_.count("ink_dmoe_dn9_nvf4_gaudi2") != 0);
        const bool nv_vmoe_ok = !m_.cfg_.has_nvfp4() ||
                                (isa_.count("ink_vmoe_up_nvf4_gaudi2") != 0 &&
                                 isa_.count("ink_vmoe_dn_nvf4_gaudi2") != 0);
        const bool mx_dmoe_ok = !m_.cfg_.has_mxfp8() ||
                                (isa_.count("ink_dmoe_up9_mxf8_gaudi2") != 0 &&
                                 isa_.count("ink_dmoe_dn9_mxf8_gaudi2") != 0);
        const bool mx_vmoe_ok = !m_.cfg_.has_mxfp8() ||
                                (isa_.count("ink_vmoe_up_mxf8_gaudi2") != 0 &&
                                 isa_.count("ink_vmoe_dn_mxf8_gaudi2") != 0);
        b_decode_ok_ = d_full_on_ && dfar_tiled_ok_ && (mask_ == 15u) && nv_dmoe_ok && mx_dmoe_ok &&
                       isa_.count("ink_dpost_b_gaudi2") && isa_.count("ink_dpost_b_full_gaudi2") &&
                       isa_.count("ink_kv_commit_d1_gaudi2") && isa_.count("ink_dattn_b_gaudi2") &&
                       isa_.count("ink_dattn_b_near_max_gaudi2") && isa_.count("ink_dgate_b_gaudi2") &&
                       isa_.count("ink_dgate_b_full_gaudi2") && isa_.count("ink_dmid_b_gaudi2") &&
                       isa_.count("ink_dtail_rw_b_gaudi2") && isa_.count("ink_dsoftcap_argmax_b_gaudi2") &&
                       mme_.count("v2_proj") && mme_.count("v4_proj") && mme_.count("b8_proj") &&
                       mme_.count("b2_lmhead") && mme_.count("b4_lmhead") && mme_.count("b8_lmhead");
        const char* bme = std::getenv("G2E_INK_BMOE_DEDUP");
        b_moe_dedup_on_ = (!bme || *bme != '0') &&
                          isa_.count("ink_vmoe_plan_gaudi2") != 0 && nv_vmoe_ok && mx_vmoe_ok;
        const char* bde = std::getenv("G2E_INK_B_DMG_FUSE");
        b_dmg_fuse_on_ = (!bde || *bde != '0') && isa_.count("ink_dmg_fgy_gaudi2") != 0;
        const char* bse = std::getenv("G2E_INK_B_SPAN");
        b_span_on_ = !bse || *bse != '0';
        const char* boe = std::getenv("G2E_INK_B_OPT_MME");
        b_opt_mme_on_ = !boe || *boe != '0';
        const char* bk4 = std::getenv("G2E_INK_B_KV4");
        b_kv4_on_ = (!bk4 || *bk4 != '0') && isa_.count("ink_kv_commit_b4_gaudi2") != 0;
        // Only emit idx = -1 for valid == 0 rows when ink_vmoe_plan consumes idx.
        b_route_b_on_ = b_moe_dedup_on_ && isa_.count("ink_droute1_b_gaudi2") != 0;
    }
    init_prefill();
}

std::string InkDirect::summary() const {
    if (!rt_) return "off";
    char b[240];
    std::snprintf(b, sizeof b, "decode mask %u (whole token in one chain: %s, batched decode: %s), prefill mask %u, SRAM scratch: %s",
                  mask_, d_full_on_ ? "yes" : "no", b_decode_ok_ ? "yes" : "no", pmask_, sram_scratch_ ? "yes" : "no");
    return b;
}

InkDirect::~InkDirect() {
    for (auto& [n, v] : chain_prof_)
        std::fprintf(stderr, "[ink-direct] chains of %u applications: %.1f us mean (%.1f us per application) over %llu\n", n,
                     v.first / double(v.second), v.first / double(v.second) / n, (unsigned long long)v.second);
    if (prof_.empty()) return;
    double tot = 0;
    uint64_t n = 0;
    for (auto& [app, v] : prof_) {
        std::fprintf(stderr, "[ink-direct] app %2u: %7.1f us mean over %llu runs\n", app, v.first / double(v.second),
                     (unsigned long long)v.second);
        tot += v.first / double(v.second);
        n++;
    }
    std::fprintf(stderr, "[ink-direct] %llu applications: %.1f us per token\n", (unsigned long long)n, tot);
}

void InkDirect::prepare_constants() {
    const auto& c = m_.cfg_;
    const uint64_t NH = c.heads, D = c.head_dim, DR = c.d_rel, RE = c.rel_extent;
    std::vector<uint8_t> host;
    auto put = [&](const void* p, uint64_t bytes) {
        host.resize(align256(host.size()));
        const uint64_t o = host.size();
        host.insert(host.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + bytes);
        return o;
    };
    auto f32 = [&](const std::string& n) {
        auto raw = m_.w_.read(n);
        std::vector<float> v(raw.size() / 4);
        std::memcpy(v.data(), raw.data(), v.size() * 4);
        return v;
    };
    auto bf16_of = [&](const std::vector<float>& v) {
        std::vector<uint16_t> o(v.size());
        for (size_t i = 0; i < v.size(); ++i) o[i] = to_bf16(v[i]);
        return o;
    };
    for (uint32_t L = 0; L < c.layers; ++L) {
        LayerConst l;
        { auto v = bf16_of(f32(Lw(L, "q_scale"))); l.qsb = put(v.data(), v.size() * 2); }
        {   // k taps then v taps: [2][4][KVD] (one tensor [KVD, 4, 2])
            auto v = bf16_of(f32(Lw(L, "k_taps")));
            const auto vv = bf16_of(f32(Lw(L, "v_taps")));
            v.insert(v.end(), vv.begin(), vv.end());
            l.kvtaps = put(v.data(), v.size() * 2);
        }
        { auto v = bf16_of(f32(Lw(L, "attn_conv"))); l.ataps = put(v.data(), v.size() * 2); }
        { auto v = bf16_of(f32(Lw(L, "mlp_conv"))); l.mtaps = put(v.data(), v.size() * 2); }
        {   // the message gates' output norm weights [3, H]: attention norm x 3 (sliding) or q / k / v norms (full)
            std::vector<uint8_t> v3;
            static const char* const kFull[3] = {"q_norm", "k_norm", "v_norm"};
            static const char* const kSliding[3] = {"attn_norm", "attn_norm", "attn_norm"};
            for (const char* n : c.full[L] ? kFull : kSliding) {
                const auto raw = m_.w_.read(Lw(L, n));
                v3.insert(v3.end(), raw.begin(), raw.end());
            }
            l.wn3 = put(v3.data(), v3.size());
        }
        {   // mlp_norm (bf16 [H]) and router_bias (f32 [E]) packed with small per-layer constants for SRAM caching
            const auto mn = m_.w_.read(Lw(L, "mlp_norm"));
            l.mlp_norm = put(mn.data(), mn.size());
            const auto rb = m_.w_.read(Lw(L, "router_bias"));
            l.router_bias = put(rb.data(), rb.size());
        }
        {   // r_scale [NH, DR] -> per head [even 8 | odd 8]
            const auto rs = f32(Lw(L, "r_scale"));
            std::vector<float> o(NH * 16, 0.f);
            for (uint64_t h = 0; h < NH; ++h)
                for (uint64_t i = 0; i < DR / 2; ++i) {
                    o[h * 16 + i] = rs[h * DR + 2 * i];
                    o[h * 16 + 8 + i] = rs[h * DR + 2 * i + 1];
                }
            l.rs = put(o.data(), o.size() * 4);
        }
        {   // rel_proj [DR, RE] -> rows reversed and transposed: projR[k, d] = proj[d, RE - 1 - k]
            auto raw = m_.w_.read(Lw(L, "rel_proj"));
            std::vector<uint16_t> pj(raw.size() / 2), o(RE * DR);
            std::memcpy(pj.data(), raw.data(), pj.size() * 2);
            for (uint64_t k = 0; k < RE; ++k)
                for (uint64_t d = 0; d < DR; ++d) o[k * DR + d] = pj[d * RE + (RE - 1 - k)];
            l.projr = put(o.data(), o.size() * 2);
        }
        lc_[L] = l;
    }
    uint64_t final_norm_off = 0;
    {
        const auto fn = m_.w_.read("final_norm");
        final_norm_off = put(fn.data(), fn.size());
    }
    // End of small per-layer constants (~2.8 MiB): align to 8 KiB (32 x 256B rows) for ink_pcopy_gaudi2 HBM->SRAM staging.
    host.resize((host.size() + 8191) & ~size_t(8191));
    const_small_bytes_ = host.size();
    for (uint32_t L = c.loop_start; L < c.loop_end; ++L) {
        // mg_{q,k,v,mlp}_fg [2H, R] natural order -> mg_fg_t [4, R, 2H] BF16 for fused TPC ink_dmg_fgy (stays in HBM)
        const uint64_t H2 = 2ull * c.hidden, R = c.gate_rank;
        std::vector<uint16_t> all(4 * R * H2);
        static const char* const kGn[4] = {"mg_q_fg", "mg_k_fg", "mg_v_fg", "mg_mlp_fg"};
        for (uint64_t g = 0; g < 4; ++g) {
            const auto raw = m_.w_.read(Lw(L, kGn[g]));
            std::vector<uint16_t> src(raw.size() / 2);
            std::memcpy(src.data(), raw.data(), raw.size());
            uint16_t* dst = all.data() + g * R * H2;
            for (uint64_t n = 0; n < H2; ++n)
                for (uint64_t k = 0; k < R; ++k) dst[k * H2 + n] = src[n * R + k];
        }
        lc_[L].mg_fg_t = put(all.data(), all.size() * 2);
    }
    host.resize(align256(host.size()));
    const_ = DeviceBuffer(m_.dev_, host.size());
    HostBuffer hb(m_.dev_, host.size());
    std::memcpy(hb.as<uint8_t>(), host.data(), host.size());
    m_.stream_.upload(hb, 0, const_, 0, host.size());
    // Also populate static expert iota [0..E-1] once so G2E_INK_DROUTE1=0 works without build_tables(1).
    HostBuffer ihb(m_.dev_, c.experts * 4);
    {
        auto* p = ihb.as<float>();
        for (uint32_t e = 0; e < c.experts; ++e) p[e] = float(e);
        m_.stream_.upload(ihb, 0, m_.tables_, m_.tab_.tau + Inkling2::kMaxT * 4, c.experts * 4);
    }
    m_.stream_.synchronize();
    const uint64_t base = const_.address();
    for (auto& [L, l] : lc_) {
        for (uint64_t* f : {&l.qsb, &l.kvtaps, &l.ataps, &l.mtaps, &l.rs, &l.projr, &l.wn3, &l.mlp_norm, &l.router_bias})
            *f += base;
        if (l.mg_fg_t) l.mg_fg_t += base;
    }
    final_norm_hbm_ = base + final_norm_off;
    (void)D;
}

// far rows of a full layer at position pos (linear-cache rows [0, nfar): positions older than the ring's W) and their
// bucket (the far templates' size)
uint32_t InkDirect::far_rows() const {
    const uint32_t W = m_.cfg_.window, pos = m_.pos_;
    return pos >= W ? pos - (W - 1) : 0;
}
uint32_t InkDirect::far_size() const { return m_.far_bucket(far_rows()); }

bool InkDirect::use_tiled_far(uint32_t fr) const {
    if (fr == 0 || !dfar_tiled_ok_) return false;
    if (fr > dfar_tile_thresh_) return true;
    const uint32_t sf = m_.far_bucket(fr);
    return sf > dfar_cap_ || !mme_.count("d_fqk_" + std::to_string(sf));
}

void InkDirect::on_slot_resized(uint32_t slot) {
    flush();
    for (auto it = plans_.begin(); it != plans_.end();) {
        if (((std::get<1>(it->first) >> 1) & 15) == int(slot)) it = plans_.erase(it);
        else ++it;
    }
    for (auto it = plans_v_.begin(); it != plans_v_.end();) {
        if (((std::get<1>(it->first) >> 1) & 15) == int(slot)) it = plans_v_.erase(it);
        else ++it;
    }
    for (auto it = span_plans_.begin(); it != span_plans_.end();) {
        bool hit = false;
        for (const auto& dk : it->first)
            if (dk.seq_slot == slot) { hit = true; break; }
        if (hit) it = span_plans_.erase(it);
        else ++it;
    }
    orders_.clear();
    for (auto it = plans_p_.begin(); it != plans_p_.end();) {
        if ((std::get<1>(it->first) >> 4) == int(slot)) it = plans_p_.erase(it);
        else ++it;
    }
    for (auto it = plans_pb_.begin(); it != plans_pb_.end();) {
        if ((std::get<1>(it->first) >> 4) == int(slot)) it = plans_pb_.erase(it);
        else ++it;
    }
    for (auto it = rollback_.begin(); it != rollback_.end();) {
        if (std::get<3>(it->first) == slot) it = rollback_.erase(it);
        else ++it;
    }
    for (auto it = v_commit_plans_.begin(); it != v_commit_plans_.end();) {
        if (std::get<1>(it->first) == slot) it = v_commit_plans_.erase(it);
        else ++it;
    }
    ++evict_gen_;
}

bool InkDirect::handles(uint32_t app) const {
    if (!mask_ || app >= m_.apps_.size()) return false;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    if (m_.cfg_.is_nvfp4(L) &&
        (!isa_.count("ink_dmoe_up9_nvf4_gaudi2") || !isa_.count("ink_dmoe_dn9_nvf4_gaudi2")))
        return false;
    if (m_.cfg_.is_mxfp8(L) &&
        (!isa_.count("ink_dmoe_up9_mxf8_gaudi2") || !isa_.count("ink_dmoe_dn9_mxf8_gaudi2")))
        return false;
    const bool full = m_.cfg_.full[L], boundary = L % m_.cfg_.attn_res_block == 0, msg = a.pass > 0;
    // bit 1: sliding layers without messages (non-boundary); bit 2: boundary layers (L0, 6, 12, 18) without messages
    if ((mask_ & 1) && !full && !boundary && !msg) return true;
    if ((mask_ & 2) && !full && boundary && !msg) return true;
    // bit 4: sliding message applications (passes 1, 2; boundary ones too when bit 2 is set)
    if ((mask_ & 4) && !full && msg && (!boundary || (mask_ & 2))) return true;
    // bit 8: full-attention layers (with messages when bit 4 is set), 1-shot <= 131K or tiled FlashDecode up to 1M+
    if ((mask_ & 8) && full && (!msg || (mask_ & 4))) {
        const uint32_t fr = far_rows();
        if (fr == 0) return true;
        if (use_tiled_far(fr)) return dfar_tiled_ok_;
        const uint32_t sf = m_.far_bucket(fr);
        return sf <= dfar_cap_ && mme_.count("d_fqk_" + std::to_string(sf)) != 0;
    }
    return false;
}

// A new plan when the runtime's plan memory is full: the pending chain runs, then stale far-bucket plans (only when
// sf != 0), idle verify plans (when !keep_verify), idle prefill plans, inactive decode/batch buckets, and finally
// remaining plans (preserving keep_bi if possible) are evicted. Every eviction of a chain-referenced plan map bumps
// evict_gen_ so callers collecting DirectPlan* chains detect invalidation and rebuild cleanly.
std::unique_ptr<DirectPlan> InkDirect::build_evicting(uint32_t sf, const std::function<std::unique_ptr<DirectPlan>()>& mk,
                                                      int keep_bi, bool keep_verify) {
    auto alloc_failure = [](const std::runtime_error& e) {
        return std::string(e.what()).find("plan alloc failed") != std::string::npos;
    };
    auto try_mk = [&]() -> std::unique_ptr<DirectPlan> {
        try {
            return mk();
        } catch (const std::runtime_error& e) {
            if (!alloc_failure(e)) throw;
            return nullptr;
        }
    };
    if (auto p = try_mk()) return p;
    flush();

    auto evict_map = [&](auto& m, bool all, uint32_t keep_sf) {
        bool dropped = false;
        for (auto it = m.begin(); it != m.end();) {
            const uint32_t ksf = std::get<5>(it->first) & 0xffffffu;
            if (all || (ksf != 0 && ksf != keep_sf)) { it = m.erase(it); dropped = true; }
            else ++it;
        }
        return dropped;
    };
    auto evict_spans_stale = [&](uint32_t keep_sf) {
        bool dropped = false;
        for (auto it = span_plans_.begin(); it != span_plans_.end();) {
            bool stale = false;
            for (const auto& k : it->first) {
                const uint32_t ksf = k.sf_tap & 0xffffffu;
                if (ksf != 0 && ksf != keep_sf) { stale = true; break; }
            }
            if (stale) { it = span_plans_.erase(it); dropped = true; }
            else ++it;
        }
        return dropped;
    };
    auto drop_verify = [&] {
        const bool dropped = !plans_v_.empty() || !rollback_.empty() || !v_embed_plans_.empty() ||
                             !v_head_plans_.empty() || !v_commit_plans_.empty();
        if (dropped) {
            plans_v_.clear();
            rollback_.clear();
            v_embed_plans_.clear();
            v_head_plans_.clear();
            v_commit_plans_.clear();
            ++evict_gen_;
        }
        return dropped;
    };
    auto drop_prefill = [&] {
        const bool dropped = !orders_.empty() || !plans_p_.empty() || !plans_pb_.empty();
        orders_.clear();
        plans_p_.clear();
        plans_pb_.clear();
        return dropped;
    };
    auto drop_prefill_far = [&] {
        bool dropped = false;
        for (auto& pool : pfar_blk_pool_) {
            if (!pool.empty()) { pool.clear(); dropped = true; }
        }
        for (int s = 0; s < 2; ++s) {
            for (int h = 0; h < 8; ++h) {
                if (pfar_load_[s][h]) { pfar_load_[s][h].reset(); dropped = true; }
                if (pfar_store_[s][h]) { pfar_store_[s][h].reset(); dropped = true; }
            }
        }
        return dropped;
    };
    auto drop_b1_decode = [&] {
        const bool dropped = !plans_.empty() || !span_plans_.empty() ||
                             d_embed_plan_[0] || d_embed_plan_[1] || !d_head_plans_.empty();
        if (dropped) {
            plans_.clear();
            span_plans_.clear();
            d_embed_plan_[0].reset();
            d_embed_plan_[1].reset();
            d_head_plans_.clear();
            ++evict_gen_;
        }
        return dropped;
    };
    auto drop_batch = [&](int skip_bi) {
        bool dropped = false;
        for (int bi = 0; bi < 3; ++bi) {
            if (bi == skip_bi) continue;
            if (!b_app_plans_[bi].empty()) dropped = true;
            b_app_plans_[bi].clear();
            b_last_slots_[bi].clear();
            b_last_opt_mask_[bi] = ~uint64_t(0);
            b_head_last_B_[bi] = 0;
            b_head_last_skip_[bi] = 0;
        }
        if (skip_bi < 0 && !b_kv_commit_pool_.empty()) {
            b_kv_commit_pool_.clear();
            dropped = true;
        }
        if (dropped) ++evict_gen_;
        return dropped;
    };
    auto drop_shared_dfar_sram = [&] {
        bool dropped = !dfar_pipe_pool_.empty() || bool(sram_const_copy_plan_);
        dfar_pipe_pool_.clear();
        for (auto& pool : dfar_pool_) {
            if (!pool.empty()) { pool.clear(); dropped = true; }
        }
        sram_const_copy_plan_.reset();
        if (dropped) ++evict_gen_;
        return dropped;
    };

    // 1: drop stale-far-bucket decode/verify/span entries (ONLY when sf != 0!)
    if (sf != 0) {
        const bool d1 = evict_map(plans_, false, sf);
        const bool d2 = evict_map(plans_v_, false, sf);
        const bool d3 = evict_spans_stale(sf);
        if (d1 || d2 || d3) {
            ++evict_gen_;
            if (auto p = try_mk()) return p;
        }
    }

    // 2: if !keep_verify, drop idle verification plans
    if (!keep_verify && drop_verify()) {
        if (auto p = try_mk()) return p;
    }

    // 3: drop idle prefill plans BEFORE touching active decode/verify/batch plans
    if (drop_prefill()) {
        if (auto p = try_mk()) return p;
    }
    if (drop_prefill_far()) {
        if (auto p = try_mk()) return p;
    }

    // 4: drop OTHER inactive buckets first (keeping the active mode's plans intact!)
    if (keep_verify) {
        const bool d1 = drop_b1_decode();
        const bool d2 = drop_shared_dfar_sram();
        const bool d3 = drop_batch(-1);
        if (d1 || d2 || d3) {
            if (auto p = try_mk()) return p;
        }
    } else if (keep_bi >= 0) {
        if (drop_b1_decode()) {
            if (auto p = try_mk()) return p;
        }
        if (drop_batch(keep_bi)) {
            if (auto p = try_mk()) return p;
        }
    } else {
        if (drop_batch(-1)) {
            if (auto p = try_mk()) return p;
        }
    }

    // 5: last resort — drop remaining plans (still bumping evict_gen_), preserving keep_bi if possible
    if (keep_verify) {
        if (sf == 0 && evict_map(plans_v_, false, 0)) {
            ++evict_gen_;
            if (auto p = try_mk()) return p;
        }
        drop_verify();
    } else if (keep_bi >= 0) {
        if (drop_shared_dfar_sram()) {
            if (auto p = try_mk()) return p;
        }
        if (!b_kv_commit_pool_.empty()) {
            b_kv_commit_pool_.clear();
            ++evict_gen_;
            if (auto p = try_mk()) return p;
        }
    } else {
        if (sf == 0) {
            const bool d1 = evict_map(plans_, false, 0);
            const bool d2 = evict_spans_stale(0);
            if (d1 || d2) {
                ++evict_gen_;
                if (auto p = try_mk()) return p;
            }
        }
        if (drop_b1_decode()) {
            if (auto p = try_mk()) return p;
        }
        drop_shared_dfar_sram();
    }
    return mk();
}

DirectPlan& InkDirect::plan(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off,
                            DecodeSplit split, DecCtx dctx, uint32_t active_sf) {
    const uint32_t sf = (split == kDecFull && m_.cfg_.full[m_.apps_[app].layer]) ? far_size() : 0;
    const uint32_t evict_sf = sf ? sf : (active_sf ? active_sf : (split == kDecFull && !use_tiled_far(far_rows()) ? far_size() : 0u));
    const bool tap_app = m_.aux_on_ && (app == 4 || app == 22 || app == 40);
    const int sbits = side | (int(m_.active_slot_) << 1) | (int(split) << 5) |
                      (int(dctx.sram_act) << 7) | (int(dctx.sram_const) << 8);
    const auto key = std::make_tuple(app, sbits, hcur, out_buf->address(), out_off, sf | (uint32_t(tap_app) << 31));
    if (auto it = plans_.find(key); it != plans_.end()) return *it->second;
    auto p = build_evicting(evict_sf, [&] { return build(app, side, hcur, out_buf, out_off, 0, split, dctx); });
    auto& ref = *p;
    plans_[key] = std::move(p);
    return ref;
}

// lim > 0: only the first lim stages, the last one signalling the host (stage timing)
std::unique_ptr<DirectPlan> InkDirect::build(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off, int lim,
                                             DecodeSplit split, DecCtx dctx) {
    auto p = std::make_unique<DirectPlan>(*rt_);
    int nsob = 0;
    if (!append_decode_app(*p, nsob, -1, 0, -1, app, side, hcur, out_buf, out_off, lim, split, false, -1, dctx)) return nullptr;
    p->prepare();
    return p;
}

bool InkDirect::append_decode_app(DirectPlan& P, int& nsob, int wait_in_sob, uint32_t wait_in_tgt, int out_sig_sob,
                                  uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off, int lim,
                                  DecodeSplit split, bool skip_in_gram, int next_fused_app, DecCtx dctx) {
    const auto& c = m_.cfg_;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const bool nvfp4 = c.is_nvfp4(L);
    const uint32_t H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim, W = c.window, RE = c.rel_extent;
    const uint32_t G = NH / KV, KVD = KV * D, DR = c.d_rel, E = c.experts, F = c.moe_inter, J = c.topk;
    const uint32_t QS = NH * D * 2 + NH * DR + NH, nbb = L / c.attn_res_block + 1;
    const bool boundary = L % c.attn_res_block == 0, msg = a.pass > 0, full = c.full[L];
    const auto& l = lc_.at(L);
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    auto wa = [&](const char* part) { return arena + w.at(Lw(L, part)).offset; };
    auto C = [&](uint64_t addr) { return dctx.sram_const ? (sram_const_base_ + (addr - const_.address())) : addr; };
    auto QIL = [&](uint64_t off) { return (dctx.sram_const ? sram_qil_base_ : m_.qil_.address()) + off; };
    auto map_act = [&](const DeviceBuffer* b) -> uint64_t {
        if (!dctx.sram_act) return b->address();
        if (b == &m_.blocks_) return s_.dblocks;
        if (b == &m_.h_[0]) return s_.dh0;
        if (b == &m_.h_[1]) return s_.dh1;
        if (b == &m_.prev_[0]) return s_.dprev0;
        if (b == &m_.prev_[1]) return s_.dprev1;
        return b->address();
    };
    const uint64_t blocks = dctx.sram_act ? s_.dblocks : m_.blocks_.address();
    const uint64_t h_in = dctx.sram_act ? (hcur == 0 ? s_.dh0 : s_.dh1) : m_.h_[hcur].address();
    const uint64_t kvs = m_.kvscratch_.address() + uint64_t(app) * 2 * KVD * Inkling2::kMaxT * 2;
    const uint64_t cin = m_.conv_addr(app, m_.active_slot_, side);
    const uint64_t cout = m_.conv_addr(app, m_.active_slot_, 1 - side);
    const uint64_t kring = m_.kring_addr(app, m_.active_slot_);
    const uint64_t vring = m_.vring_addr(app, m_.active_slot_);
    const uint64_t tab = m_.tables_.address();
    const uint64_t h_out = map_act(out_buf) + out_off;
    const bool src_copy = L + 1 == c.loop_end - c.attn_res_block && a.pass + 1 < c.passes;
    const int tap = !m_.aux_on_ ? -1 : app == 4 ? 0 : app == 22 ? 1 : app == 40 ? 2 : -1;
    const uint64_t prev_dst = dctx.sram_act ? (a.pass == 0 ? s_.dprev1 : s_.dprev0)
                                            : m_.prev_[a.pass == 0 ? 1 : 0].address();
    const uint64_t h_out2 = src_copy ? prev_dst
                            : tap >= 0 ? m_.aux_.address() + uint64_t(tap) * Inkling2::kMaxT * H * 2 : h_out;
    const float scale = 1.f / std::sqrt(float(D));
    const float conv_s = 1.f / std::sqrt(5.f);

    int nst = 0;
    bool done = false, mme_cut = false;
    auto TG = [&](uint64_t isa, std::span<const DirectTpcTensor> t, std::span<const uint32_t> sr, std::array<uint32_t, 5> grid,
                  uint32_t split_dim, uint32_t tpcs, int wsob, uint32_t wtgt, int sig) {
        if (done) return;
        if (lim > 0 && ++nst == lim) { sig = -1; done = true; }
        P.add_tpc_grid(isa, t, sr, grid, split_dim, tpcs, wsob, wtgt, sig);
    };
    auto MM = [&](const MmeTemplate& tm, uint64_t a0, uint64_t b0, uint64_t c0, uint64_t c1, int wsob, uint32_t wtgt, int sig) {
        if (done) return;
        if (lim > 0 && ++nst == lim) { mme_cut = true; done = true; return; }
        P.add_mme(tm, a0, b0, c0, c1, wsob, wtgt, sig);
    };
    const auto bf = DType::BF16, f32 = DType::F32;
    const uint32_t NT = H / 128;
    const float mscale = 1.f / std::sqrt(float(H));
    const uint32_t s0 = boundary ? std::max(nbb - 1, 1u) : nbb;
    const uint64_t apart = boundary ? blocks + uint64_t(nbb - 1) * H * 2 : h_in;
    const uint64_t prev = msg ? (dctx.sram_act ? (a.pass == 1 ? s_.dprev1 : s_.dprev0)
                                               : m_.prev_[a.pass == 1 ? 1 : 0].address()) : 0;
    const uint64_t RH = uint64_t(c.gate_rank) * 2 * H * 2;
    const uint32_t Qa = full ? 3 : 1;
    const uint32_t KR = full ? KVD + KV * 4 : KVD;
    const uint32_t NROW = QS + 2 * KR;
    const uint32_t SF = (split == kDecFull && full) ? m_.far_bucket(far_rows()) : 0;
    uint32_t proj_signals = 0;
    uint64_t in_q = 0, in_k = 0, in_v = 0;
    const bool three = msg || full;
    static const bool dmg_fuse = [] { const char* e = std::getenv("G2E_INK_DMG_FUSE"); return !e || *e != '0'; }();
    static const bool droute1 = [] { const char* e = std::getenv("G2E_INK_DROUTE1"); return !e || *e != '0'; }();

    int s_pv = -1;
    uint32_t pv_signals = 0;
    if (split != kDecPostFar) {
    const int s_in = nsob++;
    if (!msg) {
        const uint64_t q = QIL(m_.qil_off_.at(Lw(L, L == 0 ? "mlp_q" : "attn_q")));
        struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, L == 0 ? 1 : 0};
        const auto s = srf(prm);
        int s_g = wait_in_sob;
        uint32_t g_tgt = wait_in_tgt;
        if (!skip_in_gram) {
            s_g = nsob++;
            g_tgt = NT;
            const DirectTpcTensor t[] = {tt(blocks, bf, {H, 1, s0}), tt(apart, bf, {H}), tt(q, f32, {H, Qa}), tt(s_.p1, f32, {64, NT})};
            TG(isa_.at("ink_dgram_gaudi2"), t, {}, {NT, 1, 1, 1, 1}, 0, 24, wait_in_sob, wait_in_tgt, s_g);
        }
        const DirectTpcTensor t2[] = {tt(s_.p1, f32, {64, NT}), tt(blocks, bf, {H, 1, s0}), tt(apart, bf, {H}),
                                      tt(C(l.wn3), bf, {H, Qa}), tt(s_.xa, bf, {H, 1, Qa})};
        TG(isa_.at("ink_dnorm_gaudi2"), t2, s, {NT, 1, 1, 1, 1}, 0, 24, s_g, g_tgt, s_in);
        in_q = s_.xa;
        in_k = full ? s_.xa + H * 2 : s_.xa;
        in_v = full ? s_.xa + 2 * H * 2 : s_.xa;
    } else {
        const int s_cat = nsob++, s_gd = nsob++, s_gy = nsob++;
        const uint64_t q = QIL(m_.qil_off_.at(Lw(L, "attn_q")));
        const uint64_t mq = QIL(m_.qil_off_.at("msg_q") + uint64_t(L - c.loop_start) * 4 * H * 4);
        int s_g = wait_in_sob;
        uint32_t g_tgt = wait_in_tgt;
        if (!skip_in_gram) {
            s_g = nsob++;
            g_tgt = NT;
            const DirectTpcTensor t[] = {tt(blocks, bf, {H, 1, s0}), tt(apart, bf, {H}), tt(q, f32, {H, Qa}), tt(s_.p1, f32, {64, NT}),
                                         tt(prev, bf, {H, 1, 1}), tt(prev + H * 2, bf, {H}), tt(mq, f32, {H, 4}), tt(s_.p3, f32, {64, NT})};
            TG(isa_.at("ink_dgram_msg_gaudi2"), t, {}, {NT, 1, 1, 1, 1}, 0, 24, wait_in_sob, wait_in_tgt, s_g);
        }
        struct { float scale, eps; int32_t m0, write_msg; } prm{mscale, c.eps, 0, 1};
        const DirectTpcTensor t2[] = {tt(s_.p1, f32, {64, NT}), tt(blocks, bf, {H, 1, s0}), tt(apart, bf, {H}), tt(s_.p3, f32, {64, NT}),
                                      tt(prev, bf, {H, 1, 1}), tt(prev + H * 2, bf, {H}), tt(s_.cat, bf, {2 * H, 1, 3}),
                                      tt(s_.xil, f32, {H, 1, Qa}), tt(s_.msg, f32, {H, 1, 4})};
        const auto s = srf(prm);
        TG(isa_.at("ink_dnorm_msg_gaudi2"), t2, s, {NT, 1, 1, 1, 1}, 0, 24, s_g, g_tgt, s_cat);
        const uint64_t md = arena + w.at(Lw(L, "mg_down")).offset;
        const auto& td = mme_.at("d_mgd3");
        MM(td, s_.cat, md, s_.gr, 0, s_cat, NT, s_gd);
        const uint32_t gd_signals = td.signal_count;
        const int32_t m0 = 0;
        const auto so = srf(m0);
        if (dmg_fuse) {
            const DirectTpcTensor t3[] = {tt(s_.xil, f32, {H, 1, Qa}), tt(s_.msg, f32, {H, 1, 4}),
                                          tt(s_.gr, bf, {c.gate_rank, 3}), tt(l.mg_fg_t, bf, {2 * H, c.gate_rank, 3}),
                                          tt(s_.ymg, f32, {H, 1, 3}), tt(s_.p4, f32, {64, NT})};
            TG(isa_.at("ink_dmg_fgy_gaudi2"), t3, so, {NT, 1, 1, 1, 1}, 0, 24, s_gd, gd_signals, s_gy);
        } else {
            const int s_gf = nsob++;
            const auto& tf = mme_.at("d_mgf3x");
            const uint64_t mf = arena + w.at(Lw(L, "mg_fg")).offset;
            MM(tf, s_.gr, mf, s_.fgv, 0, s_gd, gd_signals, s_gf);
            DirectTpcTensor fg = tt(s_.fgv, f32, {2 * H, 1, 3});
            fg.strides[0] = 1; fg.strides[1] = 2 * H; fg.strides[2] = 3 * 2 * H + 2 * H;
            const DirectTpcTensor t3[] = {tt(s_.xil, f32, {H, 1, Qa}), tt(s_.msg, f32, {H, 1, 4}), fg, tt(s_.ymg, f32, {H, 1, 3}),
                                          tt(s_.p4, f32, {64, NT})};
            TG(isa_.at("ink_dmgy_gaudi2"), t3, so, {NT, 1, 1, 1, 1}, 0, 24, s_gf, tf.signal_count, s_gy);
        }
        const DirectTpcTensor t4[] = {tt(s_.p4, f32, {64, NT}), tt(s_.ymg, f32, {H, 1, 3}), tt(C(l.wn3), bf, {H, 3}), tt(s_.fqkv, bf, {H, 1, 3})};
        const auto se = srf(c.eps);
        TG(isa_.at("ink_dmgn_gaudi2"), t4, se, {NT, 1, 1, 1, 1}, 0, 24, s_gy, NT, s_in);
        in_q = s_.fqkv;
        in_k = s_.fqkv + H * 2;
        in_v = s_.fqkv + 2 * H * 2;
    }
    // 1: projections
    const int s_proj = nsob++;
    if (!msg && !full) {
        const auto& tp = mme_.at("d_proj_s");
        MM(tp, in_q, wa("wx"), s_.row, 0, s_in, NT, s_proj);
        proj_signals = tp.signal_count;
    } else {
        const auto& t3p = mme_.at(full ? "d_proj3f" : "d_proj3s");
        MM(t3p, in_q, wa("wx"), s_.row, 0, s_in, NT, s_proj);
        proj_signals = t3p.signal_count;
        (void)in_k; (void)in_v;
    }
    // 2: q prep, r + sink prep, k / v convs straight into KV ring (+ full linear cache)
    const int s_post = nsob++;
    {
        struct { float thr_q, thr_r, isc, cscale, eps; int32_t off_rel, off_sink, off_k, off_v; } prm{
            c.qk_clip, c.rel_clip, 1.f / scale, full ? 1.f / 3.f : conv_s, c.eps, int32_t(NH * D * 2), int32_t(NH * D * 2 + NH * DR),
            int32_t(three ? NROW + QS : QS), int32_t(three ? 2 * NROW + QS + KR : QS + KR)};
        const uint32_t kvs_d3 = handles_all() ? 1u : 2u;  // dim3 == 1 tells ink_dpost to skip unused kvscratch store
        std::vector<DirectTpcTensor> t = {tt(s_.row, bf, {three ? 3 * NROW : NROW}), tt(C(l.qsb), bf, {D, NH}), tt(s_.q, bf, {D, NH}),
                                          tt(C(l.rs), f32, {16, NH}), tt(s_.r, bf, {16, NH}), tt(s_.sink, f32, {NH}),
                                          tt(cin + m_.conv_off_[0], bf, {KVD, 3, 2}), tt(C(l.kvtaps), bf, {KVD, 4, 2}),
                                          tt(cout + m_.conv_off_[0], bf, {KVD, 3, 2}), tt(tab + m_.tab_.pos32, DType::I32, {1}),
                                          tt(kring, bf, {D, W, KV}), tt(vring, bf, {D, W, KV}), tt(kvs, bf, {D, 1, KV, kvs_d3})};
        if (full) {
            const int s = m_.kv_index_[app];
            t.push_back(tt(tab + m_.tab_.tau, f32, {1}));
            t.push_back(tt(m_.kcache_[s].address(), bf, {D, m_.cap_, KV}));
            t.push_back(tt(m_.vcache_[s].address(), bf, {D, m_.cap_, KV}));
        }
        const auto sr = srf(prm);
        TG(isa_.at(full ? "ink_dpost_full_gaudi2" : "ink_dpost_gaudi2"), t, sr, {2 * NH + 2 * KV, 1, 1, 1, 1}, 0, 24, s_proj,
           proj_signals, s_post);
    }
    // 3: bias table, ring QK, far QK (tq follows tb on the same MME engines -> no duplicate MME wait!)
    const int s_att = nsob++;
    const auto& tb = mme_.at("d_bias16");
    const auto& tq = mme_.at("d_qk16");
    MM(tb, s_.r, C(l.projr), s_.bt, 0, s_post, 24, s_att);
    MM(tq, s_.q, kring, s_.sc, 0, -1, 0, s_att);
    uint32_t att_signals = tb.signal_count + tq.signal_count;
    const MmeTemplate* tfq = SF ? &mme_.at("d_fqk_" + std::to_string(SF)) : nullptr;
    const MmeTemplate* tfv = (SF && !dfar_tpc_pv_on_) ? &mme_.at("d_fpv_" + std::to_string(SF)) : nullptr;
    if (SF) {
        const int s_fq_ign = nsob++;
        const int s = m_.kv_index_[app];
        const uint64_t kl = m_.kcache_[s].address();
        for (uint32_t h = 0; h < KV; ++h)
            MM(*tfq, s_.q + uint64_t(h) * G * D * 2, kl + uint64_t(h) * m_.cap_ * D * 2, s_.sf + uint64_t(h) * G * SF * 2, 0,
               -1, 0, h + 1 == KV ? s_att : s_fq_ign);
        att_signals += tfq->signal_count;
    }
    // 5: probabilities and row sums
    const int s_p = nsob++;
    {
        struct { float scale; int32_t G; } prm{scale, int32_t(G)};
        std::vector<DirectTpcTensor> t = {tt(s_.sc, DType::F16, {W, G, KV}), tt(s_.bt, DType::F16, {RE, NH}), tt(s_.sink, f32, {NH}),
                                          tt(tab + m_.tab_.pos32, DType::I32, {1}), tt(s_.p, bf, {W, G, KV}), tt(s_.sums, f32, {NH})};
        const char* attn_k = "ink_dattn_gaudi2";
        if (split == kDecPreFar) {
            t.push_back(tt(s_.dmax, f32, {NH}));
            attn_k = "ink_dattn_near_max_gaudi2";
        } else if (SF) {
            t.push_back(tt(s_.sf, DType::F16, {SF, G, KV}));
            t.push_back(tt(s_.pf, bf, {SF, G, KV}));
            attn_k = "ink_dattn_far_gaudi2";
        }
        const auto sr = srf(prm);
        TG(isa_.at(attn_k), t, sr, {NH, 1, 1, 1, 1}, 0, 24, s_att, att_signals, s_p);
    }
    // 6: P . V
    const auto& tv = mme_.at("d_pvb");
    if (split == kDecPreFar) {
        MM(tv, s_.p, vring, s_.pv, 0, s_p, NH, out_sig_sob);
        if (mme_cut || (lim > 0 && !done)) return false;
        return true;
    }
    s_pv = nsob++;
    if (SF && dfar_tpc_pv_on_) {
        // Overlap near MME d_pvb with 24-TPC far P.V (both wait on s_p and signal combined SOB s_pv).
        MM(tv, s_.p, vring, s_.pv, 0, s_p, NH, s_pv);
        const int s = m_.kv_index_[app];
        const uint64_t vl = m_.vcache_[s].address();
        struct { int32_t G, cap, col0, nv; } prm{int32_t(G), int32_t(m_.cap_), 0, 0};
        const DirectTpcTensor t[] = {
            tt(s_.pf, bf, {SF, G, KV}),
            tt(vl, bf, {D, 12000000u}),
            tt(s_.pvf_part, f32, {D, G, KV, 3}),
            tt(tab + m_.tab_.pos32, DType::I32, {1}),
        };
        TG(isa_.at("ink_dfar_pv_gaudi2"), t, srf(prm), {24, 1, 1, 1, 1}, 0, 24, s_p, NH, s_pv);
        pv_signals = tv.signal_count + 24u;
    } else {
        const int s_pv_ign = SF ? nsob++ : -1;
        MM(tv, s_.p, vring, s_.pv, 0, s_p, NH, SF ? s_pv_ign : s_pv);
        pv_signals = tv.signal_count;
        if (SF) {
            const int s = m_.kv_index_[app];
            const uint64_t vl = m_.vcache_[s].address();
            for (uint32_t h = 0; h < KV; ++h)
                MM(*tfv, s_.pf + uint64_t(h) * G * SF * 2, vl + uint64_t(h) * m_.cap_ * D * 2, s_.pvf + uint64_t(h) * G * D * 2, 0,
                   -1, 0, h + 1 == KV ? s_pv : s_pv_ign);
            pv_signals = tfv->signal_count;
        }
    }
    }  // split != kDecPostFar
    // 7: / sum, * sigmoid(gate)
    const int s_gate = nsob++;
    {
        struct { int32_t off_gate, G; } prm{int32_t(NH * D), int32_t(G)};
        std::vector<DirectTpcTensor> t;
        const char* gate_k = "ink_dgate_gaudi2";
        if (split == kDecPostFar) {
            t.push_back(tt(s_.o_acc, f32, {D, G, KV}));
            gate_k = "ink_dgate_f32_gaudi2";
        } else {
            t.push_back(tt(s_.pv, bf, {D, G, KV}));
            if (SF) {
                if (dfar_tpc_pv_on_) {
                    t.push_back(tt(s_.pvf_part, f32, {D, G, KV, 3}));
                    gate_k = "ink_dgate_far3_gaudi2";
                } else {
                    t.push_back(tt(s_.pvf, bf, {D, G, KV}));
                    gate_k = "ink_dgate_far_gaudi2";
                }
            }
        }
        t.push_back(tt(s_.sums, f32, {NH}));
        t.push_back(tt(s_.row, bf, {three ? 3 * NROW : NROW}));
        t.push_back(tt(s_.gated, bf, {NH * D}));
        const auto sr = srf(prm);
        const int gw = split == kDecPostFar ? wait_in_sob : s_pv;
        const uint32_t gt = split == kDecPostFar ? wait_in_tgt : pv_signals;
        TG(isa_.at(gate_k), t, sr, {NH, 1, 1, 1, 1}, 0, 24, gw, gt, s_gate);
    }
    // 8: output projection
    const int s_wo = nsob++;
    const auto& to = mme_.at("d_wo");
    MM(to, s_.gated, wa("wo"), s_.a1, 0, s_gate, NH, s_wo);
    // 9: post-attention conv + residual + MLP depth-attention + norm
    const int s_mid = nsob++, s_xn = nsob++;
    {
        const DirectTpcTensor t[] = {tt(s_.a1, bf, {H}), tt(cin + m_.conv_off_[2], bf, {H, 3}), tt(C(l.ataps), bf, {H, 4}), tt(h_in, bf, {H}),
                                     tt(blocks, bf, {H, 1, nbb}), tt(QIL(m_.qil_off_.at(Lw(L, "mlp_q"))), f32, {H, 1}),
                                     tt(cout + m_.conv_off_[2], bf, {H, 3}), tt(s_.res, f32, {H}), tt(s_.p2, f32, {64, NT})};
        struct { float cscale; int32_t has_h; } prm{conv_s, boundary ? 0 : 1};
        const auto s = srf(prm);
        TG(isa_.at("ink_dmid_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 0, 24, s_wo, to.signal_count, s_mid);
    }
    if (!msg) {
        const DirectTpcTensor t[] = {tt(s_.p2, f32, {64, NT}), tt(blocks, bf, {H, 1, nbb}), tt(s_.res, f32, {H}),
                                     tt(C(l.mlp_norm), bf, {H, 1}), tt(s_.xn, bf, {H, 1})};
        struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, 0};
        const auto s = srf(prm);
        TG(isa_.at("ink_dnorm_f32p_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 0, 24, s_mid, NT, s_xn);
    } else {
        const int s_mc = nsob++, s_md = nsob++, s_my = nsob++;
        struct { float scale, eps; int32_t m0, write_msg; } prm{mscale, c.eps, 3, 0};
        const DirectTpcTensor t[] = {tt(s_.p2, f32, {64, NT}), tt(blocks, bf, {H, 1, nbb}), tt(s_.res, f32, {H}), tt(s_.p3, f32, {64, NT}),
                                     tt(prev, bf, {H, 1, 1}), tt(prev + H * 2, bf, {H}), tt(s_.cat, bf, {2 * H, 1, 1}),
                                     tt(s_.mrmix, f32, {H, 1, 1}), tt(s_.msg, f32, {H, 1, 4})};
        const auto s = srf(prm);
        TG(isa_.at("ink_dnorm_msg_f32p_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 0, 24, s_mid, NT, s_mc);
        const uint64_t md = arena + w.at(Lw(L, "mg_down")).offset;
        const auto& td = mme_.at("d_mgd1");
        MM(td, s_.cat, md + 3 * RH, s_.gr, 0, s_mc, NT, s_md);
        const uint32_t md_signals = td.signal_count;
        const int32_t m0 = 3;
        const auto so = srf(m0);
        if (dmg_fuse) {
            const DirectTpcTensor t3[] = {tt(s_.mrmix, f32, {H, 1, 1}), tt(s_.msg, f32, {H, 1, 4}),
                                          tt(s_.gr, bf, {c.gate_rank, 1}), tt(l.mg_fg_t + 3 * RH, bf, {2 * H, c.gate_rank, 1}),
                                          tt(s_.ymg, f32, {H, 1, 1}), tt(s_.p4, f32, {64, NT})};
            TG(isa_.at("ink_dmg_fgy_gaudi2"), t3, so, {NT, 1, 1, 1, 1}, 0, 24, s_md, md_signals, s_my);
        } else {
            const int s_mf = nsob++;
            const auto& tf = mme_.at("d_mgf1");
            const uint64_t mf = arena + w.at(Lw(L, "mg_fg")).offset;
            MM(tf, s_.gr, mf + 3 * RH, s_.fgv, 0, s_md, md_signals, s_mf);
            const DirectTpcTensor t3[] = {tt(s_.mrmix, f32, {H, 1, 1}), tt(s_.msg, f32, {H, 1, 4}), tt(s_.fgv, f32, {2 * H, 1, 1}),
                                          tt(s_.ymg, f32, {H, 1, 1}), tt(s_.p4, f32, {64, NT})};
            TG(isa_.at("ink_dmgy_gaudi2"), t3, so, {NT, 1, 1, 1, 1}, 0, 24, s_mf, tf.signal_count, s_my);
        }
        const auto se = srf(c.eps);
        const DirectTpcTensor t4[] = {tt(s_.p4, f32, {64, NT}), tt(s_.ymg, f32, {H, 1, 1}), tt(C(l.mlp_norm), bf, {H, 1}),
                                      tt(s_.xn, bf, {H, 1, 1})};
        TG(isa_.at("ink_dmgn_gaudi2"), t4, se, {NT, 1, 1, 1, 1}, 0, 24, s_my, NT, s_xn);
    }
    // 10: router, top-8
    const int s_rl = nsob++, s_rt = nsob++;
    {
        const auto& tr = mme_.at("d_router");
        MM(tr, s_.xn, wa("router"), s_.logits, 0, s_xn, NT, s_rl);
        const uint32_t rl_signals = tr.signal_count;
        const auto s = srf(c.route_scale);
        if (droute1) {
            const DirectTpcTensor t[] = {tt(s_.logits, f32, {E, 1}), tt(C(l.router_bias), f32, {E}),
                                         tt(s_.idx, DType::I32, {J, 1}), tt(s_.rw, f32, {J, 1})};
            TG(isa_.at("ink_droute1_gaudi2"), t, s, {1, 1, 1, 1, 1}, 0, 1, s_rl, rl_signals, s_rt);
        } else {
            const DirectTpcTensor t[] = {tt(s_.logits, f32, {E, 1}), tt(C(l.router_bias), f32, {E}),
                                         tt(tab + m_.tab_.tau + Inkling2::kMaxT * 4, f32, {E}), tt(s_.idx, DType::I32, {J, 1}),
                                         tt(s_.rw, f32, {J, 1})};
            TG(isa_.at("ink_route_gaudi2"), t, s, {1, 1, 1, 1, 1}, 0, 24, s_rl, rl_signals, s_rt);
        }
    }
    // 11: routed experts (in-place NVFP4 or MXFP8 GEMV), slot sum
    const int s_up = nsob++, s_dn = nsob++;
    if (nvfp4) {
        if (!isa_.count("ink_dmoe_up9_nvf4_gaudi2") || !isa_.count("ink_dmoe_dn9_nvf4_gaudi2"))
            throw std::runtime_error("inkling2 direct: missing NVFP4 decode MoE kernel objects");
        {
            const DirectTpcTensor t[] = {tt(s_.xn, bf, {H, 1}), tt(s_.idx, DType::I32, {J, 1}),
                                         tt(wa("w13.wq"), DType::U8, {2 * F, H / 2, E}),
                                         tt(wa("w13.bs"), DType::F8, {2 * F, H / 16, E}),
                                         tt(m_.nv_gvec13(L), f32, {2 * F, E}), tt(s_.act, bf, {F, J, 1})};
            const auto s = srf(c.up_softcap);
            TG(isa_.at("ink_dmoe_up9_nvf4_gaudi2"), t, s, {9 * J, 1, 1, 1, 1}, 0, 24, s_rt, 1, s_up);
        }
        {
            const DirectTpcTensor t[] = {tt(s_.act, bf, {F, J, 1}), tt(s_.idx, DType::I32, {J, 1}),
                                         tt(wa("w2.wq"), DType::U8, {H, F / 2, E}),
                                         tt(wa("w2.bs"), DType::F8, {H, F / 16, E}),
                                         tt(m_.nv_gvec2(L), f32, {H, E}), tt(s_.ys, bf, {H, J, 1})};
            TG(isa_.at("ink_dmoe_dn9_nvf4_gaudi2"), t, {}, {9 * J, 1, 1, 1, 1}, 0, 24, s_up, 24, s_dn);
        }
    } else {
        if (!isa_.count("ink_dmoe_up9_mxf8_gaudi2") || !isa_.count("ink_dmoe_dn9_mxf8_gaudi2"))
            throw std::runtime_error("inkling2 direct: missing MXFP8 decode MoE kernel objects");
        {
            const DirectTpcTensor t[] = {tt(s_.xn, bf, {H, 1}), tt(s_.idx, DType::I32, {J, 1}),
                                         tt(wa("w13.wq"), DType::U8, {2 * F, H, E}),
                                         tt(wa("w13.bs"), DType::F8, {2 * F, H / 32, E}),
                                         tt(m_.nv_gvec13(L), f32, {2 * F, E}), tt(s_.act, bf, {F, J, 1})};
            const auto s = srf(c.up_softcap);
            TG(isa_.at("ink_dmoe_up9_mxf8_gaudi2"), t, s, {9 * J, 1, 1, 1, 1}, 0, 24, s_rt, 1, s_up);
        }
        {
            const DirectTpcTensor t[] = {tt(s_.act, bf, {F, J, 1}), tt(s_.idx, DType::I32, {J, 1}),
                                         tt(wa("w2.wq"), DType::U8, {H, F, E}),
                                         tt(wa("w2.bs"), DType::F8, {H, F / 32, E}),
                                         tt(m_.nv_gvec2(L), f32, {H, E}), tt(s_.ys, bf, {H, J, 1})};
            TG(isa_.at("ink_dmoe_dn9_mxf8_gaudi2"), t, {}, {9 * J, 1, 1, 1, 1}, 0, 24, s_up, 24, s_dn);
        }
    }
    // 12: slot sum + MLP conv + h_out = res + m2 (optionally fused with next app's or d_head's input dgram)
    {
        const auto s = srf(conv_s);
        if (next_fused_app >= 0 || next_fused_app == -2) {
            const bool fuse_head = (next_fused_app == -2);
            const bool nmsg = !fuse_head && m_.apps_[uint32_t(next_fused_app)].pass > 0;
            uint32_t ns0 = 4u, nQa = 1u;
            uint64_t nq = 0;
            if (fuse_head) {
                nq = QIL(m_.qil_off_.at("final_q"));
            } else {
                const auto& na = m_.apps_[uint32_t(next_fused_app)];
                const uint32_t nL = na.layer;
                const bool nfull = c.full[nL], nboundary = (nL % c.attn_res_block == 0);
                const uint32_t nnbb = nL / c.attn_res_block + 1;
                ns0 = nboundary ? std::max(nnbb - 1, 1u) : nnbb;
                nQa = nfull ? 3u : 1u;
                nq = QIL(m_.qil_off_.at(Lw(nL, nL == 0 ? "mlp_q" : "attn_q")));
            }
            if (!nmsg) {
                const DirectTpcTensor t[] = {tt(s_.ys, bf, {H, J}), tt(s_.rw, f32, {J}), tt(cin + m_.conv_off_[3], bf, {H, 3}),
                                             tt(C(l.mtaps), bf, {H, 4}), tt(s_.res, f32, {H}), tt(cout + m_.conv_off_[3], bf, {H, 3}),
                                             tt(h_out, bf, {H}), tt(h_out2, bf, {H}),
                                             tt(blocks, bf, {H, 1, ns0}), tt(nq, f32, {H, nQa}), tt(s_.p1, f32, {64, NT})};
                TG(isa_.at("ink_dtail_rw_dgram_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 0, 24, s_dn, 24, out_sig_sob);
            } else {
                const auto& na = m_.apps_[uint32_t(next_fused_app)];
                const uint32_t nL = na.layer;
                const uint64_t nprev = dctx.sram_act ? (na.pass == 1 ? s_.dprev1 : s_.dprev0)
                                                     : m_.prev_[na.pass == 1 ? 1 : 0].address();
                const uint64_t nmq = QIL(m_.qil_off_.at("msg_q") + uint64_t(nL - c.loop_start) * 4 * H * 4);
                const DirectTpcTensor t[] = {tt(s_.ys, bf, {H, J}), tt(s_.rw, f32, {J}), tt(cin + m_.conv_off_[3], bf, {H, 3}),
                                             tt(C(l.mtaps), bf, {H, 4}), tt(s_.res, f32, {H}), tt(cout + m_.conv_off_[3], bf, {H, 3}),
                                             tt(h_out, bf, {H}), tt(h_out2, bf, {H}),
                                             tt(blocks, bf, {H, 1, ns0}), tt(nq, f32, {H, nQa}), tt(s_.p1, f32, {64, NT}),
                                             tt(nprev, bf, {H, 1, 2}), tt(nmq, f32, {H, 4}), tt(s_.p3, f32, {64, NT})};
                TG(isa_.at("ink_dtail_rw_dgram_msg_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 0, 24, s_dn, 24, out_sig_sob);
            }
        } else {
            const DirectTpcTensor t[] = {tt(s_.ys, bf, {H, J}), tt(s_.rw, f32, {J}), tt(cin + m_.conv_off_[3], bf, {H, 3}),
                                         tt(C(l.mtaps), bf, {H, 4}), tt(s_.res, f32, {H}), tt(cout + m_.conv_off_[3], bf, {H, 3}),
                                         tt(h_out, bf, {H}), tt(h_out2, bf, {H})};
            TG(isa_.at("ink_dtail_rw_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 0, 24, s_dn, 24, out_sig_sob);
        }
    }
    if (mme_cut || (lim > 0 && !done)) return false;
    return true;
}

bool InkDirect::handles_all() const {
    for (uint32_t i = 0; i < m_.apps_.size(); ++i)
        if (!handles(i)) return false;
    return true;
}

bool InkDirect::handles_full_decode() const {
    return d_full_on_ && handles_all();
}

DirectPlan& InkDirect::sram_const_copy_plan(int keep_bi, uint32_t active_sf) {
    if (sram_const_copy_plan_) return *sram_const_copy_plan_;
    sram_const_copy_plan_ = build_evicting(active_sf, [&] {
        auto p = std::make_unique<DirectPlan>(*rt_);
        const uint32_t r1 = uint32_t(const_small_bytes_ / 256u);
        const uint32_t r2 = uint32_t(m_.qil_.size() / 256u);
        const DirectTpcTensor t1[] = {
            tt(const_.address(), DType::BF16, {128, r1, 1}),
            tt(sram_const_base_, DType::BF16, {128, r1, 1}),
        };
        const DirectTpcTensor t2[] = {
            tt(m_.qil_.address(), DType::BF16, {128, r2, 1}),
            tt(sram_qil_base_, DType::BF16, {128, r2, 1}),
        };
        const uint32_t g1 = r1 / 32u, g2 = r2 / 32u;
        const uint32_t t1_cnt = std::min(24u, g1);
        p->add_tpc_grid(isa_.at("ink_pcopy_gaudi2"), t1, {}, {1, g1, 1, 1, 1}, 1, 24, -1, 0, 0);
        p->add_tpc_grid(isa_.at("ink_pcopy_gaudi2"), t2, {}, {1, g2, 1, 1, 1}, 1, 24, 0, t1_cnt, -1);
        p->prepare();
        return p;
    }, keep_bi);
    return *sram_const_copy_plan_;
}

void InkDirect::append_d_embed(DirectPlan& P, int out_sig_sob, bool sram_act) {
    const auto& c = m_.cfg_;
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    const DirectTpcTensor t[] = {
        tt(m_.ids_.address(), DType::I32, {1}),
        tt(arena + w.at("embed").offset, DType::BF16, {c.hidden, c.embed_rows}),
        tt(arena + w.at("embed_norm").offset, DType::BF16, {c.hidden}),
        tt(sram_act ? s_.dblocks : m_.blocks_.address(), DType::BF16, {c.hidden, 1}),
    };
    const auto sr = srf(c.eps);
    P.add_tpc_grid(isa_.at("ink_dembed_gaudi2"), t, sr, {1, 1, 1, 1, 1}, 0, 1, -1, 0, out_sig_sob);
}

void InkDirect::append_d_head(DirectPlan& P, int& nsob, int wait_in_sob, uint32_t wait_in_tgt,
                              uint64_t last_addr, bool aux_on, bool want_logits, bool skip_in_gram, DecCtx dctx) {
    const auto& c = m_.cfg_;
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    const uint32_t H = c.hidden, NT = H / 128;
    const float mscale = 1.f / std::sqrt(float(H));
    const uint64_t xn_addr = aux_on ? (m_.aux_.address() + 3ull * Inkling2::kMaxT * H * 2) : s_.xn;
    const uint64_t blocks_addr = dctx.sram_act ? s_.dblocks : m_.blocks_.address();
    const uint64_t q_addr = (dctx.sram_const ? sram_qil_base_ : m_.qil_.address()) + m_.qil_off_.at("final_q");
    const uint64_t fn_addr = dctx.sram_const ? (sram_const_base_ + (final_norm_hbm_ - const_.address())) : final_norm_hbm_;
    const auto bf = DType::BF16, f32 = DType::F32;

    int s_g = wait_in_sob;
    uint32_t g_tgt = wait_in_tgt;
    if (!skip_in_gram) {
        s_g = nsob++;
        g_tgt = NT;
        const DirectTpcTensor t[] = {
            tt(blocks_addr, bf, {H, 1, 4}),
            tt(last_addr, bf, {H}),
            tt(q_addr, f32, {H, 1}),
            tt(s_.p1, f32, {64, NT}),
        };
        P.add_tpc_grid(isa_.at("ink_dgram_gaudi2"), t, {}, {NT, 1, 1, 1, 1}, 0, 24, wait_in_sob, wait_in_tgt, s_g);
    }
    const int s_xn = nsob++;
    {
        struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, 0};
        const auto sr = srf(prm);
        const DirectTpcTensor t[] = {
            tt(s_.p1, f32, {64, NT}),
            tt(blocks_addr, bf, {H, 1, 4}),
            tt(last_addr, bf, {H}),
            tt(fn_addr, bf, {H, 1}),
            tt(xn_addr, bf, {H, 1, 1}),
        };
        P.add_tpc_grid(isa_.at("ink_dnorm_gaudi2"), t, sr, {NT, 1, 1, 1, 1}, 0, 24, s_g, g_tgt, s_xn);
    }
    const int s_lm = nsob++;
    const auto& tm_lm = mme_.at("d_lmhead");
    P.add_mme(tm_lm, xn_addr, arena + w.at("lm_head").offset, m_.logits_.address(), 0, s_xn, NT, s_lm);
    {
        struct { float cap, inv_cap; int32_t V, skip_softcap; } prm{
            c.logit_softcap, 1.f / c.logit_softcap, int32_t(c.vocab), want_logits ? 0 : 1};
        const auto sr = srf(prm);
        const uint64_t mv_addr = d_argmax_.address();
        const uint64_t mi_addr = mv_addr + 64ull * 24 * 4;
        const DirectTpcTensor t[] = {
            tt(m_.logits_.address(), f32, {c.vocab}),
            tt(m_.logits_.address(), f32, {c.vocab}),
            tt(mv_addr, f32, {64, 24}),
            tt(mi_addr, DType::I32, {64, 24}),
        };
        P.add_tpc_grid(isa_.at("ink_dsoftcap_argmax_gaudi2"), t, sr, {24, 1, 1, 1, 1}, 0, 24, s_lm, tm_lm.signal_count, -1);
    }
}

DirectPlan& InkDirect::d_embed_plan(bool sram_act, uint32_t active_sf) {
    auto& slot = d_embed_plan_[sram_act ? 1 : 0];
    if (slot) return *slot;
    slot = build_evicting(active_sf, [&] {
        auto p = std::make_unique<DirectPlan>(*rt_);
        append_d_embed(*p, -1, sram_act);
        p->prepare();
        return p;
    });
    return *slot;
}

DirectPlan& InkDirect::d_head_plan(uint64_t last_addr, bool aux_on, bool want_logits, bool skip_in_gram, DecCtx dctx, uint32_t active_sf) {
    const auto key = std::make_tuple(last_addr, aux_on, want_logits, skip_in_gram, dctx.sram_act, dctx.sram_const);
    if (auto it = d_head_plans_.find(key); it != d_head_plans_.end()) return *it->second;
    auto pl = build_evicting(active_sf, [&] {
        auto p = std::make_unique<DirectPlan>(*rt_);
        int nsob = 0;
        append_d_head(*p, nsob, -1, 0, last_addr, aux_on, want_logits, skip_in_gram, dctx);
        p->prepare();
        return p;
    });
    auto& ref = *pl;
    d_head_plans_[key] = std::move(pl);
    return ref;
}

std::vector<uint32_t> InkDirect::compute_dfar_widths(uint32_t fr, uint32_t cap) const {
    const uint32_t max_tw = std::min<uint32_t>(dfar_cap_, std::max<uint32_t>(dfar_tile_thresh_, 1024u));
    std::vector<uint32_t> out;
    uint32_t col = 0;
    while (col < fr) {
        const uint32_t rem = fr - col;
        uint32_t chosen = kDFarWidths[0];
        for (int k = int(kNumDFarWidths) - 1; k >= 0; --k) {
            const uint32_t tw = kDFarWidths[k];
            if (tw > max_tw || col + tw > cap) continue;
            if (!mme_.count("d_fqk_" + std::to_string(tw)) || (!dfar_tpc_pv_on_ && !mme_.count("d_fpv_" + std::to_string(tw)))) continue;
            if (k == 0 || rem >= tw / 2) { chosen = tw; break; }
        }
        out.push_back(chosen);
        col += chosen;
    }
    return out;
}

void InkDirect::prebuild_dfar_for(uint32_t fr, uint32_t cap, uint32_t want_apps, int keep_bi, uint32_t active_sf) {
    if (fr == 0) return;
    const auto widths = compute_dfar_widths(fr, cap);
    auto width_idx = [&](uint32_t tw) -> uint32_t {
        for (uint32_t k = 0; k < kNumDFarWidths; ++k)
            if (kDFarWidths[k] == tw) return k;
        return 0u;
    };
    auto ensure_tiles = [&](const std::array<uint32_t, kNumDFarWidths>& need) {
        for (uint32_t k = 0; k < kNumDFarWidths; ++k) {
            if (!need[k]) continue;
            const uint32_t want = std::min<uint32_t>(kDFarPool, need[k] * want_apps);
            for (uint32_t guard = 0; dfar_pool_[k].size() < want && guard < want + 4u; ++guard) {
                DFarTileEntry ent;
                ent.plan = build_evicting(active_sf, [&] { return build_dfar_tile(kDFarWidths[k]); }, keep_bi);
                ent.nv = kDFarWidths[k];
                ent.first = 0u;
                dfar_pool_[k].push_back(std::move(ent));
            }
            if (dfar_pool_[k].empty())
                throw std::runtime_error("inkling2 direct: failed to prebuild dfar tile pool");
        }
    };
    if (!dfar_pipe_on_ || widths.size() < 2) {
        std::array<uint32_t, kNumDFarWidths> need{};
        for (uint32_t tw : widths) ++need[width_idx(tw)];
        ensure_tiles(need);
        return;
    }
    // Count how many times each <= kDFarPipeMaxTiles chunk pattern appears per full app, and ensure enough pool slots.
    std::map<std::vector<uint32_t>, uint32_t> chunk_mult;
    std::array<uint32_t, kNumDFarWidths> single_need{};
    for (size_t pos = 0; pos < widths.size();) {
        const size_t cnt = std::min<size_t>(kDFarPipeMaxTiles, widths.size() - pos);
        if (cnt == 1) {
            ++single_need[width_idx(widths[pos])];
        } else {
            std::vector<uint32_t> ch(widths.begin() + pos, widths.begin() + pos + cnt);
            ++chunk_mult[ch];
        }
        pos += cnt;
    }
    for (const auto& [ch, mult] : chunk_mult) {
        const uint32_t want = std::min<uint32_t>(kDFarPipePool, mult * want_apps);
        for (uint32_t guard = 0; dfar_pipe_pool_[ch].size() < want && guard < want + 4u; ++guard) {
            DFarPipeEntry ent;
            ent.plan = build_evicting(active_sf, [&] { return build_dfar_pipe(ch); }, keep_bi);
            ent.first = 0u;
            ent.nvs = ch;
            // Re-index dfar_pipe_pool_[ch] AFTER build_evicting (never hold map node ref across eviction).
            dfar_pipe_pool_[ch].push_back(std::move(ent));
        }
        if (dfar_pipe_pool_[ch].empty())
            throw std::runtime_error("inkling2 direct: failed to prebuild dfar pipe pool");
    }
    ensure_tiles(single_need);
}

std::unique_ptr<DirectPlan> InkDirect::build_dfar_tile(uint32_t tw) {
    const auto& c = m_.cfg_;
    const uint32_t NH = c.heads, KV = c.kv_heads, D = c.head_dim, G = NH / KV;
    const float scale = 1.f / std::sqrt(float(D));
    const auto& tfq = mme_.at("d_fqk_" + std::to_string(tw));
    auto p = std::make_unique<DirectPlan>(*rt_);
    // 0: MME Q . K_tile^T across all KV heads (patch tags: 0..7 = q[h], 10..17 = k[h])
    for (uint32_t h = 0; h < KV; ++h)
        p->add_mme(tfq, s_.q + uint64_t(h) * G * D * 2, s_.sf_buf[0], s_.sf_buf[0] + uint64_t(h) * G * tw * 2, 0,
                   -1, 0, 0, {int(h), int(10 + h), -1, -1});
    // 1: online softmax tile step (patch tags: 30 = dmax, 31 = sums; SRF tag 1 = nv)
    {
        struct { float scale; int32_t G, nv; } prm{scale, int32_t(G), int32_t(tw)};
        const DirectTpcTensor t[] = {
            tt(s_.sf_buf[0], DType::F16, {tw, G, KV}),
            tt(s_.dmax, DType::F32, {NH}),
            tt(s_.sums, DType::F32, {NH}),
            tt(s_.pf_buf[0], DType::BF16, {tw, G, KV}),
            tt(s_.dres_buf[0], DType::F32, {NH}),
        };
        const int ttags[] = {-1, 30, 31, -1, -1};
        const int stags[] = {-1, -1, 1};
        p->add_tpc_grid(isa_.at("ink_dfar_sm_gaudi2"), t, srf(prm), {NH, 1, 1, 1, 1}, 0, 24,
                        0, KV * tfq.signal_count, 1, 0xf, ttags, 0, stags);
    }
    if (dfar_tpc_pv_on_) {
        // 2: 24-TPC P_tile . V_tile (patch tag: 40 = vl; SRF tags: 41 = cap, 42 = col0, 1 = nv)
        {
            struct { int32_t G, cap, col0, nv; } prm{int32_t(G), 1024, 0, int32_t(tw)};
            const DirectTpcTensor t[] = {
                tt(s_.pf_buf[0], DType::BF16, {tw, G, KV}),
                tt(s_.pf_buf[0], DType::BF16, {D, 12000000u}),
                tt(s_.pvf_part, DType::F32, {D, G, KV, 3}),
                tt(m_.tables_.address() + m_.tab_.pos32, DType::I32, {1}),
            };
            const int ttags[] = {-1, 40, -1, -1};
            const int stags[] = {-1, 41, 42, 1};
            p->add_tpc_grid(isa_.at("ink_dfar_pv_gaudi2"), t, srf(prm), {24, 1, 1, 1, 1}, 0, 24,
                            1, NH, 2, 0xf, ttags, 0, stags);
        }
        // 3: F32 split-3 output accumulator (patch tags: 32 = pv_near, 33 = o_acc; SRF tag 2 = init_from_pv)
        {
            struct { int32_t G, init; } prm{int32_t(G), 0};
            const DirectTpcTensor t[] = {
                tt(s_.pv, DType::BF16, {D, G, KV}),
                tt(s_.pvf_part, DType::F32, {D, G, KV, 3}),
                tt(s_.dres_buf[0], DType::F32, {NH}),
                tt(s_.o_acc, DType::F32, {D, G, KV}),
            };
            const int ttags[] = {32, -1, -1, 33};
            const int stags[] = {-1, 2};
            p->add_tpc_grid(isa_.at("ink_dfar_acc3_gaudi2"), t, srf(prm), {NH, 1, 1, 1, 1}, 0, 24,
                            2, 24, -1, 0xf, ttags, 0, stags);
        }
    } else {
        const auto& tfv = mme_.at("d_fpv_" + std::to_string(tw));
        // 2: MME P_tile . V_tile across all KV heads (patch tags: 20..27 = v[h])
        for (uint32_t h = 0; h < KV; ++h)
            p->add_mme(tfv, s_.pf_buf[0] + uint64_t(h) * G * tw * 2, s_.pf_buf[0], s_.pvf_buf[0] + uint64_t(h) * G * D * 2, 0,
                       h == 0 ? 1 : -1, h == 0 ? NH : 0, 2, {-1, int(20 + h), -1, -1});
        // 3: F32 output accumulator (patch tags: 32 = pv_near, 33 = o_acc; SRF tag 2 = init_from_pv)
        {
            struct { int32_t G, init; } prm{int32_t(G), 0};
            const DirectTpcTensor t[] = {
                tt(s_.pv, DType::BF16, {D, G, KV}),
                tt(s_.pvf_buf[0], DType::BF16, {D, G, KV}),
                tt(s_.dres_buf[0], DType::F32, {NH}),
                tt(s_.o_acc, DType::F32, {D, G, KV}),
            };
            const int ttags[] = {32, -1, -1, 33};
            const int stags[] = {-1, 2};
            p->add_tpc_grid(isa_.at("ink_dfar_acc_gaudi2"), t, srf(prm), {NH, 1, 1, 1, 1}, 0, 24,
                            2, KV * tfv.signal_count, -1, 0xf, ttags, 0, stags);
        }
    }
    p->prepare();
    return p;
}

// Double-buffered MME || TPC pipelined multi-tile FlashDecode plan (N = widths.size() in [2 .. kDFarPipeMaxTiles]).
// When dfar_tpc_pv_on_:
//   FQK(0) -> SM(0) -> [FQK(t+1) || PV(t) -> s_both[t] -> ACC3(t) -> SM(t+1)] (t=0..N-2) -> PV(N-1) -> ACC3(N-1).
//   Every MME wait (s_sm[t]) and TPC wait (s_fqk0, s_sm[t], s_both[t], s_pv_last) is strictly causally monotonic ->
//   0 out-of-order fence wakeups and 100% MME FQK || 24-TPC PV overlap!
// Otherwise (MME FPV fallback):
//   MME: FQK(0) -> FQK(1) -> FPV(0) -> FQK(2) -> FPV(1) -> ... -> FQK(N-1) -> FPV(N-2) -> FPV(N-1)
//   TPC: SM(0)  -> SM(1)  -> ACC(0) -> SM(2)  -> ACC(1) -> ... -> SM(N-1)  -> ACC(N-2) -> ACC(N-1)
std::unique_ptr<DirectPlan> InkDirect::build_dfar_pipe(const std::vector<uint32_t>& widths) {
    const uint32_t N = uint32_t(widths.size());
    const auto& c = m_.cfg_;
    const uint32_t NH = c.heads, KV = c.kv_heads, D = c.head_dim, G = NH / KV;
    const float scale = 1.f / std::sqrt(float(D));
    auto p = std::make_unique<DirectPlan>(*rt_);
    int nsob = 0;

    if (dfar_tpc_pv_on_) {
        const int s_fqk0 = nsob++;
        std::vector<int> s_sm(N), s_both(N - 1);
        for (uint32_t t = 0; t < N; ++t) s_sm[t] = nsob++;
        for (uint32_t t = 0; t + 1 < N; ++t) s_both[t] = nsob++;
        const int s_pv_last = nsob++;
        const int s_ign = nsob++;

        auto emit_fqk = [&](uint32_t t, int wait_sob, uint32_t wait_tgt, int sig_sob) {
            const uint32_t tw = widths[t];
            const uint32_t buf = t & 1u;
            const auto& tfq = mme_.at("d_fqk_" + std::to_string(tw));
            for (uint32_t h = 0; h < KV; ++h)
                p->add_mme(tfq, s_.q + uint64_t(h) * G * D * 2, s_.sf_buf[buf],
                           s_.sf_buf[buf] + uint64_t(h) * G * tw * 2, 0,
                           h == 0 ? wait_sob : -1, h == 0 ? wait_tgt : 0, sig_sob,
                           {int(h), int(1000 + t * 8 + h), -1, -1});
        };
        auto emit_sm = [&](uint32_t t, int wait_sob, uint32_t wait_tgt) {
            const uint32_t tw = widths[t];
            const uint32_t buf = t & 1u;
            struct { float scale; int32_t G, nv; } prm{scale, int32_t(G), int32_t(tw)};
            const DirectTpcTensor ts[] = {
                tt(s_.sf_buf[buf], DType::F16, {tw, G, KV}),
                tt(s_.dmax, DType::F32, {NH}),
                tt(s_.sums, DType::F32, {NH}),
                tt(s_.pf_buf[buf], DType::BF16, {tw, G, KV}),
                tt(s_.dres_buf[buf], DType::F32, {NH}),
            };
            const int ttags[] = {-1, 30, 31, -1, -1};
            const int stags[] = {-1, -1, int(100 + t)};
            p->add_tpc_grid(isa_.at("ink_dfar_sm_gaudi2"), ts, srf(prm), {NH, 1, 1, 1, 1}, 0, 24,
                            wait_sob, wait_tgt, s_sm[t], 0xf, ttags, 0, stags);
        };
        auto emit_pv = [&](uint32_t t, int sig_sob) {
            const uint32_t tw = widths[t];
            const uint32_t buf = t & 1u;
            struct { int32_t G, cap, col0, nv; } prm{int32_t(G), 1024, 0, int32_t(tw)};
            const DirectTpcTensor ts[] = {
                tt(s_.pf_buf[buf], DType::BF16, {tw, G, KV}),
                tt(s_.pf_buf[buf], DType::BF16, {D, 12000000u}),
                tt(s_.pvf_part, DType::F32, {D, G, KV, 3}),
                tt(m_.tables_.address() + m_.tab_.pos32, DType::I32, {1}),
            };
            const int ttags[] = {-1, 40, -1, -1};
            const int stags[] = {-1, 41, int(200 + t), int(100 + t)};
            p->add_tpc_grid(isa_.at("ink_dfar_pv_gaudi2"), ts, srf(prm), {24, 1, 1, 1, 1}, 0, 24,
                            s_sm[t], NH, sig_sob, 0xf, ttags, 0, stags);
        };
        auto emit_acc3 = [&](uint32_t t, int wait_sob, uint32_t wait_tgt, int out_sob) {
            const uint32_t buf = t & 1u;
            struct { int32_t G, init; } prm{int32_t(G), 0};
            const DirectTpcTensor ts[] = {
                tt(s_.pv, DType::BF16, {D, G, KV}),
                tt(s_.pvf_part, DType::F32, {D, G, KV, 3}),
                tt(s_.dres_buf[buf], DType::F32, {NH}),
                tt(s_.o_acc, DType::F32, {D, G, KV}),
            };
            const int ttags[] = {t == 0 ? 32 : -1, -1, -1, 33};
            const int stags[] = {-1, t == 0 ? 2 : -1};
            p->add_tpc_grid(isa_.at("ink_dfar_acc3_gaudi2"), ts, srf(prm), {NH, 1, 1, 1, 1}, 0, 24,
                            wait_sob, wait_tgt, out_sob, 0xf, ttags, 0, stags);
        };

        const auto& tfq0 = mme_.at("d_fqk_" + std::to_string(widths[0]));
        emit_fqk(0, -1, 0, s_fqk0);
        emit_sm(0, s_fqk0, KV * tfq0.signal_count);
        for (uint32_t t = 0; t + 1 < N; ++t) {
            const auto& tfq_next = mme_.at("d_fqk_" + std::to_string(widths[t + 1]));
            emit_fqk(t + 1, s_sm[t], NH, s_both[t]);
            emit_pv(t, s_both[t]);
            emit_acc3(t, s_both[t], KV * tfq_next.signal_count + 24u, s_ign);
            emit_sm(t + 1, -1, 0);
        }
        emit_pv(N - 1, s_pv_last);
        emit_acc3(N - 1, s_pv_last, 24u, -1);
        p->prepare();
        return p;
    }

    std::vector<int> s_fqk(N), s_sm(N), s_fpv(N);
    for (uint32_t t = 0; t < N; ++t) {
        s_fqk[t] = nsob++;
        s_sm[t] = nsob++;
        s_fpv[t] = nsob++;
    }
    const int s_ign = nsob++;

    auto emit_fqk = [&](uint32_t t) {
        const uint32_t tw = widths[t];
        const uint32_t buf = t & 1u;
        const auto& tfq = mme_.at("d_fqk_" + std::to_string(tw));
        for (uint32_t h = 0; h < KV; ++h)
            p->add_mme(tfq, s_.q + uint64_t(h) * G * D * 2, s_.sf_buf[buf],
                       s_.sf_buf[buf] + uint64_t(h) * G * tw * 2, 0,
                       -1, 0, s_fqk[t], {int(h), int(1000 + t * 8 + h), -1, -1});
    };
    auto emit_sm = [&](uint32_t t) {
        const uint32_t tw = widths[t];
        const uint32_t buf = t & 1u;
        const auto& tfq = mme_.at("d_fqk_" + std::to_string(tw));
        struct { float scale; int32_t G, nv; } prm{scale, int32_t(G), int32_t(tw)};
        const DirectTpcTensor ts[] = {
            tt(s_.sf_buf[buf], DType::F16, {tw, G, KV}),
            tt(s_.dmax, DType::F32, {NH}),
            tt(s_.sums, DType::F32, {NH}),
            tt(s_.pf_buf[buf], DType::BF16, {tw, G, KV}),
            tt(s_.dres_buf[buf], DType::F32, {NH}),
        };
        const int ttags[] = {-1, 30, 31, -1, -1};
        const int stags[] = {-1, -1, int(100 + t)};
        p->add_tpc_grid(isa_.at("ink_dfar_sm_gaudi2"), ts, srf(prm), {NH, 1, 1, 1, 1}, 0, 24,
                        s_fqk[t], KV * tfq.signal_count, s_sm[t], 0xf, ttags, 0, stags);
    };
    auto emit_fpv = [&](uint32_t t) {
        const uint32_t tw = widths[t];
        const uint32_t buf = t & 1u;
        const auto& tfv = mme_.at("d_fpv_" + std::to_string(tw));
        for (uint32_t h = 0; h < KV; ++h)
            p->add_mme(tfv, s_.pf_buf[buf] + uint64_t(h) * G * tw * 2, s_.pf_buf[buf],
                       s_.pvf_buf[buf] + uint64_t(h) * G * D * 2, 0,
                       h == 0 ? s_sm[t] : -1, h == 0 ? NH : 0, s_fpv[t],
                       {-1, int(2000 + t * 8 + h), -1, -1});
    };
    auto emit_acc = [&](uint32_t t) {
        const uint32_t tw = widths[t];
        const uint32_t buf = t & 1u;
        const auto& tfv = mme_.at("d_fpv_" + std::to_string(tw));
        struct { int32_t G, init; } prm{int32_t(G), 0};
        const DirectTpcTensor ts[] = {
            tt(s_.pv, DType::BF16, {D, G, KV}),
            tt(s_.pvf_buf[buf], DType::BF16, {D, G, KV}),
            tt(s_.dres_buf[buf], DType::F32, {NH}),
            tt(s_.o_acc, DType::F32, {D, G, KV}),
        };
        const int ttags[] = {t == 0 ? 32 : -1, -1, -1, 33};
        const int stags[] = {-1, t == 0 ? 2 : -1};
        const int out_sob = (t + 1 == N) ? -1 : s_ign;
        p->add_tpc_grid(isa_.at("ink_dfar_acc_gaudi2"), ts, srf(prm), {NH, 1, 1, 1, 1}, 0, 24,
                        s_fpv[t], KV * tfv.signal_count, out_sob, 0xf, ttags, 0, stags);
    };

    emit_fqk(0);
    emit_sm(0);
    emit_fqk(1);
    emit_sm(1);
    for (uint32_t k = 0; k < N; ++k) {
        emit_fpv(k);
        emit_acc(k);
        if (k + 2 < N) {
            emit_fqk(k + 2);
            emit_sm(k + 2);
        }
    }
    p->prepare();
    return p;
}

void InkDirect::run_dfar_tiles(uint32_t app, uint32_t fr, uint32_t seq_slot, uint64_t q_addr, uint64_t pv_addr,
                               uint64_t dmax_addr, uint64_t sums_addr, uint64_t oacc_addr,
                               std::vector<DirectPlan*>& chain, DFarRunState& state, double& dev_us,
                               uint32_t* kvc_used) {
    const auto& c = m_.cfg_;
    const uint32_t KV = c.kv_heads, D = c.head_dim, G = c.heads / KV;
    const uint32_t cap = m_.slot_capacity(seq_slot);
    const uint64_t kl = m_.kcache_addr(app, seq_slot);
    const uint64_t vl = m_.vcache_addr(app, seq_slot);
    const auto widths = compute_dfar_widths(fr, cap);
    auto width_idx = [&](uint32_t tw) -> uint32_t {
        for (uint32_t k = 0; k < kNumDFarWidths; ++k)
            if (kDFarWidths[k] == tw) return k;
        return 0u;
    };
    auto flush_now = [&] {
        dev_us += rt_->run_chain(chain);
        chain.clear();
        state.reset();
        if (kvc_used) *kvc_used = 0;
    };
    auto emit_single = [&](uint32_t tw, uint32_t col, bool first) {
        const uint32_t ti = width_idx(tw);
        if (dfar_pool_[ti].empty())
            throw std::logic_error("inkling2 direct: dfar tile pool was not prebuilt before chain execution");
        if (state.tile_used[ti] >= dfar_pool_[ti].size()) flush_now();
        DFarTileEntry& ent = dfar_pool_[ti][state.tile_used[ti]++];
        DirectPlan* dp = ent.plan.get();
        if (ent.q_addr != q_addr) {
            for (uint32_t h = 0; h < KV; ++h) dp->patch_addr(int(h), q_addr + uint64_t(h) * G * D * 2);
            ent.q_addr = q_addr;
        }
        if (ent.kl != kl || ent.cap != cap || ent.col != col) {
            for (uint32_t h = 0; h < KV; ++h) dp->patch_addr(int(10 + h), kl + (uint64_t(h) * cap + col) * D * 2);
            ent.kl = kl;
        }
        if (dfar_tpc_pv_on_) {
            if (ent.vl != vl) { dp->patch_addr(40, vl); ent.vl = vl; }
            if (ent.cap != cap) dp->patch_srf(41, cap);
            if (ent.col != col) dp->patch_srf(42, col);
        } else if (ent.vl != vl || ent.cap != cap || ent.col != col) {
            for (uint32_t h = 0; h < KV; ++h) dp->patch_addr(int(20 + h), vl + (uint64_t(h) * cap + col) * D * 2);
            ent.vl = vl;
        }
        ent.cap = cap;
        ent.col = col;
        if (ent.dmax != dmax_addr) { dp->patch_addr(30, dmax_addr); ent.dmax = dmax_addr; }
        if (ent.sums != sums_addr) { dp->patch_addr(31, sums_addr); ent.sums = sums_addr; }
        if (ent.pv != pv_addr) { dp->patch_addr(32, pv_addr); ent.pv = pv_addr; }
        if (ent.oacc != oacc_addr) { dp->patch_addr(33, oacc_addr); ent.oacc = oacc_addr; }
        const uint32_t nv = std::min(fr - col, tw);
        if (ent.nv != nv) { dp->patch_srf(1, nv); ent.nv = nv; }
        const uint32_t fv = first ? 1u : 0u;
        if (ent.first != fv) { dp->patch_srf(2, fv); ent.first = fv; }
        chain.push_back(dp);
    };

    uint32_t col = 0;
    bool first = true;
    if (!dfar_pipe_on_ || widths.size() < 2) {
        for (uint32_t tw : widths) {
            emit_single(tw, col, first);
            col += tw;
            first = false;
        }
        return;
    }
    for (size_t pos = 0; pos < widths.size();) {
        const size_t cnt = std::min<size_t>(kDFarPipeMaxTiles, widths.size() - pos);
        if (cnt == 1) {
            emit_single(widths[pos], col, first);
            col += widths[pos];
            first = false;
            pos += 1;
            continue;
        }
        std::vector<uint32_t> ch(widths.begin() + pos, widths.begin() + pos + cnt);
        auto pit = dfar_pipe_pool_.find(ch);
        if (pit == dfar_pipe_pool_.end() || pit->second.empty())
            throw std::logic_error("inkling2 direct: dfar pipe pool was not prebuilt before chain execution");
        auto& pool = pit->second;
        if (state.pipe_used[ch] >= pool.size()) flush_now();
        DFarPipeEntry& ent = pool[state.pipe_used[ch]++];
        DirectPlan* dp = ent.plan.get();
        if (ent.q_addr != q_addr) {
            for (uint32_t h = 0; h < KV; ++h) dp->patch_addr(int(h), q_addr + uint64_t(h) * G * D * 2);
            ent.q_addr = q_addr;
        }
        if (ent.kl != kl || ent.cap != cap || ent.col0 != col) {
            uint32_t cc = col;
            for (uint32_t t = 0; t < cnt; ++t) {
                for (uint32_t h = 0; h < KV; ++h)
                    dp->patch_addr(int(1000 + t * 8 + h), kl + (uint64_t(h) * cap + cc) * D * 2);
                cc += ch[t];
            }
            ent.kl = kl;
        }
        if (dfar_tpc_pv_on_) {
            if (ent.vl != vl) { dp->patch_addr(40, vl); ent.vl = vl; }
            if (ent.cap != cap) dp->patch_srf(41, cap);
            if (ent.col0 != col) {
                uint32_t cc = col;
                for (uint32_t t = 0; t < cnt; ++t) {
                    dp->patch_srf(int(200 + t), cc);
                    cc += ch[t];
                }
            }
        } else if (ent.vl != vl || ent.cap != cap || ent.col0 != col) {
            uint32_t cc = col;
            for (uint32_t t = 0; t < cnt; ++t) {
                for (uint32_t h = 0; h < KV; ++h)
                    dp->patch_addr(int(2000 + t * 8 + h), vl + (uint64_t(h) * cap + cc) * D * 2);
                cc += ch[t];
            }
            ent.vl = vl;
        }
        ent.cap = cap;
        ent.col0 = col;
        if (ent.dmax != dmax_addr) { dp->patch_addr(30, dmax_addr); ent.dmax = dmax_addr; }
        if (ent.sums != sums_addr) { dp->patch_addr(31, sums_addr); ent.sums = sums_addr; }
        if (ent.pv != pv_addr) { dp->patch_addr(32, pv_addr); ent.pv = pv_addr; }
        if (ent.oacc != oacc_addr) { dp->patch_addr(33, oacc_addr); ent.oacc = oacc_addr; }
        const uint32_t fv = first ? 1u : 0u;
        if (ent.first != fv) { dp->patch_srf(2, fv); ent.first = fv; }
        uint32_t cc = col;
        for (uint32_t t = 0; t < cnt; ++t) {
            const uint32_t nv = std::min(fr - cc, ch[t]);
            if (ent.nvs[t] != nv) { dp->patch_srf(int(100 + t), nv); ent.nvs[t] = nv; }
            cc += ch[t];
        }
        chain.push_back(dp);
        col = cc;
        first = false;
        pos += cnt;
    }
}

int32_t InkDirect::run_decode_full(int32_t tok, uint32_t pos, float tau0, int side,
                                   std::span<const DecodeAppIO> apps, const DeviceBuffer* last_out, uint64_t last_off,
                                   bool need_head, bool want_logits, std::vector<float>* logits, bool want_argmax,
                                   std::vector<uint16_t>* aux_out, double* out_embed_ms,
                                   double* out_layers_ms, double* out_head_ms) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (!anchor_) { anchor(); sram_const_valid_ = false; }
    // 1: upload 12 bytes (tok, pos32, tau[0]) in ONE Direct SCAL PDMA scatter batch (0 Synapse calls)
    const int32_t pos32 = int32_t(pos);
    const DirectRuntime::PdmaH2D h2d[3] = {
        {&tok, m_.ids_.address(), 4},
        {&pos32, m_.tables_.address() + m_.tab_.pos32, 4},
        {&tau0, m_.tables_.address() + m_.tab_.tau, 4},
    };
    rt_->pdma_h2d_scatter_sync(h2d);
    const auto t1 = clk::now();

    // 2: prepare all plans before executing anything (eviction-safe), then run hardware-chained DirectPlans.
    const auto& c = m_.cfg_;
    const uint32_t NT = c.hidden / 128;
    const uint32_t fr_cur = far_rows();
    const bool tiled_cur = use_tiled_far(fr_cur);
    const uint32_t sf_cur = tiled_cur ? 0u : far_size();
    const uint8_t sslot = uint8_t(m_.active_slot_);
    static const uint32_t dspan = [] { const char* e = std::getenv("G2E_INK_DSPAN"); return e ? uint32_t(std::atoi(e)) : 8u; }();
    static const bool sram_act_env = [] { const char* e = std::getenv("G2E_INK_SRAM_ACT"); return !e || *e != '0'; }();
    static const bool fuse_tg_env = [] { const char* e = std::getenv("G2E_INK_FUSE_TAIL_GRAM"); return !e || *e != '0'; }();
    static const bool dmg_f_env = [] { const char* e = std::getenv("G2E_INK_DMG_FUSE"); return !e || *e != '0'; }();
    const DecCtx dctx{sram_scratch_ && sram_act_env, sram_const_on_};
    if (!m_.direct_prev_) sram_const_valid_ = false;
    const bool need_sram_copy = dctx.sram_const && !sram_const_valid_;
    const bool fuse_tg = fuse_tg_env && isa_.count("ink_dtail_rw_dgram_gaudi2") != 0 &&
                         isa_.count("ink_dtail_rw_dgram_msg_gaudi2") != 0;
    auto map_act_addr = [&](const DeviceBuffer* b) -> uint64_t {
        if (!dctx.sram_act) return b->address();
        if (b == &m_.blocks_) return s_.dblocks;
        if (b == &m_.h_[0]) return s_.dh0;
        if (b == &m_.h_[1]) return s_.dh1;
        if (b == &m_.prev_[0]) return s_.dprev0;
        if (b == &m_.prev_[1]) return s_.dprev1;
        return b->address();
    };
    const uint64_t head_last_addr = map_act_addr(last_out) + last_off;
    std::vector<PendingDecode> decs;
    decs.reserve(apps.size());
    for (const auto& io : apps) {
        const bool is_full = c.full[m_.apps_[io.app].layer];
        const uint32_t sf = (is_full && !tiled_cur) ? sf_cur : 0;
        const bool tap_app = m_.aux_on_ && (io.app == 4 || io.app == 22 || io.app == 40);
        decs.push_back({DecodeKey{io.app, side, io.hcur, io.out_buf->address(), io.out_off,
                                  sf | (uint32_t(tap_app) << 31), sslot, 0, 0},
                        io.out_buf});
    }
    const size_t n = decs.size();
    auto is_tiled_idx = [&](size_t k) { return tiled_cur && c.full[m_.apps_[decs[k].key.app].layer]; };
    auto can_fuse_pair = [&](uint32_t a0, uint32_t a1) {
        if (!fuse_tg || a1 != a0 + 1) return false;
        const auto& aa = m_.apps_[a0];
        return !(aa.layer + 1 == c.loop_end && aa.pass + 1 < c.passes);
    };
    auto pair_fused = [&](size_t k) -> bool {
        if (!fuse_tg || k + 1 >= n || is_tiled_idx(k) || is_tiled_idx(k + 1)) return false;
        return can_fuse_pair(decs[k].key.app, decs[k + 1].key.app);
    };
    const bool fuse_last_head = fuse_tg && need_head && n > 0 && !is_tiled_idx(n - 1) &&
                                decs[n - 1].key.app + 1 == m_.apps_.size();
    std::vector<uint8_t> skip_in(n, 0);
    std::vector<int> fuse_out(n, -1);
    for (size_t k = 0; k < n; ++k) {
        skip_in[k] = (k > 0 && pair_fused(k - 1)) ? 1u : 0u;
        if (pair_fused(k)) fuse_out[k] = int(decs[k + 1].key.app);
        else if (k + 1 == n && fuse_last_head) fuse_out[k] = -2;
    }
    struct Step {
        DirectPlan* pl = nullptr;
        uint32_t tiled_app = UINT32_MAX;  // if != UINT32_MAX, run_dfar_tiles for this app after `pl` (pre_far)
        DirectPlan* post = nullptr;
    };
    std::vector<Step> steps;
    steps.reserve(24);
    const uint32_t max_tw = (rt_->monitor_count() - 4u - 96u) / 24u;
    const uint32_t max_mw = 48u;
    const uint32_t max_sobs = rt_->sob_count() - 1u;
    const uint16_t ctx_flags = (dctx.sram_act ? 128u : 0u) | (dctx.sram_const ? 256u : 0u);
    bool plans_ready = false;
    for (int attempt = 0; attempt < kMaxPlanAttempts; ++attempt) {
        steps.clear();
        const uint64_t gen0 = evict_gen_;
        bool restarted = false;
        if (tiled_cur) {
            // Pre-build the exact needed dfar_pipe_pool_ / dfar_pool_ entries for all 8 full apps BEFORE collecting pointers.
            prebuild_dfar_for(fr_cur, m_.slot_capacity(m_.active_slot_), 8u, -1, sf_cur);
            if (evict_gen_ != gen0) restarted = true;
        }
        if (!restarted && need_sram_copy) {
            steps.push_back({&sram_const_copy_plan(-1, sf_cur), UINT32_MAX, nullptr});
            if (evict_gen_ != gen0) restarted = true;
        }
        const bool fuse_embed_span0 = (dspan > 1 && n > 0 && !is_tiled_idx(0));
        if (!restarted && !fuse_embed_span0) {
            steps.push_back({&d_embed_plan(dctx.sram_act, sf_cur), UINT32_MAX, nullptr});
            if (evict_gen_ != gen0) restarted = true;
        }
        bool head_fused = false;
        size_t i = 0;
        while (!restarted && i < n) {
            if (is_tiled_idx(i)) {
                const auto& pd = decs[i];
                DirectPlan* pre = &plan(pd.key.app, pd.key.side, pd.key.hcur, pd.out_buf, pd.key.out_off, kDecPreFar, dctx, sf_cur);
                if (evict_gen_ != gen0) { restarted = true; break; }
                DirectPlan* post = &plan(pd.key.app, pd.key.side, pd.key.hcur, pd.out_buf, pd.key.out_off, kDecPostFar, dctx, sf_cur);
                if (evict_gen_ != gen0) { restarted = true; break; }
                steps.push_back({pre, pd.key.app, post});
                ++i;
                continue;
            }
            const bool with_embed = (i == 0 && fuse_embed_span0);
            size_t j = i;
            uint32_t tpc_waits = 0, mme_waits = 0, sobs = with_embed ? 1u : 0u;
            if (dspan <= 1) {
                j = i + 1;
            } else {
                while (j < n && (j - i) < dspan) {
                    if (is_tiled_idx(j)) break;
                    const bool msg = m_.apps_[decs[j].key.app].pass > 0;
                    const bool first_in_span = (j == i);
                    const bool has_wait_in = !first_in_span || with_embed;
                    const uint32_t base_tw = (msg ? 13u : 9u) +
                                             ((dfar_tpc_pv_on_ && (decs[j].key.sf_tap & 0xffffffu) != 0) ? 1u : 0u);
                    const uint32_t in_tw = (skip_in[j] ? 0u : 1u) + (has_wait_in ? 1u : 0u);
                    const uint32_t need_tw = base_tw + in_tw;
                    const uint32_t need_mw = 5u + (msg ? (dmg_f_env ? 2u : 4u) : 0u);
                    const uint32_t need_sobs = (msg ? 24u : 18u);
                    if (j > i && (tpc_waits + need_tw > max_tw || mme_waits + need_mw > max_mw || sobs + need_sobs > max_sobs))
                        break;
                    tpc_waits += need_tw;
                    mme_waits += need_mw;
                    sobs += need_sobs;
                    ++j;
                }
            }
            const uint32_t head_tw = fuse_last_head ? 2u : 3u;
            const uint32_t head_sobs = fuse_last_head ? 3u : 4u;
            const bool with_head = (j == n && need_head && dspan > 1 &&
                                    tpc_waits + head_tw <= max_tw && mme_waits + 1u <= max_mw && sobs + head_sobs <= max_sobs);
            if (with_head) head_fused = true;
            if (j == i + 1 && !with_embed && !with_head && !skip_in[i] && fuse_out[i] == -1) {
                const auto& pd = decs[i];
                DirectPlan* pl = &plan(pd.key.app, pd.key.side, pd.key.hcur, pd.out_buf, pd.key.out_off, kDecFull, dctx, sf_cur);
                if (evict_gen_ != gen0) { restarted = true; break; }
                steps.push_back({pl, UINT32_MAX, nullptr});
            } else {
                std::vector<DecodeKey> skey;
                skey.reserve(j - i);
                for (size_t k = i; k < j; ++k) {
                    DecodeKey dk = decs[k].key;
                    uint16_t fl = ctx_flags;
                    if (skip_in[k]) fl |= 1u;
                    if (fuse_out[k] >= 0) fl |= 2u;
                    else if (fuse_out[k] == -2) fl |= 4u;
                    if (with_embed && k == i) fl |= 8u;
                    if (with_head && k + 1 == j) {
                        fl |= 16u;
                        if (want_logits) fl |= 32u;
                        if (m_.aux_on_) fl |= 64u;
                    }
                    dk.flags = fl;
                    skey.push_back(dk);
                }
                auto it = span_plans_.find(skey);
                if (it == span_plans_.end()) {
                    const uint32_t sf0 = sf_cur;
                    auto pl = build_evicting(sf0, [&] {
                        auto p = std::make_unique<DirectPlan>(*rt_);
                        int nsob = 0;
                        int prev_sob = -1;
                        uint32_t prev_tgt = 0;
                        if (with_embed) {
                            const int s_emb = nsob++;
                            append_d_embed(*p, s_emb, dctx.sram_act);
                            prev_sob = s_emb;
                            prev_tgt = 1u;
                        }
                        for (size_t k = i; k < j; ++k) {
                            const auto& pd = decs[k];
                            const bool last = (k + 1 == j);
                            const int out_sob = (last && !with_head) ? -1 : nsob++;
                            append_decode_app(*p, nsob, prev_sob, prev_tgt, out_sob,
                                              pd.key.app, pd.key.side, pd.key.hcur, pd.out_buf, pd.key.out_off, 0, kDecFull,
                                              skip_in[k] != 0, fuse_out[k], dctx);
                            prev_sob = out_sob;
                            prev_tgt = NT;
                        }
                        if (with_head)
                            append_d_head(*p, nsob, prev_sob, prev_tgt, head_last_addr, m_.aux_on_, want_logits,
                                          fuse_last_head, dctx);
                        p->prepare();
                        return p;
                    });
                    it = span_plans_.emplace(std::move(skey), std::move(pl)).first;
                    if (evict_gen_ != gen0) { restarted = true; break; }
                }
                steps.push_back({it->second.get(), UINT32_MAX, nullptr});
            }
            i = j;
        }
        if (!restarted && need_head && !head_fused) {
            DirectPlan* hp = &d_head_plan(head_last_addr, m_.aux_on_, want_logits, fuse_last_head, dctx, sf_cur);
            if (evict_gen_ != gen0) restarted = true;
            else steps.push_back({hp, UINT32_MAX, nullptr});
        }
        if (!restarted) { plans_ready = true; break; }
    }
    if (!plans_ready) throw std::runtime_error("inkling2 direct: run_decode_full plans failed to stabilize after eviction");

    std::vector<DirectPlan*> chain;
    chain.reserve(48);
    DFarRunState dfar_state;
    double us = 0.0;
    for (const auto& st : steps) {
        chain.push_back(st.pl);
        if (st.tiled_app != UINT32_MAX) {
            run_dfar_tiles(st.tiled_app, fr_cur, m_.active_slot_, s_.q, s_.pv, s_.dmax, s_.sums, s_.o_acc,
                           chain, dfar_state, us);
            chain.push_back(st.post);
        }
    }
    if (!chain.empty()) us += rt_->run_chain(chain);
    if (dctx.sram_const) sram_const_valid_ = true;
    static const bool tm = std::getenv("G2E_INK_DIRECT_TIME") != nullptr;
    if (tm) {
        auto& v = chain_prof_[uint32_t(n)];
        v.first += us;
        v.second++;
    }
    const auto t2 = clk::now();

    // 3: download ONLY what the caller asked for via Direct SCAL PDMA (12 KiB argmax OR 800 KiB logits)
    int32_t best_idx = 0;
    if (want_logits && logits) {
        logits->resize(c.vocab);
        rt_->pdma_d2h_sync(m_.logits_.address(), logits->data(), uint64_t(c.vocab) * 4);
        if (want_argmax) best_idx = int32_t(std::max_element(logits->begin(), logits->end()) - logits->begin());
    } else if (want_argmax) {
        alignas(64) uint8_t buf[2 * 64 * 24 * 4];
        rt_->pdma_d2h_sync(d_argmax_.address(), buf, sizeof(buf));
        const float* mv = reinterpret_cast<const float*>(buf);
        const int32_t* mi = reinterpret_cast<const int32_t*>(buf + 64 * 24 * 4);
        float best_v = -1.0e30f;
        best_idx = 0x7fffffff;
        for (int p = 0; p < 24; ++p) {
            const float v = mv[p * 64];
            const int32_t idx = mi[p * 64];
            if (v > best_v || (v == best_v && idx < best_idx)) {
                best_v = v;
                best_idx = idx;
            }
        }
    }
    if (aux_out) {
        const uint64_t H = c.hidden;
        const size_t base_sz = aux_out->size();
        aux_out->resize(base_sz + 4 * H);
        uint16_t* dst = aux_out->data() + base_sz;
        const DirectRuntime::PdmaD2H d2h[4] = {
            {m_.aux_.address() + 0ull * Inkling2::kMaxT * H * 2, dst + 0 * H, uint32_t(H * 2)},
            {m_.aux_.address() + 1ull * Inkling2::kMaxT * H * 2, dst + 1 * H, uint32_t(H * 2)},
            {m_.aux_.address() + 2ull * Inkling2::kMaxT * H * 2, dst + 2 * H, uint32_t(H * 2)},
            {m_.aux_.address() + 3ull * Inkling2::kMaxT * H * 2, dst + 3 * H, uint32_t(H * 2)},
        };
        rt_->pdma_d2h_scatter_sync(d2h);
    }
    const auto t3 = clk::now();
    if (out_embed_ms) *out_embed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (out_layers_ms) *out_layers_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    if (out_head_ms) *out_head_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    return best_idx;
}

bool InkDirect::handles_decode_batch() const {
    return b_decode_ok_;
}

void InkDirect::append_b_embed(DirectPlan& P, uint32_t Bp, int out_sig_sob) {
    const auto& c = m_.cfg_;
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    const DirectTpcTensor t[] = {
        tt(m_.ids_.address(), DType::I32, {Bp}),
        tt(arena + w.at("embed").offset, DType::BF16, {c.hidden, c.embed_rows}),
        tt(arena + w.at("embed_norm").offset, DType::BF16, {c.hidden}),
        tt(m_.blocks_.address(), DType::BF16, {c.hidden, Bp}),
    };
    const auto sr = srf(c.eps);
    P.add_tpc_grid(isa_.at("ink_dembed_gaudi2"), t, sr, {Bp, 1, 1, 1, 1}, 0, Bp, -1, 0, out_sig_sob);
}

std::unique_ptr<DirectPlan> InkDirect::build_b_embed(uint32_t Bp) {
    auto p = std::make_unique<DirectPlan>(*rt_);
    append_b_embed(*p, Bp, -1);
    p->prepare();
    return p;
}

void InkDirect::append_b_head(DirectPlan& P, int& nsob, int wait_in_sob, uint32_t wait_in_tgt,
                              uint32_t Bp, uint64_t last_addr, bool use_sram_const) {
    const auto& c = m_.cfg_;
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    const uint32_t H = c.hidden, NT = H / 128;
    const float mscale = 1.f / std::sqrt(float(H));
    const uint64_t q_addr = (use_sram_const ? sram_qil_base_ : m_.qil_.address()) + m_.qil_off_.at("final_q");
    const uint64_t fn_addr = use_sram_const ? (sram_const_base_ + (final_norm_hbm_ - const_.address())) : final_norm_hbm_;
    const auto bf = DType::BF16, f32 = DType::F32;

    const int s_g = nsob++, s_xn = nsob++, s_lm = nsob++;
    // 0: final depth-attention Gram/Dot partials over [b0, b1, b2, b3, h] (S0 = 4, Q = 1, T = Bp)
    {
        const DirectTpcTensor t[] = {
            tt(m_.blocks_.address(), bf, {H, Bp, 4}),
            tt(last_addr, bf, {H, Bp}),
            tt(q_addr, f32, {H, 1}),
            tt(b_.p1, f32, {64, NT, Bp}),
        };
        P.add_tpc_grid(isa_.at("ink_dgram_gaudi2"), t, {}, {NT, Bp, 1, 1, 1}, 0, 24, wait_in_sob, wait_in_tgt, s_g);
    }
    // 1: final depth-attention mix + final_norm -> b_.xn [H, Bp, 1] (split along row dim: Bp TPCs)
    {
        struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, 0};
        const auto sr = srf(prm);
        const DirectTpcTensor t[] = {
            tt(b_.p1, f32, {64, NT, Bp}),
            tt(m_.blocks_.address(), bf, {H, Bp, 4}),
            tt(last_addr, bf, {H, Bp}),
            tt(fn_addr, bf, {H, 1}),
            tt(b_.xn, bf, {H, Bp, 1}),
        };
        P.add_tpc_grid(isa_.at("ink_dnorm_gaudi2"), t, sr, {NT, Bp, 1, 1, 1}, 1, 24, s_g, NT, s_xn);
    }
    // 2: LM head [Bp, 2304] x [200058, 2304]^T -> F32 raw logits [Bp, 200058]
    const auto& tm_lm = mme_.at("b" + std::to_string(Bp) + "_lmhead");
    P.add_mme(tm_lm, b_.xn, arena + w.at("lm_head").offset, m_.logits_.address(), 0, s_xn, Bp, s_lm);
    // 3: fused [9/8] Pade logit softcap (optional skip on greedy) + 24-TPC argmax for active_B rows (SRF tags 1, 2)
    {
        struct { float cap, inv_cap; int32_t V, active_B, skip_softcap; } prm{
            c.logit_softcap, 1.f / c.logit_softcap, int32_t(c.vocab), int32_t(Bp), 0};
        const uint64_t mv_addr = b_argmax_.address();
        const uint64_t mi_addr = mv_addr + uint64_t(Bp) * 24 * 64 * 4;
        const DirectTpcTensor t[] = {
            tt(m_.logits_.address(), f32, {c.vocab, Bp}),
            tt(m_.logits_.address(), f32, {c.vocab, Bp}),
            tt(mv_addr, f32, {64, 24, Bp}),
            tt(mi_addr, DType::I32, {64, 24, Bp}),
        };
        const int stags[] = {-1, -1, -1, 1, 2};
        P.add_tpc_grid(isa_.at("ink_dsoftcap_argmax_b_gaudi2"), t, srf(prm), {24, 1, 1, 1, 1}, 0, 24,
                       s_lm, tm_lm.signal_count, -1, 0xf, {}, 0, stags);
    }
}

std::unique_ptr<DirectPlan> InkDirect::build_b_kv_commit() {
    const auto& c = m_.cfg_;
    const uint32_t KV = c.kv_heads, D = c.head_dim;
    auto p = std::make_unique<DirectPlan>(*rt_);
    const DirectTpcTensor t[] = {
        tt(b_.b_kvs, DType::BF16, {D, KV, 2, 8}),
        tt(0, DType::BF16, {D, 12000000u}),
        tt(0, DType::BF16, {D, 12000000u}),
    };
    const int ttags[] = {-1, 1, 2};
    const int stags[] = {1, 2, 3};
    const uint32_t sr[3] = {0, 0, 1024};
    p->add_tpc_grid(isa_.at("ink_kv_commit_d1_gaudi2"), t, sr, {2 * KV, 1, 1, 1, 1}, 0, 2 * KV,
                    -1, 0, -1, 0xf, ttags, 0, stags);
    p->prepare();
    return p;
}

std::unique_ptr<DirectPlan> InkDirect::build_b_kv_commit_b4(uint32_t Bp) {
    const auto& c = m_.cfg_;
    const uint32_t KV = c.kv_heads, D = c.head_dim;
    const uint32_t n_chunks = (Bp <= 4u) ? 1u : 2u;
    auto p = std::make_unique<DirectPlan>(*rt_);
    for (uint32_t ck = 0; ck < n_chunks; ++ck) {
        const uint32_t b0 = ck * 4u;
        const DirectTpcTensor t[] = {
            tt(b_.b_kvs, DType::BF16, {D, KV, 2, 8}),
            tt(0, DType::BF16, {D, 12000000u}), tt(0, DType::BF16, {D, 12000000u}),
            tt(0, DType::BF16, {D, 12000000u}), tt(0, DType::BF16, {D, 12000000u}),
            tt(0, DType::BF16, {D, 12000000u}), tt(0, DType::BF16, {D, 12000000u}),
            tt(0, DType::BF16, {D, 12000000u}), tt(0, DType::BF16, {D, 12000000u}),
        };
        const int ttags[] = {
            -1,
            10 + int(b0 + 0), 20 + int(b0 + 0),
            10 + int(b0 + 1), 20 + int(b0 + 1),
            10 + int(b0 + 2), 20 + int(b0 + 2),
            10 + int(b0 + 3), 20 + int(b0 + 3),
        };
        const int stags[] = {
            -1,
            100 + int(b0 + 0), 200 + int(b0 + 0),
            100 + int(b0 + 1), 200 + int(b0 + 1),
            100 + int(b0 + 2), 200 + int(b0 + 2),
            100 + int(b0 + 3), 200 + int(b0 + 3),
        };
        const int32_t sr[9] = {int32_t(b0), -1, 1024, -1, 1024, -1, 1024, -1, 1024};
        p->add_tpc_grid(isa_.at("ink_kv_commit_b4_gaudi2"), t, srf(sr), {4 * 2 * KV, 1, 1, 1, 1}, 0, 24,
                        -1, 0, -1, 0xf, ttags, 0, stags);
    }
    p->prepare();
    return p;
}

void InkDirect::append_b_app(DirectPlan& P, int& opt_sob, int& nsob, int wait_in_sob, uint32_t wait_in_tgt,
                             int out_sig_sob, uint32_t Bp, uint32_t app, int hcur, uint64_t h_out, DecodeSplit split,
                             bool use_sram_const) {
    const auto& c = m_.cfg_;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const bool nvfp4 = c.is_nvfp4(L);
    const uint32_t H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim, W = c.window, RE = c.rel_extent;
    const uint32_t G = NH / KV, KVD = KV * D, DR = c.d_rel, E = c.experts, F = c.moe_inter, J = c.topk;
    const uint32_t QS = NH * D * 2 + NH * DR + NH, nbb = L / c.attn_res_block + 1;
    const bool boundary = L % c.attn_res_block == 0, msg = a.pass > 0, full = c.full[L];
    const auto& l = lc_.at(L);
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    auto wa = [&](const char* part) { return arena + w.at(Lw(L, part)).offset; };
    auto C = [&](uint64_t hbm_addr) -> uint64_t {
        return use_sram_const ? (sram_const_base_ + (hbm_addr - const_.address())) : hbm_addr;
    };
    auto QIL = [&](uint64_t off) -> uint64_t {
        return (use_sram_const ? sram_qil_base_ : m_.qil_.address()) + off;
    };
    const uint64_t slab = uint64_t(Bp) * H * 2;
    const uint64_t blocks = m_.blocks_.address();
    const uint64_t h_in = m_.h_[hcur].address();
    const bool src_copy = L + 1 == c.loop_end - c.attn_res_block && a.pass + 1 < c.passes;
    const uint64_t h_out2 = src_copy ? m_.prev_[a.pass == 0 ? 1 : 0].address() : h_out;
    const uint64_t conv_app = m_.conv_pool_.address() + uint64_t(app) * m_.conv_app_bytes_;
    const uint32_t conv_stride = uint32_t(m_.conv_slot_bytes_ / 2);
    const uint32_t nss = 2 * m_.max_slots_;
    const uint64_t kr_pool = m_.kring_[m_.ring_index_[app]].address();
    const uint64_t vr_pool = m_.vring_[m_.ring_index_[app]].address();
    const float scale = 1.f / std::sqrt(float(D));
    const float conv_s = 1.f / std::sqrt(5.f);
    const uint32_t NT = H / 128;
    const float mscale = 1.f / std::sqrt(float(H));
    const uint32_t s0 = boundary ? std::max(nbb - 1, 1u) : nbb;
    const uint64_t apart = boundary ? blocks + uint64_t(nbb - 1) * slab : h_in;
    const uint64_t prev = msg ? m_.prev_[a.pass == 1 ? 1 : 0].address() : 0;
    const uint64_t RH = uint64_t(c.gate_rank) * 2 * H * 2;
    const uint32_t Qa = full ? 3 : 1;
    const uint32_t KR = full ? KVD + KV * 4 : KVD;
    const uint32_t NROW = QS + 2 * KR;
    const bool three = msg || full;
    const std::string pre = Bp == 2 ? "v2_" : Bp == 4 ? "v4_" : "b8_";
    auto TM = [&](const char* k) -> const MmeTemplate& { return mme_.at(pre + k); };
    const auto bf = DType::BF16, f32 = DType::F32, f16 = DType::F16;

    int s_pv = -1;
    uint32_t pv_signals = 0;
    if (split != kDecPostFar) {
        // 0: attention input (depth attention + norms; message gates)
        const int s_in = nsob++;
        uint64_t in3 = 0;
        if (!msg) {
            const int s_g = nsob++;
            const uint64_t q = QIL(m_.qil_off_.at(Lw(L, L == 0 ? "mlp_q" : "attn_q")));
            const DirectTpcTensor t[] = {tt(blocks, bf, {H, Bp, s0}), tt(apart, bf, {H, Bp}),
                                         tt(q, f32, {H, Qa}), tt(b_.p1, f32, {64, NT, Bp})};
            P.add_tpc_grid(isa_.at("ink_dgram_gaudi2"), t, {}, {NT, Bp, 1, 1, 1}, 0, 24, wait_in_sob, wait_in_tgt, s_g);
            struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, L == 0 ? 1 : 0};
            const DirectTpcTensor t2[] = {tt(b_.p1, f32, {64, NT, Bp}), tt(blocks, bf, {H, Bp, s0}), tt(apart, bf, {H, Bp}),
                                          tt(C(l.wn3), bf, {H, Qa}), tt(b_.xa, bf, {H, Bp, Qa})};
            P.add_tpc_grid(isa_.at("ink_dnorm_gaudi2"), t2, srf(prm), {NT, Bp, 1, 1, 1}, 1, 24, s_g, NT, s_in);
            in3 = b_.xa;
        } else {
            const int s_g = nsob++, s_cat = nsob++, s_gd = nsob++, s_gy = nsob++;
            const uint64_t q = QIL(m_.qil_off_.at(Lw(L, "attn_q")));
            const uint64_t mq = QIL(m_.qil_off_.at("msg_q") + uint64_t(L - c.loop_start) * 4 * H * 4);
            const DirectTpcTensor t[] = {tt(blocks, bf, {H, Bp, s0}), tt(apart, bf, {H, Bp}), tt(q, f32, {H, Qa}),
                                         tt(b_.p1, f32, {64, NT, Bp}), tt(prev, bf, {H, Bp, 1}), tt(prev + slab, bf, {H, Bp}),
                                         tt(mq, f32, {H, 4}), tt(b_.p3, f32, {64, NT, Bp})};
            P.add_tpc_grid(isa_.at("ink_dgram_msg_gaudi2"), t, {}, {NT, Bp, 1, 1, 1}, 0, 24, wait_in_sob, wait_in_tgt, s_g);
            struct { float scale, eps; int32_t m0, write_msg; } prm{mscale, c.eps, 0, 1};
            const DirectTpcTensor t2[] = {tt(b_.p1, f32, {64, NT, Bp}), tt(blocks, bf, {H, Bp, s0}), tt(apart, bf, {H, Bp}),
                                          tt(b_.p3, f32, {64, NT, Bp}), tt(prev, bf, {H, Bp, 1}), tt(prev + slab, bf, {H, Bp}),
                                          tt(b_.cat, bf, {2 * H, Bp, 3}), tt(b_.xil, f32, {H, Bp, Qa}), tt(b_.msg, f32, {H, Bp, 4})};
            P.add_tpc_grid(isa_.at("ink_dnorm_msg_gaudi2"), t2, srf(prm), {NT, Bp, 1, 1, 1}, 1, 24, s_g, NT, s_cat);
            const auto& td = TM("mgd3");
            const uint64_t md = arena + w.at(Lw(L, "mg_down")).offset;
            P.add_mme(td, b_.cat, md, b_.gr, 0, s_cat, Bp, s_gd);
            const int32_t m0 = 0;
            if (b_dmg_fuse_on_) {
                const DirectTpcTensor t3[] = {tt(b_.xil, f32, {H, Bp, Qa}), tt(b_.msg, f32, {H, Bp, 4}),
                                              tt(b_.gr, bf, {c.gate_rank, Bp, 3}), tt(l.mg_fg_t, bf, {2 * H, c.gate_rank, 3}),
                                              tt(b_.ymg, f32, {H, Bp, 3}), tt(b_.p4, f32, {64, NT, Bp})};
                P.add_tpc_grid(isa_.at("ink_dmg_fgy_gaudi2"), t3, srf(m0), {NT, Bp, 1, 1, 1}, 0, 24,
                               s_gd, td.signal_count, s_gy);
            } else {
                const int s_gf = nsob++;
                const auto& tf = TM("mgf3x");
                const uint64_t mf = arena + w.at(Lw(L, "mg_fg")).offset;
                P.add_mme(tf, b_.gr, mf, b_.fgv, 0, s_gd, td.signal_count, s_gf);
                DirectTpcTensor fg = tt(b_.fgv, f32, {2 * H, Bp, 3});
                fg.strides[0] = 1;
                fg.strides[1] = 3 * 2 * H;
                fg.strides[2] = Bp * 3 * 2 * H + 2 * H;
                const DirectTpcTensor t3[] = {tt(b_.xil, f32, {H, Bp, Qa}), tt(b_.msg, f32, {H, Bp, 4}), fg,
                                              tt(b_.ymg, f32, {H, Bp, 3}), tt(b_.p4, f32, {64, NT, Bp})};
                P.add_tpc_grid(isa_.at("ink_dmgy_gaudi2"), t3, srf(m0), {NT, Bp, 1, 1, 1}, 0, 24,
                               s_gf, tf.signal_count, s_gy);
            }
            const DirectTpcTensor t4[] = {tt(b_.p4, f32, {64, NT, Bp}), tt(b_.ymg, f32, {H, Bp, 3}),
                                          tt(C(l.wn3), bf, {H, 3}), tt(b_.fqkv, bf, {H, Bp, 3})};
            P.add_tpc_grid(isa_.at("ink_dmgn_gaudi2"), t4, srf(c.eps), {NT, Bp, 1, 1, 1}, 0, 24, s_gy, NT, s_in);
            in3 = b_.fqkv;
        }
        // 1: shared QKV/gate/rel/sink projections across all Bp rows
        const int s_proj = nsob++;
        const auto& tp = TM(!three ? "proj" : full ? "proj3f" : "proj3s");
        P.add_mme(tp, in3, wa("wx"), b_.row, 0, s_in, msg ? NT : Bp, s_proj);
        // 2: q prep, r/sink prep, per-slot k/v convs + ring writes (+ b_kvs for full layers)
        const int s_post = nsob++;
        {
            struct { float thr_q, thr_r, isc, cscale, eps; int32_t off_rel, off_sink, off_k, off_v, nrow, three; } prm{
                c.qk_clip, c.rel_clip, 1.f / scale, full ? 1.f / 3.f : conv_s, c.eps,
                int32_t(NH * D * 2), int32_t(NH * D * 2 + NH * DR),
                int32_t(QS), int32_t(QS + KR), int32_t(NROW), three ? 1 : 0};
            DirectTpcTensor cin_t = tt(conv_app + m_.conv_off_[0], bf, {KVD, 3, 2, nss});
            cin_t.strides[3] = conv_stride;
            const DirectTpcTensor t[] = {
                tt(b_.row, bf, {(three ? 3u : 1u) * Bp * NROW}),
                tt(C(l.qsb), bf, {D, NH}),
                tt(b_.q, bf, {D, NH, Bp}),
                tt(C(l.rs), f32, {16, NH}),
                tt(b_.r, bf, {16, NH, Bp}),
                tt(b_.sink, f32, {NH, Bp}),
                cin_t,
                tt(C(l.kvtaps), bf, {KVD, 4, 2}),
                cin_t,
                tt(bmeta_.address(), DType::I32, {8, Bp}),
                tt(kr_pool, bf, {D, W, KV, m_.max_slots_}),
                tt(vr_pool, bf, {D, W, KV, m_.max_slots_}),
                tt(b_.b_kvs, bf, {D, KV, 2, Bp}),
            };
            const uint32_t pts = (2 * NH + 2 * KV) * Bp;
            P.add_tpc_grid(isa_.at(full ? "ink_dpost_b_full_gaudi2" : "ink_dpost_b_gaudi2"), t, srf(prm),
                           {pts, 1, 1, 1, 1}, 0, 24, s_proj, tp.signal_count, s_post);
        }
        // 3: batched relative bias table + per-row patched ring QK (opt_id = b skips dummy rows when B < Bp!)
        const int s_att = opt_sob++;
        const auto& tb = TM("bias");
        const auto& tq = mme_.at("d_qk16");
        P.add_mme(tb, b_.r, C(l.projr), b_.bt, 0, s_post, 24, s_att);
        for (uint32_t b = 0; b < Bp; ++b) {
            if (b_opt_mme_on_ && b > 0) P.set_opt_id(int(b));
            P.add_mme(tq, b_.q + uint64_t(b) * NH * D * 2, kr_pool, b_.sc + uint64_t(b) * KV * G * W * 2, 0,
                      -1, 0, s_att, {-1, 1000 + int(app) * 16 + int(b), -1, -1});
        }
        if (b_opt_mme_on_) P.set_opt_id(-1);
        const uint32_t att_signals = tb.signal_count + Bp * tq.signal_count;
        // 4: batched ring softmax (full: also outputs dmax for Tiled FlashDecode)
        const int s_p = nsob++;
        {
            struct { float scale; int32_t G; } prm{scale, int32_t(G)};
            std::vector<DirectTpcTensor> t = {
                tt(b_.sc, f16, {W, G, KV, Bp}),
                tt(b_.bt, f16, {RE, NH, Bp}),
                tt(b_.sink, f32, {NH, Bp}),
                tt(bmeta_.address(), DType::I32, {8, Bp}),
                tt(b_.p, bf, {W, G, KV, Bp}),
                tt(b_.sums, f32, {NH, Bp}),
            };
            if (split == kDecPreFar) t.push_back(tt(b_.dmax, f32, {NH, Bp}));
            P.add_tpc_grid(isa_.at(split == kDecPreFar ? "ink_dattn_b_near_max_gaudi2" : "ink_dattn_b_gaudi2"), t, srf(prm),
                           {NH * Bp, 1, 1, 1, 1}, 0, 24, s_att, att_signals, s_p);
        }
        // 5: per-row patched ring P . V (opt_id = b skips dummy rows when B < Bp!)
        const auto& tv = mme_.at("d_pvb");
        const int pv_sig = (split == kDecPreFar) ? -1 : (s_pv = opt_sob++);
        for (uint32_t b = 0; b < Bp; ++b) {
            if (b_opt_mme_on_ && b > 0) P.set_opt_id(int(b));
            P.add_mme(tv, b_.p + uint64_t(b) * KV * G * W * 2, vr_pool, b_.pv + uint64_t(b) * NH * D * 2, 0,
                      b == 0 ? s_p : -1, b == 0 ? 24 : 0, pv_sig, {-1, 2000 + int(app) * 16 + int(b), -1, -1});
        }
        if (b_opt_mme_on_) P.set_opt_id(-1);
        if (split == kDecPreFar) return;
        pv_signals = Bp * tv.signal_count;
    }

    // 6: / sums, * sigmoid(gate) (full: per-row mixed near/far via bmeta[2, b] >= W)
    const int s_gate = nsob++;
    {
        struct { int32_t off_gate, G, nrow, W; } prm{int32_t(NH * D), int32_t(G), int32_t(NROW), int32_t(W)};
        std::vector<DirectTpcTensor> t = {tt(b_.pv, bf, {D, G, KV, Bp})};
        if (split == kDecPostFar) {
            t.push_back(tt(b_.o_acc, f32, {D, G, KV, Bp}));
            t.push_back(tt(bmeta_.address(), DType::I32, {8, Bp}));
        }
        t.push_back(tt(b_.sums, f32, {NH, Bp}));
        t.push_back(tt(b_.row, bf, {(three ? 3u : 1u) * Bp * NROW}));
        t.push_back(tt(b_.gated, bf, {NH * D, Bp}));
        P.add_tpc_grid(isa_.at(split == kDecPostFar ? "ink_dgate_b_full_gaudi2" : "ink_dgate_b_gaudi2"), t, srf(prm),
                       {NH * Bp, 1, 1, 1, 1}, 0, 24, split == kDecPostFar ? -1 : s_pv,
                       split == kDecPostFar ? 0 : pv_signals, s_gate);
    }
    // 7: shared output projection wo
    const int s_wo = nsob++;
    const auto& to = TM("wo");
    P.add_mme(to, b_.gated, wa("wo"), b_.a1, 0, s_gate, 24, s_wo);
    // 8: per-slot post-attention conv + residual + MLP depth-attention partials
    const int s_mid = nsob++, s_xn = nsob++;
    {
        DirectTpcTensor ca_t = tt(conv_app + m_.conv_off_[2], bf, {H, 3, nss});
        ca_t.strides[2] = conv_stride;
        const DirectTpcTensor t[] = {
            tt(b_.a1, bf, {H, Bp}), ca_t, tt(C(l.ataps), bf, {H, 4}), tt(h_in, bf, {H, Bp}),
            tt(blocks, bf, {H, Bp, nbb}), tt(QIL(m_.qil_off_.at(Lw(L, "mlp_q"))), f32, {H, 1}),
            ca_t, tt(b_.res, f32, {H, Bp}), tt(b_.p2, f32, {64, NT, Bp}),
            tt(bmeta_.address(), DType::I32, {8, Bp}),
        };
        struct { float cscale; int32_t has_h; } prm{conv_s, boundary ? 0 : 1};
        P.add_tpc_grid(isa_.at("ink_dmid_b_gaudi2"), t, srf(prm), {NT * Bp, 1, 1, 1, 1}, 0, 24,
                       s_wo, to.signal_count, s_mid);
    }
    if (!msg) {
        const DirectTpcTensor t[] = {tt(b_.p2, f32, {64, NT, Bp}), tt(blocks, bf, {H, Bp, nbb}), tt(b_.res, f32, {H, Bp}),
                                     tt(C(l.mlp_norm), bf, {H, 1}), tt(b_.xn, bf, {H, Bp, 1})};
        struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, 0};
        P.add_tpc_grid(isa_.at("ink_dnorm_f32p_gaudi2"), t, srf(prm), {NT, Bp, 1, 1, 1}, 1, 24, s_mid, 24, s_xn);
    } else {
        const int s_mc = nsob++, s_md = nsob++, s_my = nsob++;
        struct { float scale, eps; int32_t m0, write_msg; } prm{mscale, c.eps, 3, 0};
        const DirectTpcTensor t[] = {tt(b_.p2, f32, {64, NT, Bp}), tt(blocks, bf, {H, Bp, nbb}), tt(b_.res, f32, {H, Bp}),
                                     tt(b_.p3, f32, {64, NT, Bp}), tt(prev, bf, {H, Bp, 1}), tt(prev + slab, bf, {H, Bp}),
                                     tt(b_.cat, bf, {2 * H, Bp, 1}), tt(b_.mrmix, f32, {H, Bp, 1}), tt(b_.msg, f32, {H, Bp, 4})};
        P.add_tpc_grid(isa_.at("ink_dnorm_msg_f32p_gaudi2"), t, srf(prm), {NT, Bp, 1, 1, 1}, 1, 24, s_mid, 24, s_mc);
        const auto& td = TM("mgd1");
        const uint64_t md = arena + w.at(Lw(L, "mg_down")).offset;
        P.add_mme(td, b_.cat, md + 3 * RH, b_.gr, 0, s_mc, Bp, s_md);
        const int32_t m0 = 3;
        if (b_dmg_fuse_on_) {
            const DirectTpcTensor t3[] = {tt(b_.mrmix, f32, {H, Bp, 1}), tt(b_.msg, f32, {H, Bp, 4}),
                                          tt(b_.gr, bf, {c.gate_rank, Bp, 1}), tt(l.mg_fg_t + 3 * RH, bf, {2 * H, c.gate_rank, 1}),
                                          tt(b_.ymg, f32, {H, Bp, 1}), tt(b_.p4, f32, {64, NT, Bp})};
            P.add_tpc_grid(isa_.at("ink_dmg_fgy_gaudi2"), t3, srf(m0), {NT, Bp, 1, 1, 1}, 0, 24,
                           s_md, td.signal_count, s_my);
        } else {
            const int s_mf = nsob++;
            const auto& tf = TM("mgf1");
            const uint64_t mf = arena + w.at(Lw(L, "mg_fg")).offset;
            P.add_mme(tf, b_.gr, mf + 3 * RH, b_.fgv, 0, s_md, td.signal_count, s_mf);
            const DirectTpcTensor t3[] = {tt(b_.mrmix, f32, {H, Bp, 1}), tt(b_.msg, f32, {H, Bp, 4}),
                                          tt(b_.fgv, f32, {2 * H, Bp, 1}), tt(b_.ymg, f32, {H, Bp, 1}),
                                          tt(b_.p4, f32, {64, NT, Bp})};
            P.add_tpc_grid(isa_.at("ink_dmgy_gaudi2"), t3, srf(m0), {NT, Bp, 1, 1, 1}, 0, 24,
                           s_mf, tf.signal_count, s_my);
        }
        const DirectTpcTensor t4[] = {tt(b_.p4, f32, {64, NT, Bp}), tt(b_.ymg, f32, {H, Bp, 1}),
                                      tt(C(l.mlp_norm), bf, {H, 1}), tt(b_.xn, bf, {H, Bp, 1})};
        P.add_tpc_grid(isa_.at("ink_dmgn_gaudi2"), t4, srf(c.eps), {NT, Bp, 1, 1, 1}, 0, 24, s_my, NT, s_xn);
    }
    // 9: shared router + top-8 per row (ink_droute1_b sets idx=-1 for valid==0 dummy rows when b_moe_dedup_on_)
    const int s_rl = nsob++, s_rt = nsob++;
    const auto& tr = TM("router");
    P.add_mme(tr, b_.xn, wa("router"), b_.logits, 0, s_xn, msg ? NT : Bp, s_rl);
    if (b_route_b_on_) {
        const DirectTpcTensor t[] = {tt(b_.logits, f32, {E, Bp}), tt(C(l.router_bias), f32, {E}),
                                     tt(bmeta_.address(), DType::I32, {8, Bp}),
                                     tt(b_.idx, DType::I32, {J, Bp}), tt(b_.rw, f32, {J, Bp})};
        P.add_tpc_grid(isa_.at("ink_droute1_b_gaudi2"), t, srf(c.route_scale), {Bp, 1, 1, 1, 1}, 0, Bp,
                       s_rl, tr.signal_count, s_rt);
    } else {
        const DirectTpcTensor t[] = {tt(b_.logits, f32, {E, Bp}), tt(C(l.router_bias), f32, {E}),
                                     tt(b_.idx, DType::I32, {J, Bp}), tt(b_.rw, f32, {J, Bp})};
        P.add_tpc_grid(isa_.at("ink_droute1_gaudi2"), t, srf(c.route_scale), {Bp, 1, 1, 1, 1}, 0, Bp,
                       s_rl, tr.signal_count, s_rt);
    }
    // 10: routed experts (Bp x 8 slots; expert-deduplicated NVFP4 or FP8 GEMVs when b_moe_dedup_on_)
    const int s_up = nsob++, s_dn = nsob++;
    if (b_moe_dedup_on_) {
        const int s_vp = nsob++;
        const uint32_t M = J * Bp;
        {
            const DirectTpcTensor t[] = {tt(b_.idx, DType::I32, {J, Bp}),
                                         tt(b_.grp, DType::I32, {64, 4})};
            P.add_tpc_grid(isa_.at("ink_vmoe_plan_gaudi2"), t, {}, {1, 1, 1, 1, 1}, 0, 24,
                           s_rt, Bp, s_vp);
        }
        if (nvfp4) {
            {
                const DirectTpcTensor t[] = {tt(b_.xn, bf, {H, Bp}),
                                             tt(b_.grp, DType::I32, {64, 4}),
                                             tt(wa("w13.wq"), DType::U8, {2 * F, H / 2, E}),
                                             tt(wa("w13.bs"), DType::F8, {2 * F, H / 16, E}),
                                             tt(m_.nv_gvec13(L), f32, {2 * F, E}),
                                             tt(b_.act, bf, {F, J, Bp})};
                P.add_tpc_grid(isa_.at("ink_vmoe_up_nvf4_gaudi2"), t, srf(c.up_softcap), {9 * M, 1, 1, 1, 1}, 0, 24,
                               s_vp, 1, s_up);
            }
            {
                const DirectTpcTensor t[] = {tt(b_.act, bf, {F, J, Bp}),
                                             tt(b_.grp, DType::I32, {64, 4}),
                                             tt(wa("w2.wq"), DType::U8, {H, F / 2, E}),
                                             tt(wa("w2.bs"), DType::F8, {H, F / 16, E}),
                                             tt(m_.nv_gvec2(L), f32, {H, E}),
                                             tt(b_.ys, bf, {H, J, Bp})};
                P.add_tpc_grid(isa_.at("ink_vmoe_dn_nvf4_gaudi2"), t, {}, {9 * M, 1, 1, 1, 1}, 0, 24,
                               s_up, 24, s_dn);
            }
        } else {
            {
                const DirectTpcTensor t[] = {tt(b_.xn, bf, {H, Bp}),
                                             tt(b_.grp, DType::I32, {64, 4}),
                                             tt(wa("w13.wq"), DType::U8, {2 * F, H, E}),
                                             tt(wa("w13.bs"), DType::F8, {2 * F, H / 32, E}),
                                             tt(m_.nv_gvec13(L), f32, {2 * F, E}),
                                             tt(b_.act, bf, {F, J, Bp})};
                P.add_tpc_grid(isa_.at("ink_vmoe_up_mxf8_gaudi2"), t, srf(c.up_softcap), {9 * M, 1, 1, 1, 1}, 0, 24,
                               s_vp, 1, s_up);
            }
            {
                const DirectTpcTensor t[] = {tt(b_.act, bf, {F, J, Bp}),
                                             tt(b_.grp, DType::I32, {64, 4}),
                                             tt(wa("w2.wq"), DType::U8, {H, F, E}),
                                             tt(wa("w2.bs"), DType::F8, {H, F / 32, E}),
                                             tt(m_.nv_gvec2(L), f32, {H, E}),
                                             tt(b_.ys, bf, {H, J, Bp})};
                P.add_tpc_grid(isa_.at("ink_vmoe_dn_mxf8_gaudi2"), t, {}, {9 * M, 1, 1, 1, 1}, 0, 24,
                               s_up, 24, s_dn);
            }
        }
    } else if (nvfp4) {
        if (!isa_.count("ink_dmoe_up9_nvf4_gaudi2") || !isa_.count("ink_dmoe_dn9_nvf4_gaudi2"))
            throw std::runtime_error("inkling2 direct: missing NVFP4 decode MoE kernel objects");
        {
            const DirectTpcTensor t[] = {tt(b_.xn, bf, {H, Bp}), tt(b_.idx, DType::I32, {J, Bp}),
                                         tt(wa("w13.wq"), DType::U8, {2 * F, H / 2, E}),
                                         tt(wa("w13.bs"), DType::F8, {2 * F, H / 16, E}),
                                         tt(m_.nv_gvec13(L), f32, {2 * F, E}), tt(b_.act, bf, {F, J, Bp})};
            P.add_tpc_grid(isa_.at("ink_dmoe_up9_nvf4_gaudi2"), t, srf(c.up_softcap), {9 * J * Bp, 1, 1, 1, 1}, 0, 24,
                           s_rt, Bp, s_up);
        }
        {
            const DirectTpcTensor t[] = {tt(b_.act, bf, {F, J, Bp}), tt(b_.idx, DType::I32, {J, Bp}),
                                         tt(wa("w2.wq"), DType::U8, {H, F / 2, E}),
                                         tt(wa("w2.bs"), DType::F8, {H, F / 16, E}),
                                         tt(m_.nv_gvec2(L), f32, {H, E}), tt(b_.ys, bf, {H, J, Bp})};
            P.add_tpc_grid(isa_.at("ink_dmoe_dn9_nvf4_gaudi2"), t, {}, {9 * J * Bp, 1, 1, 1, 1}, 0, 24,
                           s_up, 24, s_dn);
        }
    } else {
        if (!isa_.count("ink_dmoe_up9_mxf8_gaudi2") || !isa_.count("ink_dmoe_dn9_mxf8_gaudi2"))
            throw std::runtime_error("inkling2 direct: missing MXFP8 decode MoE kernel objects");
        {
            const DirectTpcTensor t[] = {tt(b_.xn, bf, {H, Bp}), tt(b_.idx, DType::I32, {J, Bp}),
                                         tt(wa("w13.wq"), DType::U8, {2 * F, H, E}),
                                         tt(wa("w13.bs"), DType::F8, {2 * F, H / 32, E}),
                                         tt(m_.nv_gvec13(L), f32, {2 * F, E}), tt(b_.act, bf, {F, J, Bp})};
            P.add_tpc_grid(isa_.at("ink_dmoe_up9_mxf8_gaudi2"), t, srf(c.up_softcap), {9 * J * Bp, 1, 1, 1, 1}, 0, 24,
                           s_rt, Bp, s_up);
        }
        {
            const DirectTpcTensor t[] = {tt(b_.act, bf, {F, J, Bp}), tt(b_.idx, DType::I32, {J, Bp}),
                                         tt(wa("w2.wq"), DType::U8, {H, F, E}),
                                         tt(wa("w2.bs"), DType::F8, {H, F / 32, E}),
                                         tt(m_.nv_gvec2(L), f32, {H, E}), tt(b_.ys, bf, {H, J, Bp})};
            P.add_tpc_grid(isa_.at("ink_dmoe_dn9_mxf8_gaudi2"), t, {}, {9 * J * Bp, 1, 1, 1, 1}, 0, 24,
                           s_up, 24, s_dn);
        }
    }
    // 11: route-weighted slot sum + per-slot MLP conv + residual add
    {
        DirectTpcTensor cm_t = tt(conv_app + m_.conv_off_[3], bf, {H, 3, nss});
        cm_t.strides[2] = conv_stride;
        const DirectTpcTensor t[] = {
            tt(b_.ys, bf, {H, J, Bp}), tt(b_.rw, f32, {J, Bp}), cm_t, tt(C(l.mtaps), bf, {H, 4}),
            tt(b_.res, f32, {H, Bp}), cm_t, tt(h_out, bf, {H, Bp}), tt(h_out2, bf, {H, Bp}),
            tt(bmeta_.address(), DType::I32, {8, Bp}),
        };
        P.add_tpc_grid(isa_.at("ink_dtail_rw_b_gaudi2"), t, srf(conv_s), {NT * Bp, 1, 1, 1, 1}, 0, 24,
                       s_dn, 24, out_sig_sob);
    }
}

void InkDirect::ensure_batch_plans(uint32_t bi) {
    const uint32_t Bp = kBatchSizes[bi];
    const auto& c = m_.cfg_;
    const uint64_t slab = uint64_t(Bp) * c.hidden * 2;
    // Pre-build fallback single-row KV commit pool only when b_kv4_on_ is disabled.
    if (!b_kv4_on_ && b_kv_commit_pool_.empty()) {
        b_kv_commit_pool_.reserve(kBKvCommitPool);
        for (uint32_t p = 0; p < kBKvCommitPool; ++p)
            b_kv_commit_pool_.push_back(build_evicting(0, [&] { return build_b_kv_commit(); }, int(bi)));
    }
    if (!b_app_plans_[bi].empty()) return;

    b_app_plans_[bi].clear();
    b_last_slots_[bi].clear();
    b_last_opt_mask_[bi] = ~uint64_t(0);
    b_head_last_B_[bi] = Bp;
    b_head_last_skip_[bi] = 0u;
    const int keep = int(bi);
    const bool use_sc = sram_const_on_;

    enum BUnitKind : uint8_t { kBEmbed, kBApp, kBHead };
    struct BUnit {
        BUnitKind kind;
        uint32_t app = 0;
        int hcur = 0;
        uint64_t h_out = 0;
        DecodeSplit split = kDecFull;
        bool sync_after = false;
    };
    std::vector<BUnit> units;
    units.reserve(m_.apps_.size() + 12);
    units.push_back({kBEmbed, 0, 0, 0, kDecFull, false});
    int hcur = 0, pside = 0;
    uint64_t last_addr = 0;
    for (const auto& a : m_.apps_) {
        const uint32_t L = a.layer;
        const bool block_end = (L + 1) % c.attn_res_block == 0;
        const bool last_pass = a.pass + 1 == c.passes;
        const DeviceBuffer* out_buf;
        uint64_t out_off = 0;
        if (L + 1 == c.layers) {
            out_buf = &m_.h_[1 - hcur];
        } else if (block_end && L + 1 == c.loop_end && !last_pass) {
            out_buf = &m_.prev_[1 - pside];
            out_off = slab;
        } else if (block_end) {
            out_buf = &m_.blocks_;
            out_off = uint64_t((L + 1) / c.attn_res_block) * slab;
        } else {
            out_buf = &m_.h_[1 - hcur];
        }
        const uint64_t h_out = out_buf->address() + out_off;
        if (c.full[L]) {
            units.push_back({kBApp, a.index, hcur, h_out, kDecPreFar, true});
            units.push_back({kBApp, a.index, hcur, h_out, kDecPostFar, false});
        } else {
            units.push_back({kBApp, a.index, hcur, h_out, kDecFull, false});
        }
        last_addr = h_out;
        if (out_buf == &m_.h_[1 - hcur]) hcur = 1 - hcur;
        if (L + 1 == c.loop_end && !last_pass) pside = 1 - pside;
    }
    units.push_back({kBHead, 0, 0, last_addr, kDecFull, false});

    const uint32_t max_tw = (rt_->monitor_count() - 4u - 96u) / 24u;
    const uint32_t max_mw = 48u;
    const uint32_t max_sobs = rt_->sob_count() - 1u;
    const uint32_t max_opt_sobs = 64u;
    const uint32_t moe_tw = b_moe_dedup_on_ ? 3u : 2u;
    const uint32_t moe_sobs = b_moe_dedup_on_ ? 3u : 2u;
    const uint32_t dmg_mw = b_dmg_fuse_on_ ? 0u : 1u;
    const uint32_t dmg_sobs = b_dmg_fuse_on_ ? 0u : 1u;

    std::vector<BAppPlan> spans;
    spans.reserve(32);
    size_t i = 0;
    while (i < units.size()) {
        size_t j = i;
        uint32_t tw = 0, mw = 0, sobs = 0, opt_sobs = 0;
        while (j < units.size()) {
            const auto& u = units[j];
            const uint32_t hwi = (j > i) ? 1u : 0u;
            uint32_t ntw = 0, nmw = 0, nsb = 0, nopt = 0;
            if (u.kind == kBEmbed) {
                nsb = 1u;
            } else if (u.kind == kBHead) {
                ntw = 2u + hwi;
                nmw = 1u;
                nsb = 3u;
            } else {
                const bool msg = m_.apps_[u.app].pass > 0;
                if (u.split == kDecPreFar) {
                    ntw = (msg ? 5u : 3u) + hwi;
                    nmw = msg ? (4u + dmg_mw) : 3u;
                    nsb = msg ? (10u + dmg_sobs) : 6u;
                    nopt = 1u;
                } else if (u.split == kDecPostFar) {
                    ntw = (msg ? 6u : 4u) + moe_tw;
                    nmw = msg ? (3u + dmg_mw) : 2u;
                    nsb = (msg ? (11u + dmg_sobs) : 6u) + moe_sobs;
                } else {
                    ntw = (msg ? 12u : 8u) + moe_tw + hwi;
                    nmw = msg ? (7u + 2u * dmg_mw) : 5u;
                    nsb = (msg ? (21u + 2u * dmg_sobs) : 12u) + moe_sobs;
                    nopt = 2u;
                }
            }
            const uint32_t handoff_sb = (j > i) ? 1u : 0u;
            if (j > i && (!b_span_on_ || tw + ntw > max_tw || mw + nmw > max_mw ||
                          sobs + nsb + handoff_sb + nopt > max_sobs || opt_sobs + nopt > max_opt_sobs))
                break;
            tw += ntw;
            mw += nmw;
            sobs += nsb + handoff_sb + nopt;
            opt_sobs += nopt;
            ++j;
            if (u.sync_after || !b_span_on_) break;
        }
        BAppPlan entry;
        entry.main = build_evicting(0, [&] {
            auto p = std::make_unique<DirectPlan>(*rt_);
            int opt_sob = 0;
            int nsob = int(opt_sobs);
            int prev_sob = -1;
            uint32_t prev_tgt = 0;
            for (size_t k = i; k < j; ++k) {
                const auto& u = units[k];
                const bool last = (k + 1 == j);
                const int out_sob = last ? -1 : nsob++;
                if (u.kind == kBEmbed) {
                    append_b_embed(*p, Bp, out_sob);
                    prev_sob = out_sob;
                    prev_tgt = Bp;
                } else if (u.kind == kBHead) {
                    append_b_head(*p, nsob, prev_sob, prev_tgt, Bp, u.h_out, use_sc);
                } else {
                    append_b_app(*p, opt_sob, nsob, prev_sob, prev_tgt, out_sob, Bp, u.app, u.hcur, u.h_out,
                                 u.split, use_sc);
                    prev_sob = out_sob;
                    prev_tgt = 24u;
                }
            }
            p->prepare();
            return p;
        }, keep);
        for (size_t k = i; k < j; ++k) {
            const auto& u = units[k];
            if (u.kind == kBApp && u.split != kDecPostFar) {
                entry.ring_apps.push_back(u.app);
                if (b_opt_mme_on_) entry.has_opt_mme = true;
            }
            if (u.kind == kBHead) entry.has_head = true;
        }
        if (units[j - 1].sync_after) {
            entry.sync_full_app = units[j - 1].app;
            if (b_kv4_on_) {
                entry.post = build_evicting(0, [&] { return build_b_kv_commit_b4(Bp); }, keep);
            }
        }
        spans.push_back(std::move(entry));
        i = j;
    }
    b_app_plans_[bi] = std::move(spans);
}

void InkDirect::run_decode_batch(std::span<const uint32_t> slots, std::span<const int32_t> tokens,
                                 std::span<const uint8_t> want_logits, std::vector<int32_t>& out_argmax,
                                 std::vector<std::vector<float>>& out_logits) {
    const uint32_t B = uint32_t(slots.size());
    const uint32_t bi = B <= 2 ? 0u : B <= 4 ? 1u : 2u;
    const uint32_t Bp = kBatchSizes[bi];
    const auto& c = m_.cfg_;
    const uint32_t NH = c.heads, D = c.head_dim;

    // 1: ensure plans for Bp (and shared far/commit/sram_const plans) exist before assembling pointers
    if (!anchor_) { anchor(); sram_const_valid_ = false; }
    if (!m_.direct_prev_) sram_const_valid_ = false;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> fr_cnt;
    for (uint32_t b = 0; b < B; ++b) {
        const uint32_t s = slots[b];
        const uint32_t pos = m_.slot_position(s);
        const uint32_t fr = pos >= c.window ? pos - (c.window - 1) : 0u;
        if (fr > 0) ++fr_cnt[{fr, m_.slot_capacity(s)}];
    }
    DirectPlan* sram_cp = nullptr;
    bool batch_ready = false;
    for (int attempt = 0; attempt < kMaxPlanAttempts; ++attempt) {
        const uint64_t gen0 = evict_gen_;
        sram_cp = nullptr;
        ensure_batch_plans(bi);
        if (evict_gen_ != gen0) continue;
        bool restarted = false;
        for (const auto& [fc, cnt] : fr_cnt) {
            prebuild_dfar_for(fc.first, fc.second, 8u * cnt, int(bi));
            if (evict_gen_ != gen0) { restarted = true; break; }
        }
        if (restarted) continue;
        if (sram_const_on_ && !sram_const_valid_) {
            sram_cp = &sram_const_copy_plan(int(bi));
            if (evict_gen_ != gen0) continue;
        }
        batch_ready = true;
        break;
    }
    if (!batch_ready) throw std::runtime_error("inkling2 direct: run_decode_batch plans failed to stabilize after eviction");

    // 2: upload ids[Bp] + bmeta[8, Bp] in ONE SCAL PDMA scatter batch
    alignas(64) int32_t ids_host[8] = {};
    alignas(64) int32_t bmeta_host[8 * 8] = {};
    std::vector<uint32_t> pad_slots(Bp);
    bool any_logits = false;
    for (uint32_t b = 0; b < Bp; ++b) {
        const bool valid = b < B;
        if (valid && want_logits[b]) any_logits = true;
        const uint32_t s = valid ? slots[b] : slots[0];
        pad_slots[b] = s;
        const uint32_t pos = valid ? m_.slot_position(s) : 0u;
        const int side = valid ? m_.slot_conv_side(s) : 0;
        const double x = double(pos + 1) / double(c.log_floor);
        const float tau = valid ? float(1.0 + double(c.log_alpha) * std::log(std::max(x, 1.0))) : 1.0f;
        int32_t tau_bits = 0;
        std::memcpy(&tau_bits, &tau, 4);
        bmeta_host[b * 8 + 0] = int32_t(s);
        bmeta_host[b * 8 + 1] = side;
        bmeta_host[b * 8 + 2] = int32_t(pos);
        bmeta_host[b * 8 + 3] = valid ? 1 : 0;
        bmeta_host[b * 8 + 4] = tau_bits;
        ids_host[b] = valid ? tokens[b] : tokens[0];
    }
    const DirectRuntime::PdmaH2D h2d[2] = {
        {ids_host, m_.ids_.address(), Bp * 4},
        {bmeta_host, bmeta_.address(), Bp * 8 * 4},
    };
    rt_->pdma_h2d_scatter_sync(h2d);

    // 3: patch per-row ring addresses only when the active slot tuple changed, and update opt_mask / head SRFs
    if (b_last_slots_[bi] != pad_slots) {
        for (auto& sp : b_app_plans_[bi]) {
            for (uint32_t app : sp.ring_apps) {
                for (uint32_t b = 0; b < Bp; ++b) {
                    sp.main->patch_addr(1000 + int(app) * 16 + int(b), m_.kring_addr(app, pad_slots[b]));
                    sp.main->patch_addr(2000 + int(app) * 16 + int(b), m_.vring_addr(app, pad_slots[b]));
                }
            }
        }
        b_last_slots_[bi] = pad_slots;
    }
    const uint64_t opt_mask = (b_opt_mme_on_ && B < Bp) ? ((1ull << B) - 1ull) : ~uint64_t(0);
    if (b_last_opt_mask_[bi] != opt_mask) {
        for (auto& sp : b_app_plans_[bi])
            if (sp.has_opt_mme) sp.main->set_opt_mask(opt_mask);
        b_last_opt_mask_[bi] = opt_mask;
    }
    const uint32_t skip_sc = any_logits ? 0u : 1u;
    if (b_head_last_B_[bi] != B || b_head_last_skip_[bi] != skip_sc) {
        for (auto& sp : b_app_plans_[bi]) {
            if (!sp.has_head) continue;
            if (b_head_last_B_[bi] != B) sp.main->patch_srf(1, B);
            if (b_head_last_skip_[bi] != skip_sc) sp.main->patch_srf(2, skip_sc);
        }
        b_head_last_B_[bi] = B;
        b_head_last_skip_[bi] = skip_sc;
    }

    // 4: execute batched spans (with per-full-app b_kv4 commit + 24-TPC FlashDecode) in 1 hardware chain
    std::vector<DirectPlan*> chain;
    chain.reserve(96);
    if (sram_cp) chain.push_back(sram_cp);
    uint32_t kvc_used = 0;
    DFarRunState dfar_state;
    double dev_us = 0.0;
    const uint32_t n_kv4_slots = (Bp <= 4u) ? 4u : 8u;
    for (auto& sp : b_app_plans_[bi]) {
        chain.push_back(sp.main.get());
        if (sp.sync_full_app != UINT32_MAX) {
            const uint32_t app = sp.sync_full_app;
            if (sp.post) {
                DirectPlan* kp = sp.post.get();
                for (uint32_t b = 0; b < n_kv4_slots; ++b) {
                    const bool valid = b < B;
                    const uint32_t s = valid ? slots[b] : slots[0];
                    const uint64_t ka = m_.kcache_addr(app, s);
                    const uint64_t va = m_.vcache_addr(app, s);
                    const int32_t pv = valid ? int32_t(m_.slot_position(s)) : -1;
                    const uint32_t cv = valid ? m_.slot_capacity(s) : 1024u;
                    if (sp.kv_kc[b] != ka) { kp->patch_addr(10 + int(b), ka); sp.kv_kc[b] = ka; }
                    if (sp.kv_vc[b] != va) { kp->patch_addr(20 + int(b), va); sp.kv_vc[b] = va; }
                    if (sp.kv_pos[b] != pv) { kp->patch_srf(100 + int(b), uint32_t(pv)); sp.kv_pos[b] = pv; }
                    if (sp.kv_cap[b] != cv) { kp->patch_srf(200 + int(b), cv); sp.kv_cap[b] = cv; }
                }
                chain.push_back(kp);
                for (uint32_t b = 0; b < B; ++b) {
                    const uint32_t s = slots[b];
                    const uint32_t pos = m_.slot_position(s);
                    const uint32_t fr = pos >= c.window ? pos - (c.window - 1) : 0u;
                    if (fr > 0) {
                        run_dfar_tiles(app, fr, s,
                                       b_.q + uint64_t(b) * NH * D * 2,
                                       b_.pv + uint64_t(b) * NH * D * 2,
                                       b_.dmax + uint64_t(b) * NH * 4,
                                       b_.sums + uint64_t(b) * NH * 4,
                                       b_.o_acc + uint64_t(b) * NH * D * 4,
                                       chain, dfar_state, dev_us);
                    }
                }
            } else {
                for (uint32_t b = 0; b < B; ++b) {
                    const uint32_t s = slots[b];
                    const uint32_t pos = m_.slot_position(s);
                    const uint32_t cap = m_.slot_capacity(s);
                    if (kvc_used >= b_kv_commit_pool_.size()) {
                        dev_us += rt_->run_chain(chain);
                        chain.clear();
                        kvc_used = 0;
                        dfar_state.reset();
                    }
                    DirectPlan* kp = b_kv_commit_pool_[kvc_used++].get();
                    kp->patch_addr(1, m_.kcache_addr(app, s));
                    kp->patch_addr(2, m_.vcache_addr(app, s));
                    kp->patch_srf(1, b);
                    kp->patch_srf(2, pos);
                    kp->patch_srf(3, cap);
                    chain.push_back(kp);
                    const uint32_t fr = pos >= c.window ? pos - (c.window - 1) : 0u;
                    if (fr > 0) {
                        run_dfar_tiles(app, fr, s,
                                       b_.q + uint64_t(b) * NH * D * 2,
                                       b_.pv + uint64_t(b) * NH * D * 2,
                                       b_.dmax + uint64_t(b) * NH * 4,
                                       b_.sums + uint64_t(b) * NH * 4,
                                       b_.o_acc + uint64_t(b) * NH * D * 4,
                                       chain, dfar_state, dev_us, &kvc_used);
                    }
                }
            }
        }
    }
    dev_us += rt_->run_chain(chain);
    if (sram_const_on_) sram_const_valid_ = true;
    (void)dev_us;

    // 5: download compact argmax + requested logit rows in ONE SCAL PDMA scatter batch
    const uint32_t arg_bytes = 2u * Bp * 24u * 64u * 4u;
    alignas(64) uint8_t arg_buf[2 * 8 * 24 * 64 * 4];
    out_argmax.assign(B, 0);
    out_logits.clear();
    out_logits.resize(B);
    std::vector<DirectRuntime::PdmaD2H> d2h;
    d2h.reserve(1 + B);
    d2h.push_back({b_argmax_.address(), arg_buf, arg_bytes});
    for (uint32_t b = 0; b < B; ++b) {
        if (want_logits[b]) {
            out_logits[b].resize(c.vocab);
            d2h.push_back({m_.logits_.address() + uint64_t(b) * c.vocab * 4,
                           out_logits[b].data(), uint32_t(c.vocab * 4)});
        }
    }
    rt_->pdma_d2h_scatter_sync(d2h);
    const float* mv = reinterpret_cast<const float*>(arg_buf);
    const int32_t* mi = reinterpret_cast<const int32_t*>(arg_buf + uint64_t(Bp) * 24 * 64 * 4);
    for (uint32_t b = 0; b < B; ++b) {
        float best_v = -1.0e30f;
        int32_t best_idx = 0x7fffffff;
        for (uint32_t p = 0; p < 24; ++p) {
            const uint64_t off = (uint64_t(b) * 24 + p) * 64;
            const float v = mv[off];
            const int32_t idx = mi[off];
            if (v > best_v || (v == best_v && idx < best_idx)) {
                best_v = v;
                best_idx = idx;
            }
        }
        out_argmax[b] = best_idx;
    }
}

bool InkDirect::queue(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off) {
    static const bool immediate = std::getenv("G2E_INK_DIRECT_CHECK") || std::getenv("G2E_INK_DIRECT_PROF") ||
                                  std::getenv("G2E_INK_DIRECT_NOCHAIN");
    const bool was_empty = !pending();
    const bool tiled = m_.cfg_.full[m_.apps_[app].layer] && use_tiled_far(far_rows());
    if (immediate || tiled) {
        flush();
        run_app(app, side, hcur, out_buf, out_off);
        return was_empty;
    }
    static const uint32_t dspan = [] { const char* e = std::getenv("G2E_INK_DSPAN"); return e ? uint32_t(std::atoi(e)) : 8u; }();
    if (dspan <= 1) {
        queue_.push_back(&plan(app, side, hcur, out_buf, out_off));
        return was_empty;
    }
    const uint32_t sf = m_.cfg_.full[m_.apps_[app].layer] ? far_size() : 0;
    const bool tap_app = m_.aux_on_ && (app == 4 || app == 22 || app == 40);
    pending_dec_.push_back({DecodeKey{app, side, hcur, out_buf->address(), out_off, sf | (uint32_t(tap_app) << 31),
                                      uint8_t(m_.active_slot_), 0, 0},
                            out_buf});
    return was_empty;
}

void InkDirect::flush() {
    p_anchored_ = false;
    if (!anchor_ && (!pending_dec_.empty() || !queue_.empty())) { anchor(); sram_const_valid_ = false; }
    if (!pending_dec_.empty()) {
        auto decs = std::move(pending_dec_);
        pending_dec_.clear();
        static const uint32_t dspan = [] { const char* e = std::getenv("G2E_INK_DSPAN"); return e ? uint32_t(std::atoi(e)) : 8u; }();
        static const bool fuse_tg_env = [] { const char* e = std::getenv("G2E_INK_FUSE_TAIL_GRAM"); return !e || *e != '0'; }();
        static const bool dmg_f_env = [] { const char* e = std::getenv("G2E_INK_DMG_FUSE"); return !e || *e != '0'; }();
        const bool fuse_tg = fuse_tg_env && isa_.count("ink_dtail_rw_dgram_gaudi2") != 0 &&
                             isa_.count("ink_dtail_rw_dgram_msg_gaudi2") != 0;
        const auto& c = m_.cfg_;
        const uint32_t NT = c.hidden / 128;
        std::vector<DirectPlan*> chain;
        const size_t n = decs.size();
        auto can_fuse_pair = [&](uint32_t a0, uint32_t a1) {
            if (!fuse_tg || a1 != a0 + 1) return false;
            const auto& aa = m_.apps_[a0];
            return !(aa.layer + 1 == c.loop_end && aa.pass + 1 < c.passes);
        };
        std::vector<uint8_t> skip_in(n, 0);
        std::vector<int> fuse_out(n, -1);
        for (size_t k = 0; k < n; ++k) {
            if (k > 0 && can_fuse_pair(decs[k - 1].key.app, decs[k].key.app)) skip_in[k] = 1u;
            if (k + 1 < n && can_fuse_pair(decs[k].key.app, decs[k + 1].key.app)) fuse_out[k] = int(decs[k + 1].key.app);
        }
        const uint32_t max_tw = (rt_->monitor_count() - 4u - 96u) / 24u;
        const uint32_t max_mw = 48u;
        const uint32_t max_sobs = rt_->sob_count() - 1u;
        const uint32_t sf_cur = far_size();
        bool flush_ready = false;
        for (int attempt = 0; attempt < kMaxPlanAttempts; ++attempt) {
            chain.clear();
            const uint64_t gen0 = evict_gen_;
            bool restarted = false;
            size_t i = 0;
            while (i < n) {
                size_t j = i;
                if (dspan <= 1) {
                    j = i + 1;
                } else {
                    uint32_t tpc_waits = 0, mme_waits = 0, sobs = 0;
                    while (j < n && (j - i) < dspan) {
                        const bool msg = m_.apps_[decs[j].key.app].pass > 0;
                        const uint32_t base_tw = (msg ? 13u : 9u) +
                                             ((dfar_tpc_pv_on_ && (decs[j].key.sf_tap & 0xffffffu) != 0) ? 1u : 0u);
                        const uint32_t in_tw = (skip_in[j] ? 0u : 1u) + (j > i ? 1u : 0u);
                        const uint32_t need_tw = base_tw + in_tw;
                        const uint32_t need_mw = 5u + (msg ? (dmg_f_env ? 2u : 4u) : 0u);
                        const uint32_t need_sobs = (msg ? 24u : 18u);
                        if (j > i && (tpc_waits + need_tw > max_tw || mme_waits + need_mw > max_mw || sobs + need_sobs > max_sobs))
                            break;
                        tpc_waits += need_tw;
                        mme_waits += need_mw;
                        sobs += need_sobs;
                        ++j;
                    }
                }
                if (j == i + 1 && !skip_in[i] && fuse_out[i] == -1) {
                    const auto& pd = decs[i];
                    DirectPlan* pl = &plan(pd.key.app, pd.key.side, pd.key.hcur, pd.out_buf, pd.key.out_off, kDecFull, {false, false}, sf_cur);
                    if (evict_gen_ != gen0) { restarted = true; break; }
                    chain.push_back(pl);
                } else {
                    std::vector<DecodeKey> skey;
                    skey.reserve(j - i);
                    for (size_t k = i; k < j; ++k) {
                        DecodeKey dk = decs[k].key;
                        dk.flags = (skip_in[k] ? 1u : 0u) | (fuse_out[k] >= 0 ? 2u : 0u);
                        skey.push_back(dk);
                    }
                    auto it = span_plans_.find(skey);
                    if (it == span_plans_.end()) {
                        auto pl = build_evicting(sf_cur, [&] {
                            auto p = std::make_unique<DirectPlan>(*rt_);
                            int nsob = 0;
                            int prev_sob = -1;
                            for (size_t k = i; k < j; ++k) {
                                const auto& pd = decs[k];
                                const bool last = (k + 1 == j);
                                const int out_sob = last ? -1 : nsob++;
                                append_decode_app(*p, nsob, prev_sob, prev_sob >= 0 ? NT : 0, out_sob,
                                                  pd.key.app, pd.key.side, pd.key.hcur, pd.out_buf, pd.key.out_off, 0, kDecFull,
                                                  skip_in[k] != 0, fuse_out[k]);
                                prev_sob = out_sob;
                            }
                            p->prepare();
                            return p;
                        });
                        it = span_plans_.emplace(std::move(skey), std::move(pl)).first;
                        if (evict_gen_ != gen0) { restarted = true; break; }
                    }
                    chain.push_back(it->second.get());
                }
                i = j;
            }
            if (!restarted) { flush_ready = true; break; }
        }
        if (!flush_ready) throw std::runtime_error("inkling2 direct: flush plans failed to stabilize after eviction");
        const double us = rt_->run_chain(chain);
        static const bool tm = std::getenv("G2E_INK_DIRECT_TIME") != nullptr;
        if (tm) {
            auto& v = chain_prof_[uint32_t(n)];
            v.first += us;
            v.second++;
        }
    }
    if (queue_.empty()) return;
    const double us = rt_->run_chain(queue_);
    static const bool tm = std::getenv("G2E_INK_DIRECT_TIME") != nullptr;
    if (tm) {
        auto& v = chain_prof_[uint32_t(queue_.size())];
        v.first += us;
        v.second++;
    }
    queue_.clear();
}

void InkDirect::profile_stages(uint32_t app) {
    for (auto& [key, pl] : plans_) {
        if (std::get<0>(key) != app) continue;
        const auto [a, sbits, hcur, oaddr, ooff, sfk] = key;
        if (sfk >> 31) continue;  // (aux-capture variants)
        const int side = sbits & 1;
        (void)sfk;
        (void)a;
        const DeviceBuffer* ob = nullptr;  // the cached plan's output buffer (the key holds its base address)
        for (const DeviceBuffer* b : {&m_.h_[0], &m_.h_[1], &m_.blocks_, &m_.prev_[0], &m_.prev_[1]})
            if (b->address() == oaddr) ob = b;
        if (!ob) return;
        double prev = 0;
        for (int lim = 1; lim <= 40; ++lim) {
            auto pp = build(app, side, hcur, ob, ooff, lim);
            if (!pp) { std::fprintf(stderr, "[ink-direct] app %u stage %2d: MME (in the next)\n", app, lim); continue; }
            std::vector<double> ts;
            for (int r = 0; r < 25; ++r) ts.push_back(pp->run());
            std::sort(ts.begin(), ts.end());
            std::fprintf(stderr, "[ink-direct] app %u stages 1..%2d: %7.1f us (+%.1f)\n", app, lim, ts[12], ts[12] - prev);
            prev = ts[12];
        }
        return;
    }
}

double InkDirect::run_app(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off) {
    if (!anchor_) { anchor(); sram_const_valid_ = false; }
    double us = 0.0;
    const uint32_t fr = far_rows();
    if (m_.cfg_.full[m_.apps_[app].layer] && use_tiled_far(fr)) {
        DirectPlan* pre = nullptr;
        DirectPlan* post = nullptr;
        bool ready = false;
        for (int attempt = 0; attempt < kMaxPlanAttempts; ++attempt) {
            const uint64_t gen0 = evict_gen_;
            prebuild_dfar_for(fr, m_.slot_capacity(m_.active_slot_), 1u, -1, 0u);
            if (evict_gen_ != gen0) continue;
            pre = &plan(app, side, hcur, out_buf, out_off, kDecPreFar);
            if (evict_gen_ != gen0) continue;
            post = &plan(app, side, hcur, out_buf, out_off, kDecPostFar);
            if (evict_gen_ != gen0) continue;
            ready = true;
            break;
        }
        if (!ready) throw std::runtime_error("inkling2 direct: run_app tiled plans failed to stabilize after eviction");
        std::vector<DirectPlan*> chain;
        chain.push_back(pre);
        DFarRunState dfar_state;
        run_dfar_tiles(app, fr, m_.active_slot_, s_.q, s_.pv, s_.dmax, s_.sums, s_.o_acc, chain, dfar_state, us);
        chain.push_back(post);
        us += rt_->run_chain(chain);
    } else {
        auto& pl = plan(app, side, hcur, out_buf, out_off);
        us = pl.run();
    }
    static const bool prof = std::getenv("G2E_INK_DIRECT_PROF") != nullptr;
    if (prof) { auto& v = prof_[app]; v.first += us; v.second++; }
    static const int chk_at = std::getenv("G2E_INK_DIRECT_CHECK") ? std::atoi(std::getenv("G2E_INK_DIRECT_CHECK")) : -1;
    static int calls = 0;
    if (chk_at >= 0 && calls++ == chk_at) { chk_out_ = out_buf; chk_off_ = out_off; check(app, side, hcur); }
    return us;
}

namespace {
struct Rd {
    const Device& dev;
    const Stream& st;
    std::vector<uint8_t> get(uint64_t addr_base_unused, const DeviceBuffer& b, uint64_t off, uint64_t bytes) {
        (void)addr_base_unused;
        HostBuffer hb(dev, bytes + 64);
        st.download(b, off, hb, 0, bytes);
        st.synchronize();
        return std::vector<uint8_t>(hb.as<uint8_t>(), hb.as<uint8_t>() + bytes);
    }
};
std::vector<float> bf(const std::vector<uint8_t>& v) {
    std::vector<float> o(v.size() / 2);
    for (size_t i = 0; i < o.size(); ++i) { uint16_t x; std::memcpy(&x, &v[i * 2], 2); o[i] = from_bf16(x); }
    return o;
}
float h2f(uint16_t h) {
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? INFINITY : std::ldexp(float(m | 1024), int(e) - 25);
    return s ? -v : v;
}
std::vector<float> f16(const std::vector<uint8_t>& v) {
    std::vector<float> o(v.size() / 2);
    for (size_t i = 0; i < o.size(); ++i) { uint16_t x; std::memcpy(&x, &v[i * 2], 2); o[i] = h2f(x); }
    return o;
}
std::vector<float> fl(const std::vector<uint8_t>& v) {
    std::vector<float> o(v.size() / 4);
    std::memcpy(o.data(), v.data(), o.size() * 4);
    return o;
}
void report(const char* name, const std::vector<float>& got, const std::vector<float>& ref) {
    double mx = 0, num = 0, den = 0;
    size_t worst = 0;
    for (size_t i = 0; i < ref.size() && i < got.size(); ++i) {
        const double d = std::fabs(double(got[i]) - ref[i]);
        if (d > mx) { mx = d; worst = i; }
        num += d * d;
        den += double(ref[i]) * ref[i];
    }
    std::fprintf(stderr, "  [check] %-8s rel %.3e  max|d| %.4g at %zu (got %.5g ref %.5g)\n", name, std::sqrt(num / (den + 1e-30)), mx, worst,
                 worst < got.size() ? got[worst] : 0.f, worst < ref.size() ? ref[worst] : 0.f);
}
}  // namespace

void InkDirect::check(uint32_t app, int side, int hcur) {
    const auto& c = m_.cfg_;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const uint32_t H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim, W = c.window, RE = c.rel_extent;
    const uint32_t G = NH / KV, KVD = KV * D, DR = c.d_rel, QS = NH * D * 2 + NH * DR + NH, N = QS + 2 * KVD;
    Rd rd{m_.dev_, m_.stream_};
    auto sc = [&](uint64_t addr, uint64_t bytes) { return rd.get(0, scratch_, addr - scratch_.address(), bytes); };
    auto wt = [&](const char* part) { return m_.w_.read(Lw(L, part)); };
    std::fprintf(stderr, "[check] app %u layer %u side %d hcur %d pos %u\n", app, L, side, hcur, m_.pos_);
    const auto xa = bf(sc(s_.xa, H * 2));
    // projections
    const auto wx = bf(wt("wx"));
    const auto row = bf(sc(s_.row, uint64_t(N) * 2));
    {
        std::vector<float> ref(N);
        for (uint32_t n = 0; n < N; ++n) { double acc = 0; for (uint32_t k = 0; k < H; ++k) acc += double(xa[k]) * wx[uint64_t(n) * H + k]; ref[n] = from_bf16(to_bf16(float(acc))); }
        report("proj", row, ref);
    }
    // r / scale, sink
    {
        const auto rs = fl(wt("r_scale"));
        const auto r = bf(sc(s_.r, NH * 16 * 2));
        const auto sk = fl(sc(s_.sink, NH * 4));
        std::vector<float> rr(NH * 16), skr(NH);
        const float scale = 1.f / std::sqrt(float(D));
        for (uint32_t h = 0; h < NH; ++h) {
            double n2 = 0; float x[16];
            for (uint32_t d = 0; d < DR; ++d) { x[d] = row[NH * D * 2 + h * DR + d] * rs[h * DR + d]; n2 += double(x[d]) * x[d]; }
            const float f = std::min(c.rel_clip / (float(std::sqrt(n2)) + 1e-6f), 1.f);
            for (uint32_t d = 0; d < DR; ++d) rr[h * 16 + d] = from_bf16(to_bf16(from_bf16(to_bf16(x[d] * f)) / scale));
            skr[h] = row[NH * D * 2 + NH * DR + h];
        }
        report("r/scale", r, rr);
        report("sink", sk, skr);
    }
    // scores (FP16) against the ring (after the commit), bias table
    const auto q = bf(sc(s_.q, NH * D * 2));
    const uint64_t ro = m_.ring_off(m_.active_slot_);
    const auto kr = bf(rd.get(0, m_.kring_[m_.ring_index_[app]], ro, uint64_t(KV) * W * D * 2));
    const auto vr = bf(rd.get(0, m_.vring_[m_.ring_index_[app]], ro, uint64_t(KV) * W * D * 2));
    const auto s16 = f16(sc(s_.sc, uint64_t(KV) * G * W * 2));
    {
        std::vector<float> ref(uint64_t(KV) * G * W);
        for (uint32_t kv = 0; kv < KV; ++kv)
            for (uint32_t g = 0; g < G; ++g)
                for (uint32_t j = 0; j < W; ++j) {
                    double acc = 0;
                    for (uint32_t d = 0; d < D; ++d) acc += double(q[(kv * G + g) * D + d]) * kr[(uint64_t(kv) * W + j) * D + d];
                    ref[(uint64_t(kv) * G + g) * W + j] = float(acc);
                }
        report("qk", s16, ref);
    }
    {   // the new K / V rows in the ring (conv of the state and the projected row, clip / RMS per head)
        const uint64_t csin = m_.conv_off_slot(m_.active_slot_, side) + uint64_t(app) * m_.conv_app_bytes_;
        const uint64_t csout = m_.conv_off_slot(m_.active_slot_, 1 - side) + uint64_t(app) * m_.conv_app_bytes_;
        const auto st = bf(rd.get(0, m_.conv_pool_, csin + m_.conv_off_[0], uint64_t(6) * KVD * 2));
        const auto sto = bf(rd.get(0, m_.conv_pool_, csout + m_.conv_off_[0], uint64_t(6) * KVD * 2));
        const auto kt = fl(wt("k_taps")), vt = fl(wt("v_taps"));
        const uint32_t r = m_.pos_ % W;
        std::vector<float> kref(KVD), vref(KVD), kgot(KVD), vgot(KVD), stref(6 * KVD);
        for (int kv = 0; kv < 2; ++kv) {
            const auto& tp = kv == 0 ? kt : vt;
            for (uint32_t h = 0; h < KV; ++h) {
                float y[128];
                double ss = 0;
                for (uint32_t d = 0; d < D; ++d) {
                    const uint64_t ch = h * D + d;
                    const float x = row[QS + kv * KVD + ch];
                    const float* sst = &st[uint64_t(kv) * 3 * KVD];
                    auto w = [&](int i) { return from_bf16(to_bf16(tp[uint64_t(i) * KVD + ch])); };
                    const float v = ((w(0) * sst[ch] + w(1) * sst[KVD + ch]) + w(2) * sst[2 * KVD + ch]) + w(3) * x + x;
                    y[d] = from_bf16(to_bf16(v * (1.f / std::sqrt(5.f))));
                    ss += double(y[d]) * y[d];
                    stref[uint64_t(kv) * 3 * KVD + ch] = sst[KVD + ch];
                    stref[uint64_t(kv) * 3 * KVD + KVD + ch] = sst[2 * KVD + ch];
                    stref[uint64_t(kv) * 3 * KVD + 2 * KVD + ch] = x;
                }
                const float f = kv == 0 ? std::min(c.qk_clip / (float(std::sqrt(ss)) + 1e-6f), 1.f) : 1.f / std::sqrt(float(ss / 128.0) + c.eps);
                for (uint32_t d = 0; d < D; ++d) (kv == 0 ? kref : vref)[h * D + d] = from_bf16(to_bf16(y[d] * f));
                for (uint32_t d = 0; d < D; ++d) {
                    kgot[h * D + d] = kr[(uint64_t(h) * W + r) * D + d];
                    vgot[h * D + d] = vr[(uint64_t(h) * W + r) * D + d];
                }
            }
        }
        report("k ring", kgot, kref);
        report("v ring", vgot, vref);
        report("kvstate", sto, stref);
    }
    const auto r16 = bf(sc(s_.r, NH * 16 * 2));
    {
        auto raw = m_.w_.read(Lw(L, "rel_proj"));
        std::vector<uint16_t> pj(raw.size() / 2);
        std::memcpy(pj.data(), raw.data(), pj.size() * 2);
        const auto bt = f16(sc(s_.bt, uint64_t(NH) * RE * 2));
        std::vector<float> ref(uint64_t(NH) * RE);
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t k = 0; k < RE; ++k) {
                double acc = 0;
                for (uint32_t d = 0; d < DR; ++d) acc += double(r16[h * 16 + d]) * from_bf16(pj[uint64_t(d) * RE + (RE - 1 - k)]);
                ref[uint64_t(h) * RE + k] = float(acc);
            }
        report("bias", bt, ref);
        // probabilities: ages from the position (after the commit)
        const auto sk = fl(sc(s_.sink, NH * 4));
        const auto pr = bf(sc(s_.p, uint64_t(KV) * G * W * 2));
        const auto sums = fl(sc(s_.sums, NH * 4));
        std::vector<float> pref(uint64_t(KV) * G * W), sref(NH);
        const float scale = 1.f / std::sqrt(float(D));
        const uint32_t pos = m_.pos_, jw = pos % W, amax = std::min(pos, W - 1);
        for (uint32_t h = 0; h < NH; ++h) {
            const uint32_t kv = h / G, g = h % G;
            std::vector<double> y(W);
            double mx = sk[h] / scale;
            for (uint32_t j = 0; j < W; ++j) {
                const uint32_t age = (jw + W - j) % W;
                y[j] = age > amax ? -1e30 : s16[(uint64_t(kv) * G + g) * W + j] + bt[uint64_t(h) * RE + (RE - 1 - age)];
                mx = std::max(mx, y[j]);
            }
            double sum = std::exp((sk[h] / scale - mx) * scale);
            for (uint32_t j = 0; j < W; ++j) {
                const double e = y[j] < -1e29 ? 0 : std::exp((y[j] - mx) * scale);
                pref[(uint64_t(kv) * G + g) * W + j] = float(e);
                sum += e;
            }
            sref[h] = float(sum);
        }
        report("p", pr, pref);
        report("sums", sums, sref);
        // P . V, gate
        const auto pv = bf(sc(s_.pv, uint64_t(NH) * D * 2));
        std::vector<float> pvr(uint64_t(NH) * D);
        for (uint32_t kv = 0; kv < KV; ++kv)
            for (uint32_t g = 0; g < G; ++g)
                for (uint32_t d = 0; d < D; ++d) {
                    double acc = 0;
                    for (uint32_t j = 0; j < W; ++j) acc += double(pr[(uint64_t(kv) * G + g) * W + j]) * vr[(uint64_t(kv) * W + j) * D + d];
                    pvr[(uint64_t(kv) * G + g) * D + d] = float(acc);
                }
        report("pv", pv, pvr);
        const auto gated = bf(sc(s_.gated, uint64_t(NH) * D * 2));
        std::vector<float> gr(uint64_t(NH) * D);
        for (uint32_t h = 0; h < NH; ++h)
            for (uint32_t d = 0; d < D; ++d) {
                const float gt = row[NH * D + h * D + d];
                gr[h * D + d] = from_bf16(to_bf16(pv[h * D + d] / sums[h] / (1.f + std::exp(-gt))));
            }
        report("gated", gated, gr);
    }
    // ---- the non-attention stages
    const uint32_t nbb = L / c.attn_res_block + 1;
    const auto blocks = bf(rd.get(0, m_.blocks_, 0, uint64_t(nbb) * H * 2));
    const auto hin = bf(rd.get(0, m_.h_[hcur], 0, uint64_t(H) * 2));
    auto mixref = [&](const std::vector<float>& part, const char* qname, const char* wname) {
        const auto qv = fl(wt(qname));
        const auto wv = bf(wt(wname));
        const uint32_t S = nbb + 1;
        std::vector<double> rms(S), lg(S);
        auto xs = [&](uint32_t st, uint32_t k) { return st < nbb ? double(blocks[uint64_t(st) * H + k]) : double(part[k]); };
        for (uint32_t st = 0; st < S; ++st) {
            double sq = 0, d = 0;
            for (uint32_t k = 0; k < H; ++k) { sq += xs(st, k) * xs(st, k); d += xs(st, k) * qv[k]; }
            rms[st] = 1.0 / std::sqrt(sq / H + 1e-6);
            lg[st] = d * rms[st] / std::sqrt(double(H));
        }
        double mx = -1e300; for (auto v : lg) mx = std::max(mx, v);
        double se = 0; std::vector<double> pr(S); for (uint32_t st = 0; st < S; ++st) { pr[st] = std::exp(lg[st] - mx); se += pr[st]; }
        std::vector<double> mix(H, 0.0);
        for (uint32_t st = 0; st < S; ++st) for (uint32_t k = 0; k < H; ++k) mix[k] += pr[st] / se * rms[st] * xs(st, k);
        double ms = 0; for (auto v : mix) ms += v * v;
        const double rm = 1.0 / std::sqrt(ms / H + c.eps);
        std::vector<float> o(H);
        for (uint32_t k = 0; k < H; ++k) o[k] = from_bf16(to_bf16(float(mix[k] * rm * wv[k])));
        return o;
    };
    report("mix", xa, mixref(hin, "attn_q", "attn_norm"));
    auto convref = [&](const std::vector<float>& x, int which, const char* taps, float scale) {
        const auto st = bf(rd.get(0, m_.conv_pool_, m_.conv_off_slot(m_.active_slot_, side) + uint64_t(app) * m_.conv_app_bytes_ + m_.conv_off_[which], uint64_t(3) * x.size() * 2));
        const auto tp = fl(wt(taps));
        const uint64_t C = x.size();
        std::vector<float> o(C);
        for (uint64_t k = 0; k < C; ++k) {
            const float w0 = from_bf16(to_bf16(tp[k])), w1 = from_bf16(to_bf16(tp[C + k])), w2 = from_bf16(to_bf16(tp[2 * C + k])),
                        w3 = from_bf16(to_bf16(tp[3 * C + k]));
            const float v = ((w0 * st[k] + w1 * st[C + k]) + w2 * st[2 * C + k]) + w3 * x[k] + x[k];
            o[k] = from_bf16(to_bf16(v * scale));
        }
        return o;
    };
    const auto gated2 = bf(sc(s_.gated, uint64_t(NH) * D * 2));
    const auto a1 = bf(sc(s_.a1, H * 2));
    {
        const auto wo = bf(wt("wo"));
        std::vector<float> ref(H);
        for (uint32_t n = 0; n < H; ++n) { double acc = 0; for (uint32_t k = 0; k < NH * D; ++k) acc += double(gated2[k]) * wo[uint64_t(n) * NH * D + k]; ref[n] = from_bf16(to_bf16(float(acc))); }
        report("wo", a1, ref);
    }
    const auto a2 = convref(a1, 2, "attn_conv", 1.f / std::sqrt(5.f));  // (not stored by the plan)
    const auto resil = fl(sc(s_.res, H * 4));
    std::vector<float> res(H);
    for (uint32_t cc = 0; cc < H / 128; ++cc)
        for (uint32_t j = 0; j < 64; ++j) { res[cc * 128 + 2 * j] = resil[cc * 128 + j]; res[cc * 128 + 2 * j + 1] = resil[cc * 128 + 64 + j]; }
    {
        std::vector<float> ref(H);
        for (uint32_t k = 0; k < H; ++k) ref[k] = hin[k] + a2[k];
        report("res", res, ref);
    }
    const auto xn = bf(sc(s_.xn, H * 2));
    report("mlpmix", xn, mixref(res, "mlp_q", "mlp_norm"));
    const auto lgt = fl(sc(s_.logits, c.experts * 4));
    {
        const auto rw = bf(wt("router"));
        std::vector<float> ref(c.experts);
        for (uint32_t e = 0; e < c.experts; ++e) { double acc = 0; for (uint32_t k = 0; k < H; ++k) acc += double(xn[k]) * rw[uint64_t(e) * H + k]; ref[e] = float(acc); }
        report("router", lgt, ref);
    }
    {
        const auto idx = rd.get(0, scratch_, s_.idx - scratch_.address(), c.topk * 4);
        const auto rwv = fl(sc(s_.rw, c.topk * 4));
        const auto bias = fl(wt("router_bias"));
        std::vector<std::pair<double, int>> v;
        for (uint32_t e = 0; e < c.experts; ++e) v.push_back({1.0 / (1.0 + std::exp(-lgt[e])) + bias[e], int(e)});
        std::sort(v.begin(), v.end(), [](auto& x, auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
        std::fprintf(stderr, "  [check] route    got");
        for (uint32_t j = 0; j < c.topk; ++j) { int32_t e; std::memcpy(&e, &idx[j * 4], 4); std::fprintf(stderr, " %d(%.3f)", e, rwv[j]); }
        std::fprintf(stderr, "\n  [check] route    ref");
        double ws = 0; for (uint32_t j = 0; j < c.topk; ++j) ws += 1.0 / (1.0 + std::exp(-lgt[v[j].second]));
        for (uint32_t j = 0; j < c.topk; ++j) std::fprintf(stderr, " %d(%.3f)", v[j].second, 1.0 / (1.0 + std::exp(-lgt[v[j].second])) / ws * c.route_scale);
        std::fprintf(stderr, "\n");
    }
    const auto ys = bf(sc(s_.ys, uint64_t(c.topk) * H * 2));
    const auto rw_chk = fl(sc(s_.rw, c.topk * 4));
    {   // tail: h_out = bf16(res + conv(bf16(sum_j ys[j] * rw[j])))
        std::vector<float> sv(H);
        for (uint32_t k = 0; k < H; ++k) {
            float acc = 0;
            for (uint32_t j = 0; j < c.topk; ++j) acc += ys[uint64_t(j) * H + k] * rw_chk[j];
            sv[k] = from_bf16(to_bf16(acc));
        }
        const auto m2 = convref(sv, 3, "mlp_conv", 1.f / std::sqrt(5.f));
        std::vector<float> ref(H);
        for (uint32_t k = 0; k < H; ++k) ref[k] = from_bf16(to_bf16(res[k] + m2[k]));
        const auto got = bf(rd.get(0, *chk_out_, chk_off_, uint64_t(H) * 2));
        report("h_out", got, ref);
        const uint64_t csin = m_.conv_off_slot(m_.active_slot_, side) + uint64_t(app) * m_.conv_app_bytes_;
        const uint64_t csout = m_.conv_off_slot(m_.active_slot_, 1 - side) + uint64_t(app) * m_.conv_app_bytes_;
        const auto sto = bf(rd.get(0, m_.conv_pool_, csout + m_.conv_off_[3], uint64_t(3) * H * 2));
        const auto sti = bf(rd.get(0, m_.conv_pool_, csin + m_.conv_off_[3], uint64_t(3) * H * 2));
        std::vector<float> stref(3 * H);
        for (uint32_t k = 0; k < H; ++k) { stref[k] = sti[H + k]; stref[H + k] = sti[2 * H + k]; stref[2 * H + k] = sv[k]; }
        report("mstate", sto, stref);
        const auto ato = bf(rd.get(0, m_.conv_pool_, csout + m_.conv_off_[2], uint64_t(3) * H * 2));
        const auto ati = bf(rd.get(0, m_.conv_pool_, csin + m_.conv_off_[2], uint64_t(3) * H * 2));
        for (uint32_t k = 0; k < H; ++k) { stref[k] = ati[H + k]; stref[H + k] = ati[2 * H + k]; stref[2 * H + k] = a1[k]; }
        report("astate", ato, stref);
    }
    (void)resil;
}

}  // namespace g2e

namespace g2e {

// ------------------------------------------------------------------------------------------------ verification
// T = kVerifyT rows at positions pos .. pos + T - 1 (speculative verification): the same applications as build(),
// with nothing committed (K/V rows to the scratch, conv input histories to Inkling2::hist_; accept() commits).
bool InkDirect::handles_v(uint32_t app, uint32_t T) const {
    if (!mask_ || T < 2 || T > kVerifyMaxT || app >= m_.apps_.size() || !handles(app)) return false;
    const std::string pre = "v" + std::to_string(T) + "_";
    if (!mme_.count(pre + "proj")) return false;
    const auto& a = m_.apps_[app];
    if (!m_.cfg_.full[a.layer]) return true;
    const uint32_t W = m_.cfg_.window, pos = m_.pos_;
    const uint32_t sf = m_.far_bucket(pos > W ? pos - W : 0);
    return sf == 0 || (sf <= vfar_cap_ && mme_.count(pre + "fqk_" + std::to_string(sf)));
}

bool InkDirect::queue_v(uint32_t app, uint32_t T, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off) {
    const uint32_t W = m_.cfg_.window, pos = m_.pos_;
    const uint32_t sf_cur = m_.far_bucket(pos > W ? pos - W : 0);
    const uint32_t sf = m_.cfg_.full[m_.apps_[app].layer] ? sf_cur : 0;
    const bool tap_app = m_.aux_on_ && (app == 4 || app == 22 || app == 40);
    const int sbits = side | (int(m_.active_slot_) << 1);
    const auto key = std::make_tuple(app, sbits, hcur, out_buf->address(), out_off, sf | (T << 24) | (uint32_t(tap_app) << 31));
    auto it = plans_v_.find(key);
    if (it == plans_v_.end()) {
        auto p = build_evicting(sf_cur, [&] { return build_v(app, T, side, hcur, out_buf, out_off, 0); }, -1, true);
        it = plans_v_.emplace(key, std::move(p)).first;
    }
    const bool was_empty = queue_.empty();
    static const bool immediate = std::getenv("G2E_INK_DIRECT_NOCHAIN") != nullptr;
    if (immediate) {
        it->second->run();
        return was_empty;
    }
    queue_.push_back(it->second.get());
    return was_empty;
}

std::unique_ptr<DirectPlan> InkDirect::build_v(uint32_t app, uint32_t T, int side, int hcur, const DeviceBuffer* out_buf,
                                               uint64_t out_off, int lim) {
    const auto& c = m_.cfg_;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const bool nvfp4 = c.is_nvfp4(L);
    const uint32_t H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim, W = c.window, RE = c.rel_extent;
    const uint32_t G = NH / KV, KVD = KV * D, DR = c.d_rel, E = c.experts, F = c.moe_inter, J = c.topk;
    const uint32_t QS = NH * D * 2 + NH * DR + NH, nbb = L / c.attn_res_block + 1;
    const bool boundary = L % c.attn_res_block == 0, msg = a.pass > 0, full = c.full[L];
    const auto& l = lc_.at(L);
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    auto wa = [&](const char* part) { return arena + w.at(Lw(L, part)).offset; };
    const uint64_t slab = uint64_t(T) * H * 2;
    const uint64_t blocks = m_.blocks_.address();
    const uint64_t h_in = m_.h_[hcur].address();
    const uint64_t kvs = m_.kvscratch_.address() + uint64_t(app) * 2 * KVD * Inkling2::kMaxT * 2;
    const uint64_t cin = m_.conv_addr(app, m_.active_slot_, side);
    const uint64_t hist = m_.hist_.address() + uint64_t(app) * m_.hist_app_;
    const uint64_t kring = m_.kring_addr(app, m_.active_slot_);
    const uint64_t vring = m_.vring_addr(app, m_.active_slot_);
    const uint64_t tab = m_.tables_.address();
    const uint64_t h_out = out_buf->address() + out_off;
    const bool src_copy = L + 1 == c.loop_end - c.attn_res_block && a.pass + 1 < c.passes;
    // DSpark taps: the partial block entering visits 5, 23, 41 = the outputs of applications 4, 22, 40
    const int tap = !m_.aux_on_ ? -1 : app == 4 ? 0 : app == 22 ? 1 : app == 40 ? 2 : -1;
    const uint64_t h_out2 = src_copy ? m_.prev_[a.pass == 0 ? 1 : 0].address()
                            : tap >= 0 ? m_.aux_.address() + uint64_t(tap) * Inkling2::kMaxT * H * 2 : h_out;
    const float scale = 1.f / std::sqrt(float(D));
    const float conv_s = 1.f / std::sqrt(5.f);
    const uint32_t NT = H / 128;
    const float mscale = 1.f / std::sqrt(float(H));
    const uint32_t s0 = boundary ? std::max(nbb - 1, 1u) : nbb;
    const uint64_t apart = boundary ? blocks + uint64_t(nbb - 1) * slab : h_in;
    const uint64_t prev = msg ? m_.prev_[a.pass == 1 ? 1 : 0].address() : 0;
    const uint64_t RH = uint64_t(c.gate_rank) * 2 * H * 2;
    const uint32_t Qa = full ? 3 : 1;
    const uint32_t KR = full ? KVD + KV * 4 : KVD;
    const uint32_t NROW = QS + 2 * KR;
    const bool three = msg || full;
    const uint32_t pos = m_.pos_;
    const uint32_t SF = full ? m_.far_bucket(pos > W ? pos - W : 0) : 0;
    const uint32_t GT = G * T;
    const auto& v = vs_;
    const std::string pre = "v" + std::to_string(T) + "_";
    auto TM = [&](const char* k) -> const MmeTemplate& { return mme_.at(pre + k); };

    auto p = std::make_unique<DirectPlan>(*rt_);
    auto& P = *p;
    int nst = 0;
    bool done = false, mme_cut = false;
    auto TG = [&](uint64_t isa, std::span<const DirectTpcTensor> t, std::span<const uint32_t> sr, std::array<uint32_t, 5> grid,
                  int wsob, uint32_t wtgt, int sig) {
        if (done) return;
        if (lim > 0 && ++nst == lim) { sig = -1; done = true; }
        P.add_tpc_grid(isa, t, sr, grid, 0, 24, wsob, wtgt, sig);
    };
    auto sig_of = [](uint32_t g0) { return std::min<uint32_t>(24, g0); };
    // the normalize phases recompute the mix coefficients per row: split them over the rows (T TPCs, coefficients once)
    auto TR = [&](uint64_t isa, std::span<const DirectTpcTensor> t, std::span<const uint32_t> sr, int wsob, uint32_t wtgt, int sig) {
        if (done) return;
        if (lim > 0 && ++nst == lim) { sig = -1; done = true; }
        P.add_tpc_grid(isa, t, sr, {NT, T, 1, 1, 1}, 1, 24, wsob, wtgt, sig);
    };
    auto MM = [&](const MmeTemplate& tm, uint64_t a0, uint64_t b0, uint64_t c0, int wsob, uint32_t wtgt, int sig) {
        if (done) return;
        if (lim > 0 && ++nst == lim) { mme_cut = true; done = true; return; }
        P.add_mme(tm, a0, b0, c0, 0, wsob, wtgt, sig);
    };
    const auto bf = DType::BF16, f32 = DType::F32, f16 = DType::F16;
    static const bool dmg_fuse_v = [] { const char* e = std::getenv("G2E_INK_DMG_FUSE"); return !e || *e != '0'; }();

    // 0: attention input (depth attention + norms; message gates)
    uint64_t in3;
    if (!msg) {
        const uint64_t q = m_.qil_.address() + m_.qil_off_.at(Lw(L, L == 0 ? "mlp_q" : "attn_q"));
        const DirectTpcTensor t[] = {tt(blocks, bf, {H, T, s0}), tt(apart, bf, {H, T}), tt(q, f32, {H, Qa}), tt(v.p1, f32, {64, NT, T})};
        TG(isa_.at("ink_dgram_gaudi2"), t, {}, {NT, T, 1, 1, 1}, -1, 0, 0);
        struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, L == 0 ? 1 : 0};
        const DirectTpcTensor t2[] = {tt(v.p1, f32, {64, NT, T}), tt(blocks, bf, {H, T, s0}), tt(apart, bf, {H, T}),
                                      tt(full ? l.wn3 : wa("attn_norm"), bf, {H, Qa}), tt(v.xa, bf, {H, T, Qa})};
        const auto s = srf(prm);
        TR(isa_.at("ink_dnorm_gaudi2"), t2, s, 0, sig_of(NT), 10);
        in3 = v.xa;
    } else {
        const uint64_t q = m_.qil_.address() + m_.qil_off_.at(Lw(L, "attn_q"));
        const uint64_t mq = m_.qil_.address() + m_.qil_off_.at("msg_q") + uint64_t(L - c.loop_start) * 4 * H * 4;
        const DirectTpcTensor t[] = {tt(blocks, bf, {H, T, s0}), tt(apart, bf, {H, T}), tt(q, f32, {H, Qa}), tt(v.p1, f32, {64, NT, T}),
                                     tt(prev, bf, {H, T, 1}), tt(prev + slab, bf, {H, T}), tt(mq, f32, {H, 4}), tt(v.p3, f32, {64, NT, T})};
        TG(isa_.at("ink_dgram_msg_gaudi2"), t, {}, {NT, T, 1, 1, 1}, -1, 0, 0);
        struct { float scale, eps; int32_t m0, write_msg; } prm{mscale, c.eps, 0, 1};
        const DirectTpcTensor t2[] = {tt(v.p1, f32, {64, NT, T}), tt(blocks, bf, {H, T, s0}), tt(apart, bf, {H, T}), tt(v.p3, f32, {64, NT, T}),
                                      tt(prev, bf, {H, T, 1}), tt(prev + slab, bf, {H, T}), tt(v.cat, bf, {2 * H, T, 3}),
                                      tt(v.xil, f32, {H, T, Qa}), tt(v.msg, f32, {H, T, 4})};
        const auto s = srf(prm);
        TR(isa_.at("ink_dnorm_msg_gaudi2"), t2, s, 0, sig_of(NT), 20);
        const auto& td = TM("mgd3");
        const uint64_t md = arena + w.at(Lw(L, "mg_down")).offset;
        MM(td, v.cat, md, v.gr, 20, T, 21);
        const int32_t m0 = 0;
        const auto so = srf(m0);
        if (dmg_fuse_v) {
            const DirectTpcTensor t3[] = {tt(v.xil, f32, {H, T, Qa}), tt(v.msg, f32, {H, T, 4}),
                                          tt(v.gr, bf, {c.gate_rank, T, 3}), tt(l.mg_fg_t, bf, {2 * H, c.gate_rank, 3}),
                                          tt(v.ymg, f32, {H, T, 3}), tt(v.p4, f32, {64, NT, T})};
            TG(isa_.at("ink_dmg_fgy_gaudi2"), t3, so, {NT, T, 1, 1, 1}, 21, td.signal_count, 27);
        } else {
            const auto& tf = TM("mgf3x");
            const uint64_t mf = arena + w.at(Lw(L, "mg_fg")).offset;
            MM(tf, v.gr, mf, v.fgv, 21, td.signal_count, 22);
            DirectTpcTensor fg = tt(v.fgv, f32, {2 * H, T, 3});  // row g T + t of [3T, 3 * 2H], gate g's block at column g 2H
            fg.strides[0] = 1;
            fg.strides[1] = 3 * 2 * H;
            fg.strides[2] = T * 3 * 2 * H + 2 * H;
            const DirectTpcTensor t3[] = {tt(v.xil, f32, {H, T, Qa}), tt(v.msg, f32, {H, T, 4}), fg, tt(v.ymg, f32, {H, T, 3}),
                                          tt(v.p4, f32, {64, NT, T})};
            TG(isa_.at("ink_dmgy_gaudi2"), t3, so, {NT, T, 1, 1, 1}, 22, tf.signal_count, 27);
        }
        const DirectTpcTensor t4[] = {tt(v.p4, f32, {64, NT, T}), tt(v.ymg, f32, {H, T, 3}), tt(l.wn3, bf, {H, 3}), tt(v.fqkv, bf, {H, T, 3})};
        const auto se = srf(c.eps);
        TG(isa_.at("ink_dmgn_gaudi2"), t4, se, {NT, T, 1, 1, 1}, 27, sig_of(NT), 10);
        in3 = v.fqkv;
    }
    // 1: projections: [T, H] . wx^T, or the stacked [3T, H] (rows q-input, k-input, v-input)
    uint32_t proj_signals;
    {
        const auto& tp = TM(!three ? "proj" : full ? "proj3f" : "proj3s");
        MM(tp, in3, wa("wx"), v.row, 10, msg ? sig_of(NT) : T, 1);
        proj_signals = tp.signal_count;
    }
    // 2: q / r / sink per row, k / v convs over the rows (scratch + history; nothing committed)
    {
        struct { float thr_q, thr_r, isc, cscale, eps; int32_t off_rel, off_sink, off_k, off_v, nrow, three; } prm{
            c.qk_clip, c.rel_clip, 1.f / scale, full ? 1.f / 3.f : conv_s, c.eps, int32_t(NH * D * 2), int32_t(NH * D * 2 + NH * DR),
            int32_t(QS), int32_t(QS + KR), int32_t(NROW), three ? 1 : 0};
        std::vector<DirectTpcTensor> t = {tt(v.row, bf, {(three ? 3 : 1) * T * NROW}), tt(l.qsb, bf, {D, NH}), tt(v.q, bf, {D, T, G, KV}),
                                          tt(l.rs, f32, {16, NH}), tt(v.r, bf, {16, NH, T}), tt(v.sink, f32, {NH, T}),
                                          tt(cin + m_.conv_off_[0], bf, {KVD, 3, 2}), tt(l.kvtaps, bf, {KVD, 4, 2}),
                                          tt(hist + m_.hist_off(0, T), bf, {KVD, 3 + T, 2}), tt(kvs, bf, {D, T, KV, 2})};
        if (full) t.push_back(tt(tab + m_.tab_.tau, f32, {1}));
        const auto sr = srf(prm);
        const uint32_t pts = 2 * NH * T + 2 * KV;
        TG(isa_.at(full ? "ink_dpost_vf_gaudi2" : "ink_dpost_v_gaudi2"), t, sr, {pts, 1, 1, 1, 1}, 1, proj_signals, 2);
    }
    // 3: bias table rows (t, h), scores against the ring and the block's rows; full layers: far keys per KV head
    const auto& tb = TM("bias");
    const auto& tq = TM("qk");
    const auto& tqn = TM("qkn");
    MM(tb, v.r, l.projr, v.bt, 2, 24, 37);
    MM(tq, v.q, kring, v.sc, -1, 0, 38);
    MM(tqn, v.q, kvs, v.scn, -1, 0, SF ? 39 : 5);
    uint32_t att_signals = tqn.signal_count;
    const MmeTemplate* tfq = SF ? &mme_.at(pre + "fqk_" + std::to_string(SF)) : nullptr;
    const MmeTemplate* tfv = SF ? &mme_.at(pre + "fpv_" + std::to_string(SF)) : nullptr;
    if (SF) {
        const int s = m_.kv_index_[app];
        const uint64_t kl = m_.kcache_[s].address();
        for (uint32_t h = 0; h < KV; ++h)
            MM(*tfq, v.q + uint64_t(h) * GT * D * 2, kl + uint64_t(h) * m_.cap_ * D * 2, v.sf + uint64_t(h) * GT * SF * 2, -1, 0,
               h + 1 == KV ? 5 : 34);
        att_signals = tfq->signal_count;
    }
    // 4: probabilities and row sums
    {
        struct { float scale; int32_t G, full; } prm{scale, int32_t(G), full ? 1 : 0};
        std::vector<DirectTpcTensor> t = {tt(v.sc, f16, {W, T, G, KV}), tt(v.scn, f16, {T, T, G, KV}), tt(v.bt, f16, {RE, NH, T}),
                                          tt(v.sink, f32, {NH, T}), tt(tab + m_.tab_.pos32, DType::I32, {1}), tt(v.p, bf, {W, T, G, KV}),
                                          tt(v.pn, bf, {T, T, G, KV}), tt(v.sums, f32, {NH, T})};
        if (SF) {
            t.push_back(tt(v.sf, f16, {SF, T, G, KV}));
            t.push_back(tt(v.pf, bf, {SF, T, G, KV}));
        }
        const auto sr = srf(prm);
        TG(isa_.at(SF ? "ink_dattn_vf_gaudi2" : "ink_dattn_v_gaudi2"), t, sr, {NH * T, 1, 1, 1, 1}, 5, att_signals, 6);
    }
    // 5: P . V (ring, block rows, far keys)
    const auto& tv = TM("pv");
    const auto& tvn = TM("pvn");
    MM(tv, v.p, vring, v.pv, 6, 24, 35);
    MM(tvn, v.pn, kvs + uint64_t(KV) * T * D * 2, v.pvn, -1, 0, SF ? 36 : 7);
    uint32_t pv_signals = tvn.signal_count;
    if (SF) {
        const int s = m_.kv_index_[app];
        const uint64_t vl = m_.vcache_[s].address();
        for (uint32_t h = 0; h < KV; ++h)
            MM(*tfv, v.pf + uint64_t(h) * GT * SF * 2, vl + uint64_t(h) * m_.cap_ * D * 2, v.pvf + uint64_t(h) * GT * D * 2, -1, 0,
               h + 1 == KV ? 7 : 33);
        pv_signals = tfv->signal_count;
    }
    // 6: / sum, * sigmoid(gate)
    {
        struct { int32_t off_gate, G, nrow; } prm{int32_t(NH * D), int32_t(G), int32_t(NROW)};
        std::vector<DirectTpcTensor> t = {tt(v.pv, bf, {D, T, G, KV}), tt(v.pvn, bf, {D, T, G, KV})};
        if (SF) t.push_back(tt(v.pvf, bf, {D, T, G, KV}));
        t.push_back(tt(v.sums, f32, {NH, T}));
        t.push_back(tt(v.row, bf, {(three ? 3 : 1) * T * NROW}));
        t.push_back(tt(v.gated, bf, {NH * D, T}));
        const auto sr = srf(prm);
        TG(isa_.at(SF ? "ink_dgate_vf_gaudi2" : "ink_dgate_v_gaudi2"), t, sr, {NH * T, 1, 1, 1, 1}, 7, pv_signals, 8);
    }
    // 7: output projection
    const auto& to = TM("wo");
    MM(to, v.gated, wa("wo"), v.a1, 8, 24, 9);
    // 8: post-attention conv + residual + MLP partials, then the MLP input
    {
        const DirectTpcTensor t[] = {tt(v.a1, bf, {H, T}), tt(cin + m_.conv_off_[2], bf, {H, 3}), tt(l.ataps, bf, {H, 4}), tt(h_in, bf, {H, T}),
                                     tt(blocks, bf, {H, T, nbb}), tt(m_.qil_.address() + m_.qil_off_.at(Lw(L, "mlp_q")), f32, {H, 1}),
                                     tt(hist + m_.hist_off(2, T), bf, {H, 3 + T}), tt(v.res, f32, {H, T}), tt(v.p2, f32, {64, NT, T})};
        struct { float cscale; int32_t has_h; } prm{conv_s, boundary ? 0 : 1};
        const auto s = srf(prm);
        TG(isa_.at("ink_dmid_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 9, to.signal_count, 11);
    }
    if (!msg) {
        const DirectTpcTensor t[] = {tt(v.p2, f32, {64, NT, T}), tt(blocks, bf, {H, T, nbb}), tt(v.res, f32, {H, T}),
                                     tt(wa("mlp_norm"), bf, {H, 1}), tt(v.xn, bf, {H, T, 1})};
        struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, 0};
        const auto s = srf(prm);
        TR(isa_.at("ink_dnorm_f32p_gaudi2"), t, s, 11, sig_of(NT), 12);
    } else {
        struct { float scale, eps; int32_t m0, write_msg; } prm{mscale, c.eps, 3, 0};
        const DirectTpcTensor t[] = {tt(v.p2, f32, {64, NT, T}), tt(blocks, bf, {H, T, nbb}), tt(v.res, f32, {H, T}), tt(v.p3, f32, {64, NT, T}),
                                     tt(prev, bf, {H, T, 1}), tt(prev + slab, bf, {H, T}), tt(v.cat, bf, {2 * H, T, 1}),
                                     tt(v.mrmix, f32, {H, T, 1}), tt(v.msg, f32, {H, T, 4})};
        const auto s = srf(prm);
        TR(isa_.at("ink_dnorm_msg_f32p_gaudi2"), t, s, 11, sig_of(NT), 24);
        const auto& td = TM("mgd1");
        const uint64_t md = arena + w.at(Lw(L, "mg_down")).offset;
        MM(td, v.cat, md + 3 * RH, v.gr, 24, T, 25);
        const int32_t m0 = 3;
        const auto so = srf(m0);
        if (dmg_fuse_v) {
            const DirectTpcTensor t3[] = {tt(v.mrmix, f32, {H, T, 1}), tt(v.msg, f32, {H, T, 4}),
                                          tt(v.gr, bf, {c.gate_rank, T, 1}), tt(l.mg_fg_t + 3 * RH, bf, {2 * H, c.gate_rank, 1}),
                                          tt(v.ymg, f32, {H, T, 1}), tt(v.p4, f32, {64, NT, T})};
            TG(isa_.at("ink_dmg_fgy_gaudi2"), t3, so, {NT, T, 1, 1, 1}, 25, td.signal_count, 28);
        } else {
            const auto& tf = TM("mgf1");
            const uint64_t mf = arena + w.at(Lw(L, "mg_fg")).offset;
            MM(tf, v.gr, mf + 3 * RH, v.fgv, 25, td.signal_count, 26);
            const DirectTpcTensor t3[] = {tt(v.mrmix, f32, {H, T, 1}), tt(v.msg, f32, {H, T, 4}), tt(v.fgv, f32, {2 * H, T, 1}),
                                          tt(v.ymg, f32, {H, T, 1}), tt(v.p4, f32, {64, NT, T})};
            TG(isa_.at("ink_dmgy_gaudi2"), t3, so, {NT, T, 1, 1, 1}, 26, tf.signal_count, 28);
        }
        const DirectTpcTensor t4[] = {tt(v.p4, f32, {64, NT, T}), tt(v.ymg, f32, {H, T, 1}), tt(wa("mlp_norm"), bf, {H, 1}),
                                      tt(v.xn, bf, {H, T, 1})};
        const auto se = srf(c.eps);
        TG(isa_.at("ink_dmgn_gaudi2"), t4, se, {NT, T, 1, 1, 1}, 28, sig_of(NT), 12);
    }
    // 9: router, top-8 per row
    const auto& tr = TM("router");
    MM(tr, v.xn, wa("router"), v.logits, 12, msg ? sig_of(NT) : T, 13);
    static const bool droute1_v = [] { const char* e = std::getenv("G2E_INK_DROUTE1"); return !e || *e != '0'; }();
    {
        const auto s = srf(c.route_scale);
        if (droute1_v) {
            const DirectTpcTensor t[] = {tt(v.logits, f32, {E, T}), tt(wa("router_bias"), f32, {E}),
                                         tt(v.idx, DType::I32, {J, T}), tt(v.rw, f32, {J, T})};
            TG(isa_.at("ink_droute1_gaudi2"), t, s, {T, 1, 1, 1, 1}, 13, tr.signal_count, 14);
        } else {
            const DirectTpcTensor t[] = {tt(v.logits, f32, {E, T}), tt(wa("router_bias"), f32, {E}),
                                         tt(tab + m_.tab_.tau + Inkling2::kMaxT * 4, f32, {E}), tt(v.idx, DType::I32, {J, T}),
                                         tt(v.rw, f32, {J, T})};
            TG(isa_.at("ink_route_gaudi2"), t, s, {T, 1, 1, 1, 1}, 13, tr.signal_count, 14);
        }
    }
    // 10: routed experts (T x 8 slots, deduplicated NVFP4 or MXFP8 GEMVs when available), then the slot sums + MLP conv + outputs
    static const bool vmoe_dedup = [] { const char* e = std::getenv("G2E_INK_VMOE_DEDUP"); return !e || *e != '0'; }();
    const bool use_vmoe = vmoe_dedup && isa_.count("ink_vmoe_plan_gaudi2") &&
                          (nvfp4 ? (isa_.count("ink_vmoe_up_nvf4_gaudi2") && isa_.count("ink_vmoe_dn_nvf4_gaudi2"))
                                 : (isa_.count("ink_vmoe_up_mxf8_gaudi2") && isa_.count("ink_vmoe_dn_mxf8_gaudi2")));
    if (use_vmoe) {
        const uint32_t M = J * T;
        {
            const DirectTpcTensor t[] = {tt(v.idx, DType::I32, {J, T}),
                                         tt(v.grp, DType::I32, {64, 4})};
            TG(isa_.at("ink_vmoe_plan_gaudi2"), t, {}, {1, 1, 1, 1, 1}, 14, sig_of(T), 17);
        }
        if (nvfp4) {
            {
                const DirectTpcTensor t[] = {tt(v.xn, bf, {H, T}),
                                             tt(v.grp, DType::I32, {64, 4}),
                                             tt(wa("w13.wq"), DType::U8, {2 * F, H / 2, E}),
                                             tt(wa("w13.bs"), DType::F8, {2 * F, H / 16, E}),
                                             tt(m_.nv_gvec13(L), f32, {2 * F, E}),
                                             tt(v.act, bf, {F, J, T})};
                const auto s = srf(c.up_softcap);
                TG(isa_.at("ink_vmoe_up_nvf4_gaudi2"), t, s, {9 * M, 1, 1, 1, 1}, 17, 1, 15);
            }
            {
                const DirectTpcTensor t[] = {tt(v.act, bf, {F, J, T}),
                                             tt(v.grp, DType::I32, {64, 4}),
                                             tt(wa("w2.wq"), DType::U8, {H, F / 2, E}),
                                             tt(wa("w2.bs"), DType::F8, {H, F / 16, E}),
                                             tt(m_.nv_gvec2(L), f32, {H, E}),
                                             tt(v.ys, bf, {H, J, T})};
                TG(isa_.at("ink_vmoe_dn_nvf4_gaudi2"), t, {}, {9 * M, 1, 1, 1, 1}, 15, sig_of(9 * M), 16);
            }
        } else {
            {
                const DirectTpcTensor t[] = {tt(v.xn, bf, {H, T}),
                                             tt(v.grp, DType::I32, {64, 4}),
                                             tt(wa("w13.wq"), DType::U8, {2 * F, H, E}),
                                             tt(wa("w13.bs"), DType::F8, {2 * F, H / 32, E}),
                                             tt(m_.nv_gvec13(L), f32, {2 * F, E}),
                                             tt(v.act, bf, {F, J, T})};
                const auto s = srf(c.up_softcap);
                TG(isa_.at("ink_vmoe_up_mxf8_gaudi2"), t, s, {9 * M, 1, 1, 1, 1}, 17, 1, 15);
            }
            {
                const DirectTpcTensor t[] = {tt(v.act, bf, {F, J, T}),
                                             tt(v.grp, DType::I32, {64, 4}),
                                             tt(wa("w2.wq"), DType::U8, {H, F, E}),
                                             tt(wa("w2.bs"), DType::F8, {H, F / 32, E}),
                                             tt(m_.nv_gvec2(L), f32, {H, E}),
                                             tt(v.ys, bf, {H, J, T})};
                TG(isa_.at("ink_vmoe_dn_mxf8_gaudi2"), t, {}, {9 * M, 1, 1, 1, 1}, 15, sig_of(9 * M), 16);
            }
        }
    } else if (nvfp4) {
        if (!isa_.count("ink_dmoe_up9_nvf4_gaudi2") || !isa_.count("ink_dmoe_dn9_nvf4_gaudi2"))
            throw std::runtime_error("inkling2 direct: missing NVFP4 decode MoE kernel objects");
        {
            const DirectTpcTensor t[] = {tt(v.xn, bf, {H, T}), tt(v.idx, DType::I32, {J, T}),
                                         tt(wa("w13.wq"), DType::U8, {2 * F, H / 2, E}),
                                         tt(wa("w13.bs"), DType::F8, {2 * F, H / 16, E}),
                                         tt(m_.nv_gvec13(L), f32, {2 * F, E}), tt(v.act, bf, {F, J, T})};
            const auto s = srf(c.up_softcap);
            TG(isa_.at("ink_dmoe_up9_nvf4_gaudi2"), t, s, {9 * J * T, 1, 1, 1, 1}, 14, sig_of(T), 15);
        }
        {
            const DirectTpcTensor t[] = {tt(v.act, bf, {F, J, T}), tt(v.idx, DType::I32, {J, T}),
                                         tt(wa("w2.wq"), DType::U8, {H, F / 2, E}),
                                         tt(wa("w2.bs"), DType::F8, {H, F / 16, E}),
                                         tt(m_.nv_gvec2(L), f32, {H, E}), tt(v.ys, bf, {H, J, T})};
            TG(isa_.at("ink_dmoe_dn9_nvf4_gaudi2"), t, {}, {9 * J * T, 1, 1, 1, 1}, 15, sig_of(9 * J * T), 16);
        }
    } else {
        if (!isa_.count("ink_dmoe_up9_mxf8_gaudi2") || !isa_.count("ink_dmoe_dn9_mxf8_gaudi2"))
            throw std::runtime_error("inkling2 direct: missing MXFP8 decode MoE kernel objects");
        {
            const DirectTpcTensor t[] = {tt(v.xn, bf, {H, T}), tt(v.idx, DType::I32, {J, T}),
                                         tt(wa("w13.wq"), DType::U8, {2 * F, H, E}),
                                         tt(wa("w13.bs"), DType::F8, {2 * F, H / 32, E}),
                                         tt(m_.nv_gvec13(L), f32, {2 * F, E}), tt(v.act, bf, {F, J, T})};
            const auto s = srf(c.up_softcap);
            TG(isa_.at("ink_dmoe_up9_mxf8_gaudi2"), t, s, {9 * J * T, 1, 1, 1, 1}, 14, sig_of(T), 15);
        }
        {
            const DirectTpcTensor t[] = {tt(v.act, bf, {F, J, T}), tt(v.idx, DType::I32, {J, T}),
                                         tt(wa("w2.wq"), DType::U8, {H, F, E}),
                                         tt(wa("w2.bs"), DType::F8, {H, F / 32, E}),
                                         tt(m_.nv_gvec2(L), f32, {H, E}), tt(v.ys, bf, {H, J, T})};
            TG(isa_.at("ink_dmoe_dn9_mxf8_gaudi2"), t, {}, {9 * J * T, 1, 1, 1, 1}, 15, sig_of(9 * J * T), 16);
        }
    }
    {
        const auto s = srf(conv_s);
        const DirectTpcTensor t[] = {tt(v.ys, bf, {H, J, T}), tt(v.rw, f32, {J, T}), tt(cin + m_.conv_off_[3], bf, {H, 3}),
                                     tt(l.mtaps, bf, {H, 4}), tt(v.res, f32, {H, T}), tt(hist + m_.hist_off(3, T), bf, {H, 3 + T}),
                                     tt(h_out, bf, {H, T}), tt(h_out2, bf, {H, T})};
        TG(isa_.at("ink_dtail_rw_gaudi2"), t, s, {NT, 1, 1, 1, 1}, 16, sig_of(3 * J * T), -1);
    }
    if (mme_cut || (lim > 0 && !done)) return nullptr;
    P.prepare();
    return p;
}

void InkDirect::profile_stages_v(uint32_t app, uint32_t T) {
    for (auto& [key, pl] : plans_v_) {
        const auto [a, sbits, hcur, oaddr, ooff, sfk] = key;
        if (a != app || ((sfk >> 24) & 127) != T) continue;
        const int side = sbits & 1;
        const DeviceBuffer* ob = nullptr;
        for (const DeviceBuffer* b : {&m_.h_[0], &m_.h_[1], &m_.blocks_, &m_.prev_[0], &m_.prev_[1]})
            if (b->address() == oaddr) ob = b;
        if (!ob) return;
        double prev = 0;
        for (int lim = 1; lim <= 48; ++lim) {
            auto pp = build_v(app, T, side, hcur, ob, ooff, lim);
            if (!pp) { std::fprintf(stderr, "[ink-direct] v app %u stage %2d: MME (in the next)\n", app, lim); continue; }
            std::vector<double> ts;
            for (int r = 0; r < 25; ++r) ts.push_back(pp->run());
            std::sort(ts.begin(), ts.end());
            std::fprintf(stderr, "[ink-direct] v app %u T %u stages 1..%2d: %7.1f us (+%.1f)\n", app, T, lim, ts[12], ts[12] - prev);
            prev = ts[12];
        }
        return;
    }
}

bool InkDirect::handles_verify_full(uint32_t T) const {
    static const bool vfull_on = [] { const char* e = std::getenv("G2E_INK_DIRECT_VFULL"); return !e || *e != '0'; }();
    if (!vfull_on || !d_full_on_ || T < 2 || T > kVerifyMaxT) return false;
    if (!isa_.count("ink_dembed_gaudi2") || !isa_.count("ink_dsoftcap_argmax_b_gaudi2")) return false;
    if (!mme_.count("b" + std::to_string(T) + "_lmhead")) return false;
    for (uint32_t i = 0; i < m_.apps_.size(); ++i)
        if (!handles_v(i, T)) return false;
    return true;
}

bool InkDirect::handles_v_commit() const {
    return isa_.count("ink_kv_commit_bf16_gaudi2") != 0 && isa_.count("ink_rollback_gaudi2") != 0;
}

DirectPlan& InkDirect::v_embed_plan(uint32_t T, uint32_t sf) {
    if (auto it = v_embed_plans_.find(T); it != v_embed_plans_.end()) return *it->second;
    auto pl = build_evicting(sf, [&] { return build_b_embed(T); }, -1, true);
    auto& ref = *pl;
    v_embed_plans_[T] = std::move(pl);
    return ref;
}

DirectPlan& InkDirect::v_head_plan(uint32_t T, uint64_t last_addr, bool aux_on, uint32_t sf) {
    const auto key = std::make_tuple(T, last_addr, aux_on);
    if (auto it = v_head_plans_.find(key); it != v_head_plans_.end()) return *it->second;
    auto pl = build_evicting(sf, [&] {
        const auto& c = m_.cfg_;
        const auto& w = m_.w_;
        const uint64_t arena = w.arena().address();
        const uint32_t H = c.hidden, NT = H / 128;
        const float mscale = 1.f / std::sqrt(float(H));
        const uint64_t xn_addr = aux_on ? (m_.aux_.address() + 3ull * Inkling2::kMaxT * H * 2) : vs_.xn;
        const uint64_t q_addr = m_.qil_.address() + m_.qil_off_.at("final_q");
        const auto bf = DType::BF16, f32 = DType::F32;

        auto p = std::make_unique<DirectPlan>(*rt_);
        {
            const DirectTpcTensor t[] = {
                tt(m_.blocks_.address(), bf, {H, T, 4}),
                tt(last_addr, bf, {H, T}),
                tt(q_addr, f32, {H, 1}),
                tt(vs_.p1, f32, {64, NT, T}),
            };
            p->add_tpc_grid(isa_.at("ink_dgram_gaudi2"), t, {}, {NT, T, 1, 1, 1}, 0, 24, -1, 0, 0);
        }
        {
            struct { float scale, eps; int32_t plain; } prm{mscale, c.eps, 0};
            const DirectTpcTensor t[] = {
                tt(vs_.p1, f32, {64, NT, T}),
                tt(m_.blocks_.address(), bf, {H, T, 4}),
                tt(last_addr, bf, {H, T}),
                tt(arena + w.at("final_norm").offset, bf, {H, 1}),
                tt(xn_addr, bf, {H, T, 1}),
            };
            p->add_tpc_grid(isa_.at("ink_dnorm_gaudi2"), t, srf(prm), {NT, T, 1, 1, 1}, 1, 24, 0, NT, 1);
        }
        const auto& tm_lm = mme_.at("b" + std::to_string(T) + "_lmhead");
        p->add_mme(tm_lm, xn_addr, arena + w.at("lm_head").offset, m_.logits_.address(), 0, 1, T, 2);
        {
            struct { float cap, inv_cap; int32_t V, active_B, skip_softcap; } prm{
                c.logit_softcap, 1.f / c.logit_softcap, int32_t(c.vocab), int32_t(T), 0};
            const uint64_t mv_addr = b_argmax_.address();
            const uint64_t mi_addr = mv_addr + uint64_t(T) * 24 * 64 * 4;
            const DirectTpcTensor t[] = {
                tt(m_.logits_.address(), f32, {c.vocab, T}),
                tt(m_.logits_.address(), f32, {c.vocab, T}),
                tt(mv_addr, f32, {64, 24, T}),
                tt(mi_addr, DType::I32, {64, 24, T}),
            };
            p->add_tpc_grid(isa_.at("ink_dsoftcap_argmax_b_gaudi2"), t, srf(prm), {24, 1, 1, 1, 1}, 0, 24, 2,
                            tm_lm.signal_count, -1);
        }
        p->prepare();
        return p;
    }, -1, true);
    auto& ref = *pl;
    v_head_plans_[key] = std::move(pl);
    return ref;
}

void InkDirect::run_verify_full(std::span<const int32_t> tokens, uint32_t pos, float tau0, int side,
                                std::span<const DecodeAppIO> apps, const DeviceBuffer* last_out, uint64_t last_off,
                                std::vector<float>* logits, double* out_embed_ms,
                                double* out_layers_ms, double* out_head_ms) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (!anchor_) { anchor(); sram_const_valid_ = false; }
    const uint32_t T = uint32_t(tokens.size());
    const auto& c = m_.cfg_;
    const int32_t pos32 = int32_t(pos);
    const DirectRuntime::PdmaH2D h2d[3] = {
        {tokens.data(), m_.ids_.address(), T * 4},
        {&pos32, m_.tables_.address() + m_.tab_.pos32, 4},
        {&tau0, m_.tables_.address() + m_.tab_.tau, 4},
    };
    rt_->pdma_h2d_scatter_sync(h2d);
    const auto t1 = clk::now();

    if (!m_.direct_prev_) sram_const_valid_ = false;
    const uint32_t W = c.window;
    const uint32_t sf_cur = m_.far_bucket(pos > W ? pos - W : 0);
    std::vector<DirectPlan*> chain;
    chain.reserve(apps.size() + 2);
    bool verify_ready = false;
    for (int attempt = 0; attempt < kMaxPlanAttempts; ++attempt) {
        chain.clear();
        const uint64_t gen0 = evict_gen_;
        bool restarted = false;
        chain.push_back(&v_embed_plan(T, sf_cur));
        if (evict_gen_ != gen0) restarted = true;
        for (size_t i = 0; !restarted && i < apps.size(); ++i) {
            const auto& io = apps[i];
            const uint32_t sf = c.full[m_.apps_[io.app].layer] ? sf_cur : 0;
            const bool tap_app = m_.aux_on_ && (io.app == 4 || io.app == 22 || io.app == 40);
            const int sbits = side | (int(m_.active_slot_) << 1);
            const auto key = std::make_tuple(io.app, sbits, io.hcur, io.out_buf->address(), io.out_off,
                                             sf | (T << 24) | (uint32_t(tap_app) << 31));
            auto it = plans_v_.find(key);
            if (it == plans_v_.end()) {
                auto p = build_evicting(sf_cur, [&] {
                    return build_v(io.app, T, side, io.hcur, io.out_buf, io.out_off, 0);
                }, -1, true);
                it = plans_v_.emplace(key, std::move(p)).first;
                if (evict_gen_ != gen0) { restarted = true; break; }
            }
            chain.push_back(it->second.get());
        }
        if (!restarted) {
            DirectPlan* hp = &v_head_plan(T, last_out->address() + last_off, m_.aux_on_, sf_cur);
            if (evict_gen_ != gen0) restarted = true;
            else chain.push_back(hp);
        }
        if (!restarted) { verify_ready = true; break; }
    }
    if (!verify_ready) throw std::runtime_error("inkling2 direct: run_verify_full plans failed to stabilize after eviction");
    const double us = rt_->run_chain(chain);
    static const bool tm = std::getenv("G2E_INK_DIRECT_TIME") != nullptr;
    if (tm) {
        auto& v = chain_prof_[uint32_t(apps.size())];
        v.first += us;
        v.second++;
    }
    const auto t2 = clk::now();
    logits->resize(uint64_t(T) * c.vocab);
    rt_->pdma_d2h_sync(m_.logits_.address(), logits->data(), uint64_t(T) * c.vocab * 4);
    const auto t3 = clk::now();
    if (out_embed_ms) *out_embed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (out_layers_ms) *out_layers_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    if (out_head_ms) *out_head_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
}

DirectPlan& InkDirect::v_commit_plan(uint32_t T, uint32_t sf) {
    const auto key = std::make_tuple(T, m_.active_slot_, m_.cap_);
    if (auto it = v_commit_plans_.find(key); it != v_commit_plans_.end()) return *it->second;
    auto pl = build_evicting(sf, [&] {
        const auto& c = m_.cfg_;
        const uint32_t KV = c.kv_heads, D = c.head_dim, W = c.window, KVD = KV * D;
        const uint64_t tab = m_.tables_.address();
        const uint64_t rr_addr = tab + m_.tab_.rows;
        const uint64_t rl_addr = rr_addr + Inkling2::kMaxT * 4;
        const uint64_t isa = isa_.at("ink_kv_commit_bf16_gaudi2");
        const auto bf = DType::BF16;
        auto p = std::make_unique<DirectPlan>(*rt_);
        for (const auto& a : m_.apps_) {
            const uint64_t kvs = m_.kvscratch_.address() + uint64_t(a.index) * 2 * KVD * Inkling2::kMaxT * 2;
            const DirectTpcTensor tr[] = {
                tt(kvs, bf, {D, T, KV, 2}),
                tt(rr_addr, DType::I32, {T}),
                tt(m_.kring_addr(a.index, m_.active_slot_), bf, {D, W, KV}),
                tt(m_.vring_addr(a.index, m_.active_slot_), bf, {D, W, KV}),
            };
            p->add_tpc_grid(isa, tr, {}, {KV, T, 1, 1, 1}, 0, 24, -1, 0, -1);
            if (c.full[a.layer]) {
                const DirectTpcTensor tl[] = {
                    tt(kvs, bf, {D, T, KV, 2}),
                    tt(rl_addr, DType::I32, {T}),
                    tt(m_.kcache_addr(a.index, m_.active_slot_), bf, {D, m_.cap_, KV}),
                    tt(m_.vcache_addr(a.index, m_.active_slot_), bf, {D, m_.cap_, KV}),
                };
                p->add_tpc_grid(isa, tl, {}, {KV, T, 1, 1, 1}, 0, 24, -1, 0, -1);
            }
        }
        p->prepare();
        return p;
    }, -1, true);
    auto& ref = *pl;
    v_commit_plans_[key] = std::move(pl);
    return ref;
}

DirectPlan& InkDirect::rollback_plan(int side_out, uint32_t T, uint32_t n, uint32_t sf) {
    const auto key = std::make_tuple(side_out, T, n, m_.active_slot_);
    if (auto it = rollback_.find(key); it != rollback_.end()) return *it->second;
    auto pl = build_evicting(sf, [&] {
        const auto& c = m_.cfg_;
        const uint32_t KVD = c.kv_heads * c.head_dim, H = c.hidden, A = uint32_t(m_.apps_.size());
        auto p = std::make_unique<DirectPlan>(*rt_);
        const uint64_t cbase = m_.conv_pool_.address() + m_.conv_off_slot(m_.active_slot_, side_out);
        const DirectTpcTensor t[] = {tt(m_.hist_.address(), DType::BF16, {uint32_t(m_.hist_.size() / 2)}),
                                     tt(cbase, DType::BF16, {uint32_t(m_.conv_slot_bytes_ / 2)})};
        int32_t prm[14] = {int32_t(m_.hist_app_ / 2), int32_t(m_.conv_app_bytes_ / 2)};
        for (int i = 0; i < 4; ++i) {
            prm[2 + i] = int32_t(m_.hist_off(i, T) / 2);
            prm[6 + i] = int32_t(m_.conv_off_[i] / 2);
        }
        prm[10] = int32_t(KVD);
        prm[11] = int32_t(H);
        prm[12] = int32_t(n);
        prm[13] = int32_t(T);
        const auto s = srf(prm);
        const uint32_t pts = A * (2 * KVD / 128 + 2 * H / 128);
        p->add_tpc_grid(isa_.at("ink_rollback_gaudi2"), t, s, {pts, 1, 1, 1, 1}, 0, 24, -1, 0, -1);
        p->prepare();
        return p;
    }, -1, true);
    auto& ref = *pl;
    rollback_[key] = std::move(pl);
    return ref;
}

void InkDirect::rollback(int side_out, uint32_t T, uint32_t n) {
    const uint32_t W = m_.cfg_.window, pos = m_.pos_;
    const uint32_t sf_cur = m_.far_bucket(pos > W ? pos - W : 0);
    rollback_plan(side_out, T, n, sf_cur).run();
}

void InkDirect::commit_and_rollback(int side_out, uint32_t T, uint32_t n) {
    const uint32_t W = m_.cfg_.window, pos = m_.pos_;
    const uint32_t sf_cur = m_.far_bucket(pos > W ? pos - W : 0);
    if (n == 0) {
        rollback_plan(side_out, T, 0, sf_cur).run();
        return;
    }
    alignas(64) int32_t rr[8] = {}, rl[8] = {};
    for (uint32_t r = 0; r < T; ++r) {
        rr[r] = r < n ? int32_t((pos + r) % W) : -1;
        rl[r] = r < n ? int32_t(pos + r) : -1;
    }
    const uint64_t tab = m_.tables_.address();
    const DirectRuntime::PdmaH2D h2d[2] = {
        {rr, tab + m_.tab_.rows, T * 4},
        {rl, tab + m_.tab_.rows + Inkling2::kMaxT * 4, T * 4},
    };
    rt_->pdma_h2d_scatter_sync(h2d);
    DirectPlan* chain[2] = {nullptr, nullptr};
    bool ready = false;
    for (int attempt = 0; attempt < kMaxPlanAttempts; ++attempt) {
        const uint64_t gen0 = evict_gen_;
        chain[0] = &v_commit_plan(T, sf_cur);
        if (evict_gen_ != gen0) continue;
        chain[1] = &rollback_plan(side_out, T, n, sf_cur);
        if (evict_gen_ != gen0) continue;
        ready = true;
        break;
    }
    if (!ready) throw std::runtime_error("inkling2 direct: commit_and_rollback plans failed to stabilize after eviction");
    rt_->run_chain(chain);
}

}  // namespace g2e
