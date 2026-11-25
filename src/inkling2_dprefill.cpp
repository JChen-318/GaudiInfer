// Inkling-2 prefill chunks (T = 1024) on the Direct engine: Plan A (attention, convolutions, router, routing table)
// and Plan B (dynamic BF16 MoE and post-MLP conv/residual), using MME templates p_* (tools/mme_descs_ink) and ink_*
// TPC kernels synchronised by sync objects. Same math as the Synapse layer recipe (layer_recipe) with the graph's
// glue ops folded into kernels: r / sink prep (ink_prprep, BF16 sink read by ink_attn_sm16b), the attention output
// gate (ink_pgate), the convs with head-major outputs or the residuals (ink_conv_hm / resa / resm), the MoE routing
// plan (ink_rplan_*: histogram, flat offsets, wave lists) and the slot sum (ink_moe_combine_bf16).
// G2E_INK_DPREFILL: bit mask of application kinds (see docs/inkling2-direct-plan.md).
#include "g2e/inkling2_direct.hpp"

#include "g2e/graph.hpp"
#include "g2e/inkling2.hpp"
#include "g2e/numeric.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

namespace g2e {
namespace {

std::string Lp(uint32_t i, const char* part) { return "L" + std::to_string(i) + "." + part; }
uint64_t al256(uint64_t x) { return (x + 255) & ~uint64_t(255); }

// a TPC tensor: sizes dim 0 first; strides in elements (0: compact)
DirectTpcTensor tn(uint64_t addr, DType dt, std::initializer_list<uint32_t> sizes, std::initializer_list<uint32_t> strides = {}) {
    DirectTpcTensor t{};
    t.addr = addr;
    t.dtype = dt;
    t.dims = uint32_t(sizes.size());
    uint32_t i = 0;
    for (auto s : sizes) t.sizes[i++] = s;
    for (; i < 5; ++i) t.sizes[i] = 1;
    i = 0;
    for (auto s : strides) t.strides[i++] = s;
    return t;
}
template <typename P>
std::vector<uint32_t> srfv(const P& p) {
    std::vector<uint32_t> v(sizeof(P) / 4);
    std::memcpy(v.data(), &p, sizeof(P));
    return v;
}

constexpr uint32_t kT = 1024;    // rows per prefill chunk
// the Direct tables of a chunk slot (bytes): first position (i32), tau f32 [kT], expert iota f32 [128],
// 24-way token partition tpart i32 [25] (tpart[24] == valid_T <= kT), ring rows i32 [kT], linear cache rows i32 [kT]
constexpr uint64_t kTabPos = 0, kTabTau = 256, kTabIota = kTabTau + kT * 4, kTabTpart = 5120, kTabValidT = kTabTpart + 24 * 4,
                   kTabRows = 8192, kTabBytes = 16384;

// Stage durations (us, T = 1024) for the interleaving of two applications (prioritizes starting Plan B's MoE early
// so MME stays fed during Plan A's attention/message-gate TPC stages).
float stage_us(const std::string& k, uint32_t reps, uint32_t sf) {
    static const std::map<std::string, float> tpc = {
        {"ink_mix_gaudi2", 66}, {"ink_mix_f32p_gaudi2", 62}, {"ink_pnorm_gaudi2", 12}, {"ink_pcopy_gaudi2", 4},
        {"ink_qprep_bf16_gaudi2", 52}, {"ink_qprep_tau_bf16_gaudi2", 55}, {"ink_prprep_gaudi2", 29}, {"ink_prprep_tau_gaudi2", 31},
        {"ink_conv_hm_bf16_gaudi2", 35}, {"ink_conv_dyn_hm_bf16_gaudi2", 60}, {"ink_conv_hm2_bf16_gaudi2", 36}, {"ink_conv_dyn_hm2_bf16_gaudi2", 62},
        {"ink_attn_sm16b_gaudi2", 50}, {"ink_attn_sm16_near_gaudi2", 52}, {"ink_attn_sm16w_gaudi2", 32}, {"ink_zero_gaudi2", 6},
        {"ink_pgate_gaudi2", 70}, {"ink_pgate_far_gaudi2", 75}, {"ink_kv_commit_bf16_gaudi2", 5}, {"ink_conv_resa_bf16_gaudi2", 19},
        {"ink_conv_resm_bf16_gaudi2", 20}, {"ink_mg_gaudi2", 50}, {"ink_mg_out_pade_gaudi2", 138}, {"ink_mg_out_gaudi2", 200}, {"ink_route_gaudi2", 40},
        {"ink_rplan_hist_gaudi2", 16}, {"ink_rplan_dyn_tab_bf16_gaudi2", 4},
        {"ink_rplan_dyn_slots_bf16_gaudi2", 46}, {"ink_sgather_bf16_gaudi2", 43},
        {"ink_act_bf16_gaudi2", 86}, {"ink_moe_combine_bf16_gaudi2", 120}};
    static const std::map<std::string, float> mme = {
        {"p_proj_s_m512", 55}, {"p_projq_m512", 40}, {"p_projkv_m512", 8.2f}, {"p_projkvf_m512", 8.6f}, {"p_bias_r512", 0.68f},
        {"p_qk_m512", 1.27f}, {"p_pv2k", 9.9f}, {"p_wo_m512", 21}, {"p_router", 6.2f}, {"p_mgd1", 8.4f}, {"p_mgf1_m512", 7.2f},
        {"p_bdyngu", 185.f}, {"p_bdyndn", 93.f}};
    if (k == "ink_attn_sm16fb_gaudi2") return 61.f + 0.012f * float(sf);
    if (k == "p_fqk" || k == "p_fpv") return float(reps) * (1.f + 0.0009f * float(sf));
    if (auto it = tpc.find(k); it != tpc.end()) return it->second;
    if (auto it = mme.find(k); it != mme.end()) return it->second * float(reps);
    return 20.f;
}

// BF16 MoE dynamic grouped-GEMM (1 row per slot).
// Worst-case allocated flat rows across all 128 experts for any routing of S = 8192 slots:
//   sum_e alloc_rows(c_e) <= 8192 + 128*31 = 12160 <= 12288 -> ZERO overflow for ALL distributions!
constexpr uint32_t kFlatRows = 12288;
constexpr int kPbSobSl = 0, kPbSobQ = 1, kPbSobGu = 2, kPbSobA = 3, kPbSobDn = 4, kPbSobMo = 5;

// Exact flat row allocation for an expert with c slots; MUST match alloc_rows_v() in ink_rplan_dyn_tab_bf16_gaudi2.c!
static uint32_t dyn_alloc_rows(uint32_t c) {
    const uint32_t q = c >> 8, r = c & 255u;
    uint32_t rr = 256;
    if (r == 0) rr = 0;
    else if (r <= 16) rr = 16;
    else if (r <= 32) rr = 32;
    else if (r <= 64) rr = 64;
    else if (r <= 96) rr = 96;
    else if (r <= 128) rr = 128;
    else if (r <= 144) rr = 144;
    else if (r <= 160) rr = 160;
    else if (r <= 192) rr = 192;
    else if (r <= 224) rr = 224;
    return (q << 8) + rr;
}

// BF16 tile sizes indexed 0..6: {16, 32, 64, 96, 128, 192, 256}
static constexpr const char* kBGtNames[7] = {"p_bgt16", "p_bgt32", "p_bgt64", "p_bgt96", "p_bgt128", "p_bgt192", "p_bgt256"};
static constexpr const char* kBDtNames[7] = {"p_bdt16", "p_bdt32", "p_bdt64", "p_bdt96", "p_bdt128", "p_bdt192", "p_bdt256"};

// Append optimal MME tile jobs for expert e with c slots starting at flat row `row0` into per-tile-id buckets.
// Sum of emitted tile rows equals dyn_alloc_rows(c).
template <typename Emit>
static void dyn_emit_tiles(uint32_t c, uint32_t row0, Emit&& emit) {
    while (c >= 256) {
        emit(6, row0);
        row0 += 256u;
        c -= 256;
    }
    if (c == 0) return;
    if (c <= 16) emit(0, row0);
    else if (c <= 32) emit(1, row0);
    else if (c <= 64) emit(2, row0);
    else if (c <= 96) emit(3, row0);
    else if (c <= 128) emit(4, row0);
    else if (c <= 144) { emit(4, row0); emit(0, row0 + 128u); }
    else if (c <= 160) { emit(4, row0); emit(1, row0 + 128u); }
    else if (c <= 192) emit(5, row0);
    else if (c <= 224) { emit(4, row0); emit(3, row0 + 128u); }
    else emit(6, row0);
}

}  // namespace

void InkDirect::init_prefill() {
    const char* e = std::getenv("G2E_INK_DPREFILL");
    pmask_ = e ? uint32_t(std::strtoul(e, nullptr, 0)) : 0;
    if (!pmask_) return;
    const auto& c = m_.cfg_;
    const char* td = std::getenv("G2E_TPC_OBJ_DIR");
    const std::string tpc = (td && *td) ? td : std::string(G2E_TPC_OBJ_DIR);
    for (const char* k : {"ink_mix_gaudi2", "ink_mix_f32p_gaudi2", "ink_qprep_bf16_gaudi2", "ink_prprep_gaudi2", "ink_conv_hm_bf16_gaudi2",
                          "ink_pnorm_gaudi2", "ink_qprep_tau_bf16_gaudi2", "ink_prprep_tau_gaudi2", "ink_conv_dyn_hm_bf16_gaudi2", "ink_conv_hm2_bf16_gaudi2", "ink_conv_dyn_hm2_bf16_gaudi2",
                          "ink_mg_gaudi2", "ink_mg_out_pade_gaudi2", "ink_mg_out_gaudi2", "ink_attn_sm16fb_gaudi2", "ink_pgate_far_gaudi2",
                          "ink_kv_commit_bf16_gaudi2",
                          "ink_conv_resa_bf16_gaudi2", "ink_conv_resm_bf16_gaudi2", "ink_pcopy_gaudi2", "ink_attn_sm16b_gaudi2",
                          "ink_attn_sm16_near_gaudi2", "ink_attn_far_step_gaudi2", "ink_pfar_head_io_gaudi2", "ink_attn_sm16w_gaudi2",
                          "ink_pgate_gaudi2", "ink_route_gaudi2", "ink_rplan_hist_gaudi2", "ink_rplan_dyn_tab_bf16_gaudi2",
                          "ink_rplan_dyn_slots_bf16_gaudi2", "ink_sgather_bf16_gaudi2", "ink_act_bf16_gaudi2",
                          "ink_moe_combine_bf16_gaudi2", "ink_zero_gaudi2"})
        if (!isa_.count(k)) isa_[k] = rt_->load_tpc_elf(tpc + "/" + k + ".o");
    if (c.has_nvfp4() && !isa_.count("ink_nvf4_deq_bf16_gaudi2"))
        isa_["ink_nvf4_deq_bf16_gaudi2"] = rt_->load_tpc_elf(tpc + "/ink_nvf4_deq_bf16_gaudi2.o");
    if (c.has_mxfp8() && !isa_.count("ink_mxfp8_deq_bf16_gaudi2"))
        isa_["ink_mxfp8_deq_bf16_gaudi2"] = rt_->load_tpc_elf(tpc + "/ink_mxfp8_deq_bf16_gaudi2.o");
    const char* dd = std::getenv("G2E_INK_MME_DIR");
    const std::string mme = (dd && *dd) ? dd : std::string(G2E_SOURCE_DIR) + "/tools/mme_descs_ink";
    for (const char* k : {"p_proj_s_m512", "p_projq_m512", "p_projkv_m512", "p_projkvf_m512", "p_bias_r512", "p_qk_m512", "p_pv2k",
                          "p_wo_m512", "p_router", "p_mgd1", "p_mgf1_m512"})
        mme_[k] = MmeTemplate::load_dir(mme + "/" + k + "/mme_debug");
    for (int k = 0; k < 7; ++k) {
        mme_[kBGtNames[k]] = MmeTemplate::load_dir(mme + "/" + kBGtNames[k] + "/mme_debug");
        mme_[kBDtNames[k]] = MmeTemplate::load_dir(mme + "/" + kBDtNames[k] + "/mme_debug");
    }
    static constexpr uint32_t kPFarBuckets[] = {1024, 2048, 3072, 4096, 6144, 8192, 9216, 13824, 16384,
                                                20736, 31104, 32768, 46656, 65536, 69984, 104976, 131072};
    for (uint32_t b : kPFarBuckets) {
        if (b > m_.max_ctx_ && b > 3072u) continue;
        for (const char* kind : {"p_fqk_", "p_fpv_"}) {
            const std::string n = kind + std::to_string(b) + "_m512";
            if (!mme_.count(n) && std::filesystem::exists(mme + "/" + n + "/mme_debug"))
                mme_[n] = MmeTemplate::load_dir(mme + "/" + n + "/mme_debug");
        }
    }
    if (m_.max_ctx_ <= 131072u) {
        for (const char* kind : {"p_fqk_", "p_fpv_"}) {
            const std::string n = kind + std::to_string(m_.max_ctx_) + "_m512";
            if (!mme_.count(n) && std::filesystem::exists(mme + "/" + n + "/mme_debug"))
                mme_[n] = MmeTemplate::load_dir(mme + "/" + n + "/mme_debug");
        }
    }
    // per layer constants: r_scale interleaved per 128 ([even 64 | odd 64])
    const uint64_t H = c.hidden, E = c.experts, NH = c.heads, DR = c.d_rel;
    std::vector<uint8_t> host;
    auto put = [&](const void* p, uint64_t bytes) {
        host.resize(al256(host.size()));
        const uint64_t o = host.size();
        host.insert(host.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + bytes);
        return o;
    };
    for (uint32_t L = 0; L < c.layers; ++L) {
        PLayerConst l;
        {
            const auto raw = m_.w_.read(Lp(L, "r_scale"));
            std::vector<float> rs(NH * DR);
            std::memcpy(rs.data(), raw.data(), rs.size() * 4);
            const auto il = DirectRuntime::interleave_f32_scales(rs);
            l.rsil = put(il.data(), il.size() * 4);
        }
        plc_[L] = l;
    }
    {   // BF16 ones [4, H]: the message mix has no norm weights
        std::vector<uint16_t> one(4 * H, to_bf16(1.f));
        ones4_ = put(one.data(), one.size() * 2);
    }
    host.resize(al256(host.size()));
    pconst_ = DeviceBuffer(m_.dev_, host.size());
    {
        HostBuffer hb(m_.dev_, host.size());
        std::memcpy(hb.data(), host.data(), host.size());
        m_.stream_.upload(hb, 0, pconst_, 0, host.size());
        m_.stream_.synchronize();
    }
    for (auto& [L, l] : plc_) l.rsil += pconst_.address();
    ones4_ += pconst_.address();
    // scratch (HBM): the activations and the lossless MoE flat row buffers of one chunk, two sets (two chunks interleaved)
    const uint64_t T = kT, KV = c.kv_heads, D = c.head_dim, W = c.window, RE = c.rel_extent, F = c.moe_inter, K = c.topk;
    const uint64_t G = NH / KV, GT = G * T, QS = NH * D * 2 + NH * DR + NH, KRF = KV * D + KV * 4, S = T * K;
    uint64_t at = 0;
    auto take = [&](uint64_t bytes) { const uint64_t o = at; at = al256(at + std::max<uint64_t>(bytes, 256)); return o; };
    PScratch p{};
    p.xa = take(3 * T * H * 2);
    p.row = take(T * (QS + 2 * KRF) * 2);
    p.q = take(NH * T * D * 2);
    p.r16 = take(T * NH * 16 * 2);
    p.bt = take(T * NH * RE * 2);
    p.kvs = take(2 * KV * T * D * 2);
    p.vcat = take(KV * (W + T) * D * 2);
    p.sr = take(KV * GT * W * 2);
    p.sw = take(KV * GT * T * 2);
    p.p = take(KV * GT * (W + T) * 2);
    p.pw = take(KV * GT * (W + T) * 2);  // P of the windowed softmax (ink_attn_sm16w): zero outside the rows' windows
    p.sums = take(KV * GT * 4);
    p.o = take(KV * GT * D * 4);
    p.of = take(KV * GT * D * 4);
    p.gated = take(T * NH * D * 2);
    p.a1 = take(T * H * 2);
    p.res = take(T * H * 4);
    p.xn = take(T * H * 2);
    p.mixd = take(4 * T * H * 4);  // the mix kernels' unused output (up to 4 queries)
    p.logits = take(T * E * 4);
    p.idx = take(S * 4);
    p.rw = take(S * 4);
    p.hist = take(24 * 128 * 4);
    p.off = take(24 * 128 * 4);
    p.cnt = take(128 * 4);
    p.exp_base = take(128 * 4);
    p.code = take(S * 4);
    p.xil = take(3 * T * H * 4);
    p.msg = take(4 * T * H * 4);
    p.cat = take(3 * T * 2 * H * 2);
    p.gr = take(3 * T * c.gate_rank * 2);
    p.fgv = take(3 * T * 2 * H * 4);
    p.zero = take(T * H * 2);
    p.lbuf = take(4 * S * 4);
    p.lcnt = take(4 * 24 * 4);
    // flat row space (kFlatRows = 12288 BF16 rows; gu [2F, R] F32 is reused as y [H, R] F32 per wave since 2F == H)
    p.xq = take(uint64_t(kFlatRows) * H * 2);
    p.gu = take(uint64_t(kFlatRows) * std::max(2 * F, H) * 4);
    p.aq = take(uint64_t(kFlatRows) * F * 2);
    p.mo = take(T * H * 2);
    for (int sl = 0; sl < 2; ++sl) {
        pscratch_[sl] = DeviceBuffer(m_.dev_, at);
        PScratch q = p;
        const uint64_t base = pscratch_[sl].address();
        for (uint64_t* f : {&q.xa, &q.row, &q.q, &q.r16, &q.bt, &q.kvs, &q.vcat, &q.sr, &q.sw, &q.p, &q.pw, &q.sums, &q.o, &q.of, &q.gated,
                            &q.a1, &q.res, &q.xn, &q.mixd, &q.logits, &q.idx, &q.rw, &q.hist, &q.off, &q.cnt, &q.exp_base,
                            &q.code, &q.xil, &q.msg, &q.cat, &q.gr, &q.fgv, &q.zero, &q.lbuf, &q.lcnt,
                            &q.xq, &q.gu, &q.aq, &q.mo})
            *f += base;
        ps_[sl] = q;
        pdyn_[sl] = std::make_unique<DirectDynMmeSet>(*rt_, 8, 48, 384u << 10);
    }
    // Cache hot patch_b pointers and reusable job buffers once (eliminates map lookups + heap allocs on every patch_b).
    for (int k = 0; k < 7; ++k) {
        pbgt_t_[k] = &mme_.at(kBGtNames[k]);
        pbdt_t_[k] = &mme_.at(kBDtNames[k]);
        pgu_b_[k].reserve(32);
        pdn_b_[k].reserve(32);
    }
    pgu_jobs_.reserve(48);
    pdn_jobs_.reserve(48);
    ptab_ = DeviceBuffer(m_.dev_, 2 * kTabBytes);
    ptab_host_ = std::make_unique<HostBuffer>(m_.dev_, 2 * kTabBytes);
    {   // the boundary layers' zero residual input
        HostBuffer hb(m_.dev_, T * H * 2);
        std::memset(hb.data(), 0, T * H * 2);
        for (int sl = 0; sl < 2; ++sl) m_.stream_.upload(hb, 0, pscratch_[sl], p.zero, T * H * 2);
        m_.stream_.synchronize();
    }
    {   // zeros in the windowed softmax's P (it only writes the rows' windows)
        const uint64_t nb = KV * GT * (W + T) * 2;
        HostBuffer hb(m_.dev_, nb);
        std::memset(hb.data(), 0, nb);
        for (int sl = 0; sl < 2; ++sl) m_.stream_.upload(hb, 0, pscratch_[sl], p.pw, nb);
        m_.stream_.synchronize();
    }
    // Plan-A on-chip SRAM overlay (G2E_INK_DPREFILL_SRAM=1 default; off under G2E_INK_DPCHECK):
    // Always keeps 2-buffered sr[2] (12 MiB) + sw[2] (12 MiB) for full MME/TPC attention overlap, plus gated (6 MiB)
    // and a1 (4.5 MiB) aliasing sr[0..1] after SM(7) and MLP fgv (18 MiB) + gr (384 KiB) after conv_resa:
    //   - Full 48.00 MiB free (tpc_barrier_in_sram=false): adds pw[2] [24..48 MiB) (zeroed once per run_pair/queue_p)
    //   - <48 MiB (tpc_barrier_in_sram=true): leaves top 18 MiB untouched for SCAL barrier; puts q [24..30 MiB) in SRAM
    {
        const char* se = std::getenv("G2E_INK_DPREFILL_SRAM");
        const bool want_sram = (!se || *se != '0') && !std::getenv("G2E_INK_DPCHECK");
        const char* xe = std::getenv("G2E_INK_DPREFILL_SRAM_EXTRA");
        const bool want_extra = !xe || *xe != '0';
        const uint64_t sb = al256(rt_->sram_base());
        const uint64_t sa = (rt_->sram_base() + rt_->sram_usable()) > sb ? (rt_->sram_base() + rt_->sram_usable() - sb) : 0;
        const uint64_t sr2 = 2ull * GT * W * 2;        // 12 MiB
        const uint64_t sw2 = 2ull * GT * T * 2;        // 12 MiB
        const uint64_t pw2 = 2ull * GT * (W + T) * 2;  // 24 MiB
        const uint64_t qsz = uint64_t(NH) * T * D * 2; // 6 MiB
        if (want_sram && KV >= 2 && sa >= sr2 + sw2) {
            psram_on_ = true;
            psram_sr_ = sb;                             // [0 .. 12 MiB): sr[0..1]
            psram_sw_ = sb + sr2;                       // [12 .. 24 MiB): sw[0..1]
            psram_fgv_ = sb;                            // [0 .. 18 MiB): MLP 1-gate fgv (after conv_resa)
            psram_gr_ = sb + uint64_t(T) * 2 * H * 4;   // [18 .. 18.375 MiB): attn 3-gate gr
            if (want_extra) {
                psram_extra_ = true;
                psram_gated_ = psram_sr_;               // [0 .. 6 MiB): gated (after SM(7))
                psram_a1_ = psram_sr_ + uint64_t(T) * NH * D * 2;  // [6 .. 10.5 MiB): a1 (after p_wo)
            }
            if (sa >= sr2 + sw2 + pw2) {
                psram_pw_ = sb + sr2 + sw2;             // [24 .. 48 MiB): pw[0..1]
                psram_pw_on_ = true;
                pzero_pw_ = std::make_unique<DirectPlan>(*rt_);
                const uint32_t ne = uint32_t(pw2 / 2);
                const DirectTpcTensor zt[] = {tn(psram_pw_, DType::BF16, {ne})};
                pzero_pw_->add_tpc_grid(isa_.at("ink_zero_gaudi2"), zt, {}, {ne / 128, 1, 1, 1, 1}, 0, 24, -1, 0, -1);
                pzero_pw_->prepare();
            } else if (want_extra && sa >= sr2 + sw2 + qsz) {
                psram_q_ = sb + sr2 + sw2;              // [24 .. 30 MiB): q (when pw stays in HBM)
            }
        }
        // Hybrid far-key policy (G2E_INK_FAR_FLASH: unset/"auto" default, "1" = 0-HBM pure FlashAttention, "0" = legacy HBM):
        //   - fr <= 1024: inline SRAM [24..48 MiB) when pw_on (0 HBM pfar_, faster than HBM, 100% bit-exact)
        //   - 1024 < fr <= pfar_cap_: single-plan HBM pfar_ (capped at 32K = 768 MiB max, frees 2.45 GiB at 128K!)
        //   - fr > pfar_cap_ (or when pfar_ fails to allocate): tiled in-place 48 MiB SRAM FlashAttention (tw=3072 -> 1M!)
        const char* fe = std::getenv("G2E_INK_FAR_FLASH");
        const bool force_off = fe && *fe == '0';
        const bool force_pure = fe && *fe == '1';
        const bool want_flash = !force_off && !std::getenv("G2E_INK_DPCHECK");
        auto need_pfar_sram = [&](uint32_t tw) -> uint64_t {
            return 2ull * GT * tw * 2 + 3ull * GT * D * 4 + uint64_t(GT) * D * 2 + 6ull * al256(uint64_t(GT) * 4);
        };
        auto try_tw = [&](uint32_t tw) -> bool {
            return sa >= need_pfar_sram(tw) &&
                   mme_.count("p_fqk_" + std::to_string(tw) + "_m512") &&
                   mme_.count("p_fpv_" + std::to_string(tw) + "_m512");
        };
        pfar_tw_ = 0;
        if (want_flash && psram_on_) {
            if (const char* twe = std::getenv("G2E_INK_PFAR_TW")) {
                const uint32_t req = uint32_t(std::atoi(twe));
                if ((req == 1024u || req == 2048u || req == 3072u) && try_tw(req)) pfar_tw_ = req;
            }
            if (!pfar_tw_) {
                for (uint32_t tw : {3072u, 2048u, 1024u}) {
                    if (try_tw(tw)) { pfar_tw_ = tw; break; }
                }
            }
        }
        pfar_flash_ = pfar_tw_ > 0;
        if (pfar_flash_) {
            const uint64_t tile_sz = uint64_t(GT) * pfar_tw_ * 2;
            const uint64_t o_sz = uint64_t(GT) * D * 4;
            const uint64_t q_sz = uint64_t(GT) * D * 2;
            const uint64_t v_sz = al256(uint64_t(GT) * 4);
            pfar_sf_pf_[0] = sb;
            pfar_sf_pf_[1] = sb + tile_sz;
            pfar_of_tmp_[0] = sb + 2 * tile_sz;
            pfar_of_tmp_[1] = pfar_of_tmp_[0] + o_sz;
            pfar_o_h_ = pfar_of_tmp_[1] + o_sz;
            pfar_q_h_ = pfar_o_h_ + o_sz;
            pfar_m_h_ = pfar_q_h_ + q_sz;
            pfar_l_h_ = pfar_m_h_ + v_sz;
            pfar_fres_ = pfar_l_h_ + v_sz;
        }
        const uint32_t want_cap = (pfar_flash_ && force_pure) ? 0u
                                : pfar_flash_                 ? std::min<uint32_t>(m_.max_ctx_, 32768u)
                                                              : m_.max_ctx_;
        if (want_cap > 0) {
            try {
                pfar_ = DeviceBuffer(m_.dev_, 4 * GT * uint64_t(want_cap) * 2);
                pfar_ok_ = true;
                pfar_cap_ = want_cap;
            } catch (const std::exception& ex) {
                pfar_cap_ = 0;
                std::fprintf(stderr, "[ink-direct] no memory for prefill far HBM (%s): %s\n", ex.what(),
                             pfar_flash_ ? "using 0-HBM SRAM FlashAttention" : "full layers with far keys stay on Synapse");
            }
        }
        std::fprintf(stderr, "[ink-direct] prefill plans on (mask %#x): scratch 2 x %.0f MiB, SRAM %.2f MiB (on=%d, pw=%d, extra=%d, flash=%d tw=%u, far_hbm=%.0f MiB cap=%u)\n",
                     pmask_, double(at) / (1 << 20), double(sa) / (1 << 20), int(psram_on_), int(psram_pw_on_), int(psram_extra_), int(pfar_flash_), pfar_tw_,
                     pfar_ok_ ? double(pfar_.size()) / (1 << 20) : 0.0, pfar_cap_);
    }
    if (std::getenv("G2E_INK_DPREFILL_LIST")) {
        std::string ks;
        for (const auto& a : m_.apps_)
            ks += std::string(c.full[a.layer] ? "F" : "S") + (a.layer % c.attn_res_block == 0 ? "b" : "") + (a.pass > 0 ? "m" : "") + " ";
        std::fprintf(stderr, "[ink-direct] application kinds: %s\n", ks.c_str());
    }
}

// far key rows of the full layers of a chunk starting at pos (the cache rows before the ring's positions)
static uint32_t far_rows_at(uint32_t pos, uint32_t W) { return pos > W ? pos - W : 0; }

bool InkDirect::handles_p(uint32_t app, uint32_t T, uint32_t pos) const {
    if (!pmask_ || T != kT || m_.verify_ || app >= m_.apps_.size()) return false;
    static const int only = std::getenv("G2E_INK_DPREFILL_APP") ? std::atoi(std::getenv("G2E_INK_DPREFILL_APP")) : -1;
    if (only >= 0 && int(app) != only) return false;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const bool full = m_.cfg_.full[L], boundary = L % m_.cfg_.attn_res_block == 0, msg = a.pass > 0;
    if (const uint32_t fr = far_rows_at(pos, m_.cfg_.window); full && fr) {  // far keys: FlashAttention SRAM or HBM bucket
        if (!pfar_flash_ && (!mme_.count("p_fqk_" + std::to_string(m_.far_bucket(fr)) + "_m512") || !pfar_ok_)) return false;
    }
    // bit 1: sliding without messages; 2: boundary layers (layer 0 too); 4: message applications; 8: full layers
    if (!(pmask_ & 2) && boundary) return false;
    if (!(pmask_ & 4) && msg) return false;
    if (!(pmask_ & 8) && full) return false;
    return (pmask_ & 1) != 0 || boundary || msg || full;
}
bool InkDirect::handles_p(uint32_t app, uint32_t T) const { return handles_p(app, T, m_.pos_); }

// the Direct tables of chunk slot `slot` for a chunk of T rows (first valid_T real, rest masked padding) starting at pos
void InkDirect::set_tables(int slot, uint32_t pos, uint32_t T, uint32_t valid_T) {
    if (!pmask_ || !ptab_host_) return;
    const auto& c = m_.cfg_;
    const uint32_t W = c.window;
    const uint32_t vt = (valid_T && valid_T <= T) ? valid_T : T;
    auto* base = ptab_host_->as<uint8_t>() + uint64_t(slot) * kTabBytes;
    *reinterpret_cast<int32_t*>(base + kTabPos) = int32_t(pos);
    auto* tau = reinterpret_cast<float*>(base + kTabTau);
    for (uint32_t t = 0; t < T; ++t) {
        const double x = double(int64_t(pos) + t + 1) / double(c.log_floor);
        tau[t] = float(1.0 + double(c.log_alpha) * std::log(std::max(x, 1.0)));
    }
    auto* iota = reinterpret_cast<float*>(base + kTabIota);
    for (uint32_t e = 0; e < c.experts; ++e) iota[e] = float(e);
    auto* tp = reinterpret_cast<int32_t*>(base + kTabTpart);
    for (uint32_t i = 0; i <= 24; ++i) tp[i] = std::min<int32_t>(int32_t(kT * i / 24), int32_t(vt));
    auto* rr = reinterpret_cast<int32_t*>(base + kTabRows);
    auto* rl = rr + kT;
    for (uint32_t t = 0; t < T; ++t) {
        rr[t] = (t >= vt || (vt > W && t < vt - W)) ? -1 : int32_t((pos + t) % W);
        rl[t] = t >= vt ? -1 : int32_t(pos + t);
    }
    m_.stream_.upload(*ptab_host_, uint64_t(slot) * kTabBytes, ptab_, uint64_t(slot) * kTabBytes, kTabBytes);
}

// A Direct dispatch also makes each engine walk the static part of the Synapse recipe launched last: after a recipe
// without MME nodes (the embedding) the MME engines fault on it (PMMU page fault at 0x59f8xxxx, device reset). The
// anchor is a tiny recipe with one MME and one TPC node, launched before a chain of prefill plans starts.
void InkDirect::anchor() {
    if (!anchor_) {
        GraphBuilder g(m_.dev_);
        const auto a = g.input("a", DType::BF16, {16, 128});
        const auto b = g.input("b", DType::BF16, {128, 128});
        const auto c = g.input("c", DType::BF16, {16, 128});
        g.gemm(a, b, c, false, true);
        const auto h = g.input("h", DType::F32, {1, 1, 128});
        const auto m = g.input("m", DType::F32, {1, 4, 128});
        const auto cat = g.input("cat", DType::BF16, {1, 1, 256});
        struct { float eps; int32_t m0; } prm{1e-6f, 0};
        g.op("ink_mg_gaudi2", {h, m}, {cat}, &prm, sizeof(prm));
        anchor_ = g.compile("ink_direct_anchor");
        anchor_buf_ = DeviceBuffer(m_.dev_, 1 << 16);
        HostBuffer hb(m_.dev_, 1 << 16);
        std::memset(hb.data(), 0, 1 << 16);
        m_.stream_.upload(hb, 0, anchor_buf_, 0, 1 << 16);
        m_.stream_.synchronize();  // (before the host buffer goes away: the copy is asynchronous)
    }
    const Binding bd[] = {{"a", &anchor_buf_, 0}, {"b", &anchor_buf_, 4096}, {"c", &anchor_buf_, 40960}, {"h", &anchor_buf_, 45056},
                          {"m", &anchor_buf_, 49152}, {"cat", &anchor_buf_, 53248}};
    anchor_->enqueue(m_.stream_, bd);
    m_.stream_.synchronize();
}

void InkDirect::init_bf16_cache(uint64_t kv_need_bytes) {
    if (bf16_cache_init_ || !rt_) return;
    const auto& c = m_.cfg_;
    std::vector<uint32_t> looped, non_looped;
    for (uint32_t L = 0; L < c.layers; ++L) {
        if (L >= c.loop_start && L < c.loop_end) looped.push_back(L);
        else non_looped.push_back(L);
    }
    std::vector<uint32_t> order = looped;
    order.insert(order.end(), non_looped.begin(), non_looped.end());
    const uint32_t n_nat = uint32_t(order.size());
    if (n_nat == 0) {
        bf16_cache_init_ = true;
        bf16_stage_layer_ = -1;
        bf16_stage_full_ = false;
        bf16_stage_valid_.fill(0);
        return;
    }

    const uint64_t E = c.experts, H = c.hidden, F = c.moe_inter;
    bf16_w13_bytes_ = E * H * 2 * F * 2;
    bf16_w2_bytes_ = E * F * H * 2;
    const uint64_t w13_stride = al256(bf16_w13_bytes_);
    const uint64_t w2_stride = al256(bf16_w2_bytes_);
    const uint64_t layer_bytes = w13_stride + w2_stride;

    const char* env = std::getenv("G2E_INK_BF16_CACHE");
    const std::string mode = (env && *env) ? std::string(env) : std::string("auto");
    uint32_t n_cache = 0;
    if (mode == "all") {
        n_cache = n_nat;
    } else if (mode == "loop") {
        n_cache = uint32_t(looped.size());
    } else if (mode == "none") {
        n_cache = 0;
    } else if (mode == "auto") {
        // Reserve the server's 4 GiB margin plus any unallocated portion of kv_need_bytes.
        constexpr uint64_t kMarginBytes = 4ull << 30;
        const uint64_t def_rows = std::min<uint64_t>(uint64_t(m_.max_slots_) * m_.max_ctx_, 1048576ull);
        const uint64_t kv_need = kv_need_bytes ? kv_need_bytes : m_.kv_bytes_for(uint32_t(def_rows));
        const uint64_t kv_held = m_.kv_total();
        const uint64_t kv_extra = kv_need > kv_held ? (kv_need - kv_held) : 0;
        const uint64_t reserve = kMarginBytes + kv_extra;
        const uint64_t free_now = m_.dev_.info().free_bytes;
        const uint64_t avail = free_now > reserve ? (free_now - reserve) : 0;
        if (avail >= uint64_t(n_nat) * layer_bytes) {
            n_cache = n_nat;
        } else if (avail >= 2 * layer_bytes) {
            // Keep one layer_bytes for the shared staging buffer; fill the rest looped-first.
            n_cache = std::min<uint32_t>(n_nat - 1, uint32_t((avail - layer_bytes) / layer_bytes));
        } else {
            n_cache = 0;
        }
    } else {
        throw std::invalid_argument("inkling2: invalid G2E_INK_BF16_CACHE='" + mode + "' (expected auto|all|loop|none)");
    }

    const bool need_staging = (n_cache < n_nat);
    bf16_layers_.assign(c.layers, Bf16LayerAddrs{});
    uint32_t cached_nv = 0, cached_mx = 0;
    if (n_cache > 0) {
        bf16_cache_ = DeviceBuffer(m_.dev_, uint64_t(n_cache) * layer_bytes);
        for (uint32_t i = 0; i < n_cache; ++i) {
            const uint32_t L = order[i];
            const uint64_t base = bf16_cache_.address() + uint64_t(i) * layer_bytes;
            bf16_layers_[L] = {base, base + w13_stride, true};
            if (c.is_nvfp4(L)) ++cached_nv;
            else ++cached_mx;
        }
    }
    if (need_staging) {
        bf16_staging_ = DeviceBuffer(m_.dev_, layer_bytes);
        bf16_staging_w13_ = bf16_staging_.address();
        bf16_staging_w2_ = bf16_staging_w13_ + w13_stride;
    }

    double deq_ms = 0.0;
    if (n_cache > 0) {
        const char* td = std::getenv("G2E_TPC_OBJ_DIR");
        const std::string tpc = (td && *td) ? td : std::string(G2E_TPC_OBJ_DIR);
        if (cached_nv > 0 && !isa_.count("ink_nvf4_deq_bf16_gaudi2"))
            isa_["ink_nvf4_deq_bf16_gaudi2"] = rt_->load_tpc_elf(tpc + "/ink_nvf4_deq_bf16_gaudi2.o");
        if (cached_mx > 0 && !isa_.count("ink_mxfp8_deq_bf16_gaudi2"))
            isa_["ink_mxfp8_deq_bf16_gaudi2"] = rt_->load_tpc_elf(tpc + "/ink_mxfp8_deq_bf16_gaudi2.o");
        if (!p_anchored_) {
            // Do not set p_anchored_ here: the next queue_p() runs after embed_recipe() and still needs anchor + pzero_pw_.
            anchor();
            sram_const_valid_ = false;
        }
        const uint64_t arena = m_.w_.arena().address();
        const uint32_t use_mask = 0;
        std::unique_ptr<DirectPlan> deq[2];  // 0: Nvfp4, 1: Mxfp8

        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n_cache; ++i) {
            const uint32_t L = order[i];
            const bool is_nv = c.is_nvfp4(L);
            const int ki = is_nv ? 0 : 1;
            const uint64_t a_w13q = arena + m_.w_.at(Lp(L, "w13.wq")).offset;
            const uint64_t a_w13b = arena + m_.w_.at(Lp(L, "w13.bs")).offset;
            const uint64_t a_w2q  = arena + m_.w_.at(Lp(L, "w2.wq")).offset;
            const uint64_t a_w2b  = arena + m_.w_.at(Lp(L, "w2.bs")).offset;
            if (!deq[ki]) {
                const uint64_t isa_deq = isa_.at(is_nv ? "ink_nvf4_deq_bf16_gaudi2" : "ink_mxfp8_deq_bf16_gaudi2");
                const uint32_t bs_k = c.expert_block_size(L);
                const uint32_t w13_k = uint32_t(is_nv ? H / 2 : H);
                const uint32_t w2_k  = uint32_t(is_nv ? F / 2 : F);
                auto dp = std::make_unique<DirectPlan>(*rt_);
                const DirectTpcTensor t13[] = {
                    tn(a_w13q, DType::U8, {uint32_t(2 * F), w13_k, uint32_t(E)}),
                    tn(a_w13b, DType::F8, {uint32_t(2 * F), uint32_t(H / bs_k), uint32_t(E)}),
                    tn(a_w13b, DType::I32, {uint32_t(E)}),  // unused when use_mask == 0
                    tn(bf16_layers_[L].w13, DType::BF16, {uint32_t(2 * F), uint32_t(H), uint32_t(E)})};
                const int tags13[] = {0, 1, -1, 2};
                dp->add_tpc_grid(isa_deq, t13, std::span<const uint32_t>(&use_mask, 1),
                                 {uint32_t((2 * F / 256) * E), 1, 1, 1, 1}, 0, 24, -1, 0, 0, 0xf, tags13);
                const DirectTpcTensor t2[] = {
                    tn(a_w2q, DType::U8, {uint32_t(H), w2_k, uint32_t(E)}),
                    tn(a_w2b, DType::F8, {uint32_t(H), uint32_t(F / bs_k), uint32_t(E)}),
                    tn(a_w2b, DType::I32, {uint32_t(E)}),  // unused when use_mask == 0
                    tn(bf16_layers_[L].w2, DType::BF16, {uint32_t(H), uint32_t(F), uint32_t(E)})};
                const int tags2[] = {3, 4, -1, 5};
                dp->add_tpc_grid(isa_deq, t2, std::span<const uint32_t>(&use_mask, 1),
                                 {uint32_t((H / 256) * E), 1, 1, 1, 1}, 0, 24, 0, 24, -1, 0xf, tags2);
                dp->prepare();
                deq[ki] = std::move(dp);
            } else {
                deq[ki]->patch_addr(0, a_w13q);
                deq[ki]->patch_addr(1, a_w13b);
                deq[ki]->patch_addr(2, bf16_layers_[L].w13);
                deq[ki]->patch_addr(3, a_w2q);
                deq[ki]->patch_addr(4, a_w2b);
                deq[ki]->patch_addr(5, bf16_layers_[L].w2);
                DirectPlan* pp = deq[ki].get();
                rt_->flush_patches(std::span<DirectPlan* const>(&pp, 1));
            }
            deq[ki]->run();
        }
        deq_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    bf16_cache_init_ = true;
    bf16_stage_layer_ = -1;
    bf16_stage_full_ = false;
    bf16_stage_valid_.fill(0);
    const uint64_t free_after = m_.dev_.info().free_bytes;
    if (need_staging) {
        std::fprintf(stderr, "[ink-bf16-cache] mode=%s: cached %u/%u native layers (%u NVFP4, %u MXFP8; %.2f GiB) + %.2f GiB staging in %.1f ms (free %.2f GiB)\n",
                     mode.c_str(), n_cache, n_nat, cached_nv, cached_mx, double(bf16_cache_.size()) / double(1ull << 30),
                     double(bf16_staging_.size()) / double(1ull << 30), deq_ms, double(free_after) / double(1ull << 30));
    } else {
        std::fprintf(stderr, "[ink-bf16-cache] mode=%s: cached %u/%u native layers (%u NVFP4, %u MXFP8; %.2f GiB), no staging, in %.1f ms (free %.2f GiB)\n",
                     mode.c_str(), n_cache, n_nat, cached_nv, cached_mx, double(bf16_cache_.size()) / double(1ull << 30),
                     deq_ms, double(free_after) / double(1ull << 30));
    }
}

void InkDirect::download_bf16_layer(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out) const {
    if (layer >= bf16_layers_.size() || !bf16_layers_[layer].cached)
        throw std::invalid_argument("inkling2: layer L" + std::to_string(layer) + " is not in the static BF16 cache");
    const uint64_t base = bf16_cache_.address();
    const uint64_t o13 = bf16_layers_[layer].w13 - base;
    const uint64_t o2 = bf16_layers_[layer].w2 - base;
    w13_out.resize(bf16_w13_bytes_ / 2);
    w2_out.resize(bf16_w2_bytes_ / 2);
    auto dl = [&](uint64_t off, uint64_t bytes, uint16_t* dst) {
        const uint64_t chunk = m_.staging_.size();
        for (uint64_t done = 0; done < bytes; done += chunk) {
            const uint64_t n = std::min(chunk, bytes - done);
            m_.stream_.download(bf16_cache_, off + done, m_.staging_, 0, n);
            m_.stream_.synchronize();
            std::memcpy(reinterpret_cast<uint8_t*>(dst) + done, m_.staging_.data(), n);
        }
    };
    dl(o13, bf16_w13_bytes_, w13_out.data());
    dl(o2, bf16_w2_bytes_, w2_out.data());
}

void InkDirect::download_bf16_staged(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out) {
    if (layer >= m_.cfg_.layers)
        throw std::invalid_argument("inkling2: layer L" + std::to_string(layer) + " out of range");
    if (bf16_layer(layer).cached) {
        download_bf16_layer(layer, w13_out, w2_out);
        return;
    }
    if (!bf16_staging_.address())
        throw std::invalid_argument("inkling2: BF16 staging buffer is not allocated");
    if (!p_anchored_) {
        anchor();
        sram_const_valid_ = false;
    }
    ensure_bf16_staged(layer, 0, nullptr);
    const uint64_t base = bf16_staging_.address();
    const uint64_t o13 = bf16_staging_w13_ - base;
    const uint64_t o2 = bf16_staging_w2_ - base;
    w13_out.resize(bf16_w13_bytes_ / 2);
    w2_out.resize(bf16_w2_bytes_ / 2);
    auto dl = [&](uint64_t off, uint64_t bytes, uint16_t* dst) {
        const uint64_t chunk = m_.staging_.size();
        for (uint64_t done = 0; done < bytes; done += chunk) {
            const uint64_t n = std::min(chunk, bytes - done);
            m_.stream_.download(bf16_staging_, off + done, m_.staging_, 0, n);
            m_.stream_.synchronize();
            std::memcpy(reinterpret_cast<uint8_t*>(dst) + done, m_.staging_.data(), n);
        }
    };
    dl(o13, bf16_w13_bytes_, w13_out.data());
    dl(o2, bf16_w2_bytes_, w2_out.data());
}

bool InkDirect::handles_all_p(uint32_t T, uint32_t pos) const {
    for (uint32_t i = 0; i < m_.apps_.size(); ++i)
        if (!handles_p(i, T, pos)) return false;
    return true;
}
bool InkDirect::handles_all_p(uint32_t T) const { return handles_all_p(T, m_.pos_); }

// the plans (A: attention + router + rplan_tab, B: lossless MoE + post-MLP conv/res) of an application for chunk slot
// `slot` (built on first use; Plan B is shared across all far buckets / window modes to save hbm_shared and build time).
InkDirect::PPlan& InkDirect::get_p(uint32_t app, int slot, int hcur, uint64_t out_addr, int cs_in, int cs_out, uint32_t pos) {
    const bool full = m_.cfg_.full[m_.apps_[app].layer];
    const uint32_t fr = full ? far_rows_at(pos, m_.cfg_.window) : 0;
    const uint32_t fb = full && fr ? m_.far_bucket(fr) : 0;
    const bool finline = full && pfar_flash_ && psram_pw_on_ && fr > 0 && fr <= 1024u;
    const bool fsplit = full && pfar_flash_ && !finline && fr > 0 &&
                        (!pfar_ok_ || fb > pfar_cap_ || !mme_.count("p_fqk_" + std::to_string(fb) + "_m512"));
    // Constant sfk sentinel for ALL fsplit (34K .. 1M): Plan A1 + Plan A2 never rebuild; patched pfar_blk_ runs the tiles!
    const uint32_t sfk = full ? (fsplit ? 0x00fffffeu : (finline ? 1024u : fb)) : 0;
    // (the taps only change the plans of applications 4 / 22 / 40)
    const bool tap_app = m_.aux_on_ && (app == 4 || app == 22 || app == 40);
    // (sliding layers at a position that is a multiple of the chunk size take the windowed softmax)
    const bool win = !full && pos % kT == 0;
    const int sslot = int(m_.active_slot_) << 4;
    const int bbits = slot | (cs_in << 2) | (cs_out << 3) | sslot;
    const auto bkey = std::make_tuple(app, bbits, out_addr, uint32_t(tap_app) << 31);
    const auto key = std::make_tuple(app, slot | (hcur << 1) | (cs_in << 2) | (cs_out << 3) | sslot, out_addr,
                                     sfk | (uint32_t(win) << 30) | (uint32_t(tap_app) << 31));
    auto evict_and_retry = [&](const auto& fn) {
        try {
            fn();
            return;
        } catch (const std::runtime_error& e) {
            if (std::string(e.what()).find("plan alloc failed") == std::string::npos) throw;
        }
        flush();
        orders_.clear();
        auto try_fn = [&] {
            try { fn(); return true; } catch (const std::runtime_error& e2) {
                if (std::string(e2.what()).find("plan alloc failed") == std::string::npos) throw;
                return false;
            }
        };
        // 1: drop prefill plans belonging to other sequence slots (100% idle; keeps active slot & all decode plans intact)
        {
            size_t dropped = 0;
            for (auto j = plans_p_.begin(); j != plans_p_.end();) {
                if ((std::get<1>(j->first) >> 4) != int(m_.active_slot_)) { j = plans_p_.erase(j); ++dropped; }
                else ++j;
            }
            for (auto j = plans_pb_.begin(); j != plans_pb_.end();) {
                if ((std::get<1>(j->first) >> 4) != int(m_.active_slot_)) { j = plans_pb_.erase(j); ++dropped; }
                else ++j;
            }
            if (dropped && try_fn()) return;
        }
        // 2: drop stale prefill plans from old far buckets / tap modes (>128 accesses ago)
        {
            size_t dropped = 0;
            for (auto j = plans_p_.begin(); j != plans_p_.end();) {
                if (j->second.used + 128 < use_clock_) { j = plans_p_.erase(j); ++dropped; }
                else ++j;
            }
            if (dropped && try_fn()) return;
        }
        // 3: drop verification plans
        if (!plans_v_.empty() || !rollback_.empty() || !v_embed_plans_.empty() ||
            !v_head_plans_.empty() || !v_commit_plans_.empty()) {
            plans_v_.clear();
            rollback_.clear();
            v_embed_plans_.clear();
            v_head_plans_.clear();
            v_commit_plans_.clear();
            ++evict_gen_;
            if (try_fn()) return;
        }
        // 4: drop B=1 decode plans
        {
            bool any_d = !plans_.empty() || !span_plans_.empty() || d_embed_plan_[0] || d_embed_plan_[1] ||
                         !d_head_plans_.empty() || bool(sram_const_copy_plan_) || !dfar_pipe_pool_.empty();
            for (auto& pool : dfar_pool_)
                if (!pool.empty()) { pool.clear(); any_d = true; }
            if (any_d) {
                plans_.clear();
                span_plans_.clear();
                d_embed_plan_[0].reset();
                d_embed_plan_[1].reset();
                d_head_plans_.clear();
                sram_const_copy_plan_.reset();
                dfar_pipe_pool_.clear();
                ++evict_gen_;
                if (try_fn()) return;
            }
        }
        // 5: last resort — drop batched decode plans
        {
            bool any_b = !b_kv_commit_pool_.empty();
            b_kv_commit_pool_.clear();
            for (int bi = 0; bi < 3; ++bi) {
                if (!b_app_plans_[bi].empty()) any_b = true;
                b_app_plans_[bi].clear();
                b_last_slots_[bi].clear();
                b_last_opt_mask_[bi] = ~uint64_t(0);
                b_head_last_B_[bi] = 0;
                b_head_last_skip_[bi] = 0;
            }
            if (any_b) { ++evict_gen_; fn(); return; }
        }
        throw std::runtime_error("direct_mme: plan alloc failed (no plans left to evict)");
    };

    const uint64_t now_clk = ++use_clock_;
    auto bit = plans_pb_.find(bkey);
    if (bit == plans_pb_.end()) {
        PPlanB pb;
        evict_and_retry([&] {
            pb.meta_b.clear();
            pb.plan_b.reset();
            pb.plan_b = build_pb(app, slot, out_addr, cs_in, cs_out, 0, &pb.meta_b);
        });
        const bool wc = std::getenv("G2E_INK_WAIT_CHECK") != nullptr;
        if (const size_t bad = check_waits(app, pb.meta_b, wc))
            std::fprintf(stderr, "[waits] app %u slot %d planB: %zu wait pairs are not ordered\n", app, slot, bad);
        bit = plans_pb_.emplace(bkey, std::move(pb)).first;
    }
    bit->second.used = now_clk;

    auto it = plans_p_.find(key);
    if (it == plans_p_.end()) {
        PPlan pl;
        evict_and_retry([&] {
            pl.meta_a.clear();
            pl.plan_a.reset();
            pl.plan_a2.reset();
            pl.plan_a = build_pa(app, slot, hcur, cs_in, cs_out, pos, 0, &pl.meta_a);
            if (fsplit) pl.plan_a2 = build_pa2(app, slot, hcur, cs_in, cs_out);
        });
        pl.plan_b = bit->second.plan_b.get();
        pl.meta_b = &bit->second.meta_b;
        if (const char* pd = std::getenv("G2E_INK_PLAN_DUMP"); pd && uint32_t(std::atoi(pd)) == app) {  // (debug) the stages' SOBs
            auto dump_m = [&](const char* tag, const std::vector<PStage>& ms) {
                size_t k = 0;
                for (const auto& st : ms) {
                    std::string sg;
                    for (size_t j = 0; j < st.parts.size(); ++j) sg += " " + std::to_string(st.parts[j].second) + "x" + std::to_string(st.cnt[j]);
                    std::fprintf(stderr, "[plan%s] app %u stage %2zu %s %-28s wait %2d >= %-3u sig%s\n", tag, app, k++, st.eng ? "TPC" : "MME",
                                 st.name.c_str(), st.wait, st.wtgt, sg.c_str());
                }
            };
            dump_m("A", pl.meta_a);
            dump_m("B", *pl.meta_b);
        }
        const bool wc = std::getenv("G2E_INK_WAIT_CHECK") != nullptr;
        if (const size_t bad = check_waits(app, pl.meta_a, wc))
            std::fprintf(stderr, "[waits] app %u slot %d planA: %zu wait pairs are not ordered\n", app, slot, bad);
        it = plans_p_.emplace(key, std::move(pl)).first;
    }
    it->second.used = now_clk;
    return it->second;
}

// Stage an uncached native layer (NVFP4 or MXFP8) into bf16_staging_ via a lane-0 DirectPlan (outside run_waves).
// - mask_addr == 0 (run_pair): full 128-expert dequant once before (B0[i] || A1[i]); reused by B1[i] and by any
//   later application of the same layer until another uncached layer overwrites staging.
// - mask_addr != 0 (queue_p / WAVE_SEQ): masked by device cnt[128] so short prompts skip inactive experts; tracks
//   per-expert validity so repeat applications of the same layer only dequant if a new expert becomes active.
double InkDirect::ensure_bf16_staged(uint32_t layer, uint64_t mask_addr, const int32_t* cnt) {
    const auto& c = m_.cfg_;
    if (bf16_layer(layer).cached) return 0.0;
    if (!bf16_staging_w13_ || !bf16_staging_w2_)
        throw std::logic_error("inkling2 direct prefill: BF16 staging buffer not initialized for uncached layer L" +
                               std::to_string(layer));

    const uint32_t E = c.experts, H = c.hidden, F = c.moe_inter;
    const bool masked = (mask_addr != 0 && cnt != nullptr);
    if (bf16_stage_layer_ == int32_t(layer)) {
        if (bf16_stage_full_) return 0.0;
        if (masked) {
            bool all_ok = true;
            for (uint32_t e = 0; e < E; ++e)
                if (cnt[e] && !bf16_stage_valid_[e]) { all_ok = false; break; }
            if (all_ok) return 0.0;
        }
    }

    const bool is_nv = c.is_nvfp4(layer);
    const int ki = is_nv ? 0 : 1;
    const uint64_t arena = m_.w_.arena().address();
    const uint64_t a_w13q = arena + m_.w_.at(Lp(layer, "w13.wq")).offset;
    const uint64_t a_w13b = arena + m_.w_.at(Lp(layer, "w13.bs")).offset;
    const uint64_t a_w2q  = arena + m_.w_.at(Lp(layer, "w2.wq")).offset;
    const uint64_t a_w2b  = arena + m_.w_.at(Lp(layer, "w2.bs")).offset;
    const uint32_t um = masked ? 1u : 0u;
    const uint64_t maddr = masked ? mask_addr : (ps_[0].cnt ? ps_[0].cnt : a_w13b);

    if (!pbf16_deq_plan_[ki]) {
        const char* deq_name = is_nv ? "ink_nvf4_deq_bf16_gaudi2" : "ink_mxfp8_deq_bf16_gaudi2";
        if (!isa_.count(deq_name)) {
            const char* td = std::getenv("G2E_TPC_OBJ_DIR");
            const std::string tpc = (td && *td) ? td : std::string(G2E_TPC_OBJ_DIR);
            isa_[deq_name] = rt_->load_tpc_elf(tpc + "/" + deq_name + ".o");
        }
        auto pp = std::make_unique<DirectPlan>(*rt_);
        const uint64_t isa_deq = isa_.at(deq_name);
        const uint32_t bs_k = c.expert_block_size(layer);
        const uint32_t w13_k = is_nv ? H / 2 : H;
        const uint32_t w2_k  = is_nv ? F / 2 : F;
        const int stags[] = {0};
        const DirectTpcTensor t13[] = {
            tn(a_w13q, DType::U8, {2 * F, w13_k, E}),
            tn(a_w13b, DType::F8, {2 * F, H / bs_k, E}),
            tn(maddr, DType::I32, {E}),
            tn(bf16_staging_w13_, DType::BF16, {2 * F, H, E})};
        const int tags13[] = {0, 1, 2, -1};
        pp->add_tpc_grid(isa_deq, t13, std::span<const uint32_t>(&um, 1), {(2 * F / 256) * E, 1, 1, 1, 1},
                         0, 24, -1, 0, 0, 0xf, tags13, 0, stags);
        const DirectTpcTensor t2[] = {
            tn(a_w2q, DType::U8, {H, w2_k, E}),
            tn(a_w2b, DType::F8, {H, F / bs_k, E}),
            tn(maddr, DType::I32, {E}),
            tn(bf16_staging_w2_, DType::BF16, {H, F, E})};
        const int tags2[] = {3, 4, 2, -1};
        // No barrier between w13 and w2: outputs are disjoint and each core executes w13 then w2 in FIFO order.
        pp->add_tpc_grid(isa_deq, t2, std::span<const uint32_t>(&um, 1), {(H / 256) * E, 1, 1, 1, 1},
                         0, 24, -1, 0, -1, 0xf, tags2, 0, stags);
        pp->prepare();
        pbf16_deq_plan_[ki] = std::move(pp);
        pbf16_deq_last_L_[ki] = layer;
        pbf16_deq_last_mask_[ki] = maddr;
        pbf16_deq_last_um_[ki] = um;
    } else {
        bool dirty = false;
        if (layer != pbf16_deq_last_L_[ki]) {
            pbf16_deq_plan_[ki]->patch_addr(0, a_w13q);
            pbf16_deq_plan_[ki]->patch_addr(1, a_w13b);
            pbf16_deq_plan_[ki]->patch_addr(3, a_w2q);
            pbf16_deq_plan_[ki]->patch_addr(4, a_w2b);
            pbf16_deq_last_L_[ki] = layer;
            dirty = true;
        }
        if (masked && maddr != pbf16_deq_last_mask_[ki]) {
            pbf16_deq_plan_[ki]->patch_addr(2, maddr);
            pbf16_deq_last_mask_[ki] = maddr;
            dirty = true;
        }
        if (um != pbf16_deq_last_um_[ki]) {
            pbf16_deq_plan_[ki]->patch_srf(0, um);
            pbf16_deq_last_um_[ki] = um;
            dirty = true;
        }
        if (dirty) {
            DirectPlan* pp = pbf16_deq_plan_[ki].get();
            rt_->flush_patches(std::span<DirectPlan* const>(&pp, 1));
        }
    }

    const double us = pbf16_deq_plan_[ki]->run();
    if (!masked) {
        bf16_stage_layer_ = int32_t(layer);
        bf16_stage_full_ = true;
        bf16_stage_valid_.fill(1);
    } else {
        if (bf16_stage_layer_ != int32_t(layer)) {
            bf16_stage_layer_ = int32_t(layer);
            bf16_stage_full_ = false;
            bf16_stage_valid_.fill(0);
        }
        bool full = true;
        for (uint32_t e = 0; e < E; ++e) {
            if (cnt[e]) bf16_stage_valid_[e] = 1;
            if (!bf16_stage_valid_[e]) full = false;
        }
        bf16_stage_full_ = full;
    }
    return us;
}

// Host reads exact per-expert slot counts cnt[0..127] (512 B via direct SCAL PDMA) produced by Plan A of `slot`,
// computes exp_base[0..127] matching ink_rplan_dyn_tab_bf16_gaudi2, builds 4-wave diff-encoded dynamic MME ECBs
// for gu and dn (zero empty experts, zero overflow, zero per-layer static MoE code), and uploads in 1 SCAL PDMA.
void InkDirect::patch_b(uint32_t app, int slot, PPlan& pl, int32_t* out_cnt) {
    const auto& c = m_.cfg_;
    const uint32_t L = m_.apps_[app].layer;
    const uint64_t H = c.hidden, F = c.moe_inter;
    const auto bl = bf16_layer(L);
    const uint64_t w13 = bl.cached ? bl.w13 : bf16_staging_w13_;
    const uint64_t w2 = bl.cached ? bl.w2 : bf16_staging_w2_;
    if (!w13 || !w2)
        throw std::logic_error("inkling2 direct prefill: BF16 weights unavailable for layer L" + std::to_string(L));
    const uint64_t e13 = H * 2 * F * 2;
    const uint64_t e2 = F * H * 2;
    const auto& p = ps_[slot];
    auto& dyn = *pdyn_[slot];

    static double s_dl_us = 0, s_cpu_us = 0, s_fl_us = 0;
    static uint64_t s_pb_cnt = 0, s_code_tot = 0, s_rows_tot = 0, s_ecb_tot = 0;
    static const bool pprof = std::getenv("G2E_INK_PAIR_PROF") != nullptr;
    auto now_u = [] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double u0 = pprof ? now_u() : 0.0;

    alignas(16) int32_t cnt[128];
    rt_->download_sync(p.cnt, cnt, sizeof(cnt));
    if (out_cnt) std::memcpy(out_cnt, cnt, sizeof(cnt));
    const double u1 = pprof ? now_u() : 0.0;

    uint32_t exp_base[128];
    uint32_t tot_rows = 0;
    for (uint32_t e = 0; e < 128; ++e) {
        exp_base[e] = tot_rows;
        tot_rows += dyn_alloc_rows(uint32_t(cnt[e]));
    }
    if (tot_rows > kFlatRows)
        throw std::runtime_error("inkling2 direct prefill: dynamic MoE flat rows overflow (" + std::to_string(tot_rows) + ")");

    const uint32_t fence_id = slot == 0 ? 3u : 2u;
    const uint32_t sob_off = slot == 0 ? 0u : 64u;
    const uint32_t sob_gu_abs = rt_->sob_base() + sob_off + uint32_t(kPbSobGu);
    const uint32_t sob_dn_abs = rt_->sob_base() + sob_off + uint32_t(kPbSobDn);

    dyn.reset();
    uint64_t opt_mask = ~uint64_t(0);
    uint32_t cum_gu = 0, cum_dn = 0, active_waves = 0;
    bool first_gu_done = false;
    auto& gu_jobs = pgu_jobs_;
    auto& dn_jobs = pdn_jobs_;

    for (uint32_t w = 0; w < 4; ++w) {
        for (int k = 0; k < 7; ++k) { pgu_b_[k].clear(); pdn_b_[k].clear(); }
        for (uint32_t e = w * 32; e < (w + 1) * 32; ++e) {
            const uint32_t ce = uint32_t(cnt[e]);
            if (!ce) continue;
            const uint64_t wb13 = w13 + uint64_t(e) * e13;
            const uint64_t wb2 = w2 + uint64_t(e) * e2;
            dyn_emit_tiles(ce, exp_base[e], [&](int tid, uint32_t r0) {
                pgu_b_[tid].push_back({pbgt_t_[tid], p.xq + uint64_t(r0) * H * 2, wb13, p.gu + uint64_t(r0) * 2 * F * 4, 0});
                pdn_b_[tid].push_back({pbdt_t_[tid], p.aq + uint64_t(r0) * F * 2, wb2, p.gu + uint64_t(r0) * H * 4, 0});
            });
        }
        gu_jobs.clear();
        dn_jobs.clear();
        for (int k = 0; k < 7; ++k) {
            gu_jobs.insert(gu_jobs.end(), pgu_b_[k].begin(), pgu_b_[k].end());
            dn_jobs.insert(dn_jobs.end(), pdn_b_[k].begin(), pdn_b_[k].end());
        }
        if (gu_jobs.empty()) {
            opt_mask &= ~(uint64_t(1) << w);
            dyn.set_stage(w, {}, false, 0, fence_id, sob_gu_abs);
            dyn.set_stage(4 + w, {}, false, 0, fence_id, sob_dn_abs);
        } else {
            const bool gu_wait = !first_gu_done;
            first_gu_done = true;
            cum_gu += dyn.set_stage(w, gu_jobs, gu_wait, gu_wait ? 24u : 0u, fence_id, sob_gu_abs);
            pl.plan_b->set_dyn_wait(int(w), cum_gu);
            pdyn_last_[slot].tgt[w] = cum_gu;
            ++active_waves;
            cum_dn += dyn.set_stage(4 + w, dn_jobs, true, 24u * active_waves, fence_id, sob_dn_abs);
            if (w < 3) pdyn_last_[slot].tgt[5 + w] = cum_dn;
        }
    }
    pl.plan_b->set_opt_mask(opt_mask);
    pl.plan_b->set_dyn_wait(4, cum_dn);
    pdyn_last_[slot].tgt[4] = cum_dn;
    pdyn_last_[slot].opt_mask = opt_mask;
    const double u2 = pprof ? now_u() : 0.0;
    dyn.upload();

    if (pprof) {
        const double u3 = now_u();
        s_dl_us += u1 - u0; s_cpu_us += u2 - u1; s_fl_us += u3 - u2;
        s_code_tot += dyn.last_code_bytes();
        s_ecb_tot += dyn.last_ecb_count();
        s_rows_tot += tot_rows;
        if (++s_pb_cnt % 96 == 0) {
            std::fprintf(stderr, "[patch-prof] 96 calls: dl=%.1f ms (%.0f us) cpu=%.1f ms (%.0f us) ul=%.1f ms (%.0f us) | avg %.1f KiB, %.1f ECBs, %.0f rows\n",
                         s_dl_us * 1e-3, s_dl_us / 96.0, s_cpu_us * 1e-3, s_cpu_us / 96.0, s_fl_us * 1e-3, s_fl_us / 96.0,
                         double(s_code_tot) / (96.0 * 1024.0), double(s_ecb_tot) / 96.0, double(s_rows_tot) / 96.0);
            s_dl_us = s_cpu_us = s_fl_us = 0;
            s_code_tot = s_ecb_tot = s_rows_tot = 0;
        }
    }
}

bool InkDirect::queue_p(uint32_t app, int hcur, const DeviceBuffer* out_buf, uint64_t out_off, int cs_in, int cs_out) {
    const bool was_empty = queue_.empty();
    if (!was_empty) flush();
    if (!bf16_cache_init_) init_bf16_cache(0);
    if (!p_anchored_) {
        anchor();
        p_anchored_ = true;
        if (psram_pw_on_ && pzero_pw_) pzero_pw_->run();
    }
    auto& pl = get_p(app, 0, hcur, out_buf->address() + out_off, cs_in, cs_out, m_.pos_);
    double ta = pl.plan_a->run();
    if (pl.plan_a2) ta += run_pfar_loop(app, 0, m_.pos_, pl);
    int32_t cnt[128];
    patch_b(app, 0, pl, cnt);
    const double td = ensure_bf16_staged(m_.apps_[app].layer, ps_[0].cnt, cnt);
    const double tb = pl.plan_b->run();
    static const bool tm = std::getenv("G2E_INK_DIRECT_TIME") != nullptr;
    if (tm) {
        if (td > 0.0) std::fprintf(stderr, "[ink-direct] prefill app %u: A %.1f + deq %.1f + B %.1f = %.1f us\n", app, ta, td, tb, ta + td + tb);
        else std::fprintf(stderr, "[ink-direct] prefill app %u: A %.1f + B %.1f = %.1f us\n", app, ta, tb, ta + tb);
    }
    return was_empty;
}

namespace {
struct MergePick { float start; int q; size_t k; };
struct MergeResult { float makespan, mme_end, tpc_end; std::vector<MergePick> sched; };

// Shared dependency graph and list-scheduler for merging two plans' stages (Plan B of one chunk || Plan A of the other).
template <class Stage>
struct MergeDag {
    const std::vector<Stage>* m[2];
    int ns_max[2] = {0, 0};
    std::vector<std::vector<size_t>> succ[2];

    MergeDag(const std::vector<Stage>& ma, const std::vector<Stage>& mb) : m{&ma, &mb} {
        for (int q = 0; q < 2; ++q) {
            const size_t n = m[q]->size();
            for (const auto& st : *m[q]) {
                ns_max[q] = std::max(ns_max[q], st.wait + 1);
                for (const auto& pt : st.parts) ns_max[q] = std::max(ns_max[q], pt.second + 1);
            }
            // For each SOB, list (cum_credits_after, stage_idx) in plan order so consumers link to their exact producer.
            std::vector<std::vector<std::pair<uint32_t, size_t>>> prod(size_t(ns_max[q]));
            std::vector<uint32_t> cum(size_t(ns_max[q]), 0);
            for (size_t k = 0; k < n; ++k) {
                const auto& st = (*m[q])[k];
                for (size_t j = 0; j < st.parts.size(); ++j) {
                    const int sb = st.parts[j].second;
                    if (sb >= 0) {
                        cum[size_t(sb)] += st.cnt[j];
                        prod[size_t(sb)].push_back({cum[size_t(sb)], k});
                    }
                }
            }
            succ[q].assign(n, {});
            for (size_t k = 0; k < n; ++k) {
                const auto& st = (*m[q])[k];
                if (st.wait < 0) continue;
                for (const auto& [c_after, pk] : prod[size_t(st.wait)]) {
                    succ[q][pk].push_back(k);
                    if (c_after >= st.wtgt) break;
                }
            }
        }
    }

    MergeResult simulate(float alpha, int tie_q, float ready_bias, float mme_scale = 1.f, float b_bias = 0.f) const {
        constexpr float kInf = 1e30f;
        std::vector<float> cp[2];
        for (int q = 0; q < 2; ++q) {
            const size_t n = m[q]->size();
            cp[q].assign(n, 0.f);
            float eng_cp[2] = {0.f, 0.f};
            for (size_t k = n; k-- > 0;) {
                const auto& st = (*m[q])[k];
                const float sc = st.eng == 0 ? mme_scale : 1.f;
                float dur = 0.f, after = eng_cp[st.eng];
                for (const auto& pt : st.parts) dur += (sc == 1.f) ? pt.first : pt.first * sc;
                for (size_t sk : succ[q][k]) after = std::max(after, cp[q][sk]);
                cp[q][k] = dur + after;
                eng_cp[st.eng] = cp[q][k];
            }
        }
        std::vector<uint32_t> got[2];
        std::vector<std::vector<std::pair<uint32_t, float>>> sig_hist[2];
        std::vector<uint8_t> done[2];
        for (int q = 0; q < 2; ++q) {
            got[q].assign(size_t(ns_max[q]), 0);
            sig_hist[q].assign(size_t(ns_max[q]), {});
            done[q].assign(m[q]->size(), 0);
        }
        float free_at[2] = {0.f, 0.f};
        std::vector<MergePick> sched;
        const size_t total = m[0]->size() + m[1]->size();
        sched.reserve(total);
        while (sched.size() < total) {
            float min_start = kInf;
            int fq = -1;
            size_t fk = 0;
            struct Cand { int q; size_t k; float start; float score; };
            std::vector<Cand> cands;
            for (int qi = 0; qi < 2; ++qi) {
                const int q = tie_q ? (1 - qi) : qi;
                for (int eng = 0; eng < 2; ++eng) {
                    size_t k = 0;
                    while (k < m[q]->size() && (done[q][k] || (*m[q])[k].eng != eng)) ++k;
                    if (k == m[q]->size()) continue;
                    if (fq < 0) { fq = q; fk = k; }
                    const auto& st = (*m[q])[k];
                    float ready = 0.f;
                    if (st.wait >= 0) {
                        if (got[q][size_t(st.wait)] < st.wtgt) continue;
                        for (const auto& [c_after, ft] : sig_hist[q][size_t(st.wait)]) {
                            ready = ft;
                            if (c_after >= st.wtgt) break;
                        }
                    }
                    const float start = std::max(free_at[eng], ready);
                    min_start = std::min(min_start, start);
                    const float sc = start - alpha * cp[q][k] + (q == 1 ? b_bias : 0.f);
                    cands.push_back({q, k, start, sc});
                }
            }
            MergePick best{kInf, -1, 0};
            float best_sc = kInf;
            for (const auto& c : cands) {
                if (c.start > min_start + ready_bias) continue;
                if (c.score < best_sc) { best_sc = c.score; best = MergePick{c.start, c.q, c.k}; }
            }
            if (best.q < 0) best = MergePick{std::max(free_at[0], free_at[1]), fq, fk};
            const auto& st = (*m[best.q])[best.k];
            const float sc = st.eng == 0 ? mme_scale : 1.f;
            float t = best.start;
            for (size_t j = 0; j < st.parts.size(); ++j) {
                t += (sc == 1.f) ? st.parts[j].first : st.parts[j].first * sc;
                const int sb = st.parts[j].second;
                if (sb >= 0) {
                    got[best.q][size_t(sb)] += st.cnt[j];
                    sig_hist[best.q][size_t(sb)].push_back({got[best.q][size_t(sb)], t});
                }
            }
            free_at[st.eng] = t;
            done[best.q][best.k] = 1;
            sched.push_back(best);
        }
        return {std::max(free_at[0], free_at[1]), free_at[0], free_at[1], std::move(sched)};
    }

    static std::vector<uint16_t> to_order(std::vector<MergePick>& sched) {
        std::stable_sort(sched.begin(), sched.end(), [](const MergePick& x, const MergePick& y) { return x.start < y.start; });
        std::vector<uint16_t> ord;
        ord.reserve(sched.size());
        for (const auto& pk : sched) ord.push_back(uint16_t((pk.q ? 0x8000u : 0u) | uint32_t(pk.k)));
        return ord;
    }
};
}  // namespace

// The dispatch order of two plans' stages (e.g. Plan B of one chunk || Plan A of the other): tries several list
// scheduling heuristics (earliest-start, critical-path priority, tie-breaking by either plan) on the two engine types
// using the stage duration estimates and keeps the valid schedule with the smallest simulated makespan.
std::vector<uint16_t> InkDirect::merge_order(const std::vector<PStage>& ma, const std::vector<PStage>& mb) {
    const MergeDag<PStage> dag(ma, mb);
    MergeResult best{1e30f, 0.f, 0.f, {}};
    for (int tq = 0; tq < 2; ++tq)
        for (const auto& [rb, a] : {std::pair{0.f, 0.f}, std::pair{40.f, 0.5f}, std::pair{120.f, 1.0f},
                                    std::pair{300.f, 1.0f}, std::pair{1e6f, 1.0f}, std::pair{1e6f, 2.0f}}) {
            auto r = dag.simulate(a, tq, rb);
            if (r.makespan < best.makespan) best = std::move(r);
        }
    auto order = MergeDag<PStage>::to_order(best.sched);
    static const int dbg = std::getenv("G2E_INK_WAVE_DEBUG") ? std::atoi(std::getenv("G2E_INK_WAVE_DEBUG")) : 0;
    if (dbg) std::fprintf(stderr, "[ink-direct] wave pair: %zu + %zu stages, estimated %.0f us (MME until %.0f, TPC until %.0f)\n",
                          ma.size(), mb.size(), best.makespan, best.mme_end, best.tpc_end);
    if (dbg >= 2) {  // the simulated timeline: start, duration, engine, plan, stage
        float busy[2] = {0.f, 0.f};
        for (const auto& pk : best.sched) {
            const auto& st = (*dag.m[pk.q])[pk.k];
            float d = 0.f;
            for (const auto& pt : st.parts) d += pt.first;
            busy[st.eng] += d;
            std::fprintf(stderr, "[wave-sim] %7.0f +%5.0f %s %c%-2zu %s\n", pk.start, d, st.eng ? "TPC" : "MME", pk.q ? 'b' : 'a', pk.k, st.name.c_str());
        }
        std::fprintf(stderr, "[wave-sim] busy: MME %.0f us, TPC %.0f us of %.0f\n", busy[0], busy[1], best.makespan);
    }
    return order;
}

// Two chunks interleaved in a phase-locked wave:
//   Step 0:   A0[0]                                   -> patch B0[0]
//   i=0..n-1: (B0[i] || A1[i]) -> patch B1[i]         (slot 0 MME-heavy MoE || slot 1 TPC-heavy attention)
//             (B1[i] || A0[i+1]) -> patch B0[i+1]     (slot 1 MME-heavy MoE || slot 0 TPC-heavy attention)
//   Final:    B1[n-1]
// All shared per-application states (KV rings, far caches, conv[0..2]) are written by Plan A before the other chunk's
// Plan A reads them, and conv[3] is written by Plan B before the other chunk's Plan B reads it.
std::vector<std::vector<uint16_t>> InkDirect::merge_order_candidates(const std::vector<PStage>& ma, const std::vector<PStage>& mb) {
    const MergeDag<PStage> dag(ma, mb);
    // Candidate 0 is always the static merge_order baseline; 4 sharp additional candidates (converges within 1 pair
    // even for full layers which have 8 apps/pair) explore prioritizing Plan B (MoE) vs Plan A (attention).
    std::vector<std::vector<uint16_t>> out;
    out.push_back(merge_order(ma, mb));
    struct Param { float rb, a; int tq; float ms, bb; };
    static const Param kParams[] = {
        {300.f, 1.0f, 1, 1.0f, +300.f},
        {300.f, 1.0f, 1, 1.0f, +150.f},
        {120.f, 1.0f, 1, 1.0f, -150.f},
        {300.f, 1.0f, 1, 1.0f, -300.f},
        {120.f, 1.0f, 0, 1.0f, +200.f},
        {120.f, 1.0f, 1, 0.70f, +100.f},
        {0.f, 0.f, 1, 1.0f, 0.f},
    };
    for (const auto& pr : kParams) {
        auto r = dag.simulate(pr.a, pr.tq, pr.rb, pr.ms, pr.bb);
        auto ord = MergeDag<PStage>::to_order(r.sched);
        if (std::find(out.begin(), out.end(), ord) == out.end()) {
            out.push_back(std::move(ord));
            if (out.size() >= 5) break;
        }
    }
    return out;
}

double InkDirect::run_pair(uint32_t pos, std::span<const std::pair<int, uint64_t>> io0, std::span<const std::pair<int, uint64_t>> io1) {
    if (!bf16_cache_init_) init_bf16_cache(0);
    flush();
    anchor();
    if (psram_pw_on_ && pzero_pw_) pzero_pw_->run();
    const uint32_t n = uint32_t(m_.apps_.size());
    std::vector<PPlan*> pa(n), pb(n);
    for (uint32_t i = 0; i < n; ++i) pa[i] = &get_p(i, 0, io0[i].first, io0[i].second, 0, 0, pos);
    for (uint32_t i = 0; i < n; ++i) pb[i] = &get_p(i, 1, io1[i].first, io1[i].second, 0, 0, pos + kT);
    static const bool seq = std::getenv("G2E_INK_WAVE_SEQ") != nullptr;  // (debug: the two chunks one after the other)
    const auto t0 = std::chrono::steady_clock::now();
    static const bool pprof = std::getenv("G2E_INK_PAIR_PROF") != nullptr;
    double tot_dev = 0, tot_patch = 0, tot_deq = 0, by_kind[4] = {};
    int cnt_kind[4] = {};
    auto now_u = [] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    auto run_far_chain = [&](uint32_t a_app, int sl, uint32_t cpos, PPlan& pl, bool add_kind) {
        if (!pl.plan_a2) return;
        const double df = run_pfar_loop(a_app, sl, cpos, pl);
        tot_dev += df;
        if (pprof && add_kind) {
            const auto& ap = m_.apps_[a_app];
            const int kd = (m_.cfg_.full[ap.layer] ? 2 : 0) | (ap.pass > 0 ? 1 : 0);
            by_kind[kd] += df;
        }
    };
    if (seq) {
        int32_t cnt[128];
        for (uint32_t i = 0; i < n; ++i) {
            pa[i]->plan_a->run();
            run_far_chain(i, 0, pos, *pa[i], false);
            patch_b(i, 0, *pa[i], cnt);
            tot_deq += ensure_bf16_staged(m_.apps_[i].layer, ps_[0].cnt, cnt);
            pa[i]->plan_b->run();
        }
        for (uint32_t i = 0; i < n; ++i) {
            pb[i]->plan_a->run();
            run_far_chain(i, 1, pos + kT, *pb[i], false);
            patch_b(i, 1, *pb[i], cnt);
            tot_deq += ensure_bf16_staged(m_.apps_[i].layer, ps_[1].cnt, cnt);
            pb[i]->plan_b->run();
        }
    } else {
        auto run_one = [&](DirectPlan* a) {
            DirectWave w{a, nullptr, {}};
            tot_dev += rt_->run_waves(std::span<const DirectWave>(&w, 1));
        };
        static const bool autotune = [] { const char* e = std::getenv("G2E_INK_WAVE_AUTOTUNE"); return e && *e == '1'; }();
        auto topo_hash = [](const std::vector<PStage>& ma, const std::vector<PStage>& mb) -> uint64_t {
            uint64_t h = 1469598103934665603ull;
            auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
            for (const auto* mp : {&ma, &mb}) {
                mix(mp->size());
                for (const auto& st : *mp) {
                    mix((uint64_t(st.eng) << 32) | uint32_t(st.wait + 1));
                    mix((uint64_t(st.wtgt) << 16) | st.parts.size());
                    for (char ch : st.name) mix(uint8_t(ch));
                }
            }
            return h;
        };
        auto run_two = [&](uint32_t a_app, DirectPlan* a, const std::vector<PStage>& ma, DirectPlan* b, const std::vector<PStage>& mb) {
            const uint64_t th = topo_hash(ma, mb);
            const std::vector<uint16_t>* ord_ptr = nullptr;
            WaveTuner* tuner = nullptr;
            size_t picked = 0;
            if (autotune) {
                auto& wt = wave_tuners_[th];
                if (wt.cands.empty()) {
                    wt.cands = merge_order_candidates(ma, mb);
                    wt.sum_us.assign(wt.cands.size(), 0.0);
                    wt.cnt.assign(wt.cands.size(), 0);
                    if (wt.cands.size() <= 1) wt.best = 0;
                }
                if (wt.best >= 0) {
                    ord_ptr = &wt.cands[size_t(wt.best)];
                } else {
                    picked = wt.next % wt.cands.size();
                    ord_ptr = &wt.cands[picked];
                    tuner = &wt;
                }
            } else {
                auto it = orders_topo_.find(th);
                if (it == orders_topo_.end()) it = orders_topo_.emplace(th, merge_order(ma, mb)).first;
                ord_ptr = &it->second;
            }
            if (std::getenv("G2E_INK_WAVE_PROF")) orders_[std::make_pair(a, b)] = *ord_ptr;
            DirectWave w{a, b, *ord_ptr};
            const double d = rt_->run_waves(std::span<const DirectWave>(&w, 1));
            if (tuner) {
                tuner->sum_us[picked] += d;
                tuner->cnt[picked]++;
                tuner->next++;
                const auto& ap = m_.apps_[a_app];
                const uint32_t want_samples = m_.cfg_.full[ap.layer] ? 1u : 2u;
                if (tuner->next >= tuner->cands.size() * want_samples) {
                    int bi = 0;
                    double best_mean = 1e30;
                    for (size_t c = 0; c < tuner->cands.size(); ++c) {
                        const double mn = tuner->sum_us[c] / double(tuner->cnt[c]);
                        if (mn < best_mean) { best_mean = mn; bi = int(c); }
                    }
                    tuner->best = bi;
                    if (pprof)
                        std::fprintf(stderr, "[wave-tune] app=%u (full=%d msg=%d) picked cand %d/%zu: %.0f us (base cand 0: %.0f us)\n",
                                     a_app, int(m_.cfg_.full[ap.layer]), int(ap.pass > 0), bi, tuner->cands.size(),
                                     best_mean, tuner->sum_us[0] / double(tuner->cnt[0]));
                }
            }
            tot_dev += d;
            if (pprof) {
                const auto& ap = m_.apps_[a_app];
                const int kd = (m_.cfg_.full[ap.layer] ? 2 : 0) | (ap.pass > 0 ? 1 : 0);
                by_kind[kd] += d;
                cnt_kind[kd]++;
            }
        };
        auto do_patch = [&](uint32_t app_i, int sl, PPlan& pl) {
            const double u0 = pprof ? now_u() : 0.0;
            patch_b(app_i, sl, pl);
            if (pprof) tot_patch += now_u() - u0;
        };
        run_one(pa[0]->plan_a.get());
        run_far_chain(0, 0, pos, *pa[0], false);
        do_patch(0, 0, *pa[0]);
        for (uint32_t i = 0; i < n; ++i) {
            const double td = ensure_bf16_staged(m_.apps_[i].layer, 0, nullptr);
            tot_dev += td;
            tot_deq += td;
            run_two(i, pa[i]->plan_b, *pa[i]->meta_b, pb[i]->plan_a.get(), pb[i]->meta_a);
            run_far_chain(i, 1, pos + kT, *pb[i], true);
            do_patch(i, 1, *pb[i]);
            if (i + 1 < n) {
                run_two(i + 1, pb[i]->plan_b, *pb[i]->meta_b, pa[i + 1]->plan_a.get(), pa[i + 1]->meta_a);
                run_far_chain(i + 1, 0, pos, *pa[i + 1], true);
                do_patch(i + 1, 0, *pa[i + 1]);
            }
        }
        run_one(pb[n - 1]->plan_b);
    }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    if (pprof) {
        std::fprintf(stderr, "[pair-prof] pos=%u total=%.1f ms (dev=%.1f ms, patch=%.1f ms, deq=%.1f ms) | S=%.0f (%d) Sm=%.0f (%d) F=%.0f (%d) Fm=%.0f (%d) us\n",
                     pos, us * 1e-3, tot_dev * 1e-3, tot_patch * 1e-3, tot_deq * 1e-3,
                     cnt_kind[0] ? by_kind[0] / cnt_kind[0] : 0.0, cnt_kind[0],
                     cnt_kind[1] ? by_kind[1] / cnt_kind[1] : 0.0, cnt_kind[1],
                     cnt_kind[2] ? by_kind[2] / cnt_kind[2] : 0.0, cnt_kind[2],
                     cnt_kind[3] ? by_kind[3] / cnt_kind[3] : 0.0, cnt_kind[3]);
    }
    // G2E_INK_WAVE_PROF (measurements only, with inkling2_test --prefill-bench): reruns half-steps after the wave, so the
    // model state of the run is not valid afterwards. Never set it for a server.
    if (static const char* wp = std::getenv("G2E_INK_WAVE_PROF"); wp && !seq) {
        // (timing only: reruns half-step (B0[i] || A1[i]) alone and together; the states they write are garbage afterwards)
        const uint32_t i = uint32_t(std::atoi(wp));
        if (i < n) {
            pa[i]->plan_a->run();
            patch_b(i, 0, *pa[i]);
            ensure_bf16_staged(m_.apps_[i].layer, 0, nullptr);
            auto med = [&](const std::vector<DirectWave>& w) {
                std::vector<double> ts;
                for (int r = 0; r < 9; ++r) ts.push_back(rt_->run_waves(w));
                std::sort(ts.begin(), ts.end());
                return ts[4];
            };
            const auto key = std::make_pair(pa[i]->plan_b, pb[i]->plan_a.get());
            const auto& ord = orders_.at(key);
            const double ta = med({DirectWave{pa[i]->plan_b, nullptr, {}}});
            const double tb = med({DirectWave{pb[i]->plan_a.get(), nullptr, {}}});
            const double tp = med({DirectWave{pa[i]->plan_b, pb[i]->plan_a.get(), ord}});
            const double tn = med({DirectWave{pa[i]->plan_b, pb[i]->plan_a.get(), {}}});
            std::fprintf(stderr, "[ink-direct] wave half-step %u (B0 || A1): B0 alone %.0f us, A1 alone %.0f us, pair %.0f us, pair (B then A) %.0f us\n",
                         i, ta, tb, tp, tn);
            if (std::getenv("G2E_INK_WAVE_TIMELINE")) {
                // the half-step cut after each stage of its dispatch order (both plans rebuilt with the stages before
                // the cut; a cut on an MME stage of plan A is skipped): when every stage ends inside the wave
                const auto& mb = *pa[i]->meta_b;
                const auto& ma = pb[i]->meta_a;
                size_t kb = 0, ka = 0, cb = 0, ca = 0;  // stages / calls of B0 and A1 before the cut
                double prev = 0;
                for (size_t q = 0; q < ord.size(); ++q) {
                    const bool is_a = (ord[q] & 0x8000u) != 0;
                    const size_t k = ord[q] & 0x7fffu;
                    const auto& st = is_a ? ma[k] : mb[k];
                    if (is_a) { ka = std::max(ka, k + 1); ca = 0; for (size_t j = 0; j < ka; ++j) ca += ma[j].parts.size(); }
                    else { kb = std::max(kb, k + 1); cb = 0; for (size_t j = 0; j < kb; ++j) cb += mb[j].parts.size(); }
                    if (!ka || !kb) continue;
                    auto cut_b = build_pb(i, 0, io0[i].second, 0, 0, kb == mb.size() ? 0 : int(cb), nullptr);
                    auto cut_a = build_pa(i, 1, io1[i].first, 0, 0, pos + kT, ka == ma.size() ? 0 : int(ca), nullptr);
                    if (!cut_a || !cut_b) continue;
                    for (int d = 0; d < 8; ++d) cut_b->set_dyn_wait(d, pdyn_last_[0].tgt[d]);
                    cut_b->set_opt_mask(pdyn_last_[0].opt_mask);
                    // (a cut can leave MME stages nobody waits for: let them finish before the next run)
                    std::vector<double> ts;
                    for (int r = 0; r < 7; ++r) {
                        const DirectWave w{cut_b.get(), cut_a.get(), ord};
                        ts.push_back(rt_->run_waves(std::span<const DirectWave>(&w, 1)));
                        std::this_thread::sleep_for(std::chrono::microseconds(1500));
                    }
                    std::sort(ts.begin(), ts.end());
                    const double t = ts[3];
                    std::fprintf(stderr, "[wave-tl] %2zu %s %c%-2zu %-28s %7.0f us (+%.0f)\n", q, st.eng ? "TPC" : "MME", is_a ? 'a' : 'b', k,
                                 st.name.c_str(), t, t - prev);
                    prev = t;
                }
            }
        }
    }
    static const bool tm = std::getenv("G2E_INK_DIRECT_TIME") != nullptr;
    if (tm) {
        auto& v = chain_prof_[2000 + n];
        v.first += us;
        v.second++;
    }
    return us;
}

// Plan A (attention + post-attn conv + MLP mix + router + route + rplan_hist + rplan_tab): ends with hidx ready in HBM.
// lim > 0: only the first lim stages, the last one signalling the host (stage timing; nullptr when it is an MME stage)
std::unique_ptr<DirectPlan> InkDirect::build_pa(uint32_t app, int slot, int hcur, int cs_in, int cs_out, uint32_t pos,
                                                int lim, std::vector<PStage>* meta) {
    const auto& c = m_.cfg_;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const uint32_t T = kT, H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim, W = c.window, RE = c.rel_extent;
    const uint32_t G = NH / KV, GT = G * T, KVD = KV * D, DR = c.d_rel, E = c.experts, K = c.topk, S = T * K;
    const uint32_t QS = NH * D * 2 + NH * DR + NH, nbb = L / c.attn_res_block + 1, NT = H / 128;
    const bool boundary = L % c.attn_res_block == 0, msg = a.pass > 0, full = c.full[L];
    const bool three = msg || full;       // q / k / v projection inputs differ: three GEMMs, three row buffers
    const uint32_t Qa = full ? 3 : 1;     // depth-attention queries of the attention input
    const uint32_t KR = full ? KVD + KV * 4 : KVD;  // k (or v) columns (+ the dynamic conv taps of the full layers)
    const uint32_t NROW = QS + 2 * KVD;   // the one projection row of the single-input case: q | gate | rel | sink | k | v
    const uint32_t fr = full ? far_rows_at(pos, W) : 0;
    const uint32_t fb = (full && fr) ? m_.far_bucket(fr) : 0;
    const bool flash_inline = pfar_flash_ && psram_pw_on_ && full && fr > 0 && fr <= 1024u;
    const bool flash_split = pfar_flash_ && !flash_inline && full && fr > 0 &&
                             (!pfar_ok_ || fb > pfar_cap_ || !mme_.count("p_fqk_" + std::to_string(fb) + "_m512"));
    const uint32_t SF = (full && fr > 0 && !flash_split) ? (flash_inline ? 1024u : fb) : 0u;  // far key rows (bucketed) of a full layer
    const auto& l = lc_.at(L);
    const auto& pl = plc_.at(L);
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    auto wa = [&](const char* part) { return arena + w.at(Lp(L, part)).offset; };
    const auto& p = ps_[slot];
    // the chunk slot's activations: slot 0 the model's buffers, slot 1 their mirrors (the second chunk of a pair)
    const DeviceBuffer& bblk = slot ? m_.blocks1_ : m_.blocks_;
    const DeviceBuffer* bh = slot ? m_.h1_ : m_.h_;
    const DeviceBuffer* bprev = slot ? m_.prev1_ : m_.prev_;
    const uint64_t blocks = bblk.address(), h_in = bh[hcur].address();
    // the new K / V rows [2][KV][T][D], committed by the plan. With ping-pong conv states (a forward that also runs
    // Synapse recipes) they go to the model's scratch, which that forward's commit recipe writes once more.
    const uint64_t kvs = cs_in != cs_out ? m_.kvscratch_.address() + uint64_t(app) * 2 * KVD * Inkling2::kMaxT * 2 : p.kvs;
    const uint64_t cin = m_.conv_addr(app, m_.active_slot_, cs_in);
    const uint64_t cout = m_.conv_addr(app, m_.active_slot_, cs_out);  // (may be the same: in place)
    const uint64_t kring = m_.kring_addr(app, m_.active_slot_);
    const uint64_t vring = m_.vring_addr(app, m_.active_slot_);
    const uint64_t tab = ptab_.address() + uint64_t(slot) * kTabBytes;
    const float scale = 1.f / std::sqrt(float(D)), conv_s = 1.f / std::sqrt(5.f), mscale = 1.f / std::sqrt(float(H));
    struct ConvP { float scale, thr, eps; int32_t mode; };
    const auto bf = DType::BF16, f32 = DType::F32, i32 = DType::I32, f16 = DType::F16;

    static const char* const mg_out_k = [] {  // G2E_INK_PADE bit 2: the message gates' tanh
        const char* e = std::getenv("G2E_INK_PADE");
        return (!e || (std::atoi(e) & 2)) ? "ink_mg_out_pade_gaudi2" : "ink_mg_out_gaudi2";
    }();
    auto pp = std::make_unique<DirectPlan>(*rt_);
    auto& P = *pp;
    P.set_lane(1 + uint32_t(slot));  // (the two chunk slots' plans can run at the same time)
    int nsob = 0;
    int nst = 0;
    bool done = false, mme_cut = false;
    std::vector<std::string> names;
    // per plan stage: engine, the SOB it waits on, its parts (duration estimate, SOB signalled): the wave's interleaving
    auto note = [&](uint8_t eng, const std::string& k, uint32_t reps, int wsob, uint32_t wtgt, int sig, uint32_t nsig) {
        if (!meta) return;
        if (P.stage_count() > meta->size()) {
            meta->push_back(PStage{eng, wsob, {}, wtgt, {}, {}});
            meta->back().name = k;
        }
        meta->back().parts.push_back({stage_us(k, reps, SF), sig});
        meta->back().cnt.push_back(nsig);
    };
    std::map<int, uint32_t> cum_sig;  // signals of the MME calls so far, per SOB
    // TPC stage: signals min(grid[split], 24)
    // lim < 0: only the call number -lim, alone (no waits; an MME call is followed by a one-point TPC stage that waits
    // for it and signals the host): its duration on a device that does nothing else
    const int only = lim < 0 ? -lim : 0;
    auto tail = [&](uint32_t wtgt) {  // (the smallest TPC stage: one point of the copy kernel on the unused mix output)
        const DirectTpcTensor t[] = {tn(p.mixd, bf, {128, 32, 1}), tn(p.mixd + 65536, bf, {128, 32, 1})};
        P.add_tpc_grid(isa_.at("ink_pcopy_gaudi2"), t, {}, {1, 1, 1, 1, 1}, 1, 24, 0, wtgt, -1);
    };
    auto TG = [&](const char* k, std::span<const DirectTpcTensor> t, std::span<const uint32_t> sr, std::array<uint32_t, 5> grid,
                  uint32_t split, int wsob, uint32_t wtgt, int sig, uint32_t m_reps = 1) {
        const uint32_t n = std::min<uint32_t>(grid[split], 24);
        if (done) return n;
        names.push_back(k);
        if (only) {
            if (++nst == only) { P.add_tpc_grid(isa_.at(k), t, sr, grid, split, 24, -1, 0, -1); done = true; }
            return n;
        }
        if (lim > 0 && ++nst == lim) { sig = -1; done = true; }
        P.add_tpc_grid(isa_.at(k), t, sr, grid, split, 24, wsob, wtgt, sig);
        note(1, k, m_reps, wsob, wtgt, sig, n);
        return n;
    };
    // MME template issued reps times with address steps (bytes); only the first waits
    auto MM = [&](const char* k, uint32_t reps, uint64_t a0, uint64_t as, uint64_t b0, uint64_t bs, uint64_t c0, uint64_t cs,
                  int wsob, uint32_t wtgt, int sig) {
        const auto& tm = mme_.at(k);
        if (done) return reps * tm.signal_count;
        names.push_back(k);
        if (only) {
            if (++nst == only) {
                for (uint32_t i = 0; i < reps; ++i) P.add_mme(tm, a0 + i * as, b0 + i * bs, c0 + i * cs, 0, -1, 0, 0);
                tail(reps * tm.signal_count);
                done = true;
            }
            return reps * tm.signal_count;
        }
        for (uint32_t i = 0; i < reps; ++i)
            P.add_mme(tm, a0 + i * as, b0 + i * bs, c0 + i * cs, 0, i == 0 ? wsob : -1, i == 0 ? wtgt : 0, sig);
        cum_sig[sig] += reps * tm.signal_count;
        if (lim > 0 && ++nst == lim) {  // (stage timing: a one-point TPC stage waits for this call and signals the host)
            const DirectTpcTensor t[] = {tn(p.mixd, bf, {128, 32, 1}), tn(p.mixd + 65536, bf, {128, 32, 1})};
            P.add_tpc_grid(isa_.at("ink_pcopy_gaudi2"), t, {}, {1, 1, 1, 1, 1}, 1, 24, sig, cum_sig[sig], -1);
            done = true;
            return reps * tm.signal_count;
        }
        note(0, k, reps, wsob, wtgt, sig, reps * tm.signal_count);
        return reps * tm.signal_count;
    };
    // ---- attention input. Without messages: depth attention + RMSNorm -> normed [Qa][T][H] (layer 0: a plain RMSNorm
    // of the embedding). With messages: the attention-input mix and the message mix (F32, interleaved), the gate
    // GEMMs, then the gated + normed q / k / v inputs [3][T][H].
    const uint64_t slab = uint64_t(T) * H * 2;
    const uint32_t s0 = boundary ? nbb - 1 : nbb;                       // completed blocks of the attention mix
    const uint64_t apart = boundary ? blocks + uint64_t(nbb - 1) * slab : h_in;  // its partial state
    const uint64_t prev = msg ? bprev[a.pass == 1 ? 1 : 0].address() : 0;         // message sources [b1 | b2]
    const uint64_t RH = uint64_t(c.gate_rank) * 2 * H * 2;              // one gate's weights (bytes)
    const uint64_t wnq = full ? l.wn3 : wa("attn_norm");                // norm weights of the Qa mixes
    const int s_in = nsob++;
    uint32_t n_in = 0;
    if (L == 0) {
        const DirectTpcTensor t[] = {tn(blocks, bf, {H, T}), tn(wa("attn_norm"), bf, {H}), tn(p.xa, bf, {H, T})};
        n_in = TG("ink_pnorm_gaudi2", t, srfv(c.eps), {T, 1, 1, 1, 1}, 0, -1, 0, s_in);
    } else if (!msg) {
        const uint64_t q = m_.qil_.address() + m_.qil_off_.at(Lp(L, "attn_q"));
        const DirectTpcTensor t[] = {tn(blocks, bf, {H, T, s0}), tn(apart, bf, {H, T}), tn(q, f32, {H, Qa}), tn(wnq, bf, {H, Qa}),
                                     tn(p.xa, bf, {H, Qa, T}, {1, T * H, H}), tn(p.mixd, f32, {H, Qa, T})};
        struct { float scale, eps; int32_t flags; } prm{mscale, c.eps, 2};
        n_in = TG("ink_mix_gaudi2", t, srfv(prm), {24, 1, 1, 1, 1}, 0, -1, 0, s_in);
    } else {
        const int s_mx = nsob++;
        uint32_t n_mx = 0;
        {
            const uint64_t q = m_.qil_.address() + m_.qil_off_.at(Lp(L, "attn_q"));
            const DirectTpcTensor t[] = {tn(blocks, bf, {H, T, s0}), tn(apart, bf, {H, T}), tn(q, f32, {H, Qa}), tn(wnq, bf, {H, Qa}),
                                         tn(p.mixd, bf, {H, Qa, T}), tn(p.xil, f32, {H, Qa, T})};
            struct { float scale, eps; int32_t flags; } prm{mscale, c.eps, 1};
            n_mx += TG("ink_mix_gaudi2", t, srfv(prm), {24, 1, 1, 1, 1}, 0, -1, 0, s_mx);
            const uint64_t mq = m_.qil_.address() + m_.qil_off_.at("msg_q") + uint64_t(L - c.loop_start) * 4 * H * 4;
            const DirectTpcTensor t2[] = {tn(prev, bf, {H, T, 1}), tn(prev + slab, bf, {H, T}), tn(mq, f32, {H, 4}), tn(ones4_, bf, {H, 4}),
                                          tn(p.mixd, bf, {H, 4, T}), tn(p.msg, f32, {H, 4, T})};
            n_mx += TG("ink_mix_gaudi2", t2, srfv(prm), {24, 1, 1, 1, 1}, 0, -1, 0, s_mx);
        }
        const int s_cat = nsob++, s_gr = nsob++, s_fg = nsob++;
        struct { float eps; int32_t m0; } pm{c.eps, 0};
        {
            const DirectTpcTensor t[] = {tn(p.xil, f32, {H, Qa, T}), tn(p.msg, f32, {H, 4, T}), tn(p.cat, bf, {2 * H, T, 3})};
            TG("ink_mg_gaudi2", t, srfv(pm), {3, T, 1, 1, 1}, 1, s_mx, n_mx, s_cat, 3);
        }
        const uint64_t md = wa("mg_down"), mf = wa("mg_fg");
        const uint64_t gr_buf = psram_on_ ? psram_gr_ : p.gr;
        uint32_t n_gr = 0, n_fg = 0;
        for (uint32_t g = 0; g < 3; ++g)
            n_gr += MM("p_mgd1", 1, p.cat + g * uint64_t(T) * 2 * H * 2, 0, md + g * RH, 0,
                       gr_buf + g * uint64_t(T) * c.gate_rank * 2, 0, g == 0 ? s_cat : -1, 24, s_gr);
        for (uint32_t g = 0; g < 3; ++g)
            n_fg += MM("p_mgf1_m512", 2, gr_buf + g * uint64_t(T) * c.gate_rank * 2, 512ull * c.gate_rank * 2, mf + g * RH, 0,
                       p.fgv + g * uint64_t(T) * 2 * H * 4, 512ull * 2 * H * 4, g == 0 ? s_gr : -1, n_gr, s_fg);
        {
            const DirectTpcTensor t[] = {tn(p.xil, f32, {H, Qa, T}), tn(p.msg, f32, {H, 4, T}), tn(p.fgv, f32, {2 * H, T, 3}),
                                         tn(l.wn3, bf, {H, 3}), tn(p.xa, bf, {H, T, 3})};
            n_in = TG(mg_out_k, t, srfv(pm), {3, T, 1, 1, 1}, 1, s_fg, n_fg, s_in, 3);
        }
    }
    // the V ring in front of the new V rows: vcat [KV][W + T][D]
    const int s_vc = nsob++;
    uint32_t n_vc = 0;
    {
        const DirectTpcTensor t[] = {tn(vring, bf, {D, W, KV}), tn(p.vcat, bf, {D, W, KV}, {1, D, (W + T) * D})};
        n_vc += TG("ink_pcopy_gaudi2", t, {}, {KV, W / 32, 1, 1, 1}, 1, -1, 0, s_vc);
    }
    // ---- projections: one input -> row [T, NROW]; three inputs -> q side [T, QS], k [T, KR], v [T, KR]
    const int s_proj = nsob++;
    uint32_t n_proj = 0;
    const uint64_t wx = wa("wx");
    uint64_t rq = p.row, kx, vx;      // the q-side rows, the k and the v columns
    uint32_t rqs, kxs;                // their row strides (elements)
    if (!three) {
        n_proj = MM("p_proj_s_m512", 2, p.xa, 512ull * H * 2, wx, 0, p.row, 512ull * NROW * 2, s_in, n_in, s_proj);
        rqs = kxs = NROW;
        kx = p.row + uint64_t(QS) * 2;
        vx = p.row + uint64_t(QS + KVD) * 2;
    } else {
        const char* tkv = full ? "p_projkvf_m512" : "p_projkv_m512";
        rqs = QS;
        kxs = KR;
        kx = p.row + uint64_t(T) * QS * 2;
        vx = kx + uint64_t(T) * KR * 2;
        n_proj += MM("p_projq_m512", 2, p.xa, 512ull * H * 2, wx, 0, rq, 512ull * QS * 2, s_in, n_in, s_proj);
        n_proj += MM(tkv, 2, p.xa + slab, 512ull * H * 2, wx + uint64_t(QS) * H * 2, 0, kx, 512ull * KR * 2, -1, 0, s_proj);
        n_proj += MM(tkv, 2, p.xa + 2 * slab, 512ull * H * 2, wx + uint64_t(QS + KR) * H * 2, 0, vx, 512ull * KR * 2, -1, 0, s_proj);
    }
    const uint64_t tau = tab + kTabTau;  // f32 [T]: the log length scale of the full layers
    // ---- q prep (head-major q [NH][T][D]), k / v convs (head-major new rows into the KV scratch), r prep
    const uint64_t q_buf = (psram_q_ && !flash_split) ? psram_q_ : p.q;
    const uint64_t gated_buf = psram_extra_ ? psram_gated_ : p.gated;
    const uint64_t a1_buf = psram_extra_ ? psram_a1_ : p.a1;
    const float kv_s = full ? 1.f / 3.f : conv_s;
    static const bool xw = std::getenv("G2E_INK_DP_WAITS") != nullptr;  // (debug) explicit waits of r prep and the convs
    const int s_qk_in = nsob++;
    uint32_t n_qk_in = 0;
    {
        std::vector<DirectTpcTensor> t = {tn(rq, bf, {QS, T}, {1, rqs}), tn(l.qsb, bf, {D, NH})};
        if (full) t.push_back(tn(tau, f32, {T}));
        t.push_back(tn(q_buf, bf, {D, NH, T}, {1, T * D, D}));
        const float thr = c.qk_clip;
        n_qk_in += TG(full ? "ink_qprep_tau_bf16_gaudi2" : "ink_qprep_bf16_gaudi2", t, srfv(thr), {NH, 1, 1, 1, 1}, 0, s_proj, n_proj, s_qk_in);
    }
    const int s_rp = nsob++;
    {
        std::vector<DirectTpcTensor> t = {tn(rq, bf, {QS, T}, {1, rqs}), tn(pl.rsil, f32, {128, 3})};
        if (full) t.push_back(tn(tau, f32, {T}));
        t.push_back(tn(p.r16, bf, {NH * 16, T}));
        struct { float thr, isc; int32_t off_rel; } prm{c.rel_clip, 1.f / scale, int32_t(NH * D * 2)};
        // (no wait of its own, as the convs below: every core runs them after its part of the q prep, which waited
        // for the projections; a waiting TPC stage costs 24 monitors)
        TG(full ? "ink_prprep_tau_gaudi2" : "ink_prprep_gaudi2", t, srfv(prm), {T, 1, 1, 1, 1}, 0, xw ? s_proj : -1, xw ? n_proj : 0, s_rp);
    }
    // (V: the rows also behind the ring copy in vcat. A separate copy stage would wait on the V conv while the first
    // softmax, queued behind it, waits on the scores: an engine's fence takes whichever signal comes first, so the
    // copy could start on the scores' signal before the conv had finished.)
    auto kvconv = [&](uint64_t x, uint64_t st_in, uint64_t taps, uint64_t y, uint64_t y2, uint64_t st_out, int32_t mode, int sig) {
        std::vector<DirectTpcTensor> t = {tn(x, bf, {KVD, T}, {1, kxs}), tn(st_in, bf, {KVD, 3}), tn(taps, bf, {KVD, 4})};
        if (full) t.push_back(tn(x + uint64_t(KVD) * 2, bf, {KV * 4, T}, {1, kxs}));  // the dynamic taps' columns
        t.push_back(tn(y, bf, {D, KV, T}, {1, T * D, D}));
        if (y2) t.push_back(tn(y2, bf, {D, KV, T}, {1, (W + T) * D, D}));
        t.push_back(tn(st_out, bf, {KVD, 3}));
        t.push_back(tn(tab + kTabValidT, i32, {1}));
        const ConvP prm{kv_s, c.qk_clip, c.eps, mode};
        const char* k = full ? (y2 ? "ink_conv_dyn_hm2_bf16_gaudi2" : "ink_conv_dyn_hm_bf16_gaudi2")
                             : (y2 ? "ink_conv_hm2_bf16_gaudi2" : "ink_conv_hm_bf16_gaudi2");
        return TG(k, t, srfv(prm), {KVD / 128, 24, 1, 1, 1}, 1, xw ? s_proj : -1, xw ? n_proj : 0, sig);
    };
    n_qk_in += kvconv(kx, cin + m_.conv_off_[0], l.kvtaps, kvs, 0, cout + m_.conv_off_[0], 1, s_qk_in);
    kvconv(vx, cin + m_.conv_off_[1], l.kvtaps + uint64_t(4) * KVD * 2, kvs + uint64_t(KV) * T * D * 2, p.vcat + uint64_t(W) * D * 2,
           cout + m_.conv_off_[1], 2, s_vc);
    // ---- bias table br [T, NH, RE] (FP16, bias / scale, reversed distance), scores per KV head (FP16)
    const int s_bias = nsob++;
    MM("p_bias_r512", T * NH / 512, p.r16, 512ull * 16 * 2, l.projr, 0, p.bt, 512ull * RE * 2, s_rp, 24, s_bias);
    // Full layers with far keys (the linear cache rows [0, SF) before the ring's positions): FP16 scores q . K_far^T
    // and P_far . V_far per KV head through two alternating score / probability buffers, one head ahead: the MME
    // computes head h + 2's far scores under head h + 1's softmax (order: far(0), far(1), then per head PV, far PV,
    // far scores of h + 2).
    const MmeTemplate* tfq = SF ? &mme_.at("p_fqk_" + std::to_string(SF) + "_m512") : nullptr;
    const MmeTemplate* tfv = SF ? &mme_.at("p_fpv_" + std::to_string(SF) + "_m512") : nullptr;
    const uint64_t kfar = full ? m_.kcache_[m_.kv_index_[app]].address() : 0, vfar = full ? m_.vcache_[m_.kv_index_[app]].address() : 0;
    const uint64_t far_sz = uint64_t(GT) * pfar_cap_ * 2;  // one far buffer (bytes, HBM path)
    const uint64_t far_sram_sz = uint64_t(GT) * 1024ull * 2;  // 6 MiB per buffer (inline SRAM path, SF = 1024)
    auto sfar = [&](uint32_t h) { return flash_inline ? (psram_pw_ + (h & 1) * far_sram_sz) : (pfar_.address() + (h & 1) * far_sz); };
    auto pfar = [&](uint32_t h) { return flash_inline ? (psram_pw_ + (2 + (h & 1)) * far_sram_sz) : (pfar_.address() + (2 + (h & 1)) * far_sz); };
    const uint64_t fmax_addr = p.of + 2ull * GT * D * 4;  // [GT, KV] F32 running max (after 2 x 1.5 MiB of_tmp)
    const uint32_t n_fq = SF ? (GT / 512) * tfq->signal_count : 0;
    static const bool no_win = std::getenv("G2E_INK_NO_WIN") != nullptr;
    const bool win = !full && T == W && pos % W == 0 && !no_win;
    const uint64_t pbuf = win ? p.pw : p.p;
    auto sr_h = [&](uint32_t h) {
        return psram_on_ ? (psram_sr_ + uint64_t(h & 1) * GT * W * 2) : (p.sr + uint64_t(h) * GT * W * 2);
    };
    auto sw_h = [&](uint32_t h) {
        return psram_on_ ? (psram_sw_ + uint64_t(h & 1) * GT * T * 2) : (p.sw + uint64_t(h) * GT * T * 2);
    };
    auto pb_h = [&](uint32_t h) {
        return (win && psram_pw_on_) ? (psram_pw_ + uint64_t(h & 1) * GT * (W + T) * 2)
                                     : (pbuf + uint64_t(h) * GT * (W + T) * 2);
    };
    std::vector<int> s_sc(KV), s_pr(KV);
    std::vector<uint32_t> n_sc(KV, 0);
    for (uint32_t h = 0; h < KV; ++h) { s_sc[h] = nsob++; s_pr[h] = nsob++; }
    auto far_qk = [&](uint32_t h) {
        if (done) return;
        names.push_back("p_fqk");
        if (only) {
            if (++nst == only) {
                for (uint32_t i = 0; i < GT / 512; ++i)
                    P.add_mme(*tfq, q_buf + (uint64_t(h) * GT + i * 512ull) * D * 2, kfar + uint64_t(h) * m_.cap_ * D * 2,
                              sfar(h) + i * 512ull * SF * 2, 0, -1, 0, 0);
                tail((GT / 512) * tfq->signal_count);
                done = true;
            }
            return;
        }
        if (lim > 0 && ++nst == lim) { mme_cut = true; done = true; return; }
        for (uint32_t i = 0; i < GT / 512; ++i)
            P.add_mme(*tfq, q_buf + (uint64_t(h) * GT + i * 512ull) * D * 2, kfar + uint64_t(h) * m_.cap_ * D * 2,
                      sfar(h) + i * 512ull * SF * 2, 0, -1, 0, s_sc[h]);
        cum_sig[s_sc[h]] += (GT / 512) * tfq->signal_count;
        note(0, "p_fqk", GT / 512, -1, 0, s_sc[h], (GT / 512) * tfq->signal_count);
    };
    auto emit_sm = [&](uint32_t h) {
        std::vector<DirectTpcTensor> t;
        if (SF) t.push_back(tn(sfar(h), f16, {SF, GT}));
        t.push_back(tn(sr_h(h), f16, {W, GT}));
        t.push_back(tn(sw_h(h), f16, {T, GT}));
        t.push_back(tn(p.bt, f16, {RE, NH, T}));
        t.push_back(tn(rq + uint64_t(QS - NH) * 2, bf, {NH, T}, {1, rqs}));
        t.push_back(tn(tab + kTabPos, i32, {1}));
        if (SF) t.push_back(tn(pfar(h), bf, {SF, GT}));
        t.push_back(tn(pb_h(h), bf, {W + T, GT}));
        t.push_back(tn(p.sums + uint64_t(h) * GT * 4, f32, {1, GT}));
        if (flash_split) t.push_back(tn(fmax_addr + uint64_t(h) * GT * 4, f32, {1, GT}));
        struct { float scale; int32_t G, full, kv; } prm{scale, int32_t(G), full ? 1 : 0, int32_t(h)};
        const char* sm_k = SF ? "ink_attn_sm16fb_gaudi2"
                              : flash_split ? "ink_attn_sm16_near_gaudi2"
                                            : win ? "ink_attn_sm16w_gaudi2" : "ink_attn_sm16b_gaudi2";
        TG(sm_k, t, srfv(prm), {GT, 1, 1, 1, 1}, 0, s_sc[h], n_sc[h], s_pr[h]);
    };
    auto emit_fpv = [&](uint32_t h, int sig) -> uint32_t {
        const uint32_t sigs = (GT / 512) * tfv->signal_count;
        if (done) return sigs;
        names.push_back("p_fpv");
        if (only) {
            if (++nst == only) {
                for (uint32_t i = 0; i < GT / 512; ++i)
                    P.add_mme(*tfv, pfar(h) + i * 512ull * SF * 2, vfar + uint64_t(h) * m_.cap_ * D * 2,
                              p.of + (uint64_t(h) * GT + i * 512ull) * D * 4, 0, -1, 0, 0);
                tail(sigs);
                done = true;
            }
            return sigs;
        }
        if (lim > 0 && ++nst == lim) { mme_cut = true; done = true; return sigs; }
        for (uint32_t i = 0; i < GT / 512; ++i)
            P.add_mme(*tfv, pfar(h) + i * 512ull * SF * 2, vfar + uint64_t(h) * m_.cap_ * D * 2,
                      p.of + (uint64_t(h) * GT + i * 512ull) * D * 4, 0, -1, 0, sig);
        cum_sig[sig] += sigs;
        note(0, "p_fpv", GT / 512, -1, 0, sig, sigs);
        return sigs;
    };
    const int s_pv = nsob++;
    uint32_t n_pv = 0;
    if (psram_on_) {
        // 2-buffered SRAM attention pipeline in topological order:
        //   QK(0), QK(1) -> SM(0), SM(1) -> for h=0..5: [PV(h) + QK(h+2) -> SM(h+2)] -> PV(6), PV(7).
        //   PV(h) waits s_pr[h] (so SM(h) is done reading sr/sw[h&1]) and signals s_sc[h+2] along with QK(h+2)
        //   (so PV(h) is done reading pw[h&1] and QK(h+2) is done writing sr/sw[h&1] before SM(h+2) runs).
        for (uint32_t h = 0; h < 2; ++h) {
            if (h > 0 && !only && lim <= 0) P.end_mme_stage();
            const uint64_t qh = q_buf + uint64_t(h) * GT * D * 2;
            n_sc[h] = MM("p_qk_m512", GT / 512, qh, 512ull * D * 2, kring + uint64_t(h) * W * D * 2, 0, sr_h(h),
                         512ull * W * 2, h == 0 ? s_qk_in : -1, n_qk_in, s_sc[h]);
            n_sc[h] += MM("p_qk_m512", GT / 512, qh, 512ull * D * 2, kvs + uint64_t(h) * T * D * 2, 0, sw_h(h),
                          512ull * T * 2, -1, 0, s_sc[h]);
            if (SF) { far_qk(h); n_sc[h] += n_fq; }
        }
        emit_sm(0);
        emit_sm(1);
        for (uint32_t h = 0; h < KV; ++h) {
            const bool has_next = (h + 2 < KV);
            const int sig = has_next ? s_sc[h + 2] : s_pv;
            uint32_t pv_s = MM("p_pv2k", 1, pb_h(h), 0, p.vcat + uint64_t(h) * (W + T) * D * 2, 0,
                               p.o + uint64_t(h) * GT * D * 4, 0, s_pr[h], 24, sig);
            if (SF) pv_s += emit_fpv(h, sig);
            if (has_next) {
                const uint32_t hn = h + 2;
                const uint64_t qhn = q_buf + uint64_t(hn) * GT * D * 2;
                n_sc[hn] = pv_s;
                n_sc[hn] += MM("p_qk_m512", GT / 512, qhn, 512ull * D * 2, kring + uint64_t(hn) * W * D * 2, 0, sr_h(hn),
                               512ull * W * 2, -1, 0, s_sc[hn]);
                n_sc[hn] += MM("p_qk_m512", GT / 512, qhn, 512ull * D * 2, kvs + uint64_t(hn) * T * D * 2, 0, sw_h(hn),
                               512ull * T * 2, -1, 0, s_sc[hn]);
                if (SF) { far_qk(hn); n_sc[hn] += n_fq; }
                emit_sm(hn);
            } else {
                n_pv += pv_s;
            }
        }
    } else {
        for (uint32_t h = 0; h < KV; ++h) {
            if (h > 0 && !only && lim <= 0) P.end_mme_stage();
            const uint64_t qh = q_buf + uint64_t(h) * GT * D * 2;
            n_sc[h] = MM("p_qk_m512", GT / 512, qh, 512ull * D * 2, kring + uint64_t(h) * W * D * 2, 0, sr_h(h),
                         512ull * W * 2, h == 0 ? s_qk_in : -1, n_qk_in, s_sc[h]);
            n_sc[h] += MM("p_qk_m512", GT / 512, qh, 512ull * D * 2, kvs + uint64_t(h) * T * D * 2, 0, sw_h(h),
                          512ull * T * 2, -1, 0, s_sc[h]);
            n_sc[h] += n_fq;
        }
        if (SF) { far_qk(0); far_qk(1); }
        for (uint32_t h = 0; h < KV; ++h) emit_sm(h);
        for (uint32_t h = 0; h < KV; ++h) {
            n_pv += MM("p_pv2k", 1, pb_h(h), 0, p.vcat + uint64_t(h) * (W + T) * D * 2, 0,
                       p.o + uint64_t(h) * GT * D * 4, 0, s_pr[h], 24, s_pv);
            if (SF) {
                n_pv += emit_fpv(h, s_pv);
                if (h + 2 < KV) far_qk(h + 2);
            }
        }
    }
    if (flash_split) {
        // Split full-layer Plan A1 ends after near PV + KV commit (far keys read [0, pos - W), disjoint from [pos, pos + T));
        // build_pfar() runs tiled 48 MiB SRAM FlashAttention + Plan A2 (pgate .. rplan_dyn_tab) via run_chain!
        const int s_cm = nsob++;
        const DirectTpcTensor t[] = {tn(kvs, bf, {D, T, KV, 2}), tn(tab + kTabRows, i32, {T}), tn(kring, bf, {D, W, KV}),
                                     tn(vring, bf, {D, W, KV})};
        TG("ink_kv_commit_bf16_gaudi2", t, {}, {KV, T, 1, 1, 1}, 1, s_pv, n_pv, s_cm);
        const DirectTpcTensor t2[] = {tn(kvs, bf, {D, T, KV, 2}), tn(tab + kTabRows + kT * 4, i32, {T}),
                                      tn(kfar, bf, {D, m_.cap_, KV}), tn(vfar, bf, {D, m_.cap_, KV})};
        TG("ink_kv_commit_bf16_gaudi2", t2, {}, {KV, T, 1, 1, 1}, 1, -1, 0, -1);
        (void)n_vc;
        (void)s_vc;
        (void)bf;
        pnames_ = names;
        if (mme_cut || (lim > 0 && !done) || (only && !done)) return nullptr;
        P.prepare();
        return pp;
    }
    // ---- / sum, * sigmoid(gate) -> gated [T, NH D]; output projection
    const int s_gate = nsob++;
    {
        std::vector<DirectTpcTensor> t = {tn(p.o, f32, {D, GT, KV})};
        if (SF) t.push_back(tn(p.of, f32, {D, GT, KV}));
        t.push_back(tn(p.sums, f32, {GT, KV}));
        t.push_back(tn(rq, bf, {QS, T}, {1, rqs}));
        t.push_back(tn(gated_buf, bf, {NH * D, T}));
        struct { int32_t off_gate, G; } prm{int32_t(NH * D), int32_t(G)};
        TG(SF ? "ink_pgate_far_gaudi2" : "ink_pgate_gaudi2", t, srfv(prm), {NH, 1, 1, 1, 1}, 0, s_pv, n_pv, s_gate);
    }
    // ---- the new K / V rows into the ring (slot = position mod W) and, full layers, the linear cache: after every
    // read of the ring in this application (ring scores, the V ring copy; the far keys are older rows). No wait of
    // its own: every core runs it after its part of the gate, which waited for the P . V products.
    {
        const DirectTpcTensor t[] = {tn(kvs, bf, {D, T, KV, 2}), tn(tab + kTabRows, i32, {T}), tn(kring, bf, {D, W, KV}),
                                     tn(vring, bf, {D, W, KV})};
        const int s_cm = nsob++;
        TG("ink_kv_commit_bf16_gaudi2", t, {}, {KV, T, 1, 1, 1}, 1, -1, 0, s_cm);
        if (full) {
            const DirectTpcTensor t2[] = {tn(kvs, bf, {D, T, KV, 2}), tn(tab + kTabRows + kT * 4, i32, {T}),
                                          tn(kfar, bf, {D, m_.cap_, KV}), tn(vfar, bf, {D, m_.cap_, KV})};
            TG("ink_kv_commit_bf16_gaudi2", t2, {}, {KV, T, 1, 1, 1}, 1, -1, 0, s_cm);
        }
        if (flash_inline && psram_pw_on_) {
            // Re-zero [0 .. 24 MiB) SRAM after inline SF=1024 far attention (overlapped with p_wo_m512 on MME)
            const uint32_t ne = uint32_t(uint64_t(GT) * (W + T) * 2);
            const DirectTpcTensor zt[] = {tn(psram_pw_, bf, {ne})};
            TG("ink_zero_gaudi2", zt, {}, {ne / 128, 1, 1, 1, 1}, 0, -1, 0, s_cm);
        }
    }
    const int s_wo = nsob++;
    const uint32_t n_wo = MM("p_wo_m512", 2, gated_buf, 512ull * NH * D * 2, wa("wo"), 0, a1_buf, 512ull * H * 2, s_gate, 24, s_wo);
    // ---- post-attention conv + residual (res F32, interleaved), MLP input mix
    const int s_res = nsob++;
    {
        const DirectTpcTensor t[] = {tn(a1_buf, bf, {H, T}), tn(cin + m_.conv_off_[2], bf, {H, 3}), tn(l.ataps, bf, {H, 4}),
                                     tn(boundary ? p.zero : h_in, bf, {H, T}), tn(p.res, f32, {H, T}), tn(cout + m_.conv_off_[2], bf, {H, 3}),
                                     tn(tab + kTabValidT, i32, {1})};
        const ConvP prm{conv_s, c.qk_clip, c.eps, 0};
        TG("ink_conv_resa_bf16_gaudi2", t, srfv(prm), {NT, 24, 1, 1, 1}, 1, s_wo, n_wo, s_res);
    }
    // ---- MLP input: depth attention over [blocks; res] + RMSNorm (with messages: the mix, the MLP gate, the norm)
    const int s_xn = nsob++;
    uint32_t n_xn = 0;
    {
        const uint64_t q = m_.qil_.address() + m_.qil_off_.at(Lp(L, "mlp_q"));
        if (!msg) {
            const DirectTpcTensor t[] = {tn(blocks, bf, {H, T, nbb}), tn(p.res, f32, {H, T}), tn(q, f32, {H, 1}), tn(wa("mlp_norm"), bf, {H, 1}),
                                         tn(p.xn, bf, {H, 1, T}), tn(p.mixd, f32, {H, 1, T})};
            struct { float scale, eps; int32_t flags; } prm{mscale, c.eps, 2};
            n_xn = TG("ink_mix_f32p_gaudi2", t, srfv(prm), {24, 1, 1, 1, 1}, 0, s_res, 24, s_xn);
        } else {
            const int s_m2 = nsob++, s_c2 = nsob++, s_g2 = nsob++, s_f2 = nsob++;
            const DirectTpcTensor t[] = {tn(blocks, bf, {H, T, nbb}), tn(p.res, f32, {H, T}), tn(q, f32, {H, 1}), tn(wa("mlp_norm"), bf, {H, 1}),
                                         tn(p.mixd, bf, {H, 1, T}), tn(p.xil, f32, {H, 1, T})};
            struct { float scale, eps; int32_t flags; } prm{mscale, c.eps, 1};
            TG("ink_mix_f32p_gaudi2", t, srfv(prm), {24, 1, 1, 1, 1}, 0, s_res, 24, s_m2);
            struct { float eps; int32_t m0; } pm{c.eps, 3};
            const DirectTpcTensor t2[] = {tn(p.xil, f32, {H, 1, T}), tn(p.msg, f32, {H, 4, T}), tn(p.cat, bf, {2 * H, T, 1})};
            TG("ink_mg_gaudi2", t2, srfv(pm), {1, T, 1, 1, 1}, 1, s_m2, 24, s_c2);
            const uint64_t fgv_mlp = psram_on_ ? psram_fgv_ : p.fgv;
            const uint32_t n_g = MM("p_mgd1", 1, p.cat, 0, wa("mg_down") + 3 * RH, 0, p.gr, 0, s_c2, 24, s_g2);
            const uint32_t n_f = MM("p_mgf1_m512", 2, p.gr, 512ull * c.gate_rank * 2, wa("mg_fg") + 3 * RH, 0, fgv_mlp, 512ull * 2 * H * 4, s_g2, n_g, s_f2);
            const DirectTpcTensor t3[] = {tn(p.xil, f32, {H, 1, T}), tn(p.msg, f32, {H, 4, T}), tn(fgv_mlp, f32, {2 * H, T, 1}),
                                          tn(wa("mlp_norm"), bf, {H, 1}), tn(p.xn, bf, {H, T, 1})};
            n_xn = TG(mg_out_k, t3, srfv(pm), {1, T, 1, 1, 1}, 1, s_f2, n_f, s_xn);
        }
    }
    // ---- router, top-8, routing plan
    const int s_lg = nsob++;
    const uint32_t n_lg = MM("p_router", 1, p.xn, 0, wa("router"), 0, p.logits, 0, s_xn, n_xn, s_lg);
    const int s_rt = nsob++;
    {
        const DirectTpcTensor t[] = {tn(p.logits, f32, {E, T}), tn(wa("router_bias"), f32, {E}), tn(tab + kTabIota, f32, {E}),
                                     tn(p.idx, i32, {K, T}), tn(p.rw, f32, {K, T})};
        TG("ink_route_gaudi2", t, srfv(c.route_scale), {T, 1, 1, 1, 1}, 0, s_lg, n_lg, s_rt);
    }
    // routing plan: per-core histograms, then (one core) rank offsets / hot experts / counts -> signal host
    const int s_h = nsob++;
    {
        const DirectTpcTensor t[] = {tn(p.idx, i32, {S}), tn(tab + kTabTpart, i32, {25}), tn(p.hist, i32, {128, 24})};
        TG("ink_rplan_hist_gaudi2", t, srfv(int32_t(3)), {24, 1, 1, 1, 1}, 0, s_rt, 24, s_h);
    }
    {
        const DirectTpcTensor t[] = {tn(p.hist, i32, {128, 24}), tn(p.off, i32, {128, 24}), tn(p.cnt, i32, {128}),
                                     tn(p.exp_base, i32, {128})};
        TG("ink_rplan_dyn_tab_bf16_gaudi2", t, {}, {1, 1, 1, 1, 1}, 0, s_h, 24, -1);
    }
    (void)n_vc;
    (void)s_vc;
    (void)bf;
    pnames_ = names;
    if (mme_cut || (lim > 0 && !done) || (only && !done)) return nullptr;
    P.prepare();
    return pp;
}

// Per-KV-head HBM <-> SRAM staging plan for SRAM FlashAttention (dir=0: load o/q/m/l, dir=1: store o/l).
// Pre-built once per (chunk slot, KV head), ZERO runtime patching!
std::unique_ptr<DirectPlan> InkDirect::build_pfar_io(int slot, uint32_t h, int dir) {
    const auto& c = m_.cfg_;
    const uint32_t T = kT, NH = c.heads, KV = c.kv_heads, D = c.head_dim;
    const uint32_t G = NH / KV, GT = G * T;
    const auto bf = DType::BF16, f32 = DType::F32;
    const auto& p = ps_[slot];
    const uint64_t fmax_addr = p.of + 2ull * GT * D * 4;
    const DirectTpcTensor t[8] = {
        tn(p.o, f32, {D, GT, KV}),
        tn(pfar_o_h_, f32, {D, GT}),
        tn(p.q, bf, {D, GT, KV}),
        tn(pfar_q_h_, bf, {D, GT}),
        tn(fmax_addr, f32, {GT, KV}),
        tn(pfar_m_h_, f32, {GT}),
        tn(p.sums, f32, {GT, KV}),
        tn(pfar_l_h_, f32, {GT}),
    };
    struct { int32_t h, dir; } prm{int32_t(h), int32_t(dir)};
    auto pp = std::make_unique<DirectPlan>(*rt_);
    pp->add_tpc_grid(isa_.at("ink_pfar_head_io_gaudi2"), t, srfv(prm), {24, 1, 1, 1, 1}, 0, 24, -1, 0, -1);
    pp->prepare();
    return pp;
}

// Reusable slot-independent in-place 48 MiB SRAM FlashAttention tile-block plan (lane 0, B in {1, 2, 8, 24} tiles of
// width pfar_tw_ = 3072 on ONE KV head staged in SRAM). ALL scratch tensors (sf/pf in-place, of_tmp[2], o_h, q_h,
// m_h, l_h, fres[4]) live at fixed SRAM addresses -> ZERO HBM scratch traffic, ZERO Q/h patching, ZERO dirty TPC ECBs
// on full blocks! Dual MME fences (fence 3: TPC->FPV, fence 2: FPV->FQK) prevent engine-0/1 races on in-place sf/pf.
std::unique_ptr<DirectPlan> InkDirect::build_pfar_blk(uint32_t ntiles) {
    const auto& c = m_.cfg_;
    const uint32_t T = kT, NH = c.heads, KV = c.kv_heads, D = c.head_dim;
    const uint32_t G = NH / KV, GT = G * T;
    const uint32_t tw = pfar_tw_;
    const float scale = 1.f / std::sqrt(float(D));
    const auto bf = DType::BF16, f32 = DType::F32, f16 = DType::F16;
    const uint64_t fres_stride = al256(uint64_t(GT) * 4);
    auto fres_at = [&](uint32_t s) { return pfar_fres_ + uint64_t(s & 3u) * fres_stride; };

    const MmeTemplate& fqk_t = mme_.at("p_fqk_" + std::to_string(tw) + "_m512");
    const MmeTemplate& fpv_t = mme_.at("p_fpv_" + std::to_string(tw) + "_m512");
    const uint64_t isa_step = isa_.at("ink_attn_far_step_gaudi2");

    auto pp = std::make_unique<DirectPlan>(*rt_);
    auto& P = *pp;
    // SOBs: 0..1 = fqk[pr], 2..3 = fpv[pr], 4..5 = tpc[pr]
    uint32_t s_fqk[2] = {0, 0}, s_fpv[2] = {0, 0}, s_tpc[2] = {0, 0};

    // Patch tags: 10 + lk = K_tile[lk], 100 + lk = V_tile[lk]; SRF tag 1 = nv_last (only on last tile!).
    auto emit_fqk = [&](uint32_t lk, uint32_t pr, int wsob, uint32_t wtgt, int fence) {
        for (uint32_t i = 0; i < GT / 512; ++i)
            P.add_mme(fqk_t, pfar_q_h_ + i * 512ull * D * 2, pfar_o_h_, pfar_sf_pf_[pr] + i * 512ull * tw * 2, 0,
                      i == 0 ? wsob : -1, i == 0 ? wtgt : 0, int(pr), {-1, int(10 + lk), -1, -1},
                      i == 0 ? fence : -1);
        s_fqk[pr] += (GT / 512) * fqk_t.signal_count;
    };
    auto emit_fpv = [&](uint32_t lk, uint32_t pr, int wsob, uint32_t wtgt, int fence) {
        for (uint32_t i = 0; i < GT / 512; ++i)
            P.add_mme(fpv_t, pfar_sf_pf_[pr] + i * 512ull * tw * 2, pfar_o_h_, pfar_of_tmp_[pr] + i * 512ull * D * 4, 0,
                      i == 0 ? wsob : -1, i == 0 ? wtgt : 0, int(2u + pr), {-1, int(100 + lk), -1, -1},
                      i == 0 ? fence : -1);
        s_fpv[pr] += (GT / 512) * fpv_t.signal_count;
    };
    auto emit_step = [&](int sm_k, int acc_k, uint32_t pr, int wsob, uint32_t wtgt, int sig) {
        const DirectTpcTensor t[8] = {
            tn(pfar_sf_pf_[pr], f16, {tw, GT}),
            tn(pfar_sf_pf_[pr], bf, {tw, GT}),
            tn(pfar_m_h_, f32, {GT}),
            tn(pfar_l_h_, f32, {GT}),
            tn(fres_at(sm_k >= 0 ? uint32_t(sm_k) : 0u), f32, {GT}),
            tn(pfar_of_tmp_[pr], f32, {D, GT}),
            tn(pfar_o_h_, f32, {D, GT}),
            tn(fres_at(acc_k >= 0 ? uint32_t(acc_k) : 0u), f32, {GT}),
        };
        struct { float scale; int32_t nv, do_sm, do_acc; } prm{
            scale,
            sm_k >= 0 ? int32_t(tw) : 0,
            sm_k >= 0 ? 1 : 0,
            acc_k >= 0 ? 1 : 0,
        };
        const int stags[4] = {-1, (sm_k == int(ntiles) - 1) ? 1 : -1, -1, -1};
        P.add_tpc_grid(isa_step, t, srfv(prm), {GT, 1, 1, 1, 1}, 0, 24, wsob, wtgt, sig, 0xf, {}, 0, stags);
        if (sig == 4) s_tpc[0] += 24;
        else if (sig == 5) s_tpc[1] += 24;
    };

    if (ntiles == 1) {
        emit_fqk(0, 0, -1, 0, -1);
        emit_step(0, -1, 0, 0, s_fqk[0], 4);
        emit_fpv(0, 0, 4, s_tpc[0], 3);
        emit_step(-1, 0, 0, 2, s_fpv[0], -1);
    } else {
        std::vector<uint32_t> fqk_ready(ntiles, 0), fpv_ready(ntiles, 0);
        emit_fqk(0, 0, -1, 0, -1);
        fqk_ready[0] = s_fqk[0];
        emit_fqk(1, 1, -1, 0, -1);
        fqk_ready[1] = s_fqk[1];
        for (uint32_t k = 0; k < ntiles; ++k) {
            const uint32_t pr = k & 1u;
            emit_step(int(k), k >= 2 ? int(k - 2) : -1, pr, int(pr), fqk_ready[k], int(4u + pr));
            emit_fpv(k, pr, int(4u + pr), s_tpc[pr], 3);
            fpv_ready[k] = s_fpv[pr];
            if (k + 2 < ntiles) {
                emit_fqk(k + 2, pr, int(2u + pr), fpv_ready[k], 2);
                fqk_ready[k + 2] = s_fqk[pr];
            }
        }
        const uint32_t pe0 = (ntiles - 2) & 1u, pe1 = (ntiles - 1) & 1u;
        emit_step(-1, int(ntiles - 2), pe0, int(2u + pe0), fpv_ready[ntiles - 2], int(4u + pe0));
        emit_step(-1, int(ntiles - 1), pe1, int(2u + pe1), fpv_ready[ntiles - 1], -1);
    }
    P.prepare();
    return pp;
}

double InkDirect::run_pfar_loop(uint32_t app, int slot, uint32_t pos, PPlan& pl) {
    if (!pl.plan_a2) return 0.0;
    const auto& c = m_.cfg_;
    const uint32_t KV = c.kv_heads, D = c.head_dim, W = c.window;
    const uint32_t tw = pfar_tw_;
    const uint32_t fr = far_rows_at(pos, W);
    const uint64_t kfar = m_.kcache_[m_.kv_index_[app]].address();
    const uint64_t vfar = m_.vcache_[m_.kv_index_[app]].address();
    for (uint32_t h = 0; h < KV; ++h) {
        if (!pfar_load_[slot][h]) pfar_load_[slot][h] = build_pfar_io(slot, h, 0);
        if (!pfar_store_[slot][h]) pfar_store_[slot][h] = build_pfar_io(slot, h, 1);
    }

    static const bool pprof = std::getenv("G2E_INK_PAIR_PROF") != nullptr;
    static double s_pf_dev = 0, s_pf_wall = 0;
    static uint64_t s_pf_chains = 0, s_pf_calls = 0;
    auto now_u = [] { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double t_start = pprof ? now_u() : 0.0;

    std::vector<DirectPlan*> chain;
    chain.reserve(32);
    std::array<uint32_t, 4> pool_used{};
    double tot = 0.0;
    auto flush_chain = [&] {
        if (chain.empty()) return;
        tot += rt_->run_chain(chain);
        chain.clear();
        pool_used.fill(0);
        if (pprof) ++s_pf_chains;
    };

    for (uint32_t h = 0; h < KV; ++h) {
        chain.push_back(pfar_load_[slot][h].get());
        uint32_t col = 0;
        while (col < fr) {
            const uint32_t nt_rem = (fr - col + tw - 1u) / tw;
            int bi = 0;
            for (int k = 3; k >= 0; --k)
                if (kPFarBlkSizes[k] <= nt_rem) { bi = k; break; }
            const uint32_t B = kPFarBlkSizes[bi];
            if (pool_used[bi] >= kPFarBlkPoolMax[bi]) flush_chain();
            if (pool_used[bi] >= pfar_blk_pool_[bi].size()) {
                PFarBlkEntry ent;
                ent.plan = build_pfar_blk(B);
                ent.last_nv = tw;
                pfar_blk_pool_[bi].push_back(std::move(ent));
            }
            auto& ent = pfar_blk_pool_[bi][pool_used[bi]++];
            auto& bp = *ent.plan;
            for (uint32_t k = 0; k < B; ++k) {
                const uint64_t kv_off = (uint64_t(h) * m_.cap_ + col + k * tw) * D * 2;
                bp.patch_addr(int(10 + k), kfar + kv_off);
                bp.patch_addr(int(100 + k), vfar + kv_off);
            }
            const uint32_t last_col = col + (B - 1u) * tw;
            const uint32_t nv = std::min<uint32_t>(tw, fr - last_col);
            if (nv != ent.last_nv) {
                bp.patch_srf(1, nv);
                ent.last_nv = nv;
            }
            chain.push_back(&bp);
            col += B * tw;
        }
        chain.push_back(pfar_store_[slot][h].get());
    }
    chain.push_back(pl.plan_a2.get());
    flush_chain();

    if (pprof) {
        s_pf_dev += tot;
        s_pf_wall += now_u() - t_start;
        if (++s_pf_calls % 16 == 0) {
            std::fprintf(stderr, "[pfar-prof] 16 full evals (fr=%u tw=%u): wall=%.1f ms, dev=%.1f ms, patch+host=%.1f ms, chains=%llu (%.1f/eval)\n",
                         fr, tw, s_pf_wall * 1e-3, s_pf_dev * 1e-3, (s_pf_wall - s_pf_dev) * 1e-3,
                         (unsigned long long)s_pf_chains, double(s_pf_chains) / 16.0);
            s_pf_dev = s_pf_wall = 0;
            s_pf_chains = 0;
        }
    }
    return tot;
}

// Post-far Plan A2 (lane 0): pgate (merged p.o / p.sums) -> wo -> conv_resa -> MLP mix/mg -> router -> rplan_tab.
// Independent of far-key count `fr` (built once per app/slot and reused from 34K to 1,048,576).
std::unique_ptr<DirectPlan> InkDirect::build_pa2(uint32_t app, int slot, int hcur, int cs_in, int cs_out) {
    const auto& c = m_.cfg_;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const uint32_t T = kT, H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim;
    const uint32_t G = NH / KV, GT = G * T, DR = c.d_rel, E = c.experts, K = c.topk, S = T * K;
    const uint32_t QS = NH * D * 2 + NH * DR + NH, nbb = L / c.attn_res_block + 1, NT = H / 128;
    const bool boundary = L % c.attn_res_block == 0, msg = a.pass > 0;
    const auto& l = lc_.at(L);
    const auto& w = m_.w_;
    const uint64_t arena = w.arena().address();
    auto wa = [&](const char* part) { return arena + w.at(Lp(L, part)).offset; };
    const auto& p = ps_[slot];
    const DeviceBuffer& bblk = slot ? m_.blocks1_ : m_.blocks_;
    const DeviceBuffer* bh = slot ? m_.h1_ : m_.h_;
    const uint64_t blocks = bblk.address(), h_in = bh[hcur].address();
    const uint64_t cin = m_.conv_addr(app, m_.active_slot_, cs_in);
    const uint64_t cout = m_.conv_addr(app, m_.active_slot_, cs_out);
    const uint64_t tab = ptab_.address() + uint64_t(slot) * kTabBytes;
    const uint64_t RH = uint64_t(c.gate_rank) * 2 * H * 2;
    const float conv_s = 1.f / std::sqrt(5.f), mscale = 1.f / std::sqrt(float(H));
    struct ConvP { float scale, thr, eps; int32_t mode; };
    const auto bf = DType::BF16, f32 = DType::F32, i32 = DType::I32;
    static const char* const mg_out_k = [] {
        const char* e = std::getenv("G2E_INK_PADE");
        return (!e || (std::atoi(e) & 2)) ? "ink_mg_out_pade_gaudi2" : "ink_mg_out_gaudi2";
    }();
    const uint64_t gated_buf = psram_extra_ ? psram_gated_ : p.gated;
    const uint64_t a1_buf = psram_extra_ ? psram_a1_ : p.a1;

    auto pp = std::make_unique<DirectPlan>(*rt_);
    auto& P = *pp;
    int nsob = 0;
    auto TG = [&](const char* k, std::span<const DirectTpcTensor> t, std::span<const uint32_t> sr, std::array<uint32_t, 5> grid,
                  uint32_t split, int wsob, uint32_t wtgt, int sig) {
        const uint32_t n = std::min<uint32_t>(grid[split], 24);
        P.add_tpc_grid(isa_.at(k), t, sr, grid, split, 24, wsob, wtgt, sig);
        return n;
    };
    auto MM = [&](const char* k, uint32_t reps, uint64_t a0, uint64_t as, uint64_t b0, uint64_t bs, uint64_t c0, uint64_t cs,
                  int wsob, uint32_t wtgt, int sig) {
        const auto& tm = mme_.at(k);
        for (uint32_t i = 0; i < reps; ++i)
            P.add_mme(tm, a0 + i * as, b0 + i * bs, c0 + i * cs, 0, i == 0 ? wsob : -1, i == 0 ? wtgt : 0, sig);
        return reps * tm.signal_count;
    };
    const int s_gate = nsob++;
    {
        const DirectTpcTensor t[] = {tn(p.o, f32, {D, GT, KV}), tn(p.sums, f32, {GT, KV}),
                                     tn(p.row, bf, {QS, T}, {1, QS}), tn(gated_buf, bf, {NH * D, T})};
        struct { int32_t off_gate, G; } prm{int32_t(NH * D), int32_t(G)};
        TG("ink_pgate_gaudi2", t, srfv(prm), {NH, 1, 1, 1, 1}, 0, -1, 0, s_gate);
    }
    const int s_wo = nsob++;
    const uint32_t n_wo = MM("p_wo_m512", 2, gated_buf, 512ull * NH * D * 2, wa("wo"), 0, a1_buf, 512ull * H * 2, s_gate, 24, s_wo);
    if (psram_pw_on_) {
        // Re-zero [24 .. 48 MiB) SRAM touched by SRAM FlashAttention (overlapped with p_wo_m512 on MME -> 0 us extra!)
        const uint32_t ne = uint32_t(uint64_t(GT) * (c.window + T) * 2);
        const DirectTpcTensor zt[] = {tn(psram_pw_, bf, {ne})};
        const int s_z = nsob++;
        TG("ink_zero_gaudi2", zt, {}, {ne / 128, 1, 1, 1, 1}, 0, -1, 0, s_z);
    }
    const int s_res = nsob++;
    {
        const DirectTpcTensor t[] = {tn(a1_buf, bf, {H, T}), tn(cin + m_.conv_off_[2], bf, {H, 3}), tn(l.ataps, bf, {H, 4}),
                                     tn(boundary ? p.zero : h_in, bf, {H, T}), tn(p.res, f32, {H, T}), tn(cout + m_.conv_off_[2], bf, {H, 3}),
                                     tn(tab + kTabValidT, i32, {1})};
        const ConvP prm{conv_s, c.qk_clip, c.eps, 0};
        TG("ink_conv_resa_bf16_gaudi2", t, srfv(prm), {NT, 24, 1, 1, 1}, 1, s_wo, n_wo, s_res);
    }
    const int s_xn = nsob++;
    uint32_t n_xn = 0;
    {
        const uint64_t q = m_.qil_.address() + m_.qil_off_.at(Lp(L, "mlp_q"));
        if (!msg) {
            const DirectTpcTensor t[] = {tn(blocks, bf, {H, T, nbb}), tn(p.res, f32, {H, T}), tn(q, f32, {H, 1}), tn(wa("mlp_norm"), bf, {H, 1}),
                                         tn(p.xn, bf, {H, 1, T}), tn(p.mixd, f32, {H, 1, T})};
            struct { float scale, eps; int32_t flags; } prm{mscale, c.eps, 2};
            n_xn = TG("ink_mix_f32p_gaudi2", t, srfv(prm), {24, 1, 1, 1, 1}, 0, s_res, 24, s_xn);
        } else {
            const int s_m2 = nsob++, s_c2 = nsob++, s_g2 = nsob++, s_f2 = nsob++;
            const DirectTpcTensor t[] = {tn(blocks, bf, {H, T, nbb}), tn(p.res, f32, {H, T}), tn(q, f32, {H, 1}), tn(wa("mlp_norm"), bf, {H, 1}),
                                         tn(p.mixd, f32, {H, 1, T}), tn(p.xil, f32, {H, 1, T})};
            struct { float scale, eps; int32_t flags; } prm{mscale, c.eps, 1};
            TG("ink_mix_f32p_gaudi2", t, srfv(prm), {24, 1, 1, 1, 1}, 0, s_res, 24, s_m2);
            struct { float eps; int32_t m0; } pm{c.eps, 3};
            const DirectTpcTensor t2[] = {tn(p.xil, f32, {H, 1, T}), tn(p.msg, f32, {H, 4, T}), tn(p.cat, bf, {2 * H, T, 1})};
            TG("ink_mg_gaudi2", t2, srfv(pm), {1, T, 1, 1, 1}, 1, s_m2, 24, s_c2);
            const uint64_t fgv_mlp = psram_on_ ? psram_fgv_ : p.fgv;
            const uint32_t n_g = MM("p_mgd1", 1, p.cat, 0, wa("mg_down") + 3 * RH, 0, p.gr, 0, s_c2, 24, s_g2);
            const uint32_t n_f = MM("p_mgf1_m512", 2, p.gr, 512ull * c.gate_rank * 2, wa("mg_fg") + 3 * RH, 0, fgv_mlp, 512ull * 2 * H * 4, s_g2, n_g, s_f2);
            const DirectTpcTensor t3[] = {tn(p.xil, f32, {H, 1, T}), tn(p.msg, f32, {H, 4, T}), tn(fgv_mlp, f32, {2 * H, T, 1}),
                                          tn(wa("mlp_norm"), bf, {H, 1}), tn(p.xn, bf, {H, T, 1})};
            n_xn = TG(mg_out_k, t3, srfv(pm), {1, T, 1, 1, 1}, 1, s_f2, n_f, s_xn);
        }
    }
    const int s_lg = nsob++;
    const uint32_t n_lg = MM("p_router", 1, p.xn, 0, wa("router"), 0, p.logits, 0, s_xn, n_xn, s_lg);
    const int s_rt = nsob++;
    {
        const DirectTpcTensor t[] = {tn(p.logits, f32, {E, T}), tn(wa("router_bias"), f32, {E}), tn(tab + kTabIota, f32, {E}),
                                     tn(p.idx, i32, {K, T}), tn(p.rw, f32, {K, T})};
        TG("ink_route_gaudi2", t, srfv(c.route_scale), {T, 1, 1, 1, 1}, 0, s_lg, n_lg, s_rt);
    }
    const int s_h = nsob++;
    {
        const DirectTpcTensor t[] = {tn(p.idx, i32, {S}), tn(tab + kTabTpart, i32, {25}), tn(p.hist, i32, {128, 24})};
        TG("ink_rplan_hist_gaudi2", t, srfv(int32_t(3)), {24, 1, 1, 1, 1}, 0, s_rt, 24, s_h);
    }
    {
        const DirectTpcTensor t[] = {tn(p.hist, i32, {128, 24}), tn(p.off, i32, {128, 24}), tn(p.cnt, i32, {128}),
                                     tn(p.exp_base, i32, {128})};
        TG("ink_rplan_dyn_tab_bf16_gaudi2", t, {}, {1, 1, 1, 1, 1}, 0, s_h, 24, -1);
    }
    P.prepare();
    return pp;
}

// Plan B (BF16 dynamic grouped-GEMM MoE + post-MLP conv/residual + optional message-source/tap copy):
//   rplan_dyn_slots_bf16 -> sgather_bf16 -> 4-wave pipelined bdyngu[0..3] / act_bf16[0..3] / bdyndn[0..3] -> combine_bf16 -> conv_resm.
// Zero per-layer static MoE code, zero empty experts, zero overflow; all MME descriptors generated on-the-fly in pdyn_[slot].
std::unique_ptr<DirectPlan> InkDirect::build_pb(uint32_t app, int slot, uint64_t h_out, int cs_in, int cs_out, int lim,
                                                std::vector<PStage>* meta) {
    const auto& c = m_.cfg_;
    const auto& a = m_.apps_[app];
    const uint32_t L = a.layer;
    const uint32_t T = kT, H = c.hidden, E = c.experts, F = c.moe_inter, K = c.topk, S = T * K, NT = H / 128;
    const auto& l = lc_.at(L);
    const auto& p = ps_[slot];
    const DeviceBuffer* bprev = slot ? m_.prev1_ : m_.prev_;
    const DeviceBuffer& baux = slot ? m_.aux1_ : m_.aux_;
    const uint64_t cin = m_.conv_addr(app, m_.active_slot_, cs_in);
    const uint64_t cout = m_.conv_addr(app, m_.active_slot_, cs_out);
    const uint64_t tab = ptab_.address() + uint64_t(slot) * kTabBytes;
    const float conv_s = 1.f / std::sqrt(5.f);
    struct ConvP { float scale, thr, eps; int32_t mode; };
    const auto bf = DType::BF16, f32 = DType::F32, i32 = DType::I32;

    auto pp = std::make_unique<DirectPlan>(*rt_);
    auto& P = *pp;
    P.set_lane(1 + uint32_t(slot));
    int nsob = 6, nst = 0;  // SOBs 0..5 reserved for kPbSob{Sl,Q,Gu,A,Dn,Mo}
    bool done = false, mme_cut = false;
    std::vector<std::string> names;
    auto note = [&](uint8_t eng, const std::string& k, uint32_t reps, int wsob, uint32_t wtgt, int sig, uint32_t nsig) {
        if (!meta) return;
        if (P.stage_count() > meta->size()) {
            meta->push_back(PStage{eng, wsob, {}, wtgt, {}, {}});
            meta->back().name = k;
        }
        meta->back().parts.push_back({stage_us(k, reps, 0), sig});
        meta->back().cnt.push_back(nsig);
    };
    const int only = lim < 0 ? -lim : 0;
    auto TG = [&](const char* k, std::span<const DirectTpcTensor> t, std::span<const uint32_t> sr, std::array<uint32_t, 5> grid,
                  uint32_t split, int wsob, uint32_t wtgt, int sig, uint32_t m_reps = 1) {
        const uint32_t n = std::min<uint32_t>(grid[split], 24);
        if (done) return n;
        names.push_back(k);
        if (only) {
            if (++nst == only) { P.add_tpc_grid(isa_.at(k), t, sr, grid, split, 24, -1, 0, -1); done = true; }
            return n;
        }
        if (lim > 0 && ++nst == lim) { sig = -1; done = true; }
        P.add_tpc_grid(isa_.at(k), t, sr, grid, split, 24, wsob, wtgt, sig);
        note(1, k, m_reps, wsob, wtgt, sig, n);
        return n;
    };
    auto DM = [&](const char* k, uint32_t dyn_idx, int wsob, int meta_wsob, uint32_t meta_wtgt, int sig) {
        if (done) return;
        names.push_back(k);
        if (only) { ++nst; return; }
        P.add_dyn_mme(pdyn_[slot].get(), dyn_idx, wsob, sig);
        if (lim > 0 && ++nst == lim) {  // (stage timing: a one-point TPC stage waits for this wave and signals the host)
            const DirectTpcTensor t[] = {tn(p.mixd, bf, {128, 32, 1}), tn(p.mixd + 65536, bf, {128, 32, 1})};
            P.set_next_dyn_wait(dyn_idx < 4 ? int(dyn_idx) : dyn_idx == 7 ? 4 : int(dyn_idx) + 1);
            P.add_tpc_grid(isa_.at("ink_pcopy_gaudi2"), t, {}, {1, 1, 1, 1, 1}, 1, 24, sig, 0, -1);
            done = true;
            return;
        }
        note(0, k, 1, meta_wsob, meta_wtgt, sig, 2);
    };

    const uint64_t py = p.gu;  // y [H, R] F32 aliases gu [2F, R] F32 (disjoint per wave since 2F == H)
    {
        const DirectTpcTensor t[] = {tn(p.idx, i32, {S}), tn(p.off, i32, {128, 24}), tn(p.exp_base, i32, {128}),
                                     tn(tab + kTabTpart, i32, {25}), tn(p.code, i32, {S}), tn(p.lbuf, i32, {S, 4}),
                                     tn(p.lcnt, i32, {24, 4})};
        TG("ink_rplan_dyn_slots_bf16_gaudi2", t, srfv(int32_t(3)), {24, 1, 1, 1, 1}, 0, -1, 0, kPbSobSl);
    }
    {
        const DirectTpcTensor t[] = {tn(p.xn, bf, {H, T}), tn(p.code, i32, {K, T}), tn(p.xq, bf, {H, kFlatRows}),
                                     tn(tab + kTabValidT, i32, {1})};
        // (no wait: same 24-core token split as rplan_dyn_slots_bf16)
        TG("ink_sgather_bf16_gaudi2", t, {}, {T, 1, 1, 1, 1}, 0, -1, 0, kPbSobQ);
    }
    // 4-wave pipelined dynamic grouped-GEMM: bdyngu[0..3] -> act_bf16[0..3] -> bdyndn[0..3]
    for (uint32_t wv = 0; wv < 4; ++wv)
        DM("p_bdyngu", wv, kPbSobQ, wv == 0 ? kPbSobQ : -1, wv == 0 ? 24 : 0, kPbSobGu);
    const bool dyn_en = !only;  // (stage timing with lim > 0 too: profile_stages_p sets the last patch's targets)
    for (uint32_t wv = 0; wv < 4; ++wv) {
        if (dyn_en) { P.set_opt_id(int(wv)); P.set_next_dyn_wait(int(wv)); }
        const DirectTpcTensor t[] = {tn(p.gu, f32, {2 * F, kFlatRows}), tn(p.code, i32, {S}), tn(p.idx, i32, {S}),
                                     tn(m_.nv_gvec13(L), f32, {2 * F, E}), tn(p.aq, bf, {F, kFlatRows}),
                                     tn(p.lbuf + uint64_t(wv) * S * 4, i32, {S}), tn(p.lcnt + uint64_t(wv) * 24 * 4, i32, {24}),
                                     tn(tab + kTabTpart, i32, {25})};
        struct { float cu; int32_t lk; } prm{c.up_softcap, 3};
        TG("ink_act_bf16_gaudi2", t, srfv(prm), {24, 1, 1, 1, 1}, 0, kPbSobGu, 2u * (wv + 1), kPbSobA);
    }
    if (dyn_en) P.set_opt_id(-1);
    for (uint32_t wv = 0; wv < 4; ++wv)
        DM("p_bdyndn", 4 + wv, kPbSobA, kPbSobA, 24u * (wv + 1), kPbSobDn);
    // ---- slot sum -> mo BF16, MLP conv + residual -> h_out
    {
        const DirectTpcTensor tb[] = {tn(py, f32, {H, kFlatRows}), tn(m_.nv_gvec2(L), f32, {H, E}),
                                      tn(p.code, i32, {K, T}), tn(p.idx, i32, {K, T}), tn(p.rw, f32, {K, T}),
                                      tn(p.mo, bf, {H, T}), tn(tab + kTabValidT, i32, {1})};
        if (dyn_en) P.set_next_dyn_wait(4);
        TG("ink_moe_combine_bf16_gaudi2", tb, {}, {T, 1, 1, 1, 1}, 0, kPbSobDn, 8u, kPbSobMo);
    }
    {
        const DirectTpcTensor t[] = {tn(p.mo, bf, {H, T}), tn(cin + m_.conv_off_[3], bf, {H, 3}), tn(l.mtaps, bf, {H, 4}), tn(p.res, f32, {H, T}),
                                     tn(h_out, bf, {H, T}), tn(cout + m_.conv_off_[3], bf, {H, 3}), tn(tab + kTabValidT, i32, {1})};
        const ConvP prm{conv_s, c.qk_clip, c.eps, 0};
        // second copies of the output: L11 of passes 0 / 1 = the next pass's message source (block 1 of the pass);
        // applications 4 / 22 / 40 = the DSpark taps (the partial block entering visits 5 / 23 / 41)
        const bool src_copy = L + 1 == c.loop_end - c.attn_res_block && a.pass + 1 < c.passes;
        const int tap = !m_.aux_on_ ? -1 : app == 4 ? 0 : app == 22 ? 1 : app == 40 ? 2 : -1;
        const int s_out = (src_copy || tap >= 0) ? nsob++ : -1;
        TG("ink_conv_resm_bf16_gaudi2", t, srfv(prm), {NT, 24, 1, 1, 1}, 1, kPbSobMo, 24, s_out);
        auto copy_out = [&](uint64_t dst, bool last) {
            const DirectTpcTensor tc[] = {tn(h_out, bf, {128, T, NT}, {1, H, 128}), tn(dst, bf, {128, T, NT}, {1, H, 128})};
            TG("ink_pcopy_gaudi2", tc, {}, {NT, T / 32, 1, 1, 1}, 1, s_out, 24, last ? -1 : s_out + 0);
        };
        if (src_copy && tap >= 0) throw std::logic_error("inkling2 direct prefill: two output copies");
        if (src_copy) copy_out(bprev[a.pass == 0 ? 1 : 0].address(), true);
        if (tap >= 0) copy_out(baux.address() + uint64_t(tap) * Inkling2::kMaxT * H * 2, true);
    }
    pnames_ = names;
    if (mme_cut || (lim > 0 && !done) || (only && !done)) return nullptr;
    P.prepare();
    return pp;
}

// (debug) hashes of the slot-0 scratch tensors after an application: which stage's output differs between two runs
void InkDirect::hash_scratch(uint32_t app) {
    const auto& c = m_.cfg_;
    const uint64_t T = kT, H = c.hidden, NH = c.heads, KV = c.kv_heads, D = c.head_dim, GT = 3 * T, S = T * c.topk, E = c.experts;
    const auto& p = ps_[0];
    const uint64_t base = pscratch_[0].address();
    auto fnv = [&](uint64_t addr, uint64_t bytes) {
        HostBuffer hb(m_.dev_, bytes + 64);
        m_.stream_.download(pscratch_[0], addr - base, hb, 0, bytes);
        m_.stream_.synchronize();
        uint64_t h = 1469598103934665603ull;
        const uint8_t* d = hb.as<uint8_t>();
        for (uint64_t i = 0; i < bytes; ++i) h = (h ^ d[i]) * 1099511628211ull;
        return h;
    };
    const uint64_t F = c.moe_inter;
    std::vector<std::pair<const char*, std::pair<uint64_t, uint64_t>>> items = {
        {"xa", {p.xa, 3 * T * H * 2}}, {"row", {p.row, T * (6552 + 2 * 1056) * 2}}, {"q", {p.q, NH * T * D * 2}}, {"r16", {p.r16, T * NH * 16 * 2}},
        {"bt", {p.bt, T * NH * 1024 * 2}}, {"kvs", {p.kvs, 2 * KV * T * D * 2}}, {"vcat_new", {p.vcat + 1024 * D * 2, T * D * 2}},
        {"sw0", {p.sw, GT * T * 2}}, {"p0", {p.p, GT * 2048 * 2}}, {"sums", {p.sums, KV * GT * 4}}, {"o", {p.o, KV * GT * D * 4}},
        {"gated", {p.gated, T * NH * D * 2}}, {"a1", {p.a1, T * H * 2}}, {"res", {p.res, T * H * 4}}, {"xn", {p.xn, T * H * 2}},
        {"logits", {p.logits, T * E * 4}}, {"idx", {p.idx, S * 4}}, {"rw", {p.rw, S * 4}}, {"code", {p.code, S * 4}}};
    {
        alignas(16) int32_t cnt[128];
        rt_->download_sync(p.cnt, cnt, sizeof(cnt));
        uint64_t rows = 0;
        for (uint32_t e = 0; e < 128; ++e) rows += dyn_alloc_rows(uint32_t(cnt[e]));
        if (rows) {
            items.push_back({"xq_bf16", {p.xq, rows * H * 2}});
            items.push_back({"aq_bf16", {p.aq, rows * F * 2}});
            items.push_back({"y_f32", {p.gu, rows * H * 4}});
        }
    }
    items.push_back({"mo", {p.mo, T * H * 2}});
    for (const auto& it : items)
        std::fprintf(stderr, "[hash] app %u %-8s %016llx\n", app, it.first, (unsigned long long)fnv(it.second.first, it.second.second));
    for (uint64_t h = 0; h < KV; ++h) {
        std::fprintf(stderr, "[hash] app %u head%llu: vring-copy %016llx vnew %016llx P %016llx o %016llx\n", app, (unsigned long long)h,
                     (unsigned long long)fnv(p.vcat + h * 2048 * D * 2, 1024 * D * 2),
                     (unsigned long long)fnv(p.vcat + (h * 2048 + 1024) * D * 2, 1024 * D * 2),
                     (unsigned long long)fnv(p.p + h * GT * 2048 * 2, GT * 2048 * 2), (unsigned long long)fnv(p.o + h * GT * D * 4, GT * D * 4));
    }
    {   // non-finite / non-zero values: the V ring copy, P's ring columns of head 0, o
        HostBuffer hb(m_.dev_, KV * GT * D * 4 + 64);
        m_.stream_.download(pscratch_[0], p.o - base, hb, 0, KV * GT * D * 4);
        m_.stream_.synchronize();
        size_t nf = 0;
        for (uint64_t i = 0; i < KV * GT * D; ++i) nf += !std::isfinite(hb.as<float>()[i]);
        HostBuffer hv(m_.dev_, KV * 2048 * D * 2 + 64);
        m_.stream_.download(pscratch_[0], p.vcat - base, hv, 0, KV * 2048 * D * 2);
        m_.stream_.synchronize();
        size_t vnz = 0, vnf = 0;
        for (uint64_t h = 0; h < KV; ++h)
            for (uint64_t i = 0; i < 1024 * D; ++i) {
                const uint16_t v = hv.as<uint16_t>()[h * 2048 * D + i];
                vnz += (v & 0x7fff) != 0;
                vnf += (v & 0x7f80) == 0x7f80;
            }
        HostBuffer hp(m_.dev_, GT * 2048 * 2 + 64);
        m_.stream_.download(pscratch_[0], p.p - base, hp, 0, GT * 2048 * 2);
        m_.stream_.synchronize();
        size_t pnz = 0;
        for (uint64_t r = 0; r < GT; ++r)
            for (uint64_t i = 0; i < 1024; ++i) pnz += (hp.as<uint16_t>()[r * 2048 + i] & 0x7fff) != 0;
        std::fprintf(stderr, "[hash] app %u: o non-finite %zu; V ring copy non-zero %zu non-finite %zu; P head 0 ring columns non-zero %zu\n",
                     app, nf, vnz, vnf, pnz);
        // the copied new V rows against the conv's rows (kvs): rows that differ, per head
        HostBuffer hk(m_.dev_, KV * T * D * 2 + 64);
        m_.stream_.download(pscratch_[0], p.kvs + KV * T * D * 2 - base, hk, 0, KV * T * D * 2);
        m_.stream_.synchronize();
        for (uint64_t h = 0; h < KV; ++h) {
            size_t bad = 0, first = 0, last = 0;
            for (uint64_t t = 0; t < T; ++t)
                if (std::memcmp(hv.as<uint8_t>() + (h * 2048 + 1024 + t) * D * 2, hk.as<uint8_t>() + (h * T + t) * D * 2, D * 2) != 0) {
                    if (!bad) first = t;
                    last = t;
                    ++bad;
                }
            if (bad) std::fprintf(stderr, "[hash] app %u head %llu: %zu copied V rows differ from the conv output (rows %zu .. %zu)\n", app,
                                  (unsigned long long)h, bad, first, last);
        }
    }
}

// (debug) hashes of every region of slot 0's scratch (field to the next field), one line per region.
void InkDirect::hash_fields(uint32_t app) {
    static const char* const names[] = {"xa", "row", "q", "r16", "bt", "kvs", "vcat", "sr", "sw", "p", "pw", "sums", "o", "of", "gated", "a1",
        "res", "xn", "mixd", "logits", "idx", "rw", "hist", "off", "cnt", "exp_base", "code", "xil", "msg", "cat", "gr",
        "fgv", "zero", "lbuf", "lcnt", "xq", "gu", "aq", "mo"};
    constexpr size_t NF = sizeof(names) / sizeof(names[0]);
    static_assert(sizeof(PScratch) == NF * sizeof(uint64_t));
    const uint64_t* f = &ps_[0].xa;
    const uint64_t base = pscratch_[0].address(), total = pscratch_[0].size();
    std::vector<std::pair<uint64_t, size_t>> ord;
    for (size_t i = 0; i < NF; ++i) ord.push_back({f[i] - base, i});
    std::sort(ord.begin(), ord.end());
    static std::unique_ptr<HostBuffer> hb;
    constexpr uint64_t kStep = 64ull << 20;
    if (!hb) hb = std::make_unique<HostBuffer>(m_.dev_, kStep);
    for (size_t k = 0; k < ord.size(); ++k) {
        const uint64_t a0 = ord[k].first, a1 = k + 1 < ord.size() ? ord[k + 1].first : total;
        uint64_t h = 1469598103934665603ull;
        for (uint64_t o = a0; o < a1; o += kStep) {
            const uint64_t nb = std::min(kStep, a1 - o);
            m_.stream_.download(pscratch_[0], o, *hb, 0, nb);
            m_.stream_.synchronize();
            const uint64_t* d = hb->as<uint64_t>();
            for (uint64_t i = 0; i < nb / 8; ++i) h = ((h ^ d[i]) * 0x9e3779b97f4a7c15ull) ^ (h >> 29);
        }
        std::fprintf(stderr, "[fhash] app %u %-6s %016llx\n", app, names[ord[k].second], (unsigned long long)h);
    }
}

// (debug) the K / V conv outputs of the application that just ran in slot 0: a 16-bit hash per (K / V, head, the
// conv's 24 row chunks), and the rows of the P . V buffer's new V part that differ from the KV scratch's.
void InkDirect::hash_kvs(uint32_t app) {
    const auto& c = m_.cfg_;
    const uint64_t T = kT, KV = c.kv_heads, D = c.head_dim, W = c.window;
    const auto& p = ps_[0];
    const uint64_t base = pscratch_[0].address();
    HostBuffer hk(m_.dev_, 2 * KV * T * D * 2 + 64), hv(m_.dev_, KV * (W + T) * D * 2 + 64);
    m_.stream_.download(pscratch_[0], p.kvs - base, hk, 0, 2 * KV * T * D * 2);
    m_.stream_.download(pscratch_[0], p.vcat - base, hv, 0, KV * (W + T) * D * 2);
    m_.stream_.synchronize();
    for (uint64_t kv = 0; kv < 2; ++kv)
        for (uint64_t h = 0; h < KV; ++h) {
            std::string line;
            for (uint64_t j = 0; j < 24; ++j) {
                const uint64_t t0 = T * j / 24, t1 = T * (j + 1) / 24;
                uint64_t x = 1469598103934665603ull;
                const uint16_t* d = hk.as<uint16_t>() + ((kv * KV + h) * T + t0) * D;
                for (uint64_t i = 0; i < (t1 - t0) * D; ++i) x = (x ^ d[i]) * 1099511628211ull;
                char b[8];
                std::snprintf(b, sizeof b, " %04x", unsigned((x ^ (x >> 16) ^ (x >> 32) ^ (x >> 48)) & 0xffff));
                line += b;
            }
            std::fprintf(stderr, "[kvs] app %u %c head %llu:%s\n", app, kv ? 'V' : 'K', (unsigned long long)h, line.c_str());
        }
    for (uint64_t h = 0; h < KV; ++h) {
        size_t bad = 0, first = 0, last = 0;
        for (uint64_t t = 0; t < T; ++t)
            if (std::memcmp(hv.as<uint8_t>() + (h * (W + T) + W + t) * D * 2, hk.as<uint8_t>() + ((KV + h) * T + t) * D * 2, D * 2) != 0) {
                if (!bad) first = t;
                last = t;
                ++bad;
            }
        if (bad) std::fprintf(stderr, "[kvs] app %u head %llu: %zu new V rows of vcat differ from kvs (rows %zu .. %zu)\n", app,
                              (unsigned long long)h, bad, first, last);
    }
}

// The waits of a plan against the engines' fences. Every waiting stage of an engine uses the same fence counter: a
// monitor adds one when its SOB reaches the target, a waiting stage takes one. So the stage at the head of an
// engine's queue is released by whichever pending signal fires first: the waits are right only if, along each
// engine's queue, the signals fire in the order of the stages that wait on them. This derives the "not earlier
// than" relation between the events of a plan (stage starts, part ends, signal fires) and reports every pair of
// consecutive waiting stages X, Y of an engine whose signals are not ordered by it (Y's signal could fire while the
// engine sits at X's fence: X would start before its own inputs are complete). Returns the number of such pairs.
size_t InkDirect::check_waits(uint32_t app, const std::vector<PStage>& meta, bool verbose) {
    const size_t n = meta.size();
    std::vector<size_t> s_node(n), e_first(n);
    size_t nn = 0;
    for (size_t i = 0; i < n; ++i) {
        s_node[i] = nn++;
        e_first[i] = nn;
        nn += meta[i].parts.size();
    }
    std::map<std::pair<int, uint32_t>, size_t> f_node;
    for (size_t i = 0; i < n; ++i)
        if (meta[i].wait >= 0 && !f_node.count({meta[i].wait, meta[i].wtgt})) f_node[{meta[i].wait, meta[i].wtgt}] = nn++;
    std::vector<std::vector<size_t>> adj(nn);
    size_t last[2] = {SIZE_MAX, SIZE_MAX};
    for (size_t i = 0; i < n; ++i) {
        const auto& st = meta[i];
        size_t prev = s_node[i];
        for (size_t k = 0; k < st.parts.size(); ++k) {
            adj[prev].push_back(e_first[i] + k);
            prev = e_first[i] + k;
        }
        if (last[st.eng] != SIZE_MAX) adj[last[st.eng]].push_back(s_node[i]);
        last[st.eng] = prev;
        if (st.wait >= 0) adj[f_node.at({st.wait, st.wtgt})].push_back(s_node[i]);
    }
    size_t bad = 0;
    std::vector<std::vector<size_t>> ub(nn);  // per fire: the part ends it is not later than the last of
    for (const auto& [key, fn] : f_node) {  // a fire is not earlier than the parts that bring the count to the target
        uint32_t cum = 0;
        for (size_t i = 0; i < n && cum < key.second; ++i)
            for (size_t k = 0; k < meta[i].parts.size() && cum < key.second; ++k) {
                if (meta[i].parts[k].second != key.first) continue;
                cum += meta[i].cnt[k];
                // (a target reached inside a part: only the part's start bounds the fire)
                adj[cum > key.second ? (k ? e_first[i] + k - 1 : s_node[i]) : e_first[i] + k].push_back(fn);
                ub[fn].push_back(e_first[i] + k);
            }
        if (cum < key.second) {
            ++bad;
            std::fprintf(stderr, "[waits] app %u: SOB %d target %u but only %u signals\n", app, key.first, key.second, cum);
        }
        for (const auto& [k2, f2] : f_node)  // (the same SOB's larger targets fire later)
            if (k2.first == key.first && k2.second > key.second) adj[fn].push_back(f2);
    }
    auto reach = [&](size_t from, size_t to) {
        if (from == to) return true;
        std::vector<uint8_t> seen(nn, 0);
        std::vector<size_t> q{from};
        seen[from] = 1;
        while (!q.empty()) {
            const size_t u = q.back();
            q.pop_back();
            for (size_t v : adj[u]) {
                if (v == to) return true;
                if (!seen[v]) { seen[v] = 1; q.push_back(v); }
            }
        }
        return false;
    };
    for (uint8_t eng = 0; eng < 2; ++eng) {
        std::vector<size_t> w;
        for (size_t i = 0; i < n; ++i)
            if (meta[i].eng == eng && meta[i].wait >= 0) w.push_back(i);
        for (size_t a = 0; a < w.size(); ++a)
            for (size_t b = a + 1; b < w.size(); ++b) {
                const auto& x = meta[w[a]];
                const auto& y = meta[w[b]];
                const size_t fx = f_node.at({x.wait, x.wtgt}), fy = f_node.at({y.wait, y.wtgt});
                // (ordered: a path from X's fire, or from every part X's fire is not later than the last of)
                if (reach(fx, fy) || std::all_of(ub[fx].begin(), ub[fx].end(), [&](size_t u) { return reach(u, fy); })) continue;
                ++bad;
                if (verbose)
                    std::fprintf(stderr, "[waits] app %u %s: stage %zu %s (SOB %d >= %u) can be released by the signal of stage %zu %s (SOB %d >= %u)\n",
                                 app, eng ? "TPC" : "MME", w[a], x.name.c_str(), x.wait, x.wtgt, w[b], y.name.c_str(), y.wait, y.wtgt);
            }
    }
    return bad;
}

// (debug) the routing plan of the application that just ran in slot 0, recomputed on the host from the device's
// top-K experts: histogram, expert counts, flat bases, slot codes, and wave lists. Returns the number of mismatches.
size_t InkDirect::verify_rplan(uint32_t app) {
    const auto& c = m_.cfg_;
    const uint32_t T = kT, K = c.topk, S = T * K, E = c.experts;
    const auto& p = ps_[0];
    const uint64_t base = pscratch_[0].address();
    auto get = [&](uint64_t addr, uint64_t bytes) {
        HostBuffer hb(m_.dev_, bytes + 64);
        m_.stream_.download(pscratch_[0], addr - base, hb, 0, bytes);
        m_.stream_.synchronize();
        return std::vector<int32_t>(hb.as<int32_t>(), hb.as<int32_t>() + bytes / 4);
    };
    const auto idx = get(p.idx, S * 4), code = get(p.code, S * 4), lbuf = get(p.lbuf, 4 * S * 4);
    const auto lc = get(p.lcnt, 96 * 4), dcnt = get(p.cnt, 128 * 4), dbase = get(p.exp_base, 128 * 4), hist = get(p.hist, 24 * 128 * 4);
    std::vector<int> tp(25);
    if (ptab_host_) {
        const auto* htp = reinterpret_cast<const int32_t*>(ptab_host_->as<uint8_t>() + kTabTpart);
        for (uint32_t i = 0; i <= 24; ++i) tp[i] = int(htp[i]);
    } else {
        for (uint32_t i = 0; i <= 24; ++i) tp[i] = int(T * i / 24);
    }
    const uint32_t active_S = uint32_t(tp[24]) * K;
    std::vector<int> cnt(E, 0), eb(E, 0);
    for (uint32_t s2 = 0; s2 < active_S; ++s2) cnt[idx[s2]]++;
    int row_cur = 0;
    for (uint32_t e = 0; e < E; ++e) {
        eb[e] = row_cur;
        row_cur += int(dyn_alloc_rows(uint32_t(cnt[e])));
    }
    if (std::getenv("G2E_INK_LOADS")) {
        std::vector<int> sc(cnt.begin(), cnt.end());
        std::sort(sc.rbegin(), sc.rend());
        std::fprintf(stderr, "[loads] app %u: top %d %d %d %d | rank 8 %d 16 %d 32 %d 64 %d | active %d | total_rows %d\n",
                     app, sc[0], sc[1], sc[2], sc[3], sc[8], sc[16], sc[32], sc[64],
                     int(std::count_if(sc.begin(), sc.end(), [](int v) { return v > 0; })), row_cur);
    }
    size_t bad_cnt = 0, bad_eb = 0, bad_c = 0, bad_l = 0, bad_hist = 0, first = S;
    for (uint32_t e = 0; e < E; ++e) {
        bad_cnt += dcnt[e] != cnt[e];
        bad_eb += dbase[e] != eb[e];
    }
    std::vector<int> run(E, 0);
    for (int pc = 0; pc < 24; ++pc) {
        std::vector<int> hh(E, 0);
        for (int s2 = tp[pc] * int(K); s2 < tp[pc + 1] * int(K); ++s2) hh[idx[s2]]++;
        for (uint32_t e = 0; e < E; ++e) bad_hist += hist[pc * 128 + e] != hh[e];
        int nw[4] = {0, 0, 0, 0};
        for (int s2 = tp[pc] * int(K); s2 < tp[pc + 1] * int(K); ++s2) {
            const int e = idx[s2], r = run[e]++;
            const int cv = eb[e] + r;
            if (code[s2] != cv) { ++bad_c; first = std::min<size_t>(first, s2); }
            const int w = e >> 5;
            bad_l += lbuf[w * int(S) + tp[pc] * int(K) + nw[w]++] != s2;
        }
        for (int w = 0; w < 4; ++w) bad_l += lc[w * 24 + pc] != nw[w];
    }
    const size_t bad = bad_cnt + bad_eb + bad_c + bad_l + bad_hist;
    if (bad) std::fprintf(stderr, "[rplan] app %u: MISMATCH hist %zu cnt %zu exp_base %zu code %zu (first slot %zu = token %zu) lists %zu\n",
                          app, bad_hist, bad_cnt, bad_eb, bad_c, first, first / K, bad_l);
    return bad;
}

// G2E_INK_DPREFILL_PROF=<app>: Plan A and Plan B cut after each stage (median of 9 runs; the inputs of the last forward)
void InkDirect::profile_stages_p(uint32_t app) {
    for (auto& [key, pl] : plans_p_) {
        if (std::get<0>(key) != app) continue;
        const int bits = std::get<1>(key);
        const uint64_t oaddr = std::get<2>(key);
        const int sl = bits & 1, hc = (bits >> 1) & 1, ci = (bits >> 2) & 1, co = (bits >> 3) & 1;
        const uint32_t ppos = m_.pos_ >= kT ? m_.pos_ - kT : 0u;
        auto mk = [&](int part, int lim) {
            return part == 0 ? build_pa(app, sl, hc, ci, co, ppos, lim, nullptr)
                             : build_pb(app, sl, oaddr, ci, co, lim, nullptr);
        };
        auto prep_b = [&] {
            // Repopulate slot scratch, pdyn_[sl], and staging for `app` (avoid stale state from app 47).
            pl.plan_a->run();
            if (pl.plan_a2) run_pfar_loop(app, sl, ppos, pl);
            int32_t cnt[128];
            patch_b(app, sl, pl, cnt);
            ensure_bf16_staged(m_.apps_[app].layer, ps_[sl].cnt, cnt);
        };
        if (std::getenv("G2E_INK_DPREFILL_PROF_ALONE")) {  // every call alone (minus the smallest plan's time)
            double base = 0, sum[2] = {0, 0};
            {
                auto pp = std::make_unique<DirectPlan>(*rt_);
                const DirectTpcTensor t[] = {tn(ps_[0].mixd, DType::BF16, {128, 32, 1}), tn(ps_[0].mixd + 65536, DType::BF16, {128, 32, 1})};
                pp->add_tpc_grid(isa_.at("ink_pcopy_gaudi2"), t, {}, {1, 1, 1, 1, 1}, 1, 24, -1, 0, -1);
                pp->prepare();
                std::vector<double> ts;
                for (int r = 0; r < 9; ++r) ts.push_back(pp->run());
                std::sort(ts.begin(), ts.end());
                base = ts[4];
                std::fprintf(stderr, "[ink-direct] p-app %u alone: baseline %.1f us\n", app, base);
            }
            for (int part = 0; part < 2; ++part) {
                if (part == 1) prep_b();
                for (int k = 1; k <= 400; ++k) {
                    auto pp = mk(part, -k);
                    if (!pp) break;
                    std::vector<double> ts;
                    for (int r = 0; r < 9; ++r) ts.push_back(pp->run());
                    std::sort(ts.begin(), ts.end());
                    const std::string nm = pnames_.back();
                    const bool is_mme = nm.rfind("p_", 0) == 0;
                    sum[is_mme ? 0 : 1] += ts[4] - base;
                    std::fprintf(stderr, "[ink-direct] p-app %u%c alone %2d %-30s %8.1f us\n", app, part ? 'B' : 'A', k, nm.c_str(), ts[4] - base);
                }
            }
            std::fprintf(stderr, "[ink-direct] p-app %u alone: MME %.0f us, TPC %.0f us\n", app, sum[0], sum[1]);
            return;
        }
        for (int part = 0; part < 2; ++part) {
            if (part == 1) prep_b();
            double prev = 0;
            for (int lim = 1; lim <= 200; ++lim) {
                auto pp = mk(part, lim);
                const std::string nm = pnames_.empty() ? "?" : pnames_.back();
                if (int(pnames_.size()) < lim) break;
                if (pp && part == 1) {  // the dynamic MoE stages of the slot's last patch (its routing is still in the scratch)
                    for (int k = 0; k < 8; ++k) pp->set_dyn_wait(k, pdyn_last_[sl].tgt[k]);
                    pp->set_opt_mask(pdyn_last_[sl].opt_mask);
                }
                if (!pp) { std::fprintf(stderr, "[ink-direct] p-app %u%c stage %2d %-28s (MME, in the next)\n", app, part ? 'B' : 'A', lim, nm.c_str()); continue; }
                std::vector<double> ts;
                for (int r = 0; r < 9; ++r) ts.push_back(pp->run());
                std::sort(ts.begin(), ts.end());
                std::fprintf(stderr, "[ink-direct] p-app %u%c stage %2d %-28s %8.1f us (+%.1f)\n", app, part ? 'B' : 'A', lim, nm.c_str(), ts[4], ts[4] - prev);
                prev = ts[4];
            }
        }
        return;
    }
}

}  // namespace g2e
