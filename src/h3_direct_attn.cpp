#include "g2e/h3_direct_attn.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace g2e {

namespace {

constexpr uint32_t kD = 128;
constexpr uint32_t kTpcs = 24;
constexpr uint32_t kMaxTilesPerPlan = 32;  // <= 34 TPC stages (< 38 TPC monitor limit)

uint64_t align256(uint64_t x) { return (x + 255ull) & ~255ull; }

// Patch tag namespaces inside each DirectPlan:
//   10 + q_local:             Q base address for local Q-slot q_local (MME QK A operand)
//   100 + q_local*64 + kt:    K tile (q_local, kt) (MME QK B operand)
//   5000 + q_local*64 + kt:   V tile (q_local, kt) (MME PV B operand)
//   1, 2, 3:                  att_out, score_scale, d_scale_v (only patched if HBM buffer addr changes)
constexpr int kTagAttOut = 1;
constexpr int kTagScoreScale = 2;
constexpr int kTagDScaleV = 3;
int tag_q(uint32_t ql) { return 10 + int(ql); }
int tag_k(uint32_t ql, uint32_t kt) { return 100 + int(ql) * 64 + int(kt); }
int tag_v(uint32_t ql, uint32_t kt) { return 5000 + int(ql) * 64 + int(kt); }

}  // namespace

DirectH3AttnAssets DirectH3AttnAssets::load(DirectRuntime& rt, const std::string& desc_dir,
                                            const std::string& tpc_obj_dir) {
    DirectH3AttnAssets a;
    a.qk_1536_4096   = MmeTemplate::load_dir(desc_dir + "/qk_f8_1536_4096");
    a.pv_1536_4096   = MmeTemplate::load_dir(desc_dir + "/pv_f8_1536_4096");
    a.psum_1536_4096 = MmeTemplate::load_dir(desc_dir + "/psum_f8_1536_4096");
    a.qk_1536_2048   = MmeTemplate::load_dir(desc_dir + "/qk_f8_1536_2048");
    a.pv_1536_2048   = MmeTemplate::load_dir(desc_dir + "/pv_f8_1536_2048");
    a.psum_1536_2048 = MmeTemplate::load_dir(desc_dir + "/psum_f8_1536_2048");
    a.qk_1536_1024   = MmeTemplate::load_dir(desc_dir + "/qk_f8_1536_1024");
    a.pv_1536_1024   = MmeTemplate::load_dir(desc_dir + "/pv_f8_1536_1024");
    a.psum_1536_1024 = MmeTemplate::load_dir(desc_dir + "/psum_f8_1536_1024");
    a.qk_1536_512    = MmeTemplate::load_dir(desc_dir + "/qk_f8_1536_512");
    a.pv_1536_512    = MmeTemplate::load_dir(desc_dir + "/pv_f8_1536_512");
    a.psum_1536_512  = MmeTemplate::load_dir(desc_dir + "/psum_f8_1536_512");
    a.qk_1536_256    = MmeTemplate::load_dir(desc_dir + "/qk_f8_1536_256");
    a.pv_1536_256    = MmeTemplate::load_dir(desc_dir + "/pv_f8_1536_256");
    a.psum_1536_256  = MmeTemplate::load_dir(desc_dir + "/psum_f8_1536_256");

    a.qk_256_512     = MmeTemplate::load_dir(desc_dir + "/qk_f8_256_512");
    a.pv_256_512     = MmeTemplate::load_dir(desc_dir + "/pv_f8_256_512");
    a.psum_256_512   = MmeTemplate::load_dir(desc_dir + "/psum_f8_256_512");
    a.qk_256_256     = MmeTemplate::load_dir(desc_dir + "/qk_f8_256_256");
    a.pv_256_256     = MmeTemplate::load_dir(desc_dir + "/pv_f8_256_256");
    a.psum_256_256   = MmeTemplate::load_dir(desc_dir + "/psum_f8_256_256");

    if (std::filesystem::exists(desc_dir + "/pv256_f8_1536_4096")) {
        a.pv256_1536_4096 = MmeTemplate::load_dir(desc_dir + "/pv256_f8_1536_4096");
        a.pv256_1536_2048 = MmeTemplate::load_dir(desc_dir + "/pv256_f8_1536_2048");
        a.pv256_1536_1024 = MmeTemplate::load_dir(desc_dir + "/pv256_f8_1536_1024");
        a.pv256_1536_512  = MmeTemplate::load_dir(desc_dir + "/pv256_f8_1536_512");
        a.pv256_1536_256  = MmeTemplate::load_dir(desc_dir + "/pv256_f8_1536_256");
        a.pv256_256_512   = MmeTemplate::load_dir(desc_dir + "/pv256_f8_256_512");
        a.pv256_256_256   = MmeTemplate::load_dir(desc_dir + "/pv256_f8_256_256");
        a.has_pv256 = true;
    }

    a.isa_step_fp8 = rt.load_tpc_elf(tpc_obj_dir + "/h3_direct_attn_step_fp8_gaudi2.o");
    return a;
}

DirectH3Attn::DirectH3Attn(const Device& dev, DirectRuntime& rt, const DirectH3AttnAssets& assets,
                           uint32_t num_heads, uint32_t valid_u, const DirectH3AttnArgs& init_args,
                           uint64_t sram_offset, uint32_t tpc_cmd, uint32_t out_rows,
                           uint32_t kv_stride_rows, uint32_t sparse_keep_full, bool use_pv256)
    : rt_(rt), nh_(num_heads), valid_u_(valid_u), bq_(choose_bq(valid_u)),
      u_pad_(aligned_u_pad(valid_u)), k_pad_(aligned_k_pad(valid_u)),
      kv_stride_rows_(kv_stride_rows >= k_pad_ ? kv_stride_rows : k_pad_),
      q_blocks_per_head_(u_pad_ / bq_), n_full_tiles_(valid_u / 4096u),
      use_pv256_(use_pv256 && assets.has_pv256), bound_(init_args) {
    if (!nh_ || !valid_u_) throw std::invalid_argument("DirectH3Attn: invalid dims");
    const uint32_t att_out_s = (out_rows >= valid_u_) ? out_rows : u_pad_;
    const uint32_t kDv = use_pv256_ ? 256u : kD;

    // 1. K-tile decomposition:
    //    - Dense (sparse_keep_full == 0): greedy decomposition of full k_pad (always a multiple of 256).
    //    - Dynamic Block-Sparse (Route B, BQ=1536): `sparse_keep_full` dynamic 4096-wide full-tile slots
    //      (col_start=0 -> visible==4096 since every selected tile index < valid_u/4096) followed by the
    //      exact tail tiles covering [n_full_tiles*4096 .. k_pad) for 100% exact boundary masking!
    const bool small = (bq_ == 256u);
    if (!small && sparse_keep_full > 0 && sparse_keep_full < n_full_tiles_) {
        sparse_keep_full_ = sparse_keep_full;
        for (uint32_t s = 0; s < sparse_keep_full_; ++s) {
            k_tiles_.push_back({0u, 4096u});
        }
        uint32_t col = n_full_tiles_ * 4096u;
        while (col < k_pad_) {
            const uint32_t rem = k_pad_ - col;
            uint32_t w = 256u;
            if (rem >= 4096u) w = 4096u;
            else if (rem >= 2048u) w = 2048u;
            else if (rem >= 1024u) w = 1024u;
            else if (rem >= 512u) w = 512u;
            k_tiles_.push_back({col, w});
            col += w;
        }
    } else {
        uint32_t col = 0;
        while (col < k_pad_) {
            const uint32_t rem = k_pad_ - col;
            uint32_t w = 256u;
            if (small) {
                w = (rem >= 512u) ? 512u : 256u;
            } else {
                if (rem >= 4096u) w = 4096u;
                else if (rem >= 2048u) w = 2048u;
                else if (rem >= 1024u) w = 1024u;
                else if (rem >= 512u) w = 512u;
                else w = 256u;
            }
            k_tiles_.push_back({col, w});
            col += w;
        }
    }
    const uint32_t n_kt = uint32_t(k_tiles_.size());
    const uint32_t max_tile_w = k_tiles_.front().width;

    // 2. Plan packing: pack multiple Q-blocks per plan when n_kt <= kMaxTilesPerPlan (eliminates inter-Q bubbles!),
    //    or split 1 Q-block across sub-plans when n_kt > kMaxTilesPerPlan (sequences > 131k tokens).
    if (const char* e = std::getenv("G2E_H3_TPC_CMD")) {
        tpc_cmd = uint32_t(std::stoul(e, nullptr, 0));
    }
    const uint32_t total_q_blocks = nh_ * q_blocks_per_head_;
    uint32_t max_plans = 12u;  // 2 banks x 12 plans -> ~90% SRAM roofline (172 ms/L on U=37774)
    if (const char* e = std::getenv("G2E_H3_DIRECT_PLANS")) {
        uint32_t v = uint32_t(std::stoul(e));
        if (v >= 1 && v <= 64) max_plans = v;
    }
    if (n_kt <= kMaxTilesPerPlan) {
        sub_plans_per_q_ = 1u;
        q_per_plan_ = std::max(1u, std::min({kMaxTilesPerPlan / n_kt, 32u, total_q_blocks}));
        const uint32_t need_plans = (total_q_blocks + q_per_plan_ - 1u) / q_per_plan_;
        plans_per_batch_ = std::min(need_plans, max_plans);
        num_banks_ = (need_plans > plans_per_batch_ && rt.chain_enabled()) ? 2u : 1u;
    } else {
        q_per_plan_ = 1u;
        sub_plans_per_q_ = (n_kt + kMaxTilesPerPlan - 1u) / kMaxTilesPerPlan;
        const uint32_t max_q_batch = std::max(1u, max_plans / sub_plans_per_q_);
        plans_per_batch_ = std::min(total_q_blocks, max_q_batch) * sub_plans_per_q_;
        num_banks_ = (total_q_blocks > max_q_batch && rt.chain_enabled()) ? 2u : 1u;
    }
    const uint32_t slots_per_batch = (sub_plans_per_q_ == 1u)
        ? (plans_per_batch_ * q_per_plan_)
        : (plans_per_batch_ / sub_plans_per_q_);

    // Double-buffered control tables in HBM: I32 [slots_per_batch, 4] = {head_idx, q_base, 0, 0}
    for (uint32_t bk = 0; bk < num_banks_; ++bk) {
        d_ctrl_[bk] = DeviceBuffer(dev, std::max<uint64_t>(256ull, uint64_t(slots_per_batch) * 16ull));
    }
    // Constant FP8 E4M3 ones vector [4096, 1] (0x38 == 1.0f) for legacy MME P@1 fallback
    if (!use_pv256_) {
        d_ones_ = DeviceBuffer(dev, 4096ull);
        Stream st_init(dev);
        HostBuffer h_ones(dev, 4096ull);
        std::memset(h_ones.data(), 0x38, 4096ull);
        st_init.upload(h_ones, 0, d_ones_, 0, 4096ull);
        st_init.synchronize();
    }

    // 3. Allocate SRAM arena (double-buffered S[2], P[2], PV[2] [256 or 128], l_tile[2] + 1x O_acc/l_acc + 4x m/res -> 39.8 MiB!)
    uint64_t sp = align256(rt.sram_base() + sram_offset);
    auto alloc = [&](uint64_t bytes) -> uint64_t {
        const uint64_t addr = sp;
        sp = align256(sp + bytes);
        return addr;
    };
    const uint64_t s_S[2] = {
        alloc(uint64_t(bq_) * max_tile_w * 2ull),
        alloc(uint64_t(bq_) * max_tile_w * 2ull),
    };
    const uint64_t s_P[2] = {
        alloc(uint64_t(bq_) * max_tile_w * 1ull),
        alloc(uint64_t(bq_) * max_tile_w * 1ull),
    };
    const uint64_t s_PV[2] = {
        alloc(uint64_t(bq_) * kDv * 4ull),
        alloc(uint64_t(bq_) * kDv * 4ull),
    };
    const uint64_t s_l_tile[2] = {
        alloc(uint64_t(bq_) * 4ull),
        alloc(uint64_t(bq_) * 4ull),
    };
    const uint64_t s_O = alloc(uint64_t(bq_) * kD * 4ull);
    const uint64_t s_l_acc = alloc(uint64_t(bq_) * 4ull);
    uint64_t s_m[4], s_res[4];
    for (uint32_t i = 0; i < 4; ++i) s_m[i]   = alloc(uint64_t(bq_) * 4ull);
    for (uint32_t i = 0; i < 4; ++i) s_res[i] = alloc(uint64_t(bq_) * 4ull);

    sram_used_ = sp - (rt.sram_base() + sram_offset);
    if (sp > rt.sram_base() + rt.sram_size()) {
        throw std::runtime_error("DirectH3Attn: SRAM overflow (need " +
                                 std::to_string(sram_used_ / (1024 * 1024)) + " MiB, avail " +
                                 std::to_string(rt.sram_size() / (1024 * 1024)) + " MiB)");
    }

    auto qk_tmpl = [&](uint32_t w) -> const MmeTemplate& {
        if (small) return w == 512u ? assets.qk_256_512 : assets.qk_256_256;
        if (w == 4096u) return assets.qk_1536_4096;
        if (w == 2048u) return assets.qk_1536_2048;
        if (w == 1024u) return assets.qk_1536_1024;
        if (w == 512u)  return assets.qk_1536_512;
        return assets.qk_1536_256;
    };
    auto pv_tmpl = [&](uint32_t w) -> const MmeTemplate& {
        if (use_pv256_) {
            if (small) return w == 512u ? assets.pv256_256_512 : assets.pv256_256_256;
            if (w == 4096u) return assets.pv256_1536_4096;
            if (w == 2048u) return assets.pv256_1536_2048;
            if (w == 1024u) return assets.pv256_1536_1024;
            if (w == 512u)  return assets.pv256_1536_512;
            return assets.pv256_1536_256;
        }
        if (small) return w == 512u ? assets.pv_256_512 : assets.pv_256_256;
        if (w == 4096u) return assets.pv_1536_4096;
        if (w == 2048u) return assets.pv_1536_2048;
        if (w == 1024u) return assets.pv_1536_1024;
        if (w == 512u)  return assets.pv_1536_512;
        return assets.pv_1536_256;
    };
    auto psum_tmpl = [&](uint32_t w) -> const MmeTemplate& {
        if (small) return w == 512u ? assets.psum_256_512 : assets.psum_256_256;
        if (w == 4096u) return assets.psum_1536_4096;
        if (w == 2048u) return assets.psum_1536_2048;
        if (w == 1024u) return assets.psum_1536_1024;
        if (w == 512u)  return assets.psum_1536_512;
        return assets.psum_1536_256;
    };

    // 4. Build all reusable DirectPlans for one batch.
    struct StepTile {
        uint32_t q_local = 0;     // 0..q_per_plan_-1 within this plan
        uint32_t ctrl_slot = 0;   // row in d_ctrl_
        uint32_t kt_idx = 0;      // index into k_tiles_
        bool is_first = false;
        bool is_last = false;
    };

    for (uint32_t bk = 0; bk < num_banks_; ++bk) {
        plans_[bk].reserve(plans_per_batch_);
        for (uint32_t pi = 0; pi < plans_per_batch_; ++pi) {
            std::vector<StepTile> stiles;
            if (sub_plans_per_q_ == 1u) {
                stiles.reserve(q_per_plan_ * n_kt);
                for (uint32_t ql = 0; ql < q_per_plan_; ++ql) {
                    const uint32_t cslot = pi * q_per_plan_ + ql;
                    for (uint32_t kt = 0; kt < n_kt; ++kt) {
                        stiles.push_back({ql, cslot, kt, kt == 0, kt + 1 == n_kt});
                    }
                }
            } else {
                const uint32_t cslot = pi / sub_plans_per_q_;
                const uint32_t sub = pi % sub_plans_per_q_;
                const uint32_t kt0 = sub * kMaxTilesPerPlan;
                const uint32_t kt1 = std::min(n_kt, kt0 + kMaxTilesPerPlan);
                stiles.reserve(kt1 - kt0);
                for (uint32_t kt = kt0; kt < kt1; ++kt) {
                    stiles.push_back({0u, cslot, kt, kt == 0, kt + 1 == n_kt});
                }
            }

            const uint32_t count = uint32_t(stiles.size());
            auto plan = std::make_unique<DirectPlan>(rt, 0);
            uint32_t s_mme[2] = {0, 0};
            uint32_t s_tpc[2] = {0, 0};

            auto emit_qk = [&](uint32_t k, uint32_t parity, int wait_sob, uint32_t wait_val) {
                const auto& st = stiles[k];
                const auto& kt = k_tiles_[st.kt_idx];
                const auto& tmpl = qk_tmpl(kt.width);
                const uint64_t k_addr = init_args.k_addr + uint64_t(kt.col_start) * kD;
                plan->add_mme(tmpl, init_args.q_addr, k_addr, s_S[parity], 0,
                              wait_sob, wait_val, int(parity),
                              {tag_q(st.q_local), tag_k(st.q_local, st.kt_idx), -1, -1});
                s_mme[parity] += tmpl.signal_count;
            };

            auto emit_pv = [&](uint32_t k, uint32_t parity, int wait_sob, uint32_t wait_val) {
                const auto& st = stiles[k];
                const auto& kt = k_tiles_[st.kt_idx];
                const auto& tmpl = pv_tmpl(kt.width);
                const uint64_t v_addr = init_args.v_addr + uint64_t(kt.col_start) * kDv;
                plan->add_mme(tmpl, s_P[parity], v_addr, s_PV[parity], 0,
                              wait_sob, wait_val, int(parity),
                              {-1, tag_v(st.q_local, st.kt_idx), -1, -1});
                s_mme[parity] += tmpl.signal_count;
            };

            auto emit_psum = [&](uint32_t k, uint32_t parity, int wait_sob, uint32_t wait_val) {
                if (use_pv256_) return;  // Fused into pv256 col 128 -> zero extra MME & 0 SRAM P reread!
                const auto& st = stiles[k];
                const auto& kt = k_tiles_[st.kt_idx];
                const auto& tmpl = psum_tmpl(kt.width);
                plan->add_mme(tmpl, s_P[parity], d_ones_.address(), s_l_tile[parity], 0,
                              wait_sob, wait_val, int(parity),
                              {-1, -1, -1, -1});
                s_mme[parity] += tmpl.signal_count;
            };

            auto emit_tpc_step = [&](int sm_k, int acc_k, uint32_t parity,
                                     int wait_sob, uint32_t wait_val, int sig_sob) {
                const bool do_sm = sm_k >= 0;
                const bool do_acc = acc_k >= 0;
                const auto& sm_st = stiles[do_sm ? uint32_t(sm_k) : 0u];
                const auto& acc_st = stiles[do_acc ? uint32_t(acc_k) : 0u];
                const uint32_t tile_w = do_sm ? k_tiles_[sm_st.kt_idx].width : 256u;
                const uint32_t col_start = do_sm ? k_tiles_[sm_st.kt_idx].col_start : 0u;
                const uint32_t flags_sm = do_sm ? (1u | (sm_st.is_first ? 2u : 0u)) : 0u;
                const uint32_t flags_acc = do_acc
                    ? (1u | (acc_st.is_first ? 2u : 0u) | (acc_st.is_last ? 4u : 0u))
                    : 0u;

                const uint32_t sm_slot4 = sm_st.q_local & 3u;
                const uint32_t sm_r4 = (do_sm ? uint32_t(sm_k) : 0u) & 3u;
                const uint32_t acc_r4 = (do_acc ? uint32_t(acc_k) : 0u) & 3u;

                const DirectTpcTensor t[13] = {
                    {s_S[parity],          DType::BF16, 2, {tile_w, bq_, 1, 1, 1}},
                    {s_P[parity],          DType::F8,   2, {tile_w, bq_, 1, 1, 1}},
                    {s_m[sm_slot4],        DType::F32,  2, {1, bq_, 1, 1, 1}},
                    {s_l_tile[parity],     DType::F32,  2, {1, bq_, 1, 1, 1}},
                    {s_res[sm_r4],         DType::F32,  2, {1, bq_, 1, 1, 1}},
                    {s_PV[parity],         DType::F32,  2, {kDv, bq_, 1, 1, 1}},
                    {s_O,                  DType::F32,  2, {kD, bq_, 1, 1, 1}},
                    {s_res[acc_r4],        DType::F32,  2, {1, bq_, 1, 1, 1}},
                    {s_l_acc,              DType::F32,  2, {1, bq_, 1, 1, 1}},
                    {init_args.o_addr,     DType::BF16, 3, {kD, nh_, att_out_s, 1, 1}},
                    {init_args.sc_addr,    DType::F32,  1, {1, 1, 1, 1, 1}},
                    {init_args.dv_addr,    DType::F32,  1, {1, 1, 1, 1, 1}},
                    {d_ctrl_[bk].address(),DType::I32,  2, {4, slots_per_batch, 1, 1, 1}},
                };
                const int t_tags[13] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                        (do_acc && acc_st.is_last) ? kTagAttOut : -1,
                                        kTagScoreScale, kTagDScaleV, -1};
                const uint32_t srfs[5] = {
                    valid_u_,
                    col_start,
                    flags_sm,
                    flags_acc,
                    do_acc ? acc_st.ctrl_slot : 0u,
                };
                plan->add_tpc(assets.isa_step_fp8, t, srfs, bq_, kTpcs, wait_sob, wait_val, sig_sob, tpc_cmd, t_tags);
                if (sig_sob == 2) s_tpc[0] += kTpcs;
                else if (sig_sob == 3) s_tpc[1] += kTpcs;
            };

            if (count == 1u) {
                emit_qk(0, 0, -1, 0);
                const uint32_t qk0_done = s_mme[0];
                emit_tpc_step(0, -1, 0, 0, qk0_done, 2);
                emit_pv(0, 0, 2, s_tpc[0]);
                emit_psum(0, 0, -1, 0);
                const uint32_t pv0_done = s_mme[0];
                emit_tpc_step(-1, 0, 0, 0, pv0_done, -1);
            } else {
                std::vector<uint32_t> mme_ready(count + 2, 0);
                emit_qk(0, 0, -1, 0);
                mme_ready[0] = s_mme[0];
                emit_qk(1, 1, -1, 0);
                mme_ready[1] = s_mme[1];

                for (uint32_t k = 0; k < count; ++k) {
                    const uint32_t p = k & 1u;
                    emit_tpc_step(int(k), k >= 2 ? int(k - 2) : -1, p, int(p), mme_ready[k], int(2u + p));
                    emit_pv(k, p, int(2u + p), s_tpc[p]);
                    emit_psum(k, p, -1, 0);
                    if (k + 2 < count) emit_qk(k + 2, p, -1, 0);
                    mme_ready[k + 2] = s_mme[p];
                }
                const uint32_t pe0 = (count - 2) & 1u;
                const uint32_t pe1 = (count - 1) & 1u;
                emit_tpc_step(-1, int(count - 2), pe0, int(pe0), mme_ready[count], int(2u + pe0));
                emit_tpc_step(-1, int(count - 1), pe1, int(pe1), mme_ready[count + 1], -1);
            }

            plan->prepare();
            plans_[bk].push_back(std::move(plan));
        }
    }

    std::printf("[DirectH3Attn] prepared NH=%u U=%u (BQ=%u U_pad=%u K_pad=%u K_tiles=%u%s %s): "
                "q_per_plan=%u sub_plans=%u batch_plans=%u banks=%u SRAM=%.2f MiB (chain=%d)\n",
                nh_, valid_u_, bq_, u_pad_, k_pad_, n_kt,
                sparse_keep_full_ ? (" SPARSE " + std::to_string(sparse_keep_full_) + "/" + std::to_string(n_full_tiles_)).c_str() : " DENSE",
                use_pv256_ ? "PV256" : "PSUM",
                q_per_plan_, sub_plans_per_q_,
                plans_per_batch_, num_banks_, double(sram_used_) / (1024.0 * 1024.0), int(rt.chain_enabled()));
}

DirectH3Attn::~DirectH3Attn() = default;

double DirectH3Attn::run(const DirectH3AttnArgs& args, const uint16_t* active_full_tiles) {
    if (sparse_keep_full_ > 0 && !active_full_tiles) {
        throw std::invalid_argument("DirectH3Attn::run: sparse executor requires active_full_tiles");
    }
    const auto t0 = std::chrono::steady_clock::now();

    // Retarget static HBM tensors only if their addresses ever changed (typically never across 50 layers!).
    if (args.o_addr != bound_.o_addr || args.sc_addr != bound_.sc_addr || args.dv_addr != bound_.dv_addr) {
        for (uint32_t bk = 0; bk < num_banks_; ++bk) {
            for (auto& pl : plans_[bk]) {
                if (args.o_addr != bound_.o_addr) pl->patch_addr(kTagAttOut, args.o_addr);
                if (args.sc_addr != bound_.sc_addr) pl->patch_addr(kTagScoreScale, args.sc_addr);
                if (args.dv_addr != bound_.dv_addr) pl->patch_addr(kTagDScaleV, args.dv_addr);
            }
        }
        bound_.o_addr = args.o_addr;
        bound_.sc_addr = args.sc_addr;
        bound_.dv_addr = args.dv_addr;
    }

    const uint32_t total_q_blocks = nh_ * q_blocks_per_head_;
    const uint32_t slots_per_batch = (sub_plans_per_q_ == 1u)
        ? (plans_per_batch_ * q_per_plan_)
        : (plans_per_batch_ / sub_plans_per_q_);
    const uint64_t q_head_stride = uint64_t(u_pad_) * kD;          // F8 = 1 byte
    const uint64_t kv_head_stride = uint64_t(kv_stride_rows_) * kD;  // F8 = 1 byte
    const uint64_t v_head_stride = uint64_t(kv_stride_rows_) * (use_pv256_ ? 256ull : kD);
    const uint32_t n_kt = uint32_t(k_tiles_.size());

    std::vector<int32_t> ctrl_host(size_t(slots_per_batch) * 4u, 0);
    std::vector<DirectPlan*> active[2];
    active[0].reserve(plans_per_batch_);
    active[1].reserve(plans_per_batch_);

    auto prepare_batch = [&](uint32_t q_off, uint32_t bk) -> size_t {
        const uint32_t batch_q = std::min(slots_per_batch, total_q_blocks - q_off);
        const uint32_t n_plans = (sub_plans_per_q_ == 1u)
            ? ((batch_q + q_per_plan_ - 1u) / q_per_plan_)
            : (batch_q * sub_plans_per_q_);
        active[bk].clear();
        for (uint32_t pi = 0; pi < n_plans; ++pi) active[bk].push_back(plans_[bk][pi].get());

        const uint32_t active_slots = (sub_plans_per_q_ == 1u) ? (n_plans * q_per_plan_) : batch_q;
        for (uint32_t sl = 0; sl < active_slots; ++sl) {
            const bool valid_slot = (sl < batch_q);
            const uint32_t gq = valid_slot ? (q_off + sl) : 0u;
            const uint32_t h = gq / q_blocks_per_head_;
            const uint32_t qb = gq % q_blocks_per_head_;

            ctrl_host[size_t(sl) * 4u + 0] = valid_slot ? int32_t(h) : -1;
            ctrl_host[size_t(sl) * 4u + 1] = int32_t(qb * bq_);
            ctrl_host[size_t(sl) * 4u + 2] = 0;
            ctrl_host[size_t(sl) * 4u + 3] = 0;

            const uint64_t q_base = args.q_addr + uint64_t(h) * q_head_stride + uint64_t(qb) * bq_ * kD;
            const uint64_t k_head = args.k_addr + uint64_t(h) * kv_head_stride;
            const uint64_t v_head = args.v_addr + uint64_t(h) * v_head_stride;

            const uint16_t* q_tiles = (sparse_keep_full_ > 0)
                ? (active_full_tiles + size_t(gq) * sparse_keep_full_)
                : nullptr;
            auto tile_byte_off = [&](uint32_t kt) -> uint64_t {
                if (q_tiles && kt < sparse_keep_full_) {
                    return uint64_t(q_tiles[kt]) * 4096ull * kD;
                }
                return uint64_t(k_tiles_[kt].col_start) * kD;
            };

            if (sub_plans_per_q_ == 1u) {
                const uint32_t pi = sl / q_per_plan_;
                const uint32_t ql = sl % q_per_plan_;
                auto& pl = *plans_[bk][pi];
                pl.patch_addr(tag_q(ql), q_base);
                for (uint32_t kt = 0; kt < n_kt; ++kt) {
                    const uint64_t koff = tile_byte_off(kt);
                    const uint64_t voff = use_pv256_ ? (koff * 2ull) : koff;
                    pl.patch_addr(tag_k(ql, kt), k_head + koff);
                    pl.patch_addr(tag_v(ql, kt), v_head + voff);
                }
            } else {
                for (uint32_t sub = 0; sub < sub_plans_per_q_; ++sub) {
                    auto& pl = *plans_[bk][sl * sub_plans_per_q_ + sub];
                    pl.patch_addr(tag_q(0), q_base);
                    const uint32_t kt0 = sub * kMaxTilesPerPlan;
                    const uint32_t kt1 = std::min(n_kt, kt0 + kMaxTilesPerPlan);
                    for (uint32_t kt = kt0; kt < kt1; ++kt) {
                        const uint64_t koff = tile_byte_off(kt);
                        const uint64_t voff = use_pv256_ ? (koff * 2ull) : koff;
                        pl.patch_addr(tag_k(0, kt), k_head + koff);
                        pl.patch_addr(tag_v(0, kt), v_head + voff);
                    }
                }
            }
        }
        return size_t(active_slots) * 16u;
    };

    double patch_ms = 0.0, flush_ms = 0.0, chain_ms = 0.0;
    if (num_banks_ == 2u) {
        // Double-buffered CPU-patch + PDMA-flush || Gaudi2 chain execution!
        uint32_t bidx = 0;
        uint32_t prev_inflight = 0;
        for (uint32_t q_off = 0; q_off < total_q_blocks; q_off += slots_per_batch, ++bidx) {
            const uint32_t bk = bidx & 1u;
            const auto tp0 = std::chrono::steady_clock::now();
            const size_t cbytes = prepare_batch(q_off, bk);
            const auto tp1 = std::chrono::steady_clock::now();
            rt_.flush_patches_and_upload_async(active[bk], d_ctrl_[bk].address(), ctrl_host.data(), cbytes);
            const auto tp2 = std::chrono::steady_clock::now();
            if (prev_inflight > 0) rt_.wait_chain(prev_inflight);
            rt_.wait_patches();
            const auto tp3 = std::chrono::steady_clock::now();
            prev_inflight = rt_.submit_chain(active[bk]);
            patch_ms += std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            flush_ms += std::chrono::duration<double, std::milli>(tp2 - tp1).count();
            chain_ms += std::chrono::duration<double, std::milli>(tp3 - tp2).count();
        }
        const auto tw0 = std::chrono::steady_clock::now();
        if (prev_inflight > 0) rt_.wait_chain(prev_inflight);
        chain_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count();
    } else {
        for (uint32_t q_off = 0; q_off < total_q_blocks; q_off += slots_per_batch) {
            const auto tp0 = std::chrono::steady_clock::now();
            const size_t cbytes = prepare_batch(q_off, 0u);
            const auto tp1 = std::chrono::steady_clock::now();
            rt_.flush_patches_and_upload(active[0], d_ctrl_[0].address(), ctrl_host.data(), cbytes);
            const auto tp2 = std::chrono::steady_clock::now();
            chain_ms += rt_.run_chain(active[0]) / 1000.0;
            patch_ms += std::chrono::duration<double, std::milli>(tp1 - tp0).count();
            flush_ms += std::chrono::duration<double, std::milli>(tp2 - tp1).count();
        }
    }

    const double tot_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static bool prof = std::getenv("G2E_H3_DIRECT_PROF") != nullptr;
    if (prof) {
        std::printf("  [DirectH3Attn] total=%.2f ms (patch=%.2f, flush=%.2f, wait=%.2f)\n",
                    tot_ms, patch_ms, flush_ms, chain_ms);
    }
    return tot_ms;
}

}  // namespace g2e
