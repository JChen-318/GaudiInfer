#include "g2e/direct_attn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace g2e {

namespace {

constexpr uint32_t kB = 8;
constexpr uint32_t kT = 16;
constexpr uint32_t kKV = 4;
constexpr uint32_t kG = 6;
constexpr uint32_t kRHead = kG * kT;         // 96
constexpr uint32_t kRowsAll = kKV * kRHead;  // 384
constexpr uint32_t kD = 256;
constexpr uint32_t kNHD = kKV * kG * kD;     // 6144
constexpr uint32_t kTpcs = 24;
constexpr uint32_t kMaxTileW = 4096;
// Each sub-plan fits <= 32 tiles in <= 834 monitors (< 1024 private monitors).
constexpr uint32_t kMaxTilesPerPlan = 32;

uint64_t align256(uint64_t x) { return (x + 255ull) & ~255ull; }
uint32_t f32_bits(float v) { uint32_t u = 0; std::memcpy(&u, &v, 4); return u; }

// Patch tag namespaces (non-overlapping):
//   1: q_in
//   10..17: att_out[b]
//   1000 + tile_idx * 16 + h:     K tile h
//   1000 + tile_idx * 16 + 4 + h: V tile h
//   1000 + tile_idx * 16 + 8:     pos_sm SRF for tile_idx
constexpr int kTagQIn = 1;
int tag_att_out(uint32_t b) { return 10 + int(b); }
int tag_k(uint32_t tidx, uint32_t h) { return 1000 + int(tidx) * 16 + int(h); }
int tag_v(uint32_t tidx, uint32_t h) { return 1000 + int(tidx) * 16 + 4 + int(h); }

}  // namespace

DirectAttnAssets DirectAttnAssets::load(DirectRuntime& rt, const std::string& desc_dir, const std::string& tpc_obj_dir) {
    DirectAttnAssets a;
    a.qk_bf16_1024 = MmeTemplate::load_dir(desc_dir + "/attn_qk_bf16_1024");
    a.pv_bf16_1024 = MmeTemplate::load_dir(desc_dir + "/attn_pv_bf16_1024");
    a.qk_bf16_2048 = MmeTemplate::load_dir(desc_dir + "/attn_qk_bf16_2048");
    a.pv_bf16_2048 = MmeTemplate::load_dir(desc_dir + "/attn_pv_bf16_2048");
    a.qk_bf16_4096 = MmeTemplate::load_dir(desc_dir + "/attn_qk_bf16_4096");
    a.pv_bf16_4096 = MmeTemplate::load_dir(desc_dir + "/attn_pv_bf16_4096");

    a.qk_f8_1024 = MmeTemplate::load_dir(desc_dir + "/attn_qk_f8_1024");
    a.pv_f8_1024 = MmeTemplate::load_dir(desc_dir + "/attn_pv_f8_1024");
    a.qk_f8_2048 = MmeTemplate::load_dir(desc_dir + "/attn_qk_f8_2048");
    a.pv_f8_2048 = MmeTemplate::load_dir(desc_dir + "/attn_pv_f8_2048");
    a.qk_f8_4096 = MmeTemplate::load_dir(desc_dir + "/attn_qk_f8_4096");
    a.pv_f8_4096 = MmeTemplate::load_dir(desc_dir + "/attn_pv_f8_4096");

    a.isa_tile_step_bf16 = rt.load_tpc_elf(tpc_obj_dir + "/direct_attn_tile_step_bf16_gaudi2.o");
    a.isa_tile_step_f8 = rt.load_tpc_elf(tpc_obj_dir + "/direct_attn_tile_step_f8_gaudi2.o");
    a.isa_q_pack_bf16 = rt.load_tpc_elf(tpc_obj_dir + "/direct_attn_q_pack_bf16_gaudi2.o");
    a.isa_q_pack_f8 = rt.load_tpc_elf(tpc_obj_dir + "/direct_attn_q_pack_f8_gaudi2.o");
    return a;
}

DirectAttnCore::DirectAttnCore(DirectRuntime& rt, const DirectAttnAssets& assets, bool kv8, bool pack_q_from_hbm,
                               std::array<uint32_t, 8> slot_rows, const DirectAttnRunArgs& init_args,
                               uint64_t sram_offset, uint32_t tpc_cmd)
    : kv8_(kv8), pack_q_(pack_q_from_hbm), slot_rows_(slot_rows), cur_(init_args) {
    // 1. Decompose each active slot's row count into greedy {4096, 2048, 1024} tiles (slot_rows[b] == 0 skips slot b).
    for (uint32_t b = 0; b < kB; ++b) {
        const uint32_t rows = slot_rows[b];
        if (!rows) continue;
        if ((rows % 1024u) != 0)
            throw std::invalid_argument("direct_attn: slot_rows must be 0 or a multiple of 1024");
        uint32_t col = 0;
        bool first = true;
        while (col < rows) {
            const uint32_t rem = rows - col;
            const uint32_t w = rem >= 4096u ? 4096u : (rem >= 2048u ? 2048u : 1024u);
            TileDesc td{};
            td.slot = b;
            td.col_start = col;
            td.width = w;
            td.is_first = first;
            td.is_last = (col + w == rows);
            tiles_.push_back(td);
            col += w;
            first = false;
        }
    }
    total_tiles_ = uint32_t(tiles_.size());
    if (!total_tiles_) throw std::invalid_argument("direct_attn: no active slots");

    // 2. Allocate SRAM arena starting at rt.sram_base() + sram_offset.
    const uint64_t kv_es = kv8 ? 1ull : 2ull;
    const DType kv_dt = kv8 ? DType::F8 : DType::BF16;
    uint64_t sp = align256(rt.sram_base() + sram_offset);
    auto alloc = [&](uint64_t bytes) -> uint64_t {
        const uint64_t addr = sp;
        sp = align256(sp + bytes);
        return addr;
    };

    const uint64_t q_slot_bytes = uint64_t(kKV) * kRHead * kD * kv_es;
    const uint64_t q_head_bytes = uint64_t(kRHead) * kD * kv_es;
    const uint64_t s_q_sram = pack_q_from_hbm ? alloc(uint64_t(kB) * q_slot_bytes) : init_args.q_in;
    const uint64_t s_S[2] = {
        alloc(uint64_t(kRowsAll) * kMaxTileW * 2ull),
        alloc(uint64_t(kRowsAll) * kMaxTileW * 2ull),
    };
    const uint64_t s_P[2] = {
        alloc(uint64_t(kRowsAll) * kMaxTileW * kv_es),
        alloc(uint64_t(kRowsAll) * kMaxTileW * kv_es),
    };
    const uint64_t s_PV[2] = {
        alloc(uint64_t(kRowsAll) * kD * 4ull),
        alloc(uint64_t(kRowsAll) * kD * 4ull),
    };
    uint64_t s_O[kB]{}, s_m[kB]{}, s_l[kB]{};
    for (uint32_t b = 0; b < kB; ++b) if (slot_rows[b]) s_O[b] = alloc(uint64_t(kRowsAll) * kD * 4ull);
    for (uint32_t b = 0; b < kB; ++b) if (slot_rows[b]) s_m[b] = alloc(uint64_t(kRowsAll) * 4ull);
    for (uint32_t b = 0; b < kB; ++b) if (slot_rows[b]) s_l[b] = alloc(uint64_t(kRowsAll) * 4ull);
    uint64_t s_res[4];
    for (uint32_t r = 0; r < 4; ++r) s_res[r] = alloc(uint64_t(kRowsAll) * 4ull);

    sram_used_ = sp - (rt.sram_base() + sram_offset);
    if (sp > rt.sram_base() + rt.sram_size())
        throw std::runtime_error("direct_attn: SRAM overflow");

    const uint64_t isa_step = kv8 ? assets.isa_tile_step_f8 : assets.isa_tile_step_bf16;
    const uint64_t isa_pack = kv8 ? assets.isa_q_pack_f8 : assets.isa_q_pack_bf16;
    const uint32_t kl_bits = f32_bits(float((1.0 / std::sqrt(double(kD))) * 1.4426950408889634));

    auto qk_tmpl = [&](uint32_t w) -> const MmeTemplate& {
        if (kv8) return w == 4096 ? assets.qk_f8_4096 : (w == 2048 ? assets.qk_f8_2048 : assets.qk_f8_1024);
        return w == 4096 ? assets.qk_bf16_4096 : (w == 2048 ? assets.qk_bf16_2048 : assets.qk_bf16_1024);
    };
    auto pv_tmpl = [&](uint32_t w) -> const MmeTemplate& {
        if (kv8) return w == 4096 ? assets.pv_f8_4096 : (w == 2048 ? assets.pv_f8_2048 : assets.pv_f8_1024);
        return w == 4096 ? assets.pv_bf16_4096 : (w == 2048 ? assets.pv_bf16_2048 : assets.pv_bf16_1024);
    };

    // 3. Build sub-plans of <= kMaxTilesPerPlan tiles each (zero-bubble double-buffered MME || TPC).
    for (uint32_t base = 0; base < total_tiles_; base += kMaxTilesPerPlan) {
        const uint32_t count = std::min(kMaxTilesPerPlan, total_tiles_ - base);
        const uint32_t plan_idx = uint32_t(plans_.size());
        for (uint32_t k = 0; k < count; ++k) tiles_[base + k].plan_idx = plan_idx;

        // Align monitor indices across all sub-plans so mon_cfg cache hits 100% (avoids re-config LBWs)!
        const uint32_t init_mon = (base > 0 && pack_q_from_hbm) ? 2u : 0u;
        auto plan = std::make_unique<DirectPlan>(rt, init_mon);
        uint32_t s0 = 0;  // MME completion SOB
        uint32_t s2 = 0;  // TPC completion SOB

        // Optional Q pack from HBM [8, 16, 24, 256] -> SRAM [8, 4, 96, 256] (first sub-plan only)
        const bool do_pack = (base == 0 && pack_q_from_hbm);
        if (do_pack) {
            const DirectTpcTensor t[2] = {
                {init_args.q_in, DType::BF16, 4, {kD, kKV * kG, kT, kB, 1}},
                {s_q_sram, kv_dt, 4, {kD, kRHead, kKV, kB, 1}},
            };
            const int tags[2] = {kTagQIn, -1};
            plan->add_tpc(isa_pack, t, {}, kRowsAll, kTpcs, -1, 0, 2, tpc_cmd, tags);
            s2 += kTpcs;
        }

        auto emit_qk = [&](uint32_t local_k, uint32_t parity, int wait_sob, uint32_t wait_val) {
            const uint32_t tidx = base + local_k;
            const auto& td = tiles_[tidx];
            const auto& tmpl = qk_tmpl(td.width);
            const uint64_t s_head_bytes = uint64_t(kRHead) * td.width * 2ull;
            const uint64_t q_slot_base = s_q_sram + uint64_t(td.slot) * q_slot_bytes;
            for (uint32_t h = 0; h < kKV; ++h) {
                const uint64_t q_addr = q_slot_base + uint64_t(h) * q_head_bytes;
                const uint64_t k_addr = init_args.k_cache[td.slot][h] + uint64_t(td.col_start) * kD * kv_es;
                const uint64_t s_addr = s_S[parity] + uint64_t(h) * s_head_bytes;
                plan->add_mme(tmpl, q_addr, k_addr, s_addr, 0,
                              h == 0 ? wait_sob : -1, h == 0 ? wait_val : 0, 0,
                              {-1, tag_k(tidx, h), -1, -1});
                s0 += tmpl.signal_count;
            }
        };

        auto emit_pv = [&](uint32_t local_k, uint32_t parity, int wait_sob, uint32_t wait_val) {
            const uint32_t tidx = base + local_k;
            const auto& td = tiles_[tidx];
            const auto& tmpl = pv_tmpl(td.width);
            const uint64_t p_head_bytes = uint64_t(kRHead) * td.width * kv_es;
            const uint64_t pv_head_bytes = uint64_t(kRHead) * kD * 4ull;
            // Dual-engine head-parallel PV: MME0 runs heads 0,2 while MME1 runs heads 1,3 simultaneously (2x faster)!
            for (uint32_t hp = 0; hp < kKV; hp += 2) {
                const uint32_t h0 = hp, h1 = hp + 1;
                const uint64_t p0 = s_P[parity] + uint64_t(h0) * p_head_bytes;
                const uint64_t v0 = init_args.v_cache[td.slot][h0] + uint64_t(td.col_start) * kD * kv_es;
                const uint64_t o0 = s_PV[parity] + uint64_t(h0) * pv_head_bytes;
                const uint64_t p1 = s_P[parity] + uint64_t(h1) * p_head_bytes;
                const uint64_t v1 = init_args.v_cache[td.slot][h1] + uint64_t(td.col_start) * kD * kv_es;
                const uint64_t o1 = s_PV[parity] + uint64_t(h1) * pv_head_bytes;
                plan->add_mme_pair(tmpl, p0, v0, o0, p1, v1, o1,
                                   hp == 0 ? wait_sob : -1, hp == 0 ? wait_val : 0, 0,
                                   {-1, tag_v(tidx, h0), -1, -1},
                                   {-1, tag_v(tidx, h1), -1, -1});
                s0 += 2u;  // 1 engine-0 signal + 1 engine-1 signal per pair
            }
        };

        auto emit_tpc_step = [&](int sm_local_k, int acc_local_k, uint32_t parity,
                                 int wait_sob, uint32_t wait_val, int sig_sob) {
            const bool do_sm = sm_local_k >= 0;
            const bool do_acc = acc_local_k >= 0;
            const uint32_t sm_tidx = do_sm ? (base + uint32_t(sm_local_k)) : base;
            const uint32_t acc_tidx = do_acc ? (base + uint32_t(acc_local_k)) : base;
            const auto& sm_td = tiles_[sm_tidx];
            const auto& acc_td = tiles_[acc_tidx];

            const uint32_t tile_w = do_sm ? sm_td.width : 1024u;
            const uint32_t flags_sm = do_sm ? (1u | (sm_td.is_first ? 2u : 0u)) : 0u;
            const uint32_t flags_acc = do_acc ? (1u | (acc_td.is_first ? 2u : 0u) | (acc_td.is_last ? 4u : 0u)) : 0u;

            const uint64_t att_slot_addr = init_args.att_out + uint64_t(acc_td.slot) * kT * kNHD * 2ull;
            const DirectTpcTensor t[11] = {
                {s_S[parity], DType::BF16, 2, {tile_w, kRowsAll, 1, 1, 1}},
                {s_P[parity], kv_dt, 2, {tile_w, kRowsAll, 1, 1, 1}},
                {s_m[sm_td.slot], DType::F32, 2, {1, kRowsAll, 1, 1, 1}},
                {s_l[sm_td.slot], DType::F32, 2, {1, kRowsAll, 1, 1, 1}},
                {s_res[sm_tidx & 3u], DType::F32, 2, {1, kRowsAll, 1, 1, 1}},
                {s_PV[parity], DType::F32, 2, {kD, kRowsAll, 1, 1, 1}},
                {s_O[acc_td.slot], DType::F32, 2, {kD, kRowsAll, 1, 1, 1}},
                {s_res[acc_tidx & 3u], DType::F32, 2, {1, kRowsAll, 1, 1, 1}},
                {s_l[acc_td.slot], DType::F32, 2, {1, kRowsAll, 1, 1, 1}},
                {att_slot_addr, DType::BF16, 2, {kNHD, kT, 1, 1, 1}},
                {init_args.last_tab, DType::I32, 2, {32, kB, 1, 1, 1}},
            };
            const int t_tags[11] = {-1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    (do_acc && acc_td.is_last) ? tag_att_out(acc_td.slot) : -1, -1};
            const uint32_t srfs[5] = {
                kl_bits,
                do_sm ? sm_td.col_start : 0u,
                do_sm ? sm_td.slot : 0u,
                flags_sm,
                flags_acc,
            };
            plan->add_tpc(isa_step, t, srfs, kRowsAll, kTpcs, wait_sob, wait_val, sig_sob, tpc_cmd, t_tags);
            if (sig_sob == 2) s2 += kTpcs;
        };

        if (count == 1) {
            emit_qk(0, 0, do_pack ? 2 : -1, do_pack ? s2 : 0);
            const uint32_t qk0_done = s0;
            emit_tpc_step(0, -1, 0, 0, qk0_done, 2);
            emit_pv(0, 0, 2, s2);
            const uint32_t pv0_done = s0;
            emit_tpc_step(-1, 0, 0, 0, pv0_done, -1);
        } else {
            // Double-buffered zero-bubble MME || TPC pipeline!
            std::vector<uint32_t> s0_ready(count + 2, 0);
            // Prologue: QK(0) -> S[0], QK(1) -> S[1] back-to-back on MME
            emit_qk(0, 0, do_pack ? 2 : -1, do_pack ? s2 : 0);
            s0_ready[0] = s0;
            emit_qk(1, 1, -1, 0);
            s0_ready[1] = s0;

            for (uint32_t k = 0; k < count; ++k) {
                const uint32_t p = k & 1u;
                // TPC(k): SM(k) on S[p]->P[p] + (if k>=2) ACC(k-2) on PV[p]->O_acc
                emit_tpc_step(int(k), k >= 2 ? int(k - 2) : -1, p, 0, s0_ready[k], 2);
                // MME_next(k): waits TPC(k), runs PV(k) on P[p]->PV[p] + (if k+2<count) QK(k+2) -> S[p]
                emit_pv(k, p, 2, s2);
                if (k + 2 < count) emit_qk(k + 2, p, -1, 0);
                s0_ready[k + 2] = s0;
            }
            // Epilogue: ACC(count-2) and ACC(count-1)
            emit_tpc_step(-1, int(count - 2), (count - 2) & 1u, 0, s0_ready[count], 2);
            emit_tpc_step(-1, int(count - 1), (count - 1) & 1u, 0, s0_ready[count + 1], -1);
        }

        plan->prepare();
        plans_.push_back(std::move(plan));
    }
}

DirectAttnCore::~DirectAttnCore() = default;

void DirectAttnCore::update_args(const DirectAttnRunArgs& args) {
    const uint64_t kv_es = kv8_ ? 1ull : 2ull;
    if (args.q_in != cur_.q_in) {
        if (pack_q_) {
            plans_[0]->patch_addr(kTagQIn, args.q_in);
        } else {
            throw std::logic_error("direct_attn: dynamic q_in patch without pack_q not yet implemented");
        }
        cur_.q_in = args.q_in;
    }
    if (args.att_out != cur_.att_out) {
        for (uint32_t b = 0; b < kB; ++b) {
            const uint64_t addr = args.att_out + uint64_t(b) * kT * kNHD * 2ull;
            for (auto& pl : plans_) pl->patch_addr(tag_att_out(b), addr);
        }
        cur_.att_out = args.att_out;
    }
    bool any_kv_diff = false;
    for (uint32_t b = 0; b < kB && !any_kv_diff; ++b) {
        if (!slot_rows_[b]) continue;
        for (uint32_t h = 0; h < kKV; ++h) {
            if (args.k_cache[b][h] != cur_.k_cache[b][h] || args.v_cache[b][h] != cur_.v_cache[b][h]) {
                any_kv_diff = true;
                break;
            }
        }
    }
    if (!any_kv_diff) return;
    for (uint32_t tidx = 0; tidx < total_tiles_; ++tidx) {
        const auto& td = tiles_[tidx];
        const uint32_t b = td.slot;
        auto& pl = *plans_[td.plan_idx];
        for (uint32_t h = 0; h < kKV; ++h) {
            if (args.k_cache[b][h] != cur_.k_cache[b][h])
                pl.patch_addr(tag_k(tidx, h), args.k_cache[b][h] + uint64_t(td.col_start) * kD * kv_es);
            if (args.v_cache[b][h] != cur_.v_cache[b][h])
                pl.patch_addr(tag_v(tidx, h), args.v_cache[b][h] + uint64_t(td.col_start) * kD * kv_es);
        }
    }
    for (uint32_t b = 0; b < kB; ++b) {
        if (!slot_rows_[b]) continue;
        cur_.k_cache[b] = args.k_cache[b];
        cur_.v_cache[b] = args.v_cache[b];
    }
}

double DirectAttnCore::run() {
    double total_us = 0.0;
    for (auto& pl : plans_) total_us += pl->run();
    return total_us;
}

void DirectAttnCore::collect_plans(std::vector<DirectPlan*>& out) const {
    for (const auto& pl : plans_) out.push_back(pl.get());
}

}  // namespace g2e
