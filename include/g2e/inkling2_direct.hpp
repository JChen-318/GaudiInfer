// Inkling-2 on the Direct engine (include/g2e/direct_mme.hpp): layer applications as prepared DirectPlans of
// captured MME templates (tools/mme_descs_ink) and our TPC kernels, synchronised in hardware, used in place of their
// Synapse recipes: decode (T = 1, one row or a batch of 2 / 4 / 8), verification (T = 2..4) and prefill (1024-row
// chunks). G2E_INK_DIRECT / G2E_INK_DPREFILL are masks of the application kinds (15: all; the production setting).
// Needs tools/scal-g2e-ink.json as the SCAL config. See docs/inkling2.md and docs/inkling2-direct-plan.md.
#pragma once
#include <functional>

#include "g2e/direct_mme.hpp"
#include "g2e/graph.hpp"
#include "g2e/runtime.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace g2e {

class Inkling2;

class InkDirect {
public:
    explicit InkDirect(Inkling2& m);
    ~InkDirect();
    // Whether this application (at T = 1) runs on the Direct engine.
    bool handles(uint32_t app) const;
    // Runs the application's plan (synchronous). The caller has synchronised the model's stream.
    double run_app(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off);
    // Chained execution: queue() appends the application's plan; flush() runs the queued plans as one chain on the
    // scheduler (one host wait) and must precede any other device work that depends on them. queue() returns false
    // when the queue was empty (the caller then synchronises its stream first).
    bool queue(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off);
    void flush();
    bool pending() const { return !queue_.empty() || !pending_dec_.empty(); }
    // speculative verification (T = 2 .. kVerifyMaxT rows, nothing committed): see build_v
    static constexpr uint32_t kVerifyMaxT = 4;
    bool handles_v(uint32_t app, uint32_t T) const;
    bool queue_v(uint32_t app, uint32_t T, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off);
    bool handles_verify_full(uint32_t T) const;
    bool handles_v_commit() const;
    // accept(n) of a T-row verification: every application's conv states after row n - 1 into conv_[side_out]
    void rollback(int side_out, uint32_t T, uint32_t n);
    void commit_and_rollback(int side_out, uint32_t T, uint32_t n);
    // every application of a T = 1 forward runs on the Direct engine (no commit recipe needed)
    bool handles_all() const;
    // 100% Zero-Synapse Direct T = 1 decode: Direct embed + 48 apps + Direct final_q/norm/lm_head/softcap+argmax
    bool handles_full_decode() const;
    struct DecodeAppIO {
        uint32_t app;
        int hcur;
        const DeviceBuffer* out_buf;
        uint64_t out_off;
    };
    void run_verify_full(std::span<const int32_t> tokens, uint32_t pos, float tau0, int side,
                         std::span<const DecodeAppIO> apps, const DeviceBuffer* last_out, uint64_t last_off,
                         std::vector<float>* logits, double* out_embed_ms = nullptr,
                         double* out_layers_ms = nullptr, double* out_head_ms = nullptr);
    int32_t run_decode_full(int32_t tok, uint32_t pos, float tau0, int side,
                            std::span<const DecodeAppIO> apps, const DeviceBuffer* last_out, uint64_t last_off,
                            bool need_head, bool want_logits, std::vector<float>* logits, bool want_argmax,
                            std::vector<uint16_t>* aux_out, double* out_embed_ms = nullptr,
                            double* out_layers_ms = nullptr, double* out_head_ms = nullptr);
    // True Bp in {2, 4, 8} batched Direct decode across distinct sequence slots (mixed contexts 0..1M, mixed greedy/sampled).
    bool handles_decode_batch() const;
    // One line for the log: which parts run on this engine (a missing template or kernel turns a part off).
    std::string summary() const;
    void run_decode_batch(std::span<const uint32_t> slots, std::span<const int32_t> tokens,
                          std::span<const uint8_t> want_logits, std::vector<int32_t>& out_argmax,
                          std::vector<std::vector<float>>& out_logits);
    // prefill chunks (T = 1024) on the Direct engine (G2E_INK_DPREFILL, src/inkling2_dprefill.cpp)
    bool handles_p(uint32_t app, uint32_t T) const;                // at the model's position
    bool handles_p(uint32_t app, uint32_t T, uint32_t pos) const;  // for a chunk starting at pos
    // every application of a T-row forward (the plans commit the K / V rows and write the taps themselves)
    bool handles_all_p(uint32_t T) const;
    bool handles_all_p(uint32_t T, uint32_t pos) const;
    // queues the application's plan (chunk slot 0: the model's buffers); cs_in / cs_out: the conv state sides read and
    // written (the same side: updated in place)
    bool queue_p(uint32_t app, int hcur, const DeviceBuffer* out_buf, uint64_t out_off, int cs_in, int cs_out);
    // the plans' tables (position, tau, tpart/valid_T, commit rows) of a chunk slot
    void set_tables(int slot, uint32_t pos, uint32_t T, uint32_t valid_T = 0);
    // two chunks (slot 0 at pos, slot 1 at pos + 1024) interleaved, one application apart; io: per application
    // (partial block index read, output address) of each slot. Returns the device time (us).
    double run_pair(uint32_t pos, std::span<const std::pair<int, uint64_t>> io0, std::span<const std::pair<int, uint64_t>> io1);
    void profile_stages_p(uint32_t app);
    void hash_scratch(uint32_t app);
    size_t verify_rplan(uint32_t app);
    void hash_fields(uint32_t app);
    void hash_kvs(uint32_t app);
    // (G2E_INK_DPCHECK) the prefill scratch and the addresses of some of its tensors
    const DeviceBuffer& pscratch() const { return pscratch_[0]; }
    std::map<std::string, uint64_t> pdebug() const {
        const uint64_t T = 1024, NH = 24, D = 128, GT = 3 * T, KV = 8;
        const auto& ps_0 = ps_[0];
        return {{"gated", ps_0.gated}, {"a1", ps_0.a1}, {"xn", ps_0.xn}, {"mo", ps_0.mo}, {"idx", ps_0.idx}, {"res", ps_0.res},
                {"q", ps_0.q}, {"q#bytes", NH * T * D * 2}, {"sums", ps_0.sums}, {"sums#bytes", KV * GT * 4}, {"o", ps_0.o},
                {"o#bytes", KV * GT * D * 4}, {"row", ps_0.row}, {"row#bytes", T * 8600 * 2}, {"p0", ps_0.p},
                {"p0#bytes", GT * 2048 * 2}, {"r16", ps_0.r16}, {"r16#bytes", T * NH * 16 * 2}, {"bt", ps_0.bt}, {"bt#bytes", T * NH * 1024 * 2}};
    }
    // Drop cached plans tied to `slot`'s full-layer linear KV cache addresses / capacity.
    void on_slot_resized(uint32_t slot);
    // Invalidate on-chip SRAM constant cache (call after any external Synapse recipe that may clobber SRAM).
    void invalidate_sram() { sram_const_valid_ = false; }
    bool has_runtime() const { return bool(rt_); }
    // Static BF16 weight cache for native expert layers (see Inkling2::init_bf16_cache).
    struct Bf16LayerAddrs {
        uint64_t w13 = 0;
        uint64_t w2 = 0;
        bool cached = false;
    };
    void init_bf16_cache(uint64_t kv_need_bytes);
    Bf16LayerAddrs bf16_layer(uint32_t layer) const {
        return layer < bf16_layers_.size() ? bf16_layers_[layer] : Bf16LayerAddrs{};
    }
    const DeviceBuffer& bf16_cache() const { return bf16_cache_; }
    uint64_t bf16_staging_w13() const { return bf16_staging_w13_; }
    uint64_t bf16_staging_w2() const { return bf16_staging_w2_; }
    uint64_t bf16_w13_bytes() const { return bf16_w13_bytes_; }
    uint64_t bf16_w2_bytes() const { return bf16_w2_bytes_; }
    void download_bf16_layer(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out) const;
    void download_bf16_staged(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out);
    // Times the plan cut after each stage (median of 25 runs; the application's plan must have run once).
    void profile_stages(uint32_t app);
    void profile_stages_v(uint32_t app, uint32_t T);

private:
    enum DecodeSplit : uint8_t { kDecFull = 0, kDecPreFar = 1, kDecPostFar = 2 };
    struct DecCtx {
        bool sram_act;
        bool sram_const;
    };
    DirectPlan& plan(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off,
                     DecodeSplit split = kDecFull, DecCtx dctx = {false, false}, uint32_t active_sf = 0);
    // G2E_INK_DIRECT_CHECK=1: recompute each stage on the host from the device inputs and print the deviations
    void check(uint32_t app, int side, int hcur);
    std::unique_ptr<DirectPlan> build(uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off, int lim,
                                      DecodeSplit split = kDecFull, DecCtx dctx = {false, false});
    // Append one T = 1 decode application into `P` using dynamic SOBs (`nsob`); links from (wait_in_sob, wait_in_tgt)
    // and signals `out_sig_sob` (-1: host/chain completion). Returns false if cut by `lim` on an MME stage.
    // When `skip_in_gram` is true, the previous app's fused tail already populated s_.p1 (and s_.p3 if msg).
    // `next_fused_app`: >= 0 fuses that next app's input gram; -2 fuses d_head's final_q gram; -1 disables fusion.
    bool append_decode_app(DirectPlan& P, int& nsob, int wait_in_sob, uint32_t wait_in_tgt, int out_sig_sob,
                           uint32_t app, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off, int lim,
                           DecodeSplit split = kDecFull, bool skip_in_gram = false, int next_fused_app = -1,
                           DecCtx dctx = {false, false});
    void append_d_embed(DirectPlan& P, int out_sig_sob, bool sram_act);
    void append_d_head(DirectPlan& P, int& nsob, int wait_in_sob, uint32_t wait_in_tgt,
                       uint64_t last_addr, bool aux_on, bool want_logits, bool skip_in_gram, DecCtx dctx);
    struct DecodeKey {
        uint32_t app = 0;
        int side = 0, hcur = 0;
        uint64_t out_addr = 0, out_off = 0;
        uint32_t sf_tap = 0;
        uint8_t seq_slot = 0, split = 0;
        uint16_t flags = 0;
        bool operator<(const DecodeKey& o) const {
            return std::tie(app, side, hcur, out_addr, out_off, sf_tap, seq_slot, split, flags) <
                   std::tie(o.app, o.side, o.hcur, o.out_addr, o.out_off, o.sf_tap, o.seq_slot, o.split, o.flags);
        }
    };
    struct PendingDecode {
        DecodeKey key;
        const DeviceBuffer* out_buf = nullptr;
    };
    std::vector<PendingDecode> pending_dec_;
    std::map<std::vector<DecodeKey>, std::unique_ptr<DirectPlan>> span_plans_;
    bool d_full_on_ = false;
    bool dfar_tiled_ok_ = false;
    bool dfar_pipe_on_ = false;
    bool dfar_tpc_pv_on_ = false;
    uint32_t dfar_tile_thresh_ = 131072u;
    uint32_t dfar_cap_ = 0;   // far_scratch / SRAM far rows (<= 131072)
    uint32_t vfar_cap_ = 0;   // vscratch_ far rows (<= 65536)
    bool use_tiled_far(uint32_t fr) const;
    // Reusable patched Tiled MME FlashDecode plans for tw in {1024, 4096, 32768, 65536, 131072}.
    static constexpr uint32_t kNumDFarWidths = 5u;
    static constexpr uint32_t kDFarWidths[kNumDFarWidths] = {1024u, 4096u, 32768u, 65536u, 131072u};
    static constexpr uint32_t kDFarPool = 16u;
    static constexpr uint32_t kDFarPipeMaxTiles = 16u;
    static constexpr uint32_t kDFarPipePool = 16u;
    struct DFarTileEntry {
        std::unique_ptr<DirectPlan> plan;
        uint64_t q_addr = 0, kl = 0, vl = 0, dmax = 0, sums = 0, pv = 0, oacc = 0;
        uint32_t cap = 0, col = UINT32_MAX, nv = UINT32_MAX, first = UINT32_MAX;
    };
    struct DFarPipeEntry {
        std::unique_ptr<DirectPlan> plan;
        uint64_t q_addr = 0, kl = 0, vl = 0, dmax = 0, sums = 0, pv = 0, oacc = 0;
        uint32_t cap = 0, col0 = UINT32_MAX, first = UINT32_MAX;
        std::vector<uint32_t> nvs;
    };
    struct DFarRunState {
        std::array<uint32_t, kNumDFarWidths> tile_used{};
        std::map<std::vector<uint32_t>, uint32_t> pipe_used;
        void reset() { tile_used.fill(0); pipe_used.clear(); }
    };
    std::vector<DFarTileEntry> dfar_pool_[kNumDFarWidths];
    std::map<std::vector<uint32_t>, std::vector<DFarPipeEntry>> dfar_pipe_pool_;
    std::vector<uint32_t> compute_dfar_widths(uint32_t fr, uint32_t cap) const;
    void prebuild_dfar_for(uint32_t fr, uint32_t cap, uint32_t want_apps, int keep_bi = -1, uint32_t active_sf = 0);
    std::unique_ptr<DirectPlan> build_dfar_tile(uint32_t tw);
    std::unique_ptr<DirectPlan> build_dfar_pipe(const std::vector<uint32_t>& widths);
    void run_dfar_tiles(uint32_t app, uint32_t fr, uint32_t seq_slot, uint64_t q_addr, uint64_t pv_addr,
                        uint64_t dmax_addr, uint64_t sums_addr, uint64_t oacc_addr,
                        std::vector<DirectPlan*>& chain, DFarRunState& state, double& dev_us,
                        uint32_t* kvc_used = nullptr);
    DeviceBuffer d_argmax_;  // [2, 24, 64] (max_val f32 + max_idx i32, cache-line isolated per TPC)
    std::unique_ptr<DirectPlan> d_embed_plan_[2];
    std::map<std::tuple<uint64_t, bool, bool, bool, bool, bool>, std::unique_ptr<DirectPlan>> d_head_plans_;
    DirectPlan& d_embed_plan(bool sram_act = false, uint32_t active_sf = 0);
    DirectPlan& d_head_plan(uint64_t last_addr, bool aux_on, bool want_logits,
                            bool skip_in_gram = false, DecCtx dctx = {false, false}, uint32_t active_sf = 0);
    DirectPlan& sram_const_copy_plan(int keep_bi = -1, uint32_t active_sf = 0);
    std::unique_ptr<DirectPlan> sram_const_copy_plan_;
    uint64_t const_small_bytes_ = 0;
    uint64_t final_norm_hbm_ = 0;
    uint64_t sram_const_base_ = 0, sram_qil_base_ = 0;
    bool sram_const_on_ = false;
    bool sram_const_valid_ = false;
    // Batched Direct decode (Bp in {2, 4, 8}, bi in {0, 1, 2}): slot-independent patched spans + SRAM/HBM scratch
    static constexpr uint32_t kBatchSizes[3] = {2u, 4u, 8u};
    static constexpr uint32_t kBKvCommitPool = 64u;
    bool b_decode_ok_ = false;
    bool b_moe_dedup_on_ = false;
    bool b_dmg_fuse_on_ = false;
    bool b_span_on_ = false;
    bool b_opt_mme_on_ = false;
    bool b_kv4_on_ = false;
    bool b_route_b_on_ = false;
    DeviceBuffer bscratch_;  // backing HBM buffer (unused when sram_scratch_ places b_ in SRAM)
    DeviceBuffer bmeta_;     // [8, 8] i32/f32 row metadata in HBM
    DeviceBuffer b_argmax_;  // [2, 8, 24, 64] max_val f32 + max_idx i32 in HBM
    struct BScratch {
        uint64_t xa, row, q, r, sink, bt, sc, p, sums, dmax, pv, o_acc, b_kvs, gated, a1, res, xn, logits, idx, rw, act, ys,
            p1, p2, p3, p4, cat, xil, msg, gr, fgv, fqkv, mrmix, ymg, grp;
    } b_{};
    struct BAppPlan {
        std::unique_ptr<DirectPlan> main;  // batched span plan (or standalone embed/app/head when G2E_INK_B_SPAN=0)
        std::unique_ptr<DirectPlan> post;  // batched kv4 commit plan for sync_full_app (null if no sync or G2E_INK_B_KV4=0)
        uint32_t sync_full_app = UINT32_MAX;  // if != UINT32_MAX, run KV commit + dfar tiles for this full app after `main`
        std::vector<uint32_t> ring_apps;      // apps inside `main` whose kring/vring tags need slot patching
        bool has_opt_mme = false;             // `main` has opt_id bits 1..Bp-1 on d_qk16 / d_pvb
        bool has_head = false;                // `main` ends with batched head (SRF tags 1=active_B, 2=skip_softcap)
        // Dirty tracking for `post` (b_kv4 commit plan):
        uint64_t kv_kc[8] = {}, kv_vc[8] = {};
        int32_t kv_pos[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        uint32_t kv_cap[8] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
    };
    std::vector<BAppPlan> b_app_plans_[3];  // batched execution segments (spans + kv4 commit plans) per bi
    std::vector<uint32_t> b_last_slots_[3]; // last patched ring slots per bi
    uint64_t b_last_opt_mask_[3] = {~uint64_t(0), ~uint64_t(0), ~uint64_t(0)};
    uint32_t b_head_last_B_[3] = {0, 0, 0};
    uint32_t b_head_last_skip_[3] = {0, 0, 0};
    std::vector<std::unique_ptr<DirectPlan>> b_kv_commit_pool_;
    void ensure_batch_plans(uint32_t bi);
    void append_b_embed(DirectPlan& P, uint32_t Bp, int out_sig_sob);
    void append_b_head(DirectPlan& P, int& nsob, int wait_in_sob, uint32_t wait_in_tgt,
                       uint32_t Bp, uint64_t last_addr, bool use_sram_const);
    void append_b_app(DirectPlan& P, int& opt_sob, int& nsob, int wait_in_sob, uint32_t wait_in_tgt, int out_sig_sob,
                      uint32_t Bp, uint32_t app, int hcur, uint64_t h_out, DecodeSplit split, bool use_sram_const);
    std::unique_ptr<DirectPlan> build_b_embed(uint32_t Bp);
    std::unique_ptr<DirectPlan> build_b_kv_commit();
    std::unique_ptr<DirectPlan> build_b_kv_commit_b4(uint32_t Bp);
    static constexpr int kMaxPlanAttempts = 8;
    uint64_t evict_gen_ = 0;
    std::unique_ptr<DirectPlan> build_v(uint32_t app, uint32_t T, int side, int hcur, const DeviceBuffer* out_buf, uint64_t out_off, int lim);
    std::unique_ptr<DirectPlan> build_evicting(uint32_t sf, const std::function<std::unique_ptr<DirectPlan>()>& mk,
                                               int keep_bi = -1, bool keep_verify = false);
    std::map<std::tuple<uint32_t, int, int, uint64_t, uint64_t, uint32_t>, std::unique_ptr<DirectPlan>> plans_v_;
    std::map<uint32_t, std::unique_ptr<DirectPlan>> v_embed_plans_;
    std::map<std::tuple<uint32_t, uint64_t, bool>, std::unique_ptr<DirectPlan>> v_head_plans_;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, std::unique_ptr<DirectPlan>> v_commit_plans_;
    DirectPlan& v_embed_plan(uint32_t T, uint32_t sf = 0);
    DirectPlan& v_head_plan(uint32_t T, uint64_t last_addr, bool aux_on, uint32_t sf = 0);
    DirectPlan& v_commit_plan(uint32_t T, uint32_t sf = 0);
    DirectPlan& rollback_plan(int side_out, uint32_t T, uint32_t n, uint32_t sf = 0);
    DeviceBuffer vscratch_;      // HBM far verification buffers (sf, pf)
    DeviceBuffer vscratch_act_;  // HBM backing buffer for non-far vs_ (unused when sram_scratch_ places vs_ in SRAM)
    std::map<std::tuple<int, uint32_t, uint32_t, uint32_t>, std::unique_ptr<DirectPlan>> rollback_;
    struct VScratch {
        uint64_t xa, row, q, r, sink, bt, sc, scn, p, pn, sums, pv, pvn, pvf, gated, a1, res, xn, logits, idx, rw, act, ys, p1, p2,
            p3, p4, cat, xil, msg, gr, fgv, fqkv, mrmix, ymg, grp, sf, pf;
    } vs_{};
    void prepare_constants();
    uint32_t far_rows() const;
    uint32_t far_size() const;

    void init_prefill();
    // one stage of a prefill plan for the interleaving of two applications: engine (0 MME, 1 TPC), the SOB it waits
    // on (-1: none), its parts (duration estimate in us, the SOB signalled or -1)
    // (wtgt: the wait's target count; cnt: signals per part; name: the first part's kernel / template)
    struct PStage { uint8_t eng; int wait; std::vector<std::pair<float, int>> parts; uint32_t wtgt = 0; std::vector<uint32_t> cnt; std::string name; };
    static size_t check_waits(uint32_t app, const std::vector<PStage>& meta, bool verbose);
    struct PPlanB {
        std::unique_ptr<DirectPlan> plan_b;
        std::vector<PStage> meta_b;
        uint64_t used = 0;
    };
    struct PPlan {
        std::unique_ptr<DirectPlan> plan_a;
        DirectPlan* plan_b = nullptr;  // shared from plans_pb_ (independent of far bucket / win)
        std::unique_ptr<DirectPlan> plan_a2;  // post-far Plan A2 (pgate .. rplan_dyn_tab) when fsplit (lane 0)
        std::vector<PStage> meta_a;
        const std::vector<PStage>* meta_b = nullptr;
        uint64_t used = 0;
    };
    uint64_t use_clock_ = 0;  // get_p fetches (PPlan::used: the last one of a plan)
    PPlan& get_p(uint32_t app, int slot, int hcur, uint64_t out_addr, int cs_in, int cs_out, uint32_t pos);
    void patch_b(uint32_t app, int slot, PPlan& pl, int32_t* out_cnt = nullptr);
    double ensure_bf16_staged(uint32_t layer, uint64_t mask_addr = 0, const int32_t* cnt = nullptr);
    std::unique_ptr<DirectPlan> build_pa(uint32_t app, int slot, int hcur, int cs_in, int cs_out, uint32_t pos, int lim,
                                         std::vector<PStage>* meta);
    std::unique_ptr<DirectPlan> build_pa2(uint32_t app, int slot, int hcur, int cs_in, int cs_out);
    std::unique_ptr<DirectPlan> build_pfar_blk(uint32_t ntiles);
    std::unique_ptr<DirectPlan> build_pfar_io(int slot, uint32_t h, int dir);
    double run_pfar_loop(uint32_t app, int slot, uint32_t pos, PPlan& pl);
    // Reusable slot-independent pooled in-place SRAM FlashAttention tile-block plans (lane 0, tw = 3072 on 48 MiB SRAM,
    // block sizes {1, 2, 8, 24} -> up to 73,728 keys/plan, batched via run_chain; ZERO plan rebuilds from 34K to 1M!).
    static constexpr uint32_t kPFarBlkSizes[4] = {1u, 2u, 8u, 24u};
    static constexpr uint32_t kPFarBlkPoolMax[4] = {4u, 4u, 4u, 14u};
    struct PFarBlkEntry {
        std::unique_ptr<DirectPlan> plan;
        uint32_t last_nv = 0;
    };
    std::vector<PFarBlkEntry> pfar_blk_pool_[4];
    std::unique_ptr<DirectPlan> pfar_load_[2][8];
    std::unique_ptr<DirectPlan> pfar_store_[2][8];
    uint32_t pfar_tw_ = 0;
    uint64_t pfar_sf_pf_[2] = {0, 0}, pfar_of_tmp_[2] = {0, 0};
    uint64_t pfar_o_h_ = 0, pfar_q_h_ = 0, pfar_m_h_ = 0, pfar_l_h_ = 0, pfar_fres_ = 0;
    std::unique_ptr<DirectPlan> build_pb(uint32_t app, int slot, uint64_t h_out, int cs_in, int cs_out, int lim,
                                         std::vector<PStage>* meta);
    std::vector<std::string> pnames_;  // stage names of the last build_pa / build_pb
    uint32_t pmask_ = 0;
    bool p_anchored_ = false;
    struct PScratch {
        uint64_t xa, row, q, r16, bt, kvs, vcat, sr, sw, p, pw, sums, o, of, gated, a1, res, xn, mixd, logits, idx, rw, hist, off,
            cnt, exp_base, code, xil, msg, cat, gr, fgv, zero, lbuf, lcnt, xq, gu, aq, mo;
    } ps_[2]{};
    // Plan-A on-chip SRAM overlay (only one Plan A is ever active at a time in wave execution):
    //   Full 48 MiB (tpc_barrier_in_sram=false): sr[2] [0..12), sw[2] [12..24), pw[2] [24..48); gated/a1 alias sr after SM(7).
    //   Guard <48 MiB (tpc_barrier_in_sram=true): sr[2] [0..12), sw[2] [12..24), q [24..30); gated/a1 alias sr; pw in HBM.
    uint64_t psram_sr_ = 0, psram_sw_ = 0, psram_pw_ = 0, psram_fgv_ = 0, psram_gr_ = 0;
    uint64_t psram_q_ = 0, psram_gated_ = 0, psram_a1_ = 0;
    bool psram_on_ = false;
    bool psram_pw_on_ = false;
    bool psram_extra_ = false;
    bool pfar_flash_ = false;
    uint32_t pfar_cap_ = 0;  // max far rows supported by HBM pfar_ buffer (<= max_ctx_); beyond this uses SRAM FlashAttention
    std::unique_ptr<DirectPlan> pzero_pw_;
    DeviceBuffer pscratch_[2], pconst_, pfar_, ptab_;
    std::unique_ptr<DirectDynMmeSet> pdyn_[2];
    const MmeTemplate* pbgt_t_[7] = {};
    const MmeTemplate* pbdt_t_[7] = {};
    std::vector<DirectMmeJob> pgu_b_[7], pdn_b_[7], pgu_jobs_, pdn_jobs_;
    struct PDynLast { uint32_t tgt[8] = {}; uint64_t opt_mask = ~uint64_t(0); } pdyn_last_[2];  // of the slot's last patch_b
    std::unique_ptr<HostBuffer> ptab_host_;
    bool pfar_ok_ = false;
    uint64_t ones4_ = 0;  // bf16 [4, H] ones
    void anchor();
    std::unique_ptr<Recipe> anchor_;
    DeviceBuffer anchor_buf_;
    bool bf16_cache_init_ = false;
    DeviceBuffer bf16_cache_, bf16_staging_;
    std::vector<Bf16LayerAddrs> bf16_layers_;
    uint64_t bf16_staging_w13_ = 0, bf16_staging_w2_ = 0, bf16_w13_bytes_ = 0, bf16_w2_bytes_ = 0;
    std::unique_ptr<DirectPlan> pbf16_deq_plan_[2];  // 0: Nvfp4, 1: Mxfp8
    uint32_t pbf16_deq_last_L_[2] = {UINT32_MAX, UINT32_MAX};
    uint32_t pbf16_deq_last_um_[2] = {UINT32_MAX, UINT32_MAX};
    uint64_t pbf16_deq_last_mask_[2] = {0, 0};
    int32_t bf16_stage_layer_ = -1;
    bool bf16_stage_full_ = false;
    std::array<uint8_t, 128> bf16_stage_valid_{};
    struct PLayerConst { uint64_t rsil = 0; };
    std::map<uint32_t, PLayerConst> plc_;
    std::map<std::tuple<uint32_t, int, uint64_t, uint32_t>, PPlanB> plans_pb_;
    std::map<std::tuple<uint32_t, int, uint64_t, uint32_t>, PPlan> plans_p_;
    static std::vector<uint16_t> merge_order(const std::vector<PStage>& ma, const std::vector<PStage>& mb);
    static std::vector<std::vector<uint16_t>> merge_order_candidates(const std::vector<PStage>& ma, const std::vector<PStage>& mb);
    struct WaveTuner {
        std::vector<std::vector<uint16_t>> cands;
        std::vector<double> sum_us;
        std::vector<uint32_t> cnt;
        size_t next = 0;
        int best = -1;
    };
    std::map<uint64_t, WaveTuner> wave_tuners_;
    std::map<uint64_t, std::vector<uint16_t>> orders_topo_;  // interleavings keyed by stage-topology hash (4-8 entries)
    std::map<std::pair<DirectPlan*, DirectPlan*>, std::vector<uint16_t>> orders_;  // (for G2E_INK_WAVE_PROF)

    Inkling2& m_;
    std::unique_ptr<DirectRuntime> rt_;
    std::map<std::string, MmeTemplate> mme_;
    std::map<std::string, uint64_t> isa_;
    // per layer constants: q_scale (BF16), conv taps (BF16), r_scale (split order), reversed rel_proj rows,
    // transposed natural mg_fg_t [4, 64, 2H] (looped layers)
    DeviceBuffer const_;
    struct LayerConst {
        uint64_t qsb = 0, kvtaps = 0, ataps = 0, mtaps = 0, rs = 0, projr = 0, wn3 = 0, mlp_norm = 0, router_bias = 0, mg_fg_t = 0;
    };
    std::map<uint32_t, LayerConst> lc_;
    DeviceBuffer scratch_;
    DeviceBuffer far_scratch_;  // far scores / probabilities (HBM)
    struct Scratch {
        uint64_t xa, row, q, r, sink, bt, sc, p, sums, pv, gated, a1, res, xn, logits, idx, rw, act, ys,
            p1, p2, p3, cat, xil, msg, gr, fgv, fqkv, mrmix, sf, pf, pvf, pvf2, pvf_part, ymg, p4, dmax, dres, dres2, o_acc,
            dblocks, dh0, dh1, dprev0, dprev1;
        uint64_t sf_buf[2], pf_buf[2], pvf_buf[2], dres_buf[2];
    } s_{};
    std::map<std::tuple<uint32_t, int, int, uint64_t, uint64_t, uint32_t>, std::unique_ptr<DirectPlan>> plans_;
    std::vector<DirectPlan*> queue_;
    std::map<uint32_t, std::pair<double, uint64_t>> chain_prof_;
    const DeviceBuffer* chk_out_ = nullptr;
    uint64_t chk_off_ = 0;
    bool sram_scratch_ = false;
    uint32_t mask_ = 0;  // which application kinds are on (G2E_INK_DIRECT)
    std::map<uint32_t, std::pair<double, uint64_t>> prof_;  // G2E_INK_DIRECT_PROF: per application (us, runs)
};

}  // namespace g2e
