// Inkling-2-Mini (model_type inkling2_loop) on g2e graphs.
//
// 24 stored layers, 48 layer applications per token: layers 0..5, then layers 6..17 three times (passes), then
// 18..23. Every 6th layer (5, 11, 17, 23) is a full-attention layer, the others use a 1024-token sliding window.
// The residual stream is Block Attention Residuals: `h` is the partial sum of the current 6-layer block and
// `blocks` holds the completed block sums [embed, block0, block1, block2]; every sublayer input is a learned depth
// attention over (blocks, partial). Passes 2 and 3 restart from the pre-loop state and receive messages (a depth
// attention over block1 / partial block2 of the previous pass) through per-layer message gates.
//
// Caches (per layer application, per sequence slot): full layers keep a head-major linear BF16 K/V cache
// [KV, capacity, D]; sliding layers a head-major ring [KV, 1024, D] (slot = position % 1024). Every application
// owns four short-convolution states (k, v, post-attention, post-MLP: the 3 previous inputs, BF16) on ping-pong
// sides. Attention of a full layer is split into a far part (keys older than 1024 positions: plain q.k) and a near
// part (the last 1024 keys and the new rows, with the relative position bias); both share one softmax with the
// per-token sink.
//
// Launch granularity: one recipe per layer application kind (layer type, block count, pass > 1, T, far bucket);
// weights, caches and tables are bound by address at every launch.
#pragma once

#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace g2e {

struct Inkling2Config {
    uint32_t layers = 24, hidden = 2304, vocab = 200058, embed_rows = 201024;
    uint32_t heads = 24, kv_heads = 8, head_dim = 128, d_rel = 16, rel_extent = 1024, window = 1024;
    uint32_t experts = 128, topk = 8, moe_inter = 1152, conv_kernel = 4, attn_res_block = 6;
    uint32_t loop_start = 6, loop_end = 18, passes = 3, gate_rank = 64;
    float eps = 1e-6f, qk_clip = 22.627416997969522f, rel_clip = 8.0f, log_alpha = 0.1f;
    uint32_t log_floor = 8192;
    float gate_softcap = 4.f, up_softcap = 25.f, logit_softcap = 30.f, route_scale = 8.f;
    std::vector<bool> full;  // per stored layer
    int32_t eos = 200006, image_token = 200054;
    enum class ExpertKind : uint8_t { Nvfp4, Mxfp8 };
    uint32_t pack_version = 6;
    std::vector<ExpertKind> expert_kind;  // per stored layer (pack v6 native: Nvfp4 or Mxfp8)
    bool mg_stack = false;
    static Inkling2Config from_pack(const WeightPack& pack);
    static uint32_t expert_block_size(ExpertKind k) { return k == ExpertKind::Nvfp4 ? 16u : 32u; }
    uint32_t expert_block_size(uint32_t layer) const { return expert_block_size(expert_kind.at(layer)); }
    bool is_nvfp4(uint32_t layer) const { return layer < expert_kind.size() && expert_kind[layer] == ExpertKind::Nvfp4; }
    bool is_mxfp8(uint32_t layer) const { return layer < expert_kind.size() && expert_kind[layer] == ExpertKind::Mxfp8; }
    bool has_kind(ExpertKind k) const {
        for (ExpertKind x : expert_kind)
            if (x == k) return true;
        return false;
    }
    bool has_nvfp4() const { return has_kind(ExpertKind::Nvfp4); }
    bool has_mxfp8() const { return has_kind(ExpertKind::Mxfp8); }
    uint32_t applications() const { return layers + (passes - 1) * (loop_end - loop_start); }
};

class InkDirect;

class Inkling2 {
    friend class InkDirect;
public:
    Inkling2(const Device& device, const Stream& stream, const WeightPack& weights, uint32_t max_context,
             uint32_t max_slots = 1, uint32_t init_cap = UINT32_MAX);
    ~Inkling2();
    const Inkling2Config& config() const { return cfg_; }
    uint32_t position() const { return pos_; }
    uint32_t max_context() const { return max_ctx_; }
    void reset();
    // Multi-slot dynamic KV cache management (up to max_slots <= 8): rings and conv states are preallocated in
    // contiguous [max_slots] pools (1.53 GiB total for 8 slots), while full-layer linear KV caches grow/shrink
    // dynamically per slot up to 1M tokens (32 KiB/token = 32 GiB at 1M).
    uint32_t add_slot(uint32_t capacity = 0);
    void select_slot(uint32_t slot);
    uint32_t slot() const { return active_slot_; }
    uint32_t slots() const { return uint32_t(seqs_.size()); }
    uint32_t max_slots() const { return max_slots_; }
    uint32_t capacity() const { return cap_; }
    void set_capacity(uint32_t capacity);
    uint32_t round_capacity(uint32_t rows) const;
    uint64_t kv_bytes_for(uint32_t capacity) const;
    uint64_t slot_kv(uint32_t slot) const;
    uint64_t kv_total() const;
    uint32_t slot_position(uint32_t slot) const;
    uint32_t slot_capacity(uint32_t slot) const;
    int slot_conv_side(uint32_t slot) const;
    // Test helper: jump position inside an allocated/zeroed KV cache to test >131K / 1M prefill & decode quickly.
    void set_position_for_test(uint32_t pos);
    // Appends `tokens` at the current position and returns the logits (F32 [vocab]) of the last one (empty when
    // !want_logits). Splits the tokens into supported pieces.
    std::vector<float> forward(std::span<const int32_t> tokens, bool want_logits = true);
    // Greedy convenience: argmax of the logits of forward(tokens).
    int32_t forward_argmax(std::span<const int32_t> tokens);
    // True Bp in {2, 4, 8} batched Direct decode across distinct sequence slots (mixed contexts 0..1M, mixed greedy/sampled).
    struct DecodeBatchOut {
        std::vector<int32_t> argmax;
        std::vector<std::vector<float>> logits;  // populated only for rows with want_logits[i] != 0
    };
    bool can_decode_batch() const;
    // One line for the log: the engine behind this model (Synapse graphs only, or which parts the Direct engine runs).
    std::string engine_summary() const;
    bool direct_engine() const;       // the Direct engine runs (its plans serve decode and / or prefill)
    bool direct_full_decode() const;  // a decode token runs as one Direct chain, without a Synapse launch
    // Static BF16 cache for native NVFP4 layers (G2E_INK_BF16_CACHE=auto|all|loop|none). Call explicitly after all
    // packs (vision, drafter) are loaded; kv_need_bytes is the HBM to reserve for KV caches (0: default based on
    // max_slots * max_context, capped at 1M tokens). No-op on packs without NVFP4 layers.
    void init_bf16_cache(uint64_t kv_need_bytes = 0);
    struct Bf16LayerAddrs {
        uint64_t w13 = 0;  // BF16 [E, H, 2F] (per 256 cols: [gate 128 | up 128]); 0 if uncached (must stage)
        uint64_t w2 = 0;   // BF16 [E, F, H] (natural k, natural cols); 0 if uncached (must stage)
        bool cached = false;
    };
    Bf16LayerAddrs bf16_layer(uint32_t layer) const;
    uint64_t bf16_staging_w13() const;  // 1-layer staging BF16 w13 device addr (0 if all NVFP4 layers cached)
    uint64_t bf16_staging_w2() const;   // 1-layer staging BF16 w2 device addr  (0 if all NVFP4 layers cached)
    uint64_t bf16_w13_bytes() const;    // byte size of one layer's BF16 w13
    uint64_t bf16_w2_bytes() const;     // byte size of one layer's BF16 w2
    // Download one cached native layer's BF16 w13 and w2 from HBM (verification / tests).
    void download_bf16_layer(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out) const;
    // Stage (if uncached) and download one native layer's BF16 w13 and w2 from HBM (verification / tests).
    void download_bf16_staged(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out);
    DecodeBatchOut decode_batch(std::span<const uint32_t> slots, std::span<const int32_t> tokens,
                                std::span<const uint8_t> want_logits);
    // An image's features (BF16 [rows, hidden]) replace the embeddings of the positions [pos, pos + rows) when a later
    // forward reaches them (the buffer must stay alive until then). clear_injections drops the pending ones.
    void inject_rows(const DeviceBuffer* src, uint64_t offset, uint32_t pos, uint32_t rows);
    void clear_injections() { inject_.clear(); }
    // Compiles (or loads from G2E_RECIPE_CACHE) the full layers' recipes for T tokens at every far bucket up to the
    // context, so long conversations do not stall on graph compiles; returns the number of recipes.
    size_t precompile_far(uint32_t T, bool vh = false);  // vh: the verification-mode recipes (padded tails), sliding too
    // Debug: the partial-block state after every layer application of the last forward (BF16 [T, hidden] each).
    void set_dump(bool on) { dump_ = on; }
    // Direct-engine applications on / off at run time (when built with G2E_INK_DIRECT; testing)
    void set_direct(bool on) { direct_on_ = on; }
    // Invalidate Direct on-chip SRAM constant cache and force stream sync before the next Direct launch
    // (must be called after any external Synapse recipe such as InkDSpark or InklingVision).
    void invalidate_sram();
    // DSpark drafter inputs (T = 1 forwards): per token the partial block entering layer visits 5, 23, 41 and the final
    // norm output (the LM-head input), BF16 [4, hidden], appended to aux() by every forward while on
    void set_aux_capture(bool on);
    std::vector<uint16_t>& aux() { return aux_host_; }
    // the taps of the last forward's rows: tap k of row r at taps() + k * tap_stride() + r * hidden * 2 (BF16)
    const DeviceBuffer& taps() const { return tap_slot_ ? aux1_ : aux_; }
    uint64_t tap_stride() const { return uint64_t(kMaxT) * cfg_.hidden * 2; }
    // called after every forward piece while the taps are captured (pos = the piece's first position, n rows)
    void set_tap_hook(std::function<void(uint32_t, uint32_t)> hook) { tap_hook_ = std::move(hook); }
    // Speculative verification: verify() runs tokens (<= 8) at the current position and returns the logits of every
    // row (row-major [n, vocab]) without committing anything; accept(n) then commits the first n rows (K/V rows and the
    // conv states after row n - 1) and advances the position by n.
    std::vector<float> verify(std::span<const int32_t> tokens);
    // whether a verification of T rows runs on the Direct engine for every application at the current position
    bool can_verify(uint32_t T) const;
    void accept(uint32_t n);
    uint32_t app_layer(uint32_t i) const { return apps_[i].layer; }
    void direct_profile(uint32_t app);
    void direct_profile_v(uint32_t app, uint32_t T);
    uint32_t app_pass(uint32_t i) const { return apps_[i].pass; }
    const std::vector<std::vector<uint16_t>>& dumps() const { return dump_out_; }
    double last_ms() const { return last_ms_; }
    void print_profile();  // G2E_INK_PROFILE=1: device time per recipe since the last call
    struct Timing { double embed = 0, layers = 0, head = 0, commit = 0; };
    const Timing& last_timing() const { return timing_; }
    // State snapshots (prefix reuse): the rings, the conv states and the full layers' cache rows below the position
    // (state_bytes(pos)). save_state (at the current position) / load_state queue device copies on the model's stream.
    uint64_t state_bytes(uint32_t pos) const;
    void save_state(const DeviceBuffer& dst, uint64_t off) const;
    void load_state(const DeviceBuffer& src, uint64_t off, uint32_t pos);
    void save_state(const HostBuffer& dst, uint64_t off) const;
    void load_state(const HostBuffer& src, uint64_t off, uint32_t pos);
    static constexpr uint32_t kMaxT = 1024;  // tokens per forward

private:
    struct App { uint32_t layer, pass, index; };  // a layer application (index 0..47 in execution order)
    std::vector<App> apps_;
    Recipe& layer_recipe(const App& a, uint32_t T, uint32_t Sf);
    Recipe& embed_recipe(uint32_t T);
    Recipe& head_recipe(uint32_t rows, bool with_xn = false);
    TensorId attn_res(Ops& o, const std::string& n, TensorId stack, uint32_t S, uint32_t T, TensorId q, uint32_t Q);
    TensorId short_conv(Ops& o, const std::string& n, TensorId x, TensorId state, TensorId taps, uint32_t T, uint32_t C,
                        TensorId state_out);
    TensorId dyn_conv(Ops& o, const std::string& n, TensorId x, TensorId state, TensorId taps, TensorId dyn, uint32_t T,
                      uint32_t C, TensorId state_out);
    TensorId iota_input(GraphBuilder& g);
    TensorId interleave128(Ops& o, const std::string& n, TensorId x, bool inverse);
    std::pair<TensorId, TensorId> fused_mix(Ops& o, const std::string& n, TensorId blocks, TensorId part, TensorId q, TensorId w,
                                            uint32_t T, uint32_t Q, bool want_mix, bool mix_natural = true);
    bool msg_stack_ = false;
    TensorId fused_conv(Ops& o, const std::string& n, TensorId x, TensorId state, TensorId taps, TensorId dyn, uint32_t T,
                        uint32_t C, TensorId state_out, float scale, int mode);
    TensorId message_gate(Ops& o, const std::string& n, TensorId h, TensorId m, uint32_t T, const std::string& which);
    std::pair<TensorId, TensorId> route(Ops& o, const std::string& n, TensorId xn, uint32_t T);  // idx, weights
    TensorId moe_act_c256(Ops& o, const std::string& n, TensorId gu);
    TensorId moe_bf16_small(Ops& o, const std::string& n, TensorId xn, TensorId idx, TensorId rw, uint32_t T);
    TensorId moe_bf16_cap(Ops& o, const std::string& n, TensorId xn, TensorId idx, TensorId rw, uint32_t T);
    Recipe& commit_recipe(uint32_t T);  // commit new K/V rows into rings and linear caches
    static constexpr uint32_t kBf16SmallMaxT = 4;
    void enqueue(Recipe& r, const std::vector<Binding>& all);  // binds the subset of `all` the recipe declares
    TensorId l2_clip(Ops& o, const std::string& n, TensorId x, float threshold);
    void run_forward(std::span<const int32_t> tokens, bool want_logits, std::vector<float>* logits);
    int32_t run_decode_direct(int32_t token, bool want_logits, std::vector<float>* logits, bool want_argmax);
    void run_forward_pair(std::span<const int32_t> ta, std::span<const int32_t> tb, bool want_logits, std::vector<float>* logits);
    uint32_t far_bucket(uint32_t rows) const;
    uint32_t attn_block(uint32_t T, uint32_t Sf) const;
    void build_tables(uint32_t T);
    bool aux_on_ = false;
    bool verify_ = false;
    uint32_t pad_n_ = 0;  // a padded prefill tail (verification mode): only the first pad_n_ rows are real
    bool caches_zeroed_ = false;  // reset(): the KV rings / linear caches are zeroed once
    DeviceBuffer hist_;
    uint64_t hist_app_ = 0;
    uint32_t ver_T_ = 0;
    int ver_side_ = 0;
    std::function<void(uint32_t, uint32_t)> tap_hook_;
    void enqueue_commit(uint32_t T);
    void ensure_hist();
    uint64_t hist_off(int conv, uint32_t T) const;
    Recipe& headx_recipe(uint32_t rows);
    DeviceBuffer aux_;
    std::vector<uint16_t> aux_host_;
    std::unique_ptr<InkDirect> direct_;  // Direct-engine plans (G2E_INK_DIRECT)
    bool direct_on_ = true;
    bool direct_prev_ = false;  // true after Zero-Synapse Direct decode (skips idle stream_.synchronize())
    DeviceBuffer qil_;  // interleaved depth-attention queries (prepare_queries)
    std::map<std::string, uint64_t> qil_off_;
    void prepare_queries();
    // Native expert layers (pack v6): row-scale vectors [rows, E] F32 with g[e] in every row of expert e (prepare_nv_gvec)
    DeviceBuffer nv_gvec_;
    std::map<uint32_t, uint64_t> nv_gvec13_off_, nv_gvec2_off_;
    void prepare_nv_gvec();
    uint64_t nv_gvec13(uint32_t layer) const { return nv_gvec_.address() + nv_gvec13_off_.at(layer); }  // [2F, E]
    uint64_t nv_gvec2(uint32_t layer) const { return nv_gvec_.address() + nv_gvec2_off_.at(layer); }    // [H, E]
    // Synapse-only load-time BF16 expansion of native expert layers (used when Direct engine is off)
    DeviceBuffer syn_bf16_;
    struct SynBf16Off { uint64_t w13 = 0, w2 = 0; bool valid = false; };
    std::vector<SynBf16Off> syn_bf16_off_;
    uint64_t syn_bf16_w13_bytes_ = 0, syn_bf16_w2_bytes_ = 0;
    void prepare_synapse_bf16();
    void bind_bf16_experts(uint32_t layer, std::vector<Binding>& b) const;

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    Inkling2Config cfg_;
    uint32_t max_ctx_, max_slots_ = 1, pos_ = 0, cap_ = 0;
    uint32_t active_slot_ = 0;
    struct Seq {
        std::vector<DeviceBuffer> kcache, vcache;
        uint32_t pos = 0, cap = 0;
        int conv_side = 0;
        uint32_t ver_T = 0;
        int ver_side = 0;
        bool caches_zeroed = false;
    };
    std::vector<Seq> seqs_;  // seqs_[active_slot_] is empty while its state lives in the active members
    void swap_seq(Seq& s);
    // per application: KV caches (full, dynamic per slot) or rings (contiguous [max_slots]); conv_pool_ [max_slots][2][apps]
    std::vector<int> kv_index_;   // application -> full-cache index or -1
    std::vector<int> ring_index_; // application -> ring index or -1
    std::vector<DeviceBuffer> kcache_, vcache_, kring_, vring_;
    uint64_t ring_slot_bytes_ = 0;
    DeviceBuffer conv_pool_;      // [max_slots][2 sides][apps][conv_app_bytes] bf16
    uint64_t conv_app_bytes_ = 0;
    uint64_t conv_slot_bytes_ = 0;
    uint64_t conv_off_[4] = {0, 0, 0, 0};
    int conv_side_ = 0;
    uint64_t ring_off(uint32_t slot) const { return uint64_t(slot) * ring_slot_bytes_; }
    uint64_t conv_off_slot(uint32_t slot, int side) const { return (uint64_t(slot) * 2 + uint64_t(side)) * conv_slot_bytes_; }
    uint64_t kring_addr(uint32_t app, uint32_t slot) const { return kring_[ring_index_[app]].address() + ring_off(slot); }
    uint64_t vring_addr(uint32_t app, uint32_t slot) const { return vring_[ring_index_[app]].address() + ring_off(slot); }
    uint64_t conv_addr(uint32_t app, uint32_t slot, int side) const {
        return conv_pool_.address() + conv_off_slot(slot, side) + uint64_t(app) * conv_app_bytes_;
    }
    uint64_t kcache_addr(uint32_t app, uint32_t slot) const {
        const int idx = kv_index_[app];
        return (slot == active_slot_ ? kcache_[idx] : seqs_[slot].kcache[idx]).address();
    }
    uint64_t vcache_addr(uint32_t app, uint32_t slot) const {
        const int idx = kv_index_[app];
        return (slot == active_slot_ ? vcache_[idx] : seqs_[slot].vcache[idx]).address();
    }
    // activations
    DeviceBuffer blocks_, h_[2], prev_[2], kvscratch_, tables_, logits_, ids_;
    // the second chunk slot of the Direct prefill (two chunks run interleaved): mirrors of the activations
    DeviceBuffer blocks1_, h1_[2], prev1_[2], aux1_, ids1_;
    int tap_slot_ = 0;  // which slot's taps taps() returns (the tap hook of a chunk)
    HostBuffer staging_;
    std::map<std::string, std::unique_ptr<Recipe>> recipes_;
    std::map<std::string, TensorId> declared_;  // weight inputs of the recipe being built
    struct Inject { const DeviceBuffer* src; uint64_t off; uint32_t row, rows; };
    std::vector<Inject> inject_;
    bool dump_ = false;
    std::vector<std::vector<uint16_t>> dump_out_;
    double last_ms_ = 0;
    Timing timing_;
    std::map<std::string, std::pair<uint64_t, double>> profile_;
    // table layout inside tables_ (offsets, rebuilt per forward for its T)
    struct Tables { uint64_t swa_idx, swa_mask, gn_idx, gn_valid, gn_mask, gf_mask, tau, rows, slot_tok, slot_tok2, rel_rev, pos32, bytes; } tab_{};
};

}  // namespace g2e
