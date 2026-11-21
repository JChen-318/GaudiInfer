// DSpine block drafter (Qwen3.8-27B-DSpine-b16) for greedy speculative decoding with Qwen35.
//
// Mirrors dspine_mlx (model.py / bridge.py / propose.py):
//  * context: the target's residual stream after layers {1,16,31,46,61} of every committed token, fc-projected
//    (5 x [H,H] slices), hidden_norm'ed, k/v-projected per draft layer (k_norm, RoPE at the token position) into a
//    per-layer K/V cache [KV, max_ctx, D]; the draft attends to the last `window` (8192) positions.
//  * block: [anchor, mask x 15] at positions pos .. pos+15, 5 Qwen3 layers (causal inside the block), the code bridge
//    after every layer, final norm.
//  * refinement tail (greedy): top-16 candidates of rows 1..15 under the target LM head, per-row K x K transition
//    scores from the vocabulary tables TG [V,R] / TSV [V,2R], best candidate per (row, sender); the chain walk
//    (15 lookups) runs on the host.
#pragma once

#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <functional>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace g2e {

class DSpine {
public:
    DSpine(const Device& device, const Stream& stream, const WeightPack& draft, const WeightPack& target,
           uint32_t max_context);
    ~DSpine();
    static constexpr uint32_t kBlock = 16;
    static constexpr uint32_t kFeatures = 5;
    uint32_t window() const { return window_; }
    const std::vector<uint32_t>& target_layers() const { return target_layers_; }
    void reset();
    // sequence slots (as Qwen35): each owns its context ring and slot positions; slot 0 = the constructor's
    uint32_t add_slot();
    void select_slot(uint32_t slot);
    // prefix cache support: the context ring caches (queued copies on the stream) and the slot positions
    uint64_t state_bytes() const;
    void save_state(const HostBuffer& dst, uint64_t off) const;
    void load_state(const HostBuffer& src, uint64_t off, std::vector<int64_t> slot_pos);
    const std::vector<int64_t>& slot_positions() const { return slot_pos_; }
    // Build (or load) every block and append recipe the context can need.
    void precompile(const std::function<void(size_t, size_t)>& progress = {});
    // Append the target features of tokens at positions [pos, pos+n) to the draft context. `feats` holds feature k of
    // row r at byte offset k * slot_stride + r * hidden * 2 (BF16).
    // Context rows pos .. pos+n-1 from the target's captured features (rows 0..n-1). Only the first `marked` rows
    // become visible (default: all); mark() makes more of them visible later (a verification's rows are
    // appended before the accepted count is known).
    // src_row: the first features row to read (a batched verification: slot b's rows start at b * kBlock)
    void append_context(const DeviceBuffer& feats, uint64_t slot_stride, uint32_t n, uint32_t pos,
                        uint32_t marked = UINT32_MAX, uint32_t src_row = 0);
    // Batched append_context for 2..kMaxBatch distinct slots (slot b reads rows b * n .. of feats): one launch when
    // no slot wraps around the ring, else falls back per slot.
    void append_context_batch(std::span<const uint32_t> slots, const DeviceBuffer& feats, uint64_t slot_stride,
                              uint32_t n, std::span<const uint32_t> pos, uint32_t marked = UINT32_MAX);
    void mark(uint32_t pos, uint32_t n);
    // Greedy draft of the block [anchor at pos, mask x 15]: the 15 proposed tokens after the anchor.
    std::vector<int32_t> draft(int32_t anchor, uint32_t pos);
    // Sampled drafting (dspine_mlx propose.sampled_table): the candidates [15,16] and the transition logits
    // table [15,16(sender),16(candidate)]; the caller samples the chain under the request's filters.
    struct SampledDraft {
        std::vector<int32_t> cand;
        std::vector<float> table;
    };
    SampledDraft draft_table(int32_t anchor, uint32_t pos);
    // Greedy drafts of several slots in one launch (the dense work batched, attention per slot over its own ring):
    // drafts[b] for slots[b] with its anchor at pos[b]; 2 <= slots <= kMaxBatch, distinct.
    static constexpr uint32_t kMaxBatch = 8;
    std::vector<std::vector<int32_t>> draft_batch(std::span<const uint32_t> slots, std::span<const int32_t> anchors,
                                                  std::span<const uint32_t> pos);
    // draft_batch for greedy and sampled slots together: per slot the greedy chain and, with `tables`, the sampled
    // draft's candidates and transition table (as draft_table)
    struct BatchDraft {
        std::vector<int32_t> drafts;
        SampledDraft table;
    };
    std::vector<BatchDraft> draft_batch_full(std::span<const uint32_t> slots, std::span<const int32_t> anchors,
                                             std::span<const uint32_t> pos, bool tables);
    double last_device_ms() const { return last_ms_; }

private:
    Recipe& append_recipe(uint32_t n);
    Recipe& append_batch_recipe(uint32_t B, uint32_t n);
    Recipe& block_recipe(bool sampled);
    Recipe& block_batch_recipe(uint32_t B, bool sampled);  // draft_batch: B slots x kBlock rows (+ tables)
    DeviceBuffer ctlb_, append_rpos_;         // draft_batch controls, append_context_batch RoPE positions
    std::map<uint32_t, DeviceBuffer> topkb_;  // Ops::topk2 constants for B x (kBlock - 1) rows
    void run_block(int32_t anchor, uint32_t pos, bool sampled);  // results in staging_ (cand at 0, idx/table at 4096)
    void build_tables();
    uint32_t bucket_for(uint32_t rows) const;

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;   // draft
    const WeightPack& tw_;  // target (embeddings, LM head)
    uint32_t max_ctx_, layers_, hidden_, heads_, kv_heads_, head_dim_, inter_, vocab_, window_, code_, rank_;
    uint32_t mask_token_;
    float eps_, bridge_eps_;
    double theta_;
    std::vector<uint32_t> target_layers_;
    std::map<std::string, std::unique_ptr<Recipe>> recipes_;
    // per draft layer: a ring of `ring_` context rows [KV, ring, D]; position p lives in slot p % ring
    std::vector<DeviceBuffer> k_cache_, v_cache_;
    uint32_t ring_ = 0;
    std::vector<int64_t> slot_pos_;  // position held by each slot (-1: empty)
    struct Seq {
        std::vector<DeviceBuffer> k, v;
        std::vector<int64_t> slot_pos;
    };
    std::vector<Seq> seqs_ = std::vector<Seq>(1);  // seqs_[active_] is empty: its state is in the members
    uint32_t active_ = 0;
    DeviceBuffer tg_, tsv_, wtw_, qkv_w_, gu_w_, ctl_, out_idx_;  // out_idx_: cand at 0, idx / table at 4096
    DeviceBuffer topk_aux_;                 // Ops::topk2 constants
    // the target's lm_head quantised per row to FP8 (candidate logits only; the candidates are scored with the
    // BF16 rows); G2E_DRAFT_LM_BF16=1 keeps the BF16 product
    DeviceBuffer lm8_, lm8s_;
    bool lm8_on_ = false;
    void quantize_lm_head();
    // the layers' projections quantised per row to FP8 at load (opt-in, G2E_DRAFT_FP8=1): a copy
    // "<weight>.f8" [N,K] + "<weight>.s8" [N] per projection in w8_
    DeviceBuffer w8_;
    std::map<std::string, uint64_t> w8_off_;
    bool w8_on_ = false;
    void quantize_layers();
    void bind_proj(std::vector<Binding>& b, const std::string& weight) const;
    std::vector<Binding> block_bind_[2];
    static constexpr uint32_t kTopkChunk = 128;
    DeviceBuffer cos_tab_, sin_tab_;    // RoPE rows for every position [max_ctx + kBlock, D/2] f32
    DeviceBuffer causal_;               // [kBlock, kBlock] 0 / -1e30
    std::unique_ptr<Event> ev0_, ev1_;  // around a block launch (device time)    // the block recipe's bindings never change (ring cache)
    HostBuffer staging_;
    double last_ms_ = 0;
    bool zeroed_ = false;
};

}  // namespace g2e
