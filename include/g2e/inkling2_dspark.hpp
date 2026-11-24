// DSpark block drafter for Inkling-2 (mtp.safetensors -> tools/convert_dspark.py), after vLLM's implementation
// section 5 and tools/dspark_ref.py (CPU reference):
//  * context: per committed row p the target's taps (partial block entering visits 5, 23, 41 and the final-norm output)
//    are mixed per draft layer (alpha row l, RMSNorm, dflare_norm) and k/v-projected (k_norm_ctx, neox RoPE at p) into
//    per-layer position-indexed caches [cap, KV, D];
//  * block: [anchor (target embed + embed_norm), mask_embed x 7] at positions apos .. apos + 7, 3 pre-norm layers whose
//    attention sees the context (layers 0 and 2: the 1024 rows before the block; layer 1: all) and the whole block
//    (non-causal), final norm;
//  * Markov head: base logits of rows 1..7 under the target LM head, then 7 sequential GRU-like steps (previous token's
//    w1 row, folded LM-head bias of the latent), argmax on the device.
#pragma once

#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <map>
#include <memory>
#include <random>
#include <vector>

namespace g2e {

class InkDSpark {
public:
    InkDSpark(const Device& device, const Stream& stream, const WeightPack& draft, const WeightPack& target, uint32_t max_context);
    ~InkDSpark();
    static constexpr uint32_t kBlock = 8, kTaps = 4, kSampleK = 64, kTopkChunk = 256;
    uint32_t max_context() const { return max_ctx_; }
    void reset();
    // Multi-slot KV caches (lazy per-slot allocation so 8 concurrent requests don't waste HBM past max_ctx_).
    uint32_t add_slot(bool alloc = false);
    void select_slot(uint32_t slot);
    uint32_t slot() const { return active_slot_; }
    uint32_t slots() const { return uint32_t(slots_.size()); }
    bool slot_allocated(uint32_t slot) const;
    void set_slot_allocated(uint32_t slot, bool on);
    uint64_t slot_bytes() const { return uint64_t(cap_) * KV_ * D_ * 2 * 6; }
    // Taps are tap-major: tap k of row r at feats + off + k * tap_stride + r * hidden * 2 (BF16; Inkling2::taps()).
    // context rows [pos, pos + n) from rows 0 .. n - 1 of the taps at off
    void append(const DeviceBuffer& feats, uint64_t off, uint64_t tap_stride, uint32_t n, uint32_t pos);
    // ndrafts (1..7, default 7) greedy drafts after the anchor at position apos (context rows [0, apos) appended);
    // off = the taps row of position apos - 1
    std::vector<int32_t> draft(int32_t anchor, uint32_t apos, const DeviceBuffer& feats, uint64_t off, uint64_t tap_stride,
                               uint32_t ndrafts = kBlock - 1);
    struct SampledDraft {
        std::vector<int32_t> drafts;    // [ndrafts] sampled token ids
        std::vector<int32_t> cand_ids;  // [ndrafts * k] top-k candidate token ids per step
        std::vector<float> cand_q;      // [ndrafts * k] exact normalized draft prob q_i used to sample drafts[i]
        uint32_t k = kSampleK;
    };
    // ndrafts (1..7, default 3) sampled drafts + exact normalized top-k distributions for lossless rejection sampling
    SampledDraft draft_sampled(int32_t anchor, uint32_t apos, const DeviceBuffer& feats, uint64_t off,
                               uint64_t tap_stride, float temperature, int top_k, float top_p, float min_p,
                               std::mt19937_64& rng, uint32_t ndrafts = 3);
    double last_ms() const { return last_ms_; }
    // context rows [0, n) of the caches (prefix snapshots): bytes, copies to / from a device buffer (queued)
    uint64_t rows_bytes(uint32_t n) const { return uint64_t(n) * KV_ * D_ * 2 * 6; }
    void save_rows(const DeviceBuffer& dst, uint64_t off, uint32_t n) const;
    void load_rows(const DeviceBuffer& src, uint64_t off, uint32_t n);
    void save_rows(const HostBuffer& dst, uint64_t off, uint32_t n) const;
    void load_rows(const HostBuffer& src, uint64_t off, uint32_t n);
    // compile the 3-draft greedy and sampled block recipes of every context bucket up to max_context and append pieces
    void precompile();

private:
    Recipe& append_recipe(uint32_t n);
    Recipe& block_recipe(uint32_t s1, uint32_t ndrafts = kBlock - 1);
    Recipe& block_sampled_recipe(uint32_t s1, uint32_t ndrafts = 3);
    uint32_t bucket(uint32_t rows) const;
    struct SlotKv {
        std::vector<DeviceBuffer> kc, vc;
    };
    void alloc_kv(std::vector<DeviceBuffer>& kc, std::vector<DeviceBuffer>& vc);

    const Device& dev_;
    const Stream& stream_;
    const WeightPack& w_;
    const WeightPack& tw_;
    uint32_t max_ctx_, cap_;
    uint32_t H_ = 2304, NH_ = 24, KV_ = 8, D_ = 128, F_ = 9216, V_ = 200058, Vfull_ = 201024, Vpad_ = 200192, window_ = 1024;
    float eps_ = 1e-6f, theta_ = 1e6f;
    float alpha_[4][4] = {};
    std::vector<DeviceBuffer> kc_, vc_;
    std::vector<SlotKv> slots_;
    uint32_t active_slot_ = 0;
    DeviceBuffer cos_, sin_, masks_, ctl_, out_, topk2_consts_, sample_aux_;
    HostBuffer staging_;
    std::map<std::string, std::unique_ptr<Recipe>> recipes_;
    double last_ms_ = 0;
};

}  // namespace g2e
