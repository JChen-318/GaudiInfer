// Prefix cache across requests: the state a token prefix leaves in the engine, kept in host memory (and spilled to
// disk), restored instead of prefilled again.
//
// Qwen3.5 is hybrid: the attention layers keep K/V rows per position, the GDN layers one recurrent state for the
// whole prefix. So a prefix can only be resumed where a state snapshot was taken:
//   - KV chunks: 2048 rows of every attention layer (128 MiB), keyed by the chain hash of their tokens and shared by
//     every prefix that starts with them;
//   - snapshots at position n: the GDN conv / ssm states, the drafter's context ring, the KV rows past the last
//     whole chunk, keyed by the hash of tokens [0, n). Taken at the end of every request (the conversation so far)
//     and every kSnapEvery tokens of a long prefill (a shared document followed by different questions).
// restore(prompt) picks the longest snapshot that is a proper prefix of the prompt and whose chunks are all cached,
// loads it and returns its length; the caller prefills the rest. Copies run on the model's stream, so they are
// ordered with the forwards and need no synchronisation of their own; eviction to disk waits for the copy event.
#pragma once

#include "g2e/dspine.hpp"
#include "g2e/qwen35.hpp"
#include "g2e/runtime.hpp"

#include <condition_variable>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace g2e {

class PrefixCache {
public:
    static constexpr uint32_t kChunk = 2048;       // KV rows per chunk
    static constexpr uint32_t kSnapEvery = 16384;  // snapshot spacing inside a long prefill
    static constexpr uint32_t kMinTokens = 1024;   // shorter prefixes are prefilled again (cheap)

    // host_bytes: budget of host memory; disk_dir empty = no disk tier
    PrefixCache(const Device& dev, Qwen35& model, DSpine* drafter, uint64_t host_bytes, std::string disk_dir,
                uint64_t disk_bytes);
    ~PrefixCache();

    // the longest cached proper prefix of `prompt` loaded into the model (and drafter), after a reset; its length or
    // 0 (nothing loaded, the model untouched)
    uint32_t restore(const std::vector<int32_t>& prompt);
    // the engine holds the state of `tokens` (positions 0 .. n-1): cache the KV chunks not cached yet and a
    // snapshot at n (the KV rows of an open chunk included)
    void save(const std::vector<int32_t>& tokens);
    // a snapshot only (no KV chunks: their rows stay in the caches until save() of the request copies them)
    void snapshot(const std::vector<int32_t>& tokens, uint32_t n);
    // hint the background pinner to keep at least `count` chunk buffers pinned ahead (for a long prefill in flight)
    void prepare_chunks(size_t count);

    struct Stats {
        uint64_t host_bytes = 0, disk_bytes = 0;
        size_t chunks = 0, snapshots = 0;
        uint64_t hits = 0, misses = 0, restored_tokens = 0;
        double alloc_ms = 0;      // pinned host allocations on the caller's thread (no spare at hand)
        uint64_t spare_bytes = 0;  // pinned ahead by the background thread, not holding anything yet
    };
    Stats stats() const;

private:
    struct Item {  // a KV chunk or a snapshot
        bool snapshot = false;
        uint64_t key = 0;
        uint32_t pos = 0;                        // snapshot: n; chunk: first row
        std::vector<uint64_t> chunks;            // snapshot: the keys of its whole chunks, in order
        std::vector<int64_t> slot_pos;           // snapshot: drafter slot positions
        uint64_t bytes = 0;
        std::unique_ptr<HostBuffer> host;        // resident copy (null when only on disk)
        std::string file;                        // disk copy ("" none)
        std::list<uint64_t>::iterator lru;       // position in lru_ (front = most recent)
    };
    // layout of a snapshot's host buffer: [model state][drafter state][tail KV rows]
    uint64_t snapshot_bytes(uint32_t tail_rows) const;
    static uint64_t chain(uint64_t h, const int32_t* t, size_t n);
    uint64_t item_id(bool snap, uint64_t key) const { return key ^ (snap ? 0x9e3779b97f4a7c15ull : 0); }
    Item* find(bool snap, uint64_t key);
    void touch(Item& it);
    bool resident(Item& it);  // reload from disk when needed; false when lost
    void make_room(uint64_t bytes);
    void spill(Item& it);
    void drop(uint64_t id);
    // Pinned host buffers come in two sizes, a chunk and the largest snapshot, and are recycled: pinning is slow
    // (~2.3 GB/s once the kernel must reclaim page cache, 0.1-0.8 s per request end), so a background thread keeps
    // spares of both pinned ahead and the request path only takes one.
    std::unique_ptr<HostBuffer> alloc(bool snapshot);
    void recycle(bool snapshot, std::unique_ptr<HostBuffer> buf);
    uint64_t held(const Item& it) const { return class_bytes_[it.snapshot]; }
    void pinner();
    void insert(std::unique_ptr<Item> it);

    const Device& dev_;
    Qwen35& model_;
    DSpine* drafter_;
    uint64_t host_budget_, disk_budget_;
    std::string disk_dir_;
    std::unordered_map<uint64_t, std::unique_ptr<Item>> items_;  // by item_id
    std::list<uint64_t> lru_;
    uint64_t class_bytes_[2] = {0, 0};                       // chunk, snapshot (with a full tail)
    static constexpr size_t kSpares[2] = {32, 8};             // kept pinned ahead (4 GiB of chunks, ~2.2 GiB of snapshots)
    size_t spare_target_[2] = {kSpares[0], kSpares[1]};       // raised on demand for long prefills (65K-262K)
    std::vector<std::unique_ptr<HostBuffer>> spare_[2];
    mutable std::mutex spare_mu_;
    std::condition_variable spare_cv_;
    bool stop_ = false;
    std::thread pinner_;
    Stats stats_;
};

}  // namespace g2e
