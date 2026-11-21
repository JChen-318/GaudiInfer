#include "g2e/prefix_cache.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace g2e {

namespace {
constexpr uint64_t kSeed = 0x6a09e667f3bcc909ull;
uint64_t mix(uint64_t x) {  // splitmix64 finaliser
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
}  // namespace

PrefixCache::PrefixCache(const Device& dev, Qwen35& model, DSpine* drafter, uint64_t host_bytes, std::string disk_dir,
                         uint64_t disk_bytes)
    : dev_(dev), model_(model), drafter_(drafter), host_budget_(host_bytes), disk_budget_(disk_bytes),
      disk_dir_(std::move(disk_dir)) {
    class_bytes_[0] = uint64_t(kChunk) * model_.kv_row_bytes();
    class_bytes_[1] = snapshot_bytes(kChunk - 1);
    pinner_ = std::thread([this] { pinner(); });
    if (!disk_dir_.empty()) {
        std::filesystem::create_directories(disk_dir_);
        for (const auto& e : std::filesystem::directory_iterator(disk_dir_))  // files of an earlier process
            if (e.path().extension() == ".g2ekv") std::filesystem::remove(e.path());
    }
}

PrefixCache::~PrefixCache() {
    {
        std::lock_guard lk(spare_mu_);
        stop_ = true;
    }
    spare_cv_.notify_all();
    pinner_.join();
    model_.stream().synchronize();
    for (auto& [id, it] : items_)
        if (!it->file.empty()) std::filesystem::remove(it->file);
}

uint64_t PrefixCache::chain(uint64_t h, const int32_t* t, size_t n) {
    for (size_t i = 0; i < n; ++i) h = mix(h ^ (uint64_t(uint32_t(t[i])) + 0x9e3779b97f4a7c15ull));
    return h;
}

uint64_t PrefixCache::snapshot_bytes(uint32_t tail_rows) const {
    return model_.state_bytes() + (drafter_ ? drafter_->state_bytes() : 0) + uint64_t(tail_rows) * model_.kv_row_bytes();
}

PrefixCache::Item* PrefixCache::find(bool snap, uint64_t key) {
    const auto it = items_.find(item_id(snap, key));
    return it == items_.end() ? nullptr : it->second.get();
}

void PrefixCache::touch(Item& it) {
    lru_.erase(it.lru);
    lru_.push_front(item_id(it.snapshot, it.key));
    it.lru = lru_.begin();
}

void PrefixCache::pinner() {
    std::unique_lock lk(spare_mu_);
    while (!stop_) {
        int k = -1;
        bool trim = false;
        for (int c = 0; c < 2 && k < 0; ++c)
            if (spare_[c].size() < spare_target_[c]) k = c;
        for (int c = 0; c < 2 && k < 0; ++c) {  // evictions of the other kind piled up: unpin the excess
            const size_t hi = std::max(2 * kSpares[c], spare_target_[c] + kSpares[c]);
            if (spare_[c].size() > hi) k = c, trim = true;
        }
        if (k < 0) {
            spare_cv_.wait_for(lk, std::chrono::milliseconds(200));
            continue;
        }
        if (trim) {
            auto b = std::move(spare_[k].back());
            spare_[k].pop_back();
            lk.unlock();
            b.reset();
            lk.lock();
        } else {
            lk.unlock();
            std::unique_ptr<HostBuffer> b;
            try {
                b = std::make_unique<HostBuffer>(dev_, class_bytes_[k]);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[g2e] prefix cache: pinning ahead failed: %s\n", e.what());
            }
            lk.lock();
            if (!b) {
                spare_cv_.wait_for(lk, std::chrono::seconds(5));
                continue;
            }
            spare_[k].push_back(std::move(b));
        }
    }
}

void PrefixCache::recycle(bool snapshot, std::unique_ptr<HostBuffer> buf) {
    if (!buf) return;
    {
        std::lock_guard lk(spare_mu_);
        spare_[snapshot].push_back(std::move(buf));
    }
    spare_cv_.notify_one();
}

std::unique_ptr<HostBuffer> PrefixCache::alloc(bool snapshot) {
    {
        std::unique_lock lk(spare_mu_);
        if (!spare_[snapshot].empty()) {
            auto b = std::move(spare_[snapshot].back());
            spare_[snapshot].pop_back();
            lk.unlock();
            spare_cv_.notify_one();
            return b;
        }
    }
    spare_cv_.notify_one();
    const auto t0 = std::chrono::steady_clock::now();
    auto b = std::make_unique<HostBuffer>(dev_, class_bytes_[snapshot]);
    stats_.alloc_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return b;
}

void PrefixCache::spill(Item& it) {  // the stream has been synchronised by the caller
    if (it.file.empty()) {
        char name[64];
        std::snprintf(name, sizeof(name), "/%016llx.g2ekv", (unsigned long long)item_id(it.snapshot, it.key));
        const std::string path = disk_dir_ + name;
        FILE* f = std::fopen(path.c_str(), "wb");
        const bool ok = f && std::fwrite(it.host->data(), 1, it.bytes, f) == it.bytes;
        if (f) std::fclose(f);
        if (!ok) {
            std::filesystem::remove(path);
            return;  // stays resident only; the caller drops it
        }
        it.file = path;
        stats_.disk_bytes += it.bytes;
    }
    stats_.host_bytes -= held(it);
    recycle(it.snapshot, std::move(it.host));
}

void PrefixCache::drop(uint64_t id) {
    const auto f = items_.find(id);
    if (f == items_.end()) return;
    Item& it = *f->second;
    if (it.host) {
        stats_.host_bytes -= held(it);
        recycle(it.snapshot, std::move(it.host));
    }
    if (!it.file.empty()) {
        std::filesystem::remove(it.file);
        stats_.disk_bytes -= it.bytes;
    }
    (it.snapshot ? stats_.snapshots : stats_.chunks)--;
    lru_.erase(it.lru);
    items_.erase(f);
}

void PrefixCache::make_room(uint64_t bytes) {
    if (stats_.host_bytes + bytes <= host_budget_) return;
    model_.stream().synchronize();  // no copy may still read or write a buffer that is spilled or freed
    const std::vector<uint64_t> oldest(lru_.rbegin(), lru_.rend());  // least recently used first
    for (const uint64_t id : oldest) {
        if (stats_.host_bytes + bytes <= host_budget_) break;
        Item& it = *items_.at(id);
        if (!it.host) continue;
        if (!disk_dir_.empty() && stats_.disk_bytes + (it.file.empty() ? it.bytes : 0) <= disk_budget_) spill(it);
        if (it.host) drop(id);  // not spilled: gone
    }
    for (const uint64_t id : oldest) {  // the disk tier over its budget: its oldest files go
        if (stats_.disk_bytes <= disk_budget_) break;
        const auto f = items_.find(id);
        if (f != items_.end() && !f->second->host && !f->second->file.empty()) drop(id);
    }
}

bool PrefixCache::resident(Item& it) {
    if (it.host) return true;
    if (it.file.empty()) return false;
    make_room(held(it));
    auto buf = alloc(it.snapshot);
    FILE* f = std::fopen(it.file.c_str(), "rb");
    const bool ok = f && std::fread(buf->data(), 1, it.bytes, f) == it.bytes;
    if (f) std::fclose(f);
    if (!ok) return false;
    it.host = std::move(buf);
    stats_.host_bytes += held(it);
    return true;
}

void PrefixCache::insert(std::unique_ptr<Item> it) {
    const uint64_t id = item_id(it->snapshot, it->key);
    lru_.push_front(id);
    it->lru = lru_.begin();
    stats_.host_bytes += held(*it);
    (it->snapshot ? stats_.snapshots : stats_.chunks)++;
    items_[id] = std::move(it);
}

void PrefixCache::snapshot(const std::vector<int32_t>& tokens, uint32_t n) {
    if (n < kMinTokens || n > tokens.size() || model_.position() != n) return;
    const uint32_t whole = n / kChunk, tail = n - whole * kChunk;
    std::vector<uint64_t> keys;
    uint64_t h = kSeed;
    for (uint32_t c = 0; c < whole; ++c) keys.push_back(h = chain(h, tokens.data() + uint64_t(c) * kChunk, kChunk));
    const uint64_t key = mix(chain(h, tokens.data() + uint64_t(whole) * kChunk, tail) ^ n);
    if (Item* s = find(true, key)) {
        touch(*s);
        return;
    }
    auto it = std::make_unique<Item>();
    it->snapshot = true;
    it->key = key;
    it->pos = n;
    it->chunks = std::move(keys);
    it->bytes = snapshot_bytes(tail);
    make_room(class_bytes_[1]);
    it->host = alloc(true);
    uint64_t off = 0;
    model_.save_state(*it->host, off);
    off += model_.state_bytes();
    if (drafter_) {
        drafter_->save_state(*it->host, off);
        off += drafter_->state_bytes();
        it->slot_pos = drafter_->slot_positions();
    }
    if (tail) model_.save_kv(whole * kChunk, tail, *it->host, off);
    insert(std::move(it));
}

void PrefixCache::prepare_chunks(size_t count) {
    if (!class_bytes_[0] || !host_budget_) return;
    const size_t cap = std::min<size_t>(128, size_t(host_budget_ / class_bytes_[0]));
    const size_t want = std::min(std::max(count, kSpares[0]), cap);
    {
        std::lock_guard lk(spare_mu_);
        if (want <= spare_target_[0]) return;
        spare_target_[0] = want;
    }
    spare_cv_.notify_one();
}

void PrefixCache::save(const std::vector<int32_t>& tokens) {
    const uint32_t n = uint32_t(tokens.size());
    if (n < kMinTokens || model_.position() != n) return;
    const uint64_t chunk_bytes = uint64_t(kChunk) * model_.kv_row_bytes();
    uint64_t h = kSeed;
    std::vector<std::pair<uint32_t, uint64_t>> missing;
    for (uint32_t c = 0; c < n / kChunk; ++c) {
        h = chain(h, tokens.data() + uint64_t(c) * kChunk, kChunk);
        if (Item* k = find(false, h)) touch(*k);
        else missing.emplace_back(c, h);
    }
    // Make room once upfront for the whole batch (at most one stream sync and recycled chunk buffers ready in spare_[0]).
    const uint64_t total_need = uint64_t(missing.size()) * class_bytes_[0] + class_bytes_[1];
    const bool batch_evict = total_need <= host_budget_;
    if (batch_evict) make_room(total_need);
    for (const auto& [c, kh] : missing) {
        auto it = std::make_unique<Item>();
        it->key = kh;
        it->pos = c * kChunk;
        it->bytes = chunk_bytes;
        if (!batch_evict) make_room(class_bytes_[0]);
        it->host = alloc(false);
        model_.save_kv(c * kChunk, kChunk, *it->host, 0);
        insert(std::move(it));
    }
    snapshot(tokens, n);
    {
        std::lock_guard lk(spare_mu_);
        spare_target_[0] = kSpares[0];
    }
}

uint32_t PrefixCache::restore(const std::vector<int32_t>& prompt) {
    const uint32_t P = uint32_t(prompt.size());
    // the prompt's chunk keys, then the longest snapshot that is a proper prefix with every chunk cached
    std::vector<uint64_t> keys;
    uint64_t h = kSeed;
    for (uint32_t c = 0; c < P / kChunk; ++c) keys.push_back(h = chain(h, prompt.data() + uint64_t(c) * kChunk, kChunk));
    Item* best = nullptr;
    for (auto& [id, sp] : items_) {
        Item& s = *sp;
        if (!s.snapshot || s.pos >= P || (best && s.pos <= best->pos)) continue;
        const uint32_t whole = s.pos / kChunk, tail = s.pos - whole * kChunk;
        const uint64_t base = whole ? keys[whole - 1] : kSeed;
        if (mix(chain(base, prompt.data() + uint64_t(whole) * kChunk, tail) ^ s.pos) != s.key) continue;
        bool all = true;
        for (uint32_t c = 0; c < whole && all; ++c) all = s.chunks[c] == keys[c] && find(false, keys[c]);
        if (all) best = &s;
    }
    if (!best) {
        ++stats_.misses;
        return 0;
    }
    // pin the items of this restore at the front of the LRU, make them resident, then load
    const uint32_t whole = best->pos / kChunk, tail = best->pos - whole * kChunk;
    touch(*best);
    for (uint32_t c = 0; c < whole; ++c) touch(*find(false, keys[c]));
    if (!resident(*best)) {
        ++stats_.misses;
        return 0;
    }
    for (uint32_t c = 0; c < whole; ++c)
        if (!resident(*find(false, keys[c]))) {
            ++stats_.misses;
            return 0;
        }
    model_.reset();
    if (drafter_) drafter_->reset();
    for (uint32_t c = 0; c < whole; ++c) model_.load_kv(c * kChunk, kChunk, *find(false, keys[c])->host, 0);
    uint64_t off = 0;
    model_.load_state(*best->host, off, best->pos);
    off += model_.state_bytes();
    if (drafter_) {
        drafter_->load_state(*best->host, off, best->slot_pos);
        off += drafter_->state_bytes();
    }
    if (tail) model_.load_kv(whole * kChunk, tail, *best->host, off);
    ++stats_.hits;
    stats_.restored_tokens += best->pos;
    return best->pos;
}

PrefixCache::Stats PrefixCache::stats() const {
    Stats s = stats_;
    std::lock_guard lk(spare_mu_);
    for (int c = 0; c < 2; ++c) s.spare_bytes += spare_[c].size() * class_bytes_[c];
    return s;
}

}  // namespace g2e
