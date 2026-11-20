#include "g2e/qwen35.hpp"

#include <perf_lib_layer_params.h>
#include "g2e/gdn.hpp"
#include "g2e/numeric.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <map>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace g2e {

Qwen35Config Qwen35Config::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& c = j.at("config");
    Qwen35Config q;
    q.layers = j.at("layers").get<uint32_t>();
    q.hidden = c.at("hidden_size");
    q.intermediate = c.at("intermediate_size");
    q.vocab = c.at("vocab_size");
    q.heads = c.at("num_attention_heads");
    q.kv_heads = c.at("num_key_value_heads");
    q.head_dim = c.at("head_dim");
    double partial = 1.0;
    q.rope_theta = 10000.0;
    if (c.contains("rope_parameters")) {
        const auto& r = c.at("rope_parameters");
        if (r.contains("partial_rotary_factor")) partial = r.at("partial_rotary_factor");
        if (r.contains("rope_theta")) q.rope_theta = r.at("rope_theta");
    }
    if (c.contains("partial_rotary_factor")) partial = c.at("partial_rotary_factor");
    q.rotary_dim = uint32_t(q.head_dim * partial);
    q.lin_k_heads = c.at("linear_num_key_heads");
    q.lin_v_heads = c.at("linear_num_value_heads");
    q.lin_k_dim = c.at("linear_key_head_dim");
    q.lin_v_dim = c.at("linear_value_head_dim");
    q.conv_kernel = c.at("linear_conv_kernel_dim");
    q.eps = c.at("rms_norm_eps");
    if (auto rp = c.find("rope_parameters"); rp != c.end() && rp->contains("mrope_section"))
        q.mrope_section = rp->at("mrope_section").get<std::vector<uint32_t>>();
    for (const auto& k : j.at("layer_kinds")) q.attention.push_back(k.get<std::string>() == "attention");
    return q;
}

namespace {
void output(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const DType dt = d.dtype;
    const auto shape = d.shape;
    const auto out = g.input(name, dt, shape);
    g.op("identity", {src}, {out});
}
std::string L(uint32_t i, const char* part) { return "L" + std::to_string(i) + "." + part; }
std::string P(uint32_t j, const std::string& part) { return "l" + std::to_string(j) + "." + part; }
uint64_t align128(uint64_t x) { return (x + 127) & ~uint64_t(127); }
}  // namespace

Qwen35::Qwen35(const Device& device, const Stream& stream, const WeightPack& weights, uint32_t max_context,
               uint32_t block_layers)
    : dev_(device), stream_(stream), w_(weights), cfg_(Qwen35Config::from_pack(weights)), max_ctx_(max_context),
      block_(block_layers),
      staging_(device, 16u << 20) {
    if (!block_ || cfg_.layers % block_) throw std::invalid_argument("qwen35: block size must divide the layer count");
    if (max_ctx_ % 128) throw std::invalid_argument("qwen35: max_context must be a multiple of 128");
    cap_ = max_ctx_;
    kv8_ = std::getenv("G2E_KV_FP8") != nullptr;
    seqs_.resize(1);
    const uint64_t H = cfg_.hidden;
    for (int s = 0; s < 2; ++s) {
        res_[s] = DeviceBuffer(dev_, kMaxT * H * 2);
        delta_[s] = DeviceBuffer(dev_, kMaxT * H * 2);
    }
    zero_ = DeviceBuffer(dev_, kMaxT * H * 2);
    {   // RoPE tables: row p = cos/sin(p * theta^(-2i/rotary)), bound per forward at the first position's row
        const uint64_t R2 = cfg_.rotary_dim / 2, rows = uint64_t(max_ctx_) + kMaxT;
        if ((R2 * 4) % 128) throw std::runtime_error("qwen35: rotary/2 * 4 bytes must be a multiple of 128");
        cos_tab_ = DeviceBuffer(dev_, rows * R2 * 4);
        sin_tab_ = DeviceBuffer(dev_, rows * R2 * 4);
        const uint64_t per = staging_.size() / (R2 * 4) / 2;  // rows per staging fill (cos and sin halves)
        for (uint64_t r0 = 0; r0 < rows; r0 += per) {
            const uint64_t n = std::min(per, rows - r0);
            auto* c = staging_.as<float>();
            auto* s = c + per * R2;
            for (uint64_t r = 0; r < n; ++r)
                for (uint64_t i = 0; i < R2; ++i) {
                    const double inv = std::pow(cfg_.rope_theta, -double(2 * i) / double(cfg_.rotary_dim));
                    const double a = double(r0 + r) * inv;
                    c[r * R2 + i] = float(std::cos(a));
                    s[r * R2 + i] = float(std::sin(a));
                }
            stream_.upload(staging_, 0, cos_tab_, r0 * R2 * 4, n * R2 * 4);
            stream_.upload(staging_, per * R2 * 4, sin_tab_, r0 * R2 * 4, n * R2 * 4);
            stream_.synchronize();
        }
    }
    ids_ = DeviceBuffer(dev_, kIdsRopePos + kMaxT * 4);  // ids, posm1, last visible rows, RoPE positions
    rope_cs_ = DeviceBuffer(dev_, 2ull * kMaxT * (cfg_.rotary_dim / 2) * 4);
    quantize_lm_head();
    mrope_section_ = cfg_.mrope_section;
    logits_ = DeviceBuffer(dev_, uint64_t(cfg_.vocab) * 4);
    consts_ = DeviceBuffer(dev_, 256);
    spec_ctl_ = DeviceBuffer(dev_, 4096 * kMaxBatch);  // accept() controls, one region per batch index
    argmax_ = DeviceBuffer(dev_, kBestOff + 4096);      // cand8 [rows, kHeadCand] i32, best [rows] at kBestOff
    vtop_ = DeviceBuffer(dev_, 16384);
    vtopb_ = DeviceBuffer(dev_, kVtbEnd);  // up to 128 rows x kTopK
    tempb_ = DeviceBuffer(dev_, 512);
    {
        uint32_t ng = 0;
        for (bool a : cfg_.attention) ng += !a;
        (void)ng;  // spec_stash_ is sized after the layer scan (it needs num_gdn_)
    }
    {
        uint32_t na = 0;
        for (bool a : cfg_.attention) na += a;
        kv_scratch_ = DeviceBuffer(dev_, uint64_t(std::max(na, 1u)) * 2 * cfg_.kv_heads * kMaxT * cfg_.head_dim * 2);
        apre_q_ = DeviceBuffer(dev_, uint64_t(kMaxT) * cfg_.heads * cfg_.head_dim * 2);
        apre_g_ = DeviceBuffer(dev_, uint64_t(kMaxT) * cfg_.heads * cfg_.head_dim * 2);
        apre_r_ = DeviceBuffer(dev_, uint64_t(kMaxT) * cfg_.hidden * 2);
        att_ = DeviceBuffer(dev_, uint64_t(kMaxT) * cfg_.heads * cfg_.head_dim * 2);  // batched verification / prefill rows
    }
    const uint64_t kv_bytes = uint64_t(cfg_.kv_heads) * max_ctx_ * cfg_.head_dim * kv_es();
    const uint64_t conv_bytes = uint64_t(cfg_.conv_kernel - 1) * cfg_.conv_channels() * 2;
    const uint64_t ssm_bytes = uint64_t(cfg_.lin_v_heads) * cfg_.lin_k_dim * cfg_.lin_v_dim * 4;
    for (uint32_t i = 0; i < cfg_.layers; ++i) {
        if (cfg_.attention[i]) {
            slot_.push_back(int(k_cache_.size()));
            k_cache_.emplace_back(dev_, kv_bytes);
            v_cache_.emplace_back(dev_, kv_bytes);
        } else {
            slot_.push_back(int(num_gdn_++));
            conv_side_.push_back(0);
            ssm_side_.push_back(0);
        }
    }
    conv_bytes_ = conv_bytes;
    ssm_bytes_ = ssm_bytes;
    {
        const auto L = stash_layout(kSpecMaxT, spec_wy(kSpecMaxT));
        spec_stash_ = DeviceBuffer(dev_, L.base[5] + uint64_t(kMaxBatch) * std::max(num_gdn_, 1u) * L.size[5] + 128);
    }
    for (int s = 0; s < 2; ++s) {
        conv_all_[s] = DeviceBuffer(dev_, std::max(num_gdn_, 1u) * conv_bytes);
        ssm_all_[s] = DeviceBuffer(dev_, std::max(num_gdn_, 1u) * ssm_bytes);
    }
    prepare_aux();
    // i32 constants: 0 at offset 0, 1 at offset 128
    std::memset(staging_.data(), 0, 256);
    staging_.as<int32_t>()[32] = 1;
    stream_.upload(staging_, 0, consts_, 0, 256);
    stream_.synchronize();
    reset();
}
Qwen35::~Qwen35() = default;

// Host-prepared weights: Gemma norms as bf16(1 + w), -exp(A_log) in F32, conv taps transposed to [4, C].
void Qwen35::prepare_aux() {
    const uint64_t HV = cfg_.lin_v_heads, C = cfg_.conv_channels(), KC = cfg_.conv_kernel;
    std::vector<uint8_t> host;
    auto put = [&](const std::string& name, const void* data, uint64_t bytes) {
        const uint64_t off = align128(host.size());
        host.resize(off + bytes);
        std::memcpy(host.data() + off, data, bytes);
        aux_off_[name] = off;
    };
    auto plus_one = [&](const std::string& weight, const std::string& name) {
        const auto raw = w_.read(weight);
        std::vector<uint16_t> out(raw.size() / 2);
        for (size_t k = 0; k < out.size(); ++k) {
            uint16_t b;
            std::memcpy(&b, raw.data() + 2 * k, 2);
            out[k] = to_bf16(1.0f + from_bf16(b));
        }
        put(name, out.data(), out.size() * 2);
    };
    for (uint32_t i = 0; i < cfg_.layers; ++i) {
        plus_one(L(i, "in_norm"), L(i, "in_norm1p"));
        plus_one(L(i, "post_norm"), L(i, "post_norm1p"));
        if (!cfg_.attention[i]) {
            const auto a = w_.read(L(i, "A_log"));
            std::vector<float> neg(HV);
            for (uint64_t h = 0; h < HV; ++h) {
                uint16_t b;
                std::memcpy(&b, a.data() + 2 * h, 2);
                neg[h] = -std::exp(from_bf16(b));
            }
            put(L(i, "negexpA"), neg.data(), neg.size() * 4);
            const auto cw = w_.read(L(i, "conv"));  // [C, KC] bf16
            std::vector<uint16_t> t(C * KC);
            for (uint64_t c = 0; c < C; ++c)
                for (uint64_t k = 0; k < KC; ++k) std::memcpy(&t[k * C + c], cw.data() + 2 * (c * KC + k), 2);
            put(L(i, "conv_t"), t.data(), t.size() * 2);
        }
    }
    plus_one("final_norm", "final_norm1p");
    // attention mask pieces: column indices (f32) and the causal block of the new rows per width
    {
        std::vector<float> iota(uint64_t(max_ctx_) + kMaxT);
        for (size_t i = 0; i < iota.size(); ++i) iota[i] = float(i);
        put("iota", iota.data(), iota.size() * 4);
        put("iota_t", iota.data(), uint64_t(kMaxT) * 4);  // its own copy: persistent tensors must not overlap
        // FP8 SDPA scales (prefill attention): unit FP8 scales, probabilities x128 inside the SDPA
        const float one = 1.0f, s128 = 128.0f, d128 = 1.0f / 128.0f;
        put("f8_one", &one, 4);
        put("f8_s128", &s128, 4);
        put("f8_d128", &d128, 4);
        for (uint32_t T = 1; T <= kMaxT; T *= 2) {
            std::vector<float> c(uint64_t(T) * T);
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t u = 0; u < T; ++u) c[uint64_t(t) * T + u] = u <= t ? 0.f : -1e30f;
            put("causal_T" + std::to_string(T), c.data(), c.size() * 4);
        }
    }
    // Ops::topk2 constants of the verification top-k (per width), and of the FP8 head's candidates
    for (uint32_t B = 2; B <= 2 * kMaxBatch; ++B) {  // batched verification heads (B x 8 rows: widths 8 and 16)
        const uint32_t T = B * 8;
        if (T <= kSpecMaxT || (T & (T - 1)) == 0) continue;  // powers of two: below
        const auto c8 = Ops::topk2_constants(T, cfg_.vocab, kHeadCand, kTopkChunk);
        put("topk2c_T" + std::to_string(T), c8.data(), c8.size() * 4);
        const auto c = Ops::topk2_constants(T, cfg_.vocab, kTopK, kTopkChunk);
        put("topk2_T" + std::to_string(T), c.data(), c.size() * 4);
    }
    for (uint32_t T = 2; T <= kSpecMaxT * kMaxBatch; T *= 2) {
        const auto c8 = Ops::topk2_constants(T, cfg_.vocab, kHeadCand, kTopkChunk);
        put("topk2c_T" + std::to_string(T), c8.data(), c8.size() * 4);
        const auto c = Ops::topk2_constants(T, cfg_.vocab, kTopK, kTopkChunk);
        put("topk2_T" + std::to_string(T), c.data(), c.size() * 4);
    }
    // chunked-GDN constants for C = 8..128 (g2e/gdn.hpp)
    for (uint32_t c = 8; c <= 128; c *= 2)
        for (const auto& k : gdn_chunk_constants(c))
            put("chunk" + std::to_string(c) + "." + k.name, k.values.data(), k.values.size() * 4);
    aux_ = DeviceBuffer(dev_, align128(host.size()));
    for (uint64_t off = 0; off < host.size(); off += staging_.size()) {
        const uint64_t n = std::min<uint64_t>(staging_.size(), host.size() - off);
        std::memcpy(staging_.data(), host.data() + off, n);
        stream_.upload(staging_, 0, aux_, off, n);
        stream_.synchronize();
    }
}

uint64_t Qwen35::kv_row_bytes() const { return uint64_t(k_cache_.size()) * 2 * cfg_.kv_heads * cfg_.head_dim * kv_es(); }

void Qwen35::save_kv(uint32_t r0, uint32_t n, const HostBuffer& dst, uint64_t off) const {
    const uint64_t D = cfg_.head_dim, row = D * kv_es(), KV = cfg_.kv_heads;
    for (size_t s = 0; s < k_cache_.size(); ++s)
        for (int kv = 0; kv < 2; ++kv)
            for (uint64_t h = 0; h < KV; ++h) {
                const auto& c = kv ? v_cache_[s] : k_cache_[s];
                stream_.download(c, (h * cap_ + r0) * row, dst, off, n * row);
                off += n * row;
            }
}

void Qwen35::load_kv(uint32_t r0, uint32_t n, const HostBuffer& src, uint64_t off) {
    const uint64_t D = cfg_.head_dim, row = D * kv_es(), KV = cfg_.kv_heads;
    if (r0 + n > cap_) throw std::out_of_range("qwen35: load_kv past the context");
    for (size_t s = 0; s < k_cache_.size(); ++s)
        for (int kv = 0; kv < 2; ++kv)
            for (uint64_t h = 0; h < KV; ++h) {
                const auto& c = kv ? v_cache_[s] : k_cache_[s];
                stream_.upload(src, off, c, (h * cap_ + r0) * row, n * row);
                off += n * row;
            }
}

uint64_t Qwen35::state_bytes() const { return uint64_t(num_gdn_) * (conv_bytes_ + ssm_bytes_); }

void Qwen35::save_state(const HostBuffer& dst, uint64_t off) const {
    for (uint32_t s = 0; s < num_gdn_; ++s) {
        stream_.download(conv_all_[conv_side_[s]], uint64_t(s) * conv_bytes_, dst, off, conv_bytes_);
        off += conv_bytes_;
        stream_.download(ssm_all_[ssm_side_[s]], uint64_t(s) * ssm_bytes_, dst, off, ssm_bytes_);
        off += ssm_bytes_;
    }
}

void Qwen35::load_state(const HostBuffer& src, uint64_t off, uint32_t pos) {
    if (pos > cap_) throw std::out_of_range("qwen35: load_state past the context");
    for (uint32_t s = 0; s < num_gdn_; ++s) {
        stream_.upload(src, off, conv_all_[conv_side_[s]], uint64_t(s) * conv_bytes_, conv_bytes_);
        off += conv_bytes_;
        stream_.upload(src, off, ssm_all_[ssm_side_[s]], uint64_t(s) * ssm_bytes_, ssm_bytes_);
        off += ssm_bytes_;
    }
    pos_ = pos;
    pending_T_ = 0;
    rope_delta_ = 0;
}

void Qwen35::swap_seq(Seq& q) {
    std::swap(k_cache_, q.k);
    std::swap(v_cache_, q.v);
    for (int i = 0; i < 2; ++i) {
        std::swap(conv_all_[i], q.conv[i]);
        std::swap(ssm_all_[i], q.ssm[i]);
    }
    std::swap(conv_side_, q.conv_side);
    std::swap(ssm_side_, q.ssm_side);
    std::swap(pos_, q.pos);
    std::swap(cap_, q.cap);
    std::swap(rope_delta_, q.rope_delta);
    std::swap(pending_T_, q.pending_T);
    std::swap(stash_b_, q.stash_b);
    std::swap(pending_wy_, q.pending_wy);
}

uint32_t Qwen35::add_slot(uint32_t capacity) {
    if (capacity > max_ctx_ || capacity % 128) throw std::invalid_argument("qwen35: slot capacity");
    if (pending_T_) throw std::logic_error("qwen35: add_slot with a pending verification");
    Seq q;
    const uint64_t kv_bytes = uint64_t(cfg_.kv_heads) * capacity * cfg_.head_dim * kv_es();
    for (size_t i = 0; i < k_cache_.size(); ++i) {
        if (!capacity) {  // dynamic: allocated by set_capacity
            q.k.emplace_back();
            q.v.emplace_back();
            continue;
        }
        q.k.emplace_back(dev_, kv_bytes);
        q.v.emplace_back(dev_, kv_bytes);
        stream_.zero(q.k.back(), 0, kv_bytes);  // finite stale rows (masked); fresh memory may hold NaN patterns
        stream_.zero(q.v.back(), 0, kv_bytes);
    }
    for (int i = 0; i < 2; ++i) {
        q.conv[i] = DeviceBuffer(dev_, conv_all_[i].size());
        q.ssm[i] = DeviceBuffer(dev_, ssm_all_[i].size());
        stream_.zero(q.conv[i], 0, q.conv[i].size());
        stream_.zero(q.ssm[i], 0, q.ssm[i].size());
    }
    q.conv_side.assign(conv_side_.size(), 0);
    q.ssm_side.assign(ssm_side_.size(), 0);
    q.cap = capacity;
    stream_.synchronize();
    seqs_.push_back(std::move(q));
    return uint32_t(seqs_.size() - 1);
}

uint32_t Qwen35::round_capacity(uint32_t rows) const {
    uint32_t b = 16384;
    while (b < rows) {
        const uint32_t mid = b + b / 2;
        b = mid >= rows ? mid : b * 2;
    }
    return std::min(b, max_ctx_);
}

uint64_t Qwen35::kv_bytes_for(uint32_t capacity) const {
    return uint64_t(k_cache_.size()) * 2 * cfg_.kv_heads * uint64_t(capacity) * cfg_.head_dim * kv_es();
}

uint64_t Qwen35::slot_kv(uint32_t slot) const {
    if (slot >= seqs_.size()) throw std::out_of_range("qwen35: no such slot");
    uint64_t n = 0;
    if (slot == active_slot_) {
        for (const auto& b : k_cache_) n += b.size();
        for (const auto& b : v_cache_) n += b.size();
    } else {
        for (const auto& b : seqs_[slot].k) n += b.size();
        for (const auto& b : seqs_[slot].v) n += b.size();
    }
    return n;
}

uint64_t Qwen35::kv_total() const {
    uint64_t n = 0;
    for (uint32_t s = 0; s < seqs_.size(); ++s) n += slot_kv(s);
    return n;
}

void Qwen35::set_capacity(uint32_t capacity) {
    if (capacity > max_ctx_ || capacity % 128) throw std::invalid_argument("qwen35: slot capacity");
    if (pending_T_) throw std::logic_error("qwen35: set_capacity with a pending verification");
    if (capacity == cap_) return;
    auto release = [&] {
        stream_.synchronize();
        for (auto& b : k_cache_) b = DeviceBuffer();
        for (auto& b : v_cache_) b = DeviceBuffer();
        cap_ = 0;
        reset();
    };
    if (!capacity) {
        release();
        return;
    }
    if (capacity < pos_) throw std::invalid_argument("qwen35: capacity below the position");
    const uint64_t H = cfg_.kv_heads, D2 = uint64_t(cfg_.head_dim) * kv_es(), bytes = H * capacity * D2;
    try {  // one layer at a time: the peak extra memory is one layer's buffer
        for (auto* set : {&k_cache_, &v_cache_})
            for (auto& b : *set) {
                DeviceBuffer nb(dev_, bytes);
                stream_.zero(nb, 0, bytes);  // finite stale rows (masked); fresh memory may hold NaN patterns
                if (pos_ && b.size())
                    for (uint64_t h = 0; h < H; ++h)
                        stream_.copy(b, h * cap_ * D2, nb, h * capacity * D2, uint64_t(pos_) * D2);
                stream_.synchronize();  // before the old buffer is freed
                b = std::move(nb);
            }
    } catch (...) {
        release();  // layers of different capacities: start the slot over
        throw;
    }
    cap_ = capacity;
}

void Qwen35::select_slot(uint32_t slot) {
    if (slot >= seqs_.size()) throw std::out_of_range("qwen35: no such slot");
    if (slot == active_slot_) return;
    // a pending verification travels with its slot (its stash region is its own: stash_b_)
    swap_seq(seqs_[active_slot_]);  // park the selected slot
    swap_seq(seqs_[slot]);          // take the other one out
    active_slot_ = slot;
}

void Qwen35::reset() {
    // K/V rows past the position are masked, so stale (finite) rows are harmless: the caches are zeroed once, when
    // allocated (fresh device memory may hold NaN patterns). The recurrent states start from zero every time.
    pos_ = 0;
    pending_T_ = 0;
    rope_delta_ = 0;
    rope_explicit_ = false;
    inject_.clear();
    auto zero = [&](const DeviceBuffer& b) { stream_.zero(b, 0, b.size()); };
    if (!zeroed_) {
        zero(zero_);
        for (auto& b : k_cache_) if (b.size()) zero(b);
        for (auto& b : v_cache_) if (b.size()) zero(b);
        for (int s = 0; s < 2; ++s) {
            zero(conv_all_[s]);
            zero(ssm_all_[s]);
        }
        zeroed_ = true;
    } else {
        zero(conv_all_[0]);
        zero(ssm_all_[0]);
    }
    for (auto& s : conv_side_) s = 0;
    for (auto& s : ssm_side_) s = 0;
    stream_.synchronize();
}

// Context buckets: 128, 256, 384, 512, 768, 1024, 1536, 2048, ... (powers of two and 1.5x steps)
// Cache rows read by a split-attention forward of T tokens: past 2048 rows, verification / decoding
// (T <= kSpecMaxT) and full prefill pieces (T = kMaxT) round up to 2048-row steps instead of the 1.5x / 2x buckets
// (attention reads every bucket row: 105K would read 131072); the other prefill remainders keep the coarse buckets.
// Prefill pieces (T > kSpecMaxT) over >= 8K cache rows take the FP8 split attention (flash-style: a small workspace;
// the FP16 kernel path materialises [G*T, Sb] scores and probabilities per kv head, ~26 GB at T = 1024 / 256K).
// Its unmasked prefix [0, Sb - W) must lie before the piece: attn_rows' fine rows guarantee Sb - pos < T + 2048.
bool Qwen35::fp8_prefill(uint32_t T, uint32_t Sb) const {
    return T > kSpecMaxT && Sb >= std::max<uint32_t>(8192, T + 6144) && Sb % 2048 == 0;
}

uint32_t Qwen35::fp8_tail(uint32_t T) const { return std::max<uint32_t>(4096, T + 2048); }

void Qwen35::memory_report() const {
    const auto m = device_memory();
    uint64_t kv = 0;
    for (const auto& b : k_cache_) kv += b.size();
    for (const auto& b : v_cache_) kv += b.size();
    std::vector<std::pair<uint64_t, std::string>> ws;
    for (const auto& [name, r] : recipes_) ws.push_back({r->workspace_bytes(), name});
    std::sort(ws.rbegin(), ws.rend());
    std::fprintf(stderr, "[mem] device buffers %.2f GiB now, %.2f GiB peak; KV cache %.2f GiB (%u rows); %zu recipes\n",
                 double(m.live) / (1 << 30), double(m.peak) / (1 << 30), double(kv) / (1 << 30), max_ctx_, ws.size());
    for (size_t i = 0; i < std::min<size_t>(ws.size(), 6); ++i)
        std::fprintf(stderr, "[mem]   workspace %.2f GiB  %s\n", double(ws[i].first) / (1 << 30), ws[i].second.c_str());
}

uint32_t Qwen35::attn_rows(uint32_t rows, uint32_t T) const {
    constexpr uint32_t kFine = kAttnFine, kStep = kAttnStep;
    if (rows <= kFine) return bucket_for(rows);
    // prefill remainder pieces: coarse buckets up to 8K rows (their FP16 kernel attention); past that fine rows too,
    // which keeps Sb - pos < T + 2048 for the FP8 split attention (its unmasked prefix must lie before pos)
    if (T > kSpecMaxT && T != kMaxT && rows <= 8192) return bucket_for(rows);
    return std::min((rows + kStep - 1) / kStep * kStep, cap_);
}

uint32_t Qwen35::bucket_for(uint32_t rows) const {
    uint32_t b = 128;
    while (b < rows) {
        const uint32_t mid = b + b / 2;
        b = (mid >= rows && mid % 128 == 0) ? mid : b * 2;
    }
    return std::min(b, cap_);
}

void Qwen35::dbg(Ops& o, uint32_t j, const std::string& name, TensorId t) {
    if (building_debug_j_ != int(j)) return;
    if (!debug_points_.empty() && std::find(debug_points_.begin(), debug_points_.end(), name) == debug_points_.end())
        return;
    output(o.graph(), "dbg_" + name, o.cast(t, DType::F32, P(j, "dbgc_" + name)));
}

Qwen35::Pair Qwen35::mlp(Ops& o, uint32_t j, TensorId y, TensorId s) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden, F = cfg_.intermediate;
    const auto post = g.input(P(j, "post_norm1p"), DType::BF16, {H});
    const auto gu_w = g.input(P(j, "gu_w"), DType::F8, {2 * F, H});
    const auto gu_s = g.input(P(j, "gu_s"), DType::F32, {2 * F});
    const auto dn_w = g.input(P(j, "down_w"), DType::F8, {H, F});
    const auto dn_s = g.input(P(j, "down_s"), DType::F32, {H});
    const auto nq = o.add_rmsnorm_quant(y, s, post, cfg_.eps, P(j, "post"));
    dbg(o, j, "h1", nq.residual_out);
    dbg(o, j, "x2", nq.normalized);
    // G2E_DQ_FOLD=1: the gate_up GEMM's dequantisation inside the silu_mul_quant kernel (no scale epilogue)
    static const bool fold = std::getenv("G2E_DQ_FOLD") != nullptr;
    const auto act = fold ? o.fp8_gemm_silu_mul_quant(nq.codes, nq.scale, gu_w, gu_s, P(j, "gu"))
                          : o.silu_mul_quant(o.fp8_gemm(nq.codes, nq.scale, gu_w, gu_s, P(j, "gu")), P(j, "act"));
    const auto down = o.fp8_gemm(act.first, act.second, dn_w, dn_s, P(j, "down"));
    return {nq.residual_out, down};
}

// Verification GDN core through the WY-form TPC kernel (fixed head geometry, T = 16); G2E_SPEC_CHUNK=1 keeps the
// chunked graph.
// Output gate product + out_proj input quantisation in one kernel (G2E_GATE_GRAPH=1: mul, abs, max, convert).
// rows per chunk of the chunked GDN forms (G2E_GDN_CHUNK: 64 or 128)
uint32_t Qwen35::gdn_chunk() const {
    static const uint32_t c = [] {
        const char* e = std::getenv("G2E_GDN_CHUNK");
        const uint32_t v = e ? uint32_t(std::atoi(e)) : 64u;
        return v == 128 ? 128u : v == 32 ? 32u : 64u;
    }();
    return c;
}

// GDN output gate (RMSNorm x silu(z) -> FP8) as two kernels instead of graph ops + silu_mul_quant: opt-in
// (G2E_GATED_KERNEL=1) - measured 0.3 ms slower per 8K verification (GDN blocks 12.33 vs 11.98 ms): the graph's
// small ops overlap their neighbours, the kernels serialise
bool Qwen35::gated_kernel() const {
    static const bool on = std::getenv("G2E_GATED_KERNEL") != nullptr;
    return on && fused_gate_quant();
}

bool Qwen35::fused_gate_quant() const {
    static const bool graph = std::getenv("G2E_GATE_GRAPH") != nullptr;
    return !graph && debug_layer_ < 0;
}

// Attention through Synapse's fused SDPA for context buckets of >= 2048 rows (G2E_SDPA_MIN overrides; scores in
// BF16, never materialised: 3x faster prefill at 32K). Shorter contexts keep the F32 graph (faster there, and
// exact F32 softmax).
bool Qwen35::attn_sdpa(uint32_t Sb) const {
    static const uint32_t min_rows = [] {
        const char* e = std::getenv("G2E_SDPA_MIN");
        return e ? uint32_t(std::atoi(e)) : 2048u;
    }();
    return Sb >= min_rows && debug_layer_ < 0;
}

// One softmax chain for all kv heads (G2E_ATTN_PERHEAD=1: one chain per head, as before).
bool Qwen35::attn_batched() const {
    static const bool per_head = std::getenv("G2E_ATTN_PERHEAD") != nullptr;
    return !per_head;
}

TensorId Qwen35::kv_to_f8(Ops& o, TensorId x, float mult, const std::string& name) {
    const auto codes = o.temp(name, DType::F8, o.shape(x));
    ns_ConvertToFp8::ParamsV2 cp{};
    cp.round_mode = CAST_ROUND_HALF_NE;
    o.graph().op("convert_to_fp8_bf16", {x, o.constant(mult, {1}, name + "_m")}, {codes}, &cp, sizeof(cp));
    return codes;
}

TensorId Qwen35::kv_in_bf16(Ops& o, const std::string& name, uint64_t rows) {
    const auto t = o.graph().input(name, kv_dt(), {rows, uint64_t(cfg_.head_dim)});
    if (!kv8_) return t;
    // the FP8 dequantisation op (cast_f8_to_bf16 is refused in some graphs: "required data type less precise")
    const auto out = o.temp(name + "_bf", DType::BF16, {rows, uint64_t(cfg_.head_dim)});
    o.graph().op("convert_from_fp8_bf16", {t, o.constant(1.f, {1}, name + "_one")}, {out});
    return out;
}

TensorId Qwen35::kv_scores(Ops& o, TensorId q, TensorId kc, const std::string& name) {
    if (!kv8_) return o.linear_bf16(q, kc, name, DType::BF16);
    const auto qs = o.shape(q), ks = o.shape(kc);
    const auto out = o.temp(name, DType::BF16, {qs[0], ks[0]});
    o.graph().gemm(kv_to_f8(o, q, 1.f, name + "_q8"), kc, out, false, true);  // FP8 x FP8, unit scales: no epilogue
    return out;
}

TensorId Qwen35::kv_pv(Ops& o, TensorId p, TensorId vc3, const std::string& name) {
    // FP8: the unnormalised probabilities (<= 1) x 128 in E4M3 (as the FP8 SDPA's q_scale_s); the caller divides by
    // the row sums x 128
    return o.batch_gemm(kv8_ ? kv_to_f8(o, p, 128.f, name + "_p8") : p, vc3, name, false, false, DType::F32);
}

bool Qwen35::wy_geometry() const {
    return cfg_.lin_k_heads == 16 && cfg_.lin_v_heads == 48 && cfg_.lin_k_dim == 128 && cfg_.lin_v_dim == 128;
}

bool Qwen35::spec_wy(uint32_t T) const {
    static const bool chunked = std::getenv("G2E_SPEC_CHUNK") != nullptr;
    return !chunked && T == 16 && wy_geometry();
}

Qwen35::Pair Qwen35::gdn_layer(Ops& o, uint32_t j, uint32_t T, Pair in, uint32_t B) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden, HK = cfg_.lin_k_heads, HV = cfg_.lin_v_heads, DK = cfg_.lin_k_dim,
                   DV = cfg_.lin_v_dim, C = cfg_.conv_channels(), KC = cfg_.conv_kernel, REP = HV / HK;
    const uint64_t RT = uint64_t(T) * B;  // rows (B slots of T rows each)
    const auto in_norm = g.input(P(j, "in_norm1p"), DType::BF16, {H});
    const auto qkvz_w = g.input(P(j, "qkvz_w"), DType::F8, {C + HV * DV, H});
    const auto qkvz_s = g.input(P(j, "qkvz_s"), DType::F32, {C + HV * DV});
    const auto ba_w = g.input(P(j, "ba_w"), DType::BF16, {2 * HV, H});
    const auto conv_w = g.input(P(j, "conv_t"), DType::BF16, {KC, C});
    const auto negexpA = g.input(P(j, "negexpA"), DType::F32, {HV});
    const auto dt_bias = g.input(P(j, "dt_bias"), DType::BF16, {HV});
    const auto norm_w = g.input(P(j, "gdn_norm"), DType::BF16, {DV});
    const auto out_w = g.input(P(j, "out_w"), DType::F8, {H, HV * DV});
    const auto out_s = g.input(P(j, "out_s"), DType::F32, {H});

    const auto nq = o.add_rmsnorm_quant(in.delta, in.res, in_norm, cfg_.eps, P(j, "in"));
    dbg(o, j, "x", nq.normalized);
    const auto qkvz = o.fp8_gemm(nq.codes, nq.scale, qkvz_w, qkvz_s, P(j, "qkvz"));
    dbg(o, j, "qkvz", qkvz);
    const auto pz = o.split(qkvz, 1, {C, HV * DV}, P(j, "qkvz_split"));
    const auto ba = o.linear_bf16(nq.normalized, ba_w, P(j, "ba"), DType::F32);
    dbg(o, j, "ba", ba);
    const auto bap = o.split(ba, 1, {HV, HV}, P(j, "ba_split"));  // b, a
    const auto sp = o.unary("softplus", o.add(bap[1], o.cast(dt_bias, DType::F32, P(j, "dtbf")), P(j, "adt")), P(j, "sp"));
    const auto gdecay = o.mul(sp, negexpA, P(j, "g"));  // [T, HV]
    const auto decay = o.unary("exp", gdecay, P(j, "decay"));
    const auto beta = o.unary("sigmoid", bap[0], P(j, "beta"));
    dbg(o, j, "g", gdecay);
    dbg(o, j, "beta", beta);

    TensorId core;  // [RT, HV, DV] bf16
    if (B > 1 && !building_spec_) {
        // batched prefill: every slot's conv over its own window; the delta rule of all slots as one chunked
        // recurrence over B x heads (slot b's value head h = head b*HV + h uses key head b*HK + h/REP)
        if (T < 64 || T % 64) throw std::logic_error("qwen35: batched prefill pieces are multiples of 64 rows");
        const auto pz0 = o.split(pz[0], 0, std::vector<uint64_t>(B, T), P(j, "pbz"));
        const auto gsp = o.split(gdecay, 0, std::vector<uint64_t>(B, T), P(j, "pbg"));
        const auto bsp = o.split(beta, 0, std::vector<uint64_t>(B, T), P(j, "pbb"));
        const auto taps = o.split(o.cast(conv_w, DType::F32, P(j, "pcwf")), 0, std::vector<uint64_t>(KC, 1), P(j, "ptap"));
        std::vector<TensorId> qs, ks, vs, Ss;
        for (uint32_t b = 0; b < B; ++b) {
            const auto x = std::to_string(b);
            const auto conv_in = g.input(P(j, "conv_in" + x), DType::BF16, {KC - 1, C});
            Ss.push_back(g.input(P(j, "ssm_in" + x), DType::F32, {HV, DV, DK}));
            const auto seq = o.concat({conv_in, pz0[b]}, 0, P(j, "pseq" + x));
            output(g, P(j, "conv_out" + x), o.split(seq, 0, {T, KC - 1}, P(j, "ptail" + x))[1]);
            const auto seqf = o.cast(seq, DType::F32, P(j, "pseqf" + x));
            TensorId accum = kNoTensor;
            for (uint64_t k = 0; k < KC; ++k) {
                std::vector<uint64_t> sizes;
                if (k) sizes.push_back(k);
                sizes.push_back(T);
                if (KC - 1 - k) sizes.push_back(KC - 1 - k);
                const auto win = o.split(seqf, 0, sizes, P(j, "pwin" + x + "_" + std::to_string(k)))[k ? 1 : 0];
                const auto term = o.mul(win, taps[k], P(j, "pterm" + x + "_" + std::to_string(k)));
                accum = accum == kNoTensor ? term : o.add(accum, term, P(j, "pacc" + x + "_" + std::to_string(k)));
            }
            const auto conv = o.silu(accum, P(j, "pconvf" + x));
            const auto qkv = o.split(conv, 1, {HK * DK, HK * DK, HV * DV}, P(j, "pcs" + x));
            auto l2 = [&](TensorId t, uint64_t heads, uint64_t d, const std::string& n) {
                const auto t3 = o.reshape(t, {T, heads, d}, n + "3");
                const auto ss = o.add(o.reduce("sum", o.mul(t3, t3, n + "sq"), 2, n + "ss"), o.constant(1e-6f, {1}, n + "eps"), n + "sse");
                return o.mul(t3, o.unary("rsqrt", ss, n + "rs"), n + "n");
            };
            qs.push_back(o.mul(l2(qkv[0], HK, DK, P(j, "pq" + x)), o.constant(float(1.0 / std::sqrt(double(DK))), {1}, P(j, "pqsc" + x)),
                               P(j, "pqs" + x)));
            ks.push_back(l2(qkv[1], HK, DK, P(j, "pk" + x)));
            vs.push_back(o.reshape(qkv[2], {T, HV, DV}, P(j, "pv" + x)));
        }
        auto& gr = o.graph();
        const uint32_t Cc = gdn_chunk(), nc = T / Cc;
        const std::string cp = "chunk" + std::to_string(Cc) + ".";
        auto cst = [&](const std::string& name, Shape shape) {
            const auto key = cp + name;
            if (auto it = build_inputs_.find(key); it != build_inputs_.end()) return it->second;
            return build_inputs_[key] = gr.input(key, DType::F32, std::move(shape));
        };
        const auto [core_f, s_out] = g2e::gdn_chunks(
            o, P(j, "pcks_"), Cc, nc, {B * HK, B * HV, cfg_.lin_k_dim, cfg_.lin_v_dim}, cst, o.concat(Ss, 0, P(j, "pssc")),
            o.concat(qs, 1, P(j, "pqc")), o.concat(ks, 1, P(j, "pkc")), o.concat(vs, 1, P(j, "pvc")),
            o.concat(std::vector<TensorId>(gsp.begin(), gsp.end()), 1, P(j, "pgc")),
            o.concat(std::vector<TensorId>(bsp.begin(), bsp.end()), 1, P(j, "pbc")));
        const auto sparts = o.split(s_out, 0, std::vector<uint64_t>(B, HV), P(j, "psso"));
        for (uint32_t b = 0; b < B; ++b) output(g, P(j, "ssm_out" + std::to_string(b)), sparts[b]);
        const auto cparts = o.split(core_f, 1, std::vector<uint64_t>(B, HV), P(j, "pcore"));  // [T, HV, DV] each
        core = o.cast(o.concat(std::vector<TensorId>(cparts.begin(), cparts.end()), 0, P(j, "pcorec")), DType::BF16, P(j, "core"));
        goto gated_output;
    }
    if (B > 1) {
        // batched verification (WY form, T = 16): every slot's rows with its own conv window and state; the WY
        // kernel takes the slots as its batch dimension, the Gram / state products run as one batched GEMM
        if (!(building_spec_ && (T == 8 || T == 16) && wy_geometry()))
            throw std::logic_error("qwen35: batched GDN only for the WY verification");
        const auto pz0 = o.split(pz[0], 0, std::vector<uint64_t>(B, T), P(j, "bz"));
        const auto gsp = o.split(gdecay, 0, std::vector<uint64_t>(B, T), P(j, "bg"));
        const auto bsp = o.split(beta, 0, std::vector<uint64_t>(B, T), P(j, "bb"));
        std::vector<TensorId> Xs, Vs, Ss;
        for (uint32_t b = 0; b < B; ++b) {
            const auto bs = std::to_string(b);
            const auto conv_in = g.input(P(j, "conv_in" + bs), DType::BF16, {KC - 1, C});
            const auto ssm_in = g.input(P(j, "ssm_in" + bs), DType::F32, {HV, DV, DK});
            const auto seq = o.concat({conv_in, pz0[b]}, 0, P(j, "seq" + bs));
            const auto X = o.temp(P(j, "bwx" + bs), DType::F32, {HK, 2 * T, DK});
            const auto Vv = o.temp(P(j, "bwv" + bs), DType::F32, {T, HV * DV});
            g.op("flashinfer_gaudi_gdn_spec_prep_bf16_gaudi2", {seq, conv_w}, {X, Vv});
            // stashed for the commit; every one is also consumed below (see the single-slot form)
            output(g, P(j, "st_x" + bs), X);
            output(g, P(j, "st_v" + bs), Vv);
            output(g, P(j, "st_g" + bs), gsp[b]);
            output(g, P(j, "st_b" + bs), bsp[b]);
            output(g, P(j, "st_seq" + bs), seq);
            Vs.push_back(o.reshape(Vv, {1, T, HV * DV}, P(j, "bv3" + bs)));
            // per slot, straight from its bound state (concatenating the states would copy them: they are
            // persistent inputs); the products' outputs are concatenated
            Xs.push_back(o.batch_gemm(X, X, P(j, "bwgram" + bs), false, true, DType::F32));  // [HK,2T,2T]
            Ss.push_back(o.batch_gemm(X, o.reshape(ssm_in, {HK, REP * DV, DK}, P(j, "bs3" + bs)), P(j, "bwr" + bs), false,
                                      true, DType::F32));                                  // [HK,2T,REP*DV]
        }
        const auto gram = o.concat(Xs, 0, P(j, "bgr"));  // [B*HK,2T,2T]
        const auto rmat = o.concat(Ss, 0, P(j, "brm"));  // [B*HK,2T,REP*DV]
        const auto gcat = o.concat(std::vector<TensorId>(gsp.begin(), gsp.end()), 0, P(j, "bgc"));
        const auto bcat = o.concat(std::vector<TensorId>(bsp.begin(), bsp.end()), 0, P(j, "bbc"));
        const auto core4 = o.temp(P(j, "wcore"), DType::F32, {B, T, HV, DV});
        g.op(T == 8 ? "flashinfer_gaudi_gdn_mtp_wyo8_f32_gaudi2" : "flashinfer_gaudi_gdn_mtp_wyo_f32_gaudi2",
             {o.reshape(gcat, {B, T, HV}, P(j, "wg3")), o.reshape(bcat, {B, T, HV}, P(j, "wb3")),
              o.reshape(rmat, {B, HK, 2 * T, REP * DV}, P(j, "wr4")), o.reshape(gram, {B, HK, 2 * T, 2 * T}, P(j, "wgr4")),
              o.reshape(o.concat(Vs, 0, P(j, "bvc")), {B, T, HV, DV}, P(j, "wv4"))},
             {core4});
        core = o.cast(o.reshape(core4, {RT, HV, DV}, P(j, "wc3")), DType::BF16, P(j, "core"));
        goto gated_output;
    }
    if (T == 1) {
        const auto conv_in = g.input(P(j, "conv_in"), DType::BF16, {1, KC - 1, C});
        const auto conv_out = g.input(P(j, "conv_out"), DType::BF16, {1, KC - 1, C});
        // one tensor per block: persistent tensors must not overlap, and every layer uses the same constants
        if (idx_t_ == kNoTensor) {
            idx_t_ = g.input("idx", DType::I32, {1});
            acc_t_ = g.input("acc", DType::I32, {1});
        }
        const auto idx = idx_t_, acc = acc_t_;
        const auto ssm = g.input(P(j, "ssm"), DType::F32, {1, HV, DV, DK});
        const auto y = o.temp(P(j, "conv"), DType::BF16, {1, C});
        g.op("flashinfer_gaudi_spec_conv_bf16_gaudi2", {pz[0], conv_in, conv_w, idx, acc}, {y, conv_out});
        dbg(o, j, "conv", y);
        core = o.temp(P(j, "core"), DType::BF16, {1, HV, DV});
        const auto beta_b = o.cast(beta, DType::BF16, P(j, "beta_b"));
        g.op("flashinfer_gaudi_gdn_decode_packed_bf16_f32state_gaudi2", {ssm, y, decay, beta_b, idx}, {core});
    } else {
        // reference-grade prefill path: causal conv over [state; tokens], unrolled delta rule (state [HV, DV, DK])
        const auto conv_in = g.input(P(j, "conv_in"), DType::BF16, {KC - 1, C});
        const auto ssm_in = g.input(P(j, "ssm_in"), DType::F32, {HV, DV, DK});
        const auto seq = o.concat({conv_in, pz[0]}, 0, P(j, "seq"));
        if (!building_spec_) {
            const auto tail = o.split(seq, 0, {T, KC - 1}, P(j, "tail"));
            output(g, P(j, "conv_out"), tail[1]);
        }
        if (building_spec_ && spec_wy(T)) {
            // verification, WY form. One kernel: conv + SiLU, laid out as X = [k; q] per key head [HK, 2T, DK]
            // (raw, un-normalised) and V [T, HV*DV]. The products of the initial state with every k_t / q_t
            // (per key head: its 3 value heads side by side) and the Gram matrix of X run on the MME in FP32; the
            // WY kernel does the rest (normalisation, decay, delta rule) per value head. Every stashed tensor is
            // also consumed here (a graph output nothing else reads made layer recipes nondeterministic).
            const auto X = o.temp(P(j, "wx"), DType::F32, {HK, 2 * T, DK});
            const auto Vv = o.temp(P(j, "wv"), DType::F32, {T, HV * DV});
            g.op("flashinfer_gaudi_gdn_spec_prep_bf16_gaudi2", {seq, conv_w}, {X, Vv});
            output(g, P(j, "st_x"), X);
            output(g, P(j, "st_v"), Vv);
            output(g, P(j, "st_g"), gdecay);
            output(g, P(j, "st_b"), beta);
            output(g, P(j, "st_seq"), seq);
            const auto gram = o.batch_gemm(X, X, P(j, "wgram"), false, true, DType::F32);                       // [HK,2T,2T]
            const auto rmat = o.batch_gemm(X, o.reshape(ssm_in, {HK, REP * DV, DK}, P(j, "ws3")), P(j, "wr"), false, true,
                                           DType::F32);                                                         // [HK,2T,REP*DV]
            const auto core4 = o.temp(P(j, "wcore"), DType::F32, {1, T, HV, DV});
            g.op("flashinfer_gaudi_gdn_mtp_wyo_f32_gaudi2",
                 {o.reshape(gdecay, {1, T, HV}, P(j, "wg3")), o.reshape(beta, {1, T, HV}, P(j, "wb3")),
                  o.reshape(rmat, {1, HK, 2 * T, REP * DV}, P(j, "wr4")), o.reshape(gram, {1, HK, 2 * T, 2 * T}, P(j, "wgr4")),
                  o.reshape(Vv, {1, T, HV, DV}, P(j, "wv4"))},
                 {core4});
            core = o.cast(o.reshape(core4, {T, HV, DV}, P(j, "wc3")), DType::BF16, P(j, "core"));
            goto gated_output;
        }
        const auto seqf = o.cast(seq, DType::F32, P(j, "seqf"));
        const auto taps = o.split(o.cast(conv_w, DType::F32, P(j, "cwf")), 0, std::vector<uint64_t>(KC, 1), P(j, "tap"));
        TensorId accum = kNoTensor;
        for (uint64_t k = 0; k < KC; ++k) {
            std::vector<uint64_t> sizes;
            if (k) sizes.push_back(k);
            sizes.push_back(T);
            if (KC - 1 - k) sizes.push_back(KC - 1 - k);
            const auto win = o.split(seqf, 0, sizes, P(j, "win" + std::to_string(k)))[k ? 1 : 0];
            const auto term = o.mul(win, taps[k], P(j, "term" + std::to_string(k)));
            accum = accum == kNoTensor ? term : o.add(accum, term, P(j, "acc" + std::to_string(k)));
        }
        const auto conv = o.silu(accum, P(j, "convf"));
        dbg(o, j, "conv", conv);
        const auto qkv = o.split(conv, 1, {HK * DK, HK * DK, HV * DV}, P(j, "conv_split"));
        auto l2 = [&](TensorId t, uint64_t heads, uint64_t d, const std::string& n) {
            const auto t3 = o.reshape(t, {T, heads, d}, n + "3");
            const auto ss = o.add(o.reduce("sum", o.mul(t3, t3, n + "sq"), 2, n + "ss"), o.constant(1e-6f, {1}, n + "eps"), n + "sse");
            return o.mul(t3, o.unary("rsqrt", ss, n + "rs"), n + "n");
        };
        const auto q = o.mul(l2(qkv[0], HK, DK, P(j, "q")), o.constant(float(1.0 / std::sqrt(double(DK))), {1}, P(j, "qscale")), P(j, "qs"));
        const auto k = l2(qkv[1], HK, DK, P(j, "k"));
        const auto v = o.reshape(qkv[2], {T, HV, DV}, P(j, "v"));
        if (building_spec_) {
            // verification (chunked graph): stash what the commit needs, compute the outputs, leave the states
            output(g, P(j, "st_q"), q);
            output(g, P(j, "st_k"), k);
            output(g, P(j, "st_v"), v);
            output(g, P(j, "st_g"), gdecay);
            output(g, P(j, "st_b"), beta);
            output(g, P(j, "st_seq"), seq);
            const auto [core_f, s_unused] = gdn_chunk(o, P(j, "ckv_"), T, HK, HV, ssm_in, q, k, v, gdecay, beta);
            (void)s_unused;
            core = o.cast(core_f, DType::BF16, P(j, "core"));
            goto gated_output;
        }
        if (T >= 8 && (T & (T - 1)) == 0) {
            const uint32_t C = std::min<uint32_t>(T, gdn_chunk()), nc = T / C;
            auto parts = [&](TensorId t, const std::string& nm) {
                return nc == 1 ? std::vector<TensorId>{t} : o.split(t, 0, std::vector<uint64_t>(nc, C), P(j, nm));
            };
            const auto qs = parts(q, "cq"), ks = parts(k, "ck"), vs = parts(v, "cv"), gs = parts(gdecay, "cg"), bs = parts(beta, "cb");
            static const bool serial = std::getenv("G2E_GDN_SERIAL_CHUNKS") != nullptr;
            if (nc > 1 && !serial) {  // the state-independent work batched over the chunks (g2e::gdn_chunks)
                auto& gr = o.graph();
                const std::string cp = "chunk" + std::to_string(C) + ".";
                auto cst = [&](const std::string& name, Shape shape) {
                    const auto key = cp + name;
                    if (auto it = build_inputs_.find(key); it != build_inputs_.end()) return it->second;
                    return build_inputs_[key] = gr.input(key, DType::F32, std::move(shape));
                };
                const auto [core_f, s_out] = g2e::gdn_chunks(o, P(j, "cks_"), C, nc, {HK, HV, cfg_.lin_k_dim, cfg_.lin_v_dim},
                                                             cst, ssm_in, q, k, v, gdecay, beta);
                output(g, P(j, "ssm_out"), s_out);
                core = o.cast(core_f, DType::BF16, P(j, "core"));
                goto gated_output;
            }
            TensorId S = ssm_in;
            std::vector<TensorId> cores;
            for (uint32_t ci = 0; ci < nc; ++ci) {
                const auto [core_f, s_out] = gdn_chunk(o, P(j, "ck" + std::to_string(ci) + "_"), C, HK, HV, S, qs[ci], ks[ci], vs[ci],
                                                       gs[ci], bs[ci]);
                cores.push_back(core_f);
                S = s_out;
            }
            output(g, P(j, "ssm_out"), S);
            core = o.cast(nc == 1 ? cores[0] : o.concat(cores, 0, P(j, "ccat")), DType::BF16, P(j, "core"));
            goto gated_output;
        }
        {
        auto rows = [&](TensorId t, const std::string& n) { return o.split(t, 0, std::vector<uint64_t>(T, 1), n); };
        const auto qs = rows(q, P(j, "qt")), ks = rows(k, P(j, "kt")), vs = rows(v, P(j, "vt")), ds = rows(decay, P(j, "dt")),
                   bs = rows(beta, P(j, "bt"));
        const auto ones = o.constant(1.0f, {1, REP, 1}, P(j, "ones"));
        auto expand = [&](TensorId t, const std::string& n) {
            return o.reshape(o.mul(o.reshape(t, {HK, 1, DK}, n + "r"), ones, n + "e"), {HV, 1, DK}, n + "x");
        };
        TensorId S = ssm_in;
        std::vector<TensorId> outs;
        for (uint32_t t = 0; t < T; ++t) {
            const auto ts = P(j, std::to_string(t));
            const auto kt = expand(ks[t], ts + "k"), qt = expand(qs[t], ts + "q");
            S = o.mul(S, o.reshape(ds[t], {HV, 1, 1}, ts + "d"), ts + "Sd");
            const auto kv = o.batch_gemm(kt, S, ts + "kv", false, true, DType::F32);  // [HV,1,DV]
            const auto u = o.mul(o.sub(o.reshape(vs[t], {HV, 1, DV}, ts + "v"), kv, ts + "vm"),
                                 o.reshape(bs[t], {HV, 1, 1}, ts + "b"), ts + "u");
            S = o.add(S, o.batch_gemm(u, kt, ts + "uk", true, false, DType::F32), ts + "S");  // [HV,DV,DK]
            outs.push_back(o.reshape(o.batch_gemm(qt, S, ts + "o", false, true, DType::F32), {1, HV, DV}, ts + "or"));
        }
        output(g, P(j, "ssm_out"), S);
        core = o.cast(o.concat(outs, 0, P(j, "corecat")), DType::BF16, P(j, "core"));
        }
    }
gated_output:
    dbg(o, j, "core", core);
    if (gated_kernel() && DV == 128 && HV % 8 == 0 && building_debug_j_ != int(j)) {  // norm x silu(z) -> FP8 in one kernel
        const auto z3 = o.reshape(pz[1], {RT, HV, DV}, P(j, "z"));
        // stage 1 over (8-head groups x rows): gated BF16 + group maxima; stage 2 per row: FP8 codes + scale
        const auto gated = o.temp(P(j, "gq_gated"), DType::BF16, {RT, HV * DV});
        const auto pmax = o.temp(P(j, "gq_pmax"), DType::F32, {RT, HV / 8, 64});
        const auto codes = o.temp(P(j, "gq_codes"), DType::F8, {RT, HV * DV});
        const auto scale = o.temp(P(j, "gq_scale"), DType::F32, {RT, 1});
        const float eps = cfg_.eps;
        g.op("flashinfer_gaudi_gated_norm_quant_bf16_gaudi2", {o.reshape(core, {RT, HV, DV}, P(j, "core3")), z3, norm_w},
             {gated, pmax}, &eps, sizeof(eps));
        g.op("flashinfer_gaudi_row_quant_bf16_gaudi2", {gated, pmax}, {codes, scale});
        const auto y = o.fp8_gemm(codes, scale, out_w, out_s, P(j, "outp"));
        return mlp(o, j, y, nq.residual_out);
    }
    const auto normed = o.rms_norm(core, norm_w, cfg_.eps, false, P(j, "gnorm"));
    const auto z = o.reshape(pz[1], {RT, HV, DV}, P(j, "z"));
    const auto zs = o.temp(P(j, "zsilu"), DType::BF16, {RT, HV, DV});
    g.op("silu_fwd_bf16", {z}, {zs});
    TensorId gated = kNoTensor;
    std::pair<TensorId, TensorId> q8;
    if (fused_gate_quant()) {
        q8 = o.mul_quant(o.reshape(zs, {RT, HV * DV}, P(j, "zs2")), o.reshape(normed, {RT, HV * DV}, P(j, "nm2")), P(j, "gq"));
    } else {
        gated = o.reshape(o.mul(normed, zs, P(j, "gz")), {RT, HV * DV}, P(j, "g2"));
        dbg(o, j, "gated", gated);
        q8 = o.quantize(gated, P(j, "gq"));
    }
    const auto y = o.fp8_gemm(q8.first, q8.second, out_w, out_s, P(j, "outp"));
    dbg(o, j, "y", y);
    return mlp(o, j, y, nq.residual_out);
}

std::pair<TensorId, TensorId> Qwen35::gdn_chunk(Ops& o, const std::string& n, uint32_t C, uint64_t HK, uint64_t HV,
                                                TensorId S, TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta) {
    auto& gr = o.graph();
    const std::string cp = "chunk" + std::to_string(C) + ".";
    auto cst = [&](const std::string& name, Shape shape) {
        const auto key = cp + name;  // shared by every layer of the recipe (persistent tensors must not overlap)
        if (auto it = build_inputs_.find(key); it != build_inputs_.end()) return it->second;
        return build_inputs_[key] = gr.input(key, DType::F32, std::move(shape));
    };
    return g2e::gdn_chunk(o, n, C, {HK, HV, cfg_.lin_k_dim, cfg_.lin_v_dim}, cst, S, q, k, v, g, beta);
}

Qwen35::Pair Qwen35::attention_layer(Ops& o, uint32_t j, uint32_t T, uint32_t Sb, Pair in) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden, NH = cfg_.heads, KV = cfg_.kv_heads, D = cfg_.head_dim, R2 = cfg_.rotary_dim / 2;
    const uint64_t G = NH / KV;
    const auto in_norm = g.input(P(j, "in_norm1p"), DType::BF16, {H});
    const auto qkv_w = g.input(P(j, "qkv_w"), DType::F8, {NH * D * 2 + 2 * KV * D, H});
    const auto qkv_s = g.input(P(j, "qkv_s"), DType::F32, {NH * D * 2 + 2 * KV * D});
    const auto q_norm = g.input(P(j, "q_norm"), DType::BF16, {D});
    const auto k_norm = g.input(P(j, "k_norm"), DType::BF16, {D});
    const auto o_w = g.input(P(j, "o_w"), DType::F8, {H, NH * D});
    const auto o_s = g.input(P(j, "o_s"), DType::F32, {H});
    const auto cosv = g.input("cos", DType::F32, {T, 1, R2});   // shared by every attention layer of the block
    const auto sinv = g.input("sin", DType::F32, {T, 1, R2});
    const auto mask = attn_mask(o, T, Sb);

    const auto nq = o.add_rmsnorm_quant(in.delta, in.res, in_norm, cfg_.eps, P(j, "in"));
    dbg(o, j, "x", nq.normalized);
    const auto qkv = o.fp8_gemm(nq.codes, nq.scale, qkv_w, qkv_s, P(j, "qkv"));
    dbg(o, j, "qkv", qkv);
    const auto p = o.split(qkv, 1, {NH * D * 2, KV * D, KV * D}, P(j, "qkv_split"));
    const auto qg = o.split(o.reshape(p[0], {T, NH, 2 * D}, P(j, "qg3")), 2, {D, D}, P(j, "qg_split"));
    const auto gate = o.reshape(qg[1], {T, NH * D}, P(j, "gate"));
    const auto q = o.rms_norm(qg[0], q_norm, cfg_.eps, true, P(j, "qn"));
    const auto k = o.rms_norm(o.reshape(p[1], {T, KV, D}, P(j, "k3")), k_norm, cfg_.eps, true, P(j, "kn"));
    const auto v = o.reshape(p[2], {T, KV, D}, P(j, "v3"));
    auto rope = [&](TensorId t, const std::string& n) {
        const auto f = o.cast(t, DType::F32, n + "_f");
        const auto s = o.split(f, 2, {R2, R2, D - 2 * R2}, n + "_rs");
        const auto o1 = o.sub(o.mul(s[0], cosv, n + "_1c"), o.mul(s[1], sinv, n + "_2s"), n + "_o1");
        const auto o2 = o.add(o.mul(s[1], cosv, n + "_2c"), o.mul(s[0], sinv, n + "_1s"), n + "_o2");
        return o.cast(o.concat({o1, o2, s[2]}, 2, n + "_cat"), DType::BF16, n + "_b");
    };
    const auto qr = rope(q, P(j, "qr"));
    const auto kr = rope(k, P(j, "kr"));
    dbg(o, j, "qr", qr);
    dbg(o, j, "kr", kr);
    // per kv head: rows [rows, D] (heads h*G .. h*G+G-1 contiguous for queries)
    auto per_head = [&](TensorId t3, uint64_t rows, const std::string& n) {
        const auto th = o.transpose(t3, {1, 0, 2}, n + "_t");  // [heads, T, D]
        const auto r = o.reshape(th, {KV, rows, D}, n + "_r");
        auto parts = o.split(r, 0, std::vector<uint64_t>(KV, 1), n + "_h");
        for (uint64_t h = 0; h < KV; ++h) parts[h] = o.reshape(parts[h], {rows, D}, n + "_h2" + std::to_string(h));
        return parts;
    };
    const auto kn = per_head(kr, T, P(j, "knew"));
    const auto vn = per_head(v, T, P(j, "vnew"));
    const auto qh = per_head(qr, G * T, P(j, "qh"));
    const auto scale = o.constant(float(1.0 / std::sqrt(double(D))), {1}, P(j, "scale"));
    std::vector<TensorId> heads_out;
    {   // new rows of this step -> scratch [2, KV, T, D]; the commit recipe copies them into the caches
        const auto kt = o.reshape(o.transpose(kr, {1, 0, 2}, P(j, "kn_t")), {1, KV, T, D}, P(j, "kn4"));
        const auto vt = o.reshape(o.transpose(v, {1, 0, 2}, P(j, "vn_t")), {1, KV, T, D}, P(j, "vn4"));
        output(g, P(j, "kv_new"), o.concat({kt, vt}, 0, P(j, "kvn")));
    }
    if (attn_sdpa(Sb)) {
        // Synapse's fused attention per kv head: its G query heads against [cache rows; new rows] (broadcast over
        // the query heads), the additive mask as BF16. Scores never materialise in HBM (long contexts).
        const auto mask4 = o.reshape(o.cast(mask, DType::BF16, P(j, "maskb")), {1, 1, T, Sb + T}, P(j, "mask4"));
        ns_Sdpa::ParamsV7 sp{};
        sp.scale = float(1.0 / std::sqrt(double(D)));
        sp.is_causal = false;
        sp.is_inference = true;
        sp.softmax_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
        sp.wl = sp.wr = -1;
        std::vector<TensorId> outs;
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto kc = kv_in_bf16(o, P(j, "k_cache" + hs), Sb);
            const auto vc = kv_in_bf16(o, P(j, "v_cache" + hs), Sb);
            const auto kf = o.reshape(o.concat({kc, kn[h]}, 0, P(j, "kf" + hs)), {1, 1, Sb + T, D}, P(j, "kf4" + hs));
            const auto vf = o.reshape(o.concat({vc, vn[h]}, 0, P(j, "vf" + hs)), {1, 1, Sb + T, D}, P(j, "vf4" + hs));
            const auto q4 = o.reshape(qh[h], {1, G, T, D}, P(j, "q4" + hs));
            const auto oh = o.temp(P(j, "sd" + hs), DType::BF16, {1, G, T, D});
            std::vector<TensorId> ins = {q4, kf, vf, mask4, kNoTensor};
            for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
            g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(&oh, 1), &sp, sizeof(sp));
            outs.push_back(o.reshape(oh, {1, G * T, D}, P(j, "sd3" + hs)));
        }
        heads_out.push_back(o.concat(outs, 0, P(j, "sdall")));  // [KV, G*T, D] BF16
    } else if (attn_batched()) {
        // the caches are bound per head (a context bucket is not contiguous across heads), so the GEMMs stay per
        // head, but one softmax / cast / add serves all heads: per-head TPC chains ran one after another
        std::vector<TensorId> scores, oc_all, on_all;
        std::vector<TensorId> vcs;
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto kc = kv_in_bf16(o, P(j, "k_cache" + hs), Sb);
            vcs.push_back(kv_in_bf16(o, P(j, "v_cache" + hs), Sb));
            const auto sc = o.linear_bf16(qh[h], kc, P(j, "sc" + hs), DType::F32);     // [GT, Sb]
            const auto sn = o.linear_bf16(qh[h], kn[h], P(j, "sn" + hs), DType::F32);  // [GT, T]
            scores.push_back(o.reshape(o.concat({sc, sn}, 1, P(j, "s" + hs)), {1, G * T, Sb + T}, P(j, "s1" + hs)));
        }
        auto s = o.mul(o.concat(scores, 0, P(j, "sall")), scale, P(j, "ssall"));                  // [KV, GT, Sb+T]
        s = o.add(o.reshape(s, {KV, G, T, Sb + T}, P(j, "s4")), mask, P(j, "small"));
        const auto pr = o.cast(o.reshape(o.softmax(s, P(j, "sxall")), {KV, G * T, Sb + T}, P(j, "p3")), DType::BF16, P(j, "pball"));
        const auto prh = o.split(pr, 0, std::vector<uint64_t>(KV, 1), P(j, "prh"));
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto pp = o.split(o.reshape(prh[h], {G * T, Sb + T}, P(j, "p2" + hs)), 1, {Sb, T}, P(j, "ps" + hs));
            const auto oc = o.temp(P(j, "oc" + hs), DType::F32, {G * T, D});
            g.gemm(pp[0], vcs[h], oc);
            const auto onew = o.temp(P(j, "on" + hs), DType::F32, {G * T, D});
            g.gemm(pp[1], vn[h], onew);
            oc_all.push_back(o.reshape(oc, {1, G * T, D}, P(j, "oc3" + hs)));
            on_all.push_back(o.reshape(onew, {1, G * T, D}, P(j, "on3" + hs)));
        }
        heads_out.push_back(o.add(o.concat(oc_all, 0, P(j, "ocall")), o.concat(on_all, 0, P(j, "onall")), P(j, "oall")));
    }
    for (uint64_t h = 0; h < (attn_batched() || attn_sdpa(Sb) ? 0 : KV); ++h) {
        const auto hs = std::to_string(h);
        const auto kc = kv_in_bf16(o, P(j, "k_cache" + hs), Sb);
        const auto vc = kv_in_bf16(o, P(j, "v_cache" + hs), Sb);
        const auto sc = o.linear_bf16(qh[h], kc, P(j, "sc" + hs), DType::F32);     // [GT, Sb]
        const auto sn = o.linear_bf16(qh[h], kn[h], P(j, "sn" + hs), DType::F32);  // [GT, T]
        auto s = o.mul(o.concat({sc, sn}, 1, P(j, "s" + hs)), scale, P(j, "ss" + hs));
        s = o.add(o.reshape(s, {G, T, Sb + T}, P(j, "s3" + hs)), mask, P(j, "sm" + hs));
        const auto pr = o.cast(o.reshape(o.softmax(s, P(j, "sx" + hs)), {G * T, Sb + T}, P(j, "p2" + hs)), DType::BF16,
                               P(j, "pb" + hs));
        const auto pp = o.split(pr, 1, {Sb, T}, P(j, "ps" + hs));
        const auto oc = o.temp(P(j, "oc" + hs), DType::F32, {G * T, D});
        g.gemm(pp[0], vc, oc);
        const auto onew = o.temp(P(j, "on" + hs), DType::F32, {G * T, D});
        g.gemm(pp[1], vn[h], onew);
        heads_out.push_back(o.reshape(o.add(oc, onew, P(j, "o" + hs)), {1, G * T, D}, P(j, "o3" + hs)));
    }
    const auto att = o.cast(heads_out.size() == 1 ? heads_out[0] : o.concat(heads_out, 0, P(j, "att")), DType::BF16,
                            P(j, "attb"));  // [KV, G*T, D]
    const auto att2 = o.reshape(o.transpose(o.reshape(att, {NH, T, D}, P(j, "a3")), {1, 0, 2}, P(j, "at")), {T, NH * D},
                                P(j, "a2"));
    std::pair<TensorId, TensorId> q8;
    if (fused_gate_quant()) {
        q8 = o.mul_quant(o.unary("sigmoid", gate, P(j, "gsig")), att2, P(j, "gq"));
    } else {
        const auto gated = o.mul(att2, o.unary("sigmoid", gate, P(j, "gsig")), P(j, "gated"));
        dbg(o, j, "gated", gated);
        q8 = o.quantize(gated, P(j, "gq"));
    }
    const auto y = o.fp8_gemm(q8.first, q8.second, o_w, o_s, P(j, "oproj"));
    dbg(o, j, "y", y);
    return mlp(o, j, y, nq.residual_out);
}

uint32_t Qwen35::group_at(uint32_t first, uint32_t nlayers) const {
    if (block_ != 1) return block_;
    static const uint32_t gmax = [] {
        const char* e = std::getenv("G2E_GDN_GROUP");
        const int v = e ? std::atoi(e) : 1;  // 3 measured slower (verify 17.7 vs 17.2 ms): default off
        return uint32_t(v < 1 ? 1 : v > 3 ? 3 : v);
    }();
    if (cfg_.attention[first] || debug_layer_ >= 0) return 1;
    uint32_t n = 1;
    while (n < gmax && first + n < nlayers && !cfg_.attention[first + n]) ++n;
    return n;
}

Recipe& Qwen35::block_recipe(uint32_t first, uint32_t T, uint32_t Sb) {
    const uint32_t nb = launch_n_ ? launch_n_ : block_;
    std::string pattern;
    bool any_attention = false;
    for (uint32_t j = 0; j < nb; ++j) {
        pattern += cfg_.attention[first + j] ? 'A' : 'g';
        any_attention |= cfg_.attention[first + j];
    }
    building_debug_j_ = -1;
    if (debug_layer_ >= int(first) && debug_layer_ < int(first + nb)) building_debug_j_ = debug_layer_ - int(first);
    std::string key = "qwen35b_" + pattern + "_T" + std::to_string(T) + (building_spec_ ? (spec_wy(T) ? "_specwy" : "_spec") : "");
    if (!building_spec_ && T > 64 && pattern.find('g') != std::string::npos && !std::getenv("G2E_GDN_SERIAL_CHUNKS"))
        key += "_bc" + (gdn_chunk() != 64 ? std::to_string(gdn_chunk()) : std::string());  // chunks batched (gdn_chunks)
    if (any_attention) key += "_S" + std::to_string(Sb) + (attn_sdpa(Sb) ? "f" : attn_batched() ? "b" : "") + "m";  // m: in-graph mask
    if (fused_gate_quant()) key += "_mq";
    if (gated_kernel() && pattern.find('g') != std::string::npos) key += "_gk2";
    for (size_t k = 0; k < capture_.size(); ++k)
        if (nb > 1 && capture_[k] >= first && capture_[k] < first + nb) key += "_cap" + std::to_string(capture_[k] - first);
    if (building_debug_j_ >= 0) {
        key += "_dbg" + std::to_string(building_debug_j_);
        for (const auto& p : debug_points_) key += "_" + p;
    }
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden;
    GraphBuilder g(dev_);
    Ops o(g);
    idx_t_ = acc_t_ = mask_t_ = kNoTensor;
    build_inputs_.clear();
    Pair cur{g.input("res_in", DType::BF16, {T, H}), g.input("delta_in", DType::BF16, {T, H})};
    for (uint32_t j = 0; j < nb; ++j) {
        cur = cfg_.attention[first + j] ? attention_layer(o, j, T, Sb, cur) : gdn_layer(o, j, T, cur);
        for (size_t k = 0; k < capture_.size(); ++k)
            if (nb > 1 && capture_[k] == first + j) output(g, "feat_j" + std::to_string(j), o.add(cur.res, cur.delta, P(j, "feat")));
    }
    output(g, "res_out", cur.res);
    output(g, "delta_out", cur.delta);
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

// Additive attention mask [T, Sb + T] built on the device: context column c is visible iff c < pos, i.e.
// min(pos - 1 - c, 0) * 1e30 is 0 there and <= -1e30 elsewhere; the new rows see each other causally.
TensorId Qwen35::attn_mask(Ops& o, uint32_t T, uint32_t Sb) {
    if (mask_t_ != kNoTensor) return mask_t_;
    auto& g = o.graph();
    const auto posm1 = g.input("posm1", DType::F32, {1, 1});
    const auto iota = g.input("iota", DType::F32, {1, Sb});
    const auto causal = g.input("causal", DType::F32, {T, T});
    const auto ctx = o.mul(o.binary("min", o.sub(posm1, iota, "mk_d"), o.constant(0.f, {1, 1}, "mk_z"), "mk_m"),
                           o.constant(1e30f, {1, 1}, "mk_big"), "mk_c");                                   // [1, Sb]
    const auto ctxT = o.add(ctx, o.constant(0.f, {T, 1}, "mk_zt"), "mk_ct");                               // [T, Sb]
    return mask_t_ = o.concat({ctxT, causal}, 1, "mask");
}

bool Qwen35::attn_split(uint32_t T) const {
    return block_ == 1 && !k_cache_.empty() && attn_sdpa(bucket_for(std::min<uint32_t>(pos_ + T, cap_)));
}

Recipe& Qwen35::attn_pre_recipe(uint32_t T) {
    const std::string key = "qwen35_apre_T" + std::to_string(T);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, NH = cfg_.heads, KV = cfg_.kv_heads, D = cfg_.head_dim, R2 = cfg_.rotary_dim / 2;
    const uint32_t j = 0;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto res = g.input("res_in", DType::BF16, {T, H}), delta = g.input("delta_in", DType::BF16, {T, H});
    const auto in_norm = g.input(P(j, "in_norm1p"), DType::BF16, {H});
    const auto qkv_w = g.input(P(j, "qkv_w"), DType::F8, {NH * D * 2 + 2 * KV * D, H});
    const auto qkv_s = g.input(P(j, "qkv_s"), DType::F32, {NH * D * 2 + 2 * KV * D});
    const auto q_norm = g.input(P(j, "q_norm"), DType::BF16, {D});
    const auto k_norm = g.input(P(j, "k_norm"), DType::BF16, {D});
    const auto cosv = g.input("cos", DType::F32, {T, 1, R2});
    const auto sinv = g.input("sin", DType::F32, {T, 1, R2});
    // as attention_layer
    const auto nq = o.add_rmsnorm_quant(delta, res, in_norm, cfg_.eps, P(j, "in"));
    const auto qkv = o.fp8_gemm(nq.codes, nq.scale, qkv_w, qkv_s, P(j, "qkv"));
    const auto p = o.split(qkv, 1, {NH * D * 2, KV * D, KV * D}, P(j, "qkv_split"));
    const auto qg = o.split(o.reshape(p[0], {T, NH, 2 * D}, P(j, "qg3")), 2, {D, D}, P(j, "qg_split"));
    const auto q = o.rms_norm(qg[0], q_norm, cfg_.eps, true, P(j, "qn"));
    const auto k = o.rms_norm(o.reshape(p[1], {T, KV, D}, P(j, "k3")), k_norm, cfg_.eps, true, P(j, "kn"));
    const auto v = o.reshape(p[2], {T, KV, D}, P(j, "v3"));
    auto rope = [&](TensorId t, const std::string& n) {
        const auto f = o.cast(t, DType::F32, n + "_f");
        const auto s = o.split(f, 2, {R2, R2, D - 2 * R2}, n + "_rs");
        const auto o1 = o.sub(o.mul(s[0], cosv, n + "_1c"), o.mul(s[1], sinv, n + "_2s"), n + "_o1");
        const auto o2 = o.add(o.mul(s[1], cosv, n + "_2c"), o.mul(s[0], sinv, n + "_1s"), n + "_o2");
        return o.cast(o.concat({o1, o2, s[2]}, 2, n + "_cat"), DType::BF16, n + "_b");
    };
    output(g, "q_out", rope(q, P(j, "qr")));
    output(g, "gate_out", o.reshape(qg[1], {T, NH * D}, P(j, "gate")));
    output(g, "resid_out", nq.residual_out);
    const auto kh = o.split(o.transpose(rope(k, P(j, "kr")), {1, 0, 2}, "kt"), 0, std::vector<uint64_t>(KV, 1), "kh");
    const auto vh = o.split(o.transpose(v, {1, 0, 2}, "vt"), 0, std::vector<uint64_t>(KV, 1), "vh");
    for (uint64_t h = 0; h < KV; ++h) {  // rows pos .. pos+T-1 of each head's cache
        const auto k2 = o.reshape(kh[h], {T, D}, "k2_" + std::to_string(h));
        const auto v2 = o.reshape(vh[h], {T, D}, "v2_" + std::to_string(h));
        output(g, "k_out" + std::to_string(h), kv8_ ? kv_to_f8(o, k2, 1.f, "k8_" + std::to_string(h)) : k2);
        output(g, "v_out" + std::to_string(h), kv8_ ? kv_to_f8(o, v2, 1.f, "v8_" + std::to_string(h)) : v2);
    }
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Recipe& Qwen35::attn_post_recipe(uint32_t T, uint32_t Sb) {
    // up to kSpecMaxT query rows: scores -> mask -> softmax -> PV as graph ops with BF16 scores (1.8x faster than
    // SDPA at 131K: few query rows leave SDPA's softmax badly utilised); prefill pieces keep SDPA
    static const bool force_sdpa = std::getenv("G2E_VERIFY_SDPA") != nullptr;  // comparison switch
    const bool graph_attn = T <= kSpecMaxT && !force_sdpa;
    // the masked-softmax kernel + a chunked PV (default) or the graph softmax (G2E_ATTN_GRAPH_SOFTMAX=1)
    static const bool graph_softmax = std::getenv("G2E_ATTN_GRAPH_SOFTMAX") != nullptr;
    // full prefill pieces too (G2E_PREFILL_SDPA=1: SDPA): ~23% faster than SDPA at T = 512
    static const bool prefill_sdpa = std::getenv("G2E_PREFILL_SDPA") != nullptr;
    // full prefill pieces past 8K rows: FP8 SDPA (vLLM's FP8 fused attention): keys [0, Sb - 4096) are visible to
    // every query row -> no mask (the fast path, ~27% faster than the FP16 kernel at 32K); the last 4096 keys with a
    // small mask; the partial outputs merged by their row maxima / normalisers. Unit FP8 scales for q, k, v (as
    // vLLM's uncalibrated FP8 KV cache). G2E_PREFILL_NO_FP8=1: the FP16 kernel
    static const bool no_fp8 = std::getenv("G2E_PREFILL_NO_FP8") != nullptr;
    const bool fp8_attn = fp8_prefill(T, Sb) && !force_sdpa && !prefill_sdpa && !no_fp8;
    const bool kernel_attn = (graph_attn || (T > kSpecMaxT && !force_sdpa && !prefill_sdpa)) && !graph_softmax &&
                             Sb % 1024 == 0 && !fp8_attn;
    const bool f16_scores = kernel_attn && T > kSpecMaxT && !std::getenv("G2E_ATTN_BF16_SCORES") && !kv8_;
    static const bool one_softmax = std::getenv("G2E_ATTN_ONE_SOFTMAX") != nullptr;  // comparison switch
    const std::string key = "qwen35_apost_T" + std::to_string(T) + "_S" + std::to_string(Sb) + (fused_gate_quant() ? "_mq" : "") +
                            (fp8_attn ? "_f8" : kernel_attn ? (std::getenv("G2E_ATTN_HEAD_CHAINS") ? "_gk" : f16_scores ? "_gkf" : "_gks") : graph_attn ? (one_softmax ? "_gb1" : "_gb2") : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, NH = cfg_.heads, KV = cfg_.kv_heads, D = cfg_.head_dim, G = NH / KV;
    const uint32_t j = 0;
    GraphBuilder g(dev_);
    Ops o(g);
    build_inputs_.clear();
    const auto q = g.input("q_in", DType::BF16, {T, NH, D});
    const auto gate = g.input("gate_in", DType::BF16, {T, NH * D});
    const auto resid = g.input("resid_in", DType::BF16, {T, H});
    const auto o_w = g.input(P(j, "o_w"), DType::F8, {H, NH * D});
    const auto o_s = g.input(P(j, "o_s"), DType::F32, {H});
    // mask [T, Sb]: row t (position pos + t) sees cache rows c <= pos + t
    TensorId maskb = kNoTensor, mask4 = kNoTensor;
    if (!kernel_attn && !fp8_attn) {  // the kernel masks by the last visible row of each query; the others add a mask
        const auto posm1 = g.input("posm1", DType::F32, {1, 1});
        const auto iota_c = g.input("iota", DType::F32, {1, Sb});
        const auto iota_t = g.input("iota_t", DType::F32, {T, 1});
        const auto lim = o.add(o.add(posm1, o.constant(1.f, {1, 1}, "mk_one"), "mk_p"), iota_t, "mk_lim");   // [T,1]
        const auto vis = o.binary("min", o.sub(lim, iota_c, "mk_d"), o.constant(0.f, {1, 1}, "mk_z"), "mk_m"); // [T,Sb]
        maskb = o.cast(o.mul(vis, o.constant(1e30f, {1, 1}, "mk_big"), "mk_c"), DType::BF16, "mk_b");          // [T,Sb]
        mask4 = o.reshape(maskb, {1, 1, T, Sb}, "mask4");
    }
    ns_Sdpa::ParamsV7 sp{};
    sp.scale = float(1.0 / std::sqrt(double(D)));
    sp.is_causal = false;
    sp.is_inference = true;
    sp.softmax_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
    sp.wl = sp.wr = -1;
    const auto qh = o.split(o.reshape(o.transpose(q, {1, 0, 2}, "qt"), {KV, G * T, D}, "q3"), 0, std::vector<uint64_t>(KV, 1), "qh");
    std::vector<TensorId> outs;
    if (fp8_attn) {
        const uint64_t W = fp8_tail(T), Pn = Sb - W, R = G * T;  // pos >= Sb - W (attn_rows)
        const auto f8_one = g.input("f8_one", DType::F32, {1});
        const auto qs_s = g.input("f8_s128", DType::F32, {1}), ds_s = g.input("f8_d128", DType::F32, {1});
        // tail mask [T, W]: row t (position pos + t) sees cache rows c <= pos + t
        const auto posm1 = g.input("posm1", DType::F32, {1, 1});
        const auto iota_c = o.split(g.input("iota", DType::F32, {1, Sb}), 1, {Pn, W}, "mk_isp")[1];
        const auto iota_t = g.input("iota_t", DType::F32, {T, 1});
        const auto lim = o.add(o.add(posm1, o.constant(1.f, {1, 1}, "mk_one"), "mk_p"), iota_t, "mk_lim");
        const auto vis = o.binary("min", o.sub(lim, iota_c, "mk_d"), o.constant(0.f, {1, 1}, "mk_z"), "mk_m");
        const auto tmask = o.reshape(o.cast(o.mul(vis, o.constant(1e30f, {1, 1}, "mk_big"), "mk_c"), DType::BF16, "mk_b"),
                                     {1, 1, T, W}, "mk_4");
        auto f8 = [&](TensorId x, const std::string& n) {
            const auto codes = o.temp(n, DType::F8, o.shape(x));
            ns_ConvertToFp8::ParamsV2 cp{};
            cp.round_mode = CAST_ROUND_HALF_NE;
            g.op("convert_to_fp8_bf16", {x, f8_one}, {codes}, &cp, sizeof(cp));
            return codes;
        };
        ns_Sdpa::ParamsV7 fp{};
        fp.scale = sp.scale;
        fp.is_inference = false;  // row maxima and 1/sums as outputs (for the merge)
        fp.softmax_mode = SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
        fp.flags = 0x4002f00;  // as the framework sets it: d_scale_q/k/v, q_scale_s, d_scale_s present
        fp.wl = fp.wr = -1;
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto q8 = o.reshape(f8(o.reshape(qh[h], {R, D}, "fq2" + hs), "fq8" + hs), {1, G, T, D}, "fq4" + hs);
            const auto ks = o.split(g.input(P(j, "k_cache" + hs), kv_dt(), {Sb, D}), 0, {Pn, W}, "fks" + hs);
            const auto vs = o.split(g.input(P(j, "v_cache" + hs), kv_dt(), {Sb, D}), 0, {Pn, W}, "fvs" + hs);
            auto part = [&](TensorId kk, TensorId vv, uint64_t n, TensorId m4, const std::string& nm) {
                const auto oh = o.temp(nm + "o", DType::BF16, {1, G, T, D});
                const auto mx = o.temp(nm + "m", DType::BF16, {1, G, T, 1});
                const auto li = o.temp(nm + "l", DType::BF16, {1, G, T, 1});
                std::vector<TensorId> ins = {q8, o.reshape(kv8_ ? kk : f8(kk, nm + "k8"), {1, 1, n, D}, nm + "k4"),
                                             o.reshape(kv8_ ? vv : f8(vv, nm + "v8"), {1, 1, n, D}, nm + "v4"), m4, kNoTensor,
                                             f8_one, f8_one, f8_one, qs_s, kNoTensor, ds_s, kNoTensor, kNoTensor};
                const TensorId outs3[] = {oh, mx, li};
                g.op("sdpa_recomp_fwd_hf8", std::span<const TensorId>(ins), std::span<const TensorId>(outs3), &fp, sizeof(fp));
                return std::array<TensorId, 3>{o.cast(o.reshape(oh, {R, D}, nm + "o2"), DType::F32, nm + "of"),
                                               o.cast(o.reshape(mx, {R, 1}, nm + "m2"), DType::F32, nm + "mf"),
                                               o.cast(o.reshape(li, {R, 1}, nm + "l2"), DType::F32, nm + "lf")};
            };
            const auto a = part(ks[0], vs[0], Pn, kNoTensor, "fa" + hs);
            const auto b = part(ks[1], vs[1], W, tmask, "fb" + hs);
            const auto M = o.binary("max", a[1], b[1], "fmm" + hs);
            const auto wa = o.div(o.unary("exp", o.sub(a[1], M, "fda" + hs), "fea" + hs), a[2], "fwa" + hs);
            const auto wb = o.div(o.unary("exp", o.sub(b[1], M, "fdb" + hs), "feb" + hs), b[2], "fwb" + hs);
            const auto num = o.add(o.mul(a[0], wa, "fna" + hs), o.mul(b[0], wb, "fnb" + hs), "fnn" + hs);
            const auto oh = o.div(num, o.add(wa, wb, "fws" + hs), "fon" + hs);
            outs.push_back(o.reshape(o.cast(oh, DType::BF16, "fob" + hs), {1, R, D}, "sd3" + hs));
        }
    } else if (kernel_attn) {
        // per head: BF16 scores -> masked, unnormalised softmax kernel writing P in column chunks -> batched PV
        // over the chunks (keeps both MMEs busy despite the long reduction) -> sum over chunks / row sums
        const auto last = g.input("last", DType::I32, {T});
        const uint64_t C = Sb % 2048 == 0 ? 2048 : 1024, NC = Sb / C, R = G * T;
        // phases over all heads (QK, softmax kernel, PV) so the MMEs compute the next head's scores while the
        // TPCs exponentiate this head's (G2E_ATTN_HEAD_CHAINS=1: one chain per head)
        static const bool chains = std::getenv("G2E_ATTN_HEAD_CHAINS") != nullptr;
        // prefill pieces: FP16 scores + the 128-lane FP16 softmax kernel (3 more mantissa bits than BF16 scores, ~10%
        // faster at T = 512); verification keeps BF16 (faster there). G2E_ATTN_BF16_SCORES=1: BF16 everywhere
        std::vector<TensorId> qss(KV), scs(KV), prs(KV), sums(KV), vcs(KV);
        auto qscale = [&](uint64_t h) {  // unscaled q: the kernel applies the softmax scale (no TPC op ahead of QK)
            qss[h] = o.reshape(qh[h], {R, D}, "q2" + std::to_string(h));
        };
        auto qk = [&](uint64_t h) {
            const auto hs = std::to_string(h);
            const auto kc = g.input(P(j, "k_cache" + hs), kv_dt(), {Sb, D});
            vcs[h] = g.input(P(j, "v_cache" + hs), kv_dt(), {Sb, D});
            scs[h] = f16_scores ? o.linear_bf16(qss[h], kc, "sc" + hs, DType::F16) : kv_scores(o, qss[h], kc, "sc" + hs);  // [R, Sb]
        };
        auto softmax = [&](uint64_t h) {
            const auto hs = std::to_string(h);
            prs[h] = o.temp("pr" + hs, DType::BF16, {NC, R, C});
            sums[h] = o.temp("psum" + hs, DType::F32, {R, 1});
            const TensorId ins[] = {scs[h], last};
            const TensorId kouts[] = {prs[h], sums[h]};
            g.op(f16_scores ? "flashinfer_gaudi_masked_softmax_f16_gaudi2" : "flashinfer_gaudi_masked_softmax_bf16_gaudi2", ins,
                 kouts, &sp.scale, sizeof(float));
        };
        auto pv = [&](uint64_t h) {
            const auto hs = std::to_string(h);
            const auto acc = o.reduce("sum", kv_pv(o, prs[h], o.reshape(vcs[h], {NC, C, D}, "vc3" + hs), "pvc" + hs), 0,
                                      "pvs" + hs);  // [1, R, D]
            const auto den = kv8_ ? o.mul(sums[h], o.constant(128.f, {1}, "p128" + hs), "psum8" + hs) : sums[h];
            const auto oh = o.div(o.reshape(acc, {R, D}, "pv2" + hs), den, "pvn" + hs);
            outs.push_back(o.reshape(o.cast(oh, DType::BF16, "pvb" + hs), {1, R, D}, "sd3" + hs));
        };
        if (chains) {
            for (uint64_t h = 0; h < KV; ++h) qscale(h), qk(h), softmax(h), pv(h);
        } else {
            for (uint64_t h = 0; h < KV; ++h) qscale(h);
            for (uint64_t h = 0; h < KV; ++h) qk(h);
            for (uint64_t h = 0; h < KV; ++h) softmax(h);
            for (uint64_t h = 0; h < KV; ++h) pv(h);
        }
    } else if (graph_attn) {
        // all heads' QK first, then their softmaxes, then the PVs: independent chains the compiler can overlap
        // (MME of one head under the TPC softmax of another)
        std::vector<TensorId> pr(KV), vcs(KV);
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto kc = kv_in_bf16(o, P(j, "k_cache" + hs), Sb);
            vcs[h] = kv_in_bf16(o, P(j, "v_cache" + hs), Sb);
            // the softmax scale folded into q (1/16 for D = 256: exact in BF16)
            const auto qs = o.cast(o.mul(o.cast(o.reshape(qh[h], {G * T, D}, "q2" + hs), DType::F32, "qf" + hs),
                                         o.constant(sp.scale, {1}, "qsc" + hs), "qm" + hs), DType::BF16, "qb" + hs);
            pr[h] = o.reshape(o.linear_bf16(qs, kc, "sc" + hs, DType::BF16), {G, T, Sb}, "s3" + hs);
        }
        if (one_softmax) {  // one mask add + one softmax over all heads' scores
            std::vector<TensorId> parts;
            for (uint64_t h = 0; h < KV; ++h) parts.push_back(o.reshape(pr[h], {1, G, T, Sb}, "s4" + std::to_string(h)));
            const auto all = o.softmax(o.add(o.concat(parts, 0, "sall"), maskb, "small"), "sxall");
            const auto ps = o.split(o.reshape(all, {KV, G * T, Sb}, "p3"), 0, std::vector<uint64_t>(KV, 1), "ps");
            for (uint64_t h = 0; h < KV; ++h) pr[h] = o.reshape(ps[h], {G * T, Sb}, "p2" + std::to_string(h));
        } else {
            for (uint64_t h = 0; h < KV; ++h) {
                const auto hs = std::to_string(h);
                pr[h] = o.reshape(o.softmax(o.add(pr[h], maskb, "sm" + hs), "sx" + hs), {G * T, Sb}, "p2" + hs);
            }
        }
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto oh = o.temp("pv" + hs, DType::F32, {G * T, D});
            g.gemm(pr[h], vcs[h], oh);
            outs.push_back(o.reshape(o.cast(oh, DType::BF16, "pvb" + hs), {1, G * T, D}, "sd3" + hs));
        }
    }
    for (uint64_t h = 0; !graph_attn && !kernel_attn && !fp8_attn && h < KV; ++h) {  // SDPA
        const auto hs = std::to_string(h);
        const auto kc = o.reshape(kv_in_bf16(o, P(j, "k_cache" + hs), Sb), {1, 1, Sb, D}, "kc4" + hs);
        const auto vc = o.reshape(kv_in_bf16(o, P(j, "v_cache" + hs), Sb), {1, 1, Sb, D}, "vc4" + hs);
        const auto oh = o.temp("sd" + hs, DType::BF16, {1, G, T, D});
        std::vector<TensorId> ins = {o.reshape(qh[h], {1, G, T, D}, "q4" + hs), kc, vc, mask4, kNoTensor};
        for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
        g.op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(&oh, 1), &sp, sizeof(sp));
        outs.push_back(o.reshape(oh, {1, G * T, D}, "sd3" + hs));
    }
    const auto att2 = o.reshape(o.transpose(o.reshape(o.concat(outs, 0, "sdall"), {NH, T, D}, "a3"), {1, 0, 2}, "at"),
                                {T, NH * D}, "a2");
    std::pair<TensorId, TensorId> q8;
    if (fused_gate_quant()) {
        q8 = o.mul_quant(o.unary("sigmoid", gate, "gsig"), att2, "gq");
    } else {
        q8 = o.quantize(o.mul(att2, o.unary("sigmoid", gate, "gsig"), "gated"), "gq");
    }
    const auto y = o.fp8_gemm(q8.first, q8.second, o_w, o_s, P(j, "oproj"));
    const auto out = mlp(o, j, y, resid);
    output(g, "res_out", out.res);
    output(g, "delta_out", out.delta);
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Recipe& Qwen35::embed_recipe(uint32_t T) {
    const std::string key = "qwen35_embed_T" + std::to_string(T);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto table = g.input("embed", DType::BF16, {uint64_t(cfg_.vocab), uint64_t(cfg_.hidden)});
    output(g, "res", o.gather_rows(table, g.input("ids", DType::I32, {T}), "rows"));
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Recipe& Qwen35::capture_recipe(uint32_t T) {
    const std::string ck = "qwen35_capture_T" + std::to_string(T);
    if (auto it = recipes_.find(ck); it != recipes_.end()) return *it->second;
    GraphBuilder cg(dev_);
    Ops co(cg);
    const auto rr = cg.input("res", DType::BF16, {T, uint64_t(cfg_.hidden)});
    const auto dd = cg.input("delta", DType::BF16, {T, uint64_t(cfg_.hidden)});
    output(cg, "feat", co.add(rr, dd, "sum"));
    auto r = cg.compile(ck);
    return *(recipes_[ck] = std::move(r));
}

void Qwen35::precompile(const std::function<void(size_t, size_t)>& progress) {
    // the token counts a request produces: prefill pieces (512 and the power-of-two remainders), decode (1),
    // verification (kSpecMaxT, spec mode)
    struct Shape { uint32_t T; bool spec; };
    std::vector<Shape> shapes;
    for (uint32_t T = 1; T <= kMaxT; T *= 2) shapes.push_back({T, false});
    shapes.push_back({kSpecMaxT, true});
    std::vector<uint32_t> buckets;
    for (uint32_t rows = 1;;) {
        const uint32_t b = bucket_for(rows);
        buckets.push_back(b);
        if (b >= max_ctx_) break;
        rows = b + 1;
    }
    uint32_t attn = 0, gdn = 0;
    while (attn < cfg_.layers && !cfg_.attention[attn]) ++attn;
    while (gdn < cfg_.layers && cfg_.attention[gdn]) ++gdn;
    // batched verification (verify_batch): per batch size, then the per-slot attention for every context length
    const bool batched = block_ == 1 && wy_geometry() && lm8_on_ && !k_cache_.empty();
    std::vector<uint32_t> core;
    if (batched)
        for (uint32_t rows = 1024;; rows = rows < 2048 ? 2048 : rows + 2048) {
            core.push_back(core_rows(rows));
            if (rows >= max_ctx_) break;
        }
    const size_t total = shapes.size() * (buckets.size() + 3) + 2 + (batched ? 2 * (kMaxBatch - 1 + core.size()) : 0);
    size_t done = 0;
    auto tick = [&] { if (progress) progress(++done, total); };
    for (const auto& s : shapes) {
        building_spec_ = s.spec;
        if (gdn < cfg_.layers) {
            block_recipe(gdn - gdn % block_, s.T, 0);
            for (uint32_t f = 0, n = 1; f < cfg_.layers; f += n) {  // the GDN groups run_layers launches
                n = group_at(f, cfg_.layers);
                if (cfg_.attention[f]) continue;
                launch_n_ = n;
                block_recipe(f, s.T, 0);
                launch_n_ = 0;
            }
        }
        tick();
        if (attn < cfg_.layers && block_ == 1)  // the fine rows of attn_rows (verification, decode, prefill pieces)
            for (uint32_t rows = kAttnFine + 1; rows <= max_ctx_; rows += kAttnStep) {
                const uint32_t b = attn_rows(rows, s.T);
                if (std::find(buckets.begin(), buckets.end(), b) != buckets.end()) continue;
                attn_pre_recipe(s.T);
                attn_post_recipe(s.T, b);
            }
        for (uint32_t b : buckets) {
            if (attn < cfg_.layers) {
                if (block_ == 1 && attn_sdpa(b)) {  // long contexts: the split attention launches (attn_split)
                    attn_pre_recipe(s.T);
                    attn_post_recipe(s.T, b);
                } else {
                    block_recipe(attn - attn % block_, s.T, b);
                }
            }
            tick();
        }
        building_spec_ = false;
        if (!k_cache_.empty()) commit_recipe(s.T);
        tick();
        if (!capture_.empty()) capture_recipe(s.T);
        embed_recipe(s.T);
        tick();
    }
    head_recipe();
    tick();
    verify_head_recipe(kSpecMaxT);
    verify_topk_recipe(kSpecMaxT);
    spec_commit_recipe(kSpecMaxT, spec_wy(kSpecMaxT));
    tick();
    if (batched) {  // widths 16 and 8 (the batched rounds of many slots)
        for (uint32_t T : {kSpecMaxT, 8u}) {
            for (uint32_t B = 2; B <= kMaxBatch; ++B) {
                const uint32_t R = B * T;
                gdn_batch_recipe(B, T);
                apre_batch_recipe(B, T);
                apost_batch_recipe(B, T);
                embed_recipe(R);
                if (!capture_.empty()) capture_recipe(R);
                verify_head_recipe(R);
                verify_topkb_recipe(R);
                spec_commit_batch_recipe(B, T, true);
                tick();
            }
            for (uint32_t Sb : core) {
                acore_recipe(T, Sb);
                tick();
            }
        }
    }
}

Recipe& Qwen35::commit_recipe(uint32_t T) {
    const std::string key = "qwen35_commit_T" + std::to_string(T);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim, NA = k_cache_.size();
    GraphBuilder g(dev_);
    Ops o(g);
    const auto scratch = g.input("scratch", DType::BF16, {NA * 2 * KV, T, D});
    const auto rows = o.split(scratch, 0, std::vector<uint64_t>(NA * 2 * KV, 1), "rows");
    for (uint64_t n = 0; n < NA * 2 * KV; ++n) {
        const auto out = g.input("c" + std::to_string(n), kv_dt(), {T, D});
        if (!kv8_) {
            g.reshape(rows[n], out);
            continue;
        }
        ns_ConvertToFp8::ParamsV2 cp{};
        cp.round_mode = CAST_ROUND_HALF_NE;
        g.op("convert_to_fp8_bf16", {o.reshape(rows[n], {T, D}, "r2_" + std::to_string(n)), o.constant(1.f, {1}, "one" + std::to_string(n))},
             {out}, &cp, sizeof(cp));
    }
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Recipe& Qwen35::head_recipe() {
    if (head_) return *head_;
    const uint64_t H = cfg_.hidden, V = cfg_.vocab;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto res = g.input("res", DType::BF16, {1, H});
    const auto delta = g.input("delta", DType::BF16, {1, H});
    const auto nw = g.input("final_norm1p", DType::BF16, {H});
    const auto w = g.input("lm_head", DType::BF16, {V, H});
    const auto nq = o.add_rmsnorm_quant(delta, res, nw, cfg_.eps, "final");
    output(g, "logits", o.linear_bf16(nq.normalized, w, "lg", DType::F32));
    head_ = g.compile("qwen35_head2");
    return *head_;
}

void Qwen35::set_capture(std::vector<uint32_t> layers) {
    capture_ = std::move(layers);
    if (!capture_.empty()) feats_ = DeviceBuffer(dev_, capture_.size() * feature_stride());
}

std::vector<float> Qwen35::prefill(std::span<const int32_t> tokens, const std::function<void(uint32_t, uint32_t)>& on_chunk) {
    std::vector<float> logits;
    std::map<std::string, double> prefill_prof;
    size_t done = 0;
    while (done < tokens.size()) {
        static const size_t piece = [] {  // G2E_PREFILL_PIECE: smaller full pieces (a power of two <= kMaxT)
            const char* e = std::getenv("G2E_PREFILL_PIECE");
            size_t v = e ? size_t(std::atoi(e)) : kMaxT;
            return v >= 1 && v <= kMaxT && (v & (v - 1)) == 0 ? v : size_t(kMaxT);
        }();
        size_t n = std::min<size_t>(piece, tokens.size() - done);
        // a remainder of 2..15 tokens: one padded verification forward (forward_short) instead of its power-of-two
        // pieces; G2E_PREFILL_POW2=1 keeps the pieces
        static const bool pow2 = std::getenv("G2E_PREFILL_POW2") != nullptr;
        const bool short_piece = !pow2 && n >= 2 && n < kSpecMaxT && block_ == 1 && spec_wy(kSpecMaxT);
        if (n < piece && !short_piece) {  // largest power of two <= n
            size_t p = 1;
            while (p * 2 <= n) p *= 2;
            n = p;
        }
        const bool last = done + n == tokens.size();
        const uint32_t at = pos_;
        static const bool plog = std::getenv("G2E_PREFILL_LOG") != nullptr;  // every forward's wall time
        const auto f0 = std::chrono::steady_clock::now();
        logits = short_piece ? forward_short(tokens.subspan(done, n), last) : forward(tokens.subspan(done, n), nullptr, last);
        if (plog) {
            stream_.synchronize();
            std::fprintf(stderr, "[prefill] %zu rows at %u: %.1f ms (launches up to %llu)\n", n, at,
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - f0).count(),
                         (unsigned long long)launch_count());
        }
        if (profile_) {  // device time per recipe kind (context rows stripped) over the whole prefill
            for (const auto& [name, ms] : prof_.by_recipe) {
                std::string kind = name;
                if (const auto at_s = kind.find("_S"); at_s != std::string::npos) {
                    auto e = at_s + 2;
                    while (e < kind.size() && std::isdigit(static_cast<unsigned char>(kind[e]))) ++e;
                    kind.erase(at_s, e - at_s);
                }
                prefill_prof[kind] += ms;
            }
            prefill_prof["(commit)"] += prof_.commit_ms;
        }
        if (on_chunk) on_chunk(at, uint32_t(n));
        done += n;
    }
    if (profile_ && tokens.size() > kMaxT) {
        std::vector<std::pair<double, std::string>> v;
        double total = 0;
        for (const auto& [k, ms] : prefill_prof) v.push_back({ms, k}), total += ms;
        std::sort(v.rbegin(), v.rend());
        std::fprintf(stderr, "prefill profile (%zu tokens): %.0f ms device;", tokens.size(), total);
        for (size_t k = 0; k < std::min<size_t>(v.size(), 8); ++k) std::fprintf(stderr, " %s %.0f", v[k].second.c_str(), v[k].first);
        std::fprintf(stderr, "\n");
    }
    return logits;
}

std::vector<float> Qwen35::prefill_images(std::span<const int32_t> tokens, std::span<const ImageSpan> images,
                                          const std::function<void(uint32_t, uint32_t)>& on_chunk) {
    if (images.empty()) return prefill(tokens, on_chunk);
    if (mrope_section_.size() != 3) throw std::runtime_error("qwen35: images need the multimodal RoPE sections");
    // rotary (t, h, w) positions of every token
    const size_t L = tokens.size();
    std::vector<std::array<int64_t, 3>> p3(L);
    int64_t cur = int64_t(pos_) + rope_delta_;
    size_t next_img = 0;
    for (size_t i = 0; i < L;) {
        if (next_img < images.size() && images[next_img].start == i) {
            const auto& im = images[next_img++];
            if (i + im.tokens() > L) throw std::invalid_argument("qwen35: image placeholders past the prompt");
            for (uint32_t k = 0; k < im.tokens(); ++k)
                p3[i + k] = {cur, cur + k / im.grid_w, cur + k % im.grid_w};
            cur += std::max(im.grid_h, im.grid_w);
            i += im.tokens();
        } else {
            p3[i] = {cur, cur, cur};
            ++cur;
            ++i;
        }
    }
    if (next_img != images.size()) throw std::invalid_argument("qwen35: image spans out of order or overlapping");
    const uint64_t R2 = cfg_.rotary_dim / 2;
    std::vector<double> inv(R2);
    for (uint64_t i = 0; i < R2; ++i) inv[i] = std::pow(cfg_.rope_theta, -double(2 * i) / double(cfg_.rotary_dim));
    auto dim_of = [&](uint64_t i) {  // interleaved sections: h at 1, 4, ... (< 3 * sec[1]), w at 2, 5, ... (< 3 * sec[2])
        if (i % 3 == 1 && i < 3ull * mrope_section_[1]) return 1;
        if (i % 3 == 2 && i < 3ull * mrope_section_[2]) return 2;
        return 0;
    };
    std::vector<float> logits;
    size_t done = 0;
    while (done < L) {
        size_t n = std::min<size_t>(kMaxT, L - done);
        if (n < kMaxT) {
            size_t p = 1;
            while (p * 2 <= n) p *= 2;
            n = p;
        }
        // text-only chunk with consecutive positions: the table; otherwise explicit rows
        bool plain = true;
        for (size_t k = done; k < done + n && plain; ++k)
            plain = p3[k][0] == p3[k][1] && p3[k][1] == p3[k][2] && p3[k][0] == p3[done][0] + int64_t(k - done);
        if (plain) {
            rope_delta_ = p3[done][0] - int64_t(pos_);
        } else {
            rope_rows_.assign(2 * n * R2, 0.f);
            for (size_t k = 0; k < n; ++k)
                for (uint64_t i = 0; i < R2; ++i) {
                    const double a = double(p3[done + k][dim_of(i)]) * inv[i];
                    rope_rows_[k * R2 + i] = float(std::cos(a));
                    rope_rows_[n * R2 + k * R2 + i] = float(std::sin(a));
                }
            rope_explicit_ = true;
        }
        for (const auto& im : images) {  // features of the image rows in this chunk
            const size_t a = std::max<size_t>(done, im.start), b = std::min<size_t>(done + n, im.start + im.tokens());
            if (a < b)
                inject_.push_back({im.feats, im.offset + uint64_t(a - im.start) * cfg_.hidden * 2, uint32_t(a - done),
                                   uint32_t(b - a)});
        }
        const bool last = done + n == L;
        const uint32_t at = pos_;
        logits = forward(tokens.subspan(done, n), nullptr, last);
        if (on_chunk) on_chunk(at, uint32_t(n));
        done += n;
    }
    rope_delta_ = cur - int64_t(pos_);  // the next token continues the rotary positions
    return logits;
}

std::vector<float> Qwen35::forward(std::span<const int32_t> tokens, std::vector<std::vector<uint16_t>>* dump,
                                   bool want_logits) {
    if (pending_T_ || pending_total_) throw std::logic_error("qwen35: accept() the pending verification first");
    const uint32_t T = uint32_t(tokens.size());
    const uint64_t row = uint64_t(cfg_.hidden) * 2;
    Event e0(dev_), e1(dev_), eb(dev_);
    e0.record(stream_);
    run_layers(tokens, false, dump);
    eb.record(stream_);
    if (!want_logits) {
        e1.record(stream_);
        stream_.synchronize();
        last_ms_ = Event::elapsed_ms(e0, e1);
        pos_ += T;
        return {};
    }
    auto& hr = head_recipe();
    std::vector<Binding> hb = {{"res", &res_[cur_], (T - 1) * row},
                               {"delta", &delta_[cur_], (T - 1) * row},
                               {"final_norm1p", &aux_, aux_off_.at("final_norm1p")},
                               w_.bind("lm_head", "lm_head"),
                               {"logits", &logits_, 0}};
    hr.enqueue(stream_, hb);
    e1.record(stream_);
    stream_.download(logits_, 0, staging_, 0, uint64_t(cfg_.vocab) * 4);
    stream_.synchronize();
    last_ms_ = Event::elapsed_ms(e0, e1);
    if (profile_) {
        prof_.head_ms = Event::elapsed_ms(eb, e1);
        prof_.blocks_ms = Event::elapsed_ms(e0, eb);
    }
    pos_ += T;
    return std::vector<float>(staging_.as<float>(), staging_.as<float>() + cfg_.vocab);
}

std::vector<float> Qwen35::forward_short(std::span<const int32_t> tokens, bool want_logits) {
    if (pending_T_ || pending_total_) throw std::logic_error("qwen35: accept() the pending verification first");
    const uint32_t n = uint32_t(tokens.size()), T = kSpecMaxT;
    if (n < 2 || n >= T) throw std::invalid_argument("qwen35: forward_short takes 2..15 tokens");
    std::vector<int32_t> block(tokens.begin(), tokens.end());
    block.resize(T, tokens.back());
    Event e0(dev_), e1(dev_);
    e0.record(stream_);
    run_layers(block, true, nullptr);
    pending_T_ = T;
    stash_b_ = 0;
    pending_total_ = 1;
    pending_wy_ = spec_wy(T);
    // commit rows 0..n-1 (pos += n) first: its controls go through staging_ (kAcceptStage), which the logits'
    // download below overwrites; the commit does not touch the residual stream the head reads
    accept(n - 1);
    if (want_logits) {  // row n - 1 of the verified rows
        const uint64_t row = uint64_t(cfg_.hidden) * 2;
        std::vector<Binding> hb = {{"res", &res_[cur_], (n - 1) * row},
                                   {"delta", &delta_[cur_], (n - 1) * row},
                                   {"final_norm1p", &aux_, aux_off_.at("final_norm1p")},
                                   w_.bind("lm_head", "lm_head"),
                                   {"logits", &logits_, 0}};
        head_recipe().enqueue(stream_, hb);
        stream_.download(logits_, 0, staging_, 0, uint64_t(cfg_.vocab) * 4);
    }
    e1.record(stream_);
    stream_.synchronize();
    last_ms_ = Event::elapsed_ms(e0, e1);
    if (!want_logits) return {};
    return std::vector<float>(staging_.as<float>(), staging_.as<float>() + cfg_.vocab);
}

std::vector<int32_t> Qwen35::verify(std::span<const int32_t> tokens, const std::function<void()>& launched) {
    if (pending_T_ || pending_total_) throw std::logic_error("qwen35: accept() the pending verification first");
    const uint32_t T = uint32_t(tokens.size());
    if (T < 2 || T > kSpecMaxT || (T & (T - 1))) throw std::invalid_argument("qwen35: verify width must be 2..16, a power of two");
    Event e0(dev_), e1(dev_);
    e0.record(stream_);
    run_layers(tokens, true, nullptr);
    auto& hr = verify_head_recipe(T);
    std::vector<Binding> hb = {{"res", &res_[cur_], 0},
                               {"delta", &delta_[cur_], 0},
                               {"final_norm1p", &aux_, aux_off_.at("final_norm1p")},
                               w_.bind("lm_head", "lm_head")};
    if (lm8_on_) {
        hb.push_back({"lm8", &lm8_, 0});
        hb.push_back({"lm8s", &lm8s_, 0});
        hb.push_back({"tkc_consts", &aux_, aux_off_.at("topk2c_T" + std::to_string(T))});
        hb.push_back({"cand8", &argmax_, 0});
        hb.push_back({"best", &argmax_, kBestOff});
    } else {
        hb.push_back({"argmax", &argmax_, 0});
    }
    hr.enqueue(stream_, hb);
    e1.record(stream_);
    stream_.download(argmax_, 0, staging_, 0, lm8_on_ ? kBestOff + uint64_t(T) * 4 : uint64_t(T) * 4);
    if (!done_ev_) done_ev_ = std::make_unique<Event>(dev_);
    done_ev_->record(stream_);
    if (launched) launched();
    done_ev_->synchronize();
    last_ms_ = Event::elapsed_ms(e0, e1);
    pending_T_ = T;
    stash_b_ = 0;
    pending_total_ = 1;
    pending_wy_ = spec_wy(T);
    if (lm8_on_) {  // cand8 [T, kHeadCand] at 0, best [T] at kBestOff
        const auto* cand = staging_.as<int32_t>();
        const auto* best = reinterpret_cast<const int32_t*>(staging_.as<uint8_t>() + kBestOff);
        std::vector<int32_t> out(T);
        for (uint32_t t = 0; t < T; ++t) out[t] = cand[t * kHeadCand + uint32_t(best[t])];
        return out;
    }
    return std::vector<int32_t>(staging_.as<int32_t>(), staging_.as<int32_t>() + T);
}

Qwen35::VerifyTop Qwen35::verify_topk(std::span<const int32_t> tokens, float inv_temp,
                                      const std::function<void()>& launched) {
    if (pending_T_ || pending_total_) throw std::logic_error("qwen35: accept() the pending verification first");
    const uint32_t T = uint32_t(tokens.size());
    if (T < 2 || T > kSpecMaxT || (T & (T - 1))) throw std::invalid_argument("qwen35: verify width must be 2..16, a power of two");
    Event e0(dev_), e1(dev_);
    e0.record(stream_);
    run_layers(tokens, true, nullptr);
    *reinterpret_cast<float*>(staging_.as<uint8_t>() + kTempStage) = inv_temp;  // not the cos/sin or ids region
    stream_.upload(staging_, kTempStage, spec_ctl_, 3072, 4);
    auto& hr = verify_topk_recipe(T);
    // outputs in vtop_: values at 0, ids at 4096, max at 8192, sum at 8320
    std::vector<Binding> hb = {{"res", &res_[cur_], 0},
                               {"delta", &delta_[cur_], 0},
                               {"final_norm1p", &aux_, aux_off_.at("final_norm1p")},
                               w_.bind("lm_head", "lm_head"),
                               {"inv_temp", &spec_ctl_, 3072},
                               {"tk_consts", &aux_, aux_off_.at("topk2_T" + std::to_string(T))},
                               {"top_v", &vtop_, 0},
                               {"top_i", &vtop_, 4096},
                               {"mx", &vtop_, 8192},
                               {"sum", &vtop_, 8320}};
    hr.enqueue(stream_, hb);
    e1.record(stream_);
    stream_.download(vtop_, 0, staging_, 0, 8448);
    if (!done_ev_) done_ev_ = std::make_unique<Event>(dev_);
    done_ev_->record(stream_);
    if (launched) launched();
    done_ev_->synchronize();
    last_ms_ = Event::elapsed_ms(e0, e1);
    pending_T_ = T;
    stash_b_ = 0;
    pending_total_ = 1;
    pending_wy_ = spec_wy(T);
    VerifyTop v;
    const auto* base = staging_.as<uint8_t>();
    const auto* vals = reinterpret_cast<const float*>(base);
    const auto* ids = reinterpret_cast<const int32_t*>(base + 4096);
    v.values.assign(vals, vals + T * kTopK);
    v.ids.assign(ids, ids + T * kTopK);
    v.max.assign(reinterpret_cast<const float*>(base + 8192), reinterpret_cast<const float*>(base + 8192) + T);
    v.sum.assign(reinterpret_cast<const float*>(base + 8320), reinterpret_cast<const float*>(base + 8320) + T);
    return v;
}

Recipe& Qwen35::verify_topk_recipe(uint32_t T) {
    const std::string key = "qwen35_vtop2_T" + std::to_string(T);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, V = cfg_.vocab;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto res = g.input("res", DType::BF16, {T, H});
    const auto delta = g.input("delta", DType::BF16, {T, H});
    const auto nw = g.input("final_norm1p", DType::BF16, {H});
    const auto w = g.input("lm_head", DType::BF16, {V, H});
    const auto inv_temp = g.input("inv_temp", DType::F32, {1});
    const auto nq = o.add_rmsnorm_quant(delta, res, nw, cfg_.eps, "final");
    const auto scaled = o.mul(o.linear_bf16(nq.normalized, w, "lg", DType::F32), inv_temp, "scaled");  // [T, V]
    const auto top = o.topk2(scaled, kTopK, kTopkChunk, g.input("tk_consts", DType::I32, {V + 2 * uint64_t(T)}), "tk");
    output(g, "top_v", top.first);
    output(g, "top_i", top.second);
    const auto mx = o.reduce("max", scaled, 1, "mx_v");
    output(g, "mx", mx);
    output(g, "sum", o.reduce("sum", o.unary("exp", o.sub(scaled, mx, "shift"), "ex"), 1, "sum_v"));
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

void Qwen35::accept(uint32_t a) {
    const uint32_t T = pending_T_;
    if (!T || a >= T) throw std::logic_error("qwen35: accept without a pending verification or a >= width");
    // controls: mask [T,1] f32 at 0 (1 for rows 0..a), conv rows [3] i32 at 2048 (a+1 .. a+3 of [state; tokens])
    // its own staging region (no host write touches it before the upload has run: no synchronisation needed)
    // (one region per batch index: the accepts of a batched verification are queued back to back)
    const uint64_t ctl = uint64_t(stash_b_) * 4096;
    auto* m = reinterpret_cast<float*>(staging_.as<uint8_t>() + kAcceptStage + ctl);
    for (uint32_t t = 0; t < T; ++t) m[t] = t <= a ? 1.f : 0.f;
    auto* rows = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kAcceptStage + ctl + 2048);
    for (uint32_t k = 0; k + 1 < cfg_.conv_kernel; ++k) rows[k] = int32_t(a + 1 + k);
    stream_.upload(staging_, kAcceptStage + ctl, spec_ctl_, ctl, 4096);
    std::vector<Binding> b = {{"mask", &spec_ctl_, ctl}, {"rows", &spec_ctl_, ctl + 2048}};
    const std::string cp = "chunk" + std::to_string(T) + ".";
    std::vector<std::string> names = {"upper", "strict", "inclusive", "lower", "eye", "diag8"};
    for (uint32_t sz = 8; sz < T; sz *= 2) names.push_back("ll" + std::to_string(sz));
    if (T >= 8) for (const auto& nm : names) b.push_back({cp + nm, &aux_, aux_off_.at(cp + nm)});
    const auto L = stash_layout(T, pending_wy_);
    const char* fields[] = {"q", "k", "v", "g", "b", "seq"};
    if (pending_wy_) fields[0] = "x";
    const uint64_t NG = std::max(num_gdn_, 1u);
    for (int f = 0; f < 6; ++f)
        if (L.size[f]) b.push_back({fields[f], &spec_stash_, L.base[f] + uint64_t(stash_b_) * NG * L.size[f]});
    b.push_back({"ssm_in", &ssm_all_[ssm_side_[0]], 0});
    b.push_back({"ssm_out", &ssm_all_[ssm_side_[0] ^ 1], 0});
    b.push_back({"conv_out", &conv_all_[conv_side_[0] ^ 1], 0});
    spec_commit_recipe(T, pending_wy_).enqueue(stream_, b);
    for (auto& side : ssm_side_) side ^= 1;
    for (auto& side : conv_side_) side ^= 1;
    pending_T_ = 0;
    if (pending_total_) --pending_total_;
    pos_ += a + 1;
}

Qwen35::StashLayout Qwen35::stash_layout(uint32_t T, bool wy) const {
    StashLayout L{};
    const uint64_t NG = std::max(num_gdn_, 1u);
    L.size[0] = L.size[1] = uint64_t(T) * cfg_.lin_k_heads * cfg_.lin_k_dim * 4;
    if (wy) {  // X = [k; q] per key head in field 0
        L.size[0] *= 2;
        L.size[1] = 0;
    }
    L.size[2] = uint64_t(T) * cfg_.lin_v_heads * cfg_.lin_v_dim * 4;
    L.size[3] = L.size[4] = uint64_t(T) * cfg_.lin_v_heads * 4;
    L.size[5] = uint64_t(T + cfg_.conv_kernel - 1) * cfg_.conv_channels() * 2;
    uint64_t off = 0;
    for (int f = 0; f < 6; ++f) {  // field f of batch index b, GDN layer s at base[f] + (b * NG + s) * size[f]
        L.base[f] = off;
        off = align128(off + uint64_t(kMaxBatch) * NG * L.size[f]);
    }
    return L;
}

void Qwen35::quantize_lm_head() {
    lm8_on_ = std::getenv("G2E_HEAD_BF16") == nullptr;
    if (!lm8_on_) return;
    const uint64_t V = cfg_.vocab, H = cfg_.hidden, piece = 15520;  // piece * 4 bytes stays 128-aligned
    if (V % piece) {
        lm8_on_ = false;
        return;
    }
    lm8_ = DeviceBuffer(dev_, V * H);
    lm8s_ = DeviceBuffer(dev_, V * 4);
    const std::string key = "qwen35_lmq_" + std::to_string(piece) + "x" + std::to_string(H);
    auto it = recipes_.find(key);
    if (it == recipes_.end()) {
        GraphBuilder g(dev_);
        Ops o(g);
        const auto [codes, scale] = o.quantize(g.input("w", DType::BF16, {piece, H}), "q");
        output(g, "codes", codes);
        output(g, "scale", scale);
        it = recipes_.emplace(key, g.compile(key)).first;
    }
    const auto lm = w_.bind("w", "lm_head");
    for (uint64_t r = 0; r < V; r += piece) {
        const Binding b[] = {{"w", lm.buffer, lm.offset + r * H * 2}, {"codes", &lm8_, r * H}, {"scale", &lm8s_, r * 4}};
        it->second->enqueue(stream_, b);
    }
    stream_.synchronize();
}

Recipe& Qwen35::verify_head_recipe(uint32_t T) {
    const std::string key = "qwen35_vhead" + std::string(lm8_on_ ? "8" : "") + "_T" + std::to_string(T);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, V = cfg_.vocab;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto res = g.input("res", DType::BF16, {T, H});
    const auto delta = g.input("delta", DType::BF16, {T, H});
    const auto nw = g.input("final_norm1p", DType::BF16, {H});
    const auto w = g.input("lm_head", DType::BF16, {V, H});
    const auto nq = o.add_rmsnorm_quant(delta, res, nw, cfg_.eps, "final");
    if (!lm8_on_) {
        output(g, "argmax", o.argmax(o.linear_bf16(nq.normalized, w, "lg", DType::F32), "am"));
    } else {
        // FP8 logits pick kHeadCand candidates per row; their BF16 rows give the exact logits. The host takes
        // cand[t][best[t]] (exact unless the true maximum falls outside the FP8 top kHeadCand).
        const uint64_t K = kHeadCand;
        const auto approx = o.fp8_gemm(nq.codes, nq.scale, g.input("lm8", DType::F8, {V, H}), g.input("lm8s", DType::F32, {V}), "lg8");
        const auto cand = o.topk2(approx, kHeadCand, kTopkChunk, g.input("tkc_consts", DType::I32, {V + 2 * uint64_t(T)}), "tkc").second;
        const auto rows = o.reshape(o.gather_rows(w, o.reshape(cand, {T * K}, "cflat"), "crow"), {T, K, H}, "crow3");
        const auto exact = o.batch_gemm(o.reshape(nq.normalized, {T, 1, H}, "n3"), rows, "ex", false, true, DType::F32);  // [T,1,K]
        output(g, "cand8", cand);
        output(g, "best", o.argmax(exact, "best_i"));
    }
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

// Commit of a verification: per GDN layer, the state after the accepted rows (rejected rows get beta = g = 0, so
// they leave the state unchanged) and the conv window ending at the last accepted row.
Recipe& Qwen35::spec_commit_recipe(uint32_t T, bool raw_qk) {  // raw_qk: the stash holds un-normalised q/k
    const std::string key = "qwen35_speccommit2_T" + std::to_string(T) + (raw_qk ? "_x" : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t HK = cfg_.lin_k_heads, HV = cfg_.lin_v_heads, DK = cfg_.lin_k_dim, DV = cfg_.lin_v_dim,
                   C = cfg_.conv_channels(), KC = cfg_.conv_kernel, NG = num_gdn_;
    GraphBuilder g(dev_);
    Ops o(g);
    build_inputs_.clear();
    // every GDN layer at once: the layers become extra heads (value head l*HV+h uses key head l*HK+h/REP)
    const auto mask = g.input("mask", DType::F32, {T, 1});
    const auto rows = g.input("rows", DType::I32, {KC - 1});
    const auto ssm_in = g.input("ssm_in", DType::F32, {NG * HV, DV, DK});
    auto rows_first = [&](TensorId x, Shape per_layer, uint64_t heads, const std::string& nm) {
        // [NG, T, heads, d...] -> [T, NG*heads, d...]
        std::vector<unsigned> perm(per_layer.size());
        for (unsigned i = 0; i < perm.size(); ++i) perm[i] = i;
        std::swap(perm[0], perm[1]);
        Shape out = {T, NG * heads};
        for (size_t i = 3; i < per_layer.size(); ++i) out.push_back(per_layer[i]);
        return o.reshape(o.transpose(x, perm, nm + "_t"), out, nm + "_r");
    };
    TensorId q, k;
    if (raw_qk) {  // X [NG, HK, 2T, DK]: rows 0..T-1 k, T..2T-1 q -> [T, NG*HK, DK]
        const auto xs = o.split(g.input("x", DType::F32, {NG, HK, 2 * T, DK}), 2, {T, T}, "xs");
        k = o.reshape(o.transpose(xs[0], {2, 0, 1, 3}, "xk_t"), {T, NG * HK, DK}, "xk");
        q = o.reshape(o.transpose(xs[1], {2, 0, 1, 3}, "xq_t"), {T, NG * HK, DK}, "xq");
    } else {
        q = rows_first(g.input("q", DType::F32, {NG, T, HK, DK}), {NG, T, HK, DK}, HK, "q");
        k = rows_first(g.input("k", DType::F32, {NG, T, HK, DK}), {NG, T, HK, DK}, HK, "k");
    }
    if (raw_qk) {  // as gdn_layer: x / sqrt(sum x^2 + 1e-6), q also scaled by DK^-0.5
        auto l2 = [&](TensorId t3, const std::string& n) {
            const auto ss = o.add(o.reduce("sum", o.mul(t3, t3, n + "sq"), 2, n + "ss"), o.constant(1e-6f, {1}, n + "eps"), n + "sse");
            return o.mul(t3, o.unary("rsqrt", ss, n + "rs"), n + "n");
        };
        q = o.mul(l2(q, "ql2"), o.constant(float(1.0 / std::sqrt(double(DK))), {1}, "qscale"), "qs");
        k = l2(k, "kl2");
    }
    const auto v = rows_first(g.input("v", DType::F32, {NG, T, HV, DV}), {NG, T, HV, DV}, HV, "v");
    const auto gd = o.mul(rows_first(g.input("g", DType::F32, {NG, T, HV}), {NG, T, HV}, HV, "g"), mask, "gm");
    const auto bt = o.mul(rows_first(g.input("b", DType::F32, {NG, T, HV}), {NG, T, HV}, HV, "b"), mask, "bm");
    const auto [core, s_next] = gdn_chunk(o, "c_", T, NG * HK, NG * HV, ssm_in, q, k, v, gd, bt);
    (void)core;
    output(g, "ssm_out", s_next);
    const auto seq = g.input("seq", DType::BF16, {NG, T + KC - 1, C});
    output(g, "conv_out", o.gather(seq, rows, 1, "win"));
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

uint32_t Qwen35::core_bucket(uint32_t rows) {
    uint32_t b = 1024;
    while (b < rows) {
        const uint32_t mid = b + b / 2;
        b = mid >= rows && mid % 1024 == 0 ? mid : b * 2;
    }
    return b;
}

Recipe& Qwen35::acore_batch_recipe(uint32_t B, uint32_t T, uint32_t Sb) {
    // acore_recipe for B slots in one graph: slot b's queries are rows b*T.. of q_in, its visible rows last[b], its
    // cache k/v_cache<b>_<h> (Sb rows: the bucket of the longest slot; the others' extra rows are masked)
    const std::string key = "qwen35_acoreb_B" + std::to_string(B) + "_T" + std::to_string(T) + "_S" + std::to_string(Sb);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    if (Sb % 1024) throw std::logic_error("qwen35: acore rows must be a multiple of 1024");
    const uint64_t NH = cfg_.heads, KV = cfg_.kv_heads, D = cfg_.head_dim, G = NH / KV;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto qall = o.split(g.input("q_in", DType::BF16, {uint64_t(B) * T, NH, D}), 0, std::vector<uint64_t>(B, T), "qsl");
    const auto lall = o.split(g.input("last", DType::I32, {B, 32}), 0, std::vector<uint64_t>(B, 1), "lsl");
    const float scale = float(1.0 / std::sqrt(double(D)));
    const uint64_t C = Sb % 2048 == 0 ? 2048 : 1024, NC = Sb / C, R = G * T;
    std::vector<TensorId> atts;
    for (uint32_t b = 0; b < B; ++b) {
        const std::string x = "b" + std::to_string(b) + "_";
        const auto last = o.split(o.reshape(lall[b], {32}, x + "l1"), 0, {T, 32 - T}, x + "lt")[0];
        const auto qh = o.split(o.reshape(o.transpose(qall[b], {1, 0, 2}, x + "qt"), {KV, G * T, D}, x + "q3"), 0,
                                std::vector<uint64_t>(KV, 1), x + "qh");
        std::vector<TensorId> scs(KV), prs(KV), sums(KV), vcs(KV), outs;
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto kc = g.input("k_cache" + std::to_string(b) + "_" + hs, kv_dt(), {Sb, D});
            vcs[h] = g.input("v_cache" + std::to_string(b) + "_" + hs, kv_dt(), {Sb, D});
            scs[h] = kv_scores(o, o.reshape(qh[h], {R, D}, x + "q2" + hs), kc, x + "sc" + hs);
        }
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            prs[h] = o.temp(x + "pr" + hs, DType::BF16, {NC, R, C});
            sums[h] = o.temp(x + "psum" + hs, DType::F32, {R, 1});
            const TensorId ins[] = {scs[h], last};
            const TensorId kouts[] = {prs[h], sums[h]};
            g.op("flashinfer_gaudi_masked_softmax_bf16_gaudi2", ins, kouts, &scale, sizeof(float));
        }
        for (uint64_t h = 0; h < KV; ++h) {
            const auto hs = std::to_string(h);
            const auto acc = o.reduce("sum", kv_pv(o, prs[h], o.reshape(vcs[h], {NC, C, D}, x + "vc3" + hs), x + "pvc" + hs), 0,
                                      x + "pvs" + hs);
            const auto den = kv8_ ? o.mul(sums[h], o.constant(128.f, {1}, x + "p128" + hs), x + "psum8" + hs) : sums[h];
            const auto oh = o.div(o.reshape(acc, {R, D}, x + "pv2" + hs), den, x + "pvn" + hs);
            outs.push_back(o.reshape(o.cast(oh, DType::BF16, x + "pvb" + hs), {1, R, D}, x + "sd3" + hs));
        }
        atts.push_back(o.reshape(o.transpose(o.reshape(o.concat(outs, 0, x + "sdall"), {NH, T, D}, x + "a3"), {1, 0, 2}, x + "at"),
                                 {T, NH * D}, x + "a2"));
    }
    output(g, "att_out", o.concat(atts, 0, "atc"));
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

uint32_t Qwen35::core_rows(uint32_t rows) {
    if (rows <= 1024) return 1024;
    return (rows + 2047) / 2048 * 2048;
}

Recipe& Qwen35::gdn_batch_recipe(uint32_t B, uint32_t T) {
    std::string key = "qwen35_gdnb2_B" + std::to_string(B) + (T != kSpecMaxT ? "_T" + std::to_string(T) : "");
    if (fused_gate_quant()) key += "_mq";
    if (gated_kernel()) key += "_gk2";
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, R = uint64_t(T) * B;
    GraphBuilder g(dev_);
    Ops o(g);
    idx_t_ = acc_t_ = mask_t_ = kNoTensor;
    build_inputs_.clear();
    building_debug_j_ = -1;
    const bool spec = building_spec_;
    building_spec_ = true;
    Pair cur{g.input("res_in", DType::BF16, {R, H}), g.input("delta_in", DType::BF16, {R, H})};
    cur = gdn_layer(o, 0, T, cur, B);
    building_spec_ = spec;
    output(g, "res_out", cur.res);
    output(g, "delta_out", cur.delta);
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Recipe& Qwen35::apre_batch_recipe(uint32_t B, uint32_t T) {
    const std::string key = "qwen35_apreb_B" + std::to_string(B) + (T != kSpecMaxT ? "_T" + std::to_string(T) : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, NH = cfg_.heads, KV = cfg_.kv_heads, D = cfg_.head_dim, R2 = cfg_.rotary_dim / 2;
    const uint64_t R = uint64_t(T) * B;
    const uint32_t j = 0;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto res = g.input("res_in", DType::BF16, {R, H}), delta = g.input("delta_in", DType::BF16, {R, H});
    const auto in_norm = g.input(P(j, "in_norm1p"), DType::BF16, {H});
    const auto qkv_w = g.input(P(j, "qkv_w"), DType::F8, {NH * D * 2 + 2 * KV * D, H});
    const auto qkv_s = g.input(P(j, "qkv_s"), DType::F32, {NH * D * 2 + 2 * KV * D});
    const auto q_norm = g.input(P(j, "q_norm"), DType::BF16, {D});
    const auto k_norm = g.input(P(j, "k_norm"), DType::BF16, {D});
    // every row's RoPE row gathered from the tables by its position (binding each slot's rows as its own input
    // would overlap when two slots stand at the same position: persistent tensors must not)
    const uint64_t tab = uint64_t(max_ctx_) + kMaxT;
    const auto rpos = g.input("rope_pos", DType::I32, {R});
    const auto cosv = o.reshape(o.gather_rows(g.input("cos_tab", DType::F32, {tab, R2}), rpos, "cosg"), {R, 1, R2}, "cos3");
    const auto sinv = o.reshape(o.gather_rows(g.input("sin_tab", DType::F32, {tab, R2}), rpos, "sing"), {R, 1, R2}, "sin3");
    // as attn_pre_recipe
    const auto nq = o.add_rmsnorm_quant(delta, res, in_norm, cfg_.eps, P(j, "in"));
    const auto qkv = o.fp8_gemm(nq.codes, nq.scale, qkv_w, qkv_s, P(j, "qkv"));
    const auto p = o.split(qkv, 1, {NH * D * 2, KV * D, KV * D}, P(j, "qkv_split"));
    const auto qg = o.split(o.reshape(p[0], {R, NH, 2 * D}, P(j, "qg3")), 2, {D, D}, P(j, "qg_split"));
    const auto q = o.rms_norm(qg[0], q_norm, cfg_.eps, true, P(j, "qn"));
    const auto k = o.rms_norm(o.reshape(p[1], {R, KV, D}, P(j, "k3")), k_norm, cfg_.eps, true, P(j, "kn"));
    const auto v = o.reshape(p[2], {R, KV, D}, P(j, "v3"));
    auto rope = [&](TensorId t, const std::string& n) {
        const auto f = o.cast(t, DType::F32, n + "_f");
        const auto s = o.split(f, 2, {R2, R2, D - 2 * R2}, n + "_rs");
        const auto o1 = o.sub(o.mul(s[0], cosv, n + "_1c"), o.mul(s[1], sinv, n + "_2s"), n + "_o1");
        const auto o2 = o.add(o.mul(s[1], cosv, n + "_2c"), o.mul(s[0], sinv, n + "_1s"), n + "_o2");
        return o.cast(o.concat({o1, o2, s[2]}, 2, n + "_cat"), DType::BF16, n + "_b");
    };
    output(g, "q_out", rope(q, P(j, "qr")));
    output(g, "gate_out", o.reshape(qg[1], {R, NH * D}, P(j, "gate")));
    output(g, "resid_out", nq.residual_out);
    const auto kh = o.split(o.transpose(rope(k, P(j, "kr")), {1, 0, 2}, "kt"), 0, std::vector<uint64_t>(KV, 1), "kh");
    const auto vh = o.split(o.transpose(v, {1, 0, 2}, "vt"), 0, std::vector<uint64_t>(KV, 1), "vh");
    for (uint64_t h = 0; h < KV; ++h) {  // rows pos_b .. pos_b + T - 1 of each slot's cache, per head
        const auto hs = std::to_string(h);
        const auto kb = o.split(o.reshape(kh[h], {R, D}, "k2_" + hs), 0, std::vector<uint64_t>(B, T), "kb" + hs);
        const auto vb = o.split(o.reshape(vh[h], {R, D}, "v2_" + hs), 0, std::vector<uint64_t>(B, T), "vb" + hs);
        for (uint32_t b = 0; b < B; ++b) {
            const auto x = std::to_string(b) + "_" + hs;
            output(g, "k_out" + x, kv8_ ? kv_to_f8(o, kb[b], 1.f, "k8_" + x) : kb[b]);
            output(g, "v_out" + x, kv8_ ? kv_to_f8(o, vb[b], 1.f, "v8_" + x) : vb[b]);
        }
    }
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Recipe& Qwen35::acore_recipe(uint32_t T, uint32_t Sb) {
    // attn_post_recipe's verification form (BF16 scores, the masked-softmax kernel, chunked PV) without the gate,
    // o_proj and MLP (batched in apost_batch_recipe); prefill-sized pieces (T > 16) take FP16 scores and the FP16
    // softmax kernel, as attn_post_recipe's prefill form
    const bool f16 = T > kSpecMaxT && !kv8_;
    const std::string key = "qwen35_acore_T" + std::to_string(T) + "_S" + std::to_string(Sb) + (f16 ? "_f16" : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    if (Sb % 1024) throw std::logic_error("qwen35: acore rows must be a multiple of 1024");
    const uint64_t NH = cfg_.heads, KV = cfg_.kv_heads, D = cfg_.head_dim, G = NH / KV;
    const uint32_t j = 0;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto q = g.input("q_in", DType::BF16, {T, NH, D});
    const auto last = g.input("last", DType::I32, {T});
    const float scale = float(1.0 / std::sqrt(double(D)));
    const auto qh = o.split(o.reshape(o.transpose(q, {1, 0, 2}, "qt"), {KV, G * T, D}, "q3"), 0, std::vector<uint64_t>(KV, 1), "qh");
    const uint64_t C = Sb % 2048 == 0 ? 2048 : 1024, NC = Sb / C, R = G * T;
    std::vector<TensorId> qss(KV), scs(KV), prs(KV), sums(KV), vcs(KV), outs;
    for (uint64_t h = 0; h < KV; ++h) qss[h] = o.reshape(qh[h], {R, D}, "q2" + std::to_string(h));
    for (uint64_t h = 0; h < KV; ++h) {
        const auto hs = std::to_string(h);
        const auto kc = g.input(P(j, "k_cache" + hs), kv_dt(), {Sb, D});
        vcs[h] = g.input(P(j, "v_cache" + hs), kv_dt(), {Sb, D});
        scs[h] = f16 ? o.linear_bf16(qss[h], kc, "sc" + hs, DType::F16) : kv_scores(o, qss[h], kc, "sc" + hs);  // [R, Sb]
    }
    for (uint64_t h = 0; h < KV; ++h) {
        const auto hs = std::to_string(h);
        prs[h] = o.temp("pr" + hs, DType::BF16, {NC, R, C});
        sums[h] = o.temp("psum" + hs, DType::F32, {R, 1});
        const TensorId ins[] = {scs[h], last};
        const TensorId kouts[] = {prs[h], sums[h]};
        g.op(f16 ? "flashinfer_gaudi_masked_softmax_f16_gaudi2" : "flashinfer_gaudi_masked_softmax_bf16_gaudi2", ins, kouts, &scale,
             sizeof(float));
    }
    for (uint64_t h = 0; h < KV; ++h) {
        const auto hs = std::to_string(h);
        const auto acc = o.reduce("sum", kv_pv(o, prs[h], o.reshape(vcs[h], {NC, C, D}, "vc3" + hs), "pvc" + hs), 0,
                                  "pvs" + hs);  // [1, R, D]
        const auto den = kv8_ ? o.mul(sums[h], o.constant(128.f, {1}, "p128" + hs), "psum8" + hs) : sums[h];
        const auto oh = o.div(o.reshape(acc, {R, D}, "pv2" + hs), den, "pvn" + hs);
        outs.push_back(o.reshape(o.cast(oh, DType::BF16, "pvb" + hs), {1, R, D}, "sd3" + hs));
    }
    output(g, "att_out", o.reshape(o.transpose(o.reshape(o.concat(outs, 0, "sdall"), {NH, T, D}, "a3"), {1, 0, 2}, "at"),
                                   {T, NH * D}, "a2"));
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Recipe& Qwen35::apost_batch_recipe(uint32_t B, uint32_t T) {
    const std::string key = "qwen35_apostb_B" + std::to_string(B) + (T != kSpecMaxT ? "_T" + std::to_string(T) : "") +
                            (fused_gate_quant() ? "_mq" : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, NH = cfg_.heads, D = cfg_.head_dim, R = uint64_t(T) * B;
    const uint32_t j = 0;
    GraphBuilder g(dev_);
    Ops o(g);
    build_inputs_.clear();
    const auto gate = g.input("gate_in", DType::BF16, {R, NH * D});
    const auto att2 = g.input("att_in", DType::BF16, {R, NH * D});
    const auto resid = g.input("resid_in", DType::BF16, {R, H});
    const auto o_w = g.input(P(j, "o_w"), DType::F8, {H, NH * D});
    const auto o_s = g.input(P(j, "o_s"), DType::F32, {H});
    std::pair<TensorId, TensorId> q8;  // as attn_post_recipe
    if (fused_gate_quant()) {
        q8 = o.mul_quant(o.unary("sigmoid", gate, "gsig"), att2, "gq");
    } else {
        q8 = o.quantize(o.mul(att2, o.unary("sigmoid", gate, "gsig"), "gated"), "gq");
    }
    const auto y = o.fp8_gemm(q8.first, q8.second, o_w, o_s, P(j, "oproj"));
    const auto out = mlp(o, j, y, resid);
    output(g, "res_out", out.res);
    output(g, "delta_out", out.delta);
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

// Commit of several slots' batched verifications at once: as spec_commit_recipe per slot (its own mask, conv rows
// and stash), the recurrences of every slot's GDN layers as extra heads of one chunked delta rule.
Recipe& Qwen35::spec_commit_batch_recipe(uint32_t B, uint32_t T, bool raw_qk) {
    const std::string key = "qwen35_speccommitb2_B" + std::to_string(B) + "_T" + std::to_string(T) + (raw_qk ? "_x" : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t HK = cfg_.lin_k_heads, HV = cfg_.lin_v_heads, DK = cfg_.lin_k_dim, DV = cfg_.lin_v_dim,
                   C = cfg_.conv_channels(), KC = cfg_.conv_kernel, NG = num_gdn_;
    GraphBuilder g(dev_);
    Ops o(g);
    build_inputs_.clear();
    std::vector<TensorId> qs, ks, vs, gs, bs, ss;
    for (uint32_t b = 0; b < B; ++b) {
        const std::string x = std::to_string(b);
        const auto mask = g.input("mask" + x, DType::F32, {T, 1});
        const auto rows = g.input("rows" + x, DType::I32, {KC - 1});
        ss.push_back(g.input("ssm_in" + x, DType::F32, {NG * HV, DV, DK}));
        auto rows_first = [&](TensorId t, Shape per_layer, uint64_t heads, const std::string& nm) {
            std::vector<unsigned> perm(per_layer.size());
            for (unsigned i = 0; i < perm.size(); ++i) perm[i] = i;
            std::swap(perm[0], perm[1]);
            Shape out = {T, NG * heads};
            for (size_t i = 3; i < per_layer.size(); ++i) out.push_back(per_layer[i]);
            return o.reshape(o.transpose(t, perm, nm + "_t" + x), out, nm + "_r" + x);
        };
        TensorId q, k;
        if (raw_qk) {
            const auto xs = o.split(g.input("x" + x, DType::F32, {NG, HK, 2 * T, DK}), 2, {T, T}, "xs" + x);
            k = o.reshape(o.transpose(xs[0], {2, 0, 1, 3}, "xk_t" + x), {T, NG * HK, DK}, "xk" + x);
            q = o.reshape(o.transpose(xs[1], {2, 0, 1, 3}, "xq_t" + x), {T, NG * HK, DK}, "xq" + x);
            auto l2 = [&](TensorId t3, const std::string& n) {
                const auto sq = o.add(o.reduce("sum", o.mul(t3, t3, n + "sq"), 2, n + "ss"), o.constant(1e-6f, {1}, n + "eps"), n + "sse");
                return o.mul(t3, o.unary("rsqrt", sq, n + "rs"), n + "n");
            };
            q = o.mul(l2(q, "ql2" + x), o.constant(float(1.0 / std::sqrt(double(DK))), {1}, "qscale" + x), "qs" + x);
            k = l2(k, "kl2" + x);
        } else {
            q = rows_first(g.input("q" + x, DType::F32, {NG, T, HK, DK}), {NG, T, HK, DK}, HK, "q");
            k = rows_first(g.input("k" + x, DType::F32, {NG, T, HK, DK}), {NG, T, HK, DK}, HK, "k");
        }
        qs.push_back(q);
        ks.push_back(k);
        vs.push_back(rows_first(g.input("v" + x, DType::F32, {NG, T, HV, DV}), {NG, T, HV, DV}, HV, "v"));
        gs.push_back(o.mul(rows_first(g.input("g" + x, DType::F32, {NG, T, HV}), {NG, T, HV}, HV, "g"), mask, "gm" + x));
        bs.push_back(o.mul(rows_first(g.input("b" + x, DType::F32, {NG, T, HV}), {NG, T, HV}, HV, "b"), mask, "bm" + x));
        const auto seq = g.input("seq" + x, DType::BF16, {NG, T + KC - 1, C});
        output(g, "conv_out" + x, o.gather(seq, rows, 1, "win" + x));
    }
    // one chunked delta rule per slot, straight on its bound state (merging the slots' states into extra heads
    // copied every state in and out: slower than one commit launch per slot)
    for (uint32_t b = 0; b < B; ++b) {
        const std::string x = std::to_string(b);
        const auto [core, s_next] = gdn_chunk(o, "c" + x + "_", T, NG * HK, NG * HV, ss[b], qs[b], ks[b], vs[b], gs[b], bs[b]);
        (void)core;
        output(g, "ssm_out" + x, s_next);
    }
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

void Qwen35::accept_batch(std::span<const uint32_t> slots, std::span<const uint32_t> accepted) {
    const uint32_t B = uint32_t(slots.size());
    if (!B) return;
    if (accepted.size() != B) throw std::invalid_argument("qwen35: accept_batch sizes");
    // park the selected slot: every slot's state is in seqs_
    const uint32_t keep = active_slot_;
    swap_seq(seqs_[keep]);
    struct Unpark {
        Qwen35& m;
        uint32_t s;
        ~Unpark() { m.swap_seq(m.seqs_[s]); }
    } unpark{*this, keep};
    std::vector<Seq*> sq;
    uint32_t T = 0, top = 0;
    bool wy = false;
    for (uint32_t b = 0; b < B; ++b) {
        Seq& q = seqs_[slots[b]];
        if (!q.pending_T || accepted[b] >= q.pending_T) throw std::logic_error("qwen35: accept_batch without a pending verification");
        if (b && (q.pending_T != T || q.pending_wy != wy)) throw std::logic_error("qwen35: accept_batch of different verifications");
        T = q.pending_T;
        wy = q.pending_wy;
        top = std::max(top, q.stash_b + 1);
        sq.push_back(&q);
    }
    // controls: per slot (its batch index's region) mask [T,1] f32 at 0, conv rows [3] i32 at 2048; one upload
    for (uint32_t b = 0; b < B; ++b) {
        const uint64_t ctl = uint64_t(sq[b]->stash_b) * 4096;
        auto* m = reinterpret_cast<float*>(staging_.as<uint8_t>() + kAcceptStage + ctl);
        for (uint32_t t = 0; t < T; ++t) m[t] = t <= accepted[b] ? 1.f : 0.f;
        auto* rows = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kAcceptStage + ctl + 2048);
        for (uint32_t k = 0; k + 1 < cfg_.conv_kernel; ++k) rows[k] = int32_t(accepted[b] + 1 + k);
    }
    stream_.upload(staging_, kAcceptStage, spec_ctl_, 0, uint64_t(top) * 4096);
    std::vector<Binding> bnd;
    const std::string cp = "chunk" + std::to_string(T) + ".";
    std::vector<std::string> names = {"upper", "strict", "inclusive", "lower", "eye", "diag8"};
    for (uint32_t sz = 8; sz < T; sz *= 2) names.push_back("ll" + std::to_string(sz));
    for (const auto& nm : names) bnd.push_back({cp + nm, &aux_, aux_off_.at(cp + nm)});
    const auto L = stash_layout(T, wy);
    const uint64_t NG = std::max(num_gdn_, 1u);
    const char* fields[] = {wy ? "x" : "q", "k", "v", "g", "b", "seq"};
    for (uint32_t b = 0; b < B; ++b) {
        const Seq& q = *sq[b];
        const std::string x = std::to_string(b);
        const uint64_t ctl = uint64_t(q.stash_b) * 4096;
        bnd.push_back({"mask" + x, &spec_ctl_, ctl});
        bnd.push_back({"rows" + x, &spec_ctl_, ctl + 2048});
        for (int f = 0; f < 6; ++f)
            if (L.size[f]) bnd.push_back({fields[f] + x, &spec_stash_, L.base[f] + uint64_t(q.stash_b) * NG * L.size[f]});
        bnd.push_back({"ssm_in" + x, &q.ssm[q.ssm_side[0]], 0});
        bnd.push_back({"ssm_out" + x, &q.ssm[q.ssm_side[0] ^ 1], 0});
        bnd.push_back({"conv_out" + x, &q.conv[q.conv_side[0] ^ 1], 0});
    }
    spec_commit_batch_recipe(B, T, wy).enqueue(stream_, bnd);
    for (uint32_t b = 0; b < B; ++b) {
        Seq& q = *sq[b];
        for (auto& side : q.ssm_side) side ^= 1;
        for (auto& side : q.conv_side) side ^= 1;
        q.pending_T = 0;
        q.pos += accepted[b] + 1;
    }
    pending_total_ -= std::min(pending_total_, B);
}

Recipe& Qwen35::gdn_prefill_batch_recipe(uint32_t B, uint32_t T) {
    std::string key = "qwen35_gdnpb_B" + std::to_string(B) + "_T" + std::to_string(T) + "_c" + std::to_string(gdn_chunk());
    if (fused_gate_quant()) key += "_mq";
    if (gated_kernel()) key += "_gk2";
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, R = uint64_t(T) * B;
    GraphBuilder g(dev_);
    Ops o(g);
    idx_t_ = acc_t_ = mask_t_ = kNoTensor;
    build_inputs_.clear();
    building_debug_j_ = -1;
    const bool spec = building_spec_;
    building_spec_ = false;
    Pair cur{g.input("res_in", DType::BF16, {R, H}), g.input("delta_in", DType::BF16, {R, H})};
    cur = gdn_layer(o, 0, T, cur, B);
    building_spec_ = spec;
    output(g, "res_out", cur.res);
    output(g, "delta_out", cur.delta);
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

bool Qwen35::prefill_batchable(uint32_t slot, uint32_t T) const {
    if (block_ != 1 || !wy_geometry() || slot >= seqs_.size()) return false;
    const uint32_t pos = slot == active_slot_ ? pos_ : seqs_[slot].pos;
    const uint32_t cap = slot == active_slot_ ? cap_ : seqs_[slot].cap;
    const int64_t rd = slot == active_slot_ ? rope_delta_ : seqs_[slot].rope_delta;
    return rd == 0 && pos + T <= cap && core_rows(pos + T) <= kPrefillBatchMaxRows;
}

std::vector<std::vector<float>> Qwen35::prefill_batch(std::span<const uint32_t> slots, std::span<const std::vector<int32_t>> pieces,
                                                      std::span<const uint8_t> want_logits) {
    const uint32_t B = uint32_t(slots.size()), T = pieces.empty() ? 0 : uint32_t(pieces[0].size()), R = B * T;
    if (B < 2 || B > kMaxBatch || pieces.size() != B || want_logits.size() != B)
        throw std::invalid_argument("qwen35: prefill_batch takes 2..8 slots");
    if (T < 64 || T % 64 || R > kMaxT) throw std::invalid_argument("qwen35: prefill_batch pieces of 64k rows, 2048 in all");
    if (pending_T_ || pending_total_) throw std::logic_error("qwen35: accept() the pending verifications first");
    for (uint32_t b = 0; b < B; ++b) {
        if (pieces[b].size() != T) throw std::invalid_argument("qwen35: prefill_batch pieces of one length");
        if (!prefill_batchable(slots[b], T)) throw std::invalid_argument("qwen35: prefill_batch slot not batchable");
        for (uint32_t c = 0; c < b; ++c)
            if (slots[c] == slots[b]) throw std::invalid_argument("qwen35: prefill_batch slots must differ");
    }
    const uint32_t keep = active_slot_;
    swap_seq(seqs_[keep]);
    struct Unpark {
        Qwen35& m;
        uint32_t s;
        ~Unpark() { m.swap_seq(m.seqs_[s]); }
    } unpark{*this, keep};
    std::vector<Seq*> sq;
    for (uint32_t b = 0; b < B; ++b) sq.push_back(&seqs_[slots[b]]);
    const uint64_t D = cfg_.head_dim, KV = cfg_.kv_heads, NHD = cfg_.heads * D, Hd = cfg_.hidden;
    Event e0(dev_), e1(dev_);
    e0.record(stream_);
    if (stage_ev_ && stage_ev_->recorded()) stage_ev_->synchronize();
    {
        auto* ids = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kIdsStage);
        auto* last = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kIdsStage + kIdsLast);
        auto* rpos = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kIdsStage + kIdsRopePos);
        for (uint32_t b = 0; b < B; ++b)
            for (uint32_t t = 0; t < T; ++t) {
                ids[b * T + t] = pieces[b][t];
                last[b * T + t] = int32_t(sq[b]->pos + t);  // T * 4 bytes per slot: 128-byte aligned
                rpos[b * T + t] = int32_t(sq[b]->pos + t);
            }
    }
    stream_.upload(staging_, kIdsStage, ids_, 0, kIdsRopePos + uint64_t(R) * 4);
    {
        const Binding eb[] = {{"ids", &ids_, 0}, w_.bind("embed", "embed"), {"res", &res_[0], 0}};
        embed_recipe(R).enqueue(stream_, eb);
    }
    if (!stage_ev_) stage_ev_ = std::make_unique<Event>(dev_);
    stage_ev_->record(stream_);
    auto bind_w = [&](std::vector<Binding>& v, const std::string& name, const std::string& weight) {
        v.push_back(w_.bind(P(0, name), weight));
    };
    auto bind_aux = [&](std::vector<Binding>& v, const std::string& name, const std::string& key) {
        v.push_back({P(0, name), &aux_, aux_off_.at(key)});
    };
    int cur = 0;
    for (uint32_t i = 0; i < cfg_.layers; ++i) {
        const int s = slot_[i];
        if (cfg_.attention[i]) {
            std::vector<Binding> pb = {{"res_in", &res_[cur], 0}, {"delta_in", i == 0 ? &zero_ : &delta_[cur], 0},
                                       {"q_out", &apre_q_, 0}, {"gate_out", &apre_g_, 0}, {"resid_out", &apre_r_, 0},
                                       {"rope_pos", &ids_, kIdsRopePos}, {"cos_tab", &cos_tab_, 0}, {"sin_tab", &sin_tab_, 0}};
            bind_aux(pb, "in_norm1p", L(i, "in_norm1p"));
            bind_w(pb, "qkv_w", L(i, "qkv.w"));
            bind_w(pb, "qkv_s", L(i, "qkv.s"));
            bind_w(pb, "q_norm", L(i, "q_norm"));
            bind_w(pb, "k_norm", L(i, "k_norm"));
            for (uint32_t b = 0; b < B; ++b) {
                const Seq& q = *sq[b];
                const auto bs = std::to_string(b);
                for (uint64_t h = 0; h < KV; ++h) {
                    const uint64_t at = (h * uint64_t(q.cap) + q.pos) * D * kv_es();
                    pb.push_back({"k_out" + bs + "_" + std::to_string(h), &q.k[s], at});
                    pb.push_back({"v_out" + bs + "_" + std::to_string(h), &q.v[s], at});
                }
            }
            apre_batch_recipe(B, T).enqueue(stream_, pb);
            for (uint32_t b = 0; b < B; ++b) {  // each slot's attention over its cache (the new rows written above)
                const Seq& q = *sq[b];
                std::vector<Binding> cb = {{"q_in", &apre_q_, uint64_t(b) * T * NHD * 2},
                                           {"last", &ids_, kIdsLast + uint64_t(b) * T * 4},
                                           {"att_out", &att_, uint64_t(b) * T * NHD * 2}};
                for (uint64_t h = 0; h < KV; ++h) {
                    const uint64_t head = h * uint64_t(q.cap) * D * kv_es();
                    cb.push_back({P(0, "k_cache" + std::to_string(h)), &q.k[s], head});
                    cb.push_back({P(0, "v_cache" + std::to_string(h)), &q.v[s], head});
                }
                acore_recipe(T, core_rows(q.pos + T)).enqueue(stream_, cb);
            }
            std::vector<Binding> qb = {{"gate_in", &apre_g_, 0}, {"att_in", &att_, 0}, {"resid_in", &apre_r_, 0},
                                       {"res_out", &res_[cur ^ 1], 0}, {"delta_out", &delta_[cur ^ 1], 0}};
            bind_aux(qb, "post_norm1p", L(i, "post_norm1p"));
            bind_w(qb, "gu_w", L(i, "gate_up.w"));
            bind_w(qb, "gu_s", L(i, "gate_up.s"));
            bind_w(qb, "down_w", L(i, "down.w"));
            bind_w(qb, "down_s", L(i, "down.s"));
            bind_w(qb, "o_w", L(i, "o.w"));
            bind_w(qb, "o_s", L(i, "o.s"));
            apost_batch_recipe(B, T).enqueue(stream_, qb);
        } else {
            std::vector<Binding> gb = {{"res_in", &res_[cur], 0}, {"delta_in", i == 0 ? &zero_ : &delta_[cur], 0},
                                       {"res_out", &res_[cur ^ 1], 0}, {"delta_out", &delta_[cur ^ 1], 0}};
            bind_aux(gb, "in_norm1p", L(i, "in_norm1p"));
            bind_aux(gb, "post_norm1p", L(i, "post_norm1p"));
            bind_w(gb, "gu_w", L(i, "gate_up.w"));
            bind_w(gb, "gu_s", L(i, "gate_up.s"));
            bind_w(gb, "down_w", L(i, "down.w"));
            bind_w(gb, "down_s", L(i, "down.s"));
            bind_w(gb, "qkvz_w", L(i, "qkvz.w"));
            bind_w(gb, "qkvz_s", L(i, "qkvz.s"));
            bind_w(gb, "ba_w", L(i, "ba"));
            bind_aux(gb, "conv_t", L(i, "conv_t"));
            bind_aux(gb, "negexpA", L(i, "negexpA"));
            bind_w(gb, "dt_bias", L(i, "dt_bias"));
            bind_w(gb, "gdn_norm", L(i, "gdn_norm"));
            bind_w(gb, "out_w", L(i, "out.w"));
            bind_w(gb, "out_s", L(i, "out.s"));
            for (uint32_t b = 0; b < B; ++b) {
                const Seq& q = *sq[b];
                const auto bs = std::to_string(b);
                gb.push_back({P(0, "conv_in" + bs), &q.conv[q.conv_side[s]], uint64_t(s) * conv_bytes_});
                gb.push_back({P(0, "conv_out" + bs), &q.conv[q.conv_side[s] ^ 1], uint64_t(s) * conv_bytes_});
                gb.push_back({P(0, "ssm_in" + bs), &q.ssm[q.ssm_side[s]], uint64_t(s) * ssm_bytes_});
                gb.push_back({P(0, "ssm_out" + bs), &q.ssm[q.ssm_side[s] ^ 1], uint64_t(s) * ssm_bytes_});
            }
            {
                const std::string cp = "chunk" + std::to_string(gdn_chunk()) + ".";
                std::vector<std::string> names = {"upper", "strict", "inclusive", "lower", "eye", "diag8"};
                for (uint32_t sz = 8; sz < gdn_chunk(); sz *= 2) names.push_back("ll" + std::to_string(sz));
                for (const auto& nm : names) gb.push_back({cp + nm, &aux_, aux_off_.at(cp + nm)});
            }
            gdn_prefill_batch_recipe(B, T).enqueue(stream_, gb);
            for (uint32_t b = 0; b < B; ++b) {  // the slots' new states and windows are on the other sides now
                sq[b]->conv_side[s] ^= 1;
                sq[b]->ssm_side[s] ^= 1;
            }
        }
        for (size_t k = 0; k < capture_.size(); ++k)
            if (capture_[k] == i) {
                const Binding cb[] = {{"res", &res_[cur ^ 1], 0}, {"delta", &delta_[cur ^ 1], 0},
                                      {"feat", &feats_, k * feature_stride()}};
                capture_recipe(R).enqueue(stream_, cb);
            }
        cur ^= 1;
    }
    cur_ = cur;
    // the logits of the last row of every slot that wants them (one head launch each)
    std::vector<uint32_t> want;
    for (uint32_t b = 0; b < B; ++b)
        if (want_logits[b]) want.push_back(b);
    if (logits_.size() < uint64_t(cfg_.vocab) * 4 * std::max<size_t>(want.size(), 1))
        logits_ = DeviceBuffer(dev_, uint64_t(cfg_.vocab) * 4 * kMaxBatch);
    for (size_t w = 0; w < want.size(); ++w) {
        const uint64_t row = uint64_t(want[w]) * T + T - 1;
        std::vector<Binding> hb = {{"res", &res_[cur], row * Hd * 2},
                                   {"delta", &delta_[cur], row * Hd * 2},
                                   {"final_norm1p", &aux_, aux_off_.at("final_norm1p")},
                                   w_.bind("lm_head", "lm_head"),
                                   {"logits", &logits_, w * uint64_t(cfg_.vocab) * 4}};
        head_recipe().enqueue(stream_, hb);
    }
    e1.record(stream_);
    if (!want.empty()) stream_.download(logits_, 0, staging_, 0, uint64_t(cfg_.vocab) * 4 * want.size());
    stream_.synchronize();
    last_ms_ = Event::elapsed_ms(e0, e1);
    for (uint32_t b = 0; b < B; ++b) sq[b]->pos += T;
    std::vector<std::vector<float>> out(B);
    for (size_t w = 0; w < want.size(); ++w) {
        const float* l = staging_.as<float>() + w * cfg_.vocab;
        out[want[w]].assign(l, l + cfg_.vocab);
    }
    return out;
}

std::vector<std::vector<int32_t>> Qwen35::verify_batch(std::span<const uint32_t> slots,
                                                       std::span<const std::vector<int32_t>> blocks,
                                                       const std::function<void()>& launched) {
    return batch_verify(slots, blocks, {}, launched).targets;
}

std::vector<Qwen35::VerifyTop> Qwen35::verify_batch_topk(std::span<const uint32_t> slots,
                                                         std::span<const std::vector<int32_t>> blocks,
                                                         std::span<const float> inv_temps,
                                                         const std::function<void()>& launched) {
    if (inv_temps.size() != slots.size()) throw std::invalid_argument("qwen35: verify_batch_topk: one 1 / temperature per slot");
    return batch_verify(slots, blocks, inv_temps, launched).tops;
}

Recipe& Qwen35::verify_topkb_recipe(uint32_t R) {
    const std::string key = "qwen35_vtopb_R" + std::to_string(R);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, V = cfg_.vocab;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto res = g.input("res", DType::BF16, {R, H});
    const auto delta = g.input("delta", DType::BF16, {R, H});
    const auto nw = g.input("final_norm1p", DType::BF16, {H});
    const auto w = g.input("lm_head", DType::BF16, {V, H});
    const auto inv_temp = g.input("inv_temp", DType::F32, {R, 1});
    const auto nq = o.add_rmsnorm_quant(delta, res, nw, cfg_.eps, "final");
    const auto scaled = o.mul(o.linear_bf16(nq.normalized, w, "lg", DType::F32), inv_temp, "scaled");  // [R, V]
    const auto top = o.topk2(scaled, kTopK, kTopkChunk, g.input("tk_consts", DType::I32, {V + 2 * uint64_t(R)}), "tk");
    output(g, "top_v", top.first);
    output(g, "top_i", top.second);
    const auto mx = o.reduce("max", scaled, 1, "mx_v");
    output(g, "mx", mx);
    output(g, "sum", o.reduce("sum", o.unary("exp", o.sub(scaled, mx, "shift"), "ex"), 1, "sum_v"));
    auto r = g.compile(key);
    return *(recipes_[key] = std::move(r));
}

Qwen35::BatchOut Qwen35::batch_verify(std::span<const uint32_t> slots, std::span<const std::vector<int32_t>> blocks,
                                      std::span<const float> inv_temps, const std::function<void()>& launched) {
    const uint32_t B = uint32_t(slots.size()), T = blocks.empty() ? 0 : uint32_t(blocks[0].size()), R = B * T;
    if (B < 2 || B > kMaxBatch || blocks.size() != B) throw std::invalid_argument("qwen35: verify_batch takes 2..8 slots");
    if (T != 8 && T != kSpecMaxT) throw std::invalid_argument("qwen35: verify_batch blocks of 8 or 16 tokens");
    if (pending_T_ || pending_total_) throw std::logic_error("qwen35: accept() the pending verifications first");
    if (block_ != 1 || !wy_geometry() || !lm8_on_ || layer_limit_ || debug_layer_ >= 0)
        throw std::logic_error("qwen35: verify_batch needs one-layer blocks, the WY verification and the FP8 head");
    for (uint32_t b = 0; b < B; ++b) {
        if (slots[b] >= seqs_.size() || blocks[b].size() != T) throw std::invalid_argument("qwen35: verify_batch slot / block");
        for (uint32_t c = 0; c < b; ++c)
            if (slots[c] == slots[b]) throw std::invalid_argument("qwen35: verify_batch slots must differ");
    }
    // park the selected slot: every slot's state is in seqs_ while this runs
    const uint32_t keep = active_slot_;
    swap_seq(seqs_[keep]);
    struct Unpark {
        Qwen35& m;
        uint32_t s;
        ~Unpark() { m.swap_seq(m.seqs_[s]); }
    } unpark{*this, keep};
    std::vector<Seq*> sq;
    for (uint32_t b = 0; b < B; ++b) {
        Seq& q = seqs_[slots[b]];
        if (q.k.empty() || !q.k[0].size()) throw std::logic_error("qwen35: verify_batch slot without KV caches");
        if (q.pos + T > q.cap) throw std::out_of_range("qwen35: context full");
        if (q.rope_delta != 0) throw std::logic_error("qwen35: verify_batch with multimodal RoPE offsets");
        sq.push_back(&q);
    }
    const uint64_t D = cfg_.head_dim, KV = cfg_.kv_heads, NHD = cfg_.heads * D;
    Event e0(dev_), e1(dev_);
    e0.record(stream_);
    // G2E_VERIFY_PROF=1: every 50th batch, the device time by part (events between the launches)
    static const bool vprof = std::getenv("G2E_VERIFY_PROF") != nullptr;
    static int vcalls = 0;
    const bool prof = vprof && ++vcalls % 50 == 0;
    static std::vector<std::unique_ptr<Event>> pev;
    std::vector<int> pkind;
    auto mark = [&](int kind) {
        if (!prof) return;
        if (pev.size() <= pkind.size()) pev.push_back(std::make_unique<Event>(dev_));
        pev[pkind.size()]->record(stream_);
        pkind.push_back(kind);
    };
    if (stage_ev_ && stage_ev_->recorded()) stage_ev_->synchronize();
    {
        auto* ids = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kIdsStage);
        auto* last = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kIdsStage + kIdsLast);
        auto* rpos = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kIdsStage + kIdsRopePos);
        for (uint32_t b = 0; b < B; ++b)
            for (uint32_t t = 0; t < T; ++t) {
                ids[b * T + t] = blocks[b][t];
                last[b * 32 + t] = int32_t(sq[b]->pos + t);  // 128-byte aligned per slot (a binding's alignment)
                rpos[b * T + t] = int32_t(sq[b]->pos + t);   // RoPE rows (text: rope_delta 0)
            }
    }
    stream_.upload(staging_, kIdsStage, ids_, 0, kIdsRopePos + uint64_t(R) * 4);
    {
        const Binding eb[] = {{"ids", &ids_, 0}, w_.bind("embed", "embed"), {"res", &res_[0], 0}};
        embed_recipe(R).enqueue(stream_, eb);
    }
    mark(0);
    if (!stage_ev_) stage_ev_ = std::make_unique<Event>(dev_);
    stage_ev_->record(stream_);
    auto bind_w = [&](std::vector<Binding>& v, const std::string& name, const std::string& weight) {
        v.push_back(w_.bind(P(0, name), weight));
    };
    auto bind_aux = [&](std::vector<Binding>& v, const std::string& name, const std::string& key) {
        v.push_back({P(0, name), &aux_, aux_off_.at(key)});
    };
    const auto Ls = stash_layout(T, true);
    const uint64_t NG = std::max(num_gdn_, 1u);
    int cur = 0;
    for (uint32_t i = 0; i < cfg_.layers; ++i) {
        const int s = slot_[i];
        if (cfg_.attention[i]) {
            std::vector<Binding> pb = {{"res_in", &res_[cur], 0}, {"delta_in", i == 0 ? &zero_ : &delta_[cur], 0},
                                       {"q_out", &apre_q_, 0}, {"gate_out", &apre_g_, 0}, {"resid_out", &apre_r_, 0}};
            bind_aux(pb, "in_norm1p", L(i, "in_norm1p"));
            bind_w(pb, "qkv_w", L(i, "qkv.w"));
            bind_w(pb, "qkv_s", L(i, "qkv.s"));
            bind_w(pb, "q_norm", L(i, "q_norm"));
            bind_w(pb, "k_norm", L(i, "k_norm"));
            pb.push_back({"rope_pos", &ids_, kIdsRopePos});
            pb.push_back({"cos_tab", &cos_tab_, 0});
            pb.push_back({"sin_tab", &sin_tab_, 0});
            for (uint32_t b = 0; b < B; ++b) {
                const Seq& q = *sq[b];
                const auto bs = std::to_string(b);
                for (uint64_t h = 0; h < KV; ++h) {
                    const uint64_t at = (h * uint64_t(q.cap) + q.pos) * D * kv_es();
                    pb.push_back({"k_out" + bs + "_" + std::to_string(h), &q.k[s], at});
                    pb.push_back({"v_out" + bs + "_" + std::to_string(h), &q.v[s], at});
                }
            }
            apre_batch_recipe(B, T).enqueue(stream_, pb);
            mark(1);
            // one launch for every slot's attention (G2E_ACORE_MERGED=1): measured no faster at 13K contexts (the
            // launches are not the cost: each reads its K/V, and the padding to a common bucket reads more)
            static const bool core_each = std::getenv("G2E_ACORE_MERGED") == nullptr;
            uint32_t Sm = 0;
            for (uint32_t b = 0; b < B; ++b) Sm = std::max(Sm, core_bucket(sq[b]->pos + T));
            bool merged = !core_each;
            for (uint32_t b = 0; b < B && merged; ++b) merged = Sm <= sq[b]->cap;
            if (merged) {
                std::vector<Binding> cb = {{"q_in", &apre_q_, 0}, {"last", &ids_, kIdsLast}, {"att_out", &att_, 0}};
                for (uint32_t b = 0; b < B; ++b) {
                    const Seq& q = *sq[b];
                    for (uint64_t h = 0; h < KV; ++h) {
                        const uint64_t head = h * uint64_t(q.cap) * D * kv_es();
                        const auto nm = std::to_string(b) + "_" + std::to_string(h);
                        cb.push_back({"k_cache" + nm, &q.k[s], head});
                        cb.push_back({"v_cache" + nm, &q.v[s], head});
                    }
                }
                acore_batch_recipe(B, T, Sm).enqueue(stream_, cb);
            }
            for (uint32_t b = 0; b < B && !merged; ++b) {
                const Seq& q = *sq[b];
                std::vector<Binding> cb = {{"q_in", &apre_q_, uint64_t(b) * T * NHD * 2},
                                           {"last", &ids_, kIdsLast + uint64_t(b) * 128},
                                           {"att_out", &att_, uint64_t(b) * T * NHD * 2}};
                for (uint64_t h = 0; h < KV; ++h) {
                    const uint64_t head = h * uint64_t(q.cap) * D * kv_es();
                    cb.push_back({P(0, "k_cache" + std::to_string(h)), &q.k[s], head});
                    cb.push_back({P(0, "v_cache" + std::to_string(h)), &q.v[s], head});
                }
                acore_recipe(T, core_rows(q.pos + T)).enqueue(stream_, cb);
            }
            mark(2);
            std::vector<Binding> qb = {{"gate_in", &apre_g_, 0}, {"att_in", &att_, 0}, {"resid_in", &apre_r_, 0},
                                       {"res_out", &res_[cur ^ 1], 0}, {"delta_out", &delta_[cur ^ 1], 0}};
            bind_aux(qb, "post_norm1p", L(i, "post_norm1p"));
            bind_w(qb, "gu_w", L(i, "gate_up.w"));
            bind_w(qb, "gu_s", L(i, "gate_up.s"));
            bind_w(qb, "down_w", L(i, "down.w"));
            bind_w(qb, "down_s", L(i, "down.s"));
            bind_w(qb, "o_w", L(i, "o.w"));
            bind_w(qb, "o_s", L(i, "o.s"));
            apost_batch_recipe(B, T).enqueue(stream_, qb);
            mark(3);
        } else {
            std::vector<Binding> gb = {{"res_in", &res_[cur], 0}, {"delta_in", i == 0 ? &zero_ : &delta_[cur], 0},
                                       {"res_out", &res_[cur ^ 1], 0}, {"delta_out", &delta_[cur ^ 1], 0}};
            bind_aux(gb, "in_norm1p", L(i, "in_norm1p"));
            bind_aux(gb, "post_norm1p", L(i, "post_norm1p"));
            bind_w(gb, "gu_w", L(i, "gate_up.w"));
            bind_w(gb, "gu_s", L(i, "gate_up.s"));
            bind_w(gb, "down_w", L(i, "down.w"));
            bind_w(gb, "down_s", L(i, "down.s"));
            bind_w(gb, "qkvz_w", L(i, "qkvz.w"));
            bind_w(gb, "qkvz_s", L(i, "qkvz.s"));
            bind_w(gb, "ba_w", L(i, "ba"));
            bind_aux(gb, "conv_t", L(i, "conv_t"));
            bind_aux(gb, "negexpA", L(i, "negexpA"));
            bind_w(gb, "dt_bias", L(i, "dt_bias"));
            bind_w(gb, "gdn_norm", L(i, "gdn_norm"));
            bind_w(gb, "out_w", L(i, "out.w"));
            bind_w(gb, "out_s", L(i, "out.s"));
            const char* fields[] = {"st_x", nullptr, "st_v", "st_g", "st_b", "st_seq"};
            for (uint32_t b = 0; b < B; ++b) {
                const Seq& q = *sq[b];
                const auto bs = std::to_string(b);
                gb.push_back({P(0, "conv_in" + bs), &q.conv[q.conv_side[s]], uint64_t(s) * conv_bytes_});
                gb.push_back({P(0, "ssm_in" + bs), &q.ssm[q.ssm_side[s]], uint64_t(s) * ssm_bytes_});
                for (int f = 0; f < 6; ++f)
                    if (fields[f] && Ls.size[f])
                        gb.push_back({P(0, std::string(fields[f]) + bs), &spec_stash_,
                                      Ls.base[f] + (uint64_t(b) * NG + uint64_t(s)) * Ls.size[f]});
            }
            gdn_batch_recipe(B, T).enqueue(stream_, gb);
            mark(4);
        }
        for (size_t k = 0; k < capture_.size(); ++k)
            if (capture_[k] == i) {
                const Binding cb[] = {{"res", &res_[cur ^ 1], 0}, {"delta", &delta_[cur ^ 1], 0},
                                      {"feat", &feats_, k * feature_stride()}};
                capture_recipe(R).enqueue(stream_, cb);
                mark(5);
            }
        cur ^= 1;
    }
    cur_ = cur;
    const bool sampling = !inv_temps.empty();
    if (!sampling) {  // greedy: the FP8 head's candidates, rescored in BF16
        auto& hr = verify_head_recipe(R);
        std::vector<Binding> hb = {{"res", &res_[cur], 0},
                                   {"delta", &delta_[cur], 0},
                                   {"final_norm1p", &aux_, aux_off_.at("final_norm1p")},
                                   w_.bind("lm_head", "lm_head"),
                                   {"lm8", &lm8_, 0},
                                   {"lm8s", &lm8s_, 0},
                                   {"tkc_consts", &aux_, aux_off_.at("topk2c_T" + std::to_string(R))},
                                   {"cand8", &argmax_, 0},
                                   {"best", &argmax_, kBestOff}};
        hr.enqueue(stream_, hb);
        mark(6);
        e1.record(stream_);
        stream_.download(argmax_, 0, staging_, 0, kBestOff + uint64_t(R) * 4);
    } else {  // sampling: per row the top kTopK of the logits at its slot's temperature, the softmax statistics
        auto* it = reinterpret_cast<float*>(staging_.as<uint8_t>() + kTempStage);
        for (uint32_t b = 0; b < B; ++b)
            for (uint32_t t = 0; t < T; ++t) it[b * T + t] = inv_temps[b];
        stream_.upload(staging_, kTempStage, tempb_, 0, uint64_t(R) * 4);
        auto& hr = verify_topkb_recipe(R);
        std::vector<Binding> hb = {{"res", &res_[cur], 0},
                                   {"delta", &delta_[cur], 0},
                                   {"final_norm1p", &aux_, aux_off_.at("final_norm1p")},
                                   w_.bind("lm_head", "lm_head"),
                                   {"inv_temp", &tempb_, 0},
                                   {"tk_consts", &aux_, aux_off_.at("topk2_T" + std::to_string(R))},
                                   {"top_v", &vtopb_, 0},
                                   {"top_i", &vtopb_, kVtbIds},
                                   {"mx", &vtopb_, kVtbMax},
                                   {"sum", &vtopb_, kVtbSum}};
        hr.enqueue(stream_, hb);
        mark(6);
        e1.record(stream_);
        stream_.download(vtopb_, 0, staging_, 0, kVtbEnd);
    }
    if (!done_ev_) done_ev_ = std::make_unique<Event>(dev_);
    done_ev_->record(stream_);
    for (uint32_t b = 0; b < B; ++b) {  // pending until accept() (the slot selected)
        sq[b]->pending_T = T;
        sq[b]->stash_b = b;
        sq[b]->pending_wy = true;
    }
    pending_total_ = B;
    if (launched) launched();
    done_ev_->synchronize();
    last_ms_ = Event::elapsed_ms(e0, e1);
    if (prof) {
        double part[7] = {};
        part[0] = Event::elapsed_ms(e0, *pev[0]);
        for (size_t k = 1; k < pkind.size(); ++k) part[pkind[k]] += Event::elapsed_ms(*pev[k - 1], *pev[k]);
        std::fprintf(stderr, "[g2e] verify_batch %u x %u%s: %.1f ms = embed %.2f, attn pre %.2f, attn core %.2f, attn post %.2f, "
                             "gdn %.2f, capture %.2f, head %.2f\n",
                     B, T, sampling ? " (sampling)" : "", last_ms_, part[0], part[1], part[2], part[3], part[4], part[5], part[6]);
    }
    BatchOut out;
    if (!sampling) {
        const auto* cand = staging_.as<int32_t>();
        const auto* best = reinterpret_cast<const int32_t*>(staging_.as<uint8_t>() + kBestOff);
        out.targets.assign(B, std::vector<int32_t>(T));
        for (uint32_t b = 0; b < B; ++b)
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t r = b * T + t;
                out.targets[b][t] = cand[r * kHeadCand + uint32_t(best[r])];
            }
        return out;
    }
    const auto* base = staging_.as<uint8_t>();
    const auto* vals = reinterpret_cast<const float*>(base);
    const auto* ids = reinterpret_cast<const int32_t*>(base + kVtbIds);
    const auto* mx = reinterpret_cast<const float*>(base + kVtbMax);
    const auto* sm = reinterpret_cast<const float*>(base + kVtbSum);
    out.tops.resize(B);
    for (uint32_t b = 0; b < B; ++b) {
        auto& v = out.tops[b];
        const uint64_t r0 = uint64_t(b) * T;
        v.values.assign(vals + r0 * kTopK, vals + (r0 + T) * kTopK);
        v.ids.assign(ids + r0 * kTopK, ids + (r0 + T) * kTopK);
        v.max.assign(mx + r0, mx + r0 + T);
        v.sum.assign(sm + r0, sm + r0 + T);
    }
    return out;
}

void Qwen35::run_layers(std::span<const int32_t> tokens, bool spec, std::vector<std::vector<uint16_t>>* dump) {
    const uint32_t T = uint32_t(tokens.size());
    if (!T || T > kMaxT) throw std::invalid_argument("qwen35: 1.." + std::to_string(kMaxT) + " tokens per forward");
    if (pos_ + T > cap_) throw std::out_of_range("qwen35: context full");
    const uint64_t H = cfg_.hidden, R2 = cfg_.rotary_dim / 2, D = cfg_.head_dim, KV = cfg_.kv_heads;
    const uint64_t row = H * 2;
    const uint32_t Sb = bucket_for(std::max<uint32_t>(pos_, 1));
    (void)row;
    // the previous forward's uploads read the same staging regions
    if (stage_ev_ && stage_ev_->recorded()) stage_ev_->synchronize();
    // embeddings: one gather launch (a DMA per row cost ~30 us of host time each)
    std::memcpy(staging_.as<uint8_t>() + kIdsStage, tokens.data(), T * 4);
    *reinterpret_cast<float*>(staging_.as<uint8_t>() + kIdsStage + kIdsPosm1) = float(pos_) - 1.f;  // attn_mask
    {   // last visible cache row of each query row (masked softmax kernel): pos + t
        auto* last = reinterpret_cast<int32_t*>(staging_.as<uint8_t>() + kIdsStage + kIdsLast);
        for (uint32_t t = 0; t < T; ++t) last[t] = int32_t(pos_ + t);
    }
    stream_.upload(staging_, kIdsStage, ids_, 0, kIdsLast + uint64_t(T) * 4);
    {
        const Binding eb2[] = {{"ids", &ids_, 0}, w_.bind("embed", "embed"), {"res", &res_[0], 0}};
        embed_recipe(T).enqueue(stream_, eb2);
    }
    for (const auto& in : inject_)  // image features over their placeholder rows
        stream_.copy(*in.src, in.src_offset, res_[0], uint64_t(in.row) * H * 2, uint64_t(in.rows) * H * 2);
    inject_.clear();
    if (rope_explicit_) {  // a forward with image tokens: its own RoPE rows (staging [0, ...) is free here)
        std::memcpy(staging_.data(), rope_rows_.data(), rope_rows_.size() * 4);
        stream_.upload(staging_, 0, rope_cs_, 0, uint64_t(T) * R2 * 4);
        stream_.upload(staging_, uint64_t(T) * R2 * 4, rope_cs_, uint64_t(kMaxT) * R2 * 4, uint64_t(T) * R2 * 4);
    }
    // RoPE rows: the table rows of the rotary positions, or the explicit ones
    const DeviceBuffer* cos_buf = rope_explicit_ ? &rope_cs_ : &cos_tab_;
    const DeviceBuffer* sin_buf = rope_explicit_ ? &rope_cs_ : &sin_tab_;
    const uint64_t cos_at = rope_explicit_ ? 0 : uint64_t(int64_t(pos_) + rope_delta_) * R2 * 4;
    const uint64_t sin_at = rope_explicit_ ? uint64_t(kMaxT) * R2 * 4 : cos_at;
    rope_explicit_ = false;
    // RoPE rows and the attention mask come from device tables (attn_mask): no per-forward host tables
    if (!stage_ev_) stage_ev_ = std::make_unique<Event>(dev_);
    stage_ev_->record(stream_);

    Event e0(dev_), e1(dev_), eb(dev_);
    e0.record(stream_);
    int cur = 0;
    building_spec_ = spec;
    const uint32_t nlayers = layer_limit_ ? std::min(layer_limit_, cfg_.layers) : cfg_.layers;
    auto bind_w = [&](std::vector<Binding>& b, uint32_t j, const std::string& name, const std::string& weight) {
        b.push_back(w_.bind(P(j, name), weight));
    };
    auto bind_aux = [&](std::vector<Binding>& b, uint32_t j, const std::string& name, const std::string& key) {
        b.push_back({P(j, name), &aux_, aux_off_.at(key)});
    };
    std::vector<std::string> prof_names;
    if (profile_) while (prof_ev_.size() < nlayers / block_ + 2) prof_ev_.push_back(std::make_unique<Event>(dev_));
    if (profile_) prof_ev_[0]->record(stream_);
    const bool split = attn_split(T);
    const uint32_t Sb2 = attn_rows(std::min<uint32_t>(pos_ + T, cap_), T);  // split: cache rows incl. the new ones
    if (split && fp8_prefill(T, Sb2) && pos_ + fp8_tail(T) < Sb2)
        throw std::logic_error("qwen35: FP8 prefill attention prefix past the piece (pos " + std::to_string(pos_) + ", rows " +
                               std::to_string(Sb2) + ")");
    // G2E_TRACE=<file> (diagnostic): the last row's res and delta after every layer of each forward, appended
    static const char* trace_path = std::getenv("G2E_TRACE");
    if (trace_path && !trace_buf_) trace_buf_ = std::make_unique<DeviceBuffer>(dev_, uint64_t(cfg_.layers + 1) * 2 * H * 2);
    auto trace = [&](uint32_t layer, int side) {
        if (!trace_path) return;
        stream_.copy(res_[side], uint64_t(T - 1) * H * 2, *trace_buf_, uint64_t(layer) * 2 * H * 2, H * 2);
        stream_.copy(delta_[side], uint64_t(T - 1) * H * 2, *trace_buf_, (uint64_t(layer) * 2 + 1) * H * 2, H * 2);
    };
    uint32_t nb = 1, li = 0;  // layers in this launch, launch index (profile events)
    for (uint32_t first = 0; first < nlayers; first += nb, ++li) {
        nb = group_at(first, nlayers);
        if (first > 0) trace(first - 1, cur);
        if (split && cfg_.attention[first]) {
            const int s = slot_[first];
            const uint32_t i = first;
            auto& pr = attn_pre_recipe(T);
            if (profile_) prof_names.push_back(pr.name());
            std::vector<Binding> pb = {{"res_in", &res_[cur], 0},
                                       {"delta_in", first == 0 ? &zero_ : &delta_[cur], 0},
                                       {"cos", cos_buf, cos_at},
                                       {"sin", sin_buf, sin_at},
                                       {"q_out", &apre_q_, 0}, {"gate_out", &apre_g_, 0}, {"resid_out", &apre_r_, 0}};
            bind_aux(pb, 0, "in_norm1p", L(i, "in_norm1p"));
            bind_w(pb, 0, "qkv_w", L(i, "qkv.w"));
            bind_w(pb, 0, "qkv_s", L(i, "qkv.s"));
            bind_w(pb, 0, "q_norm", L(i, "q_norm"));
            bind_w(pb, 0, "k_norm", L(i, "k_norm"));
            for (uint64_t h = 0; h < KV; ++h) {
                const uint64_t at = (h * uint64_t(cap_) + pos_) * D * kv_es();
                pb.push_back({"k_out" + std::to_string(h), &k_cache_[s], at});
                pb.push_back({"v_out" + std::to_string(h), &v_cache_[s], at});
            }
            pr.enqueue(stream_, pb);
            if (profile_) prof_ev_[li + 1]->record(stream_);  // (attributed with the post launch)
            auto& qr = attn_post_recipe(T, Sb2);
            if (profile_) prof_names.back() = qr.name();
            std::vector<Binding> qb = {{"q_in", &apre_q_, 0}, {"gate_in", &apre_g_, 0}, {"resid_in", &apre_r_, 0},
                                       {"res_out", &res_[cur ^ 1], 0}, {"delta_out", &delta_[cur ^ 1], 0},
                                       {"posm1", &ids_, kIdsPosm1},
                                       {"iota", &aux_, aux_off_.at("iota")},
                                       {"iota_t", &aux_, aux_off_.at("iota_t")}};
            bind_aux(qb, 0, "post_norm1p", L(i, "post_norm1p"));
            bind_w(qb, 0, "gu_w", L(i, "gate_up.w"));
            bind_w(qb, 0, "gu_s", L(i, "gate_up.s"));
            bind_w(qb, 0, "down_w", L(i, "down.w"));
            bind_w(qb, 0, "down_s", L(i, "down.s"));
            bind_w(qb, 0, "o_w", L(i, "o.w"));
            bind_w(qb, 0, "o_s", L(i, "o.s"));
            for (uint64_t h = 0; h < KV; ++h) {
                const uint64_t head = h * uint64_t(cap_) * D * kv_es();
                qb.push_back({P(0, "k_cache" + std::to_string(h)), &k_cache_[s], head});
                qb.push_back({P(0, "v_cache" + std::to_string(h)), &v_cache_[s], head});
            }
            qb.push_back({"last", &ids_, kIdsLast});
            for (const char* c : {"f8_one", "f8_s128", "f8_d128"}) qb.push_back({c, &aux_, aux_off_.at(c)});
            {   // the mask inputs (graph / SDPA forms) or the last visible rows (kernel form): what the recipe has
                std::vector<Binding> kept;
                for (auto& bnd : qb)
                    for (const auto& t : qr.tensors())
                        if (t.name == bnd.name) { kept.push_back(bnd); break; }
                qb.swap(kept);
            }
            qr.enqueue(stream_, qb);
            for (size_t k = 0; k < capture_.size(); ++k)
                if (capture_[k] == first) {
                    const Binding cb[] = {{"res", &res_[cur ^ 1], 0}, {"delta", &delta_[cur ^ 1], 0},
                                          {"feat", &feats_, k * feature_stride()}};
                    capture_recipe(T).enqueue(stream_, cb);
                }
            if (profile_) prof_ev_[li + 1]->record(stream_);
            cur ^= 1;
            continue;
        }
        launch_n_ = nb;
        auto& r = block_recipe(first, T, Sb);
        launch_n_ = 0;
        if (profile_) prof_names.push_back(r.name());
        const int debug_j = building_debug_j_;
        std::vector<Binding> b = {{"res_in", &res_[cur], 0},
                                  {"delta_in", first == 0 ? &zero_ : &delta_[cur], 0},
                                  {"res_out", &res_[cur ^ 1], 0},
                                  {"delta_out", &delta_[cur ^ 1], 0}};
        bool has_attention = false, chunk_consts = false;
        std::vector<int> flip_conv, flip_ssm;
        for (uint32_t j = 0; j < nb; ++j) {
            const uint32_t i = first + j;
            const int s = slot_[i];
            bind_aux(b, j, "in_norm1p", L(i, "in_norm1p"));
            bind_aux(b, j, "post_norm1p", L(i, "post_norm1p"));
            bind_w(b, j, "gu_w", L(i, "gate_up.w"));
            bind_w(b, j, "gu_s", L(i, "gate_up.s"));
            bind_w(b, j, "down_w", L(i, "down.w"));
            bind_w(b, j, "down_s", L(i, "down.s"));
            if (cfg_.attention[i]) {
                has_attention = true;
                bind_w(b, j, "qkv_w", L(i, "qkv.w"));
                bind_w(b, j, "qkv_s", L(i, "qkv.s"));
                bind_w(b, j, "q_norm", L(i, "q_norm"));
                bind_w(b, j, "k_norm", L(i, "k_norm"));
                bind_w(b, j, "o_w", L(i, "o.w"));
                bind_w(b, j, "o_s", L(i, "o.s"));
                for (uint64_t h = 0; h < KV; ++h) {
                    const auto hs = std::to_string(h);
                    const uint64_t head = h * uint64_t(cap_) * D * kv_es();
                    b.push_back({P(j, "k_cache" + hs), &k_cache_[s], head});
                    b.push_back({P(j, "v_cache" + hs), &v_cache_[s], head});
                }
                b.push_back({P(j, "kv_new"), &kv_scratch_, uint64_t(s) * 2 * KV * T * D * 2});
            } else {
                bind_w(b, j, "qkvz_w", L(i, "qkvz.w"));
                bind_w(b, j, "qkvz_s", L(i, "qkvz.s"));
                bind_w(b, j, "ba_w", L(i, "ba"));
                bind_aux(b, j, "conv_t", L(i, "conv_t"));
                bind_aux(b, j, "negexpA", L(i, "negexpA"));
                bind_w(b, j, "dt_bias", L(i, "dt_bias"));
                bind_w(b, j, "gdn_norm", L(i, "gdn_norm"));
                bind_w(b, j, "out_w", L(i, "out.w"));
                bind_w(b, j, "out_s", L(i, "out.s"));
                b.push_back({P(j, "conv_in"), &conv_all_[conv_side_[s]], uint64_t(s) * conv_bytes_});
                if (!(spec && T > 1)) b.push_back({P(j, "conv_out"), &conv_all_[conv_side_[s] ^ 1], uint64_t(s) * conv_bytes_});
                flip_conv.push_back(s);
                if (T == 1) {
                    b.push_back({P(j, "ssm"), &ssm_all_[ssm_side_[s]], uint64_t(s) * ssm_bytes_});

                } else {
                    chunk_consts = T >= 8 && (T & (T - 1)) == 0 && !(spec && spec_wy(T));
                    b.push_back({P(j, "ssm_in"), &ssm_all_[ssm_side_[s]], uint64_t(s) * ssm_bytes_});
                    if (spec) {
                        const auto L = stash_layout(T, spec_wy(T));
                        const char* fields[] = {"st_q", "st_k", "st_v", "st_g", "st_b", "st_seq"};
                        if (spec_wy(T)) fields[0] = "st_x";
                        for (int f = 0; f < 6; ++f)
                            if (L.size[f]) b.push_back({P(j, fields[f]), &spec_stash_, L.base[f] + uint64_t(s) * L.size[f]});
                        flip_conv.pop_back();  // the commit writes the conv windows
                    } else {
                        b.push_back({P(j, "ssm_out"), &ssm_all_[ssm_side_[s] ^ 1], uint64_t(s) * ssm_bytes_});
                        flip_ssm.push_back(s);
                    }
                }
            }
        }
        for (size_t k = 0; k < capture_.size(); ++k)
            if (nb > 1 && capture_[k] >= first && capture_[k] < first + nb)
                b.push_back({"feat_j" + std::to_string(capture_[k] - first), &feats_, k * feature_stride()});
        if (chunk_consts) {
            const std::string cp = "chunk" + std::to_string(std::min<uint32_t>(T, gdn_chunk())) + ".";
            std::vector<std::string> names = {"upper", "strict", "inclusive", "lower", "eye", "diag8"};
            for (uint32_t sz = 8; sz < std::min<uint32_t>(T, gdn_chunk()); sz *= 2) names.push_back("ll" + std::to_string(sz));
            for (const auto& nm : names) b.push_back({cp + nm, &aux_, aux_off_.at(cp + nm)});
        }
        if (T == 1 && !flip_conv.empty()) {
            b.push_back({"idx", &consts_, 0});
            b.push_back({"acc", &consts_, 128});
        }
        if (has_attention) {
            b.push_back({"cos", cos_buf, cos_at});
            b.push_back({"sin", sin_buf, sin_at});
            b.push_back({"posm1", &ids_, kIdsPosm1});
            b.push_back({"iota", &aux_, aux_off_.at("iota")});
            b.push_back({"causal", &aux_, aux_off_.at("causal_T" + std::to_string(T))});
        }
        for (const auto& t : r.tensors()) {  // debug outputs
            if (t.name.rfind("dbg_", 0) != 0) continue;
            auto it = dbg_bufs_.find(t.name);
            if (it == dbg_bufs_.end() || it->second.size() < t.bytes())
                it = dbg_bufs_.insert_or_assign(t.name, DeviceBuffer(dev_, t.bytes())).first;
            b.push_back({t.name, &it->second, 0});
        }
        r.enqueue(stream_, b);
        if (nb == 1)  // separate capture launch: res + delta of the layer just run -> its feature slot
            for (size_t k = 0; k < capture_.size(); ++k)
                if (capture_[k] == first) {
                    const Binding cb[] = {{"res", &res_[cur ^ 1], 0}, {"delta", &delta_[cur ^ 1], 0},
                                          {"feat", &feats_, k * feature_stride()}};
                    capture_recipe(T).enqueue(stream_, cb);
                }
        if (profile_) prof_ev_[li + 1]->record(stream_);
        for (int s : flip_conv) conv_side_[s] ^= 1;
        for (int s : flip_ssm) ssm_side_[s] ^= 1;
        cur ^= 1;
        if (debug_j >= 0) {
            stream_.synchronize();
            for (const auto& t : r.tensors()) {
                if (t.name.rfind("dbg_", 0) != 0) continue;
                HostBuffer hb(dev_, t.bytes());
                stream_.download(dbg_bufs_.at(t.name), 0, hb, 0, t.bytes());
                stream_.synchronize();
                debug_out_[t.name.substr(4)] = {t.shape, std::vector<float>(hb.as<float>(), hb.as<float>() + t.elements())};
            }
        }
        if (dump) {
            stream_.download(res_[cur], 0, staging_, 0, T * row);
            stream_.download(delta_[cur], 0, staging_, T * row, T * row);
            stream_.synchronize();
            std::vector<uint16_t> h(T * H);
            const auto* a = staging_.as<uint16_t>();
            const auto* d = a + T * H;
            for (uint64_t k = 0; k < T * H; ++k) h[k] = to_bf16(from_bf16(a[k]) + from_bf16(d[k]));
            dump->push_back(std::move(h));
        }
    }
    building_debug_j_ = -1;
    if (!k_cache_.empty() && !split) {
        std::vector<Binding> cb = {{"scratch", &kv_scratch_, 0}};
        for (size_t s = 0; s < k_cache_.size(); ++s)
            for (uint64_t kv = 0; kv < 2; ++kv)
                for (uint64_t h = 0; h < KV; ++h) {
                    const uint64_t at = (h * uint64_t(cap_) + pos_) * D * kv_es();
                    cb.push_back({"c" + std::to_string((s * 2 + kv) * KV + h), kv ? &v_cache_[s] : &k_cache_[s], at});
                }
        // layer-limited runs only filled the scratch of the layers they ran; the rest is copied harmlessly
        commit_recipe(T).enqueue(stream_, cb);
    }
    trace(nlayers - 1, cur);
    if (trace_path) {  // append [layers][2][H] bf16 of this forward
        stream_.synchronize();
        HostBuffer hb(dev_, uint64_t(nlayers) * 2 * H * 2);
        stream_.download(*trace_buf_, 0, hb, 0, hb.size());
        stream_.synchronize();
        if (FILE* f = std::fopen(trace_path, "ab")) {
            std::fwrite(hb.data(), 1, hb.size(), f);
            std::fclose(f);
        }
    }
    eb.record(stream_);
    cur_ = cur;
    building_spec_ = false;
    if (profile_) {
        stream_.synchronize();
        prof_ = {};
        for (size_t k = 0; k < prof_names.size(); ++k)
            prof_.by_recipe[prof_names[k]] += Event::elapsed_ms(*prof_ev_[k], *prof_ev_[k + 1]);
        prof_.commit_ms = Event::elapsed_ms(*prof_ev_[prof_names.size()], eb);
    }
    (void)e1;
}

}  // namespace g2e
