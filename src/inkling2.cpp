#include "g2e/inkling2.hpp"
#include "g2e/inkling2_direct.hpp"
#include "g2e/numeric.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace g2e {

namespace {
void output(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto d = g.desc(src);
    const auto out = g.input(name, d.dtype, d.shape);
    g.op("identity", {src}, {out});
}
std::string Lw(uint32_t i, const char* part) { return "L" + std::to_string(i) + "." + part; }
constexpr float kMaskNeg = -1e30f;
uint64_t align128(uint64_t x) { return (x + 127) & ~uint64_t(127); }
}  // namespace

Inkling2Config Inkling2Config::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    if (j.value("model_type", std::string()) != "inkling2_loop")
        throw std::runtime_error("inkling2: pack " + pack.path() + " is not an inkling2_loop pack");
    const auto& c = j.at("config");
    Inkling2Config k;
    k.layers = c.at("num_hidden_layers");
    k.hidden = c.at("hidden_size");
    k.vocab = c.at("unpadded_vocab_size");
    k.embed_rows = c.at("vocab_size");
    k.heads = c.at("num_attention_heads");
    k.kv_heads = c.at("num_key_value_heads");
    k.head_dim = c.at("head_dim");
    k.d_rel = c.at("d_rel");
    k.rel_extent = c.at("rel_extent");
    k.window = c.at("sliding_window_size");
    k.experts = c.at("n_routed_experts");
    k.topk = c.at("num_experts_per_tok");
    k.moe_inter = c.at("intermediate_size");
    k.conv_kernel = c.at("sconv_kernel_size");
    k.attn_res_block = c.at("n_layers_per_attn_res_block");
    const auto& lc = c.at("loop_config");
    k.loop_start = lc.at("core_start");
    k.loop_end = lc.at("core_end");
    k.passes = lc.at("num_passes");
    k.gate_rank = lc.at("gate_rank");
    k.eps = c.at("rms_norm_eps");
    k.qk_clip = c.at("qk_l2_clip_threshold");
    k.rel_clip = c.at("rel_l2_clip_threshold");
    k.log_alpha = c.at("log_scaling_alpha");
    k.log_floor = c.at("log_scaling_n_floor");
    k.gate_softcap = c.at("mlp_gate_softcap");
    k.up_softcap = c.at("mlp_up_softcap");
    k.logit_softcap = c.at("final_logit_softcapping");
    k.route_scale = c.at("route_scale");
    for (const auto& s : j.at("layer_kinds")) k.full.push_back(s.get<std::string>() == "full");
    if (k.full.size() != k.layers)
        throw std::runtime_error("inkling2: pack " + pack.path() + ": layer_kinds size mismatch");
    if (j.contains("eos_token_id") && !j.at("eos_token_id").is_null()) k.eos = j.at("eos_token_id");
    if (j.contains("image_token_id")) k.image_token = j.at("image_token_id");
    k.pack_version = j.value("pack_version", 5u);
    const std::string exp_mode = j.value("experts", std::string("fp8"));
    k.mg_stack = j.value("mg_stack", false);
    auto reject_old_pack = [&]() {
        throw std::runtime_error(
            "inkling2: pack " + pack.path() +
            " uses an older FP8-row-scale expert format (v5 or pre-native-MXFP8 v6); "
            "reconvert with tools/convert_inkling2.py");
    };
    if (j.value("expert_layout", std::string()) != "ekn_gu2") reject_old_pack();
    if (k.head_dim != 128 || k.window != 1024 || k.rel_extent != 1024 || k.attn_res_block != 6 || k.conv_kernel != 4)
        throw std::runtime_error("inkling2: unexpected geometry");
    if (k.pack_version < 6) reject_old_pack();
    if (k.pack_version != 6)
        throw std::runtime_error("inkling2: pack " + pack.path() + ": unsupported pack_version " +
                                 std::to_string(k.pack_version));
    if (exp_mode != "native") reject_old_pack();
    if (!j.contains("layer_experts") || !j.at("layer_experts").is_array() || j.at("layer_experts").size() != k.layers)
        throw std::runtime_error("inkling2: pack " + pack.path() + ": missing or invalid layer_experts");
    k.expert_kind.resize(k.layers, ExpertKind::Nvfp4);
    const uint64_t E = k.experts, H = k.hidden, F = k.moe_inter;
    if ((H % 32) != 0 || (F % 32) != 0)
        throw std::runtime_error("inkling2: hidden and moe_inter must be multiples of 32");
    for (uint32_t L = 0; L < k.layers; ++L) {
        const auto kind = j.at("layer_experts").at(L).get<std::string>();
        if (kind == "fp8" || pack.has(Lw(L, "w13.w")) || !pack.has(Lw(L, "w13.wq"))) reject_old_pack();
        if (kind == "nvfp4") {
            k.expert_kind[L] = ExpertKind::Nvfp4;
            const uint64_t bs_k = expert_block_size(ExpertKind::Nvfp4);
            pack.check(Lw(L, "w13.wq"), DType::U8, {E, H / 2, 2 * F});
            pack.check(Lw(L, "w13.bs"), DType::F8, {E, H / bs_k, 2 * F});
            pack.check(Lw(L, "w13.g"), DType::F32, {E});
            pack.check(Lw(L, "w2.wq"), DType::U8, {E, F / 2, H});
            pack.check(Lw(L, "w2.bs"), DType::F8, {E, F / bs_k, H});
            pack.check(Lw(L, "w2.g"), DType::F32, {E});
        } else if (kind == "mxfp8") {
            k.expert_kind[L] = ExpertKind::Mxfp8;
            const uint64_t bs_k = expert_block_size(ExpertKind::Mxfp8);
            pack.check(Lw(L, "w13.wq"), DType::U8, {E, H, 2 * F});
            pack.check(Lw(L, "w13.bs"), DType::F8, {E, H / bs_k, 2 * F});
            pack.check(Lw(L, "w13.g"), DType::F32, {E});
            pack.check(Lw(L, "w2.wq"), DType::U8, {E, F, H});
            pack.check(Lw(L, "w2.bs"), DType::F8, {E, F / bs_k, H});
            pack.check(Lw(L, "w2.g"), DType::F32, {E});
        } else {
            throw std::runtime_error("inkling2: pack " + pack.path() + ": unknown layer_experts[" +
                                     std::to_string(L) + "]='" + kind + "'");
        }
    }
    if (k.has_nvfp4() && j.value("nvfp4_bs", std::string()) != "fp8_143_div8")
        throw std::runtime_error("inkling2: pack " + pack.path() + ": v6 pack expects nvfp4_bs='fp8_143_div8'");
    if (k.has_mxfp8() && j.value("mxfp8_bs", std::string()) != "fp8_143_le1")
        throw std::runtime_error("inkling2: pack " + pack.path() + ": v6 pack expects mxfp8_bs='fp8_143_le1'");
    return k;
}

Inkling2::Inkling2(const Device& device, const Stream& stream, const WeightPack& weights, uint32_t max_context,
                   uint32_t max_slots, uint32_t init_cap)
    : dev_(device), stream_(stream), w_(weights), cfg_(Inkling2Config::from_pack(weights)), max_ctx_(max_context),
      max_slots_(std::clamp(max_slots, 1u, 8u)), staging_(device, 64u << 20) {
    if (max_ctx_ % 1024) throw std::invalid_argument("inkling2: max_context must be a multiple of 1024");
    cap_ = (init_cap == UINT32_MAX) ? max_ctx_ : std::min(init_cap, max_ctx_);
    if (cap_ % 1024) throw std::invalid_argument("inkling2: init_cap must be a multiple of 1024");
    msg_stack_ = cfg_.mg_stack;
    const uint64_t H = cfg_.hidden, KV = cfg_.kv_heads, D = cfg_.head_dim;
    uint32_t idx = 0;
    for (uint32_t l = 0; l < cfg_.loop_start; ++l) apps_.push_back({l, 0, idx++});
    for (uint32_t p = 0; p < cfg_.passes; ++p)
        for (uint32_t l = cfg_.loop_start; l < cfg_.loop_end; ++l) apps_.push_back({l, p, idx++});
    for (uint32_t l = cfg_.loop_end; l < cfg_.layers; ++l) apps_.push_back({l, 0, idx++});
    // every application keeps a contiguous [max_slots] ring pool of its last 1024 K/V rows (the near keys); full layers
    // also keep dynamic per-slot linear caches [KV, cap, D] (the far keys).
    ring_slot_bytes_ = KV * cfg_.window * D * 2;
    for (const auto& a : apps_) {
        if (cfg_.full[a.layer]) {
            kv_index_.push_back(int(kcache_.size()));
            if (cap_ > 0) {
                kcache_.emplace_back(dev_, KV * cap_ * D * 2);
                vcache_.emplace_back(dev_, KV * cap_ * D * 2);
            } else {
                kcache_.emplace_back();
                vcache_.emplace_back();
            }
        } else {
            kv_index_.push_back(-1);
        }
        ring_index_.push_back(int(kring_.size()));
        kring_.emplace_back(dev_, uint64_t(max_slots_) * ring_slot_bytes_);
        vring_.emplace_back(dev_, uint64_t(max_slots_) * ring_slot_bytes_);
    }
    const uint64_t KVD = KV * D;
    conv_off_[0] = 0;
    conv_off_[1] = 3 * KVD * 2;
    conv_off_[2] = 6 * KVD * 2;
    conv_off_[3] = conv_off_[2] + 3 * H * 2;
    conv_app_bytes_ = align128(conv_off_[3] + 3 * H * 2);
    conv_slot_bytes_ = (apps_.size() * conv_app_bytes_ + 255) & ~uint64_t(255);
    conv_pool_ = DeviceBuffer(dev_, uint64_t(max_slots_) * 2 * conv_slot_bytes_);
    seqs_.resize(1);
    active_slot_ = 0;
    blocks_ = DeviceBuffer(dev_, 4ull * kMaxT * H * 2);
    for (auto& h : h_) h = DeviceBuffer(dev_, kMaxT * H * 2);
    for (auto& p : prev_) p = DeviceBuffer(dev_, 2ull * kMaxT * H * 2);  // [side][b1 | b2]
    kvscratch_ = DeviceBuffer(dev_, uint64_t(apps_.size()) * 2 * KVD * kMaxT * 2);
    {
        const uint64_t Sx = uint64_t(cfg_.window) + kMaxT, n = uint64_t(kMaxT) * Sx;
        tab_.swa_idx = 0;
        tab_.swa_mask = align128(tab_.swa_idx + n * 4);
        tab_.gn_idx = align128(tab_.swa_mask + n * 4);
        tab_.gn_valid = align128(tab_.gn_idx + n * 4);
        tab_.gn_mask = align128(tab_.gn_valid + n * 4);
        tab_.gf_mask = align128(tab_.gn_mask + n * 4);
        tab_.tau = align128(tab_.gf_mask + uint64_t(max_ctx_) * 4);
        tab_.rows = align128(tab_.tau + kMaxT * 4 + 1024);  // rows_ring [kMaxT] i32, rows_lin [kMaxT] i32
        tab_.slot_tok = align128(tab_.rows + 2ull * kMaxT * 4);   // per slot: token / 32, token % 32 (F32)
        tab_.slot_tok2 = align128(tab_.slot_tok + 8ull * kMaxT * 2 * 4);  // the same (a second binding)
        tab_.rel_rev = align128(tab_.slot_tok2 + 8ull * kMaxT * 2 * 4);   // RE - 1 - k (I32): the reversed bias table
        tab_.pos32 = align128(tab_.rel_rev + uint64_t(cfg_.rel_extent) * 4);  // the forward's first position (I32)
        tab_.bytes = align128(tab_.pos32 + 128);
        tables_ = DeviceBuffer(dev_, tab_.bytes);
    }
    logits_ = DeviceBuffer(dev_, uint64_t(cfg_.vocab) * 4 * 16);
    prepare_queries();
    prepare_nv_gvec();
    if (std::getenv("G2E_INK_DPREFILL")) {
        blocks1_ = DeviceBuffer(dev_, 4ull * kMaxT * H * 2);
        for (auto& h : h1_) h = DeviceBuffer(dev_, kMaxT * H * 2);
        for (auto& p : prev1_) p = DeviceBuffer(dev_, 2ull * kMaxT * H * 2);
        ids1_ = DeviceBuffer(dev_, kMaxT * 4);
    }
    if (std::getenv("G2E_INK_DIRECT") || std::getenv("G2E_INK_DPREFILL")) direct_ = std::make_unique<InkDirect>(*this);
    if (!direct_ || !direct_->has_runtime()) prepare_synapse_bf16();
    ids_ = DeviceBuffer(dev_, kMaxT * 4);
    // Zero all contiguous ring & conv pools once so every slot begins finite (no NaN/Inf in unwritten ring entries).
    stream_.zero(conv_pool_, 0, conv_pool_.size());
    for (auto& b : kring_) stream_.zero(b, 0, b.size());
    for (auto& b : vring_) stream_.zero(b, 0, b.size());
    if (cap_ > 0) {
        for (auto& b : kcache_) stream_.zero(b, 0, b.size());
        for (auto& b : vcache_) stream_.zero(b, 0, b.size());
    }
    stream_.synchronize();
    caches_zeroed_ = true;
    direct_prev_ = false;
}

Inkling2::~Inkling2() = default;

void Inkling2::direct_profile(uint32_t app) {
    if (direct_) direct_->profile_stages(app);
}

void Inkling2::direct_profile_v(uint32_t app, uint32_t T) {
    if (direct_) direct_->profile_stages_v(app, T);
}

void Inkling2::swap_seq(Seq& s) {
    std::swap(kcache_, s.kcache);
    std::swap(vcache_, s.vcache);
    std::swap(pos_, s.pos);
    std::swap(cap_, s.cap);
    std::swap(conv_side_, s.conv_side);
    std::swap(ver_T_, s.ver_T);
    std::swap(ver_side_, s.ver_side);
    std::swap(caches_zeroed_, s.caches_zeroed);
}

uint32_t Inkling2::round_capacity(uint32_t rows) const {
    if (rows == 0) return 0;
    for (uint32_t c : {4096u, 8192u, 16384u, 32768u, 65536u, 131072u})
        if (rows <= c) return std::min(c, max_ctx_);
    const uint32_t r = ((rows + 65535u) / 65536u) * 65536u;
    return std::min(r, max_ctx_);
}

uint64_t Inkling2::kv_bytes_for(uint32_t capacity) const {
    return uint64_t(kcache_.size()) * 2 * cfg_.kv_heads * capacity * cfg_.head_dim * 2;
}

uint64_t Inkling2::slot_kv(uint32_t slot) const {
    if (slot >= seqs_.size()) throw std::out_of_range("inkling2: bad slot");
    return kv_bytes_for(slot == active_slot_ ? cap_ : seqs_[slot].cap);
}

uint64_t Inkling2::kv_total() const {
    uint64_t tot = 0;
    for (uint32_t s = 0; s < seqs_.size(); ++s) tot += slot_kv(s);
    return tot;
}

uint32_t Inkling2::slot_position(uint32_t slot) const {
    if (slot >= seqs_.size()) throw std::out_of_range("inkling2: bad slot");
    return slot == active_slot_ ? pos_ : seqs_[slot].pos;
}

uint32_t Inkling2::slot_capacity(uint32_t slot) const {
    if (slot >= seqs_.size()) throw std::out_of_range("inkling2: bad slot");
    return slot == active_slot_ ? cap_ : seqs_[slot].cap;
}

int Inkling2::slot_conv_side(uint32_t slot) const {
    if (slot >= seqs_.size()) throw std::out_of_range("inkling2: bad slot");
    return slot == active_slot_ ? conv_side_ : seqs_[slot].conv_side;
}

uint32_t Inkling2::add_slot(uint32_t capacity) {
    if (seqs_.size() >= max_slots_) throw std::runtime_error("inkling2: max_slots reached");
    if (capacity > max_ctx_ || (capacity % 1024) != 0)
        throw std::invalid_argument("inkling2: bad slot capacity");
    const uint32_t id = uint32_t(seqs_.size());
    Seq s;
    s.cap = capacity;
    s.kcache.resize(kcache_.size());
    s.vcache.resize(vcache_.size());
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim;
    if (capacity > 0) {
        const uint64_t bytes = KV * capacity * D * 2;
        for (size_t i = 0; i < kcache_.size(); ++i) {
            s.kcache[i] = DeviceBuffer(dev_, bytes);
            s.vcache[i] = DeviceBuffer(dev_, bytes);
            stream_.zero(s.kcache[i], 0, bytes);
            stream_.zero(s.vcache[i], 0, bytes);
        }
    }
    stream_.zero(conv_pool_, conv_off_slot(id, 0), 2 * conv_slot_bytes_);
    stream_.synchronize();
    s.caches_zeroed = true;
    seqs_.push_back(std::move(s));
    return id;
}

void Inkling2::select_slot(uint32_t slot) {
    if (slot >= seqs_.size()) throw std::out_of_range("inkling2: bad slot");
    if (slot == active_slot_) return;
    if (ver_T_) throw std::logic_error("inkling2: select_slot with pending verification");
    if (direct_) direct_->flush();
    swap_seq(seqs_[active_slot_]);
    active_slot_ = slot;
    swap_seq(seqs_[active_slot_]);
    direct_prev_ = false;
}

void Inkling2::set_capacity(uint32_t capacity) {
    if (capacity == cap_) return;
    if (ver_T_) throw std::logic_error("inkling2: set_capacity with pending verification");
    if (capacity > max_ctx_ || (capacity % 1024) != 0)
        throw std::invalid_argument("inkling2: bad capacity");
    if (direct_) direct_->flush();
    stream_.synchronize();
    if (capacity == 0) {
        for (auto& b : kcache_) b = DeviceBuffer();
        for (auto& b : vcache_) b = DeviceBuffer();
        cap_ = 0;
        pos_ = 0;
        conv_side_ = 0;
        stream_.zero(conv_pool_, conv_off_slot(active_slot_, 0), 2 * conv_slot_bytes_);
        stream_.synchronize();
        caches_zeroed_ = false;
        direct_prev_ = false;
        if (direct_) direct_->on_slot_resized(active_slot_);
        return;
    }
    if (capacity < pos_) throw std::invalid_argument("inkling2: capacity smaller than current position");
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim;
    const uint64_t new_bytes = KV * capacity * D * 2;
    const uint64_t copy_rows = uint64_t(pos_) * D * 2;
    // Grow/shrink layer-by-layer so peak transient HBM overhead is only 1/16 of total KV (<= 2 GiB at 1M).
    for (size_t i = 0; i < kcache_.size(); ++i) {
        DeviceBuffer nk(dev_, new_bytes);
        stream_.zero(nk, 0, new_bytes);
        if (copy_rows && cap_ > 0 && kcache_[i].address()) {
            for (uint64_t h = 0; h < KV; ++h)
                stream_.copy(kcache_[i], h * uint64_t(cap_) * D * 2, nk, h * uint64_t(capacity) * D * 2, copy_rows);
        }
        stream_.synchronize();
        kcache_[i] = std::move(nk);

        DeviceBuffer nv(dev_, new_bytes);
        stream_.zero(nv, 0, new_bytes);
        if (copy_rows && cap_ > 0 && vcache_[i].address()) {
            for (uint64_t h = 0; h < KV; ++h)
                stream_.copy(vcache_[i], h * uint64_t(cap_) * D * 2, nv, h * uint64_t(capacity) * D * 2, copy_rows);
        }
        stream_.synchronize();
        vcache_[i] = std::move(nv);
    }
    cap_ = capacity;
    caches_zeroed_ = true;
    direct_prev_ = false;
    if (direct_) direct_->on_slot_resized(active_slot_);
}

void Inkling2::set_position_for_test(uint32_t pos) {
    if (pos > cap_) throw std::out_of_range("inkling2: test pos exceeds capacity");
    if (direct_) direct_->flush();
    stream_.synchronize();
    pos_ = pos;
    direct_prev_ = false;
}

void Inkling2::invalidate_sram() {
    direct_prev_ = false;
    if (direct_) direct_->invalidate_sram();
}

void Inkling2::prepare_synapse_bf16() {
    if (syn_bf16_.address()) return;
    const uint64_t E = cfg_.experts, H = cfg_.hidden, F = cfg_.moe_inter;
    std::vector<uint32_t> layers;
    uint32_t n_nv = 0, n_mx = 0;
    for (uint32_t L = 0; L < cfg_.layers; ++L) {
        layers.push_back(L);
        if (cfg_.is_nvfp4(L)) ++n_nv;
        else ++n_mx;
    }
    if (layers.empty()) return;
    syn_bf16_w13_bytes_ = E * H * 2 * F * 2;
    syn_bf16_w2_bytes_ = E * F * H * 2;
    const uint64_t w13_stride = (syn_bf16_w13_bytes_ + 255) & ~uint64_t(255);
    const uint64_t w2_stride = (syn_bf16_w2_bytes_ + 255) & ~uint64_t(255);
    const uint64_t layer_bytes = w13_stride + w2_stride;
    syn_bf16_ = DeviceBuffer(dev_, uint64_t(layers.size()) * layer_bytes);
    syn_bf16_off_.assign(cfg_.layers, SynBf16Off{});
    for (size_t i = 0; i < layers.size(); ++i) {
        const uint64_t base = uint64_t(i) * layer_bytes;
        syn_bf16_off_[layers[i]] = {base, base + w13_stride, true};
    }
    using EK = Inkling2Config::ExpertKind;
    auto deq_recipe = [&](EK kind, uint64_t K, uint64_t N) -> Recipe& {
        const bool is_nv = (kind == EK::Nvfp4);
        const char* op = is_nv ? "ink_nvf4_deq_bf16_gaudi2" : "ink_mxfp8_deq_bf16_gaudi2";
        const uint64_t w_k = is_nv ? (K / 2) : K;
        const uint64_t bs_k = K / Inkling2Config::expert_block_size(kind);
        const std::string key = std::string(is_nv ? "ink_nvf4_deq_bf16_E" : "ink_mxfp8_deq_bf16_E") +
                                std::to_string(E) + "_K" + std::to_string(K) + "_N" + std::to_string(N);
        if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
        GraphBuilder g(dev_);
        const auto wq = g.input("wq", DType::U8, {E, w_k, N});
        const auto bs = g.input("bs", DType::F8, {E, bs_k, N});
        const auto mask = g.input("mask", DType::I32, {E});
        const auto out = g.input("out", DType::BF16, {E, K, N});
        const int32_t use_mask = 0;
        g.op(op, {wq, bs, mask}, {out}, &use_mask, sizeof(use_mask));
        return *(recipes_[key] = g.compile(key));
    };
    Recipe* r13_nv = n_nv ? &deq_recipe(EK::Nvfp4, H, 2 * F) : nullptr;
    Recipe* r2_nv = n_nv ? &deq_recipe(EK::Nvfp4, F, H) : nullptr;
    Recipe* r13_mx = n_mx ? &deq_recipe(EK::Mxfp8, H, 2 * F) : nullptr;
    Recipe* r2_mx = n_mx ? &deq_recipe(EK::Mxfp8, F, H) : nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    for (uint32_t L : layers) {
        const auto& o = syn_bf16_off_[L];
        Recipe& r13 = cfg_.is_nvfp4(L) ? *r13_nv : *r13_mx;
        Recipe& r2 = cfg_.is_nvfp4(L) ? *r2_nv : *r2_mx;
        const Binding b13[] = {w_.bind("wq", Lw(L, "w13.wq")), w_.bind("bs", Lw(L, "w13.bs")),
                               {"mask", &tables_, 0}, {"out", &syn_bf16_, o.w13}};
        r13.enqueue(stream_, b13);
        const Binding b2[] = {w_.bind("wq", Lw(L, "w2.wq")), w_.bind("bs", Lw(L, "w2.bs")),
                              {"mask", &tables_, 0}, {"out", &syn_bf16_, o.w2}};
        r2.enqueue(stream_, b2);
    }
    stream_.synchronize();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    direct_prev_ = false;
    const uint64_t free_after = dev_.info().free_bytes;
    std::fprintf(stderr,
                 "[ink-bf16-synapse] expanded %zu native layers (%u NVFP4, %u MXFP8) to BF16 (%.2f GiB HBM, reducing KV budget; free %.2f GiB) in %.1f ms\n",
                 layers.size(), n_nv, n_mx, double(syn_bf16_.size()) / double(1ull << 30), double(free_after) / double(1ull << 30), ms);
}

void Inkling2::bind_bf16_experts(uint32_t layer, std::vector<Binding>& b) const {
    if (syn_bf16_.address()) {
        if (layer >= syn_bf16_off_.size() || !syn_bf16_off_[layer].valid)
            throw std::logic_error("inkling2: missing Synapse BF16 weights for layer L" + std::to_string(layer));
        b.push_back({"w13_bf16", &syn_bf16_, syn_bf16_off_[layer].w13});
        b.push_back({"w2_bf16", &syn_bf16_, syn_bf16_off_[layer].w2});
        return;
    }
    if (direct_ && direct_->has_runtime()) {
        const auto a = direct_->bf16_layer(layer);
        if (!a.cached)
            throw std::runtime_error("inkling2: Synapse graph on native layer L" + std::to_string(layer) +
                                     " requires the layer in the BF16 cache (set G2E_INK_BF16_CACHE=all and call init_bf16_cache())");
        const DeviceBuffer& cache = direct_->bf16_cache();
        b.push_back({"w13_bf16", &cache, a.w13 - cache.address()});
        b.push_back({"w2_bf16", &cache, a.w2 - cache.address()});
        return;
    }
    throw std::logic_error("inkling2: BF16 weights for native layer L" + std::to_string(layer) + " are not initialized");
}

void Inkling2::init_bf16_cache(uint64_t kv_need_bytes) {
    if (syn_bf16_.address()) return;  // already expanded at load in Synapse-only mode
    if (!direct_ || !direct_->has_runtime()) {
        prepare_synapse_bf16();
        return;
    }
    direct_->init_bf16_cache(kv_need_bytes);
}

Inkling2::Bf16LayerAddrs Inkling2::bf16_layer(uint32_t layer) const {
    if (syn_bf16_.address()) {
        if (layer >= syn_bf16_off_.size() || !syn_bf16_off_[layer].valid) return {};
        const uint64_t base = syn_bf16_.address();
        return {base + syn_bf16_off_[layer].w13, base + syn_bf16_off_[layer].w2, true};
    }
    if (!direct_) return {};
    const auto a = direct_->bf16_layer(layer);
    return {a.w13, a.w2, a.cached};
}

uint64_t Inkling2::bf16_staging_w13() const { return direct_ ? direct_->bf16_staging_w13() : 0; }
uint64_t Inkling2::bf16_staging_w2() const { return direct_ ? direct_->bf16_staging_w2() : 0; }
uint64_t Inkling2::bf16_w13_bytes() const { return syn_bf16_.address() ? syn_bf16_w13_bytes_ : (direct_ ? direct_->bf16_w13_bytes() : 0); }
uint64_t Inkling2::bf16_w2_bytes() const { return syn_bf16_.address() ? syn_bf16_w2_bytes_ : (direct_ ? direct_->bf16_w2_bytes() : 0); }

void Inkling2::download_bf16_layer(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out) const {
    if (syn_bf16_.address()) {
        if (layer >= syn_bf16_off_.size() || !syn_bf16_off_[layer].valid)
            throw std::invalid_argument("inkling2: layer L" + std::to_string(layer) + " is not in the Synapse BF16 buffer");
        w13_out.resize(syn_bf16_w13_bytes_ / 2);
        w2_out.resize(syn_bf16_w2_bytes_ / 2);
        auto dl = [&](uint64_t off, uint64_t bytes, uint16_t* dst) {
            const uint64_t chunk = staging_.size();
            for (uint64_t done = 0; done < bytes; done += chunk) {
                const uint64_t n = std::min(chunk, bytes - done);
                stream_.download(syn_bf16_, off + done, staging_, 0, n);
                stream_.synchronize();
                std::memcpy(reinterpret_cast<uint8_t*>(dst) + done, staging_.data(), n);
            }
        };
        dl(syn_bf16_off_[layer].w13, syn_bf16_w13_bytes_, w13_out.data());
        dl(syn_bf16_off_[layer].w2, syn_bf16_w2_bytes_, w2_out.data());
        return;
    }
    if (!direct_) throw std::logic_error("inkling2: download_bf16_layer requires Direct or Synapse BF16 cache");
    direct_->download_bf16_layer(layer, w13_out, w2_out);
}

void Inkling2::download_bf16_staged(uint32_t layer, std::vector<uint16_t>& w13_out, std::vector<uint16_t>& w2_out) {
    if (syn_bf16_.address()) {
        download_bf16_layer(layer, w13_out, w2_out);
        return;
    }
    if (!direct_) throw std::logic_error("inkling2: download_bf16_staged requires Direct or Synapse BF16 cache");
    direct_->download_bf16_staged(layer, w13_out, w2_out);
}

void Inkling2::reset() {
    if (direct_) direct_->flush();
    pos_ = 0;
    conv_side_ = 0;
    stream_.zero(conv_pool_, conv_off_slot(active_slot_, 0), 2 * conv_slot_bytes_);
    // the KV rings and linear caches only once (~4 GB at a 128K context: ~90 ms per request): every read of them is
    // limited to written rows (attention masks; zero probabilities against finite stale rows)
    static const bool always = std::getenv("G2E_INK_RESET_ZERO") != nullptr;
    if (!caches_zeroed_ || always) {
        const uint64_t ro = ring_off(active_slot_);
        for (auto& b : kring_) stream_.zero(b, ro, ring_slot_bytes_);
        for (auto& b : vring_) stream_.zero(b, ro, ring_slot_bytes_);
        if (cap_ > 0) {
            for (auto& b : kcache_) if (b.size()) stream_.zero(b, 0, b.size());
            for (auto& b : vcache_) if (b.size()) stream_.zero(b, 0, b.size());
        }
        caches_zeroed_ = true;
    }
    stream_.synchronize();
    direct_prev_ = false;
}

void Inkling2::inject_rows(const DeviceBuffer* src, uint64_t offset, uint32_t pos, uint32_t rows) {
    inject_.push_back({src, offset, pos, rows});
}

size_t Inkling2::precompile_far(uint32_t T, bool vh) {
    size_t n = 0;
    struct Restore { bool& v; bool old; ~Restore() { v = old; } } restore{verify_, verify_};
    verify_ = vh;
    if (vh)  // the sliding recipes too (the normal ones are compiled by any prefill)
        for (const auto& a : apps_)
            if (!cfg_.full[a.layer]) {
                const size_t before = recipes_.size();
                layer_recipe(a, T, 0);
                n += recipes_.size() - before;
            }
    std::vector<uint32_t> buckets;
    for (uint32_t rows = 1; rows <= max_ctx_;) {
        const uint32_t b = far_bucket(rows);
        if (buckets.empty() || buckets.back() != b) buckets.push_back(b);
        if (b >= max_ctx_) break;
        rows = b + 1;
    }
    for (const uint32_t Sf : buckets)
        for (const auto& a : apps_)
            if (cfg_.full[a.layer]) {
                const size_t before = recipes_.size();
                layer_recipe(a, T, Sf);
                n += recipes_.size() - before;
            }
    return n;
}

namespace { bool kon(int bit); }  // (defined below)
// full layers with far keys through the FP16 softmax kernel (ink_attn_sm16f): no query blocks (the scores are
// FP16 per KV head, the heads one after another). G2E_INK_OLD_FAR=1: the F32 graph path in query blocks.
static bool sm16_far_ok(uint32_t T, uint32_t Sf, uint32_t W) {
    static const bool off = std::getenv("G2E_INK_OLD_FAR") != nullptr;
    return !off && W == 1024 && Sf > 0 && (T == 128 || T == 256 || T == 512 || T == 1024);
}

uint32_t Inkling2::attn_block(uint32_t T, uint32_t Sf) const {
    if (sm16_far_ok(T, Sf, cfg_.window) && kon(32)) return T;
    // query rows per attention block of a full layer: the F32 scores [KV, 3 Tq, Sf + W + T] stay within ~800 MB
    static const uint64_t budget = [] {
        const char* e = std::getenv("G2E_INK_SCORE_ROWS");  // Tq * (Sf + W + T) limit
        return e ? uint64_t(std::atoll(e)) : (8ull << 20);
    }();
    uint32_t tq = T;
    while (tq > 16 && uint64_t(tq) * (Sf + cfg_.window + T) > budget) tq /= 2;
    return tq;
}

uint32_t Inkling2::far_bucket(uint32_t rows) const {
    // rows of the far region (positions older than the 1024 near rows): 0, then 1024, 2048, 3072, 4096, 6144, ...
    if (rows == 0) return 0;
    uint32_t b = 1024;
    while (b < rows) {
        const uint32_t step = b < 4096 ? 1024 : b / 2;
        b += step;
    }
    return std::min(b, cap_ > 0 ? cap_ : max_ctx_);
}

// ------------------------------------------------------------------------------------------------ graph helpers

namespace {
bool ink_ref() {  // G2E_INK_REF=1: the reference graph-op path instead of the fused TPC kernels
    static const bool on = std::getenv("G2E_INK_REF") != nullptr;
    return on;
}
// G2E_INK_KMASK (debug): which fused kernels are on (1 q prep, 2 k/v convs, 4 attn/mlp convs, 8/16 sliding/full
// k/v convs, 32 depth attention, 64 router, 128 message gates); default all
bool kon(int bit) {
    static const int m = [] { const char* e = std::getenv("G2E_INK_KMASK"); return e ? std::atoi(e) : 255; }();
    return !ink_ref() && (m & bit);
}
struct InkConvParams { float scale, thr, eps; int32_t mode; };
}  // namespace


// F32 [.., H] natural order <-> the interleaved order of the TPC kernels (every 128 values: [even 64 | odd 64])
TensorId Inkling2::interleave128(Ops& o, const std::string& n, TensorId x, bool inverse) {
    Shape s = o.shape(x);
    const uint64_t H = s.back();
    Shape a(s.begin(), s.end() - 1), b = a;
    a.push_back(H / 128); b.push_back(H / 128);
    if (!inverse) { a.push_back(64); a.push_back(2); } else { a.push_back(2); a.push_back(64); }
    std::vector<unsigned> perm;
    const unsigned r = unsigned(a.size());
    for (unsigned i = 0; i + 2 < r; ++i) perm.push_back(i);
    perm.push_back(r - 1);
    perm.push_back(r - 2);
    return o.reshape(o.transpose(o.reshape(x, a, n + "_a"), perm, n + "_t"), s, n + "_r");
}

// fused depth attention + RMSNorm (kernels/ink_mix_gaudi2.c) over [blocks (BF16 [S0, T, H]); part]:
// part BF16 [T, H] or F32 [T, H] (natural order). Returns {normed BF16 [T, Q, H], mix F32 [T, Q, H] natural (when
// want_mix, else kNoTensor)}.
std::pair<TensorId, TensorId> Inkling2::fused_mix(Ops& o, const std::string& n, TensorId blocks, TensorId part, TensorId q,
                                                  TensorId w, uint32_t T, uint32_t Q, bool want_mix, bool mix_natural) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden;
    const auto qi = q;  // bound interleaved (prepare_queries)
    const auto normed = o.temp(n + "_normed", DType::BF16, {T, Q, H});
    const auto mix = o.temp(n + "_mix", DType::F32, {T, Q, H});
    // flags: bit 0 the mix, bit 1 the normed output (the callers use one of them)
    {   // the kernel's compile-time shapes (kernels/ink_mix_gaudi2.c)
        const uint64_t S = o.shape(blocks)[0] + 1;
        if (!((Q == 1 || Q == 3) && S >= 2 && S <= 5) && !(Q == 4 && S == 2))
            throw std::invalid_argument("inkling2 fused_mix: no kernel for S=" + std::to_string(S) + " Q=" + std::to_string(Q));
    }
    struct { float scale, eps; int32_t flags; } prm{1.f / std::sqrt(float(H)), cfg_.eps, want_mix ? 1 : 2};
    if (o.dtype(part) == DType::F32) {
        g.op("ink_mix_f32p_gaudi2", {blocks, interleave128(o, n + "_pi", part, false), qi, w}, {normed, mix}, &prm, sizeof(prm));
    } else {
        g.op("ink_mix_gaudi2", {blocks, part, qi, w}, {normed, mix}, &prm, sizeof(prm));
    }
    return {normed, !want_mix ? kNoTensor : mix_natural ? interleave128(o, n + "_mo", mix, true) : mix};
}

// fused short conv (kernels/ink_conv_bf16_gaudi2.c): y BF16 [T, C]; mode 0 none, 1 L2 clip per head, 2 RMS per head
TensorId Inkling2::fused_conv(Ops& o, const std::string& n, TensorId x, TensorId state, TensorId taps, TensorId dyn,
                              uint32_t T, uint32_t C, TensorId state_out, float scale, int mode) {
    const auto tb = o.cast(taps, DType::BF16, n + "_tb");
    const auto y = o.temp(n + "_y", DType::BF16, {T, C});
    InkConvParams p{scale, cfg_.qk_clip, cfg_.eps, mode};
    if (dyn == kNoTensor) {
        o.graph().op("ink_conv_bf16_gaudi2", {x, state, tb}, {y, state_out}, &p, sizeof(p));
    } else {
        o.graph().op("ink_conv_dyn_bf16_gaudi2", {x, state, tb, o.cast(dyn, DType::F32, n + "_df")}, {y, state_out}, &p, sizeof(p));
    }
    return y;
}
// L2 clip over the innermost axis: x * min(threshold / (|x| + 1e-6), 1)  (F32 in, F32 out)
TensorId Inkling2::l2_clip(Ops& o, const std::string& n, TensorId x, float threshold) {
    Shape s = o.shape(x);
    const auto sq = o.mul(x, x, n + "_sq");
    const auto norm = o.unary("sqrt", o.reduce("sum", sq, unsigned(s.size() - 1), n + "_ss"), n + "_nrm");
    Shape one(s.size(), 1);
    const auto den = o.add(norm, o.constant(1e-6f, one, n + "_e"), n + "_den");
    const auto f = o.binary("min", o.mul(o.unary("reciprocal", den, n + "_rd"), o.constant(threshold, one, n + "_t"), n + "_q"),
                            o.constant(1.f, one, n + "_one"), n + "_f");
    return o.mul(x, f, n + "_out");
}

// Depth attention (Block Attention Residuals): stack [S, T, H] (BF16 or F32), queries q [Q, H] F32 ->
// [T, Q, H] F32 = sum_s softmax_s(n_s . q / sqrt(H)) n_s, n = unweighted RMSNorm(stack) (eps 1e-6). Exact F32 on TPC.
TensorId Inkling2::attn_res(Ops& o, const std::string& n, TensorId stack, uint32_t S, uint32_t T, TensorId q, uint32_t Q) {
    const uint64_t H = cfg_.hidden;
    const auto x = o.cast(stack, DType::F32, n + "_x");
    const auto ms = o.mul(o.reduce("sum", o.mul(x, x, n + "_sq"), 2, n + "_ss"),
                          o.constant(1.f / float(H), {1, 1, 1}, n + "_ih"), n + "_ms");
    const auto r = o.unary("rsqrt", o.add(ms, o.constant(1e-6f, {1, 1, 1}, n + "_e"), n + "_mse"), n + "_r");
    const auto nx = o.mul(x, r, n + "_n");                                                         // [S, T, H]
    const auto n4 = o.reshape(nx, {S, T, 1, H}, n + "_n4");
    const auto q4 = o.reshape(q, {1, 1, Q, H}, n + "_q4");
    const auto lg = o.reduce("sum", o.mul(n4, q4, n + "_nq"), 3, n + "_lg");                      // [S, T, Q, 1]
    const auto lgs = o.mul(o.reshape(lg, {S, T, Q}, n + "_lg3"), o.constant(1.f / std::sqrt(float(H)), {1, 1, 1}, n + "_sc"),
                           n + "_lgs");
    const auto p = o.softmax(o.transpose(lgs, {1, 2, 0}, n + "_lgt"), n + "_p");                    // [T, Q, S]
    const auto pt = o.reshape(o.transpose(p, {2, 0, 1}, n + "_pt"), {S, T, Q, 1}, n + "_p4");      // [S, T, Q, 1]
    const auto mix = o.reduce("sum", o.mul(pt, n4, n + "_pm"), 0, n + "_mix");                    // [1, T, Q, H]
    return o.reshape(mix, {T, Q, H}, n + "_out");
}

// Static short convolution (depthwise, kernel 4) with residual: y = bf16((sum_i taps[i] * xin[t + i] + x) / sqrt(5)),
// xin = [state (3 rows); x], taps F32 [4, C] in time order. The new state (last 3 rows of xin) goes to state_out.
TensorId Inkling2::short_conv(Ops& o, const std::string& n, TensorId x, TensorId state, TensorId taps, uint32_t T,
                              uint32_t /*C*/, TensorId state_out) {
    const auto xin = o.concat({state, x}, 0, n + "_xin");  // BF16 [T + 3, C]
    const auto xf = o.cast(xin, DType::F32, n + "_xf");
    const auto tp = o.split(taps, 0, {1, 1, 1, 1}, n + "_tp");
    TensorId acc = kNoTensor;
    for (uint32_t i = 0; i < 4; ++i) {
        std::vector<uint64_t> sz;
        if (i > 0) sz.push_back(i);
        sz.push_back(T);
        if (3 - i > 0) sz.push_back(3 - i);
        const auto parts = o.split(xf, 0, sz, n + "_s" + std::to_string(i));
        const auto term = o.mul(parts[i > 0 ? 1 : 0], tp[i], n + "_m" + std::to_string(i));
        acc = acc == kNoTensor ? term : o.add(acc, term, n + "_a" + std::to_string(i));
    }
    const auto y = o.mul(o.add(acc, o.cast(x, DType::F32, n + "_x32"), n + "_res"),
                         o.constant(1.f / std::sqrt(5.f), {1, 1}, n + "_k"), n + "_y");
    {
        const auto last = o.split(xin, 0, {T, 3}, n + "_st")[1];
        o.graph().op("identity", {last}, {state_out});
    }
    return o.cast(y, DType::BF16, n + "_out");
}

// Dynamic short convolution of the full layers: taps per token = static [4, C] (time order) + projected per-head taps
// dyn [T, KV * 4] (checkpoint row order head * 4 + lag, lag 0 = current token), y = bf16((sum + x) / 3).
TensorId Inkling2::dyn_conv(Ops& o, const std::string& n, TensorId x, TensorId state, TensorId taps, TensorId dyn,
                            uint32_t T, uint32_t C, TensorId state_out) {
    const uint64_t KV = cfg_.kv_heads, D = C / KV;
    const auto xin = o.concat({state, x}, 0, n + "_xin");
    const auto xf = o.cast(xin, DType::F32, n + "_xf");
    const auto d3 = o.reshape(o.cast(dyn, DType::F32, n + "_dynf"), {T, KV, 4}, n + "_d3");
    const auto lags = o.split(d3, 2, {1, 1, 1, 1}, n + "_lag");
    const auto dt = o.concat({lags[3], lags[2], lags[1], lags[0]}, 2, n + "_dt");                // time order
    const auto dtt = o.reshape(o.transpose(dt, {0, 2, 1}, n + "_dtt"), {T, 4, KV, 1}, n + "_d4");  // [T, 4, KV, 1]
    const auto st = o.reshape(taps, {1, 4, KV, D}, n + "_st4");
    const auto w = o.reshape(o.add(dtt, st, n + "_w4"), {T, 4, C}, n + "_w");                    // [T, 4, C]
    const auto wp = o.split(w, 1, {1, 1, 1, 1}, n + "_wp");
    TensorId acc = kNoTensor;
    for (uint32_t i = 0; i < 4; ++i) {
        std::vector<uint64_t> sz;
        if (i > 0) sz.push_back(i);
        sz.push_back(T);
        if (3 - i > 0) sz.push_back(3 - i);
        const auto parts = o.split(xf, 0, sz, n + "_s" + std::to_string(i));
        const auto term = o.mul(parts[i > 0 ? 1 : 0], o.reshape(wp[i], {T, C}, n + "_w2" + std::to_string(i)),
                                n + "_m" + std::to_string(i));
        acc = acc == kNoTensor ? term : o.add(acc, term, n + "_a" + std::to_string(i));
    }
    const auto y = o.mul(o.add(acc, o.cast(x, DType::F32, n + "_x32"), n + "_res"),
                         o.constant(1.f / 3.f, {1, 1}, n + "_k"), n + "_y");
    {
        const auto last = o.split(xin, 0, {T, 3}, n + "_st")[1];
        o.graph().op("identity", {last}, {state_out});
    }
    return o.cast(y, DType::BF16, n + "_out");
}

// Message gate: g = bf16(cat(rms(h), rms(m))) @ down^T (BF16), [f | mg] = tanh(g @ [forget; gain]^T) (F32),
// out = (1 - f) * h + mg * m.  h, m F32 [T, H].
TensorId Inkling2::message_gate(Ops& o, const std::string& n, TensorId h, TensorId m, uint32_t /*T*/, const std::string& which) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden, R = cfg_.gate_rank;
    const auto down = g.input("mg_" + which + "_down", DType::BF16, {R, 2 * H});
    const auto fgw = g.input("mg_" + which + "_fg", DType::BF16, {2 * H, R});
    auto urms = [&](TensorId x, const std::string& k) {
        const auto ms = o.mul(o.reduce("sum", o.mul(x, x, k + "_sq"), 1, k + "_ss"), o.constant(1.f / float(H), {1, 1}, k + "_ih"), k + "_ms");
        const auto r = o.unary("rsqrt", o.add(ms, o.constant(cfg_.eps, {1, 1}, k + "_e"), k + "_mse"), k + "_r");
        return o.mul(x, r, k + "_n");
    };
    const auto cat = o.cast(o.concat({urms(h, n + "_h"), urms(m, n + "_m")}, 1, n + "_cat"), DType::BF16, n + "_catb");
    const auto gr = o.linear_bf16(cat, down, n + "_down", DType::BF16);       // [T, R] bf16
    const auto fg = o.unary("tanh", o.linear_bf16(gr, fgw, n + "_fg", DType::F32), n + "_tanh");  // [T, 2H]
    const auto parts = o.split(fg, 1, {H, H}, n + "_fgs");
    const auto keep = o.sub(o.constant(1.f, {1, 1}, n + "_one"), parts[0], n + "_keep");
    return o.add(o.mul(keep, h, n + "_kh"), o.mul(parts[1], m, n + "_mm"), n + "_out");
}

// Routed experts (top-8 of 128 by sigmoid score + correction bias, weights renormalised x route_scale). The route
// weight multiplies the softcapped SwiGLU output before the down projection; gate/up and the activation are rounded
// to BF16; each slot's output rounded to BF16 and summed in F32. Synapse expert GEMMs run in BF16 on the exact
// dequantized weights (moe_bf16_small for T <= kBf16SmallMaxT, moe_bf16_cap for larger T).
std::pair<TensorId, TensorId> Inkling2::route(Ops& o, const std::string& n, TensorId xn, uint32_t T) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden, E = cfg_.experts, K = cfg_.topk;
    const auto router = g.input("router", DType::BF16, {E, H});
    const auto rbias = g.input("router_bias", DType::F32, {E});
    const auto logits = o.linear_bf16(xn, router, n + "_rl", DType::F32);                    // [T, E]
    if (kon(64)) {  // fused router (kernels/ink_route_gaudi2.c)
        const auto idx = o.temp(n + "_idx", DType::I32, {T, K});
        const auto rw = o.temp(n + "_w", DType::F32, {T, K});
        const float scale = cfg_.route_scale;
        g.op("ink_route_gaudi2", {logits, rbias, iota_input(g)}, {idx, rw}, &scale, sizeof(scale));
        return {idx, rw};
    }
    const auto score = o.unary("sigmoid", logits, n + "_sc");
    const auto sel = o.add(score, o.reshape(rbias, {1, E}, n + "_rb"), n + "_sel");
    const auto tk = o.topk(sel, uint32_t(K), n + "_tk");                                     // values, idx [T, K]
    const auto idx = tk.second;
    TensorId wsel = o.temp(n + "_ws", DType::F32, {T, K});
    {
        ns_GatherKernel::Params p{};
        p.axis = 0;
        g.op("gather_elements_fwd_f32", {score, idx}, {wsel}, &p, sizeof(p));
    }
    const auto wsum = o.add(o.reduce("sum", wsel, 1, n + "_wsum"), o.constant(1e-20f, {1, 1}, n + "_we"), n + "_wsd");
    const auto rw = o.mul(o.mul(wsel, o.unary("reciprocal", wsum, n + "_wr"), n + "_wn"),
                          o.constant(cfg_.route_scale, {1, 1}, n + "_rs"), n + "_rw");    // [T, K]
    return {idx, rw};
}

// Softcapped SwiGLU over the BF16 copy's w13 column order (per 256 cols: [gate 128 | up 128]):
// gu BF16 [.., 2F] -> F32 [.., F]
TensorId Inkling2::moe_act_c256(Ops& o, const std::string& k, TensorId gu) {
    const uint64_t F = cfg_.moe_inter;
    const Shape s = o.shape(gu);
    uint64_t M = 1;
    for (size_t i = 0; i + 1 < s.size(); ++i) M *= s[i];
    Shape s1(s.begin(), s.end() - 1);
    s1.push_back(F);
    const auto g4 = o.reshape(gu, {M, F / 128, 2, 128}, k + "_4");
    const auto parts = o.split(g4, 2, {1, 1}, k + "_gu");
    const auto gt = o.cast(o.reshape(parts[0], s1, k + "_gb"), DType::F32, k + "_g");
    const auto ut = o.cast(o.reshape(parts[1], s1, k + "_ub"), DType::F32, k + "_u");
    Shape one(s1.size(), 1);
    const auto gc = o.mul(o.unary("tanh", o.mul(gt, o.constant(1.f / cfg_.gate_softcap, one, k + "_ig"), k + "_g1"), k + "_gt"),
                          o.constant(cfg_.gate_softcap, one, k + "_cg"), k + "_gc");
    const auto uc = o.mul(o.unary("tanh", o.mul(ut, o.constant(1.f / cfg_.up_softcap, one, k + "_iu"), k + "_u1"), k + "_ut"),
                          o.constant(cfg_.up_softcap, one, k + "_cu"), k + "_uc");
    return o.mul(o.mul(gc, o.unary("sigmoid", gt, k + "_sg"), k + "_gs"), uc, k + "_act");
}

// the message-gate output kernel of the Synapse graphs: the lookup-table tanh (G2E_INK_MG_NOLUT=1: the LUT-free one)
static const char* mg_out_kernel() {
    static const bool nolut = std::getenv("G2E_INK_MG_NOLUT") != nullptr;
    return nolut ? "ink_mg_out_gaudi2" : "ink_mg_out_lut_gaudi2";
}

// Exact BF16 routed experts for small T (decode T = 1 and T <= kBf16SmallMaxT): gathers the R = T * K selected
// experts from the BF16 copy and runs standard BF16 batch GEMMs (F32 accumulation, per-expert factor g in F32).
TensorId Inkling2::moe_bf16_small(Ops& o, const std::string& n, TensorId xn, TensorId idx, TensorId rw, uint32_t T) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden, E = cfg_.experts, K = cfg_.topk, F = cfg_.moe_inter, R = uint64_t(T) * K;
    const auto w13 = g.input("w13_bf16", DType::BF16, {E, H, 2 * F});
    const auto g13 = g.input("w13_g", DType::F32, {E});
    const auto w2 = g.input("w2_bf16", DType::BF16, {E, F, H});
    const auto g2 = g.input("w2_g", DType::F32, {E});
    const auto ridx = o.reshape(idx, {R}, n + "_ri");
    const auto w13g = o.gather_rows(w13, ridx, n + "_w13g");
    const auto g13g = o.reshape(o.gather_rows(g13, ridx, n + "_g13g"), {R, 1, 1}, n + "_g13r");
    std::vector<TensorId> rep(K, o.reshape(xn, {T, 1, H}, n + "_x3"));
    const auto xr = o.reshape(o.concat(rep, 1, n + "_xrep"), {R, 1, H}, n + "_xr");
    const auto gu_raw = o.batch_gemm(xr, w13g, n + "_gu", false, false, DType::F32);
    const auto w2g = o.gather_rows(w2, ridx, n + "_w2g");
    const auto g2g = o.reshape(o.gather_rows(g2, ridx, n + "_g2g"), {R, 1, 1}, n + "_g2r");
    const auto gu = o.cast(o.mul(gu_raw, g13g, n + "_gus"), DType::BF16, n + "_gub");
    const auto act = o.mul(moe_act_c256(o, n + "_a", gu), o.reshape(rw, {R, 1, 1}, n + "_rw3"), n + "_ar");
    const auto actb = o.cast(act, DType::BF16, n + "_ab");
    const auto y_raw = o.batch_gemm(actb, w2g, n + "_dn", false, false, DType::F32);
    const auto yb = o.cast(o.cast(o.mul(y_raw, g2g, n + "_ys"), DType::BF16, n + "_yb"), DType::F32, n + "_yf");
    const auto sum = o.reduce("sum", o.reshape(yb, {T, K, H}, n + "_y3"), 1, n + "_sum");
    return o.cast(o.reshape(sum, {T, H}, n + "_s2d"), DType::BF16, n + "_out");
}

// Exact BF16 routed experts for T > kBf16SmallMaxT (prefill and larger tails): capacity tiers with zero overflow.
// Tier 1 covers ranks < C1 in place over all E experts; with C1 >= 4 * avg = S / 32 (capped at T), at most
// floor(S / (C1 + 1)) <= 31 experts can exceed C1, so Tier 2a (up to 16 hottest) and Tier 2b (remaining <= 15)
// cover every slot exactly with standard BF16 batch GEMMs and no overflow GEMV.
TensorId Inkling2::moe_bf16_cap(Ops& o, const std::string& n, TensorId xn, TensorId idx, TensorId rw, uint32_t T) {
    auto& g = o.graph();
    const uint64_t H = cfg_.hidden, E = cfg_.experts, K = cfg_.topk, F = cfg_.moe_inter, S = uint64_t(T) * K;
    auto r16 = [](uint64_t x) { return (x + 15) / 16 * 16; };
    const uint64_t avg = (S + E - 1) / E;
    const uint64_t C1 = std::min<uint64_t>(T, std::max<uint64_t>(16, r16(4 * avg)));
    const uint64_t EC = E * C1;
    const uint64_t max_ov = (C1 < T) ? (S / (C1 + 1)) : 0;
    const uint64_t NA = std::min<uint64_t>(16, max_ov);
    const uint64_t NB = max_ov > NA ? max_ov - NA : 0;
    const uint64_t NH = NA + NB;
    const uint64_t CA = NA ? r16(T - C1) : 0;
    const uint64_t CB = NB ? r16(S / (NA + 1) - C1) : 0;
    const auto w13 = g.input("w13_bf16", DType::BF16, {E, H, 2 * F});
    const auto g13 = g.input("w13_g", DType::F32, {E});
    const auto w2 = g.input("w2_bf16", DType::BF16, {E, F, H});
    const auto g2 = g.input("w2_g", DType::F32, {E});
    const auto one1 = o.constant(1.f, {1, 1}, n + "_one");
    const auto zero1 = o.constant(0.f, {1, 1}, n + "_zero");
    auto cf = [&](float v, const std::string& k) { return o.constant(v, {1, 1}, k); };
    auto ge = [&](TensorId a, float v, const std::string& k) {
        return o.binary("min", o.binary("max", o.add(o.sub(a, cf(v, k + "_v"), k + "_d"), one1, k + "_d1"), zero1, k + "_mx"), one1, k);
    };
    const auto ef = o.reshape(o.cast(idx, DType::F32, n + "_ef"), {S, 1}, n + "_ef2");
    const auto iota_e = iota_input(g);
    const auto iota = o.reshape(iota_e, {1, E}, n + "_iota");
    const auto onehot = o.sub(one1, o.binary("min", o.unary("abs", o.sub(ef, iota, n + "_d"), n + "_ad"), one1, n + "_md"), n + "_oh");
    const auto cum = o.temp(n + "_cum", DType::F32, {S, E});
    {
        ns_CumSumKernel::Params p{};
        p.axis = 1;
        p.exclusive = 1;
        p.reverse = 0;
        g.op("cumsum_fwd_f32", {onehot}, {cum}, &p, sizeof(p));
    }
    const auto rank = o.temp(n + "_rank", DType::F32, {S, 1});
    {
        ns_GatherKernel::Params p{};
        p.axis = 0;
        g.op("gather_elements_fwd_f32", {cum, o.reshape(idx, {S, 1}, n + "_i2")}, {rank}, &p, sizeof(p));
    }
    auto rows = [&](TensorId valid, TensorId r, uint64_t dummy, const std::string& k) {
        return o.add(o.mul(r, valid, k + "_a"), o.mul(cf(float(dummy), k + "_D"), o.sub(one1, valid, k + "_nv"), k + "_b"), k);
    };
    const auto r1_raw = o.add(o.mul(ef, cf(float(C1), n + "_C1"), n + "_eC"), rank, n + "_r1r");
    TensorId row1 = r1_raw, rowa = kNoTensor, rowb = kNoTensor;
    TensorId hidx_a = kNoTensor, hidx_b = kNoTensor;
    if (NH > 0) {
        const auto counts = o.reduce("sum", onehot, 0, n + "_cnt");
        const auto hidx = o.reshape(o.topk(counts, uint32_t(NH), n + "_hot").second, {NH}, n + "_hidx1");
        const auto inh = o.split(iota_e, 0, {NH, E - NH}, n + "_inh")[0];
        const auto hmap = o.temp(n + "_hmap", DType::F32, {E});
        {
            ns_ScatterKernel::Params p{};
            p.axis = 0;
            g.op("scatter_fwd_f32", {o.constant(-1.f, {E}, n + "_hm0"), hidx, inh}, {hmap}, &p, sizeof(p));
        }
        const auto hs = o.reshape(o.gather_rows(o.reshape(hmap, {E, 1}, n + "_hm2"), o.reshape(idx, {S}, n + "_i1"), n + "_hs"),
                                  {S, 1}, n + "_hs2");
        const auto t1 = o.sub(one1, ge(rank, float(C1), n + "_rc1"), n + "_t1");
        const auto nt1 = o.sub(one1, t1, n + "_nt1");
        const auto rmc = o.sub(rank, cf(float(C1), n + "_C1b"), n + "_rm");
        row1 = rows(t1, r1_raw, EC, n + "_row1");
        if (NB > 0) {
            const auto hb = ge(hs, float(NA), n + "_hotb");
            const auto ta = o.mul(nt1, o.sub(one1, hb, n + "_nhb"), n + "_ta");
            const auto tb = o.mul(nt1, hb, n + "_tb");
            rowa = rows(ta, o.add(o.mul(hs, cf(float(CA), n + "_CA"), n + "_hA"), rmc, n + "_rar"), NA * CA, n + "_rowa");
            rowb = rows(tb, o.add(o.mul(o.sub(hs, cf(float(NA), n + "_NA"), n + "_hsb"), cf(float(CB), n + "_CB"), n + "_hB"), rmc, n + "_rbr"),
                        NB * CB, n + "_rowb");
            const auto hgrp = o.split(hidx, 0, {NA, NB}, n + "_hg");
            hidx_a = hgrp[0];
            hidx_b = hgrp[1];
        } else {
            rowa = rows(nt1, o.add(o.mul(hs, cf(float(CA), n + "_CA"), n + "_hA"), rmc, n + "_rar"), NA * CA, n + "_rowa");
            hidx_a = hidx;
        }
    }
    const auto tokf = o.reshape(o.split(g.input("slot_tok", DType::F32, {S, 2}), 1, {1, 1}, n + "_stk")[0], {S}, n + "_tk0");
    const auto tok = o.add(o.mul(tokf, o.constant(32.f, {1}, n + "_32"), n + "_tk32"),
                           o.reshape(o.split(g.input("slot_tok2", DType::F32, {S, 2}), 1, {1, 1}, n + "_stk2")[1], {S}, n + "_tk1"), n + "_tok");
    auto scatter = [&](TensorId init, TensorId r, TensorId upd, const std::string& k) {
        const auto out = o.temp(k, DType::F32, o.shape(init));
        ns_ScatterKernel::Params p{};
        p.axis = 0;
        g.op("scatter_fwd_f32", {init, o.reshape(o.cast(r, DType::I32, k + "_ri"), {S}, k + "_r1"), upd}, {out}, &p, sizeof(p));
        return out;
    };
    const auto rwf = o.reshape(rw, {S}, n + "_rwf");
    const auto xz = o.concat({xn, o.constant(0.f, {1, H}, n + "_zr", DType::BF16)}, 0, n + "_xz");
    struct TierA { TensorId gu_raw, bw; uint64_t B, Cb; };
    auto tier_a = [&](TensorId r, uint64_t B, uint64_t Cb, TensorId wa, const std::string& k) {
        const uint64_t R = B * Cb;
        const auto btok = scatter(o.constant(float(T), {R + 1}, k + "_bt0"), r, tok, k + "_btok");
        const auto brw = scatter(o.constant(0.f, {R + 1}, k + "_bw0"), r, rwf, k + "_brw");
        const auto bt = o.split(o.cast(btok, DType::I32, k + "_bti"), 0, {R, 1}, k + "_bts")[0];
        const auto bw = o.reshape(o.split(brw, 0, {R, 1}, k + "_bws")[0], {B, Cb, 1}, k + "_bw3");
        const auto xe = o.reshape(o.gather_rows(xz, bt, k + "_xe"), {B, Cb, H}, k + "_xe3");
        const auto gu_raw = o.batch_gemm(xe, wa, k + "_gu", false, false, DType::F32);
        return TierA{gu_raw, bw, B, Cb};
    };
    auto tier_b = [&](const TierA& ta, TensorId ga, const std::string& k) {
        const auto gu = o.cast(o.mul(ta.gu_raw, o.reshape(ga, {ta.B, 1, 1}, k + "_gar"), k + "_gus"), DType::BF16, k + "_gub");
        const auto act = o.mul(moe_act_c256(o, k + "_a", gu), ta.bw, k + "_ar");
        return o.cast(act, DType::BF16, k + "_ab");
    };
    auto tier_c = [&](TensorId actb, TensorId wb, TensorId gb, uint64_t B, uint64_t Cb, const std::string& k) {
        const uint64_t R = B * Cb;
        const auto y_raw = o.batch_gemm(actb, wb, k + "_dn", false, false, DType::F32);
        const auto y = o.cast(o.mul(y_raw, o.reshape(gb, {B, 1, 1}, k + "_gbr"), k + "_ys"), DType::BF16, k + "_yb");
        return o.concat({o.reshape(y, {R, H}, k + "_y2"), o.constant(0.f, {1, H}, k + "_yz", DType::BF16)}, 0, k + "_yd");
    };
    const auto ta1 = tier_a(row1, E, C1, w13, n + "_e1");
    TensorId g13a = kNoTensor, g13b = kNoTensor;
    TierA ta2{}, ta3{};
    if (NA > 0) {
        const auto w13a = o.gather_rows(w13, hidx_a, n + "_w13a");
        g13a = o.gather_rows(g13, hidx_a, n + "_g13a");
        ta2 = tier_a(rowa, NA, CA, w13a, n + "_e2");
    }
    if (NB > 0) {
        const auto w13b = o.gather_rows(w13, hidx_b, n + "_w13b");
        g13b = o.gather_rows(g13, hidx_b, n + "_g13b");
        ta3 = tier_a(rowb, NB, CB, w13b, n + "_e3");
    }
    const auto ab1 = tier_b(ta1, g13, n + "_e1");
    TensorId w2a = kNoTensor, w2b = kNoTensor, g2a = kNoTensor, g2b = kNoTensor;
    if (NA > 0) {
        w2a = o.gather_rows(w2, hidx_a, n + "_w2a");
        g2a = o.gather_rows(g2, hidx_a, n + "_g2a");
    }
    if (NB > 0) {
        w2b = o.gather_rows(w2, hidx_b, n + "_w2b");
        g2b = o.gather_rows(g2, hidx_b, n + "_g2b");
    }
    const auto ab2 = NA > 0 ? tier_b(ta2, g13a, n + "_e2") : kNoTensor;
    const auto ab3 = NB > 0 ? tier_b(ta3, g13b, n + "_e3") : kNoTensor;
    const auto ye1 = tier_c(ab1, w2, g2, E, C1, n + "_e1");
    const auto ye2 = NA > 0 ? tier_c(ab2, w2a, g2a, NA, CA, n + "_e2") : kNoTensor;
    const auto ye3 = NB > 0 ? tier_c(ab3, w2b, g2b, NB, CB, n + "_e3") : kNoTensor;
    TensorId ys = o.cast(o.gather_rows(ye1, o.reshape(o.cast(row1, DType::I32, n + "_r1i"), {S}, n + "_r1f"), n + "_y1"), DType::F32, n + "_y1f");
    if (NA > 0) {
        const auto y2 = o.cast(o.gather_rows(ye2, o.reshape(o.cast(rowa, DType::I32, n + "_rai"), {S}, n + "_raf"), n + "_y2"), DType::F32, n + "_y2f");
        ys = o.add(ys, y2, n + "_y12");
    }
    if (NB > 0) {
        const auto y3 = o.cast(o.gather_rows(ye3, o.reshape(o.cast(rowb, DType::I32, n + "_rbi"), {S}, n + "_rbf"), n + "_y3"), DType::F32, n + "_y3f");
        ys = o.add(ys, y3, n + "_y123");
    }
    const auto sum = o.reduce("sum", o.reshape(ys, {T, K, H}, n + "_ys3"), 1, n + "_sum");
    return o.cast(o.reshape(sum, {T, H}, n + "_s2d"), DType::BF16, n + "_out");
}

TensorId Inkling2::iota_input(GraphBuilder& g) {  // F32 [E] = 0..E-1, declared once per graph
    auto it = declared_.find("iota_e");
    if (it != declared_.end()) return it->second;
    return declared_["iota_e"] = g.input("iota_e", DType::F32, {cfg_.experts});
}


// Native expert layers (pack v6, NVFP4 and MXFP8) have one output factor per expert g[e] (amax / den * 512 for NVFP4;
// 2^(s_max - 127) for MXFP8), where legacy FP8 layers had a scale per output row. The kernels and plans take row scales
// as [rows, E] F32 vectors, so every row of expert e gets g[e]: one vector per matrix serves all layouts.
void Inkling2::prepare_nv_gvec() {
    const uint64_t E = cfg_.experts;
    std::vector<float> host;
    const auto fill = [&](uint32_t L, const char* part, uint64_t rows) {
        const auto raw = w_.read(Lw(L, part));
        const uint64_t off = host.size();
        host.resize(off + E * rows);
        for (uint64_t e = 0; e < E; ++e) {
            float g;
            std::memcpy(&g, raw.data() + e * 4, 4);
            std::fill_n(host.begin() + off + e * rows, rows, g);
        }
        return off * 4;
    };
    for (uint32_t L = 0; L < cfg_.layers; ++L) {
        nv_gvec13_off_[L] = fill(L, "w13.g", 2ull * cfg_.moe_inter);
        nv_gvec2_off_[L] = fill(L, "w2.g", cfg_.hidden);
    }
    if (host.empty()) return;
    const uint64_t bytes = host.size() * 4;
    nv_gvec_ = DeviceBuffer(dev_, bytes);
    HostBuffer hb(dev_, bytes);
    std::memcpy(hb.as<uint8_t>(), host.data(), bytes);
    stream_.upload(hb, 0, nv_gvec_, 0, bytes);
    stream_.synchronize();
    direct_prev_ = false;
}

// The depth-attention queries (attn_q, mlp_q, msg_q; F32) in the interleaved order the fused mix kernel reads (every
// 128 values: [even 64 | odd 64]), once at load instead of a transpose in every launch.
void Inkling2::prepare_queries() {
    std::vector<std::string> names = {"msg_q", "final_q"};
    for (uint32_t L = 0; L < cfg_.layers; ++L) {
        if (L > 0) names.push_back(Lw(L, "attn_q"));
        names.push_back(Lw(L, "mlp_q"));
    }
    std::vector<uint8_t> host;
    for (const auto& nm : names) {
        if (!w_.has(nm)) continue;
        auto raw = w_.read(nm);
        const size_t n = raw.size() / 4;
        std::vector<float> in(n), out(n);
        std::memcpy(in.data(), raw.data(), n * 4);
        for (size_t b = 0; b + 128 <= n; b += 128)
            for (size_t j = 0; j < 64; ++j) {
                out[b + j] = in[b + 2 * j];
                out[b + 64 + j] = in[b + 2 * j + 1];
            }
        host.resize(align128(host.size()));
        qil_off_[nm] = host.size();
        host.insert(host.end(), reinterpret_cast<uint8_t*>(out.data()), reinterpret_cast<uint8_t*>(out.data()) + n * 4);
    }
    host.resize((host.size() + 8191) & ~size_t(8191));
    qil_ = DeviceBuffer(dev_, host.size());
    HostBuffer hb(dev_, host.size());
    std::memcpy(hb.as<uint8_t>(), host.data(), host.size());
    stream_.upload(hb, 0, qil_, 0, host.size());
    stream_.synchronize();
    direct_prev_ = false;
}

// ------------------------------------------------------------------------------------------------ recipes
Recipe& Inkling2::embed_recipe(uint32_t T) {
    const std::string key = "ink_embed_T" + std::to_string(T);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden;
    GraphBuilder g(dev_);
    Ops o(g);
    const auto ids = g.input("ids", DType::I32, {T});
    const auto table = g.input("embed", DType::BF16, {cfg_.embed_rows, H});
    const auto w = g.input("embed_norm", DType::BF16, {H});
    const auto x = o.cast(o.gather_rows(table, ids, "emb"), DType::F32, "embf");
    const auto y = o.rms_norm(x, o.cast(w, DType::F32, "wf"), cfg_.eps, false, "en");
    output(g, "out", o.cast(y, DType::BF16, "enb"));
    return *(recipes_[key] = g.compile(key));
}

void Inkling2::set_aux_capture(bool on) {
    aux_on_ = on;
    if (on && !aux_.address()) aux_ = DeviceBuffer(dev_, 4ull * kMaxT * cfg_.hidden * 2);
    if (on && blocks1_.address() && !aux1_.address()) aux1_ = DeviceBuffer(dev_, 4ull * kMaxT * cfg_.hidden * 2);
}

// verification history of the four conv inputs per application: [3 + T rows, C] each (k, v, attn, mlp)
uint64_t Inkling2::hist_off(int conv, uint32_t T) const {
    const uint64_t KVD = uint64_t(cfg_.kv_heads) * cfg_.head_dim, H = cfg_.hidden, R = 3 + T;
    const uint64_t off[4] = {0, R * KVD * 2, 2 * R * KVD * 2, 2 * R * KVD * 2 + R * H * 2};
    return off[conv];
}

// the final depth attention + norm of every row (the DSpark tap 48) without the LM head
Recipe& Inkling2::headx_recipe(uint32_t rows) {
    const std::string key = "ink_headx_R" + std::to_string(rows);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden;
    GraphBuilder g(dev_);
    Ops o(g);
    std::vector<TensorId> st;
    for (int k = 0; k < 4; ++k) st.push_back(o.reshape(g.input("b" + std::to_string(k), DType::BF16, {rows, H}), {1, rows, H}, "b3" + std::to_string(k)));
    st.push_back(o.reshape(g.input("h", DType::BF16, {rows, H}), {1, rows, H}, "h3"));
    const auto q = g.input("final_q", DType::F32, {1, H});
    const auto mix = o.reshape(attn_res(o, "fin", o.concat(st, 0, "stack"), 5, rows, q, 1), {rows, H}, "mix");
    const auto w = g.input("final_norm", DType::BF16, {H});
    output(g, "xn", o.cast(o.rms_norm(mix, o.cast(w, DType::F32, "nwf"), cfg_.eps, false, "fn"), DType::BF16, "fnb"));
    return *(recipes_[key] = g.compile(key));
}

void Inkling2::enqueue_commit(uint32_t T) {
    const uint64_t KVD = uint64_t(cfg_.kv_heads) * cfg_.head_dim;
    const uint64_t ro = ring_off(active_slot_);
    std::vector<Binding> cb = {{"rows_ring", &tables_, tab_.rows}, {"rows_lin", &tables_, tab_.rows + kMaxT * 4}};
    for (const auto& a : apps_) {
        const auto ai = std::to_string(a.index);
        cb.push_back({"kv" + ai, &kvscratch_, uint64_t(a.index) * 2 * KVD * kMaxT * 2});
        cb.push_back({"kr" + ai, &kring_[ring_index_[a.index]], ro});
        cb.push_back({"vr" + ai, &vring_[ring_index_[a.index]], ro});
        if (cfg_.full[a.layer]) {
            cb.push_back({"kc" + ai, &kcache_[kv_index_[a.index]], 0});
            cb.push_back({"vc" + ai, &vcache_[kv_index_[a.index]], 0});
        }
    }
    commit_recipe(T).enqueue(stream_, cb);
}

std::vector<float> Inkling2::verify(std::span<const int32_t> tokens) {
    if (tokens.empty() || tokens.size() > 8) throw std::invalid_argument("inkling2: verify takes 1..8 tokens");
    ensure_hist();
    std::vector<float> logits;
    verify_ = true;
    try {
        run_forward(tokens, true, &logits);
    } catch (...) {
        verify_ = false;
        throw;
    }
    verify_ = false;
    return logits;
}

// the conv input history of every application (verification and padded prefill tails: up to kMaxT rows)
void Inkling2::ensure_hist() {
    if (hist_.address()) return;
    const uint64_t KVD = uint64_t(cfg_.kv_heads) * cfg_.head_dim, R = 3 + kMaxT;
    hist_app_ = align128(2 * R * KVD * 2 + 2 * R * uint64_t(cfg_.hidden) * 2);
    hist_ = DeviceBuffer(dev_, hist_app_ * apps_.size());
}

bool Inkling2::can_verify(uint32_t T) const {
    if (!direct_ || !direct_on_) return false;
    for (uint32_t i = 0; i < apps_.size(); ++i)
        if (!direct_->handles_v(i, T)) return false;
    return true;
}

void Inkling2::accept(uint32_t n) {
    const uint32_t T = ver_T_;
    if (!T || n > T) throw std::invalid_argument("inkling2: accept without a verification");
    static const bool dvcommit = [] { const char* e = std::getenv("G2E_INK_DIRECT_VCOMMIT"); return !e || *e != '0'; }();
    if (dvcommit && direct_ && direct_on_ && T <= 8 && direct_->handles_v_commit()) {
        if (!direct_prev_) stream_.synchronize();
        direct_->commit_and_rollback(1 - ver_side_, T, n);
        conv_side_ = 1 - ver_side_;
        pos_ += n;
        ver_T_ = 0;
        direct_prev_ = false;
        return;
    }
    const uint64_t KVD = uint64_t(cfg_.kv_heads) * cfg_.head_dim, H = cfg_.hidden, W = cfg_.window;
    stream_.synchronize();  // the staging buffer is rewritten below
    // commit rows 0 .. n - 1 only: the rest would overwrite ring slots still in later windows
    auto* rr = staging_.as<int32_t>();
    for (uint32_t r = 0; r < T; ++r) {
        rr[r] = r < n ? int32_t((pos_ + r) % W) : -1;
        rr[kMaxT + r] = r < n ? int32_t(pos_ + r) : -1;
    }
    stream_.upload(staging_, 0, tables_, tab_.rows, T * 4);
    stream_.upload(staging_, kMaxT * 4, tables_, tab_.rows + kMaxT * 4, T * 4);
    enqueue_commit(T);
    // conv states after row n - 1: history rows n .. n + 2 (one Direct launch, else stream copies)
    if (direct_ && direct_on_ && T <= 8) {
        stream_.synchronize();
        direct_->rollback(1 - ver_side_, T, n);
    } else {
        const uint64_t cbase = conv_off_slot(active_slot_, 1 - ver_side_);
        for (const auto& a : apps_) {
            const uint64_t ca = uint64_t(a.index) * conv_app_bytes_, ha = uint64_t(a.index) * hist_app_;
            for (int i = 0; i < 4; ++i) {
                const uint64_t C = i < 2 ? KVD : H;
                stream_.copy(hist_, ha + hist_off(i, T) + uint64_t(n) * C * 2, conv_pool_, cbase + ca + conv_off_[i], 3 * C * 2);
            }
        }
        stream_.synchronize();
    }
    conv_side_ = 1 - ver_side_;
    pos_ += n;
    ver_T_ = 0;
    direct_prev_ = false;
}

Recipe& Inkling2::head_recipe(uint32_t rows, bool with_xn) {
    const std::string key = "ink_head_R" + std::to_string(rows) + (with_xn ? "_xn" : "");
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t H = cfg_.hidden, V = cfg_.vocab;
    GraphBuilder g(dev_);
    Ops o(g);
    std::vector<TensorId> st;
    for (int k = 0; k < 4; ++k) st.push_back(o.reshape(g.input("b" + std::to_string(k), DType::BF16, {rows, H}), {1, rows, H}, "b3" + std::to_string(k)));
    st.push_back(o.reshape(g.input("h", DType::BF16, {rows, H}), {1, rows, H}, "h3"));
    const auto q = g.input("final_q", DType::F32, {1, H});
    const auto mix = o.reshape(attn_res(o, "fin", o.concat(st, 0, "stack"), 5, rows, q, 1), {rows, H}, "mix");
    const auto w = g.input("final_norm", DType::BF16, {H});
    const auto xn = o.cast(o.rms_norm(mix, o.cast(w, DType::F32, "nwf"), cfg_.eps, false, "fn"), DType::BF16, "fnb");
    if (with_xn) output(g, "xn", xn);
    const auto lm = g.input("lm_head", DType::BF16, {V, H});
    const auto lg = o.linear_bf16(xn, lm, "lg", DType::F32);
    const auto cap = o.mul(o.unary("tanh", o.mul(lg, o.constant(1.f / cfg_.logit_softcap, {1, 1}, "ic"), "l1"), "lt"),
                           o.constant(cfg_.logit_softcap, {1, 1}, "c"), "lc");
    output(g, "logits", cap);
    return *(recipes_[key] = g.compile(key));
}

Recipe& Inkling2::layer_recipe(const App& a, uint32_t T, uint32_t Sf) {
    const uint32_t L = a.layer;
    const bool full = cfg_.full[L], boundary = L % cfg_.attn_res_block == 0, msg = a.pass > 0;
    const uint32_t nbb = L / cfg_.attn_res_block + 1;
    std::string key = std::string("ink_") + (full ? "F" : "S") + "_n" + std::to_string(nbb) + (boundary ? "b" : "") +
                      (L == 0 ? "z" : "") + (msg ? "m" : "") + "_bf16_T" + std::to_string(T);
    if (full) key += "_f" + std::to_string(Sf);
    if (full && attn_block(T, Sf) < T) key += "_qb" + std::to_string(attn_block(T, Sf));
    if (ink_ref()) key += "_ref";
    if (msg && kon(32) && kon(128) && msg_stack_) key += "_fmg";
    if (std::getenv("G2E_INK_KMASK")) key += std::string("_k") + std::getenv("G2E_INK_KMASK");
    if (std::getenv("G2E_INK_DBGPOST")) key += "_dbgpost";
    if (std::getenv("G2E_INK_MG_NOLUT")) key += "_mgnl";
    if (verify_) key += "_vh";  // conv state outputs = the whole input history (accept() picks the state)
    static const bool dbgout = std::getenv("G2E_INK_DBGOUT") != nullptr;  // intermediate outputs (G2E_INK_DPCHECK)
    if (dbgout) key += "_dbgo";
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;

    const uint64_t H = cfg_.hidden, NH = cfg_.heads, KV = cfg_.kv_heads, D = cfg_.head_dim, G = NH / KV;
    const uint64_t W = cfg_.window, DR = cfg_.d_rel, RE = cfg_.rel_extent, KVD = KV * D;
    const uint64_t QS = NH * D * 2 + NH * DR + NH;  // q-side rows: q, gate, rel, sink
    const uint64_t KR = full ? KVD + KV * 4 : KVD;  // k rows (+ dynamic taps)
    GraphBuilder g(dev_);
    Ops o(g);
    const auto blocks = g.input("blocks", DType::BF16, {nbb, T, H});
    TensorId h_in;
    if (boundary) {
        h_in = nbb > 1 ? o.reshape(o.split(blocks, 0, {nbb - 1, 1}, "bsplit")[1], {T, H}, "hin")
                       : o.reshape(blocks, {T, H}, "hin");
    } else {
        h_in = g.input("h_in", DType::BF16, {T, H});
    }
    declared_.clear();
    auto win = [&](const std::string& nm) {  // a BF16 [H] weight input, declared once per graph
        auto it = declared_.find(nm);
        if (it != declared_.end()) return it->second;
        const auto t = g.input(nm, DType::BF16, {H});
        declared_[nm] = t;
        return t;
    };
    // ---- sublayer inputs (F32)
    TensorId xq = kNoTensor, xk = kNoTensor, xv = kNoTensor;
    TensorId fq = kNoTensor, fk = kNoTensor, fv = kNoTensor;  // fused path: normalized BF16 inputs (no messages)
    const bool fmix = kon(32);
    const bool fmg = fmix && kon(128) && msg_stack_;  // fused message gates (interleaved F32 states)
    TensorId xil = kNoTensor;  // the attention-input mix, interleaved F32 [T, Qa, H] (fused message gates)
    if (fmix && L > 0) {
        TensorId blk = blocks, part = h_in;
        if (boundary) blk = o.split(blocks, 0, {nbb - 1, 1}, "absplit")[0];
        const uint32_t Qa = full ? 3 : 1;
        const auto q = g.input("attn_q", DType::F32, {Qa, H});
        TensorId wn;
        if (full) wn = o.concat({o.reshape(win("q_norm"), {1, H}, "qn1"), o.reshape(win("k_norm"), {1, H}, "kn1"),
                                 o.reshape(win("v_norm"), {1, H}, "vn1")}, 0, "qkvn");
        else wn = o.reshape(win("attn_norm"), {1, H}, "an1");
        const auto mr = fused_mix(o, "ar", blk, part, q, wn, T, Qa, msg, !fmg);
        if (msg && fmg) {
            xil = mr.second;
        } else if (msg) {
            if (full) {
                const auto p = o.split(mr.second, 1, {1, 1, 1}, "arq");
                xq = o.reshape(p[0], {T, H}, "xq");
                xk = o.reshape(p[1], {T, H}, "xk");
                xv = o.reshape(p[2], {T, H}, "xv");
            } else {
                xq = xk = xv = o.reshape(mr.second, {T, H}, "x");
            }
        } else if (full) {
            const auto p = o.split(mr.first, 1, {1, 1, 1}, "arn");
            fq = o.reshape(p[0], {T, H}, "fq");
            fk = o.reshape(p[1], {T, H}, "fk");
            fv = o.reshape(p[2], {T, H}, "fv");
        } else {
            fq = fk = fv = o.reshape(mr.first, {T, H}, "fq");
        }
    } else if (L == 0) {
        xq = xk = xv = o.cast(h_in, DType::F32, "x0");
    } else {
        const auto stack = boundary ? blocks : o.concat({blocks, o.reshape(h_in, {1, T, H}, "h3")}, 0, "astk");
        const uint32_t Sa = boundary ? nbb : nbb + 1, Qa = full ? 3 : 1;
        const auto q = g.input("attn_q", DType::F32, {Qa, H});
        const auto mix = attn_res(o, "ar", stack, Sa, T, q, Qa);  // [T, Qa, H]
        if (full) {
            const auto p = o.split(mix, 1, {1, 1, 1}, "arq");
            xq = o.reshape(p[0], {T, H}, "xq");
            xk = o.reshape(p[1], {T, H}, "xk");
            xv = o.reshape(p[2], {T, H}, "xv");
        } else {
            xq = xk = xv = o.reshape(mix, {T, H}, "x");
        }
    }
    TensorId mm = kNoTensor, msg_il = kNoTensor;
    if (msg) {
        const auto p1 = o.reshape(g.input("prev1", DType::BF16, {T, H}), {1, T, H}, "p1");
        const auto p2 = o.reshape(g.input("prev2", DType::BF16, {T, H}), {1, T, H}, "p2");
        const auto mq = g.input("msg_q", DType::F32, {4, H});
        if (fmg) msg_il = fused_mix(o, "msg", p1, o.reshape(p2, {T, H}, "p2d"), mq, o.constant(1.f, {4, H}, "msgw", DType::BF16),
                                    T, 4, true, false).second;  // interleaved [T, 4, H]
        const auto ms = fmg ? kNoTensor
                      : fmix ? fused_mix(o, "msg", p1, o.reshape(p2, {T, H}, "p2d"), mq,
                                         o.constant(1.f, {4, H}, "msgw", DType::BF16), T, 4, true).second
                             : attn_res(o, "msg", o.concat({p1, p2}, 0, "pstk"), 2, T, mq, 4);  // [T, 4, H]
        if (!fmg) {
        const auto mp = o.split(ms, 1, {1, 1, 1, 1}, "msp");
        const auto qm = o.reshape(mp[0], {T, H}, "qm"), km = o.reshape(mp[1], {T, H}, "km");
        const auto vm = o.reshape(mp[2], {T, H}, "vm");
        mm = o.reshape(mp[3], {T, H}, "mm");
        const auto x0 = xq;
        xq = message_gate(o, "mgq", x0, qm, T, "q");
        xk = message_gate(o, "mgk", full ? xk : x0, km, T, "k");
        xv = message_gate(o, "mgv", full ? xv : x0, vm, T, "v");
        } else {
            // q / k / v message gates at once: CAT kernel -> 2 batched GEMMs -> OUT kernel (gate + the q/k/v norms)
            const uint32_t Gh = full ? 3 : 1;
            const float prm_eps = cfg_.eps;
            struct { float eps; int32_t m0; } prm{prm_eps, 0};
            const auto cat = o.temp("mgcat", DType::BF16, {3, T, 2 * H});
            g.op("ink_mg_gaudi2", {xil, msg_il}, {cat}, &prm, sizeof(prm));
            const auto gr = o.batch_gemm(cat, g.input("mgd3", DType::BF16, {3, cfg_.gate_rank, 2 * H}), "mgg", false, true, DType::BF16);
            const auto fgv = o.batch_gemm(gr, g.input("mgf3", DType::BF16, {3, 2 * H, cfg_.gate_rank}), "mgfg", false, true, DType::F32);
            TensorId wn3;
            if (full) wn3 = o.concat({o.reshape(win("q_norm"), {1, H}, "mq1"), o.reshape(win("k_norm"), {1, H}, "mk1"),
                                      o.reshape(win("v_norm"), {1, H}, "mv1")}, 0, "mqkv");
            else { const auto a1 = o.reshape(win("attn_norm"), {1, H}, "ma1"); wn3 = o.concat({a1, a1, a1}, 0, "maaa"); }
            const auto nq = o.temp("mgnorm", DType::BF16, {3, T, H});
            g.op(mg_out_kernel(), {xil, msg_il, fgv, wn3}, {nq}, &prm, sizeof(prm));
            const auto pp = o.split(nq, 0, {1, 1, 1}, "mgsp");
            fq = o.reshape(pp[0], {T, H}, "fq");
            fk = o.reshape(pp[1], {T, H}, "fk");
            fv = o.reshape(pp[2], {T, H}, "fv");
            (void)Gh;
        }
    }
    std::map<std::string, TensorId> norm_w;  // a weight input is declared once per graph
    auto norm = [&](TensorId x, const std::string& wname, const std::string& n) {
        auto it = norm_w.find(wname);
        if (it == norm_w.end()) {
            TensorId wt = kNoTensor;
            auto d = declared_.find(wname);
            wt = d != declared_.end() ? d->second : g.input(wname, DType::BF16, {H});
            it = norm_w.emplace(wname, o.cast(wt, DType::F32, wname + "_f32")).first;
        }
        return o.cast(o.rms_norm(x, it->second, cfg_.eps, false, n), DType::BF16, n + "_b");
    };
    TensorId qh, kh, vh;
    if (fq != kNoTensor) {
        qh = fq; kh = fk; vh = fv;
    } else if (full) {
        qh = norm(xq, "q_norm", "qn");
        kh = norm(xk, "k_norm", "kn");
        vh = norm(xv, "v_norm", "vn");
    } else if (msg) {
        qh = norm(xq, "attn_norm", "qn");
        kh = norm(xk, "attn_norm", "kn");
        vh = norm(xv, "attn_norm", "vn");
    } else {
        qh = kh = vh = norm(xq, "attn_norm", "an");
    }
    // ---- projections (BF16)
    const auto wq = g.input("wq", DType::BF16, {QS, H});
    const auto wk = g.input("wk", DType::BF16, {KR, H});
    const auto wv = g.input("wv", DType::BF16, {KR, H});
    const auto qside = o.linear_bf16(qh, wq, "pq");
    const auto kside = o.linear_bf16(kh, wk, "pk");
    const auto vside = o.linear_bf16(vh, wv, "pv");
    const auto qp = o.split(qside, 1, {NH * D, NH * D, NH * DR, NH}, "qsp");
    TensorId tau = kNoTensor;
    if (full) tau = g.input("tau", DType::F32, {T, 1, 1});
    // q: * scale, L2 clip -> bf16 (full: * tau -> bf16)
    TensorId q;
    if (kon(1)) {
        const auto qsb = o.cast(g.input("q_scale", DType::F32, {NH, D}), DType::BF16, "qsb");
        q = o.temp("qk", DType::BF16, {T, NH, D});
        const float thr = cfg_.qk_clip;
        if (full) g.op("ink_qprep_tau_bf16_gaudi2", {qside, qsb, o.reshape(tau, {T}, "tau1")}, {q}, &thr, sizeof(thr));
        else g.op("ink_qprep_bf16_gaudi2", {qside, qsb}, {q}, &thr, sizeof(thr));
    } else {
        const auto qs = g.input("q_scale", DType::F32, {NH, D});
        auto x = o.mul(o.cast(o.reshape(qp[0], {T, NH, D}, "q3"), DType::F32, "qf"), o.reshape(qs, {1, NH, D}, "qs3"), "qsc");
        q = o.cast(l2_clip(o, "qc", x, cfg_.qk_clip), DType::BF16, "qb");
        if (full) q = o.cast(o.mul(o.cast(q, DType::F32, "qbf"), tau, "qtau"), DType::BF16, "qtb");
    }
    TensorId r;
    {
        const auto rs = g.input("r_scale", DType::F32, {NH, DR});
        auto x = o.mul(o.cast(o.reshape(qp[2], {T, NH, DR}, "r3"), DType::F32, "rf"), o.reshape(rs, {1, NH, DR}, "rs3"), "rsc");
        r = o.cast(l2_clip(o, "rc", x, cfg_.rel_clip), DType::BF16, "rb");
        if (full) r = o.cast(o.mul(o.cast(r, DType::F32, "rbf"), tau, "rtau"), DType::BF16, "rtb");
    }
    const auto gate = qp[1];                                            // [T, NH*D] bf16
    const auto sink = o.cast(qp[3], DType::F32, "sinkf");              // [T, NH]
    // k / v with their short convs
    const auto csk = g.input("cs_k", DType::BF16, {3, KVD});
    const auto csv = g.input("cs_v", DType::BF16, {3, KVD});
    const auto csk_o = g.input("cs_k_o", DType::BF16, {verify_ ? 3 + T : 3, KVD});
    const auto csv_o = g.input("cs_v_o", DType::BF16, {verify_ ? 3 + T : 3, KVD});
    const auto ktaps = g.input("k_taps", DType::F32, {4, KVD});
    const auto vtaps = g.input("v_taps", DType::F32, {4, KVD});
    TensorId kc, vc, knew, vnew;
    if (kon(2) && (full ? kon(16) : kon(8))) {
        TensorId kx = kside, vx = vside, kd = kNoTensor, vd = kNoTensor;
        if (full) {
            const auto kp = o.split(kside, 1, {KVD, KV * 4}, "ksp");
            const auto vp = o.split(vside, 1, {KVD, KV * 4}, "vsp");
            kx = kp[0]; kd = kp[1]; vx = vp[0]; vd = vp[1];
        }
        const float sc = full ? 1.f / 3.f : 1.f / std::sqrt(5.f);
        static const bool dbg_post = std::getenv("G2E_INK_DBGPOST") != nullptr;
        if (dbg_post) {  // debug: the kernel without its post-op, clip / RMS as graph ops
            const auto kc0 = fused_conv(o, "kcv", kx, csk, ktaps, kd, T, uint32_t(KVD), csk_o, sc, 0);
            const auto vc0 = fused_conv(o, "vcv", vx, csv, vtaps, vd, T, uint32_t(KVD), csv_o, sc, 0);
            knew = o.cast(l2_clip(o, "kcl", o.cast(o.reshape(kc0, {T, KV, D}, "k3"), DType::F32, "kf"), cfg_.qk_clip), DType::BF16, "kb");
            const auto vf = o.cast(o.reshape(vc0, {T, KV, D}, "v3"), DType::F32, "vf");
            const auto ms = o.mul(o.reduce("sum", o.mul(vf, vf, "vsq"), 2, "vss"), o.constant(1.f / float(D), {1, 1, 1}, "vid"), "vms");
            const auto rr = o.unary("rsqrt", o.add(ms, o.constant(cfg_.eps, {1, 1, 1}, "ve"), "vmse"), "vr");
            vnew = o.cast(o.mul(vf, rr, "vnorm"), DType::BF16, "vb");
        } else {
        knew = o.reshape(fused_conv(o, "kcv", kx, csk, ktaps, kd, T, uint32_t(KVD), csk_o, sc, 1), {T, KV, D}, "kb");
        vnew = o.reshape(fused_conv(o, "vcv", vx, csv, vtaps, vd, T, uint32_t(KVD), csv_o, sc, 2), {T, KV, D}, "vb");
        }
    } else {
    if (full) {
        const auto kp = o.split(kside, 1, {KVD, KV * 4}, "ksp");
        const auto vp = o.split(vside, 1, {KVD, KV * 4}, "vsp");
        kc = dyn_conv(o, "kcv", kp[0], csk, ktaps, kp[1], T, uint32_t(KVD), csk_o);
        vc = dyn_conv(o, "vcv", vp[0], csv, vtaps, vp[1], T, uint32_t(KVD), csv_o);
    } else {
        kc = short_conv(o, "kcv", kside, csk, ktaps, T, uint32_t(KVD), csk_o);
        vc = short_conv(o, "vcv", vside, csv, vtaps, T, uint32_t(KVD), csv_o);
    }
    knew = o.cast(l2_clip(o, "kcl", o.cast(o.reshape(kc, {T, KV, D}, "k3"), DType::F32, "kf"), cfg_.qk_clip),
                  DType::BF16, "kb");                                              // [T, KV, D]
    {
        const auto vf = o.cast(o.reshape(vc, {T, KV, D}, "v3"), DType::F32, "vf");
        const auto ms = o.mul(o.reduce("sum", o.mul(vf, vf, "vsq"), 2, "vss"), o.constant(1.f / float(D), {1, 1, 1}, "vid"), "vms");
        const auto rr = o.unary("rsqrt", o.add(ms, o.constant(cfg_.eps, {1, 1, 1}, "ve"), "vmse"), "vr");
        vnew = o.cast(o.mul(vf, rr, "vnorm"), DType::BF16, "vb");
    }
    }
    // head-major new rows [KV, T, D]
    const auto k8 = o.transpose(knew, {1, 0, 2}, "k8");
    const auto v8 = o.transpose(vnew, {1, 0, 2}, "v8");
    const uint64_t SN = W + T;  // near columns: the W ring rows + the T new rows
    const auto proj = g.input("rel_proj", DType::BF16, {DR, RE});
    const auto projT = o.transpose(proj, {1, 0}, "projT");                                      // [RE, DR]
    const auto idx_all = g.input(full ? "gn_idx" : "swa_idx", DType::I32, {T * SN});
    const auto valid_all = full ? g.input("gn_valid", DType::F32, {T, SN}) : kNoTensor;
    const auto mask_all = g.input(full ? "gn_mask" : "swa_mask", DType::F32, {T, SN});
    const auto scale = o.constant(1.f / std::sqrt(float(D)), {1, 1, 1}, "scl");
    const auto kn_c = g.input("k_ring", DType::BF16, {KV, W, D});  // near keys: the ring of the last W positions
    const auto vn_c = g.input("v_ring", DType::BF16, {KV, W, D});
    std::vector<TensorId> kf, vf;
    TensorId fmask = kNoTensor;
    if (full && Sf > 0) {
        fmask = g.input("gf_mask", DType::F32, {1, Sf});
        for (uint64_t h = 0; h < KV; ++h) {
            kf.push_back(g.input("k_far" + std::to_string(h), DType::BF16, {Sf, D}));
            vf.push_back(g.input("v_far" + std::to_string(h), DType::BF16, {Sf, D}));
        }
    }
    // attention of query rows [Tq] (q [Tq, NH, D], r [Tq, NH, DR], sink [Tq, NH], rel-bias rows idx [Tq * SN], masks
    // [Tq, SN]) over the far rows, the ring and all T new rows -> [KV, G*Tq, D] F32
    auto attend = [&](TensorId qb, TensorId rb, TensorId skb, TensorId idx, TensorId valid, TensorId mask_n, uint64_t Tq,
                      const std::string& x) {
        const auto q8 = o.reshape(o.transpose(qb, {1, 0, 2}, "qt" + x), {KV, G * Tq, D}, "q8" + x);  // [KV, G*Tq, D]
        const bool far16 = !kf.empty() && sm16_far_ok(uint32_t(Tq), uint32_t(Sf), uint32_t(W)) && Tq == T;
        if (kon(32) && (kf.empty() || far16) && Tq == T && T % 128 == 0) {
            // the Qwen prefill form: per KV head FP16 scores -> 128-lane FP16 softmax kernel (unnormalised BF16 P +
            // row sums) -> BF16 PV / sums; all heads' QK, then their softmaxes, then their PVs (MME under TPC)
            const float scl = 1.f / std::sqrt(float(D));
            const auto rsc = o.cast(o.mul(o.cast(rb, DType::F32, "r16f" + x), o.constant(1.f / scl, {1, 1, 1}, "r16i" + x), "r16s" + x),
                                    DType::BF16, "r16b" + x);                                          // r / scale
            const auto projR = o.gather_rows(projT, g.input("rel_rev", DType::I32, {RE}), "proj16R" + x);  // [RE, DR]
            const auto br = o.reshape(o.linear_bf16(o.reshape(rsc, {Tq * NH, DR}, "r16r" + x), projR, "b16rel" + x, DType::F16),
                                      {Tq, NH, RE}, "b16r3" + x);                                         // bias / scale
            const auto qs = o.split(q8, 0, std::vector<uint64_t>(KV, 1), "qh8" + x);
            const auto krs = o.split(kn_c, 0, std::vector<uint64_t>(KV, 1), "krh" + x);
            const auto kns = o.split(k8, 0, std::vector<uint64_t>(KV, 1), "knh" + x);
            const auto vrs = o.split(vn_c, 0, std::vector<uint64_t>(KV, 1), "vrh" + x);
            const auto vns = o.split(v8, 0, std::vector<uint64_t>(KV, 1), "vnh" + x);
            const auto pos_in = g.input("attn_pos", DType::I32, {1});
            std::vector<TensorId> sr(KV), sw(KV), sf(KV), ph(KV), pf(KV), sm(KV), outs(KV);
            auto qk = [&](uint64_t h) {
                const auto hs = std::to_string(h) + x;
                const auto qh = o.reshape(qs[h], {G * Tq, D}, "qh2" + hs);
                sr[h] = o.linear_bf16(qh, o.reshape(krs[h], {W, D}, "kr2" + hs), "s16r" + hs, DType::F16);   // [GT, W]
                sw[h] = o.linear_bf16(qh, o.reshape(kns[h], {T, D}, "kn2" + hs), "s16n" + hs, DType::F16);   // [GT, T]
                if (far16) sf[h] = o.linear_bf16(qh, kf[h], "s16f" + hs, DType::F16);                    // [GT, Sf]
            };
            auto softmax = [&](uint64_t h) {
                const auto hs = std::to_string(h) + x;
                ph[h] = o.temp("p16" + hs, DType::BF16, {G * Tq, SN});
                sm[h] = o.temp("ps16" + hs, DType::F32, {G * Tq, 1});
                struct { float scale; int32_t G, full, kv; } prm{1.f / std::sqrt(float(D)), int32_t(G), full ? 1 : 0, int32_t(h)};
                if (far16) {
                    pf[h] = o.temp("pf16" + hs, DType::BF16, {G * Tq, Sf});
                    const TensorId kins[] = {sf[h], sr[h], sw[h], br, skb, pos_in};
                    const TensorId kouts[] = {pf[h], ph[h], sm[h]};
                    g.op("ink_attn_sm16f_gaudi2", kins, kouts, &prm, sizeof(prm));
                } else {
                    const TensorId kins[] = {sr[h], sw[h], br, skb, pos_in};
                    const TensorId kouts[] = {ph[h], sm[h]};
                    g.op("ink_attn_sm16_gaudi2", kins, kouts, &prm, sizeof(prm));
                }
            };
            auto pv = [&](uint64_t h) {
                const auto hs = std::to_string(h) + x;
                const auto vh = o.concat({o.reshape(vrs[h], {W, D}, "vr2" + hs), o.reshape(vns[h], {T, D}, "vn2" + hs)}, 0, "vc16" + hs);
                auto oh = o.temp("o16" + hs, DType::F32, {G * Tq, D});
                g.gemm(ph[h], vh, oh);
                if (far16) {
                    const auto ohf = o.temp("o16f" + hs, DType::F32, {G * Tq, D});
                    g.gemm(pf[h], vf[h], ohf);
                    oh = o.add(oh, ohf, "o16s" + hs);
                }
                outs[h] = o.reshape(o.div(oh, sm[h], "o16n" + hs), {1, G * Tq, D}, "o16r" + hs);
            };
            if (!far16) {  // all heads' QK, then their softmaxes, then their PVs (MME under TPC)
                for (uint64_t h = 0; h < KV; ++h) qk(h);
                for (uint64_t h = 0; h < KV; ++h) softmax(h);
                for (uint64_t h = 0; h < KV; ++h) pv(h);
            } else {       // far keys: one head ahead (the MME computes head h + 1's scores under head h's softmax)
                qk(0);
                for (uint64_t h = 0; h < KV; ++h) {
                    if (h + 1 < KV) qk(h + 1);
                    softmax(h);
                    pv(h);
                }
            }
            return o.concat(outs, 0, "o16all" + x);                                                        // [KV, GT, D]
        }
        TensorId bias;
        {
            const auto P = o.reshape(o.gather_rows(projT, idx, "pg" + x), {Tq, SN, DR}, "P3" + x);    // [Tq, SN, DR]
            const auto b = o.batch_gemm(rb, P, "rel" + x, false, true, DType::F32);                  // [Tq, NH, SN]
            auto bb = o.cast(o.cast(b, DType::BF16, "relb" + x), DType::F32, "relbf" + x);           // bf16 table values
            if (full) bb = o.mul(bb, o.reshape(valid, {Tq, 1, SN}, "gv3" + x), "relv" + x);
            bias = o.reshape(o.transpose(bb, {1, 0, 2}, "relt" + x), {KV, G * Tq, SN}, "bias" + x);
        }
        // scores against the ring and the new rows as separate products (no copy of the 2 MiB rings)
        const auto sn = o.concat({o.batch_gemm(q8, kn_c, "snc" + x, false, true, DType::F32),
                                  o.batch_gemm(q8, k8, "snn" + x, false, true, DType::F32)}, 2, "sn" + x);  // [KV, GTq, SN]
        const auto snm = o.reshape(o.add(o.reshape(o.add(o.mul(sn, scale, "sns" + x), bias, "snb" + x), {KV, G, Tq, SN}, "sn4" + x),
                                         mask_n, "snm4" + x), {KV, G * Tq, SN}, "snm" + x);
        // far keys (full layers, Sf > 0): per-head GEMMs over the bound cache rows [0, Sf)
        TensorId sfm = kNoTensor;
        if (!kf.empty()) {
            std::vector<TensorId> sc;
            const auto q8s = o.split(q8, 0, std::vector<uint64_t>(KV, 1), "q8s" + x);
            for (uint64_t h = 0; h < KV; ++h) {
                const auto hs = std::to_string(h) + x;
                const auto s = o.linear_bf16(o.reshape(q8s[h], {G * Tq, D}, "qh" + hs), kf[h], "sf" + hs, DType::F32);  // [GTq, Sf]
                sc.push_back(o.reshape(s, {1, G * Tq, Sf}, "sf3" + hs));
            }
            sfm = o.add(o.mul(o.concat(sc, 0, "sfa" + x), scale, "sfs" + x), o.reshape(fmask, {1, 1, Sf}, "fm3" + x), "sfm" + x);
        }
        const auto sk = o.reshape(o.transpose(skb, {1, 0}, "skt" + x), {KV, G * Tq, 1}, "sk3" + x);
        std::vector<TensorId> parts;
        if (sfm != kNoTensor) parts.push_back(sfm);
        parts.push_back(snm);
        parts.push_back(sk);
        const auto p = o.softmax(o.concat(parts, 2, "sall" + x), "p" + x);                         // [KV, GTq, Stot + 1]
        const auto pb = o.cast(p, DType::BF16, "pb" + x);
        std::vector<uint64_t> psz;
        if (sfm != kNoTensor) psz.push_back(Sf);
        psz.push_back(SN);
        psz.push_back(1);
        const auto ps = o.split(pb, 2, psz, "psp" + x);
        const auto pnn = o.split(ps[sfm != kNoTensor ? 1 : 0], 2, {W, T}, "pnn" + x);
        TensorId att = o.add(o.batch_gemm(pnn[0], vn_c, "pvc" + x, false, false, DType::F32),
                             o.batch_gemm(pnn[1], v8, "pvw" + x, false, false, DType::F32), "pvn" + x);  // [KV, GTq, D]
        if (sfm != kNoTensor) {
            const auto pf = o.split(ps[0], 0, std::vector<uint64_t>(KV, 1), "pfs" + x);
            std::vector<TensorId> outs;
            for (uint64_t h = 0; h < KV; ++h) {
                const auto hs = std::to_string(h) + x;
                const auto oh = o.temp("of" + hs, DType::F32, {G * Tq, D});
                g.gemm(o.reshape(pf[h], {G * Tq, Sf}, "pf2" + hs), vf[h], oh);
                outs.push_back(o.reshape(oh, {1, G * Tq, D}, "of3" + hs));
            }
            att = o.add(att, o.concat(outs, 0, "ofa" + x), "att2" + x);
        }
        return att;
    };
    // full layers at long contexts: the scores [KV, G*Tq, Sf + SN] F32 are bounded by attending in query blocks
    const uint64_t Tq = full ? attn_block(T, Sf) : T;
    TensorId att;
    if (Tq >= T) {
        att = attend(q, r, sink, idx_all, valid_all, mask_all, T, "");
    } else {
        const uint64_t nb = T / Tq;
        const auto qs = o.split(q, 0, std::vector<uint64_t>(nb, Tq), "qblk");
        const auto rs = o.split(r, 0, std::vector<uint64_t>(nb, Tq), "rblk");
        const auto ss = o.split(sink, 0, std::vector<uint64_t>(nb, Tq), "sblk");
        const auto is = o.split(idx_all, 0, std::vector<uint64_t>(nb, Tq * SN), "iblk");
        const auto vs = o.split(valid_all, 0, std::vector<uint64_t>(nb, Tq), "vblk");
        const auto ms = o.split(mask_all, 0, std::vector<uint64_t>(nb, Tq), "mblk");
        std::vector<TensorId> outs;
        for (uint64_t i = 0; i < nb; ++i) {
            const auto x = "_q" + std::to_string(i);
            outs.push_back(o.reshape(attend(qs[i], rs[i], ss[i], is[i], vs[i], ms[i], Tq, x), {NH, Tq, D}, "atb" + x));
        }
        att = o.concat(outs, 1, "atcat");  // [NH, T, D]
    }
    // [KV, G*T, D] -> [T, NH*D], gate, out proj
    const auto at2 = o.reshape(o.transpose(o.reshape(att, {NH, T, D}, "at3"), {1, 0, 2}, "att"), {T, NH * D}, "at2");
    const auto gated = o.cast(o.mul(at2, o.unary("sigmoid", o.cast(gate, DType::F32, "gf"), "gs"), "gated"), DType::BF16, "gb");
    const auto a1 = o.linear_bf16(gated, g.input("wo", DType::BF16, {H, NH * D}), "oproj");       // [T, H] bf16
    // new K/V rows -> scratch [2, KV, T, D] (committed into the caches after the forward: a launch must not read and
    // write overlapping persistent tensors)
    output(g, "kv_new", o.concat({o.reshape(k8, {1, KV, T, D}, "k84"), o.reshape(v8, {1, KV, T, D}, "v84")}, 0, "kvn"));
    if (dbgout) {
        output(g, "dbg_gated", gated);
        output(g, "dbg_a1", a1);
    }
    // post-attention conv, residual
    const auto csa = g.input("cs_a", DType::BF16, {3, H});
    const auto csa_o = g.input("cs_a_o", DType::BF16, {verify_ ? 3 + T : 3, H});
    const auto a2 = !kon(4) ? short_conv(o, "acv", a1, csa, g.input("attn_conv", DType::F32, {4, H}), T, uint32_t(H), csa_o)
                              : fused_conv(o, "acv", a1, csa, g.input("attn_conv", DType::F32, {4, H}), kNoTensor, T, uint32_t(H),
                                           csa_o, 1.f / std::sqrt(5.f), 0);
    const auto res = boundary ? o.cast(a2, DType::F32, "resf")
                              : o.add(o.cast(h_in, DType::F32, "hinf"), o.cast(a2, DType::F32, "a2f"), "resf");
    // MLP input
    const auto mq = g.input("mlp_q", DType::F32, {1, H});
    TensorId xn;
    if (fmix) {
        const auto mnw = win("mlp_norm");
        const auto mr = fused_mix(o, "mr", blocks, res, mq, o.reshape(mnw, {1, H}, "mn1"), T, 1, msg, !fmg);
        if (msg && fmg) {
            struct { float eps; int32_t m0; } prm{cfg_.eps, 3};
            const auto cat = o.temp("mmcat", DType::BF16, {1, T, 2 * H});
            g.op("ink_mg_gaudi2", {mr.second, msg_il}, {cat}, &prm, sizeof(prm));
            const auto gr = o.batch_gemm(cat, g.input("mgd1", DType::BF16, {1, cfg_.gate_rank, 2 * H}), "mmg", false, true, DType::BF16);
            const auto fgv = o.batch_gemm(gr, g.input("mgf1", DType::BF16, {1, 2 * H, cfg_.gate_rank}), "mmfg", false, true, DType::F32);
            const auto nq = o.temp("mmnorm", DType::BF16, {1, T, H});
            g.op(mg_out_kernel(), {mr.second, msg_il, fgv, o.reshape(mnw, {1, H}, "mn2")}, {nq}, &prm, sizeof(prm));
            xn = o.reshape(nq, {T, H}, "mnb");
        } else if (msg) {
            xn = norm(message_gate(o, "mgm", o.reshape(mr.second, {T, H}, "min"), mm, T, "mlp"), "mlp_norm", "mn");
        } else {
            xn = o.reshape(mr.first, {T, H}, "mnb");
        }
    } else {
        const auto mstack = o.concat({o.cast(blocks, DType::F32, "blkf"), o.reshape(res, {1, T, H}, "res3")}, 0, "mstk");
        auto min = o.reshape(attn_res(o, "mr", mstack, nbb + 1, T, mq, 1), {T, H}, "min");
        if (msg) min = message_gate(o, "mgm", min, mm, T, "mlp");
        xn = norm(min, "mlp_norm", "mn");
    }
    const auto rt = route(o, "rt", xn, T);
    const auto mo = T <= kBf16SmallMaxT ? moe_bf16_small(o, "moe", xn, rt.first, rt.second, T)
                                        : moe_bf16_cap(o, "moe", xn, rt.first, rt.second, T);
    const auto csm = g.input("cs_m", DType::BF16, {3, H});
    const auto csm_o = g.input("cs_m_o", DType::BF16, {verify_ ? 3 + T : 3, H});
    const auto m2 = !kon(4) ? short_conv(o, "mcv", mo, csm, g.input("mlp_conv", DType::F32, {4, H}), T, uint32_t(H), csm_o)
                              : fused_conv(o, "mcv", mo, csm, g.input("mlp_conv", DType::F32, {4, H}), kNoTensor, T, uint32_t(H),
                                           csm_o, 1.f / std::sqrt(5.f), 0);
    if (dbgout) {
        output(g, "dbg_xn", xn);
        output(g, "dbg_mo", mo);
        output(g, "dbg_res", o.cast(res, DType::BF16, "dbgresb"));
        output(g, "dbg_idx", rt.first);
    }
    output(g, "h_out", o.cast(o.add(res, o.cast(m2, DType::F32, "m2f"), "hsum"), DType::BF16, "hout"));
    return *(recipes_[key] = g.compile(key));
}


void Inkling2::enqueue(Recipe& r, const std::vector<Binding>& all) {
    std::vector<Binding> b;
    b.reserve(r.tensors().size());
    for (const auto& t : r.tensors())
        for (const auto& x : all)
            if (x.name == t.name) { b.push_back(x); break; }
    static const bool prof = std::getenv("G2E_INK_PROFILE") != nullptr;
    if (!prof) {
        r.enqueue(stream_, b);
        return;
    }
    Event e0(dev_), e1(dev_);
    e0.record(stream_);
    r.enqueue(stream_, b);
    e1.record(stream_);
    e1.synchronize();
    auto& p = profile_[r.name()];
    p.first += 1;
    p.second += Event::elapsed_ms(e0, e1);
}

void Inkling2::print_profile() {
    std::vector<std::pair<std::string, std::pair<uint64_t, double>>> v(profile_.begin(), profile_.end());
    std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    double tot = 0;
    for (const auto& x : v) tot += x.second.second;
    std::fprintf(stderr, "profile: %.2f ms in recipes\n", tot);
    for (const auto& x : v)
        std::fprintf(stderr, "  %-40s %6llu x %8.3f ms = %8.2f ms (%4.1f%%)\n", x.first.c_str(), (unsigned long long)x.second.first,
                     x.second.second / double(x.second.first), x.second.second, 100.0 * x.second.second / tot);
    profile_.clear();
}


// One launch commits every application's new K/V rows (scratch -> ring, and -> linear cache for full layers).
Recipe& Inkling2::commit_recipe(uint32_t T) {
    const std::string key = "ink_commit_T" + std::to_string(T) + "_C" + std::to_string(cap_);
    if (auto it = recipes_.find(key); it != recipes_.end()) return *it->second;
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim, W = cfg_.window;
    GraphBuilder g(dev_);
    const auto rr = g.input("rows_ring", DType::I32, {T});
    const auto rl = g.input("rows_lin", DType::I32, {T});
    for (const auto& a : apps_) {
        const auto ai = std::to_string(a.index);
        const auto kv = g.input("kv" + ai, DType::BF16, {2, KV, T, D});
        g.op("ink_kv_commit_bf16_gaudi2", {kv, rr},
             {g.input("kr" + ai, DType::BF16, {KV, W, D}), g.input("vr" + ai, DType::BF16, {KV, W, D})});
        if (cfg_.full[a.layer])
            g.op("ink_kv_commit_bf16_gaudi2", {kv, rl},
                 {g.input("kc" + ai, DType::BF16, {KV, cap_, D}), g.input("vc" + ai, DType::BF16, {KV, cap_, D})});
    }
    return *(recipes_[key] = g.compile(key));
}

// ------------------------------------------------------------------------------------------------ state snapshots
// The state at a position = the linear caches' rows below it (never overwritten while the position stays) + every
// ring + the current conv states: a snapshot copies the rings and conv states (device to device).
uint64_t Inkling2::state_bytes(uint32_t pos) const {
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim;
    return uint64_t(kring_.size()) * 2 * ring_slot_bytes_ + conv_app_bytes_ * apps_.size() +
           uint64_t(kcache_.size()) * 2 * KV * pos * D * 2;
}

void Inkling2::save_state(const DeviceBuffer& dst, uint64_t off) const {
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim, ro = ring_off(active_slot_);
    if (dst.size() < off + state_bytes(pos_)) throw std::out_of_range("inkling2: snapshot buffer too small");
    uint64_t at = off;
    for (size_t i = 0; i < kring_.size(); ++i) {
        stream_.copy(kring_[i], ro, dst, at, ring_slot_bytes_);
        at += ring_slot_bytes_;
        stream_.copy(vring_[i], ro, dst, at, ring_slot_bytes_);
        at += ring_slot_bytes_;
    }
    stream_.copy(conv_pool_, conv_off_slot(active_slot_, conv_side_), dst, at, conv_app_bytes_ * apps_.size());
    at += conv_app_bytes_ * apps_.size();
    const uint64_t rows = uint64_t(pos_) * D * 2;
    if (rows)
        for (size_t i = 0; i < kcache_.size(); ++i)
            for (int kv = 0; kv < 2; ++kv)
                for (uint64_t h = 0; h < KV; ++h) {
                    stream_.copy(kv == 0 ? kcache_[i] : vcache_[i], h * uint64_t(cap_) * D * 2, dst, at, rows);
                    at += rows;
                }
}

void Inkling2::load_state(const DeviceBuffer& src, uint64_t off, uint32_t pos) {
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim, ro = ring_off(active_slot_);
    if (pos > cap_) throw std::out_of_range("inkling2: snapshot position past the context");
    if (src.size() < off + state_bytes(pos)) throw std::out_of_range("inkling2: snapshot buffer too small");
    uint64_t at = off;
    for (size_t i = 0; i < kring_.size(); ++i) {
        stream_.copy(src, at, kring_[i], ro, ring_slot_bytes_);
        at += ring_slot_bytes_;
        stream_.copy(src, at, vring_[i], ro, ring_slot_bytes_);
        at += ring_slot_bytes_;
    }
    stream_.copy(src, at, conv_pool_, conv_off_slot(active_slot_, conv_side_), conv_app_bytes_ * apps_.size());
    at += conv_app_bytes_ * apps_.size();
    const uint64_t rows = uint64_t(pos) * D * 2;
    if (rows)
        for (size_t i = 0; i < kcache_.size(); ++i)
            for (int kv = 0; kv < 2; ++kv)
                for (uint64_t h = 0; h < KV; ++h) {
                    stream_.copy(src, at, kv == 0 ? kcache_[i] : vcache_[i], h * uint64_t(cap_) * D * 2, rows);
                    at += rows;
                }
    pos_ = pos;
    direct_prev_ = false;
}

void Inkling2::save_state(const HostBuffer& dst, uint64_t off) const {
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim, ro = ring_off(active_slot_);
    if (dst.size() < off + state_bytes(pos_)) throw std::out_of_range("inkling2: host snapshot buffer too small");
    uint64_t at = off;
    for (size_t i = 0; i < kring_.size(); ++i) {
        stream_.download(kring_[i], ro, dst, at, ring_slot_bytes_);
        at += ring_slot_bytes_;
        stream_.download(vring_[i], ro, dst, at, ring_slot_bytes_);
        at += ring_slot_bytes_;
    }
    stream_.download(conv_pool_, conv_off_slot(active_slot_, conv_side_), dst, at, conv_app_bytes_ * apps_.size());
    at += conv_app_bytes_ * apps_.size();
    const uint64_t rows = uint64_t(pos_) * D * 2;
    if (rows)
        for (size_t i = 0; i < kcache_.size(); ++i)
            for (int kv = 0; kv < 2; ++kv)
                for (uint64_t h = 0; h < KV; ++h) {
                    stream_.download(kv == 0 ? kcache_[i] : vcache_[i], h * uint64_t(cap_) * D * 2, dst, at, rows);
                    at += rows;
                }
}

void Inkling2::load_state(const HostBuffer& src, uint64_t off, uint32_t pos) {
    const uint64_t KV = cfg_.kv_heads, D = cfg_.head_dim, ro = ring_off(active_slot_);
    if (pos > cap_) throw std::out_of_range("inkling2: snapshot position past the context");
    if (src.size() < off + state_bytes(pos)) throw std::out_of_range("inkling2: host snapshot buffer too small");
    uint64_t at = off;
    for (size_t i = 0; i < kring_.size(); ++i) {
        stream_.upload(src, at, kring_[i], ro, ring_slot_bytes_);
        at += ring_slot_bytes_;
        stream_.upload(src, at, vring_[i], ro, ring_slot_bytes_);
        at += ring_slot_bytes_;
    }
    stream_.upload(src, at, conv_pool_, conv_off_slot(active_slot_, conv_side_), conv_app_bytes_ * apps_.size());
    at += conv_app_bytes_ * apps_.size();
    const uint64_t rows = uint64_t(pos) * D * 2;
    if (rows)
        for (size_t i = 0; i < kcache_.size(); ++i)
            for (int kv = 0; kv < 2; ++kv)
                for (uint64_t h = 0; h < KV; ++h) {
                    stream_.upload(src, at, kv == 0 ? kcache_[i] : vcache_[i], h * uint64_t(cap_) * D * 2, rows);
                    at += rows;
                }
    pos_ = pos;
    direct_prev_ = false;
}

// ------------------------------------------------------------------------------------------------ tables
void Inkling2::build_tables(uint32_t T) {
    const uint32_t W = cfg_.window, SN = W + T, pos = pos_;
    auto* base = staging_.as<uint8_t>();
    auto* swa_idx = reinterpret_cast<int32_t*>(base + tab_.swa_idx);
    auto* swa_mask = reinterpret_cast<float*>(base + tab_.swa_mask);
    auto* gn_idx = reinterpret_cast<int32_t*>(base + tab_.gn_idx);
    auto* gn_valid = reinterpret_cast<float*>(base + tab_.gn_valid);
    auto* gn_mask = reinterpret_cast<float*>(base + tab_.gn_mask);
    auto* gf_mask = reinterpret_cast<float*>(base + tab_.gf_mask);
    auto* tau = reinterpret_cast<float*>(base + tab_.tau);
    // key positions of the near columns (-1: no key)
    std::vector<int64_t> swa_pos(SN), gn_pos(SN);
    const int64_t near_start = pos >= W ? int64_t(pos) - W : 0;
    for (uint32_t j = 0; j < W; ++j) {
        if (pos == 0) {
            swa_pos[j] = -1;
        } else {
            const int64_t last = int64_t(pos) - 1;
            const int64_t back = ((last - j) % int64_t(W) + W) % int64_t(W);
            swa_pos[j] = last - back;  // < 0 when that slot was never written
        }
        gn_pos[j] = swa_pos[j];
    }
    for (uint32_t u = 0; u < T; ++u) swa_pos[W + u] = gn_pos[W + u] = int64_t(pos) + u;
    for (uint32_t t = 0; t < T; ++t) {
        const int64_t qp = int64_t(pos) + t;
        for (uint32_t c = 0; c < SN; ++c) {
            const uint64_t at = uint64_t(t) * SN + c;
            {
                const int64_t kp = swa_pos[c];
                const int64_t d = qp - kp;
                const bool ok = kp >= 0 && d >= 0 && d < int64_t(W);
                swa_idx[at] = int32_t(std::clamp<int64_t>(d, 0, cfg_.rel_extent - 1));
                swa_mask[at] = ok ? 0.f : kMaskNeg;
            }
            {
                const int64_t kp = gn_pos[c];
                const int64_t d = qp - kp;
                const bool ok = kp >= 0 && d >= 0;
                gn_idx[at] = int32_t(std::clamp<int64_t>(d, 0, cfg_.rel_extent - 1));
                gn_valid[at] = (ok && d < int64_t(cfg_.rel_extent)) ? 1.f : 0.f;
                gn_mask[at] = ok ? 0.f : kMaskNeg;
            }
        }
        const double x = double(qp + 1) / double(cfg_.log_floor);
        tau[t] = float(1.0 + double(cfg_.log_alpha) * std::log(std::max(x, 1.0)));
    }
    const uint32_t Sf = far_bucket(uint32_t(near_start));
    for (uint32_t c = 0; c < Sf; ++c) gf_mask[c] = int64_t(c) < near_start ? 0.f : kMaskNeg;
    *reinterpret_cast<int32_t*>(base + tab_.pos32) = int32_t(pos);
    auto* iota = reinterpret_cast<float*>(base + tab_.tau + kMaxT * 4);
    for (uint32_t e = 0; e < cfg_.experts; ++e) iota[e] = float(e);
    {   // capacity-path constants (static; rewritten with every forward for simplicity)
        auto* st = reinterpret_cast<float*>(base + tab_.slot_tok);
        for (uint32_t sl = 0; sl < T * cfg_.topk; ++sl) {
            const uint32_t t = sl / cfg_.topk;
            st[2 * sl] = float(t / 32);
            st[2 * sl + 1] = float(t % 32);
        }
        stream_.upload(staging_, tab_.slot_tok, tables_, tab_.slot_tok, uint64_t(T) * cfg_.topk * 2 * 4);
        stream_.upload(staging_, tab_.slot_tok, tables_, tab_.slot_tok2, uint64_t(T) * cfg_.topk * 2 * 4);
        auto* rv = reinterpret_cast<int32_t*>(base + tab_.rel_rev);
        for (uint32_t k = 0; k < cfg_.rel_extent; ++k) rv[k] = int32_t(cfg_.rel_extent - 1 - k);
        stream_.upload(staging_, tab_.rel_rev, tables_, tab_.rel_rev, uint64_t(cfg_.rel_extent) * 4);
    }
    {   // commit rows: ring slot of each of the last min(T, W) new rows (-1 for older ones), linear cache row
        auto* rr = reinterpret_cast<int32_t*>(base + tab_.rows);
        auto* rl = rr + kMaxT;
        for (uint32_t t = 0; t < T; ++t) {
            rr[t] = (T > W && t < T - W) ? -1 : int32_t((pos + t) % W);
            rl[t] = int32_t(pos + t);
        }
        stream_.upload(staging_, tab_.rows, tables_, tab_.rows, 2ull * kMaxT * 4);
    }
    const uint64_t n = uint64_t(T) * SN;
    // upload the used parts
    stream_.upload(staging_, tab_.swa_idx, tables_, tab_.swa_idx, n * 4);
    stream_.upload(staging_, tab_.swa_mask, tables_, tab_.swa_mask, n * 4);
    stream_.upload(staging_, tab_.gn_idx, tables_, tab_.gn_idx, n * 4);
    stream_.upload(staging_, tab_.gn_valid, tables_, tab_.gn_valid, n * 4);
    stream_.upload(staging_, tab_.gn_mask, tables_, tab_.gn_mask, n * 4);
    if (Sf) stream_.upload(staging_, tab_.gf_mask, tables_, tab_.gf_mask, uint64_t(Sf) * 4);
    stream_.upload(staging_, tab_.tau, tables_, tab_.tau, kMaxT * 4 + 1024);
    stream_.upload(staging_, tab_.pos32, tables_, tab_.pos32, 4);
}

// ------------------------------------------------------------------------------------------------ forward
int32_t Inkling2::run_decode_direct(int32_t token, bool want_logits, std::vector<float>* logits, bool want_argmax) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    if (pos_ + 1 > cap_) throw std::out_of_range("inkling2: context full");
    if (!direct_prev_) stream_.synchronize();
    const uint64_t H = cfg_.hidden, slab = H * 2;
    const double x = double(pos_ + 1) / double(cfg_.log_floor);
    const float tau0 = float(1.0 + double(cfg_.log_alpha) * std::log(std::max(x, 1.0)));

    std::vector<InkDirect::DecodeAppIO> io;
    io.reserve(apps_.size());
    int hcur = 0, pside = 0;
    const DeviceBuffer* last_out = nullptr;
    uint64_t last_off = 0;
    for (const auto& a : apps_) {
        const uint32_t L = a.layer;
        const bool block_end = (L + 1) % cfg_.attn_res_block == 0;
        const bool last_pass = a.pass + 1 == cfg_.passes;
        const DeviceBuffer* out_buf;
        uint64_t out_off = 0;
        if (L + 1 == cfg_.layers) {
            out_buf = &h_[1 - hcur];
        } else if (block_end && L + 1 == cfg_.loop_end && !last_pass) {
            out_buf = &prev_[1 - pside];
            out_off = slab;
        } else if (block_end) {
            out_buf = &blocks_;
            out_off = uint64_t((L + 1) / cfg_.attn_res_block) * slab;
        } else {
            out_buf = &h_[1 - hcur];
        }
        io.push_back({a.index, hcur, out_buf, out_off});
        last_out = out_buf;
        last_off = out_off;
        if (out_buf == &h_[1 - hcur]) hcur = 1 - hcur;
        if (L + 1 == cfg_.loop_end && !last_pass) pside = 1 - pside;
    }
    const bool need_head = want_logits || want_argmax || aux_on_;
    const bool want_aux = aux_on_ && (want_logits || want_argmax);
    double em = 0, lm = 0, hm = 0;
    const int32_t tok = direct_->run_decode_full(token, pos_, tau0, conv_side_, io, last_out, last_off,
                                                 need_head, want_logits, logits, want_argmax,
                                                 want_aux ? &aux_host_ : nullptr, &em, &lm, &hm);
    conv_side_ = 1 - conv_side_;
    pos_ += 1;
    tap_slot_ = 0;
    direct_prev_ = true;
    timing_.embed = em;
    timing_.layers = lm;
    timing_.head = hm;
    last_ms_ = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    return tok;
}

void Inkling2::run_forward(std::span<const int32_t> tokens, bool want_logits, std::vector<float>* logits) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const uint32_t T = uint32_t(tokens.size());
    const uint32_t eff_T = (pad_n_ && !verify_) ? pad_n_ : T;
    if (!T || T > kMaxT) throw std::invalid_argument("inkling2: 1.." + std::to_string(kMaxT) + " tokens per forward");
    if (pos_ + eff_T > cap_) throw std::out_of_range("inkling2: context full");
    static const bool dec_dbg = std::getenv("G2E_INK_DIRECT_CHECK") || std::getenv("G2E_INK_DIRECT_PROF") ||
                                std::getenv("G2E_INK_DIRECT_NOCHAIN");
    if (T == 1 && !pad_n_ && !dec_dbg && direct_ && direct_on_ && !verify_ && !dump_ && inject_.empty() &&
        direct_->handles_full_decode()) {
        run_decode_direct(tokens[0], want_logits && logits, logits, /*want_argmax=*/false);
        return;
    }
    if (verify_ && !pad_n_ && !dump_ && inject_.empty() && want_logits && logits && !dec_dbg &&
        direct_ && direct_on_ && direct_->handles_verify_full(T)) {
        stream_.synchronize();  // ensure any async Synapse work (e.g. DSpark append / load_state) is complete
        const uint64_t H = cfg_.hidden, slab = uint64_t(T) * H * 2;
        const double x = double(pos_ + 1) / double(cfg_.log_floor);
        const float tau0 = float(1.0 + double(cfg_.log_alpha) * std::log(std::max(x, 1.0)));
        std::vector<InkDirect::DecodeAppIO> io;
        io.reserve(apps_.size());
        int hcur = 0, pside = 0;
        const DeviceBuffer* last_out = nullptr;
        uint64_t last_off = 0;
        for (const auto& a : apps_) {
            const uint32_t L = a.layer;
            const bool block_end = (L + 1) % cfg_.attn_res_block == 0;
            const bool last_pass = a.pass + 1 == cfg_.passes;
            const DeviceBuffer* out_buf;
            uint64_t out_off = 0;
            if (L + 1 == cfg_.layers) {
                out_buf = &h_[1 - hcur];
            } else if (block_end && L + 1 == cfg_.loop_end && !last_pass) {
                out_buf = &prev_[1 - pside];
                out_off = slab;
            } else if (block_end) {
                out_buf = &blocks_;
                out_off = uint64_t((L + 1) / cfg_.attn_res_block) * slab;
            } else {
                out_buf = &h_[1 - hcur];
            }
            io.push_back({a.index, hcur, out_buf, out_off});
            last_out = out_buf;
            last_off = out_off;
            if (out_buf == &h_[1 - hcur]) hcur = 1 - hcur;
            if (L + 1 == cfg_.loop_end && !last_pass) pside = 1 - pside;
        }
        double em = 0, lm = 0, hm = 0;
        direct_->run_verify_full(tokens, pos_, tau0, conv_side_, io, last_out, last_off, logits, &em, &lm, &hm);
        ver_T_ = T;
        ver_side_ = conv_side_;
        tap_slot_ = 0;
        direct_prev_ = true;
        timing_.embed = em;
        timing_.layers = lm;
        timing_.head = hm;
        last_ms_ = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        return;
    }
    direct_prev_ = false;
    const uint64_t H = cfg_.hidden, KV = cfg_.kv_heads, D = cfg_.head_dim, W = cfg_.window, KVD = KV * D;
    const uint64_t row = H * 2, slab = uint64_t(T) * row;  // one [T, H] bf16 tensor
    stream_.synchronize();  // the staging buffer is rewritten below
    // every application on the Direct engine (prefill chunk): only the position, tau, tpart/valid_T and commit rows are needed
    const bool dp_all = direct_ && direct_on_ && !verify_ && !dump_ && direct_->handles_all_p(T) && !std::getenv("G2E_INK_DPCHECK");
    if (direct_ && direct_on_ && T == kMaxT) direct_->set_tables(0, pos_, T, eff_T);  // (the prefill plans' own tables)
    if (!dp_all) build_tables(T);
    // the Direct prefill plans update the conv states in place, in side 0
    if (dp_all && conv_side_ != 0) {
        stream_.copy(conv_pool_, conv_off_slot(active_slot_, 1), conv_pool_, conv_off_slot(active_slot_, 0),
                     apps_.size() * conv_app_bytes_);
        conv_side_ = 0;
    }
    std::memcpy(staging_.as<uint8_t>() + tab_.bytes, tokens.data(), T * 4);
    stream_.upload(staging_, tab_.bytes, ids_, 0, T * 4);
    // embeddings -> block 0
    {
        const Binding b[] = {{"ids", &ids_, 0}, w_.bind("embed", "embed"), w_.bind("embed_norm", "embed_norm"),
                             {"out", &blocks_, 0}};
        embed_recipe(T).enqueue(stream_, b);
    }
    // image features over the embeddings of the positions they cover in this chunk
    for (const auto& in : inject_) {
        const uint32_t a = std::max(in.row, pos_), b = std::min(in.row + in.rows, pos_ + eff_T);
        if (a < b) stream_.copy(*in.src, in.off + uint64_t(a - in.row) * row, blocks_, uint64_t(a - pos_) * row, uint64_t(b - a) * row);
    }
    std::erase_if(inject_, [&](const Inject& in) { return in.row + in.rows <= pos_ + eff_T; });
    const auto t1 = clk::now();
    if (dump_) dump_out_.clear();
    const uint32_t near_start = pos_ >= W ? pos_ - uint32_t(W) : 0;
    const uint32_t Sf = far_bucket(near_start);
    const int side = conv_side_;
    int hcur = 0;            // h_[hcur] holds the current partial block (non-boundary inputs)
    int pside = 0;           // prev_[pside]: messages read; the pass writes prev_[1 - pside]
    auto blk = [&](uint32_t k) { return uint64_t(k) * slab; };
    const DeviceBuffer* last_out = nullptr;
    uint64_t last_off = 0;
    for (const auto& a : apps_) {
        const uint32_t L = a.layer;
        const bool full = cfg_.full[L], boundary = L % cfg_.attn_res_block == 0, msg = a.pass > 0;
        std::vector<Binding> b;
        b.reserve(96);
        b.push_back({"blocks", &blocks_, 0});
        if (!boundary) b.push_back({"h_in", &h_[hcur], 0});
        b.push_back({"kv_new", &kvscratch_, uint64_t(a.index) * 2 * KVD * kMaxT * 2});
        // output destination
        const DeviceBuffer* out_buf;
        uint64_t out_off = 0;
        const bool block_end = (L + 1) % cfg_.attn_res_block == 0;
        const bool last_pass = a.pass + 1 == cfg_.passes;
        if (L + 1 == cfg_.layers) {
            out_buf = &h_[1 - hcur];
        } else if (block_end && L + 1 == cfg_.loop_end && !last_pass) {
            out_buf = &prev_[1 - pside];  // partial block 2 of this pass: next pass's message source
            out_off = slab;
        } else if (block_end) {
            out_buf = &blocks_;
            out_off = blk((L + 1) / cfg_.attn_res_block);
        } else {
            out_buf = &h_[1 - hcur];
        }
        if (dp_all) {
            if (!direct_->pending()) stream_.synchronize();
            direct_->queue_p(a.index, hcur, out_buf, out_off, 0, 0);
        } else {
            b.push_back({"h_out", out_buf, out_off});
            if (msg) {
                b.push_back({"prev1", &prev_[pside], 0});
                b.push_back({"prev2", &prev_[pside], slab});
                if (kon(32)) b.push_back({"msg_q", &qil_, qil_off_.at("msg_q") + uint64_t(L - cfg_.loop_start) * 4 * H * 4});
                else b.push_back({"msg_q", &w_.arena(), w_.at("msg_q").offset + uint64_t(L - cfg_.loop_start) * 4 * H * 4});
                if (msg_stack_) {
                    const auto& md = w_.at(Lw(L, "mg_down"));
                    const auto& mf = w_.at(Lw(L, "mg_fg"));
                    const uint64_t RH = uint64_t(cfg_.gate_rank) * 2 * H * 2;  // one gate's weights (bytes)
                    b.push_back({"mgd3", &w_.arena(), md.offset});
                    b.push_back({"mgd1", &w_.arena(), md.offset + 3 * RH});
                    b.push_back({"mgf3", &w_.arena(), mf.offset});
                    b.push_back({"mgf1", &w_.arena(), mf.offset + 3 * RH});
                }
                for (const char* gname : {"q", "k", "v", "mlp"}) {
                    b.push_back(w_.bind(std::string("mg_") + gname + "_down", Lw(L, (std::string("mg_") + gname + "_down").c_str())));
                    b.push_back(w_.bind(std::string("mg_") + gname + "_fg", Lw(L, (std::string("mg_") + gname + "_fg").c_str())));
                }
            }
            // weights
            const auto& wx = w_.at(Lw(L, "wx"));
            const uint64_t NH = cfg_.heads, QS = NH * D * 2 + NH * cfg_.d_rel + NH, KR = full ? KVD + KV * 4 : KVD;
            b.push_back({"wq", &w_.arena(), wx.offset});
            b.push_back({"wk", &w_.arena(), wx.offset + QS * H * 2});
            b.push_back({"wv", &w_.arena(), wx.offset + (QS + KR) * H * 2});
            b.push_back(w_.bind("wo", Lw(L, "wo")));
            b.push_back(w_.bind("q_scale", Lw(L, "q_scale")));
            b.push_back(w_.bind("r_scale", Lw(L, "r_scale")));
            b.push_back(w_.bind("rel_proj", Lw(L, "rel_proj")));
            b.push_back(w_.bind("k_taps", Lw(L, "k_taps")));
            b.push_back(w_.bind("v_taps", Lw(L, "v_taps")));
            b.push_back(w_.bind("attn_conv", Lw(L, "attn_conv")));
            b.push_back(w_.bind("mlp_conv", Lw(L, "mlp_conv")));
            b.push_back(w_.bind("mlp_norm", Lw(L, "mlp_norm")));
            if (full) {
                b.push_back(w_.bind("q_norm", Lw(L, "q_norm")));
                b.push_back(w_.bind("k_norm", Lw(L, "k_norm")));
                b.push_back(w_.bind("v_norm", Lw(L, "v_norm")));
            } else {
                b.push_back(w_.bind("attn_norm", Lw(L, "attn_norm")));
            }
            if (kon(32)) {  // the fused mix reads the queries interleaved
                if (L > 0) b.push_back({"attn_q", &qil_, qil_off_.at(Lw(L, "attn_q"))});
                b.push_back({"mlp_q", &qil_, qil_off_.at(Lw(L, "mlp_q"))});
            } else {
                if (L > 0) b.push_back(w_.bind("attn_q", Lw(L, "attn_q")));
                b.push_back(w_.bind("mlp_q", Lw(L, "mlp_q")));
            }
            b.push_back(w_.bind("router", Lw(L, "router")));
            b.push_back(w_.bind("router_bias", Lw(L, "router_bias")));
            b.push_back(w_.bind("w13_g", Lw(L, "w13.g")));
            b.push_back(w_.bind("w2_g", Lw(L, "w2.g")));
            b.push_back({"iota_e", &tables_, tab_.tau + kMaxT * 4});
            b.push_back({"slot_tok", &tables_, tab_.slot_tok});
            b.push_back({"slot_tok2", &tables_, tab_.slot_tok2});
            b.push_back({"rel_rev", &tables_, tab_.rel_rev});
            b.push_back({"attn_pos", &tables_, tab_.pos32});
            // conv states
            const uint64_t ca = uint64_t(a.index) * conv_app_bytes_;
            const uint64_t cbase_in = conv_off_slot(active_slot_, side);
            const uint64_t cbase_out = conv_off_slot(active_slot_, 1 - side);
            const char* cs[4] = {"cs_k", "cs_v", "cs_a", "cs_m"};
            for (int i = 0; i < 4; ++i) {
                b.push_back({cs[i], &conv_pool_, cbase_in + ca + conv_off_[i]});
                if (verify_) b.push_back({std::string(cs[i]) + "_o", &hist_, uint64_t(a.index) * hist_app_ + hist_off(i, T)});
                else b.push_back({std::string(cs[i]) + "_o", &conv_pool_, cbase_out + ca + conv_off_[i]});
            }
            // attention caches and tables
            const uint64_t ro = ring_off(active_slot_);
            b.push_back({"k_ring", &kring_[ring_index_[a.index]], ro});
            b.push_back({"v_ring", &vring_[ring_index_[a.index]], ro});
            if (full) {
                const int s = kv_index_[a.index];
                for (uint64_t h = 0; h < KV; ++h) {
                    const auto hs = std::to_string(h);
                    const uint64_t head = h * uint64_t(cap_) * D * 2;
                    if (Sf) {
                        b.push_back({"k_far" + hs, &kcache_[s], head});
                        b.push_back({"v_far" + hs, &vcache_[s], head});
                    }
                }
                b.push_back({"gn_idx", &tables_, tab_.gn_idx});
                b.push_back({"gn_valid", &tables_, tab_.gn_valid});
                b.push_back({"gn_mask", &tables_, tab_.gn_mask});
                if (Sf) b.push_back({"gf_mask", &tables_, tab_.gf_mask});
                b.push_back({"tau", &tables_, tab_.tau});
            } else {
                b.push_back({"swa_idx", &tables_, tab_.swa_idx});
                b.push_back({"swa_mask", &tables_, tab_.swa_mask});
            }
            const bool dverify = verify_ && direct_ && direct_on_ && direct_->handles_v(a.index, T);
            if (aux_on_ && (a.index == 5 || a.index == 23 || a.index == 41) &&
                !(verify_ && direct_ && direct_on_ && direct_->handles_v(a.index - 1, T)) &&
                !(T == 1 && direct_ && direct_on_ && direct_->handles(a.index - 1)) &&
                !(direct_ && direct_on_ && direct_->handles_p(a.index - 1, T))) {  // (else the producer's plan wrote it)
                // DSpark taps: the running partial block entering this visit
                if (direct_) direct_->flush();
                stream_.copy(h_[hcur], 0, aux_, uint64_t(a.index == 5 ? 0 : a.index == 23 ? 1 : 2) * kMaxT * H * 2, slab);
            }
            // G2E_INK_DBGOUT: the recipes' intermediate outputs (compared by G2E_INK_DPCHECK)
            static DeviceBuffer* dbg = nullptr;
            const uint64_t NHD = cfg_.heads * D;
            const uint64_t o_g = 0, o_a1 = o_g + uint64_t(T) * NHD * 2, o_xn = o_a1 + slab, o_mo = o_xn + slab, o_res = o_mo + slab,
                           o_idx = o_res + slab, dbg_bytes = o_idx + uint64_t(T) * cfg_.topk * 4;
            if (std::getenv("G2E_INK_DBGOUT")) {
                if (!dbg) dbg = new DeviceBuffer(dev_, uint64_t(kMaxT) * NHD * 2 + 4 * uint64_t(kMaxT) * H * 2 + kMaxT * 8 * 4);
                for (auto [nm, off] : {std::pair<const char*, uint64_t>{"dbg_gated", o_g}, {"dbg_a1", o_a1}, {"dbg_xn", o_xn}, {"dbg_mo", o_mo},
                                       {"dbg_res", o_res}, {"dbg_idx", o_idx}})
                    b.push_back({nm, dbg, off});
            }
            (void)dbg_bytes;
            static const int dpcheck = std::getenv("G2E_INK_DPCHECK") ? std::atoi(std::getenv("G2E_INK_DPCHECK")) : -1;
            if (dpcheck == int(a.index) && direct_ && direct_on_ && direct_->handles_p(a.index, T)) {
                // G2E_INK_DPCHECK=<app>: the Synapse recipe, then the Direct plan on the same inputs; compare the outputs
                direct_->flush();
                bind_bf16_experts(L, b);
                enqueue(layer_recipe(a, T, full ? Sf : 0), b);
                const uint64_t ca = uint64_t(a.index) * conv_app_bytes_, kvo = uint64_t(a.index) * 2 * KVD * kMaxT * 2;
                auto grab = [&](const DeviceBuffer& buf, uint64_t off, uint64_t bytes) {
                    HostBuffer hb(dev_, bytes + 64);
                    stream_.download(buf, off, hb, 0, bytes);
                    stream_.synchronize();
                    return std::vector<uint16_t>(hb.as<uint16_t>(), hb.as<uint16_t>() + bytes / 2);
                };
                const auto r_h = grab(*out_buf, out_off, slab), r_kv = grab(kvscratch_, kvo, 2 * KVD * T * 2);
                const auto r_cs = grab(conv_pool_, conv_off_slot(active_slot_, 1 - side) + ca, conv_app_bytes_);
                direct_->queue_p(a.index, hcur, out_buf, out_off, side, 1 - side);
                direct_->flush();
                const auto d_h = grab(*out_buf, out_off, slab), d_kv = grab(kvscratch_, kvo, 2 * KVD * T * 2);
                const auto d_cs = grab(conv_pool_, conv_off_slot(active_slot_, 1 - side) + ca, conv_app_bytes_);
                auto cmp = [](const char* nm, const std::vector<uint16_t>& x, const std::vector<uint16_t>& y) {
                    double num = 0, den = 0, mx = 0;
                    size_t same = 0, worst = 0;
                    for (size_t i = 0; i < x.size(); ++i) {
                        const double u = from_bf16(x[i]), v = from_bf16(y[i]);
                        same += x[i] == y[i];
                        num += (u - v) * (u - v);
                        den += v * v;
                        if (std::fabs(u - v) > mx) { mx = std::fabs(u - v); worst = i; }
                    }
                    std::fprintf(stderr, "  [dpcheck] %-6s identical %6.2f%%  rel %.3e  max|d| %.4g at %zu (direct %.5g synapse %.5g)\n", nm,
                                 100.0 * double(same) / double(x.size()), std::sqrt(num / (den + 1e-30)), mx, worst, from_bf16(x[worst]),
                                 from_bf16(y[worst]));
                };
                std::fprintf(stderr, "[dpcheck] app %u (layer %u) pos %u\n", a.index, L, pos_);
                cmp("h_out", d_h, r_h);
                cmp("k_new", std::vector<uint16_t>(d_kv.begin(), d_kv.begin() + KVD * T), std::vector<uint16_t>(r_kv.begin(), r_kv.begin() + KVD * T));
                cmp("v_new", std::vector<uint16_t>(d_kv.begin() + KVD * T, d_kv.end()), std::vector<uint16_t>(r_kv.begin() + KVD * T, r_kv.end()));
                cmp("conv", d_cs, r_cs);
                if (pos_ == 0) {  // determinism: the plan again on the same inputs (the first chunk reads no ring rows)
                    for (int rep = 0; rep < 3; ++rep) {
                        direct_->queue_p(a.index, hcur, out_buf, out_off, side, 1 - side);
                        direct_->flush();
                        const auto e_h = grab(*out_buf, out_off, slab);
                        size_t nd = 0, first = 0;
                        for (size_t i = 0; i < e_h.size(); ++i)
                            if (e_h[i] != d_h[i]) { if (!nd) first = i; ++nd; }
                        std::fprintf(stderr, "  [dpcheck] rerun %d: %zu of %zu h_out values differ (first at row %zu col %zu)\n", rep, nd, e_h.size(),
                                     first / H, first % H);
                    }
                }
                if (dbg) {
                    const auto& ps = direct_->pdebug();
                    const auto& pb = direct_->pscratch();
                    auto dgrab = [&](const char* nm, uint64_t bytes) { return grab(pb, ps.at(nm) - pb.address(), bytes); };
                    cmp("gated", dgrab("gated", uint64_t(T) * NHD * 2), grab(*dbg, o_g, uint64_t(T) * NHD * 2));
                    if (const char* dd = std::getenv("G2E_INK_DPDUMP")) {  // raw tensors for offline analysis
                        auto wr = [&](const std::string& nm, const std::vector<uint16_t>& v) {
                            if (FILE* f = std::fopen((std::string(dd) + "/" + nm + "_p" + std::to_string(pos_) + ".bin").c_str(), "wb")) {
                                std::fwrite(v.data(), 2, v.size(), f);
                                std::fclose(f);
                            }
                        };
                        wr("gated_d", dgrab("gated", uint64_t(T) * NHD * 2));
                        wr("gated_s", grab(*dbg, o_g, uint64_t(T) * NHD * 2));
                        wr("mo_d", dgrab("mo", slab));
                        wr("mo_s", grab(*dbg, o_mo, slab));
                        wr("idx_d", dgrab("idx", uint64_t(T) * cfg_.topk * 4));
                        wr("idx_s", grab(*dbg, o_idx, uint64_t(T) * cfg_.topk * 4));
                        wr("xn_d", dgrab("xn", slab));
                        wr("xn_s", grab(*dbg, o_xn, slab));
                        for (const char* nm : {"q", "sums", "o", "row", "p0", "r16", "bt"})
                            if (ps.count(nm)) wr(std::string(nm) + "_d", dgrab(nm, ps.at(std::string(nm) + "#bytes")));
                    }
                    cmp("a1", dgrab("a1", slab), grab(*dbg, o_a1, slab));
                    cmp("xn", dgrab("xn", slab), grab(*dbg, o_xn, slab));
                    cmp("mo", dgrab("mo", slab), grab(*dbg, o_mo, slab));
                    const auto ri = grab(*dbg, o_idx, uint64_t(T) * cfg_.topk * 4), di = dgrab("idx", uint64_t(T) * cfg_.topk * 4);
                    size_t same = 0;
                    for (size_t i = 0; i < ri.size(); i += 2) same += ri[i] == di[i];
                    std::fprintf(stderr, "  [dpcheck] idx    identical %6.2f%%\n", 100.0 * double(same) / double(ri.size() / 2));
                }
            } else if (direct_ && direct_on_ && direct_->handles_p(a.index, T)) {
                // a prefill chunk's application on the Direct engine (the Synapse work before it must be complete)
                if (!direct_->pending()) stream_.synchronize();
                direct_->queue_p(a.index, hcur, out_buf, out_off, dp_all ? 0 : side, dp_all ? 0 : 1 - side);
            } else if (T == 1 && direct_ && direct_on_ && direct_->handles(a.index)) {
                // consecutive Direct applications run as one chain; the Synapse work before them must be complete
                if (!direct_->pending()) stream_.synchronize();
                direct_->queue(a.index, side, hcur, out_buf, out_off);
            } else if (dverify) {
                if (!direct_->pending()) stream_.synchronize();
                direct_->queue_v(a.index, T, side, hcur, out_buf, out_off);
            } else {
                if (direct_) direct_->flush();
                bind_bf16_experts(L, b);
                enqueue(layer_recipe(a, T, full ? Sf : 0), b);  // (only built when used: Direct applications need none)
            }
            if (dump_) {
                if (direct_) direct_->flush();
                stream_.synchronize();
                HostBuffer hb(dev_, slab);
                stream_.download(*out_buf, out_off, hb, 0, slab);
                stream_.synchronize();
                dump_out_.emplace_back(hb.as<uint16_t>(), hb.as<uint16_t>() + slab / 2);
            }
        }
        if (static const char* hs2 = std::getenv("G2E_INK_HASH"); hs2 && *hs2 == '2' && T == kMaxT) {  // (debug) per application
            if (direct_) direct_->flush();
            stream_.synchronize();
            HostBuffer hb(dev_, slab + 64);
            stream_.download(*out_buf, out_off, hb, 0, slab);
            stream_.synchronize();
            uint64_t h = 1469598103934665603ull;
            const uint8_t* d = hb.as<uint8_t>();
            for (uint64_t i = 0; i < slab; ++i) h = (h ^ d[i]) * 1099511628211ull;
            std::fprintf(stderr, "[hash] pos %u app %u: %016llx\n", pos_, a.index, (unsigned long long)h);
            if (direct_ && dp_all) direct_->verify_rplan(a.index);
            if (static const bool hf = std::getenv("G2E_INK_HASH_FIELDS") != nullptr; hf && direct_ && dp_all && pos_ == 0)
                direct_->hash_fields(a.index);
            if (static const bool hk = std::getenv("G2E_INK_HASH_KVS") != nullptr; hk && direct_ && dp_all && pos_ == 0)
                direct_->hash_kvs(a.index);
            if (const char* ha = std::getenv("G2E_INK_HASH_APP"); ha && uint32_t(std::atoi(ha)) == a.index && pos_ == 0 && direct_)
                direct_->hash_scratch(a.index);
        }
        last_out = out_buf;
        last_off = out_off;
        // bookkeeping for the next application
        if (out_buf == &h_[1 - hcur]) hcur = 1 - hcur;
        if (L + 1 == cfg_.loop_end && !last_pass) {
            // end of a pass: block 1 of this pass -> next pass's message source; flip the message sides (a Direct L11
            // application of this pass already wrote the copy)
            const uint32_t l11 = a.index - cfg_.attn_res_block;  // the L11 application of this pass
            if (!(T == 1 && direct_ && direct_on_ && direct_->handles(l11)) &&
                !(verify_ && direct_ && direct_on_ && direct_->handles_v(l11, T)) &&
                !(direct_ && direct_on_ && direct_->handles_p(l11, T))) {
                if (direct_) direct_->flush();
                stream_.copy(blocks_, blk(2), prev_[1 - pside], 0, slab);
            }
            pside = 1 - pside;
        }
    }
    if (direct_) direct_->flush();
    if (static const bool hs = std::getenv("G2E_INK_HASH") != nullptr; hs && T == kMaxT && last_out) {
        // (debug) a hash of the forward's final state, the four blocks and three K rings: run-to-run determinism
        stream_.synchronize();
        auto fnv = [&](const DeviceBuffer& buf, uint64_t off, uint64_t bytes) {
            HostBuffer hb(dev_, bytes + 64);
            stream_.download(buf, off, hb, 0, bytes);
            stream_.synchronize();
            uint64_t h = 1469598103934665603ull;
            const uint8_t* d = hb.as<uint8_t>();
            for (uint64_t i = 0; i < bytes; ++i) h = (h ^ d[i]) * 1099511628211ull;
            return h;
        };
        const uint64_t ro = ring_off(active_slot_);
        std::fprintf(stderr, "[hash] pos %u: h %016llx b1 %016llx b2 %016llx b3 %016llx k0 %016llx k17 %016llx k47 %016llx\n", pos_,
                     (unsigned long long)fnv(*last_out, last_off, slab), (unsigned long long)fnv(blocks_, blk(1), slab),
                     (unsigned long long)fnv(blocks_, blk(2), slab), (unsigned long long)fnv(blocks_, blk(3), slab),
                     (unsigned long long)fnv(kring_[ring_index_[0]], ro, ring_slot_bytes_),
                     (unsigned long long)fnv(kring_[ring_index_[17]], ro, ring_slot_bytes_),
                     (unsigned long long)fnv(kring_[ring_index_[47]], ro, ring_slot_bytes_));
    }
    if (static const char* pe = std::getenv("G2E_INK_DPREFILL_PROF"); pe && direct_ && T == kMaxT) {
        static int runs = 0;
        if (++runs == 2) direct_->profile_stages_p(uint32_t(std::atoi(pe)));  // (the second chunk: rings written)
    }
    // the Direct plans wrote the rings / linear caches themselves
    const bool direct_all = T == 1 && direct_ && direct_on_ && direct_->handles_all();
    const bool skip_commit = direct_all || verify_ || dp_all;  // verify: accept() commits the accepted rows
    // commit the new K/V rows: full layers rows [pos, pos + T); rings the last min(T, W) rows at slot = position % W
    static const bool prof = std::getenv("G2E_INK_PROFILE") != nullptr;
    std::unique_ptr<Event> c0, c1;
    if (prof) { c0 = std::make_unique<Event>(dev_); c1 = std::make_unique<Event>(dev_); c0->record(stream_); }
    static const bool dma_commit = std::getenv("G2E_INK_DMA_COMMIT") != nullptr;
    if (skip_commit) {
    } else if (!dma_commit) {
        const uint64_t ro = ring_off(active_slot_);
        std::vector<Binding> cb = {{"rows_ring", &tables_, tab_.rows}, {"rows_lin", &tables_, tab_.rows + kMaxT * 4}};
        for (const auto& a : apps_) {
            const auto ai = std::to_string(a.index);
            cb.push_back({"kv" + ai, &kvscratch_, uint64_t(a.index) * 2 * KVD * kMaxT * 2});
            cb.push_back({"kr" + ai, &kring_[ring_index_[a.index]], ro});
            cb.push_back({"vr" + ai, &vring_[ring_index_[a.index]], ro});
            if (cfg_.full[a.layer]) {
                cb.push_back({"kc" + ai, &kcache_[kv_index_[a.index]], 0});
                cb.push_back({"vc" + ai, &vcache_[kv_index_[a.index]], 0});
            }
        }
        commit_recipe(T).enqueue(stream_, cb);
    } else {
    const uint64_t ro = ring_off(active_slot_);
    for (const auto& a : apps_) {
        const uint64_t src0 = uint64_t(a.index) * 2 * KVD * kMaxT * 2;
        for (int kv = 0; kv < 2; ++kv) {
            for (uint64_t h = 0; h < KV; ++h) {
                const uint64_t src = src0 + (uint64_t(kv) * KV + h) * T * D * 2;
                if (cfg_.full[a.layer]) {
                    const int s = kv_index_[a.index];
                    const DeviceBuffer& c = kv == 0 ? kcache_[s] : vcache_[s];
                    stream_.copy(kvscratch_, src, c, (h * uint64_t(cap_) + pos_) * D * 2, uint64_t(T) * D * 2);
                }
                {
                    const int s = ring_index_[a.index];
                    const DeviceBuffer& ring = kv == 0 ? kring_[s] : vring_[s];
                    uint64_t u = T > W ? T - W : 0;
                    while (u < T) {
                        const uint64_t slot = (pos_ + u) % W;
                        const uint64_t n = std::min<uint64_t>(T - u, W - slot);
                        stream_.copy(kvscratch_, src + u * D * 2, ring, ro + (h * W + slot) * D * 2, n * D * 2);
                        u += n;
                    }
                }
            }
        }
    }
    }
    if (prof) {
        c1->record(stream_);
        c1->synchronize();
        auto& p = profile_["kv_commit_dma"];
        p.first += 1;
        p.second += Event::elapsed_ms(*c0, *c1);
    }
    const auto t2 = clk::now();
    if (aux_on_ && (!verify_ || pad_n_) && T > 1) {
        // the taps 48 of every row (prefill): final depth attention + norm, no LM head
        std::vector<Binding> b = {{"b0", &blocks_, blk(0)}, {"b1", &blocks_, blk(1)}, {"b2", &blocks_, blk(2)}, {"b3", &blocks_, blk(3)},
                                  {"h", last_out, last_off}, w_.bind("final_q", "final_q"), w_.bind("final_norm", "final_norm"),
                                  {"xn", &aux_, 3 * uint64_t(kMaxT) * H * 2}};
        headx_recipe(T).enqueue(stream_, b);
    }
    if (want_logits && logits) {
        // the last row (verification: every row): final depth attention over [blocks 0..3, h]
        // (padded prefill tail: the last real row)
        const uint32_t hr = verify_ && !pad_n_ ? T : 1;
        const uint64_t lr = pad_n_ ? uint64_t(pad_n_ - 1) * row : verify_ ? 0 : uint64_t(T - 1) * row;
        std::vector<Binding> b = {{"b0", &blocks_, blk(0) + lr}, {"b1", &blocks_, blk(1) + lr}, {"b2", &blocks_, blk(2) + lr},
                                  {"b3", &blocks_, blk(3) + lr}, {"h", last_out, last_off + lr},
                                  w_.bind("final_q", "final_q"), w_.bind("final_norm", "final_norm"),
                                  w_.bind("lm_head", "lm_head"), {"logits", &logits_, 0}};
        const bool cap = aux_on_ && (T == 1 || (verify_ && !pad_n_));
        if (cap) b.push_back({"xn", &aux_, 3 * uint64_t(kMaxT) * H * 2});
        head_recipe(hr, cap).enqueue(stream_, b);
        stream_.download(logits_, 0, staging_, 0, uint64_t(hr) * cfg_.vocab * 4);
        stream_.synchronize();
        logits->assign(staging_.as<float>(), staging_.as<float>() + uint64_t(hr) * cfg_.vocab);
        if (aux_on_ && T == 1) {
            HostBuffer hb(dev_, 4 * H * 2);
            for (int k = 0; k < 4; ++k) stream_.download(aux_, uint64_t(k) * kMaxT * H * 2, hb, uint64_t(k) * H * 2, H * 2);
            stream_.synchronize();
            aux_host_.insert(aux_host_.end(), hb.as<uint16_t>(), hb.as<uint16_t>() + 4 * H);
        }
    } else {
        stream_.synchronize();
    }
    const auto t3 = clk::now();
    if (verify_) {  // accept() commits, picks the conv states and advances
        ver_T_ = T;
        ver_side_ = side;
    } else {
        if (!dp_all) conv_side_ = 1 - side;  // (Direct prefill: in place)
        pos_ += eff_T;
    }
    tap_slot_ = 0;
    timing_.embed = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing_.layers = std::chrono::duration<double, std::milli>(t2 - t1).count();
    timing_.head = std::chrono::duration<double, std::milli>(t3 - t2).count();
    last_ms_ = std::chrono::duration<double, std::milli>(t3 - t0).count();
}

// Two prefill chunks of kMaxT tokens on the Direct engine, interleaved one application apart (InkDirect::run_pair):
// the first in chunk slot 0 (the model's buffers), the second in slot 1. Both chunks' embeddings (Synapse) first,
// then one wave of Direct plans (they commit the K / V rows, update the conv states in place and write the taps),
// then the taps' final norm and, on request, the logits of the last row.
void Inkling2::run_forward_pair(std::span<const int32_t> ta, std::span<const int32_t> tb, bool want_logits, std::vector<float>* logits) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const uint32_t T = kMaxT;
    if (ta.size() != T || tb.size() != T) throw std::invalid_argument("inkling2: a prefill pair needs two full chunks");
    if (pos_ + 2 * T > cap_) throw std::out_of_range("inkling2: context full");
    direct_prev_ = false;
    const uint64_t H = cfg_.hidden, row = H * 2, slab = uint64_t(T) * row;
    stream_.synchronize();  // the staging buffer is rewritten below
    if (conv_side_ != 0) {  // the plans update the conv states in place, in side 0
        stream_.copy(conv_pool_, conv_off_slot(active_slot_, 1), conv_pool_, conv_off_slot(active_slot_, 0),
                     apps_.size() * conv_app_bytes_);
        conv_side_ = 0;
    }
    DeviceBuffer* blk[2] = {&blocks_, &blocks1_};
    DeviceBuffer* hs[2] = {h_, h1_};
    DeviceBuffer* pv[2] = {prev_, prev1_};
    DeviceBuffer* ids[2] = {&ids_, &ids1_};
    DeviceBuffer* aux[2] = {&aux_, &aux1_};
    const std::span<const int32_t> toks[2] = {ta, tb};
    for (int c = 0; c < 2; ++c) {
        const uint32_t pc = pos_ + uint32_t(c) * T;
        direct_->set_tables(c, pc, T);
        std::memcpy(staging_.as<uint8_t>() + tab_.bytes + uint64_t(c) * T * 4, toks[c].data(), T * 4);
        stream_.upload(staging_, tab_.bytes + uint64_t(c) * T * 4, *ids[c], 0, T * 4);
        const Binding b[] = {{"ids", ids[c], 0}, w_.bind("embed", "embed"), w_.bind("embed_norm", "embed_norm"), {"out", blk[c], 0}};
        embed_recipe(T).enqueue(stream_, b);
        for (const auto& in : inject_) {  // image features over the embeddings of the positions they cover
            const uint32_t a = std::max(in.row, pc), e = std::min(in.row + in.rows, pc + T);
            if (a < e) stream_.copy(*in.src, in.off + uint64_t(a - in.row) * row, *blk[c], uint64_t(a - pc) * row, uint64_t(e - a) * row);
        }
    }
    std::erase_if(inject_, [&](const Inject& in) { return in.row + in.rows <= pos_ + 2 * T; });
    // per application: the partial block it reads and where its output goes (the bookkeeping of run_forward)
    std::vector<std::pair<int, uint64_t>> io[2];
    const DeviceBuffer* last_out[2] = {nullptr, nullptr};
    for (int c = 0; c < 2; ++c) {
        int hcur = 0, pside = 0;
        for (const auto& a : apps_) {
            const uint32_t L = a.layer;
            const bool block_end = (L + 1) % cfg_.attn_res_block == 0, last_pass = a.pass + 1 == cfg_.passes;
            const DeviceBuffer* ob;
            uint64_t off = 0;
            if (L + 1 == cfg_.layers) ob = &hs[c][1 - hcur];
            else if (block_end && L + 1 == cfg_.loop_end && !last_pass) { ob = &pv[c][1 - pside]; off = slab; }
            else if (block_end) { ob = blk[c]; off = uint64_t((L + 1) / cfg_.attn_res_block) * slab; }
            else ob = &hs[c][1 - hcur];
            io[c].push_back({hcur, ob->address() + off});
            last_out[c] = ob;
            if (ob == &hs[c][1 - hcur]) hcur = 1 - hcur;
            if (L + 1 == cfg_.loop_end && !last_pass) pside = 1 - pside;
        }
    }
    stream_.synchronize();
    const auto t1 = clk::now();
    direct_->run_pair(pos_, io[0], io[1]);
    const auto t2 = clk::now();
    auto blkoff = [&](uint32_t k) { return uint64_t(k) * slab; };
    if (static const bool hs = std::getenv("G2E_INK_HASH") != nullptr; hs) {  // (debug) as run_forward's: output and blocks per chunk
        auto fnv = [&](const DeviceBuffer& buf, uint64_t off, uint64_t bytes) {
            HostBuffer hb(dev_, bytes + 64);
            stream_.download(buf, off, hb, 0, bytes);
            stream_.synchronize();
            uint64_t h = 1469598103934665603ull;
            const uint8_t* d = hb.as<uint8_t>();
            for (uint64_t i = 0; i < bytes; ++i) h = (h ^ d[i]) * 1099511628211ull;
            return h;
        };
        for (int c = 0; c < 2; ++c)
            std::fprintf(stderr, "[hash] pos %u: h %016llx b1 %016llx b2 %016llx b3 %016llx\n", pos_ + uint32_t(c) * T,
                         (unsigned long long)fnv(*last_out[c], 0, slab), (unsigned long long)fnv(*blk[c], blkoff(1), slab),
                         (unsigned long long)fnv(*blk[c], blkoff(2), slab), (unsigned long long)fnv(*blk[c], blkoff(3), slab));
    }
    if (aux_on_) {  // the taps 48 of every row: final depth attention + norm
        for (int c = 0; c < 2; ++c) {
            std::vector<Binding> b = {{"b0", blk[c], blkoff(0)}, {"b1", blk[c], blkoff(1)}, {"b2", blk[c], blkoff(2)}, {"b3", blk[c], blkoff(3)},
                                      {"h", last_out[c], 0}, w_.bind("final_q", "final_q"), w_.bind("final_norm", "final_norm"),
                                      {"xn", aux[c], 3 * uint64_t(kMaxT) * H * 2}};
            headx_recipe(T).enqueue(stream_, b);
        }
    }
    if (want_logits && logits) {  // the last row of the second chunk
        const uint64_t lr = uint64_t(T - 1) * row;
        std::vector<Binding> b = {{"b0", blk[1], blkoff(0) + lr}, {"b1", blk[1], blkoff(1) + lr}, {"b2", blk[1], blkoff(2) + lr},
                                  {"b3", blk[1], blkoff(3) + lr}, {"h", last_out[1], lr},
                                  w_.bind("final_q", "final_q"), w_.bind("final_norm", "final_norm"),
                                  w_.bind("lm_head", "lm_head"), {"logits", &logits_, 0}};
        head_recipe(1, false).enqueue(stream_, b);
        stream_.download(logits_, 0, staging_, 0, uint64_t(cfg_.vocab) * 4);
        stream_.synchronize();
        logits->assign(staging_.as<float>(), staging_.as<float>() + cfg_.vocab);
    } else {
        stream_.synchronize();
    }
    const auto t3 = clk::now();
    pos_ += 2 * T;
    timing_.embed = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing_.layers = std::chrono::duration<double, std::milli>(t2 - t1).count();
    timing_.head = std::chrono::duration<double, std::milli>(t3 - t2).count();
    last_ms_ = std::chrono::duration<double, std::milli>(t3 - t0).count();
    if (std::getenv("G2E_INK_PAIR_PROF"))
        std::fprintf(stderr, "[fwd-pair] pos=%u embed=%.2f ms layers=%.2f ms head=%.2f ms total=%.2f ms\n",
                     pos_ - 2 * T, timing_.embed, timing_.layers, timing_.head, last_ms_);
}

namespace {
uint64_t kMaxQBlocks() {  // G2E_INK_QBLOCKS: query blocks per full-layer attention (1: the old chunk cap)
    static const uint64_t n = [] { const char* e = std::getenv("G2E_INK_QBLOCKS"); return e ? uint64_t(std::atoi(e)) : 16ull; }();
    return n;
}
}  // namespace

std::vector<float> Inkling2::forward(std::span<const int32_t> tokens, bool want_logits) {
    std::vector<float> logits;
    size_t at = 0;
    static const uint32_t chunk = [] {
        const char* e = std::getenv("G2E_INK_CHUNK");
        return e ? uint32_t(std::atoi(e)) : 1024u;
    }();
    while (at < tokens.size()) {
        const size_t rem = tokens.size() - at;
        // full layers attend in query blocks (attn_block) at long contexts; at most 16 blocks per chunk (Synapse only:
        // Direct FlashAttention tiles far keys in 48 MiB SRAM and handles T = 1024 at any context length)
        uint32_t cap_t = chunk;
        const bool direct_full_chunk = direct_ && direct_on_ && !verify_ && !dump_ && direct_->handles_all_p(chunk, pos_);
        if (!direct_full_chunk) {
            const uint32_t p = pos_;
            const uint32_t far = far_bucket(p >= cfg_.window ? p - cfg_.window : 0) + chunk;
            while (cap_t > 16 && uint64_t(cap_t) * (far + 2048) > (8ull << 20) * kMaxQBlocks()) cap_t /= 2;
        }
        // two full chunks on the Direct engine, interleaved one application apart (G2E_INK_DPREFILL_PAIR=0: singly).
        // The two chunk slots' plans wait on different QMAN fences (DirectPlan::set_lane); the results are those of
        // the chunks run singly, bit for bit.
        static const bool pair_on = [] { const char* e = std::getenv("G2E_INK_DPREFILL_PAIR"); return !e || *e != '0'; }();
        if (pair_on && direct_ && direct_on_ && !verify_ && !dump_ && cap_t == kMaxT && chunk == kMaxT && rem >= 2 * kMaxT &&
            blocks1_.address() && !std::getenv("G2E_INK_DPCHECK") && direct_->handles_all_p(kMaxT, pos_) &&
            direct_->handles_all_p(kMaxT, pos_ + kMaxT)) {
            const bool last = at + 2 * kMaxT == tokens.size();
            const uint32_t p0 = pos_;
            run_forward_pair(tokens.subspan(at, kMaxT), tokens.subspan(at + kMaxT, kMaxT), want_logits && last, last ? &logits : nullptr);
            if (aux_on_ && tap_hook_) {
                tap_slot_ = 0;
                tap_hook_(p0, kMaxT);
                tap_slot_ = 1;
                tap_hook_(p0 + kMaxT, kMaxT);
                tap_slot_ = 0;
            }
            at += 2 * kMaxT;
            continue;
        }
        uint32_t n = uint32_t(std::min<size_t>(rem, cap_t));
        // Zero-Synapse short & tail Direct prefill (G2E_INK_DPREFILL_TAIL=1 default):
        //   - n in 2..4: run Direct T = 1 decode steps (~8 ms/tok, 0 Synapse compiles)
        //   - n in 5..1023: pad to kMaxT = 1024 and run Direct prefill with valid_T = n (masked MoE, exact conv/KV commit)
        static const bool direct_tail = [] { const char* e = std::getenv("G2E_INK_DPREFILL_TAIL"); return !e || *e != '0'; }();
        if (direct_tail && n > 1 && n < kMaxT && direct_ && direct_on_ && !verify_ && !dump_ && !std::getenv("G2E_INK_DPCHECK")) {
            if (n <= 4 && direct_->handles_all()) {
                n = 1;  // step 1 token at a time via Direct decode (16..32 ms total, zero Synapse compiles)
            } else if (cap_t == kMaxT && direct_->handles_all_p(kMaxT, pos_)) {
                std::vector<int32_t> padded(tokens.begin() + at, tokens.begin() + at + n);
                padded.resize(kMaxT, padded.back());
                const bool last = at + n == tokens.size();
                const uint32_t p0 = pos_;
                pad_n_ = n;
                try {
                    run_forward(padded, want_logits && last, last ? &logits : nullptr);
                } catch (...) {
                    pad_n_ = 0;
                    throw;
                }
                pad_n_ = 0;
                if (aux_on_ && tap_hook_) tap_hook_(p0, n);
                at += n;
                continue;
            }
        }
        // a tail that is not a power of two: one forward padded to the next power of two in verification mode (the
        // padding rows are not committed, the conv states are taken after the last real row), instead of up to ten
        // power-of-two pieces (each small forward costs ~30 ms). G2E_INK_PAD_TAIL=0: the pieces.
        static const bool pad_tail = [] { const char* e = std::getenv("G2E_INK_PAD_TAIL"); return !e || *e != '0'; }();
        if (n < cap_t && pad_tail && (n & (n - 1)) != 0) {
            uint32_t P = 1;
            while (P < n) P *= 2;
            std::vector<int32_t> padded(tokens.begin() + at, tokens.begin() + at + n);
            padded.resize(P, padded.back());
            ensure_hist();
            const bool last = at + n == tokens.size();
            const uint32_t p0 = pos_;
            verify_ = true;
            pad_n_ = n;
            try {
                run_forward(padded, want_logits && last, last ? &logits : nullptr);
            } catch (...) {
                verify_ = false;
                pad_n_ = 0;
                throw;
            }
            verify_ = false;
            pad_n_ = 0;
            accept(n);
            if (aux_on_ && tap_hook_) tap_hook_(p0, n);
            at += n;
            continue;
        }
        if (n < cap_t) {  // the largest power of two <= rem
            n = 1;
            while (n * 2 <= rem) n *= 2;
        }
        const bool last = at + n == tokens.size();
        const uint32_t p0 = pos_;
        run_forward(tokens.subspan(at, n), want_logits && last, last ? &logits : nullptr);
        if (aux_on_ && tap_hook_) tap_hook_(p0, n);
        at += n;
    }
    return logits;
}

int32_t Inkling2::forward_argmax(std::span<const int32_t> tokens) {
    static const bool dec_dbg = std::getenv("G2E_INK_DIRECT_CHECK") || std::getenv("G2E_INK_DIRECT_PROF") ||
                                std::getenv("G2E_INK_DIRECT_NOCHAIN");
    if (tokens.size() == 1 && !pad_n_ && !dec_dbg && direct_ && direct_on_ && !verify_ && !dump_ && inject_.empty() &&
        direct_->handles_full_decode()) {
        const uint32_t p0 = pos_;
        const int32_t tok = run_decode_direct(tokens[0], /*want_logits=*/false, /*logits=*/nullptr, /*want_argmax=*/true);
        if (aux_on_ && tap_hook_) tap_hook_(p0, 1);
        return tok;
    }
    const auto lg = forward(tokens, true);
    return int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
}

std::string Inkling2::engine_summary() const {
    if (!direct_ || direct_->summary() == "off")
        return "Synapse graphs only: no speculation, no batched decode (the Direct engine needs G2E_INK_DIRECT=15 "
               "G2E_INK_DIRECT_SRAM=1 G2E_INK_DPREFILL=15 and tools/scal-g2e-ink.json, see docs/inkling2.md)";
    uint32_t vt = 0;
    for (uint32_t T = 2; T <= InkDirect::kVerifyMaxT; ++T)
        if (can_verify(T)) vt = T;
    return "Direct, " + direct_->summary() + ", verification " + (vt ? "up to " + std::to_string(vt) + " tokens" : std::string("off"));
}

bool Inkling2::direct_engine() const { return direct_ && direct_->has_runtime(); }

bool Inkling2::direct_full_decode() const { return direct_ && direct_->handles_full_decode(); }

bool Inkling2::can_decode_batch() const {
    return direct_ && direct_on_ && !verify_ && !dump_ && !aux_on_ && inject_.empty() &&
           direct_->handles_decode_batch();
}

Inkling2::DecodeBatchOut Inkling2::decode_batch(std::span<const uint32_t> slots, std::span<const int32_t> tokens,
                                                std::span<const uint8_t> want_logits) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const size_t B = slots.size();
    if (B < 2 || B > 8 || tokens.size() != B || want_logits.size() != B)
        throw std::invalid_argument("inkling2: decode_batch requires 2..8 slots and matching tokens/want_logits");
    if (!can_decode_batch()) throw std::logic_error("inkling2: decode_batch not available");
    uint64_t seen = 0;
    for (uint32_t s : slots) {
        if (s >= seqs_.size()) throw std::out_of_range("inkling2: decode_batch bad slot");
        if (seen & (1ull << s)) throw std::invalid_argument("inkling2: decode_batch duplicate slot");
        seen |= (1ull << s);
        if (slot_position(s) + 1 > slot_capacity(s)) throw std::out_of_range("inkling2: decode_batch context full");
    }
    if (direct_->pending()) direct_->flush();
    if (!direct_prev_) stream_.synchronize();
    DecodeBatchOut out;
    direct_->run_decode_batch(slots, tokens, want_logits, out.argmax, out.logits);
    for (uint32_t s : slots) {
        if (s == active_slot_) {
            pos_ += 1;
            conv_side_ = 1 - conv_side_;
        } else {
            seqs_[s].pos += 1;
            seqs_[s].conv_side = 1 - seqs_[s].conv_side;
        }
    }
    direct_prev_ = true;
    last_ms_ = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    return out;
}

}  // namespace g2e
