#include "g2e/h3_dit.hpp"
#include "g2e/numeric.hpp"

#include <nlohmann/json.hpp>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

namespace g2e {

namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void set_output(GraphBuilder& g, const std::string& name, TensorId src) {
    const auto& d = g.desc(src);
    const DType dt = d.dtype;
    const auto sh = d.shape;
    const auto out = g.input(name, dt, sh);
    g.op("identity", {src}, {out});
}

// Dynamic per-tensor FP8 E4M3 quantization for 3D [H, L, D] or 4D [1, H, L, D] BF16 -> (codes F8, d_scale [1] F32).
std::pair<TensorId, TensorId> quantize_tensor_fp8_nd(Ops& o, TensorId x, float max_val, const std::string& name) {
    const auto sh = o.shape(x);
    const size_t r = sh.size();
    const uint64_t H = sh[r - 3], L = sh[r - 2], D = sh[r - 1];
    const auto ax = o.unary("abs", o.reshape(x, {H * L, D}, name + "_2d"), name + "_abs");
    const auto m1 = o.reshape(o.reduce("max", ax, 1, name + "_m1"), {H, L}, name + "_m1r");
    const auto m2 = o.reduce("max", m1, 1, name + "_m2");
    const auto peak = o.cast(o.reshape(o.reduce("max", m2, 0, name + "_m3"), {1}, name + "_pk"), DType::F32, name + "_pkf");
    const auto scale = o.mul(o.add(peak, o.constant(1e-8f, {1}, name + "_eps", DType::F32), name + "_pe"),
                             o.constant(1.0f / max_val, {1}, name + "_imax", DType::F32), name + "_sc");
    const auto inv = o.unary("reciprocal", scale, name + "_inv");
    const auto codes = o.temp(name + "_f8", DType::F8, sh);
    ns_ConvertToFp8::ParamsV2 cp{};
    cp.round_mode = CAST_ROUND_HALF_NE;
    o.graph().op("convert_to_fp8_bf16", {x, inv}, {codes}, &cp, sizeof(cp));
    return {codes, scale};
}

std::pair<TensorId, TensorId> quantize_tensor_fp8_4d(Ops& o, TensorId x, const std::string& name) {
    return quantize_tensor_fp8_nd(o, x, 240.0f, name);
}

// Native Gaudi2 FP8 Fused Flash Attention (sdpa_recomp_fwd_hf8):
// Runs BOTH Q@K^T and P@V in 865 TFLOP/s FP8 MME, tiled in SRAM with zero HBM score spill!
// q, k, v: [1, H, L, D] BF16 -> out [1, H, L, D] BF16.
TensorId sdpa_bidir_fp8(Ops& o, TensorId q, TensorId k, TensorId v, float scale, bool fast_softmax,
                        bool allow_kv_split, const std::string& name) {
    const auto qsh = o.shape(q);
    const uint64_t H = qsh[1], Q = qsh[2], D = qsh[3];
    const uint64_t K = o.shape(k)[2];
    const uint64_t split = (allow_kv_split && K >= 4096 && (K % 128 != 0)) ? ((K / 128) * 128) : 0;

    const auto [q8, dsq] = quantize_tensor_fp8_4d(o, q, name + "_q");
    const auto [k8, dsk] = quantize_tensor_fp8_4d(o, k, name + "_k");
    const auto [v8, dsv] = quantize_tensor_fp8_4d(o, v, name + "_v");
    const auto qs_s = o.constant(128.0f, {1}, name + "_qs128", DType::F32);
    const auto ds_s = o.constant(1.0f / 128.0f, {1}, name + "_ds128", DType::F32);

    if (split == 0) {
        const auto out = o.temp(name, DType::BF16, qsh);
        ns_Sdpa::ParamsV7 fp{};
        fp.scale = scale;
        fp.is_causal = false;
        fp.is_inference = true;
        fp.softmax_mode = fast_softmax ? SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C : SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
        fp.flags = 0x4002f00;  // d_scale_q/k/v, q_scale_s, d_scale_s present
        fp.wl = fp.wr = -1;
        std::vector<TensorId> ins = {q8, k8, v8, kNoTensor, kNoTensor,
                                     dsq, dsk, dsv, qs_s, kNoTensor, ds_s, kNoTensor, kNoTensor};
        o.graph().op("sdpa_recomp_fwd_hf8", std::span<const TensorId>(ins), std::span<const TensorId>(&out, 1),
                     &fp, sizeof(fp));
        return out;
    }

    const auto ks = o.split(k8, 2, {split, K - split}, name + "_ksp");
    const auto vs = o.split(v8, 2, {split, K - split}, name + "_vsp");
    auto run_stats = [&](TensorId kk, TensorId vv, const std::string& pfx) {
        const auto out = o.temp(pfx + "_o", DType::BF16, {1, H, Q, D});
        const auto m = o.temp(pfx + "_m", DType::BF16, {1, H, Q, 1});
        const auto linv = o.temp(pfx + "_l", DType::BF16, {1, H, Q, 1});
        ns_Sdpa::ParamsV7 fp{};
        fp.scale = scale;
        fp.is_causal = false;
        fp.is_inference = false;
        fp.softmax_mode = fast_softmax ? SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C : SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
        fp.flags = 0x4002f00;
        fp.wl = fp.wr = -1;
        std::vector<TensorId> ins = {q8, kk, vv, kNoTensor, kNoTensor,
                                     dsq, dsk, dsv, qs_s, kNoTensor, ds_s, kNoTensor, kNoTensor};
        const TensorId outs[3] = {out, m, linv};
        o.graph().op("sdpa_recomp_fwd_hf8", std::span<const TensorId>(ins), std::span<const TensorId>(outs, 3),
                     &fp, sizeof(fp));
        return std::tuple<TensorId, TensorId, TensorId>{out, m, linv};
    };
    const auto [o1, m1, l1] = run_stats(ks[0], vs[0], name + "_h1");
    const auto [o2, m2, l2] = run_stats(ks[1], vs[1], name + "_h2");

    const auto m1f = o.cast(m1, DType::F32, name + "_m1f");
    const auto m2f = o.cast(m2, DType::F32, name + "_m2f");
    const auto l1f = o.cast(l1, DType::F32, name + "_l1f");
    const auto l2f = o.cast(l2, DType::F32, name + "_l2f");
    const auto mm = o.binary("max", m1f, m2f, name + "_mm");
    const auto w1 = o.div(o.unary("exp", o.sub(m1f, mm, name + "_d1"), name + "_e1"), l1f, name + "_w1");
    const auto w2 = o.div(o.unary("exp", o.sub(m2f, mm, name + "_d2"), name + "_e2"), l2f, name + "_w2");
    const auto share = o.cast(o.div(w2, o.add(w1, w2, name + "_ws"), name + "_shf"), DType::BF16, name + "_sh");
    return o.add(o1, o.mul(o.sub(o2, o1, name + "_diff"), share, name + "_upd"), name);
}

// Non-causal SDPA via Synapse sdpa_recomp_fwd_bf16.
// q, k, v: [1, H, L, D] BF16 -> out [1, H, L, D] BF16.
// When kv_split=true and L >= 4096 and L % 128 != 0, splits K/V at (L/128)*128 and merges exact softmax stats
// (matches vllm_gaudi/omni/attention.py _split_kv_forward: ~30% faster on Gaudi2).
TensorId sdpa_bidir_bf16(Ops& o, TensorId q, TensorId k, TensorId v, float scale, bool fast_softmax,
                         bool allow_kv_split, const std::string& name) {
    const auto qsh = o.shape(q);
    const uint64_t H = qsh[1], Q = qsh[2], D = qsh[3];
    const uint64_t K = o.shape(k)[2];
    const uint64_t split = (allow_kv_split && K >= 4096 && (K % 128 != 0)) ? ((K / 128) * 128) : 0;

    if (split == 0) {
        const auto out = o.temp(name, DType::BF16, qsh);
        ns_Sdpa::ParamsV7 p{};
        p.scale = scale;
        p.is_causal = false;
        p.is_inference = true;
        p.softmax_mode = fast_softmax ? SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C : SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
        p.wl = p.wr = -1;
        std::vector<TensorId> ins = {q, k, v, kNoTensor, kNoTensor};
        for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
        o.graph().op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(&out, 1),
                     &p, sizeof(p));
        return out;
    }

    // Split KV into aligned head [0..split) and short tail [split..K)
    const auto ks = o.split(k, 2, {split, K - split}, name + "_ksp");
    const auto vs = o.split(v, 2, {split, K - split}, name + "_vsp");
    auto run_stats = [&](TensorId kk, TensorId vv, const std::string& pfx) {
        const auto out = o.temp(pfx + "_o", DType::BF16, {1, H, Q, D});
        const auto m = o.temp(pfx + "_m", DType::BF16, {1, H, Q, 1});
        const auto linv = o.temp(pfx + "_l", DType::BF16, {1, H, Q, 1});
        ns_Sdpa::ParamsV7 p{};
        p.scale = scale;
        p.is_causal = false;
        p.is_inference = false;
        p.softmax_mode = fast_softmax ? SdpaSoftmaxMode_t::SDPA_SOFTMAX_HF8_1C : SdpaSoftmaxMode_t::SDPA_DEFAULT_SOFTMAX;
        p.wl = p.wr = -1;
        std::vector<TensorId> ins = {q, kk, vv, kNoTensor, kNoTensor};
        for (int i = 0; i < 7; ++i) ins.push_back(kNoTensor);
        const TensorId outs[3] = {out, m, linv};
        o.graph().op("sdpa_recomp_fwd_bf16", std::span<const TensorId>(ins), std::span<const TensorId>(outs, 3),
                     &p, sizeof(p));
        return std::tuple<TensorId, TensorId, TensorId>{out, m, linv};
    };
    const auto [o1, m1, l1] = run_stats(ks[0], vs[0], name + "_h1");
    const auto [o2, m2, l2] = run_stats(ks[1], vs[1], name + "_h2");

    const auto m1f = o.cast(m1, DType::F32, name + "_m1f");
    const auto m2f = o.cast(m2, DType::F32, name + "_m2f");
    const auto l1f = o.cast(l1, DType::F32, name + "_l1f");
    const auto l2f = o.cast(l2, DType::F32, name + "_l2f");
    const auto mm = o.binary("max", m1f, m2f, name + "_mm");
    const auto w1 = o.div(o.unary("exp", o.sub(m1f, mm, name + "_d1"), name + "_e1"), l1f, name + "_w1");
    const auto w2 = o.div(o.unary("exp", o.sub(m2f, mm, name + "_d2"), name + "_e2"), l2f, name + "_w2");
    const auto share = o.cast(o.div(w2, o.add(w1, w2, name + "_ws"), name + "_shf"), DType::BF16, name + "_sh");
    return o.add(o1, o.mul(o.sub(o2, o1, name + "_diff"), share, name + "_upd"), name);
}

// Slice [0..len) along axis 0 of a 2D or 3D tensor.
TensorId slice_prefix_axis0(Ops& o, TensorId x, uint64_t len, const std::string& name) {
    const auto sh = o.shape(x);
    if (len == sh[0]) return x;
    return o.split(x, 0, {len, sh[0] - len}, name)[0];
}

// Pad along axis 0 from `used` to `seq_len` with zeros.
TensorId zero_pad_axis0(Ops& o, TensorId x, uint64_t seq_len, const std::string& name) {
    const auto sh = o.shape(x);
    if (sh[0] == seq_len) return x;
    Shape z_sh = sh;
    z_sh[0] = seq_len - sh[0];
    const auto z = o.constant(0.0f, z_sh, name + "_z", o.dtype(x));
    return o.concat({x, z}, 0, name);
}

// Apply MiniMax-H3 partial 3D RoPE: x [S, H, 128], cos [S, 48], sin [S, 48] (all BF16).
// Rotates first 96 dims using half-split pairs (d, d+48), passes dims 96..127 unchanged.
TensorId apply_h3_rope(Ops& o, TensorId x, TensorId cos_tab, TensorId sin_tab, uint32_t rot_dim,
                       const std::string& name) {
    const auto sh = o.shape(x);
    const uint64_t S = sh[0], D = sh[2], half = rot_dim / 2, pass = D - rot_dim;
    const auto parts = o.split(x, 2, {half, half, pass}, name + "_sp");
    const auto c = o.reshape(cos_tab, {S, 1, half}, name + "_c");
    const auto s = o.reshape(sin_tab, {S, 1, half}, name + "_s");
    const auto r0 = o.sub(o.mul(parts[0], c, name + "_p0c"), o.mul(parts[1], s, name + "_p1s"), name + "_r0");
    const auto r1 = o.add(o.mul(parts[1], c, name + "_p1c"), o.mul(parts[0], s, name + "_p0s"), name + "_r1");
    return o.concat({r0, r1, parts[2]}, 2, name);
}

// Choose (h_sub, q_chunk) for split-chunk max-free attention so every binding offset is 4KiB-aligned
// and peak score+prob workspace stays bounded (~4-6 GiB even on 15s S=109,000!).
static std::pair<uint32_t, uint32_t> choose_mf_tiling(uint32_t NH, uint32_t U) {
    uint32_t h_sub = (U <= 4096) ? NH : 14u;
    uint32_t q_chunk = (U <= 4096) ? U : (U <= 45056 ? 2048u : 1024u);
    if (const char* e = std::getenv("G2E_H3_MF_HSUB")) {
        uint32_t v = uint32_t(std::stoul(e));
        if (v > 0 && NH % v == 0) h_sub = v;
    }
    if (const char* e = std::getenv("G2E_H3_MF_CHUNK")) {
        uint32_t v = uint32_t(std::stoul(e));
        if (v > 0) q_chunk = std::min(v, U);
    }
    return {h_sub, q_chunk};
}

// Helper for safetensors parsing in C++ (LoRA loader)
struct SafeTensorEntry {
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t off0 = 0, off1 = 0;
};

void load_bf16_scaled(std::ifstream& fh, uint64_t data_base, const SafeTensorEntry& e,
                      std::vector<uint16_t>& dst, float mult, bool swap_halves = false) {
    const uint64_t n_bytes = e.off1 - e.off0;
    const uint64_t n_elems = n_bytes / (e.dtype == "F32" ? 4 : 2);
    dst.resize(n_elems);
    fh.seekg(std::streamoff(data_base + e.off0));
    if (e.dtype == "BF16") {
        fh.read(reinterpret_cast<char*>(dst.data()), std::streamsize(n_bytes));
        if (!fh) throw std::runtime_error("H3LoraAdapter: failed reading BF16 tensor");
        if (mult != 1.0f) {
            for (uint64_t i = 0; i < n_elems; ++i) dst[i] = to_bf16(from_bf16(dst[i]) * mult);
        }
    } else if (e.dtype == "F32") {
        std::vector<float> tmp(n_elems);
        fh.read(reinterpret_cast<char*>(tmp.data()), std::streamsize(n_bytes));
        if (!fh) throw std::runtime_error("H3LoraAdapter: failed reading F32 tensor");
        for (uint64_t i = 0; i < n_elems; ++i) dst[i] = to_bf16(tmp[i] * mult);
    } else {
        throw std::runtime_error("H3LoraAdapter: unsupported LoRA tensor dtype " + e.dtype);
    }
    if (swap_halves) {
        // [2*F, R] -> swap [up, gate] to [gate, up]
        const uint64_t half = n_elems / 2;
        std::vector<uint16_t> sw(n_elems);
        std::memcpy(sw.data(), dst.data() + half, half * sizeof(uint16_t));
        std::memcpy(sw.data() + half, dst.data(), half * sizeof(uint16_t));
        dst.swap(sw);
    }
}

uint64_t align4k(uint64_t x) { return (x + 4095ull) & ~4095ull; }

float load_scalar_f32(std::ifstream& fh, uint64_t data_base, const SafeTensorEntry& e) {
    fh.seekg(std::streamoff(data_base + e.off0));
    if (e.dtype == "F32") {
        float v = 0.0f;
        fh.read(reinterpret_cast<char*>(&v), 4);
        if (!fh) throw std::runtime_error("H3LoraAdapter: failed reading F32 scalar");
        return v;
    } else if (e.dtype == "BF16") {
        uint16_t u = 0;
        fh.read(reinterpret_cast<char*>(&u), 2);
        if (!fh) throw std::runtime_error("H3LoraAdapter: failed reading BF16 scalar");
        return from_bf16(u);
    }
    throw std::runtime_error("H3LoraAdapter: unsupported scalar dtype " + e.dtype);
}

}  // namespace

// ------------------------------------------------------------------------------------------------ H3DiTConfig

H3DiTConfig H3DiTConfig::from_pack(const WeightPack& pack) {
    const auto j = nlohmann::json::parse(pack.header_json());
    const auto& c = j.at("config");
    H3DiTConfig cfg;
    cfg.num_layers = j.at("layers").get<uint32_t>();
    cfg.refiner_layers = j.value("refiner_layers", c.value("token_refiner_num_layers", 2u));
    cfg.hidden_size = c.at("hidden_size");
    cfg.num_heads = c.at("num_attention_heads");
    cfg.head_dim = c.at("attention_head_dim");
    cfg.ffn_hidden = c.at("ffn_hidden_size");
    cfg.latents_dim = c.at("latents_dim");
    cfg.audio_latents_dim = c.at("audio_latents_dim");
    const auto ps = c.at("patch_size");
    cfg.patch_t = ps.at(0);
    cfg.patch_h = ps.at(1);
    cfg.patch_w = ps.at(2);
    cfg.text_dim = c.at("text_dim");
    cfg.timestep_input_dim = c.at("timestep_input_dim");
    cfg.time_embed_hidden = c.at("time_embed_hidden_size");
    cfg.time_embed_dim = c.at("time_embed_dim");
    cfg.rope_inv_freq_len = c.at("rope_inv_freq_len");
    cfg.norm_eps = c.value("norm_eps", 1e-5f);
    cfg.qk_norm_eps = c.value("qk_norm_eps", 1e-5f);
    cfg.final_norm_eps = c.value("final_norm_eps", 1e-5f);
    cfg.adaln_bf16 = (j.value("adaln_dtype", "f8") == "bf16");
    cfg.audio_enabled = j.value("audio_enabled", c.value("audio_enabled", true));
    cfg.pdd_steps = j.value("pdd_steps", 0u);
    return cfg;
}

// ------------------------------------------------------------------------------------------------ H3LoraAdapter

H3LoraAdapter::H3LoraAdapter(const Device& device, const Stream& stream, const std::string& safetensors_path,
                             const H3DiTConfig& cfg, float scale)
    : path_(safetensors_path), scale_(scale) {
    const auto t0 = Clock::now();
    std::ifstream fh(safetensors_path, std::ios::binary);
    if (!fh) throw std::runtime_error("H3LoraAdapter: cannot open " + safetensors_path);
    uint64_t hdr_len = 0;
    fh.read(reinterpret_cast<char*>(&hdr_len), 8);
    std::string hdr_str(hdr_len, '\0');
    fh.read(hdr_str.data(), std::streamsize(hdr_len));
    if (!fh) throw std::runtime_error("H3LoraAdapter: truncated header in " + safetensors_path);
    const uint64_t data_base = 8 + hdr_len;

    const auto hdr = nlohmann::json::parse(hdr_str);
    if (hdr.contains("__metadata__") && hdr["__metadata__"].contains("alpha")) {
        alpha_ = std::stof(hdr["__metadata__"]["alpha"].get<std::string>());
    } else {
        alpha_ = 8.0f;  // LightX2V reference default
    }

    std::map<std::string, SafeTensorEntry> entries;
    for (auto it = hdr.begin(); it != hdr.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        SafeTensorEntry e;
        e.dtype = it.value().at("dtype").get<std::string>();
        e.shape = it.value().at("shape").get<std::vector<uint64_t>>();
        e.off0 = it.value().at("data_offsets").at(0).get<uint64_t>();
        e.off1 = it.value().at("data_offsets").at(1).get<uint64_t>();
        entries.emplace(it.key(), std::move(e));
    }

    auto find_key_opt = [&](const std::string& base, const char* side) -> const SafeTensorEntry* {
        std::string k1 = base + ".lora_" + side + ".default.weight";
        if (auto it = entries.find(k1); it != entries.end()) return &it->second;
        std::string k2 = base + ".lora_" + side + ".weight";
        if (auto it = entries.find(k2); it != entries.end()) return &it->second;
        return nullptr;
    };
    auto find_key = [&](const std::string& base, const char* side) -> const SafeTensorEntry& {
        if (const auto* e = find_key_opt(base, side)) return *e;
        throw std::runtime_error("H3LoraAdapter: missing key " + base + ".lora_" + side + ".default.weight");
    };
    bool has_per_mod_alpha = false;
    auto module_mult = [&](const std::string& base, const SafeTensorEntry& ea) -> float {
        const uint32_t r = uint32_t(ea.shape.at(0));
        float mod_alpha = alpha_;
        if (auto it = entries.find(base + ".alpha"); it != entries.end()) {
            mod_alpha = load_scalar_f32(fh, data_base, it->second);
            has_per_mod_alpha = true;
        }
        return scale_ * (mod_alpha / float(r));
    };

    rank_ = uint32_t(find_key("transformer_blocks.0.attn.to_q", "A").shape.at(0));
    if (rank_ == 0) throw std::runtime_error("H3LoraAdapter: invalid rank 0 in " + safetensors_path);
    if (const auto* e = find_key_opt("transformer_blocks.0.adaln_proj.linear", "A")) {
        adaln_rank_ = uint32_t(e->shape.at(0));
    }
    if (const auto* e = find_key_opt("norm_out.linear", "A")) {
        final_adaln_rank_ = uint32_t(e->shape.at(0));
    }

    // Pack all needed blocks (R0..R{refiner-1}, B0..B{num_layers-1}, optional norm_out) into one contiguous host buffer -> HBM arena.
    std::vector<uint8_t> host_arena;
    auto push = [&](const std::vector<uint16_t>& v) -> uint64_t {
        const uint64_t off = align4k(host_arena.size());
        host_arena.resize(off + v.size() * sizeof(uint16_t), 0);
        std::memcpy(host_arena.data() + off, v.data(), v.size() * sizeof(uint16_t));
        return off;
    };

    auto pack_block = [&](const std::string& pfx, bool is_refiner) -> BlockOffsets {
        BlockOffsets bo;
        std::vector<uint16_t> aq, ak, av, bq, bk, bv, tmp;
        const auto& eqa = find_key(pfx + "attn.to_q", "A");
        const auto& eka = find_key(pfx + "attn.to_k", "A");
        const auto& eva = find_key(pfx + "attn.to_v", "A");
        if (eqa.shape.at(0) != rank_ || eka.shape.at(0) != rank_ || eva.shape.at(0) != rank_)
            throw std::runtime_error("H3LoraAdapter: mismatched QKV rank in " + pfx);
        load_bf16_scaled(fh, data_base, eqa, aq, 1.0f);
        load_bf16_scaled(fh, data_base, eka, ak, 1.0f);
        load_bf16_scaled(fh, data_base, eva, av, 1.0f);
        std::vector<uint16_t> qkv_a;
        qkv_a.reserve(aq.size() + ak.size() + av.size());
        qkv_a.insert(qkv_a.end(), aq.begin(), aq.end());
        qkv_a.insert(qkv_a.end(), ak.begin(), ak.end());
        qkv_a.insert(qkv_a.end(), av.begin(), av.end());
        bo.qkv_a = push(qkv_a);

        load_bf16_scaled(fh, data_base, find_key(pfx + "attn.to_q", "B"), bq, module_mult(pfx + "attn.to_q", eqa));
        load_bf16_scaled(fh, data_base, find_key(pfx + "attn.to_k", "B"), bk, module_mult(pfx + "attn.to_k", eka));
        load_bf16_scaled(fh, data_base, find_key(pfx + "attn.to_v", "B"), bv, module_mult(pfx + "attn.to_v", eva));
        std::vector<uint16_t> qkv_b;
        qkv_b.reserve(bq.size() + bk.size() + bv.size());
        qkv_b.insert(qkv_b.end(), bq.begin(), bq.end());
        qkv_b.insert(qkv_b.end(), bk.begin(), bk.end());
        qkv_b.insert(qkv_b.end(), bv.begin(), bv.end());
        bo.qkv_b = push(qkv_b);

        const auto& eoa = find_key(pfx + "attn.to_out.0", "A");
        if (eoa.shape.at(0) != rank_) throw std::runtime_error("H3LoraAdapter: mismatched out rank in " + pfx);
        load_bf16_scaled(fh, data_base, eoa, tmp, 1.0f);
        bo.out_a = push(tmp);
        load_bf16_scaled(fh, data_base, find_key(pfx + "attn.to_out.0", "B"), tmp, module_mult(pfx + "attn.to_out.0", eoa));
        bo.out_b = push(tmp);

        const auto& ef1a = find_key(pfx + "ff.net.0.proj", "A");
        if (ef1a.shape.at(0) != rank_) throw std::runtime_error("H3LoraAdapter: mismatched fc1 rank in " + pfx);
        load_bf16_scaled(fh, data_base, ef1a, tmp, 1.0f);
        bo.fc1_a = push(tmp);
        // LightX2V Turbo stores [up, gate] in ff.net.0.proj.lora_B -> swap halves to [gate, up]
        load_bf16_scaled(fh, data_base, find_key(pfx + "ff.net.0.proj", "B"), tmp,
                         module_mult(pfx + "ff.net.0.proj", ef1a), /*swap_halves=*/true);
        bo.fc1_b = push(tmp);

        const auto& ef2a = find_key(pfx + "ff.net.2", "A");
        if (ef2a.shape.at(0) != rank_) throw std::runtime_error("H3LoraAdapter: mismatched fc2 rank in " + pfx);
        load_bf16_scaled(fh, data_base, ef2a, tmp, 1.0f);
        bo.fc2_a = push(tmp);
        load_bf16_scaled(fh, data_base, find_key(pfx + "ff.net.2", "B"), tmp, module_mult(pfx + "ff.net.2", ef2a));
        bo.fc2_b = push(tmp);

        if (!is_refiner && adaln_rank_ > 0) {
            const auto& eada = find_key(pfx + "adaln_proj.linear", "A");
            if (eada.shape.at(0) != adaln_rank_)
                throw std::runtime_error("H3LoraAdapter: mismatched adaln rank in " + pfx);
            load_bf16_scaled(fh, data_base, eada, tmp, 1.0f);
            bo.adaln_a = push(tmp);
            load_bf16_scaled(fh, data_base, find_key(pfx + "adaln_proj.linear", "B"), tmp,
                             module_mult(pfx + "adaln_proj.linear", eada));
            bo.adaln_b = push(tmp);
        }
        return bo;
    };

    refiner_off_.resize(cfg.refiner_layers);
    for (uint32_t i = 0; i < cfg.refiner_layers; ++i)
        refiner_off_[i] = pack_block("token_refiner.refiner_blocks." + std::to_string(i) + ".", true);

    block_off_.resize(cfg.num_layers);
    for (uint32_t i = 0; i < cfg.num_layers; ++i)
        block_off_[i] = pack_block("transformer_blocks." + std::to_string(i) + ".", false);

    if (final_adaln_rank_ > 0) {
        std::vector<uint16_t> tmp;
        const auto& ef = find_key("norm_out.linear", "A");
        load_bf16_scaled(fh, data_base, ef, tmp, 1.0f);
        final_adaln_a_ = push(tmp);
        load_bf16_scaled(fh, data_base, find_key("norm_out.linear", "B"), tmp, module_mult("norm_out.linear", ef));
        final_adaln_b_ = push(tmp);
    }

    arena_ = std::make_unique<DeviceBuffer>(device, host_arena.size());
    HostBuffer staging(device, host_arena.size());
    std::memcpy(staging.data(), host_arena.data(), host_arena.size());
    stream.upload(staging, 0, *arena_, 0, host_arena.size());
    stream.synchronize();
    std::printf("[H3Lora] loaded %s: rank=%u adaln_rank=%u final_rank=%u alpha=%.1f(%s) scale=%.2f (%.2f MiB in %.0f ms)\n",
                safetensors_path.c_str(), rank_, adaln_rank_, final_adaln_rank_, alpha_,
                has_per_mod_alpha ? "per-mod" : "meta", scale_, double(host_arena.size()) / (1 << 20), ms_since(t0));
}

void H3LoraAdapter::bind_block(std::vector<Binding>& out, bool refiner, uint32_t idx) const {
    const auto& bo = refiner ? refiner_off_.at(idx) : block_off_.at(idx);
    out.push_back({"lora_qkv_a", arena_.get(), bo.qkv_a});
    out.push_back({"lora_qkv_b", arena_.get(), bo.qkv_b});
    out.push_back({"lora_out_a", arena_.get(), bo.out_a});
    out.push_back({"lora_out_b", arena_.get(), bo.out_b});
    out.push_back({"lora_fc1_a", arena_.get(), bo.fc1_a});
    out.push_back({"lora_fc1_b", arena_.get(), bo.fc1_b});
    out.push_back({"lora_fc2_a", arena_.get(), bo.fc2_a});
    out.push_back({"lora_fc2_b", arena_.get(), bo.fc2_b});
}

void H3LoraAdapter::bind_adaln(std::vector<Binding>& out, uint32_t idx) const {
    if (adaln_rank_ == 0) return;
    const auto& bo = block_off_.at(idx);
    out.push_back({"lora_adaln_a", arena_.get(), bo.adaln_a});
    out.push_back({"lora_adaln_b", arena_.get(), bo.adaln_b});
}

void H3LoraAdapter::bind_final_adaln(std::vector<Binding>& out) const {
    if (final_adaln_rank_ == 0) return;
    out.push_back({"lora_adaln_a", arena_.get(), final_adaln_a_});
    out.push_back({"lora_adaln_b", arena_.get(), final_adaln_b_});
}

Binding H3LoraAdapter::bind_one(const std::string& name, bool refiner, uint32_t idx, int which) const {
    const auto& bo = refiner ? refiner_off_.at(idx) : block_off_.at(idx);
    uint64_t off = 0;
    switch (which) {
        case 0: off = bo.qkv_a; break;
        case 1: off = bo.qkv_b; break;
        case 2: off = bo.out_a; break;
        case 3: off = bo.out_b; break;
        case 4: off = bo.fc1_a; break;
        case 5: off = bo.fc1_b; break;
        case 6: off = bo.fc2_a; break;
        case 7: off = bo.fc2_b; break;
    }
    return {name, arena_.get(), off};
}

// ------------------------------------------------------------------------------------------------ H3PackedLayout

H3PackedLayout H3PackedLayout::build(uint32_t text_len, uint32_t latent_t, uint32_t latent_h, uint32_t latent_w,
                                     uint32_t audio_t, bool include_keyframe_cond,
                                     const std::vector<int>& keyframe_frame_indices, int frame_count,
                                     const std::vector<int32_t>& prefix_token_tags) {
    constexpr double kInterp = 32.0;
    constexpr double kFrameRescale = 5.0 / 3.0;
    constexpr int kFramePerToken[5] = {1, 4, 4, 4, 4};

    H3PackedLayout L;
    L.text_len = text_len;
    L.latent_t = latent_t;
    L.latent_h = latent_h;
    L.latent_w = latent_w;
    L.audio_t = audio_t;
    L.audio_channel = 2;

    const uint32_t ph = latent_h / 2, pw = latent_w / 2;
    const uint32_t frame_rows = ph * pw;
    const uint32_t n_cond_frames = include_keyframe_cond ? uint32_t(keyframe_frame_indices.size()) : 0u;
    L.cond_rows = n_cond_frames * frame_rows;
    L.video_rows = latent_t * frame_rows;
    L.img_rows = L.cond_rows + L.video_rows;
    L.audio_rows = audio_t * L.audio_channel;
    L.used_len = text_len + L.cond_rows + L.audio_rows + L.video_rows;
    L.seq_len = ((L.used_len + 63u) / 64u) * 64u;

    const uint32_t cond_start = text_len;
    const uint32_t audio_start = cond_start + L.cond_rows;
    const uint32_t video_start = audio_start + L.audio_rows;

    L.token_tags.assign(L.seq_len, -1);
    if (!prefix_token_tags.empty()) {
        if (prefix_token_tags.size() != text_len)
            throw std::invalid_argument("H3PackedLayout::build: prefix_token_tags size != text_len");
        for (uint32_t i = 0; i < text_len; ++i) L.token_tags[i] = prefix_token_tags[i];
    } else {
        for (uint32_t i = 0; i < text_len; ++i) L.token_tags[i] = 1;
    }
    for (uint32_t i = 0; i < L.cond_rows; ++i) L.token_tags[cond_start + i] = 0;
    for (uint32_t i = 0; i < L.audio_rows; ++i) L.token_tags[audio_start + i] = 2;
    for (uint32_t i = 0; i < L.video_rows; ++i) L.token_tags[video_start + i] = 0;

    L.update_mask.assign(L.img_rows, 0);
    for (uint32_t i = L.cond_rows; i < L.img_rows; ++i) L.update_mask[i] = 1;

    L.img_position_ids.assign(size_t(L.seq_len) * 3, 0.0);
    auto set_pos = [&](uint32_t row, double t, double h, double w) {
        L.img_position_ids[size_t(row) * 3 + 0] = t;
        L.img_position_ids[size_t(row) * 3 + 1] = h;
        L.img_position_ids[size_t(row) * 3 + 2] = w;
    };

    // Text positions: (row_idx, 0, 0)
    for (uint32_t i = 0; i < text_len; ++i) set_pos(i, double(i), 0.0, 0.0);

    // Spatial grids
    const double sqrt_area = std::sqrt(double(latent_h) * double(latent_w));
    auto axis_grid = [&](uint32_t dim, uint32_t patch) {
        const uint32_t n = dim / patch;
        const double ratio = double(dim) / sqrt_area;
        const double left = (1.0 - ratio) * 0.5;
        std::vector<double> g(n);
        for (uint32_t i = 0; i < n; ++i) {
            g[i] = (left + ratio * (double(i) / double(n))) * kInterp;
        }
        return g;
    };
    const auto h_grid = axis_grid(latent_h, 2);
    const auto w_grid = axis_grid(latent_w, 2);

    // Temporal grid for target video
    std::vector<double> t_grid(latent_t, double(text_len));
    double cum = 0.0;
    for (uint32_t k = 1; k < latent_t; ++k) {
        cum += kFrameRescale * double(kFramePerToken[(k - 1) % 5]);
        t_grid[k] = double(text_len) + cum;
    }
    double total_t_span = cum + (latent_t > 0 ? kFrameRescale * double(kFramePerToken[(latent_t - 1) % 5]) : 0.0);

    // Condition keyframes
    for (uint32_t b = 0; b < n_cond_frames; ++b) {
        int pix = keyframe_frame_indices[b];
        if (pix == -1) pix = frame_count - 1;
        double cond_t = (pix == 0) ? double(text_len) : (double(text_len) + total_t_span - kFrameRescale);
        uint32_t base = cond_start + b * frame_rows;
        for (uint32_t hi = 0; hi < ph; ++hi)
            for (uint32_t wi = 0; wi < pw; ++wi)
                set_pos(base + hi * pw + wi, cond_t, h_grid[hi], w_grid[wi]);
    }

    // Audio positions: channel-major, w pinned to w_grid[0] and w_grid[-1]
    for (uint32_t ch = 0; ch < L.audio_channel; ++ch) {
        const double w_val = (ch == 0) ? w_grid.front() : w_grid.back();
        for (uint32_t ti = 0; ti < audio_t; ++ti) {
            set_pos(audio_start + ch * audio_t + ti, double(text_len) + double(ti), 0.0, w_val);
        }
    }

    // Target video positions
    for (uint32_t ti = 0; ti < latent_t; ++ti) {
        uint32_t base = video_start + ti * frame_rows;
        for (uint32_t hi = 0; hi < ph; ++hi)
            for (uint32_t wi = 0; wi < pw; ++wi)
                set_pos(base + hi * pw + wi, t_grid[ti], h_grid[hi], w_grid[wi]);
    }
    return L;
}

H3PackedLayout H3PackedLayout::build_image(uint32_t text_len,
                                           const std::vector<std::pair<uint32_t, uint32_t>>& ref_latent_grids,
                                           uint32_t latent_h, uint32_t latent_w,
                                           const std::vector<int32_t>& prefix_token_tags) {
    constexpr double kInterp = 32.0;
    H3PackedLayout L;
    L.text_len = text_len;
    L.latent_t = 1;
    L.latent_h = latent_h;
    L.latent_w = latent_w;
    L.audio_t = 0;
    L.audio_channel = 0;
    L.audio_rows = 0;

    std::vector<std::pair<uint32_t, uint32_t>> grids = ref_latent_grids;
    grids.push_back({latent_h, latent_w});
    std::vector<uint32_t> row_counts;
    row_counts.reserve(grids.size());
    uint32_t total_img_rows = 0;
    for (const auto& [gh, gw] : grids) {
        if ((gh % 2) || (gw % 2)) throw std::invalid_argument("H3PackedLayout::build_image: latent dims must be even");
        uint32_t rc = (gh / 2) * (gw / 2);
        row_counts.push_back(rc);
        total_img_rows += rc;
    }
    L.video_rows = row_counts.back();
    L.cond_rows = total_img_rows - L.video_rows;
    L.img_rows = total_img_rows;
    L.used_len = text_len + total_img_rows;
    L.seq_len = ((L.used_len + 63u) / 64u) * 64u;

    L.token_tags.assign(L.seq_len, -1);
    if (!prefix_token_tags.empty()) {
        if (prefix_token_tags.size() != text_len)
            throw std::invalid_argument("H3PackedLayout::build_image: prefix_token_tags size != text_len");
        for (uint32_t i = 0; i < text_len; ++i) L.token_tags[i] = prefix_token_tags[i];
    } else {
        for (uint32_t i = 0; i < text_len; ++i) L.token_tags[i] = 1;
    }
    for (uint32_t i = 0; i < total_img_rows; ++i) L.token_tags[text_len + i] = 0;

    L.update_mask.assign(L.img_rows, 0);
    for (uint32_t i = L.cond_rows; i < L.img_rows; ++i) L.update_mask[i] = 1;

    L.img_position_ids.assign(size_t(L.seq_len) * 3, 0.0);
    auto set_pos = [&](uint32_t row, double t, double h, double w) {
        L.img_position_ids[size_t(row) * 3 + 0] = t;
        L.img_position_ids[size_t(row) * 3 + 1] = h;
        L.img_position_ids[size_t(row) * 3 + 2] = w;
    };
    for (uint32_t i = 0; i < text_len; ++i) set_pos(i, double(i), 0.0, 0.0);

    uint32_t cursor = text_len;
    for (size_t img_idx = 0; img_idx < grids.size(); ++img_idx) {
        const uint32_t gh = grids[img_idx].first, gw = grids[img_idx].second;
        const uint32_t ph = gh / 2, pw = gw / 2;
        const double sqrt_area = std::sqrt(double(gh) * double(gw));
        auto axis_grid = [&](uint32_t dim) {
            const uint32_t n = dim / 2;
            const double ratio = double(dim) / sqrt_area;
            const double left = (1.0 - ratio) * 0.5;
            std::vector<double> g(n);
            for (uint32_t i = 0; i < n; ++i) g[i] = (left + ratio * (double(i) / double(n))) * kInterp;
            return g;
        };
        const auto hg = axis_grid(gh), wg = axis_grid(gw);
        const double t_val = double(text_len + img_idx);
        for (uint32_t hi = 0; hi < ph; ++hi)
            for (uint32_t wi = 0; wi < pw; ++wi)
                set_pos(cursor + hi * pw + wi, t_val, hg[hi], wg[wi]);
        cursor += ph * pw;
    }
    return L;
}

// ------------------------------------------------------------------------------------------------ H3DiT

H3DiT::H3DiT(const Device& device, const Stream& stream, const WeightPack& weights)
    : dev_(device), stream_(stream), w_(weights), cfg_(H3DiTConfig::from_pack(weights)) {
    const auto raw_inv = w_.read("rope.inv_freq");
    inv_freq_.resize(cfg_.rope_inv_freq_len);
    if (raw_inv.size() != inv_freq_.size() * sizeof(float))
        throw std::runtime_error("H3DiT: rope.inv_freq size mismatch");
    std::memcpy(inv_freq_.data(), raw_inv.data(), raw_inv.size());
    d_t_freq_ = DeviceBuffer(dev_, uint64_t(kPadM) * cfg_.timestep_input_dim * 4);
    d_t_emb_ = DeviceBuffer(dev_, uint64_t(kPadM) * cfg_.time_embed_dim * 4);
}

H3DiT::~H3DiT() = default;

void H3DiT::ensure_buffers(uint32_t seq_len, uint32_t text_len, uint32_t img_rows, uint32_t audio_rows) {
    const uint64_t H = cfg_.hidden_size;
    if (seq_len > alloc_seq_) {
        alloc_seq_ = seq_len;
        d_hidden_[0] = DeviceBuffer(dev_, uint64_t(seq_len) * H * 2);
        d_hidden_[1] = DeviceBuffer(dev_, uint64_t(seq_len) * H * 2);
        d_rope_cos_ = DeviceBuffer(dev_, uint64_t(seq_len) * cfg_.head_dim * 2);
        d_rope_sin_ = DeviceBuffer(dev_, uint64_t(seq_len) * cfg_.head_dim * 2);
        d_comb_idx_ = DeviceBuffer(dev_, uint64_t(seq_len) * 4);
        d_inv_idx_ = DeviceBuffer(dev_, uint64_t(seq_len) * 4);
        d_vid_out_ = DeviceBuffer(dev_, uint64_t(seq_len) * cfg_.video_row_width() * 4);
        d_aud_out_ = DeviceBuffer(dev_, uint64_t(seq_len) * cfg_.audio_latents_dim * 4);
    }
    if (text_len > alloc_text_) {
        alloc_text_ = text_len;
        d_txt_in_ = DeviceBuffer(dev_, uint64_t(text_len) * cfg_.text_dim * 2);
        d_txt_buf_[0] = DeviceBuffer(dev_, uint64_t(text_len) * H * 2);
        d_txt_buf_[1] = DeviceBuffer(dev_, uint64_t(text_len) * H * 2);
    }
    if (img_rows > alloc_img_) {
        alloc_img_ = img_rows;
        d_vid_in_ = DeviceBuffer(dev_, uint64_t(img_rows) * cfg_.video_row_width() * 4);
        d_vid_emb_ = DeviceBuffer(dev_, uint64_t(img_rows) * H * 2);
    }
    if (audio_rows > alloc_aud_) {
        alloc_aud_ = audio_rows;
        d_aud_in_ = DeviceBuffer(dev_, uint64_t(audio_rows) * cfg_.audio_latents_dim * 4);
        d_aud_emb_ = DeviceBuffer(dev_, uint64_t(audio_rows) * H * 2);
    }
}

void H3DiT::ensure_mf_buffers(uint32_t u_pad, uint32_t k_pad, bool fp8_qkv, uint32_t v_width) {
    const uint64_t NH = cfg_.num_heads, HD = cfg_.head_dim, H = cfg_.hidden_size;
    const uint64_t qkv_elem = fp8_qkv ? 1ull : 2ull;
    const bool need_wider = (!fp8_qkv && alloc_mf_fp8_);
    if (u_pad > alloc_mf_upad_ || need_wider) {
        alloc_mf_upad_ = std::max(alloc_mf_upad_, u_pad);
        d_mf_q_ = DeviceBuffer(dev_, NH * uint64_t(alloc_mf_upad_) * HD * qkv_elem);
        if (!fp8_qkv) d_mf_b_ = DeviceBuffer(dev_, NH * uint64_t(alloc_mf_upad_) * 2);
        else d_mf_b_ = DeviceBuffer{};
        d_mf_o_ = DeviceBuffer(dev_, NH * uint64_t(alloc_mf_upad_) * HD * 2);
    }
    const bool k_grow = (k_pad > alloc_mf_kpad_) || need_wider;
    if (k_grow) {
        alloc_mf_kpad_ = std::max(alloc_mf_kpad_, k_pad);
        d_mf_k_ = DeviceBuffer(dev_, NH * uint64_t(alloc_mf_kpad_) * HD * qkv_elem);
    }
    if (k_grow || v_width > alloc_mf_v_width_) {
        alloc_mf_v_width_ = std::max(alloc_mf_v_width_, v_width);
        d_mf_v_ = DeviceBuffer(dev_, NH * uint64_t(alloc_mf_kpad_) * alloc_mf_v_width_ * qkv_elem);
    }
    alloc_mf_fp8_ = fp8_qkv;
    if (!d_mf_ad_.size()) {
        d_mf_ad_ = DeviceBuffer(dev_, 4ull * kPadM * 3 * H * 2);
        d_mf_sc_ = DeviceBuffer(dev_, 128);
        d_mf_dv_ = DeviceBuffer(dev_, 128);
    }
}

void H3DiT::upload_rope(const H3PackedLayout& layout, std::vector<float>* debug_rope_table) {
    const uint32_t S = layout.seq_len;
    const uint32_t F = cfg_.rope_inv_freq_len;  // 16
    const uint32_t half = 3 * F;                // 48
    const uint32_t HD = cfg_.head_dim;          // 128
    if (debug_rope_table == nullptr && cached_rope_seq_ == S && cached_rope_pos_ == layout.img_position_ids) return;

    const uint16_t bf16_one = to_bf16(1.0f), bf16_zero = to_bf16(0.0f);
    std::vector<uint16_t> h_cos(size_t(S) * HD, bf16_one), h_sin(size_t(S) * HD, bf16_zero);
    if (debug_rope_table) debug_rope_table->resize(size_t(S) * 2 * half);
    for (uint32_t r = 0; r < S; ++r) {
        const float pt = float(layout.img_position_ids[size_t(r) * 3 + 0]);
        const float ph = float(layout.img_position_ids[size_t(r) * 3 + 1]);
        const float pw = float(layout.img_position_ids[size_t(r) * 3 + 2]);
        for (uint32_t i = 0; i < F; ++i) {
            const float inv = inv_freq_[i];
            const float ft = pt * inv, fh = ph * inv, fw = pw * inv;
            const float cs_vals[3] = {std::cos(ft), std::cos(fh), std::cos(fw)};
            const float sn_vals[3] = {std::sin(ft), std::sin(fh), std::sin(fw)};
            for (uint32_t ax = 0; ax < 3; ++ax) {
                const uint32_t col = ax * F + i;
                const uint16_t cb = to_bf16(cs_vals[ax]);
                const uint16_t sb_pos = to_bf16(sn_vals[ax]);
                const uint16_t sb_neg = to_bf16(-sn_vals[ax]);
                h_cos[size_t(r) * HD + col] = cb;
                h_cos[size_t(r) * HD + half + col] = cb;
                h_sin[size_t(r) * HD + col] = sb_neg;
                h_sin[size_t(r) * HD + half + col] = sb_pos;
                if (debug_rope_table) {
                    (*debug_rope_table)[size_t(r) * 2 * half + col] = from_bf16(cb);
                    (*debug_rope_table)[size_t(r) * 2 * half + half + col] = from_bf16(sb_pos);
                }
            }
        }
    }
    const size_t nbytes = size_t(S) * HD * sizeof(uint16_t);
    HostBuffer stg(dev_, nbytes * 2);
    std::memcpy(stg.data(), h_cos.data(), nbytes);
    std::memcpy(static_cast<uint8_t*>(stg.data()) + nbytes, h_sin.data(), nbytes);
    stream_.upload(stg, 0, d_rope_cos_, 0, nbytes);
    stream_.upload(stg, nbytes, d_rope_sin_, 0, nbytes);
    stream_.synchronize();
    cached_rope_seq_ = S;
    cached_rope_pos_ = layout.img_position_ids;
}

// ------------------------------------------------------------------------------------------------ Graph helpers

TensorId H3DiT::add_bias(Ops& o, TensorId x, TensorId bias, const std::string& name) {
    const auto bs = o.shape(bias);
    return o.add(x, o.reshape(bias, {1, bs[0]}, name + "_b"), name);
}

TensorId H3DiT::adaln_linear(Ops& o, TensorId x_bf16, TensorId w, TensorId ws, TensorId bias, const std::string& name,
                             TensorId lora_a, TensorId lora_b) {
    TensorId y = cfg_.adaln_bf16 ? o.linear_bf16(x_bf16, w, name + "_mm") : o.fp8_linear(x_bf16, w, ws, name + "_mm");
    if (lora_a != kNoTensor) {
        const auto mid = o.linear_bf16(x_bf16, lora_a, name + "_la");
        const auto delta = o.linear_bf16(mid, lora_b, name + "_lb");
        y = o.add(y, delta, name + "_lo");
    }
    return add_bias(o, y, bias, name);
}

TensorId H3DiT::proj_with_lora(Ops& o, TensorId x_bf16, TensorId x_codes, TensorId x_scale,
                               TensorId w, TensorId ws, TensorId lora_a, TensorId lora_b,
                               bool is_qkv, const std::string& name) {
    TensorId base = o.fp8_gemm(x_codes, x_scale, w, ws, name + "_base");
    if (lora_a == kNoTensor) return base;
    if (is_qkv) {
        // lora_a: [3*R, H], lora_b: [3, 7168, R]
        const uint64_t S = o.shape(x_bf16)[0];
        const uint64_t R = o.shape(lora_a)[0] / 3;
        const uint64_t D = o.shape(lora_b)[1];
        const auto mid = o.linear_bf16(x_bf16, lora_a, name + "_la");                 // [S, 3*R]
        const auto mid3 = o.transpose(o.reshape(mid, {S, 3, R}, name + "_m3"), {1, 0, 2}, name + "_m3t");  // [3, S, R]
        const auto d3 = o.batch_gemm(mid3, lora_b, name + "_lb", false, true, DType::BF16);                // [3, S, D]
        const auto delta = o.reshape(o.transpose(d3, {1, 0, 2}, name + "_dt"), {S, 3 * D}, name + "_df");  // [S, 3*D]
        return o.add(base, delta, name);
    } else {
        const auto mid = o.linear_bf16(x_bf16, lora_a, name + "_la");
        const auto delta = o.linear_bf16(mid, lora_b, name + "_lb");
        return o.add(base, delta, name);
    }
}

// ------------------------------------------------------------------------------------------------ Recipes

Recipe& H3DiT::get_time_embed_recipe() {
    if (r_time_embed_) return *r_time_embed_;
    GraphBuilder g(dev_);
    Ops o(g, "te_");
    const auto tf = g.input("t_freq", DType::F32, {kPadM, cfg_.timestep_input_dim});
    const auto w1 = g.input("w1", DType::F32, {cfg_.time_embed_hidden, cfg_.timestep_input_dim});
    const auto b1 = g.input("b1", DType::F32, {cfg_.time_embed_hidden});
    const auto w2 = g.input("w2", DType::F32, {cfg_.time_embed_dim, cfg_.time_embed_hidden});
    const auto b2 = g.input("b2", DType::F32, {cfg_.time_embed_dim});
    const auto h1 = o.temp("h1", DType::F32, {kPadM, cfg_.time_embed_hidden});
    g.gemm(tf, w1, h1, false, true);
    const auto h1a = o.silu(add_bias(o, h1, b1, "h1b"), "h1s");
    const auto h2 = o.temp("h2", DType::F32, {kPadM, cfg_.time_embed_dim});
    g.gemm(h1a, w2, h2, false, true);
    set_output(g, "t_emb", add_bias(o, h2, b2, "h2b"));
    r_time_embed_ = g.compile("h3_time_embed");
    return *r_time_embed_;
}

Recipe& H3DiT::get_patch_proj_recipe(uint32_t n_rows, bool is_audio) {
    auto key = std::make_pair(n_rows, is_audio);
    if (auto it = r_patch_proj_.find(key); it != r_patch_proj_.end()) return *it->second;
    const uint64_t in_dim = is_audio ? cfg_.audio_latents_dim : cfg_.video_row_width();
    GraphBuilder g(dev_);
    Ops o(g, is_audio ? "ap_" : "vp_");
    const auto x = g.input("x", DType::F32, {n_rows, in_dim});
    const auto w = g.input("w", DType::F32, {cfg_.hidden_size, in_dim});
    const auto b = g.input("b", DType::F32, {cfg_.hidden_size});
    const auto y = o.temp("y", DType::F32, {n_rows, cfg_.hidden_size});
    g.gemm(x, w, y, false, true);
    set_output(g, "out", o.cast(add_bias(o, y, b, "yb"), DType::BF16, "out_bf16"));
    auto r = g.compile(std::string(is_audio ? "h3_aud_proj_" : "h3_vid_proj_") + std::to_string(n_rows));
    auto& ref = *r;
    r_patch_proj_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_cond_proj_recipe(uint32_t text_len) {
    if (auto it = r_cond_proj_.find(text_len); it != r_cond_proj_.end()) return *it->second;
    GraphBuilder g(dev_);
    Ops o(g, "cp_");
    const auto x = g.input("x", DType::BF16, {text_len, cfg_.text_dim});
    const auto w = g.input("w", DType::F8, {cfg_.hidden_size, cfg_.text_dim});
    const auto ws = g.input("ws", DType::F32, {cfg_.hidden_size});
    const auto b = g.input("b", DType::BF16, {cfg_.hidden_size});
    const auto y = o.fp8_linear(x, w, ws, "mm");
    set_output(g, "out", add_bias(o, y, b, "yb"));
    auto r = g.compile("h3_cond_proj_" + std::to_string(text_len));
    auto& ref = *r;
    r_cond_proj_.emplace(text_len, std::move(r));
    return ref;
}

Recipe& H3DiT::get_refiner_block_recipe(uint32_t text_len, uint32_t lora_rank) {
    RefinerKey key{text_len, lora_rank};
    if (auto it = r_refiner_block_.find(key); it != r_refiner_block_.end()) return *it->second;
    const uint64_t S = text_len, H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.head_dim;
    const uint64_t AD = NH * HD, FFN = cfg_.ffn_hidden;

    GraphBuilder g(dev_);
    Ops o(g, "rb_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto norm1 = g.input("norm1", DType::BF16, {H});
    const auto norm2 = g.input("norm2", DType::BF16, {H});
    const auto q_norm = g.input("q_norm", DType::BF16, {HD});
    const auto k_norm = g.input("k_norm", DType::BF16, {HD});
    const auto qkv_w = g.input("qkv_w", DType::F8, {3 * AD, H});
    const auto qkv_s = g.input("qkv_s", DType::F32, {3 * AD});
    const auto out_w = g.input("out_w", DType::F8, {H, AD});
    const auto out_s = g.input("out_s", DType::F32, {H});
    const auto fc1_w = g.input("fc1_w", DType::F8, {2 * FFN, H});
    const auto fc1_s = g.input("fc1_s", DType::F32, {2 * FFN});
    const auto fc2_w = g.input("fc2_w", DType::F8, {H, FFN});
    const auto fc2_s = g.input("fc2_s", DType::F32, {H});

    TensorId l_qkv_a = kNoTensor, l_qkv_b = kNoTensor, l_out_a = kNoTensor, l_out_b = kNoTensor;
    TensorId l_fc1_a = kNoTensor, l_fc1_b = kNoTensor, l_fc2_a = kNoTensor, l_fc2_b = kNoTensor;
    if (lora_rank > 0) {
        l_qkv_a = g.input("lora_qkv_a", DType::BF16, {3 * uint64_t(lora_rank), H});
        l_qkv_b = g.input("lora_qkv_b", DType::BF16, {3, AD, lora_rank});
        l_out_a = g.input("lora_out_a", DType::BF16, {lora_rank, AD});
        l_out_b = g.input("lora_out_b", DType::BF16, {H, lora_rank});
        l_fc1_a = g.input("lora_fc1_a", DType::BF16, {lora_rank, H});
        l_fc1_b = g.input("lora_fc1_b", DType::BF16, {2 * FFN, lora_rank});
        l_fc2_a = g.input("lora_fc2_a", DType::BF16, {lora_rank, FFN});
        l_fc2_b = g.input("lora_fc2_b", DType::BF16, {H, lora_rank});
    }

    // Attention (no RoPE in TokenRefiner)
    const auto h1 = o.rms_norm(x, norm1, cfg_.norm_eps, false, "n1");
    const auto [h1c, h1s] = o.quantize(h1, "h1q");
    const auto qkv = proj_with_lora(o, h1, h1c, h1s, qkv_w, qkv_s, l_qkv_a, l_qkv_b, true, "qkv");
    const auto qkv_parts = o.split(qkv, 1, {AD, AD, AD}, "qkv_sp");
    auto q = o.rms_norm(o.reshape(qkv_parts[0], {S, NH, HD}, "qr"), q_norm, cfg_.qk_norm_eps, false, "qn");
    auto k = o.rms_norm(o.reshape(qkv_parts[1], {S, NH, HD}, "kr"), k_norm, cfg_.qk_norm_eps, false, "kn");
    auto v = o.reshape(qkv_parts[2], {S, NH, HD}, "vr");

    q = o.reshape(o.transpose(q, {1, 0, 2}, "qt"), {1, NH, S, HD}, "qt4");
    k = o.reshape(o.transpose(k, {1, 0, 2}, "kt"), {1, NH, S, HD}, "kt4");
    v = o.reshape(o.transpose(v, {1, 0, 2}, "vt"), {1, NH, S, HD}, "vt4");
    const auto ctx4 = sdpa_bidir_bf16(o, q, k, v, float(1.0 / std::sqrt(double(HD))), false, false, "sdpa");
    const auto ctx = o.reshape(o.transpose(o.reshape(ctx4, {NH, S, HD}, "c3"), {1, 0, 2}, "ct"), {S, AD}, "cf");
    const auto [ctxc, ctxs] = o.quantize(ctx, "ctxq");
    const auto aout = proj_with_lora(o, ctx, ctxc, ctxs, out_w, out_s, l_out_a, l_out_b, false, "out");
    const auto x1 = o.add(x, aout, "res1");

    // MLP (SwiGLU)
    const auto h2 = o.rms_norm(x1, norm2, cfg_.norm_eps, false, "n2");
    const auto [h2c, h2s] = o.quantize(h2, "h2q");
    TensorId mout;
    if (lora_rank == 0) {
        const auto [mc, ms] = o.fp8_gemm_silu_mul_quant(h2c, h2s, fc1_w, fc1_s, "fc1");
        mout = o.fp8_gemm(mc, ms, fc2_w, fc2_s, "fc2");
    } else {
        const auto gu = proj_with_lora(o, h2, h2c, h2s, fc1_w, fc1_s, l_fc1_a, l_fc1_b, false, "fc1");
        const auto gup = o.split(gu, 1, {FFN, FFN}, "gup");
        const auto act = o.mul(o.silu(gup[0], "gs"), gup[1], "act");
        const auto [ac, as] = o.quantize(act, "actq");
        mout = proj_with_lora(o, act, ac, as, fc2_w, fc2_s, l_fc2_a, l_fc2_b, false, "fc2");
    }
    set_output(g, "out", o.add(x1, mout, "res2"));

    auto r = g.compile("h3_refiner_b_" + std::to_string(text_len) + "_r" + std::to_string(lora_rank));
    auto& ref = *r;
    r_refiner_block_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_refiner_norm_recipe(uint32_t text_len) {
    if (auto it = r_refiner_norm_.find(text_len); it != r_refiner_norm_.end()) return *it->second;
    GraphBuilder g(dev_);
    Ops o(g, "rn_");
    const auto x = g.input("x", DType::BF16, {text_len, cfg_.hidden_size});
    const auto w = g.input("w", DType::BF16, {cfg_.hidden_size});
    set_output(g, "out", o.rms_norm(x, w, cfg_.final_norm_eps, false, "norm"));
    auto r = g.compile("h3_refiner_norm_" + std::to_string(text_len));
    auto& ref = *r;
    r_refiner_norm_.emplace(text_len, std::move(r));
    return ref;
}

Recipe& H3DiT::get_dit_block_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank, uint32_t adaln_rank) {
    const bool fast_sm = (std::getenv("G2E_H3_SDPA_FAST") != nullptr);
    const bool kv_split = (std::getenv("G2E_H3_NO_KV_SPLIT") == nullptr);
    const bool fp8_sdpa = (std::getenv("G2E_H3_FP8_SDPA") != nullptr);
    const uint32_t bflags = (fast_sm ? 1u : 0u) | (kv_split ? 2u : 0u) | (fp8_sdpa ? 4u : 0u);
    ShapeKey key{seq_len, used_len, lora_rank, adaln_rank, bflags};
    if (auto it = r_dit_block_.find(key); it != r_dit_block_.end()) return *it->second;
    const uint64_t S = seq_len, U = used_len, H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.head_dim;
    const uint64_t AD = NH * HD, FFN = cfg_.ffn_hidden, R2 = cfg_.rot_dim() / 2;

    GraphBuilder g(dev_);
    Ops o(g, "db_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto t_emb = g.input("t_emb", DType::F32, {kPadM, cfg_.time_embed_dim});
    const auto comb_idx = g.input("comb_idx", DType::I32, {S});
    const auto rope_cos = g.input("rope_cos", DType::BF16, {S, HD});
    const auto rope_sin = g.input("rope_sin", DType::BF16, {S, HD});

    const auto norm1 = g.input("norm1", DType::BF16, {H});
    const auto norm2 = g.input("norm2", DType::BF16, {H});
    const auto q_norm = g.input("q_norm", DType::BF16, {HD});
    const auto k_norm = g.input("k_norm", DType::BF16, {HD});
    const auto qkv_w = g.input("qkv_w", DType::F8, {3 * AD, H});
    const auto qkv_s = g.input("qkv_s", DType::F32, {3 * AD});
    const auto out_w = g.input("out_w", DType::F8, {H, AD});
    const auto out_s = g.input("out_s", DType::F32, {H});
    const auto fc1_w = g.input("fc1_w", DType::F8, {2 * FFN, H});
    const auto fc1_s = g.input("fc1_s", DType::F32, {2 * FFN});
    const auto fc2_w = g.input("fc2_w", DType::F8, {H, FFN});
    const auto fc2_s = g.input("fc2_s", DType::F32, {H});
    const auto adaln_w = g.input("adaln_w", cfg_.adaln_bf16 ? DType::BF16 : DType::F8, {18 * H, cfg_.time_embed_dim});
    const auto adaln_s = cfg_.adaln_bf16 ? kNoTensor : g.input("adaln_s", DType::F32, {18 * H});
    const auto adaln_b = g.input("adaln_b", DType::BF16, {18 * H});

    TensorId l_qkv_a = kNoTensor, l_qkv_b = kNoTensor, l_out_a = kNoTensor, l_out_b = kNoTensor;
    TensorId l_fc1_a = kNoTensor, l_fc1_b = kNoTensor, l_fc2_a = kNoTensor, l_fc2_b = kNoTensor;
    TensorId l_adaln_a = kNoTensor, l_adaln_b = kNoTensor;
    if (lora_rank > 0) {
        l_qkv_a = g.input("lora_qkv_a", DType::BF16, {3 * uint64_t(lora_rank), H});
        l_qkv_b = g.input("lora_qkv_b", DType::BF16, {3, AD, lora_rank});
        l_out_a = g.input("lora_out_a", DType::BF16, {lora_rank, AD});
        l_out_b = g.input("lora_out_b", DType::BF16, {H, lora_rank});
        l_fc1_a = g.input("lora_fc1_a", DType::BF16, {lora_rank, H});
        l_fc1_b = g.input("lora_fc1_b", DType::BF16, {2 * FFN, lora_rank});
        l_fc2_a = g.input("lora_fc2_a", DType::BF16, {lora_rank, FFN});
        l_fc2_b = g.input("lora_fc2_b", DType::BF16, {H, lora_rank});
    }
    if (adaln_rank > 0) {
        l_adaln_a = g.input("lora_adaln_a", DType::BF16, {adaln_rank, cfg_.time_embed_dim});
        l_adaln_b = g.input("lora_adaln_b", DType::BF16, {18 * H, adaln_rank});
    }

    // 1. AdaLN projection: [kPadM, 2688] -> [kPadM*3, 6*H] -> 6 x [kPadM*3, H]
    const auto t_act = o.cast(o.silu(t_emb, "ts"), DType::BF16, "tsb");
    const auto adaln_raw = adaln_linear(o, t_act, adaln_w, adaln_s, adaln_b, "adaln", l_adaln_a, l_adaln_b);
    const auto adaln_3 = o.reshape(adaln_raw, {kPadM * 3, 6 * H}, "ad3");
    const auto ad = o.split(adaln_3, 1, {H, H, H, H, H, H}, "ad_sp");
    // ad[0..5] = shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    const bool fused_adaln = (std::getenv("G2E_H3_UNFUSED_ADALN") == nullptr);
    struct H3NormQuantParams { float eps, inv_w, inv_range; };
    const H3NormQuantParams nqp{cfg_.norm_eps, 1.0f / float(H), 0.004180908203125f};

    // 2. Norm1 + Indexed AdaLN + Dynamic Row FP8 Quant
    TensorId h1, h1c, h1s;
    if (fused_adaln) {
        h1c = o.temp("h1_codes", DType::F8, {S, H});
        h1s = o.temp("h1_scale", DType::F32, {S, 1});
        h1 = o.temp("h1", DType::BF16, {S, H});
        g.op("flashinfer_gaudi_h3_adaln_rmsnorm_quant_bf16_gaudi2",
             {x, norm1, ad[0], ad[1], comb_idx}, {h1c, h1s, h1}, &nqp, sizeof(nqp));
    } else {
        const auto one_f = o.constant(1.0f, {1}, "one1", DType::F32);
        const auto n1 = o.rms_norm(x, norm1, cfg_.norm_eps, false, "n1");
        const auto sh_msa = o.cast(o.gather_rows(ad[0], comb_idx, "sh_msa"), DType::F32, "sh_msaf");
        const auto sc_msa = o.add(o.cast(o.gather_rows(ad[1], comb_idx, "sc_msa"), DType::F32, "sc_msaf"), one_f, "sc1_msa");
        h1 = o.cast(o.add(o.mul(o.cast(n1, DType::F32, "n1f"), sc_msa, "n1s"), sh_msa, "n1a"), DType::BF16, "h1");
        std::tie(h1c, h1s) = o.quantize(h1, "h1q");
    }

    // 3 & 4. QKV + fused QK-Norm + 3D RoPE + prefix slice + transpose -> [1, NH, U, HD]
    const auto qkv = proj_with_lora(o, h1, h1c, h1s, qkv_w, qkv_s, l_qkv_a, l_qkv_b, true, "qkv");
    const bool fused_rope = (std::getenv("G2E_H3_UNFUSED_ROPE") == nullptr) && HD == 128 && R2 == 48;
    TensorId qu, ku, vu;
    if (fused_rope) {
        const auto q3 = o.temp("qf3", DType::BF16, {NH, U, HD});
        const auto k3 = o.temp("kf3", DType::BF16, {NH, U, HD});
        const auto v3 = o.temp("vf3", DType::BF16, {NH, U, HD});
        const float qk_eps = cfg_.qk_norm_eps;
        g.op("flashinfer_gaudi_h3_qk_norm_rope_bf16_gaudi2",
             {qkv, q_norm, k_norm, rope_cos, rope_sin}, {q3, k3, v3}, &qk_eps, sizeof(qk_eps));
        qu = o.reshape(q3, {1, NH, U, HD}, "qt4");
        ku = o.reshape(k3, {1, NH, U, HD}, "kt4");
        vu = o.reshape(v3, {1, NH, U, HD}, "vt4");
    } else {
        const auto qkv_parts = o.split(qkv, 1, {AD, AD, AD}, "qkv_sp");
        auto q = o.rms_norm(o.reshape(qkv_parts[0], {S, NH, HD}, "qr"), q_norm, cfg_.qk_norm_eps, false, "qn");
        auto k = o.rms_norm(o.reshape(qkv_parts[1], {S, NH, HD}, "kr"), k_norm, cfg_.qk_norm_eps, false, "kn");
        auto v = o.reshape(qkv_parts[2], {S, NH, HD}, "vr");
        const auto cos48 = o.split(rope_cos, 1, {R2, HD - R2}, "c48")[0];
        const auto sin48 = o.split(rope_sin, 1, {R2, R2, HD - 2 * R2}, "s48")[1];
        q = apply_h3_rope(o, q, cos48, sin48, cfg_.rot_dim(), "qrope");
        k = apply_h3_rope(o, k, cos48, sin48, cfg_.rot_dim(), "krope");
        qu = o.reshape(o.transpose(slice_prefix_axis0(o, q, U, "qu"), {1, 0, 2}, "qt"), {1, NH, U, HD}, "qt4");
        ku = o.reshape(o.transpose(slice_prefix_axis0(o, k, U, "ku"), {1, 0, 2}, "kt"), {1, NH, U, HD}, "kt4");
        vu = o.reshape(o.transpose(slice_prefix_axis0(o, v, U, "vu"), {1, 0, 2}, "vt"), {1, NH, U, HD}, "vt4");
    }

    const float attn_scale = float(1.0 / std::sqrt(double(HD)));
    const auto ctx4 = fp8_sdpa ? sdpa_bidir_fp8(o, qu, ku, vu, attn_scale, fast_sm, kv_split, "sdpa8")
                               : sdpa_bidir_bf16(o, qu, ku, vu, attn_scale, fast_sm, kv_split, "sdpa");
    auto ctx_u = o.reshape(o.transpose(o.reshape(ctx4, {NH, U, HD}, "c3"), {1, 0, 2}, "ct"), {U, AD}, "cf");
    const auto ctx = zero_pad_axis0(o, ctx_u, S, "ctx");

    // 5 & 6. Out proj + Gated residual 1 + Norm2 + Indexed AdaLN + Dynamic Row FP8 Quant
    TensorId ctxc, ctxs;
    if (fused_adaln && (AD % 128 == 0) && AD <= 8192) {
        ctxc = o.temp("ctxq_codes", DType::F8, {S, AD});
        ctxs = o.temp("ctxq_scale", DType::F32, {S, 1});
        g.op("flashinfer_gaudi_h3_row_quant_bf16_gaudi2", {ctx}, {ctxc, ctxs});
    } else {
        std::tie(ctxc, ctxs) = o.quantize(ctx, "ctxq");
    }
    const auto aout = proj_with_lora(o, ctx, ctxc, ctxs, out_w, out_s, l_out_a, l_out_b, false, "out");
    TensorId x1, h2, h2c, h2s;
    if (fused_adaln) {
        h2c = o.temp("h2_codes", DType::F8, {S, H});
        h2s = o.temp("h2_scale", DType::F32, {S, 1});
        h2 = o.temp("h2", DType::BF16, {S, H});
        x1 = o.temp("x1", DType::BF16, {S, H});
        g.op("flashinfer_gaudi_h3_gate_adaln_rmsnorm_quant_bf16_gaudi2",
             {x, aout, ad[2], norm2, ad[3], ad[4], comb_idx}, {h2c, h2s, h2, x1}, &nqp, sizeof(nqp));
    } else {
        const auto one_f = o.constant(1.0f, {1}, "one2", DType::F32);
        const auto g_msa = o.cast(o.gather_rows(ad[2], comb_idx, "g_msa"), DType::F32, "g_msaf");
        x1 = o.cast(o.add(o.cast(x, DType::F32, "xf"), o.mul(g_msa, o.cast(aout, DType::F32, "af"), "ga"), "x1f"),
                    DType::BF16, "x1");
        const auto n2 = o.rms_norm(x1, norm2, cfg_.norm_eps, false, "n2");
        const auto sh_mlp = o.cast(o.gather_rows(ad[3], comb_idx, "sh_mlp"), DType::F32, "sh_mlpf");
        const auto sc_mlp = o.add(o.cast(o.gather_rows(ad[4], comb_idx, "sc_mlp"), DType::F32, "sc_mlpf"), one_f, "sc1_mlp");
        h2 = o.cast(o.add(o.mul(o.cast(n2, DType::F32, "n2f"), sc_mlp, "n2s"), sh_mlp, "n2a"), DType::BF16, "h2");
        std::tie(h2c, h2s) = o.quantize(h2, "h2q");
    }

    // 7. SwiGLU MLP + Gated residual 2
    TensorId mout;
    if (lora_rank == 0) {
        const auto [mc, ms] = o.fp8_gemm_silu_mul_quant(h2c, h2s, fc1_w, fc1_s, "fc1");
        mout = o.fp8_gemm(mc, ms, fc2_w, fc2_s, "fc2");
    } else {
        const auto fc1_raw = o.temp("fc1_raw", DType::BF16, {S, 2 * FFN});
        ns_Fp8Gemm::ParamsV2 gp{};
        gp.transpose_a = false;
        gp.transpose_b = true;
        g.op("fp8_gemm_bf16", {h2c, fc1_w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {fc1_raw}, &gp, sizeof(gp));
        const auto fc1_lo = o.linear_bf16(h2, l_fc1_a, "fc1_la");
        const auto fc1_dl = o.linear_bf16(fc1_lo, l_fc1_b, "fc1_lb");
        const auto ac = o.temp("actq_codes", DType::F8, {S, FFN});
        const auto as = o.temp("actq_scale", DType::F32, {S, 1});
        const auto act = o.temp("act", DType::BF16, {S, FFN});
        g.op("flashinfer_gaudi_h3_silu_mul_quant_dual_bf16_gaudi2",
             {fc1_raw, h2s, o.cast(fc1_s, DType::BF16, "fc1sb"), fc1_dl}, {ac, as, act});
        mout = proj_with_lora(o, act, ac, as, fc2_w, fc2_s, l_fc2_a, l_fc2_b, false, "fc2");
    }
    TensorId x2;
    if (fused_adaln) {
        x2 = o.temp("x2", DType::BF16, {S, H});
        g.op("flashinfer_gaudi_h3_gate_residual_bf16_gaudi2", {x1, mout, ad[5], comb_idx}, {x2});
    } else {
        const auto g_mlp = o.cast(o.gather_rows(ad[5], comb_idx, "g_mlp"), DType::F32, "g_mlpf");
        x2 = o.cast(o.add(o.cast(x1, DType::F32, "x1ff"), o.mul(g_mlp, o.cast(mout, DType::F32, "mf"), "gm"), "x2f"),
                    DType::BF16, "x2");
    }
    set_output(g, "out", x2);

    const std::string rname = "h3_dit_b_" + std::to_string(S) + "_u" + std::to_string(U) +
                              "_r" + std::to_string(lora_rank) +
                              (adaln_rank > 0 ? ("_ar" + std::to_string(adaln_rank)) : "") +
                              (fp8_sdpa ? "_fp8sdpa" : "_sdpa") +
                              (fast_sm ? "_fast" : "") + (kv_split ? "_ks" : "") + "_v3";
    auto r = g.compile(rname);
    auto& ref = *r;
    r_dit_block_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_dit_pre_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank,
                                  uint32_t h_sub, uint32_t q_chunk, uint32_t adaln_rank) {
    MfKey key{seq_len, used_len, lora_rank, adaln_rank, h_sub, q_chunk, 0u};
    if (auto it = r_dit_pre_.find(key); it != r_dit_pre_.end()) return *it->second;
    const uint64_t S = seq_len, U = used_len, H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.head_dim;
    const uint64_t AD = NH * HD;
    const uint64_t K_pad = ((U + 127) / 128) * 128;
    const uint64_t pad_k = K_pad - U;
    const uint64_t Q_blocks = (U + q_chunk - 1) / q_chunk;
    const uint64_t U_pad = Q_blocks * q_chunk;
    const uint64_t pad_q = U_pad - U;
    const uint64_t H_groups = NH / h_sub;

    GraphBuilder g(dev_);
    Ops o(g, "dp_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto t_emb = g.input("t_emb", DType::F32, {kPadM, cfg_.time_embed_dim});
    const auto comb_idx = g.input("comb_idx", DType::I32, {S});
    const auto rope_cos = g.input("rope_cos", DType::BF16, {S, HD});
    const auto rope_sin = g.input("rope_sin", DType::BF16, {S, HD});
    const auto norm1 = g.input("norm1", DType::BF16, {H});
    const auto q_norm = g.input("q_norm", DType::BF16, {HD});
    const auto k_norm = g.input("k_norm", DType::BF16, {HD});
    const auto qkv_w = g.input("qkv_w", DType::F8, {3 * AD, H});
    const auto qkv_s = g.input("qkv_s", DType::F32, {3 * AD});
    const auto adaln_w = g.input("adaln_w", cfg_.adaln_bf16 ? DType::BF16 : DType::F8, {18 * H, cfg_.time_embed_dim});
    const auto adaln_s = cfg_.adaln_bf16 ? kNoTensor : g.input("adaln_s", DType::F32, {18 * H});
    const auto adaln_b = g.input("adaln_b", DType::BF16, {18 * H});

    TensorId l_qkv_a = kNoTensor, l_qkv_b = kNoTensor, l_adaln_a = kNoTensor, l_adaln_b = kNoTensor;
    if (lora_rank > 0) {
        l_qkv_a = g.input("lora_qkv_a", DType::BF16, {3 * uint64_t(lora_rank), H});
        l_qkv_b = g.input("lora_qkv_b", DType::BF16, {3, AD, lora_rank});
    }
    if (adaln_rank > 0) {
        l_adaln_a = g.input("lora_adaln_a", DType::BF16, {adaln_rank, cfg_.time_embed_dim});
        l_adaln_b = g.input("lora_adaln_b", DType::BF16, {18 * H, adaln_rank});
    }

    const auto t_act = o.cast(o.silu(t_emb, "ts"), DType::BF16, "tsb");
    const auto adaln_raw = adaln_linear(o, t_act, adaln_w, adaln_s, adaln_b, "adaln", l_adaln_a, l_adaln_b);
    const auto adaln_3 = o.reshape(adaln_raw, {kPadM * 3, 6 * H}, "ad3");
    const auto ad = o.split(adaln_3, 1, {H, H, H, H, H, H}, "ad_sp");

    struct H3NormQuantParams { float eps, inv_w, inv_range; };
    const H3NormQuantParams nqp{cfg_.norm_eps, 1.0f / float(H), 0.004180908203125f};
    const auto h1c = o.temp("h1_codes", DType::F8, {S, H});
    const auto h1s = o.temp("h1_scale", DType::F32, {S, 1});
    const auto h1 = o.temp("h1", DType::BF16, {S, H});
    g.op("flashinfer_gaudi_h3_adaln_rmsnorm_quant_bf16_gaudi2",
         {x, norm1, ad[0], ad[1], comb_idx}, {h1c, h1s, h1}, &nqp, sizeof(nqp));

    const auto qkv = proj_with_lora(o, h1, h1c, h1s, qkv_w, qkv_s, l_qkv_a, l_qkv_b, true, "qkv");
    const auto q3 = o.temp("qf3", DType::BF16, {NH, U, HD});
    const auto k3 = o.temp("kf3", DType::BF16, {NH, U, HD});
    const auto v3 = o.temp("vf3", DType::BF16, {NH, U, HD});
    const float qk_eps = cfg_.qk_norm_eps;
    g.op("flashinfer_gaudi_h3_qk_norm_rope_bf16_gaudi2",
         {qkv, q_norm, k_norm, rope_cos, rope_sin}, {q3, k3, v3}, &qk_eps, sizeof(qk_eps));

    // Scale Q and compute Cauchy-Schwarz bound [NH, U, 1] BF16
    const float attn_scale = float(1.0 / std::sqrt(double(HD)));
    const auto q_sc = o.mul(q3, o.constant(attn_scale, {1}, "sc", DType::BF16), "qsc");
    const auto qf = o.cast(q_sc, DType::F32, "qf");
    const auto kf = o.cast(k3, DType::F32, "kf");
    const auto q_norm2 = o.unary("sqrt", o.reduce("sum", o.mul(qf, qf, "q2"), 2, "q2s"), "qn");
    const auto k_norm2 = o.unary("sqrt", o.reduce("sum", o.mul(kf, kf, "k2"), 2, "k2s"), "kn");
    const auto k_max = o.reduce("max", k_norm2, 1, "km");  // [NH, 1, 1]
    const auto bound = o.cast(o.mul(q_norm2, k_max, "bf"), DType::BF16, "b");  // [NH, U, 1]

    // Pad K, V along U -> K_pad (multiple of 128)
    TensorId k_pad = k3, v_pad = v3;
    if (pad_k > 0) {
        const auto zk = o.constant(0.0f, {NH, pad_k, HD}, "zk", DType::BF16);
        k_pad = o.concat({k3, zk}, 1, "kp");
        v_pad = o.concat({v3, zk}, 1, "vp");
    }

    // Pad Q_sc and bound to U_pad = Q_blocks * q_chunk, then pack to [H_groups, Q_blocks, h_sub, q_chunk, D]
    TensorId q_pad = q_sc, b_pad = bound;
    if (pad_q > 0) {
        q_pad = o.concat({q_sc, o.constant(0.0f, {NH, pad_q, HD}, "zq", DType::BF16)}, 1, "qp");
        b_pad = o.concat({bound, o.constant(0.0f, {NH, pad_q, 1}, "zb", DType::BF16)}, 1, "bp");
    }
    TensorId q_out = q_pad, b_out = b_pad;
    if (Q_blocks > 1) {
        q_out = o.transpose(o.reshape(q_pad, {H_groups, h_sub, Q_blocks, q_chunk, HD}, "qr5"),
                            {0, 2, 1, 3, 4}, "qt5");
        b_out = o.transpose(o.reshape(b_pad, {H_groups, h_sub, Q_blocks, q_chunk, 1}, "br5"),
                            {0, 2, 1, 3, 4}, "bt5");
    }
    set_output(g, "q_chunks", o.reshape(q_out, {H_groups, Q_blocks, h_sub, q_chunk, HD}, "qo"));
    set_output(g, "b_chunks", o.reshape(b_out, {H_groups, Q_blocks, h_sub, q_chunk, 1}, "bo"));
    set_output(g, "k_pad", o.reshape(k_pad, {H_groups, h_sub, K_pad, HD}, "ko"));
    set_output(g, "v_pad", o.reshape(v_pad, {H_groups, h_sub, K_pad, HD}, "vo"));
    set_output(g, "ad_post", o.concat({ad[2], ad[3], ad[4], ad[5]}, 0, "adp"));

    const std::string rname = "h3_dit_pre_" + std::to_string(S) + "_u" + std::to_string(U) +
                              "_r" + std::to_string(lora_rank) +
                              (adaln_rank > 0 ? ("_ar" + std::to_string(adaln_rank)) : "") +
                              "_h" + std::to_string(h_sub) +
                              "_q" + std::to_string(q_chunk);
    auto r = g.compile(rname);
    auto& ref = *r;
    r_dit_pre_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_maxfree_chunk_recipe(uint32_t h_sub, uint32_t q_chunk, uint32_t k_pad, uint32_t pad_k) {
    MfChunkKey key{h_sub, q_chunk, k_pad, pad_k};
    if (auto it = r_mf_chunk_.find(key); it != r_mf_chunk_.end()) return *it->second;
    const uint64_t H = h_sub, Q = q_chunk, K = k_pad, D = cfg_.head_dim, R = H * Q;
    const bool use_tpc = (std::getenv("G2E_H3_UNFUSED_MF_EXP") == nullptr);

    GraphBuilder g(dev_);
    Ops o(g, "mfc_");
    const auto qc = g.input("q_c", DType::BF16, {H, Q, D});
    const auto kg = g.input("k_g", DType::BF16, {H, K, D});
    const auto vg = g.input("v_g", DType::BF16, {H, K, D});
    const auto bc = g.input("b_c", DType::BF16, {R, 1});

    const auto scores = o.batch_gemm(qc, kg, "sc", false, true, DType::BF16);  // [H, Q, K]
    TensorId probs3, inv_sum3;
    if (use_tpc) {
        const auto sc2 = o.reshape(scores, {R, K}, "sc2");
        const auto pr2 = o.temp("pr2", DType::BF16, {R, K});
        const auto is2 = o.temp("is2", DType::F32, {R, 1});
        const float pad_f = float(pad_k);
        g.op("flashinfer_gaudi_h3_maxfree_exp_bf16_gaudi2", {sc2, bc}, {pr2, is2}, &pad_f, sizeof(pad_f));
        probs3 = o.reshape(pr2, {H, Q, K}, "pr3");
        inv_sum3 = o.reshape(is2, {H, Q, 1}, "is3");
    } else {
        const auto b3 = o.reshape(bc, {H, Q, 1}, "b3");
        const auto shifted = o.binary("min", o.sub(scores, b3, "sh"), o.constant(30.0f, {1}, "c30", DType::BF16), "cl");
        probs3 = o.unary("exp", shifted, "pr");
        TensorId rsum = o.reduce("sum", o.cast(probs3, DType::F32, "prf"), 2, "rs");
        if (pad_k > 0) {
            const auto neg_b = o.unary("neg", o.cast(b3, DType::F32, "bf"), "nb");
            rsum = o.sub(rsum, o.mul(o.constant(float(pad_k), {1}, "pf", DType::F32), o.unary("exp", neg_b, "ep"), "pc"), "rsc");
        }
        rsum = o.binary("max", rsum, o.constant(1e-30f, {1}, "eps", DType::F32), "rsm");
        inv_sum3 = o.div(o.constant(1.0f, {1}, "one", DType::F32), rsum, "is3");
    }
    const auto pv = o.batch_gemm(probs3, vg, "pv", false, false, DType::BF16);  // [H, Q, D]
    set_output(g, "out", o.cast(o.mul(o.cast(pv, DType::F32, "pvf"), inv_sum3, "norm"), DType::BF16, "ob"));

    const std::string rname = "h3_mfc_h" + std::to_string(H) + "_q" + std::to_string(Q) +
                              "_k" + std::to_string(K) + "_p" + std::to_string(pad_k) +
                              (use_tpc ? "_tpc" : "_syn");
    auto r = g.compile(rname);
    auto& ref = *r;
    r_mf_chunk_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_dit_pre_fp8_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank,
                                      uint32_t h_sub, uint32_t q_chunk, bool kv_use_upad, bool with_blk_scores,
                                      bool v_aug_256, uint32_t adaln_rank) {
    const bool env_unfused_prep = (std::getenv("G2E_H3_UNFUSED_PREP") != nullptr);
    const uint32_t flags = (kv_use_upad ? 1u : 0u) | (with_blk_scores ? 2u : 0u) |
                           (v_aug_256 ? 4u : 0u) | (env_unfused_prep ? 8u : 0u);
    MfKey key{seq_len, used_len, lora_rank, adaln_rank, h_sub, q_chunk, flags};
    if (auto it = r_dit_pre_fp8_.find(key); it != r_dit_pre_fp8_.end()) return *it->second;
    const uint64_t S = seq_len, U = used_len, H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.head_dim;
    const uint64_t AD = NH * HD;
    const uint64_t Q_blocks = (U + q_chunk - 1) / q_chunk;
    const uint64_t U_pad = Q_blocks * q_chunk;
    const uint64_t K_pad = kv_use_upad ? U_pad : (((U + 255) / 256) * 256);
    const uint64_t pad_k = K_pad - U;
    const uint64_t pad_q = U_pad - U;
    const uint64_t H_groups = NH / h_sub;
    const uint64_t V_W = v_aug_256 ? 256ull : HD;
    const bool use_fused_prep = kv_use_upad && (lora_rank == 0) && (Q_blocks == 1) && (U_pad % 256 == 0) &&
                                (NH % 2 == 0) && (HD == 128) && (std::getenv("G2E_H3_UNFUSED_PREP") == nullptr);
    const bool emit_blk_scores = kv_use_upad && with_blk_scores && (U_pad % 1536 == 0) && (U >= 8192);

    GraphBuilder g(dev_);
    Ops o(g, "dp8_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto t_emb = g.input("t_emb", DType::F32, {kPadM, cfg_.time_embed_dim});
    const auto comb_idx = g.input("comb_idx", DType::I32, {S});
    const auto rope_cos = g.input("rope_cos", DType::BF16, {S, HD});
    const auto rope_sin = g.input("rope_sin", DType::BF16, {S, HD});
    const auto norm1 = g.input("norm1", DType::BF16, {H});
    const auto q_norm = g.input("q_norm", DType::BF16, {HD});
    const auto k_norm = g.input("k_norm", DType::BF16, {HD});
    const auto qkv_w = g.input("qkv_w", DType::F8, {3 * AD, H});
    const auto qkv_s = g.input("qkv_s", DType::F32, {3 * AD});
    const auto adaln_w = g.input("adaln_w", cfg_.adaln_bf16 ? DType::BF16 : DType::F8, {18 * H, cfg_.time_embed_dim});
    const auto adaln_s = cfg_.adaln_bf16 ? kNoTensor : g.input("adaln_s", DType::F32, {18 * H});
    const auto adaln_b = g.input("adaln_b", DType::BF16, {18 * H});

    TensorId l_qkv_a = kNoTensor, l_qkv_b = kNoTensor, l_adaln_a = kNoTensor, l_adaln_b = kNoTensor;
    if (lora_rank > 0) {
        l_qkv_a = g.input("lora_qkv_a", DType::BF16, {3 * uint64_t(lora_rank), H});
        l_qkv_b = g.input("lora_qkv_b", DType::BF16, {3, AD, lora_rank});
    }
    if (adaln_rank > 0) {
        l_adaln_a = g.input("lora_adaln_a", DType::BF16, {adaln_rank, cfg_.time_embed_dim});
        l_adaln_b = g.input("lora_adaln_b", DType::BF16, {18 * H, adaln_rank});
    }

    const auto t_act = o.cast(o.silu(t_emb, "ts"), DType::BF16, "tsb");
    const auto adaln_raw = adaln_linear(o, t_act, adaln_w, adaln_s, adaln_b, "adaln", l_adaln_a, l_adaln_b);
    const auto adaln_3 = o.reshape(adaln_raw, {kPadM * 3, 6 * H}, "ad3");
    const auto ad = o.split(adaln_3, 1, {H, H, H, H, H, H}, "ad_sp");

    struct H3NormQuantParams { float eps, inv_w, inv_range; };
    const H3NormQuantParams nqp{cfg_.norm_eps, 1.0f / float(H), 0.004180908203125f};
    const auto h1c = o.temp("h1_codes", DType::F8, {S, H});
    const auto h1s = o.temp("h1_scale", DType::F32, {S, 1});
    const auto h1 = o.temp("h1", DType::BF16, {S, H});
    g.op("flashinfer_gaudi_h3_adaln_rmsnorm_quant_bf16_gaudi2",
         {x, norm1, ad[0], ad[1], comb_idx}, {h1c, h1s, h1}, &nqp, sizeof(nqp));

    const float attn_scale = float(1.0 / std::sqrt(double(HD)));
    TensorId q_out = kNoTensor, k_pad = kNoTensor, v_int = kNoTensor, sc = kNoTensor, dsv = kNoTensor;
    TensorId q_b_blk = kNoTensor, k_b_blk = kNoTensor;

    if (use_fused_prep) {
        // Raw FP8 QKV GEMM (no separate dequant pass!) -> fused dequant + 64-row strip Q/K RMSNorm + 3D RoPE + pad + V interleave + block max!
        const auto qkv_raw = o.temp("qkv_raw", DType::BF16, {S, 3 * AD});
        ns_Fp8Gemm::ParamsV2 gp{};
        gp.transpose_a = false;
        gp.transpose_b = true;
        g.op("fp8_gemm_bf16", {h1c, qkv_w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {qkv_raw}, &gp, sizeof(gp));
        const auto qkv_sb = o.cast(qkv_s, DType::BF16, "qkvsb");

        const auto q_b = o.temp("qb", DType::BF16, {NH, U_pad, HD});
        const auto k_b = o.temp("kb", DType::BF16, {NH, U_pad, HD});
        const auto v_b = o.temp("vb", DType::BF16, {NH, U_pad, HD});
        const auto max_qkv = o.temp("mqkv", DType::F32, {U_pad / 256, 3});
        struct PrepParams { float eps; int32_t valid_u; };
        const PrepParams pp{cfg_.qk_norm_eps, int32_t(U)};
        g.op("flashinfer_gaudi_h3_qk_norm_rope_prep_bf16_gaudi2",
             {qkv_raw, h1s, qkv_sb, q_norm, k_norm, rope_cos, rope_sin}, {q_b, k_b, v_b, max_qkv}, &pp, sizeof(pp));
        q_b_blk = q_b;
        k_b_blk = k_b;

        const auto pk3 = o.reshape(o.reduce("max", max_qkv, 0, "pk3"), {3}, "pk3f");
        const auto pks = o.split(pk3, 0, {1, 1, 1}, "pks");
        auto convert_fp8_with_peak = [&](TensorId src, TensorId pk, float max_val, const std::string& nm) {
            const auto scale = o.mul(o.add(pk, o.constant(1e-8f, {1}, nm + "_eps", DType::F32), nm + "_pe"),
                                     o.constant(1.0f / max_val, {1}, nm + "_imax", DType::F32), nm + "_sc");
            const auto inv = o.unary("reciprocal", scale, nm + "_inv");
            const auto codes = o.temp(nm + "_f8", DType::F8, o.shape(src));
            ns_ConvertToFp8::ParamsV2 cp{};
            cp.round_mode = CAST_ROUND_HALF_NE;
            g.op("convert_to_fp8_bf16", {src, inv}, {codes}, &cp, sizeof(cp));
            return std::make_pair(codes, scale);
        };
        const auto [q8, dsq] = convert_fp8_with_peak(q_b, pks[0], 22.0f, "q8");
        const auto [k8, dsk] = convert_fp8_with_peak(k_b, pks[1], 22.0f, "k8");
        q_out = q8;
        k_pad = k8;
        if (v_aug_256) {
            const auto v_scale = o.mul(o.add(pks[2], o.constant(1e-8f, {1}, "v8_eps", DType::F32), "v8_pe"),
                                       o.constant(1.0f / 240.0f, {1}, "v8_imax", DType::F32), "v8_sc");
            const auto v_inv = o.unary("reciprocal", v_scale, "v8_inv");
            const auto v8 = o.temp("v8_aug", DType::F8, {NH, U_pad, 256});
            g.op("flashinfer_gaudi_h3_quant_v256_bf16_gaudi2", {v_b, v_inv}, {v8});
            v_int = v8;
            dsv = v_scale;
        } else {
            const auto [v8, dv] = convert_fp8_with_peak(v_b, pks[2], 240.0f, "v8");
            v_int = v8;
            dsv = dv;
        }
        sc = o.mul(o.mul(dsq, dsk, "dsqk"), o.constant(attn_scale, {1}, "as", DType::F32), "sc");
    } else {
        const auto qkv = proj_with_lora(o, h1, h1c, h1s, qkv_w, qkv_s, l_qkv_a, l_qkv_b, true, "qkv");
        const auto q3 = o.temp("qf3", DType::BF16, {NH, U, HD});
        const auto k3 = o.temp("kf3", DType::BF16, {NH, U, HD});
        const auto v3 = o.temp("vf3", DType::BF16, {NH, U, HD});
        const float qk_eps = cfg_.qk_norm_eps;
        g.op("flashinfer_gaudi_h3_qk_norm_rope_bf16_gaudi2",
             {qkv, q_norm, k_norm, rope_cos, rope_sin}, {q3, k3, v3}, &qk_eps, sizeof(qk_eps));

        // Pad K, V along U -> K_pad (multiple of 256) and Q along U -> U_pad in BF16 before FP8 quant
        TensorId k_b = k3, v_b = v3, q_b = q3;
        if (pad_k > 0) {
            const auto zk = o.constant(0.0f, {NH, pad_k, HD}, "zk", DType::BF16);
            k_b = o.concat({k3, zk}, 1, "kp");
            v_b = o.concat({v3, zk}, 1, "vp");
        }
        if (pad_q > 0) {
            q_b = o.concat({q3, o.constant(0.0f, {NH, pad_q, HD}, "zq", DType::BF16)}, 1, "qp");
        }
        q_b_blk = q_b;
        k_b_blk = k_b;
        const auto [q_pad, dsq] = quantize_tensor_fp8_nd(o, q_b, 22.0f, "q8");
        const auto [k8, dsk] = quantize_tensor_fp8_nd(o, k_b, 22.0f, "k8");
        k_pad = k8;
        sc = o.mul(o.mul(dsq, dsk, "dsqk"), o.constant(attn_scale, {1}, "as", DType::F32), "sc");
        q_out = q_pad;
        if (Q_blocks > 1) {
            q_out = o.transpose(o.reshape(q_pad, {H_groups, h_sub, Q_blocks, q_chunk, HD}, "qr5"),
                                {0, 2, 1, 3, 4}, "qt5");
        }
        if (v_aug_256) {
            const auto v_bi = o.reshape(
                o.transpose(o.reshape(v_b, {NH, K_pad / 256, 2, 128, HD}, "vbr5"), {0, 1, 3, 2, 4}, "vbt5"),
                {NH, K_pad, HD}, "vbi");
            const auto ax = o.unary("abs", o.reshape(v_bi, {NH * K_pad, HD}, "v8_2d"), "v8_abs");
            const auto m1 = o.reshape(o.reduce("max", ax, 1, "v8_m1"), {NH, K_pad}, "v8_m1r");
            const auto m2 = o.reduce("max", m1, 1, "v8_m2");
            const auto peak = o.cast(o.reshape(o.reduce("max", m2, 0, "v8_m3"), {1}, "v8_pk"), DType::F32, "v8_pkf");
            dsv = o.mul(o.add(peak, o.constant(1e-8f, {1}, "v8_eps", DType::F32), "v8_pe"),
                        o.constant(1.0f / 240.0f, {1}, "v8_imax", DType::F32), "v8_sc");
            const auto v_inv = o.unary("reciprocal", dsv, "v8_inv");
            v_int = o.temp("v8_aug", DType::F8, {NH, K_pad, 256});
            g.op("flashinfer_gaudi_h3_quant_v256_bf16_gaudi2", {v_bi, v_inv}, {v_int});
        } else {
            const auto [v_pad, dv] = quantize_tensor_fp8_nd(o, v_b, 240.0f, "v8");
            dsv = dv;
            v_int = o.transpose(o.reshape(v_pad, {NH, K_pad / 256, 2, 128, HD}, "vr5"),
                                {0, 1, 3, 2, 4}, "vt5");
        }
    }

    if (emit_blk_scores) {
        // Route B dynamic block-importance scores [NH, U_pad/1536, U/4096] F32 (~419 KiB on 15s!):
        // Pool Q into 512-tok (~0.5 frame) sub-blocks and K into 1024-tok (~1 frame) sub-blocks,
        // dot-product in F32, then max-pool over (3 x 4) sub-blocks per (1536-Q, 4096-K) tile
        // to avoid cross-frame 3D-RoPE cancellation while keeping full global & temporal fidelity!
        const uint64_t qb = U_pad / 1536;
        const uint64_t nf = U / 4096;
        const auto q_sub = o.reshape(o.reduce("sum", o.reshape(q_b_blk, {NH * qb * 3, 512, HD}, "sqr3"), 1, "sqsm"),
                                     {NH, qb * 3, HD}, "sqsub");
        const auto k_full = (nf * 4096 < U_pad)
            ? o.split(k_b_blk, 1, {nf * 4096, U_pad - nf * 4096}, "skfsp")[0]
            : k_b_blk;
        const auto k_sub = o.reshape(o.reduce("sum", o.reshape(k_full, {NH * nf * 4, 1024, HD}, "skr3"), 1, "sksm"),
                                     {NH, nf * 4, HD}, "sksub");
        const auto sub_sc = o.batch_gemm(q_sub, k_sub, "sbsc", false, true, DType::F32);
        const auto mq = o.reshape(o.reduce("max", o.reshape(sub_sc, {NH * qb, 3, nf * 4}, "sbsq"), 1, "sbsmq"),
                                  {NH * qb, nf, 4}, "sbsk");
        set_output(g, "blk_scores", o.reshape(o.reduce("max", mq, 2, "sbsmk"), {NH, qb, nf}, "sbso"));
    }

    set_output(g, "q_chunks", o.reshape(q_out, {H_groups, Q_blocks, h_sub, q_chunk, HD}, "qo"));
    set_output(g, "k_pad", o.reshape(k_pad, {H_groups, h_sub, K_pad, HD}, "ko"));
    set_output(g, "v_pad", o.reshape(v_int, {H_groups, h_sub, K_pad, V_W}, "vo"));
    set_output(g, "score_scale", sc);
    set_output(g, "d_scale_v", dsv);
    set_output(g, "ad_post", o.concat({ad[2], ad[3], ad[4], ad[5]}, 0, "adp"));

    const std::string rname = "h3_dit_pre8_" + std::to_string(S) + "_u" + std::to_string(U) +
                              "_r" + std::to_string(lora_rank) +
                              (adaln_rank > 0 ? ("_ar" + std::to_string(adaln_rank)) : "") +
                              "_h" + std::to_string(h_sub) +
                              "_q" + std::to_string(q_chunk) +
                              (use_fused_prep ? "_fp3" : (kv_use_upad ? "_ku" : "")) +
                              (emit_blk_scores ? "_sp" : "") +
                              (v_aug_256 ? "_v256" : "");
    auto r = g.compile(rname);
    auto& ref = *r;
    r_dit_pre_fp8_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_fast_fp8_chunk_recipe(uint32_t h_sub, uint32_t q_chunk, uint32_t k_pad, uint32_t valid_k) {
    MfChunkKey key{h_sub, q_chunk, k_pad, valid_k};
    if (auto it = r_fp8_chunk_.find(key); it != r_fp8_chunk_.end()) return *it->second;
    const uint64_t H = h_sub, Q = q_chunk, K = k_pad, D = cfg_.head_dim, R = H * Q;

    GraphBuilder g(dev_);
    Ops o(g, "ff8_");
    const auto qc = g.input("q_c", DType::F8, {H, Q, D});
    const auto kg = g.input("k_g", DType::F8, {H, K, D});
    const auto vg = g.input("v_g", DType::F8, {H, K, D});
    const auto sc = g.input("score_scale", DType::F32, {1});
    const auto dv = g.input("d_scale_v", DType::F32, {1});

    // FP8 Q@K^T -> BF16 scores (865 TFLOP/s FP8 MME)
    const auto scores = o.batch_gemm(qc, kg, "sc", false, true, DType::BF16);
    const auto sc2 = o.reshape(scores, {R, K}, "sc2");
    const auto pr2 = o.temp("pr2", DType::F8, {R, K});
    const auto is2 = o.temp("is2", DType::F32, {R, 1});
    const int32_t vk = int32_t(valid_k);
    g.op("flashinfer_gaudi_h3_fast_softmax_fp8_gaudi2", {sc2, sc, dv}, {pr2, is2}, &vk, sizeof(vk));

    // FP8 P@V -> F32 (865 TFLOP/s FP8 MME + 50% HBM P traffic)
    const auto pv = o.batch_gemm(o.reshape(pr2, {H, Q, K}, "pr3"), vg, "pv", false, false, DType::F32);
    set_output(g, "out", o.cast(o.mul(pv, o.reshape(is2, {H, Q, 1}, "is3"), "nm"), DType::BF16, "ob"));

    const std::string rname = "h3_ff8_h" + std::to_string(H) + "_q" + std::to_string(Q) +
                              "_k" + std::to_string(K) + "_v" + std::to_string(valid_k);
    auto r = g.compile(rname);
    auto& ref = *r;
    r_fp8_chunk_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_dit_post_recipe(uint32_t seq_len, uint32_t used_len, uint32_t lora_rank,
                                   uint32_t h_sub, uint32_t q_chunk) {
    MfKey key{seq_len, used_len, lora_rank, 0u, h_sub, q_chunk, 0u};
    if (auto it = r_dit_post_.find(key); it != r_dit_post_.end()) return *it->second;
    const uint64_t S = seq_len, U = used_len, H = cfg_.hidden_size, NH = cfg_.num_heads, HD = cfg_.head_dim;
    const uint64_t AD = NH * HD, FFN = cfg_.ffn_hidden;
    const bool direct_flat = (q_chunk == 0);
    const uint64_t Q_blocks = direct_flat ? 1ull : ((U + q_chunk - 1) / q_chunk);
    const uint64_t U_pad = direct_flat ? S : (Q_blocks * q_chunk);
    const uint64_t H_groups = NH / h_sub;

    GraphBuilder g(dev_);
    Ops o(g, "dpo_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto ctx_chunks = direct_flat
        ? g.input("ctx_chunks", DType::BF16, {S, AD})
        : g.input("ctx_chunks", DType::BF16, {H_groups, Q_blocks, h_sub, q_chunk, HD});
    const auto ad_post = g.input("ad_post", DType::BF16, {4 * kPadM * 3, H});
    const auto comb_idx = g.input("comb_idx", DType::I32, {S});
    const auto norm2 = g.input("norm2", DType::BF16, {H});
    const auto out_w = g.input("out_w", DType::F8, {H, AD});
    const auto out_s = g.input("out_s", DType::F32, {H});
    const auto fc1_w = g.input("fc1_w", DType::F8, {2 * FFN, H});
    const auto fc1_s = g.input("fc1_s", DType::F32, {2 * FFN});
    const auto fc2_w = g.input("fc2_w", DType::F8, {H, FFN});
    const auto fc2_s = g.input("fc2_s", DType::F32, {H});

    TensorId l_out_a = kNoTensor, l_out_b = kNoTensor;
    TensorId l_fc1_a = kNoTensor, l_fc1_b = kNoTensor, l_fc2_a = kNoTensor, l_fc2_b = kNoTensor;
    if (lora_rank > 0) {
        l_out_a = g.input("lora_out_a", DType::BF16, {lora_rank, AD});
        l_out_b = g.input("lora_out_b", DType::BF16, {H, lora_rank});
        l_fc1_a = g.input("lora_fc1_a", DType::BF16, {lora_rank, H});
        l_fc1_b = g.input("lora_fc1_b", DType::BF16, {2 * FFN, lora_rank});
        l_fc2_a = g.input("lora_fc2_a", DType::BF16, {lora_rank, FFN});
        l_fc2_b = g.input("lora_fc2_b", DType::BF16, {H, lora_rank});
    }

    TensorId ctx;
    if (direct_flat) {
        // DirectH3Attn already wrote [S, NH, 128] == [S, AD] row-major with [U..S) pre-zeroed!
        ctx = ctx_chunks;
    } else {
        // Unpack [H_groups, Q_blocks, h_sub, q_chunk, HD] -> [NH, U_pad, HD] -> slice U -> [U, AD] -> pad S
        TensorId c3 = (Q_blocks > 1)
            ? o.reshape(o.transpose(ctx_chunks, {0, 2, 1, 3, 4}, "ct5"), {NH, U_pad, HD}, "c3p")
            : o.reshape(ctx_chunks, {NH, U_pad, HD}, "c3p");
        if (U_pad > U) c3 = o.split(c3, 1, {U, U_pad - U}, "cu")[0];
        const auto ctx_u = o.reshape(o.transpose(c3, {1, 0, 2}, "ct"), {U, AD}, "cf");
        ctx = zero_pad_axis0(o, ctx_u, S, "ctx");
    }

    const uint64_t M3 = kPadM * 3;
    const auto ad = o.split(ad_post, 0, {M3, M3, M3, M3}, "ads");
    // ad[0]=gate_msa, ad[1]=shift_mlp, ad[2]=scale_mlp, ad[3]=gate_mlp

    const auto ctxc = o.temp("ctxq_codes", DType::F8, {S, AD});
    const auto ctxs = o.temp("ctxq_scale", DType::F32, {S, 1});
    g.op("flashinfer_gaudi_h3_row_quant_bf16_gaudi2", {ctx}, {ctxc, ctxs});
    const auto aout = proj_with_lora(o, ctx, ctxc, ctxs, out_w, out_s, l_out_a, l_out_b, false, "out");

    struct H3NormQuantParams { float eps, inv_w, inv_range; };
    const H3NormQuantParams nqp{cfg_.norm_eps, 1.0f / float(H), 0.004180908203125f};
    const auto h2c = o.temp("h2_codes", DType::F8, {S, H});
    const auto h2s = o.temp("h2_scale", DType::F32, {S, 1});
    const auto h2 = o.temp("h2", DType::BF16, {S, H});
    const auto x1 = o.temp("x1", DType::BF16, {S, H});
    g.op("flashinfer_gaudi_h3_gate_adaln_rmsnorm_quant_bf16_gaudi2",
         {x, aout, ad[0], norm2, ad[1], ad[2], comb_idx}, {h2c, h2s, h2, x1}, &nqp, sizeof(nqp));

    TensorId mout;
    if (lora_rank == 0) {
        const auto [mc, ms] = o.fp8_gemm_silu_mul_quant(h2c, h2s, fc1_w, fc1_s, "fc1");
        mout = o.fp8_gemm(mc, ms, fc2_w, fc2_s, "fc2");
    } else {
        const auto fc1_raw = o.temp("fc1_raw", DType::BF16, {S, 2 * FFN});
        ns_Fp8Gemm::ParamsV2 gp{};
        gp.transpose_a = false;
        gp.transpose_b = true;
        g.op("fp8_gemm_bf16", {h2c, fc1_w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {fc1_raw}, &gp, sizeof(gp));
        const auto fc1_lo = o.linear_bf16(h2, l_fc1_a, "fc1_la");
        const auto fc1_dl = o.linear_bf16(fc1_lo, l_fc1_b, "fc1_lb");
        const auto ac = o.temp("actq_codes", DType::F8, {S, FFN});
        const auto as = o.temp("actq_scale", DType::F32, {S, 1});
        const auto act = o.temp("act", DType::BF16, {S, FFN});
        g.op("flashinfer_gaudi_h3_silu_mul_quant_dual_bf16_gaudi2",
             {fc1_raw, h2s, o.cast(fc1_s, DType::BF16, "fc1sb"), fc1_dl}, {ac, as, act});
        mout = proj_with_lora(o, act, ac, as, fc2_w, fc2_s, l_fc2_a, l_fc2_b, false, "fc2");
    }
    const auto x2 = o.temp("x2", DType::BF16, {S, H});
    g.op("flashinfer_gaudi_h3_gate_residual_bf16_gaudi2", {x1, mout, ad[3], comb_idx}, {x2});
    set_output(g, "out", x2);

    const std::string rname = "h3_dit_post_" + std::to_string(S) + "_u" + std::to_string(U) +
                              "_r" + std::to_string(lora_rank) + "_h" + std::to_string(h_sub) +
                              "_q" + std::to_string(q_chunk) + "_v3";
    auto r = g.compile(rname);
    auto& ref = *r;
    r_dit_post_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_final_recipe(uint32_t seq_len, uint32_t adaln_rank) {
    const auto key = std::make_pair(seq_len, adaln_rank);
    if (auto it = r_final_.find(key); it != r_final_.end()) return *it->second;
    const uint64_t S = seq_len, H = cfg_.hidden_size, VW = cfg_.video_row_width(), AW = cfg_.audio_latents_dim;
    GraphBuilder g(dev_);
    Ops o(g, "fn_");
    const auto x = g.input("x", DType::BF16, {S, H});
    const auto t_emb = g.input("t_emb", DType::F32, {kPadM, cfg_.time_embed_dim});
    const auto inv_idx = g.input("inv_idx", DType::I32, {S});
    const auto norm_w = g.input("norm_w", DType::BF16, {H});
    const auto adaln_w = g.input("adaln_w", cfg_.adaln_bf16 ? DType::BF16 : DType::F8, {2 * H, cfg_.time_embed_dim});
    const auto adaln_s = cfg_.adaln_bf16 ? kNoTensor : g.input("adaln_s", DType::F32, {2 * H});
    const auto adaln_b = g.input("adaln_b", DType::BF16, {2 * H});
    TensorId l_adaln_a = kNoTensor, l_adaln_b = kNoTensor;
    if (adaln_rank > 0) {
        l_adaln_a = g.input("lora_adaln_a", DType::BF16, {adaln_rank, cfg_.time_embed_dim});
        l_adaln_b = g.input("lora_adaln_b", DType::BF16, {2 * H, adaln_rank});
    }
    const auto vid_w = g.input("vid_w", DType::F32, {VW, H});
    const auto vid_b = g.input("vid_b", DType::F32, {VW});
    TensorId aud_w = kNoTensor, aud_b = kNoTensor;
    if (cfg_.audio_enabled) {
        aud_w = g.input("aud_w", DType::F32, {AW, H});
        aud_b = g.input("aud_b", DType::F32, {AW});
    }

    const auto t_act = o.cast(o.silu(t_emb, "ts"), DType::BF16, "tsb");
    const auto adaln_raw = adaln_linear(o, t_act, adaln_w, adaln_s, adaln_b, "adaln", l_adaln_a, l_adaln_b);
    const auto ad = o.split(adaln_raw, 1, {H, H}, "ad_sp");
    const auto one_f = o.constant(1.0f, {1}, "one", DType::F32);
    const auto nh = o.rms_norm(x, norm_w, cfg_.final_norm_eps, false, "norm");
    const auto sh = o.cast(o.gather_rows(ad[0], inv_idx, "sh"), DType::F32, "shf");
    const auto sc = o.add(o.cast(o.gather_rows(ad[1], inv_idx, "sc"), DType::F32, "scf"), one_f, "sc1");
    // Match reference: indexed_scale_shift_ rounds to BF16, then final_layer casts to FP32 before heads.
    const auto hb = o.cast(o.add(o.mul(o.cast(nh, DType::F32, "nhf"), sc, "nhs"), sh, "nha"), DType::BF16, "hb");
    const auto hf = o.cast(hb, DType::F32, "hf");

    const auto vo = o.temp("vo", DType::F32, {S, VW});
    g.gemm(hf, vid_w, vo, false, true);
    set_output(g, "video_out", add_bias(o, vo, vid_b, "vob"));

    if (cfg_.audio_enabled) {
        const auto ao = o.temp("ao", DType::F32, {S, AW});
        g.gemm(hf, aud_w, ao, false, true);
        set_output(g, "audio_out", add_bias(o, ao, aud_b, "aob"));
    }

    auto r = g.compile("h3_final_" + std::to_string(S) + (cfg_.audio_enabled ? "" : "_noaud") +
                       (adaln_rank > 0 ? ("_ar" + std::to_string(adaln_rank)) : ""));
    auto& ref = *r;
    r_final_.emplace(key, std::move(r));
    return ref;
}

Recipe& H3DiT::get_merge_weight_recipe(uint64_t N, uint64_t K, uint64_t R, uint64_t G) {
    auto key = std::make_tuple(N, K, R, G);
    if (auto it = r_merge_w_.find(key); it != r_merge_w_.end()) return *it->second;
    GraphBuilder g(dev_);
    Ops o(g, "ml_");
    const auto w_base = g.input("w_base", DType::F8, {N, K});
    const auto s_base = g.input("s_base", DType::F32, {N});
    const auto lora_a = g.input("lora_a", DType::BF16, {G * R, K});
    const auto lora_b = (G > 1) ? g.input("lora_b", DType::BF16, {G, N / G, R})
                                : g.input("lora_b", DType::BF16, {N, R});

    const auto a3 = o.reshape(lora_a, {G, R, K}, "a3");
    const auto b3 = (G > 1) ? lora_b : o.reshape(lora_b, {1, N, R}, "b3");
    const auto delta = o.reshape(o.batch_gemm(b3, a3, "dm", false, false, DType::F32), {N, K}, "df");

    const auto w_f32 = o.temp("wf", DType::F32, {N, K});
    ns_CastKernel::Params cp{};
    cp.round_mode = CAST_ROUND_HALF_NE;
    g.op("cast_hf8_to_f32", {w_base}, {w_f32}, &cp, sizeof(cp));
    const auto w_deq = o.mul(w_f32, o.reshape(s_base, {N, 1}, "sb"), "wd");
    const auto w_sum = o.cast(o.add(w_deq, delta, "ws"), DType::BF16, "wb");
    const auto [codes, sc2] = o.quantize(w_sum, "wq");
    set_output(g, "w_out", codes);
    set_output(g, "s_out", o.reshape(sc2, {N}, "so"));

    auto r = g.compile("h3_merge_w_" + std::to_string(N) + "_" + std::to_string(K) +
                       "_r" + std::to_string(R) + "_g" + std::to_string(G));
    auto& ref = *r;
    r_merge_w_.emplace(key, std::move(r));
    return ref;
}

void H3DiT::restore_base_block_weights() {
    if (!merged_active_) return;
    auto st = w_.reload_subset_in_place(dev_, stream_, [](const WeightEntry& e) {
        if (e.name.size() < 6 || e.name[0] != 'B') return false;
        return e.name.find(".qkv.") != std::string::npos ||
               e.name.find(".out.") != std::string::npos ||
               e.name.find(".fc1.") != std::string::npos ||
               e.name.find(".fc2.") != std::string::npos;
    });
    merged_active_ = false;
    merged_lora_path_.clear();
    merged_lora_scale_ = 0.0f;
    std::printf("[H3DiT] restored base FP8 block weights (%.2f GiB in %.2f s)\n", st.gib, st.seconds);
}

void H3DiT::ensure_merged_lora(const H3LoraAdapter* lora) {
    if (!lora) {
        if (merged_active_) restore_base_block_weights();
        return;
    }
    if (merged_active_ && merged_lora_path_ == lora->path() && merged_lora_scale_ == lora->scale()) {
        return;
    }
    if (merged_active_) restore_base_block_weights();

    const auto t0 = Clock::now();
    const uint64_t H = cfg_.hidden_size, AD = cfg_.attn_dim(), FFN = cfg_.ffn_hidden, R = lora->rank();
    const uint64_t max_w_bytes = std::max({3 * AD * H, H * AD, 2 * FFN * H, H * FFN});
    const uint64_t max_s_bytes = std::max({3 * AD, 2 * FFN, H}) * 4;
    DeviceBuffer tmp_w(dev_, align4k(max_w_bytes));
    DeviceBuffer tmp_s(dev_, align4k(max_s_bytes));

    auto& r_qkv = get_merge_weight_recipe(3 * AD, H, R, 3);
    auto& r_out = get_merge_weight_recipe(H, AD, R, 1);
    auto& r_fc1 = get_merge_weight_recipe(2 * FFN, H, R, 1);
    auto& r_fc2 = get_merge_weight_recipe(H, FFN, R, 1);

    for (uint32_t bi = 0; bi < cfg_.num_layers; ++bi) {
        const std::string p = "B" + std::to_string(bi) + ".";
        auto run_m = [&](Recipe& rec, const std::string& wname, int ia, int ib) {
            const auto& ew = w_.at(p + wname + ".w");
            const auto& es = w_.at(p + wname + ".s");
            std::vector<Binding> b = {
                {"w_base", &w_.arena(), ew.offset},
                {"s_base", &w_.arena(), es.offset},
                lora->bind_one("lora_a", false, bi, ia),
                lora->bind_one("lora_b", false, bi, ib),
                {"w_out", &tmp_w, 0},
                {"s_out", &tmp_s, 0},
            };
            rec.enqueue(stream_, b);
            stream_.copy(tmp_w, 0, w_.arena(), ew.offset, ew.bytes);
            stream_.copy(tmp_s, 0, w_.arena(), es.offset, es.bytes);
        };
        run_m(r_qkv, "qkv", 0, 1);
        run_m(r_out, "out", 2, 3);
        run_m(r_fc1, "fc1", 4, 5);
        run_m(r_fc2, "fc2", 6, 7);
    }
    stream_.synchronize();
    merged_active_ = true;
    merged_lora_path_ = lora->path();
    merged_lora_scale_ = lora->scale();
    std::printf("[H3DiT] dynamically merged LoRA in-place (rank=%u, %u blocks, 0 GiB extra HBM) in %.0f ms\n",
                uint32_t(R), cfg_.num_layers, ms_since(t0));
}

void H3DiT::prepare_lora(const H3LoraAdapter* lora) {
    // Re-quantizing low-magnitude (0.2-1.3%) BF16 distillation LoRA deltas into 3-bit-mantissa FP8 E4M3
    // loses the LoRA signal (L39 cos collapses from 0.69 -> 0.02, dark/blurry video). Keep exact BF16 runtime
    // LoRA by default (+0.3 s/step only); allow opt-in FP8 merge only via G2E_H3_MERGE_LORA=1.
    const char* ml = std::getenv("G2E_H3_MERGE_LORA");
    if (!ml || ml[0] == '0' || std::getenv("G2E_H3_NO_MERGE_LORA") != nullptr) {
        if (merged_active_) restore_base_block_weights();
        return;
    }
    ensure_merged_lora(lora);
}

// ------------------------------------------------------------------------------------------------ Forward

H3StepOutput H3DiT::forward(const H3StepInput& in, H3ForwardDebug* debug) {
    if (!in.layout || !in.video_rows || !in.prompt_embeds || (cfg_.audio_enabled && in.layout->audio_rows > 0 && !in.audio_rows))
        throw std::invalid_argument("H3DiT::forward: missing required input pointers");

    const auto t_all = Clock::now();
    const auto& L = *in.layout;
    const uint32_t S = L.seq_len, U = L.used_len;
    const uint32_t lora_rank = in.lora ? in.lora->rank() : 0u;
    ensure_buffers(S, L.text_len, L.img_rows, L.audio_rows);
    upload_rope(L, debug ? &debug->rope_table : nullptr);

    // Build per-token timesteps -> sorted unique_timesteps (padded to kPadM) + inverse_indices + combined_indices
    const uint32_t cond_start = L.text_len;
    const uint32_t audio_start = cond_start + L.cond_rows;
    const uint32_t video_start = audio_start + L.audio_rows;

    std::vector<float> tok_t(S, in.t_video);
    for (uint32_t i = 0; i < L.cond_rows; ++i) tok_t[cond_start + i] = in.imgvid_cond_t;
    for (uint32_t i = 0; i < L.audio_rows; ++i) tok_t[audio_start + i] = in.t_audio;
    for (uint32_t i = 0; i < L.video_rows; ++i) tok_t[video_start + i] = in.t_video;

    std::vector<float> uniq = tok_t;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    if (uniq.size() > kPadM) throw std::runtime_error("H3DiT: too many unique timesteps (" + std::to_string(uniq.size()) + ")");

    std::vector<int32_t> inv_idx(S), comb_idx(S);
    for (uint32_t i = 0; i < S; ++i) {
        int32_t ui = int32_t(std::lower_bound(uniq.begin(), uniq.end(), tok_t[i]) - uniq.begin());
        inv_idx[i] = ui;
        int32_t tag = std::max<int32_t>(L.token_tags[i], 0);
        comb_idx[i] = ui * 3 + tag;
    }

    // Sinusoidal t_freq [kPadM, 256] F32: [cos, sin]
    const uint32_t half_tf = cfg_.timestep_input_dim / 2;
    std::vector<float> t_freq(size_t(kPadM) * cfg_.timestep_input_dim, 0.0f);
    for (size_t m = 0; m < uniq.size(); ++m) {
        for (uint32_t i = 0; i < half_tf; ++i) {
            const float freq = std::exp(-std::log(10000.0f) * float(i) / float(half_tf));
            const float arg = uniq[m] * freq;
            t_freq[m * cfg_.timestep_input_dim + i] = std::cos(arg);
            t_freq[m * cfg_.timestep_input_dim + half_tf + i] = std::sin(arg);
        }
    }

    // Upload inputs
    auto upload_vec = [&](const void* src, size_t bytes, DeviceBuffer& dst) {
        HostBuffer stg(dev_, bytes);
        std::memcpy(stg.data(), src, bytes);
        stream_.upload(stg, 0, dst, 0, bytes);
        stream_.synchronize();
    };
    upload_vec(t_freq.data(), t_freq.size() * 4, d_t_freq_);
    upload_vec(inv_idx.data(), inv_idx.size() * 4, d_inv_idx_);
    upload_vec(comb_idx.data(), comb_idx.size() * 4, d_comb_idx_);
    upload_vec(in.video_rows, size_t(L.img_rows) * cfg_.video_row_width() * 4, d_vid_in_);
    if (cfg_.audio_enabled && L.audio_rows > 0) {
        upload_vec(in.audio_rows, size_t(L.audio_rows) * cfg_.audio_latents_dim * 4, d_aud_in_);
    }
    upload_vec(in.prompt_embeds, size_t(L.text_len) * cfg_.text_dim * 2, d_txt_in_);

    H3StepOutput out;
    const auto t_emb_start = Clock::now();

    // 1. Time embedder
    {
        auto& r = get_time_embed_recipe();
        std::vector<Binding> b = {
            {"t_freq", &d_t_freq_, 0},
            w_.bind("w1", "time_embedder.proj_in.w"),
            w_.bind("b1", "time_embedder.proj_in.b"),
            w_.bind("w2", "time_embedder.proj_out.w"),
            w_.bind("b2", "time_embedder.proj_out.b"),
            {"t_emb", &d_t_emb_, 0},
        };
        r.enqueue(stream_, b);
    }

    // 2. Video & Audio patch projections + Condition proj + TokenRefiner
    {
        auto& rv = get_patch_proj_recipe(L.img_rows, false);
        std::vector<Binding> bv = {
            {"x", &d_vid_in_, 0},
            w_.bind("w", "video_patch_proj.w"),
            w_.bind("b", "video_patch_proj.b"),
            {"out", &d_vid_emb_, 0},
        };
        rv.enqueue(stream_, bv);

        if (cfg_.audio_enabled && L.audio_rows > 0) {
            auto& ra = get_patch_proj_recipe(L.audio_rows, true);
            std::vector<Binding> ba = {
                {"x", &d_aud_in_, 0},
                w_.bind("w", "audio_patch_proj.w"),
                w_.bind("b", "audio_patch_proj.b"),
                {"out", &d_aud_emb_, 0},
            };
            ra.enqueue(stream_, ba);
        }

        auto& rc = get_cond_proj_recipe(L.text_len);
        std::vector<Binding> bc = {
            {"x", &d_txt_in_, 0},
            w_.bind("w", "condition_proj.w"),
            w_.bind("ws", "condition_proj.s"),
            w_.bind("b", "condition_proj.b"),
            {"out", &d_txt_buf_[0], 0},
        };
        rc.enqueue(stream_, bc);

        int tp = 0;
        auto& rrb = get_refiner_block_recipe(L.text_len, lora_rank);
        for (uint32_t i = 0; i < cfg_.refiner_layers; ++i) {
            const std::string p = "R" + std::to_string(i) + ".";
            std::vector<Binding> br = {
                {"x", &d_txt_buf_[tp], 0},
                w_.bind("norm1", p + "norm1"),
                w_.bind("norm2", p + "norm2"),
                w_.bind("q_norm", p + "q_norm"),
                w_.bind("k_norm", p + "k_norm"),
                w_.bind("qkv_w", p + "qkv.w"),
                w_.bind("qkv_s", p + "qkv.s"),
                w_.bind("out_w", p + "out.w"),
                w_.bind("out_s", p + "out.s"),
                w_.bind("fc1_w", p + "fc1.w"),
                w_.bind("fc1_s", p + "fc1.s"),
                w_.bind("fc2_w", p + "fc2.w"),
                w_.bind("fc2_s", p + "fc2.s"),
                {"out", &d_txt_buf_[1 - tp], 0},
            };
            if (in.lora) in.lora->bind_block(br, true, i);
            rrb.enqueue(stream_, br);
            tp = 1 - tp;
        }
        auto& rrn = get_refiner_norm_recipe(L.text_len);
        std::vector<Binding> brn = {
            {"x", &d_txt_buf_[tp], 0},
            w_.bind("w", "refiner.final_norm"),
            {"out", &d_txt_buf_[1 - tp], 0},
        };
        rrn.enqueue(stream_, brn);
        const int txt_final = 1 - tp;

        // Pack into d_hidden_[0] contiguous slices: [text | cond_video | audio | target_video | zeros]
        const uint64_t row_bytes = uint64_t(cfg_.hidden_size) * 2;
        stream_.zero(d_hidden_[0], 0, uint64_t(S) * row_bytes);
        stream_.copy(d_txt_buf_[txt_final], 0, d_hidden_[0], 0, uint64_t(L.text_len) * row_bytes);
        if (L.cond_rows > 0) {
            stream_.copy(d_vid_emb_, 0, d_hidden_[0], uint64_t(cond_start) * row_bytes, uint64_t(L.cond_rows) * row_bytes);
        }
        if (cfg_.audio_enabled && L.audio_rows > 0) {
            stream_.copy(d_aud_emb_, 0, d_hidden_[0], uint64_t(audio_start) * row_bytes, uint64_t(L.audio_rows) * row_bytes);
        }
        stream_.copy(d_vid_emb_, uint64_t(L.cond_rows) * row_bytes, d_hidden_[0], uint64_t(video_start) * row_bytes,
                     uint64_t(L.video_rows) * row_bytes);

        if (debug) {
            stream_.synchronize();
            // Dump t_emb [uniq.size(), 2688]
            HostBuffer stg_te(dev_, uniq.size() * cfg_.time_embed_dim * 4);
            stream_.download(d_t_emb_, 0, stg_te, 0, stg_te.size());
            stream_.synchronize();
            debug->t_emb.assign(stg_te.as<float>(), stg_te.as<float>() + uniq.size() * cfg_.time_embed_dim);

            // Dump token_refiner_out [text_len, 5376]
            HostBuffer stg_tr(dev_, uint64_t(L.text_len) * row_bytes);
            stream_.download(d_txt_buf_[txt_final], 0, stg_tr, 0, stg_tr.size());
            stream_.synchronize();
            debug->token_refiner_out.resize(size_t(L.text_len) * cfg_.hidden_size);
            const auto* p16 = stg_tr.as<uint16_t>();
            for (size_t i = 0; i < debug->token_refiner_out.size(); ++i) debug->token_refiner_out[i] = from_bf16(p16[i]);
        }
    }
    stream_.synchronize();
    out.embed_ms = ms_since(t_emb_start);

    // 3. 50 DiT blocks
    const auto t_blk_start = Clock::now();
    uint32_t mf_min = 4096;
    if (const char* e = std::getenv("G2E_H3_MAXFREE_MIN")) mf_min = uint32_t(std::stoul(e));
    auto resolve_h3_desc_dir = []() -> std::string {
        if (const char* e = std::getenv("G2E_H3_MME_DESC_DIR"); e && e[0] != '\0') return e;
        const std::string cands[] = {
            "/opt/gaudiinfer/tools/mme_descs_h3",
            std::string(G2E_SOURCE_DIR) + "/tools/mme_descs_h3",
            "tools/mme_descs_h3",
        };
        for (const auto& c : cands) {
            if (std::filesystem::exists(c + "/qk_f8_1536_4096")) return c;
        }
        return {};
    };
    auto resolve_tpc_dir = []() -> std::string {
        if (const char* e = std::getenv("G2E_TPC_OBJ_DIR_OVERRIDE"); e && e[0] != '\0') return e;
        if (const char* e = std::getenv("G2E_TPC_OBJ_DIR"); e && e[0] != '\0') return e;
        if (std::filesystem::exists("/opt/gaudiinfer/lib/h3_direct_attn_step_fp8_gaudi2.o"))
            return "/opt/gaudiinfer/lib";
        return std::string(G2E_TPC_OBJ_DIR);
    };
    const char* da_env = std::getenv("G2E_H3_DIRECT_ATTN");
    const char* ff8_env = std::getenv("G2E_H3_FAST_FP8_ATTN");
    const char* mf_env = std::getenv("G2E_H3_MAXFREE");
    const bool explicit_other = (ff8_env && ff8_env[0] != '0') || (mf_env && mf_env[0] != '0');
    const char* scal_env = std::getenv("SCAL_CONFIG_FILE_PATH");
    const bool has_scal_direct = (scal_env && scal_env[0] != '\0');
    const std::string h3_desc_dir = resolve_h3_desc_dir();
    const bool use_direct_attn = (da_env ? (da_env[0] != '0') : (!explicit_other && has_scal_direct && !h3_desc_dir.empty()));
    const bool use_fast_fp8 = !use_direct_attn && (ff8_env && ff8_env[0] != '0');
    const bool use_maxfree = !use_direct_attn && !use_fast_fp8 && (mf_env && mf_env[0] != '0') && U >= mf_min;
    static const std::set<uint32_t> kMaxFreeSkip = {45, 48, 49};

    uint32_t h_sub = 0, q_chunk = 0, k_pad = 0, pad_k = 0, q_blocks = 0, h_groups = 0;
    DirectH3Attn* direct_attn = nullptr;
    DirectH3Attn* direct_attn_sparse = nullptr;
    DirectH3AttnArgs direct_args{};
    uint32_t sparse_keep_full = 0;
    uint32_t sparse_dense_edges = 2u;
    bool sparse_dense_step0 = false;
    uint32_t sparse_prefix_tiles = 1u;
    bool direct_use_pv256 = false;
    if (use_direct_attn) {
        h_sub = cfg_.num_heads;
        q_chunk = DirectH3Attn::aligned_u_pad(U);
        k_pad = DirectH3Attn::aligned_k_pad(U);
        pad_k = k_pad - U;
        q_blocks = 1u;
        h_groups = 1u;
        if (!direct_rt_) {
            direct_rt_ = std::make_unique<DirectRuntime>(dev_);
            const std::string desc_dir = h3_desc_dir.empty()
                ? (std::string(G2E_SOURCE_DIR) + "/tools/mme_descs_h3")
                : h3_desc_dir;
            const std::string tpc_dir = resolve_tpc_dir();
            direct_assets_ = std::make_unique<DirectH3AttnAssets>(
                DirectH3AttnAssets::load(*direct_rt_, desc_dir, tpc_dir));
        }
        direct_use_pv256 = direct_assets_->has_pv256 && (std::getenv("G2E_H3_DIRECT_LEGACY_PSUM") == nullptr);
        ensure_mf_buffers(q_chunk, q_chunk, /*fp8_qkv=*/true, /*v_width=*/direct_use_pv256 ? 256u : 128u);
        direct_args = {d_mf_q_.address(), d_mf_k_.address(), d_mf_v_.address(),
                       d_mf_o_.address(), d_mf_sc_.address(), d_mf_dv_.address()};
        auto get_or_build_direct = [&](uint32_t keep_full) -> DirectH3Attn* {
            const auto key = std::make_pair(U, keep_full | (direct_use_pv256 ? 0x80000000u : 0u));
            auto it = direct_attn_cache_.find(key);
            if (it == direct_attn_cache_.end()) {
                it = direct_attn_cache_.emplace(
                    key, std::make_unique<DirectH3Attn>(dev_, *direct_rt_, *direct_assets_,
                                                        cfg_.num_heads, U, direct_args, 0, 0xa, S, q_chunk,
                                                        keep_full, direct_use_pv256)).first;
            }
            return it->second.get();
        };
        direct_attn = get_or_build_direct(0u);

        // Route A (U <= 45,000, e.g. <= 6s 768p): 100% Dense Lossless.
        // Route B (U > 45,000, e.g. 7-15s 768p): Dynamic Block-Sparse DirectH3Attn on middle layers!
        uint32_t sparse_min_u = 45000u;
        if (const char* e = std::getenv("G2E_H3_SPARSE_MIN_U")) sparse_min_u = uint32_t(std::stoul(e));
        const char* sp_en = std::getenv("G2E_H3_SPARSE");
        const bool sparse_allowed = (!sp_en || sp_en[0] != '0') && (sparse_min_u > 0) && (U > sparse_min_u);
        if (const char* e = std::getenv("G2E_H3_SPARSE_DENSE_EDGES")) sparse_dense_edges = uint32_t(std::stoul(e));
        if (const char* e = std::getenv("G2E_H3_SPARSE_DENSE_STEP0")) sparse_dense_step0 = (e[0] != '0');
        const uint32_t n_full_tiles = U / 4096u;
        if (sparse_allowed && n_full_tiles >= 10u && !(sparse_dense_step0 && in.step_idx == 0)) {
            double keep_frac = 0.30;
            if (const char* e = std::getenv("G2E_H3_SPARSE_KEEP_FRAC")) {
                double v = std::atof(e);
                if (v > 0.05 && v < 1.0) keep_frac = v;
            }
            sparse_prefix_tiles = std::max<uint32_t>(1u, (video_start + 4095u) / 4096u);
            const uint32_t min_keep = std::min(n_full_tiles, sparse_prefix_tiles + 5u);
            uint32_t kf = uint32_t(std::lround(double(n_full_tiles) * keep_frac));
            if (const char* e = std::getenv("G2E_H3_SPARSE_KEEP_TILES")) kf = uint32_t(std::stoul(e));
            kf = std::clamp(kf, min_keep, n_full_tiles);
            if (kf < n_full_tiles) {
                sparse_keep_full = kf;
                direct_attn_sparse = get_or_build_direct(sparse_keep_full);
                const uint64_t sc_bytes = uint64_t(cfg_.num_heads) * direct_attn_sparse->q_blocks_per_head() * n_full_tiles * 4ull;
                if (sc_bytes > alloc_blk_scores_bytes_) {
                    alloc_blk_scores_bytes_ = sc_bytes;
                    d_blk_scores_ = DeviceBuffer(dev_, std::max<uint64_t>(256ull, sc_bytes));
                    h_blk_scores_ = std::make_unique<HostBuffer>(dev_, std::max<uint64_t>(256ull, sc_bytes));
                }
            }
        }
        if (S > U) {
            // Pre-zero [U..S) tail of flat [S, AD] output ONCE (DirectH3Attn only writes rows < U).
            const uint64_t ad_bytes = uint64_t(cfg_.num_heads) * cfg_.head_dim * 2ull;
            stream_.zero(d_mf_o_, uint64_t(U) * ad_bytes, uint64_t(S - U) * ad_bytes);
        }
    } else if (use_fast_fp8 || use_maxfree) {
        std::tie(h_sub, q_chunk) = choose_mf_tiling(cfg_.num_heads, U);
        const uint32_t k_align = use_fast_fp8 ? 256u : 128u;
        k_pad = ((U + k_align - 1u) / k_align) * k_align;
        pad_k = k_pad - U;
        q_blocks = (U + q_chunk - 1u) / q_chunk;
        h_groups = cfg_.num_heads / h_sub;
        ensure_mf_buffers(q_blocks * q_chunk, k_pad, /*fp8_qkv=*/use_fast_fp8);
    }

    double direct_attn_ms_total = 0.0, direct_pre_ms = 0.0, direct_post_ms = 0.0;
    const bool direct_prof = (std::getenv("G2E_H3_DIRECT_PROF") != nullptr);
    const char* ml_env = std::getenv("G2E_H3_MERGE_LORA");
    const bool merge_lora = (in.lora != nullptr) && (ml_env && ml_env[0] != '0') &&
                            (std::getenv("G2E_H3_NO_MERGE_LORA") == nullptr);
    ensure_merged_lora(merge_lora ? in.lora : nullptr);
    const uint32_t blk_lora_rank = merge_lora ? 0u : lora_rank;
    const H3LoraAdapter* blk_lora = merge_lora ? nullptr : in.lora;
    const uint32_t adaln_rank = in.lora ? in.lora->adaln_rank() : 0u;
    const uint32_t final_adaln_rank = in.lora ? in.lora->final_adaln_rank() : 0u;
    std::vector<uint16_t> sparse_active_tiles;
    if (direct_attn_sparse) {
        sparse_active_tiles.resize(size_t(cfg_.num_heads) * direct_attn_sparse->q_blocks_per_head() * sparse_keep_full);
    }
    auto select_sparse_tiles = [&](const float* scores, uint32_t qb_per_h, uint32_t n_full, uint32_t keep_full) {
        const uint32_t total_qb = cfg_.num_heads * qb_per_h;
        for (uint32_t gq = 0; gq < total_qb; ++gq) {
            const uint32_t qb = gq % qb_per_h;
            const float* sc = scores + size_t(gq) * n_full;
            uint16_t* dst = sparse_active_tiles.data() + size_t(gq) * keep_full;
            uint64_t picked_mask = 0;
            uint32_t cnt = 0;
            auto add_tile = [&](uint32_t t) {
                if (t >= n_full || cnt >= keep_full) return;
                const uint64_t bit = 1ull << (t & 63u);
                if (t < 64u) {
                    if (picked_mask & bit) return;
                    picked_mask |= bit;
                } else {
                    for (uint32_t i = 0; i < cnt; ++i) if (dst[i] == t) return;
                }
                dst[cnt++] = uint16_t(t);
            };
            // 1. Always attend to prefix tiles (text + cond keyframes + full stereo audio!)
            for (uint32_t t = 0; t < sparse_prefix_tiles; ++t) add_tile(t);
            // 2. Always attend to self + +-1 local temporal window (~4-8 neighboring latent frames)
            const uint32_t row0 = qb * 1536u;
            const uint32_t row1 = row0 + 1535u;
            const uint32_t t_lo = std::min(n_full - 1u, row0 / 4096u);
            const uint32_t t_hi = std::min(n_full - 1u, row1 / 4096u);
            add_tile(t_lo);
            add_tile(t_hi);
            if (t_lo > 0) add_tile(t_lo - 1u);
            if (t_hi + 1u < n_full) add_tile(t_hi + 1u);
            // 3. Fill remaining slots with dynamic highest-scoring global K-tiles
            while (cnt < keep_full) {
                float best_v = -1.0e30f;
                uint32_t best_t = 0;
                bool found = false;
                for (uint32_t t = 0; t < n_full; ++t) {
                    if (t < 64u) {
                        if (picked_mask & (1ull << t)) continue;
                    } else {
                        bool dup = false;
                        for (uint32_t i = 0; i < cnt; ++i) if (dst[i] == t) { dup = true; break; }
                        if (dup) continue;
                    }
                    if (!found || sc[t] > best_v) { best_v = sc[t]; best_t = t; found = true; }
                }
                if (!found) break;
                add_tile(best_t);
            }
            std::sort(dst, dst + cnt);
        }
    };

    int hp = 0;
    for (uint32_t bi = 0; bi < cfg_.num_layers; ++bi) {
        const bool mf = use_maxfree && (kMaxFreeSkip.count(bi) == 0);
        const std::string p = "B" + std::to_string(bi) + ".";
        auto bind_bw = [&](const std::string& tname, const std::string& field) -> Binding {
            return w_.bind(tname, p + field);
        };
        if (use_direct_attn) {
            const bool layer_sparse = (direct_attn_sparse != nullptr) &&
                                      (bi >= sparse_dense_edges) &&
                                      (bi + sparse_dense_edges < cfg_.num_layers);
            auto& r_pre = get_dit_pre_fp8_recipe(S, U, blk_lora_rank, h_sub, q_chunk,
                                                 /*kv_use_upad=*/true, /*with_blk_scores=*/layer_sparse,
                                                 /*v_aug_256=*/direct_use_pv256, adaln_rank);
            std::vector<Binding> bpre = {
                {"x", &d_hidden_[hp], 0},
                {"t_emb", &d_t_emb_, 0},
                {"comb_idx", &d_comb_idx_, 0},
                {"rope_cos", &d_rope_cos_, 0},
                {"rope_sin", &d_rope_sin_, 0},
                w_.bind("norm1", p + "norm1"),
                w_.bind("q_norm", p + "q_norm"),
                w_.bind("k_norm", p + "k_norm"),
                bind_bw("qkv_w", "qkv.w"),
                bind_bw("qkv_s", "qkv.s"),
                w_.bind("adaln_w", p + "adaln.w"),
                w_.bind("adaln_b", p + "adaln.b"),
                {"q_chunks", &d_mf_q_, 0},
                {"k_pad", &d_mf_k_, 0},
                {"v_pad", &d_mf_v_, 0},
                {"score_scale", &d_mf_sc_, 0},
                {"d_scale_v", &d_mf_dv_, 0},
                {"ad_post", &d_mf_ad_, 0},
            };
            if (layer_sparse) bpre.push_back({"blk_scores", &d_blk_scores_, 0});
            if (!cfg_.adaln_bf16) bpre.push_back(w_.bind("adaln_s", p + "adaln.s"));
            if (blk_lora) {
                std::vector<Binding> all_l;
                blk_lora->bind_block(all_l, false, bi);
                for (const auto& lb : all_l)
                    if (lb.name == "lora_qkv_a" || lb.name == "lora_qkv_b") bpre.push_back(lb);
            }
            if (in.lora && adaln_rank > 0) in.lora->bind_adaln(bpre, bi);
            const auto t_pre0 = Clock::now();
            r_pre.enqueue(stream_, bpre);
            if (layer_sparse) {
                const uint64_t sc_bytes = uint64_t(cfg_.num_heads) * direct_attn_sparse->q_blocks_per_head() *
                                          direct_attn_sparse->n_full_tiles() * 4ull;
                stream_.download(d_blk_scores_, 0, *h_blk_scores_, 0, sc_bytes);
            }
            stream_.synchronize();
            if (direct_prof) direct_pre_ms += ms_since(t_pre0);

            if (layer_sparse) {
                select_sparse_tiles(h_blk_scores_->as<float>(), direct_attn_sparse->q_blocks_per_head(),
                                    direct_attn_sparse->n_full_tiles(), sparse_keep_full);
                direct_attn_ms_total += direct_attn_sparse->run(direct_args, sparse_active_tiles.data());
            } else {
                direct_attn_ms_total += direct_attn->run(direct_args);
            }

            auto& r_post = get_dit_post_recipe(S, U, blk_lora_rank, h_sub, /*q_chunk=*/0);
            std::vector<Binding> bpost = {
                {"x", &d_hidden_[hp], 0},
                {"ctx_chunks", &d_mf_o_, 0},
                {"ad_post", &d_mf_ad_, 0},
                {"comb_idx", &d_comb_idx_, 0},
                w_.bind("norm2", p + "norm2"),
                bind_bw("out_w", "out.w"),
                bind_bw("out_s", "out.s"),
                bind_bw("fc1_w", "fc1.w"),
                bind_bw("fc1_s", "fc1.s"),
                bind_bw("fc2_w", "fc2.w"),
                bind_bw("fc2_s", "fc2.s"),
                {"out", &d_hidden_[1 - hp], 0},
            };
            if (blk_lora) {
                std::vector<Binding> all_l;
                blk_lora->bind_block(all_l, false, bi);
                for (const auto& lb : all_l)
                    if (lb.name != "lora_qkv_a" && lb.name != "lora_qkv_b") bpost.push_back(lb);
            }
            const auto t_post0 = Clock::now();
            r_post.enqueue(stream_, bpost);
            if (direct_prof) {
                stream_.synchronize();
                direct_post_ms += ms_since(t_post0);
            }
        } else if (use_fast_fp8) {
            // Ultra-fast 256-wide direct BF16->FP8 attention (works on ALL 50 layers including 45, 48, 49!)
            auto& r_pre = get_dit_pre_fp8_recipe(S, U, blk_lora_rank, h_sub, q_chunk, false, false, false, adaln_rank);
            std::vector<Binding> bpre = {
                {"x", &d_hidden_[hp], 0},
                {"t_emb", &d_t_emb_, 0},
                {"comb_idx", &d_comb_idx_, 0},
                {"rope_cos", &d_rope_cos_, 0},
                {"rope_sin", &d_rope_sin_, 0},
                w_.bind("norm1", p + "norm1"),
                w_.bind("q_norm", p + "q_norm"),
                w_.bind("k_norm", p + "k_norm"),
                bind_bw("qkv_w", "qkv.w"),
                bind_bw("qkv_s", "qkv.s"),
                w_.bind("adaln_w", p + "adaln.w"),
                w_.bind("adaln_b", p + "adaln.b"),
                {"q_chunks", &d_mf_q_, 0},
                {"k_pad", &d_mf_k_, 0},
                {"v_pad", &d_mf_v_, 0},
                {"score_scale", &d_mf_sc_, 0},
                {"d_scale_v", &d_mf_dv_, 0},
                {"ad_post", &d_mf_ad_, 0},
            };
            if (!cfg_.adaln_bf16) bpre.push_back(w_.bind("adaln_s", p + "adaln.s"));
            if (blk_lora) {
                std::vector<Binding> all_l;
                blk_lora->bind_block(all_l, false, bi);
                for (const auto& lb : all_l)
                    if (lb.name == "lora_qkv_a" || lb.name == "lora_qkv_b") bpre.push_back(lb);
            }
            if (in.lora && adaln_rank > 0) in.lora->bind_adaln(bpre, bi);
            r_pre.enqueue(stream_, bpre);

            auto& r_chk = get_fast_fp8_chunk_recipe(h_sub, q_chunk, k_pad, U);
            const uint64_t HD = cfg_.head_dim;
            const uint64_t kv_stride = uint64_t(h_sub) * k_pad * HD;      // F8 = 1 byte
            const uint64_t q_stride = uint64_t(h_sub) * q_chunk * HD;     // F8 = 1 byte
            const uint64_t o_stride = uint64_t(h_sub) * q_chunk * HD * 2; // BF16 = 2 bytes
            for (uint32_t hg = 0; hg < h_groups; ++hg) {
                const uint64_t kv_off = uint64_t(hg) * kv_stride;
                for (uint32_t ci = 0; ci < q_blocks; ++ci) {
                    const uint64_t idx = uint64_t(hg) * q_blocks + ci;
                    std::vector<Binding> bc = {
                        {"q_c", &d_mf_q_, idx * q_stride},
                        {"k_g", &d_mf_k_, kv_off},
                        {"v_g", &d_mf_v_, kv_off},
                        {"score_scale", &d_mf_sc_, 0},
                        {"d_scale_v", &d_mf_dv_, 0},
                        {"out", &d_mf_o_, idx * o_stride},
                    };
                    r_chk.enqueue(stream_, bc);
                }
            }

            auto& r_post = get_dit_post_recipe(S, U, blk_lora_rank, h_sub, q_chunk);
            std::vector<Binding> bpost = {
                {"x", &d_hidden_[hp], 0},
                {"ctx_chunks", &d_mf_o_, 0},
                {"ad_post", &d_mf_ad_, 0},
                {"comb_idx", &d_comb_idx_, 0},
                w_.bind("norm2", p + "norm2"),
                bind_bw("out_w", "out.w"),
                bind_bw("out_s", "out.s"),
                bind_bw("fc1_w", "fc1.w"),
                bind_bw("fc1_s", "fc1.s"),
                bind_bw("fc2_w", "fc2.w"),
                bind_bw("fc2_s", "fc2.s"),
                {"out", &d_hidden_[1 - hp], 0},
            };
            if (blk_lora) {
                std::vector<Binding> all_l;
                blk_lora->bind_block(all_l, false, bi);
                for (const auto& lb : all_l)
                    if (lb.name != "lora_qkv_a" && lb.name != "lora_qkv_b") bpost.push_back(lb);
            }
            r_post.enqueue(stream_, bpost);
        } else if (mf) {
            // Split-chunk max-free path: r_dit_pre -> (H_groups x Q_blocks) x r_maxfree_chunk -> r_dit_post
            auto& r_pre = get_dit_pre_recipe(S, U, blk_lora_rank, h_sub, q_chunk, adaln_rank);
            std::vector<Binding> bpre = {
                {"x", &d_hidden_[hp], 0},
                {"t_emb", &d_t_emb_, 0},
                {"comb_idx", &d_comb_idx_, 0},
                {"rope_cos", &d_rope_cos_, 0},
                {"rope_sin", &d_rope_sin_, 0},
                w_.bind("norm1", p + "norm1"),
                w_.bind("q_norm", p + "q_norm"),
                w_.bind("k_norm", p + "k_norm"),
                w_.bind("qkv_w", p + "qkv.w"),
                w_.bind("qkv_s", p + "qkv.s"),
                w_.bind("adaln_w", p + "adaln.w"),
                w_.bind("adaln_b", p + "adaln.b"),
                {"q_chunks", &d_mf_q_, 0},
                {"b_chunks", &d_mf_b_, 0},
                {"k_pad", &d_mf_k_, 0},
                {"v_pad", &d_mf_v_, 0},
                {"ad_post", &d_mf_ad_, 0},
            };
            if (!cfg_.adaln_bf16) bpre.push_back(w_.bind("adaln_s", p + "adaln.s"));
            if (blk_lora) {
                std::vector<Binding> all_l;
                blk_lora->bind_block(all_l, false, bi);
                for (const auto& lb : all_l)
                    if (lb.name == "lora_qkv_a" || lb.name == "lora_qkv_b") bpre.push_back(lb);
            }
            if (in.lora && adaln_rank > 0) in.lora->bind_adaln(bpre, bi);
            r_pre.enqueue(stream_, bpre);

            auto& r_chk = get_maxfree_chunk_recipe(h_sub, q_chunk, k_pad, pad_k);
            const uint64_t HD = cfg_.head_dim;
            const uint64_t kv_stride = uint64_t(h_sub) * k_pad * HD * 2;
            const uint64_t qo_stride = uint64_t(h_sub) * q_chunk * HD * 2;
            const uint64_t b_stride = uint64_t(h_sub) * q_chunk * 2;
            for (uint32_t hg = 0; hg < h_groups; ++hg) {
                const uint64_t kv_off = uint64_t(hg) * kv_stride;
                for (uint32_t ci = 0; ci < q_blocks; ++ci) {
                    const uint64_t idx = uint64_t(hg) * q_blocks + ci;
                    std::vector<Binding> bc = {
                        {"q_c", &d_mf_q_, idx * qo_stride},
                        {"k_g", &d_mf_k_, kv_off},
                        {"v_g", &d_mf_v_, kv_off},
                        {"b_c", &d_mf_b_, idx * b_stride},
                        {"out", &d_mf_o_, idx * qo_stride},
                    };
                    r_chk.enqueue(stream_, bc);
                }
            }

            auto& r_post = get_dit_post_recipe(S, U, blk_lora_rank, h_sub, q_chunk);
            std::vector<Binding> bpost = {
                {"x", &d_hidden_[hp], 0},
                {"ctx_chunks", &d_mf_o_, 0},
                {"ad_post", &d_mf_ad_, 0},
                {"comb_idx", &d_comb_idx_, 0},
                w_.bind("norm2", p + "norm2"),
                w_.bind("out_w", p + "out.w"),
                w_.bind("out_s", p + "out.s"),
                w_.bind("fc1_w", p + "fc1.w"),
                w_.bind("fc1_s", p + "fc1.s"),
                w_.bind("fc2_w", p + "fc2.w"),
                w_.bind("fc2_s", p + "fc2.s"),
                {"out", &d_hidden_[1 - hp], 0},
            };
            if (blk_lora) {
                std::vector<Binding> all_l;
                blk_lora->bind_block(all_l, false, bi);
                for (const auto& lb : all_l)
                    if (lb.name != "lora_qkv_a" && lb.name != "lora_qkv_b") bpost.push_back(lb);
            }
            r_post.enqueue(stream_, bpost);
        } else {
            auto& r = get_dit_block_recipe(S, U, blk_lora_rank, adaln_rank);
            std::vector<Binding> b = {
                {"x", &d_hidden_[hp], 0},
                {"t_emb", &d_t_emb_, 0},
                {"comb_idx", &d_comb_idx_, 0},
                {"rope_cos", &d_rope_cos_, 0},
                {"rope_sin", &d_rope_sin_, 0},
                w_.bind("norm1", p + "norm1"),
                w_.bind("norm2", p + "norm2"),
                w_.bind("q_norm", p + "q_norm"),
                w_.bind("k_norm", p + "k_norm"),
                w_.bind("qkv_w", p + "qkv.w"),
                w_.bind("qkv_s", p + "qkv.s"),
                w_.bind("out_w", p + "out.w"),
                w_.bind("out_s", p + "out.s"),
                w_.bind("fc1_w", p + "fc1.w"),
                w_.bind("fc1_s", p + "fc1.s"),
                w_.bind("fc2_w", p + "fc2.w"),
                w_.bind("fc2_s", p + "fc2.s"),
                w_.bind("adaln_w", p + "adaln.w"),
                w_.bind("adaln_b", p + "adaln.b"),
                {"out", &d_hidden_[1 - hp], 0},
            };
            if (!cfg_.adaln_bf16) b.push_back(w_.bind("adaln_s", p + "adaln.s"));
            if (blk_lora) blk_lora->bind_block(b, false, bi);
            if (in.lora && adaln_rank > 0) in.lora->bind_adaln(b, bi);
            r.enqueue(stream_, b);
        }
        hp = 1 - hp;

        if (debug && debug->dump_all_layers) {
            stream_.synchronize();
            const uint32_t rows = std::min<uint32_t>(S, debug->layer_sample_rows);
            const size_t n = size_t(rows) * cfg_.hidden_size;
            HostBuffer stg(dev_, n * 2);
            stream_.download(d_hidden_[hp], 0, stg, 0, n * 2);
            stream_.synchronize();
            if (debug->layer_samples.size() <= bi) debug->layer_samples.resize(bi + 1);
            auto& dst = debug->layer_samples[bi];
            dst.resize(n);
            const auto* p16 = stg.as<uint16_t>();
            for (size_t k = 0; k < n; ++k) dst[k] = from_bf16(p16[k]);
        } else if (debug && (bi == 0 || bi == 1)) {
            stream_.synchronize();
            const size_t n = size_t(S) * cfg_.hidden_size;
            HostBuffer stg(dev_, n * 2);
            stream_.download(d_hidden_[hp], 0, stg, 0, n * 2);
            stream_.synchronize();
            auto& dst = (bi == 0) ? debug->b0_out : debug->b1_out;
            dst.resize(n);
            const auto* p16 = stg.as<uint16_t>();
            for (size_t k = 0; k < n; ++k) dst[k] = from_bf16(p16[k]);
        }
    }
    stream_.synchronize();
    out.blocks_ms = ms_since(t_blk_start);
    if (use_direct_attn) {
        if (direct_prof) {
            std::printf("[H3DiT] DirectH3Attn sum=%.1f ms (%.2f ms/L), pre=%.1f ms (%.2f ms/L), post=%.1f ms (%.2f ms/L), non-attn=%.1f ms\n",
                        direct_attn_ms_total, direct_attn_ms_total / double(cfg_.num_layers),
                        direct_pre_ms, direct_pre_ms / double(cfg_.num_layers),
                        direct_post_ms, direct_post_ms / double(cfg_.num_layers),
                        out.blocks_ms - direct_attn_ms_total);
        } else {
            std::printf("[H3DiT] DirectH3Attn sum=%.1f ms (%.2f ms/layer), non-attn=%.1f ms\n",
                        direct_attn_ms_total, direct_attn_ms_total / double(cfg_.num_layers),
                        out.blocks_ms - direct_attn_ms_total);
        }
    }

    // 4. Final layer
    const auto t_fin_start = Clock::now();
    {
        auto& rf = get_final_recipe(S, final_adaln_rank);
        const std::string vid_w_key = (in.pdd_head_idx >= 0)
            ? ("pdd.video_out." + std::to_string(in.pdd_head_idx) + ".w")
            : "final.video_out.w";
        const std::string vid_b_key = (in.pdd_head_idx >= 0)
            ? ("pdd.video_out." + std::to_string(in.pdd_head_idx) + ".b")
            : "final.video_out.b";
        std::vector<Binding> b = {
            {"x", &d_hidden_[hp], 0},
            {"t_emb", &d_t_emb_, 0},
            {"inv_idx", &d_inv_idx_, 0},
            w_.bind("norm_w", "final.norm"),
            w_.bind("adaln_w", "final.adaln.w"),
            w_.bind("adaln_b", "final.adaln.b"),
            w_.bind("vid_w", vid_w_key),
            w_.bind("vid_b", vid_b_key),
            {"video_out", &d_vid_out_, 0},
        };
        if (cfg_.audio_enabled) {
            b.push_back(w_.bind("aud_w", "final.audio_out.w"));
            b.push_back(w_.bind("aud_b", "final.audio_out.b"));
            b.push_back({"audio_out", &d_aud_out_, 0});
        }
        if (!cfg_.adaln_bf16) b.push_back(w_.bind("adaln_s", "final.adaln.s"));
        if (in.lora && final_adaln_rank > 0) in.lora->bind_final_adaln(b);
        rf.enqueue(stream_, b);
    }

    // Download video (& optional audio) logits and select img_pos / audio_pos + apply update_mask
    const uint64_t vw = cfg_.video_row_width(), aw = cfg_.audio_latents_dim;
    HostBuffer stg_v(dev_, uint64_t(S) * vw * 4);
    stream_.download(d_vid_out_, 0, stg_v, 0, stg_v.size());
    std::unique_ptr<HostBuffer> stg_a;
    if (cfg_.audio_enabled && L.audio_rows > 0) {
        stg_a = std::make_unique<HostBuffer>(dev_, uint64_t(S) * aw * 4);
        stream_.download(d_aud_out_, 0, *stg_a, 0, stg_a->size());
    }
    stream_.synchronize();
    out.final_ms = ms_since(t_fin_start);

    const float* vp = stg_v.as<float>();
    out.video_logits.assign(size_t(L.img_rows) * vw, 0.0f);
    std::memcpy(out.video_logits.data() + size_t(L.cond_rows) * vw,
                vp + size_t(video_start) * vw,
                size_t(L.video_rows) * vw * sizeof(float));

    if (cfg_.audio_enabled && L.audio_rows > 0) {
        const float* ap = stg_a->as<float>();
        out.audio_logits.resize(size_t(L.audio_rows) * aw);
        std::memcpy(out.audio_logits.data(),
                    ap + size_t(audio_start) * aw,
                    size_t(L.audio_rows) * aw * sizeof(float));
    }

    out.total_ms = ms_since(t_all);
    return out;
}

// ------------------------------------------------------------------------------------------------ Scheduler + Denoise

std::vector<float> H3DiT::time_shift_sigmas(uint32_t num_steps, float shift_scale) {
    std::vector<float> sigmas;
    sigmas.reserve(num_steps + 1);
    for (uint32_t i = 0; i < num_steps; ++i) {
        float base = (num_steps > 1) ? (1.0f - float(i) / float(num_steps - 1)) : 1.0f;
        float s = shift_scale * base / (1.0f + (shift_scale - 1.0f) * base);
        if (sigmas.empty() || s != sigmas.back()) sigmas.push_back(s);
    }
    if (num_steps > 1 && sigmas.back() > 0.0f) sigmas.push_back(0.0f);
    return sigmas;
}

void H3DiT::denoise(const H3PackedLayout& layout, std::vector<float>& video_rows, std::vector<float>& audio_rows,
                    const uint16_t* prompt_embeds, const std::vector<float>& sigmas_video,
                    const std::vector<float>& sigmas_audio, const H3LoraAdapter* lora,
                    const float* cond_anchor, float imgvid_cond_noise_aug) {
    const uint32_t vw = cfg_.video_row_width(), aw = cfg_.audio_latents_dim;
    if (cond_anchor && layout.cond_rows > 0) {
        std::memcpy(video_rows.data(), cond_anchor, size_t(layout.cond_rows) * vw * sizeof(float));
    }
    const size_t n_steps = sigmas_video.size() - 1;
    for (size_t step = 0; step < n_steps; ++step) {
        const float sv = sigmas_video[step], sv_next = sigmas_video[step + 1];
        const float sa = sigmas_audio[step], sa_next = sigmas_audio[step + 1];
        const float tv = 1.0f - sv, ta = 1.0f - sa;
        const float cond_t = std::max(tv, imgvid_cond_noise_aug);

        H3StepInput sin;
        sin.layout = &layout;
        sin.video_rows = video_rows.data();
        sin.audio_rows = audio_rows.data();
        sin.prompt_embeds = prompt_embeds;
        sin.t_video = tv;
        sin.t_audio = ta;
        sin.imgvid_cond_t = cond_t;
        sin.lora = lora;
        sin.step_idx = uint32_t(step);

        auto res = forward(sin);
        std::printf("[H3DiT] step %zu/%zu: total=%.1f ms (embed=%.1f, blocks=%.1f, final=%.1f)\n",
                    step + 1, n_steps, res.total_ms, res.embed_ms, res.blocks_ms, res.final_ms);

        // Rectified-flow v->x0 + Euler eta=0 step (in FP32)
        const float rv = (sv > 0.0f) ? (sv_next / sv) : 1.0f;
        const size_t v_off = size_t(layout.cond_rows) * vw;
        const size_t v_cnt = size_t(layout.video_rows) * vw;
        for (size_t i = 0; i < v_cnt; ++i) {
            const float xt = video_rows[v_off + i];
            const float x0 = xt + sv * res.video_logits[v_off + i];
            video_rows[v_off + i] = rv * xt + (1.0f - rv) * x0;
        }
        if (cond_anchor && layout.cond_rows > 0) {
            std::memcpy(video_rows.data(), cond_anchor, size_t(layout.cond_rows) * vw * sizeof(float));
        }

        const float ra = (sa > 0.0f) ? (sa_next / sa) : 1.0f;
        const size_t a_cnt = size_t(layout.audio_rows) * aw;
        for (size_t i = 0; i < a_cnt; ++i) {
            const float xt = audio_rows[i];
            const float x0 = xt + sa * res.audio_logits[i];
            audio_rows[i] = ra * xt + (1.0f - ra) * x0;
        }
    }
}

void H3DiT::denoise_image(const H3PackedLayout& layout, std::vector<float>& img_rows,
                          const uint16_t* prompt_embeds, const float* ref_anchor,
                          float ref_timestep) {
    const uint32_t vw = cfg_.video_row_width();
    if (ref_anchor && layout.cond_rows > 0) {
        std::memcpy(img_rows.data(), ref_anchor, size_t(layout.cond_rows) * vw * sizeof(float));
    }
    // Preserve exact LightX2V/Diffusers float32 round trip: timesteps = 1 - sigma[:-1], sigmas = [1 - t, 0]
    float t_steps[6], sigmas[7];
    for (int i = 0; i < 6; ++i) {
        t_steps[i] = 1.0f - kImagePddSigmas[i];
        sigmas[i] = 1.0f - t_steps[i];
    }
    sigmas[6] = 0.0f;

    const size_t v_off = size_t(layout.cond_rows) * vw;
    const size_t v_cnt = size_t(layout.video_rows) * vw;
    for (int step = 0; step < 6; ++step) {
        if (ref_anchor && layout.cond_rows > 0) {
            std::memcpy(img_rows.data(), ref_anchor, size_t(layout.cond_rows) * vw * sizeof(float));
        }
        H3StepInput sin;
        sin.layout = &layout;
        sin.video_rows = img_rows.data();
        sin.audio_rows = nullptr;
        sin.prompt_embeds = prompt_embeds;
        sin.t_video = t_steps[step];
        sin.t_audio = t_steps[step];
        sin.imgvid_cond_t = ref_timestep;
        sin.lora = nullptr;
        sin.step_idx = uint32_t(step);
        sin.pdd_head_idx = (cfg_.pdd_steps >= 6) ? step : -1;

        auto res = forward(sin);
        std::printf("[H3ImageDiT] step %d/6 (t=%.4f, U=%u): total=%.1f ms (embed=%.1f, blocks=%.1f, final=%.1f)\n",
                    step + 1, t_steps[step], layout.used_len, res.total_ms, res.embed_ms, res.blocks_ms, res.final_ms);

        // Data-ward velocity Euler update: target = target + (sigma[i] - sigma[i+1]) * v_pred
        const float delta = sigmas[step] - sigmas[step + 1];
        for (size_t i = 0; i < v_cnt; ++i) {
            img_rows[v_off + i] += delta * res.video_logits[v_off + i];
        }
    }
}

}  // namespace g2e
