// Inkling-2-Mini correctness / speed test against golden HF dumps (tools/inkling2_ref/dump_golden.py).
//
//   inkling2_test --pack P.g2w --golden DIR [--max-ctx N] [--dump] [--decode N] [--bench N]
//
// DIR holds ids.i32 (prompt), logits_last.f32 (HF logits of the last prompt row), dec_ids.i32 / dec_logits.f32 (HF
// greedy continuation and its logits) and hs.f32 ([49, T, H]: embedding + every layer application's output).
#include "g2e/inkling2.hpp"
#include "g2e/inkling2_dspark.hpp"
#include "g2e/numeric.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace g2e;

// the drafter's pack (tools/convert_dspark.py) for the DSpark modes
static const std::string& need_dspark_pack(const std::string& path) {
    if (path.empty()) throw std::runtime_error("this mode needs --dspark-pack <pack from tools/convert_dspark.py>");
    return path;
}

template <class T>
static std::vector<T> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = size_t(f.tellg());
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * sizeof(T)));
    return v;
}

struct Cmp { double maxdiff, kl; bool top1; int ref_top, top; };
static Cmp compare(const float* ref, const std::vector<float>& got) {
    const size_t V = got.size();
    double mr = -1e30, mg = -1e30;
    for (size_t i = 0; i < V; ++i) { mr = std::max<double>(mr, ref[i]); mg = std::max<double>(mg, got[i]); }
    double sr = 0, sg = 0, md = 0;
    for (size_t i = 0; i < V; ++i) { sr += std::exp(ref[i] - mr); sg += std::exp(got[i] - mg); md = std::max(md, std::fabs(double(ref[i]) - got[i])); }
    double kl = 0;
    for (size_t i = 0; i < V; ++i) {
        const double lr = ref[i] - mr - std::log(sr), lg = got[i] - mg - std::log(sg);
        kl += std::exp(lr) * (lr - lg);
    }
    const int rt = int(std::max_element(ref, ref + V) - ref);
    const int gt = int(std::max_element(got.begin(), got.end()) - got.begin());
    return {md, kl, rt == gt, rt, gt};
}

int main(int argc, char** argv) {
    std::string pack_path, golden;
    uint32_t max_ctx = 8192, n_decode = 16, bench = 0, pdecode = 0, tbench = 0;
    std::vector<uint32_t> precompile, precompile_vh, pbenches;
    uint32_t split = 0;
    bool direct_check = false, verify_check = false;
    int direct_prof = -1;
    uint32_t long_check = 0, dspark_dump = 0, dspark_step = 4, spec_test = 0, spec_sampled_test = 0;
    std::string dspark_file, dspark_test, dspark_pack;
    bool dump = false, ctx1m_test = false, batch_check = false, bf16_cache_check = false, decode_prefill = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--pack") pack_path = next();
        else if (a == "--golden") golden = next();
        else if (a == "--max-ctx") max_ctx = uint32_t(std::stoul(next()));
        else if (a == "--decode") n_decode = uint32_t(std::stoul(next()));
        else if (a == "--bench") bench = uint32_t(std::stoul(next()));
        else if (a == "--prefill-bench") pbenches.push_back(uint32_t(std::stoul(next())));
        else if (a == "--direct-check") direct_check = true;
        else if (a == "--verify-check") verify_check = true;  // a 4-row verification: Synapse vs Direct, per row and application
        else if (a == "--dspark-dump") { dspark_dump = uint32_t(std::stoul(next())); dspark_file = next(); }  // N tokens, file
        else if (a == "--dspark-test") dspark_test = next();      // a --dspark-dump file: device drafter acceptance
        else if (a == "--dspark-pack") dspark_pack = next();
        else if (a == "--spec-test") spec_test = uint32_t(std::stoul(next()));  // greedy speculative decoding of N tokens
        else if (a == "--spec-sampled-test") spec_sampled_test = uint32_t(std::stoul(next()));  // lossless sampled spec decode of N tokens
        else if (a == "--dspark-step") dspark_step = uint32_t(std::stoul(next()));
        else if (a == "--long-check") long_check = uint32_t(std::stoul(next()));  // P-token prefill, then 32 decode steps Synapse vs Direct
        else if (a == "--direct-prof") direct_prof = int(std::stoi(next()));  // stage timing of one Direct application  // one decode token: Synapse vs Direct, per application
        else if (a == "--split") split = uint32_t(std::stoul(next()));  // prefill the first N tokens on their own
        else if (a == "--precompile") precompile.push_back(uint32_t(std::stoul(next())));  // full-layer recipes, all far buckets
        else if (a == "--precompile-vh") precompile_vh.push_back(uint32_t(std::stoul(next())));  // verification-mode recipes (padded tails)
        else if (a == "--tbench") tbench = uint32_t(std::stoul(next()));  // forward of T tokens (verify-sized)
        else if (a == "--prefill-decode") pdecode = uint32_t(std::stoul(next()));  // decode bench after the prefill bench
        else if (a == "--ctx-1m-test") ctx1m_test = true;  // test dynamic growth + real 1M prefill & decode
        else if (a == "--batch-check") batch_check = true; // test B in {2..8} batched decode vs B=1 + bench B in {1,2,3,4,8}
        else if (a == "--bf16-cache-check") bf16_cache_check = true;  // verify the v6 BF16 cache contents against a host decode
        else if (a == "--decode-prefill") decode_prefill = true;      // feed the prompt token by token through decode (no prefill forward)
        else if (a == "--dump") dump = true;
    }
    // filling the recipe cache or checking the v6 BF16 cache needs no golden reference
    const bool precompile_only = golden.empty() && (!precompile.empty() || !precompile_vh.empty());
    if (pack_path.empty() || (golden.empty() && !precompile_only && !bf16_cache_check)) {
        std::fprintf(stderr, "usage: %s --pack P.g2w --golden DIR [--max-ctx N] [--dump] [--decode N] [--bench N]\n"
                             "       %s --pack P.g2w [--max-ctx N] --precompile T ... [--precompile-vh T ...]\n"
                             "       %s --pack P.g2w --bf16-cache-check\n", argv[0], argv[0], argv[0]);
        return 2;
    }
    for (uint32_t pb : pbenches)
        if (pb + pdecode + 64 > max_ctx) max_ctx = ((pb + pdecode + 1087) / 1024) * 1024;
    try {
        Device dev;
        Stream stream(dev);
        WeightPack pack(pack_path);
        const auto up = pack.upload(dev, stream);
        std::fprintf(stderr, "weights: %.2f GiB in %.1f s\n", up.gib, up.seconds);
        Inkling2 model(dev, stream, pack, max_ctx, batch_check ? 8u : 1u, batch_check ? 8192u : UINT32_MAX);
        model.init_bf16_cache();
        if (bf16_cache_check) {
            const auto& cfg = model.config();
            if (cfg.pack_version != 6)
                throw std::runtime_error("bf16-cache-check requires a v6 native pack");
            std::vector<uint32_t> native_layers, nvfp4_layers, mxfp8_layers;
            std::vector<uint32_t> cached_layers, cached_nv, cached_mx, uncached_nv, uncached_mx;
            for (uint32_t L = 0; L < cfg.layers; ++L) {
                native_layers.push_back(L);
                const bool is_nv = cfg.is_nvfp4(L);
                (is_nv ? nvfp4_layers : mxfp8_layers).push_back(L);
                if (model.bf16_layer(L).cached) {
                    cached_layers.push_back(L);
                    (is_nv ? cached_nv : cached_mx).push_back(L);
                } else {
                    (is_nv ? uncached_nv : uncached_mx).push_back(L);
                }
            }
            const bool need_staging = cached_layers.size() < native_layers.size();
            const bool has_staging = model.bf16_staging_w13() != 0 && model.bf16_staging_w2() != 0;
            std::printf("v6 pack: %zu native layers (%zu NVFP4, %zu MXFP8), %zu cached (%zu NVFP4, %zu MXFP8), staging=%d (expected=%d)\n",
                        native_layers.size(), nvfp4_layers.size(), mxfp8_layers.size(),
                        cached_layers.size(), cached_nv.size(), cached_mx.size(),
                        int(has_staging), int(need_staging));
            if (has_staging != need_staging)
                throw std::runtime_error("bf16-cache-check: staging allocation mismatch");
            const char* cm = std::getenv("G2E_INK_BF16_CACHE");
            const bool expect_none = (cm && std::string(cm) == "none");
            if (expect_none) {
                if (!cached_layers.empty())
                    throw std::runtime_error("bf16-cache-check: expected 0 cached layers for mode=none");
            } else if (cached_layers.size() < 3) {
                throw std::runtime_error("bf16-cache-check: expected at least 3 cached layers");
            }
            // Select cached + staged layers so NVFP4 and at least two MXFP8 layers are always verified.
            std::vector<std::pair<uint32_t, bool>> check_layers;  // (layer, via_staging)
            auto add_check = [&](uint32_t L, bool staged) {
                for (const auto& [x, _] : check_layers)
                    if (x == L) return;
                check_layers.push_back({L, staged});
            };
            if (!cached_nv.empty()) {
                add_check(cached_nv.front(), false);
                add_check(cached_nv[cached_nv.size() / 2], false);
                add_check(cached_nv.back(), false);
            } else if (!uncached_nv.empty()) {
                add_check(uncached_nv.front(), true);
            }
            if (!cached_mx.empty()) {
                add_check(cached_mx.front(), false);
                add_check(cached_mx.back(), false);
            }
            if (!uncached_mx.empty()) {
                add_check(uncached_mx.front(), true);
                add_check(uncached_mx.back(), true);
            }
            size_t mx_checked = 0;
            for (const auto& [L, _] : check_layers) mx_checked += cfg.is_mxfp8(L);
            if (mxfp8_layers.size() >= 2 && mx_checked < 2)
                throw std::runtime_error("bf16-cache-check: expected at least 2 MXFP8 layers checked");

            static const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                            -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
            float f8_lut[256], e4m3fn_lut[256];
            for (int i = 0; i < 256; ++i) {
                f8_lut[i] = from_fp8_143(uint8_t(i));
                const float sgn = (i & 0x80) ? -1.f : 1.f;
                const int ex = (i >> 3) & 15, mn = i & 7;
                if (ex == 0) e4m3fn_lut[i] = sgn * std::ldexp(float(mn), -9);
                else if (ex == 15 && mn == 7) e4m3fn_lut[i] = std::nanf("");
                else e4m3fn_lut[i] = sgn * std::ldexp(1.f + float(mn) / 8.f, ex - 7);
            }
            const uint64_t E = cfg.experts, H = cfg.hidden, F = cfg.moe_inter;
            std::vector<uint16_t> got13, got2;
            uint64_t total_diffs = 0;
            auto check_mat = [&](const char* tag, bool is_nv, const std::string& wq_name, const std::string& bs_name,
                                 uint64_t K, uint64_t N, const std::vector<uint16_t>& got) {
                const auto wq = pack.read(wq_name);
                const auto bs = pack.read(bs_name);
                uint64_t diffs = 0, exact_bits = 0;
                const uint64_t NG = N / 256;
                if (is_nv) {
                    for (uint64_t e = 0; e < E; ++e) {
                        const uint8_t* wqe = wq.data() + e * (K / 2) * N;
                        const uint8_t* bse = bs.data() + e * (K / 16) * N;
                        const uint16_t* ge = got.data() + e * K * N;
                        for (uint64_t kb = 0; kb < K / 16; ++kb) {
                            const uint8_t* bsk = bse + kb * N;
                            for (uint64_t k_in = 0; k_in < 16; ++k_in) {
                                const uint64_t k = kb * 16 + k_in;
                                const uint8_t* wkr = wqe + (k >> 1) * N;
                                const uint16_t* gkr = ge + k * N;
                                const bool hi = (k & 1) != 0;
                                for (uint64_t g = 0; g < NG; ++g) {
                                    const uint64_t c0 = g * 256;
                                    for (uint64_t i = 0; i < 128; ++i) {
                                        const uint8_t bev = wkr[c0 + 2 * i];
                                        const uint8_t bod = wkr[c0 + 2 * i + 1];
                                        const uint8_t nev = hi ? (bev >> 4) : (bev & 15);
                                        const uint8_t nod = hi ? (bod >> 4) : (bod & 15);
                                        const float sev = f8_lut[bsk[c0 + 2 * i]];
                                        const float sod = f8_lut[bsk[c0 + 2 * i + 1]];
                                        const uint16_t rev = to_bf16((kE2M1[nev] * 0.015625f) * sev);
                                        const uint16_t rod = to_bf16((kE2M1[nod] * 0.015625f) * sod);
                                        const uint16_t aev = gkr[c0 + i];
                                        const uint16_t aod = gkr[c0 + 128 + i];
                                        exact_bits += (rev == aev) + (rod == aod);
                                        const bool eq_ev = (rev == aev) || (((rev | aev) & 0x7fff) == 0);
                                        const bool eq_od = (rod == aod) || (((rod | aod) & 0x7fff) == 0);
                                        diffs += (!eq_ev) + (!eq_od);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for (uint64_t e = 0; e < E; ++e) {
                        const uint8_t* wqe = wq.data() + e * K * N;
                        const uint8_t* bse = bs.data() + e * (K / 32) * N;
                        const uint16_t* ge = got.data() + e * K * N;
                        for (uint64_t kb = 0; kb < K / 32; ++kb) {
                            const uint8_t* bsk = bse + kb * N;
                            for (uint64_t k_in = 0; k_in < 32; ++k_in) {
                                const uint64_t k = kb * 32 + k_in;
                                const uint8_t* wkr = wqe + k * N;
                                const uint16_t* gkr = ge + k * N;
                                for (uint64_t g = 0; g < NG; ++g) {
                                    const uint64_t c0 = g * 256;
                                    for (uint64_t i = 0; i < 128; ++i) {
                                        const uint8_t bev = wkr[c0 + 2 * i];
                                        const uint8_t bod = wkr[c0 + 2 * i + 1];
                                        const float sev = f8_lut[bsk[c0 + 2 * i]];
                                        const float sod = f8_lut[bsk[c0 + 2 * i + 1]];
                                        const uint16_t rev = to_bf16(e4m3fn_lut[bev] * sev);
                                        const uint16_t rod = to_bf16(e4m3fn_lut[bod] * sod);
                                        const uint16_t aev = gkr[c0 + i];
                                        const uint16_t aod = gkr[c0 + 128 + i];
                                        exact_bits += (rev == aev) + (rod == aod);
                                        const bool eq_ev = (rev == aev) || (((rev | aev) & 0x7fff) == 0);
                                        const bool eq_od = (rod == aod) || (((rod | aod) & 0x7fff) == 0);
                                        diffs += (!eq_ev) + (!eq_od);
                                    }
                                }
                            }
                        }
                    }
                }
                const uint64_t tot = E * K * N;
                std::printf("  %s: %llu diffs / %llu values (bit-exact %llu / %llu)\n", tag,
                            (unsigned long long)diffs, (unsigned long long)tot,
                            (unsigned long long)exact_bits, (unsigned long long)tot);
                total_diffs += diffs;
            };
            for (const auto& [L, staged] : check_layers) {
                if (staged) model.download_bf16_staged(L, got13, got2);
                else model.download_bf16_layer(L, got13, got2);
                const bool is_nv = cfg.is_nvfp4(L);
                const std::string p = "L" + std::to_string(L) + ".";
                const std::string meta = std::string(is_nv ? "nvfp4" : "mxfp8") + "," + (staged ? "staged" : "cached");
                check_mat((p + "w13 [" + meta + "]").c_str(), is_nv, p + "w13.wq", p + "w13.bs", H, 2 * F, got13);
                check_mat((p + "w2  [" + meta + "]").c_str(), is_nv, p + "w2.wq", p + "w2.bs", F, H, got2);
            }
            if (total_diffs != 0)
                throw std::runtime_error("bf16-cache-check: found " + std::to_string(total_diffs) + " differences");
            std::printf("bf16-cache-check: PASS (%zu layers checked, %zu MXFP8)\n", check_layers.size(), mx_checked);
            return 0;
        }
        for (const uint32_t T : precompile) {
            const auto c0 = std::chrono::steady_clock::now();
            const size_t n = model.precompile_far(T);
            std::printf("precompile T=%u: %zu recipes in %.1f s\n", T, n,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count());
        }
        for (const uint32_t T : precompile_vh) {
            const auto c0 = std::chrono::steady_clock::now();
            const size_t n = model.precompile_far(T, true);
            std::printf("precompile vh T=%u: %zu recipes in %.1f s\n", T, n,
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count());
        }
        if (precompile_only) return 0;
        const auto& cfg = model.config();
        const auto ids = read_raw<int32_t>(golden + "/ids.i32");
        const auto ref_last = read_raw<float>(golden + "/logits_last.f32");
        const auto dec_ids = read_raw<int32_t>(golden + "/dec_ids.i32");
        const auto dec_logits = read_raw<float>(golden + "/dec_logits.f32");
        const size_t V = cfg.vocab, T = ids.size(), H = cfg.hidden;
        model.set_dump(dump);
        auto run_prefill = [&](std::span<const int32_t> toks, bool want_logits) {
            if (!decode_prefill) return model.forward(toks, want_logits);
            std::vector<float> out;
            for (size_t i = 0; i < toks.size(); ++i) {
                const bool last = (i + 1 == toks.size());
                auto lg_i = model.forward(std::span<const int32_t>(&toks[i], 1), want_logits && last);
                if (last) out = std::move(lg_i);
            }
            return out;
        };
        auto t0 = std::chrono::steady_clock::now();
        if (split && split < ids.size()) run_prefill(std::span<const int32_t>(ids.data(), split), false);
        auto lg = split && split < ids.size() ? run_prefill(std::span<const int32_t>(ids.data() + split, ids.size() - split), true)
                                              : run_prefill(ids, true);
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "%s T=%zu: %.1f ms%s\n", decode_prefill ? "decode-prefill" : "prefill", T, ms,
                     decode_prefill ? "" : " (first call includes graph compiles)");
        auto c = compare(ref_last.data(), lg);
        std::printf("prefill last row: top1 %s (ref %d got %d) KL %.5f maxdiff %.3f\n", c.top1 ? "OK" : "MISMATCH", c.ref_top, c.top,
                    c.kl, c.maxdiff);
        if (dump) {
            const auto hs = read_raw<float>(golden + "/hs.f32");
            const auto& d = model.dumps();
            const size_t Tl = d.empty() ? 0 : d[0].size() / H;  // rows of the last forward piece
            const size_t r0 = T - Tl;
            for (size_t k = 0; k < d.size(); ++k) {
                double num = 0, den = 0, mx = 0;
                for (size_t r = 0; r < Tl; ++r)
                    for (size_t j = 0; j < H; ++j) {
                        const double ref = hs[((k + 1) * T + r0 + r) * H + j];
                        const double got = from_bf16(d[k][r * H + j]);
                        num += (ref - got) * (ref - got);
                        den += ref * ref;
                        mx = std::max(mx, std::fabs(ref - got));
                    }
                std::printf("  app %2zu: rel %.4f maxabs %.4f\n", k, std::sqrt(num / std::max(den, 1e-30)), mx);
            }
            model.set_dump(false);
        }
        // teacher-forced decode against the HF greedy continuation
        int ok = 0;
        double kl_sum = 0, dms = 0;
        const size_t nd = std::min<size_t>(n_decode, dec_ids.size());
        for (size_t i = 0; i < nd; ++i) {
            const int32_t tok = dec_ids[i];
            t0 = std::chrono::steady_clock::now();
            lg = model.forward(std::span<const int32_t>(&tok, 1), true);
            dms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            c = compare(dec_logits.data() + i * V, lg);
            ok += c.top1;
            kl_sum += c.kl;
            if (!c.top1) std::printf("  decode %zu: ref %d got %d KL %.4f\n", i, c.ref_top, c.top, c.kl);
        }
        std::printf("decode: %d/%zu top1, mean KL %.5f, %.2f ms/token (incl. first-call compiles)\n", ok, nd, kl_sum / std::max<size_t>(nd, 1),
                    dms / std::max<size_t>(nd, 1));
        if (bench) {
            int32_t tok = dec_ids.back();
            model.print_profile();  // drop the warm-up part
            t0 = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < bench; ++i) tok = model.forward_argmax(std::span<const int32_t>(&tok, 1));
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            const auto& tm = model.last_timing();
            std::printf("bench decode: %.2f ms/token (%.1f tok/s); last: embed %.2f layers %.2f head %.2f ms\n", ms / bench,
                        1000.0 * bench / ms, tm.embed, tm.layers, tm.head);
            model.print_profile();
        }
        if (direct_check) {
            // the same decode token through the Synapse path and the Direct path from the same state
            DeviceBuffer snap(dev, model.state_bytes(model.position()));
            const uint32_t p0 = model.position();
            model.save_state(snap, 0);
            stream.synchronize();
            const int32_t tok = dec_ids.empty() ? ids.back() : dec_ids[0];
            model.set_dump(true);
            model.set_direct(false);
            const auto lg0 = model.forward(std::span<const int32_t>(&tok, 1), true);
            const auto d0 = model.dumps();
            model.load_state(snap, 0, p0);
            stream.synchronize();
            model.set_direct(true);
            const auto lg1 = model.forward(std::span<const int32_t>(&tok, 1), true);
            const auto d1 = model.dumps();
            model.set_dump(false);
            for (size_t i = 0; i < d0.size() && i < d1.size(); ++i) {
                double mx = 0, num = 0, den = 0;
                for (size_t k = 0; k < d0[i].size(); ++k) {
                    const double x = from_bf16(d1[i][k]), y = from_bf16(d0[i][k]);
                    mx = std::max(mx, std::fabs(x - y));
                    num += (x - y) * (x - y);
                    den += y * y;
                }
                std::printf("app %2zu (layer %2u pass %u): max |diff| %.4f rel %.2e\n", i, model.app_layer(uint32_t(i)), model.app_pass(uint32_t(i)), mx,
                            std::sqrt(num / (den + 1e-30)));
            }
            auto c = compare(lg0.data(), lg1);
            std::printf("logits Direct vs Synapse: top1 %s KL %.5f maxdiff %.3f\n", c.top1 ? "same" : "DIFF", c.kl, c.maxdiff);
        }
        if (verify_check) {
            DeviceBuffer snap(dev, model.state_bytes(model.position()));
            const uint32_t p0 = model.position();
            model.save_state(snap, 0);
            stream.synchronize();
            std::vector<int32_t> blk;
            for (int i = 0; i < 4; ++i) blk.push_back(dec_ids.size() > size_t(i) ? dec_ids[i] : ids[ids.size() - 4 + i]);
            if (decode_prefill) {
                // Direct-only verification check (no Synapse): compare Direct verify(T=4) and verify(T=2)+accept(2)
                // against sequential Direct B=1 decode from the same saved state.
                std::vector<std::vector<float>> ref_l(4);
                for (int r = 0; r < 4; ++r) ref_l[r] = model.forward(std::span<const int32_t>(&blk[r], 1), true);
                for (uint32_t vt : {4u, 2u}) {
                    model.load_state(snap, 0, p0);
                    stream.synchronize();
                    const auto vl = model.verify(std::span<const int32_t>(blk.data(), vt));
                    for (uint32_t r = 0; r < vt; ++r) {
                        std::vector<float> got(vl.begin() + long(r) * cfg.vocab, vl.begin() + long(r + 1) * cfg.vocab);
                        auto c = compare(ref_l[r].data(), got);
                        std::printf("verify T=%u row %u: Direct verify vs B=1 top1 %s KL %.5f maxdiff %.3f\n",
                                    vt, r, c.top1 ? "same" : "DIFF", c.kl, c.maxdiff);
                    }
                    const uint32_t acc = (vt == 4u) ? 2u : 0u;
                    model.accept(acc);
                    if (acc == 2u) {
                        const auto after = model.forward(std::span<const int32_t>(&blk[2], 1), true);
                        auto c = compare(ref_l[2].data(), after);
                        std::printf("verify T=4 accept(2) -> row 2: top1 %s KL %.5f maxdiff %.3f\n",
                                    c.top1 ? "same" : "DIFF", c.kl, c.maxdiff);
                    }
                }
                model.load_state(snap, 0, p0);
                stream.synchronize();
            } else {
                model.set_dump(true);
                model.set_direct(false);
                const auto l0 = model.verify(blk);
                const auto d0 = model.dumps();
                model.accept(0);
                model.load_state(snap, 0, p0);
                stream.synchronize();
                model.set_direct(true);
                const auto l1 = model.verify(blk);
                const auto d1 = model.dumps();
                model.accept(0);
                model.set_dump(false);
                for (size_t i = 0; i < d0.size() && i < d1.size(); ++i) {
                    double mx = 0, num = 0, den = 0;
                    for (size_t k = 0; k < d0[i].size(); ++k) {
                        const double x = from_bf16(d1[i][k]), y = from_bf16(d0[i][k]);
                        mx = std::max(mx, std::fabs(x - y));
                        num += (x - y) * (x - y);
                        den += y * y;
                    }
                    std::printf("app %2zu (layer %2u pass %u): max |diff| %.4f rel %.2e  rows", i, model.app_layer(uint32_t(i)), model.app_pass(uint32_t(i)), mx,
                                std::sqrt(num / (den + 1e-30)));
                    const size_t H = cfg.hidden;
                    for (size_t r = 0; r * H < d0[i].size(); ++r) {
                        double n2 = 0, d2 = 0;
                        for (size_t k = r * H; k < (r + 1) * H && k < d0[i].size(); ++k) {
                            const double x = from_bf16(d1[i][k]), y = from_bf16(d0[i][k]);
                            n2 += (x - y) * (x - y);
                            d2 += y * y;
                        }
                        std::printf(" %.1e", std::sqrt(n2 / (d2 + 1e-30)));
                    }
                    std::printf("\n");
                }
                for (int r = 0; r < 4; ++r) {
                    std::vector<float> got(l1.begin() + long(r) * cfg.vocab, l1.begin() + long(r + 1) * cfg.vocab);
                    auto c = compare(l0.data() + uint64_t(r) * cfg.vocab, got);
                    std::printf("verify row %d: Direct vs Synapse top1 %s KL %.5f maxdiff %.3f\n", r, c.top1 ? "same" : "DIFF", c.kl, c.maxdiff);
                }
            }
        }
        if (direct_prof >= 0) model.direct_profile(uint32_t(direct_prof));
        if (spec_test) {
            // greedy speculative decoding with the DSpark drafter vs plain greedy decoding of the same prompt
            WeightPack dp(need_dspark_pack(dspark_pack));
            dp.upload(dev, stream);
            InkDSpark ds(dev, stream, dp, pack, max_ctx);
            auto argmax = [&](const float* v) { return int32_t(std::max_element(v, v + cfg.vocab) - v); };
            const uint32_t vt = std::clamp<uint32_t>(std::getenv("G2E_SPEC_T") ? uint32_t(std::atoi(std::getenv("G2E_SPEC_T"))) : 4u, 2u, InkDSpark::kBlock);
            const uint32_t ndrafts = vt - 1;
            // plain greedy reference
            model.reset();
            std::vector<int32_t> ref;
            {
                auto lg = model.forward(ids, true);
                int32_t t = argmax(lg.data());
                const auto c0 = std::chrono::steady_clock::now();
                for (uint32_t i = 0; i < spec_test; ++i) {
                    ref.push_back(t);
                    lg = model.forward(std::span<const int32_t>(&t, 1), true);
                    t = argmax(lg.data());
                }
                std::printf("plain greedy: %u tokens, %.2f ms/token\n", spec_test,
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count() / spec_test);
            }
            // speculative
            model.reset();
            ds.reset();
            model.set_aux_capture(true);
            uint32_t last_pos0 = 0;
            model.set_tap_hook([&](uint32_t p0, uint32_t n) {
                ds.append(model.taps(), 0, model.tap_stride(), n, p0);
                model.invalidate_sram();
                last_pos0 = p0;
            });
            auto lg = model.forward(ids, true);
            model.set_tap_hook({});
            std::vector<int32_t> out;
            int32_t anchor = argmax(lg.data());
            uint32_t apos = uint32_t(ids.size());
            uint64_t last_row = apos - 1 - last_pos0;  // the taps row of position apos - 1
            uint32_t rounds = 0;
            double ms_draft = 0, ms_verify = 0, ms_accept = 0, ss_ms = 0, ss_v = 0;
            uint32_t ss_tok = 0, ss_rounds = 0;
            const auto c0 = std::chrono::steady_clock::now();
            while (out.size() < spec_test) {
                out.push_back(anchor);
                auto t0 = std::chrono::steady_clock::now();
                const auto d = ds.draft(anchor, apos, model.taps(), last_row * cfg.hidden * 2, model.tap_stride(), ndrafts);
                model.invalidate_sram();
                auto t1 = std::chrono::steady_clock::now();
                std::vector<int32_t> blk = {anchor};
                blk.insert(blk.end(), d.begin(), d.end());
                const auto vl = model.verify(blk);
                auto t2 = std::chrono::steady_clock::now();
                uint32_t j = 0;
                while (j + 1 < vt && d[j] == argmax(vl.data() + uint64_t(j) * cfg.vocab)) ++j;
                for (uint32_t i = 0; i < j; ++i) out.push_back(d[i]);
                model.accept(j + 1);
                ds.append(model.taps(), 0, model.tap_stride(), j + 1, apos);
                model.invalidate_sram();
                auto t3 = std::chrono::steady_clock::now();
                anchor = argmax(vl.data() + uint64_t(j) * cfg.vocab);
                apos += j + 1;
                last_row = j;
                ++rounds;
                ms_draft += std::chrono::duration<double, std::milli>(t1 - t0).count();
                ms_verify += std::chrono::duration<double, std::milli>(t2 - t1).count();
                ms_accept += std::chrono::duration<double, std::milli>(t3 - t2).count();
                if (rounds > 24) {  // steady state (the first rounds build the plans)
                    ss_ms += std::chrono::duration<double, std::milli>(t3 - t0).count();
                    ss_v += std::chrono::duration<double, std::milli>(t2 - t1).count();
                    ss_tok += j + 1;
                    ++ss_rounds;
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
            model.set_aux_capture(false);
            out.resize(spec_test);
            uint32_t same = 0;
            while (same < spec_test && out[same] == ref[same]) ++same;
            if (std::getenv("G2E_SPEC_PROF")) model.direct_profile_v(uint32_t(std::atoi(std::getenv("G2E_SPEC_PROF"))), vt);
            std::printf("speculative: %u tokens in %u rounds (%.2f tokens/round), %.2f ms/token (%.1f tok/s); per round draft %.2f, verify %.2f, "
                        "accept %.2f ms; identical to greedy for the first %u tokens\n",
                        spec_test, rounds, double(spec_test) / rounds, ms / spec_test, 1000.0 * spec_test / ms, ms_draft / rounds, ms_verify / rounds,
                        ms_accept / rounds, same);
            if (ss_rounds)
                std::printf("  steady state (rounds after 24): %.2f ms/token (%.1f tok/s), verify %.2f ms, %.2f tokens/round\n",
                            ss_ms / ss_tok, 1000.0 * ss_tok / ss_ms, ss_v / ss_rounds, double(ss_tok) / ss_rounds);
        }
        if (spec_sampled_test) {
            // Lossless sampled DSpark speculative decoding verification + benchmark (temp=0.7, top_k=50, top_p=0.95, min_p=0.0)
            WeightPack dp(need_dspark_pack(dspark_pack));
            dp.upload(dev, stream);
            InkDSpark ds(dev, stream, dp, pack, max_ctx);
            const uint32_t vt = std::clamp<uint32_t>(std::getenv("G2E_SPEC_T") ? uint32_t(std::atoi(std::getenv("G2E_SPEC_T"))) : 4u, 2u, InkDSpark::kBlock);
            const uint32_t ndrafts = vt - 1;
            const float temp = 0.7f, top_p = 0.95f, min_p = 0.0f;
            const int top_k = 50;

            auto build_target_dist = [&](const float* lg_ptr, float t_use) {
                const float inv_t = 1.0f / t_use;
                std::vector<std::pair<float, int32_t>> c;
                auto cmp = [](const auto& a, const auto& b) {
                    return a.first > b.first || (a.first == b.first && a.second < b.second);
                };
                if (top_k > 0 && size_t(top_k) <= 512 && size_t(top_k) < cfg.vocab) {
                    const size_t K = size_t(top_k);
                    std::vector<std::pair<float, int32_t>> h(K);
                    for (size_t i = 0; i < K; ++i) h[i] = {lg_ptr[i], int32_t(i)};
                    auto worse = [](const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) {
                        return a.first < b.first || (a.first == b.first && a.second > b.second);
                    };
                    auto sift_down = [&](size_t pos) {
                        const auto val = h[pos];
                        while (true) {
                            const size_t l = 2 * pos + 1;
                            if (l >= K) break;
                            const size_t r = l + 1;
                            const size_t sm = (r < K && worse(h[r], h[l])) ? r : l;
                            if (!worse(h[sm], val)) break;
                            h[pos] = h[sm];
                            pos = sm;
                        }
                        h[pos] = val;
                    };
                    for (size_t i = K / 2; i-- > 0;) sift_down(i);
                    float thresh = h[0].first;
                    for (size_t i = K; i < cfg.vocab; ++i) {
                        const float v = lg_ptr[i];
                        if (__builtin_expect(v > thresh, 0)) {
                            h[0] = {v, int32_t(i)};
                            sift_down(0);
                            thresh = h[0].first;
                        }
                    }
                    std::sort(h.begin(), h.end(), cmp);
                    const float mx = h[0].first;
                    c.reserve(K);
                    for (const auto& [v, id] : h) {
                        const float z = (v - mx) * inv_t;
                        if (z > -30.f) c.push_back({z, id});
                        else break;
                    }
                } else {
                    const float mx = *std::max_element(lg_ptr, lg_ptr + cfg.vocab);
                    const float cut_z = (top_k <= 0 && top_p >= 1.0f && min_p > 0.0f)
                                            ? std::max(-30.f, std::log(min_p) - 1e-4f)
                                            : -30.f;
                    const float cut_lg = mx + cut_z * t_use;
                    c.reserve(4096);
                    for (size_t i = 0; i < cfg.vocab; ++i)
                        if (lg_ptr[i] > cut_lg) c.push_back({(lg_ptr[i] - mx) * inv_t, int32_t(i)});
                    if (top_k > 0 && size_t(top_k) < c.size()) {
                        std::nth_element(c.begin(), c.begin() + long(top_k), c.end(), cmp);
                        c.resize(size_t(top_k));
                    }
                    std::sort(c.begin(), c.end(), cmp);
                }
                double sum = 0.0;
                std::vector<double> pr(c.size());
                for (size_t i = 0; i < c.size(); ++i) sum += pr[i] = std::exp(double(c[i].first));
                size_t n = c.size();
                if (top_p < 1.0f) {
                    double acc = 0.0;
                    for (size_t i = 0; i < c.size(); ++i) {
                        acc += pr[i] / sum;
                        if (acc >= top_p) { n = i + 1; break; }
                    }
                }
                if (min_p > 0.0f) {
                    const double thr = min_p * pr[0];
                    size_t m = 0;
                    while (m < n && pr[m] >= thr) ++m;
                    n = std::max<size_t>(m, 1);
                }
                double tot = 0.0;
                for (size_t i = 0; i < n; ++i) tot += pr[i];
                std::vector<std::pair<int32_t, double>> dist;
                dist.reserve(n);
                for (size_t i = 0; i < n; ++i) dist.push_back({c[i].second, pr[i] / tot});
                return dist;
            };
            auto draw_dist = [](const std::vector<std::pair<int32_t, double>>& dist, std::mt19937_64& rng) {
                double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
                for (const auto& [id, pr] : dist) {
                    u -= pr;
                    if (u <= 0.0) return id;
                }
                return dist.back().first;
            };
            auto check_sd = [&](const InkDSpark::SampledDraft& sd, const char* ctx) {
                if (sd.drafts.size() != ndrafts || sd.cand_ids.size() != ndrafts * sd.k || sd.cand_q.size() != ndrafts * sd.k)
                    throw std::runtime_error(std::string(ctx) + ": bad SampledDraft sizes");
                for (uint32_t i = 0; i < ndrafts; ++i) {
                    double qsum = 0.0, qdraft = 0.0;
                    bool found = false;
                    for (uint32_t k = 0; k < sd.k; ++k) {
                        const int32_t cid = sd.cand_ids[size_t(i) * sd.k + k];
                        const float qv = sd.cand_q[size_t(i) * sd.k + k];
                        if (cid < 0 || uint32_t(cid) >= cfg.vocab)
                            throw std::runtime_error(std::string(ctx) + ": cand_id out of vocab range");
                        if (!std::isfinite(qv) || qv < -1e-7f)
                            throw std::runtime_error(std::string(ctx) + ": negative/non-finite cand_q");
                        qsum += double(qv);
                        if (cid == sd.drafts[i]) { found = true; qdraft += double(qv); }
                    }
                    if (std::fabs(qsum - 1.0) > 1e-3)
                        throw std::runtime_error(std::string(ctx) + ": sum(cand_q) != 1 (" + std::to_string(qsum) + ")");
                    if (!found || qdraft <= 0.0)
                        throw std::runtime_error(std::string(ctx) + ": draft not in cand_ids with q > 0");
                }
            };

            // Prefill prompt with taps
            model.reset();
            ds.reset();
            model.set_aux_capture(true);
            uint32_t last_pos0 = 0;
            model.set_tap_hook([&](uint32_t p0, uint32_t n) {
                ds.append(model.taps(), 0, model.tap_stride(), n, p0);
                model.invalidate_sram();
                last_pos0 = p0;
            });
            auto lg = model.forward(ids, true);
            model.set_tap_hook({});

            std::mt19937_64 rng(12345);
            const auto first_dist = build_target_dist(lg.data(), temp);
            int32_t anchor = draw_dist(first_dist, rng);
            uint32_t apos = uint32_t(ids.size());
            uint64_t last_row = apos - 1 - last_pos0;

            // ---- 1. Statistical losslessness check on row 0 (both temp=0.7 and high-entropy temp=2.5) ----
            {
                const auto sd_probe = ds.draft_sampled(anchor, apos, model.taps(), last_row * cfg.hidden * 2, model.tap_stride(),
                                                       temp, top_k, top_p, min_p, rng, ndrafts);
                model.invalidate_sram();
                check_sd(sd_probe, "spec-sampled row0");
                std::vector<int32_t> blk0 = {anchor};
                blk0.insert(blk0.end(), sd_probe.drafts.begin(), sd_probe.drafts.end());
                const auto vl0 = model.verify(blk0);
                model.accept(0);

                for (const float chk_t : {temp, 2.5f}) {
                    const auto p0 = build_target_dist(vl0.data(), chk_t);
                    std::mt19937_64 dev_rng(777);
                    const auto sd_ref = ds.draft_sampled(anchor, apos, model.taps(), last_row * cfg.hidden * 2, model.tap_stride(),
                                                         chk_t, top_k, top_p, min_p, dev_rng, ndrafts);
                    model.invalidate_sram();
                    check_sd(sd_ref, "spec-sampled dev-check");
                    std::unordered_map<int32_t, int> dev_cnt;
                    constexpr int kDevDraws = 128;
                    dev_cnt[sd_ref.drafts[0]]++;
                    for (int d = 1; d < kDevDraws; ++d) {
                        const auto sdi = ds.draft_sampled(anchor, apos, model.taps(), last_row * cfg.hidden * 2, model.tap_stride(),
                                                          chk_t, top_k, top_p, min_p, dev_rng, ndrafts);
                        model.invalidate_sram();
                        for (uint32_t k = 0; k < sd_ref.k; ++k) {
                            if (sdi.cand_ids[k] != sd_ref.cand_ids[k] || std::fabs(sdi.cand_q[k] - sd_ref.cand_q[k]) > 1e-6f)
                                throw std::runtime_error("spec-sampled dev-check: non-deterministic cand_ids/cand_q");
                        }
                        dev_cnt[sdi.drafts[0]]++;
                    }
                    double dev_l1 = 0.0;
                    for (uint32_t k = 0; k < sd_ref.k; ++k) {
                        const double qk = double(sd_ref.cand_q[k]);
                        auto it = dev_cnt.find(sd_ref.cand_ids[k]);
                        const double emp = it != dev_cnt.end() ? double(it->second) / double(kDevDraws) : 0.0;
                        dev_l1 += std::fabs(emp - qk);
                    }
                    const double dev_tv = 0.5 * dev_l1;

                    // 20,000 Monte-Carlo rejection-sampler draws using actual device q0 and target p0
                    std::unordered_map<int32_t, double> pmap, qmap;
                    for (const auto& [id, pr] : p0) pmap[id] += pr;
                    std::vector<std::pair<int32_t, double>> q0_dist;
                    q0_dist.reserve(sd_ref.k);
                    for (uint32_t k = 0; k < sd_ref.k; ++k) {
                        const double qk = double(sd_ref.cand_q[k]);
                        if (qk > 0.0) {
                            qmap[sd_ref.cand_ids[k]] += qk;
                            q0_dist.push_back({sd_ref.cand_ids[k], qk});
                        }
                    }
                    std::vector<std::pair<int32_t, double>> resid;
                    double rsum = 0.0;
                    for (const auto& [id, pr] : p0) {
                        const double r = std::max(0.0, pr - qmap[id]);
                        if (r > 0.0) { resid.push_back({id, r}); rsum += r; }
                    }
                    if (rsum > 0.0) for (auto& [id, r] : resid) r /= rsum;
                    else resid = p0;

                    std::mt19937_64 mc_rng(2026);
                    std::uniform_real_distribution<double> u01(0.0, 1.0);
                    std::unordered_map<int32_t, uint32_t> mc_cnt;
                    constexpr uint32_t kMC = 20000;
                    uint32_t mc_acc = 0;
                    for (uint32_t m = 0; m < kMC; ++m) {
                        const int32_t x = draw_dist(q0_dist, mc_rng);
                        const double pd = pmap.count(x) ? pmap.at(x) : 0.0;
                        const double qd = qmap.at(x);
                        int32_t emitted = x;
                        if (u01(mc_rng) < std::min(1.0, pd / std::max(qd, 1e-30))) {
                            ++mc_acc;
                        } else {
                            emitted = draw_dist(resid, mc_rng);
                        }
                        mc_cnt[emitted]++;
                    }
                    double l1 = 0.0;
                    for (const auto& [id, pr] : p0) {
                        auto it = mc_cnt.find(id);
                        const double emp = it != mc_cnt.end() ? double(it->second) / double(kMC) : 0.0;
                        l1 += std::fabs(emp - pr);
                    }
                    for (const auto& [id, cnt] : mc_cnt) if (!pmap.count(id)) l1 += double(cnt) / double(kMC);
                    const double tv = 0.5 * l1;
                    std::printf("spec-sampled losslessness (row 0, temp=%.2f top_k=%d top_p=%.2f): MC=%u acc=%.3f TV=%.5f L1=%.5f (p_supp=%zu q_supp=%zu); "
                                "device-draw TV=%.4f (%d draws, %zu unique)\n",
                                chk_t, top_k, top_p, kMC, double(mc_acc) / double(kMC), tv, l1, p0.size(), q0_dist.size(), dev_tv, kDevDraws, dev_cnt.size());
                    if (tv >= 0.03)
                        throw std::runtime_error("spec-sampled losslessness failed: TV=" + std::to_string(tv));
                }
            }

            // ---- 2. Benchmark N generated tokens with lossless sampled speculative decoding ----
            std::vector<int32_t> out;
            uint32_t rounds = 0;
            double ms_draft = 0, ms_verify = 0, ms_accept = 0, ss_ms = 0, ss_v = 0;
            uint32_t ss_tok = 0, ss_rounds = 0;
            std::uniform_real_distribution<double> u01(0.0, 1.0);
            const auto c0 = std::chrono::steady_clock::now();
            while (out.size() < spec_sampled_test) {
                out.push_back(anchor);
                auto t0 = std::chrono::steady_clock::now();
                const auto sd = ds.draft_sampled(anchor, apos, model.taps(), last_row * cfg.hidden * 2, model.tap_stride(),
                                                 temp, top_k, top_p, min_p, rng, ndrafts);
                model.invalidate_sram();
                auto t1 = std::chrono::steady_clock::now();
                check_sd(sd, "spec-sampled bench");
                std::vector<int32_t> blk = {anchor};
                blk.insert(blk.end(), sd.drafts.begin(), sd.drafts.end());
                const auto vl = model.verify(blk);
                auto t2 = std::chrono::steady_clock::now();
                const uint32_t K = sd.k;
                auto q_of = [&](uint32_t step, int32_t id) {
                    double q = 0.0;
                    const size_t base = size_t(step) * K;
                    for (uint32_t k = 0; k < K; ++k)
                        if (sd.cand_ids[base + k] == id) q += double(sd.cand_q[base + k]);
                    return q;
                };
                uint32_t j = 0;
                std::vector<std::pair<int32_t, double>> rej_p;
                while (j + 1 < vt) {
                    auto pa = build_target_dist(vl.data() + uint64_t(j) * cfg.vocab, temp);
                    double pd = 0.0;
                    for (const auto& [id, pr] : pa) if (id == sd.drafts[j]) { pd = pr; break; }
                    const double qd = q_of(j, sd.drafts[j]);
                    if (u01(rng) < std::min(1.0, pd / std::max(qd, 1e-30))) {
                        ++j;
                    } else {
                        rej_p = std::move(pa);
                        break;
                    }
                }
                int32_t bonus = 0;
                if (j + 1 == vt) {
                    bonus = draw_dist(build_target_dist(vl.data() + uint64_t(j) * cfg.vocab, temp), rng);
                } else {
                    std::vector<double> w;
                    w.reserve(rej_p.size());
                    double total = 0.0;
                    for (const auto& [id, pr] : rej_p) {
                        const double r = std::max(0.0, pr - q_of(j, id));
                        w.push_back(r);
                        total += r;
                    }
                    if (total <= 0.0) {
                        bonus = draw_dist(rej_p, rng);
                    } else {
                        double u = std::uniform_real_distribution<double>(0.0, total)(rng);
                        bonus = rej_p.back().first;
                        for (size_t idx = 0; idx < rej_p.size(); ++idx) {
                            u -= w[idx];
                            if (u <= 0.0) { bonus = rej_p[idx].first; break; }
                        }
                    }
                }
                for (uint32_t i = 0; i < j; ++i) out.push_back(sd.drafts[i]);
                model.accept(j + 1);
                ds.append(model.taps(), 0, model.tap_stride(), j + 1, apos);
                model.invalidate_sram();
                auto t3 = std::chrono::steady_clock::now();
                anchor = bonus;
                apos += j + 1;
                last_row = j;
                ++rounds;
                ms_draft += std::chrono::duration<double, std::milli>(t1 - t0).count();
                ms_verify += std::chrono::duration<double, std::milli>(t2 - t1).count();
                ms_accept += std::chrono::duration<double, std::milli>(t3 - t2).count();
                if (rounds > 24) {
                    ss_ms += std::chrono::duration<double, std::milli>(t3 - t0).count();
                    ss_v += std::chrono::duration<double, std::milli>(t2 - t1).count();
                    ss_tok += j + 1;
                    ++ss_rounds;
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
            model.set_aux_capture(false);
            out.resize(spec_sampled_test);
            std::printf("speculative sampled: %u tokens in %u rounds (%.2f tokens/round), %.2f ms/token (%.1f tok/s); "
                        "per round draft %.2f, verify %.2f, accept %.2f ms\n",
                        spec_sampled_test, rounds, double(spec_sampled_test) / rounds, ms / spec_sampled_test,
                        1000.0 * spec_sampled_test / ms, ms_draft / rounds, ms_verify / rounds, ms_accept / rounds);
            if (ss_rounds)
                std::printf("  steady state (rounds after 24): %.2f ms/token (%.1f tok/s), verify %.2f ms, %.2f tokens/round\n",
                            ss_ms / ss_tok, 1000.0 * ss_tok / ss_ms, ss_v / ss_rounds, double(ss_tok) / ss_rounds);
        }
        if (!dspark_test.empty()) {
            // replay a --dspark-dump through the device drafter: per generated position p the drafts after the anchor
            // (the target token at p + 1) from the taps of rows <= p, against the target's greedy tokens
            FILE* f = std::fopen(dspark_test.c_str(), "rb");
            int32_t hdr[2];
            if (!f || std::fread(hdr, 4, 2, f) != 2) throw std::runtime_error("dspark-test: cannot read " + dspark_test);
            const uint32_t n = uint32_t(hdr[0]), H = uint32_t(hdr[1]);
            std::vector<int32_t> toks(n), nxt(n);
            std::vector<uint16_t> feats(uint64_t(n) * 4 * H);
            if (std::fread(toks.data(), 4, n, f) != n || std::fread(nxt.data(), 4, n, f) != n ||
                std::fread(feats.data(), 2, feats.size(), f) != feats.size())
                throw std::runtime_error("dspark-test: short file");
            std::fclose(f);
            WeightPack dp(need_dspark_pack(dspark_pack));
            dp.upload(dev, stream);
            InkDSpark ds(dev, stream, dp, pack, max_ctx);
            DeviceBuffer fb(dev, feats.size() * 2);
            const uint64_t ts = uint64_t(n) * H * 2;  // tap-major on the device: tap k of row r at k * ts + r * H * 2
            {
                HostBuffer hb(dev, feats.size() * 2);
                for (uint32_t r = 0; r < n; ++r)
                    for (int k = 0; k < 4; ++k)
                        std::memcpy(hb.as<uint16_t>() + (uint64_t(k) * n + r) * H, feats.data() + (uint64_t(r) * 4 + k) * H, H * 2);
                stream.upload(hb, 0, fb, 0, feats.size() * 2);
                stream.synchronize();
            }
            const uint32_t n_prompt = uint32_t(ids.size());
            uint32_t appended = 0;
            std::vector<int> acc;
            double ms = 0;
            for (uint32_t p = n_prompt - 1; p + 9 < n; p += dspark_step) {
                if (appended < p + 1) {
                    ds.append(fb, uint64_t(appended) * H * 2, ts, p + 1 - appended, appended);
                    model.invalidate_sram();
                    appended = p + 1;
                }
                const auto d = ds.draft(nxt[p], p + 1, fb, uint64_t(p) * H * 2, ts);
                model.invalidate_sram();
                ms += ds.last_ms();
                int k = 0;
                while (k < 7 && d[k] == toks[p + 2 + k]) ++k;
                acc.push_back(k);
                if (acc.size() <= 3) {
                    std::printf("p %u anchor %d drafts", p, nxt[p]);
                    for (int t : d) std::printf(" %d", t);
                    std::printf(" | target");
                    for (int i = 0; i < 7; ++i) std::printf(" %d", toks[p + 2 + i]);
                    std::printf("\n");
                }
            }
            double mean = 0;
            std::vector<int> hist(8, 0);
            for (int k : acc) { mean += k; hist[k]++; }
            std::printf("dspark device: %zu rounds, accepted drafts per round %.2f (tokens per round %.2f), %.2f ms per draft; histogram",
                        acc.size(), mean / acc.size(), mean / acc.size() + 1, ms / acc.size());
            for (int h : hist) std::printf(" %d", h);
            std::printf("\n");
        }
        if (dspark_dump) {
            // the prompt and N greedy tokens one at a time (T = 1) with the DSpark taps: file = header (n, H), tokens
            // i32 [n], target argmax after each token i32 [n], features bf16 [n, 4, H]
            model.reset();
            model.set_aux_capture(true);
            model.aux().clear();
            std::vector<int32_t> toks(ids.begin(), ids.end()), next_tok;
            for (size_t i = 0; i < toks.size() || next_tok.size() < ids.size() + dspark_dump; ++i) {
                if (i >= toks.size()) toks.push_back(next_tok.back());
                const auto lg = model.forward(std::span<const int32_t>(&toks[i], 1), true);
                next_tok.push_back(int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin()));
                if (next_tok.size() >= ids.size() + dspark_dump) break;
            }
            model.set_aux_capture(false);
            FILE* f = std::fopen(dspark_file.c_str(), "wb");
            const int32_t hdr[2] = {int32_t(next_tok.size()), int32_t(model.config().hidden)};
            std::fwrite(hdr, 4, 2, f);
            std::fwrite(toks.data(), 4, next_tok.size(), f);
            std::fwrite(next_tok.data(), 4, next_tok.size(), f);
            std::fwrite(model.aux().data(), 2, model.aux().size(), f);
            std::fclose(f);
            std::printf("dspark dump: %zu positions (%zu prompt) -> %s\n", next_tok.size(), ids.size(), dspark_file.c_str());
        }
        if (long_check) {
            // prefill long_check tokens (the prompt repeated), then 32 greedy decode steps on the Synapse path and the
            // same tokens (teacher forced) on the Direct path from the same state: per-step logits KL / top-1
            model.reset();
            std::vector<int32_t> p;
            while (p.size() < long_check) p.insert(p.end(), ids.begin(), ids.end());
            p.resize(long_check);
            model.set_direct(false);
            auto lg = model.forward(p, true);
            DeviceBuffer snap(dev, model.state_bytes(model.position()));
            const uint32_t p0 = model.position();
            model.save_state(snap, 0);
            stream.synchronize();
            std::vector<int32_t> toks;
            std::vector<std::vector<float>> ref;
            int32_t t = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
            const auto ts0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 32; ++i) {
                toks.push_back(t);
                ref.push_back(model.forward(std::span<const int32_t>(&t, 1), true));
                t = int32_t(std::max_element(ref.back().begin(), ref.back().end()) - ref.back().begin());
            }
            const double ms_syn = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts0).count();
            model.load_state(snap, 0, p0);
            stream.synchronize();
            model.set_direct(true);
            double kl = 0, klmax = 0;
            int same = 0;
            std::vector<std::vector<float>> got_all;
            const auto td0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 32; ++i) got_all.push_back(model.forward(std::span<const int32_t>(&toks[i], 1), true));
            const double ms_dir = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
            for (int i = 0; i < 32; ++i) {
                const auto& got = got_all[i];
                auto c = compare(ref[i].data(), got);
                kl += c.kl;
                klmax = std::max(klmax, double(c.kl));
                same += c.top1;
            }
            std::printf("long check at %u: Direct vs Synapse over 32 steps: top1 %d/32, mean KL %.5f, max KL %.5f; "
                        "%.2f vs %.2f ms/token (incl. first plans / recipes)\n", p0, same, kl / 32, klmax, ms_dir / 32, ms_syn / 32);
        }
        if (tbench) {
            std::vector<int32_t> tt(tbench, dec_ids.back());
            for (int i = 0; i < 3; ++i) model.forward(tt, true);  // compiles
            model.print_profile();
            const auto b0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 16; ++i) model.forward(tt, true);
            const double bms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count() / 16;
            std::printf("forward T=%u: %.2f ms (%.2f ms/token)\n", tbench, bms, bms / tbench);
            model.print_profile();
        }
        for (const uint32_t pbench : pbenches) {  // prefill throughput: pbench tokens (the prompt repeated), twice (the first run compiles)
            std::vector<int32_t> p;
            // G2E_TEST_CYCLE=N: the prompt is a cycle of the first N prompt tokens (very skewed expert routing); the
            // bench then prints 8 greedy tokens after the prefill (for comparisons between configurations)
            const char* cyc = std::getenv("G2E_TEST_CYCLE");
            const std::vector<int32_t> src = cyc ? std::vector<int32_t>(ids.begin() + 1, ids.begin() + 1 + std::stoul(cyc)) : ids;
            while (p.size() < pbench) p.insert(p.end(), src.begin(), src.end());
            p.resize(pbench);
            for (int run = 0; run < 2; ++run) {
                model.reset();
                const auto p0 = std::chrono::steady_clock::now();
                model.forward(p, true);
                const double pms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
                std::printf("prefill bench %u tokens: %.0f ms (%.0f tok/s)%s\n", pbench, pms, 1000.0 * pbench / pms, run ? "" : " [compile run]");
            }
            if (cyc || std::getenv("G2E_TEST_GREEDY")) {  // greedy tokens after the prefill (comparisons between configurations)
                int32_t tok = p.back();
                std::printf("greedy after %u:", pbench);
                for (int i = 0; i < 16; ++i) { tok = model.forward_argmax(std::span<const int32_t>(&tok, 1)); std::printf(" %d", tok); }
                std::printf("\n");
            }
            if (pdecode) {
                const uint32_t pos0 = model.position();
                int32_t tok = ids.back();
                for (int i = 0; i < 4; ++i) tok = model.forward_argmax(std::span<const int32_t>(&tok, 1));  // compiles
                model.print_profile();
                const auto d0 = std::chrono::steady_clock::now();
                for (uint32_t i = 0; i < pdecode; ++i) tok = model.forward_argmax(std::span<const int32_t>(&tok, 1));
                const double dms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
                std::printf("decode at %u (%.1fK): %.2f ms/token (%.1f tok/s)\n", pos0, double(pos0) / 1024.0,
                            dms / pdecode, 1000.0 * pdecode / dms);
                model.print_profile();
            }
        }
        if (ctx1m_test) {
            // Test real 1M-context Direct prefill (patched SRAM FlashAttention) & Direct decode (Tiled MME FlashDecode):
            // jump to pos = 135168 (>131K) and pos = 1044480 (~1M), run 2048-token pair prefill + 16 decode tokens each.
            std::vector<int32_t> chunk2k;
            while (chunk2k.size() < 2048) chunk2k.insert(chunk2k.end(), ids.begin(), ids.end());
            chunk2k.resize(2048);
            const std::vector<uint32_t> targets = {135168u, 1044480u};
            for (uint32_t target : targets) {
                if (target + 2048 + 32 > model.max_context()) continue;
                model.reset();
                model.set_position_for_test(target);
                const auto p0 = std::chrono::steady_clock::now();
                auto lg1 = model.forward(chunk2k, true);
                const double pms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
                bool ok_finite = !lg1.empty();
                for (float v : lg1) if (!std::isfinite(v)) { ok_finite = false; break; }
                int32_t tok = int32_t(std::max_element(lg1.begin(), lg1.end()) - lg1.begin());
                if (n_decode == 0) {
                    std::printf("ctx-1m-test at pos=%u: 2K prefill %.1f ms (%.0f tok/s, finite=%d), top1=%d\n",
                                target, pms, 2048000.0 / pms, int(ok_finite), tok);
                    if (!ok_finite) throw std::runtime_error("ctx-1m-test: non-finite logits!");
                    continue;
                }
                // Warmup 2 decode steps, then time 16 decode steps
                for (int i = 0; i < 2; ++i) tok = model.forward_argmax(std::span<const int32_t>(&tok, 1));
                const auto d0 = std::chrono::steady_clock::now();
                std::vector<int32_t> gen;
                for (int i = 0; i < 16; ++i) {
                    tok = model.forward_argmax(std::span<const int32_t>(&tok, 1));
                    gen.push_back(tok);
                }
                const double dms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
                std::printf("ctx-1m-test at pos=%u: 2K prefill %.1f ms (%.0f tok/s, finite=%d), 16-tok decode %.2f ms/tok (%.1f tok/s), first4=%d %d %d %d\n",
                            target, pms, 2048000.0 / pms, int(ok_finite), dms / 16.0, 16000.0 / dms,
                            gen[0], gen[1], gen[2], gen[3]);
                if (!ok_finite) throw std::runtime_error("ctx-1m-test: non-finite logits!");
            }
        }
        if (batch_check) {
            // Stage F2 verification: mixed short (<1024) + long (>=1024 far keys) contexts across 8 distinct slots,
            // verifying exact parity between decode_batch(B=2, 3, 4, 8) and sequential B=1 decode, plus throughput bench!
            while (model.slots() < 8) model.add_slot(8192);
            const uint32_t lens[8] = {128, 300, 600, 1024, 1500, 2048, 3000, 4096};
            std::vector<int32_t> long_p;
            while (long_p.size() < 4096) long_p.insert(long_p.end(), ids.begin(), ids.end());
            std::vector<int32_t> init_tok(8);
            std::vector<DeviceBuffer> snaps;
            snaps.reserve(8);
            for (uint32_t s = 0; s < 8; ++s) {
                model.select_slot(s);
                model.reset();
                std::vector<int32_t> p(long_p.begin(), long_p.begin() + lens[s]);
                // Make each slot's prompt slightly distinct so KV states & routing truly diverge
                p[0] = (p[0] + int32_t(s * 13 + 7)) % int32_t(V);
                p.back() = (p.back() + int32_t(s * 29 + 3)) % int32_t(V);
                auto l0 = run_prefill(p, true);
                init_tok[s] = int32_t(std::max_element(l0.begin(), l0.end()) - l0.begin());
                snaps.emplace_back(dev, model.state_bytes(model.position()));
                model.save_state(snaps.back(), 0);
            }
            stream.synchronize();
            // Reference sequential B=1 decode for 16 steps on all 8 slots (enough steps for a per-slot median: a context
            // can sit at a routing knife-edge for several consecutive steps)
            constexpr int kSteps = 16;
            std::vector<std::vector<std::vector<float>>> ref_logits(8, std::vector<std::vector<float>>(kSteps));
            std::vector<std::vector<int32_t>> ref_toks(8, std::vector<int32_t>(kSteps));
            for (uint32_t s = 0; s < 8; ++s) {
                model.select_slot(s);
                int32_t t = init_tok[s];
                for (int step = 0; step < kSteps; ++step) {
                    ref_logits[s][step] = model.forward(std::span<const int32_t>(&t, 1), true);
                    t = int32_t(std::max_element(ref_logits[s][step].begin(), ref_logits[s][step].end()) - ref_logits[s][step].begin());
                    ref_toks[s][step] = t;
                }
            }
            // Test B = 2, 3, 4, 5, 6, 7, 8 (aligned + asymmetric padded) batched decode against B = 1 reference
            for (uint32_t B : {2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
                // Choose a mix of short (<1024) and far (>=1024) slots for every B!
                std::vector<uint32_t> slots;
                if (B == 2) slots = {0, 5};
                else if (B == 3) slots = {1, 3, 6};
                else if (B == 4) slots = {0, 2, 4, 7};
                else if (B == 5) slots = {0, 1, 3, 5, 7};
                else if (B == 6) slots = {0, 1, 2, 4, 6, 7};
                else if (B == 7) slots = {0, 1, 2, 3, 4, 5, 7};
                else slots = {0, 1, 2, 3, 4, 5, 6, 7};
                for (uint32_t s : slots) {
                    model.select_slot(s);
                    model.load_state(snaps[s], 0, lens[s]);
                }
                stream.synchronize();
                std::vector<int32_t> cur(B);
                for (uint32_t b = 0; b < B; ++b) cur[b] = init_tok[slots[b]];
                std::vector<uint8_t> want(B, 1);
                int top1_ok = 0, arg_ok = 0;
                double kl_sum = 0, kl_max = 0;
                std::vector<std::vector<double>> kl_rows(B);  // per batch row, the KL of every step
                for (int step = 0; step < kSteps; ++step) {
                    auto out = model.decode_batch(slots, cur, want);
                    for (uint32_t b = 0; b < B; ++b) {
                        const uint32_t s = slots[b];
                        const auto cmp = compare(ref_logits[s][step].data(), out.logits[b]);
                        top1_ok += cmp.top1;
                        arg_ok += (out.argmax[b] == cmp.top);
                        kl_sum += cmp.kl;
                        kl_max = std::max(kl_max, cmp.kl);
                        kl_rows[b].push_back(cmp.kl);
                        if (!cmp.top1 || cmp.kl > 0.005) {
                            const float r_top_v = ref_logits[s][step][cmp.ref_top];
                            const float r_got_v = ref_logits[s][step][cmp.top];
                            std::printf("  [diff] B=%u b=%u slot=%u(pos=%u) step=%d: kl=%.6f maxd=%.4f ref=%d(%.4f) got=%d(%.4f) margin=%.4f\n",
                                        B, b, s, lens[s] + step, step, cmp.kl, cmp.maxdiff, cmp.ref_top, r_top_v, cmp.top, r_got_v, r_top_v - r_got_v);
                        }
                        cur[b] = ref_toks[s][step];  // feed ref token to measure pure per-step accuracy
                    }
                }
                const int tot = int(B) * kSteps;
                // Pass criteria: the on-device argmax is the argmax of the row's logits, and every slot's median KL against
                // B=1 over the steps is small. A wrong slot, row or padding mapping breaks every step of a slot; an isolated
                // large KL is a routing knife-edge (an expert at the top-8 cut whose choice flips with BF16-level rounding:
                // Direct B=1 and the Synapse graphs differ that way too, about 1 row in 10 on odd prompts), not a failure.
                double worst_median = 0;
                for (uint32_t b = 0; b < B; ++b) {
                    std::vector<double> v(kl_rows[b]);
                    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                    worst_median = std::max(worst_median, v[v.size() / 2]);
                }
                std::printf("batch-check B=%u (mixed short+far): top1 %d/%d, argmax-match %d/%d, mean KL %.6f, max KL %.6f, worst slot median KL %.6f\n",
                            B, top1_ok, tot, arg_ok, tot, kl_sum / tot, kl_max, worst_median);
                if (arg_ok != tot || worst_median > 0.01)
                    throw std::runtime_error("batch-check accuracy failure!");
            }
            // Throughput benchmark: B = 1, 2, 3, 4, 8 (greedy argmax-only AND sampled with logits, 32 steps after 4 warmup)
            for (uint32_t B : {1u, 2u, 3u, 4u, 8u}) {
                std::vector<uint32_t> slots(B);
                std::iota(slots.begin(), slots.end(), 0u);
                for (uint32_t s : slots) {
                    model.select_slot(s);
                    model.load_state(snaps[s], 0, lens[s]);
                }
                stream.synchronize();
                std::vector<int32_t> cur(B);
                for (uint32_t b = 0; b < B; ++b) cur[b] = init_tok[slots[b]];
                std::vector<uint8_t> want0(B, 0), want1(B, 1);
                for (int want_lg = 0; want_lg <= 1; ++want_lg) {
                    for (int w = 0; w < 4; ++w) {
                        if (B == 1) {
                            if (want_lg) {
                                auto l = model.forward(std::span<const int32_t>(&cur[0], 1), true);
                                cur[0] = int32_t(std::max_element(l.begin(), l.end()) - l.begin());
                            } else {
                                cur[0] = model.forward_argmax(std::span<const int32_t>(&cur[0], 1));
                            }
                        } else {
                            auto out = model.decode_batch(slots, cur, want_lg ? want1 : want0);
                            cur = out.argmax;
                        }
                    }
                    const auto b0 = std::chrono::steady_clock::now();
                    constexpr int kBench = 32;
                    for (int i = 0; i < kBench; ++i) {
                        if (B == 1) {
                            if (want_lg) {
                                auto l = model.forward(std::span<const int32_t>(&cur[0], 1), true);
                                cur[0] = int32_t(std::max_element(l.begin(), l.end()) - l.begin());
                            } else {
                                cur[0] = model.forward_argmax(std::span<const int32_t>(&cur[0], 1));
                            }
                        } else {
                            auto out = model.decode_batch(slots, cur, want_lg ? want1 : want0);
                            cur = out.argmax;
                        }
                    }
                    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count();
                    const double step_ms = ms / kBench;
                    std::printf("batch-bench B=%u (%s): %.2f ms/step -> %.1f agg tok/s (%.1f tok/s/user)\n",
                                B, want_lg ? "sampled" : "greedy ", step_ms, 1000.0 * B / step_ms, 1000.0 / step_ms);
                }
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
