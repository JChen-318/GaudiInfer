// Inkling-2 Direct-engine tools.
//   ink_direct_tool capture <out_dir> [filter]   MME descriptor templates of every Inkling GEMM shape: one-node
//       Synapse graphs compiled with ENABLE_EXPERIMENTAL_FLAGS=true PRINT_MME_DESCRIPTORS=1 (Synapse writes the
//       dumps to $HABANA_LOGS/mme_debug; they are moved to <out_dir>/<name>/mme_debug, the layout
//       MmeTemplate::load_dir reads). Run with G2E_RECIPE_CACHE unset (the graphs must compile).
//   ink_direct_tool gemmtest <desc_dir> <name>    one captured GEMM through the Direct runtime vs the Synapse recipe
//       (needs SCAL_CFG_FILE_PATH / SCAL_CONFIG_FILE_PATH = tools/scal-g2e-ink.json)
#include "g2e/direct_mme.hpp"
#include "g2e/graph.hpp"
#include "g2e/numeric.hpp"
#include "g2e/ops.hpp"
#include "g2e/weights.hpp"

#include <perf_lib_layer_params.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace g2e;
namespace fs = std::filesystem;

// the runtime options of the Inkling-2 engine (InkDirect)
static constexpr DirectRuntime::Options kInkOptions{.spin_wait = true, .mme_diff = true, .pdma_patches = true};

struct GemmSpec {
    std::string name;
    uint64_t B, M, K, N;   // batch (1: plain GEMM)
    DType a, b, out;
    bool tb;               // B stored [N, K] (x W^T) instead of [K, N]
    bool ta = false;       // A stored [K, M]
};

static fs::path dump_dir() {
    const char* l = std::getenv("HABANA_LOGS");
    const char* h = std::getenv("HOME");
    return fs::path(l && *l ? l : (std::string(h ? h : ".") + "/.habana_logs")) / "mme_debug";
}

static std::unique_ptr<Recipe> build(Device& dev, const GemmSpec& s, const std::string& key) {
    GraphBuilder g(dev);
    TensorId a, b, c;
    if (s.B == 1) {
        a = s.ta ? g.input("a", s.a, {s.K, s.M}) : g.input("a", s.a, {s.M, s.K});
        b = s.tb ? g.input("b", s.b, {s.N, s.K}) : g.input("b", s.b, {s.K, s.N});
        c = g.input("c", s.out, {s.M, s.N});
        g.gemm(a, b, c, s.ta, s.tb);
    } else {
        a = g.input("a", s.a, {s.B, s.M, s.K});
        b = s.tb ? g.input("b", s.b, {s.B, s.N, s.K}) : g.input("b", s.b, {s.B, s.K, s.N});
        c = g.input("c", s.out, {s.B, s.M, s.N});
        g.batch_gemm(a, b, c, false, s.tb);
    }
    return g.compile(key);
}

static void capture(Device& dev, const std::string& root, const GemmSpec& s) {
    const fs::path dir = fs::path(root) / s.name / "mme_debug";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path dd = dump_dir();
    if (fs::exists(dd)) for (auto& e : fs::directory_iterator(dd)) fs::remove(e.path());
    build(dev, s, "inkcap_" + s.name);
    int n = 0;
    if (fs::exists(dd))
        for (auto& e : fs::directory_iterator(dd)) {
            fs::copy_file(e.path(), dir / e.path().filename(), fs::copy_options::overwrite_existing);
            fs::remove(e.path());
            ++n;
        }
    std::printf("%-28s B=%llu M=%llu K=%llu N=%llu: %d descriptor dumps\n", s.name.c_str(), (unsigned long long)s.B,
                (unsigned long long)s.M, (unsigned long long)s.K, (unsigned long long)s.N, n);
}

static std::vector<GemmSpec> specs() {
    const auto BF = DType::BF16;
    std::vector<GemmSpec> v = {
        // decode (T = 1), sliding layers without messages
        {"d_proj_s", 1, 1, 2304, 8600, BF, BF, BF, true},          // x . wx^T (q | gate | rel | sink | k | v)
        {"d_wo", 1, 1, 3072, 2304, BF, BF, BF, true},               // gated attention . wo^T
        {"d_router", 1, 1, 2304, 128, BF, BF, DType::F32, true},    // x . router^T
        {"d_lmhead", 1, 1, 2304, 200058, BF, BF, DType::F32, true}, // final xn . lm_head^T -> raw F32 logits
        {"d_qk16", 8, 3, 128, 1024, BF, BF, DType::F16, true},      // FP16 scores (the 128-lane softmax kernel)
        {"d_bias16", 1, 24, 16, 1024, BF, BF, DType::F16, true},    // FP16 bias / scale table
        {"d_pvb", 8, 3, 1024, 128, BF, BF, BF, false},              // BF16 P . V (the gate kernel's split order)
        // decode, message applications (separate q-side / k / v inputs) and full layers (k / v with dynamic taps)
        {"d_mgd3", 3, 1, 4608, 64, BF, BF, BF, true},              // message gates q, k, v: cat . mg_down^T
        {"d_mgd1", 1, 1, 4608, 64, BF, BF, BF, true},              // MLP gate
        {"d_mgf1", 1, 1, 64, 4608, BF, BF, DType::F32, true},
        // three inputs at once (rows fq, fk, fv / the three gates' r): one weight read, the off-diagonal blocks unused
        {"d_proj3s", 1, 3, 2304, 8600, BF, BF, BF, true},         // [fq; fk; fv] . wx^T (sliding)
        {"d_proj3f", 1, 3, 2304, 8664, BF, BF, BF, true},         // full layers
        {"d_mgf3x", 1, 3, 64, 13824, BF, BF, DType::F32, true},   // [r_q; r_k; r_v] . [fg_q; fg_k; fg_v]^T
        // batched continuous-batching decode (Bp = 2, 4, 8): b8_* (b2/b4 reuse v2_*/v4_*) + b{2,4,8}_lmhead
        {"b8_proj", 1, 8, 2304, 8600, BF, BF, BF, true},
        {"b8_proj3s", 1, 24, 2304, 8600, BF, BF, BF, true},
        {"b8_proj3f", 1, 24, 2304, 8664, BF, BF, BF, true},
        {"b8_wo", 1, 8, 3072, 2304, BF, BF, BF, true},
        {"b8_router", 1, 8, 2304, 128, BF, BF, DType::F32, true},
        {"b8_bias", 1, 192, 16, 1024, BF, BF, DType::F16, true},
        {"b8_mgd3", 3, 8, 4608, 64, BF, BF, BF, true},
        {"b8_mgf3x", 1, 24, 64, 13824, BF, BF, DType::F32, true},
        {"b8_mgd1", 1, 8, 4608, 64, BF, BF, BF, true},
        {"b8_mgf1", 1, 8, 64, 4608, BF, BF, DType::F32, true},
        {"b2_lmhead", 1, 2, 2304, 200058, BF, BF, DType::F32, true},
        {"b4_lmhead", 1, 4, 2304, 200058, BF, BF, DType::F32, true},
        {"b8_lmhead", 1, 8, 2304, 200058, BF, BF, DType::F32, true},
    };
    // speculative verification, T = 2, 3, 4 rows (M = T, 3T per KV head, 24T for the bias table): v<T>_*
    for (uint64_t T = 2; T <= 4; ++T) {
        const std::string p = "v" + std::to_string(T) + "_";
        for (const auto& x : std::vector<GemmSpec>{
                 {p + "proj", 1, T, 2304, 8600, BF, BF, BF, true},          // [T, H] . wx^T
                 {p + "proj3s", 1, 3 * T, 2304, 8600, BF, BF, BF, true},    // [3T, H]: q-, k-, v-input rows
                 {p + "proj3f", 1, 3 * T, 2304, 8664, BF, BF, BF, true},
                 {p + "wo", 1, T, 3072, 2304, BF, BF, BF, true},
                 {p + "router", 1, T, 2304, 128, BF, BF, DType::F32, true},
                 {p + "bias", 1, 24 * T, 16, 1024, BF, BF, DType::F16, true},   // r rows (t, h) . projR^T
                 {p + "qk", 8, 3 * T, 128, 1024, BF, BF, DType::F16, true},     // per KV head: q rows (g, t) . K_ring^T
                 {p + "qkn", 8, 3 * T, 128, T, BF, BF, DType::F16, true},       // . K_new^T (the block's rows)
                 {p + "pv", 8, 3 * T, 1024, 128, BF, BF, BF, false},
                 {p + "pvn", 8, 3 * T, T, 128, BF, BF, BF, false},
                 {p + "mgd3", 3, T, 4608, 64, BF, BF, BF, true},
                 {p + "mgf3x", 1, 3 * T, 64, 13824, BF, BF, DType::F32, true},
                 {p + "mgd1", 1, T, 4608, 64, BF, BF, BF, true},
                 {p + "mgf1", 1, T, 64, 4608, BF, BF, DType::F32, true},
             })
            v.push_back(x);
    }
    // prefill chunks (T = 1024): p_*. Batched templates (B > 1) capture one activation of several batches; the plan
    // issues it B / batches times with offset addresses (see batches_of()).
    {
        const uint64_t T = 1024;
        for (const auto& x : std::vector<GemmSpec>{
                 {"p_router", 1, T, 2304, 128, BF, BF, DType::F32, true},
                 {"p_mgd1", 1, T, 4608, 64, BF, BF, BF, true},               // message gate: cat . mg_down^T
                 // BF16 MoE tiles for native expert layers (F32 output)
                 {"p_bgt16",  1,  16, 2304, 2304, BF, BF, DType::F32, false},
                 {"p_bgt32",  1,  32, 2304, 2304, BF, BF, DType::F32, false},
                 {"p_bgt64",  1,  64, 2304, 2304, BF, BF, DType::F32, false},
                 {"p_bgt96",  1,  96, 2304, 2304, BF, BF, DType::F32, false},
                 {"p_bgt128", 1, 128, 2304, 2304, BF, BF, DType::F32, false},
                 {"p_bgt192", 1, 192, 2304, 2304, BF, BF, DType::F32, false},
                 {"p_bgt256", 1, 256, 2304, 2304, BF, BF, DType::F32, false},
                 {"p_bdt16",  1,  16, 1152, 2304, BF, BF, DType::F32, false},
                 {"p_bdt32",  1,  32, 1152, 2304, BF, BF, DType::F32, false},
                 {"p_bdt64",  1,  64, 1152, 2304, BF, BF, DType::F32, false},
                 {"p_bdt96",  1,  96, 1152, 2304, BF, BF, DType::F32, false},
                 {"p_bdt128", 1, 128, 1152, 2304, BF, BF, DType::F32, false},
                 {"p_bdt192", 1, 192, 1152, 2304, BF, BF, DType::F32, false},
                 {"p_bdt256", 1, 256, 1152, 2304, BF, BF, DType::F32, false},
                 // 512-row tiles of the large prefill GEMMs (whole-M captures are sliced by Synapse)
                 {"p_proj_s_m512", 1, 512, 2304, 8600, BF, BF, BF, true},
                 {"p_projq_m512", 1, 512, 2304, 6552, BF, BF, BF, true},
                 {"p_projkv_m512", 1, 512, 2304, 1024, BF, BF, BF, true},
                 {"p_projkvf_m512", 1, 512, 2304, 1056, BF, BF, BF, true},
                 {"p_qk_m512", 1, 512, 128, 1024, BF, BF, DType::F16, true},
                 {"p_wo_m512", 1, 512, 3072, 2304, BF, BF, BF, true},
                 {"p_mgf1_m512", 1, 512, 64, 4608, BF, BF, DType::F32, true},
                 {"p_bias_r512", 1, 512, 16, 1024, BF, BF, DType::F16, true},
                 {"p_pv2k", 1, 3 * T, 2048, 128, BF, BF, DType::F32, false},  // P [GT, ring | new] . V
             })
            v.push_back(x);
        // full layers' far keys in prefill chunks: 512 query rows against the cache rows [0, Sf) (FP16 scores, F32 P . V)
        for (uint32_t sf : {1024u, 2048u, 3072u, 4096u, 6144u, 8192u, 9216u, 13824u, 16384u, 20736u, 31104u, 32768u, 46656u, 65536u,
                            69984u, 104976u, 131072u}) {
            v.push_back({"p_fqk_" + std::to_string(sf) + "_m512", 1, 512, 128, sf, BF, BF, DType::F16, true});
            v.push_back({"p_fpv_" + std::to_string(sf) + "_m512", 1, 512, sf, 128, BF, BF, DType::F32, false});
        }
    }
    // full layers' far keys (linear cache rows [0, n), n bucketed): per KV head q . K^T (FP16) and P . V (BF16)
    for (uint32_t sf : {1024u, 2048u, 3072u, 4096u, 6144u, 8192u, 9216u, 13824u, 16384u, 20736u, 31104u, 32768u, 46656u, 65536u,
                        69984u, 104976u, 131072u}) {
        v.push_back({"d_fqk_" + std::to_string(sf), 1, 3, 128, sf, BF, BF, DType::F16, true});
        v.push_back({"d_fpv_" + std::to_string(sf), 1, 3, sf, 128, BF, BF, BF, false});
        if (sf <= 65536u) {
            for (uint32_t T = 2; T <= 4; ++T) {
                v.push_back({"v" + std::to_string(T) + "_fqk_" + std::to_string(sf), 1, 3 * T, 128, sf, BF, BF, DType::F16, true});
                v.push_back({"v" + std::to_string(T) + "_fpv_" + std::to_string(sf), 1, 3 * T, sf, 128, BF, BF, BF, false});
            }
        }
    }
    return v;
}

// batches covered by one activation of a captured template: tensorB validElements[2] / (K * N)
static uint64_t batches_of(const std::string& dir, uint64_t kn) {
    for (auto& e : fs::directory_iterator(dir)) {
        std::ifstream f(e.path());
        std::string line;
        bool inb = false;
        while (std::getline(f, line)) {
            if (line.find("Tensor: tensorB") != std::string::npos) inb = true;
            else if (inb && line.find("validElements") != std::string::npos) {
                uint64_t v[5] = {0, 0, 0, 0, 0};
                std::sscanf(line.c_str() + line.find('[') + 1, "%lu, %lu, %lu, %lu, %lu", &v[0], &v[1], &v[2], &v[3], &v[4]);
                return std::max<uint64_t>(1, v[2] / kn);
            }
        }
    }
    return 1;
}

static int gemmtest(const std::string& desc_dir, const std::string& name) {
    GemmSpec s{};
    bool found = false;
    for (const auto& x : specs()) if (x.name == name) { s = x; found = true; }
    if (!found) { std::fprintf(stderr, "unknown spec %s\n", name.c_str()); return 2; }
    Device dev;
    Stream st(dev);
    auto esz = [](DType d) { return d == DType::F8 ? 1u : d == DType::F32 ? 4u : 2u; };  // F16 / BF16: 2
    const uint64_t na = s.B * s.M * s.K, nb = s.B * s.K * s.N, nc = s.B * s.M * s.N;
    DeviceBuffer da(dev, na * esz(s.a)), db(dev, nb * esz(s.b)), dc(dev, nc * esz(s.out)), dr(dev, nc * esz(s.out));
    HostBuffer hb(dev, std::max({na * esz(s.a), nb * esz(s.b), nc * esz(s.out)}) + 4096);
    uint64_t seed = 12345;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return float(int((seed >> 33) % 2001) - 1000) / 1000.f; };
    auto fill = [&](DeviceBuffer& d, uint64_t n, DType t) {
        for (uint64_t i = 0; i < n; ++i) {
            if (t == DType::F8) hb.as<uint8_t>()[i] = to_fp8_143(rnd());
            else hb.as<uint16_t>()[i] = to_bf16(rnd());
        }
        st.upload(hb, 0, d, 0, n * esz(t));
        st.synchronize();
    };
    fill(da, na, s.a);
    fill(db, nb, s.b);
    // Synapse reference
    auto r = build(dev, s, "inkgt_" + s.name);
    std::vector<Binding> bd = {{"a", &da, 0}, {"b", &db, 0}, {"c", &dr, 0}};
    r->enqueue(st, bd);
    st.synchronize();
    // Direct
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const auto t = MmeTemplate::load_dir(desc_dir + "/" + name + "/mme_debug");
    DirectPlan plan(rt);
    // a batched template covers `nb` batches (one activation): issue it B / nb times with offset addresses
    const uint64_t nbt = batches_of(desc_dir + "/" + name + "/mme_debug", s.K * s.N);
    const uint64_t reps = s.B > nbt ? s.B / nbt : 1;
    for (uint64_t i = 0; i < reps; ++i)
        plan.add_mme(t, da.address() + i * nbt * s.M * s.K * esz(s.a), db.address() + i * nbt * s.K * s.N * esz(s.b),
                     dc.address() + i * nbt * s.M * s.N * esz(s.out));
    std::printf("template: %llu batches per activation, %llu issues\n", (unsigned long long)nbt, (unsigned long long)reps);
    plan.prepare();
    double us = plan.run();
    us = plan.run();
    // compare
    const uint32_t oe = esz(s.out);
    std::vector<uint8_t> c(nc * oe), ref(nc * oe);
    st.download(dc, 0, hb, 0, nc * oe); st.synchronize(); std::memcpy(c.data(), hb.data(), nc * oe);
    st.download(dr, 0, hb, 0, nc * oe); st.synchronize(); std::memcpy(ref.data(), hb.data(), nc * oe);
    uint64_t same = 0; double num = 0, den = 0;
    for (uint64_t i = 0; i < nc; ++i) {
        double x, y;
        if (oe == 4) { float a, b; std::memcpy(&a, &c[i * 4], 4); std::memcpy(&b, &ref[i * 4], 4); x = a; y = b; same += a == b; }
        else { uint16_t a, b; std::memcpy(&a, &c[i * 2], 2); std::memcpy(&b, &ref[i * 2], 2); same += a == b; x = a; y = b; if (s.out == DType::BF16) { x = from_bf16(a); y = from_bf16(b); } }
        num += (x - y) * (x - y); den += y * y;
    }
    std::printf("%s: %zu descriptors, engines 0x%x; direct %.1f us; bit-identical %.4f%%, rel diff %.2e\n", name.c_str(),
                t.descs.size(), t.engine_mask, us, 100.0 * double(same) / double(nc), std::sqrt(num / (den + 1e-30)));
    // the Synapse recipe's time for comparison
    Event e0(dev), e1(dev);
    e0.record(st);
    for (int i = 0; i < 10; ++i) r->enqueue(st, bd);
    e1.record(st);
    st.synchronize();
    std::printf("  synapse recipe: %.1f us per launch\n", 1000.0 * Event::elapsed_ms(e0, e1) / 10);
    return same == nc ? 0 : 1;
}

static int gemmtest_prefix(const std::string& desc_dir, const std::vector<std::string>& prefixes) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<GemmSpec> todo;
    for (const auto& x : specs()) {
        for (const auto& p : prefixes)
            if (x.name.rfind(p, 0) == 0) { todo.push_back(x); break; }
    }
    if (todo.empty()) { std::fprintf(stderr, "no specs matching given prefix(es)\n"); return 2; }
    auto esz = [](DType d) { return d == DType::F8 ? 1u : d == DType::F32 ? 4u : 2u; };
    uint64_t max_a = 0, max_b = 0, max_c = 0;
    for (const auto& s : todo) {
        max_a = std::max(max_a, s.B * s.M * s.K * esz(s.a));
        max_b = std::max(max_b, s.B * s.K * s.N * esz(s.b));
        max_c = std::max(max_c, s.B * s.M * s.N * esz(s.out));
    }
    Device dev;
    Stream st(dev);
    DeviceBuffer da(dev, max_a + 4096), db(dev, max_b + 4096), dc(dev, max_c + 4096), dr(dev, max_c + 4096);
    HostBuffer hb(dev, std::max({max_a, max_b, max_c}) + 4096);
    uint64_t seed = 12345;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return float(int((seed >> 33) % 2001) - 1000) / 1000.f; };
    auto fill = [&](DeviceBuffer& d, uint64_t n, DType t) {
        for (uint64_t i = 0; i < n; ++i) {
            if (t == DType::F8) hb.as<uint8_t>()[i] = to_fp8_143(rnd());
            else hb.as<uint16_t>()[i] = to_bf16(rnd());
        }
        st.upload(hb, 0, d, 0, n * esz(t));
        st.synchronize();
    };
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    int failed = 0;
    for (const auto& s : todo) {
        const uint64_t na = s.B * s.M * s.K, nb = s.B * s.K * s.N, nc = s.B * s.M * s.N;
        fill(da, na, s.a);
        fill(db, nb, s.b);
        auto r = build(dev, s, "inkgt_" + s.name);
        std::vector<Binding> bd = {{"a", &da, 0}, {"b", &db, 0}, {"c", &dr, 0}};
        r->enqueue(st, bd);
        st.synchronize();
        const auto t = MmeTemplate::load_dir(desc_dir + "/" + s.name + "/mme_debug");
        DirectPlan plan(rt);
        const uint64_t nbt = batches_of(desc_dir + "/" + s.name + "/mme_debug", s.K * s.N);
        const uint64_t reps = s.B > nbt ? s.B / nbt : 1;
        for (uint64_t i = 0; i < reps; ++i)
            plan.add_mme(t, da.address() + i * nbt * s.M * s.K * esz(s.a), db.address() + i * nbt * s.K * s.N * esz(s.b),
                         dc.address() + i * nbt * s.M * s.N * esz(s.out));
        plan.prepare();
        double us = plan.run();
        us = plan.run();
        const uint32_t oe = esz(s.out);
        std::vector<uint8_t> c(nc * oe), ref(nc * oe);
        st.download(dc, 0, hb, 0, nc * oe); st.synchronize(); std::memcpy(c.data(), hb.data(), nc * oe);
        st.download(dr, 0, hb, 0, nc * oe); st.synchronize(); std::memcpy(ref.data(), hb.data(), nc * oe);
        uint64_t same = 0; double num = 0, den = 0;
        for (uint64_t i = 0; i < nc; ++i) {
            double x, y;
            if (oe == 4) {
                float a, b; std::memcpy(&a, &c[i * 4], 4); std::memcpy(&b, &ref[i * 4], 4);
                x = a; y = b; same += a == b;
            } else if (oe == 1) {
                same += c[i] == ref[i];
                x = from_fp8_143(c[i]); y = from_fp8_143(ref[i]);
            } else {
                uint16_t a, b; std::memcpy(&a, &c[i * 2], 2); std::memcpy(&b, &ref[i * 2], 2);
                same += a == b;
                x = (s.out == DType::BF16) ? from_bf16(a) : double(a);
                y = (s.out == DType::BF16) ? from_bf16(b) : double(b);
            }
            num += (x - y) * (x - y); den += y * y;
        }
        const bool ok = (same == nc);
        if (!ok) ++failed;
        std::printf("%-16s %s: %2zu descs, eng 0x%x; direct %6.1f us; bit-id %8.4f%% (%llu/%llu), rel %.2e\n",
                    s.name.c_str(), ok ? "PASS" : "FAIL", t.descs.size(), t.engine_mask, us,
                    100.0 * double(same) / double(nc), (unsigned long long)same, (unsigned long long)nc,
                    std::sqrt(num / (den + 1e-30)));
    }
    std::printf("gemmtest_prefix: %zu/%zu passed\n", todo.size() - size_t(failed), todo.size());
    return failed ? 1 : 0;
}

// one ink_* TPC kernel through the Direct runtime vs its Synapse launch: ink_mg_gaudi2 (CAT) at T = 16

static void synapse_warmup(Device& dev, Stream& st);
// mixbench OBJDIR KERNEL S0 Q FLAGS: time a depth-mix kernel (ink_mix layout) over T = 1024 rows on the Direct engine
static int mixbench(const std::string& objdir, const std::string& kernel, uint32_t S0, uint32_t Q, int flags) {
    Device dev;
    Stream st(dev);
    const uint64_t H = 2304, T = 1024;
    DeviceBuffer db(dev, H * T * S0 * 2 + 4096), dp(dev, H * T * 4), dq(dev, H * Q * 4), dw(dev, H * Q * 2),
        dn(dev, H * Q * T * 2), dx(dev, H * Q * T * 4);
    HostBuffer hb(dev, H * T * 4 + 4096);
    for (uint64_t i = 0; i < H * T * 2; ++i) hb.as<uint16_t>()[i] = 0x3f80 + uint16_t(i % 37);
    st.upload(hb, 0, db, 0, std::min<uint64_t>(H * T * S0 * 2, H * T * 4)); st.upload(hb, 0, dp, 0, H * T * 4);
    st.upload(hb, 0, dq, 0, H * Q * 4); st.upload(hb, 0, dw, 0, H * Q * 2); st.synchronize();
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/" + kernel + ".o");
    const bool f32p = kernel.find("f32p") != std::string::npos;
    DirectTpcTensor tb{db.address(), DType::BF16, 3, {uint32_t(H), uint32_t(T), S0, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tp{dp.address(), f32p ? DType::F32 : DType::BF16, 2, {uint32_t(H), uint32_t(T), 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tq{dq.address(), DType::F32, 2, {uint32_t(H), Q, 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tw{dw.address(), DType::BF16, 2, {uint32_t(H), Q, 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tn{dn.address(), DType::BF16, 3, {uint32_t(H), Q, uint32_t(T), 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tx{dx.address(), DType::F32, 3, {uint32_t(H), Q, uint32_t(T), 1, 1}, {0, 0, 0, 0, 0}, 0};
    const DirectTpcTensor ts[] = {tb, tp, tq, tw, tn, tx};
    struct { float scale, eps; int32_t flags; } prm{1.f / 48.f, 1e-6f, flags};
    uint32_t srfs[3];
    std::memcpy(srfs, &prm, sizeof(prm));
    DirectPlan plan(rt);
    plan.add_tpc_grid(isa, ts, srfs, {24, 1, 1, 1, 1}, 0);
    plan.prepare();
    double us = plan.run();
    double best = 1e30;
    for (int i = 0; i < 5; ++i) best = std::min(best, plan.run());
    (void)us;
    std::printf("%s S0=%u Q=%u flags=%d: %.1f us\n", kernel.c_str(), S0, Q, flags, best);
    return 0;
}


// tilebench DESC_DIR NAME N: N issues of one captured GEMM in one plan (the same operands): the time per issue
static int tilebench(const std::string& desc_dir, const std::string& name, uint32_t N, uint32_t chunk_limit = 9) {
    GemmSpec s{};
    bool found = false;
    for (const auto& x : specs()) if (x.name == name) { s = x; found = true; }
    if (!found) { std::fprintf(stderr, "unknown spec %s\n", name.c_str()); return 2; }
    Device dev;
    Stream st(dev);
    auto esz = [](DType d) { return d == DType::F8 ? 1u : d == DType::F32 ? 4u : 2u; };
    const uint64_t na = s.B * s.M * s.K, nb = s.B * s.K * s.N, nc = s.B * s.M * s.N;
    DeviceBuffer da(dev, na * esz(s.a) + 4096), db(dev, nb * esz(s.b) + 4096), dc(dev, nc * esz(s.out) + 4096);
    st.zero(da, 0, na * esz(s.a)); st.zero(db, 0, nb * esz(s.b)); st.synchronize();
    // (the anchor: the engines walk the static part of the recipe launched last, which must have an MME node)
    auto r = build(dev, s, "inkgt_" + s.name);
    std::vector<Binding> bd = {{"a", &da, 0}, {"b", &db, 0}, {"c", &dc, 0}};
    r->enqueue(st, bd);
    st.synchronize();
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const auto t = MmeTemplate::load_dir(desc_dir + "/" + name + "/mme_debug");
    auto run = [&](uint32_t n) {
        DirectPlan plan(rt);
        plan.set_mme_chunk_limit(chunk_limit);
        for (uint32_t i = 0; i < n; ++i) plan.add_mme(t, da.address(), db.address(), dc.address());
        plan.prepare();
        plan.run();
        double best = 1e30;
        for (int i = 0; i < 9; ++i) best = std::min(best, plan.run());
        return best;
    };
    const double t1 = run(1), tn = run(N);
    std::printf("%s (%u per chunk): %zu descriptors per issue; 1 issue %.1f us, %u issues %.1f us: %.2f us per issue (%.0f T MAC/s)\n",
                name.c_str(), chunk_limit, t.descs.size(), t1, N, tn, (tn - t1) / (N - 1),
                double(s.B * s.M * s.K * s.N) / ((tn - t1) / (N - 1)) * 1e-6);
    return 0;
}

// lanetest DESC_DIR OBJDIR LANE: a plan of the given lane (lane 2 waits on QMAN fence 2) whose stages each wait on
// the one before: copy a -> b (TPC), copy b -> c (TPC), GEMM (MME), copy c -> d (TPC). 50 runs; d must equal a.
static int lanetest(const std::string& desc_dir, const std::string& objdir, uint32_t lane) {
    GemmSpec s{};
    for (const auto& x : specs()) if (x.name == "p_bgt128") s = x;
    Device dev;
    Stream st(dev);
    const uint64_t na = s.M * s.K * 2, nb = s.K * s.N * 2, nc = s.M * s.N * 4, NB = 128 * 768 * 2;  // (24 points of 32 rows)
    DeviceBuffer da(dev, na + 4096), db(dev, nb + 4096), dc(dev, nc + 4096), ta(dev, NB), tb(dev, NB), tc(dev, NB), td(dev, NB);
    HostBuffer hb(dev, NB + 4096);
    st.zero(da, 0, na); st.zero(db, 0, nb); st.synchronize();
    auto r = build(dev, s, "inkgt_" + s.name);
    std::vector<Binding> bd = {{"a", &da, 0}, {"b", &db, 0}, {"c", &dc, 0}};
    r->enqueue(st, bd);
    st.synchronize();
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/ink_pcopy_gaudi2.o");
    const auto t = MmeTemplate::load_dir(desc_dir + "/p_bgt128/mme_debug");
    auto T = [](const DeviceBuffer& d) { return DirectTpcTensor{d.address(), DType::BF16, 3, {128, 768, 1, 1, 1}, {0, 0, 0, 0, 0}, 0}; };
    DirectPlan plan(rt);
    if (lane) plan.set_lane(lane);
    const DirectTpcTensor s1[] = {T(ta), T(tb)}, s2[] = {T(tb), T(tc)}, s3[] = {T(tc), T(td)};
    plan.add_tpc_grid(isa, s1, {}, {1, 24, 1, 1, 1}, 1, 24, -1, 0, 0);
    plan.add_tpc_grid(isa, s2, {}, {1, 24, 1, 1, 1}, 1, 24, 0, 24, 1);
    plan.add_mme(t, da.address(), db.address(), dc.address(), 0, 1, 24, 2);
    plan.add_tpc_grid(isa, s3, {}, {1, 24, 1, 1, 1}, 1, 24, 2, t.signal_count, -1);
    plan.prepare();
    int bad = 0;
    double us = 0;
    for (int it = 0; it < 50; ++it) {
        for (uint64_t i = 0; i < NB / 2; ++i) hb.as<uint16_t>()[i] = uint16_t(i * 7 + it * 131 + 1);
        st.upload(hb, 0, ta, 0, NB); st.zero(tb, 0, NB); st.zero(tc, 0, NB); st.zero(td, 0, NB); st.synchronize();
        us = plan.run();
        std::vector<uint16_t> ref(hb.as<uint16_t>(), hb.as<uint16_t>() + NB / 2);
        st.download(td, 0, hb, 0, NB); st.synchronize();
        bad += std::memcmp(ref.data(), hb.data(), NB) != 0;
    }
    std::printf("lane %u: 50 runs, %d wrong; last run %.1f us\n", lane, bad, us);
    return bad != 0;
}

// hostrt OBJDIR: the cost of a host round trip inside a prefill layer: the smallest plan (one copy point) run and
// waited for, a 512-byte download, and both in sequence (median / 10th / 90th percentile of 300)
static int hostrt(const std::string& objdir) {
    Device dev;
    Stream st(dev);
    DeviceBuffer da(dev, 1 << 20), db(dev, 1 << 20);
    HostBuffer hb(dev, 4096);
    st.zero(da, 0, 1 << 20); st.zero(db, 0, 1 << 20); st.synchronize();
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/ink_pcopy_gaudi2.o");
    const DirectTpcTensor ts[] = {{da.address(), DType::BF16, 3, {128, 32, 1, 1, 1}, {0, 0, 0, 0, 0}, 0},
                                  {db.address(), DType::BF16, 3, {128, 32, 1, 1, 1}, {0, 0, 0, 0, 0}, 0}};
    DirectPlan plan(rt);
    plan.add_tpc_grid(isa, ts, {}, {1, 1, 1, 1, 1}, 0);
    plan.prepare();
    plan.run();
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    std::vector<double> tp, td, tb;
    for (int i = 0; i < 300; ++i) {
        auto t0 = now();
        plan.run();
        auto t1 = now();
        st.download(da, 0, hb, 0, 512);
        st.synchronize();
        auto t2 = now();
        tp.push_back(us(t0, t1)); td.push_back(us(t1, t2)); tb.push_back(us(t0, t2));
    }
    auto pr = [](const char* n, std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        std::printf("%-28s median %7.1f us  p10 %7.1f  p90 %7.1f  max %8.1f\n", n, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], v.back());
    };
    pr("smallest plan run + wait", tp);
    pr("512-byte download + sync", td);
    pr("both", tb);
    HostBuffer big(dev, 1 << 20);
    for (uint64_t bytes : {4096ull, 65536ull, 262144ull, 1048576ull}) {  // uploads (plan patches, tables)
        std::vector<double> tu;
        for (int i = 0; i < 100; ++i) {
            auto t0 = now();
            st.upload(big, 0, da, 0, bytes);
            st.synchronize();
            tu.push_back(us(t0, now()));
        }
        char nm[64];
        std::snprintf(nm, sizeof nm, "%llu-byte upload + sync", (unsigned long long)bytes);
        pr(nm, tu);
    }
    return 0;
}

// attnbench OBJDIR KERNEL POS0 FULL: time the prefill attention-softmax kernel (ink_attn_sm16 layout: one KV head, G = 3,
// W = T = 1024, RE = 1024) on the Direct engine
static int attnbench(const std::string& objdir, const std::string& kernel, int pos0, int full) {
    Device dev;
    Stream st(dev);
    const uint64_t W = 1024, T = 1024, G = 3, GT = G * T, RE = 1024, NH = 24, SN = W + T;
    DeviceBuffer dsr(dev, W * GT * 2), dsn(dev, T * GT * 2), dbr(dev, RE * NH * T * 2), dsk(dev, NH * T * 4), dpos(dev, 4096),
        dp(dev, SN * GT * 2), dsum(dev, GT * 4 + 4096);
    HostBuffer hb(dev, RE * NH * T * 2 + 4096);
    for (uint64_t i = 0; i < RE * NH * T; ++i) hb.as<uint16_t>()[i] = uint16_t(0x3c00 + (i * 7919) % 2048);  // f16 1..3
    st.upload(hb, 0, dsr, 0, W * GT * 2); st.upload(hb, 0, dsn, 0, T * GT * 2); st.upload(hb, 0, dbr, 0, RE * NH * T * 2);
    for (uint64_t i = 0; i < NH * T; ++i) hb.as<float>()[i] = 0.5f;
    st.upload(hb, 0, dsk, 0, NH * T * 4);
    hb.as<int32_t>()[0] = pos0;
    st.upload(hb, 0, dpos, 0, 4); st.synchronize();
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/" + kernel + ".o");
    DirectTpcTensor tsr{dsr.address(), DType::F16, 2, {uint32_t(W), uint32_t(GT), 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tsn{dsn.address(), DType::F16, 2, {uint32_t(T), uint32_t(GT), 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tbr{dbr.address(), DType::F16, 3, {uint32_t(RE), uint32_t(NH), uint32_t(T), 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tsk{dsk.address(), DType::F32, 2, {uint32_t(NH), uint32_t(T), 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tpo{dpos.address(), DType::I32, 1, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tp{dp.address(), DType::BF16, 2, {uint32_t(SN), uint32_t(GT), 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tsu{dsum.address(), DType::F32, 2, {1, uint32_t(GT), 1, 1, 1}, {0, 0, 0, 0, 0}, 0};
    const DirectTpcTensor ts[] = {tsr, tsn, tbr, tsk, tpo, tp, tsu};
    struct { float scale; int32_t G, full, kv; } prm{1.f / std::sqrt(128.f), int32_t(G), full, 0};
    uint32_t srfs[4];
    std::memcpy(srfs, &prm, sizeof(prm));
    DirectPlan plan(rt);
    const bool p24 = std::getenv("ATT_P24") != nullptr;
    plan.add_tpc_grid(isa, ts, srfs, {p24 ? 24u : uint32_t(GT), 1, 1, 1, 1}, 0);
    plan.prepare();
    (void)plan.run();
    double best = 1e30;
    for (int i = 0; i < 5; ++i) best = std::min(best, plan.run());
    std::printf("%s pos0=%d full=%d: %.1f us\n", kernel.c_str(), pos0, full, best);
    return 0;
}


// dnormbench OBJDIR KERNEL S0 Q: time the decode depth-mix phase 2 for message applications (ink_dnorm_msg layout,
// T = 1, H = 2304) on the Direct engine
static int dnormbench(const std::string& objdir, const std::string& kernel, uint32_t S0, uint32_t Q) {
    Device dev;
    Stream st(dev);
    const uint64_t H = 2304, NT = H / 128, G = 3;
    DeviceBuffer dpa(dev, 64 * NT * 4), db(dev, H * S0 * 2 + 4096), dp(dev, H * 4), dmp(dev, 64 * NT * 4), dp1(dev, H * 2),
        dp2(dev, H * 2), dc(dev, 2 * H * G * 2), dmx(dev, H * Q * 4), dms(dev, H * 4 * 4);
    HostBuffer hb(dev, 1 << 20);
    for (uint64_t i = 0; i < 64 * NT; ++i) hb.as<float>()[i] = 0.01f * float(1 + (i % 64));
    st.upload(hb, 0, dpa, 0, 64 * NT * 4); st.upload(hb, 0, dmp, 0, 64 * NT * 4);
    for (uint64_t i = 0; i < H * S0; ++i) hb.as<uint16_t>()[i] = 0x3f80;
    st.upload(hb, 0, db, 0, H * S0 * 2); st.upload(hb, 0, dp, 0, H * 2); st.upload(hb, 0, dp1, 0, H * 2); st.upload(hb, 0, dp2, 0, H * 2);
    st.synchronize();
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/" + kernel + ".o");
    auto T3 = [](uint64_t a, DType t, uint32_t d0, uint32_t d1, uint32_t d2) {
        return DirectTpcTensor{a, t, 3, {d0, d1, d2, 1, 1}, {0, 0, 0, 0, 0}, 0};
    };
    const DirectTpcTensor ts[] = {T3(dpa.address(), DType::F32, 64, uint32_t(NT), 1), T3(db.address(), DType::BF16, uint32_t(H), 1, S0),
                                  T3(dp.address(), DType::BF16, uint32_t(H), 1, 1), T3(dmp.address(), DType::F32, 64, uint32_t(NT), 1),
                                  T3(dp1.address(), DType::BF16, uint32_t(H), 1, 1), T3(dp2.address(), DType::BF16, uint32_t(H), 1, 1),
                                  T3(dc.address(), DType::BF16, uint32_t(2 * H), 1, uint32_t(G)), T3(dmx.address(), DType::F32, uint32_t(H), 1, Q),
                                  T3(dms.address(), DType::F32, uint32_t(H), 1, 4)};
    struct { float scale, eps; int32_t m0, write_msg; } prm{1.f / 48.f, 1e-6f, 0, 1};
    uint32_t srfs[4];
    std::memcpy(srfs, &prm, sizeof(prm));
    // a chain of N copies: the per-launch cost without the plan's fixed run overhead
    const int N = 20;
    DirectPlan plan(rt);
    for (int i = 0; i < N; ++i) plan.add_tpc_grid(isa, ts, srfs, {uint32_t(NT), 1, 1, 1, 1}, 0, 24, i ? 0 : -1, i ? uint32_t(NT) : 0, i + 1 < N ? 0 : -1);
    plan.prepare();
    (void)plan.run();
    double best = 1e30;
    for (int i = 0; i < 5; ++i) best = std::min(best, plan.run());
    std::printf("%s S0=%u Q=%u: %.2f us per launch (chain of %d)\n", kernel.c_str(), S0, Q, best / N, N);
    return 0;
}

static int tpctest(const std::string& objdir) {
    Device dev;
    Stream st(dev);
    const uint64_t H = 2304, G = 3, Gh = 3, T = 16;
    const uint64_t nh = H * Gh * T, nm = H * 4 * T, nc = 2 * H * T * G;
    DeviceBuffer dh(dev, nh * 4), dm(dev, nm * 4), dc(dev, nc * 2), dr(dev, nc * 2);
    HostBuffer hb(dev, std::max(nm * 4, nc * 2) + 4096);
    uint64_t seed = 99;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return float(int((seed >> 33) % 2001) - 1000) / 250.f; };
    for (uint64_t i = 0; i < nh; ++i) hb.as<float>()[i] = rnd();
    st.upload(hb, 0, dh, 0, nh * 4); st.synchronize();
    for (uint64_t i = 0; i < nm; ++i) hb.as<float>()[i] = rnd();
    st.upload(hb, 0, dm, 0, nm * 4); st.synchronize();
    struct { float eps; int32_t m0; } prm{1e-6f, 0};
    {   // Synapse
        GraphBuilder g(dev);
        const auto hi = g.input("h", DType::F32, {T, Gh, H});
        const auto mi = g.input("m", DType::F32, {T, 4, H});
        const auto co = g.input("c", DType::BF16, {G, T, 2 * H});
        g.op("ink_mg_gaudi2", {hi, mi}, {co}, &prm, sizeof(prm));
        auto r = g.compile("inkdt_mg");
        std::vector<Binding> b = {{"h", &dh, 0}, {"m", &dm, 0}, {"c", &dr, 0}};
        r->enqueue(st, b);
        st.synchronize();
    }
    // Direct: tensors in main() order with the glue's shapes (dim 0 fastest); SRFs = the parameter struct words;
    // index space [G, T] split over the TPCs along T
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/ink_mg_gaudi2.o");
    DirectTpcTensor th{dh.address(), DType::F32, 3, {uint32_t(H), uint32_t(Gh), uint32_t(T), 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tm{dm.address(), DType::F32, 3, {uint32_t(H), 4, uint32_t(T), 1, 1}, {0, 0, 0, 0, 0}, 0};
    DirectTpcTensor tc{dc.address(), DType::BF16, 3, {uint32_t(2 * H), uint32_t(T), uint32_t(G), 1, 1}, {0, 0, 0, 0, 0}, 0};
    const DirectTpcTensor ts[] = {th, tm, tc};
    uint32_t srfs[2];
    std::memcpy(srfs, &prm, sizeof(prm));
    DirectPlan plan(rt);
    plan.add_tpc_grid(isa, ts, srfs, {uint32_t(G), uint32_t(T), 1, 1, 1}, 1);
    plan.prepare();
    double us = plan.run();
    us = plan.run();
    std::vector<uint16_t> c(nc), ref(nc);
    st.download(dc, 0, hb, 0, nc * 2); st.synchronize(); std::memcpy(c.data(), hb.data(), nc * 2);
    st.download(dr, 0, hb, 0, nc * 2); st.synchronize(); std::memcpy(ref.data(), hb.data(), nc * 2);
    uint64_t same = 0;
    for (uint64_t i = 0; i < nc; ++i) same += c[i] == ref[i];
    std::printf("ink_mg via Direct: %.1f us, bit-identical %.4f%%\n", us, 100.0 * double(same) / double(nc));
    return 0;
}

// One Synapse TPC launch: it programs the TPC special-function table base. Every kernel's prologue invalidates that
// table cache, so a Direct TPC launch in a process that never ran a Synapse TPC kernel page-faults.
static void synapse_warmup(Device& dev, Stream& st) {
    const uint64_t H = 2304, T = 1;
    DeviceBuffer a(dev, H * 3 * T * 4), b(dev, H * 4 * T * 4), c(dev, 2 * H * T * 3 * 2);
    struct { float eps; int32_t m0; } prm{1e-6f, 0};
    GraphBuilder g(dev);
    const auto hi = g.input("h", DType::F32, {T, 3, H});
    const auto mi = g.input("m", DType::F32, {T, 4, H});
    const auto co = g.input("c", DType::BF16, {3, T, 2 * H});
    g.op("ink_mg_gaudi2", {hi, mi}, {co}, &prm, sizeof(prm));
    auto r = g.compile("inkdt_warmup");
    std::vector<Binding> bd = {{"h", &a, 0}, {"m", &b, 0}, {"c", &c, 0}};
    r->enqueue(st, bd);
    st.synchronize();
}

// LUT check: ink_mg_out_gaudi2 (tanh through v_sigmoid_f32, a LUT function) through Direct vs Synapse. The Synapse
// launch comes first: a Direct TPC launch before any Synapse launch on the device is a device error.
static int luttest(const std::string& objdir) {
    Device dev;
    Stream st(dev);
    const uint64_t H = 2304, G = 3, Gh = 3, T = 16;
    const uint64_t nh = H * Gh * T, nm = H * 4 * T, nf = 2 * H * T * G, nw = H * G, no = H * T * G;
    DeviceBuffer dh(dev, nh * 4), dm(dev, nm * 4), df(dev, nf * 4), dw(dev, nw * 2), dout(dev, no * 2), dr(dev, no * 2);
    HostBuffer hb(dev, nf * 4 + 4096);
    uint64_t seed = 7;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return float(int((seed >> 33) % 2001) - 1000) / 300.f; };
    auto up32 = [&](DeviceBuffer& d, uint64_t n) { for (uint64_t i = 0; i < n; ++i) hb.as<float>()[i] = rnd(); st.upload(hb, 0, d, 0, n * 4); st.synchronize(); };
    up32(dh, nh); up32(dm, nm); up32(df, nf);
    for (uint64_t i = 0; i < nw; ++i) hb.as<uint16_t>()[i] = to_bf16(1.f + rnd() * 0.1f);
    st.upload(hb, 0, dw, 0, nw * 2); st.synchronize();
    struct { float eps; int32_t m0; } prm{1e-6f, 0};
    auto synapse = [&]() {
        GraphBuilder g(dev);
        const auto hi = g.input("h", DType::F32, {T, Gh, H});
        const auto mi = g.input("m", DType::F32, {T, 4, H});
        const auto fi = g.input("f", DType::F32, {G, T, 2 * H});
        const auto wi = g.input("w", DType::BF16, {G, H});
        const auto oo = g.input("o", DType::BF16, {G, T, H});
        g.op("ink_mg_out_gaudi2", {hi, mi, fi, wi}, {oo}, &prm, sizeof(prm));
        auto r = g.compile("inkdt_mgout");
        std::vector<Binding> b = {{"h", &dh, 0}, {"m", &dm, 0}, {"f", &df, 0}, {"w", &dw, 0}, {"o", &dr, 0}};
        r->enqueue(st, b);
        st.synchronize();
    };
    synapse();
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/ink_mg_out_gaudi2.o");
    const DirectTpcTensor ts[] = {
        {dh.address(), DType::F32, 3, {uint32_t(H), uint32_t(Gh), uint32_t(T), 1, 1}, {0, 0, 0, 0, 0}, 0},
        {dm.address(), DType::F32, 3, {uint32_t(H), 4, uint32_t(T), 1, 1}, {0, 0, 0, 0, 0}, 0},
        {df.address(), DType::F32, 3, {uint32_t(2 * H), uint32_t(T), uint32_t(G), 1, 1}, {0, 0, 0, 0, 0}, 0},
        {dw.address(), DType::BF16, 2, {uint32_t(H), uint32_t(G), 1, 1, 1}, {0, 0, 0, 0, 0}, 0},
        {dout.address(), DType::BF16, 3, {uint32_t(H), uint32_t(T), uint32_t(G), 1, 1}, {0, 0, 0, 0, 0}, 0}};
    uint32_t srfs[2];
    std::memcpy(srfs, &prm, sizeof(prm));
    DirectPlan plan(rt);
    plan.add_tpc_grid(isa, ts, srfs, {uint32_t(G), uint32_t(T), 1, 1, 1}, 1);
    plan.prepare();
    plan.run();
    std::vector<uint16_t> c(no), ref(no);
    st.download(dout, 0, hb, 0, no * 2); st.synchronize(); std::memcpy(c.data(), hb.data(), no * 2);
    st.download(dr, 0, hb, 0, no * 2); st.synchronize(); std::memcpy(ref.data(), hb.data(), no * 2);
    uint64_t same = 0;
    for (uint64_t i = 0; i < no; ++i) same += c[i] == ref[i];
    std::printf("ink_mg_out (LUT) via Direct: bit-identical %.4f%%\n", 100.0 * double(same) / double(no));
    return 0;
}

static int pdmatest() {
    Device dev;
    Stream st(dev);
    DeviceBuffer da(dev, 1 << 20);
    HostBuffer hb(dev, 4096);
    uint32_t* p = hb.as<uint32_t>();
    for (int i = 0; i < 64; ++i) p[i] = 0xdead0000u + uint32_t(i);
    st.upload(hb, 0, da, 0, 256);
    st.synchronize();
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    uint32_t got[64] = {};
    rt.pdma_d2h_sync(da.address(), got, sizeof(got));
    int bad = 0;
    for (int i = 0; i < 64; ++i) bad += (got[i] != 0xdead0000u + uint32_t(i));
    std::printf("pdma_d2h_sync 256B: bad=%d (got[0]=%#x got[63]=%#x)\n", bad, got[0], got[63]);
    if (bad) return 1;
    uint32_t send[64];
    for (int i = 0; i < 64; ++i) send[i] = 0xcafe0000u + uint32_t(i);
    rt.pdma_h2d_sync(send, da.address() + 4096, sizeof(send));
    std::memset(got, 0, sizeof(got));
    rt.pdma_d2h_sync(da.address() + 4096, got, sizeof(got));
    for (int i = 0; i < 64; ++i) bad += (got[i] != 0xcafe0000u + uint32_t(i));
    std::printf("pdma_h2d_sync 256B: bad=%d (got[0]=%#x got[63]=%#x)\n", bad, got[0], got[63]);
    if (bad) return 1;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
    std::vector<double> td, tu, tda;
    for (int i = 0; i < 100; ++i) {
        auto t0 = now();
        rt.pdma_d2h_sync(da.address(), got, 256);
        td.push_back(us(t0, now()));
    }
    std::vector<uint8_t> big(65536, 0x5a);
    for (int i = 0; i < 100; ++i) {
        auto t0 = now();
        rt.pdma_h2d_sync(big.data(), da.address(), 65536);
        tu.push_back(us(t0, now()));
    }
    for (int i = 0; i < 20; ++i) {
        auto w0 = now();
        while (us(w0, now()) < 2500.0) {}  // busy-spin 2.5 ms like wait_fence
        auto t0 = now();
        rt.pdma_d2h_sync(da.address(), got, 256);
        tda.push_back(us(t0, now()));
    }
    auto pr = [](const char* n, std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        std::printf("%-30s med %6.1f us  p90 %6.1f  max %6.1f\n", n, v[v.size()/2], v[v.size()*9/10], v.back());
    };
    pr("SCAL pdma_d2h 256B", td);
    pr("SCAL pdma_h2d 64KiB", tu);
    pr("SCAL pdma_d2h after 2.5ms", tda);
    return 0;
}

// Unit test and timing for ink_nvf4_deq_bf16_gaudi2 (scale/8 convention, optional expert mask).
static int deqtest(const std::string& objdir) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Device dev;
    Stream st(dev);
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/ink_nvf4_deq_bf16_gaudi2.o");
    auto T = [](uint64_t a, DType dt, std::initializer_list<uint32_t> sz) {
        DirectTpcTensor t{};
        t.addr = a; t.dtype = dt; t.dims = uint32_t(sz.size());
        uint32_t i = 0;
        for (auto v : sz) t.sizes[i++] = v;
        for (; i < 5; ++i) t.sizes[i] = 1;
        return t;
    };
    static const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    float f8_lut[256];
    for (int i = 0; i < 256; ++i) f8_lut[i] = from_fp8_143(uint8_t(i));
    auto eq_bf = [](uint16_t a, uint16_t b) { return a == b || (((a | b) & 0x7fff) == 0); };

    const uint32_t N = 2304, K13 = 2304, K2 = 1152;
    uint64_t seed = 42;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(seed >> 33); };
    auto check_case = [&](const char* label, uint32_t E, uint32_t K, const std::vector<int32_t>& mask, uint32_t use_mask) {
        std::vector<uint8_t> w(uint64_t(E) * (K / 2) * N), bs(uint64_t(E) * (K / 16) * N);
        for (auto& v : w) v = uint8_t(rnd() & 0xff);
        // All 87 valid E4M3FN codes [0x28..0x7e] shifted by -0x18 -> [0x10..0x66] (scale / 8 in [0.03125, 56])
        for (auto& v : bs) v = uint8_t((0x28 + (rnd() % 87)) - 0x18);
        const uint64_t obytes = uint64_t(E) * K * N * 2;
        DeviceBuffer dw(dev, w.size()), dbs(dev, bs.size()), dm(dev, std::max<uint64_t>(E * 4, 256)), dout(dev, obytes);
        HostBuffer hb(dev, obytes + 256);
        auto up = [&](DeviceBuffer& d, const void* p, uint64_t nb) {
            std::memcpy(hb.data(), p, nb); st.upload(hb, 0, d, 0, nb); st.synchronize();
        };
        up(dw, w.data(), w.size()); up(dbs, bs.data(), bs.size()); up(dm, mask.data(), E * 4);
        std::fill_n(hb.as<uint16_t>(), obytes / 2, uint16_t(0x5a5au));
        st.upload(hb, 0, dout, 0, obytes);
        st.synchronize();

        DirectPlan p(rt);
        const DirectTpcTensor ts[] = {
            T(dw.address(), DType::U8, {N, K / 2, E}), T(dbs.address(), DType::F8, {N, K / 16, E}),
            T(dm.address(), DType::I32, {E}), T(dout.address(), DType::BF16, {N, K, E})};
        p.add_tpc_grid(isa, ts, std::span<const uint32_t>(&use_mask, 1), {(N / 256) * E, 1, 1, 1, 1}, 0, 24);
        p.prepare();
        p.run();
        st.download(dout, 0, hb, 0, obytes);
        st.synchronize();
        const uint16_t* got = hb.as<uint16_t>();
        uint64_t diffs = 0, skipped_dirty = 0;
        for (uint32_t e = 0; e < E; ++e) {
            const uint16_t* ge = got + uint64_t(e) * K * N;
            if (use_mask && mask[e] == 0) {
                for (uint64_t i = 0; i < uint64_t(K) * N; ++i) skipped_dirty += (ge[i] != 0x5a5au);
                continue;
            }
            const uint8_t* we = w.data() + uint64_t(e) * (K / 2) * N;
            const uint8_t* bse = bs.data() + uint64_t(e) * (K / 16) * N;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t r = k >> 1, kb = k >> 4, hi = k & 1;
                for (uint32_t g = 0; g < N / 256; ++g) {
                    const uint32_t c0 = g * 256;
                    for (uint32_t i = 0; i < 128; ++i) {
                        const uint8_t bev = we[uint64_t(r) * N + c0 + 2 * i];
                        const uint8_t bod = we[uint64_t(r) * N + c0 + 2 * i + 1];
                        const uint8_t nev = hi ? (bev >> 4) : (bev & 15);
                        const uint8_t nod = hi ? (bod >> 4) : (bod & 15);
                        const uint16_t rev = to_bf16((kE2M1[nev] * 0.015625f) * f8_lut[bse[uint64_t(kb) * N + c0 + 2 * i]]);
                        const uint16_t rod = to_bf16((kE2M1[nod] * 0.015625f) * f8_lut[bse[uint64_t(kb) * N + c0 + 2 * i + 1]]);
                        diffs += !eq_bf(rev, ge[uint64_t(k) * N + c0 + i]) + !eq_bf(rod, ge[uint64_t(k) * N + c0 + 128 + i]);
                    }
                }
            }
        }
        std::printf("[deqtest] %-20s E=%u K=%u mask=%u: diffs=%llu skipped_dirty=%llu\n", label, E, K, use_mask,
                    (unsigned long long)diffs, (unsigned long long)skipped_dirty);
        return (diffs == 0 && skipped_dirty == 0);
    };
    if (!check_case("E=1 w13 unmasked", 1, K13, {0}, 0)) return 1;
    if (!check_case("E=1 w2  unmasked", 1, K2,  {0}, 0)) return 1;
    if (!check_case("E=4 w13 masked",   4, K13, {1, 0, 5, 0}, 1)) return 1;
    if (!check_case("E=4 w2  masked",   4, K2,  {0, 3, 0, 2}, 1)) return 1;

    // Full-layer timing (E = 128)
    const uint32_t E = 128;
    DeviceBuffer dw13(dev, uint64_t(E) * (K13 / 2) * N), dbs13(dev, uint64_t(E) * (K13 / 16) * N), do13(dev, uint64_t(E) * K13 * N * 2);
    DeviceBuffer dw2(dev, uint64_t(E) * (K2 / 2) * N), dbs2(dev, uint64_t(E) * (K2 / 16) * N), do2(dev, uint64_t(E) * K2 * N * 2);
    DeviceBuffer dm(dev, 4096);
    st.zero(dw13, 0, dw13.size()); st.zero(dbs13, 0, dbs13.size());
    st.zero(dw2, 0, dw2.size()); st.zero(dbs2, 0, dbs2.size());
    st.synchronize();
    auto bench = [&](const char* label, const DeviceBuffer& dw, const DeviceBuffer& dbs, const DeviceBuffer& dout,
                     uint32_t K, uint32_t use_mask, uint32_t active) {
        std::vector<int32_t> m(E, 0);
        for (uint32_t i = 0; i < E; ++i) m[i] = (i * active) / E != ((i + 1) * active) / E ? 1 : 0;
        rt.pdma_h2d_sync(m.data(), dm.address(), E * 4);
        DirectPlan p(rt);
        const DirectTpcTensor ts[] = {
            T(dw.address(), DType::U8, {N, K / 2, E}), T(dbs.address(), DType::F8, {N, K / 16, E}),
            T(dm.address(), DType::I32, {E}), T(dout.address(), DType::BF16, {N, K, E})};
        p.add_tpc_grid(isa, ts, std::span<const uint32_t>(&use_mask, 1), {(N / 256) * E, 1, 1, 1, 1}, 0, 24);
        p.prepare();
        std::vector<double> t;
        for (int r = 0; r < 15; ++r) t.push_back(p.run());
        std::sort(t.begin(), t.end());
        const double us = t[7];
        const double bytes = double(active) * N * (K / 2 + K / 16 + K * 2);
        std::printf("[deqtest] %-20s act=%3u/128: %7.1f us (%.0f GB/s)\n", label, active, us, bytes / us * 1e-3);
    };
    bench("w13 full unmasked", dw13, dbs13, do13, K13, 0, 128);
    bench("w2  full unmasked", dw2,  dbs2,  do2,  K2,  0, 128);
    bench("w13 masked 64",     dw13, dbs13, do13, K13, 1, 64);
    bench("w2  masked 64",     dw2,  dbs2,  do2,  K2,  1, 64);
    return 0;
}

// Exhaustive 256-code x representative-scale exactness test, masked test, and E=128 timing for ink_mxfp8_deq_bf16_gaudi2.
static int mxdeqtest(const std::string& objdir) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Device dev;
    Stream st(dev);
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    const uint64_t isa = rt.load_tpc_elf(objdir + "/ink_mxfp8_deq_bf16_gaudi2.o");
    auto T = [](uint64_t a, DType dt, std::initializer_list<uint32_t> sz) {
        DirectTpcTensor t{};
        t.addr = a; t.dtype = dt; t.dims = uint32_t(sz.size());
        uint32_t i = 0;
        for (auto v : sz) t.sizes[i++] = v;
        for (; i < 5; ++i) t.sizes[i] = 1;
        return t;
    };
    float f8_lut[256], mx_lut[256];
    for (int i = 0; i < 256; ++i) {
        f8_lut[i] = from_fp8_143(uint8_t(i));
        const float sgn = (i & 0x80) ? -1.f : 1.f;
        const int ex = (i >> 3) & 15, mn = i & 7;
        if (ex == 0) mx_lut[i] = sgn * std::ldexp(float(mn), -9);
        else mx_lut[i] = sgn * std::ldexp(1.f + float(mn) / 8.f, ex - 7);  // 0x7f/0xff -> +-480.f (FORM_C4 deterministic)
    }
    auto eq_bf = [](uint16_t a, uint16_t b) { return a == b || (((a | b) & 0x7fff) == 0); };

    // 1. Compact exhaustive matrix: all 256 weight codes in both even and odd lanes across all 4 row phases x 12 scales
    //    (all 7 MXFP8 pow2 scales [2^-6..1.0] + edge fp8_143 scales 0, 2^-9, 1.5, 240, -1) + mixed-scale/lane blocks.
    {
        const uint8_t scales[12] = {0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x00, 0x01, 0x3c, 0x77, 0xb8};
        const uint32_t E = 1, N = 512, KB = 16, K = KB * 32;
        std::vector<uint8_t> w(uint64_t(K) * N), bs(uint64_t(KB) * N);
        for (uint32_t kb = 0; kb < KB; ++kb) {
            for (uint32_t c = 0; c < N; ++c)
                bs[uint64_t(kb) * N + c] = (kb < 12) ? scales[kb] : scales[(c + kb * 3) % 12];
            for (uint32_t kin = 0; kin < 32; ++kin) {
                const uint32_t k = kb * 32 + kin;
                for (uint32_t i = 0; i < N / 2; ++i) {
                    w[uint64_t(k) * N + 2 * i]     = uint8_t((i + kin) & 0xff);
                    w[uint64_t(k) * N + 2 * i + 1] = uint8_t(((i ^ 0x78) + kin * 3) & 0xff);
                }
            }
        }
        const uint64_t obytes = uint64_t(K) * N * 2;
        DeviceBuffer dw(dev, w.size()), dbs(dev, bs.size()), dm(dev, 256), dout(dev, obytes);
        HostBuffer hb(dev, obytes + 256);
        std::memcpy(hb.data(), w.data(), w.size()); st.upload(hb, 0, dw, 0, w.size()); st.synchronize();
        std::memcpy(hb.data(), bs.data(), bs.size()); st.upload(hb, 0, dbs, 0, bs.size()); st.synchronize();
        std::fill_n(hb.as<uint16_t>(), obytes / 2, uint16_t(0x5a5au));
        st.upload(hb, 0, dout, 0, obytes); st.synchronize();
        const uint32_t use_mask = 0;
        DirectPlan p(rt);
        const DirectTpcTensor ts[] = {
            T(dw.address(), DType::U8, {N, K, E}), T(dbs.address(), DType::F8, {N, KB, E}),
            T(dm.address(), DType::I32, {E}), T(dout.address(), DType::BF16, {N, K, E})};
        p.add_tpc_grid(isa, ts, std::span<const uint32_t>(&use_mask, 1), {(N / 256) * E, 1, 1, 1, 1}, 0, 24);
        p.prepare();
        p.run();
        st.download(dout, 0, hb, 0, obytes); st.synchronize();
        const uint16_t* got = hb.as<uint16_t>();
        uint64_t fin_diffs = 0, exp15_diffs = 0, sub_diffs = 0, nz_diffs = 0, nan_diffs = 0;
        for (uint32_t k = 0; k < K; ++k) {
            const uint32_t kb = k >> 5;
            for (uint32_t g = 0; g < N / 256; ++g) {
                const uint32_t c0 = g * 256;
                for (uint32_t i = 0; i < 128; ++i) {
                    const uint8_t codes[2] = {w[uint64_t(k) * N + c0 + 2 * i], w[uint64_t(k) * N + c0 + 2 * i + 1]};
                    const uint8_t bss[2]   = {bs[uint64_t(kb) * N + c0 + 2 * i], bs[uint64_t(kb) * N + c0 + 2 * i + 1]};
                    const uint16_t act[2]  = {got[uint64_t(k) * N + c0 + i], got[uint64_t(k) * N + c0 + 128 + i]};
                    for (int lane = 0; lane < 2; ++lane) {
                        const uint8_t c = codes[lane];
                        const uint16_t ref = to_bf16(mx_lut[c] * f8_lut[bss[lane]]);
                        const bool bad = !eq_bf(ref, act[lane]);
                        if ((c & 0x7f) == 0x7f) {
                            nan_diffs += bad;
                        } else {
                            fin_diffs += bad;
                            if (((c >> 3) & 15) == 15) exp15_diffs += bad;
                            if (((c >> 3) & 15) == 0 && (c & 7) != 0) sub_diffs += bad;
                            if (c == 0x80) nz_diffs += bad;
                        }
                    }
                }
            }
        }
        std::printf("[mxdeqtest] all-256 x 12 scales (E=1 K=%u N=%u): finite_diffs=%llu exp15_diffs=%llu subnorm_diffs=%llu neg0_diffs=%llu nan_code_diffs=%llu\n",
                    K, N, (unsigned long long)fin_diffs, (unsigned long long)exp15_diffs,
                    (unsigned long long)sub_diffs, (unsigned long long)nz_diffs, (unsigned long long)nan_diffs);
        if (fin_diffs || exp15_diffs || sub_diffs || nz_diffs || nan_diffs) return 1;
    }

    // 2. Production shapes unmasked & masked (finite E4M3FN incl. exp=15, subnormals, -0.0; pow2 scales [0x08..0x38])
    const uint32_t N = 2304, K13 = 2304, K2 = 1152;
    uint64_t seed = 20261125ull;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(seed >> 33); };
    auto check_case = [&](const char* label, uint32_t E, uint32_t K, const std::vector<int32_t>& mask, uint32_t use_mask) {
        std::vector<uint8_t> w(uint64_t(E) * K * N), bs(uint64_t(E) * (K / 32) * N);
        for (auto& v : w) {
            uint8_t c = uint8_t(rnd() & 0xff);
            if ((c & 0x7f) == 0x7f) c ^= 1;  // keep finite E4M3FN
            v = c;
        }
        for (auto& v : bs) v = uint8_t(0x08 + (rnd() % 7) * 0x08);  // 2^-6 .. 1.0
        const uint64_t obytes = uint64_t(E) * K * N * 2;
        DeviceBuffer dw(dev, w.size()), dbs(dev, bs.size()), dm(dev, std::max<uint64_t>(E * 4, 256)), dout(dev, obytes);
        HostBuffer hb(dev, obytes + 256);
        auto up = [&](DeviceBuffer& d, const void* p, uint64_t nb) {
            std::memcpy(hb.data(), p, nb); st.upload(hb, 0, d, 0, nb); st.synchronize();
        };
        up(dw, w.data(), w.size()); up(dbs, bs.data(), bs.size()); up(dm, mask.data(), E * 4);
        std::fill_n(hb.as<uint16_t>(), obytes / 2, uint16_t(0x5a5au));
        st.upload(hb, 0, dout, 0, obytes); st.synchronize();

        DirectPlan p(rt);
        const DirectTpcTensor ts[] = {
            T(dw.address(), DType::U8, {N, K, E}), T(dbs.address(), DType::F8, {N, K / 32, E}),
            T(dm.address(), DType::I32, {E}), T(dout.address(), DType::BF16, {N, K, E})};
        p.add_tpc_grid(isa, ts, std::span<const uint32_t>(&use_mask, 1), {(N / 256) * E, 1, 1, 1, 1}, 0, 24);
        p.prepare();
        p.run();
        st.download(dout, 0, hb, 0, obytes); st.synchronize();
        const uint16_t* got = hb.as<uint16_t>();
        uint64_t diffs = 0, skipped_dirty = 0;
        for (uint32_t e = 0; e < E; ++e) {
            const uint16_t* ge = got + uint64_t(e) * K * N;
            if (use_mask && mask[e] == 0) {
                for (uint64_t i = 0; i < uint64_t(K) * N; ++i) skipped_dirty += (ge[i] != 0x5a5au);
                continue;
            }
            const uint8_t* we = w.data() + uint64_t(e) * K * N;
            const uint8_t* bse = bs.data() + uint64_t(e) * (K / 32) * N;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t kb = k >> 5;
                for (uint32_t g = 0; g < N / 256; ++g) {
                    const uint32_t c0 = g * 256;
                    for (uint32_t i = 0; i < 128; ++i) {
                        const uint16_t rev = to_bf16(mx_lut[we[uint64_t(k) * N + c0 + 2 * i]]     * f8_lut[bse[uint64_t(kb) * N + c0 + 2 * i]]);
                        const uint16_t rod = to_bf16(mx_lut[we[uint64_t(k) * N + c0 + 2 * i + 1]] * f8_lut[bse[uint64_t(kb) * N + c0 + 2 * i + 1]]);
                        diffs += !eq_bf(rev, ge[uint64_t(k) * N + c0 + i]) + !eq_bf(rod, ge[uint64_t(k) * N + c0 + 128 + i]);
                    }
                }
            }
        }
        std::printf("[mxdeqtest] %-20s E=%u K=%u mask=%u: diffs=%llu skipped_dirty=%llu\n", label, E, K, use_mask,
                    (unsigned long long)diffs, (unsigned long long)skipped_dirty);
        return (diffs == 0 && skipped_dirty == 0);
    };
    if (!check_case("E=1 w13 unmasked", 1, K13, {0}, 0)) return 1;
    if (!check_case("E=1 w2  unmasked", 1, K2,  {0}, 0)) return 1;
    if (!check_case("E=4 w13 masked",   4, K13, {1, 0, 5, 0}, 1)) return 1;
    if (!check_case("E=4 w2  masked",   4, K2,  {0, 3, 0, 2}, 1)) return 1;

    // 3. Full-layer timing (E = 128)
    const uint32_t E = 128;
    DeviceBuffer dw13(dev, uint64_t(E) * K13 * N), dbs13(dev, uint64_t(E) * (K13 / 32) * N), do13(dev, uint64_t(E) * K13 * N * 2);
    DeviceBuffer dw2(dev, uint64_t(E) * K2 * N), dbs2(dev, uint64_t(E) * (K2 / 32) * N), do2(dev, uint64_t(E) * K2 * N * 2);
    DeviceBuffer dm(dev, 4096);
    st.zero(dw13, 0, dw13.size()); st.zero(dbs13, 0, dbs13.size());
    st.zero(dw2, 0, dw2.size()); st.zero(dbs2, 0, dbs2.size());
    st.synchronize();
    auto bench = [&](const char* label, const DeviceBuffer& dw, const DeviceBuffer& dbs, const DeviceBuffer& dout,
                     uint32_t K, uint32_t use_mask, uint32_t active) {
        std::vector<int32_t> m(E, 0);
        for (uint32_t i = 0; i < E; ++i) m[i] = (i * active) / E != ((i + 1) * active) / E ? 1 : 0;
        rt.pdma_h2d_sync(m.data(), dm.address(), E * 4);
        DirectPlan p(rt);
        const DirectTpcTensor ts[] = {
            T(dw.address(), DType::U8, {N, K, E}), T(dbs.address(), DType::F8, {N, K / 32, E}),
            T(dm.address(), DType::I32, {E}), T(dout.address(), DType::BF16, {N, K, E})};
        p.add_tpc_grid(isa, ts, std::span<const uint32_t>(&use_mask, 1), {(N / 256) * E, 1, 1, 1, 1}, 0, 24);
        p.prepare();
        std::vector<double> t;
        for (int r = 0; r < 15; ++r) t.push_back(p.run());
        std::sort(t.begin(), t.end());
        const double us = t[7];
        const double bytes = double(active) * N * (K + K / 32 + K * 2);
        std::printf("[mxdeqtest] %-20s act=%3u/128: %7.1f us (%.0f GB/s)\n", label, active, us, bytes / us * 1e-3);
    };
    bench("w13 full unmasked", dw13, dbs13, do13, K13, 0, 128);
    bench("w2  full unmasked", dw2,  dbs2,  do2,  K2,  0, 128);
    bench("w13 masked 64",     dw13, dbs13, do13, K13, 1, 64);
    bench("w2  masked 64",     dw2,  dbs2,  do2,  K2,  1, 64);
    return 0;
}

// Compact exactness unit test for all 5 BF16 prefill MoE TPC kernels:
// ink_rplan_dyn_{tab,slots}_bf16, ink_sgather_bf16, ink_act_bf16, ink_moe_combine_bf16.
static int bf16pbtest(const std::string& objdir) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Device dev;
    Stream st(dev);
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);
    auto isa = [&](const char* k) { return rt.load_tpc_elf(objdir + "/" + k + ".o"); };
    auto tn = [](uint64_t a, DType dt, std::initializer_list<uint32_t> sz) {
        DirectTpcTensor t{};
        t.addr = a; t.dtype = dt; t.dims = uint32_t(sz.size());
        uint32_t i = 0;
        for (auto v : sz) t.sizes[i++] = v;
        for (; i < 5; ++i) t.sizes[i] = 1;
        return t;
    };
    auto srfv = [](const auto& p) {
        std::vector<uint32_t> v(sizeof(p) / 4);
        std::memcpy(v.data(), &p, sizeof(p));
        return v;
    };
    auto bf16_eq = [](uint16_t a, uint16_t b) { return a == b || (((a | b) & 0x7fffu) == 0); };
    auto bf16_ulp = [](uint16_t a, uint16_t b) -> int {
        if (a == b || (((a | b) & 0x7fffu) == 0)) return 0;
        auto ord = [](uint16_t u) -> int { return (u & 0x8000u) ? -(int(u & 0x7fffu)) : int(u & 0x7fffu); };
        return std::abs(ord(a) - ord(b));
    };
    auto alloc_bf16 = [](uint32_t c) {
        const uint32_t q = c >> 8, r = c & 255u;
        uint32_t rr = 256;
        if (r == 0) rr = 0;
        else if (r <= 16) rr = 16;
        else if (r <= 32) rr = 32;
        else if (r <= 64) rr = 64;
        else if (r <= 96) rr = 96;
        else if (r <= 128) rr = 128;
        else if (r <= 144) rr = 144;
        else if (r <= 160) rr = 160;
        else if (r <= 192) rr = 192;
        else if (r <= 224) rr = 224;
        return (q << 8) + rr;
    };

    const auto bf = DType::BF16, f32 = DType::F32, i32 = DType::I32;
    constexpr uint32_t H = 2304, F = 1152, E = 128, K = 8, Tu = 32, Su = Tu * K, VLu = 27;
    uint64_t rng = 0x5eed123456789abfull;
    auto nxt = [&]() { rng = rng * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(rng >> 32); };
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * (float(nxt() & 0xffffu) / 65535.f); };

    std::vector<int32_t> h_idx(Su), h_tp(25);
    for (uint32_t i = 0; i <= 24; ++i) h_tp[i] = int32_t(Tu * i / 24);
    for (uint32_t t = 0; t < Tu; ++t) {
        const int32_t ex[8] = {0, int32_t(1 + (t % 7)), int32_t(32 + (t % 5)), int32_t(37 + (t % 11)),
                               int32_t(64 + (t % 3)), int32_t(70 + (t % 13)), int32_t(96 + (t % 4)), int32_t(101 + (t % 17))};
        for (uint32_t j = 0; j < K; ++j) h_idx[t * K + j] = ex[j];
    }
    std::vector<int32_t> ref_hist(24 * E, 0), ref_off(24 * E, 0), ref_cnt(E, 0), ref_eb(E, 0);
    for (int p = 0; p < 24; ++p)
        for (int s = h_tp[p] * int(K); s < h_tp[p + 1] * int(K); ++s) ref_hist[p * int(E) + h_idx[s]]++;
    uint32_t Ru = 0;
    for (uint32_t e = 0; e < E; ++e) {
        int acc = 0;
        for (int p = 0; p < 24; ++p) { ref_off[p * int(E) + int(e)] = acc; acc += ref_hist[p * int(E) + int(e)]; }
        ref_cnt[e] = acc;
        ref_eb[e] = int32_t(Ru);
        Ru += alloc_bf16(uint32_t(acc));
    }
    std::vector<int32_t> ref_code(Su, -1), ref_lbuf(4 * Su, 0), ref_lcnt(4 * 24, 0);
    for (int p = 0; p < 24; ++p) {
        int run[E];
        for (uint32_t e = 0; e < E; ++e) run[e] = ref_off[p * int(E) + int(e)];
        int nw[4] = {0, 0, 0, 0};
        const int s0 = h_tp[p] * int(K), s1 = h_tp[p + 1] * int(K);
        for (int s = s0; s < s1; ++s) {
            const int e = h_idx[s], r = run[e]++;
            ref_code[s] = ref_eb[e] + r;
            const int w = e >> 5;
            ref_lbuf[w * int(Su) + s0 + nw[w]++] = s;
        }
        for (int w = 0; w < 4; ++w) ref_lcnt[w * 24 + p] = nw[w];
    }

    DeviceBuffer d_idx(dev, Su * 4 + 256), d_tp(dev, 256), d_vlen(dev, 256);
    DeviceBuffer d_hist(dev, 24 * E * 4 + 256), d_off(dev, 24 * E * 4 + 256), d_cnt(dev, E * 4 + 256), d_eb(dev, E * 4 + 256);
    DeviceBuffer d_code(dev, Su * 4 + 256), d_lbuf(dev, 4 * Su * 4 + 256), d_lcnt(dev, 4 * 24 * 4 + 256);
    DeviceBuffer d_x(dev, Tu * H * 2 + 256), d_xb(dev, uint64_t(Ru) * H * 2 + 256);
    DeviceBuffer d_gu(dev, uint64_t(Ru) * 2 * F * 4 + 256), d_ws(dev, uint64_t(E) * 2 * F * 4 + 256), d_ab(dev, uint64_t(Ru) * F * 2 + 256);
    DeviceBuffer d_y(dev, uint64_t(Ru) * H * 4 + 256), d_cs(dev, uint64_t(E) * H * 4 + 256);
    DeviceBuffer d_rw(dev, Su * 4 + 256), d_out(dev, Tu * H * 2 + 256);
    HostBuffer hb(dev, std::max<uint64_t>(uint64_t(Ru) * 2 * F * 4, uint64_t(E) * 2 * F * 4) + 4096);
    auto up = [&](DeviceBuffer& d, const void* ptr, uint64_t nb) {
        std::memcpy(hb.data(), ptr, nb); st.upload(hb, 0, d, 0, nb); st.synchronize();
    };
    auto dl = [&](DeviceBuffer& d, void* ptr, uint64_t nb) {
        st.download(d, 0, hb, 0, nb); st.synchronize(); std::memcpy(ptr, hb.data(), nb);
    };
    const int32_t vl_i = int32_t(VLu);
    up(d_idx, h_idx.data(), Su * 4); up(d_tp, h_tp.data(), 25 * 4); up(d_vlen, &vl_i, 4);
    st.zero(d_lbuf, 0, 4 * Su * 4); st.synchronize();
    int failures = 0;

    // 1. ink_rplan_dyn_tab_bf16 + ink_rplan_dyn_slots_bf16
    {
        DirectPlan p(rt);
        const DirectTpcTensor t1[] = {tn(d_idx.address(), i32, {Su}), tn(d_tp.address(), i32, {25}), tn(d_hist.address(), i32, {E, 24})};
        p.add_tpc_grid(isa("ink_rplan_hist_gaudi2"), t1, srfv(int32_t(3)), {24, 1, 1, 1, 1}, 0, 24, -1, 0, 0);
        const DirectTpcTensor t2[] = {tn(d_hist.address(), i32, {E, 24}), tn(d_off.address(), i32, {E, 24}),
                                      tn(d_cnt.address(), i32, {E}), tn(d_eb.address(), i32, {E})};
        p.add_tpc_grid(isa("ink_rplan_dyn_tab_bf16_gaudi2"), t2, {}, {1, 1, 1, 1, 1}, 0, 24, 0, 24, 1);
        const DirectTpcTensor t3[] = {tn(d_idx.address(), i32, {Su}), tn(d_off.address(), i32, {E, 24}),
                                      tn(d_eb.address(), i32, {E}), tn(d_tp.address(), i32, {25}),
                                      tn(d_code.address(), i32, {Su}), tn(d_lbuf.address(), i32, {Su, 4}), tn(d_lcnt.address(), i32, {24, 4})};
        p.add_tpc_grid(isa("ink_rplan_dyn_slots_bf16_gaudi2"), t3, srfv(int32_t(3)), {24, 1, 1, 1, 1}, 0, 24, 1, 1, -1);
        p.prepare();
        p.run();
        std::vector<int32_t> go(24 * E), gc(E), geb(E), gcode(Su), glb(4 * Su), glc(4 * 24);
        dl(d_off, go.data(), go.size() * 4); dl(d_cnt, gc.data(), gc.size() * 4); dl(d_eb, geb.data(), geb.size() * 4);
        dl(d_code, gcode.data(), gcode.size() * 4); dl(d_lbuf, glb.data(), glb.size() * 4); dl(d_lcnt, glc.data(), glc.size() * 4);
        size_t bad = 0;
        for (size_t i = 0; i < go.size(); ++i) bad += go[i] != ref_off[i];
        for (size_t i = 0; i < gc.size(); ++i) bad += (gc[i] != ref_cnt[i]) + (geb[i] != ref_eb[i]);
        for (size_t i = 0; i < gcode.size(); ++i) bad += gcode[i] != ref_code[i];
        for (size_t i = 0; i < glc.size(); ++i) bad += glc[i] != ref_lcnt[i];
        for (int w = 0; w < 4; ++w)
            for (int pc = 0; pc < 24; ++pc)
                for (int k = 0; k < ref_lcnt[w * 24 + pc]; ++k)
                    bad += glb[w * int(Su) + h_tp[pc] * int(K) + k] != ref_lbuf[w * int(Su) + h_tp[pc] * int(K) + k];
        if (bad) ++failures;
        std::printf("[bf16pbtest] %-28s %s: T=%u Ru=%u mismatches=%zu\n", "rplan_dyn_{tab,slots}_bf16", bad ? "FAIL" : "PASS", Tu, Ru, bad);
    }

    // 2. ink_sgather_bf16 (vlen=27/32)
    {
        std::vector<uint16_t> hx(Tu * H);
        for (auto& v : hx) v = to_bf16(rf(-4.f, 4.f));
        up(d_x, hx.data(), hx.size() * 2);
        st.zero(d_xb, 0, uint64_t(Ru) * H * 2); st.synchronize();
        DirectPlan p(rt);
        const DirectTpcTensor ts[] = {tn(d_x.address(), bf, {H, Tu}), tn(d_code.address(), i32, {K, Tu}),
                                      tn(d_xb.address(), bf, {H, Ru}), tn(d_vlen.address(), i32, {1})};
        p.add_tpc_grid(isa("ink_sgather_bf16_gaudi2"), ts, {}, {Tu, 1, 1, 1, 1}, 0, 24);
        p.prepare();
        p.run();
        std::vector<uint16_t> got(uint64_t(Ru) * H), exp(uint64_t(Ru) * H, 0);
        dl(d_xb, got.data(), got.size() * 2);
        for (uint32_t t = 0; t < VLu; ++t)
            for (uint32_t j = 0; j < K; ++j)
                std::memcpy(&exp[uint64_t(ref_code[t * K + j]) * H], &hx[uint64_t(t) * H], H * 2);
        uint64_t same = 0;
        for (size_t i = 0; i < got.size(); ++i) same += got[i] == exp[i];
        const bool ok = (same == got.size());
        if (!ok) ++failures;
        std::printf("[bf16pbtest] %-28s %s: %llu / %zu bit-exact (vlen=%u/%u)\n", "ink_sgather_bf16", ok ? "PASS" : "FAIL",
                    (unsigned long long)same, got.size(), VLu, Tu);
    }

    // 3. ink_act_bf16 (4 waves, F32 gu in [gate 128 | up 128], per-row F32 ws)
    {
        const float cu = 25.f;
        std::vector<float> gu(uint64_t(Ru) * 2 * F), ws(uint64_t(E) * 2 * F);
        for (uint32_t r = 0; r < Ru; ++r)
            for (uint32_t c = 0; c < 2 * F; ++c)
                gu[uint64_t(r) * 2 * F + c] = (r == 0) ? 0.f : (r == 1) ? rf(-35.f, 35.f) : rf(-8.f, 8.f);
        for (auto& v : ws) v = rf(0.05f, 1.8f);
        up(d_gu, gu.data(), gu.size() * 4);
        up(d_ws, ws.data(), ws.size() * 4);
        st.zero(d_ab, 0, uint64_t(Ru) * F * 2); st.synchronize();
        DirectPlan p(rt);
        struct { float cu; int32_t lk; } prm{cu, 3};
        for (uint32_t wv = 0; wv < 4; ++wv) {
            const DirectTpcTensor ts[] = {tn(d_gu.address(), f32, {2 * F, Ru}), tn(d_code.address(), i32, {Su}),
                                          tn(d_idx.address(), i32, {Su}), tn(d_ws.address(), f32, {2 * F, E}),
                                          tn(d_ab.address(), bf, {F, Ru}), tn(d_lbuf.address() + uint64_t(wv) * Su * 4, i32, {Su}),
                                          tn(d_lcnt.address() + uint64_t(wv) * 24 * 4, i32, {24}), tn(d_tp.address(), i32, {25})};
            p.add_tpc_grid(isa("ink_act_bf16_gaudi2"), ts, srfv(prm), {24, 1, 1, 1, 1}, 0, 24, -1, 0, wv == 3 ? -1 : int(wv));
        }
        p.prepare();
        p.run();
        std::vector<uint16_t> got(uint64_t(Ru) * F);
        dl(d_ab, got.data(), got.size() * 2);
        uint64_t same = 0, tot = 0, bad = 0;
        int max_ulp = 0;
        double max_abs_small = 0.0;
        for (uint32_t s = 0; s < Su; ++s) {
            const uint32_t r = uint32_t(ref_code[s]), e = uint32_t(h_idx[s]);
            for (uint32_t g = 0; g < F / 128; ++g) {
                for (uint32_t i = 0; i < 128; ++i) {
                    const uint32_t cg = g * 256 + i, cu_i = g * 256 + 128 + i;
                    const double gd = double(from_bf16(to_bf16(gu[uint64_t(r) * 2 * F + cg] * ws[uint64_t(e) * 2 * F + cg])));
                    const double ud = double(from_bf16(to_bf16(gu[uint64_t(r) * 2 * F + cu_i] * ws[uint64_t(e) * 2 * F + cu_i])));
                    const double ref_d = (4.0 * std::tanh(gd * 0.25)) * (1.0 / (1.0 + std::exp(-gd))) * (double(cu) * std::tanh(ud / double(cu)));
                    const uint16_t ref_b = to_bf16(float(ref_d)), got_b = got[uint64_t(r) * F + g * 128 + i];
                    same += bf16_eq(got_b, ref_b);
                    if (std::fabs(ref_d) > 1e-6) {
                        const int u = bf16_ulp(got_b, ref_b);
                        max_ulp = std::max(max_ulp, u);
                        if (u > 1) ++bad;
                    } else {
                        const double ae = std::fabs(double(from_bf16(got_b)) - double(from_bf16(ref_b)));
                        max_abs_small = std::max(max_abs_small, ae);
                        if (ae > 1e-6) ++bad;
                    }
                    ++tot;
                }
            }
        }
        const bool ok = (bad == 0);
        if (!ok) ++failures;
        std::printf("[bf16pbtest] %-28s %s: %llu / %llu exact (%.4f%%), |ref|>1e-6 max_ulp=%d, |ref|<=1e-6 max_abs=%.2e\n",
                    "ink_act_bf16", ok ? "PASS" : "FAIL", (unsigned long long)same, (unsigned long long)tot,
                    100.0 * double(same) / double(tot), max_ulp, max_abs_small);
    }

    // 4. ink_moe_combine_bf16 (vlen=27/32, F32 y, per-row F32 cs)
    {
        std::vector<float> y(uint64_t(Ru) * H), cs(uint64_t(E) * H), h_rw(Su);
        for (auto& v : y) v = rf(-5.f, 5.f);
        for (auto& v : cs) v = rf(0.05f, 1.5f);
        for (auto& v : h_rw) v = rf(0.05f, 0.5f);
        up(d_y, y.data(), y.size() * 4);
        up(d_cs, cs.data(), cs.size() * 4);
        up(d_rw, h_rw.data(), h_rw.size() * 4);
        std::vector<uint16_t> init_out(Tu * H, 0x3c00u);
        up(d_out, init_out.data(), init_out.size() * 2);
        DirectPlan p(rt);
        const DirectTpcTensor ts[] = {tn(d_y.address(), f32, {H, Ru}), tn(d_cs.address(), f32, {H, E}),
                                      tn(d_code.address(), i32, {K, Tu}), tn(d_idx.address(), i32, {K, Tu}),
                                      tn(d_rw.address(), f32, {K, Tu}), tn(d_out.address(), bf, {H, Tu}),
                                      tn(d_vlen.address(), i32, {1})};
        p.add_tpc_grid(isa("ink_moe_combine_bf16_gaudi2"), ts, {}, {Tu, 1, 1, 1, 1}, 0, 24);
        p.prepare();
        p.run();
        std::vector<uint16_t> got(Tu * H);
        dl(d_out, got.data(), got.size() * 2);
        uint64_t same = 0, tail_dirty = 0;
        int max_ulp = 0;
        for (uint32_t t = 0; t < Tu; ++t) {
            for (uint32_t c = 0; c < H; ++c) {
                if (t >= VLu) { tail_dirty += (got[t * H + c] != 0x3c00u); continue; }
                float acc = 0.f;
                for (uint32_t j = 0; j < K; ++j) {
                    const uint32_t r = uint32_t(ref_code[t * K + j]), e = uint32_t(h_idx[t * K + j]);
                    const float yj = from_bf16(to_bf16(y[uint64_t(r) * H + c] * cs[uint64_t(e) * H + c]));
                    acc = std::fma(yj, h_rw[t * K + j], acc);
                }
                const uint16_t ref_b = to_bf16(acc), got_b = got[t * H + c];
                same += bf16_eq(got_b, ref_b);
                max_ulp = std::max(max_ulp, bf16_ulp(got_b, ref_b));
            }
        }
        const uint64_t tot = uint64_t(VLu) * H;
        const bool ok = (same == tot && tail_dirty == 0);
        if (!ok) ++failures;
        std::printf("[bf16pbtest] %-28s %s: %llu / %llu bit-exact (max_ulp=%d, tail_dirty=%llu)\n",
                    "ink_moe_combine_bf16", ok ? "PASS" : "FAIL", (unsigned long long)same,
                    (unsigned long long)tot, max_ulp, (unsigned long long)tail_dirty);
    }
    return failures ? 1 : 0;
}

// Exactness and timing test for the 4 production NVFP4 decode/batched/verify TPC kernels:
// ink_dmoe_up9_nvf4, ink_dmoe_dn9_nvf4, ink_vmoe_up_nvf4, ink_vmoe_dn_nvf4.
static uint8_t fp8_143_exact(float v) {
    if (v == 0.f) return 0;
    const uint8_t sgn = v < 0 ? 0x80 : 0;
    float a = std::fabs(v);
    int e = 0;
    const float m = std::frexp(a, &e);
    int ex = e - 1 + 7;
    if (ex <= 0) return uint8_t(sgn | uint8_t(std::lround(a * 512.f)));
    const int man = int(std::lround((m * 2.f - 1.f) * 8.f));
    return uint8_t(sgn | (ex << 3) | man);
}
static double fp8_e4m3fn_to_f64(uint8_t c) {
    const double sgn = (c & 0x80) ? -1.0 : 1.0;
    const int ex = (c >> 3) & 15, mn = c & 7;
    if (ex == 0) return sgn * std::ldexp(double(mn), -9);
    return sgn * std::ldexp(1.0 + double(mn) / 8.0, ex - 7);
}
static double bf16_to_f64(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return double(f);
}
static double swiglu4_f64(double g, double u, double cu) {
    const double gr = bf16_to_f64(to_bf16(float(g))), ur = bf16_to_f64(to_bf16(float(u)));
    const double g4 = 4.0 * std::tanh(gr * 0.25);
    const double sg = g4 / (1.0 + std::exp(-gr));
    return sg * (cu * std::tanh(ur / cu));
}
static int nvf4decodetest(const std::string& objdir) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Device dev;
    Stream st(dev);
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);

    auto T = [](uint64_t a, DType dt, std::initializer_list<uint32_t> sz) {
        DirectTpcTensor t{};
        t.addr = a; t.dtype = dt; t.dims = uint32_t(sz.size());
        uint32_t i = 0;
        for (auto v : sz) t.sizes[i++] = v;
        for (; i < 5; ++i) t.sizes[i] = 1;
        return t;
    };
    auto upload = [&](const void* p, uint64_t n) {
        auto d = std::make_unique<DeviceBuffer>(dev, n + 256);
        HostBuffer hb(dev, n + 64);
        std::memcpy(hb.data(), p, n);
        st.upload(hb, 0, *d, 0, n);
        st.synchronize();
        return d;
    };
    auto download_u16 = [&](const DeviceBuffer& d, uint64_t cnt) {
        HostBuffer hb(dev, cnt * 2 + 64);
        st.download(d, 0, hb, 0, cnt * 2);
        st.synchronize();
        std::vector<uint16_t> v(cnt);
        std::memcpy(v.data(), hb.data(), cnt * 2);
        return v;
    };
    auto cmp_bf16 = [](const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref,
                       uint64_t& exact, uint64_t& ulp1, uint64_t& ulp1_big, uint64_t& big_n,
                       double& rel_l2, double& max_abs) {
        exact = ulp1 = ulp1_big = big_n = 0;
        max_abs = 0.0;
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const bool eq = (got[i] == ref[i]) || (((got[i] | ref[i]) & 0x7fff) == 0);
            const bool u1 = eq || (std::abs(int(got[i]) - int(ref[i])) <= 1 && ((got[i] ^ ref[i]) & 0x8000) == 0);
            exact += eq;
            ulp1 += u1;
            const double g = bf16_to_f64(got[i]), r = bf16_to_f64(ref[i]);
            if (std::fabs(r) >= 1e-3) { ++big_n; ulp1_big += u1; }
            const double e = std::fabs(g - r);
            if (e > max_abs) max_abs = e;
            num += e * e;
            den += r * r;
        }
        rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    };

    constexpr uint32_t H = 2304, F = 1152, N2 = 2 * F, E = 16, J = 8, Tv = 8;
    static const float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    auto w2col = [](uint32_t c, uint32_t q) { return c + (q < 128u ? 2u * q : 2u * (q - 128u) + 1u); };
    uint64_t seed = 20261125ull;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(seed >> 33); };
    const float cu = 25.f;
    uint32_t cu_srf;
    std::memcpy(&cu_srf, &cu, 4);

    const uint64_t isa_up9_nvf4 = rt.load_tpc_elf(objdir + "/ink_dmoe_up9_nvf4_gaudi2.o");
    const uint64_t isa_dn9_nvf4 = rt.load_tpc_elf(objdir + "/ink_dmoe_dn9_nvf4_gaudi2.o");
    const uint64_t isa_plan     = rt.load_tpc_elf(objdir + "/ink_vmoe_plan_gaudi2.o");
    const uint64_t isa_vup_nvf4 = rt.load_tpc_elf(objdir + "/ink_vmoe_up_nvf4_gaudi2.o");
    const uint64_t isa_vdn_nvf4 = rt.load_tpc_elf(objdir + "/ink_vmoe_dn_nvf4_gaudi2.o");

    // Routing idx [J, Tv]: covers nm=1..4, cnt>4 splits (7->4+3, 5->4+1), and token 7 padded with -1.
    std::vector<int32_t> h_idx(J * Tv, -1);
    const int32_t pat[7][8] = {
        {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 8, 9, 10}, {0, 1, 2, 3, 5, 11, 12, 13},
        {0, 1, 2, 6, 8, 11, 14, 15}, {0, 1, 5, 6, 9, 12, 14, 7}, {0, 8, 9, 10, 11, 13, 15, 7},
        {0, 10, 12, 13, 14, 15, 5, 6},
    };
    for (uint32_t t = 0; t < 7; ++t)
        for (uint32_t j = 0; j < J; ++j) h_idx[t * J + j] = pat[t][j];
    auto didx8 = upload(h_idx.data(), h_idx.size() * 4);
    DeviceBuffer dgrp(dev, 64 * 4 * 4 + 256);
    st.zero(dgrp, 0, 64 * 4 * 4); st.synchronize();
    {
        DirectPlan pp(rt);
        const DirectTpcTensor tp[] = {T(didx8->address(), DType::I32, {J, Tv}), T(dgrp.address(), DType::I32, {64, 4})};
        pp.add_tpc_grid(isa_plan, tp, {}, {1, 1, 1, 1, 1}, 0, 1);
        pp.prepare(); pp.run();
    }

    // Input activations x [Tv, H] (token 2 has 320x high-DR outliers)
    std::vector<uint16_t> h_x(uint64_t(Tv) * H);
    for (uint32_t t = 0; t < Tv; ++t)
        for (uint32_t k = 0; k < H; ++k) {
            const float v = (t == 2) ? ((k % 16 == 0) ? ((k & 16) ? 16.f : -16.f) : (float(int(rnd() % 201) - 100) / 2000.f))
                                     : (float(int(rnd() % 2001) - 1000) / 500.f);
            h_x[uint64_t(t) * H + k] = to_bf16(v);
        }
    auto dx = upload(h_x.data(), h_x.size() * 2);
    DeviceBuffer dact(dev, uint64_t(Tv) * J * F * 2 + 256), dys(dev, uint64_t(Tv) * J * H * 2 + 256);

    int failures = 0;
    auto check_res = [&](const char* label, const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref,
                         uint32_t pad_rows, double us) {
        uint64_t ex, u1, u1b, bn; double rl2, ma;
        cmp_bf16(got, ref, ex, u1, u1b, bn, rl2, ma);
        uint64_t pad_nz = 0;
        if (pad_rows) {
            const size_t stride = got.size() / Tv;
            for (size_t i = (Tv - pad_rows) * stride; i < got.size(); ++i) pad_nz += (got[i] != 0);
        }
        const double u1b_pct = bn ? 100.0 * double(u1b) / double(bn) : 100.0;
        const bool ok = (rl2 < 2e-3 && u1b_pct > 99.5 && pad_nz == 0);
        if (!ok) ++failures;
        if (us > 0.0)
            std::printf("  %-30s exact %6.2f%% <=1ulp(|r|>=1e-3) %6.2f%% relL2 %.2e padNZ=%llu | %6.1f us %s\n",
                        label, 100.0 * double(ex) / double(got.size()), u1b_pct, rl2,
                        (unsigned long long)pad_nz, us, ok ? "PASS" : "FAIL");
        else
            std::printf("  %-30s exact %6.2f%% <=1ulp(|r|>=1e-3) %6.2f%% relL2 %.2e padNZ=%llu %s\n",
                        label, 100.0 * double(ex) / double(got.size()), u1b_pct, rl2,
                        (unsigned long long)pad_nz, ok ? "PASS" : "FAIL");
    };
    auto run_grid = [&](uint64_t isa, std::span<const DirectTpcTensor> ts, std::span<const uint32_t> sr,
                        uint32_t pts, uint32_t ntpc, DeviceBuffer& out, uint64_t n_u16, int nruns) {
        st.zero(out, 0, n_u16 * 2); st.synchronize();
        DirectPlan p(rt);
        p.add_tpc_grid(isa, ts, sr, {pts, 1, 1, 1, 1}, 0, ntpc);
        p.prepare();
        std::vector<double> tv;
        for (int r = 0; r < nruns; ++r) tv.push_back(p.run());
        std::sort(tv.begin(), tv.end());
        return std::make_pair(tv[tv.size() / 2], download_u16(out, n_u16));
    };

    // Arbitrary Inkling E4M3FN scales [6..448] (25% > 240, constant si/expert) vs host FP64 (1 & 24 TPC)
    std::printf("[nvf4decodetest] arbitrary E4M3FN scales [6..448] (incl >240) vs FP64 host ref (1 & 24 TPC)\n");
    {
        std::vector<uint8_t> nib13(uint64_t(E) * H * N2), wq_13(nib13.size() / 2), bs13_raw(nib13.size() / 16), bs13_f8(nib13.size() / 16);
        for (auto& v : nib13) v = uint8_t(rnd() & 15);
        for (size_t i = 0; i < bs13_raw.size(); ++i) {
            const uint8_t code = (i % 4 == 0) ? uint8_t(0x78 + (rnd() % 7)) : uint8_t(0x4c + (rnd() % (0x7f - 0x4c)));
            bs13_raw[i] = code;
            bs13_f8[i] = uint8_t(code - 0x18);
        }
        for (uint32_t e = 0; e < E; ++e)
            for (uint32_t k = 0; k < H; k += 2)
                for (uint32_t n = 0; n < N2; ++n) {
                    const uint64_t i = (uint64_t(e) * H + k) * N2 + n;
                    wq_13[(uint64_t(e) * (H / 2) + k / 2) * N2 + n] = uint8_t(nib13[i] | (nib13[i + N2] << 4));
                }
        std::vector<uint8_t> nib2(uint64_t(E) * F * H), wq_2(nib2.size() / 2), bs2_raw(nib2.size() / 16), bs2_f8(nib2.size() / 16);
        for (auto& v : nib2) v = uint8_t(rnd() & 15);
        for (uint32_t e = 0; e < E; ++e)
            for (uint32_t kb = 0; kb < F / 16; ++kb)
                for (uint32_t c = 0; c < H; c += 256)
                    for (uint32_t q = 0; q < 256; ++q) {
                        const uint8_t code = ((kb + q) % 4 == 0) ? uint8_t(0x78 + (rnd() % 7)) : uint8_t(0x4c + (rnd() % (0x7f - 0x4c)));
                        bs2_raw[(uint64_t(e) * (F / 16) + kb) * H + c + q] = code;
                        bs2_f8 [(uint64_t(e) * (F / 16) + kb) * H + w2col(c, q)] = uint8_t(code - 0x18);
                    }
        for (uint32_t e = 0; e < E; ++e)
            for (uint32_t k = 0; k < F; k += 2)
                for (uint32_t c = 0; c < H; c += 256)
                    for (uint32_t q = 0; q < 256; ++q) {
                        const uint32_t bcol = w2col(c, q), lcol = c + q;
                        const uint8_t n0 = nib2[(uint64_t(e) * F + k) * H + lcol];
                        const uint8_t n1 = nib2[(uint64_t(e) * F + k + 1) * H + lcol];
                        wq_2[(uint64_t(e) * (F / 2) + k / 2) * H + bcol] = uint8_t(n0 | (n1 << 4));
                    }
        std::vector<float> g13_e(E), g2_e(E), si13_5(uint64_t(E) * N2), si2_5(uint64_t(E) * H);
        for (uint32_t e = 0; e < E; ++e) {
            g13_e[e] = 1.5e-4f + float(e) * 1.1e-5f;
            g2_e[e]  = 1.8e-4f + float(e) * 1.3e-5f;
            std::fill_n(si13_5.begin() + e * N2, N2, g13_e[e] * 512.f);
            std::fill_n(si2_5.begin()  + e * H,  H,  g2_e[e]  * 512.f);
        }
        std::vector<uint16_t> ref_up(uint64_t(Tv) * J * F, 0), ref_dn(uint64_t(Tv) * J * H, 0);
        for (uint32_t t = 0; t < Tv; ++t) {
            for (uint32_t j = 0; j < J; ++j) {
                const int32_t e = h_idx[t * J + j];
                if (e < 0) continue;
                const uint64_t ba = (uint64_t(t) * J + j) * F;
                for (uint32_t f = 0; f < F; ++f) {
                    double ag = 0.0, au = 0.0;
                    const uint32_t ng = 2 * f, nu = 2 * f + 1;
                    for (uint32_t k = 0; k < H; ++k) {
                        const double xv = bf16_to_f64(h_x[uint64_t(t) * H + k]);
                        ag += xv * (double(kE2M1[nib13[(uint64_t(e) * H + k) * N2 + ng]]) *
                                    fp8_e4m3fn_to_f64(bs13_raw[(uint64_t(e) * (H / 16) + k / 16) * N2 + ng]));
                        au += xv * (double(kE2M1[nib13[(uint64_t(e) * H + k) * N2 + nu]]) *
                                    fp8_e4m3fn_to_f64(bs13_raw[(uint64_t(e) * (H / 16) + k / 16) * N2 + nu]));
                    }
                    ref_up[ba + f] = to_bf16(float(swiglu4_f64(ag * double(g13_e[e]), au * double(g13_e[e]), double(cu))));
                }
                const uint64_t by = (uint64_t(t) * J + j) * H;
                for (uint32_t n = 0; n < H; ++n) {
                    double acc = 0.0;
                    for (uint32_t k = 0; k < F; ++k)
                        acc += bf16_to_f64(ref_up[ba + k]) *
                               (double(kE2M1[nib2[(uint64_t(e) * F + k) * H + n]]) *
                                fp8_e4m3fn_to_f64(bs2_raw[(uint64_t(e) * (F / 16) + k / 16) * H + n]));
                    ref_dn[by + n] = to_bf16(float(acc * double(g2_e[e])));
                }
            }
        }
        auto dwq13 = upload(wq_13.data(), wq_13.size()), dbs13 = upload(bs13_f8.data(), bs13_f8.size()),
             dsi13 = upload(si13_5.data(), si13_5.size() * 4);
        auto dwq2 = upload(wq_2.data(), wq_2.size()), dbs2 = upload(bs2_f8.data(), bs2_f8.size()),
             dsi2 = upload(si2_5.data(), si2_5.size() * 4);
        auto dact_ref = upload(ref_up.data(), ref_up.size() * 2);
        const std::span<const uint32_t> sr_cu(&cu_srf, 1);

        // dmoe_up9_nvf4 & dmoe_dn9_nvf4 on token 0 (idx row 0 = {0..7})
        std::vector<uint16_t> ref_up0(ref_up.begin(), ref_up.begin() + J * F);
        std::vector<uint16_t> ref_dn0(ref_dn.begin(), ref_dn.begin() + J * H);
        const DirectTpcTensor t_up9[] = {
            T(dx->address(), DType::BF16, {H, 1}), T(didx8->address(), DType::I32, {J, 1}),
            T(dwq13->address(), DType::U8, {N2, H / 2, E}), T(dbs13->address(), DType::F8, {N2, H / 16, E}),
            T(dsi13->address(), DType::F32, {N2, E}), T(dact.address(), DType::BF16, {F, J, 1})};
        const DirectTpcTensor t_dn9[] = {
            T(dact_ref->address(), DType::BF16, {F, J, 1}), T(didx8->address(), DType::I32, {J, 1}),
            T(dwq2->address(), DType::U8, {H, F / 2, E}), T(dbs2->address(), DType::F8, {H, F / 16, E}),
            T(dsi2->address(), DType::F32, {H, E}), T(dys.address(), DType::BF16, {H, J, 1})};
        const DirectTpcTensor t_vup[] = {
            T(dx->address(), DType::BF16, {H, Tv}), T(dgrp.address(), DType::I32, {64, 4}),
            T(dwq13->address(), DType::U8, {N2, H / 2, E}), T(dbs13->address(), DType::F8, {N2, H / 16, E}),
            T(dsi13->address(), DType::F32, {N2, E}), T(dact.address(), DType::BF16, {F, J, Tv})};
        const DirectTpcTensor t_vdn[] = {
            T(dact_ref->address(), DType::BF16, {F, J, Tv}), T(dgrp.address(), DType::I32, {64, 4}),
            T(dwq2->address(), DType::U8, {H, F / 2, E}), T(dbs2->address(), DType::F8, {H, F / 16, E}),
            T(dsi2->address(), DType::F32, {H, E}), T(dys.address(), DType::BF16, {H, J, Tv})};
        for (uint32_t ntpc : {1u, 24u}) {
            const int nr = (ntpc == 24u) ? 21 : 1;
            const std::string tag = (ntpc == 24u) ? "[24TPC] " : "[ 1TPC] ";
            auto r_up9 = run_grid(isa_up9_nvf4, t_up9, sr_cu, 9 * J, ntpc, dact, uint64_t(F) * J, nr);
            auto r_dn9 = run_grid(isa_dn9_nvf4, t_dn9, {},    9 * J, ntpc, dys,  uint64_t(H) * J, nr);
            auto r_vup = run_grid(isa_vup_nvf4, t_vup, sr_cu, 9 * J * Tv, ntpc, dact, uint64_t(Tv) * J * F, nr);
            auto r_vdn = run_grid(isa_vdn_nvf4, t_vdn, {},    9 * J * Tv, ntpc, dys,  uint64_t(Tv) * J * H, nr);
            check_res((tag + "ink_dmoe_up9_nvf4").c_str(), r_up9.second, ref_up0, 0, ntpc == 24u ? r_up9.first : 0);
            check_res((tag + "ink_dmoe_dn9_nvf4").c_str(), r_dn9.second, ref_dn0, 0, ntpc == 24u ? r_dn9.first : 0);
            check_res((tag + "ink_vmoe_up_nvf4").c_str(),  r_vup.second, ref_up,  1, ntpc == 24u ? r_vup.first : 0);
            check_res((tag + "ink_vmoe_dn_nvf4").c_str(),  r_vdn.second, ref_dn,  1, ntpc == 24u ? r_vdn.first : 0);
        }
    }
    std::printf("[nvf4decodetest] %s (failures=%d)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}

// Exactness and timing test for the 4 native MXFP8 decode/batched/verify TPC kernels:
// ink_dmoe_up9_mxf8, ink_dmoe_dn9_mxf8, ink_vmoe_up_mxf8, ink_vmoe_dn_mxf8.
static int mxf8decodetest(const std::string& objdir, const std::string& routing_bin, bool verify_only) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    Device dev;
    Stream st(dev);
    synapse_warmup(dev, st);
    DirectRuntime rt(dev);
    rt.set_options(kInkOptions);

    auto T = [](uint64_t a, DType dt, std::initializer_list<uint32_t> sz) {
        DirectTpcTensor t{};
        t.addr = a; t.dtype = dt; t.dims = uint32_t(sz.size());
        uint32_t i = 0;
        for (auto v : sz) t.sizes[i++] = v;
        for (; i < 5; ++i) t.sizes[i] = 1;
        return t;
    };
    auto upload = [&](const void* p, uint64_t n) {
        auto d = std::make_unique<DeviceBuffer>(dev, n + 256);
        const uint64_t chunk = std::min<uint64_t>(n, 32ull << 20);
        HostBuffer hb(dev, chunk + 64);
        for (uint64_t off = 0; off < n; off += chunk) {
            const uint64_t len = std::min<uint64_t>(chunk, n - off);
            std::memcpy(hb.data(), static_cast<const uint8_t*>(p) + off, len);
            st.upload(hb, 0, *d, off, len);
            st.synchronize();
        }
        return d;
    };
    auto upload_fill_pattern = [&](uint64_t n, uint8_t seed_byte, uint8_t max_exp_mask) {
        auto d = std::make_unique<DeviceBuffer>(dev, n + 256);
        const uint64_t chunk = std::min<uint64_t>(n, 32ull << 20);
        HostBuffer hb(dev, chunk + 64);
        auto* p = static_cast<uint8_t*>(hb.data());
        for (uint64_t i = 0; i < chunk; ++i)
            p[i] = uint8_t((((i * 131u + seed_byte) & max_exp_mask) | 0x08u) | ((max_exp_mask > 0x38u) ? ((i & 1u) << 7) : 0u));
        for (uint64_t off = 0; off < n; off += chunk) {
            const uint64_t len = std::min<uint64_t>(chunk, n - off);
            st.upload(hb, 0, *d, off, len);
            st.synchronize();
        }
        return d;
    };
    auto download_u16 = [&](const DeviceBuffer& d, uint64_t cnt) {
        HostBuffer hb(dev, cnt * 2 + 64);
        st.download(d, 0, hb, 0, cnt * 2);
        st.synchronize();
        std::vector<uint16_t> v(cnt);
        std::memcpy(v.data(), hb.data(), cnt * 2);
        return v;
    };
    auto cmp_bf16 = [](const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref,
                       uint64_t& exact, uint64_t& ulp1, uint64_t& ulp1_big, uint64_t& big_n,
                       double& rel_l2, double& max_abs) {
        exact = ulp1 = ulp1_big = big_n = 0;
        max_abs = 0.0;
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const bool eq = (got[i] == ref[i]) || (((got[i] | ref[i]) & 0x7fff) == 0);
            const bool u1 = eq || (std::abs(int(got[i]) - int(ref[i])) <= 1 && ((got[i] ^ ref[i]) & 0x8000) == 0);
            exact += eq;
            ulp1 += u1;
            const double g = bf16_to_f64(got[i]), r = bf16_to_f64(ref[i]);
            if (std::fabs(r) >= 1e-3) { ++big_n; ulp1_big += u1; }
            const double e = std::fabs(g - r);
            if (e > max_abs) max_abs = e;
            num += e * e;
            den += r * r;
        }
        rel_l2 = den > 0.0 ? std::sqrt(num / den) : 0.0;
    };
    auto run_grid = [&](uint64_t isa, std::span<const DirectTpcTensor> ts, std::span<const uint32_t> sr,
                        uint32_t pts, uint32_t ntpc, DeviceBuffer& out, uint64_t n_u16, int nruns) {
        st.zero(out, 0, n_u16 * 2); st.synchronize();
        DirectPlan p(rt);
        p.add_tpc_grid(isa, ts, sr, {pts, 1, 1, 1, 1}, 0, ntpc);
        p.prepare();
        std::vector<double> tv;
        for (int r = 0; r < nruns; ++r) tv.push_back(p.run());
        std::sort(tv.begin(), tv.end());
        return std::make_pair(tv[tv.size() / 2], download_u16(out, n_u16));
    };

    constexpr uint32_t H = 2304, F = 1152, N2 = 2 * F, J = 8;
    auto w2col = [](uint32_t c, uint32_t q) { return c + (q < 128u ? 2u * q : 2u * (q - 128u) + 1u); };
    uint64_t seed = 2026112502ull;
    auto rnd = [&]() { seed = seed * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(seed >> 33); };
    // Realistic MXFP8 weight code matching scan_mxfp8.log (~10% e=15, 0 NaNs, ~1% subnormals/zeros, ~89% e=1..14)
    auto rand_mxf8_code = [&]() -> uint8_t {
        const uint32_t u = rnd() % 1000u;
        const uint8_t sgn = uint8_t((rnd() & 1u) << 7);
        if (u < 100u) return uint8_t(sgn | 0x78u | (rnd() % 7u));           // e=15, mant 0..6 (no NaN)
        if (u < 110u) return uint8_t(sgn | (rnd() & 7u));                   // subnormal or +/-0
        return uint8_t(sgn | ((1u + (rnd() % 14u)) << 3) | (rnd() & 7u));   // normal e=1..14
    };
    const float cu = 25.f;
    uint32_t cu_srf;
    std::memcpy(&cu_srf, &cu, 4);
    const std::span<const uint32_t> sr_cu(&cu_srf, 1);

    const uint64_t isa_up9_mxf8 = rt.load_tpc_elf(objdir + "/ink_dmoe_up9_mxf8_gaudi2.o");
    const uint64_t isa_dn9_mxf8 = rt.load_tpc_elf(objdir + "/ink_dmoe_dn9_mxf8_gaudi2.o");
    const uint64_t isa_vup_mxf8 = rt.load_tpc_elf(objdir + "/ink_vmoe_up_mxf8_gaudi2.o");
    const uint64_t isa_vdn_mxf8 = rt.load_tpc_elf(objdir + "/ink_vmoe_dn_mxf8_gaudi2.o");
    const uint64_t isa_plan     = rt.load_tpc_elf(objdir + "/ink_vmoe_plan_gaudi2.o");

    int failures = 0;

    // =========================================================================
    // Part 0: Smallest case (1 TPC) - ALL 256 byte codes (0x00..0xFF) x 9 scales
    //         across boundary k in {0, 1, 31, K-1}, all 4 kernels and nm=1..4.
    // =========================================================================
    std::printf("[mxf8decodetest] Part 0: 1-TPC exhaustive 256 codes (0x00..0xFF) x 9 scales (k={0,1,31,K-1}, nm=1..4)\n");
    {
        constexpr uint32_t E0 = 8, T0 = 4, J0 = 4;
        // 7 pow2 scales spanning checkpoint range 2^-6..2^0 + 2 non-pow2 scales (1.25, 3.25)
        const uint8_t bs9[9] = {0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x3A, 0x45};
        // 4 distinct k-blocks (kb=0,1,2,NB-1) covering intra-block offsets 0, 1, 31, and final k=K-1
        const uint32_t kh_up[4] = {0, 33, 95, H - 1};
        const uint32_t kh_dn[4] = {0, 33, 95, F - 1};

        // --- w2 exhaustive: at each kh_dn[t], all 256 codes in each of the 9 groups with bs9[g] ---
        std::vector<uint8_t> w2_wq(uint64_t(E0) * F * H);
        std::vector<uint8_t> w2_bs(uint64_t(E0) * (F / 32) * H, 0x38);
        for (size_t i = 0; i < w2_wq.size(); ++i) w2_wq[i] = uint8_t((i * 37u + 19u) & 0xffu);
        for (uint32_t e = 0; e < E0; ++e) {
            for (uint32_t t = 0; t < T0; ++t) {
                const uint32_t k = kh_dn[t], kb = k / 32;
                for (uint32_t g = 0; g < 9; ++g) {
                    const uint32_t c = g * 256;
                    for (uint32_t q = 0; q < 256; ++q) {
                        const uint32_t bcol = w2col(c, q);
                        w2_wq[(uint64_t(e) * F + k) * H + bcol] = uint8_t(q);
                        w2_bs[(uint64_t(e) * (F / 32) + kb) * H + bcol] = bs9[g];
                    }
                }
            }
        }
        // --- w13 exhaustive: kh_up[0]: gate=0..127; [1]: gate=128..255; [2]: up=0..127; [3]: up=128..255 ---
        // Use checkpoint-like g13 = 2^-9 (with partner code 0x70 = 128 => partner val = 0.25) so all 256 codes
        // including E=15 (-480..+480) stay strictly inside SWIGLU4's unclamped domain (|val| <= 3.05 < 26).
        const float g13_0 = std::ldexp(1.0f, -9);
        std::vector<uint8_t> w13_wq(uint64_t(E0) * H * N2);
        std::vector<uint8_t> w13_bs(uint64_t(E0) * (H / 32) * N2, 0x38);
        for (size_t i = 0; i < w13_wq.size(); ++i) w13_wq[i] = uint8_t((i * 53u + 11u) & 0xffu);
        for (uint32_t e = 0; e < E0; ++e) {
            for (uint32_t t = 0; t < T0; ++t) {
                const uint32_t k = kh_up[t], kb = k / 32;
                for (uint32_t g = 0; g < 9; ++g) {
                    for (uint32_t p = 0; p < 128; ++p) {
                        const uint32_t ng = g * 256 + 2 * p, nu = ng + 1;
                        const uint8_t cg = (t == 0) ? uint8_t(p) : (t == 1) ? uint8_t(128 + p) : uint8_t(0x70);
                        const uint8_t cu_c = (t == 2) ? uint8_t(p) : (t == 3) ? uint8_t(128 + p) : uint8_t(0x70);
                        const uint8_t bsg = (t < 2) ? bs9[g] : uint8_t(0x38);
                        const uint8_t bsu = (t >= 2) ? bs9[g] : uint8_t(0x38);
                        w13_wq[(uint64_t(e) * H + k) * N2 + ng] = cg;
                        w13_wq[(uint64_t(e) * H + k) * N2 + nu] = cu_c;
                        w13_bs[(uint64_t(e) * (H / 32) + kb) * N2 + ng] = bsg;
                        w13_bs[(uint64_t(e) * (H / 32) + kb) * N2 + nu] = bsu;
                    }
                }
            }
        }
        std::vector<float> si13(uint64_t(E0) * N2, g13_0), si2(uint64_t(E0) * H, 1.0f);
        // Single-hot activations: x0[t, kh_up[t]] = 1.0, act0[t, j, kh_dn[t]] = 1.0
        std::vector<uint16_t> x0(uint64_t(T0) * H, 0), act0(uint64_t(T0) * J0 * F, 0);
        for (uint32_t t = 0; t < T0; ++t) {
            x0[uint64_t(t) * H + kh_up[t]] = to_bf16(1.0f);
            for (uint32_t j = 0; j < J0; ++j)
                act0[(uint64_t(t) * J0 + j) * F + kh_dn[t]] = to_bf16(1.0f);
        }
        auto dx0 = upload(x0.data(), x0.size() * 2);
        auto dact0 = upload(act0.data(), act0.size() * 2);
        auto dw13_wq = upload(w13_wq.data(), w13_wq.size()), dw13_bs = upload(w13_bs.data(), w13_bs.size());
        auto dw2_wq  = upload(w2_wq.data(),  w2_wq.size()),  dw2_bs  = upload(w2_bs.data(),  w2_bs.size());
        auto dsi13   = upload(si13.data(), si13.size() * 4), dsi2    = upload(si2.data(), si2.size() * 4);
        DeviceBuffer dout_up(dev, uint64_t(T0) * J0 * F * 2 + 256);
        DeviceBuffer dout_dn(dev, uint64_t(T0) * J0 * H * 2 + 256);

        // Expected reference vectors for one slot of token t
        std::vector<uint16_t> ref_dn_one(H);
        for (uint32_t g = 0; g < 9; ++g)
            for (uint32_t q = 0; q < 256; ++q)
                ref_dn_one[g * 256 + q] = to_bf16(float(fp8_e4m3fn_to_f64(uint8_t(q)) * fp8_e4m3fn_to_f64(bs9[g])));
        std::vector<uint16_t> ref_up_t[4];
        for (uint32_t t = 0; t < 4; ++t) {
            ref_up_t[t].resize(F);
            for (uint32_t g = 0; g < 9; ++g) {
                for (uint32_t p = 0; p < 128; ++p) {
                    const uint8_t cg = (t == 0) ? uint8_t(p) : (t == 1) ? uint8_t(128 + p) : uint8_t(0x70);
                    const uint8_t cu_c = (t == 2) ? uint8_t(p) : (t == 3) ? uint8_t(128 + p) : uint8_t(0x70);
                    const double gv = fp8_e4m3fn_to_f64(cg) * fp8_e4m3fn_to_f64((t < 2) ? bs9[g] : uint8_t(0x38)) * double(g13_0);
                    const double uv = fp8_e4m3fn_to_f64(cu_c) * fp8_e4m3fn_to_f64((t >= 2) ? bs9[g] : uint8_t(0x38)) * double(g13_0);
                    ref_up_t[t][g * 128 + p] = to_bf16(float(swiglu4_f64(gv, uv, double(cu))));
                }
            }
        }

        // 0a: Literally smallest case: 1 TPC, 1 expert (J=1, T=1) on dmoe_dn9_mxf8 + (J=1, T=4) covering all 4 kh_dn
        const int32_t idx_t4[4] = {0, 1, 2, 3};
        auto didx_t4 = upload(idx_t4, sizeof(idx_t4));
        {
            const DirectTpcTensor t_dn1[] = {
                T(dact0->address(), DType::BF16, {F, 1, 1}), T(didx_t4->address(), DType::I32, {1, 1}),
                T(dw2_wq->address(), DType::U8, {H, F, E0}), T(dw2_bs->address(), DType::F8, {H, F / 32, E0}),
                T(dsi2->address(), DType::F32, {H, E0}), T(dout_dn.address(), DType::BF16, {H, 1, 1})};
            auto got = run_grid(isa_dn9_mxf8, t_dn1, {}, 9, 1, dout_dn, H, 1).second;
            uint64_t mis = 0;
            for (uint32_t i = 0; i < H; ++i) {
                const bool eq = (got[i] == ref_dn_one[i]) || (((got[i] | ref_dn_one[i]) & 0x7fff) == 0);
                if (!eq) {
                    if (mis < 12)
                        std::printf("    MISMATCH dn9(1x1) g=%u q=0x%02x bs=0x%02x got=0x%04x(%.6g) ref=0x%04x(%.6g)\n",
                                    i / 256, i % 256, bs9[i / 256], got[i], bf16_to_f64(got[i]), ref_dn_one[i], bf16_to_f64(ref_dn_one[i]));
                    ++mis;
                }
            }
            if (mis) ++failures;
            std::printf("  [1TPC 1exp] ink_dmoe_dn9_mxf8  256x9=%u codes: exact=%llu/%u (mis=%llu) %s\n",
                        H, (unsigned long long)(H - mis), H, (unsigned long long)mis, mis == 0 ? "PASS" : "FAIL");
        }
        // 0b: dmoe_dn9_mxf8 & dmoe_up9_mxf8 on (J=1, T=4) covering kh={0,1,31,K-1}
        {
            const DirectTpcTensor t_dn4[] = {
                T(dact0->address(), DType::BF16, {F, J0, T0}), T(didx_t4->address(), DType::I32, {1, T0}),
                T(dw2_wq->address(), DType::U8, {H, F, E0}), T(dw2_bs->address(), DType::F8, {H, F / 32, E0}),
                T(dsi2->address(), DType::F32, {H, E0}), T(dout_dn.address(), DType::BF16, {H, 1, T0})};
            // Note: act0 has J0=4 stride; build compact J=1 act for T0=4
            std::vector<uint16_t> act_j1(uint64_t(T0) * F, 0);
            for (uint32_t t = 0; t < T0; ++t) act_j1[uint64_t(t) * F + kh_dn[t]] = to_bf16(1.0f);
            auto dact_j1 = upload(act_j1.data(), act_j1.size() * 2);
            DirectTpcTensor t_dn4c[6];
            std::memcpy(t_dn4c, t_dn4, sizeof(t_dn4));
            t_dn4c[0] = T(dact_j1->address(), DType::BF16, {F, 1, T0});
            auto got_dn = run_grid(isa_dn9_mxf8, t_dn4c, {}, 9 * T0, 1, dout_dn, uint64_t(T0) * H, 1).second;
            std::vector<uint16_t> ref_dn4(uint64_t(T0) * H);
            for (uint32_t t = 0; t < T0; ++t) std::copy(ref_dn_one.begin(), ref_dn_one.end(), ref_dn4.begin() + t * H);
            uint64_t ex, u1, u1b, bn; double rl2, ma;
            cmp_bf16(got_dn, ref_dn4, ex, u1, u1b, bn, rl2, ma);
            const bool ok_dn = (ex == ref_dn4.size());
            if (!ok_dn) ++failures;
            std::printf("  [1TPC 4k  ] ink_dmoe_dn9_mxf8  exact=%llu/%zu maxAbs=%.2e %s\n",
                        (unsigned long long)ex, ref_dn4.size(), ma, ok_dn ? "PASS" : "FAIL");

            const DirectTpcTensor t_up4[] = {
                T(dx0->address(), DType::BF16, {H, T0}), T(didx_t4->address(), DType::I32, {1, T0}),
                T(dw13_wq->address(), DType::U8, {N2, H, E0}), T(dw13_bs->address(), DType::F8, {N2, H / 32, E0}),
                T(dsi13->address(), DType::F32, {N2, E0}), T(dout_up.address(), DType::BF16, {F, 1, T0})};
            auto got_up = run_grid(isa_up9_mxf8, t_up4, sr_cu, 9 * T0, 1, dout_up, uint64_t(T0) * F, 1).second;
            std::vector<uint16_t> ref_up4(uint64_t(T0) * F);
            for (uint32_t t = 0; t < T0; ++t) std::copy(ref_up_t[t].begin(), ref_up_t[t].end(), ref_up4.begin() + t * F);
            cmp_bf16(got_up, ref_up4, ex, u1, u1b, bn, rl2, ma);
            const bool ok_up = (u1 == ref_up4.size() && rl2 < 1e-3);
            if (!ok_up) {
                ++failures;
                int shown = 0;
                for (size_t i = 0; i < ref_up4.size() && shown < 10; ++i) {
                    const bool eq = (got_up[i] == ref_up4[i]) || (((got_up[i] | ref_up4[i]) & 0x7fff) == 0);
                    const bool is_u1 = eq || (std::abs(int(got_up[i]) - int(ref_up4[i])) <= 1 && ((got_up[i] ^ ref_up4[i]) & 0x8000) == 0);
                    if (!is_u1) {
                        std::printf("    MISMATCH up9 t=%zu g=%zu p=%zu got=0x%04x(%.6g) ref=0x%04x(%.6g)\n",
                                    i / F, (i % F) / 128, i % 128, got_up[i], bf16_to_f64(got_up[i]), ref_up4[i], bf16_to_f64(ref_up4[i]));
                        ++shown;
                    }
                }
            }
            std::printf("  [1TPC 4k  ] ink_dmoe_up9_mxf8  exact=%6.2f%% <=1ulp=%llu/%zu relL2=%.2e maxAbs=%.2e %s\n",
                        100.0 * double(ex) / double(ref_up4.size()), (unsigned long long)u1, ref_up4.size(),
                        rl2, ma, ok_up ? "PASS" : "FAIL");
        }
        // 0c: vmoe_dn_mxf8 & vmoe_up_mxf8 on 1 TPC exercising ALL nm=1,2,3,4 on all 256 codes x 9 scales x boundary k
        {
            // grp [64, 4] (row-major in memory: grp[row * 64 + u], dims {64, 4})
            // 7 active groups (13 active slots out of J0*T0=16, 3 unused slots):
            //   u=0..3: nm=1 (e=0..3 on t=0..3, slot=0) -> tests all 4 kh on nm=1
            //   u=4:    nm=2 (e=4 on t=0,1, slot=1)
            //   u=5:    nm=3 (e=5 on t=0,1,2, slot=2)
            //   u=6:    nm=4 (e=6 on t=0,1,2,3, slot=3)
            std::vector<int32_t> grp0(64 * 4, 0);
            for (uint32_t u = 0; u < 64; ++u) grp0[u] = -1;
            auto set_grp = [&](uint32_t u, int32_t e, int32_t cnt, std::initializer_list<std::pair<int,int>> ts) {
                grp0[0 * 64 + u] = e;
                grp0[1 * 64 + u] = cnt;
                int32_t tsp = 0, m = 0;
                for (auto [t, s] : ts) tsp |= (((t << 4) | s) << (8 * m++));
                grp0[2 * 64 + u] = tsp;
            };
            set_grp(0, 0, 1, {{0, 0}});
            set_grp(1, 1, 1, {{1, 0}});
            set_grp(2, 2, 1, {{2, 0}});
            set_grp(3, 3, 1, {{3, 0}});
            set_grp(4, 4, 2, {{0, 1}, {1, 1}});
            set_grp(5, 5, 3, {{0, 2}, {1, 2}, {2, 2}});
            set_grp(6, 6, 4, {{0, 3}, {1, 3}, {2, 3}, {3, 3}});
            auto dgrp0 = upload(grp0.data(), grp0.size() * 4);

            const DirectTpcTensor t_vdn0[] = {
                T(dact0->address(), DType::BF16, {F, J0, T0}), T(dgrp0->address(), DType::I32, {64, 4}),
                T(dw2_wq->address(), DType::U8, {H, F, E0}), T(dw2_bs->address(), DType::F8, {H, F / 32, E0}),
                T(dsi2->address(), DType::F32, {H, E0}), T(dout_dn.address(), DType::BF16, {H, J0, T0})};
            auto got_vdn = run_grid(isa_vdn_mxf8, t_vdn0, {}, 9 * J0 * T0, 1, dout_dn, uint64_t(T0) * J0 * H, 1).second;

            const DirectTpcTensor t_vup0[] = {
                T(dx0->address(), DType::BF16, {H, T0}), T(dgrp0->address(), DType::I32, {64, 4}),
                T(dw13_wq->address(), DType::U8, {N2, H, E0}), T(dw13_bs->address(), DType::F8, {N2, H / 32, E0}),
                T(dsi13->address(), DType::F32, {N2, E0}), T(dout_up.address(), DType::BF16, {F, J0, T0})};
            auto got_vup = run_grid(isa_vup_mxf8, t_vup0, sr_cu, 9 * J0 * T0, 1, dout_up, uint64_t(T0) * J0 * F, 1).second;

            const std::pair<int, std::vector<std::pair<int,int>>> nm_slots[4] = {
                {1, {{0,0},{1,0},{2,0},{3,0}}},
                {2, {{0,1},{1,1}}},
                {3, {{0,2},{1,2},{2,2}}},
                {4, {{0,3},{1,3},{2,3},{3,3}}},
            };
            for (const auto& [nm, list] : nm_slots) {
                std::vector<uint16_t> gd, rd, gu, ru;
                for (auto [t, s] : list) {
                    const size_t bd = (size_t(t) * J0 + s) * H;
                    gd.insert(gd.end(), got_vdn.begin() + bd, got_vdn.begin() + bd + H);
                    rd.insert(rd.end(), ref_dn_one.begin(), ref_dn_one.end());
                    const size_t bu = (size_t(t) * J0 + s) * F;
                    gu.insert(gu.end(), got_vup.begin() + bu, got_vup.begin() + bu + F);
                    ru.insert(ru.end(), ref_up_t[t].begin(), ref_up_t[t].end());
                }
                uint64_t ex, u1, u1b, bn; double rl2, ma;
                cmp_bf16(gd, rd, ex, u1, u1b, bn, rl2, ma);
                const bool okd = (ex == rd.size());
                if (!okd) ++failures;
                std::printf("  [1TPC nm=%d] ink_vmoe_dn_mxf8   exact=%llu/%zu maxAbs=%.2e %s\n",
                            nm, (unsigned long long)ex, rd.size(), ma, okd ? "PASS" : "FAIL");
                cmp_bf16(gu, ru, ex, u1, u1b, bn, rl2, ma);
                const bool oku = (u1 == ru.size() && rl2 < 1e-3);
                if (!oku) ++failures;
                std::printf("  [1TPC nm=%d] ink_vmoe_up_mxf8   exact=%6.2f%% <=1ulp=%llu/%zu relL2=%.2e maxAbs=%.2e %s\n",
                            nm, 100.0 * double(ex) / double(ru.size()), (unsigned long long)u1, ru.size(),
                            rl2, ma, oku ? "PASS" : "FAIL");
            }
        }
    }

    // =========================================================================
    // Part 1: Realistic checkpoint-stat MXFP8 weights (E=16, Tv=8, J=8, 1 & 24 TPC)
    //         vs FP64 host ref, nm=1..4 + cnt>4 splits + -1 padded token.
    // =========================================================================
    std::printf("\n[mxf8decodetest] Part 1: realistic checkpoint-stat MXFP8 (10%% e=15, s in [-14,-9]) vs FP64 ref\n");
    {
        constexpr uint32_t E1 = 16, Tv = 8;
        std::vector<int32_t> h_idx(J * Tv, -1);
        const int32_t pat[7][8] = {
            {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 8, 9, 10}, {0, 1, 2, 3, 5, 11, 12, 13},
            {0, 1, 2, 6, 8, 11, 14, 15}, {0, 1, 5, 6, 9, 12, 14, 7}, {0, 8, 9, 10, 11, 13, 15, 7},
            {0, 10, 12, 13, 14, 15, 5, 6},
        };
        for (uint32_t t = 0; t < 7; ++t)
            for (uint32_t j = 0; j < J; ++j) h_idx[t * J + j] = pat[t][j];
        auto didx8 = upload(h_idx.data(), h_idx.size() * 4);
        DeviceBuffer dgrp(dev, 64 * 4 * 4 + 256);
        st.zero(dgrp, 0, 64 * 4 * 4); st.synchronize();
        {
            DirectPlan pp(rt);
            const DirectTpcTensor tp[] = {T(didx8->address(), DType::I32, {J, Tv}), T(dgrp.address(), DType::I32, {64, 4})};
            pp.add_tpc_grid(isa_plan, tp, {}, {1, 1, 1, 1, 1}, 0, 1);
            pp.prepare(); pp.run();
        }
        // Read back grp on host to report per-nm errors
        HostBuffer hgrp(dev, 64 * 4 * 4 + 64);
        st.download(dgrp, 0, hgrp, 0, 64 * 4 * 4); st.synchronize();
        std::vector<int32_t> grp_h(64 * 4);
        std::memcpy(grp_h.data(), hgrp.data(), 64 * 4 * 4);

        // Input activations x [Tv, H] (token 2 has 320x high-DR outliers)
        std::vector<uint16_t> h_x(uint64_t(Tv) * H);
        for (uint32_t t = 0; t < Tv; ++t)
            for (uint32_t k = 0; k < H; ++k) {
                const float v = (t == 2) ? ((k % 16 == 0) ? ((k & 16) ? 16.f : -16.f) : (float(int(rnd() % 201) - 100) / 2000.f))
                                         : (float(int(rnd() % 2001) - 1000) / 500.f);
                h_x[uint64_t(t) * H + k] = to_bf16(v);
            }
        auto dx = upload(h_x.data(), h_x.size() * 2);

        std::vector<uint8_t> w13_wq(uint64_t(E1) * H * N2), w13_bs(uint64_t(E1) * (H / 32) * N2);
        std::vector<uint8_t> w2_raw(uint64_t(E1) * F * H), w2_wq(uint64_t(E1) * F * H);
        std::vector<uint8_t> w2_bs_raw(uint64_t(E1) * (F / 32) * H), w2_bs(uint64_t(E1) * (F / 32) * H);
        std::vector<float> g13(E1), g2(E1), si13(uint64_t(E1) * N2), si2(uint64_t(E1) * H);
        for (uint32_t e = 0; e < E1; ++e) {
            const int smax13 = (e & 1) ? -10 : -9;
            const int smax2  = (e & 2) ? -10 : -9;
            g13[e] = std::ldexp(1.0f, smax13);
            g2[e]  = std::ldexp(1.0f, smax2);
            std::fill_n(si13.begin() + e * N2, N2, g13[e]);
            std::fill_n(si2.begin()  + e * H,  H,  g2[e]);
            for (uint32_t kb = 0; kb < H / 32; ++kb)
                for (uint32_t n = 0; n < N2; ++n) {
                    const int s = -14 + int(rnd() % uint32_t(smax13 + 15)) - ((smax13 == -9 && (rnd() % 32 == 0)) ? 1 : 0);
                    w13_bs[(uint64_t(e) * (H / 32) + kb) * N2 + n] = fp8_143_exact(std::ldexp(1.0f, s - smax13));
                }
            for (uint64_t i = uint64_t(e) * H * N2; i < uint64_t(e + 1) * H * N2; ++i)
                w13_wq[i] = rand_mxf8_code();
            for (uint32_t kb = 0; kb < F / 32; ++kb)
                for (uint32_t c = 0; c < H; c += 256)
                    for (uint32_t q = 0; q < 256; ++q) {
                        const int s = -14 + int(rnd() % uint32_t(smax2 + 15)) - ((smax2 == -9 && (rnd() % 32 == 0)) ? 1 : 0);
                        const uint8_t b = fp8_143_exact(std::ldexp(1.0f, s - smax2));
                        w2_bs_raw[(uint64_t(e) * (F / 32) + kb) * H + c + q] = b;
                        w2_bs    [(uint64_t(e) * (F / 32) + kb) * H + w2col(c, q)] = b;
                    }
            for (uint32_t k = 0; k < F; ++k)
                for (uint32_t c = 0; c < H; c += 256)
                    for (uint32_t q = 0; q < 256; ++q) {
                        const uint8_t v = rand_mxf8_code();
                        w2_raw[(uint64_t(e) * F + k) * H + c + q] = v;
                        w2_wq [(uint64_t(e) * F + k) * H + w2col(c, q)] = v;
                    }
        }
        // Host FP64 reference
        std::vector<uint16_t> ref_up(uint64_t(Tv) * J * F, 0), ref_dn(uint64_t(Tv) * J * H, 0);
        for (uint32_t t = 0; t < Tv; ++t) {
            for (uint32_t j = 0; j < J; ++j) {
                const int32_t e = h_idx[t * J + j];
                if (e < 0) continue;
                const uint64_t ba = (uint64_t(t) * J + j) * F;
                for (uint32_t f = 0; f < F; ++f) {
                    double ag = 0.0, au = 0.0;
                    const uint32_t ng = 2 * f, nu = 2 * f + 1;
                    for (uint32_t kb = 0; kb < H / 32; ++kb) {
                        double bg = 0.0, bu = 0.0;
                        const double sg = fp8_e4m3fn_to_f64(w13_bs[(uint64_t(e) * (H / 32) + kb) * N2 + ng]);
                        const double su = fp8_e4m3fn_to_f64(w13_bs[(uint64_t(e) * (H / 32) + kb) * N2 + nu]);
                        for (uint32_t dk = 0; dk < 32; ++dk) {
                            const uint32_t k = kb * 32 + dk;
                            const double xv = bf16_to_f64(h_x[uint64_t(t) * H + k]);
                            bg += xv * fp8_e4m3fn_to_f64(w13_wq[(uint64_t(e) * H + k) * N2 + ng]);
                            bu += xv * fp8_e4m3fn_to_f64(w13_wq[(uint64_t(e) * H + k) * N2 + nu]);
                        }
                        ag += bg * sg; au += bu * su;
                    }
                    ref_up[ba + f] = to_bf16(float(swiglu4_f64(ag * double(g13[e]), au * double(g13[e]), double(cu))));
                }
                const uint64_t by = (uint64_t(t) * J + j) * H;
                for (uint32_t n = 0; n < H; ++n) {
                    double acc = 0.0;
                    for (uint32_t kb = 0; kb < F / 32; ++kb) {
                        double ba_k = 0.0;
                        const double sb = fp8_e4m3fn_to_f64(w2_bs_raw[(uint64_t(e) * (F / 32) + kb) * H + n]);
                        for (uint32_t dk = 0; dk < 32; ++dk) {
                            const uint32_t k = kb * 32 + dk;
                            ba_k += bf16_to_f64(ref_up[ba + k]) * fp8_e4m3fn_to_f64(w2_raw[(uint64_t(e) * F + k) * H + n]);
                        }
                        acc += ba_k * sb;
                    }
                    ref_dn[by + n] = to_bf16(float(acc * double(g2[e])));
                }
            }
        }
        auto dw13_wq = upload(w13_wq.data(), w13_wq.size()), dw13_bs = upload(w13_bs.data(), w13_bs.size());
        auto dw2_wq  = upload(w2_wq.data(),  w2_wq.size()),  dw2_bs  = upload(w2_bs.data(),  w2_bs.size());
        auto dsi13   = upload(si13.data(), si13.size() * 4), dsi2    = upload(si2.data(), si2.size() * 4);
        auto dact_ref = upload(ref_up.data(), ref_up.size() * 2);
        DeviceBuffer dact(dev, uint64_t(Tv) * J * F * 2 + 256), dys(dev, uint64_t(Tv) * J * H * 2 + 256);

        std::vector<uint16_t> ref_up0(ref_up.begin(), ref_up.begin() + J * F);
        std::vector<uint16_t> ref_dn0(ref_dn.begin(), ref_dn.begin() + J * H);

        auto check_p1 = [&](const char* label, const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref,
                            uint32_t pad_rows) {
            uint64_t ex, u1, u1b, bn; double rl2, ma;
            cmp_bf16(got, ref, ex, u1, u1b, bn, rl2, ma);
            uint64_t pad_nz = 0;
            if (pad_rows) {
                const size_t stride = got.size() / Tv;
                for (size_t i = (Tv - pad_rows) * stride; i < got.size(); ++i) pad_nz += (got[i] != 0);
            }
            const double u1b_pct = bn ? 100.0 * double(u1b) / double(bn) : 100.0;
            const bool ok = (rl2 < 2e-3 && u1b_pct > 99.5 && pad_nz == 0);
            if (!ok) ++failures;
            std::printf("  %-28s exact %6.2f%% <=1ulp(|r|>=1e-3) %6.2f%% relL2 %.2e maxAbs %.2e padNZ=%llu %s\n",
                        label, 100.0 * double(ex) / double(got.size()), u1b_pct, rl2, ma,
                        (unsigned long long)pad_nz, ok ? "PASS" : "FAIL");
        };

        std::vector<uint16_t> last_vup, last_vdn;
        for (uint32_t ntpc : {1u, 24u}) {
            const DirectTpcTensor t_up9[] = {
                T(dx->address(), DType::BF16, {H, 1}), T(didx8->address(), DType::I32, {J, 1}),
                T(dw13_wq->address(), DType::U8, {N2, H, E1}), T(dw13_bs->address(), DType::F8, {N2, H / 32, E1}),
                T(dsi13->address(), DType::F32, {N2, E1}), T(dact.address(), DType::BF16, {F, J, 1})};
            const DirectTpcTensor t_dn9[] = {
                T(dact_ref->address(), DType::BF16, {F, J, 1}), T(didx8->address(), DType::I32, {J, 1}),
                T(dw2_wq->address(), DType::U8, {H, F, E1}), T(dw2_bs->address(), DType::F8, {H, F / 32, E1}),
                T(dsi2->address(), DType::F32, {H, E1}), T(dys.address(), DType::BF16, {H, J, 1})};
            const DirectTpcTensor t_vup[] = {
                T(dx->address(), DType::BF16, {H, Tv}), T(dgrp.address(), DType::I32, {64, 4}),
                T(dw13_wq->address(), DType::U8, {N2, H, E1}), T(dw13_bs->address(), DType::F8, {N2, H / 32, E1}),
                T(dsi13->address(), DType::F32, {N2, E1}), T(dact.address(), DType::BF16, {F, J, Tv})};
            const DirectTpcTensor t_vdn[] = {
                T(dact_ref->address(), DType::BF16, {F, J, Tv}), T(dgrp.address(), DType::I32, {64, 4}),
                T(dw2_wq->address(), DType::U8, {H, F, E1}), T(dw2_bs->address(), DType::F8, {H, F / 32, E1}),
                T(dsi2->address(), DType::F32, {H, E1}), T(dys.address(), DType::BF16, {H, J, Tv})};
            auto g_up9 = run_grid(isa_up9_mxf8, t_up9, sr_cu, 9 * J, ntpc, dact, uint64_t(F) * J, 1).second;
            auto g_dn9 = run_grid(isa_dn9_mxf8, t_dn9, {},    9 * J, ntpc, dys,  uint64_t(H) * J, 1).second;
            auto g_vup = run_grid(isa_vup_mxf8, t_vup, sr_cu, 9 * J * Tv, ntpc, dact, uint64_t(Tv) * J * F, 1).second;
            auto g_vdn = run_grid(isa_vdn_mxf8, t_vdn, {},    9 * J * Tv, ntpc, dys,  uint64_t(Tv) * J * H, 1).second;
            const std::string tag = (ntpc == 24u) ? "[24TPC] " : "[ 1TPC] ";
            check_p1((tag + "ink_dmoe_up9_mxf8").c_str(), g_up9, ref_up0, 0);
            check_p1((tag + "ink_dmoe_dn9_mxf8").c_str(), g_dn9, ref_dn0, 0);
            check_p1((tag + "ink_vmoe_up_mxf8").c_str(),  g_vup, ref_up,  1);
            check_p1((tag + "ink_vmoe_dn_mxf8").c_str(),  g_vdn, ref_dn,  1);
            if (ntpc == 24u) { last_vup = std::move(g_vup); last_vdn = std::move(g_vdn); }
        }
        // Per-nm breakdown on 24-TPC vmoe outputs
        for (int nm = 1; nm <= 4; ++nm) {
            std::vector<uint16_t> gu, ru, gd, rd;
            int ngrp = 0;
            for (int u = 0; u < 64; ++u) {
                if (grp_h[u] < 0) break;
                const int cnt = grp_h[64 + u];
                if (cnt != nm) continue;
                ++ngrp;
                const int tsp = grp_h[128 + u];
                for (int m = 0; m < cnt; ++m) {
                    const int c = (tsp >> (8 * m)) & 0xff;
                    const int t = c >> 4, s = c & 15;
                    const size_t bu = (size_t(t) * J + s) * F, bd = (size_t(t) * J + s) * H;
                    gu.insert(gu.end(), last_vup.begin() + bu, last_vup.begin() + bu + F);
                    ru.insert(ru.end(), ref_up.begin()   + bu, ref_up.begin()   + bu + F);
                    gd.insert(gd.end(), last_vdn.begin() + bd, last_vdn.begin() + bd + H);
                    rd.insert(rd.end(), ref_dn.begin()   + bd, ref_dn.begin()   + bd + H);
                }
            }
            if (gu.empty()) continue;
            uint64_t ex, u1, u1b, bn; double rl2u, mau, rl2d, mad;
            cmp_bf16(gu, ru, ex, u1, u1b, bn, rl2u, mau);
            const double u1bu = bn ? 100.0 * double(u1b) / double(bn) : 100.0;
            cmp_bf16(gd, rd, ex, u1, u1b, bn, rl2d, mad);
            const double u1bd = bn ? 100.0 * double(u1b) / double(bn) : 100.0;
            std::printf("    nm=%d (%2d grps): vmoe_up <=1ulp=%6.2f%% relL2=%.2e maxAbs=%.2e | vmoe_dn <=1ulp=%6.2f%% relL2=%.2e maxAbs=%.2e\n",
                        nm, ngrp, u1bu, rl2u, mau, u1bd, rl2d, mad);
        }
    }

    if (verify_only || failures != 0) {
        std::printf("[mxf8decodetest] %s (failures=%d%s)\n",
                    failures == 0 ? "PASS" : "FAIL", failures, verify_only ? ", verify-only" : ", aborting timing");
        return failures == 0 ? 0 : 1;
    }

    // =========================================================================
    // Part 2: 24-TPC Timing (E=128 full layer in HBM) - NVFP4 vs MXFP8
    // =========================================================================
    std::printf("\n[mxf8decodetest] Part 2: 24-TPC Timing (E=128 full layer in HBM): NVFP4 vs MXFP8\n");
    {
        constexpr uint32_t E = 128, Tmax = 8;
        const uint64_t isa_up9_nvf4 = rt.load_tpc_elf(objdir + "/ink_dmoe_up9_nvf4_gaudi2.o");
        const uint64_t isa_dn9_nvf4 = rt.load_tpc_elf(objdir + "/ink_dmoe_dn9_nvf4_gaudi2.o");
        const uint64_t isa_vup_nvf4 = rt.load_tpc_elf(objdir + "/ink_vmoe_up_nvf4_gaudi2.o");
        const uint64_t isa_vdn_nvf4 = rt.load_tpc_elf(objdir + "/ink_vmoe_dn_nvf4_gaudi2.o");

        // Allocate E=128 weight tensors for NVFP4 and MXFP8 (clean non-NaN patterns)
        auto dw13_nvq  = upload_fill_pattern(uint64_t(E) * (H / 2) * N2, 3, 0x7f);
        auto dw13_nvbs = upload_fill_pattern(uint64_t(E) * (H / 16) * N2, 4, 0x30);
        auto dw2_nvq   = upload_fill_pattern(uint64_t(E) * (F / 2) * H, 5, 0x7f);
        auto dw2_nvbs  = upload_fill_pattern(uint64_t(E) * (F / 16) * H, 6, 0x30);
        auto dw13_mxq  = upload_fill_pattern(uint64_t(E) * H * N2, 7, 0x7e);
        auto dw13_mxbs = upload_fill_pattern(uint64_t(E) * (H / 32) * N2, 8, 0x30);
        auto dw2_mxq   = upload_fill_pattern(uint64_t(E) * F * H, 9, 0x7e);
        auto dw2_mxbs  = upload_fill_pattern(uint64_t(E) * (F / 32) * H, 10, 0x30);
        std::vector<float> h_si13(uint64_t(E) * N2, 0.002f), h_si2(uint64_t(E) * H, 0.002f);
        auto dsi13 = upload(h_si13.data(), h_si13.size() * 4);
        auto dsi2  = upload(h_si2.data(),  h_si2.size() * 4);
        std::vector<uint16_t> h_xb(uint64_t(Tmax) * H, to_bf16(0.25f));
        std::vector<uint16_t> h_ab(uint64_t(Tmax) * J * F, to_bf16(0.25f));
        auto dxb = upload(h_xb.data(), h_xb.size() * 2);
        auto dab = upload(h_ab.data(), h_ab.size() * 2);
        DeviceBuffer dact(dev, uint64_t(Tmax) * J * F * 2 + 256);
        DeviceBuffer dys (dev, uint64_t(Tmax) * J * H * 2 + 256);
        DeviceBuffer didx(dev, uint64_t(Tmax) * J * 4 + 256);
        DeviceBuffer dgrp(dev, 64 * 4 * 4 + 256);

        auto med_run = [](DirectPlan& p, int nruns) {
            std::vector<double> tv;
            tv.reserve(nruns);
            for (int r = 0; r < nruns; ++r) tv.push_back(p.run());
            std::sort(tv.begin(), tv.end());
            return tv[tv.size() / 2];
        };

        // --- 2a: B=1 decode dmoe_up9 & dmoe_dn9 (J=8, 72 pts = 3 waves, median of 41) ---
        double b1_up_nv = 0, b1_up_mx = 0;
        double b1_dn_nv = 0, b1_dn_mx = 0;
        {
            const int32_t idx1[J] = {3, 19, 37, 52, 71, 88, 105, 121};
            auto didx1 = upload(idx1, sizeof(idx1));
            auto time_b1 = [&](uint64_t isa, std::span<const DirectTpcTensor> ts, std::span<const uint32_t> sr) {
                DirectPlan p(rt);
                p.add_tpc_grid(isa, ts, sr, {9 * J, 1, 1, 1, 1}, 0, 24);
                p.prepare();
                return med_run(p, 41);
            };
            const DirectTpcTensor tu_nv[] = {T(dxb->address(), DType::BF16, {H, 1}), T(didx1->address(), DType::I32, {J, 1}),
                T(dw13_nvq->address(), DType::U8, {N2, H / 2, E}), T(dw13_nvbs->address(), DType::F8, {N2, H / 16, E}),
                T(dsi13->address(), DType::F32, {N2, E}), T(dact.address(), DType::BF16, {F, J, 1})};
            const DirectTpcTensor tu_mx[] = {T(dxb->address(), DType::BF16, {H, 1}), T(didx1->address(), DType::I32, {J, 1}),
                T(dw13_mxq->address(), DType::U8, {N2, H, E}), T(dw13_mxbs->address(), DType::F8, {N2, H / 32, E}),
                T(dsi13->address(), DType::F32, {N2, E}), T(dact.address(), DType::BF16, {F, J, 1})};
            const DirectTpcTensor td_nv[] = {T(dab->address(), DType::BF16, {F, J, 1}), T(didx1->address(), DType::I32, {J, 1}),
                T(dw2_nvq->address(), DType::U8, {H, F / 2, E}), T(dw2_nvbs->address(), DType::F8, {H, F / 16, E}),
                T(dsi2->address(), DType::F32, {H, E}), T(dys.address(), DType::BF16, {H, J, 1})};
            const DirectTpcTensor td_mx[] = {T(dab->address(), DType::BF16, {F, J, 1}), T(didx1->address(), DType::I32, {J, 1}),
                T(dw2_mxq->address(), DType::U8, {H, F, E}), T(dw2_mxbs->address(), DType::F8, {H, F / 32, E}),
                T(dsi2->address(), DType::F32, {H, E}), T(dys.address(), DType::BF16, {H, J, 1})};
            b1_up_nv = time_b1(isa_up9_nvf4, tu_nv, sr_cu);
            b1_up_mx = time_b1(isa_up9_mxf8, tu_mx, sr_cu);
            b1_dn_nv = time_b1(isa_dn9_nvf4, td_nv, {});
            b1_dn_mx = time_b1(isa_dn9_mxf8, td_mx, {});
            const double up_mb = double(J) * H * N2 / 1e6, dn_mb = double(J) * F * H / 1e6;
            std::printf("  --- 2a: B=1 decode (J=8, 72 pts = 3 waves, median of 41) ---\n");
            std::printf("  dmoe_up9 : NVF4=%5.1f us (%4.0f GB/s) | MXF8=%5.1f us (%4.0f GB/s)\n",
                        b1_up_nv, up_mb * 0.5625 * 1e3 / b1_up_nv, b1_up_mx, up_mb * 1.03125 * 1e3 / b1_up_mx);
            std::printf("  dmoe_dn9 : NVF4=%5.1f us (%4.0f GB/s) | MXF8=%5.1f us (%4.0f GB/s)\n",
                        b1_dn_nv, dn_mb * 0.5625 * 1e3 / b1_dn_nv, b1_dn_mx, dn_mb * 1.03125 * 1e3 / b1_dn_mx);
            std::printf("  B=1 sum  : NVF4=%5.1f us | MXF8=%5.1f us\n",
                        b1_up_nv + b1_dn_nv, b1_up_mx + b1_dn_mx);
        }

        // Helpers to build vmoe_up / vmoe_dn plans for kind: 0=NVF4, 1=MXF8
        auto build_vup = [&](int kind, uint32_t Tcur) {
            auto p = std::make_unique<DirectPlan>(rt);
            if (kind == 0) {
                const DirectTpcTensor ts[] = {T(dxb->address(), DType::BF16, {H, Tcur}), T(dgrp.address(), DType::I32, {64, 4}),
                    T(dw13_nvq->address(), DType::U8, {N2, H / 2, E}), T(dw13_nvbs->address(), DType::F8, {N2, H / 16, E}),
                    T(dsi13->address(), DType::F32, {N2, E}), T(dact.address(), DType::BF16, {F, J, Tcur})};
                p->add_tpc_grid(isa_vup_nvf4, ts, sr_cu, {9 * J * Tcur, 1, 1, 1, 1}, 0, 24);
            } else {
                const DirectTpcTensor ts[] = {T(dxb->address(), DType::BF16, {H, Tcur}), T(dgrp.address(), DType::I32, {64, 4}),
                    T(dw13_mxq->address(), DType::U8, {N2, H, E}), T(dw13_mxbs->address(), DType::F8, {N2, H / 32, E}),
                    T(dsi13->address(), DType::F32, {N2, E}), T(dact.address(), DType::BF16, {F, J, Tcur})};
                p->add_tpc_grid(isa_vup_mxf8, ts, sr_cu, {9 * J * Tcur, 1, 1, 1, 1}, 0, 24);
            }
            p->prepare();
            return p;
        };
        auto build_vdn = [&](int kind, uint32_t Tcur) {
            auto p = std::make_unique<DirectPlan>(rt);
            if (kind == 0) {
                const DirectTpcTensor ts[] = {T(dab->address(), DType::BF16, {F, J, Tcur}), T(dgrp.address(), DType::I32, {64, 4}),
                    T(dw2_nvq->address(), DType::U8, {H, F / 2, E}), T(dw2_nvbs->address(), DType::F8, {H, F / 16, E}),
                    T(dsi2->address(), DType::F32, {H, E}), T(dys.address(), DType::BF16, {H, J, Tcur})};
                p->add_tpc_grid(isa_vdn_nvf4, ts, {}, {9 * J * Tcur, 1, 1, 1, 1}, 0, 24);
            } else {
                const DirectTpcTensor ts[] = {T(dab->address(), DType::BF16, {F, J, Tcur}), T(dgrp.address(), DType::I32, {64, 4}),
                    T(dw2_mxq->address(), DType::U8, {H, F, E}), T(dw2_mxbs->address(), DType::F8, {H, F / 32, E}),
                    T(dsi2->address(), DType::F32, {H, E}), T(dys.address(), DType::BF16, {H, J, Tcur})};
                p->add_tpc_grid(isa_vdn_mxf8, ts, {}, {9 * J * Tcur, 1, 1, 1, 1}, 0, 24);
            }
            p->prepare();
            return p;
        };

        // --- 2b: Pure nm=1,2,3,4 (16 unique experts = 144 pts = 6 full waves on 24 TPCs, median of 41) ---
        std::printf("\n  --- 2b: Pure nm=1..4 (16 unique experts = 6 waves on 24 TPCs, median of 41) ---\n");
        HostBuffer hbg(dev, 64 * 4 * 4 + 64);
        for (int nm = 1; nm <= 4; ++nm) {
            const uint32_t Tcur = uint32_t(2 * nm); // 16 * nm slots = J(=8) * Tcur
            std::vector<int32_t> gh(64 * 4, 0);
            for (int u = 0; u < 64; ++u) gh[u] = -1;
            int flat = 0;
            for (int u = 0; u < 16; ++u) {
                gh[0 * 64 + u] = (u * 7 + 3) % int(E);
                gh[1 * 64 + u] = nm;
                int32_t tsp = 0;
                for (int m = 0; m < nm; ++m) {
                    const int t = flat / int(J), s = flat % int(J);
                    ++flat;
                    tsp |= (((t << 4) | s) << (8 * m));
                }
                gh[2 * 64 + u] = tsp;
            }
            std::memcpy(hbg.data(), gh.data(), gh.size() * 4);
            st.upload(hbg, 0, dgrp, 0, gh.size() * 4); st.synchronize();
            auto pu_nv = build_vup(0, Tcur), pu_mx = build_vup(1, Tcur);
            auto pd_nv = build_vdn(0, Tcur), pd_mx = build_vdn(1, Tcur);
            const double unv = med_run(*pu_nv, 41), umx = med_run(*pu_mx, 41);
            const double dnv = med_run(*pd_nv, 41), dmx = med_run(*pd_mx, 41);
            std::printf("  pure nm=%d (T=%u) | up: NVF4=%5.1f MXF8=%5.1f | dn: NVF4=%5.1f MXF8=%5.1f\n",
                        nm, Tcur, unv, umx, dnv, dmx);
        }

        // --- 2c: Synthetic dedup patterns (Zipf/correlated, median of 41) ---
        std::printf("\n  --- 2c: Synthetic dedup patterns (median of 41) ---\n");
        auto sample_zipf = [&](uint32_t Tcur, double alpha, double corr, uint64_t rseed) {
            std::vector<double> cdf(E);
            double s = 0.0;
            for (uint32_t e = 0; e < E; ++e) { s += std::pow(double(e + 1), -alpha); cdf[e] = s; }
            for (auto& v : cdf) v /= s;
            uint64_t st_r = rseed;
            auto r01 = [&]() { st_r = st_r * 6364136223846793005ull + 1442695040888963407ull; return double(st_r >> 11) * (1.0 / double(1ull << 53)); };
            std::vector<int32_t> idx(Tcur * J);
            for (uint32_t t = 0; t < Tcur; ++t) {
                uint32_t picked = 0;
                while (picked < J) {
                    int32_t e = -1;
                    if (t > 0 && corr > 0.0 && r01() < corr) e = idx[(t - 1) * J + uint32_t(r01() * J)];
                    else {
                        const double u = r01();
                        e = int32_t(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
                        if (e >= int32_t(E)) e = int32_t(E - 1);
                    }
                    bool dup = false;
                    for (uint32_t k = 0; k < picked; ++k) if (idx[t * J + k] == e) { dup = true; break; }
                    if (!dup) idx[t * J + picked++] = e;
                }
            }
            return idx;
        };
        struct SynCase { const char* name; uint32_t Tcur; double alpha; double corr; uint64_t rseed; };
        const SynCase scases[] = {
            {"B=8 uniform",        8, 0.00, 0.00, 101},
            {"B=8 realistic-skew", 8, 1.05, 0.00, 303},
            {"B=4 realistic-skew", 4, 1.05, 0.00, 606},
            {"Verify T=2 corr",    2, 0.90, 0.45, 909},
            {"Verify T=3 corr",    3, 0.90, 0.45, 1010},
            {"Verify T=4 corr",    4, 0.90, 0.45, 1111},
        };
        HostBuffer hbi(dev, 64 * 4 + 64);
        for (const auto& sc : scases) {
            auto idx_v = sample_zipf(sc.Tcur, sc.alpha, sc.corr, sc.rseed);
            std::memcpy(hbi.data(), idx_v.data(), idx_v.size() * 4);
            st.upload(hbi, 0, didx, 0, idx_v.size() * 4); st.synchronize();
            st.zero(dgrp, 0, 64 * 4 * 4); st.synchronize();
            {
                DirectPlan pp(rt);
                const DirectTpcTensor tp[] = {T(didx.address(), DType::I32, {J, sc.Tcur}), T(dgrp.address(), DType::I32, {64, 4})};
                pp.add_tpc_grid(isa_plan, tp, {}, {1, 1, 1, 1, 1}, 0, 24);
                pp.prepare(); pp.run();
            }
            auto pu_nv = build_vup(0, sc.Tcur), pu_mx = build_vup(1, sc.Tcur);
            auto pd_nv = build_vdn(0, sc.Tcur), pd_mx = build_vdn(1, sc.Tcur);
            const double unv = med_run(*pu_nv, 41), umx = med_run(*pu_mx, 41);
            const double dnv = med_run(*pd_nv, 41), dmx = med_run(*pd_mx, 41);
            const double tnv = unv + dnv, tmx = umx + dmx;
            std::printf("  %-19s | up: nv=%5.1f mx=%5.1f | dn: nv=%5.1f mx=%5.1f | tot: nv=%5.1f mx=%5.1f\n",
                        sc.name, unv, umx, dnv, dmx, tnv, tmx);
        }

        // --- 2d: Real v5 routing replay (MXFP8 layers app>=44 + reference app<44 step0) ---
        if (!routing_bin.empty()) {
            std::printf("\n  --- 2d: Real routing replay (%s, median of 9/app) ---\n", routing_bin.c_str());
            FILE* f = std::fopen(routing_bin.c_str(), "rb");
            if (!f) { std::fprintf(stderr, "cannot open %s\n", routing_bin.c_str()); return 1; }
            std::fseek(f, 0, SEEK_END);
            const size_t fsz = size_t(std::ftell(f));
            std::fseek(f, 0, SEEK_SET);
            const size_t nrec = fsz / (324 * 4);
            std::vector<int32_t> raw(nrec * 324);
            if (std::fread(raw.data(), 4, raw.size(), f) != raw.size()) { std::fclose(f); return 1; }
            std::fclose(f);

            struct ReplaySpec { const char* name; int mode; uint32_t Tcur; int max_steps; int app_lo; int app_hi; };
            const ReplaySpec rspecs[] = {
                {"MX(44..47) B=8 (4x4)", 0, 8, 4, 44, 47},
                {"MX(44..47) B=4 (4x4)", 0, 4, 4, 44, 47},
                {"MX(44..47) B=2 (4x4)", 0, 2, 4, 44, 47},
                {"MX(44..47) Ver T=2",   1, 2, 1, 44, 47},
                {"MX(44..47) Ver T=3",   1, 3, 1, 44, 47},
                {"MX(44..47) Ver T=4",   1, 4, 1, 44, 47},
                {"NV( 0..43) B=8 st0",   0, 8, 1,  0, 43},
                {"NV( 0..43) Ver T=4",   1, 4, 1,  0, 43},
            };
            constexpr int NREP = 9;
            for (const auto& rs : rspecs) {
                auto pu_nv = build_vup(0, rs.Tcur), pu_mx = build_vup(1, rs.Tcur);
                auto pd_nv = build_vdn(0, rs.Tcur), pd_mx = build_vdn(1, rs.Tcur);
                DirectPlan p_plan(rt);
                {
                    const DirectTpcTensor tp[] = {T(didx.address(), DType::I32, {J, rs.Tcur}), T(dgrp.address(), DType::I32, {64, 4})};
                    p_plan.add_tpc_grid(isa_plan, tp, {}, {1, 1, 1, 1, 1}, 0, 24);
                    p_plan.prepare();
                }
                double su_nv = 0, su_mx = 0, sd_nv = 0, sd_mx = 0;
                int cnt_apps = 0;
                for (size_t r = 0; r < nrec; ++r) {
                    const int32_t* v = raw.data() + r * 324;
                    const int mode = v[0], bt = v[1], app = v[2], step = v[3];
                    if (app < rs.app_lo || app > rs.app_hi) continue;
                    if (rs.mode == 0) {
                        if (mode != 0 || uint32_t(bt) != rs.Tcur || step >= rs.max_steps) continue;
                    } else {
                        if (mode != 1 || step >= rs.max_steps) continue;
                    }
                    std::memcpy(hbi.data(), v + 4, rs.Tcur * J * 4);
                    st.upload(hbi, 0, didx, 0, rs.Tcur * J * 4); st.synchronize();
                    st.zero(dgrp, 0, 64 * 4 * 4); st.synchronize();
                    p_plan.run();
                    su_nv += med_run(*pu_nv, NREP); su_mx += med_run(*pu_mx, NREP);
                    sd_nv += med_run(*pd_nv, NREP); sd_mx += med_run(*pd_mx, NREP);
                    ++cnt_apps;
                }
                if (!cnt_apps) continue;
                const double inv = 1.0 / double(cnt_apps);
                const double mnv = (su_nv + sd_nv) * inv, mmx = (su_mx + sd_mx) * inv;
                std::printf("  %-20s (%2d apps) | up: nv=%5.1f mx=%5.1f | dn: nv=%5.1f mx=%5.1f | tot: nv=%5.1f mx=%5.1f\n",
                            rs.name, cnt_apps, su_nv * inv, su_mx * inv, sd_nv * inv, sd_mx * inv, mnv, mmx);
            }
        }
    }

    std::printf("[mxf8decodetest] %s (failures=%d)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "pdmatest") return pdmatest();
    if (argc >= 4 && std::string(argv[1]) == "gemmtest") return gemmtest(argv[2], argv[3]);
    if (argc >= 4 && std::string(argv[1]) == "gemmtest_prefix") {
        std::vector<std::string> ps;
        for (int i = 3; i < argc; ++i) ps.emplace_back(argv[i]);
        return gemmtest_prefix(argv[2], ps);
    }
    if (argc >= 3 && std::string(argv[1]) == "deqtest") return deqtest(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "mxdeqtest") return mxdeqtest(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "hostrt") return hostrt(argv[2]);
    if (argc >= 5 && std::string(argv[1]) == "lanetest") return lanetest(argv[2], argv[3], uint32_t(std::atoi(argv[4])));
    if (argc >= 5 && std::string(argv[1]) == "tilebench")
        return tilebench(argv[2], argv[3], uint32_t(std::atoi(argv[4])), argc >= 6 ? uint32_t(std::atoi(argv[5])) : 9u);
    if (argc >= 3 && std::string(argv[1]) == "tpctest") return tpctest(argv[2]);
    if (argc >= 6 && std::string(argv[1]) == "dnormbench")
        return dnormbench(argv[2], argv[3], uint32_t(std::atoi(argv[4])), uint32_t(std::atoi(argv[5])));
    if (argc >= 6 && std::string(argv[1]) == "attnbench")
        return attnbench(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]));
    if (argc >= 7 && std::string(argv[1]) == "mixbench")
        return mixbench(argv[2], argv[3], uint32_t(std::atoi(argv[4])), uint32_t(std::atoi(argv[5])), std::atoi(argv[6]));
    if (argc >= 3 && std::string(argv[1]) == "luttest") return luttest(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "bf16pbtest") return bf16pbtest(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "nvf4decodetest") return nvf4decodetest(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "mxf8decodetest") {
        const std::string arg3 = argc >= 4 ? argv[3] : "";
        const bool vo = (arg3 == "verify");
        return mxf8decodetest(argv[2], vo ? "" : arg3, vo);
    }
    if (argc < 3 || std::string(argv[1]) != "capture") {
        std::fprintf(stderr, "usage: %s capture <out_dir> [filter]      (empties $HABANA_LOGS/mme_debug)\n"
                             "       %s gemmtest <desc_dir> <name>      (the other test / bench modes: see main())\n", argv[0], argv[0]);
        return 2;
    }
    setenv("ENABLE_EXPERIMENTAL_FLAGS", "true", 1);
    setenv("PRINT_MME_DESCRIPTORS", "1", 1);
    const std::string root = fs::absolute(argv[2]).string();
    const std::string filt = argc > 3 ? argv[3] : "";
    Device dev;
    for (const auto& s : specs())
        if (filt.empty() || s.name.find(filt) != std::string::npos) capture(dev, root, s);
    return 0;
}
