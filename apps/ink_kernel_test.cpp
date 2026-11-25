// Unit test of the Inkling TPC kernels against a CPU reference (development tool).
#include "g2e/graph.hpp"
#include "g2e/numeric.hpp"
#include "g2e/ops.hpp"
#include <perf_lib_layer_params.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cstring>
using namespace g2e;
struct P { float scale, thr, eps; int32_t mode; };
// ink_attn_sm16 timing at the prefill shape (one KV head: GT = 3 * 1024 rows, W = T = RE = 1024), objects given by
// SM16_OBJ (default: the built kernel) so debug variants can be compared
static void test_sm16(Device& dev, Stream& st) {
    const uint64_t W = 1024, T = 1024, RE = 1024, NH = 24, GT = 3 * T, SN = W + T;
    DeviceBuffer bsr(dev, GT * W * 2), bsn(dev, GT * T * 2), bbr(dev, T * NH * RE * 2), bsk(dev, T * NH * 4 + 4096),
        bpos(dev, 4096), bp(dev, GT * SN * 2), bs(dev, GT * 4 + 4096);
    HostBuffer hb(dev, 4096);
    hb.as<int32_t>()[0] = 4096;
    st.upload(hb, 0, bpos, 0, 4); st.synchronize();
    for (int full = 0; full < 2; ++full) {
        GraphBuilder g(dev);
        const auto sr = g.input("sr", DType::F16, {GT, W});
        const auto sn = g.input("sn", DType::F16, {GT, T});
        const auto br = g.input("br", DType::F16, {T, NH, RE});
        const auto sk = g.input("sk", DType::F32, {T, NH});
        const auto ps = g.input("ps", DType::I32, {1});
        const auto po = g.input("po", DType::BF16, {GT, SN});
        const auto so = g.input("so", DType::F32, {GT, 1});
        struct { float scale; int32_t G, full, kv; } prm{0.088f, 3, full, 2};
        const TensorId ins[] = {sr, sn, br, sk, ps};
        const TensorId outs[] = {po, so};
        g.op("ink_attn_sm16_gaudi2", ins, outs, &prm, sizeof(prm));
        auto r = g.compile("inktest_sm16_" + std::to_string(full));
        std::vector<Binding> b = {{"sr", &bsr, 0}, {"sn", &bsn, 0}, {"br", &bbr, 0}, {"sk", &bsk, 0}, {"ps", &bpos, 0},
                                  {"po", &bp, 0}, {"so", &bs, 0}};
        r->enqueue(st, b);
        st.synchronize();
        Event e0(dev), e1(dev);
        e0.record(st);
        for (int i = 0; i < 10; ++i) r->enqueue(st, b);
        e1.record(st);
        st.synchronize();
        std::printf("attn_sm16 full=%d: %.1f us\n", full, 1000.0 * Event::elapsed_ms(e0, e1) / 10);
    }
}

int main() {
    Device dev;
    Stream st(dev);
    if (std::getenv("TEST_SM16")) { test_sm16(dev, st); return 0; }
    if (std::getenv("TEST_TOPK")) {  // is topk's output sorted (descending)?
        GraphBuilder g(dev);
        Ops o(g);
        const auto x = g.input("x", DType::F32, {1, 128});
        const auto r = o.topk(x, 32, "tk");
        const auto vo = g.input("v", DType::F32, {1, 32});
        const auto io = g.input("i", DType::I32, {1, 32});
        g.op("identity", {r.first}, {vo});
        g.op("identity", {r.second}, {io});
        auto rc = g.compile("inktest_topk");
        HostBuffer hb(dev, 4096);
        DeviceBuffer bx(dev, 4096), bv(dev, 4096), bi(dev, 4096);
        std::mt19937 rng(3);
        bool sorted = true;
        for (int trial = 0; trial < 20; ++trial) {
            for (int i = 0; i < 128; ++i) hb.as<float>()[i] = float(rng() % 300);
            st.upload(hb, 0, bx, 0, 512); st.synchronize();
            std::vector<Binding> b = {{"x", &bx, 0}, {"v", &bv, 0}, {"i", &bi, 0}};
            rc->enqueue(st, b);
            st.download(bv, 0, hb, 1024, 128); st.synchronize();
            for (int i = 1; i < 32; ++i) if (hb.as<float>()[256 + i] > hb.as<float>()[256 + i - 1]) sorted = false;
        }
        std::printf("topk sorted descending: %s\n", sorted ? "yes" : "NO");
        return 0;
    }
    if (std::getenv("TEST_CUMSUM")) {
        for (int axis = 0; axis < 2; ++axis) {
            GraphBuilder g(dev);
            auto xi = g.input("x", DType::F32, {4, 3});
            auto yo = g.input("y", DType::F32, {4, 3});
            ns_CumSumKernel::Params p{};
            p.axis = axis; p.exclusive = 1; p.reverse = 0;
            g.op("cumsum_fwd_f32", {xi}, {yo}, &p, sizeof(p));
            auto r = g.compile("inktest_cumsum" + std::to_string(axis));
            HostBuffer hb(dev, 4096);
            DeviceBuffer bx(dev, 4096), by(dev, 4096);
            for (int i = 0; i < 12; ++i) hb.as<float>()[i] = float(i + 1);
            st.upload(hb, 0, bx, 0, 48); st.synchronize();
            std::vector<Binding> b = {{"x", &bx, 0}, {"y", &by, 0}};
            r->enqueue(st, b);
            st.download(by, 0, hb, 0, 48); st.synchronize();
            std::printf("axis %d:", axis);
            for (int i = 0; i < 12; ++i) std::printf(" %g", hb.as<float>()[i]);
            std::printf("\n");
        }
        return 0;
    }
    if (std::getenv("TEST_SCATTER")) {  // out = data; out[idx[i]] = upd[i]
        GraphBuilder g(dev);
        auto d = g.input("d", DType::F32, {8});
        auto ix = g.input("ix", DType::I32, {4});
        auto up = g.input("up", DType::F32, {4});
        auto out = g.input("out", DType::F32, {8});
        ns_ScatterKernel::Params p{};
        p.axis = 0;
        g.op("scatter_fwd_f32", {d, ix, up}, {out}, &p, sizeof(p));
        auto r = g.compile("inktest_scatter");
        HostBuffer hb(dev, 4096);
        DeviceBuffer bd(dev, 4096), bi(dev, 4096), bu(dev, 4096), bo(dev, 4096);
        for (int i = 0; i < 8; ++i) hb.as<float>()[i] = -1.f;
        st.upload(hb, 0, bd, 0, 32); st.synchronize();
        const int32_t idx[4] = {5, 2, 7, 2};
        std::memcpy(hb.data(), idx, 16); st.upload(hb, 0, bi, 0, 16); st.synchronize();
        for (int i = 0; i < 4; ++i) hb.as<float>()[i] = float(10 + i);
        st.upload(hb, 0, bu, 0, 16); st.synchronize();
        std::vector<Binding> b = {{"d", &bd, 0}, {"ix", &bi, 0}, {"up", &bu, 0}, {"out", &bo, 0}};
        r->enqueue(st, b);
        st.download(bo, 0, hb, 0, 32); st.synchronize();
        for (int i = 0; i < 8; ++i) std::printf(" %g", hb.as<float>()[i]);
        std::printf("\n");
        return 0;
    }
    const uint32_t T = 3, C = 1024, KV = 8;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 1);
    auto bf = [&](float x) { return from_bf16(to_bf16(x)); };
    std::vector<float> x(T * C), s(3 * C), w(4 * C), d(T * KV * 4);
    for (auto& v : x) v = bf(nd(rng));
    for (auto& v : s) v = bf(nd(rng));
    for (auto& v : w) v = bf(0.3f * nd(rng));
    const char* dm = std::getenv("DMODE");
    const int dmode = dm ? std::atoi(dm) : 0;
    if (dmode == 5) { for (auto& v : x) v = 1.f; for (auto& v : s) v = 0.f; for (auto& v : w) v = 0.f; }
    for (size_t i = 0; i < d.size(); ++i) {
        const size_t t = i / (KV * 4), h = (i / 4) % KV, lag = i % 4;
        d[i] = dmode == 5 ? 0.5f : dmode == 0 ? bf(0.3f * nd(rng)) : dmode == 1 ? 0.f : dmode == 2 ? 0.5f : (lag == 0 && h == 0 && t == 1 ? 1.f : 0.f);
    }
    for (int mode = 0; mode < 3; ++mode) for (int dyn = 0; dyn < 2; ++dyn) {
        GraphBuilder g(dev);
        Ops o(g);
        auto xi = g.input("x", DType::BF16, {T, C}), si = g.input("s", DType::BF16, {3, C}), wi = g.input("w", DType::BF16, {4, C});
        auto di = g.input("d", DType::F32, {T, KV * 4});
        auto yo = g.input("y", DType::BF16, {T, C}), so = g.input("so", DType::BF16, {3, C});
        P p{dyn ? 1.f / 3.f : 1.f / std::sqrt(5.f), 22.6f, 1e-6f, mode};
        if (dyn) g.op("ink_conv_dyn_bf16_gaudi2", {xi, si, wi, di}, {yo, so}, &p, sizeof(p));
        else { g.op("ink_conv_bf16_gaudi2", {xi, si, wi}, {yo, so}, &p, sizeof(p)); g.op("identity", {di}, {g.temp("dd", DType::F32, {T, KV * 4})}); }
        auto r = g.compile(std::string("inktest_conv_") + std::to_string(mode) + std::to_string(dyn));
        HostBuffer hb(dev, 1 << 20);
        DeviceBuffer bx(dev, T * C * 2), bs(dev, 3 * C * 2), bw(dev, 4 * C * 2), bd(dev, 4096), by(dev, T * C * 2), bso(dev, 3 * C * 2);
        auto up16 = [&](const std::vector<float>& v, DeviceBuffer& b) {
            for (size_t i = 0; i < v.size(); ++i) hb.as<uint16_t>()[i] = to_bf16(v[i]);
            st.upload(hb, 0, b, 0, v.size() * 2); st.synchronize();
        };
        up16(x, bx); up16(s, bs); up16(w, bw);
        for (size_t i = 0; i < d.size(); ++i) hb.as<float>()[i] = d[i];
        st.upload(hb, 0, bd, 0, d.size() * 4); st.synchronize();
        std::vector<Binding> b = {{"x", &bx, 0}, {"s", &bs, 0}, {"w", &bw, 0}, {"y", &by, 0}, {"so", &bso, 0}};
        b.push_back({"d", &bd, 0});
        r->enqueue(st, b);
        st.download(by, 0, hb, 0, T * C * 2); st.synchronize();
        std::vector<float> y(T * C);
        for (size_t i = 0; i < y.size(); ++i) y[i] = from_bf16(hb.as<uint16_t>()[i]);
        // CPU reference
        double maxerr = 0;
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<float> yr(C);
            for (uint32_t c = 0; c < C; ++c) {
                auto xin = [&](int k) { int r = int(t) + k; return r < 3 ? s[r * C + c] : x[(r - 3) * C + c]; };
                float acc = 0;
                for (int i = 0; i < 4; ++i) {
                    float wt = w[i * C + c];
                    if (dyn) wt += d[t * KV * 4 + (c / 128) * 4 + (3 - i)];
                    acc = i == 0 ? xin(i) * wt : acc + xin(i) * wt;
                }
                yr[c] = bf((acc + x[t * C + c]) * p.scale);
            }
            if (mode) for (uint32_t h = 0; h < KV; ++h) {
                double ss = 0;
                for (int k = 0; k < 128; ++k) ss += double(yr[h * 128 + k]) * yr[h * 128 + k];
                float f = mode == 1 ? std::min(p.thr / float(std::sqrt(ss) + 1e-6), 1.f) : float(1.0 / std::sqrt(ss / 128 + 1e-6));
                for (int k = 0; k < 128; ++k) yr[h * 128 + k] = bf(yr[h * 128 + k] * f);
            }
            for (uint32_t c = 0; c < C; ++c) maxerr = std::max(maxerr, double(std::fabs(yr[c] - y[t * C + c])) / (std::fabs(yr[c]) + 1e-2));
            if (dmode == 5 && dyn && mode == 0) std::printf("   t%u: ref %g got %g (c=0) got %g (c=500)\n", t, yr[0], y[t * C], y[t * C + 500]);
            if (t == 1 && dyn && mode == 0) std::printf("   sample t1: ref %g %g %g got %g %g %g\n", yr[0], yr[200], yr[1000], y[C], y[C + 200], y[C + 1000]);
        }
        std::printf("conv mode %d dyn %d: max rel err %.4g\n", mode, dyn, maxerr);
    }
}

