#include "g2e/gdn.hpp"

#include <cstdlib>

namespace g2e {

bool gdn_prefill_fast_enabled() {
    static const bool off = std::getenv("G2E_OLD_PREFILL_GDN") != nullptr;
    return !off;
}

std::vector<GdnConstant> gdn_chunk_constants(uint32_t c) {
    std::vector<float> upper(c * c), strict(c * c), inclusive(c * c), lower(c * c), eye(c * c), diag8(c * c);
    for (uint32_t t = 0; t < c; ++t)
        for (uint32_t u = 0; u < c; ++u) {
            // [row t][column u]: upper is the cumulative-sum operator (G = g x upper); strict / inclusive are added to
            // the log-decay difference before exp; lower is the Newton residual mask
            upper[t * c + u] = t <= u ? 1.f : 0.f;
            strict[t * c + u] = u < t ? 0.f : -1e30f;
            inclusive[t * c + u] = u <= t ? 0.f : -1e30f;
            lower[t * c + u] = u < t ? 1.f : 0.f;
            eye[t * c + u] = t == u ? 1.f : 0.f;
            diag8[t * c + u] = t / 8 == u / 8 ? 1.f : 0.f;
        }
    const Shape sq = {c, c}, sq1 = {1, c, c};
    std::vector<GdnConstant> out = {{"upper", sq, upper},     {"strict", sq1, strict}, {"inclusive", sq1, inclusive},
                                    {"lower", sq1, lower},    {"eye", sq1, eye},       {"diag8", sq1, diag8}};
    for (uint32_t s = 8; s < c; s *= 2) {  // the lower-left SxS block of every 2Sx2S diagonal block
        std::vector<float> ll(c * c);
        for (uint32_t t = 0; t < c; ++t)
            for (uint32_t u = 0; u < c; ++u)
                ll[t * c + u] = t / (2 * s) == u / (2 * s) && (t / s) % 2 == 1 && (u / s) % 2 == 0 ? 1.f : 0.f;
        out.push_back({"ll" + std::to_string(s), sq1, ll});
    }
    return out;
}

std::pair<TensorId, TensorId> gdn_chunk(Ops& o, const std::string& n, uint32_t C, const GdnChunkDims& d,
                                        const std::function<TensorId(const std::string&, Shape)>& cst,
                                        TensorId S, TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta) {
    auto& gr = o.graph();
    const uint64_t HK = d.HK, HV = d.HV, DK = d.DK, REP = d.HV / d.HK;
    const uint64_t c = C;
    const auto upper = cst("upper", {c, c}), strict = cst("strict", {1, c, c}), inclusive = cst("inclusive", {1, c, c});
    const auto lower = cst("lower", {1, c, c}), eye = cst("eye", {1, c, c}), diag8 = cst("diag8", {1, c, c});
    std::vector<TensorId> ll;
    for (uint32_t sz = 8; sz < C; sz *= 2) ll.push_back(cst("ll" + std::to_string(sz), {1, c, c}));

    const auto kc = o.transpose(k, {1, 0, 2}, n + "kc");   // [HK, C, DK]
    const auto qc = o.transpose(q, {1, 0, 2}, n + "qc");
    const auto vb0 = o.transpose(v, {1, 0, 2}, n + "vc");  // [HV, C, DV]
    const auto kk = o.batch_gemm(kc, kc, n + "kk", false, true, DType::F32);  // [HK, C, C]
    const auto qk = o.batch_gemm(qc, kc, n + "qk", false, true, DType::F32);
    auto widen = [&](TensorId x, TensorId factor, uint64_t r, uint64_t xc, uint64_t fc, const std::string& name) {
        const auto x4 = o.reshape(x, {HK, 1, r, xc}, name + "_x4");
        const auto f4 = o.reshape(factor, {HK, REP, r, fc}, name + "_f4");
        return o.reshape(o.mul(x4, f4, name + "_m"), {HV, r, xc}, name);
    };

    const auto gates = o.transpose(g, {1, 0}, n + "gates");
    const auto cum = o.temp(n + "cum", DType::F32, {HV, c});
    gr.gemm(gates, upper, cum);
    const auto g_col = o.reshape(cum, {HV, c, 1}, n + "gcol");
    const auto g_row = o.reshape(cum, {HV, 1, c}, n + "grow");
    const auto beta_col = o.reshape(o.transpose(beta, {1, 0}, n + "bt"), {HV, c, 1}, n + "bcol");
    const auto g_last = o.reshape(o.split(cum, 1, {c - 1, 1}, n + "gsplit")[1], {HV, 1, 1}, n + "glast");
    auto decay = [&](TensorId mask, const std::string& name) {
        return o.unary("exp", o.add(o.sub(g_col, g_row, n + name + "_d"), mask, n + name + "_m"), n + name);
    };
    const auto d_strict = decay(strict, "dstrict"), d_incl = decay(inclusive, "dincl");
    const auto gamma = o.unary("exp", g_col, n + "gamma");
    const auto tail = o.unary("exp", o.sub(g_last, g_col, n + "tl"), n + "tail");
    const auto gamma_last = o.unary("exp", g_last, n + "gl");
    const auto a = o.mul(widen(kk, d_strict, c, c, c, n + "au"), beta_col, n + "a");
    const auto p_intra = widen(qk, d_incl, c, c, c, n + "pin");
    const auto a8 = o.mul(a, diag8, n + "a8");
    const auto l8 = o.add(a8, eye, n + "l8");
    TensorId x = o.sub(eye, a8, n + "x0");
    for (int i = 1; i <= 2; ++i) {
        const auto st = n + "nt" + std::to_string(i);
        const auto res = o.mul(o.batch_gemm(l8, x, st + "lx", false, false, DType::F32), lower, st + "r");
        x = o.sub(x, o.batch_gemm(x, res, st + "st", false, false, DType::F32), st + "x");
    }
    for (size_t level = 0; level < ll.size(); ++level) {
        const auto st = n + "db" + std::to_string(level);
        const auto qx = o.batch_gemm(o.mul(a, ll[level], st + "q"), x, st + "qx", false, false, DType::F32);
        x = o.sub(x, o.batch_gemm(x, qx, st + "st", false, false, DType::F32), st + "x");
    }
    const auto kw = widen(kc, o.mul(beta_col, gamma, n + "ws"), c, DK, 1, n + "kw");
    const auto qd = widen(qc, gamma, c, DK, 1, n + "qd");
    const auto kd = widen(kc, tail, c, DK, 1, n + "kd");  // [HV, C, DK]
    const auto vb = o.mul(vb0, beta_col, n + "vb");
    const auto u = o.batch_gemm(x, vb, n + "u", false, false, DType::F32);   // [HV, C, DV]
    const auto w = o.batch_gemm(x, kw, n + "w", false, false, DType::F32);   // [HV, C, DK]
    // S is [HV, DV, DK] (value-major, as the decode kernel keeps it): S^T products use transpose_b
    const auto delta = o.sub(u, o.batch_gemm(w, S, n + "wst", false, true, DType::F32), n + "delta");  // [HV, C, DV]
    const auto out = o.add(o.batch_gemm(qd, S, n + "o1", false, true, DType::F32),
                           o.batch_gemm(p_intra, delta, n + "o2", false, false, DType::F32), n + "o");  // [HV, C, DV]
    const auto s_next = o.add(o.mul(S, gamma_last, n + "sd"), o.batch_gemm(delta, kd, n + "ds", true, false, DType::F32),
                              n + "sn");  // [HV, DV, DK]
    return {o.transpose(out, {1, 0, 2}, n + "core"), s_next};
}

std::pair<TensorId, TensorId> gdn_chunks(Ops& o, const std::string& n, uint32_t C, uint32_t nc, const GdnChunkDims& d,
                                         const std::function<TensorId(const std::string&, Shape)>& cst,
                                         TensorId S, TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta) {
    auto& gr = o.graph();
    const uint64_t HK = d.HK, HV = d.HV, DK = d.DK, DV = d.DV, REP = d.HV / d.HK;
    const uint64_t c = C, N = nc, BK = N * HK, BV = N * HV;
    const auto upper = cst("upper", {c, c}), strict = cst("strict", {1, c, c}), inclusive = cst("inclusive", {1, c, c});
    const auto lower = cst("lower", {1, c, c}), eye = cst("eye", {1, c, c}), diag8 = cst("diag8", {1, c, c});
    std::vector<TensorId> ll;
    for (uint32_t sz = 8; sz < C; sz *= 2) ll.push_back(cst("ll" + std::to_string(sz), {1, c, c}));
    // [N*C, H, X] -> [N*H, C, X] (chunk-major, then head)
    auto heads = [&](TensorId x, uint64_t H, uint64_t X, const std::string& name) {
        return o.reshape(o.transpose(o.reshape(x, {N, c, H, X}, name + "_4"), {0, 2, 1, 3}, name + "_t"), {N * H, c, X}, name);
    };
    const auto kc = heads(k, HK, DK, n + "kc");   // [N*HK, C, DK]
    const auto qc = heads(q, HK, DK, n + "qc");
    const auto vb0 = heads(v, HV, DV, n + "vc");  // [N*HV, C, DV]
    const auto kk = o.batch_gemm(kc, kc, n + "kk", false, true, DType::F32);  // [N*HK, C, C]
    const auto qk = o.batch_gemm(qc, kc, n + "qk", false, true, DType::F32);
    auto widen = [&](TensorId x, TensorId factor, uint64_t r, uint64_t xc, uint64_t fc, const std::string& name) {
        const auto x4 = o.reshape(x, {BK, 1, r, xc}, name + "_x4");
        const auto f4 = o.reshape(factor, {BK, REP, r, fc}, name + "_f4");
        return o.reshape(o.mul(x4, f4, name + "_m"), {BV, r, xc}, name);
    };
    auto rows = [&](TensorId t, const std::string& name) {
        return o.reshape(o.transpose(o.reshape(t, {N, c, HV}, name + "_3"), {0, 2, 1}, name + "_t"), {BV, c}, name);
    };
    const auto gates = rows(g, n + "gates");
    const auto cum = o.temp(n + "cum", DType::F32, {BV, c});
    gr.gemm(gates, upper, cum);
    const auto g_col = o.reshape(cum, {BV, c, 1}, n + "gcol");
    const auto g_row = o.reshape(cum, {BV, 1, c}, n + "grow");
    const auto beta_col = o.reshape(rows(beta, n + "bt"), {BV, c, 1}, n + "bcol");
    const auto g_last = o.reshape(o.split(cum, 1, {c - 1, 1}, n + "gsplit")[1], {BV, 1, 1}, n + "glast");
    auto decay = [&](TensorId mask, const std::string& name) {
        return o.unary("exp", o.add(o.sub(g_col, g_row, n + name + "_d"), mask, n + name + "_m"), n + name);
    };
    const auto d_strict = decay(strict, "dstrict"), d_incl = decay(inclusive, "dincl");
    const auto gamma = o.unary("exp", g_col, n + "gamma");
    const auto tail = o.unary("exp", o.sub(g_last, g_col, n + "tl"), n + "tail");
    const auto gamma_last = o.unary("exp", g_last, n + "gl");
    const auto a = o.mul(widen(kk, d_strict, c, c, c, n + "au"), beta_col, n + "a");
    const auto p_intra = widen(qk, d_incl, c, c, c, n + "pin");
    const auto a8 = o.mul(a, diag8, n + "a8");
    const auto l8 = o.add(a8, eye, n + "l8");
    TensorId x = o.sub(eye, a8, n + "x0");
    for (int i = 1; i <= 2; ++i) {
        const auto st = n + "nt" + std::to_string(i);
        const auto res = o.mul(o.batch_gemm(l8, x, st + "lx", false, false, DType::F32), lower, st + "r");
        x = o.sub(x, o.batch_gemm(x, res, st + "st", false, false, DType::F32), st + "x");
    }
    for (size_t level = 0; level < ll.size(); ++level) {
        const auto st = n + "db" + std::to_string(level);
        const auto qx = o.batch_gemm(o.mul(a, ll[level], st + "q"), x, st + "qx", false, false, DType::F32);
        x = o.sub(x, o.batch_gemm(x, qx, st + "st", false, false, DType::F32), st + "x");
    }
    const auto kw = widen(kc, o.mul(beta_col, gamma, n + "ws"), c, DK, 1, n + "kw");
    const auto qd = widen(qc, gamma, c, DK, 1, n + "qd");
    const auto kd = widen(kc, tail, c, DK, 1, n + "kd");  // [N*HV, C, DK]
    const auto vb = o.mul(vb0, beta_col, n + "vb");
    const auto u = o.batch_gemm(x, vb, n + "u", false, false, DType::F32);   // [N*HV, C, DV]
    const auto w = o.batch_gemm(x, kw, n + "w", false, false, DType::F32);   // [N*HV, C, DK]
    const std::vector<uint64_t> per(N, HV);
    const auto us = N == 1 ? std::vector<TensorId>{u} : o.split(u, 0, per, n + "us");
    const auto wsp = N == 1 ? std::vector<TensorId>{w} : o.split(w, 0, per, n + "wsp");
    const auto kds = N == 1 ? std::vector<TensorId>{kd} : o.split(kd, 0, per, n + "kds");
    const auto gls = N == 1 ? std::vector<TensorId>{gamma_last} : o.split(gamma_last, 0, per, n + "gls");
    std::vector<TensorId> entry, deltas;
    for (uint64_t i = 0; i < N; ++i) {
        const auto is = n + "s" + std::to_string(i);
        const auto delta = o.sub(us[i], o.batch_gemm(wsp[i], S, is + "wst", false, true, DType::F32), is + "delta");
        entry.push_back(S);
        deltas.push_back(delta);
        S = o.add(o.mul(S, gls[i], is + "sd"), o.batch_gemm(delta, kds[i], is + "ds", true, false, DType::F32), is + "sn");
    }
    const auto S_in = N == 1 ? entry[0] : o.concat(entry, 0, n + "sall");      // [N*HV, DV, DK]
    const auto delta = N == 1 ? deltas[0] : o.concat(deltas, 0, n + "dall");  // [N*HV, C, DV]
    const auto out = o.add(o.batch_gemm(qd, S_in, n + "o1", false, true, DType::F32),
                           o.batch_gemm(p_intra, delta, n + "o2", false, false, DType::F32), n + "o");
    const auto core = o.reshape(o.transpose(o.reshape(out, {N, HV, c, DV}, n + "o4"), {0, 2, 1, 3}, n + "ot"),
                                {N * c, HV, DV}, n + "core");
    return {core, S};
}

std::pair<TensorId, TensorId> gdn_chunks_fast(Ops& o, const std::string& n, uint32_t nc, const GdnChunkDims& d,
                                              const std::function<TensorId(const std::string&, Shape)>& cst,
                                              TensorId S, TensorId qc, TensorId kc, TensorId vc,
                                              TensorId g, TensorId beta) {
    auto& gr = o.graph();
    const uint64_t HK = d.HK, HV = d.HV, DK = d.DK, REP = HV / HK;
    const uint64_t c = 64, N = nc, BK = N * HK, BV = N * HV;

    // 1. BF16 MME Gram matrices -> BF16 [N*HK, 64, 64] reshaped to [N*HK, 32, 128] for 256B loads
    const auto kk = o.reshape(o.batch_gemm(kc, kc, n + "kk", false, true, DType::BF16), {BK, 32, 128}, n + "kkr");
    const auto qk = o.reshape(o.batch_gemm(qc, kc, n + "qk", false, true, DType::BF16), {BK, 32, 128}, n + "qkr");
    auto rows = [&](TensorId t, const std::string& nm) {
        return o.reshape(o.transpose(o.reshape(t, {N, c, HV}, nm + "_3"), {0, 2, 1}, nm + "_t"), {BV, c}, nm);
    };
    const auto gates = rows(g, n + "gates");
    const auto bt_in = rows(beta, n + "btin");
    const auto cum = o.temp(n + "cum", DType::F32, {BV, c});
    gr.gemm(gates, cst("upper", {c, c}), cum);

    // 2. Fused 24-TPC decay + mask + initial Newton/Doubling matrices in BF16 (ZERO scalar HBM loads!)
    const auto a_raw  = o.temp(n + "araw",  DType::BF16, {BV, 32, 128});
    const auto l8_raw = o.temp(n + "l8raw", DType::BF16, {BV, 32, 128});
    const auto x0_raw = o.temp(n + "x0raw", DType::BF16, {BV, 32, 128});
    const auto p_raw  = o.temp(n + "praw",  DType::BF16, {BV, 32, 128});
    const auto ws2    = o.temp(n + "ws2",   DType::BF16, {BV, c});
    const auto gm2    = o.temp(n + "gm2",   DType::BF16, {BV, c});
    const auto tl2    = o.temp(n + "tl2",   DType::BF16, {BV, c});
    const auto bt2    = o.temp(n + "bt2",   DType::BF16, {BV, c});
    const auto gl     = o.temp(n + "gl",    DType::F32,  {BV, 1, 1});
    gr.op("flashinfer_gaudi_gdn_prefill_decay_prep_bf16_gaudi2",
          {kk, qk, cum, bt_in}, {a_raw, l8_raw, x0_raw, p_raw, ws2, gm2, tl2, bt2, gl});

    const auto a       = o.reshape(a_raw,  {BV, c, c}, n + "a");
    const auto l8      = o.reshape(l8_raw, {BV, c, c}, n + "l8");
    TensorId x         = o.reshape(x0_raw, {BV, c, c}, n + "x0");
    const auto p_intra = o.reshape(p_raw,  {BV, c, c}, n + "pin");

    // 3. Newton (2x) + Doubling (3x) in pure BF16 MME (direct BF16 writeout, ZERO intermediate casts!)
    const auto lower_b = o.cast(cst("lower", {1, c, c}), DType::BF16, n + "lowb");
    for (int i = 1; i <= 2; ++i) {
        const auto st = n + "nt" + std::to_string(i);
        const auto res = o.mul(o.batch_gemm(l8, x, st + "lx", false, false, DType::BF16), lower_b, st + "r");
        x = o.sub(x, o.batch_gemm(x, res, st + "st", false, false, DType::BF16), st + "x");
    }
    for (uint32_t sz = 8; sz < 64; sz *= 2) {
        const auto st = n + "db" + std::to_string(sz);
        const auto ll_b = o.cast(cst("ll" + std::to_string(sz), {1, c, c}), DType::BF16, st + "llb");
        const auto qx = o.batch_gemm(o.mul(a, ll_b, st + "q"), x, st + "qx", false, false, DType::BF16);
        x = o.sub(x, o.batch_gemm(x, qx, st + "st", false, false, DType::BF16), st + "x");
    }

    // 4. BF16 scaled keys/queries/values -> u (F32 for delta accuracy), w (BF16 directly!)
    auto widen_bf16 = [&](TensorId src, TensorId fac2, const std::string& nm) {
        const auto x4 = o.reshape(src, {BK, 1, c, DK}, nm + "_x4");
        const auto f4 = o.reshape(fac2, {BK, REP, c, 1}, nm + "_f4");
        return o.reshape(o.mul(x4, f4, nm + "_m"), {BV, c, DK}, nm);
    };
    const auto kw = widen_bf16(kc, ws2, n + "kw");
    const auto qd = widen_bf16(qc, gm2, n + "qd");
    const auto kd = widen_bf16(kc, tl2, n + "kd");
    const auto vb = o.mul(vc, o.reshape(bt2, {BV, c, 1}, n + "bt3"), n + "vb");
    const auto u  = o.batch_gemm(x, vb, n + "u", false, false, DType::F32);   // [BV, 64, 128] F32
    const auto w  = o.batch_gemm(x, kw, n + "w", false, false, DType::BF16);  // [BV, 64, 128] BF16

    // 5. State chain: FP32 state S, BF16 MME products (w @ Sb^T -> F32, delta_b^T @ kd -> F32)
    const std::vector<uint64_t> per(N, HV);
    const auto us  = N == 1 ? std::vector<TensorId>{u}  : o.split(u,  0, per, n + "us");
    const auto wsp = N == 1 ? std::vector<TensorId>{w}  : o.split(w,  0, per, n + "wsp");
    const auto kds = N == 1 ? std::vector<TensorId>{kd} : o.split(kd, 0, per, n + "kds");
    const auto gls = N == 1 ? std::vector<TensorId>{gl} : o.split(gl, 0, per, n + "gls");
    std::vector<TensorId> entry, deltas;
    entry.reserve(N);
    deltas.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        const auto is = n + "s" + std::to_string(i);
        const auto Sb = o.cast(S, DType::BF16, is + "Sb");
        const auto delta = o.sub(us[i], o.batch_gemm(wsp[i], Sb, is + "wst", false, true, DType::F32), is + "delta");
        const auto delta_b = o.cast(delta, DType::BF16, is + "db");
        entry.push_back(Sb);
        deltas.push_back(delta_b);
        S = o.add(o.mul(S, gls[i], is + "sd"), o.batch_gemm(delta_b, kds[i], is + "ds", true, false, DType::F32), is + "sn");
    }
    const auto S_in  = N == 1 ? entry[0]  : o.concat(entry,  0, n + "sall");
    const auto delta = N == 1 ? deltas[0] : o.concat(deltas, 0, n + "dall");
    const auto out_c = o.add(o.batch_gemm(qd, S_in, n + "o1", false, true, DType::BF16),
                             o.batch_gemm(p_intra, delta, n + "o2", false, false, DType::BF16), n + "o");  // [BV, 64, 128] BF16
    return {out_c, S};
}

}  // namespace g2e
