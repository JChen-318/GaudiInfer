#include "g2e/gdn.hpp"

namespace g2e {

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
    const uint64_t HK = d.HK, HV = d.HV, DK = d.DK, DV = d.DV, REP = d.HV / d.HK;
    const uint64_t c = C;
    const auto upper = cst("upper", {c, c}), strict = cst("strict", {1, c, c}), inclusive = cst("inclusive", {1, c, c});
    const auto lower = cst("lower", {1, c, c}), eye = cst("eye", {1, c, c}), diag8 = cst("diag8", {1, c, c});
    std::vector<TensorId> ll;
    for (uint32_t sz = 8; sz < C; sz *= 2) ll.push_back(cst("ll" + std::to_string(sz), {1, c, c}));
    const auto gates = o.transpose(g, {1, 0}, n + "gates");                    // [HV, C]
    const auto cum = o.temp(n + "cum", DType::F32, {HV, c});
    gr.gemm(gates, upper, cum);                                                // G = cumulative sum
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
    const auto kc = o.transpose(k, {1, 0, 2}, n + "kc");   // [HK, C, DK]
    const auto qc = o.transpose(q, {1, 0, 2}, n + "qc");
    const auto vb0 = o.transpose(v, {1, 0, 2}, n + "vc");  // [HV, C, DV]
    const auto kk = o.batch_gemm(kc, kc, n + "kk", false, true, DType::F32);  // [HK, C, C]
    const auto qk = o.batch_gemm(qc, kc, n + "qk", false, true, DType::F32);
    // compact [HK, R, X] times a per-value-head factor [HV, R, F] -> [HV, R, X]; value head h uses key head h/REP
    auto widen = [&](TensorId x, TensorId factor, uint64_t r, uint64_t xc, uint64_t fc, const std::string& name) {
        const auto x4 = o.reshape(x, {HK, 1, r, xc}, name + "_x4");
        const auto f4 = o.reshape(factor, {HK, REP, r, fc}, name + "_f4");
        return o.reshape(o.mul(x4, f4, name + "_m"), {HV, r, xc}, name);
    };
    const auto a = o.mul(widen(kk, d_strict, c, c, c, n + "au"), beta_col, n + "a");
    const auto p_intra = widen(qk, d_incl, c, c, c, n + "pin");
    const auto kw = widen(kc, o.mul(beta_col, gamma, n + "ws"), c, DK, 1, n + "kw");
    const auto qd = widen(qc, gamma, c, DK, 1, n + "qd");
    const auto kd = widen(kc, tail, c, DK, 1, n + "kd");  // [HV, C, DK]
    const auto vb = o.mul(vb0, beta_col, n + "vb");
    // (I + A)^-1: 8x8 diagonal blocks by Newton's iteration, then exact doublings
    const auto a8 = o.mul(a, diag8, n + "a8");
    const auto l8 = o.add(a8, eye, n + "l8");
    auto x = o.sub(eye, a8, n + "x0");
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
    auto rows = [&](TensorId x, const std::string& name) {  // [N*C, HV] -> [N*HV, C]
        return o.reshape(o.transpose(o.reshape(x, {N, c, HV}, name + "_3"), {0, 2, 1}, name + "_t"), {BV, c}, name);
    };
    const auto gates = rows(g, n + "gates");                                   // [N*HV, C]
    const auto cum = o.temp(n + "cum", DType::F32, {BV, c});
    gr.gemm(gates, upper, cum);                                                // G = cumulative sum per chunk
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
    const auto a = o.mul(widen(kk, d_strict, c, c, c, n + "au"), beta_col, n + "a");
    const auto p_intra = widen(qk, d_incl, c, c, c, n + "pin");
    const auto kw = widen(kc, o.mul(beta_col, gamma, n + "ws"), c, DK, 1, n + "kw");
    const auto qd = widen(qc, gamma, c, DK, 1, n + "qd");
    const auto kd = widen(kc, tail, c, DK, 1, n + "kd");  // [N*HV, C, DK]
    const auto vb = o.mul(vb0, beta_col, n + "vb");
    const auto a8 = o.mul(a, diag8, n + "a8");
    const auto l8 = o.add(a8, eye, n + "l8");
    auto x = o.sub(eye, a8, n + "x0");
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
    const auto u = o.batch_gemm(x, vb, n + "u", false, false, DType::F32);   // [N*HV, C, DV]
    const auto w = o.batch_gemm(x, kw, n + "w", false, false, DType::F32);   // [N*HV, C, DK]
    // the state chain: delta_i = u_i - w_i S_i^T, S_{i+1} = gamma_last_i S_i + delta_i^T kd_i
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
                           o.batch_gemm(p_intra, delta, n + "o2", false, false, DType::F32), n + "o");  // [N*HV, C, DV]
    const auto core = o.reshape(o.transpose(o.reshape(out, {N, HV, c, DV}, n + "o4"), {0, 2, 1, 3}, n + "ot"),
                                {N * c, HV, DV}, n + "core");
    return {core, S};
}

}  // namespace g2e
