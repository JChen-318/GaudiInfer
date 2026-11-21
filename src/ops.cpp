#include <cstdlib>
#include "g2e/ops.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <stdexcept>

namespace g2e {

const char* suffix(DType t) {
    switch (t) {
        case DType::F32: return "f32";
        case DType::BF16: return "bf16";
        case DType::F16: return "f16";
        case DType::F8: return "f8";
        case DType::I32: return "i32";
        case DType::I64: return "i64";
        case DType::I16: return "i16";
        case DType::I8: return "i8";
        case DType::U8: return "u8";
    }
    return "?";
}

namespace {
std::string guid(const char* base, DType t) { return std::string(base) + "_" + suffix(t); }
unsigned syn_axis(size_t rank, unsigned axis) {
    if (axis >= rank) throw std::invalid_argument("g2e ops: axis out of range");
    return unsigned(rank - 1 - axis);
}
Shape broadcast(const Shape& a, const Shape& b) {
    const size_t r = std::max(a.size(), b.size());
    Shape out(r);
    for (size_t i = 0; i < r; ++i) {
        const uint64_t da = i < r - a.size() ? 1 : a[i - (r - a.size())];
        const uint64_t db = i < r - b.size() ? 1 : b[i - (r - b.size())];
        if (da != db && da != 1 && db != 1) throw std::invalid_argument("g2e ops: shapes do not broadcast");
        out[i] = std::max(da, db);
    }
    return out;
}
}  // namespace

TensorId Ops::temp(const std::string& name, DType dtype, Shape shape) { return g_.temp(n(name), dtype, std::move(shape)); }

TensorId Ops::binary(const char* op, TensorId a, TensorId b, const std::string& name) {
    if (dtype(a) != dtype(b)) throw std::invalid_argument(std::string("g2e ops: ") + op + " dtype mismatch at " + name);
    const auto out = temp(name, dtype(a), broadcast(shape(a), shape(b)));
    const auto gd = guid((std::string(op) + "_fwd").c_str(), dtype(a));
    g_.op(gd.c_str(), {a, b}, {out});
    return out;
}

TensorId Ops::unary(const char* op, TensorId x, const std::string& name) {
    const auto out = temp(name, dtype(x), shape(x));
    const auto gd = guid((std::string(op) + "_fwd").c_str(), dtype(x));
    if (std::string(op) == "softplus") {
        ns_Softplus::Params p{};
        p.beta = 1.0f;
        p.threshold = 20.0f;
        g_.op(gd.c_str(), {x}, {out}, &p, sizeof(p));
    } else {
        g_.op(gd.c_str(), {x}, {out});
    }
    return out;
}

TensorId Ops::constant(float value, Shape s, const std::string& name, DType dt) {
    const auto out = temp(name, dt, std::move(s));
    ns_ConstantKernel::Params p{};
    if (dt == DType::F32 || dt == DType::BF16) p.constant.f = value;
    else p.constant.i = int(value);
    const auto gd = guid("constant", dt);
    g_.op(gd.c_str(), std::span<const TensorId>(), std::span<const TensorId>(&out, 1), &p, sizeof(p));
    return out;
}

TensorId Ops::cast(TensorId x, DType to, const std::string& name) {
    if (dtype(x) == to) return x;
    const auto out = temp(name, to, shape(x));
    const auto gd = std::string("cast_") + suffix(dtype(x)) + "_to_" + suffix(to);
    ns_CastKernel::Params p{};
    p.round_mode = CAST_ROUND_HALF_NE;
    g_.op(gd.c_str(), {x}, {out}, &p, sizeof(p));
    return out;
}

TensorId Ops::reshape(TensorId x, Shape s, const std::string& name) {
    const auto out = temp(name, dtype(x), std::move(s));
    g_.reshape(x, out);
    return out;
}

TensorId Ops::transpose(TensorId x, const std::vector<unsigned>& perm, const std::string& name) {
    const auto& in = shape(x);
    const size_t r = in.size();
    if (perm.size() != r) throw std::invalid_argument("g2e ops: transpose rank");
    Shape out_shape(r);
    synTransposeParams p{};
    p.tensorDim = unsigned(r);
    for (size_t i = 0; i < r; ++i) {
        out_shape[i] = in[perm[i]];
        p.permutation[r - 1 - i] = static_cast<TransposePermutationDim>(r - 1 - perm[i]);
    }
    const auto out = temp(name, dtype(x), out_shape);
    g_.op("transpose", {x}, {out}, &p, sizeof(p));
    return out;
}

std::vector<TensorId> Ops::split(TensorId x, unsigned axis, const std::vector<uint64_t>& sizes, const std::string& name) {
    const auto& in = shape(x);
    uint64_t total = 0;
    for (auto s : sizes) total += s;
    if (axis >= in.size() || total != in[axis]) throw std::invalid_argument("g2e ops: split sizes at " + name);
    std::vector<TensorId> outs;
    for (size_t i = 0; i < sizes.size(); ++i) {
        Shape s = in;
        s[axis] = sizes[i];
        outs.push_back(temp(name + std::to_string(i), dtype(x), s));
    }
    synSplitParams p{};
    p.axis = syn_axis(in.size(), axis);
    g_.op("split", std::span<const TensorId>(&x, 1), outs, &p, sizeof(p));
    return outs;
}

TensorId Ops::concat(const std::vector<TensorId>& xs, unsigned axis, const std::string& name) {
    Shape s = shape(xs.at(0));
    uint64_t total = 0;
    for (auto t : xs) total += shape(t).at(axis);
    s[axis] = total;
    const auto out = temp(name, dtype(xs[0]), s);
    synConcatenateParams p{};
    p.axis = syn_axis(s.size(), axis);
    g_.op("concat", xs, std::span<const TensorId>(&out, 1), &p, sizeof(p));
    return out;
}

TensorId Ops::reduce(const char* op, TensorId x, unsigned axis, const std::string& name) {
    Shape s = shape(x);
    const unsigned a = syn_axis(s.size(), axis);
    s[axis] = 1;
    const auto out = temp(name, dtype(x), s);
    ns_Reduction::Params p{};
    p.reductionDimension = a;
    const auto gd = guid((std::string("reduce_") + op + "_fwd").c_str(), dtype(x));
    g_.op(gd.c_str(), {x}, {out}, &p, sizeof(p));
    return out;
}

TensorId Ops::softmax(TensorId x, const std::string& name) {
    const auto out = temp(name, dtype(x), shape(x));
    ns_Softmax::Params p{};
    p.dim = 0;
    const auto gd = guid("softmax_fwd", dtype(x));
    g_.op(gd.c_str(), {x}, {out}, &p, sizeof(p));
    return out;
}

TensorId Ops::linear_bf16(TensorId x, TensorId w, const std::string& name, DType out_type) {
    const auto& xs = shape(x);
    const auto& ws = shape(w);
    const auto out = temp(name, out_type, {xs.at(0), ws.at(0)});
    g_.gemm(x, w, out, false, true);
    return out;
}

TensorId Ops::fp8_linear(TensorId x, TensorId w, TensorId ws, const std::string& name) {
    const auto& xs = shape(x);
    const auto& wsh = shape(w);
    if (xs.size() != 2 || wsh.size() != 2 || xs[1] != wsh[1] || dtype(w) != DType::F8 || dtype(ws) != DType::F32)
        throw std::invalid_argument("g2e fp8_linear: x[M,K] bf16, w[N,K] f8, ws[N] f32 at " + name);
    const uint64_t M = xs[0], N = wsh[0];
    const auto xb = cast(x, DType::BF16, name + "_xb");
    const auto peak = cast(reduce("max", unary("abs", xb, name + "_abs"), 1, name + "_peak"), DType::F32, name + "_peakf");
    const auto eps = constant(1e-8f, {1}, name + "_eps");
    const auto inv240 = constant(1.0f / 240.0f, {1}, name + "_inv240");
    const auto scale = mul(add(peak, eps, name + "_pe"), inv240, name + "_scale");  // [M,1]
    const auto inv = unary("reciprocal", scale, name + "_inv");
    const auto codes = temp(name + "_codes", DType::F8, {M, xs[1]});
    {
        ns_ConvertToFp8::ParamsV2 p{};
        p.round_mode = CAST_ROUND_HALF_NE;
        g_.op("convert_to_fp8_bf16", {xb, inv}, {codes}, &p, sizeof(p));
    }
    const auto out = temp(name, DType::BF16, {M, N});
    ns_Fp8Gemm::ParamsV2 p{};
    p.transpose_a = false;
    p.transpose_b = true;
    // G2E_TIMING_NO_DEQUANT=1 (timing experiment only, wrong numerics): no scale inputs (no dequant epilogue)
    static const bool no_dq = std::getenv("G2E_TIMING_NO_DEQUANT") != nullptr;
    // G2E_TIMING_SCALAR_SCALES=1 (timing experiment only, wrong numerics): per-tensor scales (the first row's / column's)
    static const bool scalar = std::getenv("G2E_TIMING_SCALAR_SCALES") != nullptr;
    if (no_dq) {
        g_.op("fp8_gemm_bf16", {codes, w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {out}, &p, sizeof(p));
    } else if (scalar) {
        const Shape ss = shape(scale), wss = shape(ws);
        uint64_t se = 1, we = 1;
        for (auto d : ss) se *= d;
        for (auto d : wss) we *= d;
        const auto sf = reshape(scale, {se}, name + "_sflat");
        const auto s1 = se > 1 ? split(sf, 0, {1, se - 1}, name + "_s1")[0] : sf;
        const auto wf = reshape(ws, {we}, name + "_wflat");
        const auto w1 = we > 1 ? split(wf, 0, {1, we - 1}, name + "_w1")[0] : wf;
        g_.op("fp8_gemm_bf16", {codes, w, s1, w1, kNoTensor, kNoTensor, kNoTensor}, {out}, &p, sizeof(p));
    } else {
        g_.op("fp8_gemm_bf16", {codes, w, scale, ws, kNoTensor, kNoTensor, kNoTensor}, {out}, &p, sizeof(p));
    }
    return out;
}

TensorId Ops::rms_norm(TensorId x, TensorId w, float eps, bool plus_one, const std::string& name) {
    const auto out = temp(name, dtype(x), shape(x));
    Shape rs = shape(x);
    rs.back() = 1;
    const auto inv = temp(name + "_invrms", DType::F32, rs);
    ns_LayerNormKernel::ParamsRmsNormV3 p{};
    p.epsValid = true;
    p.eps = eps;
    p.fastMath = false;
    p.hasResidual = false;
    p.addOneToWeight = plus_one;
    const auto gd = guid("rms_norm_ex_fwd", dtype(x));
    g_.op(gd.c_str(), {x, w}, {out, inv}, &p, sizeof(p));
    return out;
}

TensorId Ops::layer_norm(TensorId x, TensorId w, TensorId b, float eps, const std::string& name) {
    const Shape s = shape(x);
    if (s.size() != 2) throw std::invalid_argument("g2e layer_norm: x [M, N] at " + name);
    const auto x4 = reshape(x, {1, 1, s[0], s[1]}, name + "_x4");
    const auto y4 = temp(name + "_y4", dtype(x), {1, 1, s[0], s[1]});
    const auto mean = temp(name + "_mean", DType::F32, {1, 1, s[0], 1});
    const auto rstd = temp(name + "_rstd", DType::F32, {1, 1, s[0], 1});
    ns_LayerNormKernel::Params p{};
    p.epsValid = true;
    p.eps = eps;
    const auto gd = guid("layer_norm_fwd", dtype(x));
    const TensorId ins[] = {x4, b, w};  // bias before weight (as the PyTorch bridge)
    const TensorId outs[] = {y4, mean, rstd};
    g_.op(gd.c_str(), ins, outs, &p, sizeof(p));
    return reshape(y4, s, name);
}

TensorId Ops::gelu(TensorId x, bool tanh_approx, const std::string& name) {
    const auto out = temp(name, dtype(x), shape(x));
    const auto keep = temp(name + "_keep", dtype(x), shape(x));
    ns_GeluKernel::Params p{};
    p.approximation = tanh_approx;
    const auto gd = guid("gelu_fwd", dtype(x));
    const TensorId outs[] = {out, keep};
    g_.op(gd.c_str(), std::span<const TensorId>(&x, 1), outs, &p, sizeof(p));
    return out;
}

TensorId Ops::silu(TensorId x, const std::string& name) { return mul(x, unary("sigmoid", x, name + "_sig"), name); }

TensorId Ops::argmax(TensorId x, const std::string& name) {
    Shape s = shape(x);
    s.back() = 1;
    const auto out = temp(name, DType::I32, s);
    ns_Reduction::ParamsV2 p{};
    p.reductionDimensionMask = 1;  // innermost (TPC dimension 0)
    p.keepDim = true;
    const auto gd = guid("argmax_multi_dim_fwd", dtype(x));
    g_.op(gd.c_str(), {x}, {out}, &p, sizeof(p));
    return out;
}

std::pair<TensorId, TensorId> Ops::topk(TensorId x, uint32_t k, const std::string& name) {
    Shape s = shape(x);
    s.back() = k;
    const auto values = temp(name + "_v", dtype(x), s);
    const auto idx = temp(name + "_i", DType::I32, s);
    synBeamParams p{};
    p.bsw = k;
    p.axis = 0;
    p.bottomK = false;
    g_.op("topk", {x}, {values, idx}, &p, sizeof(p));
    return {values, idx};
}

std::pair<TensorId, TensorId> Ops::topk2(TensorId x, uint32_t k, uint32_t chunk, TensorId consts, const std::string& name,
                                         TensorId precomputed_cmax) {
    const Shape s = shape(x);
    if (s.size() != 2 || s[1] % chunk || s[1] / chunk < k) throw std::invalid_argument("topk2: [R, V], V % chunk == 0");
    const uint64_t R = s[0], V = s[1], nc = V / chunk, K = k;
    const auto cp = split(consts, 0, {V, R, R}, name + "_c");  // iota, r * nc, r * k * chunk
    const auto x2 = reshape(x, {R * nc, chunk}, name + "_x2");
    const auto cmax = precomputed_cmax != kNoTensor
                          ? precomputed_cmax
                          : reshape(reduce("max", reshape(x, {R, nc, chunk}, name + "_x3"), 2, name + "_cm"), {R, nc}, name + "_cm2");
    const auto ci = topk(cmax, k, name + "_1").second;                                                    // [R, k] chunks
    const auto rows = reshape(add(ci, reshape(cp[1], {R, 1}, name + "_r1"), name + "_cr"), {R * K}, name + "_crf");
    const auto vals = reshape(gather_rows(x2, rows, name + "_gv"), {R, K * chunk}, name + "_gv2");      // their values
    const auto ids = reshape(gather_rows(reshape(cp[0], {nc, chunk}, name + "_io"), reshape(ci, {R * K}, name + "_cif"), name + "_gi"),
                             {R * K * chunk, 1}, name + "_gi2");                                          // and columns
    const auto [v, p] = topk(vals, k, name + "_2");                                                      // [R, k]
    const auto at = reshape(add(p, reshape(cp[2], {R, 1}, name + "_r2"), name + "_pr"), {R * K}, name + "_prf");
    return {v, reshape(gather_rows(ids, at, name + "_ids"), {R, K}, name + "_i")};
}

std::vector<int32_t> Ops::topk2_constants(uint64_t rows, uint64_t cols, uint32_t k, uint32_t chunk) {
    const uint64_t nc = cols / chunk;
    std::vector<int32_t> c(cols + 2 * rows);
    for (uint64_t i = 0; i < cols; ++i) c[i] = int32_t(i);
    for (uint64_t r = 0; r < rows; ++r) {
        c[cols + r] = int32_t(r * nc);
        c[cols + rows + r] = int32_t(r * k * chunk);
    }
    return c;
}

TensorId Ops::gather(TensorId table, TensorId idx, unsigned axis, const std::string& name) {
    Shape s = shape(table);
    s[axis] = shape(idx)[0];
    const auto out = temp(name, dtype(table), s);
    ns_GatherKernel::Params p{};
    p.axis = int(syn_axis(s.size(), axis));
    const auto gd = guid("gather_fwd", dtype(table));
    g_.op(gd.c_str(), {table, idx}, {out}, &p, sizeof(p));
    return out;
}

TensorId Ops::gather_rows(TensorId table, TensorId idx, const std::string& name) {
    Shape s = shape(table);
    const Shape is = shape(idx);
    s[0] = is[0];
    const auto out = temp(name, dtype(table), s);
    ns_GatherKernel::Params p{};
    p.axis = int(s.size()) - 1;  // the outermost axis in TPC order
    const auto gd = guid("gather_fwd", dtype(table));
    g_.op(gd.c_str(), {table, idx}, {out}, &p, sizeof(p));
    return out;
}

namespace {
struct AddRmsNormQuantParams {  // tpc/flashinfer_gaudi/norm_quant_params.h
    float epsilon, inverse_width, inverse_range;
};
constexpr float kInverseRangeBf16 = 0.004180908203125f;  // bf16(1/240), as vLLM's row quantization
}  // namespace

Ops::NormQuant Ops::add_rmsnorm_quant(TensorId x, TensorId residual, TensorId w, float eps, const std::string& name,
                                      bool need_normalized) {
    const Shape s = shape(x);
    if (s.size() != 2 || dtype(x) != DType::BF16 || dtype(residual) != DType::BF16 || dtype(w) != DType::BF16)
        throw std::invalid_argument("g2e add_rmsnorm_quant: BF16 [rows, width] at " + name);
    NormQuant r;
    r.codes = temp(name + "_codes", DType::F8, s);
    r.scale = temp(name + "_scale", DType::F32, {s[0], 1});
    r.residual_out = temp(name + "_res", DType::BF16, s);
    AddRmsNormQuantParams p{eps, 1.0f / float(s[1]), kInverseRangeBf16};
    if (!need_normalized && s[1] <= 8192) {
        r.normalized = kNoTensor;
        g_.op("flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2", {x, residual, w},
              {r.codes, r.scale, r.residual_out}, &p, sizeof(p));
    } else {
        r.normalized = temp(name + "_norm", DType::BF16, s);
        g_.op("flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2", {x, residual, w},
              {r.codes, r.scale, r.normalized, r.residual_out}, &p, sizeof(p));
    }
    return r;
}

Ops::NormQuant Ops::fp8_gemm_add_rmsnorm_quant(TensorId codes, TensorId scale, TensorId w, TensorId ws,
                                               TensorId residual, TensorId norm_w, float eps, const std::string& name) {
    const Shape cs = shape(codes), wsh = shape(w);
    if (cs.size() != 2 || wsh.size() != 2 || cs[1] != wsh[1] || dtype(codes) != DType::F8 || dtype(w) != DType::F8)
        throw std::invalid_argument("g2e fp8_gemm_add_rmsnorm_quant at " + name);
    const uint64_t M = cs[0], N = wsh[0];
    const auto raw = temp(name + "_raw", DType::BF16, {M, N});
    ns_Fp8Gemm::ParamsV2 gp{};
    gp.transpose_a = false;
    gp.transpose_b = true;
    g_.op("fp8_gemm_bf16", {codes, w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {raw}, &gp, sizeof(gp));
    NormQuant r;
    r.codes = temp(name + "_codes", DType::F8, {M, N});
    r.scale = temp(name + "_scale", DType::F32, {M, 1});
    r.normalized = kNoTensor;
    r.residual_out = temp(name + "_res", DType::BF16, {M, N});
    AddRmsNormQuantParams p{eps, 1.0f / float(N), kInverseRangeBf16};
    g_.op("flashinfer_gaudi_add_rmsnorm_quant_bf16_gaudi2",
          {raw, scale, cast(ws, DType::BF16, name + "_wsb"), residual, norm_w},
          {r.codes, r.scale, r.residual_out}, &p, sizeof(p));
    return r;
}

TensorId Ops::fp8_gemm(TensorId codes, TensorId scale, TensorId w, TensorId ws, const std::string& name) {
    const Shape cs = shape(codes), wsh = shape(w);
    if (cs.size() != 2 || wsh.size() != 2 || cs[1] != wsh[1] || dtype(codes) != DType::F8 || dtype(w) != DType::F8)
        throw std::invalid_argument("g2e fp8_gemm: codes[M,K] f8, w[N,K] f8 at " + name);
    const auto out = temp(name, DType::BF16, {cs[0], wsh[0]});
    ns_Fp8Gemm::ParamsV2 p{};
    p.transpose_a = false;
    p.transpose_b = true;
    // G2E_TIMING_NO_DEQUANT=1 (timing experiment only, wrong numerics): no scale inputs (no dequant epilogue)
    static const bool no_dq = std::getenv("G2E_TIMING_NO_DEQUANT") != nullptr;
    // G2E_TIMING_SCALAR_SCALES=1 (timing experiment only, wrong numerics): per-tensor scales (the first row's / column's)
    static const bool scalar = std::getenv("G2E_TIMING_SCALAR_SCALES") != nullptr;
    if (no_dq) {
        g_.op("fp8_gemm_bf16", {codes, w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {out}, &p, sizeof(p));
    } else if (scalar) {
        const Shape ss = shape(scale), wss = shape(ws);
        uint64_t se = 1, we = 1;
        for (auto d : ss) se *= d;
        for (auto d : wss) we *= d;
        const auto sf = reshape(scale, {se}, name + "_sflat");
        const auto s1 = se > 1 ? split(sf, 0, {1, se - 1}, name + "_s1")[0] : sf;
        const auto wf = reshape(ws, {we}, name + "_wflat");
        const auto w1 = we > 1 ? split(wf, 0, {1, we - 1}, name + "_w1")[0] : wf;
        g_.op("fp8_gemm_bf16", {codes, w, s1, w1, kNoTensor, kNoTensor, kNoTensor}, {out}, &p, sizeof(p));
    } else {
        g_.op("fp8_gemm_bf16", {codes, w, scale, ws, kNoTensor, kNoTensor, kNoTensor}, {out}, &p, sizeof(p));
    }
    return out;
}

std::pair<TensorId, TensorId> Ops::quantize(TensorId x, const std::string& name) {
    const Shape xs = shape(x);
    const auto xb = cast(x, DType::BF16, name + "_xb");
    const auto peak = cast(reduce("max", unary("abs", xb, name + "_abs"), 1, name + "_peak"), DType::F32, name + "_peakf");
    const auto eps = constant(1e-8f, {1}, name + "_eps");
    const auto inv240 = constant(kInverseRangeBf16, {1}, name + "_inv240");
    const auto scale = mul(add(peak, eps, name + "_pe"), inv240, name + "_scale");
    const auto inv = unary("reciprocal", scale, name + "_inv");
    const auto codes = temp(name + "_codes", DType::F8, xs);
    ns_ConvertToFp8::ParamsV2 p{};
    p.round_mode = CAST_ROUND_HALF_NE;
    g_.op("convert_to_fp8_bf16", {xb, inv}, {codes}, &p, sizeof(p));
    return {codes, scale};
}

std::pair<TensorId, TensorId> Ops::mul_quant(TensorId a, TensorId b, const std::string& name) {
    const Shape s = shape(a);
    if (s.size() != 2 || shape(b) != s || dtype(a) != DType::BF16 || dtype(b) != DType::BF16)
        throw std::invalid_argument("g2e mul_quant: a, b BF16 [M,F] at " + name);
    static const bool old_smq = std::getenv("G2E_OLD_SMQ") != nullptr;
    const auto codes = temp(name + "_codes", DType::F8, s);
    const auto scale = temp(name + "_scale", DType::F32, {s[0], 1});
    if (old_smq) {
        const auto ab = concat({a, b}, 1, name + "_ab");
        g_.op("flashinfer_gaudi_silu_mul_quant_bf16_gaudi2", {ab, a}, {codes, scale});
    } else {
        g_.op("flashinfer_gaudi_silu_mul_quant_bf16_gaudi2", {b, a}, {codes, scale});
    }
    return {codes, scale};
}

std::pair<TensorId, TensorId> Ops::fp8_gemm_silu_mul_quant(TensorId codes, TensorId scale, TensorId w, TensorId ws,
                                                           const std::string& name) {
    const Shape cs = shape(codes), wsh = shape(w);
    if (cs.size() != 2 || wsh.size() != 2 || cs[1] != wsh[1] || wsh[0] % 2 || dtype(codes) != DType::F8 || dtype(w) != DType::F8)
        throw std::invalid_argument("g2e fp8_gemm_silu_mul_quant at " + name);
    const uint64_t M = cs[0], F = wsh[0] / 2;
    const auto raw = temp(name + "_raw", DType::BF16, {M, 2 * F});
    ns_Fp8Gemm::ParamsV2 p{};
    p.transpose_a = false;
    p.transpose_b = true;
    g_.op("fp8_gemm_bf16", {codes, w, kNoTensor, kNoTensor, kNoTensor, kNoTensor, kNoTensor}, {raw}, &p, sizeof(p));
    const auto out = temp(name + "_codes", DType::F8, {M, F});
    const auto rs = temp(name + "_scale", DType::F32, {M, 1});
    g_.op("flashinfer_gaudi_silu_mul_quant_dq_bf16_gaudi2", {raw, scale, cast(ws, DType::BF16, name + "_wsb")}, {out, rs});
    return {out, rs};
}

std::pair<TensorId, TensorId> Ops::silu_mul_quant(TensorId gu, const std::string& name) {
    const Shape s = shape(gu);
    if (s.size() != 2 || s[1] % 2 || dtype(gu) != DType::BF16) throw std::invalid_argument("g2e silu_mul_quant at " + name);
    const uint64_t F = s[1] / 2;
    const auto halves = split(gu, 1, {F, F}, name + "_gu");
    const auto act = temp(name + "_silu", DType::BF16, {s[0], F});
    g_.op("silu_fwd_bf16", {halves[0]}, {act});
    const auto codes = temp(name + "_codes", DType::F8, {s[0], F});
    const auto scale = temp(name + "_scale", DType::F32, {s[0], 1});
    g_.op("flashinfer_gaudi_silu_mul_quant_bf16_gaudi2", {gu, act}, {codes, scale});
    return {codes, scale};
}

TensorId Ops::batch_gemm(TensorId a, TensorId b, const std::string& name, bool ta, bool tb, DType out_type) {
    const auto& as = shape(a);
    const auto& bs = shape(b);
    Shape o = as;
    const size_t r = as.size();
    o[r - 2] = as[ta ? r - 1 : r - 2];
    o[r - 1] = bs[tb ? r - 2 : r - 1];
    const auto out = temp(name, out_type, o);
    g_.batch_gemm(a, b, out, ta, tb);
    return out;
}

}  // namespace g2e
