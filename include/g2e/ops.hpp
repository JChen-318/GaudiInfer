// g2e ops: model building blocks on top of GraphBuilder. Every helper creates its graph-internal output tensor
// (named <prefix><name>) and returns its id. Shapes are outermost-first; `axis` arguments are outermost-first
// too and converted to Synapse's innermost-first dimension index internally.
#pragma once

#include "g2e/graph.hpp"

#include <string>
#include <vector>

namespace g2e {

using Shape = std::vector<uint64_t>;

class Ops {
public:
    Ops(GraphBuilder& graph, std::string prefix = "") : g_(graph), prefix_(std::move(prefix)) {}
    GraphBuilder& graph() { return g_; }
    Shape shape(TensorId t) const { return g_.desc(t).shape; }  // a copy: creating tensors may reallocate descriptors
    DType dtype(TensorId t) const { return g_.desc(t).dtype; }

    TensorId temp(const std::string& name, DType dtype, Shape shape);
    // Elementwise with numpy-style broadcasting; output shape is the broadcast shape, dtype as `a` unless given.
    TensorId binary(const char* op, TensorId a, TensorId b, const std::string& name);  // op: add, sub, mult, div, max, min
    TensorId add(TensorId a, TensorId b, const std::string& n) { return binary("add", a, b, n); }
    TensorId sub(TensorId a, TensorId b, const std::string& n) { return binary("sub", a, b, n); }
    TensorId mul(TensorId a, TensorId b, const std::string& n) { return binary("mult", a, b, n); }
    TensorId div(TensorId a, TensorId b, const std::string& n) { return binary("div", a, b, n); }
    TensorId unary(const char* op, TensorId x, const std::string& name);  // exp, sigmoid, abs, reciprocal, sqrt, rsqrt, neg, log, softplus, silu, tanh
    TensorId constant(float value, Shape shape, const std::string& name, DType dtype = DType::F32);
    TensorId cast(TensorId x, DType to, const std::string& name);
    TensorId reshape(TensorId x, Shape shape, const std::string& name);
    // Permute axes (outermost-first permutation: out.shape[i] = in.shape[perm[i]]).
    TensorId transpose(TensorId x, const std::vector<unsigned>& perm, const std::string& name);
    std::vector<TensorId> split(TensorId x, unsigned axis, const std::vector<uint64_t>& sizes, const std::string& name);
    TensorId concat(const std::vector<TensorId>& xs, unsigned axis, const std::string& name);
    TensorId reduce(const char* op, TensorId x, unsigned axis, const std::string& name);  // sum, max: keeps the axis as 1
    TensorId softmax(TensorId x, const std::string& name);  // over the innermost axis

    // y = x @ w^T for a BF16 x [M,K] and BF16 w [N,K].
    TensorId linear_bf16(TensorId x, TensorId w, const std::string& name, DType out = DType::BF16);
    // vLLM-equivalent dynamic FP8 linear: x [M,K] (BF16) quantized per row with scale max|x|/240, FP8 GEMM against
    // w [N,K] (FP8) with per-row weight scale ws [N] (F32); BF16 output [M,N].
    TensorId fp8_linear(TensorId x, TensorId w, TensorId ws, const std::string& name);
    // RMSNorm over the innermost axis: y = x / sqrt(mean(x^2) + eps) * (plus_one ? 1 + w : w)
    TensorId rms_norm(TensorId x, TensorId w, float eps, bool plus_one, const std::string& name);
    TensorId silu(TensorId x, const std::string& name);
    // LayerNorm over the innermost axis of x [M, N] with weight and bias [N]
    TensorId layer_norm(TensorId x, TensorId w, TensorId b, float eps, const std::string& name);
    // GELU, tanh approximation or exact (erf)
    TensorId gelu(TensorId x, bool tanh_approx, const std::string& name);

    // ---- selection ----
    // Index of the maximum along the innermost axis: [.., N] -> I32 [.., 1].
    TensorId argmax(TensorId x, const std::string& name);
    // k largest along the innermost axis: values [.., k] (dtype of x), indices I32 [.., k].
    std::pair<TensorId, TensorId> topk(TensorId x, uint32_t k, const std::string& name);
    // Top-k along the last axis of x [R, V] without sorting the row (a bitonic sort of a vocabulary takes ~1 ms):
    // the k largest values lie in the k chunks (of `chunk` columns) with the largest maxima, so only those k*chunk
    // values are sorted. `consts` [V + 2R] i32 = topk2_constants(R, V, k, chunk). Same set as topk; ties may order
    // differently.
    std::pair<TensorId, TensorId> topk2(TensorId x, uint32_t k, uint32_t chunk, TensorId consts, const std::string& name,
                                        TensorId precomputed_cmax = kNoTensor);
    static std::vector<int32_t> topk2_constants(uint64_t rows, uint64_t cols, uint32_t k, uint32_t chunk);
    // Rows of table [R, ...] selected by idx I32 [n] -> [n, ...].
    TensorId gather_rows(TensorId table, TensorId idx, const std::string& name);
    // Entries of `axis` (outermost-first) selected by idx I32 [n].
    TensorId gather(TensorId table, TensorId idx, unsigned axis, const std::string& name);

    // ---- fused custom TPC kernels (tpc/flashinfer_gaudi) ----
    struct NormQuant { TensorId codes, scale, normalized, residual_out; };
    // s = x + residual; n = bf16(s * rsqrt(mean(s^2) + eps) * w); codes/scale = vLLM row FP8 quantization of n.
    // `w` is the final BF16 weight (pass 1 + w for Gemma-style norms).
    NormQuant add_rmsnorm_quant(TensorId x, TensorId residual, TensorId w, float eps, const std::string& name,
                                bool need_normalized = true);
    // FP8 GEMM (no scale epilogue) + add_rmsnorm_quant with GEMM dequantisation folded in (3 outputs: codes, scale, residual_out).
    NormQuant fp8_gemm_add_rmsnorm_quant(TensorId codes, TensorId scale, TensorId w, TensorId ws,
                                         TensorId residual, TensorId norm_w, float eps, const std::string& name);
    // FP8 GEMM of pre-quantized rows: codes [M,K] (scale [M,1]) x w [N,K] (ws [N]) -> BF16 [M,N]
    TensorId fp8_gemm(TensorId codes, TensorId scale, TensorId w, TensorId ws, const std::string& name);
    // gate_up GEMM + silu(gate) * up -> FP8 codes and row scales, the GEMM's dequantisation folded into the kernel
    // (no scale epilogue after the GEMM): codes [M,K] f8 with row scales [M,1], w [2F,K] f8, ws [2F] f32
    std::pair<TensorId, TensorId> fp8_gemm_silu_mul_quant(TensorId codes, TensorId scale, TensorId w, TensorId ws,
                                                          const std::string& name);
    // Row FP8 quantization (vLLM rule) of BF16 x [M,K] -> codes, scale [M,1]
    std::pair<TensorId, TensorId> quantize(TensorId x, const std::string& name);
    // a * b (BF16 [M,F] each), quantized per row like silu_mul_quant (its kernel reads only the second half of its
    // [M, 2F] input and multiplies by the given activation): -> codes [M,F], scale [M,1]
    std::pair<TensorId, TensorId> mul_quant(TensorId a, TensorId b, const std::string& name);
    // silu(gate) * up for gu = [gate | up] [M, 2F], quantized: -> codes [M,F], scale [M,1]
    std::pair<TensorId, TensorId> silu_mul_quant(TensorId gu, const std::string& name);
    TensorId batch_gemm(TensorId a, TensorId b, const std::string& name, bool ta, bool tb, DType out);

private:
    std::string n(const std::string& name) const { return prefix_ + name; }
    GraphBuilder& g_;
    std::string prefix_;
};

const char* suffix(DType t);  // "bf16", "f32", ...

}  // namespace g2e
