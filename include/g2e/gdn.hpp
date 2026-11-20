// Chunked gated delta rule (after strata/hpu/src/gdn_chunked.cpp) as a graph fragment, one chunk of C rows.
//
// Per value head h (value head h uses key head h / (HV/HK)), with S the [value][key] state entering the chunk, G the
// within-chunk cumulative log decay and Gamma = exp(G):
//   A[t][u] = beta[t] exp(G[t]-G[u]) (k[t].k[u]) for u < t;  X = (I + A)^-1 (8x8 Newton blocks, then doublings)
//   delta = X (beta v) - X (beta Gamma k) S^T;  out = (Gamma q) S^T + (exp(G[t]-G[u]) (q[t].k[u]))_{u<=t} delta
//   S' = Gamma[C-1] S + delta^T (exp(G[C-1]-G) k)
// q/k must already be L2-normalized (q also scaled); g is the log decay; everything F32.
#pragma once

#include "g2e/ops.hpp"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace g2e {

struct GdnChunkDims {
    uint64_t HK, HV, DK, DV;
};

// Host values of the chunk constants for C in {8, 16, 32, 64}: (name, shape, values). Names: upper [C,C],
// strict, inclusive, lower, eye, diag8, ll8.. [1,C,C].
struct GdnConstant {
    std::string name;
    Shape shape;
    std::vector<float> values;
};
std::vector<GdnConstant> gdn_chunk_constants(uint32_t C);

// `constant(name, shape)` returns the graph tensor holding chunk constant `name` (the caller decides whether it is a
// persistent input and shares it). Returns (core [C, HV, DV], S' [HV, DV, DK]).
std::pair<TensorId, TensorId> gdn_chunk(Ops& o, const std::string& prefix, uint32_t C, const GdnChunkDims& d,
                                        const std::function<TensorId(const std::string&, Shape)>& constant,
                                        TensorId state, TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta);

// nc consecutive chunks of C rows at once (q/k [nc*C, HK, DK], v [nc*C, HV, DV], g/beta [nc*C, HV]): every part that
// does not depend on the state (decays, (I + A)^-1, u, w, ...) runs batched over the chunks; only delta and the
// state update are chained chunk by chunk, and the outputs are batched again over the saved entry states.
// Returns (core [nc*C, HV, DV], S' [HV, DV, DK]).
std::pair<TensorId, TensorId> gdn_chunks(Ops& o, const std::string& prefix, uint32_t C, uint32_t nc, const GdnChunkDims& d,
                                         const std::function<TensorId(const std::string&, Shape)>& constant,
                                         TensorId state, TensorId q, TensorId k, TensorId v, TensorId g, TensorId beta);

}  // namespace g2e
