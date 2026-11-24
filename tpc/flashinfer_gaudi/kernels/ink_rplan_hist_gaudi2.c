// SPDX-License-Identifier: Apache-2.0
// Inkling-2 MoE routing plan, stage 1 of 3 (Direct-engine prefill; replaces the cumsum / topk / scatter graph ops of
// moe_cap2): per-core expert histograms of the routed slots. Core p takes the tokens [tpart[p], tpart[p + 1]) (the
// 24-way split of T the TPC stages use), their slots in order (s = t K + k), and counts them per expert.
// Tensors (dim 0 fastest): idx i32 [S] (the router's top-K experts, flat), tpart i32 [25], hist i32 [128, 24].
// Index space [24]. Params: lk (log2 K).
__local__ int cnt[128];

void main(tensor idx, tensor tpart, tensor hist, int lk)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int p = start[0]; p < end[0]; ++p) {
        for (int e = 0; e < 128; ++e) cnt[e] = 0;
        int5 pc = {p, 0, 0, 0, 0};
        const int s0 = s_i32_ld_g((__global__ int*)gen_addr(pc, tpart)) << lk;
        pc[0] = p + 1;
        const int s1 = s_i32_ld_g((__global__ int*)gen_addr(pc, tpart)) << lk;
        int5 c = {s0, 0, 0, 0, 0};
        for (int s = s0; s < s1; ++s) {
            c[0] = s;
            const int e = s_i32_ld_g((__global__ int*)gen_addr(c, idx));
            cnt[e] = cnt[e] + 1;
        }
        int5 o = {0, p, 0, 0, 0};
        for (int e = 0; e < 128; ++e) {
            o[0] = e;
            s_i32_st_g((__global__ int*)gen_addr(o, hist), cnt[e]);
        }
    }
}
