// SPDX-License-Identifier: Apache-2.0
// Inkling-2 MoE dynamic grouped-GEMM routing plan, stage 3 of 3 (Direct-engine prefill, BF16 1 row per slot):
// Every slot's flat row in the dynamic row space (zero overflow). Core p walks the slots of its tokens
// [tpart[p], tpart[p + 1]) in order; rank r = off[p][e] + seen so far, row = exp_base[e] + r.
// Partitions slots into 4 waves by expert index (w = e >> 5: 0..31, 32..63, 64..95, 96..127) in lbuf[S, 4] / lcnt[24, 4].
// Tensors (dim 0 fastest): idx i32 [S], off i32 [128, 24], exp_base i32 [128], tpart i32 [25],
// code i32 [S], lbuf i32 [S, 4], lcnt i32 [24, 4]. Index space [24]. Params: lk (log2 K).
__local__ int run[128];
__local__ int eb[128];
__local__ int nw[4];

void main(tensor idx, tensor off, tensor exp_base, tensor tpart, tensor code, tensor lbuf, tensor lcnt, int lk)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int p = start[0]; p < end[0]; ++p) {
        int5 oc = {0, p, 0, 0, 0};
        int5 ec = {0, 0, 0, 0, 0};
        for (int e = 0; e < 128; ++e) {
            oc[0] = e;
            ec[0] = e;
            run[e] = s_i32_ld_g((__global__ int*)gen_addr(oc, off));
            eb[e] = s_i32_ld_g((__global__ int*)gen_addr(ec, exp_base));
        }
        nw[0] = 0; nw[1] = 0; nw[2] = 0; nw[3] = 0;
        int5 pc = {p, 0, 0, 0, 0};
        const int t0 = s_i32_ld_g((__global__ int*)gen_addr(pc, tpart));
        pc[0] = p + 1;
        const int s0 = t0 << lk, s1 = s_i32_ld_g((__global__ int*)gen_addr(pc, tpart)) << lk;
        int5 sc = {0, 0, 0, 0, 0};
        int5 lc = {0, 0, 0, 0, 0};
        for (int s = s0; s < s1; ++s) {
            sc[0] = s;
            const int e = s_i32_ld_g((__global__ int*)gen_addr(sc, idx));
            const int r = run[e];
            run[e] = r + 1;
            const int cv = eb[e] + r;
            s_i32_st_g((__global__ int*)gen_addr(sc, code), cv);
            const int w = e >> 5;
            const int k = nw[w];
            nw[w] = k + 1;
            lc[0] = s0 + k;
            lc[1] = w;
            s_i32_st_g((__global__ int*)gen_addr(lc, lbuf), s);
        }
        for (int w = 0; w < 4; ++w) {
            int5 nc = {p, w, 0, 0, 0};
            s_i32_st_g((__global__ int*)gen_addr(nc, lcnt), nw[w]);
        }
    }
}
