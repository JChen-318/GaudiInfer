// SPDX-License-Identifier: Apache-2.0
// Inkling-2 verify / batched-decode MoE expert deduplication planner (1 TPC, vectorized output, LUT-free).
// Deduplicates valid routed expert IDs (e >= 0; ignores e < 0 padding rows) across T tokens (T <= 8, J <= 8, M = J*T <= 64)
// and writes ONE compact 1 KiB tensor grp i32 [64, 4] via 4 vector stores (zero scalar st_g!):
//   grp[u, 0] = unique expert ID (0..127) for u < num_groups, or -1 for unused slots u >= num_groups
//   grp[u, 1] = group member count cnt (1..4) for u < num_groups, or 0 for unused slots
//   grp[u, 2] = packed ts0: byte m (m=0..3) holds (tok[m] << 4) | slot[m]
//   grp[u, 3] = 0 (reserved; experts with > 4 picks at B=5..8 are split into <= 4-pick groups so 2 TPCs run them
//               in parallel with zero 2-pass stragglers, guaranteed <= 64 total groups!)
// Groups are emitted sorted by cnt descending (4, 3, 2, 1) so paired-snake 24-TPC mapping balances work evenly.
// Tensors (dim 0 fastest): idx i32 [J, T], grp i32 [64, 4].
// Index space [1]. Params: none.

__local__ int e2g[128];
__local__ int raw_e[64];
__local__ int raw_cnt[64];
__local__ int raw_ts0[64];
__local__ int row_e[8];

static inline int add_pick(int e, int t, int j, int nu) {
    if (e < 0 || e >= 128) return nu;
    int g = e2g[e];
    if (g < 0 || raw_cnt[g] >= 4) {
        if (nu >= 64) return nu;
        g = nu;
        nu = nu + 1;
        e2g[e] = g;
        raw_e[g] = e;
        raw_cnt[g] = 0;
        raw_ts0[g] = 0;
    }
    const int m = raw_cnt[g];
    const int code = (t << 4) | j;
    raw_ts0[g] = raw_ts0[g] | (code << (m << 3));
    raw_cnt[g] = m + 1;
    return nu;
}

void main(tensor idx, tensor grp)
{
    const int J = get_dim_size(idx, 0);
    const int T = get_dim_size(idx, 1);

    for (int e = 0; e < 128; e += 4) {
        e2g[e]     = -1;
        e2g[e + 1] = -1;
        e2g[e + 2] = -1;
        e2g[e + 3] = -1;
    }

    int nu = 0;
    if (J == 8) {
        // Issue all 8 scalar loads per token row in parallel to hide SRAM/HBM load latency,
        // then deduplicate in a compact single-copy loop (keeps .text < 5 KiB for I-cache!).
#pragma loop_unroll(1)
        for (int t = 0; t < T; ++t) {
            int5 c0 = {0, t, 0, 0, 0}, c1 = {1, t, 0, 0, 0}, c2 = {2, t, 0, 0, 0}, c3 = {3, t, 0, 0, 0};
            int5 c4 = {4, t, 0, 0, 0}, c5 = {5, t, 0, 0, 0}, c6 = {6, t, 0, 0, 0}, c7 = {7, t, 0, 0, 0};
            const int e0 = s_i32_ld_g((__global__ int*)gen_addr(c0, idx));
            const int e1 = s_i32_ld_g((__global__ int*)gen_addr(c1, idx));
            const int e2 = s_i32_ld_g((__global__ int*)gen_addr(c2, idx));
            const int e3 = s_i32_ld_g((__global__ int*)gen_addr(c3, idx));
            const int e4 = s_i32_ld_g((__global__ int*)gen_addr(c4, idx));
            const int e5 = s_i32_ld_g((__global__ int*)gen_addr(c5, idx));
            const int e6 = s_i32_ld_g((__global__ int*)gen_addr(c6, idx));
            const int e7 = s_i32_ld_g((__global__ int*)gen_addr(c7, idx));
            row_e[0] = e0; row_e[1] = e1; row_e[2] = e2; row_e[3] = e3;
            row_e[4] = e4; row_e[5] = e5; row_e[6] = e6; row_e[7] = e7;
#pragma loop_unroll(1)
            for (int j = 0; j < 8; ++j) {
                nu = add_pick(row_e[j], t, j, nu);
            }
        }
    } else {
        for (int t = 0; t < T; ++t) {
            int5 ic = {0, t, 0, 0, 0};
            for (int j = 0; j < J; ++j) {
                ic[0] = j;
                const int e = s_i32_ld_g((__global__ int*)gen_addr(ic, idx));
                nu = add_pick(e, t, j, nu);
            }
        }
    }

    const int64 lanes = (int64)read_lane_id_4b_b();
    int64 ve = (int64)-1;
    int64 vc = (int64)0;
    int64 vts0 = (int64)0;
    int64 vts1 = (int64)0;

    const int cmax = T < 4 ? T : 4;
    int out_u = 0;
    for (int c = cmax; c >= 1; --c) {
        for (int g = 0; g < nu; ++g) {
            if (raw_cnt[g] != c) continue;
            ve   = v_i32_sel_eq_i32_b(lanes, out_u, raw_e[g],   ve);
            vc   = v_i32_sel_eq_i32_b(lanes, out_u, c,          vc);
            vts0 = v_i32_sel_eq_i32_b(lanes, out_u, raw_ts0[g], vts0);
            out_u = out_u + 1;
        }
    }

    // 4 vector stores = entire 1 KiB plan buffer (ZERO scalar st_g)!
    int5 oc = {0, 0, 0, 0, 0};
    v_i32_st_tnsr(oc, grp, ve);
    oc[1] = 1; v_i32_st_tnsr(oc, grp, vc);
    oc[1] = 2; v_i32_st_tnsr(oc, grp, vts0);
    oc[1] = 3; v_i32_st_tnsr(oc, grp, vts1);
}
