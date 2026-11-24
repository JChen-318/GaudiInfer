// SPDX-License-Identifier: Apache-2.0
// Inkling-2 prefill MoE BF16 row gather (Direct engine, table-free): copies every valid token's BF16 row
// into each of its K routed expert slots at flat row `code[j, t]` (1 row per slot).
// Same 24-core token split as ink_rplan_dyn_slots_bf16 (no wait needed between them).
// Tensors (dim 0 fastest): x bf16 [H, T], code i32 [K, T], xb bf16 [H, R], vlen i32 [1].
// Index space [T]. H % 256 == 0, H <= 128 * MAX_T.
#define MAX_T 20
__local__ bfloat128 va[MAX_T];

void main(tensor x, tensor code, tensor xb, tensor vlen)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NT = get_dim_size(x, 0) >> 7;  // H / 128 = 18
    const int K = get_dim_size(code, 0);
    int5 vc = {0, 0, 0, 0, 0};
    const int TV = s_i32_ld_g((__global__ int*)gen_addr(vc, vlen));
    const int t_end = end[0] < TV ? end[0] : TV;

    for (int t = start[0]; t < t_end; ++t) {
        int5 xc = {0, t, 0, 0, 0};
        for (int i = 0; i < NT; i += 2) {
            xc[0] = i << 7;
            va[i] = v_bf16_ld_tnsr_b(xc, x);
            xc[0] = (i + 1) << 7;
            va[i + 1] = v_bf16_ld_tnsr_b(xc, x);
        }
        for (int j = 0; j < K; ++j) {
            int5 cc = {j, t, 0, 0, 0};
            const int row = s_i32_ld_g((__global__ int*)gen_addr(cc, code));
            int5 oc = {0, row, 0, 0, 0};
            for (int i = 0; i < NT; i += 2) {
                oc[0] = i << 7;
                v_bf16_st_tnsr(oc, xb, va[i]);
                oc[0] = (i + 1) << 7;
                v_bf16_st_tnsr(oc, xb, va[i + 1]);
            }
        }
    }
}
