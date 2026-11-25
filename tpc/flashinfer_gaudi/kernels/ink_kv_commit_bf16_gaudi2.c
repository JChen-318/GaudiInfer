// SPDX-License-Identifier: Apache-2.0
// Inkling-2 K/V commit: the new rows of one layer application (written to scratch by its layer launch) into a ring
// or a linear cache, at the row given per token (rows[t] < 0: skipped). A launch reads the scratch and writes the
// cache: separate persistent tensors (a layer launch itself must not write the cache it reads).
// Tensors (dim 0 fastest): kv bf16 [D, T, KV, 2] (k rows then v rows), rows i32 [T], kc bf16 [D, R, KV],
// vc bf16 [D, R, KV]. Index space [KV, T].
void main(tensor kv, tensor rows, tensor kc, tensor vc)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int t = start[1]; t < end[1]; ++t) {
        int5 rc = {t, 0, 0, 0, 0};
        const int r = s_i32_ld_g((__global__ int*)gen_addr(rc, rows));
        if (r < 0) continue;
        for (int h = start[0]; h < end[0]; ++h) {
            int5 src = {0, t, h, 0, 0};
            int5 dst = {0, r, h, 0, 0};
            v_bf16_st_tnsr(dst, kc, v_bf16_ld_tnsr_b(src, kv));
            src[3] = 1;
            v_bf16_st_tnsr(dst, vc, v_bf16_ld_tnsr_b(src, kv));
        }
    }
}
