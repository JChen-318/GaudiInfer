// SPDX-License-Identifier: Apache-2.0
// Inkling-2 single-row full-layer linear KV cache commit (capacity-independent 2D view [128, 8*1048576]).
// Copies b_kvs[:, h, 0, b] -> kcache[:, h*cap + pos] and b_kvs[:, h, 1, b] -> vcache[:, h*cap + pos].
// Tensors (dim 0 fastest): b_kvs bf16 [128, KV, 2, Bp], kcache bf16 [128, MAX_ROWS], vcache bf16 [128, MAX_ROWS].
// Index space: [2 * KV]. Params: int b, int pos, int cap.

void main(tensor b_kvs, tensor kcache, tensor vcache, int b, int pos, int cap)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int KV = get_dim_size(b_kvs, 1);
    for (int p = start[0]; p < end[0]; ++p) {
        const int kvs = p / KV;
        const int h = p - kvs * KV;
        int5 sc = {0, h, kvs, b, 0};
        const bfloat128 v = v_bf16_ld_tnsr_b(sc, b_kvs);
        int5 dc = {0, h * cap + pos, 0, 0, 0};
        if (kvs == 0) v_bf16_st_tnsr(dc, kcache, v);
        else v_bf16_st_tnsr(dc, vcache, v);
    }
}
