// SPDX-License-Identifier: Apache-2.0
// Inkling-2 batched full-layer linear KV cache commit: up to 4 batch rows (K + V) in ONE 24-TPC launch.
// Copies b_kvs[:, h, 0, b0+r] -> k_r[:, h*cap_r + pos_r] and b_kvs[:, h, 1, b0+r] -> v_r[:, h*cap_r + pos_r]
// for r in 0..3 whenever pos_r >= 0 (skips dummy/unused rows with pos_r < 0).
// Tensors (dim 0 fastest, 9 total):
//   0: b_kvs       bf16 [128, KV, 2, 8]
//   1..8: k0, v0, k1, v1, k2, v2, k3, v3  bf16 [128, MAX_ROWS] (capacity-independent flat 2D views)
// Index space: [4 * 2 * KV] (= [64], 24 TPCs).
// Params (9 int32): b0, pos0, cap0, pos1, cap1, pos2, cap2, pos3, cap3.

void main(tensor b_kvs,
          tensor k0, tensor v0, tensor k1, tensor v1,
          tensor k2, tensor v2, tensor k3, tensor v3,
          int b0,
          int pos0, int cap0,
          int pos1, int cap1,
          int pos2, int cap2,
          int pos3, int cap3)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int KV = get_dim_size(b_kvs, 1);
    const int KV2 = KV << 1;

    for (int p = start[0]; p < end[0]; ++p) {
        const int r = p / KV2;
        const int rem = p - r * KV2;
        const int kvs = rem / KV;
        const int h = rem - kvs * KV;

        const int pos = (r == 0) ? pos0 : (r == 1 ? pos1 : (r == 2 ? pos2 : pos3));
        if (pos < 0) continue;
        const int cap = (r == 0) ? cap0 : (r == 1 ? cap1 : (r == 2 ? cap2 : cap3));

        int5 sc = {0, h, kvs, b0 + r, 0};
        const bfloat128 val = v_bf16_ld_tnsr_b(sc, b_kvs);
        int5 dc = {0, h * cap + pos, 0, 0, 0};

        if (r == 0) {
            if (kvs == 0) v_bf16_st_tnsr(dc, k0, val);
            else          v_bf16_st_tnsr(dc, v0, val);
        } else if (r == 1) {
            if (kvs == 0) v_bf16_st_tnsr(dc, k1, val);
            else          v_bf16_st_tnsr(dc, v1, val);
        } else if (r == 2) {
            if (kvs == 0) v_bf16_st_tnsr(dc, k2, val);
            else          v_bf16_st_tnsr(dc, v2, val);
        } else {
            if (kvs == 0) v_bf16_st_tnsr(dc, k3, val);
            else          v_bf16_st_tnsr(dc, v3, val);
        }
    }
}
