// SPDX-License-Identifier: Apache-2.0
// Inkling-2 Direct-engine copy of BF16 rows of 128 (the V ring of each KV head in front of its new rows, for one
// P . V product over [ring | new]). Tensors (dim 0 fastest): src bf16 [128, R, B], dst bf16 [128, R, B] (any strides).
// Index space [B, R / 32]: 32 rows per point.
void main(tensor src, tensor dst)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    for (int b = start[0]; b < end[0]; ++b)
        for (int j = start[1]; j < end[1]; ++j) {
            int5 c = {0, j * 32, b, 0, 0};
            for (int i = 0; i < 32; i += 4) {
                const bfloat128 x0 = v_bf16_ld_tnsr_b(c, src);
                c[1] += 1;
                const bfloat128 x1 = v_bf16_ld_tnsr_b(c, src);
                c[1] += 1;
                const bfloat128 x2 = v_bf16_ld_tnsr_b(c, src);
                c[1] += 1;
                const bfloat128 x3 = v_bf16_ld_tnsr_b(c, src);
                c[1] -= 3;
                v_bf16_st_tnsr(c, dst, x0);
                c[1] += 1;
                v_bf16_st_tnsr(c, dst, x1);
                c[1] += 1;
                v_bf16_st_tnsr(c, dst, x2);
                c[1] += 1;
                v_bf16_st_tnsr(c, dst, x3);
                c[1] += 1;
            }
        }
}
