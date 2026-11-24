// SPDX-License-Identifier: Apache-2.0
// Zero a contiguous BF16 buffer of length N (multiple of 128) in SRAM or HBM.
// Tensors (dim 0 fastest): dst bf16 [N]. Index space [N / 128].
void main(tensor dst)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const bfloat128 z = 0;
    for (int i = start[0]; i < end[0]; ++i) {
        int5 c = {i * 128, 0, 0, 0, 0};
        v_bf16_st_tnsr(c, dst, z);
    }
}
