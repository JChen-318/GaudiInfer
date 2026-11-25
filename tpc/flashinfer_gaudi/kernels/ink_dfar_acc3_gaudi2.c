// SPDX-License-Identifier: Apache-2.0
// Inkling-2 Tiled FlashDecode F32 output accumulator for split-K TPC PV (T = 1, Direct engine, 100% LUT-free).
// Reduces pvf_part[0..127, g, kv, 0..NS-1] (split F32 lane order: [even 64 | odd 64], matching
// v_bf16_mac_acc32_b / v_convert_bf16_to_f32_all_b with zero shuffle, consumed by ink_dgate_f32_gaudi2):
//   f = sum_{sp=0..NS-1} pvf_part[:, g, kv, sp]
//   o_acc = (init_from_pv ? f32(pv_near) : o_acc) * rescale[h] + f
// Tensors (dim 0 fastest): pv_near bf16 [128, G, KV], pvf_part f32 [128, G, KV, NS],
// rescale f32 [NH], o_acc f32 [128, G, KV].
// Index space [NH]. Params: int G, int init_from_pv.

void main(tensor pv_near, tensor pvf_part, tensor rescale, tensor o_acc, int G, int init_from_pv)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NS = get_dim_size(pvf_part, 3);
    for (int h = start[0]; h < end[0]; ++h) {
        const int kv = h / G, g = h - kv * G;
        int5 rc = {h, 0, 0, 0, 0};
        const float64 alpha = v_broadcast_element_f32(v_f32_ld_tnsr_b(rc, rescale), 0);
        int5 pc = {0, g, kv, 0, 0};
        float64_pair_t o;
        if (init_from_pv) {
            o = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pv_near));
        } else {
            o.v1 = v_f32_ld_tnsr_b(pc, o_acc);
            pc[0] = 64;
            o.v2 = v_f32_ld_tnsr_b(pc, o_acc);
            pc[0] = 0;
        }
        float64 f1, f2;
        if (NS == 3) {
            int5 fc0 = {0, g, kv, 0, 0};
            int5 fc1 = {0, g, kv, 1, 0};
            int5 fc2 = {0, g, kv, 2, 0};
            const float64 a0 = v_f32_ld_tnsr_b(fc0, pvf_part); fc0[0] = 64;
            const float64 b0 = v_f32_ld_tnsr_b(fc0, pvf_part);
            const float64 a1 = v_f32_ld_tnsr_b(fc1, pvf_part); fc1[0] = 64;
            const float64 b1 = v_f32_ld_tnsr_b(fc1, pvf_part);
            const float64 a2 = v_f32_ld_tnsr_b(fc2, pvf_part); fc2[0] = 64;
            const float64 b2 = v_f32_ld_tnsr_b(fc2, pvf_part);
            f1 = (a0 + a1) + a2;
            f2 = (b0 + b1) + b2;
        } else {
            int5 fc = {0, g, kv, 0, 0};
            f1 = v_f32_ld_tnsr_b(fc, pvf_part);
            fc[0] = 64;
            f2 = v_f32_ld_tnsr_b(fc, pvf_part);
            for (int sp = 1; sp < NS; ++sp) {
                fc[0] = 0; fc[3] = sp;
                f1 = f1 + v_f32_ld_tnsr_b(fc, pvf_part);
                fc[0] = 64;
                f2 = f2 + v_f32_ld_tnsr_b(fc, pvf_part);
            }
        }
        // Round split-K sum to BF16 to match MME d_fpv output rounding bit-for-bit before F32 accumulation.
        float64_pair_t fp;
        fp.v1 = f1;
        fp.v2 = f2;
        fp = v_convert_bf16_to_f32_all_b(v_convert_f32_to_bf16_all_b(fp));
        o.v1 = v_f32_mac_b(o.v1, alpha, fp.v1);
        o.v2 = v_f32_mac_b(o.v2, alpha, fp.v2);
        v_f32_st_tnsr(pc, o_acc, o.v1);
        pc[0] = 64;
        v_f32_st_tnsr(pc, o_acc, o.v2);
    }
}
