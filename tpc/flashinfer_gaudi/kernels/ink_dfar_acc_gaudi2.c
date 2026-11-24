// SPDX-License-Identifier: Apache-2.0
// Inkling-2 Tiled MME FlashDecode F32 output accumulator (T = 1, Direct engine, 100% LUT-free).
// Stores o_acc[0..63, g, kv] = even 64 lanes and o_acc[64..127, g, kv] = odd 64 lanes (matching
// v_convert_bf16_to_f32_all_b lane order with zero shuffle, consumed by ink_dgate_f32_gaudi2).
//   o_acc = (init_from_pv ? f32(pv_near) : o_acc) * rescale[h] + f32(pvf)
// Tensors (dim 0 fastest): pv_near bf16 [128, G, KV], pvf bf16 [128, G, KV], rescale f32 [NH], o_acc f32 [128, G, KV].
// Index space [NH]. Params: int G, int init_from_pv.

void main(tensor pv_near, tensor pvf, tensor rescale, tensor o_acc, int G, int init_from_pv)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
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
        const float64_pair_t f = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(pc, pvf));
        o.v1 = v_f32_mac_b(o.v1, alpha, f.v1);
        o.v2 = v_f32_mac_b(o.v2, alpha, f.v2);
        v_f32_st_tnsr(pc, o_acc, o.v1);
        pc[0] = 64;
        v_f32_st_tnsr(pc, o_acc, o.v2);
    }
}
