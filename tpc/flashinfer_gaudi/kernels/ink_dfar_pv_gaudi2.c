// SPDX-License-Identifier: Apache-2.0
// Inkling-2 FlashDecode split-K 24-TPC far-key P . V kernel (T = 1, Direct engine, 100% LUT-free).
// Replaces MME d_fpv for both tiled long-context (135K..1M, nv > 0) and non-tiled far buckets (2K..131K, nv <= 0).
// Splits [0, nval) across NS = dim3(pvf_part) TPCs per KV head (NS = 3 -> 8 KV x 3 = 24 TPCs):
//   pvf_part[:, g, kv, sp] = sum_{t in [t0, t1)} pf[t, g, kv] * v[:, kv * cap + col0 + t]
// Output uses split FP32 lane order [even 64 | odd 64] matching v_bf16_mac_acc32_b (zero shuffle,
// reduced by ink_dfar_acc3_gaudi2 / ink_dgate_far3_gaudi2).
// Tensors (dim 0 fastest):
//   0: pf       bf16 [TW, G, KV]       (SRAM)
//   1: v        bf16 [128, 16777216]   (flat 2D row view of HBM vcache_, row = kv * cap + col0 + t)
//   2: pvf_part f32  [128, G, KV, NS]  (SRAM, split F32 order)
//   3: pos_t    i32  [1]               (read only when nv <= 0: nval = clamp(pos - 1023, 0, TW))
// Index space [KV * NS] (= [24]). Params: int G, int cap, int col0, int nv.

void main(tensor pf, tensor v, tensor pvf_part, tensor pos_t, int G, int cap, int col0, int nv)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int NS = get_dim_size(pvf_part, 3);
    const int TW = get_dim_size(pf, 0);
    int nval = nv;
    if (nval <= 0) {
        int5 z = {0, 0, 0, 0, 0};
        const int pos = s_i32_ld_g((__global__ int*)gen_addr(z, pos_t));
        nval = pos > 1023 ? pos - 1023 : 0;
    }
    if (nval > TW) nval = TW;

    for (int pt = start[0]; pt < end[0]; ++pt) {
        const int kv = pt / NS;
        const int sp = pt - kv * NS;
        const int t0 = sp == 0 ? 0 : (((nval * sp) / NS) & ~3);
        const int t1 = sp + 1 == NS ? nval : (((nval * (sp + 1)) / NS) & ~3);
        const int vbase = kv * cap + col0;

        if (G == 3) {
            float128 o0a = {0}, o1a = {0}, o2a = {0};
            float128 o0b = {0}, o1b = {0}, o2b = {0};
            int t = t0;
            // 12-deep continuous SWP on HBM v (3 KiB/TPC = 72 KiB across 24 TPCs, SW_L0CS keeps pf in L0),
            // v_bf16_ld_g broadcast weights (zero scalar latency), 6 dual FP32 accumulators (zero MAC stalls).
            if (t0 + 12 <= t1) {
                const int n12 = (t1 - t0 - 12) / 12;
                int5 vc = {0, vbase + t0, 0, 0, 0};
                int5 p0c = {t0, 0, kv, 0, 0};
                int5 p1c = {t0, 1, kv, 0, 0};
                int5 p2c = {t0, 2, kv, 0, 0};

                bfloat128 v0  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v1  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v2  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v3  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v4  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v5  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v6  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v7  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v8  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v9  = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v10 = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                bfloat128 v11 = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;

                bfloat128 a0 = v_bf16_ld_g((__global__ bf16*)gen_addr(p0c, pf)); p0c[0] += 1;
                bfloat128 a1 = v_bf16_ld_g((__global__ bf16*)gen_addr(p1c, pf)); p1c[0] += 1;
                bfloat128 a2 = v_bf16_ld_g((__global__ bf16*)gen_addr(p2c, pf)); p2c[0] += 1;
                bfloat128 b0 = v_bf16_ld_g((__global__ bf16*)gen_addr(p0c, pf)); p0c[0] += 1;
                bfloat128 b1 = v_bf16_ld_g((__global__ bf16*)gen_addr(p1c, pf)); p1c[0] += 1;
                bfloat128 b2 = v_bf16_ld_g((__global__ bf16*)gen_addr(p2c, pf)); p2c[0] += 1;

#define VSTEP(VV, OA, OB, OC, P0, P1, P2, DO_V, DO_P) do {                                   \
                    (OA) = v_bf16_mac_acc32_b((VV), (P0), (OA), 0);                          \
                    if (DO_P) { (P0) = v_bf16_ld_g((__global__ bf16*)gen_addr(p0c, pf)); p0c[0] += 1; } \
                    (OB) = v_bf16_mac_acc32_b((VV), (P1), (OB), 0);                          \
                    if (DO_P) { (P1) = v_bf16_ld_g((__global__ bf16*)gen_addr(p1c, pf)); p1c[0] += 1; } \
                    (OC) = v_bf16_mac_acc32_b((VV), (P2), (OC), 0);                          \
                    if (DO_P) { (P2) = v_bf16_ld_g((__global__ bf16*)gen_addr(p2c, pf)); p2c[0] += 1; } \
                    if (DO_V) { (VV) = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1; }       \
                } while (0)

#pragma loop_unroll(1)
                for (int k = 0; k < n12; ++k) {
                    VSTEP(v0,  o0a, o1a, o2a, a0, a1, a2, 1, 1);
                    VSTEP(v1,  o0b, o1b, o2b, b0, b1, b2, 1, 1);
                    VSTEP(v2,  o0a, o1a, o2a, a0, a1, a2, 1, 1);
                    VSTEP(v3,  o0b, o1b, o2b, b0, b1, b2, 1, 1);
                    VSTEP(v4,  o0a, o1a, o2a, a0, a1, a2, 1, 1);
                    VSTEP(v5,  o0b, o1b, o2b, b0, b1, b2, 1, 1);
                    VSTEP(v6,  o0a, o1a, o2a, a0, a1, a2, 1, 1);
                    VSTEP(v7,  o0b, o1b, o2b, b0, b1, b2, 1, 1);
                    VSTEP(v8,  o0a, o1a, o2a, a0, a1, a2, 1, 1);
                    VSTEP(v9,  o0b, o1b, o2b, b0, b1, b2, 1, 1);
                    VSTEP(v10, o0a, o1a, o2a, a0, a1, a2, 1, 1);
                    VSTEP(v11, o0b, o1b, o2b, b0, b1, b2, 1, 1);
                }
                VSTEP(v0,  o0a, o1a, o2a, a0, a1, a2, 0, 1);
                VSTEP(v1,  o0b, o1b, o2b, b0, b1, b2, 0, 1);
                VSTEP(v2,  o0a, o1a, o2a, a0, a1, a2, 0, 1);
                VSTEP(v3,  o0b, o1b, o2b, b0, b1, b2, 0, 1);
                VSTEP(v4,  o0a, o1a, o2a, a0, a1, a2, 0, 1);
                VSTEP(v5,  o0b, o1b, o2b, b0, b1, b2, 0, 1);
                VSTEP(v6,  o0a, o1a, o2a, a0, a1, a2, 0, 1);
                VSTEP(v7,  o0b, o1b, o2b, b0, b1, b2, 0, 1);
                VSTEP(v8,  o0a, o1a, o2a, a0, a1, a2, 0, 1);
                VSTEP(v9,  o0b, o1b, o2b, b0, b1, b2, 0, 1);
                VSTEP(v10, o0a, o1a, o2a, a0, a1, a2, 0, 0);
                VSTEP(v11, o0b, o1b, o2b, b0, b1, b2, 0, 0);
                t = t0 + (n12 + 1) * 12;
#undef VSTEP
            }
            if (t < t1) {
                int5 vc = {0, vbase + t, 0, 0, 0};
                int5 p0c = {t, 0, kv, 0, 0};
                int5 p1c = {t, 1, kv, 0, 0};
                int5 p2c = {t, 2, kv, 0, 0};
                for (; t < t1; ++t) {
                    const bfloat128 vv = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                    const bf16 p0 = s_bf16_ld_g((__global__ bf16*)gen_addr(p0c, pf)); p0c[0] += 1;
                    const bf16 p1 = s_bf16_ld_g((__global__ bf16*)gen_addr(p1c, pf)); p1c[0] += 1;
                    const bf16 p2 = s_bf16_ld_g((__global__ bf16*)gen_addr(p2c, pf)); p2c[0] += 1;
                    o0a = v_bf16_mac_acc32_b(vv, p0, o0a, 0);
                    o1a = v_bf16_mac_acc32_b(vv, p1, o1a, 0);
                    o2a = v_bf16_mac_acc32_b(vv, p2, o2a, 0);
                }
            }
            int5 oc = {0, 0, kv, sp, 0};
            v_f32_st_tnsr(oc, pvf_part, o0a.v1 + o0b.v1); oc[0] = 64;
            v_f32_st_tnsr(oc, pvf_part, o0a.v2 + o0b.v2);
            oc[0] = 0; oc[1] = 1;
            v_f32_st_tnsr(oc, pvf_part, o1a.v1 + o1b.v1); oc[0] = 64;
            v_f32_st_tnsr(oc, pvf_part, o1a.v2 + o1b.v2);
            oc[0] = 0; oc[1] = 2;
            v_f32_st_tnsr(oc, pvf_part, o2a.v1 + o2b.v1); oc[0] = 64;
            v_f32_st_tnsr(oc, pvf_part, o2a.v2 + o2b.v2);
        } else {
            for (int g = 0; g < G; ++g) {
                float128 oa = {0};
                int5 vc = {0, vbase + t0, 0, 0, 0};
                int5 pc = {t0, g, kv, 0, 0};
                for (int t = t0; t < t1; ++t) {
                    const bfloat128 vv = v_bf16_ld_tnsr_b(vc, v, SW_L0CS); vc[1] += 1;
                    const bf16 pw = s_bf16_ld_g((__global__ bf16*)gen_addr(pc, pf)); pc[0] += 1;
                    oa = v_bf16_mac_acc32_b(vv, pw, oa, 0);
                }
                int5 oc = {0, g, kv, sp, 0};
                v_f32_st_tnsr(oc, pvf_part, oa.v1); oc[0] = 64;
                v_f32_st_tnsr(oc, pvf_part, oa.v2);
            }
        }
    }
}
