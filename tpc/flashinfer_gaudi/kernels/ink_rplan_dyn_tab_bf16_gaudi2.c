// SPDX-License-Identifier: Apache-2.0
// Inkling-2 MoE dynamic grouped-GEMM routing plan, stage 2 of 3 (one core, BF16 1 row per slot):
//   off[p][e]    = exclusive prefix sum of hist across cores before p
//   cnt[e]       = total slots routed to expert e (downloaded by host via PDMA)
//   exp_base[e]  = exclusive prefix sum of alloc_rows(cnt[e]) across experts 0..127,
//                  matching the host's exact tile decomposition over {256, 192, 128, 96, 64, 32, 16}
//                  (zero overflow for all routing distributions).
// Tensors (dim 0 fastest): hist i32 [128, 24], off i32 [128, 24], cnt i32 [128], exp_base i32 [128].
// Index space [1].

static inline float64 alloc_rows_v(int64 c) {
    const float64 qf = v_convert_i32_to_f32_b(c >> 8, 0);
    const float64 rf = v_convert_i32_to_f32_b(c & 255, 0);
    float64 rr = (float64)256.0f;
    rr = v_f32_sel_leq_f32_b(rf, 224.0f, 224.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf, 192.0f, 192.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf, 160.0f, 160.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf, 144.0f, 144.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf, 128.0f, 128.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf,  96.0f,  96.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf,  64.0f,  64.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf,  32.0f,  32.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf,  16.0f,  16.0f, rr);
    rr = v_f32_sel_leq_f32_b(rf,   0.0f,   0.0f, rr);
    return qf * 256.0f + rr;
}

void main(tensor hist, tensor off, tensor cnt, tensor exp_base)
{
    int64 ra = 0, rb = 0;  // running sums: experts 0..63, 64..127
    for (int j = 0; j < 24; ++j) {
        int5 c = {0, j, 0, 0, 0};
        const int64 ha = v_i32_ld_tnsr_b(c, hist);
        c[0] = 64;
        const int64 hb = v_i32_ld_tnsr_b(c, hist);
        c[0] = 0;
        v_i32_st_tnsr(c, off, ra);
        c[0] = 64;
        v_i32_st_tnsr(c, off, rb);
        ra += ha;
        rb += hb;
    }
    int5 z = {0, 0, 0, 0, 0};
    v_i32_st_tnsr(z, cnt, ra);
    z[0] = 64;
    v_i32_st_tnsr(z, cnt, rb);

    const float64 rows_a = alloc_rows_v(ra);
    const float64 rows_b = alloc_rows_v(rb);
    const float64 lane = v_convert_i32_to_f32_b(as_int64(read_lane_id_4b_b()), 0);
    float64 psa = 0, psb = 0;
#define PS(K) do { \
        const float64 ba_ = v_broadcast_element_f32(rows_a, (K)); \
        const float64 bb_ = v_broadcast_element_f32(rows_b, (K)); \
        psa += v_f32_sel_grt_f32_b(lane, (float)(K), ba_, 0.0f); \
        psb += v_f32_sel_grt_f32_b(lane, (float)(K), bb_, 0.0f); \
    } while (0)
#define PS8(I) do { PS(I); PS(I + 1); PS(I + 2); PS(I + 3); PS(I + 4); PS(I + 5); PS(I + 6); PS(I + 7); } while (0)
    PS8(0); PS8(8); PS8(16); PS8(24); PS8(32); PS8(40); PS8(48); PS8(56);
#undef PS8
#undef PS
    const float64 tot_a = v_broadcast_element_f32(psa + rows_a, 63);
    psb += tot_a;
    z[0] = 0;
    v_i32_st_tnsr(z, exp_base, v_convert_f32_to_i32_b(psa, SW_RHNE));
    z[0] = 64;
    v_i32_st_tnsr(z, exp_base, v_convert_f32_to_i32_b(psb, SW_RHNE));
}
