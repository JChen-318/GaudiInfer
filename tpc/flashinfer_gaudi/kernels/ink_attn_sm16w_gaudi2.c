// SPDX-License-Identifier: Apache-2.0
// ink_attn_sm16b for the sliding layers' chunks of T = 1024 rows at a position that is a multiple of W = 1024 (every
// chunk of a chunked prefill): only each row's window. Ring slot s then holds the key at distance W - s before the
// chunk, so on the key axis x = [ring slots | new rows] row t sees exactly the keys t < x <= t + W: nine of the
// sixteen 128-key chunks, seven of them without a mask. The other chunks are neither read nor written: p must be a
// buffer that only this kernel writes, zero outside the windows (the P . V product reads all of it).
// The first chunk of a prompt (position 0, no ring keys) takes the new rows of the window and zeroes its ring part.
// Same values as ink_attn_sm16b for every visible key and for the sums (hidden keys: 0 instead of 2^-126).
// Tensors as ink_attn_sm16b: sr f16 [W, GT], sn f16 [T, GT], br f16 [RE, NH, T], sink bf16 [NH, T] (the q-side rows'
// sink columns), pos i32 [1], p bf16 [W + T, GT], sums f32 [1, GT]. Index space [GT]. Params: scale, G, full (0), kv.
#define SINK_BF16
#define ROW_WINDOW
#include "ink_attn_sm16_gaudi2.c"

#define NORING 0
#define B 0
#define ROWFN roww_0
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 1
#define ROWFN roww_1
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 2
#define ROWFN roww_2
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 3
#define ROWFN roww_3
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 4
#define ROWFN roww_4
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 5
#define ROWFN roww_5
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 6
#define ROWFN roww_6
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 7
#define ROWFN roww_7
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#undef NORING
#define NORING 1
#define B 0
#define ROWFN row0w_0
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 1
#define ROWFN row0w_1
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 2
#define ROWFN row0w_2
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 3
#define ROWFN row0w_3
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 4
#define ROWFN row0w_4
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 5
#define ROWFN row0w_5
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 6
#define ROWFN row0w_6
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#define B 7
#define ROWFN row0w_7
#include "ink_attn_roww_gaudi2.h"
#undef B
#undef ROWFN
#undef NORING

void main(tensor sr, tensor sn, tensor br, tensor sink, tensor pos_t, tensor p, tensor sums, float scale, int G,
          int full, int kv)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int RE = get_dim_size(br, 0);
    (void)full;
    int5 z = {0, 0, 0, 0, 0};
    const int pos0 = s_i32_ld_g((__global__ int*)gen_addr(z, pos_t));  // the chunk's first position (a multiple of W)
    const short128 lane = (short128)read_lane_id_2b_b();
    const half128 neg = (half128)-65504.0f;
    const short128 negb = *(short128*)&neg;
    float128 kf;
    kf.v1 = 128.0f * 1.4426950408889634f * scale;
    kf.v2 = kf.v1;
    const half128 kh = v_convert_f32_to_f16_all_b(kf);
    const float isc = ink_srecip(scale);
    // runs of rows of one band (t = r mod 1024, band = t / 128, query head g = r / 1024)
#define RUN_(FN) for (; r < re; ++r) FN(sr, sn, br, sink, p, sums, r, r & 1023, kv * G + (r >> 10), RE, kh, isc, scale, lane, negb)
    int r = start[0];
    while (r < end[0]) {
        const int b = (r & 1023) >> 7;
        int re = (r | 127) + 1;
        if (re > end[0]) re = end[0];
        if (pos0 > 0) {
            if (b == 0) RUN_(roww_0);
            else if (b == 1) RUN_(roww_1);
            else if (b == 2) RUN_(roww_2);
            else if (b == 3) RUN_(roww_3);
            else if (b == 4) RUN_(roww_4);
            else if (b == 5) RUN_(roww_5);
            else if (b == 6) RUN_(roww_6);
            else RUN_(roww_7);
        } else {
            if (b == 0) RUN_(row0w_0);
            else if (b == 1) RUN_(row0w_1);
            else if (b == 2) RUN_(row0w_2);
            else if (b == 3) RUN_(row0w_3);
            else if (b == 4) RUN_(row0w_4);
            else if (b == 5) RUN_(row0w_5);
            else if (b == 6) RUN_(row0w_6);
            else RUN_(row0w_7);
        }
    }
#undef RUN_
}
