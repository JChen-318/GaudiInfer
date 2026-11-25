// SPDX-License-Identifier: Apache-2.0
// Inkling-2 decode / verification depth attention, phase 1 (ink_dmix_gaudi2.h): Gram / query partials of one tile and
// row per index point.
// Tensors (dim 0 fastest): blocks bf16 [H, T, S0], part bf16 [H, T], query f32 [H, Q]
// (split order), partials f32 [64, H / 128, T]; MSG: also the message mix's partials, states [prev1, prev2]: mblocks
// bf16 [H, T, 1] (prev1), mpart bf16 [H, T] (prev2), mquery f32 [H, 4] (msg_q), mpartials f32 [64, H / 128, T].
// Index space [H / 128, T].
#include "ink_dmix_gaudi2.h"

void main(tensor blocks, tensor part, tensor query, tensor partials
#ifdef MSG
          , tensor mblocks, tensor mpart, tensor mquery, tensor mpartials
#endif
          )
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int S0 = get_dim_size(blocks, 2);
    const int Q = get_dim_size(query, 1);
    for (int r = start[1]; r < end[1]; ++r) {
        for (int tile = start[0]; tile < end[0]; ++tile) {
            const int c = tile * 128;
            dmix_states_t xs;
            for (int s = 0; s < S0; ++s) {
                int5 bc = {c, r, s, 0, 0};
                xs.v[s] = dmix_bf16_tile(blocks, bc);
            }
            int5 pc = {c, r, 0, 0, 0};
            xs.v[S0] = dmix_bf16_tile(part, pc);
            dmix_gram_store(xs, S0 + 1, query, Q, c, partials, tile, r);
#ifdef MSG
            {
                dmix_states_t ms;
                int5 bc = {c, r, 0, 0, 0};
                ms.v[0] = dmix_bf16_tile(mblocks, bc);
                ms.v[1] = dmix_bf16_tile(mpart, bc);
                dmix_gram_store(ms, 2, mquery, 4, c, mpartials, tile, r);
            }
#endif
        }
    }
}
