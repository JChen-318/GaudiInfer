// SPDX-License-Identifier: Apache-2.0
// Inkling-2 near attention for full layers with tiled SRAM FlashAttention (Direct engine):
// identical to ink_attn_sm16b, also outputs the FP16-rounded near row max m_out f32 [1, GT] for online softmax merge.
#define SINK_BF16
#define SAVE_MAX
#include "ink_attn_sm16_gaudi2.c"
