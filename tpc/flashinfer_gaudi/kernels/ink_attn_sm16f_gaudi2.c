// SPDX-License-Identifier: Apache-2.0
// ink_attn_sm16 with far keys (full layers at long contexts): far scores sf f16 [Sf, GT] (no bias, cache rows
// j < pos0 - W valid), far probabilities pf bf16 [Sf, GT] (zeros past the valid rows), the near part as ink_attn_sm16.
// Tensors: sf, sr, sn, br, sink, pos -> pf, p, sums. W = 1024, T % 128 == 0, T <= 1024 (Sf: any bucket).
#define ROW_FAR
#include "ink_attn_sm16_gaudi2.c"
