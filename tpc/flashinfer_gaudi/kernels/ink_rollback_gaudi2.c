// SPDX-License-Identifier: Apache-2.0
// Inkling-2 speculative verification, accept(n): the conv states after row n - 1 of every application and conv from
// the verification's input histories (Inkling2::hist_: per application, per conv [3 + T rows, C]; the state after
// an accepted prefix of n rows is rows n .. n + 2) into the conv state buffer of the next forward.
// One (application, conv, 128-channel tile) per index point; conv c has C_c channels (KVD, KVD, H, H).
// Tensors: hist bf16 [elements of Inkling2::hist_], conv bf16 [elements of a conv state side].
// Params: hist_app, conv_app (elements per application), hoff[4], coff[4] (element offsets of the convs), ckv, ch
// (channels: k / v convs, attn / mlp convs), n, T.
void main(tensor hist, tensor conv, int hist_app, int conv_app, int hoff0, int hoff1, int hoff2, int hoff3, int coff0, int coff1,
          int coff2, int coff3, int ckv, int ch, int n, int T)
{
    const int5 start = get_index_space_offset();
    const int5 end = start + get_index_space_size();
    const int tkv = ckv / 128, th = ch / 128;
    const int per_app = 2 * tkv + 2 * th;
    (void)T;
    for (int p = start[0]; p < end[0]; ++p) {
        const int app = p / per_app;
        int q = p - app * per_app;
        int cv, tile;
        if (q < tkv) { cv = 0; tile = q; }
        else if (q < 2 * tkv) { cv = 1; tile = q - tkv; }
        else if (q < 2 * tkv + th) { cv = 2; tile = q - 2 * tkv; }
        else { cv = 3; tile = q - 2 * tkv - th; }
        const int C = cv < 2 ? ckv : ch;
        const int hoff = cv == 0 ? hoff0 : cv == 1 ? hoff1 : cv == 2 ? hoff2 : hoff3;
        const int coff = cv == 0 ? coff0 : cv == 1 ? coff1 : cv == 2 ? coff2 : coff3;
        for (int r = 0; r < 3; ++r) {
            int5 src = {app * hist_app + hoff + (n + r) * C + tile * 128, 0, 0, 0, 0};
            int5 dst = {app * conv_app + coff + r * C + tile * 128, 0, 0, 0, 0};
            v_bf16_st_tnsr(dst, conv, v_bf16_ld_tnsr_b(src, hist));
        }
    }
}
