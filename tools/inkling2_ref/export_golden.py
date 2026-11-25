"""golden/<case>/*.npy -> raw little-endian files for the C++ test (ids.i32, logits_last.f32, dec_ids.i32, dec_logits.f32, hs.f32)"""
import sys, numpy as np, os
root = sys.argv[1]
for case in ('short', 'long'):
    d = os.path.join(root, case)
    ids = np.load(f'{d}/ids.npy').astype(np.int32); ids.tofile(f'{d}/ids.i32')
    lg = np.load(f'{d}/logits.npy').astype(np.float32); lg[-1].tofile(f'{d}/logits_last.f32')
    np.load(f'{d}/dec_ids.npy').astype(np.int32).tofile(f'{d}/dec_ids.i32')
    np.load(f'{d}/dec_logits.npy').astype(np.float32).tofile(f'{d}/dec_logits.f32')
    hs = np.load(f'{d}/hs.npy').astype(np.float32); hs.tofile(f'{d}/hs.f32')
    print(case, ids.shape, lg.shape, hs.shape)
