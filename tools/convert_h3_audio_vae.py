"""Convert MiniMax-H3 Audio VAE (BigVGAN 32kHz decoder) into a g2e weight pack (.g2w).

    python3 convert_h3_audio_vae.py <audio_vae_dir> <out.g2w>

Pre-folds PyTorch weight_norm pairs (weight_g, weight_v) into plain `weight` tensors in FP32
so the C++ BigVGAN decoder executes pure Conv1d / ConvTranspose1d / SnakeBeta without runtime
norm division. Total size is ~0.52 GiB F32.
"""
import argparse
import json
import os
import struct
import time
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vae_dir", help="Path to FL2VA/audio_vae directory")
    ap.add_argument("out", help="Output .g2w path")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.vae_dir, "config.json")))
    st_path = os.path.join(args.vae_dir, cfg.get("source_safetensors_path", "model.safetensors"))

    t0 = time.time()
    with open(st_path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
    hdr.pop("__metadata__", None)

    def load(k):
        info = hdr[k]
        a, b = info["data_offsets"]
        with open(st_path, "rb") as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        return np.frombuffer(buf, np.float32).reshape(info["shape"]).copy()

    dec_keys = sorted(k for k in hdr if k.startswith("dec_in_proj.") or k.startswith("decoder."))
    folded = {}
    skip = set()
    for k in dec_keys:
        if k in skip:
            continue
        if k.endswith(".weight_g"):
            prefix = k[:-len(".weight_g")]
            kv = prefix + ".weight_v"
            wg = load(k)
            wv = load(kv)
            # PyTorch weight_norm default dim=0: norm over all axes except axis 0
            axes = tuple(range(1, wv.ndim))
            norm = np.sqrt(np.sum(wv.astype(np.float64) ** 2, axis=axes, keepdims=True) + 1e-12)
            w = (wg.astype(np.float64) * (wv.astype(np.float64) / norm)).astype(np.float32)
            folded[prefix + ".weight"] = np.ascontiguousarray(w)
            skip.add(kv)
        elif k.endswith(".weight_v"):
            continue
        else:
            folded[k] = np.ascontiguousarray(load(k))

    entries = {}
    offset = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as dst:
        dst.seek(DATA_START)
        for name in sorted(folded):
            arr = folded[name]
            raw = arr.tobytes()
            pad = (-offset) % ALIGN
            if pad:
                dst.write(b"\0" * pad)
                offset += pad
            entries[name] = {"dtype": "f32", "shape": list(arr.shape), "offset": offset, "bytes": len(raw)}
            dst.write(raw)
            offset += len(raw)

        header = json.dumps({
            "format": "g2e-weights-1",
            "model": "minimax-h3-audio-vae-decoder",
            "data_start": DATA_START,
            "data_bytes": offset,
            "config": cfg,
            "tensors": entries,
        }, separators=(",", ":")).encode()
        assert 16 + len(header) <= DATA_START
        dst.seek(0)
        dst.write(b"G2EW0001" + struct.pack("<Q", len(header)) + header)
        dst.write(b"\0" * (DATA_START - 16 - len(header)))
    print(f"wrote {args.out}: {len(entries)} tensors, {offset / (1 << 30):.3f} GiB in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
