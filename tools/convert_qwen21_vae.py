"""Convert Qwen/Qwen-Image-2.1-Turbo AutoencoderKLQwenImage21 (encoder + decoder) to a g2e weight pack (.g2w).

    python3 convert_qwen21_vae.py <vae_dir> <out.g2w>

All 2D conv weights [Cout, Cin, kH, kW] are pre-transposed to SynapseAI NHWC spatial_convolution
layout [kH, kW, Cin, Cout] in BF16 (1x1 convs in mid_block attention are stored as 2D [Cout, Cin]
for direct BF16 GEMM), norm gammas are flattened to 1D [C] BF16, and unused temporal time_conv
weights (never invoked for T=1 still images) are omitted.
"""
import argparse
import json
import os
import struct
import time
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20


def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return (r >> np.uint32(16)).astype(np.uint16)


def read_safetensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
    hdr.pop("__metadata__", None)
    return base, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vae_dir", help="Path to Qwen-Image-2.1-Turbo vae directory")
    ap.add_argument("out", help="Output .g2w path")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.vae_dir, "config.json")))
    st_path = os.path.join(args.vae_dir, "diffusion_pytorch_model.safetensors")
    base, hdr = read_safetensors(st_path)

    t0 = time.time()
    entries = {}
    offset = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(st_path, "rb") as src, open(args.out, "wb") as dst:
        dst.seek(DATA_START)
        for k in sorted(hdr.keys()):
            if ".time_conv." in k:
                # Temporal 1x1 conv is never called when feat_cache=None (single-frame image).
                continue
            info = hdr[k]
            a, b = info["data_offsets"]
            src.seek(base + a)
            buf = src.read(b - a)
            dt_in = info["dtype"]
            if dt_in == "BF16":
                arr = np.frombuffer(buf, np.uint16).reshape(info["shape"]).copy()
            elif dt_in == "F32":
                arr = bf16_bits(np.frombuffer(buf, np.float32).reshape(info["shape"]).copy())
            elif dt_in == "F16":
                arr = bf16_bits(np.frombuffer(buf, np.float16).reshape(info["shape"]).astype(np.float32))
            else:
                raise ValueError(f"Unsupported dtype {dt_in} for {k}")

            if k.endswith(".gamma"):
                arr = np.ascontiguousarray(arr.reshape(-1))
            elif k.endswith(".weight") and arr.ndim == 4:
                if ".attentions." in k or k in ("quant_conv.weight", "post_quant_conv.weight") or k.endswith("conv_shortcut.weight"):
                    # 1x1 conv -> [Cout, Cin] for direct MME GEMM (faster & zero conv padding overhead!)
                    assert arr.shape[2] == 1 and arr.shape[3] == 1, (k, arr.shape)
                    arr = np.ascontiguousarray(arr.reshape(arr.shape[0], arr.shape[1]))
                else:
                    # 3x3 conv -> SynapseAI NHWC spatial_convolution layout [kH, kW, C_in, C_out]
                    arr = np.ascontiguousarray(arr.transpose(2, 3, 1, 0))
            else:
                arr = np.ascontiguousarray(arr)

            raw = arr.tobytes()
            pad = (-offset) % ALIGN
            if pad:
                dst.write(b"\0" * pad)
                offset += pad
            entries[k] = {"dtype": "bf16", "shape": list(arr.shape), "offset": offset, "bytes": len(raw)}
            dst.write(raw)
            offset += len(raw)

        header = json.dumps({
            "format": "g2e-weights-1",
            "model": "qwen-image-2.1-vae",
            "data_start": DATA_START,
            "data_bytes": offset,
            "config": cfg,
            "tensors": entries,
        }, separators=(",", ":")).encode()
        assert 16 + len(header) <= DATA_START
        dst.seek(0)
        dst.write(b"G2EW0001" + struct.pack("<Q", len(header)) + header)
        dst.write(b"\0" * (DATA_START - 16 - len(header)))
    print(f"wrote {args.out}: {len(entries)} tensors, {offset / (1 << 20):.1f} MiB in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
