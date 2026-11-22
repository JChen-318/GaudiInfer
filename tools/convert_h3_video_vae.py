"""Convert MiniMax-H3 Video VAE (ViT3D decoder) into a g2e weight pack (.g2w).

    python3 convert_h3_video_vae.py <video_vae_dir> <out.g2w>

Decoder transformer weights (36 layers, hidden=2048, SwiGLU FFN=8192) are stored in BF16
(~4.83 GiB, matching vLLM-Gaudi's qualified VLLM_GAUDI_H3_VAE_PERSIST_BF16_WEIGHTS=1),
while post_quant_conv, x_embedder, norm_out, proj_out, register_tokens stay F32.
"""
import argparse
import json
import os
import struct
import time
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20
FP8_MAX = 240.0


def bf16_round(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return (r >> np.uint32(16)).astype(np.uint16)


def to_fp8_143(x):
    x = np.asarray(x, np.float32)
    sign = (np.signbit(x).astype(np.uint8) << np.uint8(7))
    a = np.minimum(np.abs(x), np.float32(FP8_MAX))
    u = a.view(np.uint32)
    r = (u + np.uint32(0x7FFFF) + ((u >> np.uint32(20)) & np.uint32(1))) & np.uint32(0xFFF00000)
    exp = (r >> np.uint32(23)).astype(np.int32) - 127 + 7
    man = ((r >> np.uint32(20)) & np.uint32(7)).astype(np.int32)
    normal = np.clip(exp, 0, 15) * 8 + man
    sub = np.rint(a * np.float32(512.0)).astype(np.int32)
    code = np.where(a < np.float32(2.0 ** -6), sub, normal)
    code = np.minimum(code, 0x77).astype(np.uint8)
    return code | sign


def quant_rows(w):
    peak = np.abs(w).max(axis=1)
    scale = bf16_round(bf16_round(peak + np.float32(1e-8)) / np.float32(FP8_MAX))
    inv = bf16_round(np.float32(1.0) / scale)
    return np.ascontiguousarray(to_fp8_143(w * inv[:, None])), np.ascontiguousarray(scale.astype(np.float32))


def read_safetensors(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
    hdr.pop("__metadata__", None)
    return base, hdr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vae_dir", help="Path to FL2VA/video_vae directory")
    ap.add_argument("out", help="Output .g2w path")
    ap.add_argument("--f32-all", action="store_true", help="Keep all decoder weights in F32 (~9.7 GiB)")
    ap.add_argument("--fp8", action="store_true", help="Quantize 36 ViT block linear weights to per-row FP8 E4M3 (~2.3 GiB, 2x faster MME)")
    ap.add_argument("--no-encoder", action="store_true", help="Omit 2D-sliced single-frame CNN encoder weights (~88 MiB F32)")
    args = ap.parse_args()

    top_cfg = json.load(open(os.path.join(args.vae_dir, "config.json")))
    is_image_vae = bool(top_cfg.get("vit_decoder_2d", False))
    if is_image_vae:
        src_cfg = dict(top_cfg)
        src_cfg.setdefault("vae_ratio", 16)
        src_cfg.setdefault("vae_ratio_t", 1)
        st_path = os.path.join(args.vae_dir, "diffusion_pytorch_model.safetensors")
    else:
        src_dir = os.path.join(args.vae_dir, top_cfg.get("source_path", "source"))
        src_cfg = json.load(open(os.path.join(src_dir, "config.json")))
        st_path = os.path.join(src_dir, top_cfg.get("source_safetensors_path", "model.safetensors"))

    t0 = time.time()
    base, hdr = read_safetensors(st_path)
    dec_keys = sorted(k for k in hdr if k.startswith("decoder.") or k.startswith("post_quant_conv."))
    enc_keys = [] if args.no_encoder else sorted(k for k in hdr if k.startswith("encoder.") or k.startswith("quant_conv."))
    all_keys = dec_keys + enc_keys

    entries = {}
    offset = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(st_path, "rb") as src, open(args.out, "wb") as dst:
        dst.seek(DATA_START)
        for k in all_keys:
            info = hdr[k]
            a, b = info["data_offsets"]
            src.seek(base + a)
            buf = src.read(b - a)
            dt_in = info["dtype"]
            if dt_in == "F32":
                arr = np.frombuffer(buf, np.float32).reshape(info["shape"]).copy()
            elif dt_in == "BF16":
                arr = (np.frombuffer(buf, np.uint16).astype(np.uint32) << np.uint32(16)).view(np.float32).reshape(info["shape"]).copy()
            elif dt_in == "F16":
                arr = np.frombuffer(buf, np.float16).astype(np.float32).reshape(info["shape"]).copy()
            else:
                raise ValueError(f"Unsupported dtype {dt_in} for {k}")

            if k == "post_quant_conv.weight":
                arr = arr.reshape(arr.shape[0], arr.shape[1])
            elif (k.startswith("encoder.") or k.startswith("quant_conv.")) and arr.ndim == 5:
                # Causal 3D conv on single image frame (D=1, left zero-pad [0, 0, x]) is mathematically
                # 100% identical to 2D conv with the last temporal slice weight[:, :, -1, :, :]!
                # Store directly in Synapse 2D spatial_convolution layout [kH, kW, C_in, C_out] F32.
                arr = np.ascontiguousarray(arr[:, :, -1, :, :].transpose(2, 3, 1, 0))

            # Keep transformer_blocks in FP8 (if --fp8) or BF16 (unless --f32-all), boundary & encoder layers in F32
            is_blk_lin = args.fp8 and k.startswith("decoder.transformer_blocks.") and k.endswith(".weight") and arr.ndim == 2
            use_bf16 = (not args.f32_all) and k.startswith("decoder.transformer_blocks.")
            items = []
            if is_blk_lin:
                c, s = quant_rows(arr)
                items.append((k, "f8", c))
                items.append((k[:-7] + ".scale", "f32", s))
            elif use_bf16:
                items.append((k, "bf16", np.ascontiguousarray(bf16_bits(arr))))
            else:
                items.append((k, "f32", np.ascontiguousarray(arr.astype(np.float32))))

            for out_k, out_dt, out_arr in items:
                raw = out_arr.tobytes()
                pad = (-offset) % ALIGN
                if pad:
                    dst.write(b"\0" * pad)
                    offset += pad
                entries[out_k] = {"dtype": out_dt, "shape": list(out_arr.shape), "offset": offset, "bytes": len(raw)}
                dst.write(raw)
                offset += len(raw)

        header = json.dumps({
            "format": "g2e-weights-1",
            "model": "minimax-h3-image-vae-decoder" if is_image_vae else "minimax-h3-video-vae-decoder",
            "vit_decoder_2d": is_image_vae,
            "layers": src_cfg["vit_decoder_kwargs"]["num_layers"],
            "blocks_dtype": "f8" if args.fp8 else ("f32" if args.f32_all else "bf16"),
            "has_encoder": not args.no_encoder,
            "data_start": DATA_START,
            "data_bytes": offset,
            "config": {"wrapper": top_cfg, "source": src_cfg},
            "tensors": entries,
        }, separators=(",", ":")).encode()
        assert 16 + len(header) <= DATA_START
        dst.seek(0)
        dst.write(b"G2EW0001" + struct.pack("<Q", len(header)) + header)
        dst.write(b"\0" * (DATA_START - 16 - len(header)))
    print(f"wrote {args.out}: {len(entries)} tensors, {offset / (1 << 30):.2f} GiB in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
