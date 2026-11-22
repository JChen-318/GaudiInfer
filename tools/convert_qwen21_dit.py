"""Convert Qwen/Qwen-Image-2.1-Turbo transformer checkpoint into a g2e weight pack (.g2w).

    python3 convert_qwen21_dit.py <transformer_dir> <out.g2w> [--workers N] [--bf16-all]

Weight naming in .g2w:
  Top-level exact BF16 boundary & shared weights:
    img_in.w              BF16 [4096, 64]
    txt_in.text_norm      BF16 [4096]         (ZeroCenterRMSNorm: weight + 1)
    txt_in.in_layer.w     BF16 [4096, 4096]
    txt_in.out_layer.w    BF16 [4096, 4096]
    time_embed.linear_1.w BF16 [4096, 256]
    time_embed.linear_2.w BF16 [4096, 4096]
    modulation.w          BF16 [16384, 4096]  (shared across all 32 blocks: scale1, gate1, scale2, gate2)
    norm_out.linear.w     BF16 [4096, 4096]
    proj_out.w            BF16 [64, 4096]
  Per-block B0..B31 (FP8 per-row E4M3 linears + BF16 QK-RMSNorms):
    B{i}.norm_q           BF16 [128]
    B{i}.norm_k           BF16 [128]
    B{i}.qkv.w / .s       F8 [12288, 4096] / F32 [12288]  ([to_q; to_k; to_v])
    B{i}.out.w / .s       F8 [4096, 4096]  / F32 [4096]   (to_out.0)
    B{i}.fc1.w / .s       F8 [24576, 4096] / F32 [24576]  ([gate_layer; proj] for fused silu_mul_quant)
    B{i}.fc2.w / .s       F8 [4096, 12288] / F32 [4096]   (img_mlp.out)
"""
import argparse
import json
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20
FP8_MAX = 240.0


def bf16_round(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def bf16_bits(x):
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def from_bf16_bits(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


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


def quant_rows_from_bf16(b_u16):
    w = from_bf16_bits(b_u16)
    peak = np.abs(w).max(axis=1)
    scale = bf16_round(bf16_round(peak + np.float32(1e-8)) / np.float32(FP8_MAX))
    inv = bf16_round(np.float32(1.0) / scale)
    codes = to_fp8_143(w * inv[:, None])
    return codes, scale.astype(np.float32)


class Checkpoint:
    def __init__(self, root):
        self.root = root
        self.where = {}
        for f in sorted(os.listdir(root)):
            if not f.endswith(".safetensors"):
                continue
            with open(os.path.join(root, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for name, info in h.items():
                self.where[name] = (f, 8 + n, info)

    def get_bf16(self, name):
        f, base, info = self.where[name]
        a, b = info["data_offsets"]
        dt = info["dtype"]
        with open(os.path.join(self.root, f), "rb") as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        if dt == "BF16":
            return np.frombuffer(buf, np.uint16).reshape(info["shape"]).copy()
        if dt == "F32":
            return bf16_bits(np.frombuffer(buf, np.float32).reshape(info["shape"]).copy())
        if dt == "F16":
            return bf16_bits(np.frombuffer(buf, np.float16).reshape(info["shape"]).astype(np.float32))
        raise ValueError(f"Unsupported dtype {dt} for {name}")


CKPT = None


def _init(root):
    global CKPT
    CKPT = Checkpoint(root)


def run_job(job):
    kind = job[0]
    if kind == "bf16":
        out_name, src_key = job[1], job[2]
        return [(out_name, "bf16", np.ascontiguousarray(CKPT.get_bf16(src_key)))]
    if kind == "cat_bf16":
        out_name, src_keys = job[1], job[2]
        arrs = [CKPT.get_bf16(k) for k in src_keys]
        arr = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]
        return [(out_name, "bf16", np.ascontiguousarray(arr))]
    if kind == "fp8_cat":
        out_prefix, src_keys = job[1], job[2]
        codes, scales = [], []
        for k in src_keys:
            c, s = quant_rows_from_bf16(CKPT.get_bf16(k))
            codes.append(c)
            scales.append(s)
        c_all = np.concatenate(codes, axis=0) if len(codes) > 1 else codes[0]
        s_all = np.concatenate(scales, axis=0) if len(scales) > 1 else scales[0]
        return [(out_prefix + ".w", "f8", np.ascontiguousarray(c_all)),
                (out_prefix + ".s", "f32", np.ascontiguousarray(s_all))]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to Qwen-Image-2.1-Turbo transformer directory")
    ap.add_argument("out", help="Output .g2w path")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--bf16-all", action="store_true", help="Keep block linears in BF16 instead of FP8")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    L = cfg["num_layers"]
    jobs = [
        ("bf16", "img_in.w", "img_in.weight"),
        ("bf16", "txt_in.text_norm", "txt_in.text_norm.weight"),
        ("bf16", "txt_in.in_layer.w", "txt_in.in_layer.weight"),
        ("bf16", "txt_in.out_layer.w", "txt_in.out_layer.weight"),
        ("bf16", "time_embed.linear_1.w", "time_text_embed.timestep_embedder.linear_1.weight"),
        ("bf16", "time_embed.linear_2.w", "time_text_embed.timestep_embedder.linear_2.weight"),
        ("bf16", "modulation.w", "modulation.1.weight"),
        ("bf16", "norm_out.linear.w", "norm_out.linear.weight"),
        ("bf16", "proj_out.w", "proj_out.weight"),
    ]
    lin_kind = "cat_bf16" if args.bf16_all else "fp8_cat"
    lin_suf = ".w" if args.bf16_all else ""
    for i in range(L):
        p = f"transformer_blocks.{i}."
        o = f"B{i}."
        jobs.extend([
            ("bf16", o + "norm_q", p + "attn.norm_q.weight"),
            ("bf16", o + "norm_k", p + "attn.norm_k.weight"),
            (lin_kind, o + "qkv" + lin_suf, [p + "attn.to_q.weight", p + "attn.to_k.weight", p + "attn.to_v.weight"]),
            (lin_kind, o + "out" + lin_suf, [p + "attn.to_out.0.weight"]),
            (lin_kind, o + "fc1" + lin_suf, [p + "img_mlp.gate_layer.weight", p + "img_mlp.proj.weight"]),
            (lin_kind, o + "fc2" + lin_suf, [p + "img_mlp.out.weight"]),
        ])

    t0 = time.time()
    entries = {}
    offset = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f, ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.model,)) as ex:
        f.seek(DATA_START)
        for done, outs in enumerate(ex.map(run_job, jobs), 1):
            for name, dtype, arr in outs:
                raw = arr.tobytes()
                pad = (-offset) % ALIGN
                if pad:
                    f.write(b"\0" * pad)
                    offset += pad
                entries[name] = {"dtype": dtype, "shape": list(arr.shape), "offset": offset, "bytes": len(raw)}
                f.write(raw)
                offset += len(raw)
            if done % 40 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} jobs  {offset / (1 << 30):.2f} GiB  {time.time() - t0:.1f}s", flush=True)

        header = json.dumps({
            "format": "g2e-weights-1",
            "model": "qwen-image-2.1-dit",
            "layers": L,
            "blocks_dtype": "bf16" if args.bf16_all else "f8",
            "data_start": DATA_START,
            "data_bytes": offset,
            "config": cfg,
            "tensors": entries,
        }, separators=(",", ":")).encode()
        assert 16 + len(header) <= DATA_START
        f.seek(0)
        f.write(b"G2EW0001" + struct.pack("<Q", len(header)) + header)
        f.write(b"\0" * (DATA_START - 16 - len(header)))
    print(f"wrote {args.out}: {len(entries)} tensors, {offset / (1 << 30):.2f} GiB in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
