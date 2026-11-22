"""Convert MiniMax-H3 FL2VA DiT checkpoint into a g2e weight pack (.g2w).

    python3 convert_h3_dit.py <transformer_dir> <out.g2w> [--workers N] [--limit-layers L]
                              [--lora <lora.safetensors> --lora-scale 1.0] [--bf16-adaln]

Design for swappable acceleration LoRAs:
  - By default (no --lora), produces the clean base FP8 weight pack (~30.9 GiB).
    Any LightX2V/Diffusers Turbo LoRA (.safetensors, ~1.3 GiB BF16) can then be
    loaded and hot-swapped directly at runtime in C++ with <0.3 s load time and
    ~0.5 s total BF16 LoRA overhead across 4 denoising steps.
  - With --lora <path>, merges the LoRA delta in FP32/BF16 before per-row FP8
    quantization to produce a zero-runtime-overhead baked pack for that LoRA.

Weight naming in .g2w:
  Top-level FP32 contract tensors (exact F32):
    video_patch_proj.w [5376, 96], video_patch_proj.b [5376]
    audio_patch_proj.w [5376, 32], audio_patch_proj.b [5376]
    time_embedder.proj_in.w [5376, 256], time_embedder.proj_in.b [5376]
    time_embedder.proj_out.w [2688, 5376], time_embedder.proj_out.b [2688]
    final.video_out.w [96, 5376], final.video_out.b [96]
    final.audio_out.w [32, 5376], final.audio_out.b [32]
    rope.inv_freq [16]
  Top-level BF16/FP8 tensors:
    condition_proj.w (F8 [5376, 5120]), condition_proj.s (F32 [5376]), condition_proj.b (BF16 [5376])
    refiner.final_norm (BF16 [5376])
    final.norm (BF16 [5376])
    final.adaln.w (F8 [10752, 2688]), final.adaln.s (F32 [10752]), final.adaln.b (BF16 [10752])
  Token refiner blocks R0..R1 (same linear layout as DiT blocks, no adaln):
    R{i}.norm1, R{i}.norm2 (BF16 [5376])
    R{i}.q_norm, R{i}.k_norm (BF16 [128])
    R{i}.qkv.w (F8 [21504, 5376] reordered to [Q, K, V]), R{i}.qkv.s (F32 [21504])
    R{i}.out.w (F8 [5376, 7168]), R{i}.out.s (F32 [5376])
    R{i}.fc1.w (F8 [28672, 5376] in [gate, up] order), R{i}.fc1.s (F32 [28672])
    R{i}.fc2.w (F8 [5376, 14336]), R{i}.fc2.s (F32 [5376])
  DiT blocks B0..B{L-1}:
    B{i}.norm1, B{i}.norm2 (BF16 [5376])
    B{i}.q_norm, B{i}.k_norm (BF16 [128])
    B{i}.qkv.w (F8 [21504, 5376] reordered to [Q, K, V]), B{i}.qkv.s (F32 [21504])
    B{i}.out.w (F8 [5376, 7168]), B{i}.out.s (F32 [5376])
    B{i}.fc1.w (F8 [28672, 5376] in [gate, up] order), B{i}.fc1.s (F32 [28672])
    B{i}.fc2.w (F8 [5376, 14336]), B{i}.fc2.s (F32 [5376])
    B{i}.adaln.w (F8 [96768, 2688], or BF16 if --bf16-adaln), B{i}.adaln.s (F32 [96768]), B{i}.adaln.b (BF16 [96768])
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
    """float32 array -> float32 array holding the nearest-even BF16 value."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def bf16_bits(x):
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def from_bf16_bits(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def to_fp8_143(x):
    """float32 -> uint8 Gaudi2 fp8_143 (bias 7), round to nearest even, saturating at +-240."""
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


def quant_rows_from_f32(w_f32):
    """BF16-rounded weight [N, K] -> (fp8_143 codes [N, K], f32 per-row scales [N])."""
    w = bf16_round(w_f32)
    peak = np.abs(w).max(axis=1)
    scale = bf16_round(bf16_round(peak + np.float32(1e-8)) / np.float32(FP8_MAX))
    inv = bf16_round(np.float32(1.0) / scale)
    codes = to_fp8_143(w * inv[:, None])
    return codes, scale.astype(np.float32)


def reorder_grouped_qkv(w, num_heads, head_dim):
    """[num_heads * 3 * head_dim, K] grouped -> [Q(H*D), K(H*D), V(H*D), K]."""
    k_dim = w.shape[1]
    g = w.reshape(num_heads, 3, head_dim, k_dim)
    q = g[:, 0, :, :].reshape(num_heads * head_dim, k_dim)
    k = g[:, 1, :, :].reshape(num_heads * head_dim, k_dim)
    v = g[:, 2, :, :].reshape(num_heads * head_dim, k_dim)
    return np.ascontiguousarray(np.concatenate([q, k, v], axis=0))


class SafetensorsStore:
    def __init__(self, path):
        self.where = {}
        self.metadata = {}
        if os.path.isdir(path):
            files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
        else:
            files = [path]
        for fp in files:
            with open(fp, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            meta = h.pop("__metadata__", None)
            if meta:
                self.metadata.update(meta)
            for name, info in h.items():
                self.where[name] = (fp, 8 + n, info)

    def has(self, name):
        return name in self.where

    def get_raw(self, name):
        fp, base, info = self.where[name]
        a, b = info["data_offsets"]
        dt = {"F8_E4M3": np.uint8, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}[info["dtype"]]
        with open(fp, "rb") as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        return np.frombuffer(buf, dt).reshape(info["shape"]).copy(), info["dtype"]

    def get_f32(self, name):
        arr, dt = self.get_raw(name)
        if dt == "BF16":
            return from_bf16_bits(arr)
        return arr.astype(np.float32)


CKPT = None
LORA = None
LORA_SCALE = 1.0
LORA_META_ALPHA = 8.0
ARCH = None
PDD_HEADS = None


def _init(model_dir, lora_path, lora_scale, arch, pdd_heads_path):
    global CKPT, LORA, LORA_SCALE, LORA_META_ALPHA, ARCH, PDD_HEADS
    CKPT = SafetensorsStore(model_dir)
    ARCH = arch
    PDD_HEADS = SafetensorsStore(pdd_heads_path) if pdd_heads_path else None
    if lora_path:
        LORA = SafetensorsStore(lora_path)
        LORA_SCALE = float(lora_scale)
        LORA_META_ALPHA = float(LORA.metadata.get("alpha", 8.0))
    else:
        LORA = None
        LORA_SCALE = 1.0
        LORA_META_ALPHA = 8.0


def _lora_delta(prefix_key):
    """Return float32 B @ A * (scale * mod_alpha / mod_rank) for a Diffusers-style LoRA key prefix, or None."""
    if LORA is None:
        return None
    ka = prefix_key + ".lora_A.default.weight"
    kb = prefix_key + ".lora_B.default.weight"
    if not LORA.has(ka):
        ka = prefix_key + ".lora_A.weight"
        kb = prefix_key + ".lora_B.weight"
    if not LORA.has(ka):
        return None
    a = LORA.get_f32(ka)  # [rank, in]
    b = LORA.get_f32(kb)  # [out, rank]
    rank = float(a.shape[0])
    alpha_k = prefix_key + ".alpha"
    mod_alpha = float(LORA.get_f32(alpha_k).reshape(-1)[0]) if LORA.has(alpha_k) else LORA_META_ALPHA
    mult = LORA_SCALE * (mod_alpha / rank)
    if ".ff.net.0.proj" in prefix_key:
        # LightX2V Turbo stores [up, gate] in ff.net.0.proj.lora_B; swap to [gate, up].
        half = b.shape[0] // 2
        b = np.concatenate([b[half:], b[:half]], axis=0)
    return (b @ a) * np.float32(mult)


def run_job(job):
    kind = job[0]
    if kind == "f32":
        out_name, src_key = job[1], job[2]
        arr = CKPT.get_f32(src_key)
        return [(out_name, "f32", np.ascontiguousarray(arr))]
    if kind == "bf16":
        out_name, src_key = job[1], job[2]
        arr, dt = CKPT.get_raw(src_key)
        if dt != "BF16":
            arr = bf16_bits(arr.astype(np.float32))
        return [(out_name, "bf16", np.ascontiguousarray(arr))]
    if kind == "fp8":
        out_prefix, src_key, lora_keys, is_qkv, keep_bf16 = job[1], job[2], job[3], job[4], job[5]
        w = CKPT.get_f32(src_key)
        if is_qkv:
            w = reorder_grouped_qkv(w, ARCH["num_attention_heads"], ARCH["attention_head_dim"])
        if LORA is not None and lora_keys:
            if is_qkv:
                # lora_keys = [to_q, to_k, to_v]
                d = ARCH["num_attention_heads"] * ARCH["attention_head_dim"]
                for idx, lk in enumerate(lora_keys):
                    delta = _lora_delta(lk)
                    if delta is not None:
                        w[idx * d : (idx + 1) * d] += delta
            else:
                delta = _lora_delta(lora_keys[0])
                if delta is not None:
                    w += delta
        if keep_bf16:
            return [(out_prefix + ".w", "bf16", np.ascontiguousarray(bf16_bits(w)))]
        codes, scales = quant_rows_from_f32(w)
        return [(out_prefix + ".w", "f8", np.ascontiguousarray(codes)),
                (out_prefix + ".s", "f32", np.ascontiguousarray(scales))]
    if kind == "pdd_f32":
        out_name, src_key, step_idx = job[1], job[2], job[3]
        arr = PDD_HEADS.get_f32(src_key)[step_idx]
        return [(out_name, "f32", np.ascontiguousarray(arr))]
    raise ValueError(f"Unknown job kind: {kind}")


def build_jobs(cfg, limit_layers, bf16_adaln, has_pdd_heads=False):
    L = cfg["num_layers"] if limit_layers is None else min(limit_layers, cfg["num_layers"])
    R = cfg["token_refiner_num_layers"]
    audio_enabled = bool(cfg.get("audio_enabled", True))
    jobs = [
        # Exact FP32 contract tensors
        ("f32", "video_patch_proj.w", "video_patch_proj.weight"),
        ("f32", "video_patch_proj.b", "video_patch_proj.bias"),
        ("f32", "time_embedder.proj_in.w", "time_embedder.proj_in.weight"),
        ("f32", "time_embedder.proj_in.b", "time_embedder.proj_in.bias"),
        ("f32", "time_embedder.proj_out.w", "time_embedder.proj_out.weight"),
        ("f32", "time_embedder.proj_out.b", "time_embedder.proj_out.bias"),
        ("f32", "rope.inv_freq", "rope.inv_freq"),
        # Top-level BF16/FP8
        ("fp8", "condition_proj", "condition_proj.weight", None, False, False),
        ("bf16", "condition_proj.b", "condition_proj.bias"),
        ("bf16", "refiner.final_norm", "token_refiner.final_norm.weight"),
        ("bf16", "final.norm", "final_layer.norm.weight"),
        ("fp8", "final.adaln", "final_layer.adaln_proj.linear.weight", ["norm_out.linear"], False, bf16_adaln),
        ("bf16", "final.adaln.b", "final_layer.adaln_proj.linear.bias"),
    ]
    if audio_enabled:
        jobs.extend([
            ("f32", "audio_patch_proj.w", "audio_patch_proj.weight"),
            ("f32", "audio_patch_proj.b", "audio_patch_proj.bias"),
            ("f32", "final.audio_out.w", "final_layer.audio_out.weight"),
            ("f32", "final.audio_out.b", "final_layer.audio_out.bias"),
        ])
    if has_pdd_heads:
        for s in range(6):
            jobs.append(("pdd_f32", f"pdd.video_out.{s}.w", "video_out.weight", s))
            jobs.append(("pdd_f32", f"pdd.video_out.{s}.b", "video_out.bias", s))
    else:
        jobs.extend([
            ("f32", "final.video_out.w", "final_layer.video_out.weight"),
            ("f32", "final.video_out.b", "final_layer.video_out.bias"),
        ])
    for i in range(R):
        sp = f"token_refiner.blocks.{i}."
        lp = f"token_refiner.refiner_blocks.{i}."
        op = f"R{i}."
        jobs.extend([
            ("bf16", op + "norm1", sp + "norm1.weight"),
            ("bf16", op + "norm2", sp + "norm2.weight"),
            ("bf16", op + "q_norm", sp + "attn.q_norm.weight"),
            ("bf16", op + "k_norm", sp + "attn.k_norm.weight"),
            ("fp8", op + "qkv", sp + "attn.qkv_proj.weight",
             [lp + "attn.to_q", lp + "attn.to_k", lp + "attn.to_v"], True, False),
            ("fp8", op + "out", sp + "attn.out_proj.weight", [lp + "attn.to_out.0"], False, False),
            ("fp8", op + "fc1", sp + "mlp.fc1.weight", [lp + "ff.net.0.proj"], False, False),
            ("fp8", op + "fc2", sp + "mlp.fc2.weight", [lp + "ff.net.2"], False, False),
        ])
    for i in range(L):
        sp = f"blocks.{i}."
        lp = f"transformer_blocks.{i}."
        op = f"B{i}."
        jobs.extend([
            ("bf16", op + "norm1", sp + "norm1.weight"),
            ("bf16", op + "norm2", sp + "norm2.weight"),
            ("bf16", op + "q_norm", sp + "attn.q_norm.weight"),
            ("bf16", op + "k_norm", sp + "attn.k_norm.weight"),
            ("fp8", op + "qkv", sp + "attn.qkv_proj.weight",
             [lp + "attn.to_q", lp + "attn.to_k", lp + "attn.to_v"], True, False),
            ("fp8", op + "out", sp + "attn.out_proj.weight", [lp + "attn.to_out.0"], False, False),
            ("fp8", op + "fc1", sp + "mlp.fc1.weight", [lp + "ff.net.0.proj"], False, False),
            ("fp8", op + "fc2", sp + "mlp.fc2.weight", [lp + "ff.net.2"], False, False),
            ("fp8", op + "adaln", sp + "adaln_proj.linear.weight", [lp + "adaln_proj.linear"], False, bf16_adaln),
            ("bf16", op + "adaln.b", sp + "adaln_proj.linear.bias"),
        ])
    return jobs, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to FL2VA/transformer directory")
    ap.add_argument("out", help="Output .g2w file path")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-layers", type=int, default=None)
    ap.add_argument("--lora", type=str, default=None, help="Optional Diffusers/LightX2V LoRA safetensors to merge offline")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--bf16-adaln", action="store_true", help="Keep adaln_proj linear weights in BF16 instead of FP8")
    ap.add_argument("--pdd-heads", type=str, default=None, help="Path to MiniMax-H3-Image pdd_fused_heads.safetensors")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    pdd_path = args.pdd_heads
    if pdd_path is None and not cfg.get("audio_enabled", True):
        cand = os.path.join(os.path.dirname(os.path.abspath(args.model)), "pdd_fused_heads.safetensors")
        if os.path.isfile(cand):
            pdd_path = cand
    jobs, L = build_jobs(cfg, args.limit_layers, args.bf16_adaln, has_pdd_heads=(pdd_path is not None))
    t0 = time.time()
    entries = {}
    offset = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f, ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(args.model, args.lora, args.lora_scale, cfg, pdd_path)
    ) as ex:
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
            if done % 25 == 0 or done == len(jobs):
                dt = time.time() - t0
                print(f"  {done}/{len(jobs)} jobs  {offset / (1 << 30):.2f} GiB  {dt:.1f}s", flush=True)

        header = json.dumps({
            "format": "g2e-weights-1",
            "model": "minimax-h3-image-dit" if not cfg.get("audio_enabled", True) else "minimax-h3-dit",
            "layers": L,
            "refiner_layers": cfg["token_refiner_num_layers"],
            "audio_enabled": bool(cfg.get("audio_enabled", True)),
            "pdd_steps": 6 if pdd_path is not None else 0,
            "adaln_dtype": "bf16" if args.bf16_adaln else "f8",
            "merged_lora": os.path.basename(args.lora) if args.lora else None,
            "lora_scale": args.lora_scale if args.lora else 0.0,
            "data_start": DATA_START,
            "data_bytes": offset,
            "config": cfg,
            "tensors": entries,
        }, separators=(",", ":")).encode()
        assert 16 + len(header) <= DATA_START, f"header too large: {len(header)}"
        f.seek(0)
        f.write(b"G2EW0001" + struct.pack("<Q", len(header)) + header)
        f.write(b"\0" * (DATA_START - 16 - len(header)))
    print(f"wrote {args.out}: {len(entries)} tensors, {offset / (1 << 30):.2f} GiB in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
