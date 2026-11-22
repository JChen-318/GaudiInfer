"""Convert MiniMax-H3 Qwen3-VL-32B text/vision encoder (layers 0..49 only) to a g2e weight pack (.g2w).

    python3 convert_h3_text_encoder.py <text_encoder_dir> <out.g2w> [--workers N] [--limit-layers L]

MiniMax-H3 only consumes hidden_states[50] (the unnormalized output of decoder layer 49,
before final norm and lm_head), so layers 50..63, model.language_model.norm, and lm_head
are omitted — saving ~22% HBM and compute (~24.8 GiB total in FP8 vs ~66.7 GB BF16).

Weight naming in .g2w:
  Language backbone:
    embed (BF16 [151936, 5120])
    L{i}.in_norm, L{i}.post_norm (BF16 [5120])
    L{i}.q_norm, L{i}.k_norm (BF16 [128])
    L{i}.qkv.w (F8 [10240, 5120] = [Q(8192); K(1024); V(1024)]), L{i}.qkv.s (F32 [10240])
    L{i}.o.w (F8 [5120, 8192]), L{i}.o.s (F32 [5120])
    L{i}.gate_up.w (F8 [51200, 5120] = [gate(25600); up(25600)]), L{i}.gate_up.s (F32 [51200])
    L{i}.down.w (F8 [5120, 25600]), L{i}.down.s (F32 [5120])
  Vision encoder (BF16, matching src/vision.cpp + DeepStack mergers):
    vis.patch_embed.proj.weight ([1152, 1536]), vis.patch_embed.proj.bias ([1152])
    vis.pos_embed.weight ([2304, 1152])
    vis.blocks.{0..26}.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}.{weight,bias}
    vis.merger.{norm,linear_fc1,linear_fc2}.{weight,bias}
    vis.deepstack_merger_list.{0..2}.{norm,linear_fc1,linear_fc2}.{weight,bias}
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
SELECTED_LM_LAYERS = 50


def bf16_round(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


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

    def keys(self):
        return self.where.keys()

    def get(self, name):
        f, base, info = self.where[name]
        a, b = info["data_offsets"]
        dt = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16}[info["dtype"]]
        with open(os.path.join(self.root, f), "rb") as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        return np.frombuffer(buf, dt).reshape(info["shape"]).copy(), info["dtype"]


CKPT = None


def _init(root):
    global CKPT
    CKPT = Checkpoint(root)


def run_job(job):
    kind = job[0]
    if kind == "bf16":
        out_name, src_key, flatten_2d = job[1], job[2], job[3]
        arr, dt = CKPT.get(src_key)
        assert dt == "BF16", (src_key, dt)
        if flatten_2d:
            arr = arr.reshape(arr.shape[0], -1)
        return [(out_name, "bf16", np.ascontiguousarray(arr))]
    if kind == "bf16_cat":
        out_name, src_keys = job[1], job[2]
        arrs = []
        for k in src_keys:
            arr, dt = CKPT.get(k)
            assert dt == "BF16", (k, dt)
            arrs.append(arr)
        arr_all = np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]
        return [(out_name, "bf16", np.ascontiguousarray(arr_all))]
    if kind == "fp8_cat":
        out_prefix, src_keys = job[1], job[2]
        codes, scales = [], []
        for k in src_keys:
            arr, dt = CKPT.get(k)
            assert dt == "BF16", (k, dt)
            c, s = quant_rows_from_bf16(arr)
            codes.append(c)
            scales.append(s)
        c_all = np.concatenate(codes, axis=0) if len(codes) > 1 else codes[0]
        s_all = np.concatenate(scales, axis=0) if len(scales) > 1 else scales[0]
        return [(out_prefix + ".w", "f8", np.ascontiguousarray(c_all)),
                (out_prefix + ".s", "f32", np.ascontiguousarray(s_all))]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to FL2VA/text_encoder directory")
    ap.add_argument("out", help="Output .g2w path")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-layers", type=int, default=SELECTED_LM_LAYERS)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="Keep LM linear weights in BF16 (exact numerics for 8B encoders)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    ckpt = Checkpoint(args.model)
    L = min(args.limit_layers, cfg["text_config"]["num_hidden_layers"])
    pre = "model.language_model."

    lin_kind = "bf16_cat" if args.bf16 else "fp8_cat"
    lin_suf = ".w" if args.bf16 else ""
    jobs = [("bf16", "embed", pre + "embed_tokens.weight", False)]
    for i in range(L):
        p = f"{pre}layers.{i}."
        o = f"L{i}."
        jobs.extend([
            ("bf16", o + "in_norm", p + "input_layernorm.weight", False),
            ("bf16", o + "post_norm", p + "post_attention_layernorm.weight", False),
            ("bf16", o + "q_norm", p + "self_attn.q_norm.weight", False),
            ("bf16", o + "k_norm", p + "self_attn.k_norm.weight", False),
            (lin_kind, o + "qkv" + lin_suf, [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"]),
            (lin_kind, o + "o" + lin_suf, [p + "self_attn.o_proj.weight"]),
            (lin_kind, o + "gate_up" + lin_suf, [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
            (lin_kind, o + "down" + lin_suf, [p + "mlp.down_proj.weight"]),
        ])
    if not args.no_vision:
        for k in sorted(ckpt.keys()):
            if k.startswith("model.visual."):
                short = "vis." + k[len("model.visual."):]
                jobs.append(("bf16", short, k, short == "vis.patch_embed.proj.weight"))

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
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} jobs  {offset / (1 << 30):.2f} GiB  {time.time() - t0:.1f}s", flush=True)

        header = json.dumps({
            "format": "g2e-weights-1",
            "model": "minimax-h3-qwen3vl",
            "layers": L,
            "blocks_dtype": "bf16" if args.bf16 else "f8",
            "has_vision": not args.no_vision,
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
