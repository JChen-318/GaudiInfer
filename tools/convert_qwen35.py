"""Convert the Qwen3.8-27B FP8 (128x128 block-scaled) checkpoint into a g2e weight pack (.g2w).

    python3 convert_qwen35.py <model_dir> <out.g2w> [--workers N] [--limit-layers L]

The block-scaled e4m3fn weights are dequantized and requantized per output row exactly the way vLLM-gaudi does it
with VLLM_HPU_FORCE_CHANNEL_FP8 (the production setting), emulating its BF16 arithmetic:
    w_bf16 = bf16(fp8 * bf16_block_scale)
    scale  = bf16(bf16(max|row| + 1e-8) / 240)          (Gaudi2 FP8 max = 240)
    codes  = fp8_143(w_bf16 * bf16(1 / scale))           (round to nearest even, saturating)
and the scale is stored as F32. Fused projections are concatenated along the output dimension:
    attention  qkv = [q_proj (q and gate interleaved per head, as in HF); k_proj; v_proj]   14336 rows
    GDN        qkvz = [in_proj_qkv; in_proj_z]                                                16384 rows
    GDN        ba   = [in_proj_b; in_proj_a]  (BF16)                                          96 rows
    MLP        gate_up = [gate_proj; up_proj]                                                 34816 rows

File layout (little endian): "G2EW0001", u64 header bytes, JSON header, zero padding up to DATA_START (4 MiB),
tensor data. Every tensor starts at a 4096-aligned offset relative to DATA_START, so the data region can be copied
to one HBM arena as-is and each tensor bound at (arena, offset).
"""
import argparse, json, os, re, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20
FP8_MAX = 240.0


# ---------------------------------------------------------------------------------------------------------------- numerics
def bf16_round(x):
    """float32 array -> float32 array holding the nearest-even BF16 value."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def bf16_bits(x):
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def from_bf16_bits(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _e4m3fn_lut():
    lut = np.zeros(256, np.float32)
    for v in range(256):
        s, e, m = v >> 7, (v >> 3) & 15, v & 7
        if e == 15 and m == 7:
            x = np.nan
        elif e == 0:
            x = (m / 8.0) * 2.0 ** -6
        else:
            x = (1 + m / 8.0) * 2.0 ** (e - 7)
        lut[v] = -x if s else x
    return lut


E4M3FN = _e4m3fn_lut()


def to_fp8_143(x):
    """float32 -> uint8 Gaudi2 fp8_143 (bias 7), round to nearest even, saturating at +-240."""
    x = np.asarray(x, np.float32)
    sign = (np.signbit(x).astype(np.uint8) << np.uint8(7))
    a = np.minimum(np.abs(x), np.float32(FP8_MAX))
    u = a.view(np.uint32)
    # normal range: keep 3 mantissa bits with RNE, carry into the exponent naturally
    r = (u + np.uint32(0x7FFFF) + ((u >> np.uint32(20)) & np.uint32(1))) & np.uint32(0xFFF00000)
    exp = (r >> np.uint32(23)).astype(np.int32) - 127 + 7
    man = ((r >> np.uint32(20)) & np.uint32(7)).astype(np.int32)
    normal = np.clip(exp, 0, 15) * 8 + man
    # subnormal range (|x| < 2^-6): quantum 2^-9; rint is round-half-even; 8 == smallest normal code
    sub = np.rint(a * np.float32(512.0)).astype(np.int32)
    code = np.where(a < np.float32(2.0 ** -6), sub, normal)
    code = np.minimum(code, 0x77).astype(np.uint8)  # 240 is code 0x77
    return code | sign


def requant_rows(w_codes, block_scale, block=128):
    """e4m3fn codes [N,K] + BF16 block scales [N/128,K/128] -> (fp8_143 codes [N,K], f32 scales [N])."""
    n, k = w_codes.shape
    w = E4M3FN[w_codes]
    s = np.repeat(np.repeat(block_scale, block, axis=0)[:n], block, axis=1)[:, :k]
    w = bf16_round(w * s)
    del s
    peak = np.abs(w).max(axis=1)
    scale = bf16_round(bf16_round(peak + np.float32(1e-8)) / np.float32(FP8_MAX))
    inv = bf16_round(np.float32(1.0) / scale)
    codes = to_fp8_143(w * inv[:, None])
    return codes, scale.astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------- safetensors
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

    def has(self, name):
        return name in self.where

    def get(self, name):
        f, base, info = self.where[name]
        a, b = info["data_offsets"]
        dt = {"F8_E4M3": np.uint8, "BF16": np.uint16, "F32": np.float32}[info["dtype"]]
        with open(os.path.join(self.root, f), "rb") as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        return np.frombuffer(buf, dt).reshape(info["shape"]), info["dtype"]


# ---------------------------------------------------------------------------------------------------------------- jobs
# A job produces one or more output tensors: list of (name, dtype, ndarray).
CKPT = None


def _init(root):
    global CKPT
    CKPT = Checkpoint(root)


def job_fp8(out, parts):
    codes, scales = [], []
    for p in parts:
        w, dt = CKPT.get(p + ".weight")
        assert dt == "F8_E4M3", p
        s, sdt = CKPT.get(p + ".weight_scale_inv")
        s = from_bf16_bits(s) if sdt == "BF16" else s.astype(np.float32)
        c, sc = requant_rows(w, s)
        codes.append(c)
        scales.append(sc)
    return [(out + ".w", "f8", np.concatenate(codes)), (out + ".s", "f32", np.concatenate(scales))]


def job_bf16(out, parts, shape=None):
    arrs = []
    for p in parts:
        a, dt = CKPT.get(p)
        assert dt == "BF16", (p, dt)
        arrs.append(a)
    a = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    if shape is not None:
        a = a.reshape(shape)
    return [(out, "bf16", np.ascontiguousarray(a))]


def run_job(job):
    kind, args = job
    return {"fp8": job_fp8, "bf16": job_bf16}[kind](*args)


def plan(ckpt, cfg, limit_layers):
    t = cfg.get("text_config", cfg)
    L = t["num_hidden_layers"] if limit_layers is None else limit_layers
    pre = "model.language_model."
    jobs = [("bf16", ("embed", [pre + "embed_tokens.weight"])),
            ("bf16", ("final_norm", [pre + "norm.weight"])),
            ("bf16", ("lm_head", ["lm_head.weight"]))]
    kinds = []
    for i in range(L):
        p = f"{pre}layers.{i}."
        o = f"L{i}."
        jobs.append(("bf16", (o + "in_norm", [p + "input_layernorm.weight"])))
        jobs.append(("bf16", (o + "post_norm", [p + "post_attention_layernorm.weight"])))
        jobs.append(("fp8", (o + "gate_up", [p + "mlp.gate_proj", p + "mlp.up_proj"])))
        jobs.append(("fp8", (o + "down", [p + "mlp.down_proj"])))
        if ckpt.has(p + "self_attn.q_proj.weight"):
            kinds.append("attention")
            a = p + "self_attn."
            jobs.append(("fp8", (o + "qkv", [a + "q_proj", a + "k_proj", a + "v_proj"])))
            jobs.append(("fp8", (o + "o", [a + "o_proj"])))
            jobs.append(("bf16", (o + "q_norm", [a + "q_norm.weight"])))
            jobs.append(("bf16", (o + "k_norm", [a + "k_norm.weight"])))
        else:
            kinds.append("gdn")
            g = p + "linear_attn."
            jobs.append(("fp8", (o + "qkvz", [g + "in_proj_qkv", g + "in_proj_z"])))
            jobs.append(("fp8", (o + "out", [g + "out_proj"])))
            jobs.append(("bf16", (o + "ba", [g + "in_proj_b.weight", g + "in_proj_a.weight"])))
            jobs.append(("bf16", (o + "conv", [g + "conv1d.weight"], (-1, t["linear_conv_kernel_dim"]))))
            jobs.append(("bf16", (o + "A_log", [g + "A_log"])))
            jobs.append(("bf16", (o + "dt_bias", [g + "dt_bias"])))
            jobs.append(("bf16", (o + "gdn_norm", [g + "norm.weight"])))
    return jobs, kinds, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-layers", type=int, default=None)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    ckpt = Checkpoint(args.model)
    jobs, kinds, L = plan(ckpt, cfg, args.limit_layers)
    t0 = time.time()
    entries = {}
    offset = 0
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
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} jobs, {offset / 2**30:.1f} GiB, {time.time() - t0:.0f} s", flush=True)
        f.write(b"\0" * ((-offset) % ALIGN))
        t = cfg.get("text_config", cfg)
        header = {"format": "g2e-weights-1", "model": os.path.basename(os.path.normpath(args.model)),
                  "layers": L, "layer_kinds": kinds, "config": t, "data_start": DATA_START,
                  "data_bytes": offset, "tensors": entries}
        hj = json.dumps(header).encode()
        head = b"G2EW0001" + struct.pack("<Q", len(hj)) + hj
        if len(head) > DATA_START:
            raise SystemExit("header larger than DATA_START")
        f.seek(0)
        f.write(head)
    print(f"wrote {args.out}: {len(entries)} tensors, {offset / 2**30:.2f} GiB data, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
