"""Convert the Inkling-2-Mini checkpoint (NVFP4 / MXFP8 experts, BF16 rest) into a g2e v6 weight pack (.g2w).

    python3 convert_inkling2.py <model_dir> <out.g2w> [--workers N] [--experts native]

Experts (128 per layer; w13 rows stay in checkpoint order gate_0, up_0, gate_1, ...; w2 natural k, 256-col interleaved):
  NVFP4 (layers 0..19): wq u8 [E, K/2, N] (low nibble = even k), bs f8 [E, K/16, N] (fp8_143 code of scale / 8 =
      E4M3FN code - 0x18), g f32 [E] = amax / den * 512. Value: (E2M1 as fp8_143 s00eem00 = elem*2^-6) * (scale / 8)
      * (amax / den * 512) = elem * scale * amax / den.
  MXFP8 (layers 20..23): wq u8 [E, K, N] (checkpoint E4M3FN codes), bs f8 [E, K/32, N] (fp8_143 code of 2^(s - s_max)
      in [2^-6, 1.0] where s_max is the matrix's largest block exponent in the layer), g f32 [E] = 2^(s_max - 127).
      Value: E4M3FN(code) * 2^(s - s_max) * 2^(s_max - 127) = E4M3FN(code) * 2^(s - 127).

Attention projections stay BF16, fused per layer along the output dimension as one tensor L<i>.wx:
  sliding layers: [q 3072 | gate 3072 | rel 384 | sink 24 | k 1024 | v 1024]                          8600 rows
  full layers:    [q 3072 | gate 3072 | rel 384 | sink 24 | k 1024 | k_taps 32 | v 1024 | v_taps 32]  8664 rows
(q-side rows 0..6551 read the query input, k rows the key input, v rows the value input; the k/v tap projections
of the dynamic convs read the same inputs as k / v).
Short-conv taps are stored F32 [4, C] in time order (tap 3 multiplies the current token); the dynamic convs'
static taps likewise ([4, 1024], time order) and their projected taps keep the checkpoint row order head*4 + lag.

File layout as convert_qwen35.py: "G2EW0001", u64 header bytes, JSON header, zero padding up to DATA_START (4 MiB),
4096-aligned tensors.
"""
import argparse, json, os, struct, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


def bf16_round(x):
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
# all 256 byte values -> the two E2M1 values (low nibble first)
NIB = np.stack([E2M1[np.arange(256) & 15], E2M1[np.arange(256) >> 4]], axis=-1).astype(np.float32)


class Checkpoint:
    def __init__(self, root, files=None):
        self.root = root
        self.where = {}
        for f in sorted(os.listdir(root)):
            if not f.endswith(".safetensors") or (files and f not in files):
                continue
            with open(os.path.join(root, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for name, info in h.items():
                self.where[name] = (f, 8 + n, info)

    def has(self, name):
        return name in self.where

    def info(self, name):
        return self.where[name][2]

    def get(self, name, rows=None):
        """rows: optional (first, count) slice of the outermost dimension"""
        f, base, info = self.where[name]
        a, b = info["data_offsets"]
        dt = {"F8_E4M3": np.uint8, "U8": np.uint8, "BF16": np.uint16, "F32": np.float32}[info["dtype"]]
        shape = list(info["shape"])
        isz = np.dtype(dt).itemsize
        if rows is not None:
            inner = int(np.prod(shape[1:])) * isz
            a, b = a + rows[0] * inner, a + (rows[0] + rows[1]) * inner
            shape[0] = rows[1]
        with open(os.path.join(self.root, f), "rb") as fh:
            fh.seek(base + a)
            buf = fh.read(b - a)
        return np.frombuffer(buf, dt).reshape(shape), info["dtype"]

    def f32(self, name):
        a, dt = self.get(name)
        if dt == "BF16":
            return from_bf16_bits(a)
        if dt == "F32":
            return a.astype(np.float32)
        raise ValueError(f"{name}: {dt}")

    def bf16(self, name):
        a, dt = self.get(name)
        if dt == "BF16":
            return a
        if dt == "F32":
            return bf16_bits(a)
        raise ValueError(f"{name}: {dt}")


CKPT = None
LLM = "model.llm."


def _init(root):
    global CKPT
    CKPT = Checkpoint(root)


# ------------------------------------------------------------------------------------------------------------- experts
def expert_inner(layer, which, e0, ne):
    """inner values f32 [ne, N, K] of experts e0..e0+ne (w13 de-interleaved), expert factors f32 [ne], kind"""
    p = f"{LLM}layers.{layer}.mlp.experts.{which}_weight"
    w, wdt = CKPT.get(p, (e0, ne))
    s, sdt = CKPT.get(p + "_scale", (e0, ne))
    if wdt == "U8":  # NVFP4
        v = NIB[w].reshape(ne, w.shape[1], w.shape[2] * 2)         # [ne, N, K]
        bs = E4M3FN[s]                                              # [ne, N, K/16]
        v = (v.reshape(ne, v.shape[1], -1, 16) * bs[..., None]).reshape(v.shape)
        amax = CKPT.f32(p + "_amax")[e0:e0 + ne]
        den = CKPT.f32(p + "_outer_scale_denominator").reshape(-1)[0]
        g = (amax / den).astype(np.float32)
        kind = "nvfp4"
    else:  # MXFP8
        v = E4M3FN[w]
        sc = np.exp2(s.astype(np.float32) - 127.0)
        v = (v.reshape(ne, v.shape[1], -1, 32) * sc[..., None]).reshape(v.shape)
        g = np.ones(ne, np.float32)
        kind = "mxfp8"
    # w13 rows stay in the checkpoint order: gate_0, up_0, gate_1, up_1, ...
    return np.ascontiguousarray(v, np.float32), g, kind


def _interleave256_cols(a):
    """[E, K_dim, N] -> per 256-column chunk: stored[2i] = n_i, stored[2i + 1] = n_{128 + i}."""
    E_, K_, N_ = a.shape
    return a.reshape(E_, K_, N_ // 256, 2, 128).transpose(0, 1, 2, 4, 3).reshape(E_, K_, N_)


def job_experts_nvfp4(layer, which):
    p = f"{LLM}layers.{layer}.mlp.experts.{which}_weight"
    w, wdt = CKPT.get(p)
    s, sdt = CKPT.get(p + "_scale")
    if wdt != "U8" or sdt != "F8_E4M3":
        raise ValueError(f"L{layer}.{which}: expected U8 weights and F8_E4M3 scales, got {wdt}/{sdt}")
    # E4M3FN codes in [0x28, 0x7e] correspond strictly to positive normal values in [0.25, 448.0];
    # subtracting 0x18 yields exact normal Gaudi2 fp8_143 codes for scale / 8 in [0.03125, 56.0].
    bad = (s < 0x28) | (s > 0x7e)
    if np.any(bad):
        idx = tuple(int(x[0]) for x in np.nonzero(bad))
        code = int(s[idx])
        val = float(E4M3FN[code])
        raise ValueError(
            f"L{layer}.{which}: {int(bad.sum())} NVFP4 block scale(s) outside [0.25, 448] or zero/negative/NaN "
            f"(first at {idx}: code=0x{code:02x}, value={val})"
        )
    amax = CKPT.f32(p + "_amax")
    den = CKPT.f32(p + "_outer_scale_denominator").reshape(-1)[0]
    if not np.isfinite(den) or den == 0.0 or not np.all(np.isfinite(amax)):
        raise ValueError(f"L{layer}.{which}: invalid outer scale amax/denominator (den={den})")
    g = ((amax / den).astype(np.float32) * np.float32(512.0)).astype(np.float32)

    wq = w.transpose(0, 2, 1)                         # [E, K/2, N]
    bs = (s - np.uint8(0x18)).transpose(0, 2, 1)      # [E, K/16, N]
    if which == "w2":
        wq = _interleave256_cols(wq)
        bs = _interleave256_cols(bs)
    o = f"L{layer}.{which}"
    return [
        (o + ".wq", "u8", np.ascontiguousarray(wq)),
        (o + ".bs", "f8", np.ascontiguousarray(bs)),
        (o + ".g", "f32", np.ascontiguousarray(g)),
    ]


def job_experts_mxfp8(layer, which):
    p = f"{LLM}layers.{layer}.mlp.experts.{which}_weight"
    w, wdt = CKPT.get(p)
    s, sdt = CKPT.get(p + "_scale")
    if wdt != "F8_E4M3" or sdt != "U8":
        raise ValueError(f"L{layer}.{which}: expected F8_E4M3 weights and U8 scales, got {wdt}/{sdt}")
    nan_mask = (w & np.uint8(0x7F)) == np.uint8(0x7F)
    if np.any(nan_mask):
        idx = tuple(int(x[0]) for x in np.nonzero(nan_mask))
        raise ValueError(
            f"L{layer}.{which}: {int(nan_mask.sum())} NaN E4M3FN weight code(s) (first at {idx}: 0x{int(w[idx]):02x})"
        )
    s_min = int(s.min())
    s_max = int(s.max())
    if s_max >= 255 or s_max < 1:
        raise ValueError(f"L{layer}.{which}: invalid UE8M0 block scale range [{s_min}, {s_max}]")
    if s_max - s_min > 6:
        raise ValueError(
            f"L{layer}.{which}: MXFP8 block exponent span {s_max - s_min} [{s_min - 127}, {s_max - 127}] exceeds 6 "
            f"(normal fp8_143 range <= 1)"
        )
    # Stored bs = Gaudi2 fp8_143 code of 2^(s - s_max) in [2^-6, 1.0]: bias 7 -> exp field in [1, 7], mantissa 0.
    bs_code = ((7 + s.astype(np.int16) - s_max).astype(np.uint8) << np.uint8(3))
    g = np.full(w.shape[0], np.float32(2.0 ** (s_max - 127)), dtype=np.float32)

    wq = w.transpose(0, 2, 1)        # [E, K, N]
    bs = bs_code.transpose(0, 2, 1)  # [E, K/32, N]
    if which == "w2":
        wq = _interleave256_cols(wq)
        bs = _interleave256_cols(bs)
    o = f"L{layer}.{which}"
    return [
        (o + ".wq", "u8", np.ascontiguousarray(wq)),
        (o + ".bs", "f8", np.ascontiguousarray(bs)),
        (o + ".g", "f32", np.ascontiguousarray(g)),
    ]


def job_experts(layer, which):
    p = f"{LLM}layers.{layer}.mlp.experts.{which}_weight"
    dt = CKPT.info(p)["dtype"]
    if dt == "U8":
        return job_experts_nvfp4(layer, which)
    if dt == "F8_E4M3":
        return job_experts_mxfp8(layer, which)
    raise ValueError(f"L{layer}.{which}: unsupported checkpoint expert weight dtype {dt!r}")


# ------------------------------------------------------------------------------------------------------------- dense
def job_bf16(out, names, shape=None):
    arrs = [CKPT.bf16(n) for n in names]
    a = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    if shape is not None:
        a = a.reshape(shape)
    return [(out, "bf16", np.ascontiguousarray(a))]


def job_f32(out, names, shape=None):
    arrs = [CKPT.f32(n) for n in names]
    a = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
    if shape is not None:
        a = a.reshape(shape)
    return [(out, "f32", np.ascontiguousarray(a, np.float32))]


def job_lm_head(out, unpadded):
    a = CKPT.bf16(LLM + "unembed.weight")[:unpadded]
    return [(out, "bf16", np.ascontiguousarray(a))]


def job_conv(out, name):
    """conv1d weight [C, 1, 4] (index 3 = current token) -> F32 [4, C] time order"""
    w = CKPT.f32(name).reshape(-1, 4)
    return [(out, "f32", np.ascontiguousarray(w.T))]


def job_static_taps(out, name):
    """dynamic conv static_w [4 (lag), C] (lag 0 = current) -> F32 [4, C] time order (row 3 = current)"""
    w = CKPT.f32(name)
    return [(out, "f32", np.ascontiguousarray(w[::-1]))]


def job_msg_queries(out):
    q = CKPT.f32(LLM + "middle_cycle_cross_pass_queries").reshape(-1, 2, 2304)      # [12, (query, mlp)]
    kv = CKPT.f32(LLM + "middle_cycle_cross_pass_kv_queries").reshape(-1, 2, 2304)  # [12, (key, value)]
    m = np.concatenate([q[:, :1], kv, q[:, 1:]], axis=1)                           # [12, 4 (q, k, v, mlp), H]
    return [(out, "f32", np.ascontiguousarray(m))]


def job_mg_stack(o, p):
    downs, fgs = [], []
    for ck in ("attn", "k", "v", "mlp"):
        c = f"{p}controllers.{ck}."
        downs.append(CKPT.bf16(c + "down_weight"))
        fg = np.concatenate([CKPT.bf16(c + "forget_weight"), CKPT.bf16(c + "gain_weight")])  # [2H, R]
        n2, r = fg.shape
        fgs.append(fg.reshape(n2 // 128, 64, 2, r).transpose(0, 2, 1, 3).reshape(n2, r))
    return [(o + "mg_down", "bf16", np.ascontiguousarray(np.stack(downs))), (o + "mg_fg", "bf16", np.ascontiguousarray(np.stack(fgs)))]


def run_job(job):
    kind, args = job
    return {"experts": job_experts, "bf16": job_bf16, "f32": job_f32, "lm_head": job_lm_head, "conv": job_conv,
            "static_taps": job_static_taps, "msgq": job_msg_queries, "mg_stack": job_mg_stack}[kind](*args)


def plan(ckpt, cfg):
    t = cfg["text_config"]
    L = t["num_hidden_layers"]
    full = [(i + 1) % 6 == 0 for i in range(L)]
    lc = t["loop_config"]
    jobs = [("bf16", ("embed", [LLM + "embed.weight"])),
            ("bf16", ("embed_norm", [LLM + "embed_norm.weight"])),
            ("bf16", ("final_norm", [LLM + "norm.weight"])),
            ("f32", ("final_q", [LLM + "attn_res.output_query"])),
            ("lm_head", ("lm_head", t["unpadded_vocab_size"])),
            ("msgq", ("msg_q",))]
    layer_experts = []
    for i in range(L):
        p = f"{LLM}layers.{i}."
        a = p + "attn."
        o = f"L{i}."
        dt13 = ckpt.info(p + "mlp.experts.w13_weight")["dtype"]
        dt2 = ckpt.info(p + "mlp.experts.w2_weight")["dtype"]
        if dt13 != dt2:
            raise ValueError(f"L{i}: mismatched expert weight dtypes w13={dt13} w2={dt2}")
        if dt13 == "U8":
            layer_experts.append("nvfp4")
        elif dt13 == "F8_E4M3":
            layer_experts.append("mxfp8")
        else:
            raise ValueError(f"L{i}: unsupported checkpoint expert weight dtype {dt13!r}")
        side = [a + "wq_du.weight", a + "gate_attn.weight", a + "wr_du.weight", a + "attn_sink_proj.weight"]
        if full[i]:
            names = side + [a + "wk_dv.weight", a + "k_sconv.weight_proj", a + "wv_dv.weight", a + "v_sconv.weight_proj"]
            jobs.append(("static_taps", (o + "k_taps", a + "k_sconv.static_w")))
            jobs.append(("static_taps", (o + "v_taps", a + "v_sconv.static_w")))
            for n in ("q", "k", "v"):
                jobs.append(("bf16", (o + n + "_norm", [p + n + "_input_norm.weight"])))
        else:
            names = side + [a + "wk_dv.weight", a + "wv_dv.weight"]
            jobs.append(("conv", (o + "k_taps", a + "k_sconv.weight")))
            jobs.append(("conv", (o + "v_taps", a + "v_sconv.weight")))
            jobs.append(("bf16", (o + "attn_norm", [p + "attn_norm.weight"])))
        jobs.append(("bf16", (o + "wx", names)))
        jobs.append(("bf16", (o + "wo", [a + "wo_ud.weight"])))
        jobs.append(("f32", (o + "q_scale", [a + "q_l2_clip.scale"])))
        jobs.append(("f32", (o + "r_scale", [a + "rel_l2_clip.scale"])))
        jobs.append(("bf16", (o + "rel_proj", [a + "rel_logits_proj.proj"])))
        jobs.append(("conv", (o + "attn_conv", p + "attn_sconv.weight")))
        jobs.append(("conv", (o + "mlp_conv", p + "mlp_sconv.weight")))
        jobs.append(("bf16", (o + "mlp_norm", [p + "mlp_norm.weight"])))
        if i > 0:
            jobs.append(("f32", (o + "attn_q", [f"{LLM}attn_res.attn_queries.{i - 1}"], (-1, 2304))))
        jobs.append(("f32", (o + "mlp_q", [f"{LLM}attn_res.mlp_queries.{i}"], (1, 2304))))
        jobs.append(("bf16", (o + "router", [p + "mlp.gate.weight"])))
        jobs.append(("f32", (o + "router_bias", [p + "mlp.gate.bias"])))
        jobs.append(("experts", (i, "w13")))
        jobs.append(("experts", (i, "w2")))
        if lc["core_start"] <= i < lc["core_end"]:
            for gname, ck in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
                c = f"{p}controllers.{ck}."
                jobs.append(("bf16", (f"{o}mg_{gname}_down", [c + "down_weight"])))
                # [forget | gain] stacked: [2H, rank]
                jobs.append(("bf16", (f"{o}mg_{gname}_fg", [c + "forget_weight", c + "gain_weight"])))
            # the same, stacked for the fused message gates: mg_down [4 (q, k, v, mlp), rank, 2H]; mg_fg [4, 2H, rank]
            # with the rows of every 128 in the interleaved order of the TPC kernels ([even 64 | odd 64])
            jobs.append(("mg_stack", (o, p)))
    return jobs, full, layer_experts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--experts", choices=["native"], default="native")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    ckpt = Checkpoint(args.model, files=["model-00001-of-00001.safetensors"])
    jobs, full, layer_experts = plan(ckpt, cfg)
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
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} jobs, {offset / 2**30:.1f} GiB, {time.time() - t0:.0f} s", flush=True)
        f.write(b"\0" * ((-offset) % ALIGN))
        t = dict(cfg["text_config"])
        header = {"format": "g2e-weights-1", "model_type": "inkling2_loop", "expert_layout": "ekn_gu2", "mg_stack": True,
                  "w2_rows": "natural", "pack_version": 6, "experts": "native", "layer_experts": layer_experts,
                  "nvfp4_bs": "fp8_143_div8", "mxfp8_bs": "fp8_143_le1",
                  "model": os.path.basename(os.path.normpath(args.model)),
                  "layers": t["num_hidden_layers"], "layer_kinds": ["full" if x else "sliding" for x in full],
                  "config": t, "eos_token_id": cfg.get("eos_token_id"), "image_token_id": cfg.get("image_token_id", 200054),
                  "data_start": DATA_START, "data_bytes": offset, "tensors": entries}
        hj = json.dumps(header).encode()
        head = b"G2EW0001" + struct.pack("<Q", len(hj)) + hj
        if len(head) > DATA_START:
            raise SystemExit("header larger than DATA_START")
        f.seek(0)
        f.write(head)
    print(f"wrote {args.out}: {len(entries)} tensors, {offset / 2**30:.2f} GiB data, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
