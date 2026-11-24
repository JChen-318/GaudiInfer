"""Convert the Inkling-2-Mini DSpark drafter (mtp.safetensors) into a g2e weight pack (.g2w), all BF16.

    python3 convert_dspark.py <model_dir> <out.g2w>

Layout for the engine (vllm.md section 5 for the semantics):
  alpha           bf16(softmax(fuse_weight, -1)) [4, 4]: rows = draft layers 0..2 + the Markov head, columns = taps
  L<l>.qkv_w      [q 3072 | k 1024 | v 1024] rows of the block-side projections        [5120, H]
  L<l>.kvc_w      [k_ctx 1024 | v_ctx 1024] rows of the context-side projections       [2048, H]
  L<l>.gu_w       [gate 9216 | up 9216]                                                [18432, H]
  L<l>.o_w, L<l>.down_w, L<l>.{input_ln, post_ln, dflare_norm, q_norm, k_norm, k_norm_ctx}
  norm, mask_embed
  mk.dflare_norm, mk.w1 [V, 256], rnn_proj split by input: mk.ws (state) [768, 256], mk.wp (w1[prev]) [768, 256],
  mk.wh (h) [768, H], mk.wf (f_rnn) [768, H]; mk.rb [768],
  mk.fold_w = unembed @ hidden_out.W [V, 256] and mk.fold_b = unembed @ hidden_out.b [V] (fp64, rounded once)
The target's unembed is read from the model's safetensors for the folds only (the engine uses the target pack's).
Same file layout as convert_qwen35.py.
"""
import json, os, struct, sys
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20


def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return (r >> np.uint32(16)).astype(np.uint16)


def f32(b):
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


class ST:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.h.pop("__metadata__", None)
        self.base = 8 + n

    def get(self, name):
        info = self.h[name]
        a, b = info["data_offsets"]
        assert info["dtype"] == "BF16", (name, info["dtype"])
        self.f.seek(self.base + a)
        return np.frombuffer(self.f.read(b - a), np.uint16).reshape(info["shape"])


def main():
    src, out = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json")))["mtp_config"]
    m = ST(os.path.join(src, "mtp.safetensors"))
    t = ST(os.path.join(src, "model-00001-of-00001.safetensors"))
    tensors = []
    fw = f32(m.get("fuse_weight")).astype(np.float64)
    a = np.exp(fw - fw.max(-1, keepdims=True))
    tensors.append(("alpha", bf16_bits(a / a.sum(-1, keepdims=True))))
    for l in range(cfg["num_hidden_layers"]):
        p = f"layers.{l}."
        g = lambda n: m.get(p + n)
        tensors.append((f"L{l}.qkv_w", np.concatenate([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"),
                                                         g("self_attn.v_proj.weight")])))
        tensors.append((f"L{l}.kvc_w", np.concatenate([g("self_attn.k_proj_ctx.weight"), g("self_attn.v_proj_ctx.weight")])))
        tensors.append((f"L{l}.gu_w", np.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])))
        tensors.append((f"L{l}.o_w", g("self_attn.o_proj.weight")))
        tensors.append((f"L{l}.down_w", g("mlp.down_proj.weight")))
        for short, name in [("input_ln", "input_layernorm.weight"), ("post_ln", "post_attention_layernorm.weight"),
                            ("dflare_norm", "dflare_norm.weight"), ("q_norm", "self_attn.q_norm.weight"),
                            ("k_norm", "self_attn.k_norm.weight"), ("k_norm_ctx", "self_attn.k_norm_ctx.weight")]:
            tensors.append((f"L{l}.{short}", g(name)))
    tensors.append(("norm", m.get("norm.weight")))
    tensors.append(("mask_embed", m.get("mask_embed")))
    tensors.append(("mk.dflare_norm", m.get("markov_head.dflare_norm.weight")))
    tensors.append(("mk.w1", m.get("markov_head.markov_w1.weight")))
    rp = m.get("markov_head.rnn_proj.weight")
    H = cfg["hidden_size"]
    tensors.append(("mk.ws", np.ascontiguousarray(rp[:, :256])))                  # state columns
    tensors.append(("mk.wp", np.ascontiguousarray(rp[:, 256:512])))               # w1[prev] columns
    tensors.append(("mk.wh", np.ascontiguousarray(rp[:, 512:512 + H])))           # h columns
    tensors.append(("mk.wf", np.ascontiguousarray(rp[:, 512 + H:512 + 2 * H])))   # f_rnn columns
    tensors.append(("mk.rb", m.get("markov_head.rnn_proj.bias")))
    un = f32(t.get("model.llm.unembed.weight")).astype(np.float64)
    hw = f32(m.get("markov_head.hidden_out.weight")).astype(np.float64)  # [H, 256]
    hb = f32(m.get("markov_head.hidden_out.bias")).astype(np.float64)    # [H]
    tensors.append(("mk.fold_w", bf16_bits(un @ hw)))
    tensors.append(("mk.fold_b", bf16_bits(un @ hb)))

    entries, offset = {}, 0
    with open(out, "wb") as o:
        o.seek(DATA_START)
        for name, arr in tensors:
            raw = np.ascontiguousarray(arr).tobytes()
            pad = (-offset) % ALIGN
            o.write(b"\0" * pad)
            offset += pad
            entries[name] = {"dtype": "bf16", "shape": list(arr.shape), "offset": offset, "bytes": len(raw)}
            o.write(raw)
            offset += len(raw)
        o.write(b"\0" * ((-offset) % ALIGN))
        header = {"format": "g2e-weights-1", "model": "inkling2-dspark", "layers": cfg["num_hidden_layers"],
                  "layer_kinds": ["draft"] * cfg["num_hidden_layers"], "config": cfg, "data_start": DATA_START,
                  "data_bytes": offset, "tensors": entries}
        hj = json.dumps(header).encode()
        o.seek(0)
        o.write(b"G2EW0001" + struct.pack("<Q", len(hj)) + hj)
    print(f"wrote {out}: {len(entries)} tensors, {offset / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
