"""CPU reference of the Inkling-2 DSpark drafter (run with TORCH_DEVICE_BACKEND_AUTOLOAD=0: CPU only) (mtp.safetensors), following vLLM's
implementation, and an offline acceptance measurement on a target dump from inkling2_test --dspark-dump:
for every generated position p, draft the 7 tokens after the anchor (the target's token at p + 1) from the target's
taps at positions <= p and count the leading drafts equal to the target's greedy tokens."""
import os
import struct
import sys

import numpy as np
import torch
from safetensors import safe_open

torch.set_num_threads(8)
MODEL = os.environ.get('INKLING2_MODEL_DIR', '/models/Inkling-2-Mini')  # the checkpoint directory
H, NH, KV, D, F = 2304, 24, 8, 128, 9216
EPS = 1e-6
THETA = 1e6
VOCAB = 200058


def bf(t):
    return t.to(torch.bfloat16).to(torch.float32)


def rms(x, w=None):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y * w if w is not None else y


def load():
    m = {}
    with safe_open(f"{MODEL}/mtp.safetensors", "pt") as f:
        for k in f.keys():
            m[k] = f.get_tensor(k).to(torch.float32)
    with safe_open(f"{MODEL}/model-00001-of-00001.safetensors", "pt") as f:
        for k in ["model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.unembed.weight"]:
            m[k.split(".")[-2]] = f.get_tensor(k)  # BF16 (big)
    return m


def rope(x, pos):
    # neox RoPE (rotate halves [0:64] / [64:128]); x [n, heads, 128], pos [n]; output rounded to BF16
    inv = THETA ** (-(torch.arange(64, dtype=torch.float64) * 2 / 128))
    ang = pos.to(torch.float64)[:, None] * inv[None, :]
    c = torch.cos(ang).to(torch.float32)[:, None, :]
    s = torch.sin(ang).to(torch.float32)[:, None, :]
    x1, x2 = x[..., :64], x[..., 64:]
    return bf(torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1))


class Drafter:
    def __init__(self, m):
        self.m = m
        a = torch.softmax(m["fuse_weight"], -1)
        self.alpha = bf(a)  # [4 rows (layers 0..2, markov), 4 taps]
        W = m["markov_head.rnn_proj.weight"]  # [768, 5120]
        self.Wsx = W[:, :512]
        self.Whf = W[:, 512:]
        self.rb = m["markov_head.rnn_proj.bias"]
        un = m["unembed"].to(torch.float64)
        self.foldW = bf((un @ m["markov_head.hidden_out.weight"].to(torch.float64)).to(torch.float32))  # [V, 256]
        self.foldb = bf((un @ m["markov_head.hidden_out.bias"].to(torch.float64)).to(torch.float32))
        self.un = m["unembed"]  # bf16 [V, H]
        self.w1 = m["markov_head.markov_w1.weight"]
        self.ctxK = [None] * 3
        self.ctxV = [None] * 3

    def ctx(self, f, r):
        # f [n, 4, H] bf16 values (f32); row r of the fusion
        s = (self.alpha[r][None, :, None] * f).sum(1)
        w = self.m[f"layers.{r}.dflare_norm.weight"] if r < 3 else self.m["markov_head.dflare_norm.weight"]
        return bf(rms(s) * w)

    def context_kv(self, f):
        n = f.shape[0]
        pos = torch.arange(n)
        for l in range(3):
            p = f"layers.{l}.self_attn."
            c = self.ctx(f, l)
            k = bf(c @ self.m[p + "k_proj_ctx.weight"].T).view(n, KV, D)
            k = bf(rms(k) * self.m[p + "k_norm_ctx.weight"])
            self.ctxK[l] = rope(k, pos)
            self.ctxV[l] = bf(c @ self.m[p + "v_proj_ctx.weight"].T).view(n, KV, D)

    def draft(self, f_last, anchor, apos):
        m = self.m
        x0 = bf(rms(m["embed"][anchor].to(torch.float32), m["embed_norm"].to(torch.float32)))
        x = torch.stack([x0] + [m["mask_embed"]] * 7)  # [8, H]
        pos = torch.arange(8) + apos
        for l in range(3):
            p = f"layers.{l}."
            a = bf(rms(x, m[p + "input_layernorm.weight"]))
            q = bf(a @ m[p + "self_attn.q_proj.weight"].T).view(8, NH, D)
            q = rope(bf(rms(q) * m[p + "self_attn.q_norm.weight"]), pos)
            k = bf(a @ m[p + "self_attn.k_proj.weight"].T).view(8, KV, D)
            k = rope(bf(rms(k) * m[p + "self_attn.k_norm.weight"]), pos)
            v = bf(a @ m[p + "self_attn.v_proj.weight"].T).view(8, KV, D)
            lo = 0 if l == 1 else max(0, apos - 1024)
            K = torch.cat([self.ctxK[l][lo:apos], k], 0)  # [S, KV, D]
            V = torch.cat([self.ctxV[l][lo:apos], v], 0)
            qh = q.view(8, KV, NH // KV, D)
            sc = torch.einsum("tkgd,skd->tkgs", qh, K) / np.sqrt(D)
            pr = torch.softmax(sc, -1)
            o = bf(torch.einsum("tkgs,skd->tkgd", pr, V).reshape(8, NH * D))
            x = bf(x + bf(o @ m[p + "self_attn.o_proj.weight"].T))
            b = bf(rms(x, m[p + "post_attention_layernorm.weight"]))
            g = bf(b @ m[p + "mlp.gate_proj.weight"].T)
            u = bf(b @ m[p + "mlp.up_proj.weight"].T)
            h = bf(g * torch.sigmoid(g) * u)
            x = bf(x + bf(h @ m[p + "mlp.down_proj.weight"].T))
        hf = bf(rms(x, m["norm.weight"]))
        base = bf((hf[1:].to(torch.bfloat16) @ self.un.T).to(torch.float32))  # [7, V]
        frnn = self.ctx(f_last[None], 3)[0]
        C = self.rb[None, :] + torch.cat([hf[1:], frnn[None].expand(7, H)], 1) @ self.Whf.T  # [7, 768]
        state = torch.zeros(256)
        prev = anchor
        out = []
        for i in range(7):
            raw = C[i] + torch.cat([state, self.w1[prev]]) @ self.Wsx.T
            g, c, o = bf(raw[:256]), bf(raw[256:512]), bf(raw[512:])
            g = bf(torch.sigmoid(g))
            c = bf(torch.tanh(c))
            state = bf(bf(g * state) + bf(bf(1 - g) * c))
            lat = bf(torch.tanh(o))
            bias = bf(self.foldb + lat @ self.foldW.T)
            lg = bf(30 * torch.tanh(bf(base[i] + bias) / 30))[:VOCAB]
            prev = int(torch.argmax(lg))
            out.append(prev)
        return out


def main(path, n_prompt, step):
    with open(path, "rb") as fh:
        n, hid = struct.unpack("<ii", fh.read(8))
        toks = np.frombuffer(fh.read(4 * n), dtype=np.int32)
        nxt = np.frombuffer(fh.read(4 * n), dtype=np.int32)
        feats = np.frombuffer(fh.read(2 * n * 4 * hid), dtype=np.uint16).reshape(n, 4, hid)
    f = torch.from_numpy((feats.astype(np.uint32) << 16).view(np.float32).copy())
    m = load()
    d = Drafter(m)
    d.context_kv(f)
    acc, rounds = [], 0
    for p in range(n_prompt - 1, n - 9, step):
        anchor = int(nxt[p])  # = toks[p + 1] in the generated region
        dr = d.draft(f[p], anchor, p + 1)
        tgt = [int(t) for t in toks[p + 2:p + 9]]
        k = 0
        while k < 7 and dr[k] == tgt[k]:
            k += 1
        acc.append(k)
        rounds += 1
        if rounds % 20 == 0:
            print(f"  {rounds} rounds: mean accepted drafts {np.mean(acc):.2f} (+1 bonus)", flush=True)
    acc = np.array(acc)
    print(f"{path}: {rounds} rounds, accepted drafts per round {acc.mean():.2f} (tokens per round {acc.mean() + 1:.2f}); "
          f"histogram {np.bincount(acc, minlength=8).tolist()}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1)
