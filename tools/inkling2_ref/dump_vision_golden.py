# Golden for the Inkling-2 vision tower: HF Inkling2VisionModel (bf16, CPU) with the checkpoint's vision weights.
import json, os, struct, sys
import numpy as np, torch
from PIL import Image, ImageDraw
from transformers import AutoConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel
import math

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cfg = AutoConfig.from_pretrained(src)
vc = cfg.vision_config
vc._attn_implementation = "sdpa"
model = Inkling2VisionModel(vc).to(torch.bfloat16).eval()
index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
fn = os.path.join(src, index["model.visual.proj_in.weight"])
f = open(fn, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
def get(name):
    info = hdr["model.visual." + name]; a, b = info["data_offsets"]; f.seek(base + a)
    return torch.from_numpy(np.frombuffer(f.read(b - a), np.int16).copy()).view(torch.bfloat16).reshape(info["shape"])
sd = {"embeddings.patch_embedding.weight": get("backbone.embeddings.patch_embedding.weight"),
      "embeddings.position_embedding.weight": get("backbone.embeddings.position_embedding"),
      "embeddings.class_embedding": get("backbone.embeddings.class_embedding"),
      "embeddings.cls_position_embedding": get("backbone.embeddings.cls_position_embedding"),
      "pre_layernorm.weight": get("backbone.ln_pre.weight"), "pre_layernorm.bias": get("backbone.ln_pre.bias"),
      "merger.linear_1.weight": get("proj_in.weight"), "merger.linear_2.weight": get("proj_out.weight")}
for L in range(vc.num_hidden_layers):
    p, q = f"backbone.layers.{L}.", f"layers.{L}."
    for a, b in (("ln_1", "layernorm_before"), ("ln_2", "layernorm_after")):
        sd[q + b + ".weight"] = get(p + a + ".weight"); sd[q + b + ".bias"] = get(p + a + ".bias")
    sd[q + "lambda_1"] = get(p + "ls_1.gamma"); sd[q + "lambda_2"] = get(p + "ls_2.gamma")
    sd[q + "mlp.fc1.weight"] = get(p + "mlp.c_fc.weight"); sd[q + "mlp.fc1.bias"] = get(p + "mlp.c_fc.bias")
    sd[q + "mlp.fc2.weight"] = get(p + "mlp.c_proj.weight"); sd[q + "mlp.fc2.bias"] = get(p + "mlp.c_proj.bias")
    for x in ("q", "k", "v", "out"):
        sd[q + f"self_attn.{x}_proj.weight"] = get(p + f"self_attn.{x}_proj.weight")
        sd[q + f"self_attn.{x}_proj.bias"] = get(p + f"self_attn.{x}_proj.bias")
missing, unexpected = model.load_state_dict(sd, strict=False)
missing = [m for m in missing if "inv_freq" not in m]
assert not missing and not unexpected, (missing, unexpected)

def make(w, h, seed):
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (240, 235, 220)); d = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, w - 10)), int(rng.integers(0, h - 10))
        d.rectangle([x0, y0, x0 + int(rng.integers(8, w // 3)), y0 + int(rng.integers(8, h // 3))],
                    fill=tuple(int(c) for c in rng.integers(0, 255, 3)))
    d.ellipse([w // 4, h // 4, w // 2, h // 2], fill=(200, 30, 30)); d.text((10, 10), "Gaudi 2", fill=(0, 0, 0))
    return im

def preprocess(im, S=40):  # Inkling2ImageProcessor: pad to the 40-pixel grid with -1 (0-255 scale), rescale, normalize
    a = np.asarray(im.convert("RGB"), np.float32)
    H, W = a.shape[:2]; rows, cols = math.ceil(H / S), math.ceil(W / S)
    pad = np.full((rows * S, cols * S, 3), -1.0, np.float32); pad[:H, :W] = a
    x = (pad * np.float32(1 / 255) - np.float32(0.5)) / np.float32(0.5)
    x = x.reshape(rows, S, cols, S, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, S, S, 3)
    return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor([[1, rows, cols]])
for name, (w, h) in {"small": (130, 90), "mid": (640, 400)}.items():
    im = make(w, h, len(name)); im.save(f"{out}/{name}.png")
    pv, thw = preprocess(im)
    with torch.no_grad():
        o = model(pixel_values=pv, image_grid_thw=thw)
    feat = o.pooler_output.float().numpy()
    m32 = model.float()
    with torch.no_grad():
        f32 = m32(pixel_values=pv.float(), image_grid_thw=thw).pooler_output.numpy()
    model.to(torch.bfloat16)
    f32.astype(np.float32).tofile(f"{out}/{name}.feat32.f32")
    print(name, "hf bf16 vs fp32 rel err %.4f" % (np.linalg.norm(feat - f32) / np.linalg.norm(f32)))
    pv.float().numpy().astype(np.float32).tofile(f"{out}/{name}.pix.f32")
    feat.astype(np.float32).tofile(f"{out}/{name}.feat.f32")
    json.dump({"w": w, "h": h, "grid": thw.tolist()[0], "feat_shape": list(feat.shape), "pix_shape": list(pv.shape)},
              open(f"{out}/{name}.json", "w"))
    print(name, thw.tolist(), pv.shape, feat.shape, float(np.abs(feat).mean()))
