"""Pack the vision tower of an Inkling-2 checkpoint ("model.visual.*", all BF16) into a g2e weight pack (.g2w).

    python3 convert_inkling2_vision.py <model_dir> <out.g2w>

Tensors (BF16): patch_embed [E, P*P*3] (patch vector order: row, column, channel), pos_embed [G*G, E],
cls [E] (class_embedding + cls_position_embedding, as the model adds them), ln_pre.{weight,bias}, per layer L:
L.ln1.{weight,bias}, L.qkv.{weight [3E, E], bias [3E]} (q | k | v), L.out.{weight,bias}, L.ls1, L.ln2.{weight,bias},
L.fc1.{weight,bias}, L.fc2.{weight,bias}, L.ls2; proj_in [D, 4E], proj_out [text hidden, D] (no biases).
The header's "config" is the checkpoint's vision_config plus image_token_id and the image processor settings.
"""
import json, os, struct, sys
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20


def bf16_to_f32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(a):
    a = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((a + 0x7FFF + ((a >> 16) & 1)) >> 16).astype(np.uint16)


def main():
    src, out = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json")))
    vcfg = dict(cfg["vision_config"])
    vcfg["image_token_id"] = cfg.get("image_token_id", 200054)
    vcfg.setdefault("rope_theta", 10000.0)
    vcfg["image_processor"] = json.load(open(os.path.join(src, "processor_config.json")))["image_processor"]
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    files = {}

    def get(name):
        name = "model.visual." + name
        fn = os.path.join(src, index[name])
        if fn not in files:
            f = open(fn, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            files[fn] = (f, json.loads(f.read(n)), 8 + n)
        f, hdr, base = files[fn]
        info = hdr[name]
        if info["dtype"] != "BF16":
            raise SystemExit(f"{name}: expected BF16, got {info['dtype']}")
        a, b = info["data_offsets"]
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), np.uint16).reshape(info["shape"])

    bb = "backbone."
    tensors = [("patch_embed", get(bb + "embeddings.patch_embedding.weight")),
               ("pos_embed", get(bb + "embeddings.position_embedding")),
               ("cls", f32_to_bf16(bf16_to_f32(get(bb + "embeddings.class_embedding")) +
                                   bf16_to_f32(get(bb + "embeddings.cls_position_embedding")))),
               ("ln_pre.weight", get(bb + "ln_pre.weight")), ("ln_pre.bias", get(bb + "ln_pre.bias"))]
    depth = vcfg["num_hidden_layers"]
    for L in range(depth):
        p = f"{bb}layers.{L}."
        a = p + "self_attn."
        tensors += [
            (f"{L}.ln1.weight", get(p + "ln_1.weight")), (f"{L}.ln1.bias", get(p + "ln_1.bias")),
            (f"{L}.qkv.weight", np.concatenate([get(a + f"{x}_proj.weight") for x in "qkv"])),
            (f"{L}.qkv.bias", np.concatenate([get(a + f"{x}_proj.bias") for x in "qkv"])),
            (f"{L}.out.weight", get(a + "out_proj.weight")), (f"{L}.out.bias", get(a + "out_proj.bias")),
            (f"{L}.ls1", get(p + "ls_1.gamma")),
            (f"{L}.ln2.weight", get(p + "ln_2.weight")), (f"{L}.ln2.bias", get(p + "ln_2.bias")),
            (f"{L}.fc1.weight", get(p + "mlp.c_fc.weight")), (f"{L}.fc1.bias", get(p + "mlp.c_fc.bias")),
            (f"{L}.fc2.weight", get(p + "mlp.c_proj.weight")), (f"{L}.fc2.bias", get(p + "mlp.c_proj.bias")),
            (f"{L}.ls2", get(p + "ls_2.gamma")),
        ]
    tensors += [("proj_in", get("proj_in.weight")), ("proj_out", get("proj_out.weight"))]

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
        header = {"format": "g2e-weights-1", "model": os.path.basename(os.path.normpath(src)) + "-vision",
                  "layers": depth, "layer_kinds": ["vision"] * depth, "config": vcfg,
                  "data_start": DATA_START, "data_bytes": offset, "tensors": entries}
        hj = json.dumps(header).encode()
        o.seek(0)
        o.write(b"G2EW0001" + struct.pack("<Q", len(hj)) + hj)
    print(f"wrote {out}: {len(entries)} tensors, {offset / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
