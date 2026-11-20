"""Pack the vision encoder of a Qwen3.5/3.8 checkpoint ("model.visual.*", all BF16) into a g2e weight pack (.g2w).

    python3 convert_vision.py <model_dir> <out.g2w>

Names lose the "model.visual." prefix. patch_embed.proj.weight [E, 3, 2, P, P] becomes a linear weight [E, 3*2*P*P]
(the processor's patch vector order: channel, temporal copy, row, column). Same file layout as convert_dspine.py;
the header's "config" is the checkpoint's vision_config plus the text model's image/vision token ids.
"""
import json, os, struct, sys
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20


def main():
    src, out = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json")))
    vcfg = dict(cfg["vision_config"])
    for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        vcfg[k] = cfg[k]
    proc = json.load(open(os.path.join(src, "processor_config.json")))["image_processor"]
    vcfg["image_processor"] = proc
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    names = sorted(k for k in index if k.startswith("model.visual."))
    files = {}

    def get(name):
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

    tensors = []
    for name in names:
        short = name.removeprefix("model.visual.")
        w = get(name)
        if short == "patch_embed.proj.weight":
            w = w.reshape(w.shape[0], -1)
        tensors.append((short, w))

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
                  "layers": vcfg["depth"], "layer_kinds": ["vision"] * vcfg["depth"], "config": vcfg,
                  "data_start": DATA_START, "data_bytes": offset, "tensors": entries}
        hj = json.dumps(header).encode()
        o.seek(0)
        o.write(b"G2EW0001" + struct.pack("<Q", len(hj)) + hj)
    print(f"wrote {out}: {len(entries)} tensors, {offset / 2**30:.2f} GiB")
    for name in sorted(entries):
        if not name.startswith("blocks."):
            print(f"  {name:32s} {entries[name]['shape']}")


if __name__ == "__main__":
    main()
