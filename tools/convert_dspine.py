"""Convert the Qwen3.8-27B-DSpine-b16 drafter checkpoint into a g2e weight pack (.g2w), all BF16.

    python3 convert_dspine.py <draft_dir> <out.g2w>

Layout changes for the engine:
  fc.k            fc.weight[:, k*H:(k+1)*H] for k in 0..4 (one [H, H] matrix per captured target layer)
  bridge.packed   [gate; predecessor_gate; prev] stacked -> [3R, C]
  bridge.read.i   early_read_weights[i] for i < 4, read_weight for i = 4   ([C, H])
  bridge.write.i  feature_write_weights[i]                                 ([H, R])
everything else keeps its HF name without the "model." prefix. Same file layout as convert_qwen35.py.
"""
import json, os, struct, sys
import numpy as np

ALIGN = 4096
DATA_START = 4 << 20


def main():
    src, out = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(src, "config.json")))
    f = open(os.path.join(src, "model.safetensors"), "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    base = 8 + n

    def get(name):
        info = hdr[name]
        a, b = info["data_offsets"]
        f.seek(base + a)
        dt = {"BF16": np.uint16, "I64": np.int64, "F32": np.float32}[info["dtype"]]
        return np.frombuffer(f.read(b - a), dt).reshape(info["shape"])

    H = cfg["hidden_size"]
    tensors = []
    for name in sorted(hdr):
        short = name.removeprefix("model.")
        if hdr[name]["dtype"] == "I64":
            v = int(get(name).reshape(-1)[0])
            if v != 1:
                raise SystemExit(f"unsupported {short} = {v}")
            continue
        if short == "fc.weight":
            w = get(name)
            for k in range(w.shape[1] // H):
                tensors.append((f"fc.{k}", np.ascontiguousarray(w[:, k * H:(k + 1) * H])))
        elif short == "selected_prefix_bridge.early_read_weights":
            w = get(name)
            for i in range(w.shape[0]):
                tensors.append((f"bridge.read.{i}", np.ascontiguousarray(w[i])))
        elif short == "selected_prefix_bridge.read_weight":
            tensors.append((f"bridge.read.{cfg['num_hidden_layers'] - 1}", get(name)))
        elif short == "selected_prefix_bridge.feature_write_weights":
            w = get(name)
            for i in range(w.shape[0]):
                tensors.append((f"bridge.write.{i}", np.ascontiguousarray(w[i])))
        elif short in ("selected_prefix_bridge.gate.weight", "selected_prefix_bridge.predecessor_gate.weight",
                       "selected_prefix_bridge.prev.weight"):
            continue  # stacked below
        else:
            tensors.append((short.replace("selected_prefix_bridge.", "bridge."), get(name)))
    packed = np.concatenate([get("selected_prefix_bridge.gate.weight"), get("selected_prefix_bridge.predecessor_gate.weight"),
                             get("selected_prefix_bridge.prev.weight")])
    tensors.append(("bridge.packed", packed))
    out_w = get("selected_prefix_bridge.out.weight")
    if np.any(out_w & 0x7FFF):
        raise SystemExit("terminal-disabled checkpoint has a nonzero bridge.out projection")

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
        header = {"format": "g2e-weights-1", "model": os.path.basename(os.path.normpath(src)), "layers": cfg["num_hidden_layers"],
                  "layer_kinds": ["draft"] * cfg["num_hidden_layers"], "config": cfg, "data_start": DATA_START,
                  "data_bytes": offset, "tensors": entries}
        hj = json.dumps(header).encode()
        o.seek(0)
        o.write(b"G2EW0001" + struct.pack("<Q", len(hj)) + hj)
    print(f"wrote {out}: {len(entries)} tensors, {offset / 2**30:.2f} GiB")
    for name in sorted(entries):
        if not name.startswith("layers."):
            print(f"  {name:32s} {entries[name]['shape']}")


if __name__ == "__main__":
    main()
