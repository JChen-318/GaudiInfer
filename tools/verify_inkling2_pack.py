"""Independently decode a v6 native Inkling-2-Mini .g2w pack and verify all expert weights against checkpoint expert_inner.

    python3 tools/verify_inkling2_pack.py <model_dir> <pack.g2w> [--workers N]

Does not share any encoding code with convert_inkling2.py (only calls convert_inkling2.expert_inner as the checkpoint
reference). For every layer (all 24 layers), expert (128), and matrix (w13 and w2):
  - NVFP4 layers: decodes wq (u8) + bs (Gaudi2 fp8_143 = scale/8) + g (f32 = amax/den*512) using the device BF16
    value convention (E2M1 as fp8_143 s00eem00) * bs -> BF16 * 512, and checks 0 differences against expert_inner and g.
  - MXFP8 layers: decodes wq (u8 E4M3FN) + bs (Gaudi2 fp8_143 = 2^(s - s_max) <= 1) + g (f32 = 2^(s_max - 127))
    using the device TPC conversion rule (exp=15 converted as (code - 0x08) * 2) * bs -> BF16 * g, and checks
    0 differences against expert_inner.
Exits 1 on any difference or header/shape/dtype mismatch in any layer.
"""
import argparse
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_inkling2 as conv


def _gaudi2_fp8_143_lut():
    """Gaudi2 fp8_143 (bias 7, exp=15 reserved NaN/Inf, max normal 240.0) -> float32 LUT."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = (b >> 7) & 1
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 15:
            val = np.nan
        elif e == 0:
            val = (m / 8.0) * (2.0 ** -6)
        else:
            val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        lut[b] = -val if s else val
    return lut


FP8_143 = _gaudi2_fp8_143_lut()
# Precompute BF16-rounded value of (fp8_143(s00eem00)) for both nibbles of every uint8 byte.
_BYTES = np.arange(256, dtype=np.uint8)
_LO_F8 = ((_BYTES & np.uint8(0x08)) << np.uint8(4)) | ((_BYTES & np.uint8(0x07)) << np.uint8(2))
_HI_F8 = (_BYTES & np.uint8(0x80)) | ((_BYTES & np.uint8(0x70)) >> np.uint8(2))
NIB_F8_PAIR = np.stack([FP8_143[_LO_F8], FP8_143[_HI_F8]], axis=-1).astype(np.float32)  # [256, 2]
# Device TPC E4M3FN conversion rule: exponent field 15 -> fp8_143(code - 0x08) * 2.0, else fp8_143(code).
_IS_EXP15 = (_BYTES & np.uint8(0x78)) == np.uint8(0x78)
_ADJ_F8 = np.where(_IS_EXP15, (_BYTES.astype(np.int16) - 8).astype(np.uint8), _BYTES)
MX_E4M3_TPC = np.where(_IS_EXP15, FP8_143[_ADJ_F8] * np.float32(2.0), FP8_143[_ADJ_F8]).astype(np.float32)


def _round_bf16_f32(x):
    """Independent IEEE round-to-nearest-even float32 -> BF16 (returned as float32)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> np.uint32(16)) & np.uint32(1)
    r = (u + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return r.view(np.float32)


def _read_pack_header(path):
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if magic != b"G2EW0001":
            raise ValueError(f"bad pack magic: {magic!r}")
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    return hdr


def _read_tensor(path, data_start, entry, e_slice=None):
    dt_map = {"u8": np.uint8, "f8": np.uint8, "bf16": np.uint16, "f32": np.float32, "i32": np.int32}
    dt = dt_map[entry["dtype"]]
    shape = list(entry["shape"])
    off = data_start + entry["offset"]
    nbytes = entry["bytes"]
    if e_slice is not None:
        e0, ne = e_slice
        inner_bytes = int(np.prod(shape[1:])) * np.dtype(dt).itemsize
        off += e0 * inner_bytes
        nbytes = ne * inner_bytes
        shape[0] = ne
    with open(path, "rb") as fh:
        fh.seek(off)
        buf = fh.read(nbytes)
    if len(buf) != nbytes:
        raise IOError(f"short read: got {len(buf)}, expected {nbytes}")
    return np.frombuffer(buf, dtype=dt).reshape(shape)


def _deinterleave256_cols(a):
    """Undo 256-column interleave: stored[2i]=n_i, stored[2i+1]=n_{128+i} -> natural [E, K_dim, N]."""
    E_, K_, N_ = a.shape
    return a.reshape(E_, K_, N_ // 256, 128, 2).transpose(0, 1, 2, 4, 3).reshape(E_, K_, N_)


_PACK_PATH = None
_PACK_HDR = None


def _worker_init(model_dir, pack_path):
    global _PACK_PATH, _PACK_HDR
    conv._init(model_dir)
    _PACK_PATH = pack_path
    _PACK_HDR = _read_pack_header(pack_path)


def _verify_layer(layer):
    hdr = _PACK_HDR
    tensors = hdr["tensors"]
    data_start = hdr["data_start"]
    kind = hdr["layer_experts"][layer]
    E = hdr["config"]["n_routed_experts"]
    H = hdr["config"]["hidden_size"]
    F = hdr["config"]["intermediate_size"]
    chunk = 16

    layer_report = {"layer": layer, "kind": kind, "fail": False, "matrices": {}}

    for which, N, K in (("w13", 2 * F, H), ("w2", H, F)):
        prefix = f"L{layer}.{which}"
        if kind == "nvfp4":
            e_wq = tensors[prefix + ".wq"]
            e_bs = tensors[prefix + ".bs"]
            e_g = tensors[prefix + ".g"]
            assert e_wq["dtype"] == "u8" and e_wq["shape"] == [E, K // 2, N], f"{prefix}.wq bad meta"
            assert e_bs["dtype"] == "f8" and e_bs["shape"] == [E, K // 16, N], f"{prefix}.bs bad meta"
            assert e_g["dtype"] == "f32" and e_g["shape"] == [E], f"{prefix}.g bad meta"

            g_pack = _read_tensor(_PACK_PATH, data_start, e_g)
            val_diffs = 0
            g_diffs = 0
            nan_cnt = 0
            for e0 in range(0, E, chunk):
                ne = min(chunk, E - e0)
                ref_v, ref_g, ref_kind = conv.expert_inner(layer, which, e0, ne)
                assert ref_kind == "nvfp4", f"{prefix}: checkpoint kind {ref_kind} != nvfp4"

                wq = _read_tensor(_PACK_PATH, data_start, e_wq, (e0, ne))
                bs = _read_tensor(_PACK_PATH, data_start, e_bs, (e0, ne))
                if which == "w2":
                    wq = _deinterleave256_cols(wq)
                    bs = _deinterleave256_cols(bs)

                # [ne, K/2, N] -> [ne, N, K/2] -> [ne, N, K] via fp8_143(s00eem00)
                elem_div64 = NIB_F8_PAIR[wq.transpose(0, 2, 1)].reshape(ne, N, K)
                scale_div8 = FP8_143[bs.transpose(0, 2, 1)]  # [ne, N, K/16]
                # Device BF16 dequant product: bf16(elem * 2^-6) * bf16(scale / 8) -> BF16, then * 512 in F32
                prod_bf16 = _round_bf16_f32(
                    (elem_div64.reshape(ne, N, K // 16, 16) * scale_div8[..., None]).reshape(ne, N, K)
                )
                got_v = prod_bf16 * np.float32(512.0)

                nan_cnt += int(np.isnan(got_v).sum())
                val_diffs += int(np.sum(got_v != ref_v))
                g_rec = g_pack[e0:e0 + ne] / np.float32(512.0)
                g_diffs += int(np.sum(g_rec != ref_g))

            bad = (val_diffs > 0) or (g_diffs > 0) or (nan_cnt > 0)
            if bad:
                layer_report["fail"] = True
            layer_report["matrices"][which] = {
                "total": E * N * K,
                "diffs": val_diffs,
                "g_diffs": g_diffs,
                "nans": nan_cnt,
            }
        elif kind == "mxfp8":
            e_wq = tensors[prefix + ".wq"]
            e_bs = tensors[prefix + ".bs"]
            e_g = tensors[prefix + ".g"]
            assert e_wq["dtype"] == "u8" and e_wq["shape"] == [E, K, N], f"{prefix}.wq bad meta"
            assert e_bs["dtype"] == "f8" and e_bs["shape"] == [E, K // 32, N], f"{prefix}.bs bad meta"
            assert e_g["dtype"] == "f32" and e_g["shape"] == [E], f"{prefix}.g bad meta"

            g_pack = _read_tensor(_PACK_PATH, data_start, e_g)
            val_diffs = 0
            g_diffs = 0
            nan_cnt = 0
            bad_bs = 0
            if not np.all(g_pack == g_pack[0]) or not np.isfinite(g_pack[0]) or g_pack[0] <= 0.0:
                g_diffs = int(E)
            for e0 in range(0, E, chunk):
                ne = min(chunk, E - e0)
                ref_v, ref_g, ref_kind = conv.expert_inner(layer, which, e0, ne)
                assert ref_kind == "mxfp8", f"{prefix}: checkpoint kind {ref_kind} != mxfp8"
                assert np.all(ref_g == 1.0), f"{prefix}: unexpected mxfp8 g"

                wq = _read_tensor(_PACK_PATH, data_start, e_wq, (e0, ne))
                bs = _read_tensor(_PACK_PATH, data_start, e_bs, (e0, ne))
                nan_cnt += int(np.sum((wq & np.uint8(0x7F)) == np.uint8(0x7F)))
                # Valid stored MXFP8 block scales: positive normal fp8_143 powers of two in [2^-6, 1.0] (0x08..0x38, m=0)
                bad_bs += int(np.sum((bs < np.uint8(0x08)) | (bs > np.uint8(0x38)) | ((bs & np.uint8(0x87)) != 0)))
                if which == "w2":
                    wq = _deinterleave256_cols(wq)
                    bs = _deinterleave256_cols(bs)

                elem = MX_E4M3_TPC[wq.transpose(0, 2, 1)]  # [ne, N, K]
                scale = FP8_143[bs.transpose(0, 2, 1)]     # [ne, N, K/32]
                prod_bf16 = _round_bf16_f32(
                    (elem.reshape(ne, N, K // 32, 32) * scale[..., None]).reshape(ne, N, K)
                )
                got_v = prod_bf16 * g_pack[e0:e0 + ne, None, None]

                nan_cnt += int(np.isnan(got_v).sum())
                val_diffs += int(np.sum(got_v != ref_v))

            bad = (val_diffs > 0) or (g_diffs > 0) or (nan_cnt > 0) or (bad_bs > 0)
            if bad:
                layer_report["fail"] = True
            layer_report["matrices"][which] = {
                "total": E * N * K,
                "diffs": val_diffs,
                "g_diffs": g_diffs,
                "nans": nan_cnt,
                "bad_bs": bad_bs,
            }
        else:
            raise ValueError(f"L{layer}: unknown layer_experts kind {kind!r}")

    return layer_report


def main():
    ap = argparse.ArgumentParser(description="Verify v6 native Inkling-2-Mini pack against checkpoint expert_inner.")
    ap.add_argument("model", help="checkpoint directory")
    ap.add_argument("pack", help="v6 .g2w pack path")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    hdr = _read_pack_header(args.pack)
    if (hdr.get("pack_version") != 6 or hdr.get("experts") != "native" or
            hdr.get("nvfp4_bs") != "fp8_143_div8" or hdr.get("mxfp8_bs") != "fp8_143_le1"):
        print(
            f"ERROR: expected v6 native pack (nvfp4_bs=fp8_143_div8, mxfp8_bs=fp8_143_le1), got "
            f"pack_version={hdr.get('pack_version')!r}, experts={hdr.get('experts')!r}, "
            f"nvfp4_bs={hdr.get('nvfp4_bs')!r}, mxfp8_bs={hdr.get('mxfp8_bs')!r}",
            file=sys.stderr,
        )
        return 1

    L = hdr["layers"]
    le = hdr.get("layer_experts")
    if not isinstance(le, list) or len(le) != L:
        print(f"ERROR: invalid layer_experts header field: {le!r}", file=sys.stderr)
        return 1

    t0 = time.time()
    total_vals = 0
    total_diffs = 0
    total_g_diffs = 0
    any_fail = False

    with ProcessPoolExecutor(args.workers, initializer=_worker_init, initargs=(args.model, args.pack)) as ex:
        for rep in ex.map(_verify_layer, range(L)):
            l = rep["layer"]
            kind = rep["kind"]
            m13 = rep["matrices"]["w13"]
            m2 = rep["matrices"]["w2"]
            total_vals += m13["total"] + m2["total"]
            total_diffs += m13["diffs"] + m2["diffs"]
            total_g_diffs += m13["g_diffs"] + m2["g_diffs"]
            extra = ""
            if kind == "mxfp8":
                extra = f" bad_bs={m13['bad_bs'] + m2['bad_bs']}"
            print(
                f"L{l:02d} [{kind:5s}]: w13 diffs={m13['diffs']} g_diffs={m13['g_diffs']} nans={m13['nans']} | "
                f"w2 diffs={m2['diffs']} g_diffs={m2['g_diffs']} nans={m2['nans']}{extra}",
                flush=True,
            )
            if rep["fail"]:
                any_fail = True

    dt = time.time() - t0
    print(
        f"SUMMARY ({dt:.1f}s): ALL {L} layers total={total_vals} value_diffs={total_diffs} g_diffs={total_g_diffs}; "
        f"status={'FAIL' if any_fail else 'PASS'}",
        flush=True,
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
