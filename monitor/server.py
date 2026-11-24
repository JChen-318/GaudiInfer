#!/usr/bin/env python3
"""GaudiInfer Studio & Monitor WebUI server (pure Python 3 stdlib, zero pip deps).

Unified workbench + real-time hardware telemetry + one-click engine switcher for:
  - llm       : Qwen3.8-27B FP8 + DSpine-b16 + Vision (:8001)
  - qwen21    : Qwen-Image-2.1-Turbo T2I & 1-4 Ref Edit (:8098)
  - h3_image  : MiniMax-H3-Image 2K/4K 16MP T2I & 1-16 Ref Edit (:8099)
  - h3        : MiniMax-H3 Video + Stereo Audio T2VA / FL2VA (:8097)
"""
import base64
import collections
import http.client
import json
import mimetypes
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_DIR = pathlib.Path(os.environ.get("G2E_REPO_DIR", "/home/a/gaudi2-engine"))
BUILD_DIR = REPO_DIR / "build"
DATA_DIR = pathlib.Path(os.environ.get("G2E_DATA_DIR", "/home/a/g2e-data"))
PACKS_DIR = pathlib.Path(os.environ.get("G2E_PACKS_DIR", "/mnt/sas/g2e"))
MODELS_DIR = pathlib.Path(os.environ.get("G2E_MODELS_DIR", "/mnt/sas/models"))
GALLERY_DIR = pathlib.Path(os.environ.get("G2E_STUDIO_OUTPUT_DIR", "/mnt/sas/outputs/g2e_studio"))
LOG_DIR = pathlib.Path(os.environ.get("G2E_STUDIO_LOG_DIR", "/tmp/g2e_studio_logs"))
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

CONTAINER_NAME = os.environ.get("G2E_CONTAINER_NAME", "gaudiinfer")
MONITOR_HOST = os.environ.get("G2E_MONITOR_HOST", "0.0.0.0")
MONITOR_PORT = int(os.environ.get("G2E_MONITOR_PORT", "8002"))
DOCKER_SOCK = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")
HL_SMI = os.environ.get("HL_SMI_BIN", "/usr/bin/hl-smi")
POLL_SEC = float(os.environ.get("G2E_MONITOR_POLL_SEC", "1.0"))

ENGINE_SPECS = {
    "llm": {
        "id": "llm",
        "name": "Qwen3.8-27B",
        "short": "27B 对话",
        "subtitle": "FP8 + DSpine-b16 + Vision · 256K 上下文",
        "kind": "chat",
        "port": int(os.environ.get("G2E_LLM_PORT", "8001")),
        "binary": "g2e_server",
        "hbm_est": "~33 GiB + 动态 KV",
        "speed_hint": "~180–210 tok/s · 看图 + 思考",
    },
    "qwen21": {
        "id": "qwen21",
        "name": "Qwen-Image-2.1-Turbo",
        "short": "Qwen 2.1 生图",
        "subtitle": "8 步文生图 / 1–4 图编辑 · 1024² 极速",
        "kind": "image",
        "port": int(os.environ.get("G2E_QWEN21_PORT", "8098")),
        "binary": "g2e_qwen21_server",
        "hbm_est": "22.6 GiB",
        "speed_hint": "1024² 约 2.3s · 编辑约 4.2s",
    },
    "h3_image": {
        "id": "h3_image",
        "name": "MiniMax-H3-Image",
        "short": "H3 图像",
        "subtitle": "33.1B 6 步 PDD · 2K / 4K 16MP & 1–16 图编辑",
        "kind": "image",
        "port": int(os.environ.get("G2E_H3_IMAGE_PORT", "8099")),
        "binary": "g2e_h3_image_server",
        "hbm_est": "56.9 GiB",
        "speed_hint": "2K 约 4.6s · 4K 16MP 约 22s",
    },
    "h3": {
        "id": "h3",
        "name": "MiniMax-H3 Video",
        "short": "H3 视频",
        "subtitle": "文生/首尾帧音视频 · 4步/8步 Turbo LoRA",
        "kind": "video",
        "port": int(os.environ.get("G2E_H3_PORT", "8097")),
        "binary": "g2e_h3_server",
        "hbm_est": "~62 GiB",
        "speed_hint": "480p/768p 5–15s 同步双声道",
    },
}

ENGINE_ORDER = ["llm", "qwen21", "h3_image", "h3"]

REQ_RE = re.compile(
    r"\[g2e\]\s+prompt\s+(\d+)\s+\(cached\s+(\d+),\s+([\d.]+)%;\s+all requests\s+([\d.]+)%\)\s+"
    r"([\d.]+)\s+ms\s+\|\s+generated\s+(\d+)\s+in\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)\s+\|\s+"
    r"rounds\s+(\d+),\s+([\d.]+)\s+tokens/round\s+\|\s+(\S+)"
)


def pick_existing(*candidates) -> str:
    for c in candidates:
        if c and os.path.exists(str(c)):
            return str(c)
    return str(candidates[-1])


def discover_h3_loras():
    loras = []
    candidates = [
        ("v1.2_8step", "8步 Turbo v1.2（推荐·画质最佳）", 8,
         pick_existing("/dev/shm/h3/minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors",
                       MODELS_DIR / "minimax-h3/minimax_h3_fl2v_turbo_8step_v1.2_768p_bf16.safetensors")),
        ("v1.0_4step", "4步 Turbo v1.0（极速出片）", 4,
         pick_existing("/dev/shm/h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
                       MODELS_DIR / "minimax-h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors")),
    ]
    for lid, label, steps, path in candidates:
        if os.path.exists(path):
            loras.append({"id": lid, "label": label, "steps": steps, "path": path})
    loras.append({"id": "none", "label": "无 LoRA（Base 50 步原版）", "steps": 50, "path": ""})
    return loras


def build_engine_command(engine_id: str):
    """Build (argv, env) for launching one of the 4 native C++ Gaudi2 servers."""
    env = os.environ.copy()
    # Always use a user-writable HABANA_LOGS dir (avoids root-owned logs left by Docker containers)
    habana_logs = LOG_DIR / "habana_logs"
    habana_logs.mkdir(parents=True, exist_ok=True)
    env["HABANA_LOGS"] = str(habana_logs)
    recipe_dir = DATA_DIR / "recipes"
    if not (recipe_dir.is_dir() and os.access(recipe_dir, os.W_OK)):
        recipe_dir = LOG_DIR / "recipes"
        recipe_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("G2E_RECIPE_CACHE", str(recipe_dir))
    scal_cfg = str(REPO_DIR / "tools/scal-g2e-direct.json")
    if os.path.exists(scal_cfg):
        env["SCAL_CONFIG_FILE_PATH"] = scal_cfg
        env["SCAL_CFG_FILE_PATH"] = scal_cfg

    spec = ENGINE_SPECS[engine_id]
    bin_path = str(BUILD_DIR / spec["binary"])
    port = str(spec["port"])

    if engine_id == "llm":
        shm = pathlib.Path("/dev/shm/g2e")
        pdir = shm if all((shm / f).is_file() for f in ("qwen3.8-27b.g2w", "dspine-b16.g2w", "vision.g2w")) else PACKS_DIR
        cmd = [
            bin_path,
            "--model-dir", str(MODELS_DIR / "qwen3.8-27b-fp8-mtp"),
            "--target-pack", str(pdir / "qwen3.8-27b.g2w"),
            "--draft-pack", str(pdir / "dspine-b16.g2w"),
            "--vision-pack", str(pdir / "vision.g2w"),
            "--served-model-name", "qwen3.8-27b",
            "--host", "0.0.0.0",
            "--port", port,
            "--max-model-len", os.environ.get("G2E_MAX_MODEL_LEN", "262144"),
            "--slots", os.environ.get("G2E_SLOTS", "8"),
            "--kv-cache-gib", "0",
            "--reasoning-parser", "qwen3",
        ]
    elif engine_id == "qwen21":
        cmd = [
            bin_path,
            "--host", "0.0.0.0",
            "--port", port,
            "--served-model-name", "Qwen/Qwen-Image-2.1-Turbo",
            "--tokenizer", str(MODELS_DIR / "qwen-image-2.1-turbo/processor/tokenizer.json"),
            "--text-enc", str(PACKS_DIR / "qwen21_text_encoder_bf16.g2w"),
            "--dit", str(PACKS_DIR / "qwen21_dit_fp8.g2w"),
            "--vae", str(PACKS_DIR / "qwen21_vae_bf16.g2w"),
        ]
    elif engine_id == "h3_image":
        cmd = [
            bin_path,
            "--host", "0.0.0.0",
            "--port", port,
            "--served-model-name", "MiniMaxAI/MiniMax-H3-Image",
            "--tokenizer", str(MODELS_DIR / "minimax-h3/FL2VA/tokenizer/tokenizer.json"),
            "--text-enc", pick_existing("/dev/shm/h3/h3_text_encoder_fp8.g2w", PACKS_DIR / "h3_text_encoder_fp8.g2w"),
            "--dit", pick_existing("/dev/shm/h3/h3_image_dit_fp8.g2w", PACKS_DIR / "h3_image_dit_fp8.g2w"),
            "--vae", pick_existing("/dev/shm/h3/h3_image_vae_fp8.g2w", PACKS_DIR / "h3_image_vae_fp8.g2w"),
        ]
    elif engine_id == "h3":
        loras = discover_h3_loras()
        default_lora = loras[0]["path"] if loras else ""
        ffmpeg_bin = pick_existing("/home/a/h3-deploy/media-bin/ffmpeg", shutil.which("ffmpeg") or "ffmpeg")
        cmd = [
            bin_path,
            "--host", "0.0.0.0",
            "--port", port,
            "--served-model-name", "MiniMaxAI/MiniMax-H3",
            "--tokenizer", str(MODELS_DIR / "minimax-h3/FL2VA/tokenizer/tokenizer.json"),
            "--text-enc", pick_existing("/dev/shm/h3/h3_text_encoder_fp8.g2w", PACKS_DIR / "h3_text_encoder_fp8.g2w"),
            "--dit", pick_existing("/dev/shm/h3/h3_dit_base_fp8.g2w", PACKS_DIR / "h3_dit_base_fp8.g2w"),
            "--vvae", pick_existing("/dev/shm/h3/h3_video_vae_dec_fp8.g2w", PACKS_DIR / "h3_video_vae_dec_fp8.g2w",
                                    PACKS_DIR / "h3_video_vae_dec.g2w"),
            "--avae", pick_existing("/dev/shm/h3/h3_audio_vae_dec.g2w", PACKS_DIR / "h3_audio_vae_dec.g2w"),
            "--ffmpeg", ffmpeg_bin,
        ]
        if default_lora:
            cmd.extend(["--lora", default_lora])
    else:
        raise ValueError(f"Unknown engine: {engine_id}")

    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        cmd = [stdbuf, "-oL", "-eL"] + cmd
    return cmd, env


# ------------------------------------------------------------------------------------------------ Docker / HTTP helpers

class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 3.0):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def docker_request(method: str, path: str, timeout: float = 4.0):
    if not os.path.exists(DOCKER_SOCK):
        return None, None
    conn = UnixHTTPConnection(DOCKER_SOCK, timeout=timeout)
    try:
        conn.request(method, path)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, body
    except Exception:
        return None, None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def docker_get(path: str, timeout: float = 3.0):
    code, body = docker_request("GET", path, timeout=timeout)
    return body if code == 200 else None


def decode_docker_logs(raw: bytes) -> list:
    if not raw:
        return []
    if len(raw) >= 8 and raw[0] in (0, 1, 2) and raw[1:4] == b"\x00\x00\x00":
        i, n = 0, len(raw)
        buf = bytearray()
        while i + 8 <= n:
            size = int.from_bytes(raw[i + 4 : i + 8], "big")
            i += 8
            if size > n - i:
                buf.extend(raw[i:])
                break
            buf.extend(raw[i : i + size])
            i += size
        text = buf.decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return [ln for ln in text.splitlines() if ln.strip()]


def http_json(url: str, timeout: float = 2.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_status(url: str, timeout: float = 1.5) -> int:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        return resp.status


def post_json_req(url: str, payload: dict, timeout: float = 600.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def build_multipart(fields: list, files: list):
    """Build multipart/form-data body. fields=[(k,v)], files=[(k,filename,mime,bytes)]."""
    boundary = "----G2EStudio" + uuid.uuid4().hex
    parts = []
    for k, v in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8")
        )
    for k, fn, mime, raw in files:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{fn}"\r\n'
            f"Content-Type: {mime}\r\n\r\n".encode("utf-8")
            + raw
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def post_multipart_req(url: str, fields: list, files: list, timeout: float = 600.0):
    ctype, body = build_multipart(fields, files)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": ctype, "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def decode_image_item(item):
    """Accept either a data URL string, raw base64 string, or {'gallery_id': ...} -> (filename, mime, bytes)."""
    if isinstance(item, dict) and item.get("gallery_id"):
        meta = GALLERY.get_item(item["gallery_id"])
        if not meta:
            raise ValueError(f"Gallery item not found: {item['gallery_id']}")
        p = GALLERY_DIR / meta["filename"]
        if not p.is_file():
            raise ValueError(f"Gallery file missing: {meta['filename']}")
        mime = meta.get("mime") or mimetypes.guess_type(str(p))[0] or "image/png"
        return meta["filename"], mime, p.read_bytes()

    if isinstance(item, dict):
        s = item.get("data_url") or item.get("b64") or ""
        fn = item.get("name") or "ref.png"
    else:
        s = str(item or "")
        fn = "ref.png"
    mime = "image/png"
    if s.startswith("data:"):
        hdr, b64 = s.split(",", 1)
        m = re.match(r"data:([^;]+);base64", hdr)
        if m:
            mime = m.group(1)
        raw = base64.b64decode(b64)
    else:
        raw = base64.b64decode(s)
    ext = ".jpg" if "jpeg" in mime or "jpg" in mime else ".webp" if "webp" in mime else ".png"
    if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        fn = fn + ext
    return fn, mime, raw


# ------------------------------------------------------------------------------------------------ Hardware & Host

def read_sysfs(path: str) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def parse_num(s: str, default=None):
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    if not m:
        return default
    v = float(m.group(0))
    return int(v) if v.is_integer() else v


def collect_hardware():
    dev = "/sys/class/accel/accel0/device"
    hw = {
        "available": False,
        "name": "Gaudi2",
        "temp_c": None,
        "power_w": None,
        "power_max_w": None,
        "power_54v_w": None,
        "power_54v_max_w": None,
        "power_12v_w": None,
        "power_12v_max_w": None,
        "util_pct": None,
        "mem_used_mib": None,
        "mem_total_mib": None,
        "clock_mhz": None,
        "clock_max_mhz": None,
        "driver_ver": read_sysfs(f"{dev}/driver_ver"),
        "pcie_speed": read_sysfs(f"{dev}/current_link_speed"),
        "pcie_width": read_sysfs(f"{dev}/current_link_width"),
        "hard_reset_cnt": parse_num(read_sysfs(f"{dev}/hard_reset_cnt"), 0),
        "device_type": read_sysfs(f"{dev}/device_type"),
    }
    if os.path.exists(HL_SMI):
        try:
            out = subprocess.check_output(
                [
                    HL_SMI,
                    "-Q",
                    "index,name,temperature.aip,power.draw,power.max,utilization.aip,memory.used,memory.total,clocks.current.soc,clocks.max.soc",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.5,
            ).strip()
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                if len(parts) >= 10:
                    hw.update(
                        available=True,
                        name=parts[1],
                        temp_c=parse_num(parts[2]),
                        power_w=parse_num(parts[3]),
                        power_max_w=parse_num(parts[4]),
                        util_pct=parse_num(parts[5]),
                        mem_used_mib=parse_num(parts[6]),
                        mem_total_mib=parse_num(parts[7]),
                        clock_mhz=parse_num(parts[8]),
                        clock_max_mhz=parse_num(parts[9]),
                    )
        except Exception:
            pass
        try:
            pout = subprocess.check_output(
                [HL_SMI, "-q", "-d", "POWER"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2.5,
            )
            rails = {}
            cur = None
            for line in pout.splitlines():
                m = re.search(r"Power Readings\s*\(([^)]+)\)", line)
                if m:
                    cur = m.group(1).strip()
                    rails[cur] = {}
                    continue
                if cur and ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    if k in ("Power Draw", "Power Max", "Power Limit"):
                        rails[cur][k] = parse_num(v)
            p54 = next((v for k, v in rails.items() if "54V" in k.upper()), None)
            p12 = next((v for k, v in rails.items() if "12V" in k.upper()), None)
            if p54 is not None or p12 is not None:
                d54, m54 = (p54 or {}).get("Power Draw"), (p54 or {}).get("Power Max")
                d12, m12 = (p12 or {}).get("Power Draw"), (p12 or {}).get("Power Max")
                hw["power_54v_w"], hw["power_54v_max_w"] = d54, m54
                hw["power_12v_w"], hw["power_12v_max_w"] = d12, m12
                draws = [x for x in (d54, d12) if x is not None]
                maxes = [x for x in (m54, m12) if x is not None]
                if draws:
                    hw["power_w"] = round(sum(draws), 1) if any(isinstance(x, float) for x in draws) else sum(draws)
                if maxes:
                    hw["power_max_w"] = round(sum(maxes), 1) if any(isinstance(x, float) for x in maxes) else sum(maxes)
        except Exception:
            pass
    if not hw["available"]:
        clk = parse_num(read_sysfs(f"{dev}/clk_cur_freq_mhz"))
        if clk is not None:
            hw.update(available=True, clock_mhz=clk, clock_max_mhz=parse_num(read_sysfs(f"{dev}/clk_max_freq_mhz")))
    return hw


class HostSampler:
    def __init__(self):
        self._prev_total = 0
        self._prev_idle = 0

    def sample(self):
        cpu_pct = 0.0
        try:
            vals = [int(x) for x in pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]
            idle, total = vals[3] + vals[4], sum(vals)
            if self._prev_total > 0 and total > self._prev_total:
                cpu_pct = round(100.0 * (1.0 - (idle - self._prev_idle) / (total - self._prev_total)), 1)
            self._prev_total, self._prev_idle = total, idle
        except Exception:
            pass
        mem_total_gib = mem_used_gib = 0.0
        try:
            info = {}
            for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
                k, v = line.split(":", 1)
                info[k.strip()] = int(v.strip().split()[0])
            tot, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
            mem_total_gib = round(tot / (1024 * 1024), 1)
            mem_used_gib = round((tot - avail) / (1024 * 1024), 1)
        except Exception:
            pass
        shm_used_gib = shm_total_gib = 0.0
        try:
            st = os.statvfs("/dev/shm")
            shm_total_gib = round((st.f_blocks * st.f_frsize) / (1 << 30), 1)
            shm_used_gib = round(((st.f_blocks - st.f_bavail) * st.f_frsize) / (1 << 30), 1)
        except Exception:
            pass
        try:
            load1 = round(os.getloadavg()[0], 2)
        except Exception:
            load1 = 0.0
        return {
            "cpu_pct": cpu_pct,
            "mem_used_gib": mem_used_gib,
            "mem_total_gib": mem_total_gib,
            "shm_used_gib": shm_used_gib,
            "shm_total_gib": shm_total_gib,
            "load1": load1,
        }


# ------------------------------------------------------------------------------------------------ Persistent Gallery

class GalleryStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, media_bytes: bytes, ext: str, meta: dict) -> dict:
        item_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        filename = f"{item_id}.{ext.lstrip('.')}"
        record = dict(meta)
        record.update(
            id=item_id,
            filename=filename,
            url=f"/media/{filename}",
            mime="video/mp4" if ext.lstrip(".") == "mp4" else "image/png",
            bytes=len(media_bytes),
            created_at=time.time(),
        )
        with self.lock:
            (self.root / filename).write_bytes(media_bytes)
            (self.root / f"{item_id}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return record

    def list_items(self, limit: int = 80) -> list:
        items = []
        with self.lock:
            for jp in sorted(self.root.glob("*.json"), reverse=True):
                if len(items) >= limit:
                    break
                try:
                    d = json.loads(jp.read_text(encoding="utf-8"))
                    if (self.root / d.get("filename", "")).is_file():
                        items.append(d)
                except Exception:
                    continue
        return items

    def get_item(self, item_id: str):
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "", item_id or "")
        jp = self.root / f"{safe}.json"
        if not jp.is_file():
            return None
        try:
            return json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            return None

    def delete(self, item_id: str) -> bool:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "", item_id or "")
        with self.lock:
            jp = self.root / f"{safe}.json"
            if not jp.is_file():
                return False
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
                mp = self.root / d.get("filename", "")
                if mp.is_file():
                    mp.unlink()
            except Exception:
                pass
            jp.unlink(missing_ok=True)
            return True

    def count(self) -> int:
        try:
            return sum(1 for _ in self.root.glob("*.json"))
        except Exception:
            return 0


GALLERY = GalleryStore(GALLERY_DIR)


# ------------------------------------------------------------------------------------------------ Engine Manager

class EngineManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.proc_engine = None
        self.proc_log_path = None
        self.state = "idle"       # idle, stopping, loading, running, error
        self.target = None        # engine_id currently starting/running
        self.error = None
        self.started_ts = None
        self.status_msg = "空闲"
        self._switch_seq = 0
        self.can_manage = BUILD_DIR.is_dir() and any((BUILD_DIR / s["binary"]).is_file() for s in ENGINE_SPECS.values())
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        try:
            stat_txt = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="ignore")
            rp = stat_txt.rfind(")")
            if rp != -1:
                fields = stat_txt[rp + 2 :].split()
                if fields and fields[0] == "Z":
                    return False  # zombie already exited
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _wait_device_ready(self, timeout: float = 45.0, seq=None) -> bool:
        dev_path = "/dev/accel/accel0"
        if not os.path.exists(dev_path):
            return True
        status_path = "/sys/class/accel/accel0/device/status"
        deadline = time.time() + timeout
        waited = False
        while time.time() < deadline:
            if seq is not None:
                with self.lock:
                    if self._switch_seq != seq:
                        return False
            st = read_sysfs(status_path)
            if not st or st.lower() == "operational":
                try:
                    fd = os.open(dev_path, os.O_RDWR)
                    os.close(fd)
                    if waited:
                        time.sleep(0.5)  # brief settle after kernel teardown finishes
                    return True
                except OSError:
                    pass
            if not waited:
                waited = True
                if seq is not None:
                    with self.lock:
                        if self._switch_seq == seq:
                            self.status_msg = "正在等待 Gaudi2 设备完全释放…"
            time.sleep(0.5)
        return False

    def _kill_all_engines(self, seq=None):
        # 1. Stop our managed subprocess first and ALWAYS reap it (even after SIGKILL)
        with self.lock:
            p = self.proc
            self.proc = None
            self.proc_engine = None
        if p is not None:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                try:
                    p.wait(timeout=12)
                except Exception:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass
                    try:
                        p.wait(timeout=15)
                    except Exception:
                        pass
            else:
                try:
                    p.wait(timeout=1)
                except Exception:
                    pass

        # 2. Stop Docker container `gaudiinfer` if running
        raw = docker_get(f"/containers/{urllib.parse.quote(CONTAINER_NAME)}/json", timeout=2.0)
        if raw:
            try:
                d = json.loads(raw.decode("utf-8"))
                if d.get("State", {}).get("Running"):
                    docker_request("POST", f"/containers/{urllib.parse.quote(CONTAINER_NAME)}/stop?t=10", timeout=15.0)
            except Exception:
                pass

        # 3. Stop any externally launched host g2e_*_server processes holding /dev/accel/accel0
        bins = [s["binary"] for s in ENGINE_SPECS.values()]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            pids = []
            for b in bins:
                try:
                    out = subprocess.check_output(["pgrep", "-f", f"/{b}(\\s|$)"], text=True, stderr=subprocess.DEVNULL)
                    for ln in out.splitlines():
                        pid = int(ln.strip())
                        if pid != os.getpid() and self._is_pid_alive(pid):
                            pids.append(pid)
                except Exception:
                    pass
            if not pids:
                break
            for pid in set(pids):
                try:
                    os.kill(pid, sig)
                except Exception:
                    pass
            deadline = time.time() + (10.0 if sig == signal.SIGTERM else 5.0)
            while time.time() < deadline:
                if not any(self._is_pid_alive(pid) for pid in set(pids)):
                    break
                time.sleep(0.3)

        # 4. Wait until habanalabs finishes releasing / resetting /dev/accel/accel0
        self._wait_device_ready(timeout=45.0, seq=seq)

    def stop_async(self):
        if not self.can_manage:
            return False, "当前运行在只读容器内，无法管理宿主机引擎"
        with self.lock:
            self._switch_seq += 1
            seq = self._switch_seq
            self.state = "stopping"
            self.target = None
            self.error = None
            self.status_msg = "正在停止当前引擎并释放 Gaudi2 HBM…"

        def _worker():
            try:
                self._kill_all_engines(seq=seq)
                with self.lock:
                    if self._switch_seq == seq:
                        self.state = "idle"
                        self.status_msg = "Gaudi2 空闲，可随时启动任意模型"
                        self.started_ts = None
            except Exception as e:
                with self.lock:
                    if self._switch_seq == seq:
                        self.state = "error"
                        self.error = str(e)
                        self.status_msg = f"停止失败：{e}"

        threading.Thread(target=_worker, daemon=True).start()
        return True, "stopping"

    def switch_async(self, engine_id: str):
        if engine_id not in ENGINE_SPECS:
            return False, f"未知引擎: {engine_id}"
        if not self.can_manage:
            return False, "当前环境未挂载宿主机编译目录，无法自动切换引擎"
        spec = ENGINE_SPECS[engine_id]
        bin_path = BUILD_DIR / spec["binary"]
        if not bin_path.is_file():
            return False, f"未找到可执行文件: {bin_path}"

        with self.lock:
            self._switch_seq += 1
            seq = self._switch_seq
            self.state = "loading"
            self.target = engine_id
            self.error = None
            self.started_ts = time.time()
            self.status_msg = f"正在释放旧引擎并启动 {spec['name']}…"

        def _worker():
            try:
                self._kill_all_engines(seq=seq)
                with self.lock:
                    if self._switch_seq != seq:
                        return
                    self.status_msg = f"正在将 {spec['name']} 权重加载至 Gaudi2 HBM…"

                cmd, env = build_engine_command(engine_id)
                log_path = LOG_DIR / f"g2e_{engine_id}.log"
                health_url = f"http://127.0.0.1:{spec['port']}/health"
                max_attempts = 3

                for attempt in range(max_attempts):
                    with self.lock:
                        if self._switch_seq != seq:
                            return
                    lf = open(log_path, "wb")
                    try:
                        lf.write(f"[studio] launching {spec['name']}: {' '.join(cmd)}\n".encode("utf-8"))
                        lf.flush()
                        proc = subprocess.Popen(
                            cmd,
                            cwd=str(REPO_DIR),
                            env=env,
                            stdout=lf,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    finally:
                        lf.close()

                    with self.lock:
                        if self._switch_seq != seq:
                            try:
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                            except Exception:
                                pass
                            return
                        self.proc = proc
                        self.proc_engine = engine_id
                        self.proc_log_path = log_path

                    # Wait up to 600s for /health == 200 (covers cold 624-recipe cache warmup on 27B)
                    launch_ts = time.time()
                    deadline = launch_ts + 600.0
                    retry_acquire = False
                    while time.time() < deadline:
                        with self.lock:
                            if self._switch_seq != seq:
                                return
                        if proc.poll() is not None:
                            tail = ""
                            try:
                                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
                                tail = "\n".join(lines)
                            except Exception:
                                pass
                            is_dev_busy = (
                                "Synapse status 8" in tail
                                or "Device not found" in tail
                                or "synDeviceAcquire" in tail
                            )
                            if is_dev_busy and (time.time() - launch_ts) < 20.0 and attempt + 1 < max_attempts:
                                with self.lock:
                                    if self._switch_seq == seq:
                                        self.status_msg = (
                                            f"Gaudi2 设备仍在复位，等待后重试启动 {spec['name']} ({attempt + 2}/{max_attempts})…"
                                        )
                                self._wait_device_ready(timeout=30.0, seq=seq)
                                time.sleep(2.0)
                                retry_acquire = True
                                break
                            with self.lock:
                                if self._switch_seq == seq:
                                    self.state = "error"
                                    self.error = f"进程异常退出 (rc={proc.returncode}): {tail}"
                                    self.status_msg = f"{spec['name']} 启动失败 (rc={proc.returncode})"
                            return
                        try:
                            if http_status(health_url, timeout=1.5) == 200:
                                with self.lock:
                                    if self._switch_seq == seq:
                                        self.state = "running"
                                        self.status_msg = f"{spec['name']} 已就绪（:{spec['port']}）"
                                return
                        except Exception:
                            pass
                        time.sleep(1.0)

                    if retry_acquire:
                        continue

                    with self.lock:
                        if self._switch_seq == seq:
                            self.state = "error"
                            self.error = "等待 /health 超时 (600s)"
                            self.status_msg = f"{spec['name']} 启动超时"
                    return
            except Exception as e:
                with self.lock:
                    if self._switch_seq == seq:
                        self.state = "error"
                        self.error = str(e)
                        self.status_msg = f"切换失败：{e}"

        threading.Thread(target=_worker, daemon=True).start()
        return True, f"switching to {engine_id}"

    def read_managed_log(self, engine_id=None, max_bytes=98304):
        paths = []
        with self.lock:
            if self.proc_log_path and self.proc_log_path.is_file():
                paths.append(self.proc_log_path)
        if engine_id:
            paths.append(LOG_DIR / f"g2e_{engine_id}.log")
        if engine_id == "h3":
            paths.append(pathlib.Path("/tmp/g2e_h3_server.log"))
        if engine_id == "llm":
            paths.append(pathlib.Path("/mnt/sas/g2e/server.log"))
        for p in paths:
            if p.is_file():
                try:
                    data = p.read_bytes()
                    return data[-max_bytes:]
                except Exception:
                    pass
        return b""

    def info(self, healthy_map: dict):
        with self.lock:
            p_alive = self.proc is not None and self.proc.poll() is None
            st = self.state
            tgt = self.target
            err = self.error
            msg = self.status_msg
            started = self.started_ts

        active = next((eid for eid in ENGINE_ORDER if healthy_map.get(eid)), None)
        if st == "running" and not active and not p_alive:
            st = "idle"
            msg = "Gaudi2 空闲"
        elif st == "idle" and active:
            st = "running"
            tgt = active
            msg = f"{ENGINE_SPECS[active]['name']} 运行中（:{ENGINE_SPECS[active]['port']}）"

        items = []
        for eid in ENGINE_ORDER:
            sp = ENGINE_SPECS[eid]
            items.append(
                {
                    **sp,
                    "url": f"http://127.0.0.1:{sp['port']}",
                    "healthy": bool(healthy_map.get(eid)),
                    "loading": st == "loading" and tgt == eid and not healthy_map.get(eid),
                    "installed": (BUILD_DIR / sp["binary"]).is_file(),
                }
            )
        return {
            "can_manage": self.can_manage,
            "state": st,
            "active": active,
            "target": tgt,
            "error": err,
            "status_msg": msg,
            "elapsed_s": round(time.time() - started, 1) if (started and st in ("loading", "stopping")) else None,
            "items": items,
            "h3_loras": discover_h3_loras(),
        }


MANAGER = EngineManager()


# ------------------------------------------------------------------------------------------------ Video Job Tracker

class VideoJobTracker:
    """Tracks MiniMax-H3 async video jobs and archives completed MP4s into GALLERY."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs = {}  # job_id -> {meta, status, gallery_item, error, submitted_at, ...}

    def register(self, job_id: str, meta: dict):
        with self.lock:
            self.jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "progress": 0,
                "meta": meta,
                "gallery_item": None,
                "error": None,
                "submitted_at": time.time(),
            }

    def poll(self, job_id: str):
        base = f"http://127.0.0.1:{ENGINE_SPECS['h3']['port']}"
        with self.lock:
            rec = self.jobs.get(job_id)
        if rec and rec.get("gallery_item"):
            return {
                "id": job_id,
                "status": "completed",
                "progress": 100,
                "item": rec["gallery_item"],
                "timing": rec["gallery_item"].get("timing"),
            }
        _, st = http_json(f"{base}/v1/videos/{urllib.parse.quote(job_id)}", timeout=5.0)
        status = st.get("status", "unknown")
        if status == "completed":
            # Download MP4 and persist into Gallery once
            with self.lock:
                rec = self.jobs.get(job_id)
                if rec and rec.get("gallery_item"):
                    return {
                        "id": job_id,
                        "status": "completed",
                        "progress": 100,
                        "item": rec["gallery_item"],
                        "timing": rec["gallery_item"].get("timing"),
                    }
            req = urllib.request.Request(f"{base}/v1/videos/{urllib.parse.quote(job_id)}/content")
            with urllib.request.urlopen(req, timeout=60.0) as resp:
                mp4_bytes = resp.read()
            meta = dict((rec or {}).get("meta") or {})
            meta.update(
                engine="h3",
                model="MiniMaxAI/MiniMax-H3",
                kind="video",
                width=st.get("width") or meta.get("width"),
                height=st.get("height") or meta.get("height"),
                num_frames=st.get("num_frames"),
                timing=st.get("timing"),
                job_id=job_id,
            )
            item = GALLERY.save(mp4_bytes, "mp4", meta)
            with self.lock:
                if rec is not None:
                    rec["status"] = "completed"
                    rec["progress"] = 100
                    rec["gallery_item"] = item
            return {"id": job_id, "status": "completed", "progress": 100, "item": item, "timing": st.get("timing")}

        elapsed = round(time.time() - rec["submitted_at"], 1) if rec else None
        return {
            "id": job_id,
            "status": status,
            "progress": st.get("progress", 0),
            "elapsed_s": elapsed,
            "error": (st.get("error") or {}).get("message") if isinstance(st.get("error"), dict) else st.get("error"),
        }


VIDEO_TRACKER = VideoJobTracker()


# ------------------------------------------------------------------------------------------------ Telemetry Store

class TelemetryStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = collections.deque(maxlen=300)
        self.requests = []
        self.logs = []
        self.engine_banner = {}
        self.latest = {}
        self.host_sampler = HostSampler()
        self._last_log_ts = 0.0
        self._container_cache = {}
        self._models_cache = {}
        self._version_cache = None
        self._prev_live = None

    def _poll_engines(self):
        healthy_map = {}
        active_id = None
        for eid in ENGINE_ORDER:
            port = ENGINE_SPECS[eid]["port"]
            try:
                ok = http_status(f"http://127.0.0.1:{port}/health", timeout=0.8) == 200
            except Exception:
                ok = False
            healthy_map[eid] = ok
            if ok and active_id is None:
                active_id = eid

        eng = {
            "healthy": active_id is not None,
            "url": f"http://127.0.0.1:{ENGINE_SPECS[active_id]['port']}" if active_id else f"http://127.0.0.1:{ENGINE_SPECS['llm']['port']}",
            "mode": active_id or "idle",
            "version": self._version_cache or "gaudi2-engine",
            "model": ENGINE_SPECS[active_id]["name"] if active_id else None,
            "max_model_len": None,
            "started_at": None,
            "stats": None,
            "error": None if active_id else "当前无活跃引擎（可在上方一键启动）",
        }
        if not active_id:
            return eng, healthy_map

        active_url = eng["url"]
        try:
            _, eng["stats"] = http_json(f"{active_url}/stats", timeout=1.5)
            if eng["stats"] and eng["stats"].get("model"):
                eng["model"] = eng["stats"]["model"]
        except Exception as e:
            eng["error"] = str(e)

        now_i = int(time.time())
        if active_id not in self._models_cache or now_i % 15 == 0:
            try:
                _, self._models_cache[active_id] = http_json(f"{active_url}/v1/models", timeout=1.5)
            except Exception:
                pass
        mc = self._models_cache.get(active_id)
        if mc and mc.get("data"):
            m0 = mc["data"][0]
            eng["model"] = m0.get("id") or eng["model"]
            eng["max_model_len"] = m0.get("max_model_len")
            eng["started_at"] = m0.get("created")

        if active_id == "llm" and self._version_cache is None:
            try:
                _, ver = http_json(f"{active_url}/version", timeout=1.5)
                self._version_cache = ver.get("version")
                eng["version"] = self._version_cache
            except Exception:
                pass
        return eng, healthy_map

    def _poll_logs_and_docker(self, active_mode: str, target_mode):
        now = time.time()
        if now - self._last_log_ts < 1.5 and self.logs:
            return self._container_cache, self.logs, self.requests, self.engine_banner
        self._last_log_ts = now

        cinfo = {"available": False, "name": CONTAINER_NAME}
        raw_info = docker_get(f"/containers/{urllib.parse.quote(CONTAINER_NAME)}/json", timeout=1.5)
        if raw_info:
            try:
                d = json.loads(raw_info.decode("utf-8"))
                st = d.get("State", {})
                cinfo = {
                    "available": True,
                    "name": d.get("Name", "").lstrip("/"),
                    "id": d.get("Id", "")[:12],
                    "image": d.get("Config", {}).get("Image", ""),
                    "status": st.get("Status", "unknown"),
                    "running": st.get("Running", False),
                    "health": (st.get("Health") or {}).get("Status", "none"),
                    "started_at": st.get("StartedAt", ""),
                    "restart_count": d.get("RestartCount", 0),
                    "oom_killed": st.get("OOMKilled", False),
                }
            except Exception:
                pass
        self._container_cache = cinfo

        log_eid = target_mode or (active_mode if active_mode != "idle" else None)
        raw_logs = MANAGER.read_managed_log(log_eid)
        if not raw_logs and cinfo.get("running"):
            raw_logs = docker_get(
                f"/containers/{urllib.parse.quote(CONTAINER_NAME)}/logs?stdout=1&stderr=1&timestamps=1&tail=450",
                timeout=2.0,
            ) or b""

        parsed_logs, parsed_reqs = [], []
        banner = dict(self.engine_banner)
        if raw_logs:
            for line in decode_docker_logs(raw_logs):
                ts_str, msg = "", line
                if len(line) > 31 and line[4] == "-" and line[10] == "T" and " " in line[:40]:
                    sp = line.index(" ")
                    ts_str, msg = line[:sp], line[sp + 1 :]
                parsed_logs.append({"ts": ts_str, "msg": msg})

                short = msg.replace("[g2e] ", "")
                if "Direct MME+SRAM engine active" in msg:
                    banner["direct_engine"] = short
                elif "dynamic KV caches:" in msg:
                    banner["kv_summary"] = short
                elif "target weights " in msg:
                    banner["target_weights"] = short
                elif "drafter weights " in msg:
                    banner["drafter_weights"] = short
                elif "vision weights " in msg:
                    banner["vision_weights"] = short
                elif "prefix cache:" in msg:
                    banner["prefix_cache"] = short

                m = REQ_RE.search(msg)
                if m:
                    p_tok, c_tok = int(m.group(1)), int(m.group(2))
                    p_ms, g_ms = float(m.group(5)), float(m.group(7))
                    uncached = max(0, p_tok - c_tok)
                    parsed_reqs.append(
                        {
                            "ts": ts_str,
                            "prompt_tokens": p_tok,
                            "cached_tokens": c_tok,
                            "cached_pct": float(m.group(3)),
                            "prompt_ms": p_ms,
                            "prefill_tps": round(uncached / (p_ms / 1000.0), 1) if p_ms > 0 and uncached > 0 else 0.0,
                            "eff_prefill_tps": round(p_tok / (p_ms / 1000.0), 1) if p_ms > 0 else 0.0,
                            "gen_tokens": int(m.group(6)),
                            "decode_ms": g_ms,
                            "decode_tps": float(m.group(8)),
                            "rounds": int(m.group(9)),
                            "tok_per_round": float(m.group(10)),
                            "finish": m.group(11),
                        }
                    )
        return cinfo, parsed_logs[-260:], parsed_reqs[-100:], banner

    def poll_once(self):
        ts = time.time()
        eng, healthy_map = self._poll_engines()
        engines_info = MANAGER.info(healthy_map)
        hw = collect_hardware()
        host = self.host_sampler.sample()
        cinfo, logs, reqs, banner = self._poll_logs_and_docker(eng["mode"], engines_info.get("target"))

        slots = (eng.get("stats") or {}).get("slots") or {}
        pc = (eng.get("stats") or {}).get("prefix_cache") or {}
        raw_live = (eng.get("stats") or {}).get("live") or {}
        live_dec_tps = live_pre_tps = live_tpr = 0.0
        if "decode_tokens" in raw_live:
            cur_dt = int(raw_live.get("decode_tokens", 0))
            cur_dr = int(raw_live.get("decode_rounds", 0))
            cur_pt = int(raw_live.get("prefill_tokens", 0))
            if self._prev_live is not None:
                pts, pdt, pdr, ppt = self._prev_live
                dt = max(0.05, ts - pts)
                if cur_dt >= pdt and cur_pt >= ppt:
                    ddt, ddr, dpt = cur_dt - pdt, max(0, cur_dr - pdr), cur_pt - ppt
                    live_dec_tps = round(ddt / dt, 1)
                    live_pre_tps = round(dpt / dt, 1)
                    if ddr > 0:
                        live_tpr = round(ddt / ddr, 2)
            self._prev_live = (ts, cur_dt, cur_dr, cur_pt)
        else:
            self._prev_live = None

        live = {
            "supported": "decode_tokens" in raw_live,
            "decode_tps": live_dec_tps,
            "prefill_tps": live_pre_tps,
            "tok_per_round": live_tpr,
            "decode_tokens": int(raw_live.get("decode_tokens", 0)),
            "prefill_tokens": int(raw_live.get("prefill_tokens", 0)),
            "decoding_slots": int(raw_live.get("decoding_slots", slots.get("decoding", 0))),
            "prefilling_slots": int(raw_live.get("prefilling_slots", slots.get("prefilling", 0))),
        }

        point = {
            "t": round(ts, 1),
            "util": hw.get("util_pct") or 0,
            "power": hw.get("power_w") or 0,
            "temp": hw.get("temp_c") or 0,
            "hbm_gib": round((hw.get("mem_used_mib") or 0) / 1024.0, 2),
            "active_slots": slots.get("active", 0),
            "queued": slots.get("queued", 0),
            "kv_gib": round(slots.get("kv_used_gib", 0.0), 2),
            "prefix_gib": round(pc.get("host_gib", 0.0), 2),
            "cpu": host.get("cpu_pct", 0.0),
            "decode_tps": live_dec_tps,
            "prefill_tps": live_pre_tps,
        }
        with self.lock:
            self.history.append(point)
            self.logs = logs
            self.requests = reqs
            self.engine_banner = banner
            self.latest = {
                "timestamp": ts,
                "engine": eng,
                "engines": engines_info,
                "live": live,
                "hardware": hw,
                "host": host,
                "container": cinfo,
                "banner": banner,
                "gallery_count": GALLERY.count(),
            }

    def snapshot(self):
        with self.lock:
            snap = dict(self.latest)
            snap["history"] = list(self.history)
            snap["requests"] = list(reversed(self.requests))
            snap["log_tail"] = [l["msg"] for l in self.logs[-12:]]
            return snap

    def get_logs(self, tail: int = 250):
        with self.lock:
            return list(self.logs[-max(1, min(tail, 500)) :])


STORE = TelemetryStore()


def poller_loop():
    while True:
        try:
            STORE.poll_once()
        except Exception:
            pass
        time.sleep(POLL_SEC)


# ------------------------------------------------------------------------------------------------ Chat Streaming

def stream_chat_events(messages: list, max_tokens: int, temperature: float, enable_thinking: bool = True):
    llm_url = f"http://127.0.0.1:{ENGINE_SPECS['llm']['port']}"
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max(1, min(int(max_tokens), 8192)),
        "temperature": max(0.0, min(float(temperature), 2.0)),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    req = urllib.request.Request(
        f"{llm_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_ms = None
    content, reasoning = [], []
    deltas = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = b""
        done = False
        while not done:
            chunk = resp.read1(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw_line, buf = buf.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    done = True
                    break
                try:
                    ev = json.loads(body)
                except Exception:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                choices = ev.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                c = delta.get("content") or ""
                r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if not (c or r):
                    continue
                now_ms = (time.perf_counter() - t0) * 1000.0
                if ttft_ms is None:
                    ttft_ms = now_ms
                deltas += 1
                if c:
                    content.append(c)
                if r:
                    reasoning.append(r)
                yield {"type": "delta", "c": c, "r": r, "t": round(now_ms, 1), "n": deltas}
    total_ms = (time.perf_counter() - t0) * 1000.0
    comp_tokens = usage.get("completion_tokens") or deltas
    ttft = ttft_ms if ttft_ms is not None else total_ms
    decode_ms = max(1.0, total_ms - ttft)
    yield {
        "type": "done",
        "ok": True,
        "ttft_ms": round(ttft, 1),
        "total_ms": round(total_ms, 1),
        "decode_tps": round(comp_tokens / (decode_ms / 1000.0), 1) if comp_tokens > 1 else 0.0,
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        "completion_tokens": comp_tokens,
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
    }


# ------------------------------------------------------------------------------------------------ HTTP Handler

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def _send_json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: pathlib.Path, ctype: str, cache="no-cache"):
        if not path.is_file():
            self.send_error(404, "Not Found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, path: pathlib.Path):
        if not path.is_file():
            self.send_error(404, "Not Found")
            return
        total = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                start = int(m.group(1)) if m.group(1) else 0
                end = int(m.group(2)) if m.group(2) else total - 1
                end = min(end, total - 1)
                if start <= end < total:
                    length = end - start + 1
                    self.send_response(206)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.end_headers()
                    with open(path, "rb") as fh:
                        fh.seek(start)
                        rem = length
                        while rem > 0:
                            chunk = fh.read(min(262144, rem))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            rem -= len(chunk)
                    return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(total))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(262144)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _read_json(self, limit: int = 128 * 1024 * 1024):
        length = int(self.headers.get("Content-Length", "0"))
        if length > limit:
            raise ValueError(f"Request body too large ({length} > {limit})")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if parsed.path == "/health" else "text/html; charset=utf-8")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_error(404, "Not Found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path
        if p in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if p == "/studio.css":
            return self._send_file(STATIC_DIR / "studio.css", "text/css; charset=utf-8")
        if p == "/studio.js":
            return self._send_file(STATIC_DIR / "studio.js", "application/javascript; charset=utf-8")
        if p == "/health":
            return self._send_json(200, {"status": "ok", "studio": True})
        if p == "/api/snapshot":
            return self._send_json(200, STORE.snapshot())
        if p == "/api/logs":
            qs = urllib.parse.parse_qs(parsed.query)
            return self._send_json(200, {"lines": STORE.get_logs(int(qs.get("tail", ["250"])[0]))})
        if p == "/api/gallery":
            qs = urllib.parse.parse_qs(parsed.query)
            limit = int(qs.get("limit", ["80"])[0])
            return self._send_json(200, {"items": GALLERY.list_items(limit)})
        if p.startswith("/media/"):
            fn = os.path.basename(urllib.parse.unquote(p[len("/media/"):]))
            if not fn or "/" in fn or "\\" in fn:
                return self.send_error(400, "Bad filename")
            return self._send_media(GALLERY_DIR / fn)
        if p.startswith("/api/video/status/"):
            jid = urllib.parse.unquote(p[len("/api/video/status/"):])
            try:
                return self._send_json(200, VIDEO_TRACKER.poll(jid))
            except Exception as e:
                return self._send_json(502, {"ok": False, "error": str(e)})
        self.send_error(404, "Not Found")

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/gallery/"):
            item_id = urllib.parse.unquote(parsed.path[len("/api/gallery/"):])
            ok = GALLERY.delete(item_id)
            return self._send_json(200 if ok else 404, {"ok": ok})
        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path

        if p == "/api/engine/switch":
            try:
                req = self._read_json(65536)
                eid = str(req.get("engine") or "").strip()
                ok, msg = MANAGER.switch_async(eid)
                return self._send_json(200 if ok else 400, {"ok": ok, "message": msg})
            except Exception as e:
                return self._send_json(400, {"ok": False, "error": str(e)})

        if p == "/api/engine/stop":
            ok, msg = MANAGER.stop_async()
            return self._send_json(200 if ok else 400, {"ok": ok, "message": msg})

        if p in ("/api/chat/stream", "/api/probe", "/api/probe/stream"):
            return self._handle_chat(p)

        if p == "/api/image/generate":
            return self._handle_image_generate()

        if p == "/api/video/submit":
            return self._handle_video_submit()

        self.send_error(404, "Not Found")

    def _handle_chat(self, path: str):
        try:
            req = self._read_json()
            max_tokens = int(req.get("max_tokens", 1024 if path == "/api/chat/stream" else 256))
            temperature = float(req.get("temperature", 0.7 if path == "/api/chat/stream" else 0.0))
            enable_thinking = bool(req.get("enable_thinking", True))
            if "messages" in req and isinstance(req["messages"], list):
                messages = []
                for m in req["messages"]:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    # Resolve any gallery_id image attachments in user message
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "gallery_image" and part.get("gallery_id"):
                                _, mime, raw = decode_image_item({"gallery_id": part["gallery_id"]})
                                url = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
                                parts.append({"type": "image_url", "image_url": {"url": url}})
                            else:
                                parts.append(part)
                        messages.append({"role": role, "content": parts})
                    else:
                        messages.append({"role": role, "content": content})
            else:
                prompt = str(req.get("prompt") or "用一句话介绍 Intel Gaudi2 推理加速卡。").strip()
                messages = [{"role": "user", "content": prompt}]
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": f"bad request: {e}"})

        if path == "/api/probe":
            try:
                final = {}
                for ev in stream_chat_events(messages, max_tokens, temperature, enable_thinking):
                    if ev["type"] == "done":
                        final = ev
                return self._send_json(200, final)
            except Exception as e:
                return self._send_json(502, {"ok": False, "error": str(e)})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        try:
            for ev in stream_chat_events(messages, max_tokens, temperature, enable_thinking):
                self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            try:
                err = {"type": "error", "ok": False, "error": str(e)}
                self.wfile.write(("data: " + json.dumps(err, ensure_ascii=False) + "\n\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def _handle_image_generate(self):
        try:
            req = self._read_json()
            engine = str(req.get("engine") or "qwen21").strip()
            if engine not in ("qwen21", "h3_image"):
                raise ValueError(f"Unsupported image engine: {engine}")
            prompt = str(req.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("请输入提示词 (prompt)")
            width = int(req.get("width", 1024 if engine == "qwen21" else 2048))
            height = int(req.get("height", 1024 if engine == "qwen21" else 2048))
            seed = int(req.get("seed", int(time.time() * 1000) % 2147483647))
            neg = str(req.get("negative_prompt") or "")
            steps = int(req.get("steps", 8 if engine == "qwen21" else 6))
            images = req.get("images") or []
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": str(e)})

        base = f"http://127.0.0.1:{ENGINE_SPECS[engine]['port']}"
        t0 = time.perf_counter()
        try:
            if not images:
                payload = {
                    "model": ENGINE_SPECS[engine]["name"],
                    "prompt": prompt,
                    "size": f"{width}x{height}",
                    "seed": seed,
                }
                if engine == "qwen21":
                    payload["num_inference_steps"] = steps
                    if neg:
                        payload["negative_prompt"] = neg
                _, data = post_json_req(f"{base}/v1/images/generations", payload, timeout=600.0)
            else:
                fields = [
                    ("model", ENGINE_SPECS[engine]["name"]),
                    ("prompt", prompt),
                    ("size", f"{width}x{height}"),
                    ("seed", str(seed)),
                ]
                if engine == "qwen21":
                    fields.append(("num_inference_steps", str(steps)))
                    if neg:
                        fields.append(("negative_prompt", neg))
                files = []
                for idx, item in enumerate(images):
                    fn, mime, raw = decode_image_item(item)
                    files.append(("image", fn or f"ref_{idx}.png", mime, raw))
                _, data = post_multipart_req(f"{base}/v1/images/edits", fields, files, timeout=600.0)

            b64 = data["data"][0]["b64_json"]
            png_bytes = base64.b64decode(b64)
            wall_s = round(time.perf_counter() - t0, 2)
            item = GALLERY.save(
                png_bytes,
                "png",
                {
                    "engine": engine,
                    "model": ENGINE_SPECS[engine]["name"],
                    "kind": "image",
                    "mode": "Edit" if images else "T2I",
                    "n_ref": len(images),
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "steps": steps,
                    "timing": data.get("timing"),
                    "wall_s": wall_s,
                },
            )
            return self._send_json(200, {"ok": True, "item": item, "wall_s": wall_s, "timing": data.get("timing")})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return self._send_json(e.code, {"ok": False, "error": body or str(e)})
        except Exception as e:
            return self._send_json(502, {"ok": False, "error": str(e)})

    def _handle_video_submit(self):
        try:
            req = self._read_json()
            prompt = str(req.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("请输入视频提示词 (prompt)")
            width = int(req.get("width", 1280))
            height = int(req.get("height", 768))
            duration = float(req.get("duration", 5.0))
            fps = int(req.get("fps", 24))
            steps = int(req.get("steps", 8))
            seed = int(req.get("seed", int(time.time() * 1000) % 2147483647))
            flow_shift = float(req.get("flow_shift", 6.0))
            audio_flow_shift = float(req.get("audio_flow_shift", 3.0))
            lora_id = str(req.get("lora") or "v1.2_8step")
            first_frame = req.get("first_frame")
            last_frame = req.get("last_frame")
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": str(e)})

        loras = {l["id"]: l for l in discover_h3_loras()}
        lora_path = (loras.get(lora_id) or {}).get("path") or "none"
        base = f"http://127.0.0.1:{ENGINE_SPECS['h3']['port']}"

        fields = [
            ("prompt", prompt),
            ("size", f"{width}x{height}"),
            ("seconds", f"{duration:.2f}"),
            ("fps", str(fps)),
            ("steps", str(steps)),
            ("seed", str(seed)),
            ("flow_shift", str(flow_shift)),
            ("audio_flow_shift", str(audio_flow_shift)),
            ("lora_path", lora_path),
        ]
        files = []
        frame_indices = []
        try:
            if first_frame and last_frame:
                fn1, m1, r1 = decode_image_item(first_frame)
                fn2, m2, r2 = decode_image_item(last_frame)
                files.append(("input_references", fn1, m1, r1))
                files.append(("input_references", fn2, m2, r2))
                frame_indices = [0, -1]
            elif first_frame:
                fn1, m1, r1 = decode_image_item(first_frame)
                files.append(("input_reference", fn1, m1, r1))
                frame_indices = [0]
            elif last_frame:
                fn2, m2, r2 = decode_image_item(last_frame)
                files.append(("input_reference", fn2, m2, r2))
                frame_indices = [-1]
            if frame_indices:
                fields.append(("frame_indices", json.dumps(frame_indices)))

            if files:
                _, data = post_multipart_req(f"{base}/v1/videos", fields, files, timeout=60.0)
            else:
                payload = {
                    "prompt": prompt,
                    "size": f"{width}x{height}",
                    "seconds": duration,
                    "fps": fps,
                    "steps": steps,
                    "seed": seed,
                    "flow_shift": flow_shift,
                    "audio_flow_shift": audio_flow_shift,
                    "lora_path": lora_path,
                }
                _, data = post_json_req(f"{base}/v1/videos", payload, timeout=60.0)

            job_id = data["id"]
            mode = "FL2VA" if len(files) == 2 else ("I2VA" if len(files) == 1 else "T2VA")
            VIDEO_TRACKER.register(
                job_id,
                {
                    "mode": mode,
                    "prompt": prompt,
                    "width": width,
                    "height": height,
                    "duration": duration,
                    "fps": fps,
                    "steps": steps,
                    "lora": lora_id,
                    "seed": seed,
                },
            )
            return self._send_json(200, {"ok": True, "id": job_id, "status": data.get("status", "queued")})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return self._send_json(e.code, {"ok": False, "error": body or str(e)})
        except Exception as e:
            return self._send_json(502, {"ok": False, "error": str(e)})


def main():
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STORE.poll_once()
    threading.Thread(target=poller_loop, daemon=True).start()
    srv = ThreadingHTTPServer((MONITOR_HOST, MONITOR_PORT), Handler)
    srv.daemon_threads = True
    print(
        f"[gaudiinfer-studio] listening on http://{MONITOR_HOST}:{MONITOR_PORT} "
        f"(manage={MANAGER.can_manage}, gallery={GALLERY_DIR})",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
