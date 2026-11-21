#!/usr/bin/env python3
"""GaudiInfer real-time monitoring WebUI server (pure Python 3 stdlib, zero pip deps)."""
import collections
import http.client
import json
import os
import pathlib
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENGINE_URL = os.environ.get("G2E_ENGINE_URL", "http://127.0.0.1:8001").rstrip("/")
CONTAINER_NAME = os.environ.get("G2E_CONTAINER_NAME", "gaudiinfer")
MONITOR_HOST = os.environ.get("G2E_MONITOR_HOST", "0.0.0.0")
MONITOR_PORT = int(os.environ.get("G2E_MONITOR_PORT", "8002"))
DOCKER_SOCK = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")
HL_SMI = os.environ.get("HL_SMI_BIN", "/usr/bin/hl-smi")
POLL_SEC = float(os.environ.get("G2E_MONITOR_POLL_SEC", "1.0"))
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "static"

# Example log line from g2e_server.cpp:
# [g2e] prompt 60 (cached 0, 0%; all requests 0.0%) 48 ms | generated 88 in 521 ms (168.9 tok/s) | rounds 26, 3.35 tokens/round | stop
REQ_RE = re.compile(
    r"\[g2e\]\s+prompt\s+(\d+)\s+\(cached\s+(\d+),\s+([\d.]+)%;\s+all requests\s+([\d.]+)%\)\s+"
    r"([\d.]+)\s+ms\s+\|\s+generated\s+(\d+)\s+in\s+([\d.]+)\s+ms\s+\(([\d.]+)\s+tok/s\)\s+\|\s+"
    r"rounds\s+(\d+),\s+([\d.]+)\s+tokens/round\s+\|\s+(\S+)"
)


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float = 3.0):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def docker_get(path: str, timeout: float = 3.0):
    if not os.path.exists(DOCKER_SOCK):
        return None
    conn = UnixHTTPConnection(DOCKER_SOCK, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        body = resp.read()
        if resp.status != 200:
            return None
        return body
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def decode_docker_logs(raw: bytes) -> list:
    """Decode Docker multiplexed log stream (8-byte header frames) or plain text."""
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


def http_status(url: str, timeout: float = 2.0) -> int:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as resp:
        return resp.status


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
        # Read both 54V PSU and 12V PSU rails from `hl-smi -q -d POWER` and sum them
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
                d54 = (p54 or {}).get("Power Draw")
                m54 = (p54 or {}).get("Power Max")
                d12 = (p12 or {}).get("Power Draw")
                m12 = (p12 or {}).get("Power Max")
                hw["power_54v_w"] = d54
                hw["power_54v_max_w"] = m54
                hw["power_12v_w"] = d12
                hw["power_12v_max_w"] = m12
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


class TelemetryStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = collections.deque(maxlen=300)
        self.requests = []
        self.logs = []
        self.engine_banner = {}
        self.latest = {}
        self.host_sampler = HostSampler()
        self._last_docker_ts = 0.0
        self._container_cache = {}
        self._models_cache = None
        self._version_cache = None
        self._prev_live = None  # (ts, decode_tokens, decode_rounds, prefill_tokens)

    def _poll_engine(self):
        eng = {
            "healthy": False,
            "url": ENGINE_URL,
            "version": self._version_cache or "unknown",
            "model": None,
            "max_model_len": None,
            "started_at": None,
            "stats": None,
            "error": None,
        }
        try:
            eng["healthy"] = http_status(f"{ENGINE_URL}/health", timeout=1.5) == 200
        except Exception as e:
            eng["error"] = str(e)
            return eng
        try:
            _, eng["stats"] = http_json(f"{ENGINE_URL}/stats", timeout=1.5)
        except Exception as e:
            eng["error"] = str(e)
        if self._models_cache is None or int(time.time()) % 15 == 0:
            try:
                _, self._models_cache = http_json(f"{ENGINE_URL}/v1/models", timeout=1.5)
            except Exception:
                pass
        if self._models_cache and self._models_cache.get("data"):
            m0 = self._models_cache["data"][0]
            eng["model"] = m0.get("id")
            eng["max_model_len"] = m0.get("max_model_len")
            eng["started_at"] = m0.get("created")
        if self._version_cache is None:
            try:
                _, ver = http_json(f"{ENGINE_URL}/version", timeout=1.5)
                self._version_cache = ver.get("version")
                eng["version"] = self._version_cache
            except Exception:
                pass
        return eng

    def _poll_docker(self):
        now = time.time()
        if now - self._last_docker_ts < 2.0 and self._container_cache:
            return self._container_cache, self.logs, self.requests, self.engine_banner
        self._last_docker_ts = now

        cinfo = {"available": False, "name": CONTAINER_NAME}
        raw_info = docker_get(f"/containers/{urllib.parse.quote(CONTAINER_NAME)}/json", timeout=2.0)
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

        raw_logs = docker_get(
            f"/containers/{urllib.parse.quote(CONTAINER_NAME)}/logs?stdout=1&stderr=1&timestamps=1&tail=450",
            timeout=2.5,
        )
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
        return cinfo, parsed_logs[-250:], parsed_reqs[-100:], banner

    def poll_once(self):
        ts = time.time()
        eng = self._poll_engine()
        hw = collect_hardware()
        host = self.host_sampler.sample()
        cinfo, logs, reqs, banner = self._poll_docker()
        slots = (eng.get("stats") or {}).get("slots") or {}
        pc = (eng.get("stats") or {}).get("prefix_cache") or {}
        raw_live = (eng.get("stats") or {}).get("live") or {}
        live_dec_tps = 0.0
        live_pre_tps = 0.0
        live_tpr = 0.0
        if "decode_tokens" in raw_live:
            cur_dt = int(raw_live.get("decode_tokens", 0))
            cur_dr = int(raw_live.get("decode_rounds", 0))
            cur_pt = int(raw_live.get("prefill_tokens", 0))
            if self._prev_live is not None:
                pts, pdt, pdr, ppt = self._prev_live
                dt = max(0.05, ts - pts)
                if cur_dt >= pdt and cur_pt >= ppt:
                    ddt = cur_dt - pdt
                    ddr = max(0, cur_dr - pdr)
                    dpt = cur_pt - ppt
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
                "live": live,
                "hardware": hw,
                "host": host,
                "container": cinfo,
                "banner": banner,
            }

    def snapshot(self):
        with self.lock:
            snap = dict(self.latest)
            snap["history"] = list(self.history)
            snap["requests"] = list(reversed(self.requests))
            return snap

    def get_logs(self, tail: int = 200):
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


def probe_events(prompt: str, max_tokens: int, temperature: float):
    """Stream a chat completion from the engine; yield delta events, then a final metrics event."""
    payload = {
        "model": (STORE.latest.get("engine") or {}).get("model") or "qwen3.8-27b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, min(int(max_tokens), 1024)),
        "temperature": max(0.0, min(float(temperature), 2.0)),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{ENGINE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_ms = None
    content, reasoning = [], []
    deltas = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=180) as resp:
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


class Handler(BaseHTTPRequestHandler):
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

    def _send_file(self, path: pathlib.Path, ctype: str):
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(min(length, 65536)) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_HEAD(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if parsed.path == "/health" else "text/html; charset=utf-8")
            self.end_headers()
            return
        self.send_error(404, "Not Found")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if parsed.path == "/health":
            return self._send_json(200, {"status": "ok"})
        if parsed.path == "/api/snapshot":
            return self._send_json(200, STORE.snapshot())
        if parsed.path == "/api/logs":
            qs = urllib.parse.parse_qs(parsed.query)
            return self._send_json(200, {"lines": STORE.get_logs(int(qs.get("tail", ["200"])[0]))})
        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ("/api/probe", "/api/probe/stream"):
            return self.send_error(404, "Not Found")
        try:
            req = self._read_json()
            prompt = str(req.get("prompt") or "用一句话介绍 Intel Gaudi2 推理加速卡。").strip()
            max_tokens = int(req.get("max_tokens", 128))
            temperature = float(req.get("temperature", 0.0))
        except Exception as e:
            return self._send_json(400, {"ok": False, "error": f"bad request: {e}"})

        if parsed.path == "/api/probe":
            try:
                final = {}
                for ev in probe_events(prompt, max_tokens, temperature):
                    if ev["type"] == "done":
                        final = ev
                return self._send_json(200, final)
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e)})

        # Server-sent events: relay token deltas live, then the final metrics.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for ev in probe_events(prompt, max_tokens, temperature):
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


def main():
    STORE.poll_once()
    threading.Thread(target=poller_loop, daemon=True).start()
    srv = ThreadingHTTPServer((MONITOR_HOST, MONITOR_PORT), Handler)
    srv.daemon_threads = True
    print(f"[gaudiinfer-monitor] listening on http://{MONITOR_HOST}:{MONITOR_PORT} (engine={ENGINE_URL}, container={CONTAINER_NAME})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
