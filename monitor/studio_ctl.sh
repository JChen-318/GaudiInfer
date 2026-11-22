#!/usr/bin/env bash
# Manage the unified GaudiInfer Studio & Monitor WebUI on host :8002.
# Usage: monitor/studio_ctl.sh start|stop|restart|status
set -euo pipefail

ACTION="${1:-restart}"
REPO_DIR="${G2E_REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
DATA_DIR="${G2E_DATA_DIR:-${HOME}/g2e-data}"
GALLERY_DIR="${G2E_STUDIO_OUTPUT_DIR:-/mnt/sas/outputs/g2e_studio}"
PIDFILE="${DATA_DIR}/studio_webui.pid"
LOGFILE="/tmp/g2e_studio_webui.log"

stop_studio() {
    # Stop legacy docker monitor container if it holds :8002
    if command -v docker >/dev/null 2>&1; then
        docker rm -f gaudiinfer-monitor >/dev/null 2>&1 || true
    fi
    if [ -f "$PIDFILE" ]; then
        PID="$(cat "$PIDFILE")"
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            for _ in $(seq 1 10); do
                kill -0 "$PID" 2>/dev/null || break
                sleep 0.3
            done
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
    fi
    for pid in $(pgrep -f '^python3.*monitor/server\.py' 2>/dev/null || true); do
        [ "$pid" = "$$" ] && continue
        kill "$pid" 2>/dev/null || true
    done
}

start_studio() {
    mkdir -p "$DATA_DIR" "$GALLERY_DIR" /tmp/g2e_studio_logs
    cd "$REPO_DIR"
    export G2E_REPO_DIR="$REPO_DIR" G2E_DATA_DIR="$DATA_DIR" G2E_STUDIO_OUTPUT_DIR="$GALLERY_DIR"
    nohup python3 -u "$REPO_DIR/monitor/server.py" > "$LOGFILE" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 20); do
        if curl -sf http://127.0.0.1:8002/health >/dev/null 2>&1; then
            echo "[studio] running pid=$(cat "$PIDFILE") on http://0.0.0.0:8002 (log=$LOGFILE)"
            return 0
        fi
        kill -0 "$(cat "$PIDFILE")" 2>/dev/null || break
        sleep 0.5
    done
    echo "[studio] failed to start! Log:"
    cat "$LOGFILE"
    exit 1
}

case "$ACTION" in
    start)   start_studio ;;
    stop)    stop_studio ;;
    restart) stop_studio; start_studio ;;
    status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "[studio] active pid=$(cat "$PIDFILE")"
            curl -sf http://127.0.0.1:8002/health || true
            echo ""
        else
            echo "[studio] stopped"
            exit 1
        fi
        ;;
    *) echo "Usage: $0 start|stop|restart|status"; exit 1 ;;
esac
