#!/bin/sh
# server_ctl.sh start|stop|restart [port] [max_model_len]: run g2e_server in the background as root (pid in /home/a/g2e-data/server.pid,
# log in /mnt/sas/g2e/server.log).
ACTION=${1:-restart}
PORT=${2:-8001}
MAXLEN=${3:-131072}
SLOTS=${4:-8}          # requests resident at once (interleaved step by step)
KVGIB=${5:-0}          # KV cache budget of all slots in GiB (0: the device memory left after loading, less 4)
# weight packs: the RAM copy in /dev/shm/g2e when there (faster loading), else the SAS disk
PACKS=/mnt/sas/g2e
[ -f /dev/shm/g2e/qwen3.8-27b.g2w ] && [ -f /dev/shm/g2e/dspine-b16.g2w ] && [ -f /dev/shm/g2e/vision.g2w ] && PACKS=/dev/shm/g2e
PIDFILE=/home/a/g2e-data/server.pid
cd /home/a/gaudi2-engine
stop() {
    if [ -f $PIDFILE ]; then
        kill "$(cat $PIDFILE)" 2>/dev/null
        for i in $(seq 1 30); do kill -0 "$(cat $PIDFILE)" 2>/dev/null || break; sleep 1; done
        rm -f $PIDFILE
    fi
}
start() {
    export G2E_RECIPE_CACHE=/home/a/g2e-data/recipes HABANA_LOGS=/home/a/g2e-data/habana_logs
    nohup ./build/g2e_server --model-dir /mnt/sas/models/qwen3.8-27b-fp8-mtp \
        --target-pack "$PACKS/qwen3.8-27b.g2w" --draft-pack "$PACKS/dspine-b16.g2w" --vision-pack "$PACKS/vision.g2w" \
        --served-model-name qwen3.8-27b --port "$PORT" --max-model-len "$MAXLEN" --slots "$SLOTS" --kv-cache-gib "$KVGIB" --reasoning-parser qwen3 \
        > /mnt/sas/g2e/server.log 2>&1 < /dev/null &
    echo $! > $PIDFILE
    echo "started pid $(cat $PIDFILE) on port $PORT (max_model_len $MAXLEN, $SLOTS slots, KV budget ${KVGIB} GiB (0: auto), packs from $PACKS)"
}
case "$ACTION" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
esac
