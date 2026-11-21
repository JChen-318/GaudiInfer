#!/usr/bin/env bash
set -euo pipefail

BIN="/opt/gaudiinfer/bin/g2e_server"

# Allow passing raw g2e_server CLI flags directly.
if [ "$#" -gt 0 ]; then
    exec "$BIN" "$@"
fi

mkdir -p "${G2E_RECIPE_CACHE:-/data/recipes}" "${HABANA_LOGS:-/data/habana_logs}"

# Prefer RAM-cached weight packs in /dev/shm/g2e when all required packs exist;
# otherwise fall back to disk-mounted G2E_PACKS_DIR (/packs).
PACKS_DIR="${G2E_PACKS_DIR:-/packs}"
SHM_DIR="${G2E_SHM_PACKS_DIR:-/dev/shm/g2e}"
TARGET_NAME="${G2E_TARGET_PACK_NAME:-qwen3.8-27b.g2w}"
DRAFT_NAME="${G2E_DRAFT_PACK_NAME:-dspine-b16.g2w}"
VISION_NAME="${G2E_VISION_PACK_NAME:-vision.g2w}"

if [ -z "${G2E_TARGET_PACK:-}" ] && [ -f "${SHM_DIR}/${TARGET_NAME}" ]; then
    if { [ -z "${DRAFT_NAME}" ] || [ -f "${SHM_DIR}/${DRAFT_NAME}" ]; } && \
       { [ -z "${VISION_NAME}" ] || [ -f "${SHM_DIR}/${VISION_NAME}" ]; }; then
        PACKS_DIR="${SHM_DIR}"
    fi
fi

TARGET_PACK="${G2E_TARGET_PACK:-${PACKS_DIR}/${TARGET_NAME}}"
DRAFT_PACK="${G2E_DRAFT_PACK:-${PACKS_DIR}/${DRAFT_NAME}}"
VISION_PACK="${G2E_VISION_PACK:-${PACKS_DIR}/${VISION_NAME}}"

ARGS=(
    --model-dir "${G2E_MODEL_DIR:-/models/qwen3.8-27b-fp8-mtp}"
    --target-pack "${TARGET_PACK}"
    --served-model-name "${G2E_SERVED_MODEL_NAME:-qwen3.8-27b}"
    --host "${G2E_HOST:-0.0.0.0}"
    --port "${G2E_PORT:-8001}"
    --max-model-len "${G2E_MAX_MODEL_LEN:-262144}"
    --slots "${G2E_SLOTS:-8}"
    --kv-cache-gib "${G2E_KV_CACHE_GIB:-0}"
)

if [ -n "${DRAFT_PACK}" ] && [ -f "${DRAFT_PACK}" ]; then
    ARGS+=(--draft-pack "${DRAFT_PACK}")
fi
if [ -n "${VISION_PACK}" ] && [ -f "${VISION_PACK}" ]; then
    ARGS+=(--vision-pack "${VISION_PACK}")
fi
if [ -n "${G2E_REASONING_PARSER:-qwen3}" ]; then
    ARGS+=(--reasoning-parser "${G2E_REASONING_PARSER:-qwen3}")
fi
if [ -n "${G2E_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${G2E_EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

echo "[gaudiinfer] starting g2e_server (packs=${PACKS_DIR}, port=${G2E_PORT:-8001}, max_model_len=${G2E_MAX_MODEL_LEN:-262144}, slots=${G2E_SLOTS:-8})"
exec "$BIN" "${ARGS[@]}"
