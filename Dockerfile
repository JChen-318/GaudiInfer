# Multi-stage production Dockerfile for GaudiInfer (g2e_server) on Intel Gaudi2.
# Pure C++ runtime on Ubuntu 24.04 + Habana SynapseAI (no PyTorch / Python bloat).
#
# Build:
#   docker build -t gaudiinfer:latest .
#   docker build --build-arg HABANA_VERSION=1.24.1-482 -t gaudiinfer:latest .

ARG HABANA_VERSION=1.24.1-482

# ---- Shared base: Ubuntu 24.04 + Habana SynapseAI C++ runtime ----
FROM ubuntu:24.04 AS base

ARG HABANA_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public \
        | gpg --dearmor -o /usr/share/keyrings/habanalabs.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/habanalabs.gpg] https://vault.habana.ai/artifactory/debian noble main" \
        > /etc/apt/sources.list.d/habanalabs.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        habanalabs-rdma-core=${HABANA_VERSION} \
        habanalabs-thunk=${HABANA_VERSION} \
        habanalabs-graph=${HABANA_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# ---- Builder: add C++ toolchain + tpc-clang, build g2e_server + TPC kernels ----
FROM base AS builder

ARG HABANA_VERSION
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake ninja-build g++ binutils libpng-dev \
        habanalabs-tools=${HABANA_VERSION} \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target g2e_server g2e_h3_server g2e_h3_cli g2e_qwen21_server g2e_h3_image_server -j"$(nproc)" \
    && mkdir -p /stage/opt/gaudiinfer/bin /stage/opt/gaudiinfer/lib /stage/opt/gaudiinfer/tools \
    && cp build/g2e_server build/g2e_h3_server build/g2e_h3_cli build/g2e_qwen21_server build/g2e_h3_image_server /stage/opt/gaudiinfer/bin/ \
    && cp build/tpc/flashinfer_gaudi/libflashinfer_gaudi_kernels.so /stage/opt/gaudiinfer/lib/ \
    && find build/tpc/flashinfer_gaudi -maxdepth 1 -name '*.o' ! -name '*_embedded.o' \
        -exec cp {} /stage/opt/gaudiinfer/lib/ \; \
    && cp tools/scal-g2e-direct.json tools/scal-g2e-ink.json /stage/opt/gaudiinfer/tools/ \
    && cp -r tools/mme_descs_R128 tools/mme_descs_R16 tools/mme_descs_h3 tools/mme_descs_ink /stage/opt/gaudiinfer/tools/ \
    && cp docker/entrypoint.sh /stage/opt/gaudiinfer/entrypoint.sh \
    && sed -i 's/\r$//' /stage/opt/gaudiinfer/entrypoint.sh \
    && chmod +x /stage/opt/gaudiinfer/entrypoint.sh

# ---- Runtime: minimal SynapseAI C++ image without build tools or tpc-clang ----
FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl3t64-gnutls libpng16-16t64 libwebp7 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data/recipes /data/habana_logs

COPY --from=builder /stage/opt/gaudiinfer /opt/gaudiinfer

ENV G2E_TPC_LIB=/opt/gaudiinfer/lib/libflashinfer_gaudi_kernels.so \
    G2E_TPC_OBJ_DIR=/opt/gaudiinfer/lib \
    G2E_MME_DESC_DIR=/opt/gaudiinfer/tools/mme_descs_R128 \
    G2E_MME_DESC_DIR_R16=/opt/gaudiinfer/tools/mme_descs_R16 \
    G2E_H3_MME_DESC_DIR=/opt/gaudiinfer/tools/mme_descs_h3 \
    G2E_INK_MME_DIR=/opt/gaudiinfer/tools/mme_descs_ink \
    SCAL_CONFIG_FILE_PATH=/opt/gaudiinfer/tools/scal-g2e-direct.json \
    SCAL_CFG_FILE_PATH=/opt/gaudiinfer/tools/scal-g2e-direct.json \
    HABANA_SCAL_BIN_PATH=/opt/habanalabs/engines_fw \
    HABANA_PLUGINS_LIB_PATH=/opt/habanalabs/habana_plugins \
    G2E_RECIPE_CACHE=/data/recipes \
    HABANA_LOGS=/data/habana_logs \
    LD_LIBRARY_PATH=/usr/lib/habanalabs

WORKDIR /opt/gaudiinfer
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${G2E_PORT:-8001}/health" || exit 1

ENTRYPOINT ["/opt/gaudiinfer/entrypoint.sh"]
