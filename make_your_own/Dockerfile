# =============================================================================
# llama.cpp RunPod SERVERLESS Dockerfile
# =============================================================================
# Strategy: lightweight image — weights live on a RunPod *network volume* and
# download once (NOT baked in, NOT re-downloaded per cold start). The image
# ships llama.cpp + the RunPod SDK + a thin pass-through handler.
#
# The handler (handler.py) boots `llama-server` inside the worker on a cold
# start, waits for /health, then forwards every job to it. The endpoint is
# OpenAI-compatible at  https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
#
# Base image: official llama.cpp "server" build, CUDA 12. Pin to a specific
# build tag (e.g. :server-cuda-bXXXX) in production. Use :server-cuda13 on a
# CUDA-13 host.
# =============================================================================

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# ---- System + Python deps ----
#   aria2 -> fast model download   |   python3/pip -> run the handler
#   curl  -> debugging             |   runpod -> serverless SDK   |   requests -> forwarding
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        aria2 ca-certificates curl python3 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir --break-system-packages runpod requests

# ---- App files ----
COPY download_models.sh /download_models.sh
COPY handler.py /handler.py
RUN chmod +x /download_models.sh

# ---- Defaults (override in the RunPod endpoint "Environment Variables") ----
ENV R2_BUCKET_URL="[REPLACE ! ENTER THE DOWNLOAD_URL- Such as a cloudflare R2 bucket public endpoint]" \
    MODEL_DIR="/runpod-volume/models" \
    MODEL_FILE="Qwen3.6-27B-uncensored-heretic-v2-Q8_0.gguf" \
    MMPROJ_FILE="Qwen3.6-27B-mmproj-BF16.gguf" \
    MODEL_ALIAS="Qwen3.6-27B-uncensored-heretic-v2" \
    LLAMA_PORT="8080" \
    LLAMA_CTX="62768" \
    LLAMA_NGL="999" \
    MAX_CONCURRENCY="1"
    # Optional:
    #   LLAMA_EXTRA_ARGS="--flash-attn auto --parallel 2"
    #   STARTUP_TIMEOUT / REQUEST_TIMEOUT (seconds)

# IMPORTANT: clear the base image's llama-server ENTRYPOINT and run the handler
# (the handler boots llama-server itself).
ENTRYPOINT ["python3", "-u", "/handler.py"]
