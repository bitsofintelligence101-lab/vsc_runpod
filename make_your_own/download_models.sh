#!/usr/bin/env bash
# =============================================================================
# download_models.sh — runs once at container startup before llama-server boots
# =============================================================================
# Downloads the GGUF model + mmproj (vision projector) only if they don't
# already exist (idempotent). If MODEL_DIR points at a RunPod network volume,
# this is a no-op on every restart after the first.
#
# Everything is env-driven (see Dockerfile ENV block) so you can swap models
# without rebuilding the image.
# =============================================================================
set -euo pipefail

# Defaults (constants)
readonly DEFAULT_R2_BUCKET_URL="[REPLACE ! YOUR R2 BUCKET URL HERE - such as a cloudflare R2 bucket public endpoint]"
# On RunPod Serverless a network volume mounts at /runpod-volume.
readonly DEFAULT_MODEL_DIR="/runpod-volume/models"
readonly DEFAULT_MODEL_FILE="Qwen3.6-27B-uncensored-heretic-v2-Q8_0.gguf"
readonly DEFAULT_MMPROJ_FILE="Qwen3.6-27B-mmproj-BF16.gguf"

# Effective config (environment can override defaults)
R2_BUCKET_URL="${R2_BUCKET_URL:-$DEFAULT_R2_BUCKET_URL}"
MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
MODEL_FILE="${MODEL_FILE:-$DEFAULT_MODEL_FILE}"
MMPROJ_FILE="${MMPROJ_FILE:-$DEFAULT_MMPROJ_FILE}"

# Multi-connection download. --continue resumes partial files (matters a lot for
# a ~29 GB GGUF if a cold start gets interrupted).
ARIA2_OPTS="--split=16 --max-connection-per-server=16 --min-split-size=50M \
--file-allocation=none --summary-interval=0 --console-log-level=warn --continue=true"

mkdir -p "$MODEL_DIR"

# ---------------------------------------------------------------------------
# dl: download <file> if it doesn't already exist
# ---------------------------------------------------------------------------
dl() {
    local file="$1" label="$2"
    if [ -f "$MODEL_DIR/$file" ]; then
        echo "[skip]     $label ($file already exists)"
        return 0
    fi
    echo "[download] $label → $file"
    # shellcheck disable=SC2086
    aria2c $ARIA2_OPTS --dir="$MODEL_DIR" --out="$file" "$R2_BUCKET_URL/$file"
    echo "[done]     $label"
}

echo "========================================"
echo "  Model Download   (dir: $MODEL_DIR)"
echo "========================================"

# NOTE: unlike the ComfyUI script (many smallish files in parallel), these are
# two very large files. Downloading them sequentially keeps each one at full
# bandwidth instead of halving it. The mmproj is small and quick.
dl "$MODEL_FILE"  "Main model (Q8_0 GGUF)"
dl "$MMPROJ_FILE" "Vision projector (mmproj BF16)"

echo "========================================"
echo "  All models ready in $MODEL_DIR"
echo "========================================"
