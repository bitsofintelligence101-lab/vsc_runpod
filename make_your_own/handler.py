#!/usr/bin/env python3
# =============================================================================
# handler.py — RunPod Serverless worker for llama.cpp
# =============================================================================
# On worker COLD START (once per worker, before serving any job):
#   1. download_models.sh  (idempotent — no-op if the network volume is warm)
#   2. boot `llama-server` as a local subprocess on 127.0.0.1:8080
#   3. wait until /health is green
#
# Then the handler just FORWARDS jobs to that local server. Because llama-server
# already speaks the OpenAI API (and emits raw OpenAI SSE when streaming), the
# handler is a thin pass-through.
#
# OpenAI-compatible usage (recommended, for agentic / OpenAI-SDK clients):
#   base_url = https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1
#   api_key  = <your RunPod API key>
#   -> RunPod injects job["input"]["openai_route"] + ["openai_input"]
#
# Plain RunPod queue usage (/run, /runsync):
#   {"input": {"messages": [...], "stream": false}}     -> chat
#   {"input": {"prompt": "...", "stream": false}}       -> completion
#
# COLD STARTS: the model load is paid once per worker. Set the endpoint's
# "Idle Timeout" (e.g. 60s) so a burst of agentic calls reuses one warm worker
# and only eats a single cold start. Keep the model on a network volume so the
# 29 GB download happens once, not every cold start.
# =============================================================================

import os
import atexit
import time
import subprocess
from shutil import which

import requests
import runpod

# ---- Config (env-overridable; defaults set in the Dockerfile) ----
LLAMA_HOST       = os.environ.get("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT       = int(os.environ.get("LLAMA_PORT", "8080"))
LLAMA_BASE       = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

MODEL_DIR        = os.environ.get("MODEL_DIR", "/runpod-volume/models")
MODEL_FILE       = os.environ.get("MODEL_FILE", "Qwen3.6-27B-uncensored-heretic-v2-Q8_0.gguf")
MMPROJ_FILE      = os.environ.get("MMPROJ_FILE", "Qwen3.6-27B-mmproj-BF16.gguf")
MODEL_ALIAS      = os.environ.get("MODEL_ALIAS", "Qwen3.6-27B-uncensored-heretic-v2")

LLAMA_CTX        = os.environ.get("LLAMA_CTX", "32768")
LLAMA_NGL        = os.environ.get("LLAMA_NGL", "999")
LLAMA_EXTRA_ARGS = os.environ.get("LLAMA_EXTRA_ARGS", "")

STARTUP_TIMEOUT  = int(os.environ.get("STARTUP_TIMEOUT", "1800"))  # secs to reach /health
REQUEST_TIMEOUT  = int(os.environ.get("REQUEST_TIMEOUT", "1800"))  # per-request ceiling
MAX_CONCURRENCY  = int(os.environ.get("MAX_CONCURRENCY", "1"))     # jobs per worker

_llama_proc = None


# ---------------------------------------------------------------------------
# Boot llama-server (runs once per worker, at import time)
# ---------------------------------------------------------------------------
def _find_llama_bin():
    for p in ("/app/llama-server", "/usr/local/bin/llama-server", "/usr/bin/llama-server"):
        if os.path.exists(p):
            return p
    return which("llama-server") or "/app/llama-server"


def start_llama_server():
    global _llama_proc

    print("[handler] ensuring models are present...", flush=True)
    subprocess.run(["/download_models.sh"], check=True)

    model_path  = os.path.join(MODEL_DIR, MODEL_FILE)
    mmproj_path = os.path.join(MODEL_DIR, MMPROJ_FILE)
    if not os.path.exists(model_path):
        raise RuntimeError(f"model not found: {model_path}")

    args = [
        _find_llama_bin(),
        "-m", model_path,
        "--host", LLAMA_HOST,
        "--port", str(LLAMA_PORT),
        "-c", str(LLAMA_CTX),
        "-ngl", str(LLAMA_NGL),
        "--jinja",                 # embedded chat template + tool calls
        "-a", MODEL_ALIAS,         # clean name in /v1/models
    ]
    if os.path.exists(mmproj_path):
        print(f"[handler] multimodal: attaching mmproj {mmproj_path}", flush=True)
        args += ["--mmproj", mmproj_path]
    else:
        print("[handler] no mmproj found — text-only", flush=True)
    if LLAMA_EXTRA_ARGS.strip():
        args += LLAMA_EXTRA_ARGS.split()

    print(f"[handler] starting: {' '.join(args)}", flush=True)
    _llama_proc = subprocess.Popen(args)

    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if _llama_proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {_llama_proc.returncode})")
        try:
            if requests.get(f"{LLAMA_BASE}/health", timeout=3).status_code == 200:
                print("[handler] llama-server healthy.", flush=True)
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("llama-server did not become healthy within STARTUP_TIMEOUT")


def _cleanup():
    if _llama_proc and _llama_proc.poll() is None:
        _llama_proc.terminate()


atexit.register(_cleanup)


# ---------------------------------------------------------------------------
# Forwarding helpers
# ---------------------------------------------------------------------------
def _forward_nonstream(path, payload, method="POST"):
    if method == "GET":
        r = requests.get(f"{LLAMA_BASE}{path}", timeout=REQUEST_TIMEOUT)
    else:
        r = requests.post(f"{LLAMA_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    try:
        return r.json()
    except ValueError:
        return {"error": r.text, "status_code": r.status_code}


def _forward_stream(path, payload):
    # llama-server emits raw OpenAI SSE ("data: {...}\n\n" ... "data: [DONE]\n\n").
    # We yield those bytes verbatim; RunPod's /openai layer passes them straight
    # through to the client, preserving OpenAI streaming semantics.
    with requests.post(f"{LLAMA_BASE}{path}", json=payload,
                       stream=True, timeout=REQUEST_TIMEOUT) as r:
        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                yield chunk.decode("utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Handler (generator -> supports streaming)
# ---------------------------------------------------------------------------
def handler(job):
    job_input = job.get("input", {}) or {}

    # --- OpenAI-compatible route (hit via /v2/<id>/openai/...) ---
    openai_route = job_input.get("openai_route")
    if openai_route:
        route   = openai_route                      # e.g. /v1/chat/completions
        payload = job_input.get("openai_input") or {}

        # GET-style route: model listing
        if route.rstrip("/").endswith("/models"):
            yield _forward_nonstream("/v1/models", None, method="GET")
            return

        if payload.get("stream"):
            yield from _forward_stream(route, payload)
        else:
            yield _forward_nonstream(route, payload)
        return

    # --- Plain RunPod queue path (/run, /runsync) ---
    if "messages" in job_input:
        path = "/v1/chat/completions"
    elif "prompt" in job_input:
        path = "/v1/completions"
    else:
        yield {"error": "Provide 'messages' or 'prompt' in input, "
                        "or call the /openai/v1/... route."}
        return

    if job_input.get("stream"):
        yield from _forward_stream(path, job_input)
    else:
        yield _forward_nonstream(path, job_input)


# ---------------------------------------------------------------------------
# Cold-start boot, then serve  (runs when executed as the worker entrypoint)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start_llama_server()
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True,
        "concurrency_modifier": lambda current: MAX_CONCURRENCY,
    })
