# ollama_llama_proxy.py  (RunPod Serverless edition)
#
# PURPOSE: VS Code Copilot (as of May 2026) only supports Ollama as a local model
# provider. This proxy runs LOCALLY, pretends to BE Ollama on port 11434, and
# forwards generation requests to your RunPod Serverless endpoint's
# OpenAI-compatible API.
#
# Flow:
#   VS Code Copilot  -->  this proxy (localhost:11434, speaks Ollama API)
#                              |
#                              v
#         https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/...  (Bearer auth)
#                              |
#                              v
#                  RunPod worker -> llama-server (OpenAI-compatible)
#
# IMPORTANT — cost control:
#   Every request forwarded to RunPod can wake a (billable) GPU worker. So this
#   proxy answers all of Ollama's METADATA probes LOCALLY (/api/tags, /api/version,
#   /api/show, /v1/models) and only forwards actual GENERATION
#   (/api/chat, /api/generate, /v1/chat/completions, /v1/responses). The model
#   name / context length are taken from config below, NOT probed from the server.
#
# Setup:
#   export RUNPOD_OPENAI_BASE="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1"
#   export RUNPOD_API_KEY="<your RunPod API key>"
#   python3 ollama_llama_proxy.py
#   VS Code: Settings -> "GitHub Copilot: Local Provider" -> Ollama (localhost:11434)

import os
import json
import time
import queue
import threading
import itertools
import http.client
import http.server
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime


ENV_FILE = ".env"


def _strip_quotes(value):
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def load_dotenv_file(path=ENV_FILE):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                if key in os.environ:
                    continue
                os.environ[key] = _strip_quotes(value)
    except Exception as e:
        print(f"[proxy] warning: failed reading {path}: {e}")


def _quote_env_value(value):
    return '"' + value.replace('"', '\\"') + '"'


def write_env_values(updates, path=ENV_FILE):
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    remaining = dict(updates)
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue

        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            out.append(f"{key}={_quote_env_value(remaining.pop(key))}")
        else:
            out.append(line)

    if out and out[-1].strip() != "":
        out.append("")

    for key, value in remaining.items():
        out.append(f"{key}={_quote_env_value(value)}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out).rstrip() + "\n")


def normalize_runpod_base(endpoint_or_base):
    value = (endpoint_or_base or "").strip().rstrip("/")
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        if value.endswith("/v1"):
            return value[:-3].rstrip("/")
        return value
    return f"https://api.runpod.ai/v2/{value}/openai"


def endpoint_id_from_base_or_endpoint(endpoint_or_base):
    value = (endpoint_or_base or "").strip().rstrip("/")
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        parts = [p for p in urllib.parse.urlparse(value).path.split("/") if p]
        try:
            i = parts.index("v2")
            if i + 1 < len(parts):
                return parts[i + 1]
        except ValueError:
            return ""
        return ""
    return value


def is_config_ready():
    if not RUNPOD_API_KEY.strip():
        return False
    if not RUNPOD_BASE.strip():
        return False
    if "REPLACE_ENDPOINT_ID" in RUNPOD_BASE:
        return False
    return True


def apply_runtime_config(api_key, endpoint_or_base, enable_vision=None):
    global RUNPOD_API_KEY, RUNPOD_BASE, ENABLE_VISION
    RUNPOD_API_KEY = (api_key or "").strip()
    RUNPOD_BASE = normalize_runpod_base(endpoint_or_base)
    if enable_vision is not None:
        ENABLE_VISION = enable_vision


def current_endpoint_display():
    eid = endpoint_id_from_base_or_endpoint(RUNPOD_BASE)
    return eid or RUNPOD_BASE


def config_page_html(message="", error=False):
    status_text = "ready" if is_config_ready() else "waiting for configuration"
    status_color = "#0f766e" if is_config_ready() else "#b45309"
    notice = ""
    if message:
        box_bg = "#dcfce7" if not error else "#fee2e2"
        box_fg = "#14532d" if not error else "#7f1d1d"
        notice = (
            f"<div style='margin:12px 0;padding:10px 12px;border-radius:8px;"
            f"background:{box_bg};color:{box_fg};font-size:14px'>{message}</div>"
        )

    endpoint_value = current_endpoint_display().replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    masked_key = RUNPOD_API_KEY[:6] + ("..." if RUNPOD_API_KEY else "")
    model_alias_value = MODEL_ALIAS.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    quant_value = DEFAULT_QUANTIZATION.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    context_value = str(CONTEXT_LEN)
    request_timeout_value = "" if REQUEST_TIMEOUT is None else str(REQUEST_TIMEOUT)
    drip_interval_value = str(DRIP_INTERVAL)
    drip_token_value = DRIP_TOKEN.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")
    vision_checked = "checked" if ENABLE_VISION else ""

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>RunPod Proxy Config</title>
  <style>
    :root {{ --bg:#f8fafc; --card:#ffffff; --line:#e2e8f0; --text:#0f172a; --muted:#475569; --brand:#0f766e; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Segoe UI, Tahoma, sans-serif; background:var(--bg); color:var(--text); }}
    .wrap {{ max-width:620px; margin:32px auto; padding:0 14px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; box-shadow:0 6px 24px rgba(15,23,42,.06); }}
    h1 {{ margin:0 0 8px; font-size:22px; }}
    p {{ margin:0 0 14px; color:var(--muted); }}
    .status {{ margin:8px 0 16px; font-weight:600; color:{status_color}; }}
    label {{ display:block; margin:12px 0 6px; font-size:14px; font-weight:600; }}
    input {{ width:100%; padding:10px 11px; border:1px solid var(--line); border-radius:8px; font-size:14px; }}
    button {{ margin-top:14px; background:var(--brand); color:#fff; border:0; border-radius:8px; padding:10px 14px; font-weight:600; cursor:pointer; }}
    .hint {{ margin-top:12px; font-size:13px; color:var(--muted); }}
    code {{ background:#f1f5f9; padding:1px 5px; border-radius:5px; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <div class=\"card\">
      <h1>Proxy Configuration</h1>
      <p>Set your RunPod values once. They will be saved to <code>.env</code>.</p>
      <div class=\"status\">Status: {status_text}</div>
      {notice}
      <form method=\"post\" action=\"/config\">
        <label for=\"api_key\">RunPod API Key</label>
        <input id=\"api_key\" name=\"api_key\" type=\"password\" placeholder=\"rpa_...\" autocomplete=\"off\" />
        <div class=\"hint\">Current key: {masked_key or "not set"}</div>

        <label for=\"runpod_endpoint\">RunPod Endpoint ID or OpenAI Base URL</label>
        <input id=\"runpod_endpoint\" name=\"runpod_endpoint\" value=\"{endpoint_value}\" placeholder=\"4zymmya1e6kh25 or https://api.runpod.ai/v2/.../openai/v1\" />
        <div class=\"hint\">Examples: <code>4zymmya1e6kh25</code> or full URL.</div>

                <label for=\"model_alias\">Model Alias</label>
                <input id=\"model_alias\" name=\"model_alias\" value=\"{model_alias_value}\" placeholder=\"Qwen3.6-27B-uncensored-heretic-v2\" />
                <div class=\"hint\">Advertised model name before the quantization suffix is appended.</div>

                <label for=\"default_quantization\">Default Quantization</label>
                <input id=\"default_quantization\" name=\"default_quantization\" value=\"{quant_value}\" placeholder=\"Q8_0\" />
                <div class=\"hint\">Combined with the model alias to form the Ollama-visible model tag.</div>

                <label for=\"llama_ctx\">LLAMA_CTX</label>
                <input id=\"llama_ctx\" name=\"llama_ctx\" value=\"{context_value}\" placeholder=\"64000\" />
                <div class=\"hint\">Context length advertised in local metadata responses.</div>

                <label for=\"request_timeout\">REQUEST_TIMEOUT</label>
                <input id=\"request_timeout\" name=\"request_timeout\" value=\"{request_timeout_value}\" placeholder=\"leave blank for no timeout\" />
                <div class=\"hint\">Optional upstream timeout in seconds. Leave blank to allow long cold starts.</div>

                <label for=\"drip_interval\">DRIP_INTERVAL</label>
                <input id=\"drip_interval\" name=\"drip_interval\" value=\"{drip_interval_value}\" placeholder=\"5\" />
                <div class=\"hint\">Seconds between keepalive drip chunks during cold start.</div>

                <label for=\"drip_token\">DRIP_TOKEN</label>
                <input id=\"drip_token\" name=\"drip_token\" value=\"{drip_token_value}\" placeholder=\".\" />
                <div class=\"hint\">Placeholder token emitted while waiting for the first real streamed bytes.</div>
                <label style="display:flex;align-items:center;gap:8px;font-weight:600;font-size:14px;margin-top:14px">
                  <input id="enable_vision" name="enable_vision" type="checkbox" {vision_checked} style="width:auto" />
                  Enable Vision (advertise model as multimodal)
                </label>
                <div class="hint">Adds <code>&quot;vision&quot;</code> capability so VS Code Copilot shows this model for image inputs.</div>
        <button type=\"submit\">Save Configuration</button>
      </form>
      <div class=\"hint\" style=\"margin-top:16px\">Then use VS Code Copilot local Ollama at <code>http://localhost:{PROXY_PORT}</code>.</div>
    </div>
  </div>
</body>
</html>
"""


def send_html(handler, html, status=200):
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


load_dotenv_file()

# --- Configuration (env-overridable) ---
PROXY_PORT = int(os.environ.get("PROXY_PORT", "11434"))   # Ollama's default port

# RunPod Serverless OpenAI base. You can paste either ".../openai" or ".../openai/v1";
# we normalise to the part WITHOUT the trailing /v1 so the /v1/... paths below append cleanly.
_raw_base = os.environ.get(
    "RUNPOD_OPENAI_BASE",
    "https://api.runpod.ai/v2/REPLACE_ENDPOINT_ID/openai/v1",
).rstrip("/")
RUNPOD_BASE = _raw_base[:-3].rstrip("/") if _raw_base.endswith("/v1") else _raw_base

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# Model metadata shown to VS Code. Static config — no server probe.
MODEL_ALIAS         = os.environ.get("MODEL_ALIAS", "Qwen3.6-27B-Q8-heretic-v2")
DEFAULT_QUANTIZATION = os.environ.get("DEFAULT_QUANTIZATION", "Q8_0")
MODEL_NAME          = f"{MODEL_ALIAS}:{DEFAULT_QUANTIZATION}"   # Ollama needs 'name:tag'
CONTEXT_LEN         = int(os.environ.get("LLAMA_CTX", "64000"))

# No request timeout by default: the first call may trigger a cold start (model
# load) that takes a while. Set REQUEST_TIMEOUT (seconds) to cap it if you prefer.
_t = os.environ.get("REQUEST_TIMEOUT")
REQUEST_TIMEOUT = int(_t) if _t else None

# Vision capability — advertise the model as multimodal so VS Code Copilot
# surfaces it as a vision model in addition to a tool model.
ENABLE_VISION = os.environ.get("ENABLE_VISION", "true").lower() not in ("0", "false", "no")

# --- Cold-start keepalive ("drip") ---
# While RunPod spins up a worker (model load can take much longer than the
# client's ~30s patience), no bytes flow and VS Code aborts with "Response
# contained no choices". To prevent that, we emit a valid choices-bearing chunk
# immediately, then drip a placeholder token every DRIP_INTERVAL seconds until
# the real tokens arrive. The placeholder DOES enter the output/context.
#   DRIP_INTERVAL : seconds between drips (keep comfortably under ~30s)
#   DRIP_TOKEN    : the placeholder text. "." by default. Set to "" to drip
#                   empty (choices-bearing but zero-content) chunks instead —
#                   no context pollution, if your client tolerates them.
#TODO: Reconsider this? It doesn't work exactly as intended and letting it fail normally may be better and more consistent with what vs code expects, since it will default to showing a 'try again' message.
DRIP_INTERVAL = float(os.environ.get("DRIP_INTERVAL", "5"))
DRIP_TOKEN    = os.environ.get("DRIP_TOKEN", ".")

# --- IncompleteRead retry ---
# RunPod can close the connection with 0 bytes during a cold start before the
# worker is ready. Retry up to this many times (with a short backoff) before
# surfacing the error to VS Code.
INITIAL_INCOMPLETE_READ_RETRIES = int(os.environ.get("INITIAL_INCOMPLETE_READ_RETRIES", "3"))
INITIAL_INCOMPLETE_READ_BACKOFF = float(os.environ.get("INITIAL_INCOMPLETE_READ_BACKOFF", "1.5"))

_chunk_id_counter = itertools.count()
SHUTDOWN_EVENT = threading.Event()
_shutdown_started = threading.Event()
_active_upstreams = set()
_active_upstreams_lock = threading.Lock()


def _track_upstream(resp):
    with _active_upstreams_lock:
        _active_upstreams.add(resp)


def _untrack_upstream(resp):
    with _active_upstreams_lock:
        _active_upstreams.discard(resp)


def _close_active_upstreams():
    with _active_upstreams_lock:
        upstreams = list(_active_upstreams)
        _active_upstreams.clear()

    for resp in upstreams:
        try:
            resp.close()
        except Exception:
            pass


def shutdown_server(server, reason="shutdown requested"):
    if _shutdown_started.is_set():
        return

    _shutdown_started.set()
    SHUTDOWN_EVENT.set()
    print(f"[proxy] initiating shutdown ({reason})")
    _close_active_upstreams()

    try:
        server.shutdown()
    except Exception as e:
        print(f"[proxy] shutdown() warning: {e}")

    try:
        server.server_close()
    except Exception as e:
        print(f"[proxy] server_close() warning: {e}")

    print("[proxy] shutdown complete")


def get_model_info():
    """Static (name, quant, context) — no backend probe, so polling never wakes a worker."""
    return MODEL_NAME, DEFAULT_QUANTIZATION, CONTEXT_LEN


def _runpod_headers():
    h = {"Content-Type": "application/json"}
    if RUNPOD_API_KEY:
        h["Authorization"] = f"Bearer {RUNPOD_API_KEY}"
    return h


def forward_to_runpod(path, body):
    """Non-streaming POST to RunPod's OpenAI endpoint -> (status_code, response_text)."""
    url = f"{RUNPOD_BASE}{path}"
    data = json.dumps(body).encode()
    print(f"[proxy] -> RunPod POST {url}  (non-stream)")
    print(f"[proxy]    body: {json.dumps(body)[:300]}")
    req = urllib.request.Request(url, data=data, headers=_runpod_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            text = resp.read().decode()
            print(f"[proxy] <- RunPod {resp.status}  response: {text[:300]}")
            return resp.status, text
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="ignore")
        print(f"[proxy] <- RunPod HTTPError {e.code}: {text[:300]}")
        return e.code, text


def _sse(obj):
    """Serialize a dict as one SSE event."""
    return f"data: {json.dumps(obj)}\n\n".encode()


def _chunk(model, cid, delta=None, finish=None):
    """Build a minimal OpenAI chat.completion.chunk."""
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }


def forward_stream_to_runpod(path, body, wfile):
    """
    Stream from RunPod's OpenAI endpoint to the client, with a cold-start
    keepalive drip.

    A background thread connects to RunPod (which may block for the whole cold
    start before any byte arrives). Meanwhile this thread:
      1. sends an immediate role chunk so 'choices' exist right away,
      2. drips a placeholder token every DRIP_INTERVAL seconds *until* the first
         real byte arrives (safe: nothing partial has been written yet),
      3. once real bytes flow, forwards RunPod's raw SSE verbatim and NEVER
         injects again (injecting mid-event would corrupt the stream).
    """
    url = f"{RUNPOD_BASE}{path}"
    data = json.dumps(body).encode()
    model = body.get("model", MODEL_NAME)
    cid = f"chatcmpl-proxy-{next(_chunk_id_counter)}"
    print(f"[proxy] -> RunPod POST {url}  (streaming, drip every {DRIP_INTERVAL}s)")
    print(f"[proxy]    body: {json.dumps(body)[:300]}")

    q = queue.Queue()
    stop = threading.Event()

    def reader():
        attempts_left = INITIAL_INCOMPLETE_READ_RETRIES + 1
        while attempts_left > 0 and not SHUTDOWN_EVENT.is_set() and not stop.is_set():
            attempts_left -= 1
            req = urllib.request.Request(url, data=data, headers=_runpod_headers(), method="POST")
            tracked_resp = None
            retry = False
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    tracked_resp = resp
                    _track_upstream(resp)
                    while not stop.is_set():
                        if SHUTDOWN_EVENT.is_set():
                            break
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        q.put(("bytes", chunk))
            except http.client.IncompleteRead as e:
                partial = getattr(e, "partial", b"") or b""
                if not partial and attempts_left > 0 and not SHUTDOWN_EVENT.is_set():
                    print(f"[proxy] IncompleteRead(0 bytes), retrying "
                          f"({INITIAL_INCOMPLETE_READ_RETRIES - attempts_left + 1}/"
                          f"{INITIAL_INCOMPLETE_READ_RETRIES})"
                          f" in {INITIAL_INCOMPLETE_READ_BACKOFF}s")
                    retry = True
                elif not SHUTDOWN_EVENT.is_set():
                    q.put(("error", str(e)))
            except urllib.error.HTTPError as e:
                if not SHUTDOWN_EVENT.is_set():
                    q.put(("error", f"HTTP {e.code}: {e.read().decode(errors='ignore')[:500]}"))
            except Exception as e:  # noqa: BLE001 — surface any upstream failure to the client
                if not SHUTDOWN_EVENT.is_set():
                    q.put(("error", str(e)))
            finally:
                if tracked_resp is not None:
                    _untrack_upstream(tracked_resp)
                if not retry:
                    q.put(("done", None))
                    return
            # retry path — sleep outside finally so done isn't put prematurely
            time.sleep(INITIAL_INCOMPLETE_READ_BACKOFF)
        # Safety net: if the while loop exits because SHUTDOWN_EVENT / stop was set
        # between retry iterations, make sure the main thread can unblock.
        q.put(("done", None))

    threading.Thread(target=reader, daemon=True).start()

    def safe_write(b):
        try:
            wfile.write(b)
            wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            stop.set()
            return False

    # 1. immediate role chunk -> client sees 'choices' instantly
    if not safe_write(_sse(_chunk(model, cid, delta={"role": "assistant", "content": ""}))):
        return

    real_started = False
    drips = 0
    try:
        while True:
            if SHUTDOWN_EVENT.is_set():
                break

            if real_started:
                kind, payload = q.get()                 # pump raw bytes, no injection
            else:
                try:
                    kind, payload = q.get(timeout=DRIP_INTERVAL)
                except queue.Empty:
                    # still cold — drip a placeholder to reset the client's timer
                    delta = {"content": DRIP_TOKEN} if DRIP_TOKEN else {"content": ""}
                    if not safe_write(_sse(_chunk(model, cid, delta=delta))):
                        return
                    drips += 1
                    continue

            if kind == "bytes":
                if not real_started:
                    real_started = True
                    print(f"[proxy] first real bytes after {drips} drip(s); piping raw SSE")
                if not safe_write(payload):
                    return
            elif kind == "error":
                if not real_started:
                    msg = f"[proxy: upstream error] {payload} \n The runpod worker likely failed to start. This can happen if the model files fail to load or if the worker runs out of memory. You can check your RunPod endpoint's logs for more details. If this happens consistently, consider trying a smaller model or checking your RunPod resource limits. Large context window is usually the issue, stay under 64k unless you're using the 80gb+ GPUs\n\nOr just start a new chat and retry if this was your first request or it's been awhile"
                    safe_write(_sse(_chunk(model, cid, delta={"content": msg})))
                    safe_write(_sse(_chunk(model, cid, finish="stop")))
                print(f"[proxy] <- RunPod error: {payload}")
                break
            elif kind == "done":
                break
    finally:
        stop.set()
        # Always terminate the SSE stream. If RunPod already sent [DONE] in the
        # raw bytes, a second one is harmless (clients stop at the first).
        safe_write(b"data: [DONE]\n\n")
        print(f"[proxy] <- stream closed (real_started={real_started}, drips={drips})")


def responses_to_chat_completions(parsed):
    """
    Translate an OpenAI Responses API request body into a Chat Completions body.
      - "input" (string or list) instead of "messages"
      - "instructions" instead of a system message
    """
    messages = []

    instructions = parsed.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    input_field = parsed.get("input", [])
    if isinstance(input_field, str):
        messages.append({"role": "user", "content": input_field})
    elif isinstance(input_field, list):
        for item in input_field:
            role = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                content = "\n".join(text_parts)
            messages.append({"role": role, "content": content})

    chat_body = {
        "model": parsed.get("model", MODEL_NAME),
        "messages": messages,
        "stream": parsed.get("stream", False),
    }
    for key in ("temperature", "top_p", "max_tokens", "stop"):
        if key in parsed:
            chat_body[key] = parsed[key]
    return chat_body


class OllamaProxyHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[proxy] {self.command} {self.path} -> {args[0] if args else ''}")

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    # ------------------------------------------------------------------
    # GET handlers — ALL answered locally (never forwarded -> never wakes a worker)
    # ------------------------------------------------------------------
    def do_GET(self):
        if SHUTDOWN_EVENT.is_set():
            self.send_json(503, {"error": "Server is shutting down"})
            return

        if self.path == "/config":
            send_html(self, config_page_html())
            return

        if not is_config_ready():
            self.send_json(503, {
                "error": "Proxy not configured yet. Open /config and set RUNPOD_API_KEY and endpoint.",
                "config_url": f"http://localhost:{PROXY_PORT}/config",
            })
            return

        model_name, quant, _ = get_model_info()
        print(f"[proxy] GET {self.path}")

        if self.path in ("/", "/api/version"):
            self.send_json(200, {"version": "0.6.4"})

        elif self.path == "/api/tags":
            families = ["qwen36"]
            if ENABLE_VISION:
                families.append("clip")
            self.send_json(200, {
                "models": [{
                    "name": model_name,
                    "model": model_name,
                    "modified_at": datetime.utcnow().isoformat() + "Z",
                    "size": 22000000000,
                    "digest": "aaaaaaaaaaaaaaaa",
                    "details": {
                        "parent_model": "", "format": "gguf",
                        "family": "qwen36", "families": families,
                        "parameter_size": "27B", "quantization_level": quant,
                    },
                }]
            })

        elif self.path == "/v1/models":
            self.send_json(200, {
                "object": "list",
                "data": [{
                    "id": model_name, "object": "model",
                    "created": int(datetime.utcnow().timestamp()), "owned_by": "local",
                }],
            })

        else:
            print(f"[proxy]   -> UNHANDLED GET '{self.path}'")
            self.send_json(404, {"error": f"Unhandled GET: {self.path}"})

    # ------------------------------------------------------------------
    # POST handlers
    # ------------------------------------------------------------------
    def do_POST(self):
        if SHUTDOWN_EVENT.is_set():
            self.send_json(503, {"error": "Server is shutting down"})
            return

        if self.path == "/config":
            length = int(self.headers.get("Content-Length", 0))
            form_data = self.rfile.read(length).decode("utf-8", errors="ignore") if length else ""
            fields = urllib.parse.parse_qs(form_data)

            api_key = (fields.get("api_key", [""])[0] or "").strip()
            endpoint = (fields.get("runpod_endpoint", [""])[0] or "").strip()
            model_alias = (fields.get("model_alias", [""])[0] or MODEL_ALIAS).strip()
            default_quantization = (fields.get("default_quantization", [""])[0] or DEFAULT_QUANTIZATION).strip()
            llama_ctx = (fields.get("llama_ctx", [""])[0] or str(CONTEXT_LEN)).strip()
            request_timeout = (fields.get("request_timeout", [""])[0] or "").strip()
            drip_interval = (fields.get("drip_interval", [""])[0] or str(DRIP_INTERVAL)).strip()
            drip_token = fields.get("drip_token", [DRIP_TOKEN])[0]
            enable_vision = "enable_vision" in fields

            if not api_key and not RUNPOD_API_KEY:
                send_html(self, config_page_html("API key is required.", error=True), status=400)
                return
            if not endpoint and not RUNPOD_BASE:
                send_html(self, config_page_html("RunPod endpoint is required.", error=True), status=400)
                return
            if not model_alias:
                send_html(self, config_page_html("Model alias is required.", error=True), status=400)
                return
            if not default_quantization:
                send_html(self, config_page_html("Default quantization is required.", error=True), status=400)
                return

            try:
                int(llama_ctx)
            except ValueError:
                send_html(self, config_page_html("LLAMA_CTX must be an integer.", error=True), status=400)
                return

            if request_timeout:
                try:
                    int(request_timeout)
                except ValueError:
                    send_html(self, config_page_html("REQUEST_TIMEOUT must be an integer when set.", error=True), status=400)
                    return

            try:
                float(drip_interval)
            except ValueError:
                send_html(self, config_page_html("DRIP_INTERVAL must be a number.", error=True), status=400)
                return

            final_key = api_key or RUNPOD_API_KEY
            final_endpoint = endpoint or RUNPOD_BASE
            final_base = normalize_runpod_base(final_endpoint)

            if not final_base:
                send_html(self, config_page_html("RunPod endpoint value is invalid.", error=True), status=400)
                return

            endpoint_id = endpoint_id_from_base_or_endpoint(final_endpoint)
            updates = {
                "RUNPOD_API_KEY": final_key,
                "RUNPOD_OPENAI_BASE": f"{final_base}/v1",
                "MODEL_ALIAS": model_alias,
                "DEFAULT_QUANTIZATION": default_quantization,
                "LLAMA_CTX": llama_ctx,
                "DRIP_INTERVAL": drip_interval,
                "DRIP_TOKEN": drip_token,
                "ENABLE_VISION": "true" if enable_vision else "false",
            }
            if request_timeout:
                updates["REQUEST_TIMEOUT"] = request_timeout
            if endpoint_id:
                updates["RUNPOD_ENDPOINT_ID"] = endpoint_id

            try:
                write_env_values(updates)
            except Exception as e:
                send_html(self, config_page_html(f"Failed writing .env: {e}", error=True), status=500)
                return

            apply_runtime_config(final_key, final_endpoint, enable_vision=enable_vision)
            print("[proxy] configuration saved to .env")
            if is_config_ready():
                print("[proxy] server is ready to use with Visual Studio Code")
            send_html(self, config_page_html("Saved. Server is ready to use with Visual Studio Code."))
            return

        if not is_config_ready():
            self.send_json(503, {
                "error": "Proxy not configured yet. Open /config and set RUNPOD_API_KEY and endpoint.",
                "config_url": f"http://localhost:{PROXY_PORT}/config",
            })
            return

        model_name, quant, ctx_len = get_model_info()
        try:
            parsed = self.read_body()
            is_stream = parsed.get("stream", False)
            print(f"[proxy] POST {self.path}  stream={is_stream}")

            # --- /api/show --- answered locally (metadata only)
            if self.path == "/api/show":
                families = ["qwen36"]
                capabilities = ["completion", "tools"]
                if ENABLE_VISION:
                    families.append("clip")
                    capabilities.append("vision")
                self.send_json(200, {
                    "model": model_name,
                    "modelfile": f"FROM {model_name}",
                    "parameters": "temperature 1.0\ntop_p 0.95",
                    "template": "{{ .Prompt }}",
                    "details": {
                        "parent_model": "", "format": "gguf",
                        "family": "qwen36", "families": families,
                        "parameter_size": "27B", "quantization_level": quant,
                    },
                    "model_info": {
                        "general.architecture": "qwen36",
                        "general.parameter_count": 27000000000,
                        "general.quantization_version": 2,
                        "qwen36.context_length": ctx_len,
                        "qwen36.attention.head_count": 32,
                    },
                    "capabilities": capabilities,
                })

            # --- /api/chat -> /v1/chat/completions (Ollama native chat path) ---
            elif self.path == "/api/chat":
                opts = parsed.get("options", {}) or {}
                llama_body = {
                    "model": model_name,
                    "messages": parsed.get("messages", []),
                    "stream": is_stream,
                }
                if opts.get("temperature") is not None:
                    llama_body["temperature"] = opts["temperature"]
                if opts.get("top_p") is not None:
                    llama_body["top_p"] = opts["top_p"]
                if is_stream:
                    self._start_sse()
                    forward_stream_to_runpod("/v1/chat/completions", llama_body, self.wfile)
                else:
                    status, body = forward_to_runpod("/v1/chat/completions", llama_body)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            # --- /api/generate -> /v1/completions ---
            elif self.path == "/api/generate":
                llama_body = {
                    "model": model_name,
                    "prompt": parsed.get("prompt", ""),
                    "stream": is_stream,
                }
                if is_stream:
                    self._start_sse()
                    forward_stream_to_runpod("/v1/completions", llama_body, self.wfile)
                else:
                    status, body = forward_to_runpod("/v1/completions", llama_body)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            # --- /v1/chat/completions --- direct passthrough ---
            elif self.path == "/v1/chat/completions":
                parsed["model"] = parsed.get("model", model_name)
                if is_stream:
                    self._start_sse()
                    forward_stream_to_runpod("/v1/chat/completions", parsed, self.wfile)
                else:
                    status, body = forward_to_runpod("/v1/chat/completions", parsed)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            # --- /v1/responses --- (newer Copilot agent mode) -> translate ---
            elif self.path == "/v1/responses":
                chat_body = responses_to_chat_completions(parsed)
                is_stream = chat_body.get("stream", False)
                if is_stream:
                    self._start_sse()
                    forward_stream_to_runpod("/v1/chat/completions", chat_body, self.wfile)
                else:
                    status, body = forward_to_runpod("/v1/chat/completions", chat_body)
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body.encode())

            else:
                print(f"[proxy]   -> UNHANDLED POST '{self.path}' body={json.dumps(parsed)[:300]}")
                self.send_json(404, {"error": f"Unhandled POST: {self.path}"})

        except Exception as e:
            print(f"[proxy] ERROR on {self.path}: {e}")
            self.send_json(500, {"error": str(e)})


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), OllamaProxyHandler)
    server.daemon_threads = True

    print("\n", "=" * 60)
    print("Ollama <-> RunPod Serverless proxy")
    print("=" * 60)
    if not is_config_ready():
        print("-- Proxy is NOT configured yet! --\n\n")
        print(f"open config: http://localhost:{PROXY_PORT}/config")
        print("Go to that local config page")
        print("Provide RUNPOD_API_KEY and Runpodendpoint, then save.")
        print("You won't need to restart the proxy; it picks up config changes immediately.")
        print("\n\n------------BELOW WORKS AFTER CONFIGURATION SAVE---------------------")
    else:
        print("  Server is ready to use with Visual Studio Code.")
    print(f"  Listening (as Ollama):  http://0.0.0.0:{PROXY_PORT}")
    print(f"  Forwarding to RunPod:    {RUNPOD_BASE}/v1/...")
    print(f"  Auth header:             {'set' if RUNPOD_API_KEY else 'MISSING — set RUNPOD_API_KEY!'}")
    print(f"  Advertised model:        {MODEL_NAME}  (context {CONTEXT_LEN})")
    print(f"  Vision capability:       {'enabled' if ENABLE_VISION else 'disabled'}")
    print(f"  Cold-start drip:         '{DRIP_TOKEN}' every {DRIP_INTERVAL}s until real tokens")
    
    print()
    print("  Metadata (GET /api/tags, /api/version, /api/show, /v1/models)")
    print("  is answered LOCALLY and never wakes a GPU worker.")
    print("  Only /api/chat, /api/generate, /v1/chat/completions, /v1/responses")
    print("  are forwarded to RunPod.")
    print("=" * 60)
    print("NOTE on first deployment or deployment after a long idle period:")
    print("If you have just deployed the model on runpod, you need to wait about 10 minutes for the first cold start")
    print("if you get error in visual studo code like \n'............[proxy: upstream error] IncompleteRead(0 bytes read)'\n it's not ready yet, wait and try again in about 5 minutes\nthe server is still probably downloading the large model files")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        shutdown_server(server, reason="KeyboardInterrupt")