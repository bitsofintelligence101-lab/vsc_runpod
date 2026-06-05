#!/usr/bin/env python3
"""
deploy.py — Programmatic RunPod Serverless Endpoint Deployer
=============================================================
Automates what Step 2 of the README does manually in the console.

Flow:
  1. Read RUNPOD_API_KEY from environment (or .env file)
  2. Check if a serverless template with our name already exists
  3. Create it if not (saveTemplate mutation)
  4. Check if an endpoint with our name already exists
  5. Create it if not (saveEndpoint mutation)
  6. Write the endpoint ID and base URL to .env so the proxy picks it up

Usage:
  python3 deploy.py
    → prompts for API key on first run, saves it to .env, then deploys.
    → on subsequent runs the key is loaded from .env automatically.

Requirements:
  pip install requests python-dotenv
"""

import json
import os
import sys
import importlib
import subprocess
import requests

# Resolve paths relative to this script so behavior is stable even when
# launched from a different working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, ".env")

# ── optional: load a .env file if present ────────────────────────────────────
try:
    dotenv = importlib.import_module("dotenv")
    dotenv.load_dotenv(dotenv_path=ENV_PATH)
except ImportError:
    pass  # python-dotenv not installed — that's fine, env vars still work


def _load_api_key_from_env_file() -> str:
    """
    Fallback parser for RUNPOD_API_KEY from ENV_PATH when python-dotenv
    is unavailable or did not populate os.environ for any reason.
    """
    if not os.path.exists(ENV_PATH):
        return ""

    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            if key.strip() != "RUNPOD_API_KEY":
                continue

            value = value.strip()
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            return value.strip()

    return ""


def resolve_api_key() -> str:
    """
    Return the RunPod API key, prompting the user if it isn't already set.

    Priority:
      1. RUNPOD_API_KEY already in environment (or loaded from .env above)
      2. Interactive prompt — the entered key is written to .env so future
         runs skip the prompt entirely.

    The .env file is written in simple KEY=value format so python-dotenv,
    bash 'source .env', and Docker --env-file all understand it.
    """
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()

    # Fallback: parse .env ourselves in case python-dotenv is not installed
    # or failed to load due to an unexpected runtime environment.
    if not api_key:
        api_key = _load_api_key_from_env_file()
        if api_key:
            os.environ["RUNPOD_API_KEY"] = api_key

    if api_key:
        # Key was already present in the environment — nothing to do.
        return api_key

    # No key found — ask the user.
    print("No RUNPOD_API_KEY found in environment or .env file.")
    api_key = input("Enter your RunPod API key: ").strip()

    if not api_key:
        print("ERROR: API key cannot be empty.")
        sys.exit(1)

    # Persist the key to .env so the user never has to enter it again.
    # If .env already exists we append/overwrite just the RUNPOD_API_KEY line
    # to avoid clobbering any other variables already in the file.
    existing_lines: list[str] = []

    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            # Keep every line that isn't RUNPOD_API_KEY
            existing_lines = [
                line for line in f.readlines()
                if not line.startswith("RUNPOD_API_KEY")
            ]

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(existing_lines)
        f.write(f"RUNPOD_API_KEY={api_key}\n")

    print(f"  API key saved to {ENV_PATH}\n")

    # Also inject into the current process so downstream code can use os.environ
    os.environ["RUNPOD_API_KEY"] = api_key

    return api_key

# ── configuration ─────────────────────────────────────────────────────────────

# The Docker image to deploy.
# Change this if you fork the project and push your own image.
IMAGE_NAME = "marsdefender5/qwen27b_q8_heritic:v1"

# Names used to identify the template and endpoint in RunPod.
# If these already exist under your account, the script reuses them.
TEMPLATE_NAME = "qwen27b-q8-heretic-template"
ENDPOINT_NAME = "qwen27b-q8-heretic-endpoint"

# Container disk in GB.
# 40 GB is required for the Q8_0 27B model weights + runtime overhead.
CONTAINER_DISK_GB = 40

# GPU type preference list (REST API `gpuTypeIds`).
# The order determines fallback priority when capacity is limited.
# Source: https://docs.runpod.io/api-reference/endpoints/POST/endpoints
GPU_TYPE_IDS = [
  "NVIDIA RTX A6000",
  "NVIDIA L40S",
  "NVIDIA A100 80GB PCIe",
]

# Minimum CUDA version accepted by workers (now supported via REST API).
MIN_CUDA_VERSION = "12.6"

# Scaling config — mirrors the README recommendations.
WORKERS_MIN = 0          # scale to zero when idle (saves money)
WORKERS_MAX = 1          # raise this for higher concurrency
IDLE_TIMEOUT = 60        # seconds before an idle worker shuts down

# RunPod GraphQL endpoint.
# Auth is passed as a query parameter (not a header) per RunPod's API design.
# Source: https://docs.runpod.io/sdks/graphql/configurations
GRAPHQL_URL = "https://api.runpod.io/graphql"

# RunPod REST endpoint for modern Serverless endpoint creation.
REST_BASE_URL = "https://rest.runpod.io/v1"

# ── helpers ───────────────────────────────────────────────────────────────────

def gql(api_key: str, query: str) -> dict:
    """
    Send a GraphQL query/mutation to RunPod and return the parsed JSON.
    Raises RuntimeError if the response contains GraphQL errors.
    """
    resp = requests.post(
        GRAPHQL_URL,
        params={"api_key": api_key},          # RunPod auth goes in query string
        headers={"Content-Type": "application/json"},
        json={"query": query},
        timeout=30,
    )
    try:
        resp.raise_for_status()  # surface HTTP-level errors early
    except requests.HTTPError as exc:
        # Include response body so schema/validation failures are visible.
        detail = resp.text.strip()
        raise RuntimeError(
            f"RunPod GraphQL HTTP {resp.status_code}: {detail or '<empty response body>'}"
        ) from exc
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"GraphQL error: {body['errors']}")
    return body["data"]


def find_existing_template(api_key: str) -> str | None:
    """
    List all templates on the account and return the ID of the one matching
    TEMPLATE_NAME, or None if it doesn't exist yet.
    """
    data = gql(api_key, """
        query {
          myself {
            podTemplates {
              id
              name
              isServerless
            }
          }
        }
    """)
    for t in data["myself"]["podTemplates"]:
        if t["name"] != TEMPLATE_NAME:
            continue
        if t.get("isServerless"):
            return t["id"]
        raise RuntimeError(
            f"A template named '{TEMPLATE_NAME}' already exists but is not serverless. "
            "Rename TEMPLATE_NAME in this script or remove/rename that template in RunPod."
        )
    return None


def create_template(api_key: str) -> str:
    """
    Create a serverless template and return its ID.

    Key points confirmed from RunPod docs
    (https://docs.runpod.io/sdks/graphql/manage-pod-templates):
      - isServerless: true   marks it as a serverless (not GPU Pod) template
      - volumeInGb: 0        serverless workers have no persistent volume
      - containerDiskInGb    ephemeral scratch space for the worker container
    """
    print(f"  Creating serverless template '{TEMPLATE_NAME}'...")
    data = gql(api_key, f"""
        mutation {{
          saveTemplate(input: {{
            name:              "{TEMPLATE_NAME}",
            imageName:         "{IMAGE_NAME}",
            isServerless:      true,
            containerDiskInGb: {CONTAINER_DISK_GB},
                        volumeInGb:        0,
                        dockerArgs:        "python handler.py",
                        env:               []
          }}) {{
            id
            name
          }}
        }}
    """)
    template_id = data["saveTemplate"]["id"]
    print(f"  Template created: id={template_id}")
    return template_id


def find_existing_endpoint(api_key: str) -> str | None:
    """
    List all endpoints on the account and return the ID of the one matching
    ENDPOINT_NAME, or None if it doesn't exist yet.
    """
    data = gql(api_key, """
        query {
          myself {
            endpoints {
              id
              name
            }
          }
        }
    """)
    for e in data["myself"]["endpoints"]:
        if e["name"] == ENDPOINT_NAME:
            return e["id"]
    return None


def create_endpoint(api_key: str, template_id: str) -> str:
    """
    Create a serverless endpoint and return its ID.

    Uses the REST API schema documented at:
      https://docs.runpod.io/api-reference/endpoints/POST/endpoints

    Field notes:
      - gpuTypeIds  : list of concrete RunPod GPU model names
      - flashboot   : boolean
      - scalerType  : QUEUE_DELAY scales on queue latency
      - workersMin 0: enables scale-to-zero
    """
    print(f"  Creating endpoint '{ENDPOINT_NAME}'...")
    payload = {
        "name": ENDPOINT_NAME,
        "templateId": template_id,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": GPU_TYPE_IDS,
        "workersMin": WORKERS_MIN,
        "workersMax": WORKERS_MAX,
        "idleTimeout": IDLE_TIMEOUT,
        "flashboot": True,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "minCudaVersion": MIN_CUDA_VERSION,
    }

    resp = requests.post(
        f"{REST_BASE_URL}/endpoints",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = resp.text.strip()
        raise RuntimeError(
            f"RunPod REST HTTP {resp.status_code}: {detail or '<empty response body>'}"
        ) from exc

    data = resp.json()
    endpoint_id = data["id"]
    print(f"  Endpoint created: id={endpoint_id}")
    return endpoint_id


def save_config(endpoint_id: str) -> None:
    """
    Persist deployment results in .env — writes (or overwrites) two variables:
      RUNPOD_ENDPOINT_ID   the bare endpoint ID
      RUNPOD_OPENAI_BASE   the full OpenAI-compatible base URL for direct API use
    Any other existing variables in .env are left untouched.
    """
    # ── update .env ───────────────────────────────────────────────────────────
    openai_base = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"

    # Keys we manage — strip any existing lines for these before re-writing,
    # so re-running deploy.py after a re-deploy always reflects the latest IDs.
    managed_keys = {"RUNPOD_ENDPOINT_ID", "RUNPOD_OPENAI_BASE"}

    existing_lines: list[str] = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            existing_lines = [
                line for line in f.readlines()
                if not any(line.startswith(k) for k in managed_keys)
            ]

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(existing_lines)
        # Blank line separator if the file already had content
        if existing_lines and not existing_lines[-1].endswith("\n\n"):
            f.write("\n")
        f.write(f"RUNPOD_ENDPOINT_ID={endpoint_id}\n")
        f.write(f'RUNPOD_OPENAI_BASE="{openai_base}"\n')

    print(f"  .env updated:")
    print(f"    RUNPOD_ENDPOINT_ID={endpoint_id}")
    print(f"    RUNPOD_OPENAI_BASE={openai_base}")


def check_local_llama() -> bool:
    """
    Check whether a llama.cpp server is running on localhost:8080.

    llama.cpp exposes an OpenAI-compatible API at /v1/models by default.
    A successful GET means the local instance is ready to use.
    """
    try:
        resp = requests.get(
            "http://localhost:8080/v1/models",
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        print(f"  Local llama.cpp instance found on port 8080")
        if models:
            print(f"  Available model(s): {[m.get('id', '?') for m in models]}")
        return True
    except requests.ConnectionError:
        return False
    except requests.Timeout:
        print("  Local llama.cpp on port 8080 timed out (is it still starting up?)")
        return False
    except Exception as exc:
        print(f"  Error checking localhost:8080: {exc}")
        return False


def launch_proxy_local() -> None:
    """
    Start the proxy server pointing to the local llama.cpp instance instead
    of a RunPod endpoint.

    We set RUNPOD_BASE to localhost:8080 so the proxy treats it as the
    OpenAI-compatible backend.  No API key is needed for local mode.
    """
    # Temporarily override env so the proxy targets localhost:8080
    os.environ["RUNPOD_BASE"] = "http://localhost:8080"
    os.environ["RUNPOD_OPENAI_BASE"] = "http://localhost:8080/v1"

    proxy_script = os.path.join(os.path.dirname(__file__), "ollama_llama_runpod_proxy.py")
    print("\nLaunching proxy server (local llama.cpp mode)...")
    print(f"  {sys.executable} {proxy_script}")
    print(f"  Target: http://localhost:8080")
    proc = subprocess.Popen([sys.executable, proxy_script])
    print("  Proxy server started.  Press Ctrl+C to stop.\n")
    try:
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def launch_proxy() -> None:
    """Start the local Ollama-compatible proxy in the current console.

    Blocks until the proxy exits so that Ctrl+C in the terminal propagates to
    the child process (both share the same console process group on Windows and
    the same process group on Unix).  The parent then waits up to 5 seconds for
    the proxy to shut down gracefully before force-killing it.
    """
    proxy_script = os.path.join(os.path.dirname(__file__), "ollama_llama_runpod_proxy.py")
    print("\nLaunching proxy server...")
    print(f"  {sys.executable} {proxy_script}")
    proc = subprocess.Popen([sys.executable, proxy_script])
    print("  Proxy server started.  Press Ctrl+C to stop.\n")
    try:
        proc.wait()
    except KeyboardInterrupt:
        # Ctrl+C already delivered to the child via the shared console group;
        # just wait briefly for it to finish its own shutdown handler.
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 0. ask whether to use local llama.cpp ────────────────────────────────
    use_local = input(
        "Before launching Runpod, do you want to proxy to a local llama.cpp instance instead? [y/N]: "
    ).strip().lower()

    if use_local in ("y", "yes"):
        if check_local_llama():
            print("\nUsing local llama.cpp instance.\n")
            launch_proxy_local()
            return
        else:
            print("No llama instance found on localhost:8080.")
            use_runpod = input(
                "Should I launch a RunPod connection instead? [Y/n]: "
            ).strip().lower()
            if use_runpod not in ("n", "no"):
                pass  # fall through to the RunPod flow below
            else:
                print("\nExiting. Start llama.cpp on port 8080 and try again.")
                return

    # ── 1. resolve API key (prompt + save to .env on first run) ─────────────
    api_key = resolve_api_key()

    # ── fast path: if an endpoint ID is already saved in .env, skip deploy ──
    existing_endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    if not existing_endpoint_id:
        # Try reading directly from the env file in case it wasn't loaded
        import re as _re
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _m = _re.match(r'^RUNPOD_ENDPOINT_ID\s*=\s*"?([^"\n]+)"?\s*$', _line)
                    if _m:
                        existing_endpoint_id = _m.group(1).strip()
                        break
    if existing_endpoint_id:
        print(
            f"A RunPod endpoint exists (id={existing_endpoint_id}), launching proxy server using it.\n"
            "Delete the RUNPOD_ENDPOINT_ID line from .env if you want to reset."
        )
        launch_proxy()
        return

    print("=== RunPod Serverless Deployer ===\n")

    # ── 2. template: reuse or create ─────────────────────────────────────────
    print("Step 1/2 — Serverless template")
    template_id = find_existing_template(api_key)
    if template_id:
        print(f"  Found existing template '{TEMPLATE_NAME}': id={template_id} (reusing)")
    else:
        template_id = create_template(api_key)

    # ── 3. endpoint: reuse or create ─────────────────────────────────────────
    print("\nStep 2/2 — Serverless endpoint")
    endpoint_id = find_existing_endpoint(api_key)
    if endpoint_id:
        print(f"  Found existing endpoint '{ENDPOINT_NAME}': id={endpoint_id} (reusing)")
    else:
        endpoint_id = create_endpoint(api_key, template_id)

    # ── 4. persist config for the proxy ──────────────────────────────────────
    save_config(endpoint_id)

    print(f"""
=== Done ===

Endpoint ID : {endpoint_id}
GPU types   : {', '.join(GPU_TYPE_IDS)}
Workers     : min={WORKERS_MIN}, max={WORKERS_MAX}, idle timeout={IDLE_TIMEOUT}s
Min CUDA    : {MIN_CUDA_VERSION}

Now start the proxy:
  python3 ollama_llama_runpod_proxy.py
""")

    launch_now = input(
        "Do you want to launch the proxy server now so you can use the endpoint in Visual Studio Code? [y/N]: "
    ).strip().lower()
    if launch_now in ("y", "yes"):
        launch_proxy()
    else:
        print("\nYou can run it later by running this script or the ollama_llama_runpod_proxy.py script directly.")


if __name__ == "__main__":
    main()