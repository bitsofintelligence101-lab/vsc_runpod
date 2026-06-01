# llama.cpp · RunPod · VS Code Copilot

Run a RunPod Hosted Model and use it as a local Ollama provider in Visual Studio Code Copilot.

---

## 1 · Clone the repo

```bash
git clone https://github.com/bitsofintelligence101-lab/vsc_runpod.git
cd vsc_runpod
```

---

## 2 · Start here: run the deploy script

Please use my referal link if you don't have a runpod account yet, it gives us both $10 in free credits:
https://runpod.io?ref=0tg2p4r0

This project is designed have a simple one-command deployment that handles both the RunPod serverless setup and the local proxy configuration. just run:

```bash
python3 runpod_deploy.py
```

`runpod_deploy.py` is the default entrypoint and will:

1. Prompt for your `RUNPOD_API_KEY` on first run (will auto save it to `.env` for future runs)
2. Create/reuse the serverless template and endpoint automatically
3. Configures the endpoint with:
    - **Workers Max:** `1`
    - **Idle Timeout:** `60` seconds
    - **Context Window:** `64000` tokens (Increasing this means you need bigger GPU workers, so it's set to 64k by default. Anything larger may fail to load on smaller 48GB GPUs. If you need a larger context window, change the `LLAMA_CTX` variable in `.env` and make sure to select a GPU with enough VRAM when configuring the endpoint.)
4. Launch the local proxy 

---


## 3 · Connect VS Code Copilot (Ollama local provider)

VS Code Copilot only supports Ollama as a local model provider (as of this writing). This proxy pretends to be Ollama and forwards generation requests to RunPod. Metadata probes (`/api/tags`, `/api/version`) are answered locally — they never wake a GPU worker else it would burn credits unnecessarily.

If you started with `python3 runpod_deploy.py`, the proxy is launched for you.

Directly running the proxy is still technically supported, but it is no longer the recommended starting flow:

```bash
python3 ollama_llama_runpod_proxy.py
```

If running the proxy directly, go to (`http://localhost:11434/config`) and provide your RunPod API key and Endpoint ID. Save the config, and the proxy picks it up immediately — no restart needed.

Then in VS Code: **Settings → GitHub Copilot: Local Provider → Ollama** (`localhost:11434`).

### Video walkthrough Connecting to VS Code Copilot

[![Watch how to connect the server to Visual Studio Code](https://img.youtube.com/vi/agnECMF3nVs/hqdefault.jpg)](https://youtu.be/agnECMF3nVs)

---

## OPTIONAL: Browser Chat UI

Open `chat.html` directly in your browser (no server needed). Enter your RunPod API key and Endpoint ID in the config bar at the top and click **Save**. Supports streaming text and image attachments (vision). Your config is persisted in `localStorage` between sessions.

The chat UI calls `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions` directly from the browser — no proxy required.

---

## OPTIONAL: Direct API usage

```python
from openai import OpenAI
client = OpenAI(
    api_key="<RUNPOD_API_KEY>",
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)
resp = client.chat.completions.create(
    model="Qwen3.6-27B-uncensored-heretic-v2",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

If you need different scaling/cost behavior, log in to the RunPod GUI after deployment and edit endpoint settings there.

# OPTIONAL: Model Download
> **Network Volume (one-time setup):** A 27B Q8_0 model is ~29 GB. Create a Network Volume, mount it to a temporary Pod, and run `./download_models.sh` once. The serverless worker mounts it at `/runpod-volume` on every start — no re-downloading.

# Advanced: 
This repo is currently hardcoded to the Q8 version of Qwen3.6-27B-uncensored-heretic-v2 model, but you can modify the dockerfile and proxy script to use any model you like (see make_your_own/README.md). You'd need to make the docker image, deploy it to docker hub and make sure the model files are included in the image or downloaded on startup from an accessible endpoint. Then you can change the `MODEL_ALIAS` and `DEFAULT_QUANTIZATION` variables in `.env` to point to your new model. Make sure to also update the `MODEL_NAME` variable in the proxy script to match the new model name and quantization tag.  If I get around to it i'll make this more plug-and-play in the future but for now it's a bit of a manual process to change models. Or drop the make_your_own files in to an LLM and it can tell you how to modify the dockerfile and proxy script to use a different model.