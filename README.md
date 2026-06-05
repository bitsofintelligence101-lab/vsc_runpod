# llama.cpp · runpod · VS Code Copilot

Run a runpod Hosted Model and use it as a local Ollama provider in Visual Studio Code Copilot.

---

## 1 · Create a runpod account and get API key
[Get Started with API Keys](https://docs.runpod.io/get-started/api-keys)

If you don't have a runpod account yet, use this link:
[New runpod Account](https://runpod.io?ref=0tg2p4r0)
it gives us both $10 in free credits:

## 2 · Clone the repo

```bash
git clone https://github.com/bitsofintelligence101-lab/vsc_runpod.git
cd vsc_runpod
```

---

## 3 · Run the deploy script
run the deploy script:

```bash
python3 runpod_deploy.py
```

You'll be asked **Do you want to use a local llama model instead?** — answer `no` (or just press Enter) to proceed with the RunPod deployment.

# DONE
You should see the endpoint configuration printed in the terminal, the local proxy should be running. 
You can skip to step 4 to connect your endpoint to VS Code Copilot now, or read on for more details about how the deployment works and how to use the proxy directly if you want.

## More details on what the deploy script does:

This project is designed have a simple one-command deployment that handles both the RunPod serverless setup and the local proxy configuration.

`runpod_deploy.py` is the default entrypoint and will:

1. Prompt for your `RUNPOD_API_KEY` on first run (will auto save it to `.env` for future runs)
2. Create/reuse the serverless template and endpoint automatically
3. Configures the endpoint with:
    - **Workers Max:** `1`
    - **Idle Timeout:** `60` seconds
    - **Context Window:** `64000` tokens (Increasing this means you need bigger GPU workers, so it's set to 64k by default. Anything larger may fail to load on smaller 48GB GPUs. If you need a larger context window, change the `LLAMA_CTX` variable in `.env` and make sure to select a GPU with enough VRAM when configuring the endpoint.)
4. Launch the local proxy 

---


## 4 · Connect VS Code Copilot (Ollama local provider)

VS Code Copilot only supports Ollama as a local model provider (as of this writing). This proxy pretends to be Ollama and forwards generation requests to RunPod. Metadata probes (`/api/tags`, `/api/version`) are answered locally — they never wake a GPU worker else it would burn credits unnecessarily. 

Then in VS Code: **Settings → GitHub Copilot: Local Provider → Ollama**
Enter a name for your model host
then use the default URL (`localhost:11434`).

If you started with `python3 runpod_deploy.py`, the proxy is launched for you.

Technically you can also directly run the proxy (`ollama_llama_runpod_proxy.py`), but it is no longer the recommended starting flow. 


### Video walkthrough Connecting to VS Code Copilot

[![Watch how to connect the server to Visual Studio Code](https://img.youtube.com/vi/agnECMF3nVs/hqdefault.jpg)](https://youtu.be/agnECMF3nVs)

---
## NOTES ON RUNPOD: Brittle
RunPod's API and serverless environment are powerful but can be a bit brittle. If you encounter issues, the first step is usually to check the RunPod dashboard for any error logs or status messages related to your endpoint. Common issues include:
- **Worker Fails to Start:** This can happen if the model fails to load due to insufficient VRAM or other resource constraints. Check the endpoint logs for any error messages during startup.
- **Available GPU Types Change:** RunPod occasionally updates their available GPU types, which can affect your endpoint configuration. If your endpoint fails to start, verify that the selected GPU type is still available and compatible with your model's VRAM requirements.
- **Time of Day Variability:** RunPod's performance can vary based on the time of day and overall demand. If you experience slow response times or worker startup, it may be worth trying again during off-peak hours.

## OPTIONAL TESTING: Browser Chat UI

go to the  `chat` endpoint directly in your browser http://localhost:11434/chat. It should populate with your runpod endpoint and API key. Supports streaming text and image attachments (vision). Your config is persisted in `localStorage` between sessions. Technically you can use this as a chat interface but it is very light weight and lacks features, it's really more for testing.

The chat UI calls `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/chat/completions` directly from the browser — it doesn't go through the proxy.

---

## OPTIONAL: Direct API usage
This is what the chat UI calls under the hood. You can use the same approach to call the RunPod endpoint directly from your own code without going through the proxy, just make sure to use the correct base URL and include your API key in the headers. Here's an example using the OpenAI Python client:

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

# OPTIONAL: Run the model locally (Windows)

A `windows_llama_qwen_launch.bat` file is included for Windows users who want to run the same Qwen3.6-27B model locally instead of using RunPod. Just double-click it (or run it from PowerShell) to start the local model server. Once it's running, you can then run `python3 runpod_deploy.py` (answer `yes` when asked about using a local model) to have the proxy connect to your local instance instead.

# OPTIONAL: Model Download
**Network Volume (one-time setup):** A 27B Q8_0 model is ~29 GB. Create a Network Volume, mount it to a temporary Pod, and run `./download_models.sh` once. The serverless worker mounts it at `/runpod-volume` on every start — no re-downloading.
The downside of a fixed storage location is that you are locked to the data center that the volume is in.  This can limit avalibility since runpod is unable to move workers between data centers.  The upside is startup speed and reliability assuming GPU availity, the endpoint is a lot less brittle.

# Advanced: 
This repo is currently hardcoded to the Q8 version of Qwen3.6-27B-uncensored-heretic-v2 model due to the docker image used, but you can modify the dockerfile and proxy script to use any model you like (see make_your_own/README.md). You'd need to make the docker image, deploy it to docker hub and make sure the model files are included in the image or downloaded on startup from an accessible endpoint. Then you can change the `MODEL_ALIAS` and `DEFAULT_QUANTIZATION` variables in `.env` to point to your new model. Make sure to also update the `MODEL_NAME` variable in the proxy script to match the new model name and quantization tag.  If I get around to it i'll make this more plug-and-play in the future but for now it's a bit of a manual process to change models. Or drop the make_your_own files in to an LLM and it can tell you how to modify the dockerfile and proxy script to use a different model.

# To do:
- configure to use load balaned endpoints. this will allow actually allow parallelism to be taken advantage of.  currently it uses queue based, so one gpu can only manage. with load balanced multiple requests go to the same worker which brings down aggragate cost per token when multiple parallel requests are being made.
https://docs.runpod.io/serverless/load-balancing/overview