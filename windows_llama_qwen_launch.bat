@echo off
setlocal enabledelayedexpansion

:: FORCE the script to run in the folder where the .bat file is located
cd /d "%~dp0"

:: -------------------------------------------------------
:: CONTEXT SIZE: Read from .env (LLAMA_CTX) with a fallback
::   Q8_0 (~28.5 GB): 16384 or 24576 on a 32 GB card
::   Q6_K (~22 GB):   65000 is typically safe
::   Q4_K (~16 GB):   64576 is typically safe
:: -------------------------------------------------------
set MODEL_QUANT=Q8_0

:: Try to load LLAMA_CTX from .env; fall back to 120000 if not found
set MODEL_CTX_SIZE=120000
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if /i "%%A"=="LLAMA_CTX" (
            set "MODEL_CTX_SIZE=%%~B"
        )
    )
)

:: -------------------------------------------------------
:: HUGGING FACE CACHE: Set your preferred cache directory
:: Comment out to use Hugging Face's default location
:: -------------------------------------------------------
:: set "HF_HUB_CACHE=C:\Users\name\.cache\huggingface\hub"

echo ====================================================================
echo llmfan-Qwen3.6-27B-abliterated- (!MODEL_QUANT!) Setup ^& Run (Blackwell sm_120)
echo ====================================================================
echo.

:: 1. Check Prerequisites & Detect Python Safely
echo [1/6] Checking prerequisites...
where git >nul 2>&1 || (echo ERROR: Git is not installed or not in your PATH. & pause & exit /b)
where cmake >nul 2>&1 || (echo ERROR: CMake is not installed or not in your PATH. & pause & exit /b)
where nvcc >nul 2>&1 || (echo ERROR: CUDA Toolkit is not installed or not in your PATH. & pause & exit /b)

set PY_CMD=python
where py >nul 2>&1
if %ERRORLEVEL% equ 0 set PY_CMD=py

%PY_CMD% --version >nul 2>&1 || (echo ERROR: Python is not functioning properly. & pause & exit /b)
echo All required tools found! (Using %PY_CMD% for Python)
echo.

:: 2. Setup MSVC Environment
echo [2/6] Initializing Visual Studio C++ Environment...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Please install Visual Studio.
    pause
    exit /b
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo ERROR: Could not find Visual Studio.
    pause
    exit /b
)
call "!VS_PATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
echo.

:: 3. Install Hugging Face Hub
echo [3/6] Installing huggingface_hub...
%PY_CMD% -m pip install "huggingface_hub[cli]" >nul 2>&1
echo.

:: 4. Clone and Build llama.cpp
echo [4/6] Setting up and building llama.cpp...
if not exist "llama.cpp" (
    echo Cloning llama.cpp repository...
    git clone https://github.com/ggml-org/llama.cpp
)
cd llama.cpp

if not exist "build\bin\llama-server.exe" (
    echo Configuring CMake for CUDA ^(Targeting Blackwell architecture 120^)...
    set "ASM=cl.exe"
    cmake -B build -G Ninja -DCMAKE_POLICY_DEFAULT_CMP0194=OLD -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DCMAKE_ASM_COMPILER=cl.exe -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
    if !errorlevel! neq 0 ( echo ERROR: CMake configuration failed. & pause & exit /b )

    echo Building llama-server...
    cmake --build build --config Release -j %NUMBER_OF_PROCESSORS% --target llama-server
    if !errorlevel! neq 0 ( echo ERROR: Build failed. & pause & exit /b )
) else (
    echo Build already successfully completed. Skipping compilation!
)
echo.

:: 5. Download Model Files using Native Python
echo [5/6] Downloading models...

:: Create temporary python script to safely fetch the model
echo from huggingface_hub import snapshot_download > dl_model.py
echo path = snapshot_download(repo_id="llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF", allow_patterns="*!MODEL_QUANT!*") >> dl_model.py
echo with open("model_path.txt", "w") as f: f.write(path) >> dl_model.py

:: Create temporary python script to safely fetch the Vision Projector
echo from huggingface_hub import hf_hub_download > dl_vision.py
echo path = hf_hub_download(repo_id="llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF", filename="Qwen3.6-27B-mmproj-BF16.gguf") >> dl_vision.py
echo with open("vision_path.txt", "w") as f: f.write(path) >> dl_vision.py

echo.
echo Downloading llmfan46-Qwen3.6-27B-uncensored-heretic-v2-GGUF (!MODEL_QUANT!)...
%PY_CMD% dl_model.py
set /p MODEL_SNAP_DIR=<model_path.txt

echo.
echo Downloading Qwen3.6-27B-mmproj-BF16 vision projector...
%PY_CMD% dl_vision.py
set /p MMPROJ_FILE=<vision_path.txt

:: Clean up temp scripts
del dl_model.py dl_vision.py model_path.txt vision_path.txt

:: Automatically find the exact Q6_K file inside the snapshot directory
for %%f in ("!MODEL_SNAP_DIR!\*!MODEL_QUANT!*.gguf") do set "MODEL_FILE=%%f"
if not defined MODEL_FILE (
    echo ERROR: !MODEL_QUANT! model file not found in cache.
    pause
    exit /b
)

:: 6. Run Server
echo.
echo [6/6] Starting OpenAI-Compatible Server...
echo Model File:  !MODEL_FILE!
echo Vision File: !MMPROJ_FILE!
echo.
echo ====================================================================
echo The server Web UI will be available at: http://localhost:8080
echo Press Ctrl+C in this window to stop the server when you are done.
echo ====================================================================
echo.
echo model downloaded from https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-GGUF

:: Note: Windows cmd requires escaping inner quotes in JSON string arguments
:: set "CHAT_TEMPLATE={\"enable_thinking\":false}"
set "CHAT_TEMPLATE={\"enable_thinking\":true}"

:: =======================================================
:: GPU SELECTION:
:: Set to 0 to use ONLY the RTX 5090 (Device 0)
:: Set to 1 to use ONLY the RTX 5060 (Device 1)
:: Delete or comment out the line below to use BOTH GPUs
:: =======================================================
::set CUDA_VISIBLE_DEVICES=0

:: =======================================================
:: TENSOR SPLIT (VRAM distribution across GPUs):
:: Don't let llama.cpp decide automatically - you can specify how to split the model across multiple GPUs.
:: Format: --tensor-split "GPU0_weight,GPU1_weight"
:: Weights are proportional. Higher = more layers/VRAM used.
:: Examples:
::   "0,1"  -> 100% on 5060, 0% on 5090
::   "1,1"  -> 50/50 split (default)
::   "1,3"  -> ~25% on 5090, ~75% on 5060
::   "1,4"  -> ~20% on 5090, ~80% on 5060
:: Comment out or delete to use default even split.
:: =======================================================
set "TENSOR_SPLIT=2.6,1"
echo Setting tensor split for multi-GPU: !TENSOR_SPLIT! (5090,5060 Ti)

build\bin\llama-server.exe ^
  -m "!MODEL_FILE!" ^
  --mmproj "!MMPROJ_FILE!" ^
  --tensor-split "!TENSOR_SPLIT!" ^
  -ngl 99 ^
  --ctx-size !MODEL_CTX_SIZE! ^
  --flash-attn on ^
  --jinja ^
  --temp 1.0 ^
  --top-p 0.95 ^
  --top-k 20 ^
  --presence-penalty 0.5 ^
  --min-p 0.05 ^
  --host 0.0.0.0 ^
  --port 8080 ^
  --chat-template-kwargs "!CHAT_TEMPLATE!"
  --reasoning on

pause