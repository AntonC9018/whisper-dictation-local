# Local Whisper dictation

A small local web app for English and Russian dictation. It records from the browser, accepts audio uploads, and includes two short synthetic test clips.

The repository contains two backends:

- **Faster Whisper CPU.** Runs multilingual `base` or `small` with int8 weights. Works on Windows, macOS, and Linux.
- **whisper.cpp with OpenVINO.** A WSL2 Ubuntu trial for Intel GPUs. It runs the encoder on the GPU and the decoder on the CPU. This page currently supports `base` only.

Model files are not stored in this repository. The Faster Whisper backend downloads the selected model when it starts. The OpenVINO installer downloads the `base` model and converts its encoder into the cache directory outside the checkout.

## Requirements

For Faster Whisper, install Python 3.10 or newer. The first model download needs internet access and about 1 GB of free space.

The OpenVINO backend requires WSL2, Ubuntu 22.04, an Intel GPU exposed to WSL, and an up-to-date Intel graphics driver on Windows. Its setup also installs compiler tools and a Python conversion environment, so allow several GB of free space. The initial model conversion can take a few minutes.

## Run on Windows

Open PowerShell in the repository folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run-windows.ps1
```

Open `http://127.0.0.1:8000`. Leave the PowerShell window open while you use the app. Use `run-windows.ps1 --port 8001` if port 8000 is busy.

## Run on macOS or Linux with CPU

Open a terminal in the repository folder:

```sh
bash scripts/install-linux-cpu.sh
bash scripts/run-linux-cpu.sh
```

Open `http://127.0.0.1:8000`. The installer creates a `.venv` folder and installs the pinned Faster Whisper package. Python model files go in the operating system's user cache, outside this checkout.

## Run the Intel GPU trial in WSL2

Use Ubuntu 22.04 under WSL2 and run these commands from the repository folder inside Ubuntu:

```sh
bash scripts/install-openvino-wsl.sh
bash scripts/run-openvino-wsl.sh
```

Open `http://127.0.0.1:8001`. The installer adds Intel's graphics package key and source, installs the OpenCL and Level Zero runtimes, builds the OpenVINO-enabled `whisper.cpp` CLI, and converts the multilingual `base` encoder. It keeps the source checkout, conversion environment, compiled binaries, and model files under `~/.cache/whisper-dictation` by default. Set `WHISPER_DICTATION_CACHE` to use another cache directory.

If OpenVINO cannot see the GPU after installing the Windows graphics driver, run `wsl.exe --update` and `wsl.exe --shutdown` in Windows PowerShell, reopen Ubuntu, then run the app again. See [Intel's WSL GPU setup guide](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2025-1/configure-wsl-2-for-gpu-workflows.html).

The OpenVINO path is based on [whisper.cpp's OpenVINO support](https://github.com/ggml-org/whisper.cpp#openvino-support). The installer pins the upstream source revision used for this prototype.

## Use the app

- **Transcribe sample** runs the included synthetic English clip.
- **Record a phrase** records through the browser microphone. Allow microphone access when the browser asks.
- **Choose audio file** accepts WAV, MP3, M4A, OGG, and WebM files up to 25 MB.

Choose Auto for English, Russian, or speech that switches between them. Whisper may miss very brief language changes. The OpenVINO server uses `ffmpeg` to convert recordings to 16 kHz mono WAV before transcription. Uploads are written to a temporary directory and removed after each request.

Both servers bind to `127.0.0.1`, so only the same computer can reach them. Setup and first run download packages and model files from PyPI, Ubuntu and Intel package repositories, Hugging Face, and OpenAI model hosting. Transcription itself runs locally.

## Transcribe a file from Python

After installing the Faster Whisper dependencies, run:

```sh
.venv/bin/python transcribe_file.py path/to/recording.wav --model base --language auto
```

On Windows:

```powershell
.\.venv\Scripts\python.exe .\transcribe_file.py .\sample_phrase.wav --model base --language auto
```

Use `--model small` for the larger model or `--language en` / `--language ru` to force one language.

## Trial result

On an Intel Iris Xe in WSL2, the cached OpenVINO endpoint transcribed the 3.81-second sample in 0.96 seconds. The first OpenVINO run took about 4.3 seconds while it prepared the GPU cache. A CPU-only `whisper.cpp` run on the same clip took 2.58 seconds. These are measurements from one short sample, not a general benchmark. OpenVINO accelerates the encoder; the decoder still runs on the CPU.

