#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANO_REPO="${SCRIPT_DIR}/../MOSS-TTS-Nano"

# Conda environment (set CONDA_ENV to override; defaults differ per variant)
CONDA_ENV="${CONDA_ENV:-}"

# Defaults
DEVICE="${DEVICE:-cuda:0}"
PORT="${PORT:-7861}"
HOST="${LISTEN_HOST:-127.0.0.1}"
MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-TTS}"
ATTN="${ATTN:-auto}"
SHARE="${SHARE:-}"
MODE="8b"     # 8b | nano | full
RELOAD=1

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Launch the MOSS-TTS Multilingual Demo (English, French, Spanish, German).

Model variants:
  (default)                 8B model via llama.cpp (Q4_K_M, high quality)
  -n, --nano                Nano (~100M params, CPU/GPU, fast)
  -f, --full                Full 8B via PyTorch (needs 16GB+ VRAM)

Options:
  -d, --device DEVICE       Torch device (default: auto)
  -p, --port PORT           Server port (default: 7862/7863/7861)
  -H, --host HOST           Server host (default: 0.0.0.0)
  -e, --env ENV             Conda environment name (default: auto per variant)
  -R, --no-reload           Disable hot reload
  -s, --share               Create a public Gradio share link
  -h, --help                Show this help message

Examples:
  $(basename "$0")                    # 8B llama.cpp (default, best quality)
  $(basename "$0") --nano             # Nano 100M (fast, lighter)
  $(basename "$0") --full             # Full 8B PyTorch (needs big GPU)
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--nano)    MODE="nano"; shift ;;
        -f|--full)    MODE="full"; shift ;;
        -R|--no-reload) RELOAD=0; shift ;;
        -d|--device)  DEVICE="$2"; shift 2 ;;
        -p|--port)    PORT="$2"; shift 2 ;;
        -H|--host)    HOST="$2"; shift 2 ;;
        -m|--model)   MODEL_PATH="$2"; shift 2 ;;
        -a|--attn)    ATTN="$2"; shift 2 ;;
        -e|--env)     CONDA_ENV="$2"; shift 2 ;;
        -s|--share)   SHARE="--share"; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

activate_conda() {
    _SAVE_HOST="$HOST"; _SAVE_DEVICE="$DEVICE"; _SAVE_PORT="$PORT"
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
    HOST="$_SAVE_HOST"; DEVICE="$_SAVE_DEVICE"; PORT="$_SAVE_PORT"
}

# ── Nano mode ───────────────────────────────────────────────────────────────
if [[ "$MODE" == "nano" ]]; then
    [[ "$DEVICE" == "cuda:0" ]] && DEVICE="auto"
    [[ "$PORT" == "7861" ]] && PORT="7862"
    [[ -z "$CONDA_ENV" ]] && CONDA_ENV="moss-nano"

    echo "=== MOSS-TTS-Nano Multilingual Demo ==="
    echo "  Conda env: $CONDA_ENV"
    echo "  Device:    $DEVICE"
    echo "  Port:      $PORT"
    echo "  Host:      $HOST"
    echo "========================================"

    activate_conda

    if [[ ! -d "$NANO_REPO" ]]; then
        echo "[Setup] Cloning MOSS-TTS-Nano..."
        git clone https://github.com/OpenMOSS/MOSS-TTS-Nano.git "$NANO_REPO"
    fi

    if ! python -c "import moss_tts_nano" 2>/dev/null; then
        echo "[Setup] Installing MOSS-TTS-Nano dependencies..."
        pip install -r "$NANO_REPO/requirements.txt"
        pip install -e "$NANO_REPO"
    fi

    export MOSS_DEVICE="$DEVICE"
    export MOSS_DTYPE="auto"

    if [[ "$RELOAD" -eq 1 ]]; then
        echo "[Hot reload] UI changes auto-apply — model stays loaded"
        exec gradio "$SCRIPT_DIR/clis/moss_tts_nano_multilingual_demo.py"
    else
        exec python "$SCRIPT_DIR/clis/moss_tts_nano_multilingual_demo.py" \
            --device "$DEVICE" --port "$PORT" --host "$HOST" $SHARE
    fi

# ── 8B llama.cpp mode ───────────────────────────────────────────────────────
elif [[ "$MODE" == "8b" ]]; then
    [[ "$PORT" == "7861" ]] && PORT="7863"
    [[ -z "$CONDA_ENV" ]] && CONDA_ENV="moss-8b"

    echo "=== MOSS-TTS 8B Multilingual Demo (llama.cpp) ==="
    echo "  Conda env: $CONDA_ENV"
    echo "  Port:      $PORT"
    echo "  Host:      $HOST"
    echo "  Config:    configs/llama_cpp/onnx-8gb.yaml"
    echo "=================================================="

    activate_conda

    if [[ ! -f "$SCRIPT_DIR/weights/MOSS-TTS-GGUF/first_class/MOSS_TTS_FIRST_CLASS_Q4_K_M.gguf" ]]; then
        echo "[Error] First-class GGUF not found. Run ./setup_8b_llamacpp.sh first."
        exit 1
    fi

    LLAMA_MOSS_DIR="${SCRIPT_DIR}/llama.cpp-moss"
    export MOSS_8B_MODEL="$SCRIPT_DIR/weights/MOSS-TTS-GGUF/first_class/MOSS_TTS_FIRST_CLASS_Q4_K_M.gguf"
    export MOSS_8B_MODEL_F16="$SCRIPT_DIR/weights/MOSS-TTS-GGUF/first_class/MOSS_TTS_FIRST_CLASS_F16.gguf"
    export MOSS_8B_ENCODER="$SCRIPT_DIR/weights/MOSS-Audio-Tokenizer-GGUF/encoder_f16.gguf"
    export MOSS_8B_DECODER="$SCRIPT_DIR/weights/MOSS-Audio-Tokenizer-GGUF/decoder_f16.gguf"
    export MOSS_8B_BINARY="$LLAMA_MOSS_DIR/build-cuda/bin/llama-moss-tts"
    export MOSS_8B_BINARY_INTERACTIVE="$LLAMA_MOSS_DIR/build-cuda/bin/llama-moss-tts-interactive"
    export LD_LIBRARY_PATH="$LLAMA_MOSS_DIR/build-cuda/bin:${LD_LIBRARY_PATH:-}"

    exec python "$SCRIPT_DIR/clis/moss_tts_8b_multilingual_demo.py" \
        --port "$PORT" --host "$HOST" $SHARE

# ── Full PyTorch mode ───────────────────────────────────────────────────────
else
    [[ -z "$CONDA_ENV" ]] && CONDA_ENV="ai"

    echo "=== MOSS-TTS Multilingual Demo (PyTorch) ==="
    echo "  Conda env: $CONDA_ENV"
    echo "  Device:    $DEVICE"
    echo "  Port:      $PORT"
    echo "  Host:      $HOST"
    echo "=============================================="

    activate_conda

    if ! python -c "import moss_tts_delay" 2>/dev/null; then
        echo "[Setup] Installing MOSS-TTS..."
        pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e "$SCRIPT_DIR[torch-runtime]"
    fi

    exec python "$SCRIPT_DIR/clis/moss_tts_multilingual_demo.py" \
        --device "$DEVICE" --port "$PORT" --host "$HOST" \
        --model_path "$MODEL_PATH" --attn_implementation "$ATTN" $SHARE
fi
