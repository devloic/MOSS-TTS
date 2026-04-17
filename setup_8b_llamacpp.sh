#!/usr/bin/env bash
set -euo pipefail

# MOSS-TTS 8B llama.cpp Backend Setup
# For 8GB GPUs (RTX 5060, RTX 4060, etc.)
# Uses Q4_K_M quantization + ONNX audio tokenizer + low-memory mode

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$SCRIPT_DIR/weights"
LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp-moss"
CONDA_ENV="${CONDA_ENV:-moss-8b}"

echo "=== MOSS-TTS 8B llama.cpp Setup ==="
echo "  Weights dir:  $WEIGHTS_DIR"
echo "  llama.cpp:    $LLAMA_CPP_DIR"
echo "  Conda env:    $CONDA_ENV"
echo "====================================="
echo

# --- Step 1: Conda environment ---
echo "[1/5] Setting up conda environment '$CONDA_ENV'..."
set +u
eval "$(conda shell.bash hook)"
set -u

if ! conda env list 2>/dev/null | grep -q "^${CONDA_ENV} "; then
    echo "  Creating new environment..."
    conda create -n "$CONDA_ENV" python=3.11 -y
fi

set +u
conda activate "$CONDA_ENV"
set -u

# --- Step 2: Install Python deps ---
echo "[2/5] Installing Python dependencies..."
pip install -e "$SCRIPT_DIR[llama-cpp-onnx]" 2>&1 | tail -3
echo "  Done."

# --- Step 3: Download weights ---
echo "[3/5] Downloading model weights (this may take a while)..."

if [[ ! -f "$WEIGHTS_DIR/MOSS-TTS-GGUF/MOSS_TTS_Q4_K_M.gguf" ]]; then
    echo "  Downloading Q4_K_M backbone (~5GB)..."
    hf download OpenMOSS-Team/MOSS-TTS-GGUF \
        MOSS_TTS_Q4_K_M.gguf \
        --local-dir "$WEIGHTS_DIR/MOSS-TTS-GGUF"
else
    echo "  GGUF backbone already present, skipping."
fi

NEED_EMB=0; NEED_LM=0; NEED_TOK=0
[[ ! -d "$WEIGHTS_DIR/MOSS-TTS-GGUF/embeddings" ]] || [[ $(ls "$WEIGHTS_DIR/MOSS-TTS-GGUF/embeddings/"*.npy 2>/dev/null | wc -l) -lt 33 ]] && NEED_EMB=1
[[ ! -d "$WEIGHTS_DIR/MOSS-TTS-GGUF/lm_heads" ]] || [[ $(ls "$WEIGHTS_DIR/MOSS-TTS-GGUF/lm_heads/"*.npy 2>/dev/null | wc -l) -lt 33 ]] && NEED_LM=1
[[ ! -f "$WEIGHTS_DIR/MOSS-TTS-GGUF/tokenizer/tokenizer.json" ]] && NEED_TOK=1

if [[ "$NEED_EMB" -eq 1 ]]; then
    echo "  Downloading embeddings (33 files)..."
    hf download OpenMOSS-Team/MOSS-TTS-GGUF \
        --include "embeddings/*.npy" \
        --local-dir "$WEIGHTS_DIR/MOSS-TTS-GGUF"
else
    echo "  Embeddings already present, skipping."
fi

if [[ "$NEED_LM" -eq 1 ]]; then
    echo "  Downloading LM heads (33 files)..."
    hf download OpenMOSS-Team/MOSS-TTS-GGUF \
        --include "lm_heads/*.npy" \
        --local-dir "$WEIGHTS_DIR/MOSS-TTS-GGUF"
else
    echo "  LM heads already present, skipping."
fi

if [[ "$NEED_TOK" -eq 1 ]]; then
    echo "  Downloading tokenizer..."
    hf download OpenMOSS-Team/MOSS-TTS-GGUF \
        --include "tokenizer/*" \
        --local-dir "$WEIGHTS_DIR/MOSS-TTS-GGUF"
else
    echo "  Tokenizer already present, skipping."
fi

if [[ ! -f "$WEIGHTS_DIR/MOSS-Audio-Tokenizer-ONNX/decoder.onnx" ]]; then
    echo "  Downloading ONNX audio tokenizer..."
    hf download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX \
        --local-dir "$WEIGHTS_DIR/MOSS-Audio-Tokenizer-ONNX"
else
    echo "  ONNX audio tokenizer already present, skipping."
fi

# --- Step 4: Build llama.cpp ---
echo "[4/5] Building llama.cpp with CUDA support..."

if [[ ! -d "$LLAMA_CPP_DIR" ]]; then
    echo "  Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

if [[ ! -f "$LLAMA_CPP_DIR/build/bin/llama-cli" ]] && [[ ! -f "$LLAMA_CPP_DIR/build/libllama.so" ]]; then
    echo "  Building llama.cpp..."
    cd "$LLAMA_CPP_DIR"

    # Check if nvcc supports the GPU architecture
    NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "0")
    NVCC_MAJOR=$(echo "$NVCC_VER" | cut -d. -f1)

    if [[ "$NVCC_MAJOR" -ge 13 ]]; then
        echo "  CUDA $NVCC_VER detected — building with CUDA support"
        cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    else
        echo "  CUDA $NVCC_VER too old for this GPU — building CPU-only"
        echo "  (llama.cpp CPU is still fast for 8B inference)"
        rm -rf build 2>/dev/null
        cmake -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    fi

    cmake --build build --config Release -j "$(nproc)"
    cd "$SCRIPT_DIR"
else
    echo "  llama.cpp already built, skipping."
fi

# --- Step 5: Build C bridge ---
echo "[5/5] Building backbone C bridge..."

BRIDGE_SO="$SCRIPT_DIR/moss_tts_delay/llama_cpp/libbackbone_bridge.so"
if [[ ! -f "$BRIDGE_SO" ]]; then
    cd "$SCRIPT_DIR/moss_tts_delay/llama_cpp"
    bash build_bridge.sh "$LLAMA_CPP_DIR"
    cd "$SCRIPT_DIR"
else
    echo "  C bridge already built, skipping."
fi

echo
echo "=== Setup Complete ==="
echo
echo "Test with:"
echo "  conda activate $CONDA_ENV"
echo "  python -m moss_tts_delay.llama_cpp \\"
echo "      --config configs/llama_cpp/onnx-8gb.yaml \\"
echo "      --text 'Hello, this is a test of the MOSS TTS system.' \\"
echo "      --output test_output.wav --profile"
echo
echo "Voice cloning:"
echo "  python -m moss_tts_delay.llama_cpp \\"
echo "      --config configs/llama_cpp/onnx-8gb.yaml \\"
echo "      --text 'Hello world' \\"
echo "      --reference path/to/reference.wav \\"
echo "      --output cloned_output.wav"
echo
echo "Peak VRAM: ~5.6 GB (low-memory mode, staged loading)"
