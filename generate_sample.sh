#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_MOSS_DIR="${SCRIPT_DIR}/llama.cpp-moss"
BINARY="$LLAMA_MOSS_DIR/build-cuda/bin/llama-moss-tts"
MODEL="$SCRIPT_DIR/weights/MOSS-TTS-GGUF/first_class/MOSS_TTS_FIRST_CLASS_Q4_K_M.gguf"
ENCODER="$SCRIPT_DIR/weights/MOSS-Audio-Tokenizer-GGUF/encoder_f16.gguf"
DECODER="$SCRIPT_DIR/weights/MOSS-Audio-Tokenizer-GGUF/decoder_f16.gguf"

export LD_LIBRARY_PATH="$LLAMA_MOSS_DIR/build-cuda/bin:${LD_LIBRARY_PATH:-}"

DEFAULT_TEXT="Bonjour et bienvenue. Nous allons explorer les merveilles de la cuisine française et découvrir les secrets des grands chefs."
TEXT="${1:-$DEFAULT_TEXT}"
OUTPUT="${2:-output.wav}"
REF_AUDIO="${3:-}"

echo "Text:   ${TEXT:0:80}..."
echo "Output: $OUTPUT"

TMPTEXT=$(mktemp /tmp/moss-tts-XXXX.txt)
printf '%s' "$TEXT" > "$TMPTEXT"
CMD=("$BINARY" -m "$MODEL" --audio-decoder-model "$DECODER" --text-file "$TMPTEXT" --wav-out "$OUTPUT" -ngl -1)

if [[ -n "$REF_AUDIO" ]]; then
    echo "Voice:  $REF_AUDIO"
    CMD+=(--audio-encoder-model "$ENCODER" --reference-audio "$REF_AUDIO")
else
    echo "Voice:  default"
fi

echo "Generating..."
START=$(date +%s%N)
"${CMD[@]}" 2>&1 | grep -vE "create_tensor|load:|print_info|llama_model_loader|init_tokenizer|sched_reserve|graph_reserve|set_|done_getting|load_tensors|llama_context|enumerating|backend_ptrs"
END=$(date +%s%N)

rm -f "$TMPTEXT"
ELAPSED=$(( (END - START) / 1000000 ))
echo "Done in ${ELAPSED}ms → $OUTPUT"
