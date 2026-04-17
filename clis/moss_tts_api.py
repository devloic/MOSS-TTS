"""
MOSS-TTS 8B REST API

POST /tts — Generate speech from text, returns MP3.

Usage:
    python clis/moss_tts_api.py --port 7864

Example requests:
    # List all voices (with IDs)
    curl http://127.0.0.1:7864/voices

    # List French voices only
    curl http://127.0.0.1:7864/voices?language=fr

    # Generate MP3 with voice ID
    curl -X POST http://127.0.0.1:7864/tts \
        -F "text=Bonjour le monde" \
        -F "voice=fr_henri" \
        -o output.mp3

    # With custom reference audio
    curl -X POST http://127.0.0.1:7864/tts \
        -F "text=Hello world" \
        -F "reference_audio=@my_voice.wav" \
        -o output.mp3

    # All options
    curl -X POST http://127.0.0.1:7864/tts \
        -F "text=Guten Tag" \
        -F "voice=de_conrad" \
        -F "language=de" \
        -F "audio_temperature=1.7" \
        -F "seed=42" \
        -F "format=wav" \
        -o output.wav

    # Health check
    curl http://127.0.0.1:7864/health
"""

import argparse
import io
import json
import os
import subprocess
import tempfile
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydub import AudioSegment
from scipy.signal import resample_poly

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SAMPLE_RATE = 24000

# Paths from environment
LLAMA_MOSS_TTS_INTERACTIVE = os.environ.get("MOSS_8B_BINARY_INTERACTIVE", "")
LLAMA_MOSS_TTS = os.environ.get("MOSS_8B_BINARY", "")
MODEL_Q4 = os.environ.get("MOSS_8B_MODEL", "")
MODEL_F16 = os.environ.get("MOSS_8B_MODEL_F16", "")
ENCODER_PATH = os.environ.get("MOSS_8B_ENCODER", "")
DECODER_PATH = os.environ.get("MOSS_8B_DECODER", "")
VOICES_DIR = PROJECT_DIR / "voices"

# Model registry: id -> {name, path, ngl, description}
MODEL_REGISTRY = {}
if MODEL_Q4 and Path(MODEL_Q4).exists():
    MODEL_REGISTRY["q4_k_m"] = {"name": "Q4_K_M", "path": MODEL_Q4, "ngl": -1, "description": "4.9 GB, full GPU, fast"}
MODEL_Q5 = str(Path(MODEL_Q4).parent / "MOSS_TTS_FIRST_CLASS_Q5_K_M.gguf") if MODEL_Q4 else ""
if MODEL_Q5 and Path(MODEL_Q5).exists():
    MODEL_REGISTRY["q5_k_m"] = {"name": "Q5_K_M", "path": MODEL_Q5, "ngl": -1, "description": "5.7 GB, full GPU, better quality"}
if MODEL_F16 and Path(MODEL_F16).exists():
    MODEL_REGISTRY["f16"] = {"name": "F16", "path": MODEL_F16, "ngl": 12, "description": "16 GB, mixed CPU/GPU, best quality"}
DEFAULT_MODEL_ID = list(MODEL_REGISTRY.keys())[0] if MODEL_REGISTRY else ""

# Voice registry: id -> {name, file, language, gender, accent}
VOICE_REGISTRY = {
    "en_ava":     {"name": "Ava",     "file": "en_2.wav",    "language": "en", "gender": "female", "accent": "American"},
    "en_bella":   {"name": "Bella",   "file": "en_3.wav",    "language": "en", "gender": "female", "accent": "American"},
    "en_adam":    {"name": "Adam",    "file": "en_4.wav",    "language": "en", "gender": "male",   "accent": "American"},
    "fr_henri":   {"name": "Henri",   "file": "fr_1.wav",    "language": "fr", "gender": "male",   "accent": "France"},
    "fr_denise":  {"name": "Denise",  "file": "fr_2.wav",    "language": "fr", "gender": "female", "accent": "France"},
    "es_alvaro":  {"name": "Alvaro",  "file": "es_1.wav",    "language": "es", "gender": "male",   "accent": "Spain"},
    "es_elvira":  {"name": "Elvira",  "file": "es_2.wav",    "language": "es", "gender": "female", "accent": "Spain"},
    "es_jorge":   {"name": "Jorge",   "file": "es_mx_1.wav", "language": "es", "gender": "male",   "accent": "Mexico"},
    "es_dalia":   {"name": "Dalia",   "file": "es_mx_2.wav", "language": "es", "gender": "female", "accent": "Mexico"},
    "es_tomas":   {"name": "Tomas",   "file": "es_ar_1.wav", "language": "es", "gender": "male",   "accent": "Argentina"},
    "es_elena":   {"name": "Elena",   "file": "es_ar_2.wav", "language": "es", "gender": "female", "accent": "Argentina"},
    "es_gonzalo": {"name": "Gonzalo", "file": "es_co_1.wav", "language": "es", "gender": "male",   "accent": "Colombia"},
    "es_salome":  {"name": "Salome",  "file": "es_co_2.wav", "language": "es", "gender": "female", "accent": "Colombia"},
    "de_conrad":  {"name": "Conrad",  "file": "de_1.wav",    "language": "de", "gender": "male",   "accent": "Germany"},
    "de_katja":   {"name": "Katja",   "file": "de_2.wav",    "language": "de", "gender": "female", "accent": "Germany"},
}


QUALITY_PRESETS = {
    "stable": {
        "name": "Stable",
        "description": "Clear and consistent, best for voice cloning",
        "audio_temperature": 0.8,
        "audio_top_p": 0.7,
        "audio_top_k": 15,
        "audio_repetition_penalty": 1.2,
    },
    "balanced": {
        "name": "Balanced",
        "description": "Official MOSS-TTS defaults, good all-round",
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
    },
    "expressive": {
        "name": "Expressive",
        "description": "More varied and emotional, may be less stable",
        "audio_temperature": 2.2,
        "audio_top_p": 0.9,
        "audio_top_k": 40,
        "audio_repetition_penalty": 1.0,
    },
}
DEFAULT_PRESET_ID = "balanced"


def _resolve_voice_file(voice_id: str) -> str | None:
    """Resolve a voice ID to a file path. Checks registry, then custom voices."""
    if voice_id in VOICE_REGISTRY:
        candidate = VOICES_DIR / VOICE_REGISTRY[voice_id]["file"]
        if candidate.exists():
            return str(candidate)
    # Check custom voices
    custom = VOICES_DIR / "custom" / f"{voice_id}.wav"
    if custom.exists():
        return str(custom)
    return None

# ── Interactive process manager ──────────────────────────────────────────────

_interactive_procs: dict[str, subprocess.Popen] = {}
_interactive_lock = threading.Lock()


def _get_interactive_proc(model_path: str, ngl: int) -> subprocess.Popen:
    key = f"{model_path}:{ngl}"
    with _interactive_lock:
        proc = _interactive_procs.get(key)
        if proc and proc.poll() is None:
            return proc

        cmd = [
            LLAMA_MOSS_TTS_INTERACTIVE,
            "-m", model_path,
            "--audio-decoder-model", DECODER_PATH,
            "-ngl", str(ngl),
            "--interactive",
        ]
        if ENCODER_PATH and Path(ENCODER_PATH).exists():
            cmd.extend(["--audio-encoder-model", ENCODER_PATH])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            env={**os.environ, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
        )

        while True:
            line = proc.stdout.readline()
            if not line:
                proc.kill()
                raise RuntimeError("Interactive process died during startup")
            if '"ready"' in line.strip():
                break

        _interactive_procs[key] = proc
        return proc


def _send_request(proc, request: dict) -> dict:
    line = json.dumps(request, ensure_ascii=False)
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

    while True:
        resp_line = proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("Interactive process died")
        resp_line = resp_line.strip()
        if resp_line.startswith("{"):
            return json.loads(resp_line)


def convert_reference_audio(audio_bytes: bytes) -> str:
    """Convert uploaded audio to 24kHz mono 16-bit PCM WAV."""
    tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_in.write(audio_bytes)
    tmp_in.close()

    data, sr = sf.read(tmp_in.name)
    os.unlink(tmp_in.name)

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 24000:
        g = gcd(sr, 24000)
        data = resample_poly(data, 24000 // g, sr // g).astype(np.float64)

    data_16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_out.name, data_16, 24000, subtype="PCM_16")
    return tmp_out.name


def wav_to_mp3(wav_path: str, bitrate: str = "192k") -> bytes:
    """Convert WAV to MP3 bytes using pydub."""
    audio = AudioSegment.from_wav(wav_path)
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


# ── FastAPI app ──────────────────────────────────────────────────────────────

from fastapi.responses import FileResponse as StaticFileResponse, HTMLResponse

app = FastAPI(title="MOSS-TTS API", version="1.0")

CLIENT_HTML = SCRIPT_DIR / "moss_tts_client.html"


@app.get("/", response_class=HTMLResponse)
def serve_client():
    """Serve the web client UI."""
    return CLIENT_HTML.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "default_model": DEFAULT_MODEL_ID, "models": list(MODEL_REGISTRY.keys())}


@app.get("/models")
def list_models():
    """List available models with IDs."""
    models = []
    for mid, info in MODEL_REGISTRY.items():
        models.append({
            "id": mid,
            "name": info["name"],
            "description": info["description"],
            "gpu_layers": info["ngl"],
        })
    return {"models": models, "default": DEFAULT_MODEL_ID}


@app.get("/presets")
def list_presets():
    """List quality presets with IDs and sampling parameters."""
    presets = []
    for pid, info in QUALITY_PRESETS.items():
        presets.append({
            "id": pid,
            "name": info["name"],
            "description": info["description"],
            "audio_temperature": info["audio_temperature"],
            "audio_top_p": info["audio_top_p"],
            "audio_top_k": info["audio_top_k"],
            "audio_repetition_penalty": info["audio_repetition_penalty"],
        })
    return {"presets": presets, "default": DEFAULT_PRESET_ID}


@app.get("/voices")
def list_voices(language: str = ""):
    """List all available voices. Optionally filter by language code (en, fr, es, de)."""
    voices = []
    for vid, info in VOICE_REGISTRY.items():
        if language and info["language"] != language:
            continue
        voices.append({
            "id": vid,
            "name": info["name"],
            "language": info["language"],
            "gender": info["gender"],
            "accent": info["accent"],
        })

    # Add custom voices (with metadata if available)
    custom_dir = VOICES_DIR / "custom"
    if custom_dir.exists():
        for f in sorted(custom_dir.glob("*.wav")):
            vid = f.stem
            meta_path = custom_dir / f"{vid}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                v_name = meta.get("name", vid.replace("_", " ").title())
                v_lang = meta.get("language", "any")
            else:
                v_name = vid.replace("_", " ").title()
                v_lang = "any"

            if language and v_lang != language and v_lang != "any":
                continue

            voices.append({
                "id": vid,
                "name": v_name,
                "language": v_lang,
                "gender": "unknown",
                "accent": "custom",
            })

    return {"voices": voices}


@app.post("/voices/save")
async def save_voice(
    name: str = Form(...),
    language: str = Form("en"),
    audio: UploadFile = File(...),
):
    """Save a recording as a custom voice for reuse."""
    name = name.strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)

    audio_bytes = await audio.read()
    if len(audio_bytes) < 100:
        return JSONResponse({"error": "audio too short"}, status_code=400)

    ref_path = convert_reference_audio(audio_bytes)
    safe_name = name.lower().replace(" ", "_")
    lang = language.strip().lower() or "en"

    custom_dir = VOICES_DIR / "custom"
    custom_dir.mkdir(parents=True, exist_ok=True)
    dest = custom_dir / f"{safe_name}.wav"

    # Save metadata (language, original name)
    meta_path = custom_dir / f"{safe_name}.json"
    meta = {"name": name, "language": lang, "id": safe_name}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    import shutil
    shutil.move(ref_path, str(dest))

    return {"id": safe_name, "name": name, "language": lang}


@app.post("/tts")
async def tts(
    text: str = Form(...),
    voice: str = Form(""),
    model: str = Form(""),
    preset: str = Form(""),
    language: str = Form(""),
    reference_audio: UploadFile = File(None),
    audio_temperature: float = Form(0),
    audio_top_p: float = Form(0),
    audio_top_k: int = Form(0),
    audio_repetition_penalty: float = Form(0),
    seed: int = Form(0),
    format: str = Form("mp3"),
):
    text = (text or "").strip()
    if not text:
        return JSONResponse({"error": "text is required"}, status_code=400)

    # Resolve preset: preset values are defaults, explicit params override
    preset_id = preset.strip() if preset else DEFAULT_PRESET_ID
    if preset_id not in QUALITY_PRESETS:
        return JSONResponse({"error": f"unknown preset: {preset_id}", "hint": "GET /presets for available IDs"}, status_code=400)
    p = QUALITY_PRESETS[preset_id]
    eff_temperature = audio_temperature if audio_temperature > 0 else p["audio_temperature"]
    eff_top_p = audio_top_p if audio_top_p > 0 else p["audio_top_p"]
    eff_top_k = audio_top_k if audio_top_k > 0 else p["audio_top_k"]
    eff_rep_penalty = audio_repetition_penalty if audio_repetition_penalty > 0 else p["audio_repetition_penalty"]

    # Resolve reference audio: uploaded file > voice ID
    ref_path = ""
    if reference_audio and reference_audio.size > 0:
        audio_bytes = await reference_audio.read()
        ref_path = convert_reference_audio(audio_bytes)
    elif voice:
        resolved = _resolve_voice_file(voice)
        if resolved:
            ref_path = resolved
        else:
            return JSONResponse({"error": f"unknown voice id: {voice}", "hint": "GET /voices for available IDs"}, status_code=400)

    # Generate
    out_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_file.close()

    # Resolve model
    model_id = model.strip() if model else DEFAULT_MODEL_ID
    if model_id and model_id not in MODEL_REGISTRY:
        return JSONResponse({"error": f"unknown model: {model_id}", "hint": "GET /models for available IDs"}, status_code=400)
    model_info = MODEL_REGISTRY.get(model_id, {})
    model_path = model_info.get("path", MODEL_Q4 or MODEL_F16)
    ngl = model_info.get("ngl", -1)

    try:
        if LLAMA_MOSS_TTS_INTERACTIVE and Path(LLAMA_MOSS_TTS_INTERACTIVE).exists():
            proc = _get_interactive_proc(model_path, ngl)
            request = {
                "text": text,
                "wav_out": out_file.name,
                "language": language,
                "reference_audio": ref_path,
                "seed": seed if seed > 0 else int(time.time()) % 999999,
                "max_new_tokens": 512,
                "audio_temperature": eff_temperature,
                "audio_top_p": eff_top_p,
                "audio_top_k": eff_top_k,
                "audio_repetition_penalty": eff_rep_penalty,
            }
            resp = _send_request(proc, request)
            if resp.get("status") != "ok":
                return JSONResponse({"error": resp.get("message", "generation failed")}, status_code=500)
        else:
            return JSONResponse({"error": "interactive binary not found"}, status_code=500)

        if not Path(out_file.name).exists() or Path(out_file.name).stat().st_size == 0:
            return JSONResponse({"error": "no audio generated"}, status_code=500)

        # Return audio
        fmt = format.lower()
        if fmt == "wav":
            audio_bytes = Path(out_file.name).read_bytes()
            media_type = "audio/wav"
            filename = "speech.wav"
        else:
            audio_bytes = wav_to_mp3(out_file.name)
            media_type = "audio/mpeg"
            filename = "speech.mp3"

        os.unlink(out_file.name)

        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        if Path(out_file.name).exists():
            os.unlink(out_file.name)
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOSS-TTS REST API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7864)
    args = parser.parse_args()

    print(f"MOSS-TTS API starting on http://{args.host}:{args.port}")
    print(f"  Model: {MODEL_Q4 or MODEL_F16}")
    print(f"  Docs:  http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)
