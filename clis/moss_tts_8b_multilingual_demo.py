"""
MOSS-TTS 8B Multilingual Demo — French, Spanish, English, German

Uses the 8B model via the first-class llama.cpp pipeline (Q4_K_M, full GPU).
All inference runs on GPU: backbone + LM heads + audio decoder.
24 kHz output. Supports quality presets and best-of-N generation.
"""

import os
import random
import subprocess
import tempfile
import time
from math import gcd
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_PORT = 7863
SAMPLE_RATE = 24000

# Paths from environment (set by run_multilingual_demo.sh)
LLAMA_MOSS_TTS = os.environ.get("MOSS_8B_BINARY", "")
LLAMA_MOSS_TTS_INTERACTIVE = os.environ.get("MOSS_8B_BINARY_INTERACTIVE", "")
MODEL_Q4 = os.environ.get("MOSS_8B_MODEL", "")
MODEL_F16 = os.environ.get("MOSS_8B_MODEL_F16", "")
ENCODER_PATH = os.environ.get("MOSS_8B_ENCODER", "")
DECODER_PATH = os.environ.get("MOSS_8B_DECODER", "")

# Model variants: name -> (gguf_path, n_gpu_layers)
MODEL_VARIANTS = {}
if MODEL_Q4 and Path(MODEL_Q4).exists():
    MODEL_VARIANTS["Q4_K_M (full GPU, fast)"] = (MODEL_Q4, -1)
MODEL_Q5 = str(Path(MODEL_Q4).parent / "MOSS_TTS_FIRST_CLASS_Q5_K_M.gguf") if MODEL_Q4 else ""
if MODEL_Q5 and Path(MODEL_Q5).exists():
    MODEL_VARIANTS["Q5_K_M (full GPU, better quality)"] = (MODEL_Q5, -1)
if MODEL_F16 and Path(MODEL_F16).exists():
    MODEL_VARIANTS["F16 (mixed CPU/GPU, best quality)"] = (MODEL_F16, 12)

# Reference voice audio (24kHz 16-bit PCM, compatible with first-class binary)
VOICES_DIR = PROJECT_DIR / "voices"

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}

VOICE_PRESETS = {
    "Ava (English female)":    ("en_2.wav", "English female voice A"),
    "Bella (English female)":  ("en_3.wav", "English female voice B"),
    "Adam (English male)":     ("en_4.wav", "English male voice A"),
    "Henri (French male)":     ("fr_1.wav", "French male voice"),
    "Denise (French female)":  ("fr_2.wav", "French female voice"),
    "Alvaro (Spanish male)":       ("es_1.wav", "Spain male voice"),
    "Elvira (Spanish female)":     ("es_2.wav", "Spain female voice"),
    "Jorge (Mexican male)":        ("es_mx_1.wav", "Mexican male voice"),
    "Dalia (Mexican female)":      ("es_mx_2.wav", "Mexican female voice"),
    "Tomas (Argentine male)":      ("es_ar_1.wav", "Argentine male voice"),
    "Elena (Argentine female)":    ("es_ar_2.wav", "Argentine female voice"),
    "Gonzalo (Colombian male)":    ("es_co_1.wav", "Colombian male voice"),
    "Salome (Colombian female)":   ("es_co_2.wav", "Colombian female voice"),
    "Conrad (German male)":    ("de_1.wav", "German male voice"),
    "Katja (German female)":   ("de_2.wav", "German female voice"),
}

LANGUAGE_DEFAULT_VOICE = {
    "English": "Ava (English female)",
    "French": "Henri (French male)",
    "Spanish": "Jorge (Mexican male)",
    "German": "Conrad (German male)",
}

EXAMPLE_TEXTS = {
    "English": [
        "In the quiet hours before dawn, the world looks unfinished.\nStreets are empty, windows are dark, and the air holds its breath as if waiting for a cue.\nBut beneath the stillness, everything is moving.\nWater is traveling through pipes. Electricity is humming along invisible lines.\nSeeds are pushing against soil.\nSomewhere, a hand reaches for a switch, and a day begins.",
        "Look, I know what you're thinking. Here he goes again.\nThe guy in the suit, about to make a speech like it's a press conference.\nRelax. This one isn't for the cameras.\nNo sponsors, no applause, no clever angle.\nI just want to say something honest for once.",
        "Tonight, I just want to take a second and breathe this in with you.\nBecause moments like this don't happen by accident.\nThey're built, one step at a time, one brave decision at a time.\nThey're built by people who keep showing up,\neven when life is loud,\neven when the world is heavy.",
    ],
    "French": [
        "Dans les heures calmes avant l'aube, le monde semble inachevé.\nLes rues sont vides, les fenêtres sont sombres,\net l'air retient son souffle comme s'il attendait un signal.\nMais sous cette quiétude, tout bouge.\nL'eau voyage dans les tuyaux, l'électricité fredonne le long de fils invisibles.\nQuelque part, une main se tend vers un interrupteur, et une journée commence.",
        "Tu sais, il y a des jours où tout semble difficile.\nOù tu te demandes si ça vaut la peine de continuer.\nMais regarde autour de toi.\nChaque personne que tu croises mène un combat que tu ne connais pas.\nEt pourtant, on avance tous, pas après pas.\nC'est ça, le courage. Pas l'absence de peur,\nmais la décision de marcher malgré elle.",
        "Ce soir, je voudrais simplement m'arrêter un instant\net respirer ce moment avec vous.\nParce que des instants comme celui-ci ne sont pas le fruit du hasard.\nIls se construisent, un mot à la fois, une nuit blanche à la fois,\nune décision courageuse à la fois.\nIls sont faits par des gens qui continuent à se montrer,\nmême quand la vie est bruyante.",
    ],
    "Spanish": [
        "Mira, hay algo que necesito decirte.\nNo es facil, y probablemente no es lo que esperas escuchar.\nPero a veces las verdades mas importantes son las que nadie se atreve a pronunciar.\nEl mundo no te debe nada.\nPero tu te debes a ti mismo la honestidad de intentarlo,\nde levantarte cada manana y elegir ser mejor que ayer.",
        "En las horas tranquilas antes del amanecer, el mundo parece incompleto.\nLas calles estan vacias, las ventanas oscuras,\ny el aire contiene la respiracion como esperando una senal.\nPero debajo de esa quietud, todo se mueve.\nEl agua viaja por las tuberias, la electricidad zumba por cables invisibles.\nEn algun lugar, una mano enciende un interruptor, y un nuevo dia comienza.",
        "Esta noche, solo quiero tomarme un segundo\ny respirar este momento con ustedes.\nPorque momentos como este no suceden por accidente.\nSe construyen, un paso a la vez, una decision valiente a la vez.\nLos construyen personas que siguen apareciendo,\nincluso cuando la vida es ruidosa,\nincluso cuando el mundo pesa demasiado.",
    ],
    "German": [
        "In den stillen Stunden vor der Morgendämmerung wirkt die Welt unvollendet.\nDie Straßen sind leer, die Fenster dunkel,\nund die Luft hält den Atem an, als warte sie auf ein Zeichen.\nDoch unter der Stille bewegt sich alles.\nWasser fließt durch Rohre, Elektrizität summt entlang unsichtbarer Leitungen.\nIrgendwo greift eine Hand nach einem Schalter, und ein Tag beginnt.",
        "Schau, ich weiß was du denkst. Da geht er wieder.\nDer Typ im Anzug, der eine Rede halten will,\nals wäre es eine Pressekonferenz.\nEntspann dich. Diese hier ist nicht für die Kameras.\nKeine Sponsoren, kein Applaus.\nIch möchte einfach nur einmal etwas Ehrliches sagen.",
        "Heute Abend möchte ich mir nur einen Moment nehmen\nund diesen Augenblick mit euch einatmen.\nDenn solche Momente passieren nicht zufällig.\nSie werden aufgebaut, ein Schritt nach dem anderen,\neine mutige Entscheidung nach der anderen.\nSie werden von Menschen geschaffen, die immer wieder auftauchen,\nauch wenn das Leben laut ist,\nauch wenn die Welt schwer wiegt.",
    ],
}

# ── Quality presets ──────────────────────────────────────────────────────────

QUALITY_PRESETS = {
    "Stable": {
        "audio_temperature": 0.8,
        "audio_top_p": 0.7,
        "audio_top_k": 15,
        "audio_repetition_penalty": 1.2,
        "description": "Clear and consistent. Best for voice cloning.",
    },
    "Balanced": {
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
        "description": "Official MOSS-TTS defaults. Good all-round.",
    },
    "Expressive": {
        "audio_temperature": 2.2,
        "audio_top_p": 0.9,
        "audio_top_k": 40,
        "audio_repetition_penalty": 1.0,
        "description": "More varied and emotional. May be less stable.",
    },
}

# ── Voice management ─────────────────────────────────────────────────────────

CUSTOM_VOICES_DIR = VOICES_DIR / "custom"
CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)


def get_saved_voices() -> dict[str, str]:
    saved = {}
    for f in sorted(CUSTOM_VOICES_DIR.glob("*.wav")):
        name = f.stem.replace("_", " ").title()
        saved[f"* {name} (saved)"] = str(f)
    return saved


def get_voices_for_language(language: str) -> list[str]:
    lang_lower = language.lower()
    native = [k for k in VOICE_PRESETS if lang_lower in k.lower()]
    others = [k for k in VOICE_PRESETS if lang_lower not in k.lower()]
    saved = list(get_saved_voices().keys())
    return saved + native + others


def resolve_ref_path(voice: str, reference_audio: str | None) -> tuple[str | None, str]:
    """Returns (ref_path, voice_label)."""
    if reference_audio:
        data, sr = sf.read(reference_audio)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 24000:
            g = gcd(sr, 24000)
            data = resample_poly(data, 24000 // g, sr // g).astype(np.float64)
        data_16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        ref_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(ref_tmp.name, data_16, 24000, subtype="PCM_16")
        return ref_tmp.name, "custom reference"

    if voice in VOICE_PRESETS:
        candidate = str(VOICES_DIR / VOICE_PRESETS[voice][0])
        if Path(candidate).exists():
            return candidate, voice

    saved = get_saved_voices()
    if voice in saved and Path(saved[voice]).exists():
        return saved[voice], voice

    return None, "default"


# ── Interactive process manager ───────────────────────────────────────────────

import json
import threading

_interactive_procs: dict[str, subprocess.Popen] = {}
_interactive_lock = threading.Lock()


def _get_interactive_proc(model_path: str, ngl: int) -> subprocess.Popen:
    """Get or start a persistent interactive llama-moss-tts process for this model."""
    key = f"{model_path}:{ngl}"
    with _interactive_lock:
        proc = _interactive_procs.get(key)
        if proc and proc.poll() is None:
            return proc

        # Start new interactive process
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
            stderr=None,  # let stderr go to console for debug logs
            text=True,
            env={**os.environ, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
        )

        # Wait for "ready" response (model loading takes time)
        while True:
            ready_line = proc.stdout.readline()
            if not ready_line:
                proc.kill()
                raise RuntimeError("Interactive process died during startup")
            ready_line = ready_line.strip()
            if '"ready"' in ready_line:
                break

        _interactive_procs[key] = proc
        return proc


def _send_interactive_request(proc: subprocess.Popen, request: dict) -> dict:
    """Send a JSON request to the interactive process and read the JSON response."""
    line = json.dumps(request, ensure_ascii=False)
    proc.stdin.write(line + "\n")
    proc.stdin.flush()

    # Read lines until we get a valid JSON response (skip log output)
    while True:
        resp_line = proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("Interactive process died during generation")
        resp_line = resp_line.strip()
        if not resp_line:
            continue
        if resp_line.startswith("{"):
            return json.loads(resp_line)


# ── Generation ───────────────────────────────────────────────────────────────

def generate_single(text: str, ref_path: str | None, audio_temp: float,
                    audio_top_p: float, audio_top_k: int,
                    audio_rep_penalty: float, seed: int | None,
                    model_variant: str = "", gpu_layers_override: int = -1) -> tuple[np.ndarray, int]:
    """Run a single generation. Returns (audio_data, sample_rate)."""
    variant = MODEL_VARIANTS.get(model_variant)
    if variant:
        model_path, default_ngl = variant
    else:
        model_path, default_ngl = MODEL_Q4, -1

    ngl = int(gpu_layers_override) if "F16" in model_variant else -1

    out_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out_file.close()

    # Use interactive mode if available (keeps model loaded between requests)
    if LLAMA_MOSS_TTS_INTERACTIVE and Path(LLAMA_MOSS_TTS_INTERACTIVE).exists():
        proc = _get_interactive_proc(model_path, ngl)
        request = {
            "text": text,
            "wav_out": out_file.name,
            "reference_audio": ref_path or "",
            "seed": seed if seed is not None else random.randint(1, 999999),
            "max_new_tokens": 512,
            "audio_temperature": audio_temp,
            "audio_top_p": audio_top_p,
            "audio_top_k": audio_top_k,
            "audio_repetition_penalty": audio_rep_penalty,
        }
        resp = _send_interactive_request(proc, request)
        if resp.get("status") != "ok":
            raise gr.Error(f"Generation failed: {resp.get('message', 'unknown error')}")
    else:
        # Fallback: one-shot subprocess
        text_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        text_file.write(text)
        text_file.close()

        cmd = [
            LLAMA_MOSS_TTS,
            "-m", model_path,
            "--audio-decoder-model", DECODER_PATH,
            "--text-file", text_file.name,
            "--wav-out", out_file.name,
            "-ngl", str(ngl),
            "--max-new-tokens", "512",
            "--audio-temperature", str(audio_temp),
            "--audio-top-p", str(audio_top_p),
            "--audio-top-k", str(audio_top_k),
            "--audio-repetition-penalty", str(audio_rep_penalty),
        ]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        if ref_path:
            cmd.extend(["--audio-encoder-model", ENCODER_PATH])
            cmd.extend(["--reference-audio", ref_path])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env={**os.environ, "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")},
            )
        finally:
            os.unlink(text_file.name)

        if result.returncode != 0:
            error_msg = (result.stderr or "")[-500:]
            raise gr.Error(f"Generation failed (exit {result.returncode}): {error_msg}")

    if not Path(out_file.name).exists() or Path(out_file.name).stat().st_size == 0:
        raise gr.Error("No audio generated.")

    audio_data, sample_rate = sf.read(out_file.name, dtype="int16")
    os.unlink(out_file.name)
    return audio_data, sample_rate


def run_inference(text, language, voice, reference_audio,
                  model_variant, gpu_layers_val, preset, audio_temp, audio_top_p, audio_top_k,
                  audio_rep_penalty, seed_text, multi_variants):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter text to synthesize.")

    ref_path, voice_label = resolve_ref_path(voice, reference_audio)

    # Parse seed
    seed_val = None
    seed_str = (seed_text or "").strip()
    if seed_str:
        try:
            seed_val = int(seed_str)
        except ValueError:
            raise gr.Error("Seed must be a number or empty for random.")

    n_variants = int(multi_variants) if multi_variants else 1
    started_at = time.monotonic()

    model_label = model_variant.split(" (")[0] if model_variant else "Q4_K_M"
    ngl_display = int(gpu_layers_val) if "F16" in (model_variant or "") else "all"
    model_label = f"{model_label} (ngl={ngl_display})" if ngl_display != "all" else model_label

    if n_variants <= 1:
        audio_data, sample_rate = generate_single(
            text, ref_path, audio_temp, audio_top_p, audio_top_k,
            audio_rep_penalty, seed_val, model_variant, gpu_layers_val,
        )
        elapsed = time.monotonic() - started_at
        duration = len(audio_data) / sample_rate
        rtf = elapsed / duration if duration > 0 else 0
        seed_info = f"seed={seed_val}" if seed_val is not None else "seed=random"
        status = (
            f"Done | {language} | {voice_label} | {elapsed:.1f}s gen | {duration:.1f}s audio | RTF {rtf:.1f}x\n"
            f"Model: {model_label} | {preset} | temp={audio_temp:.1f} top_p={audio_top_p:.2f} top_k={audio_top_k} | {seed_info}"
        )
        return (
            (sample_rate, audio_data),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            status,
        )
    else:
        # Best-of-N: generate multiple variants with different seeds
        results = []
        base_seed = seed_val if seed_val is not None else random.randint(1, 999999)
        for i in range(n_variants):
            s = base_seed + i
            audio_data, sample_rate = generate_single(
                text, ref_path, audio_temp, audio_top_p, audio_top_k,
                audio_rep_penalty, s, model_variant,
            )
            results.append((sample_rate, audio_data, s))

        elapsed = time.monotonic() - started_at
        seeds_used = ", ".join(str(r[2]) for r in results)
        status = (
            f"Done | {language} | {voice_label} | {n_variants} variants in {elapsed:.1f}s\n"
            f"Preset: {preset} | temp={audio_temp:.1f} top_p={audio_top_p:.2f} top_k={audio_top_k} | seeds: {seeds_used}"
        )

        outputs = []
        for i in range(3):
            if i < len(results):
                outputs.append(gr.update(value=(results[i][0], results[i][1]), visible=True,
                                         label=f"Variant {i+1} (seed {results[i][2]})"))
            else:
                outputs.append(gr.update(visible=False))

        return (
            (results[0][0], results[0][1]),  # main audio = first variant
            outputs[0], outputs[1], outputs[2],
            status,
        )


# ── UI ───────────────────────────────────────────────────────────────────────

APP_CSS = """
:root { --accent: #b45309; --panel: #ffffff; --line: #e5e7eb; --muted: #4d5562; }
.gradio-container { background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%); }
.card-8b { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); padding: 14px; }
.title-8b { font-size: 24px; font-weight: 700; margin-bottom: 4px; color: #000000; }
.subtitle-8b { color: var(--muted); font-size: 14px; margin-bottom: 8px; }
.badge-8b { display: inline-block; color: white; font-size: 11px;
            padding: 2px 8px; border-radius: 8px; margin-left: 8px; vertical-align: middle; }
#run-btn { background: var(--accent); border: none; font-size: 16px; }
.example-btn { text-align: left !important; white-space: normal !important; line-height: 1.4; padding: 8px 12px !important; height: auto !important; }
"""


def build_demo():
    with gr.Blocks(title="MOSS-TTS 8B Multilingual Demo") as demo:
        gr.Markdown(
            """
            <div class="card-8b">
              <div class="title-8b">MOSS-TTS 8B Multilingual Demo
                <span class="badge-8b" style="background:#b45309">8B Q4_K_M</span>
                <span class="badge-8b" style="background:#16a34a">Full GPU (first-class)</span>
              </div>
              <div class="subtitle-8b">High-quality TTS in English, French, Spanish, German — 24 kHz, all on GPU</div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Language",
                )
                text_input = gr.Textbox(
                    label="Speech text",
                    lines=5,
                    placeholder="Enter text to synthesize...",
                    value=EXAMPLE_TEXTS["English"][0],
                )

                with gr.Accordion("Voice", open=True):
                    voice = gr.Dropdown(
                        choices=get_voices_for_language("English"),
                        value=LANGUAGE_DEFAULT_VOICE["English"],
                        label="Preset Voice",
                        info="Native voices listed first; or upload your own below",
                    )
                    gr.Markdown("**Example texts** (click to use)")
                    example_btns = []
                    init_texts = EXAMPLE_TEXTS["English"]
                    for i in range(3):
                        btn = gr.Button(
                            value=init_texts[i] if i < len(init_texts) else "",
                            elem_classes=["example-btn"],
                            variant="secondary",
                            visible=i < len(init_texts),
                        )
                        example_btns.append(btn)
                    reference_audio = gr.Audio(
                        label="Custom Reference Audio (overrides preset voice)",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )
                    with gr.Row():
                        save_name = gr.Textbox(
                            label="Voice name",
                            placeholder="e.g. My Voice French",
                            scale=3,
                        )
                        save_btn = gr.Button("Save Voice", scale=1, variant="secondary")
                    save_status = gr.Markdown("")

            with gr.Column(scale=2):
                # Model variant
                default_model = list(MODEL_VARIANTS.keys())[0] if MODEL_VARIANTS else ""
                model_variant = gr.Dropdown(
                    choices=list(MODEL_VARIANTS.keys()),
                    value=default_model,
                    label="Model",
                    info="Q4_K_M = fast, full GPU | F16 = best quality, mixed CPU/GPU (slower)",
                )
                gpu_layers_row = gr.Row(visible=False)
                with gpu_layers_row:
                    gpu_layers = gr.Slider(
                        0, 36, step=1, value=12,
                        label="GPU layers",
                        info="0 = all CPU | 12 = safe default | 20+ = faster, needs free VRAM",
                    )

                # Quality preset
                preset = gr.Dropdown(
                    choices=list(QUALITY_PRESETS.keys()),
                    value="Balanced",
                    label="Quality Preset",
                    info="Stable = clear cloning | Balanced = defaults | Expressive = emotional",
                )

                with gr.Accordion("Fine-tune sampling", open=False):
                    audio_temp = gr.Slider(0.1, 3.0, step=0.1, value=1.7, label="Audio Temperature",
                                           info="Lower = stable, higher = expressive")
                    audio_top_p = gr.Slider(0.1, 1.0, step=0.05, value=0.8, label="Audio Top-p")
                    audio_top_k = gr.Slider(1, 100, step=1, value=25, label="Audio Top-k")
                    audio_rep_penalty = gr.Slider(0.8, 2.0, step=0.05, value=1.0, label="Repetition Penalty",
                                                   info="Higher = reduces slurring")
                    seed_text = gr.Textbox(label="Seed", placeholder="Empty = random", value="")
                    multi_variants = gr.Slider(1, 4, step=1, value=1,
                                               label="Generate variants",
                                               info="1 = single, 2-4 = best-of-N (compare and pick)")

                run_btn = gr.Button("Generate Speech", variant="primary", elem_id="run-btn")
                output_audio = gr.Audio(label="Output Audio", type="numpy")

                # Extra variant players (hidden by default)
                variant_audio_1 = gr.Audio(label="Variant 1", type="numpy", visible=False)
                variant_audio_2 = gr.Audio(label="Variant 2", type="numpy", visible=False)
                variant_audio_3 = gr.Audio(label="Variant 3", type="numpy", visible=False)

                status = gr.Textbox(label="Status", lines=3, interactive=False)

                with gr.Accordion("GPU Memory", open=False):
                    gpu_info = gr.Markdown("")
                    with gr.Row():
                        gpu_refresh_btn = gr.Button("Refresh", scale=1, variant="secondary", size="sm")
                        gpu_kill_btn = gr.Button("Free GPU memory", scale=1, variant="stop", size="sm")
                    gpu_kill_status = gr.Markdown("")

        # ── GPU management ───────────────────────────────────────────────────
        def get_gpu_info():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                mem_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
                used, free, total = mem_result.stdout.strip().split(", ")
                md = f"**VRAM:** {used.strip()} used / {free.strip()} free / {total.strip()} total\n\n"
                if lines:
                    md += "| PID | Process | VRAM |\n|-----|---------|------|\n"
                    for line in lines:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == 3:
                            md += f"| {parts[0]} | {parts[1]} | {parts[2]} |\n"
                else:
                    md += "No GPU processes found."
                return md
            except Exception as e:
                return f"Error: {e}"

        def kill_gpu_processes():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
                killed = []
                my_pid = os.getpid()
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        pid = int(parts[0])
                        name = parts[1]
                        if pid == my_pid:
                            continue
                        try:
                            os.kill(pid, 9)
                            killed.append(f"{name} (PID {pid})")
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            killed.append(f"{name} (PID {pid}) - permission denied")
                if killed:
                    return get_gpu_info(), "Killed: " + ", ".join(killed)
                return get_gpu_info(), "No other GPU processes to kill."
            except Exception as e:
                return get_gpu_info(), f"Error: {e}"

        gpu_refresh_btn.click(fn=get_gpu_info, outputs=[gpu_info])
        gpu_kill_btn.click(fn=kill_gpu_processes, outputs=[gpu_info, gpu_kill_status])

        # Auto-refresh GPU info on page load
        demo.load(fn=get_gpu_info, outputs=[gpu_info])

        # ── Show/hide GPU layers when model changes ─────────────────────────
        def on_model_change(variant_name):
            show = "F16" in (variant_name or "")
            return gr.update(visible=show)

        model_variant.change(
            fn=on_model_change,
            inputs=[model_variant],
            outputs=[gpu_layers_row],
            concurrency_limit=None,
        )

        # ── Preset changes fill sliders ──────────────────────────────────────
        def apply_preset(preset_name):
            p = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS["Balanced"])
            return p["audio_temperature"], p["audio_top_p"], p["audio_top_k"], p["audio_repetition_penalty"]

        preset.change(
            fn=apply_preset,
            inputs=[preset],
            outputs=[audio_temp, audio_top_p, audio_top_k, audio_rep_penalty],
        )

        # ── Save voice ───────────────────────────────────────────────────────
        def save_voice(audio_path, name, current_language):
            if not audio_path:
                return gr.update(), gr.update(), "No audio to save. Record or upload first."
            name = (name or "").strip()
            if not name:
                return gr.update(), gr.update(), "Enter a name for the voice."
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != 24000:
                g = gcd(sr, 24000)
                data = resample_poly(data, 24000 // g, sr // g).astype(np.float64)
            data_16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
            safe_name = name.lower().replace(" ", "_")
            out_path = CUSTOM_VOICES_DIR / f"{safe_name}.wav"
            sf.write(str(out_path), data_16, 24000, subtype="PCM_16")
            voices = get_voices_for_language(current_language)
            display_name = f"* {name.replace('_', ' ').title()} (saved)"
            return (
                gr.Dropdown(choices=voices, value=display_name),
                "",
                f"Saved as **{name}** ({len(data_16)/24000:.1f}s)",
            )

        save_btn.click(
            fn=save_voice,
            inputs=[reference_audio, save_name, language],
            outputs=[voice, save_name, save_status],
        )

        # ── Language change ──────────────────────────────────────────────────
        example_state = gr.State(EXAMPLE_TEXTS["English"])

# IMPORTANT: Use a SINGLE .change() handler for language.
        # Multiple .change() handlers on the same component cause Gradio 6
        # hot-reload to reset outputs not in a handler's output list,
        # replacing the textbox value with its label.
        def on_language_change(lang):
            texts = EXAMPLE_TEXTS.get(lang, [])
            voices = get_voices_for_language(lang)
            default_voice = LANGUAGE_DEFAULT_VOICE.get(lang, voices[0])
            btn0 = texts[0] if len(texts) > 0 else ""
            btn1 = texts[1] if len(texts) > 1 else ""
            btn2 = texts[2] if len(texts) > 2 else ""
            return (
                texts[0] if texts else "",
                gr.Dropdown(choices=voices, value=default_voice),
                gr.Button(value=btn0, visible=bool(btn0)),
                gr.Button(value=btn1, visible=bool(btn1)),
                gr.Button(value=btn2, visible=bool(btn2)),
                texts,
            )

        language.change(
            fn=on_language_change,
            inputs=[language],
            outputs=[text_input, voice, example_btns[0], example_btns[1], example_btns[2], example_state],
        )

        # ── Example clicks ───────────────────────────────────────────────────
        example_btns[0].click(fn=lambda s: s[0] if s else "", inputs=[example_state], outputs=[text_input])
        example_btns[1].click(fn=lambda s: s[1] if len(s) > 1 else "", inputs=[example_state], outputs=[text_input])
        example_btns[2].click(fn=lambda s: s[2] if len(s) > 2 else "", inputs=[example_state], outputs=[text_input])

        # ── Generate ─────────────────────────────────────────────────────────
        run_btn.click(
            fn=run_inference,
            inputs=[
                text_input, language, voice, reference_audio,
                model_variant, gpu_layers, preset, audio_temp, audio_top_p, audio_top_k,
                audio_rep_penalty, seed_text, multi_variants,
            ],
            outputs=[output_audio, variant_audio_1, variant_audio_2, variant_audio_3, status],
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        css=APP_CSS,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
