"""
MOSS-TTS-Nano Multilingual Demo — French, Spanish, English, German

A lightweight Gradio app using the ~100M parameter MOSS-TTS-Nano model.
Supports voice cloning with optional reference audio.
Runs on CPU (4 cores) or GPU. 48 kHz stereo output.
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress torchaudio deprecation warnings and duplicate WeText logs
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*StreamReader.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*StreamWriter.*", category=UserWarning)
logging.getLogger("wetext-zh_normalizer").setLevel(logging.WARNING)
logging.getLogger("wetext-en_normalizer").setLevel(logging.WARNING)

import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torch.cuda
import torchaudio

# MOSS-TTS-Nano repo must be on sys.path
NANO_REPO = Path(__file__).resolve().parent.parent.parent / "MOSS-TTS-Nano"
if str(NANO_REPO) not in sys.path:
    sys.path.insert(0, str(NANO_REPO))

from moss_tts_nano_runtime import NanoTTSService, build_default_voice_presets
from text_normalization_pipeline import WeTextProcessingManager, prepare_tts_request_texts

DEFAULT_PORT = 7862

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}

NANO_AUDIO_DIR = NANO_REPO / "assets" / "audio"

# Voice presets: name -> (audio file, description)
VOICE_PRESETS = {
    # English
    "Ava (English female)":    ("en_2.wav", "English female voice A"),
    "Bella (English female)":  ("en_3.wav", "English female voice B"),
    "Adam (English male)":     ("en_4.wav", "English male voice A"),
    # French
    "Henri (French male)":     ("fr_1.wav", "French male voice"),
    "Denise (French female)":  ("fr_2.wav", "French female voice"),
    # Spanish
    "Alvaro (Spanish male)":   ("es_1.wav", "Spanish male voice"),
    "Elvira (Spanish female)": ("es_2.wav", "Spanish female voice"),
    # German
    "Conrad (German male)":    ("de_1.wav", "German male voice"),
    "Katja (German female)":   ("de_2.wav", "German female voice"),
}

# Map language to its default voice
LANGUAGE_DEFAULT_VOICE = {
    "English": "Ava (English female)",
    "French": "Henri (French male)",
    "Spanish": "Alvaro (Spanish male)",
    "German": "Conrad (German male)",
}

def get_voices_for_language(language: str) -> list[str]:
    """Return voice names matching a language, plus all others."""
    lang_lower = language.lower()
    native = [k for k in VOICE_PRESETS if lang_lower in k.lower()]
    others = [k for k in VOICE_PRESETS if lang_lower not in k.lower()]
    return native + others

EXAMPLE_TEXTS = {
    "English": [
        "Good morning. Today we're going to talk about the future of artificial intelligence and how it will transform our daily lives in the coming decades.",
        "The sun was setting behind the mountains, painting the sky in shades of orange and purple. She stood at the edge of the cliff, feeling the cool wind against her face.",
        "Welcome to our podcast. In this episode, we explore the fascinating world of deep-sea creatures and their remarkable adaptations to extreme environments.",
    ],
    "French": [
        "Bonjour et bienvenue. Aujourd'hui, nous allons explorer les merveilles de la cuisine française et découvrir les secrets des grands chefs étoilés.",
        "La pluie tombait doucement sur les toits de Paris. Les passants pressaient le pas, leurs parapluies formant une mosaïque colorée sur les boulevards.",
        "Il est essentiel de comprendre que l'intelligence artificielle n'est pas simplement un outil technologique, mais une révolution qui transforme notre façon de vivre et de travailler.",
    ],
    "Spanish": [
        "Buenos días a todos. Hoy vamos a hablar sobre la importancia de la sostenibilidad y cómo cada uno de nosotros puede contribuir a un futuro mejor.",
        "El mar brillaba bajo el sol de mediodía mientras las olas rompían suavemente contra la orilla. Los niños jugaban en la arena, construyendo castillos efímeros.",
        "La música es un lenguaje universal que trasciende fronteras y culturas. Desde los ritmos del flamenco hasta las melodías del tango, cada género cuenta una historia única.",
    ],
    "German": [
        "Guten Morgen und herzlich willkommen. Heute sprechen wir über die Zukunft der erneuerbaren Energien und ihre Bedeutung für unsere Gesellschaft.",
        "Der Herbstwind wehte durch die bunten Blätter der alten Eiche. Im Garten lagen Äpfel verstreut, und der Duft von frisch gebackenem Brot zog aus dem Haus.",
        "Die Wissenschaft hat in den letzten Jahrzehnten enorme Fortschritte gemacht. Besonders im Bereich der Quantencomputer eröffnen sich völlig neue Möglichkeiten.",
    ],
}

_service: NanoTTSService | None = None
_text_normalizer: WeTextProcessingManager | None = None


TARGET_SR = 48000
TARGET_CHANNELS = 2
MAX_REF_DURATION = 15.0  # seconds — longer refs don't improve cloning
MIN_REF_DURATION = 1.0   # seconds — too short gives bad results


def preprocess_reference_audio(audio_path: str) -> str:
    """Resample to 48kHz stereo WAV, trim silence, and validate duration."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    # Convert to stereo if mono
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(TARGET_CHANNELS, 1)
    elif waveform.shape[0] > TARGET_CHANNELS:
        waveform = waveform[:TARGET_CHANNELS]

    # Normalize volume
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * 0.95

    # Trim leading/trailing silence (threshold -40dB)
    energy = waveform.abs().max(dim=0).values
    threshold = 0.01
    active = (energy > threshold).nonzero(as_tuple=True)[0]
    if len(active) > 0:
        start = max(0, active[0].item() - int(TARGET_SR * 0.05))
        end = min(waveform.shape[1], active[-1].item() + int(TARGET_SR * 0.05))
        waveform = waveform[:, start:end]

    # Enforce duration limits
    duration = waveform.shape[1] / TARGET_SR
    if duration > MAX_REF_DURATION:
        waveform = waveform[:, :int(TARGET_SR * MAX_REF_DURATION)]
    if duration < MIN_REF_DURATION:
        raise gr.Error(f"Reference audio too short ({duration:.1f}s). Record at least {MIN_REF_DURATION}s.")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    torchaudio.save(tmp.name, waveform, TARGET_SR)
    return tmp.name


def get_service() -> NanoTTSService:
    assert _service is not None, "Service not initialized"
    return _service


def run_inference(
    text: str,
    language: str,
    voice: str,
    reference_audio: str | None,
    audio_temperature: float,
    audio_top_p: float,
    audio_top_k: int,
    audio_repetition_penalty: float,
    max_new_frames: int,
):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter text to synthesize.")

    service = get_service()
    started_at = time.monotonic()

    # Normalize text
    prepared = prepare_tts_request_texts(
        text=text,
        prompt_text="",
        voice="",
        enable_wetext=_text_normalizer is not None,
        enable_normalize_tts_text=True,
        text_normalizer_manager=_text_normalizer,
    )
    normalized_text = str(prepared["text"])

    # Determine prompt audio: custom upload > preset voice > continuation
    prompt_audio_path = None
    voice_label = voice or ""
    if reference_audio:
        prompt_audio_path = preprocess_reference_audio(reference_audio)
        voice_label = "custom reference"
    elif voice in VOICE_PRESETS:
        audio_file = VOICE_PRESETS[voice][0]
        prompt_audio_path = str(NANO_AUDIO_DIR / audio_file)
        voice_label = voice

    synth_kwargs = dict(
        text=normalized_text,
        mode="voice_clone",
        prompt_audio_path=prompt_audio_path,
        voice=None,
        max_new_frames=int(max_new_frames),
        do_sample=True,
        audio_temperature=float(audio_temperature),
        audio_top_p=float(audio_top_p),
        audio_top_k=int(audio_top_k),
        audio_repetition_penalty=float(audio_repetition_penalty),
    )

    if not prompt_audio_path:
        synth_kwargs["mode"] = "continuation"
        synth_kwargs.pop("prompt_audio_path")
        synth_kwargs.pop("voice")
        voice_label = "no voice (continuation)"

    result = service.synthesize(**synth_kwargs)

    # Read generated audio
    audio_path = str(result["audio_path"])
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    elapsed = time.monotonic() - started_at

    # Device & GPU info
    device = service.device
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats(device)
        device_info = f"GPU: {gpu_name} | VRAM peak: {mem_used:.0f} MB"
    else:
        device_info = "CPU"

    status = (
        f"Done | {language} | voice: {voice_label} | {elapsed:.2f}s | {sample_rate} Hz\n"
        f"{device_info}\n"
        f"audio_temperature={audio_temperature:.2f}, audio_top_p={audio_top_p:.2f}, "
        f"audio_top_k={audio_top_k}, audio_repetition_penalty={audio_repetition_penalty:.2f}"
    )
    return (sample_rate, audio_data), status



def _device_badge() -> str:
    service = get_service()
    if service.device.type == "cuda":
        name = torch.cuda.get_device_name(service.device)
        return f'<span class="nano-badge" style="background:#16a34a">GPU: {name}</span>'
    return '<span class="nano-badge" style="background:#6b7280">CPU</span>'


APP_CSS = """
:root { --accent: #6d28d9; --panel: #ffffff; --line: #e5e7eb; --muted: #4d5562; }
.gradio-container { background: linear-gradient(180deg, #faf5ff 0%, #f3f0f7 100%); }
.nano-card { border: 1px solid var(--line); border-radius: 16px; background: var(--panel); padding: 14px; }
.nano-title { font-size: 24px; font-weight: 700; margin-bottom: 4px; color: #000000; }
.nano-subtitle { color: var(--muted); font-size: 14px; margin-bottom: 8px; }
.nano-badge { display: inline-block; background: #6d28d9; color: white; font-size: 11px;
              padding: 2px 8px; border-radius: 8px; margin-left: 8px; vertical-align: middle; }
#run-btn { background: var(--accent); border: none; font-size: 16px; }
.example-btn { text-align: left !important; white-space: normal !important; line-height: 1.4; padding: 8px 12px !important; height: auto !important; }
"""


def build_demo():
    device_badge = _device_badge()

    with gr.Blocks(title="MOSS-TTS-Nano Multilingual Demo") as demo:
        gr.Markdown(
            f"""
            <div class="nano-card">
              <div class="nano-title">MOSS-TTS-Nano Multilingual Demo
                <span class="nano-badge">~100M params</span>
                {device_badge}
              </div>
              <div class="nano-subtitle">Lightweight TTS in English, French, Spanish, German — runs on CPU or GPU, 48 kHz stereo</div>
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
                text = gr.Textbox(
                    label="Text",
                    lines=5,
                    placeholder="Enter text to synthesize...",
                    value=EXAMPLE_TEXTS["English"][0],
                )

                with gr.Accordion("Voice", open=True):
                    voice = gr.Dropdown(
                        choices=get_voices_for_language("English"),
                        value=LANGUAGE_DEFAULT_VOICE["English"],
                        label="Preset Voice",
                        info="Native voices are listed first; select any voice or upload your own below",
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
                    gr.Markdown(
                        "<small>Tip: record 3-10s of clear speech, avoid background noise. "
                        "Audio is auto-converted to 48kHz stereo.</small>"
                    )

            with gr.Column(scale=2):
                with gr.Accordion("Sampling Parameters", open=False):
                    audio_temperature = gr.Slider(0.1, 3.0, step=0.05, value=0.5, label="Audio Temperature",
                        info="Lower = more stable/clear, higher = more expressive/risky")
                    audio_top_p = gr.Slider(0.1, 1.0, step=0.01, value=0.8, label="Audio Top-p")
                    audio_top_k = gr.Slider(1, 200, step=1, value=15, label="Audio Top-k")
                    audio_repetition_penalty = gr.Slider(0.8, 2.0, step=0.05, value=1.35, label="Audio Repetition Penalty",
                        info="Higher = less repetition/distortion")
                    max_new_frames = gr.Slider(50, 750, step=25, value=375, label="Max New Frames")

                run_btn = gr.Button("Generate Speech", variant="primary", elem_id="run-btn")
                output_audio = gr.Audio(label="Output Audio", type="numpy")
                status = gr.Textbox(label="Status", lines=4, interactive=False)

        def on_language_change(lang):
            texts = EXAMPLE_TEXTS.get(lang, [])
            first_text = texts[0] if texts else ""
            voices = get_voices_for_language(lang)
            default_voice = LANGUAGE_DEFAULT_VOICE.get(lang, voices[0])
            btn_updates = []
            for i in range(3):
                if i < len(texts):
                    btn_updates.append(gr.update(value=texts[i], visible=True))
                else:
                    btn_updates.append(gr.update(value="", visible=False))
            return [gr.update(value=first_text), gr.update(choices=voices, value=default_voice)] + btn_updates

        language.change(
            fn=on_language_change,
            inputs=[language],
            outputs=[text, voice] + example_btns,
        )

        for btn in example_btns:
            btn.click(fn=lambda t: t, inputs=[btn], outputs=[text])

        run_btn.click(
            fn=run_inference,
            inputs=[
                text, language, voice, reference_audio,
                audio_temperature, audio_top_p, audio_top_k,
                audio_repetition_penalty, max_new_frames,
            ],
            outputs=[output_audio, status],
        )

    return demo


def main():
    global _service, _text_normalizer

    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano Multilingual Demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path (default: HuggingFace)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda/cuda:0")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # Init service (skip if already loaded by module-level init)
    if _service is None:
        os.environ["MOSS_DEVICE"] = args.device
        os.environ["MOSS_DTYPE"] = args.dtype
        _init_service_once()

    demo = build_demo()

    launch_kwargs = dict(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )

    demo.queue(max_size=16, default_concurrency_limit=1).launch(css=APP_CSS, **launch_kwargs)


def _init_service_once():
    """Load model + text normalizer (idempotent — safe to call on every hot reload)."""
    global _service, _text_normalizer
    if _service is not None:
        return

    try:
        _text_normalizer = WeTextProcessingManager()
        snapshot = _text_normalizer.ensure_ready()
        if not snapshot.ready:
            _text_normalizer = None
    except Exception:
        _text_normalizer = None

    device = os.environ.get("MOSS_DEVICE", "auto")
    dtype = os.environ.get("MOSS_DTYPE", "auto")
    print(f"[Startup] Loading MOSS-TTS-Nano (device={device}, dtype={dtype})...", flush=True)
    started = time.monotonic()
    _service = NanoTTSService(device=device, dtype=dtype)
    _service.warmup()
    print(f"[Startup] Model loaded in {time.monotonic() - started:.2f}s", flush=True)


# Always create module-level demo for Gradio hot-reload discovery.
# Model loads once and persists across UI reloads.
_init_service_once()
demo = build_demo()

if __name__ == "__main__":
    main()
