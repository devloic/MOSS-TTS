"""
MOSS-TTS Multilingual Demo — French, Spanish, English, German

A Gradio app for generating speech in 4 languages with optional voice cloning.
"""

import argparse
import functools
import importlib.util
import time

import gradio as gr
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

# Disable the broken cuDNN SDPA backend
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

MODEL_PATH = "OpenMOSS-Team/MOSS-TTS"
DEFAULT_ATTN_IMPLEMENTATION = "auto"
DEFAULT_MAX_NEW_TOKENS = 4096

LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}

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


def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> str | None:
    requested_norm = (requested or "").strip().lower()
    if requested_norm in {"none"}:
        return None
    if requested_norm not in {"", "auto"}:
        return requested
    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return "flash_attention_2"
    if device.type == "cuda":
        return "sdpa"
    return "eager"


@functools.lru_cache(maxsize=1)
def load_backend(model_path: str, device_str: str, attn_implementation: str):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    resolved_attn = resolve_attn_implementation(requested=attn_implementation, device=device, dtype=dtype)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if resolved_attn:
        model_kwargs["attn_implementation"] = resolved_attn

    model = AutoModel.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
    return model, processor, device, sample_rate


def run_inference(
    text: str,
    language: str,
    reference_audio: str | None,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    model_path: str,
    device: str,
    attn_implementation: str,
):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please enter text to synthesize.")

    lang_code = LANGUAGES.get(language)
    started_at = time.monotonic()
    model, processor, torch_device, sample_rate = load_backend(
        model_path=model_path,
        device_str=device,
        attn_implementation=attn_implementation,
    )

    user_kwargs = {"text": text}
    if lang_code:
        user_kwargs["language"] = lang_code
    if reference_audio:
        user_kwargs["reference"] = [reference_audio]

    conversations = [[processor.build_user_message(**user_kwargs)]]
    mode = "generation"

    batch = processor(conversations, mode=mode)
    input_ids = batch["input_ids"].to(torch_device)
    attention_mask = batch["attention_mask"].to(torch_device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            audio_temperature=float(temperature),
            audio_top_p=float(top_p),
            audio_top_k=int(top_k),
            audio_repetition_penalty=float(repetition_penalty),
        )

    messages = processor.decode(outputs)
    if not messages or messages[0] is None:
        raise gr.Error("The model did not return a decodable audio result.")

    audio = messages[0].audio_codes_list[0]
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().cpu().numpy()
    else:
        audio_np = np.asarray(audio, dtype=np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)
    audio_np = audio_np.astype(np.float32, copy=False)

    elapsed = time.monotonic() - started_at
    clone_status = "with voice cloning" if reference_audio else "direct generation"
    status = (
        f"Done | {language} ({lang_code}) | {clone_status} | {elapsed:.2f}s\n"
        f"temperature={temperature:.2f}, top_p={top_p:.2f}, top_k={top_k}, "
        f"repetition_penalty={repetition_penalty:.2f}, max_new_tokens={max_new_tokens}"
    )
    return (sample_rate, audio_np), status


def fill_example(language, evt: gr.SelectData):
    if evt is None or evt.index is None:
        return gr.update()
    row_idx = int(evt.index[0]) if isinstance(evt.index, (tuple, list)) else int(evt.index)
    texts = EXAMPLE_TEXTS.get(language, [])
    if 0 <= row_idx < len(texts):
        return texts[row_idx]
    return gr.update()


def get_examples_for_language(language):
    texts = EXAMPLE_TEXTS.get(language, [])
    return [[t[:80] + "..." if len(t) > 80 else t] for t in texts]


def build_demo(args: argparse.Namespace):
    css = """
    :root {
      --accent: #0f766e;
      --panel: #ffffff;
      --line: #e5e7eb;
      --muted: #4d5562;
    }
    .gradio-container {
      background: linear-gradient(180deg, #f7f8fa 0%, #f3f5f7 100%);
    }
    .lang-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      padding: 14px;
    }
    .lang-title {
      font-size: 24px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .lang-subtitle {
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 8px;
    }
    #run-btn {
      background: var(--accent);
      border: none;
      font-size: 16px;
    }
    """

    with gr.Blocks(title="MOSS-TTS Multilingual Demo", css=css) as demo:
        gr.Markdown(
            """
            <div class="lang-card">
              <div class="lang-title">MOSS-TTS Multilingual Demo</div>
              <div class="lang-subtitle">Text-to-Speech in English, French, Spanish, and German — with optional voice cloning</div>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                language = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Language",
                    info="Select the language of the text to synthesize",
                )
                text = gr.Textbox(
                    label="Text",
                    lines=6,
                    placeholder="Enter text to synthesize...",
                    value=EXAMPLE_TEXTS["English"][0],
                )
                reference_audio = gr.Audio(
                    label="Reference Audio (Optional — for voice cloning)",
                    type="filepath",
                )
                clone_hint = gr.Markdown(
                    "**Mode:** Direct generation (upload reference audio to enable voice cloning)"
                )

                with gr.Accordion("Sampling Parameters", open=False):
                    temperature = gr.Slider(0.1, 3.0, step=0.05, value=1.7, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, step=0.01, value=0.8, label="Top-p")
                    top_k = gr.Slider(1, 200, step=1, value=25, label="Top-k")
                    repetition_penalty = gr.Slider(0.8, 2.0, step=0.05, value=1.0, label="Repetition Penalty")
                    max_new_tokens = gr.Slider(256, 8192, step=128, value=DEFAULT_MAX_NEW_TOKENS, label="Max New Tokens")

                run_btn = gr.Button("Generate Speech", variant="primary", elem_id="run-btn")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Output Audio", type="numpy")
                status = gr.Textbox(label="Status", lines=3, interactive=False)

                examples_table = gr.Dataframe(
                    headers=["Example Text"],
                    value=get_examples_for_language("English"),
                    datatype=["str"],
                    interactive=False,
                    wrap=True,
                    label="Example texts (click a row to use)",
                )

        # Update examples table and default text when language changes
        def on_language_change(lang):
            examples = get_examples_for_language(lang)
            first_text = EXAMPLE_TEXTS.get(lang, [""])[0]
            return examples, first_text

        language.change(
            fn=on_language_change,
            inputs=[language],
            outputs=[examples_table, text],
        )

        # Update clone hint when reference audio changes
        reference_audio.change(
            fn=lambda ref: (
                "**Mode:** Voice cloning (using uploaded reference audio)"
                if ref
                else "**Mode:** Direct generation (upload reference audio to enable voice cloning)"
            ),
            inputs=[reference_audio],
            outputs=[clone_hint],
        )

        # Fill text from example selection
        examples_table.select(
            fn=fill_example,
            inputs=[language],
            outputs=[text],
        )

        # Run inference
        run_btn.click(
            fn=lambda text, language, reference_audio, temperature, top_p, top_k, repetition_penalty, max_new_tokens: run_inference(
                text=text,
                language=language,
                reference_audio=reference_audio,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                model_path=args.model_path,
                device=args.device,
                attn_implementation=args.attn_implementation,
            ),
            inputs=[text, language, reference_audio, temperature, top_p, top_k, repetition_penalty, max_new_tokens],
            outputs=[output_audio, status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="MOSS-TTS Multilingual Demo")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_implementation", type=str, default=DEFAULT_ATTN_IMPLEMENTATION)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    runtime_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime_dtype = torch.bfloat16 if runtime_device.type == "cuda" else torch.float32
    args.attn_implementation = resolve_attn_implementation(
        requested=args.attn_implementation,
        device=runtime_device,
        dtype=runtime_dtype,
    ) or "none"
    print(f"[INFO] attn_implementation={args.attn_implementation}", flush=True)

    # Preload model at startup
    preload_start = time.monotonic()
    print(f"[Startup] Preloading: model={args.model_path}, device={args.device}, attn={args.attn_implementation}", flush=True)
    load_backend(
        model_path=args.model_path,
        device_str=args.device,
        attn_implementation=args.attn_implementation,
    )
    print(f"[Startup] Preload done in {time.monotonic() - preload_start:.2f}s", flush=True)

    demo = build_demo(args)
    demo.queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
