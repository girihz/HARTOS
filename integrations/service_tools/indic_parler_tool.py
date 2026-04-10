"""
Indic Parler TTS tool — 22 Indian languages + English (GPU).

Supports: hi, ta, te, bn, gu, kn, ml, mr, or, pa, ur, as, bho, doi,
          kok, mai, mni, ne, sa, sat, sd, en
VRAM: 1.8GB model size, 2GB recommended.
Requires: pip install indic-parler-tts

SUBPROCESS ISOLATED: this module is BOTH the parent-side tool AND the
worker subprocess entry point. Model + tokenizer live in the worker;
the parent just forwards requests.

Public API (parent):
  indic_parler_synthesize(text, language, voice, output_path) → JSON
  unload_indic_parler() → None

Worker entry:
  python -m integrations.service_tools.indic_parler_tool
"""

import re
from typing import Optional

from integrations.service_tools.gpu_worker import ToolWorker

# Fallback sample rate used by the parent-side default_sample_rate kwarg
# when the worker response doesn't carry one. The real value comes from
# model.config.sampling_rate at runtime (44100 for Indic Parler TTS).
SAMPLE_RATE = 44100

# Recommended voices per language (from Nunba's _LazyIndicParler.SPEAKERS).
# Indic Parler's output character depends heavily on which named speaker
# appears in the description — wrong language-speaker pairing gives poor
# pronunciation.
_SPEAKERS = {
    'ta': 'Jaya',  'hi': 'Divya',   'bn': 'Aditi',   'te': 'Lalitha',
    'kn': 'Anu',   'ml': 'Anjali',  'gu': 'Neha',    'mr': 'Sunita',
    'as': 'Sita',  'ur': 'Divya',   'ne': 'Amrita',  'or': 'Debjani',
    'sa': 'Aryan', 'mai': 'Aditi',  'mni': 'Laishram','sd': 'Divya',
    'kok': 'Sunita','brx': 'Maya',  'doi': 'Karan',  'sat': 'Maya',
    'pa': 'Divya', 'en': 'Divya',
}

# Tuning constants — match Nunba's _LazyIndicParler exactly so output is
# acoustically identical after the port.
_INTER_SENTENCE_GAP_S = 0.15
_END_PAD_S = 0.5
_PEAK_TARGET_DB = -1.0
_SPLIT_THRESHOLD_CHARS = 80
_MIN_CHUNK_CHARS = 20      # merge any sub-20-char fragment into neighbor
_TAIL_MERGE_CHARS = 15     # merge any ≤15-char trailing fragment backwards
_MAX_NEW_TOKENS_MIN = 3000
_MAX_NEW_TOKENS_MAX = 8000
_MAX_NEW_TOKENS_PER_CHAR = 50


def _build_description(language: str) -> str:
    """Build a style description with the recommended speaker for language."""
    speaker = _SPEAKERS.get(language, 'Divya')
    return (
        f"{speaker} speaks with a confident, clear and expressive voice "
        f"at a moderate pace. The recording is of very high quality with no "
        f"background noise, the speaker's voice is loud, clear and very "
        f"close to the microphone."
    )


def _split_sentences(text: str) -> list:
    """Split text at real sentence boundaries (not mid-ellipsis).

    Handles Latin + Indic punctuation (. ? ! । ৷). Protects "..." so
    ellipses don't trigger splits. Merges fragments shorter than
    _MIN_CHUNK_CHARS into their neighbor, and pulls any ≤
    _TAIL_MERGE_CHARS tail back into the previous chunk.
    """
    protected = text.replace('...', '\x00ELLIPSIS\x00')
    parts = re.split(r'(?<=[^\.\s])[.?!।৷]\s+', protected)
    parts = [p.replace('\x00ELLIPSIS\x00', '...') for p in parts]
    merged = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if merged and len(merged[-1]) < _MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + ' ' + p
        else:
            merged.append(p)
    if len(merged) > 1 and len(merged[-1]) < _TAIL_MERGE_CHARS:
        merged[-2] = merged[-2] + ' ' + merged[-1]
        merged.pop()
    return merged if len(merged) > 1 else [text]


# ── Worker callbacks (run in subprocess) ──────────────────────────

def _load():
    """Load Indic Parler TTS + both tokenizers.

    Ported from Nunba's _LazyIndicParler:
      - Loads ParlerTTSForConditionalGeneration on CUDA
      - Uses TWO tokenizers: one for the prompt text, one for the
        description. The description tokenizer comes from the model's
        own text encoder (different vocab from the prompt tokenizer).
    """
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    model = ParlerTTSForConditionalGeneration.from_pretrained(
        'ai4bharat/indic-parler-tts',
    ).to('cuda')
    # Prompt tokenizer: lowercases Indic text, matches the model's decoder.
    prompt_tokenizer = AutoTokenizer.from_pretrained(
        'ai4bharat/indic-parler-tts',
    )
    # Description tokenizer: matches the model's text encoder (different
    # vocab — English-centric since descriptions are always English).
    desc_tokenizer = AutoTokenizer.from_pretrained(
        model.config.text_encoder._name_or_path,
    )
    return {
        'model': model,
        'prompt_tokenizer': prompt_tokenizer,
        'desc_tokenizer': desc_tokenizer,
        'sample_rate': model.config.sampling_rate,
    }


def _generate_chunk(state: dict, text: str, language: str):
    """Generate audio for one text chunk; returns a 1-D numpy float32 array."""
    import torch

    model = state['model']
    prompt_tokenizer = state['prompt_tokenizer']
    desc_tokenizer = state['desc_tokenizer']

    description = _build_description(language)
    desc_inputs = desc_tokenizer(description, return_tensors='pt').to('cuda')
    prompt_inputs = prompt_tokenizer(text, return_tensors='pt').to('cuda')
    max_tokens = max(
        _MAX_NEW_TOKENS_MIN,
        min(_MAX_NEW_TOKENS_MAX, len(text) * _MAX_NEW_TOKENS_PER_CHAR),
    )

    with torch.no_grad():
        generation = model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
            max_new_tokens=max_tokens,
        )
    return generation.cpu().float().numpy().squeeze()


def _synthesize(state, req: dict) -> dict:
    text = req.get('text', '')
    if not text or not text.strip():
        return {'error': 'Text is required'}
    output_path = req.get('output_path')
    if not output_path:
        return {'error': 'output_path is required'}

    import numpy as np
    import soundfile as sf

    language = req.get('language', 'hi')
    sr = state['sample_rate']

    # Split long text to prevent Indic Parler's tendency to clip long
    # utterances' tails. Threshold 80 chars matches Nunba's tuning.
    if len(text) > _SPLIT_THRESHOLD_CHARS:
        sentences = _split_sentences(text)
    else:
        sentences = [text]

    if len(sentences) == 1:
        audio = _generate_chunk(state, text, language)
    else:
        gap = np.zeros(int(sr * _INTER_SENTENCE_GAP_S), dtype=np.float32)
        chunks = []
        for i, sent in enumerate(sentences):
            chunk_audio = _generate_chunk(state, sent, language)
            if chunk_audio is not None and len(chunk_audio) > 0:
                chunks.append(chunk_audio)
                if i < len(sentences) - 1:
                    chunks.append(gap)
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)

    # Pad trailing silence to prevent chopped endings
    end_pad = np.zeros(int(sr * _END_PAD_S), dtype=np.float32)
    audio = np.concatenate([audio, end_pad])

    # Peak-normalize to the target dBFS
    peak = float(np.abs(audio).max())
    if peak > 0:
        target_peak = 10 ** (_PEAK_TARGET_DB / 20.0)
        audio = audio * (target_peak / peak)

    sf.write(output_path, audio, sr)

    return {
        'path': output_path,
        'duration': round(len(audio) / sr, 2),
        'sample_rate': sr,
        'engine': 'indic-parler-tts',
        'device': 'cuda',
        'language': language,
        'voice': f"{_SPEAKERS.get(language, 'Divya')} ({language})",
    }


# ── Parent-side: ToolWorker instance ─────────────────────────────

_tool = ToolWorker(
    tool_name='indic_parler',
    tool_module='integrations.service_tools.indic_parler_tool',
    vram_budget='tts_indic_parler',
    output_subdir='indic_parler/output',
    engine='indic-parler-tts',
    startup_timeout=120.0,
    request_timeout=120.0,
)


def indic_parler_synthesize(
    text: str,
    language: str = 'hi',
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Synthesize with Indic Parler TTS (22 Indic languages, GPU subprocess).

    `voice` here is a style description (e.g. "A female speaker with
    calm tone"), not a reference audio path. Indic Parler uses
    text-conditioned styles, not voice cloning.
    """
    return _tool.synthesize(
        text=text,
        language=language,
        voice=voice,
        output_path=output_path,
        default_sample_rate=SAMPLE_RATE,
    )


def unload_indic_parler():
    """Stop the Indic Parler worker subprocess and free its VRAM."""
    _tool.stop()


class IndicParlerTool:
    """Register Indic Parler as an in-process service tool."""

    @classmethod
    def register_functions(cls):
        from .registry import ServiceToolInfo, service_tool_registry
        tool_info = ServiceToolInfo(
            name="indic_parler",
            description=(
                "Indic Parler TTS: 22 Indian languages + English. "
                "Style-conditioned synthesis (no voice cloning). "
                "1.8GB VRAM. Requires: pip install indic-parler-tts"
            ),
            base_url="inprocess://indic_parler",
            endpoints={
                "synthesize": {
                    "path": "/synthesize",
                    "method": "POST",
                    "description": "Synthesize with Indic Parler TTS (22 Indic languages, GPU).",
                    "params_schema": {
                        "text": {"type": "string"},
                        "language": {"type": "string"},
                        "voice": {"type": "string", "description": "Style description text"},
                    },
                },
            },
            tags=["tts", "speech", "gpu", "indic", "multilingual"],
            timeout=60,
        )
        tool_info.is_healthy = True
        service_tool_registry._tools["indic_parler"] = tool_info
        return True

# NOTE: no `if __name__ == '__main__':` block — the centralized
# dispatcher (gpu_worker) imports this module and calls _load/_synthesize.
