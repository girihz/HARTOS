"""
Chatterbox TTS tool — GPU-accelerated emotional speech synthesis.

Two variants, each in its own subprocess worker:
  - Turbo: English, 3.8GB VRAM, [laugh]/[chuckle] tags
  - ML:    23 languages, 12GB VRAM, voice cloning

SUBPROCESS ISOLATED: this module is BOTH the parent-side tool AND the
worker subprocess entry point. Variant is selected via CLI arg:
  python -m integrations.service_tools.chatterbox_tool turbo
  python -m integrations.service_tools.chatterbox_tool ml

CUDA OOM (especially likely with the 12GB ML model on consumer GPUs)
only kills the worker. Parent receives `transient: true` and falls back.

Public API (parent side):
  chatterbox_synthesize(text, language, voice, output_path) → JSON  [Turbo]
  chatterbox_ml_synthesize(text, language, voice, output_path) → JSON  [ML]
  unload_chatterbox() → None
"""

from typing import Optional

import os
import sys

from integrations.service_tools.gpu_worker import ToolWorker

# Default reference voice for Chatterbox voice cloning — same path Nunba
# used historically so existing ref audio keeps working.
_DEFAULT_REF_VOICE = os.path.join(os.path.expanduser('~'), 'Downloads', 'Lily.mp3')

# Silence pad appended after every generation to prevent chopped endings.
_END_PAD_SECONDS = 0.3


def _resolve_ref_voice(req: dict) -> str:
    ref = req.get('voice')
    if not ref and os.path.isfile(_DEFAULT_REF_VOICE):
        ref = _DEFAULT_REF_VOICE
    return ref or ''


def _save_wav_with_padding(wav, sample_rate: int, output_path: str) -> float:
    """Save a tensor as WAV with 0.3s end silence pad. Returns duration."""
    import torch
    import torchaudio

    # Normalize shape to (channels, samples)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav = wav.cpu()

    # Pad trailing silence to prevent chopped endings
    pad = torch.zeros(
        wav.shape[0], int(sample_rate * _END_PAD_SECONDS),
        dtype=wav.dtype, device=wav.device,
    )
    wav_padded = torch.cat([wav, pad], dim=-1)

    torchaudio.save(output_path, wav_padded, sample_rate)
    return wav_padded.shape[-1] / sample_rate


# ── Chatterbox Turbo (English, 3.8GB VRAM) ──────────────────────

def _load_turbo():
    """Load Chatterbox Turbo (English, voice cloning) on CUDA.

    Ported from Nunba's _LazyChatterboxTurbo:
      - Uses ChatterboxTurboTTS (not the base ChatterboxTTS)
      - Applies Windows safetensors CPU-first workaround to avoid
        segfaults on sequential CUDA loads (known safetensors bug
        on Windows).
    """
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if sys.platform == 'win32':
        # safetensors segfaults on sequential CUDA loads on Windows.
        # Patch load_file to always load to CPU first, then let
        # .to(device) do the CUDA transfer.
        import safetensors.torch as _st
        _orig_load = _st.load_file

        def _cpu_first_load(path, device=None):
            return _orig_load(path, device='cpu')

        _st.load_file = _cpu_first_load
        try:
            return ChatterboxTurboTTS.from_pretrained(device='cuda')
        finally:
            _st.load_file = _orig_load

    return ChatterboxTurboTTS.from_pretrained(device='cuda')


def _synthesize_turbo(model, req: dict) -> dict:
    text = req.get('text', '')
    if not text or not text.strip():
        return {'error': 'Text is required'}
    output_path = req.get('output_path')
    if not output_path:
        return {'error': 'output_path is required'}

    ref = _resolve_ref_voice(req)
    wav = model.generate(text, audio_prompt_path=ref)
    duration = _save_wav_with_padding(wav, model.sr, output_path)

    return {
        'path': output_path,
        'duration': round(duration, 2),
        'sample_rate': model.sr,
        'engine': 'chatterbox-turbo',
        'device': 'cuda',
        'voice': ref or 'default',
    }


# ── Chatterbox Multilingual (23 languages, 12GB VRAM) ───────────

def _load_ml():
    """Load Chatterbox Multilingual (23 languages, 12GB VRAM) on CUDA.

    Ported from Nunba's _LazyChatterboxMultilingual:
      - Uses ChatterboxMultilingualTTS (not base ChatterboxTTS)
    """
    from chatterbox.tts import ChatterboxMultilingualTTS
    return ChatterboxMultilingualTTS.from_pretrained(device='cuda')


def _synthesize_ml(model, req: dict) -> dict:
    text = req.get('text', '')
    if not text or not text.strip():
        return {'error': 'Text is required'}
    output_path = req.get('output_path')
    if not output_path:
        return {'error': 'output_path is required'}

    language = req.get('language', 'en')
    ref = _resolve_ref_voice(req)
    # ChatterboxMultilingualTTS uses language_id=, not lang=
    wav = model.generate(text, audio_prompt_path=ref, language_id=language)
    duration = _save_wav_with_padding(wav, model.sr, output_path)

    return {
        'path': output_path,
        'duration': round(duration, 2),
        'sample_rate': model.sr,
        'engine': 'chatterbox-ml',
        'device': 'cuda',
        'language': language,
        'voice': ref or 'default',
    }


# ── Parent-side: one ToolWorker per variant ──────────────────────

_turbo = ToolWorker(
    tool_name='chatterbox_turbo',
    tool_module='integrations.service_tools.chatterbox_tool',
    variant='turbo',
    vram_budget='tts_chatterbox_turbo',
    output_subdir='chatterbox/output',
    engine='chatterbox-turbo',
    startup_timeout=120.0,
    request_timeout=120.0,
)

_ml = ToolWorker(
    tool_name='chatterbox_ml',
    tool_module='integrations.service_tools.chatterbox_tool',
    variant='ml',
    vram_budget='tts_chatterbox_ml',
    output_subdir='chatterbox/output',
    engine='chatterbox-ml',
    startup_timeout=240.0,   # 12GB model takes a while
    request_timeout=180.0,
)


def chatterbox_synthesize(
    text: str,
    language: str = 'en',
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Synthesize with Chatterbox Turbo (English, GPU subprocess)."""
    return _turbo.synthesize(
        text=text, language='en', voice=voice, output_path=output_path,
    )


def chatterbox_ml_synthesize(
    text: str,
    language: str = 'en',
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Synthesize with Chatterbox ML (23 languages, GPU subprocess)."""
    return _ml.synthesize(
        text=text, language=language, voice=voice, output_path=output_path,
    )


def unload_chatterbox():
    """Stop both Chatterbox worker subprocesses and free VRAM."""
    _turbo.stop()
    _ml.stop()


class ChatterboxTool:
    """Register Chatterbox as an in-process service tool."""

    @classmethod
    def register_functions(cls):
        from .registry import ServiceToolInfo, service_tool_registry
        tool_info = ServiceToolInfo(
            name="chatterbox",
            description=(
                "GPU-accelerated emotional TTS. Turbo: English + [laugh]/[chuckle] tags, "
                "3.8GB VRAM. ML: 23 languages, 12GB VRAM. Voice cloning. "
                "Requires: pip install chatterbox"
            ),
            base_url="inprocess://chatterbox",
            endpoints={
                "synthesize": {
                    "path": "/synthesize",
                    "method": "POST",
                    "description": "Synthesize with Chatterbox Turbo (English, GPU).",
                    "params_schema": {
                        "text": {"type": "string"},
                        "voice": {"type": "string", "description": "Reference audio path"},
                    },
                },
                "synthesize_ml": {
                    "path": "/synthesize_ml",
                    "method": "POST",
                    "description": "Synthesize with Chatterbox ML (23 languages, GPU).",
                    "params_schema": {
                        "text": {"type": "string"},
                        "language": {"type": "string"},
                        "voice": {"type": "string"},
                    },
                },
            },
            tags=["tts", "speech", "voice-cloning", "gpu", "chatterbox"],
            timeout=60,
        )
        tool_info.is_healthy = True
        service_tool_registry._tools["chatterbox"] = tool_info
        return True


# NOTE: no `if __name__ == '__main__':` block here. The centralized
# dispatcher in gpu_worker picks up `_load_turbo`/`_synthesize_turbo`
# when spawned with variant='turbo', and `_load_ml`/`_synthesize_ml`
# when variant='ml'.
