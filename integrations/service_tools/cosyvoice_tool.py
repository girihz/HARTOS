"""
CosyVoice 3 TTS tool — multilingual zero-shot voice cloning (GPU).

Supports: zh, ja, ko, de, es, fr, it, ru, en (9 languages).
VRAM: 3.5GB model size, 4GB recommended.
Requires: pip install cosyvoice

SUBPROCESS ISOLATED: this module is BOTH the parent-side tool AND the
worker subprocess entry point. CUDA OOM or DLL crashes stay contained
in the worker; the parent falls back gracefully.

Public API (parent):
  cosyvoice_synthesize(text, language, voice, output_path) → JSON
  unload_cosyvoice() → None

Worker entry:
  python -m integrations.service_tools.cosyvoice_tool
"""

import os
from typing import Optional

from integrations.service_tools.gpu_worker import ToolWorker

# Fallback sample rate used by the parent-side default_sample_rate kwarg
# when the worker response doesn't carry one. The actual sample rate is
# reported by model.sample_rate at runtime (22050 for CosyVoice3-0.5B).
SAMPLE_RATE = 22050

# Default reference voice for CosyVoice zero-shot cloning — same path
# Nunba used historically.
_DEFAULT_REF_VOICE = os.path.join(os.path.expanduser('~'), 'Downloads', 'Lily.mp3')

# CosyVoice 3 lives in a dev clone (not pip). The clone includes a
# Matcha-TTS dependency that also needs to be on sys.path.
_COSYVOICE_CLONE = os.path.join(
    os.path.expanduser('~'), 'PycharmProjects', 'CosyVoice',
)
_COSYVOICE_MODEL_DIR = os.path.join(
    _COSYVOICE_CLONE, 'pretrained_models', 'CosyVoice3-0.5B',
)
_COSYVOICE_HF_REPO = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'

# CosyVoice 3 requires every prompt to be prefixed with this token.
_COSYVOICE_PROMPT_PREFIX = 'You are a helpful assistant.<|endofprompt|>'

# Trailing silence pad to prevent chopped endings.
_END_PAD_SECONDS = 0.3


# ── Worker callbacks (run in subprocess) ──────────────────────────

def _load():
    """Load CosyVoice 3 0.5B from the dev clone.

    Ported from Nunba's _LazyCosyVoice3:
      - Requires ~/PycharmProjects/CosyVoice clone + its Matcha-TTS deps
      - Uses cosyvoice.cli.cosyvoice.AutoModel (not the pip CosyVoice class)
      - Auto-downloads CosyVoice3-0.5B from HuggingFace if missing
    """
    import sys

    if not os.path.isdir(_COSYVOICE_CLONE):
        raise FileNotFoundError(
            f"CosyVoice 3 not found at {_COSYVOICE_CLONE} — clone the "
            f"CosyVoice repo to that path."
        )

    # Prepend the clone + its Matcha-TTS bundled dependency to sys.path
    if _COSYVOICE_CLONE not in sys.path:
        sys.path.insert(0, _COSYVOICE_CLONE)
    matcha = os.path.join(_COSYVOICE_CLONE, 'third_party', 'Matcha-TTS')
    if os.path.isdir(matcha) and matcha not in sys.path:
        sys.path.insert(0, matcha)

    from cosyvoice.cli.cosyvoice import AutoModel

    # Auto-download CosyVoice3 model weights if missing
    if not os.path.isdir(_COSYVOICE_MODEL_DIR):
        from huggingface_hub import snapshot_download
        snapshot_download(_COSYVOICE_HF_REPO, local_dir=_COSYVOICE_MODEL_DIR)

    return AutoModel(model_dir=_COSYVOICE_MODEL_DIR)


def _synthesize(model, req: dict) -> dict:
    text = req.get('text', '')
    if not text or not text.strip():
        return {'error': 'Text is required'}

    output_path = req.get('output_path')
    if not output_path:
        return {'error': 'output_path is required'}

    # CosyVoice 3 requires an explicit assistant prefix token
    cv3_text = f'{_COSYVOICE_PROMPT_PREFIX}{text}'

    # Resolve reference voice for zero-shot cloning; fall back to
    # inference_sft with the first available built-in speaker.
    ref = req.get('voice')
    if not ref and os.path.isfile(_DEFAULT_REF_VOICE):
        ref = _DEFAULT_REF_VOICE

    audio = None
    if ref and os.path.isfile(ref):
        for chunk in model.inference_cross_lingual(cv3_text, ref, stream=False):
            audio = chunk['tts_speech']
            break
    else:
        spks = model.list_available_spks() if hasattr(model, 'list_available_spks') else []
        if not spks:
            return {'error': 'CosyVoice3: no reference voice and no built-in speakers'}
        spk = spks[0]
        for chunk in model.inference_sft(cv3_text, spk, stream=False):
            audio = chunk['tts_speech']
            break

    if audio is None:
        return {'error': 'CosyVoice3: synthesis returned no audio'}

    # Pad 0.3s silence to prevent chopped endings
    import torch
    sr = model.sample_rate
    pad = torch.zeros(
        audio.shape[0] if audio.ndim > 1 else 1,
        int(sr * _END_PAD_SECONDS),
        dtype=audio.dtype, device=audio.device,
    )
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    audio = torch.cat([audio, pad], dim=-1)

    import torchaudio
    torchaudio.save(output_path, audio.cpu(), sr)

    return {
        'path': output_path,
        'duration': round(audio.shape[-1] / sr, 2),
        'sample_rate': sr,
        'engine': 'cosyvoice3',
        'device': 'cuda',
        'voice': ref or 'default',
    }


# ── Parent-side: ToolWorker instance ─────────────────────────────

_tool = ToolWorker(
    tool_name='cosyvoice3',
    tool_module='integrations.service_tools.cosyvoice_tool',
    vram_budget='tts_cosyvoice3',
    output_subdir='cosyvoice/output',
    engine='cosyvoice3',
    startup_timeout=120.0,
    request_timeout=120.0,
)


def cosyvoice_synthesize(
    text: str,
    language: str = 'zh',
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Synthesize with CosyVoice 3 (9 languages, GPU subprocess)."""
    return _tool.synthesize(
        text=text,
        language=language,
        voice=voice,
        output_path=output_path,
        default_sample_rate=SAMPLE_RATE,
    )


def unload_cosyvoice():
    """Stop the CosyVoice worker subprocess and free its VRAM."""
    _tool.stop()


class CosyVoiceTool:
    """Register CosyVoice as an in-process service tool."""

    @classmethod
    def register_functions(cls):
        from .registry import ServiceToolInfo, service_tool_registry
        tool_info = ServiceToolInfo(
            name="cosyvoice",
            description=(
                "CosyVoice 3: multilingual zero-shot TTS. "
                "9 languages (zh/ja/ko/de/es/fr/it/ru/en), 3.5GB VRAM. "
                "Requires: pip install cosyvoice"
            ),
            base_url="inprocess://cosyvoice",
            endpoints={
                "synthesize": {
                    "path": "/synthesize",
                    "method": "POST",
                    "description": "Synthesize with CosyVoice 3 (9 languages, GPU).",
                    "params_schema": {
                        "text": {"type": "string"},
                        "language": {"type": "string"},
                        "voice": {"type": "string", "description": "Reference audio path"},
                    },
                },
            },
            tags=["tts", "speech", "voice-cloning", "gpu", "cosyvoice", "multilingual"],
            timeout=60,
        )
        tool_info.is_healthy = True
        service_tool_registry._tools["cosyvoice"] = tool_info
        return True

# NOTE: no `if __name__ == '__main__':` block — the centralized
# dispatcher (gpu_worker) imports this module and calls _load/_synthesize.
