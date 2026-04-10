"""
F5-TTS tool — flow-matching voice cloning (English + Chinese, GPU).

VRAM: 1.3GB model size, 2GB recommended.
Requires: pip install f5-tts

SUBPROCESS ISOLATED: this module is BOTH the parent-side tool AND the
worker subprocess entry point. When imported normally, `f5_synthesize()`
dispatches to a subprocess running this same module via `python -m`.
CUDA OOM or any C-level crash only kills the subprocess — the parent
receives `{"error": ..., "transient": true}` and can fall back to Piper.

Public API (parent side):
  f5_synthesize(text, language, voice, output_path) → JSON
  unload_f5_tts() → None

Worker entry (child side):
  python -m integrations.service_tools.f5_tts_tool
"""

from typing import Optional

from integrations.service_tools.gpu_worker import ToolWorker

# ── Worker callbacks (run in subprocess) ──────────────────────────
#
# Ported from Nunba's tts/tts_engine.py::_LazyF5 — this module is now
# the SINGLE source of truth for F5-TTS synthesis. Nunba's TTSEngine
# routes here via a subprocess adapter; no more parallel in-process
# implementation.


import os


# Default reference voice for F5 voice cloning. Same path Nunba used
# historically, so existing users' ref audio keeps working.
_DEFAULT_REF_VOICE = os.path.join(os.path.expanduser('~'), 'Downloads', 'Lily.mp3')


def _load():
    """Load F5-TTS model once at subprocess startup (~40s).

    Uses F5TTS_v1_Base on CUDA — this worker exists because the parent
    decided it wants GPU F5. If CUDA isn't available the load will fail
    and the parent (Nunba TTSEngine / HARTOS tool caller) falls back to
    Piper.
    """
    from f5_tts.api import F5TTS
    return F5TTS(model='F5TTS_v1_Base', device='cuda')


def _synthesize(model, req: dict) -> dict:
    """Run one synthesis request inside the worker.

    Writes directly to output_path via F5's file_wave= arg (avoids a
    second soundfile.write pass).
    """
    text = req.get('text', '')
    if not text or not text.strip():
        return {'error': 'Text is required'}

    output_path = req.get('output_path')
    if not output_path:
        return {'error': 'output_path is required'}

    # Resolve reference voice: request override → default Lily.mp3 → empty
    # string (F5 auto-picks a voice).
    ref_voice = req.get('voice')
    if not ref_voice and os.path.isfile(_DEFAULT_REF_VOICE):
        ref_voice = _DEFAULT_REF_VOICE
    ref_voice = ref_voice or ''

    # Speed is forwarded from the adapter so synthesize_text(..., speed=0.8)
    # reaches F5's infer() — preserves behavior of the old _LazyF5 class.
    speed = float(req.get('speed') or 1.0)

    wav, sr, _ = model.infer(
        ref_file=ref_voice,
        ref_text='',  # empty = auto-transcribe, cached by F5
        gen_text=text,
        file_wave=output_path,  # writes WAV directly
        speed=speed,
    )

    return {
        'path': output_path,
        'duration': round(len(wav) / sr, 2),
        'sample_rate': sr,
        'engine': 'f5-tts',
        'device': 'cuda',
        'voice': ref_voice or 'default',
    }


# ── Parent-side: one ToolWorker instance ─────────────────────────

_tool = ToolWorker(
    tool_name='f5_tts',
    tool_module='integrations.service_tools.f5_tts_tool',
    vram_budget='tts_f5',
    output_subdir='f5_tts/output',
    engine='f5-tts',
    startup_timeout=90.0,
    request_timeout=120.0,
)


def f5_synthesize(
    text: str,
    language: str = 'en',
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
    speed: float = 1.0,
) -> str:
    """Synthesize speech using F5-TTS (GPU subprocess).

    Args:
        speed: Synthesis speed multiplier passed through to F5's
               infer() call. 1.0 = normal, >1 = faster, <1 = slower.
               Preserved from the legacy _LazyF5 behavior.

    Returns JSON. On subprocess crash the response contains
    `{"error": ..., "transient": true}` so the caller can fall back.
    """
    return _tool.synthesize(
        text=text,
        language=language,
        voice=voice,
        output_path=output_path,
        extra_request={'speed': speed} if speed != 1.0 else None,
    )


def unload_f5_tts():
    """Stop the F5 worker subprocess and free its VRAM."""
    _tool.stop()


class F5TTSTool:
    """Register F5-TTS as an in-process service tool."""

    @classmethod
    def register_functions(cls):
        from .registry import ServiceToolInfo, service_tool_registry
        tool_info = ServiceToolInfo(
            name="f5_tts",
            description=(
                "F5-TTS: flow-matching voice cloning. "
                "English + Chinese, 1.3GB VRAM. "
                "Requires: pip install f5-tts"
            ),
            base_url="inprocess://f5_tts",
            endpoints={
                "synthesize": {
                    "path": "/synthesize",
                    "method": "POST",
                    "description": "Synthesize with F5-TTS (English + Chinese, GPU).",
                    "params_schema": {
                        "text": {"type": "string"},
                        "language": {"type": "string"},
                        "voice": {"type": "string", "description": "Reference audio path"},
                    },
                },
            },
            tags=["tts", "speech", "voice-cloning", "gpu", "f5"],
            timeout=60,
        )
        tool_info.is_healthy = True
        service_tool_registry._tools["f5_tts"] = tool_info
        return True

# NOTE: no `if __name__ == '__main__':` block here. The centralized
# dispatcher at integrations.service_tools.gpu_worker imports this
# module and calls `_load` / `_synthesize` directly when spawned.
