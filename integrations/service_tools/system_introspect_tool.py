"""System self-introspection tool for agent self-awareness.

Exposes the Nunba/HARTOS runtime state (GPU tier, active models, TTS
backend, WAMP router state, draft-gate decision rationale) so the LLM
can answer "what model is running?", "why is chat slow?", "do I have
a GPU?", "is speculation on?" using real live data instead of
hallucinating.

Every function in this module:
  1. Calls the LOCAL Nunba HTTP API (http://127.0.0.1:5000) — the
     authoritative source of runtime state.  No direct module imports
     from Nunba internals, so this works in both in-process (agent
     running inside Flask) and cross-process (agent running in its
     own Python) deployments.
  2. Returns a dict with both structured fields AND a natural-language
     summary in `summary` — so the LLM can quote the summary verbatim
     for conversational replies, or read structured fields for
     follow-up logic.
  3. Has a short timeout (3s) — if the local Flask is down, returns
     `{available: False, reason: ...}` instead of hanging the agent.

Dual registration:
  - LangChain: `get_langchain_tools()` wraps each function as a
    `langchain.tools.Tool` — importable from Nunba's langchain path.
  - AutoGen:   `register_autogen(agent, user_proxy)` calls
    `autogen.register_function` for each.

Extension pattern:
  - To add a new introspection endpoint, write a small Python function
    here, append it to `_TOOL_FUNCTIONS`.  Both loaders pick it up
    automatically — no two-sided wiring.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_NUNBA_BASE = os.environ.get(
    'NUNBA_BASE_URL', 'http://127.0.0.1:5000',
).rstrip('/')
_TIMEOUT = 3.0


# ═══════════════════════════════════════════════════════════════════
# Low-level HTTP helper
# ═══════════════════════════════════════════════════════════════════

def _get(path: str) -> Dict[str, Any]:
    """GET a Nunba admin endpoint.  Returns `{available: False, ...}` on
    timeout / connection error so callers never hang + never raise into
    the LLM's reasoning loop."""
    url = f"{_NUNBA_BASE}{path}"
    try:
        r = requests.get(url, timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        return {
            'available': False,
            'status_code': r.status_code,
            'reason': f"GET {path} returned {r.status_code}",
        }
    except requests.exceptions.ConnectionError:
        return {'available': False, 'reason': f"Nunba Flask not reachable at {_NUNBA_BASE}"}
    except requests.exceptions.Timeout:
        return {'available': False, 'reason': f"Nunba {path} timed out after {_TIMEOUT}s"}
    except Exception as e:
        return {'available': False, 'reason': f"GET {path} failed: {e!s}"}


def _summarize_gpu(h: Dict[str, Any]) -> str:
    """Translate `/backend/health` payload into a one-line English."""
    if h.get('available') is False:
        return "GPU status unavailable (backend not reachable)."
    tier = h.get('gpu_tier', 'unknown')
    name = h.get('gpu_name', 'unknown GPU')
    total = h.get('vram_total_gb', 0) or 0
    free = h.get('vram_free_gb', 0) or 0
    spec = 'on' if h.get('speculation_enabled') else 'off'
    tier_human = {
        'ultra':    'Ultra (≥24GB) — every backend fits concurrently',
        'full':     'Full (≥10GB) — speculative decoding unlocked for all languages',
        'standard': 'Standard (4-10GB) — heavy model only, except cohort fast-path',
        'none':     'No GPU (<4GB or CUDA unavailable) — CPU-only',
    }.get(tier, tier)
    if total > 0:
        return (
            f"{tier_human}.  Device: {name}, {total:.1f}GB VRAM, "
            f"{free:.1f}GB free.  Speculative decoding: {spec}."
        )
    return f"{tier_human}.  Device: {name}.  Speculative decoding: {spec}."


# ═══════════════════════════════════════════════════════════════════
# Public tool functions (registered to langchain + autogen)
# ═══════════════════════════════════════════════════════════════════

def get_gpu_tier() -> Dict[str, Any]:
    """Report the GPU tier + VRAM + whether speculative decoding is active.

    Use this when the user asks: "do I have a GPU?", "why is chat
    slow?", "is speculative decoding on?", "how much VRAM do I have?",
    "what tier am I?".
    """
    h = _get('/backend/health')
    return {
        'gpu_tier': h.get('gpu_tier'),
        'gpu_name': h.get('gpu_name'),
        'vram_total_gb': h.get('vram_total_gb'),
        'vram_free_gb': h.get('vram_free_gb'),
        'cuda_available': h.get('cuda_available'),
        'speculation_enabled': h.get('speculation_enabled'),
        'available': h.get('available', True),
        'summary': _summarize_gpu(h),
    }


def list_running_models() -> Dict[str, Any]:
    """List all models the system knows about + their load state.

    Use this when the user asks: "what models are installed?", "what
    language model is running?", "what TTS voice am I using?", "do I
    have vision?".
    """
    h = _get('/api/admin/models')
    if h.get('available') is False:
        return {'available': False, 'summary': h.get('reason', 'models API unreachable'), 'models': []}
    models = h.get('models') or h.get('data') or []
    loaded = [m for m in models if m.get('loaded') or m.get('status') == 'loaded']
    lines: List[str] = []
    lines.append(f"{len(models)} model(s) registered, {len(loaded)} currently loaded.")
    for m in loaded[:15]:
        name = m.get('name') or m.get('id') or '?'
        mtype = m.get('model_type') or m.get('type') or '?'
        lang = ','.join(m.get('lang_priority') or []) or 'any'
        lines.append(f"  - {name} ({mtype}, lang={lang})")
    return {
        'available': True,
        'total_count': len(models),
        'loaded_count': len(loaded),
        'loaded': loaded,
        'all': models,
        'summary': '\n'.join(lines),
    }


def get_tts_status() -> Dict[str, Any]:
    """Report the active TTS backend + language ladder + fallback chain.

    Use this when the user asks: "what voice am I using?", "why does
    Tamil sound weird?", "why is the voice robotic?", "which TTS
    engines are loaded?".
    """
    h = _get('/api/admin/tts/status')
    if h.get('available') is False:
        # fallback: derive from /backend/health if tts/status endpoint
        # isn't live yet
        gh = _get('/backend/health')
        return {
            'available': False,
            'summary': (
                'TTS status endpoint unavailable.  '
                + (f"GPU state: {_summarize_gpu(gh)}" if gh.get('available') is not False else '')
            ),
        }
    active = h.get('active_backend') or '?'
    lang = h.get('language') or h.get('preferred_lang') or 'en'
    ladder = h.get('ladder') or []
    return {
        'available': True,
        'active_backend': active,
        'language': lang,
        'ladder': ladder,
        'summary': (
            f"Active TTS backend: {active} for language '{lang}'.  "
            f"Fallback ladder: {' → '.join(ladder) if ladder else '(unknown)'}."
        ),
    }


def get_tier_thresholds() -> Dict[str, Any]:
    """Return the canonical GPU tier threshold table.

    Use this when the user asks: "what's the difference between
    standard and full tier?", "what GPU do I need for speculation?",
    "at what VRAM does Indic Parler load?".
    """
    h = _get('/api/v1/system/tiers')
    if h.get('available') is False:
        # static fallback — must mirror core/gpu_tier.py TIER_THRESHOLDS
        return {
            'available': True,
            'source': 'fallback',
            'tiers': [
                {'name': 'ultra',    'min_vram_gb': 24, 'description': 'Every backend fits concurrently'},
                {'name': 'full',     'min_vram_gb': 10, 'description': 'Speculative decoding for all languages'},
                {'name': 'standard', 'min_vram_gb': 4,  'description': 'Heavy model only, cohort fast-path for English+Kokoro/Piper'},
                {'name': 'none',     'min_vram_gb': 0,  'description': 'CPU-only'},
            ],
            'summary': 'Tier thresholds: ultra≥24GB, full≥10GB, standard≥4GB, none<4GB.',
        }
    tiers = h.get('tiers') or []
    lines = ['GPU tier thresholds:']
    for t in tiers:
        lines.append(f"  - {t.get('name')}: ≥{t.get('min_vram_gb')}GB — {t.get('description', '')}")
    return {**h, 'summary': '\n'.join(lines)}


def get_boot_decision() -> Dict[str, Any]:
    """Report why the current draft-gate / speculation state was chosen.

    Reads the last line of `~/Documents/Nunba/logs/draft_decision.jsonl`
    (written by `LlamaConfig.should_boot_draft` — commit 12c9304).

    Use when the user asks: "why is speculation off on my 8GB GPU?",
    "why didn't Nunba load the draft model?", "what's the cohort
    fast-path?".
    """
    import json
    from pathlib import Path
    log_path = Path.home() / 'Documents' / 'Nunba' / 'logs' / 'draft_decision.jsonl'
    if not log_path.exists():
        return {
            'available': False,
            'summary': (
                "Draft decision log not yet written — this usually means "
                "Nunba has not been booted since the cohort-aware gate "
                "landed (commit 12c9304), or log directory is missing."
            ),
        }
    try:
        with log_path.open(encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        if not lines:
            return {'available': False, 'summary': 'Draft decision log empty.'}
        last = json.loads(lines[-1])
        return {
            'available': True,
            'decision': last.get('decision'),
            'reason': last.get('reason'),
            'lang': last.get('lang'),
            'vram_total_gb': last.get('vram_total_gb'),
            'vram_free_gb': last.get('vram_free_gb'),
            'active_tts': last.get('active_tts'),
            'ts': last.get('ts'),
            'summary': (
                f"Last boot decision (ts={last.get('ts')}): "
                f"{last.get('decision')} — reason: {last.get('reason')}.  "
                f"Context: lang={last.get('lang')}, "
                f"VRAM={last.get('vram_total_gb')}GB total / "
                f"{last.get('vram_free_gb')}GB free, "
                f"active_tts={last.get('active_tts')}."
            ),
        }
    except Exception as e:
        return {
            'available': False,
            'summary': f"Could not parse draft decision log: {e!s}",
        }


def get_system_health() -> Dict[str, Any]:
    """Top-level system health — combines GPU tier + Flask liveness.

    Use when the user asks: "is Nunba healthy?", "what's broken?",
    "can you diagnose why X isn't working?".
    """
    gpu = get_gpu_tier()
    hb = _get('/health')
    flask_ok = hb.get('available') is not False
    parts = [f"Nunba Flask: {'up' if flask_ok else 'down'}."]
    parts.append(gpu.get('summary', ''))
    # If we can, add counts
    models = _get('/api/admin/models')
    if models.get('available') is not False:
        m = models.get('models') or models.get('data') or []
        loaded = [x for x in m if x.get('loaded') or x.get('status') == 'loaded']
        parts.append(f"{len(m)} models registered, {len(loaded)} loaded.")
    return {
        'flask_ok': flask_ok,
        'gpu': gpu,
        'summary': '  '.join(p for p in parts if p),
    }


# ═══════════════════════════════════════════════════════════════════
# Registry + dual-loader
# ═══════════════════════════════════════════════════════════════════

_TOOL_FUNCTIONS: List[Callable[..., Dict[str, Any]]] = [
    get_gpu_tier,
    list_running_models,
    get_tts_status,
    get_tier_thresholds,
    get_boot_decision,
    get_system_health,
]


def get_tool_functions() -> List[Callable[..., Dict[str, Any]]]:
    """Canonical list for anyone who wants the raw functions."""
    return list(_TOOL_FUNCTIONS)


def get_langchain_tools() -> List[Any]:
    """Wrap each introspect function as a LangChain Tool.

    Returns `[]` if langchain isn't importable — no-op for non-agent
    deployments.  Each Tool's `name` matches the function name so agent
    prompts can reference them directly ("call get_gpu_tier").
    """
    try:
        from langchain_core.tools import Tool
    except ImportError:
        try:
            from langchain.agents import Tool  # type: ignore
        except ImportError:
            logger.debug("langchain not importable — system_introspect tools unavailable")
            return []

    tools = []
    for fn in _TOOL_FUNCTIONS:
        def _make_runner(_fn: Callable) -> Callable[[str], str]:
            # LangChain Tool.func receives a single string.  Our
            # functions take no args — just call and return the summary.
            def _run(_ignored_query: str = '') -> str:
                result = _fn()
                return result.get('summary') or str(result)
            _run.__name__ = _fn.__name__
            return _run

        tools.append(Tool(
            name=fn.__name__,
            func=_make_runner(fn),
            description=(fn.__doc__ or fn.__name__).strip().split('\n')[0],
        ))
    return tools


def register_autogen(assistant_agent: Any, user_proxy_agent: Any) -> int:
    """Register every introspect function with an autogen agent pair.

    autogen's `register_function` wires the agent's tool-calling layer
    to the Python callable.  We register the `caller` (assistant) so it
    CAN call, and the `executor` (user proxy) so it RUNS the call.

    Returns count registered.  Silent no-op if autogen isn't importable.
    """
    try:
        from autogen import register_function
    except ImportError:
        logger.debug("autogen not importable — system_introspect tools unavailable")
        return 0
    count = 0
    for fn in _TOOL_FUNCTIONS:
        try:
            register_function(
                fn,
                caller=assistant_agent,
                executor=user_proxy_agent,
                name=fn.__name__,
                description=(fn.__doc__ or fn.__name__).strip().split('\n')[0],
            )
            count += 1
        except Exception as e:
            logger.warning(f"autogen register_function failed for {fn.__name__}: {e}")
    return count


__all__ = [
    'get_gpu_tier',
    'list_running_models',
    'get_tts_status',
    'get_tier_thresholds',
    'get_boot_decision',
    'get_system_health',
    'get_tool_functions',
    'get_langchain_tools',
    'register_autogen',
]
