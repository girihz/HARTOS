"""
core.diag — canonical thread-stack dump for live-hang diagnosis.

WHY THIS EXISTS
───────────────
Three call sites used to maintain their own thread-dump implementations:
  1. app.py `_dump_all_thread_stacks` (startup watchdog)
  2. main.py `/api/admin/diag/thread-dump` admin endpoint
  3. HARTOS security/node_watchdog.py (NodeWatchdog FROZEN restart)

Sites 2 and 3 each carried fragile `getattr(__main__, '_dump_...')` lookup
chains because in frozen mode `app.py` is the entry script and the function
isn't reachable by normal import.  Whenever the symbol moved or the frozen
bundle layout changed, the chain silently fell through to "thread dump
unavailable" — defeating the entire diagnostic purpose.

This module is the SINGLE canonical implementation.  It also publishes
itself on `builtins._nunba_dump_threads` so frozen-mode lookups (where
neither `core.diag` nor `app` may be import-resolvable from a watchdog
spawned by HARTOS) keep working via the same trick `app.py` uses for
`_nunba_trace`.

WHO CALLS IT
────────────
- app.py `_startup_watchdog` (15s into a stalled startup phase)
- main.py `/api/admin/diag/thread-dump` (operator on-demand)
- HARTOS security/node_watchdog.py (BEFORE killing a frozen daemon)

WHERE THE DUMP GOES
───────────────────
Two sinks ALWAYS:
  - The Python logger (may be delayed if MainThread holds the GIL).
  - `_nunba_trace` builtin (flushes immediately to startup_trace.log,
    survives GIL-held hangs — the whole reason this module exists).

Callers can inject extra sinks (e.g., WAMP publish, metrics counter,
crash-reporter breadcrumb) via the `sinks` parameter without monkey-patching.

NOT FOR PRODUCTION TELEMETRY
────────────────────────────
Stack dumps leak file paths, env hints (cwd in tracebacks), and call-graph
shape.  Admin endpoint MUST stay behind `require_local_or_token` on flat
tier and `require_central` on regional/central tier.
"""
from __future__ import annotations

import builtins
import logging
import sys
import threading
import traceback
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def _trace_sink(payload: str) -> None:
    """Write to the startup trace channel (immediate flush, GIL-resilient).

    The `_nunba_trace` builtin is published by app.py at process boot.  If
    the watchdog runs before app.py finished initialising (or in a stripped
    test environment), this is a silent no-op.
    """
    _t = getattr(builtins, '_nunba_trace', None)
    if _t is None:
        return
    try:
        _t(payload)
    except Exception:
        # The trace sink itself failing is a last-line-of-defence failure;
        # we cannot recurse into logger.error here because that's what the
        # caller already tried.  Swallow.
        pass


def _logger_sink(payload: str) -> None:
    """Write to the Python logger.  May be delayed if MainThread is wedged."""
    try:
        logger.error(payload)
    except Exception:
        pass


def dump_all_thread_stacks(
    reason: str,
    sinks: Optional[List[Callable[[str], None]]] = None,
) -> str:
    """Dump EVERY live thread (including MainThread) with its current Python
    stack frame.

    Args:
        reason: Human-readable why-this-fired (e.g. "Phase 'wmic_probe'
            stuck 30s", "admin diag", "NodeWatchdog FROZEN restart: tts").
            Included as the dump header so cross-referencing logs is easy.
        sinks: Extra one-arg callables that receive the formatted payload
            string.  The default [logger, trace] sinks are ALWAYS invoked
            in addition to whatever you pass.  Use this for WAMP publish,
            metrics, crash-reporter breadcrumb — anything that must NOT
            replace the canonical sinks.

    Returns:
        The formatted multi-line dump string (also returned so test code
        can assert against it without monkey-patching the logger).
    """
    # Build the dump payload first — no I/O while collecting frames so we
    # capture a coherent snapshot even if a thread is mid-syscall.
    lines = [f"[THREAD DUMP] {reason}"]
    try:
        frames = sys._current_frames()
    except Exception as e:
        # _current_frames is a CPython feature; on a non-CPython runtime
        # it could raise.  Emit a stub so the caller still gets feedback.
        payload = f"[THREAD DUMP] {reason}\n  (_current_frames unavailable: {e})"
        for sink in (_logger_sink, _trace_sink, *(sinks or [])):
            try:
                sink(payload)
            except Exception:
                pass
        return payload

    name_by_id = {t.ident: t.name for t in threading.enumerate()}
    try:
        main_ident = threading.main_thread().ident
    except Exception:
        main_ident = None

    for tid, frame in frames.items():
        tname = name_by_id.get(tid, 'unknown')
        marker = ' [MAIN]' if tid == main_ident else ''
        lines.append(f"  ── Thread {tname}{marker} (id={tid}) ──")
        try:
            formatted = traceback.format_stack(frame)
            lines.append('    ' + '    '.join(formatted).rstrip())
        except Exception as fe:
            lines.append(f"    (format_stack failed: {fe})")

    payload = '\n'.join(lines)

    # Always-invoked sinks: logger + trace.  Caller-supplied sinks run AFTER
    # the canonical pair so a buggy custom sink can't suppress diagnostics.
    for sink in (_logger_sink, _trace_sink, *(sinks or [])):
        try:
            sink(payload)
        except Exception:
            # A failing sink must not stop the others.  We deliberately do
            # not log the failure (would recurse if logger sink is broken).
            pass

    return payload


# ── Builtin publication for frozen-mode cross-module lookup ──────────────
# In frozen bundles HARTOS-side watchdogs may not be able to `import core.diag`
# because the bundle's importlib-machinery only sees Nunba's own pyc cache
# and HARTOS sits in a sibling site-packages.  Exposing the function on
# `builtins` mirrors how `app.py` publishes `_nunba_trace` — the watchdog
# does `getattr(__import__('builtins'), '_nunba_dump_threads', None)` and
# gets a working callable regardless of import topology.
#
# Idempotent: re-publishing during hot-reload / repeated test imports is OK.
try:
    builtins._nunba_dump_threads = dump_all_thread_stacks  # type: ignore[attr-defined]
except Exception:
    pass


__all__ = ['dump_all_thread_stacks']
