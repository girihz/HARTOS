"""Tests for core.diag — the canonical thread-stack dumper.

Replaces ad-hoc verification of three (now-removed) parallel
implementations.  These are unit tests — no network, no Flask app.
"""
from __future__ import annotations

import builtins
import threading

import pytest


@pytest.fixture
def diag_module():
    """Re-import core.diag so the builtin publication runs fresh.

    Other tests may have stomped `builtins._nunba_dump_threads`; this
    guarantees a clean baseline per-test.
    """
    import importlib

    import core.diag as _d
    importlib.reload(_d)
    return _d


def test_dumps_main_thread_with_marker(diag_module):
    """The dump must include the [MAIN] marker on MainThread so operators
    reading a 30-thread dump can find the relevant frame in O(1)."""
    payload = diag_module.dump_all_thread_stacks("unit-test reason")
    assert "[THREAD DUMP] unit-test reason" in payload
    assert "[MAIN]" in payload
    # All non-main threads should NOT carry [MAIN]
    main_marker_count = payload.count("[MAIN]")
    assert main_marker_count == 1, (
        f"Expected exactly one [MAIN] marker, got {main_marker_count}"
    )


def test_handles_missing_logger(diag_module, monkeypatch):
    """If `logger.error` raises (e.g., handler closed mid-shutdown), the
    dump must still complete and return its payload — the trace channel
    is the safety net.  Prior to refactor this worked by accident; the
    new module guarantees it."""
    def _bad_error(*_a, **_kw):
        raise RuntimeError("logger detonated")

    monkeypatch.setattr(diag_module.logger, 'error', _bad_error)

    # Must not raise — sink failures are swallowed individually.
    payload = diag_module.dump_all_thread_stacks("logger-broken")
    assert "[THREAD DUMP] logger-broken" in payload


def test_writes_to_trace_builtin_when_available(diag_module):
    """The `_nunba_trace` builtin (published by app.py at boot) is the
    GIL-resilient sink.  Verify the dumper invokes it.

    We replace the builtin with a capturing list, run the dump, then
    restore.  This is the contract that lets startup_trace.log capture
    dumps even when MainThread is wedged in a GIL-holding C call.
    """
    captured: list[str] = []
    original = getattr(builtins, '_nunba_trace', None)
    builtins._nunba_trace = captured.append  # type: ignore[attr-defined]
    try:
        diag_module.dump_all_thread_stacks("trace-sink-test")
    finally:
        if original is None:
            try:
                delattr(builtins, '_nunba_trace')
            except AttributeError:
                pass
        else:
            builtins._nunba_trace = original  # type: ignore[attr-defined]

    assert len(captured) == 1, (
        f"trace sink must be invoked exactly once, got {len(captured)}"
    )
    assert "[THREAD DUMP] trace-sink-test" in captured[0]


def test_extra_sink_receives_payload(diag_module):
    """Callers can inject additional sinks (WAMP publish, metrics counter,
    crash-reporter breadcrumb) without monkey-patching the canonical pair."""
    received: list[str] = []
    diag_module.dump_all_thread_stacks(
        "extra-sink", sinks=[received.append],
    )
    assert len(received) == 1
    assert "[THREAD DUMP] extra-sink" in received[0]


def test_failing_extra_sink_does_not_break_others(diag_module):
    """A buggy custom sink must not stop the canonical sinks from running.
    This is the contract that makes `sinks=` safe to expose to plugins."""
    good: list[str] = []

    def _bad_sink(_payload: str) -> None:
        raise RuntimeError("plugin sink crashed")

    payload = diag_module.dump_all_thread_stacks(
        "buggy-plugin-sink",
        sinks=[_bad_sink, good.append],
    )
    assert len(good) == 1
    assert "[THREAD DUMP] buggy-plugin-sink" in payload


def test_builtin_published_for_frozen_lookup(diag_module):
    """The module must publish itself on `builtins._nunba_dump_threads`
    so frozen-mode HARTOS watchdogs can find it without an import path."""
    fn = getattr(builtins, '_nunba_dump_threads', None)
    assert callable(fn), (
        "core.diag must publish dump_all_thread_stacks on builtins for "
        "frozen-mode cross-bundle lookup (mirrors `_nunba_trace` pattern)"
    )
    payload = fn("from-builtin")
    assert "[THREAD DUMP] from-builtin" in payload


def test_includes_a_secondary_thread(diag_module):
    """Sanity: when we spawn a thread, the dump must include it.  Without
    this assertion the test would pass even if the dumper only ever
    captured MainThread."""
    started = threading.Event()
    stop = threading.Event()

    def _worker():
        started.set()
        stop.wait(timeout=5)

    t = threading.Thread(target=_worker, name="diag-test-worker")
    t.start()
    try:
        assert started.wait(timeout=2), "worker thread failed to start"
        payload = diag_module.dump_all_thread_stacks("multi-thread")
        assert "diag-test-worker" in payload
    finally:
        stop.set()
        t.join(timeout=5)
