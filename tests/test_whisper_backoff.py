"""NFT — faster-whisper retry storm containment.

Stage-A Symptom #10 guard (2026-04-16).

Previously: frozen_debug.log showed ~2Hz 'faster-whisper transcription
failed: HF_HUB_OFFLINE' — every streaming chunk hit the same broken
load path with no backoff, no breaker, no user-visible error.

Now: module-level circuit breaker (5 failures -> OPEN for 300s) and
exponential backoff (1s -> 2s -> 4s -> ... -> cap 300s) gate the load
path so one failed streaming session produces O(log N) log lines
instead of N.

Contract tested:
1. Fresh module: breaker CLOSED, backoff clear.
2. 5 recorded failures -> breaker OPENS -> subsequent _get_faster_
   whisper_model raises without touching the network.
3. get_whisper_last_error surfaces the most recent reason string.
4. record_success clears breaker, backoff, and last-error.
5. _faster_whisper_transcribe returns None (graceful, no log spam)
   when the breaker is OPEN.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest


def _reset_whisper_module():
    """Reset the shared module-level breaker/backoff state for test isolation."""
    from integrations.service_tools import whisper_tool as wt
    if wt._whisper_load_breaker is not None:
        wt._whisper_load_breaker.reset()
    if wt._whisper_load_backoff is not None:
        # PeerBackoff has no reset API — clear the dict via the method
        wt._whisper_load_backoff._entries.clear()
    wt._whisper_last_error = None
    wt._faster_whisper_model = None
    wt._faster_whisper_model_size = None


class WhisperBreakerTests(unittest.TestCase):
    def setUp(self):
        _reset_whisper_module()

    def test_fresh_module_has_closed_breaker(self):
        from integrations.service_tools import whisper_tool as wt
        self.assertIsNotNone(wt._whisper_load_breaker,
                             "CircuitBreaker must be importable at module load")
        self.assertFalse(wt._whisper_load_breaker.is_open())
        self.assertIsNone(wt.get_whisper_last_error())

    def test_five_failures_open_the_breaker(self):
        from integrations.service_tools import whisper_tool as wt
        for i in range(5):
            wt._record_whisper_failure(f"attempt {i}: fake failure")
        self.assertTrue(wt._whisper_load_breaker.is_open(),
                        "Breaker must OPEN after 5 consecutive failures")
        self.assertIn("fake failure", wt.get_whisper_last_error() or "")

    def test_get_model_refuses_when_breaker_open(self):
        """_get_faster_whisper_model MUST raise RuntimeError without
        even importing faster_whisper when the breaker is OPEN."""
        from integrations.service_tools import whisper_tool as wt
        for _ in range(5):
            wt._record_whisper_failure("fake")

        with self.assertRaises(RuntimeError) as ctx:
            wt._get_faster_whisper_model("base")
        self.assertIn("circuit breaker OPEN", str(ctx.exception))

    def test_transcribe_returns_none_when_breaker_open(self):
        """_faster_whisper_transcribe must fast-return None without
        logging at 2Hz when the breaker is OPEN."""
        from integrations.service_tools import whisper_tool as wt
        for _ in range(5):
            wt._record_whisper_failure("fake")
        # No patching needed — breaker OPEN short-circuits before the
        # real faster_whisper import is touched.
        result = wt._faster_whisper_transcribe("/tmp/fake.wav")
        self.assertIsNone(result)

    def test_success_clears_breaker_and_backoff(self):
        from integrations.service_tools import whisper_tool as wt
        for _ in range(5):
            wt._record_whisper_failure("fake")
        self.assertTrue(wt._whisper_load_breaker.is_open())

        wt._record_whisper_success()

        self.assertFalse(wt._whisper_load_breaker.is_open(),
                         "Success must close the breaker")
        self.assertIsNone(wt.get_whisper_last_error(),
                          "Success must clear last_error")
        self.assertFalse(
            wt._whisper_load_backoff.is_backed_off('faster_whisper'),
            "Success must clear backoff"
        )

    def test_backoff_exists_on_failure(self):
        from integrations.service_tools import whisper_tool as wt
        wt._record_whisper_failure("first failure")
        self.assertTrue(
            wt._whisper_load_backoff.is_backed_off('faster_whisper'),
            "First failure must set the backoff window"
        )

    def test_last_error_captures_reason_string(self):
        from integrations.service_tools import whisper_tool as wt
        wt._record_whisper_failure("HF_HUB_OFFLINE — specific reason")
        self.assertEqual(
            wt.get_whisper_last_error(),
            "HF_HUB_OFFLINE — specific reason",
        )

    def test_breaker_cooldown_is_five_minutes(self):
        """Symptom #10 spec: cap backoff at 300s, breaker cooldown 300s."""
        from integrations.service_tools import whisper_tool as wt
        self.assertEqual(wt._whisper_load_breaker.cooldown, 300.0)
        self.assertEqual(wt._whisper_load_backoff.maximum, 300.0)


if __name__ == "__main__":
    unittest.main()
