"""Stage-C Symptom #6 FT — agent_approval publishes consent on WAMP.

House-rule 6 (crossbar-realtime.md + _house_rules.md section 6) is
categorical: "ALL real-time push uses Crossbar WAMP. Never SSE,
never raw WebSocket." The original consent flow relied on an HTTP
response + a raw WebSocket on port 5460 for frame ingress. The
consent event itself had no push surface — UI components either
polled or relied on side effects.

This test guards the fix: /api/agent/approval now calls
publish_async('com.hertzai.hevolve.vision.<user_id>', ...) on BOTH
approve and deny paths. Subscribers (VisionService, frontend SPA,
React Native) receive the decision via the canonical WAMP topic.

AST-only — the full Flask app imports are too heavy for this file
to instantiate (requires LangChain+autogen). We grep the source for
the publish call + topic string.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parent.parent / "hart_intelligence_entry.py"


class AgentApprovalWAMPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SRC_PATH.read_text(encoding="utf-8")
        # Pull out the agent_approval function body so we only check
        # within its scope, not elsewhere in the file.
        match = re.search(
            r"def agent_approval\(\):.*?\n(?=^(?:def |@app\.route))",
            cls.src,
            re.DOTALL | re.MULTILINE,
        )
        cls.func_body = match.group(0) if match else ""

    def test_function_found(self):
        self.assertTrue(
            self.func_body,
            "agent_approval() function body could not be located",
        )

    def test_approve_publishes_wamp_topic(self):
        self.assertIn(
            "com.hertzai.hevolve.vision",
            self.func_body,
            "agent_approval must publish to com.hertzai.hevolve.vision.<user_id>"
        )

    def test_publish_async_called(self):
        self.assertIn(
            "publish_async(",
            self.func_body,
            "agent_approval must call publish_async() on approve/deny"
        )

    def test_publish_payload_has_type_consent(self):
        self.assertIn(
            "'type': 'consent'",
            self.func_body,
            "WAMP payload must set type='consent' so subscribers can filter"
        )

    def test_publish_fires_on_both_approve_and_deny(self):
        """Count publish_async occurrences — one for approve, one for deny.
        Tolerates more (e.g. unknown-action path) as long as there are >=2."""
        count = self.func_body.count("publish_async(")
        self.assertGreaterEqual(
            count, 2,
            f"Expected >= 2 publish_async calls (approve + deny), got {count}"
        )

    def test_publish_failure_does_not_break_http_response(self):
        """The WAMP publish is wrapped in try/except — a publish failure
        must NOT roll back the approval or change the HTTP response."""
        # Crude but effective: find the publish_async call and verify
        # an `except` clause appears within 30 lines.
        lines = self.func_body.split("\n")
        for i, line in enumerate(lines):
            if "publish_async(" in line:
                context = "\n".join(lines[i:i + 30])
                self.assertIn(
                    "except",
                    context,
                    f"publish_async at line ~{i} must be wrapped in try/except "
                    "(a WAMP failure must not roll back an approval)"
                )
                return
        self.fail("No publish_async call found to validate try/except wrapping")

    def test_topic_is_keyed_by_user_id(self):
        """Topic must end in the user_id so per-user WAMP subscribers
        receive only their own consent events."""
        # Look for f-string or format patterns that interpolate user_id
        self.assertTrue(
            re.search(
                r"com\.hertzai\.hevolve\.vision\.\{[^}]*(?:user_id|cu|_cu)",
                self.func_body,
            ),
            "WAMP topic must be keyed by user_id"
        )


if __name__ == "__main__":
    unittest.main()
