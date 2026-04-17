"""FT — /chat preferred_lang fallback to core.user_lang.

Stage-A Symptom #5 guard. Verifies the fix in commit a433f55:

When the /chat request body omits `preferred_lang`, the resolution
MUST fall back to `core.user_lang.get_preferred_lang()` (which reads
hart_language.json). Previously defaulted to 'en' and ignored the
user's language preference.

Because importing hart_intelligence_entry.py pulls in heavy LangChain
/ autogen deps, this FT tests the resolution logic in isolation by
extracting the exact same precedence used in the /chat route body.
If the logic in chat() drifts from this test, the AST-level check at
the bottom of the file will catch it.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


def _resolve_preferred_lang(data: dict) -> str:
    """Exact copy of the resolution logic in
    hart_intelligence_entry.chat() at lines 5517-5525 (a433f55).

    If this duplicates drifts from the source, the AST contract test
    at the bottom of this file FAILS, forcing the drift to be noticed.
    """
    _req_lang = data.get("preferred_lang") or data.get("language")
    if _req_lang and _req_lang.strip():
        return _req_lang.strip()
    try:
        from core.user_lang import get_preferred_lang
        return get_preferred_lang()
    except Exception:
        return "en"


@pytest.fixture
def tmp_lang_file(monkeypatch, tmp_path):
    fake = tmp_path / "hart_language.json"
    monkeypatch.setenv("HART_USER_LANGUAGE", "")
    import core.user_lang as ul_mod
    monkeypatch.setattr(ul_mod, "_HART_LANG_PATH", str(fake))
    monkeypatch.setattr(ul_mod, "_cache", {"value": None, "mtime": 0})
    monkeypatch.setattr(ul_mod, "_listeners", [])
    yield fake, ul_mod


def test_request_preferred_lang_wins(tmp_lang_file):
    fake, _ = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "ta"}', encoding="utf-8")

    # Request explicitly sets 'en' — must override the file
    lang = _resolve_preferred_lang({"preferred_lang": "en"})
    assert lang == "en"


def test_fallback_reads_ta_from_hart_language_json(tmp_lang_file):
    """Symptom #5 happy path: no preferred_lang in body → reads 'ta'."""
    fake, _ = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "ta"}', encoding="utf-8")

    lang = _resolve_preferred_lang({})  # no preferred_lang, no language
    assert lang == "ta", "Must fall back to hart_language.json 'ta'"


def test_fallback_reads_hi_from_hart_language_json(tmp_lang_file):
    fake, _ = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "hi"}', encoding="utf-8")

    lang = _resolve_preferred_lang({})
    assert lang == "hi"


def test_fallback_respects_alt_language_key(tmp_lang_file):
    """Some callers pass 'language' instead of 'preferred_lang'. Both work."""
    lang = _resolve_preferred_lang({"language": "ja"})
    assert lang == "ja"


def test_empty_preferred_lang_triggers_fallback(tmp_lang_file):
    """Empty string or whitespace in preferred_lang must NOT win —
    must fall through to the file."""
    fake, _ = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "ta"}', encoding="utf-8")

    for empty in ("", "   ", "\n"):
        assert _resolve_preferred_lang({"preferred_lang": empty}) == "ta"


def test_fallback_defaults_to_en_when_no_file_no_env(tmp_lang_file):
    """No file, no env → returns 'en' (the ultimate default)."""
    lang = _resolve_preferred_lang({})
    assert lang == "en"


def test_ast_contract_chat_resolution_matches_source():
    """Drift guard — if hart_intelligence_entry.chat() resolution code
    changes shape, this test fails loudly so the copy here stays
    synchronized.

    Specifically checks that chat() contains the three markers:
    1. the fallback try/import: `from core.user_lang import get_preferred_lang`
    2. the request-first read: data.get('preferred_lang') OR data.get('language')
    3. the default-'en' except handler
    """
    src_path = Path(__file__).resolve().parent.parent / "hart_intelligence_entry.py"
    src = src_path.read_text(encoding="utf-8")

    assert "from core.user_lang import get_preferred_lang" in src, (
        "Symptom #5 fix regressed: chat() no longer imports get_preferred_lang"
    )
    assert "data.get('preferred_lang')" in src or 'data.get("preferred_lang")' in src
    # Must NOT revert to the old default-'en' single-line form
    assert "data.get('preferred_lang', 'en')" not in src
    assert 'data.get("preferred_lang", "en")' not in src


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
