"""Tests for the monoscript invariant on Indic tone blocks.

Background:
  Commit 77245da (2026-04-13) reverted an explicit "use native script"
  rule under the rationale that Piper TTS handled romanised code-mixing
  fine.  That rationale no longer holds: once Indic Parler / Kokoro
  are in the backend ladder, the TTS synthesises from native script
  and can't pronounce Latin-transliterated Tamil/Hindi/etc.

  These tests pin the current behaviour — the `full`-tier Indic
  language tone blocks (ta, hi, bn, te, mr) MUST contain only
  native-script output examples, with English loanwords transliterated
  into the target script inside the examples.  The romanised vocabulary
  survives only as a pipe-separated HINT ("நண்பா|nanba") for the LLM's
  own use — never as standalone output.
"""
from __future__ import annotations

import re

import pytest


# Script ranges we want to see in the `full` Indic tone blocks.
_SCRIPT_RANGES = {
    'ta': ('\u0B80', '\u0BFF'),   # Tamil
    'hi': ('\u0900', '\u097F'),   # Devanagari
    'bn': ('\u0980', '\u09FF'),   # Bengali
    'te': ('\u0C00', '\u0C7F'),   # Telugu
    'mr': ('\u0900', '\u097F'),   # Devanagari (shared with Hindi)
}


@pytest.mark.parametrize('lang_code', list(_SCRIPT_RANGES))
def test_full_tier_indic_blocks_contain_native_script_examples(lang_code):
    """Every full-tier Indic language must have at least one example
    sentence written in its native script — proves we stopped shipping
    pure Tanglish/Hinglish tone prompts."""
    from core.agent_personality import _REGIONAL_TONE_DATA
    entry = _REGIONAL_TONE_DATA[lang_code]
    _name, tier, phrases = entry
    assert tier == 'full', f"{lang_code} expected full tier"
    lo, hi = _SCRIPT_RANGES[lang_code]
    has_native = any(lo <= ch <= hi for ch in phrases)
    assert has_native, (
        f"Tone block for {lang_code} has no native-script characters "
        f"in range {lo!r}..{hi!r}; reverted to Tanglish/Hinglish?"
    )


@pytest.mark.parametrize('lang_code', list(_SCRIPT_RANGES))
def test_full_tier_indic_blocks_declare_monoscript_rule(lang_code):
    """The block must explicitly forbid Latin output — so a future
    refactor can't silently drop the rule without this test failing."""
    from core.agent_personality import _REGIONAL_TONE_DATA
    _name, _tier, phrases = _REGIONAL_TONE_DATA[lang_code]
    lower = phrases.lower()
    assert any(phrase in lower for phrase in (
        'no latin letters',
        'fully in',
        'entirely in',
    )), (
        f"Tone block for {lang_code} doesn't declare monoscript rule. "
        f"TTS needs native-script-only output; a new refactor can't "
        f"drop this invariant silently."
    )


def test_build_tone_prompt_tamil_includes_monoscript_directive():
    """The assembled prompt (what the LLM actually sees) must include
    the SCRIPT: directive — otherwise _build_tone_prompt regressed."""
    from core.agent_personality import _build_tone_prompt
    prompt = _build_tone_prompt('ta')
    assert 'SCRIPT:' in prompt, "Tamil prompt missing SCRIPT directive"
    assert 'Tamil' in prompt
    # Must also contain real Tamil script somewhere in the assembled prompt
    assert any('\u0B80' <= ch <= '\u0BFF' for ch in prompt), (
        "Assembled Tamil prompt has no Tamil script characters"
    )


def test_build_tone_prompt_english_returns_empty():
    """Sanity: English users get no regional tone prompt."""
    from core.agent_personality import _build_tone_prompt
    assert _build_tone_prompt('en') == ''


def test_roman_hint_convention_uses_pipe_separator():
    """The 'script|roman' vocabulary convention lets the LLM see both
    the target-script word and its pronunciation hint without ever
    emitting romanised output.  If the convention breaks, the LLM
    loses its phonetic anchor and may fabricate incorrect native-
    script spellings."""
    from core.agent_personality import _REGIONAL_TONE_DATA
    _name, _tier, phrases = _REGIONAL_TONE_DATA['ta']
    # Expect at least a few pipe-separated pairs: native|roman
    assert '|' in phrases, "Tamil vocabulary lost the pipe-separated roman hint format"
    # Spot-check a known anchor word
    assert 'நண்பா|nanba' in phrases, (
        "Tamil vocabulary missing the நண்பா|nanba anchor — the LLM "
        "needs both the script form and the pronunciation hint"
    )
