"""Module-level constants shared across HARTOS.

This file is the single source of truth for literal values that were
previously hardcoded in multiple modules. Before this file existed the
channel registry, flask integration, dynamic agent registry, test
fixtures, and example scripts each carried their own copy of
``10077`` / ``8888`` with no mechanism to keep them in sync.

Import from here instead of repeating literals:

    from core.constants import DEFAULT_USER_ID, DEFAULT_PROMPT_ID

Why these specific values:
    DEFAULT_USER_ID = 10077 — the guest/unauthenticated Hevolve user
        account used by channel adapters, test fixtures, and
        standalone entry points that haven't resolved a real user yet.
        Any real user_id comes from UserChannelBinding resolution,
        JWT auth, or the frontend session — the default only fires
        when every other source is empty.
    DEFAULT_PROMPT_ID = 8888 — the pre-registered default agent prompt
        that serves generic chat when no custom agent_id is provided.
        Tests and the channel fallback path both point here so a
        brand-new install answers chat requests out of the box.
"""

DEFAULT_USER_ID: int = 10077
DEFAULT_PROMPT_ID: int = 8888

# ISO 639-1 → language name mapping.
# Used by hart_intelligence_entry (system prompt), speculative_dispatcher
# (draft language prompt), and _persist_language (validation).
SUPPORTED_LANG_DICT = {
    "ar": "Arabic", "bg": "Bulgarian", "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)", "nl": "Dutch", "fi": "Finnish",
    "fr": "French", "de": "German", "el": "Greek", "he": "Hebrew",
    "hu": "Hungarian", "is": "Icelandic", "id": "Indonesian",
    "ko": "Korean", "lv": "Latvian", "ms": "Malay", "fa": "Persian",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "es": "Spanish", "sw": "Swahili", "sv": "Swedish", "th": "Thai",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
    "cy": "Welsh", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
    "pa": "Punjabi", "gu": "Gujarati", "kn": "Kannada", "te": "Telugu",
    "mr": "Marathi", "ml": "Malayalam", "en": "English",
    "ja": "Japanese", "it": "Italian", "ne": "Nepali", "si": "Sinhala",
    "or": "Odia", "as": "Assamese", "sd": "Sindhi", "ks": "Kashmiri",
    "doi": "Dogri", "mni": "Manipuri", "sa": "Sanskrit", "kok": "Konkani",
    "mai": "Maithili", "brx": "Bodo", "sat": "Santali",
    # SEA Brahmi-derived scripts — added for NON_LATIN_SCRIPT_LANGS
    # membership so the sub-1B draft-skip gate recognises them.
    "km": "Khmer", "lo": "Lao", "my": "Burmese",
    # Cyrillic / Greek — weaker but non-zero 0.8B coverage; listed so
    # NON_LATIN_SCRIPT_LANGS assertion passes.
    "sr": "Serbian",
}


# Indic-language ISO 639-1 codes (Brahmi-family scripts + Urdu/Sindhi
# in Perso-Arabic).  Subset used by TTS routing (Indic Parler) and by
# NON_LATIN_SCRIPT_LANGS below.  Single source for any code that needs
# "is this an Indic language?" — previously duplicated as _INDIC_LANGS
# in tts/tts_engine.py.
INDIC_LANGS = frozenset({
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta",
    "te", "ur",
})


# ISO 639-1 codes where sub-1B LLMs (the Qwen3.5-0.8B-class draft
# model) produce Latin-transliterated output ("Vanakkam" instead of
# native Tamil script) due to weak Unicode-script tokenizer coverage.
#
# Single source of truth — consumed by:
#   - integrations.agent_engine.speculative_dispatcher
#     (dispatch_draft_first skip-gate at runtime)
#   - integrations.service_tools.model_lifecycle
#     (on_lang_change subscriber — evicts draft on switch TO these)
#
# Derived from INDIC_LANGS plus the other non-Latin script families.
# Do NOT inline a duplicate frozenset anywhere else — import this.
NON_LATIN_SCRIPT_LANGS = INDIC_LANGS | frozenset({
    # CJK
    "zh", "ja", "ko",
    # RTL (Arabic / Hebrew / Persian)
    "ar", "he", "fa",
    # Southeast Asian Brahmi-derived
    "th", "lo", "km", "my",
    # Cyrillic + Greek (historically included by HIE's inline
    # _NON_LATIN_LANGS; kept here for parity + weaker 0.8B coverage)
    "ru", "uk", "bg", "sr", "el",
})

# Invariant: every code in NON_LATIN_SCRIPT_LANGS must be a registered
# language in SUPPORTED_LANG_DICT.  Fails loud at import time on drift,
# so adding a code to the set without registering its display name is
# a build-time error, not a runtime mystery.
assert NON_LATIN_SCRIPT_LANGS <= set(SUPPORTED_LANG_DICT), (
    f"NON_LATIN_SCRIPT_LANGS has codes not in SUPPORTED_LANG_DICT: "
    f"{NON_LATIN_SCRIPT_LANGS - set(SUPPORTED_LANG_DICT)}"
)
