"""Invariants for the canonical language collections in core.constants.

Failing tests = DRY violation: someone inlined a duplicate frozenset
for "non-Latin script" or "Indic" instead of importing from the single
source of truth.  Fix by importing from core.constants.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_non_latin_script_langs_exists():
    from core.constants import NON_LATIN_SCRIPT_LANGS, SUPPORTED_LANG_DICT
    assert isinstance(NON_LATIN_SCRIPT_LANGS, frozenset)
    # Every code in the set must be a valid ISO 639-1 from the canonical
    # language-name dict.  If this fails, a new lang was added to the
    # set without registering its display name.
    unknown = NON_LATIN_SCRIPT_LANGS - set(SUPPORTED_LANG_DICT)
    assert not unknown, f"Unknown codes in NON_LATIN_SCRIPT_LANGS: {unknown}"


def test_indic_langs_exists_and_subset():
    from core.constants import INDIC_LANGS, NON_LATIN_SCRIPT_LANGS
    assert isinstance(INDIC_LANGS, frozenset)
    assert INDIC_LANGS <= NON_LATIN_SCRIPT_LANGS, (
        "INDIC_LANGS must be a subset of NON_LATIN_SCRIPT_LANGS"
    )


def test_critical_scripts_in_non_latin():
    from core.constants import NON_LATIN_SCRIPT_LANGS
    for code in ('ta', 'hi', 'bn', 'te', 'mr', 'zh', 'ja', 'ko',
                 'ar', 'he', 'th'):
        assert code in NON_LATIN_SCRIPT_LANGS, (
            f"{code!r} must be in NON_LATIN_SCRIPT_LANGS — the 0.8B draft "
            f"cannot produce its native script."
        )


def test_english_not_in_non_latin():
    from core.constants import NON_LATIN_SCRIPT_LANGS
    assert 'en' not in NON_LATIN_SCRIPT_LANGS


def test_dispatcher_imports_canonical_set():
    """speculative_dispatcher.py MUST NOT define its own frozenset.
    If this fails, revert the inline `_skip_draft_langs = frozenset({...})`
    and `from core.constants import NON_LATIN_SCRIPT_LANGS`."""
    src = Path(
        'C:/Users/sathi/PycharmProjects/HARTOS/integrations/agent_engine/'
        'speculative_dispatcher.py'
    ).read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                    '_skip_draft_langs', '_indic_or_script', '_non_latin',
                ):
                    pytest.fail(
                        f"Inline frozenset `{t.id}` at line {node.lineno} — "
                        f"import from core.constants.NON_LATIN_SCRIPT_LANGS instead",
                    )


def test_hart_intelligence_entry_imports_canonical_set():
    src = Path(
        'C:/Users/sathi/PycharmProjects/HARTOS/hart_intelligence_entry.py'
    ).read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                    '_skip_draft_langs', '_indic_or_script', '_non_latin',
                ):
                    pytest.fail(
                        f"Inline frozenset `{t.id}` at line {node.lineno} — "
                        f"import from core.constants.NON_LATIN_SCRIPT_LANGS",
                    )


def test_tts_engine_imports_indic_langs_from_canonical():
    src = Path(
        'C:/Users/sathi/PycharmProjects/Nunba-HART-Companion/tts/tts_engine.py'
    ).read_text(encoding='utf-8')
    # Must have the import line; must NOT still have `_INDIC_LANGS = {...}`
    assert 'from core.constants import' in src, (
        "tts_engine.py must import INDIC_LANGS from core.constants"
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == '_INDIC_LANGS':
                    # Only acceptable if it's a re-export/alias (RHS is Name
                    # pointing to the imported canonical).  A set literal
                    # means DRY violation.
                    if not isinstance(node.value, ast.Name):
                        pytest.fail(
                            f"_INDIC_LANGS inline at line {node.lineno} — "
                            f"import from core.constants.INDIC_LANGS instead"
                        )


def test_no_cjk_rtl_duplicates_across_files():
    """No two source files should each define their own frozenset of
    CJK/RTL codes.  There's only one concept; it lives in constants."""
    offenders = []
    for path in (
        'C:/Users/sathi/PycharmProjects/HARTOS/integrations/agent_engine/'
        'speculative_dispatcher.py',
        'C:/Users/sathi/PycharmProjects/HARTOS/hart_intelligence_entry.py',
        'C:/Users/sathi/PycharmProjects/Nunba-HART-Companion/tts/tts_engine.py',
    ):
        text = Path(path).read_text(encoding='utf-8')
        # Heuristic: CJK + RTL codes appearing together in a `frozenset({...})`
        # literal is the signature of a non-Latin-script set.
        if ("'zh'" in text or '"zh"' in text) and (
            "'ar'" in text or '"ar"' in text
        ):
            # OK if they only appear via `from core.constants import`
            if 'NON_LATIN_SCRIPT_LANGS' not in text:
                offenders.append(path)
    assert not offenders, (
        f"Files with inline CJK+RTL frozensets (DRY violation): {offenders}"
    )
