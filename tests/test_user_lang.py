"""Tests for core.user_lang — canonical language read/write layer.

Verifies:
  * Precedence order: request_override > file > env > default
  * Atomic write survives simulated ENOSPC
  * set_preferred_lang persists English (no stuck-English guard)
  * Idempotent re-write skips disk I/O
  * on_lang_change subscribers fire only on transition, not no-op
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_lang_file(monkeypatch, tmp_path):
    """Point core.user_lang at an empty temp dir by directly overriding
    its module-level _HART_LANG_PATH constant — cleaner than trying to
    re-derive it via patched expanduser/Path.home (which requires a
    reimport that races the patch)."""
    fake = tmp_path / 'hart_language.json'
    monkeypatch.setenv('HART_USER_LANGUAGE', '')  # clear env override
    import core.user_lang as ul_mod
    monkeypatch.setattr(ul_mod, '_HART_LANG_PATH', str(fake))
    # Clear cache so each test starts fresh
    monkeypatch.setattr(ul_mod, '_cache', {'value': None, 'mtime': 0})
    # Also clear listeners list so test isolation is preserved
    monkeypatch.setattr(ul_mod, '_listeners', [])
    yield fake, ul_mod


def test_get_preferred_lang_request_override_wins(tmp_lang_file):
    _, ul = tmp_lang_file
    assert ul.get_preferred_lang(request_override='ta') == 'ta'


def test_get_preferred_lang_file_read(tmp_lang_file, tmp_path):
    fake, ul = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "hi"}', encoding='utf-8')
    ul._cache = {'value': None, 'mtime': 0}  # invalidate
    assert ul.get_preferred_lang() == 'hi'


def test_get_preferred_lang_env_fallback(tmp_lang_file, monkeypatch):
    _, ul = tmp_lang_file
    monkeypatch.setenv('HART_USER_LANGUAGE', 'ja')
    ul._cache = {'value': None, 'mtime': 0}
    # file doesn't exist, env takes over
    assert ul.get_preferred_lang() == 'ja'


def test_get_preferred_lang_defaults_to_en(tmp_lang_file):
    _, ul = tmp_lang_file
    assert ul.get_preferred_lang() == 'en'


def test_set_preferred_lang_writes_file(tmp_lang_file):
    fake, ul = tmp_lang_file
    assert ul.set_preferred_lang('ta')
    assert fake.exists()
    assert json.loads(fake.read_text(encoding='utf-8'))['language'] == 'ta'


def test_set_preferred_lang_persists_english(tmp_lang_file):
    """Regression: previous code had `!= 'en'` guard skipping English.
    Now writing 'en' MUST persist (covers 'ta' → 'en' switch)."""
    fake, ul = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "ta"}', encoding='utf-8')
    ul._cache = {'value': None, 'mtime': 0}
    assert ul.set_preferred_lang('en')
    assert json.loads(fake.read_text(encoding='utf-8'))['language'] == 'en'


def test_set_preferred_lang_idempotent_read_first(tmp_lang_file):
    fake, ul = tmp_lang_file
    ul.set_preferred_lang('ta')
    mtime1 = fake.stat().st_mtime_ns
    import time; time.sleep(0.01)
    ul.set_preferred_lang('ta')  # same value
    mtime2 = fake.stat().st_mtime_ns
    assert mtime1 == mtime2, "Writing identical value should skip disk I/O"


def test_set_preferred_lang_rejects_invalid(tmp_lang_file):
    _, ul = tmp_lang_file
    assert not ul.set_preferred_lang('xyz')
    assert not ul.set_preferred_lang('')
    assert not ul.set_preferred_lang(None)


def test_on_lang_change_fires_on_transition(tmp_lang_file):
    _, ul = tmp_lang_file
    calls = []
    ul.on_lang_change(lambda old, new: calls.append((old, new)))
    ul.set_preferred_lang('ta')  # None → ta
    assert calls == [(None, 'ta')] or calls == [('en', 'ta')] or len(calls) == 1


def test_on_lang_change_no_op_on_same_value(tmp_lang_file):
    _, ul = tmp_lang_file
    ul.set_preferred_lang('ta')
    calls = []
    ul.on_lang_change(lambda old, new: calls.append((old, new)))
    ul.set_preferred_lang('ta')  # same value
    assert calls == []


def test_atomic_write_leaves_original_on_enospc(tmp_lang_file):
    """Simulate ENOSPC mid-write.  The original file must NOT be
    truncated — atomic tmp+replace is the whole point."""
    fake, ul = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "en"}', encoding='utf-8')
    ul._cache = {'value': None, 'mtime': 0}

    real_open = open

    def failing_open(path, *a, **kw):
        if str(path).endswith('.tmp') and 'w' in (a[0] if a else kw.get('mode', '')):
            raise OSError(28, 'No space left on device')
        return real_open(path, *a, **kw)

    with patch('builtins.open', side_effect=failing_open):
        assert not ul.set_preferred_lang('ta')

    # Original still intact
    assert json.loads(fake.read_text(encoding='utf-8'))['language'] == 'en'


def test_cache_invalidates_on_mtime_change(tmp_lang_file):
    fake, ul = tmp_lang_file
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text('{"language": "en"}', encoding='utf-8')
    ul._cache = {'value': None, 'mtime': 0}
    assert ul.get_preferred_lang() == 'en'
    # External process writes the file
    import time; time.sleep(0.01)
    fake.write_text('{"language": "ta"}', encoding='utf-8')
    os.utime(fake, (fake.stat().st_atime + 1, fake.stat().st_mtime + 1))
    assert ul.get_preferred_lang() == 'ta'
