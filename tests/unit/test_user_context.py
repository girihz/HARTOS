"""Tests for core/user_context.py — the canonical resolver that
replaced three drifted copies of get_action_user_details.

Covers:
  - TTL cache hit path
  - Hard budget timeout → cheap defaults + background refresh
  - Mode-specific formatting (create vs reuse)
  - Profile .get() safety for guest / local users
  - Regression guard: NO Python-side message classification

Explicitly does NOT cover any chat-intent classifier inside
core/user_context — that responsibility lives in the draft 0.8B
model (speculative_dispatcher.dispatch_draft_first) with its
augmented classifier prompt. An earlier draft of this module had
a `_is_casual_greeting` regex short-circuit; it was reverted on
operator directive because keyword regex hacks are parallel-path
shadow code. The module now has zero classification logic.
"""
import time
import unittest
from unittest.mock import patch, MagicMock

from core.user_context import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_TTL_SECONDS,
    UserContextCache,
    _cheap_defaults,
    _format_action_rich,
    _format_action_simple,
    _format_profile,
    get_user_context,
    get_user_context_cache,
    invalidate_user_context,
)


# ═══════════════════════════════════════════════════════════════════
# Regression guard: no Python-side classifier
# ═══════════════════════════════════════════════════════════════════


class TestNoPythonSideClassifier(unittest.TestCase):
    """core/user_context must not contain any message-content
    classifier — regex, keyword list, or otherwise. Chat intent
    classification is owned by the draft 0.8B model in
    speculative_dispatcher.dispatch_draft_first with its augmented
    classifier prompt.

    If a future change re-adds ``_is_casual_greeting`` / ``_GREETING_RE``
    / ``_GREETING_MAX_LEN`` (or any other per-message Python-side
    classifier), these tests fail and the reviewer catches the
    regression before it ships.
    """

    def test_module_has_no_greeting_classifier_symbols(self):
        import core.user_context as uc
        forbidden = (
            '_is_casual_greeting',
            '_GREETING_RE',
            '_GREETING_MAX_LEN',
            '_classify_query',
            '_classify_message',
        )
        for name in forbidden:
            self.assertFalse(
                hasattr(uc, name),
                f"core.user_context.{name} exists — Python-side "
                f"classification is forbidden here. Use the draft "
                f"0.8B envelope from dispatch_draft_first instead.",
            )

    def test_get_user_context_signature_has_no_query_param(self):
        import inspect
        from core.user_context import get_user_context
        sig = inspect.signature(get_user_context)
        self.assertNotIn(
            'query', sig.parameters,
            "get_user_context must not accept `query` — the presence "
            "of a query parameter implies per-message classification "
            "inside this module, which is exactly the anti-pattern "
            "being blocked.",
        )
        # The allowed parameters are user_id, mode, timeout_budget_s,
        # ttl_s — anything else is a sign of classifier creep.
        allowed = {'user_id', 'mode', 'timeout_budget_s', 'ttl_s'}
        self.assertTrue(
            set(sig.parameters.keys()).issubset(allowed),
            f"get_user_context has unexpected params: "
            f"{set(sig.parameters.keys()) - allowed}",
        )


# ═══════════════════════════════════════════════════════════════════
# Caching layer
# ═══════════════════════════════════════════════════════════════════


class TestUserContextCache(unittest.TestCase):
    """UserContextCache wraps the shared TTLCache with
    per-user + per-mode keying so create and reuse modes don't
    collide. Behavior must match that contract exactly."""

    def test_separate_keys_per_mode(self):
        cache = UserContextCache()
        cache.set(101, 'create', ('create_details', 'create_actions'))
        cache.set(101, 'reuse', ('reuse_details', 'reuse_actions'))

        self.assertEqual(
            cache.get(101, 'create'),
            ('create_details', 'create_actions'),
        )
        self.assertEqual(
            cache.get(101, 'reuse'),
            ('reuse_details', 'reuse_actions'),
        )

    def test_get_miss_returns_none(self):
        cache = UserContextCache()
        self.assertIsNone(cache.get(404, 'reuse'))

    def test_invalidate_single_mode(self):
        cache = UserContextCache()
        cache.set(1, 'create', ('a', 'b'))
        cache.set(1, 'reuse', ('c', 'd'))
        cache.invalidate(1, 'create')
        self.assertIsNone(cache.get(1, 'create'))
        self.assertEqual(cache.get(1, 'reuse'), ('c', 'd'))

    def test_invalidate_all_modes(self):
        cache = UserContextCache()
        cache.set(1, 'create', ('a', 'b'))
        cache.set(1, 'reuse', ('c', 'd'))
        cache.invalidate(1)
        self.assertIsNone(cache.get(1, 'create'))
        self.assertIsNone(cache.get(1, 'reuse'))

    def test_singleton_accessor(self):
        c1 = get_user_context_cache()
        c2 = get_user_context_cache()
        self.assertIs(c1, c2)


# ═══════════════════════════════════════════════════════════════════
# Formatting layer — mode-specific behavior
# ═══════════════════════════════════════════════════════════════════


class TestFormatters(unittest.TestCase):
    """_format_action_simple (create) vs _format_action_rich (reuse)
    must produce outputs that match their historical callers so
    agent prompts don't subtly shift."""

    _SAMPLE = [
        {
            'action': 'opened_file',
            'zeroshot_label': 'Tool Use',
            'gpt3_label': '',
            'created_date': '2026-04-11T10:00:00Z',
        },
        {
            'action': 'search_code',
            'zeroshot_label': 'Tool Use',
            'gpt3_label': '',
            'created_date': '2026-04-11T10:05:00Z',
        },
        {
            'action': 'Topic Confirmation',  # in _UNWANTED_ACTIONS
            'zeroshot_label': 'Probe',
            'gpt3_label': '',
            'created_date': '2026-04-11T10:06:00Z',
        },
    ]

    def test_simple_excludes_unwanted(self):
        out = _format_action_simple(self._SAMPLE)
        self.assertIn('opened_file', out)
        self.assertIn('search_code', out)
        self.assertNotIn('Topic Confirmation', out)

    def test_simple_empty_returns_placeholder(self):
        out = _format_action_simple([])
        self.assertEqual(out, 'user has not performed any actions yet.')

    def test_rich_excludes_unwanted(self):
        out = _format_action_rich(self._SAMPLE)
        self.assertIn('opened_file', out)
        self.assertNotIn('Topic Confirmation', out)

    def test_rich_includes_datetime_hint(self):
        """The rich formatter appends a 'Today's datetime in Asia/Kolkata'
        hint so the LLM can answer 'what time is it' questions."""
        out = _format_action_rich(self._SAMPLE)
        self.assertIn('<PREVIOUS_USER_ACTION_END>', out)
        self.assertIn("Today's datetime", out)

    def test_rich_empty_still_has_datetime_hint(self):
        out = _format_action_rich([])
        self.assertIn("Today's datetime", out)
        self.assertIn('user has not performed any actions yet', out)


class TestProfileFormatter(unittest.TestCase):
    """_format_profile must handle guest users with no cloud profile
    (the 2026-04 privacy-first fix) without raising KeyError."""

    def test_full_profile(self):
        data = {
            'name': 'Ada',
            'gender': 'female',
            'preferred_language': 'en',
            'dob': '1815-12-10',
            'english_proficiency': 'native',
            'created_date': '2020-01-01',
            'standard': 'expert',
            'who_pays_for_course': 'self',
        }
        out = _format_profile(data, verbose=True)
        self.assertIn('Ada', out)
        self.assertIn('female', out)
        self.assertIn('native', out)

    def test_guest_profile_missing_fields(self):
        """Guest user with only a username should NOT crash."""
        data = {'username': 'anon42'}
        out = _format_profile(data, verbose=True)
        self.assertIn('anon42', out)
        self.assertIn('not specified', out)

    def test_empty_profile(self):
        """An empty dict is indistinguishable from 'no profile at all',
        so we return the 'no user details' message — don't fabricate
        a fake 'User' placeholder that a downstream LLM might greet
        by that exact name."""
        out = _format_profile({}, verbose=True)
        self.assertEqual(out, 'No user details available.')

    def test_none_profile(self):
        out = _format_profile(None, verbose=True)
        self.assertEqual(out, 'No user details available.')

    def test_simple_mode_shorter(self):
        data = {'name': 'Ada', 'gender': 'female'}
        verbose = _format_profile(data, verbose=True)
        simple = _format_profile(data, verbose=False)
        self.assertLess(len(simple), len(verbose))


# ═══════════════════════════════════════════════════════════════════
# Orchestration — the end-to-end contract
# ═══════════════════════════════════════════════════════════════════


class TestGetUserContextOrchestration(unittest.TestCase):
    """get_user_context is the only function external callers touch.
    Its contract (post Python-classifier removal):

      1. Cache hit → cached tuple, ZERO HTTP.
      2. Cache miss + fast backend → fresh fetch + cache populated.
      3. Cache miss + slow backend → cheap defaults within budget.

    No message-content branching. Every call path above is identical
    regardless of the user's query — the speed wins are cache + budget,
    not classification.
    """

    def setUp(self):
        # Clean singleton cache between tests so state doesn't leak.
        cache = get_user_context_cache()
        cache._cache._data.clear()
        cache._cache._timestamps.clear()

    def test_fresh_fetch_calls_both_http_functions(self):
        """With an empty cache and a healthy backend, BOTH
        _fetch_actions_raw and _fetch_profile_raw fire exactly once
        and the formatted profile surfaces to the caller."""
        with patch('core.user_context._fetch_actions_raw',
                   return_value=[]) as mock_actions, \
             patch('core.user_context._fetch_profile_raw',
                   return_value={'name': 'Ada'}) as mock_profile:
            details, actions = get_user_context(
                user_id=54321, mode='reuse',
            )
        mock_actions.assert_called()
        mock_profile.assert_called()
        self.assertIn('Ada', details)

    def test_cache_hit_skips_http(self):
        """Second call for the same user within the TTL never re-fetches."""
        with patch('core.user_context._fetch_actions_raw',
                   return_value=[]) as mock_actions, \
             patch('core.user_context._fetch_profile_raw',
                   return_value={'name': 'Cached'}) as mock_profile:
            get_user_context(user_id=77, mode='reuse')
            # Small sleep to let the future return.
            time.sleep(0.1)
            mock_actions.reset_mock()
            mock_profile.reset_mock()
            # Second call — should hit cache.
            details2, _ = get_user_context(user_id=77, mode='reuse')
        self.assertIn('Cached', details2)
        mock_actions.assert_not_called()
        mock_profile.assert_not_called()

    def test_budget_timeout_returns_defaults(self):
        """If the backend stalls, the hot path returns defaults
        within the budget — not 33.8s later."""
        def _slow_fetch(*args, **kwargs):
            time.sleep(5.0)  # Far past the budget
            return []

        start = time.monotonic()
        with patch('core.user_context._fetch_actions_raw',
                   side_effect=_slow_fetch), \
             patch('core.user_context._fetch_profile_raw',
                   side_effect=_slow_fetch):
            details, actions = get_user_context(
                user_id=999, mode='reuse',
                timeout_budget_s=0.5,  # Tight budget for the test
            )
        elapsed = time.monotonic() - start
        # Must return within ~budget + small overhead.
        self.assertLess(elapsed, 1.5,
                        f"hot path blocked {elapsed:.2f}s beyond budget")
        # Defaults surface for the user.
        self.assertIn('No user details', details)

    def test_create_vs_reuse_mode_different_output(self):
        """mode='create' gets the simple formatter, mode='reuse' gets
        the rich one with datetime hint."""
        with patch('core.user_context._fetch_actions_raw',
                   return_value=[]), \
             patch('core.user_context._fetch_profile_raw',
                   return_value={'name': 'Same'}):
            _, create_actions = get_user_context(user_id=1, mode='create')
            _, reuse_actions = get_user_context(user_id=2, mode='reuse')
        self.assertNotIn('<PREVIOUS_USER_ACTION_END>', create_actions)
        self.assertIn('<PREVIOUS_USER_ACTION_END>', reuse_actions)

    def test_invalidate_forces_refetch(self):
        """invalidate_user_context drops cache so the next call re-fetches."""
        with patch('core.user_context._fetch_actions_raw',
                   return_value=[]), \
             patch('core.user_context._fetch_profile_raw',
                   return_value={'name': 'First'}):
            get_user_context(user_id=7, mode='reuse')
            time.sleep(0.1)
        invalidate_user_context(7)
        with patch('core.user_context._fetch_actions_raw',
                   return_value=[]) as mock_actions2, \
             patch('core.user_context._fetch_profile_raw',
                   return_value={'name': 'Second'}):
            details, _ = get_user_context(user_id=7, mode='reuse')
            time.sleep(0.1)
        mock_actions2.assert_called()


class TestCheapDefaults(unittest.TestCase):
    """_cheap_defaults is the zero-HTTP safety net for budget-blown
    fetches with an empty cache. No relation to classification."""

    def test_reuse_defaults_include_datetime(self):
        details, actions = _cheap_defaults('reuse')
        self.assertIn("Today's datetime", actions)

    def test_create_defaults_simpler(self):
        _, actions = _cheap_defaults('create')
        self.assertNotIn("Today's datetime", actions)


if __name__ == '__main__':
    unittest.main()
