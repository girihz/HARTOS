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
