"""navigate_tool — LangChain facade for the page registry.

Exposed to the chat agent as `Navigate_App`. When the user says
'take me to social' / 'open model management' / 'connect whatsapp
on the admin page', this tool resolves the best-matching page from
PAGE_REGISTRY and returns a structured sentinel the Flask /chat
handler intercepts and surfaces to the frontend LiquidActionBar.

The tool ALSO writes the resolved ui_action into thread_local_data
so chatbot_routes.py can attach it to the final response without
having to parse the tool string. The string return value is just
what the LLM sees (human-readable so it weaves a natural reply
like 'opening Model Management for you').

Single source of truth: this tool, page_registry.PAGE_REGISTRY, and
the frontend `pageRegistry.js` all share the same `id` keys. Adding
a new page means editing one entry in page_registry.py and mirroring
the id + route in the frontend file.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from .page_registry import (
    PAGE_REGISTRY,
    page_to_ui_action,
    resolve_page,
    list_pages,
)

logger = logging.getLogger(__name__)


def handle_navigate_app(
    input_text: str,
    user_role: str = 'flat',
    thread_local=None,
) -> str:
    """Resolve a natural-language destination and attach a ui_action.

    Args:
        input_text: the user's destination phrase (e.g. "model management")
        user_role: viewer role for filtering admin-only pages
        thread_local: optional thread-local object with set_ui_actions().
            When present the resolved ui_action list is stored on it so
            the /chat handler can lift it into the response without
            parsing the tool's string return value.

    Returns:
        A human-readable string the LLM can weave into its final reply.
        On no-match, returns a hint listing a few valid destinations so
        the LLM asks for clarification.
    """
    query = (input_text or '').strip()
    if not query:
        visible = list_pages(user_role=user_role)
        names = ', '.join(p.label for p in visible[:6])
        return (
            f"Navigate_App needs a destination. Available pages include: {names}. "
            f"Try calling me again with the page name."
        )

    matches = resolve_page(query, user_role=user_role, top_k=3)
    if not matches:
        visible = list_pages(user_role=user_role)
        names = ', '.join(p.label for p in visible[:6])
        return (
            f"I couldn't match '{query}' to any app page. "
            f"Known destinations include: {names}."
        )

    actions = [page_to_ui_action(page) for page, _score in matches]

    # Best match drives the reply text; secondary matches still surface
    # as chips so the user can pick another option if we guessed wrong.
    best_page, best_score = matches[0]
    logger.info(
        f"Navigate_App resolved '{query}' → {best_page.id} "
        f"(score={best_score}, +{len(matches)-1} runner-ups)"
    )

    if thread_local is not None:
        try:
            thread_local.set_ui_actions(actions)
        except Exception as e:
            logger.debug(f"thread_local.set_ui_actions failed: {e}")

    runner_up = ''
    if len(matches) > 1:
        runner_up = ' Other options: ' + ', '.join(
            f"{p.label}" for p, _ in matches[1:]
        ) + '.'
    return (
        f"Opening {best_page.label} ({best_page.description}).{runner_up}"
    )


def navigate_tool_json_payload(user_role: str = 'flat') -> str:
    """JSON blob the Flask /chat handler can return so the frontend
    renders the full action bar on first load without waiting for the
    user to invoke Navigate_App. Used by the initial LiquidActionBar
    hydration call (GET /ui-actions/pages)."""
    return json.dumps({
        'pages': [page_to_ui_action(p) for p in list_pages(user_role=user_role)],
    })
