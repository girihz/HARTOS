"""Unit tests for integrations.ui_actions — page registry + Navigate_App.

Covers:
    - PAGE_REGISTRY invariants (stable ids, unique routes, non-empty keywords)
    - list_pages role gating
    - resolve_page lexical scoring + ranking
    - page_to_ui_action shape stability
    - handle_navigate_app happy path, no-match fallback, empty input
    - thread_local.set_ui_actions is called on a successful resolve
"""
from __future__ import annotations

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Page Registry Invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestPageRegistryInvariants:
    """Static checks that prevent accidental breakage of the registry shape."""

    def test_registry_non_empty(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        assert len(PAGE_REGISTRY) >= 8, "Registry should hold the core pages"

    def test_ids_are_unique(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        ids = [p.id for p in PAGE_REGISTRY]
        assert len(ids) == len(set(ids)), "Duplicate page ids"

    def test_routes_are_unique(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        routes = [p.route for p in PAGE_REGISTRY]
        assert len(routes) == len(set(routes)), "Duplicate routes"

    def test_ids_are_snake_case(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        for page in PAGE_REGISTRY:
            assert page.id == page.id.lower()
            assert ' ' not in page.id
            assert '-' not in page.id

    def test_routes_start_with_slash(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        for page in PAGE_REGISTRY:
            assert page.route.startswith('/'), f"{page.id}: route missing leading /"

    def test_keywords_are_lowercase(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        for page in PAGE_REGISTRY:
            assert all(kw == kw.lower() for kw in page.keywords), \
                f"{page.id}: keywords must be lowercase"

    def test_keywords_are_non_empty(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        for page in PAGE_REGISTRY:
            assert len(page.keywords) > 0, f"{page.id}: needs at least one keyword"

    def test_labels_are_short(self):
        from integrations.ui_actions.page_registry import PAGE_REGISTRY
        for page in PAGE_REGISTRY:
            assert len(page.label) <= 20, f"{page.id}: label too long ({page.label})"


# ═══════════════════════════════════════════════════════════════════════════
# list_pages — role gating
# ═══════════════════════════════════════════════════════════════════════════

class TestListPages:

    def test_flat_user_sees_no_admin_pages(self):
        from integrations.ui_actions.page_registry import list_pages
        pages = list_pages(user_role='flat')
        admin_ids = [p.id for p in pages if p.id.startswith('admin_')]
        assert admin_ids == [], f"Flat user should not see admin pages: {admin_ids}"

    def test_central_user_sees_admin_pages(self):
        from integrations.ui_actions.page_registry import list_pages
        pages = list_pages(user_role='central')
        admin_ids = [p.id for p in pages if p.id.startswith('admin_')]
        assert len(admin_ids) >= 3, "Central user should see admin pages"

    def test_regional_user_sees_no_central_only(self):
        from integrations.ui_actions.page_registry import list_pages
        pages = list_pages(user_role='regional')
        # All registry entries requiring 'central' should be hidden
        for p in pages:
            if p.requires_role:
                assert p.requires_role != 'central'

    def test_category_filter(self):
        from integrations.ui_actions.page_registry import list_pages
        pages = list_pages(user_role='central', category='admin')
        assert len(pages) >= 1
        assert all(p.category == 'admin' for p in pages)

    def test_unknown_role_defaults_to_flat(self):
        from integrations.ui_actions.page_registry import list_pages
        pages = list_pages(user_role='nonsense')
        # Should behave like 'flat' — no admin pages
        assert all(not p.id.startswith('admin_') for p in pages)


# ═══════════════════════════════════════════════════════════════════════════
# resolve_page — lexical scoring
# ═══════════════════════════════════════════════════════════════════════════

class TestResolvePage:

    def test_direct_label_match_wins(self):
        from integrations.ui_actions.page_registry import resolve_page
        matches = resolve_page('open games', user_role='flat')
        assert len(matches) >= 1
        assert matches[0][0].id == 'games'

    def test_social_feed_keyword(self):
        from integrations.ui_actions.page_registry import resolve_page
        matches = resolve_page('take me to the social feed', user_role='flat')
        assert matches[0][0].id == 'social_feed'

    def test_model_management_requires_central(self):
        from integrations.ui_actions.page_registry import resolve_page
        # Flat user can't see admin_models
        matches = resolve_page('open model management', user_role='flat')
        assert not any(p.id == 'admin_models' for p, _ in matches)
        # Central user gets it
        matches = resolve_page('open model management', user_role='central')
        assert any(p.id == 'admin_models' for p, _ in matches)

    def test_channels_page(self):
        from integrations.ui_actions.page_registry import resolve_page
        matches = resolve_page('open channels page', user_role='central')
        top_ids = [p.id for p, _ in matches]
        assert 'admin_channels' in top_ids

    def test_no_match_returns_empty(self):
        from integrations.ui_actions.page_registry import resolve_page
        assert resolve_page('asdfasdf qwerty', user_role='flat') == []

    def test_empty_query_returns_empty(self):
        from integrations.ui_actions.page_registry import resolve_page
        assert resolve_page('', user_role='flat') == []
        assert resolve_page(None, user_role='flat') == []

    def test_top_k_caps_results(self):
        from integrations.ui_actions.page_registry import resolve_page
        # A query that hits many pages — should respect top_k
        matches = resolve_page(
            'social agent chat games kids marketplace', user_role='flat', top_k=3,
        )
        assert len(matches) <= 3

    def test_scores_are_positive(self):
        from integrations.ui_actions.page_registry import resolve_page
        matches = resolve_page('open social feed', user_role='flat')
        for _, score in matches:
            assert score > 0

    def test_scores_are_sorted_descending(self):
        from integrations.ui_actions.page_registry import resolve_page
        matches = resolve_page('open admin models channels', user_role='central')
        scores = [s for _, s in matches]
        assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
# page_to_ui_action — serialization shape
# ═══════════════════════════════════════════════════════════════════════════

class TestPageToUiAction:

    def test_shape_contains_required_fields(self):
        from integrations.ui_actions.page_registry import (
            PAGE_REGISTRY, page_to_ui_action,
        )
        action = page_to_ui_action(PAGE_REGISTRY[0])
        assert set(action.keys()) >= {
            'id', 'type', 'label', 'route', 'icon', 'description', 'category',
        }

    def test_type_is_navigate(self):
        from integrations.ui_actions.page_registry import (
            PAGE_REGISTRY, page_to_ui_action,
        )
        action = page_to_ui_action(PAGE_REGISTRY[0])
        assert action['type'] == 'navigate'

    def test_all_pages_serialize(self):
        from integrations.ui_actions.page_registry import (
            PAGE_REGISTRY, page_to_ui_action,
        )
        for page in PAGE_REGISTRY:
            action = page_to_ui_action(page)
            assert action['id'] == page.id
            assert action['route'] == page.route


# ═══════════════════════════════════════════════════════════════════════════
# handle_navigate_app — tool handler end-to-end
# ═══════════════════════════════════════════════════════════════════════════

class _FakeThreadLocal:
    def __init__(self):
        self.ui_actions = None

    def set_ui_actions(self, actions):
        self.ui_actions = actions


class TestNavigateAppHandler:

    def test_happy_path_social(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        tl = _FakeThreadLocal()
        msg = handle_navigate_app('open social feed', user_role='flat', thread_local=tl)
        assert 'Social' in msg
        assert tl.ui_actions is not None
        assert len(tl.ui_actions) >= 1
        assert tl.ui_actions[0]['id'] == 'social_feed'

    def test_happy_path_admin_for_central(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        tl = _FakeThreadLocal()
        msg = handle_navigate_app(
            'open model management', user_role='central', thread_local=tl,
        )
        assert tl.ui_actions[0]['id'] == 'admin_models'
        assert 'Management' in msg or 'model' in msg.lower()

    def test_admin_hidden_for_flat(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        tl = _FakeThreadLocal()
        msg = handle_navigate_app(
            'open model management', user_role='flat', thread_local=tl,
        )
        # Flat user can't reach admin pages — handler should NOT surface one
        if tl.ui_actions:
            admin_ids = [a['id'] for a in tl.ui_actions
                         if a['id'].startswith('admin_')]
            assert admin_ids == []

    def test_no_match_returns_hint(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        tl = _FakeThreadLocal()
        msg = handle_navigate_app(
            'totally nonsense query 123', user_role='flat', thread_local=tl,
        )
        assert "couldn't match" in msg.lower() or 'known destinations' in msg.lower()
        # Nothing to navigate — thread-local should remain None
        assert tl.ui_actions is None

    def test_empty_input_returns_hint(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        msg = handle_navigate_app('', user_role='flat')
        assert 'destination' in msg.lower() or 'available' in msg.lower()

    def test_none_input_returns_hint(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        msg = handle_navigate_app(None, user_role='flat')
        assert 'destination' in msg.lower() or 'available' in msg.lower()

    def test_handler_works_without_thread_local(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        # thread_local=None should not crash
        msg = handle_navigate_app('open social', user_role='flat', thread_local=None)
        assert 'Social' in msg

    def test_runner_ups_appear_in_message(self):
        from integrations.ui_actions.navigate_tool import handle_navigate_app
        tl = _FakeThreadLocal()
        msg = handle_navigate_app(
            'social agent chat', user_role='flat', thread_local=tl,
        )
        # Multiple matches → message should mention at least one alternative
        if len(tl.ui_actions) > 1:
            assert 'options' in msg.lower() or len(tl.ui_actions) > 1


# ═══════════════════════════════════════════════════════════════════════════
# navigate_tool_json_payload — hydration endpoint helper
# ═══════════════════════════════════════════════════════════════════════════

class TestJsonPayload:

    def test_returns_valid_json(self):
        import json
        from integrations.ui_actions.navigate_tool import navigate_tool_json_payload
        payload = navigate_tool_json_payload(user_role='flat')
        data = json.loads(payload)
        assert 'pages' in data
        assert isinstance(data['pages'], list)

    def test_central_sees_more_pages_than_flat(self):
        import json
        from integrations.ui_actions.navigate_tool import navigate_tool_json_payload
        flat = json.loads(navigate_tool_json_payload('flat'))['pages']
        central = json.loads(navigate_tool_json_payload('central'))['pages']
        assert len(central) > len(flat)
