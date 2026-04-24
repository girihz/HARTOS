"""Unit tests for the Encounter Icebreaker blueprint (PR-A alpha).

Covers:
  - Encounter WAMP topics constants exist + share namespace
  - Seeded goals (encounter_icebreaker_agent, social_media_curator_agent)
    are present in SEED_BOOTSTRAP_GOALS
  - /encounter/* routes require JWT auth (return 401 without one)
  - discoverable POST requires age_claim_18 to enable
  - discoverable TTL caps at ENCOUNTER_DISCOVERABLE_TTL_SEC
  - discoverable 24h toggle-count limit enforced
  - sighting returns 404 when peer is not discoverable (no leak)
  - sighting rejects self-sighting
  - swipe creates mutual match when both users swipe 'like' in window
  - PRIVACY INVARIANT: one-sided 'like' never surfaces via /matches or
    swipe-response 'match_id' (it must be null)
  - icebreaker approve enforces per-user side + draft length cap
  - icebreaker decline writes reason without sending

Design doc: Claude-memory/project_encounter_icebreaker.md
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'),
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import constants as C


# ══════════════════════════════════════════════════════════════════════
# Constants + invariants
# ══════════════════════════════════════════════════════════════════════


def test_encounter_topics_share_prefix():
    prefix = 'com.hevolve.encounter.'
    assert all(t.startswith(prefix) for t in C.ENCOUNTER_TOPICS)
    assert C.ENCOUNTER_TOPIC_SIGHTING in C.ENCOUNTER_TOPICS
    assert C.ENCOUNTER_TOPIC_SWIPE in C.ENCOUNTER_TOPICS
    assert C.ENCOUNTER_TOPIC_MATCH in C.ENCOUNTER_TOPICS
    assert C.ENCOUNTER_TOPIC_ICEBREAKER in C.ENCOUNTER_TOPICS


def test_sighting_correlation_tunables_sensible():
    # Sighting must require close-range + dwell + facing for autonomous
    # pairing to work.  Loosening any one of these degrades to random-
    # pairs-in-a-crowd.
    assert C.ENCOUNTER_SIGHTING_RSSI_PEAK_DBM <= -50     # < ~2m
    assert C.ENCOUNTER_SIGHTING_MIN_DWELL_SEC >= 2
    assert 0 < C.ENCOUNTER_SIGHTING_COMPASS_TOL_DEG <= 90
    # TTL gate: discoverable must auto-expire within 24h
    assert 0 < C.ENCOUNTER_DISCOVERABLE_TTL_SEC <= 24 * 3600
    assert C.ENCOUNTER_DISCOVERABLE_MAX_TOGGLES_24H >= 2
    # Match window: both sightings must be close in time
    assert 60 <= C.ENCOUNTER_MATCH_WINDOW_SEC <= 30 * 60
    # Draft length cap keeps openers short + auditable
    assert 80 <= C.ENCOUNTER_DRAFT_MAX_CHARS <= 500


# ══════════════════════════════════════════════════════════════════════
# Seeded goals present
# ══════════════════════════════════════════════════════════════════════


def test_encounter_icebreaker_goal_seeded():
    from integrations.agent_engine.goal_seeding import SEED_BOOTSTRAP_GOALS
    slugs = {g['slug'] for g in SEED_BOOTSTRAP_GOALS}
    assert 'encounter_icebreaker_agent' in slugs
    assert 'social_media_curator_agent' in slugs


def test_encounter_icebreaker_goal_has_no_autosend_gate():
    from integrations.agent_engine.goal_seeding import SEED_BOOTSTRAP_GOALS
    g = next(
        g for g in SEED_BOOTSTRAP_GOALS
        if g['slug'] == 'encounter_icebreaker_agent'
    )
    cfg = g['config']
    assert cfg.get('no_autosend') is True, \
        "icebreaker MUST NOT auto-send — user approval required"
    assert cfg.get('camera_consent_required') is False, \
        "encounter is explicitly camera-free per user directive 2026-04-24"
    assert 'no_autosend' in cfg.get('constitutional_gates', [])
    assert 'consent_required' in cfg.get('constitutional_gates', [])
    assert cfg.get('max_draft_length_chars', 0) \
        <= C.ENCOUNTER_DRAFT_MAX_CHARS


def test_social_media_curator_goal_has_no_autosend():
    from integrations.agent_engine.goal_seeding import SEED_BOOTSTRAP_GOALS
    g = next(
        g for g in SEED_BOOTSTRAP_GOALS
        if g['slug'] == 'social_media_curator_agent'
    )
    cfg = g['config']
    assert cfg.get('no_autosend') is True
    assert 'no_autosend' in cfg.get('constitutional_gates', [])


# ══════════════════════════════════════════════════════════════════════
# Flask app fixture with encounter_bp mounted + auth monkey-patched so
# tests don't need a real DB user + JWT pipeline.
# ══════════════════════════════════════════════════════════════════════


_TEST_TOKEN_PREFIX = 'TEST-USER-'  # Authorization: Bearer TEST-USER-<id>


class _MockDB:
    """Minimum duck-type for the db object require_auth pulls from
    _get_user_from_token.  Real require_auth calls .commit() / .close()
    / .is_active / .rollback() — mock them all as no-ops so the
    decorator's db lifecycle doesn't AttributeError.
    """

    is_active = True

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def _fake_get_user_from_token(token):
    """Test-mode replacement for auth._get_user_from_token.

    Accepts 'TEST-USER-<N>' tokens (set by our test client) and returns
    a minimal user + mock db pair.  Patching this internal function —
    instead of swapping the @require_auth decorator by module-reload —
    is robust to test collection order.  The real require_auth keeps
    all its logic (Bearer prefix check, g.user population, db
    lifecycle); we only replace the token → user resolver.
    """
    if not isinstance(token, str) or not token.startswith(_TEST_TOKEN_PREFIX):
        return None, None
    try:
        uid = int(token[len(_TEST_TOKEN_PREFIX):])
    except ValueError:
        return None, None
    return (
        SimpleNamespace(id=uid, is_admin=False, is_moderator=False),
        _MockDB(),
    )


@pytest.fixture
def app(monkeypatch):
    """Minimal Flask app with just the encounter blueprint mounted.

    The real require_auth decorator runs (bearer check, token
    resolution, g.user population, db lifecycle) with
    _get_user_from_token swapped for a test-mode version.  No module
    reloading, no blueprint reconstruction — robust to test
    collection order and parallel execution.
    """
    from flask import Flask

    from integrations.social import auth as _auth_mod
    from integrations.social import encounter_api

    monkeypatch.setattr(
        _auth_mod, '_get_user_from_token', _fake_get_user_from_token,
    )

    encounter_api.ENCOUNTER_STORE.clear()
    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True
    flask_app.register_blueprint(encounter_api.encounter_bp)
    flask_app.encounter_module = encounter_api  # type: ignore[attr-defined]
    yield flask_app
    # Blueprint registration is per-app; garbage-collected with flask_app.
    encounter_api.ENCOUNTER_STORE.clear()


@pytest.fixture
def client(app):
    return app.test_client()


def _as_user(uid: int) -> dict:
    """Auth header that pairs with _fake_get_user_from_token above."""
    return {'Authorization': f'Bearer {_TEST_TOKEN_PREFIX}{uid}'}


# ══════════════════════════════════════════════════════════════════════
# Auth required
# ══════════════════════════════════════════════════════════════════════


def test_all_routes_require_auth(client):
    # Hit every route without the test-user header; all must 401.
    probes = [
        ('GET', '/api/social/encounter/discoverable', {}),
        ('POST', '/api/social/encounter/discoverable',
            {'enabled': False, 'age_claim_18': True}),
        ('POST', '/api/social/encounter/sighting',
            {'peer_pubkey': 'a' * 32, 'rssi_peak': -40, 'dwell_sec': 4}),
        ('POST', '/api/social/encounter/swipe',
            {'sighting_id': 'x', 'decision': 'like'}),
        ('GET', '/api/social/encounter/matches', {}),
        ('GET', '/api/social/encounter/map-pins', {}),
        ('POST', '/api/social/encounter/icebreaker/approve',
            {'match_id': 'x', 'text': 'hi'}),
        ('POST', '/api/social/encounter/icebreaker/decline',
            {'match_id': 'x'}),
        ('POST', '/api/social/encounter/register-pubkey',
            {'pubkey': 'a' * 32}),
        ('GET', '/api/social/encounter/topics', {}),
    ]
    for method, path, body in probes:
        if method == 'GET':
            resp = client.get(path)
        else:
            resp = client.post(path, json=body)
        assert resp.status_code == 401, \
            f'{method} {path} did not 401 (got {resp.status_code})'


# ══════════════════════════════════════════════════════════════════════
# discoverable toggle
# ══════════════════════════════════════════════════════════════════════


def test_discoverable_default_off(client):
    resp = client.get(
        '/api/social/encounter/discoverable', headers=_as_user(10),
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['success']
    assert body['data']['enabled'] is False
    assert body['data']['toggle_count_24h'] == 0


def test_discoverable_requires_age_claim(client):
    resp = client.post(
        '/api/social/encounter/discoverable',
        json={'enabled': True, 'age_claim_18': False},
        headers=_as_user(10),
    )
    assert resp.status_code == 403
    assert 'age_claim_18' in resp.get_json()['error']


def test_discoverable_enable_with_age_claim(client):
    resp = client.post(
        '/api/social/encounter/discoverable',
        json={'enabled': True, 'age_claim_18': True,
              'vibe_tags': ['hiking', 'chai']},
        headers=_as_user(10),
    )
    assert resp.status_code == 200
    data = resp.get_json()['data']
    assert data['enabled'] is True
    assert data['remaining_sec'] > 0
    assert data['remaining_sec'] <= C.ENCOUNTER_DISCOVERABLE_TTL_SEC


def test_discoverable_ttl_clamped(client):
    resp = client.post(
        '/api/social/encounter/discoverable',
        json={'enabled': True, 'age_claim_18': True,
              'ttl_sec': 10 * C.ENCOUNTER_DISCOVERABLE_TTL_SEC},
        headers=_as_user(10),
    )
    data = resp.get_json()['data']
    assert data['remaining_sec'] <= C.ENCOUNTER_DISCOVERABLE_TTL_SEC


def test_discoverable_toggle_limit(client):
    uid = 11
    # Toggle MAX+1 times; the last should 429.
    for i in range(C.ENCOUNTER_DISCOVERABLE_MAX_TOGGLES_24H):
        resp = client.post(
            '/api/social/encounter/discoverable',
            json={'enabled': bool(i % 2), 'age_claim_18': True},
            headers=_as_user(uid),
        )
        assert resp.status_code == 200
    resp = client.post(
        '/api/social/encounter/discoverable',
        json={'enabled': True, 'age_claim_18': True},
        headers=_as_user(uid),
    )
    assert resp.status_code == 429


# ══════════════════════════════════════════════════════════════════════
# sighting → swipe-card
# ══════════════════════════════════════════════════════════════════════


def _make_discoverable(client, uid, tags=None):
    client.post(
        '/api/social/encounter/discoverable',
        json={'enabled': True, 'age_claim_18': True,
              'vibe_tags': tags or ['hiking']},
        headers=_as_user(uid),
    )


def _register_pubkey(client, uid, pk):
    resp = client.post(
        '/api/social/encounter/register-pubkey',
        json={'pubkey': pk},
        headers=_as_user(uid),
    )
    assert resp.status_code == 200


def test_sighting_of_non_discoverable_peer_404s(client):
    pk = 'deadbeef' * 4
    # Peer 20 is NOT discoverable and has not registered pubkey.
    resp = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert resp.status_code == 404
    # Message must be neutral — no leak about why (toggled off, never
    # existed, expired, etc.)
    assert resp.get_json()['error'] == 'peer not discoverable'


def test_sighting_rejects_self(client):
    pk = 'cafebabe' * 4
    _make_discoverable(client, 10)
    _register_pubkey(client, 10, pk)
    resp = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert resp.status_code == 400
    assert 'self-sighting' in resp.get_json()['error']


def test_sighting_returns_swipe_card(client):
    pk = 'feedface' * 4
    _make_discoverable(client, 20, tags=['indie_film'])
    _register_pubkey(client, 20, pk)
    resp = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk, 'rssi_peak': -40, 'dwell_sec': 4,
              'lat': 12.97, 'lng': 77.59},
        headers=_as_user(10),
    )
    assert resp.status_code == 200
    data = resp.get_json()['data']
    assert data['sighting_id'].startswith('sight_')
    assert data['peer_anon_id'] == pk[:12]
    assert 'indie_film' in data['vibe_tags']


# ══════════════════════════════════════════════════════════════════════
# swipe → match (mutual like)
# ══════════════════════════════════════════════════════════════════════


def _setup_mutual_sighting(client):
    """Both users discoverable + register pubkeys + report sightings."""
    pk_a = 'aaaabbbb' * 4
    pk_b = 'ccccdddd' * 4
    _make_discoverable(client, 1)
    _make_discoverable(client, 2)
    _register_pubkey(client, 1, pk_a)
    _register_pubkey(client, 2, pk_b)
    # User 1 sees user 2
    r1 = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk_b, 'rssi_peak': -40, 'dwell_sec': 4,
              'lat': 12.97, 'lng': 77.59},
        headers=_as_user(1),
    )
    # User 2 sees user 1
    r2 = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk_a, 'rssi_peak': -40, 'dwell_sec': 4,
              'lat': 12.97, 'lng': 77.59},
        headers=_as_user(2),
    )
    return (
        r1.get_json()['data']['sighting_id'],
        r2.get_json()['data']['sighting_id'],
    )


def test_mutual_like_creates_match(client):
    s1, s2 = _setup_mutual_sighting(client)
    # User 1 likes first — no match yet.
    r = client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'like'},
        headers=_as_user(1),
    )
    assert r.status_code == 200
    assert r.get_json()['data']['match_id'] is None
    # User 2 likes → mutual match.
    r = client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s2, 'decision': 'like'},
        headers=_as_user(2),
    )
    assert r.status_code == 200
    assert r.get_json()['data']['match_id'] is not None


def test_one_sided_like_never_leaks_to_other_user(client):
    """PRIVACY INVARIANT: if A likes but B dislikes (or never swipes),
    B must see NOTHING about the like.  /matches returns empty for B,
    and B's swipe response for a subsequent dislike must not reveal
    that A had liked first.
    """
    s1, s2 = _setup_mutual_sighting(client)
    # A likes.
    client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'like'},
        headers=_as_user(1),
    )
    # B's /matches is empty.
    r_b = client.get(
        '/api/social/encounter/matches',
        headers=_as_user(2),
    )
    assert r_b.status_code == 200
    assert r_b.get_json()['data']['matches'] == []
    assert r_b.get_json()['data']['count'] == 0

    # B dislikes.  Response must NOT include match_id or any peer-like
    # signal.
    r = client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s2, 'decision': 'dislike'},
        headers=_as_user(2),
    )
    data = r.get_json()['data']
    assert data['match_id'] is None
    # Exhaustive shape check — no field leaks that A liked.
    leaky_keys = {'peer_liked', 'other_decision', 'reciprocal',
                  'liked_by', 'was_liked'}
    assert leaky_keys.isdisjoint(set(data))

    # Even A's /matches is empty (no mutual).
    r_a = client.get(
        '/api/social/encounter/matches',
        headers=_as_user(1),
    )
    assert r_a.get_json()['data']['matches'] == []


def test_swipe_non_owner_404(client):
    s1, s2 = _setup_mutual_sighting(client)
    # User 3 (unrelated) tries to swipe on user 1's sighting.
    r = client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'like'},
        headers=_as_user(3),
    )
    assert r.status_code == 404


def test_swipe_double_409(client):
    s1, _ = _setup_mutual_sighting(client)
    client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'like'},
        headers=_as_user(1),
    )
    r = client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'dislike'},
        headers=_as_user(1),
    )
    assert r.status_code == 409


# ══════════════════════════════════════════════════════════════════════
# icebreaker approve / decline
# ══════════════════════════════════════════════════════════════════════


def _matched_pair(client):
    s1, s2 = _setup_mutual_sighting(client)
    client.post('/api/social/encounter/swipe',
                json={'sighting_id': s1, 'decision': 'like'},
                headers=_as_user(1))
    r = client.post('/api/social/encounter/swipe',
                    json={'sighting_id': s2, 'decision': 'like'},
                    headers=_as_user(2))
    return r.get_json()['data']['match_id']


def test_icebreaker_approve_happy_path(client):
    m = _matched_pair(client)
    r = client.post(
        '/api/social/encounter/icebreaker/approve',
        json={'match_id': m, 'text': 'Hey! Nice to actually be across '
                                      'the table.'},
        headers=_as_user(1),
    )
    assert r.status_code == 200
    assert r.get_json()['data']['status'] == 'sent'


def test_icebreaker_length_cap(client):
    m = _matched_pair(client)
    r = client.post(
        '/api/social/encounter/icebreaker/approve',
        json={'match_id': m, 'text': 'x' * (C.ENCOUNTER_DRAFT_MAX_CHARS + 1)},
        headers=_as_user(1),
    )
    assert r.status_code == 413


def test_icebreaker_non_participant_404(client):
    m = _matched_pair(client)
    r = client.post(
        '/api/social/encounter/icebreaker/approve',
        json={'match_id': m, 'text': 'hi'},
        headers=_as_user(99),
    )
    assert r.status_code == 404


def test_icebreaker_decline_records_reason(client):
    m = _matched_pair(client)
    r = client.post(
        '/api/social/encounter/icebreaker/decline',
        json={'match_id': m, 'reason': 'vibe mismatch'},
        headers=_as_user(1),
    )
    assert r.status_code == 200
    assert r.get_json()['data']['status'] == 'declined'


def test_topics_endpoint_matches_constants(client):
    r = client.get(
        '/api/social/encounter/topics',
        headers=_as_user(1),
    )
    assert r.status_code == 200
    topics = r.get_json()['data']['topics']
    assert topics['sighting'] == C.ENCOUNTER_TOPIC_SIGHTING
    assert topics['swipe'] == C.ENCOUNTER_TOPIC_SWIPE
    assert topics['match'] == C.ENCOUNTER_TOPIC_MATCH
    assert topics['icebreaker'] == C.ENCOUNTER_TOPIC_ICEBREAKER
