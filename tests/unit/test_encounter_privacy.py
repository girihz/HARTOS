"""NFT / privacy adversarial tests for encounter_api (closes #403).

Augments tests/unit/test_encounter_api.py (23 functional tests) with
focused adversarial cases on the privacy + lifecycle invariants the
design doc (project_encounter_icebreaker.md) treats as blocking:

  * Discoverable TTL auto-off — advance time past expires_at, peer
    must no longer be sightable (the stalker-walks-away invariant).
  * Match-window enforcement — sightings outside
    ENCOUNTER_MATCH_WINDOW_SEC must NOT mutual-match even when both
    sides like (the "I swiped yes a week ago" replay invariant).
  * Sighting expiry — once a sighting passes expires_at the swipe
    must 410 (the ephemeral-state-vanishes invariant).
  * Pubkey rotation — when peer rotates their pubkey, the OLD
    pubkey must no longer resolve a sighting (the rotating-handle
    invariant; the whole point of rotation is unlinkability).

Time is controlled via monkeypatching `encounter_api._now_dt` rather
than freezegun (not in HARTOS's dep set).  Each test gets a fresh
in-memory SQLite engine via the existing `app` fixture so DB state
does NOT bleed across cases.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# Re-use the rich `app` / `client` / `_as_user` fixtures from the
# functional suite — same fake-auth, same fresh in-memory engine,
# same _make_discoverable / _register_pubkey helpers.  Keeping a
# single fixture surface (one truth) avoids the parallel-test-fixture
# anti-pattern.
from .test_encounter_api import (  # noqa: F401, E402
    _as_user,
    _make_discoverable,
    _register_pubkey,
    _setup_mutual_sighting,
    app,
    client,
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'),
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import constants as C  # noqa: E402


def _patch_clock(monkeypatch, now_dt):
    """Set encounter_api._now_dt to return a fixed datetime."""
    from integrations.social import encounter_api
    monkeypatch.setattr(encounter_api, '_now_dt', lambda: now_dt)


# ══════════════════════════════════════════════════════════════════════
# Discoverable TTL auto-off
# ══════════════════════════════════════════════════════════════════════


def test_discoverable_auto_off_after_ttl_blocks_sighting(client, monkeypatch):
    """After the TTL elapses, a sighting against the now-expired peer
    must 404 'peer not discoverable' — same neutral message as a peer
    who never enabled in the first place (no leak about why).
    """
    pk = 'aabbccdd' * 4
    t0 = datetime(2026, 4, 25, 12, 0, 0)
    _patch_clock(monkeypatch, t0)

    _make_discoverable(client, 20)
    _register_pubkey(client, 20, pk)

    # Sighting at t0 succeeds.
    r = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert r.status_code == 200

    # Advance past TTL — peer's discoverable expires_at lapsed.
    t_after = t0 + timedelta(
        seconds=C.ENCOUNTER_DISCOVERABLE_TTL_SEC + 60,
    )
    _patch_clock(monkeypatch, t_after)

    r = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert r.status_code == 404
    assert r.get_json()['error'] == 'peer not discoverable'


# ══════════════════════════════════════════════════════════════════════
# Match-window enforcement
# ══════════════════════════════════════════════════════════════════════


def test_sightings_outside_match_window_do_not_match(client, monkeypatch):
    """Two reciprocal 'like' sightings far apart in time must NOT
    create a match — even when both swipe like.  The whole reason
    the window exists is to refuse forced-pairing-via-replay.
    """
    pk_a = 'aaaa1111' * 4
    pk_b = 'bbbb2222' * 4

    t0 = datetime(2026, 4, 25, 12, 0, 0)
    _patch_clock(monkeypatch, t0)
    _make_discoverable(client, 1)
    _make_discoverable(client, 2)
    _register_pubkey(client, 1, pk_a)
    _register_pubkey(client, 2, pk_b)

    # User 1 sees user 2 at t0.
    r1 = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk_b, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(1),
    )
    assert r1.status_code == 200
    s1 = r1.get_json()['data']['sighting_id']

    # Advance well beyond the match window.
    t_far = t0 + timedelta(
        seconds=C.ENCOUNTER_MATCH_WINDOW_SEC + 600,
    )
    _patch_clock(monkeypatch, t_far)

    # User 2 sees user 1 — at a time outside the window.
    r2 = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk_a, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(2),
    )
    assert r2.status_code == 200
    s2 = r2.get_json()['data']['sighting_id']

    # Both swipe like; reciprocal-check window-test must FAIL.
    client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'like'},
        headers=_as_user(1),
    )
    r = client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s2, 'decision': 'like'},
        headers=_as_user(2),
    )
    assert r.status_code == 200
    # No mutual match because sightings are out-of-window.
    assert r.get_json()['data']['match_id'] is None

    # /matches returns nothing for either side.
    for uid in (1, 2):
        m = client.get(
            '/api/social/encounter/matches', headers=_as_user(uid),
        )
        assert m.status_code == 200
        assert m.get_json()['data']['matches'] == []


# ══════════════════════════════════════════════════════════════════════
# Sighting auto-expiry
# ══════════════════════════════════════════════════════════════════════


def test_swipe_on_expired_sighting_410(client, monkeypatch):
    """Sightings auto-expire after ENCOUNTER_SIGHTING_EXPIRES_SEC.
    A swipe attempt on an expired row must 410 Gone (not 200) so the
    client knows the row is past TTL and refuses to gamble on stale
    state.
    """
    pk = 'cafe' * 8
    t0 = datetime(2026, 4, 25, 12, 0, 0)
    _patch_clock(monkeypatch, t0)
    _make_discoverable(client, 20)
    _register_pubkey(client, 20, pk)

    r = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert r.status_code == 200
    sid = r.get_json()['data']['sighting_id']

    # Travel past the sighting's expires_at.
    t_after = t0 + timedelta(
        seconds=C.ENCOUNTER_SIGHTING_EXPIRES_SEC + 60,
    )
    _patch_clock(monkeypatch, t_after)

    r = client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': sid, 'decision': 'like'},
        headers=_as_user(10),
    )
    assert r.status_code == 410
    assert 'expired' in r.get_json()['error'].lower()


# ══════════════════════════════════════════════════════════════════════
# Pubkey rotation unlinkability
# ══════════════════════════════════════════════════════════════════════


def test_old_pubkey_no_longer_resolves_after_rotation(
    client, monkeypatch,
):
    """When a peer rotates their pubkey, the OLD pubkey must no
    longer resolve a sighting — that's the WHOLE POINT of rotation
    (unlinkability across observation windows).  Any test that
    finds the old pubkey still resolving is a privacy regression.
    """
    pk_old = 'old0' * 8
    pk_new = 'new1' * 8
    t0 = datetime(2026, 4, 25, 12, 0, 0)
    _patch_clock(monkeypatch, t0)

    _make_discoverable(client, 20)
    _register_pubkey(client, 20, pk_old)

    # Sighting against pk_old works.
    r = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk_old, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert r.status_code == 200

    # Peer rotates pubkey.
    _register_pubkey(client, 20, pk_new)

    # Sighting against pk_old must now 404 — no DiscoverablePref row
    # has current_pubkey == pk_old anymore.
    r = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk_old, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert r.status_code == 404

    # Sighting against pk_new resolves.
    r = client.post(
        '/api/social/encounter/sighting',
        json={'peer_pubkey': pk_new, 'rssi_peak': -40, 'dwell_sec': 4},
        headers=_as_user(10),
    )
    assert r.status_code == 200


# ══════════════════════════════════════════════════════════════════════
# Cross-user payload isolation on /matches
# ══════════════════════════════════════════════════════════════════════


def test_unrelated_user_matches_endpoint_is_empty(client):
    """A user who never sighted anyone must see /matches return [],
    even if other users elsewhere in the system have matches.  No
    cross-user enumeration via the matches list.
    """
    s1, s2 = _setup_mutual_sighting(client)
    client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'like'},
        headers=_as_user(1),
    )
    client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s2, 'decision': 'like'},
        headers=_as_user(2),
    )

    # User 99 is unrelated.
    r = client.get(
        '/api/social/encounter/matches', headers=_as_user(99),
    )
    assert r.status_code == 200
    body = r.get_json()['data']
    assert body['matches'] == []
    assert body['count'] == 0


# ══════════════════════════════════════════════════════════════════════
# Map-pin visibility gate
# ══════════════════════════════════════════════════════════════════════


def test_map_pins_skip_unmatched_pairs(client):
    """An unmatched sighting (one-sided like) must NOT produce a
    map pin.  Pins are post-match adornment only.
    """
    s1, _ = _setup_mutual_sighting(client)
    client.post(
        '/api/social/encounter/swipe',
        json={'sighting_id': s1, 'decision': 'like'},
        headers=_as_user(1),
    )
    # User 2 never swipes.

    r = client.get(
        '/api/social/encounter/map-pins', headers=_as_user(1),
    )
    assert r.status_code == 200
    body = r.get_json()['data']
    assert body['pins'] == []
    assert body['count'] == 0


# ══════════════════════════════════════════════════════════════════════
# Icebreaker decline-then-approve refusal
# ══════════════════════════════════════════════════════════════════════


def test_icebreaker_approve_after_decline_409(client):
    """Once a side declines an icebreaker, attempting to approve a
    new draft for the same side must 409 — declined state is
    terminal so the agent's learn-from-decline signal isn't
    overwritten by a later edge-case flip.
    """
    from .test_encounter_api import _matched_pair  # noqa: WPS433

    m = _matched_pair(client)
    # Decline first.
    r = client.post(
        '/api/social/encounter/icebreaker/decline',
        json={'match_id': m, 'reason': 'not interested'},
        headers=_as_user(1),
    )
    assert r.status_code == 200
    # Then attempt approve — must 409 (status terminal).
    r = client.post(
        '/api/social/encounter/icebreaker/approve',
        json={'match_id': m, 'text': 'changed my mind'},
        headers=_as_user(1),
    )
    assert r.status_code == 409
