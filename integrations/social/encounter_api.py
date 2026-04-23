"""
HevolveSocial - Encounter Icebreaker Blueprint (PR-A alpha skeleton)

Physical-world P2P encounter flow: two nearby Nunba users both set
'discoverable' on Hevolve Android, their phones do autonomous BLE
sighting correlation via close-range RSSI + dwell + compass alignment,
each user swipes like/dislike on the other's diffusion-styled avatar
(NO real photo, NO camera, NO upload), and a mutual-like fires the
encounter_icebreaker_agent to draft a warm opener for user approval.

Full design: Claude-memory/project_encounter_icebreaker.md
Seeded agent goal: integrations.agent_engine.goal_seeding
 .SEED_BOOTSTRAP_GOALS[slug='encounter_icebreaker_agent']

Endpoints all mounted at /api/social/encounter/*  (JWT-auth required):

  POST /discoverable     enable/disable broadcast + TTL + age gate
  GET  /discoverable     current state + remaining TTL + toggle count
  POST /sighting         phone reports a BLE sighting; returns swipe card
  POST /swipe            like/dislike decision (signed event)
  GET  /matches          list of MUTUAL matches (one-sided never leaks)
  GET  /map-pins         post-match encounter pins the user kept visible
  POST /icebreaker/approve   send approved draft (agent integration: PR-C)
  POST /icebreaker/decline   reject draft; agent learns from reason

Invariants enforced server-side (the blocking privacy gates):

  1. One-sided likes are write-only from the liker's perspective — no
     endpoint returns them.  The likee never learns they were liked
     unless THEY also swiped like within the match window.
  2. A match row is created only when BOTH sightings are within
     ENCOUNTER_MATCH_WINDOW_SEC of each other AND both swiped 'like'.
  3. Location is captured at match time from the sightings, never
     from the user's current reported location (prevents forged pins).
  4. Discoverable default OFF, auto-expires after ENCOUNTER_DISCOVERABLE
     _TTL_SEC, max ENCOUNTER_DISCOVERABLE_MAX_TOGGLES_24H per day.
  5. 18+ age claim required at toggle time.
  6. All pubkeys are rotating (scheme rotates every
     ENCOUNTER_PUBKEY_ROTATION_SEC on the phone); server stores only
     the rotating value, never the user's master identity.

STORAGE (this PR-A alpha):
  In-process dicts + threading.RLock.  NOT DB-backed yet — that lands
  in PR-A beta along with the migration v38.  Persistence is
  process-lifetime only; suitable for dev + CI smoke tests, NOT for
  production regional/central nodes.  The shape of the in-memory
  records matches the planned schema 1:1 so the PR-A beta swap is
  mechanical.

  The ENCOUNTER_STORE sentinel is exposed at module-level so tests
  can reach in and assert state without going through HTTP.
"""
from __future__ import annotations

import logging
import secrets
import threading
import time
from typing import Any, Optional

from flask import Blueprint, g, jsonify, request

from core.constants import (
    ENCOUNTER_DISCOVERABLE_MAX_TOGGLES_24H,
    ENCOUNTER_DISCOVERABLE_TTL_SEC,
    ENCOUNTER_DRAFT_MAX_CHARS,
    ENCOUNTER_MATCH_WINDOW_SEC,
    ENCOUNTER_SIGHTING_EXPIRES_SEC,
    ENCOUNTER_TOPIC_ICEBREAKER,
    ENCOUNTER_TOPIC_MATCH,
    ENCOUNTER_TOPIC_SIGHTING,
    ENCOUNTER_TOPIC_SWIPE,
)
from .auth import require_auth

logger = logging.getLogger('hevolve_social')

encounter_bp = Blueprint('encounter', __name__, url_prefix='/api/social')


# ──────────────────────────────────────────────────────────────────────
# ENCOUNTER_STORE — in-memory skeleton state (PR-A alpha only).
# Replaced by DB models in PR-A beta (migration v38).
#
# Shape matches the target SQLAlchemy model shape 1:1 so the swap in
# PR-A beta is a drop-in replacement, not a refactor.
# ──────────────────────────────────────────────────────────────────────
class _EncounterStore:
    """Process-lifetime encounter state.  RLock-guarded.

    Not thread-stress-tested; the real concurrency story is the DB
    in PR-A beta.  This is enough for dev + single-worker pytest.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # user_id -> dict(enabled, enabled_at, expires_at,
        #                 age_claim_18, face_visible, avatar_style,
        #                 vibe_tags, toggle_count_24h, last_toggle_at)
        self.discoverable: dict[int, dict[str, Any]] = {}
        # sighting_id -> dict(owner_user_id, peer_pubkey, rssi_peak,
        #                     dwell_sec, lat, lng, sighted_at,
        #                     swipe_decision, expires_at)
        self.sightings: dict[str, dict[str, Any]] = {}
        # match_id -> dict(user_a, user_b, lat, lng, matched_at,
        #                  icebreaker_a_status, icebreaker_b_status,
        #                  map_pin_visible)
        self.matches: dict[str, dict[str, Any]] = {}
        # rotating_pubkey (hex) -> user_id  (reverse index; rotated
        # entries expire after ENCOUNTER_PUBKEY_ROTATION_SEC)
        self.pubkey_to_user: dict[str, tuple[int, float]] = {}

    def clear(self) -> None:
        """Test-only: reset the store."""
        with self._lock:
            self.discoverable.clear()
            self.sightings.clear()
            self.matches.clear()
            self.pubkey_to_user.clear()


ENCOUNTER_STORE = _EncounterStore()


# ──────────────────────────────────────────────────────────────────────
# Tiny helpers mirroring the rest of the social blueprints' style so
# response shapes are consistent across /api/social/*.
# ──────────────────────────────────────────────────────────────────────
def _ok(data: Any = None, meta: Any = None, status: int = 200):
    r: dict[str, Any] = {'success': True}
    if data is not None:
        r['data'] = data
    if meta is not None:
        r['meta'] = meta
    return jsonify(r), status


def _err(msg: str, status: int = 400):
    return jsonify({'success': False, 'error': msg}), status


def _json() -> dict[str, Any]:
    return request.get_json(force=True, silent=True) or {}


def _now() -> float:
    return time.time()


def _user_id() -> Optional[int]:
    """Resolve current user id from auth context populated by
    require_auth (sets g.user + g.user_id).  Returns None if unauth.

    User ids in HevolveSocial are VARCHAR(64) in some paths (see
    agent_bridge) and INT in others; we coerce to a consistent string
    hash key for match-row pairing so ordering (user_a, user_b) is
    lexicographic-stable regardless of backing type.
    """
    user = getattr(g, 'user', None)
    if user is None:
        return None
    uid = getattr(user, 'id', None)
    if uid is None and isinstance(user, dict):
        uid = user.get('id')
    if uid is None:
        return None
    # HevolveSocial User.id is a string on some deployments (VARCHAR
    # primary key to allow UUID agents).  Fall back to g.user_id
    # (already stringified by require_auth) if the raw value isn't
    # coerceable to int.
    try:
        return int(uid)
    except (TypeError, ValueError):
        # Hash-stable int for pairing.  Not cryptographic — just need
        # a deterministic ordering so (A,B) and (B,A) collapse.
        return hash(str(uid)) & 0x7FFFFFFF


def _new_id(prefix: str) -> str:
    """16-byte urlsafe id.  Not a secret — used as a handle only."""
    return f"{prefix}_{secrets.token_urlsafe(12)}"


# ──────────────────────────────────────────────────────────────────────
# /discoverable — toggle the BLE broadcast + age + TTL state.
# ──────────────────────────────────────────────────────────────────────
@encounter_bp.route('/encounter/discoverable', methods=['GET'])
@require_auth
def get_discoverable():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    with ENCOUNTER_STORE._lock:
        state = ENCOUNTER_STORE.discoverable.get(uid)
    if not state:
        return _ok({
            'enabled': False,
            'expires_at': None,
            'remaining_sec': 0,
            'toggle_count_24h': 0,
            'age_claim_18': False,
            'face_visible': False,
            'avatar_style': 'studio_ghibli',
            'vibe_tags': [],
        })
    now = _now()
    remaining = max(0, int((state.get('expires_at') or now) - now))
    still_on = bool(state.get('enabled')) and remaining > 0
    return _ok({
        'enabled': still_on,
        'expires_at': state.get('expires_at'),
        'remaining_sec': remaining,
        'toggle_count_24h': state.get('toggle_count_24h', 0),
        'age_claim_18': state.get('age_claim_18', False),
        'face_visible': state.get('face_visible', False),
        'avatar_style': state.get('avatar_style', 'studio_ghibli'),
        'vibe_tags': state.get('vibe_tags', []),
    })


@encounter_bp.route('/encounter/discoverable', methods=['POST'])
@require_auth
def set_discoverable():
    """Turn discoverable on/off.  Server-side invariants:
      * 18+ age claim required to enable
      * TTL capped at ENCOUNTER_DISCOVERABLE_TTL_SEC
      * Max ENCOUNTER_DISCOVERABLE_MAX_TOGGLES_24H toggles per 24h
    """
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    body = _json()
    enable = bool(body.get('enabled', False))
    ttl = int(body.get('ttl_sec', ENCOUNTER_DISCOVERABLE_TTL_SEC))
    if ttl <= 0 or ttl > ENCOUNTER_DISCOVERABLE_TTL_SEC:
        ttl = ENCOUNTER_DISCOVERABLE_TTL_SEC
    age_claim = bool(body.get('age_claim_18', False))
    face_visible = bool(body.get('face_visible', False))
    avatar_style = str(body.get('avatar_style', 'studio_ghibli'))[:64]
    vibe_tags = body.get('vibe_tags', []) or []
    if not isinstance(vibe_tags, list):
        return _err('vibe_tags must be a list of strings')
    vibe_tags = [str(t)[:40] for t in vibe_tags[:10]]

    now = _now()
    with ENCOUNTER_STORE._lock:
        state = ENCOUNTER_STORE.discoverable.setdefault(uid, {
            'enabled': False,
            'enabled_at': None,
            'expires_at': None,
            'age_claim_18': False,
            'face_visible': False,
            'avatar_style': 'studio_ghibli',
            'vibe_tags': [],
            'toggle_count_24h': 0,
            'last_toggle_at': 0.0,
            'toggle_window_start': now,
        })
        # 24h sliding window
        if now - state.get('toggle_window_start', now) > 24 * 3600:
            state['toggle_window_start'] = now
            state['toggle_count_24h'] = 0
        if state['toggle_count_24h'] >= ENCOUNTER_DISCOVERABLE_MAX_TOGGLES_24H:
            return _err(
                f'toggle limit reached '
                f'({ENCOUNTER_DISCOVERABLE_MAX_TOGGLES_24H} per 24h)',
                429,
            )
        if enable and not age_claim:
            return _err('age_claim_18 must be true to enable discoverable', 403)

        state['enabled'] = enable
        state['enabled_at'] = now if enable else state.get('enabled_at')
        state['expires_at'] = (now + ttl) if enable else None
        state['age_claim_18'] = age_claim
        state['face_visible'] = face_visible
        state['avatar_style'] = avatar_style
        state['vibe_tags'] = vibe_tags
        state['toggle_count_24h'] += 1
        state['last_toggle_at'] = now

    return _ok({
        'enabled': enable,
        'expires_at': state['expires_at'],
        'remaining_sec': ttl if enable else 0,
    })


# ──────────────────────────────────────────────────────────────────────
# /sighting — phone reports a BLE sighting.  Returns swipe-card payload
# if the peer is still discoverable and opted in; returns 404 otherwise
# (the likee's phone would simply never produce a card for a non-
# discoverable peer — no leak surface).
# ──────────────────────────────────────────────────────────────────────
@encounter_bp.route('/encounter/sighting', methods=['POST'])
@require_auth
def report_sighting():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    body = _json()
    peer_pubkey = str(body.get('peer_pubkey', '')).strip().lower()
    rssi_peak = int(body.get('rssi_peak', 0))
    dwell_sec = int(body.get('dwell_sec', 0))
    lat = body.get('lat')
    lng = body.get('lng')
    if not peer_pubkey or len(peer_pubkey) < 16:
        return _err('peer_pubkey required (hex, >=16 chars)')

    now = _now()
    # Look up peer user_id from rotating-pubkey index.  If the peer
    # never registered their current pubkey (not discoverable, or
    # pubkey expired) → 404 with a neutral message.  No information
    # is leaked about which case it is.
    with ENCOUNTER_STORE._lock:
        peer_entry = ENCOUNTER_STORE.pubkey_to_user.get(peer_pubkey)
        if not peer_entry:
            return _err('peer not discoverable', 404)
        peer_uid, registered_at = peer_entry
        if peer_uid == uid:
            return _err('self-sighting rejected')
        peer_state = ENCOUNTER_STORE.discoverable.get(peer_uid) or {}
        peer_exp = peer_state.get('expires_at') or 0
        if not peer_state.get('enabled') or peer_exp < now:
            return _err('peer not discoverable', 404)

        sighting_id = _new_id('sight')
        sighting = {
            'id': sighting_id,
            'owner_user_id': uid,
            'peer_user_id': peer_uid,        # resolved, for internal use
            'peer_pubkey': peer_pubkey,
            'rssi_peak': rssi_peak,
            'dwell_sec': dwell_sec,
            'lat': lat,
            'lng': lng,
            'sighted_at': now,
            'swipe_decision': 'pending',
            'expires_at': now + ENCOUNTER_SIGHTING_EXPIRES_SEC,
        }
        ENCOUNTER_STORE.sightings[sighting_id] = sighting

    # Swipe-card payload: only what the liker needs to decide.
    return _ok({
        'sighting_id': sighting_id,
        'peer_anon_id': peer_pubkey[:12],   # shown as handle on card
        'avatar_style': peer_state.get('avatar_style', 'studio_ghibli'),
        'vibe_tags': peer_state.get('vibe_tags', []),
        'face_visible': peer_state.get('face_visible', False),
        'expires_at': sighting['expires_at'],
    })


# ──────────────────────────────────────────────────────────────────────
# /swipe — like/dislike decision.  Server checks for a mutual like
# within ENCOUNTER_MATCH_WINDOW_SEC and, if found, creates an
# encounter_match row.  ONE-SIDED LIKES NEVER LEAK — they live only on
# the liker's sighting row as swipe_decision='like' with no peer-side
# visibility.
# ──────────────────────────────────────────────────────────────────────
@encounter_bp.route('/encounter/swipe', methods=['POST'])
@require_auth
def swipe():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    body = _json()
    sighting_id = str(body.get('sighting_id', ''))
    decision = str(body.get('decision', '')).lower()
    if decision not in {'like', 'dislike'}:
        return _err("decision must be 'like' or 'dislike'")
    if not sighting_id:
        return _err('sighting_id required')

    now = _now()
    matched = None
    with ENCOUNTER_STORE._lock:
        sighting = ENCOUNTER_STORE.sightings.get(sighting_id)
        if not sighting or sighting['owner_user_id'] != uid:
            return _err('sighting not found', 404)
        if sighting.get('expires_at', 0) < now:
            return _err('sighting expired', 410)
        if sighting['swipe_decision'] != 'pending':
            return _err('already swiped', 409)
        sighting['swipe_decision'] = decision

        if decision == 'like':
            # Check for reciprocal like within match window.
            peer_uid = sighting['peer_user_id']
            for other_id, other in ENCOUNTER_STORE.sightings.items():
                if other_id == sighting_id:
                    continue
                if other['owner_user_id'] != peer_uid:
                    continue
                if other['peer_user_id'] != uid:
                    continue
                if other.get('swipe_decision') != 'like':
                    continue
                if abs(other['sighted_at'] - sighting['sighted_at']) \
                        > ENCOUNTER_MATCH_WINDOW_SEC:
                    continue
                # Mutual match found.  Pin at the midpoint of the two
                # sighting locations (if both reported lat/lng).
                lat = None
                lng = None
                la = sighting.get('lat')
                lb = other.get('lat')
                if la is not None and lb is not None:
                    lat = (la + lb) / 2
                lga = sighting.get('lng')
                lgb = other.get('lng')
                if lga is not None and lgb is not None:
                    lng = (lga + lgb) / 2
                # Canonical ordering of the pair (user_a < user_b) so
                # we never double-create match rows for (A,B) and (B,A).
                a_uid = min(uid, peer_uid)
                b_uid = max(uid, peer_uid)
                match_id = _new_id('match')
                matched = {
                    'id': match_id,
                    'user_a': a_uid,
                    'user_b': b_uid,
                    'lat': lat,
                    'lng': lng,
                    'matched_at': now,
                    'icebreaker_a_status': 'pending',
                    'icebreaker_b_status': 'pending',
                    'map_pin_visible': True,
                }
                ENCOUNTER_STORE.matches[match_id] = matched
                break

    return _ok({
        'sighting_id': sighting_id,
        'decision': decision,
        # Matched flag tells the liker's CLIENT that *something*
        # happened, but the response is SYMMETRIC: even when there's
        # no mutual, this shape is identical aside from 'match_id'.
        # We do NOT include any signal about whether the peer swiped
        # dislike — only the positive-match signal is surfaced, and
        # only to the two matched parties.
        'match_id': matched['id'] if matched else None,
    })


# ──────────────────────────────────────────────────────────────────────
# /matches — list mutual matches the user is part of.
# ──────────────────────────────────────────────────────────────────────
@encounter_bp.route('/encounter/matches', methods=['GET'])
@require_auth
def list_matches():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    with ENCOUNTER_STORE._lock:
        mine = [
            m for m in ENCOUNTER_STORE.matches.values()
            if m['user_a'] == uid or m['user_b'] == uid
        ]
    mine.sort(key=lambda m: m['matched_at'], reverse=True)
    return _ok({'matches': mine, 'count': len(mine)})


@encounter_bp.route('/encounter/map-pins', methods=['GET'])
@require_auth
def map_pins():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    with ENCOUNTER_STORE._lock:
        pins = [
            {
                'match_id': m['id'],
                'lat': m['lat'],
                'lng': m['lng'],
                'matched_at': m['matched_at'],
            }
            for m in ENCOUNTER_STORE.matches.values()
            if (m['user_a'] == uid or m['user_b'] == uid)
            and m.get('map_pin_visible')
            and m.get('lat') is not None
            and m.get('lng') is not None
        ]
    return _ok({'pins': pins, 'count': len(pins)})


# ──────────────────────────────────────────────────────────────────────
# /icebreaker — approve or decline a draft.  The actual agent that
# PRODUCES the draft lives in integrations/agent_engine/ and lands in
# PR-C; until then, these endpoints accept user-supplied draft text so
# the RN/React UI can be integration-tested end-to-end.
# ──────────────────────────────────────────────────────────────────────
@encounter_bp.route('/encounter/icebreaker/approve', methods=['POST'])
@require_auth
def icebreaker_approve():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    body = _json()
    match_id = str(body.get('match_id', ''))
    text_val = str(body.get('text', '')).strip()
    if not match_id:
        return _err('match_id required')
    if not text_val:
        return _err('text required')
    if len(text_val) > ENCOUNTER_DRAFT_MAX_CHARS:
        return _err(
            f'text exceeds {ENCOUNTER_DRAFT_MAX_CHARS} chars', 413,
        )
    with ENCOUNTER_STORE._lock:
        match = ENCOUNTER_STORE.matches.get(match_id)
        if not match or uid not in (match['user_a'], match['user_b']):
            return _err('match not found', 404)
        side = 'a' if match['user_a'] == uid else 'b'
        key = f'icebreaker_{side}_status'
        if match[key] in {'sent', 'declined'}:
            return _err(f'icebreaker already {match[key]}', 409)
        match[key] = 'sent'
        match[f'icebreaker_{side}_text'] = text_val
        match[f'icebreaker_{side}_sent_at'] = _now()
    logger.info(
        'encounter.icebreaker sent side=%s match=%s len=%d',
        side, match_id, len(text_val),
    )
    return _ok({'match_id': match_id, 'status': 'sent'})


@encounter_bp.route('/encounter/icebreaker/decline', methods=['POST'])
@require_auth
def icebreaker_decline():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    body = _json()
    match_id = str(body.get('match_id', ''))
    reason = str(body.get('reason', ''))[:400]
    if not match_id:
        return _err('match_id required')
    with ENCOUNTER_STORE._lock:
        match = ENCOUNTER_STORE.matches.get(match_id)
        if not match or uid not in (match['user_a'], match['user_b']):
            return _err('match not found', 404)
        side = 'a' if match['user_a'] == uid else 'b'
        match[f'icebreaker_{side}_status'] = 'declined'
        match[f'icebreaker_{side}_decline_reason'] = reason
    return _ok({'match_id': match_id, 'status': 'declined'})


# ──────────────────────────────────────────────────────────────────────
# INTERNAL — phone registers its current rotating pubkey after it
# rotates (every ENCOUNTER_PUBKEY_ROTATION_SEC).  This lets the server
# resolve sightings.  Unauthenticated peers of the user can't look up
# the user's pubkey via this endpoint — it's a POST and self-scoped
# to the authenticated user.
# ──────────────────────────────────────────────────────────────────────
@encounter_bp.route('/encounter/register-pubkey', methods=['POST'])
@require_auth
def register_pubkey():
    uid = _user_id()
    if uid is None:
        return _err('unauthenticated', 401)
    body = _json()
    pk = str(body.get('pubkey', '')).strip().lower()
    if not pk or len(pk) < 16 or len(pk) > 128:
        return _err('pubkey hex 16..128 chars')
    now = _now()
    with ENCOUNTER_STORE._lock:
        ENCOUNTER_STORE.pubkey_to_user[pk] = (uid, now)
    return _ok({'registered_at': now})


# ──────────────────────────────────────────────────────────────────────
# Topic constants re-exported for clients that need them (Nunba's
# crossbarWorker + RN subscription manager).  Single-source import.
# ──────────────────────────────────────────────────────────────────────
WAMP_TOPICS = {
    'sighting': ENCOUNTER_TOPIC_SIGHTING,
    'swipe': ENCOUNTER_TOPIC_SWIPE,
    'match': ENCOUNTER_TOPIC_MATCH,
    'icebreaker': ENCOUNTER_TOPIC_ICEBREAKER,
}


@encounter_bp.route('/encounter/topics', methods=['GET'])
@require_auth
def list_topics():
    return _ok({'topics': WAMP_TOPICS})
