"""Public Flask endpoints for the Hive Contest.

Surface:
  GET  /api/hive/contest/info              — rules, tracks, onramp
  GET  /api/hive/contest/leaderboard       — ranked entries
  GET  /api/hive/contest/claude-code.mcp   — paste-ready MCP snippet
  POST /api/hive/contest/join              — idempotent registration

All endpoints are public except POST /join which needs an authenticated
user (the @require_auth decorator matches the pattern used by
api_audit.py — no new auth mechanism, no parallel path)."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, g

from integrations.agent_engine.hive_contest import (
    ContestTrack,
    claude_code_mcp_snippet,
    get_contest_info,
    get_leaderboard,
    register_participant,
)
from integrations.social.models import get_db
from integrations.social.auth import require_auth

logger = logging.getLogger(__name__)

hive_contest_bp = Blueprint(
    'hive_contest', __name__, url_prefix='/api/hive/contest'
)


def _parse_track(raw) -> ContestTrack | None:
    if not raw:
        return None
    try:
        return ContestTrack(raw.lower())
    except ValueError:
        return None


@hive_contest_bp.route('/info', methods=['GET'])
def contest_info():
    """Public: rules, tracks, dates, how-to-join, Claude Code snippet."""
    return jsonify({'data': get_contest_info()})


@hive_contest_bp.route('/leaderboard', methods=['GET'])
def contest_leaderboard():
    """Public: ranked entries.  Query param:
        ?track=digital|embodied|human_wellness   (default: overall)
        ?limit=50                                (max 200)
    """
    track = _parse_track(request.args.get('track'))
    try:
        limit = min(int(request.args.get('limit', 50)), 200)
    except (TypeError, ValueError):
        limit = 50

    db = get_db()
    try:
        rows = get_leaderboard(db, track=track, limit=limit)
        return jsonify({
            'data': rows,
            'meta': {
                'track': track.value if track else 'overall',
                'count': len(rows),
            },
        })
    finally:
        db.close()


@hive_contest_bp.route('/claude-code.mcp', methods=['GET'])
def contest_mcp_snippet():
    """Public paste-ready snippet for Claude Code -> HARTOS MCP.
    Served as text/plain so the user can pipe it straight into
    their settings file:

        curl -s $HOST/api/hive/contest/claude-code.mcp > ~/.config/claude-code/settings.json
    """
    from flask import Response
    return Response(claude_code_mcp_snippet(), mimetype='text/plain')


@hive_contest_bp.route('/join', methods=['POST'])
@require_auth
def contest_join():
    """Idempotent: register the authenticated user for the contest.

    Body:
      { "track": "digital" | "embodied" | "human_wellness",
        "github": "optional-handle",
        "email": "optional" }
    """
    body = request.get_json(silent=True) or {}
    track = _parse_track(body.get('track')) or ContestTrack.DIGITAL
    github = (body.get('github') or '').strip() or None
    email = (body.get('email') or '').strip() or None

    user_id = getattr(g.user, 'id', None)
    if not user_id:
        return jsonify({'error': 'auth required'}), 401

    db = get_db()
    try:
        result = register_participant(
            db, user_id=user_id, track=track,
            github_handle=github, email=email,
        )
        try:
            db.commit()
        except Exception as exc:  # pragma: no cover
            db.rollback()
            logger.debug(f'commit failed: {exc}')
        return jsonify({'data': result})
    finally:
        db.close()
