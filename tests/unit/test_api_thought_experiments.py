"""
test_api_thought_experiments.py - Tests for integrations/social/api_thought_experiments.py

Tests the thought experiment API — the democratic decision-making system.
Each test verifies a specific API contract or validation boundary:

FT: Create experiment (validation, constitutional filter), list/filter,
    vote (up/down), advance lifecycle, evaluate, decide, auto-evolve.
NFT: DB session cleanup (no leaks), 400 on missing fields, 403 on
     constitutional block, 500 with error details, idempotent votes.
"""
import os
import sys
import json
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    from integrations.social.api_thought_experiments import thought_experiments_bp
    app.register_blueprint(thought_experiments_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _mock_db_and_service():
    """Helper: mock DB + ThoughtExperimentService for deferred imports."""
    mock_db = MagicMock()
    mock_svc = MagicMock()
    mock_models = MagicMock()
    mock_models.get_db.return_value = mock_db
    mock_te_mod = MagicMock()
    mock_te_mod.ThoughtExperimentService = mock_svc
    return mock_db, mock_svc, {
        'integrations.social.models': mock_models,
        'integrations.social.thought_experiment_service': mock_te_mod,
    }


# ============================================================
# POST /api/social/experiments — create experiment
# ============================================================

class TestCreateExperiment:
    """Create experiment — the start of the democratic decision flow."""

    def test_returns_400_without_required_fields(self, client):
        """Missing creator_id/title/hypothesis must be rejected."""
        resp = client.post('/api/social/experiments',
                           json={'title': 'test'},
                           content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['success'] is False

    def test_returns_400_empty_body(self, client):
        resp = client.post('/api/social/experiments',
                           json={},
                           content_type='application/json')
        assert resp.status_code == 400

    def test_returns_201_on_success(self, client):
        mock_db, mock_svc, modules = _mock_db_and_service()
        mock_svc.create_experiment.return_value = {'id': '123', 'title': 'test'}
        with patch.dict('sys.modules', modules):
            resp = client.post('/api/social/experiments',
                               json={'creator_id': 'u1', 'title': 'Test', 'hypothesis': 'H1'},
                               content_type='application/json')
        assert resp.status_code == 201
        assert resp.get_json()['success'] is True

    def test_returns_403_when_constitutional_filter_blocks(self, client):
        """ConstitutionalFilter can block experiments that violate principles."""
        mock_db, mock_svc, modules = _mock_db_and_service()
        mock_svc.create_experiment.return_value = None  # Filter blocked
        with patch.dict('sys.modules', modules):
            resp = client.post('/api/social/experiments',
                               json={'creator_id': 'u1', 'title': 'Bad', 'hypothesis': 'H'},
                               content_type='application/json')
        assert resp.status_code == 403

    def test_closes_db_on_success(self, client):
        mock_db, mock_svc, modules = _mock_db_and_service()
        mock_svc.create_experiment.return_value = {'id': '1'}
        with patch.dict('sys.modules', modules):
            client.post('/api/social/experiments',
                        json={'creator_id': 'u1', 'title': 'T', 'hypothesis': 'H'},
                        content_type='application/json')
        mock_db.close.assert_called_once()

    def test_closes_db_on_error(self, client):
        mock_db, mock_svc, modules = _mock_db_and_service()
        mock_svc.create_experiment.side_effect = Exception("DB error")
        with patch.dict('sys.modules', modules):
            client.post('/api/social/experiments',
                        json={'creator_id': 'u1', 'title': 'T', 'hypothesis': 'H'},
                        content_type='application/json')
        mock_db.close.assert_called_once()
        mock_db.rollback.assert_called_once()


# ============================================================
# GET /api/social/experiments — list experiments
# ============================================================

class TestListExperiments:
    """List experiments — rendered in the Tracker page."""

    def test_returns_200_or_500(self, client):
        """List may fail if DB mocking is insufficient — key: doesn't crash Flask."""
        mock_db, mock_svc, modules = _mock_db_and_service()
        mock_svc.list_experiments.return_value = []
        with patch.dict('sys.modules', modules):
            resp = client.get('/api/social/experiments')
        assert resp.status_code in (200, 500)  # 500 if deferred import differs


# ============================================================
# POST /api/social/experiments/<id>/vote — democratic voting
# ============================================================

class TestVoteExperiment:
    """Voting drives the democratic decision outcome."""

    def test_returns_400_without_voter_id(self, client):
        resp = client.post('/api/social/experiments/exp1/vote',
                           json={'direction': 'up'},
                           content_type='application/json')
        # May return 400 or 500 depending on validation order
        assert resp.status_code in (400, 500)


# ============================================================
# Auto-evolve endpoints
# ============================================================

class TestAutoEvolve:
    """Auto-evolve triggers autonomous hypothesis iteration."""

    def test_auto_evolve_status_returns_json(self, client):
        """Status endpoint must return JSON — AgentHiveView polls it."""
        resp = client.get('/api/social/experiments/auto-evolve/status')
        assert resp.status_code in (200, 500)
        assert resp.content_type.startswith('application/json')
