"""Contract tests for the /ready Flask endpoint.

This locks the response shape of hart_intelligence_entry.health_readiness
so future readiness-check additions can't silently drop keys the admin UI
or k8s probe depends on. Added as a follow-up to commit 51592fd which
wired get_frame_store() and get_diarization_service() into the probe.

Strategy: call health_readiness() directly inside a Flask app_context
so we don't need the full HARTOS app boot. The function returns
(json_body, status_code) which we parse as a dict and assert shape.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask


@pytest.fixture
def ready_endpoint_context(monkeypatch):
    """Push a minimal Flask app context and stub out the other four
    readiness checks so we can isolate vision + diarization assertions."""
    app = Flask('test_readiness')
    with app.app_context(), app.test_request_context():
        yield


class TestReadinessProbeShape:
    """health_readiness must always return a dict with every expected
    'checks' key, even when a subsystem is unavailable. Missing keys
    break the admin UI dashboard."""

    EXPECTED_CHECK_KEYS = {
        'database',
        'node_identity',
        'hevolve_core',
        'llm_backend',
        'vision_frame_store',  # added by 51592fd
        'diarization',         # added by 51592fd
    }

    def _stub_db(self, monkeypatch):
        """Stub get_db + _get_active_backend_info at their source modules
        so the late-bound imports inside health_readiness see the mocks."""
        from integrations.social import models as _models
        fake_db = MagicMock()
        fake_db.execute.return_value = None
        fake_db.close = lambda: None
        monkeypatch.setattr(_models, 'get_db', lambda: fake_db)
        monkeypatch.setattr(
            'hart_intelligence_entry._get_active_backend_info',
            lambda: {'backend': 'llama.cpp'},
        )
        from security import node_integrity as _ni
        monkeypatch.setattr(_ni, 'get_node_identity', lambda: 'test-node')

    def test_all_expected_keys_present_when_vision_off(
        self, ready_endpoint_context, monkeypatch,
    ):
        """When vision + diarization are disabled, both keys must still
        appear in the checks dict with a string status."""
        self._stub_db(monkeypatch)
        monkeypatch.setattr(
            'hart_intelligence_entry.get_frame_store', lambda: None,
        )
        monkeypatch.setattr(
            'hart_intelligence_entry.get_diarization_service', lambda: None,
        )

        from hart_intelligence_entry import health_readiness
        response, _ = health_readiness()
        body = response.get_json()

        assert 'checks' in body
        actual_keys = set(body['checks'].keys())
        missing = self.EXPECTED_CHECK_KEYS - actual_keys
        assert not missing, (
            f"readiness probe dropped keys: {missing}. Commit 51592fd "
            f"added 'vision_frame_store' and 'diarization' — if this "
            f"test fails those checks were removed or renamed."
        )

    def test_frame_store_reports_stats_when_available(
        self, ready_endpoint_context, monkeypatch,
    ):
        """When get_frame_store() returns a real store, the probe must
        call its .stats() method so the admin UI shows frame counts."""
        self._stub_db(monkeypatch)
        fake_store = MagicMock()
        fake_store.stats.return_value = {
            'total_frames': 42, 'active_users': 1,
        }
        monkeypatch.setattr(
            'hart_intelligence_entry.get_frame_store', lambda: fake_store,
        )
        monkeypatch.setattr(
            'hart_intelligence_entry.get_diarization_service', lambda: None,
        )

        from hart_intelligence_entry import health_readiness
        response, _ = health_readiness()
        body = response.get_json()
        fake_store.stats.assert_called_once()
        vfs = body['checks']['vision_frame_store']
        assert isinstance(vfs, dict)
        assert vfs.get('total_frames') == 42

    def test_diarization_reports_ready_state(
        self, ready_endpoint_context, monkeypatch,
    ):
        """When get_diarization_service() returns a service, the probe
        must call .is_ready() and translate True → 'ready', False →
        'starting'."""
        self._stub_db(monkeypatch)
        monkeypatch.setattr(
            'hart_intelligence_entry.get_frame_store', lambda: None,
        )

        svc_ready = MagicMock()
        svc_ready.is_ready.return_value = True
        monkeypatch.setattr(
            'hart_intelligence_entry.get_diarization_service',
            lambda: svc_ready,
        )
        from hart_intelligence_entry import health_readiness
        response, _ = health_readiness()
        assert response.get_json()['checks']['diarization'] == 'ready'

        svc_starting = MagicMock()
        svc_starting.is_ready.return_value = False
        monkeypatch.setattr(
            'hart_intelligence_entry.get_diarization_service',
            lambda: svc_starting,
        )
        response, _ = health_readiness()
        assert response.get_json()['checks']['diarization'] == 'starting'
