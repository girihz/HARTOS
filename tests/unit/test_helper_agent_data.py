"""Tests for helper.py agent data backup + info helpers.

Covers two dead-code wirings:

1. backup_agent_data_file now auto-prunes via cleanup_old_backups so
   backups don't accumulate forever. Verify the prune runs and keeps
   only keep_count most-recent files.

2. get_agent_data_info returns a diagnostic dict for admin UI — exists
   flag, size_bytes, modified_at, saved_at, data_keys. The new admin
   endpoint /api/admin/agent-data/<id>/info wraps it.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from flask import Flask


@pytest.fixture
def tmp_agent_data_dir(tmp_path, monkeypatch):
    """Redirect helper's AGENT_DATA_DIR to an isolated tmp dir for every test.

    helper.py uses `current_app.logger` everywhere, which is a Flask
    LocalProxy that needs an application context to resolve. Push a
    minimal Flask app context around every test so those logger calls
    don't blow up with 'Working outside of application context'.
    """
    import helper
    monkeypatch.setattr(helper, 'AGENT_DATA_DIR', str(tmp_path))
    app = Flask('test_helper_agent_data')
    with app.app_context():
        yield tmp_path


class TestCleanupOldBackups:
    """cleanup_old_backups keeps only the N newest backup files per prompt_id."""

    def _make_backup(self, tmp_path, prompt_id, age_seconds):
        """Create a fake backup file with a modtime age_seconds in the past."""
        fname = f'{prompt_id}_agent_data_backup_{age_seconds}.json'
        fpath = os.path.join(tmp_path, fname)
        with open(fpath, 'w') as f:
            f.write('{}')
        stamp = (datetime.now() - timedelta(seconds=age_seconds)).timestamp()
        os.utime(fpath, (stamp, stamp))
        return fpath

    def test_keeps_newest_five_by_default(self, tmp_agent_data_dir):
        from helper import cleanup_old_backups
        for age in range(8):
            self._make_backup(tmp_agent_data_dir, prompt_id=42, age_seconds=age)
        deleted = cleanup_old_backups(42, keep_count=5)
        assert deleted == 3
        remaining = [
            f for f in os.listdir(tmp_agent_data_dir)
            if '42_agent_data_backup_' in f
        ]
        assert len(remaining) == 5

    def test_respects_custom_keep_count(self, tmp_agent_data_dir):
        from helper import cleanup_old_backups
        for age in range(10):
            self._make_backup(tmp_agent_data_dir, prompt_id=99, age_seconds=age)
        deleted = cleanup_old_backups(99, keep_count=2)
        assert deleted == 8
        remaining = [
            f for f in os.listdir(tmp_agent_data_dir)
            if '99_agent_data_backup_' in f
        ]
        assert len(remaining) == 2

    def test_per_prompt_isolation(self, tmp_agent_data_dir):
        """Cleanup for prompt 1 must NOT touch prompt 2's backups."""
        from helper import cleanup_old_backups
        for age in range(6):
            self._make_backup(tmp_agent_data_dir, prompt_id=1, age_seconds=age)
        for age in range(4):
            self._make_backup(tmp_agent_data_dir, prompt_id=2, age_seconds=age)
        cleanup_old_backups(1, keep_count=2)
        p1 = [f for f in os.listdir(tmp_agent_data_dir) if '1_agent_data_backup_' in f]
        p2 = [f for f in os.listdir(tmp_agent_data_dir) if '2_agent_data_backup_' in f]
        assert len(p1) == 2  # pruned
        assert len(p2) == 4  # untouched

    def test_zero_backups_returns_zero(self, tmp_agent_data_dir):
        """No files to prune is not an error — return 0."""
        from helper import cleanup_old_backups
        assert cleanup_old_backups(123, keep_count=5) == 0


class TestBackupAutoPrune:
    """backup_agent_data_file now calls cleanup_old_backups so the caller
    doesn't have to remember the two-step dance."""

    def test_backup_triggers_cleanup(self, tmp_agent_data_dir):
        from helper import backup_agent_data_file, get_agent_data_file_path

        # Create the primary agent data file that backup will copy
        primary = get_agent_data_file_path(7)
        with open(primary, 'w') as f:
            json.dump({'data': {'foo': 'bar'}, 'saved_at': 'test'}, f)

        with patch('helper.cleanup_old_backups') as mock_prune:
            result = backup_agent_data_file(7, keep_count=3)

        assert result is True
        mock_prune.assert_called_once_with(7, keep_count=3)


class TestGetAgentDataInfo:
    """get_agent_data_info returns a diagnostic dict for the admin UI."""

    def test_missing_file_returns_exists_false(self, tmp_agent_data_dir):
        from helper import get_agent_data_info
        info = get_agent_data_info(999)
        assert info['exists'] is False
        assert 'path' in info

    def test_existing_file_returns_full_diagnostic(self, tmp_agent_data_dir):
        from helper import get_agent_data_info, get_agent_data_file_path
        path = get_agent_data_file_path(77)
        payload = {
            'saved_at': '2026-04-10T18:00:00',
            'data': {'history': [], 'persona': {}, 'tools': []},
        }
        with open(path, 'w') as f:
            json.dump(payload, f)

        info = get_agent_data_info(77)

        assert info['exists'] is True
        assert info['path'] == path
        assert info['size_bytes'] > 0
        assert info['saved_at'] == '2026-04-10T18:00:00'
        assert set(info['data_keys']) == {'history', 'persona', 'tools'}
        assert 'modified_at' in info
