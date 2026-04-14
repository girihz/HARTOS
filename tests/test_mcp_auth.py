"""Tests for MCP HTTP bridge auth gate env-var overrides.

Covers the three override paths added to `mcp_http_bridge.py` so HARTOS
standalone deployments (Docker/K8s/air-gapped) no longer require the
`%LOCALAPPDATA%/Nunba/mcp.token` disk file or the Nunba admin UI:

  1. HARTOS_MCP_DISABLE_AUTH=1   → auth gate is skipped entirely
  2. HARTOS_MCP_TOKEN=<literal>  → use env-var token instead of disk file
  3. HARTOS_MCP_TOKEN_FILE=<p>   → read token from a custom file path

Each test isolates the module-level token cache and restores env state
so tests don't leak into one another.
"""

import importlib
import os
import sys

import pytest


@pytest.fixture
def mcp_bridge(monkeypatch, tmp_path):
    """Import `mcp_http_bridge` fresh with a clean env and token cache.

    Clears the three override env vars + LOCALAPPDATA/HOME so default
    disk-path behaviour is predictable, then reloads the module so the
    `_MCP_TOKEN_CACHE` / `_MCP_AUTH_DISABLED_WARNED` globals are reset.
    """
    for var in (
        'HARTOS_MCP_DISABLE_AUTH',
        'HARTOS_MCP_TOKEN',
        'HARTOS_MCP_TOKEN_FILE',
    ):
        monkeypatch.delenv(var, raising=False)
    # Redirect default-disk-path writes into tmp_path so we don't touch
    # the real %LOCALAPPDATA%/Nunba/mcp.token during tests.
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('USERPROFILE', str(tmp_path))

    mod_name = 'integrations.mcp.mcp_http_bridge'
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    # Hard-reset the cache in case import-time side effects populated it.
    mod._MCP_TOKEN_CACHE = None
    mod._MCP_AUTH_DISABLED_WARNED = False
    return mod


def test_env_var_token_literal_wins_over_disk(mcp_bridge, tmp_path, monkeypatch):
    """HARTOS_MCP_TOKEN literal value must be returned verbatim and must
    NOT trigger any filesystem write (Docker/K8s secret injection path)."""
    monkeypatch.setenv('HARTOS_MCP_TOKEN', 'k8s-secret-abc-123')
    token = mcp_bridge._ensure_mcp_token()
    assert token == 'k8s-secret-abc-123'
    # Default disk path must NOT have been created
    default = mcp_bridge._mcp_token_path()
    assert not os.path.exists(default), (
        f"literal env-var token must not write {default}"
    )


def test_env_var_token_file_is_read(mcp_bridge, tmp_path, monkeypatch):
    """HARTOS_MCP_TOKEN_FILE must read the token from a custom path
    (Vault/cert-manager/K8s mounted-secret path)."""
    secret_file = tmp_path / 'vault' / 'mcp.token'
    secret_file.parent.mkdir(parents=True)
    secret_file.write_text('vault-mounted-token-xyz\n', encoding='utf-8')

    monkeypatch.setenv('HARTOS_MCP_TOKEN_FILE', str(secret_file))
    token = mcp_bridge._ensure_mcp_token()
    # `.strip()` inside _ensure_mcp_token drops the trailing newline.
    assert token == 'vault-mounted-token-xyz'
    # Default disk path must NOT have been created
    default = mcp_bridge._mcp_token_path()
    assert not os.path.exists(default)


def test_disable_auth_env_flag_bypasses_gate(mcp_bridge, monkeypatch):
    """HARTOS_MCP_DISABLE_AUTH=1 must short-circuit the before_request
    gate — no token required — and emit exactly one WARN on first hit."""
    import flask

    monkeypatch.setenv('HARTOS_MCP_DISABLE_AUTH', '1')

    app = flask.Flask(__name__)
    app.register_blueprint(mcp_bridge.mcp_local_bp)

    warn_calls = []
    original_warning = mcp_bridge.logger.warning

    def _capture_warning(msg, *args, **kwargs):
        warn_calls.append(msg % args if args else msg)
        return original_warning(msg, *args, **kwargs)

    monkeypatch.setattr(mcp_bridge.logger, 'warning', _capture_warning)

    with app.test_client() as client:
        # First call — no Authorization header — should succeed (not 403)
        r1 = client.get('/api/mcp/local/tools/list')
        assert r1.status_code != 403, (
            "disable-auth env flag must bypass the 403 gate"
        )
        # Second call — same, still bypassed
        r2 = client.get('/api/mcp/local/tools/list')
        assert r2.status_code != 403

    # Exactly one WARN emitted across the two requests
    disable_warns = [
        w for w in warn_calls if 'MCP auth disabled via env' in str(w)
    ]
    assert len(disable_warns) == 1, (
        f"expected exactly one 'auth disabled' WARN, got {len(disable_warns)}: "
        f"{disable_warns}"
    )
