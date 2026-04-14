"""Tests for MCP HTTP Bridge — REST exposure of local HARTOS MCP tools."""

import json
import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture(autouse=True)
def reset_tools():
    """Reset tool registry between tests."""
    from integrations.mcp import mcp_http_bridge
    mcp_http_bridge._tools_loaded = False
    mcp_http_bridge._local_tools.clear()
    yield
    mcp_http_bridge._tools_loaded = False
    mcp_http_bridge._local_tools.clear()


@pytest.fixture
def app():
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    from integrations.mcp.mcp_http_bridge import mcp_local_bp
    app.register_blueprint(mcp_local_bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


# ── Health endpoint ────────────────────────────────────────────

class TestMCPHealth:
    def test_health_returns_ok(self, client):
        resp = client.get('/api/mcp/local/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert 'tools' in data
        assert data['server'] == 'hartos-mcp-local'

    def test_health_reports_tool_count(self, client):
        resp = client.get('/api/mcp/local/health')
        data = resp.get_json()
        assert isinstance(data['tools'], int)
        assert data['tools'] >= 14  # At least 14 tools (grows as new tools are added)


# ── Tools list endpoint ────────────────────────────────────────

class TestMCPToolsList:
    def test_list_returns_tools_array(self, client):
        resp = client.get('/api/mcp/local/tools/list')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tools' in data
        assert isinstance(data['tools'], list)

    def test_each_tool_has_required_fields(self, client):
        resp = client.get('/api/mcp/local/tools/list')
        data = resp.get_json()
        for tool in data['tools']:
            assert 'name' in tool
            assert 'description' in tool
            assert 'parameters' in tool

    def test_known_tools_present(self, client):
        resp = client.get('/api/mcp/local/tools/list')
        data = resp.get_json()
        tool_names = {t['name'] for t in data['tools']}
        # Core tools that must always exist
        required = {
            'list_agents', 'list_goals', 'agent_status',
            'list_recipes', 'system_health', 'social_query',
            'remember', 'recall',
        }
        assert required.issubset(tool_names), f"Missing: {required - tool_names}"

    def test_tool_parameters_have_schema(self, client):
        resp = client.get('/api/mcp/local/tools/list')
        data = resp.get_json()
        for tool in data['tools']:
            params = tool['parameters']
            assert params['type'] == 'object'
            assert 'properties' in params


# ── Tool execution endpoint ────────────────────────────────────

class TestMCPToolExecution:
    def test_execute_missing_tool_name_returns_400(self, client):
        resp = client.post('/api/mcp/local/tools/execute',
                          data=json.dumps({"arguments": {}}),
                          content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['success'] is False

    def test_execute_unknown_tool_returns_404(self, client):
        resp = client.post('/api/mcp/local/tools/execute',
                          data=json.dumps({"tool": "nonexistent_tool_xyz"}),
                          content_type='application/json')
        assert resp.status_code == 404
        data = resp.get_json()
        assert data['success'] is False
        assert 'available_tools' in data

    def test_execute_empty_body_returns_400(self, client):
        resp = client.post('/api/mcp/local/tools/execute',
                          data='{}', content_type='application/json')
        assert resp.status_code == 400

    def test_execute_list_recipes(self, client):
        resp = client.post('/api/mcp/local/tools/execute',
                          data=json.dumps({"tool": "list_recipes", "arguments": {}}),
                          content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert 'result' in data
        assert 'count' in data['result']
        assert 'recipes' in data['result']

    def test_execute_with_bad_arguments(self, client):
        resp = client.post('/api/mcp/local/tools/execute',
                          data=json.dumps({"tool": "social_query", "arguments": {"bad_arg": 1}}),
                          content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['success'] is False
        assert 'Invalid arguments' in data['error']


# ── Parameter extraction ───────────────────────────────────────

class TestParameterExtraction:
    def test_extract_parameters_basic(self):
        from integrations.mcp.mcp_http_bridge import _extract_parameters

        def sample_fn(name: str, count: int = 5):
            pass

        schema = _extract_parameters(sample_fn)
        assert schema['type'] == 'object'
        assert 'name' in schema['properties']
        assert 'count' in schema['properties']
        assert schema['properties']['count']['default'] == 5
        assert 'name' in schema['required']
        assert 'count' not in schema['required']

    def test_extract_parameters_no_args(self):
        from integrations.mcp.mcp_http_bridge import _extract_parameters
        schema = _extract_parameters(lambda: None)
        assert schema['properties'] == {}

    def test_extract_parameters_none(self):
        from integrations.mcp.mcp_http_bridge import _extract_parameters
        assert _extract_parameters(None) == {}

    def test_extract_parameters_types(self):
        from integrations.mcp.mcp_http_bridge import _extract_parameters

        def typed_fn(name: str, count: int, rate: float, flag: bool):
            pass

        schema = _extract_parameters(typed_fn)
        assert schema['properties']['name']['type'] == 'string'
        assert schema['properties']['count']['type'] == 'number'
        assert schema['properties']['rate']['type'] == 'number'
        assert schema['properties']['flag']['type'] == 'boolean'


# ── Tool loading ───────────────────────────────────────────────

class TestToolLoading:
    def test_loads_tools(self):
        from integrations.mcp.mcp_http_bridge import _load_tools, _local_tools
        _load_tools()
        assert len(_local_tools) >= 14  # Grows as new MCP tools are added

    def test_idempotent(self):
        from integrations.mcp import mcp_http_bridge
        mcp_http_bridge._load_tools()
        count1 = len(mcp_http_bridge._local_tools)
        mcp_http_bridge._load_tools()
        count2 = len(mcp_http_bridge._local_tools)
        assert count1 == count2

    def test_all_tools_have_callables(self):
        from integrations.mcp.mcp_http_bridge import _load_tools, _local_tools
        _load_tools()
        for t in _local_tools:
            assert callable(t['fn']), f"Tool {t['name']} has no callable"


# ── Auto-registration ─────────────────────────────────────────

class TestAutoRegistration:
    def test_auto_register_adds_to_registry(self):
        from integrations.mcp.mcp_http_bridge import auto_register_local_mcp
        from integrations.mcp.mcp_integration import mcp_registry
        mcp_registry.servers.pop('hartos_local', None)
        auto_register_local_mcp()
        assert 'hartos_local' in mcp_registry.servers
        connector = mcp_registry.servers['hartos_local']
        assert connector.connected is True
        assert '127.0.0.1' in connector.server_url

    def test_auto_register_idempotent(self):
        from integrations.mcp.mcp_http_bridge import auto_register_local_mcp
        from integrations.mcp.mcp_integration import mcp_registry
        mcp_registry.servers.pop('hartos_local', None)
        auto_register_local_mcp()
        auto_register_local_mcp()
        assert 'hartos_local' in mcp_registry.servers


# ── Port registry ─────────────────────────────────────────────

class TestPortRegistry:
    def test_mcp_port_registered(self):
        from core.port_registry import get_port
        port = get_port('mcp')
        assert port > 0
        assert port == 6791 or port == 682

    def test_mcp_port_env_override(self):
        from core import port_registry
        old_cache = port_registry._os_mode_cached
        try:
            port_registry._os_mode_cached = False
            with patch.dict(os.environ, {'HART_MCP_PORT': '9999'}):
                port = port_registry.get_port('mcp')
                assert port == 9999
        finally:
            port_registry._os_mode_cached = old_cache


# ── MCPServerConnector compatibility ───────────────────────────

class TestConnectorCompatibility:
    """Verify the REST API contract matches what MCPServerConnector expects."""

    def test_health_contract(self, client):
        """MCPServerConnector checks {url}/health for 200."""
        resp = client.get('/api/mcp/local/health')
        assert resp.status_code == 200

    def test_tools_list_contract(self, client):
        """MCPServerConnector expects {"tools": [...]} from {url}/tools/list."""
        resp = client.get('/api/mcp/local/tools/list')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'tools' in data
        assert isinstance(data['tools'], list)

    def test_tools_execute_contract(self, client):
        """MCPServerConnector sends POST {"tool": "...", "arguments": {...}}."""
        resp = client.post('/api/mcp/local/tools/execute',
                          data=json.dumps({"tool": "list_recipes", "arguments": {}}),
                          content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True


# ── End-to-end roundtrip ──────────────────────────────────────

class TestE2ERoundtrip:
    def test_discover_then_execute(self, client):
        """Simulate what MCPServerConnector does: discover, then execute."""
        # 1. Health check
        resp = client.get('/api/mcp/local/health')
        assert resp.status_code == 200

        # 2. Discover tools
        resp = client.get('/api/mcp/local/tools/list')
        tools = resp.get_json()['tools']
        tool_names = {t['name'] for t in tools}
        assert 'list_recipes' in tool_names

        # 3. Execute a tool
        resp = client.post('/api/mcp/local/tools/execute',
                          data=json.dumps({"tool": "list_recipes", "arguments": {}}),
                          content_type='application/json')
        assert resp.status_code == 200
        result = resp.get_json()
        assert result['success'] is True
