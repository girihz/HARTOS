"""
test_vlm_agent_integration.py - Tests for integrations/vlm/vlm_agent_integration.py

Tests VLMAgentContext: the bridge between VLM visual computer-use and agent ledger.

FT: Availability checks, screen context retrieval, context injection into ledger tasks,
    action execution, visual feedback generation, tool definition creation, status summary,
    singleton getter.
NFT: History size caps, error resilience, network failure handling, empty-state behavior.
"""
import os
import sys
import json
import pytest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import integrations.vlm.vlm_agent_integration as vai_mod
from integrations.vlm.vlm_agent_integration import VLMAgentContext, get_vlm_context


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def ctx():
    """Fresh VLMAgentContext with no network dependency."""
    return VLMAgentContext(
        vlm_server_url="http://localhost:5001",
        omniparser_url="http://localhost:8080",
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    vai_mod._vlm_context = None
    yield
    vai_mod._vlm_context = None


# ============================================================
# __init__ - Constructor defaults
# ============================================================

class TestVLMAgentContextInit:
    """VLMAgentContext constructor and default URLs."""

    def test_default_urls_from_env(self):
        with patch.dict(os.environ, {'VLM_GUI_PORT': '7001', 'OMNIPARSER_PORT': '7002'}):
            ctx = VLMAgentContext()
            assert '7001' in ctx.vlm_server_url
            assert '7002' in ctx.omniparser_url

    def test_explicit_urls_override_env(self):
        ctx = VLMAgentContext(
            vlm_server_url="http://custom:9999",
            omniparser_url="http://custom:8888",
        )
        assert ctx.vlm_server_url == "http://custom:9999"
        assert ctx.omniparser_url == "http://custom:8888"

    def test_histories_start_empty(self, ctx):
        assert ctx.screen_history == []
        assert ctx.action_history == []

    def test_default_ports_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove VLM_GUI_PORT and OMNIPARSER_PORT
            os.environ.pop('VLM_GUI_PORT', None)
            os.environ.pop('OMNIPARSER_PORT', None)
            ctx = VLMAgentContext()
            assert '5001' in ctx.vlm_server_url
            assert '8080' in ctx.omniparser_url


# ============================================================
# is_vlm_available
# ============================================================

class TestIsVlmAvailable:
    """VLM server health check."""

    def test_returns_true_when_health_200(self, ctx):
        mock_resp = Mock()
        mock_resp.status_code = 200
        with patch('integrations.vlm.vlm_agent_integration.pooled_get', return_value=mock_resp):
            assert ctx.is_vlm_available() is True

    def test_returns_false_when_health_500(self, ctx):
        mock_resp = Mock()
        mock_resp.status_code = 500
        with patch('integrations.vlm.vlm_agent_integration.pooled_get', return_value=mock_resp):
            assert ctx.is_vlm_available() is False

    def test_returns_false_on_connection_error(self, ctx):
        with patch('integrations.vlm.vlm_agent_integration.pooled_get', side_effect=ConnectionError("refused")):
            assert ctx.is_vlm_available() is False

    def test_returns_false_on_timeout(self, ctx):
        with patch('integrations.vlm.vlm_agent_integration.pooled_get', side_effect=TimeoutError("timeout")):
            assert ctx.is_vlm_available() is False


# ============================================================
# is_omniparser_available
# ============================================================

class TestIsOmniparserAvailable:
    """OmniParser probe check."""

    def test_returns_true_when_probe_200(self, ctx):
        mock_resp = Mock()
        mock_resp.status_code = 200
        with patch('integrations.vlm.vlm_agent_integration.pooled_get', return_value=mock_resp):
            assert ctx.is_omniparser_available() is True

    def test_returns_false_on_network_error(self, ctx):
        with patch('integrations.vlm.vlm_agent_integration.pooled_get', side_effect=Exception("err")):
            assert ctx.is_omniparser_available() is False


# ============================================================
# get_screen_context
# ============================================================

class TestGetScreenContext:
    """Screen context retrieval from OmniParser."""

    def test_returns_none_when_omniparser_unavailable(self, ctx):
        with patch.object(ctx, 'is_omniparser_available', return_value=False):
            assert ctx.get_screen_context() is None

    def test_returns_screen_data_on_success(self, ctx):
        screen_data = {
            "screen_info": "Calculator app open",
            "parsed_content_list": [{"idx": 1, "text": "="}],
            "width": 1920,
            "height": 1080,
        }
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = screen_data
        with patch.object(ctx, 'is_omniparser_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                result = ctx.get_screen_context()
        assert result == screen_data

    def test_appends_to_screen_history(self, ctx):
        screen_data = {"screen_info": "Desktop", "parsed_content_list": []}
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = screen_data
        with patch.object(ctx, 'is_omniparser_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                ctx.get_screen_context()
        assert len(ctx.screen_history) == 1
        assert ctx.screen_history[0]['screen_info'] == "Desktop"

    def test_screen_history_capped_at_10(self, ctx):
        """History should not exceed 10 entries."""
        screen_data = {"screen_info": "X", "parsed_content_list": []}
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = screen_data
        with patch.object(ctx, 'is_omniparser_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                for _ in range(15):
                    ctx.get_screen_context()
        assert len(ctx.screen_history) == 10

    def test_returns_none_on_http_error(self, ctx):
        mock_resp = Mock()
        mock_resp.status_code = 500
        with patch.object(ctx, 'is_omniparser_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                assert ctx.get_screen_context() is None

    def test_returns_none_on_exception(self, ctx):
        with patch.object(ctx, 'is_omniparser_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', side_effect=Exception("network")):
                assert ctx.get_screen_context() is None


# ============================================================
# inject_visual_context_into_ledger_task
# ============================================================

class TestInjectVisualContext:
    """Context injection into agent ledger tasks."""

    def test_injects_visual_context_when_available(self, ctx):
        screen_data = {
            "screen_info": "Code editor with Python file",
            "parsed_content_list": [{"idx": 1}, {"idx": 2}, {"idx": 3}],
            "width": 1920,
            "height": 1080,
        }
        with patch.object(ctx, 'get_screen_context', return_value=screen_data):
            task = {"task_id": "t1", "description": "Write code"}
            result = ctx.inject_visual_context_into_ledger_task(task)

        assert result["visual_context"]["has_screen_info"] is True
        assert result["visual_context"]["visible_elements"] == 3
        assert result["visual_context"]["screen_dimensions"]["width"] == 1920

    def test_injects_unavailable_note_when_no_context(self, ctx):
        with patch.object(ctx, 'get_screen_context', return_value=None):
            task = {"task_id": "t2"}
            result = ctx.inject_visual_context_into_ledger_task(task)

        assert result["visual_context"]["has_screen_info"] is False
        assert "not available" in result["visual_context"]["note"]

    def test_preserves_existing_task_context(self, ctx):
        with patch.object(ctx, 'get_screen_context', return_value=None):
            task = {"task_id": "t3", "existing_key": "keep me"}
            result = ctx.inject_visual_context_into_ledger_task(task)

        assert result["existing_key"] == "keep me"

    def test_screen_summary_truncated_to_500_chars(self, ctx):
        screen_data = {
            "screen_info": "A" * 1000,
            "parsed_content_list": [],
            "width": 800,
            "height": 600,
        }
        with patch.object(ctx, 'get_screen_context', return_value=screen_data):
            result = ctx.inject_visual_context_into_ledger_task({})

        assert len(result["visual_context"]["screen_summary"]) == 500


# ============================================================
# execute_vlm_action
# ============================================================

class TestExecuteVlmAction:
    """Action execution via VLM agent."""

    def test_returns_error_when_vlm_unavailable(self, ctx):
        with patch.object(ctx, 'is_vlm_available', return_value=False):
            result = ctx.execute_vlm_action("left_click", {"coordinate": [100, 200]})
        assert result["status"] == "error"
        assert "not available" in result["message"]

    def test_successful_action_returns_result(self, ctx):
        action_result = {"status": "success", "output": "Clicked at (100,200)"}
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = action_result
        with patch.object(ctx, 'is_vlm_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                result = ctx.execute_vlm_action("left_click", {"coordinate": [100, 200]})
        assert result["status"] == "success"

    def test_action_appended_to_history(self, ctx):
        action_result = {"status": "success"}
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = action_result
        with patch.object(ctx, 'is_vlm_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                ctx.execute_vlm_action("type", {"text": "hello"})
        assert len(ctx.action_history) == 1
        assert ctx.action_history[0]["action"] == "type"

    def test_action_history_capped_at_50(self, ctx):
        action_result = {"status": "success"}
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = action_result
        with patch.object(ctx, 'is_vlm_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                for i in range(55):
                    ctx.execute_vlm_action("type", {"text": f"msg{i}"})
        assert len(ctx.action_history) == 50

    def test_http_error_returns_error_dict(self, ctx):
        mock_resp = Mock()
        mock_resp.status_code = 503
        with patch.object(ctx, 'is_vlm_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp):
                result = ctx.execute_vlm_action("scroll_up")
        assert result["status"] == "error"
        assert "503" in result["message"]

    def test_network_exception_returns_error(self, ctx):
        with patch.object(ctx, 'is_vlm_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', side_effect=ConnectionError("timeout")):
                result = ctx.execute_vlm_action("left_click")
        assert result["status"] == "error"
        assert result["action"] == "left_click"

    def test_default_parameters_empty_dict(self, ctx):
        """When no parameters given, payload.parameters should be {}."""
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "success"}
        with patch.object(ctx, 'is_vlm_available', return_value=True):
            with patch('integrations.vlm.vlm_agent_integration.pooled_post', return_value=mock_resp) as mock_post:
                ctx.execute_vlm_action("wait")
        payload = mock_post.call_args[1]['json']
        assert payload["parameters"] == {}


# ============================================================
# get_visual_feedback_for_task
# ============================================================

class TestGetVisualFeedback:
    """Visual feedback text generation."""

    def test_unavailable_message_when_no_context(self, ctx):
        with patch.object(ctx, 'get_screen_context', return_value=None):
            feedback = ctx.get_visual_feedback_for_task("Open Notepad")
        assert "unavailable" in feedback.lower()

    def test_includes_task_description(self, ctx):
        screen_data = {
            "screen_info": "Desktop view",
            "parsed_content_list": [{"idx": 1}, {"idx": 2}],
            "width": 1920,
            "height": 1080,
        }
        with patch.object(ctx, 'get_screen_context', return_value=screen_data):
            feedback = ctx.get_visual_feedback_for_task("Open Notepad")
        assert "Open Notepad" in feedback

    def test_includes_element_count(self, ctx):
        screen_data = {
            "screen_info": "Browser",
            "parsed_content_list": [{"idx": i} for i in range(7)],
            "width": 1920,
            "height": 1080,
        }
        with patch.object(ctx, 'get_screen_context', return_value=screen_data):
            feedback = ctx.get_visual_feedback_for_task("Browse web")
        assert "7" in feedback

    def test_includes_recent_actions(self, ctx):
        ctx.action_history = [
            {"timestamp": "T1", "action": "left_click", "result": "success"},
            {"timestamp": "T2", "action": "type", "result": "success"},
        ]
        screen_data = {"screen_info": "", "parsed_content_list": [], "width": 800, "height": 600}
        with patch.object(ctx, 'get_screen_context', return_value=screen_data):
            feedback = ctx.get_visual_feedback_for_task("Continue task")
        assert "left_click" in feedback
        assert "type" in feedback

    def test_screen_info_truncated_in_feedback(self, ctx):
        screen_data = {
            "screen_info": "X" * 1000,
            "parsed_content_list": [],
            "width": 800,
            "height": 600,
        }
        with patch.object(ctx, 'get_screen_context', return_value=screen_data):
            feedback = ctx.get_visual_feedback_for_task("task")
        # screen_info should be at most 500 chars in the feedback
        lines = feedback.split('\n')
        screen_line = [l for l in lines if 'XXXXX' in l]
        if screen_line:
            assert len(screen_line[0]) <= 510  # some margin for prefix


# ============================================================
# create_vlm_enabled_tool
# ============================================================

class TestCreateVlmEnabledTool:
    """Tool definition creation for agent registration."""

    def test_tool_has_correct_name(self, ctx):
        tool = ctx.create_vlm_enabled_tool("screen_control", "Control the screen")
        assert tool["function"]["name"] == "screen_control"

    def test_tool_has_action_enum(self, ctx):
        tool = ctx.create_vlm_enabled_tool("vlm_tool", "VLM tool")
        action_prop = tool["function"]["parameters"]["properties"]["action"]
        assert "enum" in action_prop
        assert "left_click" in action_prop["enum"]
        assert "type" in action_prop["enum"]
        assert "shell" not in action_prop["enum"]  # shell is not in the tool enum

    def test_tool_type_is_function(self, ctx):
        tool = ctx.create_vlm_enabled_tool("t", "d")
        assert tool["type"] == "function"

    def test_action_is_required(self, ctx):
        tool = ctx.create_vlm_enabled_tool("t", "d")
        assert "action" in tool["function"]["parameters"]["required"]


# ============================================================
# get_status_summary
# ============================================================

class TestGetStatusSummary:
    """Status summary for admin/monitoring."""

    def test_all_keys_present(self, ctx):
        with patch.object(ctx, 'is_vlm_available', return_value=False):
            with patch.object(ctx, 'is_omniparser_available', return_value=False):
                status = ctx.get_status_summary()
        expected_keys = {
            'vlm_available', 'omniparser_available',
            'screen_history_count', 'action_history_count',
            'last_screen_capture', 'last_action',
        }
        assert set(status.keys()) == expected_keys

    def test_empty_history_defaults(self, ctx):
        with patch.object(ctx, 'is_vlm_available', return_value=False):
            with patch.object(ctx, 'is_omniparser_available', return_value=False):
                status = ctx.get_status_summary()
        assert status['screen_history_count'] == 0
        assert status['action_history_count'] == 0
        assert status['last_screen_capture'] is None
        assert status['last_action'] is None

    def test_reflects_history_after_actions(self, ctx):
        ctx.action_history.append({"timestamp": "T1", "action": "click", "result": "ok"})
        ctx.screen_history.append({"timestamp": "T2", "screen_info": "", "element_count": 0})
        with patch.object(ctx, 'is_vlm_available', return_value=True):
            with patch.object(ctx, 'is_omniparser_available', return_value=True):
                status = ctx.get_status_summary()
        assert status['screen_history_count'] == 1
        assert status['action_history_count'] == 1
        assert status['last_screen_capture'] == "T2"
        assert status['last_action']['action'] == "click"


# ============================================================
# get_vlm_context singleton
# ============================================================

class TestGetVlmContext:
    """Singleton factory."""

    def test_returns_same_instance(self):
        ctx1 = get_vlm_context()
        ctx2 = get_vlm_context()
        assert ctx1 is ctx2

    def test_creates_new_after_reset(self):
        ctx1 = get_vlm_context()
        vai_mod._vlm_context = None
        ctx2 = get_vlm_context()
        assert ctx1 is not ctx2

    def test_passes_urls_on_first_creation(self):
        ctx = get_vlm_context(
            vlm_server_url="http://custom:1111",
            omniparser_url="http://custom:2222",
        )
        assert ctx.vlm_server_url == "http://custom:1111"
        assert ctx.omniparser_url == "http://custom:2222"
