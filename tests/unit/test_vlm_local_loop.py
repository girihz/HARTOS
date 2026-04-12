"""
test_vlm_local_loop.py - Tests for integrations/vlm/local_loop.py helper functions.

Tests the pure-logic helpers in the VLM agentic loop that do NOT require
pyautogui or a running LLM server:
- _parse_vlm_response: JSON extraction from LLM output
- _build_action_payload: VLM response -> action payload conversion
- _build_vision_prompt: multimodal prompt construction
- run_local_agentic_loop: loop orchestration (mocked deps)

FT: Happy path parsing, code-block extraction, Box ID resolution, prompt format.
NFT: Malformed JSON resilience, empty inputs, injection attempts, iteration limits.
"""
import os
import sys
import json
import time
import pytest
from unittest.mock import patch, Mock, MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from integrations.vlm.local_loop import (
    _parse_vlm_response,
    _build_action_payload,
    _build_vision_prompt,
    run_local_agentic_loop,
    MAX_ITERATIONS,
    SYSTEM_PROMPT,
    _VLM_ACTION_LIST,
)


# ============================================================
# _parse_vlm_response
# ============================================================

class TestParseVlmResponse:
    """JSON extraction from LLM text output."""

    def test_parses_clean_json(self):
        text = '{"Next Action": "left_click", "coordinate": [100, 200], "Status": "IN_PROGRESS"}'
        result = _parse_vlm_response(text)
        assert result["Next Action"] == "left_click"
        assert result["coordinate"] == [100, 200]

    def test_parses_json_in_code_block(self):
        text = '```json\n{"Next Action": "type", "value": "hello", "Status": "IN_PROGRESS"}\n```'
        result = _parse_vlm_response(text)
        assert result["Next Action"] == "type"
        assert result["value"] == "hello"

    def test_parses_json_in_bare_code_block(self):
        text = '```\n{"Next Action": "None", "Status": "DONE"}\n```'
        result = _parse_vlm_response(text)
        assert result["Status"] == "DONE"

    def test_parses_json_with_surrounding_text(self):
        text = 'I see the screen. Here is my action:\n{"Next Action": "scroll_down", "Status": "IN_PROGRESS"}\nDone.'
        result = _parse_vlm_response(text)
        assert result["Next Action"] == "scroll_down"

    def test_empty_string_returns_done(self):
        result = _parse_vlm_response("")
        assert result["Next Action"] == "None"
        assert result["Status"] == "DONE"

    def test_none_input_returns_done(self):
        result = _parse_vlm_response(None)
        assert result["Next Action"] == "None"
        assert result["Status"] == "DONE"

    def test_unparseable_text_returns_done_with_reasoning(self):
        text = "I cannot determine the next action from this screenshot."
        result = _parse_vlm_response(text)
        assert result["Next Action"] == "None"
        assert result["Status"] == "DONE"
        assert text[:200] in result["Reasoning"]

    def test_malformed_json_in_code_block_falls_through(self):
        text = '```json\n{"Next Action": "left_click",}\n```'  # trailing comma
        result = _parse_vlm_response(text)
        # Should either parse or fall through to DONE
        assert "Next Action" in result

    def test_nested_json_objects(self):
        text = '{"Reasoning": "Click button", "Next Action": "left_click", "coordinate": [50, 60], "Status": "IN_PROGRESS"}'
        result = _parse_vlm_response(text)
        assert result["Reasoning"] == "Click button"

    def test_json_with_reasoning_field(self):
        text = '{"Reasoning": "I see a Save button at the top", "Next Action": "left_click", "coordinate": [400, 30], "Status": "IN_PROGRESS"}'
        result = _parse_vlm_response(text)
        assert "Save button" in result["Reasoning"]

    def test_json_with_box_id(self):
        text = '{"Reasoning": "Click element", "Next Action": "left_click", "Box ID": 7, "Status": "IN_PROGRESS"}'
        result = _parse_vlm_response(text)
        assert result["Box ID"] == 7

    def test_json_with_command_field_for_shell(self):
        text = '{"Next Action": "shell", "command": "notepad.exe", "Status": "IN_PROGRESS"}'
        result = _parse_vlm_response(text)
        assert result["command"] == "notepad.exe"

    def test_json_with_path_field_for_open_file_gui(self):
        text = '{"Next Action": "open_file_gui", "path": "C:\\\\test.pdf", "Status": "IN_PROGRESS"}'
        result = _parse_vlm_response(text)
        assert result["Next Action"] == "open_file_gui"


# ============================================================
# _build_action_payload
# ============================================================

class TestBuildActionPayload:
    """Convert VLM response to execute_action payload."""

    def test_basic_click_action(self):
        action_json = {"Next Action": "left_click", "coordinate": [100, 200]}
        parsed = {"parsed_content_list": []}
        payload = _build_action_payload(action_json, parsed)
        assert payload["action"] == "left_click"
        assert payload["coordinate"] == [100, 200]

    def test_type_action_with_text(self):
        action_json = {"Next Action": "type", "value": "hello world"}
        parsed = {"parsed_content_list": []}
        payload = _build_action_payload(action_json, parsed)
        assert payload["action"] == "type"
        assert payload["text"] == "hello world"

    def test_box_id_resolves_to_coordinate(self):
        """When coordinate is None but Box ID matches a parsed element, resolve center of bbox."""
        action_json = {"Next Action": "left_click", "Box ID": 3}
        parsed = {
            "parsed_content_list": [
                {"idx": 1, "bbox": [0, 0, 100, 100]},
                {"idx": 3, "bbox": [200, 300, 400, 500]},
            ]
        }
        payload = _build_action_payload(action_json, parsed)
        # Center of bbox [200,300,400,500] = (300, 400)
        assert payload["coordinate"] == [300, 400]

    def test_box_id_with_id_field(self):
        """Some parsers use 'id' instead of 'idx'."""
        action_json = {"Next Action": "left_click", "Box ID": 5}
        parsed = {
            "parsed_content_list": [
                {"id": 5, "bbox": [10, 20, 30, 40]},
            ]
        }
        payload = _build_action_payload(action_json, parsed)
        assert payload["coordinate"] == [20, 30]  # center of [10,20,30,40]

    def test_box_id_not_found_leaves_no_coordinate(self):
        action_json = {"Next Action": "left_click", "Box ID": 999}
        parsed = {"parsed_content_list": [{"idx": 1, "bbox": [0, 0, 100, 100]}]}
        payload = _build_action_payload(action_json, parsed)
        assert "coordinate" not in payload

    def test_explicit_coordinate_takes_precedence_over_box_id(self):
        action_json = {"Next Action": "left_click", "coordinate": [50, 50], "Box ID": 1}
        parsed = {"parsed_content_list": [{"idx": 1, "bbox": [200, 200, 400, 400]}]}
        payload = _build_action_payload(action_json, parsed)
        assert payload["coordinate"] == [50, 50]

    def test_shell_action_passes_command(self):
        action_json = {"Next Action": "shell", "command": "dir C:\\"}
        parsed = {"parsed_content_list": []}
        payload = _build_action_payload(action_json, parsed)
        assert payload["command"] == "dir C:\\"

    def test_open_file_gui_passes_path(self):
        action_json = {"Next Action": "open_file_gui", "path": "notepad"}
        parsed = {"parsed_content_list": []}
        payload = _build_action_payload(action_json, parsed)
        assert payload["path"] == "notepad"

    def test_write_file_passes_content(self):
        action_json = {"Next Action": "write_file", "path": "/tmp/test.txt", "content": "hello"}
        parsed = {"parsed_content_list": []}
        payload = _build_action_payload(action_json, parsed)
        assert payload["content"] == "hello"
        assert payload["path"] == "/tmp/test.txt"

    def test_no_extra_keys_when_not_present(self):
        action_json = {"Next Action": "wait", "value": ""}
        parsed = {"parsed_content_list": []}
        payload = _build_action_payload(action_json, parsed)
        assert "command" not in payload
        assert "path" not in payload
        assert "content" not in payload

    def test_empty_parsed_content_list(self):
        action_json = {"Next Action": "left_click", "Box ID": 1}
        parsed = {"parsed_content_list": []}
        payload = _build_action_payload(action_json, parsed)
        assert "coordinate" not in payload


# ============================================================
# _build_vision_prompt
# ============================================================

class TestBuildVisionPrompt:
    """Multimodal prompt construction."""

    def test_first_iteration_prompt(self):
        content = _build_vision_prompt("UI: Button[OK]", "base64data", iteration=0)
        assert len(content) == 2  # text + image
        assert content[0]["type"] == "text"
        assert "current screen state" in content[0]["text"]
        assert "UI: Button[OK]" in content[0]["text"]

    def test_subsequent_iteration_prompt(self):
        content = _build_vision_prompt("UI: Menu open", "base64data", iteration=3)
        assert "updated screen" in content[0]["text"]
        assert "previous action" in content[0]["text"].lower()

    def test_includes_image(self):
        content = _build_vision_prompt("info", "abc123", iteration=0)
        assert content[1]["type"] == "image_url"
        assert "abc123" in content[1]["image_url"]["url"]

    def test_image_is_base64_data_url(self):
        content = _build_vision_prompt("info", "xyz", iteration=0)
        url = content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")


# ============================================================
# Module constants
# ============================================================

class TestModuleConstants:
    """Verify module-level constants are sensible."""

    def test_max_iterations_positive(self):
        assert MAX_ITERATIONS > 0
        assert MAX_ITERATIONS <= 100

    def test_system_prompt_contains_os_name(self):
        import platform
        assert platform.system() in SYSTEM_PROMPT

    def test_system_prompt_contains_json_format(self):
        assert "JSON" in SYSTEM_PROMPT

    def test_system_prompt_mentions_done_status(self):
        assert "DONE" in SYSTEM_PROMPT

    def test_action_list_has_gui_actions(self):
        assert "left_click" in _VLM_ACTION_LIST
        assert "type" in _VLM_ACTION_LIST
        assert "scroll_up" in _VLM_ACTION_LIST

    def test_action_list_has_deterministic_actions(self):
        assert "shell" in _VLM_ACTION_LIST
        assert "open_file_gui" in _VLM_ACTION_LIST


# ============================================================
# run_local_agentic_loop (mocked)
# ============================================================

class TestRunLocalAgenticLoop:
    """Loop orchestration with all external deps mocked."""

    def test_completes_on_done_action(self):
        """Loop ends when VLM returns Status=DONE."""
        mock_screenshot = "fakebase64"
        done_response = '{"Next Action": "None", "Status": "DONE", "Reasoning": "Task complete"}'

        mock_lct = MagicMock()
        mock_lct.take_screenshot.return_value = mock_screenshot
        mock_lct.execute_action.return_value = {"status": "ok"}
        # VLM_IMG_W/VLM_IMG_H needed for unified mode but not legacy
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            with patch('integrations.vlm.local_loop._call_local_llm', return_value=done_response):
                mock_omni = MagicMock()
                mock_omni.parse_screen.return_value = {'screen_info': 'Desktop', 'parsed_content_list': []}
                with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                    result = run_local_agentic_loop(
                        {"instruction_to_vlm_agent": "Open notepad"},
                        tier='inprocess',
                        max_iterations=5,
                    )

        assert result["status"] == "success"
        assert result["exit_reason"] == "done"
        assert len(result["extracted_responses"]) == 1
        assert result["extracted_responses"][0]["type"] == "completion"

    def test_stops_at_max_iterations(self):
        """Loop stops at max_iterations and returns incomplete."""
        action_response = '{"Next Action": "left_click", "coordinate": [100, 200], "Status": "IN_PROGRESS"}'

        mock_lct = MagicMock()
        mock_lct.take_screenshot.return_value = "base64"
        mock_lct.execute_action.return_value = {"status": "ok", "output": "clicked"}
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            with patch('integrations.vlm.local_loop._call_local_llm', return_value=action_response):
                mock_omni = MagicMock()
                mock_omni.parse_screen.return_value = {'screen_info': '', 'parsed_content_list': []}
                with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                    with patch('integrations.vlm.local_loop.time.sleep'):
                        result = run_local_agentic_loop(
                            {"instruction_to_vlm_agent": "Keep clicking"},
                            tier='inprocess',
                            max_iterations=3,
                        )

        assert result["status"] == "incomplete"
        assert result["exit_reason"] == "max_iterations"
        assert len(result["extracted_responses"]) == 3

    def test_stops_on_eta_timeout(self):
        """Loop stops when max_ETA_in_seconds is exceeded."""
        mock_lct = MagicMock()
        mock_lct.take_screenshot.return_value = "base64"
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            mock_omni = MagicMock()
            mock_omni.parse_screen.return_value = {'screen_info': '', 'parsed_content_list': []}
            with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                result = run_local_agentic_loop(
                    {
                        "instruction_to_vlm_agent": "Long task",
                        "max_ETA_in_seconds": 0,  # already expired
                    },
                    tier='inprocess',
                    max_iterations=10,
                )

        assert result["status"] == "incomplete"
        assert result["exit_reason"] == "timeout"

    def test_stops_after_3_consecutive_action_errors(self):
        """3 consecutive action errors triggers abort."""
        action_response = '{"Next Action": "left_click", "coordinate": [10, 10], "Status": "IN_PROGRESS"}'

        mock_lct = MagicMock()
        mock_lct.take_screenshot.return_value = "base64"
        mock_lct.execute_action.return_value = {"status": "error", "output": "pyautogui not available"}
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            with patch('integrations.vlm.local_loop._call_local_llm', return_value=action_response):
                mock_omni = MagicMock()
                mock_omni.parse_screen.return_value = {'screen_info': '', 'parsed_content_list': []}
                with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                    with patch('integrations.vlm.local_loop.time.sleep'):
                        result = run_local_agentic_loop(
                            {"instruction_to_vlm_agent": "Test errors"},
                            tier='inprocess',
                            max_iterations=10,
                        )

        assert result["exit_reason"] == "action_error"
        assert len(result["extracted_responses"]) == 3

    def test_stops_after_3_consecutive_exceptions(self):
        """3 consecutive iteration exceptions triggers abort."""
        mock_lct = MagicMock()
        mock_lct.take_screenshot.side_effect = RuntimeError("no display")
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            mock_omni = MagicMock()
            with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                with patch('integrations.vlm.local_loop.time.sleep'):
                    result = run_local_agentic_loop(
                        {"instruction_to_vlm_agent": "Test crash"},
                        tier='inprocess',
                        max_iterations=10,
                    )

        assert result["exit_reason"] == "action_error"
        assert len(result["extracted_responses"]) == 3
        assert result["extracted_responses"][0]["type"] == "error"

    def test_execution_time_tracked(self):
        """Result includes execution_time_seconds."""
        done_response = '{"Next Action": "None", "Status": "DONE"}'

        mock_lct = MagicMock()
        mock_lct.take_screenshot.return_value = "base64"
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            with patch('integrations.vlm.local_loop._call_local_llm', return_value=done_response):
                mock_omni = MagicMock()
                mock_omni.parse_screen.return_value = {'screen_info': '', 'parsed_content_list': []}
                with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                    result = run_local_agentic_loop(
                        {"instruction_to_vlm_agent": "Quick task"},
                        tier='inprocess',
                    )
        assert "execution_time_seconds" in result
        assert result["execution_time_seconds"] >= 0

    def test_extracted_responses_include_iteration_number(self):
        """Each response includes its iteration index."""
        done_response = '{"Next Action": "None", "Status": "DONE", "Reasoning": "Done"}'

        mock_lct = MagicMock()
        mock_lct.take_screenshot.return_value = "base64"
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            with patch('integrations.vlm.local_loop._call_local_llm', return_value=done_response):
                mock_omni = MagicMock()
                mock_omni.parse_screen.return_value = {'screen_info': '', 'parsed_content_list': []}
                with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                    result = run_local_agentic_loop(
                        {"instruction_to_vlm_agent": "test"},
                        tier='http',
                    )
        assert result["extracted_responses"][0]["iteration"] == 1

    def test_uses_enhanced_instruction_when_provided(self):
        """enhanced_instruction overrides instruction_to_vlm_agent for prompts."""
        done_response = '{"Next Action": "None", "Status": "DONE"}'

        mock_lct = MagicMock()
        mock_lct.take_screenshot.return_value = "base64"
        mock_lct.VLM_IMG_W = 1280
        mock_lct.VLM_IMG_H = 720

        with patch.dict('sys.modules', {'integrations.vlm.local_computer_tool': mock_lct}):
            with patch('integrations.vlm.local_loop._call_local_llm', return_value=done_response) as mock_llm:
                mock_omni = MagicMock()
                mock_omni.parse_screen.return_value = {'screen_info': '', 'parsed_content_list': []}
                with patch.dict('sys.modules', {'integrations.vlm.local_omniparser': mock_omni}):
                    run_local_agentic_loop(
                        {
                            "instruction_to_vlm_agent": "basic",
                            "enhanced_instruction": "enhanced version",
                        },
                        tier='inprocess',
                    )
        # The messages passed to LLM should contain the enhanced instruction
        messages = mock_llm.call_args[0][0]
        user_msg = messages[1]["content"]
        assert user_msg == "enhanced version"
