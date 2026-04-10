"""
Unit tests for core.peer_link.ui_commands — the agentic UI bridge facade.

Verifies:
  - Required-field validation (user_id, screen, layout shape)
  - Topic selection: device_id present → fleet.command, absent → fleet.command.user
  - Payload shape: always emits cmd_type key (NOT 'command'), includes id
  - Returns None on publish failure, command_id on success
  - Prevents the pre-existing bootstrap.py bug class via an explicit
    "never emit payloads with 'command' key" regression test.

MessageBus is mocked so tests run without any transport.
"""

from unittest.mock import MagicMock, patch

import pytest


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_bus():
    """Mock get_message_bus() — ui_commands imports it lazily inside _publish
    so we patch at its source module (core.peer_link.message_bus).
    """
    bus = MagicMock()
    bus.publish = MagicMock(return_value='msg-abc')
    with patch('core.peer_link.message_bus.get_message_bus', return_value=bus):
        yield bus


# ─── Required-field validation ──────────────────────────────────────────────

class TestInputValidation:
    def test_ui_navigate_requires_user_id(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        with pytest.raises(ValueError, match='user_id is required'):
            ui_navigate(user_id='', screen='KidsHub')

    def test_ui_navigate_requires_screen(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        with pytest.raises(ValueError, match='screen is required'):
            ui_navigate(user_id='u1', screen='')

    def test_ui_overlay_show_requires_user_id(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_show
        with pytest.raises(ValueError, match='user_id is required'):
            ui_overlay_show(user_id='', layout={'type': 'view'})

    def test_ui_overlay_show_rejects_non_dict_layout(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_show
        with pytest.raises(ValueError, match='layout must be a dict'):
            ui_overlay_show(user_id='u1', layout='not a dict')

    def test_ui_overlay_show_rejects_layout_without_type(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_show
        with pytest.raises(ValueError, match="'type' field"):
            ui_overlay_show(user_id='u1', layout={'foo': 'bar'})

    def test_ui_overlay_dismiss_requires_user_id(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_dismiss
        with pytest.raises(ValueError, match='user_id is required'):
            ui_overlay_dismiss(user_id='')


# ─── Topic selection (the C1 regression — multi-device contract) ────────────

class TestTopicSelection:
    def test_ui_navigate_with_device_id_uses_per_device_topic(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='u1', screen='KidsHub', device_id='dev-1')
        assert mock_bus.publish.called
        topic = mock_bus.publish.call_args[0][0]
        assert topic == 'fleet.command'

    def test_ui_navigate_without_device_id_uses_user_fanout_topic(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='u1', screen='KidsHub')
        assert mock_bus.publish.called
        topic = mock_bus.publish.call_args[0][0]
        assert topic == 'fleet.command.user'

    def test_ui_overlay_show_with_device_id_uses_per_device_topic(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_show
        ui_overlay_show(
            user_id='u1',
            layout={'type': 'view'},
            device_id='dev-2',
        )
        topic = mock_bus.publish.call_args[0][0]
        assert topic == 'fleet.command'

    def test_ui_overlay_dismiss_without_device_id_uses_user_fanout_topic(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_dismiss
        ui_overlay_dismiss(user_id='u1')
        topic = mock_bus.publish.call_args[0][0]
        assert topic == 'fleet.command.user'


# ─── Payload shape (the H2 regression — cmd_type key, not 'command') ────────

class TestPayloadShape:
    def test_ui_navigate_payload_has_cmd_type_key(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='u1', screen='KidsHub')
        payload = mock_bus.publish.call_args[0][1]
        assert payload['cmd_type'] == 'ui_navigate'
        assert 'command' not in payload  # H2 regression — never emit 'command' key

    def test_ui_overlay_show_payload_has_cmd_type_key(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_show
        ui_overlay_show(user_id='u1', layout={'type': 'view'})
        payload = mock_bus.publish.call_args[0][1]
        assert payload['cmd_type'] == 'ui_overlay_show'
        assert 'command' not in payload

    def test_ui_overlay_dismiss_payload_has_cmd_type_key(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_dismiss
        ui_overlay_dismiss(user_id='u1')
        payload = mock_bus.publish.call_args[0][1]
        assert payload['cmd_type'] == 'ui_overlay_dismiss'
        assert 'command' not in payload

    def test_payload_includes_generated_command_id(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='u1', screen='KidsHub')
        payload = mock_bus.publish.call_args[0][1]
        assert payload['id'].startswith('ui-')
        assert len(payload['id']) >= 15  # 'ui-' + 12 hex chars

    def test_payload_uses_caller_supplied_command_id(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='u1', screen='KidsHub', command_id='my-custom-id')
        payload = mock_bus.publish.call_args[0][1]
        assert payload['id'] == 'my-custom-id'

    def test_ui_navigate_payload_carries_screen_and_params(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='u1', screen='KidsGame', params={'gameId': 'abc'})
        payload = mock_bus.publish.call_args[0][1]
        assert payload['screen'] == 'KidsGame'
        assert payload['params'] == {'gameId': 'abc'}

    def test_ui_overlay_show_payload_carries_layout_data_agent_name(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_show
        layout = {'type': 'card', 'children': []}
        ui_overlay_show(
            user_id='u1',
            layout=layout,
            data={'title': 'Hi'},
            agent_name='TherapyAgent',
        )
        payload = mock_bus.publish.call_args[0][1]
        assert payload['layout'] == layout
        assert payload['data'] == {'title': 'Hi'}
        assert payload['agent_name'] == 'TherapyAgent'

    def test_ui_overlay_show_defaults_agent_name(self, mock_bus):
        from core.peer_link.ui_commands import ui_overlay_show
        ui_overlay_show(user_id='u1', layout={'type': 'view'})
        payload = mock_bus.publish.call_args[0][1]
        assert payload['agent_name'] == 'Agent'


# ─── Return value on success / failure (M3 regression) ─────────────────────

class TestReturnValues:
    def test_returns_command_id_on_success(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        cid = ui_navigate(user_id='u1', screen='KidsHub')
        assert cid is not None
        assert cid.startswith('ui-')

    def test_returns_none_on_publish_failure(self, mock_bus):
        mock_bus.publish.side_effect = RuntimeError('transport down')
        from core.peer_link.ui_commands import ui_navigate
        cid = ui_navigate(user_id='u1', screen='KidsHub')
        assert cid is None  # caller can detect failure

    def test_overlay_show_returns_none_on_publish_failure(self, mock_bus):
        mock_bus.publish.side_effect = RuntimeError('transport down')
        from core.peer_link.ui_commands import ui_overlay_show
        cid = ui_overlay_show(user_id='u1', layout={'type': 'view'})
        assert cid is None

    def test_overlay_dismiss_returns_none_on_publish_failure(self, mock_bus):
        mock_bus.publish.side_effect = RuntimeError('transport down')
        from core.peer_link.ui_commands import ui_overlay_dismiss
        cid = ui_overlay_dismiss(user_id='u1')
        assert cid is None


# ─── user_id + device_id passthrough to message_bus ────────────────────────

class TestRouting:
    def test_publish_passes_user_id_as_kwarg(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='user-42', screen='KidsHub')
        kwargs = mock_bus.publish.call_args[1]
        assert kwargs['user_id'] == 'user-42'

    def test_publish_passes_device_id_as_kwarg(self, mock_bus):
        from core.peer_link.ui_commands import ui_navigate
        ui_navigate(user_id='user-42', screen='KidsHub', device_id='dev-xyz')
        kwargs = mock_bus.publish.call_args[1]
        assert kwargs['device_id'] == 'dev-xyz'


# ─── Constants exported for test injection ─────────────────────────────────

class TestCmdTypeConstants:
    def test_cmd_type_constants_exported(self):
        from core.peer_link import ui_commands
        assert ui_commands.CMD_UI_NAVIGATE == 'ui_navigate'
        assert ui_commands.CMD_UI_OVERLAY_SHOW == 'ui_overlay_show'
        assert ui_commands.CMD_UI_OVERLAY_DISMISS == 'ui_overlay_dismiss'
