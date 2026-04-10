"""
Agentic UI Commands — single source for all HARTOS→phone UI dispatches.

ANY HARTOS agent (speech therapy, marketing, learning, recipe, fleet, etc.)
can use these helpers to drive the phone UI. All calls go through the
existing fleet.command MessageBus topic family — the single dispatch channel
for every device command (TTS, consent, navigation, overlays, game dispatch).

Topic selection (automatic based on targeting):
  - device_id provided → 'fleet.command'       (per-device topic)
  - device_id omitted  → 'fleet.command.user'  (fan-out to all user's devices)

Single responsibility:
  - Formats the payload with the correct cmd_type so the RN handler's
    registry can route to the matching handler in fleetCommandHandler.js.
  - Selects the right topic template (per-device vs user-fanout) based on
    whether the caller specified a device_id.
  - Does NOT own its own transport or state — delegates to
    get_message_bus().publish(...).

No parallel paths:
  - Uses the same fleet.command topic family already subscribed by
    AutobahnConnectionManager.
  - Uses the same FCM fallback already handled by MyFirebaseMessagingService.
  - Uses the same HTTP polling fallback in pollFleetCommands.

IMPORTANT — when NOT to use ui_overlay_show:
  Consent-class interactions (grant microphone access, approve agent action,
  etc.) MUST use the cmd_type 'agent_consent' instead of 'ui_overlay_show'.
  The agent_consent path has countdown, auto-deny, ack semantics, TV D-pad
  support, and accessibility — none of which the generic overlay has. A
  helper for agent_consent lives in this module's future work; until then,
  call get_message_bus().publish('fleet.command', {cmd_type: 'agent_consent',
  ...}) directly.

Layout theming:
  Layouts passed to ui_overlay_show are rendered via SocialLiquidUI → which
  wraps ServerDrivenUI with socialTokens theme. If your agent needs a
  different theme, inject theme tokens via the layout's style.

Usage:
    from core.peer_link.ui_commands import ui_navigate, ui_overlay_show

    # Targeting a specific device (exact delivery)
    cid = ui_navigate(user_id='10077', device_id='dev-abc',
                       screen='KidsGame', params={'gameId': 'abc'})
    if cid is None:
        log.warning("UI command failed to publish")

    # Fan-out to all of user's devices (phone + tablet + watch)
    ui_overlay_show(
        user_id='10077',
        layout={'type': 'card', 'children': [...]},
        data={'gameTitle': 'Balloon Pop'},
        agent_name='SpeechTherapyAgent',
    )
"""

import logging
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger('hevolve.ui_commands')


# ─── cmd_type constants (single source of truth, mirror RN COMMAND_HANDLERS) ──
# If you add a new cmd_type here, add the matching handler in:
#   services/fleetCommandHandler.js → COMMAND_HANDLERS registry
CMD_UI_NAVIGATE = 'ui_navigate'
CMD_UI_OVERLAY_SHOW = 'ui_overlay_show'
CMD_UI_OVERLAY_DISMISS = 'ui_overlay_dismiss'


def _publish(
    cmd_type: str,
    params: Dict[str, Any],
    user_id: str,
    device_id: str = '',
    command_id: Optional[str] = None,
) -> Optional[str]:
    """Internal: format and publish a fleet.command for the UI bridge.

    Returns:
        The command_id on success, or None if the publish failed.
        Callers MUST check the return value — a None means the phone
        will never receive this command.
    """
    if not user_id:
        raise ValueError("user_id is required for UI commands (per-user scoping)")

    cid = command_id or f"ui-{uuid.uuid4().hex[:12]}"
    payload = {
        'cmd_type': cmd_type,
        'id': cid,
        **params,
    }

    # Select topic: per-device when device_id specified, else user-fanout
    topic = 'fleet.command' if device_id else 'fleet.command.user'

    try:
        from core.peer_link.message_bus import get_message_bus
        get_message_bus().publish(
            topic,
            payload,
            user_id=user_id,
            device_id=device_id,
        )
        logger.debug("UI command published: %s topic=%s id=%s user=%s device=%s",
                     cmd_type, topic, cid, user_id, device_id or '*')
        return cid
    except Exception as e:
        logger.warning("UI command publish failed (%s topic=%s): %s",
                       cmd_type, topic, e)
        return None


def ui_navigate(
    user_id: str,
    screen: str,
    params: Optional[Dict[str, Any]] = None,
    device_id: str = '',
    command_id: Optional[str] = None,
) -> Optional[str]:
    """Navigate the phone to a specific screen.

    The RN handler validates the screen name against an allowlist — screens
    not in the allowlist will be rejected with an ack failure. The allowlist
    is maintained in services/fleetCommandHandler.js (NAVIGATION_ALLOWLIST).

    Args:
        user_id: User whose device(s) should receive the command.
        screen: Screen name as registered in home.routes.js (e.g. 'KidsGame',
            'Encounters', 'CommunityDetail').
        params: Screen params passed to React Navigation (e.g. {'gameId': 'abc'}).
        device_id: Target specific device. Empty string = fan out to all
            user's devices via 'fleet.command.user' topic.
        command_id: Optional caller-supplied id for ack correlation.

    Returns:
        The command_id on successful publish, or None on failure.
    """
    if not screen:
        raise ValueError("screen is required for ui_navigate")

    return _publish(
        CMD_UI_NAVIGATE,
        {'screen': screen, 'params': params or {}},
        user_id=user_id,
        device_id=device_id,
        command_id=command_id,
    )


def ui_overlay_show(
    user_id: str,
    layout: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
    agent_name: str = 'Agent',
    device_id: str = '',
    command_id: Optional[str] = None,
) -> Optional[str]:
    """Float an overlay on the phone with a ServerDrivenUI layout.

    The layout JSON is rendered by the RN app's LiquidOverlay →
    SocialLiquidUI → ServerDrivenUI chain. SocialLiquidUI injects
    socialTokens theme, so layouts pick up the social color palette by
    default. The raw JSON fallback in LiquidOverlay handles layouts that
    fail to render through SocialLiquidUI.

    NOT FOR CONSENT: see module docstring. Use 'agent_consent' cmd_type
    instead for permission / approval flows.

    NOT SHOWN ON TV: the RN handler rejects this on TV devices (the
    LiquidOverlay component is not mounted on TV per CommunityView.js).
    Use ui_navigate to a dedicated screen for TV agents.

    Args:
        user_id: User whose device(s) should receive the command.
        layout: ServerDrivenUI JSON descriptor (type, props, style, children,
            bind, action, visible). Max depth 20. Must be a dict with a
            'type' field.
        data: Context object for data bindings (e.g. {'gameTitle': 'Pong'}).
        agent_name: Display name shown in the overlay header.
        device_id: Target specific device. Empty string = fan out to all
            user's devices via 'fleet.command.user' topic.
        command_id: Optional caller-supplied id for ack correlation.

    Returns:
        The command_id on successful publish, or None on failure.
    """
    if not isinstance(layout, dict) or 'type' not in layout:
        raise ValueError("layout must be a dict with a 'type' field (ServerDrivenUI schema)")

    return _publish(
        CMD_UI_OVERLAY_SHOW,
        {
            'layout': layout,
            'data': data or {},
            'agent_name': agent_name,
        },
        user_id=user_id,
        device_id=device_id,
        command_id=command_id,
    )


def ui_overlay_dismiss(
    user_id: str,
    device_id: str = '',
    command_id: Optional[str] = None,
) -> Optional[str]:
    """Dismiss the active overlay on the phone.

    Args:
        user_id: User whose device(s) should receive the command.
        device_id: Target specific device. Empty string = fan out to all
            user's devices via 'fleet.command.user' topic.
        command_id: Optional caller-supplied id for ack correlation.

    Returns:
        The command_id on successful publish, or None on failure.
    """
    return _publish(
        CMD_UI_OVERLAY_DISMISS,
        {},
        user_id=user_id,
        device_id=device_id,
        command_id=command_id,
    )
