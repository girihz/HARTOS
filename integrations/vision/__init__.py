"""
Vision Sidecar — packages MiniCPM + frame handling for desktop apps.

Replaces external Redis with in-process FrameStore.
Manages MiniCPM as a subprocess sidecar with auto-download.
Provides continuous scene descriptions for the embodied AI agent.
"""
from .frame_store import FrameStore
from .vision_service import VisionService

__all__ = ['FrameStore', 'VisionService', 'get_vision_service']


_vision_service_singleton: 'VisionService | None' = None


def get_vision_service() -> VisionService:
    """Return the process-wide VisionService singleton.

    Callers (admin toggle, agent approval handler, parse_visual_context
    tool) should always go through this helper so we never end up with
    two VisionService instances fighting over the WebSocket port and
    the MiniCPM sidecar subprocess. Lazy-instantiates on first call —
    actual start/stop is still explicit via .start() / .stop().
    """
    global _vision_service_singleton
    if _vision_service_singleton is None:
        _vision_service_singleton = VisionService()
    return _vision_service_singleton
