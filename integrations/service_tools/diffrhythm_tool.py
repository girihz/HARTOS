"""
DiffRhythm tool wrapper — AI singing voice synthesis.

Service: DiffRhythm (singing voice, vocal synthesis with lyrics)
Default port: 8002
VRAM: 4GB GPU required
"""

from .registry import ServiceToolInfo, service_tool_registry


class DiffRhythmTool:
    """Thin wrapper to register DiffRhythm with the ServiceToolRegistry."""

    DEFAULT_URL = "http://localhost:8002"

    @classmethod
    def create_tool_info(cls, base_url: str = None) -> ServiceToolInfo:
        return ServiceToolInfo(
            name="diffrhythm",
            description=(
                "AI singing voice synthesis. Generates sung vocals from lyrics "
                "with melody, rhythm, and vocal style control. Produces natural "
                "singing audio from text lyrics and style prompts."
            ),
            base_url=base_url or cls.DEFAULT_URL,
            endpoints={
                "generate": {
                    "path": "/generate",
                    "method": "POST",
                    "description": (
                        "Generate singing audio from lyrics. "
                        "Input: JSON with 'lyrics' (text to sing), "
                        "'style' (vocal style: pop/opera/folk/etc), "
                        "'tempo' (BPM), 'duration' (seconds). "
                        "Returns audio URL when complete."
                    ),
                    "params_schema": {
                        "lyrics": {"type": "string", "description": "Lyrics to sing"},
                        "style": {"type": "string", "description": "Vocal style (pop, opera, folk, ballad)", "default": "pop"},
                        "tempo": {"type": "integer", "description": "BPM tempo", "default": 120},
                        "duration": {"type": "integer", "description": "Duration in seconds", "default": 30},
                    },
                },
            },
            health_endpoint="/health",
            tags=["singing", "vocals", "audio", "generation", "music"],
            timeout=120,
        )

    @classmethod
    def register(cls, base_url: str = None) -> bool:
        """Register DiffRhythm with the global service_tool_registry."""
        tool_info = cls.create_tool_info(base_url)
        return service_tool_registry.register_tool(tool_info)
