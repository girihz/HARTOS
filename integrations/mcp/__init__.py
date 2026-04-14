"""MCP (Model Context Protocol) Integration

Public API surface for cross-package consumers (Nunba, third-party clients).

Token management:
  - get_mcp_token()        — return current bearer token (idempotent, cached)
  - rotate_mcp_token()     — generate + persist a new token, invalidate cache
  - get_mcp_token_path()   — on-disk path to the token file

Underscore-prefixed names (`_ensure_mcp_token`, `_mcp_token_path`) are
INTERNAL and may move/rename without notice.  New code MUST use the
public names above.
"""
from .mcp_integration import (
    MCPServerConnector,
    MCPToolRegistry,
    get_mcp_tools_for_autogen,
    load_user_mcp_servers,
    mcp_registry,
)
from .mcp_http_bridge import (
    auto_register_local_mcp,
    get_mcp_token,
    get_mcp_token_path,
    mcp_local_bp,
    rotate_mcp_token,
)

__all__ = [
    # Tool registry
    'MCPServerConnector', 'MCPToolRegistry', 'load_user_mcp_servers',
    'get_mcp_tools_for_autogen', 'mcp_registry',
    # HTTP bridge
    'mcp_local_bp', 'auto_register_local_mcp',
    # Token management — public contract for Nunba/cross-package use
    'get_mcp_token', 'rotate_mcp_token', 'get_mcp_token_path',
]
