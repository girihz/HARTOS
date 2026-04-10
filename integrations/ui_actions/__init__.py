"""ui_actions — registry + tools for in-app navigation and action surfacing.

The agent chat screen in Nunba/Hevolve has a companion "LiquidUI action bar"
that shows context-aware quick-actions pointing at other app pages. This
package owns:

- page_registry: single source of truth for what pages exist and which
  natural-language phrases map to them
- navigate_tool: LangChain tool that resolves a user phrase to a
  concrete ui_action dict the frontend can execute (navigate to route,
  open overlay, etc.)

Having the registry on the backend (not frontend) means the LLM sees the
available destinations as part of its tool context, so 'take me to social'
lands deterministically instead of becoming text chatter.
"""
from .page_registry import (
    PageEntry,
    PAGE_REGISTRY,
    resolve_page,
    list_pages,
)

__all__ = ['PageEntry', 'PAGE_REGISTRY', 'resolve_page', 'list_pages']
