"""Redirect stdout/stderr to devnull in frozen builds.

cx_Freeze frozen builds may have stdout/stderr closed or pointing to
invalid file descriptors. This causes crashes when any library tries to
print (LangChain, autogen, etc.). Redirecting to devnull prevents these
crashes while preserving logging (which uses its own handlers).

Single source of truth — imported by create_recipe.py, reuse_recipe.py,
and hart_intelligence_entry.py instead of copy-pasting the guard.
"""
import os
import sys


def silence_stdio():
    """Redirect stdout/stderr to devnull if they're broken (frozen builds)."""
    try:
        if sys.stdout is None or sys.stdout.closed:
            sys.stdout = open(os.devnull, 'w')
    except Exception:
        sys.stdout = open(os.devnull, 'w')

    try:
        if sys.stderr is None or sys.stderr.closed:
            sys.stderr = open(os.devnull, 'w')
    except Exception:
        sys.stderr = open(os.devnull, 'w')
