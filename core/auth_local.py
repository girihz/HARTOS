"""Localhost-or-token decorator for HARTOS admin/control routes.

Port of the canonical Nunba `routes/auth.py:require_local_or_token`
pattern into HARTOS, where multiple endpoints (vlm_stop, prompts/sync,
diag) need the same "trusted localhost OR valid Bearer token" semantic.

Why this pattern instead of plain @require_auth:
    Nunba's bundled install runs HARTOS on 127.0.0.1:5000 and has the
    desktop tray (Tk indicator window, Python `app.py` / `main.py`)
    POST to /api/vlm/stop directly — without a logged-in JWT context.
    Adding plain @require_auth would break that user flow on every
    Stop-button click.  This decorator preserves the local-trust UX
    while still rejecting remote unauthenticated callers.

Threat model coverage:
    ✓ Remote attacker on the LAN — rejected (remote_addr != localhost)
    ✓ Remote attacker via DNS rebind — rejected (post-rebind remote_addr
      is still the attacker's IP, not localhost)
    ✓ Browser CSRF from same-origin localhost page — accepted (correct;
      that's the intended Nunba SPA flow)
    ✗ Browser CSRF from cross-origin page targeting localhost — partial.
      The browser submits with remote_addr=127.0.0.1 (because the
      browser IS local).  Add an Origin/Referer check at the route
      level for endpoints handling destructive actions if this matters.

Env vars:
    HARTOS_API_TOKEN — optional shared secret.  When set, callers may
    send `Authorization: Bearer <token>` to bypass the localhost check.
    Used by remote ops tooling and inter-node admin calls.

    TRUSTED_PROXY — when HARTOS sits behind a reverse proxy (nginx,
    Traefik), all requests appear as remote_addr=127.0.0.1 by default.
    Setting this env to the proxy's address makes the decorator inspect
    X-Forwarded-For instead.  Without it, only direct-connection
    remote_addr is trusted (safe default).
"""
from __future__ import annotations

import hmac
import os
from functools import wraps

from flask import jsonify, request

# Read once at import time (not per-request) so token rotation requires
# a HARTOS restart — same model as Nunba.
API_TOKEN = os.environ.get('HARTOS_API_TOKEN', '')


def _is_local_request() -> bool:
    """True if the request is from localhost, honouring TRUSTED_PROXY."""
    trusted_proxy = os.environ.get('TRUSTED_PROXY', '')
    if trusted_proxy and request.remote_addr == trusted_proxy:
        forwarded_for = (request.headers.get('X-Forwarded-For', '')
                         .split(',')[0].strip())
        return forwarded_for in ('127.0.0.1', '::1', 'localhost')
    return request.remote_addr in ('127.0.0.1', '::1')


def require_local_or_token(f):
    """Allow localhost callers; require Bearer token for remote callers.

    Returns 401 with a clear message when neither condition holds — the
    error body is JSON to match the rest of the HARTOS API surface so
    the React SPA can surface it via its existing error toast pipeline.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if _is_local_request():
            return f(*args, **kwargs)
        if API_TOKEN:
            auth = request.headers.get('Authorization', '')
            if auth.startswith('Bearer '):
                token = auth[7:]
                # hmac.compare_digest is constant-time — defends against
                # timing-oracle leaks on the token comparison.
                if hmac.compare_digest(token, API_TOKEN):
                    return f(*args, **kwargs)
        return jsonify({
            'error': 'unauthorized',
            'message': ('This endpoint requires local access or a '
                        'valid HARTOS_API_TOKEN bearer header.'),
        }), 401
    return decorated
