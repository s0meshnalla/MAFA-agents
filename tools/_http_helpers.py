"""Shared HTTP helpers for MAFA-B API tools.

Centralises the _fetch_json / _post_json / _put_json / _delete_json / _unwrap
pattern that was previously duplicated across profile_tools, alert_tools,
strategy_tools, and execute_trade_tools.
"""

import os
from typing import Any, Dict

from http_client import get, post, put, delete

API_BASE = os.getenv("BROKER_API_URL", "http://localhost:8080")


# ── Standard error-response builder ──────────────────────────────────────

_STATUS_HINTS = {
    401: "Authentication failed — your session may have expired. Please log in again.",
    403: "Access denied — you may not have permission for this operation.",
    404: "The requested resource was not found on the broker.",
    429: "Rate-limited by the broker API — please wait a moment and retry.",
    500: "The broker API encountered an internal error. Please try again shortly.",
    502: "The broker API is temporarily unreachable (bad gateway).",
    503: "The broker API is temporarily unavailable (maintenance). Try again shortly.",
}


def make_error_response(exc: Exception, context: str = "") -> dict:
    """Build a structured error dict that agents can interpret and relay to users.

    Extracts HTTP status from *requests.RequestException* when available and
    adds a human-friendly hint so the LLM can produce a useful reply.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    hint = _STATUS_HINTS.get(status, f"Broker API request failed: {exc}")
    result: Dict[str, Any] = {"error": hint}
    if context:
        result["context"] = context
    if status:
        result["http_status"] = status
    return result


class AuthError(Exception):
    """Raised when a MAFA-B call returns 401/403 — lets callers handle auth
    separately from transient failures."""
    pass


def raise_on_auth(exc: Exception) -> None:
    """If *exc* wraps a 401/403 HTTP response, raise AuthError instead."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        raise AuthError(_STATUS_HINTS[status]) from exc


# ── HTTP wrappers ────────────────────────────────────────────────────────

def fetch_json(url: str, timeout: int = 10) -> Any:
    """GET a URL and return the parsed JSON body."""
    response = get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def post_json(url: str, body: Dict[str, Any], timeout: int = 15) -> Any:
    """POST JSON to a URL and return the parsed JSON body."""
    response = post(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def put_json(url: str, body: Dict[str, Any], timeout: int = 15) -> Any:
    """PUT JSON to a URL and return the parsed JSON body."""
    response = put(url, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()


def delete_json(url: str, timeout: int = 10) -> Any:
    """DELETE a URL and return the parsed JSON body."""
    response = delete(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def unwrap(payload: Any) -> Any:
    """Extract .data from an ApiResponse wrapper when present."""
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return payload
