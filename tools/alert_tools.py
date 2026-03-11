"""Alert tools – wired to MAFA-B AlertController.

MAFA-B contract
────────────────
POST   /alerts            body: AlertRequestDto → ApiResponse { data: AlertResponseDto }
GET    /alerts?status=X   → ApiResponse { data: List<AlertResponseDto> }
DELETE /alerts/{id}       → ApiResponse { data: AlertResponseDto }  (soft-delete → CANCELLED)

AlertRequestDto:
    symbol:      String (ticker)
    condition:   enum ABOVE | BELOW
    targetPrice: Double
    channel:     enum IN_APP | USER

AlertResponseDto:
    id, symbol, condition, targetPrice, status, channel, createdAt
    status enum: ACTIVE | TRIGGERED | CANCELLED
"""

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from requests import RequestException

from tools._http_helpers import fetch_json as _fetch_json, post_json as _post_json, delete_json as _delete_json, unwrap as _unwrap, API_BASE, make_error_response as _err

logger = logging.getLogger(__name__)


# ── tools ───────────────────────────────────────────────────────────────────

@tool
def create_alert(symbol: str, condition: str, target_price: float, channel: str = "IN_APP") -> str:
    """Create a price alert for a stock.

    Params:
        symbol:       ticker (e.g. "AAPL")
        condition:    "ABOVE" or "BELOW"
        target_price: the price threshold
        channel:      "IN_APP" (default) or "USER"

    Calls MAFA-B  POST /alerts
    Body: { "symbol": "AAPL", "condition": "ABOVE", "targetPrice": 200.0, "channel": "IN_APP" }
    Returns JSON: AlertResponseDto {id, symbol, condition, targetPrice, status, channel, createdAt}
    """
    symbol = symbol.upper().strip()
    condition = condition.upper().strip()
    channel = channel.upper().strip()

    if condition not in ("ABOVE", "BELOW"):
        return json.dumps({"error": f"Invalid condition '{condition}'. Use ABOVE or BELOW."})
    if channel not in ("IN_APP", "USER"):
        return json.dumps({"error": f"Invalid channel '{channel}'. Use IN_APP or USER."})
    if target_price <= 0:
        return json.dumps({"error": "target_price must be positive."})

    try:
        body = {
            "symbol": symbol,
            "condition": condition,
            "targetPrice": target_price,
            "channel": channel,
        }
        payload = _post_json(f"{API_BASE}/alerts", body)
        data = _unwrap(payload)
        return json.dumps(data, default=str) if data else json.dumps({"error": "Alert creation failed"})
    except RequestException as exc:
        logger.warning(f"Error creating alert for {symbol}: {exc}")
        return json.dumps(_err(exc, f"create alert for {symbol}"))


@tool
def get_alerts(status: Optional[str] = None) -> str:
    """Fetch the user's price alerts, optionally filtered by status.

    Params:
        status: (optional) "ACTIVE" | "TRIGGERED" | "CANCELLED"
                If omitted, returns all alerts.

    Calls MAFA-B  GET /alerts?status=X
    Returns JSON: List<AlertResponseDto> [{id, symbol, condition, targetPrice, status, channel, createdAt}, ...]
    """
    try:
        url = f"{API_BASE}/alerts"
        if status:
            url += f"?status={status.upper().strip()}"
        payload = _fetch_json(url)
        data = _unwrap(payload)
        return json.dumps(data, default=str) if isinstance(data, list) else json.dumps([])
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching alerts: {exc}")
        return json.dumps(_err(exc, "get alerts"))


@tool
def delete_alert(alert_id: int) -> str:
    """Delete (cancel) a price alert by its ID.

    Calls MAFA-B  DELETE /alerts/{id}
    The alert is soft-deleted (status set to CANCELLED).
    Returns JSON: the cancelled AlertResponseDto.
    """
    try:
        payload = _delete_json(f"{API_BASE}/alerts/{alert_id}")
        data = _unwrap(payload)
        return json.dumps(data, default=str) if data else json.dumps({"error": "Delete failed"})
    except RequestException as exc:
        if hasattr(exc, 'response') and exc.response is not None and exc.response.status_code == 404:
            return json.dumps({"error": f"Alert {alert_id} not found"})
        logger.warning(f"Error deleting alert {alert_id}: {exc}")
        return json.dumps(_err(exc, f"delete alert {alert_id}"))
