"""Trade-execution tools – wired to real MAFA-B endpoints.

MAFA-B contract
────────────────
POST /execute/buy   body: { "quantity": long, "symbol": String }
POST /execute/sell  body: { "quantity": long, "symbol": String }

Both return  TransactionDto { id, type, asset, assetQuantity, amount, createdAt }
"""

import json
import logging
from typing import Any, Dict

from langchain_core.tools import tool
from requests import RequestException

from tools._http_helpers import post_json as _post_json, fetch_json as _fetch_json, unwrap as _unwrap, API_BASE, make_error_response as _err, raise_on_auth, AuthError

logger = logging.getLogger(__name__)


# ── tools ───────────────────────────────────────────────────────────────────

@tool
def buy_stock(symbol: str, quantity: int) -> str:
    """Buy shares of a stock.

    Calls MAFA-B  POST /execute/buy
    Body: { "quantity": <int>, "symbol": "<TICKER>" }
    Returns a TransactionDto JSON string on success, or an error object.
    """
    symbol = symbol.upper().strip()
    if quantity <= 0:
        return json.dumps({"error": "Quantity must be a positive integer.", "symbol": symbol, "quantity": quantity})
    if not symbol.isalpha() or len(symbol) > 5:
        return json.dumps({"error": f"Invalid ticker symbol: '{symbol}'", "symbol": symbol})
    try:
        # Pre-trade safety check: verify sufficient balance
        try:
            bal_payload = _fetch_json(f"{API_BASE}/balance")
            balance = float(_unwrap(bal_payload) or 0)
            price_payload = _fetch_json(f"{API_BASE}/stockprice?symbol={symbol}")
            price = float(price_payload)
            estimated_cost = price * quantity
            if estimated_cost > balance:
                return json.dumps({"error": f"Insufficient balance: estimated cost ${estimated_cost:,.2f} exceeds available ${balance:,.2f}.", "symbol": symbol, "quantity": quantity})
        except Exception:
            pass  # Let the broker do final validation if pre-check fails

        body = {"quantity": quantity, "symbol": symbol}
        result = _post_json(f"{API_BASE}/execute/buy", body)
        if result is None or (isinstance(result, dict) and not result):
            return json.dumps({"error": "Trade was not executed — the broker returned an empty response. This usually means insufficient funds or an unsupported ticker.", "symbol": symbol, "quantity": quantity})
        return json.dumps(result)
    except RequestException as exc:
        raise_on_auth(exc)
        logger.warning("Buy order failed for %dx %s: %s", quantity, symbol, exc)
        return json.dumps(_err(exc, f"buy {quantity}x {symbol}"))


@tool
def sell_stock(symbol: str, quantity: int) -> str:
    """Sell shares of a stock.

    Calls MAFA-B  POST /execute/sell
    Body: { "quantity": <int>, "symbol": "<TICKER>" }
    Returns a TransactionDto JSON string on success, or an error object.
    """
    symbol = symbol.upper().strip()
    if quantity <= 0:
        return json.dumps({"error": "Quantity must be a positive integer.", "symbol": symbol, "quantity": quantity})
    if not symbol.isalpha() or len(symbol) > 5:
        return json.dumps({"error": f"Invalid ticker symbol: '{symbol}'", "symbol": symbol})
    try:
        # Pre-trade safety check: verify sufficient holdings
        try:
            hold_payload = _fetch_json(f"{API_BASE}/holdings")
            holdings = _unwrap(hold_payload)
            owned = 0
            if isinstance(holdings, list):
                for h in holdings:
                    if isinstance(h, dict) and h.get("symbol", "").upper() == symbol:
                        owned = h.get("quantity", 0)
                        break
            if owned < quantity:
                return json.dumps({"error": f"Insufficient shares: you own {owned} of {symbol} but are trying to sell {quantity}.", "symbol": symbol, "quantity": quantity})
        except Exception:
            pass  # Let the broker do final validation if pre-check fails

        body = {"quantity": quantity, "symbol": symbol}
        result = _post_json(f"{API_BASE}/execute/sell", body)
        if result is None or (isinstance(result, dict) and not result):
            return json.dumps({"error": "Trade was not executed — the broker returned an empty response. This usually means you don't own enough shares.", "symbol": symbol, "quantity": quantity})
        return json.dumps(result)
    except RequestException as exc:
        raise_on_auth(exc)
        logger.warning("Sell order failed for %dx %s: %s", quantity, symbol, exc)
        return json.dumps(_err(exc, f"sell {quantity}x {symbol}"))
