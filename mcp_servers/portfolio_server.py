"""TRUE MCP Server for Portfolio Management.

Exposes portfolio analysis, holdings tracking, and metrics via official MCP stdio protocol.
Run: python mcp_servers/portfolio_server.py
"""

from __future__ import annotations

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# Bridge auth token from parent process → this subprocess
from http_client import set_request_token
_auth_token = os.environ.get("MAFA_AUTH_TOKEN")
if _auth_token:
    set_request_token(_auth_token)

# Import existing tools (LangChain StructuredTool objects — use .invoke())
from tools.profile_tools import get_user_holdings, get_current_stock_price, get_user_balance

# Create TRUE MCP Server
mcp = FastMCP("portfolio-server")


# ---------------------------------------------------------------------------
# Helper: safely call a LangChain StructuredTool
# ---------------------------------------------------------------------------
import inspect


def _call_tool(tool_obj, params: dict | None = None):
    """Invoke a LangChain @tool-decorated function safely.

    Calls the underlying function directly to bypass StructuredTool
    wrapper issues.
    """
    fn = getattr(tool_obj, 'func', None)
    if fn is not None and callable(fn):
        if params:
            sig = inspect.signature(fn)
            accepted = {k: v for k, v in params.items() if k in sig.parameters}
            return fn(**accepted)
        return fn()
    return tool_obj.invoke(params or {})


def _parse_holdings(raw) -> dict:
    """Parse holdings result into a dict of {symbol: quantity}.

    The broker returns List<Share> [{symbol, quantity, price, id}, ...].
    We convert this into {symbol: quantity} for portfolio calculations.
    """
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, list):
        # List of Share objects: [{symbol, quantity, price, id}, ...]
        return {
            item.get("symbol", ""): item.get("quantity", 0)
            for item in data
            if isinstance(item, dict) and item.get("symbol")
        }
    if isinstance(data, dict):
        if data.get("error"):
            raise ValueError(str(data.get("error")))
        # Could be {"data": [...]} wrapper
        inner = data.get("data", data)
        if isinstance(inner, dict) and inner.get("error"):
            raise ValueError(str(inner.get("error")))
        if isinstance(inner, list):
            return {
                item.get("symbol", ""): item.get("quantity", 0)
                for item in inner
                if isinstance(item, dict) and item.get("symbol")
            }
        return {}
    return {}


def _parse_balance(raw) -> float:
    """Parse balance result into a float."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        if data.get("error"):
            raise ValueError(str(data.get("error")))
        inner = data.get("data", data.get("balance", 0))
        if isinstance(inner, dict) and inner.get("error"):
            raise ValueError(str(inner.get("error")))
        return float(inner if not isinstance(inner, dict) else inner.get("balance", 0))
    return float(data)


def _parse_price(raw) -> float:
    """Parse price result into a float."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        if data.get("error"):
            raise ValueError(str(data.get("error")))
        inner = data.get("data", data.get("price", 0))
        if isinstance(inner, dict) and inner.get("error"):
            raise ValueError(str(inner.get("error")))
        return float(inner if not isinstance(inner, dict) else inner.get("price", 0))
    return float(data)


@mcp.tool()
def get_portfolio_snapshot() -> str:
    """Get complete portfolio snapshot with current values.
    
    Returns:
        JSON string with portfolio holdings, values, and totals
    """
    try:
        holdings = _parse_holdings(_call_tool(get_user_holdings))
        balance = _parse_balance(_call_tool(get_user_balance))
        
        portfolio = {
            "holdings": [],
            "cash_balance": balance,
            "total_invested": 0.0,
            "total_portfolio_value": balance,
            "success": True
        }
        
        for symbol, quantity in holdings.items():
            quantity_int = int(quantity)
            try:
                price = _parse_price(_call_tool(get_current_stock_price, {"symbol": symbol}))
                value = price * quantity_int
                portfolio["holdings"].append({
                    "symbol": symbol,
                    "quantity": quantity_int,
                    "current_price": price,
                    "market_value": value
                })
                portfolio["total_invested"] += value
                portfolio["total_portfolio_value"] += value
            except Exception:
                portfolio["holdings"].append({
                    "symbol": symbol,
                    "quantity": quantity_int,
                    "current_price": None,
                    "market_value": None,
                    "error": "Could not fetch price"
                })
        
        return json.dumps(portfolio)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def get_portfolio_allocation() -> str:
    """Get portfolio allocation percentages by holding.
    
    Returns:
        JSON string with allocation breakdown
    """
    try:
        holdings = _parse_holdings(_call_tool(get_user_holdings))
        balance = _parse_balance(_call_tool(get_user_balance))
        
        allocations = []
        total_value = balance
        
        for symbol, quantity in holdings.items():
            try:
                price = _parse_price(_call_tool(get_current_stock_price, {"symbol": symbol}))
                value = price * int(quantity)
                total_value += value
                allocations.append({
                    "symbol": symbol,
                    "quantity": int(quantity),
                    "value": value
                })
            except Exception:
                pass
        
        result = {
            "allocations": [],
            "cash_percentage": (balance / total_value * 100) if total_value > 0 else 100,
            "total_value": total_value,
            "success": True
        }
        
        for alloc in allocations:
            percentage = (alloc["value"] / total_value * 100) if total_value > 0 else 0
            result["allocations"].append({
                "symbol": alloc["symbol"],
                "quantity": alloc["quantity"],
                "value": alloc["value"],
                "percentage": round(percentage, 2)
            })
        
        return json.dumps(result)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def analyze_portfolio_risk() -> str:
    """Analyze portfolio risk metrics and concentration.
    
    Returns:
        JSON string with risk analysis
    """
    try:
        holdings = _parse_holdings(_call_tool(get_user_holdings))
        balance = _parse_balance(_call_tool(get_user_balance))
        
        positions = []
        total_invested = 0.0
        
        for symbol, quantity in holdings.items():
            try:
                price = _parse_price(_call_tool(get_current_stock_price, {"symbol": symbol}))
                value = price * int(quantity)
                positions.append({"symbol": symbol, "value": value})
                total_invested += value
            except Exception:
                pass
        
        total_portfolio = total_invested + balance
        
        risk_analysis = {
            "total_portfolio_value": total_portfolio,
            "cash_percentage": (balance / total_portfolio * 100) if total_portfolio > 0 else 100,
            "invested_percentage": (total_invested / total_portfolio * 100) if total_portfolio > 0 else 0,
            "position_count": len(positions),
            "concentration_risk": "low",
            "warnings": [],
            "success": True
        }
        
        for pos in positions:
            pct = (pos["value"] / total_portfolio * 100) if total_portfolio > 0 else 0
            if pct > 25:
                risk_analysis["concentration_risk"] = "high"
                risk_analysis["warnings"].append(
                    f"{pos['symbol']} represents {pct:.1f}% of portfolio (>25%)"
                )
            elif pct > 15:
                if risk_analysis["concentration_risk"] != "high":
                    risk_analysis["concentration_risk"] = "medium"
                risk_analysis["warnings"].append(
                    f"{pos['symbol']} represents {pct:.1f}% of portfolio (>15%)"
                )
        
        cash_pct = risk_analysis["cash_percentage"]
        if cash_pct < 5:
            risk_analysis["warnings"].append(
                f"Low cash reserves ({cash_pct:.1f}%). Consider maintaining 5-10%."
            )
        elif cash_pct > 50:
            risk_analysis["warnings"].append(
                f"High cash allocation ({cash_pct:.1f}%). Consider investing."
            )
        
        return json.dumps(risk_analysis)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def check_position(symbol: str) -> str:
    """Check if user has a position in a specific stock.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        JSON string with position details
    """
    symbol = symbol.upper().strip()
    
    try:
        holdings = _parse_holdings(_call_tool(get_user_holdings))
        quantity = int(holdings.get(symbol, 0))
        
        result = {
            "symbol": symbol,
            "has_position": quantity > 0,
            "quantity": quantity,
            "success": True
        }
        
        if quantity > 0:
            try:
                price = _parse_price(_call_tool(get_current_stock_price, {"symbol": symbol}))
                result["current_price"] = price
                result["market_value"] = price * quantity
            except Exception:
                pass
        
        return json.dumps(result)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "success": False
        })


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
