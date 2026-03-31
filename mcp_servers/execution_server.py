"""TRUE MCP Server for Trade Execution.

Exposes trade execution, balance checking, and holdings via official MCP stdio protocol.
Run: python mcp_servers/execution_server.py
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

# Import existing tools (these are LangChain StructuredTool objects — use .invoke())
from tools.execute_trade_tools import buy_stock, sell_stock
from tools.profile_tools import get_user_balance, get_user_holdings, get_current_stock_price

# Create TRUE MCP Server
mcp = FastMCP("execution-server")


# ---------------------------------------------------------------------------
# Helper: safely call a LangChain StructuredTool
# ---------------------------------------------------------------------------
import inspect


def _call_tool(tool_obj, params: dict = None):
    """Invoke a LangChain @tool-decorated function safely.

    Calls the underlying function directly via tool_obj.func to bypass
    StructuredTool wrapper issues. Uses inspect.signature to pass only
    accepted parameters.
    """
    fn = getattr(tool_obj, 'func', None)
    if fn is not None and callable(fn):
        if params:
            sig = inspect.signature(fn)
            accepted = {k: v for k, v in params.items() if k in sig.parameters}
            return fn(**accepted)
        return fn()
    # Fallback: try invoke
    return tool_obj.invoke(params or {})


@mcp.tool()
def execute_trade(symbol: str, quantity: int, action: str) -> str:
    """Execute a trade order (buy/sell) with validation.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
        quantity: Number of shares to trade (must be positive integer)
        action: Trade action - either "buy" or "sell"
        
    Returns:
        JSON string with execution result or error message
    """
    action = action.lower().strip()
    symbol = symbol.upper().strip()
    
    if action not in ["buy", "sell"]:
        return json.dumps({
            "error": f"Invalid action '{action}'. Must be 'buy' or 'sell'.",
            "success": False
        })
    
    if quantity <= 0:
        return json.dumps({
            "error": "Quantity must be a positive integer.",
            "success": False
        })
    
    try:
        if action == "buy":
            result = _call_tool(buy_stock, {"symbol": symbol, "quantity": quantity})
        else:
            result = _call_tool(sell_stock, {"symbol": symbol, "quantity": quantity})
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "quantity": quantity,
            "action": action,
            "success": False
        })


@mcp.tool()
def check_balance() -> str:
    """Check user's available trading balance.
    
    Returns:
        JSON string with current balance or error message
    """
    try:
        result = _call_tool(get_user_balance)
        balance = json.loads(result) if isinstance(result, str) else result
        # Handle both raw number and dict responses
        if isinstance(balance, dict):
            bal_val = float(balance.get("balance", balance.get("data", 0)))
        else:
            bal_val = float(balance)
        return json.dumps({
            "balance": bal_val,
            "currency": "USD",
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def check_holdings() -> str:
    """Get user's current stock holdings.
    
    Returns:
        JSON string with holdings dictionary or error message
    """
    try:
        result = _call_tool(get_user_holdings)
        holdings = json.loads(result) if isinstance(result, str) else result
        return json.dumps({
            "holdings": holdings,
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "success": False
        })


@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """Get current stock price for a symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)
        
    Returns:
        JSON string with current price or error message
    """
    symbol = symbol.upper().strip()
    
    try:
        result = _call_tool(get_current_stock_price, {"symbol": symbol})
        price = json.loads(result) if isinstance(result, str) else result
        if isinstance(price, dict):
            price_val = float(price.get("price", price.get("data", 0)))
        else:
            price_val = float(price)
        return json.dumps({
            "symbol": symbol,
            "price": price_val,
            "currency": "USD",
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "success": False
        })


@mcp.tool()
def validate_trade(symbol: str, quantity: int, action: str) -> str:
    """Validate a trade before execution (check balance/holdings).
    
    Args:
        symbol: Stock ticker symbol
        quantity: Number of shares
        action: Trade action - "buy" or "sell"
        
    Returns:
        JSON string with validation result
    """
    action = action.lower().strip()
    symbol = symbol.upper().strip()
    
    validation = {
        "symbol": symbol,
        "quantity": quantity,
        "action": action,
        "valid": True,
        "issues": []
    }
    
    try:
        if action == "buy":
            bal_result = _call_tool(get_user_balance)
            balance_raw = json.loads(bal_result) if isinstance(bal_result, str) else bal_result
            balance = float(balance_raw.get("balance", balance_raw.get("data", balance_raw)) if isinstance(balance_raw, dict) else balance_raw)

            price_result = _call_tool(get_current_stock_price, {"symbol": symbol})
            price_raw = json.loads(price_result) if isinstance(price_result, str) else price_result
            price = float(price_raw.get("price", price_raw.get("data", price_raw)) if isinstance(price_raw, dict) else price_raw)

            total_cost = price * quantity
            
            if total_cost > balance:
                validation["valid"] = False
                validation["issues"].append(
                    f"Insufficient balance. Need ${total_cost:.2f}, have ${balance:.2f}"
                )
            else:
                validation["estimated_cost"] = total_cost
                validation["remaining_balance"] = balance - total_cost
                
        elif action == "sell":
            hold_result = _call_tool(get_user_holdings)
            holdings = json.loads(hold_result) if isinstance(hold_result, str) else hold_result
            # Holdings can be a list of {symbol, quantity, ...} or a dict
            if isinstance(holdings, list):
                current_qty = next(
                    (int(h.get("quantity", 0)) for h in holdings
                     if isinstance(h, dict) and h.get("symbol", "").upper() == symbol),
                    0
                )
            elif isinstance(holdings, dict):
                current_qty = int(holdings.get(symbol, 0))
            else:
                current_qty = 0
            
            if current_qty < quantity:
                validation["valid"] = False
                validation["issues"].append(
                    f"Insufficient shares. Own {current_qty}, trying to sell {quantity}"
                )
            else:
                price_result = _call_tool(get_current_stock_price, {"symbol": symbol})
                price_raw = json.loads(price_result) if isinstance(price_result, str) else price_result
                price = float(price_raw.get("price", price_raw.get("data", price_raw)) if isinstance(price_raw, dict) else price_raw)
                validation["estimated_proceeds"] = price * quantity
        else:
            validation["valid"] = False
            validation["issues"].append(f"Invalid action: {action}")
            
    except Exception as exc:
        validation["valid"] = False
        validation["issues"].append(f"Validation error: {str(exc)}")
    
    validation["success"] = True
    return json.dumps(validation)


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
