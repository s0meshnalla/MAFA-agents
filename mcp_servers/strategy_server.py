"""TRUE MCP Server for Investment Strategy.

Exposes investment recommendations, risk assessment, and rebalancing via official MCP stdio protocol.
Run: python mcp_servers/strategy_server.py
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
from tools.investment_strategy_tools import (
    assess_risk_tolerance,
    analyze_portfolio_alignment,
    generate_personalized_strategy,
    calculate_optimal_allocation,
    track_strategy_adherence,
)
from tools.profile_tools import get_user_holdings, get_current_stock_price, get_user_balance
from tools.market_research_tools import predict

# Create TRUE MCP Server
mcp = FastMCP("strategy-server")

# Valid tickers for predictions
VALID_TICKERS = ["AAPL", "AMZN", "ADBE", "GOOGL", "IBM", "JPM", "META", "MSFT", "NVDA", "ORCL", "TSLA"]


# ---------------------------------------------------------------------------
# Helper: safely call a LangChain StructuredTool
# ---------------------------------------------------------------------------
import inspect


def _call_tool(tool_obj, params: dict = None):
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
    """
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, list):
        return {
            item.get("symbol", ""): item.get("quantity", 0)
            for item in data
            if isinstance(item, dict) and item.get("symbol")
        }
    if isinstance(data, dict):
        inner = data.get("data", data)
        if isinstance(inner, list):
            return {
                item.get("symbol", ""): item.get("quantity", 0)
                for item in inner
                if isinstance(item, dict) and item.get("symbol")
            }
        return inner
    return {}


def _parse_balance(raw) -> float:
    """Parse balance result into a float."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        return float(data.get("balance", data.get("data", 0)))
    return float(data)


def _parse_price(raw) -> float:
    """Parse price result into a float."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    if isinstance(data, dict):
        return float(data.get("price", data.get("data", 0)))
    return float(data)


@mcp.tool()
def generate_investment_recommendation(symbol: str, user_id: int) -> str:
    """Generate a complete investment recommendation for a symbol.
    
    Combines risk profile, portfolio fit, and price prediction to produce
    a buy/sell/hold recommendation with confidence and reasoning.
    
    Args:
        symbol: Stock ticker symbol
        user_id: User ID for personalization
        
    Returns:
        JSON string with recommendation details
    """
    symbol = symbol.upper().strip()
    
    recommendation = {
        "symbol": symbol,
        "user_id": user_id,
        "action": "hold",
        "confidence": 0.5,
        "position_size_pct": 0.0,
        "reasoning": [],
        "success": True
    }
    
    try:
        # 1. Assess user risk tolerance
        risk_result = _call_tool(assess_risk_tolerance, {"user_id": user_id})
        risk_data = json.loads(risk_result) if isinstance(risk_result, str) else risk_result
        risk_level = risk_data.get("risk_level", "moderate") if isinstance(risk_data, dict) else "moderate"
        recommendation["risk_profile"] = risk_level
        recommendation["reasoning"].append(f"Risk profile: {risk_level}")
        
        # 2. Check portfolio alignment
        alignment_result = _call_tool(analyze_portfolio_alignment, {"user_id": user_id, "strategy_id": None})
        alignment_data = json.loads(alignment_result) if isinstance(alignment_result, str) else alignment_result
        recommendation["portfolio_alignment"] = alignment_data
        
        # 3. Get price prediction if available
        if symbol in VALID_TICKERS:
            try:
                pred_result = _call_tool(predict, {"symbol": symbol})
                pred_data = json.loads(pred_result) if isinstance(pred_result, str) else pred_result
                if isinstance(pred_data, dict):
                    predicted_price = float(pred_data.get("predicted_close", pred_data.get("data", pred_data.get("prediction", 0))))
                else:
                    predicted_price = float(pred_data)

                price_result = _call_tool(get_current_stock_price, {"symbol": symbol})
                current_price = _parse_price(price_result)

                price_change_pct = ((predicted_price - current_price) / current_price) * 100 if current_price > 0 else 0
                
                recommendation["prediction"] = {
                    "current_price": current_price,
                    "predicted_price": predicted_price,
                    "expected_change_pct": round(price_change_pct, 2)
                }
                
                if price_change_pct > 3:
                    recommendation["action"] = "buy"
                    recommendation["confidence"] = min(0.85, 0.6 + price_change_pct / 20)
                    recommendation["reasoning"].append(f"Bullish: Predicted {price_change_pct:.1f}% upside")
                elif price_change_pct < -3:
                    recommendation["action"] = "sell"
                    recommendation["confidence"] = min(0.85, 0.6 + abs(price_change_pct) / 20)
                    recommendation["reasoning"].append(f"Bearish: Predicted {price_change_pct:.1f}% downside")
                else:
                    recommendation["action"] = "hold"
                    recommendation["confidence"] = 0.6
                    recommendation["reasoning"].append(f"Neutral: Predicted {price_change_pct:.1f}% change")
                    
            except Exception as exc:
                recommendation["reasoning"].append(f"Prediction unavailable: {str(exc)}")
        else:
            recommendation["reasoning"].append(f"No LSTM model available for {symbol}")
        
        # 4. Calculate position size based on risk
        risk_multiplier = {"conservative": 0.03, "moderate": 0.05, "aggressive": 0.08}.get(risk_level, 0.05)
        recommendation["position_size_pct"] = risk_multiplier * 100
        recommendation["reasoning"].append(
            f"Suggested position size: {recommendation['position_size_pct']:.1f}% of portfolio"
        )
        
    except Exception as exc:
        recommendation["error"] = str(exc)
        recommendation["success"] = False
    
    return json.dumps(recommendation)


@mcp.tool()
def get_risk_profile(user_id: int) -> str:
    """Get user's risk tolerance profile.
    
    Args:
        user_id: User ID
        
    Returns:
        JSON string with risk profile details
    """
    try:
        result = _call_tool(assess_risk_tolerance, {"user_id": user_id})
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "success": False
        })


@mcp.tool()
def portfolio_rebalancing_proposal(user_id: int) -> str:
    """Generate a complete portfolio rebalancing proposal.
    
    Analyzes current allocation vs target and suggests trades.
    
    Args:
        user_id: User ID
        
    Returns:
        JSON string with rebalancing trades and reasoning
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
                positions.append({
                    "symbol": symbol,
                    "quantity": int(quantity),
                    "value": value,
                    "price": price
                })
                total_invested += value
            except Exception:
                pass
        
        total_portfolio = total_invested + balance
        
        # Get target allocation
        alloc_result = _call_tool(calculate_optimal_allocation, {"user_id": user_id, "strategy_type": "balanced"})
        allocation_data = json.loads(alloc_result) if isinstance(alloc_result, str) else alloc_result
        
        proposal = {
            "user_id": user_id,
            "current_portfolio_value": total_portfolio,
            "cash_balance": balance,
            "current_positions": positions,
            "suggested_trades": [],
            "reasoning": [],
            "success": True
        }
        
        for pos in positions:
            current_pct = (pos["value"] / total_portfolio * 100) if total_portfolio > 0 else 0
            if current_pct > 20:
                trim_pct = current_pct - 15
                trim_value = total_portfolio * (trim_pct / 100)
                trim_shares = int(trim_value / pos["price"]) if pos["price"] > 0 else 0
                
                if trim_shares > 0:
                    proposal["suggested_trades"].append({
                        "action": "sell",
                        "symbol": pos["symbol"],
                        "quantity": trim_shares,
                        "reason": f"Reduce concentration from {current_pct:.1f}% to ~15%"
                    })
                    proposal["reasoning"].append(f"{pos['symbol']} is overweight at {current_pct:.1f}%")
        
        cash_pct = (balance / total_portfolio * 100) if total_portfolio > 0 else 100
        if cash_pct > 30:
            proposal["reasoning"].append(f"High cash allocation ({cash_pct:.1f}%). Consider deploying capital.")
        
        return json.dumps(proposal)
        
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "success": False
        })


@mcp.tool()
def get_strategy_for_goal(user_id: int, goal: str, time_horizon: str) -> str:
    """Generate personalized investment strategy based on goal.
    
    Args:
        user_id: User ID
        goal: Investment goal (e.g., "retirement", "growth", "income")
        time_horizon: Time horizon (e.g., "short", "medium", "long")
        
    Returns:
        JSON string with strategy recommendation
    """
    try:
        result = _call_tool(generate_personalized_strategy, {
            "user_id": user_id,
            "goal": goal,
            "time_horizon": time_horizon,
        })
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "goal": goal,
            "time_horizon": time_horizon,
            "success": False
        })


@mcp.tool()
def check_strategy_adherence(user_id: int) -> str:
    """Check how well user is following their investment strategy.
    
    Args:
        user_id: User ID
        
    Returns:
        JSON string with adherence metrics
    """
    try:
        result = _call_tool(track_strategy_adherence, {"user_id": user_id})
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "user_id": user_id,
            "success": False
        })


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
