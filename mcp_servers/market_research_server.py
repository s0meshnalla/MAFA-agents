"""TRUE MCP Server for Market Research.

Exposes LSTM predictions and live news search via official MCP stdio protocol.
Run: python mcp_servers/market_research_server.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
from tools.market_research_tools import predict, search_live_news

# Create TRUE MCP Server
mcp = FastMCP("market-research-server")

# Valid ticker symbols supported by our LSTM models
VALID_TICKERS = ["AAPL", "AMZN", "ADBE", "GOOGL", "IBM", "JPM", "META", "MSFT", "NVDA", "ORCL", "TSLA"]


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


def _parse_prediction_value(prediction_payload) -> float:
    """Extract a predicted_close-like float from a tool payload."""
    if isinstance(prediction_payload, dict):
        pred_val = prediction_payload.get("predicted_close", prediction_payload.get("data", prediction_payload.get("prediction")))
        if pred_val is not None:
            return float(pred_val)
        if prediction_payload:
            return float(list(prediction_payload.values())[0])
        return 0.0
    return float(prediction_payload)


def _load_cached_prediction(symbol: str):
    """Load latest precomputed prediction from lstm/output/{symbol}/predictions.csv.

    This avoids expensive runtime inference in MCP subprocesses.
    """
    path = Path(__file__).resolve().parents[1] / "lstm" / "output" / symbol / "predictions.csv"
    if not path.exists():
        return None

    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return None
        # CSV shape: date,y_true,y_pred
        last = lines[-1].split(",")
        if len(last) < 3:
            return None
        as_of = last[0].strip()
        predicted_close = float(last[2].strip())
        return {"predicted_close": predicted_close, "as_of": as_of, "source": "offline_artifact"}
    except Exception:
        return None


def _predict_symbol(symbol: str):
    """Prediction helper with cached-first strategy and live-tool fallback."""
    cached = _load_cached_prediction(symbol)
    if cached:
        return cached

    result = _call_tool(predict, {"ticker": symbol})
    prediction = json.loads(result) if isinstance(result, str) else result
    return {"predicted_close": _parse_prediction_value(prediction), "source": "live_lstm"}


@mcp.tool()
def predict_next_day(symbol: str) -> str:
    """Predict next day's closing price using LSTM model.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA, GOOGL)
        
    Returns:
        JSON string with prediction result or error message
    """
    symbol = symbol.upper().strip()
    
    if symbol not in VALID_TICKERS:
        return json.dumps({
            "error": f"Ticker '{symbol}' not supported. Valid tickers: {VALID_TICKERS}",
            "success": False
        })
    
    try:
        prediction_obj = _predict_symbol(symbol)
        return json.dumps({
            "symbol": symbol,
            "predicted_close": float(prediction_obj["predicted_close"]),
            "prediction_source": prediction_obj.get("source", "unknown"),
            "as_of": prediction_obj.get("as_of"),
            "model": "LSTM",
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "symbol": symbol,
            "success": False
        })


@mcp.tool()
def get_live_news(query: str, num_results: int = 5) -> str:
    """Search for live financial news related to a stock or topic.
    
    Args:
        query: Search query (e.g., "AAPL earnings", "Tesla stock news")
        num_results: Number of news results to return (default: 5)
        
    Returns:
        JSON string with news articles or error message
    """
    try:
        result = _call_tool(search_live_news, {"query": query})
        # search_live_news returns plain-text markdown lines, not JSON
        if isinstance(result, str):
            try:
                news = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                news = {"text": result}
        else:
            news = result
        return json.dumps({
            "query": query,
            "results": news,
            "success": True
        })
    except Exception as exc:
        return json.dumps({
            "error": str(exc),
            "query": query,
            "success": False
        })


@mcp.tool()
def get_market_analysis(symbol: str) -> str:
    """Get combined market analysis: LSTM prediction + recent news.
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        JSON string with comprehensive market analysis
    """
    symbol = symbol.upper().strip()
    
    if symbol not in VALID_TICKERS:
        return json.dumps({
            "error": f"Ticker '{symbol}' not supported. Valid tickers: {VALID_TICKERS}",
            "success": False
        })
    
    analysis = {
        "symbol": symbol,
        "prediction": None,
        "news": None,
        "success": True
    }
    
    # Get LSTM prediction
    try:
        prediction_obj = _predict_symbol(symbol)
        analysis["prediction"] = {
            "predicted_close": float(prediction_obj["predicted_close"]),
            "model": "LSTM",
            "source": prediction_obj.get("source", "unknown"),
            "as_of": prediction_obj.get("as_of"),
        }
    except Exception as exc:
        analysis["prediction"] = {"error": str(exc)}
    
    # Get live news
    try:
        result = _call_tool(search_live_news, {"query": f"{symbol} stock"})
        # search_live_news returns plain-text markdown, not JSON
        if isinstance(result, str):
            try:
                news = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                news = {"text": result}
        else:
            news = result
        analysis["news"] = news
    except Exception as exc:
        analysis["news"] = {"error": str(exc)}
    
    return json.dumps(analysis)


if __name__ == "__main__":
    # Run as TRUE MCP server via stdio transport
    mcp.run(transport="stdio")
