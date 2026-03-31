"""Market research tools with fallback support for OHLCV data.

Provides LSTM predictions and news search with multiple data sources.
"""

import json
import logging
import os
from typing import Iterable, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from requests import RequestException

from http_client import get
from lstm.predict_next_day import predict_next_day_price
from tools._http_helpers import make_error_response as _err, API_BASE

logger = logging.getLogger(__name__)

# Configuration with environment variable support
USE_FALLBACK = os.getenv("USE_FALLBACK_DATA", "true").lower() == "true"
REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]

load_dotenv()
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
CUSTOM_SEARCH_CX = os.getenv("CUSTOM_SEARCH_CX")


def _fetch_ohlcv_from_broker(ticker: str) -> List[dict]:
    """Fetch OHLCV data from primary broker API."""
    url = f"{API_BASE}/stockdailyprices?symbol={ticker}"
    response = get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", payload)
    if not isinstance(data, Iterable):
        raise ValueError("Unexpected OHLCV payload format")
    return list(data)


def _fetch_ohlcv_from_yfinance(ticker: str, period: str = "3mo") -> Optional[List[dict]]:
    """Fetch OHLCV data from Yahoo Finance as fallback."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        
        if df.empty:
            logger.warning(f"No yfinance data for {ticker}")
            return None
        
        # Reset index to get date as column
        df = df.reset_index()
        
        # Rename columns to match expected format
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        # Convert to list of dicts
        records = df[["date", "open", "high", "low", "close", "volume"]].to_dict("records")
        logger.debug(f"Using Yahoo Finance OHLCV for {ticker}: {len(records)} records")
        return records
        
    except Exception as e:
        logger.debug(f"Yahoo Finance OHLCV fallback failed for {ticker}: {e}")
        return None


def _fetch_ohlcv(ticker: str) -> List[dict]:
    """Fetch OHLCV data with fallback support."""
    ticker = ticker.upper()
    
    # Try primary: Broker API
    try:
        return _fetch_ohlcv_from_broker(ticker)
    except Exception as e:
        logger.debug(f"Broker API unavailable for {ticker} OHLCV: {e}")
    
    # Try fallback: Yahoo Finance
    if USE_FALLBACK:
        yf_data = _fetch_ohlcv_from_yfinance(ticker)
        if yf_data:
            return yf_data
    
    raise ValueError(f"All OHLCV data sources failed for {ticker}")


def _to_dataframe(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in OHLCV data: {', '.join(missing)}")
    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 21:
        raise ValueError("Need at least 21 days of OHLCV data to predict")
    return df


def _fetch_live_news(query: str, num: int = 5) -> List[dict]:
    if not CUSTOM_SEARCH_API_KEY:
        raise RuntimeError("Missing GOOGLE_CUSTOM_SEARCH_API_KEY or CUSTOM_SEARCH_API_KEY")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": CUSTOM_SEARCH_API_KEY,
        "cx": CUSTOM_SEARCH_CX,
        "q": query,
        "num": num,
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items") or []
    if not isinstance(items, list):
        return []
    return items[:num]


@tool
def predict(ticker: str) -> str:
    """Predict the next day's closing price using recent OHLCV data and an LSTM model.

    Returns a JSON string with the predicted price or an error message.
    Supported tickers: AAPL, AMZN, ADBE, GOOGL, IBM, JPM, META, MSFT, NVDA, ORCL, TSLA.
    """
    ticker = ticker.upper().strip()
    SUPPORTED = {"AAPL", "AMZN", "ADBE", "GOOGL", "IBM", "JPM", "META", "MSFT", "NVDA", "ORCL", "TSLA"}
    if ticker not in SUPPORTED:
        return json.dumps({
            "error": f"Ticker '{ticker}' is not supported for prediction.",
            "supported_tickers": sorted(SUPPORTED),
        })
    try:
        records = _fetch_ohlcv(ticker)
        prices_df = _to_dataframe(records)
        prediction_df = predict_next_day_price(ticker, prices_df)
        predicted_price = float(prediction_df)
        return json.dumps({"ticker": ticker, "predicted_close": round(predicted_price, 2)})
    except (RequestException, ValueError, TypeError, RuntimeError, ImportError) as exc:
        logger.debug("Prediction failed for %s: %s", ticker, exc)
        return json.dumps({"ticker": ticker, "error": f"Prediction failed: {exc}"})


@tool
def search_live_news(query: str) -> str:
    """Search live news via Google Custom Search and return concise headlines with links.

    If the Custom Search API is not configured, returns a helpful fallback message.
    """
    if not CUSTOM_SEARCH_API_KEY or not CUSTOM_SEARCH_CX:
        return (
            "Live news search is not configured (missing CUSTOM_SEARCH_API_KEY or CUSTOM_SEARCH_CX). "
            "Try using google_search on the General Agent for web results, or ask me about "
            "price data which is available via the broker API."
        )
    try:
        items = _fetch_live_news(query)
    except (RequestException, ValueError, RuntimeError) as exc:
        logger.warning("News search failed for '%s': %s", query, exc)
        return f"News search unavailable: {exc}"
    if not items:
        return f"No recent news found for '{query}'."
    lines = []
    for item in items:
        title = item.get("title", "Untitled")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        lines.append(f"- **{title}** — {snippet} [{link}]")
    return "\n".join(lines)


@tool
def get_all_companies() -> str:
    """Get the list of all tradable companies.

    Calls MAFA-B  GET /companies
    Response shape: ApiResponse { data: List<CompanyDto> }
        CompanyDto: { id, name, symbol, sector }
    """
    try:
        response = get(f"{API_BASE}/companies", timeout=10)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload)
        return json.dumps(data) if data else "[]"
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching companies: {exc}")
        return json.dumps(_err(exc, "get all companies"))


@tool
def get_all_sectors() -> str:
    """Get the list of all market sectors.

    Calls MAFA-B  GET /sectors
    Response shape: ApiResponse { data: List<SectorDto> }
        SectorDto: { id, name }
    """
    try:
        response = get(f"{API_BASE}/sectors", timeout=10)
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload)
        return json.dumps(data) if data else "[]"
    except (RequestException, ValueError, TypeError) as exc:
        logger.warning(f"Error fetching sectors: {exc}")
        return json.dumps(_err(exc, "get all sectors"))
