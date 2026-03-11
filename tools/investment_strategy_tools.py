"""Investment strategy tools – wired to real MAFA-B data.

All strategy-related analysis is grounded in real user holdings, transactions,
preferences, and dashboard data fetched from MAFA-B endpoints via http_client
(which attaches the user's JWT automatically).

MAFA-B endpoints consumed
──────────────────────────
GET  /profile/preferences          → ApiResponse { data: PreferenceResponse }
GET  /holdings                     → ApiResponse { data: List<Share> }
GET  /dashboard                    → List<StockDto>
GET  /transactions                 → List<TransactionDto>
GET  /stockprice?symbol=X          → Double
POST /bulkstockprice               → ApiResponse { data: List<StockPriceDto> }
GET  /companies/{symbol}           → ApiResponse { data: CompanyDto }  (cached)
POST /companies/by-symbols         → ApiResponse { data: List<CompanyDto> }
GET  /strategy                     → ApiResponse { data: StrategyDto }  (active; 404 if none)
GET  /portfolio/history?period=&interval= → ApiResponse { data: List<PortfolioDailySnapshotDTO> }
"""

import contextvars
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from requests import RequestException

from http_client import get
from tools._http_helpers import (
    AuthError, raise_on_auth, make_error_response as _err,
    fetch_json as _fetch_json, unwrap as _unwrap, API_BASE,
)

logger = logging.getLogger(__name__)


def _get_preferences() -> Dict:
    try:
        return _unwrap(_fetch_json(f"{API_BASE}/profile/preferences")) or {}
    except Exception as exc:
        raise_on_auth(exc)
        logger.warning(f"Could not fetch preferences: {exc}")
        return {}


def _get_holdings() -> List[Dict]:
    """List<Share>  each: {symbol, quantity, price, id}"""
    try:
        data = _unwrap(_fetch_json(f"{API_BASE}/holdings"))
        return data if isinstance(data, list) else []
    except Exception as exc:
        raise_on_auth(exc)
        logger.warning(f"Could not fetch holdings: {exc}")
        return []


def _get_dashboard() -> List[Dict]:
    """List<StockDto>  each: {symbol, shares, totalAmount, currentPrice, avgBuyPrice, gainLoss}"""
    try:
        data = _fetch_json(f"{API_BASE}/dashboard")
        return data if isinstance(data, list) else []
    except Exception as exc:
        raise_on_auth(exc)
        logger.warning(f"Could not fetch dashboard: {exc}")
        return []


def _get_transactions() -> List[Dict]:
    """List<TransactionDto>  each: {id, type, asset, assetQuantity, amount, createdAt}"""
    try:
        data = _fetch_json(f"{API_BASE}/transactions")
        return data if isinstance(data, list) else []
    except Exception as exc:
        raise_on_auth(exc)
        logger.warning(f"Could not fetch transactions: {exc}")
        return []


def _get_stock_price(symbol: str) -> float:
    try:
        return float(_fetch_json(f"{API_BASE}/stockprice?symbol={symbol}"))
    except Exception:
        return 0.0


def _get_companies() -> List[Dict]:
    try:
        return _unwrap(_fetch_json(f"{API_BASE}/companies")) or []
    except Exception as exc:
        raise_on_auth(exc)
        return []


def _get_sectors() -> List[Dict]:
    try:
        return _unwrap(_fetch_json(f"{API_BASE}/sectors")) or []
    except Exception as exc:
        raise_on_auth(exc)
        return []


def _get_companies_by_symbols(symbols: List[str]) -> List[Dict]:
    """Bulk company lookup including sector info."""
    if not symbols:
        return []
    try:
        from http_client import post as http_post
        resp = http_post(f"{API_BASE}/companies/by-symbols", json={"symbols": symbols}, timeout=10)
        resp.raise_for_status()
        return _unwrap(resp.json()) or []
    except Exception as exc:
        logger.warning(f"Bulk company lookup failed: {exc}")
        return []


def _get_active_strategy() -> Dict:
    """Fetch the user's current saved strategy."""
    try:
        return _unwrap(_fetch_json(f"{API_BASE}/strategy")) or {}
    except Exception as exc:
        raise_on_auth(exc)
        return {}


def _get_portfolio_history(period: str = "LAST_30_DAYS", interval: str = "DAILY") -> List[Dict]:
    try:
        data = _unwrap(_fetch_json(f"{API_BASE}/portfolio/history?period={period}&interval={interval}"))
        return data if isinstance(data, list) else []
    except Exception as exc:
        raise_on_auth(exc)
        return []


def _json_dump(payload: Dict) -> str:
    return json.dumps(payload, default=str)


def _fetch_parallel(*funcs):
    """Run callables concurrently, propagating contextvars (auth token, cache).

    Each callable gets its own snapshot of the current context so the JWT
    token and request-scoped GET cache are available in every worker thread.
    Returns results in the same order as the input callables.
    """
    if len(funcs) <= 1:
        return [f() for f in funcs]
    contexts = [contextvars.copy_context() for _ in funcs]
    with ThreadPoolExecutor(max_workers=min(len(funcs), 6)) as pool:
        futures = [pool.submit(ctx.run, fn) for ctx, fn in zip(contexts, funcs)]
        return [f.result() for f in futures]


# ---------------------------------------------------------------------------
# Tool definitions – all grounded in real MAFA-B data
# ---------------------------------------------------------------------------

@tool
def assess_risk_tolerance() -> str:
    """Assess the user's risk tolerance from their saved preferences and actual
    trading behaviour (transaction history, portfolio concentration).

    Sources:
    - GET /profile/preferences  → riskTolerance setting
    - GET /transactions         → trade frequency & size distribution
    - GET /dashboard            → position concentration
    """
    try:
        prefs, txns, dashboard = _fetch_parallel(_get_preferences, _get_transactions, _get_dashboard)
    except AuthError as e:
        return json.dumps({"error": str(e), "context": "assess risk tolerance"})

    # Risk from explicit preference
    risk_setting = prefs.get("riskTolerance", "unknown")

    # Trade frequency (proxy for risk appetite)
    total_trades = len(txns)
    buy_count = sum(1 for t in txns if str(t.get("type", "")).upper() == "BUY")
    sell_count = total_trades - buy_count

    # Concentration analysis from dashboard
    total_value = sum(d.get("totalAmount", 0) for d in dashboard)
    max_position = 0.0
    max_symbol = ""
    concentrations: Dict[str, float] = {}
    for d in dashboard:
        val = d.get("totalAmount", 0)
        sym = d.get("symbol", "???")
        pct = round(val / total_value, 4) if total_value > 0 else 0.0
        concentrations[sym] = pct
        if pct > max_position:
            max_position = pct
            max_symbol = sym

    # Sector concentration via bulk company lookup
    symbols = [d.get("symbol", "") for d in dashboard if d.get("symbol")]
    companies = _get_companies_by_symbols(symbols)
    sym_to_sector: Dict[str, str] = {}
    for c in companies:
        s = c.get("sector")
        sector_name = s.get("name", "Other") if isinstance(s, dict) else str(s) if s else "Other"
        sym_to_sector[c.get("symbol", "")] = sector_name

    sector_weights: Dict[str, float] = {}
    for sym, pct in concentrations.items():
        sec = sym_to_sector.get(sym, "Other")
        sector_weights[sec] = round(sector_weights.get(sec, 0.0) + pct, 4)

    # Derive a risk level heuristic
    if risk_setting.lower() in ("high", "aggressive"):
        risk_level = "aggressive"
        risk_score = 0.85
    elif risk_setting.lower() in ("low", "conservative"):
        risk_level = "conservative"
        risk_score = 0.30
    else:
        risk_level = "moderate"
        risk_score = 0.55

    # Adjust score by concentration – heavy concentration → higher risk
    if max_position > 0.40:
        risk_score = min(risk_score + 0.15, 1.0)
        risk_level = "aggressive" if risk_score > 0.7 else risk_level

    payload = {
        "risk_level": risk_level,
        "risk_score": round(risk_score, 2),
        "stated_risk_tolerance": risk_setting,
        "factors": {
            "total_trades": total_trades,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "max_single_position_pct": round(max_position * 100, 1),
            "max_position_symbol": max_symbol,
            "portfolio_value": round(total_value, 2),
        },
        "sector_concentrations": {k: round(v * 100, 1) for k, v in sector_weights.items()},
        "recommendations": _risk_recommendations(risk_level, max_position),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return _json_dump(payload)


def _risk_recommendations(risk_level: str, max_pos: float) -> List[str]:
    recs: List[str] = []
    if max_pos > 0.25:
        recs.append(f"Your largest position is {round(max_pos*100,1)}% of portfolio – consider trimming to <20%.")
    if risk_level == "aggressive":
        recs.append("With aggressive risk, maintain a 5–10% cash buffer for buying dips.")
    elif risk_level == "conservative":
        recs.append("Conservative posture – consider keeping ≥15% in low-volatility holdings or cash.")
    recs.append("Review risk posture quarterly or after any drawdown >10%.")
    return recs


@tool
def analyze_portfolio_alignment(target_strategy: Optional[str] = None) -> str:
    """Compare the user's current portfolio allocation against a target.

    Uses real holdings from GET /dashboard.  If *target_strategy* is provided
    (JSON mapping symbol→weight or sector→weight), compare against that;
    otherwise compare against an equal-weight benchmark across held stocks.

    Sources: GET /dashboard, GET /profile/preferences
    """
    try:
        dashboard, prefs = _fetch_parallel(_get_dashboard, _get_preferences)
    except AuthError as e:
        return json.dumps({"error": str(e), "context": "analyze portfolio alignment"})

    total_value = sum(d.get("totalAmount", 0) for d in dashboard)
    current: Dict[str, float] = {}
    for d in dashboard:
        sym = d.get("symbol", "???")
        pct = round(d.get("totalAmount", 0) / total_value, 4) if total_value > 0 else 0.0
        current[sym] = pct

    # Resolve target
    if target_strategy:
        try:
            target = json.loads(target_strategy)
        except json.JSONDecodeError:
            target = {sym: round(1.0 / len(current), 4) for sym in current} if current else {}
    else:
        # Equal-weight across current symbols as a simple benchmark
        target = {sym: round(1.0 / len(current), 4) for sym in current} if current else {}

    deviations: List[Dict] = []
    for sym in set(list(current.keys()) + list(target.keys())):
        cur = current.get(sym, 0.0)
        tgt = target.get(sym, 0.0)
        diff = round(cur - tgt, 4)
        if abs(diff) >= 0.03:
            deviations.append({
                "symbol": sym,
                "current_pct": round(cur * 100, 1),
                "target_pct": round(tgt * 100, 1),
                "difference_pct": round(diff * 100, 1),
                "status": "overweight" if diff > 0 else "underweight",
            })

    alignment_score = 1 - sum(abs(d["difference_pct"]) for d in deviations) / 200
    alignment_score = max(round(alignment_score, 2), 0.0)

    payload = {
        "alignment_score": alignment_score,
        "portfolio_value": round(total_value, 2),
        "current_allocation": {k: round(v * 100, 1) for k, v in current.items()},
        "target_allocation": {k: round(v * 100, 1) for k, v in target.items()},
        "deviations": deviations,
        "user_goals": prefs.get("investmentGoals", "not set"),
        "recommended_actions": [
            f"Trim overweight positions by redirecting to underweight ones."
            if deviations else "Portfolio is well-aligned.",
            "Rebalance whenever drift exceeds 5% on any position.",
        ],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return _json_dump(payload)


@tool
def generate_personalized_strategy(goal: str, time_horizon: str) -> str:
    """Generate a personalized investment strategy based on the user's goal,
    time horizon, and their stored preferences.

    Sources: GET /profile/preferences
    """
    try:
        prefs = _get_preferences()
    except AuthError as e:
        return json.dumps({"error": str(e), "context": "generate personalized strategy"})

    goal_lower = goal.lower().strip()
    horizon = time_horizon.lower().strip()
    risk = (prefs.get("riskTolerance") or "moderate").lower()

    # Determine strategy type from goal + risk + horizon
    if goal_lower in ("growth", "wealth_building") and horizon in ("long", "long_term", "long-term"):
        strategy_type = "aggressive_growth" if risk in ("high", "aggressive") else "growth"
        allocation = {"stocks": 0.80, "bonds": 0.15, "cash": 0.05}
    elif goal_lower in ("income", "retirement"):
        strategy_type = "conservative_income"
        allocation = {"stocks": 0.40, "bonds": 0.45, "cash": 0.15}
    elif goal_lower in ("preservation", "capital_preservation"):
        strategy_type = "capital_preservation"
        allocation = {"stocks": 0.25, "bonds": 0.50, "cash": 0.25}
    else:
        strategy_type = "balanced"
        allocation = {"stocks": 0.60, "bonds": 0.30, "cash": 0.10}

    # Enrich with user's preferred sectors / companies
    preferred_sectors = prefs.get("sectorIds", [])
    preferred_companies = prefs.get("companyIds", [])

    payload = {
        "strategy_type": strategy_type,
        "goal": goal_lower,
        "time_horizon": horizon,
        "stated_risk_tolerance": risk,
        "asset_allocation": allocation,
        "preferred_sectors": preferred_sectors,
        "preferred_companies": preferred_companies,
        "risk_rules": {
            "max_single_position": 0.15 if risk in ("high", "aggressive") else 0.10,
            "stop_loss": 0.20 if risk in ("high", "aggressive") else 0.12,
            "rebalance_trigger": 0.05,
            "cash_floor": allocation["cash"],
        },
        "milestones": _milestones(strategy_type),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return _json_dump(payload)


def _milestones(strategy: str) -> Dict:
    if strategy in ("aggressive_growth", "growth"):
        return {"3m": {"target_return": 0.04}, "6m": {"target_return": 0.10}, "12m": {"target_return": 0.18}}
    if strategy == "conservative_income":
        return {"3m": {"target_return": 0.01}, "6m": {"target_return": 0.03}, "12m": {"target_return": 0.06}}
    return {"3m": {"target_return": 0.02}, "6m": {"target_return": 0.05}, "12m": {"target_return": 0.10}}


@tool
def calculate_optimal_allocation(target_allocation: str) -> str:
    """Suggest concrete trades to move the portfolio toward a target allocation.

    *target_allocation* should be a JSON string mapping symbols to target weights
    (e.g. '{"AAPL":0.25, "MSFT":0.25, "GOOGL":0.25, "JPM":0.25}').

    Sources: GET /dashboard, GET /stockprice
    """
    try:
        dashboard = _get_dashboard()
    except AuthError as e:
        return json.dumps({"error": str(e), "context": "calculate optimal allocation"})
    total_value = sum(d.get("totalAmount", 0) for d in dashboard)

    try:
        target = json.loads(target_allocation)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid target_allocation JSON."})

    current: Dict[str, Dict] = {}
    for d in dashboard:
        sym = d.get("symbol", "???")
        current[sym] = {
            "shares": d.get("shares", 0),
            "value": d.get("totalAmount", 0),
            "current_price": d.get("currentPrice", 0),
            "weight": round(d.get("totalAmount", 0) / total_value, 4) if total_value > 0 else 0.0,
        }

    trades: List[Dict] = []
    for sym, target_weight in target.items():
        cur = current.get(sym)
        cur_value = cur["value"] if cur else 0.0
        cur_price = cur["current_price"] if cur else _get_stock_price(sym)
        target_value = total_value * target_weight
        diff_value = target_value - cur_value

        if cur_price <= 0:
            continue
        diff_shares = int(diff_value / cur_price)
        if diff_shares == 0:
            continue
        trades.append({
            "action": "buy" if diff_shares > 0 else "sell",
            "symbol": sym,
            "quantity": abs(diff_shares),
            "estimated_value": round(abs(diff_shares) * cur_price, 2),
            "reason": f"Move {sym} from {round((cur_value/total_value)*100 if total_value else 0,1)}% to {round(target_weight*100,1)}%",
            "priority": "high" if abs(diff_value) / max(total_value, 1) > 0.05 else "medium",
        })

    # Also flag symbols held but not in target
    for sym in current:
        if sym not in target:
            trades.append({
                "action": "sell",
                "symbol": sym,
                "quantity": current[sym]["shares"],
                "estimated_value": round(current[sym]["value"], 2),
                "reason": f"{sym} not in target allocation – consider selling.",
                "priority": "low",
            })

    payload = {
        "portfolio_value": round(total_value, 2),
        "target": target,
        "suggested_trades": trades,
        "notes": "Trades are estimates. Review prices before executing.",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return _json_dump(payload)


@tool
def track_strategy_adherence() -> str:
    """Measure how well the user's actual portfolio tracks their stated strategy
    and preferences, including their saved strategy from MAFA-B.

    Sources: GET /dashboard, GET /profile/preferences, GET /transactions,
             GET /strategy/active, POST /companies/by-symbols, GET /portfolio/history
    """
    try:
        dashboard, prefs, txns = _fetch_parallel(_get_dashboard, _get_preferences, _get_transactions)
    except AuthError as e:
        return json.dumps({"error": str(e), "context": "track strategy adherence"})

    total_value = sum(d.get("totalAmount", 0) for d in dashboard)
    num_positions = len(dashboard)

    # Gain/loss analysis
    winners = [d for d in dashboard if (d.get("gainLoss") or 0) > 0]
    losers = [d for d in dashboard if (d.get("gainLoss") or 0) < 0]
    total_gain = sum(d.get("gainLoss", 0) for d in dashboard)

    # Concentration (largest position)
    max_pct = 0.0
    max_sym = ""
    for d in dashboard:
        pct = d.get("totalAmount", 0) / total_value if total_value > 0 else 0
        if pct > max_pct:
            max_pct = pct
            max_sym = d.get("symbol", "???")

    # Recent activity
    recent_buys = sum(1 for t in txns if str(t.get("type", "")).upper() == "BUY")
    recent_sells = len(txns) - recent_buys

    # Adherence scoring (simple heuristic)
    score = 1.0
    alerts: List[str] = []
    if max_pct > 0.30:
        score -= 0.15
        alerts.append(f"{max_sym} is {round(max_pct*100,1)}% of portfolio – consider trimming to <25%.")
    if num_positions < 3 and total_value > 0:
        score -= 0.10
        alerts.append("Portfolio has very few positions – diversification is low.")
    if total_gain < 0:
        alerts.append(f"Portfolio is down ${abs(round(total_gain,2))} overall – review underperformers.")

    risk = (prefs.get("riskTolerance") or "moderate").lower()
    goals = prefs.get("investmentGoals", "not set")

    # Fetch saved strategy + portfolio history in parallel (both independent)
    saved, history = _fetch_parallel(
        _get_active_strategy,
        lambda: _get_portfolio_history("LAST_30_DAYS", "WEEKLY"),
    )
    saved_allocation = saved.get("targetAllocation") if saved else None
    saved_sector_limits = saved.get("sectorLimits") if saved else None
    saved_type = saved.get("strategyType", "none") if saved else "none"

    if saved_allocation:
        alerts.append(f"Active strategy '{saved_type}' target allocation: {json.dumps(saved_allocation)}. Compare with your current portfolio.")

    # Sector-level drift check
    symbols = [d.get("symbol", "") for d in dashboard if d.get("symbol")]
    companies = _get_companies_by_symbols(symbols)
    sym_to_sector: Dict[str, str] = {}
    for c in companies:
        s = c.get("sector")
        sec_name = s.get("name", "Other") if isinstance(s, dict) else str(s) if s else "Other"
        sym_to_sector[c.get("symbol", "")] = sec_name

    sector_weights: Dict[str, float] = {}
    for d in dashboard:
        sym = d.get("symbol", "")
        val = d.get("totalAmount", 0)
        sec = sym_to_sector.get(sym, "Other")
        sector_weights[sec] = sector_weights.get(sec, 0.0) + (val / total_value if total_value > 0 else 0.0)

    if saved_sector_limits and isinstance(saved_sector_limits, dict):
        for sec, limit in saved_sector_limits.items():
            actual = sector_weights.get(sec, 0.0) * 100
            limit_val = float(limit)
            if actual > limit_val + 5:
                score -= 0.10
                alerts.append(f"Sector '{sec}' at {actual:.1f}% exceeds limit of {limit_val}% by {actual - limit_val:.1f}pp.")
    trend_note = ""
    if len(history) >= 2:
        first_val = history[0].get("totalValue", 0)
        last_val = history[-1].get("totalValue", 0)
        if first_val > 0:
            pct_change = ((last_val - first_val) / first_val) * 100
            trend_note = f"Portfolio {'up' if pct_change >= 0 else 'down'} {abs(pct_change):.1f}% over last month."

    payload = {
        "adherence_score": round(max(score, 0.0), 2),
        "portfolio_value": round(total_value, 2),
        "num_positions": num_positions,
        "total_gain_loss": round(total_gain, 2),
        "winning_positions": len(winners),
        "losing_positions": len(losers),
        "max_concentration": {"symbol": max_sym, "pct": round(max_pct * 100, 1)},
        "sector_weights": {k: round(v * 100, 1) for k, v in sector_weights.items()},
        "recent_activity": {"buys": recent_buys, "sells": recent_sells, "total": len(txns)},
        "stated_risk": risk,
        "stated_goals": goals,
        "saved_strategy": saved_type,
        "monthly_trend": trend_note,
        "alerts": alerts,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    return _json_dump(payload)
