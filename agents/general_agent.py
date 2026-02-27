"""General Financial Agent — account info, lookups, and light guidance."""

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from agents.base import model, run_agent_turn, normalize_content
from tools.execute_trade_tools import buy_stock, sell_stock
from tools.profile_tools import (
    get_current_stock_price,
    get_bulk_stock_prices,
    get_user_balance,
    get_user_holdings,
    get_user_profile,
    get_user_transactions,
    get_dashboard,
    get_stock_change,
    get_company_by_symbol,
    get_companies_by_symbols,
    get_portfolio_history,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
)
from tools.alert_tools import create_alert, get_alerts, delete_alert
from tools.memory_tools import search_user_memory, store_user_note

# Google Search via Gemini’s built-in grounding
_model_with_search = model.bind_tools([{"google_search": {}}])


@tool
def google_search(query: str) -> str:
    """Perform a Google search and return summarized and processed results."""
    try:
        result = _model_with_search.invoke(query)
        return normalize_content(result.content) if hasattr(result, "content") else str(result)
    except Exception as exc:
        return f"Google search unavailable: {exc}"

# Expose all account-related tools.
tools = [
    get_user_balance,
    get_user_holdings,
    get_user_profile,
    get_current_stock_price,
    get_bulk_stock_prices,
    get_stock_change,
    get_user_transactions,
    get_dashboard,
    get_company_by_symbol,
    get_companies_by_symbols,
    get_portfolio_history,
    get_watchlist,
    add_to_watchlist,
    remove_from_watchlist,
    create_alert,
    get_alerts,
    delete_alert,
    google_search,
    buy_stock,
    sell_stock,
    search_user_memory,
    store_user_note,
]

BASE_SYSTEM_PROMPT = """
You are the General Financial Agent. Act as a fast, trustworthy concierge for account info, quick lookups, and light guidance. Keep one shared memory with other agents via Supabase.

Tools
- get_user_balance, get_user_holdings, get_user_profile, get_user_transactions, get_dashboard: account info.
- get_current_stock_price(symbol), get_bulk_stock_prices(symbols): real-time prices (bulk for multiple tickers).
- get_stock_change(symbol): price change info (price, change, changePercent).
- get_company_by_symbol(symbol): look up a company's details and sector.
- get_companies_by_symbols(symbols): bulk company lookup with sector info.
- get_portfolio_history(period, interval): portfolio value over time (periods: LAST_24_HOURS, LAST_7_DAYS, LAST_30_DAYS, LAST_90_DAYS, LAST_1_YEAR, ALL; intervals: DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY).
- get_watchlist, add_to_watchlist(symbol), remove_from_watchlist(symbol): manage the user's stock watchlist.
- create_alert(symbol, condition, target_price, channel?): set a price alert. condition: ABOVE|BELOW. channel: IN_APP (default) or USER.
- get_alerts(status?): list price alerts (ACTIVE, TRIGGERED, CANCELLED).
- delete_alert(alert_id): cancel a price alert.
- google_search: use for freshness (news, announcements, current figures).
- buy_stock, sell_stock: for simple trade execution after explicit confirmation.
- search_user_memory, store_user_note: recall/store brief context to keep conversations coherent across agents.

Operating rules
1) Core scope: balances, holdings, price checks, account status, company overviews, watchlist management, portfolio history, headlines, and general finance Q&A.
2) Routing: for complex trades with analysis, redirect to the Execution Agent. For forecasts/deep research, redirect to the Market Research Agent. For strategy, redirect to the Investment Strategy Agent.
3) Recency: when the question depends on latest info, call google_search, then summarize top takeaways with source mentions.
4) Memory: when context matters, search memory for recent intents or preferences; state only what you find. After useful interactions, store a short note (topic, ticker, preference) to shared memory.
5) Safety & tone: avoid personalized investment advice; mark stale/approximate data; ask one clarifying question if needed.
6) Style: concise first answer with key figures, then one clear next step or option.
"""


agent = create_react_agent(model=model, tools=tools, prompt=BASE_SYSTEM_PROMPT)

# Limited agent: no broker API tools, for unsupported-company queries
NO_BROKER_TOOLS = [
    google_search,
    search_user_memory,
    store_user_note,
]

NO_BROKER_PROMPT = BASE_SYSTEM_PROMPT + """

Broker limits
- You must NOT call broker API tools in this mode. Use only google_search or memory tools.
- Start by politely noting the company is not supported by the broker yet, and that it might be supported in the future.
- Then answer any general, non-broker questions the user asked (company overview, public info, news).
"""

agent_no_broker = create_react_agent(model=model, tools=NO_BROKER_TOOLS, prompt=NO_BROKER_PROMPT)


def run_general_agent(user_message: str, user_id: int) -> str:
    return run_agent_turn("general_agent", agent, user_message, user_id)


def run_general_agent_no_broker(user_message: str, user_id: int) -> str:
    return run_agent_turn("general_agent", agent_no_broker, user_message, user_id)
