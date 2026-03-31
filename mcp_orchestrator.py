"""TRUE MCP Orchestrator — LLM-planned, protocol-native MCP orchestration.

Replaces the old 3-tier keyword/heuristic/shallow-LLM router with a
genuine MCP-powered pipeline:

    User Query  →  Tool Discovery  →  LLM Query Planner  →  MCP Executor  →  LLM Synthesizer  →  Response

Key capabilities:
  • Dynamic tool discovery from MCP servers at startup (via mcp_tool_registry)
  • LLM reasons over the FULL tool catalog to produce execution plans (via query_planner)
  • Tools invoked via actual MCP stdio protocol with parallel execution (via mcp_executor)
  • Structured result synthesis with all tool outputs available
  • Graceful degradation to in-process agents when MCP servers are unavailable
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agents.base import model as _shared_model, sanitize_user_response
from vectordbsupabase import SupabaseVectorDB
from tools.memory_tools import store_user_context
from event_bus import MCPTopics, get_event_bus
from http_client import get

from mcp_tool_registry import MCPToolRegistry, get_tool_registry
from query_planner import QueryPlanner, ExecutionPlan
from mcp_executor import MCPExecutor, PlanExecutionResult

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BROKER_API_URL = os.getenv("BROKER_API_URL", "http://localhost:8080")


# ---------------------------------------------------------------------------
# TRUE MCP Orchestrator
# ---------------------------------------------------------------------------

class TrueMCPOrchestrator:
    """TRUE MCP Orchestrator with dynamic tool discovery, LLM planning,
    and protocol-native execution.

    Architecture
    ------------
    1. **Tool Registry** discovers all MCP tools at startup via stdio.
    2. **Query Planner** (LLM) reads the full tool catalog and produces
       a structured execution plan for each user query.
    3. **MCP Executor** runs the plan — invoking tools via MCP protocol
       with support for parallel execution and per-tool timeouts.
    4. **Synthesizer** (LLM) merges tool results into a cohesive user response.
    5. **Fallback** — if MCP execution fails, falls back to in-process agents.
    """

    def __init__(self):
        self.vector_db = SupabaseVectorDB()
        self.event_bus = get_event_bus()
        self._company_cache: List[Dict[str, Any]] = []
        self._company_cache_ts: float = 0.0

        # Core pipeline components
        self.registry = get_tool_registry()
        self.llm = _shared_model
        self.planner: Optional[QueryPlanner] = None
        self.executor: Optional[MCPExecutor] = None

        self._initialized = False

    # -- Lifecycle ----------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize MCP connections, tool discovery, and pipeline."""
        if self._initialized:
            return

        logger.info("Initializing TrueMCPOrchestrator …")

        # Connect to event bus
        await self.event_bus.connect()

        # Discover tools from all MCP servers
        try:
            await self.registry.discover_all()
            logger.info(
                "Tool discovery complete: %d tools found across %d servers",
                self.registry.tool_count,
                len(self.registry._server_tools),
            )
        except Exception as exc:
            logger.warning("Tool discovery failed: %s — will use fallback agents", exc)

        # Initialize the planner and executor
        self.planner = QueryPlanner(self.registry, llm=self.llm)
        self.executor = MCPExecutor(self.registry)

        self._initialized = True
        logger.info("TrueMCPOrchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown all connections."""
        await self.event_bus.disconnect()
        self._initialized = False

    # -- Main orchestration entry point -------------------------------------

    async def orchestrate(
        self,
        user_id: int,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main orchestration workflow — true MCP pipeline.

        Pipeline:
            1. Company support check (broker compatibility)
            2. LLM Query Planner → structured execution plan
            3. Confirmation check for trade tools
            4. MCP Executor → parallel tool invocation
            5. LLM Synthesizer → cohesive user response
            6. Fallback to in-process agents on MCP failure

        Args:
            user_id: User ID
            query: User query
            session_id: Optional session ID

        Returns:
            Orchestration result with response and metadata
        """
        start_time = time.time()

        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info(
            "MCP orchestration started: user_id=%s, query=%.80s…",
            user_id, query,
        )

        # Publish query event
        await self.event_bus.publish_raw(
            MCPTopics.MCP_QUERY,
            user_id,
            {"query": query, "session_id": session_id},
        )

        try:
            # 0. Company support pre-check
            company_block = await self._check_company_support(query)
            if company_block:
                from agents.general_agent import run_general_agent_no_broker
                general_response = run_general_agent_no_broker(query, user_id, session_id)
                combined = general_response or company_block
                await self._publish_result(user_id, query, combined, [], session_id)
                return self._build_response(combined, [], {}, True, start_time, user_id, session_id)

            # 1. Plan — LLM reasons over full tool catalog
            plan = self.planner.plan(query, user_id) if self.planner else None

            if plan is None or (not plan.steps and plan.is_general_query):
                # General/greeting or planner unavailable → agent fallback
                response = await self._run_fallback_agent(
                    plan.fallback_agent if plan else "general",
                    query, user_id, session_id,
                )
                routing = {
                    "source": "fallback_agent",
                    "reason": plan.reasoning if plan else "planner unavailable",
                    "agent": plan.fallback_agent if plan else "general",
                }
                await self._publish_result(user_id, query, response, [], session_id, routing)
                return self._build_response(response, [], routing, True, start_time, user_id, session_id)

            # 2. Confirmation check for trade tools
            if plan.requires_confirmation and plan.has_trade_tools:
                confirm_msg = plan.confirmation_message or self._build_confirmation_message(plan)
                routing = {
                    "source": "planner",
                    "awaiting_confirmation": True,
                    "planned_tools": plan.tool_names,
                    "servers": plan.server_keys,
                }
                return self._build_response(confirm_msg, plan.server_keys, routing, True, start_time, user_id, session_id)

            # 3. Execute — invoke tools via MCP protocol
            exec_result = await self._execute_plan(plan)

            if exec_result and exec_result.success:
                # 4. Synthesize — LLM merges tool outputs
                response = self._synthesize_tool_results(
                    query, plan, exec_result,
                )
                response = sanitize_user_response(response, user_message=query)

                routing = {
                    "source": "mcp_pipeline",
                    "tools_called": exec_result.tools_called,
                    "servers_used": exec_result.servers_used,
                    "plan_confidence": plan.confidence,
                    "plan_reasoning": plan.reasoning,
                    "execution_time": exec_result.total_time,
                    "partial_failure": exec_result.partial_failure,
                }

                error_msg = self._extract_response_error(response)
                success = error_msg is None

                await self._save_turn(user_id, query, response)
                await self._publish_result(
                    user_id, query, response,
                    exec_result.servers_used, session_id, routing,
                )

                result = self._build_response(
                    response, exec_result.servers_used, routing,
                    success, start_time, user_id, session_id,
                )
                result["error"] = error_msg
                return result

            # 5. MCP execution returned partial data — still try to synthesize
            if exec_result and exec_result.results:
                any_data = any(r.success and r.data for r in exec_result.results)
                if any_data:
                    logger.info("MCP partial success — synthesizing available results")
                    response = self._synthesize_tool_results(
                        query, plan, exec_result,
                    )
                    response = sanitize_user_response(response, user_message=query)
                    routing = {
                        "source": "mcp_pipeline",
                        "tools_called": exec_result.tools_called,
                        "servers_used": exec_result.servers_used,
                        "plan_confidence": plan.confidence,
                        "execution_time": exec_result.total_time,
                        "partial_failure": True,
                    }
                    await self._save_turn(user_id, query, response)
                    await self._publish_result(
                        user_id, query, response,
                        exec_result.servers_used, session_id, routing,
                    )
                    return self._build_response(
                        response, exec_result.servers_used, routing,
                        True, start_time, user_id, session_id,
                    )

            # 6. MCP execution fully failed → fallback to in-process agents
            logger.warning("MCP execution failed — falling back to in-process agents")
            response = await self._run_plan_as_agents(plan, query, user_id, session_id)
            response = sanitize_user_response(response, user_message=query)

            routing = {
                "source": "agent_fallback",
                "reason": "MCP execution failed",
                "planned_tools": plan.tool_names,
                "servers": plan.server_keys,
            }
            await self._save_turn(user_id, query, response)
            await self._publish_result(user_id, query, response, plan.server_keys, session_id, routing)
            return self._build_response(response, plan.server_keys, routing, True, start_time, user_id, session_id)

        except Exception as exc:
            logger.error("MCP orchestration error: %s", exc)
            await self.event_bus.publish_raw(
                MCPTopics.MCP_ERRORS,
                user_id,
                {"query": query, "error": str(exc)},
            )
            return {
                "response": f"Error processing request: {str(exc)}",
                "success": False,
                "error": str(exc),
                "execution_time": time.time() - start_time,
            }

    # -- Execution ----------------------------------------------------------

    async def _execute_plan(self, plan: ExecutionPlan) -> Optional[PlanExecutionResult]:
        """Execute the plan via MCP protocol (with parallel support)."""
        if not self.executor or not plan.steps:
            return None

        try:
            return await self.executor.execute(plan)
        except Exception as exc:
            logger.error("MCP plan execution error: %s", exc)
            return None

    async def _run_fallback_agent(
        self,
        agent_key: Optional[str],
        query: str,
        user_id: int,
        session_id: str,
    ) -> str:
        """Run an in-process agent as a fallback."""
        handlers = self._get_agent_handlers()
        handler = handlers.get(agent_key or "general", handlers["general"])
        try:
            return handler(query, user_id, session_id)
        except Exception as exc:
            logger.error("Fallback agent '%s' error: %s", agent_key, exc)
            return f"Error processing request: {str(exc)}"

    async def _run_plan_as_agents(
        self,
        plan: ExecutionPlan,
        query: str,
        user_id: int,
        session_id: str,
    ) -> str:
        """Use the plan's server_keys to run in-process agents as MCP fallback."""
        handlers = self._get_agent_handlers()
        servers = plan.server_keys or ["general"]

        if len(servers) == 1:
            handler = handlers.get(servers[0], handlers["general"])
            return handler(query, user_id, session_id)

        # Multiple servers → run each agent and synthesize
        responses: Dict[str, str] = {}
        for server in servers:
            handler = handlers.get(server)
            if handler:
                try:
                    responses[server] = handler(query, user_id, session_id)
                except Exception as exc:
                    logger.warning("Agent fallback '%s' error: %s", server, exc)

        if not responses:
            return handlers["general"](query, user_id, session_id)

        if len(responses) == 1:
            return list(responses.values())[0]

        return self._synthesize_agent_responses(query, responses)

    @staticmethod
    def _get_agent_handlers() -> Dict[str, Any]:
        """Lazy import and return all in-process agent handler functions."""
        from agents.market_search_agent import run_market_research_agent
        from agents.execution_agent import run_execute_agent
        from agents.portfolio_manager_agent import run_portfolio_manager_agent
        from agents.investment_strategy_agent import run_investment_strategy_agent
        from agents.general_agent import run_general_agent
        return {
            "market": run_market_research_agent,
            "execution": run_execute_agent,
            "portfolio": run_portfolio_manager_agent,
            "strategy": run_investment_strategy_agent,
            "general": run_general_agent,
        }

    # -- Synthesis ----------------------------------------------------------

    def _synthesize_tool_results(
        self,
        query: str,
        plan: ExecutionPlan,
        exec_result: PlanExecutionResult,
    ) -> str:
        """Use LLM to synthesize MCP tool results into a cohesive user response."""
        tool_data = exec_result.get_combined_data_text()

        if not tool_data.strip():
            return "I wasn't able to retrieve the data needed. Please try rephrasing your question."

        # If only one tool was called and succeeded, and the output is
        # already well-structured, use an abbreviated synthesis prompt
        successful = [r for r in exec_result.results if r.success]
        if len(successful) == 1:
            synthesis = self._quick_synthesis(query, successful[0])
            if synthesis:
                return synthesis

        # Full synthesis prompt for multi-tool results
        prompt = f"""You are MAFA, a Multi-Agent Financial Advisor. You have just gathered data
from multiple financial analysis tools. Synthesize all the tool outputs below into
ONE clear, cohesive response for the user.

Rules:
1. Lead with the most actionable insight.
2. Merge overlapping data — don't repeat the same numbers twice.
3. Use dollar amounts and percentages where appropriate.
4. Keep the response concise: 4-8 sentences plus optional bullets.
5. If any tool failed, note what data is unavailable.
6. End with one clear next step or question for the user.
7. Never mention internal tools, MCP servers, backend processes, or system internals.
8. Never mention tool names, server keys, or technical pipeline details.

User Query: {query}

Plan reasoning: {plan.reasoning}
Synthesis hint: {plan.synthesis_hint}

Tool Results:
{tool_data}

Provide a unified, user-friendly response:"""

        try:
            result = self.llm.invoke(prompt)
            text = str(result.content) if hasattr(result, "content") else str(result)
            return text.strip() if text.strip() else "I processed your request but couldn't formulate a response. Please try again."
        except Exception as exc:
            logger.warning("Synthesis LLM failed: %s — returning raw data", exc)
            # Graceful degradation: return raw tool outputs
            return tool_data

    def _quick_synthesis(self, query: str, result) -> Optional[str]:
        """Fast synthesis for single-tool results that are already user-friendly."""
        data = result.data
        if not data:
            return None

        # Try to parse as JSON and provide a quick formatted summary
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict) and parsed.get("success") is False:
                error = parsed.get("error", "Unknown error")
                return f"Sorry, I couldn't complete that: {error}"
        except (json.JSONDecodeError, TypeError):
            pass

        # For single-tool results, use a focused synthesis prompt
        prompt = f"""You are MAFA, a financial advisor. The user asked: "{query}"

Here is the data retrieved:
{data}

Provide a clear, concise response (3-5 sentences). Lead with the key insight.
Do not mention internal tools, backends, or system details."""

        try:
            result = self.llm.invoke(prompt)
            text = str(result.content) if hasattr(result, "content") else str(result)
            return text.strip() if text.strip() else None
        except Exception:
            return None

    def _synthesize_agent_responses(self, query: str, responses: Dict[str, str]) -> str:
        """Synthesize multiple in-process agent responses (fallback path)."""
        if len(responses) == 1:
            return list(responses.values())[0]

        prompt = f"""You are MAFA, synthesising multiple specialist agent responses into one unified answer for the user.

Rules:
1. Merge overlapping insights — don't repeat the same data twice.
2. Resolve any contradictions by noting both perspectives briefly.
3. Lead with the most actionable insight, then supporting details.
4. Keep the total response concise (aim for 4-8 sentences + optional bullets).
5. End with one clear next step or question for the user.
6. Never mention internal systems, tool calls, memory stores, prompts, or backend process details.

User Query: {query}

Agent Responses:
"""
        for server, response in responses.items():
            prompt += f"\n--- {server.upper()} ---\n{response}\n"

        prompt += "\nProvide a unified response that integrates all insights:"

        try:
            result = self.llm.invoke(prompt)
            text = str(result.content) if hasattr(result, "content") else str(result)
            return sanitize_user_response(text, user_message=query)
        except Exception:
            return sanitize_user_response(
                "\n\n".join(f"**{k.title()}**: {v}" for k, v in responses.items()),
                user_message=query,
            )

    def _build_confirmation_message(self, plan: ExecutionPlan) -> str:
        """Build a user-facing confirmation prompt for trade tools."""
        trade_steps = [s for s in plan.steps if s.tool_name == "execute_trade"]
        if not trade_steps:
            return "Please confirm you'd like to proceed with this action."

        parts = []
        for step in trade_steps:
            symbol = step.params.get("symbol", "?")
            quantity = step.params.get("quantity", "?")
            action = step.params.get("action", "?").upper()
            parts.append(f"{action} {quantity} shares of {symbol}")

        actions = " and ".join(parts)
        return (
            f"I'm ready to {actions}. "
            f"Please confirm by saying 'Yes, proceed' or 'Confirm'. "
            f"Note: Market prices can change between now and execution."
        )

    # -- Company support check (preserved from original) --------------------

    async def _check_company_support(self, query: str) -> Optional[str]:
        """Return a user-facing message if the company isn't supported by the broker."""
        query_text = (query or "").strip()
        if not query_text:
            return None

        q_pred = query_text.lower()
        if any(k in q_pred for k in ("predict", "prediction", "forecast", "next-day", "next day")):
            return None

        if any(k in q_pred for k in ("stock price", "price of", "current price", "quote")):
            return None

        _account_kw = [
            "my portfolio", "my holdings", "my stock", "my shares",
            "my balance", "my watchlist", "my transactions", "my position",
            "my dashboard", "my profile", "my account", "my strategy",
            "portfolio", "holdings", "balance", "watchlist", "dashboard",
            "transactions", "all companies", "all stocks", "tradable",
            "companies can", "what companies", "list companies", "available companies",
            "supported stocks", "which stocks",
        ]
        q_lower = query_text.lower()
        if any(kw in q_lower for kw in _account_kw):
            return None

        companies = self._fetch_supported_companies()
        if not companies:
            return None

        symbols: List[str] = []
        names: List[str] = []
        for company in companies:
            symbol = str(company.get("symbol", "")).upper().strip()
            name = str(company.get("name", "")).strip()
            if symbol:
                symbols.append(symbol)
            if name:
                names.append(name)

        if not symbols:
            return None

        query_lower = query_text.lower()
        symbol_set = set(symbols)
        for symbol in symbol_set:
            if re.search(rf"\b{re.escape(symbol.lower())}\b", query_lower):
                return None

        for name in names:
            if name.lower() in query_lower:
                return None

        if not self._query_mentions_company(query_text):
            return None

        return (
            "Thanks for asking. That company is not supported by our broker yet, "
            "but it might be in the future. Please keep an eye out for updates."
        )

    @staticmethod
    def _query_mentions_company(query: str) -> bool:
        """Detect whether a query references a SPECIFIC company or ticker.

        This must be conservative — it should NOT trigger on generic financial
        queries like "promising stocks" or "investment guide".  It should only
        trigger when there's an actual ticker-like token (2-5 uppercase letters)
        that isn't a common English word.
        """
        # Generic financial phrases should NEVER trigger company check
        _GENERIC_PHRASES = [
            "promising stocks", "best stocks", "top stocks", "worst stocks",
            "invest in", "investment guide", "investment strategy",
            "market trends", "market analysis", "stock market",
            "portfolio", "rebalance", "diversify", "allocation",
            "most promising", "which stocks", "what stocks",
            "based on", "recommend", "suggestion",
        ]
        q_lower = query.lower()
        if any(phrase in q_lower for phrase in _GENERIC_PHRASES):
            return False

        # Look for specific ticker-like tokens (2-5 uppercase letters)
        _COMMON_WORDS = {
            "I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO",
            "GO", "HE", "IF", "IN", "IS", "IT", "ME", "MY", "NO",
            "OF", "OK", "ON", "OR", "SO", "TO", "UP", "US", "WE",
            "THE", "AND", "BUT", "FOR", "NOT", "YOU", "ALL", "CAN",
            "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW",
            "ITS", "LET", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY",
            "WHO", "DID", "GET", "HIM", "HIT", "HAD", "SAY", "SHE",
            "TOO", "USE", "DAD", "MOM", "SET", "RUN", "TRY", "ASK",
            "MEN", "RAN", "ANY", "DAY", "FEW", "GOT", "END",
            "WHAT", "WHEN", "WILL", "WITH", "THIS", "THAT", "FROM",
            "HAVE", "BEEN", "WANT", "SOME", "MUCH", "MANY", "ALSO",
            "BEST", "LAST", "NEXT", "HELP", "SHOW", "TELL", "GIVE",
            "MAKE", "LIKE", "LOOK", "NEED", "DOES", "THAN",
            "BUY", "SELL", "HOLD", "PUT", "CALL", "ETF", "IPO",
            "ATH", "CEO", "CFO", "GDP", "FED", "SEC",
        }
        for token in re.findall(r"\b[A-Z]{2,5}\b", query):
            if token not in _COMMON_WORDS:
                return True
        return False

    def _fetch_supported_companies(self) -> List[Dict[str, Any]]:
        """Fetch broker-supported companies with a short TTL cache."""
        now = time.time()
        if self._company_cache and (now - self._company_cache_ts) < 300:
            return self._company_cache

        try:
            response = get(f"{BROKER_API_URL}/companies", timeout=10)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", payload)
            if isinstance(data, list):
                self._company_cache = data
                self._company_cache_ts = now
                return data
        except Exception as exc:
            logger.warning("Company catalog lookup failed: %s", exc)

        return []

    # -- Error handling (preserved from original) ---------------------------

    @staticmethod
    def _extract_response_error(response: str) -> Optional[str]:
        """Return a normalized error message when agent output signals failure."""
        if not isinstance(response, str):
            return None

        text = response.strip()
        lower = text.lower()

        if lower.startswith("error processing request"):
            return text
        if "quota exceeded" in lower:
            return "Upstream LLM quota exceeded"
        if "429" in lower and "quota" in lower:
            return "Upstream LLM quota exceeded"
        if "rate limit" in lower and "openrouter" in lower:
            return "Upstream LLM rate limit exceeded"

        return None

    # -- Persistence --------------------------------------------------------

    async def _save_turn(self, user_id: int, query: str, response: str) -> None:
        """Save conversation turn to Supabase."""
        try:
            content = f"User: {query}\nAssistant: {response}"
            emb = self.vector_db.embed_text(content)
            store_user_context(
                user_id=str(user_id),
                agent="mcp_orchestrator",
                content=content,
                metadata={"user_message": query, "agent_response": response},
                embedding=emb,
            )
        except Exception as exc:
            logger.warning("Failed to save turn: %s", exc)

    async def _publish_result(
        self,
        user_id: int,
        query: str,
        response: str,
        servers_used: List[str],
        session_id: str,
        routing: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish result event to event bus."""
        await self.event_bus.publish_raw(
            MCPTopics.MCP_RESULTS,
            user_id,
            {
                "query": query,
                "response": response,
                "servers_used": servers_used,
                "routing": routing or {},
                "session_id": session_id,
            },
        )

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _build_response(
        response: str,
        servers_used: List[str],
        routing: Dict[str, Any],
        success: bool,
        start_time: float,
        user_id: int,
        session_id: str,
    ) -> Dict[str, Any]:
        """Build the standard orchestration response dict."""
        return {
            "response": response,
            "mcp_servers_used": servers_used,
            "routing": routing,
            "success": success,
            "error": None,
            "execution_time": time.time() - start_time,
            "user_id": user_id,
            "session_id": session_id,
        }

    def process_query(
        self,
        user_id: int,
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for orchestrate()."""
        return asyncio.run(
            self.orchestrate(user_id, query, session_id)
        )


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------

_orchestrator: Optional[TrueMCPOrchestrator] = None


def get_mcp_orchestrator() -> TrueMCPOrchestrator:
    """Get the global MCP orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TrueMCPOrchestrator()
    return _orchestrator
