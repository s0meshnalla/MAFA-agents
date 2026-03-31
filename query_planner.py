"""LLM Query Planner — replaces keyword routing with tool-aware execution planning.

Given a user query and the full MCP tool catalog, the planner produces a
structured execution plan: an ordered list of tool invocations with parameter
mappings.  The plan is validated, safety-checked, and ready for the
MCPExecutor to run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agents.base import model as _shared_model, sanitize_user_response
from mcp_tool_registry import MCPToolRegistry, MCPToolDescriptor

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolStep:
    """A single tool invocation within an execution plan."""

    tool_name: str
    server_key: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: Optional[int] = None  # index of a prior step this depends on
    description: str = ""  # why this step is needed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionPlan:
    """Structured execution plan produced by the LLM planner."""

    steps: List[ToolStep] = field(default_factory=list)
    reasoning: str = ""
    requires_confirmation: bool = False
    confirmation_message: str = ""
    synthesis_hint: str = ""
    confidence: float = 0.0
    fallback_agent: Optional[str] = None  # if no tools match, use this agent
    is_general_query: bool = False  # true if query doesn't need MCP tools

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @property
    def tool_names(self) -> List[str]:
        return [s.tool_name for s in self.steps]

    @property
    def server_keys(self) -> List[str]:
        seen: List[str] = []
        for s in self.steps:
            if s.server_key not in seen:
                seen.append(s.server_key)
        return seen

    @property
    def has_trade_tools(self) -> bool:
        return any(s.tool_name == "execute_trade" for s in self.steps)

    def get_parallel_groups(self) -> List[List[int]]:
        """Return groups of step indices that can run in parallel.

        Steps without dependencies (depends_on is None) and steps that
        depend on the same prior step are grouped together.
        """
        groups: Dict[Optional[int], List[int]] = {}
        for i, step in enumerate(self.steps):
            key = step.depends_on
            if key not in groups:
                groups[key] = []
            groups[key].append(i)

        # Order groups by dependency chain
        ordered: List[List[int]] = []
        # None-dependency group first (independent steps)
        if None in groups:
            ordered.append(groups.pop(None))
        # Then groups that depend on completed steps
        remaining = sorted(groups.keys(), key=lambda k: k if k is not None else -1)
        for k in remaining:
            ordered.append(groups[k])
        return ordered


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

# Safety: tools that should never be auto-executed
CONFIRMATION_REQUIRED_TOOLS = {"execute_trade"}

# Common financial question patterns that don't need MCP tools
_GENERAL_PATTERNS = [
    r"\b(what is|explain|define|how does|what does|tell me about)\b.*\b(stock market|investing|dividend|mutual fund|etf|bond|crypto|interest rate|inflation)\b",
    r"\b(hi|hello|hey|good morning|good evening|thanks|thank you|help)\b",
]


class QueryPlanner:
    """Plans MCP tool execution by having the LLM reason over the full tool catalog.

    Usage
    -----
    planner = QueryPlanner(registry)
    plan = planner.plan(query, user_id, context=...)
    """

    def __init__(
        self,
        registry: MCPToolRegistry,
        *,
        llm=None,
        max_steps: int = 6,
    ):
        self.registry = registry
        self.llm = llm or _shared_model
        self.max_steps = int(os.getenv("MCP_PLANNER_MAX_STEPS", str(max_steps)))
        self._confirmation_tools = CONFIRMATION_REQUIRED_TOOLS

    # -- Public API ---------------------------------------------------------

    def plan(
        self,
        query: str,
        user_id: int,
        context: Optional[str] = None,
    ) -> ExecutionPlan:
        """Produce an execution plan for the given user query.

        Args:
            query: The raw user query text
            user_id: Authenticated user ID (injected into tool params)
            context: Optional prior conversation context

        Returns:
            ExecutionPlan with ordered tool steps
        """
        query_text = (query or "").strip()
        if not query_text:
            return ExecutionPlan(
                is_general_query=True,
                fallback_agent="general",
                reasoning="Empty query",
            )

        # Fast-path: obviously general/greeting queries
        if self._is_general_query(query_text):
            return ExecutionPlan(
                is_general_query=True,
                fallback_agent="general",
                reasoning="General/greeting query — no MCP tools needed",
                confidence=1.0,
            )

        # Build the planner prompt with the full tool catalog
        prompt = self._build_planner_prompt(query_text, user_id, context)

        try:
            result = self.llm.invoke(prompt)
            text = str(result.content) if hasattr(result, "content") else str(result)
            plan = self._parse_plan(text, user_id)

            if plan.steps:
                logger.info(
                    "Query plan generated: %d steps, tools=%s, confidence=%.2f",
                    len(plan.steps),
                    plan.tool_names,
                    plan.confidence,
                )
            else:
                logger.info("Query plan: no tools needed, fallback=%s", plan.fallback_agent)

            return plan

        except Exception as exc:
            logger.error("Query planning failed: %s", exc)
            return ExecutionPlan(
                is_general_query=True,
                fallback_agent="general",
                reasoning=f"Planning error: {exc}",
                confidence=0.0,
            )

    # -- Prompt construction ------------------------------------------------

    def _build_planner_prompt(self, query: str, user_id: int, context: Optional[str]) -> str:
        tool_block = self.registry.get_planner_prompt_block()
        confirmation_tools = self.registry.list_confirmation_tools()
        confirmation_note = ""
        if confirmation_tools:
            confirmation_note = (
                f"\n⚠️ SAFETY: These tools require explicit user confirmation before execution: "
                f"{', '.join(confirmation_tools)}. If the user has NOT confirmed in this query, "
                f"set requires_confirmation=true and provide a confirmation_message.\n"
            )

        context_block = ""
        if context:
            context_block = f"\n--- Prior conversation context ---\n{context}\n---\n"

        return f"""You are the query planner for MAFA (Multi-Agent Financial Advisor).
Your job is to analyze the user's query and produce an execution plan that specifies
which MCP tools to call, with what parameters, and in what order.

═══ AVAILABLE MCP TOOLS ═══
{tool_block}
{confirmation_note}
═══ PLANNING RULES ═══
1. Select ONLY tools that directly help answer the user's question.
2. Prefer fewer, more targeted tool calls over broad sweeps.
3. If the query asks about a specific ticker, extract it and pass it as a parameter.
4. For portfolio/account queries, no ticker parameter is usually needed.
5. If a later tool needs output from an earlier tool, set depends_on to the earlier step index (0-based).
6. Steps WITHOUT dependencies can run in PARALLEL — use this for independent data fetches.
7. Maximum {self.max_steps} steps per plan.
8. For trade/execution requests (buy/sell), ALWAYS include validation tools before the trade tool.
9. For general conversation, greetings, or non-financial questions, return an empty steps array with fallback_agent="general".
10. The user_id for this request is {user_id}. Inject it into tool params where needed.
{context_block}
═══ USER QUERY ═══
{query}

═══ OUTPUT FORMAT ═══
Respond with ONLY a JSON object (no markdown, no commentary):
{{
  "reasoning": "Brief explanation of why these tools were selected",
  "steps": [
    {{
      "tool_name": "exact_tool_name",
      "server_key": "server_key",
      "params": {{"param1": "value1"}},
      "depends_on": null,
      "description": "why this step is needed"
    }}
  ],
  "requires_confirmation": false,
  "confirmation_message": "",
  "synthesis_hint": "How to combine tool results into a user response",
  "confidence": 0.85,
  "fallback_agent": null,
  "is_general_query": false
}}
"""

    # -- Plan parsing -------------------------------------------------------

    def _parse_plan(self, llm_output: str, user_id: int) -> ExecutionPlan:
        """Parse LLM output into a validated ExecutionPlan."""
        raw_json = self._extract_json(llm_output)
        if not raw_json:
            logger.warning("Planner returned no parseable JSON: %s", llm_output[:200])
            return ExecutionPlan(
                is_general_query=True,
                fallback_agent="general",
                reasoning="Failed to parse planner output",
            )

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.warning("Planner JSON parse error: %s", exc)
            return ExecutionPlan(
                is_general_query=True,
                fallback_agent="general",
                reasoning=f"JSON parse error: {exc}",
            )

        # Build steps
        steps: List[ToolStep] = []
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            raw_steps = []

        for i, raw_step in enumerate(raw_steps[:self.max_steps]):
            if not isinstance(raw_step, dict):
                continue

            tool_name = str(raw_step.get("tool_name", "")).strip()
            server_key = str(raw_step.get("server_key", "")).strip()

            # Validate tool exists in registry
            descriptor = self.registry.get_tool(tool_name)
            if not descriptor:
                logger.warning("Planner referenced unknown tool: %s — skipping", tool_name)
                continue

            # Override server_key from registry (planner might get it wrong)
            server_key = descriptor.server_key

            # Extract and validate params
            params = raw_step.get("params", {})
            if not isinstance(params, dict):
                params = {}

            # Auto-inject user_id where the tool expects it
            if "user_id" in descriptor.parameters.get("properties", {}) and "user_id" not in params:
                params["user_id"] = user_id

            depends_on = raw_step.get("depends_on")
            if depends_on is not None:
                try:
                    depends_on = int(depends_on)
                    if depends_on < 0 or depends_on >= i:
                        depends_on = None  # invalid dependency
                except (TypeError, ValueError):
                    depends_on = None

            description = str(raw_step.get("description", ""))

            steps.append(ToolStep(
                tool_name=tool_name,
                server_key=server_key,
                params=params,
                depends_on=depends_on,
                description=description,
            ))

        # Check if any steps require confirmation
        requires_confirmation = data.get("requires_confirmation", False)
        if any(s.tool_name in self._confirmation_tools for s in steps):
            requires_confirmation = True

        confidence = self._coerce_float(data.get("confidence"), 0.0)
        fallback_agent = data.get("fallback_agent")
        if fallback_agent and fallback_agent not in ("market", "execution", "portfolio", "strategy", "general"):
            fallback_agent = "general"

        is_general = bool(data.get("is_general_query", False))
        if not steps and not fallback_agent:
            is_general = True
            fallback_agent = "general"

        return ExecutionPlan(
            steps=steps,
            reasoning=str(data.get("reasoning", "")),
            requires_confirmation=requires_confirmation,
            confirmation_message=str(data.get("confirmation_message", "")),
            synthesis_hint=str(data.get("synthesis_hint", "")),
            confidence=confidence,
            fallback_agent=fallback_agent,
            is_general_query=is_general,
        )

    # -- Utilities ----------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first JSON object from LLM output."""
        if not text:
            return None
        cleaned = text.strip()

        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Drop first and last line (fences)
            inner_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                if line.strip() == "```" and in_block:
                    break
                if in_block:
                    inner_lines.append(line)
            cleaned = "\n".join(inner_lines).strip()

        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return cleaned[start: end + 1]

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_general_query(query: str) -> bool:
        """Quick heuristic for obviously non-financial queries."""
        q = query.lower().strip()

        # Greetings and thanks
        greetings = ("hi", "hello", "hey", "good morning", "good evening",
                     "good afternoon", "thanks", "thank you", "bye", "goodbye")
        if q in greetings or any(q.startswith(g + " ") for g in greetings[:3]):
            # Exception: "hey what's the price of AAPL" is financial
            financial_markers = ("price", "stock", "buy", "sell", "portfolio",
                                 "predict", "balance", "holding", "invest",
                                 "strategy", "risk", "trade", "news", "forecast")
            if not any(m in q for m in financial_markers):
                return True

        # Pure meta questions
        meta = ("who are you", "what can you do", "help me", "what is mafa")
        if any(m in q for m in meta):
            return True

        return False
