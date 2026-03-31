"""MCP Execution Engine — runs tool plans via actual MCP protocol.

Takes an ExecutionPlan from the QueryPlanner and invokes each tool via
stdio transport against the appropriate MCP server.  Supports parallel
execution of independent steps.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from http_client import _request_token as _parent_token_var
from mcp_tool_registry import MCP_SERVER_CONFIGS, MCPToolRegistry
from query_planner import ExecutionPlan, ToolStep

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of a single MCP tool invocation."""

    tool_name: str
    server_key: str
    success: bool
    data: str = ""
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "server_key": self.server_key,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
        }


@dataclass
class PlanExecutionResult:
    """Aggregate result of executing a full plan."""

    results: List[ToolResult] = field(default_factory=list)
    total_time: float = 0.0
    servers_used: List[str] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)
    success: bool = True
    partial_failure: bool = False  # some tools failed but others succeeded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_time": self.total_time,
            "servers_used": self.servers_used,
            "tools_called": self.tools_called,
            "success": self.success,
            "partial_failure": self.partial_failure,
        }

    def get_result_by_tool(self, tool_name: str) -> Optional[ToolResult]:
        """Find the result for a specific tool invocation."""
        for r in self.results:
            if r.tool_name == tool_name:
                return r
        return None

    def get_successful_data(self) -> Dict[str, str]:
        """Return a mapping of tool_name → data for all successful results."""
        return {r.tool_name: r.data for r in self.results if r.success}

    def get_combined_data_text(self) -> str:
        """Combine all successful tool outputs into a single text block."""
        parts: List[str] = []
        for r in self.results:
            if r.success and r.data:
                parts.append(f"--- {r.tool_name} (from {r.server_key} server) ---\n{r.data}")
            elif not r.success:
                parts.append(f"--- {r.tool_name} (FAILED) ---\n{r.error or 'Unknown error'}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class MCPExecutor:
    """Executes MCP tool plans via stdio transport.

    Usage
    -----
    executor = MCPExecutor(registry)
    result = await executor.execute(plan)
    """

    def __init__(
        self,
        registry: MCPToolRegistry,
        server_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_timeout: float = 30.0,
    ):
        self.registry = registry
        self._configs = server_configs or MCP_SERVER_CONFIGS
        self.tool_timeout = float(os.getenv("MCP_TOOL_TIMEOUT", str(tool_timeout)))
        # Per-server locks prevent concurrent subprocess spawns to the same
        # MCP server, which causes TaskGroup errors.  Parallel execution
        # still works across different servers.
        self._server_locks: Dict[str, asyncio.Lock] = {}

    # -- Public API ---------------------------------------------------------

    async def execute(self, plan: ExecutionPlan) -> PlanExecutionResult:
        """Execute an entire plan, respecting dependencies and parallelism.

        Steps without dependencies are run in parallel.  Steps that depend
        on earlier steps wait for those to complete first.
        """
        start = time.time()

        if not plan.steps:
            return PlanExecutionResult(total_time=time.time() - start)

        # Build result slots
        results: List[Optional[ToolResult]] = [None] * len(plan.steps)

        # Execute in parallel groups
        groups = plan.get_parallel_groups()
        for group in groups:
            # Run all steps in this group concurrently
            tasks = [
                self._execute_step(plan.steps[idx], idx, results)
                for idx in group
            ]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for idx, result in zip(group, group_results):
                if isinstance(result, Exception):
                    results[idx] = ToolResult(
                        tool_name=plan.steps[idx].tool_name,
                        server_key=plan.steps[idx].server_key,
                        success=False,
                        error=str(result),
                    )
                # If result is ToolResult, it was already stored by _execute_step

        # Collect results
        final_results = [r for r in results if r is not None]
        servers_used: List[str] = []
        tools_called: List[str] = []
        any_success = False
        any_failure = False

        for r in final_results:
            if r.server_key not in servers_used:
                servers_used.append(r.server_key)
            if r.tool_name not in tools_called:
                tools_called.append(r.tool_name)
            if r.success:
                any_success = True
            else:
                any_failure = True

        total_time = time.time() - start

        return PlanExecutionResult(
            results=final_results,
            total_time=total_time,
            servers_used=servers_used,
            tools_called=tools_called,
            success=any_success,
            partial_failure=any_success and any_failure,
        )

    async def execute_single_tool(
        self,
        tool_name: str,
        server_key: str,
        params: Dict[str, Any],
    ) -> ToolResult:
        """Execute a single tool directly (for ad-hoc calls outside a plan)."""
        step = ToolStep(tool_name=tool_name, server_key=server_key, params=params)
        return await self._run_tool_via_mcp(step)

    # -- Internal -----------------------------------------------------------

    async def _execute_step(
        self,
        step: ToolStep,
        step_idx: int,
        results: List[Optional[ToolResult]],
    ) -> None:
        """Execute one step and store the result."""
        tool_result = await self._run_tool_via_mcp(step)
        results[step_idx] = tool_result

    async def _run_tool_via_mcp(self, step: ToolStep) -> ToolResult:
        """Connect to the MCP server and invoke the tool.

        Uses a per-server lock so that multiple calls to the same server
        are serialized (each spawns a subprocess; concurrent spawns race).
        Calls to *different* servers still run in parallel.
        """
        # Lazy-create a lock per server key
        if step.server_key not in self._server_locks:
            self._server_locks[step.server_key] = asyncio.Lock()

        async with self._server_locks[step.server_key]:
            return await self._run_tool_via_mcp_locked(step)

    async def _run_tool_via_mcp_locked(self, step: ToolStep) -> ToolResult:
        """Actual MCP invocation (called under server lock)."""
        start = time.time()
        config = self._configs.get(step.server_key)
        if not config:
            return ToolResult(
                tool_name=step.tool_name,
                server_key=step.server_key,
                success=False,
                error=f"Unknown server key: {step.server_key}",
                execution_time=time.time() - start,
            )

        server_params = StdioServerParameters(
            command=config["command"][0],
            args=config["command"][1:],
            env=self._build_subprocess_env(),
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Call tool with timeout
                    result = await asyncio.wait_for(
                        session.call_tool(step.tool_name, step.params),
                        timeout=self.tool_timeout,
                    )

                    # Extract text content from MCP result
                    data = self._extract_text(result)

                    execution_time = time.time() - start
                    logger.info(
                        "MCP tool %s.%s completed in %.2fs",
                        step.server_key,
                        step.tool_name,
                        execution_time,
                    )

                    return ToolResult(
                        tool_name=step.tool_name,
                        server_key=step.server_key,
                        success=True,
                        data=data,
                        execution_time=execution_time,
                    )

        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.warning(
                "MCP tool %s.%s timed out after %.1fs",
                step.server_key,
                step.tool_name,
                elapsed,
            )
            return ToolResult(
                tool_name=step.tool_name,
                server_key=step.server_key,
                success=False,
                error=f"Tool timed out after {self.tool_timeout}s",
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(
                "MCP tool %s.%s failed: %s",
                step.server_key,
                step.tool_name,
                exc,
            )
            return ToolResult(
                tool_name=step.tool_name,
                server_key=step.server_key,
                success=False,
                error=str(exc),
                execution_time=elapsed,
            )

    @staticmethod
    def _build_subprocess_env() -> dict:
        """Build env dict for MCP subprocess, injecting the parent's auth token.

        MCP servers run as separate processes and have no access to the parent's
        contextvars.  We bridge the gap by setting MAFA_AUTH_TOKEN in the env.
        The MCP server reads this at startup and calls set_request_token().
        """
        env = os.environ.copy()
        try:
            token = _parent_token_var.get()
            if token:
                env["MAFA_AUTH_TOKEN"] = token
        except LookupError:
            pass
        return env

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Normalize MCP tool result into a plain text string."""
        content = getattr(result, "content", None)
        if content is None:
            return "{}"
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else "{}"
        return str(content)
