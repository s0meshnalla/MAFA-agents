"""MCP Tool Registry — dynamic tool discovery from MCP servers.

Connects to each configured MCP server via stdio, discovers available tools
(names, descriptions, parameter schemas) and maintains a live, queryable
tool catalog.  The planner uses this catalog to reason about which tools
can satisfy a user query.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

import sys

load_dotenv()

logger = logging.getLogger(__name__)

# Use the same Python interpreter that's running the orchestrator
# so MCP server sub-processes have access to all installed dependencies.
_PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Server configuration — mirrors the old MCP_SERVERS dict, plus extras
# ---------------------------------------------------------------------------

MCP_SERVER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "market": {
        "name": "market-research-server",
        "command": [_PYTHON, "mcp_servers/market_research_server.py"],
        "description": "LSTM price predictions, live financial news search, and combined market analysis",
        "domain": "market_data",
    },
    "execution": {
        "name": "execution-server",
        "command": [_PYTHON, "mcp_servers/execution_server.py"],
        "description": "Trade execution (buy/sell), balance checking, holdings, and trade validation",
        "domain": "trade_execution",
        "requires_confirmation": True,
    },
    "portfolio": {
        "name": "portfolio-server",
        "command": [_PYTHON, "mcp_servers/portfolio_server.py"],
        "description": "Portfolio snapshots, allocation breakdowns, risk analysis, and position checks",
        "domain": "portfolio_analysis",
    },
    "strategy": {
        "name": "strategy-server",
        "command": [_PYTHON, "mcp_servers/strategy_server.py"],
        "description": "Investment recommendations, risk profiling, rebalancing proposals, strategy adherence",
        "domain": "investment_strategy",
    },
}


# ---------------------------------------------------------------------------
# Tool descriptor — one per discovered MCP tool
# ---------------------------------------------------------------------------

@dataclass
class MCPToolDescriptor:
    """Describes a single tool exposed by an MCP server."""

    name: str
    server_key: str
    server_name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    domain: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_planner_text(self) -> str:
        """Human-readable description for inclusion in the planner prompt."""
        params_text = ""
        if self.parameters:
            properties = self.parameters.get("properties", {})
            required = set(self.parameters.get("required", []))
            parts: List[str] = []
            for pname, pinfo in properties.items():
                ptype = pinfo.get("type", "any")
                req = " (required)" if pname in required else " (optional)"
                pdesc = pinfo.get("description", "")
                parts.append(f"      - {pname}: {ptype}{req}{' — ' + pdesc if pdesc else ''}")
            if parts:
                params_text = "\n    Parameters:\n" + "\n".join(parts)
            else:
                params_text = "\n    Parameters: none"
        else:
            params_text = "\n    Parameters: none"

        confirm_note = "  ⚠️ REQUIRES USER CONFIRMATION" if self.requires_confirmation else ""
        return (
            f"  - {self.name} (server: {self.server_key}){confirm_note}\n"
            f"    {self.description}{params_text}"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MCPToolRegistry:
    """Discovers and caches tool metadata from all configured MCP servers.

    Usage
    -----
    registry = MCPToolRegistry()
    await registry.discover_all()      # connect to each server, list tools
    catalog = registry.get_catalog()    # all discovered tools
    text = registry.get_planner_prompt_block()  # for LLM planner prompt
    """

    def __init__(self, server_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        self._configs = server_configs or MCP_SERVER_CONFIGS
        self._catalog: Dict[str, MCPToolDescriptor] = {}  # tool_name → descriptor
        self._server_tools: Dict[str, List[str]] = {}  # server_key → [tool_name, ...]
        self._last_discovery: float = 0.0
        self._discovery_ttl: float = float(os.getenv("MCP_DISCOVERY_TTL", "300"))
        self._discovered = False

    # -- Public API ---------------------------------------------------------

    async def discover_all(self, force: bool = False) -> Dict[str, MCPToolDescriptor]:
        """Discover tools from every configured MCP server.

        Connects to each server sequentially via stdio, calls list_tools(),
        and caches the results.  Skips servers that fail to connect.
        """
        now = time.time()
        if self._discovered and not force and (now - self._last_discovery) < self._discovery_ttl:
            return self._catalog

        logger.info("MCP Tool Registry: starting tool discovery across %d servers …", len(self._configs))

        tasks = [self._discover_server(key, cfg) for key, cfg in self._configs.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for key, result in zip(self._configs.keys(), results):
            if isinstance(result, Exception):
                logger.warning("Tool discovery failed for server '%s': %s", key, result)

        self._last_discovery = time.time()
        self._discovered = True
        logger.info(
            "MCP Tool Registry: discovered %d tools across %d servers",
            len(self._catalog),
            len(self._server_tools),
        )
        return self._catalog

    def get_catalog(self) -> Dict[str, MCPToolDescriptor]:
        """Return the full tool catalog (tool_name → descriptor)."""
        return dict(self._catalog)

    def get_tools_for_server(self, server_key: str) -> List[MCPToolDescriptor]:
        """Return all tools belonging to a specific server."""
        names = self._server_tools.get(server_key, [])
        return [self._catalog[n] for n in names if n in self._catalog]

    def get_tool(self, tool_name: str) -> Optional[MCPToolDescriptor]:
        """Look up a single tool by name."""
        return self._catalog.get(tool_name)

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Return the server key that owns a given tool."""
        desc = self._catalog.get(tool_name)
        return desc.server_key if desc else None

    def get_planner_prompt_block(self) -> str:
        """Generate a formatted text block listing all tools for the LLM planner."""
        if not self._catalog:
            return "(No MCP tools discovered — discovery may have failed)"

        sections: List[str] = []
        for server_key in self._configs:
            tools = self.get_tools_for_server(server_key)
            if not tools:
                continue
            cfg = self._configs[server_key]
            header = f"Server: {server_key} — {cfg.get('description', '')}"
            tool_lines = "\n".join(t.to_planner_text() for t in tools)
            sections.append(f"{header}\n{tool_lines}")

        return "\n\n".join(sections)

    def list_confirmation_tools(self) -> List[str]:
        """Return names of tools that require user confirmation before execution."""
        return [name for name, desc in self._catalog.items() if desc.requires_confirmation]

    @property
    def is_discovered(self) -> bool:
        return self._discovered

    @property
    def tool_count(self) -> int:
        return len(self._catalog)

    # -- Internal -----------------------------------------------------------

    async def _discover_server(self, server_key: str, cfg: Dict[str, Any]) -> None:
        """Connect to one MCP server via stdio and discover its tools."""
        command_parts = cfg["command"]
        server_params = StdioServerParameters(
            command=command_parts[0],
            args=command_parts[1:],
            env=os.environ.copy(),
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()

                    tools_list = tools_result.tools if hasattr(tools_result, "tools") else tools_result
                    if not isinstance(tools_list, list):
                        tools_list = []

                    server_requires_confirm = cfg.get("requires_confirmation", False)
                    domain = cfg.get("domain", "")
                    tool_names: List[str] = []

                    for tool in tools_list:
                        tool_name = getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None)
                        if not tool_name:
                            continue

                        description = getattr(tool, "description", "") or (tool.get("description", "") if isinstance(tool, dict) else "")

                        # Extract input schema
                        input_schema = getattr(tool, "inputSchema", None) or (tool.get("inputSchema") if isinstance(tool, dict) else None)
                        if input_schema is None:
                            input_schema = getattr(tool, "input_schema", None) or (tool.get("input_schema") if isinstance(tool, dict) else None)
                        params = input_schema if isinstance(input_schema, dict) else {}

                        # Mark execution tools as requiring confirmation
                        is_trade_tool = tool_name in ("execute_trade",)
                        requires_confirm = server_requires_confirm and is_trade_tool

                        descriptor = MCPToolDescriptor(
                            name=tool_name,
                            server_key=server_key,
                            server_name=cfg.get("name", server_key),
                            description=description,
                            parameters=params,
                            requires_confirmation=requires_confirm,
                            domain=domain,
                        )

                        self._catalog[tool_name] = descriptor
                        tool_names.append(tool_name)

                    self._server_tools[server_key] = tool_names
                    logger.info(
                        "Discovered %d tools from server '%s': %s",
                        len(tool_names),
                        server_key,
                        tool_names,
                    )

        except Exception as exc:
            logger.error("Failed to discover tools from MCP server '%s': %s", server_key, exc)
            raise


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_registry: Optional[MCPToolRegistry] = None


def get_tool_registry() -> MCPToolRegistry:
    """Return the global MCPToolRegistry instance."""
    global _registry
    if _registry is None:
        _registry = MCPToolRegistry()
    return _registry
