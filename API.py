"""MAFA Agents API - FastAPI Backend with TRUE MCP Integration.

Provides REST endpoints for all agents plus WebSocket streaming for real-time updates.
Features: Rate limiting, input validation, structured logging, health checks.
"""

import base64
import binascii
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from mcp_orchestrator import get_mcp_orchestrator
from event_bus import get_event_bus, MCPTopics, MCPEvent, shutdown_event_bus
from http_client import set_request_token, init_request_cache, clear_request_cache
from monitoring import (
    setup_logging, log_mcp_request, metrics,
    check_redis, check_supabase, check_broker_api, check_mcp_servers
)

# Configure structured logging
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

# Environment configuration
BROKER_API_URL = os.getenv("BROKER_API_URL", "http://localhost:8080")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")


# ---------------------------------------------------------------------------
# Rate Limiting (simple in-memory implementation)
# ---------------------------------------------------------------------------
from collections import defaultdict
import time

class RateLimiter:
    """Simple in-memory rate limiter with periodic stale-key cleanup."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict = defaultdict(list)
        self._last_cleanup: float = 0.0
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for given key."""
        now = time.time()
        self._maybe_cleanup(now)
        window_start = now - self.window_seconds
        
        # Clean old requests
        self._requests[key] = [t for t in self._requests[key] if t > window_start]
        
        if len(self._requests[key]) >= self.max_requests:
            return False
        
        self._requests[key].append(now)
        return True

    def _maybe_cleanup(self, now: float) -> None:
        """Prune stale IP keys every 5 minutes to prevent unbounded memory growth."""
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        cutoff = now - self.window_seconds
        stale = [k for k, v in self._requests.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._requests[k]

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


# ---------------------------------------------------------------------------
# Request Models with Validation
# ---------------------------------------------------------------------------

class ExecuteAgentRequest(BaseModel):
    """Request model for legacy agent endpoints."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    userId: int = Field(..., ge=1, description="User ID must be positive")
    sessionId: Optional[str] = Field(None, max_length=100)


class MCPQueryRequest(BaseModel):
    """Request model for MCP orchestration endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    userId: int = Field(..., ge=1, description="User ID must be positive")
    sessionId: Optional[str] = Field(None, max_length=100)
    
    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        """Input sanitization that avoids false positives on normal finance language.

        Important: keep regular user phrases like "stock drops 10%" valid.
        """
        text = v.strip()
        lower = text.lower()

        # Allow educational/meta references to SQL-like text when clearly
        # framed as quoted examples, not executable instructions.
        meta_context_markers = (
            "sql-like",
            "example",
            "quoted",
            "string",
            "contains",
            "my query includes",
            "treat this safely",
            "ignore that part",
            "from another app",
            "pasting this text",
        )
        has_meta_context = any(m in lower for m in meta_context_markers)

        # XSS-style payloads
        if "<script" in lower or "javascript:" in lower:
            raise ValueError("Query contains potentially dangerous content")

        # SQL-ish patterns (word-boundary + command context) to avoid matching
        # benign words like "drop" in market scenarios.
        sql_patterns = [
            r"\b(drop|delete|truncate)\s+(table|database|from)\b",
            r"\bunion\s+select\b",
            r"\binsert\s+into\b",
            r"\bupdate\s+\w+\s+set\b",
            r"\bselect\s+.+\s+from\b",
            r"\b(or|and)\s+1\s*=\s*1\b",
            r"--",
            r"/\*|\*/",
        ]
        for pattern in sql_patterns:
            if re.search(pattern, lower):
                if not has_meta_context:
                    raise ValueError("Query contains potentially dangerous content")

        return text


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as exc:
                logger.error(f"Error broadcasting to WebSocket: {exc}")


ws_manager = ConnectionManager()


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize MCP and event bus."""
    logger.info("Starting MAFA Agents API...")

    # Log memory backend readiness early for production diagnostics.
    supabase_status = await check_supabase(SUPABASE_URL, SUPABASE_API_KEY)
    if supabase_status.get("status") == "healthy":
        logger.info(
            "Supabase memory backend healthy",
            extra={"extra_data": {"event": "supabase_startup_ok", **supabase_status}},
        )
    else:
        logger.warning(
            "Supabase memory backend unavailable at startup",
            extra={"extra_data": {"event": "supabase_startup_unhealthy", **supabase_status}},
        )
    
    # Initialize MCP orchestrator
    orchestrator = get_mcp_orchestrator()
    await orchestrator.initialize()
    
    # Initialize event bus (Redis) — gracefully degrade if unavailable
    _event_bus_available = False
    try:
        event_bus = get_event_bus()
        await event_bus.connect()

        # Subscribe to MCP results for WebSocket streaming
        async def broadcast_event(event: MCPEvent):
            await ws_manager.broadcast(event.to_dict())

        await event_bus.subscribe(MCPTopics.MCP_RESULTS, broadcast_event)
        _event_bus_available = True
        logger.info("Event bus (Redis) connected")
    except Exception as exc:
        logger.warning(f"Redis unavailable — event bus disabled: {exc}")
        logger.warning("WebSocket streaming and pub/sub will not work. Core agent endpoints are unaffected.")
    
    logger.info("MAFA Agents API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MAFA Agents API...")
    await orchestrator.shutdown()
    if _event_bus_available:
        await shutdown_event_bus()
    logger.info("MAFA Agents API shutdown complete")


app = FastAPI(
    title="MAFA Agents API",
    description="Multi-Agent Financial Advisor with TRUE MCP Integration",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware with configurable origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to all requests."""
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests", "message": "Please try again later"}
        )
    
    response = await call_next(request)
    return response


# Use TRUE MCP Orchestrator (replaces old MCPCoordinator)
mcp_orchestrator = get_mcp_orchestrator()

# ---------------------------------------------------------------------------
# Auth Dependency – eliminates repeated header validation in every endpoint
# ---------------------------------------------------------------------------
_bearer_scheme = HTTPBearer(auto_error=True)


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without signature verification for basic claim checks.

    This is used for fast rejection of obviously invalid/expired tokens at the
    API boundary. Signature/authorization is still enforced by downstream broker
    calls that require valid credentials.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Invalid bearer token format")

    payload_b64 = parts[1]
    # JWT uses URL-safe base64 and may omit padding.
    payload_b64 += "=" * (-len(payload_b64) % 4)

    try:
        payload_raw = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
        payload = json.loads(payload_raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid bearer token payload")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="Invalid bearer token payload")

    return payload


def _validate_token_claims(token: str) -> None:
    """Validate minimal JWT claims to reject malformed/expired tokens early."""
    payload = _decode_jwt_payload(token)
    exp = payload.get("exp")
    if exp is None:
        return

    try:
        exp_int = int(exp)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token expiry claim")

    if exp_int <= int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")


def _is_upstream_quota_error(exc: Exception) -> bool:
    """Detect quota/rate-limit failures from upstream LLM providers."""
    text = str(exc).lower()
    return (
        "quota exceeded" in text
        or ("429" in text and "quota" in text)
        or ("429" in text and "rate limit" in text)
        or "generate_content_free_tier_requests" in text
    )


def get_token(credentials: HTTPAuthorizationCredentials = Security(_bearer_scheme)) -> str:
    """Extract Bearer token.  FastAPI returns 401/403 automatically if missing."""
    token = credentials.credentials
    _validate_token_claims(token)
    return token


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint for monitoring."""
    # Check all dependencies
    redis_status = await check_redis(REDIS_URL)
    supabase_status = await check_supabase(SUPABASE_URL, SUPABASE_API_KEY)
    broker_status = await check_broker_api(BROKER_API_URL)
    mcp_status = check_mcp_servers()
    
    checks = {
        "redis": redis_status,
        "supabase": supabase_status,
        "broker_api": broker_status,
        "mcp_servers": mcp_status,
    }
    
    # Determine overall status — 'unavailable' means a dependency is down (degraded, not healthy)
    critical_healthy = all(
        checks[k].get("status") == "healthy"
        for k in ["redis", "supabase", "mcp_servers"]
    )
    
    return {
        "status": "healthy" if critical_healthy else "degraded",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mcp_enabled": True,
        "websocket_connections": len(ws_manager.active_connections),
        "checks": checks,
    }


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics for monitoring."""
    return metrics.get_metrics()


@app.get("/mcp/servers")
async def list_mcp_servers():
    """List available MCP servers and their dynamically discovered tools."""
    registry = mcp_orchestrator.registry
    if not registry.is_discovered:
        return {
            "status": "not_ready",
            "message": "Tool discovery has not completed yet. Try again shortly.",
            "servers": {},
        }

    servers_info = {}
    for server_key in registry._server_tools:
        tools = registry.get_tools_for_server(server_key)
        servers_info[server_key] = {
            "name": tools[0].server_name if tools else server_key,
            "tools": [t.name for t in tools],
            "tool_count": len(tools),
            "status": "available",
            "details": [
                {
                    "name": t.name,
                    "description": t.description,
                    "requires_confirmation": t.requires_confirmation,
                }
                for t in tools
            ],
        }

    return {
        "status": "ready",
        "total_tools": registry.tool_count,
        "servers": servers_info,
    }


# ---------------------------------------------------------------------------
# WebSocket Streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/mcp-stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time MCP event streaming."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            await websocket.send_json({"type": "ack", "message": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}")
        ws_manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Agent Endpoints — ALL route through the MCP orchestrator pipeline
# ---------------------------------------------------------------------------

async def _run_via_orchestrator(payload: ExecuteAgentRequest, token: str) -> dict:
    """Route any agent request through the MCP orchestrator pipeline.

    Every endpoint now benefits from dynamic tool discovery, LLM-planned
    execution, and MCP-protocol tool invocation.  The orchestrator falls
    back to in-process agents automatically when MCP servers are unavailable.
    """
    set_request_token(token)
    init_request_cache()
    session_id = payload.sessionId or str(uuid.uuid4())
    try:
        result = await mcp_orchestrator.orchestrate(
            user_id=payload.userId,
            query=payload.query,
            session_id=session_id,
        )
        return {
            "data": result.get("response", ""),
            "userId": payload.userId,
            "sessionId": session_id,
            "success": result.get("success", True),
            "error": result.get("error"),
            "routing": result.get("routing", {}),
            "mcp_servers_used": result.get("mcp_servers_used", []),
            "execution_time": result.get("execution_time", 0),
        }
    except Exception as exc:
        logger.error(f"MCP orchestrator error: {exc}")
        if _is_upstream_quota_error(exc):
            return {
                "data": f"Error processing request: {str(exc)}",
                "userId": payload.userId,
                "sessionId": session_id,
                "success": False,
                "error": str(exc),
            }
        raise HTTPException(status_code=500, detail="Failed to process your request. Please try again later.") from exc
    finally:
        clear_request_cache()
        set_request_token(None)


@app.post("/execute-agent")
async def execute_agent_endpoint(payload: ExecuteAgentRequest, token: str = Depends(get_token)):
	"""HTTP endpoint — routes through MCP orchestrator for intelligent tool selection."""
	return await _run_via_orchestrator(payload, token)


@app.post("/investment-strategy-agent")
async def investment_strategy_agent_endpoint(payload: ExecuteAgentRequest, token: str = Depends(get_token)):
	"""HTTP endpoint — routes through MCP orchestrator for intelligent tool selection."""
	return await _run_via_orchestrator(payload, token)

@app.post("/market-research-agent")
async def market_research_agent_endpoint(payload: ExecuteAgentRequest, token: str = Depends(get_token)):
	"""HTTP endpoint — routes through MCP orchestrator for intelligent tool selection."""
	return await _run_via_orchestrator(payload, token)


@app.post("/general-agent")
async def general_agent_endpoint(payload: ExecuteAgentRequest, token: str = Depends(get_token)):
	"""HTTP endpoint — routes through MCP orchestrator for intelligent tool selection."""
	return await _run_via_orchestrator(payload, token)

@app.post("/portfolio-manager-agent")
async def portfolio_manager_agent_endpoint(payload: ExecuteAgentRequest, token: str = Depends(get_token)):
	"""HTTP endpoint — routes through MCP orchestrator for intelligent tool selection."""
	return await _run_via_orchestrator(payload, token)


@app.post("/mcp/query")
@log_mcp_request
async def mcp_query_endpoint(payload: MCPQueryRequest, token: str = Depends(get_token)):
	"""TRUE MCP orchestration endpoint that routes queries across MCP servers."""
	set_request_token(token)
	init_request_cache()
	try:
		session_id = payload.sessionId or str(uuid.uuid4())
		# Use TRUE MCP Orchestrator
		result = await mcp_orchestrator.orchestrate(
			user_id=payload.userId,
			query=payload.query,
			session_id=session_id,
		)
		return result
	except Exception as exc:
		logger.error(f"MCP orchestration error: {exc}")
		# Return proper HTTP error instead of 200 with error body
		raise HTTPException(status_code=500, detail="Failed to process your request. Please try again later.") from exc
	finally:
		clear_request_cache()
		set_request_token(None)


# ---------------------------------------------------------------------------
# Direct MCP Server Endpoints (for testing individual servers)
# ---------------------------------------------------------------------------

@app.post("/mcp/market/predict")
async def mcp_market_predict(symbol: str, token: str = Depends(get_token)):
	"""Direct call to market research MCP server for prediction."""
	import json as _json
	from tools.market_research_tools import predict
	try:
		# Use .invoke() for LangChain StructuredTool objects
		if hasattr(predict, 'invoke'):
			result = predict.invoke({"ticker": symbol.upper()})
		elif hasattr(predict, 'run'):
			result = predict.run(symbol.upper())
		else:
			result = predict.func(symbol.upper()) if hasattr(predict, 'func') else '{"error": "no predict function"}'
		# predict tool returns a JSON string like '{"ticker":"AAPL","predicted_close":255.73}'
		parsed = _json.loads(result) if isinstance(result, str) else result
		if "error" in parsed:
			return {"symbol": symbol.upper(), "error": parsed["error"], "success": False}
		return {"symbol": symbol.upper(), "predicted_close": float(parsed["predicted_close"]), "success": True}
	except Exception as exc:
		logger.warning(f"Market predict error for {symbol}: {exc}")
		return {"symbol": symbol.upper(), "error": str(exc), "success": False}


@app.post("/mcp/execution/validate")
async def mcp_execution_validate(
	symbol: str,
	quantity: int,
	action: str,
	token: str = Depends(get_token),
):
	"""Direct call to execution MCP server for trade validation."""
	set_request_token(token)
	init_request_cache()
	
	import json as _json
	from tools.profile_tools import get_user_balance, get_user_holdings, get_current_stock_price
	try:
		# Tools return JSON strings — parse them into native Python values
		balance_raw = get_user_balance.invoke({}) if hasattr(get_user_balance, 'invoke') else get_user_balance.func()
		holdings_raw = get_user_holdings.invoke({}) if hasattr(get_user_holdings, 'invoke') else get_user_holdings.func()
		price_raw = get_current_stock_price.invoke({"symbol": symbol.upper()}) if hasattr(get_current_stock_price, 'invoke') else get_current_stock_price.func(symbol.upper())
		
		# Parse JSON strings: {"balance": 172412.70}, {"symbol":"AAPL","price":264.72}
		balance_data = _json.loads(balance_raw) if isinstance(balance_raw, str) else balance_raw
		balance = float(balance_data.get("balance", 0)) if isinstance(balance_data, dict) else float(balance_data)
		
		price_data = _json.loads(price_raw) if isinstance(price_raw, str) else price_raw
		price = float(price_data.get("price", 0)) if isinstance(price_data, dict) else float(price_data)
		
		try:
			holdings_list = _json.loads(holdings_raw) if isinstance(holdings_raw, str) else holdings_raw
		except (ValueError, TypeError):
			holdings_list = []
		
		validation = {"symbol": symbol.upper(), "quantity": quantity, "action": action, "valid": True, "issues": []}
		
		if action.lower() == "buy":
			total_cost = price * quantity
			if total_cost > balance:
				validation["valid"] = False
				validation["issues"].append(f"Insufficient balance: need ${total_cost:.2f}, have ${balance:.2f}")
		elif action.lower() == "sell":
			owned = 0
			if isinstance(holdings_list, list):
				for h in holdings_list:
					if isinstance(h, dict) and h.get("symbol", "").upper() == symbol.upper():
						owned = h.get("quantity", 0)
						break
			if owned < quantity:
				validation["valid"] = False
				validation["issues"].append(f"Insufficient shares: own {owned}, selling {quantity}")
		
		validation["success"] = True
		return validation
	except Exception as exc:
		return {"symbol": symbol.upper(), "error": str(exc), "success": False}
	finally:
		clear_request_cache()
		set_request_token(None)
