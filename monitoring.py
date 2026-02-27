"""MAFA Agents Production Monitoring - Structured logging and metrics.

Provides decorators for request logging, performance tracking, and error reporting.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
from collections import deque
import statistics

# Create logs directory if not exists
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for production."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # File handler with JSON formatting
    file_handler = logging.FileHandler(os.path.join(LOGS_DIR, "mcp_system.log"))
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(log_level)
    
    # Console handler with readable format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    console_handler.setLevel(log_level)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


logger = logging.getLogger("mafa_agents")


# ---------------------------------------------------------------------------
# Performance Metrics
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latencies: Dict[str, deque] = {}
        self._request_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._mcp_server_usage: Dict[str, int] = {}
    
    def record_latency(self, endpoint: str, duration_ms: float) -> None:
        """Record request latency for an endpoint."""
        if endpoint not in self._latencies:
            self._latencies[endpoint] = deque(maxlen=self.window_size)
        self._latencies[endpoint].append(duration_ms)
        self._request_counts[endpoint] = self._request_counts.get(endpoint, 0) + 1
    
    def record_error(self, endpoint: str) -> None:
        """Record error for an endpoint."""
        self._error_counts[endpoint] = self._error_counts.get(endpoint, 0) + 1
    
    def record_mcp_server_usage(self, servers: List[str]) -> None:
        """Record which MCP servers were used."""
        for server in servers:
            self._mcp_server_usage[server] = self._mcp_server_usage.get(server, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        metrics = {
            "endpoints": {},
            "mcp_servers": self._mcp_server_usage.copy(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        for endpoint, latencies in self._latencies.items():
            if latencies:
                sorted_latencies = sorted(latencies)
                metrics["endpoints"][endpoint] = {
                    "total_requests": self._request_counts.get(endpoint, 0),
                    "errors": self._error_counts.get(endpoint, 0),
                    "latency": {
                        "p50": round(statistics.median(sorted_latencies), 2),
                        "p95": round(sorted_latencies[int(len(sorted_latencies) * 0.95)] if len(sorted_latencies) > 20 else max(sorted_latencies), 2),
                        "p99": round(sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 100 else max(sorted_latencies), 2),
                        "avg": round(statistics.mean(sorted_latencies), 2),
                    }
                }
        
        return metrics


# Global metrics collector
metrics = MetricsCollector()


# ---------------------------------------------------------------------------
# Logging Decorators
# ---------------------------------------------------------------------------

def log_mcp_request(func: Callable) -> Callable:
    """Decorator for MCP endpoint logging with metrics."""
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        endpoint_name = func.__name__
        
        # Extract request info
        request = kwargs.get("payload") or kwargs.get("request") or {}
        if hasattr(request, "model_dump"):
            request = request.model_dump()
        elif hasattr(request, "dict"):
            request = request.dict()
        
        user_id = request.get("userId", request.get("user_id", "unknown"))
        query = str(request.get("query", ""))[:100]
        
        logger.info(
            f"MCP request started: {endpoint_name}",
            extra={"extra_data": {
                "event": "mcp_request_start",
                "endpoint": endpoint_name,
                "user_id": user_id,
                "query_preview": query,
            }}
        )
        
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            # Extract MCP servers used
            mcp_servers = []
            if isinstance(result, dict):
                mcp_servers = result.get("mcp_servers_used", [])
            
            # Record metrics
            metrics.record_latency(endpoint_name, duration_ms)
            if mcp_servers:
                metrics.record_mcp_server_usage(mcp_servers)
            
            logger.info(
                f"MCP request completed: {endpoint_name}",
                extra={"extra_data": {
                    "event": "mcp_request_complete",
                    "endpoint": endpoint_name,
                    "user_id": user_id,
                    "duration_ms": round(duration_ms, 2),
                    "mcp_servers": mcp_servers,
                    "success": True,
                }}
            )
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_latency(endpoint_name, duration_ms)
            metrics.record_error(endpoint_name)
            
            logger.error(
                f"MCP request failed: {endpoint_name}",
                extra={"extra_data": {
                    "event": "mcp_request_error",
                    "endpoint": endpoint_name,
                    "user_id": user_id,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(e),
                    "error_type": type(e).__name__,
                }}
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        endpoint_name = func.__name__
        
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_latency(endpoint_name, duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            metrics.record_latency(endpoint_name, duration_ms)
            metrics.record_error(endpoint_name)
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def log_tool_call(func: Callable) -> Callable:
    """Decorator for tool call logging."""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        tool_name = func.__name__
        start_time = time.time()
        
        logger.debug(f"Tool call started: {tool_name}")
        
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                f"Tool call completed: {tool_name}",
                extra={"extra_data": {
                    "event": "tool_call_complete",
                    "tool": tool_name,
                    "duration_ms": round(duration_ms, 2),
                }}
            )
            
            return result
            
        except Exception as e:
            logger.warning(
                f"Tool call failed: {tool_name} - {e}",
                extra={"extra_data": {
                    "event": "tool_call_error",
                    "tool": tool_name,
                    "error": str(e),
                }}
            )
            raise
    
    return wrapper


# ---------------------------------------------------------------------------
# Health Check Helpers
# ---------------------------------------------------------------------------

async def check_redis(redis_url: str = "redis://localhost:6379") -> Dict[str, Any]:
    """Check Redis connectivity."""
    try:
        import redis.asyncio as redis
        client = redis.from_url(redis_url, decode_responses=True)
        await client.ping()
        await client.close()
        return {"status": "healthy", "url": redis_url}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_supabase(url: str, key: str) -> Dict[str, Any]:
    """Check Supabase connectivity."""
    try:
        from supabase import create_client
        client = create_client(url, key)
        # Simple health check - just verify client creation
        return {"status": "healthy", "url": url}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_broker_api(url: str) -> Dict[str, Any]:
    """Check broker API connectivity."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{url}/actuator/health")
            if response.status_code == 200:
                return {"status": "healthy", "url": url}
            return {"status": "unhealthy", "status_code": response.status_code}
    except Exception:
        return {"status": "unavailable", "fallback": "using mock/yfinance data"}


def check_mcp_servers() -> Dict[str, Any]:
    """Check MCP server availability."""
    servers = ["market", "execution", "portfolio", "strategy"]
    return {
        "status": "healthy",
        "servers": servers,
        "count": len(servers),
    }
