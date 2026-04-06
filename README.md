# MAFA-Agents

Multi-Agent Financial Assistant powered by Model Context Protocol(MCP) and OpenRouter.

## Overview

MAFA-Agents is a sophisticated multi-agent system for financial analysis and trading assistance. It uses the official MCP protocol to coordinate multiple specialized agents:

- **Market Research Agent** - LSTM-based price predictions and live news search
- **Execution Agent** - Trade execution and balance checking
- **Portfolio Manager Agent** - Portfolio analysis and risk metrics
- **Investment Strategy Agent** - Recommendations and rebalancing strategies
- **General Agent** - Handles general queries and routing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                        │
│  (REST API + WebSocket Streaming + Rate Limiting)           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  MCP Orchestrator                           │
│  (OpenRouter LLM + Intent Classification + Tool Routing)    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   MCP Servers (stdio)                       │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│   Market    │  Execution  │  Portfolio  │     Strategy      │
│   Server    │   Server    │   Server    │      Server       │
└─────────────┴─────────────┴─────────────┴───────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      Data Layer                             │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Supabase   │    Redis     │    LSTM      │   Yahoo        │
│  Vector DB   │  Event Bus   │   Models     │  Finance       │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for Redis)
- API Keys (see `.env.example`)

### Installation

```bash
# Clone and navigate to project
cd MAFA-agents

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Start Services

```bash
# Start Redis (Docker)
docker run -d --name mafa-redis -p 6379:6379 redis:alpine

# Start API Server
.\start_server.ps1  # Windows
# python -m uvicorn API:app --host 0.0.0.0 --port 5001  # Linux/Mac
```

### Smoke Test the System

```bash
python -m uvicorn API:app --host 0.0.0.0 --port 5001
# Then call /health and /mcp/query with a valid bearer token.
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/mcp/query` | POST | Main orchestrator endpoint |
| `/mcp/servers` | GET | List available MCP servers |
| `/mcp/market/predict` | POST | Direct LSTM prediction |
| `/ws/mcp-stream` | WS | Real-time event streaming |

## Project Structure

```
MAFA-agents/
├── API.py                 # FastAPI application
├── mcp_orchestrator.py    # MCP orchestration logic
├── event_bus.py           # Redis pub/sub event bus
├── monitoring.py          # Logging & metrics
├── vectordbsupabase.py    # Supabase vector DB client
├── http_client.py         # HTTP client utilities
│
├── agents/                # LangChain agent implementations
│   ├── execution_agent.py
│   ├── general_agent.py
│   ├── investment_strategy_agent.py
│   ├── market_search_agent.py
│   └── portfolio_manager_agent.py
│
├── mcp_servers/           # MCP protocol servers
│   ├── execution_server.py
│   ├── market_research_server.py
│   ├── portfolio_server.py
│   └── strategy_server.py
│
├── tools/                 # Agent tools
│   ├── execute_trade_tools.py
│   ├── investment_strategy_tools.py
│   ├── market_research_tools.py
│   ├── memory_tools.py
│   └── profile_tools.py
│
├── lstm/                  # LSTM prediction models
│   ├── infer.py
│   ├── predict_next_day.py
│   └── output/            # Trained models per ticker
│
└── logs/                  # Application logs
```

## Configuration

Required environment variables (see `.env.example`):

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OPENROUTER_MODEL` | OpenRouter model identifier (e.g. `openai/gpt-4o-mini`) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_API_KEY` | Supabase anon/service key |
| `CUSTOM_SEARCH_API_KEY` | Google Custom Search API key |
| `CUSTOM_SEARCH_CX` | Custom Search Engine ID |
| `USE_FALLBACK_DATA` | Use Yahoo Finance when broker unavailable |
| `REDIS_URL` | Redis connection URL |

## Features

- **TRUE MCP Protocol** - Official MCP stdio client integration
- **Multi-Agent Orchestration** - Automatic intent classification and agent routing
- **LSTM Price Predictions** - Pre-trained models for 11 stock tickers
- **Real-time Events** - Redis pub/sub + WebSocket streaming
- **Fallback Data** - Yahoo Finance when broker API unavailable
- **Production Monitoring** - Structured JSON logging, metrics, health checks
- **Rate Limiting** - 100 requests/minute per IP
- **Input Validation** - Pydantic models with sanitization

## License

MIT
