from __future__ import annotations

import os
import socket
import logging
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ISP DNS-hijack bypass: Excitel (and similar ISPs) resolve *.supabase.co to
# a blocking proxy.  We resolve via Google DoH and patch socket.getaddrinfo
# so every library (httpx, supabase-py, etc.) connects to the real IP.
# ---------------------------------------------------------------------------
_SUPABASE_HOST = os.getenv("SUPABASE_URL", "").replace("https://", "").replace("http://", "").strip("/")
_DNS_OVERRIDE: dict[str, str] = {}  # hostname -> real IP

def _resolve_via_doh(hostname: str) -> str | None:
    """Resolve *hostname* using Google DNS-over-HTTPS (bypasses ISP DNS)."""
    try:
        import urllib.request, json
        url = f"https://8.8.8.8/resolve?name={hostname}&type=A"
        req = urllib.request.Request(url, headers={"Accept": "application/dns-json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            for ans in data.get("Answer", []):
                if ans.get("type") == 1:  # A record
                    return ans["data"]
    except Exception as exc:
        logger.debug("DoH resolution failed for %s: %s", hostname, exc)
    return None

def _bootstrap_dns_override():
    """Resolve the Supabase host once at import time and patch getaddrinfo."""
    if not _SUPABASE_HOST:
        return
    real_ip = _resolve_via_doh(_SUPABASE_HOST)
    if not real_ip:
        logger.warning("Could not resolve %s via DoH; Supabase may be unreachable", _SUPABASE_HOST)
        return
    _DNS_OVERRIDE[_SUPABASE_HOST] = real_ip
    logger.info("DNS override: %s -> %s (bypassing ISP hijack)", _SUPABASE_HOST, real_ip)

    _original_getaddrinfo = socket.getaddrinfo

    def _patched_getaddrinfo(host, port, *args, **kwargs):
        override = _DNS_OVERRIDE.get(host)
        if override:
            return _original_getaddrinfo(override, port, *args, **kwargs)
        return _original_getaddrinfo(host, port, *args, **kwargs)

    socket.getaddrinfo = _patched_getaddrinfo

_bootstrap_dns_override()

DEFAULT_TABLE = os.getenv("SUPABASE_VECTOR_TABLE", "agent_memory")
DEFAULT_RPC_FN = os.getenv("SUPABASE_VECTOR_MATCH_FN", "match_agent_context")
DEFAULT_EMBED_DIM = int(os.getenv("SUPABASE_VECTOR_DIM", "768"))
DEFAULT_EMBED_MODEL = os.getenv("SUPABASE_EMBEDDING_MODEL", "gemini-embedding-001")


def build_schema_sql(
    table_name: str = DEFAULT_TABLE,
    embedding_dim: int = DEFAULT_EMBED_DIM,
    rpc_fn: str = DEFAULT_RPC_FN,
) -> str:
    return f"""
create extension if not exists vector;
create extension if not exists pgcrypto;

create table if not exists {table_name} (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    agent text not null,
    content text not null,
    metadata jsonb default '{{}}'::jsonb,
    embedding vector({embedding_dim}) not null,
    created_at timestamptz not null default now()
);

create index if not exists {table_name}_user_agent_idx on {table_name} (user_id, agent);
create index if not exists {table_name}_ivfflat_embedding_idx on {table_name} using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create or replace function {rpc_fn}(
    query_embedding vector({embedding_dim}),
    match_count int default 5,
    match_threshold float default 0.0,
    filter_user_id text default null,
    filter_agent text default null
) returns table (
    id uuid,
    user_id text,
    agent text,
    content text,
    metadata jsonb,
    created_at timestamptz,
    similarity float
) language plpgsql as $$
begin
    return query
    select
        m.id,
        m.user_id,
        m.agent,
        m.content,
        m.metadata,
        m.created_at,
        1 - (m.embedding <=> query_embedding) as similarity
    from {table_name} as m
    where (filter_user_id is null or m.user_id = filter_user_id)
      and (filter_agent is null or m.agent = filter_agent)
      and (match_threshold <= 0 or 1 - (m.embedding <=> query_embedding) >= match_threshold)
    order by m.embedding <=> query_embedding
    limit match_count;
end;
$$;
"""


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_API_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_API_KEY must be set in the environment.")
    # Use a custom httpx client that skips SSL verification to work around
    # ISP-level TLS interception (Excitel MITM proxy replaces Supabase certs).
    options = SyncClientOptions(httpx_client=httpx.Client(verify=False))
    return create_client(url, key, options=options)


class SupabaseVectorDB:
    _genai_module = None  # Class-level cache: import + configure genai once

    def __init__(
        self,
        client: Optional[Client] = None,
        table_name: str = DEFAULT_TABLE,
        rpc_fn: str = DEFAULT_RPC_FN,
        embedding_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        self.client = client or get_supabase_client()
        self.table_name = table_name
        self.rpc_fn = rpc_fn
        self.embedding_dim = embedding_dim

    def schema_sql(self) -> str:
        return build_schema_sql(self.table_name, self.embedding_dim, self.rpc_fn)

    @classmethod
    def _init_genai(cls):
        """Import and configure google.genai Client exactly once."""
        if cls._genai_module is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY is required for automatic embeddings.")
            from google import genai as _genai
            cls._genai_module = _genai.Client(api_key=api_key)
        return cls._genai_module

    def embed_text(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings using google.genai SDK (gemini-embedding-001)."""
        client = self._init_genai()
        from google.genai import types as _gentypes

        target_model = model or DEFAULT_EMBED_MODEL
        try:
            response = client.models.embed_content(
                model=target_model,
                contents=text,
                config=_gentypes.EmbedContentConfig(
                    output_dimensionality=self.embedding_dim,
                ),
            )
            embedding = list(response.embeddings[0].values)
            return self._validate_embedding(embedding)
        except Exception as exc:
            logger.error("Embedding failed for model %s: %s", target_model, exc)
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc

    def upsert_record(
        self,
        user_id: str,
        agent: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        vector = self._validate_embedding(embedding)
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "agent": agent,
            "content": content,
            "metadata": metadata or {},
            "embedding": vector,
        }
        try:
            result = self.client.table(self.table_name).upsert(payload, returning="representation").execute()
        except Exception as exc:
            raise RuntimeError(f"Failed to upsert context: {exc}") from exc
        rows = getattr(result, "data", None)
        if rows and isinstance(rows, list) and rows[0].get("id"):
            return str(rows[0]["id"])
        return ""

    def similarity_search(
        self,
        user_id: Optional[str],
        agent: Optional[str],
        query_embedding: List[float],
        match_count: int = 5,
        match_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        vector = self._validate_embedding(query_embedding)
        params: Dict[str, Any] = {
            "query_embedding": vector,
            "match_count": match_count,
            "match_threshold": match_threshold,
            "filter_user_id": user_id,
            "filter_agent": agent,
        }
        try:
            result = self.client.rpc(self.rpc_fn, params).execute()
        except Exception as exc:
            raise RuntimeError(f"Failed to run similarity search: {exc}") from exc
        rows = getattr(result, "data", None)
        return rows if isinstance(rows, list) else []

    def latest_records(
        self,
        user_id: Optional[str],
        agent: Optional[str],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        try:
            query = self.client.table(self.table_name).select("*").order("created_at", desc=True).limit(limit)
            if user_id:
                query = query.eq("user_id", user_id)
            if agent:
                query = query.eq("agent", agent)
            result = query.execute()
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch recent context: {exc}") from exc
        rows = getattr(result, "data", None)
        return rows if isinstance(rows, list) else []

    def _validate_embedding(self, embedding: List[float]) -> List[float]:
        if not isinstance(embedding, list):
            raise ValueError("Embedding must be a list of floats.")
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}.")
        return embedding
