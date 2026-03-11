import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from vectordbsupabase import SupabaseVectorDB

logger = logging.getLogger(__name__)

SHARED_AGENT_NAME = "shared_context"

vector_db = SupabaseVectorDB()


def store_user_context(
    user_id: str,
    agent: str,
    content: str,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Store user context in shared Supabase vector memory (no agent separation)."""
    merged_metadata: Dict[str, Any] = {"source_agent": agent}
    if metadata:
        merged_metadata.update(metadata)
    try:
        vector = embedding or vector_db.embed_text(content)
        return vector_db.upsert_record(
            user_id=user_id,
            agent=SHARED_AGENT_NAME,
            content=content,
            embedding=vector,
            metadata=merged_metadata,
        )
    except Exception as exc:
        logger.warning("Error storing context for user %s: %s", user_id, exc)
        return ""


def retrieve_user_context(
    user_id: str,
    agent: str,
    query_embedding: List[float],
    top_k: int = 5,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    """Retrieve similar context entries for a user across all agents using vector search."""
    try:
        return vector_db.similarity_search(
            user_id=user_id,
            agent=None,
            query_embedding=query_embedding,
            match_count=top_k,
            match_threshold=min_score,
        )
    except Exception as exc:
        logger.warning("Error retrieving context for user %s: %s", user_id, exc)
        return []


def supabase_vector_schema_sql() -> str:
    """Return SQL required to provision the Supabase vector table and RPC."""
    try:
        return vector_db.schema_sql()
    except Exception as exc:
        logger.warning("Error generating schema SQL: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Tool-wrapped helpers (agent-specific) for recalling/storing short notes
# ---------------------------------------------------------------------------


def _render_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No relevant memory found."
    lines = []
    for row in rows:
        meta = row.get("metadata", {})
        source = meta.get("source_agent", "unknown")
        user_msg = meta.get("user_message", "")
        agent_msg = meta.get("agent_response", "")
        if user_msg or agent_msg:
            lines.append(f"- [{source}] Q: {user_msg} → A: {agent_msg}")
    return "\n".join(lines) if lines else "No relevant memory found."


@tool
def search_user_memory(query: str, user_id: str) -> str:
    """Search recent Supabase memory for this user (shared across agents).

    Returns the most relevant past interactions matching the query.
    """
    try:
        emb = vector_db.embed_text(query)
        rows = retrieve_user_context(
            user_id=str(user_id),
            agent=SHARED_AGENT_NAME,
            query_embedding=emb,
            top_k=5,
            min_score=0.25,
        )
    except Exception as exc:
        logger.warning("Memory search failed for user %s: %s", user_id, exc)
        return f"Memory search unavailable: {exc}"
    return _render_rows(rows)


@tool
def store_user_note(note: str, user_id: str) -> str:
    """Store a short note to shared Supabase memory (visible to all agents)."""
    if not note or not note.strip():
        return "Cannot store an empty note."
    try:
        store_user_context(
            user_id=str(user_id),
            agent=SHARED_AGENT_NAME,
            content=note.strip(),
            metadata={"user_message": note.strip(), "agent_response": "stored_note"},
        )
        return "Saved to memory."
    except Exception as exc:
        logger.warning("Could not save memory for user %s: %s", user_id, exc)
        return f"Could not save memory: {exc}"

