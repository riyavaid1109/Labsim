"""
observability.py — structured tracing for every LLM call.

Each call through llm.client is logged to a local SQLite DB with:
prompt, response, provider, model, latency, token counts, caller tag,
and whether structured output was requested. No external service needed —
query traces with plain SQL or the summary helper below.

    from llm.observability import trace_summary
    print(trace_summary())
"""

import json
import time
import sqlite3
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger("lab_sim")

DEFAULT_DB = "./llm_traces.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    caller TEXT,
    structured INTEGER DEFAULT 0,
    prompt TEXT,
    response TEXT,
    latency_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ts ON llm_calls (ts);
CREATE INDEX IF NOT EXISTS idx_llm_calls_caller ON llm_calls (caller);
"""

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path = DEFAULT_DB
_enabled = True


def configure(db_path: str = DEFAULT_DB, enabled: bool = True) -> None:
    """Point tracing at a different DB file, or disable it entirely."""
    global _db_path, _enabled, _conn
    with _lock:
        _db_path = db_path
        _enabled = enabled
        if _conn is not None:
            _conn.close()
            _conn = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(_db_path, check_same_thread=False)
        _conn.executescript(_SCHEMA)
    return _conn


def record(
    provider: str,
    model: str,
    prompt: str,
    response: str,
    latency_ms: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    caller: str | None = None,
    structured: bool = False,
    error: str | None = None,
) -> None:
    if not _enabled:
        return
    try:
        with _lock:
            conn = _get_conn()
            conn.execute(
                "INSERT INTO llm_calls (ts, provider, model, caller, structured, prompt, "
                "response, latency_ms, prompt_tokens, completion_tokens, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), provider, model, caller, int(structured), prompt,
                 response, latency_ms, prompt_tokens, completion_tokens, error),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Trace logging failed: {e}")


@contextmanager
def traced_call(provider: str, model: str, prompt: str,
                caller: str | None = None, structured: bool = False):
    """
    Context manager used by the client layer. Usage:

        with traced_call("ollama", model, prompt, caller) as trace:
            resp = ...
            trace["response"] = resp_text
            trace["prompt_tokens"] = ...
    """
    trace = {"response": "", "prompt_tokens": None,
             "completion_tokens": None, "error": None}
    start = time.perf_counter()
    try:
        yield trace
    except Exception as e:
        trace["error"] = str(e)
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        record(
            provider=provider, model=model, prompt=prompt,
            response=trace["response"], latency_ms=latency_ms,
            prompt_tokens=trace["prompt_tokens"],
            completion_tokens=trace["completion_tokens"],
            caller=caller, structured=structured, error=trace["error"],
        )


def trace_summary(db_path: str | None = None) -> str:
    """Aggregate stats: calls, latency, tokens — per caller and per model."""
    conn = sqlite3.connect(db_path or _db_path)
    try:
        rows = conn.execute(
            "SELECT caller, model, COUNT(*), AVG(latency_ms), "
            "SUM(COALESCE(prompt_tokens,0)), SUM(COALESCE(completion_tokens,0)), "
            "SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM llm_calls GROUP BY caller, model ORDER BY COUNT(*) DESC"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No LLM calls traced yet."
    lines = [f"{'caller':<28}{'model':<18}{'calls':>6}{'avg ms':>9}"
             f"{'in tok':>9}{'out tok':>9}{'errs':>6}"]
    for caller, model, n, avg_ms, tin, tout, errs in rows:
        lines.append(f"{(caller or '-'):<28}{model:<18}{n:>6}{avg_ms:>9.0f}"
                     f"{tin:>9}{tout:>9}{errs:>6}")
    return "\n".join(lines)


def export_traces(out_path: str = "./llm_traces.jsonl", db_path: str | None = None) -> int:
    """Dump all traces to JSONL for offline analysis. Returns row count."""
    conn = sqlite3.connect(db_path or _db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM llm_calls ORDER BY ts").fetchall()
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + "\n")
        return len(rows)
    finally:
        conn.close()
