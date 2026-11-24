"""
SQLite database layer.
Uses WAL mode and check_same_thread=False with a module-level lock
so the connection is safe across threads (FastAPI/uvicorn workers).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional

from app.config import get_settings

_lock = threading.Lock()
_db_path: Optional[str] = None


# ─────────────────────────── connection ───────────────────────────────────────

def _get_db_path() -> str:
    global _db_path
    if _db_path is None:
        _db_path = get_settings().db_path
    return _db_path


def _connect() -> sqlite3.Connection:
    path = _get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    with _lock:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ─────────────────────────── schema ───────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    system_prompt TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


# ─────────────────────────── helpers ──────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────── sessions ─────────────────────────────────────────

def get_or_create_session(session_id: str, system_prompt: str) -> str:
    """Return (possibly persisted) system prompt for a session."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row:
            return row["system_prompt"]
        conn.execute(
            "INSERT INTO sessions (session_id, system_prompt, created_at) VALUES (?,?,?)",
            (session_id, system_prompt, _now_iso()),
        )
        return system_prompt


def session_exists(session_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None


def update_session_system_prompt(session_id: str, system_prompt: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET system_prompt = ? WHERE session_id = ?",
            (system_prompt, session_id),
        )


# ─────────────────────────── messages ─────────────────────────────────────────

def get_session_history(session_id: str, limit: int = 40) -> List[Dict[str, Any]]:
    """Return last `limit` messages (oldest first) for session."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, metadata
            FROM (
                SELECT id, role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) sub
            ORDER BY id ASC
            """,
            (session_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        msg: Dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["metadata"]:
            msg["metadata"] = json.loads(r["metadata"])
        result.append(msg)
    return result


def store_exchange(
    session_id: str,
    user_content: str,
    assistant_content: str,
    metadata: Dict[str, Any],
) -> None:
    """Atomically store user + assistant messages after a successful reply."""
    meta_json = json.dumps(metadata)
    now = _now_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?,?,?,NULL,?)",
            (session_id, "user", user_content, now),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
            (session_id, "assistant", assistant_content, meta_json, now),
        )


def get_all_messages(session_id: str) -> List[Dict[str, Any]]:
    """Return all messages for session, oldest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        msg: Dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["metadata"]:
            msg["metadata"] = json.loads(r["metadata"])
        result.append(msg)
    return result
