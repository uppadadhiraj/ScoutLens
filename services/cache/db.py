"""SQLite plumbing shared by the search cache and (later) the investigation repository.

A new short-lived connection is opened per operation: Streamlit reruns scripts on different
threads, and sqlite3 connections must not be shared across threads. The default rollback
journal is used instead of WAL because the project may live in a cloud-synced folder
(OneDrive/Dropbox), where WAL's side files cause lock contention.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS search_cache (
        cache_key     TEXT PRIMARY KEY,
        engine        TEXT NOT NULL,
        query         TEXT NOT NULL,
        response_json TEXT NOT NULL,
        fetched_at    REAL NOT NULL
    )
    """,
)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success / roll back on error, always close."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        for statement in SCHEMA:
            conn.execute(statement)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
