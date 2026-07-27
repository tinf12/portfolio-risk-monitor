"""SQLite connection handling."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.db.schema import create_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "risk.db"


def resolve_db_path() -> Path:
    """Return the database path, honouring the RISK_DB_PATH override."""
    override = os.environ.get("RISK_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_DB_PATH


@contextmanager
def get_connection(
    db_path: Path | str | None = None,
    *,
    ensure_schema: bool = True,
) -> Iterator[sqlite3.Connection]:
    """Yield a connection with foreign keys on and rows as sqlite3.Row.

    Commits on clean exit, rolls back if the body raises. The daily job depends
    on this: a partially written day must not be committed, because the next
    run would treat it as complete and skip it.
    """
    path = Path(db_path) if db_path is not None else resolve_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        if ensure_schema:
            create_schema(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
