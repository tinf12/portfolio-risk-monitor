"""Shared fixtures."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.db.schema import create_schema


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """An empty database with the real schema applied.

    File-backed rather than in-memory so tests exercise the same sqlite
    behaviour as production, including AUTOINCREMENT.
    """
    connection = sqlite3.connect(tmp_path / "test.db")
    connection.row_factory = sqlite3.Row
    create_schema(connection)
    yield connection
    connection.close()
