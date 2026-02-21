from __future__ import annotations

import sqlite3
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

DB_PATH = Path("appraisal.sqlite")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            location TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS appraisals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            version_name TEXT NOT NULL,
            assumptions_json TEXT NOT NULL,
            outputs_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(project_id) REFERENCES projects(id)
        )
        """)
        conn.commit()


def list_projects() -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM projects ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def create_project(name: str, location: str = "") -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO projects (name, location) VALUES (?, ?)",
            (name, location),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_appraisals(project_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM appraisals WHERE project_id=? ORDER BY id DESC",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def save_appraisal(project_id: int, version_name: str, assumptions: Dict[str, Any], outputs: Dict[str, Any]) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO appraisals (project_id, version_name, assumptions_json, outputs_json) VALUES (?, ?, ?, ?)",
            (project_id, version_name, json.dumps(assumptions), json.dumps(outputs)),
        )
        conn.commit()
        return int(cur.lastrowid)


def load_appraisal(appraisal_id: int) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM appraisals WHERE id=?", (appraisal_id,)).fetchone()
        if not row:
            return None
        assumptions = json.loads(row["assumptions_json"])
        outputs = json.loads(row["outputs_json"])
        return assumptions, outputs
