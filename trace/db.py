"""
trace/db.py
SQLite trace database storage for per-incident surveillance event lifecycle.

Assumptions:
- SQLite is used for prototype traceability (swap-in path to Postgres later).
- Database file defaults to 'trace.db' in project root or outputs.
- Thread-safe connection handling.
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional
from bus.schemas import SurveillanceEvent

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "trace.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Ensure table exists
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS incident_traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            stage TEXT NOT NULL,
            anomaly_score REAL DEFAULT 0.0,
            anomaly_type TEXT,
            distress_gesture INTEGER DEFAULT 0,
            vlm_report TEXT,
            severity TEXT,
            payload_ref TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes SQLite trace schema."""
    conn = get_connection(db_path)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_incident_id ON incident_traces(incident_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stage ON incident_traces(stage)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON incident_traces(timestamp)")
    conn.commit()
    conn.close()


def insert_event(event: SurveillanceEvent, db_path: str = DEFAULT_DB_PATH) -> int:
    """Inserts a surveillance event into trace DB."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO incident_traces (
            incident_id, camera_id, timestamp, stage,
            anomaly_score, anomaly_type, distress_gesture,
            vlm_report, severity, payload_ref, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.incident_id,
            event.camera_id,
            event.timestamp,
            str(event.stage),
            event.anomaly_score,
            event.anomaly_type,
            1 if event.distress_gesture else 0,
            event.vlm_report,
            str(event.severity) if event.severity else None,
            event.payload_ref,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    inserted_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_recent_events(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Retrieves most recent trace events."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM incident_traces ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_incident_trace(incident_id: str, db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Retrieves full chronological stage history for a specific incident."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM incident_traces WHERE incident_id = ? ORDER BY id ASC",
        (incident_id,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_all_incidents(db_path: str = DEFAULT_DB_PATH) -> List[Dict]:
    """Aggregates all unique incidents with their latest state and stages."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            incident_id,
            camera_id,
            MIN(timestamp) as first_detected,
            MAX(timestamp) as last_updated,
            MAX(anomaly_score) as peak_anomaly_score,
            MAX(distress_gesture) as has_distress_gesture,
            GROUP_CONCAT(DISTINCT stage) as stages_reached,
            MAX(severity) as severity,
            MAX(vlm_report) as latest_vlm_report
        FROM incident_traces
        GROUP BY incident_id
        ORDER BY id DESC
        """
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
