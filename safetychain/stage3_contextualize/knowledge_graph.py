"""SafetyChain — Stage 3: CONTEXTUALIZE — Knowledge Graph

SQLite-backed knowledge graph with tables matching the ER diagram
from the architecture document (Section 5).

Tables: sites, zones, norms, history, cameras, sops
Initialization from zones.json and sops.json.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from ..utils.logger import get_logger

logger = get_logger("stage3.knowledge_graph")


class KnowledgeGraph:
    """SQLite-backed knowledge graph for zone norms, history, and SOPs.
    
    Provides the contextual memory layer that enables the system to
    'learn without retraining' through operator feedback.
    """

    def __init__(self, db_path: str, zones_path: str, sops_path: str):
        """Initialize the knowledge graph.
        
        Args:
            db_path: Path to SQLite database file
            zones_path: Path to zones.json configuration
            sops_path: Path to sops.json configuration
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._load_initial_data(zones_path, sops_path)

        logger.info(f"KnowledgeGraph initialized at {db_path}")

    def _create_tables(self):
        """Create the knowledge graph schema matching the architecture ER diagram."""
        cursor = self.conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sites (
                site_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                address TEXT,
                timezone TEXT DEFAULT 'UTC'
            );

            CREATE TABLE IF NOT EXISTS zones (
                zone_id TEXT PRIMARY KEY,
                site_id TEXT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                boundary_coords TEXT,
                FOREIGN KEY (site_id) REFERENCES sites(site_id)
            );

            CREATE TABLE IF NOT EXISTS norms (
                norm_id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL,
                norm_type TEXT NOT NULL,
                rule_description TEXT,
                active_hours TEXT,
                parameters TEXT,
                FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
            );

            CREATE TABLE IF NOT EXISTS history (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone_id TEXT NOT NULL,
                camera_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                description TEXT,
                chain_id TEXT,
                was_false_positive BOOLEAN DEFAULT 0,
                detection_class TEXT,
                operator_note TEXT,
                FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
            );

            CREATE TABLE IF NOT EXISTS cameras (
                camera_id TEXT PRIMARY KEY,
                zone_id TEXT NOT NULL,
                name TEXT,
                stream_url TEXT,
                field_of_view TEXT,
                adjacent_cameras TEXT,
                FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
            );

            CREATE TABLE IF NOT EXISTS sops (
                sop_id TEXT PRIMARY KEY,
                zone_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                title TEXT NOT NULL,
                procedure TEXT,
                contacts TEXT,
                FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
            );
        """)

        self.conn.commit()

    def _load_initial_data(self, zones_path: str, sops_path: str):
        """Load zone and SOP data from JSON configuration files."""
        cursor = self.conn.cursor()

        # Load zones
        if os.path.exists(zones_path):
            with open(zones_path, "r") as f:
                zones_data = json.load(f)

            # Insert site
            site = zones_data.get("site", {})
            cursor.execute(
                "INSERT OR REPLACE INTO sites (site_id, name, address, timezone) VALUES (?, ?, ?, ?)",
                (site.get("site_id", "site-001"), site.get("name", "Default"),
                 site.get("address", ""), site.get("timezone", "UTC"))
            )

            # Insert zones, norms, cameras
            for zone in zones_data.get("zones", []):
                cursor.execute(
                    "INSERT OR REPLACE INTO zones (zone_id, site_id, name, type, boundary_coords) VALUES (?, ?, ?, ?, ?)",
                    (zone["zone_id"], site.get("site_id", "site-001"),
                     zone["name"], zone["type"],
                     json.dumps(zone.get("boundary_coords", [])))
                )

                for norm in zone.get("norms", []):
                    cursor.execute(
                        "INSERT INTO norms (zone_id, norm_type, rule_description, active_hours, parameters) VALUES (?, ?, ?, ?, ?)",
                        (zone["zone_id"], norm["norm_type"],
                         norm.get("rule_description", ""),
                         norm.get("active_hours", ""),
                         json.dumps(norm.get("parameters", {})))
                    )

                for cam in zone.get("cameras", []):
                    cursor.execute(
                        "INSERT OR REPLACE INTO cameras (camera_id, zone_id, name, stream_url, field_of_view, adjacent_cameras) VALUES (?, ?, ?, ?, ?, ?)",
                        (cam["camera_id"], zone["zone_id"], cam.get("name", ""),
                         cam.get("stream_url", ""),
                         json.dumps(cam.get("field_of_view", {})),
                         json.dumps(cam.get("adjacent_cameras", [])))
                    )

        # Load SOPs
        if os.path.exists(sops_path):
            with open(sops_path, "r") as f:
                sops_data = json.load(f)

            for sop in sops_data.get("sops", []):
                cursor.execute(
                    "INSERT OR REPLACE INTO sops (sop_id, zone_id, alert_type, title, procedure, contacts) VALUES (?, ?, ?, ?, ?, ?)",
                    (sop["sop_id"], sop["zone_id"], sop["alert_type"],
                     sop["title"], sop.get("procedure", ""),
                     json.dumps(sop.get("contacts", {})))
                )

        self.conn.commit()

    def get_zone(self, zone_id: str) -> Optional[Dict]:
        """Get zone details by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM zones WHERE zone_id = ?", (zone_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_zone_norms(self, zone_id: str) -> List[Dict]:
        """Get all norms for a zone."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM norms WHERE zone_id = ?", (zone_id,))
        return [dict(row) for row in cursor.fetchall()]

    def get_history(self, zone_id: str, days: int = 30,
                    camera_id: str = None) -> List[Dict]:
        """Get recent event history for a zone.
        
        Args:
            zone_id: Zone to query
            days: Number of days to look back
            camera_id: Optional camera filter
            
        Returns:
            List of historical events
        """
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        if camera_id:
            cursor.execute(
                "SELECT * FROM history WHERE zone_id = ? AND camera_id = ? AND timestamp > ? ORDER BY timestamp DESC",
                (zone_id, camera_id, cutoff)
            )
        else:
            cursor.execute(
                "SELECT * FROM history WHERE zone_id = ? AND timestamp > ? ORDER BY timestamp DESC",
                (zone_id, cutoff)
            )

        return [dict(row) for row in cursor.fetchall()]

    def get_false_positive_rate(self, zone_id: str, camera_id: str = None,
                                days: int = 30) -> float:
        """Calculate the false positive rate for a zone/camera combo.
        
        Args:
            zone_id: Zone to query
            camera_id: Optional camera filter
            days: Window in days
            
        Returns:
            False positive rate (0.0 to 1.0)
        """
        history = self.get_history(zone_id, days, camera_id)
        if not history:
            return 0.0

        total = len(history)
        fps = sum(1 for h in history if h.get("was_false_positive"))
        return fps / total

    def get_known_fp_patterns(self, zone_id: str, camera_id: str = None) -> Optional[str]:
        """Get known false positive patterns for a zone/camera.
        
        Returns the most common FP description if any patterns exist.
        """
        cursor = self.conn.cursor()

        if camera_id:
            cursor.execute(
                "SELECT description, COUNT(*) as cnt FROM history WHERE zone_id = ? AND camera_id = ? AND was_false_positive = 1 GROUP BY description ORDER BY cnt DESC LIMIT 1",
                (zone_id, camera_id)
            )
        else:
            cursor.execute(
                "SELECT description, COUNT(*) as cnt FROM history WHERE zone_id = ? AND was_false_positive = 1 GROUP BY description ORDER BY cnt DESC LIMIT 1",
                (zone_id,)
            )

        row = cursor.fetchone()
        return dict(row)["description"] if row else None

    def get_sop(self, zone_id: str, alert_type: str = None) -> Optional[Dict]:
        """Get the matching SOP for a zone and alert type.
        
        Args:
            zone_id: Zone to query
            alert_type: Type of alert (e.g., "vehicle_breakin", "school_intrusion")
            
        Returns:
            SOP dict with title, procedure, contacts, or None
        """
        cursor = self.conn.cursor()

        if alert_type:
            cursor.execute(
                "SELECT * FROM sops WHERE zone_id = ? AND alert_type = ?",
                (zone_id, alert_type)
            )
        else:
            cursor.execute(
                "SELECT * FROM sops WHERE zone_id = ? LIMIT 1",
                (zone_id,)
            )

        row = cursor.fetchone()
        if row:
            sop = dict(row)
            sop["contacts"] = json.loads(sop["contacts"]) if sop.get("contacts") else {}
            return sop
        return None

    def record_event(self, zone_id: str, camera_id: str, event_type: str,
                     description: str, chain_id: str = None,
                     was_false_positive: bool = False,
                     detection_class: str = None):
        """Record an event in the history table.
        
        Args:
            zone_id: Zone where the event occurred
            camera_id: Camera that captured the event
            event_type: "alert", "false_positive", "true_positive"
            description: Human-readable description
            chain_id: UUID linking to the verification chain
            was_false_positive: Whether this was marked as FP
            detection_class: YOLO class that triggered it
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO history (zone_id, camera_id, event_type, description, chain_id, was_false_positive, detection_class) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (zone_id, camera_id, event_type, description, chain_id,
             was_false_positive, detection_class)
        )
        self.conn.commit()

    def update_from_feedback(self, chain_id: str, was_false_positive: bool,
                             operator_note: str = None):
        """Update event history based on operator feedback.
        
        This is the 'Learn Without Retraining' mechanism — feedback
        enriches the knowledge graph for future context checks.
        """
        cursor = self.conn.cursor()
        event_type = "false_positive" if was_false_positive else "true_positive"

        cursor.execute(
            "UPDATE history SET was_false_positive = ?, event_type = ?, operator_note = ? WHERE chain_id = ?",
            (was_false_positive, event_type, operator_note, chain_id)
        )
        self.conn.commit()

        logger.info(
            f"Knowledge graph updated: chain_id={chain_id}, "
            f"was_fp={was_false_positive}, note={operator_note}"
        )

    def close(self):
        """Close the database connection."""
        self.conn.close()
