"""SafetyChain — Configuration Module

All tunable parameters in a single config class.
Matches Section 8 of the design document.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SafetyChainConfig:
    """Central configuration for the entire SafetyChain pipeline."""

    # ── Stage 1: PERCEIVE ──
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35
    YOLO_CLASSES_OF_INTEREST: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "knife",
        "scissors", "backpack", "fire"
    ])
    MOTION_DELTA_THRESHOLD: float = 0.15
    FRAME_SKIP: int = 2  # Process every Nth frame

    # ── Stage 2: DESCRIBE ──
    GEMMA_MODEL: str = "gemma-4-26b-a4b-it"
    GEMMA_MAX_TOKENS: int = 512
    GEMMA_TEMPERATURE: float = 0.1  # Low = deterministic

    # ── Stage 3: CONTEXTUALIZE ──
    KNOWLEDGE_GRAPH_DB: str = "data/knowledge_graph.db"
    ZONES_CONFIG: str = "data/zones.json"
    SOPS_CONFIG: str = "data/sops.json"
    FP_HISTORY_WINDOW_DAYS: int = 30

    # ── Stage 4: VERIFY ──
    ZEROTHINK_CLASSES: List[str] = field(default_factory=lambda: [
        "knife", "fire"
    ])
    ZEROTHINK_ZONE_TYPES: List[str] = field(default_factory=lambda: [
        "school"
    ])
    ZEROTHINK_CONFIDENCE: float = 0.95
    FULLTHINK_TIMEOUT_MS: int = 500

    # ── Stage 5: ACT ──
    ESCALATION_UNACK_NOTIFY_MIN: int = 5
    ESCALATION_UNACK_ALERT_MIN: int = 3
    EVIDENCE_RETENTION_DAYS_DISMISSED: int = 7
    EVIDENCE_RETENTION_DAYS_CONFIRMED: int = 90

    # ── Dashboard ──
    DASHBOARD_PORT: int = 8000
    DASHBOARD_HOST: str = "0.0.0.0"
    MAX_ALERTS_DISPLAYED: int = 50

    # ── Demo ──
    DEMO_VIDEO_DIR: str = "demo/videos/"
    DEMO_MODE: bool = False  # True = use pre-recorded videos

    # ── API Key ──
    GOOGLE_API_KEY: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))

    def get_db_path(self, base_dir: str) -> str:
        """Get absolute path to the knowledge graph database."""
        return os.path.join(base_dir, self.KNOWLEDGE_GRAPH_DB)

    def get_zones_path(self, base_dir: str) -> str:
        """Get absolute path to zones config."""
        return os.path.join(base_dir, self.ZONES_CONFIG)

    def get_sops_path(self, base_dir: str) -> str:
        """Get absolute path to SOPs config."""
        return os.path.join(base_dir, self.SOPS_CONFIG)

    def get_demo_video_dir(self, base_dir: str) -> str:
        """Get absolute path to demo videos directory."""
        return os.path.join(base_dir, self.DEMO_VIDEO_DIR)
