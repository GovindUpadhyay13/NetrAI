"""SafetyChain — Data Models

All data models as Python dataclasses. Matches Section 2 of the design document exactly.
Inter-stage communication contracts:
  Stage 1 → Stage 2: AnomalyCandidate
  Stage 2 → Stage 3: SceneDescription
  Stage 3 → Stage 4: ContextReport
  Stage 4 → Stage 5: Verdict
  Stage 5 → Dashboard: Alert
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import uuid


# ═══════════════════════════════════════════════════════════
# Stage 1 → Stage 2: Detection & AnomalyCandidate
# ═══════════════════════════════════════════════════════════

@dataclass
class Detection:
    """A single object detection from YOLOv8."""
    class_name: str          # "person", "vehicle", "knife", etc.
    confidence: float        # 0.0 - 1.0
    bbox: tuple              # (x1, y1, x2, y2)


@dataclass
class AnomalyCandidate:
    """Output of Stage 1 — a frame that passed the Anomaly Gate."""
    id: str                          # UUID
    timestamp: datetime              # When the frame was captured
    frame: np.ndarray                # Raw frame (OpenCV BGR)
    frame_annotated: np.ndarray      # Frame with YOLO bboxes drawn
    detections: List[Detection]      # All relevant detections
    zone_id: str                     # Which zone this camera covers
    camera_id: str                   # Camera identifier
    motion_delta: float              # Frame-diff motion score
    trigger_reason: str              # "person_in_restricted_zone", "rapid_motion", etc.


# ═══════════════════════════════════════════════════════════
# Stage 2 → Stage 3: SceneDescription
# ═══════════════════════════════════════════════════════════

@dataclass
class Person:
    """A person identified in the scene by the VLM."""
    id: str                  # "P1", "P2", etc.
    description: str         # "Adult male, dark hoodie, face partially obscured"
    position: str            # "Near driver-side door of silver sedan"
    posture: str             # "Crouching, looking around repeatedly"
    movement: str            # "Intermittent — pauses, then moves quickly"


@dataclass
class ObjectOfInterest:
    """A notable object identified in the scene."""
    type: str                # "vehicle", "tool", "bag", etc.
    description: str         # "Slim metallic object in right hand"


@dataclass
class SceneDescription:
    """Output of Stage 2 — VLM-generated scene understanding."""
    candidate_id: str                # Links back to AnomalyCandidate
    scene_environment: str           # "Parking lot, nighttime, poorly lit"
    people: List[Person]             # People in the scene
    objects: List[ObjectOfInterest]  # Notable objects
    visible_text: List[str]          # Signs, license plates
    activity: str                    # "Possible vehicle break-in attempt"
    norm_violation: str              # "Person using tool on vehicle door lock"
    suspiciousness: str              # "NORMAL" | "UNUSUAL" | "CONCERNING" | "ALARMING"
    raw_json: dict                   # Full Gemma output for evidence trail


# ═══════════════════════════════════════════════════════════
# Stage 3 → Stage 4: ContextReport
# ═══════════════════════════════════════════════════════════

@dataclass
class ZoneContext:
    """Zone-specific context from the knowledge graph."""
    zone_id: str
    zone_name: str
    zone_type: str           # "parking", "school", "corridor"
    active_hours: str        # "06:00-23:00"
    currently_active: bool
    expected_occupancy: str  # "residents_with_badge"


@dataclass
class TemporalContext:
    """Time-based context."""
    current_time: datetime
    day_of_week: str
    is_holiday: bool
    holiday_name: Optional[str]
    is_within_active_hours: bool


@dataclass
class HistoricalContext:
    """Historical event data for this zone/camera."""
    similar_events_count: int          # In last 30 days
    false_positive_rate: float         # For this camera + similar detection
    known_fp_pattern: Optional[str]    # "Swaying tree branch" etc.
    last_event_in_zone: Optional[datetime]


@dataclass
class ProtocolContext:
    """Matching SOP and emergency contacts."""
    matching_sop: Optional[str]        # "SOP-014: Vehicle Theft Response"
    procedure_summary: Optional[str]
    contacts: Optional[Dict]           # {"patrol": "ext. 2200", "police": "911"}
    policy_notes: Optional[str]        # "Do NOT approach suspect"


@dataclass
class ContextReport:
    """Output of Stage 3 — aggregated context for verification."""
    candidate_id: str
    zone: ZoneContext
    temporal: TemporalContext
    historical: HistoricalContext
    protocol: ProtocolContext
    verdict: str                 # "SUPPORTS_ANOMALY" | "NEUTRAL" | "REFUTES_ANOMALY"
    confidence_modifier: float   # -0.3 to +0.3 (context boost/penalty)
    suppress: bool               # True = kill alert (known FP pattern)
    suppress_reason: Optional[str]


# ═══════════════════════════════════════════════════════════
# Stage 4 → Stage 5: Verdict
# ═══════════════════════════════════════════════════════════

@dataclass
class ReasoningStep:
    """A single step in the CoT reasoning chain."""
    step_number: int         # 1-5
    title: str               # "Evidence Consistency"
    content: str             # Full reasoning text
    passed: bool             # True = supports anomaly


@dataclass
class Verdict:
    """Output of Stage 4 — the final verification result."""
    candidate_id: str
    chain_id: str                        # UUID for forensic tracing
    classification: str                  # "FALSE_POSITIVE" | "SUSPICIOUS" | "CONFIRMED_ANOMALY"
    confidence: float                    # 0.0 - 1.0
    severity: str                        # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    urgency: str                         # "MONITOR" | "INVESTIGATE" | "INTERVENE" | "EMERGENCY"
    reasoning_strategy: str              # "ZeroThink" | "LessThink" | "FullThink" | "MoreThink"
    reasoning_chain: List[ReasoningStep] # Empty for ZeroThink
    alternative_hypotheses: List[str]
    recommended_action: str
    consequences_if_ignored: str
    reasoning_latency_ms: int


# ═══════════════════════════════════════════════════════════
# Stage 5 → Dashboard: Alert
# ═══════════════════════════════════════════════════════════

@dataclass
class Alert:
    """Output of Stage 5 — the operator-facing alert."""
    alert_id: str                        # UUID
    chain_id: str                        # Links to Verdict
    timestamp: datetime
    severity: str                        # "LOG" | "NOTIFY" | "ALERT" | "EMERGENCY"
    title: str                           # "Possible Vehicle Break-in"
    zone_name: str
    confidence: float
    frame_b64: str                       # Base64-encoded annotated JPEG
    verdict: Verdict
    context_summary: str                 # One-line context
    sop: Optional[str]                   # Retrieved SOP text
    contacts: Optional[Dict]
    status: str = "active"               # "active" | "acknowledged" | "dismissed" | "resolved"
    operator_feedback: Optional[str] = None  # "true_positive" | "false_positive"
