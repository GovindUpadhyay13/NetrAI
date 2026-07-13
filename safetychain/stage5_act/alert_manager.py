"""SafetyChain — Stage 5: ACT — Alert Manager

Creates alerts from verdicts, applies the escalation matrix, and
manages alert lifecycle (active → acknowledged → dismissed → resolved).

Architecture ref: Section 7 (Stage 5 — ACT: Alert Escalation State Machine)
Design ref: Section 5 (Escalation Logic Design — Decision Matrix)

Escalation Matrix:
                    | <40%  | 40-70% | 70-90% | >90%  |
  LOW               | LOG   | NOTIFY | NOTIFY | ALERT |
  MEDIUM            | LOG   | NOTIFY | ALERT  | ALERT |
  HIGH              | NOTIFY| ALERT  | ALERT  | EMERG |
  CRITICAL          | ALERT | EMERG  | EMERG  | EMERG |
"""

import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable

from ..config import SafetyChainConfig
from ..models import Verdict, Alert, ContextReport
from ..utils.frame_utils import frame_to_base64
from ..utils.logger import get_logger, log_stage_start, log_stage_end

logger = get_logger("stage5.alert_manager")


# Escalation matrix from design doc Section 5
ESCALATION_MATRIX = {
    # (severity, confidence_bucket) → alert_level
    ("LOW", "very_low"):    "LOG",
    ("LOW", "low"):         "NOTIFY",
    ("LOW", "medium"):      "NOTIFY",
    ("LOW", "high"):        "ALERT",
    ("MEDIUM", "very_low"): "LOG",
    ("MEDIUM", "low"):      "NOTIFY",
    ("MEDIUM", "medium"):   "ALERT",
    ("MEDIUM", "high"):     "ALERT",
    ("HIGH", "very_low"):   "NOTIFY",
    ("HIGH", "low"):        "ALERT",
    ("HIGH", "medium"):     "ALERT",
    ("HIGH", "high"):       "EMERGENCY",
    ("CRITICAL", "very_low"): "ALERT",
    ("CRITICAL", "low"):    "EMERGENCY",
    ("CRITICAL", "medium"): "EMERGENCY",
    ("CRITICAL", "high"):   "EMERGENCY",
}


def _confidence_bucket(confidence: float) -> str:
    """Map confidence score to bucket for escalation matrix lookup."""
    if confidence < 0.40:
        return "very_low"
    elif confidence < 0.70:
        return "low"
    elif confidence < 0.90:
        return "medium"
    else:
        return "high"


class AlertManager:
    """Manages alert creation, escalation, and lifecycle.
    
    Implements the escalation state machine:
    Idle → Evaluating → LOG/NOTIFY/ALERT/EMERGENCY
    
    With time-based escalation:
    - NOTIFY unacknowledged for 5 min → ALERT
    - ALERT unacknowledged for 3 min → EMERGENCY
    """

    def __init__(self, config: SafetyChainConfig):
        """Initialize the alert manager.
        
        Args:
            config: SafetyChain configuration
        """
        self.config = config
        self.alerts: Dict[str, Alert] = {}  # alert_id → Alert
        self.alert_history: List[Alert] = []
        self._on_new_alert: Optional[Callable] = None  # WebSocket callback

        logger.info("AlertManager initialized")

    def set_alert_callback(self, callback: Callable):
        """Set callback function for new alerts (used by WebSocket server).
        
        Args:
            callback: Async function called with Alert when a new alert is created
        """
        self._on_new_alert = callback

    def create_alert(
        self,
        verdict: Verdict,
        context: ContextReport,
        frame_annotated,
    ) -> Optional[Alert]:
        """Create an alert from a verdict, applying the escalation matrix.
        
        Args:
            verdict: The verification verdict
            context: The context report
            frame_annotated: Annotated frame (OpenCV BGR numpy array)
            
        Returns:
            Alert object, or None if verdict is FALSE_POSITIVE with very low confidence
        """
        log_stage_start(logger, "ACT", verdict.candidate_id)
        start_time = time.time()

        # Skip creating alerts for clear false positives
        if verdict.classification == "FALSE_POSITIVE" and verdict.confidence < 0.3:
            log_stage_end(logger, "ACT", verdict.candidate_id, 0, "DROPPED_FP")
            return None

        # Determine alert severity using escalation matrix
        bucket = _confidence_bucket(verdict.confidence)
        severity = ESCALATION_MATRIX.get(
            (verdict.severity, bucket), "NOTIFY"
        )

        # ZeroThink always escalates to EMERGENCY
        if verdict.reasoning_strategy == "ZeroThink":
            severity = "EMERGENCY"

        # Build context summary
        context_summary = self._build_context_summary(verdict, context)

        # Encode frame to base64
        frame_b64 = frame_to_base64(frame_annotated) if frame_annotated is not None else ""

        # Build alert title
        title = self._build_title(verdict, context)

        alert = Alert(
            alert_id=f"a-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:4]}",
            chain_id=verdict.chain_id,
            timestamp=datetime.now(),
            severity=severity,
            title=title,
            zone_name=context.zone.zone_name,
            confidence=verdict.confidence,
            frame_b64=frame_b64,
            verdict=verdict,
            context_summary=context_summary,
            sop=context.protocol.procedure_summary,
            contacts=context.protocol.contacts,
        )

        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        latency_ms = int((time.time() - start_time) * 1000)
        log_stage_end(logger, "ACT", verdict.candidate_id, latency_ms, severity)

        logger.info(
            f"Alert created: {alert.alert_id} | {severity} | {title} | "
            f"confidence={verdict.confidence:.0%} | strategy={verdict.reasoning_strategy}"
        )

        return alert

    def process_feedback(self, alert_id: str, feedback: str,
                         note: str = None) -> bool:
        """Process operator feedback (TP/FP) for an alert.
        
        Args:
            alert_id: The alert to update
            feedback: "true_positive" or "false_positive"
            note: Optional operator note
            
        Returns:
            True if the alert was found and updated
        """
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.operator_feedback = feedback
        alert.status = "resolved"

        logger.info(
            f"Feedback received: {alert_id} → {feedback}"
            + (f" (note: {note})" if note else "")
        )

        return True

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged by the operator."""
        if alert_id not in self.alerts:
            return False
        self.alerts[alert_id].status = "acknowledged"
        return True

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert."""
        if alert_id not in self.alerts:
            return False
        self.alerts[alert_id].status = "dismissed"
        return True

    def get_active_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts, most recent first."""
        sorted_alerts = sorted(
            self.alert_history, key=lambda a: a.timestamp, reverse=True
        )
        return sorted_alerts[:limit]

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert by ID."""
        return self.alerts.get(alert_id)

    def get_stats(self) -> Dict:
        """Get dashboard statistics."""
        total = len(self.alert_history)
        tp = sum(1 for a in self.alert_history if a.operator_feedback == "true_positive")
        fp = sum(1 for a in self.alert_history if a.operator_feedback == "false_positive")
        active = sum(1 for a in self.alert_history if a.status == "active")

        return {
            "total_alerts": total,
            "true_positives": tp,
            "false_positives": fp,
            "active_alerts": active,
            "suppressed": sum(
                1 for a in self.alert_history
                if a.verdict.reasoning_strategy == "Suppressed"
            ),
        }

    def _build_title(self, verdict: Verdict, context: ContextReport) -> str:
        """Build a human-readable alert title."""
        zone_type = context.zone.zone_type

        if verdict.reasoning_strategy == "ZeroThink":
            if zone_type == "school":
                return "⚠️ PERIMETER BREACH — School Zone"
            return "⚠️ CRITICAL THREAT DETECTED"

        # Build from classification
        title_map = {
            "vehicle_breakin": "Possible Vehicle Break-in",
            "school_intrusion": "Perimeter Intrusion — School Zone",
            "loitering": "Suspicious Loitering Detected",
        }

        # Try to infer from context
        if context.protocol.matching_sop:
            sop_title = context.protocol.matching_sop
            if "Vehicle" in sop_title:
                return "Possible Vehicle Break-in"
            elif "School" in sop_title or "Perimeter" in sop_title:
                return "Perimeter Intrusion — School Zone"
            elif "Loitering" in sop_title:
                return "Suspicious Loitering Detected"

        if verdict.severity == "CRITICAL":
            return "⚠️ Critical Security Event"
        elif verdict.severity == "HIGH":
            return f"Security Alert — {context.zone.zone_name}"
        else:
            return f"Anomaly Detected — {context.zone.zone_name}"

    def _build_context_summary(self, verdict: Verdict,
                                context: ContextReport) -> str:
        """Build a one-line context summary for the alert."""
        parts = []

        # Time context
        time_str = context.temporal.current_time.strftime("%H:%M")
        if not context.temporal.is_within_active_hours:
            parts.append(f"After-hours ({time_str})")
        else:
            parts.append(f"During active hours ({time_str})")

        # Zone
        parts.append(f"in {context.zone.zone_name}")

        # Verdict context
        if context.verdict == "SUPPORTS_ANOMALY":
            parts.append("— context supports anomaly")
        elif context.verdict == "REFUTES_ANOMALY":
            parts.append("— context refutes anomaly")

        return " ".join(parts)
