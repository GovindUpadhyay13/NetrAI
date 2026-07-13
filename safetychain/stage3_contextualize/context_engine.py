"""SafetyChain — Stage 3: CONTEXTUALIZE — Context Engine

Aggregates zone norms, temporal context, historical FP patterns, and SOPs
to produce a ContextReport that supports, is neutral to, or refutes the anomaly.

Architecture ref: Section 5 (Stage 3 — Context Aggregation Flow)
Design ref: Section 2.3 (ContextReport data model)
"""

import json
import time
from datetime import datetime
from typing import Optional

from ..config import SafetyChainConfig
from ..models import (
    AnomalyCandidate, SceneDescription, ContextReport,
    ZoneContext, TemporalContext, HistoricalContext, ProtocolContext,
)
from .knowledge_graph import KnowledgeGraph
from ..utils.logger import get_logger, log_stage_start, log_stage_end

logger = get_logger("stage3.context_engine")


class ContextEngine:
    """Aggregates contextual information to enrich anomaly assessment.
    
    Combines four context sources (run in parallel conceptually):
    1. Knowledge Graph — zone norms and rules
    2. Temporal Engine — current time, day, calendar
    3. History Lookup — known FP patterns, recent events
    4. Protocol RAG — SOPs, contacts
    """

    def __init__(self, config: SafetyChainConfig, knowledge_graph: KnowledgeGraph):
        """Initialize the context engine.
        
        Args:
            config: SafetyChain configuration
            knowledge_graph: SQLite-backed knowledge graph instance
        """
        self.config = config
        self.kg = knowledge_graph
        logger.info("ContextEngine initialized")

    def contextualize(
        self,
        candidate: AnomalyCandidate,
        scene: SceneDescription,
    ) -> ContextReport:
        """Generate a ContextReport by aggregating all context sources.
        
        Args:
            candidate: The anomaly candidate
            scene: The VLM-generated scene description
            
        Returns:
            ContextReport with verdict (SUPPORTS/NEUTRAL/REFUTES) and suppress flag
        """
        log_stage_start(logger, "CONTEXTUALIZE", candidate.id)
        start_time = time.time()

        # 1. Zone context from knowledge graph
        zone_ctx = self._get_zone_context(candidate.zone_id)

        # 2. Temporal context
        temporal_ctx = self._get_temporal_context(zone_ctx)

        # 3. Historical context
        historical_ctx = self._get_historical_context(
            candidate.zone_id, candidate.camera_id
        )

        # 4. Protocol context (SOP lookup)
        protocol_ctx = self._get_protocol_context(
            candidate.zone_id, scene, candidate.detections
        )

        # 5. Aggregate verdict
        verdict, confidence_modifier, suppress, suppress_reason = self._aggregate_verdict(
            zone_ctx, temporal_ctx, historical_ctx, scene
        )

        latency_ms = int((time.time() - start_time) * 1000)
        log_stage_end(logger, "CONTEXTUALIZE", candidate.id, latency_ms, verdict)

        return ContextReport(
            candidate_id=candidate.id,
            zone=zone_ctx,
            temporal=temporal_ctx,
            historical=historical_ctx,
            protocol=protocol_ctx,
            verdict=verdict,
            confidence_modifier=confidence_modifier,
            suppress=suppress,
            suppress_reason=suppress_reason,
        )

    def _get_zone_context(self, zone_id: str) -> ZoneContext:
        """Query the knowledge graph for zone norms and status."""
        zone = self.kg.get_zone(zone_id)
        norms = self.kg.get_zone_norms(zone_id)

        if not zone:
            return ZoneContext(
                zone_id=zone_id,
                zone_name="Unknown Zone",
                zone_type="general",
                active_hours="00:00-23:59",
                currently_active=True,
                expected_occupancy="unknown",
            )

        # Find the time_window norm for active hours
        active_hours = "00:00-23:59"
        expected_occupancy = "general"
        for norm in norms:
            if norm["norm_type"] == "time_window":
                active_hours = norm.get("active_hours", active_hours)
                params = json.loads(norm.get("parameters", "{}"))
                expected_occupancy = params.get("expected_occupancy",
                    "residents_with_badge" if zone["type"] == "parking" else "none"
                )

        # Check if currently within active hours
        currently_active = self._is_within_hours(active_hours)

        return ZoneContext(
            zone_id=zone_id,
            zone_name=zone["name"],
            zone_type=zone["type"],
            active_hours=active_hours,
            currently_active=currently_active,
            expected_occupancy=expected_occupancy,
        )

    def _get_temporal_context(self, zone_ctx: ZoneContext) -> TemporalContext:
        """Get current temporal context."""
        now = datetime.now()

        return TemporalContext(
            current_time=now,
            day_of_week=now.strftime("%A"),
            is_holiday=False,       # Could integrate a holiday calendar
            holiday_name=None,
            is_within_active_hours=zone_ctx.currently_active,
        )

    def _get_historical_context(self, zone_id: str,
                                 camera_id: str) -> HistoricalContext:
        """Look up historical event data for this zone/camera."""
        history = self.kg.get_history(
            zone_id, self.config.FP_HISTORY_WINDOW_DAYS, camera_id
        )
        fp_rate = self.kg.get_false_positive_rate(
            zone_id, camera_id, self.config.FP_HISTORY_WINDOW_DAYS
        )
        known_fp = self.kg.get_known_fp_patterns(zone_id, camera_id)

        last_event = None
        if history:
            try:
                last_event = datetime.fromisoformat(history[0]["timestamp"])
            except (ValueError, TypeError):
                pass

        return HistoricalContext(
            similar_events_count=len(history),
            false_positive_rate=fp_rate,
            known_fp_pattern=known_fp,
            last_event_in_zone=last_event,
        )

    def _get_protocol_context(self, zone_id: str, scene: SceneDescription,
                               detections) -> ProtocolContext:
        """Retrieve matching SOP and contacts."""
        # Determine alert type from scene and detections
        alert_type = self._infer_alert_type(scene, detections)

        sop = self.kg.get_sop(zone_id, alert_type)

        if sop:
            return ProtocolContext(
                matching_sop=sop["title"],
                procedure_summary=sop["procedure"],
                contacts=sop.get("contacts"),
                policy_notes=None,
            )

        # Try getting any SOP for this zone
        sop = self.kg.get_sop(zone_id)
        if sop:
            return ProtocolContext(
                matching_sop=sop["title"],
                procedure_summary=sop["procedure"],
                contacts=sop.get("contacts"),
                policy_notes="Note: No exact SOP match. Using general zone SOP.",
            )

        return ProtocolContext(
            matching_sop=None,
            procedure_summary=None,
            contacts=None,
            policy_notes="No SOP configured for this zone/alert type.",
        )

    def _infer_alert_type(self, scene: SceneDescription, detections) -> Optional[str]:
        """Infer the alert type from scene analysis and detections."""
        # Check for weapon-related detections
        weapon_classes = {"knife", "scissors"}
        for det in detections:
            if det.class_name in weapon_classes:
                return "weapon"

        # Check scene description keywords
        activity_lower = scene.activity.lower() if scene.activity else ""
        if any(w in activity_lower for w in ["break-in", "breakin", "theft", "vehicle"]):
            return "vehicle_breakin"
        if any(w in activity_lower for w in ["intrusion", "perimeter", "fence", "breach"]):
            return "school_intrusion"
        if any(w in activity_lower for w in ["loiter", "suspicious"]):
            return "loitering"

        return None

    def _aggregate_verdict(
        self,
        zone_ctx: ZoneContext,
        temporal_ctx: TemporalContext,
        historical_ctx: HistoricalContext,
        scene: SceneDescription,
    ) -> tuple:
        """Aggregate all context into a verdict.
        
        Returns:
            (verdict, confidence_modifier, suppress, suppress_reason)
        """
        confidence_modifier = 0.0
        suppress = False
        suppress_reason = None

        # ── Check 1: After-hours activity in restricted zone ──
        if not temporal_ctx.is_within_active_hours:
            confidence_modifier += 0.15  # Boost confidence
            verdict_direction = "SUPPORTS"
        else:
            verdict_direction = "NEUTRAL"

        # ── Check 2: School zone is always critical ──
        if zone_ctx.zone_type == "school":
            confidence_modifier += 0.2
            verdict_direction = "SUPPORTS"

        # ── Check 3: Known FP pattern ──
        if historical_ctx.known_fp_pattern:
            # Check if this looks like the known FP
            if historical_ctx.false_positive_rate > 0.7:
                suppress = True
                suppress_reason = f"Known FP pattern: {historical_ctx.known_fp_pattern} (FP rate: {historical_ctx.false_positive_rate:.0%})"
                verdict_direction = "REFUTES"
                confidence_modifier = -0.3

        # ── Check 4: Scene suspiciousness ──
        susp = scene.suspiciousness
        if susp == "ALARMING":
            confidence_modifier += 0.1
            if verdict_direction == "NEUTRAL":
                verdict_direction = "SUPPORTS"
        elif susp == "NORMAL":
            confidence_modifier -= 0.1
            if verdict_direction == "NEUTRAL":
                verdict_direction = "REFUTES"

        # Clamp modifier
        confidence_modifier = max(-0.3, min(0.3, confidence_modifier))

        # Map direction to final verdict string
        verdict_map = {
            "SUPPORTS": "SUPPORTS_ANOMALY",
            "NEUTRAL": "NEUTRAL",
            "REFUTES": "REFUTES_ANOMALY",
        }

        return (
            verdict_map.get(verdict_direction, "NEUTRAL"),
            confidence_modifier,
            suppress,
            suppress_reason,
        )

    def _is_within_hours(self, active_hours: str) -> bool:
        """Check if current time is within the active hours range.
        
        Args:
            active_hours: String like "06:00-23:00"
            
        Returns:
            True if current time is within the range
        """
        try:
            start_str, end_str = active_hours.split("-")
            now = datetime.now()
            start_h, start_m = map(int, start_str.split(":"))
            end_h, end_m = map(int, end_str.split(":"))

            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            now_minutes = now.hour * 60 + now.minute

            if start_minutes <= end_minutes:
                return start_minutes <= now_minutes <= end_minutes
            else:
                # Overnight range (e.g., 23:00-06:00)
                return now_minutes >= start_minutes or now_minutes <= end_minutes
        except (ValueError, AttributeError):
            return True  # Default to active if parsing fails
