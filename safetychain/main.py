"""SafetyChain — Pipeline Orchestrator

Entry point that wires all 5 stages together.
Video loop: read frame → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5

Architecture ref: Section 2 (Data Flow — Inter-Stage Communication)
Implementation plan ref: Phase 7
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from typing import Optional, Callable

from .config import SafetyChainConfig
from .models import AnomalyCandidate, Alert
from .stage1_perceive.detector import ObjectDetector
from .stage1_perceive.anomaly_gate import AnomalyGate
from .stage2_describe.scene_describer import SceneDescriber
from .stage3_contextualize.knowledge_graph import KnowledgeGraph
from .stage3_contextualize.context_engine import ContextEngine
from .stage4_verify.cot_verifier import CoTVerifier
from .stage5_act.alert_manager import AlertManager
from .stage5_act.evidence_packager import EvidencePackager
from .utils.frame_utils import extract_frames
from .utils.logger import get_logger

logger = get_logger("pipeline")


class SafetyChainPipeline:
    """The main SafetyChain pipeline orchestrator.
    
    Connects all 5 stages into a continuous processing pipeline:
    PERCEIVE → DESCRIBE → CONTEXTUALIZE → VERIFY → ACT
    """

    def __init__(self, config: SafetyChainConfig = None, base_dir: str = None):
        """Initialize all pipeline components.
        
        Args:
            config: Configuration (uses defaults if None)
            base_dir: Base directory for data files
        """
        self.config = config or SafetyChainConfig()
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))

        # Pipeline status tracking
        self.status = {
            "active_stage": "idle",
            "fps": 0.0,
            "frames_processed": 0,
            "alerts_today": 0,
            "pipeline_active": False,
            "cumulative_latency_ms": 0,
        }

        # Initialize all stages
        self._init_stages()

        # Callback for broadcasting status/alerts
        self._on_alert: Optional[Callable] = None
        self._on_status: Optional[Callable] = None

        logger.info("SafetyChain Pipeline initialized")

    def _init_stages(self):
        """Initialize all pipeline stage components."""
        # Stage 1: PERCEIVE
        self.detector = ObjectDetector(self.config)
        self.gate = AnomalyGate(self.config)

        # Stage 2: DESCRIBE
        self.describer = SceneDescriber(self.config)

        # Stage 3: CONTEXTUALIZE
        db_path = self.config.get_db_path(self.base_dir)
        zones_path = self.config.get_zones_path(self.base_dir)
        sops_path = self.config.get_sops_path(self.base_dir)

        self.knowledge_graph = KnowledgeGraph(db_path, zones_path, sops_path)
        self.context_engine = ContextEngine(self.config, self.knowledge_graph)

        # Stage 4: VERIFY
        self.verifier = CoTVerifier(self.config)

        # Stage 5: ACT
        self.alert_manager = AlertManager(self.config)
        self.evidence_packager = EvidencePackager(
            os.path.join(self.base_dir, "evidence")
        )

    def set_callbacks(self, on_alert: Callable = None, on_status: Callable = None):
        """Set callbacks for alert and status broadcasting.
        
        Args:
            on_alert: Called with (Alert) when a new alert is created
            on_status: Called with (status_dict) on pipeline status updates
        """
        self._on_alert = on_alert
        self._on_status = on_status

    def process_frame(
        self,
        frame,
        zone_id: str = "zone-a",
        camera_id: str = "cam-a1",
    ) -> Optional[Alert]:
        """Process a single frame through the full 5-stage pipeline.
        
        Args:
            frame: OpenCV BGR frame
            zone_id: Zone this camera covers
            camera_id: Camera identifier
            
        Returns:
            Alert if an anomaly was detected and verified, None otherwise
        """
        pipeline_start = time.time()

        # ═══ Stage 1: PERCEIVE ═══
        self._update_status("PERCEIVE")
        detections = self.detector.detect(frame)
        candidate = self.gate.evaluate(frame, detections, zone_id, camera_id)

        if candidate is None:
            self._update_status("idle")
            self.status["frames_processed"] += 1
            return None

        # ═══ Stage 2: DESCRIBE ═══
        self._update_status("DESCRIBE")

        # Get zone info for the describer
        zone = self.knowledge_graph.get_zone(zone_id)
        zone_name = zone["name"] if zone else "Unknown Zone"
        zone_type = zone["type"] if zone else "general"

        scene = self.describer.describe(candidate, zone_name, zone_type)

        # ═══ Stage 3: CONTEXTUALIZE ═══
        self._update_status("CONTEXTUALIZE")
        context = self.context_engine.contextualize(candidate, scene)

        # Check if context suppresses the alert
        if context.suppress:
            logger.info(
                f"Alert suppressed by context: {context.suppress_reason}"
            )
            self._update_status("idle")
            return None

        # ═══ Stage 4: VERIFY ═══
        self._update_status("VERIFY")
        verdict = self.verifier.verify(candidate, scene, context)

        # ═══ Stage 5: ACT ═══
        self._update_status("ACT")
        alert = self.alert_manager.create_alert(
            verdict, context, candidate.frame_annotated
        )

        if alert:
            # Save evidence
            self.evidence_packager.save_evidence_json(alert)

            # Record in knowledge graph
            self.knowledge_graph.record_event(
                zone_id=zone_id,
                camera_id=camera_id,
                event_type="alert",
                description=alert.title,
                chain_id=alert.chain_id,
                detection_class=candidate.detections[0].class_name
                if candidate.detections else None,
            )

            self.status["alerts_today"] += 1

            # Notify via callback
            if self._on_alert:
                self._on_alert(alert)

        # Update pipeline metrics
        pipeline_latency = int((time.time() - pipeline_start) * 1000)
        self.status["cumulative_latency_ms"] = pipeline_latency
        self.status["frames_processed"] += 1
        self._update_status("idle")

        return alert

    def process_video(
        self,
        video_path: str,
        zone_id: str = "zone-a",
        camera_id: str = "cam-a1",
        max_frames: int = None,
    ) -> list:
        """Process a video file through the pipeline.
        
        Args:
            video_path: Path to video file
            zone_id: Zone this camera covers
            camera_id: Camera identifier
            max_frames: Maximum frames to process (None = all)
            
        Returns:
            List of Alert objects generated
        """
        logger.info(f"Processing video: {video_path}")
        self.status["pipeline_active"] = True
        alerts = []
        frame_times = []

        for frame_num, frame in extract_frames(video_path, self.config.FRAME_SKIP):
            frame_start = time.time()

            alert = self.process_frame(frame, zone_id, camera_id)
            if alert:
                alerts.append(alert)

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)

            # Update FPS
            if frame_times:
                avg_time = sum(frame_times[-30:]) / len(frame_times[-30:])
                self.status["fps"] = 1.0 / avg_time if avg_time > 0 else 0

            if max_frames and frame_num >= max_frames:
                break

        self.status["pipeline_active"] = False
        logger.info(
            f"Video processing complete: {len(alerts)} alerts from "
            f"{self.status['frames_processed']} frames"
        )

        return alerts

    def process_feedback(self, alert_id: str, feedback: str,
                         note: str = None) -> bool:
        """Process operator feedback and update knowledge graph.
        
        Args:
            alert_id: Alert to update
            feedback: "true_positive" or "false_positive"
            note: Optional operator note
            
        Returns:
            True if feedback was processed successfully
        """
        # Update alert
        success = self.alert_manager.process_feedback(alert_id, feedback, note)

        if success:
            alert = self.alert_manager.get_alert(alert_id)
            if alert:
                # Update knowledge graph (feedback loop)
                self.knowledge_graph.update_from_feedback(
                    chain_id=alert.chain_id,
                    was_false_positive=(feedback == "false_positive"),
                    operator_note=note,
                )

        return success

    def get_status(self) -> dict:
        """Get current pipeline status."""
        return self.status.copy()

    def _update_status(self, stage: str):
        """Update the active pipeline stage."""
        self.status["active_stage"] = stage
        if self._on_status:
            self._on_status(self.status.copy())

    def shutdown(self):
        """Clean shutdown of all components."""
        self.knowledge_graph.close()
        logger.info("SafetyChain Pipeline shut down")
