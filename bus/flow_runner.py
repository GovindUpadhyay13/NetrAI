"""
bus/flow_runner.py
Orchestrates the detect -> analyze pipeline across Milestones 1-3 wired to the Event Bus:
1. AnomalyCLIP branch -> publishes 'anomaly_detected' event.
2. MediaPipe Gesture branch -> publishes 'gesture_flagged' event under the same incident_id.
3. VLM Reasoning -> merges signals, constructs 3x3 grid, invokes Gemini, publishes 'vlm_analyzed' event.
4. TraceLogger -> writes each event into SQLite trace.db.
"""

import os
import sys
from uuid import uuid4
from typing import Dict, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bus.schemas import SeverityEnum, StageEnum, SurveillanceEvent
from bus.publisher import EventBus
from bus.consumers.trace_logger import TraceLoggerConsumer
from detection.anomaly_clip.scorer import VideoAnomalyScorer
from detection.gesture.pose_extractor import PoseExtractor
from detection.gesture.distress_classifier import DistressClassifier
from reasoning.frame_grid import FrameGridBuilder
from reasoning.gemini_analyzer import GeminiIncidentAnalyzer
from ingestion.video_reader import VideoReader


class SurveillancePipelineRunner:
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        db_path: Optional[str] = None,
        output_dir: str = "outputs/pipeline",
    ):
        self.bus = event_bus or EventBus()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize consumers
        self.trace_logger = TraceLoggerConsumer(bus=self.bus, db_path=db_path or "trace.db")

        # Initialize detection & reasoning engines
        self.anomaly_scorer = VideoAnomalyScorer()
        self.distress_classifier = DistressClassifier()
        self.grid_builder = FrameGridBuilder()
        self.vlm_analyzer = GeminiIncidentAnalyzer()

    def process_video_incident(
        self,
        video_path: str,
        camera_id: str = "CAM-EAST-CORRIDOR-02",
        incident_id: Optional[str] = None,
        anomaly_threshold: float = 0.20,
    ) -> Dict:
        """
        Runs the full detection and VLM reasoning flow on a video,
        publishing events to the bus at every stage.
        """
        inc_id = incident_id or str(uuid4())
        print(f"\n=======================================================")
        print(f" Starting Surveillance Event Flow | Incident: {inc_id}")
        print(f" Camera: {camera_id} | Video: {video_path}")
        print(f"=======================================================")

        with VideoReader(video_path) as reader:
            frames, timestamps = reader.read_all_frames()

        if not frames:
            raise ValueError(f"Could not load frames from {video_path}")

        # -------------------------------------------------------------
        # Branch 1: AnomalyCLIP scoring
        # -------------------------------------------------------------
        print("\n[Stage 1] Running AnomalyCLIP branch...")
        anomaly_res = self.anomaly_scorer.score_video(video_path, threshold=anomaly_threshold)
        flagged_windows = anomaly_res.get("flagged_windows", [])
        peak_anomaly_score = max(anomaly_res.get("scores", [0.0])) if anomaly_res.get("scores") else 0.0

        anomaly_type = "Suspicious Activity"
        start_sec, end_sec = 0.0, float(timestamps[-1]) if timestamps else 0.0

        if flagged_windows:
            win = flagged_windows[0]
            anomaly_type = win["anomaly_type"]
            peak_anomaly_score = win["peak_score"]
            start_sec = win["start_time_sec"]
            end_sec = win["end_time_sec"]

        # Publish Stage 1: anomaly_detected
        event_stage1 = SurveillanceEvent(
            incident_id=inc_id,
            camera_id=camera_id,
            stage=StageEnum.ANOMALY_DETECTED,
            anomaly_score=round(peak_anomaly_score, 4),
            anomaly_type=anomaly_type,
            distress_gesture=False,
            payload_ref=video_path,
        )
        self.bus.publish(event_stage1)

        # -------------------------------------------------------------
        # Branch 2: MediaPipe Gesture Branch (under same incident_id)
        # -------------------------------------------------------------
        print("\n[Stage 2] Running MediaPipe Distress Gesture branch...")
        with PoseExtractor() as extractor:
            pose_frames = extractor.process_video_frames(frames, timestamps)
        gesture_res = self.distress_classifier.classify_clip(pose_frames)

        # Publish Stage 2: gesture_flagged
        event_stage2 = SurveillanceEvent(
            incident_id=inc_id,
            camera_id=camera_id,
            stage=StageEnum.GESTURE_FLAGGED,
            anomaly_score=round(peak_anomaly_score, 4),
            anomaly_type=anomaly_type,
            distress_gesture=gesture_res.is_distress,
            severity=SeverityEnum.HIGH if gesture_res.is_distress else SeverityEnum.LOW,
            payload_ref=video_path,
        )
        self.bus.publish(event_stage2)

        # -------------------------------------------------------------
        # Stage 3: VLM Reasoning (Merge signals -> 3x3 Grid -> Gemini)
        # -------------------------------------------------------------
        print("\n[Stage 3] Merging signals and invoking Gemini VLM Reasoner...")
        grid_out_path = os.path.join(self.output_dir, f"{inc_id}_grid_3x3.png")
        grid_img = self.grid_builder.build_grid_from_frames(frames, timestamps, output_path=grid_out_path)

        vlm_report = self.vlm_analyzer.analyze_incident(
            grid_image=grid_img,
            camera_id=camera_id,
            start_sec=start_sec,
            end_sec=end_sec,
            anomaly_type_prior=anomaly_type,
            anomaly_score=peak_anomaly_score,
            distress_gesture_flag=gesture_res.is_distress,
            distress_gesture_type=gesture_res.gesture_type,
            gesture_confidence=gesture_res.confidence,
        )

        sev_enum = SeverityEnum(vlm_report.severity) if vlm_report.severity in [s.value for s in SeverityEnum] else SeverityEnum.MEDIUM

        # Publish Stage 3: vlm_analyzed
        event_stage3 = SurveillanceEvent(
            incident_id=inc_id,
            camera_id=camera_id,
            stage=StageEnum.VLM_ANALYZED,
            anomaly_score=round(peak_anomaly_score, 4),
            anomaly_type=anomaly_type,
            distress_gesture=gesture_res.is_distress,
            vlm_report=f"[{vlm_report.recommended_department}] {vlm_report.incident_description}",
            severity=sev_enum,
            payload_ref=grid_out_path,
        )
        self.bus.publish(event_stage3)

        print("\n[Stage Complete] All events published and traced into SQLite.")
        return {
            "incident_id": inc_id,
            "camera_id": camera_id,
            "anomaly_score": peak_anomaly_score,
            "distress_detected": gesture_res.is_distress,
            "severity": vlm_report.severity,
            "recommended_department": vlm_report.recommended_department,
            "grid_image": grid_out_path,
        }
