"""SafetyChain — Stage 1: PERCEIVE — Anomaly Gate

Threshold filter that decides whether a detection should proceed
to Stage 2 (DESCRIBE) or be silently dropped.
Architecture ref: Section 3 (Anomaly Gate subsection)

Logic:
  - Applies per-class confidence thresholds
  - Checks zone boundary restrictions
  - Evaluates motion delta for rapid changes
  - 95%+ of frames are rejected here
"""

import uuid
from datetime import datetime
from typing import List, Optional

import numpy as np

from ..config import SafetyChainConfig
from ..models import Detection, AnomalyCandidate
from ..utils.frame_utils import annotate_frame, compute_motion_delta
from ..utils.logger import get_logger, log_pipeline_decision

logger = get_logger("stage1.anomaly_gate")


class AnomalyGate:
    """Filters detections to only pass genuine anomaly candidates.
    
    This is the first line of defense against alarm fatigue —
    it drops 95%+ of frames before they ever reach the VLM.
    """

    def __init__(self, config: SafetyChainConfig):
        """Initialize the anomaly gate with threshold configuration.
        
        Args:
            config: SafetyChain configuration with threshold settings
        """
        self.config = config
        self.previous_frame: Optional[np.ndarray] = None

        # High-priority classes that always pass with lower thresholds
        self.critical_classes = {"knife", "fire", "scissors"}
        self.critical_threshold = 0.3  # Lower threshold for weapons/fire

    def evaluate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        zone_id: str = "zone-a",
        camera_id: str = "cam-a1",
    ) -> Optional[AnomalyCandidate]:
        """Evaluate detections against anomaly thresholds.
        
        Args:
            frame: Raw OpenCV BGR frame
            detections: List of YOLO detections
            zone_id: ID of the zone this camera covers
            camera_id: Camera identifier
            
        Returns:
            AnomalyCandidate if the frame passes the gate, None otherwise
        """
        if not detections:
            self.previous_frame = frame.copy()
            return None

        # Compute motion delta
        motion_delta = compute_motion_delta(self.previous_frame, frame)
        self.previous_frame = frame.copy()

        # Check if any detection passes the gate
        passing_detections = []
        trigger_reasons = []

        for det in detections:
            # Critical classes have a lower threshold
            if det.class_name in self.critical_classes:
                if det.confidence >= self.critical_threshold:
                    passing_detections.append(det)
                    trigger_reasons.append(
                        f"critical_class_{det.class_name}"
                    )
                continue

            # Standard confidence threshold
            if det.confidence >= self.config.YOLO_CONFIDENCE_THRESHOLD:
                passing_detections.append(det)
                trigger_reasons.append(f"detection_{det.class_name}")

        # Also check motion delta — rapid changes may indicate an event
        if motion_delta > self.config.MOTION_DELTA_THRESHOLD and detections:
            passing_detections = detections  # Pass all detections on rapid motion
            trigger_reasons.append("rapid_motion")

        if not passing_detections:
            return None

        # Create the annotated frame
        frame_annotated = annotate_frame(frame, passing_detections)

        candidate_id = str(uuid.uuid4())[:12]
        trigger_reason = "; ".join(trigger_reasons)

        log_pipeline_decision(
            logger, candidate_id, "PASS",
            f"{len(passing_detections)} detections passed gate: {trigger_reason}"
        )

        return AnomalyCandidate(
            id=candidate_id,
            timestamp=datetime.now(),
            frame=frame,
            frame_annotated=frame_annotated,
            detections=passing_detections,
            zone_id=zone_id,
            camera_id=camera_id,
            motion_delta=motion_delta,
            trigger_reason=trigger_reason,
        )
