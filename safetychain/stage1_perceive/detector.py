"""SafetyChain — Stage 1: PERCEIVE — Object Detector

Wraps YOLOv8n via ultralytics for object detection.
Processes frames and returns list[Detection] filtered to classes of interest.
Architecture ref: Section 3 (Stage 1 — PERCEIVE: Detection Architecture)
"""

from ultralytics import YOLO
import numpy as np
from typing import List

from ..config import SafetyChainConfig
from ..models import Detection
from ..utils.logger import get_logger

logger = get_logger("stage1.detector")


class ObjectDetector:
    """YOLOv8n object detection wrapper.
    
    Loads the YOLOv8n model and runs inference on frames,
    filtering results to classes of interest for safety monitoring.
    """

    def __init__(self, config: SafetyChainConfig):
        """Initialize the detector with YOLOv8n model.
        
        Args:
            config: SafetyChain configuration with YOLO settings
        """
        self.config = config
        self.model = YOLO(config.YOLO_MODEL_PATH)
        self.classes_of_interest = set(config.YOLO_CLASSES_OF_INTEREST)
        logger.info(
            f"ObjectDetector initialized with model={config.YOLO_MODEL_PATH}, "
            f"threshold={config.YOLO_CONFIDENCE_THRESHOLD}, "
            f"classes={config.YOLO_CLASSES_OF_INTEREST}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLOv8n detection on a single frame.
        
        Args:
            frame: OpenCV BGR frame to process
            
        Returns:
            List of Detection objects for classes of interest above confidence threshold
        """
        results = self.model(
            frame,
            conf=self.config.YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                # Filter to classes of interest
                if class_name not in self.classes_of_interest:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                ))

        return detections
