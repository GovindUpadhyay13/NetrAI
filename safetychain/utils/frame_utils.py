"""SafetyChain — Frame Utilities

Frame annotation (draw YOLO bounding boxes), JPEG encoding to base64,
and frame extraction helpers.
"""

import base64
import cv2
import numpy as np
from typing import List
from ..models import Detection


# Severity-based colors (BGR for OpenCV)
COLORS = {
    "person": (0, 165, 255),     # Orange
    "car": (255, 200, 0),        # Cyan-ish
    "truck": (255, 200, 0),      # Cyan-ish
    "knife": (0, 0, 255),        # Red
    "scissors": (0, 0, 255),     # Red
    "backpack": (0, 255, 255),   # Yellow
    "fire": (0, 0, 255),         # Red
    "default": (0, 255, 0),      # Green
}


def annotate_frame(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw YOLO bounding boxes and labels on a frame.
    
    Args:
        frame: Raw OpenCV BGR frame
        detections: List of Detection objects with class_name, confidence, bbox
        
    Returns:
        Annotated frame with bounding boxes and labels drawn
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = COLORS.get(det.class_name, COLORS["default"])
        label = f"{det.class_name} {det.confidence:.0%}"

        # Draw bounding box
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated,
            (int(x1), int(y1) - label_h - baseline - 4),
            (int(x1) + label_w, int(y1)),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (int(x1), int(y1) - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode an OpenCV frame as base64 JPEG string.
    
    Args:
        frame: OpenCV BGR frame
        quality: JPEG compression quality (0-100)
        
    Returns:
        Base64-encoded JPEG string
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_frame(b64_string: str) -> np.ndarray:
    """Decode a base64 JPEG string back to an OpenCV frame.
    
    Args:
        b64_string: Base64-encoded JPEG string
        
    Returns:
        OpenCV BGR frame
    """
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def extract_frames(video_path: str, frame_skip: int = 1):
    """Generator that yields frames from a video file.
    
    Args:
        video_path: Path to the video file
        frame_skip: Process every Nth frame (1 = every frame)
        
    Yields:
        Tuple of (frame_number, frame) for each extracted frame
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                yield frame_count, frame

            frame_count += 1
    finally:
        cap.release()


def compute_motion_delta(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute motion delta between two frames using absolute difference.
    
    Args:
        frame1: Previous frame (OpenCV BGR)
        frame2: Current frame (OpenCV BGR)
        
    Returns:
        Normalized motion score (0.0 = no motion, 1.0 = maximum motion)
    """
    if frame1 is None or frame2 is None:
        return 0.0

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Normalized by total pixels
    motion_score = np.count_nonzero(thresh) / thresh.size
    return float(motion_score)
