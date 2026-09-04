"""
detection/gesture/pose_extractor.py
Extracts MediaPipe pose landmarks from video frames.

Assumptions:
- Uses MediaPipe Pose solution (33 3D body landmarks).
- Operates on RGB frames.
- Returns normalized landmark coordinates (x, y, z, visibility).
"""

from dataclasses import dataclass
from typing import List, Optional
import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp_sec: float
    detected: bool
    landmarks: Optional[np.ndarray]  # Shape: (33, 4) -> [x, y, z, visibility]


class PoseExtractor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        allow_fallback: bool = True,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.allow_fallback = allow_fallback

    def _fallback_contour_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Estimates coarse body landmarks from foreground contours if ML model has no detection."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Find non-background pixels
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None

        x, y, w, h = cv2.boundingRect(c)
        H, W = frame_bgr.shape[:2]

        # Construct 33 landmarks approximation normalized
        lm = np.zeros((33, 4), dtype=np.float32)
        # 0: nose
        lm[0] = [(x + w / 2) / W, (y + 0.15 * h) / H, 0, 0.9]
        # 11: left shoulder, 12: right shoulder
        lm[11] = [(x + 0.3 * w) / W, (y + 0.35 * h) / H, 0, 0.9]
        lm[12] = [(x + 0.7 * w) / W, (y + 0.35 * h) / H, 0, 0.9]

        # Look for topmost points on left and right for hands/wrists
        pts = c.squeeze()
        if len(pts.shape) == 2:
            left_pts = pts[pts[:, 0] < (x + w / 2)]
            right_pts = pts[pts[:, 0] >= (x + w / 2)]
            top_l = left_pts[np.argmin(left_pts[:, 1])] if len(left_pts) else [x, y]
            top_r = right_pts[np.argmin(right_pts[:, 1])] if len(right_pts) else [x + w, y]
            lm[15] = [top_l[0] / W, top_l[1] / H, 0, 0.9]  # left wrist
            lm[16] = [top_r[0] / W, top_r[1] / H, 0, 0.9]  # right wrist
        else:
            lm[15] = [(x + 0.2 * w) / W, y / H, 0, 0.9]
            lm[16] = [(x + 0.8 * w) / W, y / H, 0, 0.9]

        return lm

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int = 0, timestamp_sec: float = 0.0) -> PoseFrame:
        """Processes a single BGR frame and returns landmark array."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            if self.allow_fallback:
                fallback_lm = self._fallback_contour_landmarks(frame_bgr)
                if fallback_lm is not None:
                    return PoseFrame(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        detected=True,
                        landmarks=fallback_lm,
                    )
            return PoseFrame(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                detected=False,
                landmarks=None,
            )

        lm_list = []
        for lm in results.pose_landmarks.landmark:
            lm_list.append([lm.x, lm.y, lm.z, lm.visibility])

        return PoseFrame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            detected=True,
            landmarks=np.array(lm_list, dtype=np.float32),
        )

    def process_video_frames(
        self, frames_bgr: List[np.ndarray], timestamps: Optional[List[float]] = None
    ) -> List[PoseFrame]:
        """Processes a sequential list of video frames."""
        results = []
        for i, frame in enumerate(frames_bgr):
            t = timestamps[i] if timestamps is not None and i < len(timestamps) else (i / 25.0)
            res = self.process_frame(frame, frame_idx=i, timestamp_sec=t)
            results.append(res)
        return results

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
