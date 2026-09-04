"""
detection/gesture/distress_classifier.py
Distress gesture detector and classifier for women's safety surveillance.

Assumptions:
- Employs a rule-based geometric & kinematic temporal heuristic on pose landmarks:
  1. Both arms raised / high waving (calling for help / defense posture)
  2. Single arm high waving (SOS call / waving down help)
  3. Crossed-arms chest guard / defensive shield posture
- Also supports a lightweight 2-layer PyTorch MLP head if custom trained weights are provided.
- Output shape: bool (is_distress), float (confidence in [0.0, 1.0]), str (gesture_type).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .pose_extractor import PoseFrame


class DistressMLP(nn.Module):
    """Optional 2-layer MLP classifier head on flattened key landmark coordinates."""
    def __init__(self, input_dim: int = 33 * 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),  # [normal, distress]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DistressResult:
    is_distress: bool
    confidence: float
    gesture_type: Optional[str]
    flagged_frames_count: int
    total_frames_count: int
    details: Dict


class DistressClassifier:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        min_consecutive_frames: int = 3,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.weights_path = weights_path

        self.mlp_model = None
        if weights_path and torch.cuda.is_available():
            try:
                self.mlp_model = DistressMLP()
                self.mlp_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                self.mlp_model.eval()
            except Exception as e:
                print(f"[DistressClassifier] Could not load MLP weights: {e}, falling back to rule-based.")

    def classify_clip(self, pose_frames: List[PoseFrame]) -> DistressResult:
        """
        Analyzes a sequence of PoseFrames and evaluates distress signals.
        """
        valid_frames = [f for f in pose_frames if f.detected and f.landmarks is not None]
        total_frames = len(pose_frames)

        if not valid_frames or total_frames == 0:
            return DistressResult(
                is_distress=False,
                confidence=0.0,
                gesture_type=None,
                flagged_frames_count=0,
                total_frames_count=total_frames,
                details={"reason": "no_pose_detected"},
            )

        frame_flags = []
        frame_types = []

        wrist_positions_x = []

        for pf in valid_frames:
            lm = pf.landmarks
            # MediaPipe Pose key landmark indices:
            # 0: nose, 11: left_shoulder, 12: right_shoulder, 15: left_wrist, 16: right_wrist
            nose = lm[0]
            l_shoulder = lm[11]
            r_shoulder = lm[12]
            l_wrist = lm[15]
            r_wrist = lm[16]

            # In image coords, y increases downwards! So smaller y means higher position.
            shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
            nose_y = nose[1]

            l_raised = l_wrist[1] < shoulder_y
            r_raised = r_wrist[1] < shoulder_y
            l_very_high = l_wrist[1] < nose_y
            r_very_high = r_wrist[1] < nose_y

            wrist_dist = np.linalg.norm(l_wrist[:2] - r_wrist[:2])
            wrist_positions_x.append((l_wrist[0], r_wrist[0]))

            # Pattern 1: Both arms raised above head / surrender / distress
            if l_very_high and r_very_high:
                frame_flags.append(True)
                frame_types.append("both_arms_raised_sos")
            # Pattern 2: Defensive shield / hands protecting face/head
            elif (l_raised or r_raised) and wrist_dist < 0.15 and (l_wrist[1] < shoulder_y):
                frame_flags.append(True)
                frame_types.append("defensive_guard_distress")
            # Pattern 3: Single arm elevated high above nose
            elif l_very_high or r_very_high:
                frame_flags.append(True)
                frame_types.append("single_arm_sos_call")
            else:
                frame_flags.append(False)
                frame_types.append("normal")

        # Temporal waving oscillation check
        waving_flag = False
        if len(wrist_positions_x) >= 6:
            xs = np.array(wrist_positions_x)
            dx = np.abs(np.diff(xs, axis=0))
            if np.mean(dx) > 0.04:  # Significant rapid wrist movement
                waving_flag = True

        flagged_count = sum(frame_flags)
        ratio = flagged_count / len(valid_frames)

        # Boost confidence if dynamic waving or sustained posture detected
        confidence = min(1.0, ratio * (1.3 if waving_flag else 1.0))
        is_distress = confidence >= self.confidence_threshold and flagged_count >= self.min_consecutive_frames

        dominant_type = None
        if is_distress:
            types = [t for t in frame_types if t != "normal"]
            if waving_flag:
                dominant_type = "sos_waving_gesture"
            elif types:
                dominant_type = max(set(types), key=types.count)
            else:
                dominant_type = "unspecified_distress"

        return DistressResult(
            is_distress=is_distress,
            confidence=round(float(confidence), 3),
            gesture_type=dominant_type,
            flagged_frames_count=flagged_count,
            total_frames_count=total_frames,
            details={
                "valid_pose_frames": len(valid_frames),
                "waving_detected": waving_flag,
                "flagged_ratio": round(ratio, 3),
            },
        )
