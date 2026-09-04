"""
detection/anomaly_clip/scorer.py
Evaluates videos using AnomalyCLIP to produce continuous per-frame anomaly scores
and extract flagged time windows exceeding a configurable threshold.

Assumptions:
- Input is a path to a video file or an instantiated VideoReader.
- Uses AnomalyCLIPWrapper with the UCF-Crime checkpoint.
- Frames are converted to RGB PIL images for CLIP feature extraction.
- Output contains per-frame score curves and structured list of flagged intervals.
"""

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ingestion.video_reader import VideoReader
from .model_loader import AnomalyCLIPWrapper


@dataclass
class FlaggedWindow:
    window_id: int
    start_time_sec: float
    end_time_sec: float
    start_frame: int
    end_frame: int
    peak_score: float
    mean_score: float
    anomaly_type: str

    def to_dict(self) -> Dict:
        return asdict(self)


class VideoAnomalyScorer:
    def __init__(
        self,
        model_wrapper: Optional[AnomalyCLIPWrapper] = None,
        checkpoint_path: Optional[str] = None,
        default_threshold: float = 0.45,
        min_window_duration_sec: float = 0.2,
        smoothing_window: int = 5,
    ):
        if model_wrapper is None:
            self.model = AnomalyCLIPWrapper(checkpoint_path=checkpoint_path)
        else:
            self.model = model_wrapper

        self.threshold = default_threshold
        self.min_window_duration_sec = min_window_duration_sec
        self.smoothing_window = smoothing_window

    def score_features(
        self, features: torch.Tensor, timestamps: List[float]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Runs AnomalyCLIP forward pass given pre-extracted CLIP visual features (N, 512).
        Returns:
            scores: (N,) numpy array of anomaly probabilities [0.0, 1.0]
            top_classes: (N,) list of predicted anomaly types per frame
        """
        num_frames = features.shape[0]
        if num_frames == 0:
            return np.array([]), []

        device = self.model.device
        features = features.to(device)

        # AnomalyCLIP temporal model expects 32 segments * 16 frames = 512 frames
        target_len = 512
        if num_frames != target_len:
            # Interpolate features along temporal dimension
            feat_perm = features.unsqueeze(0).permute(0, 2, 1)  # (1, 512, N)
            interp_feat = F.interpolate(feat_perm, size=target_len, mode="linear", align_corners=False)
            input_feats = interp_feat.permute(0, 2, 1).unsqueeze(0)  # (1, 1, 512, 512)
        else:
            input_feats = features.unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

        with torch.no_grad():
            similarity, raw_scores = self.model.model(
                input_feats,
                labels=None,
                ncentroid=self.model.ncentroid,
                test_mode=True,
            )

            # Map raw_scores (512 or 8192) to probabilities using sigmoid / min-max scaling
            scores_tensor = raw_scores.float().cpu()
            # Normalize to 0-1 range cleanly
            s_min, s_max = scores_tensor.min(), scores_tensor.max()
            if s_max - s_min > 1e-6:
                norm_scores = (scores_tensor - s_min) / (s_max - s_min)
            else:
                norm_scores = torch.sigmoid(scores_tensor)

            # Interpolate scores back to original frame length
            norm_scores_2d = norm_scores.view(1, 1, -1)
            final_scores = F.interpolate(norm_scores_2d, size=num_frames, mode="linear", align_corners=False)
            final_scores = final_scores.squeeze().numpy()
            if final_scores.ndim == 0:
                final_scores = np.array([float(final_scores)])
            elif self.smoothing_window > 1 and len(final_scores) >= self.smoothing_window:
                # Apply moving average filter
                kernel = np.ones(self.smoothing_window) / self.smoothing_window
                final_scores = np.convolve(final_scores, kernel, mode="same")

            # Identify top predicted anomaly class from text projection similarity
            sim_tensor = similarity.float().cpu()
            if sim_tensor.shape[0] != num_frames:
                sim_2d = sim_tensor.unsqueeze(0).permute(0, 2, 1)  # (1, C, T)
                interp_sim = F.interpolate(sim_2d, size=num_frames, mode="linear", align_corners=False)
                final_sim = interp_sim.squeeze(0).permute(1, 0)  # (num_frames, C)
            else:
                final_sim = sim_tensor

            top_class_indices = torch.argmax(final_sim, dim=-1).numpy()
            top_classes = [self.model.abnormal_classes[idx % len(self.model.abnormal_classes)] for idx in top_class_indices]

        return final_scores, top_classes

    def score_video(
        self,
        video_path: str,
        threshold: Optional[float] = None,
        target_fps: Optional[float] = 5.0,
    ) -> Dict:
        """
        Reads a video file, scores every frame, and extracts flagged time windows.
        """
        th = threshold if threshold is not None else self.threshold

        with VideoReader(video_path) as reader:
            metadata = reader.get_metadata()
            frames_bgr, timestamps = reader.read_all_frames(target_fps=target_fps)

        if not frames_bgr:
            return {
                "metadata": asdict(metadata),
                "timestamps": [],
                "scores": [],
                "flagged_windows": [],
            }

        # Convert to PIL RGB
        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]

        # Extract features
        features = self.model.extract_image_features(pil_images, batch_size=32)

        # Compute per-frame scores
        scores, top_classes = self.score_features(features, timestamps)

        # Detect contiguous windows above threshold
        flagged_windows = self._extract_windows(scores, timestamps, top_classes, th)

        return {
            "metadata": asdict(metadata),
            "timestamps": [round(t, 3) for t in timestamps],
            "scores": [round(float(s), 4) for s in scores],
            "threshold": th,
            "flagged_windows": [w.to_dict() for w in flagged_windows],
        }

    def _extract_windows(
        self,
        scores: np.ndarray,
        timestamps: List[float],
        top_classes: List[str],
        threshold: float,
    ) -> List[FlaggedWindow]:
        windows: List[FlaggedWindow] = []
        in_window = False
        start_idx = 0

        for i, score in enumerate(scores):
            if score >= threshold and not in_window:
                in_window = True
                start_idx = i
            elif score < threshold and in_window:
                in_window = False
                end_idx = i - 1
                duration = timestamps[end_idx] - timestamps[start_idx]
                if duration >= self.min_window_duration_sec:
                    window_scores = scores[start_idx : end_idx + 1]
                    window_classes = top_classes[start_idx : end_idx + 1]
                    # dominant class in window
                    dominant_class = max(set(window_classes), key=window_classes.count) if window_classes else "Unknown Anomaly"

                    windows.append(
                        FlaggedWindow(
                            window_id=len(windows) + 1,
                            start_time_sec=round(timestamps[start_idx], 2),
                            end_time_sec=round(timestamps[end_idx], 2),
                            start_frame=start_idx,
                            end_frame=end_idx,
                            peak_score=round(float(np.max(window_scores)), 4),
                            mean_score=round(float(np.mean(window_scores)), 4),
                            anomaly_type=dominant_class,
                        )
                    )

        # If ended while still above threshold
        if in_window:
            end_idx = len(scores) - 1
            duration = timestamps[end_idx] - timestamps[start_idx]
            if duration >= self.min_window_duration_sec:
                window_scores = scores[start_idx : end_idx + 1]
                window_classes = top_classes[start_idx : end_idx + 1]
                dominant_class = max(set(window_classes), key=window_classes.count) if window_classes else "Unknown Anomaly"
                windows.append(
                    FlaggedWindow(
                        window_id=len(windows) + 1,
                        start_time_sec=round(timestamps[start_idx], 2),
                        end_time_sec=round(timestamps[end_idx], 2),
                        start_frame=start_idx,
                        end_frame=end_idx,
                        peak_score=round(float(np.max(window_scores)), 4),
                        mean_score=round(float(np.mean(window_scores)), 4),
                        anomaly_type=dominant_class,
                    )
                )

        return windows
