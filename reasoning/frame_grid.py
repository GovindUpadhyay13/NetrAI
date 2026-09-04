"""
reasoning/frame_grid.py
Constructs a 3x3 temporal grid image from 9 uniformly sampled frames across a flagged incident clip.

Assumptions:
- Input is a list of BGR/RGB frames or a VideoReader with start_sec and end_sec.
- Uniformly samples 9 frames across the duration.
- Annotates each sub-frame with relative timestamp / index.
- Outputs a high-resolution 3x3 stitched image for VLM reasoning.
"""

import os
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from ingestion.video_reader import VideoReader


class FrameGridBuilder:
    def __init__(self, tile_size: Tuple[int, int] = (320, 240)):
        self.tile_w, self.tile_h = tile_size

    def sample_9_frames(
        self, frames: List[np.ndarray], timestamps: Optional[List[float]] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Uniformly samples 9 frames across the list."""
        n = len(frames)
        if n == 0:
            raise ValueError("No frames provided to FrameGridBuilder.")

        if n <= 9:
            indices = np.linspace(0, n - 1, 9, dtype=int)
        else:
            indices = np.linspace(0, n - 1, 9, dtype=int)

        sampled_frames = [frames[i] for i in indices]
        if timestamps is not None and len(timestamps) == n:
            sampled_times = [timestamps[i] for i in indices]
        else:
            sampled_times = [float(i) for i in range(9)]

        return sampled_frames, sampled_times

    def build_grid_from_frames(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[float]] = None,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Builds a 3x3 stitched composite image from given frames.
        """
        sampled_frames, sampled_times = self.sample_9_frames(frames, timestamps)

        annotated_tiles = []
        for idx, (frame, t) in enumerate(zip(sampled_frames, sampled_times)):
            tile = cv2.resize(frame, (self.tile_w, self.tile_h))
            # Annotate tile with frame number and timestamp
            header_text = f"T+{t:.2f}s [F{idx + 1}]"
            # Background banner
            cv2.rectangle(tile, (0, 0), (self.tile_w, 24), (20, 20, 20), -1)
            cv2.putText(
                tile,
                header_text,
                (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # Thin border
            cv2.rectangle(tile, (0, 0), (self.tile_w - 1, self.tile_h - 1), (80, 80, 80), 1)
            annotated_tiles.append(tile)

        # Stitch 3 rows of 3 tiles
        row1 = np.hstack(annotated_tiles[0:3])
        row2 = np.hstack(annotated_tiles[3:6])
        row3 = np.hstack(annotated_tiles[6:9])
        grid_bgr = np.vstack([row1, row2, row3])

        # Convert to RGB PIL Image
        grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
        pil_grid = Image.fromarray(grid_rgb)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            pil_grid.save(output_path)

        return pil_grid

    def build_grid_from_video(
        self,
        video_path: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """Extracts frames from video clip window and builds 3x3 grid."""
        with VideoReader(video_path) as reader:
            all_frames, all_times = reader.read_all_frames()

        if not all_frames:
            raise ValueError(f"Could not read frames from {video_path}")

        duration = all_times[-1] if all_times else 0.0
        e_sec = end_sec if end_sec is not None else duration

        clip_frames = []
        clip_times = []
        for f, t in zip(all_frames, all_times):
            if start_sec <= t <= e_sec:
                clip_frames.append(f)
                clip_times.append(t)

        if not clip_frames:
            clip_frames = all_frames
            clip_times = all_times

        return self.build_grid_from_frames(clip_frames, clip_times, output_path=output_path)
