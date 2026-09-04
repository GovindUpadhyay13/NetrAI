"""
ingestion/video_reader.py
Video ingestion module for reading video files and streams frame-by-frame or in batches,
extracting frame timestamps, resolution, FPS, and saving sub-clips.

Assumptions:
- Input is a valid local video file path (mp4, avi, mkv, mov) or RTSP/webcam stream string/int.
- Uses OpenCV (cv2) for cross-platform video decoding.
- Timestamps are computed based on frame index and FPS.
"""

import os
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    source_path: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int


class VideoReader:
    def __init__(self, source_path: str):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Video file not found: {source_path}")
        self.source_path = source_path
        self.cap = cv2.VideoCapture(source_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 25.0  # fallback standard fps

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.total_frames > 0 else 0.0

    def get_metadata(self) -> VideoMetadata:
        return VideoMetadata(
            source_path=self.source_path,
            fps=self.fps,
            total_frames=self.total_frames,
            duration_sec=self.duration_sec,
            width=self.width,
            height=self.height,
        )

    def read_all_frames(self, target_fps: Optional[float] = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Reads all frames (or sampled to target_fps) into memory.
        Returns: (frames_bgr_list, timestamps_sec_list)
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        timestamps = []

        step = 1
        if target_fps is not None and target_fps < self.fps:
            step = max(1, int(round(self.fps / target_fps)))

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                t_sec = frame_idx / self.fps
                frames.append(frame)
                timestamps.append(t_sec)
            frame_idx += 1

        return frames, timestamps

    def stream_batches(self, batch_size: int = 32) -> Generator[Tuple[List[np.ndarray], List[float], List[int]], None, None]:
        """
        Yields batches of (frames, timestamps, frame_indices) without loading entire video into memory.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        batch_frames = []
        batch_times = []
        batch_indices = []

        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            t_sec = frame_idx / self.fps
            batch_frames.append(frame)
            batch_times.append(t_sec)
            batch_indices.append(frame_idx)

            if len(batch_frames) == batch_size:
                yield batch_frames, batch_times, batch_indices
                batch_frames, batch_times, batch_indices = [], [], []

            frame_idx += 1

        if batch_frames:
            yield batch_frames, batch_times, batch_indices

    def extract_clip(self, start_sec: float, end_sec: float, output_path: str) -> str:
        """
        Extracts and writes a sub-clip between start_sec and end_sec.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        start_frame = max(0, int(start_sec * self.fps))
        end_frame = min(self.total_frames, int(end_sec * self.fps))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            writer.write(frame)
            current_frame += 1

        writer.release()
        return output_path

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
