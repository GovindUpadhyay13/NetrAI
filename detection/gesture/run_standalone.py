"""
detection/gesture/run_standalone.py
Standalone runner for Milestone 2: MediaPipe pose extraction -> distress gesture classifier.

Usage:
    python -m detection.gesture.run_standalone --video path/to/clip.mp4
    python -m detection.gesture.run_standalone --demo
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingestion.video_reader import VideoReader
from detection.gesture.pose_extractor import PoseExtractor
from detection.gesture.distress_classifier import DistressClassifier


def generate_synthetic_distress_clip(output_path: str = "demo_distress_clip.mp4", fps: int = 20, duration_sec: int = 3) -> str:
    """Creates a synthetic clip with a person performing a high-arm SOS waving gesture."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = fps * duration_sec
    for i in range(total_frames):
        t = i / fps
        frame = np.full((height, width, 3), 235, dtype=np.uint8)

        # Realistic human body stick-figure with head, torso, shoulders, and waving arms
        cx, cy = 320, 240

        # Head
        cv2.circle(frame, (cx, cy - 80), 28, (40, 40, 40), -1)
        # Face skin tone
        cv2.circle(frame, (cx, cy - 80), 22, (180, 200, 240), -1)

        # Torso
        cv2.line(frame, (cx, cy - 50), (cx, cy + 60), (30, 80, 180), 16)

        # Shoulders
        cv2.line(frame, (cx - 45, cy - 40), (cx + 45, cy - 40), (30, 80, 180), 12)

        # High Waving Arms (Both hands raised high above head with sinusoidal waving motion)
        wave_offset_l = int(np.sin(t * 10.0) * 35)
        wave_offset_r = int(np.cos(t * 10.0) * 35)

        # Left arm raised
        cv2.line(frame, (cx - 45, cy - 40), (cx - 70, cy - 90), (30, 80, 180), 8)
        cv2.line(frame, (cx - 70, cy - 90), (cx - 90 + wave_offset_l, cy - 140), (30, 80, 180), 8)
        cv2.circle(frame, (cx - 90 + wave_offset_l, cy - 140), 10, (180, 200, 240), -1)

        # Right arm raised
        cv2.line(frame, (cx + 45, cy - 40), (cx + 70, cy - 90), (30, 80, 180), 8)
        cv2.line(frame, (cx + 70, cy - 90), (cx + 90 + wave_offset_r, cy - 140), (30, 80, 180), 8)
        cv2.circle(frame, (cx + 90 + wave_offset_r, cy - 140), 10, (180, 200, 240), -1)

        # Legs
        cv2.line(frame, (cx, cy + 60), (cx - 35, cy + 170), (40, 40, 40), 10)
        cv2.line(frame, (cx, cy + 60), (cx + 35, cy + 170), (40, 40, 40), 10)

        cv2.putText(frame, f"SOS GESTURE DEMO | T={t:.2f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        writer.write(frame)

    writer.release()
    print(f"[Demo] Created synthetic distress gesture clip at: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Distress Gesture Classifier Standalone Runner")
    parser.add_argument("--video", type=str, default=None, help="Path to input video clip")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--demo", action="store_true", help="Generate and test on synthetic distress clip")

    args = parser.parse_args()

    video_path = args.video
    if video_path is None or args.demo:
        demo_dir = "outputs/milestone2"
        os.makedirs(demo_dir, exist_ok=True)
        demo_file = os.path.join(demo_dir, "synthetic_distress.mp4")
        video_path = generate_synthetic_distress_clip(demo_file)

    print(f"\n=======================================================")
    print(f" MediaPipe Distress Gesture Branch (Milestone 2)")
    print(f"=======================================================")
    print(f"Target Video: {video_path}")
    print(f"Threshold:    {args.threshold}")

    with VideoReader(video_path) as reader:
        frames, timestamps = reader.read_all_frames()

    print(f"Read {len(frames)} frames. Extracting pose landmarks...")
    with PoseExtractor(static_image_mode=False) as extractor:
        pose_frames = extractor.process_video_frames(frames, timestamps)

    classifier = DistressClassifier(confidence_threshold=args.threshold)
    result = classifier.classify_clip(pose_frames)

    print(f"\n--- CLASSIFICATION RESULT ---")
    print(f"Distress Flagged:    {result.is_distress}")
    print(f"Confidence:          {result.confidence:.3f}")
    print(f"Gesture Type:        {result.gesture_type}")
    print(f"Flagged Frames:      {result.flagged_frames_count} / {result.total_frames_count}")
    print(f"Details:             {result.details}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
