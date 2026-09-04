"""
detection/anomaly_clip/run_standalone.py
Standalone CLI entry point for Milestone 1: Ingestion + AnomalyCLIP scoring.

Usage:
    python -m detection.anomaly_clip.run_standalone --video path/to/video.mp4 --threshold 0.45
    python -m detection.anomaly_clip.run_standalone --demo
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detection.anomaly_clip.scorer import VideoAnomalyScorer
from detection.anomaly_clip.visualizer import plot_anomaly_curve


def create_demo_video(output_path: str = "demo_surveillance.mp4", duration_sec: int = 6, fps: int = 20) -> str:
    """Generates a synthetic surveillance video for standalone verification."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        t = i / fps
        # Surveillance corridor background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Floor lines
        cv2.line(frame, (0, 400), (640, 400), (80, 80, 80), 2)
        cv2.line(frame, (100, 480), (250, 400), (70, 70, 70), 2)
        cv2.line(frame, (540, 480), (390, 400), (70, 70, 70), 2)

        # Subject walking normally from t=0 to t=2.5
        x = int(120 + (t / duration_sec) * 350)
        y = 320

        # Person 1 (walking)
        cv2.circle(frame, (x, y - 50), 18, (200, 200, 200), -1)  # head
        cv2.line(frame, (x, y - 32), (x, y + 30), (200, 180, 100), 6)  # torso
        cv2.line(frame, (x, y + 30), (x - 15, y + 70), (180, 160, 80), 5)  # leg 1
        cv2.line(frame, (x, y + 30), (x + 15, y + 70), (180, 160, 80), 5)  # leg 2

        # Second person appears around t=2.2 - t=4.5 (harassment/stalking/assault gesture pattern)
        if 2.0 <= t <= 4.5:
            x2 = x - 40
            y2 = y
            cv2.circle(frame, (x2, y2 - 50), 18, (80, 80, 220), -1)  # head
            cv2.line(frame, (x2, y2 - 32), (x2, y2 + 30), (50, 50, 200), 6)  # torso
            # Raised arms / aggressive posture
            cv2.line(frame, (x2, y2 - 20), (x, y - 20), (50, 50, 200), 4)
            # Distress text on frame
            cv2.putText(frame, "SIMULATED INCIDENT WINDOW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Camera watermark
        cv2.putText(frame, f"CAM-01 | T={t:.2f}s", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        writer.write(frame)

    writer.release()
    print(f"[Demo] Created synthetic test video at: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AnomalyCLIP Video Scoring Standalone Runner")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--threshold", type=float, default=0.45, help="Anomaly detection threshold (default: 0.45)")
    parser.add_argument("--output-dir", type=str, default="outputs/milestone1", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Run with auto-generated demo surveillance video")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_path = args.video
    if video_path is None or args.demo:
        demo_path = os.path.join(args.output_dir, "synthetic_surveillance.mp4")
        video_path = create_demo_video(demo_path)

    print(f"\n=======================================================")
    print(f" AnomalyCLIP Surveillance Anomaly Detection (Milestone 1)")
    print(f"=======================================================")
    print(f"Target Video: {video_path}")
    print(f"Threshold:    {args.threshold}")
    print(f"Output Dir:   {args.output_dir}")

    scorer = VideoAnomalyScorer(
        checkpoint_path=args.checkpoint,
        default_threshold=args.threshold,
    )

    print("\nRunning inference...")
    result = scorer.score_video(video_path, threshold=args.threshold)

    # Save JSON report
    json_path = os.path.join(args.output_dir, "flagged_windows.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Saved JSON] -> {json_path}")

    # Plot anomaly curve
    plot_path = os.path.join(args.output_dir, "anomaly_curve.png")
    plot_anomaly_curve(result, plot_path)
    print(f"[Saved Plot] -> {plot_path}")

    # Print summary
    flagged = result["flagged_windows"]
    print(f"\nTotal Flagged Windows: {len(flagged)}")
    for w in flagged:
        print(f"  * Window #{w['window_id']}: [{w['start_time_sec']}s - {w['end_time_sec']}s] | "
              f"Peak Score: {w['peak_score']:.4f} | Anomaly: {w['anomaly_type']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
