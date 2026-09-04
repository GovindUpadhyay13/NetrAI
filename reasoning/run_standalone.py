"""
reasoning/run_standalone.py
Standalone runner for Milestone 3: 3x3 Grid Builder + Gemini Analyzer.

Usage:
    python -m reasoning.run_standalone --video path/to/clip.mp4 --anomaly-prior "Assault" --distress
    python -m reasoning.run_standalone --demo
"""

import argparse
import json
import os
import sys

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from reasoning.frame_grid import FrameGridBuilder
from reasoning.gemini_analyzer import GeminiIncidentAnalyzer
from detection.gesture.run_standalone import generate_synthetic_distress_clip


def main():
    parser = argparse.ArgumentParser(description="Milestone 3: 3x3 Grid Builder + Gemini Analyzer Standalone")
    parser.add_argument("--video", type=str, default=None, help="Path to video clip")
    parser.add_argument("--anomaly-prior", type=str, default="Stalking / Harassment Pattern", help="Anomaly type prior")
    parser.add_argument("--anomaly-score", type=float, default=0.82, help="Anomaly score")
    parser.add_argument("--distress", action="store_true", default=True, help="Flag distress gesture as present")
    parser.add_argument("--gesture-type", type=str, default="both_arms_raised_sos", help="Distress gesture type")
    parser.add_argument("--gesture-confidence", type=float, default=0.91, help="Distress gesture confidence")
    parser.add_argument("--output-dir", type=str, default="outputs/milestone3", help="Output directory")
    parser.add_argument("--demo", action="store_true", help="Generate and use demo clip")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    video_path = args.video
    if video_path is None or args.demo:
        demo_clip = os.path.join(args.output_dir, "demo_incident_clip.mp4")
        video_path = generate_synthetic_distress_clip(demo_clip)

    print(f"\n=======================================================")
    print(f" 3x3 Temporal Grid Builder + Gemini Analyzer (Milestone 3)")
    print(f"=======================================================")
    print(f"Target Clip:        {video_path}")
    print(f"AnomalyCLIP Prior:  {args.anomaly_prior} (Score: {args.anomaly_score})")
    print(f"Distress Gesture:   {args.distress} ({args.gesture_type}, Conf: {args.gesture_confidence})")

    # 1. Build 3x3 Grid
    grid_builder = FrameGridBuilder()
    grid_img_path = os.path.join(args.output_dir, "incident_grid_3x3.png")
    grid_image = grid_builder.build_grid_from_video(video_path, output_path=grid_img_path)
    print(f"\n[1] Built 3x3 temporal grid -> {grid_img_path}")

    # 2. Run Gemini VLM Analyzer
    analyzer = GeminiIncidentAnalyzer()
    print(f"[2] Invoking Gemini Incident Analyzer...")
    report = analyzer.analyze_incident(
        grid_image=grid_image,
        camera_id="CAM-SOUTH-CORRIDOR-04",
        start_sec=1.5,
        end_sec=4.5,
        anomaly_type_prior=args.anomaly_prior,
        anomaly_score=args.anomaly_score,
        distress_gesture_flag=args.distress,
        distress_gesture_type=args.gesture_type,
        gesture_confidence=args.gesture_confidence,
    )

    report_dict = report.to_dict()
    report_json_path = os.path.join(args.output_dir, "incident_report.json")
    with open(report_json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\n--- STRUCTURED INCIDENT REPORT ---")
    print(f"Description:            {report.incident_description}")
    print(f"Severity:               {report.severity.upper()}")
    print(f"Recommended Department: {report.recommended_department}")
    print(f"Confidence:             {report.confidence}")
    print(f"Threat Indicators:      {report.threat_indicators}")
    print(f"Key Observations:")
    for obs in report.key_observations:
        print(f"  - {obs}")
    print(f"\n[Saved Report JSON] -> {report_json_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
