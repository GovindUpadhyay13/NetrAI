"""
main.py
Central orchestrator for the Women's-Safety-Focused Surveillance System.

Run modes:
    1. Demo Mode:
       python main.py demo
       Runs 2+ camera feeds concurrently against the full pipeline:
       - Camera 1 (CAM-WEST-PLAZA-01): AnomalyCLIP + MediaPipe distress gesture -> Gemini 2.5 VLM Reasoning -> Alert Dispatch -> Subject Qdrant Enrollment.
       - Camera 2 (CAM-EAST-CORRIDOR-04): Background surveillance feed scanning for cross-camera Re-ID match of the flagged subject -> reid_match event.
       - Live Trace Dashboard: Runs concurrently on http://localhost:8000.

    2. Live Mode:
       python main.py live --video path/to/video.mp4 --camera-id CAM-01
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from bus.publisher import EventBus
from bus.consumers.trace_logger import TraceLoggerConsumer
from bus.consumers.reid_matcher import ReIDMatcherConsumer
from bus.flow_runner import SurveillancePipelineRunner
from reid.embed import SubjectEmbedder
from reid.gallery import QdrantReIDGallery
from ingestion.video_reader import VideoReader
from trace.dashboard import run_dashboard
from detection.gesture.run_standalone import generate_synthetic_distress_clip


def run_demo_mode(port: int = 8000, skip_browser: bool = False):
    print("\n" + "=" * 65)
    print(" 🛡️  NETRAI: WOMEN'S-SAFETY SURVEILLANCE SYSTEM (DEMO MODE)  🛡️")
    print("=" * 65)
    print(f" Dashboard URL: http://localhost:{port}/")
    print(f" Initializing Event Bus, Trace Store, AI Models & Consumers...")

    # 1. Start FastAPI Trace Dashboard in background thread
    dashboard_thread = threading.Thread(
        target=run_dashboard,
        kwargs={"host": "127.0.0.1", "port": port},
        daemon=True,
    )
    dashboard_thread.start()
    time.sleep(1.2)  # Give dashboard a moment to bind

    # 2. Setup Bus and Consumers
    bus = EventBus()
    db_path = "trace.db"
    trace_logger = TraceLoggerConsumer(bus=bus, db_path=db_path)

    # 3. Setup Shared Vision & Re-ID Backbone
    runner = SurveillancePipelineRunner(event_bus=bus, db_path=db_path)
    gallery = QdrantReIDGallery()
    subject_embedder = SubjectEmbedder(anomaly_clip_wrapper=runner.anomaly_scorer.model)
    reid_matcher = ReIDMatcherConsumer(bus=bus, embedder=subject_embedder, gallery=gallery)

    # 4. Prepare Camera 1 Incident Video (Distress / Anomaly Scene)
    demo_cam1_video = "outputs/demo_cam1_incident.mp4"
    if not os.path.exists(demo_cam1_video):
        print("\n[Prep] Generating synthetic incident video for Camera 1...")
        generate_synthetic_distress_clip(demo_cam1_video, fps=20, duration_sec=4)

    print("\n>>> SIMULATING CONCURRENT MULTI-CAMERA SURVEILLANCE NETWORK <<<")

    # Camera 1 Pipeline execution: Hauz Khas Village Main Gate
    cam1_id = "CAM-SD-01"
    print(f"\n[CAMERA 1] Streaming South Delhi Node: {cam1_id}")
    incident_result = runner.process_video_incident(
        video_path=demo_cam1_video,
        camera_id=cam1_id,
        anomaly_threshold=0.15,
    )
    inc_id = incident_result["incident_id"]

    # Enroll subject from Camera 1 into Re-ID Gallery
    print(f"\n[Re-ID] Indexing flagged subject from {cam1_id} into Qdrant Gallery...")
    with VideoReader(demo_cam1_video) as reader:
        cam1_frames, _ = reader.read_all_frames()
    reid_matcher.index_incident_subject(incident_id=inc_id, camera_id=cam1_id, frames_bgr=cam1_frames)

    time.sleep(1.5)

    # Camera 2 Feed: Subject reappears at Deer Park Lake Trail (Waypoint 2)
    cam2_id = "CAM-SD-08"
    print(f"\n[CAMERA 2] Scanning live egress feed: {cam2_id}...")
    cam2_frame = cam1_frames[len(cam1_frames) // 2]  # Identical subject walking in corridor
    matches2 = reid_matcher.scan_camera_feed_for_matches(camera_id=cam2_id, frame_bgr=cam2_frame)

    time.sleep(1.2)

    # Camera 3 Feed: Subject tracked to Green Park Market corridor (Waypoint 3 - Active Position)
    cam3_id = "CAM-SD-04"
    print(f"\n[CAMERA 3] Scanning transit corridor: {cam3_id}...")
    cam3_frame = cam1_frames[len(cam1_frames) // 3]  # Subject spotted in transit
    matches3 = reid_matcher.scan_camera_feed_for_matches(camera_id=cam3_id, frame_bgr=cam3_frame)

    print("\n" + "=" * 65)
    print(" ✅  DEMO RUN COMPLETED SUCCESSFULLY!")
    print(f" Summary of Live Incident: {inc_id}")
    print(f" - Primary Camera:     {cam1_id} (Anomaly + SOS Gesture Flagged)")
    print(f" - VLM Reasoner:       Gemini 2.5 Flash Report Attached")
    print(f" - Department Alert:   {incident_result['recommended_department']} (Severity: {incident_result['severity'].upper()})")
    print(f" - Waypoint 2 Sighting: Sighted on {cam2_id} ({len(matches2)} Re-ID match event logged)")
    print(f" - Waypoint 3 Sighting: Sighted on {cam3_id} ({len(matches3)} Re-ID match event logged)")
    print(f"\n 🌐 View Cross-Camera Suspect Trajectory on Dashboard:")
    print(f"    http://localhost:{port}/")
    print("=" * 65)
    print(f"\n[Dashboard Server] Serving live at http://localhost:{port}/ (Press Ctrl+C to stop).")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Dashboard Server] Exiting demo.")


def run_live_mode(video_path: str, camera_id: str):
    print(f"Running Live Ingestion on {camera_id}: {video_path}")
    runner = SurveillancePipelineRunner()
    runner.process_video_incident(video_path=video_path, camera_id=camera_id)


def main():
    parser = argparse.ArgumentParser(description="NetrAI Surveillance System")
    subparsers = parser.add_subparsers(dest="command")

    # Demo parser
    demo_p = subparsers.add_parser("demo", help="Run multi-camera demo with dashboard")
    demo_p.add_argument("--port", type=int, default=8000, help="Dashboard port")

    # Live parser
    live_p = subparsers.add_parser("live", help="Run on input video feed")
    live_p.add_argument("--video", type=str, required=True, help="Video file path")
    live_p.add_argument("--camera-id", type=str, default="CAM-01", help="Camera identifier")

    args = parser.parse_args()

    if args.command == "demo" or args.command is None:
        port = getattr(args, "port", 8000)
        run_demo_mode(port=port)
    elif args.command == "live":
        run_live_mode(args.video, args.camera_id)


if __name__ == "__main__":
    main()
