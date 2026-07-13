"""SafetyChain — Scripted Demo Runner

Orchestrates the full demo:
1. Generates synthetic test videos (if not present)
2. Starts the dashboard server
3. Runs Scenario A: Vehicle Break-in (FullThink path)
4. Runs Scenario B: School Intrusion (ZeroThink path)
5. Shows the contrast on the dashboard

Implementation plan ref: Phase 9
"""

import os
import sys
import time
import threading
import webbrowser
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetychain.config import SafetyChainConfig
from safetychain.main import SafetyChainPipeline
from safetychain.dashboard.server import app, manager, alert_to_dict, broadcast_alert
import uvicorn


def generate_videos_if_needed(demo_dir: str):
    """Generate test videos if they don't exist."""
    videos_dir = os.path.join(demo_dir, "videos")
    video_a = os.path.join(videos_dir, "scenario_a_vehicle_breakin.mp4")

    if not os.path.exists(video_a):
        print("🎬 Generating synthetic test videos...")
        from safetychain.demo.generate_test_video import generate_all_videos
        generate_all_videos(videos_dir)
    else:
        print("✅ Test videos already exist")


def run_demo_scenarios(pipeline: SafetyChainPipeline, demo_dir: str):
    """Run both demo scenarios through the pipeline."""
    videos_dir = os.path.join(demo_dir, "videos")
    loop = asyncio.new_event_loop()

    print()
    print("=" * 60)
    print("⛓️  SafetyChain Demo — Running Scenarios")
    print("=" * 60)

    # ═══ Scenario A: Vehicle Break-in (FullThink) ═══
    print()
    print("━" * 50)
    print("📹 SCENARIO A: Vehicle Break-in")
    print("   Zone: Parking Lot - Zone A")
    print("   Time: 03:42 AM (after hours)")
    print("   Expected: FullThink → 5-step reasoning → ALERT")
    print("━" * 50)

    video_a = os.path.join(videos_dir, "scenario_a_vehicle_breakin.mp4")
    if os.path.exists(video_a):
        start = time.time()
        alerts_a = pipeline.process_video(
            video_a, zone_id="zone-a", camera_id="cam-a1", max_frames=90
        )
        elapsed = time.time() - start

        print(f"   ✅ Processed in {elapsed:.1f}s")
        print(f"   📊 Alerts generated: {len(alerts_a)}")
        for alert in alerts_a:
            print(f"      → [{alert.severity}] {alert.title} "
                  f"({alert.confidence:.0%}, {alert.verdict.reasoning_strategy})")

            # Broadcast to dashboard via WebSocket
            loop.run_until_complete(broadcast_alert(alert))

    # Brief pause between scenarios
    time.sleep(2)

    # ═══ Scenario B: School Intrusion (ZeroThink) ═══
    print()
    print("━" * 50)
    print("📹 SCENARIO B: School Intrusion")
    print("   Zone: School Perimeter - Zone B")
    print("   Time: 10:15 AM (during school hours)")
    print("   Expected: ZeroThink → instant EMERGENCY")
    print("━" * 50)

    video_b = os.path.join(videos_dir, "scenario_b_school_intrusion.mp4")
    if os.path.exists(video_b):
        start = time.time()
        alerts_b = pipeline.process_video(
            video_b, zone_id="zone-b", camera_id="cam-b1", max_frames=60
        )
        elapsed = time.time() - start

        print(f"   ✅ Processed in {elapsed:.1f}s")
        print(f"   📊 Alerts generated: {len(alerts_b)}")
        for alert in alerts_b:
            print(f"      → [{alert.severity}] {alert.title} "
                  f"({alert.confidence:.0%}, {alert.verdict.reasoning_strategy})")

            # Broadcast to dashboard
            loop.run_until_complete(broadcast_alert(alert))

    # ═══ Summary ═══
    print()
    print("=" * 60)
    print("📊 DEMO SUMMARY")
    print("=" * 60)
    stats = pipeline.alert_manager.get_stats()
    print(f"   Total alerts: {stats['total_alerts']}")
    print(f"   Active:       {stats['active_alerts']}")
    print()
    print("🌐 Dashboard is live at http://localhost:8000")
    print("   • View both scenarios side-by-side")
    print("   • Click alerts to see reasoning chains")
    print("   • Try the TP/FP feedback buttons")
    print()
    print("Press Ctrl+C to stop the server.")

    loop.close()


def main():
    """Main demo entry point."""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ⛓️  SafetyChain — AI Public Safety Demo            ║")
    print("║  5-Stage Chain-of-Thought Verification Pipeline     ║")
    print("║  Powered by Gemma + YOLOv8                          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    safetychain_dir = os.path.join(base_dir, "safetychain")
    demo_dir = os.path.join(safetychain_dir, "demo")

    # Step 1: Generate test videos
    generate_videos_if_needed(demo_dir)

    # Step 2: Initialize pipeline
    print()
    print("🔧 Initializing SafetyChain pipeline...")
    config = SafetyChainConfig()
    config.DEMO_MODE = True
    config.FRAME_SKIP = 3  # Process every 3rd frame for speed

    pipeline = SafetyChainPipeline(config, base_dir=safetychain_dir)

    # Step 3: Set up dashboard server
    from safetychain.dashboard import server
    server.pipeline = pipeline

    # Step 4: Start server in background thread
    print("🌐 Starting dashboard server...")

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(
            app, host="0.0.0.0", port=8000, log_level="warning"
        ),
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)  # Wait for server to start

    # Step 5: Open browser
    print("🌐 Opening dashboard in browser...")
    webbrowser.open("http://localhost:8000")
    time.sleep(2)

    # Step 6: Run scenarios
    run_demo_scenarios(pipeline, demo_dir)

    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        pipeline.shutdown()


if __name__ == "__main__":
    main()
