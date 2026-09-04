"""
tests/test_milestone6.py
Verification test for Milestone 6: Cross-camera Re-ID.
"""

import sys
import os
import cv2
import numpy as np
from uuid import uuid4

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bus.publisher import EventBus
from bus.consumers.trace_logger import TraceLoggerConsumer
from bus.consumers.reid_matcher import ReIDMatcherConsumer
from trace.db import get_incident_trace


def create_subject_frame(color=(50, 120, 220), tag="CAM-A") -> np.ndarray:
    """Creates a frame containing a distinctive colored figure."""
    frame = np.full((480, 640, 3), 220, dtype=np.uint8)
    cx, cy = 320, 240
    # Head
    cv2.circle(frame, (cx, cy - 80), 25, (40, 40, 40), -1)
    # Distinctive colored jacket / torso
    cv2.rectangle(frame, (cx - 40, cy - 50), (cx + 40, cy + 80), color, -1)
    # Legs
    cv2.rectangle(frame, (cx - 30, cy + 80), (cx - 10, cy + 180), (30, 30, 30), -1)
    cv2.rectangle(frame, (cx + 10, cy + 80), (cx + 30, cy + 180), (30, 30, 30), -1)
    cv2.putText(frame, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return frame


def main():
    print("Testing Milestone 6: Cross-Camera Re-ID...")
    bus = EventBus()
    trace_logger = TraceLoggerConsumer(bus=bus, db_path="trace.db")
    reid_consumer = ReIDMatcherConsumer(bus=bus, similarity_threshold=0.60)

    incident_id = str(uuid4())

    # Camera 1 (CAM-ENTRY-A): Subject flagged in incident
    print(f"\n[1] Enrolling subject from Camera-1 (Incident: {incident_id[:8]})...")
    cam1_frame = create_subject_frame(color=(30, 70, 200), tag="CAM-ENTRY-A")
    reid_consumer.index_incident_subject(
        incident_id=incident_id,
        camera_id="CAM-ENTRY-A",
        frames_bgr=[cam1_frame],
    )

    # Camera 2 (CAM-EXIT-B): Same subject appears on secondary camera feed
    print("\n[2] Scanning Camera-2 feed for subject re-appearance...")
    cam2_frame = create_subject_frame(color=(30, 70, 200), tag="CAM-EXIT-B")
    matches = reid_consumer.scan_camera_feed_for_matches(
        camera_id="CAM-EXIT-B",
        frame_bgr=cam2_frame,
    )

    assert len(matches) > 0, "Expected a Cross-Camera Re-ID match!"
    match = matches[0]
    print(f"[3] Match Verified: Origin Incident {match['origin_incident_id'][:8]} | Similarity: {match['score']}")
    assert match["origin_incident_id"] == incident_id, "Match must link to originating incident ID!"

    # Verify 'reid_match' logged in Trace DB
    traces = get_incident_trace(incident_id, db_path="trace.db")
    stages = [t["stage"] for t in traces]
    print(f"[4] Trace Stages in SQLite: {stages}")
    assert "reid_match" in stages, "Expected 'reid_match' recorded in trace database!"

    print("\nSUCCESS: Milestone 6 Cross-Camera Re-ID verified.")


if __name__ == "__main__":
    main()
