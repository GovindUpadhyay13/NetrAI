"""
tests/test_milestone4.py
Verification test for Milestone 4: Event Bus + Trace Logger + SQLite trace history.
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bus.flow_runner import SurveillancePipelineRunner
from trace.db import get_incident_trace, get_all_incidents


def main():
    print("Testing Milestone 4: Full flow detect -> analyze -> Event Bus -> Trace DB...")
    runner = SurveillancePipelineRunner(db_path="trace.db")

    test_video = "outputs/milestone2/synthetic_distress.mp4"
    if not os.path.exists(test_video):
        from detection.gesture.run_standalone import generate_synthetic_distress_clip
        test_video = generate_synthetic_distress_clip(test_video)

    res = runner.process_video_incident(test_video, camera_id="CAM-NORTH-PARKING-03")
    inc_id = res["incident_id"]

    traces = get_incident_trace(inc_id, db_path="trace.db")
    print(f"\n[Trace Verification] Found {len(traces)} stages for incident: {inc_id}")
    for t in traces:
        print(f"  * Stage: {t['stage']:<20} | Score: {t['anomaly_score']:.3f} | Distress: {t['distress_gesture']} | Sev: {t['severity']}")
        if t["vlm_report"]:
            print(f"    Report: {t['vlm_report'][:90]}...")

    assert len(traces) >= 3, f"Expected at least 3 stages, found {len(traces)}"
    print("\nSUCCESS: Milestone 4 Event Bus + Trace Logger verified.")


if __name__ == "__main__":
    main()
