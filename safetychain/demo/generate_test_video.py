"""SafetyChain — Synthetic Test Video Generator

Generates synthetic test videos using OpenCV for the two demo scenarios:
  - Scenario A: Vehicle Break-in (parking lot, nighttime)
  - Scenario B: School Intrusion (school perimeter, daytime)

Implementation plan ref: Phase 9
"""

import os
import cv2
import numpy as np
import random


def generate_parking_lot_video(output_path: str, num_frames: int = 90,
                                 fps: int = 15, width: int = 640,
                                 height: int = 480):
    """Generate a synthetic parking lot break-in scenario.
    
    Simulates: nighttime parking lot with a person crouching near a vehicle,
    holding a slim object. Progressively more suspicious.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_num in range(num_frames):
        # Dark nighttime background
        frame = np.full((height, width, 3), (15, 12, 10), dtype=np.uint8)

        # Add some noise for realism
        noise = np.random.randint(0, 8, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Ground / parking surface
        cv2.rectangle(frame, (0, height - 100), (width, height), (35, 30, 28), -1)

        # Parking lines
        for x in range(50, width, 120):
            cv2.line(frame, (x, height - 100), (x, height), (80, 75, 70), 2)

        # Street lamp glow (top-right area)
        center = (width - 80, 40)
        for r in range(120, 20, -10):
            alpha = max(0, 30 - r // 5)
            overlay = frame.copy()
            cv2.circle(overlay, center, r, (40 + alpha, 35 + alpha, 20 + alpha), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.circle(frame, center, 8, (180, 170, 140), -1)

        # Vehicle (silver sedan) — static
        car_x, car_y = 250, height - 180
        # Car body
        cv2.rectangle(frame, (car_x, car_y + 20), (car_x + 180, car_y + 80), (140, 140, 150), -1)
        # Car roof
        pts_roof = np.array([
            [car_x + 30, car_y + 20],
            [car_x + 50, car_y - 10],
            [car_x + 140, car_y - 10],
            [car_x + 160, car_y + 20]
        ], np.int32)
        cv2.fillPoly(frame, [pts_roof], (130, 130, 140))
        # Windows
        cv2.rectangle(frame, (car_x + 55, car_y - 5), (car_x + 95, car_y + 18), (40, 50, 60), -1)
        cv2.rectangle(frame, (car_x + 100, car_y - 5), (car_x + 135, car_y + 18), (40, 50, 60), -1)
        # Wheels
        cv2.circle(frame, (car_x + 40, car_y + 80), 15, (20, 20, 20), -1)
        cv2.circle(frame, (car_x + 150, car_y + 80), 15, (20, 20, 20), -1)

        # Person (progressively approaching and crouching)
        progress = frame_num / num_frames
        person_x = int(500 - progress * 150)
        person_y = int(height - 200 + progress * 40)  # Crouching down

        # Body
        body_height = int(80 - progress * 25)  # Getting shorter as crouching
        # Dark hoodie
        cv2.rectangle(frame, (person_x - 15, person_y),
                       (person_x + 15, person_y + body_height), (30, 25, 35), -1)
        # Head
        cv2.circle(frame, (person_x, person_y - 5), 12, (120, 95, 80), -1)
        # Hood
        cv2.ellipse(frame, (person_x, person_y - 5), (15, 12), 0, -180, 0,
                     (30, 25, 35), -1)

        # Legs
        cv2.line(frame, (person_x - 5, person_y + body_height),
                 (person_x - 10, person_y + body_height + 30), (25, 25, 30), 4)
        cv2.line(frame, (person_x + 5, person_y + body_height),
                 (person_x + 10, person_y + body_height + 30), (25, 25, 30), 4)

        # Slim metallic object in hand (appears after 40% progress)
        if progress > 0.4:
            tool_x = person_x + 18
            tool_y = person_y + int(body_height * 0.6)
            cv2.line(frame, (tool_x, tool_y), (tool_x + 20, tool_y + 5),
                     (180, 180, 190), 2)

        # Arms reaching toward car (after 60%)
        if progress > 0.6:
            cv2.line(frame, (person_x + 15, person_y + 30),
                     (car_x + 180, car_y + 40), (30, 25, 35), 3)

        # Timestamp overlay
        cv2.putText(frame, "03:42 AM | Parking Lot - Zone A | CAM-A1",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)

        # Add subtle grain
        grain = np.random.randint(-3, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + grain, 0, 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    print(f"Generated: {output_path} ({num_frames} frames)")


def generate_school_intrusion_video(output_path: str, num_frames: int = 60,
                                     fps: int = 15, width: int = 640,
                                     height: int = 480):
    """Generate a synthetic school perimeter intrusion scenario.
    
    Simulates: daytime, a person approaching a school fence.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_num in range(num_frames):
        # Daytime sky gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            ratio = y / height
            b = int(200 - ratio * 120)
            g = int(180 - ratio * 100)
            r = int(140 - ratio * 80)
            frame[y, :] = (max(b, 30), max(g, 40), max(r, 30))

        # Ground
        ground_y = height - 120
        cv2.rectangle(frame, (0, ground_y), (width, height), (50, 90, 45), -1)

        # School building in background
        bldg_x, bldg_y = 100, ground_y - 150
        cv2.rectangle(frame, (bldg_x, bldg_y), (bldg_x + 400, ground_y),
                       (160, 155, 140), -1)
        # Roof
        pts_roof = np.array([
            [bldg_x - 10, bldg_y],
            [bldg_x + 200, bldg_y - 40],
            [bldg_x + 410, bldg_y]
        ], np.int32)
        cv2.fillPoly(frame, [pts_roof], (120, 60, 50))
        # Windows
        for wx in range(bldg_x + 20, bldg_x + 380, 50):
            for wy in [bldg_y + 20, bldg_y + 70, bldg_y + 120]:
                cv2.rectangle(frame, (wx, wy), (wx + 30, wy + 35),
                               (100, 150, 200), -1)
                cv2.rectangle(frame, (wx, wy), (wx + 30, wy + 35),
                               (80, 80, 70), 1)

        # "SCHOOL" sign
        cv2.putText(frame, "ELEMENTARY SCHOOL", (bldg_x + 80, bldg_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Fence
        fence_y = ground_y - 60
        for x in range(0, width, 15):
            cv2.line(frame, (x, fence_y - 10), (x, ground_y), (100, 100, 100), 1)
        cv2.line(frame, (0, fence_y), (width, fence_y), (120, 120, 120), 2)
        cv2.line(frame, (0, fence_y + 20), (width, fence_y + 20), (120, 120, 120), 2)

        # Person approaching fence
        progress = frame_num / num_frames
        person_x = int(width - 80 - progress * 200)
        person_y = ground_y - 70

        # Body
        cv2.rectangle(frame, (person_x - 12, person_y),
                       (person_x + 12, person_y + 55), (40, 40, 50), -1)
        # Head
        cv2.circle(frame, (person_x, person_y - 8), 10, (140, 110, 90), -1)
        # Legs
        cv2.line(frame, (person_x - 5, person_y + 55),
                 (person_x - 8, person_y + 75), (35, 35, 45), 4)
        cv2.line(frame, (person_x + 5, person_y + 55),
                 (person_x + 8, person_y + 75), (35, 35, 45), 4)

        # Person reaching for fence after 70%
        if progress > 0.7:
            # Hands on fence
            cv2.line(frame, (person_x + 12, person_y + 20),
                     (person_x + 25, fence_y + 10), (140, 110, 90), 3)

        # Timestamp
        cv2.putText(frame, "10:15 AM | School Perimeter - Zone B | CAM-B1",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated: {output_path} ({num_frames} frames)")


def generate_normal_activity_video(output_path: str, num_frames: int = 60,
                                    fps: int = 15, width: int = 640,
                                    height: int = 480):
    """Generate a normal activity video (no anomalies — should produce zero alerts).
    
    Simulates: daytime parking lot, person casually walking to their car.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_num in range(num_frames):
        # Daytime
        frame = np.full((height, width, 3), (180, 170, 150), dtype=np.uint8)

        # Ground
        cv2.rectangle(frame, (0, height - 80), (width, height), (120, 120, 115), -1)

        # Parking lines
        for x in range(50, width, 120):
            cv2.line(frame, (x, height - 80), (x, height), (200, 200, 190), 2)

        # Vehicle
        cv2.rectangle(frame, (200, height - 160), (380, height - 100), (60, 80, 160), -1)
        cv2.circle(frame, (230, height - 100), 12, (30, 30, 30), -1)
        cv2.circle(frame, (350, height - 100), 12, (30, 30, 30), -1)

        # Person walking normally
        progress = frame_num / num_frames
        person_x = int(100 + progress * 200)
        person_y = height - 170

        cv2.rectangle(frame, (person_x - 10, person_y),
                       (person_x + 10, person_y + 50), (50, 100, 150), -1)
        cv2.circle(frame, (person_x, person_y - 8), 10, (180, 150, 130), -1)

        cv2.putText(frame, "14:30 PM | Parking Lot - Zone A | CAM-A1",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 0), 1)

        writer.write(frame)

    writer.release()
    print(f"Generated: {output_path} ({num_frames} frames)")


def generate_all_videos(output_dir: str):
    """Generate all test videos for the demo."""
    os.makedirs(output_dir, exist_ok=True)

    print("🎬 Generating synthetic test videos...")
    print()

    generate_parking_lot_video(
        os.path.join(output_dir, "scenario_a_vehicle_breakin.mp4")
    )
    generate_school_intrusion_video(
        os.path.join(output_dir, "scenario_b_school_intrusion.mp4")
    )
    generate_normal_activity_video(
        os.path.join(output_dir, "scenario_c_normal_activity.mp4")
    )

    print()
    print("✅ All test videos generated!")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "videos")
    generate_all_videos(output_dir)
