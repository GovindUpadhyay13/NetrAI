"""
detection/anomaly_clip/visualizer.py
Plots the continuous per-frame anomaly score curve with threshold and flagged anomaly windows.

Assumptions:
- Input is the output dictionary from VideoAnomalyScorer.
- Saves high-resolution PNG image suitable for inspection and UI display.
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def plot_anomaly_curve(
    scoring_result: Dict,
    output_path: str,
    title: Optional[str] = "AnomalyCLIP Surveillance Anomaly Curve",
) -> str:
    """
    Renders and saves a per-frame anomaly score plot.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    timestamps = scoring_result.get("timestamps", [])
    scores = scoring_result.get("scores", [])
    threshold = scoring_result.get("threshold", 0.5)
    flagged_windows = scoring_result.get("flagged_windows", [])

    plt.figure(figsize=(12, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot score line
    plt.plot(timestamps, scores, label="AnomalyCLIP Score", color="#e63946", linewidth=2.0)

    # Plot threshold line
    plt.axhline(
        y=threshold,
        color="#1d3557",
        linestyle="--",
        linewidth=1.5,
        label=f"Detection Threshold ({threshold:.2f})",
    )

    # Shade flagged windows
    for i, win in enumerate(flagged_windows):
        start = win["start_time_sec"]
        end = win["end_time_sec"]
        lbl = f"Flagged [{win['anomaly_type']}]" if i == 0 else ""
        plt.axvspan(start, end, color="#ff4d6d", alpha=0.35, label=lbl)
        # Text annotation for peak score
        mid_time = (start + end) / 2
        plt.text(
            mid_time,
            min(0.95, win["peak_score"] + 0.03),
            f"{win['anomaly_type']}\n({win['peak_score']:.2f})",
            color="#800f2f",
            fontsize=8,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Time (seconds)", fontsize=11)
    plt.ylabel("Anomaly Probability", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, max(timestamps) if timestamps else 1.0)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
