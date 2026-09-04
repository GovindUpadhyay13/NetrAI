# NetrAI: Women's-Safety-Focused Surveillance System

A camera-based anomaly detection, distress gesture recognition, and cross-camera tracking system designed for women's safety in urban and campus environments.

---

## System Architecture

```
                       [Video Ingestion]
                               |
            +------------------+------------------+
            |                                     |
    [AnomalyCLIP Branch]                 [MediaPipe Pose]
   (ViT-B/16 + Temporal)                (Distress Gestures)
            |                                     |
    anomaly_detected                       gesture_flagged
            \                                     /
             +-------------> [Event Bus] <-------+
                            (Redis Streams)
                                  |
                                  v
                        [VLM Reasoner: Gemini]
                        (3x3 Multi-Frame Grid)
                                  |
                            vlm_analyzed
                                  |
                    [Cross-Camera Re-ID]
                  (YOLOv8 + CLIP + Qdrant)
                                  |
                             reid_match
                                  |
                       [SQLite Trace DB]
                                  |
                                  v
                        [FastAPI Live Dashboard]
```

---

## Directory Layout

```
NetrAI/
  ingestion/            # Video reader and frame extraction utilities
  detection/
    anomaly_clip/       # AnomalyCLIP inference wrapper and per-frame scoring
    gesture/            # MediaPipe body pose extraction and distress gesture classifier
  reasoning/
    frame_grid.py        # 3x3 temporal grid image builder
    gemini_analyzer.py   # Gemini 2.5 Flash multimodal VLM incident analyzer
  bus/
    schemas.py          # Pydantic schemas for the pinned event format
    publisher.py        # Redis Streams publisher with resilient in-memory fallback
    flow_runner.py      # End-to-end incident pipeline runner
    consumers/
      trace_logger.py   # Consumes all stages -> writes to SQLite trace.db
      reid_matcher.py   # Enrolls flagged subjects & searches Qdrant on new feeds
  reid/
    embed.py            # YOLOv8 crop detector + shared CLIP image encoder
    gallery.py          # Qdrant vector database interface (512-dim cosine)
  trace/
    db.py               # SQLite schema, indices, and CRUD operations
    dashboard.py        # FastAPI live operations center dashboard with HTML UI
  tests/
    test_milestone4.py  # Standalone test for Event Bus + Trace Logger
    test_milestone6.py  # Standalone test for Cross-Camera Re-ID
  main.py               # Main CLI runner (demo and live modes)
  requirements.txt
```

---

## Pinned Event Schema

All pipeline stages publish JSON events conforming strictly to:

```json
{
  "incident_id": "uuid4",
  "camera_id": "string",
  "timestamp": "iso8601",
  "stage": "anomaly_detected | gesture_flagged | vlm_analyzed | dispatched | reid_match | trace",
  "anomaly_score": 0.0,
  "anomaly_type": "string or null",
  "distress_gesture": "bool",
  "vlm_report": "string or null",
  "severity": "low | medium | high or null",
  "payload_ref": "path or object key to frames/crops, not raw bytes in the event"
}
```

---

## Quickstart & Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Keys** (`.env`):
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   # Optional:
   DISPATCH_WEBHOOK_URL=https://your-webhook-endpoint.com
   TWILIO_ACCOUNT_SID=...
   TWILIO_AUTH_TOKEN=...
   TWILIO_PHONE_NUMBER=...
   ```

---

## Running Milestones Standalone

Each component can be run and tested independently:

### Milestone 1 — Ingestion + AnomalyCLIP Scoring
```bash
python -m detection.anomaly_clip.run_standalone --video path/to/video.mp4 --threshold 0.45
# Or with auto-generated test footage:
python -m detection.anomaly_clip.run_standalone --demo
```
*Outputs: `flagged_windows.json` + `anomaly_curve.png`*

### Milestone 2 — MediaPipe Distress Gesture Branch
```bash
python -m detection.gesture.run_standalone --video path/to/clip.mp4
# Or with auto-generated SOS gesture footage:
python -m detection.gesture.run_standalone --demo
```
*Outputs: Distress flag boolean + confidence score*

### Milestone 3 — 3x3 Grid Builder + Gemini VLM Reasoner
```bash
python -m reasoning.run_standalone --demo
```
*Outputs: `incident_grid_3x3.png` + structured JSON report from Gemini 2.5 Flash*

### Milestone 4 — Event Bus + Trace Logger
```bash
python tests/test_milestone4.py
```
*Outputs: Full stage-by-stage event lifecycle stored in `trace.db`*

### Milestone 5 — Cross-Camera Subject Re-ID
```bash
python tests/test_milestone6.py
```
*Outputs: Subject detected on Camera 1 is recognized and linked when spotted on Camera 2*

---

## Running the Complete System (Demo Mode)

Run `main.py demo` to launch the multi-camera concurrent simulation and live operations dashboard:

```bash
python main.py demo --port 8000
```

Open your browser to: **`http://localhost:8000`** to view real-time incident traces, severity classification, and cross-camera sightings.
