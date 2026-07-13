# SafetyChain — Implementation Plan

## Goal

Implement the full SafetyChain system: a 5-stage Chain-of-Thought verification pipeline for public safety, targeting the Google Gemma Hackathon Track 2 (AI for Public Safety). The system runs entirely on-device, processes video to detect anomalies, reasons about them through a structured CoT pipeline, and presents results on a real-time dashboard.

---

## User Review Required

> [!IMPORTANT]
> **Gemma API Backend:** The design documents mention multiple inference options. For this implementation I will use the **Google AI Studio / Gemini API** approach (using the `google-generativeai` Python SDK with `gemma-4-26b-a4b-it` per hackathon requirements) since this is the most practical for a hackathon demo without requiring local GPU setup. The architecture abstracts this so swapping to local inference (Ollama, LM Studio) or Kaggle GPU is a config change.

> [!IMPORTANT]
> **YOLO Model:** I will use `ultralytics` YOLOv8n. The model weights will be auto-downloaded on first run. No pre-downloaded ONNX needed for the demo.

> [!IMPORTANT]
> **Demo Videos:** Since we need pre-recorded test clips for the two scenarios (vehicle break-in, school intrusion), I will create a `generate_test_video.py` script that generates **synthetic test videos** using OpenCV (drawing simulated scenes with shapes/text). For a real demo, you would swap in actual footage.

> [!WARNING]
> **Audio:** Audio is strictly not used currently.

---

## Proposed Changes

### Project Structure

```
Gemma_2/
├── safetychain/
│   ├── __init__.py
│   ├── main.py                    # Entry point — orchestrates the full pipeline
│   ├── config.py                  # All configuration, thresholds, paths
│   ├── models.py                  # All data models (dataclasses)
│   │
│   ├── stage1_perceive/
│   │   ├── __init__.py
│   │   ├── detector.py            # YOLOv8n object detection wrapper
│   │   └── anomaly_gate.py        # Threshold filter — should we proceed?
│   │
│   ├── stage2_describe/
│   │   ├── __init__.py
│   │   └── scene_describer.py     # Gemma VLM scene description (PerCoAct prompt)
│   │
│   ├── stage3_contextualize/
│   │   ├── __init__.py
│   │   ├── context_engine.py      # Temporal + zone + history aggregation
│   │   └── knowledge_graph.py     # SQLite-backed zone/norm/history graph
│   │
│   ├── stage4_verify/
│   │   ├── __init__.py
│   │   └── cot_verifier.py        # Chain-of-thought structured reasoning (Gemma)
│   │
│   ├── stage5_act/
│   │   ├── __init__.py
│   │   ├── alert_manager.py       # Alert creation + escalation state machine
│   │   └── evidence_packager.py   # Bundle frame + reasoning into evidence
│   │
│   ├── dashboard/
│   │   ├── server.py              # FastAPI + WebSocket server
│   │   └── static/
│   │       ├── index.html         # Single-page dashboard
│   │       ├── style.css          # Dark theme security operations aesthetic
│   │       └── app.js             # WebSocket client + DOM rendering
│   │
│   ├── demo/
│   │   ├── videos/                # Pre-recorded/synthetic test clips
│   │   ├── run_demo.py            # Scripted demo runner (both scenarios)
│   │   └── generate_test_video.py # Creates synthetic test clips with OpenCV
│   │
│   ├── data/
│   │   ├── zones.json             # Zone definitions, norms, rules
│   │   └── sops.json              # Standard Operating Procedures
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── frame_utils.py         # Frame annotation, encoding, extraction
│   │   └── logger.py              # Structured logging for evidence trail
│   │
│   ├── requirements.txt
│   └── README.md
```

---

### Phase 1: Foundation

#### [NEW] [config.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/config.py)
All tunable parameters in a single config dataclass: YOLO thresholds, Gemma model settings, zone paths, escalation timers, dashboard ports, demo mode flags.

#### [NEW] [models.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/models.py)
All data models as Python dataclasses: `Detection`, `AnomalyCandidate`, `Person`, `ObjectOfInterest`, `SceneDescription`, `ZoneContext`, `TemporalContext`, `HistoricalContext`, `ProtocolContext`, `ContextReport`, `ReasoningStep`, `Verdict`, `Alert`. Exactly as specified in the design document.

#### [NEW] [utils/logger.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/utils/logger.py)
Structured logging with JSON output for evidence trail. Logs pipeline stages, latencies, and decisions.

#### [NEW] [utils/frame_utils.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/utils/frame_utils.py)
Frame annotation (draw YOLO bounding boxes), JPEG encoding to base64, frame extraction from video.

#### [NEW] [requirements.txt](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/requirements.txt)
Dependencies: `ultralytics`, `opencv-python`, `fastapi`, `uvicorn`, `websockets`, `google-generativeai`, `Pillow`, `numpy`.

---

### Phase 2: Stage 1 — PERCEIVE

#### [NEW] [stage1_perceive/detector.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage1_perceive/detector.py)
- Wraps YOLOv8n via `ultralytics`
- Processes frames from video file (or RTSP in production)
- Returns `list[Detection]` with class, confidence, bbox
- Filters to classes of interest: person, car, truck, backpack, knife, scissors

#### [NEW] [stage1_perceive/anomaly_gate.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage1_perceive/anomaly_gate.py)
- Receives detections + frame + zone context
- Applies threshold rules (confidence, zone restrictions, motion delta)
- Returns `AnomalyCandidate` or `None` (95%+ of frames rejected here)

---

### Phase 3: Stage 2 — DESCRIBE

#### [NEW] [stage2_describe/scene_describer.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage2_describe/scene_describer.py)
- Takes `AnomalyCandidate` (with annotated frame)
- Constructs the PerCoAct prompt (Perception + Cognition, as specified in design)
- Sends frame + prompt to Gemma via Google AI Studio API (multimodal)
- Parses structured JSON response into `SceneDescription`
- Falls back to YOLO-only description if Gemma fails

---

### Phase 4: Stage 3 — CONTEXTUALIZE

#### [NEW] [stage3_contextualize/knowledge_graph.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage3_contextualize/knowledge_graph.py)
- SQLite-backed with tables: `sites`, `zones`, `norms`, `history`, `cameras`, `sops`
- Schema matches the ER diagram from the architecture document
- Initialization from `zones.json` and `sops.json`
- Methods: `get_zone_norms()`, `get_history()`, `get_sop()`, `record_event()`, `update_from_feedback()`

#### [NEW] [stage3_contextualize/context_engine.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage3_contextualize/context_engine.py)
- Takes `AnomalyCandidate` + `SceneDescription`
- Queries knowledge graph for zone norms
- Checks current time against zone active hours
- Looks up false-positive history
- Retrieves matching SOP
- Returns `ContextReport` with verdict (SUPPORTS/NEUTRAL/REFUTES) and suppress flag

#### [NEW] [data/zones.json](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/data/zones.json)
Pre-configured for two demo zones: Parking Lot (Zone A) and School Perimeter (Zone B).

#### [NEW] [data/sops.json](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/data/sops.json)
Standard Operating Procedures for demo scenarios: vehicle theft response, school intrusion response.

---

### Phase 5: Stage 4 — VERIFY

#### [NEW] [stage4_verify/cot_verifier.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage4_verify/cot_verifier.py)
- Implements the Depth Router: ZeroThink / LessThink / FullThink / MoreThink
- **ZeroThink**: For CRITICAL events (weapon/fire + school zone) — skip reasoning, instant verdict in <50ms
- **FullThink**: Constructs the 5-step CoT verification prompt, sends to Gemma, parses structured JSON
- Returns `Verdict` with full reasoning chain, confidence, severity, and recommended action

---

### Phase 6: Stage 5 — ACT

#### [NEW] [stage5_act/alert_manager.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage5_act/alert_manager.py)
- Receives `Verdict`, applies escalation matrix (LOG/NOTIFY/ALERT/EMERGENCY)
- Creates `Alert` object with all evidence
- Pushes to dashboard via WebSocket
- Handles operator feedback (TP/FP) → feeds back to knowledge graph

#### [NEW] [stage5_act/evidence_packager.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage5_act/evidence_packager.py)
- Bundles: annotated frame (base64) + reasoning chain + context report + verdict
- Saves as JSON evidence file
- Can export as HTML report for forensic review

---

### Phase 7: Pipeline Orchestrator

#### [NEW] [main.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/main.py)
- Entry point that wires all 5 stages together
- Video loop: read frame → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
- Pipeline status tracking (which stage is active, cumulative latency)
- Broadcasts pipeline status via WebSocket

---

### Phase 8: Dashboard

#### [NEW] [dashboard/server.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/server.py)
- FastAPI with REST endpoints: `GET /api/alerts`, `GET /api/alerts/{id}`, `POST /api/alerts/{id}/feedback`, `GET /api/pipeline/status`, `GET /api/stats`
- WebSocket endpoint: `/ws/alerts` for real-time alert streaming
- Serves static files (index.html, style.css, app.js)
- SQLite alert store for persistence

#### [NEW] [dashboard/static/index.html](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/static/index.html)
3-panel layout:
- **Left**: Alert feed (severity-coded, clickable)
- **Center**: Annotated evidence frame viewer
- **Right**: Reasoning chain panel (collapsible 5-step chain)
- **Header**: System status, pipeline stage indicator
- **Footer**: Action bar (Acknowledge / Investigate / Dismiss / Feedback)

#### [NEW] [dashboard/static/style.css](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/static/style.css)
- Dark theme (security operations center aesthetic)
- CSS Grid for 3-panel layout
- Color-coded severity: 🟢 LOG → 🟡 NOTIFY → 🟠 ALERT → 🔴 EMERGENCY
- Monospace reasoning chain (forensic tool feel)
- Pulse animation on new alerts, smooth transitions
- Glassmorphism panels with subtle borders
- Premium typography (Inter font)

#### [NEW] [dashboard/static/app.js](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/static/app.js)
- WebSocket connection + auto-reconnect
- Receives alert JSON → renders into DOM
- Click alert → show frame + reasoning chain
- Pipeline status bar updates (animated stage indicators)
- Feedback buttons (TP/FP) with POST to API

---

### Phase 9: Demo System

#### [NEW] [demo/generate_test_video.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/demo/generate_test_video.py)
Generates synthetic test videos using OpenCV for the two demo scenarios.

#### [NEW] [demo/run_demo.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/demo/run_demo.py)
Scripted demo runner:
1. Starts dashboard server
2. Opens browser
3. Runs Scenario 1: Vehicle Break-in (FullThink path, ~670ms)
4. Runs Scenario 2: School Intrusion (ZeroThink path, ~70ms)
5. Shows the contrast on the dashboard

---

## Verification Plan

### Automated Tests
```bash
# Install dependencies
pip install -r safetychain/requirements.txt

# Run the demo
python safetychain/demo/run_demo.py
```

### Manual Verification
1. Dashboard loads at `http://localhost:8000` with dark theme
2. Vehicle break-in scenario: full 5-step reasoning chain visible, ~89% confidence, ALERT level
3. School intrusion scenario: ZeroThink fires, EMERGENCY level, near-instant
4. Both alerts appear in left panel with correct severity badges
5. Clicking an alert shows annotated frame + reasoning chain
6. Pipeline status bar shows stage progression in real-time
7. Feedback buttons work (TP/FP updates knowledge graph)
