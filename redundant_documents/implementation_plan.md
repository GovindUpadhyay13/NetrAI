# SafetyChain — Implementation Plan

---

## Decisions Locked

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Demo scenarios | **Both**: Vehicle break-in + School intrusion | Vehicle = full chain demo; School = ZeroThink fast-path demo |
| Hardware target | **RPi 5** (~$110 narrative) | Cheaper = more compelling story; prove it works on minimal hardware |
| Scope | **Full 5-stage pipeline** | No mocking; every stage runs real inference |
| Audio | **No** | Video-only simplifies scope significantly; audio can be a stretch goal |
| Dashboard | **Functional prototype** | Working > pretty; clean but not over-designed |

---

## Proposed Changes

### Project Structure

#### [NEW] Project directory layout

```
Gemma_2/
├── safetychain/
│   ├── main.py                    # Entry point — orchestrates the pipeline
│   ├── config.py                  # All configuration, thresholds, paths
│   │
│   ├── stage1_perceive/
│   │   ├── __init__.py
│   │   ├── detector.py            # YOLOv8n object detection wrapper
│   │   ├── anomaly_gate.py        # Threshold logic — should we proceed?
│   │   └── models/                # YOLO weights (downloaded at setup)
│   │
│   ├── stage2_describe/
│   │   ├── __init__.py
│   │   └── scene_describer.py     # Gemma multimodal scene description
│   │
│   ├── stage3_contextualize/
│   │   ├── __init__.py
│   │   ├── context_engine.py      # Time + zone + history lookup
│   │   ├── knowledge_graph.py     # SQLite-backed zone/norm graph
│   │   └── data/
│   │       └── zones.json         # Zone definitions, norms, rules
│   │
│   ├── stage4_verify/
│   │   ├── __init__.py
│   │   └── cot_verifier.py        # Chain-of-thought structured reasoning
│   │
│   ├── stage5_act/
│   │   ├── __init__.py
│   │   ├── alert_manager.py       # Alert creation, escalation logic
│   │   └── evidence_packager.py   # Bundle frames + reasoning into report
│   │
│   ├── dashboard/
│   │   ├── server.py              # FastAPI WebSocket server
│   │   ├── static/
│   │   │   ├── index.html         # Single-page dashboard
│   │   │   ├── style.css          # Minimal functional styling
│   │   │   └── app.js             # WebSocket client, DOM updates
│   │   └── templates/             # (if needed)
│   │
│   ├── demo/
│   │   ├── videos/                # Pre-recorded test clips
│   │   │   ├── vehicle_breakin.mp4
│   │   │   └── school_intrusion.mp4
│   │   ├── run_demo.py            # Scripted demo runner
│   │   └── generate_test_video.py # Script to create synthetic test clips
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── frame_utils.py         # Frame extraction, annotation, encoding
│   │   └── logger.py              # Structured logging for evidence trail
│   │
│   ├── requirements.txt
│   ├── setup.sh                   # One-command setup script
│   └── README.md
```

---

### Stage 1: PERCEIVE

#### [NEW] [detector.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage1_perceive/detector.py)

- Wraps **YOLOv8n** via `ultralytics` library
- Processes frames from video file or RTSP stream
- Returns list of detections: `{class, confidence, bbox, timestamp}`
- Classes of interest: `person`, `car`, `truck`, `backpack`, `knife`, `scissors`, `fire`

#### [NEW] [anomaly_gate.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage1_perceive/anomaly_gate.py)

- Receives detections from YOLO
- Applies threshold rules:
  - Person in restricted zone? → proceed
  - Person + suspicious object? → proceed
  - Vehicle + unusual time? → proceed
  - Nothing interesting? → **stop here** (95%+ of frames)
- Also detects: rapid motion changes, person count changes, new person entering frame
- Output: `AnomalyCandidate` or `None`

**Key decision:** Use **pre-recorded demo videos** for the hackathon. No need for live camera setup during judging. The architecture supports live RTSP, but demo uses files.

---

### Stage 2: DESCRIBE

#### [NEW] [scene_describer.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage2_describe/scene_describer.py)

Two approaches depending on environment:

**Option A — Kaggle/Cloud (for hackathon submission):**
- Use **Gemma 3 27B** or **Gemma 4** via Kaggle notebook with GPU
- Full multimodal: pass frame image directly to Gemma's vision encoder
- Richest scene descriptions

**Option B — Edge demo (RPi 5 narrative):**
- Use **PaliGemma 2 3B** (quantized) for visual captioning
- Pass caption text to **Gemma 4 E2B** (quantized) for structured reasoning
- Two-step but fits in RPi 5 memory

**For the hackathon, we build Option A first (Kaggle), then show Option B is architecturally possible.**

The structured prompt from the deep dive is used here — Perception + Cognition output as JSON.

---

### Stage 3: CONTEXTUALIZE

#### [NEW] [knowledge_graph.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage3_contextualize/knowledge_graph.py)

SQLite-backed lightweight graph with three tables:
- `zones` — zone_id, name, type (parking, school, corridor, loading_dock)
- `norms` — zone_id, norm_type (occupancy, time_window, expected_objects, expected_activities)
- `history` — zone_id, timestamp, event_type, was_false_positive

#### [NEW] [zones.json](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage3_contextualize/data/zones.json)

Pre-configured for demo:

```json
{
  "zones": [
    {
      "id": "zone_a",
      "name": "Parking Lot",
      "type": "parking",
      "norms": {
        "active_hours": "06:00-23:00",
        "expected_occupants": "residents_with_badge",
        "suspicious_objects": ["slim_jim", "crowbar", "glass_cutter"],
        "suspicious_behaviors": ["crouching_at_vehicle", "looking_around_furtively"]
      }
    },
    {
      "id": "zone_b",
      "name": "School Perimeter",
      "type": "school",
      "norms": {
        "school_hours": "08:00-15:30",
        "authorized_entry": "main_gate_only",
        "fence_climbing": "ALWAYS_CRITICAL",
        "unknown_adults_during_school": "ALWAYS_CRITICAL"
      }
    }
  ]
}
```

#### [NEW] [context_engine.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage3_contextualize/context_engine.py)

- Takes `AnomalyCandidate` + `SceneDescription`
- Queries knowledge graph for zone norms
- Checks current time against zone rules
- Checks for known false-positive patterns in history
- Returns `ContextReport { supports_anomaly: bool, context_details: str, known_fp: bool }`

---

### Stage 4: VERIFY

#### [NEW] [cot_verifier.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage4_verify/cot_verifier.py)

The core reasoning engine. Two modes:

**ZeroThink (for CRITICAL events):**
```python
if context.fence_climbing and context.zone_type == "school" and context.school_hours:
    return Verdict(
        classification="CONFIRMED_ANOMALY",
        confidence=0.99,
        severity="CRITICAL",
        urgency="EMERGENCY",
        reasoning="ZeroThink: School perimeter breach during hours — immediate escalation",
        chain_id=generate_chain_id()
    )
```

**FullThink (for ambiguous events):**
- Constructs the 5-step verification prompt from the deep dive
- Passes to Gemma: evidence summary → consistency → context check → alternative hypotheses → severity → verdict
- Parses structured JSON response
- Returns `Verdict` with full reasoning chain

**Implementation detail:** The prompt template is stored as a `.txt` file and filled with f-string interpolation. This makes it easy to iterate on the prompt without touching code.

---

### Stage 5: ACT

#### [NEW] [alert_manager.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage5_act/alert_manager.py)

- Receives `Verdict` from Stage 4
- Applies escalation rules (LOG / NOTIFY / ALERT / EMERGENCY)
- Pushes to dashboard via WebSocket
- Logs to evidence store

#### [NEW] [evidence_packager.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/stage5_act/evidence_packager.py)

- Bundles: annotated frame + reasoning chain + context report + verdict
- Saves as JSON evidence file (one per alert)
- Can export as simple HTML report for forensic review

---

### Dashboard

#### [NEW] [server.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/server.py)

- **FastAPI** with WebSocket endpoint
- Serves static files (index.html, style.css, app.js)
- Receives alerts from pipeline, broadcasts to connected dashboard clients
- REST endpoint: `GET /alerts` for history

#### [NEW] [index.html](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/static/index.html)

Functional prototype layout:

```
┌──────────────────────────────────────────────────┐
│  ⛓️ SafetyChain           [🟢 Pipeline Active]   │
├────────────┬─────────────────┬───────────────────┤
│ ALERT LIST │  VIDEO FRAME    │ REASONING CHAIN   │
│            │  (annotated     │                   │
│ Clickable  │   screenshot    │ Step 1: ✅/❌     │
│ list with  │   with YOLO     │ Step 2: ✅/❌     │
│ severity   │   bounding      │ Step 3: ✅/❌     │
│ badges     │   boxes)        │ Step 4: ✅/❌     │
│            │                 │ Step 5: verdict   │
│            │                 │                   │
│            │                 │ Confidence: 89%   │
│            │                 │ Action: ...       │
├────────────┴─────────────────┴───────────────────┤
│ Pipeline: PERCEIVE → DESCRIBE → CONTEXT → VERIFY │
│ Latency: 0.00s    Stage: Idle                     │
└──────────────────────────────────────────────────┘
```

- **Left panel:** Alert feed (newest first, color-coded by severity)
- **Center panel:** Annotated frame from the triggering event
- **Right panel:** Full reasoning chain, collapsible per step
- **Bottom bar:** Pipeline status indicator showing which stage is active + cumulative latency

**Tech:** Vanilla HTML/CSS/JS. No React — keeps it simple, no build step, works anywhere.

#### [NEW] [style.css](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/static/style.css)

- Dark theme (security operations aesthetic)
- CSS Grid layout for the 3-panel design
- Color-coded severity badges: 🟢 LOG, 🟡 NOTIFY, 🟠 ALERT, 🔴 EMERGENCY
- Monospace font for reasoning chain (feels like a terminal/forensic tool)
- Minimal animations: pulse on new alert, fade-in for reasoning steps

#### [NEW] [app.js](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/dashboard/static/app.js)

- WebSocket connection to FastAPI server
- Receives alert JSON → renders into DOM
- Click alert → show its frame + reasoning chain
- Pipeline status bar updates in real-time
- Auto-scroll alert feed

---

### Demo Runner

#### [NEW] [run_demo.py](file:///c:/Users/HP/Desktop/VS/Gemma_2/safetychain/demo/run_demo.py)

Scripted demo that:
1. Starts the dashboard server
2. Opens browser to `http://localhost:8000`
3. Runs **Scenario 1: Vehicle Break-in** — plays video, shows full 5-stage chain completing in real-time
4. Pauses for presenter to explain the reasoning chain
5. Runs **Scenario 2: School Intrusion** — shows ZeroThink fast-path, EMERGENCY alert fires in <100ms
6. Shows the contrast: full deliberation vs. instant escalation

---

## Verification Plan

### Automated Tests

```bash
# Unit tests for each stage
pytest tests/test_stage1_perceive.py    # YOLO detects person in test frame
pytest tests/test_stage3_context.py     # Knowledge graph returns correct norms
pytest tests/test_stage4_zerothink.py   # School intrusion triggers ZeroThink
pytest tests/test_pipeline_e2e.py       # Full pipeline on test video produces alert
```

### Manual Verification

1. Run `run_demo.py` — both scenarios should produce correct alerts
2. Vehicle break-in: full reasoning chain visible, 80%+ confidence, ALERT level
3. School intrusion: ZeroThink fires, EMERGENCY level, <200ms total latency
4. Dashboard shows both alerts with clickable reasoning chains
5. False positive suppression: run a "normal activity" video — should produce zero alerts

---

## Implementation Order

> [!IMPORTANT]
> Build bottom-up: pipeline first, dashboard last. A working pipeline with console output is more valuable than a pretty dashboard with no brain.

| Phase | Components | Deliverable |
|-------|-----------|-------------|
| **Phase 1** | `config.py`, `stage1_perceive/`, `utils/` | YOLO running on demo video, console output |
| **Phase 2** | `stage2_describe/` | Gemma generating scene descriptions from flagged frames |
| **Phase 3** | `stage3_contextualize/`, `zones.json` | Context engine enriching descriptions with zone/time data |
| **Phase 4** | `stage4_verify/` | CoT reasoning producing verdicts with evidence chains |
| **Phase 5** | `stage5_act/`, `main.py` | Full pipeline end-to-end, alerts printed to console |
| **Phase 6** | `dashboard/` | Web UI displaying alerts + reasoning chains |
| **Phase 7** | `demo/run_demo.py` | Scripted demo with both scenarios |

---

## Open Questions

> [!WARNING]
> **Gemma API vs. Local inference:** For the hackathon demo, are you planning to:
> - **(A)** Run on **Kaggle** with GPU and use Gemma via `transformers` library (easier, more powerful)
> - **(B)** Run **locally** with a Gemma API endpoint (Ollama, LM Studio, etc.)
> - **(C)** Use **Google AI Studio / Vertex AI** API calls to Gemma
>
> This affects how we wire up Stages 2 and 4. The pipeline architecture stays the same regardless — only the inference backend changes.
