# ⛓️ SafetyChain — System Design Document

---

## 1. Design Philosophy

### Core Principles

| Principle | What It Means | How We Enforce It |
|-----------|--------------|-------------------|
| **System 2 Thinking** | Don't react — reason. Every alert has a chain of evidence. | 5-stage pipeline with mandatory verification before any operator-facing alert |
| **Progressive Refinement** | Each stage adds semantic depth and can kill false positives early | Funnel design: 95% of raw detections never reach the operator |
| **Privacy by Architecture** | Not a policy — a hardware guarantee. Data doesn't leave the device. | All inference on-device (RPi 5). No cloud calls. No telemetry. |
| **Explainability as Product** | The reasoning chain IS the product, not just the alert | Every alert ships with a 5-step verification trace, alternative hypotheses, and confidence breakdown |
| **Adaptive Depth** | Spend compute where it matters | ZeroThink for emergencies (<50ms), FullThink for ambiguity (~300ms) |
| **Learn Without Retraining** | Get smarter from operator feedback without touching model weights | Knowledge graph enrichment from TP/FP feedback loop |

---

## 2. Data Models

### 2.1 AnomalyCandidate (Stage 1 → Stage 2)

```python
@dataclass
class Detection:
    class_name: str          # "person", "vehicle", "knife", etc.
    confidence: float        # 0.0 - 1.0
    bbox: tuple[int,int,int,int]  # x1, y1, x2, y2
    
@dataclass
class AnomalyCandidate:
    id: str                  # UUID
    timestamp: datetime
    frame: np.ndarray        # Raw frame (OpenCV BGR)
    frame_annotated: np.ndarray  # Frame with YOLO bboxes drawn
    detections: list[Detection]
    zone_id: str             # Which zone this camera covers
    camera_id: str
    motion_delta: float      # Frame-diff motion score
    trigger_reason: str      # "person_in_restricted_zone", "rapid_motion", etc.
```

### 2.2 SceneDescription (Stage 2 → Stage 3)

```python
@dataclass
class Person:
    id: str                  # "P1", "P2", etc.
    description: str         # "Adult male, dark hoodie, face partially obscured"
    position: str            # "Near driver-side door of silver sedan"
    posture: str             # "Crouching, looking around repeatedly"
    movement: str            # "Intermittent — pauses, then moves quickly"

@dataclass
class ObjectOfInterest:
    type: str                # "vehicle", "tool", "bag", etc.
    description: str         # "Slim metallic object in right hand"

@dataclass
class SceneDescription:
    candidate_id: str        # Links back to AnomalyCandidate
    scene_environment: str   # "Parking lot, nighttime, poorly lit"
    people: list[Person]
    objects: list[ObjectOfInterest]
    visible_text: list[str]  # Signs, license plates
    activity: str            # "Possible vehicle break-in attempt"
    norm_violation: str      # "Person using tool on vehicle door lock"
    suspiciousness: str      # "NORMAL" | "UNUSUAL" | "CONCERNING" | "ALARMING"
    raw_json: dict           # Full Gemma output for evidence trail
```

### 2.3 ContextReport (Stage 3 → Stage 4)

```python
@dataclass
class ZoneContext:
    zone_id: str
    zone_name: str
    zone_type: str           # "parking", "school", "corridor"
    active_hours: str        # "06:00-23:00"
    currently_active: bool
    expected_occupancy: str  # "residents_with_badge"

@dataclass 
class TemporalContext:
    current_time: datetime
    day_of_week: str
    is_holiday: bool
    holiday_name: str | None
    is_within_active_hours: bool
    
@dataclass
class HistoricalContext:
    similar_events_count: int      # In last 30 days
    false_positive_rate: float     # For this camera + similar detection
    known_fp_pattern: str | None   # "Swaying tree branch" etc.
    last_event_in_zone: datetime | None

@dataclass
class ProtocolContext:
    matching_sop: str | None       # "SOP-014: Vehicle Theft Response"
    procedure_summary: str | None  
    contacts: dict | None          # {"patrol": "ext. 2200", "police": "911"}
    policy_notes: str | None       # "Do NOT approach suspect"

@dataclass
class ContextReport:
    candidate_id: str
    zone: ZoneContext
    temporal: TemporalContext
    historical: HistoricalContext
    protocol: ProtocolContext
    verdict: str             # "SUPPORTS_ANOMALY" | "NEUTRAL" | "REFUTES_ANOMALY"
    confidence_modifier: float  # -0.3 to +0.3 (context boost/penalty)
    suppress: bool           # True = kill alert (known FP pattern)
    suppress_reason: str | None
```

### 2.4 Verdict (Stage 4 → Stage 5)

```python
@dataclass
class ReasoningStep:
    step_number: int         # 1-5
    title: str               # "Evidence Consistency"
    content: str             # Full reasoning text
    passed: bool             # ✅ or ❌

@dataclass
class Verdict:
    candidate_id: str
    chain_id: str            # UUID for forensic tracing
    classification: str      # "FALSE_POSITIVE" | "SUSPICIOUS" | "CONFIRMED_ANOMALY"
    confidence: float        # 0.0 - 1.0
    severity: str            # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    urgency: str             # "MONITOR" | "INVESTIGATE" | "INTERVENE" | "EMERGENCY"
    reasoning_strategy: str  # "ZeroThink" | "LessThink" | "FullThink" | "MoreThink"
    reasoning_chain: list[ReasoningStep]  # Empty for ZeroThink
    alternative_hypotheses: list[str]
    recommended_action: str
    consequences_if_ignored: str
    reasoning_latency_ms: int
```

### 2.5 Alert (Stage 5 → Dashboard)

```python
@dataclass
class Alert:
    alert_id: str            # UUID
    chain_id: str            # Links to Verdict
    timestamp: datetime
    severity: str            # "LOG" | "NOTIFY" | "ALERT" | "EMERGENCY"
    title: str               # "Possible Vehicle Break-in"
    zone_name: str
    confidence: float
    frame_b64: str           # Base64-encoded annotated JPEG
    verdict: Verdict
    context_summary: str     # One-line context
    sop: str | None          # Retrieved SOP text
    contacts: dict | None
    status: str              # "active" | "acknowledged" | "dismissed" | "resolved"
    operator_feedback: str | None  # "true_positive" | "false_positive"
```

---

## 3. API Design

### 3.1 REST Endpoints (FastAPI)

| Method | Endpoint | Purpose | Request | Response |
|--------|----------|---------|---------|----------|
| `GET` | `/api/alerts` | List recent alerts | `?limit=20&severity=ALERT,EMERGENCY` | `Alert[]` |
| `GET` | `/api/alerts/{alert_id}` | Get full alert with reasoning | — | `Alert` (full) |
| `POST` | `/api/alerts/{alert_id}/feedback` | Operator marks TP/FP | `{feedback: "true_positive"/"false_positive", note: "..."}` | `{status: "updated"}` |
| `GET` | `/api/pipeline/status` | Pipeline health check | — | `{stage: "idle"/"perceive"/..., fps: 28.4, alerts_today: 7}` |
| `GET` | `/api/zones` | List configured zones | — | `Zone[]` |
| `GET` | `/api/stats` | Dashboard statistics | — | `{total_alerts, true_positives, false_positives, suppressed}` |

### 3.2 WebSocket Protocol

**Endpoint:** `ws://localhost:8000/ws/alerts`

**Server → Client messages:**

```jsonc
// New alert
{
  "type": "new_alert",
  "data": {
    "alert_id": "a-2026-07-14-0342",
    "severity": "ALERT",
    "title": "Possible Vehicle Break-in",
    "zone": "Parking Lot - Zone A",
    "confidence": 0.89,
    "frame_b64": "<base64 JPEG>",
    "reasoning_chain": [
      {"step": 1, "title": "Evidence Consistency", "passed": true, "content": "..."},
      {"step": 2, "title": "Context Check", "passed": true, "content": "..."},
      // ...
    ],
    "recommended_action": "Alert security patrol",
    "sop": "SOP-014: Vehicle Theft/Break-in Response"
  }
}

// Pipeline status update
{
  "type": "pipeline_status",
  "data": {
    "active_stage": "VERIFY",
    "stage_latency_ms": 245,
    "cumulative_latency_ms": 520,
    "fps": 28.4
  }
}

// Alert status change
{
  "type": "alert_updated",
  "data": {
    "alert_id": "a-2026-07-14-0342",
    "status": "acknowledged",
    "feedback": "true_positive"
  }
}
```

---

## 4. Prompt Engineering Design

### 4.1 Scene Description Prompt (Stage 2)

**Design rationale:** Inspired by PerCoAct-CoT (Vad-R1-Plus). Separating Perception from Cognition forces the model to first describe what it sees objectively, then interpret — reducing hallucination where the model jumps to conclusions.

```
SYSTEM: You are a trained security camera analyst. You observe
scenes carefully and report with precision. You never speculate
beyond what is visible. You always structure your output as JSON.

USER: Analyze this camera frame. A detection system has flagged:
- Object class: {detection.class_name} ({detection.confidence:.0%})
- Location in frame: {detection.bbox}
- Camera zone: {zone_name} ({zone_type})

PERCEPTION — Describe ONLY what you can see:
1. Scene: Describe the overall environment
2. People: For each person, describe clothing, posture, position
3. Objects: List all notable objects, especially unusual ones
4. Text: Any visible signage or text

COGNITION — Interpret what you see:
1. Activity: What is happening?
2. Norm violation: Does anything seem wrong for a {zone_type}?
3. Relationships: How do people and objects relate?
4. Suspiciousness: Rate as NORMAL, UNUSUAL, CONCERNING, or ALARMING

Respond ONLY with valid JSON matching this schema:
{schema}
```

**Key design decisions:**
- Zone type injected into prompt so Gemma knows "parking lot" norms vs "school" norms
- Detection metadata from YOLO gives Gemma a starting point (reduces hallucination)
- Strict JSON schema enforcement prevents free-form rambling

### 4.2 CoT Verification Prompt (Stage 4)

**Design rationale:** Inspired by AD-FM (AAAI 2026) multi-stage reasoning and SafeChain (ACL 2025). Forcing explicit alternative hypotheses is the key anti-hallucination mechanism — the model must argue against its own conclusion.

```
SYSTEM: You are a deliberative safety verification system. You
NEVER jump to conclusions. You evaluate evidence systematically
and always consider alternative explanations. If in doubt, you
err on the side of caution but document your uncertainty.

USER: Verify this potential anomaly using the evidence below.

═══ EVIDENCE ═══
Visual observation: {scene_description}
Detection class: {detection.class_name} ({detection.confidence:.0%})
Zone: {zone_name} ({zone_type})
Time: {current_time} ({day_of_week})
Zone active hours: {active_hours}
Currently active: {is_active}
Historical FP rate for this camera: {fp_rate:.0%}
Known FP patterns: {known_fp_patterns}

═══ VERIFY IN 5 STEPS ═══

STEP 1 — EVIDENCE QUALITY
  How clear is the visual evidence? Rate: HIGH/MEDIUM/LOW
  Confidence in primary detection: ___%

STEP 2 — CONTEXT ALIGNMENT
  Is this behavior abnormal for {zone_type} at {current_time}?
  Context verdict: SUPPORTS_ANOMALY / NEUTRAL / REFUTES_ANOMALY

STEP 3 — ALTERNATIVE HYPOTHESES
  List 2-3 benign explanations for what is observed.
  For each, rate: PLAUSIBLE / UNLIKELY / REFUTED
  Can ANY benign hypothesis explain ALL evidence?

STEP 4 — SEVERITY
  If genuine: Threat level (LOW/MEDIUM/HIGH/CRITICAL)
  Urgency: MONITOR / INVESTIGATE / INTERVENE / EMERGENCY
  What happens if we ignore this?

STEP 5 — VERDICT
  Classification: FALSE_POSITIVE / SUSPICIOUS / CONFIRMED_ANOMALY
  Confidence: ___%
  Recommended action: ___

Respond ONLY with valid JSON matching this schema:
{schema}
```

---

## 5. Escalation Logic Design

### Decision Matrix

```
                        ┌─────────────────────────────────┐
                        │       CONFIDENCE LEVEL           │
                        ├────────┬────────┬────────┬───────┤
                        │ <40%   │ 40-70% │ 70-90% │ >90%  │
    ┌───────────────────┼────────┼────────┼────────┼───────┤
    │ LOW (loitering,   │  LOG   │ NOTIFY │ NOTIFY │ ALERT │
S   │  minor trespass)  │        │        │        │       │
E   ├───────────────────┼────────┼────────┼────────┼───────┤
V   │ MEDIUM (break-in, │  LOG   │ NOTIFY │ ALERT  │ ALERT │
E   │  vandalism)       │        │        │        │       │
R   ├───────────────────┼────────┼────────┼────────┼───────┤
I   │ HIGH (assault,    │ NOTIFY │ ALERT  │ ALERT  │ EMERG │
T   │  theft in prog.)  │        │        │        │       │
Y   ├───────────────────┼────────┼────────┼────────┼───────┤
    │ CRITICAL (weapon, │ ALERT  │ EMERG  │ EMERG  │ EMERG │
    │  fire, school)    │        │        │        │       │
    └───────────────────┴────────┴────────┴────────┴───────┘
```

### Time-Based Escalation

If an alert remains **unacknowledged** for:
- **NOTIFY:** 5 minutes → auto-escalate to ALERT
- **ALERT:** 3 minutes → auto-escalate to EMERGENCY
- **EMERGENCY:** 1 minute → trigger secondary notification (SMS/alarm if configured)

---

## 6. Error Handling & Graceful Degradation

| Failure | Degradation Strategy |
|---------|---------------------|
| **Gemma inference fails** (Stage 2/4) | Fall back to YOLO-only detection with elevated sensitivity; alert operators that reasoning is offline |
| **Knowledge graph unavailable** | Skip Stage 3 context enrichment; all detections pass through to Stage 4 with `context: UNKNOWN` |
| **Camera feed drops** | Log disconnection, display "Camera Offline" on dashboard, alert if >30 seconds |
| **Dashboard disconnected** | Pipeline continues logging alerts to SQLite; dashboard auto-reconnects and pulls missed alerts |
| **Disk full** | Circular log with 7-day retention; oldest evidence archives deleted first; critical alerts preserved indefinitely |
| **Model outputs invalid JSON** | Retry once with simplified prompt; if still fails, log raw output and pass candidate with `reasoning: PARSE_ERROR` |
| **Pipeline latency >2s** | Bypass Stage 4 (CoT), treat as LessThink; log performance warning |

---

## 7. Security & Privacy Design

| Concern | Design Decision |
|---------|----------------|
| **Data sovereignty** | All processing on-device. No network egress for video/inference data. |
| **No face recognition** | System identifies people by behavior and clothing, never by biometrics. YOLO class = "person", not identity. |
| **Evidence retention** | Configurable. Default: 7 days for dismissed alerts, 90 days for confirmed alerts, indefinite for EMERGENCY. |
| **Dashboard access** | Local network only (no port forwarding to internet). Basic auth via environment variable. |
| **Evidence export** | Manual operator action only. Exported as self-contained HTML (frame + reasoning). No auto-sharing. |
| **Bias audit** | Every alert includes a behavioral-only justification. System never stores or reasons about demographic attributes. |
| **Operator privacy** | Dashboard does not track operator identity in feedback (anonymous TP/FP feedback). |

---

## 8. Configuration Design

All tunable parameters in a single `config.py`:

```python
class SafetyChainConfig:
    # ── Stage 1: PERCEIVE ──
    YOLO_MODEL_PATH: str = "models/yolov8n.onnx"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35
    YOLO_CLASSES_OF_INTEREST: list = [
        "person", "car", "truck", "knife", 
        "scissors", "backpack", "fire"
    ]
    MOTION_DELTA_THRESHOLD: float = 0.15
    FRAME_SKIP: int = 2  # Process every Nth frame
    
    # ── Stage 2: DESCRIBE ──
    GEMMA_MODEL: str = "gemma-4-e2b"  # or API endpoint
    GEMMA_MAX_TOKENS: int = 512
    GEMMA_TEMPERATURE: float = 0.1  # Low = deterministic
    
    # ── Stage 3: CONTEXTUALIZE ──
    KNOWLEDGE_GRAPH_DB: str = "data/knowledge_graph.db"
    ZONES_CONFIG: str = "data/zones.json"
    FP_HISTORY_WINDOW_DAYS: int = 30
    
    # ── Stage 4: VERIFY ──
    ZEROTHINK_CLASSES: list = ["knife", "fire"]  # Skip CoT
    ZEROTHINK_ZONE_TYPES: list = ["school"]       # School = always critical
    ZEROTHINK_CONFIDENCE: float = 0.95
    FULLTHINK_TIMEOUT_MS: int = 500
    
    # ── Stage 5: ACT ──
    ESCALATION_UNACK_NOTIFY_MIN: int = 5
    ESCALATION_UNACK_ALERT_MIN: int = 3
    EVIDENCE_RETENTION_DAYS_DISMISSED: int = 7
    EVIDENCE_RETENTION_DAYS_CONFIRMED: int = 90
    
    # ── Dashboard ──
    DASHBOARD_PORT: int = 8000
    DASHBOARD_HOST: str = "0.0.0.0"
    MAX_ALERTS_DISPLAYED: int = 50
    
    # ── Demo ──
    DEMO_VIDEO_DIR: str = "demo/videos/"
    DEMO_MODE: bool = False  # True = use pre-recorded videos
```

---

## 9. Demo Scenario Design

### Scenario A: Vehicle Break-in (FullThink path)

```
TIME: 03:42 AM | ZONE: Parking Lot (Zone A) | STRATEGY: FullThink

Frame 1 (t=0s):
  YOLO → person detected (0.87), near vehicle
  Gate → PASS (person in parking lot after hours)

Frame 1 → Gemma DESCRIBE:
  "Adult in dark hoodie, crouching near driver-side door 
   of silver sedan, holding slim metallic object"
  Suspiciousness: ALARMING

Context Engine:
  Zone A active hours: 06:00-23:00 → OUTSIDE
  No badge scan in 2 hours → NO AUTHORIZED PERSON
  No scheduled maintenance → SUPPORTS ANOMALY
  Known FP history: 0 for this angle → NO SUPPRESSION

CoT Verification (FullThink, 5 steps):
  Step 1: Visual clear, person + tool at vehicle ✅
  Step 2: After hours, unauthorized ✅  
  Step 3: Alt hypotheses — owner lockout? (unlikely at 3:42AM) ⚠️
  Step 4: HIGH severity, INVESTIGATE urgency ✅
  Step 5: CONFIRMED_ANOMALY at 89% ✅

Alert:
  Severity: 🟠 ALERT
  Title: "Possible Vehicle Break-in"
  Action: "Alert patrol ext. 2200, do NOT approach alone"
  SOP: SOP-014 loaded
  Total latency: ~670ms
```

### Scenario B: School Intrusion (ZeroThink path)

```
TIME: 10:15 AM | ZONE: School Perimeter (Zone B) | STRATEGY: ZeroThink

Frame 1 (t=0s):
  YOLO → person detected (0.91), at fence line
  Gate → PASS (person at school perimeter fence)

Depth Router:
  Zone type = "school" ∈ ZEROTHINK_ZONE_TYPES
  Detection confidence (0.91) ≥ 0.90
  → TRIGGER ZEROTHINK

ZeroThink (bypass Stages 2-4):
  Classification: CONFIRMED_ANOMALY
  Confidence: 0.99
  Severity: CRITICAL
  Urgency: EMERGENCY
  Reasoning: "School perimeter breach during school hours.
              Automatic critical escalation per policy."

Alert:
  Severity: 🔴 EMERGENCY
  Title: "⚠️ PERIMETER BREACH — School Zone"
  Action: "DISPATCH IMMEDIATELY. Lock down building."
  Total latency: ~70ms
```

### Dashboard Contrast

The demo shows both alerts side-by-side on the dashboard:

| | Vehicle Break-in | School Intrusion |
|---|---|---|
| **Strategy** | FullThink (5-step chain) | ZeroThink (instant) |
| **Latency** | ~670ms | ~70ms |
| **Reasoning visible** | Full 5 steps, expandable | Single line: "ZeroThink: auto-escalation" |
| **Severity** | 🟠 ALERT | 🔴 EMERGENCY |
| **Key insight for judges** | "Look how deeply it reasons" | "But when lives are at stake, it doesn't waste a millisecond" |

---

## 10. Testing Strategy

### Unit Tests

| Test | Stage | Validates |
|------|-------|-----------|
| `test_yolo_detects_person` | 1 | YOLO finds person in test frame |
| `test_gate_passes_afterhours` | 1 | Person in restricted zone after hours → PASS |
| `test_gate_drops_empty_frame` | 1 | Empty frame → no candidate |
| `test_scene_description_schema` | 2 | Gemma output parses to valid SceneDescription |
| `test_context_supports_anomaly` | 3 | After-hours parking lot → SUPPORTS |
| `test_context_refutes_anomaly` | 3 | Construction zone during work hours → REFUTES |
| `test_context_known_fp` | 3 | Known tree-branch camera → SUPPRESS |
| `test_zerothink_school` | 4 | School zone + person → ZeroThink EMERGENCY |
| `test_fullthink_produces_5_steps` | 4 | Ambiguous case → 5 reasoning steps |
| `test_escalation_matrix` | 5 | HIGH severity + 85% confidence → ALERT |
| `test_feedback_updates_kg` | Loop | FP feedback → knowledge graph updated |

### Integration Tests

| Test | Validates |
|------|-----------|
| `test_pipeline_vehicle_breakin` | Full e2e: video → alert with 5-step chain |
| `test_pipeline_school_intrusion` | Full e2e: video → ZeroThink EMERGENCY |
| `test_pipeline_normal_activity` | Normal video → zero alerts |
| `test_dashboard_receives_alert` | Pipeline → WebSocket → dashboard client |
| `test_latency_under_budget` | FullThink pipeline < 700ms, ZeroThink < 100ms |
