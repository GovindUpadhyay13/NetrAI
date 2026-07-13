# ⛓️ SafetyChain

**A 5-Stage Chain-of-Thought Verification Pipeline for AI-Powered Public Safety**

> 🏆 Built for Google Gemma Hackathon — Track 2: AI for Public Safety

---

## 🎯 Problem

Traditional public safety monitoring suffers from **"alarm fatigue"** due to passive, high-noise surveillance that lacks semantic context. Current systems face two critical failures:

1. **Cloud-based processing** creates latency and privacy risks
2. **"Black-box" pattern matching** lacks the reasoning layer needed for forensic-grade verification

## 💡 Solution

SafetyChain introduces a **5-stage reasoning pipeline** that transforms raw video into explainable, forensic-grade safety alerts:

```
📹 Video → ⚡ PERCEIVE → 🧠 DESCRIBE → 🌍 CONTEXTUALIZE → ⛓️ VERIFY → 🚨 ACT
           (YOLOv8)     (Gemma VLM)    (Knowledge Graph)   (CoT Reasoning) (Dashboard)
```

Every alert ships with a **transparent chain of evidence** — not just "something happened," but a structured, 5-step verification trace that an operator can trust.

---

## 🏗️ Architecture

| Stage | Component | Purpose | Latency |
|-------|-----------|---------|---------|
| **1. PERCEIVE** | YOLOv8n | Object detection + anomaly gate | ~20ms |
| **2. DESCRIBE** | Gemma 4 VLM | PerCoAct scene understanding | ~200ms |
| **3. CONTEXTUALIZE** | SQLite KG | Zone norms + temporal + history | ~100ms |
| **4. VERIFY** | Gemma CoT | 5-step chain-of-thought reasoning | ~300ms |
| **5. ACT** | Dashboard | Escalation + evidence packaging | ~50ms |

**Total pipeline latency: <700ms** (FullThink path)

### Adaptive Reasoning Depth

| Strategy | When | Latency | Description |
|----------|------|---------|-------------|
| ⚡ **ZeroThink** | Weapon/fire/school breach | <50ms | Skip reasoning, instant EMERGENCY |
| 🟠 **LessThink** | High confidence + high severity | ~150ms | 3-step abbreviated chain |
| 🟡 **FullThink** | Ambiguous situations | ~300ms | Full 5-step verification |
| ⚪ **MoreThink** | Novel/unknown patterns | ~500ms | Extended + extra hypotheses |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google AI Studio API key (for Gemma)

### Setup

```bash
# Install dependencies
pip install -r safetychain/requirements.txt

# Set your API key
set GOOGLE_API_KEY=your_api_key_here

# Run the demo
python -m safetychain.demo.run_demo
```

The demo will:
1. Generate synthetic test videos
2. Start the dashboard at `http://localhost:8000`
3. Run **Scenario A**: Vehicle Break-in (FullThink → 5-step reasoning)
4. Run **Scenario B**: School Intrusion (ZeroThink → instant EMERGENCY)

### Without API Key

The system works without a Gemma API key using **rule-based fallback** for Stages 2 and 4. You'll see YOLO-only detection with deterministic reasoning chains.

---

## 📊 Dashboard

The real-time Security Operations Center dashboard features:

- **Alert Feed** — Severity-coded cards (🔴 EMERGENCY, 🟠 ALERT, 🟡 NOTIFY, ⚫ LOG)
- **Evidence Viewer** — Annotated frames with detection bounding boxes
- **Reasoning Chain** — Expandable 5-step verification trace
- **Pipeline Indicator** — Live stage progression visualization
- **Feedback Loop** — TP/FP buttons that update the knowledge graph

---

## 🔑 Key Innovation: Chain-of-Thought Verification

### FullThink 5-Step Chain

```
Step 1: Evidence Quality → Is the visual evidence clear?
Step 2: Context Alignment → Is this abnormal for this place/time?
Step 3: Alternative Hypotheses → Can a benign explanation cover ALL evidence?
Step 4: Severity Assessment → How bad is this if real?
Step 5: Final Verdict → Classification + confidence + recommended action
```

### Why It Matters

- **Anti-hallucination**: Model must argue against its own conclusion (Step 3)
- **Explainability**: Every alert includes a forensic evidence chain
- **Accountability**: Chain IDs enable complete audit trails
- **Adaptive compute**: ZeroThink for emergencies, FullThink for ambiguity

---

## 📁 Project Structure

```
safetychain/
├── main.py                     # Pipeline orchestrator
├── config.py                   # All configuration
├── models.py                   # Data models
├── stage1_perceive/            # YOLOv8n + Anomaly Gate
├── stage2_describe/            # Gemma VLM + PerCoAct prompt
├── stage3_contextualize/       # Knowledge Graph + Context Engine
├── stage4_verify/              # Adaptive CoT Reasoning
├── stage5_act/                 # Alert Manager + Evidence Packager
├── dashboard/                  # FastAPI + WebSocket + Frontend
├── demo/                       # Synthetic videos + demo runner
├── data/                       # Zone configs + SOPs
└── utils/                      # Logger + Frame utilities
```

---

## 🔒 Privacy by Architecture

- **All processing on-device** — no cloud calls, no telemetry
- **No face recognition** — identifies people by behavior, not biometrics
- **No demographic analysis** — behavioral-only justification
- **Local network only** — dashboard accessible on LAN
- **Configurable retention** — 7 days dismissed, 90 days confirmed

---

## 🛡️ Graceful Degradation

| Failure | Fallback |
|---------|----------|
| Gemma unavailable | YOLO-only detection + rule-based reasoning |
| Knowledge graph down | Skip context enrichment, pass-through to verification |
| Camera feed drops | Log disconnection, alert operator after 30s |
| Invalid model output | Retry once, then log with PARSE_ERROR flag |
| Pipeline latency >2s | Bypass CoT, treat as LessThink |

---

## 📄 License

Built for the Google Gemma Hackathon. See hackathon rules for usage terms.
