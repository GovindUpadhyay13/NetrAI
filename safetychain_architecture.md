# ⛓️ SafetyChain — System Architecture

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph INPUT["📥 Input Sources"]
        CAM["📹 Camera Feed<br/>(RTSP / USB / File)"]
        SENS["🌡️ Sensors<br/>(PIR / Door / Temp)"]
    end

    subgraph S1["⚡ STAGE 1 — PERCEIVE  (~20ms)"]
        YOLO["YOLOv8n<br/>Object Detection<br/>(INT8 quantized)"]
        GATE["Anomaly Gate<br/>Threshold Filter"]
    end

    subgraph S2["🧠 STAGE 2 — DESCRIBE  (~200ms)"]
        VLM["Gemma 4 E2B<br/>Multimodal VLM<br/>(INT4 quantized)"]
        PROMPT["PerCoAct Prompt<br/>Template Engine"]
    end

    subgraph S3["🌍 STAGE 3 — CONTEXTUALIZE  (~100ms)"]
        KG["Knowledge Graph<br/>(SQLite)"]
        TEMP["Temporal Engine<br/>(Time / Calendar)"]
        RAG["On-Device RAG<br/>(Gecko Embeddings)"]
        CTX["Context Aggregator"]
    end

    subgraph S4["⛓️ STAGE 4 — VERIFY  (~300ms)"]
        DEPTH["Depth Router<br/>Zero/Less/Full/More"]
        COT["CoT Reasoning<br/>Gemma 4 E2B"]
        ZERO["ZeroThink<br/>Instant Alert"]
    end

    subgraph S5["🚨 STAGE 5 — ACT  (~50ms)"]
        ESC["Escalation Engine<br/>LOG/NOTIFY/ALERT/EMERGENCY"]
        EVID["Evidence Packager<br/>Frame + Chain + Context"]
        DASH["Operator Dashboard<br/>(WebSocket)"]
    end

    subgraph LEARN["🔄 Feedback Loop"]
        FB["Operator Feedback<br/>TP / FP"]
        MEM["Anomaly Memory<br/>KG Update"]
    end

    CAM --> YOLO
    SENS --> GATE
    YOLO --> GATE

    GATE -->|"Anomaly<br/>Candidate"| VLM
    GATE -->|"Nothing<br/>interesting"| DROP["🗑️ Drop<br/>(95% of frames)"]
    VLM --> PROMPT
    PROMPT -->|"Scene<br/>Description"| CTX

    KG --> CTX
    TEMP --> CTX
    RAG --> CTX

    CTX -->|"Context<br/>Report"| DEPTH
    DEPTH -->|"CRITICAL"| ZERO
    DEPTH -->|"Other"| COT

    ZERO --> ESC
    COT -->|"Verdict"| ESC

    ESC --> EVID
    EVID --> DASH

    DASH --> FB
    FB --> MEM
    MEM --> KG

    style S1 fill:#1a1a2e,color:#e0e0e0,stroke:#e94560
    style S2 fill:#1a1a2e,color:#e0e0e0,stroke:#f5a623
    style S3 fill:#1a1a2e,color:#e0e0e0,stroke:#50fa7b
    style S4 fill:#1a1a2e,color:#e0e0e0,stroke:#8be9fd
    style S5 fill:#1a1a2e,color:#e0e0e0,stroke:#bd93f9
    style LEARN fill:#1a1a2e,color:#e0e0e0,stroke:#ffb86c
    style DROP fill:#333,color:#888,stroke:#555
```

---

## 2. Data Flow — Inter-Stage Communication

```mermaid
sequenceDiagram
    participant C as 📹 Camera
    participant S1 as ⚡ PERCEIVE
    participant S2 as 🧠 DESCRIBE
    participant S3 as 🌍 CONTEXTUALIZE
    participant S4 as ⛓️ VERIFY
    participant S5 as 🚨 ACT
    participant D as 📊 Dashboard
    participant O as 👤 Operator

    C->>S1: Video frame (30fps continuous)
    Note over S1: YOLOv8n detection<br/>~20ms/frame

    alt Nothing detected
        S1-->>S1: Drop frame (95%+ of all frames)
    else Anomaly candidate
        S1->>S2: AnomalyCandidate {type, confidence, bbox, timestamp}
        S1->>S2: Annotated frame (JPEG bytes)
        
        Note over S2: Gemma 4 VLM<br/>~200ms

        S2->>S3: SceneDescription {perception, cognition, suspiciousness}
        
        Note over S3: KG + Temporal + RAG<br/>~100ms

        S3->>S4: ContextReport {supports_anomaly, zone_norms, history, protocols}

        alt CRITICAL (weapon/fire/intrusion at school)
            Note over S4: ZeroThink<br/><50ms
            S4->>S5: Verdict {EMERGENCY, 99%, skip_reasoning}
        else Ambiguous
            Note over S4: FullThink CoT<br/>~300ms
            S4->>S5: Verdict {classification, confidence, reasoning_chain, chain_id}
        end

        S5->>D: Alert {verdict, annotated_frame, reasoning_chain, SOP, contacts}
        D->>O: Visual alert + reasoning display
        
        O->>D: Feedback (True Positive / False Positive)
        D->>S3: Update knowledge graph with feedback
    end
```

---

## 3. Stage 1 — PERCEIVE: Detection Architecture

```mermaid
graph LR
    subgraph "Frame Ingestion"
        SRC["Video Source<br/>(RTSP/USB/File)"]
        DEC["Frame Decoder<br/>(OpenCV)"]
        BUF["Frame Buffer<br/>(Ring Buffer, 30 frames)"]
    end

    subgraph "Object Detection"
        YOLO["YOLOv8n<br/>(ONNX Runtime, INT8)"]
        CLASSES["Class Filter<br/>person, vehicle, weapon,<br/>fire, backpack, knife"]
    end

    subgraph "Anomaly Gate"
        THRESH["Confidence Threshold<br/>(configurable per class)"]
        ZONE_CHK["Zone Boundary Check<br/>(is detection in monitored area?)"]
        MOTION["Motion Delta<br/>(frame diff for rapid changes)"]
        DECIDE{Pass?}
    end

    SRC --> DEC --> BUF
    BUF --> YOLO --> CLASSES
    CLASSES --> THRESH
    THRESH --> ZONE_CHK
    ZONE_CHK --> MOTION
    MOTION --> DECIDE

    DECIDE -->|"Yes"| OUT["AnomalyCandidate<br/>→ Stage 2"]
    DECIDE -->|"No"| LOG["Silent Log<br/>(for forensic review)"]

    style OUT fill:#e94560,color:#fff
    style LOG fill:#333,color:#888
```

---

## 4. Stage 2 — DESCRIBE: VLM Scene Understanding

```mermaid
graph TB
    subgraph "Input"
        FRAME["Annotated Frame<br/>(YOLO bounding boxes)"]
        CAND["AnomalyCandidate<br/>{type, confidence, bbox}"]
    end

    subgraph "Prompt Construction"
        TMPL["PerCoAct Template<br/>(perception + cognition)"]
        INJECT["Variable Injection<br/>{bbox, class, confidence}"]
    end

    subgraph "VLM Inference"
        GEMMA["Gemma 4 E2B<br/>Multimodal (INT4)<br/>~200ms"]
    end

    subgraph "Output Parsing"
        PARSE["JSON Parser<br/>(structured output)"]
        VALID["Schema Validator<br/>(ensure all fields present)"]
    end

    FRAME --> INJECT
    CAND --> INJECT
    TMPL --> INJECT
    INJECT --> GEMMA
    GEMMA --> PARSE --> VALID

    VALID --> SD["SceneDescription<br/>{perception, cognition,<br/>suspiciousness}"]

    style SD fill:#f5a623,color:#000
```

### PerCoAct Prompt Structure

```mermaid
graph LR
    subgraph "Prompt Template"
        P["🔍 PERCEPTION<br/>• Scene environment<br/>• People (age, clothing, posture)<br/>• Objects (unusual items)<br/>• Text (signs, plates)"]
        C["🧠 COGNITION<br/>• Activity identification<br/>• Norm violations<br/>• Object-person relationships<br/>• Suspiciousness rating"]
    end

    P --> C --> OUT["Structured JSON Output"]

    style P fill:#2d3436,color:#dfe6e9
    style C fill:#2d3436,color:#dfe6e9
```

---

## 5. Stage 3 — CONTEXTUALIZE: Knowledge Graph & Context Engine

### Knowledge Graph Schema

```mermaid
erDiagram
    SITE ||--o{ ZONE : contains
    ZONE ||--o{ NORM : has
    ZONE ||--o{ HISTORY : logs
    ZONE ||--o{ CAMERA : monitors
    ZONE ||--o{ SOP : applies

    SITE {
        string site_id PK
        string name
        string address
        string timezone
    }

    ZONE {
        string zone_id PK
        string site_id FK
        string name
        string type "parking|school|corridor|loading_dock|entrance"
        json boundary_coords
    }

    NORM {
        int norm_id PK
        string zone_id FK
        string norm_type "occupancy|time_window|expected_objects|behaviors"
        string rule_description
        string active_hours
        json parameters
    }

    HISTORY {
        int event_id PK
        string zone_id FK
        datetime timestamp
        string event_type "alert|false_positive|true_positive"
        string description
        string chain_id
        bool was_false_positive
    }

    CAMERA {
        string camera_id PK
        string zone_id FK
        string name
        string stream_url
        json field_of_view
        json adjacent_cameras
    }

    SOP {
        string sop_id PK
        string zone_id FK
        string alert_type
        string title
        text procedure
        json contacts
    }
```

### Context Aggregation Flow

```mermaid
graph TB
    SD["SceneDescription<br/>(from Stage 2)"] --> AGG

    subgraph "Context Sources"
        KG["Knowledge Graph<br/>Zone norms & rules"]
        TIME["Temporal Engine<br/>Current time, day,<br/>calendar, season"]
        HIST["History Lookup<br/>Known FP patterns,<br/>recent events"]
        RAGS["Protocol RAG<br/>SOPs, floor plans,<br/>contacts"]
    end

    KG --> AGG["Context Aggregator"]
    TIME --> AGG
    HIST --> AGG
    RAGS --> AGG

    AGG --> CR["ContextReport"]

    CR --> SUPPORT{"Supports<br/>Anomaly?"}
    SUPPORT -->|"SUPPORTS"| S4A["→ Stage 4<br/>(elevated confidence)"]
    SUPPORT -->|"NEUTRAL"| S4B["→ Stage 4<br/>(unchanged)"]
    SUPPORT -->|"REFUTES"| KILL["🗑️ Suppress Alert<br/>(known false positive)"]

    style KILL fill:#333,color:#888
    style S4A fill:#e94560,color:#fff
```

---

## 6. Stage 4 — VERIFY: Adaptive CoT Reasoning

### Reasoning Depth Router

```mermaid
flowchart TD
    INPUT["Anomaly Candidate +<br/>Scene Description +<br/>Context Report"] --> ASSESS

    ASSESS{"Confidence &<br/>Severity?"}

    ASSESS -->|">95% AND<br/>CRITICAL<br/>(weapon/fire/school breach)"| ZT["⚡ ZeroThink<br/>Skip reasoning<br/>< 50ms"]
    ASSESS -->|">80% AND<br/>HIGH"| LT["🟠 LessThink<br/>3-step abbreviated<br/>~150ms"]
    ASSESS -->|"50-80% OR<br/>ambiguous"| FT["🟡 FullThink<br/>5-step complete chain<br/>~300ms"]
    ASSESS -->|"30-50% OR<br/>novel/unknown"| MT["⚪ MoreThink<br/>Extended + extra hypotheses<br/>~500ms"]

    ZT --> VERDICT
    LT --> VERDICT
    FT --> VERDICT
    MT --> VERDICT

    VERDICT["Verdict<br/>{classification, confidence,<br/>reasoning_chain, severity,<br/>recommended_action,<br/>chain_id}"]

    VERDICT --> S5["→ Stage 5: ACT"]

    style ZT fill:#e94560,color:#fff
    style LT fill:#f5a623,color:#000
    style FT fill:#f1c40f,color:#000
    style MT fill:#636e72,color:#fff
```

### FullThink 5-Step Chain

```mermaid
graph TD
    E["Evidence Summary<br/>Visual + Context"] --> STEP1

    STEP1["STEP 1: Evidence Consistency<br/>• Visual signals consistent?<br/>• Matches known anomaly pattern?<br/>• Per-source confidence scores"] --> STEP2

    STEP2["STEP 2: Context Check<br/>• Abnormal for location + time?<br/>• Knowledge graph supports/refutes?<br/>• Known false-positive match?"] --> STEP3

    STEP3["STEP 3: Alternative Hypotheses<br/>• Generate 2-3 benign explanations<br/>• Can any explain ALL evidence?<br/>• Rate each hypothesis"] --> STEP4

    STEP4["STEP 4: Severity Assessment<br/>• Threat level (LOW→CRITICAL)<br/>• Urgency (MONITOR→EMERGENCY)<br/>• Consequences if ignored"] --> STEP5

    STEP5["STEP 5: Final Verdict<br/>• Classification<br/>• Confidence %<br/>• Recommended action<br/>• Evidence chain ID"]

    STEP5 --> V["Verdict Output"]

    style STEP1 fill:#2d3436,color:#dfe6e9
    style STEP2 fill:#2d3436,color:#dfe6e9
    style STEP3 fill:#2d3436,color:#dfe6e9
    style STEP4 fill:#2d3436,color:#dfe6e9
    style STEP5 fill:#2d3436,color:#dfe6e9
    style V fill:#e94560,color:#fff
```

---

## 7. Stage 5 — ACT: Alert Escalation State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle

    Idle --> Evaluating : Verdict received

    Evaluating --> LOG : confidence < 40%
    Evaluating --> NOTIFY : 40% ≤ confidence < 70%
    Evaluating --> ALERT : 70% ≤ confidence < 90%
    Evaluating --> EMERGENCY : confidence ≥ 90% OR ZeroThink

    LOG --> Idle : Auto-archive
    
    NOTIFY --> Idle : Operator dismisses
    NOTIFY --> ALERT : Escalation (time-based or corroborated)
    
    ALERT --> Idle : Operator acknowledges
    ALERT --> EMERGENCY : Escalation (new evidence)
    ALERT --> LOG : Operator marks as FP
    
    EMERGENCY --> Idle : Resolved by operator
    EMERGENCY --> ALERT : Downgraded after investigation

    LOG --> FeedbackLoop : Periodic review
    ALERT --> FeedbackLoop : Operator feedback
    EMERGENCY --> FeedbackLoop : Post-incident review
    FeedbackLoop --> KnowledgeGraphUpdate : TP/FP recorded
    KnowledgeGraphUpdate --> [*]
```

---

## 8. Anomaly Memory — Feedback Loop Architecture

```mermaid
graph LR
    subgraph "Runtime"
        ALERT["Alert Displayed"]
        OP["👤 Operator"]
    end

    subgraph "Feedback Processing"
        FB["Feedback Ingestion<br/>(TP / FP + optional note)"]
        CLASS["Classifier<br/>• Environmental FP<br/>• Equipment FP<br/>• Behavioral FP<br/>• True Positive"]
    end

    subgraph "Knowledge Update"
        KG_UP["Knowledge Graph<br/>Update"]
        PATTERN["Pattern Store<br/>(scene_hash → verdict)"]
        THRESH["Threshold Tuner<br/>(per-zone, per-class)"]
    end

    ALERT --> OP
    OP -->|"Feedback"| FB
    FB --> CLASS
    CLASS -->|"Environmental FP<br/>(tree, shadow, animal)"| KG_UP
    CLASS -->|"Equipment FP<br/>(camera glitch)"| THRESH
    CLASS -->|"Behavioral FP<br/>(authorized person)"| PATTERN
    CLASS -->|"True Positive"| KG_UP

    KG_UP --> S3["Stage 3<br/>Context Engine<br/>(enriched)"]
    PATTERN --> S3
    THRESH --> S1["Stage 1<br/>Anomaly Gate<br/>(tuned)"]

    style S3 fill:#50fa7b,color:#000
    style S1 fill:#e94560,color:#fff
```

---

## 9. Deployment Architecture — RPi 5 Edge Device

```mermaid
graph TB
    subgraph "Edge Device: Raspberry Pi 5 (~$110)"
        subgraph "Hardware"
            CPU["BCM2712<br/>Cortex-A76 Quad-core"]
            RAM_HW["8GB LPDDR4X"]
            HAILO["Hailo-8L NPU<br/>13 TOPS"]
            STORAGE["128GB microSD<br/>(OS + Models + Data)"]
        end

        subgraph "Model Memory Budget"
            YOLO_M["YOLOv8n (INT8)<br/>~50MB RAM"]
            GEMMA_M["Gemma 4 E2B (INT4)<br/>~2.5GB RAM"]
            SHIELD_M["ShieldGemma<br/>~1.2GB RAM"]
            GECKO_M["Gecko Embeddings<br/>~300MB RAM"]
            OS_M["OS + App<br/>~1GB RAM"]
        end

        subgraph "Software Runtime"
            PIPE["SafetyChain Pipeline<br/>(Python)"]
            API["FastAPI Server<br/>:8000"]
            WS["WebSocket Server<br/>:8001"]
            DB["SQLite<br/>(Knowledge Graph)"]
        end
    end

    subgraph "Peripherals"
        CAM_P["IP Camera<br/>(RTSP, PoE)"]
        DISPLAY["Monitor<br/>(HDMI, optional)"]
    end

    subgraph "Network"
        LAN["Local Network<br/>(LAN only — no cloud)"]
        BROWSER["Any Browser<br/>Dashboard Client"]
    end

    CAM_P -->|"RTSP"| PIPE
    PIPE --> API
    API --> WS
    WS -->|"WebSocket"| BROWSER
    PIPE --> DB

    HAILO -.->|"Accelerates"| YOLO_M
    CPU -.->|"Runs"| GEMMA_M
    CPU -.->|"Runs"| SHIELD_M

    style HAILO fill:#50fa7b,color:#000
    style GEMMA_M fill:#f5a623,color:#000
```

### Memory Budget Breakdown

```mermaid
pie title RAM Usage on RPi 5 (8GB)
    "OS + Runtime" : 1.0
    "YOLOv8n" : 0.05
    "Gemma 4 E2B (INT4)" : 2.5
    "ShieldGemma" : 1.2
    "Gecko Embeddings" : 0.3
    "SQLite + RAG Index" : 0.2
    "Frame Buffer" : 0.15
    "Available Headroom" : 2.6
```

---

## 10. Dashboard Component Architecture

```mermaid
graph TB
    subgraph "Backend (FastAPI)"
        REST["REST API<br/>GET /alerts<br/>GET /alerts/:id<br/>POST /feedback"]
        WSS["WebSocket Server<br/>/ws/alerts"]
        STORE["Alert Store<br/>(SQLite)"]
    end

    subgraph "Frontend (Vanilla JS)"
        subgraph "Layout"
            HEAD["Header Bar<br/>System status, pipeline indicator"]
            LEFT["Alert Feed Panel<br/>Scrollable list, severity badges"]
            CENTER["Evidence Panel<br/>Annotated frame viewer"]
            RIGHT["Reasoning Chain Panel<br/>Collapsible 5-step chain"]
            BOTTOM["Action Bar<br/>Acknowledge / Investigate / Dismiss / Report"]
        end

        WSCL["WebSocket Client"]
        RENDER["DOM Renderer"]
    end

    WSS -->|"New Alert JSON"| WSCL
    WSCL --> RENDER
    RENDER --> HEAD
    RENDER --> LEFT
    RENDER --> CENTER
    RENDER --> RIGHT
    RENDER --> BOTTOM

    BOTTOM -->|"Feedback POST"| REST
    REST --> STORE
    LEFT -->|"Click alert"| REST
    REST -->|"Alert details"| CENTER
    REST -->|"Reasoning chain"| RIGHT
```

---

## 11. End-to-End Latency Budget

```mermaid
gantt
    title Pipeline Latency Budget (target: <700ms total)
    dateFormat X
    axisFormat %L ms

    section Stage 1
    YOLO Detection        :s1, 0, 20
    Anomaly Gate           :s1g, 20, 5

    section Stage 2
    Frame Encoding         :s2a, 25, 10
    Gemma VLM Inference    :s2b, 35, 200
    JSON Parsing           :s2c, 235, 5

    section Stage 3
    KG Query               :s3a, 240, 20
    Temporal Check         :s3b, 240, 10
    RAG Retrieval          :s3c, 240, 50
    Context Aggregation    :s3d, 290, 10

    section Stage 4
    Depth Routing          :s4a, 300, 5
    CoT Reasoning (Gemma)  :s4b, 305, 300
    Verdict Generation     :s4c, 605, 10

    section Stage 5
    Escalation Logic       :s5a, 615, 5
    Evidence Packaging     :s5b, 620, 20
    WebSocket Push         :s5c, 640, 10

    section Total
    End-to-end             :total, 0, 670
```

> [!NOTE]
> Stages 3a, 3b, 3c run **in parallel** (async). The Gantt shows wall-clock time.
> ZeroThink path skips Stages 2-4 and completes in **<70ms**.
