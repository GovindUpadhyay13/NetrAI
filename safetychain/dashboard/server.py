"""SafetyChain — Dashboard Server

FastAPI server with REST endpoints and WebSocket for real-time alerts.

Implementation plan ref: Phase 8
Design ref: Section 3 (API Design)

REST Endpoints:
  GET  /api/alerts              — List recent alerts
  GET  /api/alerts/{alert_id}   — Get full alert with reasoning
  POST /api/alerts/{alert_id}/feedback — Operator marks TP/FP
  GET  /api/pipeline/status     — Pipeline health check
  GET  /api/stats               — Dashboard statistics
  GET  /api/zones               — List configured zones

WebSocket:
  ws://localhost:8000/ws/alerts  — Real-time alert streaming
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ..main import SafetyChainPipeline
from ..config import SafetyChainConfig
from ..models import Alert

app = FastAPI(title="SafetyChain Dashboard", version="1.0.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (set by run_server)
pipeline: SafetyChainPipeline = None

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time alert broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


def alert_to_dict(alert: Alert) -> dict:
    """Convert an Alert dataclass to a JSON-serializable dict."""
    return {
        "alert_id": alert.alert_id,
        "chain_id": alert.chain_id,
        "timestamp": alert.timestamp.isoformat(),
        "severity": alert.severity,
        "title": alert.title,
        "zone_name": alert.zone_name,
        "confidence": alert.confidence,
        "frame_b64": alert.frame_b64,
        "classification": alert.verdict.classification,
        "reasoning_strategy": alert.verdict.reasoning_strategy,
        "reasoning_chain": [
            {
                "step": step.step_number,
                "title": step.title,
                "content": step.content,
                "passed": step.passed,
            }
            for step in alert.verdict.reasoning_chain
        ],
        "alternative_hypotheses": alert.verdict.alternative_hypotheses,
        "recommended_action": alert.verdict.recommended_action,
        "consequences_if_ignored": alert.verdict.consequences_if_ignored,
        "reasoning_latency_ms": alert.verdict.reasoning_latency_ms,
        "context_summary": alert.context_summary,
        "sop": alert.sop,
        "contacts": alert.contacts,
        "status": alert.status,
        "operator_feedback": alert.operator_feedback,
    }


# ═══ REST Endpoints ═══

@app.get("/api/alerts")
async def get_alerts(limit: int = 20, severity: str = None):
    """List recent alerts with optional severity filter."""
    alerts = pipeline.alert_manager.get_active_alerts(limit)
    if severity:
        severity_filter = set(severity.upper().split(","))
        alerts = [a for a in alerts if a.severity in severity_filter]
    return [alert_to_dict(a) for a in alerts]


@app.get("/api/alerts/{alert_id}")
async def get_alert(alert_id: str):
    """Get full alert details including reasoning chain."""
    alert = pipeline.alert_manager.get_alert(alert_id)
    if not alert:
        return {"error": "Alert not found"}, 404
    return alert_to_dict(alert)


@app.post("/api/alerts/{alert_id}/feedback")
async def post_feedback(alert_id: str, body: dict):
    """Submit operator feedback (true_positive/false_positive)."""
    feedback = body.get("feedback")
    note = body.get("note")

    if feedback not in ("true_positive", "false_positive"):
        return {"error": "feedback must be 'true_positive' or 'false_positive'"}, 400

    success = pipeline.process_feedback(alert_id, feedback, note)

    if success:
        # Broadcast status update
        await manager.broadcast({
            "type": "alert_updated",
            "data": {
                "alert_id": alert_id,
                "status": "resolved",
                "feedback": feedback,
            }
        })
        return {"status": "updated"}

    return {"error": "Alert not found"}, 404


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    success = pipeline.alert_manager.acknowledge_alert(alert_id)
    if success:
        await manager.broadcast({
            "type": "alert_updated",
            "data": {
                "alert_id": alert_id,
                "status": "acknowledged",
            }
        })
        return {"status": "acknowledged"}
    return {"error": "Alert not found"}, 404


@app.post("/api/alerts/{alert_id}/dismiss")
async def dismiss_alert(alert_id: str):
    """Dismiss an alert."""
    success = pipeline.alert_manager.dismiss_alert(alert_id)
    if success:
        await manager.broadcast({
            "type": "alert_updated",
            "data": {
                "alert_id": alert_id,
                "status": "dismissed",
            }
        })
        return {"status": "dismissed"}
    return {"error": "Alert not found"}, 404


@app.get("/api/pipeline/status")
async def get_pipeline_status():
    """Get pipeline health status."""
    status = pipeline.get_status()
    return {
        "stage": status["active_stage"],
        "fps": round(status["fps"], 1),
        "alerts_today": status["alerts_today"],
        "frames_processed": status["frames_processed"],
        "pipeline_active": status["pipeline_active"],
    }


@app.get("/api/stats")
async def get_stats():
    """Get dashboard statistics."""
    return pipeline.alert_manager.get_stats()


@app.get("/api/zones")
async def get_zones():
    """List configured zones."""
    zones_path = pipeline.config.get_zones_path(pipeline.base_dir)
    if os.path.exists(zones_path):
        with open(zones_path) as f:
            data = json.load(f)
        return data.get("zones", [])
    return []


# ═══ WebSocket ═══

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time alert streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive — listen for client messages
            data = await websocket.receive_text()
            # Client can send ping/pong or commands
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_alert(alert: Alert):
    """Broadcast a new alert to all WebSocket clients."""
    await manager.broadcast({
        "type": "new_alert",
        "data": alert_to_dict(alert),
    })


async def broadcast_pipeline_status(status: dict):
    """Broadcast pipeline status update."""
    await manager.broadcast({
        "type": "pipeline_status",
        "data": {
            "active_stage": status.get("active_stage", "idle"),
            "cumulative_latency_ms": status.get("cumulative_latency_ms", 0),
            "fps": round(status.get("fps", 0), 1),
        }
    })


# ═══ Static Files ═══

# Serve the dashboard frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard page."""
    return FileResponse(os.path.join(static_dir, "index.html"))


# Mount static files AFTER the root route
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def run_server(pipeline_instance: SafetyChainPipeline, host: str = "0.0.0.0", port: int = 8000):
    """Start the dashboard server.
    
    Args:
        pipeline_instance: Initialized SafetyChainPipeline
        host: Bind host
        port: Bind port
    """
    global pipeline
    pipeline = pipeline_instance

    uvicorn.run(app, host=host, port=port, log_level="info")
