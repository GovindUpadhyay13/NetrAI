"""
bus/schemas.py
Pydantic event models matching the pinned surveillance event schema.

Every stage publishes one JSON event conforming to this exact shape:
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
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class StageEnum(str, Enum):
    ANOMALY_DETECTED = "anomaly_detected"
    GESTURE_FLAGGED = "gesture_flagged"
    VLM_ANALYZED = "vlm_analyzed"
    DISPATCHED = "dispatched"
    REID_MATCH = "reid_match"
    TRACE = "trace"


class SeverityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SurveillanceEvent(BaseModel):
    incident_id: str = Field(default_factory=lambda: str(uuid4()))
    camera_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: StageEnum
    anomaly_score: float = 0.0
    anomaly_type: Optional[str] = None
    distress_gesture: bool = False
    vlm_report: Optional[str] = None
    severity: Optional[SeverityEnum] = None
    payload_ref: str = ""

    class Config:
        use_enum_values = True

    def to_redis_dict(self) -> dict:
        """Converts model to flat dict suitable for Redis Streams XADD."""
        return {
            "incident_id": self.incident_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "stage": str(self.stage),
            "anomaly_score": str(self.anomaly_score),
            "anomaly_type": self.anomaly_type or "",
            "distress_gesture": "1" if self.distress_gesture else "0",
            "vlm_report": self.vlm_report or "",
            "severity": str(self.severity) if self.severity else "",
            "payload_ref": self.payload_ref or "",
        }

    @classmethod
    def from_redis_dict(cls, data: dict) -> "SurveillanceEvent":
        """Reconstructs model from Redis Streams flat string mapping."""
        # Convert byte keys/values if received from raw redis
        cleaned = {}
        for k, v in data.items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            val = v.decode() if isinstance(v, bytes) else str(v)
            cleaned[key] = val

        sev = cleaned.get("severity")
        severity_val = SeverityEnum(sev) if sev and sev in [s.value for s in SeverityEnum] else None

        return cls(
            incident_id=cleaned.get("incident_id", str(uuid4())),
            camera_id=cleaned.get("camera_id", "UNKNOWN_CAM"),
            timestamp=cleaned.get("timestamp", datetime.now(timezone.utc).isoformat()),
            stage=StageEnum(cleaned.get("stage", "trace")),
            anomaly_score=float(cleaned.get("anomaly_score", 0.0)),
            anomaly_type=cleaned.get("anomaly_type") or None,
            distress_gesture=cleaned.get("distress_gesture") in ["1", "true", "True"],
            vlm_report=cleaned.get("vlm_report") or None,
            severity=severity_val,
            payload_ref=cleaned.get("payload_ref", ""),
        )
