"""SafetyChain — Structured Logger

JSON-formatted logging for evidence trail.
Logs pipeline stages, latencies, and decisions.
"""

import logging
import json
import sys
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON for structured evidence trails."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "stage"):
            log_entry["stage"] = record.stage
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms
        if hasattr(record, "candidate_id"):
            log_entry["candidate_id"] = record.candidate_id
        if hasattr(record, "chain_id"):
            log_entry["chain_id"] = record.chain_id
        if hasattr(record, "decision"):
            log_entry["decision"] = record.decision
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        return json.dumps(log_entry)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a structured JSON logger for SafetyChain components.
    
    Args:
        name: Logger name (typically module name like 'stage1.detector')
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"safetychain.{name}")
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def log_stage_start(logger: logging.Logger, stage: str, candidate_id: str):
    """Log the start of a pipeline stage."""
    logger.info(
        f"Stage {stage} started",
        extra={"stage": stage, "candidate_id": candidate_id}
    )


def log_stage_end(logger: logging.Logger, stage: str, candidate_id: str, 
                  latency_ms: int, decision: str = None):
    """Log the end of a pipeline stage with latency."""
    extra = {
        "stage": stage,
        "candidate_id": candidate_id,
        "latency_ms": latency_ms,
    }
    if decision:
        extra["decision"] = decision
    logger.info(
        f"Stage {stage} completed in {latency_ms}ms",
        extra=extra
    )


def log_pipeline_decision(logger: logging.Logger, candidate_id: str, 
                          decision: str, reason: str):
    """Log a pipeline decision (pass/drop/suppress)."""
    logger.info(
        f"Pipeline decision: {decision} — {reason}",
        extra={
            "candidate_id": candidate_id,
            "decision": decision,
            "extra_data": {"reason": reason}
        }
    )
