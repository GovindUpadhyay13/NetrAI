"""SafetyChain — Stage 5: ACT — Evidence Packager

Bundles annotated frame + reasoning chain + context report + verdict
into a self-contained evidence package for forensic review.

Architecture ref: Section 7 (Evidence Packager component)
"""

import json
import os
from datetime import datetime
from typing import Optional

from ..models import Alert, Verdict, ContextReport
from ..utils.logger import get_logger

logger = get_logger("stage5.evidence_packager")


class EvidencePackager:
    """Packages alert evidence into JSON and HTML formats for forensic review."""

    def __init__(self, evidence_dir: str = "evidence"):
        """Initialize the evidence packager.
        
        Args:
            evidence_dir: Directory to store evidence files
        """
        self.evidence_dir = evidence_dir
        os.makedirs(evidence_dir, exist_ok=True)
        logger.info(f"EvidencePackager initialized at {evidence_dir}")

    def package_evidence(self, alert: Alert) -> dict:
        """Package an alert into a JSON evidence structure.
        
        Args:
            alert: The alert to package
            
        Returns:
            Evidence dict containing all forensic data
        """
        evidence = {
            "evidence_id": alert.chain_id,
            "alert_id": alert.alert_id,
            "timestamp": alert.timestamp.isoformat(),
            "severity": alert.severity,
            "title": alert.title,
            "zone": alert.zone_name,
            "confidence": alert.confidence,
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
            "frame_b64": alert.frame_b64[:50] + "..." if alert.frame_b64 else None,
        }

        return evidence

    def save_evidence_json(self, alert: Alert) -> str:
        """Save evidence as a JSON file.
        
        Args:
            alert: The alert to save
            
        Returns:
            Path to the saved evidence file
        """
        evidence = self.package_evidence(alert)

        # Include full frame in saved file
        evidence["frame_b64"] = alert.frame_b64

        filename = f"evidence_{alert.chain_id[:8]}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.evidence_dir, filename)

        with open(filepath, "w") as f:
            json.dump(evidence, f, indent=2, default=str)

        logger.info(f"Evidence saved: {filepath}")
        return filepath

    def export_html_report(self, alert: Alert) -> str:
        """Export evidence as a self-contained HTML report.
        
        Args:
            alert: The alert to export
            
        Returns:
            Path to the exported HTML file
        """
        severity_colors = {
            "LOG": "#6c757d",
            "NOTIFY": "#ffc107",
            "ALERT": "#fd7e14",
            "EMERGENCY": "#dc3545",
        }
        color = severity_colors.get(alert.severity, "#6c757d")

        # Build reasoning chain HTML
        chain_html = ""
        for step in alert.verdict.reasoning_chain:
            icon = "✅" if step.passed else "❌"
            chain_html += f"""
            <div class="step">
                <h3>{icon} Step {step.step_number}: {step.title}</h3>
                <p>{step.content}</p>
            </div>"""

        # Build hypotheses HTML
        hypotheses_html = ""
        for h in alert.verdict.alternative_hypotheses:
            hypotheses_html += f"<li>{h}</li>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SafetyChain Evidence — {alert.alert_id}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', system-ui, sans-serif; background: #0a0a1a; color: #e0e0e0; padding: 2rem; }}
        .header {{ background: {color}22; border: 1px solid {color}; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; }}
        .header h1 {{ color: {color}; font-size: 1.5rem; }}
        .header .meta {{ color: #888; margin-top: 0.5rem; }}
        .section {{ background: #12122a; border: 1px solid #2a2a4a; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
        .section h2 {{ color: #8be9fd; margin-bottom: 1rem; font-size: 1.2rem; }}
        .step {{ background: #1a1a2e; border-radius: 8px; padding: 1rem; margin-bottom: 0.75rem; }}
        .step h3 {{ color: #f5a623; font-size: 0.95rem; margin-bottom: 0.5rem; }}
        .step p {{ color: #ccc; font-size: 0.9rem; line-height: 1.5; }}
        .evidence-frame {{ max-width: 100%; border-radius: 8px; border: 2px solid #2a2a4a; }}
        .badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }}
        .contacts {{ display: grid; gap: 0.5rem; }}
        .contact {{ background: #1a1a2e; padding: 0.5rem 1rem; border-radius: 6px; }}
        ul {{ padding-left: 1.5rem; }}
        li {{ margin-bottom: 0.5rem; color: #ccc; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⛓️ SafetyChain Evidence Report</h1>
        <div class="meta">
            Alert: {alert.alert_id} | Chain: {alert.chain_id} | 
            {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} |
            <span class="badge" style="background: {color}33; color: {color};">{alert.severity}</span>
        </div>
    </div>

    <div class="section">
        <h2>📋 Alert Summary</h2>
        <p><strong>Title:</strong> {alert.title}</p>
        <p><strong>Zone:</strong> {alert.zone_name}</p>
        <p><strong>Confidence:</strong> {alert.confidence:.0%}</p>
        <p><strong>Classification:</strong> {alert.verdict.classification}</p>
        <p><strong>Strategy:</strong> {alert.verdict.reasoning_strategy}</p>
        <p><strong>Context:</strong> {alert.context_summary}</p>
    </div>

    {"<div class='section'><h2>📸 Evidence Frame</h2><img class='evidence-frame' src='data:image/jpeg;base64," + alert.frame_b64 + "'/></div>" if alert.frame_b64 else ""}

    <div class="section">
        <h2>⛓️ Reasoning Chain ({alert.verdict.reasoning_strategy})</h2>
        {chain_html if chain_html else "<p>No reasoning chain (ZeroThink — instant verdict)</p>"}
    </div>

    <div class="section">
        <h2>🔄 Alternative Hypotheses</h2>
        <ul>{hypotheses_html if hypotheses_html else "<li>None considered</li>"}</ul>
    </div>

    <div class="section">
        <h2>🚨 Recommended Action</h2>
        <p>{alert.verdict.recommended_action}</p>
        <p><strong>If ignored:</strong> {alert.verdict.consequences_if_ignored}</p>
    </div>

    {self._build_sop_html(alert)}
    {self._build_contacts_html(alert)}

    <div class="section">
        <h2>📊 Metadata</h2>
        <p><strong>Reasoning latency:</strong> {alert.verdict.reasoning_latency_ms}ms</p>
        <p><strong>Status:</strong> {alert.status}</p>
        <p><strong>Operator feedback:</strong> {alert.operator_feedback or 'Pending'}</p>
    </div>
</body>
</html>"""

        filename = f"report_{alert.chain_id[:8]}_{alert.timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.evidence_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report exported: {filepath}")
        return filepath

    def _build_sop_html(self, alert: Alert) -> str:
        """Build SOP section HTML."""
        if not alert.sop:
            return ""
        # Replace newlines with <br> for HTML
        sop_html = alert.sop.replace("\n", "<br>")
        return f"""
    <div class="section">
        <h2>📖 Standard Operating Procedure</h2>
        <p>{sop_html}</p>
    </div>"""

    def _build_contacts_html(self, alert: Alert) -> str:
        """Build contacts section HTML."""
        if not alert.contacts:
            return ""
        contacts_items = ""
        for role, number in alert.contacts.items():
            contacts_items += f'<div class="contact"><strong>{role}:</strong> {number}</div>'
        return f"""
    <div class="section">
        <h2>📞 Emergency Contacts</h2>
        <div class="contacts">{contacts_items}</div>
    </div>"""
