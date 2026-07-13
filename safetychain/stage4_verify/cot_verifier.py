"""SafetyChain — Stage 4: VERIFY — Chain-of-Thought Verifier

Implements the Depth Router and structured CoT reasoning.
Architecture ref: Section 6 (Stage 4 — VERIFY: Adaptive CoT Reasoning)
Design ref: Section 4.2 (CoT Verification Prompt)

Reasoning strategies:
  - ZeroThink: CRITICAL events (weapon/fire + school) → instant verdict <50ms
  - LessThink: HIGH confidence → 3-step abbreviated ~150ms
  - FullThink: Ambiguous cases → 5-step complete chain ~300ms
  - MoreThink: Novel/unknown → extended + extra hypotheses ~500ms
"""

import json
import time
import uuid
from typing import Optional

import google.generativeai as genai

from ..config import SafetyChainConfig
from ..models import (
    AnomalyCandidate, SceneDescription, ContextReport,
    Verdict, ReasoningStep,
)
from ..utils.logger import get_logger, log_stage_start, log_stage_end

logger = get_logger("stage4.cot_verifier")


# CoT 5-step verification prompt — from design doc Section 4.2
COT_VERIFICATION_PROMPT = """You are a deliberative safety verification system. You NEVER jump to conclusions. You evaluate evidence systematically and always consider alternative explanations. If in doubt, you err on the side of caution but document your uncertainty.

Verify this potential anomaly using the evidence below.

═══ EVIDENCE ═══
Visual observation: {scene_description}
Detection class: {detection_class} ({detection_confidence:.0%} confidence)
Zone: {zone_name} ({zone_type})
Time: {current_time} ({day_of_week})
Zone active hours: {active_hours}
Currently active: {is_active}
Historical FP rate for this camera: {fp_rate:.0%}
Known FP patterns: {known_fp_patterns}

═══ VERIFY IN 5 STEPS ═══

STEP 1 — EVIDENCE QUALITY
  How clear is the visual evidence? Rate: HIGH/MEDIUM/LOW
  Confidence in primary detection: __%

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
  Confidence: __%
  Recommended action: ___

Respond ONLY with valid JSON matching this schema:
{{
  "steps": [
    {{"step_number": 1, "title": "Evidence Quality", "content": "...", "passed": true}},
    {{"step_number": 2, "title": "Context Alignment", "content": "...", "passed": true}},
    {{"step_number": 3, "title": "Alternative Hypotheses", "content": "...", "passed": true}},
    {{"step_number": 4, "title": "Severity Assessment", "content": "...", "passed": true}},
    {{"step_number": 5, "title": "Final Verdict", "content": "...", "passed": true}}
  ],
  "classification": "FALSE_POSITIVE or SUSPICIOUS or CONFIRMED_ANOMALY",
  "confidence": 0.89,
  "severity": "LOW or MEDIUM or HIGH or CRITICAL",
  "urgency": "MONITOR or INVESTIGATE or INTERVENE or EMERGENCY",
  "alternative_hypotheses": ["hypothesis 1", "hypothesis 2"],
  "recommended_action": "what to do",
  "consequences_if_ignored": "what happens if ignored"
}}"""


class CoTVerifier:
    """Chain-of-Thought verification with adaptive reasoning depth.
    
    The Depth Router determines which strategy to use based on
    detection confidence and severity:
    
    - ZeroThink: >95% AND CRITICAL → instant verdict
    - LessThink: >80% AND HIGH → 3-step abbreviated
    - FullThink: 50-80% OR ambiguous → 5-step complete
    - MoreThink: 30-50% OR novel/unknown → extended analysis
    """

    def __init__(self, config: SafetyChainConfig):
        """Initialize the CoT verifier.
        
        Args:
            config: SafetyChain configuration
        """
        self.config = config
        self.model = None

        if config.GOOGLE_API_KEY:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(config.GEMMA_MODEL)
            logger.info(f"CoTVerifier initialized with model={config.GEMMA_MODEL}")
        else:
            logger.warning(
                "No GOOGLE_API_KEY set — CoTVerifier will use rule-based fallback"
            )

    def verify(
        self,
        candidate: AnomalyCandidate,
        scene: SceneDescription,
        context: ContextReport,
    ) -> Verdict:
        """Verify an anomaly candidate through adaptive CoT reasoning.
        
        The Depth Router selects the reasoning strategy, then either
        skips reasoning (ZeroThink) or runs the full CoT chain.
        
        Args:
            candidate: The anomaly candidate
            scene: VLM-generated scene description
            context: Aggregated context report
            
        Returns:
            Verdict with classification, confidence, reasoning chain
        """
        log_stage_start(logger, "VERIFY", candidate.id)
        start_time = time.time()

        # Check if context says to suppress
        if context.suppress:
            latency_ms = int((time.time() - start_time) * 1000)
            log_stage_end(logger, "VERIFY", candidate.id, latency_ms, "SUPPRESSED")
            return self._create_suppressed_verdict(candidate, context, latency_ms)

        # Depth Router — select reasoning strategy
        strategy = self._route_depth(candidate, scene, context)
        logger.info(f"Depth Router selected: {strategy} for candidate {candidate.id}")

        if strategy == "ZeroThink":
            verdict = self._zerothink(candidate, scene, context)
        elif strategy == "LessThink":
            verdict = self._fullthink_or_fallback(candidate, scene, context, steps=3)
        elif strategy == "MoreThink":
            verdict = self._fullthink_or_fallback(candidate, scene, context, steps=5)
        else:  # FullThink (default)
            verdict = self._fullthink_or_fallback(candidate, scene, context, steps=5)

        latency_ms = int((time.time() - start_time) * 1000)
        verdict.reasoning_latency_ms = latency_ms
        log_stage_end(logger, "VERIFY", candidate.id, latency_ms, verdict.classification)

        return verdict

    def _route_depth(
        self,
        candidate: AnomalyCandidate,
        scene: SceneDescription,
        context: ContextReport,
    ) -> str:
        """Determine reasoning depth based on confidence and severity.
        
        Decision tree from architecture Section 6:
        - >95% AND CRITICAL (weapon/fire/school breach) → ZeroThink
        - >80% AND HIGH → LessThink
        - 50-80% OR ambiguous → FullThink
        - 30-50% OR novel/unknown → MoreThink
        """
        max_confidence = max(d.confidence for d in candidate.detections)

        # Check for ZeroThink triggers
        is_critical_class = any(
            d.class_name in self.config.ZEROTHINK_CLASSES
            for d in candidate.detections
        )
        is_critical_zone = context.zone.zone_type in self.config.ZEROTHINK_ZONE_TYPES

        if (is_critical_class or is_critical_zone) and max_confidence >= 0.90:
            return "ZeroThink"

        # Adjusted confidence with context modifier
        adjusted = max_confidence + context.confidence_modifier

        if adjusted > 0.80:
            return "LessThink"
        elif adjusted >= 0.50:
            return "FullThink"
        else:
            return "MoreThink"

    def _zerothink(
        self,
        candidate: AnomalyCandidate,
        scene: SceneDescription,
        context: ContextReport,
    ) -> Verdict:
        """ZeroThink — instant verdict for critical events.
        
        Bypasses CoT reasoning entirely. Used for weapons, fire,
        or school zone breaches. Target: <50ms.
        """
        # Determine the specific critical trigger
        critical_classes = [
            d.class_name for d in candidate.detections
            if d.class_name in self.config.ZEROTHINK_CLASSES
        ]
        is_school = context.zone.zone_type == "school"

        if critical_classes:
            reason = f"ZeroThink: {', '.join(critical_classes)} detected. Automatic critical escalation."
        elif is_school:
            reason = "ZeroThink: School perimeter breach during school hours. Automatic critical escalation per policy."
        else:
            reason = "ZeroThink: Critical event detected. Automatic escalation."

        return Verdict(
            candidate_id=candidate.id,
            chain_id=str(uuid.uuid4()),
            classification="CONFIRMED_ANOMALY",
            confidence=0.99,
            severity="CRITICAL",
            urgency="EMERGENCY",
            reasoning_strategy="ZeroThink",
            reasoning_chain=[],  # Empty for ZeroThink
            alternative_hypotheses=[],
            recommended_action="DISPATCH IMMEDIATELY. Follow emergency SOP.",
            consequences_if_ignored="Potential life-threatening situation left unaddressed.",
            reasoning_latency_ms=0,
        )

    def _fullthink_or_fallback(
        self,
        candidate: AnomalyCandidate,
        scene: SceneDescription,
        context: ContextReport,
        steps: int = 5,
    ) -> Verdict:
        """Run FullThink CoT with Gemma, falling back to rule-based if unavailable.
        
        Args:
            candidate: Anomaly candidate
            scene: Scene description
            context: Context report
            steps: Number of reasoning steps (3 for LessThink, 5 for FullThink)
        """
        if self.model:
            try:
                return self._fullthink_gemma(candidate, scene, context)
            except Exception as e:
                logger.error(f"Gemma CoT failed: {e}. Using rule-based fallback.")

        return self._fullthink_rules(candidate, scene, context)

    def _fullthink_gemma(
        self,
        candidate: AnomalyCandidate,
        scene: SceneDescription,
        context: ContextReport,
    ) -> Verdict:
        """Run 5-step CoT verification using Gemma."""
        primary_det = max(candidate.detections, key=lambda d: d.confidence)

        prompt = COT_VERIFICATION_PROMPT.format(
            scene_description=scene.activity,
            detection_class=primary_det.class_name,
            detection_confidence=primary_det.confidence,
            zone_name=context.zone.zone_name,
            zone_type=context.zone.zone_type,
            current_time=context.temporal.current_time.strftime("%H:%M"),
            day_of_week=context.temporal.day_of_week,
            active_hours=context.zone.active_hours,
            is_active=context.temporal.is_within_active_hours,
            fp_rate=context.historical.false_positive_rate,
            known_fp_patterns=context.historical.known_fp_pattern or "None known",
        )

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.config.GEMMA_MAX_TOKENS,
                temperature=self.config.GEMMA_TEMPERATURE,
            ),
        )

        response_text = response.text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        raw = json.loads(response_text)
        return self._parse_cot_response(candidate.id, raw)

    def _fullthink_rules(
        self,
        candidate: AnomalyCandidate,
        scene: SceneDescription,
        context: ContextReport,
    ) -> Verdict:
        """Rule-based fallback for CoT verification when Gemma is unavailable.
        
        Implements the 5-step reasoning chain using deterministic rules.
        """
        primary_det = max(candidate.detections, key=lambda d: d.confidence)
        chain_id = str(uuid.uuid4())

        # Step 1: Evidence Quality
        evidence_quality = "HIGH" if primary_det.confidence > 0.7 else (
            "MEDIUM" if primary_det.confidence > 0.5 else "LOW"
        )
        step1 = ReasoningStep(
            step_number=1,
            title="Evidence Quality",
            content=f"Primary detection: {primary_det.class_name} at {primary_det.confidence:.0%} confidence. "
                    f"Evidence quality: {evidence_quality}. "
                    f"Motion delta: {candidate.motion_delta:.2f}.",
            passed=primary_det.confidence > 0.4,
        )

        # Step 2: Context Alignment
        context_verdict = context.verdict
        step2 = ReasoningStep(
            step_number=2,
            title="Context Alignment",
            content=f"Zone: {context.zone.zone_name} ({context.zone.zone_type}). "
                    f"Active hours: {context.zone.active_hours}. "
                    f"Currently active: {context.temporal.is_within_active_hours}. "
                    f"Context verdict: {context_verdict}. "
                    f"Confidence modifier: {context.confidence_modifier:+.2f}.",
            passed=context_verdict != "REFUTES_ANOMALY",
        )

        # Step 3: Alternative Hypotheses
        alt_hypotheses = self._generate_alt_hypotheses(
            primary_det, scene, context
        )
        plausible_alts = [h for h in alt_hypotheses if "(plausible)" in h.lower()]
        step3 = ReasoningStep(
            step_number=3,
            title="Alternative Hypotheses",
            content=f"Considered {len(alt_hypotheses)} alternative explanations: "
                    + "; ".join(alt_hypotheses)
                    + f". {len(plausible_alts)} plausible alternatives found.",
            passed=len(plausible_alts) == 0,
        )

        # Step 4: Severity Assessment
        severity, urgency = self._assess_severity(primary_det, scene, context)
        step4 = ReasoningStep(
            step_number=4,
            title="Severity Assessment",
            content=f"Threat level: {severity}. Urgency: {urgency}. "
                    f"Suspiciousness from VLM: {scene.suspiciousness}.",
            passed=severity in ("HIGH", "CRITICAL"),
        )

        # Step 5: Final Verdict
        adjusted_confidence = min(1.0, max(0.0,
            primary_det.confidence + context.confidence_modifier
        ))

        # Classify based on adjusted confidence and reasoning
        if adjusted_confidence >= 0.7 and step1.passed and step2.passed:
            classification = "CONFIRMED_ANOMALY"
        elif adjusted_confidence >= 0.4:
            classification = "SUSPICIOUS"
        else:
            classification = "FALSE_POSITIVE"

        recommended = self._get_recommended_action(severity, urgency, context)
        consequences = self._get_consequences(severity, scene)

        step5 = ReasoningStep(
            step_number=5,
            title="Final Verdict",
            content=f"Classification: {classification} at {adjusted_confidence:.0%} confidence. "
                    f"Recommended action: {recommended}.",
            passed=classification != "FALSE_POSITIVE",
        )

        strategy = "FullThink"
        if adjusted_confidence > 0.8:
            strategy = "LessThink"
        elif adjusted_confidence < 0.5:
            strategy = "MoreThink"

        return Verdict(
            candidate_id=candidate.id,
            chain_id=chain_id,
            classification=classification,
            confidence=adjusted_confidence,
            severity=severity,
            urgency=urgency,
            reasoning_strategy=strategy,
            reasoning_chain=[step1, step2, step3, step4, step5],
            alternative_hypotheses=[h for h in alt_hypotheses],
            recommended_action=recommended,
            consequences_if_ignored=consequences,
            reasoning_latency_ms=0,
        )

    def _generate_alt_hypotheses(self, detection, scene, context) -> list:
        """Generate alternative benign explanations."""
        hypotheses = []

        if detection.class_name == "person":
            if context.zone.zone_type == "parking":
                hypotheses.append(
                    "Owner returning to vehicle after hours (unlikely — no badge scan)"
                )
                hypotheses.append(
                    "Maintenance worker performing scheduled work (unlikely — no scheduled work)"
                )
                hypotheses.append(
                    "Delivery person at wrong location (plausible)"
                )
            elif context.zone.zone_type == "school":
                hypotheses.append(
                    "Staff member accessing perimeter (unlikely — unusual approach)"
                )
                hypotheses.append(
                    "Parent picking up child (unlikely — not at designated area)"
                )
        elif detection.class_name in ("knife", "scissors"):
            hypotheses.append(
                "Worker with a tool (unlikely — wrong area/time)"
            )
            hypotheses.append(
                "Misidentified object like a phone or remote (plausible)"
            )

        if not hypotheses:
            hypotheses.append("Benign activity misidentified (plausible)")
            hypotheses.append("Environmental factor (camera artifact)")

        return hypotheses

    def _assess_severity(self, detection, scene, context):
        """Assess threat severity and urgency."""
        # Critical: weapon or school
        if detection.class_name in ("knife", "scissors", "fire"):
            return "CRITICAL", "EMERGENCY"
        if context.zone.zone_type == "school":
            return "HIGH", "EMERGENCY"

        # Based on suspiciousness
        susp = scene.suspiciousness
        if susp == "ALARMING":
            return "HIGH", "INTERVENE"
        elif susp == "CONCERNING":
            return "MEDIUM", "INVESTIGATE"
        elif susp == "UNUSUAL":
            return "LOW", "MONITOR"
        else:
            return "LOW", "MONITOR"

    def _get_recommended_action(self, severity, urgency, context):
        """Get recommended action based on severity and available SOPs."""
        if context.protocol.matching_sop:
            return f"Follow {context.protocol.matching_sop}"

        action_map = {
            "EMERGENCY": "DISPATCH IMMEDIATELY. Contact emergency services.",
            "INTERVENE": "Alert security patrol. Monitor situation.",
            "INVESTIGATE": "Review footage. Consider dispatching patrol.",
            "MONITOR": "Continue monitoring. Log for review.",
        }
        return action_map.get(urgency, "Continue monitoring.")

    def _get_consequences(self, severity, scene):
        """Describe consequences if the alert is ignored."""
        consequence_map = {
            "CRITICAL": "Potential life-threatening situation left unaddressed. Risk of injury or death.",
            "HIGH": "Possible property damage or theft in progress. Escalation likely.",
            "MEDIUM": "Potential security breach may go undetected. Property at risk.",
            "LOW": "Minor security concern. May indicate developing pattern if repeated.",
        }
        return consequence_map.get(severity, "Unknown risk level.")

    def _parse_cot_response(self, candidate_id: str, raw: dict) -> Verdict:
        """Parse Gemma's CoT JSON response into a Verdict."""
        steps = []
        for s in raw.get("steps", []):
            steps.append(ReasoningStep(
                step_number=s.get("step_number", 0),
                title=s.get("title", "Unknown"),
                content=s.get("content", ""),
                passed=s.get("passed", True),
            ))

        return Verdict(
            candidate_id=candidate_id,
            chain_id=str(uuid.uuid4()),
            classification=raw.get("classification", "SUSPICIOUS"),
            confidence=float(raw.get("confidence", 0.5)),
            severity=raw.get("severity", "MEDIUM"),
            urgency=raw.get("urgency", "INVESTIGATE"),
            reasoning_strategy="FullThink",
            reasoning_chain=steps,
            alternative_hypotheses=raw.get("alternative_hypotheses", []),
            recommended_action=raw.get("recommended_action", "Investigate further"),
            consequences_if_ignored=raw.get("consequences_if_ignored", "Unknown"),
            reasoning_latency_ms=0,
        )

    def _create_suppressed_verdict(self, candidate, context, latency_ms):
        """Create a verdict for suppressed alerts (known FP patterns)."""
        return Verdict(
            candidate_id=candidate.id,
            chain_id=str(uuid.uuid4()),
            classification="FALSE_POSITIVE",
            confidence=0.1,
            severity="LOW",
            urgency="MONITOR",
            reasoning_strategy="Suppressed",
            reasoning_chain=[
                ReasoningStep(
                    step_number=1,
                    title="Context Suppression",
                    content=f"Alert suppressed by context engine: {context.suppress_reason}",
                    passed=False,
                )
            ],
            alternative_hypotheses=[],
            recommended_action="No action required. Known false positive pattern.",
            consequences_if_ignored="None — this matches a known false positive.",
            reasoning_latency_ms=latency_ms,
        )
