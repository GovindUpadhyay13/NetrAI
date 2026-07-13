"""SafetyChain — Stage 2: DESCRIBE — Scene Describer

Uses Gemma 4 VLM (via Google AI Studio API) with the PerCoAct prompt
to generate structured scene descriptions from annotated frames.

Architecture ref: Section 4 (Stage 2 — DESCRIBE: VLM Scene Understanding)
Design ref: Section 4.1 (Scene Description Prompt)
"""

import json
import time
from typing import Optional

import google.generativeai as genai
from PIL import Image
import numpy as np
import cv2

from ..config import SafetyChainConfig
from ..models import AnomalyCandidate, SceneDescription, Person, ObjectOfInterest
from ..utils.logger import get_logger, log_stage_start, log_stage_end

logger = get_logger("stage2.scene_describer")


# PerCoAct prompt template — from design doc Section 4.1
SCENE_DESCRIPTION_PROMPT = """You are a trained security camera analyst. You observe scenes carefully and report with precision. You never speculate beyond what is visible. You always structure your output as JSON.

Analyze this camera frame. A detection system has flagged:
- Object class: {class_name} ({confidence:.0%} confidence)
- Location in frame: bounding box {bbox}
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
{{
  "scene_environment": "string describing the scene",
  "people": [
    {{
      "id": "P1",
      "description": "physical description",
      "position": "where in scene",
      "posture": "body posture",
      "movement": "movement pattern"
    }}
  ],
  "objects": [
    {{
      "type": "object type",
      "description": "object description"
    }}
  ],
  "visible_text": ["any visible text or signs"],
  "activity": "what is happening",
  "norm_violation": "what seems wrong or 'none'",
  "suspiciousness": "NORMAL or UNUSUAL or CONCERNING or ALARMING"
}}"""


class SceneDescriber:
    """Generates structured scene descriptions using Gemma VLM.
    
    Uses the PerCoAct prompt pattern (Perception + Cognition) to force
    the model to first describe what it sees, then interpret — reducing
    hallucination where the model jumps to conclusions.
    """

    def __init__(self, config: SafetyChainConfig):
        """Initialize the scene describer with Gemma configuration.
        
        Args:
            config: SafetyChain configuration with Gemma settings
        """
        self.config = config
        self.model = None

        if config.GOOGLE_API_KEY:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(config.GEMMA_MODEL)
            logger.info(f"SceneDescriber initialized with model={config.GEMMA_MODEL}")
        else:
            logger.warning(
                "No GOOGLE_API_KEY set — SceneDescriber will use fallback mode"
            )

    def describe(
        self,
        candidate: AnomalyCandidate,
        zone_name: str = "Unknown Zone",
        zone_type: str = "general",
    ) -> SceneDescription:
        """Generate a structured scene description for an anomaly candidate.
        
        Args:
            candidate: The anomaly candidate with annotated frame
            zone_name: Human-readable zone name
            zone_type: Zone type (parking, school, corridor, etc.)
            
        Returns:
            SceneDescription with perception and cognition analysis
        """
        log_stage_start(logger, "DESCRIBE", candidate.id)
        start_time = time.time()

        # Use the primary detection for prompt context
        primary_detection = max(candidate.detections, key=lambda d: d.confidence)

        try:
            if self.model:
                result = self._describe_with_gemma(
                    candidate, primary_detection, zone_name, zone_type
                )
            else:
                result = self._describe_fallback(
                    candidate, primary_detection, zone_name, zone_type
                )
        except Exception as e:
            logger.error(f"Scene description failed: {e}")
            result = self._describe_fallback(
                candidate, primary_detection, zone_name, zone_type
            )

        latency_ms = int((time.time() - start_time) * 1000)
        log_stage_end(logger, "DESCRIBE", candidate.id, latency_ms)

        return result

    def _describe_with_gemma(
        self,
        candidate: AnomalyCandidate,
        primary_detection,
        zone_name: str,
        zone_type: str,
    ) -> SceneDescription:
        """Use Gemma VLM to generate scene description.
        
        Sends the annotated frame + PerCoAct prompt to Gemma and
        parses the structured JSON response.
        """
        # Construct the prompt
        prompt = SCENE_DESCRIPTION_PROMPT.format(
            class_name=primary_detection.class_name,
            confidence=primary_detection.confidence,
            bbox=primary_detection.bbox,
            zone_name=zone_name,
            zone_type=zone_type,
        )

        # Convert OpenCV BGR frame to PIL Image for Gemma
        frame_rgb = cv2.cvtColor(candidate.frame_annotated, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Call Gemma multimodal API
        response = self.model.generate_content(
            [prompt, pil_image],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.config.GEMMA_MAX_TOKENS,
                temperature=self.config.GEMMA_TEMPERATURE,
            ),
        )

        # Parse the JSON response
        response_text = response.text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1])

        raw_json = json.loads(response_text)
        return self._parse_scene_json(candidate.id, raw_json)

    def _describe_fallback(
        self,
        candidate: AnomalyCandidate,
        primary_detection,
        zone_name: str,
        zone_type: str,
    ) -> SceneDescription:
        """Fallback scene description using only YOLO detection data.
        
        Used when Gemma is unavailable or fails.
        """
        logger.info("Using YOLO-only fallback for scene description")

        # Build a description from detection metadata
        det_summary = ", ".join(
            f"{d.class_name} ({d.confidence:.0%})" for d in candidate.detections
        )

        people = []
        for i, det in enumerate(candidate.detections):
            if det.class_name == "person":
                people.append(Person(
                    id=f"P{i + 1}",
                    description=f"Person detected with {det.confidence:.0%} confidence",
                    position=f"Bounding box {det.bbox}",
                    posture="Unknown (YOLO-only mode)",
                    movement="Unknown (YOLO-only mode)",
                ))

        objects = []
        for det in candidate.detections:
            if det.class_name != "person":
                objects.append(ObjectOfInterest(
                    type=det.class_name,
                    description=f"{det.class_name} detected ({det.confidence:.0%})",
                ))

        # Determine suspiciousness from detection types
        suspicious_classes = {"knife", "fire", "scissors"}
        has_suspicious = any(
            d.class_name in suspicious_classes for d in candidate.detections
        )
        suspiciousness = "ALARMING" if has_suspicious else "CONCERNING"

        raw_json = {
            "mode": "yolo_fallback",
            "detections": det_summary,
            "trigger_reason": candidate.trigger_reason,
        }

        return SceneDescription(
            candidate_id=candidate.id,
            scene_environment=f"{zone_name} ({zone_type})",
            people=people,
            objects=objects,
            visible_text=[],
            activity=f"Detected: {det_summary}. Trigger: {candidate.trigger_reason}",
            norm_violation=candidate.trigger_reason,
            suspiciousness=suspiciousness,
            raw_json=raw_json,
        )

    def _parse_scene_json(self, candidate_id: str, raw_json: dict) -> SceneDescription:
        """Parse Gemma's JSON response into a SceneDescription dataclass.
        
        Handles missing or malformed fields gracefully.
        """
        people = []
        for p in raw_json.get("people", []):
            people.append(Person(
                id=p.get("id", "P?"),
                description=p.get("description", "Unknown"),
                position=p.get("position", "Unknown"),
                posture=p.get("posture", "Unknown"),
                movement=p.get("movement", "Unknown"),
            ))

        objects = []
        for o in raw_json.get("objects", []):
            objects.append(ObjectOfInterest(
                type=o.get("type", "unknown"),
                description=o.get("description", "Unknown"),
            ))

        return SceneDescription(
            candidate_id=candidate_id,
            scene_environment=raw_json.get("scene_environment", "Unknown"),
            people=people,
            objects=objects,
            visible_text=raw_json.get("visible_text", []),
            activity=raw_json.get("activity", "Unknown"),
            norm_violation=raw_json.get("norm_violation", "none"),
            suspiciousness=raw_json.get("suspiciousness", "UNUSUAL"),
            raw_json=raw_json,
        )
