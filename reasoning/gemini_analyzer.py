"""
reasoning/gemini_analyzer.py
VLM incident reasoner using the Gemini API.
Sends a 3x3 temporal frame grid together with AnomalyCLIP prior and distress gesture flags,
and parses a structured incident report.

Assumptions:
- Uses google-genai or google.generativeai with model 'gemini-2.5-flash' (or fallback 'gemini-1.5-flash').
- Parses structured JSON response conforming to incident report schema.
- If no API key is set in environment, provides deterministic fallback reasoning for offline testing.
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


@dataclass
class IncidentAnalysisReport:
    incident_description: str
    severity: str  # "low", "medium", "high"
    recommended_department: str  # "Campus Security", "Police Dispatch", "Medical First Responder", etc.
    confidence: float
    key_observations: List[str]
    threat_indicators: Dict[str, bool]

    def to_dict(self) -> Dict:
        return asdict(self)


PROMPT_TEMPLATE = """You are an expert AI surveillance security analyst specializing in women's safety, situational threat detection, and distress pattern recognition.

Input Context:
- Camera ID: {camera_id}
- Time Window: {start_sec:.2f}s - {end_sec:.2f}s
- AnomalyCLIP Prior Anomaly Type: {anomaly_type_prior}
- AnomalyCLIP Anomaly Score: {anomaly_score:.2f}
- MediaPipe Distress Gesture Flagged: {distress_gesture_flag} (Type: {distress_gesture_type}, Confidence: {gesture_confidence:.2f})

You are provided with an attached 3x3 chronological grid of 9 video frames (read left-to-right, top-to-bottom: F1 to F9) capturing the incident.

Analyze the sequence carefully and output a STRICT JSON object with these exact keys:
{{
  "incident_description": "Precise summary of the actions, spatial positioning, body language, and threat progression across frames.",
  "severity": "low | medium | high",
  "recommended_department": "Campus Security | Police Dispatch | Medical First Responder | Escort Patrol",
  "confidence": 0.0 to 1.0,
  "key_observations": [
    "Observation 1",
    "Observation 2"
  ],
  "threat_indicators": {{
    "stalking_isolation": true/false,
    "distress_gestures_observed": true/false,
    "physical_contact": true/false
  }}
}}

Return ONLY the raw JSON object, without markdown wrapping or commentary.
"""


class GeminiIncidentAnalyzer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                print(f"[GeminiAnalyzer] Initialized GenAI client with model: {self.model_name}")
            except Exception as e:
                print(f"[GeminiAnalyzer] Could not initialize google.genai: {e}, falling back to google.generativeai")
                try:
                    import google.generativeai as gai
                    gai.configure(api_key=self.api_key)
                    self.client = gai.GenerativeModel(self.model_name)
                except Exception as e2:
                    print(f"[GeminiAnalyzer] Failed to initialize Gemini API: {e2}")

    def analyze_incident(
        self,
        grid_image: Image.Image,
        camera_id: str = "CAM-01",
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        anomaly_type_prior: Optional[str] = "Harassment / Physical Dispute",
        anomaly_score: float = 0.85,
        distress_gesture_flag: bool = False,
        distress_gesture_type: Optional[str] = None,
        gesture_confidence: float = 0.0,
    ) -> IncidentAnalysisReport:
        """
        Sends the 3x3 frame grid and detection priors to Gemini, returning a structured IncidentAnalysisReport.
        """
        prompt = PROMPT_TEMPLATE.format(
            camera_id=camera_id,
            start_sec=start_sec,
            end_sec=end_sec,
            anomaly_type_prior=anomaly_type_prior or "Unspecified Anomaly",
            anomaly_score=anomaly_score,
            distress_gesture_flag=distress_gesture_flag,
            distress_gesture_type=distress_gesture_type or "None",
            gesture_confidence=gesture_confidence,
        )

        if self.client and self.api_key:
            try:
                # Call live Gemini API
                raw_text = self._call_gemini(grid_image, prompt)
                report = self._parse_response(raw_text)
                return report
            except Exception as e:
                print(f"[GeminiAnalyzer] Live API call failed: {e}. Falling back to structured heuristic analysis.")

        # Deterministic heuristic reasoner (when offline or API key missing)
        return self._heuristic_reasoning(
            anomaly_type_prior=anomaly_type_prior,
            anomaly_score=anomaly_score,
            distress_gesture_flag=distress_gesture_flag,
            distress_gesture_type=distress_gesture_type,
            gesture_confidence=gesture_confidence,
        )

    def _call_gemini(self, grid_image: Image.Image, prompt: str) -> str:
        """Invokes Gemini model with multimodal image + text prompt."""
        # Using google.genai Client
        if hasattr(self.client, "models"):
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[grid_image, prompt],
            )
            return response.text
        # Fallback google.generativeai
        else:
            response = self.client.generate_content([grid_image, prompt])
            return response.text

    def _parse_response(self, text: str) -> IncidentAnalysisReport:
        """Parses and validates structured JSON response from LLM."""
        cleaned = text.strip()
        # Remove ```json ... ``` blocks if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
            cleaned = re.sub(r"\n```$", "", cleaned)

        data = json.loads(cleaned)

        # Normalize severity
        sev = str(data.get("severity", "medium")).lower()
        if sev not in ["low", "medium", "high"]:
            sev = "medium"

        return IncidentAnalysisReport(
            incident_description=data.get("incident_description", "Incident detected by surveillance pipeline."),
            severity=sev,
            recommended_department=data.get("recommended_department", "Campus Security"),
            confidence=float(data.get("confidence", 0.85)),
            key_observations=data.get("key_observations", ["Anomaly flagged in visual stream"]),
            threat_indicators=data.get(
                "threat_indicators",
                {
                    "stalking_isolation": False,
                    "distress_gestures_observed": False,
                    "physical_contact": False,
                },
            ),
        )

    def _heuristic_reasoning(
        self,
        anomaly_type_prior: Optional[str],
        anomaly_score: float,
        distress_gesture_flag: bool,
        distress_gesture_type: Optional[str],
        gesture_confidence: float,
    ) -> IncidentAnalysisReport:
        """Synthesizes structured reasoning when Gemini API key is not present."""
        observations = []
        threats = {
            "stalking_isolation": False,
            "distress_gestures_observed": distress_gesture_flag,
            "physical_contact": False,
        }

        prior = (anomaly_type_prior or "Harassment").lower()
        if distress_gesture_flag:
            observations.append(f"Distress gesture detected ({distress_gesture_type}) with confidence {gesture_confidence:.2f}")
            if "stalk" in prior or "follow" in prior or "isolation" in prior:
                severity = "high"
                threats["stalking_isolation"] = True
                dept = "Police Dispatch"
                desc = "Urgent: Subject flagged in isolated area exhibiting active distress gestures while being trailed or approached."
            else:
                severity = "high" if anomaly_score > 0.6 else "medium"
                dept = "Campus Security"
                desc = f"Subject signaling distress ({distress_gesture_type}) during anomalous event categorized under {anomaly_type_prior}."
        else:
            if "assault" in prior or "fight" in prior or "abuse" in prior or anomaly_score > 0.7:
                severity = "high"
                threats["physical_contact"] = True
                dept = "Police Dispatch"
                desc = f"High-confidence physical altercation or aggressive encounter ({anomaly_type_prior}) captured on surveillance feed."
            elif anomaly_score > 0.4:
                severity = "medium"
                dept = "Campus Security"
                desc = f"Suspicious activity and movement pattern detected ({anomaly_type_prior}) exceeding baseline anomaly threshold."
            else:
                severity = "low"
                dept = "Escort Patrol"
                desc = f"Mild spatial irregularity ({anomaly_type_prior}) flagged for precautionary monitoring."

        observations.append(f"Visual anomaly scoring model registered peak anomaly probability of {anomaly_score:.2f}")
        observations.append(f"Class prior aligned with '{anomaly_type_prior}' surveillance profile")

        return IncidentAnalysisReport(
            incident_description=desc,
            severity=severity,
            recommended_department=dept,
            confidence=round(max(anomaly_score, gesture_confidence, 0.75), 2),
            key_observations=observations,
            threat_indicators=threats,
        )
