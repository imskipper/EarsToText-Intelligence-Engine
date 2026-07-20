from typing import List

from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

from config.settings import settings


class IncidentUrgency(BaseModel):
    priority_level: str = Field(..., description="Classification: LOW, MEDIUM, HIGH, CRITICAL")
    justification: str = Field(..., description="Brief technical justification for the assigned priority.")

class ActionableEntity(BaseModel):
    entity_name: str = Field(..., description="Extracted actionable unit (e.g., patient name, drug name, location, core skill).")
    contextual_metadata: str = Field(..., description="Surrounding contextual details or explicit descriptions mentioned.")

class StructuredIntelligenceReport(BaseModel):
    executive_summary: str = Field(..., description="High-density objective recap of the verbal intake.")
    triage_metrics: IncidentUrgency = Field(..., description="System generated risk/urgency scoring parameters.")
    extracted_entities: List[ActionableEntity] = Field(default=[], description="Extracted critical entities.")
    domain_terms: List[ActionableEntity] = Field(default=[], description="Medical, technical, or operational jargon mapped from the transcript.")
    recommended_next_steps: List[str] = Field(default=[], description="Automated logical responses based on the intake.")


class StructuredAnalyticsPipeline:
    def __init__(self):
        if not settings.has_openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        self.client = instructor.from_openai(OpenAI(api_key=settings.OPENAI_API_KEY))

    def analyze_transcript(self, raw_text: str, acoustic_context: str = "") -> StructuredIntelligenceReport:
        """Enforces schema compliance on unstructured conversational streams."""
        return self.client.chat.completions.create(
            model=settings.INFERENCE_MODEL,
            response_model=StructuredIntelligenceReport,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert redacted audio transcripts and acoustic metrics into strict, "
                        "auditable intelligence reports. Preserve uncertainty, extract domain jargon, "
                        "and never reconstruct redacted PII."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Transcript:\n"
                        f"{raw_text}\n\n"
                        "Acoustic context:\n"
                        f"{acoustic_context or 'No acoustic context provided.'}"
                    ),
                },
            ],
            temperature=0.0,
        )
