from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


SourceType = Literal["personal_notes", "model_only"]


class SophiaRequest(BaseModel):
    """
    Contract for asking a question to the pipeline.
    """
    question: str = Field(..., description="The user's technical question.")
    context_limit: int = Field(3, ge=1, le=10, description="Max number of context chunks to retrieve.")

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        return v.strip()


class SophiaResponse(BaseModel):
    """
    Contract for the pipeline's output.
    """
    topic: str = Field(..., description="The predicted topic category.")
    source_type: SourceType = Field(..., description="Where the answer was derived from.")
    technical_answer: str = Field(..., description="The detailed engineering answer.")
    simple_explanation: str = Field(..., description="ELI5 version of the answer.")
    follow_up_question: str = Field(..., description="A suggested follow-up to deepen understanding.")
    confidence_score: Optional[float] = Field(None, description="Model confidence (0.0 to 1.0).")
    latency_ms: Optional[float] = Field(None, description="Processing time in milliseconds.")


class ModelMetadata(BaseModel):
    """
    Metadata for model governance.
    """
    model_version: str
    architecture: str = "Transformer (Decoder-Only)"
    base_model: str = "Qwen-2.5-1.5B"
    created_at: datetime = Field(default_factory=datetime.now)
    parameters_count: str = "1.5B"
