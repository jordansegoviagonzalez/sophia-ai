from enum import Enum


class Topic(str, Enum):
    ML_BASICS = "ml_basics"
    LOSS_FUNCTIONS = "loss_functions"
    LLM_TRANSFORMERS = "llm_transformers"
    RAG = "rag"
    DEPLOYMENT = "deployment"

    @classmethod
    def from_label(cls, label: str) -> "Topic":
        """Convert a raw string label into a Topic, with a safe fallback."""
        normalized = (label or "").strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        # Fallback to ML_BASICS if we don't recognize it
        return cls.ML_BASICS
