from dataclasses import dataclass
from typing import Literal


SourceType = Literal["personal_notes", "model_only"]


@dataclass
class SophiaResponse:
    topic: str
    source_type: SourceType
    technical_answer: str
    simple_explanation: str
    follow_up_question: str
