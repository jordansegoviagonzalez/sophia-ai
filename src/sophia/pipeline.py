from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Union

from .classifier import TopicClassifier
from .config import AppConfig
from .knowledge_base import KnowledgeBase
from .llm_client import LLMClient
from .schemas import SophiaRequest, SophiaResponse


@dataclass
class PipelineComponents:
    config: AppConfig
    classifier: TopicClassifier
    kb: KnowledgeBase
    llm: LLMClient


class SophiaPipeline:
    """
    High-level pipeline:

    question -> topic classifier -> notes retrieval -> LLMClient -> SophiaResponse
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.components = self._init_components(config)

    def _init_components(self, config: Optional[AppConfig]) -> PipelineComponents:
        cfg = config or AppConfig()
        cfg.ensure_directories()

        classifier = TopicClassifier(cfg.model_path)
        kb = KnowledgeBase(cfg.notes_dir)
        llm = LLMClient()

        # Load resources
        kb.load()
        try:
            classifier.load()
        except FileNotFoundError:
            # Defer training to user; CLI will surface a clean message.
            pass

        return PipelineComponents(config=cfg, classifier=classifier, kb=kb, llm=llm)

    def answer(self, input_data: Union[str, SophiaRequest]) -> SophiaResponse:
        """
        Run the full pipeline for a single question.
        """
        start_time = time.perf_counter()

        # Normalize input to Pydantic model
        if isinstance(input_data, str):
            request = SophiaRequest(question=input_data)
        else:
            request = input_data

        # Predict topic (fallback to ml_basics if model is missing)
        try:
            topic = self.components.classifier.predict_topic(request.question)
            confidence = self.components.classifier.predict_proba(request.question)
        except (FileNotFoundError, AttributeError):
            # Fallback if model missing or untrained
            topic = "ml_basics"
            confidence = 0.5  # Low confidence default

        # Retrieve notes for that topic
        chunks = self.components.kb.retrieve(topic_hint=topic, k=request.context_limit)

        # Ask LLM (Note: llm_client.ask returns a partial dict/object, we wrap it)
        # We assume llm.ask returns a structure compatible with our fields.
        # Ideally, LLMClient should also be refactored, but we map it here for now.
        raw_response = self.components.llm.ask(
            question=request.question, topic=topic, notes=chunks
        )

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # ms

        # Construct final Pydantic response
        return SophiaResponse(
            topic=topic,
            source_type=raw_response.source_type,
            technical_answer=raw_response.technical_answer,
            simple_explanation=raw_response.simple_explanation,
            follow_up_question=raw_response.follow_up_question,
            latency_ms=round(latency, 2),
            confidence_score=round(confidence, 4),
        )
