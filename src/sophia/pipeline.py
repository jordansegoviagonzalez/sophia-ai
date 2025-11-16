from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .classifier import TopicClassifier
from .config import AppConfig
from .knowledge_base import KnowledgeBase
from .llm_client import LLMClient
from .schemas import SophiaResponse


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

    def answer(self, question: str) -> SophiaResponse:
        """
        Run the full pipeline for a single question.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Predict topic (fallback to ml_basics if model is missing)
        try:
            topic = self.components.classifier.predict_topic(question)
        except FileNotFoundError:
            topic = "ml_basics"

        # Retrieve notes for that topic
        chunks = self.components.kb.retrieve(topic_hint=topic, k=3)

        # Ask LLM
        response = self.components.llm.ask(question=question, topic=topic, notes=chunks)
        return response
