from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class NoteChunk:
    text: str
    source: str


class KnowledgeBase:
    """
    Very simple note-based knowledge store.

    Loads full .md files as single chunks. Retrieval is keyword-based.
    """

    def __init__(self, notes_dir: Path) -> None:
        self.notes_dir = Path(notes_dir)
        self.chunks: List[NoteChunk] = []

    def load(self) -> None:
        self.chunks.clear()
        if not self.notes_dir.exists():
            return
        for path in self.notes_dir.glob("*.md"):
            text = path.read_text(encoding="utf-8")
            self.chunks.append(NoteChunk(text=text, source=path.name))

    def retrieve(self, topic_hint: str, k: int = 3) -> List[NoteChunk]:
        """
        Naive retrieval: count occurrences of topic_hint in each chunk.
        """
        topic_hint_lower = (topic_hint or "").lower()
        scored: List[tuple[int, NoteChunk]] = []

        for chunk in self.chunks:
            text_lower = chunk.text.lower()
            score = text_lower.count(topic_hint_lower)
            # Small bonus if topic appears in filename
            if topic_hint_lower in chunk.source.lower():
                score += 2
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:k]]
