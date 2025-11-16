from pathlib import Path

from sophia.knowledge_base import KnowledgeBase


def test_knowledge_base_retrieval(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "ml_basics.md").write_text("Overfitting happens when a model memorizes the training data.")
    (notes_dir / "rag.md").write_text("RAG uses retrieval plus generation.")

    kb = KnowledgeBase(notes_dir)
    kb.load()

    results = kb.retrieve(topic_hint="ml_basics", k=2)
    assert results
    assert any("overfitting" in chunk.text.lower() for chunk in results)
