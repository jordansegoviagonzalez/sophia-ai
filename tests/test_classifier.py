from pathlib import Path

from sophia.classifier import TopicClassifier


def test_classifier_train_and_predict(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.joblib"

    csv_path.write_text(
        "question,topic\n"
        "What is overfitting?,ml_basics\n"
        "Explain cross-entropy loss.,loss_functions\n"
        "How does attention work in Transformers?,llm_transformers\n"
        "Describe a RAG system.,rag\n"
        "How would you deploy an LLM API?,deployment\n"
    )

    clf = TopicClassifier(model_path)
    clf.train(csv_path)

    pred = clf.predict_topic("How do you prevent overfitting?")
    assert pred in {
        "ml_basics",
        "loss_functions",
        "llm_transformers",
        "rag",
        "deployment",
    }
