from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class TopicClassifier:
    """
    Simple text classifier: question -> topic label.

    Uses TF-IDF features + Logistic Regression.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.pipeline: Optional[Pipeline] = None

    def train(self, csv_path: Path, test_size: float = 0.2, random_state: int = 42) -> None:
        df = pd.read_csv(csv_path)
        if "question" not in df.columns or "topic" not in df.columns:
            raise ValueError("CSV must contain 'question' and 'topic' columns.")

        X_train, X_test, y_train, y_test = train_test_split(
            df["question"],
            df["topic"],
            test_size=test_size,
            random_state=random_state,
            stratify=df["topic"],
        )

        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("Classification report:\n", report)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"No model found at {self.model_path}. Train it first.")
        self.pipeline = joblib.load(self.model_path)

    def predict_topic(self, question: str) -> str:
        if self.pipeline is None:
            self.load()
        assert self.pipeline is not None
        return str(self.pipeline.predict([question])[0])
