from pathlib import Path

from sophia.classifier import TopicClassifier
from sophia.config import AppConfig


def main() -> None:
    cfg = AppConfig()
    cfg.ensure_directories()

    if not cfg.data_path.exists():
        raise SystemExit(f"Data file not found at {cfg.data_path}. Create questions_labeled.csv first.")

    print(f"Using data: {cfg.data_path}")
    print(f"Saving model to: {cfg.model_path}")

    clf = TopicClassifier(cfg.model_path)
    clf.train(cfg.data_path)


if __name__ == "__main__":
    main()
