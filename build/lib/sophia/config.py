from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    """
    Central configuration for Sophia.
    """

    project_root: Path = Path(__file__).resolve().parents[2]
    data_path: Path = project_root / "data" / "questions_labeled.csv"
    model_path: Path = project_root / "models" / "topic_classifier.joblib"
    notes_dir: Path = project_root / "notes"

    # Modes: "study" (notes/model only) vs "research" (future: web/search)
    mode: str = "study"

    def ensure_directories(self) -> None:
        """Create expected directories if they don't exist."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
