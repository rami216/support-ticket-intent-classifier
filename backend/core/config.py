from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Support Ticket Classifier API"
    app_version: str = "1.0.0"

    # RNN
    model_path: Path = Path(
        "artifacts/models/best_rnn.pt"
    )
    vocab_path: Path = Path(
        "artifacts/vocab/word_to_id.json"
    )
    model_config_path: Path = Path(
        "artifacts/config/model_config.json"
    )
    label_mapping_path: Path = Path(
        "artifacts/config/label_to_id.json"
    )

    # DistilBERT
    distilbert_model_dir: Path = Path(
        "artifacts/distilbert/model"
    )
    distilbert_label_mapping_path: Path = Path(
        "artifacts/distilbert/label_to_id.json"
    )

    max_concurrent_predictions: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()