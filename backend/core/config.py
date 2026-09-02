from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Support Ticket Classifier API"
    app_version: str = "1.0.0"

    # RNN ONNX
    rnn_onnx_model_path: Path = Path(
        "artifacts/rnn/onnx/rnn.onnx"
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

    # DistilBERT ONNX INT8
    distilbert_onnx_model_path: Path = Path(
        "artifacts/distilbert/onnx/model_quantized.onnx"
    )
    distilbert_tokenizer_path: Path = Path(
        "artifacts/distilbert/onnx"
    )
    distilbert_label_mapping_path: Path = Path(
        "artifacts/distilbert/label_to_id.json"
    )

    max_concurrent_predictions: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()