from pathlib import Path

import torch

from src.inference.predictor import TicketPredictor


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    predictor = TicketPredictor(
        model_path=Path(
            "artifacts/models/best_rnn.pt"
        ),
        vocab_path=Path(
            "artifacts/vocab/word_to_id.json"
        ),
        config_path=Path(
            "artifacts/config/model_config.json"
        ),
        label_mapping_path=Path(
            "artifacts/config/label_to_id.json"
        ),
        device=device,
    )

    test_texts = [
        "I want to cancel my order",
        "My refund still has not arrived",
        "I forgot my password",
        "Can I change my shipping address?",
        "What payment methods do you accept?",
        "Where is my order?",
    ]

    for text in test_texts:
        label, confidence = predictor.predict(
            text
        )

        print()
        print(f"Text: {text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()