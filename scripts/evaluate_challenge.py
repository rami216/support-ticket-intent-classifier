from pathlib import Path
import json

import pandas as pd
import torch

from src.inference.predictor import TicketPredictor


CHALLENGE_PATH = Path(
    "data/challenge/challenge.csv"
)

MODEL_PATH = Path(
    "artifacts/models/best_rnn.pt"
)

VOCAB_PATH = Path(
    "artifacts/vocab/word_to_id.json"
)

CONFIG_PATH = Path(
    "artifacts/config/model_config.json"
)

LABEL_MAPPING_PATH = Path(
    "artifacts/config/label_to_id.json"
)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    predictor = TicketPredictor(
        model_path=MODEL_PATH,
        vocab_path=VOCAB_PATH,
        config_path=CONFIG_PATH,
        label_mapping_path=LABEL_MAPPING_PATH,
        device=device,
    )

    df = pd.read_csv(
        CHALLENGE_PATH
    )

    correct = 0

    results = []

    for _, row in df.iterrows():
        text = row["text"]
        true_label = row["label"]

        predicted_label, confidence = (
            predictor.predict(text)
        )

        is_correct = (
            predicted_label == true_label
        )

        if is_correct:
            correct += 1

        results.append(
            {
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "correct": is_correct,
            }
        )

    accuracy = correct / len(df)

    print()
    print("Challenge Set Results")
    print("---------------------")

    print(
        f"Total examples: {len(df)}"
    )

    print(
        f"Correct: {correct}"
    )

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    print()
    print("Wrong Predictions")
    print("-----------------")

    for result in results:
        if not result["correct"]:
            print()

            print(
                f"Text: {result['text']}"
            )

            print(
                f"True: {result['true_label']}"
            )

            print(
                f"Predicted: "
                f"{result['predicted_label']}"
            )

            print(
                f"Confidence: "
                f"{result['confidence']:.2%}"
            )


if __name__ == "__main__":
    main()