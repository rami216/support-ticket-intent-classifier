import pandas as pd
import torch

from src.inference.distilbert_predictor import (
    DistilBertPredictor,
)


CHALLENGE_PATH = "data/challenge/challenge.csv"

MODEL_DIR = "artifacts/distilbert/model"

LABEL_MAPPING_PATH = (
    "artifacts/distilbert/label_to_id.json"
)


def main() -> None:
    device = torch.device("cpu")

    predictor = DistilBertPredictor(
        model_dir=MODEL_DIR,
        label_mapping_path=LABEL_MAPPING_PATH,
        device=device,
    )

    df = pd.read_csv(CHALLENGE_PATH)

    total = len(df)
    correct = 0

    wrong_predictions = []

    for _, row in df.iterrows():
        text = row["text"]
        true_label = row["label"]

        predicted_label, confidence = (
            predictor.predict(text)
        )

        if predicted_label == true_label:
            correct += 1
        else:
            wrong_predictions.append(
                {
                    "text": text,
                    "true": true_label,
                    "predicted": predicted_label,
                    "confidence": confidence,
                }
            )

    accuracy = correct / total

    print()
    print("DistilBERT Challenge Set Results")
    print("--------------------------------")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    print()
    print("Wrong Predictions")
    print("-----------------")

    for item in wrong_predictions:
        print()
        print(f'Text: {item["text"]}')
        print(f'True: {item["true"]}')
        print(
            f'Predicted: {item["predicted"]}'
        )
        print(
            f'Confidence: '
            f'{item["confidence"]:.2%}'
        )


if __name__ == "__main__":
    main()