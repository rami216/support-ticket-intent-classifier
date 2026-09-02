import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer


CHALLENGE_PATH = Path("data/challenge/challenge.csv")
ONNX_DIR = Path("artifacts/distilbert/onnx")
ONNX_PATH = ONNX_DIR / "model_quantized.onnx"
LABEL_MAPPING_PATH = Path("artifacts/distilbert/label_to_id.json")


def main():
    df = pd.read_csv(CHALLENGE_PATH)

    tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR)

    with open(LABEL_MAPPING_PATH, "r") as file:
        label_to_id = json.load(file)

    id_to_label = {
        label_id: label
        for label, label_id in label_to_id.items()
    }

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    correct = 0

    for _, row in df.iterrows():
        text = row["text"]
        true_label = row["label"]

        encoded = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        outputs = session.run(
            None,
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )

        logits = outputs[0]
        predicted_id = int(np.argmax(logits, axis=1)[0])
        predicted_label = id_to_label[predicted_id]

        if predicted_label == true_label:
            correct += 1
        else:
            print(
                f"WRONG | true={true_label} | "
                f"pred={predicted_label} | "
                f"text={text}"
            )

    total = len(df)
    accuracy = correct / total

    print()
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()