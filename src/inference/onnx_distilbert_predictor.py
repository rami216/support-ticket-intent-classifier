import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXDistilBertPredictor:
    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path,
        label_mapping_path: str | Path,
    ):
        with open(label_mapping_path, "r") as file:
            label_to_id = json.load(file)

        self.id_to_label = {
            label_id: label
            for label, label_id in label_to_id.items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path
        )

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(self, text: str) -> tuple[str, float]:
        encoded = self.tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=128,
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": encoded["input_ids"].astype(np.int64),
                "attention_mask": encoded["attention_mask"].astype(np.int64),
            },
        )

        logits = outputs[0]

        logits = logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )

        probabilities = np.exp(logits)
        probabilities = probabilities / np.sum(
            probabilities,
            axis=1,
            keepdims=True,
        )

        predicted_id = int(
            np.argmax(probabilities, axis=1)[0]
        )

        confidence = float(
            probabilities[0, predicted_id]
        )

        predicted_label = self.id_to_label[
            predicted_id
        ]

        return predicted_label, confidence