from pathlib import Path
import json

import numpy as np
import onnxruntime as ort

from src.data.vocabulary import encode_text


class ONNXRNNPredictor:
    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        config_path: str | Path,
        label_mapping_path: str | Path,
        max_length: int = 128,
    ):
        self.max_length = max_length

        with open(vocab_path, "r") as file:
            self.word_to_id = json.load(file)

        with open(config_path, "r") as file:
            model_config = json.load(file)

        with open(label_mapping_path, "r") as file:
            label_to_id = json.load(file)

        self.padding_idx = model_config["padding_idx"]

        self.id_to_label = {
            label_id: label
            for label, label_id in label_to_id.items()
        }

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

    def predict(
        self,
        text: str,
    ) -> tuple[str, float]:

        input_ids = encode_text(
            text=text,
            word_to_id=self.word_to_id,
        )

        # Keep at most 128 tokens.
        input_ids = input_ids[:self.max_length]

        length = len(input_ids)

        # Pad to exactly 128 tokens.
        padded_ids = input_ids + [
            self.padding_idx
        ] * (
            self.max_length - length
        )

        input_array = np.array(
            [padded_ids],
            dtype=np.int64,
        )

        lengths_array = np.array(
            [length],
            dtype=np.int64,
        )

        outputs = self.session.run(
            None,
            {
                "input_ids": input_array,
                "lengths": lengths_array,
            },
        )

        logits = outputs[0]

        # Stable softmax
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
            np.argmax(
                probabilities,
                axis=1,
            )[0]
        )

        confidence = float(
            probabilities[
                0,
                predicted_id,
            ]
        )

        predicted_label = self.id_to_label[
            predicted_id
        ]

        return predicted_label, confidence