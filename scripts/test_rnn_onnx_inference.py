from pathlib import Path
import json

import numpy as np
import onnxruntime as ort

from src.data.vocabulary import encode_text


MODEL_PATH = Path("artifacts/rnn/onnx/rnn.onnx")
VOCAB_PATH = Path("artifacts/vocab/word_to_id.json")
CONFIG_PATH = Path("artifacts/config/model_config.json")
LABEL_MAPPING_PATH = Path("artifacts/config/label_to_id.json")

MAX_LENGTH = 128


def main():
    with open(VOCAB_PATH, "r") as file:
        word_to_id = json.load(file)

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    with open(LABEL_MAPPING_PATH, "r") as file:
        label_to_id = json.load(file)

    id_to_label = {
        label_id: label
        for label, label_id in label_to_id.items()
    }

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    text = "Where is my package?"

    input_ids = encode_text(
        text=text,
        word_to_id=word_to_id,
    )

    input_ids = input_ids[:MAX_LENGTH]

    length = len(input_ids)

    padded_ids = input_ids + [
        config["padding_idx"]
    ] * (
        MAX_LENGTH - length
    )

    input_array = np.array(
        [padded_ids],
        dtype=np.int64,
    )

    lengths_array = np.array(
        [length],
        dtype=np.int64,
    )

    outputs = session.run(
        None,
        {
            "input_ids": input_array,
            "lengths": lengths_array,
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
        np.argmax(
            probabilities,
            axis=1,
        )[0]
    )

    confidence = float(
        probabilities[0, predicted_id]
    )

    predicted_label = id_to_label[
        predicted_id
    ]

    print("Text:", text)
    print("Logits shape:", logits.shape)
    print("Predicted ID:", predicted_id)
    print("Predicted label:", predicted_label)
    print("Confidence:", confidence)


if __name__ == "__main__":
    main()