import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


ONNX_DIR = Path("artifacts/distilbert/onnx")
ONNX_PATH = ONNX_DIR / "model_quantized.onnx"
LABEL_MAPPING_PATH = Path("artifacts/distilbert/label_to_id.json")


def main():
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

    text = "Where is my package?"

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

    print("Text:", text)
    print("Logits shape:", logits.shape)
    print("Predicted ID:", predicted_id)
    print("Predicted label:", predicted_label)


if __name__ == "__main__":
    main()