from pathlib import Path

import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification


MODEL_DIR = Path("artifacts/distilbert/model")
ONNX_DIR = Path("artifacts/distilbert/onnx")
ONNX_PATH = ONNX_DIR / "model.onnx"


def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    model.eval()

    sample = tokenizer(
        "Where is my package?",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )

    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        ONNX_PATH,
        input_names=[
            "input_ids",
            "attention_mask",
        ],
        output_names=[
            "logits",
        ],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
                1: "sequence_length",
            },
            "attention_mask": {
                0: "batch_size",
                1: "sequence_length",
            },
            "logits": {
                0: "batch_size",
            },
        },
        opset_version=17,
    )

    tokenizer.save_pretrained(ONNX_DIR)

    print("ONNX export completed.")
    print(f"Saved to: {ONNX_PATH}")


if __name__ == "__main__":
    main()