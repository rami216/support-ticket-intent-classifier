from pathlib import Path
import json

import torch

from src.models.rnn import RNNClassifier


MODEL_PATH = Path("artifacts/models/best_rnn.pt")
CONFIG_PATH = Path("artifacts/config/model_config.json")

OUTPUT_DIR = Path("artifacts/rnn/onnx")
OUTPUT_PATH = OUTPUT_DIR / "rnn.onnx"

MAX_LENGTH = 128


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    model = RNNClassifier(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        num_classes=config["num_classes"],
        padding_idx=config["padding_idx"],
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
        )
    )

    model.eval()

    # Example batch: 1 sentence, always 128 positions.
    dummy_input_ids = torch.full(
        (1, MAX_LENGTH),
        fill_value=config["padding_idx"],
        dtype=torch.long,
    )

    # Pretend the real sentence contains 3 tokens.
    dummy_lengths = torch.tensor(
        [3],
        dtype=torch.long,
    )

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_lengths),
        OUTPUT_PATH,
        input_names=[
            "input_ids",
            "lengths",
        ],
        output_names=[
            "logits",
        ],
        dynamic_axes={
            "input_ids": {
                0: "batch_size",
            },
            "lengths": {
                0: "batch_size",
            },
            "logits": {
                0: "batch_size",
            },
        },
        opset_version=17,
    )

    print("RNN ONNX export completed.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()