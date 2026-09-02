from pathlib import Path
import json

import torch

from src.data.vocabulary import encode_text
from src.models.rnn import RNNClassifier


class TicketPredictor:
    def __init__(
        self,
        model_path: str | Path,
        vocab_path: str | Path,
        config_path: str | Path,
        label_mapping_path: str | Path,
        device: torch.device,
    ):
        self.device = device

        with open(vocab_path, "r") as file:
            self.word_to_id = json.load(file)

        with open(config_path, "r") as file:
            model_config = json.load(file)

        with open(label_mapping_path, "r") as file:
            label_to_id = json.load(file)

        self.id_to_label = {
            label_id: label
            for label, label_id in label_to_id.items()
        }

        self.model = RNNClassifier(
            vocab_size=model_config["vocab_size"],
            embedding_dim=model_config["embedding_dim"],
            hidden_size=model_config["hidden_size"],
            num_classes=model_config["num_classes"],
            padding_idx=model_config["padding_idx"],
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            )
        )

        self.model = self.model.to(device)

        self.model.eval()

    def predict(
        self,
        text: str,
    ) -> tuple[str, float]:

        input_ids = encode_text(
            text=text,
            word_to_id=self.word_to_id,
        )

        input_tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        lengths = torch.tensor(
            [len(input_ids)],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=input_tensor,
                lengths=lengths,
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predicted_id = torch.argmax(
                probabilities,
                dim=1,
            ).item()

            confidence = probabilities[
                0,
                predicted_id,
            ].item()

        predicted_label = self.id_to_label[
            predicted_id
        ]

        return predicted_label, confidence