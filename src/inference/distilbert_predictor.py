import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers import DistilBertForSequenceClassification


class DistilBertPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        label_mapping_path: str | Path,
        device: torch.device,
    ):
        self.device = device

        with open(label_mapping_path, "r") as file:
            label_to_id = json.load(file)

        self.id_to_label = {
            label_id: label
            for label, label_id in label_to_id.items()
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir
        )

        self.model = (
            DistilBertForSequenceClassification
            .from_pretrained(model_dir)
        )

        self.model = self.model.to(device)
        self.model.eval()

    def predict(
        self,
        text: str,
    ) -> tuple[str, float]:

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encoded[
            "input_ids"
        ].to(self.device)

        attention_mask = encoded[
            "attention_mask"
        ].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

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