from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.collate import collate_batch
from src.data.dataset import TicketDataset
from src.models.rnn import RNNClassifier
from src.training.trainer import evaluate
from src.training.metrics import (
    create_classification_report,
    create_confusion_matrix,
)


TEST_PATH = Path("data/splits/test.csv")

VOCAB_PATH = Path(
    "artifacts/vocab/word_to_id.json"
)

LABEL_MAPPING_PATH = Path(
    "artifacts/config/label_to_id.json"
)

MODEL_CONFIG_PATH = Path(
    "artifacts/config/model_config.json"
)

MODEL_PATH = Path(
    "artifacts/models/best_rnn.pt"
)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    with open(
        VOCAB_PATH,
        "r",
    ) as file:
        word_to_id = json.load(file)

    with open(
        LABEL_MAPPING_PATH,
        "r",
    ) as file:
        label_to_id = json.load(file)

    with open(
        MODEL_CONFIG_PATH,
        "r",
    ) as file:
        model_config = json.load(file)

    id_to_label = {
        label_id: label
        for label, label_id
        in label_to_id.items()
    }

    test_dataset = TicketDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        word_to_id=word_to_id,
        label_to_id=label_to_id,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = RNNClassifier(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_size=model_config["hidden_size"],
        num_classes=model_config["num_classes"],
        padding_idx=model_config["padding_idx"],
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    (
        test_loss,
        test_metrics,
        test_labels,
        test_predictions,
    ) = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print()
    print("Test Results")
    print("------------")

    print(
        f"Loss: {test_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.2%}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.2%}"
    )

    print(
        f"Recall: "
        f"{test_metrics['recall']:.2%}"
    )

    print(
        f"F1: "
        f"{test_metrics['f1']:.2%}"
    )

    target_names = [
        id_to_label[i]
        for i in range(
            len(id_to_label)
        )
    ]

    report = create_classification_report(
        y_true=test_labels,
        y_pred=test_predictions,
        target_names=target_names,
    )

    print()
    print("Classification Report")
    print(report)

    matrix = create_confusion_matrix(
        y_true=test_labels,
        y_pred=test_predictions,
    )

    print()
    print("Confusion Matrix")

    for row in matrix:
        print(row)


if __name__ == "__main__":
    main()