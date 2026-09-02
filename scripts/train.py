from pathlib import Path
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.collate import collate_batch
from src.data.dataset import (
    TicketDataset,
    build_label_mapping,
)
from src.data.vocabulary import (
    build_vocabulary,
    PAD_ID,
)
from src.models.rnn import RNNClassifier
from src.training.trainer import (
    train_one_epoch,
    evaluate,
)
from src.training.metrics import (
    create_classification_report,
    create_confusion_matrix,
)


TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"
TEST_PATH = "data/splits/test.csv"


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    word_to_id = build_vocabulary(
        texts=train_df["text"].tolist(),
        min_frequency=1,
    )

    label_to_id, id_to_label = build_label_mapping(
        train_df["label"].tolist()
    )

    vocab_path = Path(
        "artifacts/vocab/word_to_id.json"
    )

    vocab_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        vocab_path,
        "w",
    ) as file:
        json.dump(
            word_to_id,
            file,
            indent=4,
        )

    label_mapping_path = Path(
        "artifacts/config/label_to_id.json"
    )

    label_mapping_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        label_mapping_path,
        "w",
    ) as file:
        json.dump(
            label_to_id,
            file,
            indent=4,
        )

    model_config = {
        "vocab_size": len(word_to_id),
        "embedding_dim": 64,
        "hidden_size": 128,
        "num_classes": len(label_to_id),
        "padding_idx": PAD_ID,
    }

    config_path = Path(
        "artifacts/config/model_config.json"
    )

    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        config_path,
        "w",
    ) as file:
        json.dump(
            model_config,
            file,
            indent=4,
        )

    train_dataset = TicketDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].tolist(),
        word_to_id=word_to_id,
        label_to_id=label_to_id,
    )

    validation_dataset = TicketDataset(
        texts=validation_df["text"].tolist(),
        labels=validation_df["label"].tolist(),
        word_to_id=word_to_id,
        label_to_id=label_to_id,
    )

    test_dataset = TicketDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].tolist(),
        word_to_id=word_to_id,
        label_to_id=label_to_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_batch,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Vocabulary size: {len(word_to_id)}"
    )

    print(
        f"Number of classes: {len(label_to_id)}"
    )

    print(
        f"Train examples: {len(train_dataset)}"
    )

    print(
        f"Validation examples: {len(validation_dataset)}"
    )

    print(
        f"Test examples: {len(test_dataset)}"
    )

    model = RNNClassifier(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_size=model_config["hidden_size"],
        num_classes=model_config["num_classes"],
        padding_idx=model_config["padding_idx"],
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    num_epochs = 30
    patience = 3

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    checkpoint_path = Path(
        "artifacts/models/best_rnn.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            val_loss,
            val_metrics,
            _,
            _,
        ) = evaluate(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.2%} | "
            f"Val F1: {val_metrics['f1']:.2%}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print(
                "Best model saved."
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            print(
                "Early stopping."
            )
            break

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

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
    print("Final Test Results")

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.2%}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.2%}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.2%}"
    )

    print(
        f"Test F1: "
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