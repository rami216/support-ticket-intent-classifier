import json
from pathlib import Path

import pandas as pd
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import build_label_mapping
from src.models.distilbert import build_distilbert_model
from src.transformer.dataset import DistilBertTicketDataset
from transformers import get_linear_schedule_with_warmup

MODEL_NAME = "distilbert-base-uncased"

TRAIN_PATH = "data/splits/train.csv"
VALIDATION_PATH = "data/splits/validation.csv"

OUTPUT_DIR = Path("artifacts/distilbert")

BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
EPOCHS = 4
PATIENCE = 2


def evaluate(
    model,
    data_loader,
    device,
):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch[
                "attention_mask"
            ].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_examples += batch_size

    average_loss = (
        total_loss / total_examples
    )

    accuracy = (
        total_correct / total_examples
    )

    return average_loss, accuracy


def main() -> None:
    train_df = pd.read_csv(
        TRAIN_PATH
    )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    label_to_id, id_to_label = (
        build_label_mapping(
            train_df["label"].tolist()
        )
    )

    print(
        f"Number of classes: "
        f"{len(label_to_id)}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    train_dataset = (
        DistilBertTicketDataset(
            texts=train_df[
                "text"
            ].tolist(),
            labels=train_df[
                "label"
            ].tolist(),
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            max_length=MAX_LENGTH,
        )
    )

    validation_dataset = (
        DistilBertTicketDataset(
            texts=validation_df[
                "text"
            ].tolist(),
            labels=validation_df[
                "label"
            ].tolist(),
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            max_length=MAX_LENGTH,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    total_training_steps = (
      len(train_loader) * EPOCHS
    )
    warmup_steps = int(
      0.1 * total_training_steps
    )
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cpu")

    print(f"Device: {device}")

    model = build_distilbert_model(
        model_name=MODEL_NAME,
        num_classes=len(
            label_to_id
        ),
    )

    model = model.to(device)
    

    optimizer = torch.optim.AdamW(
      model.parameters(),
      lr=2e-5,
      weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_training_steps,
      )
    
    

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_DIR / "label_to_id.json",
        "w",
    ) as file:
        json.dump(
            label_to_id,
            file,
            indent=2,
        )

    best_validation_loss = (
        float("inf")
    )

    epochs_without_improvement = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):
            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "label"
            ].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            loss.backward()

            optimizer.step()
            scheduler.step()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            current_batch_size = (
                labels.size(0)
            )

            total_train_loss += (
                loss.item()
                * current_batch_size
            )

            total_train_correct += (
                predictions == labels
            ).sum().item()

            total_train_examples += (
                current_batch_size
            )

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Batch {batch_index}/"
                    f"{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = (
            total_train_loss
            / total_train_examples
        )

        train_accuracy = (
            total_train_correct
            / total_train_examples
        )

        validation_loss, validation_accuracy = (
            evaluate(
                model=model,
                data_loader=validation_loader,
                device=device,
            )
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            model.save_pretrained(
                OUTPUT_DIR / "model"
            )

            tokenizer.save_pretrained(
                OUTPUT_DIR / "model"
            )

            print(
                "Saved new best model."
            )

        else:
            epochs_without_improvement += 1

            print(
                "Validation loss "
                "did not improve."
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                "Early stopping."
            )
            break


if __name__ == "__main__":
    main()