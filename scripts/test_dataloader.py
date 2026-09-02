import pandas as pd
from torch.utils.data import DataLoader

from src.data.collate import collate_batch
from src.data.dataset import TicketDataset
from src.data.vocabulary import build_vocabulary


TRAIN_PATH = "data/splits/train.csv"


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)

    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()

    word_to_id = build_vocabulary(
        texts=texts,
        min_frequency=1,
    )

    train_dataset = TicketDataset(
        texts=texts,
        labels=labels,
        word_to_id=word_to_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=3,
        shuffle=True,
        collate_fn=collate_batch,
    )

    batch = next(iter(train_loader))

    print("Input IDs:")
    print(batch["input_ids"])

    print("\nLengths:")
    print(batch["lengths"])

    print("\nLabels:")
    print(batch["labels"])

    print("\nShapes:")
    print("input_ids:", batch["input_ids"].shape)
    print("lengths:", batch["lengths"].shape)
    print("labels:", batch["labels"].shape)


if __name__ == "__main__":
    main()