import torch

from src.data.vocabulary import PAD_ID


def collate_batch(batch):
    lengths = torch.tensor(
        [len(sample["input_ids"]) for sample in batch],
        dtype=torch.long,
    )

    max_length = lengths.max().item()
    batch_size = len(batch)

    padded_input_ids = torch.full(
        size=(batch_size, max_length),
        fill_value=PAD_ID,
        dtype=torch.long,
    )

    for i, sample in enumerate(batch):
        input_ids = sample["input_ids"]

        padded_input_ids[
            i,
            :len(input_ids)
        ] = input_ids

    labels = torch.stack(
        [sample["label"] for sample in batch]
    )

    return {
        "input_ids": padded_input_ids,
        "lengths": lengths,
        "labels": labels,
    }