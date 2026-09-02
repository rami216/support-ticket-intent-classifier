import torch
from torch.utils.data import Dataset

from src.data.vocabulary import encode_text


def build_label_mapping(
    labels: list[str],
) -> tuple[
    dict[str, int],
    dict[int, str],
]:
    unique_labels = sorted(
        set(labels)
    )

    label_to_id = {
        label: index
        for index, label in enumerate(unique_labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    return label_to_id, id_to_label


class TicketDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        word_to_id: dict[str, int],
        label_to_id: dict[str, int],
    ):
        self.texts = texts
        self.labels = labels
        self.word_to_id = word_to_id
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(
        self,
        index: int,
    ):
        text = self.texts[index]
        label = self.labels[index]

        input_ids = encode_text(
            text=text,
            word_to_id=self.word_to_id,
        )

        label_id = self.label_to_id[label]

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "label": torch.tensor(
                label_id,
                dtype=torch.long,
            ),
        }