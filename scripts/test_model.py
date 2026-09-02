import torch

from src.models.rnn import RNNClassifier


def main() -> None:
    vocab_size = 20
    embedding_dim = 5
    hidden_size = 6
    num_classes = 7
    padding_idx = 0

    model = RNNClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        padding_idx=padding_idx,
    )

    input_ids = torch.tensor(
        [
            [2, 3, 4, 0],
            [5, 6, 7, 8],
        ],
        dtype=torch.long,
    )

    lengths = torch.tensor(
        [3, 4],
        dtype=torch.long,
    )

    logits = model(
        input_ids=input_ids,
        lengths=lengths,
    )

    print("Input IDs:")
    print(input_ids)

    print("\nInput shape:")
    print(input_ids.shape)

    print("\nLengths:")
    print(lengths)

    print("\nLogits:")
    print(logits)

    print("\nLogits shape:")
    print(logits.shape)


if __name__ == "__main__":
    main()