import torch
import torch.nn as nn


def main() -> None:
    logits = torch.tensor(
        [
            [0.2, -0.4, 0.1, 0.6, 2.8, -0.7, 0.3],
            [1.9, 0.2, 0.4, 0.1, 0.0, -0.3, 0.5],
        ],
        dtype=torch.float,
    )

    labels = torch.tensor(
        [4, 0],
        dtype=torch.long,
    )

    criterion = nn.CrossEntropyLoss()

    loss = criterion(
        logits,
        labels,
    )

    print("Logits shape:")
    print(logits.shape)

    print("\nLabels shape:")
    print(labels.shape)

    print("\nLoss:")
    print(loss.item())


if __name__ == "__main__":
    main()