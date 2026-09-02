import pandas as pd

from src.data.vocabulary import (
    build_vocabulary,
    encode_text,
)


TRAIN_PATH = "data/splits/train.csv"


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)

    word_to_id = build_vocabulary(
        texts=train_df["text"].tolist(),
        min_frequency=1,
    )

    print("Vocabulary size:")
    print(len(word_to_id))

    print("\nFirst vocabulary entries:")
    for word, word_id in list(word_to_id.items())[:15]:
        print(word, "->", word_id)

    example_text = "I want a refund"

    encoded = encode_text(
        text=example_text,
        word_to_id=word_to_id,
    )

    print("\nExample text:")
    print(example_text)

    print("\nEncoded:")
    print(encoded)


if __name__ == "__main__":
    main()