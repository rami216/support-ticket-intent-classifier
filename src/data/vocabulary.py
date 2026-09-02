from collections import Counter

from src.data.preprocessing import tokenize


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1


def build_vocabulary(
    texts: list[str],
    min_frequency: int = 1,
) -> dict[str, int]:
    word_counts = Counter()

    for text in texts:
        word_counts.update(tokenize(text))

    word_to_id = {
        PAD_TOKEN: PAD_ID,
        UNK_TOKEN: UNK_ID,
    }

    next_id = 2

    for word, count in word_counts.items():
        if count >= min_frequency:
            word_to_id[word] = next_id
            next_id += 1

    return word_to_id


def encode_text(
    text: str,
    word_to_id: dict[str, int],
) -> list[int]:
    tokens = tokenize(text)

    token_ids = []

    for token in tokens:
        token_id = word_to_id.get(token, UNK_ID)
        token_ids.append(token_id)

    return token_ids