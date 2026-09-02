from src.data.dataset import TicketDataset
from src.data.vocabulary import build_vocabulary


texts = [
    "I want a refund",
    "My package is late",
    "I cannot login",
]

labels = [
    "refund",
    "delivery",
    "account",
]


word_to_id = build_vocabulary(texts)

dataset = TicketDataset(
    texts=texts,
    labels=labels,
    word_to_id=word_to_id,
)


print("Vocabulary:")
print(word_to_id)

print("\nDataset length:")
print(len(dataset))

print("\nSample 0:")
print(dataset[0])

print("\nSample 1:")
print(dataset[1])