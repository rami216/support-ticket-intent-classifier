from src.data.preprocessing import normalize_text, tokenize


def main() -> None:
    text = "   I WANT a Refund!!!   "

    normalized = normalize_text(text)
    tokens = tokenize(text)

    print("Original:")
    print(text)

    print("\nNormalized:")
    print(normalized)

    print("\nTokens:")
    print(tokens)


if __name__ == "__main__":
    main()