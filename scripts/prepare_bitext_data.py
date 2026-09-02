from pathlib import Path

import pandas as pd


SOURCE_PATH = Path(
    "data/source/"
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

OUTPUT_PATH = Path(
    "data/raw/tickets.csv"
)


def main():
    df = pd.read_csv(
        SOURCE_PATH
    )

    required_columns = {
        "instruction",
        "intent",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    prepared_df = df[
        [
            "instruction",
            "intent",
        ]
    ].copy()

    prepared_df = prepared_df.rename(
        columns={
            "instruction": "text",
            "intent": "label",
        }
    )

    prepared_df["text"] = (
        prepared_df["text"]
        .astype(str)
        .str.strip()
    )

    prepared_df["label"] = (
        prepared_df["label"]
        .astype(str)
        .str.strip()
    )

    prepared_df = prepared_df[
        (prepared_df["text"] != "")
        &
        (prepared_df["label"] != "")
    ]

    prepared_df = prepared_df.drop_duplicates(
        subset=[
            "text",
            "label",
        ]
    )

    prepared_df = prepared_df.reset_index(
        drop=True
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Source rows: {len(df)}"
    )

    print(
        f"Prepared rows: {len(prepared_df)}"
    )

    print(
        f"Number of labels: "
        f"{prepared_df['label'].nunique()}"
    )

    print()

    print(
        prepared_df["label"]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()