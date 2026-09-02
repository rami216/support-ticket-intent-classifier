from pathlib import Path

import pandas as pd

from src.data.schema import REQUIRED_COLUMNS


class DataValidationError(Exception):
    pass


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise DataValidationError(
            f"Dataset does not exist: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise DataValidationError(
            f"Expected a CSV file, got: {file_path.suffix}"
        )

    try:
        df = pd.read_csv(file_path)

    except Exception as exc:
        raise DataValidationError(
            f"Failed to read dataset: {exc}"
        ) from exc

    return df


def validate_columns(df: pd.DataFrame) -> None:
    actual_columns = set(df.columns)

    missing_columns = REQUIRED_COLUMNS - actual_columns

    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    missing_text = df["text"].isna().sum()
    missing_labels = df["label"].isna().sum()

    errors = []

    if missing_text > 0:
        errors.append(
            f"'text' contains {missing_text} missing values"
        )

    if missing_labels > 0:
        errors.append(
            f"'label' contains {missing_labels} missing values"
        )

    if errors:
        raise DataValidationError("; ".join(errors))


def validate_empty_text(df: pd.DataFrame) -> None:
    empty_mask = (
        df["text"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    empty_count = empty_mask.sum()

    if empty_count > 0:
        raise DataValidationError(
            f"'text' contains {empty_count} empty strings"
        )


def validate_labels(df: pd.DataFrame) -> None:
    unique_labels = df["label"].unique()

    if len(unique_labels) < 2:
        raise DataValidationError(
            "Dataset must contain at least 2 labels."
        )

def validate_duplicates(df: pd.DataFrame) -> None:
    duplicate_count = df.duplicated(
        subset=["text", "label"]
    ).sum()

    if duplicate_count > 0:
        raise DataValidationError(
            f"Dataset contains {duplicate_count} duplicate rows"
        )


def validate_conflicting_labels(df: pd.DataFrame) -> None:
    label_counts_per_text = (
        df.groupby("text")["label"]
        .nunique()
    )

    conflicting_count = (
        label_counts_per_text > 1
    ).sum()

    if conflicting_count > 0:
        raise DataValidationError(
            f"Found {conflicting_count} text samples "
            "with conflicting labels"
        )


def validate_dataset(df: pd.DataFrame) -> None:
    validate_columns(df)
    validate_missing_values(df)
    validate_empty_text(df)
    validate_labels(df)
    validate_duplicates(df)
    validate_conflicting_labels(df)