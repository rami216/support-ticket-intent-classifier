from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def calculate_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, float]:

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    
def create_classification_report(
    y_true: list[int],
    y_pred: list[int],
    target_names: list[str],
) -> str:

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0,
    )

    return report

def create_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
) -> list[list[int]]:

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    return matrix.tolist()