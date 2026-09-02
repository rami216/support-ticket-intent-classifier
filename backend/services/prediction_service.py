from threading import BoundedSemaphore

from src.inference.predictor import TicketPredictor
from src.inference.distilbert_predictor import (
    DistilBertPredictor,
)


class PredictionService:
    def __init__(
        self,
        rnn_predictor: TicketPredictor,
        distilbert_predictor: DistilBertPredictor,
        max_concurrent_predictions: int,
    ):
        self.rnn_predictor = rnn_predictor
        self.distilbert_predictor = (
            distilbert_predictor
        )

        self.semaphore = BoundedSemaphore(
            value=max_concurrent_predictions
        )

    def predict(
        self,
        text: str,
        model_name: str,
    ) -> tuple[str, float]:

        with self.semaphore:
            if model_name == "rnn":
                return self.rnn_predictor.predict(
                    text
                )

            return (
                self.distilbert_predictor.predict(
                    text
                )
            )

    def compare(
        self,
        text: str,
    ) -> dict:

        with self.semaphore:
            rnn_label, rnn_confidence = (
                self.rnn_predictor.predict(text)
            )

            (
                distilbert_label,
                distilbert_confidence,
            ) = (
                self.distilbert_predictor.predict(
                    text
                )
            )

        return {
            "rnn": {
                "label": rnn_label,
                "confidence": rnn_confidence,
            },
            "distilbert": {
                "label": distilbert_label,
                "confidence": distilbert_confidence,
            },
        }