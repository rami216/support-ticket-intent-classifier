import threading


class PredictionService:
    def __init__(
        self,
        rnn_predictor,
        distilbert_predictor,
        max_concurrent_predictions: int,
    ):
        self.rnn_predictor = rnn_predictor
        self.distilbert_predictor = distilbert_predictor

        self.semaphore = threading.BoundedSemaphore(
            max_concurrent_predictions
        )

    def predict(
        self,
        text: str,
        model_name: str,
    ):
        with self.semaphore:
            if model_name == "rnn":
                label, confidence = self.rnn_predictor.predict(text)

            elif model_name == "distilbert":
                label, confidence = self.distilbert_predictor.predict(text)

            else:
                raise ValueError(
                    f"Unknown model: {model_name}"
                )

        return {
            "model": model_name,
            "label": label,
            "confidence": confidence,
        }

    def compare(
        self,
        text: str,
    ):
        with self.semaphore:
            rnn_label, rnn_confidence = (
                self.rnn_predictor.predict(text)
            )

            distilbert_label, distilbert_confidence = (
                self.distilbert_predictor.predict(text)
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