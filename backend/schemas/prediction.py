from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
    )

    model: Literal[
        "rnn",
        "distilbert",
    ] = "distilbert"

    @field_validator("text")
    @classmethod
    def validate_text(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "text must not be empty"
            )

        return value


class PredictionResponse(BaseModel):
    model: str
    label: str
    confidence: float


class ModelPrediction(BaseModel):
    label: str
    confidence: float


class ComparisonPredictionResponse(BaseModel):
    rnn: ModelPrediction
    distilbert: ModelPrediction