from fastapi import APIRouter, Request

from backend.schemas.prediction import (
    ComparisonPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["prediction"],
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:

    service = (
        request.app.state.prediction_service
    )

    label, confidence = service.predict(
        text=payload.text,
        model_name=payload.model,
    )

    return PredictionResponse(
        model=payload.model,
        label=label,
        confidence=confidence,
    )


@router.post(
    "/predict/compare",
    response_model=ComparisonPredictionResponse,
)
def compare_predictions(
    payload: PredictionRequest,
    request: Request,
) -> ComparisonPredictionResponse:

    service = (
        request.app.state.prediction_service
    )

    result = service.compare(
        text=payload.text
    )

    return ComparisonPredictionResponse(
        **result
    )