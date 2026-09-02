from fastapi import (
    APIRouter,
    Request,
    Response,
    status,
)

router = APIRouter(
    tags=["health"],
)


@router.get("/health")
def health():
    return {
        "status": "ok"
    }


@router.get("/ready")
def readiness(
    request: Request,
    response: Response,
):
    prediction_service = getattr(
        request.app.state,
        "prediction_service",
        None,
    )

    if prediction_service is None:
        response.status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
        )

        return {
            "status": "not_ready"
        }

    return {
        "status": "ready"
    }