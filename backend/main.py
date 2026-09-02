from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.health import router as health_router
from backend.api.routes.prediction import router as prediction_router
from backend.core.config import get_settings
from backend.core.exceptions import global_exception_handler
from backend.core.logging import configure_logging
from backend.middleware.request_logging import (
    request_logging_middleware,
)
from backend.services.prediction_service import PredictionService

from src.inference.predictor import TicketPredictor
from src.inference.distilbert_predictor import (
    DistilBertPredictor,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    rnn_predictor = TicketPredictor(
        model_path=settings.model_path,
        vocab_path=settings.vocab_path,
        config_path=settings.model_config_path,
        label_mapping_path=settings.label_mapping_path,
        device=device,
    )

    distilbert_predictor = DistilBertPredictor(
        model_dir=settings.distilbert_model_name,
        label_mapping_path=(
            settings.distilbert_label_mapping_path
        ),
        device=device,
    )

    app.state.prediction_service = PredictionService(
        rnn_predictor=rnn_predictor,
        distilbert_predictor=distilbert_predictor,
        max_concurrent_predictions=(
            settings.max_concurrent_predictions
        ),
    )

    yield

    app.state.prediction_service = None


settings = get_settings()

configure_logging()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(
    request_logging_middleware
)

app.add_exception_handler(
    Exception,
    global_exception_handler,
)

app.include_router(
    health_router,
)

app.include_router(
    prediction_router,
)