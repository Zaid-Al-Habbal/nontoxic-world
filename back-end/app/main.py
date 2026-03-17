import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    HealthResponse,
    ModelsResponse,
    ModelInfo,
    LabelThresholds,
    PredictRequest,
    PredictResponse,
)
from app.services import (
    load_all_models,
    get_loaded_model_names,
    predict,
    _loaded,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model metadata for the /models endpoint
# Hardcoded from project_summary.md results
# ---------------------------------------------------------------------------

_MODEL_META: dict[str, dict] = {
    "StackedBiGRUModel": {
        "tokenizer_type": "bbpe",
        "pr_auc": 0.62,
        "macro_f1": 0.60,
    },
    "StackedBiGRUWithPretrainedEmbedModel": {
        "tokenizer_type": "bert",
        "pr_auc": 0.69,
        "macro_f1": 0.67,
    },
    "StackedBiGRUWithScaledAttention": {
        "tokenizer_type": "bert",
        "pr_auc": 0.69,
        "macro_f1": 0.67,
    },
}

# ---------------------------------------------------------------------------
# Lifespan: load models on startup, release on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    load_all_models()
    loaded = get_loaded_model_names()
    if not loaded:
        logger.error("No models were loaded successfully. All predictions will fail.")
    else:
        logger.info(f"Ready. Loaded models: {loaded}")
    yield
    logger.info("Shutting down.")
    _loaded.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nontoxic World API",
    description=(
        "Multi-label toxicity classifier for Wikipedia comments. "
        "Detects: toxic, severe_toxic, obscene, threat, insult, identity_hate."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Utility"],
)
def health() -> HealthResponse:
    return HealthResponse(loaded_models=get_loaded_model_names())


@app.get(
    "/models",
    response_model=ModelsResponse,
    summary="List available models with their metadata",
    tags=["Utility"],
)
def list_models() -> ModelsResponse:
    loaded_names = get_loaded_model_names()
    models = []
    for name, meta in _MODEL_META.items():
        if name not in loaded_names:
            continue  # skip models that failed to load
        entry = _loaded[name]
        models.append(
            ModelInfo(
                model_name=name,
                tokenizer_type=meta["tokenizer_type"],
                pr_auc=meta["pr_auc"],
                macro_f1=meta["macro_f1"],
                thresholds=LabelThresholds(**entry.thresholds),
            )
        )
    return ModelsResponse(models=models)


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a comment for toxicity across 6 labels",
    tags=["Prediction"],
)
def predict_toxicity(request: PredictRequest) -> PredictResponse:
    loaded_names = get_loaded_model_names()

    if request.model_name not in loaded_names:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Model '{request.model_name}' is currently unavailable. "
                f"Available models: {loaded_names}"
            ),
        )

    try:
        return predict(model_name=request.model_name, text=request.text)
    except Exception as exc:
        logger.exception(f"Prediction failed for model '{request.model_name}'.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        ) from exc