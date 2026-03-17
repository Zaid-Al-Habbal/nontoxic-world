from pydantic import BaseModel, Field, field_validator
from typing import Literal


LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

ModelName = Literal[
    "StackedBiGRUModel",
    "StackedBiGRUWithPretrainedEmbedModel",
    "StackedBiGRUWithScaledAttention",
]


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    model_name: ModelName = Field(
        default="StackedBiGRUWithScaledAttention",
        description="Which model to use for inference.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank or whitespace only.")
        return v


class LabelProbabilities(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float


class LabelPredictions(BaseModel):
    toxic: bool
    severe_toxic: bool
    obscene: bool
    threat: bool
    insult: bool
    identity_hate: bool


class LabelThresholds(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float


class PredictResponse(BaseModel):
    original_text: str
    preprocessed_text: str
    probabilities: LabelProbabilities
    predictions: LabelPredictions
    thresholds_used: LabelThresholds
    model_used: ModelName
    is_toxic: bool = Field(
        description="True if at least one label is predicted as positive."
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    loaded_models: list[str]


class ModelInfo(BaseModel):
    model_name: ModelName
    tokenizer_type: Literal["bbpe", "bert"]
    pr_auc: float
    macro_f1: float
    thresholds: LabelThresholds


class ModelsResponse(BaseModel):
    models: list[ModelInfo]