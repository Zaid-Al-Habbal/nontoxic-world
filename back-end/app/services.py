import json
import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import tokenizers

from app.models import (
    StackedBiGRUModel,
    StackedBiGRUWithPretrainedEmbedModel,
    StackedBiGRUWithScaledAttention,
)
from app.preprocessing import preprocess
from app.schemas import (
    LABELS,
    LabelPredictions,
    LabelProbabilities,
    LabelThresholds,
    PredictResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TokenizerFamily = Literal["bbpe", "bert"]

# ---------------------------------------------------------------------------
# Model registry
# Each entry declares the HuggingFace repo, filenames, and tokenizer family.
# The threshold file must be a JSON mapping label -> float, e.g.:
#   {"toxic": 0.21, "severe_toxic": 0.18, ...}
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, dict] = {
    "StackedBiGRUModel": {
        "hf_repo": "Zaid-Al-Habbal/nontoxic-world",
        "checkpoint_file": "stacked_bigru_model.pth",
        "threshold_file": "stacked_bigru_model_thresholds.json",
        "tokenizer_family": "bbpe",
        "tokenizer_file": "bbpe_tokenizer.json",  # uploaded alongside checkpoint
    },
    "StackedBiGRUWithPretrainedEmbedModel": {
        "hf_repo": "Zaid-Al-Habbal/nontoxic-world",
        "checkpoint_file": "pretrained_embed_stacked_bigru.pth",
        "threshold_file": "pretrained_embed_stacked_bigru_thresholds.json",
        "tokenizer_family": "bert",
        "tokenizer_file": None,  # loaded directly from bert-base-uncased
    },
    "StackedBiGRUWithScaledAttention": {
        "hf_repo": "Zaid-Al-Habbal/nontoxic-world",
        "checkpoint_file": "stacked_bigru_with_attention.pth",
        "threshold_file": "stacked_bigru_with_attention_thresholds.json",
        "tokenizer_family": "bert",
        "tokenizer_file": None,
    },
}

# ---------------------------------------------------------------------------
# Internal dataclass for a fully loaded model entry
# ---------------------------------------------------------------------------

@dataclass
class _LoadedModel:
    model: nn.Module
    tokenizer: PreTrainedTokenizerFast
    thresholds: dict[str, float]  # label -> optimal threshold
    tokenizer_family: TokenizerFamily


# ---------------------------------------------------------------------------
# Module-level state (populated during app startup)
# ---------------------------------------------------------------------------

_loaded: dict[str, _LoadedModel] = {}
_device: torch.device = torch.device("cpu")

# Shared BERT embeddings — loaded once and reused by both BERT-family models
_bert_embeddings: nn.Embedding | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_bert_embeddings() -> nn.Embedding:
    hf_repo = "Zaid-Al-Habbal/nontoxic-world"
    logger.info(f"Downloading checkpoint for bert_word_embeddings from {hf_repo}...")
    embeddings = hf_hub_download(
        repo_id=hf_repo, filename="bert_word_embeddings.pt"
    )
    # Detach from the full BERT model to avoid carrying unnecessary weights
    embeddings = nn.Embedding.from_pretrained(
        embeddings.weight.data.clone(), freeze=True
    )
    return embeddings


def _load_bbpe_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    """Reconstruct a PreTrainedTokenizerFast from a saved BBPE JSON file."""
    tok = tokenizers.Tokenizer.from_file(tokenizer_path)
    return PreTrainedTokenizerFast(tokenizer_object=tok)


def _build_model(
    model_name: str,
    checkpoint: dict,
    bert_embeddings: nn.Embedding | None,
) -> nn.Module:
    """Reconstruct a model from its checkpoint config dict."""
    cfg = checkpoint.get("config", {})
    n_layers = cfg.get("num_layers", 2)
    hidden_dim = cfg.get("hidden_dim", 32)
    dropout = cfg.get("dropout", 0.4)

    if model_name == "StackedBiGRUModel":
        vocab_size = cfg.get("vocab_size", 30_000)
        embed_dim = cfg.get("embed_dim", 256)
        return StackedBiGRUModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if model_name == "StackedBiGRUWithPretrainedEmbedModel":
        return StackedBiGRUWithPretrainedEmbedModel(
            pretrained_embeddings=bert_embeddings,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    if model_name == "StackedBiGRUWithScaledAttention":
        return StackedBiGRUWithScaledAttention(
            pretrained_embeddings=bert_embeddings,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    raise ValueError(f"Unknown model name: {model_name}")


def _load_single_model(model_name: str) -> _LoadedModel:
    entry = _REGISTRY[model_name]
    hf_repo = entry["hf_repo"]
    family: TokenizerFamily = entry["tokenizer_family"]

    logger.info(f"Downloading checkpoint for {model_name} from {hf_repo}...")
    checkpoint_path = hf_hub_download(
        repo_id=hf_repo, filename=entry["checkpoint_file"]
    )
    threshold_path = hf_hub_download(
        repo_id=hf_repo, filename=entry["threshold_file"]
    )

    checkpoint = torch.load(checkpoint_path, map_location=_device)

    with open(threshold_path) as f:
        thresholds: dict[str, float] = json.load(f)

    # Validate threshold keys match expected labels
    missing = set(LABELS) - set(thresholds.keys())
    if missing:
        raise ValueError(f"Threshold file for {model_name} is missing labels: {missing}")

    # Tokenizer
    if family == "bbpe":
        tokenizer_path = hf_hub_download(
            repo_id=hf_repo, filename=entry["tokenizer_file"]
        )
        tokenizer = _load_bbpe_tokenizer(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Model
    global _bert_embeddings
    if family == "bert" and _bert_embeddings is None:
        _bert_embeddings = _load_bert_embeddings().to(_device)

    model = _build_model(
        model_name,
        checkpoint,
        bert_embeddings=_bert_embeddings if family == "bert" else None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()

    logger.info(f"{model_name} loaded successfully on {_device}.")
    return _LoadedModel(
        model=model,
        tokenizer=tokenizer,
        thresholds=thresholds,
        tokenizer_family=family,
    )


# ---------------------------------------------------------------------------
# Public lifecycle functions (called from main.py lifespan)
# ---------------------------------------------------------------------------

def load_all_models() -> None:
    """Load every registered model into memory. Called once at app startup."""
    global _device
    _device = _get_device()
    logger.info(f"Inference device: {_device}")

    for model_name in _REGISTRY:
        try:
            _loaded[model_name] = _load_single_model(model_name)
        except Exception:
            logger.exception(f"Failed to load {model_name}. It will be unavailable.")


def get_loaded_model_names() -> list[str]:
    return list(_loaded.keys())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _tokenize(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, torch.Tensor]:
    """Tokenize a single string and return tensors on the correct device."""
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=256,
        padding=False,
        return_tensors="pt",
    )
    return {k: v.to(_device) for k, v in encoding.items()}


def predict(model_name: str, text: str) -> PredictResponse:
    if model_name not in _loaded:
        raise KeyError(
            f"Model '{model_name}' is not loaded. "
            f"Available: {list(_loaded.keys())}"
        )

    entry = _loaded[model_name]
    preprocessed = preprocess(text)

    encoding = _tokenize(preprocessed, entry.tokenizer)

    with torch.no_grad():
        logits = entry.model(encoding)          # (1, 6)
        probs = torch.sigmoid(logits).squeeze(0)  # (6,)

    probs_list = probs.cpu().tolist()

    prob_dict = dict(zip(LABELS, probs_list))
    pred_dict = {
        label: prob >= entry.thresholds[label]
        for label, prob in prob_dict.items()
    }

    return PredictResponse(
        original_text=text,
        preprocessed_text=preprocessed,
        probabilities=LabelProbabilities(**prob_dict),
        predictions=LabelPredictions(**pred_dict),
        thresholds_used=LabelThresholds(**entry.thresholds),
        model_used=model_name,
        is_toxic=any(pred_dict.values()),
    )