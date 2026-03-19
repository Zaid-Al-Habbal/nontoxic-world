"""
services/api_client.py
Handles all HTTP communication with the FastAPI backend.
"""
import requests
import streamlit as st
from typing import Optional

# ── Demo / fallback data ──────────────────────────────────────────────────────
DEMO_MODELS = [
    {
        "model_name": "StackedBiGRUModel",
        "tokenizer_type": "bbpe",
        "pr_auc": 0.62,
        "macro_f1": 0.60,
        "thresholds": {
            "toxic": 0.21, "severe_toxic": 0.09, "obscene": 0.18,
            "threat": 0.07, "insult": 0.18, "identity_hate": 0.08,
        },
    },
    {
        "model_name": "StackedBiGRUWithPretrainedEmbedModel",
        "tokenizer_type": "bert",
        "pr_auc": 0.69,
        "macro_f1": 0.67,
        "thresholds": {
            "toxic": 0.25, "severe_toxic": 0.10, "obscene": 0.22,
            "threat": 0.08, "insult": 0.21, "identity_hate": 0.09,
        },
    },
    {
        "model_name": "StackedBiGRUWithScaledAttention",
        "tokenizer_type": "bert",
        "pr_auc": 0.69,
        "macro_f1": 0.7,
        "thresholds": {
            "toxic": 0.25, "severe_toxic": 0.10, "obscene": 0.22,
            "threat": 0.08, "insult": 0.21, "identity_hate": 0.09,
        },
    },
]

DEMO_PREDICTION = {
    "original_text": "",
    "preprocessed_text": "",
    "probabilities": {
        "toxic": 0.87, "severe_toxic": 0.10, "obscene": 0.23,
        "threat": 0.04, "insult": 0.79, "identity_hate": 0.03,
    },
    "predictions": {
        "toxic": True, "severe_toxic": False, "obscene": False,
        "threat": False, "insult": True, "identity_hate": False,
    },
    "thresholds_used": {
        "toxic": 0.25, "severe_toxic": 0.10, "obscene": 0.22,
        "threat": 0.08, "insult": 0.21, "identity_hate": 0.09,
    },
    "model_used": "StackedBiGRUWithScaledAttention",
    "is_toxic": True,
}


def get_available_models(backend_url: str) -> Optional[list]:
    """Fetch model list from /models. Returns None on failure."""
    try:
        resp = requests.get(f"{backend_url}/models", timeout=3)
        if resp.status_code == 200:
            return resp.json().get("models", [])
    except Exception:
        pass
    return None


def predict(backend_url: str, text: str, model_name: str) -> Optional[dict]:
    """
    Call POST /predict.
    Returns the response dict or None on error.
    Falls back to a demo prediction if backend is unreachable.
    """
    try:
        resp = requests.post(
            f"{backend_url}/predict",
            json={"text": text, "model_name": model_name},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Backend error {resp.status_code}: {resp.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        # Return demo data so users can still explore the UI
        demo = dict(DEMO_PREDICTION)
        demo["original_text"] = text
        demo["preprocessed_text"] = text.lower()
        demo["model_used"] = model_name
        return demo
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def health_check(backend_url: str) -> bool:
    """Quick connectivity check."""
    try:
        resp = requests.get(f"{backend_url}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False