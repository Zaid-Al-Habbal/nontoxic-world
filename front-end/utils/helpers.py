"""
utils/helpers.py
Shared utilities: CSS loading, session state, formatting helpers.
"""
import streamlit as st
from pathlib import Path
from datetime import datetime


# ── CSS ───────────────────────────────────────────────────────────────────────

def load_css():
    css_path = Path(__file__).parent.parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "backend_url": "http://localhost:8000",
        "models_data": None,
        "prediction_history": [],
        "last_result": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def add_to_history(text: str, model_name: str, result: dict):
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "date": datetime.now().strftime("%b %d"),
        "text": text[:120] + ("..." if len(text) > 120 else ""),
        "model": model_name,
        "is_toxic": result.get("is_toxic", False),
        "probabilities": result.get("probabilities", {}),
        "predictions": result.get("predictions", {}),
        "result": result,
    }
    st.session_state["prediction_history"].insert(0, entry)
    st.session_state["prediction_history"] = st.session_state["prediction_history"][:50]


# ── Formatting ────────────────────────────────────────────────────────────────

LABEL_DISPLAY = {
    "toxic":         "Toxic",
    "severe_toxic":  "Severe Toxic",
    "obscene":       "Obscene",
    "threat":        "Threat",
    "insult":        "Insult",
    "identity_hate": "Identity Hate",
}

MODEL_DISPLAY = {
    "StackedBiGRUModel":                   "BiGRU + BBPE",
    "StackedBiGRUWithPretrainedEmbedModel": "BiGRU + BERT Embed",
    "StackedBiGRUWithScaledAttention":     "BiGRU + Attention",
}

MODEL_DESCRIPTIONS = {
    "StackedBiGRUModel":
        "Lightest & fastest. Uses a custom BBPE tokenizer trained on the dataset.",
    "StackedBiGRUWithPretrainedEmbedModel":
        "Balanced performance. Frozen BERT word embeddings with a stacked BiGRU encoder.",
    "StackedBiGRUWithScaledAttention":
        "Best accuracy. Adds scaled dot-product self-attention on top of BERT embeddings.",
}

MODEL_TOKENIZER_LABEL = {
    "StackedBiGRUModel":                   "BBPE tokenizer",
    "StackedBiGRUWithPretrainedEmbedModel": "BERT tokenizer",
    "StackedBiGRUWithScaledAttention":     "BERT tokenizer",
}

MODEL_SPEED_LABEL = {
    "StackedBiGRUModel":                   "Fast",
    "StackedBiGRUWithPretrainedEmbedModel": "Balanced",
    "StackedBiGRUWithScaledAttention":     "Accurate",
}


def fmt_model_name(api_name: str) -> str:
    return MODEL_DISPLAY.get(api_name, api_name)


def fmt_label(key: str) -> str:
    return LABEL_DISPLAY.get(key, key.replace("_", " ").title())


def pct(prob: float) -> str:
    return f"{prob * 100:.1f}%"


def toxic_count(predictions: dict) -> int:
    return sum(1 for v in predictions.values() if v)