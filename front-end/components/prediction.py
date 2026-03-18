"""
components/prediction.py
Renders the prediction result after a successful API call.
"""
import streamlit as st
from services.api_client import predict
from utils.helpers import add_to_history, fmt_label, fmt_model_name, pct, toxic_count

LABEL_ORDER = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _label_bars(probabilities: dict, predictions: dict):
    """Render one animated row per toxicity label."""
    for label in LABEL_ORDER:
        prob     = probabilities.get(label, 0.0)
        is_toxic = predictions.get(label, False)
        bar_cls  = "toxic" if is_toxic else "safe"
        badge    = "TOXIC" if is_toxic else "SAFE"
        fill_pct = f"{prob * 100:.1f}%"

        st.markdown(
            f"""
            <div class="label-row">
                <span class="label-name">{fmt_label(label)}</span>
                <div class="bar-track">
                    <div class="bar-fill {bar_cls}" style="width:{fill_pct};"></div>
                </div>
                <span class="bar-pct">{pct(prob)}</span>
                <span class="bar-badge {bar_cls}">{badge}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-600);margin-top:0.6rem;">'
        'Width = sigmoid probability · Badge = prob ≥ per-label threshold</p>',
        unsafe_allow_html=True,
    )


def render_prediction_results(text: str, model_name: str, backend_url: str):
    """Run prediction and render results."""
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Analysis Results</p>', unsafe_allow_html=True)

    with st.spinner("Analyzing…"):
        result = predict(backend_url, text, model_name)

    if result is None:
        st.error("Prediction failed. Please check the backend connection.")
        return

    add_to_history(text, model_name, result)
    st.session_state["last_result"] = result

    is_toxic    = result.get("is_toxic", False)
    predictions = result.get("predictions", {})
    n_toxic     = toxic_count(predictions)

    if is_toxic:
        banner_cls = "toxic"
        icon       = "⚠️"
        title      = "TOXIC CONTENT DETECTED"
        sub        = f"{n_toxic} of 6 labels flagged · {fmt_model_name(model_name)}"
    else:
        banner_cls = "safe"
        icon       = "✅"
        title      = "TEXT APPEARS SAFE"
        sub        = f"No toxicity detected · {fmt_model_name(model_name)}"

    # ── Result banner ─────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="result-banner {banner_cls}">
            <div class="result-icon-wrap {banner_cls}">{icon}</div>
            <div>
                <p class="result-title">{title}</p>
                <p class="result-sub">{sub}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two columns ───────────────────────────────────────────────────────────
    col_bars, col_detail = st.columns([3, 2], gap="large")

    with col_bars:
        st.markdown('<p class="section-label">Label Probabilities</p>', unsafe_allow_html=True)
        _label_bars(result.get("probabilities", {}), predictions)

    with col_detail:
        st.markdown('<p class="section-label">Details</p>', unsafe_allow_html=True)

        probs           = result.get("probabilities", {})
        max_label       = max(probs, key=probs.get) if probs else "—"
        max_val         = probs.get(max_label, 0.0)

        m1, m2 = st.columns(2)
        m1.metric("Flagged", f"{n_toxic} / 6")
        m2.metric("Peak", pct(max_val))

        st.markdown("<br>", unsafe_allow_html=True)

        tabs = st.tabs(["Original", "Preprocessed"])
        with tabs[0]:
            st.markdown(
                f'<div class="text-preview">{result.get("original_text", text)[:500]}</div>',
                unsafe_allow_html=True,
            )
        with tabs[1]:
            st.markdown(
                f'<div class="text-preview">{result.get("preprocessed_text", text.lower())[:500]}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("Thresholds used"):
            th = result.get("thresholds_used", {})
            for label in LABEL_ORDER:
                v = th.get(label, "—")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
                    f'border-bottom:1px solid var(--glass-border);">'
                    f'<span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-400);">'
                    f'{fmt_label(label)}</span>'
                    f'<span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-200);">'
                    f'{v}</span></div>',
                    unsafe_allow_html=True,
                )