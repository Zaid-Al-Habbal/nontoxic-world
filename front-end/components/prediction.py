"""
components/prediction.py
Renders the prediction result after a successful API call.
"""
import streamlit as st
from services.api_client import predict
from utils.helpers import add_to_history, fmt_label, fmt_model_name, pct, toxic_count

LABEL_ORDER = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _label_bars(probabilities: dict, predictions: dict, thresholds: dict):
    """Render one row per toxicity label."""
    for label in LABEL_ORDER:
        prob       = probabilities.get(label, 0.0)
        is_toxic   = predictions.get(label, False)
        threshold  = thresholds.get(label, 0.5)
        bar_cls    = "toxic" if is_toxic else "safe"
        badge_text = "TOXIC" if is_toxic else "SAFE"
        fill_pct   = f"{prob * 100:.1f}%"

        st.markdown(
            f"""
            <div class="label-row fade-in">
                <span class="label-name">{fmt_label(label)}</span>
                <div class="label-bar-track">
                    <div class="label-bar-fill {bar_cls}" style="width:{fill_pct};"></div>
                </div>
                <span class="label-pct">{pct(prob)}</span>
                <span class="label-badge {bar_cls}">{badge_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="font-family:var(--font-mono);font-size:0.63rem;color:var(--text-muted);margin-top:0.5rem;">'
        'Bar width = raw sigmoid probability · Badge = probability ≥ per-label threshold</p>',
        unsafe_allow_html=True,
    )


def render_prediction_results(text: str, model_name: str, backend_url: str):
    """Main entry point — runs prediction and renders results."""

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Analysis Results</p>', unsafe_allow_html=True)

    with st.spinner("Analyzing…"):
        result = predict(backend_url, text, model_name)

    if result is None:
        st.error("Prediction failed. Please check the backend connection.")
        return

    # Save to history
    add_to_history(text, model_name, result)
    st.session_state["last_result"] = result

    # ── Banner ────────────────────────────────────────────────────────────────
    is_toxic    = result.get("is_toxic", False)
    predictions = result.get("predictions", {})
    n_toxic     = toxic_count(predictions)

    if is_toxic:
        banner_cls = "toxic"
        icon       = "⚠️"
        title      = "Toxic Content Detected"
        subtitle   = f"{n_toxic} of 6 label{'s' if n_toxic != 1 else ''} flagged · Model: {fmt_model_name(model_name)}"
    else:
        banner_cls = "safe"
        icon       = "✅"
        title      = "This Text Appears Safe"
        subtitle   = f"No toxicity detected across 6 labels · Model: {fmt_model_name(model_name)}"

    st.markdown(
        f"""
        <div class="result-banner {banner_cls} fade-in">
            <span class="result-icon">{icon}</span>
            <div>
                <p class="result-title">{title}</p>
                <p class="result-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two columns: bars + details ───────────────────────────────────────────
    col_bars, col_detail = st.columns([3, 2], gap="large")

    with col_bars:
        st.markdown('<p class="section-label">Label Probabilities</p>', unsafe_allow_html=True)
        _label_bars(
            result.get("probabilities", {}),
            predictions,
            result.get("thresholds_used", {}),
        )

    with col_detail:
        st.markdown('<p class="section-label">Details</p>', unsafe_allow_html=True)

        # Summary metrics
        m1, m2 = st.columns(2)
        probs  = result.get("probabilities", {})
        max_prob_label = max(probs, key=probs.get) if probs else "—"
        max_prob_val   = probs.get(max_prob_label, 0.0)

        m1.metric("Flagged Labels", f"{n_toxic} / 6")
        m2.metric("Peak Probability", pct(max_prob_val))

        st.markdown("<br>", unsafe_allow_html=True)

        # Original & preprocessed text
        tabs = st.tabs(["Original", "Preprocessed"])
        with tabs[0]:
            st.markdown(
                f"""
                <div class="text-preview">
                    {result.get('original_text', text)[:500]}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with tabs[1]:
            st.markdown(
                f"""
                <div class="text-preview">
                    {result.get('preprocessed_text', text.lower())[:500]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Thresholds used
        with st.expander("🔧 Thresholds used"):
            th = result.get("thresholds_used", {})
            for label in LABEL_ORDER:
                v = th.get(label, "—")
                st.markdown(
                    f'<span style="font-family:var(--font-mono);font-size:0.75rem;">'
                    f'<span style="color:var(--text-muted);">{fmt_label(label)}</span>'
                    f'<span style="color:var(--text-primary);float:right;">{v}</span></span><br>',
                    unsafe_allow_html=True,
                )