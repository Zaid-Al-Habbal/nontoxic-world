"""
components/visualization.py
Model comparison mode and prediction history page.
"""

import streamlit as st
import json
from services.api_client import predict, DEMO_MODELS
from utils.helpers import fmt_label, fmt_model_name, pct, toxic_count

LABEL_ORDER = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# ── Comparison mode ───────────────────────────────────────────────────────────


def render_comparison_mode(backend_url: str):
    st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card anim-1" style="margin-bottom:1.2rem;">
            <p style="margin:0;font-size:0.85rem;color:var(--text-400);font-family:var(--font-body);
               font-weight:300;line-height:1.6;">
                Run the same comment through all three models simultaneously
                and compare their predictions side-by-side.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Comment for comparison",
        height=140,
        max_chars=10_000,
        placeholder="Type a comment to compare across all models…",
        key="compare_text",
        label_visibility="collapsed",
    )

    char_count = len(text)
    st.markdown(
        f'<p class="char-counter">{char_count:,} / 10,000</p>',
        unsafe_allow_html=True,
    )

    run = st.button(
        "Run All Models →",
        type="primary",
        disabled=not text.strip(),
    )

    if not run or not text.strip():
        return

    models_data = st.session_state.get("models_data") or DEMO_MODELS
    results = {}

    with st.spinner("Running all models…"):
        for model in models_data:
            name = model["model_name"]
            res = predict(backend_url, text, name)
            if res:
                results[name] = res

    if not results:
        st.error("All model predictions failed.")
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Results</p>', unsafe_allow_html=True)

    # Top banners
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        with col:
            is_toxic = res.get("is_toxic", False)
            n_toxic = toxic_count(res.get("predictions", {}))
            banner_cls = "toxic" if is_toxic else "safe"
            icon = "⚠️" if is_toxic else "✅"
            st.markdown(
                f"""
                <div class="result-banner {banner_cls}" style="flex-direction:column;align-items:flex-start;padding:1.2rem 1.4rem;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                        <span style="font-size:1.1rem;">{icon}</span>
                        <span class="result-title" style="font-size:1.2rem;">{fmt_model_name(name)}</span>
                    </div>
                    <span class="result-sub">{n_toxic}/6 labels flagged</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p class="section-label">Per-Label Comparison</p>', unsafe_allow_html=True
    )

    for label in LABEL_ORDER:
        st.markdown(
            f'<p style="font-family:var(--font-body);font-size:0.82rem;font-weight:600;'
            f'color:var(--text-200);margin-bottom:8px;">{fmt_label(label)}</p>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(results))
        for col, (name, res) in zip(cols, results.items()):
            prob = res.get("probabilities", {}).get(label, 0.0)
            is_toxic = res.get("predictions", {}).get(label, False)
            bar_cls = "toxic" if is_toxic else "safe"
            with col:
                st.markdown(
                    f"""
                    <div style="margin-bottom:6px;">
                        <p class="compare-model-label" style="font-family:var(--font-mono);font-size:0.6rem;
                           color:var(--text-600);margin:0 0 4px;">{fmt_model_name(name)}</p>
                        <div class="bar-track" style="margin-bottom:4px;">
                            <div class="bar-fill {bar_cls}" style="width:{prob * 100:.1f}%;"></div>
                        </div>
                        <span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-200);">{pct(prob)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div style="height:6px;border-bottom:1px solid var(--glass-border);margin-bottom:14px;"></div>',
            unsafe_allow_html=True,
        )


# ── History page ──────────────────────────────────────────────────────────────


def render_history():
    st.markdown('<p class="section-label">Session Log</p>', unsafe_allow_html=True)

    history = st.session_state.get("prediction_history", [])

    if not history:
        st.markdown(
            """
            <div class="glass-card" style="text-align:center;padding:3rem 2rem;">
                <p style="font-size:2.2rem;margin-bottom:0.6rem;">📭</p>
                <p style="color:var(--text-400);font-size:0.88rem;margin:0;font-weight:300;">
                    No predictions yet. Go to <em>Analyze Text</em> to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_info, col_btn = st.columns([4, 1])
    with col_info:
        st.markdown(
            f'<p style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-600);">'
            f"{len(history)} prediction{'s' if len(history) != 1 else ''} this session</p>",
            unsafe_allow_html=True,
        )
    with col_btn:
        if st.button("Clear", type="secondary"):
            st.session_state["prediction_history"] = []
            st.rerun()

    export_data = [
        {
            "timestamp": h["timestamp"],
            "date": h["date"],
            "text": h["text"],
            "model": h["model"],
            "is_toxic": h["is_toxic"],
            "probabilities": h["probabilities"],
            "predictions": h["predictions"],
        }
        for h in history
    ]
    st.download_button(
        "Export JSON",
        data=json.dumps(export_data, indent=2),
        file_name="nontoxic_world_history.json",
        mime="application/json",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    for entry in history:
        is_toxic = entry["is_toxic"]
        n_toxic = toxic_count(entry.get("predictions", {}))
        icon = "⚠️" if is_toxic else "✅"
        label = entry["text"][:80] + ("…" if len(entry["text"]) > 80 else "")

        with st.expander(f"{icon}  {label}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Result", "Toxic" if is_toxic else "Safe")
            c2.metric("Flagged", f"{n_toxic}/6")
            c3.metric("Model", fmt_model_name(entry["model"]))

            st.markdown(
                f'<p class="history-meta">🕐 {entry["date"]} at {entry["timestamp"]}</p>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            probs = entry.get("probabilities", {})
            preds = entry.get("predictions", {})
            for label_key in LABEL_ORDER:
                prob = probs.get(label_key, 0.0)
                is_lbl = preds.get(label_key, False)
                bar_cls = "toxic" if is_lbl else "safe"
                st.markdown(
                    f"""
                    <div class="label-row">
                        <span class="label-name">{fmt_label(label_key)}</span>
                        <div class="bar-track">
                            <div class="bar-fill {bar_cls}" style="width:{prob * 100:.1f}%;"></div>
                        </div>
                        <span class="bar-pct">{pct(prob)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
