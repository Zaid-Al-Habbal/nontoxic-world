"""
components/input.py
Text input area + model selector.
"""
import streamlit as st
from services.api_client import DEMO_MODELS
from utils.helpers import fmt_model_name, MODEL_DESCRIPTIONS, MODEL_DISPLAY

MAX_CHARS = 10_000
EXAMPLE_TEXTS = [
    "Thank you for your contribution to this article!",
    "You are an absolute idiot and I hope you disappear.",
    "This edit is completely wrong and whoever made it is incompetent.",
    "I strongly disagree with your point of view on this topic.",
    "Get out of here you worthless piece of garbage.",
]


def render_input_section():
    """
    Renders the text input area + model selector.
    Returns (text, selected_model_name, submitted).
    """
    models_data = st.session_state.get("models_data") or DEMO_MODELS
    model_names  = [m["model_name"] for m in models_data]

    col_input, col_config = st.columns([3, 1], gap="large")

    # ── Left: text area ───────────────────────────────────────────────────────
    with col_input:
        st.markdown('<p class="section-label">Comment to Analyze</p>', unsafe_allow_html=True)

        # Quick example buttons
        st.markdown("**Try an example:**")
        eg_cols = st.columns(len(EXAMPLE_TEXTS))
        selected_example = None
        for i, (col, ex) in enumerate(zip(eg_cols, EXAMPLE_TEXTS)):
            with col:
                label = f"#{i+1}"
                if st.button(label, key=f"ex_{i}", use_container_width=True):
                    selected_example = ex

        # Text area — pre-fill with example if clicked
        default_text = selected_example or st.session_state.get("input_text_val", "")
        if selected_example:
            st.session_state["input_text_val"] = selected_example

        text = st.text_area(
            "Enter text",
            value=default_text,
            height=160,
            max_chars=MAX_CHARS,
            placeholder="Paste or type a comment here…",
            label_visibility="collapsed",
            key="main_text_area",
        )
        st.session_state["input_text_val"] = text

        # Character counter
        char_count = len(text)
        warn_class  = "warn" if char_count > MAX_CHARS * 0.9 else ""
        st.markdown(
            f'<p class="char-counter {warn_class}">{char_count:,} / {MAX_CHARS:,} characters</p>',
            unsafe_allow_html=True,
        )

        # Submit button
        submitted = st.button(
            "🔍  Analyze Comment",
            type="primary",
            use_container_width=True,
            disabled=not text.strip(),
        )

        if submitted and not text.strip():
            st.warning("Please enter some text before analyzing.")
            submitted = False

    # ── Right: model selector ─────────────────────────────────────────────────
    with col_config:
        st.markdown('<p class="section-label">Select Model</p>', unsafe_allow_html=True)

        selected_idx = st.session_state.get("selected_model_idx", 2)  # default: attention model

        for i, model in enumerate(models_data):
            api_name    = model["model_name"]
            is_selected = i == selected_idx
            badge_html  = '<span class="model-badge">DEFAULT</span>' if i == 2 else ""
            active_cls  = "active" if is_selected else ""

            st.markdown(
                f"""
                <div class="model-card {active_cls}" id="mc_{i}">
                    {badge_html}
                    <div class="model-name">{fmt_model_name(api_name)}</div>
                    <div class="model-meta">PR-AUC {model['pr_auc']:.2f} · F1 {model['macro_f1']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Select" if not is_selected else "✓ Selected",
                key=f"sel_model_{i}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["selected_model_idx"] = i
                st.rerun()

        # Description of selected model
        selected_api_name = model_names[selected_idx]
        st.markdown(
            f"""
            <div class="card fade-in" style="margin-top:0.6rem; padding: 0.9rem 1rem;">
                <p class="text-preview-label">ABOUT THIS MODEL</p>
                <p style="font-size:0.78rem; color: var(--text-secondary); line-height:1.55; margin:0;">
                    {MODEL_DESCRIPTIONS.get(selected_api_name, "")}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return text, model_names[selected_idx], submitted