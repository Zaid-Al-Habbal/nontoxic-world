"""
components/input.py
Text input area + model selector.
"""

import streamlit as st
from services.api_client import DEMO_MODELS
from utils.helpers import fmt_model_name, MODEL_DESCRIPTIONS

MAX_CHARS = 10_000
EXAMPLE_TEXTS = [
    "Thank you for your contribution!",
    "You are an absolute idiot.",
    "This edit is completely wrong.",
    "I strongly disagree with you.",
    "Get out you worthless garbage.",
]


def render_input_section():
    """Renders the text input area + model selector. Returns (text, model_name, submitted)."""
    models_data = st.session_state.get("models_data") or DEMO_MODELS
    model_names = [m["model_name"] for m in models_data]
    selected_idx = st.session_state.get("selected_model_idx", 2)
    if selected_idx >= len(models_data):
        selected_idx = len(models_data) - 1
    st.session_state["selected_model_idx"] = selected_idx

    col_input, col_config = st.columns([3, 1], gap="large")

    # ── Left: text area ───────────────────────────────────────────────────────
    with col_input:
        st.markdown(
            '<p class="section-label">Comment to Analyze</p>', unsafe_allow_html=True
        )

        st.markdown(
            '<p class="example-label">Quick examples</p>', unsafe_allow_html=True
        )
        eg_cols = st.columns(len(EXAMPLE_TEXTS))
        for i, (col, ex) in enumerate(zip(eg_cols, EXAMPLE_TEXTS)):
            with col:
                if st.button(
                    f"#{i + 1}", key=f"ex_{i}", use_container_width=True, help=ex
                ):
                    st.session_state["main_text_area"] = ex
                    st.session_state["input_text_val"] = ex

        default_text = st.session_state.get("input_text_val", "")

        text = st.text_area(
            "Enter text",
            value=default_text,
            height=170,
            max_chars=MAX_CHARS,
            placeholder="Paste or type a comment here...",
            label_visibility="collapsed",
            key="main_text_area",
        )
        st.session_state["input_text_val"] = text

        char_count = len(text)
        warn_cls = "warn" if char_count > MAX_CHARS * 0.9 else ""
        st.markdown(
            f'<p class="char-counter {warn_cls}">{char_count:,} / {MAX_CHARS:,}</p>',
            unsafe_allow_html=True,
        )

        submitted = st.button(
            "Analyze Comment",
            type="primary",
            use_container_width=True,
            disabled=not text.strip(),
        )
        if submitted and not text.strip():
            st.warning("Please enter some text before analyzing.")
            submitted = False

    # ── Right: model selector ─────────────────────────────────────────────────
    with col_config:
        st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)

        model_options = [fmt_model_name(m["model_name"]) for m in models_data]
        selected_option = st.selectbox(
            "Select Model",
            options=model_options,
            index=selected_idx,
            label_visibility="collapsed",
            key="model_select",
        )
        selected_idx = model_options.index(selected_option)
        st.session_state["selected_model_idx"] = selected_idx

        # Description of selected model
        desc = MODEL_DESCRIPTIONS.get(model_names[selected_idx], "")
        st.markdown(
            f"""
            <div class="msc-desc-panel">
                <p class="msc-desc-eyebrow">About this model</p>
                <p class="msc-desc-text">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return text, model_names[selected_idx], submitted
