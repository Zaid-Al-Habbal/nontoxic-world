"""
components/input.py
Text input area + model selector.
Uses st.pills (Streamlit 1.40+) for example selection.

Fix: Streamlit raises a warning when a widget has both key= and value= set,
AND the session state key is also written to directly. The rule is:
  - If you give st.text_area a key, control its value ONLY through
    st.session_state[key] — never also pass value= at the same time.
"""

import streamlit as st
from services.api_client import DEMO_MODELS
from utils.helpers import (
    fmt_model_name,
    MODEL_DESCRIPTIONS,
    MODEL_TOKENIZER_LABEL,
    MODEL_SPEED_LABEL,
)

MAX_CHARS = 10_000

# Label → example text
EXAMPLES = {
    "Safe ✅":         "Thank you for your contribution to this article!",
    "Insult ⚠️":       "You are an absolute idiot and I hope you disappear.",
    "Criticism 🔍":    "This edit is completely wrong and misleading.",
    "Disagreement 💬": "I strongly disagree with your point of view here.",
    "Threat 🚨":       "Get out of here you worthless piece of garbage.",
}

# Icon per model for the info card
MODEL_ICONS = {
    "StackedBiGRUModel":                   "⚡",
    "StackedBiGRUWithPretrainedEmbedModel": "⚖️",
    "StackedBiGRUWithScaledAttention":     "🎯",
}


def _model_info_card(model: dict) -> None:
    """Render a rich info card for the currently selected model."""
    api_name = model["model_name"]
    icon     = MODEL_ICONS.get(api_name, "🤖")
    tok      = MODEL_TOKENIZER_LABEL.get(api_name, "")
    speed    = MODEL_SPEED_LABEL.get(api_name, "")
    desc     = MODEL_DESCRIPTIONS.get(api_name, "")
    pr_auc   = model["pr_auc"]
    f1       = model["macro_f1"]

    st.markdown(
        f"""
        <div class="model-info-card">
            <div class="model-info-header">
                <span class="model-info-icon">{icon}</span>
                <div class="model-info-tags">
                    <span class="model-tag">{tok}</span>
                    <span class="model-tag model-tag-speed">{speed}</span>
                </div>
            </div>
            <p class="model-info-desc">{desc}</p>
            <div class="model-info-scores">
                <div class="model-score-item">
                    <span class="model-score-label">PR-AUC</span>
                    <div class="model-score-bar">
                        <div class="model-score-fill" style="width:{pr_auc*100:.1f}%"></div>
                    </div>
                    <span class="model-score-val">{pr_auc:.2f}</span>
                </div>
                <div class="model-score-item">
                    <span class="model-score-label">Macro F1</span>
                    <div class="model-score-bar">
                        <div class="model-score-fill" style="width:{f1*100:.1f}%"></div>
                    </div>
                    <span class="model-score-val">{f1:.2f}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_section():
    """Renders text input + model selector. Returns (text, model_name, submitted)."""
    models_data  = st.session_state.get("models_data") or DEMO_MODELS
    model_names  = [m["model_name"] for m in models_data]
    selected_idx = st.session_state.get("selected_model_idx", 2)
    selected_idx = min(selected_idx, len(models_data) - 1)
    st.session_state["selected_model_idx"] = selected_idx

    # ── Initialise the textarea's session-state key on first load only ────────
    # We do this ONCE before the widget is rendered. After that, Streamlit owns
    # the value through the key — we never pass value= to st.text_area.
    if "main_text_area" not in st.session_state:
        st.session_state["main_text_area"] = ""

    col_input, col_config = st.columns([3, 1], gap="large")

    # ── Left: text input ──────────────────────────────────────────────────────
    with col_input:
        st.markdown('<p class="section-label">Comment to Analyze</p>', unsafe_allow_html=True)

        # Quick example pills (Streamlit 1.40+)
        st.markdown('<p class="example-label">Quick examples</p>', unsafe_allow_html=True)
        pill = st.pills(
            label="Examples",
            options=list(EXAMPLES.keys()),
            selection_mode="single",
            label_visibility="collapsed",
            key="example_pills",
        )

        # When a pill is selected, write the example text into the textarea's
        # session-state key BEFORE st.text_area renders on this same run.
        # Because we only set st.session_state[key] and never pass value=,
        # there is no conflict and Streamlit raises no warning.
        if pill and EXAMPLES[pill] != st.session_state["main_text_area"]:
            st.session_state["main_text_area"] = EXAMPLES[pill]

        # No value= argument — Streamlit reads st.session_state["main_text_area"]
        text = st.text_area(
            label="Comment",
            height=170,
            max_chars=MAX_CHARS,
            placeholder="Paste or type a comment here...",
            label_visibility="collapsed",
            key="main_text_area",
        )

        char_count = len(text)
        warn_cls   = "warn" if char_count > MAX_CHARS * 0.9 else ""
        st.markdown(
            f'<p class="char-counter {warn_cls}">{char_count:,} / {MAX_CHARS:,}</p>',
            unsafe_allow_html=True,
        )

        submitted = st.button(
            "Analyze Comment",
            type="primary",
            use_container_width=True,
            disabled=not text.strip(),
            key="analyze_btn",
        )
        if submitted and not text.strip():
            st.warning("Please enter some text before analyzing.")
            submitted = False

    # ── Right: model selector ─────────────────────────────────────────────────
    with col_config:
        st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)

        display_names    = [fmt_model_name(m["model_name"]) for m in models_data]
        selected_display = st.selectbox(
            label="Select model",
            options=display_names,
            index=selected_idx,
            label_visibility="collapsed",
            key="model_selectbox",
        )
        new_idx = display_names.index(selected_display)
        if new_idx != selected_idx:
            st.session_state["selected_model_idx"] = new_idx
            selected_idx = new_idx

        # Rich info card for the currently selected model
        _model_info_card(models_data[selected_idx])

    return text, model_names[selected_idx], submitted