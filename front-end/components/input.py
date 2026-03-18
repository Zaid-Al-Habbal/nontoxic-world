"""
components/input.py
Text input area + model selector.
"""
import streamlit as st
from services.api_client import DEMO_MODELS
from utils.helpers import fmt_model_name, MODEL_DESCRIPTIONS, MODEL_TOKENIZER_LABEL, MODEL_SPEED_LABEL

MAX_CHARS = 10_000
EXAMPLE_TEXTS = [
    "Thank you for your contribution!",
    "You are an absolute idiot.",
    "This edit is completely wrong.",
    "I strongly disagree with you.",
    "Get out you worthless garbage.",
]


def _score_bar(value: float) -> str:
    pct = value * 100
    return (
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.1f}%"></div>'
        f'</div>'
    )


def render_input_section():
    """Renders the text input area + model selector. Returns (text, model_name, submitted)."""
    models_data  = st.session_state.get("models_data") or DEMO_MODELS
    model_names  = [m["model_name"] for m in models_data]
    selected_idx = st.session_state.get("selected_model_idx", 2)

    col_input, col_config = st.columns([3, 1], gap="large")

    # ── Left: text area ───────────────────────────────────────────────────────
    with col_input:
        st.markdown('<p class="section-label">Comment to Analyze</p>', unsafe_allow_html=True)

        st.markdown('<p class="example-label">Quick examples</p>', unsafe_allow_html=True)
        eg_cols = st.columns(len(EXAMPLE_TEXTS))
        selected_example = None
        for i, (col, ex) in enumerate(zip(eg_cols, EXAMPLE_TEXTS)):
            with col:
                if st.button(f"#{i+1}", key=f"ex_{i}", use_container_width=True, help=ex):
                    selected_example = ex

        default_text = selected_example if selected_example else st.session_state.get("input_text_val", "")
        if selected_example:
            st.session_state["input_text_val"] = selected_example

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

        # Render all cards as one HTML block
        cards_html = '<div class="model-selector">'
        for i, model in enumerate(models_data):
            api_name    = model["model_name"]
            is_selected = i == selected_idx
            active_cls  = "msc-active" if is_selected else ""
            check       = '<span class="msc-check">&#10003;</span>' if is_selected else '<span class="msc-check msc-check-empty"></span>'
            badge       = '<span class="msc-badge">Best</span>' if i == 2 else ""
            tok         = MODEL_TOKENIZER_LABEL.get(api_name, "")
            speed       = MODEL_SPEED_LABEL.get(api_name, "")
            pr_bar      = _score_bar(model["pr_auc"])
            f1_bar      = _score_bar(model["macro_f1"])

            cards_html += f"""
<div class="msc-card {active_cls}" data-idx="{i}">
  <div class="msc-header">
    {check}
    <span class="msc-name">{fmt_model_name(api_name)}</span>
    {badge}
  </div>
  <div class="msc-tags">
    <span class="msc-tag">{tok}</span>
    <span class="msc-tag msc-tag-speed">{speed}</span>
  </div>
  <div class="msc-scores">
    <div class="msc-score-row">
      <span class="msc-score-label">PR-AUC</span>
      {pr_bar}
      <span class="msc-score-val">{model["pr_auc"]:.2f}</span>
    </div>
    <div class="msc-score-row">
      <span class="msc-score-label">F1</span>
      {f1_bar}
      <span class="msc-score-val">{model["macro_f1"]:.2f}</span>
    </div>
  </div>
</div>"""
        cards_html += '\n</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # Hidden trigger buttons — one per model.
        # We hide them via CSS (class "msc-trigger-btn") rather than label_visibility.
        st.markdown('<div class="msc-trigger-wrap">', unsafe_allow_html=True)
        for i in range(len(models_data)):
            if st.button(f"sel{i}", key=f"sel_model_{i}"):
                st.session_state["selected_model_idx"] = i
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # JS: map card clicks → matching hidden Streamlit button
        # We match on the button text "sel0", "sel1", "sel2"
        st.markdown(
            """
            <script>
            (function() {
                function attach() {
                    document.querySelectorAll('.msc-card').forEach(function(card) {
                        card.addEventListener('click', function() {
                            var idx = card.getAttribute('data-idx');
                            var target = 'sel' + idx;
                            var allBtns = window.parent.document.querySelectorAll('button');
                            allBtns.forEach(function(btn) {
                                if (btn.innerText.trim() === target) {
                                    btn.click();
                                }
                            });
                        });
                    });
                }
                var tries = 0;
                var iv = setInterval(function() {
                    if (document.querySelectorAll('.msc-card').length > 0 || tries++ > 30) {
                        clearInterval(iv);
                        attach();
                    }
                }, 100);
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )

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