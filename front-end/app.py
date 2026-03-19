import streamlit as st
from components.input import render_input_section
from components.prediction import render_prediction_results
from components.visualization import render_comparison_mode, render_history
from services.api_client import get_available_models
from utils.helpers import load_css, init_session_state

# ── Backend URL (hardcoded — no user-facing config) ───────────────────────────
BACKEND_URL = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nontoxic World",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load styles & session ─────────────────────────────────────────────────────
load_css()
init_session_state()

# ── Silently probe backend once per session ───────────────────────────────────
if "models_data" not in st.session_state or st.session_state["models_data"] is None:
    models_data = get_available_models(BACKEND_URL)
    st.session_state["models_data"] = models_data  # None → demo mode
st.session_state["backend_url"] = BACKEND_URL

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Animated logo mark
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-orb"></div>
            <div class="brand-text">
                <span class="brand-title">Nontoxic</span>
                <span class="brand-sub">World</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)

    # Nav
    st.markdown('<p class="nav-eyebrow">Navigate</p>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        options=["Analyze Text", "Compare Models", "History"],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)

    # Connection pill (read-only, no config exposed)
    is_connected = st.session_state.get("models_data") is not None
    dot_cls   = "dot-live" if is_connected else "dot-demo"
    label_txt = "Live · Backend connected" if is_connected else "Demo mode · No backend"
    st.markdown(
        f"""
        <div class="conn-pill">
            <span class="conn-dot {dot_cls}"></span>
            <span class="conn-label">{label_txt}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-footer">
            <p class="footer-line">6-label toxicity classifier</p>
            <p class="footer-line">PyTorch · FastAPI · Streamlit</p>
            <p class="footer-line footer-dim">v1.0.0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Ambient background canvas ─────────────────────────────────────────────────
st.markdown(
    """
    <div class="ambient-bg" aria-hidden="true">
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Page header ───────────────────────────────────────────────────────────────
page_meta = {
    "Analyze Text":   ("Analyze", "Detect toxicity across six dimensions in any comment."),
    "Compare Models": ("Compare", "Run all three models side-by-side on the same input."),
    "History":        ("History", "Browse and export your session prediction log."),
}
h_word, h_sub = page_meta.get(page, ("Nontoxic World", ""))

st.markdown(
    f"""
    <header class="page-header">
        <div class="header-eyebrow">Nontoxic World</div>
        <h1 class="page-title">{h_word}</h1>
        <p class="page-subtitle">{h_sub}</p>
    </header>
    """,
    unsafe_allow_html=True,
)

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "Analyze Text":
    text, selected_model, submitted = render_input_section()
    if submitted and text:
        render_prediction_results(text, selected_model, BACKEND_URL)

elif page == "Compare Models":
    render_comparison_mode(BACKEND_URL)

elif page == "History":
    render_history()