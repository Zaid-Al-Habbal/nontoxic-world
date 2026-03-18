import streamlit as st
from components.input import render_input_section
from components.prediction import render_prediction_results
from components.visualization import render_comparison_mode, render_history
from services.api_client import get_available_models
from utils.helpers import load_css, init_session_state

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-logo">
            <span class="logo-icon">🌿</span>
            <span class="logo-text">Nontoxic World</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Navigation
    st.markdown('<p class="sidebar-label">NAVIGATION</p>', unsafe_allow_html=True)
    page = st.radio(
        "",
        options=["🔍 Analyze Text", "⚖️ Compare Models", "📜 History"],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Backend URL
    st.markdown('<p class="sidebar-label">CONFIGURATION</p>', unsafe_allow_html=True)
    backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.get("backend_url", "http://localhost:8000"),
        key="backend_url_input",
        help="FastAPI backend address",
    )
    st.session_state["backend_url"] = backend_url

    # Connection status
    models_data = get_available_models(backend_url)
    if models_data:
        st.markdown(
            '<div class="status-badge status-online">● Connected</div>',
            unsafe_allow_html=True,
        )
        st.session_state["models_data"] = models_data
    else:
        st.markdown(
            '<div class="status-badge status-offline">● Offline — using demo mode</div>',
            unsafe_allow_html=True,
        )
        if "models_data" not in st.session_state:
            st.session_state["models_data"] = None

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-footer">
            <p>Multi-label toxicity classifier</p>
            <p>Built with PyTorch · FastAPI · Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <h1 class="page-title">Nontoxic<span class="title-accent"> World</span></h1>
        <p class="page-subtitle">Multi-label toxic comment classification · 6 categories · Real-time analysis</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "🔍 Analyze Text":
    text, selected_model, submitted = render_input_section()
    if submitted and text:
        render_prediction_results(text, selected_model, backend_url)

elif page == "⚖️ Compare Models":
    render_comparison_mode(backend_url)

elif page == "📜 History":
    render_history()