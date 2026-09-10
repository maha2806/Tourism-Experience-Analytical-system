"""
Tourism Experience Analytics — Streamlit Application
Classification, Prediction, and Recommendation System for Tourism Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import logging


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="Tourism Analytics",
    page_icon="assets/logo.svg" if False else "✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

_resolved_theme = st.session_state["theme"]

THEMES = {
    "dark": {
        "bg-0": "#0f0e0c",
        "bg-1": "#1a1814",
        "bg-2": "#201e19",
        "line": "#2e2b25",
        "text-0": "#f5f1ea",
        "text-1": "#c6beaf",
        "text-2": "#857d6e",
        "accent": "#0097b8",
        "accent-dim": "#061e24",
        "accent-ink": "#ffffff",
        "warn": "#d4943a",
        "danger": "#d96b5f",
        "grid": "#2e2b25",
        "shadow": "rgba(0, 151, 184, 0.25)",
        "shadow-strong": "rgba(0, 151, 184, 0.42)",
        "glow-r1": "radial-gradient(circle at 15% 0%, #091820 0%, transparent 38%)",
        "glow-r2": "radial-gradient(circle at 95% 18%, #0a1a1f 0%, transparent 40%)",
        "color-scheme": "dark",
    },
    "light": {
        "bg-0": "#f5f0e6",
        "bg-1": "#ede7d8",
        "bg-2": "#ffffff",
        "line": "#e0d8c8",
        "text-0": "#16140f",
        "text-1": "#4e4738",
        "text-2": "#8a8070",
        "accent": "#006b8f",
        "accent-dim": "#dff0f5",
        "accent-ink": "#ffffff",
        "warn": "#8a5e00",
        "danger": "#a83228",
        "grid": "#e8e0cf",
        "shadow": "rgba(0, 107, 143, 0.15)",
        "shadow-strong": "rgba(0, 107, 143, 0.28)",
        "glow-r1": "radial-gradient(circle at 15% 0%, #ddf0f7 0%, transparent 40%)",
        "glow-r2": "radial-gradient(circle at 95% 18%, #d8f2f8 0%, transparent 44%)",
        "color-scheme": "light",
    },
}

_T = THEMES[_resolved_theme]

_css_vars = "\n".join(
    f"        --{k}: {v};" for k, v in _T.items() if k != "color-scheme"
)

st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

    :root {{
{_css_vars}
        color-scheme: {_T["color-scheme"]};
    }}

    :root {{
        --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-serif:   'Source Serif 4', Georgia, serif;
        --font-mono:    'JetBrains Mono', monospace;
        --radius-xs: 4px;
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 10px;
        --radius-xl: 50px;
        --motion-fast:   200ms;
        --motion-normal: 250ms;
    }}

    html, body, [class*="css"] {{ font-family: var(--font-primary); }}

    .stApp {{
        background: var(--glow-r1), var(--glow-r2), var(--bg-0) !important;
        color: var(--text-0) !important;
    }}

    [data-testid="stMain"], [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"], .main {{
        background-color: var(--bg-0) !important;
        color: var(--text-0) !important;
    }}

    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}

    header[data-testid="stHeader"] {{
        background: transparent;
        box-shadow: none;
        height: 3.2rem !important;
        min-height: 3.2rem !important;
        z-index: 999999 !important;
    }}

    div[data-testid="stToolbar"] {{ visibility: hidden !important; }}

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        display: flex !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 12px !important;
        left: 12px !important;
        z-index: 999999 !important;
    }}

    button[data-testid="stSidebarCollapsedControl"],
    button[data-testid="baseButton-headerNoPadding"],
    button[data-testid="stBaseButton-headerNoPadding"] {{
        visibility: visible !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        color: var(--accent) !important;
        background: var(--bg-2) !important;
        border: 1px solid var(--accent) !important;
        border-radius: var(--radius-lg) !important;
        width: 42px !important;
        height: 42px !important;
        box-shadow: 0 0 14px var(--shadow) !important;
        transition: box-shadow var(--motion-fast) ease, transform var(--motion-fast) ease;
    }}

    button[data-testid="stSidebarCollapsedControl"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover,
    button[data-testid="stBaseButton-headerNoPadding"]:hover {{
        box-shadow: 0 0 22px var(--shadow-strong) !important;
        transform: translateY(-1px);
    }}

    button[data-testid="stSidebarCollapsedControl"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg,
    button[data-testid="stBaseButton-headerNoPadding"] svg {{
        fill: var(--accent) !important;
        color: var(--accent) !important;
        width: 22px !important;
        height: 22px !important;
    }}

    section[data-testid="stSidebar"] {{
        background: var(--bg-1) !important;
        border-right: 1px solid var(--line);
    }}
    section[data-testid="stSidebar"] * {{ color: var(--text-0); }}

    h1, h2, h3 {{
        font-family: var(--font-serif);
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--text-0) !important;
    }}

    .eyebrow {{
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 0.4rem;
        display: block;
    }}

    .subtext {{
        color: var(--text-1);
        font-size: 0.98rem;
        line-height: 1.6;
    }}

    hr {{ border-color: var(--line) !important; }}

    .stat-card {{
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 1.1rem 1.3rem;
        height: 100%;
        transition: border-color var(--motion-fast) ease;
    }}
    .stat-card:hover {{ border-color: var(--accent); }}
    .stat-card .label {{
        font-family: var(--font-mono);
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-2);
        margin-bottom: 0.35rem;
        display: block;
    }}
    .stat-card .value {{
        font-family: var(--font-mono);
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text-0);
        line-height: 1.1;
    }}
    .stat-card .delta {{
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--accent);
        margin-top: 0.3rem;
        display: block;
    }}

    .result-banner {{
        background: linear-gradient(135deg, var(--accent-dim), var(--bg-2));
        border: 1px solid var(--accent);
        border-radius: var(--radius-lg);
        padding: 1.8rem 2rem;
        margin-top: 1rem;
    }}
    .result-banner .tag {{
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
    }}
    .result-banner .title {{
        font-family: var(--font-serif);
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-0);
        margin: 0.25rem 0 0.5rem 0;
    }}
    .result-banner .desc {{
        color: var(--text-1);
        font-size: 0.98rem;
        line-height: 1.55;
    }}

    .rec-card {{
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.9rem;
        transition: border-color var(--motion-fast) ease;
    }}
    .rec-card:hover {{ border-color: var(--accent); }}
    .rec-rank {{
        font-family: var(--font-mono);
        font-size: 0.85rem;
        color: var(--accent);
        background: var(--accent-dim);
        border-radius: var(--radius-sm);
        padding: 0.25rem 0.55rem;
        min-width: 2.1rem;
        text-align: center;
    }}
    .rec-name {{
        font-size: 0.95rem;
        color: var(--text-0);
        font-weight: 500;
    }}
    .rec-score {{
        margin-left: auto;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--text-2);
    }}

    .badge {{
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        letter-spacing: 0.06em;
        padding: 0.2rem 0.55rem;
        border-radius: var(--radius-xl);
        border: 1px solid var(--line);
        color: var(--text-1);
        margin-right: 0.4rem;
    }}

    .stButton > button {{
        background: var(--accent);
        color: var(--accent-ink);
        border: none;
        border-radius: var(--radius-md);
        font-weight: 700;
        font-family: var(--font-primary);
        padding: 0.6rem 1.4rem;
        transition: transform 0.1s ease, box-shadow var(--motion-fast) ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 18px var(--shadow);
        color: var(--accent-ink);
    }}
    .stButton > button:focus-visible {{
        outline: 2px solid var(--accent) !important;
        outline-offset: 2px !important;
    }}

    .stTextInput input, .stNumberInput input {{
        background: var(--bg-2) !important;
        color: var(--text-0) !important;
        border: 1px solid var(--line) !important;
        border-radius: var(--radius-md) !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }}

    .stSelectbox > div > div {{
        background: var(--bg-2) !important;
        border: 1px solid var(--line) !important;
        color: var(--text-0) !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid var(--line);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: var(--font-mono);
        font-size: 0.82rem;
        color: var(--text-1);
    }}
    .stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}

    [data-testid="stDataFrame"] {{
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        overflow: hidden;
    }}

    .empty-state {{
        border: 1px dashed var(--line);
        border-radius: var(--radius-md);
        padding: 1.4rem 1.6rem;
        color: var(--text-2);
        font-size: 0.9rem;
        text-align: center;
    }}

    .footer-note {{
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--text-2);
        text-align: center;
        padding: 2rem 0 0.5rem 0;
    }}

    /* Radio buttons — accent colour in both themes */
    [data-baseweb="radio"] [data-checked="true"] div,
    [data-baseweb="radio"] div[aria-checked="true"] {{
        background-color: var(--accent) !important;
        border-color:     var(--accent) !important;
    }}
    [data-baseweb="radio"]:hover [data-checked="false"] > div:first-child {{
        border-color: var(--accent) !important;
    }}
    [data-baseweb="radio"] label p,
    [data-baseweb="radio"] label span {{
        color: var(--text-0) !important;
    }}

    /* Slider — accent track and thumb */
    [data-testid="stSlider"] [role="slider"] {{
        background-color: var(--accent) !important;
        border-color:     var(--accent) !important;
    }}
    [data-testid="stSlider"] div[data-testid="stSliderTrack"] > div:nth-child(2) {{
        background-color: var(--accent) !important;
    }}
    [data-testid="stSlider"] p {{ color: var(--text-0) !important; }}

    /* All widget labels */
    label, [data-testid="stWidgetLabel"] p,
    .stSelectbox label, .stNumberInput label,
    .stSlider label, .stRadio label,
    .stCheckbox label, .stTextInput label {{
        color: var(--text-1) !important;
        font-size: 0.85rem !important;
    }}

    /* Number input stepper buttons */
    [data-testid="stNumberInput"] button {{
        background: var(--bg-2)  !important;
        border-color: var(--line) !important;
        color: var(--text-0)     !important;
    }}
    [data-testid="stNumberInput"] button:hover {{
        border-color: var(--accent) !important;
        color: var(--accent)        !important;
    }}

    /* Selectbox dropdown list */
    [data-baseweb="select"] ul {{
        background: var(--bg-2) !important;
        border: 1px solid var(--line) !important;
    }}
    [data-baseweb="select"] li {{
        color: var(--text-0) !important;
    }}
    [data-baseweb="select"] li:hover {{
        background: var(--accent-dim) !important;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        background: var(--bg-2) !important;
        border: 1px solid var(--line) !important;
        border-radius: var(--radius-md) !important;
    }}
    [data-testid="stExpander"] summary {{ color: var(--text-0) !important; }}

    /* Caption / small text */
    [data-testid="stCaptionContainer"] p,
    .stCaption p {{ color: var(--text-2) !important; }}

    /* Tab bar */
    [data-baseweb="tab"] button,
    [data-baseweb="tab"] span {{ color: var(--text-1) !important; }}
    [data-baseweb="tab"][aria-selected="true"] button,
    [data-baseweb="tab"][aria-selected="true"] span {{ color: var(--accent) !important; }}

    /* Download button */
    [data-testid="stDownloadButton"] button {{
        background: transparent        !important;
        border: 1px solid var(--accent) !important;
        color: var(--accent)           !important;
    }}
    [data-testid="stDownloadButton"] button:hover {{
        background: var(--accent)  !important;
        color: var(--accent-ink)   !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] * {{ color: var(--text-0) !important; }}
    [data-testid="stDataFrame"] th {{ background: var(--bg-1) !important; }}
    [data-testid="stDataFrame"] td {{ background: var(--bg-2) !important; }}

    /* Alert / error / warning boxes */
    [data-testid="stAlert"] {{
        background: var(--bg-2) !important;
        color: var(--text-0)    !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar              {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track        {{ background: var(--bg-1); }}
    ::-webkit-scrollbar-thumb        {{ background: var(--line); border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover  {{ background: var(--accent); }}

    </style>
    """,
    unsafe_allow_html=True,
)


def icon(name, size=22, sw=1.6, color="currentColor"):
    """Return inline SVG string for a given icon name."""
    paths = {
        "compass": (
            '<circle cx="12" cy="12" r="8.5"/>'
            '<path d="M16.2 7.8l-2.4 5.8-5.8 2.4 2.4-5.8z"/>'
        ),
        "star": (
            '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 '
            '12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>'
        ),
        "map": (
            '<polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/>'
            '<line x1="8" y1="2" x2="8" y2="18"/>'
            '<line x1="16" y1="6" x2="16" y2="22"/>'
        ),
        "bar-chart": (
            '<line x1="18" y1="20" x2="18" y2="10"/>'
            '<line x1="12" y1="20" x2="12" y2="4"/>'
            '<line x1="6"  y1="20" x2="6"  y2="14"/>'
            '<line x1="2"  y1="20" x2="22" y2="20"/>'
        ),
        "users": (
            '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
            '<circle cx="9" cy="7" r="4"/>'
            '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/>'
            '<path d="M16 3.13a4 4 0 0 1 0 7.75"/>'
        ),
        "cpu": (
            '<rect x="9" y="9" width="6" height="6"/>'
            '<path d="M18 9V6a1 1 0 0 0-1-1h-3M9 5H6a1 1 0 0 0-1 1v3M5 18v1a1 1 0 0 0 1 1h3m6 0h3a1 1 0 0 0 1-1v-3M18 15v3"/>'
            '<line x1="9" y1="1" x2="9" y2="4"/>'
            '<line x1="15" y1="1" x2="15" y2="4"/>'
            '<line x1="23" y1="9" x2="20" y2="9"/>'
            '<line x1="23" y1="15" x2="20" y2="15"/>'
            '<line x1="1" y1="9" x2="4" y2="9"/>'
            '<line x1="1" y1="15" x2="4" y2="15"/>'
            '<line x1="9" y1="23" x2="9" y2="20"/>'
            '<line x1="15" y1="23" x2="15" y2="20"/>'
        ),
        "info": (
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="16" x2="12" y2="12"/>'
            '<line x1="12" y1="8" x2="12.01" y2="8"/>'
        ),
        "plane": (
            '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 '
            "19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 "
            "2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 "
            '6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>'
        ),
    }
    body = paths.get(name, "")
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:-4px; display:inline-block;">'
        f"{body}</svg>"
    )


ICON_PLANE = icon("plane", size=22)
ICON_MAP = icon("map", size=26)
ICON_STAR = icon("star", size=26)
ICON_USERS = icon("users", size=26)
ICON_CHART = icon("bar-chart", size=26)
ICON_COMPASS = icon("compass", size=26)
ICON_CPU = icon("cpu", size=26)
ICON_INFO = icon("info", size=26)

MODE_INFO = {
    "Business": {
        "desc": "Corporate travellers visiting for work. Prioritise business facilities, reliable Wi-Fi, and express services.",
        "color": "#0097b8",
    },
    "Couples": {
        "desc": "Romantic visits — scenic, peaceful, and intimate experiences matter most. Highlight couple-specific packages.",
        "color": "#d4943a",
    },
    "Family": {
        "desc": "Family groups with mixed ages. Safety, accessibility, and child-friendly amenities drive satisfaction.",
        "color": "#5cb85c",
    },
    "Friends": {
        "desc": "Social travel groups — group activities, variety, and social spaces are key satisfaction drivers.",
        "color": "#9b59b6",
    },
    "Solo": {
        "desc": "Independent travellers prioritising flexibility, discovery, and authentic local experiences.",
        "color": "#e67e22",
    },
}


print("=" * 70)
print("MODEL DEPLOYMENT DIAGNOSTIC")
print("Current working directory:", Path.cwd())
print("app.py location:", Path(__file__).resolve())
print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)
print("MODEL_DIR exists:", MODEL_DIR.exists())

if MODEL_DIR.exists():
    print("Files in models/:")
    for p in MODEL_DIR.iterdir():
        print(f"  {p.name} -> {p.stat().st_size:,} bytes")

classification_path = MODEL_DIR / "best_classification_model.pkl"

print("Classification model exists:", classification_path.exists())

if classification_path.exists():
    print("Classification model size:", classification_path.stat().st_size, "bytes")

print("=" * 70)


@st.cache_resource
def load_models():
    """Load all saved model artifacts from models/ folder."""
    try:
        reg_model = joblib.load(MODEL_DIR / "best_regression_model.pkl")

        reg_scaler = joblib.load(MODEL_DIR / "regression_scaler.pkl")

        use_sc_reg = joblib.load(MODEL_DIR / "use_sc_reg.pkl")

        clf_model = joblib.load(MODEL_DIR / "best_classification_model.pkl")

        clf_scaler = joblib.load(MODEL_DIR / "classification_scaler.pkl")

        le_mode = joblib.load(MODEL_DIR / "label_encoder_mode.pkl")

        use_sc_clf = joblib.load(MODEL_DIR / "use_sc_clf.pkl")

        collab_sim = joblib.load(MODEL_DIR / "collab_similarity.pkl")

        content_sim = joblib.load(MODEL_DIR / "content_similarity.pkl")

        uim = joblib.load(MODEL_DIR / "user_item_matrix.pkl")

        meta = joblib.load(MODEL_DIR / "feature_meta.pkl")

        return (
            reg_model,
            reg_scaler,
            use_sc_reg,
            clf_model,
            clf_scaler,
            le_mode,
            use_sc_clf,
            collab_sim,
            content_sim,
            uim,
            meta,
        ), True

    except Exception:
        logging.exception("Model loading failed")
        return None, False


@st.cache_data
def load_data():
    """Load master DataFrame and item catalogue."""
    try:
        df = pd.read_csv("models/master_df.csv", low_memory=False)
        df_item = pd.read_csv("models/df_item.csv")
        return df, df_item, True
    except FileNotFoundError:
        return None, None, False


artifacts, MODELS_OK = load_models()
df, df_item, DATA_OK = load_data()

if MODELS_OK and artifacts:
    (
        reg_model,
        reg_scaler,
        use_sc_reg,
        clf_model,
        clf_scaler,
        le_mode,
        use_sc_clf,
        collab_sim,
        content_sim,
        uim,
        meta,
    ) = artifacts


def _fig_layout(**overrides):
    """Return Plotly layout kwargs for the active theme. Pass overrides to avoid duplicate-key errors."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=_T["text-1"], size=12),
        xaxis=dict(gridcolor=_T["grid"], linecolor=_T["line"]),
        yaxis=dict(gridcolor=_T["grid"], linecolor=_T["line"]),
        margin=dict(l=30, r=20, t=45, b=30),
        hoverlabel=dict(
            bgcolor=_T["bg-2"], bordercolor=_T["accent"], font=dict(color=_T["text-0"])
        ),
    )
    base.update(overrides)
    return base


COLORS = [
    "#0097b8",
    "#00c4e0",
    "#40d4e8",
    "#80e8f4",
    "#b0f0f8",
    "#d4943a",
    "#9b59b6",
    "#5cb85c",
    "#e67e22",
    "#c6beaf",
]


def stat_card(label, value, delta=None):
    """Render a stat card component."""
    d = f'<span class="delta">{delta}</span>' if delta else ""
    st.markdown(
        f'<div class="stat-card"><span class="label">{label}</span>'
        f'<div class="value">{value}</div>{d}</div>',
        unsafe_allow_html=True,
    )


# Sidebar
st.sidebar.markdown(
    f"""
    <div style="padding: 0.4rem 0 1.2rem 0;">
        <span style="font-family:var(--font-mono); font-size:0.7rem; letter-spacing:0.16em;
        color:var(--accent); text-transform:uppercase;">tourism · ML</span>
        <h2 style="margin:0.15rem 0 0 0; font-size:1.45rem;">
            {ICON_PLANE} Tourism Analytics
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_OPTIONS = [
    "Home",
    "EDA Explorer",
    "Rating Predictor",
    "Mode Predictor",
    "Recommendations",
    "Model Performance",
]

if st.session_state.get("_nav_hint"):
    st.session_state["_nav_radio"] = st.session_state.pop("_nav_hint")

page = st.sidebar.radio(
    "Navigation",
    options=NAV_OPTIONS,
    label_visibility="collapsed",
    key="_nav_radio",
)

st.sidebar.markdown("<hr style='margin:1.0rem 0;'>", unsafe_allow_html=True)

st.sidebar.markdown(
    '<span style="font-family:var(--font-mono); font-size:0.7rem; '
    'letter-spacing:0.1em; color:var(--text-2); text-transform:uppercase;">theme</span>',
    unsafe_allow_html=True,
)

theme_choice = st.sidebar.radio(
    "Theme",
    options=["Dark", "Light"],
    index=0 if st.session_state["theme"] == "dark" else 1,
    horizontal=True,
    label_visibility="collapsed",
    key="_theme_radio",
)

new_theme = "dark" if theme_choice == "Dark" else "light"
if new_theme != st.session_state["theme"]:
    st.session_state["theme"] = new_theme
    st.rerun()

st.sidebar.markdown("<hr style='margin:1.0rem 0;'>", unsafe_allow_html=True)

_model_info = ""
if MODELS_OK:
    _model_info = (
        f"REG &nbsp;&nbsp;&nbsp; {meta.get('best_reg_name','—')[:18]}<br>"
        f"CLF &nbsp;&nbsp;&nbsp; {meta.get('best_clf_name','—')[:18]}<br>"
        f"REC &nbsp;&nbsp;&nbsp; Hybrid (collab + content)"
    )
else:
    _model_info = "Models not loaded<br>Run notebook first"

st.sidebar.markdown(
    f'<div style="font-family:var(--font-mono); font-size:0.72rem; color:var(--text-2); line-height:2;">'
    f"{_model_info}</div>",
    unsafe_allow_html=True,
)

# HOME
if page == "Home":

    st.markdown(
        '<span class="eyebrow">tourism · machine learning · recommendation</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f"{ICON_PLANE} Tourism Experience Analytics</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="subtext" style="max-width:660px;">Predicts attraction ratings and visit modes from '
        "traveller demographics, and recommends personalised attractions using collaborative and "
        "content-based filtering — built on 50,000+ tourism transactions across 33,000+ global users.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    if DATA_OK and df is not None:
        with c1:
            stat_card("Transactions", f"{len(df):,}")
        with c2:
            stat_card("Unique Users", f"{df['UserId'].nunique():,}")
        with c3:
            stat_card("Attractions", f"{df['AttractionId'].nunique()}")
        with c4:
            stat_card("Avg Rating", f"{df['Rating'].mean():.2f}", delta="out of 5.0")
    else:
        with c1:
            stat_card("Transactions", "50,000+")
        with c2:
            stat_card("Users", "33,000+")
        with c3:
            stat_card("Attractions", "30")
        with c4:
            stat_card("ML Tasks", "3", delta="Regression · Class · Rec")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("### What this does")
        st.markdown(
            """
            <p class="subtext">
            <b style="color:var(--text-0);">Predict ratings</b> — enter a traveller profile
            and attraction details to get a predicted satisfaction score (1–5), powered by the
            best-performing regression model from an 11-model comparison.
            </p>
            <p class="subtext">
            <b style="color:var(--text-0);">Predict visit mode</b> — predict whether a trip
            is Business, Couples, Family, Friends, or Solo travel. Results include full
            probability breakdowns across all five modes.
            </p>
            <p class="subtext">
            <b style="color:var(--text-0);">Recommend attractions</b> — item-based collaborative
            filtering identifies attractions most visited by similar users. A hybrid system
            combines this with TF-IDF content similarity for cold-start users.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("→ Predict rating", use_container_width=True):
                st.session_state["_nav_hint"] = "Rating Predictor"
                st.rerun()
        with b2:
            if st.button("→ Predict mode", use_container_width=True):
                st.session_state["_nav_hint"] = "Mode Predictor"
                st.rerun()
        with b3:
            if st.button("→ Get recs", use_container_width=True):
                st.session_state["_nav_hint"] = "Recommendations"
                st.rerun()

    with right:
        st.markdown("### Visit mode reference")
        for name, meta_m in MODE_INFO.items():
            st.markdown(
                f'<div class="rec-card" style="border-left: 3px solid {meta_m["color"]};">'
                f'<div><div class="rec-name">{name}</div>'
                f'<div style="color:var(--text-2); font-size:0.82rem; margin-top:0.2rem;">'
                f'{meta_m["desc"]}</div></div></div>',
                unsafe_allow_html=True,
            )

# EDA EXPLORER
elif page == "EDA Explorer":

    st.markdown(
        '<span class="eyebrow">exploratory analysis</span>', unsafe_allow_html=True
    )
    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f"{ICON_CHART} EDA Explorer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtext" style="max-width:600px;">Interactive exploration of the tourism dataset '
        "with dynamic filters. All charts update in real time.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not DATA_OK or df is None:
        st.markdown(
            '<div class="empty-state">Run the notebook first to generate <code>models/master_df.csv</code>.</div>',
            unsafe_allow_html=True,
        )
    else:
        with st.expander("Filters", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            cont_opts = sorted([c for c in df["Continent"].unique() if c != "Unknown"])
            sel_cont = fc1.multiselect("Continent", cont_opts, default=cont_opts[:3])
            type_opts = sorted(
                [t for t in df["AttractionType"].unique() if t != "Unknown"]
            )
            sel_type = fc2.multiselect(
                "Attraction Type", type_opts, default=type_opts[:5]
            )
            yr_min, yr_max = int(df["VisitYear"].min()), int(df["VisitYear"].max())
            sel_yr = fc3.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))

        fdf = df[
            df["Continent"].isin(sel_cont if sel_cont else cont_opts)
            & df["AttractionType"].isin(sel_type if sel_type else type_opts)
            & df["VisitYear"].between(sel_yr[0], sel_yr[1])
        ]
        st.caption(f"**{len(fdf):,}** transactions match your filters")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Ratings", "Visit Modes", "Geographic", "Temporal"]
        )

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                rc = fdf["Rating"].value_counts().sort_index().reset_index()
                rc.columns = ["Rating", "Count"]
                fig = px.bar(
                    rc,
                    x="Rating",
                    y="Count",
                    color="Rating",
                    color_continuous_scale=[
                        "#006b8f",
                        "#0097b8",
                        "#00c4e0",
                        "#40d4e8",
                        "#80e8f4",
                    ],
                    title="Rating Distribution",
                )
                fig.update_layout(**_fig_layout(), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                tr = (
                    fdf[fdf["AttractionType"] != "Unknown"]
                    .groupby("AttractionType")["Rating"]
                    .mean()
                    .sort_values(ascending=True)
                    .reset_index()
                )
                tr.columns = ["Type", "AvgRating"]
                fig = px.bar(
                    tr,
                    x="AvgRating",
                    y="Type",
                    orientation="h",
                    color="AvgRating",
                    color_continuous_scale="Teal",
                    title="Mean Rating by Attraction Type",
                )
                fig.update_layout(**_fig_layout(), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            piv = fdf[
                (fdf["Continent"] != "Unknown") & (fdf["VisitModeLabel"] != "Unknown")
            ].pivot_table(
                values="Rating",
                index="Continent",
                columns="VisitModeLabel",
                aggfunc="mean",
            )
            fig = px.imshow(
                piv,
                color_continuous_scale="YlOrRd",
                text_auto=".2f",
                title="Mean Rating Heatmap — Continent × Visit Mode",
            )
            fig.update_layout(**_fig_layout())
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                mc = fdf[fdf["VisitModeLabel"] != "Unknown"][
                    "VisitModeLabel"
                ].value_counts()
                fig = go.Figure(
                    go.Pie(
                        labels=mc.index,
                        values=mc.values,
                        hole=0.5,
                        marker_colors=COLORS[: len(mc)],
                        textinfo="percent+label",
                    )
                )
                fig.update_layout(**_fig_layout(), title="Visit Mode Distribution")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                myr = (
                    fdf[fdf["VisitModeLabel"] != "Unknown"]
                    .groupby(["VisitYear", "VisitModeLabel"])["TransactionId"]
                    .count()
                    .reset_index()
                )
                myr.columns = ["Year", "Mode", "Visits"]
                fig = px.bar(
                    myr,
                    x="Year",
                    y="Visits",
                    color="Mode",
                    color_discrete_sequence=COLORS,
                    title="Visit Mode by Year",
                    barmode="stack",
                )
                fig.update_layout(**_fig_layout())
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                cu = (
                    fdf[fdf["Continent"] != "Unknown"]
                    .groupby("Continent")["UserId"]
                    .nunique()
                    .sort_values()
                    .reset_index()
                )
                cu.columns = ["Continent", "Users"]
                fig = px.bar(
                    cu,
                    x="Users",
                    y="Continent",
                    orientation="h",
                    color="Users",
                    color_continuous_scale="Teal",
                    title="Unique Users by Continent",
                )
                fig.update_layout(**_fig_layout(), coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                sb = (
                    fdf[
                        (fdf["Continent"] != "Unknown")
                        & (fdf["VisitModeLabel"] != "Unknown")
                    ]
                    .groupby(["Continent", "VisitModeLabel"])["TransactionId"]
                    .count()
                    .reset_index()
                )
                sb.columns = ["Continent", "Mode", "Count"]
                fig = px.sunburst(
                    sb,
                    path=["Continent", "Mode"],
                    values="Count",
                    color_discrete_sequence=COLORS,
                    title="Continent → Visit Mode Sunburst",
                )
                fig.update_layout(**_fig_layout(margin=dict(l=5, r=5, t=45, b=5)))
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            c1, c2 = st.columns(2)
            month_map = {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
            with c1:
                mo = fdf.groupby("VisitMonth")["TransactionId"].count().reset_index()
                mo.columns = ["Month", "Visits"]
                mo["MonthName"] = mo["Month"].map(month_map)
                fig = px.area(
                    mo,
                    x="MonthName",
                    y="Visits",
                    color_discrete_sequence=[COLORS[0]],
                    title="Monthly Visit Volume",
                )
                fig.update_traces(
                    line_color=COLORS[0], fillcolor=f"rgba(0,151,184,0.12)"
                )
                fig.update_layout(**_fig_layout())
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                yr_r = fdf.groupby("VisitYear")["Rating"].mean().reset_index()
                yr_r.columns = ["Year", "AvgRating"]
                fig = go.Figure(
                    go.Scatter(
                        x=yr_r["Year"],
                        y=yr_r["AvgRating"],
                        mode="lines+markers",
                        line=dict(color=COLORS[0], width=2.5),
                        marker=dict(size=9, color=COLORS[0]),
                    )
                )
                fig.add_hline(
                    y=fdf["Rating"].mean(),
                    line_dash="dash",
                    line_color=_T["warn"],
                    annotation_text=f"Mean {fdf['Rating'].mean():.2f}",
                )
                fig.update_layout(
                    **_fig_layout(
                        yaxis=dict(
                            range=[3, 5.5], gridcolor=_T["grid"], linecolor=_T["line"]
                        )
                    ),
                    title="Average Rating by Year",
                )
                st.plotly_chart(fig, use_container_width=True)

# RATING PREDICTOR
elif page == "Rating Predictor":

    st.markdown('<span class="eyebrow">module 01</span>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f"{ICON_STAR} Rating Predictor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtext" style="max-width:600px;">Enter traveller and attraction details to '
        "predict the satisfaction rating using the best-performing regression model.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not MODELS_OK:
        st.markdown(
            '<div class="empty-state">Run the notebook to generate model artifacts.</div>',
            unsafe_allow_html=True,
        )
    else:
        input_col, result_col = st.columns([1, 1.15], gap="large")

        with input_col:
            st.markdown("##### Traveller profile")
            c1, c2 = st.columns(2)
            visit_year = c1.number_input("Visit Year", 2000, 2030, 2023, step=1)
            visit_month = c2.selectbox(
                "Visit Month",
                range(1, 13),
                format_func=lambda m: [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ][m - 1],
            )
            visit_mode = st.selectbox(
                "Visit Mode",
                [1, 2, 3, 4, 5],
                format_func=lambda v: [
                    "Business",
                    "Couples",
                    "Family",
                    "Friends",
                    "Solo",
                ][v - 1],
            )

            c3, c4, c5 = st.columns(3)
            continent_id = c3.selectbox(
                "Continent",
                [1, 2, 3, 4, 5],
                format_func=lambda v: {
                    1: "Africa",
                    2: "Americas",
                    3: "Asia",
                    4: "Europe",
                    5: "Oceania",
                }[v],
            )
            region_id = c4.number_input("Region ID", 1, 50, 1)
            country_id = c5.number_input("Country ID", 1, 300, 1)

            st.markdown("##### Attraction details")
            attr_opts = []
            if DATA_OK and df_item is not None:
                attr_opts = sorted(df_item["AttractionId"].tolist())
            attraction_id = st.selectbox(
                "Attraction ID", attr_opts if attr_opts else [1]
            )
            attr_type_id = st.number_input("Attraction Type ID", 1, 30, 1)

            st.markdown("##### Behavioural context")
            c6, c7 = st.columns(2)
            user_avg = c6.slider("Your Avg Rating", 1.0, 5.0, 4.0, 0.1)
            user_cnt = c7.number_input("Your Visit Count", 1, 200, 5)
            c8, c9 = st.columns(2)
            attr_avg = c8.slider("Attraction Avg Rating", 1.0, 5.0, 4.3, 0.1)
            attr_cnt = c9.number_input("Attraction Visit Count", 1, 20000, 1000)

            predict_clicked = st.button("Predict rating", use_container_width=True)

        with result_col:
            st.markdown("##### Where this trip sits vs the dataset")
            if DATA_OK and df is not None:
                sample = df.sample(min(1500, len(df)), random_state=42)
                fig = px.scatter(
                    sample,
                    x="UserAvgRating",
                    y="Rating",
                    opacity=0.3,
                    color_discrete_sequence=[_T["accent"]],
                )
                fig.add_trace(
                    go.Scatter(
                        x=[user_avg],
                        y=[3.0],
                        mode="markers",
                        marker=dict(
                            size=18,
                            color=_T["warn"],
                            symbol="star",
                            line=dict(width=1, color=_T["text-0"]),
                        ),
                        name="This traveller",
                    )
                )
                fig.update_layout(
                    **_fig_layout(),
                    height=320,
                    showlegend=False,
                    xaxis_title="User Avg Rating",
                    yaxis_title="Rating",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(
                    '<div class="empty-state">Load data to see population chart.</div>',
                    unsafe_allow_html=True,
                )

        if predict_clicked:
            season_enc = {
                1: 3,
                2: 3,
                3: 0,
                4: 0,
                5: 0,
                6: 2,
                7: 2,
                8: 2,
                9: 1,
                10: 1,
                11: 1,
                12: 3,
            }[visit_month]
            quarter_enc = {
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 1,
                6: 1,
                7: 2,
                8: 2,
                9: 2,
                10: 3,
                11: 3,
                12: 3,
            }[visit_month]

            X_in = np.array(
                [
                    [
                        visit_year,
                        visit_month,
                        visit_mode,
                        continent_id,
                        region_id,
                        country_id,
                        attraction_id,
                        attr_type_id,
                        user_avg,
                        user_cnt,
                        attr_avg,
                        attr_cnt,
                        season_enc,
                        quarter_enc,
                    ]
                ]
            )
            try:
                X_scaled = reg_scaler.transform(X_in) if use_sc_reg else X_in
                pred = float(np.clip(reg_model.predict(X_scaled)[0], 1.0, 5.0))
                stars = "★" * round(pred) + "☆" * (5 - round(pred))

                st.markdown(
                    f"""
                    <div class="result-banner">
                        <span class="tag">prediction result · {meta.get('best_reg_name','model')}</span>
                        <div class="title">{pred:.2f} / 5.00</div>
                        <div style="font-size:1.4rem; letter-spacing:0.12em; color:var(--warn); margin-bottom:0.5rem;">{stars}</div>
                        <div class="desc">Predicted satisfaction rating for this traveller–attraction combination.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)
                with m1:
                    stat_card("Predicted Rating", f"{pred:.2f}")
                with m2:
                    stat_card("Your Avg Rating", f"{user_avg:.1f}")
                with m3:
                    stat_card("Attraction Avg", f"{attr_avg:.1f}")

            except Exception as e:
                st.error(f"Prediction failed — {e}")

# MODE PREDICTOR
elif page == "Mode Predictor":

    st.markdown('<span class="eyebrow">module 02</span>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f"{ICON_USERS} Visit Mode Predictor</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtext" style="max-width:600px;">Predict whether a trip is Business, Couples, '
        "Family, Friends, or Solo — with full probability breakdown across all five modes.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not MODELS_OK:
        st.markdown(
            '<div class="empty-state">Run the notebook to generate model artifacts.</div>',
            unsafe_allow_html=True,
        )
    else:
        MODES = list(le_mode.classes_)

        input_col, chart_col = st.columns([1, 1.15], gap="large")

        with input_col:
            st.markdown("##### Traveller profile")
            c1, c2 = st.columns(2)
            visit_year = c1.number_input(
                "Visit Year", 2000, 2030, 2023, step=1, key="mc_yr"
            )
            visit_month = c2.selectbox(
                "Visit Month",
                range(1, 13),
                key="mc_mo",
                format_func=lambda m: [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ][m - 1],
            )
            c3, c4, c5 = st.columns(3)
            continent_id = c3.selectbox(
                "Continent",
                [1, 2, 3, 4, 5],
                key="mc_cont",
                format_func=lambda v: {
                    1: "Africa",
                    2: "Americas",
                    3: "Asia",
                    4: "Europe",
                    5: "Oceania",
                }[v],
            )
            region_id = c4.number_input("Region ID", 1, 50, 1, key="mc_reg")
            country_id = c5.number_input("Country ID", 1, 300, 1, key="mc_cty")

            st.markdown("##### Trip details")
            attr_opts = (
                sorted(df_item["AttractionId"].tolist())
                if DATA_OK and df_item is not None
                else [1]
            )
            attraction_id = st.selectbox("Attraction ID", attr_opts, key="mc_attr")
            attr_type_id = st.number_input(
                "Attraction Type ID", 1, 30, 1, key="mc_atype"
            )
            rating = st.slider("Rating Given", 1, 5, 4, key="mc_rat")

            st.markdown("##### Behavioural context")
            c6, c7 = st.columns(2)
            user_avg = c6.slider("Your Avg Rating", 1.0, 5.0, 4.0, 0.1, key="mc_uavg")
            user_cnt = c7.number_input("Your Visit Count", 1, 200, 5, key="mc_ucnt")
            c8, c9 = st.columns(2)
            attr_avg = c8.slider(
                "Attraction Avg Rating", 1.0, 5.0, 4.3, 0.1, key="mc_aavg"
            )
            attr_cnt = c9.number_input(
                "Attraction Visit Count", 1, 20000, 1000, key="mc_acnt"
            )

            predict_mode = st.button("Predict visit mode", use_container_width=True)

        with chart_col:
            st.markdown("##### Mode probability breakdown")
            st.markdown(
                (
                    '<div class="empty-state" style="margin-top:2rem;">Run prediction to see probabilities.</div>'
                    if not predict_mode
                    else ""
                ),
                unsafe_allow_html=True,
            )

        if predict_mode:
            season_enc = {
                1: 3,
                2: 3,
                3: 0,
                4: 0,
                5: 0,
                6: 2,
                7: 2,
                8: 2,
                9: 1,
                10: 1,
                11: 1,
                12: 3,
            }[visit_month]
            quarter_enc = {
                1: 0,
                2: 0,
                3: 0,
                4: 1,
                5: 1,
                6: 1,
                7: 2,
                8: 2,
                9: 2,
                10: 3,
                11: 3,
                12: 3,
            }[visit_month]

            X_in = np.array(
                [
                    [
                        visit_year,
                        visit_month,
                        continent_id,
                        region_id,
                        country_id,
                        attraction_id,
                        attr_type_id,
                        rating,
                        user_avg,
                        user_cnt,
                        attr_avg,
                        attr_cnt,
                        season_enc,
                        quarter_enc,
                    ]
                ]
            )
            try:
                X_scaled = clf_scaler.transform(X_in) if use_sc_clf else X_in
                pred_code = int(clf_model.predict(X_scaled)[0])
                pred_name = le_mode.inverse_transform([pred_code])[0]
                pred_meta = MODE_INFO.get(
                    pred_name, {"desc": "", "color": _T["accent"]}
                )

                try:
                    probs = clf_model.predict_proba(X_scaled)[0]
                except AttributeError:
                    probs = np.zeros(len(MODES))
                    probs[pred_code] = 1.0

                st.markdown(
                    f"""
                    <div class="result-banner" style="border-color:{pred_meta['color']};
                        background: linear-gradient(135deg, {pred_meta['color']}20, var(--bg-2));">
                        <span class="tag" style="color:{pred_meta['color']};">prediction · cluster {pred_code}</span>
                        <div class="title">{pred_name}</div>
                        <div class="desc">{pred_meta['desc']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<br>", unsafe_allow_html=True)

                prob_df = pd.DataFrame({"Mode": MODES, "Probability": probs * 100})
                prob_df = prob_df.sort_values("Probability", ascending=True)

                fig = go.Figure(
                    go.Bar(
                        x=prob_df["Probability"],
                        y=prob_df["Mode"],
                        orientation="h",
                        marker_color=[
                            pred_meta["color"] if m == pred_name else _T["line"]
                            for m in prob_df["Mode"]
                        ],
                        text=[f"{v:.1f}%" for v in prob_df["Probability"]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    **_fig_layout(
                        xaxis=dict(
                            range=[0, 115],
                            title="Probability (%)",
                            gridcolor=_T["grid"],
                            linecolor=_T["line"],
                        ),
                        height=300,
                    ),
                    title=f"Probability per Visit Mode — {meta.get('best_clf_name','')}",
                )
                st.plotly_chart(fig, use_container_width=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    stat_card("Predicted Mode", pred_name)
                with m2:
                    stat_card("Confidence", f"{probs[pred_code]*100:.1f}%")
                with m3:
                    stat_card("Cluster ID", str(pred_code))

            except Exception as e:
                st.error(f"Prediction failed — {e}")

# RECOMMENDATIONS
elif page == "Recommendations":

    st.markdown('<span class="eyebrow">module 03</span>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f"{ICON_COMPASS} Recommendation Engine</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtext" style="max-width:600px;">Hybrid collaborative + content-based filtering. '
        "Enter a user to get personalised attraction suggestions, or enter an attraction to find similar ones.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not MODELS_OK:
        st.markdown(
            '<div class="empty-state">Run the notebook to generate model artifacts.</div>',
            unsafe_allow_html=True,
        )
    else:
        name_map = {}
        if DATA_OK and df_item is not None:
            name_map = df_item.set_index("AttractionId")["Attraction"].to_dict()

        mode_col, param_col = st.columns([1.4, 1], gap="large")

        with mode_col:
            rec_type = st.radio(
                "Type",
                ["User-based (Hybrid)", "Content-based (Attraction)"],
                horizontal=True,
                key="rec_type",
            )

        with param_col:
            top_n = st.slider("Recommendations", 3, 10, 5, key="rec_n")
            if "Hybrid" in rec_type:
                alpha = st.slider(
                    "CF weight (α)",
                    0.0,
                    1.0,
                    0.6,
                    0.1,
                    key="rec_alpha",
                    help="1.0 = pure collaborative, 0.0 = pure content",
                )

        if "Hybrid" in rec_type:
            st.markdown("##### User selection")
            sample_users = []
            if DATA_OK and df is not None:
                sample_users = df["UserId"].value_counts().head(200).index.tolist()

            uid_mode = st.radio(
                "Input method",
                ["Pick from sample", "Enter User ID"],
                horizontal=True,
                key="uid_mode",
            )
            if uid_mode == "Pick from sample" and sample_users:
                user_id = int(
                    st.selectbox("Sample User", sample_users, key="rec_user_sel")
                )
            else:
                user_id = int(
                    st.number_input(
                        "User ID",
                        min_value=1,
                        value=int(sample_users[0]) if sample_users else 1,
                        key="rec_user_in",
                    )
                )

            rec_clicked = st.button(
                "Find recommendations", use_container_width=True, key="rec_go"
            )

            if rec_clicked:
                try:
                    if user_id not in uim.index:
                        st.warning("User not in training data.")
                    else:
                        row = uim.loc[user_id]
                        visited = row[row > 0].index.tolist()

                        valid = [v for v in visited if v in collab_sim]
                        if not valid:
                            top_items = (
                                df["AttractionId"]
                                .value_counts()
                                .head(top_n)
                                .index.tolist()
                                if DATA_OK and df is not None
                                else []
                            )
                            recs_final = [
                                (aid, name_map.get(aid, str(aid)), 0.0)
                                for aid in top_items
                            ]
                        else:
                            score_agg = {}
                            for v in valid:
                                for aid, sim_val in collab_sim[v].items():
                                    if aid not in visited:
                                        score_agg[aid] = score_agg.get(aid, 0) + float(
                                            sim_val
                                        ) * float(row[v])

                            if alpha < 1.0 and content_sim is not None:
                                cb_list = [
                                    content_sim[v]
                                    for v in visited
                                    if v in content_sim.index
                                ]
                                if cb_list:
                                    import pandas as _pd

                                    cb_avg = _pd.concat(cb_list, axis=1).mean(axis=1)
                                    for aid in cb_avg.index:
                                        if aid not in visited:
                                            c_v = score_agg.get(aid, 0)
                                            b_v = float(cb_avg[aid])
                                            score_agg[aid] = (
                                                alpha * c_v + (1 - alpha) * b_v
                                            )

                            sorted_recs = sorted(
                                score_agg.items(), key=lambda x: x[1], reverse=True
                            )
                            recs_final = [
                                (aid, name_map.get(aid, str(aid)), s)
                                for aid, s in sorted_recs[:top_n]
                            ]

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            f'<span class="badge">user {user_id}</span> '
                            f'<span style="color:var(--text-1); font-size:0.88rem;">'
                            f"visited {len(visited)} attraction(s) · showing {len(recs_final)} recommendations</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)

                        for rank, (aid, aname, score) in enumerate(recs_final, 1):
                            avg_r = (
                                df[df["AttractionId"] == aid]["Rating"].mean()
                                if DATA_OK and df is not None
                                else 0
                            )
                            st.markdown(
                                f'<div class="rec-card">'
                                f'<div class="rec-rank">#{rank}</div>'
                                f'<div class="rec-name">{aname}</div>'
                                f'<div class="rec-score">avg {avg_r:.2f} · score {score:.3f}</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        with st.expander(f"Visit history ({len(visited)} attractions)"):
                            if visited:
                                h = pd.DataFrame(
                                    {
                                        "AttractionId": visited,
                                        "Name": [
                                            name_map.get(a, str(a)) for a in visited
                                        ],
                                        "Rating": [float(row[a]) for a in visited],
                                    }
                                )
                                st.dataframe(
                                    h, use_container_width=True, hide_index=True
                                )

                except Exception as e:
                    st.error(f"Error — {e}")

        else:
            st.markdown("##### Attraction selection")
            attr_opts = sorted(name_map.keys()) if name_map else [1]
            sel_attr = st.selectbox(
                "Attraction",
                attr_opts,
                format_func=lambda a: name_map.get(a, str(a)),
                key="rec_attr_sel",
            )
            cb_clicked = st.button(
                "Find similar attractions", use_container_width=True, key="rec_cb_go"
            )

            if cb_clicked:
                try:
                    if content_sim is None or sel_attr not in content_sim.index:
                        st.warning("Attraction not in content similarity matrix.")
                    else:
                        sims = content_sim[sel_attr].sort_values(ascending=False)
                        sims = sims[sims.index != sel_attr].head(top_n)
                        anchor = name_map.get(sel_attr, str(sel_attr))

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            f'<span class="badge">based on</span> '
                            f'<span style="color:var(--text-0); font-weight:600;">{anchor}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("<br>", unsafe_allow_html=True)

                        for rank, (aid, score) in enumerate(sims.items(), 1):
                            st.markdown(
                                f'<div class="rec-card">'
                                f'<div class="rec-rank">#{rank}</div>'
                                f'<div class="rec-name">{name_map.get(aid,str(aid))}</div>'
                                f'<div class="rec-score">similarity {score:.4f}</div>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                        rec_df = pd.DataFrame(
                            {
                                "attraction": [
                                    name_map.get(a, str(a)) for a in sims.index
                                ],
                                "similarity": sims.values,
                            }
                        )
                        st.download_button(
                            "Download as CSV",
                            data=rec_df.to_csv(index=False).encode("utf-8"),
                            file_name="content_recommendations.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"Error — {e}")

# MODEL PERFORMANCE
elif page == "Model Performance":

    st.markdown('<span class="eyebrow">evaluation</span>', unsafe_allow_html=True)
    st.markdown(
        f'<h1 style="display:flex; align-items:center; gap:0.55rem; margin-bottom:0;">'
        f"{ICON_CPU} Model Performance</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtext" style="max-width:600px;">Full comparison of all trained models '
        "across regression and classification tasks.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if not MODELS_OK:
        st.markdown(
            '<div class="empty-state">Run the notebook to generate model artifacts.</div>',
            unsafe_allow_html=True,
        )
    else:
        tab_r, tab_c = st.tabs(["Regression Models", "Classification Models"])

        with tab_r:
            reg_res = meta.get("reg_results", [])
            if reg_res:
                rdf = pd.DataFrame(reg_res).sort_values("RMSE")
                best = meta.get("best_reg_name", "")
                br = rdf.iloc[0]

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    stat_card("Best Model", br["Model"][:14])
                with m2:
                    stat_card("RMSE", f"{br['RMSE']:.4f}")
                with m3:
                    stat_card("MAE", f"{br['MAE']:.4f}")
                with m4:
                    stat_card("R²", f"{br['R2']:.4f}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### RMSE by model")

                fig = go.Figure(
                    go.Bar(
                        y=rdf["Model"],
                        x=rdf["RMSE"],
                        orientation="h",
                        marker_color=[
                            _T["accent"] if m == best else _T["line"]
                            for m in rdf["Model"]
                        ],
                        text=[f"{v:.4f}" for v in rdf["RMSE"]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    **_fig_layout(
                        margin=dict(l=5, r=60, t=20, b=20),
                        yaxis=dict(
                            autorange="reversed",
                            gridcolor=_T["grid"],
                            linecolor=_T["line"],
                        ),
                        height=380,
                    ),
                    xaxis_title="RMSE",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### Full leaderboard")
                d = rdf.copy()
                d.insert(
                    0, "Best?", d["Model"].apply(lambda m: "🏆" if m == best else "")
                )
                st.dataframe(
                    d[["Best?", "Model", "RMSE", "MAE", "R2"]],
                    use_container_width=True,
                    hide_index=True,
                )

        with tab_c:
            clf_res = meta.get("clf_results", [])
            if clf_res:
                cdf = pd.DataFrame(clf_res).sort_values("F1", ascending=False)
                # Best benchmark model = highest weighted F1
                bc = cdf.iloc[0]

                # Deployment model = model selected for Streamlit deployment
                deployment_name = meta.get("best_clf_name", "")
                deployment_rows = cdf[cdf["Model"] == deployment_name]
                dc = deployment_rows.iloc[0] if not deployment_rows.empty else None

                m1, m2, m3, m4 = st.columns(4)

                with m1:
                    stat_card("Best Benchmark", bc["Model"][:14])

                with m2:
                    stat_card("Accuracy", f"{bc['Accuracy']:.4f}")

                with m3:
                    stat_card("F1 (Weighted)", f"{bc['F1']:.4f}")

                with m4:
                    stat_card("Precision", f"{bc['Precision']:.4f}")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### F1 score by model")

                fig = go.Figure(
                    go.Bar(
                        y=cdf.sort_values("F1")["Model"],
                        x=cdf.sort_values("F1")["F1"],
                        orientation="h",
                        marker_color=[
                            _T["accent"] if m == best else _T["line"]
                            for m in cdf.sort_values("F1")["Model"]
                        ],
                        text=[f"{v:.4f}" for v in cdf.sort_values("F1")["F1"]],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    **_fig_layout(
                        margin=dict(l=5, r=60, t=20, b=20),
                        height=380,
                    ),
                    xaxis_title="Weighted F1",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("##### Radar — top 5 models")
                top5 = cdf.head(5)
                cats = ["Accuracy", "Precision", "Recall", "F1"]
                fig2 = go.Figure()
                for i, (_, row) in enumerate(top5.iterrows()):
                    vals = [row[c] for c in cats] + [row[cats[0]]]
                    fig2.add_trace(
                        go.Scatterpolar(
                            r=vals,
                            theta=cats + [cats[0]],
                            fill="toself",
                            opacity=0.5,
                            name=row["Model"],
                            line=dict(color=COLORS[i % len(COLORS)], width=2),
                        )
                    )
                fig2.update_layout(
                    **_fig_layout(),
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Multi-metric Radar — Top 5",
                    height=400,
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("##### Full leaderboard")
                d2 = cdf.copy()
                d2.insert(
                    0, "Best?", d2["Model"].apply(lambda m: "🏆" if m == best else "")
                )
                st.dataframe(
                    d2[["Best?", "Model", "Accuracy", "Precision", "Recall", "F1"]],
                    use_container_width=True,
                    hide_index=True,
                )

st.markdown(
    '<div class="footer-note">Tourism Experience Analytics · Streamlit · Scikit-learn · Plotly</div>',
    unsafe_allow_html=True,
)
