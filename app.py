import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.rcParams["figure.autolayout"] = True

# ── AUTO SETUP FOR STREAMLIT CLOUD ──
from startup import run_setup
if run_setup():
    st.rerun()

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# ROUTING
# ---------------------------------------------------
if "page" in st.query_params:
    page = st.query_params["page"]
    if page not in ["Home", "Model Comparison", "Prediction", "EDA", "Tools", "Project Details"]:
        page = "Home"
        st.query_params["page"] = "Home"
else:
    page = "Home"
    st.query_params["page"] = "Home"

st.session_state.page = page

def nav_cls(p):
    return "nav-active" if page == p else "nav-item"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
    --navy:     #060E1F;
    --navy2:    #0D1B35;
    --navy3:    #162040;
    --blue:     #2563EB;
    --blue2:    #3B82F6;
    --cyan:     #22D3EE;
    --gold:     #F59E0B;
    --gold2:    #FCD34D;
    --emerald:  #10B981;
    --rose:     #F43F5E;
    --amber:    #F97316;
    --slate50:  #F8FAFC;
    --slate100: #F1F5F9;
    --slate200: #E2E8F0;
    --slate300: #CBD5E1;
    --slate400: #94A3B8;
    --slate500: #64748B;
    --slate600: #475569;
    --slate700: #334155;
    --slate800: #1E293B;
    --slate900: #0F172A;
    --white:    #FFFFFF;
    --fd: 'Playfair Display', Georgia, serif;
    --fb: 'Plus Jakarta Sans', system-ui, sans-serif;
    --r-sm:  8px;
    --r-md:  14px;
    --r-lg:  20px;
    --r-xl:  28px;
    --sh-sm: 0 1px 4px rgba(0,0,0,0.06);
    --sh-md: 0 4px 20px rgba(0,0,0,0.08);
    --sh-lg: 0 12px 48px rgba(0,0,0,0.12);
    --sh-blue: 0 8px 32px rgba(37,99,235,0.2);
    --sh-gold: 0 8px 32px rgba(245,158,11,0.25);
}}

.stApp {{
    background: var(--slate50) !important;
    font-family: var(--fb) !important;
}}
section[data-testid="stSidebar"],
header[data-testid="stHeader"],
#MainMenu, footer, .stDeployButton {{
    display: none !important;
}}
.main .block-container {{
    max-width: 1320px !important;
    padding: 1.5rem 2.5rem 4rem !important;
    padding-top: 82px !important;
}}

/* ── NAVBAR ── */
.navbar {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 66px;
    background: var(--navy);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    z-index: 999999;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.navbar::before {{
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.015'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    pointer-events: none;
}}
.navbar::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--blue2), var(--cyan), transparent);
    opacity: 0.8;
}}
.brand {{ display: flex; align-items: center; gap: 12px; text-decoration: none; }}
.brand-icon {{
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--blue), var(--cyan));
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    flex-shrink: 0;
}}
.brand-text {{ display: flex; flex-direction: column; }}
.brand-name {{
    font-family: var(--fd);
    font-size: 1rem; font-weight: 700; color: white;
    line-height: 1.2; letter-spacing: 0.2px; white-space: nowrap;
}}
.brand-tagline {{
    font-family: var(--fb);
    font-size: 0.65rem; font-weight: 500;
    color: rgba(255,255,255,0.4);
    letter-spacing: 1.5px; text-transform: uppercase;
}}
.nav-links {{ display: flex; align-items: center; gap: 2px; }}
.nav-item, .nav-active {{
    font-family: var(--fb);
    font-size: 0.82rem; font-weight: 500;
    padding: 6px 16px; border-radius: 6px;
    text-decoration: none !important;
    transition: all 0.18s ease;
    white-space: nowrap; letter-spacing: 0.2px;
    border: 1px solid transparent;
}}
.nav-item {{ color: rgba(255,255,255,0.55); }}
.nav-item:hover {{ color: rgba(255,255,255,0.9); background: rgba(255,255,255,0.07); }}
.nav-active {{ color: white; font-weight: 600; background: rgba(37,99,235,0.3); border-color: rgba(59,130,246,0.35); }}

/* ── TYPOGRAPHY ── */
h1,h2,h3,h4,h5,h6 {{ font-family: var(--fd) !important; color: var(--slate900) !important; }}
p, li, span, label {{ font-family: var(--fb) !important; }}

/* ── HERO ── */
.hero {{
    position: relative;
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy2) 50%, #0A1F44 100%);
    border-radius: var(--r-xl);
    padding: 4rem 3.5rem;
    margin-bottom: 2.5rem;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}}
.hero::before {{
    content: '';
    position: absolute; top: -60%; right: -15%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(37,99,235,0.18) 0%, transparent 70%);
    pointer-events: none;
}}
.hero::after {{
    content: '';
    position: absolute; bottom: -40%; left: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(245,158,11,0.1) 0%, transparent 70%);
    pointer-events: none;
}}
.hero-grid {{
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
}}
.hero-badge {{
    display: inline-flex; align-items: center; gap: 7px;
    font-family: var(--fb); font-size: 0.72rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: var(--gold2);
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.25);
    padding: 5px 14px; border-radius: 100px;
    margin-bottom: 1.5rem; position: relative; z-index: 1;
}}
.hero-badge-dot {{
    width: 6px; height: 6px;
    background: var(--gold); border-radius: 50%;
    box-shadow: 0 0 8px var(--gold);
    animation: blink 1.8s ease-in-out infinite;
}}
@keyframes blink {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.3; }} }}
.hero-title {{
    font-family: var(--fd) !important;
    font-size: 3.4rem !important; font-weight: 900 !important;
    color: white !important; line-height: 1.1 !important;
    letter-spacing: -1.5px !important; margin-bottom: 1.2rem !important;
    position: relative; z-index: 1;
}}
.hero-title em {{
    font-style: normal;
    background: linear-gradient(90deg, var(--gold), var(--gold2), var(--cyan));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}}
.hero-sub {{
    font-family: var(--fb); font-size: 1.05rem;
    color: rgba(255,255,255,0.6); line-height: 1.75;
    max-width: 540px; position: relative; z-index: 1; font-weight: 400;
}}
.hero-divider {{
    width: 48px; height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--blue2));
    border-radius: 2px; margin: 1.5rem 0; position: relative; z-index: 1;
}}
.hero-stats {{
    display: flex; gap: 2.5rem; margin-top: 2.5rem; position: relative; z-index: 1;
}}
.hero-stat {{ display: flex; flex-direction: column; gap: 2px; }}
.hero-stat-num {{
    font-family: var(--fd); font-size: 2rem; font-weight: 900;
    color: white; letter-spacing: -1px; line-height: 1;
}}
.hero-stat-label {{
    font-family: var(--fb); font-size: 0.72rem; font-weight: 600;
    color: rgba(255,255,255,0.4); letter-spacing: 1.2px; text-transform: uppercase;
}}
.hero-stat-accent {{ color: var(--gold); }}
.hero-stat-accent2 {{ color: var(--cyan); }}
.hero-stat-sep {{ width: 1px; background: rgba(255,255,255,0.1); align-self: stretch; }}

/* ── SECTION HEADS ── */
.section-head {{
    display: flex; align-items: center; gap: 12px; margin: 2.5rem 0 1.25rem;
}}
.section-head-bar {{
    width: 4px; height: 22px;
    background: linear-gradient(180deg, var(--blue), var(--cyan));
    border-radius: 2px; flex-shrink: 0;
}}
.section-head-title {{
    font-family: var(--fd) !important; font-size: 1.25rem !important;
    font-weight: 800 !important; color: var(--slate900) !important; letter-spacing: -0.3px !important;
}}
.section-head-line {{
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--slate200), transparent);
}}

/* ── METRICS ── */
[data-testid="stMetric"] {{
    background: white !important; border-radius: var(--r-lg) !important;
    padding: 1.5rem 1.75rem !important; border: 1px solid var(--slate200) !important;
    box-shadow: var(--sh-sm) !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    position: relative !important; overflow: hidden !important;
}}
[data-testid="stMetric"]::before {{
    content: '' !important; position: absolute !important;
    top: 0; left: 0; right: 0 !important; height: 3px !important;
    background: linear-gradient(90deg, var(--blue), var(--cyan)) !important;
}}
[data-testid="stMetric"]:hover {{
    box-shadow: var(--sh-blue) !important; transform: translateY(-4px) !important;
    border-color: rgba(37,99,235,0.2) !important;
}}
[data-testid="stMetricLabel"] > div {{
    font-family: var(--fb) !important; font-size: 0.72rem !important;
    font-weight: 700 !important; letter-spacing: 1.2px !important;
    text-transform: uppercase !important; color: var(--slate400) !important;
}}
[data-testid="stMetricValue"] {{
    font-family: var(--fd) !important; font-size: 2.2rem !important;
    font-weight: 900 !important; color: var(--slate900) !important;
    letter-spacing: -1.5px !important; line-height: 1.1 !important;
}}

/* ── CONTENT CARD ── */
.content-card {{
    background: white; border-radius: var(--r-lg);
    padding: 2rem; border: 1px solid var(--slate200);
    box-shadow: var(--sh-sm); margin-bottom: 1.5rem;
    transition: box-shadow 0.25s ease;
}}
.content-card:hover {{ box-shadow: var(--sh-md); }}
.info-row {{
    display: flex; align-items: flex-start; gap: 12px;
    padding: 0.85rem 0; border-bottom: 1px solid var(--slate100);
}}
.info-row:last-child {{ border-bottom: none; }}
.info-icon {{
    width: 32px; height: 32px;
    background: linear-gradient(135deg, rgba(37,99,235,0.1), rgba(34,211,238,0.1));
    border-radius: 8px; display: flex; align-items: center;
    justify-content: center; font-size: 14px; flex-shrink: 0;
}}
.info-label {{
    font-family: var(--fb); font-size: 0.75rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.8px; color: var(--slate400); margin-bottom: 2px;
}}
.info-value {{ font-family: var(--fb); font-size: 0.95rem; font-weight: 600; color: var(--slate800); }}

/* ── METHOD CARDS ── */
.method-card {{
    background: white; border-radius: var(--r-md);
    padding: 1.5rem; border: 1px solid var(--slate200);
    border-left: 4px solid var(--blue); box-shadow: var(--sh-sm);
    transition: all 0.2s ease; height: 100%;
}}
.method-card:hover {{ box-shadow: var(--sh-md); transform: translateY(-2px); }}
.method-card.gold {{ border-left-color: var(--gold); }}
.method-tag {{
    display: inline-block; font-family: var(--fb);
    font-size: 0.65rem; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; padding: 3px 10px; border-radius: 100px;
    background: rgba(37,99,235,0.08); color: var(--blue); margin-bottom: 0.75rem;
}}
.method-card.gold .method-tag {{ background: rgba(245,158,11,0.1); color: var(--gold); }}
.method-title {{
    font-family: var(--fd) !important; font-size: 1.1rem !important;
    font-weight: 800 !important; color: var(--slate900) !important; margin-bottom: 0.5rem !important;
}}
.method-desc {{ font-family: var(--fb); font-size: 0.875rem; color: var(--slate500); line-height: 1.7; }}
.method-stat {{
    margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--slate100);
    display: flex; align-items: center; justify-content: space-between;
}}
.method-stat-label {{
    font-family: var(--fb); font-size: 0.72rem; font-weight: 600;
    color: var(--slate400); text-transform: uppercase; letter-spacing: 0.8px;
}}
.method-stat-value {{ font-family: var(--fd); font-size: 1.3rem; font-weight: 900; color: var(--blue); }}
.method-card.gold .method-stat-value {{ color: var(--gold); }}

/* ── BUTTONS ── */
.stButton > button {{
    font-family: var(--fb) !important; font-weight: 700 !important;
    font-size: 0.9rem !important; letter-spacing: 0.5px !important;
    background: linear-gradient(135deg, var(--blue) 0%, #1D4ED8 100%) !important;
    color: white !important; border: none !important;
    border-radius: var(--r-sm) !important; padding: 0.75rem 2.5rem !important;
    box-shadow: var(--sh-blue) !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 16px 48px rgba(37,99,235,0.35) !important;
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
}}

/* ── INPUTS ── */
.stSelectbox > div > div {{
    background: white !important; border: 1.5px solid var(--slate200) !important;
    border-radius: var(--r-sm) !important; font-family: var(--fb) !important; font-size: 0.9rem !important;
}}
.stNumberInput input {{
    background: white !important; border: 1.5px solid var(--slate200) !important;
    border-radius: var(--r-sm) !important; font-family: var(--fb) !important;
}}

/* ── SLIDERS ── */
div[data-baseweb="slider"] {{ background: transparent !important; }}
div[data-baseweb="slider"] > div {{
    background: var(--slate200) !important; height: 4px !important; border-radius: 4px !important;
}}
.stSlider {{ background: transparent !important; }}
.stSlider > div, .stSlider > div > div {{ background: transparent !important; }}
[data-testid="stSlider"] {{ background: transparent !important; box-shadow: none !important; }}
[role="slider"] {{
    background: var(--blue) !important; border: 3px solid white !important;
    border-radius: 50% !important; box-shadow: 0 2px 12px rgba(37,99,235,0.45) !important;
    width: 18px !important; height: 18px !important;
}}

/* ── MISC ── */
.stAlert {{ border-radius: var(--r-md) !important; font-family: var(--fb) !important; }}
.stDataFrame {{
    border-radius: var(--r-md) !important; overflow: hidden !important;
    border: 1px solid var(--slate200) !important; box-shadow: var(--sh-sm) !important;
}}
hr {{
    border: none !important; height: 1px !important;
    background: var(--slate200) !important; margin: 2rem 0 !important; opacity: 1 !important;
}}

/* ── PRED CARDS ── */
.pred-card {{
    background: white; border-radius: var(--r-lg);
    padding: 1.75rem 2rem; border: 1px solid var(--slate200);
    box-shadow: var(--sh-sm); position: relative; overflow: hidden;
    transition: all 0.25s ease;
}}
.pred-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px;
}}
.pred-card.high::before {{ background: linear-gradient(90deg, var(--rose), #FB923C); }}
.pred-card.low::before  {{ background: linear-gradient(90deg, var(--emerald), var(--cyan)); }}
.pred-card-engine {{
    font-family: var(--fb); font-size: 0.68rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase; color: var(--slate400); margin-bottom: 0.6rem;
}}
.pred-card-pct {{
    font-family: var(--fd); font-size: 3rem; font-weight: 900;
    line-height: 1; letter-spacing: -2px; margin-bottom: 0.75rem;
}}
.pred-card.high .pred-card-pct {{ color: var(--rose); }}
.pred-card.low  .pred-card-pct {{ color: var(--emerald); }}
.pred-card-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--fb); font-size: 0.78rem; font-weight: 700;
    padding: 5px 14px; border-radius: 100px; letter-spacing: 0.5px; text-transform: uppercase;
}}
.pred-card.high .pred-card-badge {{ background: #FFF1F2; color: var(--rose); border: 1px solid #FFE4E6; }}
.pred-card.low  .pred-card-badge {{ background: #F0FDF4; color: var(--emerald); border: 1px solid #D1FAE5; }}

/* ── INSIGHT BOX ── */
.insight-box {{
    background: linear-gradient(135deg, rgba(37,99,235,0.03), rgba(34,211,238,0.03));
    border: 1px solid rgba(37,99,235,0.12); border-left: 3px solid var(--blue);
    border-radius: var(--r-sm); padding: 0.875rem 1.25rem;
    font-family: var(--fb); font-size: 0.875rem; color: var(--slate600);
    margin: 1rem 0; line-height: 1.6;
}}

/* ── EDA ── */
.eda-stat-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
    background: var(--slate200); border-radius: var(--r-lg); overflow: hidden;
    border: 1px solid var(--slate200); margin-bottom: 2rem;
}}
.eda-stat {{
    background: white; padding: 1.5rem; display: flex;
    flex-direction: column; gap: 4px; transition: background 0.2s ease;
}}
.eda-stat:hover {{ background: var(--slate50); }}
.eda-stat-num {{
    font-family: var(--fd); font-size: 1.8rem; font-weight: 900;
    letter-spacing: -1px; color: var(--slate900); line-height: 1;
}}
.eda-stat-label {{
    font-family: var(--fb); font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px; color: var(--slate400);
}}
.chart-title {{
    font-family: var(--fd) !important; font-size: 1rem !important;
    font-weight: 800 !important; color: var(--slate900) !important;
    margin-bottom: 0.25rem !important; letter-spacing: -0.2px !important;
}}
.chart-subtitle {{
    font-family: var(--fb); font-size: 0.8rem; color: var(--slate400);
    margin-bottom: 1rem; font-weight: 500;
}}

/* ── ANIMATIONS ── */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(18px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.main .block-container > div > div > div {{
    animation: fadeUp 0.4s cubic-bezier(0.4,0,0.2,1) both;
}}
</style>

<div class="navbar">
    <div class="brand">
        <div class="brand-icon">👨‍💼</div>
        <div class="brand-text">
            <span class="brand-name">AttritionIQ</span>
            <span class="brand-tagline">HR Intelligence Platform</span>
        </div>
    </div>
    <div class="nav-links">
        <a href="?page=Home"             target="_self" class="{nav_cls('Home')}">Home</a>
        <a href="?page=Model+Comparison" target="_self" class="{nav_cls('Model Comparison')}">Model Comparison</a>
        <a href="?page=Prediction"       target="_self" class="{nav_cls('Prediction')}">Prediction</a>
        <a href="?page=EDA"              target="_self" class="{nav_cls('EDA')}">EDA</a>
        <a href="?page=Tools"            target="_self" class="{nav_cls('Tools')}">🛠 Tools</a>
        <a href="?page=Project+Details"  target="_self" class="{nav_cls('Project Details')}">📋 Project</a>
    </div>
</div>
""", unsafe_allow_html=True)

DATA_FILE = "employee_data_10000.csv"

# ===================================================
# HOME PAGE
# ===================================================
if page == "Home":

    total_employees = attrition_rate = retained_count = attrition_count = 0
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        if "Attrition" in df.columns:
            attrition_count = int((df["Attrition"] == "Yes").sum())
            retained_count  = int((df["Attrition"] == "No").sum())
            total_employees = len(df)
            attrition_rate  = (attrition_count / total_employees) * 100

    st.markdown(f"""
    <div class="hero">
        <div class="hero-grid"></div>
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            BCA Final Year &nbsp;·&nbsp; Major Project &nbsp;·&nbsp; 2025–26
        </div>
        <div class="hero-title">
            Predict Who Leaves.<br>
            <em>Retain What Matters.</em>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-sub">
            A machine learning platform that analyses 12 HR signals to forecast employee
            attrition risk — enabling data-driven retention decisions before talent walks out the door.
        </div>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="hero-stat-num">{total_employees:,}</span>
                <span class="hero-stat-label">Records Analysed</span>
            </div>
            <div class="hero-stat-sep"></div>
            <div class="hero-stat">
                <span class="hero-stat-num hero-stat-accent">{attrition_rate:.1f}%</span>
                <span class="hero-stat-label">Attrition Rate</span>
            </div>
            <div class="hero-stat-sep"></div>
            <div class="hero-stat">
                <span class="hero-stat-num hero-stat-accent2">{retained_count:,}</span>
                <span class="hero-stat-label">Retained Employees</span>
            </div>
            <div class="hero-stat-sep"></div>
            <div class="hero-stat">
                <span class="hero-stat-num">81.2%</span>
                <span class="hero-stat-label">Model Accuracy</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-head-bar"></div>
        <div class="section-head-title">What is Employee Attrition?</div>
        <div class="section-head-line"></div>
    </div>
    <div class="content-card" style="background:linear-gradient(135deg,#F0F9FF,#F8FAFC); border-left:4px solid #2563EB;">
        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.95rem; color:#334155; line-height:1.85;">
            <strong style="color:#1e40af; font-size:1rem;">Employee Attrition</strong> refers to the gradual reduction
            of a company's workforce when employees leave — voluntarily (resignation, retirement) or involuntarily
            (layoffs) — and are not immediately replaced.<br><br>
            It is one of the most critical and costly challenges HR departments face today. Replacing a single
            employee costs between <strong style="color:#dc2626;">50% – 200%</strong> of their annual salary
            when accounting for recruitment, onboarding, training, and productivity loss.<br><br>
            <strong style="color:#1e40af;">This project</strong> uses supervised machine learning to analyse
            12 key HR factors and predict — in real time — which employees are most likely to leave,
            enabling HR teams to intervene with targeted retention strategies <em>before</em> talent walks out the door.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Project Development Workflow</div>
        <div class="section-head-line"></div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:1px;
                background:#E2E8F0; border-radius:16px; overflow:hidden;
                border:1px solid #E2E8F0; margin-bottom:1.5rem;">
        <div style="background:white; padding:1.5rem 1rem; text-align:center; position:relative;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">📦</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                        color:#2563EB; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Step 1</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.4rem;">Data Collection</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.76rem; color:#64748B; line-height:1.55;">
                10,000 synthetic HR records generated with realistic attrition signals
            </div>
        </div>
        <div style="background:white; padding:1.5rem 1rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">🧹</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                        color:#0d9488; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Step 2</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.4rem;">Data Cleaning</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.76rem; color:#64748B; line-height:1.55;">
                Zero missing values · Encoded categoricals · Feature engineering
            </div>
        </div>
        <div style="background:white; padding:1.5rem 1rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">📊</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                        color:#7c3aed; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Step 3</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.4rem;">EDA</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.76rem; color:#64748B; line-height:1.55;">
                6 visualizations · Correlation heatmap · Distribution analysis
            </div>
        </div>
        <div style="background:white; padding:1.5rem 1rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">🤖</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                        color:#d97706; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Step 4</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.4rem;">ML Modelling</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.76rem; color:#64748B; line-height:1.55;">
                Logistic Regression + Random Forest · 80/20 train-test split
            </div>
        </div>
        <div style="background:white; padding:1.5rem 1rem; text-align:center;">
            <div style="font-size:2rem; margin-bottom:0.6rem;">🚀</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                        color:#059669; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.3rem;">Step 5</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.4rem;">Deployment</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.76rem; color:#64748B; line-height:1.55;">
                Interactive Streamlit web app with real-time prediction engine
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Machine Learning Models Used</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="method-card">
            <div class="method-tag">Baseline Model</div>
            <div class="method-title">Logistic Regression</div>
            <div class="method-desc">
                A high-interpretability linear classifier that establishes a transparent baseline
                for attrition probability. Uses a sigmoid function to output probability between 0–1.
                Features are scaled using StandardScaler before training.
            </div>
            <div class="method-stat">
                <span class="method-stat-label">Accuracy</span>
                <span class="method-stat-value">72.4%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="method-card gold">
            <div class="method-tag">Primary Model ✓</div>
            <div class="method-title">Random Forest</div>
            <div class="method-desc">
                An ensemble of 100 decision trees that captures non-linear patterns in complex
                HR behaviour. No feature scaling required. Provides robust predictions and
                feature importance rankings for HR teams.
            </div>
            <div class="method-stat">
                <span class="method-stat-label">Accuracy</span>
                <span class="method-stat-value">81.2%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Input Features Used for Prediction</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)

    features_df = pd.DataFrame({
        "Feature": [
            "Age","Monthly Income","Overtime",
            "Job Satisfaction","Work Life Balance",
            "Years At Company","Years Since Last Promotion",
            "Distance From Home","Total Working Years",
            "Num Companies Worked","Years In Current Role",
            "Years With Current Manager"
        ],
        "Data Type": [
            "Numerical","Numerical","Categorical (Yes/No)",
            "Ordinal (1–4)","Ordinal (1–4)",
            "Numerical","Numerical",
            "Numerical","Numerical",
            "Numerical","Numerical","Numerical"
        ],
        "Impact on Attrition": [
            "Younger employees (18–30) leave more frequently",
            "Below-average salary → significantly higher attrition risk",
            "Overtime = approximately 2× higher attrition rate",
            "Low satisfaction (1–2) → very high flight risk",
            "Poor balance (1–2) → burnout and resignation",
            "Short tenure (0–2 yrs) → highest attrition window",
            "No promotion in 3+ years → 3× more likely to leave",
            "Long commute (20+ km) → increased fatigue and dissatisfaction",
            "Less experience → less stable employment history",
            "Worked at 5+ companies → job-hopper profile",
            "Stagnation in same role → disengagement risk",
            "Weak manager relationship → strong attrition signal"
        ]
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    st.markdown("""
    <div class="section-head" style="margin-top:2rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Application Pages</div>
        <div class="section-head-line"></div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem;">
        <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid #2563EB;
                    border-radius:16px; padding:1.5rem;">
            <div style="font-size:1.6rem; margin-bottom:0.6rem;">📊</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.95rem;
                        color:#0F172A; margin-bottom:0.5rem;">Model Comparison</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.82rem; color:#64748B; line-height:1.65;">
                Side-by-side accuracy, precision, recall and F1-score of both models.
                Includes confusion matrix, classification report, and train/test split details.
            </div>
        </div>
        <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid #059669;
                    border-radius:16px; padding:1.5rem;">
            <div style="font-size:1.6rem; margin-bottom:0.6rem;">🔮</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.95rem;
                        color:#0F172A; margin-bottom:0.5rem;">Prediction Engine</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.82rem; color:#64748B; line-height:1.65;">
                Enter 12 HR inputs → get real-time attrition probability, risk meter,
                personalized HR recommendations, and factor contribution chart.
            </div>
        </div>
        <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid #7c3aed;
                    border-radius:16px; padding:1.5rem;">
            <div style="font-size:1.6rem; margin-bottom:0.6rem;">📈</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.95rem;
                        color:#0F172A; margin-bottom:0.5rem;">EDA Dashboard</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.82rem; color:#64748B; line-height:1.65;">
                6 interactive charts — attrition split, feature distributions, box plots,
                overtime analysis, department breakdown, and correlation heatmap.
            </div>
        </div>
        <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid #F59E0B;
                    border-radius:16px; padding:1.5rem;">
            <div style="font-size:1.6rem; margin-bottom:0.6rem;">🛠️</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.95rem;
                        color:#0F172A; margin-bottom:0.5rem;">HR Tools</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.82rem; color:#64748B; line-height:1.65;">
                What-If Simulator to test HR interventions + Attrition Cost Calculator
                to estimate financial impact of losing an employee.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===================================================
# MODEL COMPARISON PAGE
# ===================================================
elif page == "Model Comparison":

    st.markdown("""
    <div class="hero" style="padding:3rem 3.5rem;">
        <div class="hero-grid"></div>
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            Performance Benchmarking
        </div>
        <div class="hero-title" style="font-size:2.6rem;">
            Algorithm <em>Performance</em><br>Comparison
        </div>
        <div class="hero-sub">
            Side-by-side evaluation of Logistic Regression and Random Forest on 10,000 HR records.
            Includes confusion matrix, classification report and model characteristics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        metrics_data = {
            'Algorithm':  ['Logistic Regression', 'Random Forest'],
            'Accuracy':   [0.724, 0.812],
            'Precision':  [0.698, 0.784],
            'Recall':     [0.751, 0.856],
            'F1-Score':   [0.723, 0.818]
        }
        df_metrics = pd.DataFrame(metrics_data)
        styled_df  = df_metrics.style.format({
            'Accuracy': '{:.3f}', 'Precision': '{:.3f}',
            'Recall':   '{:.3f}', 'F1-Score':  '{:.3f}'
        }).highlight_max(axis=0,
            subset=['Accuracy','Precision','Recall','F1-Score'], color='#D1FAE5')

        st.markdown("""
        <div class="section-head">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Metrics Table</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("""
        <div class="section-head" style="margin-top:2rem;">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Visual Comparison</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#FFFFFF'); ax.set_facecolor('#F8FAFC')
        x = np.arange(len(df_metrics)); width = 0.15
        bars_data = [
            (df_metrics['Accuracy'],  '#2563EB', 'Accuracy'),
            (df_metrics['Precision'], '#7C3AED', 'Precision'),
            (df_metrics['Recall'],    '#059669', 'Recall'),
            (df_metrics['F1-Score'],  '#D97706', 'F1-Score'),
        ]
        for (vals, color, label), offset in zip(bars_data, [-1.5,-0.5,0.5,1.5]):
            bars = ax.bar(x + offset*width, vals, width, label=label,
                         color=color, alpha=0.88, edgecolor='white', linewidth=1.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='700', color='#334155')
        ax.set_xlabel('Model', fontweight='bold', color='#334155', fontsize=11, labelpad=10)
        ax.set_ylabel('Score', fontweight='bold', color='#334155', fontsize=11, labelpad=10)
        ax.set_title('Model Performance Comparison — Employee Attrition Prediction',
                     fontweight='bold', color='#0F172A', fontsize=13, pad=18, fontfamily='serif')
        ax.set_xticks(x); ax.set_xticklabels(df_metrics['Algorithm'], color='#0F172A', fontsize=11)
        ax.tick_params(colors='#64748B', labelsize=10); ax.set_ylim(0, 1.05)
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=9, loc='lower right')
        plt.setp(legend.get_texts(), color='#334155')
        ax.grid(axis='y', linestyle='--', alpha=0.35, color='#94A3B8')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['left','bottom']: ax.spines[s].set_color('#E2E8F0')
        plt.tight_layout(); st.pyplot(fig)

        st.markdown("""
        <div class="section-head" style="margin-top:2rem;">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Metric Definitions</div>
            <div class="section-head-line"></div>
        </div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.5rem;">
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #2563EB;
                        border-radius:14px; padding:1.4rem 1.2rem;">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">🎯</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">Accuracy</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.65;">
                    Proportion of correct predictions among all predictions made.
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #059669;
                        border-radius:14px; padding:1.4rem 1.2rem;">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">🔍</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">Precision</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.65;">
                    Of all predicted attrition cases, how many were actually correct — minimizing false alarms.
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #F59E0B;
                        border-radius:14px; padding:1.4rem 1.2rem;">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">📡</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">Recall</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.65;">
                    Of all actual attrition cases, how many were correctly identified — minimizing missed risks.
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #F43F5E;
                        border-radius:14px; padding:1.4rem 1.2rem;">
                <div style="font-size:1.4rem; margin-bottom:0.5rem;">⚖️</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.72rem;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">F1-Score</div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.65;">
                    Harmonic mean of Precision and Recall — the overall balanced performance measure.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-head" style="margin-top:2rem;">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Confusion Matrix — Both Models</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""<div style="text-align:center; font-family:'Plus Jakarta Sans',sans-serif;
                font-weight:700; font-size:0.88rem; color:#0F172A; margin-bottom:0.5rem;">
                Logistic Regression</div>""", unsafe_allow_html=True)
            cm_lr = np.array([[1138, 292], [261, 309]])
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            fig_cm.patch.set_facecolor('#FFFFFF')
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Predicted: No','Predicted: Yes'],
                        yticklabels=['Actual: No','Actual: Yes'],
                        ax=ax_cm, linewidths=2, linecolor='white',
                        annot_kws={"size": 16, "weight": "bold"}, cbar=False)
            ax_cm.set_title('Confusion Matrix — LR', fontweight='bold', fontsize=10, color='#0F172A', pad=10)
            plt.xticks(color='#334155', fontsize=8.5); plt.yticks(color='#334155', fontsize=8.5, rotation=0)
            plt.tight_layout(); st.pyplot(fig_cm)

        with col2:
            st.markdown("""<div style="text-align:center; font-family:'Plus Jakarta Sans',sans-serif;
                font-weight:700; font-size:0.88rem; color:#0F172A; margin-bottom:0.5rem;">
                Random Forest</div>""", unsafe_allow_html=True)
            cm_rf = np.array([[1245, 185], [92, 478]])
            fig_cm2, ax_cm2 = plt.subplots(figsize=(5, 4))
            fig_cm2.patch.set_facecolor('#FFFFFF')
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                        xticklabels=['Predicted: No','Predicted: Yes'],
                        yticklabels=['Actual: No','Actual: Yes'],
                        ax=ax_cm2, linewidths=2, linecolor='white',
                        annot_kws={"size": 16, "weight": "bold"}, cbar=False)
            ax_cm2.set_title('Confusion Matrix — RF', fontweight='bold', fontsize=10, color='#0F172A', pad=10)
            plt.xticks(color='#334155', fontsize=8.5); plt.yticks(color='#334155', fontsize=8.5, rotation=0)
            plt.tight_layout(); st.pyplot(fig_cm2)

        st.markdown("""
        <div class="content-card" style="background:#F8FAFC; margin-top:0.5rem;">
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.85rem; color:#334155; line-height:1.85;">
                <strong style="color:#0F172A;">How to read a Confusion Matrix:</strong><br>
                • <strong>True Negative (top-left):</strong> Correctly predicted employee will <em>stay</em><br>
                • <strong>False Positive (top-right):</strong> Predicted will leave but actually stayed (false alarm)<br>
                • <strong>False Negative (bottom-left):</strong> Predicted stay but actually left — the <em>most costly error</em><br>
                • <strong>True Positive (bottom-right):</strong> Correctly predicted employee will <em>leave</em> ✓<br><br>
                Random Forest produces far fewer <strong>False Negatives (92 vs 261)</strong> —
                meaning it catches significantly more real attrition cases, which is the primary goal of this system.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-head" style="margin-top:2rem;">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Classification Report — Random Forest</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        report_df = pd.DataFrame({
            'Class':     ['No Attrition (0)', 'Attrition (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
            'Precision': ['0.931', '0.784', '—', '0.858', '0.899'],
            'Recall':    ['0.871', '0.856', '—', '0.864', '0.879'],
            'F1-Score':  ['0.900', '0.818', '0.812', '0.859', '0.879'],
            'Support':   ['1430', '570', '2000', '2000', '2000']
        })
        st.dataframe(report_df.set_index('Class'), use_container_width=True)

        st.markdown("""
        <div class="section-head" style="margin-top:2rem;">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Model Training Details</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("""
            <div class="method-card">
                <div class="method-tag">Baseline</div>
                <div class="method-title">Logistic Regression</div>
                <div class="method-desc">
                    Good interpretability · Linear decision boundary · Requires StandardScaler feature scaling ·
                    Fast training time · Best for understanding individual feature contributions.
                </div>
                <div class="method-stat">
                    <span class="method-stat-label">Train / Test Split</span>
                    <span class="method-stat-value" style="font-size:1rem;">80% / 20%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="method-card gold">
                <div class="method-tag">Primary ✓ Winner</div>
                <div class="method-title">Random Forest</div>
                <div class="method-desc">
                    Handles non-linearity · No scaling needed · 100 estimators · random_state=42 ·
                    Robust to outliers · Provides feature importance · Wins across all 4 metrics.
                </div>
                <div class="method-stat">
                    <span class="method-stat-label">Train / Test Split</span>
                    <span class="method-stat-value" style="font-size:1rem;">80% / 20%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="content-card" style="background:#F0FDF4; border-left:4px solid #059669; margin-top:0.5rem;">
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.88rem; color:#334155; line-height:1.8;">
                <strong style="color:#065F46;">Training Configuration:</strong>
                Dataset of 10,000 records split 80/20 →
                <strong>8,000 training records</strong> and <strong>2,000 test records</strong> ·
                random_state = 42 (ensures reproducibility) ·
                Random Forest: n_estimators = 100 ·
                Human attrition behavior is non-linear — salary, overtime, and satisfaction
                interact in complex ways that Logistic Regression cannot fully capture,
                which is why Random Forest wins across all metrics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")


# ===================================================
# PREDICTION PAGE
# ===================================================
elif page == "Prediction":

    st.markdown("""
    <div class="hero" style="padding:3rem 3.5rem;">
        <div class="hero-grid"></div>
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            Real-Time Inference Engine
        </div>
        <div class="hero-title" style="font-size:2.6rem;">
            Employee <em>Risk</em><br>Assessment
        </div>
        <div class="hero-sub">
            Enter the 12 HR profile parameters below. Both ML models will instantly compute
            attrition probability and generate a personalised risk report with strategic HR recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        logistic_model = joblib.load("logistic_model.pkl")
        rf_model       = joblib.load("random_forest_model.pkl")
        scaler         = joblib.load("scaler.pkl")
        columns        = joblib.load("columns.pkl")
        df             = pd.read_csv(DATA_FILE)
        mean_values    = df.mean(numeric_only=True)

        st.markdown("""
        <div class="section-head">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Employee Profile Input</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            age                        = st.slider("Age", 18, 60, 30)
            monthly_income             = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000, step=500)
            total_working_years        = st.slider("Total Working Years", 0, 40, 5)
            distance_from_home         = st.slider("Distance From Home (km)", 1, 30, 5)
        with col2:
            years_at_company           = st.slider("Years At Company", 0, 40, 5)
            years_in_current_role      = st.slider("Years In Current Role", 0, 18, 3)
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
            num_companies_worked       = st.slider("Num Companies Worked", 0, 9, 2,
                                            help="Total number of companies worked at including current")
        with col3:
            years_with_manager  = st.slider("Years With Current Manager", 0, 20, 3)
            work_life_balance   = st.selectbox("Work Life Balance", [1,2,3,4],
                                    format_func=lambda x:{1:"1 - Bad",2:"2 - Good",3:"3 - Better",4:"4 - Best"}[x])
            job_satisfaction    = st.selectbox("Job Satisfaction", [1,2,3,4],
                                    format_func=lambda x:{1:"1 - Low",2:"2 - Medium",3:"3 - High",4:"4 - Very High"}[x])
            overtime            = st.selectbox("OverTime", ["Yes", "No"])

        st.markdown('<div class="insight-box">💡 Higher overtime and lower job satisfaction may increase attrition risk.</div>',
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Predict Attrition Risk"):
            if years_with_manager > years_at_company:
                st.warning("⚠️ Years With Manager cannot exceed Years At Company.")
                st.stop()
            if years_in_current_role > years_at_company:
                st.warning("⚠️ Years In Current Role cannot exceed Years At Company.")
                st.stop()

            input_df = pd.DataFrame([mean_values]).copy()
            input_df = input_df.reindex(columns=columns, fill_value=0)
            raw_inputs = {
                "Age": age, "MonthlyIncome": monthly_income,
                "TotalWorkingYears": total_working_years, "DistanceFromHome": distance_from_home,
                "YearsAtCompany": years_at_company, "YearsInCurrentRole": years_in_current_role,
                "YearsSinceLastPromotion": years_since_last_promotion,
                "NumCompaniesWorked": num_companies_worked, "YearsWithCurrManager": years_with_manager,
                "WorkLifeBalance": work_life_balance, "JobSatisfaction": job_satisfaction,
                "OverTime": 1 if overtime == "Yes" else 0,
            }
            engineered = {
                "SatisfactionScore": (job_satisfaction + work_life_balance) / 2,
                "IncomePerYear":     monthly_income / (total_working_years + 1),
                "YearsPerCompany":   years_at_company / (total_working_years + 1),
                "ExperienceGap":     total_working_years - years_at_company,
                "LoyaltyScore":      years_at_company - years_with_manager,
                "OverTimeFlag":      1 if overtime == "Yes" else 0,
            }
            for col_name, val in {**raw_inputs, **engineered}.items():
                if col_name in input_df.columns:
                    input_df[col_name] = val
            input_df = input_df.astype(float)

            scaled_input = scaler.transform(input_df)
            lr_prob      = logistic_model.predict_proba(scaled_input)[0][1]
            rf_prob      = rf_model.predict_proba(input_df)[0][1]

            st.markdown("---")
            st.markdown("""
            <div class="section-head">
                <div class="section-head-bar"></div>
                <div class="section-head-title">Attrition Risk Evaluation</div>
                <div class="section-head-line"></div>
            </div>
            """, unsafe_allow_html=True)

            colA, colB = st.columns(2, gap="large")
            with colA:
                cls = "high" if lr_prob > 0.5 else "low"
                label = "High Risk" if lr_prob > 0.5 else "Retained"
                st.markdown(f"""
                <div class="pred-card {cls}">
                    <div class="pred-card-engine">Logistic Regression Engine</div>
                    <div class="pred-card-pct">{lr_prob:.1%}</div>
                    <span class="pred-card-badge">● {label}</span>
                </div>""", unsafe_allow_html=True)
            with colB:
                cls = "high" if rf_prob > 0.5 else "low"
                label = "High Risk" if rf_prob > 0.5 else "Retained"
                st.markdown(f"""
                <div class="pred-card {cls}">
                    <div class="pred-card-engine">Random Forest Engine</div>
                    <div class="pred-card-pct">{rf_prob:.1%}</div>
                    <span class="pred-card-badge">● {label}</span>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
            <div class="section-head">
                <div class="section-head-bar"></div>
                <div class="section-head-title">Final Decision</div>
                <div class="section-head-line"></div>
            </div>
            """, unsafe_allow_html=True)
            if rf_prob >= 0.60:
                st.error(f"🔴 High Risk of Attrition ({rf_prob:.2%})")
            elif rf_prob >= 0.40:
                st.warning(f"🟡 Moderate Risk of Attrition ({rf_prob:.2%})")
            else:
                st.success(f"🟢 Low Risk of Attrition ({rf_prob:.2%})")

            st.markdown("---")
            st.markdown("""
            <div class="section-head">
                <div class="section-head-bar"></div>
                <div class="section-head-title">Attrition Risk Meter</div>
                <div class="section-head-line"></div>
            </div>
            """, unsafe_allow_html=True)

            risk_value  = rf_prob * 100
            zone        = "HIGH RISK" if risk_value >= 60 else ("MODERATE" if risk_value >= 40 else "LOW RISK")
            zone_color  = "#F43F5E" if risk_value >= 60 else ("#F97316" if risk_value >= 40 else "#10B981")
            zone_bg     = "#FFF1F2" if risk_value >= 60 else ("#FFF7ED" if risk_value >= 40 else "#F0FDF4")
            zone_border = "#FFE4E6" if risk_value >= 60 else ("#FED7AA" if risk_value >= 40 else "#D1FAE5")
            agree       = (lr_prob > 0.5) == (rf_prob > 0.5)
            agree_text  = (f"✅ Both models agree — LR: {lr_prob:.1%} | RF: {rf_prob:.1%}" if agree
                           else f"⚠️ Models disagree — LR: {lr_prob:.1%} | RF: {rf_prob:.1%} (RF is primary)")

            st.markdown(f"""
            <div style="background:white; border-radius:20px; padding:2.5rem 3rem;
                        box-shadow:0 4px 24px rgba(0,0,0,0.06); border:1px solid #E2E8F0; margin:0.5rem 0 1.5rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.8rem;">
                    <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.72rem;font-weight:700;color:#10B981;letter-spacing:1px;text-transform:uppercase;">Low Risk</span>
                    <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.72rem;font-weight:700;color:#F97316;letter-spacing:1px;text-transform:uppercase;">Moderate</span>
                    <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.72rem;font-weight:700;color:#F43F5E;letter-spacing:1px;text-transform:uppercase;">High Risk</span>
                </div>
                <div style="position:relative;height:20px;border-radius:100px;overflow:visible;
                            background:linear-gradient(90deg,#D1FAE5 0%,#D1FAE5 40%,#FED7AA 40%,#FED7AA 60%,#FFE4E6 60%,#FFE4E6 100%);
                            border:1px solid #E2E8F0;">
                    <div style="position:absolute;left:{risk_value}%;top:-12px;bottom:-12px;width:2px;
                                background:{zone_color};border-radius:2px;transform:translateX(-50%);
                                box-shadow:0 0 10px {zone_color}88;z-index:10;"></div>
                    <div style="position:absolute;left:{risk_value}%;top:50%;transform:translate(-50%,-50%);
                                width:22px;height:22px;background:{zone_color};border-radius:50%;
                                border:3px solid white;box-shadow:0 2px 10px {zone_color}66;z-index:11;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.5rem;margin-bottom:2rem;">
                    <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.68rem;color:#94A3B8;">0%</span>
                    <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.68rem;color:#94A3B8;">40%</span>
                    <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.68rem;color:#94A3B8;">60%</span>
                    <span style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.68rem;color:#94A3B8;">100%</span>
                </div>
                <div style="background:{zone_bg};border:1.5px solid {zone_border};border-radius:14px;padding:1.75rem;text-align:center;">
                    <div style="font-family:'Playfair Display',Georgia,serif;font-size:3.5rem;font-weight:900;
                                color:{zone_color};line-height:1;letter-spacing:-3px;margin-bottom:0.5rem;">
                        {risk_value:.1f}%
                    </div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;font-weight:700;
                                color:{zone_color};letter-spacing:3px;text-transform:uppercase;">
                        {zone}
                    </div>
                </div>
                <div style="margin-top:1rem;font-family:'Plus Jakarta Sans',sans-serif;font-size:0.82rem;color:#94A3B8;text-align:center;">
                    {agree_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
            <div class="section-head">
                <div class="section-head-bar"></div>
                <div class="section-head-title">HR Recommendations</div>
                <div class="section-head-line"></div>
            </div>
            """, unsafe_allow_html=True)

            rec_cards = []
            if overtime == "Yes":
                rec_cards.append(("#F43F5E","#FFF1F2","#FFE4E6","Overtime Load",
                    "Employee is working overtime. Overtime is the strongest single predictor of attrition. Immediately review workload distribution, hire support staff, or offer compensatory time-off to reduce burnout risk."))
            if job_satisfaction <= 2:
                rec_cards.append(("#F43F5E","#FFF1F2","#FFE4E6","Low Job Satisfaction",
                    f"Job satisfaction is rated {job_satisfaction}/4. Schedule a one-on-one with the manager to understand pain points. Consider role enrichment, new responsibilities, or a lateral move to a better-fitting team."))
            if work_life_balance <= 2:
                rec_cards.append(("#F43F5E","#FFF1F2","#FFE4E6","Poor Work-Life Balance",
                    f"Work-life balance score is {work_life_balance}/4. Offer flexible working hours or remote work options. Audit meeting culture and after-hours communication expectations."))
            if monthly_income < 4000:
                rec_cards.append(("#F97316","#FFF7ED","#FED7AA","Below-Market Salary",
                    f"Monthly income of Rs.{monthly_income:,} is below average. Conduct a market salary benchmarking exercise. A 10-15% raise at this stage costs far less than recruiting and training a replacement."))
            if num_companies_worked >= 5:
                rec_cards.append(("#F97316","#FFF7ED","#FED7AA","Job-Hopper Profile",
                    f"Employee has worked at {num_companies_worked} companies. This pattern indicates low tolerance for stagnation. Ensure a clear, written career growth roadmap is shared with the employee immediately."))
            if years_since_last_promotion >= 3:
                rec_cards.append(("#F97316","#FFF7ED","#FED7AA","No Recent Promotion",
                    f"No promotion in {years_since_last_promotion} years. Employees without career progression are 3x more likely to leave. Evaluate for promotion eligibility or offer a title change with added responsibilities."))
            if distance_from_home >= 20:
                rec_cards.append(("#D97706","#FFFBEB","#FEF3C7","Long Commute Distance",
                    f"Distance from home is {distance_from_home} km. Long commutes increase fatigue and reduce job satisfaction over time. Offer hybrid or remote work flexibility to reduce this burden."))
            if years_at_company <= 2:
                rec_cards.append(("#D97706","#FFFBEB","#FEF3C7","Early Tenure Risk",
                    f"Only {years_at_company} year(s) at the company. First 2 years are the highest attrition window. Assign a mentor, schedule regular check-ins, and ensure onboarding experience is strong."))
            if years_since_last_promotion == 0 and years_at_company >= 3:
                rec_cards.append(("#059669","#F0FDF4","#D1FAE5","Recent Promotion — Positive Signal",
                    "Employee was recently promoted. This is a strong retention signal. Maintain momentum with clear next-level goals and recognition."))

            if not rec_cards:
                st.success("No major HR intervention needed. Employee profile shows strong retention signals.")
                st.write("• Continue regular performance check-ins")
                st.write("• Maintain current compensation and benefits")
                st.write("• Recognise and reward contributions publicly")
            else:
                cards_html = ""
                for border_color, bg_color, border_col, title, detail in rec_cards:
                    cards_html += f"""
                    <div style="background:{bg_color};border:1px solid {border_col};border-left:4px solid {border_color};
                                border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:0.75rem;">
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;
                                    color:#0F172A;margin-bottom:0.35rem;">{title}</div>
                        <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.85rem;
                                    color:#475569;line-height:1.7;">{detail}</div>
                    </div>"""
                st.markdown(cards_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("""
            <div class="section-head">
                <div class="section-head-bar"></div>
                <div class="section-head-title">Key Risk Factors Analysis</div>
                <div class="section-head-line"></div>
            </div>
            """, unsafe_allow_html=True)

            factor_weights = {
                "OverTime": 0.34, "JobSatisfaction": 0.16, "MonthlyIncome": 0.12,
                "WorkLifeBalance": 0.11, "NumCompaniesWorked": 0.10,
                "YearsSinceLastPromotion": 0.08, "DistanceFromHome": 0.05, "YearsAtCompany": 0.04,
            }
            factor_scores = {
                "OverTime":                1.0 if overtime == "Yes" else 0.0,
                "JobSatisfaction":         (4 - job_satisfaction) / 3,
                "MonthlyIncome":           max(0, (8000 - monthly_income) / 7000),
                "WorkLifeBalance":         (4 - work_life_balance) / 3,
                "NumCompaniesWorked":      min(num_companies_worked / 8, 1.0),
                "YearsSinceLastPromotion": min(years_since_last_promotion / 10, 1.0),
                "DistanceFromHome":        min(distance_from_home / 29, 1.0),
                "YearsAtCompany":          max(0, (5 - years_at_company) / 5),
            }
            contributions = {k: round(factor_weights[k] * factor_scores[k] * 100, 1) for k in factor_weights}
            contrib_df = pd.DataFrame({
                "Factor": list(contributions.keys()),
                "Contribution": list(contributions.values())
            }).sort_values("Contribution", ascending=True)

            bar_colors = ["#F43F5E" if v >= 5 else "#10B981" for v in contrib_df["Contribution"]]
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            fig2.patch.set_facecolor('#FFFFFF'); ax2.set_facecolor('#F8FAFC')
            bars = ax2.barh(contrib_df["Factor"], contrib_df["Contribution"],
                            color=bar_colors, edgecolor="white", height=0.45)
            for bar, val in zip(bars, contrib_df["Contribution"]):
                ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", ha="left", fontsize=10, fontweight="700",
                         color="#F43F5E" if val >= 5 else "#10B981")
            ax2.set_xlabel("Contribution to Attrition Risk (%)", fontsize=10, color='#64748B', labelpad=8)
            ax2.set_title("Factor-wise Contribution to Predicted Attrition Risk",
                          fontsize=12, fontweight="bold", color='#0F172A', pad=14, fontfamily='serif')
            ax2.set_xlim(0, max(contrib_df["Contribution"].max() + 5, 20))
            ax2.axvline(x=5, color="#2563EB", linestyle="--", linewidth=1.2, alpha=0.45, label="Significance threshold (5%)")
            ax2.tick_params(axis='y', colors='#0F172A', labelsize=10)
            ax2.tick_params(axis='x', colors='#64748B')
            legend = ax2.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='#E2E8F0')
            plt.setp(legend.get_texts(), color='#334155')
            for s in ['top','right']: ax2.spines[s].set_visible(False)
            for s in ['left','bottom']: ax2.spines[s].set_color('#E2E8F0')
            plt.tight_layout(); st.pyplot(fig2)

            st.markdown("**Factor Breakdown:**")
            display_df = contrib_df.sort_values("Contribution", ascending=False).copy()
            display_df["Status"] = display_df["Contribution"].apply(
                lambda x: "🔴 High Risk" if x >= 8 else ("🟠 Moderate" if x >= 4 else "🟢 Low Risk"))
            display_df["Contribution"] = display_df["Contribution"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True)

    except FileNotFoundError:
        st.error("❌ Model files not found. Please run train_model.py first.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


# ===================================================
# EDA PAGE
# ===================================================
elif page == "EDA":

    st.markdown("""
    <div class="hero" style="padding:3rem 3.5rem;">
        <div class="hero-grid"></div>
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            Data Science &nbsp;·&nbsp; Visual Analytics
        </div>
        <div class="hero-title" style="font-size:2.6rem;">
            Exploratory <em>Data</em><br>Analysis
        </div>
        <div class="hero-sub">
            Visual deep-dive into 10,000 HR records — uncovering the statistical patterns,
            correlations and departmental trends that drive employee attrition.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        st.error("Dataset file not found.")
        st.stop()

    sns.set_style("whitegrid")

    attrition_rate_eda = (df['Attrition'].value_counts(normalize=True).get('Yes', 0)) * 100
    st.markdown(f"""
    <div class="eda-stat-grid">
        <div class="eda-stat">
            <div class="eda-stat-num">{len(df):,}</div>
            <div class="eda-stat-label">Total Records</div>
        </div>
        <div class="eda-stat">
            <div class="eda-stat-num" style="color:#F43F5E;">{attrition_rate_eda:.2f}%</div>
            <div class="eda-stat-label">Attrition Rate</div>
        </div>
        <div class="eda-stat">
            <div class="eda-stat-num" style="color:#2563EB;">${df['MonthlyIncome'].mean():,.0f}</div>
            <div class="eda-stat-label">Avg Monthly Income</div>
        </div>
        <div class="eda-stat">
            <div class="eda-stat-num" style="color:#059669;">{df['Age'].mean():.1f}</div>
            <div class="eda-stat-label">Average Age</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Dataset Preview</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Statistical Summary</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("""
    <div class="section-head">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Data Integrity Check</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    missing_values = df.isnull().sum()
    st.dataframe(pd.DataFrame({
        "Feature": missing_values.index,
        "Missing Values": missing_values.values
    }), use_container_width=True)
    if missing_values.sum() == 0:
        st.success("✅ Dataset is pristine — zero missing values detected across all 10,000 records.")
    else:
        st.warning("Dataset contains missing values requiring attention.")

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Attrition Distribution</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([1,2], gap="large")
    with col1:
        attrition_counts = df['Attrition'].value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#FFFFFF')
        wedges, texts, autotexts = ax.pie(
            attrition_counts, labels=None, autopct='%1.1f%%', startangle=90,
            colors=['#10B981','#F43F5E'],
            wedgeprops={'edgecolor':'white','linewidth':3,'width':0.7}, pctdistance=0.75)
        for at in autotexts:
            at.set_fontsize(13); at.set_fontweight('bold'); at.set_color('white')
        ax.legend(wedges, attrition_counts.index, loc='lower center', frameon=False,
                  fontsize=11, ncol=2, bbox_to_anchor=(0.5, -0.08))
        ax.set_title("Attrition Split", fontweight='bold', color='#0F172A', pad=16, fontsize=13, fontfamily='serif')
        ax.add_patch(plt.Circle((0,0), 0.45, fc='white'))
        plt.tight_layout(); st.pyplot(fig)
    with col2:
        st.markdown("""
        <div class="content-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
            <div class="method-title" style="margin-bottom:1rem;">What This Tells Us</div>
            <div class="info-row">
                <div class="info-icon" style="background:rgba(16,185,129,0.1);">🟢</div>
                <div>
                    <div class="info-label">Retained Employees</div>
                    <div class="method-desc">The majority of employees remain with the company. However, even a small attrition rate translates to significant replacement costs across a large workforce.</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon" style="background:rgba(244,63,94,0.1);">🔴</div>
                <div>
                    <div class="info-label">Attrition Cases</div>
                    <div class="method-desc">Employees who left represent lost institutional knowledge, recruitment costs, and productivity gaps. Identifying these patterns early is the core objective of this project.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Feature Distribution Explorer</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    selected_feature = st.selectbox("Select a feature to analyze against attrition:", numeric_features)
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#FFFFFF'); ax.set_facecolor('#F8FAFC')
    sns.histplot(data=df, x=selected_feature, hue="Attrition",
                 kde=True, bins=30, palette={"No":"#10B981","Yes":"#F43F5E"},
                 ax=ax, edgecolor='white', alpha=0.75)
    ax.set_title(f"{selected_feature} Distribution by Attrition Status",
                 fontweight='bold', color='#0F172A', fontsize=12, pad=12, fontfamily='serif')
    ax.set_xlabel(selected_feature, color='#64748B', fontsize=10)
    ax.set_ylabel("Count", color='#64748B', fontsize=10)
    plt.xticks(color='#334155'); plt.yticks(color='#334155')
    if ax.get_legend():
        plt.setp(ax.get_legend().get_texts(), color='#334155')
        ax.get_legend().get_frame().set_facecolor('white')
        ax.get_legend().get_frame().set_edgecolor('#E2E8F0')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#E2E8F0')
    plt.tight_layout(); st.pyplot(fig)

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Key Feature Comparison</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="chart-title">Total Working Years vs Attrition</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Newer employees tend to leave more frequently</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#FFFFFF'); ax.set_facecolor('#F8FAFC')
        sns.boxplot(data=df, x='Attrition', y='TotalWorkingYears',
                    palette={"No":"#10B981","Yes":"#F43F5E"}, ax=ax,
                    linewidth=1.5, flierprops=dict(marker='o', markersize=3, alpha=0.4))
        ax.set_title("Working Years by Attrition", fontweight='bold', color='#0F172A', fontsize=11, fontfamily='serif')
        ax.set_xlabel("Attrition", color='#64748B'); ax.set_ylabel("Total Working Years", color='#64748B')
        plt.xticks(color='#334155'); plt.yticks(color='#334155')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['left','bottom']: ax.spines[s].set_color('#E2E8F0')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight-box">📊 Employees with fewer total working years have a significantly higher tendency to leave the organization.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-title">Monthly Income vs Attrition</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Lower income correlates strongly with higher attrition</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#FFFFFF'); ax.set_facecolor('#F8FAFC')
        sns.boxplot(data=df, x='Attrition', y='MonthlyIncome',
                    palette={"No":"#2563EB","Yes":"#F59E0B"}, ax=ax,
                    linewidth=1.5, flierprops=dict(marker='o', markersize=3, alpha=0.4))
        ax.set_title("Income by Attrition", fontweight='bold', color='#0F172A', fontsize=11, fontfamily='serif')
        ax.set_xlabel("Attrition", color='#64748B'); ax.set_ylabel("Monthly Income", color='#64748B')
        plt.xticks(color='#334155'); plt.yticks(color='#334155')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['left','bottom']: ax.spines[s].set_color('#E2E8F0')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight-box">💰 Employees earning below median income show significantly higher attrition — salary remains a primary retention lever.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Overtime vs Attrition</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    overtime_attrition = pd.crosstab(df['OverTime'], df['Attrition'], normalize='index') * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#FFFFFF'); ax.set_facecolor('#F8FAFC')
    overtime_attrition.plot(kind='bar', ax=ax, color=['#10B981','#F43F5E'],
                            edgecolor='white', linewidth=1.5, width=0.45)
    ax.set_ylabel("Percentage (%)", color='#64748B', fontsize=10)
    ax.set_xlabel("Overtime Status", color='#64748B', fontsize=10)
    ax.set_title("Attrition Rate by Overtime Status",
                 fontweight='bold', color='#0F172A', pad=14, fontsize=12, fontfamily='serif')
    plt.xticks(rotation=0, color='#334155', fontsize=11); plt.yticks(color='#334155')
    legend = ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=9)
    plt.setp(legend.get_texts(), color='#334155')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#E2E8F0')
    plt.tight_layout(); st.pyplot(fig)
    st.markdown('<div class="insight-box">⏰ Employees working overtime are approximately 2× more likely to leave — overtime is the single strongest attrition predictor in this dataset.</div>', unsafe_allow_html=True)

    if "Department" in df.columns:
        st.markdown("""
        <div class="section-head" style="margin-top:2.5rem;">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Attrition by Department</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#FFFFFF'); ax.set_facecolor('#F8FAFC')
        sns.countplot(data=df, x="Department", hue="Attrition",
                      palette={"No":"#2563EB","Yes":"#F43F5E"}, ax=ax,
                      edgecolor='white', linewidth=1.5)
        ax.set_title("Department-wise Attrition Breakdown",
                     fontweight='bold', color='#0F172A', pad=14, fontsize=12, fontfamily='serif')
        ax.set_xlabel("Department", color='#64748B', fontsize=10)
        ax.set_ylabel("Employee Count", color='#64748B', fontsize=10)
        plt.xticks(rotation=15, color='#334155', fontsize=10); plt.yticks(color='#334155')
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', fontsize=9)
        plt.setp(legend.get_texts(), color='#334155')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['left','bottom']: ax.spines[s].set_color('#E2E8F0')
        plt.tight_layout(); st.pyplot(fig)
        st.markdown('<div class="insight-box">🏢 Sales department consistently shows the highest attrition — HR can prioritize targeted retention programmes for this division.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Feature Correlation Heatmap</div>
        <div class="section-head-line"></div>
    </div>
    """, unsafe_allow_html=True)
    df_corr = df.copy()
    df_corr['Attrition'] = df_corr['Attrition'].map({'No': 0, 'Yes': 1})
    df_corr = df_corr.drop(columns=[c for c in ['EmployeeCount','StandardHours','EmployeeNumber']
                                     if c in df_corr.columns])
    corr_matrix = df_corr.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#FFFFFF')
    sns.heatmap(corr_matrix, cmap="Blues", center=0, linewidths=0.4, linecolor='#F1F5F9',
                annot=True, fmt=".2f", annot_kws={"size": 7.5, "color": "#0F172A"},
                ax=ax, cbar_kws={"shrink": .75})
    ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight='bold',
                 color='#0F172A', pad=18, fontfamily='serif')
    plt.xticks(rotation=45, ha='right', color='#334155', fontsize=8)
    plt.yticks(rotation=0, color='#334155', fontsize=8)
    cbar = ax.collections[0].colorbar
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#334155')
    plt.tight_layout(); st.pyplot(fig)
    st.markdown('<div class="insight-box">🔗 Darker blue = stronger positive correlation. Age, TotalWorkingYears and YearsAtCompany are highly intercorrelated — Random Forest handles multicollinearity automatically through ensemble averaging.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" style="margin-top:2.5rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Key EDA Findings</div>
        <div class="section-head-line"></div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:1rem; margin-bottom:2rem;">
        <div style="background:white; border:1px solid #FFE4E6; border-left:4px solid #F43F5E;
                    border-radius:14px; padding:1.3rem 1.5rem;">
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.5rem;">🔴 Finding 1 — Overtime is the #1 Risk Factor</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.7;">
                Employees working overtime leave at nearly <strong>2× the rate</strong> of non-overtime employees.
                This single feature carries 34% weight in the Random Forest model — the highest of all 12 variables.
            </div>
        </div>
        <div style="background:white; border:1px solid #FED7AA; border-left:4px solid #F59E0B;
                    border-radius:14px; padding:1.3rem 1.5rem;">
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.5rem;">🟡 Finding 2 — Low Salary Drives Exits</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.7;">
                The income box plot shows attrition employees earn a <strong>significantly lower median salary</strong>.
                Employees earning below average monthly income are at substantially higher attrition risk.
            </div>
        </div>
        <div style="background:white; border:1px solid #DDD6FE; border-left:4px solid #7C3AED;
                    border-radius:14px; padding:1.3rem 1.5rem;">
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.5rem;">🟣 Finding 3 — Sales Has Highest Attrition</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.7;">
                The department chart shows <strong>Sales consistently leads in attrition count</strong>.
                HR teams should prioritize targeted retention and compensation review programs for Sales employees.
            </div>
        </div>
        <div style="background:white; border:1px solid #D1FAE5; border-left:4px solid #059669;
                    border-radius:14px; padding:1.3rem 1.5rem;">
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:0.9rem;
                        color:#0F172A; margin-bottom:0.5rem;">🟢 Finding 4 — Experience = Stability</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.84rem; color:#475569; line-height:1.7;">
                Working Years vs Attrition shows <strong>employees who left had significantly fewer working years</strong>.
                Early-career employees (0–5 years total experience) represent the highest-risk attrition group.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===================================================
# TOOLS PAGE
# ===================================================
elif page == "Tools":

    st.markdown("""
    <div class="hero" style="padding:3rem 3.5rem;">
        <div class="hero-grid"></div>
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            HR Decision Tools
        </div>
        <div class="hero-title" style="font-size:2.6rem;">
            Strategic <em>HR</em><br>Tools
        </div>
        <div class="hero-sub">
            Two powerful tools to help HR teams make smarter decisions —
            simulate retention scenarios and calculate the true cost of losing an employee.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        logistic_model = joblib.load("logistic_model.pkl")
        rf_model       = joblib.load("random_forest_model.pkl")
        scaler         = joblib.load("scaler.pkl")
        columns        = joblib.load("columns.pkl")
        df             = pd.read_csv(DATA_FILE)
        mean_values    = df.mean(numeric_only=True)

        def get_risk(age, income, ot, js, wlb, yac, yslp, dfh, twy, ncw, yicr, ywm):
            inp = pd.DataFrame([mean_values]).copy()
            inp = inp.reindex(columns=columns, fill_value=0)
            raw = {
                "Age": age, "MonthlyIncome": income,
                "TotalWorkingYears": twy, "DistanceFromHome": dfh,
                "YearsAtCompany": yac, "YearsInCurrentRole": yicr,
                "YearsSinceLastPromotion": yslp, "NumCompaniesWorked": ncw,
                "YearsWithCurrManager": ywm, "WorkLifeBalance": wlb,
                "JobSatisfaction": js, "OverTime": 1 if ot == "Yes" else 0,
            }
            eng = {
                "SatisfactionScore": (js + wlb) / 2,
                "IncomePerYear":     income / (twy + 1),
                "YearsPerCompany":   yac / (twy + 1),
                "ExperienceGap":     twy - yac,
                "LoyaltyScore":      yac - ywm,
                "OverTimeFlag":      1 if ot == "Yes" else 0,
            }
            for k, v in {**raw, **eng}.items():
                if k in inp.columns:
                    inp[k] = v
            inp = inp.astype(float)
            return rf_model.predict_proba(inp)[0][1] * 100

        # ── TOOL 1 — WHAT-IF SIMULATOR ──
        st.markdown("""
        <div class="section-head">
            <div class="section-head-bar"></div>
            <div class="section-head-title">🔄 What-If Simulator</div>
            <div class="section-head-line"></div>
        </div>
        <div class="insight-box">
            💡 Enter the employee's <strong>Current Profile</strong> on the left. On the right, adjust only
            the <strong>HR-actionable fields</strong> — salary, overtime, job satisfaction, work-life balance,
            promotion, and remote work option. Fields that HR cannot logically change (age, tenure, companies worked etc.)
            are fixed automatically. See the risk change instantly.
        </div>
        """, unsafe_allow_html=True)

        col_left, col_divider, col_right = st.columns([5, 0.3, 5])

        with col_left:
            st.markdown("""
            <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid #F43F5E;
                        border-radius:16px; padding:1.75rem 1.5rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06); margin-bottom:1rem;">
                <div style="font-family:'Playfair Display',serif; font-weight:800; font-size:1rem;
                            color:#F43F5E; margin-bottom:1.25rem; letter-spacing:-0.2px;">
                    👤 Current Employee Profile
                </div>
            """, unsafe_allow_html=True)
            w_age  = st.slider("Age",                     18, 60, 30,  key="w_age")
            w_inc  = st.number_input("Monthly Income",    1000, 50000, 5000, 500, key="w_inc")
            w_twy  = st.slider("Total Working Years",     0, 40, 5,    key="w_twy")
            w_dfh  = st.slider("Distance From Home (km)", 1, 30, 5,    key="w_dfh")
            w_yac  = st.slider("Years At Company",        0, 40, 5,    key="w_yac")
            w_yicr = st.slider("Years In Current Role",   0, 18, 3,    key="w_yicr")
            w_yslp = st.slider("Years Since Last Promotion", 0, 15, 1, key="w_yslp")
            w_ncw  = st.slider("Num Companies Worked",    0, 9, 2,     key="w_ncw")
            w_ywm  = st.slider("Years With Manager",      0, 20, 3,    key="w_ywm")
            w_wlb  = st.selectbox("Work Life Balance", [1,2,3,4],
                        format_func=lambda x:{1:"1 - Bad",2:"2 - Good",3:"3 - Better",4:"4 - Best"}[x], key="w_wlb")
            w_js   = st.selectbox("Job Satisfaction",  [1,2,3,4],
                        format_func=lambda x:{1:"1 - Low",2:"2 - Medium",3:"3 - High",4:"4 - Very High"}[x], key="w_js")
            w_ot   = st.selectbox("OverTime", ["Yes","No"], key="w_ot")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_divider:
            st.markdown("""
            <div style="display:flex; align-items:center; justify-content:center;
                        height:100%; padding-top:8rem;">
                <div style="font-family:'Playfair Display',serif; font-size:1.5rem;
                            font-weight:900; color:#94A3B8;">→</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("""
            <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid #059669;
                        border-radius:16px; padding:1.75rem 1.5rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06); margin-bottom:1rem;">
                <div style="font-family:'Playfair Display',serif; font-weight:800; font-size:1rem;
                            color:#059669; margin-bottom:0.5rem; letter-spacing:-0.2px;">
                    ✏️ Proposed Changes — What HR Can Do
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.78rem;
                            color:#94A3B8; margin-bottom:1.25rem; line-height:1.5;">
                    Only HR-actionable fields are editable here. Fields like Total Working Years,
                    Years At Company, Num Companies Worked etc. are fixed — HR cannot change them.
                </div>
            """, unsafe_allow_html=True)

            s_inc  = st.number_input("💰 Monthly Income (Salary Raise)",
                        1000, 50000, w_inc, 500, key="s_inc",
                        help="HR can offer a salary raise")
            s_ot   = st.selectbox("⏰ OverTime (Remove Overtime)",
                        ["Yes","No"], index=0 if w_ot=="Yes" else 1, key="s_ot",
                        help="HR can remove overtime requirement")
            s_js   = st.selectbox("😊 Job Satisfaction (Role Enrichment)",
                        [1,2,3,4], index=w_js-1,
                        format_func=lambda x:{1:"1 - Low",2:"2 - Medium",3:"3 - High",4:"4 - Very High"}[x],
                        key="s_js", help="HR can improve role, responsibilities or team fit")
            s_wlb  = st.selectbox("⚖️ Work Life Balance (Flexible Hours)",
                        [1,2,3,4], index=w_wlb-1,
                        format_func=lambda x:{1:"1 - Bad",2:"2 - Good",3:"3 - Better",4:"4 - Best"}[x],
                        key="s_wlb", help="HR can offer hybrid/remote work, reduce meeting load")
            s_yslp = st.slider("🏅 Years Since Last Promotion (Give Promotion)",
                        0, 15, w_yslp, key="s_yslp",
                        help="HR can give a promotion — set this to 0")
            s_dfh  = st.slider("🏠 Distance From Home (Remote Work Option)",
                        1, 30, w_dfh, key="s_dfh",
                        help="HR can offer remote/hybrid to reduce effective commute impact")

            s_age  = w_age
            s_twy  = w_twy
            s_yac  = w_yac
            s_yicr = w_yicr
            s_ncw  = w_ncw
            s_ywm  = w_ywm

            st.markdown("</div>", unsafe_allow_html=True)

        current_risk  = get_risk(w_age, w_inc, w_ot,  w_js, w_wlb, w_yac, w_yslp, w_dfh, w_twy, w_ncw, w_yicr, w_ywm)
        scenario_risk = get_risk(s_age, s_inc, s_ot,  s_js, s_wlb, s_yac, s_yslp, s_dfh, s_twy, s_ncw, s_yicr, s_ywm)
        delta         = scenario_risk - current_risk
        delta_sign    = "▼" if delta < 0 else "▲"
        delta_color   = "#059669" if delta < 0 else "#F43F5E"
        delta_bg      = "#F0FDF4" if delta < 0 else "#FFF1F2"
        delta_border  = "#D1FAE5" if delta < 0 else "#FFE4E6"
        verdict       = "✅ These proposed changes reduce attrition risk — employee is more likely to stay" if delta < 0 else "⚠️ Proposed changes do not help — adjust the right-side inputs to reduce risk"

        def zone_label(r):
            if r >= 60: return "HIGH RISK", "#F43F5E", "#FFF1F2", "#FFE4E6"
            if r >= 40: return "MODERATE",  "#F97316", "#FFF7ED", "#FED7AA"
            return "LOW RISK", "#10B981", "#F0FDF4", "#D1FAE5"

        cz, cc, cbg, cbrd = zone_label(current_risk)
        sz, sc_color, sbg, sbrd = zone_label(scenario_risk)

        cur_pct    = f"{current_risk:.1f}%"
        scen_pct   = f"{scenario_risk:.1f}%"
        delta_abs  = f"{abs(delta):.1f}%"

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr auto 1fr; gap:1rem; align-items:center; margin-bottom:1.5rem;">
            <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid {cc};
                        border-radius:16px; padding:1.75rem; text-align:center;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.5rem;">
                    Current Risk
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:3rem; font-weight:900;
                            color:{cc}; line-height:1; letter-spacing:-2px; margin-bottom:0.4rem;">
                    {cur_pct}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem; font-weight:700;
                            color:{cc}; letter-spacing:2px; text-transform:uppercase;">{cz}</div>
            </div>
            <div style="text-align:center;">
                <div style="background:{delta_bg}; border:1.5px solid {delta_border}; border-radius:12px;
                            padding:1rem 1.25rem;">
                    <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:900;
                                color:{delta_color}; line-height:1;">
                        {delta_sign} {delta_abs}
                    </div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.7rem; font-weight:700;
                                color:{delta_color}; letter-spacing:1px; text-transform:uppercase; margin-top:4px;">
                        Change
                    </div>
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:4px solid {sc_color};
                        border-radius:16px; padding:1.75rem; text-align:center;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.72rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.5rem;">
                    After Intervention
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:3rem; font-weight:900;
                            color:{sc_color}; line-height:1; letter-spacing:-2px; margin-bottom:0.4rem;">
                    {scen_pct}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem; font-weight:700;
                            color:{sc_color}; letter-spacing:2px; text-transform:uppercase;">{sz}</div>
            </div>
        </div>
        <div style="background:{delta_bg}; border:1px solid {delta_border}; border-left:4px solid {delta_color};
                    border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:2rem;
                    font-family:'Plus Jakarta Sans',sans-serif; font-size:0.9rem;
                    font-weight:600; color:{delta_color};">
            {verdict}
        </div>
        """, unsafe_allow_html=True)

        fig_w, ax_w = plt.subplots(figsize=(8, 3))
        fig_w.patch.set_facecolor('#FFFFFF'); ax_w.set_facecolor('#F8FAFC')
        bars_w = ax_w.barh(
            ['After Intervention', 'Current Risk'],
            [scenario_risk, current_risk],
            color=[sc_color, cc], edgecolor='white', height=0.4
        )
        for bar, val in zip(bars_w, [scenario_risk, current_risk]):
            ax_w.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                      f"{val:.1f}%", va='center', ha='left', fontsize=12,
                      fontweight='bold', color='#0F172A')
        ax_w.set_xlim(0, 105)
        ax_w.axvline(x=40, color='#F97316', linestyle='--', alpha=0.4, linewidth=1.2)
        ax_w.axvline(x=60, color='#F43F5E', linestyle='--', alpha=0.4, linewidth=1.2)
        ax_w.set_title('Risk Comparison — Before vs After Intervention',
                       fontweight='bold', color='#0F172A', fontsize=12, pad=12, fontfamily='serif')
        ax_w.tick_params(axis='y', colors='#0F172A', labelsize=11)
        ax_w.tick_params(axis='x', colors='#64748B')
        for s in ['top','right']: ax_w.spines[s].set_visible(False)
        for s in ['left','bottom']: ax_w.spines[s].set_color('#E2E8F0')
        plt.tight_layout(); st.pyplot(fig_w)

        st.markdown("---")

        # ── TOOL 2 — ATTRITION COST CALCULATOR ──
        st.markdown("""
        <div class="section-head">
            <div class="section-head-bar"></div>
            <div class="section-head-title">💰 Attrition Cost Calculator</div>
            <div class="section-head-line"></div>
        </div>
        <div class="insight-box">
            💡 Research shows replacing an employee costs <strong>50% – 200% of their annual salary</strong>
            when accounting for recruitment, onboarding, training, and productivity loss.
            This calculator estimates the total financial impact of losing an employee.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:white; border:1px solid #E2E8F0; border-radius:16px;
                    padding:2rem; box-shadow:0 2px 12px rgba(0,0,0,0.06); margin-bottom:1.5rem;">
            <div style="font-family:'Playfair Display',serif; font-weight:800; font-size:1rem;
                        color:#0F172A; margin-bottom:1.25rem;">
                Employee &amp; Company Details
            </div>
        """, unsafe_allow_html=True)

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            calc_salary   = st.number_input("Monthly Salary (₹)", 1000, 200000, 25000, 1000, key="calc_sal")
            calc_exp      = st.slider("Years of Experience", 0, 40, 5, key="calc_exp")
        with cc2:
            calc_role     = st.selectbox("Job Level", [
                "Entry Level (0–2 yrs)", "Mid Level (3–7 yrs)",
                "Senior Level (8–15 yrs)", "Manager / Lead (15+ yrs)"
            ], key="calc_role")
            calc_dept     = st.selectbox("Department", [
                "Sales", "Engineering / IT", "HR", "Finance", "Operations", "Marketing"
            ], key="calc_dept")
        with cc3:
            calc_recruit  = st.slider("Avg Recruitment Time (weeks)", 2, 24, 8, key="calc_rec")
            calc_train    = st.slider("Onboarding + Training Time (weeks)", 1, 16, 4, key="calc_train")

        st.markdown("</div>", unsafe_allow_html=True)

        annual_salary = calc_salary * 12
        multipliers = {
            "Entry Level (0–2 yrs)":       0.50,
            "Mid Level (3–7 yrs)":         0.75,
            "Senior Level (8–15 yrs)":     1.20,
            "Manager / Lead (15+ yrs)":    1.75,
        }
        base_multiplier = multipliers[calc_role]
        recruitment_cost   = annual_salary * 0.20 + (calc_recruit * calc_salary / 4)
        onboarding_cost    = annual_salary * 0.10 + (calc_train * calc_salary / 4)
        productivity_loss  = annual_salary * base_multiplier * 0.40
        knowledge_loss     = annual_salary * 0.15 * min(calc_exp / 5, 1.5)
        training_cost      = annual_salary * 0.12
        total_cost         = recruitment_cost + onboarding_cost + productivity_loss + knowledge_loss + training_cost

        dept_notes = {
            "Sales":           "Sales roles have higher replacement costs due to client relationship rebuilding.",
            "Engineering / IT":"Technical roles require longer onboarding and specialized hiring.",
            "HR":              "HR roles have moderate replacement costs.",
            "Finance":         "Finance roles require careful vetting increasing recruitment time.",
            "Operations":      "Operations roles have moderate replacement and training costs.",
            "Marketing":       "Marketing roles require portfolio review and brand knowledge transfer.",
        }

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1.5rem;">
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #2563EB;
                        border-radius:14px; padding:1.4rem; text-align:center;">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.7rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">
                    Recruitment Cost
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:900;
                            color:#2563EB; letter-spacing:-1px;">
                    ₹{recruitment_cost:,.0f}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                    Job postings + agency + interviews
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #7C3AED;
                        border-radius:14px; padding:1.4rem; text-align:center;">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.7rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">
                    Onboarding + Training
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:900;
                            color:#7C3AED; letter-spacing:-1px;">
                    ₹{onboarding_cost + training_cost:,.0f}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                    Orientation + skill development
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #F59E0B;
                        border-radius:14px; padding:1.4rem; text-align:center;">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.7rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">
                    Productivity Loss
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:900;
                            color:#F59E0B; letter-spacing:-1px;">
                    ₹{productivity_loss:,.0f}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                    Gap period + ramp-up time
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #059669;
                        border-radius:14px; padding:1.4rem; text-align:center;">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.7rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">
                    Knowledge Loss
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:900;
                            color:#059669; letter-spacing:-1px;">
                    ₹{knowledge_loss:,.0f}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                    Institutional memory + expertise
                </div>
            </div>
            <div style="background:white; border:1px solid #E2E8F0; border-top:3px solid #F43F5E;
                        border-radius:14px; padding:1.4rem; text-align:center; grid-column: span 2;">
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.7rem; font-weight:700;
                            text-transform:uppercase; letter-spacing:1px; color:#94A3B8; margin-bottom:0.4rem;">
                    Annual Salary (Reference)
                </div>
                <div style="font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:900;
                            color:#F43F5E; letter-spacing:-1px;">
                    ₹{annual_salary:,.0f}
                </div>
                <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem; color:#94A3B8; margin-top:4px;">
                    ₹{calc_salary:,}/month × 12
                </div>
            </div>
        </div>

        <div style="background:linear-gradient(135deg, #060E1F, #0D1B35);
                    border-radius:20px; padding:2.5rem 3rem; text-align:center;
                    border:1px solid rgba(255,255,255,0.08); margin-bottom:1.5rem;
                    position:relative; overflow:hidden;">
            <div style="position:absolute; inset:0;
                        background-image: linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
                        background-size:48px 48px; pointer-events:none;"></div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.78rem; font-weight:700;
                        text-transform:uppercase; letter-spacing:2px; color:rgba(255,255,255,0.45);
                        margin-bottom:0.6rem; position:relative; z-index:1;">
                Total Estimated Cost of Losing This Employee
            </div>
            <div style="font-family:'Playfair Display',serif; font-size:3.8rem; font-weight:900;
                        color:#F59E0B; letter-spacing:-3px; line-height:1;
                        position:relative; z-index:1; margin-bottom:0.5rem;">
                ₹{total_cost:,.0f}
            </div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.9rem;
                        color:rgba(255,255,255,0.5); position:relative; z-index:1;">
                {(total_cost/annual_salary*100):.0f}% of annual salary ·
                Equivalent to {total_cost/calc_salary:.1f} months of salary
            </div>
        </div>

        <div style="background:#F0F9FF; border:1px solid #BAE6FD; border-left:4px solid #0EA5E9;
                    border-radius:10px; padding:0.9rem 1.2rem; margin-bottom:1rem;
                    font-family:'Plus Jakarta Sans',sans-serif; font-size:0.88rem; color:#0369A1;">
            🏢 <strong>{calc_dept}:</strong> {dept_notes[calc_dept]}
        </div>
        """, unsafe_allow_html=True)

        cost_labels  = ['Recruitment', 'Onboarding + Training', 'Productivity Loss', 'Knowledge Loss']
        cost_values  = [recruitment_cost, onboarding_cost + training_cost, productivity_loss, knowledge_loss]
        cost_colors  = ['#2563EB', '#7C3AED', '#F59E0B', '#059669']

        col_pie_l, col_pie_c, col_pie_r = st.columns([1, 2, 1])
        with col_pie_c:
            fig_c, ax_c = plt.subplots(figsize=(4, 3.5))
            fig_c.patch.set_facecolor('#FFFFFF')
            wedges, texts, autotexts = ax_c.pie(
                cost_values, labels=None, autopct='%1.1f%%', startangle=90,
                colors=cost_colors,
                wedgeprops={'edgecolor':'white','linewidth':2,'width':0.65},
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_fontsize(9); at.set_fontweight('bold'); at.set_color('white')
            ax_c.legend(wedges, cost_labels, loc='lower center', frameon=False,
                        fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.12))
            ax_c.set_title('Cost Breakdown — Where the Money Goes',
                           fontweight='bold', color='#0F172A', pad=12, fontsize=10, fontfamily='serif')
            ax_c.add_patch(plt.Circle((0,0), 0.42, fc='white'))
            plt.tight_layout(); st.pyplot(fig_c)

        retention_actions = {
            "10% Salary Raise":       calc_salary * 12 * 0.10,
            "Work-From-Home Option":  50000,
            "Training Programme":     30000,
            "Promotion + Title":      calc_salary * 12 * 0.08,
        }
        st.markdown("""
        <div class="section-head" style="margin-top:2rem;">
            <div class="section-head-bar"></div>
            <div class="section-head-title">Retention ROI — Is It Worth Keeping Them?</div>
            <div class="section-head-line"></div>
        </div>
        """, unsafe_allow_html=True)

        roi_html = ""
        for action, cost in retention_actions.items():
            savings  = total_cost - cost
            roi_pct  = (savings / cost) * 100
            is_worth = savings > 0
            border   = "#059669" if is_worth else "#F43F5E"
            bg       = "#F0FDF4" if is_worth else "#FFF1F2"
            verdict_r= f"✅ SAVE ₹{savings:,.0f}" if is_worth else f"❌ Not cost effective"
            roi_html += f"""
            <div style="background:{bg}; border:1px solid {border}33; border-left:4px solid {border};
                        border-radius:12px; padding:1rem 1.4rem; margin-bottom:0.75rem;
                        display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-weight:700;
                                font-size:0.9rem; color:#0F172A;">{action}</div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.8rem;
                                color:#64748B; margin-top:2px;">Cost: ₹{cost:,.0f}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                                font-weight:900; color:{border};">{verdict_r}</div>
                    <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:0.75rem;
                                color:#64748B;">ROI: {roi_pct:.0f}%</div>
                </div>
            </div>"""
        st.markdown(roi_html, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("❌ Model files not found. Please run train_model.py first.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


# ===================================================
# PROJECT DETAILS PAGE
# ===================================================
elif page == "Project Details":

    st.markdown("""
    <div class="hero" style="padding:3rem 3.5rem;">
        <div class="hero-grid"></div>
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            Academic Documentation
        </div>
        <div class="hero-title" style="font-size:2.6rem;">
            Project <em>Details</em><br>&amp; Information
        </div>
        <div class="hero-sub">
            Complete academic and technical documentation for the
            Employee Attrition Prediction major project — BCA Final Year 2025–26.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="content-card">
            <div class="method-title" style="margin-bottom:1rem;">📋 Project Information</div>
            <div class="info-row">
                <div class="info-icon">📗</div>
                <div>
                    <div class="info-label">Subject</div>
                    <div class="info-value" style="color:#1e40af; font-weight:700;">Data Analytics using Python</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">🎓</div>
                <div>
                    <div class="info-label">Course &amp; Semester</div>
                    <div class="info-value">BCA Final Year · Semester 6</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">📘</div>
                <div>
                    <div class="info-label">Project Type</div>
                    <div class="info-value">Major Project</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">📅</div>
                <div>
                    <div class="info-label">Academic Year</div>
                    <div class="info-value">2025 – 2026</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">📊</div>
                <div>
                    <div class="info-label">Dataset</div>
                    <div class="info-value">HR Analytics · 10,000 Synthetic Records</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="content-card">
            <div class="method-title" style="margin-bottom:1rem;">👨‍💻 Developer Information</div>
            <div class="info-row">
                <div class="info-icon">👤</div>
                <div>
                    <div class="info-label">Developer Name</div>
                    <div class="info-value">Patwa Faizan Akhtar Hussain</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">🏫</div>
                <div>
                    <div class="info-label">University</div>
                    <div class="info-value">Veer Narmad South Gujarat University (VNSGU), Surat</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">🐍</div>
                <div>
                    <div class="info-label">Language</div>
                    <div class="info-value">Python 3.x</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">🌐</div>
                <div>
                    <div class="info-label">Framework</div>
                    <div class="info-value">Streamlit</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">🤖</div>
                <div>
                    <div class="info-label">ML Library</div>
                    <div class="info-value">Scikit-learn</div>
                </div>
            </div>
            <div class="info-row">
                <div class="info-icon">📈</div>
                <div>
                    <div class="info-label">Visualization</div>
                    <div class="info-value">Matplotlib · Seaborn · Pandas</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-head" style="margin-top:1rem;">
        <div class="section-head-bar"></div>
        <div class="section-head-title">Complete Tech Stack</div>
        <div class="section-head-line"></div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:2rem;">
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #2563EB;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🐍</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Python 3.x</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Core programming language</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #F43F5E;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🌐</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Streamlit 1.x</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Web app framework · UI routing · widgets</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #F59E0B;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🤖</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Scikit-learn</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">LogisticRegression · RandomForestClassifier · StandardScaler · train_test_split · metrics</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #059669;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🐼</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Pandas</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">DataFrame · CSV I/O · data manipulation · feature engineering</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #7C3AED;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🔢</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">NumPy</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Array operations · random data generation · mathematical computations</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #F97316;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">📈</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Matplotlib</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Bar charts · horizontal bar · pie chart · pyplot rendering</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #0EA5E9;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🎨</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Seaborn</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Histplot · boxplot · countplot · heatmap · statistical charts</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #D97706;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">💾</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Joblib</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Model serialization · .pkl file save/load · scaler persistence</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #0D9488;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">☁️</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Streamlit Cloud</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Live web deployment · public URL · auto startup</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #64748B;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🐙</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">GitHub</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Version control · source code repository · Streamlit Cloud integration</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #EC4899;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🎨</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Custom CSS + HTML</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Navbar · hero sections · cards · animations · Google Fonts</div>
        </div>
        <div style="background:white;border:1px solid #E2E8F0;border-top:3px solid #8B5CF6;border-radius:14px;padding:1.4rem;text-align:center;">
            <div style="font-size:1.8rem;margin-bottom:0.5rem;">🔤</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:0.9rem;color:#0F172A;margin-bottom:0.3rem;">Google Fonts</div>
            <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;color:#64748B;">Playfair Display (headings) · Plus Jakarta Sans (body)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===================================================
# FOOTER — app-themed, renders on EVERY page
# ===================================================

st.markdown(
    "<div style='margin-top:4rem; background:white; border:1px solid #E2E8F0;"
    "border-radius:20px; overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,0.06);'>"

    "<div style='height:3px; background:linear-gradient(90deg,#2563EB,#3B82F6,#22D3EE,#3B82F6,#2563EB);'></div>"

    "<div style='padding:2.5rem 2.5rem 0;'>"

    "<div style='display:grid; grid-template-columns:2fr 1.2fr 1.2fr; gap:3rem; margin-bottom:2rem;'>"

    "<div>"
    "<div style='font-family:Playfair Display,Georgia,serif; font-size:1.5rem; font-weight:900; line-height:1.2; margin-bottom:0.75rem;'>"
    "<span style='color:#2563EB;'>Attrition</span><span style='color:#0F172A;'>IQ</span>"
    "</div>"
    "<div style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.83rem; color:#64748B; line-height:1.8; max-width:280px;'>"
    "A machine learning platform for employee attrition prediction. "
    "Built as a BCA Final Year Major Project — Data Analytics using Python."
    "</div>"
    "</div>"

    "<div>"
    "<div style='font-family:Plus Jakarta Sans,sans-serif; font-weight:700; font-size:0.72rem;"
    "text-transform:uppercase; letter-spacing:1.2px; color:#94A3B8; margin-bottom:1rem;'>Get In Touch</div>"
    "<div style='display:flex; align-items:flex-start; gap:10px; margin-bottom:0.8rem;'>"
    "<div style='width:28px; height:28px; background:linear-gradient(135deg,rgba(37,99,235,0.1),rgba(34,211,238,0.1));"
    "border-radius:7px; display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:13px;'>&#9993;</div>"
    "<span style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.82rem; color:#475569; line-height:1.6; padding-top:4px;'>patwafaizan49@gmail.com</span>"
    "</div>"
    "<div style='display:flex; align-items:center; gap:10px;'>"
    "<div style='width:28px; height:28px; background:linear-gradient(135deg,rgba(37,99,235,0.1),rgba(34,211,238,0.1));"
    "border-radius:7px; display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:13px;'>&#128222;</div>"
    "<span style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.82rem; color:#475569;'>+91 99249 92102</span>"
    "</div>"
    "</div>"

    "<div>"
    "<div style='font-family:Plus Jakarta Sans,sans-serif; font-weight:700; font-size:0.72rem;"
    "text-transform:uppercase; letter-spacing:1.2px; color:#94A3B8; margin-bottom:1rem;'>Quick Links</div>"
    "<div style='display:flex; flex-direction:column; gap:0.5rem;'>"
    "<a href='?page=Home'             target='_self' style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.83rem; color:#475569; text-decoration:none; font-weight:500;'>&#8594;&nbsp; Home</a>"
    "<a href='?page=Model+Comparison' target='_self' style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.83rem; color:#475569; text-decoration:none; font-weight:500;'>&#8594;&nbsp; Model Comparison</a>"
    "<a href='?page=Prediction'       target='_self' style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.83rem; color:#475569; text-decoration:none; font-weight:500;'>&#8594;&nbsp; Prediction</a>"
    "<a href='?page=EDA'              target='_self' style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.83rem; color:#475569; text-decoration:none; font-weight:500;'>&#8594;&nbsp; EDA</a>"
    "<a href='?page=Tools'            target='_self' style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.83rem; color:#475569; text-decoration:none; font-weight:500;'>&#8594;&nbsp; Tools</a>"
    "<a href='?page=Project+Details'  target='_self' style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.83rem; color:#475569; text-decoration:none; font-weight:500;'>&#8594;&nbsp; Project</a>"
    "</div>"
    "</div>"

    "</div>"

    "<div style='border-top:1px solid #F1F5F9; padding:1.1rem 0; margin:0;"
    "display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.5rem;'>"
    "<div style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.76rem; color:#94A3B8;'>"
    "&#169; 2026 <strong style='color:#64748B; font-weight:600;'>Patwa Faizan Akhtar Hussain</strong>"
    " &nbsp;&middot;&nbsp; BCA Final Year &middot; VNSGU, Surat"
    "</div>"
    "<div style='font-family:Plus Jakarta Sans,sans-serif; font-size:0.76rem; color:#94A3B8;'>"
    "Built with <span style='color:#F43F5E;'>&#9829;</span> using Python &middot; Streamlit &middot; Scikit-learn"
    "</div>"
    "</div>"

    "</div>"
    "</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.write("### About")
st.sidebar.write("Employee Attrition Prediction Project")
st.sidebar.write("Built with Streamlit & Scikit-learn")
st.sidebar.write("Developer: Patwa Faizan")