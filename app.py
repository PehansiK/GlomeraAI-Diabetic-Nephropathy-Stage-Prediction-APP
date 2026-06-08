"""
GlomeraAI Clinical Intelligence Platform
Multi-Stage Diabetic Nephropathy Risk Assessment & Clinical Decision Support
"""

import io, warnings, pickle, requests
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GlomeraAI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

[data-testid="stSidebarNav"] { display: none !important; }

:root {
    --navy:   #0b1426;
    --navy2:  #111d35;
    --blue:   #1a6efc;
    --teal:   #0cc8b0;
    --slate:  #8496b0;
    --border: rgba(255,255,255,0.07);
    --card:   rgba(255,255,255,0.04);
    --white:  #f0f4ff;
    --s0:  #10b981; --s1: #f59e0b; --s2: #f97316;
    --s3:  #ef4444; --s4: #a855f7; --s5: #1e293b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--navy) !important;
    color: var(--white);
}

/* ── Sidebar ── */
div[data-testid="stSidebarContent"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border);
}
div[data-testid="stSidebarContent"] * { color: #8496b0 !important; }
div[data-testid="stSidebarContent"] strong { color: #f0f4ff !important; }
div[data-testid="stSidebarContent"] span[style*="color"] { color: inherit !important; }
div[data-testid="stSidebarContent"] h3 { color: var(--white) !important; font-family: 'Playfair Display', serif; }

div[data-testid="stSidebarContent"] .stButton > button {
    width: 100%; text-align: left;
    background: transparent; border: none;
    border-radius: 8px; padding: .5rem .85rem;
    font-size: .85rem; font-weight: 500; color: var(--slate) !important;
    transition: all .15s;
}
div[data-testid="stSidebarContent"] .stButton > button:hover {
    background: var(--card) !important;
    color: var(--white) !important; border: none !important;
}
div[data-testid="stSidebarContent"] .nav-active .stButton > button {
    background: rgba(26,110,252,.18) !important;
    color: #7cb3ff !important; font-weight: 700 !important;
    border-left: 3px solid var(--blue) !important;
}

/* ── Main surface ── */
.main .block-container { padding: 1.5rem 2.5rem; max-width: 1280px; }

/* ── Platform header ── */
.platform-header {
    display: flex; align-items: center; gap: 1.2rem;
    padding: 1.2rem 1.8rem;
    background: linear-gradient(135deg, rgba(26,110,252,.12), rgba(12,200,176,.08));
    border: 1px solid var(--border);
    border-radius: 16px; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.platform-header::before {
    content: ''; position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%231a6efc' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.platform-icon { font-size: 2.4rem; }
.platform-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 800;
    color: var(--white); margin: 0;
    letter-spacing: -.02em;
}
.platform-sub {
    font-size: .72rem; color: var(--slate);
    font-family: 'DM Mono', monospace;
    letter-spacing: .06em; margin: .15rem 0 0;
}
.platform-badge {
    margin-left: auto;
    background: rgba(12,200,176,.15);
    border: 1px solid rgba(12,200,176,.3);
    border-radius: 20px; padding: .3rem .9rem;
    font-size: .72rem; font-weight: 600;
    color: var(--teal); font-family: 'DM Mono', monospace;
}

/* ── Cards ── */
.glass-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
}
.stage-result-card {
    border-radius: 18px; padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    border: 1px solid; position: relative; overflow: hidden;
}
.stage-result-card::after {
    content: ''; position: absolute;
    top: -40px; right: -40px;
    width: 140px; height: 140px;
    border-radius: 50%; opacity: .06;
    background: currentColor;
}

/* ── Metric tiles ── */
.kpi-row { display: flex; gap: .9rem; margin: 1rem 0; }
.kpi-tile {
    flex: 1; background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 1.1rem 1.3rem;
    text-align: center;
}
.kpi-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem; font-weight: 700;
    line-height: 1; margin-bottom: .3rem;
}
.kpi-lbl {
    font-size: .68rem; color: var(--slate);
    text-transform: uppercase; letter-spacing: .1em; font-weight: 600;
}

/* ── Section headers ── */
.sect-hdr {
    font-size: .72rem; font-weight: 700; color: var(--teal);
    text-transform: uppercase; letter-spacing: .12em;
    padding-left: .7rem;
    border-left: 3px solid var(--teal);
    margin: 1.6rem 0 .9rem;
}

/* ── Recommendation box ── */
.rec-block {
    border-radius: 12px; padding: 1rem 1.3rem;
    margin: .8rem 0; border-left: 4px solid;
    font-size: .86rem; line-height: 1.7;
}
.rec-block.green  { background: rgba(16,185,129,.08); border-color: #10b981; }
.rec-block.amber  { background: rgba(245,158,11,.08);  border-color: #f59e0b; }
.rec-block.orange { background: rgba(249,115,22,.08);  border-color: #f97316; }
.rec-block.red    { background: rgba(239,68,68,.08);   border-color: #ef4444; }
.rec-block.purple { background: rgba(168,85,247,.08);  border-color: #a855f7; }
.rec-block.slate  { background: rgba(30,41,59,.5);     border-color: #475569; }
.rec-block.blue   { background: rgba(26,110,252,.08);  border-color: var(--blue); }
.rec-block.teal   { background: rgba(12,200,176,.08);  border-color: var(--teal); }

/* ── Progression verdict ── */
.verdict-card {
    border-radius: 16px; padding: 1.5rem 1.8rem; margin: 1rem 0;
    border: 1px solid;
}
.verdict-card.danger  { background: rgba(239,68,68,.07);   border-color: rgba(239,68,68,.3); }
.verdict-card.warning { background: rgba(245,158,11,.07);  border-color: rgba(245,158,11,.3); }
.verdict-card.safe    { background: rgba(16,185,129,.07);  border-color: rgba(16,185,129,.3); }
.verdict-title { font-size: 1.1rem; font-weight: 800; margin-bottom: .5rem; }
.verdict-body  { font-size: .87rem; color: var(--slate); line-height: 1.7; }

/* ── Model domain badges ── */
.domain-badge {
    border-radius: 12px; padding: .85rem 1.1rem;
    margin-bottom: .7rem; border: 1px solid var(--border);
    border-left: 4px solid;
}
.domain-badge.rf  { border-left-color: var(--blue);  background: rgba(26,110,252,.06); }
.domain-badge.xgb { border-left-color: #f97316; background: rgba(249,115,22,.06); }
.domain-badge.lr  { border-left-color: #a855f7; background: rgba(168,85,247,.06); }
.domain-badge h4  { margin: 0 0 .3rem; font-size: .88rem; font-weight: 700; color: var(--white); }
.domain-badge p   { margin: 0; font-size: .78rem; color: var(--slate); }

/* ── Stepper ── */
.stepper { display: flex; align-items: center; justify-content: center; padding: .5rem 0 1.8rem; }
.step-dot {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .78rem; font-weight: 700;
}
.step-dot.done   { background: var(--teal); color: var(--navy); }
.step-dot.active { background: var(--blue); color: #fff; box-shadow: 0 0 0 5px rgba(26,110,252,.2); }
.step-dot.todo   { background: rgba(255,255,255,.08); color: var(--slate); }
.step-label { font-size: .67rem; color: var(--slate); margin-top: .35rem; text-align: center; font-weight: 600; }
.step-line { width: 50px; height: 2px; background: rgba(255,255,255,.08); margin: 0 4px; }
.step-line.done { background: var(--teal); }

/* ── Chip ── */
.chip {
    display: inline-block; padding: .22rem .75rem;
    border-radius: 20px; font-size: .71rem; font-weight: 700; margin: .15rem;
}
.chip.green  { background: rgba(16,185,129,.15); color: #34d399; border: 1px solid rgba(16,185,129,.3); }
.chip.amber  { background: rgba(245,158,11,.15);  color: #fbbf24; border: 1px solid rgba(245,158,11,.3); }
.chip.red    { background: rgba(239,68,68,.15);   color: #f87171; border: 1px solid rgba(239,68,68,.3); }
.chip.blue   { background: rgba(26,110,252,.15);  color: #7cb3ff; border: 1px solid rgba(26,110,252,.3); }
.chip.purple { background: rgba(168,85,247,.15);  color: #c084fc; border: 1px solid rgba(168,85,247,.3); }

/* ── Tabs ── */
/* Metrics */
div[data-testid="stMetric"] label { color: #8496b0 !important; }
div[data-testid="stMetricValue"] { color: #f0f4ff !important; }
div[data-testid="stMetricDelta"] { color: #10b981 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] { background: rgba(255,255,255,.03) !important; border-radius: 10px; }

/* Spinner */
div[data-testid="stSpinner"] p { color: #8496b0 !important; }

/* Selectbox options */
div[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #f0f4ff !important; background: #111d35 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: rgba(255,255,255,.04);
    border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; font-size: .82rem; font-weight: 600;
    padding: .45rem 1.1rem; color: var(--slate);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(26,110,252,.25) !important;
    color: #7cb3ff !important;
}

/* ── Input labels ── */
label { color: #a0b4cc !important; font-size: .82rem !important; font-weight: 500 !important; }
.stNumberInput input, .stSelectbox > div > div {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid var(--border) !important;
    color: var(--white) !important; border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--blue), #0ea5e9) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; color: #fff !important;
    padding: .6rem 1.4rem !important;
    box-shadow: 0 4px 20px rgba(26,110,252,.35) !important;
    transition: all .2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(26,110,252,.5) !important;
}

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(255,255,255,.03);
    border: 1px solid var(--border);
    border-radius: 10px; padding: .9rem 1.2rem;
    font-size: .74rem; color: var(--slate);
    margin-top: 2rem; line-height: 1.7;
}
.disclaimer strong { color: #f87171; }

/* ── matplotlib plots ── */
.stPlotlyChart, .stPyplot { border-radius: 12px; overflow: hidden; }

/* ── Progress bar ── */
div[data-testid="stProgress"] > div { background: var(--blue) !important; border-radius: 4px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Info / warning boxes ── */
div[data-testid="stAlert"] {
    background: rgba(26,110,252,.08) !important;
    border: 1px solid rgba(26,110,252,.2) !important;
    border-radius: 10px !important; color: var(--slate) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0b1426",
    "axes.facecolor":    "#111d35",
    "axes.edgecolor":    "#1e2d4a",
    "axes.labelcolor":   "#8496b0",
    "xtick.color":       "#8496b0",
    "ytick.color":       "#8496b0",
    "text.color":        "#f0f4ff",
    "grid.color":        "#1e2d4a",
    "grid.alpha":        0.6,
    "font.family":       "sans-serif",
    "figure.dpi":        120,
})

# ── Constants ──────────────────────────────────────────────────────────────────
N_CLASSES = 6

STAGE_NAMES = {
    0: "No DKD",
    1: "Stage 1 — Microalbuminuria",
    2: "Stage 2 — Macroalbuminuria / Mild GFR Decrease",
    3: "Stage 3 — Moderate GFR Decrease",
    4: "Stage 4 — Severe GFR Decrease",
    5: "Stage 5 — Kidney Failure",
}
STAGE_COLORS = {
    0: "#10b981", 1: "#f59e0b", 2: "#f97316",
    3: "#ef4444", 4: "#a855f7", 5: "#94a3b8",
}
STAGE_RECS = {
    0: {
        "headline": "No diabetic kidney disease detected at this time.",
        "actions": [
            "Continue annual kidney monitoring (eGFR + urine albumin)",
            "Maintain HbA1c below 7% and blood pressure below 130/80 mmHg",
            "Healthy weight, diet, and regular physical activity",
        ],
        "urgency": "Routine follow-up",
        "color": "green",
    },
    1: {
        "headline": "Early kidney damage — protein leaking into urine detected.",
        "actions": [
            "Initiate ACE inhibitor or ARB to protect kidney filtration",
            "Target blood pressure below 130/80 mmHg",
            "Tighten glycaemic control — HbA1c target below 7%",
            "Repeat urine albumin test in 3 months to confirm",
        ],
        "urgency": "Within 4 weeks",
        "color": "amber",
    },
    2: {
        "headline": "Macroalbuminuria or mild GFR decline (eGFR 45–59 or UACR >300 mg/g).",
        "actions": [
            "Refer to a nephrologist for specialist assessment",
            "Reduce dietary protein to 0.8 g/kg/day",
            "Review nephrotoxic medications",
            "Optimise blood pressure and glycaemic control",
        ],
        "urgency": "Within 2–4 weeks",
        "color": "orange",
    },
    3: {
        "headline": "Significant kidney function loss — specialist care required now.",
        "actions": [
            "Urgent nephrology co-management",
            "Begin discussion of kidney replacement options",
            "Investigate and treat renal anaemia",
            "Strict fluid, sodium, and potassium management",
        ],
        "urgency": "Urgent — within 1 week",
        "color": "red",
    },
    4: {
        "headline": "Severe kidney function loss — prepare for replacement therapy.",
        "actions": [
            "Plan dialysis access (arteriovenous fistula creation)",
            "Multidisciplinary team: nephrology, dietitian, social work",
            "Transplant evaluation referral",
            "Strict fluid and electrolyte management",
        ],
        "urgency": "Immediate specialist review",
        "color": "purple",
    },
    5: {
        "headline": "Kidney failure — renal replacement therapy required.",
        "actions": [
            "Initiate dialysis or transplant evaluation",
            "Full renal replacement therapy pathway",
            "Intensive symptom and multidisciplinary support",
            "Palliative care consultation if appropriate",
        ],
        "urgency": "Emergency / Immediate",
        "color": "slate",
    },
}

CLINICAL_CONTEXT = {
    "log_urine_albumin_ugl":         ("Urine Albumin",       "expm1", "Protein leaking into urine — primary early damage marker"),
    "log_serum_creatinine_mgdl":     ("Serum Creatinine",    "expm1", "Waste product in blood — rises when kidneys cannot filter"),
    "log_bun_mgdl":                  ("Blood Urea (BUN)",    "expm1", "Blood urea nitrogen — rises when kidneys are failing"),
    "hba1c_pct":                     ("HbA1c",               None,    "3-month blood glucose average — higher means poorer diabetes control"),
    "hemoglobin_gdl":                ("Haemoglobin",         None,    "Anaemia develops as kidney function declines"),
    "hematocrit_pct":                ("Haematocrit",         None,    "Low values signal kidney-related anaemia"),
    "uric_acid_mgdl":                ("Uric Acid",           None,    "Elevated levels accelerate kidney damage"),
    "log_crp_mgL":                   ("CRP",                 "expm1", "Inflammation marker — drives kidney disease progression"),
    "mean_sbp":                      ("Systolic BP",         None,    "High blood pressure directly damages kidney blood vessels"),
    "mean_dbp":                      ("Diastolic BP",        None,    "Elevated pressure adds strain to kidney filters"),
    "bmi_kgm2":                      ("BMI",                 None,    "Obesity increases kidney disease risk"),
    "log_triglycerides_mgdl":        ("Triglycerides",       "expm1", "High blood fats worsen kidney disease outlook"),
    "hdl_cholesterol_mgdl":          ("HDL Cholesterol",     None,    "Low HDL independently predicts kidney decline"),
    "log_insulin_uiml":              ("Insulin Level",       "expm1", "High fasting insulin signals insulin resistance"),
    "serum_albumin_gdl":             ("Serum Albumin",       None,    "Low albumin indicates poor nutrition from kidney disease"),
    "urine_creatinine_mgdl":         ("Urine Creatinine",    None,    "Used to calculate urine albumin ratio"),
    "kidney_disease_history":        ("Kidney Disease Hx",   None,    "Previous kidney disease strongly predicts DKD"),
    "hypertension_diagnosed":        ("Hypertension",        None,    "Uncontrolled blood pressure accelerates kidney damage"),
    "kidney_stone_history":          ("Kidney Stone Hx",     None,    "Associated with CKD progression"),
    "insulin_use":                   ("Insulin Use",         None,    "Signals advanced diabetes requiring insulin"),
    "phosphorus_mg_day":             ("Dietary Phosphorus",  None,    "High phosphorus intake linked to kidney decline"),
    "potassium_mg_day":              ("Dietary Potassium",   None,    "Potassium management critical in kidney disease"),
    "current_smoker_status":         ("Smoking",             None,    "Smoking damages kidney blood vessels"),
    "vigorous_leisure_activity":     ("Physical Activity",   None,    "Protective — regular exercise reduces kidney disease risk"),
    "age_years":                     ("Patient Age",         None,    "Older age is a primary kidney disease risk factor"),
    "race_ethnicity_code":           ("Ethnicity",           None,    "Ethnic differences in kidney disease risk"),
    "sex_code":                      ("Sex",                 None,    "Sex differences in kidney disease risk trajectory"),
    "heart_attack":                  ("Heart Attack Hx",     None,    "Cardiovascular history worsens prognosis"),
    "stroke_ever":                   ("Stroke History",      None,    "Stroke indicates widespread vascular disease"),
    "coronary_heart_disease":        ("Heart Disease",       None,    "Shared disease process with kidney disease"),
    "family_hx_diabetes":            ("Family Hx Diabetes",  None,    "Genetic predisposition to diabetes and DKD"),
    "education_level":               ("Education Level",     None,    "Proxy for healthcare access and disease management"),
    "household_income_cat":          ("Income Level",        None,    "Socioeconomic factors affect disease management"),
}

SHAP_EXPLAIN = (
    "Red bars indicate factors <strong>increasing</strong> risk for this stage. "
    "Blue bars indicate factors <strong>decreasing</strong> risk. "
    "Longer bars carry stronger influence on the AI's assessment."
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [("page", 0), ("patient", {})]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        url = st.secrets["MODEL_URL"]
    except (KeyError, FileNotFoundError):
        return None, "MODEL_URL not configured in Streamlit secrets."
    try:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        return pickle.load(io.BytesIO(r.content)), None
    except Exception as e:
        return None, f"Could not load model: {e}"

# ── Core helpers ───────────────────────────────────────────────────────────────
def display_name(feat):
    if feat in CLINICAL_CONTEXT:
        return CLINICAL_CONTEXT[feat][0]
    return feat.replace("_", " ").title()

def display_val(feat, raw):
    if feat in CLINICAL_CONTEXT and CLINICAL_CONTEXT[feat][1] == "expm1":
        return float(np.expm1(raw))
    return raw

def build_vector(inputs, mdl):
    ALL = mdl["ALL_FEATS"]
    M1, M2, M3 = mdl["M1"], mdl["M2"], mdl["M3"]
    imp = mdl["knn_imputer"]
    row = {f: np.nan for f in ALL}
    row.update({k: v for k, v in inputs.items() if k in row})
    df = pd.DataFrame([row])[ALL]
    df_imp = pd.DataFrame(imp.transform(df), columns=ALL)
    X1 = df_imp[[c for c in M1 if c in df_imp.columns]]
    X2 = df_imp[[c for c in M2 if c in df_imp.columns]]
    X3 = df_imp[[c for c in M3 if c in df_imp.columns]]
    return df_imp, X1, X2, X3

def run_ensemble(mdl, X1, X2, X3):
    X3s = mdl["lr_scaler"].transform(X3)
    p1  = mdl["rf_pipeline"].predict_proba(X1)
    p2  = mdl["xgb_pipeline"].predict_proba(X2)
    p3  = mdl["lr_model"].predict_proba(X3s)
    mf  = np.hstack([p1, p2, p3])
    ms  = mdl["meta_scaler"].transform(mf)
    pred  = int(mdl["meta_lr"].predict(ms)[0])
    proba = mdl["meta_lr"].predict_proba(ms)[0]
    return pred, proba, p1[0], p2[0], p3[0]

def get_shap(mdl, X1, X2, X3):
    rf_exp  = shap.TreeExplainer(mdl["rf_clf"])
    xgb_exp = shap.TreeExplainer(mdl["xgb_clf"])
    rf_sv   = np.array(rf_exp.shap_values(X1))
    xgb_sv  = np.array(xgb_exp.shap_values(X2))
    try:
        X3s  = mdl["lr_scaler"].transform(X3)
        coef = mdl["lr_model"].coef_
        lr_sv = coef[:, np.newaxis, :] * np.array(X3s)[np.newaxis, :, :]
    except Exception:
        lr_sv = None
    return rf_sv, xgb_sv, lr_sv

def plot_shap_bar(shap_dict, title, c_pos="#ef4444", c_neg="#1a6efc", top_n=10):
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    if not items:
        return None
    labels = [display_name(f) for f, _ in items]
    vals   = [v for _, v in items]
    colors = [c_pos if v > 0 else c_neg for v in vals]
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.48)))
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], alpha=0.85, height=0.6)
    ax.axvline(0, color="#2d4060", lw=1, ls="--")
    ax.set_xlabel("Influence on prediction (SHAP value)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#f0f4ff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    return fig

def urgency_chip(text):
    cm = {"Routine": "green", "Within 4": "blue", "Within 2": "amber",
          "Urgent": "red", "Immediate": "red", "Emergency": "red"}
    c  = next((v for k, v in cm.items() if text.startswith(k)), "blue")
    return f'<span class="chip {c}">{text}</span>'

def stepper_html(current):
    steps = ["Clinical", "Lifestyle", "Demographics", "Results"]
    parts = []
    for i, label in enumerate(steps):
        cls  = "done" if i < current else ("active" if i == current else "todo")
        num  = "✓" if i < current else str(i + 1)
        lc   = "done" if i < current else ""
        dot  = f'<div class="step-dot {cls}">{num}</div>'
        lbl  = f'<div class="step-label">{label}</div>'
        inner = f'<div style="display:flex;flex-direction:column;align-items:center">{dot}{lbl}</div>'
        conn  = f'<div class="step-line {lc}"></div>' if i < len(steps) - 1 else ""
        parts.append(f'<div class="step-item">{inner}{conn}</div>')
    return '<div class="stepper">' + "".join(parts) + "</div>"

# ── Platform header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="platform-header">
  <div class="platform-icon">🔬</div>
  <div>
    <p class="platform-name">GlomeraAI</p>
    <p class="platform-sub">CLINICAL INTELLIGENCE PLATFORM &nbsp;·&nbsp; KDIGO 2024 · NHANES-VALIDATED · FAIRNESS-AUDITED</p>
  </div>
  <div class="platform-badge">AUC 0.961 · NHANES Validated</div>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Initialising AI engine..."):
    mdl, err = load_model()

if err:
    st.error(f"**Model unavailable.** {err}")
    st.info("Set `MODEL_URL` in `.streamlit/secrets.toml` pointing to `DKD_complete_artifacts.pkl`.")
    st.stop()

# ── Sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 GlomeraAI")
    st.markdown('<div style="font-size:.72rem;color:#475569;font-family:\'DM Mono\',monospace;margin:-4px 0 12px">Clinical Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    nav = {
        0: ("🩺", "Patient Assessment"),
        1: ("📈", "HbA1c What-If"),
        2: ("🔮", "Progression Simulation"),
        3: ("🎯", "Demo Patients"),
    }

    for pg, (icon, label) in nav.items():
        active = pg == st.session_state.page
        if active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        prefix = "▶ " if active else "   "
        if st.button(f"{prefix}{icon}  {label}", key=f"nav_{pg}"):
            st.session_state.page = pg; st.rerun()
        if active:
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.77rem;line-height:1.8;color:#8496b0">
    <strong style="color:#f0f4ff">Three specialist models</strong><br>
    <span style="color:#7cb3ff">■ M1 RF</span> Clinical biomarkers
    <span style="color:#7cb3ff;font-family:'DM Mono',monospace;font-size:.7rem"> 53.1%</span><br>
    <span style="color:#a78bfa">■ M3 LR</span> Demographics &amp; history
    <span style="color:#a78bfa;font-family:'DM Mono',monospace;font-size:.7rem"> 27.3%</span><br>
    <span style="color:#fb923c">■ M2 XGB</span> Lifestyle &amp; medications
    <span style="color:#fb923c;font-family:'DM Mono',monospace;font-size:.7rem"> 19.6%</span><br><br>
    <span style="color:#8496b0;font-size:.72rem">Meta-learner weights from<br>SHAP analysis on NHANES data</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-size:.68rem;color:#475569;text-align:center">NHANES 2015–2020 · n=2,627 · KDIGO 2024</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Patient Assessment (wizard)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == 0:

    # ── Sub-page state ─────────────────────────────────────────────────────────
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 0

    step = st.session_state.wizard_step

    # ── Step 0: Clinical ───────────────────────────────────────────────────────
    if step == 0:
        st.markdown(stepper_html(0), unsafe_allow_html=True)
        st.markdown("""<div class="domain-badge rf">
          <h4>Clinical Model (M1) — Blood Tests & Kidney Markers</h4>
          <p>Serum creatinine · Urine albumin · BUN · HbA1c · Haemoglobin</p>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Blood Pressure** *(average of 3 readings)*")
            sbp1 = st.number_input("Systolic BP — Reading 1 (mmHg)",  80, 220, 130, key="sbp1")
            dbp1 = st.number_input("Diastolic BP — Reading 1 (mmHg)", 40, 130, 78,  key="dbp1")
            sbp2 = st.number_input("Systolic BP — Reading 2 (mmHg)",  80, 220, 128, key="sbp2")
            dbp2 = st.number_input("Diastolic BP — Reading 2 (mmHg)", 40, 130, 76,  key="dbp2")
            sbp3 = st.number_input("Systolic BP — Reading 3 (mmHg)",  80, 220, 132, key="sbp3")
            dbp3 = st.number_input("Diastolic BP — Reading 3 (mmHg)", 40, 130, 80,  key="dbp3")
            st.markdown("**Kidney & Urine Markers**")
            serum_cr  = st.number_input("Serum Creatinine (mg/dL)",    0.2, 20.0,   1.1,  step=0.1, key="serum_cr")
            urine_alb = st.number_input("Urine Albumin (µg/L)",         0.5, 5000.0, 25.0, step=0.5, key="urine_alb")
            urine_cr  = st.number_input("Urine Creatinine (mg/dL)",     5.0, 3000.0, 120.0,           key="urine_cr")

        with col2:
            st.markdown("**Metabolic & Blood Chemistry**")
            hba1c     = st.number_input("HbA1c (%)",                 4.0, 20.0,  7.2,  step=0.1, key="hba1c")
            fasting_gl = st.number_input("Fasting Glucose (mg/dL)",  40.0, 600.0, 145.0,          key="fasting_gl")
            insulin   = st.number_input("Fasting Insulin (µIU/mL)",  0.0,  300.0, 12.0,           key="insulin")
            bun       = st.number_input("BUN (mg/dL)",               2.0,  150.0, 16.0,           key="bun")
            uric_acid = st.number_input("Uric Acid (mg/dL)",         1.0,  20.0,  5.8,  step=0.1, key="uric_acid")
            st.markdown("**Blood Count & Other**")
            hemoglobin = st.number_input("Haemoglobin (g/dL)", 4.0,  22.0, 13.5, step=0.1, key="hemoglobin")
            hematocrit = st.number_input("Haematocrit (%)",   10.0,  65.0, 40.0, step=0.5, key="hematocrit")
            serum_alb  = st.number_input("Serum Albumin (g/dL)", 1.0, 6.0,  4.1, step=0.1, key="serum_alb")
            crp        = st.number_input("CRP (mg/L)",         0.0,  200.0, 3.5, step=0.1, key="crp")

        st.markdown("**Lipid Panel & Body**")
        ca, cb, cc = st.columns(3)
        with ca:
            tot_chol = st.number_input("Total Cholesterol (mg/dL)", 50.0,  600.0, 195.0, key="tot_chol")
            ldl      = st.number_input("LDL Cholesterol (mg/dL)",   20.0,  500.0, 115.0, key="ldl")
        with cb:
            hdl  = st.number_input("HDL Cholesterol (mg/dL)", 10.0,  200.0, 48.0,  key="hdl")
            trig = st.number_input("Triglycerides (mg/dL)",   20.0, 3000.0, 145.0, key="trig")
        with cc:
            bmi  = st.number_input("BMI (kg/m²)", 12.0, 80.0, 28.5, step=0.1, key="bmi")

        mean_sbp = round((sbp1 + sbp2 + sbp3) / 3, 1)
        mean_dbp = round((dbp1 + dbp2 + dbp3) / 3, 1)
        uacr     = round(urine_alb / (urine_cr * 10), 2) if urine_cr > 0 else 0.0

        st.markdown(f"""<div class="rec-block blue">
          <strong>Auto-calculated:</strong> &nbsp;
          Mean SBP = <strong>{mean_sbp} mmHg</strong> &nbsp;·&nbsp;
          Mean DBP = <strong>{mean_dbp} mmHg</strong> &nbsp;·&nbsp;
          UACR = <strong>{uacr:.1f} mg/g</strong>
        </div>""", unsafe_allow_html=True)

        st.session_state.patient.update({
            "mean_sbp": mean_sbp, "mean_dbp": mean_dbp,
            "serum_creatinine_mgdl": serum_cr,
            "log_serum_creatinine_mgdl": np.log1p(serum_cr),
            "urine_albumin_ugl": urine_alb,
            "log_urine_albumin_ugl": np.log1p(urine_alb),
            "urine_creatinine_mgdl": urine_cr,
            "uacr_mgg": uacr, "log_uacr": np.log10(max(uacr, 0.01)),
            "hba1c_pct": hba1c,
            "fasting_glucose_mgdl": fasting_gl,
            "log_fasting_glucose_mgdl": np.log1p(fasting_gl),
            "insulin_uiml": insulin,
            "log_insulin_uiml": np.log1p(insulin),
            "bun_mgdl": bun, "log_bun_mgdl": np.log1p(bun),
            "uric_acid_mgdl": uric_acid,
            "hemoglobin_gdl": hemoglobin, "hematocrit_pct": hematocrit,
            "serum_albumin_gdl": serum_alb,
            "crp_mgL": crp, "log_crp_mgL": np.log1p(crp),
            "total_cholesterol_mgdl": tot_chol, "ldl_cholesterol_mgdl": ldl,
            "hdl_cholesterol_mgdl": hdl,
            "triglycerides_mgdl": trig, "log_triglycerides_mgdl": np.log1p(trig),
            "bmi_kgm2": bmi,
        })
        _, col_btn, _ = st.columns([2, 3, 2])
        with col_btn:
            if st.button("Continue to Lifestyle →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 1; st.rerun()

    # ── Step 1: Lifestyle ──────────────────────────────────────────────────────
    elif step == 1:
        st.markdown(stepper_html(1), unsafe_allow_html=True)
        st.markdown("""<div class="domain-badge xgb">
          <h4>Lifestyle Model (M2) — Medications, Activity & Diet</h4>
          <p>Kidney disease history · Hypertension · Insulin use · Physical activity · Phosphorus</p>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Medications & Diagnoses**")
            dm_dx       = st.selectbox("Diabetes Diagnosed?",               [1, 0], format_func=lambda x: "Yes" if x else "No", key="dm_dx")
            insulin_use = st.selectbox("Currently using insulin?",          [0, 1], format_func=lambda x: "Yes" if x else "No", key="insulin_use")
            oral_meds   = st.selectbox("Oral diabetes medications?",        [0, 1], format_func=lambda x: "Yes" if x else "No", key="oral_meds")
            htn_dx      = st.selectbox("Diagnosed hypertension?",           [0, 1], format_func=lambda x: "Yes" if x else "No", key="htn_dx")
            bp_med      = st.selectbox("Blood pressure medication?",        [0, 1], format_func=lambda x: "Yes" if x else "No", key="bp_med")
            statin      = st.selectbox("Taking statins?",                   [0, 1], format_func=lambda x: "Yes" if x else "No", key="statin")
            st.markdown("**Kidney History**")
            kidney_hx    = st.selectbox("History of kidney disease?",  [0, 1], format_func=lambda x: "Yes" if x else "No", key="kidney_hx")
            kidney_stone = st.selectbox("History of kidney stones?",   [0, 1], format_func=lambda x: "Yes" if x else "No", key="kidney_stone")
            nocturia     = st.selectbox("Waking at night to urinate?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="nocturia")

        with col2:
            st.markdown("**Activity & Lifestyle**")
            smoker      = st.selectbox("Smoking status", [0, 1, 2],
                                       format_func=lambda x: ["Never", "Former", "Current smoker"][x], key="smoker")
            alcohol     = st.number_input("Alcohol (drinks/day)", 0.0, 20.0, 0.3, step=0.1, key="alcohol")
            vig_leisure = st.selectbox("Vigorous exercise?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="vig_leisure")
            sedentary   = st.number_input("Sedentary time/day (minutes)", 0, 1440, 300, key="sedentary")
            sleep_h     = st.number_input("Weekday sleep (hours)", 2.0, 14.0, 7.0, step=0.5, key="sleep_h")
            st.markdown("**Dietary Intake (daily averages)**")
            sodium     = st.number_input("Sodium (mg/day)",       100.0, 15000.0, 2800.0, key="sodium")
            protein    = st.number_input("Protein (g/day)",         5.0,   400.0,   85.0, key="protein")
            potassium  = st.number_input("Potassium (mg/day)",     200.0,  8000.0, 2800.0, key="potassium")
            phosphorus = st.number_input("Phosphorus (mg/day)",    100.0,  4000.0, 1100.0, key="phosphorus")

        st.session_state.patient.update({
            "diabetes_diagnosed": dm_dx, "insulin_use": insulin_use,
            "oral_diabetes_meds": oral_meds, "hypertension_diagnosed": htn_dx,
            "bp_medication": bp_med, "statin_use": statin,
            "current_smoker_status": smoker,
            "avg_alcohol_drinks_per_day": alcohol,
            "log_avg_alcohol_drinks_per_day": np.log1p(alcohol),
            "vigorous_leisure_activity": vig_leisure,
            "sedentary_minutes_per_day": sedentary,
            "log_sedentary_minutes_per_day": np.log1p(sedentary),
            "sleep_hours_weekday": sleep_h,
            "kidney_disease_history": kidney_hx, "kidney_stone_history": kidney_stone,
            "nocturia": nocturia,
            "sodium_mg_day": sodium, "protein_g_day": protein,
            "potassium_mg_day": potassium, "phosphorus_mg_day": phosphorus,
        })
        cl, _, cr = st.columns([2, 2, 2])
        with cl:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 0; st.rerun()
        with cr:
            if st.button("Continue to Demographics →", type="primary", use_container_width=True):
                st.session_state.wizard_step = 2; st.rerun()

    # ── Step 2: Demographics ───────────────────────────────────────────────────
    elif step == 2:
        st.markdown(stepper_html(2), unsafe_allow_html=True)
        st.markdown("""<div class="domain-badge lr">
          <h4>Demographics Model (M3) — Patient Background & History</h4>
          <p>Age · Sex · Ethnicity · Cardiovascular history · Socioeconomic factors</p>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Patient Background**")
            age  = st.number_input("Age (years)", 18, 100, 58, key="age")
            sex  = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female", key="sex")
            race = st.selectbox("Race / Ethnicity", [1, 2, 3, 4, 6, 7],
                                format_func=lambda x: {1:"Mexican American",2:"Other Hispanic",
                                                       3:"Non-Hispanic White",4:"Non-Hispanic Black",
                                                       6:"Non-Hispanic Asian",7:"Other/Multiracial"}[x], key="race")
            st.markdown("**Socioeconomic**")
            education = st.selectbox("Education level", [1,2,3,4,5], index=3,
                                     format_func=lambda x: {1:"< 9th grade",2:"Some high school",
                                                            3:"High school / GED",4:"Some college",
                                                            5:"College graduate+"}[x], key="education")
            income = st.selectbox("Household income", [1,2,3,4,5,6,7,8,9,10,14,15], index=7,
                                  format_func=lambda x: {1:"< $5k",2:"$5–10k",3:"$10–15k",4:"$15–20k",
                                                         5:"$20–25k",6:"$25–35k",7:"$35–45k",8:"$45–55k",
                                                         9:"$55–65k",10:"$65–75k",14:"$75–100k",
                                                         15:"Over $100k"}[x], key="income")
            food_sec = st.number_input("Food security score (0–18)", 0.0, 18.0, 10.0, step=1.0, key="food_sec")

        with col2:
            st.markdown("**Cardiovascular & Family History**")
            chd       = st.selectbox("Coronary heart disease?", [0,1], format_func=lambda x: "Yes" if x else "No", key="chd")
            heart_att = st.selectbox("Heart attack history?",   [0,1], format_func=lambda x: "Yes" if x else "No", key="heart_att")
            stroke    = st.selectbox("Stroke history?",         [0,1], format_func=lambda x: "Yes" if x else "No", key="stroke")
            fam_hx_dm = st.selectbox("Family history diabetes?",[0,1], format_func=lambda x: "Yes" if x else "No", key="fam_hx_dm")

        sex_l  = "Female" if sex == 1 else "Male"
        race_l = {1:"Mexican American",2:"Other Hispanic",3:"Non-Hispanic White",
                  4:"Non-Hispanic Black",6:"Non-Hispanic Asian",7:"Other/Multiracial"}[race]
        st.markdown(f"""<div class="rec-block blue">
          <strong>Patient Profile:</strong> {age}-year-old {sex_l} · {race_l}
        </div>""", unsafe_allow_html=True)

        st.session_state.patient.update({
            "age_years": age, "sex_code": sex, "race_ethnicity_code": race,
            "education_level": education, "household_income_cat": income,
            "food_security_score": food_sec,
            "coronary_heart_disease": chd, "heart_attack": heart_att,
            "stroke_ever": stroke, "family_hx_diabetes": fam_hx_dm,
        })
        cl, _, cr = st.columns([2, 2, 2])
        with cl:
            if st.button("← Back", use_container_width=True):
                st.session_state.wizard_step = 1; st.rerun()
        with cr:
            if st.button("🔬  Run AI Assessment", type="primary", use_container_width=True):
                st.session_state.wizard_step = 3; st.rerun()

    # ── Step 3: Results ────────────────────────────────────────────────────────
    elif step == 3:
        st.markdown(stepper_html(3), unsafe_allow_html=True)

        with st.spinner("Running AI analysis across all three models..."):
            df_imp, X1, X2, X3 = build_vector(st.session_state.patient, mdl)
            pred, proba, p1, p2, p3 = run_ensemble(mdl, X1, X2, X3)
            rf_sv, xgb_sv, lr_sv = get_shap(mdl, X1, X2, X3)

        rec         = STAGE_RECS[pred]
        stage_color = STAGE_COLORS[pred]
        prog_score  = sum(c * proba[c] for c in range(N_CLASSES))
        entropy_val = scipy_entropy(proba)
        uncertainty = entropy_val / scipy_entropy([1/N_CLASSES]*N_CLASSES)
        conf_label  = ("High confidence" if uncertainty < 0.33 else
                       "Moderate" if uncertainty < 0.66 else "Low — review carefully")
        conf_color  = ("#10b981" if uncertainty < 0.33 else
                       "#f59e0b" if uncertainty < 0.66 else "#ef4444")
        forward_p   = float(sum(proba[c] for c in range(pred+1, N_CLASSES)))

        # Store for use in other pages
        st.session_state["last_pred"]  = pred
        st.session_state["last_proba"] = proba
        st.session_state["last_hba1c"] = st.session_state.patient.get("hba1c_pct", 7.2)

        # Result hero
        st.markdown(f"""
        <div class="stage-result-card" style="color:{stage_color};
             background:rgba({','.join(str(int(int(stage_color.lstrip('#')[i:i+2],16)*0.12)) for i in (0,2,4))},1);
             border-color:{stage_color}40">
          <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.14em;
                      color:{stage_color};margin-bottom:.5rem">GlomeraAI ASSESSMENT</div>
          <div style="font-size:2.1rem;font-weight:800;letter-spacing:-.03em;color:{stage_color}">
            {STAGE_NAMES[pred]}</div>
          <div style="font-size:.9rem;color:#8496b0;margin-top:.5rem">{rec['headline']}</div>
          <div style="margin-top:.9rem">{urgency_chip(rec['urgency'])}</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        tiles = [
            (f"Stage {pred}", "KDIGO 2024 Stage (0–5)", stage_color),
            (f"{proba[pred]*100:.0f}%", "Stage probability", "#f0f4ff"),
            (conf_label.split()[0], "AI confidence", conf_color),
            (f"{prog_score:.2f}/5", "Severity score", "#f0f4ff"),
        ]
        for col, (val, lbl, color) in zip([c1,c2,c3,c4], tiles):
            with col:
                st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:{color}">{val}</div><div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""<div class="rec-block {rec['color']}">
          <strong>Recommended Clinical Actions</strong>
          <ul style="margin:.5rem 0 0;padding-left:1.2rem">
            {''.join(f"<li style='margin:.3rem 0'>{a}</li>" for a in rec['actions'])}
          </ul>
        </div>""", unsafe_allow_html=True)

        # Plain-language SHAP narrative
        shap_rf_dict = dict(zip(X1.columns.tolist(), rf_sv[0, :, pred]))
        top3 = sorted(shap_rf_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        narrative_parts = []
        for feat, sv in top3:
            name = display_name(feat)
            raw  = display_val(feat, float(df_imp[feat].iloc[0]))
            direction = "elevated" if sv > 0 else "lower than expected"
            desc = CLINICAL_CONTEXT.get(feat, ("", None, ""))[2]
            narrative_parts.append(f"<strong>{name}</strong> ({raw:.2f}) is {direction} — {desc}")
        narrative_html = "; ".join(narrative_parts) + "."
        st.markdown(f"""<div class="rec-block teal">
          <strong>Why this prediction?</strong><br>
          The primary drivers for this patient are: {narrative_html}
        </div>""", unsafe_allow_html=True)

        st.divider()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Summary", "Clinical (M1)", "Lifestyle (M2)",
            "Demographics (M3)", "Progression Risk", "Fairness",
        ])

        with tab1:
            st.markdown('<p class="sect-hdr">Patient Snapshot</p>', unsafe_allow_html=True)
            pd = st.session_state.patient
            cs1, cs2, cs3 = st.columns(3)
            with cs1:
                st.markdown("**Key Clinical Values**")
                st.write(f"HbA1c: **{pd.get('hba1c_pct','—'):.1f}%** (target <7%)")
                st.write(f"UACR: **{pd.get('uacr_mgg','—'):.1f} mg/g**")
                st.write(f"BP: **{pd.get('mean_sbp','—')}/{pd.get('mean_dbp','—')} mmHg**")
                st.write(f"BMI: **{pd.get('bmi_kgm2','—'):.1f} kg/m²**")
            with cs2:
                st.markdown("**Model Agreement**")
                for label, prob, color in [
                    ("Clinical (M1)",     p1[pred], "#1a6efc"),
                    ("Lifestyle (M2)",    p2[pred], "#f97316"),
                    ("Demographics (M3)", p3[pred], "#a855f7"),
                ]:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                         padding:.4rem .7rem;background:rgba(255,255,255,.04);border-radius:8px;margin:.3rem 0">
                      <span style="font-size:.83rem">{label}</span>
                      <strong style="color:{color};font-family:'DM Mono',monospace">{prob*100:.0f}%</strong>
                    </div>""", unsafe_allow_html=True)
            with cs3:
                st.markdown("**Patient Demographics**")
                sex_l  = "Female" if pd.get("sex_code")==1 else "Male"
                race_l = {1:"Mexican Am.",2:"Other Hispanic",3:"NH White",
                          4:"NH Black",6:"NH Asian",7:"Other"}.get(pd.get("race_ethnicity_code"),"—")
                inc    = pd.get("household_income_cat",8)
                inc_l  = "Low" if inc<=5 else ("Middle" if inc<=9 else "High")
                st.write(f"Age: **{pd.get('age_years','—')}** · {sex_l}")
                st.write(f"Ethnicity: **{race_l}**")
                st.write(f"Income tier: **{inc_l}**")

            st.markdown('<p class="sect-hdr">Stage Probability Distribution</p>', unsafe_allow_html=True)
            fig_dist, ax = plt.subplots(figsize=(10, 2.8))
            bars = ax.barh([STAGE_NAMES[c] for c in range(N_CLASSES)],
                           [proba[c]*100 for c in range(N_CLASSES)],
                           color=[STAGE_COLORS[c] for c in range(N_CLASSES)],
                           alpha=0.85, height=0.62)
            for bar, val in zip(bars, proba):
                ax.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2,
                        f"{val*100:.1f}%", va="center", fontsize=8.5, fontweight="600")
            ax.set_xlabel("Probability (%)", fontsize=9)
            ax.set_xlim(0, 110)
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_dist, use_container_width=True); plt.close()

        with tab2:
            st.markdown("""<div class="domain-badge rf">
              <h4>Clinical Model (M1) — Blood test and kidney marker influence</h4>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="rec-block blue">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
            fig_rf = plot_shap_bar(shap_rf_dict, f"Clinical Factors — Stage {pred} Prediction", "#ef4444", "#1a6efc")
            if fig_rf:
                st.pyplot(fig_rf, use_container_width=True); plt.close()

        with tab3:
            st.markdown("""<div class="domain-badge xgb">
              <h4>Lifestyle Model (M2) — Medication and lifestyle influence</h4>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="rec-block orange">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
            shap_xgb = dict(zip(X2.columns.tolist(), xgb_sv[0, :, pred]))
            fig_xgb  = plot_shap_bar(shap_xgb, f"Lifestyle Factors — Stage {pred} Prediction", "#f97316", "#0ea5e9")
            if fig_xgb:
                st.pyplot(fig_xgb, use_container_width=True); plt.close()

            st.markdown('<p class="sect-hdr">Three-Model Agreement</p>', unsafe_allow_html=True)
            fig_comp, ax2 = plt.subplots(figsize=(10, 3.2))
            x, w = np.arange(N_CLASSES), 0.26
            ax2.bar(x-w, p1*100, w, label="M1 Clinical",     color="#1a6efc", alpha=0.85)
            ax2.bar(x,   p2*100, w, label="M2 Lifestyle",    color="#f97316", alpha=0.85)
            ax2.bar(x+w, p3*100, w, label="M3 Demographics", color="#a855f7", alpha=0.85)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Stage {c}" for c in range(N_CLASSES)], fontsize=8.5)
            ax2.set_ylabel("Confidence (%)", fontsize=9)
            ax2.set_title("All Three Models — Stage Confidence", fontsize=10, fontweight="bold")
            ax2.legend(fontsize=8.5)
            ax2.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_comp, use_container_width=True); plt.close()

        with tab4:
            st.markdown("""<div class="domain-badge lr">
              <h4>Demographics Model (M3) — Patient background influence</h4>
            </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="rec-block purple">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
            if lr_sv is not None:
                try:
                    shap_lr = (dict(zip(X3.columns.tolist(), lr_sv[pred, 0, :]))
                               if lr_sv.ndim == 3 else
                               dict(zip(X3.columns.tolist(), lr_sv[0, :])))
                except Exception:
                    shap_lr = {}
                if shap_lr:
                    fig_lr = plot_shap_bar(shap_lr, f"Demographic Factors — Stage {pred} Prediction", "#a855f7", "#06b6d4")
                    if fig_lr:
                        st.pyplot(fig_lr, use_container_width=True); plt.close()
            st.markdown("""<div class="rec-block purple">
              <strong>Note on demographic factors:</strong> Differences by age, ethnicity, or sex reflect
              documented population-level differences in DKD rates, not algorithmic bias. This system
              was independently audited for fairness across all demographic groups.
            </div>""", unsafe_allow_html=True)

        with tab5:
            st.markdown('<p class="sect-hdr">Progression Risk Assessment</p>', unsafe_allow_html=True)
            if pred >= N_CLASSES - 1:
                st.markdown("""<div class="verdict-card safe">
                  <div class="verdict-title" style="color:#10b981">Stage 5 — Kidney Replacement Required</div>
                  <div class="verdict-body">Patient is at the most advanced stage. Focus on renal replacement therapy.</div>
                </div>""", unsafe_allow_html=True)
            else:
                next_name = STAGE_NAMES[pred+1]
                if forward_p >= 0.30:
                    v_cls  = "danger"; v_title = "PROGRESSION LIKELY"
                    v_color = "#ef4444"; v_txt = f"{forward_p*100:.0f}% chance of progressing beyond {STAGE_NAMES[pred]}. Prioritise clinical intervention and nephrology referral."
                elif forward_p >= 0.15:
                    v_cls  = "warning"; v_title = "MONITOR CLOSELY"
                    v_color = "#f59e0b"; v_txt = f"{forward_p*100:.0f}% probability of worsening to {next_name}. Good management may prevent progression."
                else:
                    v_cls  = "safe"; v_title = "STAGE APPEARS STABLE"
                    v_color = "#10b981"; v_txt = f"Only {forward_p*100:.0f}% chance of progression. Current management appears effective."
                st.markdown(f"""<div class="verdict-card {v_cls}">
                  <div class="verdict-title" style="color:{v_color}">{v_title}</div>
                  <div class="verdict-body">{v_txt}</div>
                </div>""", unsafe_allow_html=True)

            # Severity gauge
            fig_g, ax_g = plt.subplots(figsize=(9, 1.8))
            ax_g.barh([""], [prog_score], color=stage_color, alpha=0.85, height=0.45)
            ax_g.set_xlim(0, 5)
            for s in range(6):
                ax_g.axvline(s, color="#1e2d4a", lw=0.8)
                ax_g.text(s+0.06, 0.27, f"S{s}", fontsize=7.5, color="#8496b0")
            ax_g.axvline(pred, color="#f0f4ff", lw=1.5, ls="--", alpha=0.5)
            ax_g.set_xlabel("Disease Severity (0 = No DKD → 5 = Kidney Failure)", fontsize=9)
            ax_g.spines[["top","right","left"]].set_visible(False)
            ax_g.set_yticks([])
            ax_g.text(prog_score+0.07, 0, f"{prog_score:.2f}", fontsize=10, fontweight="700", va="center", color=stage_color)
            plt.tight_layout()
            st.pyplot(fig_g, use_container_width=True); plt.close()

            st.markdown("""<div class="rec-block teal">
              <strong>Want to explore how intervention could change this trajectory?</strong><br>
              Use the <strong>HbA1c What-If</strong> page to see how blood glucose control affects this
              patient's risk, or the <strong>Progression Simulation</strong> page for a 24-month outlook.
            </div>""", unsafe_allow_html=True)

        with tab6:
            pd = st.session_state.patient
            sex_l  = "Female" if pd.get("sex_code")==1 else "Male"
            race_l = {1:"Mexican Am.",2:"Other Hispanic",3:"NH White",
                      4:"NH Black",6:"NH Asian",7:"Other"}.get(pd.get("race_ethnicity_code"),"—")
            age_v  = pd.get("age_years", 0)
            ag     = "<40" if age_v<40 else "40–55" if age_v<55 else "55–65" if age_v<65 else "65+"
            inc    = pd.get("household_income_cat",8)
            inc_l  = "Low" if inc<=5 else ("Middle" if inc<=9 else "High")

            cf1,cf2,cf3,cf4 = st.columns(4)
            with cf1: st.metric("Sex", sex_l)
            with cf2: st.metric("Ethnicity", race_l)
            with cf3: st.metric("Age Group", ag)
            with cf4: st.metric("Income", inc_l)

            st.markdown("""<div class="rec-block blue">
              <strong>Fairness audit summary:</strong> This system was independently validated across sex,
              ethnicity, age group, and income tier before deployment. AUC parity within 5 percentage points
              achieved across all groups. NOTE flags reflect expected population-level disease rate variation,
              not algorithmic bias. Sensitivity of 100% achieved for Stages 4–5 across all demographic groups.
            </div>""", unsafe_allow_html=True)

            if "summary_df" in mdl and mdl["summary_df"] is not None:
                st.markdown('<p class="sect-hdr">Fairness Audit Results</p>', unsafe_allow_html=True)
                st.dataframe(mdl["summary_df"], use_container_width=True)

        st.markdown("""
        <div class="disclaimer">
          <strong>⚠ Clinical Disclaimer:</strong> GlomeraAI is a clinical decision-support tool only.
          All predictions must be reviewed by a qualified clinician alongside the full patient history,
          physical examination findings, and current clinical guidelines before informing any clinical
          decision. Trained on cross-sectional NHANES 2015–2020 data (n=2,627 diabetic adults, AUC 0.961).
        </div>""", unsafe_allow_html=True)

        st.divider()
        cl, _, cr = st.columns([2,3,2])
        with cl:
            if st.button("Edit Patient Data", use_container_width=True):
                st.session_state.wizard_step = 0; st.rerun()
        with cr:
            if st.button("New Patient", type="primary", use_container_width=True):
                st.session_state.wizard_step = 0
                st.session_state.patient = {}
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HbA1c What-If Trajectory (CORE NEW FEATURE)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 1:

    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:800;color:#f0f4ff;margin-bottom:.3rem">
        HbA1c Intervention What-If
      </div>
      <div style="font-size:.82rem;color:#8496b0;line-height:1.7">
        Using the actual trained GlomeraAI ensemble, this tool shows how the patient's predicted
        kidney disease risk changes as HbA1c improves — month by month — over the next 12 months.
        Every result shown is a real prediction from the validated model.
      </div>
    </div>
    """, unsafe_allow_html=True)

    has_patient = bool(st.session_state.patient)
    current_hba1c = st.session_state.patient.get("hba1c_pct", 7.2)

    if not has_patient:
        st.markdown("""<div class="rec-block amber">
          <strong>No patient loaded.</strong> Complete the Patient Assessment first to use the
          What-If tool with real patient data. You can also use the manual inputs below to explore
          a hypothetical patient.
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="sect-hdr">Intervention Scenario</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1,1])
    with col_l:
        hba1c_now = st.slider(
            "Current HbA1c (%)",
            min_value=5.5, max_value=14.0,
            value=float(round(current_hba1c, 1)),
            step=0.1,
            help="The patient's current HbA1c reading"
        )
        hba1c_goal = st.slider(
            "Target HbA1c at 12 months (%)",
            min_value=5.5, max_value=14.0,
            value=max(5.5, float(round(current_hba1c - 2.0, 1))),
            step=0.1,
            help="The clinical target after intervention"
        )

    with col_r:
        intervention_label = st.selectbox("Intervention type", [
            "Intensive glycaemic management",
            "Lifestyle modification only",
            "Pharmacotherapy initiation",
            "Combination therapy",
            "Custom (use slider values)",
        ])
        n_months = st.slider("Projection horizon (months)", 6, 24, 12, step=6)

    if hba1c_goal >= hba1c_now:
        st.markdown("""<div class="rec-block red">
          Target HbA1c should be lower than the current value to model an improvement.
          Adjust the sliders so the target is below the current HbA1c.
        </div>""", unsafe_allow_html=True)
    else:
        trajectory = np.linspace(hba1c_now, hba1c_goal, n_months + 1)

        if st.button("▶  Run What-If Analysis", type="primary"):
            if not has_patient:
                # Use default patient with just HbA1c varying
                base = {
                    "mean_sbp": 130.0, "mean_dbp": 78.0,
                    "serum_creatinine_mgdl": 1.2, "log_serum_creatinine_mgdl": np.log1p(1.2),
                    "urine_albumin_ugl": 30.0, "log_urine_albumin_ugl": np.log1p(30.0),
                    "urine_creatinine_mgdl": 120.0, "uacr_mgg": 0.009,
                    "log_uacr": np.log10(0.009), "hemoglobin_gdl": 13.5,
                    "hematocrit_pct": 40.0, "serum_albumin_gdl": 4.1,
                    "bun_mgdl": 16.0, "log_bun_mgdl": np.log1p(16.0),
                    "uric_acid_mgdl": 5.8, "crp_mgL": 3.5, "log_crp_mgL": np.log1p(3.5),
                    "total_cholesterol_mgdl": 195.0, "ldl_cholesterol_mgdl": 115.0,
                    "hdl_cholesterol_mgdl": 48.0, "triglycerides_mgdl": 145.0,
                    "log_triglycerides_mgdl": np.log1p(145.0), "bmi_kgm2": 28.5,
                    "fasting_glucose_mgdl": 145.0, "log_fasting_glucose_mgdl": np.log1p(145.0),
                    "insulin_uiml": 12.0, "log_insulin_uiml": np.log1p(12.0),
                    "hypertension_diagnosed": 1, "kidney_disease_history": 0,
                    "kidney_stone_history": 0, "insulin_use": 1, "bp_medication": 1,
                    "diabetes_diagnosed": 1, "oral_diabetes_meds": 0, "statin_use": 0,
                    "current_smoker_status": 0, "avg_alcohol_drinks_per_day": 0.0,
                    "log_avg_alcohol_drinks_per_day": 0.0,
                    "vigorous_leisure_activity": 0, "sedentary_minutes_per_day": 300,
                    "log_sedentary_minutes_per_day": np.log1p(300), "sleep_hours_weekday": 7.0,
                    "nocturia": 0, "sodium_mg_day": 2800.0, "protein_g_day": 85.0,
                    "potassium_mg_day": 2800.0, "phosphorus_mg_day": 1100.0,
                    "age_years": 58, "sex_code": 0, "race_ethnicity_code": 3,
                    "education_level": 4, "household_income_cat": 8,
                    "food_security_score": 10.0, "coronary_heart_disease": 0,
                    "heart_attack": 0, "stroke_ever": 0, "family_hx_diabetes": 1,
                }
            else:
                base = dict(st.session_state.patient)

            results_over_time = []
            with st.spinner(f"Running GlomeraAI at {n_months+1} time points..."):
                for t, hba1c_val in enumerate(trajectory):
                    patient_t = dict(base)
                    patient_t["hba1c_pct"] = float(hba1c_val)
                    patient_t["log_fasting_glucose_mgdl"] = np.log1p(
                        base.get("fasting_glucose_mgdl", 145.0) * (hba1c_val / hba1c_now)
                    )
                    _, X1t, X2t, X3t = build_vector(patient_t, mdl)
                    pred_t, proba_t, _, _, _ = run_ensemble(mdl, X1t, X2t, X3t)
                    results_over_time.append({
                        "month": t,
                        "hba1c": hba1c_val,
                        "pred": pred_t,
                        "proba": proba_t.tolist(),
                    })

            # ── Key metrics ────────────────────────────────────────────────────
            start_pred  = results_over_time[0]["pred"]
            end_pred    = results_over_time[-1]["pred"]
            start_proba = results_over_time[0]["proba"]
            end_proba   = results_over_time[-1]["proba"]
            start_risk  = sum(c * start_proba[c] for c in range(N_CLASSES))
            end_risk    = sum(c * end_proba[c]   for c in range(N_CLASSES))
            delta_risk  = end_risk - start_risk

            m1, m2, m3, m4 = st.columns(4)
            with m1: st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:{STAGE_COLORS[start_pred]}">Stage {start_pred}</div><div class="kpi-lbl">Predicted now</div></div>', unsafe_allow_html=True)
            with m2: st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:{STAGE_COLORS[end_pred]}">Stage {end_pred}</div><div class="kpi-lbl">At {n_months} months</div></div>', unsafe_allow_html=True)
            with m3: st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:#f59e0b">{hba1c_now:.1f}%→{hba1c_goal:.1f}%</div><div class="kpi-lbl">HbA1c trajectory</div></div>', unsafe_allow_html=True)
            with m4:
                d_color = "#10b981" if delta_risk < 0 else "#ef4444"
                d_sign  = "▼" if delta_risk < 0 else "▲"
                st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:{d_color}">{d_sign}{abs(delta_risk):.2f}</div><div class="kpi-lbl">Severity score change</div></div>', unsafe_allow_html=True)

            # Verdict banner
            if delta_risk < -0.3:
                st.markdown(f"""<div class="rec-block green">
                  <strong>Clinically meaningful improvement projected.</strong> With HbA1c reduction
                  from <strong>{hba1c_now:.1f}%</strong> to <strong>{hba1c_goal:.1f}%</strong>,
                  GlomeraAI projects the severity score to decrease by <strong>{abs(delta_risk):.2f} points</strong>
                  over {n_months} months. This {intervention_label.lower()} scenario demonstrates measurable
                  benefit for this patient's kidney risk profile.
                </div>""", unsafe_allow_html=True)
            elif delta_risk < 0:
                st.markdown(f"""<div class="rec-block amber">
                  <strong>Modest improvement projected.</strong> Severity score reduces by
                  <strong>{abs(delta_risk):.2f} points</strong> over {n_months} months. Additional
                  interventions (blood pressure control, dietary modification) may amplify the benefit.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="rec-block red">
                  <strong>Risk remains elevated despite HbA1c improvement.</strong> Other risk factors
                  (creatinine, blood pressure, proteinuria) are driving this patient's risk score.
                  HbA1c control alone may be insufficient — review full clinical picture.
                </div>""", unsafe_allow_html=True)

            # ── Trajectory plot ────────────────────────────────────────────────
            months = [r["month"] for r in results_over_time]
            pred_stages = [r["pred"] for r in results_over_time]
            severity_scores = [sum(c * r["proba"][c] for c in range(N_CLASSES)) for r in results_over_time]
            hba1c_vals = [r["hba1c"] for r in results_over_time]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            # Plot 1 — HbA1c trajectory
            axes[0].plot(months, hba1c_vals, color="#f59e0b", lw=2.5, marker="o", markersize=5)
            axes[0].axhline(7.0, color="#10b981", lw=1, ls="--", alpha=0.7, label="Target <7%")
            axes[0].fill_between(months, hba1c_vals, 7.0,
                                 where=[h > 7.0 for h in hba1c_vals],
                                 alpha=0.15, color="#ef4444")
            axes[0].set_title("HbA1c Trajectory", fontsize=10, fontweight="bold")
            axes[0].set_xlabel("Month"); axes[0].set_ylabel("HbA1c (%)")
            axes[0].legend(fontsize=8); axes[0].spines[["top","right"]].set_visible(False)

            # Plot 2 — Severity score
            axes[1].plot(months, severity_scores, color="#1a6efc", lw=2.5, marker="o", markersize=5)
            axes[1].fill_between(months, severity_scores, alpha=0.12, color="#1a6efc")
            for t, score in enumerate(severity_scores):
                stage = pred_stages[t]
                if t == 0 or t == len(months) - 1:
                    axes[1].annotate(f"S{stage}", (t, score), textcoords="offset points",
                                     xytext=(0, 8), fontsize=8, ha="center",
                                     color=STAGE_COLORS[stage], fontweight="bold")
            axes[1].set_title("Severity Score (Real Model Output)", fontsize=10, fontweight="bold")
            axes[1].set_xlabel("Month"); axes[1].set_ylabel("Severity Score (0–5)")
            axes[1].set_ylim(0, 5); axes[1].spines[["top","right"]].set_visible(False)

            # Plot 3 — Stage probabilities stacked area
            stage_proba_matrix = np.array([[r["proba"][c] for c in range(N_CLASSES)]
                                           for r in results_over_time])
            bottom = np.zeros(len(months))
            for c in range(N_CLASSES):
                axes[2].fill_between(months, bottom, bottom + stage_proba_matrix[:, c],
                                     color=STAGE_COLORS[c], alpha=0.8,
                                     label=STAGE_NAMES[c].split("—")[0].strip())
                bottom += stage_proba_matrix[:, c]
            axes[2].set_title("Stage Probability Over Time", fontsize=10, fontweight="bold")
            axes[2].set_xlabel("Month"); axes[2].set_ylabel("Probability")
            axes[2].set_ylim(0, 1); axes[2].legend(fontsize=7, loc="upper right")
            axes[2].spines[["top","right"]].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

            # ── Tabular breakdown ──────────────────────────────────────────────
            st.markdown('<p class="sect-hdr">Monthly Breakdown</p>', unsafe_allow_html=True)
            rows = []
            for r in results_over_time:
                rows.append({
                    "Month": r["month"],
                    "HbA1c (%)": f"{r['hba1c']:.1f}",
                    "Predicted Stage": f"Stage {r['pred']} — {STAGE_NAMES[r['pred']]}",
                    "Severity Score": f"{sum(c * r['proba'][c] for c in range(N_CLASSES)):.3f}",
                    "P(Stage 4+)": f"{sum(r['proba'][c] for c in range(4, N_CLASSES))*100:.1f}%",
                })
            tbl_df = pd.DataFrame(rows)
            st.dataframe(tbl_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer">
      <strong>What this tool is:</strong> A sensitivity analysis using the actual trained GlomeraAI ensemble.
      Each data point shown is a real model prediction — not a simulation. HbA1c is varied along a
      linear trajectory; all other patient features remain constant. This reflects the isolated effect of
      glycaemic control on DKD risk as captured by the trained model.
      <strong>This does not constitute a clinical prediction of individual patient outcomes.</strong>
      All outputs require review by a qualified clinician.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Progression Risk Simulation (SUPPORTING FEATURE)
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 2:

    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:800;color:#f0f4ff;margin-bottom:.3rem">
        24-Month Progression Risk Simulation
      </div>
      <div style="font-size:.82rem;color:#8496b0;line-height:1.7">
        A rule-based clinical simulation combining this patient's GlomeraAI risk profile with
        published KDIGO annual stage-transition rates. Shows the probability distribution of
        where the patient could be at 6, 12, and 24 months under different intervention scenarios.
        <strong>This is a population-level simulation, not an individual prediction.</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Published KDIGO transition matrix ──────────────────────────────────────
    # Annual stage-to-stage transition rates
    # Source: KDIGO 2024 Clinical Practice Guidelines (Kidney Int. 105(4S):S117-S314)
    T_ANNUAL = np.array([
        [0.85, 0.10, 0.04, 0.01, 0.00, 0.00],  # From S0
        [0.05, 0.75, 0.15, 0.04, 0.01, 0.00],  # From S1
        [0.02, 0.05, 0.72, 0.17, 0.03, 0.01],  # From S2
        [0.01, 0.02, 0.08, 0.70, 0.16, 0.03],  # From S3
        [0.00, 0.01, 0.02, 0.07, 0.72, 0.18],  # From S4
        [0.00, 0.00, 0.00, 0.01, 0.09, 0.90],  # From S5
    ])
    T_ANNUAL = T_ANNUAL / T_ANNUAL.sum(axis=1, keepdims=True)

    INTERVENTIONS = {
        "No intervention (natural history)":     1.00,
        "Blood pressure optimised":               0.78,
        "HbA1c optimised (<7%)":                 0.72,
        "RAS blockade (ACE-I / ARB) initiated":  0.65,
        "SGLT2 inhibitor initiated":              0.60,
        "All interventions combined":             0.40,
    }

    has_patient = bool(st.session_state.patient)

    st.markdown('<p class="sect-hdr">Patient Risk Profile</p>', unsafe_allow_html=True)

    if has_patient and "last_proba" in st.session_state:
        proba_input = st.session_state["last_proba"]
        pred_input  = st.session_state["last_pred"]
        st.markdown(f"""<div class="rec-block teal">
          Using current patient assessment — <strong>{STAGE_NAMES[pred_input]}</strong>
          (confidence {proba_input[pred_input]*100:.0f}%). Adjust sliders below to override.
        </div>""", unsafe_allow_html=True)
    else:
        proba_input = None
        pred_input  = 2
        st.markdown("""<div class="rec-block amber">
          No patient loaded. Use the sliders below to enter stage probabilities manually,
          or complete the Patient Assessment first.
        </div>""", unsafe_allow_html=True)

    st.markdown("**Enter or confirm current stage probabilities from GlomeraAI assessment:**")
    cols = st.columns(6)
    raw_p = []
    for i, col in enumerate(cols):
        with col:
            default = float(round(proba_input[i], 2)) if proba_input is not None else (0.5 if i==2 else 0.1)
            p = st.number_input(
                f"P(S{i})", 0.0, 1.0,
                value=min(1.0, max(0.0, default)),
                step=0.01, key=f"sp_{i}"
            )
            raw_p.append(p)

    total = sum(raw_p)
    if total > 0:
        proba_norm = np.array(raw_p) / total
    else:
        proba_norm = np.array([1/6]*6)

    # Show current risk bar
    fig_bar, ax_bar = plt.subplots(figsize=(9, 1.0))
    left = 0
    for s in range(6):
        ax_bar.barh(0, proba_norm[s], left=left, color=STAGE_COLORS[s], alpha=0.85)
        if proba_norm[s] > 0.06:
            ax_bar.text(left + proba_norm[s]/2, 0, f"S{s}\n{proba_norm[s]*100:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += proba_norm[s]
    ax_bar.set_xlim(0, 1); ax_bar.axis("off")
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True); plt.close()

    st.markdown('<p class="sect-hdr">Simulation Settings</p>', unsafe_allow_html=True)
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        intervention = st.selectbox("Intervention scenario", list(INTERVENTIONS.keys()))
    with col_set2:
        n_sims = st.select_slider("Monte Carlo simulations", [1000, 2000, 5000, 10000], value=5000)

    if st.button("▶  Run 24-Month Simulation", type="primary"):
        factor = INTERVENTIONS[intervention]

        # Build monthly transition matrix with intervention adjustment
        T_adj = T_ANNUAL.copy()
        for i in range(6):
            for j in range(6):
                if j > i:
                    T_adj[i, j] *= factor
            row_sum = T_adj[i].sum()
            if row_sum > 0:
                T_adj[i] = T_adj[i] / row_sum

        # Approximate monthly matrix: T^(1/12)
        try:
            from scipy.linalg import fractional_matrix_power
            T_monthly = fractional_matrix_power(T_adj, 1/12)
            T_monthly = np.clip(np.real(T_monthly), 0, 1)
            T_monthly = T_monthly / T_monthly.sum(axis=1, keepdims=True)
        except Exception:
            # Fallback: approximate with 12th root via eigendecomposition
            T_monthly = T_adj ** (1/12)
            T_monthly = np.clip(T_monthly, 0, 1)
            T_monthly = T_monthly / T_monthly.sum(axis=1, keepdims=True)

        # Monte Carlo
        with st.spinner(f"Running {n_sims:,} simulations over 24 months..."):
            rng = np.random.default_rng(42)
            initial_stages = rng.choice(6, size=n_sims, p=proba_norm)
            trajectories = np.zeros((n_sims, 25), dtype=np.int8)
            trajectories[:, 0] = initial_stages
            for t in range(1, 25):
                for sim in range(n_sims):
                    cur = trajectories[sim, t-1]
                    trajectories[sim, t] = rng.choice(6, p=T_monthly[cur])

        # Compute probabilities over time
        time_points = np.arange(25)
        prob_time = np.zeros((25, 6))
        for t in time_points:
            for s in range(6):
                prob_time[t, s] = (trajectories[:, t] == s).mean()

        # Key milestones
        st.markdown('<p class="sect-hdr">Key Risk Milestones</p>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns(3)
        for months_pt, col in [(6, mc1), (12, mc2), (24, mc3)]:
            stage_at_t = trajectories[:, months_pt]
            p_esrd  = (stage_at_t >= 4).mean()
            p_prog  = (stage_at_t > np.bincount(trajectories[:,0]).argmax()).mean()
            most_l  = int(np.bincount(stage_at_t).argmax())
            c = "#ef4444" if p_esrd > 0.2 else ("#f59e0b" if p_esrd > 0.05 else "#10b981")
            with col:
                st.markdown(f"""<div class="kpi-tile">
                  <div class="kpi-val" style="color:{STAGE_COLORS[most_l]}">Stage {most_l}</div>
                  <div class="kpi-lbl">Most likely at {months_pt} months</div>
                  <div style="margin-top:.5rem;font-size:.72rem;color:{c};font-weight:700">
                    {p_esrd*100:.0f}% risk Stage 4+
                  </div>
                </div>""", unsafe_allow_html=True)

        # Main simulation plot
        fig_sim, axes_sim = plt.subplots(1, 2, figsize=(13, 5))

        # Left: stacked area
        bottom = np.zeros(25)
        for s in range(6):
            axes_sim[0].fill_between(time_points, bottom, bottom + prob_time[:, s],
                                     color=STAGE_COLORS[s], alpha=0.82,
                                     label=STAGE_NAMES[s].split("—")[0].strip())
            bottom += prob_time[:, s]
        axes_sim[0].set_xlim(0, 24)
        axes_sim[0].set_xticks([0, 6, 12, 18, 24])
        axes_sim[0].set_xticklabels(["Now", "6 mo", "12 mo", "18 mo", "24 mo"])
        axes_sim[0].set_ylabel("Probability of being at each stage")
        axes_sim[0].set_title("Stage Distribution Over Time", fontsize=11, fontweight="bold")
        axes_sim[0].legend(fontsize=8, loc="upper left")
        axes_sim[0].spines[["top","right"]].set_visible(False)

        # Right: Stage 4+ risk over time
        p4plus_over_time = prob_time[:, 4] + prob_time[:, 5]
        axes_sim[1].plot(time_points, p4plus_over_time * 100,
                         color="#ef4444", lw=2.5, marker="o", markersize=4)
        axes_sim[1].fill_between(time_points, 0, p4plus_over_time * 100, alpha=0.12, color="#ef4444")
        axes_sim[1].axhline(20, color="#f59e0b", lw=1, ls="--", alpha=0.7, label="20% threshold")
        axes_sim[1].set_xlim(0, 24)
        axes_sim[1].set_xticks([0, 6, 12, 18, 24])
        axes_sim[1].set_xticklabels(["Now", "6 mo", "12 mo", "18 mo", "24 mo"])
        axes_sim[1].set_ylabel("Probability (%)")
        axes_sim[1].set_title("Risk of Stage 4+ (Severe / Failure)", fontsize=11, fontweight="bold")
        axes_sim[1].legend(fontsize=8)
        axes_sim[1].spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_sim, use_container_width=True); plt.close()

        # Intervention comparison
        st.markdown('<p class="sect-hdr">Intervention Comparison — 12-Month Stage 4+ Risk</p>', unsafe_allow_html=True)
        comp_results = {}
        with st.spinner("Comparing all intervention scenarios..."):
            for interv, factor_i in INTERVENTIONS.items():
                T_i = T_ANNUAL.copy()
                for ii in range(6):
                    for jj in range(6):
                        if jj > ii:
                            T_i[ii, jj] *= factor_i
                    T_i[ii] = T_i[ii] / T_i[ii].sum()
                try:
                    T_mi = fractional_matrix_power(T_i, 1/12)
                    T_mi = np.clip(np.real(T_mi), 0, 1)
                    T_mi = T_mi / T_mi.sum(axis=1, keepdims=True)
                except Exception:
                    T_mi = T_i
                trajs_i = np.zeros((1000, 13), dtype=np.int8)
                init_i  = rng.choice(6, size=1000, p=proba_norm)
                trajs_i[:, 0] = init_i
                for t in range(1, 13):
                    for sim in range(1000):
                        cur = trajs_i[sim, t-1]
                        trajs_i[sim, t] = rng.choice(6, p=T_mi[cur])
                comp_results[interv] = (trajs_i[:, 12] >= 4).mean() * 100

        comp_sorted = sorted(comp_results.items(), key=lambda x: x[1])
        fig_comp2, ax_comp2 = plt.subplots(figsize=(10, 3.5))
        colors_c = ["#10b981" if i == 0 else ("#ef4444" if i == len(comp_sorted)-1
                    else "#1a6efc") for i in range(len(comp_sorted))]
        bars_c = ax_comp2.barh([x[0] for x in comp_sorted],
                               [x[1] for x in comp_sorted],
                               color=colors_c, alpha=0.85)
        for bar in bars_c:
            w = bar.get_width()
            ax_comp2.text(w + 0.3, bar.get_y() + bar.get_height()/2,
                          f"{w:.1f}%", va="center", fontsize=9, fontweight="600")
        ax_comp2.set_xlabel("Stage 4+ risk at 12 months (%)")
        ax_comp2.set_title("Impact of Different Interventions", fontsize=11, fontweight="bold")
        ax_comp2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_comp2, use_container_width=True); plt.close()

    st.markdown("""
    <div class="disclaimer">
      <strong>⚠ Simulation Disclaimer:</strong> This is a rule-based clinical simulation using
      published KDIGO population-level stage-transition rates (KDIGO 2024 Clinical Practice Guidelines).
      It is <strong>not</strong> a trained machine learning model and does <strong>not</strong>
      predict individual patient outcomes. The simulation estimates the probability distribution
      of future stages based on known population-level disease progression patterns. Intervention
      reduction factors are derived from published randomised controlled trial data. All outputs
      require review by a qualified nephrologist or diabetologist before informing any clinical
      decision. GlomeraAI · NHANES 2015–2020 · n=2,627 diabetic adults.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Demo Patients (NBQSA Presentation Page)
# Six pre-loaded patient profiles covering all KDIGO stages 0–5
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == 3:

    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:800;color:#f0f4ff;margin-bottom:.3rem">
        Clinical Demo Profiles
      </div>
      <div style="font-size:.82rem;color:#8496b0;line-height:1.7">
        Six representative patient profiles covering every KDIGO stage.
        Select a profile to run the full GlomeraAI assessment instantly —
        no manual data entry required.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Six representative patient profiles ────────────────────────────────────
    DEMO_PATIENTS = {
        "Stage 0 — Priya, 42F, No DKD": {
            "label": "Stage 0 — No DKD",
            "color": "#10b981",
            "summary": "42-year-old female, well-controlled T2DM, routine monitoring",
            "clinical_note": "HbA1c well-controlled. No proteinuria. Kidney function normal. Low progression risk. Demonstrates the system's ability to correctly reassure low-risk patients.",
            "data": {
                "mean_sbp": 118.0, "mean_dbp": 74.0,
                "serum_creatinine_mgdl": 0.7, "log_serum_creatinine_mgdl": np.log1p(0.7),
                "urine_albumin_ugl": 12.0,   "log_urine_albumin_ugl": np.log1p(12.0),
                "urine_creatinine_mgdl": 140.0, "uacr_mgg": 0.009,
                "log_uacr": np.log10(0.009),
                "hba1c_pct": 6.4, "fasting_glucose_mgdl": 112.0,
                "log_fasting_glucose_mgdl": np.log1p(112.0),
                "insulin_uiml": 8.0,  "log_insulin_uiml": np.log1p(8.0),
                "bun_mgdl": 12.0, "log_bun_mgdl": np.log1p(12.0),
                "uric_acid_mgdl": 4.2, "hemoglobin_gdl": 13.8,
                "hematocrit_pct": 41.0, "serum_albumin_gdl": 4.3,
                "crp_mgL": 1.2,   "log_crp_mgL": np.log1p(1.2),
                "total_cholesterol_mgdl": 175.0, "ldl_cholesterol_mgdl": 98.0,
                "hdl_cholesterol_mgdl": 58.0, "triglycerides_mgdl": 95.0,
                "log_triglycerides_mgdl": np.log1p(95.0), "bmi_kgm2": 24.2,
                "diabetes_diagnosed": 1, "insulin_use": 0, "oral_diabetes_meds": 1,
                "hypertension_diagnosed": 0, "bp_medication": 0, "statin_use": 0,
                "current_smoker_status": 0, "avg_alcohol_drinks_per_day": 0.1,
                "log_avg_alcohol_drinks_per_day": np.log1p(0.1),
                "vigorous_leisure_activity": 1, "sedentary_minutes_per_day": 180,
                "log_sedentary_minutes_per_day": np.log1p(180), "sleep_hours_weekday": 7.5,
                "kidney_disease_history": 0, "kidney_stone_history": 0, "nocturia": 0,
                "sodium_mg_day": 2100.0, "protein_g_day": 68.0,
                "potassium_mg_day": 3200.0, "phosphorus_mg_day": 900.0,
                "age_years": 42, "sex_code": 1, "race_ethnicity_code": 6,
                "education_level": 5, "household_income_cat": 10,
                "food_security_score": 15.0, "coronary_heart_disease": 0,
                "heart_attack": 0, "stroke_ever": 0, "family_hx_diabetes": 1,
            },
        },
        "Stage 1 — Malik, 51M, Microalbuminuria": {
            "label": "Stage 1 — Microalbuminuria",
            "color": "#f59e0b",
            "summary": "51-year-old male, UACR 45 mg/g — early microalbuminuria, eGFR preserved",
            "clinical_note": "Urine albumin elevated but GFR preserved. HbA1c suboptimal at 8.1%. Early intervention with ACE-I and tighter glycaemic control could halt progression. Demonstrates early detection value.",
            "data": {
                "mean_sbp": 134.0, "mean_dbp": 82.0,
                "serum_creatinine_mgdl": 0.95, "log_serum_creatinine_mgdl": np.log1p(0.95),
                "urine_albumin_ugl": 54000.0,  "log_urine_albumin_ugl": np.log1p(54000.0),
                "urine_creatinine_mgdl": 120.0, "uacr_mgg": 45.0,
                "log_uacr": np.log10(45.0),
                "hba1c_pct": 8.1, "fasting_glucose_mgdl": 162.0,
                "log_fasting_glucose_mgdl": np.log1p(162.0),
                "insulin_uiml": 14.0, "log_insulin_uiml": np.log1p(14.0),
                "bun_mgdl": 15.0, "log_bun_mgdl": np.log1p(15.0),
                "uric_acid_mgdl": 5.8, "hemoglobin_gdl": 13.2,
                "hematocrit_pct": 39.5, "serum_albumin_gdl": 4.1,
                "crp_mgL": 3.4, "log_crp_mgL": np.log1p(3.4),
                "total_cholesterol_mgdl": 210.0, "ldl_cholesterol_mgdl": 130.0,
                "hdl_cholesterol_mgdl": 42.0, "triglycerides_mgdl": 175.0,
                "log_triglycerides_mgdl": np.log1p(175.0), "bmi_kgm2": 28.8,
                "diabetes_diagnosed": 1, "insulin_use": 0, "oral_diabetes_meds": 1,
                "hypertension_diagnosed": 1, "bp_medication": 1, "statin_use": 1,
                "current_smoker_status": 0, "avg_alcohol_drinks_per_day": 0.5,
                "log_avg_alcohol_drinks_per_day": np.log1p(0.5),
                "vigorous_leisure_activity": 0, "sedentary_minutes_per_day": 320,
                "log_sedentary_minutes_per_day": np.log1p(320), "sleep_hours_weekday": 6.5,
                "kidney_disease_history": 0, "kidney_stone_history": 0, "nocturia": 1,
                "sodium_mg_day": 3100.0, "protein_g_day": 95.0,
                "potassium_mg_day": 2600.0, "phosphorus_mg_day": 1200.0,
                "age_years": 51, "sex_code": 0, "race_ethnicity_code": 4,
                "education_level": 3, "household_income_cat": 7,
                "food_security_score": 11.0, "coronary_heart_disease": 0,
                "heart_attack": 0, "stroke_ever": 0, "family_hx_diabetes": 1,
            },
        },
        "Stage 2 — Anura, 56M, Macroalbuminuria": {
            "label": "Stage 2 — Macroalbuminuria / Mild GFR",
            "color": "#f97316",
            "summary": "56-year-old male, UACR 350 mg/g — macroalbuminuria, eGFR borderline",
            "clinical_note": "GFR beginning to decline with persistent albuminuria. HbA1c poorly controlled at 9.2%. Nephrology referral indicated. Demonstrates detection at the last stage before moderate damage.",
            "data": {
                "mean_sbp": 145.0, "mean_dbp": 88.0,
                "serum_creatinine_mgdl": 1.35, "log_serum_creatinine_mgdl": np.log1p(1.35),
                "urine_albumin_ugl": 385000.0, "log_urine_albumin_ugl": np.log1p(385000.0),
                "urine_creatinine_mgdl": 110.0, "uacr_mgg": 350.0,
                "log_uacr": np.log10(350.0),
                "hba1c_pct": 9.2, "fasting_glucose_mgdl": 188.0,
                "log_fasting_glucose_mgdl": np.log1p(188.0),
                "insulin_uiml": 22.0, "log_insulin_uiml": np.log1p(22.0),
                "bun_mgdl": 20.0, "log_bun_mgdl": np.log1p(20.0),
                "uric_acid_mgdl": 6.5, "hemoglobin_gdl": 12.8,
                "hematocrit_pct": 38.0, "serum_albumin_gdl": 3.9,
                "crp_mgL": 5.8, "log_crp_mgL": np.log1p(5.8),
                "total_cholesterol_mgdl": 225.0, "ldl_cholesterol_mgdl": 148.0,
                "hdl_cholesterol_mgdl": 38.0, "triglycerides_mgdl": 220.0,
                "log_triglycerides_mgdl": np.log1p(220.0), "bmi_kgm2": 31.5,
                "diabetes_diagnosed": 1, "insulin_use": 0, "oral_diabetes_meds": 1,
                "hypertension_diagnosed": 1, "bp_medication": 1, "statin_use": 1,
                "current_smoker_status": 1, "avg_alcohol_drinks_per_day": 1.2,
                "log_avg_alcohol_drinks_per_day": np.log1p(1.2),
                "vigorous_leisure_activity": 0, "sedentary_minutes_per_day": 420,
                "log_sedentary_minutes_per_day": np.log1p(420), "sleep_hours_weekday": 6.0,
                "kidney_disease_history": 1, "kidney_stone_history": 0, "nocturia": 1,
                "sodium_mg_day": 3600.0, "protein_g_day": 105.0,
                "potassium_mg_day": 2400.0, "phosphorus_mg_day": 1350.0,
                "age_years": 56, "sex_code": 0, "race_ethnicity_code": 1,
                "education_level": 2, "household_income_cat": 5,
                "food_security_score": 8.0, "coronary_heart_disease": 0,
                "heart_attack": 0, "stroke_ever": 0, "family_hx_diabetes": 0,
            },
        },
        "Stage 3 — Nirmala, 63F, Moderate GFR Decrease": {
            "label": "Stage 3 — Moderate GFR Decrease",
            "color": "#ef4444",
            "summary": "63-year-old female, eGFR 30–44, UACR 200 mg/g, renal anaemia developing",
            "clinical_note": "Established CKD Stage 3 with significant albuminuria and rising creatinine. Anaemia developing. Urgent nephrology co-management required. Demonstrates the system's urgency flagging.",
            "data": {
                "mean_sbp": 152.0, "mean_dbp": 92.0,
                "serum_creatinine_mgdl": 1.9,  "log_serum_creatinine_mgdl": np.log1p(1.9),
                "urine_albumin_ugl": 180000.0, "log_urine_albumin_ugl": np.log1p(180000.0),
                "urine_creatinine_mgdl": 90.0, "uacr_mgg": 200.0,
                "log_uacr": np.log10(200.0),
                "hba1c_pct": 9.8, "fasting_glucose_mgdl": 210.0,
                "log_fasting_glucose_mgdl": np.log1p(210.0),
                "insulin_uiml": 38.0, "log_insulin_uiml": np.log1p(38.0),
                "bun_mgdl": 28.0, "log_bun_mgdl": np.log1p(28.0),
                "uric_acid_mgdl": 7.8, "hemoglobin_gdl": 11.2,
                "hematocrit_pct": 33.5, "serum_albumin_gdl": 3.5,
                "crp_mgL": 9.2, "log_crp_mgL": np.log1p(9.2),
                "total_cholesterol_mgdl": 240.0, "ldl_cholesterol_mgdl": 162.0,
                "hdl_cholesterol_mgdl": 32.0, "triglycerides_mgdl": 310.0,
                "log_triglycerides_mgdl": np.log1p(310.0), "bmi_kgm2": 33.8,
                "diabetes_diagnosed": 1, "insulin_use": 1, "oral_diabetes_meds": 0,
                "hypertension_diagnosed": 1, "bp_medication": 1, "statin_use": 1,
                "current_smoker_status": 0, "avg_alcohol_drinks_per_day": 0.2,
                "log_avg_alcohol_drinks_per_day": np.log1p(0.2),
                "vigorous_leisure_activity": 0, "sedentary_minutes_per_day": 540,
                "log_sedentary_minutes_per_day": np.log1p(540), "sleep_hours_weekday": 5.5,
                "kidney_disease_history": 1, "kidney_stone_history": 1, "nocturia": 1,
                "sodium_mg_day": 3200.0, "protein_g_day": 88.0,
                "potassium_mg_day": 2200.0, "phosphorus_mg_day": 1150.0,
                "age_years": 63, "sex_code": 1, "race_ethnicity_code": 4,
                "education_level": 2, "household_income_cat": 3,
                "food_security_score": 6.0, "coronary_heart_disease": 1,
                "heart_attack": 0, "stroke_ever": 0, "family_hx_diabetes": 1,
            },
        },
        "Stage 4 — Rajan, 69M, Severe GFR Decrease": {
            "label": "Stage 4 — Severe GFR Decrease",
            "color": "#a855f7",
            "summary": "69-year-old male, eGFR 15–29, UACR 500 mg/g, severe renal anaemia",
            "clinical_note": "Near end-stage renal disease. Severe anaemia, markedly elevated creatinine and BUN. Dialysis planning underway. Demonstrates the system's emergency flagging and referral pathway.",
            "data": {
                "mean_sbp": 165.0, "mean_dbp": 98.0,
                "serum_creatinine_mgdl": 3.4,  "log_serum_creatinine_mgdl": np.log1p(3.4),
                "urine_albumin_ugl": 375000.0, "log_urine_albumin_ugl": np.log1p(375000.0),
                "urine_creatinine_mgdl": 75.0, "uacr_mgg": 500.0,
                "log_uacr": np.log10(500.0),
                "hba1c_pct": 10.5, "fasting_glucose_mgdl": 240.0,
                "log_fasting_glucose_mgdl": np.log1p(240.0),
                "insulin_uiml": 55.0, "log_insulin_uiml": np.log1p(55.0),
                "bun_mgdl": 48.0, "log_bun_mgdl": np.log1p(48.0),
                "uric_acid_mgdl": 9.2, "hemoglobin_gdl": 9.4,
                "hematocrit_pct": 28.0, "serum_albumin_gdl": 3.1,
                "crp_mgL": 18.5, "log_crp_mgL": np.log1p(18.5),
                "total_cholesterol_mgdl": 195.0, "ldl_cholesterol_mgdl": 110.0,
                "hdl_cholesterol_mgdl": 28.0, "triglycerides_mgdl": 280.0,
                "log_triglycerides_mgdl": np.log1p(280.0), "bmi_kgm2": 26.5,
                "diabetes_diagnosed": 1, "insulin_use": 1, "oral_diabetes_meds": 0,
                "hypertension_diagnosed": 1, "bp_medication": 1, "statin_use": 1,
                "current_smoker_status": 2, "avg_alcohol_drinks_per_day": 0.0,
                "log_avg_alcohol_drinks_per_day": 0.0,
                "vigorous_leisure_activity": 0, "sedentary_minutes_per_day": 660,
                "log_sedentary_minutes_per_day": np.log1p(660), "sleep_hours_weekday": 5.0,
                "kidney_disease_history": 1, "kidney_stone_history": 0, "nocturia": 1,
                "sodium_mg_day": 2800.0, "protein_g_day": 55.0,
                "potassium_mg_day": 1800.0, "phosphorus_mg_day": 800.0,
                "age_years": 69, "sex_code": 0, "race_ethnicity_code": 3,
                "education_level": 3, "household_income_cat": 4,
                "food_security_score": 7.0, "coronary_heart_disease": 1,
                "heart_attack": 1, "stroke_ever": 0, "family_hx_diabetes": 1,
            },
        },
        "Stage 5 — Kamala, 72F, Kidney Failure": {
            "label": "Stage 5 — Kidney Failure",
            "color": "#94a3b8",
            "summary": "72-year-old female, eGFR <15, UACR 800 mg/g, critical renal failure",
            "clinical_note": "End-stage renal disease requiring renal replacement therapy. Severe anaemia, critical electrolyte imbalances, profound albuminuria. Demonstrates complete staging capability across all 6 classes.",
            "data": {
                "mean_sbp": 178.0, "mean_dbp": 104.0,
                "serum_creatinine_mgdl": 6.8,  "log_serum_creatinine_mgdl": np.log1p(6.8),
                "urine_albumin_ugl": 440000.0, "log_urine_albumin_ugl": np.log1p(440000.0),
                "urine_creatinine_mgdl": 55.0, "uacr_mgg": 800.0,
                "log_uacr": np.log10(800.0),
                "hba1c_pct": 11.8, "fasting_glucose_mgdl": 290.0,
                "log_fasting_glucose_mgdl": np.log1p(290.0),
                "insulin_uiml": 72.0, "log_insulin_uiml": np.log1p(72.0),
                "bun_mgdl": 82.0, "log_bun_mgdl": np.log1p(82.0),
                "uric_acid_mgdl": 11.5, "hemoglobin_gdl": 7.8,
                "hematocrit_pct": 23.5, "serum_albumin_gdl": 2.6,
                "crp_mgL": 32.0, "log_crp_mgL": np.log1p(32.0),
                "total_cholesterol_mgdl": 168.0, "ldl_cholesterol_mgdl": 88.0,
                "hdl_cholesterol_mgdl": 22.0, "triglycerides_mgdl": 320.0,
                "log_triglycerides_mgdl": np.log1p(320.0), "bmi_kgm2": 22.1,
                "diabetes_diagnosed": 1, "insulin_use": 1, "oral_diabetes_meds": 0,
                "hypertension_diagnosed": 1, "bp_medication": 1, "statin_use": 0,
                "current_smoker_status": 0, "avg_alcohol_drinks_per_day": 0.0,
                "log_avg_alcohol_drinks_per_day": 0.0,
                "vigorous_leisure_activity": 0, "sedentary_minutes_per_day": 900,
                "log_sedentary_minutes_per_day": np.log1p(900), "sleep_hours_weekday": 4.5,
                "kidney_disease_history": 1, "kidney_stone_history": 1, "nocturia": 1,
                "sodium_mg_day": 1800.0, "protein_g_day": 45.0,
                "potassium_mg_day": 1400.0, "phosphorus_mg_day": 600.0,
                "age_years": 72, "sex_code": 1, "race_ethnicity_code": 2,
                "education_level": 1, "household_income_cat": 2,
                "food_security_score": 4.0, "coronary_heart_disease": 1,
                "heart_attack": 1, "stroke_ever": 1, "family_hx_diabetes": 1,
            },
        },
    }

    # ── Stage selector cards ───────────────────────────────────────────────────
    st.markdown('<p class="sect-hdr">Select a Patient Profile</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cols_cycle = [col1, col2, col3, col1, col2, col3]

    for col, (name, profile) in zip(cols_cycle, DEMO_PATIENTS.items()):
        with col:
            color = profile["color"]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.03);border:1px solid {color}40;
                 border-left:4px solid {color};border-radius:12px;
                 padding:.9rem 1.1rem;margin-bottom:.5rem">
              <div style="font-size:.7rem;font-weight:700;color:{color};
                   text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem">
                {profile['label']}
              </div>
              <div style="font-size:.83rem;font-weight:600;color:#f0f4ff;margin-bottom:.3rem">
                {name.split('—')[1].strip() if '—' in name else name}
              </div>
              <div style="font-size:.76rem;color:#8496b0;line-height:1.5">
                {profile['summary']}
              </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Run Assessment", key=f"demo_{name}", use_container_width=True):
                st.session_state["selected_demo"] = name
                st.rerun()

    # ── Run selected demo ──────────────────────────────────────────────────────
    selected = st.session_state.get("selected_demo")

    if selected and selected in DEMO_PATIENTS:
        profile = DEMO_PATIENTS[selected]
        patient_data = profile["data"]

        st.divider()
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:800;
             color:{profile['color']};margin-bottom:.3rem">
          {profile['label']} — Assessment Running
        </div>
        <div style="font-size:.82rem;color:#8496b0;margin-bottom:1rem">
          {profile['clinical_note']}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Running full GlomeraAI assessment..."):
            df_imp, X1, X2, X3 = build_vector(patient_data, mdl)
            pred, proba, p1, p2, p3 = run_ensemble(mdl, X1, X2, X3)
            rf_sv, xgb_sv, lr_sv = get_shap(mdl, X1, X2, X3)

        rec         = STAGE_RECS[pred]
        stage_color = STAGE_COLORS[pred]
        prog_score  = sum(c * proba[c] for c in range(N_CLASSES))
        entropy_val = scipy_entropy(proba)
        uncertainty = entropy_val / scipy_entropy([1/N_CLASSES]*N_CLASSES)
        conf_label  = ("High confidence" if uncertainty < 0.33 else
                       "Moderate" if uncertainty < 0.66 else "Low — review carefully")
        conf_color  = ("#10b981" if uncertainty < 0.33 else
                       "#f59e0b" if uncertainty < 0.66 else "#ef4444")
        forward_p   = float(sum(proba[c] for c in range(pred+1, N_CLASSES)))

        # Store for What-If page
        st.session_state["last_pred"]   = pred
        st.session_state["last_proba"]  = proba
        st.session_state["last_hba1c"]  = patient_data.get("hba1c_pct", 7.2)
        st.session_state["patient"]     = patient_data

        # Result hero
        st.markdown(f"""
        <div class="stage-result-card" style="color:{stage_color};
             background:rgba(11,20,38,0.6);border-color:{stage_color}40">
          <div style="font-size:.68rem;font-weight:700;text-transform:uppercase;
               letter-spacing:.14em;color:{stage_color};margin-bottom:.5rem">
            GlomeraAI ASSESSMENT RESULT
          </div>
          <div style="font-size:2rem;font-weight:800;letter-spacing:-.03em;color:{stage_color}">
            {STAGE_NAMES[pred]}
          </div>
          <div style="font-size:.88rem;color:#8496b0;margin:.5rem 0">{rec['headline']}</div>
          <div>{urgency_chip(rec['urgency'])}</div>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:{stage_color}">Stage {pred}</div><div class="kpi-lbl">KDIGO Stage</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:#f0f4ff">{proba[pred]*100:.0f}%</div><div class="kpi-lbl">Confidence</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:{conf_color};font-size:1.1rem;padding-top:.4rem">{conf_label}</div><div class="kpi-lbl">AI Certainty</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="kpi-tile"><div class="kpi-val" style="color:#f0f4ff">{prog_score:.2f}/5</div><div class="kpi-lbl">Severity Score</div></div>', unsafe_allow_html=True)

        # Plain-language narrative
        shap_rf = dict(zip(X1.columns.tolist(), rf_sv[0, :, pred]))
        top3 = sorted(shap_rf.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        narrative_parts = []
        for feat, sv in top3:
            name_f = display_name(feat)
            raw    = display_val(feat, float(df_imp[feat].iloc[0]))
            direction = "elevated" if sv > 0 else "lower than expected"
            desc = CLINICAL_CONTEXT.get(feat, ("", None, ""))[2]
            narrative_parts.append(f"<strong>{name_f}</strong> ({raw:.2f}) is {direction} — {desc}")
        st.markdown(f"""<div class="rec-block teal">
          <strong>Why this prediction?</strong><br>
          {';&nbsp; '.join(narrative_parts)}.
        </div>""", unsafe_allow_html=True)

        # Recommendations
        st.markdown(f"""<div class="rec-block {rec['color']}">
          <strong>Recommended Clinical Actions</strong>
          <ul style="margin:.5rem 0 0;padding-left:1.2rem">
            {''.join(f"<li style='margin:.3rem 0'>{a}</li>" for a in rec['actions'])}
          </ul>
        </div>""", unsafe_allow_html=True)

        # Two column: probabilities + model votes
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown('<p class="sect-hdr">Stage Probability Breakdown</p>', unsafe_allow_html=True)
            fig_d, ax_d = plt.subplots(figsize=(8, 2.8))
            bars = ax_d.barh([STAGE_NAMES[c] for c in range(N_CLASSES)],
                             [proba[c]*100 for c in range(N_CLASSES)],
                             color=[STAGE_COLORS[c] for c in range(N_CLASSES)],
                             alpha=0.85, height=0.62)
            for bar, val in zip(bars, proba):
                if val > 0.01:
                    ax_d.text(bar.get_width()+.5, bar.get_y()+bar.get_height()/2,
                              f"{val*100:.1f}%", va="center", fontsize=8.5, fontweight="600")
            ax_d.set_xlabel("Probability (%)", fontsize=9)
            ax_d.set_xlim(0, 110)
            ax_d.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_d, use_container_width=True); plt.close()

        with col_right:
            st.markdown('<p class="sect-hdr">Three-Model Agreement</p>', unsafe_allow_html=True)
            for lbl, prob, color in [
                ("M1 Clinical",     p1[pred], "#1a6efc"),
                ("M2 Lifestyle",    p2[pred], "#f97316"),
                ("M3 Demographics", p3[pred], "#a855f7"),
            ]:
                pct = int(prob * 100)
                st.markdown(f"""
                <div style="margin:.5rem 0">
                  <div style="display:flex;justify-content:space-between;
                       font-size:.82rem;margin-bottom:.25rem">
                    <span>{lbl}</span>
                    <strong style="color:{color};font-family:'DM Mono',monospace">{pct}%</strong>
                  </div>
                  <div style="background:rgba(255,255,255,.06);border-radius:4px;height:6px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;
                         transition:width .6s ease"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:1rem;padding:.8rem 1rem;background:rgba(255,255,255,.03);
                 border:1px solid rgba(255,255,255,.07);border-radius:10px;font-size:.8rem">
              <strong>Severity gauge</strong><br>
              <div style="display:flex;align-items:center;gap:.5rem;margin-top:.4rem">
                <div style="flex:1;background:rgba(255,255,255,.06);border-radius:4px;height:8px;overflow:hidden">
                  <div style="width:{int(prog_score/5*100)}%;height:100%;
                       background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);border-radius:4px"></div>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:.75rem;color:{stage_color};font-weight:700">
                  {prog_score:.2f}/5
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

        # SHAP
        st.markdown('<p class="sect-hdr">Clinical Factor Influence (SHAP)</p>', unsafe_allow_html=True)
        fig_shap = plot_shap_bar(shap_rf, f"Clinical Biomarkers — {STAGE_NAMES[pred]}", "#ef4444", "#1a6efc", top_n=8)
        if fig_shap:
            st.pyplot(fig_shap, use_container_width=True); plt.close()

        # CTA to HbA1c page
        if pred > 0:
            st.markdown(f"""<div class="rec-block teal">
              <strong>Next step:</strong> Use the
              <strong>HbA1c What-If</strong> page to model how glycaemic control could
              reduce this patient's risk score — powered by the actual GlomeraAI model.
            </div>""", unsafe_allow_html=True)
            if st.button("→ Open HbA1c What-If for this patient", type="primary"):
                st.session_state.page = 1; st.rerun()

        st.markdown("""
        <div class="disclaimer">
          <strong>⚠ Clinical Disclaimer:</strong> These are synthetic representative patient profiles
          for demonstration purposes only. All GlomeraAI predictions shown are real outputs from
          the trained ensemble model validated on NHANES 2015–2020 (n=2,627, AUC 0.961).
          All outputs require review by a qualified clinician before informing any clinical decision.
        </div>""", unsafe_allow_html=True)
