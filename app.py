"""
GlomeraAI-DKD Clinical Decision Support System
Multimodel, Explainable and Fair AI for Predicting Diabetic Nephropathy Progression
"""
 
import io
import warnings
import pickle
import requests
 
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
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
 
[data-testid="stSidebarNav"] { display: none !important; }
 
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #f8fafc;
}
 
.app-header {
    background: #fff;
    border-bottom: 1px solid #e2e8f0;
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #10b981);
}
.app-header-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-header-sub {
    font-size: 0.72rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    margin: 0;
}
 
.metric-tile {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    transition: transform .12s, box-shadow .12s;
}
.metric-tile:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,.08);
}
.metric-tile .value {
    font-size: 2.1rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: .3rem;
}
.metric-tile .label {
    font-size: .7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 600;
}
 
.sect-hdr {
    font-size: .78rem;
    font-weight: 700;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: .1em;
    border-left: 3px solid #0ea5e9;
    padding-left: .6rem;
    margin: 1.5rem 0 .8rem;
}
 
.verdict-yes {
    background: #fff1f2;
    border: 2px solid #f43f5e;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin: 1rem 0;
}
.verdict-no {
    background: #f0fdf4;
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin: 1rem 0;
}
.verdict-watch {
    background: #fffbeb;
    border: 2px solid #f59e0b;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin: 1rem 0;
}
.verdict-title {
    font-size: 1.25rem;
    font-weight: 800;
    margin-bottom: .5rem;
}
.verdict-body {
    font-size: .88rem;
    line-height: 1.6;
}
 
.rec-box {
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin: .8rem 0;
    border-left: 4px solid;
    font-size: .87rem;
    line-height: 1.6;
}
.rec-box.blue   { background: #eff6ff; border-color: #3b82f6; color: #1e3a5f; }
.rec-box.green  { background: #f0fdf4; border-color: #22c55e; color: #14532d; }
.rec-box.amber  { background: #fffbeb; border-color: #f59e0b; color: #78350f; }
.rec-box.red    { background: #fff1f2; border-color: #f43f5e; color: #881337; }
.rec-box.purple { background: #faf5ff; border-color: #8b5cf6; color: #3b0764; }
.rec-box.slate  { background: #f1f5f9; border-color: #475569; color: #1e293b; }
 
.model-badge {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .7rem;
    border: 1px solid #e2e8f0;
    border-left: 4px solid;
}
.model-badge.rf  { border-left-color: #3b82f6; background: #f8fbff; }
.model-badge.xgb { border-left-color: #f97316; background: #fffaf5; }
.model-badge.lr  { border-left-color: #8b5cf6; background: #faf8ff; }
.model-badge h4  { margin: 0 0 .35rem; font-size: .9rem; font-weight: 700; }
.model-badge p   { margin: 0; font-size: .8rem; color: #475569; }
 
.stepper {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: .5rem 0 1.5rem;
}
.step-item { display: flex; align-items: center; }
.step-dot {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .78rem; font-weight: 700;
}
.step-dot.done   { background: #22c55e; color: #fff; }
.step-dot.active { background: #0ea5e9; color: #fff; box-shadow: 0 0 0 4px #bae6fd; }
.step-dot.todo   { background: #e2e8f0; color: #94a3b8; }
.step-label { font-size: .68rem; color: #64748b; margin-top: .3rem; text-align: center; font-weight: 500; }
.step-line { width: 44px; height: 2px; background: #e2e8f0; margin: 0 4px; }
.step-line.done { background: #22c55e; }
 
.chip {
    display: inline-block;
    padding: .2rem .65rem;
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 600;
    margin: .15rem;
}
.chip.green  { background: #dcfce7; color: #166534; }
.chip.red    { background: #fee2e2; color: #991b1b; }
.chip.blue   { background: #dbeafe; color: #1e40af; }
.chip.amber  { background: #fef3c7; color: #92400e; }
.chip.purple { background: #ede9fe; color: #5b21b6; }
 
div[data-testid="stSidebarContent"] { background: #0f172a !important; }
div[data-testid="stSidebarContent"] .stMarkdown,
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] p { color: #94a3b8 !important; }
div[data-testid="stSidebarContent"] h3 { color: #38bdf8 !important; }
div[data-testid="stSidebarContent"] h4 { color: #e2e8f0 !important; }
 
div[data-testid="stSidebarContent"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: .45rem .75rem;
    font-size: .84rem;
    font-weight: 500;
    color: #94a3b8;
    transition: background .15s, color .15s;
    cursor: pointer;
}
div[data-testid="stSidebarContent"] .stButton > button:hover {
    background: rgba(255,255,255,.06) !important;
    color: #e2e8f0 !important;
    border: none !important;
}
div[data-testid="stSidebarContent"] .nav-active .stButton > button {
    background: rgba(14,165,233,.18) !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
    border-left: 3px solid #38bdf8 !important;
}
 
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #f1f5f9;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: .83rem;
    font-weight: 600;
    padding: .45rem 1rem;
    color: #64748b;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #fff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
 
.disclaimer {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: .76rem;
    color: #64748b;
    margin-top: 1.5rem;
    line-height: 1.6;
}
 
.driver-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: .5rem .8rem;
    border-radius: 8px;
    margin-bottom: .4rem;
    font-size: .84rem;
}
.driver-row.risk { background: #fff1f2; }
.driver-row.prot { background: #f0fdf4; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
N_CLASSES = 6
 
STAGE_NAMES = {
    0: "No DKD",
    1: "Stage 1 — Microalbuminuria",
    2: "Stage 2 — Mild GFR Decrease",
    3: "Stage 3 — Moderate GFR Decrease",
    4: "Stage 4 — Severe GFR Decrease",
    5: "Stage 5 — Kidney Failure",
}
STAGE_COLORS = {
    0: "#22c55e", 1: "#f59e0b", 2: "#f97316",
    3: "#ef4444", 4: "#8b5cf6", 5: "#0f172a",
}
STAGE_BG = {
    0: "#f0fdf4", 1: "#fffbeb", 2: "#fff7ed",
    3: "#fff1f2", 4: "#faf5ff", 5: "#f1f5f9",
}
 
STAGE_RECOMMENDATIONS = {
    0: {
        "headline": "No diabetic kidney disease detected at this time.",
        "actions": [
            "Continue annual kidney monitoring (eGFR + urine albumin test)",
            "Keep blood sugar well controlled (HbA1c target below 7%)",
            "Maintain blood pressure below 130/80 mmHg",
            "Encourage healthy weight, diet, and regular activity",
        ],
        "urgency": "Routine follow-up",
    },
    1: {
        "headline": "Early kidney damage — protein leaking into urine detected.",
        "actions": [
            "Start blood pressure medication (ACE inhibitor or ARB) to protect kidneys",
            "Set blood pressure target below 130/80 mmHg",
            "Tighten blood sugar control (HbA1c below 7%)",
            "Repeat urine albumin test in 3 months to confirm",
        ],
        "urgency": "Within 4 weeks",
    },
    2: {
        "headline": "Kidney filtering function is beginning to decline.",
        "actions": [
            "Refer to a kidney specialist (nephrologist)",
            "Reduce dietary protein intake (0.8 g per kg body weight per day)",
            "Review and adjust any medications that may strain the kidneys",
            "Optimise blood pressure and blood sugar control",
        ],
        "urgency": "Within 2–4 weeks",
    },
    3: {
        "headline": "Significant kidney function loss — specialist care needed now.",
        "actions": [
            "Urgent nephrology co-management required",
            "Begin discussions about future kidney replacement options",
            "Check and treat anaemia (low haemoglobin)",
            "Strict fluid, salt, and potassium management",
        ],
        "urgency": "Urgent — within 1 week",
    },
    4: {
        "headline": "Severe kidney function loss — prepare for kidney replacement.",
        "actions": [
            "Plan for dialysis access surgery (fistula creation)",
            "Strict fluid and electrolyte management with dietitian",
            "Full multidisciplinary team care (nephrology, dietitian, social work)",
            "Transplant evaluation referral",
        ],
        "urgency": "Immediate specialist review",
    },
    5: {
        "headline": "Kidney failure — renal replacement therapy required.",
        "actions": [
            "Initiate dialysis or arrange kidney transplant evaluation",
            "Full renal replacement therapy pathway",
            "Intensive symptom management and multidisciplinary support",
            "Palliative care consultation if appropriate",
        ],
        "urgency": "Emergency/Immediate",
    },
}
 
CLINICAL_CONTEXT = {
    "log_urine_albumin_ugl":         ("Urine Albumin",        "expm1", "Protein leaking into urine — key early damage marker"),
    "log_serum_creatinine_mgdl":     ("Creatinine",           "expm1", "Waste product in blood — rises when kidneys cannot filter"),
    "log_bun_mgdl":                  ("Blood Urea (BUN)",     "expm1", "Blood urea nitrogen — rises when kidneys are failing"),
    "uacr_mgg":                      ("Urine Albumin Ratio",  None,    "Albumin-to-creatinine ratio — key DKD screening marker"),
    "hba1c_pct":                     ("HbA1c",                None,    "3-month average blood glucose — higher means poorer diabetes control"),
    "hemoglobin_gdl":                ("Haemoglobin",          None,    "Anaemia develops as kidney function declines"),
    "hematocrit_pct":                ("Haematocrit",          None,    "Low values signal kidney-related anaemia"),
    "uric_acid_mgdl":                ("Uric Acid",            None,    "Elevated levels accelerate kidney damage"),
    "log_crp_mgL":                   ("CRP (Inflammation)",   "expm1", "Inflammation marker — drives kidney disease progression"),
    "mean_sbp":                      ("Systolic BP",          None,    "High blood pressure directly damages kidney blood vessels"),
    "mean_dbp":                      ("Diastolic BP",         None,    "Elevated pressure adds strain to kidney filters"),
    "bmi_kgm2":                      ("BMI",                  None,    "Obesity increases kidney disease risk"),
    "log_triglycerides_mgdl":        ("Triglycerides",        "expm1", "High blood fats worsen kidney disease outlook"),
    "hdl_cholesterol_mgdl":          ("HDL Cholesterol",      None,    "Low HDL independently predicts kidney decline"),
    "log_insulin_uiml":              ("Insulin Level",        "expm1", "High fasting insulin signals insulin resistance"),
    "serum_albumin_gdl":             ("Serum Albumin",        None,    "Low albumin indicates poor nutrition from kidney disease"),
    "urine_creatinine_mgdl":         ("Urine Creatinine",     None,    "Used to calculate urine albumin ratio (UACR)"),
    "kidney_disease_history":        ("Kidney Disease Hx",    None,    "Previous kidney disease strongly predicts DKD"),
    "hypertension_diagnosed":        ("Hypertension",         None,    "Uncontrolled blood pressure accelerates kidney damage"),
    "kidney_stone_history":          ("Kidney Stone Hx",      None,    "Associated with chronic kidney disease progression"),
    "insulin_use":                   ("Insulin Use",          None,    "Signals advanced diabetes requiring insulin"),
    "phosphorus_mg_day":             ("Dietary Phosphorus",   None,    "High phosphorus intake linked to kidney decline"),
    "potassium_mg_day":              ("Dietary Potassium",    None,    "Potassium management important in kidney disease"),
    "current_smoker_status":         ("Smoking",              None,    "Smoking damages kidney blood vessels"),
    "sedentary_minutes_per_day":     ("Sedentary Time",       None,    "Physical inactivity worsens metabolic health"),
    "log_sedentary_minutes_per_day": ("Sedentary Time",       "expm1", "Physical inactivity worsens metabolic health"),
    "vigorous_leisure_activity":     ("Physical Activity",    None,    "Protective — regular exercise reduces kidney disease risk"),
    "age_years":                     ("Patient Age",          None,    "Older age is a primary kidney disease risk factor"),
    "race_ethnicity_code":           ("Ethnicity",            None,    "Ethnic differences in kidney disease risk"),
    "sex_code":                      ("Sex",                  None,    "Sex differences in kidney disease risk trajectory"),
    "heart_attack":                  ("Heart Attack Hx",      None,    "Cardiovascular history worsens kidney disease prognosis"),
    "stroke_ever":                   ("Stroke History",       None,    "Stroke indicates widespread vascular disease"),
    "coronary_heart_disease":        ("Heart Disease",        None,    "Shared disease process with kidney disease"),
    "family_hx_diabetes":            ("Family Hx Diabetes",   None,    "Genetic predisposition to diabetes and kidney disease"),
    "education_level":               ("Education Level",      None,    "Proxy for healthcare access and disease management"),
    "household_income_cat":          ("Income Level",         None,    "Socioeconomic factors affect disease management"),
}
 
RF_EXPECTED  = ["log_serum_creatinine_mgdl", "log_urine_albumin_ugl", "log_bun_mgdl", "hba1c_pct", "hemoglobin_gdl"]
XGB_EXPECTED = ["hypertension_diagnosed", "kidney_disease_history", "kidney_stone_history", "insulin_use", "phosphorus_mg_day"]
LR_EXPECTED  = ["age_years", "race_ethnicity_code", "sex_code", "bmi_kgm2", "heart_attack"]
 
SHAP_EXPLAIN = (
    "The bars show how much each factor <b>increases</b> (red) or "
    "<b>decreases</b> (blue) the AI's assessment for this patient. "
    "Longer bars mean stronger influence."
)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [("current_page", 0), ("patient_data", {})]:
    if key not in st.session_state:
        st.session_state[key] = default
 
# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_from_hf():
    try:
        url = st.secrets["MODEL_URL"]
    except (KeyError, FileNotFoundError):
        return None, "MODEL_URL secret not configured. Add it to your Streamlit secrets."
    try:
        resp = requests.get(url, timeout=240)
        resp.raise_for_status()
        return pickle.load(io.BytesIO(resp.content)), None
    except Exception as exc:
        return None, f"Could not load model: {exc}"

# ── Core helpers ───────────────────────────────────────────────────────────────
def display_value(feat, raw_val):
    if feat in CLINICAL_CONTEXT:
        name, transform, _ = CLINICAL_CONTEXT[feat]
        if transform == "expm1":
            return name, float(np.expm1(raw_val))
        return name, raw_val
    return feat.replace("_", " ").title(), raw_val
 
 
def build_patient_vector(inputs, mdl):
    ALL_FEATS   = mdl["ALL_FEATS"]
    M1, M2, M3  = mdl["M1"], mdl["M2"], mdl["M3"]
    knn_imputer = mdl["knn_imputer"]
    row    = {f: np.nan for f in ALL_FEATS}
    row.update({k: v for k, v in inputs.items() if k in row})
    df_row = pd.DataFrame([row])[ALL_FEATS]
    df_imp = pd.DataFrame(knn_imputer.transform(df_row), columns=ALL_FEATS)
    X1 = df_imp[[c for c in M1 if c in df_imp.columns]]
    X2 = df_imp[[c for c in M2 if c in df_imp.columns]]
    X3 = df_imp[[c for c in M3 if c in df_imp.columns]]
    return df_imp, X1, X2, X3
 
 
def run_ensemble(mdl, X1, X2, X3):
    X3_sc     = mdl["lr_scaler"].transform(X3)
    p1        = mdl["rf_pipeline"].predict_proba(X1)
    p2        = mdl["xgb_pipeline"].predict_proba(X2)
    p3        = mdl["lr_model"].predict_proba(X3_sc)
    meta_feat = np.hstack([p1, p2, p3])
    meta_sc   = mdl["meta_scaler"].transform(meta_feat)
    pred      = mdl["meta_lr"].predict(meta_sc)[0]
    proba     = mdl["meta_lr"].predict_proba(meta_sc)[0]
    return int(pred), proba, p1[0], p2[0], p3[0]
 
 
def compute_shap_patient(mdl, X1, X2, X3):
    rf_exp  = shap.TreeExplainer(mdl["rf_clf"])
    xgb_exp = shap.TreeExplainer(mdl["xgb_clf"])
    rf_sv   = np.array(rf_exp.shap_values(X1))
    xgb_sv  = np.array(xgb_exp.shap_values(X2))
    try:
        X3_sc = mdl["lr_scaler"].transform(X3)
        coef  = mdl["lr_model"].coef_
        lr_sv = coef[:, np.newaxis, :] * np.array(X3_sc)[np.newaxis, :, :]
    except Exception:
        lr_sv = None
    return rf_sv, xgb_sv, lr_sv
 
 
def progression_risk_profile(rf_sv, X1, pred_class):
    if pred_class >= N_CLASSES - 1:
        return None, None
    shap_next = rf_sv[0, :, pred_class + 1] if rf_sv.ndim == 3 else rf_sv[:, :, pred_class + 1][0]
    feat_shap = dict(zip(X1.columns.tolist(), shap_next))
    risk = sorted([(f, v) for f, v in feat_shap.items() if v > 0 and f in CLINICAL_CONTEXT],
                  key=lambda x: x[1], reverse=True)[:5]
    prot = sorted([(f, v) for f, v in feat_shap.items() if v < 0 and f in CLINICAL_CONTEXT],
                  key=lambda x: x[1])[:3]
    return risk, prot
 
 
def plot_shap_bar(shap_dict, title, color_pos="#ef4444", color_neg="#3b82f6",
                  top_n=10, highlight_feats=None):
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    if not items:
        return None
    labels = [display_value(f, 0)[0] for f, _ in items]
    vals   = [sv for _, sv in items]
    colors = []
    for feat, sv in items:
        if sv > 0:
            colors.append("#f97316" if (highlight_feats and feat in highlight_feats) else color_pos)
        else:
            colors.append("#0ea5e9" if (highlight_feats and feat in highlight_feats) else color_neg)
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.48)))
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], alpha=0.88, height=0.62)
    ax.axvline(0, color="#374151", lw=0.9, ls="--")
    ax.set_xlabel("Influence on prediction (SHAP value)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="700")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    return fig
 
 
def stepper_html(current):
    steps = ["Clinical Data", "Lifestyle", "Demographics", "Results"]
    parts = []
    for i, label in enumerate(steps):
        cls      = "done" if i < current else ("active" if i == current else "todo")
        num      = "✓"   if i < current else str(i + 1)
        line_cls = "done" if i < current else ""
        dot      = f'<div class="step-dot {cls}">{num}</div>'
        lbl      = f'<div class="step-label">{label}</div>'
        inner    = f'<div style="display:flex;flex-direction:column;align-items:center">{dot}{lbl}</div>'
        conn     = f'<div class="step-line {line_cls}"></div>' if i < len(steps) - 1 else ""
        parts.append(f'<div class="step-item">{inner}{conn}</div>')
    return '<div class="stepper">' + "".join(parts) + "</div>"
 
 
def urgency_chip(text):
    color_map = {
        "Routine": "green", "Within 4": "blue", "Within 2": "amber",
        "Urgent": "red", "Immediate": "red", "Emergency": "red",
    }
    chip_color = next((v for k, v in color_map.items() if text.startswith(k)), "blue")
    return f'<span class="chip {chip_color}">{text}</span>'

# ── App header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div style="font-size:1.8rem;line-height:1">🔬</div>
  <div>
    <p class="app-header-title">GlomeraAI</p>
    <p class="app-header-sub">DKD Clinical Decision Support &nbsp;·&nbsp; Multimodel Ensemble · KDIGO 2022 Staging · Fairness-Audited · Explainable AI</p>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── Load model ─────────────────────────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    mdl, load_err = load_model_from_hf()
 
if load_err:
    st.error(f"Model loading failed. {load_err}")
    st.info("Ensure MODEL_URL is set in Streamlit secrets pointing to DKD_complete_artifacts.pkl.")
    st.stop()
 
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔬 GlomeraAI")
    st.markdown("---")
 
    current = st.session_state.current_page
    nav_labels = {
        0: "Clinical Data",
        1: "Lifestyle",
        2: "Demographics",
        3: "Results",
        4: "Demo",
    }
 
    for pg, label in nav_labels.items():
        is_active = pg == current
        if is_active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        prefix = "▶ " if is_active else ("✓ " if pg < current else "  ")
        if st.button(f"{prefix}{label}", key=f"nav_btn_{pg}"):
            st.session_state.current_page = pg
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("#### How this AI works")
    st.markdown("""
    <div style="font-size:.78rem;line-height:1.7;color:#94a3b8">
    Three models analyse different aspects of your patient and combine their findings:<br><br>
    <span style="color:#60a5fa;font-weight:600">Clinical Model (M1)</span><br>
    Blood tests and kidney markers<br><br>
    <span style="color:#fb923c;font-weight:600">Lifestyle Model (M2)</span><br>
    Medications, activity, diet<br><br>
    <span style="color:#a78bfa;font-weight:600">Demographics Model (M3)</span><br>
    Age, sex, ethnicity, history<br><br>
    <span style="color:#94a3b8;font-weight:600">Combined Decision</span><br>
    All three models vote; the meta-learner weighs their agreement
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:.7rem;color:#475569;text-align:center">
    NHANES 2015–2020 · n ≈ 6,600 · AUC 0.96
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Clinical Data
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.current_page == 0:
    st.markdown(stepper_html(0), unsafe_allow_html=True)
 
    st.markdown("""
    <div class="model-badge rf">
      <h4>Clinical Model (M1) — Blood Tests & Kidney Markers</h4>
      <p>Key indicators: Creatinine · Urine Albumin · Blood Urea · HbA1c · Haemoglobin</p>
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
        serum_cr  = st.number_input("Serum Creatinine (mg/dL)",    0.2,  20.0,   1.1,   step=0.1, key="serum_cr")
        urine_alb = st.number_input("Urine Albumin (ug/L)",         0.5,  5000.0, 25.0,  step=0.5, key="urine_alb")
        urine_cr  = st.number_input("Urine Creatinine (mg/dL)",     5.0,  3000.0, 120.0,           key="urine_cr")
 
    with col2:
        st.markdown("**Metabolic & Blood Chemistry**")
        hba1c      = st.number_input("HbA1c (%)",                 4.0,  20.0,  7.2,  step=0.1, key="hba1c")
        fasting_gl = st.number_input("Fasting Glucose (mg/dL)",   40.0, 600.0, 145.0,           key="fasting_gl")
        insulin    = st.number_input("Fasting Insulin (uIU/mL)",  0.0,  300.0, 12.0,            key="insulin")
        bun        = st.number_input("Blood Urea Nitrogen (mg/dL)",2.0, 150.0, 16.0,            key="bun")
        uric_acid  = st.number_input("Uric Acid (mg/dL)",         1.0,  20.0,  5.8,  step=0.1, key="uric_acid")
 
        st.markdown("**Blood Count & Other**")
        hemoglobin = st.number_input("Haemoglobin (g/dL)",  4.0,  22.0, 13.5, step=0.1, key="hemoglobin")
        hematocrit = st.number_input("Haematocrit (%)",    10.0,  65.0, 40.0, step=0.5, key="hematocrit")
        serum_alb  = st.number_input("Serum Albumin (g/dL)", 1.0,  6.0,  4.1, step=0.1, key="serum_alb")
        crp        = st.number_input("CRP (mg/L)",          0.0,  200.0, 3.5, step=0.1, key="crp")
 
    st.markdown("**Lipid Panel & Body**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tot_chol = st.number_input("Total Cholesterol (mg/dL)", 50.0,  600.0, 195.0, key="tot_chol")
        ldl      = st.number_input("LDL Cholesterol (mg/dL)",   20.0,  500.0, 115.0, key="ldl")
    with col_b:
        hdl  = st.number_input("HDL Cholesterol (mg/dL)", 10.0,  200.0, 48.0,  key="hdl")
        trig = st.number_input("Triglycerides (mg/dL)",   20.0, 3000.0, 145.0, key="trig")
    with col_c:
        bmi = st.number_input("BMI (kg/m²)", 12.0, 80.0, 28.5, step=0.1, key="bmi")
 
    mean_sbp = round((sbp1 + sbp2 + sbp3) / 3, 1)
    mean_dbp = round((dbp1 + dbp2 + dbp3) / 3, 1)
    uacr     = round(urine_alb / (urine_cr * 10), 2) if urine_cr > 0 else 0.0
 
    st.markdown(f"""
    <div class="rec-box blue">
      <strong>Auto-calculated values:</strong><br>
      Mean Systolic BP = <strong>{mean_sbp} mmHg</strong> &nbsp;·&nbsp;
      Mean Diastolic BP = <strong>{mean_dbp} mmHg</strong> &nbsp;·&nbsp;
      UACR = <strong>{uacr:.1f} mg/g</strong>
    </div>""", unsafe_allow_html=True)
 
    st.session_state.patient_data.update({
        "mean_sbp": mean_sbp, "mean_dbp": mean_dbp,
        "serum_creatinine_mgdl": serum_cr,
        "log_serum_creatinine_mgdl": np.log1p(serum_cr),
        "urine_albumin_ugl": urine_alb,
        "log_urine_albumin_ugl": np.log1p(urine_alb),
        "urine_creatinine_mgdl": urine_cr,
        "uacr_mgg": uacr,
        "log_uacr": np.log10(max(uacr, 0.01)),
        "hba1c_pct": hba1c,
        "fasting_glucose_mgdl": fasting_gl,
        "log_fasting_glucose_mgdl": np.log1p(fasting_gl),
        "insulin_uiml": insulin,
        "log_insulin_uiml": np.log1p(insulin),
        "bun_mgdl": bun,
        "log_bun_mgdl": np.log1p(bun),
        "uric_acid_mgdl": uric_acid,
        "hemoglobin_gdl": hemoglobin,
        "hematocrit_pct": hematocrit,
        "serum_albumin_gdl": serum_alb,
        "crp_mgL": crp,
        "log_crp_mgL": np.log1p(crp),
        "total_cholesterol_mgdl": tot_chol,
        "ldl_cholesterol_mgdl": ldl,
        "hdl_cholesterol_mgdl": hdl,
        "triglycerides_mgdl": trig,
        "log_triglycerides_mgdl": np.log1p(trig),
        "bmi_kgm2": bmi,
    })
 
    _, col_btn, _ = st.columns([2, 3, 2])
    with col_btn:
        if st.button("Continue to Lifestyle", type="primary", use_container_width=True):
            st.session_state.current_page = 1
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Lifestyle
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == 1:
    st.markdown(stepper_html(1), unsafe_allow_html=True)
 
    st.markdown("""
    <div class="model-badge xgb">
      <h4>Lifestyle Model (M2) — Medications, Activity & Diet</h4>
      <p>Key indicators: Kidney Disease History · Hypertension · Insulin Use · Physical Activity · Phosphorus</p>
    </div>""", unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Medications & Diagnoses**")
        dm_dx       = st.selectbox("Diabetes Diagnosed?",                  [1, 0], format_func=lambda x: "Yes" if x else "No", key="dm_dx")
        insulin_use = st.selectbox("Currently using insulin?",             [0, 1], format_func=lambda x: "Yes" if x else "No", key="insulin_use")
        oral_meds   = st.selectbox("Taking oral diabetes medications?",    [0, 1], format_func=lambda x: "Yes" if x else "No", key="oral_meds")
        htn_dx      = st.selectbox("Diagnosed with high blood pressure?",  [0, 1], format_func=lambda x: "Yes" if x else "No", key="htn_dx")
        bp_med      = st.selectbox("Taking blood pressure medication?",    [0, 1], format_func=lambda x: "Yes" if x else "No", key="bp_med")
        statin      = st.selectbox("Taking statins?",                      [0, 1], format_func=lambda x: "Yes" if x else "No", key="statin")
 
        st.markdown("**Kidney History**")
        kidney_hx    = st.selectbox("History of kidney disease?",  [0, 1], format_func=lambda x: "Yes" if x else "No", key="kidney_hx")
        kidney_stone = st.selectbox("History of kidney stones?",   [0, 1], format_func=lambda x: "Yes" if x else "No", key="kidney_stone")
        nocturia     = st.selectbox("Waking at night to urinate?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="nocturia")
 
    with col2:
        st.markdown("**Physical Activity & Lifestyle**")
        smoker      = st.selectbox("Smoking status", [0, 1, 2],
                                   format_func=lambda x: ["Never smoked", "Former smoker", "Current smoker"][x],
                                   key="smoker")
        alcohol     = st.number_input("Average alcohol (drinks/day)", 0.0, 20.0, 0.3, step=0.1, key="alcohol")
        vig_leisure = st.selectbox("Does the patient do vigorous exercise?", [0, 1],
                                   format_func=lambda x: "Yes" if x else "No", key="vig_leisure")
        sedentary   = st.number_input("Sedentary time per day (minutes)", 0, 1440, 300, key="sedentary")
        sleep_h     = st.number_input("Average weekday sleep (hours)", 2.0, 14.0, 7.0, step=0.5, key="sleep_h")
 
        st.markdown("**Dietary Intake (daily averages)**")
        sodium     = st.number_input("Dietary Sodium (mg/day)",     100.0, 15000.0, 2800.0, key="sodium")
        protein    = st.number_input("Dietary Protein (g/day)",       5.0,   400.0,   85.0, key="protein")
        potassium  = st.number_input("Dietary Potassium (mg/day)",  200.0,  8000.0, 2800.0, key="potassium")
        phosphorus = st.number_input("Dietary Phosphorus (mg/day)", 100.0,  4000.0, 1100.0, key="phosphorus")
 
    st.session_state.patient_data.update({
        "diabetes_diagnosed": dm_dx,
        "insulin_use": insulin_use,
        "oral_diabetes_meds": oral_meds,
        "hypertension_diagnosed": htn_dx,
        "bp_medication": bp_med,
        "statin_use": statin,
        "current_smoker_status": smoker,
        "avg_alcohol_drinks_per_day": alcohol,
        "log_avg_alcohol_drinks_per_day": np.log1p(alcohol),
        "vigorous_leisure_activity": vig_leisure,
        "sedentary_minutes_per_day": sedentary,
        "log_sedentary_minutes_per_day": np.log1p(sedentary),
        "sleep_hours_weekday": sleep_h,
        "kidney_disease_history": kidney_hx,
        "kidney_stone_history": kidney_stone,
        "nocturia": nocturia,
        "sodium_mg_day": sodium,
        "protein_g_day": protein,
        "potassium_mg_day": potassium,
        "phosphorus_mg_day": phosphorus,
    })
 
    col_l, _, col_r = st.columns([2, 2, 2])
    with col_l:
        if st.button("Back", use_container_width=True):
            st.session_state.current_page = 0; st.rerun()
    with col_r:
        if st.button("Continue to Demographics", type="primary", use_container_width=True):
            st.session_state.current_page = 2; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Demographics
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == 2:
    st.markdown(stepper_html(2), unsafe_allow_html=True)
 
    st.markdown("""
    <div class="model-badge lr">
      <h4>Demographics Model (M3) — Patient Background & History</h4>
      <p>Key indicators: Age · Sex · Ethnicity · BMI · Heart Attack History</p>
    </div>""", unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Basic Information**")
        age  = st.number_input("Patient age (years)", 18, 100, 58, key="age")
        sex  = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female", key="sex")
        race = st.selectbox("Race/Ethnicity", [1, 2, 3, 4, 6, 7],
                            format_func=lambda x: {1: "Mexican American", 2: "Other Hispanic",
                                                   3: "Non-Hispanic White", 4: "Non-Hispanic Black",
                                                   6: "Non-Hispanic Asian", 7: "Other/Multiracial"}[x],
                            key="race")
        st.markdown("**Socioeconomic Background**")
        education = st.selectbox("Highest education level", [1, 2, 3, 4, 5], index=3,
                                 format_func=lambda x: {1: "Less than 9th grade", 2: "Some high school",
                                                         3: "High school / GED", 4: "Some college",
                                                         5: "College graduate or higher"}[x],
                                 key="education")
        income = st.selectbox("Household income category",
                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15], index=7,
                              format_func=lambda x: {1: "< $5,000", 2: "$5–10k", 3: "$10–15k",
                                                      4: "$15–20k", 5: "$20–25k", 6: "$25–35k",
                                                      7: "$35–45k", 8: "$45–55k", 9: "$55–65k",
                                                      10: "$65–75k", 14: "$75–100k",
                                                      15: "Over $100,000"}[x],
                              key="income")
        food_sec = st.number_input("Food security score (0–18)", 0.0, 18.0, 10.0, step=1.0, key="food_sec")
 
    with col2:
        st.markdown("**Cardiovascular & Family History**")
        chd       = st.selectbox("History of coronary heart disease?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="chd")
        heart_att = st.selectbox("History of heart attack?",           [0, 1], format_func=lambda x: "Yes" if x else "No", key="heart_att")
        stroke    = st.selectbox("History of stroke?",                 [0, 1], format_func=lambda x: "Yes" if x else "No", key="stroke")
        fam_hx_dm = st.selectbox("Family history of diabetes?",        [0, 1], format_func=lambda x: "Yes" if x else "No", key="fam_hx_dm")
 
    race_label = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                  4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other/Multiracial"}[race]
    sex_label  = "Female" if sex == 1 else "Male"
    st.markdown(f"""
    <div class="rec-box blue">
      <strong>Patient Profile:</strong> {age}-year-old {sex_label} · {race_label}
    </div>""", unsafe_allow_html=True)
 
    st.session_state.patient_data.update({
        "age_years": age, "sex_code": sex, "race_ethnicity_code": race,
        "education_level": education, "household_income_cat": income,
        "food_security_score": food_sec,
        "coronary_heart_disease": chd, "heart_attack": heart_att,
        "stroke_ever": stroke, "family_hx_diabetes": fam_hx_dm,
    })
 
    col_l, _, col_r = st.columns([2, 2, 2])
    with col_l:
        if st.button("Back", use_container_width=True):
            st.session_state.current_page = 1; st.rerun()
    with col_r:
        if st.button("Run AI Assessment", type="primary", use_container_width=True):
            st.session_state.current_page = 3; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Results
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.current_page == 3:
    st.markdown(stepper_html(3), unsafe_allow_html=True)
 
    with st.spinner("Running AI analysis..."):
        df_imp, X1, X2, X3 = build_patient_vector(st.session_state.patient_data, mdl)
        pred_stage, proba_vec, p1, p2, p3 = run_ensemble(mdl, X1, X2, X3)
        rf_sv, xgb_sv, lr_sv = compute_shap_patient(mdl, X1, X2, X3)
        risk_drivers, prot_drivers = progression_risk_profile(rf_sv, X1, pred_stage)
 
    stage_color = STAGE_COLORS[pred_stage]
    stage_bg    = STAGE_BG[pred_stage]
    stage_name  = STAGE_NAMES[pred_stage]
    rec         = STAGE_RECOMMENDATIONS[pred_stage]
    prog_score  = sum(c * proba_vec[c] for c in range(N_CLASSES))
 
    entropy_val  = scipy_entropy(proba_vec)
    max_entropy  = scipy_entropy([1 / N_CLASSES] * N_CLASSES)
    uncertainty  = entropy_val / max_entropy
    confidence_label = ("High confidence" if uncertainty < 0.33 else
                         "Moderate confidence" if uncertainty < 0.66 else
                         "Low confidence — review carefully")
    confidence_color = ("#22c55e" if uncertainty < 0.33 else
                         "#f59e0b" if uncertainty < 0.66 else "#ef4444")
 
    forward_prob = float(sum(proba_vec[c] for c in range(pred_stage + 1, N_CLASSES)))
    if pred_stage < N_CLASSES - 1:
        if forward_prob >= 0.30:
            progression_verdict, prog_icon = "YES", "Warning"
        elif forward_prob >= 0.15:
            progression_verdict, prog_icon = "POSSIBLE", "Notice"
        else:
            progression_verdict, prog_icon = "NO", "OK"
 
    # Result hero card
    st.markdown(f"""
    <div style="background:{stage_bg};border:2px solid {stage_color};border-radius:16px;
                padding:1.6rem 2rem;margin-bottom:1.2rem">
      <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
                  color:{stage_color};margin-bottom:.4rem">AI ASSESSMENT RESULT</div>
      <div style="font-size:2rem;font-weight:800;color:{stage_color};letter-spacing:-.03em">{stage_name}</div>
      <div style="font-size:.9rem;color:#475569;margin-top:.4rem">{rec['headline']}</div>
      <div style="margin-top:.8rem">{urgency_chip(rec['urgency'])}</div>
    </div>""", unsafe_allow_html=True)
 
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-tile"><div class="value" style="color:{stage_color}">Stage {pred_stage}</div><div class="label">KDIGO Stage (0–5)</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-tile"><div class="value" style="color:#0f172a">{proba_vec[pred_stage]*100:.0f}%</div><div class="label">Stage probability</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-tile"><div class="value" style="color:{confidence_color};font-size:1.1rem;padding-top:.5rem">{confidence_label.split(" ")[0]} {confidence_label.split(" ")[1]}</div><div class="label">AI confidence</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-tile"><div class="value" style="color:#0f172a">{prog_score:.1f}/5</div><div class="label">Severity score</div></div>', unsafe_allow_html=True)
 
    st.markdown("<div style='margin-top:.5rem'></div>", unsafe_allow_html=True)
 
    rec_color_map = {0: "green", 1: "blue", 2: "amber", 3: "red", 4: "red", 5: "slate"}
    actions_html  = "".join(f"<li style='margin:.3rem 0'>{a}</li>" for a in rec["actions"])
    st.markdown(f"""
    <div class="rec-box {rec_color_map[pred_stage]}">
      <strong>Recommended Clinical Actions:</strong>
      <ul style="margin:.5rem 0 0;padding-left:1.2rem">{actions_html}</ul>
    </div>""", unsafe_allow_html=True)
 
    st.divider()
 
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Summary", "Clinical Factors (M1)", "Lifestyle Factors (M2)",
        "Demographic Factors (M3)", "Progression Risk", "Fairness Audit",
    ])
 
    with tab1:
        st.markdown('<p class="sect-hdr">Patient Snapshot</p>', unsafe_allow_html=True)
        pd_data = st.session_state.patient_data
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            st.markdown("**Key Clinical Values**")
            st.write(f"HbA1c: **{pd_data.get('hba1c_pct','N/A'):.1f}%** (target <7%)")
            st.write(f"UACR: **{pd_data.get('uacr_mgg','N/A'):.1f} mg/g**")
            st.write(f"BP: **{pd_data.get('mean_sbp','N/A')}/{pd_data.get('mean_dbp','N/A')} mmHg**")
            st.write(f"BMI: **{pd_data.get('bmi_kgm2','N/A'):.1f} kg/m²**")
        with cs2:
            st.markdown("**Model Votes**")
            st.markdown(f"""
            <div style="font-size:.85rem">
              <div style="display:flex;justify-content:space-between;padding:.4rem .6rem;background:#f0f7ff;border-radius:8px;margin:.3rem 0">
                <span>Clinical (M1)</span><strong style="color:#3b82f6">{p1[pred_stage]*100:.0f}%</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:.4rem .6rem;background:#fff7f0;border-radius:8px;margin:.3rem 0">
                <span>Lifestyle (M2)</span><strong style="color:#f97316">{p2[pred_stage]*100:.0f}%</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:.4rem .6rem;background:#faf5ff;border-radius:8px;margin:.3rem 0">
                <span>Demographics (M3)</span><strong style="color:#8b5cf6">{p3[pred_stage]*100:.0f}%</strong>
              </div>
            </div>""", unsafe_allow_html=True)
        with cs3:
            st.markdown("**Patient Demographics**")
            sex_l  = "Female" if pd_data.get("sex_code") == 1 else "Male"
            race_l = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
                      4: "NH Black", 6: "NH Asian", 7: "Other/Multi"}.get(pd_data.get("race_ethnicity_code"), "N/A")
            age    = pd_data.get("age_years", 0)
            inc    = pd_data.get("household_income_cat", 8)
            inc_l  = "Low (<$25k)" if inc <= 5 else ("Middle ($25–64k)" if inc <= 9 else "High (≥$65k)")
            st.write(f"Age: **{age} years** · {sex_l}")
            st.write(f"Ethnicity: **{race_l}**")
            st.write(f"Income: **{inc_l}**")
 
        st.markdown('<p class="sect-hdr">Stage Probability Breakdown</p>', unsafe_allow_html=True)
        fig_dist, ax_dist = plt.subplots(figsize=(10, 2.8))
        bars = ax_dist.barh([STAGE_NAMES[c] for c in range(N_CLASSES)],
                            [proba_vec[c] * 100 for c in range(N_CLASSES)],
                            color=[STAGE_COLORS[c] for c in range(N_CLASSES)],
                            alpha=0.85, edgecolor="white", linewidth=0.5, height=0.65)
        for bar, val in zip(bars, proba_vec):
            ax_dist.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{val*100:.1f}%", va="center", fontsize=8.5, fontweight="600")
        ax_dist.set_xlabel("Probability (%)", fontsize=9)
        ax_dist.set_xlim(0, 108)
        ax_dist.spines[["top", "right"]].set_visible(False)
        ax_dist.tick_params(labelsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close()
 
    with tab2:
        st.markdown("""
        <div class="model-badge rf">
          <h4>Clinical Model (M1) — Which blood test results drove this prediction?</h4>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box blue">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
 
        shap_rf_dict = dict(zip(X1.columns.tolist(), rf_sv[0, :, pred_stage]))
        fig_rf = plot_shap_bar(shap_rf_dict, f"Clinical Factors — Stage {pred_stage} Prediction",
                               "#ef4444", "#3b82f6", highlight_feats=RF_EXPECTED)
        if fig_rf:
            st.pyplot(fig_rf, use_container_width=True); plt.close()
 
        st.markdown('<p class="sect-hdr">Clinical Factor Influence Across All Stages</p>', unsafe_allow_html=True)
        top_feats = sorted(shap_rf_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        feat_labels = [display_value(f, 0)[0] for f, _ in top_feats]
        shap_matrix = np.array([[rf_sv[0, list(X1.columns).index(f), c] for f, _ in top_feats]
                                 for c in range(N_CLASSES)])
        fig_heat, ax_heat = plt.subplots(figsize=(10, 4))
        im = ax_heat.imshow(shap_matrix, cmap="RdBu_r", aspect="auto",
                            norm=mcolors.TwoSlopeNorm(vcenter=0))
        ax_heat.set_xticks(range(len(feat_labels)))
        ax_heat.set_xticklabels(feat_labels, rotation=35, ha="right", fontsize=8.5)
        ax_heat.set_yticks(range(N_CLASSES))
        ax_heat.set_yticklabels([STAGE_NAMES[c] for c in range(N_CLASSES)], fontsize=8.5)
        plt.colorbar(im, ax=ax_heat, shrink=0.8, label="SHAP value")
        ax_heat.set_title("Clinical Factor Influence Heatmap", fontsize=10, fontweight="700")
        plt.tight_layout()
        st.pyplot(fig_heat, use_container_width=True); plt.close()
 
    with tab3:
        st.markdown("""
        <div class="model-badge xgb">
          <h4>Lifestyle Model (M2) — Which lifestyle factors drove this prediction?</h4>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box amber">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
 
        shap_xgb_dict = dict(zip(X2.columns.tolist(), xgb_sv[0, :, pred_stage]))
        fig_xgb = plot_shap_bar(shap_xgb_dict, f"Lifestyle Factors — Stage {pred_stage} Prediction",
                                "#f97316", "#0ea5e9", highlight_feats=XGB_EXPECTED)
        if fig_xgb:
            st.pyplot(fig_xgb, use_container_width=True); plt.close()
 
        st.markdown('<p class="sect-hdr">Model Agreement</p>', unsafe_allow_html=True)
        fig_comp, ax_comp = plt.subplots(figsize=(10, 3.2))
        x, w = np.arange(N_CLASSES), 0.26
        ax_comp.bar(x - w, p1 * 100, w, label="M1 Clinical",     color="#3b82f6", alpha=0.85)
        ax_comp.bar(x,     p2 * 100, w, label="M2 Lifestyle",    color="#f97316", alpha=0.85)
        ax_comp.bar(x + w, p3 * 100, w, label="M3 Demographics", color="#8b5cf6", alpha=0.85)
        ax_comp.set_xticks(x)
        ax_comp.set_xticklabels([f"Stage {c}" for c in range(N_CLASSES)], fontsize=8.5)
        ax_comp.set_ylabel("Confidence (%)", fontsize=9)
        ax_comp.set_title("Agreement Between All Three Models", fontsize=10, fontweight="700")
        ax_comp.legend(fontsize=8.5)
        ax_comp.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_comp, use_container_width=True); plt.close()
 
    with tab4:
        st.markdown("""
        <div class="model-badge lr">
          <h4>Demographics Model (M3) — Which patient background factors drove this prediction?</h4>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box purple">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
 
        if lr_sv is not None:
            try:
                shap_lr_dict = (dict(zip(X3.columns.tolist(), lr_sv[pred_stage, 0, :]))
                                if lr_sv.ndim == 3 else
                                dict(zip(X3.columns.tolist(), lr_sv[0, :])))
            except Exception:
                shap_lr_dict = {}
 
            if shap_lr_dict:
                fig_lr = plot_shap_bar(shap_lr_dict, f"Demographic Factors — Stage {pred_stage} Prediction",
                                       "#8b5cf6", "#06b6d4", highlight_feats=LR_EXPECTED)
                if fig_lr:
                    st.pyplot(fig_lr, use_container_width=True); plt.close()
 
            if lr_sv.ndim == 3 and lr_sv.shape[0] == N_CLASSES:
                st.markdown('<p class="sect-hdr">Demographic Factor Influence Across All Stages</p>', unsafe_allow_html=True)
                top_lr = sorted(shap_lr_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:7]
                lr_feat_labels = [display_value(f, 0)[0] for f, _ in top_lr]
                lr_shap_matrix = np.array([[lr_sv[c, 0, list(X3.columns).index(f)] for f, _ in top_lr]
                                            for c in range(N_CLASSES)])
                fig_lrheat, ax_lrheat = plt.subplots(figsize=(9, 4))
                im2 = ax_lrheat.imshow(lr_shap_matrix, cmap="RdBu_r", aspect="auto",
                                       norm=mcolors.TwoSlopeNorm(vcenter=0))
                ax_lrheat.set_xticks(range(len(lr_feat_labels)))
                ax_lrheat.set_xticklabels(lr_feat_labels, rotation=35, ha="right", fontsize=8.5)
                ax_lrheat.set_yticks(range(N_CLASSES))
                ax_lrheat.set_yticklabels([STAGE_NAMES[c] for c in range(N_CLASSES)], fontsize=8.5)
                plt.colorbar(im2, ax=ax_lrheat, shrink=0.8, label="SHAP value")
                ax_lrheat.set_title("Demographic Factor Influence Heatmap", fontsize=10, fontweight="700")
                plt.tight_layout()
                st.pyplot(fig_lrheat, use_container_width=True); plt.close()
 
            st.markdown("""
            <div class="rec-box purple">
              <strong>Note on demographic factors:</strong> Differences by age, ethnicity, or sex reflect
              documented population-level differences in kidney disease rates, not model bias.
              This system was independently audited for fairness across all demographic groups.
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Demographic SHAP analysis unavailable for this patient.")
 
    with tab5:
        st.markdown('<p class="sect-hdr">Will This Patient\'s Kidney Disease Progress?</p>', unsafe_allow_html=True)
 
        if pred_stage >= N_CLASSES - 1:
            st.markdown("""
            <div class="verdict-no">
              <div class="verdict-title" style="color:#14532d">Stage 5 — Kidney Replacement Required</div>
              <div class="verdict-body">Patient is at the most advanced stage. Focus on renal replacement therapy.</div>
            </div>""", unsafe_allow_html=True)
        else:
            next_stage_name = STAGE_NAMES[pred_stage + 1]
            if progression_verdict == "YES":
                vhtml = f"""<div class="verdict-yes">
                  <div class="verdict-title" style="color:#be123c">YES — DISEASE PROGRESSION IS LIKELY</div>
                  <div class="verdict-body">
                    <strong>{forward_prob*100:.0f}%</strong> chance of progressing beyond {stage_name}
                    toward <strong>{next_stage_name}</strong> if risk factors are not addressed.
                    Prioritise clinical actions and consider nephrology referral.
                  </div></div>"""
            elif progression_verdict == "POSSIBLE":
                vhtml = f"""<div class="verdict-watch">
                  <div class="verdict-title" style="color:#92400e">POSSIBLE — MONITOR CLOSELY</div>
                  <div class="verdict-body">
                    <strong>{forward_prob*100:.0f}%</strong> probability of worsening to <strong>{next_stage_name}</strong>.
                    With good management, progression may be preventable.
                    Increase monitoring frequency and review modifiable risk factors.
                  </div></div>"""
            else:
                vhtml = f"""<div class="verdict-no">
                  <div class="verdict-title" style="color:#14532d">NO — STAGE APPEARS STABLE</div>
                  <div class="verdict-body">
                    Only <strong>{forward_prob*100:.0f}%</strong> chance of progression at this time.
                    Current management appears effective. Continue routine monitoring.
                  </div></div>"""
            st.markdown(vhtml, unsafe_allow_html=True)
 
            if risk_drivers:
                st.markdown('<p class="sect-hdr">Factors Driving Progression Risk</p>', unsafe_allow_html=True)
                col_risk, col_prot = st.columns([3, 2])
                with col_risk:
                    r_feats = [display_value(f, 0)[0] for f, _ in risk_drivers]
                    r_vals  = [v for _, v in risk_drivers]
                    r_raws  = [display_value(f, X1.iloc[0].get(f, np.nan))[1] for f, _ in risk_drivers]
                    fig_prog, ax_prog = plt.subplots(figsize=(7, max(2.8, len(r_feats) * 0.6)))
                    bars_p = ax_prog.barh(r_feats[::-1], r_vals[::-1], color="#f43f5e", alpha=0.85, height=0.6)
                    for bar, raw in zip(bars_p, r_raws[::-1]):
                        ax_prog.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                                     f"{raw:.2f}", va="center", fontsize=8)
                    ax_prog.set_xlabel(f"Influence toward {next_stage_name}", fontsize=9)
                    ax_prog.set_title("Risk Factors to Address", fontsize=10, fontweight="700")
                    ax_prog.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_prog, use_container_width=True); plt.close()
                with col_prot:
                    if prot_drivers:
                        st.markdown("**Protective factors (slowing progression)**")
                        for feat, sv in prot_drivers:
                            disp_name, _ = display_value(feat, X1.iloc[0].get(feat, np.nan))
                            _, _, desc   = CLINICAL_CONTEXT.get(feat, (None, None, ""))
                            st.markdown(f"""
                            <div class="driver-row prot">
                              <span><strong>{disp_name}</strong><br>
                                <small style="color:#475569">{desc}</small></span>
                              <span style="color:#16a34a;font-weight:700">✓</span>
                            </div>""", unsafe_allow_html=True)
 
        st.markdown('<p class="sect-hdr">Disease Severity Gauge</p>', unsafe_allow_html=True)
        fig_gauge, ax_g = plt.subplots(figsize=(9, 1.8))
        ax_g.barh(["Score"], [prog_score], color=stage_color, alpha=0.85, height=0.45)
        ax_g.set_xlim(0, 5)
        for s in range(6):
            ax_g.axvline(s, color="#e2e8f0", lw=0.8)
            ax_g.text(s + 0.07, 0.28, f"Stage {s}", fontsize=7.5, color="#64748b")
        ax_g.axvline(pred_stage, color="#0f172a", lw=2, ls="--", alpha=0.7)
        ax_g.set_xlabel("Disease Severity (0 = No DKD → 5 = Kidney Failure)", fontsize=9)
        ax_g.spines[["top", "right", "left"]].set_visible(False)
        ax_g.set_yticks([])
        ax_g.text(prog_score + 0.07, 0, f"{prog_score:.2f}", fontsize=10,
                  fontweight="800", va="center", color=stage_color)
        plt.tight_layout()
        st.pyplot(fig_gauge, use_container_width=True); plt.close()
 
    with tab6:
        pd_data   = st.session_state.patient_data
        sex_label = "Female" if pd_data.get("sex_code") == 1 else "Male"
        race_map  = {1: "Mexican American", 2: "Other Hispanic", 3: "NH White",
                     4: "NH Black", 6: "NH Asian", 7: "Other/Multi"}
        race_label = race_map.get(pd_data.get("race_ethnicity_code"), "Unknown")
        age_v      = pd_data.get("age_years", 0)
        age_group  = "<40" if age_v < 40 else "40–55" if age_v < 55 else "55–65" if age_v < 65 else "65+"
        inc        = pd_data.get("household_income_cat", 8)
        inc_label  = "Low (<$25k)" if inc <= 5 else ("Middle ($25–64k)" if inc <= 9 else "High (≥$65k)")
 
        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1: st.metric("Sex", sex_label)
        with cf2: st.metric("Ethnicity", race_label)
        with cf3: st.metric("Age Group", age_group)
        with cf4: st.metric("Income Tier", inc_label)
 
        st.markdown("""
        <div class="rec-box blue">
          <strong>Fairness audit summary:</strong> This system was tested across sex, ethnicity, age,
          and income groups before deployment. Performance thresholds applied:
          <ul style="margin:.5rem 0 0;padding-left:1.2rem">
            <li>Accuracy gap between groups: no more than 5 percentage points</li>
            <li>Sensitivity: at least 60% for all groups</li>
            <li>Sensitivity gap between groups: no more than 15 percentage points</li>
          </ul>
          Observed differences reflect real population-level disease rate variation, not AI bias.
        </div>""", unsafe_allow_html=True)
 
        if "summary_df" in mdl and mdl["summary_df"] is not None:
            st.markdown('<p class="sect-hdr">Fairness Audit Results</p>', unsafe_allow_html=True)
            st.dataframe(mdl["summary_df"], use_container_width=True)
            st.caption("PASS = meets threshold · FAIL = below threshold · NOTE = expected population difference")
        else:
            st.info("Fairness audit table not available in the loaded model file.")
 
    st.markdown("""
    <div class="disclaimer">
      <strong>Clinical Disclaimer:</strong> This system is a decision support tool only and does not replace
      clinical judgement. All predictions must be reviewed by a qualified clinician alongside the full patient
      history, physical examination, and current clinical guidelines. Trained on cross-sectional NHANES data
      (2015–2020, n ≈ 6,600); not validated for longitudinal monitoring without repeated assessment.
    </div>""", unsafe_allow_html=True)
 
    st.divider()
    col_back, _, col_new = st.columns([2, 3, 2])
    with col_back:
        if st.button("Edit Patient Data", use_container_width=True):
            st.session_state.current_page = 0; st.rerun()
    with col_new:
        if st.button("New Patient", type="primary", use_container_width=True):
            st.session_state.current_page = 0; st.rerun()
