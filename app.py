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
