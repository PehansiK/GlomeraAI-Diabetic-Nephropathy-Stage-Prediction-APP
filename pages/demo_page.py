"""
GlomeraAI — DKD Clinical Decision Support System — Demo Page
Pre-filled synthetic patient profiles for KDIGO stages 0–5.
Each tab walks through the full Clinical → Lifestyle → Demographics form
and renders a complete results view identical to the main app.
"""
 
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import entropy as scipy_entropy
 
# ══════════════════════════════════════════════════════════════════════════════
# eGFR HELPER  (CKD-EPI 2021, race-free)
# ══════════════════════════════════════════════════════════════════════════════
def _ckd_epi_2021(scr: float, age: int, sex_code: int) -> float:
    """
    CKD-EPI 2021 creatinine equation (race-free).
 
    Parameters
    ----------
    scr      : serum creatinine (mg/dL)
    age      : age in years
    sex_code : demo-page convention — 0 = Male, 1 = Female
               (internally converted to NHANES: 1 = Male, 2 = Female)
    """
    sex_nhanes = sex_code + 1           # 0→1 (Male), 1→2 (Female)
    kappa = 0.7  if sex_nhanes == 2 else 0.9
    alpha = -0.241 if sex_nhanes == 2 else -0.302
    sf    =  1.012 if sex_nhanes == 2 else 1.000
    r = scr / kappa
    egfr = (
        142
        * (min(r, 1.0) ** alpha)
        * (max(r, 1.0) ** -1.2)
        * (0.9938 ** age)
        * sf
    )
    return round(egfr, 1)

# ══════════════════════════════════════════════════════════════════════════════
# CLINICALLY GROUNDED DEMO PROFILES  (KDIGO 2022 + NHANES reference ranges)
# ══════════════════════════════════════════════════════════════════════════════
def _build_profiles():
    def uacr(alb, cr):
        return round(alb / (cr * 10), 2)
 
    profiles = {}
 
    alb0, cr0 = 5.0, 140.0
    profiles[0] = {
        "label":    "Stage 0 — No DKD",
        "subtitle": "62-year-old Male · Non-Hispanic White · Well-controlled T2DM · No kidney involvement",
        "icon": "🟢", "color": "#22c55e", "bg": "#f0fdf4",
        "clinical_notes": [
            "Serum creatinine 0.90 mg/dL → eGFR ≈ 97 mL/min/1.73 m² (normal)",
            "Urine albumin 5 µg/L → UACR < 3 mg/g (normal, well below 30 mg/g threshold)",
            "HbA1c 6.5% — excellent glycaemic control on oral agents alone",
            "BP 118/74 mmHg — optimal; no antihypertensive required",
            "No kidney disease history, no cardiovascular comorbidities; regular exercise",
        ],
        "sbp1": 118, "sbp2": 117, "sbp3": 119,
        "dbp1": 74,  "dbp2": 73,  "dbp3": 75,
        "serum_cr": 0.90, "urine_alb": alb0, "urine_cr": cr0,
        "hba1c": 6.5,  "fasting_gl": 118.0, "insulin": 8.5,
        "bun": 13.0, "uric_acid": 4.8,
        "hemoglobin": 14.9, "hematocrit": 44.5, "serum_alb": 4.3, "crp": 1.1,
        "tot_chol": 182.0, "ldl": 106.0, "hdl": 57.0, "trig": 102.0, "bmi": 26.0,
        # Lifestyle
        "dm_dx": 1, "insulin_use": 0, "oral_meds": 1, "htn_dx": 0, "bp_med": 0, "statin": 1,
        "kidney_hx": 0, "kidney_stone": 0, "nocturia": 0,
        "smoker": 0, "alcohol": 0.3, "vig_leisure": 1, "sedentary": 240, "sleep_h": 7.5,
        "sodium": 2100.0, "protein": 76.0, "potassium": 3200.0, "phosphorus": 960.0,
        # Demographics
        "age": 62, "sex": 0, "race": 3, "education": 5, "income": 10, "food_sec": 13.0,
        "chd": 0, "heart_att": 0, "stroke": 0, "fam_hx_dm": 1,
    }
 
    alb1, cr1 = 72.0, 105.0
    profiles[1] = {
        "label":    "Stage 1 — Microalbuminuria",
        "subtitle": "55-year-old Female · Non-Hispanic Black · T2DM 8 yrs · Early kidney protein leak",
        "icon": "🟡", "color": "#f59e0b", "bg": "#fffbeb",
        "clinical_notes": [
            "Serum creatinine 1.05 mg/dL → eGFR ≈ 63 mL/min/1.73 m² (preserved)",
            "Urine albumin 72 µg/L → UACR 34 mg/g — confirmed microalbuminuria (30–300 mg/g)",
            "HbA1c 8.1% — suboptimal control; elevated glucose driving albumin leak",
            "BP 138/88 mmHg — Stage 1 hypertension; no ACE inhibitor started yet",
            "Hypertension diagnosed; sedentary lifestyle; BMI 31.4 kg/m²",
        ],
        "sbp1": 138, "sbp2": 137, "sbp3": 139,
        "dbp1": 88,  "dbp2": 87,  "dbp3": 89,
        "serum_cr": 1.05, "urine_alb": alb1, "urine_cr": cr1,
        "hba1c": 8.1, "fasting_gl": 178.0, "insulin": 16.5,
        "bun": 17.0, "uric_acid": 6.0,
        "hemoglobin": 13.0, "hematocrit": 39.5, "serum_alb": 4.1, "crp": 4.6,
        "tot_chol": 212.0, "ldl": 130.0, "hdl": 43.0, "trig": 188.0, "bmi": 31.4,
        "dm_dx": 1, "insulin_use": 0, "oral_meds": 1, "htn_dx": 1, "bp_med": 0, "statin": 1,
        "kidney_hx": 0, "kidney_stone": 0, "nocturia": 1,
        "smoker": 0, "alcohol": 0.1, "vig_leisure": 0, "sedentary": 360, "sleep_h": 6.5,
        "sodium": 3100.0, "protein": 90.0, "potassium": 2600.0, "phosphorus": 1060.0,
        "age": 55, "sex": 1, "race": 4, "education": 3, "income": 6, "food_sec": 9.0,
        "chd": 0, "heart_att": 0, "stroke": 0, "fam_hx_dm": 1,
    }
 
    # Supporting markers tightened to clearly sit in mid-Stage 2 territory
    alb2, cr2 = 215.0, 102.0          # UACR ≈ 106 mg/g (macroalbuminuria approaching)
    profiles[2] = {
        "label":    "Stage 2 — Mild GFR Decrease",
        "subtitle": "67-year-old Male · Mexican American · T2DM 14 yrs · eGFR 45–59 (CKD Stage 3a)",
        "icon": "🟠", "color": "#f97316", "bg": "#fff7ed",
        "clinical_notes": [
            "Serum creatinine 1.47 mg/dL → eGFR ≈ 52 mL/min/1.73 m² (mild GFR decrease, 45–59 range)",
            "Urine albumin 215 µg/L → UACR ≈ 106 mg/g — macroalbuminuria approaching (>300 threshold)",
            "HbA1c 8.7% — poor glycaemic control despite oral agents and insulin",
            "BP 148/93 mmHg despite antihypertensive medication; BUN 20 mg/dL (mildly elevated)",
            "Kidney disease history confirmed; nocturia present; uric acid 7.0 mg/dL",
        ],
        "sbp1": 148, "sbp2": 147, "sbp3": 149,
        "dbp1": 93,  "dbp2": 92,  "dbp3": 94,
        # ── KEY FIX: creatinine corrected to give eGFR ≈ 52 (mid 45–59 band) ──
        "serum_cr": 1.47, "urine_alb": alb2, "urine_cr": cr2,
        "hba1c": 8.7, "fasting_gl": 205.0, "insulin": 22.0,
        # BUN moderate (20) — Stage 2 level, not Stage 3
        "bun": 20.0, "uric_acid": 7.0,
        # Haemoglobin slightly reduced but not anaemic
        "hemoglobin": 12.8, "hematocrit": 38.0, "serum_alb": 3.9, "crp": 7.5,
        "tot_chol": 224.0, "ldl": 142.0, "hdl": 37.0, "trig": 238.0, "bmi": 33.9,
        "dm_dx": 1, "insulin_use": 1, "oral_meds": 1, "htn_dx": 1, "bp_med": 1, "statin": 1,
        "kidney_hx": 1, "kidney_stone": 0, "nocturia": 1,
        "smoker": 0, "alcohol": 0.2, "vig_leisure": 0, "sedentary": 420, "sleep_h": 6.0,
        "sodium": 3400.0, "protein": 96.0, "potassium": 2900.0, "phosphorus": 1210.0,
        "age": 67, "sex": 0, "race": 1, "education": 2, "income": 4, "food_sec": 7.0,
        "chd": 0, "heart_att": 0, "stroke": 0, "fam_hx_dm": 1,
    }
 
    alb3, cr3 = 870.0, 102.0          # UACR ≈ 427 mg/g (heavy proteinuria)
    profiles[3] = {
        "label":    "Stage 3 — Moderate GFR Decrease",
        "subtitle": "71-year-old Female · Non-Hispanic White · T2DM 20 yrs · CKD stage 3b with anaemia & CVD",
        "icon": "🔴", "color": "#ef4444", "bg": "#fff1f2",
        "clinical_notes": [
            "Serum creatinine 1.50 mg/dL → eGFR ≈ 37 mL/min/1.73 m² (moderate GFR decrease, 30–44 range)",
            "Urine albumin 870 µg/L → UACR ≈ 427 mg/g — heavy proteinuria",
            "BUN 28 mg/dL — uraemia accumulating; GFR significantly impaired",
            "Haemoglobin 11.5 g/dL — early CKD-related anaemia",
            "Heart attack and coronary heart disease history; on insulin + BP medication",
        ],
        "sbp1": 158, "sbp2": 157, "sbp3": 159,
        "dbp1": 96,  "dbp2": 95,  "dbp3": 97,
        # ── KEY FIX: creatinine corrected from 2.15 → 1.50 mg/dL ──
        # Old 2.15 → eGFR 24 (Stage 4 band!) | New 1.50 → eGFR 37 (Stage 3 band ✓)
        "serum_cr": 1.50, "urine_alb": alb3, "urine_cr": cr3,
        "hba1c": 9.3, "fasting_gl": 238.0, "insulin": 30.0,
        # BUN 28 — elevated but consistent with eGFR 37, not as severe as Stage 4
        "bun": 28.0, "uric_acid": 8.0,
        # Haemoglobin 11.5 — mild-moderate anaemia appropriate for eGFR 30–44
        "hemoglobin": 11.5, "hematocrit": 34.5, "serum_alb": 3.7, "crp": 12.0,
        "tot_chol": 196.0, "ldl": 116.0, "hdl": 32.0, "trig": 295.0, "bmi": 35.0,
        "dm_dx": 1, "insulin_use": 1, "oral_meds": 0, "htn_dx": 1, "bp_med": 1, "statin": 1,
        "kidney_hx": 1, "kidney_stone": 1, "nocturia": 1,
        "smoker": 0, "alcohol": 0.0, "vig_leisure": 0, "sedentary": 540, "sleep_h": 5.5,
        "sodium": 3600.0, "protein": 100.0, "potassium": 2400.0, "phosphorus": 1360.0,
        "age": 71, "sex": 1, "race": 3, "education": 3, "income": 5, "food_sec": 6.0,
        "chd": 1, "heart_att": 1, "stroke": 0, "fam_hx_dm": 1,
    }
 
    alb4, cr4 = 1850.0, 92.0          # UACR ≈ 1005 mg/g (nephrotic-range)
    profiles[4] = {
        "label":    "Stage 4 — Severe GFR Decrease",
        "subtitle": "74-year-old Male · Non-Hispanic Black · T2DM 25 yrs · Pre-dialysis, full CVD burden",
        "icon": "🟣", "color": "#8b5cf6", "bg": "#faf5ff",
        "clinical_notes": [
            "Serum creatinine 2.90 mg/dL → eGFR ≈ 22 mL/min/1.73 m² (severe GFR decrease, 15–29 range)",
            "BUN 52 mg/dL — significant uraemia; early uraemic symptoms expected",
            "Urine albumin 1850 µg/L → UACR ≈ 1005 mg/g — nephrotic-range proteinuria",
            "Haemoglobin 9.8 g/dL — severe CKD anaemia; EPO therapy under consideration",
            "Stroke, heart attack, and coronary artery disease; current smoker",
        ],
        "sbp1": 168, "sbp2": 167, "sbp3": 169,
        "dbp1": 100, "dbp2": 99,  "dbp3": 101,
        # ── KEY FIX: creatinine corrected from 3.85 → 2.90 mg/dL ──
        # Old 3.85 → eGFR 15.7 (Stage 4/5 border!) | New 2.90 → eGFR 22 (Stage 4 mid-range ✓)
        "serum_cr": 2.90, "urine_alb": alb4, "urine_cr": cr4,
        "hba1c": 10.2, "fasting_gl": 275.0, "insulin": 46.0,
        # BUN 52 — uraemic but consistent with eGFR 22 (not as severe as eGFR <15)
        "bun": 52.0, "uric_acid": 9.2,
        # Haemoglobin 9.8 — severe anaemia but not as extreme as Stage 5
        "hemoglobin": 9.8, "hematocrit": 29.5, "serum_alb": 3.3, "crp": 20.0,
        "tot_chol": 186.0, "ldl": 103.0, "hdl": 27.0, "trig": 385.0, "bmi": 30.4,
        "dm_dx": 1, "insulin_use": 1, "oral_meds": 0, "htn_dx": 1, "bp_med": 1, "statin": 1,
        "kidney_hx": 1, "kidney_stone": 1, "nocturia": 1,
        "smoker": 2, "alcohol": 0.0, "vig_leisure": 0, "sedentary": 660, "sleep_h": 5.0,
        "sodium": 3800.0, "protein": 105.0, "potassium": 2200.0, "phosphorus": 1510.0,
        "age": 74, "sex": 0, "race": 4, "education": 2, "income": 3, "food_sec": 4.0,
        "chd": 1, "heart_att": 1, "stroke": 1, "fam_hx_dm": 1,
    }
 
    alb5, cr5 = 3250.0, 76.0
    profiles[5] = {
        "label":    "Stage 5 — Kidney Failure",
        "subtitle": "69-year-old Male · Non-Hispanic Black · T2DM 30 yrs · ESKD — dialysis candidate",
        "icon": "⚫", "color": "#0f172a", "bg": "#f1f5f9",
        "clinical_notes": [
            "Serum creatinine 7.3 mg/dL → eGFR ≈ 8 mL/min/1.73 m² — end-stage kidney disease (<15)",
            "BUN 98 mg/dL — severe uraemia; uraemic symptoms expected",
            "Urine albumin 3250 µg/L → UACR ≈ 2138 mg/g — massive nephrotic proteinuria",
            "Haemoglobin 7.9 g/dL — severe anaemia; serum albumin 2.8 g/dL (protein-energy wasting)",
            "Full CVD burden (MI, stroke, CHD); current smoker; lowest income/education bracket",
        ],
        "sbp1": 178, "sbp2": 177, "sbp3": 179,
        "dbp1": 106, "dbp2": 105, "dbp3": 107,
        "serum_cr": 7.3, "urine_alb": alb5, "urine_cr": cr5,
        "hba1c": 11.0, "fasting_gl": 320.0, "insulin": 65.0,
        "bun": 98.0, "uric_acid": 11.3,
        "hemoglobin": 7.9, "hematocrit": 23.5, "serum_alb": 2.8, "crp": 38.5,
        "tot_chol": 170.0, "ldl": 90.0, "hdl": 23.0, "trig": 485.0, "bmi": 27.6,
        "dm_dx": 1, "insulin_use": 1, "oral_meds": 0, "htn_dx": 1, "bp_med": 1, "statin": 1,
        "kidney_hx": 1, "kidney_stone": 1, "nocturia": 1,
        "smoker": 2, "alcohol": 0.0, "vig_leisure": 0, "sedentary": 780, "sleep_h": 4.5,
        "sodium": 4000.0, "protein": 110.0, "potassium": 2000.0, "phosphorus": 1660.0,
        "age": 69, "sex": 0, "race": 4, "education": 1, "income": 2, "food_sec": 2.0,
        "chd": 1, "heart_att": 1, "stroke": 1, "fam_hx_dm": 1,
    }
 
    return profiles

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _stepper_html(current):
    steps = ["Clinical Data", "Lifestyle", "Demographics", "Results"]
    parts = []
    for i, label in enumerate(steps):
        cls  = "done"   if i < current  else ("active" if i == current else "todo")
        num  = "✓"      if i < current  else str(i + 1)
        line = "done"   if i < current  else ""
        dot  = f'<div class="step-dot {cls}">{num}</div>'
        lbl  = f'<div class="step-label">{label}</div>'
        inner = f'<div style="display:flex;flex-direction:column;align-items:center">{dot}{lbl}</div>'
        conn  = f'<div class="step-line {line}"></div>' if i < len(steps) - 1 else ""
        parts.append(f'<div class="step-item">{inner}{conn}</div>')
    return '<div class="stepper">' + "".join(parts) + "</div>"
 
def _urgency_chip(text):
    color_map = {
        "Routine": "green", "Within 4": "blue", "Within 2": "amber",
        "Urgent":  "red",   "Immediate": "red", "Emergency": "red",
    }
    chip_color = next((v for k, v in color_map.items() if text.startswith(k)), "blue")
    return f'<span class="chip {chip_color}">{text}</span>'
 
def _display_value(feat, raw_val, CLINICAL_CONTEXT):
    if feat in CLINICAL_CONTEXT:
        name, transform, _ = CLINICAL_CONTEXT[feat]
        if transform == "expm1":
            return name, float(np.expm1(raw_val))
        return name, raw_val
    return feat.replace("_", " ").title(), raw_val
 
def _plot_shap_bar(shap_dict, title, color_pos, color_neg, top_n,
                   CLINICAL_CONTEXT, highlight_feats=None):
    items = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    if not items:
        return None
    labels, vals = [], []
    for feat, sv in items:
        name, _ = _display_value(feat, 0, CLINICAL_CONTEXT)[:2]
        labels.append(name)
        vals.append(sv)
    bar_colors = []
    for feat, sv in items:
        if sv > 0:
            bar_colors.append("#f97316" if (highlight_feats and feat in highlight_feats) else color_pos)
        else:
            bar_colors.append("#0ea5e9" if (highlight_feats and feat in highlight_feats) else color_neg)
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(labels) * 0.48)))
    ax.barh(labels[::-1], vals[::-1], color=bar_colors[::-1], alpha=0.88, height=0.62)
    ax.axvline(0, color="#374151", lw=0.9, ls="--")
    ax.set_xlabel("Influence on prediction (SHAP value)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="700")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PAGE SECTIONS
# ══════════════════════════════════════════════════════════════════════════════
 
def _render_clinical_form(s, p):
    """Render Clinical Data form for stage s, pre-filled from profile p."""
    k = f"d{s}_"  # unique key prefix
 
    st.markdown("""
    <div class="model-badge rf">
      <h4>🌲 Clinical Model (M1) — Blood Tests & Kidney Markers</h4>
      <p>Key indicators: Creatinine · Urine Albumin · Blood Urea · HbA1c · Haemoglobin</p>
    </div>""", unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🩸 Blood Pressure** *(average of 3 readings)*")
        sbp1 = st.number_input("Systolic BP — Reading 1 (mmHg)",  80, 220, p["sbp1"], key=f"{k}sbp1")
        dbp1 = st.number_input("Diastolic BP — Reading 1 (mmHg)", 40, 130, p["dbp1"], key=f"{k}dbp1")
        sbp2 = st.number_input("Systolic BP — Reading 2 (mmHg)",  80, 220, p["sbp2"], key=f"{k}sbp2")
        dbp2 = st.number_input("Diastolic BP — Reading 2 (mmHg)", 40, 130, p["dbp2"], key=f"{k}dbp2")
        sbp3 = st.number_input("Systolic BP — Reading 3 (mmHg)",  80, 220, p["sbp3"], key=f"{k}sbp3")
        dbp3 = st.number_input("Diastolic BP — Reading 3 (mmHg)", 40, 130, p["dbp3"], key=f"{k}dbp3")
 
        st.markdown("**🔬 Kidney & Urine Markers**")
        serum_cr  = st.number_input("Serum Creatinine (mg/dL)",        0.2,  20.0,    p["serum_cr"],  step=0.01, key=f"{k}serum_cr")
        urine_alb = st.number_input("Urine Albumin (ug/L)",             0.5,  5000.0,  p["urine_alb"], step=0.5, key=f"{k}urine_alb")
        urine_cr  = st.number_input("Urine Creatinine (mg/dL)",         5.0,  3000.0,  p["urine_cr"],           key=f"{k}urine_cr")
 
    with col2:
        st.markdown("**💉 Metabolic & Blood Chemistry**")
        hba1c      = st.number_input("HbA1c — Blood Sugar Control (%)", 4.0,  20.0,   p["hba1c"],     step=0.1, key=f"{k}hba1c")
        fasting_gl = st.number_input("Fasting Glucose (mg/dL)",         40.0, 600.0,  p["fasting_gl"],          key=f"{k}fasting_gl")
        insulin    = st.number_input("Fasting Insulin (uIU/mL)",        0.0,  300.0,  p["insulin"],             key=f"{k}insulin")
        bun        = st.number_input("Blood Urea Nitrogen — BUN (mg/dL)",2.0, 150.0,  p["bun"],                 key=f"{k}bun")
        uric_acid  = st.number_input("Uric Acid (mg/dL)",               1.0,  20.0,   p["uric_acid"], step=0.1, key=f"{k}uric_acid")
 
        st.markdown("**🩺 Blood Count & Other**")
        hemoglobin = st.number_input("Haemoglobin (g/dL)",  4.0,  22.0, p["hemoglobin"], step=0.1, key=f"{k}hemoglobin")
        hematocrit = st.number_input("Haematocrit (%)",    10.0,  65.0, p["hematocrit"], step=0.5, key=f"{k}hematocrit")
        serum_alb  = st.number_input("Serum Albumin (g/dL)", 1.0,  6.0, p["serum_alb"],  step=0.1, key=f"{k}serum_alb")
        crp        = st.number_input("CRP — Inflammation Marker (mg/L)", 0.0, 200.0, p["crp"], step=0.1, key=f"{k}crp")
 
    st.markdown("**🍳 Lipid Panel & Body**")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        tot_chol = st.number_input("Total Cholesterol (mg/dL)", 50.0,  600.0, p["tot_chol"], key=f"{k}tot_chol")
        ldl      = st.number_input("LDL Cholesterol (mg/dL)",   20.0,  500.0, p["ldl"],      key=f"{k}ldl")
    with col_b:
        hdl  = st.number_input("HDL Cholesterol (mg/dL)",       10.0,  200.0, p["hdl"],  key=f"{k}hdl")
        trig = st.number_input("Triglycerides (mg/dL)",         20.0, 3000.0, p["trig"], key=f"{k}trig")
    with col_c:
        bmi = st.number_input("BMI (kg/m²)", 12.0, 80.0, p["bmi"], step=0.1, key=f"{k}bmi")
 
    mean_sbp = round((sbp1 + sbp2 + sbp3) / 3, 1)
    mean_dbp = round((dbp1 + dbp2 + dbp3) / 3, 1)
    uacr_val = round(urine_alb / (urine_cr * 10), 2) if urine_cr > 0 else 0.0
 
    # ── Derive eGFR using CKD-EPI 2021 (race-free) ──
    egfr_val = _ckd_epi_2021(serum_cr, p["age"], p["sex"])
 
    # eGFR stage label helper
    if egfr_val >= 90:
        egfr_stage = "G1 (normal)"
    elif egfr_val >= 60:
        egfr_stage = "G2 (mildly reduced)"
    elif egfr_val >= 45:
        egfr_stage = "G3a (mild–moderate)"
    elif egfr_val >= 30:
        egfr_stage = "G3b (moderate)"
    elif egfr_val >= 15:
        egfr_stage = "G4 (severely reduced)"
    else:
        egfr_stage = "G5 (kidney failure)"
 
    st.markdown(f"""
    <div class="rec-box blue">
      <strong>📊 Auto-calculated values:</strong><br>
      Mean Systolic BP = <strong>{mean_sbp} mmHg</strong> &nbsp;·&nbsp;
      Mean Diastolic BP = <strong>{mean_dbp} mmHg</strong> &nbsp;·&nbsp;
      Urine Albumin Ratio (UACR) = <strong>{uacr_val:.1f} mg/g</strong>
      <br>
      eGFR (CKD-EPI 2021, race-free) = <strong>{egfr_val} mL/min/1.73 m²</strong>
      &nbsp;<span style="background:#e0f2fe;color:#0369a1;padding:2px 8px;border-radius:12px;
      font-size:.8rem;font-weight:700">{egfr_stage}</span>
    </div>""", unsafe_allow_html=True)
 
    # Collect data dict
    data = {
        "mean_sbp": mean_sbp, "mean_dbp": mean_dbp,
        "serum_creatinine_mgdl": serum_cr,
        "log_serum_creatinine_mgdl": np.log1p(serum_cr),
        "urine_albumin_ugl": urine_alb,
        "log_urine_albumin_ugl": np.log1p(urine_alb),
        "urine_creatinine_mgdl": urine_cr,
        "uacr_mgg": uacr_val,
        "log_uacr": np.log10(max(uacr_val, 0.01)),
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
        # Derived eGFR stored for display in results
        "egfr_ml_min": egfr_val,
    }
    return data
 
def _render_lifestyle_form(s, p):
    """Render Lifestyle form for stage s, pre-filled from profile p."""
    k = f"d{s}_"
 
    st.markdown("""
    <div class="model-badge xgb">
      <h4>⚡ Lifestyle Model (M2) — Medications, Activity & Diet</h4>
      <p>Key indicators: Kidney Disease History · Hypertension · Insulin Use · Physical Activity · Phosphorus intake</p>
    </div>""", unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**💊 Medications & Diagnoses**")
        dm_dx       = st.selectbox("Diabetes Diagnosed?",             [1, 0],          format_func=lambda x: "Yes" if x else "No", index=0 if p["dm_dx"] else 1, key=f"{k}dm_dx")
        insulin_use = st.selectbox("Currently using insulin?",        [0, 1],          format_func=lambda x: "Yes" if x else "No", index=p["insulin_use"],        key=f"{k}insulin_use")
        oral_meds   = st.selectbox("Taking oral diabetes medications?",[0, 1],          format_func=lambda x: "Yes" if x else "No", index=p["oral_meds"],           key=f"{k}oral_meds")
        htn_dx      = st.selectbox("Diagnosed with high blood pressure?",[0, 1],        format_func=lambda x: "Yes" if x else "No", index=p["htn_dx"],              key=f"{k}htn_dx")
        bp_med      = st.selectbox("Taking blood pressure medication?", [0, 1],         format_func=lambda x: "Yes" if x else "No", index=p["bp_med"],              key=f"{k}bp_med")
        statin      = st.selectbox("Taking statins (cholesterol medication)?", [0, 1],  format_func=lambda x: "Yes" if x else "No", index=p["statin"],              key=f"{k}statin")
 
        st.markdown("**🏥 Kidney History**")
        kidney_hx    = st.selectbox("History of kidney disease?",  [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["kidney_hx"],    key=f"{k}kidney_hx")
        kidney_stone = st.selectbox("History of kidney stones?",   [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["kidney_stone"], key=f"{k}kidney_stone")
        nocturia     = st.selectbox("Waking at night to urinate?", [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["nocturia"],     key=f"{k}nocturia")
 
    with col2:
        st.markdown("**🏃 Physical Activity & Lifestyle**")
        smoker_options = [0, 1, 2]
        smoker_idx = smoker_options.index(p["smoker"]) if p["smoker"] in smoker_options else 0
        smoker      = st.selectbox("Smoking status", smoker_options,
                                   format_func=lambda x: ["Never smoked", "Former smoker", "Current smoker"][x],
                                   index=smoker_idx, key=f"{k}smoker")
        alcohol     = st.number_input("Average alcohol (drinks/day)",  0.0,  20.0, float(p["alcohol"]),   step=0.1, key=f"{k}alcohol")
        vig_leisure = st.selectbox("Does the patient do vigorous exercise?", [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["vig_leisure"], key=f"{k}vig_leisure")
        sedentary   = st.number_input("Hours spent sitting/lying per day (minutes)", 0, 1440, int(p["sedentary"]), key=f"{k}sedentary")
        sleep_h     = st.number_input("Average sleep hours on weekdays", 2.0, 14.0, float(p["sleep_h"]), step=0.5, key=f"{k}sleep_h")
 
        st.markdown("**🥗 Dietary Intake (daily averages)**")
        sodium     = st.number_input("Dietary Sodium (mg/day)",     100.0, 15000.0, float(p["sodium"]),     key=f"{k}sodium")
        protein    = st.number_input("Dietary Protein (g/day)",       5.0,   400.0, float(p["protein"]),    key=f"{k}protein")
        potassium  = st.number_input("Dietary Potassium (mg/day)",  200.0,  8000.0, float(p["potassium"]),  key=f"{k}potassium")
        phosphorus = st.number_input("Dietary Phosphorus (mg/day)", 100.0,  4000.0, float(p["phosphorus"]), key=f"{k}phosphorus")
 
    data = {
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
    }
    return data
 
def _render_demographics_form(s, p):
    """Render Demographics form for stage s, pre-filled from profile p."""
    k = f"d{s}_"
 
    st.markdown("""
    <div class="model-badge lr">
      <h4>📊 Demographics Model (M3) — Patient Background & History</h4>
      <p>Key indicators: Age · Sex · Ethnicity · BMI · Heart Attack History</p>
    </div>""", unsafe_allow_html=True)
 
    race_options  = [1, 2, 3, 4, 6, 7]
    income_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15]
    edu_options   = [1, 2, 3, 4, 5]
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**👤 Basic Information**")
        age  = st.number_input("Patient age (years)", 18, 100, int(p["age"]), key=f"{k}age")
        sex  = st.selectbox("Sex", [0, 1],
                            format_func=lambda x: "Male" if x == 0 else "Female",
                            index=p["sex"], key=f"{k}sex")
        race_idx = race_options.index(p["race"]) if p["race"] in race_options else 0
        race = st.selectbox("Race/Ethnicity", race_options,
                            format_func=lambda x: {1:"Mexican American",2:"Other Hispanic",
                                                    3:"Non-Hispanic White",4:"Non-Hispanic Black",
                                                    6:"Non-Hispanic Asian",7:"Other/Multiracial"}[x],
                            index=race_idx, key=f"{k}race")
 
        st.markdown("**📊 Socioeconomic Background**")
        edu_idx = edu_options.index(p["education"]) if p["education"] in edu_options else 3
        education = st.selectbox("Highest education level", edu_options, index=edu_idx,
                                 format_func=lambda x: {1:"Less than 9th grade",2:"Some high school",
                                                         3:"High school / GED",4:"Some college",
                                                         5:"College graduate or higher"}[x],
                                 key=f"{k}education")
        inc_idx = income_options.index(p["income"]) if p["income"] in income_options else 7
        income = st.selectbox("Household income category", income_options, index=inc_idx,
                              format_func=lambda x: {1:"< $5,000",2:"$5–10k",3:"$10–15k",4:"$15–20k",
                                                      5:"$20–25k",6:"$25–35k",7:"$35–45k",8:"$45–55k",
                                                      9:"$55–65k",10:"$65–75k",14:"$75–100k",
                                                      15:"Over $100,000"}[x],
                              key=f"{k}income")
        food_sec = st.number_input("Food security score (0–18)", 0.0, 18.0, float(p["food_sec"]), step=1.0, key=f"{k}food_sec")
 
    with col2:
        st.markdown("**❤️ Cardiovascular & Family History**")
        chd       = st.selectbox("History of coronary heart disease?", [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["chd"],       key=f"{k}chd")
        heart_att = st.selectbox("History of heart attack?",           [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["heart_att"], key=f"{k}heart_att")
        stroke    = st.selectbox("History of stroke?",                 [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["stroke"],    key=f"{k}stroke")
        fam_hx_dm = st.selectbox("Family history of diabetes?",        [0, 1], format_func=lambda x: "Yes" if x else "No", index=p["fam_hx_dm"], key=f"{k}fam_hx_dm")
 
    race_label = {1:"Mexican American",2:"Other Hispanic",3:"Non-Hispanic White",
                  4:"Non-Hispanic Black",6:"Non-Hispanic Asian",7:"Other/Multiracial"}[race]
    sex_label  = "Female" if sex == 1 else "Male"
    st.markdown(f"""
    <div class="rec-box blue">
      <strong>👤 Patient Profile:</strong>
      {age}-year-old {sex_label} · {race_label}
    </div>""", unsafe_allow_html=True)
 
    data = {
        "age_years": age, "sex_code": sex, "race_ethnicity_code": race,
        "education_level": education, "household_income_cat": income,
        "food_security_score": food_sec,
        "coronary_heart_disease": chd, "heart_attack": heart_att,
        "stroke_ever": stroke, "family_hx_diabetes": fam_hx_dm,
    }
    return data
 
def _render_results(
    s, patient_data, mdl,
    CLINICAL_CONTEXT, STAGE_NAMES, STAGE_COLORS, STAGE_BG,
    STAGE_RECOMMENDATIONS, N_CLASSES,
    build_patient_vector, run_ensemble,
    compute_shap_patient, progression_risk_profile,
):
    """Run the model and render a full main-app-style results section for stage s."""
 
    res_key = f"demo_res_{s}"
 
    # Run prediction (cache in session state so it doesn't re-run on every widget interaction)
    if res_key not in st.session_state:
        with st.spinner("🔄 Running AI analysis — please wait…"):
            try:
                df_imp, X1, X2, X3 = build_patient_vector(patient_data, mdl)
                pred_stage, proba_vec, p1, p2, p3 = run_ensemble(mdl, X1, X2, X3)
                rf_sv, xgb_sv, lr_sv = compute_shap_patient(mdl, X1, X2, X3)
                risk_drivers, prot_drivers = progression_risk_profile(rf_sv, X1, pred_stage)
                st.session_state[res_key] = dict(
                    pred_stage=pred_stage, proba_vec=proba_vec, p1=p1, p2=p2, p3=p3,
                    rf_sv=rf_sv, xgb_sv=xgb_sv, lr_sv=lr_sv,
                    X1=X1, X2=X2, X3=X3,
                    risk_drivers=risk_drivers, prot_drivers=prot_drivers,
                )
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                return
 
    r = st.session_state[res_key]
    pred_stage   = r["pred_stage"]
    proba_vec    = r["proba_vec"]
    p1, p2, p3   = r["p1"], r["p2"], r["p3"]
    rf_sv        = r["rf_sv"]
    xgb_sv       = r["xgb_sv"]
    lr_sv        = r["lr_sv"]
    X1_, X2_, X3_ = r["X1"], r["X2"], r["X3"]
    risk_drivers = r["risk_drivers"]
    prot_drivers = r["prot_drivers"]
 
    stage_color = STAGE_COLORS[pred_stage]
    stage_bg    = STAGE_BG[pred_stage]
    stage_name  = STAGE_NAMES[pred_stage]
    rec         = STAGE_RECOMMENDATIONS[pred_stage]
 
    prog_score  = sum(c * proba_vec[c] for c in range(N_CLASSES))
    entropy_val = scipy_entropy(proba_vec)
    max_entropy = scipy_entropy([1 / N_CLASSES] * N_CLASSES)
    uncertainty = entropy_val / max_entropy
    confidence_label = ("High confidence"   if uncertainty < 0.33 else
                        "Moderate confidence" if uncertainty < 0.66 else
                        "Low confidence — review carefully")
    confidence_color = ("#22c55e" if uncertainty < 0.33 else
                        "#f59e0b" if uncertainty < 0.66 else "#ef4444")
 
    forward_prob = float(sum(proba_vec[c] for c in range(pred_stage + 1, N_CLASSES)))
    if pred_stage < N_CLASSES - 1:
        if   forward_prob >= 0.30: progression_verdict, prog_icon, prog_color = "YES",      "⚠️", "#f43f5e"
        elif forward_prob >= 0.15: progression_verdict, prog_icon, prog_color = "POSSIBLE", "🔔", "#f59e0b"
        else:                      progression_verdict, prog_icon, prog_color = "NO",       "✅", "#22c55e"
 
    # ── Match badge ───────────────────────────────────────────────────────────
    if pred_stage == s:
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;
                    padding:.65rem 1rem;font-size:.84rem;color:#166534;font-weight:600;
                    margin-bottom:.8rem">
          ✅ Correct prediction — Model predicted <strong>Stage {pred_stage}</strong>
          (expected Stage {s})
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#fffbeb;border:1.5px solid #fcd34d;border-radius:10px;
                    padding:.65rem 1rem;font-size:.84rem;color:#92400e;font-weight:600;
                    margin-bottom:.8rem">
          ⚠️ Model predicted <strong>Stage {pred_stage}</strong>
          (expected Stage {s}) — review the probability distribution below
        </div>""", unsafe_allow_html=True)
 
    # ── Result Hero ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{stage_bg};border:2px solid {stage_color};border-radius:16px;
                padding:1.6rem 2rem;margin-bottom:1.2rem">
      <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
                  color:{stage_color};margin-bottom:.4rem">AI ASSESSMENT RESULT</div>
      <div style="font-size:2rem;font-weight:800;color:{stage_color};letter-spacing:-.03em">
        {stage_name}
      </div>
      <div style="font-size:.9rem;color:#475569;margin-top:.4rem">{rec['headline']}</div>
      <div style="margin-top:.8rem">{_urgency_chip(rec['urgency'])}</div>
    </div>""", unsafe_allow_html=True)
 
    # ── Metric Tiles ──────────────────────────────────────────────────────────
    egfr_display = patient_data.get("egfr_ml_min", "N/A")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="value" style="color:{stage_color}">Stage {pred_stage}</div>
          <div class="label">KDIGO Stage (0–5)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="value" style="color:#0f172a">{proba_vec[pred_stage]*100:.0f}%</div>
          <div class="label">Probability of this stage</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="value" style="color:{confidence_color};font-size:1.1rem;padding-top:.5rem">
            {confidence_label.split(" ")[0]} {confidence_label.split(" ")[1]}
          </div>
          <div class="label">AI confidence level</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="value" style="color:#0f172a">{prog_score:.1f}/5</div>
          <div class="label">Disease severity score</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-tile">
          <div class="value" style="color:#0369a1">{egfr_display}</div>
          <div class="label">eGFR mL/min/1.73 m²</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<div style='margin-top:.5rem'></div>", unsafe_allow_html=True)
 
    # ── Clinical Action Plan ──────────────────────────────────────────────────
    rec_color_map = {0:"green", 1:"blue", 2:"amber", 3:"red", 4:"red", 5:"slate"}
    box_color     = rec_color_map[pred_stage]
    actions_html  = "".join(f"<li style='margin:.3rem 0'>{a}</li>" for a in rec["actions"])
    st.markdown(f"""
    <div class="rec-box {box_color}">
      <strong>Recommended Clinical Actions:</strong>
      <ul style="margin:.5rem 0 0;padding-left:1.2rem">{actions_html}</ul>
    </div>""", unsafe_allow_html=True)
 
    st.divider()
 
    # ── Result Tabs ───────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Summary",
        "🌲 Clinical Factors (M1)",
        "⚡ Lifestyle Factors (M2)",
        "📊 Demographic Factors (M3)",
        "📈 Will Disease Progress?",
        "⚖️ Fairness & Equity",
    ])
 
    SHAP_EXPLAIN = (
        "The bars show how much each factor <b>increases</b> (red) or "
        "<b>decreases</b> (blue) the AI's assessment for this patient. "
        "Longer bars mean stronger influence."
    )
    RF_EXPECTED  = ["log_serum_creatinine_mgdl","log_urine_albumin_ugl","log_bun_mgdl","hba1c_pct","hemoglobin_gdl"]
    XGB_EXPECTED = ["hypertension_diagnosed","kidney_disease_history","kidney_stone_history","insulin_use","phosphorus_mg_day"]
    LR_EXPECTED  = ["age_years","race_ethnicity_code","sex_code","bmi_kgm2","heart_attack"]
 
    # ── Tab 1: Summary ────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<p class="sect-hdr">Patient Snapshot</p>', unsafe_allow_html=True)
        cs1, cs2, cs3 = st.columns(3)
        with cs1:
            st.markdown("**Key Clinical Values**")
            st.write(f"• HbA1c: **{patient_data.get('hba1c_pct','N/A'):.1f}%** (target <7%)")
            st.write(f"• Urine Albumin Ratio: **{patient_data.get('uacr_mgg','N/A'):.1f} mg/g**")
            st.write(f"• eGFR: **{patient_data.get('egfr_ml_min','N/A')} mL/min/1.73 m²**")
            st.write(f"• Blood Pressure: **{patient_data.get('mean_sbp','N/A')}/{patient_data.get('mean_dbp','N/A')} mmHg**")
            st.write(f"• BMI: **{patient_data.get('bmi_kgm2','N/A'):.1f} kg/m²**")
        with cs2:
            st.markdown("**How Each AI Model Voted**")
            st.markdown(f"""
            <div style="font-size:.85rem">
              <div style="display:flex;justify-content:space-between;padding:.4rem .6rem;
                          background:#f0f7ff;border-radius:8px;margin:.3rem 0">
                <span>🌲 Clinical Model</span>
                <strong style="color:#3b82f6">{p1[pred_stage]*100:.0f}% confidence</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:.4rem .6rem;
                          background:#fff7f0;border-radius:8px;margin:.3rem 0">
                <span>⚡ Lifestyle Model</span>
                <strong style="color:#f97316">{p2[pred_stage]*100:.0f}% confidence</strong>
              </div>
              <div style="display:flex;justify-content:space-between;padding:.4rem .6rem;
                          background:#faf5ff;border-radius:8px;margin:.3rem 0">
                <span>📊 Demographics Model</span>
                <strong style="color:#8b5cf6">{p3[pred_stage]*100:.0f}% confidence</strong>
              </div>
            </div>""", unsafe_allow_html=True)
        with cs3:
            st.markdown("**Patient Demographics**")
            sex_l  = "Female" if patient_data.get("sex_code") == 1 else "Male"
            race_l = {1:"Mexican American",2:"Other Hispanic",3:"NH White",
                      4:"NH Black",6:"NH Asian",7:"Other/Multi"}.get(patient_data.get("race_ethnicity_code"),"N/A")
            age_v  = patient_data.get("age_years", 0)
            inc    = patient_data.get("household_income_cat", 8)
            inc_l  = "Low (<$25k)" if inc <= 5 else ("Middle ($25–64k)" if inc <= 9 else "High (≥$65k)")
            st.write(f"• Age: **{age_v} years** · {sex_l}")
            st.write(f"• Ethnicity: **{race_l}**")
            st.write(f"• Income bracket: **{inc_l}**")
 
        st.markdown('<p class="sect-hdr">Stage Probability Breakdown</p>', unsafe_allow_html=True)
        st.caption("Probability the patient belongs to each DKD stage.")
        fig_dist, ax_dist = plt.subplots(figsize=(10, 2.8))
        bars_d = ax_dist.barh([STAGE_NAMES[c] for c in range(N_CLASSES)],
                              [proba_vec[c] * 100 for c in range(N_CLASSES)],
                              color=[STAGE_COLORS[c] for c in range(N_CLASSES)],
                              alpha=0.85, edgecolor="white", linewidth=0.5, height=0.65)
        for bar, val in zip(bars_d, proba_vec):
            ax_dist.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f"{val*100:.1f}%", va="center", fontsize=8.5, fontweight="600")
        ax_dist.set_xlabel("Probability (%)", fontsize=9)
        ax_dist.set_xlim(0, 108)
        ax_dist.spines[["top", "right"]].set_visible(False)
        ax_dist.tick_params(labelsize=8.5)
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close()
 
    # ── Tab 2: M1 RF SHAP ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="model-badge rf">
          <h4>🌲 Clinical Model (M1) — Which blood test results drove this prediction?</h4>
          <p>The chart shows which clinical measurements most influenced the AI's decision for this patient.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box blue">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
 
        try:
            shap_rf_dict = dict(zip(X1_.columns.tolist(), rf_sv[0, :, pred_stage]))
            fig_rf = _plot_shap_bar(shap_rf_dict, f"Clinical Factors Influencing the Stage {pred_stage} Prediction",
                                    "#ef4444", "#3b82f6", 10, CLINICAL_CONTEXT, RF_EXPECTED)
            if fig_rf:
                st.pyplot(fig_rf, use_container_width=True); plt.close()
 
            st.markdown("""
            <p style="font-size:.8rem;text-align:center;color:#64748b;margin-top:.4rem">
              <span style="color:#ef4444;font-weight:600">■ Red</span> = increases risk &nbsp;|&nbsp;
              <span style="color:#3b82f6;font-weight:600">■ Blue</span> = decreases risk &nbsp;|&nbsp;
              <span style="color:#f97316;font-weight:600">■ Orange</span> = key marker confirmed by research
            </p>""", unsafe_allow_html=True)
 
            st.markdown('<p class="sect-hdr">How Each Stage Is Affected By Clinical Markers</p>', unsafe_allow_html=True)
            top_feats_rf = sorted(shap_rf_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            feat_labels  = [_display_value(f, 0, CLINICAL_CONTEXT)[0] for f, _ in top_feats_rf]
            shap_matrix  = np.array([[rf_sv[0, list(X1_.columns).index(f), c] for f, _ in top_feats_rf]
                                      for c in range(N_CLASSES)])
            fig_heat, ax_heat = plt.subplots(figsize=(10, 4))
            im = ax_heat.imshow(shap_matrix, cmap="RdBu_r", aspect="auto",
                                norm=mcolors.TwoSlopeNorm(vcenter=0))
            ax_heat.set_xticks(range(len(feat_labels)))
            ax_heat.set_xticklabels(feat_labels, rotation=35, ha="right", fontsize=8.5)
            ax_heat.set_yticks(range(N_CLASSES))
            ax_heat.set_yticklabels([STAGE_NAMES[c] for c in range(N_CLASSES)], fontsize=8.5)
            plt.colorbar(im, ax=ax_heat, shrink=0.8, label="Influence (SHAP value)")
            ax_heat.set_title("Clinical Factor Influence Across All Stages", fontsize=10, fontweight="700")
            plt.tight_layout()
            st.pyplot(fig_heat, use_container_width=True); plt.close()
        except Exception as exc:
            st.warning(f"Clinical SHAP unavailable: {exc}")
 
    # ── Tab 3: M2 XGB SHAP ────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="model-badge xgb">
          <h4>⚡ Lifestyle Model (M2) — Which lifestyle factors drove this prediction?</h4>
          <p>The chart shows how medications, activity levels, and diet influenced the AI's decision.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box amber">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
 
        try:
            shap_xgb_dict = dict(zip(X2_.columns.tolist(), xgb_sv[0, :, pred_stage]))
            fig_xgb = _plot_shap_bar(shap_xgb_dict, f"Lifestyle Factors Influencing the Stage {pred_stage} Prediction",
                                     "#f97316", "#0ea5e9", 10, CLINICAL_CONTEXT, XGB_EXPECTED)
            if fig_xgb:
                st.pyplot(fig_xgb, use_container_width=True); plt.close()
 
            st.markdown("""
            <p style="font-size:.8rem;text-align:center;color:#64748b;margin-top:.4rem">
              <span style="color:#f97316;font-weight:600">■ Orange</span> = increases risk &nbsp;|&nbsp;
              <span style="color:#0ea5e9;font-weight:600">■ Blue</span> = decreases risk
            </p>""", unsafe_allow_html=True)
 
            st.markdown('<p class="sect-hdr">How All Three Models Compare for This Patient</p>', unsafe_allow_html=True)
            fig_comp, ax_comp = plt.subplots(figsize=(10, 3.2))
            x = np.arange(N_CLASSES); w = 0.26
            ax_comp.bar(x - w, p1 * 100, w, label="M1 Clinical",     color="#3b82f6", alpha=0.85)
            ax_comp.bar(x,     p2 * 100, w, label="M2 Lifestyle",    color="#f97316", alpha=0.85)
            ax_comp.bar(x + w, p3 * 100, w, label="M3 Demographics", color="#8b5cf6", alpha=0.85)
            ax_comp.set_xticks(x)
            ax_comp.set_xticklabels([f"Stage {c}" for c in range(N_CLASSES)], fontsize=8.5)
            ax_comp.set_ylabel("Model confidence (%)", fontsize=9)
            ax_comp.set_title("Agreement Between All Three AI Models", fontsize=10, fontweight="700")
            ax_comp.legend(fontsize=8.5)
            ax_comp.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_comp, use_container_width=True); plt.close()
        except Exception as exc:
            st.warning(f"Lifestyle SHAP unavailable: {exc}")
 
    # ── Tab 4: M3 LR SHAP ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="model-badge lr">
          <h4>📊 Demographics Model (M3) — Which patient background factors drove this prediction?</h4>
          <p>The chart shows how age, sex, ethnicity, and medical history influenced the AI's decision.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="rec-box purple">{SHAP_EXPLAIN}</div>', unsafe_allow_html=True)
 
        if lr_sv is not None:
            try:
                if lr_sv.ndim == 3:
                    shap_lr_dict = dict(zip(X3_.columns.tolist(), lr_sv[pred_stage, 0, :]))
                else:
                    shap_lr_dict = dict(zip(X3_.columns.tolist(), lr_sv[0, :]))
 
                if shap_lr_dict:
                    fig_lr = _plot_shap_bar(shap_lr_dict,
                                            f"Demographic Factors Influencing the Stage {pred_stage} Prediction",
                                            "#8b5cf6", "#06b6d4", 10, CLINICAL_CONTEXT, LR_EXPECTED)
                    if fig_lr:
                        st.pyplot(fig_lr, use_container_width=True); plt.close()
 
                if lr_sv.ndim == 3 and lr_sv.shape[0] == N_CLASSES:
                    st.markdown('<p class="sect-hdr">How Demographic Factors Affect Each Stage</p>', unsafe_allow_html=True)
                    top_lr = sorted(shap_lr_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:7]
                    lr_feat_labels = [_display_value(f, 0, CLINICAL_CONTEXT)[0] for f, _ in top_lr]
                    lr_shap_matrix = np.array([[lr_sv[c, 0, list(X3_.columns).index(f)] for f, _ in top_lr]
                                               for c in range(N_CLASSES)])
                    fig_lrheat, ax_lrheat = plt.subplots(figsize=(9, 4))
                    im2 = ax_lrheat.imshow(lr_shap_matrix, cmap="RdBu_r", aspect="auto",
                                           norm=mcolors.TwoSlopeNorm(vcenter=0))
                    ax_lrheat.set_xticks(range(len(lr_feat_labels)))
                    ax_lrheat.set_xticklabels(lr_feat_labels, rotation=35, ha="right", fontsize=8.5)
                    ax_lrheat.set_yticks(range(N_CLASSES))
                    ax_lrheat.set_yticklabels([STAGE_NAMES[c] for c in range(N_CLASSES)], fontsize=8.5)
                    plt.colorbar(im2, ax=ax_lrheat, shrink=0.8, label="Influence (SHAP value)")
                    ax_lrheat.set_title("Demographic Factor Influence Across All Stages", fontsize=10, fontweight="700")
                    plt.tight_layout()
                    st.pyplot(fig_lrheat, use_container_width=True); plt.close()
            except Exception as exc:
                st.warning(f"Demographic SHAP unavailable: {exc}")
 
            st.markdown("""
            <div class="rec-box purple">
              <strong>ℹ️ Important note on demographic factors:</strong><br>
              The demographics model identifies statistical patterns in the training data.
              Differences by age, ethnicity, or sex reflect documented differences in kidney disease
              rates in the population — not model bias. This AI was independently audited for fairness.
            </div>""", unsafe_allow_html=True)
        else:
            st.warning("Demographic SHAP analysis unavailable for this patient.")
 
    # ── Tab 5: Progression ────────────────────────────────────────────────────
    with tab5:
        st.markdown('<p class="sect-hdr">Will This Patient\'s Kidney Disease Progress?</p>', unsafe_allow_html=True)
 
        if pred_stage >= N_CLASSES - 1:
            st.markdown("""
            <div class="verdict-no">
              <div class="verdict-title" style="color:#14532d">ℹ️ Stage 5 — Kidney Replacement Required</div>
              <div class="verdict-body">Already at the most advanced stage. Focus on renal replacement therapy.</div>
            </div>""", unsafe_allow_html=True)
        else:
            next_stage_name = STAGE_NAMES[pred_stage + 1]
            if progression_verdict == "YES":
                vhtml = f"""<div class="verdict-yes">
                  <div class="verdict-title" style="color:#be123c">{prog_icon} YES — DISEASE PROGRESSION IS LIKELY</div>
                  <div class="verdict-body">
                    <strong>{forward_prob*100:.0f}%</strong> chance of progressing beyond <em>{stage_name}</em>
                    toward <strong>{next_stage_name}</strong> if risk factors are not addressed.<br><br>
                    <strong>Action:</strong> Prioritise clinical actions listed above. Consider referral and closer monitoring.
                  </div>
                </div>"""
            elif progression_verdict == "POSSIBLE":
                vhtml = f"""<div class="verdict-watch">
                  <div class="verdict-title" style="color:#92400e">{prog_icon} POSSIBLE — MONITOR CLOSELY</div>
                  <div class="verdict-body">
                    <strong>{forward_prob*100:.0f}%</strong> probability of worsening to <strong>{next_stage_name}</strong>.
                    With good management, progression may be preventable.<br><br>
                    <strong>Action:</strong> Increase monitoring frequency and review modifiable risk factors.
                  </div>
                </div>"""
            else:
                vhtml = f"""<div class="verdict-no">
                  <div class="verdict-title" style="color:#14532d">{prog_icon} NO — STAGE APPEARS STABLE</div>
                  <div class="verdict-body">
                    Only <strong>{forward_prob*100:.0f}%</strong> chance of progression at this time.
                    Current management appears effective.<br><br>
                    <strong>Action:</strong> Continue current management plan and routine monitoring.
                  </div>
                </div>"""
            st.markdown(vhtml, unsafe_allow_html=True)
 
            if risk_drivers:
                st.markdown('<p class="sect-hdr">Factors Pushing Toward the Next Stage</p>', unsafe_allow_html=True)
                col_risk, col_prot = st.columns([3, 2])
                with col_risk:
                    r_feats = [_display_value(f, 0, CLINICAL_CONTEXT)[0] for f, _ in risk_drivers]
                    r_vals  = [v for _, v in risk_drivers]
                    r_raws  = [_display_value(f, X1_.iloc[0].get(f, np.nan), CLINICAL_CONTEXT)[1]
                               for f, _ in risk_drivers]
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
                        st.markdown("**✅ Protective factors (slowing progression)**")
                        for feat, sv in prot_drivers:
                            disp_name, _ = _display_value(feat, X1_.iloc[0].get(feat, np.nan), CLINICAL_CONTEXT)
                            _, _, desc   = CLINICAL_CONTEXT.get(feat, (None, None, ""))
                            st.markdown(f"""
                            <div class="driver-row prot">
                              <span><strong>{disp_name}</strong><br>
                                <small style="color:#475569">{desc}</small>
                              </span>
                              <span style="color:#16a34a;font-weight:700;font-size:.9rem">✓</span>
                            </div>""", unsafe_allow_html=True)
 
        # Severity gauge
        st.markdown('<p class="sect-hdr">Disease Severity Gauge</p>', unsafe_allow_html=True)
        fig_gauge, ax_g = plt.subplots(figsize=(9, 1.8))
        ax_g.barh(["Score"], [prog_score], color=stage_color, alpha=0.85, height=0.45)
        ax_g.set_xlim(0, 5)
        for sv in range(6):
            ax_g.axvline(sv, color="#e2e8f0", lw=0.8)
            ax_g.text(sv + 0.07, 0.28, f"Stage {sv}", fontsize=7.5, color="#64748b")
        ax_g.axvline(pred_stage, color="#0f172a", lw=2, ls="--", alpha=0.7)
        ax_g.set_xlabel("Disease Severity (0 = No DKD → 5 = Kidney Failure)", fontsize=9)
        ax_g.spines[["top", "right", "left"]].set_visible(False)
        ax_g.set_yticks([])
        ax_g.text(prog_score + 0.07, 0, f"{prog_score:.2f}", fontsize=10,
                  fontweight="800", va="center", color=stage_color)
        plt.tight_layout()
        st.pyplot(fig_gauge, use_container_width=True); plt.close()
 
    # ── Tab 6: Fairness ───────────────────────────────────────────────────────
    with tab6:
        sex_label  = "Female" if patient_data.get("sex_code") == 1 else "Male"
        race_map   = {1:"Mexican American",2:"Other Hispanic",3:"NH White",
                      4:"NH Black",6:"NH Asian",7:"Other/Multi"}
        race_label = race_map.get(patient_data.get("race_ethnicity_code"), "Unknown")
        age_v      = patient_data.get("age_years", 0)
        age_group  = "<40" if age_v < 40 else "40–55" if age_v < 55 else "55–65" if age_v < 65 else "65+"
        inc        = patient_data.get("household_income_cat", 8)
        inc_label  = "Low (<$25k)" if inc <= 5 else ("Middle ($25–64k)" if inc <= 9 else "High (≥$65k)")
 
        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1: st.metric("Sex",        sex_label)
        with cf2: st.metric("Ethnicity",  race_label)
        with cf3: st.metric("Age Group",  age_group)
        with cf4: st.metric("Income Tier",inc_label)
 
        st.markdown("""
        <div class="rec-box blue">
          <strong>What does the fairness audit mean for this patient?</strong><br>
          Before deployment this AI was tested across sex, ethnicity, age, and income groups
          to ensure consistent performance:
          <ul style="margin:.5rem 0 0;padding-left:1.2rem">
            <li>Accuracy difference between groups: ≤ 5 percentage points</li>
            <li>Sensitivity (catching real cases): ≥ 60% for all groups</li>
            <li>No group has > 15 percentage point sensitivity gap</li>
          </ul>
          Any differences reflect real population-level disease rate differences, not AI bias.
        </div>""", unsafe_allow_html=True)
 
        if "summary_df" in mdl and mdl["summary_df"] is not None:
            st.markdown('<p class="sect-hdr">Official Fairness Audit Results</p>', unsafe_allow_html=True)
            st.dataframe(mdl["summary_df"], use_container_width=True)
            st.caption("PASS = meets threshold · FAIL = below threshold · NOTE = expected clinical difference")
        else:
            st.info("Fairness audit table not available in the loaded model file.")
 
    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      <strong>⚕️ Demo Disclaimer:</strong>
      These are synthetic patient profiles for demonstration purposes only — not real patient data.
      All AI outputs are for illustrative use and must not be used for clinical decision-making.
      The full system was trained on NHANES 2015–2020 cross-sectional data (n ≈ 6,600).
    </div>""", unsafe_allow_html=True)
