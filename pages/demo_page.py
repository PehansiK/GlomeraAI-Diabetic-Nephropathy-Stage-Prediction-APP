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
