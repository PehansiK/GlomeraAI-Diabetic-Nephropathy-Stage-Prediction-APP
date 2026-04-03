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
