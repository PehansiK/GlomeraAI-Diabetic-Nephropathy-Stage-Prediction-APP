# 🔬 GlomeraAI - DKD Clinical Decision Support System

A multimodel, explainable, and fairness-audited AI system for predicting Diabetic Kidney Disease (DKD) progression across six KDIGO 2022 stages.

Built with a stacked ensemble of three sub-models (Random Forest · XGBoost · Logistic Regression) trained on NHANES 2015–2020 cross-sectional data.

---
## Model Development & Streamlit App

>**Model Repo:** [Diabetic-Nephropathy-Stage-Prediction](https://github.com/PehansiK/Diabetic-Nephropathy-Stage-Prediction)

>**Streamlit App:** [🔬GlomeraAI](https://glomera-ai.streamlit.app/)

---
## Features

- **Six-class KDIGO staging** — predicts DKD stage 0 (No DKD) through stage 5 (Kidney Failure)
- **Three-model ensemble** — clinical, lifestyle, and demographic sub-models combined via a meta-learner
- **SHAP explainability** — per-patient factor-level explanations for each sub-model
- **Progression risk** — identifies factors driving progression to the next stage
- **Fairness audit** — pre-deployment evaluation across sex, ethnicity, age, and income groups
- **Demo mode** — six clinically calibrated synthetic patient profiles (one per stage)

---

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── pages/
│   └── demo_page.py        # Demo mode — pre-filled synthetic patient profiles
├── .streamlit/
│   └── config.toml         # Theme and server settings
├── requirements.txt        # Python dependencies
└── packages.txt            # System-level apt packages (Streamlit Cloud)
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/PehansiK/GlomeraAI-Diabetic-Nephropathy-Stage-Prediction-APP.git
cd GlomeraAI-Diabetic-Nephropathy-Stage-Prediction-APP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the model secret

The trained model (`DKD_complete_artifacts.pkl`) is hosted on Hugging Face and fetched at runtime.

Create `.streamlit/secrets.toml` (this file is gitignored):

```toml
MODEL_URL = "https://huggingface.co/pehansiK-20/Diabetic-Nephropathy-Model/resolve/main/DKD_complete_artifacts.pkl"
```

### 4. Run locally

```bash
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub.
2. Connect the repo on [share.streamlit.io](https://share.streamlit.io).
3. Add `MODEL_URL` under **Settings → Secrets**.
4. Deploy — Streamlit Cloud will install `packages.txt` (apt) then `requirements.txt` (pip) automatically.

---

## Model Architecture

| Sub-model | Algorithm | Feature group |
|-----------|-----------|---------------|
| M1 | Random Forest | Clinical blood tests and kidney markers |
| M2 | XGBoost | Lifestyle, medications, and dietary intake |
| M3 | Logistic Regression | Demographics and medical history |
| Meta-learner | Logistic Regression | Stacked probabilities from M1, M2, M3 |

eGFR is derived using the CKD-EPI 2021 (race-free) equation. DKD staging follows KDIGO 2022 criteria.

---

## Data Source

NHANES 2015–2016 and 2017–2020 Pre-pandemic cycles. Diabetic participants with valid eGFR and UACR measurements only.

---

## Disclaimer

This system is a clinical decision support tool and does not replace clinical judgement. All predictions must be reviewed by a qualified clinician alongside the full patient history, physical examination, and current clinical guidelines.
