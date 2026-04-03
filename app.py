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
