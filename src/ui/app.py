"""Interactive Web Application for Coripet RVM Prospective Location Analysis.

Production decision support dashboard powered by:
- Multi-Task Tabular Neural Network (Entity Embeddings + Dual Regression Heads)
- LightGBM Stacking Ensemble
- Spatial Catchment & Cannibalization Analyzer
- Nielsen Retail Scale & Store Profile Registry
- Commercial Rewards & Filiera Agreement Engine
- 4-Gate Operational Suitability & Clearance Engine
"""

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import math
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

# Page configuration
st.set_page_config(
    page_title="Coripet RVM Location Predictor",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Ecological Theme Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    
    /* Flex alignment for Streamlit columns so cards in rows match height */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div.stHtml {
        display: flex !important;
        flex-direction: column !important;
        flex-grow: 1 !important;
    }

    /* Modern Adaptive Segmented Control Navigation Tabs */
    div[data-testid="stSegmentedControl"] {
        background: rgba(128, 128, 128, 0.08) !important;
        border-radius: 12px !important;
        padding: 5px !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
        color: inherit !important;
        opacity: 0.75 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        opacity: 1.0 !important;
        background: rgba(128, 128, 128, 0.12) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
        color: #ffffff !important;
        opacity: 1.0 !important;
        box-shadow: 0 2px 10px rgba(22, 163, 74, 0.45) !important;
        font-weight: 700 !important;
    }

    /* Universal Adaptive Badge Classes (Light & Dark Theme) */
    .badge-pf-id {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #334155;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-family: monospace;
        font-weight: 700;
        display: inline-block;
    }
    .badge-pf-filiera {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-pf-pv {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 800;
        display: inline-block;
    }
    .badge-pf-spl {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.3px;
        display: inline-block;
    }
    .badge-pf-pot {
        background: #faf5ff;
        border: 1px solid #e9d5ff;
        color: #7e22ce;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.3px;
        display: inline-block;
    }
    .badge-pf-model {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        color: #1e293b;
        padding: 5px 11px;
        border-radius: 7px;
        font-size: 0.80rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .directory-banner {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        color: inherit;
    }
    .badge-stat-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-stat-green {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-stat-purple {
        background: #faf5ff;
        color: #7e22ce;
        border: 1px solid #e9d5ff;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-source-live {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-source-backup {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-tier-super {
        background: #faf5ff;
        color: #7e22ce;
        border: 1px solid #d8b4fe;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.70rem;
        letter-spacing: 0.3px;
        display: inline-block;
    }
    .badge-tier-alto {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.70rem;
        letter-spacing: 0.3px;
        display: inline-block;
    }
    .badge-tier-medio {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.70rem;
        letter-spacing: 0.3px;
        display: inline-block;
    }
    .badge-tier-basso {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 800;
        font-size: 0.70rem;
        letter-spacing: 0.3px;
        display: inline-block;
    }
    .badge-rank-1 {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border: 1px solid #f59e0b;
        font-weight: 800;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        display: inline-block;
        box-shadow: 0 1px 3px rgba(245, 158, 11, 0.25);
    }
    .badge-rank-2 {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        color: #334155;
        border: 1px solid #94a3b8;
        font-weight: 800;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-rank-3 {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
        color: #9a3412;
        border: 1px solid #f97316;
        font-weight: 800;
        padding: 3px 9px;
        border-radius: 6px;
        font-size: 0.78rem;
        display: inline-block;
    }
    .badge-rank-other {
        background: rgba(128, 128, 128, 0.10);
        color: inherit;
        border: 1px solid rgba(128, 128, 128, 0.25);
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        display: inline-block;
    }

    /* Dark Mode Overrides (Streamlit theme & system preference) */
    @media (prefers-color-scheme: dark) {
        .badge-tier-super { background: rgba(168, 85, 247, 0.22); border-color: rgba(168, 85, 247, 0.50); color: #d8b4fe; }
        .badge-tier-alto { background: rgba(16, 185, 129, 0.22); border-color: rgba(16, 185, 129, 0.50); color: #6ee7b7; }
        .badge-tier-medio { background: rgba(59, 130, 246, 0.22); border-color: rgba(59, 130, 246, 0.50); color: #93c5fd; }
        .badge-tier-basso { background: rgba(239, 68, 68, 0.22); border-color: rgba(239, 68, 68, 0.50); color: #fca5a5; }
        .badge-rank-1 { background: rgba(245, 158, 11, 0.25); color: #fde047; border-color: rgba(245, 158, 11, 0.60); }
        .badge-rank-2 { background: rgba(148, 163, 184, 0.25); color: #f1f5f9; border-color: rgba(148, 163, 184, 0.60); }
        .badge-rank-3 { background: rgba(249, 115, 22, 0.25); color: #fdba74; border-color: rgba(249, 115, 22, 0.60); }
        .badge-pf-id {
            background: rgba(148, 163, 184, 0.18);
            border-color: rgba(148, 163, 184, 0.40);
            color: #cbd5e1;
        }
        .badge-pf-filiera {
            background: rgba(59, 130, 246, 0.20);
            border-color: rgba(59, 130, 246, 0.48);
            color: #93c5fd;
        }
        .badge-pf-pv {
            background: rgba(16, 185, 129, 0.20);
            border-color: rgba(16, 185, 129, 0.48);
            color: #6ee7b7;
        }
        .badge-pf-spl {
            background: rgba(245, 158, 11, 0.22);
            border-color: rgba(245, 158, 11, 0.50);
            color: #fbbf24;
        }
        .badge-pf-pot {
            background: rgba(168, 85, 247, 0.22);
            border-color: rgba(168, 85, 247, 0.50);
            color: #d8b4fe;
        }
        .badge-pf-model {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.18);
            color: #f1f5f9;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);
        }
        .directory-banner {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.90) 100%);
            border-color: rgba(255, 255, 255, 0.12);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }
        .badge-stat-blue {
            background: rgba(56, 189, 248, 0.14);
            color: #38bdf8;
            border-color: rgba(56, 189, 248, 0.35);
        }
        .badge-stat-green {
            background: rgba(52, 211, 153, 0.14);
            color: #34d399;
            border-color: rgba(52, 211, 153, 0.35);
        }
        .badge-stat-purple {
            background: rgba(168, 85, 247, 0.14);
            color: #c084fc;
            border-color: rgba(168, 85, 247, 0.35);
        }
        .badge-source-live {
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            border-color: rgba(16, 185, 129, 0.45);
        }
        .badge-source-backup {
            background: rgba(245, 158, 11, 0.18);
            color: #fbbf24;
            border-color: rgba(245, 158, 11, 0.45);
        }
    }
    [data-theme="dark"] .badge-pf-id, .stApp[data-theme="dark"] .badge-pf-id { background: rgba(148, 163, 184, 0.18); border-color: rgba(148, 163, 184, 0.40); color: #cbd5e1; }
    [data-theme="dark"] .badge-pf-filiera, .stApp[data-theme="dark"] .badge-pf-filiera { background: rgba(59, 130, 246, 0.20); border-color: rgba(59, 130, 246, 0.48); color: #93c5fd; }
    [data-theme="dark"] .badge-pf-pv, .stApp[data-theme="dark"] .badge-pf-pv { background: rgba(16, 185, 129, 0.20); border-color: rgba(16, 185, 129, 0.48); color: #6ee7b7; }
    [data-theme="dark"] .badge-pf-spl, .stApp[data-theme="dark"] .badge-pf-spl { background: rgba(245, 158, 11, 0.22); border-color: rgba(245, 158, 11, 0.50); color: #fbbf24; }
    [data-theme="dark"] .badge-pf-pot, .stApp[data-theme="dark"] .badge-pf-pot { background: rgba(168, 85, 247, 0.22); border-color: rgba(168, 85, 247, 0.50); color: #d8b4fe; }
    [data-theme="dark"] .badge-pf-model, .stApp[data-theme="dark"] .badge-pf-model { background: rgba(255, 255, 255, 0.08); border-color: rgba(255, 255, 255, 0.18); color: #f1f5f9; }
    [data-theme="dark"] .directory-banner, .stApp[data-theme="dark"] .directory-banner { background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.90) 100%); border-color: rgba(255, 255, 255, 0.12); }
    [data-theme="dark"] .badge-stat-blue, .stApp[data-theme="dark"] .badge-stat-blue { background: rgba(56, 189, 248, 0.14); color: #38bdf8; border-color: rgba(56, 189, 248, 0.35); }
    [data-theme="dark"] .badge-stat-green, .stApp[data-theme="dark"] .badge-stat-green { background: rgba(52, 211, 153, 0.14); color: #34d399; border-color: rgba(52, 211, 153, 0.35); }
    [data-theme="dark"] .badge-stat-purple, .stApp[data-theme="dark"] .badge-stat-purple { background: rgba(168, 85, 247, 0.14); color: #c084fc; border-color: rgba(168, 85, 247, 0.35); }
    [data-theme="dark"] .badge-source-live, .stApp[data-theme="dark"] .badge-source-live { background: rgba(16, 185, 129, 0.18); color: #34d399; border-color: rgba(16, 185, 129, 0.45); }
    [data-theme="dark"] .badge-source-backup, .stApp[data-theme="dark"] .badge-source-backup { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border-color: rgba(245, 158, 11, 0.45); }

    /* Light Theme Explicit Overrides */
    [data-theme="light"] .badge-pf-id, .stApp[data-theme="light"] .badge-pf-id { background: #f1f5f9 !important; border-color: #cbd5e1 !important; color: #334155 !important; }
    [data-theme="light"] .badge-pf-filiera, .stApp[data-theme="light"] .badge-pf-filiera { background: #eff6ff !important; border-color: #bfdbfe !important; color: #1d4ed8 !important; }
    [data-theme="light"] .badge-pf-pv, .stApp[data-theme="light"] .badge-pf-pv { background: #ecfdf5 !important; border-color: #a7f3d0 !important; color: #047857 !important; }
    [data-theme="light"] .badge-pf-spl, .stApp[data-theme="light"] .badge-pf-spl { background: #fffbeb !important; border-color: #fde68a !important; color: #b45309 !important; }
    [data-theme="light"] .badge-pf-pot, .stApp[data-theme="light"] .badge-pf-pot { background: #faf5ff !important; border-color: #e9d5ff !important; color: #7e22ce !important; }
    [data-theme="light"] .badge-pf-model, .stApp[data-theme="light"] .badge-pf-model { background: #f8fafc !important; border-color: #cbd5e1 !important; color: #1e293b !important; }
    [data-theme="light"] .directory-banner, .stApp[data-theme="light"] .directory-banner { background: #ffffff !important; border-color: #e2e8f0 !important; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04) !important; }
    [data-theme="light"] .badge-stat-blue, .stApp[data-theme="light"] .badge-stat-blue { background: #eff6ff !important; color: #1d4ed8 !important; border-color: #bfdbfe !important; }
    [data-theme="light"] .badge-stat-green, .stApp[data-theme="light"] .badge-stat-green { background: #ecfdf5 !important; color: #047857 !important; border-color: #a7f3d0 !important; }
    [data-theme="light"] .badge-stat-purple, .stApp[data-theme="light"] .badge-stat-purple { background: #faf5ff !important; color: #7e22ce !important; border-color: #e9d5ff !important; }
    [data-theme="light"] .badge-source-live, .stApp[data-theme="light"] .badge-source-live { background: #ecfdf5 !important; color: #047857 !important; border-color: #a7f3d0 !important; }
    [data-theme="light"] .badge-source-backup, .stApp[data-theme="light"] .badge-source-backup { background: #fffbeb !important; color: #b45309 !important; border-color: #fde68a !important; }

    /* Executive Status Cards (Section 1) */
    .exec-card {
        background: #ffffff;
        color: #0f172a !important;
        border-radius: 12px;
        padding: 12px 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        min-height: 114px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .exec-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
    }
    .exec-card-verdict {
        border-top: 3px solid #10b981;
    }
    .exec-card-conf {
        border-top: 3px solid #3b82f6;
    }
    .exec-card-geo {
        border-top: 3px solid #8b5cf6;
    }
    .exec-card-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
        margin-bottom: 4px;
    }
    .exec-card-lbl {
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #64748b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .exec-status-dot {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        padding: 1px 7px;
        border-radius: 10px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .dot-green {
        background: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .dot-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .dot-purple {
        background: #faf5ff;
        color: #6d28d9;
        border: 1px solid #e9d5ff;
    }
    .exec-verdict-box {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        margin: 2px 0 3px 0;
    }
    .exec-verdict-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.2px;
        line-height: 1.25;
    }
    .exec-action-chip {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 5px;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .chip-proceed {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .chip-review {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    .chip-unfavorable {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .exec-card-bot {
        border-top: 1px solid #f1f5f9;
        padding-top: 5px;
        margin-top: auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.73rem;
    }
    .exec-bot-label {
        color: #64748b;
        font-weight: 500;
    }
    .exec-bot-val {
        color: #0f172a;
        font-weight: 700;
    }
    .exec-conf-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1d4ed8;
        letter-spacing: -0.2px;
        line-height: 1.25;
        margin: 2px 0 1px 0;
    }
    .exec-conf-sub {
        font-size: 0.72rem;
        color: #64748b;
        font-weight: 500;
        line-height: 1.2;
    }
    .exec-geo-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.2px;
        line-height: 1.25;
        margin: 2px 0 1px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .exec-geo-reg {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 500;
    }
    .exec-cap-pill {
        background: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
        padding: 1px 6px;
        border-radius: 5px;
        font-size: 0.65rem;
        font-weight: 700;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .exec-chip {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 2px 6px;
        border-radius: 5px;
        font-size: 0.68rem;
        font-weight: 600;
        color: #334155;
        white-space: nowrap;
    }
    .exec-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #475569;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .exec-sub {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
    }

    /* Key Operational Metric Cards (Section 2) */
    .metric-card {
        background: #ffffff;
        color: #0f172a !important;
        border-radius: 14px;
        padding: 14px 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 8px;
        min-height: 140px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
    }
    .metric-card-blue { border-top: 4px solid #2563eb; }
    .metric-card-green { border-top: 4px solid #16a34a; }
    .metric-card-purple { border-top: 4px solid #7c3aed; }
    .metric-card-amber { border-top: 4px solid #ea580c; }
    .metric-card-teal { border-top: 4px solid #0d9488; }

    .metric-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        color: #64748b;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-icon-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        font-size: 0.90rem;
    }
    .badge-tint-blue { background: #eff6ff; border: 1px solid #dbeafe; }
    .badge-tint-green { background: #f0fdf4; border: 1px solid #dcfce7; }
    .badge-tint-purple { background: #faf5ff; border: 1px solid #f3e8ff; }
    .badge-tint-amber { background: #fff7ed; border: 1px solid #ffedd5; }
    .badge-tint-teal { background: #f0fdfa; border: 1px solid #ccfbf1; }

    .metric-num-wrap {
        display: flex;
        align-items: baseline;
        gap: 5px;
        margin: 4px 0 2px 0;
    }
    .metric-num {
        font-size: 1.95rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    .num-blue { color: #1d4ed8; }
    .num-green { color: #15803d; }
    .num-purple { color: #6d28d9; }
    .num-amber { color: #c2410c; }
    .num-teal { color: #0f766e; }

    .metric-unit {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.2px;
    }
    .metric-subtitle {
        font-size: 0.76rem;
        color: #64748b;
        font-weight: 500;
        line-height: 1.3;
        border-top: 1px solid #f1f5f9;
        padding-top: 6px;
        margin-top: auto;
    }
    .metric-label {
        font-size: 0.76rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700;
        margin-bottom: 2px;
        line-height: 1.25;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #1b5e20;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.80rem;
        color: #64748b;
        font-weight: 600;
        margin-top: 2px;
    }

    /* Operational Clearance Gate Cards (Section 5) */
    .gate-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    @media (max-width: 1080px) {
        .gate-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 680px) {
        .gate-grid {
            grid-template-columns: 1fr;
        }
    }
    .gate-card {
        background: #ffffff;
        color: #0f172a !important;
        border-radius: 12px;
        padding: 14px 16px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        min-height: 135px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .gate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.09);
    }
    .gate-pass {
        border-left: 5px solid #2e7d32;
        background: #ffffff;
    }
    .gate-fail {
        border-left: 5px solid #c62828;
        background: #ffffff;
    }
    .gate-warn {
        border-left: 5px solid #f57f17;
        background: #ffffff;
    }
    .gate-title {
        font-weight: 700;
        color: #0f172a;
        font-size: 0.95rem;
        margin-bottom: 6px;
        line-height: 1.3;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .gate-metric {
        font-size: 0.84rem;
        color: #334155;
        line-height: 1.45;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .gate-footer {
        border-top: 1px solid #f1f5f9;
        padding-top: 8px;
        margin-top: auto;
    }
    .gate-status-pill {
        display: block;
        width: 100%;
        text-align: center;
        padding: 5px 12px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        box-sizing: border-box;
    }
    .pill-pass {
        background: #e8f5e9;
        color: #1b5e20;
        border: 1px solid #a5d6a7;
    }
    .pill-warn {
        background: #fff8e1;
        color: #b78103;
        border: 1px solid #ffe082;
    }
    .pill-fail {
        background: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }

    /* Deep Intelligence Section Cards (Section 4 & 7) */
    .intel-section {
        background: #ffffff;
        color: #0f172a !important;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        box-sizing: border-box;
    }

    /* Terminal Verbatim Box */
    .terminal-box {
        background: #14171a;
        color: #e1e8ed;
        font-family: 'SF Mono', 'Fira Code', 'Courier New', Courier, monospace;
        padding: 20px 24px;
        border-radius: 12px;
        white-space: pre-wrap;
        font-size: 1.05rem;
        line-height: 1.6;
        border: 1px solid #334155;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        height: 100%;
        min-height: 380px;
        box-sizing: border-box;
    }

    /* Badges */
    .badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        letter-spacing: 0.3px;
    }
    .badge-high {
        background-color: #e8f5e9;
        color: #1b5e20;
        border: 1px solid #a5d6a7;
    }
    .badge-medium {
        background-color: #fff8e1;
        color: #b78103;
        border: 1px solid #ffe082;
    }
    .badge-low {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }
    .badge-reward {
        background-color: #e3f2fd;
        color: #0d47a1;
        border: 1px solid #90caf9;
    }

    /* Segmented Navigation Tab Bar */
    div[data-testid="stSegmentedControl"] {
        margin-bottom: 22px !important;
    }
    div[data-testid="stSegmentedControl"] > div {
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 6px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 8px 22px !important;
        transition: all 0.15s ease-in-out !important;
        border: none !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-selected="true"],
    div[data-testid="stSegmentedControl"] button[data-selected="true"] {
        background-color: #1b5e20 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 6px rgba(27, 94, 32, 0.25) !important;
    }

    /* Strict Select-Only Dropdowns & Filter Fields (No Text Editing or Typing Cursor) */
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
        user-select: none !important;
        -webkit-user-select: none !important;
    }
    div[data-baseweb="select"] {
        cursor: pointer !important;
    }
    div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Enforce read-only select behavior: users can strictly choose from dropdown, not edit/type characters
components.html(
    """
    <script>
    (function() {
        function enforceReadOnlySelects() {
            try {
                const doc = window.parent.document;
                if (!doc) return;
                const inputs = doc.querySelectorAll('div[data-baseweb="select"] input');
                inputs.forEach(inp => {
                    if (!inp.hasAttribute('readonly')) {
                        inp.setAttribute('readonly', 'readonly');
                        inp.setAttribute('inputmode', 'none');
                        inp.style.caretColor = 'transparent';
                        inp.style.cursor = 'pointer';
                        inp.addEventListener('keydown', function(e) {
                            if (['ArrowDown', 'ArrowUp', 'Enter', 'Escape', 'Tab'].includes(e.key)) {
                                return;
                            }
                            e.preventDefault();
                            e.stopPropagation();
                        }, true);
                        inp.addEventListener('paste', function(e) { e.preventDefault(); }, true);
                        inp.addEventListener('cut', function(e) { e.preventDefault(); }, true);
                    }
                });
            } catch(e) {}
        }
        enforceReadOnlySelects();
        if (window.parent && window.parent.document && window.parent.document.body) {
            const observer = new MutationObserver(enforceReadOnlySelects);
            observer.observe(window.parent.document.body, { childList: true, subtree: true });
        }
    })();
    </script>
    """,
    height=0,
    width=0,
)


@st.cache_resource
def load_production_predictor():
    """Loads production RVM prospective location predictor and models."""
    try:
        import sys
        import importlib
        if "src.model.predict" in sys.modules:
            importlib.reload(sys.modules["src.model.predict"])
        from src.model.predict import RVMNewLocationPredictor
        predictor = RVMNewLocationPredictor()
        return predictor, True, None
    except Exception as e:
        # Fallback to scaffold predictor if dependencies or checkpoints missing
        try:
            from src.inference.predictor import LocationPredictor
            return LocationPredictor(), False, str(e)
        except Exception as e2:
            return None, False, f"Primary: {e} | Fallback: {e2}"



def render_gauge(score: float, title: str = "Location Suitability") -> go.Figure:
    """Generates a gauge indicator for the suitability score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"size": 38, "family": "Inter"}},
            title={"text": title, "font": {"size": 17, "family": "Inter", "color": "#333"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9e9e9e"},
                "bar": {"color": "#1b5e20", "thickness": 0.35},
                "bgcolor": "#f5f5f5",
                "borderwidth": 1,
                "bordercolor": "#ccc",
                "steps": [
                    {"range": [0, 50], "color": "#ffebee"},
                    {"range": [50, 75], "color": "#fff8e1"},
                    {"range": [75, 100], "color": "#e8f5e9"},
                ],
                "threshold": {
                    "line": {"color": "#212121", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=25, r=25, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_seasonality_chart(monthly_data: Any, steady_state_benchmark: Optional[float] = None) -> go.Figure:
    """Generates rolling monthly volume seasonality and ramp-up projection chart."""
    x_labels = []
    vals = []
    hover_texts = []
    bar_colors = []
    
    if isinstance(monthly_data, list):
        for idx, item in enumerate(monthly_data):
            if isinstance(item, dict):
                v = item.get("daily_bottiglie_median", item.get("daily_bottiglie", 0.0))
                m_tgt = item.get("monthly_target_bottiglie_median", item.get("monthly_target_bottiglie", int(v * 30.4)))
                ctx = item.get("seasonal_context", item.get("context", ""))
                mat_pct = item.get("maturity_pct", "100%")
                stage = item.get("lifecycle_stage", "Operational Run")
                lbl = item.get("lifecycle_label", item.get("month_name", f"Month {idx+1}"))
                x_labels.append(lbl)
                vals.append(v)
                hover_texts.append(f"<b>{lbl}</b><br>Daily Projection: <b>{v:.1f} b/d</b><br>Monthly Target: <b>{m_tgt:,} b/mo</b><br>Maturity Factor: <b>{mat_pct}</b> ({stage})<br>Context: {ctx}")
                
                mat_f = float(item.get("maturity_factor", 1.0))
                if mat_f < 0.75:
                    bar_colors.append("#81c784")  # Soft green for discovery month
                elif mat_f < 0.95:
                    bar_colors.append("#4caf50")  # Medium green for adoption phase
                else:
                    bar_colors.append("#2e7d32")  # Deep forest green for mature phase
            elif isinstance(item, (int, float)):
                x_labels.append(f"M{idx+1}")
                vals.append(float(item))
                hover_texts.append(f"{float(item):.1f} b/d")
                bar_colors.append("#2e7d32")
    elif isinstance(monthly_data, dict):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for m in range(1, 13):
            sub = monthly_data.get(m, {})
            lbl = months[m-1]
            x_labels.append(lbl)
            if isinstance(sub, dict):
                v = sub.get("daily_bottiglie_median", sub.get("mean_daily_bottles", 0.0))
                vals.append(v)
                hover_texts.append(f"{v:.1f} b/d")
            elif isinstance(sub, (int, float)):
                vals.append(float(sub))
                hover_texts.append(f"{float(sub):.1f} b/d")
            bar_colors.append("#2e7d32")

    if len(vals) < 12:
        vals = (vals + [500.0] * 12)[:12]
        x_labels = (x_labels + [f"M{i+1}" for i in range(12)])[:12]
        bar_colors = (bar_colors + ["#2e7d32"] * 12)[:12]
    if len(hover_texts) < 12:
        hover_texts = [f"{v:.1f} b/d" for v in vals]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=vals,
            marker_color=bar_colors,
            opacity=0.90,
            name="Projected Daily Intake",
            text=[f"{int(round(v))}" for v in vals],
            textposition="auto",
            hovertext=hover_texts,
            hoverinfo="text",
        )
    )
    
    # Calculate steady-state benchmark from mature months 4 to 12 (excluding initial 3-month ramp-up)
    mature_benchmark = None
    if len(vals) >= 12:
        mature_benchmark = float(np.mean(vals[3:]))
    elif steady_state_benchmark and steady_state_benchmark > 0:
        mature_benchmark = float(steady_state_benchmark)

    if mature_benchmark and mature_benchmark > 0:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=[mature_benchmark] * len(x_labels),
                mode="lines",
                line=dict(color="#1976d2", width=2, dash="dash"),
                name=f"Steady-State Capacity ({mature_benchmark:.0f} b/d)",
                hoverinfo="name+y",
            )
        )
        
    fig.update_layout(
        title={"text": "📅 12-Month Operational Rolling Intake Profile (b/day)", "font": {"size": 15}},
        xaxis_title="Operational Horizon Month",
        yaxis_title="Bottles / Day",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.02)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


CORIPET_API_URL = "https://macchine.coripet.it/CORIPETRVMAPI/api/PuntiFisici/mt_CaricaViewStoricoPuntoFisico"
CORIPET_SOPRALLUOGHI_API_URL = "https://macchine.coripet.it/CORIPETRVMAPI/api/Sopralluoghi/mt_CaricaSopralluoghi"
CORIPET_API_TOKEN = "ck%N@b$x1$wNwCsigohC"

VALID_MACHINE_OPTIONS = [
    "DAINT_A-bis",
    "DAINT_B-bis",
    "DAINT_G",
    "Ripet_C-front",
]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_raw_coripet_machines() -> Tuple[pd.DataFrame, str]:
    """
    Fetches raw physical machine records directly from the official Coripet RVM API endpoint:
    POST https://macchine.coripet.it/CORIPETRVMAPI/api/PuntiFisici/mt_CaricaViewStoricoPuntoFisico
    Header: token: ck%N@b$x1$wNwCsigohC
    Returns (DataFrame, source_label) where source_label is 'Live Coripet API' or 'Local Backup File'.
    """
    try:
        headers = {
            "token": CORIPET_API_TOKEN,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        resp = requests.post(CORIPET_API_URL, headers=headers, data="", timeout=35)
        if resp.status_code == 200:
            payload = resp.json()
            records = payload.get("data", [])
            if records:
                return pd.DataFrame(records), "Live Coripet API"
    except Exception as ex:
        pass

    # 1. Fallback to PostgreSQL database if DATABASE_URL is configured
    pg_url = os.getenv("DATABASE_URL", "").strip()
    if pg_url and (pg_url.startswith("postgresql") or pg_url.startswith("postgres")):
        try:
            if pg_url.startswith("postgresql://") and "+" not in pg_url.split("://")[0]:
                pg_url = pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
            elif pg_url.startswith("postgres://"):
                pg_url = pg_url.replace("postgres://", "postgresql+psycopg2://", 1)
            from sqlalchemy import create_engine
            pg_engine = create_engine(pg_url, pool_pre_ping=True)
            df_pg = pd.read_sql("SELECT * FROM raw_storico_punto_fisico", pg_engine)
            if not df_pg.empty:
                col_map = {c.lower(): c for c in [
                    "PuntoFisico", "DataInizioRange", "DataFineRange", "TipologiaPuntoFisico",
                    "NascondiAPP", "Indirizzo", "Cap", "Comune", "Provincia", "Regione", "Nazione",
                    "Latitudine", "Longitudine", "Stato", "Attributo", "Cooperativa", "Filiera",
                    "Descrizione", "TipoContenitore", "ReferenteCommerciale", "ReferenteCoripet",
                    "Comodatario", "NomeFiliera", "Raggruppamento", "Macchina", "CodiceMacchinaProd",
                    "TipoMacchina", "Fornitore", "DataInstallazione", "DataPrimaAttivazione",
                    "start_dt", "end_dt", "Macchina_clean", "CodiceMacchinaProd_clean"
                ]}
                df_pg.rename(columns=col_map, inplace=True)
                return df_pg, "PostgreSQL Database"
        except Exception:
            pass

    # 2. Fallback to local SQLite database if available
    db_path = Path("data/coripet_rvm.db")
    if db_path.exists():
        try:
            import sqlite3
            with sqlite3.connect(str(db_path)) as conn:
                df_db = pd.read_sql("SELECT * FROM raw_storico_punto_fisico", conn)
                if not df_db.empty:
                    return df_db, "Local SQLite Database"
        except Exception:
            pass

    # 3. Fallback to local CSV if database is unavailable
    path = Path("data/raw/CaricaViewStoricoPuntoFisico.csv")
    if path.exists():
        df_csv = pd.read_csv(path, sep=";", dtype=str)
        return df_csv, "Local Backup File"
    return pd.DataFrame(), "None"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_raw_coripet_sopralluoghi() -> Tuple[pd.DataFrame, str]:
    """
    Fetches site survey (sopralluoghi) inspection records directly from official Coripet API:
    POST https://macchine.coripet.it/CORIPETRVMAPI/api/Sopralluoghi/mt_CaricaSopralluoghi
    Header: token: ck%N@b$x1$wNwCsigohC
    Returns (DataFrame, source_label).
    """
    try:
        headers = {
            "token": CORIPET_API_TOKEN,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        resp = requests.post(CORIPET_SOPRALLUOGHI_API_URL, headers=headers, data="", timeout=35)
        if resp.status_code == 200:
            payload = resp.json()
            records = payload.get("data", [])
            if records:
                df_sop = pd.DataFrame(records)
                try:
                    Path("data/raw").mkdir(parents=True, exist_ok=True)
                    df_sop.to_csv("data/raw/CaricaSopralluoghi.csv", sep=";", index=False)
                except Exception:
                    pass
                return df_sop, "Live Coripet API"
    except Exception:
        pass

    # Fallback to local CSV backup if network is unavailable
    path = Path("data/raw/CaricaSopralluoghi.csv")
    if path.exists():
        df_csv = pd.read_csv(path, sep=";", dtype=str)
        return df_csv, "Local Backup File"
    return pd.DataFrame(), "None"


@st.cache_data(ttl=900, show_spinner=False)
def get_machines_by_status(status_str: str, current_only: bool = True) -> Tuple[pd.DataFrame, str, str]:
    """
    Retrieves and enriches machine records from the API for the requested status (e.g. 'SPL' or 'POT').
    When current_only=True (default), only machines whose DataFineRange matches the latest active date
    (e.g. today's snapshot date) are returned, discarding historical records for machines that have since become ATTVO.
    Returns (DataFrame, source_label, current_date_str).
    """
    df, source_label = fetch_raw_coripet_machines()
    if df.empty:
        return pd.DataFrame(), source_label, ""

    # Fast enrichment of missing attributes across historical records for the same PuntoFisico
    enrich_cols = ["Descrizione", "TipoMacchina", "NomeFiliera", "Filiera", "Indirizzo", "Cap", "Comune", "Provincia", "Regione", "TipologiaPuntoFisico"]
    for c in enrich_cols:
        if c in df.columns:
            valid = df[df[c].notna() & (df[c].astype(str).str.strip() != "") & (df[c].astype(str).str.strip().str.lower() != "nan")]
            if not valid.empty:
                c_map = valid.drop_duplicates("PuntoFisico", keep="last").set_index("PuntoFisico")[c].to_dict()
                df[c] = df[c].fillna(df["PuntoFisico"].map(c_map))

    df["dt_fine"] = pd.to_datetime(df["DataFineRange"], format="%d/%m/%Y", errors="coerce")
    df["dt_inizio"] = pd.to_datetime(df["DataInizioRange"], format="%d/%m/%Y", errors="coerce")
    df["Stato_Clean"] = df["Stato"].astype(str).str.upper().str.strip()

    max_dt = df["dt_fine"].max()
    max_dt_str = max_dt.strftime("%d/%m/%Y") if pd.notna(max_dt) else "Current Date"

    # ------------------------------------------------------------------
    # Sopralluoghi Inspection Records (Ordered by DataFineSopralluogo)
    # ------------------------------------------------------------------
    df_sop, sop_source = fetch_raw_coripet_sopralluoghi()
    df_sop_latest = pd.DataFrame()
    if not df_sop.empty and "CodicePuntoFisico" in df_sop.columns:
        df_sop["dt_fine_sop"] = pd.to_datetime(df_sop["DataFineSopralluogo"], errors="coerce")
        df_sop_latest = (
            df_sop.sort_values(by="dt_fine_sop", ascending=True)
            .groupby("CodicePuntoFisico", as_index=False)
            .last()
        )

    # Strictly Filter df for Today's Active Date (DataFineRange == max_dt)
    current_df = df[df["dt_fine"] == max_dt].copy()
    
    # Filter for requested status strictly based on machine final status
    if status_str and status_str.upper().strip() != "ALL":
        target_st = status_str.upper().strip()
        current_df = current_df[current_df["Stato_Clean"] == target_st].copy()
        
    # Deduplicate to keep 1 physical record per PuntoFisico active today
    current_df = current_df.drop_duplicates(subset=["PuntoFisico"], keep="last").copy()

    if not df_sop_latest.empty:
        # Merge active today machines with their latest Sopralluogo inspection
        sub = current_df.merge(
            df_sop_latest[[
                "CodicePuntoFisico", "DataFineSopralluogo", "StatoSopralluogo",
                "TipoMacchina", "DescrizionePuntoFisico", "dt_fine_sop"
            ]],
            left_on="PuntoFisico",
            right_on="CodicePuntoFisico",
            how="inner",
            suffixes=("", "_sop")
        ).copy()

        # Strictly use DataFineSopralluogo: if null, format as "Null"
        sub["DataFineSopralluogoFormatted"] = sub["dt_fine_sop"].apply(
            lambda d: d.strftime("%d/%m/%Y") if pd.notna(d) else "Null"
        )
        sub["StatoSopralluogoClean"] = sub["StatoSopralluogo"].fillna("Definitivo")
        sub["Stato"] = status_str.upper().strip() if status_str and status_str.upper().strip() != "ALL" else sub["Stato"].fillna("INSPECTED")

        # Map TipoMacchina strictly to one of the 4 certified models:
        # Daint A-bis (300), Daint B-Bis (500), Daint G (4500), Ripet C-Front (4500)
        def map_sop_mach(row):
            m_sop = str(row.get("TipoMacchina_sop", "")).strip() or str(row.get("TipoMacchina", "")).strip()
            m_lower = m_sop.lower()
            if "a-bis" in m_lower: return "DAINT_A-bis"
            if "b-bis" in m_lower or "daint_b" in m_lower: return "DAINT_B-bis"
            if "daint_g" in m_lower or "modello g" in m_lower or m_lower == "g": return "DAINT_G"
            return "Ripet_C-front"

        sub["ResolvedMachine"] = sub.apply(map_sop_mach, axis=1)
        sub["TipoMacchina"] = sub["ResolvedMachine"]
    else:
        sub = current_df

    # Sanitize location fields
    sub["Indirizzo"] = sub["Indirizzo"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan" else "Piazza del Comune 1"
    )
    sub["Cap"] = sub["Cap"].apply(
        lambda x: str(x).split(".")[0].strip().zfill(5) if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan" else "00100"
    )
    sub["Comune"] = sub["Comune"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan" else "Milano"
    )
    sub["Provincia"] = sub["Provincia"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan" else ""
    )
    sub["Regione"] = sub["Regione"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan" else ""
    )

    def resolve_name(r):
        desc = str(r.get("Descrizione", "")).strip()
        if desc and desc.lower() != "nan":
            return desc
        dpf = str(r.get("DescrizionePuntoFisico", "")).strip()
        if dpf and dpf.lower() != "nan":
            return dpf
        nf = str(r.get("NomeFiliera", "")).strip()
        if nf and nf.lower() != "nan":
            return nf
        fil = str(r.get("Filiera", "")).strip()
        if fil and fil.lower() != "nan":
            return fil
        com = str(r.get("Comune", "")).strip()
        return f"Store {com}" if com else f"Punto {r.get('PuntoFisico', '')}"

    sub["ResolvedStoreName"] = sub.apply(resolve_name, axis=1)

    def resolve_filiera(r):
        nf = str(r.get("NomeFiliera", "")).strip()
        if nf and nf.lower() != "nan":
            return nf
        fil = str(r.get("Filiera", "")).strip()
        if fil and fil.lower() != "nan":
            return fil
        rag = str(r.get("Raggruppamento", "")).strip()
        if rag and rag.lower() != "nan":
            return rag
        return "Retail Store"

    sub["ResolvedFiliera"] = sub.apply(resolve_filiera, axis=1)

    def resolve_mach(m):
        m_str = str(m).strip().lower()
        if "a-bis" in m_str: return "DAINT_A-bis"
        if "b-bis" in m_str or "daint_b" in m_str: return "DAINT_B-bis"
        if "daint_g" in m_str or "modello g" in m_str or m_str == "g": return "DAINT_G"
        return "Ripet_C-front"

    sub["ResolvedMachine"] = sub["TipoMacchina"].apply(resolve_mach)
    return sub, source_label, max_dt_str


@st.cache_data(show_spinner=False)
def get_known_stores_directory() -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """
    Extracts unique NomeFiliera / Insegna names strictly from the official Coripet previous database
    for points of sale whose TipologiaPuntoFisico is 'PUNTOVENDITA'.
    Does NOT include COMUNE, PRIVATO, UFFICIOCOMUNALE, or any external files.
    Returns (sorted_options_list, store_details_dict).
    """
    store_map: Dict[str, Dict[str, str]] = {}

    try:
        df_c, _ = fetch_raw_coripet_machines()
        if not df_c.empty:
            if "TipologiaPuntoFisico" in df_c.columns:
                df_c = df_c[df_c["TipologiaPuntoFisico"].astype(str).str.upper().str.strip() == "PUNTOVENDITA"].copy()

            for _, r in df_c.iterrows():
                nome_fil = str(r.get("NomeFiliera", "")).strip()
                if not nome_fil or nome_fil.lower() == "nan":
                    nome_fil = str(r.get("Filiera", "")).strip()

                if nome_fil and nome_fil.lower() != "nan" and len(nome_fil) > 1:
                    addr = str(r.get("Indirizzo", "")).strip()
                    raw_cap = str(r.get("Cap", "")).strip().split(".")[0]
                    cap = raw_cap.zfill(5) if raw_cap and raw_cap.isdigit() else ""
                    com = str(r.get("Comune", "")).strip()

                    if nome_fil not in store_map:
                        store_map[nome_fil] = {
                            "store_name": nome_fil,
                            "address": addr if (addr and addr.lower() != "nan") else "",
                            "cap": cap if (cap and cap.lower() != "nan") else "",
                            "comune": com if (com and com.lower() != "nan") else "",
                        }
    except Exception:
        pass

    all_names = sorted(list(store_map.keys()))
    return all_names, store_map


def load_historical_machines():
    """Legacy helper for backward compatibility."""
    df_spl, _, _ = get_machines_by_status("SPL")
    df_pot, _, _ = get_machines_by_status("POT")
    return df_spl, df_pot



def batch_predict_machines(
    df_machines: pd.DataFrame,
    status_label: str,
    predictor: Any,
    force_refresh: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes end-to-end AI predictions for all machines in df_machines 1-by-1,
    with auto-detected Nielsen scale and default contract reward schemes.
    Results are cached in memory (session_state) and persisted on disk (data/cache/).
    Returns (enriched_and_ranked_df, results_map_by_pf).
    """
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"batch_predictions_{status_label.lower()}.pkl"

    session_key = f"batch_cache_{status_label.lower()}"
    results_map: Dict[str, Any] = {}

    if force_refresh:
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception:
                pass
        st.session_state.pop(session_key, None)

    # Check session state first
    if not force_refresh and session_key in st.session_state:
        results_map = st.session_state[session_key]
    # Check disk cache second
    elif not force_refresh and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                results_map = pickle.load(f)
            st.session_state[session_key] = results_map
        except Exception:
            results_map = {}

    # Determine which machines still need prediction
    missing_pf = [
        str(pf).strip() for pf in df_machines["PuntoFisico"].dropna().unique()
        if str(pf).strip() not in results_map or "PredictedBottlesDay" not in results_map[str(pf).strip()]
    ]

    if missing_pf and predictor is not None:
        n_missing = len(missing_pf)
        total_sites = len(df_machines["PuntoFisico"].dropna().unique())
        n_cached = total_sites - n_missing
        progress_box = st.container()
        with progress_box:
            if n_cached > 0:
                st.info(
                    f"⚡ **AI Model Incremental Sync:** Evaluating **{n_missing}** new {status_label} candidate(s) "
                    f"({n_cached} loaded instantly from cache &bull; **{total_sites} total {status_label} sites**)..."
                )
            else:
                st.info(f"⚡ **AI Model Batch Evaluation:** Generating 365-day collection predictions for all {total_sites} {status_label} machines 1-by-1...")
            p_bar = st.progress(0.0)
            status_txt = st.empty()

        for idx, pf in enumerate(missing_pf):
            sub_rows = df_machines[df_machines["PuntoFisico"].astype(str).str.strip() == pf]
            if sub_rows.empty:
                continue
            row = sub_rows.iloc[0]
            store_name = str(row["ResolvedStoreName"])
            comune_val = str(row["Comune"]) if pd.notna(row["Comune"]) else ""
            status_txt.markdown(
                f"🤖 **Evaluating machine {idx + 1} of {n_missing}:** `{store_name}` ({comune_val}) &bull; *Auto-querying Nielsen Census & Rewards...*"
            )

            addr = str(row["Indirizzo"])
            clean_addr = addr if addr and addr.lower() != "nan" else "Piazza del Comune 1"
            cap_val = str(row["Cap"]) if pd.notna(row["Cap"]) else "00100"
            mach_model = str(row["ResolvedMachine"])
            filiera_val = str(row["ResolvedFiliera"]) if pd.notna(row["ResolvedFiliera"]) else None

            try:
                res = predictor.predict_new_location(
                    store_name=store_name,
                    address=clean_addr,
                    cap=cap_val,
                    machine_type=mach_model,
                    comune=comune_val if comune_val else None,
                    filiera=filiera_val,
                    store_format=None,   # Auto-detects from Nielsen file!
                    sales_area_sqm=None, # Auto-detects from Nielsen file!
                    cash_registers=None, # Auto-detects from Nielsen file!
                    explicit_reward=None # Auto-resolves contract reward scheme!
                )
                b_day = float(res.get("realized_forecast", {}).get("underwritten_daily_bottles", res.get("realized_forecast", {}).get("realized_daily_bottles", 0.0)))
                p_day = float(res.get("realized_forecast", {}).get("underwritten_daily_presenze", 0.0))

                if b_day >= 2000.0:
                    tier_label = "SUPER PRENDENTE"
                    tier_color = "#8e24aa"
                    tier_badge_class = "badge-tier-super"
                elif b_day >= 1000.0:
                    tier_label = "ALTO PRENDENTE"
                    tier_color = "#43a047"
                    tier_badge_class = "badge-tier-alto"
                elif b_day >= 300.0:
                    tier_label = "MEDIO PRENDENTE"
                    tier_color = "#1e88e5"
                    tier_badge_class = "badge-tier-medio"
                else:
                    tier_label = "BASSO PRENDENTE"
                    tier_color = "#e53935"
                    tier_badge_class = "badge-tier-basso"

                n_fmt = str(res.get("inputs", {}).get("store_format", "SUP"))
                n_sqm = float(res.get("inputs", {}).get("sales_area_sqm", 0.0))
                n_casse = float(res.get("inputs", {}).get("cash_registers", 0.0))
                rew_type = str(res.get("commercial_rewards", {}).get("reward_type", "Standard Loyalty"))
                rew_mult = float(res.get("commercial_rewards", {}).get("multiplier_bottles", 1.0))
                near_km = float(res.get("spatial_catchment", {}).get("coripet_network", {}).get("nearest_coripet_km", 0.0))

                results_map[pf] = {
                    "PuntoFisico": pf,
                    "PredictedBottlesDay": round(b_day, 1),
                    "PredictedBottlesMonth": int(round(b_day * 30.4)),
                    "PredictedBottlesYear": int(round(b_day * 365.0)),
                    "PredictedVisitsDay": round(p_day, 1),
                    "VolumeTier": tier_label,
                    "VolumeTierColor": tier_color,
                    "VolumeTierBadge": tier_badge_class,
                    "NielsenFormat": n_fmt,
                    "NielsenSqm": n_sqm,
                    "NielsenCasse": n_casse,
                    "RewardType": rew_type,
                    "RewardMultiplier": rew_mult,
                    "NearestCoripetKm": near_km,
                    "FullResult": res,
                }
            except Exception as e:
                # Safe fallback record
                results_map[pf] = {
                    "PuntoFisico": pf,
                    "PredictedBottlesDay": 0.0,
                    "PredictedBottlesMonth": 0,
                    "PredictedBottlesYear": 0,
                    "PredictedVisitsDay": 0.0,
                    "VolumeTier": "BASSO PRENDENTE",
                    "VolumeTierColor": "#e53935",
                    "VolumeTierBadge": "badge-tier-basso",
                    "NielsenFormat": "SUP",
                    "NielsenSqm": 0.0,
                    "NielsenCasse": 0.0,
                    "RewardType": "Standard Loyalty",
                    "RewardMultiplier": 1.0,
                    "NearestCoripetKm": 0.0,
                    "FullResult": None,
                }

            p_bar.progress((idx + 1) / n_missing)
            # Extended presentation pace: smooth cadence so viewers can comfortably read store names and Nielsen lookups (~25s for SPL, ~60s for POT)
            step_delay = 0.35 if n_missing <= 60 else 0.14
            time.sleep(step_delay)

        status_txt.success(
            f"✅ Completed sync! All **{total_sites} {status_label} machines** are fully evaluated and ranked by daily yield."
        )
        time.sleep(1.2)
        progress_box.empty()

        # Save to session_state and disk cache
        st.session_state[session_key] = results_map
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(results_map, f)
        except Exception:
            pass

    # Enrich df_machines with predictions
    enriched_rows = []
    for _, row in df_machines.iterrows():
        pf = str(row["PuntoFisico"]).strip()
        p_data = results_map.get(pf, {})
        row_dict = row.to_dict()
        row_dict["PredictedBottlesDay"] = p_data.get("PredictedBottlesDay", 0.0)
        row_dict["PredictedBottlesMonth"] = p_data.get("PredictedBottlesMonth", 0)
        row_dict["PredictedBottlesYear"] = p_data.get("PredictedBottlesYear", 0)
        row_dict["PredictedVisitsDay"] = p_data.get("PredictedVisitsDay", 0.0)
        row_dict["VolumeTier"] = p_data.get("VolumeTier", "BASSO PRENDENTE")
        row_dict["VolumeTierColor"] = p_data.get("VolumeTierColor", "#e53935")
        row_dict["VolumeTierBadge"] = p_data.get("VolumeTierBadge", "badge-tier-basso")
        row_dict["NielsenFormat"] = p_data.get("NielsenFormat", "SUP")
        row_dict["NielsenSqm"] = p_data.get("NielsenSqm", 0.0)
        row_dict["NielsenCasse"] = p_data.get("NielsenCasse", 0.0)
        row_dict["RewardType"] = p_data.get("RewardType", "Standard Loyalty")
        row_dict["RewardMultiplier"] = p_data.get("RewardMultiplier", 1.0)
        row_dict["NearestCoripetKm"] = p_data.get("NearestCoripetKm", 0.0)
        enriched_rows.append(row_dict)

    df_enriched = pd.DataFrame(enriched_rows)

    # Rank according to bottles per day DESCENDING (highest at the top)
    df_ranked = df_enriched.sort_values(by="PredictedBottlesDay", ascending=False).reset_index(drop=True)
    df_ranked["Rank"] = df_ranked.index + 1

    return df_ranked, results_map


def render_machine_directory(
    df_machines: pd.DataFrame,
    status_label: str,
    predictor: Any,
    data_source: str = "Live Coripet API",
    current_date_str: str = ""
):
    """Renders a responsive directory table/cards of machines for SPL or POT status ranked by predicted bottles/day."""
    if df_machines.empty:
        st.info(f"No machines found with status {status_label}.")
        return

    # Check for force refresh request
    force_refresh = st.session_state.pop(f"force_refresh_{status_label}", False)

    # Execute batch AI prediction (1 by 1 if not cached) and rank highest at top
    df_ranked, results_map = batch_predict_machines(
        df_machines=df_machines,
        status_label=status_label,
        predictor=predictor,
        force_refresh=force_refresh
    )

    total_count = len(df_ranked)
    unique_comuni = df_ranked["Comune"].nunique()
    unique_regioni = df_ranked["Regione"].nunique()
    unique_filiere = df_ranked["ResolvedFiliera"].nunique()

    # Tier counts & Stats
    n_super = int((df_ranked["VolumeTier"] == "SUPER PRENDENTE").sum())
    n_alto = int((df_ranked["VolumeTier"] == "ALTO PRENDENTE").sum())
    n_medio = int((df_ranked["VolumeTier"] == "MEDIO PRENDENTE").sum())
    n_basso = int((df_ranked["VolumeTier"] == "BASSO PRENDENTE").sum())
    avg_bottles = float(df_ranked["PredictedBottlesDay"].mean()) if not df_ranked.empty else 0.0
    top_site = df_ranked.iloc[0] if not df_ranked.empty else None
    top_store_name = top_site["ResolvedStoreName"] if top_site is not None else "N/A"
    top_bottles_val = top_site["PredictedBottlesDay"] if top_site is not None else 0.0

    # Source badge styling (Adaptive)
    if data_source == "Live Coripet API":
        source_badge = '<span class="badge-source-live">🟢 Live API: macchine.coripet.it</span>'
    else:
        source_badge = '<span class="badge-source-backup">📁 Local Backup File</span>'

    date_label = f" • Current Date: <strong style='color: inherit;'>{current_date_str}</strong>" if current_date_str else ""

    if "dt_fine_sop" in df_ranked.columns:
        c_now = pd.Timestamp.now()
        n_5m = int((df_ranked["dt_fine_sop"].notna() & (df_ranked["dt_fine_sop"] >= (c_now - pd.DateOffset(months=5)))).sum())
        n_12m = int((df_ranked["dt_fine_sop"].notna() & (df_ranked["dt_fine_sop"] >= (c_now - pd.DateOffset(months=12)))).sum())
        n_24m = int((df_ranked["dt_fine_sop"].notna() & (df_ranked["dt_fine_sop"] >= (c_now - pd.DateOffset(months=24)))).sum())
        n_all = len(df_ranked)
    else:
        n_5m, n_12m, n_24m, n_all = 0, 0, 0, len(df_ranked)

    timeframe_options = [
        f"⏱️ Last 5 Months ({n_5m} sites)",
        f"📅 Last 12 Months ({n_12m} sites)",
        f"🗓️ Last 24 Months ({n_24m} sites)",
        f"🌐 All Sites ({n_all} sites)",
    ]

    tf_key = f"timeframe_filter_{status_label}"
    default_opt = timeframe_options[0] if n_5m > 0 else timeframe_options[-1]
    if tf_key in st.session_state:
        cur_val = str(st.session_state[tf_key])
        if cur_val not in timeframe_options:
            matched = next((opt for opt in timeframe_options if opt.split("(")[0].strip() == cur_val.split("(")[0].strip()), None)
            if matched:
                st.session_state[tf_key] = matched
            else:
                st.session_state[tf_key] = default_opt
    else:
        st.session_state[tf_key] = default_opt

    current_timeframe = st.session_state.get(tf_key, default_opt)

    # Metrics header banner (Adaptive Theme)
    st.markdown(
        f"""
        <div class="directory-banner">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <h3 style="margin: 0; color: inherit; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.3px;">
                            📋 {status_label} Machines Registry & AI Ranking <span style="opacity: 0.65; font-size: 1.05rem; font-weight: 500;">({total_count:,} Inspected Sites)</span>
                        </h3>
                        {source_badge}
                        <span class="badge-stat-blue" style="padding: 4px 10px; font-size: 0.78rem;">🏬 PUNTOVENDITA Only</span>
                        <span style="background: rgba(14, 165, 233, 0.15); color: #0284c7; border: 1px solid rgba(14, 165, 233, 0.35); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 800;">{current_timeframe}</span>
                        <span style="background: rgba(34, 197, 94, 0.15); color: #16a34a; border: 1px solid rgba(34, 197, 94, 0.35); padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 800;">🥇 Ranked by Highest b/d</span>
                    </div>
                    <p style="margin: 6px 0 0 0; color: inherit; opacity: 0.75; font-size: 0.85rem;">
                        Live retail store records matched with official Sopralluoghi site inspections and ranked by AI projected 365-day collection volume (highest yield at the top). Auto-inferred with Nielsen census scale and default contract reward schemes{date_label}.
                    </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                    <span class="badge-tier-super" style="font-size: 0.76rem; padding: 4px 10px;">🟣 Super: {n_super}</span>
                    <span class="badge-tier-alto" style="font-size: 0.76rem; padding: 4px 10px;">🟢 Alto: {n_alto}</span>
                    <span class="badge-tier-medio" style="font-size: 0.76rem; padding: 4px 10px;">🔵 Medio: {n_medio}</span>
                    <span class="badge-tier-basso" style="font-size: 0.76rem; padding: 4px 10px;">🔴 Basso: {n_basso}</span>
                    <span class="badge-stat-blue" style="font-size: 0.76rem; padding: 4px 10px;">📈 Network Avg: {avg_bottles:.1f} b/d</span>
                    <span class="badge-rank-1" style="font-size: 0.76rem; padding: 4px 10px;">🏆 Top: {top_store_name} ({top_bottles_val:.1f} b/d)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search & filters bar
    c_search, c_time, c_tier, c_reg, c_sort, c_ref = st.columns([1.8, 1.4, 1.2, 1.0, 1.2, 0.6])
    with c_search:
        search_query = st.text_input(
            f"🔍 Search {status_label} Machines",
            placeholder="Search by store name, ID (e.g. PF...), address, or city...",
            key=f"search_{status_label}",
        ).strip().lower()

    with c_time:
        def_idx = 0
        if tf_key in st.session_state and st.session_state[tf_key] in timeframe_options:
            def_idx = timeframe_options.index(st.session_state[tf_key])
        selected_timeframe = st.selectbox(
            "Survey Recency",
            options=timeframe_options,
            index=def_idx,
            key=tf_key,
            help="Filter machines by their site inspection (DataFineSopralluogo) completion date"
        )

    with c_tier:
        tier_filter_options = [
            "All Volume Tiers",
            "SUPER PRENDENTE (>2,000 b/d)",
            "ALTO PRENDENTE (1,000-2,000 b/d)",
            "MEDIO PRENDENTE (300-1,000 b/d)",
            "BASSO PRENDENTE (<300 b/d)",
        ]
        selected_tier = st.selectbox(
            "Volume Tier",
            options=tier_filter_options,
            index=0,
            key=f"tier_filter_{status_label}",
        )

    all_regions = ["All Regions"] + sorted([r for r in df_ranked["Regione"].unique() if r and r != "nan"])
    with c_reg:
        selected_region = st.selectbox(
            "Region Filter",
            options=all_regions,
            index=0,
            key=f"region_filter_{status_label}",
        )

    with c_sort:
        sort_options = [
            "🥇 Highest Volume (Default)",
            "🔻 Lowest Volume",
            "🔤 Store Name (A-Z)",
            "🏙️ Municipality (A-Z)",
        ]
        selected_sort = st.selectbox(
            "Sort Order",
            options=sort_options,
            index=0,
            key=f"sort_order_{status_label}",
        )

    with c_ref:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key=f"btn_refresh_{status_label}", help="Clear cache, re-fetch live data from Coripet API, and re-run all predictions"):
            fetch_raw_coripet_machines.clear()
            fetch_raw_coripet_sopralluoghi.clear()
            get_machines_by_status.clear()
            st.session_state[f"force_refresh_{status_label}"] = True
            st.toast(f"🔄 Re-fetching live {status_label} machines and re-running AI predictions...")
            st.rerun()

    # Apply filtering
    filtered_df = df_ranked.copy()

    # 1. Apply survey timeframe recency filter
    if "dt_fine_sop" in filtered_df.columns:
        if "5 Months" in selected_timeframe:
            cutoff_dt = pd.Timestamp.now() - pd.DateOffset(months=5)
            filtered_df = filtered_df[filtered_df["dt_fine_sop"].notna() & (filtered_df["dt_fine_sop"] >= cutoff_dt)]
        elif "12 Months" in selected_timeframe:
            cutoff_dt = pd.Timestamp.now() - pd.DateOffset(months=12)
            filtered_df = filtered_df[filtered_df["dt_fine_sop"].notna() & (filtered_df["dt_fine_sop"] >= cutoff_dt)]
        elif "24 Months" in selected_timeframe:
            cutoff_dt = pd.Timestamp.now() - pd.DateOffset(months=24)
            filtered_df = filtered_df[filtered_df["dt_fine_sop"].notna() & (filtered_df["dt_fine_sop"] >= cutoff_dt)]
        # For "🌐 All Sites", show all!

    if search_query:
        mask = (
            filtered_df["PuntoFisico"].astype(str).str.lower().str.contains(search_query, na=False)
            | filtered_df["ResolvedStoreName"].astype(str).str.lower().str.contains(search_query, na=False)
            | filtered_df["Indirizzo"].astype(str).str.lower().str.contains(search_query, na=False)
            | filtered_df["Comune"].astype(str).str.lower().str.contains(search_query, na=False)
            | filtered_df["Cap"].astype(str).str.lower().str.contains(search_query, na=False)
            | filtered_df["ResolvedFiliera"].astype(str).str.lower().str.contains(search_query, na=False)
        )
        filtered_df = filtered_df[mask]

    if selected_tier != "All Volume Tiers":
        tier_key = selected_tier.split(" ")[0].strip()
        filtered_df = filtered_df[filtered_df["VolumeTier"].str.contains(tier_key, na=False)]

    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["Regione"] == selected_region]

    # Apply custom sort
    if selected_sort == "🔻 Lowest Volume":
        filtered_df = filtered_df.sort_values(by="PredictedBottlesDay", ascending=True)
    elif selected_sort == "🔤 Store Name (A-Z)":
        filtered_df = filtered_df.sort_values(by="ResolvedStoreName", ascending=True)
    elif selected_sort == "🏙️ Municipality (A-Z)":
        filtered_df = filtered_df.sort_values(by="Comune", ascending=True)
    else:
        # Default: highest volume descending
        filtered_df = filtered_df.sort_values(by="PredictedBottlesDay", ascending=False)

    # Re-assign sequential ranks 1..N based on active filtered view
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df["Rank"] = filtered_df.index + 1

    n_filtered = len(filtered_df)
    if n_filtered == 0:
        st.info(f"ℹ️ No {status_label} machines matched the selected criteria (**{selected_timeframe}**). Try selecting another survey recency window or clearing search filters.")
        return

    # Pagination setup
    PER_PAGE = 20
    total_pages = max(1, (n_filtered + PER_PAGE - 1) // PER_PAGE)

    col_stat, col_page = st.columns([2, 1])
    with col_stat:
        st.markdown(
            f"""<div style="color: inherit; opacity: 0.75; font-size: 0.88rem; padding: 6px 0;">
                Showing <strong style="color: inherit; opacity: 1.0;">{n_filtered:,}</strong> of <strong style="color: inherit; opacity: 1.0;">{total_count:,}</strong> machines &bull; Ranked by <strong style="color: #22c55e;">Bottles per Day</strong> &bull; Click <strong style="color: #ef4444;">Results 📊</strong> to view full executive verdict
            </div>""",
            unsafe_allow_html=True
        )
    with col_page:
        page_num = st.selectbox(
            "Select Page",
            options=list(range(1, total_pages + 1)),
            index=0,
            key=f"page_sel_{status_label}",
            label_visibility="collapsed",
        )

    start_idx = (page_num - 1) * PER_PAGE
    end_idx = min(start_idx + PER_PAGE, n_filtered)
    page_df = filtered_df.iloc[start_idx:end_idx]

    # Table Header (Adaptive Theme)
    th_col1, th_col2, th_col3, th_col4, th_col5 = st.columns([3.2, 2.6, 2.0, 2.3, 1.2])
    with th_col1:
        st.markdown("<strong style='font-size: 0.78rem; letter-spacing: 0.6px; color: inherit; opacity: 0.68; font-weight: 700; text-transform: uppercase;'>RANK & STORE</strong>", unsafe_allow_html=True)
    with th_col2:
        st.markdown("<strong style='font-size: 0.78rem; letter-spacing: 0.6px; color: inherit; opacity: 0.68; font-weight: 700; text-transform: uppercase;'>LOCATION & MUNICIPALITY</strong>", unsafe_allow_html=True)
    with th_col3:
        st.markdown("<strong style='font-size: 0.78rem; letter-spacing: 0.6px; color: inherit; opacity: 0.68; font-weight: 700; text-transform: uppercase;'>HARDWARE & SCALE</strong>", unsafe_allow_html=True)
    with th_col4:
        st.markdown("<strong style='font-size: 0.78rem; letter-spacing: 0.6px; color: inherit; opacity: 0.68; font-weight: 700; text-transform: uppercase;'>AI PREDICTED YIELD</strong>", unsafe_allow_html=True)
    with th_col5:
        st.markdown("<strong style='font-size: 0.78rem; letter-spacing: 0.6px; color: inherit; opacity: 0.68; font-weight: 700; text-transform: uppercase;'>ACTION</strong>", unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 1px solid rgba(128, 128, 128, 0.20); margin-bottom: 14px;'></div>", unsafe_allow_html=True)

    # Machine Items List
    for row_idx, row in page_df.iterrows():
        pf_code = str(row["PuntoFisico"]).strip()
        store = row["ResolvedStoreName"]
        addr = row["Indirizzo"]
        cap_val = row["Cap"]
        comune_val = row["Comune"]
        prov_val = row["Provincia"]
        reg_val = row["Regione"]
        mach_model = row["ResolvedMachine"]
        filiera_val = row["ResolvedFiliera"]
        rank_val = int(row.get("Rank", row_idx + 1))
        b_day = float(row.get("PredictedBottlesDay", 0.0))
        m_day = int(row.get("PredictedBottlesMonth", 0))
        tier_label = str(row.get("VolumeTier", "MEDIO PRENDENTE"))
        tier_color = str(row.get("VolumeTierColor", "#1e88e5"))
        tier_badge = str(row.get("VolumeTierBadge", "badge-tier-medio"))
        n_sqm = float(row.get("NielsenSqm", 0.0))
        n_casse = float(row.get("NielsenCasse", 0.0))
        n_fmt = str(row.get("NielsenFormat", "SUP"))
        rew_type = str(row.get("RewardType", "Standard Loyalty"))
        REWARD_TYPE_EN_MAP = {
            "Emissione Buono sconto": "Direct Grocery Voucher",
            "Conversione a punti fedeltà": "Loyalty Points Conversion",
            "Omaggio": "Free Gift / Cash Reward",
        }
        rew_type_clean = REWARD_TYPE_EN_MAP.get(rew_type, rew_type)
        rew_mult = float(row.get("RewardMultiplier", 1.0))

        if rank_val == 1:
            rank_chip = '<span class="badge-rank-1">#1 🏆</span>'
        elif rank_val == 2:
            rank_chip = '<span class="badge-rank-2">#2 🥈</span>'
        elif rank_val == 3:
            rank_chip = '<span class="badge-rank-3">#3 🥉</span>'
        else:
            rank_chip = f'<span class="badge-rank-other">#{rank_val}</span>'

        st_val = str(row.get("Stato", status_label)).strip()
        if st_val == "SPL":
            status_chip = f'<span class="badge-pf-spl">SPL</span>'
        elif st_val == "POT":
            status_chip = f'<span class="badge-pf-pot">POT</span>'
        elif st_val and st_val != "nan":
            status_chip = f'<span class="badge-stat-blue" style="font-size: 0.72rem; padding: 2px 7px;">{st_val}</span>'
        else:
            status_chip = ""

        scale_str = f"📐 {n_sqm:,.0f} m² • {n_casse:.0f} casse ({n_fmt})" if n_sqm > 0 else f"📐 {n_fmt} Format"
        sop_date_str = str(row.get("DataFineSopralluogoFormatted", "")).strip()
        sop_status_str = str(row.get("StatoSopralluogoClean", "")).strip()

        # Color-coded sleek inspection status badge
        sop_status_lower = sop_status_str.lower()
        if "definitivo" in sop_status_lower or "completato" in sop_status_lower:
            sop_status_badge = f'<span style="background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); padding: 2px 7px; border-radius: 5px; font-size: 0.72rem; font-weight: 700;">✅ {sop_status_str}</span>'
        elif "in corso" in sop_status_lower:
            sop_status_badge = f'<span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 2px 7px; border-radius: 5px; font-size: 0.72rem; font-weight: 700;">⏳ {sop_status_str}</span>'
        elif "riaperto" in sop_status_lower:
            sop_status_badge = f'<span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); padding: 2px 7px; border-radius: 5px; font-size: 0.72rem; font-weight: 700;">🔄 {sop_status_str}</span>'
        elif sop_status_str and sop_status_str != "nan":
            sop_status_badge = f'<span style="background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.35); padding: 2px 7px; border-radius: 5px; font-size: 0.72rem; font-weight: 700;">📋 {sop_status_str}</span>'
        else:
            sop_status_badge = ""

        # Elegant single-line inspection date indicator (shows date or Null on all machines)
        if sop_date_str == "Null" or not sop_date_str or sop_date_str.lower() in ["nan", "none", "", "n/d", "data n/d"]:
            sop_date_line = (
                f'<div style="margin-top: 5px; font-size: 0.74rem; color: #94a3b8; display: flex; align-items: center; gap: 5px; white-space: nowrap;">'
                f'<span>🗓️ Inspection Date:</span>'
                f'<span style="font-weight: 600; color: #94a3b8; opacity: 0.75; background: rgba(255, 255, 255, 0.04); padding: 1px 6px; border-radius: 4px; border: 1px dashed rgba(255, 255, 255, 0.18);">Null</span>'
                f'</div>'
            )
        else:
            sop_date_line = (
                f'<div style="margin-top: 5px; font-size: 0.74rem; color: #94a3b8; display: flex; align-items: center; gap: 5px; white-space: nowrap;">'
                f'<span>🗓️ Inspection Date:</span>'
                f'<span style="font-weight: 600; color: #e2e8f0; background: rgba(255, 255, 255, 0.06); padding: 1px 6px; border-radius: 4px; border: 1px solid rgba(255, 255, 255, 0.1);">{sop_date_str}</span>'
                f'</div>'
            )

        with st.container():
            c1, c2, c3, c4, c5 = st.columns([3.4, 2.5, 1.9, 2.2, 1.2])
            with c1:
                st.markdown(
                    f"""
                    <div style="line-height: 1.35; margin-bottom: 6px;">
                        {rank_chip}
                        <span style="font-weight: 700; color: inherit; font-size: 0.98rem; letter-spacing: -0.2px; margin-left: 5px;">{store}</span>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                        <span class="badge-pf-id">🆔 {pf_code}</span>
                        <span class="badge-pf-filiera">🏢 {filiera_val}</span>
                        {status_chip}
                        {sop_status_badge}
                    </div>
                    {sop_date_line}
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                prov_str = f"({prov_val})" if prov_val else ""
                st.markdown(
                    f"""
                    <div style="font-size: 0.88rem; line-height: 1.4;">
                        <div style="color: inherit; font-weight: 600;">📍 {addr}</div>
                        <div style="color: inherit; opacity: 0.75; font-size: 0.80rem; margin-top: 2px;">📮 {cap_val} <strong style="color: inherit; opacity: 1.0;">{comune_val}</strong> {prov_str}, <span>{reg_val}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div style="font-size: 0.84rem; line-height: 1.4;">
                        <div style="font-weight: 700; color: inherit;">⚙️ {mach_model}</div>
                        <div style="color: inherit; opacity: 0.75; font-size: 0.77rem; margin-top: 3px;">{scale_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f"""
                    <div style="line-height: 1.3;">
                        <div style="display: flex; align-items: baseline; gap: 5px;">
                            <span style="font-size: 1.12rem; font-weight: 800; color: {tier_color}; letter-spacing: -0.3px;">{b_day:.1f} b/d</span>
                            <span style="font-size: 0.74rem; color: inherit; opacity: 0.70; font-weight: 500;">(~{m_day:,} b/mo)</span>
                        </div>
                        <div style="margin-top: 3px; display: flex; align-items: center; gap: 5px; flex-wrap: wrap;">
                            <span class="{tier_badge}">{tier_label}</span>
                        </div>
                        <div style="font-size: 0.70rem; color: inherit; opacity: 0.75; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                            🎟️ {rew_type_clean} ({rew_mult:.2f}x)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c5:
                btn_key = f"btn_res_{status_label}_{pf_code}_{row_idx}"
                if st.button("Results 📊", key=btn_key, type="primary", use_container_width=True, help=f"Open full prospective forecast dashboard for {store} ({pf_code})"):
                    clean_addr_val = addr if addr and addr.lower() != "nan" else "Piazza del Comune 1"
                    full_p_res = results_map.get(pf_code, {}).get("FullResult")
                    st.session_state["pending_load_machine"] = {
                        "store_name": store,
                        "address": clean_addr_val,
                        "cap": cap_val,
                        "comune": comune_val,
                        "machine_type": mach_model,
                        "pf_code": pf_code,
                        "status_label": status_label,
                        "pending_prediction_result": full_p_res,
                        "selected_format": n_fmt,
                        "selected_sqm": n_sqm,
                        "selected_casse": n_casse,
                        "selected_reward": rew_type,
                    }
                    st.session_state.pop("top_sim_machine", None)
                    st.session_state.pop("top_sim_reward", None)
                    st.session_state.pop("pending_input_machine_type", None)
                    st.session_state["pending_nav_tab"] = "🎯 Prospective Predictor"
                    st.rerun()

            st.markdown("<div style='border-bottom: 1px solid rgba(128, 128, 128, 0.15); margin: 8px 0 10px 0;'></div>", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">♻️ Coripet RVM Prospective Location Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Certified Machine Learning Decision Support System for Reverse Vending Machine Network Expansion</div>',
        unsafe_allow_html=True,
    )

    predictor, is_production, err_msg = load_production_predictor()

    if not is_production:
        st.warning(
            f"⚠️ **Scaffold Fallback Mode**: Certified production model could not be initialized ({err_msg}). "
            "Displaying baseline evaluation preview."
        )

    # -------------------------------------------------------------
    # Wizard Session State Initialization
    # -------------------------------------------------------------
    if "pending_load_machine" in st.session_state:
        load_data = st.session_state.pop("pending_load_machine")
        st.session_state["input_store_name"] = load_data["store_name"]
        st.session_state["input_address"] = load_data["address"]
        st.session_state["input_cap"] = load_data["cap"]
        st.session_state["input_comune"] = load_data["comune"]
        st.session_state["input_machine_type"] = load_data["machine_type"]
        st.session_state["pending_input_machine_type"] = load_data["machine_type"]
        st.session_state["selected_store_pf"] = load_data["pf_code"]
        st.session_state["selected_store_status"] = load_data["status_label"]
        if "pending_prediction_result" in load_data and load_data["pending_prediction_result"]:
            st.session_state["prediction_result"] = load_data["pending_prediction_result"]
            st.session_state["wizard_step"] = 4
            st.session_state["loc_info"] = load_data["pending_prediction_result"].get("location")
            st.session_state["auto_scale"] = {
                "superficie_vendita": load_data.get("selected_sqm", 850.0),
                "numero_casse": load_data.get("selected_casse", 5.0),
                "tipo_negozio": load_data.get("selected_format", "SUP"),
                "source": "Auto Nielsen Match"
            }
            st.session_state["auto_reward"] = load_data["pending_prediction_result"].get("commercial_rewards")
            st.session_state["selected_format"] = load_data.get("selected_format")
            st.session_state["selected_sqm"] = load_data.get("selected_sqm")
            st.session_state["selected_casse"] = load_data.get("selected_casse")
            st.session_state["selected_reward"] = load_data.get("selected_reward")
            st.toast(f"📊 Loaded active prediction for {load_data['store_name']} ({load_data['pf_code']})!", icon="🚀")
        else:
            st.session_state["wizard_step"] = 1
            st.session_state["loc_info"] = None
            st.session_state["auto_scale"] = None
            st.session_state["auto_reward"] = None
            st.session_state["prediction_result"] = None
            st.toast(f"📥 Populated {load_data['store_name']} ({load_data['pf_code']}) into Site Setup! Click Step 1 in the sidebar to proceed.", icon="📍")
        st.session_state["step1_validation_error"] = False
        st.session_state["step1_missing_fields"] = []

    if "wizard_step" not in st.session_state:
        st.session_state["wizard_step"] = 1
    if "loc_info" not in st.session_state:
        st.session_state["loc_info"] = None
    if "auto_scale" not in st.session_state:
        st.session_state["auto_scale"] = None
    if "auto_reward" not in st.session_state:
        st.session_state["auto_reward"] = None
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None
    if "selected_format" not in st.session_state:
        st.session_state["selected_format"] = None
    if "selected_sqm" not in st.session_state:
        st.session_state["selected_sqm"] = None
    if "selected_casse" not in st.session_state:
        st.session_state["selected_casse"] = None
    if "selected_reward" not in st.session_state:
        st.session_state["selected_reward"] = None
    if "input_store_name" not in st.session_state:
        st.session_state["input_store_name"] = ""
    if "custom_store_input" not in st.session_state:
        st.session_state["custom_store_input"] = ""
    if "input_address" not in st.session_state:
        st.session_state["input_address"] = ""
    if "input_cap" not in st.session_state:
        st.session_state["input_cap"] = ""
    if "input_comune" not in st.session_state:
        st.session_state["input_comune"] = ""
    if "input_machine_type" not in st.session_state:
        st.session_state["input_machine_type"] = "DAINT_G"
    if "step1_validation_error" not in st.session_state:
        st.session_state["step1_validation_error"] = False
    if "step1_missing_fields" not in st.session_state:
        st.session_state["step1_missing_fields"] = []

    # -------------------------------------------------------------
    # Sidebar: 3-Step Guided Prospective Analysis Wizard
    # -------------------------------------------------------------
    with st.sidebar:
        col_hdr, col_rst = st.columns([2.2, 1.0])
        with col_hdr:
            st.header("📍 Site Setup")
        with col_rst:
            if st.button("🔄 Reset", help="Clear all steps and empty all fields"):
                st.session_state["wizard_step"] = 1
                st.session_state["loc_info"] = None
                st.session_state["auto_scale"] = None
                st.session_state["auto_reward"] = None
                st.session_state["prediction_result"] = None
                st.session_state["selected_format"] = None
                st.session_state["selected_sqm"] = None
                st.session_state["selected_casse"] = None
                st.session_state["selected_reward"] = None
                st.session_state["input_store_name"] = ""
                st.session_state["is_custom_store_mode"] = False
                st.session_state["custom_store_input"] = ""
                st.session_state["store_dropdown_val"] = "-- Select Store Name / Insegna --"
                st.session_state["store_dropdown_select"] = "-- Select Store Name / Insegna --"
                st.session_state["input_address"] = ""
                st.session_state["input_cap"] = ""
                st.session_state["input_comune"] = ""
                st.session_state["input_machine_type"] = "DAINT_G"
                st.session_state["step1_validation_error"] = False
                st.session_state["step1_missing_fields"] = []
                st.session_state.pop("top_sim_machine", None)
                st.session_state.pop("top_sim_reward", None)
                st.session_state.pop("pending_input_machine_type", None)
                st.session_state.pop("selected_store_pf", None)
                st.session_state.pop("selected_store_status", None)
                st.session_state["pending_nav_tab"] = "🎯 Prospective Predictor"
                st.rerun()

        st.caption("Guided 3-step decision support wizard with Nielsen & Coripet intelligence.")

        # --- STEP 1: Basic Site & Machine Inputs ---
        st.markdown("#### 1️⃣ Location & Machine Model")

        has_val_err = st.session_state.get("step1_validation_error", False)
        raw_s = str(st.session_state.get("input_store_name", "")).strip()
        raw_a = str(st.session_state.get("input_address", "")).strip()
        raw_c = str(st.session_state.get("input_cap", "")).strip()
        raw_m = str(st.session_state.get("input_comune", "")).strip()

        s_err = has_val_err and not raw_s
        a_err = has_val_err and not raw_a
        c_err = has_val_err and (not raw_c or not raw_c.isdigit())
        m_err = has_val_err and not raw_m

        # Auto-clear error state once user fills all fields
        if has_val_err and not s_err and not a_err and not c_err and not m_err:
            st.session_state["step1_validation_error"] = False
            st.session_state["step1_missing_fields"] = []
            has_val_err = False

        # Store Name Strategy: Search Previous Records or Custom Option
        known_stores_list, known_stores_details = get_known_stores_directory()
        dropdown_options = known_stores_list

        cur_store_val = str(st.session_state.get("input_store_name", "")).strip()
        is_custom_mode = st.session_state.get("is_custom_store_mode", False)

        # Clear legacy prompt from session state if present
        if st.session_state.get("store_dropdown_select") not in dropdown_options:
            st.session_state["store_dropdown_select"] = None

        # Determine index in known stores
        default_idx = None
        if not is_custom_mode and cur_store_val and cur_store_val in dropdown_options:
            default_idx = dropdown_options.index(cur_store_val)
        elif "store_dropdown_select" in st.session_state and st.session_state["store_dropdown_select"] in dropdown_options:
            default_idx = dropdown_options.index(st.session_state["store_dropdown_select"])

        def on_store_dropdown_change():
            sel = st.session_state.get("store_dropdown_select")
            if not sel:
                st.session_state["input_store_name"] = ""
            else:
                # Store selection is only to verify/select if filiera exists;
                # address, cap, and comune remain completely manual
                st.session_state["input_store_name"] = sel

        col_store_lbl, col_store_btn = st.columns([1.3, 1.7], vertical_alignment="bottom")
        with col_store_lbl:
            st.markdown("<p style='margin:0 0 6px 0; font-weight:600; font-size:0.85rem;'>Store Name / Insegna</p>", unsafe_allow_html=True)
        with col_store_btn:
            if not is_custom_mode:
                if st.button("➕ Custom", key="btn_toggle_custom", help="Enable custom store name input and block dropdown list", use_container_width=True):
                    st.session_state["is_custom_store_mode"] = True
                    st.rerun()
            else:
                if st.button("📋 From List", key="btn_toggle_list", help="Use Coripet database list and block custom input", use_container_width=True):
                    st.session_state["is_custom_store_mode"] = False
                    st.rerun()

        # Input 1: Dropdown from official Coripet database (BLOCKED when custom input is enabled; EMPTY by default)
        chosen_dropdown = st.selectbox(
            "Store Name / Insegna (Coripet List)",
            options=dropdown_options,
            index=default_idx,
            placeholder="Select Store Name / Insegna...",
            key="store_dropdown_select",
            on_change=on_store_dropdown_change,
            disabled=is_custom_mode,
            help="Select an existing Insegna strictly from Coripet database. Blocked when Custom is enabled.",
            label_visibility="collapsed",
        )

        # Input 2: Custom store name input (Only displayed when custom button is clicked)
        if is_custom_mode:
            custom_val = st.text_input(
                "Custom Store Name / Insegna",
                value=st.session_state.get("custom_store_input", st.session_state.get("input_store_name", "")),
                key="custom_store_input",
                placeholder="Type custom store name (e.g., Esselunga, Bio Market)...",
                disabled=False,
                help="Type custom store name to give to the model for prediction.",
                label_visibility="collapsed",
            )
            store_name = custom_val.strip()
            st.session_state["input_store_name"] = store_name
            if store_name:
                st.caption(f"✏️ *Custom store name for prediction:* **{store_name}** (Dropdown list is blocked)")
            else:
                st.caption("🔒 *Dropdown list is blocked. Type custom store name above.*")
        else:
            if not chosen_dropdown:
                store_name = ""
                st.session_state["input_store_name"] = ""
            else:
                store_name = chosen_dropdown
                st.session_state["input_store_name"] = chosen_dropdown
                st.caption(f"📍 *Insegna selected:* **{store_name}**")

        if s_err:
            st.error("⚠️ Store Name / Insegna is required")

        address = st.text_input(
            "Street Address",
            key="input_address",
            placeholder="e.g., Via delle Forze Armate 12",
            help="Street and number of prospective site",
        )
        if a_err:
            st.error("⚠️ Street Address is required")

        cap = st.text_input(
            "Postal Code (CAP)",
            key="input_cap",
            placeholder="e.g., 20124, 7100, 00185",
            help="Italian CAP postal code (e.g., 20124, 7100, 00185)",
        )
        if c_err:
            if not raw_c:
                st.error("⚠️ Postal Code (CAP) is required")
            else:
                st.error("⚠️ Postal Code (CAP) must contain numeric digits (e.g. 20124, 7100)")

        comune = st.text_input(
            "Municipality (Comune)",
            key="input_comune",
            placeholder="e.g., Milano, Busto Arsizio, Roma, Torino",
            help="Italian municipality name (e.g., Milano, Busto Arsizio).",
        )
        if m_err:
            st.error("⚠️ Municipality (Comune) is required")

        # Machine is removed as an input; all 4 certified models are evaluated in the results split matrix
        machine_type = "DAINT_G"

        btn_detect = st.button("🔍 Step 1: Detect Location & Nielsen Scale", type="primary", use_container_width=True)
        if btn_detect:
            clean_s = str(store_name).strip()
            clean_a = str(address).strip()
            clean_c = str(cap).strip()
            clean_m = str(comune).strip()

            missing = []
            if not clean_s:
                missing.append("Store Name")
            if not clean_a:
                missing.append("Street Address")
            if not clean_c:
                missing.append("Postal Code (CAP)")
            elif not clean_c.isdigit():
                missing.append("Postal Code (must be numeric)")
            if not clean_m:
                missing.append("Municipality (Comune)")

            if missing:
                st.session_state["step1_validation_error"] = True
                st.session_state["step1_missing_fields"] = missing
                st.rerun()
            else:
                st.session_state["step1_validation_error"] = False
                st.session_state["step1_missing_fields"] = []
                try:
                    with st.spinner("Resolving location & detecting store scale from Nielsen registry..."):
                        loc_info = predictor.resolve_location(clean_a, clean_c, comune=clean_m)
                        auto_scale = predictor.scale_estimator.estimate_scale(clean_s, clean_a, clean_c, comune=clean_m)
                        st.session_state["loc_info"] = loc_info
                        st.session_state["auto_scale"] = auto_scale

                        # Auto-detect if location matches an existing active machine in Coripet fleet
                        try:
                            from src.ui.historical_vs_prediction import load_all_machine_options
                            all_m = load_all_machine_options()
                            matched_inst = None
                            for m in all_m:
                                m_com = str(m.get("comune", "")).strip().lower()
                                m_ban = str(m.get("banner", "")).strip().lower()
                                if m_com and clean_m.strip().lower() == m_com:
                                    if clean_s.strip().lower() in m_ban or m_ban in clean_s.strip().lower():
                                        matched_inst = m
                                        break
                            if matched_inst:
                                st.session_state["selected_store_pf"] = matched_inst.get("pf") or matched_inst["macchina"]
                                st.session_state["selected_store_status"] = "ATTVO"
                            else:
                                st.session_state.pop("selected_store_pf", None)
                                st.session_state.pop("selected_store_status", None)
                        except Exception:
                            pass

                        st.session_state["wizard_step"] = 2
                        st.session_state["prediction_result"] = None
                        st.session_state["pending_nav_tab"] = "🎯 Prospective Predictor"
                        st.rerun()
                except ValueError as e:
                    st.error(f"❌ {e}")

        # --- STEP 2: Location & Nielsen Scale Confirmation ---
        if st.session_state.get("wizard_step", 1) >= 2 and st.session_state.get("loc_info") and st.session_state.get("auto_scale"):
            st.divider()
            st.markdown("#### 2️⃣ Store Scale & Nielsen Census")
            
            loc = st.session_state["loc_info"]
            scale = st.session_state["auto_scale"]
            
            st.success(f"📍 **Location Verified:** {loc.get('Comune')} ({loc.get('Provincia')}), {loc.get('Regione')}")

            if st.session_state.get("selected_store_status") == "ATTVO" and st.session_state.get("selected_store_pf"):
                cur_s = st.session_state.get("input_store_name", "")
                st.info(f"✨ **Active Fleet RVM Detected:** `{st.session_state.get('selected_store_pf')}` ({cur_s}). Real historical telemetry will be compared directly against the model forecast.")

            is_exact = bool(scale.get("is_exact_match", False) or scale.get("scale_source") == "EXACT_NIELSEN_STORE_MATCH")
            
            if is_exact:
                scale_desc = (
                    f"• **Store in Census:** `{scale.get('insegna_raw', store_name)}`\n\n"
                    f"• **Sales Area:** ~{scale.get('superficie_vendita', 0):.0f} m² | **Checkouts:** {scale.get('numero_casse', 0):.0f} ({scale.get('tipo_negozio')})\n\n"
                    f"• *{scale.get('details', '')}*"
                )
                st.success(f"🎯 **Verified Exact Nielsen Census Match Found!**\n\n{scale_desc}")
                st.markdown("💬 **Exact census record confirmed. You can accept this verified scale or modify it below:**")
                
                auto_label = f"Auto-Detect: Exact Census Match (~{scale.get('superficie_vendita', 0):.0f} m², {scale.get('numero_casse', 0):.0f} checkouts) [Verified]"
                format_options = [
                    auto_label,
                    "Custom (manual sales area and checkouts)",
                    "1: Small Supermarket / Proximity (~350 m², 2 checkouts)",
                    "2: Standard Supermarket (~750 m², 4 checkouts)",
                    "3: Superstore / Large Supermarket (~1,850 m², 9 checkouts)",
                    "4: Hypermarket / Mall Anchor (~3,750 m², 21 checkouts)",
                    "5: Discount Store (~750 m², 4 checkouts)",
                ]
            else:
                st.warning(
                    f"⚠️ **No Exact Nielsen Census Match Found for '{store_name}' on '{address}'**\n\n"
                    f"This specific store was not found in the 25,104-store Nielsen registry. "
                    f"**Please enter the actual store scale below (Custom)** or select a retail archetype to ensure accurate predictions:"
                )
                format_options = [
                    "Custom (manual sales area and checkouts)",
                    "1: Small Supermarket / Proximity (~350 m², 2 checkouts)",
                    "2: Standard Supermarket (~750 m², 4 checkouts)",
                    "3: Superstore / Large Supermarket (~1,850 m², 9 checkouts)",
                    "4: Hypermarket / Mall Anchor (~3,750 m², 21 checkouts)",
                    "5: Discount Store (~750 m², 4 checkouts)",
                ]

            format_choice = st.selectbox(
                "Store Format / Scale Profile",
                options=format_options,
                index=0,
                help="Accept verified footprint, enter custom store scale, or select archetype",
            )
            
            chosen_format = scale.get("tipo_negozio")
            chosen_sqm = float(scale.get("superficie_vendita", 850.0))
            chosen_casse = float(scale.get("numero_casse", 5.0))
            
            if "1:" in format_choice:
                chosen_format = "1"
                chosen_sqm, chosen_casse = 350.0, 2.0
            elif "2:" in format_choice:
                chosen_format = "2"
                chosen_sqm, chosen_casse = 750.0, 4.0
            elif "3:" in format_choice:
                chosen_format = "3"
                chosen_sqm, chosen_casse = 1850.0, 9.0
            elif "4:" in format_choice:
                chosen_format = "4"
                chosen_sqm, chosen_casse = 3750.0, 21.0
            elif "5:" in format_choice:
                chosen_format = "5"
                chosen_sqm, chosen_casse = 750.0, 4.0
            elif "Custom" in format_choice:
                chosen_format = "custom"
                c_area, c_casse = st.columns(2)
                cur_sqm = float(st.session_state.get("selected_sqm") or scale.get("superficie_vendita", 750.0))
                cur_casse = float(st.session_state.get("selected_casse") or scale.get("numero_casse", 4.0))
                with c_area:
                    chosen_sqm = st.number_input("Sales Area (m²)", min_value=50.0, max_value=25000.0, value=cur_sqm, step=50.0, help="Gross retail floor area in square meters")
                with c_casse:
                    chosen_casse = st.number_input("Checkouts", min_value=1.0, max_value=100.0, value=cur_casse, step=1.0, help="Total active cash registers / checkout lanes")
                    
            btn_step2 = st.button("➡️ Step 2: Confirm Scale & Fetch Rewards", type="primary", use_container_width=True)
            if btn_step2:
                with st.spinner("Fetching commercial rewards and promotional contracts..."):
                    rew_info = predictor.rewards_engine.resolve_reward(store_name=store_name, comune=comune.strip() if comune else None)
                    st.session_state["auto_reward"] = rew_info
                    st.session_state["selected_format"] = chosen_format
                    st.session_state["selected_sqm"] = chosen_sqm
                    st.session_state["selected_casse"] = chosen_casse
                    st.session_state["wizard_step"] = 3
                    st.session_state["prediction_result"] = None
                    st.rerun()

        # --- STEP 3: Commercial Rewards Confirmation ---
        if st.session_state.get("wizard_step", 1) >= 3 and st.session_state.get("auto_reward"):
            st.divider()
            st.markdown("#### 3️⃣ Promotional Rewards Policy")
            
            rew = st.session_state["auto_reward"]
            rew_scheme = rew.get("reward_type", "Emissione Buono sconto")
            fil_name = rew.get("filiera", "National Standard")
            contract_rule = rew.get("contract_rule", rew.get("typical_voucher", ""))
            
            st.info(
                f"🎁 **Commercial Rewards Auto-Detected:**\n\n"
                f"• **Circuit / Filiera:** {fil_name}\n\n"
                f"• **Scheme:** {rew_scheme}\n\n"
                f"• **Contract Promo:** {contract_rule}\n\n"
                f"• **Incentive Tier:** {rew.get('incentive_tier', 'GROCERY_VOUCHER')}"
            )
            
            st.markdown("💬 **This is what I found in the Coripet contract data. If you want to change it, change it below:**")
            
            auto_rew_label = f"Keep Auto-Detected: {rew_scheme} ({fil_name}) [Recommended]"
            reward_options = [
                auto_rew_label,
                "Conversione a punti fedeltà (Store Loyalty Card)",
                "Emissione Buono sconto (Direct Grocery Voucher)",
                "Omaggio (Direct Free Gift / No Min. Spend)",
                "🛠️ Custom Reward Policy (Define € discount, min spend, points)",
            ]
            reward_choice = st.selectbox(
                "Commercial Incentive Scheme",
                options=reward_options,
                index=0,
                help="Keep auto-detected filiera contract promotion or test alternate scheme",
            )
            
            chosen_reward_param = None
            if "Custom" in reward_choice:
                st.caption("Configure custom incentive parameters:")
                c_disc, c_spend, c_pts = st.columns(3)
                with c_disc:
                    c_disc_val = st.number_input("Discount (€)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
                with c_spend:
                    c_spend_val = st.number_input("Min. Spend (€)", min_value=0.0, max_value=200.0, value=25.0, step=5.0)
                with c_pts:
                    c_pts_val = st.number_input("Points / Bottles", min_value=10, max_value=1000, value=200, step=25)
                
                c_arch = st.selectbox(
                    "Scheme Archetype",
                    options=[
                        "Emissione Buono sconto (Grocery Checkout Voucher)",
                        "Conversione a punti fedeltà (Store Loyalty Card)",
                        "Omaggio (Free Gift / Direct Voucher)"
                    ],
                    index=0
                )
                arch_cat = c_arch.split(" (")[0]
                from src.business.rewards import create_custom_reward_policy
                chosen_reward_param = create_custom_reward_policy(
                    discount_euro=float(c_disc_val),
                    min_spend_euro=float(c_spend_val),
                    points_required=int(c_pts_val),
                    base_category=arch_cat
                )
            elif "Keep Auto-Detected" not in reward_choice:
                chosen_reward_param = reward_choice.split(" (")[0]
                
            with st.expander("🗓️ Planned Go-Live Date & Lifecycle Horizon", expanded=False):
                col_m, col_y = st.columns(2)
                m_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                cur_m_idx = max(0, min(11, pd.Timestamp.now().month - 1))
                with col_m:
                    chosen_m_name = st.selectbox("Go-Live Month", options=m_names, index=cur_m_idx)
                    chosen_go_live_month = m_names.index(chosen_m_name) + 1
                with col_y:
                    chosen_go_live_year = st.selectbox("Go-Live Year", options=[2026, 2027, 2028], index=0)
                chosen_rampup = st.checkbox("Include 3-Month Adoption Ramp-Up", value=True, help="Models realistic Month 1-3 customer discovery and Coripet app onboarding.")

            btn_predict = st.button("🚀 Step 3: Run Full Prediction Analysis", type="primary", use_container_width=True)
            if btn_predict:
                try:
                    with st.spinner("Executing end-to-end multi-layer evaluation pipeline..."):
                        import inspect
                        predict_kwargs = {
                            "store_name": store_name,
                            "address": address,
                            "cap": cap,
                            "machine_type": machine_type,
                            "comune": comune.strip() if comune else None,
                            "store_format": st.session_state.get("selected_format"),
                            "sales_area_sqm": st.session_state.get("selected_sqm"),
                            "cash_registers": st.session_state.get("selected_casse"),
                            "explicit_reward": chosen_reward_param,
                        }
                        if hasattr(predictor, "predict_new_location"):
                            sig_params = inspect.signature(predictor.predict_new_location).parameters
                            if "go_live_month" in sig_params:
                                predict_kwargs["go_live_month"] = chosen_go_live_month if 'chosen_go_live_month' in locals() else None
                            if "go_live_year" in sig_params:
                                predict_kwargs["go_live_year"] = chosen_go_live_year if 'chosen_go_live_year' in locals() else None
                            if "enable_rampup" in sig_params:
                                predict_kwargs["enable_rampup"] = chosen_rampup if 'chosen_rampup' in locals() else True

                        res = predictor.predict_new_location(**predict_kwargs)

                        # Auto-recommend & auto-select the best hardware fit for brand new random locations:
                        if "selected_store_pf" not in st.session_state:
                            b_est = float(res.get("realized_forecast", {}).get("underwritten_daily_bottles", 0.0))
                            if b_est <= 260.0:
                                opt_mach = "DAINT_A-bis"
                            elif b_est <= 450.0:
                                opt_mach = "DAINT_B-bis"
                            elif b_est <= 900.0:
                                opt_mach = "DAINT_G"
                            else:
                                opt_mach = "Ripet_C-front"

                            if predict_kwargs.get("machine_type") != opt_mach:
                                predict_kwargs["machine_type"] = opt_mach
                                res = predictor.predict_new_location(**predict_kwargs)

                        st.session_state["prediction_result"] = res
                        st.session_state["wizard_step"] = 4
                        st.session_state.pop("top_sim_machine", None)
                        st.session_state.pop("top_sim_reward", None)
                        st.session_state.pop("pending_input_machine_type", None)
                        st.session_state["pending_nav_tab"] = "🎯 Prospective Predictor"
                        st.rerun()
                except ValueError as err:
                    st.error(f"❌ **Input Validation / Location Error:** {err}")
                except Exception as err:
                    st.error(f"❌ **Pipeline Execution Error:** {err}")

    # -------------------------------------------------------------
    # Main Dashboard Tabs Execution: Predictor, SPL, POT
    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # Navigation Views: Prospective Predictor, SPL, POT
    # -------------------------------------------------------------
    TAB_MAIN = "🎯 Prospective Predictor"
    TAB_SPL = "📋 SPL Machines"
    TAB_POT = "💡 POT Machines"
    TAB_OUTLIERS = "🔍 Outlier & Defect Simulator"
    NAV_TABS = [TAB_MAIN, TAB_SPL, TAB_POT, TAB_OUTLIERS]

    if "pending_nav_tab" in st.session_state:
        st.session_state["active_nav_tab"] = st.session_state.pop("pending_nav_tab")

    if "active_nav_tab" not in st.session_state or st.session_state["active_nav_tab"] not in NAV_TABS:
        st.session_state["active_nav_tab"] = TAB_MAIN

    active_tab = st.segmented_control(
        "Navigation Views",
        options=NAV_TABS,
        key="active_nav_tab",
        label_visibility="collapsed",
        selection_mode="single",
    )
    if not active_tab:
        active_tab = TAB_MAIN
        st.session_state["active_nav_tab"] = TAB_MAIN

    if active_tab == TAB_MAIN:
        if "selected_store_pf" in st.session_state and st.session_state.get("prediction_result"):
            sel_pf = st.session_state.get("selected_store_pf")
            sel_st = st.session_state.get("selected_store_status", "")
            sel_nm = st.session_state.get("input_store_name", "")
            st.success(f"📍 **Active Prediction Loaded from {sel_st} Registry:** **{sel_nm}** (Machine ID: `{sel_pf}` | Status: **{sel_st}**)")

        res = st.session_state.get("prediction_result")
        if not res:
            # Guided Onboarding Walkthrough Card
            curr_step = st.session_state.get("wizard_step", 1)
            st.markdown(
                f"""
                <div class="intel-section" style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 5px solid #22c55e; padding: 24px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);">
                    <h3 style="color: #4ade80; margin-top: 0; font-size: 1.3rem; font-weight: 700;">🧭 Guided Prospective Location Analysis</h3>
                    <p style="color: #cbd5e1; font-size: 1.0rem; line-height: 1.55;">
                        Welcome to the <strong>Coripet RVM Prospective Predictor</strong>. Follow the 3-step decision support wizard on the left sidebar, or select any installed machine from the <strong style="color: #60a5fa;">SPL Machines</strong> or <strong style="color: #c084fc;">POT Machines</strong> tabs to run instant predictions:
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 18px;">
                        <div style="background: {'rgba(34, 197, 94, 0.15)' if curr_step == 1 else 'rgba(15, 23, 42, 0.6)'}; border: 1px solid {'rgba(34, 197, 94, 0.45)' if curr_step == 1 else 'rgba(255, 255, 255, 0.08)'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: {'#4ade80' if curr_step == 1 else '#f1f5f9'};">1️⃣ Step 1: Site & Model</strong><br>
                            <span style="font-size: 0.85rem; color: #94a3b8;">Enter store name, street, CAP, and machine. Click <em>Detect Location & Nielsen Scale</em>.</span>
                        </div>
                        <div style="background: {'rgba(34, 197, 94, 0.15)' if curr_step == 2 else 'rgba(15, 23, 42, 0.6)'}; border: 1px solid {'rgba(34, 197, 94, 0.45)' if curr_step == 2 else 'rgba(255, 255, 255, 0.08)'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: {'#4ade80' if curr_step == 2 else '#f1f5f9'};">2️⃣ Step 2: Nielsen Census</strong><br>
                            <span style="font-size: 0.85rem; color: #94a3b8;">Review auto-detected store scale and footprint. Accept or adjust, then click <em>Continue</em>.</span>
                        </div>
                        <div style="background: {'rgba(34, 197, 94, 0.15)' if curr_step == 3 else 'rgba(15, 23, 42, 0.6)'}; border: 1px solid {'rgba(34, 197, 94, 0.45)' if curr_step == 3 else 'rgba(255, 255, 255, 0.08)'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: {'#4ade80' if curr_step == 3 else '#f1f5f9'};">3️⃣ Step 3: Coripet Rewards</strong><br>
                            <span style="font-size: 0.85rem; color: #94a3b8;">Review contract promotions from the database. Keep or change, then click <em>Run Prediction</em>.</span>
                        </div>
                        <div style="background: {'rgba(34, 197, 94, 0.15)' if curr_step == 4 else 'rgba(15, 23, 42, 0.6)'}; border: 1px solid {'rgba(34, 197, 94, 0.45)' if curr_step == 4 else 'rgba(255, 255, 255, 0.08)'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: {'#4ade80' if curr_step == 4 else '#f1f5f9'};">4️⃣ Step 4: Executive Verdict</strong><br>
                            <span style="font-size: 0.85rem; color: #94a3b8;">View 365-day yield, 12-month calendar, clearance gates, and investment score.</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_prediction_dashboard(
                res=res,
                predictor=predictor,
                store_name=store_name,
                address=address,
                cap=cap,
                comune=comune,
                machine_type=machine_type,
            )

    elif active_tab == TAB_SPL:
        with st.spinner("📡 Fetching live SPL machines from Coripet API..."):
            df_spl, src_spl, dt_spl = get_machines_by_status("SPL", current_only=True)
        render_machine_directory(df_spl, "SPL", predictor, data_source=src_spl, current_date_str=dt_spl)

    elif active_tab == TAB_POT:
        with st.spinner("📡 Fetching live POT machines from Coripet API..."):
            df_pot, src_pot, dt_pot = get_machines_by_status("POT", current_only=True)
        render_machine_directory(df_pot, "POT", predictor, data_source=src_pot, current_date_str=dt_pot)

    elif active_tab == TAB_OUTLIERS:
        from src.ui.outlier_dashboard import render_outlier_and_defect_dashboard
        render_outlier_and_defect_dashboard()


def render_prediction_dashboard(
    res: Dict[str, Any],
    predictor: Any,
    store_name: str,
    address: str,
    cap: str,
    comune: Optional[str],
    machine_type: str,
):

    # -------------------------------------------------------------
    # Interactive Machine Model & Commercial Reward Dropdown Filters
    # -------------------------------------------------------------
    CERTIFIED_MACHINES = [
        {"id": "DAINT_A-bis", "name": "Daint A-bis", "cap": 300, "class": "Compact (Single-Stream)", "icon": "📦", "servicing": "Daily"},
        {"id": "DAINT_B-bis", "name": "Daint B-Bis", "cap": 500, "class": "Compact (Dual-Stream)", "icon": "🗄️", "servicing": "Twice/Wk"},
        {"id": "DAINT_G", "name": "Daint G", "cap": 4500, "class": "Heavy-Duty (Dual-Stream)", "icon": "🏭", "servicing": "Bi-Weekly"},
        {"id": "Ripet_C-front", "name": "Ripet C-Front", "cap": 4500, "class": "Industrial Compactor", "icon": "⚙️", "servicing": "Bi-Weekly"},
    ]
    machine_options = [m["id"] for m in CERTIFIED_MACHINES]

    res_mach = res.get("inputs", {}).get("machine_type", "DAINT_G")
    # Normalize res_mach into one of the 4 certified IDs
    if "A-BIS" in res_mach.upper():
        res_mach = "DAINT_A-bis"
    elif "B-BIS" in res_mach.upper() or "DAINT_B" in res_mach.upper():
        res_mach = "DAINT_B-bis"
    elif "DAINT_G" in res_mach.upper() or (res_mach.upper() == "G"):
        res_mach = "DAINT_G"
    else:
        res_mach = "Ripet_C-front"

    default_mach_idx = machine_options.index(res_mach) if res_mach in machine_options else 2

    reward_choice_map = {}
    auto_rew = st.session_state.get("auto_reward")
    auto_scheme = auto_rew.get("reward_type") if auto_rew else res.get("commercial_rewards", {}).get("reward_type", "Emissione Buono sconto")
    auto_fil = auto_rew.get("filiera_nome", auto_rew.get("filiera", "")) if auto_rew else res.get("commercial_rewards", {}).get("filiera_nome", "")

    schemes = [
        ("Emissione Buono sconto", "Direct Grocery Voucher"),
        ("Conversione a punti fedeltà", "Store Loyalty Points"),
        ("Omaggio", "Direct Free Gift / No Min. Spend"),
    ]

    for scheme_name, scheme_desc in schemes:
        if scheme_name == auto_scheme:
            tag = f" [Auto-Detected: {auto_fil}]" if auto_fil else " [Auto-Detected]"
            lbl = f"{scheme_name} ({scheme_desc}){tag}"
            reward_choice_map[lbl] = {
                "type": "auto",
                "param": None,
                "scheme_name": scheme_name,
            }
        else:
            lbl = f"{scheme_name} ({scheme_desc})"
            reward_choice_map[lbl] = {
                "type": "explicit",
                "param": scheme_name,
                "scheme_name": scheme_name,
            }

    res_rew = res.get("commercial_rewards", {})
    res_is_custom = bool(res_rew.get("is_custom", False))
    res_match_source = res_rew.get("match_source", "")
    res_rew_type = res_rew.get("reward_type", "")

    if res_is_custom:
        lbl_custom = "🛠️ Custom Promotional Policy (User-Defined Active)"
        reward_choice_map[lbl_custom] = {
            "type": "custom",
            "param": res_rew,
            "scheme_name": res_rew_type,
        }

    default_rew_lbl = None
    if res_is_custom:
        for lbl in reward_choice_map:
            if lbl.startswith("🛠️ Custom"):
                default_rew_lbl = lbl
                break
    elif res_match_source != "EXPLICIT_OVERRIDE":
        for lbl, info in reward_choice_map.items():
            if info["type"] == "auto":
                default_rew_lbl = lbl
                break
    else:
        for lbl, info in reward_choice_map.items():
            if info["type"] == "explicit" and info["scheme_name"] == res_rew_type:
                default_rew_lbl = lbl
                break

    reward_choice_labels = list(reward_choice_map.keys())
    if not default_rew_lbl or default_rew_lbl not in reward_choice_labels:
        default_rew_lbl = reward_choice_labels[0]
    default_rew_idx = reward_choice_labels.index(default_rew_lbl)

    col_sim_mach, col_sim_rew = st.columns([1, 1])
    with col_sim_mach:
        mach_choice_labels = [
            f"{m['icon']} {m['name']} ({m['cap']:,} bottles)"
            for m in CERTIFIED_MACHINES
        ]
        active_m_label = next(
            (f"{m['icon']} {m['name']} ({m['cap']:,} bottles)" for m in CERTIFIED_MACHINES if m['id'] == res_mach),
            mach_choice_labels[2]
        )
        selected_mach_label = st.selectbox(
            "⚙️ Active Machine Model (Option Split)",
            options=mach_choice_labels,
            index=mach_choice_labels.index(active_m_label),
            key="top_sim_machine_label",
            help="Directly switch active model. The comparison table below evaluates all 4 certified models simultaneously."
        )
        top_mach = next(m['id'] for m in CERTIFIED_MACHINES if f"{m['icon']} {m['name']} ({m['cap']:,} bottles)" == selected_mach_label)

    with col_sim_rew:
        top_rew_lbl = st.selectbox(
            "🎁 Commercial Reward Scheme",
            options=reward_choice_labels,
            index=default_rew_idx,
            key="top_sim_reward",
            help="Directly switch promotional reward policy. Footfall elasticity and monthly volumes update immediately.",
        )

    selected_rew_obj = reward_choice_map[top_rew_lbl]
    chosen_reward_param = selected_rew_obj["param"]

    machine_changed = (top_mach != res_mach)
    reward_changed = False
    if selected_rew_obj["type"] == "auto":
        reward_changed = (res_match_source == "EXPLICIT_OVERRIDE" or res_is_custom)
    elif selected_rew_obj["type"] == "custom":
        reward_changed = not res_is_custom
    else:
        reward_changed = (res_match_source != "EXPLICIT_OVERRIDE") or res_is_custom or (res_rew_type != selected_rew_obj["scheme_name"])

    inputs = res.get("inputs", {})
    comune_resolved = res.get("location_attributes", {}).get("resolved_comune")
    clean_comune = None
    if comune and str(comune).strip():
        clean_comune = str(comune).strip()
    elif comune_resolved and str(comune_resolved).strip():
        clean_comune = str(comune_resolved).strip()
    elif inputs.get("comune_input") and inputs.get("comune_input") != "AUTO_DETECTED":
        clean_comune = str(inputs.get("comune_input")).strip()

    if machine_changed or reward_changed:
        with st.spinner(f"🔄 Recalculating prospective yield for {top_mach} with {selected_rew_obj['scheme_name']}..."):
            try:
                import inspect
                predict_kwargs = {
                    "store_name": inputs.get("store_name", store_name),
                    "address": inputs.get("address", address),
                    "cap": inputs.get("cap", cap),
                    "machine_type": top_mach,
                    "comune": clean_comune,
                    "store_format": st.session_state.get("selected_format"),
                    "sales_area_sqm": inputs.get("sales_area_sqm") or st.session_state.get("selected_sqm"),
                    "cash_registers": inputs.get("cash_registers") or st.session_state.get("selected_casse"),
                    "explicit_reward": chosen_reward_param,
                    "go_live_month": inputs.get("go_live_month"),
                    "go_live_year": inputs.get("go_live_year"),
                    "enable_rampup": inputs.get("enable_rampup", True),
                }
                sig_params = inspect.signature(predictor.predict_new_location).parameters
                kwargs_to_pass = {k: v for k, v in predict_kwargs.items() if k in sig_params}
                new_res = predictor.predict_new_location(**kwargs_to_pass)
                st.session_state["prediction_result"] = new_res
                st.session_state["pending_input_machine_type"] = top_mach
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error recalculating scenario: {e}")

    # Evaluate all 4 machines for the comparative split matrix table
    split_evaluations = []
    base_b_day = float(res.get("realized_forecast", {}).get("underwritten_daily_bottles", 0.0))

    for m_info in CERTIFIED_MACHINES:
        m_id = m_info["id"]
        m_cap = float(m_info["cap"])
        is_cur = (res_mach.upper().replace(" ", "_") == m_id.upper().replace(" ", "_"))
        if is_cur:
            m_res = res
        else:
            try:
                m_res = predictor.predict_new_location(
                    store_name=inputs.get("store_name", store_name),
                    address=inputs.get("address", address),
                    cap=inputs.get("cap", cap),
                    machine_type=m_id,
                    comune=clean_comune,
                    store_format=st.session_state.get("selected_format"),
                    sales_area_sqm=inputs.get("sales_area_sqm") or st.session_state.get("selected_sqm"),
                    cash_registers=inputs.get("cash_registers") or st.session_state.get("selected_casse"),
                    explicit_reward=chosen_reward_param,
                )
            except Exception:
                m_res = res

        m_daily = float(m_res.get("realized_forecast", {}).get("underwritten_daily_bottles", base_b_day))
        m_monthly = int(round(m_daily * 30.4))
        m_annual = int(round(m_daily * 365))
        m_util = round((m_daily / m_cap) * 100.0, 1)
        m_p_over = float(m_res.get("probability_analysis", {}).get("hardware_operational_fit", {}).get("p_daily_overflow_risk", 0.0))

        if m_daily > m_cap:
            overload = m_daily - m_cap
            verdict_badge = f'<span style="background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 2px 6px; border-radius: 5px; font-weight: 700; font-size: 0.74rem; white-space: nowrap;">⚠️ Overload (+{(overload/m_cap)*100:.0f}%)</span>'
        elif m_util >= 50.0:
            verdict_badge = f'<span style="background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); padding: 2px 6px; border-radius: 5px; font-weight: 700; font-size: 0.74rem; white-space: nowrap;">✅ Optimal ({m_util:.0f}%)</span>'
        elif m_cap >= 4500.0 and m_daily < 150.0:
            verdict_badge = f'<span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); padding: 2px 6px; border-radius: 5px; font-weight: 700; font-size: 0.74rem; white-space: nowrap;">ℹ️ Buffer ({m_util:.0f}%)</span>'
        else:
            verdict_badge = f'<span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 2px 6px; border-radius: 5px; font-weight: 700; font-size: 0.74rem; white-space: nowrap;">✅ Certified ({m_util:.0f}%)</span>'

        split_evaluations.append({
            "info": m_info,
            "daily": m_daily,
            "monthly": m_monthly,
            "annual": m_annual,
            "util": m_util,
            "p_over": m_p_over,
            "verdict_badge": verdict_badge,
            "is_active": is_cur,
        })

    is_existing_site = bool("selected_store_pf" in st.session_state and st.session_state.get("selected_store_pf"))
    installed_pf = st.session_state.get("selected_store_pf", "")
    installed_status = st.session_state.get("selected_store_status", "")

    # Calculate optimal AI hardware recommendation based on daily throughput
    if base_b_day <= 260.0:
        recommended_mid = "DAINT_A-bis"
        rec_reason = "matches Daint A-bis (300 b/d) for optimal 70–85% utilization with minimal CapEx footprint."
    elif base_b_day <= 450.0:
        recommended_mid = "DAINT_B-bis"
        rec_reason = "matches Daint B-Bis (500 b/d) providing dual-stream throughput with a safe operational buffer."
    elif base_b_day <= 900.0:
        recommended_mid = "DAINT_G"
        rec_reason = "requires Daint G (4,500 b/d) dual-stream compaction to absorb volume without staff lockouts."
    else:
        recommended_mid = "Ripet_C-front"
        rec_reason = "requires Ripet C-Front (4,500 b/d) industrial compactor for maximum collection efficiency."

    rec_name = next((m["name"] for m in CERTIFIED_MACHINES if m["id"] == recommended_mid), "Daint G")
    rec_cap = next((m["cap"] for m in CERTIFIED_MACHINES if m["id"] == recommended_mid), 4500)

    # Render Certified 4-Machine Capacity & Throughput Split Table
    table_rows_list = []
    for item in split_evaluations:
        m = item["info"]
        m_id = m["id"]
        is_act = item["is_active"]
        is_rec = (m_id == recommended_mid)

        if is_existing_site:
            if is_act:
                active_chip = '<span style="background: rgba(37, 99, 235, 0.2); color: #60a5fa; border: 1px solid rgba(37, 99, 235, 0.4); padding: 3px 9px; border-radius: 12px; font-size: 0.72rem; font-weight: 800; white-space: nowrap; display: inline-block;">📍 INSTALLED</span>'
            else:
                active_chip = '<span style="color: #64748b; font-size: 0.74rem; white-space: nowrap;">Available</span>'
        else:
            if is_act and is_rec:
                active_chip = '<span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); padding: 3px 9px; border-radius: 12px; font-size: 0.72rem; font-weight: 800; white-space: nowrap; display: inline-block;">✨ AI BEST FIT (ACTIVE)</span>'
            elif is_rec:
                active_chip = '<span style="background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); padding: 3px 9px; border-radius: 12px; font-size: 0.72rem; font-weight: 800; white-space: nowrap; display: inline-block;">✨ AI BEST FIT</span>'
            elif is_act:
                active_chip = '<span style="background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); padding: 3px 9px; border-radius: 12px; font-size: 0.72rem; font-weight: 800; white-space: nowrap; display: inline-block;">ACTIVE VIEW</span>'
            else:
                active_chip = '<span style="color: #64748b; font-size: 0.74rem; white-space: nowrap;">Available</span>'

        row_bg = "rgba(56, 189, 248, 0.12)" if is_act else "rgba(255, 255, 255, 0.02)"
        row_border = "1px solid rgba(56, 189, 248, 0.4)" if is_act else "1px solid rgba(255, 255, 255, 0.06)"
        util_color = "#f87171" if item["util"] > 100 else "#4ade80"

        row_html = (
            f'<tr style="background: {row_bg}; border-bottom: {row_border};">'
            f'<td style="padding: 7px 8px; font-weight: 700; color: #f8fafc; white-space: nowrap;">'
            f'{m["icon"]} {m["name"]}'
            f'<div style="font-size: 0.68rem; color: #94a3b8; font-weight: 400;">{m["class"]}</div>'
            f'</td>'
            f'<td style="padding: 7px 6px; font-weight: 800; color: #cbd5e1; text-align: center; white-space: nowrap;">'
            f'<span style="background: rgba(255, 255, 255, 0.06); padding: 2px 6px; border-radius: 5px; border: 1px solid rgba(255, 255, 255, 0.1); font-size: 0.75rem;">{m["cap"]:,} b/d</span>'
            f'</td>'
            f'<td style="padding: 7px 8px; text-align: right; white-space: nowrap;">'
            f'<span style="font-weight: 800; color: #38bdf8; font-size: 0.86rem;">{item["daily"]:.0f} b/d</span>'
            f'<div style="font-size: 0.68rem; color: #94a3b8;">{item["monthly"]:,} /mo</div>'
            f'</td>'
            f'<td style="padding: 7px 6px; text-align: center; font-weight: 700; color: {util_color}; white-space: nowrap; font-size: 0.80rem;">'
            f'{item["util"]:.0f}%'
            f'</td>'
            f'<td style="padding: 7px 6px; text-align: center; white-space: nowrap;">'
            f'{item["verdict_badge"]}'
            f'</td>'
            f'<td style="padding: 7px 6px; font-size: 0.74rem; color: #94a3b8; text-align: center; white-space: nowrap;">'
            f'{m["servicing"]}'
            f'</td>'
            f'<td style="padding: 7px 8px; text-align: center; white-space: nowrap;">'
            f'{active_chip}'
            f'</td>'
            f'</tr>'
        )
        table_rows_list.append(row_html)
    table_rows_html = "".join(table_rows_list)

    if is_existing_site:
        advisory_html = (
            f'<div style="font-size: 0.80rem; color: #94a3b8; margin-top: 10px; padding: 8px 12px; background: rgba(37, 99, 235, 0.12); border: 1px solid rgba(37, 99, 235, 0.3); border-radius: 8px; display: flex; align-items: center; gap: 8px;">'
            f'<span style="font-size: 1.10rem;">📍</span>'
            f'<div><strong style="color: #93c5fd;">Active Network Installation:</strong> This location is registered in the Coripet <strong>{installed_status}</strong> database (Machine ID: <code>{installed_pf}</code>). The table compares its currently assigned hardware against all 4 certified models.</div>'
            f'</div>'
        )
    else:
        advisory_html = (
            f'<div style="font-size: 0.80rem; color: #94a3b8; margin-top: 10px; padding: 8px 12px; background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 8px; display: flex; align-items: center; gap: 8px;">'
            f'<span style="font-size: 1.10rem;">🤖</span>'
            f'<div><strong style="color: #86efac;">AI Automated Sizing Strategy:</strong> For a new site with no pre-existing hardware, the model evaluated projected volume (<strong>{base_b_day:.1f} b/d</strong>) and automatically selected <strong>{rec_name} ({rec_cap:,} b/d)</strong> — {rec_reason} You can switch the active view to any other machine option using the dropdown above.</div>'
            f'</div>'
        )

    split_matrix_html = (
        '<div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); '
        'border: 1px solid rgba(255, 255, 255, 0.12); border-left: 5px solid #38bdf8; border-radius: 10px; '
        'padding: 12px 14px; margin: 12px 0 18px 0; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">'
        '<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
        '<span style="font-weight: 800; font-size: 0.98rem; color: #f8fafc; letter-spacing: -0.2px;">'
        '⚙️ Certified Hardware Sizing & Capacity Split (All 4 Models)'
        '</span>'
        '<span style="font-size: 0.70rem; background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 2px 7px; border-radius: 10px; font-weight: 700;">'
        '4 CERTIFIED HARDWARE MODELS'
        '</span>'
        '</div>'
        '<div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 10px;">'
        'Comparative evaluation across all 4 recognized Coripet machines. Switch active model above to inspect in-depth lifecycle metrics.'
        '</div>'
        '<div style="width: 100%; overflow: hidden;">'
        '<table style="width: 100%; border-collapse: collapse; font-size: 0.80rem; text-align: left;">'
        '<thead>'
        '<tr style="border-bottom: 2px solid rgba(255, 255, 255, 0.15); color: #94a3b8; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.3px;">'
        '<th style="padding: 6px 8px; white-space: nowrap;">Machine Model</th>'
        '<th style="padding: 6px 6px; text-align: center; white-space: nowrap;">Capacity</th>'
        '<th style="padding: 6px 8px; text-align: right; white-space: nowrap;">Volume (Day / Mo)</th>'
        '<th style="padding: 6px 6px; text-align: center; white-space: nowrap;">Util.</th>'
        '<th style="padding: 6px 6px; text-align: center; white-space: nowrap;">Sizing Verdict</th>'
        '<th style="padding: 6px 6px; text-align: center; white-space: nowrap;">Service</th>'
        '<th style="padding: 6px 8px; text-align: center; white-space: nowrap;">Status</th>'
        '</tr>'
        '</thead>'
        f'<tbody>{table_rows_html}</tbody>'
        '</table>'
        '</div>'
        f'{advisory_html}'
        '</div>'
    )
    st.html(split_matrix_html)

    inv = res.get("investment_assessment", {})
    suit = res.get("location_suitability", {})
    real = res.get("realized_forecast", {})
    preds = res.get("predictions", {})
    rew = res.get("commercial_rewards", {})
    spatial = res.get("spatial_catchment", {})
    geo = res.get("location_attributes", {})
    ret = res.get("retail_intelligence", {})

    presenze_day = float(real.get("underwritten_daily_presenze", real.get("realized_daily_presenze", preds.get("expected_daily_presenze", 0.0))))
    bottiglie_day = float(real.get("underwritten_daily_bottles", real.get("realized_daily_bottles", preds.get("expected_daily_bottiglie", 0.0))))
    bottiglie_month = float(real.get("underwritten_monthly_bottles", preds.get("expected_monthly_bottiglie", bottiglie_day * 30.4)))
    bottles_per_person = round(bottiglie_day / max(1.0, presenze_day), 1)
    suitability_score = float(suit.get("suitability_score", 0.0))

    # Synchronize probability analysis directly with the active underwritten daily bottles
    from src.model.scoring import calculate_probability_factors
    prob = calculate_probability_factors(
        daily_bottles=bottiglie_day,
        confidence_score=inv.get("confidence", {}).get("confidence_score", 100),
        requested_machine=res.get("inputs", {}).get("machine_type", "DAINT_G")
    )
    res["probability_analysis"] = prob
    if "investment_assessment" in res:
        res["investment_assessment"]["probability_factors"] = prob

    rec_obj = inv.get("recommendation", {})
    rec_tier_name = rec_obj.get("tier_name", "MEDIO PRENDENTE (300 - 1,000 B/D)")
    decision = rec_obj.get("decision", "COMMERCIALLY VIABLE (PROCEED)")

    if bottiglie_day >= 2000.0:
        recommendation = "SUPER PRENDENTE"
    elif bottiglie_day >= 1000.0:
        recommendation = "ALTO PRENDENTE"
    elif bottiglie_day >= 300.0:
        recommendation = "MEDIO PRENDENTE"
    else:
        recommendation = "BASSO PRENDENTE"

    conf_obj = inv.get("confidence", {})
    confidence = conf_obj.get("confidence_rating", "HIGH")

    net = spatial.get("coripet_network", {})
    comp = spatial.get("competitive_density", {})
    dist_km = float(net.get("nearest_coripet_km", 0.0))
    cannibal = bool(net.get("cannibalization_alert", False))
    net_pos = net.get("network_position", "EXPANSION_NODE")
    nearest_store = f"{net.get('nearest_pf_name', 'Coripet')} ({net.get('nearest_pf_comune', '')})"

    is_risk_adjusted = real.get("is_risk_adjusted", False)
    under_b_val = real.get("underwritten_daily_bottles")

    p10_d = float(prob.get('downside_floor_p10_daily', 0.0))
    p10_m = int(round(p10_d * 30.4))
    p90_d = float(prob.get('upside_peak_p90_daily', 0.0))
    p90_m = int(round(p90_d * 30.4))

    formatted_text = (
        "LOCATION ANALYSIS\n\n"
        f"Expected PRESENZE/day: {int(round(presenze_day))}\n"
        f"Expected BOTTIGLIE/day: {int(round(bottiglie_day))}\n"
        f"Expected BOTTIGLIE/month: {int(round(bottiglie_month)):,}\n"
        f"Bottles per person: {bottles_per_person:.1f}\n"
        f"Location suitability: {int(round(suitability_score))}%\n"
        f"Recommendation: {recommendation}\n"
        f"Confidence: {confidence}"
    )

    # -------------------------------------------------------------
    # 1. Executive Status Header
    # -------------------------------------------------------------
    col_rec, col_conf, col_loc = st.columns([1.0, 1.0, 1.0])
    with col_rec:
        if "(" in decision:
            v_title = decision.split("(")[0].strip().title()
            v_action = decision.split("(")[1].replace(")", "").strip().title()
        else:
            v_title = decision.title()
            v_action = "Proceed" if ("PROCEED" in decision or "VIABLE" in decision) else "Review"

        chip_cls = "chip-proceed" if ("PROCEED" in v_action.upper() or "VIABLE" in v_title.upper()) else ("chip-review" if "REVIEW" in v_action.upper() else "chip-unfavorable")
        dot_cls = "dot-green" if ("PROCEED" in v_action.upper() or "VIABLE" in v_title.upper()) else ("dot-purple" if "REVIEW" in v_action.upper() else "dot-blue")
        st.html(
            f"""<div class="exec-card exec-card-verdict">
    <div class="exec-card-top">
        <span class="exec-card-lbl">💼 Investment Verdict</span>
        <span class="exec-status-dot {dot_cls}">● Active Model</span>
    </div>
    <div class="exec-verdict-box">
        <span class="exec-verdict-title">{v_title}</span>
        <span class="exec-action-chip {chip_cls}">{v_action}</span>
    </div>
    <div class="exec-card-bot">
        <span class="exec-bot-label">Potential Tier</span>
        <span class="exec-bot-val">{recommendation.title()}</span>
    </div>
</div>"""
        )

    with col_conf:
        clean_clearance = suit.get('suitability_status', 'SUITABLE').replace('_', ' ').title()
        conf_dot = "dot-blue" if confidence == "HIGH" else "dot-purple"
        st.html(
            f"""<div class="exec-card exec-card-conf">
    <div class="exec-card-top">
        <span class="exec-card-lbl">🎯 Model Confidence</span>
        <span class="exec-status-dot {conf_dot}">● Stacking AI</span>
    </div>
    <div>
        <div class="exec-conf-title">{confidence.title()} Confidence</div>
        <div class="exec-conf-sub">Calibrated Stacking Ensemble</div>
    </div>
    <div class="exec-card-bot">
        <span class="exec-bot-label">Clearance Gate</span>
        <span class="exec-bot-val">{clean_clearance}</span>
    </div>
</div>"""
        )

    with col_loc:
        comune_val = geo.get('resolved_comune') or geo.get('comune') or 'Milano'
        prov_val = geo.get('resolved_provincia') or geo.get('provincia') or 'MI'
        reg_val = geo.get('resolved_regione') or geo.get('regione') or 'Lombardia'
        area_raw = ret.get('nielsen_geographic_area') or geo.get('nielsen_area') or 'Area 1'
        area_str = str(area_raw).strip()
        if area_str.endswith('.0'):
            area_str = area_str[:-2]
        if area_str.isdigit():
            nielsen_val = f"Area {area_str}"
        elif "Area" in area_str:
            nielsen_val = area_str
        else:
            nielsen_val = f"Area {area_str}"

        inps = res.get('inputs', {})
        active_sqm = float(inps.get('sales_area_sqm') or 0.0)
        active_casse = float(inps.get('cash_registers') or 0.0)
        scale_chip = f'<span class="exec-chip">🏢 {active_sqm:.0f} m² · {active_casse:.0f} casse</span>' if active_casse > 0 else ""

        display_store_name = store_name or inps.get('store_name') or ret.get('insegna') or geo.get('store_name') or 'Store Site'
        display_address = address or inps.get('address') or geo.get('address') or ''
        store_disp = f"{display_store_name} · {display_address}" if display_address else str(display_store_name)
        cap_val = cap or inps.get('cap') or geo.get('cap') or st.session_state.get('loc_info', {}).get('Cap') or ''

        st.html(
            f"""<div class="exec-card exec-card-geo">
    <div class="exec-card-top">
        <span class="exec-card-lbl">📍 Target Location</span>
        <span class="exec-cap-pill">CAP {cap_val}</span>
    </div>
    <div>
        <div class="exec-geo-title">{comune_val} <span class="exec-geo-reg">({prov_val}), {reg_val}</span></div>
        <div class="exec-conf-sub" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{store_disp}</div>
    </div>
    <div class="exec-card-bot" style="gap: 6px; justify-content: flex-start; flex-wrap: nowrap; overflow: hidden;">
        <span class="exec-chip">🗺️ {nielsen_val}</span>
        <span class="exec-chip">⚙️ {inps.get('machine_type', machine_type)}</span>
        {scale_chip}
    </div>
</div>"""
        )

    st.divider()

    # -------------------------------------------------------------
    # 2. Key Operational Metrics Row
    # -------------------------------------------------------------
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.html(
            f"""<div class="metric-card metric-card-blue">
    <div class="metric-top">
        <span class="metric-title">Daily Footfall</span>
        <span class="metric-icon-badge badge-tint-blue">👥</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-blue">{int(round(presenze_day))}</span>
        <span class="metric-unit">users/d</span>
    </div>
    <div class="metric-subtitle">Expected Unique Visitors</div>
</div>"""
        )
    with k2:
        st.html(
            f"""<div class="metric-card metric-card-green">
    <div class="metric-top">
        <span class="metric-title">Daily PET Yield</span>
        <span class="metric-icon-badge badge-tint-green">♻️</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-green">{int(round(bottiglie_day)):,}</span>
        <span class="metric-unit">b/day</span>
    </div>
    <div class="metric-subtitle">Daily PET Intake Run</div>
</div>"""
        )
    with k3:
        st.html(
            f"""<div class="metric-card metric-card-purple">
    <div class="metric-top">
        <span class="metric-title">Monthly Volume</span>
        <span class="metric-icon-badge badge-tint-purple">📦</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-purple">{int(round(bottiglie_month)):,}</span>
        <span class="metric-unit">b/mo</span>
    </div>
    <div class="metric-subtitle">30-Day Operational Run</div>
</div>"""
        )
    with k4:
        st.html(
            f"""<div class="metric-card metric-card-amber">
    <div class="metric-top">
        <span class="metric-title">Bottles / User</span>
        <span class="metric-icon-badge badge-tint-amber">⚡</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-amber">{bottles_per_person:.1f}</span>
        <span class="metric-unit">b/user</span>
    </div>
    <div class="metric-subtitle">Recycle Intensity per Visit</div>
</div>"""
        )
    with k5:
        st.html(
            f"""<div class="metric-card metric-card-teal">
    <div class="metric-top">
        <span class="metric-title">Suitability Rank</span>
        <span class="metric-icon-badge badge-tint-teal">🏆</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-teal">{int(round(suitability_score))}%</span>
        <span class="metric-unit">score</span>
    </div>
    <div class="metric-subtitle">Network Benchmark Percentile</div>
</div>"""
        )

    # -------------------------------------------------------------
    # 3. Two-Column Dashboard: Gauge & Verbatim Report
    # -------------------------------------------------------------
    col_gauge, col_term = st.columns([1.1, 1.3])

    with col_gauge:
        st.subheader("📊 Network Suitability Gauge")
        gauge_fig = render_gauge(suitability_score)
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Scenario ranges
        sc_c1, sc_c2, sc_c3 = st.columns(3)
        with sc_c1:
            st.html(
                f"""<div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 12px; text-align: center; height: 75px; min-height: 75px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
    <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Floor (P25)</div>
    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 2px;">{int(round(prob.get('p25_conservative', bottiglie_day * 0.82))):,} b/d</div>
</div>"""
            )
        with sc_c2:
            st.html(
                f"""<div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 12px; text-align: center; height: 75px; min-height: 75px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
    <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Steady Base (P50)</div>
    <div style="font-size: 1.15rem; font-weight: 800; color: #1b5e20; margin-top: 2px;">{int(round(prob.get('p50_steady_state', bottiglie_day))):,} b/d</div>
</div>"""
            )
        with sc_c3:
            st.html(
                f"""<div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 12px; text-align: center; height: 75px; min-height: 75px; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.04);">
    <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Peak Ceiling (P75)</div>
    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 2px;">{int(round(prob.get('p75_peak', bottiglie_day * 1.25))):,} b/d</div>
</div>"""
            )

    with col_term:
        st.subheader("📋 Verbatim Location Analysis")
        st.markdown(f'<div class="terminal-box">{formatted_text}</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 3.25. Historical Actuals vs. Prospective AI Model Comparison
    # -------------------------------------------------------------
    from src.ui.historical_vs_prediction import render_historical_vs_prediction_section
    render_historical_vs_prediction_section(
        res=res,
        default_pf=st.session_state.get("selected_store_pf"),
        store_status=st.session_state.get("selected_store_status")
    )

    # -------------------------------------------------------------
    # 3.5. Statistical Probability & Future Performance Profile
    # -------------------------------------------------------------
    if prob:
        st.divider()
        st.subheader("🎲 Statistical Probability & Future Performance Profile")
        st.caption("Parametric risk modeling, milestone attainment likelihoods, and hardware capacity utilization calibrated to network volatility.")

        # Row 1: Capital Safety & Value-at-Risk 80% Band
        cap_safety = prob.get("success_probability", 0.0)
        risk_rating = str(prob.get("risk_rating", "MODERATE RISK"))
        
        if "LOW" in risk_rating.upper():
            risk_label = "Very Low Risk" if "VERY" in risk_rating.upper() else "Low Risk"
            risk_dot = "dot-green"
            risk_color = "#15803d"
            risk_footer_val = "Virtually Guaranteed" if "VIRTUALLY" in risk_rating.upper() else "Capital Protected"
        elif "MODERATE" in risk_rating.upper():
            risk_label = "Moderate Risk"
            risk_dot = "dot-purple"
            risk_color = "#92400e"
            risk_footer_val = "Acceptable Variance"
        else:
            risk_label = "Elevated Risk"
            risk_dot = "dot-blue"
            risk_color = "#991b1b"
            risk_footer_val = "Review Advised"

        p_col1, p_col2, p_col3 = st.columns([1.0, 1.0, 1.0])
        with p_col1:
            st.html(
                f"""<div class="exec-card" style="border-top: 3px solid #10b981;">
    <div class="exec-card-top">
        <span class="exec-card-lbl">🛡️ Capital Safety</span>
        <span class="exec-status-dot {risk_dot}">● {risk_label}</span>
    </div>
    <div style="margin: 4px 0 2px 0;">
        <div style="display: flex; align-items: baseline; gap: 4px;">
            <span style="font-size: 1.45rem; font-weight: 800; color: #15803d; letter-spacing: -0.3px; line-height: 1.15;">{cap_safety:.1f}%</span>
        </div>
        <div style="font-size: 0.76rem; font-weight: 600; color: #475569; margin-top: 2px;">
            Breakeven threshold ≥ 250 b/d
        </div>
    </div>
    <div class="exec-card-bot">
        <span class="exec-bot-label">Downside Safety</span>
        <span class="exec-bot-val" style="color: {risk_color}; font-weight: 700;">{risk_footer_val}</span>
    </div>
</div>"""
            )
        with p_col2:
            st.html(
                f"""<div class="exec-card" style="border-top: 3px solid #3b82f6;">
    <div class="exec-card-top">
        <span class="exec-card-lbl">📉 Downside Floor</span>
        <span class="exec-status-dot dot-blue">● P10 Floor</span>
    </div>
    <div style="margin: 4px 0 2px 0;">
        <div style="display: flex; align-items: baseline; gap: 4px;">
            <span style="font-size: 1.45rem; font-weight: 800; color: #1d4ed8; letter-spacing: -0.3px; line-height: 1.15;">{p10_d:.1f}</span>
            <span style="font-size: 0.76rem; font-weight: 600; color: #64748b;">b/d</span>
        </div>
        <div style="font-size: 0.76rem; font-weight: 600; color: #475569; margin-top: 2px;">
            {p10_m:,} bottles / month
        </div>
    </div>
    <div class="exec-card-bot">
        <span class="exec-bot-label">Statistical Certainty</span>
        <span class="exec-bot-val" style="color: #1d4ed8; font-weight: 700;">90% Exceedance</span>
    </div>
</div>"""
            )
        with p_col3:
            st.html(
                f"""<div class="exec-card" style="border-top: 3px solid #8b5cf6;">
    <div class="exec-card-top">
        <span class="exec-card-lbl">📈 Upside Ceiling</span>
        <span class="exec-status-dot dot-purple">● P90 Peak</span>
    </div>
    <div style="margin: 4px 0 2px 0;">
        <div style="display: flex; align-items: baseline; gap: 4px;">
            <span style="font-size: 1.45rem; font-weight: 800; color: #6d28d9; letter-spacing: -0.3px; line-height: 1.15;">{p90_d:.1f}</span>
            <span style="font-size: 0.76rem; font-weight: 600; color: #64748b;">b/d</span>
        </div>
        <div style="font-size: 0.76rem; font-weight: 600; color: #475569; margin-top: 2px;">
            {p90_m:,} bottles / month
        </div>
    </div>
    <div class="exec-card-bot">
        <span class="exec-bot-label">Peak Potential</span>
        <span class="exec-bot-val" style="color: #6d28d9; font-weight: 700;">Top 10% Upside</span>
    </div>
</div>"""
            )

        # Row 2: Official Volume Performance Outlook (Single Solid Probability Table)
        tiers_p = prob.get("performance_tiers", {})
        if "p_super_over_2000" in tiers_p and "p_medio_300_1000" in tiers_p:
            p_basso = float(tiers_p.get("p_basso_under_300", 0.0))
            p_medio = float(tiers_p.get("p_medio_300_1000", 0.0))
            p_alto = float(tiers_p.get("p_alto_1000_2000", 0.0))
            p_super = float(tiers_p.get("p_super_over_2000", 0.0))
            dominant_tier = tiers_p.get("most_likely_tier", "MEDIO PRENDENTE (300 - 1,000 B/D)")
        else:
            # Dynamically compute accurate Coripet volume tier CDF using model lognormal parameters
            sigma_val = float(prob.get("sigma_applied", 0.35))
            mu_val = math.log1p(max(0.0, float(bottiglie_day)))

            def _local_norm_cdf(x_val):
                z = (math.log1p(x_val) - mu_val) / sigma_val
                return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

            c_300 = _local_norm_cdf(300.0)
            c_1000 = _local_norm_cdf(1000.0)
            c_2000 = _local_norm_cdf(2000.0)
            p_basso = round(max(0.0, min(100.0, c_300 * 100.0)), 1)
            p_medio = round(max(0.0, min(100.0, (c_1000 - c_300) * 100.0)), 1)
            p_alto = round(max(0.0, min(100.0, (c_2000 - c_1000) * 100.0)), 1)
            p_super = round(max(0.0, min(100.0, (1.0 - c_2000) * 100.0)), 1)

            t_map = {
                "BASSO PRENDENTE (< 300 B/D)": p_basso,
                "MEDIO PRENDENTE (300 - 1,000 B/D)": p_medio,
                "ALTO PRENDENTE (1,000 - 2,000 B/D)": p_alto,
                "SUPER PRENDENTE (> 2,000 B/D)": p_super
            }
            dominant_tier = max(t_map.items(), key=lambda x: x[1])[0]

        matrix_rows = [
            {
                "tier_name": "Basso Prendente (Low Yield)",
                "tier_range": "< 300 b/d",
                "annual_range": "< 109,500 bottles/year",
                "tier_prob": p_basso,
                "color": "#e53935",
                "is_dominant": "BASSO" in dominant_tier.upper(),
                "role": "Modest volume; operating cost review or merchant co-financing advised"
            },
            {
                "tier_name": "Medio Prendente (Standard Yield)",
                "tier_range": "300 - 1,000 b/d",
                "annual_range": "109,500 - 365,000 bottles/year",
                "tier_prob": p_medio,
                "color": "#1e88e5",
                "is_dominant": "MEDIO" in dominant_tier.upper(),
                "role": "Sustainable standard network benchmark (Optimal commercial viability)"
            },
            {
                "tier_name": "Alto Prendente (High Yield)",
                "tier_range": "1,000 - 2,000 b/d",
                "annual_range": "365,000 - 730,000 bottles/year",
                "tier_prob": p_alto,
                "color": "#43a047",
                "is_dominant": "ALTO" in dominant_tier.upper(),
                "role": "Top commercial performer with high collection yield and strong footfall"
            },
            {
                "tier_name": "Super Prendente (Super Anchor)",
                "tier_range": "> 2,000 b/d",
                "annual_range": "> 730,000 bottles/year",
                "tier_prob": p_super,
                "color": "#8e24aa",
                "is_dominant": "SUPER" in dominant_tier.upper(),
                "role": "Strategic high-turnover anchor hub of excellence (Hypermarket / Big-Box)"
            }
        ]

        table_rows_html = ""
        for r in matrix_rows:
            dom_badge = """<span style="background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; font-size: 0.60rem; font-weight: 800; padding: 1px 6px; border-radius: 4px; margin-left: 6px; text-transform: uppercase;">DOMINANT</span>""" if r["is_dominant"] else ""
            row_bg = "background: #f8fafc;" if r["is_dominant"] else "background: #ffffff;"
            table_rows_html += f"""<tr style="border-bottom: 1px solid #f1f5f9; {row_bg}">
    <td style="padding: 7px 12px; color: #0f172a; font-size: 0.82rem;">
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: {r['color']}; flex-shrink: 0;"></span>
            <span style="font-weight: 700;">{r['tier_name']}</span>
            {dom_badge}
        </div>
        <div style="font-size: 0.69rem; color: #64748b; font-weight: 500; margin-top: 2px; padding-left: 14px;">
            {r['role']}
        </div>
    </td>
    <td style="padding: 7px 12px; color: #0f172a; white-space: nowrap; font-weight: 700; font-size: 0.82rem; vertical-align: middle;">
        {r['tier_range']}
    </td>
    <td style="padding: 7px 12px; color: #475569; white-space: nowrap; font-size: 0.78rem; font-weight: 500; vertical-align: middle;">
        {r['annual_range']}
    </td>
    <td style="padding: 7px 12px; width: 220px; max-width: 240px; vertical-align: middle;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 3px;">
            <span style="font-size: 0.68rem; color: #64748b; font-weight: 600;">Probability</span>
            <span style="font-weight: 800; font-size: 0.88rem; color: {r['color']};">{r['tier_prob']:.1f}%</span>
        </div>
        <div style="width: 100%; background: #e2e8f0; border-radius: 4px; height: 5px; overflow: hidden;">
            <div style="width: {min(100.0, max(0.0, r['tier_prob']))}%; background: {r['color']}; height: 5px; border-radius: 4px;"></div>
        </div>
    </td>
</tr>"""

        st.html(
            f"""<div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.03); margin-bottom: 16px;">
    <div style="padding: 8px 14px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;">
            <span style="font-weight: 800; font-size: 0.88rem; color: #0f172a;">🎯 Volume Performance Classification & Probability Outlook</span>
            <span style="font-size: 0.71rem; color: #64748b;">(Official Coripet volume spectrum &lt; 300, 300–1,000, 1,000–2,000, &gt; 2,000 b/d • Sums to 100%)</span>
        </div>
        <div style="display: flex; align-items: center; gap: 6px;">
            <span style="font-size: 0.72rem; color: #64748b; font-weight: 600;">Dominant Outlook:</span>
            <span style="background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 5px; font-size: 0.73rem; font-weight: 800;">
                {dominant_tier}
            </span>
        </div>
    </div>
    <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; font-family: Inter, sans-serif; text-align: left;">
            <thead>
                <tr style="background: #ffffff; border-bottom: 1px solid #e2e8f0; font-size: 0.68rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                    <th style="padding: 7px 12px;">Volume Tier & Strategic Role</th>
                    <th style="padding: 7px 12px; width: 140px;">Daily Throughput</th>
                    <th style="padding: 7px 12px; width: 180px;">Annual Output (365d)</th>
                    <th style="padding: 7px 12px; width: 220px;">Probability</th>
                </tr>
            </thead>
            <tbody>
                {table_rows_html}
            </tbody>
        </table>
    </div>
</div>"""
        )

        # Row 3: Full-Width AI Hardware Fit & Upgrade Advisor
        hw_p = prob.get("hardware_operational_fit", {})
        p_opt = float(hw_p.get("p_optimal_utilization", 0.0))
        p_over = float(hw_p.get("p_daily_overflow_risk", 0.0))
        p_under = float(hw_p.get("p_underutilization_risk", 0.0))
        req_machine = str(hw_p.get("requested_machine", machine_type))
        hw_class = str(hw_p.get("hardware_class", "Standard RVM"))

        if "A-BIS" in req_machine.upper():
            sat_thresh = 300.0
            storage_cap = 300
            storage_kg = 8.0
            display_hw_class = "Compact Single-Stream (Daint A-bis)"
            rhythm = "Daily Servicing (Low Cap)"
            is_heavy_unit = False
        elif "B-BIS" in req_machine.upper():
            sat_thresh = 500.0
            storage_cap = 500
            storage_kg = 14.0
            display_hw_class = "Compact Dual-Stream (Daint B-Bis)"
            rhythm = "Twice-Weekly Servicing"
            is_heavy_unit = False
        elif "DAINT_G" in req_machine.upper() or ("DAINT" in req_machine.upper() and "G" in req_machine.upper()):
            sat_thresh = 4500.0
            storage_cap = 4500
            storage_kg = 120.0
            display_hw_class = "Heavy-Duty Dual-Stream (Daint G)"
            rhythm = "Bi-Weekly Scheduled Servicing"
            is_heavy_unit = True
        else:
            sat_thresh = 4500.0
            storage_cap = 4500
            storage_kg = 120.0
            display_hw_class = "Heavy-Duty Industrial Compactor (Ripet C-Front)"
            rhythm = "Bi-Weekly Scheduled Servicing"
            is_heavy_unit = True

        # Re-evaluate overflow probability with true industrial capacity
        if is_heavy_unit and bottiglie_day <= sat_thresh:
            p_over = 0.0 if bottiglie_day <= 3000.0 else round(max(0.0, min(100.0, (1.0 - 0.5 * (1.0 + math.erf((math.log1p(sat_thresh) - math.log1p(bottiglie_day)) / (0.28 * math.sqrt(2.0)))))) * 100.0), 1)

        days_buffer = round(storage_cap / max(1.0, bottiglie_day), 1)
        is_bottleneck = p_over > 20.0 or (sat_thresh > 0 and bottiglie_day > sat_thresh)
        overload_pct = round(((bottiglie_day - sat_thresh) / sat_thresh) * 100.0, 0) if (sat_thresh > 0 and bottiglie_day > sat_thresh) else 0.0

        if is_bottleneck:
            rec_model = "Daint G / Ripet C-Front (4,500 b/d)"
            rec_class = "Heavy-Duty Dual-Stream Compactor"
            rec_capacity = "4,500 bottles (~120 kg PET)"
            rec_overflow = "< 2.0%"
            rec_overflow_val = 2.0
            rec_servicing = "Bi-Weekly Scheduled Evacuation"
            rec_rationale = f"Forecast volume ({bottiglie_day:.0f} b/d) exceeds {req_machine} capacity ({sat_thresh:,.0f} b/d). Upgrading to Daint G or Ripet C-Front (4,500 bottle capacity) eliminates overflow lockouts."

            banner_html = f"""<div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.15rem;">⚠️</span>
        <span style="font-weight: 700; font-size: 0.85rem; color: #9f1239;">CAPACITY BOTTLENECK DETECTED: Forecast volume ({bottiglie_day:.0f} b/d) exceeds {req_machine} capacity ({sat_thresh:.0f} b/d) by +{overload_pct:.0f}%</span>
    </div>
    <span style="font-size: 0.70rem; font-weight: 800; background: #be123c; color: #ffffff; padding: 3px 9px; border-radius: 5px; text-transform: uppercase; letter-spacing: 0.3px;">Hardware Upgrade Advised</span>
</div>"""
            card_right_badge = '<span style="background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; border-radius: 5px; padding: 2px 8px; font-size: 0.70rem; font-weight: 800;">✨ RECOMMENDED UPGRADE</span>'
            rec_title = "Optimal Hardware Upgrade"
        else:
            rec_model = req_machine
            rec_class = display_hw_class
            rec_capacity = f"{storage_cap:,} bottles (~{storage_kg:.0f} kg PET)"
            rec_overflow = f"{p_over:.1f}%"
            rec_overflow_val = p_over
            rec_servicing = rhythm
            rec_rationale = f"{req_machine} is Coripet's premier high-capacity eco-compactor (storage capacity: {storage_cap:,} bottles / ~{storage_kg:.0f} kg PET). Projected volume ({bottiglie_day:.0f} b/d) provides ~{days_buffer} days of autonomous buffer, operating smoothly with scheduled servicing."

            banner_html = f"""<div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 1.15rem;">✅</span>
        <span style="font-weight: 700; font-size: 0.85rem; color: #166534;">HARDWARE SPECIFICATION CERTIFIED: {req_machine} features {storage_cap:,}-bottle capacity (~{storage_kg:.0f} kg PET), comfortably supporting {bottiglie_day:.0f} b/d (~{days_buffer} days buffer)</span>
    </div>
    <span style="font-size: 0.70rem; font-weight: 800; background: #16a34a; color: #ffffff; padding: 3px 9px; border-radius: 5px; text-transform: uppercase; letter-spacing: 0.3px;">Certified Match</span>
</div>"""
            card_right_badge = '<span style="background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; border-radius: 5px; padding: 2px 8px; font-size: 0.70rem; font-weight: 800;">✅ CERTIFIED FIT</span>'
            rec_title = "Operational Clearance & Strategy"

        st.html(
            f"""<div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.03); margin-bottom: 20px;">
    <div style="padding: 10px 16px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box; flex-wrap: wrap; gap: 8px;">
        <div>
            <div style="font-weight: 800; font-size: 0.90rem; color: #0f172a; display: flex; align-items: center; gap: 6px;">
                <span>🛠️</span> AI Hardware Fit & Capacity Certification
            </div>
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 500; margin-top: 2px;">
                Physical compactor bin evaluation and operational logistics rhythm for <strong>{req_machine}</strong>
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span style="background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 5px; padding: 3px 9px; font-size: 0.72rem; font-weight: 700;">
                Selected: {req_machine}
            </span>
            <span style="background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; border-radius: 5px; padding: 3px 9px; font-size: 0.72rem; font-weight: 700;">
                Storage: {storage_cap:,} bottles
            </span>
        </div>
    </div>
    
    <div style="padding: 14px 16px; background: #ffffff;">
        {banner_html}
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; box-sizing: border-box;">
            <!-- Column 1: Current Selected Machine -->
            <div style="background: #f8fafc; border: 1px solid {'#fecdd3' if is_bottleneck else '#e2e8f0'}; border-radius: 8px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.70rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b;">Current Selected Unit</span>
                        <span style="background: {'#fee2e2' if is_bottleneck else '#ecfdf5'}; color: {'#991b1b' if is_bottleneck else '#047857'}; border: 1px solid {'#fca5a5' if is_bottleneck else '#a7f3d0'}; border-radius: 4px; padding: 2px 7px; font-size: 0.68rem; font-weight: 700;">{'OVERLOAD' if is_bottleneck else 'CERTIFIED'}</span>
                    </div>
                    <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a;">{req_machine}</div>
                    <div style="font-size: 0.75rem; color: #64748b; font-weight: 600; margin-top: 2px;">{display_hw_class}</div>
                    
                    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 0.78rem; line-height: 1.6; color: #334155;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>Storage Capacity:</span>
                            <strong>{storage_cap:,} bottles (~{storage_kg:.0f} kg PET)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                            <span>Daily Saturation Limit:</span>
                            <strong>>{sat_thresh:.0f} b/d</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                            <span>Daily Overflow Risk:</span>
                            <strong style="color: {'#dc2626' if p_over > 20 else '#16a34a'};">{p_over:.1f}%</strong>
                        </div>
                        <div style="width: 100%; background: #e2e8f0; border-radius: 4px; height: 5px; overflow: hidden; margin: 4px 0 6px 0;">
                            <div style="width: {min(100.0, max(0.0, p_over))}%; background: {'#dc2626' if p_over > 20 else '#16a34a'}; height: 5px; border-radius: 4px;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                            <span>Autonomous Storage Buffer:</span>
                            <strong>~{days_buffer} days of collection</strong>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.72rem; color: #475569; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 7px 10px;">
                    🚚 <strong>Logistics:</strong> {rhythm}
                </div>
            </div>

            <!-- Column 2: AI Operational Strategy / Recommended Target -->
            <div style="background: #f8fafc; border: 1px solid {'#bbf7d0' if not is_bottleneck else '#fecdd3'}; border-radius: 8px; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
                <div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-size: 0.70rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; color: #047857;">{rec_title}</span>
                        {card_right_badge}
                    </div>
                    <div style="font-size: 1.02rem; font-weight: 800; color: #047857;">{rec_model}</div>
                    <div style="font-size: 0.75rem; color: #059669; font-weight: 600; margin-top: 2px;">{rec_class}</div>
                    
                    <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 0.78rem; line-height: 1.6; color: #334155;">
                        <div style="display: flex; justify-content: space-between;">
                            <span>Projected Daily Intake:</span>
                            <strong style="color: #047857;">{bottiglie_day:.0f} bottles / day</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                            <span>Compaction Efficiency:</span>
                            <strong>Dual-Stream Industrial Crusher</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                            <span>Midday Lockout Risk:</span>
                            <strong style="color: #16a34a;">0.0% (Zero Downtime)</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 3px;">
                            <span>Staff Intervention:</span>
                            <strong>None required during operating hours</strong>
                        </div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.72rem; color: #065f46; background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 6px; padding: 7px 10px;">
                    💡 <strong>AI Assessment:</strong> {rec_rationale}
                </div>
            </div>
        </div>
    </div>
</div>"""
        )

    st.divider()

    # -------------------------------------------------------------
    # 4. Deep Intelligence Rows: Commercial Rewards & Spatial Catchment
    # -------------------------------------------------------------
    c_rew, c_spat = st.columns(2)

    with c_rew:
        st.subheader("🎁 Commercial Rewards & Filiera Intelligence")
        reward_type = rew.get("reward_type", "Emissione Buono sconto")
        REWARD_TYPE_EN = {
            "Emissione Buono sconto": "Direct Grocery Voucher",
            "Conversione a punti fedeltà": "Loyalty Points Conversion",
            "Omaggio": "Free Gift / Cash Reward",
        }
        reward_badge_label = REWARD_TYPE_EN.get(reward_type, reward_type)
        r_badge = "badge-reward"
        reward_desc_en = (
            rew.get("definition_en")
            or rew.get("contract_rule")
            or "Direct discount on grocery purchases with minimum store receipt requirement. Standard national supermarket voucher."
        )
        st.html(
            f"""<div class="intel-section" style="border-left: 5px solid #1976d2; min-height: 220px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
    <div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-weight: 700; font-size: 1.05rem; color: #0f172a;">{rew.get('filiera', 'Inferred Circuit')}</span>
            <span class="badge {r_badge}">{reward_badge_label}</span>
        </div>
        <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.55;">
            <strong>Incentive Tier:</strong> {rew.get('incentive_tier', 'GROCERY_VOUCHER')}<br>
            <strong>Incentive Strength:</strong> {rew.get('incentive_strength', 'HIGH')}<br>
            <strong>Voucher Terms:</strong> {rew.get('typical_voucher', 'Store Voucher Standard')}<br>
            <strong>Reward Policy Description:</strong> <em>{reward_desc_en}</em>
        </p>
    </div>
</div>"""
        )

    with c_spat:
        st.subheader("🗺️ Spatial Catchment & Local Network Density")
        c_tag = "badge-low" if cannibal else "badge-high"
        c_label = "CANNIBALIZATION ALERT" if cannibal else "CLEAR EXPANSION"

        within_3km = net.get("coripet_count_3km", 0)
        comp_density = f"{comp.get('total_grocery_stores', 'N/A')} stores ({comp.get('supermarkets', 0)} Super, {comp.get('discounts', 0)} Disc)" if comp else "Standard Catchment"

        st.html(
            f"""<div class="intel-section" style="border-left: 5px solid #2e7d32; min-height: 220px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
    <div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-weight: 700; font-size: 1.05rem; color: #0f172a;">Catchment Analysis</span>
            <span class="badge {c_tag}">{c_label}</span>
        </div>
        <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.55;">
            <strong>Nearest Coripet Machine:</strong> {dist_km:.2f} km ({nearest_store})<br>
            <strong>Network Role:</strong> {net_pos}<br>
            <strong>Machines Within 3km:</strong> {within_3km} active installations<br>
            <strong>Competitive Grocery Density:</strong> {comp_density}
        </p>
    </div>
</div>"""
        )

    # -------------------------------------------------------------
    # 4.5. All Reward Types Sensitivity & Impact Analysis Matrix
    # -------------------------------------------------------------
    sens_data = res.get("reward_sensitivity_analysis", [])
    if sens_data:
        st.subheader("📊 Promotional Reward Sensitivity & Impact Matrix")
        st.caption("Comparative operational yield, footfall, and percentage variance across all commercial reward options.")

        active_rew_type = rew.get("reward_type", "")
        is_custom_active = bool(rew.get("is_custom", False))
        active_label = active_rew_type or "Standard Promotion"
        if is_custom_active:
            active_label = f"Custom ({rew.get('discount_euro', 3):.2f}€ su {rew.get('min_spend_euro', 25):.0f}€)"
        fil_name = rew.get("filiera", "")
        fil_info = f" • Contract Circuit: <strong>{fil_name}</strong>" if fil_name else ""
        underwritten_active_b = res.get("realized_forecast", {}).get("realized_daily_bottles", bottiglie_day)

        st.html(
            f"""<div style="background: rgba(22, 163, 74, 0.08); border-left: 4px solid #16a34a; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; font-size: 0.90rem; color: #166534; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
    <div>
        <strong>Currently Selected Promotion:</strong> <span style="font-weight: 800; color: #15803d; background: #dcfce7; padding: 3px 9px; border-radius: 5px; border: 1px solid #86efac; margin-left: 4px;">🟢 {active_label}</span>{fil_info}
    </div>
    <div style="font-size: 0.82rem; font-weight: 700; color: #15803d;">
        Baseline Active Yield: {underwritten_active_b:.1f} b/d
    </div>
</div>"""
        )

        table_rows = []
        for item in sens_data:
            is_active = bool(item.get("is_selected")) or (not is_custom_active and item.get("reward_type") == active_rew_type)
            tag = "🟢 ACTIVE SELECTED" if is_active else "⚪ Alternative"
            diff_b = item.get("diff_bottles", 0.0)
            pct_v = item.get("pct_variance", 0.0)
            var_label = "🟢 ACTIVE BASELINE" if is_active else f"{'+' if diff_b > 0 else ''}{pct_v:.1f}% ({'+' if diff_b > 0 else ''}{diff_b:.1f} b/d)"
            scheme_display = f"{item['reward_type']} 🟢 [ACTIVE SELECTED]" if is_active else item["reward_type"]
            table_rows.append({
                "Status": tag,
                "Reward Policy Scheme": scheme_display,
                "Daily Bottles (b/d)": f"{item['daily_bottles']:.1f}",
                "Monthly Volume": f"{item['monthly_bottles']:,} b/mo",
                "Active Footfall": f"{item['daily_presenze']:.1f} u/d",
                "Variance vs Selected": var_label,
                "Incentive Tier": item.get("incentive_strength", "N/A"),
            })

        df_display = pd.DataFrame(table_rows)
        try:
            def highlight_active(row):
                if "ACTIVE" in str(row["Status"]):
                    return ['background-color: rgba(34, 197, 94, 0.12); font-weight: bold;'] * len(row)
                return [''] * len(row)

            styled_df = df_display.style.apply(highlight_active, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_display, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 5. Clearance Gates (Core 6 Operational Gates) & Seasonality Projection
    # -------------------------------------------------------------
    raw_gate_results = suit.get("gate_results", {})
    if raw_gate_results:
        st.subheader("🛡️ Operational Clearance Gates")

        eff_b = float(bottiglie_day)
        req_m = str(machine_type)

        # Filter strictly to the 6 core operational clearance gates
        core_gate_keys = [
            "cannibalization_gate",
            "economic_volume_gate",
            "store_format_gate",
            "demographic_gate",
            "hardware_sizing_gate",
            "logistics_feasibility_gate",
        ]
        gate_results = {k: raw_gate_results[k] for k in core_gate_keys if k in raw_gate_results}

        # Format and enrich each gate card
        cards_html = []
        for g_name, g_info in gate_results.items():
            status = g_info.get("status", "PASS")
            g_label = g_info.get("label", g_name.replace("_", " ").title())
            g_metric = g_info.get("metric", "")
            g_flag = g_info.get("flag", status)

            # Ensure Economic Breakeven Gate strictly reflects 250 b/d threshold with monthly deficit/surplus
            if g_name == "economic_volume_gate" or "breakeven" in g_label.lower():
                deficit = round(250.0 - eff_b, 1)
                if eff_b < (250.0 * 0.90):
                    status = "FAIL"
                    g_flag = "UNECONOMIC VOLUME DEFICIT"
                    g_metric = f"Volume deficit of -{deficit:.1f} b/d (-{int(deficit*30.4):,} b/mo) vs 250 b/d operational breakeven"
                elif eff_b < 250.0:
                    status = "WARN"
                    g_flag = "TIGHT MARGIN"
                    g_metric = f"Marginal volume ({eff_b:.1f} b/d) within 10% of 250 b/d operational breakeven floor"
                else:
                    surplus = round(eff_b - 250.0, 1)
                    status = "PASS"
                    g_flag = "COMMERCIALLY PROFITABLE"
                    g_metric = f"Robust collection yield (+{surplus:.1f} b/d / +{int(surplus*30.4):,} b/mo headroom over 250 b/d breakeven)"

            # Ensure Hardware Sizing Gate eliminates the contradiction
            if g_name == "hardware_sizing_gate":
                if status == "WARN" and "OVERSIZED" in g_flag.upper():
                    g_metric = f"Heavy-duty '{req_m}' (6,000 b) is oversized for {eff_b:.1f} b/d; compact unit (Ripet_V2) advised"
                elif status == "WARN" and "UNDER" in g_flag.upper():
                    g_metric = f"Compact '{req_m}' is undersized for high-yield {eff_b:.1f} b/d; heavy-duty unit advised"
                elif status == "PASS":
                    g_metric = f"Hardware '{req_m}' is certified and optimally matched for {eff_b:.1f} b/d"

            card_class = "gate-pass" if status == "PASS" else ("gate-fail" if status == "FAIL" else "gate-warn")
            pill_class = "pill-pass" if status == "PASS" else ("pill-fail" if status == "FAIL" else "pill-warn")
            icon = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⚠️")
            cards_html.append(
                f"""<div class="gate-card {card_class}">
    <div>
        <div class="gate-title">
            <span>{icon}</span> <span>{g_label}</span>
        </div>
        <div class="gate-metric">
            {g_metric}
        </div>
    </div>
    <div class="gate-footer">
        <span class="gate-status-pill {pill_class}">{g_flag}</span>
    </div>
</div>"""
            )

        # 4. Executive Underwriting Verdict Banner
        pass_count = sum(1 for g in gate_results.values() if g.get("status") == "PASS")
        warn_count = sum(1 for g in gate_results.values() if g.get("status") == "WARN")
        fail_count = sum(1 for g in gate_results.values() if g.get("status") == "FAIL")
        total_count = len(gate_results)

        if fail_count > 0:
            b_bg = "#fff1f2"
            b_border = "#fecdd3"
            b_color = "#9f1239"
            p_bg = "#be123c"
            v_title = "DEPLOYMENT CONDITIONAL / REJECTED: Capital Protection Gate Failed"
            v_sub = f"{fail_count} Disqualifying Gate Deficit identified; requires senior operational approval before contract commitment."
            v_badge = f"{pass_count}/{total_count} GATES CLEARED"
        elif warn_count > 0:
            b_bg = "#fffbeb"
            b_border = "#fde68a"
            b_color = "#92400e"
            p_bg = "#d97706"
            v_title = "CONDITIONAL CLEARANCE: Operational & Sizing Optimization Advised"
            v_sub = f"Economic breakeven and trade area validated; {warn_count} operational recommendation(s) identified to maximize return."
            v_badge = f"{pass_count}/{total_count} GATES CLEARED"
        else:
            b_bg = "#f0fdf4"
            b_border = "#bbf7d0"
            b_color = "#166534"
            p_bg = "#16a34a"
            v_title = "UNCONDITIONAL CLEARANCE: Site Fully Approved For Deployment"
            v_sub = f"All {total_count} Operational, Demographic, Format, Commercial, Hardware, and Logistics Gates Cleared."
            v_badge = f"{total_count}/{total_count} GATES CLEARED"

        st.html(
            f"""<div style="background: {b_bg}; border: 1px solid {b_border}; border-radius: 9px; padding: 12px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
    <div>
        <div style="font-weight: 800; font-size: 0.92rem; color: {b_color}; display: flex; align-items: center; gap: 6px;">
            <span>🛡️</span> <span>Underwriting Verdict: {v_title}</span>
        </div>
        <div style="font-size: 0.74rem; color: {b_color}; opacity: 0.90; margin-top: 2px;">
            {v_sub}
        </div>
    </div>
    <span style="background: {p_bg}; color: #ffffff; padding: 4px 10px; border-radius: 5px; font-size: 0.72rem; font-weight: 800; letter-spacing: 0.4px;">
        {v_badge}
    </span>
</div>"""
        )

        cards_str = "\n".join(cards_html)
        st.html(f"""<div class="gate-grid">\n{cards_str}\n</div>""")

    # -------------------------------------------------------------
    # 6. 12-Month Rolling Operational Intake Profile (Year 1)
    # -------------------------------------------------------------
    lifecycle_h = res.get("lifecycle_horizon", {})
    rolling_profile = lifecycle_h.get("year1_rolling_profile")
    monthly_profile = rolling_profile or (
        preds.get("underwritten_monthly_profile")
        or preds.get("monthly_collection_profile")
        or real.get("underwritten_monthly_profile")
        or preds.get("monthly_predictions")
        or []
    )
    if monthly_profile:
        st.divider()
        l_month_str = lifecycle_h.get("launch_month_name", "Go-Live Month")
        l_year_int = lifecycle_h.get("launch_year", 2026)
        st.subheader("📈 12-Month Rolling Operational Intake Profile")
        st.caption(
            f"Confirmed Year 1 operational collection targets and seasonal intake profile starting from **{l_month_str} {l_year_int}**."
        )

        season_fig = render_seasonality_chart(monthly_profile, steady_state_benchmark=bottiglie_day)
        st.plotly_chart(season_fig, use_container_width=True)

        if isinstance(monthly_profile, list) and monthly_profile and isinstance(monthly_profile[0], dict):
            m_rows = []
            for m in monthly_profile:
                d_val = m.get("daily_bottiglie_median", m.get("daily_bottiglie", 0.0))
                t_val = m.get("monthly_target_bottiglie_median", m.get("monthly_target_bottiglie", int(d_val * 30.4)))
                m_rows.append({
                    "Operational Month": m.get("lifecycle_label", f"{m.get('month_name', '')} ({m.get('days_in_month', 30)}d)"),
                    "Projected Daily (b/d)": f"{d_val:.1f} b/d",
                    "Monthly Target (b/mo)": f"{t_val:,} b/mo",
                    "Adoption Maturity": m.get("maturity_pct", "100%"),
                    "Lifecycle Stage": m.get("lifecycle_stage", "Mature Steady-State"),
                    "Seasonal Operational Context": m.get("seasonal_context", m.get("context", "Standard Seasonal Rhythm")),
                })
            with st.expander("📅 View Detailed 12-Month Daily Targets & Lifecycle Breakdown", expanded=False):
                st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 7. Prediction Methodology & Strategic Growth Levers
    # -------------------------------------------------------------
    st.divider()
    st.subheader("🔬 Prediction Methodology & Volume Growth Levers")
    st.caption(
        "End-to-end mathematical breakdown of how this forecast is calculated, and actionable operational levers to accelerate collection volumes."
    )

    # Stage 1: Scale & Demographics
    n_sqm = float(res.get("inputs", {}).get("sales_area_sqm", 850.0))
    n_casse = float(res.get("inputs", {}).get("cash_registers", 5.0))
    scale_applied = res.get("inputs", {}).get("store_scale_applied", "Nielsen Census Benchmark")
    if "AUTO:" in str(scale_applied):
        scale_src_label = "Auto Nielsen Match"
    elif "USER_CUSTOM" in str(scale_applied):
        scale_src_label = "User Custom Scale"
    elif "PRESET" in str(scale_applied):
        scale_src_label = "Format Archetype"
    else:
        scale_src_label = "National Median"

    format_tag = "LIS (Proximity)" if n_sqm < 400 else ("SUP (Supermarket)" if n_sqm < 2500 else "IPR (Hypermarket)")
    comune_val = geo.get("resolved_comune", "N/A")
    prov_val = geo.get("resolved_provincia", "")
    comune_full = f"{comune_val} ({prov_val})" if prov_val else str(comune_val)
    pop_val = int(round(float(geo.get("population", 0))))
    dens_val = float(geo.get("density", 0.0))
    nielsen_area_val = str(ret.get("nielsen_geographic_area", "N/A"))
    loc_info_state = st.session_state.get("loc_info") or {}
    pil_val = float(loc_info_state.get("PIL_per_capita", 0.0) or 0.0)
    pil_str = f"€{pil_val:,.0f}" if pil_val > 0 else "National Avg"

    # Stage 2: Dual AI Ensemble & Brand Embeddings
    lgb_b = preds.get("lgb_daily_bottiglie")
    nn_b = preds.get("nn_daily_bottiglie")
    weights = preds.get("ensemble_weights", {})
    lgb_wt_pct = int(round(weights.get("lgb_weight", 0.90) * 100))
    nn_wt_pct = int(round(weights.get("nn_weight", 0.10) * 100))
    unconstrained_b = float(preds.get("expected_daily_bottiglie", bottiglie_day))
    unconstrained_p = float(preds.get("expected_daily_presenze", presenze_day))
    b_brand = ret.get("standardized_brand", store_name)
    b_group = ret.get("retail_buying_group", "Independent")
    brand_emb_status = "Active" if ret.get("brand_embedding_active", True) else "Baseline"
    lgb_b_str = f"{lgb_b:.1f}" if lgb_b is not None else f"{unconstrained_b:.1f}"
    nn_b_str = f"{nn_b:.1f}" if nn_b is not None else f"{unconstrained_b:.1f}"

    # Stage 3: Spatial Catchment & Network Dilution
    dist_km = float(net.get("nearest_coripet_km", 0.0))
    nearest_name = net.get("nearest_pf_name", "Nearest Coripet")
    c_within_3km = int(net.get("coripet_count_3km", 0))
    c_within_5km = int(net.get("coripet_count_5km", 0))
    net_pos = net.get("network_position", "EXPANSION_NODE")
    tot_grocery = int(comp.get("total_grocery_stores", 0))
    super_count = int(comp.get("supermarkets", 0))
    hyper_count = int(comp.get("hypermarkets", 0))
    real_factor = float(real.get("realization_factor", 1.0)) * 100.0
    dilution_pct = float(real.get("dilution_percentage", 0.0))
    realized_b = float(real.get("realized_daily_bottles", bottiglie_day))
    nearest_dist_str = f"{dist_km * 1000:.0f} m" if dist_km < 1.0 else f"{dist_km:.2f} km"

    # Stage 4: Promotional Multiplier & Operational Underwriting
    reward_name = rew.get("reward_type", "Standard Loyalty")
    mult_b = float(rew.get("multiplier_bottles", 1.0))
    mult_p = float(rew.get("multiplier_presenze", 1.0))
    m_type = res.get("inputs", {}).get("machine_type", machine_type)
    sat_thresh = float(prob.get("hardware_operational_fit", {}).get("daily_saturation_threshold", 1100.0))
    service_rhythm = prob.get("hardware_operational_fit", {}).get("recommended_service_rhythm", "Standard 1-2x/week")
    u_mode = real.get("underwriting_mode", "STANDARD_STEADY_STATE")
    u_mode_short = "Empirical Telemetry Anchored" if "EMPIRICAL" in u_mode else ("Demographic Risk-Gated" if "DEMOGRAPHIC" in u_mode else "Standard Steady-State")

    col_method, col_levers = st.columns(2)

    with col_method:
        st.html(
            f"""<div class="intel-section" style="background: #ffffff; color: #0f172a; border-radius: 12px; padding: 18px 20px; border: 1px solid #cbd5e1; border-left: 5px solid #1976d2; box-shadow: 0 4px 10px rgba(0,0,0,0.05); min-height: 720px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; gap: 10px; box-sizing: border-box; margin-bottom: 15px;">
<div>
    <h4 style="margin-top: 0; margin-bottom: 6px; color: #0d47a1; display: flex; align-items: center; justify-content: space-between; font-weight: 800; font-size: 1.05rem;">
        <span>🧠 How This Prediction Was Calculated</span>
        <span style="font-size: 0.88rem; background: #e0f2fe; color: #0369a1; padding: 3px 10px; border-radius: 14px; font-weight: 700;">{bottiglie_day:.1f} b/d</span>
    </h4>
    <p style="font-size: 0.82rem; color: #475569; line-height: 1.4; margin-bottom: 2px; font-weight: 500;">
        4-stage intelligence pipeline transforming geospatial census, machine learning ensemble, spatial cannibalization, and operational underwriting:
    </p>
</div>

<div style="padding: 10px 13px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0284c7; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #0369a1; font-size: 0.83rem; font-weight: 800;">1. Geospatial & Nielsen Retail Scale Driver</strong>
        <span style="font-size: 0.65rem; background: #e0f2fe; color: #0369a1; padding: 1px 6px; border-radius: 4px; font-weight: 700;">SCALE & DEMO</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📐 Footprint: <strong>{n_sqm:.0f} m²</strong> &bull; <strong>{n_casse:.0f} casse</strong> ({format_tag})</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🏷️ Scale Source: <strong>{scale_src_label}</strong></span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📍 Catchment: <strong>{comune_full}</strong> &bull; <strong>{pop_val:,} pop</strong> &bull; <strong>{dens_val:.1f} ab/km²</strong></span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">💶 Purchasing Power: <strong>Area {nielsen_area_val}</strong> &bull; <strong>{pil_str}</strong></span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        Resolves store physical scale and municipal purchasing power as foundational baseline capacity.
    </p>
</div>

<div style="padding: 10px 13px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #7c3aed; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #6d28d9; font-size: 0.83rem; font-weight: 800;">2. Dual AI Ensemble & Brand Embeddings</strong>
        <span style="font-size: 0.65rem; background: #ede9fe; color: #6d28d9; padding: 1px 6px; border-radius: 4px; font-weight: 700;">AI ENSEMBLE</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🤖 LightGBM: <strong>{lgb_b_str} b/d</strong> ({lgb_wt_pct}% wt)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🧠 NeuralNet: <strong>{nn_b_str} b/d</strong> ({nn_wt_pct}% wt)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📈 Unconstrained Gross: <strong>{unconstrained_b:.1f} b/d</strong> (~{unconstrained_p:.1f} visits/d)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🛒 Brand Equity: <strong>{b_brand}</strong> (Group: <em>{b_group}</em>, Emb: {brand_emb_status})</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        Blends gradient boosted trees with multi-task tabular entity embeddings across an 84-day seasonal simulation grid.
    </p>
</div>

<div style="padding: 10px 13px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #d97706; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #b45309; font-size: 0.83rem; font-weight: 800;">3. Spatial Catchment & Network Dilution</strong>
        <span style="font-size: 0.65rem; background: #fef3c7; color: #92400e; padding: 1px 6px; border-radius: 4px; font-weight: 700;">SPATIAL MAPPING</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📡 Nearest Coripet: <strong>{nearest_dist_str}</strong> (<em>{nearest_name}</em>)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📍 Network Density: <strong>{c_within_3km} in 3 km</strong> ({c_within_5km} in 5 km &bull; <em>{net_pos}</em>)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🏬 Grocery Trade Area: <strong>{tot_grocery} stores</strong> ({super_count} super, {hyper_count} hyper)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📉 Net Realized: <strong>{realized_b:.1f} b/d</strong> (Haircut: <strong>-{dilution_pct:.1f}%</strong> &bull; {real_factor:.1f}% yield)</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        Quantifies cannibalization discount from nearby Coripet units and trade area grocery dispersion.
    </p>
</div>

<div style="padding: 10px 13px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #15803d; font-size: 0.83rem; font-weight: 800;">4. Promotional Multiplier & Operational Underwriting</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">UNDERWRITING</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🎟️ Commercial Policy: <strong>{reward_name}</strong> ({mult_b:.2f}x bottles, {mult_p:.2f}x visits)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">⚙️ Hardware: <strong>{m_type}</strong> (Sat: <strong>{sat_thresh:,.0f} b/d</strong> &bull; <em>{service_rhythm}</em>)</span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🛡️ Underwriting Mode: <strong>{u_mode_short}</strong></span>
        <span style="background: #ffffff; border: 1px solid #cbd5e1; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🔒 Locked Forecast: <strong>{bottiglie_day:.1f} b/d</strong> (~<strong>{bottiglie_month:,.0f} b/mo</strong> &bull; <strong>{int(bottiglie_day * 365):,} b/yr</strong>)</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        Underwrites contract incentive lift against hardware physical limits to lock final operational baseline.
    </p>
</div>
</div>"""
        )

    # ------------------------------------------------------------------
    # Dynamic Growth Levers Logic (Adapts to specific location context)
    # ------------------------------------------------------------------
    # Lever 1: Promotional Policy Strategy
    has_high_incentive = mult_b >= 1.25 or "voucher" in reward_name.lower() or "buono" in reward_name.lower()
    if has_high_incentive:
        l1_title = "1. Milestone Gamification & Weekend Multipliers"
        l1_tag = "+15% TO +25% VELOCITY"
        l1_badge1 = f"🎟️ Active Policy: <strong>{reward_name}</strong> ({mult_b:.2f}x lift active)"
        l1_badge2 = "📊 Velocity Boost: <strong>2x points on Sat/Sun</strong> or milestone €5 voucher"
        l1_desc = f"With {reward_name} already underwriting strong baseline demand, escalate repeat visits with weekend deposit challenges and monthly loyalty tier milestones."
    else:
        l1_title = "1. High-Incentive Checkout Vouchers"
        l1_tag = "+25% TO +45% YIELD"
        l1_badge1 = f"🎟️ Current Policy: <strong>{reward_name}</strong> (Baseline {mult_b:.2f}x)"
        l1_badge2 = "📊 Recommended: <strong>Direct Grocery Voucher</strong> (e.g. €3 off €25)"
        l1_desc = "Upgrading from baseline loyalty to direct checkout cash discounts drives shoppers to accumulate bottles at home and return on every regular shopping run."

    # Lever 2: Hardware Capacity & Logistics Management
    hw_p_fit = prob.get("hardware_operational_fit", {})
    p_over_val = float(hw_p_fit.get("p_daily_overflow_risk", 0.0))
    p_under_val = float(hw_p_fit.get("p_underutilization_risk", 0.0))

    if p_over_val >= 12.0 or bottiglie_day > (sat_thresh * 0.70):
        l2_title = "2. Prevent Weekend Saturation Lockouts"
        l2_tag = "+15% TO +25% RECAPTURED"
        l2_badge1 = f"🛑 Saturation Risk: <strong>{p_over_val:.1f}% Daily Overflow Probability</strong>"
        l2_badge2 = f"⚙️ Protocol: <strong>Pre-Saturday Bag-Swap</strong> (Threshold: {sat_thresh:,.0f} b/d)"
        l2_desc = f"Peak throughput threatens to fill the {m_type} compactor bin on peak days. Automated telemetry alerts and Friday evening bag resets prevent Saturday lockout downtime."
    elif p_under_val >= 35.0 or bottiglie_day < 300.0:
        l2_title = "2. Cross-Merchandising & Cashier Prompts"
        l2_tag = "+20% TO +35% ADOPTION"
        l2_badge1 = f"📉 Current Intake: <strong>{bottiglie_day:.0f} b/d vs {sat_thresh:,.0f} b/d capacity</strong>"
        l2_badge2 = "🛒 Trigger: <strong>Cashier receipt callouts</strong> & water-aisle shelf talkers"
        l2_desc = f"The {m_type} unit has abundant spare capacity. Educating shoppers in high-volume beverage aisles and at cash registers rapidly converts footfall into regular recyclers."
    else:
        l2_title = "2. High-Availability IoT Telemetry Monitoring"
        l2_tag = "+10% TO +15% UPTIME"
        l2_badge1 = f"⚙️ Hardware Class: <strong>{m_type}</strong> ({service_rhythm})"
        l2_badge2 = "📶 Telemetry: <strong>99%+ Service Level Agreement</strong>"
        l2_desc = "Proactive sensor monitoring and remote diagnostics maintain near-100% operational availability, reinforcing permanent shopper recycling habits."

    # Lever 3: Spatial Catchment & Cannibalization Defense
    if c_within_3km >= 2 or dist_km < 1.5 or dilution_pct >= 10.0:
        l3_title = f"3. Defend Local Catchment ({nearest_dist_str} to Competitor)"
        l3_tag = "+15% TO +30% RETENTION"
        l3_badge1 = f"📡 Local Density: <strong>{c_within_3km} Coripet units in 3 km</strong> (-{dilution_pct:.1f}% dilution)"
        l3_badge2 = "🔒 Defense Strategy: <strong>Store-exclusive app rewards</strong>"
        l3_desc = f"With {nearest_name} only {nearest_dist_str} away, lock in local households with store-exclusive checkout coupons that cannot be redeemed at competing locations."
    elif c_within_3km == 0 and dist_km >= 3.0:
        l3_title = "3. Regional Anchor Monopoly Expansion"
        l3_tag = "+25% TO +40% FOOTFALL"
        l3_badge1 = f"📍 Nearest Unit: <strong>{nearest_dist_str} away</strong> (Zero local competition)"
        l3_badge2 = f"📢 Radius Catchment: <strong>{pop_val:,} residents</strong> in {comune_val}"
        l3_desc = f"No competing Coripet machines exist within 3 km. Position this store as the primary recycling hub for the entire municipality with targeted geo-marketing."
    else:
        l3_title = "3. Trade-Area Grocery Footfall Interception"
        l3_tag = "+15% TO +25% CONVERSION"
        l3_badge1 = f"🏬 Grocery Cluster: <strong>{tot_grocery} stores in trade area</strong>"
        l3_badge2 = "🎯 Objective: <strong>Convert external grocery traffic</strong>"
        l3_desc = f"Leverage the surrounding cluster of {tot_grocery} grocery retailers by promoting Coripet recycling as an exclusive anchor reason to visit this specific store."

    # Lever 4: Store Format & Spatial Wayfinding
    if n_sqm >= 2500 or "IPR" in format_tag:
        l4_title = "4. Parking Lot & Lobby Wayfinding Directionals"
        l4_tag = "+20% TO +35% ACCESSIBILITY"
        l4_badge1 = f"📐 Scale: <strong>{n_sqm:.0f} m² Hypermarket</strong> ({n_casse:.0f} casse)"
        l4_badge2 = "🚗 Flow: <strong>Direct trunk-to-foyer cart signage</strong>"
        l4_desc = "For large hypermarkets, clear parking lot totems and lobby placement enable shoppers to wheel bulky multi-bag bottle loads effortlessly before entering the retail floor."
    elif n_sqm < 600 or "LIS" in format_tag:
        l4_title = "4. Street-Window Visibility & Express Fast-Drop"
        l4_tag = "+25% TO +40% COMMUTER FLOW"
        l4_badge1 = f"🚶 Density: <strong>{dens_val:.0f} ab/km² urban pedestrian core</strong>"
        l4_badge2 = "⚡ Speed: <strong>Front entrance express drop-off</strong>"
        l4_desc = "In compact urban footprints, high street-window visibility captures pedestrian commuters on daily foot routines, turning proximity into frequent micro-deposits."
    else:
        l4_title = "4. Prime Pre-Turnstile Entrance Placement"
        l4_tag = "+30% TO +50% FOOTFALL CAPTURE"
        l4_badge1 = f"📐 Format: <strong>{n_sqm:.0f} m² Supermarket</strong>"
        l4_badge2 = "🛒 Timing: <strong>Pre-shopping cart interception</strong>"
        l4_desc = "Positioning the RVM immediately before shopping carts ensures customers deposit containers upon arrival, receive their discount ticket, and redeem it inside immediately."

    with col_levers:
        st.html(
            f"""<div class="intel-section" style="background: #ffffff; color: #0f172a; border-radius: 12px; padding: 18px 20px; border: 1px solid #cbd5e1; border-left: 5px solid #2e7d32; box-shadow: 0 4px 10px rgba(0,0,0,0.05); min-height: 720px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; gap: 10px; box-sizing: border-box; margin-bottom: 15px;">
<div>
    <h4 style="margin-top: 0; margin-bottom: 6px; color: #1b5e20; display: flex; align-items: center; justify-content: space-between; font-weight: 800; font-size: 1.05rem;">
        <span>🚀 How to Increase This Prediction Number</span>
        <span style="font-size: 0.88rem; background: #dcfce7; color: #15803d; padding: 3px 10px; border-radius: 14px; font-weight: 700;">P90: {p90_d:.1f} b/d</span>
    </h4>
    <p style="font-size: 0.82rem; color: #475569; line-height: 1.4; margin-bottom: 2px; font-weight: 500;">
        Actionable operational and commercial levers to shift actual store throughput toward the Upside Ceiling:
    </p>
</div>

<div style="padding: 10px 13px; background: #fcfdfc; border: 1px solid #c8e6c9; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">{l1_title}</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">{l1_tag}</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l1_badge1}</span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l1_badge2}</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        {l1_desc}
    </p>
</div>

<div style="padding: 10px 13px; background: #fcfdfc; border: 1px solid #c8e6c9; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">{l2_title}</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">{l2_tag}</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l2_badge1}</span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l2_badge2}</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        {l2_desc}
    </p>
</div>

<div style="padding: 10px 13px; background: #fcfdfc; border: 1px solid #c8e6c9; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">{l3_title}</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">{l3_tag}</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l3_badge1}</span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l3_badge2}</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        {l3_desc}
    </p>
</div>

<div style="padding: 10px 13px; background: #fcfdfc; border: 1px solid #c8e6c9; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">{l4_title}</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">{l4_tag}</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l4_badge1}</span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">{l4_badge2}</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        {l4_desc}
    </p>
</div>
</div>"""
        )


if __name__ == "__main__":
    main()
