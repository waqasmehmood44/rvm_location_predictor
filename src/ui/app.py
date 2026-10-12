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
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
        margin-top: 8px;
        margin-bottom: 14px;
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
    </style>
    """,
    unsafe_allow_html=True,
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


@st.cache_data
def load_historical_machines():
    """Loads and enriches historical machine records for SPL and POT statuses from CaricaViewStoricoPuntoFisico.csv."""
    path = Path("data/raw/CaricaViewStoricoPuntoFisico.csv")
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(path, sep=";", dtype=str)

    # Fast enrichment of missing attributes across historical records for the same PuntoFisico
    enrich_cols = ["Descrizione", "TipoMacchina", "NomeFiliera", "Filiera", "Indirizzo", "Cap", "Comune", "Provincia", "Regione"]
    for c in enrich_cols:
        valid = df[df[c].notna() & (df[c].str.strip() != "") & (df[c].str.strip().str.lower() != "nan")]
        if not valid.empty:
            c_map = valid.drop_duplicates("PuntoFisico", keep="last").set_index("PuntoFisico")[c].to_dict()
            df[c] = df[c].fillna(df["PuntoFisico"].map(c_map))

    df["Stato_Clean"] = df["Stato"].astype(str).str.upper().str.strip()

    valid_machine_options = [
        "Ripet_C-front", "DAINT_G", "DAINT_A-bis", "Ripet_C-Bis", "Ripet_V2",
        "Ripet_C-Ter", "DAINT_B-bis", "RVM_Mod.E_Nuova", "Ripet_Smarpet",
        "RVM_Mod.E", "Ripet_V1", "SUPERMARPET01S"
    ]

    def process_status(status_str: str) -> pd.DataFrame:
        sub = df[df["Stato_Clean"] == status_str].copy()
        if sub.empty:
            return pd.DataFrame()

        # Deduplicate to keep the latest physical record per PuntoFisico
        sub = sub.drop_duplicates(subset=["PuntoFisico"], keep="last").copy()

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
            m_str = str(m).strip()
            if m_str in valid_machine_options:
                return m_str
            for vm in valid_machine_options:
                if vm.lower() == m_str.lower():
                    return vm
            return "Ripet_C-front"

        sub["ResolvedMachine"] = sub["TipoMacchina"].apply(resolve_mach)
        return sub

    df_spl = process_status("SPL")
    df_pot = process_status("POT")
    return df_spl, df_pot


def render_machine_directory(df_machines: pd.DataFrame, status_label: str, predictor: Any):
    """Renders a responsive directory table/cards of machines for SPL or POT status."""
    if df_machines.empty:
        st.info(f"No machines found with status {status_label}.")
        return

    total_count = len(df_machines)
    unique_comuni = df_machines["Comune"].nunique()
    unique_regioni = df_machines["Regione"].nunique()
    unique_filiere = df_machines["ResolvedFiliera"].nunique()

    # Metrics header
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%); border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h3 style="margin: 0; color: #1e293b; font-size: 1.25rem;">
                        📋 {status_label} Machines Registry ({total_count:,} Installed / Prospective Sites)
                    </h3>
                    <p style="margin: 4px 0 0 0; color: #64748b; font-size: 0.85rem;">
                        Verified historical physical network locations from CaricaViewStoricoPuntoFisico with status <strong>{status_label}</strong>.
                    </p>
                </div>
                <div style="display: flex; gap: 10px;">
                    <span style="background: #e0f2fe; color: #0369a1; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 0.82rem;">
                        🏙️ {unique_comuni} Municipalities
                    </span>
                    <span style="background: #f0fdf4; color: #15803d; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 0.82rem;">
                        🗺️ {unique_regioni} Regions
                    </span>
                    <span style="background: #faf5ff; color: #7e22ce; padding: 5px 12px; border-radius: 8px; font-weight: 700; font-size: 0.82rem;">
                        🏪 {unique_filiere} Commercial Partners
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search & filters bar
    c_search, c_reg, c_mach = st.columns([2.5, 1.2, 1.2])
    with c_search:
        search_query = st.text_input(
            f"🔍 Search {status_label} Machines",
            placeholder="Search by store name, ID (e.g. PF...), address, or city...",
            key=f"search_{status_label}",
        ).strip().lower()

    all_regions = ["All Regions"] + sorted([r for r in df_machines["Regione"].unique() if r and r != "nan"])
    with c_reg:
        selected_region = st.selectbox(
            "Region Filter",
            options=all_regions,
            index=0,
            key=f"region_filter_{status_label}",
        )

    all_models = ["All Machine Models"] + sorted([m for m in df_machines["ResolvedMachine"].unique() if m])
    with c_mach:
        selected_model = st.selectbox(
            "Hardware Model",
            options=all_models,
            index=0,
            key=f"model_filter_{status_label}",
        )

    # Apply filtering
    filtered_df = df_machines.copy()
    if search_query:
        mask = (
            filtered_df["PuntoFisico"].str.lower().str.contains(search_query, na=False)
            | filtered_df["ResolvedStoreName"].str.lower().str.contains(search_query, na=False)
            | filtered_df["Indirizzo"].str.lower().str.contains(search_query, na=False)
            | filtered_df["Comune"].str.lower().str.contains(search_query, na=False)
            | filtered_df["Cap"].str.lower().str.contains(search_query, na=False)
            | filtered_df["ResolvedFiliera"].str.lower().str.contains(search_query, na=False)
        )
        filtered_df = filtered_df[mask]

    if selected_region != "All Regions":
        filtered_df = filtered_df[filtered_df["Regione"] == selected_region]

    if selected_model != "All Machine Models":
        filtered_df = filtered_df[filtered_df["ResolvedMachine"] == selected_model]

    n_filtered = len(filtered_df)
    if n_filtered == 0:
        st.warning("⚠️ No machines matched the search / filter criteria.")
        return

    # Pagination setup
    PER_PAGE = 20
    total_pages = max(1, (n_filtered + PER_PAGE - 1) // PER_PAGE)

    col_stat, col_page = st.columns([2, 1])
    with col_stat:
        st.caption(f"Showing **{n_filtered:,}** of {total_count:,} machines | Click **Results 📊** to analyze prospective AI yield")
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

    # Table Header
    th_col1, th_col2, th_col3, th_col4 = st.columns([3.5, 3.0, 1.8, 1.3])
    with th_col1:
        st.markdown("<strong style='font-size: 0.82rem; color: #475569;'>STORE & NETWORK ID</strong>", unsafe_allow_html=True)
    with th_col2:
        st.markdown("<strong style='font-size: 0.82rem; color: #475569;'>LOCATION & ADDRESS</strong>", unsafe_allow_html=True)
    with th_col3:
        st.markdown("<strong style='font-size: 0.82rem; color: #475569;'>HARDWARE MODEL</strong>", unsafe_allow_html=True)
    with th_col4:
        st.markdown("<strong style='font-size: 0.82rem; color: #475569;'>AI PREDICTION</strong>", unsafe_allow_html=True)

    st.markdown("<div style='border-bottom: 2px solid #e2e8f0; margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    # Machine Items List
    for row_idx, row in page_df.iterrows():
        pf_code = row["PuntoFisico"]
        store = row["ResolvedStoreName"]
        addr = row["Indirizzo"]
        cap_val = row["Cap"]
        comune_val = row["Comune"]
        prov_val = row["Provincia"]
        reg_val = row["Regione"]
        mach_model = row["ResolvedMachine"]
        filiera_val = row["ResolvedFiliera"]

        with st.container():
            c1, c2, c3, c4 = st.columns([3.5, 3.0, 1.8, 1.3])
            with c1:
                st.markdown(
                    f"""
                    <div style="line-height: 1.35; margin-bottom: 4px;">
                        <span style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">{store}</span>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                        <span style="background: #f1f5f9; color: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-family: monospace; font-weight: 600;">🆔 {pf_code}</span>
                        <span style="background: #e0f2fe; color: #0369a1; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">🏢 {filiera_val}</span>
                        <span style="background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 700;">{status_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                prov_str = f"({prov_val})" if prov_val else ""
                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; color: #334155; line-height: 1.35;">
                        <div>📍 {addr}</div>
                        <div style="color: #64748b; font-size: 0.78rem;">📮 {cap_val} <strong>{comune_val}</strong> {prov_str}, {reg_val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; height: 100%;">
                        <span style="background: #f8fafc; border: 1px solid #cbd5e1; color: #1e293b; padding: 4px 8px; border-radius: 6px; font-size: 0.78rem; font-weight: 600;">
                            ⚙️ {mach_model}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c4:
                btn_key = f"btn_res_{status_label}_{pf_code}_{row_idx}"
                if st.button("Results 📊", key=btn_key, type="primary", use_container_width=True):
                    with st.spinner(f"Predicting yield for {store}..."):
                        try:
                            clean_addr_val = addr if addr and addr.lower() != "nan" else "Piazza del Comune 1"
                            st.session_state["pending_load_machine"] = {
                                "store_name": store,
                                "address": clean_addr_val,
                                "cap": cap_val,
                                "comune": comune_val,
                                "machine_type": mach_model,
                                "pf_code": pf_code,
                                "status_label": status_label,
                            }
                            st.session_state.pop("top_sim_machine", None)
                            st.session_state.pop("top_sim_reward", None)
                            st.session_state["pending_nav_tab"] = "🎯 Prospective Predictor"

                            import inspect
                            predict_kwargs = {
                                "store_name": store,
                                "address": clean_addr_val,
                                "cap": cap_val,
                                "machine_type": mach_model,
                                "comune": comune_val if comune_val else None,
                                "enable_rampup": True,
                            }
                            sig_params = inspect.signature(predictor.predict_new_location).parameters
                            kwargs_to_pass = {k: v for k, v in predict_kwargs.items() if k in sig_params}
                            res = predictor.predict_new_location(**kwargs_to_pass)
                            st.session_state["prediction_result"] = res
                            st.rerun()
                        except Exception as err:
                            st.error(f"❌ Error predicting location: {err}")

            st.markdown("<div style='border-bottom: 1px solid #f1f5f9; margin: 8px 0 10px 0;'></div>", unsafe_allow_html=True)


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
        st.session_state["wizard_step"] = 4
        try:
            st.session_state["loc_info"] = predictor.resolve_location(
                load_data["address"],
                load_data["cap"],
                comune=load_data["comune"] if load_data["comune"] else None,
            )
        except Exception:
            st.session_state["loc_info"] = None

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
        st.session_state["input_store_name"] = "Esselunga Milano Certosa"
    if "input_address" not in st.session_state:
        st.session_state["input_address"] = "Via delle Forze Armate 12"
    if "input_cap" not in st.session_state:
        st.session_state["input_cap"] = "20147"
    if "input_comune" not in st.session_state:
        st.session_state["input_comune"] = "Milano"
    if "input_machine_type" not in st.session_state:
        st.session_state["input_machine_type"] = "Ripet_C-front"
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
                st.session_state["input_address"] = ""
                st.session_state["input_cap"] = ""
                st.session_state["input_comune"] = ""
                st.session_state["input_machine_type"] = "Ripet_C-front"
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

        store_name = st.text_input(
            "Store Name / Insegna",
            key="input_store_name",
            placeholder="e.g., Esselunga, Conad Superstore, Coop.fi, Carrefour Market",
            help="e.g., Esselunga, Conad Superstore, Coop.fi, Carrefour Market",
        )
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

        machine_options = [
            "Ripet_C-front",
            "DAINT_G",
            "DAINT_A-bis",
            "Ripet_C-Bis",
            "Ripet_V2",
            "Ripet_C-Ter",
            "DAINT_B-bis",
            "RVM_Mod.E_Nuova",
            "Ripet_Smarpet",
            "RVM_Mod.E",
            "Ripet_V1",
            "SUPERMARPET01S",
        ]

        if "pending_input_machine_type" in st.session_state:
            st.session_state["input_machine_type"] = st.session_state.pop("pending_input_machine_type")

        if st.session_state.get("input_machine_type") not in machine_options:
            st.session_state["input_machine_type"] = machine_options[0]

        machine_type = st.selectbox(
            "RVM Machine Model",
            options=machine_options,
            key="input_machine_type",
            help="Target Coripet RVM hardware installation type",
        )

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
            
            scale_desc = (
                f"• **Format:** {scale.get('label', scale.get('tipo_negozio'))}\n\n"
                f"• **Sales Area:** ~{scale.get('superficie_vendita', 0):.0f} m² | **Checkouts:** {scale.get('numero_casse', 0):.0f}\n\n"
                f"• *{scale.get('details', '')}*"
            )
            st.info(f"🏢 **Nielsen Scale Auto-Detected:**\n\n{scale_desc}")
            
            st.markdown("💬 **This is what was detected from the Nielsen file. You can accept it (Recommended) or change it below:**")
            
            auto_label = f"Auto-Detect: Use Detected Scale (~{scale.get('superficie_vendita', 0):.0f} m², {scale.get('numero_casse', 0):.0f} checkouts)"
            format_options = [
                auto_label,
                "1: Small Supermarket / Proximity (~350 m², 2 checkouts)",
                "2: Standard Supermarket (~750 m², 4 checkouts)",
                "3: Superstore / Large Supermarket (~1,850 m², 9 checkouts)",
                "4: Hypermarket / Mall Anchor (~3,750 m², 21 checkouts)",
                "5: Discount Store (~750 m², 4 checkouts)",
                "Custom (manual sales area and checkouts)",
            ]
            format_choice = st.selectbox(
                "Store Format / Scale Profile",
                options=format_options,
                index=0,
                help="Accept auto-detected footprint or select custom archetype",
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
                with c_area:
                    chosen_sqm = st.number_input("Sales Area (m²)", min_value=100.0, max_value=15000.0, value=float(scale.get("superficie_vendita", 850.0)), step=50.0)
                with c_casse:
                    chosen_casse = st.number_input("Checkouts", min_value=1.0, max_value=60.0, value=float(scale.get("numero_casse", 5.0)), step=1.0)
                    
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
    NAV_TABS = [TAB_MAIN, TAB_SPL, TAB_POT]

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
                <div class="intel-section" style="border-left: 5px solid #2e7d32; padding: 24px;">
                    <h3 style="color: #1b5e20; margin-top: 0;">🧭 Guided Prospective Location Analysis</h3>
                    <p style="color: #444; font-size: 1.05rem;">
                        Welcome to the <strong>Coripet RVM Prospective Predictor</strong>. Follow the 3-step decision support wizard on the left sidebar, or select any installed machine from the <strong>SPL Machines</strong> or <strong>POT Machines</strong> tabs to run instant predictions:
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 18px;">
                        <div style="background: {'#e8f5e9' if curr_step == 1 else '#f7fafc'}; border: 1px solid {'#81c784' if curr_step == 1 else '#e2e8f0'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: #1b5e20;">1️⃣ Step 1: Site & Model</strong><br>
                            <span style="font-size: 0.85rem; color: #555;">Enter store name, street, CAP, and machine. Click <em>Detect Location & Nielsen Scale</em>.</span>
                        </div>
                        <div style="background: {'#e8f5e9' if curr_step == 2 else '#f7fafc'}; border: 1px solid {'#81c784' if curr_step == 2 else '#e2e8f0'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: #1b5e20;">2️⃣ Step 2: Nielsen Census</strong><br>
                            <span style="font-size: 0.85rem; color: #555;">Review auto-detected store scale and footprint. Accept or adjust, then click <em>Continue</em>.</span>
                        </div>
                        <div style="background: {'#e8f5e9' if curr_step == 3 else '#f7fafc'}; border: 1px solid {'#81c784' if curr_step == 3 else '#e2e8f0'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: #1b5e20;">3️⃣ Step 3: Coripet Rewards</strong><br>
                            <span style="font-size: 0.85rem; color: #555;">Review contract promotions from the database. Keep or change, then click <em>Run Prediction</em>.</span>
                        </div>
                        <div style="background: {'#e8f5e9' if curr_step == 4 else '#f7fafc'}; border: 1px solid {'#81c784' if curr_step == 4 else '#e2e8f0'}; border-radius: 8px; padding: 14px;">
                            <strong style="color: #1b5e20;">4️⃣ Step 4: Executive Verdict</strong><br>
                            <span style="font-size: 0.85rem; color: #555;">View 365-day yield, 12-month calendar, clearance gates, and investment score.</span>
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
        df_spl, _ = load_historical_machines()
        render_machine_directory(df_spl, "SPL", predictor)

    elif active_tab == TAB_POT:
        _, df_pot = load_historical_machines()
        render_machine_directory(df_pot, "POT", predictor)


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
    machine_options = [
        "Ripet_C-front",
        "DAINT_G",
        "DAINT_A-bis",
        "Ripet_C-Bis",
        "Ripet_V2",
        "Ripet_C-Ter",
        "DAINT_B-bis",
        "RVM_Mod.E_Nuova",
        "Ripet_Smarpet",
        "RVM_Mod.E",
        "Ripet_V1",
        "SUPERMARPET01S",
    ]

    res_mach = res.get("inputs", {}).get("machine_type", machine_options[0])
    default_mach_idx = machine_options.index(res_mach) if res_mach in machine_options else 0

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
        top_mach = st.selectbox(
            "⚙️ Hardware Machine Model",
            options=machine_options,
            index=default_mach_idx,
            key="top_sim_machine",
            help="Directly switch the RVM machine model. Yield, hardware capacity, and clearance gate update immediately.",
        )
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

    if machine_changed or reward_changed:
        with st.spinner(f"🔄 Recalculating prospective yield for {top_mach} with {selected_rew_obj['scheme_name']}..."):
            try:
                inputs = res.get("inputs", {})
                comune_resolved = res.get("location_attributes", {}).get("resolved_comune")
                clean_comune = None
                if comune and str(comune).strip():
                    clean_comune = str(comune).strip()
                elif comune_resolved and str(comune_resolved).strip():
                    clean_comune = str(comune_resolved).strip()
                elif inputs.get("comune_input") and inputs.get("comune_input") != "AUTO_DETECTED":
                    clean_comune = str(inputs.get("comune_input")).strip()

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

    inv = res.get("investment_assessment", {})
    suit = res.get("location_suitability", {})
    real = res.get("realized_forecast", {})
    preds = res.get("predictions", {})
    rew = res.get("commercial_rewards", {})
    spatial = res.get("spatial_catchment", {})
    geo = res.get("location_attributes", {})
    ret = res.get("retail_intelligence", {})
    prob = res.get("probability_analysis", {})

    presenze_day = float(real.get("underwritten_daily_presenze", real.get("realized_daily_presenze", preds.get("expected_daily_presenze", 0.0))))
    bottiglie_day = float(real.get("underwritten_daily_bottles", real.get("realized_daily_bottles", preds.get("expected_daily_bottiglie", 0.0))))
    bottiglie_month = float(real.get("underwritten_monthly_bottles", preds.get("expected_monthly_bottiglie", bottiglie_day * 30.4)))
    bottles_per_person = round(bottiglie_day / max(1.0, presenze_day), 1)
    suitability_score = float(suit.get("suitability_score", 0.0))

    rec_obj = inv.get("recommendation", {})
    rec_tier_name = rec_obj.get("tier_name", "TIER 2 - VIABLE STANDARD EXPANSION")
    decision = rec_obj.get("decision", "COMMERCIALLY VIABLE (PROCEED)")

    if suitability_score >= 75 or bottiglie_day >= 750.0:
        recommendation = "HIGH POTENTIAL"
    elif suitability_score >= 50 or bottiglie_day >= 448.0:
        recommendation = "MEDIUM POTENTIAL"
    else:
        recommendation = "LOW POTENTIAL"

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
    ann_med_base = round(preds.get("expected_daily_bottiglie_median", 0.0) * real.get("realization_factor", 1.0), 1)
    underwrite_scale = (under_b_val / ann_med_base) if (is_risk_adjusted and ann_med_base > 0 and under_b_val) else 1.0

    p10_d = round(prob.get('downside_floor_p10_daily', 0.0) * underwrite_scale, 1) if (is_risk_adjusted and underwrite_scale < 1.0) else prob.get('downside_floor_p10_daily', 0.0)
    p10_m = int(round(p10_d * 30.4))
    p90_d = round(prob.get('upside_peak_p90_daily', 0.0) * underwrite_scale, 1) if (is_risk_adjusted and underwrite_scale < 1.0) else prob.get('upside_peak_p90_daily', 0.0)
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
        if str(area_raw).isdigit():
            nielsen_val = f"Area {area_raw}"
        elif "Area" in str(area_raw):
            nielsen_val = str(area_raw)
        else:
            nielsen_val = f"Area {area_raw}"
        store_disp = str(res.get('inputs', {}).get('store_name', store_name)).strip()
        st.html(
            f"""<div class="exec-card exec-card-geo">
    <div class="exec-card-top">
        <span class="exec-card-lbl">📍 Target Location & Hardware</span>
        <span class="exec-cap-pill">CAP {cap}</span>
    </div>
    <div>
        <div class="exec-geo-title">{comune_val} <span class="exec-geo-reg">({prov_val}), {reg_val}</span></div>
        <div class="exec-conf-sub" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{store_disp}</div>
    </div>
    <div class="exec-card-bot" style="gap: 6px; justify-content: flex-start;">
        <span class="exec-chip">🗺️ {nielsen_val}</span>
        <span class="exec-chip">⚙️ {res.get('inputs', {}).get('machine_type', machine_type)}</span>
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
    # 3.5. Statistical Probability & Future Performance Profile
    # -------------------------------------------------------------
    if prob:
        st.divider()
        st.subheader("🎲 Statistical Probability & Future Performance Profile")
        st.caption("Parametric risk modeling, milestone attainment likelihoods, and hardware capacity utilization calibrated to network volatility.")

        # Row 1: Capital Safety & Value-at-Risk 80% Band
        cap_safety = prob.get("success_probability", 0.0)
        risk_rating = prob.get("risk_rating", "MODERATE RISK")
        
        if "LOW" in risk_rating:
            chip_bg = "#dcfce7"
            chip_color = "#166534"
            chip_border = "#86efac"
        elif "MODERATE" in risk_rating:
            chip_bg = "#fef3c7"
            chip_color = "#92400e"
            chip_border = "#fcd34d"
        else:
            chip_bg = "#fee2e2"
            chip_color = "#991b1b"
            chip_border = "#fca5a5"

        p_col1, p_col2, p_col3 = st.columns([1.0, 1.0, 1.0])
        with p_col1:
            st.html(
                f"""<div style="background: #ffffff; color: #0f172a; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0; border-top: 3px solid #16a34a; box-shadow: 0 2px 6px rgba(0,0,0,0.03); min-height: 114px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
    <div>
        <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 2px;">Capital Safety Probability</div>
        <div style="font-size: 1.25rem; font-weight: 800; color: #15803d; letter-spacing: -0.3px; line-height: 1.15;">{cap_safety:.1f}%</div>
    </div>
    <div style="margin: 4px 0;">
        <span style="display: inline-block; padding: 2px 7px; border-radius: 5px; font-size: 0.63rem; font-weight: 700; background: {chip_bg}; color: {chip_color}; border: 1px solid {chip_border}; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%;">{risk_rating}</span>
    </div>
    <div style="border-top: 1px solid #f1f5f9; padding-top: 4px; font-size: 0.70rem; color: #64748b; font-weight: 500;">
        P(Volume ≥ 120 b/d Breakeven)
    </div>
</div>"""
            )
        with p_col2:
            st.html(
                f"""<div style="background: #ffffff; color: #0f172a; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0; border-top: 3px solid #2563eb; box-shadow: 0 2px 6px rgba(0,0,0,0.03); min-height: 114px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
    <div>
        <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 2px;">Downside Floor (P10)</div>
        <div style="display: flex; align-items: baseline; gap: 4px;">
            <span style="font-size: 1.25rem; font-weight: 800; color: #1d4ed8; letter-spacing: -0.3px; line-height: 1.15;">{p10_d:.1f}</span>
            <span style="font-size: 0.75rem; font-weight: 600; color: #64748b;">b/d</span>
        </div>
    </div>
    <div style="margin: 2px 0; font-size: 0.76rem; font-weight: 700; color: #334155;">
        {p10_m:,} bottles / month
    </div>
    <div style="border-top: 1px solid #f1f5f9; padding-top: 4px; font-size: 0.70rem; color: #64748b; font-weight: 500;">
        90% Statistical Certainty
    </div>
</div>"""
            )
        with p_col3:
            st.html(
                f"""<div style="background: #ffffff; color: #0f172a; border-radius: 10px; padding: 12px 14px; border: 1px solid #e2e8f0; border-top: 3px solid #7c3aed; box-shadow: 0 2px 6px rgba(0,0,0,0.03); min-height: 114px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
    <div>
        <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #64748b; margin-bottom: 2px;">Upside Ceiling (P90)</div>
        <div style="display: flex; align-items: baseline; gap: 4px;">
            <span style="font-size: 1.25rem; font-weight: 800; color: #6d28d9; letter-spacing: -0.3px; line-height: 1.15;">{p90_d:.1f}</span>
            <span style="font-size: 0.75rem; font-weight: 600; color: #64748b;">b/d</span>
        </div>
    </div>
    <div style="margin: 2px 0; font-size: 0.76rem; font-weight: 700; color: #334155;">
        {p90_m:,} bottles / month
    </div>
    <div style="border-top: 1px solid #f1f5f9; padding-top: 4px; font-size: 0.70rem; color: #64748b; font-weight: 500;">
        Top 10% Peak Potential
    </div>
</div>"""
            )

        # Row 2: Future Performance Outlook (Tiers) & Cumulative Milestones
        c_tiers, c_milestones = st.columns([1.2, 1.2])

        tiers_p = prob.get("performance_tiers", {})
        with c_tiers:
            st.markdown("#### 🎯 Future Performance Outlook (Volume Tiers)")
            st.caption(f"Dominant Outlook: **{tiers_p.get('most_likely_tier', 'N/A')}**")
            
            p_low = tiers_p.get("p_low_under_200", 0.0)
            p_mod = tiers_p.get("p_moderate_200_400", 0.0)
            p_high = tiers_p.get("p_high_400_700", 0.0)
            p_star = tiers_p.get("p_star_over_700", 0.0)
            
            tier_items = [
                ("Low Yield / Review (< 200 b/d)", p_low, "#e53935", "Capital underperformance risk"),
                ("Moderate / Viable (200 - 400 b/d)", p_mod, "#1e88e5", "Standard network benchmark"),
                ("High Performer Hub (400 - 700 b/d)", p_high, "#43a047", "Top quartile retail asset"),
                ("Star / Strategic Anchor (> 700 b/d)", p_star, "#8e24aa", "Exceptional destination node"),
            ]
            
            for t_label, t_val, t_color, t_sub in tier_items:
                st.html(
                    f"""<div style="margin-bottom: 8px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px; height: 68px; min-height: 68px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">{t_label}</span>
        <span style="font-weight: 800; font-size: 0.88rem; color: {t_color};">{t_val:.1f}%</span>
    </div>
    <div style="width: 100%; background: #e2e8f0; border-radius: 6px; height: 6px; overflow: hidden; margin: 3px 0;">
        <div style="width: {min(100.0, max(0.0, t_val))}%; background: {t_color}; height: 6px; border-radius: 6px;"></div>
    </div>
    <div style="font-size: 0.70rem; color: #64748b; font-weight: 500;">{t_sub}</div>
</div>"""
                )

        milestones_p = prob.get("annual_milestones", {})
        with c_milestones:
            st.markdown("#### 🏆 Cumulative Year 1 Collection Milestones")
            st.caption("Probability of reaching annual collection volume targets operating 365 days.")
            
            m_50k = milestones_p.get("p_annual_50k", 0.0)
            m_100k = milestones_p.get("p_annual_100k", 0.0)
            m_150k = milestones_p.get("p_annual_150k", 0.0)
            m_200k = milestones_p.get("p_annual_200k", 0.0)
            
            milestone_items = [
                ("P(Year 1 ≥ 50,000 bottles)", m_50k, "#1e88e5", "Base Survival Volume (137 b/d)"),
                ("P(Year 1 ≥ 100,000 bottles)", m_100k, "#43a047", "Core Operational Milestone (274 b/d)"),
                ("P(Year 1 ≥ 150,000 bottles)", m_150k, "#fb8c00", "Strong Commercial Target (411 b/d)"),
                ("P(Year 1 ≥ 200,000 bottles)", m_200k, "#8e24aa", "Top Tier Asset Output (548 b/d)"),
            ]
            
            for m_label, m_val, m_color, m_sub in milestone_items:
                st.html(
                    f"""<div style="margin-bottom: 8px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px; height: 68px; min-height: 68px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">{m_label}</span>
        <span style="font-weight: 800; font-size: 0.88rem; color: {m_color};">{m_val:.1f}%</span>
    </div>
    <div style="width: 100%; background: #e2e8f0; border-radius: 6px; height: 6px; overflow: hidden; margin: 3px 0;">
        <div style="width: {min(100.0, max(0.0, m_val))}%; background: {m_color}; height: 6px; border-radius: 6px;"></div>
    </div>
    <div style="font-size: 0.70rem; color: #64748b; font-weight: 500;">{m_sub}</div>
</div>"""
                )

        # Row 3: Target Attainment Probabilities & Hardware Operational Fit
        c_attain, c_hw = st.columns([1.2, 1.2])

        with c_attain:
            st.markdown("#### 🎯 Commercial Target Attainment Hurdles")
            st.caption("Threshold probability of sustaining target daily volume collection rates.")
            
            p_120 = prob.get("p_breakeven_120", 0.0)
            p_300 = prob.get("p_standard_300", 0.0)
            p_500 = prob.get("p_high_yield_500", 0.0)
            p_1000 = prob.get("p_anchor_1000", 0.0)
            
            attain_items = [
                ("P(Volume ≥ 120 b/d - Breakeven)", p_120, "#2e7d32", "Capital Recovery Threshold"),
                ("P(Volume ≥ 300 b/d - Standard)", p_300, "#1976d2", "Commercial Network Viability"),
                ("P(Volume ≥ 500 b/d - High-Yield)", p_500, "#f57f17", "Top-Quartile Performance Hub"),
                ("P(Volume ≥ 1,000 b/d - Anchor)", p_1000, "#c2185b", "Hypermarket Capacity Demand"),
            ]
            
            for a_label, a_val, a_color, a_sub in attain_items:
                st.html(
                    f"""<div style="margin-bottom: 8px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px; height: 68px; min-height: 68px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">{a_label}</span>
        <span style="font-weight: 800; font-size: 0.88rem; color: {a_color};">{a_val:.1f}%</span>
    </div>
    <div style="width: 100%; background: #e2e8f0; border-radius: 6px; height: 6px; overflow: hidden; margin: 3px 0;">
        <div style="width: {min(100.0, max(0.0, a_val))}%; background: {a_color}; height: 6px; border-radius: 6px;"></div>
    </div>
    <div style="font-size: 0.70rem; color: #64748b; font-weight: 500;">{a_sub}</div>
</div>"""
                )

        hw_p = prob.get("hardware_operational_fit", {})
        with c_hw:
            st.markdown("#### ⚙️ Hardware Operational Fit & Capacity Utilization")
            st.caption(f"Evaluation of hardware model: **{hw_p.get('requested_machine', machine_type)}**")
            
            p_opt = hw_p.get("p_optimal_utilization", 0.0)
            p_over = hw_p.get("p_daily_overflow_risk", 0.0)
            p_under = hw_p.get("p_underutilization_risk", 0.0)
            sat_thresh = hw_p.get("daily_saturation_threshold", 0.0)
            rhythm = hw_p.get("recommended_service_rhythm", "Weekly Servicing")
            hw_class = hw_p.get("hardware_class", "Standard RVM")

            opt_w = max(0.0, min(100.0, p_opt))
            und_w = max(0.0, min(100.0 - opt_w, p_under))
            ovr_w = max(0.0, min(100.0 - opt_w - und_w, p_over))

            # Card 1: Hardware Model & Unit Class Specifications
            st.html(
                f"""<div style="margin-bottom: 8px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px; height: 68px; min-height: 68px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.68rem; color: #475569; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Installed Unit Class</span>
        <span style="background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; border-radius: 4px; padding: 1px 6px; font-size: 0.68rem; font-weight: 700;">{hw_p.get('requested_machine', machine_type)}</span>
    </div>
    <div style="font-weight: 800; font-size: 0.82rem; color: #0f172a; line-height: 1.2; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        {hw_class}
    </div>
    <div style="font-size: 0.70rem; color: #1976d2; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
        🚚 Recommended Logistics: {rhythm}
    </div>
</div>"""
            )

            hw_items = [
                ("Optimal Utilization Probability", p_opt, "#2e7d32", "Expected volume matches hardware intake capacity"),
                ("Underutilization Risk", p_under, "#f57f17", "Hardware capacity oversized relative to demand"),
                (f"Daily Saturation / Overflow Risk (>{sat_thresh:.0f} b/d)", p_over, "#c62828" if p_over > 5 else "#757575", "Peak collection volume exceeding physical compactor bin"),
            ]

            for h_label, h_val, h_color, h_sub in hw_items:
                st.html(
                    f"""<div style="margin-bottom: 8px; background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 12px; height: 68px; min-height: 68px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box; box-shadow: 0 2px 4px rgba(0,0,0,0.03);">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 700; font-size: 0.82rem; color: #0f172a;">{h_label}</span>
        <span style="font-weight: 800; font-size: 0.88rem; color: {h_color};">{h_val:.1f}%</span>
    </div>
    <div style="width: 100%; background: #e2e8f0; border-radius: 6px; height: 6px; overflow: hidden; margin: 3px 0;">
        <div style="width: {min(100.0, max(0.0, h_val))}%; background: {h_color}; height: 6px; border-radius: 6px;"></div>
    </div>
    <div style="font-size: 0.70rem; color: #64748b; font-weight: 500;">{h_sub}</div>
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
        r_badge = "badge-reward"
        st.html(
            f"""<div class="intel-section" style="border-left: 5px solid #1976d2; min-height: 220px; height: 100%; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
    <div>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <span style="font-weight: 700; font-size: 1.05rem; color: #0f172a;">{rew.get('filiera', 'Inferred Circuit')}</span>
            <span class="badge {r_badge}">{reward_type}</span>
        </div>
        <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.55;">
            <strong>Incentive Tier:</strong> {rew.get('incentive_tier', 'GROCERY_VOUCHER')}<br>
            <strong>Incentive Strength:</strong> {rew.get('incentive_strength', 'HIGH')}<br>
            <strong>Voucher Terms:</strong> {rew.get('typical_voucher', 'Store Voucher Standard')}<br>
            <strong>Italian Definition:</strong> <em>{rew.get('definition_it', '')}</em>
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
        
        table_rows = []
        for item in sens_data:
            tag = "⭐ SELECTED" if item.get("is_selected") else ""
            diff_b = item.get("diff_bottles", 0.0)
            pct_v = item.get("pct_variance", 0.0)
            var_label = "SELECTED POLICY" if item.get("is_selected") else f"{'+' if diff_b > 0 else ''}{pct_v:.1f}% ({'+' if diff_b > 0 else ''}{diff_b:.1f} b/d)"
            table_rows.append({
                "Status": tag,
                "Reward Policy Scheme": item["reward_type"],
                "Daily Bottles (b/d)": f"{item['daily_bottles']:.1f}",
                "Monthly Volume": f"{item['monthly_bottles']:,} b/mo",
                "Active Footfall": f"{item['daily_presenze']:.1f} u/d",
                "Variance vs Selected": var_label,
                "Incentive Tier": item.get("incentive_strength", "N/A"),
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # 5. Clearance Gates & Seasonality Projection
    # -------------------------------------------------------------
    gate_results = suit.get("gate_results", {})
    if gate_results:
        st.subheader("🛡️ Operational Clearance Gates")
        cards_html = []
        for g_name, g_info in gate_results.items():
            status = g_info.get("status", "PASS")
            g_label = g_info.get("label", g_name.replace("_", " ").title())
            g_metric = g_info.get("metric", "")
            g_flag = g_info.get("flag", status)
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
        cards_str = "\n".join(cards_html)
        st.html(f"""<div class="gate-grid">\n{cards_str}\n</div>""")

    # -------------------------------------------------------------
    # 6. Operational Rolling Targets & Multi-Year Lifecycle Horizon
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
        st.subheader("📈 12-Month Operational Target & Multi-Year Lifecycle Horizon")
        st.caption(
            f"Rolling 12-month operational adoption roadmap starting from **{l_month_str} {l_year_int}**, combined with 3-year commercial compounding lifecycle."
        )

        # 3 Multi-Year Executive Horizon Cards
        my_forecast = lifecycle_h.get("multi_year_forecast", [])
        if my_forecast and len(my_forecast) >= 3:
            y1 = my_forecast[0]
            y2 = my_forecast[1]
            y3 = my_forecast[2]

            c_y1, c_y2, c_y3 = st.columns(3)
            with c_y1:
                st.html(
                    f"""<div class="metric-card metric-card-blue" style="min-height: 145px;">
    <div class="metric-top">
        <span class="metric-title">{y1['horizon']}</span>
        <span class="metric-icon-badge badge-tint-blue">🌱</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-blue">{y1['annual_bottles']:,}</span>
        <span class="metric-unit">b/yr</span>
    </div>
    <div style="font-size: 0.80rem; font-weight: 700; color: #1565c0; margin: 3px 0;">
        Daily Average: <strong>{y1['daily_average']:.1f} b/d</strong> ({y1['capacity_utilization']} of Mature Capacity)
    </div>
    <div class="metric-subtitle">{y1['operational_focus']}</div>
</div>"""
                )
            with c_y2:
                st.html(
                    f"""<div class="metric-card metric-card-green" style="min-height: 145px;">
    <div class="metric-top">
        <span class="metric-title">{y2['horizon']}</span>
        <span class="metric-icon-badge badge-tint-green">⭐</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-green">{y2['annual_bottles']:,}</span>
        <span class="metric-unit">b/yr</span>
    </div>
    <div style="font-size: 0.80rem; font-weight: 700; color: #2e7d32; margin: 3px 0;">
        Daily Average: <strong>{y2['daily_average']:.1f} b/d</strong> (100% Mature Steady-State)
    </div>
    <div class="metric-subtitle">{y2['operational_focus']}</div>
</div>"""
                )
            with c_y3:
                st.html(
                    f"""<div class="metric-card metric-card-purple" style="min-height: 145px;">
    <div class="metric-top">
        <span class="metric-title">{y3['horizon']}</span>
        <span class="metric-icon-badge badge-tint-purple">🚀</span>
    </div>
    <div class="metric-num-wrap">
        <span class="metric-num num-purple">{y3['annual_bottles']:,}</span>
        <span class="metric-unit">b/yr</span>
    </div>
    <div style="font-size: 0.80rem; font-weight: 700; color: #6a1b9a; margin: 3px 0;">
        Daily Average: <strong>{y3['daily_average']:.1f} b/d</strong> (+5.5% YoY Coripet Network Growth)
    </div>
    <div class="metric-subtitle">{y3['operational_focus']}</div>
</div>"""
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
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">1. High-Incentive Checkout Vouchers</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">+25% TO +45% YIELD</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🎟️ Mechanism: <strong>Direct Grocery Voucher</strong> (e.g. €3 off min. €25 spend)</span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📊 Velocity Lift: <strong>+35% repeat deposit velocity</strong> vs generic points</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        Direct checkout discounts drive shoppers to accumulate bottles at home and return on every regular shopping run.
    </p>
</div>

<div style="padding: 10px 13px; background: #fcfdfc; border: 1px solid #c8e6c9; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">2. High-Visibility Entrance Placement</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">+30% TO +50% FOOTFALL</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📍 Prime Location: <strong>Store Main Entrance / Lobby</strong></span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📊 Discovery Lift: <strong>+40% footfall conversion</strong> vs exterior / rear lots</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        Placing the RVM directly along natural customer arrival flow captures shoppers before they begin grocery shopping.
    </p>
</div>

<div style="padding: 10px 13px; background: #fcfdfc; border: 1px solid #c8e6c9; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">3. Eliminate Bin-Full Lockouts</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">+15% TO +20% RECAPTURED</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">🛑 Downtime Cause: <strong>"Cassonetto Pieno" Saturday Peak Lockout</strong></span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📊 Recapture Lift: <strong>Recaptures 15-20% weekend volume</strong> via dual-feed bins</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        Proactive Friday/Saturday bag change protocols prevent machines halting during maximum shopper volume hours.
    </p>
</div>

<div style="padding: 10px 13px; background: #fcfdfc; border: 1px solid #c8e6c9; border-left: 4px solid #16a34a; border-radius: 8px; display: flex; flex-direction: column; gap: 6px; box-sizing: border-box;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="color: #1b5e20; font-size: 0.83rem; font-weight: 800;">4. In-Store Point-of-Sale Awareness</strong>
        <span style="font-size: 0.65rem; background: #dcfce7; color: #15803d; padding: 1px 6px; border-radius: 4px; font-weight: 700;">+10% TO +15% ADOPTION</span>
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 4px;">
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📢 Campaign: <strong>Checkout Flyers & Floor Graphic Wayfinding</strong></span>
        <span style="background: #ffffff; border: 1px solid #c8e6c9; padding: 2px 7px; border-radius: 5px; font-size: 0.73rem; color: #1e293b;">📊 Habit Lift: <strong>+12% household adoption</strong> within 60 days of launch</span>
    </div>
    <p style="font-size: 0.75rem; color: #475569; margin: 0; line-height: 1.35; font-weight: 500;">
        In-store promotional collateral habituates local families to bring empty PET bottles from home on each grocery visit.
    </p>
</div>
</div>"""
        )


if __name__ == "__main__":
    main()
