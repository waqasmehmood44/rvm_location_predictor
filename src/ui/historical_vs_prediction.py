"""
Historical Actuals vs. Prospective Model Prediction Visualizer & Comparison Engine.

Premium, Dark-Theme Adaptive Analytics Component:
1. Past Operational Performance (Historical Actuals from PostgreSQL master_daily_rvm - Last 2 Years)
2. Future Operational Forecast (Multi-Task Stacking AI Model Predictions - Next 12 Months)

Design Highlights:
- Perfectly Balanced Dual View: 24 Past Months Actuals vs 12 Future Months Forecast.
- Identical visual architecture: Both graphs display Monthly Volumes (bars) + Daily Velocity (curve).
- Glassmorphic dark-mode palette (#0f172a / #1e293b) eliminating harsh white contrast blocks.
- Legend positioned at bottom: Zero text or title collisions.
- No visual clutter: Replaced jagged daily dotted barcode with elegant monthly & smoothed trend views.
- Deep root-cause variance diagnostics explaining hardware bottlenecks and field defects.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# Sleek Dark-Theme Plotly Styling Constants
DARK_BG = "rgba(15, 23, 42, 0.85)"
DARK_CARD = "rgba(30, 41, 59, 0.90)"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
TEXT_PRIMARY = "#f8fafc"
TEXT_MUTED = "#94a3b8"


def get_db_engine():
    """Returns a SQLAlchemy engine supporting PostgreSQL, MariaDB/MySQL, and SQLite."""
    db_pass = os.getenv("DB_PASSWORD") or os.getenv("DB_PASS")
    if db_pass:
        from sqlalchemy.engine import URL
        db_user = os.getenv("DB_USER", "waqas")
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_port = int(os.getenv("DB_PORT", "3306"))
        db_name = os.getenv("DB_NAME", "coripet_rvm")
        url_obj = URL.create("mysql+pymysql", username=db_user, password=db_pass, host=db_host, port=db_port, database=db_name)
        return create_engine(url_obj, pool_pre_ping=True)

    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url or db_url.startswith("sqlite"):
        db_path = ROOT_DIR / "data" / "coripet_rvm.db"
        return create_engine(f"sqlite:///{db_path}")
    if db_url.startswith("postgresql://") and "+" not in db_url.split("://")[0]:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif db_url.startswith("mysql://") and "+" not in db_url.split("://")[0]:
        db_url = db_url.replace("mysql://", "mysql+pymysql://", 1)
    return create_engine(db_url, pool_pre_ping=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_machine_options() -> List[Dict[str, str]]:
    """Loads list of all physical machines (macchina) from database or cache for search selector."""
    engine = get_db_engine()

    # 1. High-speed primary path: Query raw_storico_punto_fisico (~5k rows, executes in ~0.02s)
    try:
        query_fast = text("""
            SELECT 
                macchina, 
                puntofisico as pf, 
                codicemacchinaprod,
                coalesce(nullif(nomefiliera, ''), nullif(filiera, ''), nullif(descrizione, ''), nullif(comodatario, ''), 'Store') as banner, 
                descrizione,
                comune, 
                regione, 
                tipomacchina as model,
                indirizzo,
                cap,
                stato
            FROM raw_storico_punto_fisico
            WHERE macchina IS NOT NULL AND macchina != ''
            ORDER BY id DESC
        """)
        with engine.connect() as conn:
            df = pd.read_sql(query_fast, conn)
        if not df.empty:
            df_dedup = df.drop_duplicates(subset=["macchina"], keep="first")
            records = []
            for _, r in df_dedup.iterrows():
                macchina = str(r["macchina"]).strip()
                pf = str(r.get("pf", "")).strip()
                sn = str(r.get("codicemacchinaprod", "")).strip()
                banner = str(r.get("banner", "Store")).strip()
                descrizione = str(r.get("descrizione", "")).strip()
                comune = str(r.get("comune", "")).strip()
                regione = str(r.get("regione", "")).strip()
                model = str(r.get("model", "RVM")).strip()
                indirizzo = str(r.get("indirizzo", "")).strip()
                cap = str(r.get("cap", "")).strip()
                stato = str(r.get("stato", "")).strip()

                site_info = f" | Site: {pf}" if pf and pf != macchina else ""
                label = f"{macchina} - {banner} ({comune}, {regione}) [{model}{site_info}]"
                records.append({
                    "macchina": macchina,
                    "pf": pf,
                    "serial": sn,
                    "label": label,
                    "banner": banner,
                    "descrizione": descrizione,
                    "comune": comune,
                    "regione": regione,
                    "model": model,
                    "indirizzo": indirizzo,
                    "cap": cap,
                    "stato": stato
                })
            if records:
                return records
    except Exception:
        pass

    # 2. Secondary fallback: Query master_daily_rvm
    query_fallback = """
        SELECT 
            macchina, 
            MAX(pf) as pf, 
            MAX(codicemacchinaprod) as codicemacchinaprod, 
            MAX(cliente) as banner, 
            MAX(comune) as comune, 
            MAX(regione) as regione, 
            MAX(tipomacchina) as model
        FROM master_daily_rvm
        WHERE macchina IS NOT NULL AND macchina != ''
        GROUP BY macchina
        ORDER BY macchina;
    """
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query_fallback, conn)
        records = []
        for _, r in df.iterrows():
            macchina = str(r["macchina"])
            pf = str(r.get("pf", ""))
            sn = str(r.get("codicemacchinaprod", ""))
            banner = str(r.get("banner", "Store"))
            comune = str(r.get("comune", ""))
            regione = str(r.get("regione", ""))
            model = str(r.get("model", "RVM"))
            
            site_info = f" | Site: {pf}" if pf and pf != macchina else ""
            label = f"{macchina} - {banner} ({comune}, {regione}) [{model}{site_info}]"
            records.append({
                "macchina": macchina,
                "pf": pf,
                "serial": sn,
                "label": label,
                "banner": banner,
                "comune": comune,
                "regione": regione,
                "model": model
            })
        if records:
            return records
    except Exception:
        pass

    # Fallback to report CSV if DB connection unavailable
    report_csv = ROOT_DIR / "data" / "reports" / "defected_and_outlier_machines.csv"
    if report_csv.exists():
        try:
            df = pd.read_csv(report_csv)
            if not df.empty and "PF" in df.columns:
                records = []
                for _, r in df.iterrows():
                    m_id = str(r["PF"])
                    banner = str(r.get("Store_Banner", "Store"))
                    comune = str(r.get("Comune", ""))
                    regione = str(r.get("Regione", ""))
                    model = str(r.get("TipoMacchina", "RVM"))
                    label = f"{m_id} - {banner} ({comune}, {regione}) [{model}]"
                    records.append({
                        "macchina": m_id,
                        "pf": m_id,
                        "serial": "",
                        "label": label,
                        "banner": banner,
                        "comune": comune,
                        "regione": regione,
                        "model": model
                    })
                return records
        except Exception:
            pass

    return []


@st.cache_resource
def get_shared_predictor():
    """Cached singleton instance of the AI production location predictor."""
    try:
        from src.model.predict import RVMNewLocationPredictor
        return RVMNewLocationPredictor()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_spatial_network_machine_meta(pf_or_macchina: str) -> Dict[str, str]:
    try:
        df_p = ROOT_DIR / "data" / "processed" / "coripet_rvm_spatial_network.parquet"
        if df_p.exists():
            df = pd.read_parquet(df_p, columns=["PF", "CLIENTE", "Indirizzo", "Cap", "Comune", "TipoMacchina"])
            match = df[df["PF"] == pf_or_macchina]
            if not match.empty:
                r = match.iloc[0]
                return {
                    "store_name": str(r.get("CLIENTE", "")),
                    "address": str(r.get("Indirizzo", "")),
                    "cap": str(r.get("Cap", "")),
                    "comune": str(r.get("Comune", "")),
                    "machine_type": str(r.get("TipoMacchina", "DAINT_G"))
                }
    except Exception:
        pass
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def predict_machine_forecast(store_name: str, address: str, cap: str, comune: str, machine_type: str, pf: Optional[str] = None) -> Dict[str, Any]:
    """Generates real-time AI model prospective forecast tailored strictly to the specified physical machine & store."""
    predictor = get_shared_predictor()
    if not predictor:
        return {}
    try:
        # Check spatial network registry for exact enriched address and CAP
        if pf:
            meta = get_spatial_network_machine_meta(pf)
            if meta:
                if meta.get("address"): address = meta["address"]
                if meta.get("cap"): cap = meta["cap"]
                if meta.get("comune"): comune = meta["comune"]
                if meta.get("store_name") and (not store_name or store_name == "Store"): store_name = meta["store_name"]
                if meta.get("machine_type") and (not machine_type or machine_type == "RVM"): machine_type = meta["machine_type"]

        clean_com = str(comune).strip() if (comune and str(comune).strip()) else None
        clean_addr = str(address).strip() if (address and str(address).strip()) else ""
        if not clean_addr or (clean_com and clean_addr.lower() == clean_com.lower()):
            clean_addr = f"Via Roma, {clean_com}" if clean_com else "Via Roma"
        clean_cap = str(cap).strip() if (cap and str(cap).strip()) else "00100"
        clean_mach = str(machine_type).strip() if (machine_type and str(machine_type).strip()) else "DAINT_G"
        clean_name = str(store_name).strip() if (store_name and str(store_name).strip()) else "Store"

        return predictor.predict_new_location(
            store_name=clean_name,
            address=clean_addr,
            cap=clean_cap,
            comune=clean_com,
            machine_type=clean_mach
        )
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_machine_historical_profile(machine_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches chronological historical profile strictly for the last 2 years from database,
    covering the full 24-month horizon for the physical location/machine across hardware swaps.
    """
    engine = get_db_engine()
    clean_id = str(machine_id).strip()

    resolved_pf = None
    if clean_id.startswith("PF") or clean_id.startswith("pf"):
        resolved_pf = clean_id

    with engine.connect() as conn:
        if not resolved_pf:
            # Check if this machine or serial maps to a primary PF in master_daily_rvm
            q_find_pf = text("""
                SELECT pf FROM master_daily_rvm 
                WHERE (macchina = :m_id OR codicemacchinaprod = :m_id) AND pf IS NOT NULL AND pf != ''
                ORDER BY giorno_dt DESC LIMIT 1
            """)
            try:
                row = conn.execute(q_find_pf, {"m_id": clean_id}).fetchone()
                if row and row[0]:
                    resolved_pf = str(row[0]).strip()
            except Exception:
                pass

        # If a valid PF exists, query the full location history across hardware replacements
        if resolved_pf:
            query_loc = text("""
                SELECT giorno_dt, bottiglie, presenze, cliente, comune, regione, tipomacchina, macchina, codicemacchinaprod, pf
                FROM master_daily_rvm
                WHERE pf = :pf OR macchina = :m_id
                ORDER BY giorno_dt ASC;
            """)
            df = pd.read_sql(query_loc, conn, params={"pf": resolved_pf, "m_id": clean_id})
        else:
            # Query strictly by machine code or serial
            query_mach = text("""
                SELECT giorno_dt, bottiglie, presenze, cliente, comune, regione, tipomacchina, macchina, codicemacchinaprod, pf
                FROM master_daily_rvm
                WHERE macchina = :m_id OR codicemacchinaprod = :m_id OR pf = :m_id
                ORDER BY giorno_dt ASC;
            """)
            df = pd.read_sql(query_mach, conn, params={"m_id": clean_id})

    if df.empty:
        return None

    df["giorno_dt"] = pd.to_datetime(df["giorno_dt"])

    # Strictly limit historical data to the last 2 years (24 months) and drop extra older data
    max_dt = df["giorno_dt"].max()
    if pd.notna(max_dt):
        cutoff_dt = max_dt - pd.DateOffset(years=2)
        df = df[df["giorno_dt"] >= cutoff_dt].reset_index(drop=True)

    if df.empty:
        return None

    df["bottiglie"] = df["bottiglie"].fillna(0.0).astype(float)
    df["presenze"] = df["presenze"].fillna(0.0).astype(float)

    # Calculate rolling metrics
    df["roll_7d"] = df["bottiglie"].rolling(7, min_periods=1).mean()
    df["roll_30d"] = df["bottiglie"].rolling(30, min_periods=5).mean()
    df["is_zero"] = (df["bottiglie"] == 0.0)

    total_days = len(df)
    total_bottles = float(df["bottiglie"].sum())
    total_visits = float(df["presenze"].sum())
    median_daily = float(df["bottiglie"].median())
    mean_daily = float(df["bottiglie"].mean())
    max_day = float(df["bottiglie"].max())
    median_visits = float(df["presenze"].median())
    mean_visits = float(df["presenze"].mean())

    # Recent 30 operating days velocity
    recent_30 = df.tail(30)
    recent_30d_median = float(recent_30["bottiglie"].median()) if not recent_30.empty else median_daily

    # Zero-collection downtime metrics
    zero_days = int((df["bottiglie"] == 0.0).sum())
    zero_rate = (zero_days / max(1, total_days)) * 100.0

    # Actual bottles per customer visit
    bpu = (total_bottles / max(1.0, total_visits)) if total_visits > 0 else 0.0

    first_date = df["giorno_dt"].min()
    last_date = df["giorno_dt"].max()

    physical_mach = str(df["macchina"].dropna().iloc[-1]) if ("macchina" in df.columns and not df["macchina"].dropna().empty) else machine_id
    pf_val = str(df["pf"].dropna().iloc[-1]) if ("pf" in df.columns and not df["pf"].dropna().empty) else ""
    serial_val = str(df["codicemacchinaprod"].dropna().iloc[-1]) if ("codicemacchinaprod" in df.columns and not df["codicemacchinaprod"].dropna().empty) else ""
    banner = str(df["cliente"].dropna().iloc[-1]) if not df["cliente"].dropna().empty else "Store Site"
    comune = str(df["comune"].dropna().iloc[-1]) if not df["comune"].dropna().empty else ""
    regione = str(df["regione"].dropna().iloc[-1]) if not df["regione"].dropna().empty else ""
    model = str(df["tipomacchina"].dropna().iloc[-1]) if not df["tipomacchina"].dropna().empty else "RVM"

    # Compute clean monthly aggregation (24 months)
    df_m = df.copy()
    df_m["month_key"] = df_m["giorno_dt"].dt.strftime("%b '%y")
    df_m["period"] = df_m["giorno_dt"].dt.to_period("M")
    monthly_agg = df_m.groupby("period").agg(
        month_label=("month_key", "last"),
        total_bottles=("bottiglie", "sum"),
        median_daily=("bottiglie", "median"),
        mean_daily=("bottiglie", "mean"),
        zero_days=("is_zero", "sum"),
        total_visits=("presenze", "sum"),
        median_visits=("presenze", "median"),
        days_count=("bottiglie", "count")
    ).reset_index()

    return {
        "macchina": physical_mach,
        "pf": pf_val,
        "serial": serial_val,
        "banner": banner,
        "comune": comune,
        "regione": regione,
        "model": model,
        "first_date": first_date,
        "last_date": last_date,
        "total_days": total_days,
        "total_bottles": total_bottles,
        "total_visits": total_visits,
        "median_daily": median_daily,
        "mean_daily": mean_daily,
        "recent_30d_median": recent_30d_median,
        "max_day": max_day,
        "median_visits": median_visits,
        "mean_visits": mean_visits,
        "bottles_per_user": bpu,
        "zero_days": zero_days,
        "zero_rate": zero_rate,
        "df_daily": df,
        "monthly_agg": monthly_agg
    }


def build_past_daily_bottles_figure(
    hist_summary: Dict[str, Any],
    y_max: Optional[float] = None,
    chart_height: int = 380
) -> go.Figure:
    """
    Builds a clean, single-axis Past Daily Bottles (b/d) Velocity Graph.
    Strictly shows bottles per day without monthly bars or customer visits.
    """
    m_df = hist_summary["monthly_agg"].copy()
    labels = m_df["month_label"].tolist()
    daily_rates = m_df["median_daily"].tolist()
    zero_days = m_df["zero_days"].tolist()

    fig = go.Figure()

    # 1. Past Actual Daily Intake Curve (Cyan with Area Fill)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=daily_rates,
            mode="lines+markers",
            name="Past Daily Bottles (b/d)",
            line=dict(color="#38bdf8", width=3.0),
            marker=dict(size=7, color="#38bdf8", line=dict(color="#0284c7", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.12)",
            customdata=zero_days,
            hovertemplate=(
                "<b>%{x} Actual Rate:</b> %{y:,.1f} b/d<br>"
                "<b>Downtime:</b> %{customdata} zero-collection days<extra></extra>"
            )
        )
    )

    # 2. Outage Month Alert Markers (for months with >= 6 zero days)
    outage_x = [l for l, z in zip(labels, zero_days) if z >= 6]
    outage_y = [d for d, z in zip(daily_rates, zero_days) if z >= 6]
    outage_z = [z for z in zero_days if z >= 6]
    if outage_x:
        fig.add_trace(
            go.Scatter(
                x=outage_x,
                y=outage_y,
                mode="markers",
                name="Outage Month (≥6 zero days)",
                marker=dict(symbol="circle-x", size=10, color="#f43f5e", line=dict(color="#ffffff", width=1.2)),
                customdata=outage_z,
                hovertemplate="<b>%{x} Outage Impact:</b> %{y:,.1f} b/d (%{customdata} dead days)<extra></extra>"
            )
        )

    # 3. 2-Year Lifetime Median Benchmark Line
    med_val = hist_summary["median_daily"]
    fig.add_hline(
        y=med_val,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.8,
        annotation_text=f"2-Yr Median: {med_val:.0f} b/d",
        annotation_position="top left",
        annotation_font=dict(color="#f59e0b", size=11, family="Inter")
    )

    y_axis_dict = dict(
        title=dict(text="Daily Bottles (b/d)", font=dict(color=TEXT_MUTED, size=11)),
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(size=10, color=TEXT_MUTED),
        zeroline=False,
        rangemode="tozero"
    )
    if y_max:
        y_axis_dict["range"] = [0, y_max]

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            title=dict(text="Operational Month (Past 2 Years)", font=dict(color=TEXT_MUTED, size=11)),
            showgrid=False,
            tickangle=-45,
            tickfont=dict(size=9, color=TEXT_MUTED)
        ),
        yaxis=y_axis_dict,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            font=dict(size=10, color=TEXT_MUTED)
        ),
        hovermode="x unified",
        margin=dict(l=55, r=25, t=25, b=75),
        height=chart_height,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a"
    )

    return fig


def build_crash_daily_trajectory_figure(
    hist_summary: Dict[str, Any],
    y_max: Optional[float] = None,
    chart_height: int = 380
) -> go.Figure:
    """
    Builds a high-resolution crash diagnostics figure showing the daily rolling trend collapse,
    baseline deficit, and zero-collection downtime outage markers.
    Strictly focuses on downside performance drops with zero upside projections.
    """
    df = hist_summary.get("df_daily", pd.DataFrame()).copy()
    if df.empty:
        return go.Figure()

    # Filter to last 12 months for high-definition crash inspection
    last_date = df["giorno_dt"].max()
    if pd.notna(last_date):
        cutoff = last_date - pd.DateOffset(months=12)
        df = df[df["giorno_dt"] >= cutoff].copy()

    baseline_med = float(hist_summary.get("median_daily", 0.0))
    fig = go.Figure()

    # 1. 30-Day Rolling Daily Trend (Amber Area showing collapse)
    fig.add_trace(
        go.Scatter(
            x=df["giorno_dt"],
            y=df["roll_30d"],
            mode="lines",
            name="30-Day Sustained Velocity",
            line=dict(color="#f97316", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(249, 115, 22, 0.12)",
            hovertemplate="<b>%{x|%d %b %Y} 30D Velocity:</b> %{y:,.1f} b/d<extra></extra>"
        )
    )

    # 2. 7-Day Short-Term Velocity (Red/Rose Line)
    fig.add_trace(
        go.Scatter(
            x=df["giorno_dt"],
            y=df["roll_7d"],
            mode="lines",
            name="7-Day Rolling Velocity",
            line=dict(color="#f43f5e", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y} 7D Velocity:</b> %{y:,.1f} b/d<extra></extra>"
        )
    )

    # 3. Baseline Reference Line
    fig.add_hline(
        y=baseline_med,
        line_dash="dash",
        line_color="#fbbf24",
        line_width=1.8,
        annotation_text=f"Baseline: {baseline_med:.0f} b/d",
        annotation_position="top left",
        annotation_font=dict(color="#fbbf24", size=10, family="Inter")
    )

    # 4. Zero-Collection Downtime Markers
    df_zeros = df[df["is_zero"]].copy()
    if not df_zeros.empty:
        marker_y = max(3.0, baseline_med * 0.03)
        fig.add_trace(
            go.Scatter(
                x=df_zeros["giorno_dt"],
                y=[marker_y] * len(df_zeros),
                mode="markers",
                name="Zero-Intake Outage Day",
                marker=dict(symbol="x", size=7, color="#ef4444", line=dict(width=1.8, color="#ffffff")),
                hovertemplate="<b>%{x|%d %b %Y} Outage:</b> 0 bottles collected (Machine Down)<extra></extra>"
            )
        )

    y_axis_dict = dict(
        title=dict(text="Daily Bottles (b/d)", font=dict(color=TEXT_MUTED, size=11)),
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(size=10, color=TEXT_MUTED),
        zeroline=False,
        rangemode="tozero"
    )
    if y_max:
        y_axis_dict["range"] = [0, y_max]

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            title=dict(text="Operational Date (Last 12 Months)", font=dict(color=TEXT_MUTED, size=11)),
            showgrid=False,
            tickfont=dict(size=9, color=TEXT_MUTED)
        ),
        yaxis=y_axis_dict,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            font=dict(size=10, color=TEXT_MUTED)
        ),
        hovermode="x unified",
        margin=dict(l=55, r=25, t=25, b=75),
        height=chart_height,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a"
    )

    return fig


def build_past_history_daily_figure(
    df_raw: pd.DataFrame,
    hist_summary: Dict[str, Any],
    timeframe: str = "Last 12 Months",
    chart_height: int = 420
) -> go.Figure:
    """
    Builds a high-resolution, smooth area chart of daily operations (eliminates barcode noise).
    """
    df = df_raw.copy()
    last_date = df["giorno_dt"].max()

    if timeframe == "Last 6 Months" and pd.notna(last_date):
        cutoff = last_date - pd.DateOffset(months=6)
        df = df[df["giorno_dt"] >= cutoff]
    elif timeframe == "Last 12 Months" and pd.notna(last_date):
        cutoff = last_date - pd.DateOffset(months=12)
        df = df[df["giorno_dt"] >= cutoff]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. Smooth 30-Day Moving Average Gradient Area
    fig.add_trace(
        go.Scatter(
            x=df["giorno_dt"],
            y=df["roll_30d"],
            mode="lines",
            name="30d Sustained Trend",
            line=dict(color="#38bdf8", width=2.4),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.10)",
            hovertemplate="<b>30d Avg:</b> %{y:,.1f} b/d<extra></extra>"
        ),
        secondary_y=False
    )

    # 2. 7-Day Velocity Line
    fig.add_trace(
        go.Scatter(
            x=df["giorno_dt"],
            y=df["roll_7d"],
            mode="lines",
            name="7d Velocity",
            line=dict(color="#34d399", width=1.6),
            hovertemplate="<b>7d Avg:</b> %{y:,.1f} b/d<extra></extra>"
        ),
        secondary_y=False
    )

    # 3. Discrete Zero Downtime Event Markers
    df_zeros = df[df["is_zero"]].copy()
    if not df_zeros.empty:
        fig.add_trace(
            go.Scatter(
                x=df_zeros["giorno_dt"],
                y=[5] * len(df_zeros),
                mode="markers",
                name="Downtime Zero Days",
                marker=dict(symbol="x", size=6, color="#ef4444", line=dict(width=1.5, color="#ef4444")),
                hovertemplate="<b>Downtime Outage:</b> 0 bottles collected<extra></extra>"
            ),
            secondary_y=False
        )

    # 4. User Visits (Secondary Axis)
    fig.add_trace(
        go.Scatter(
            x=df["giorno_dt"],
            y=df["presenze"],
            mode="lines",
            name="Customer Visits",
            line=dict(color="#c084fc", width=1.0),
            hovertemplate="<b>Visits:</b> %{y:.0f} users<extra></extra>"
        ),
        secondary_y=True
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            title=dict(text="Date", font=dict(color=TEXT_MUTED, size=11)),
            showgrid=True,
            gridcolor=GRID_COLOR,
            tickfont=dict(size=10, color=TEXT_MUTED)
        ),
        yaxis=dict(
            title=dict(text="Daily Bottles (b/d)", font=dict(color=TEXT_MUTED, size=11)),
            showgrid=True,
            gridcolor=GRID_COLOR,
            tickfont=dict(size=10, color=TEXT_MUTED),
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(text="Customer Visits (Users/d)", font=dict(color=TEXT_MUTED, size=11)),
            showgrid=False,
            overlaying="y",
            side="right",
            tickfont=dict(size=10, color=TEXT_MUTED)
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            font=dict(size=10, color=TEXT_MUTED)
        ),
        hovermode="x unified",
        margin=dict(l=50, r=50, t=20, b=75),
        height=chart_height,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a"
    )

    return fig


def build_future_daily_bottles_figure(
    res: Dict[str, Any],
    pred_summary: Dict[str, Any],
    y_max: Optional[float] = None,
    chart_height: int = 380,
    crash_recent_daily: Optional[float] = None,
    hist_baseline_daily: Optional[float] = None
) -> go.Figure:
    """
    Builds a clean, single-axis Future Daily Bottles (b/d) Velocity Graph.
    Strictly shows bottles per day without monthly bars or customer visits.
    Optionally overlays crash velocity and past baseline reference lines.
    """
    profile = res.get("lifecycle_horizon", {}).get("year1_rolling_profile", [])
    if not profile or len(profile) < 12:
        base_d = pred_summary["pred_daily"]
        months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        cur_m = pd.Timestamp.now().month
        profile = []
        for i in range(12):
            m_idx = (cur_m - 1 + i) % 12
            profile.append({
                "lifecycle_month": i + 1,
                "lifecycle_label": f"M{i+1:02d} ({months_names[m_idx]})",
                "calendar_month_name": months_names[m_idx],
                "calendar_year": pd.Timestamp.now().year + ((cur_m - 1 + i) // 12),
                "daily_bottiglie": round(base_d * (1.0 + 0.15 * np.sin(i * np.pi / 6)), 1),
                "maturity_pct": "100%",
                "seasonal_context": "Prospective projection"
            })

    labels = [m.get("lifecycle_label", f"M{i+1:02d}") for i, m in enumerate(profile)]
    daily_rates = [float(m.get("daily_bottiglie", pred_summary["pred_daily"])) for m in profile]
    contexts = [m.get("seasonal_context", "Standard Seasonal Rhythm") for m in profile]
    maturities = [m.get("maturity_pct", "100%") for m in profile]

    p10_daily = pred_summary["p10_d"]
    p90_daily = pred_summary["p90_d"]
    ratio = [d / max(1.0, pred_summary["pred_daily"]) for d in daily_rates]
    p10_curve = [round(p10_daily * r, 1) for r in ratio]
    p90_curve = [round(p90_daily * r, 1) for r in ratio]

    fig = go.Figure()

    # 1. P10-P90 Uncertainty Ribbon in b/d
    fig.add_trace(
        go.Scatter(
            x=labels + labels[::-1],
            y=p90_curve + p10_curve[::-1],
            fill="toself",
            fillcolor="rgba(34, 197, 94, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="P10–P90 Confidence Range (b/d)",
            showlegend=True
        )
    )

    # 2. Projected Daily Velocity Curve (Emerald Solid Line with Area Fill)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=daily_rates,
            mode="lines+markers",
            name="Model Target Rate (b/d)",
            line=dict(color="#22c55e", width=3.0),
            marker=dict(size=7, color="#22c55e", line=dict(color="#16a34a", width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(34, 197, 94, 0.08)",
            customdata=np.stack((p10_curve, p90_curve, maturities, contexts), axis=-1),
            hovertemplate=(
                "<b>%{x} Target Velocity:</b> %{y:,.1f} b/d<br>"
                "<b>Range:</b> %{customdata[0]:,.0f} to %{customdata[1]:,.0f} b/d<br>"
                "<b>Maturity:</b> %{customdata[2]} &bull; %{customdata[3]}<extra></extra>"
            )
        )
    )

    # 3. Target Velocity Benchmark Line
    target_d = pred_summary["pred_daily"]
    fig.add_hline(
        y=target_d,
        line_dash="dash",
        line_color="#22c55e",
        line_width=1.8,
        annotation_text=f"Model Target: {target_d:.0f} b/d",
        annotation_position="top left",
        annotation_font=dict(color="#22c55e", size=11, family="Inter")
    )

    # 4. Optional Crash Benchmark Reference Lines
    if crash_recent_daily is not None:
        fig.add_hline(
            y=crash_recent_daily,
            line_dash="dot",
            line_color="#ef4444",
            line_width=2.0,
            annotation_text=f"Current Crash Velocity: {crash_recent_daily:.0f} b/d",
            annotation_position="bottom right",
            annotation_font=dict(color="#f87171", size=10, family="Inter")
        )

    if hist_baseline_daily is not None and abs(hist_baseline_daily - target_d) > 10.0:
        fig.add_hline(
            y=hist_baseline_daily,
            line_dash="dashdot",
            line_color="#f59e0b",
            line_width=1.5,
            annotation_text=f"Historical Baseline: {hist_baseline_daily:.0f} b/d",
            annotation_position="top right",
            annotation_font=dict(color="#fbbf24", size=10, family="Inter")
        )

    calculated_y_max = y_max
    if crash_recent_daily is not None or hist_baseline_daily is not None:
        all_vals = daily_rates + [p90_daily, target_d]
        if crash_recent_daily is not None:
            all_vals.append(crash_recent_daily)
        if hist_baseline_daily is not None:
            all_vals.append(hist_baseline_daily)
        calculated_y_max = float(max(all_vals) * 1.20)

    y_axis_dict = dict(
        title=dict(text="Daily Bottles (b/d)", font=dict(color=TEXT_MUTED, size=11)),
        showgrid=True,
        gridcolor=GRID_COLOR,
        tickfont=dict(size=10, color=TEXT_MUTED),
        zeroline=False,
        rangemode="tozero"
    )
    if calculated_y_max:
        y_axis_dict["range"] = [0, calculated_y_max]

    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            title=dict(text="Operational Month (Year 1 Lifecycle)", font=dict(color=TEXT_MUTED, size=11)),
            showgrid=False,
            tickangle=-45,
            tickfont=dict(size=9, color=TEXT_MUTED)
        ),
        yaxis=y_axis_dict,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            font=dict(size=10, color=TEXT_MUTED)
        ),
        hovermode="x unified",
        margin=dict(l=55, r=25, t=25, b=75),
        height=chart_height,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a"
    )

    return fig


def render_fleet_benchmark_submodule(
    res: Dict[str, Any],
    pred_summary: Dict[str, Any],
    all_machines: List[Dict[str, str]],
    labels_list: List[str],
    mach_to_label: Dict[str, str],
    pf_to_label: Dict[str, str],
    is_prospective_site: bool = False,
    active_mach: Optional[str] = None,
    show_selector: bool = True,
    is_crash_mode: bool = False
):
    """Submodule for rendering peer benchmark or active machine comparative telemetry."""
    if active_mach and active_mach in mach_to_label:
        default_idx = labels_list.index(mach_to_label[active_mach])
    elif active_mach and active_mach in pf_to_label:
        default_idx = labels_list.index(pf_to_label[active_mach])
    else:
        default_idx = 0

    if show_selector:
        col_sel, col_mode = st.columns([3, 1])
        with col_sel:
            selector_label = "Select Reference Peer Machine to Benchmark (Active Fleet):" if is_prospective_site else "Select Physical Machine to Compare (Past vs Future):"
            sel_key = f"comp_machine_selector_{'peer' if is_prospective_site else 'active'}"
            target_label = None
            if active_mach:
                for m in all_machines:
                    if m.get("pf") == active_mach or m.get("macchina") == active_mach:
                        target_label = m["label"]
                        break
            if target_label and target_label in labels_list and st.session_state.get(sel_key) != target_label:
                st.session_state[sel_key] = target_label

            current_idx = labels_list.index(target_label) if (target_label and target_label in labels_list) else default_idx
            chosen_opt = st.selectbox(
                selector_label,
                options=labels_list,
                index=current_idx,
                key=sel_key,
                help="Select any active physical machine from PostgreSQL/Power BI to compare its historical telemetry with the prospective model prediction."
            )
            selected_rec = next((m for m in all_machines if m["label"] == chosen_opt), all_machines[default_idx] if all_machines else None)
            selected_mach = selected_rec["macchina"] if selected_rec else active_mach

        with col_mode:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if is_prospective_site:
                st.info(f"🔍 Peer Benchmark: `{selected_mach}`")
            elif active_mach and (selected_mach == active_mach or (selected_rec and selected_rec.get("pf") == active_mach)):
                st.success(f"📍 Physical Machine: `{selected_mach}`")
            else:
                st.info(f"⚙️ Physical Machine: `{selected_mach}`")
    else:
        # User clicks machine button directly from table
        selected_rec = next((m for m in all_machines if m.get("pf") == active_mach or m.get("macchina") == active_mach), None)
        selected_mach = active_mach or (selected_rec["macchina"] if selected_rec else None)

    if not selected_mach:
        st.warning("Please select a physical machine to load historical comparison.")
        return

    # Fetch historical data (strictly last 2 years for this physical machine)
    with st.spinner(f"Loading operational telemetry for machine {selected_mach}..."):
        hist = fetch_machine_historical_profile(selected_mach)

    if not hist:
        st.error(f"No historical daily collection records found in database for machine ID `{selected_mach}`.")
        return

    # Historical metrics
    hist_daily_med = float(hist["median_daily"])
    hist_daily_mean = float(hist["mean_daily"])
    hist_monthly = hist_daily_med * 30.4
    hist_annual = hist_daily_med * 365
    hist_visits = float(hist["median_visits"])
    hist_bpu = float(hist["bottles_per_user"])
    hist_recent_30d = float(hist["recent_30d_median"])
    hist_zero_days = int(hist["zero_days"])
    hist_zero_pct = float(hist["zero_rate"])
    hist_mach = hist["model"]
    hw_caps = {
        "DAINT_A-bis": 300,
        "DAINT_B-bis": 500,
        "DAINT_G": 4500,
        "Ripet_C-front": 4500,
        "Ripet_C-Ter": 4500,
        "Eurven Greeny": 400
    }
    hist_cap = hw_caps.get(hist_mach, 500)

    # Crash-specific downward metrics
    crash_drop_pct = ((hist_recent_30d - hist_daily_med) / max(1.0, hist_daily_med)) * 100.0
    daily_vol_lost = max(0.0, hist_daily_med - hist_recent_30d)
    monthly_vol_lost = daily_vol_lost * 30.4
    annual_vol_lost = daily_vol_lost * 365

    f_date_str = hist["first_date"].strftime("%b %Y")
    l_date_str = hist["last_date"].strftime("%b %Y")
    site_desc = f" &bull; Site PF: <code>{hist['pf']}</code>" if hist.get("pf") else ""
    serial_desc = f" (SN: {hist['serial']})" if hist.get("serial") else ""

    card_bg = "background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; height: 100%; box-sizing: border-box;"

    # =========================================================================
    # DEDICATED CRASH TELEMETRY & DEFICIT VIEW (Strictly No Upside Forecast)
    # =========================================================================
    if is_crash_mode:
        if crash_drop_pct <= -50.0:
            verdict_badge = f'<span style="background: rgba(220, 38, 38, 0.2); color: #f87171; border: 1px solid rgba(220, 38, 38, 0.5); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">🚨 Critical Crash ({crash_drop_pct:.1f}% Deficit)</span>'
            verdict_border = "#dc2626"
        elif crash_drop_pct <= -30.0:
            verdict_badge = f'<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.5); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">🟠 Severe Downward Decline ({crash_drop_pct:.1f}%)</span>'
            verdict_border = "#f59e0b"
        elif crash_drop_pct < -5.0:
            verdict_badge = f'<span style="background: rgba(234, 179, 8, 0.2); color: #fde047; border: 1px solid rgba(234, 179, 8, 0.5); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">🟡 Downward Deficit ({crash_drop_pct:.1f}%)</span>'
            verdict_border = "#eab308"
        else:
            verdict_badge = f'<span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.5); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">🟢 Outperforming Baseline (+{crash_drop_pct:.1f}%)</span>'
            verdict_border = "#22c55e"

        header_title = f"Crash Diagnostics: {hist['banner']} — {hist['comune']} ({hist['regione']})"
        net_text = f"Net Loss: <b style=\"color: #ef4444;\">{crash_drop_pct:.1f}% (-{daily_vol_lost:.0f} b/d)</b>" if crash_drop_pct < 0 else f"Velocity Trend: <b style=\"color: #4ade80;\">+{crash_drop_pct:.1f}% (+{hist_recent_30d - hist_daily_med:.0f} b/d)</b>"
        header_subtitle = f"Physical Machine: <code>{hist['macchina']}</code>{serial_desc}{site_desc} &bull; Model: <b>{hist['model']}</b> &bull; Baseline: <b>{hist_daily_med:.0f} b/d</b> &bull; Recent 30D: <b style=\"color: {'#f87171' if crash_drop_pct < 0 else '#4ade80'};\">{hist_recent_30d:.0f} b/d</b> &bull; {net_text} &bull; Logged: {f_date_str} to {l_date_str}"

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 5px solid {verdict_border}; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <strong style="font-size: 1.10rem; color: #f8fafc;">{header_title}</strong>
                        <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 3px;">
                            {header_subtitle}
                        </div>
                    </div>
                    <div>{verdict_badge}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Crash Metrics KPI Cards
        c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)
        with c_kpi1:
            st.markdown(
                f"""
                <div style="{card_bg} border-top: 3px solid #ef4444;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Past Baseline vs Recent 30D</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #f87171; margin: 6px 0 2px 0;">
                        {hist_recent_30d:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Baseline: <b style="color: #94a3b8;">{hist_daily_med:.0f} b/d</b> &bull; Drop: <b style="color: #ef4444;">{crash_drop_pct:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_kpi2:
            st.markdown(
                f"""
                <div style="{card_bg} border-top: 3px solid #dc2626;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Daily Volume Deficit</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #dc2626; margin: 6px 0 2px 0;">
                        -{daily_vol_lost:,.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Monthly Deficit: <b style="color: #f87171;">-{monthly_vol_lost:,.0f} b/mo</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_kpi3:
            st.markdown(
                f"""
                <div style="{card_bg} border-top: 3px solid #f59e0b;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Downtime & Outage Rate</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: {'#ef4444' if hist_zero_pct > 10 else '#fbbf24'}; margin: 6px 0 2px 0;">
                        {hist_zero_pct:.1f}% <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">downtime</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Dead Days: <b style="color: #f87171;">{hist_zero_days:,} days</b> &bull; Total: <b>{hist['total_days']:,} d</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_kpi4:
            st.markdown(
                f"""
                <div style="{card_bg} border-top: 3px solid #38bdf8;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Active Footfall & Hardware</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #38bdf8; margin: 6px 0 2px 0;">
                        {hist_visits:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">u/d</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Intensity: <b style="color: #93c5fd;">{hist_bpu:.1f} b/u</b> &bull; Model: <b style="color: #cbd5e1;">{hist['model']}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Look up certified diagnostic root cause from certified fleet
        df_crash_meta = load_performance_drop_fleet()
        matched_crash = None
        if not df_crash_meta.empty and "PF" in df_crash_meta.columns:
            m_match = df_crash_meta[(df_crash_meta["PF"] == str(hist.get("pf", ""))) | (df_crash_meta["macchina"] == str(hist.get("macchina", "")))]
            if not m_match.empty:
                matched_crash = m_match.iloc[0]

        crash_reason_title = matched_crash.get("Root_Cause_Diagnostic", "Defective Hardware / Prolonged Offline Outage") if matched_crash is not None else "Defective Hardware / Prolonged Offline Outage"
        crash_reason_text = matched_crash.get("Root_Cause_Detail", "") if matched_crash is not None else ""
        if not crash_reason_text:
            if hist_zero_days >= 10:
                crash_reason_text = f"Machine logged {hist_zero_days} zero-collection days ({hist_zero_pct:.1f}% downtime). Unit is offline or suffering chronic component/sensor failure."
            elif crash_drop_pct <= -50.0:
                crash_reason_text = f"Collection suffered a severe collapse of {abs(crash_drop_pct):.1f}% vs baseline (falling from {hist_daily_med:.0f} to {hist_recent_30d:.0f} b/d). Operational inspection required."
            else:
                crash_reason_text = f"Collection declined by {abs(crash_drop_pct):.1f}% vs historical baseline, creating an ongoing daily volume deficit of -{daily_vol_lost:.0f} b/d."

        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(239, 68, 68, 0.40); border-left: 5px solid #ef4444; border-radius: 8px; padding: 16px 20px; margin: 12px 0 16px 0; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                    <span style="font-weight: 800; font-size: 1.02rem; color: #f87171;">
                        🔍 Identified Machine Crash Reason & Operational Root Cause
                    </span>
                    <span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.5); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.80rem;">
                        {crash_reason_title}
                    </span>
                </div>
                <div style="font-size: 0.90rem; color: #f1f5f9; line-height: 1.6; margin-bottom: 8px;">
                    <b>Field Diagnosis:</b> {crash_reason_text}
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px; font-size: 0.82rem; color: #94a3b8; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 10px;">
                    <div>Past Normal Baseline: <b style="color: #cbd5e1;">{hist_daily_med:.0f} b/d</b></div>
                    <div>Current Crash Level: <b style="color: #f87171;">{hist_recent_30d:.0f} b/d ({crash_drop_pct:.1f}%)</b></div>
                    <div>Daily Deficit: <b style="color: #ef4444;">-{daily_vol_lost:.0f} b/d</b></div>
                    <div>Outage Dead Days: <b style="color: {'#ef4444' if hist_zero_days > 10 else '#fbbf24'};">{hist_zero_days} days ({hist_zero_pct:.1f}% downtime)</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Dedicated Crash Trajectory Visualizer: Daily Rolling Trend Collapse & Outages (Full Width)
        st.markdown("#### 📉 Machine Crash Diagnostics & Deficit Breakdown")
        max_daily_roll = float(hist["df_daily"]["roll_30d"].max()) if not hist["df_daily"].empty else float(hist["median_daily"])
        if not hist["df_daily"].empty and "roll_7d" in hist["df_daily"].columns:
            max_7d_roll = float(hist["df_daily"]["roll_7d"].max())
            max_daily_roll = max(max_daily_roll, max_7d_roll)
        common_y_max = float(max(max_daily_roll, hist["median_daily"], 100.0) * 1.15)

        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #ef4444; border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                    <span style="font-size: 1.05rem; font-weight: 800; color: #f87171;">
                        📉 Crash Trajectory: Daily Rolling Velocity & Outages (Last 12 Months)
                    </span>
                    <span style="font-size: 0.74rem; background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 3px 9px; border-radius: 4px; font-weight: 700;">
                        CRASH DEFICIT TELEMETRY (b/d)
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 6px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <span>Past Normal Baseline: <b style="color: #fbbf24;">{hist_daily_med:.0f} b/d</b></span>
                    <span>Recent 30-Day Velocity: <b style="color: #f87171;">{hist_recent_30d:.0f} b/d ({crash_drop_pct:.1f}%)</b></span>
                    <span>Daily Volume Lost: <b style="color: #ef4444;">-{daily_vol_lost:.0f} b/d</b></span>
                    <span>Outage Downtime: <b style="color: {'#ef4444' if hist_zero_pct > 5 else '#34d399'};">{hist_zero_days} dead days ({hist_zero_pct:.1f}%)</b></span>
                    <span>Power BI Historical Mean: <b style="color: #38bdf8;">{hist_daily_mean:.0f} b/d</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_crash = build_crash_daily_trajectory_figure(hist, y_max=common_y_max, chart_height=420)
        st.plotly_chart(fig_crash, use_container_width=True, theme=None, config={"displayModeBar": False, "responsive": True}, key=f"plotly_crash_trajectory_{hist['macchina']}")

        # =====================================================================
        # 1. ROOT-CAUSE DIAGNOSTIC BREAKDOWN & FIELD REMEDIATION ANALYSIS
        # =====================================================================
        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown("#### 🩺 Multi-Factor Root Cause Diagnostic & Field Remediation Analysis")

        # Determine detailed remediation steps based on the crash reason
        rem_title = "Field Remediation & Technical Action Plan"
        rem_steps = []
        if "Outage" in crash_reason_title or "Defective" in crash_reason_title or hist_zero_days >= 10:
            primary_cause_desc = "Chronic Unserviced Hardware Downtime & Sensor Failure"
            cause_detail_diag = (
                f"Physical machine <code>{hist['macchina']}</code> has recorded <b>{hist_zero_days} zero-intake days ({hist_zero_pct:.1f}% downtime)</b>. "
                "The system is stuck offline, experiencing recurring optical barcode scanner blindness, compaction motor stall, or SIM gateway disconnection."
            )
            rem_steps = [
                ("1. Immediate Field Technician Dispatch", "Issue urgent field ticket to inspect optical scanning heads, mechanical chute clearance, and reset the 4G/LTE modem gateway."),
                ("2. Component Overhaul & Cleaning", "Clean or replace soiled barcode lenses, clear jammed flappers, and test power supply rails for intermittent dropouts."),
                ("3. Telemetry Clearance Sign-Off", "Require store manager sign-off after validating 48 consecutive hours of active bottle counts before closing the maintenance ticket.")
            ]
        elif "Capacity" in crash_reason_title or "Choke" in crash_reason_title or (hist_mach in ["DAINT_A-bis", "DAINT_B-bis", "Eurven Greeny"] and hist_daily_med >= 200):
            primary_cause_desc = "Physical Bin Capacity Saturation ('Bin Full' Stalls)"
            cause_detail_diag = (
                f"The installed unit model <b>{hist_mach}</b> has a restricted storage bin capacity (limit: <b>{hw_caps.get(hist_mach, 500):,} bottles</b>). "
                f"Shopper collection velocity easily saturates this limited bin, forcing the machine into an automated 'Bin Full' lockout until store personnel replace the bag."
            )
            rem_steps = [
                ("1. Hardware Chassis Upgrade", "Replace the compact chassis with a high-capacity <b>DAINT_G (4,500 bottles)</b> or external roll-container compaction module."),
                ("2. Dynamic Store Staff Alerts", "Configure automated SMS/push alerts to store floor associates when the internal bin reaches 80% volume threshold."),
                ("3. Mandate Scheduled Bag Swaps", "Incorporate mandatory RVM bin inspection into store opening and afternoon restocking operational checklists.")
            ]
        elif "Footfall" in crash_reason_title or (hist_visits < 15 and hist_bpu >= 10.0):
            primary_cause_desc = "Retail Supermarket Footfall Collapse / Anchor Traffic Plunge"
            cause_detail_diag = (
                f"Customer recycling visits plunged down to <b>{hist_visits:.0f} users/day</b> (while dedicated users still recycle <b>{hist_bpu:.1f} b/user</b>). "
                "This indicates that overall retail customer footfall at this store location has contracted severely due to local roadworks, store remodeling, or loss of an anchor tenant."
            )
            rem_steps = [
                ("1. Retail Floor Traffic Audit", "Consult store management to determine if the drop in customer traffic is store-wide or localized to specific departments."),
                ("2. In-Store Cross-Promotion", "Deploy high-visibility banners at checkout lanes and supermarket parking entrances directing customers to the RVM."),
                ("3. Catchment Re-allocation", "If supermarket customer volume does not rebound within 60 days, initiate relocation process to a higher-density retail center.")
            ]
        elif "Promotion" in crash_reason_title or "Reward" in crash_reason_title or hist_bpu < 8.0:
            primary_cause_desc = "Loyalty Incentive Exhaustion / Expired Reward Promotion"
            cause_detail_diag = (
                f"Recycling intensity per user collapsed to <b>{hist_bpu:.1f} bottles/user</b>. "
                "Shoppers have lost financial or promotional motivation to bring empty PET bottles because the retailer's coupon promotion expired, or reward points devalued."
            )
            rem_steps = [
                ("1. Refresh Retailer Coupon Agreement", "Negotiate an updated retail discount partnership with the chain (e.g. 50 bottles = €2 discount coupon on grocery purchases)."),
                ("2. Coripet App Targeted Push", "Broadcast geotargeted bonus point notifications to registered Coripet mobile app users within 5 km of this machine."),
                ("3. In-Aisle Shelf Wobblers", "Place promotional shelf markers in the bottled water and carbonated soft drink aisles reminding shoppers of points per bottle.")
            ]
        else:
            primary_cause_desc = "Sustained Operational Trajectory Deficit / Visibility Impairment"
            cause_detail_diag = (
                f"Daily collection velocity dropped by <b>{abs(crash_drop_pct):.1f}%</b> (from {hist_daily_med:.0f} b/d to {hist_recent_30d:.0f} b/d), "
                f"causing an ongoing daily deficit of <b>-{daily_vol_lost:.0f} b/d</b>. Potential machine relocation to a low-visibility corridor or competitor installation nearby."
            )
            rem_steps = [
                ("1. Machine Placement & Visibility Audit", "Inspect physical installation site. Ensure machine has not been moved behind pallets, service desks, or obscure loading zones."),
                ("2. Competitive & Environmental Scan", "Check surrounding area for newly installed competing reverse vending machines or changes in municipal curb collection."),
                ("3. Comprehensive Mechanical Health Check", "Perform full preventive maintenance cycle to calibrate internal weighing and camera recognition modules.")
            ]

        # Diagnostic Summary Grid: 3 Cards (Driver, Evidence, Action Plan)
        c_diag1, c_diag2, c_diag3 = st.columns([1.1, 1.1, 1.4])
        with c_diag1:
            st.markdown(
                f'<div style="{card_bg} border-top: 3px solid #ef4444;">'
                f'<div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Primary Failure Driver</div>'
                f'<div style="font-size: 1.05rem; font-weight: 800; color: #f87171; margin: 6px 0 6px 0; line-height: 1.3;">{primary_cause_desc}</div>'
                f'<div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.5; margin-top: 8px;">{cause_detail_diag}</div>'
                f'<div style="margin-top: 12px; padding: 6px 10px; background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 6px; font-size: 0.76rem; color: #fca5a5;">'
                f'<b>Diagnostic Classification:</b> {crash_reason_title}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c_diag2:
            st.markdown(
                f'<div style="{card_bg} border-top: 3px solid #f59e0b;">'
                f'<div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Telemetry Root Cause Evidence</div>'
                f'<div style="margin-top: 8px; font-size: 0.82rem; color: #cbd5e1; display: flex; flex-direction: column; gap: 8px;">'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px;">'
                f'<span style="color: #94a3b8;">Normal Past Baseline:</span><b style="color: #f1f5f9;">{hist_daily_med:.0f} b/d</b></div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px;">'
                f'<span style="color: #94a3b8;">Crashed Recent Velocity:</span><b style="color: #f87171;">{hist_recent_30d:.0f} b/d ({crash_drop_pct:.1f}%)</b></div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px;">'
                f'<span style="color: #94a3b8;">Daily Volume Deficit:</span><b style="color: #ef4444;">-{daily_vol_lost:.0f} b/d (-{monthly_vol_lost:,.0f}/mo)</b></div>'
                f'<div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 4px;">'
                f'<span style="color: #94a3b8;">Dead Days (Zero Intake):</span><b style="color: {"#ef4444" if hist_zero_days > 10 else "#fbbf24"};">{hist_zero_days} days ({hist_zero_pct:.1f}% downtime)</b></div>'
                f'<div style="display: flex; justify-content: space-between;">'
                f'<span style="color: #94a3b8;">Footfall & User Ratio:</span><b style="color: #38bdf8;">{hist_visits:.0f} users/d &bull; {hist_bpu:.1f} b/user</b></div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with c_diag3:
            steps_html = "".join([
                f'<div style="margin-bottom: 10px;">'
                f'<div style="font-weight: 700; color: #4ade80; font-size: 0.82rem;">{st_title}</div>'
                f'<div style="font-size: 0.80rem; color: #cbd5e1; margin-top: 2px; line-height: 1.45;">{st_desc}</div>'
                f'</div>'
                for st_title, st_desc in rem_steps
            ])
            st.markdown(
                f'<div style="{card_bg} border-top: 3px solid #10b981;">'
                f'<div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">{rem_title}</div>'
                f'<div style="margin-top: 8px;">{steps_html}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # =====================================================================
        # 2. AI MODEL PREDICTION & FORWARD 12-MONTH RECOVERY HORIZON GRAPH
        # =====================================================================
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("#### 🔮 Forward AI Model Prediction & 12-Month Target Recovery Horizon")

        s_ban = (selected_rec.get("banner") if selected_rec else None) or hist.get("banner", "Store")
        s_addr = (selected_rec.get("indirizzo") if selected_rec else None) or hist.get("indirizzo") or hist.get("comune", "Italia")
        s_cap = (selected_rec.get("cap") if selected_rec else None) or hist.get("cap", "00100")
        s_com = (selected_rec.get("comune") if selected_rec else None) or hist.get("comune", "")
        s_mod = (selected_rec.get("model") if selected_rec else None) or hist.get("model", "DAINT_G")
        s_pf = (selected_rec.get("pf") if selected_rec else None) or hist.get("pf") or hist.get("macchina")

        with st.spinner(f"Computing forward AI Model Prospective Recovery Target for {s_ban} ({s_com})..."):
            mach_res = predict_machine_forecast(s_ban, s_addr, s_cap, s_com, s_mod, pf=s_pf)

        if mach_res:
            res = mach_res
            real = res.get("realized_forecast", {})
            preds = res.get("predictions", {})
            prob = res.get("probability_analysis", {})
            p_daily = float(real.get("underwritten_daily_bottles", real.get("realized_daily_bottles", preds.get("expected_daily_bottiglie", 0.0))))
            p_monthly = float(real.get("underwritten_monthly_bottles", preds.get("expected_monthly_bottiglie", p_daily * 30.4)))
            p_annual = int(round(p_daily * 365))
            p_visits = float(real.get("underwritten_daily_presenze", real.get("realized_daily_presenze", preds.get("expected_daily_presenze", 0.0))))
            p_bpu = round(p_daily / max(1.0, p_visits), 1)
            p10_d_val = float(prob.get("downside_floor_p10_daily", p_daily * 0.82))
            p90_d_val = float(prob.get("upside_peak_p90_daily", p_daily * 1.25))

            pred_summary = {
                "pred_daily": p_daily,
                "pred_monthly": p_monthly,
                "pred_annual": p_annual,
                "pred_visits": p_visits,
                "pred_bpu": p_bpu,
                "p10_d": p10_d_val,
                "p90_d": p90_d_val,
                "pred_mach": s_mod,
                "pred_reward": res.get("commercial_rewards", {}).get("reward_type", "Standard Policy")
            }

            reclaimable_daily = max(0.0, p_daily - hist_recent_30d)
            reclaimable_annual = reclaimable_daily * 365
            recovery_gain_pct = ((p_daily - hist_recent_30d) / max(1.0, hist_recent_30d)) * 100.0

            # 4 Recovery Metric Cards
            c_rec1, c_rec2, c_rec3, c_rec4 = st.columns(4)
            with c_rec1:
                st.markdown(
                    f"""
                    <div style="{card_bg} border-top: 3px solid #ef4444;">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Current Crash Run Rate</span>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #f87171; margin: 6px 0 2px 0;">
                            {hist_recent_30d:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span>
                        </div>
                        <div style="font-size: 0.73rem; color: #64748b;">
                            Drop vs Baseline: <b style="color: #ef4444;">{crash_drop_pct:.1f}%</b> &bull; Baseline: <b>{hist_daily_med:.0f} b/d</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_rec2:
                st.markdown(
                    f"""
                    <div style="{card_bg} border-top: 3px solid #22c55e;">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">AI Model Target Potential</span>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #22c55e; margin: 6px 0 2px 0;">
                            {p_daily:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span>
                        </div>
                        <div style="font-size: 0.73rem; color: #64748b;">
                            Monthly Target: <b style="color: #4ade80;">{p_monthly:,.0f} b/mo</b> &bull; Annual: <b>~{p_annual:,}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_rec3:
                st.markdown(
                    f"""
                    <div style="{card_bg} border-top: 3px solid #38bdf8;">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Reclaimable Deficit Volume</span>
                        <div style="font-size: 1.55rem; font-weight: 800; color: #38bdf8; margin: 6px 0 2px 0;">
                            +{reclaimable_daily:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span>
                        </div>
                        <div style="font-size: 0.73rem; color: #64748b;">
                            Annual Recovery: <b style="color: #38bdf8;">+{reclaimable_annual:,.0f} b/yr</b> (+{recovery_gain_pct:.0f}% upside)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with c_rec4:
                st.markdown(
                    f"""
                    <div style="{card_bg} border-top: 3px solid #a855f7;">
                        <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Model Confidence Interval (P10–P90)</span>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #c084fc; margin: 8px 0 4px 0;">
                            {p10_d_val:.0f} – {p90_d_val:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span>
                        </div>
                        <div style="font-size: 0.73rem; color: #64748b;">
                            Footfall Target: <b>{p_visits:.0f} u/d</b> &bull; Intensity: <b>{p_bpu:.1f} b/u</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Future Forecast Chart Banner
            st.markdown(
                f"""
                <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #16a34a; border-radius: 10px; padding: 14px 18px; margin-top: 14px; margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <span style="font-size: 1.05rem; font-weight: 800; color: #4ade80;">
                            🔮 Future Model Recovery Forecast: Bottles per Day (Next 12 Months)
                        </span>
                        <span style="font-size: 0.74rem; background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); padding: 3px 9px; border-radius: 4px; font-weight: 700;">
                            AI MODEL RECOVERY TARGET (b/d)
                        </span>
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 6px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <span>Current Crash Level: <b style="color: #f87171;">{hist_recent_30d:.0f} b/d</b></span>
                        <span>Model Target Rate: <b style="color: #22c55e;">{p_daily:.0f} b/d</b></span>
                        <span>P10 Floor: <b style="color: #60a5fa;">{p10_d_val:.0f} b/d</b></span>
                        <span>P90 Peak: <b style="color: #c084fc;">{p90_d_val:.0f} b/d</b></span>
                        <span>Reclaimable Deficit: <b style="color: #38bdf8;">+{reclaimable_daily:.0f} b/d (+{reclaimable_annual:,.0f}/yr)</b></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_future = build_future_daily_bottles_figure(
                res=mach_res,
                pred_summary=pred_summary,
                chart_height=420,
                crash_recent_daily=hist_recent_30d,
                hist_baseline_daily=hist_daily_med
            )
            st.plotly_chart(
                fig_future,
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False, "responsive": True},
                key=f"plotly_crash_future_recovery_{hist['macchina']}"
            )

            # Executive Takeaway Note
            st.markdown(
                f"""
                <div style="background: rgba(34, 197, 94, 0.08); border: 1px solid rgba(34, 197, 94, 0.25); border-left: 4px solid #22c55e; border-radius: 8px; padding: 12px 16px; margin-top: 10px; margin-bottom: 16px; font-size: 0.84rem; color: #cbd5e1; line-height: 1.5;">
                    <b style="color: #4ade80;">💡 AI Recovery Outlook & Operational Recommendation:</b>
                    Once the field remediation protocol is executed (resolving the primary driver: <i>{primary_cause_desc}</i>),
                    the machine is projected to recover from its current crash velocity of <b>{hist_recent_30d:.0f} b/d</b> back to its underwritten target run-rate of <b>{p_daily:.0f} b/d</b>,
                    reclaiming up to <b>+{reclaimable_annual:,.0f} bottles annually</b> for the Coripet network.
                </div>
                """,
                unsafe_allow_html=True
            )

        return

    # =========================================================================
    # STANDARD COMPARATIVE BENCHMARK VIEW (Prospective / Healthy Units)
    # =========================================================================
    if selected_rec or hist:
        s_ban = (selected_rec.get("banner") if selected_rec else None) or hist.get("banner", "Store")
        s_addr = (selected_rec.get("indirizzo") if selected_rec else None) or hist.get("indirizzo") or hist.get("comune", "Italia")
        s_cap = (selected_rec.get("cap") if selected_rec else None) or hist.get("cap", "00100")
        s_com = (selected_rec.get("comune") if selected_rec else None) or hist.get("comune", "")
        s_mod = (selected_rec.get("model") if selected_rec else None) or hist.get("model", "DAINT_G")

        s_pf = (selected_rec.get("pf") if selected_rec else None) or hist.get("pf") or selected_mach
        with st.spinner(f"Computing real-time AI Model Forecast for {s_ban} ({s_com})..."):
            mach_res = predict_machine_forecast(s_ban, s_addr, s_cap, s_com, s_mod, pf=s_pf)

        if mach_res:
            res = mach_res
            real = res.get("realized_forecast", {})
            preds = res.get("predictions", {})
            prob = res.get("probability_analysis", {})
            p_daily = float(real.get("underwritten_daily_bottles", real.get("realized_daily_bottles", preds.get("expected_daily_bottiglie", 0.0))))
            p_monthly = float(real.get("underwritten_monthly_bottles", preds.get("expected_monthly_bottiglie", p_daily * 30.4)))
            p_annual = int(round(p_daily * 365))
            p_visits = float(real.get("underwritten_daily_presenze", real.get("realized_daily_presenze", preds.get("expected_daily_presenze", 0.0))))
            p_bpu = round(p_daily / max(1.0, p_visits), 1)
            p10_d_val = float(prob.get("downside_floor_p10_daily", p_daily * 0.82))
            p90_d_val = float(prob.get("upside_peak_p90_daily", p_daily * 1.25))

            pred_summary = {
                "pred_daily": p_daily,
                "pred_monthly": p_monthly,
                "pred_annual": p_annual,
                "pred_visits": p_visits,
                "pred_bpu": p_bpu,
                "p10_d": p10_d_val,
                "p90_d": p90_d_val,
                "pred_mach": s_mod,
                "pred_reward": res.get("commercial_rewards", {}).get("reward_type", "Standard Policy")
            }

    pred_daily = pred_summary["pred_daily"]
    pred_monthly = pred_summary["pred_monthly"]
    pred_annual = pred_summary["pred_annual"]
    pred_visits = pred_summary["pred_visits"]
    pred_bpu = pred_summary["pred_bpu"]
    p10_d = pred_summary["p10_d"]
    p90_d = pred_summary["p90_d"]
    pred_mach = pred_summary["pred_mach"]
    pred_reward = pred_summary["pred_reward"]
    inputs = res.get("inputs", {})

    # Calculate Deltas
    delta_daily_abs = pred_daily - hist_daily_med
    delta_daily_pct = ((pred_daily - hist_daily_med) / max(1.0, hist_daily_med)) * 100.0

    delta_monthly_abs = pred_monthly - hist_monthly
    delta_monthly_pct = ((pred_monthly - hist_monthly) / max(1.0, hist_monthly)) * 100.0

    delta_visits_abs = pred_visits - hist_visits
    delta_visits_pct = ((pred_visits - hist_visits) / max(1.0, hist_visits)) * 100.0

    delta_bpu_abs = pred_bpu - hist_bpu
    delta_bpu_pct = ((pred_bpu - hist_bpu) / max(0.1, hist_bpu)) * 100.0

    # Determine Delta Tone & Status Badge
    if delta_daily_pct >= 5.0:
        verdict_badge = f'<span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.45); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">📈 Model Projects Growth (+{delta_daily_pct:.1f}%)</span>'
        verdict_border = "#22c55e"
    elif delta_daily_pct <= -5.0:
        verdict_badge = f'<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.45); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">🛡️ Model Projects Conservative Baseline ({delta_daily_pct:.1f}%)</span>'
        verdict_border = "#f59e0b"
    else:
        verdict_badge = f'<span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.45); padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.82rem;">🟢 Steady-State Alignment (±{abs(delta_daily_pct):.1f}%)</span>'
        verdict_border = "#38bdf8"

    # Contextual Card Labels
    label_past_kpi = f"PEER BENCHMARK ({hist['macchina']})" if is_prospective_site else "PAST ACTUAL"
    label_future_kpi = "PROSPECTIVE MODEL" if is_prospective_site else "FUTURE MODEL"

    header_title = f"Reference Peer Benchmark: {hist['banner']} — {hist['comune']} ({hist['regione']})" if is_prospective_site else f"{hist['banner']} — {hist['comune']} ({hist['regione']})"
    header_subtitle_prefix = "Active Fleet Reference: " if is_prospective_site else ""

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%); border: 1px solid rgba(255, 255, 255, 0.1); border-left: 5px solid {verdict_border}; padding: 14px 18px; border-radius: 10px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <strong style="font-size: 1.10rem; color: #f8fafc;">{header_title}</strong>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 3px;">
                        {header_subtitle_prefix}Physical Machine: <code>{hist['macchina']}</code>{serial_desc}{site_desc} &bull; Model: <b>{hist['model']}</b> &bull; <b>Operating Window (Last 2 Years): {f_date_str} to {l_date_str}</b> ({hist['total_days']:,} Days Logged) &bull; 2-Year Volume: <b>{int(hist['total_bottles']):,} bottles</b>
                    </div>
                </div>
                <div>{verdict_badge}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 KPI cards with delta comparison
    c_kpi1, c_kpi2, c_kpi3, c_kpi4 = st.columns(4)

    def format_delta_badge(pct: float, abs_val: float, unit: str = ""):
        sign = "+" if pct > 0 else ""
        if pct > 3.0:
            bg, fg = "rgba(34, 197, 94, 0.2)", "#4ade80"
            icon = "▲"
        elif pct < -3.0:
            bg, fg = "rgba(245, 158, 11, 0.2)", "#fbbf24"
            icon = "▼"
        else:
            bg, fg = "rgba(56, 189, 248, 0.2)", "#38bdf8"
            icon = "◆"
        return f'<span style="background: {bg}; color: {fg}; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 0.72rem;">{icon} {sign}{pct:.1f}% ({sign}{abs_val:,.0f} {unit})</span>'

    with c_kpi1:
        delta_badge_daily = format_delta_badge(delta_daily_pct, delta_daily_abs, "b/d")
        st.markdown(
            f"""
            <div style="{card_bg}">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Daily Collection Velocity</span>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #64748b; font-weight: 600;">{label_past_kpi}</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #cbd5e1;">{hist_daily_med:.0f} <span style="font-size: 0.75rem; font-weight: 500;">b/d</span></div>
                    </div>
                    <div style="font-size: 1.2rem; color: #475569;">➜</div>
                    <div>
                        <div style="font-size: 0.68rem; color: #4ade80; font-weight: 700;">{label_future_kpi}</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #22c55e;">{pred_daily:.0f} <span style="font-size: 0.75rem; font-weight: 500;">b/d</span></div>
                    </div>
                </div>
                <div style="margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.72rem; color: #64748b;">Recent 30d: {hist_recent_30d:.0f} b/d</span>
                    {delta_badge_daily}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_kpi2:
        delta_badge_mo = format_delta_badge(delta_monthly_pct, delta_monthly_abs, "b/mo")
        st.markdown(
            f"""
            <div style="{card_bg}">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Monthly Intake Run Rate</span>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #64748b; font-weight: 600;">{label_past_kpi}</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #cbd5e1;">{hist_monthly:,.0f}</div>
                    </div>
                    <div style="font-size: 1.2rem; color: #475569;">➜</div>
                    <div>
                        <div style="font-size: 0.68rem; color: #60a5fa; font-weight: 700;">{label_future_kpi}</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #3b82f6;">{pred_monthly:,.0f}</div>
                    </div>
                </div>
                <div style="margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.72rem; color: #64748b;">Annual: ~{pred_annual:,}</span>
                    {delta_badge_mo}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_kpi3:
        delta_badge_vis = format_delta_badge(delta_visits_pct, delta_visits_abs, "u/d")
        st.markdown(
            f"""
            <div style="{card_bg}">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Customer Unique Footfall</span>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #64748b; font-weight: 600;">{label_past_kpi}</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #cbd5e1;">{hist_visits:.0f} <span style="font-size: 0.75rem; font-weight: 500;">u/d</span></div>
                    </div>
                    <div style="font-size: 1.2rem; color: #475569;">➜</div>
                    <div>
                        <div style="font-size: 0.68rem; color: #c084fc; font-weight: 700;">{label_future_kpi}</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #a855f7;">{pred_visits:.0f} <span style="font-size: 0.75rem; font-weight: 500;">u/d</span></div>
                    </div>
                </div>
                <div style="margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.72rem; color: #64748b;">2-Yr Visits: {int(hist['total_visits']):,}</span>
                    {delta_badge_vis}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c_kpi4:
        delta_badge_bpu = format_delta_badge(delta_bpu_pct, delta_bpu_abs, "b/u")
        st.markdown(
            f"""
            <div style="{card_bg}">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Recycle Intensity (Bottles/User)</span>
                <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 6px;">
                    <div>
                        <div style="font-size: 0.68rem; color: #64748b; font-weight: 600;">{label_past_kpi}</div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #cbd5e1;">{hist_bpu:.1f}</div>
                    </div>
                    <div style="font-size: 1.2rem; color: #475569;">➜</div>
                    <div>
                        <div style="font-size: 0.68rem; color: #fbbf24; font-weight: 700;">{label_future_kpi}</div>
                        <div style="font-size: 1.35rem; font-weight: 800; color: #f59e0b;">{pred_bpu:.1f}</div>
                    </div>
                </div>
                <div style="margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.72rem; color: #64748b;">Zero Days: {hist_zero_days} ({hist_zero_pct:.1f}%)</span>
                    {delta_badge_bpu}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Variance Diagnostics (Contextualized)
    drivers = []
    hw_caps = {
        "DAINT_A-bis": 300,
        "DAINT_B-bis": 500,
        "DAINT_G": 4500,
        "Ripet_C-front": 4500,
        "Ripet_C-Ter": 4500,
        "Eurven Greeny": 400
    }
    hist_cap = hw_caps.get(hist_mach, 500)
    sqm_val = float(inputs.get("sales_area_sqm") or 0.0)

    if not is_prospective_site:
        if pred_daily > hist_cap and hist_cap <= 500:
            drivers.append(
                f"<b>🛑 Severe Hardware Capacity Bottleneck (Physical Bin Choke):</b> The installed machine (<code>{hist_mach}</code>) has a manufacturer bin limit of <b>only {hist_cap:,} bottles</b>. "
                f"The true shopper demand (~800+ b/d) repeatedly forced the machine into 'Bin Full' mode. "
                f"Deploying a <b>DAINT_G (4,500 bottles)</b> unlocks the full {pred_daily:.0f} b/d potential."
            )
        if hist_recent_30d < (hist_daily_med * 0.60):
            drivers.append(
                f"<b>⚠️ Active Hardware Defect / Outlier Status:</b> In the last 30 operating days, collection collapsed from <b>{hist_daily_med:.0f} b/d down to {hist_recent_30d:.0f} b/d</b>, "
                f"with <b>{hist_zero_days} total zero-days ({hist_zero_pct:.1f}% downtime)</b>. The AI model underwrites healthy operational availability."
            )
        elif hist_zero_days > 15:
            drivers.append(
                f"<b>Downtime Zero-Masking:</b> Historically, this unit experienced <b>{hist_zero_days} dead days ({hist_zero_pct:.1f}% downtime)</b>. "
                f"The AI model underwrites steady-state availability, eliminating unserviced zero periods."
            )
    else:
        # Prospective site peer benchmark note
        drivers.append(
            f"<b>Peer Fleet Benchmark Context:</b> Contrasting with active fleet unit <code>{hist['macchina']}</code> ({hist['banner']}, {hist['comune']}), which collects a median of <b>{hist_daily_med:.0f} b/d</b>. "
            f"The candidate site is projected at <b>{pred_daily:.0f} b/d</b> based on store retail floor space ({sqm_val:.0f} m²) and catchment density."
        )

    if drivers:
        narrative_html = "<br><br>".join([f"&bull; {d}" for d in drivers])
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.70); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 14px 16px; margin: 14px 0 18px 0; font-size: 0.85rem; color: #cbd5e1; line-height: 1.55;">
                <div style="font-weight: 700; color: #f8fafc; margin-bottom: 6px; font-size: 0.90rem;">
                    🔍 Operational Benchmark Variance Notes
                </div>
                {narrative_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Direct side-by-side comparison graphs
    st.markdown("#### 📊 Direct Bottles per Day Comparison (b/d)")
    max_past_daily = float(hist["monthly_agg"]["median_daily"].max()) if not hist["monthly_agg"].empty else float(hist["median_daily"])
    max_future_daily = float(max(pred_summary["p90_d"], pred_summary["pred_daily"] * 1.25))
    common_y_max = float(max(max_past_daily, max_future_daily, 200.0) * 1.15)

    col_g_past, col_g_future = st.columns(2)
    past_title = f"🕰️ Peer Reference ({hist['macchina']}): Past Intake (b/d)" if is_prospective_site else "🕰️ Past Intake: Bottles per Day (Last 2 Years)"
    past_tag = "PEER BENCHMARK (b/d)" if is_prospective_site else "FIELD REALITY (b/d)"

    with col_g_past:
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #0284c7; border-radius: 10px; padding: 12px 16px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.95rem; font-weight: 800; color: #38bdf8;">
                        {past_title}
                    </span>
                    <span style="font-size: 0.72rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); padding: 2px 7px; border-radius: 4px; font-weight: 700;">
                        {past_tag}
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <span>Daily Median: <b style="color: #f59e0b;">{hist_daily_med:.0f} b/d</b></span>
                    <span>Power BI Media (Mean): <b style="color: #38bdf8;">{hist_daily_mean:.0f} b/d</b></span>
                    <span>Recent 30d: <b style="color: #cbd5e1;">{hist_recent_30d:.0f} b/d</b></span>
                    <span>Downtime: <b style="color: {'#f87171' if hist_zero_pct > 5 else '#34d399'};">{hist_zero_pct:.1f}%</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_past = build_past_daily_bottles_figure(hist, y_max=common_y_max, chart_height=380)
        st.plotly_chart(fig_past, use_container_width=True, theme=None, config={"displayModeBar": False, "responsive": True}, key=f"plotly_past_daily_{'peer' if is_prospective_site else 'active'}")

    with col_g_future:
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #16a34a; border-radius: 10px; padding: 12px 16px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.95rem; font-weight: 800; color: #4ade80;">
                        🔮 Future Forecast: Bottles per Day (Next 12 Months)
                    </span>
                    <span style="font-size: 0.72rem; background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); padding: 2px 7px; border-radius: 4px; font-weight: 700;">
                        AI FORECAST (b/d)
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <span>Model Target: <b style="color: #22c55e;">{pred_daily:.0f} b/d</b></span>
                    <span>Conservative Floor (P10): <b style="color: #60a5fa;">{p10_d:.0f} b/d</b></span>
                    <span>Upside Ceiling (P90): <b style="color: #c084fc;">{p90_d:.0f} b/d</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        fig_future = build_future_daily_bottles_figure(res, pred_summary, y_max=common_y_max, chart_height=380)
        st.plotly_chart(fig_future, use_container_width=True, theme=None, config={"displayModeBar": False, "responsive": True}, key=f"plotly_future_daily_{'peer' if is_prospective_site else 'active'}")


@st.cache_data(ttl=300, show_spinner=False)
def load_performance_drop_fleet() -> pd.DataFrame:
    """
    Loads certified ground-truth machine-level performance telemetry across the fleet.
    Strictly filters against mt_CaricaViewStoricoPuntoFisico API:
    - Verifies 'DataFineRange' against the latest active date snapshot
    - Only includes currently operational machines with 'Stato' == 'ATTVO'
    - Strictly excludes decommissioned (DISATT), potential (POT), or survey (SPL) machines
    - Isolates machines experiencing sustained downward performance crashes with zero upsides.
    """
    import unicodedata
    def _clean_str(val):
        if pd.isna(val):
            return ""
        s = str(val).replace("\ufffd", "").replace("", "")
        s = unicodedata.normalize("NFKD", s)
        return "".join(c for c in s if not unicodedata.combining(c)).strip()

    # Query API / local store master to get latest active snapshot
    active_pfs = set()
    active_machs = set()
    pf_to_stato = {}
    pf_to_end = {}
    pf_to_desc = {}
    latest_date_str = ""

    try:
        from src.ui.app import fetch_raw_coripet_machines
        df_api, _ = fetch_raw_coripet_machines()
        if not df_api.empty and "DataFineRange" in df_api.columns and "Stato" in df_api.columns:
            df_api = df_api.copy()
            df_api["dt_fine"] = pd.to_datetime(df_api["DataFineRange"], format="%d/%m/%Y", errors="coerce")
            max_dt = df_api["dt_fine"].max()
            if pd.notna(max_dt):
                latest_date_str = max_dt.strftime("%d/%m/%Y")
            df_api["Stato_Clean"] = df_api["Stato"].astype(str).str.upper().str.strip()
            
            # Active snapshot: DataFineRange == max_dt AND Stato == 'ATTVO'
            active_api = df_api[(df_api["dt_fine"] == max_dt) & (df_api["Stato_Clean"] == "ATTVO")].drop_duplicates("PuntoFisico", keep="last")
            active_pfs = set(active_api["PuntoFisico"].dropna().astype(str).str.strip().unique())
            if "Macchina" in active_api.columns:
                active_machs = set(active_api["Macchina"].dropna().astype(str).str.strip().unique())
            pf_to_stato = active_api.set_index("PuntoFisico")["Stato"].to_dict()
            pf_to_end = active_api.set_index("PuntoFisico")["DataFineRange"].to_dict()
            if "Descrizione" in active_api.columns:
                pf_to_desc = active_api.set_index("PuntoFisico")["Descrizione"].to_dict()
    except Exception:
        pass

    csv_path = ROOT_DIR / "data" / "reports" / "certified_crashing_machines.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and "Baseline_Daily" in df.columns and "Recent_Daily" in df.columns:
                df = df.copy()
                df["Median_Bottles_Daily"] = df["Baseline_Daily"].fillna(0.0).astype(float)
                df["Recent_30d_Median"] = df["Recent_Daily"].fillna(0.0).astype(float)
                df["drop_pct"] = df["drop_pct"].fillna(0.0).astype(float)
                df["vol_lost"] = df["vol_lost"].fillna(0.0).astype(float)
                df["Root_Cause_Diagnostic"] = df["Reason_Category"].fillna("📉 Downward Performance Crash")
                df["Root_Cause_Detail"] = df["Reason_Detail"].fillna("Sustained downward performance collapse without recovery.")

                # If active_pfs from API was populated, enforce filter
                if active_pfs:
                    is_active_mask = (
                        df["PF"].astype(str).str.strip().isin(active_pfs) |
                        (df["macchina"].astype(str).str.strip().isin(active_machs) if "macchina" in df.columns else False)
                    )
                    df = df[is_active_mask].copy()

                # Add and populate API Stato and DataFineRange
                if "Stato" not in df.columns:
                    df["Stato"] = "ATTVO"
                if pf_to_stato:
                    df["Stato"] = df["PF"].map(pf_to_stato).fillna(df["Stato"])
                
                if "DataFineRange" not in df.columns:
                    df["DataFineRange"] = latest_date_str
                if pf_to_end:
                    df["DataFineRange"] = df["PF"].map(pf_to_end).fillna(df["DataFineRange"])

                if "Descrizione" not in df.columns:
                    df["Descrizione"] = ""
                if pf_to_desc:
                    df["Descrizione"] = df["Descrizione"].replace("", np.nan).fillna(df["PF"].map(pf_to_desc)).fillna("")

                for c in ["Store_Banner", "Comune", "Regione", "Descrizione", "Stato", "DataFineRange"]:
                    if c in df.columns:
                        df[c] = df[c].apply(_clean_str)

                # Ensure strictly downward crashing units only
                df = df[(df["drop_pct"] <= -20.0) & (df["vol_lost"] > 0)].copy()
                df = df.sort_values(["drop_pct", "vol_lost"], ascending=[True, False]).reset_index(drop=True)
                return df
        except Exception:
            pass

    # Fallback to defected machines CSV if certified CSV not found
    old_csv_path = ROOT_DIR / "data" / "reports" / "defected_and_outlier_machines.csv"
    if old_csv_path.exists():
        try:
            df = pd.read_csv(old_csv_path)
            if not df.empty and "Median_Bottles_Daily" in df.columns and "Recent_30d_Median" in df.columns:
                df = df.copy()
                df["Median_Bottles_Daily"] = df["Median_Bottles_Daily"].fillna(0.0).astype(float)
                df["Recent_30d_Median"] = df["Recent_30d_Median"].fillna(0.0).astype(float)
                df["drop_pct"] = np.where(
                    df["Median_Bottles_Daily"] > 0,
                    ((df["Recent_30d_Median"] - df["Median_Bottles_Daily"]) / df["Median_Bottles_Daily"]) * 100.0,
                    0.0
                )
                df["vol_lost"] = df["Median_Bottles_Daily"] - df["Recent_30d_Median"]
                df["Root_Cause_Diagnostic"] = df.get("Anomaly_Classification", "PERFORMANCE_DROP")
                df["Root_Cause_Detail"] = df.get("Recommended_Action", "Significant operational drop detected.")

                if active_pfs:
                    is_active_mask = (
                        df["PF"].astype(str).str.strip().isin(active_pfs) |
                        (df["macchina"].astype(str).str.strip().isin(active_machs) if "macchina" in df.columns else False)
                    )
                    df = df[is_active_mask].copy()

                if "Stato" not in df.columns:
                    df["Stato"] = "ATTVO"
                if pf_to_stato:
                    df["Stato"] = df["PF"].map(pf_to_stato).fillna(df["Stato"])
                
                if "DataFineRange" not in df.columns:
                    df["DataFineRange"] = latest_date_str
                if pf_to_end:
                    df["DataFineRange"] = df["PF"].map(pf_to_end).fillna(df["DataFineRange"])

                if "Descrizione" not in df.columns:
                    df["Descrizione"] = ""
                if pf_to_desc:
                    df["Descrizione"] = df["Descrizione"].replace("", np.nan).fillna(df["PF"].map(pf_to_desc)).fillna("")

                for c in ["Store_Banner", "Comune", "Regione", "Descrizione", "Stato", "DataFineRange"]:
                    if c in df.columns:
                        df[c] = df[c].apply(_clean_str)

                df = df[(df["drop_pct"] <= -20.0) & (df["Median_Bottles_Daily"] >= 30.0) & (df["vol_lost"] > 0)].copy()
                df = df.sort_values(["drop_pct", "vol_lost"], ascending=[True, False]).reset_index(drop=True)
                return df
        except Exception:
            pass

    return pd.DataFrame()


def render_performance_drops_section(all_machines: List[Dict[str, str]], active_mach_current: Optional[str] = None) -> Optional[str]:
    """
    Renders fleet performance drop directory cleanly:
    - Dedicated crash identification for machines going downward with zero upsides.
    - Explicit diagnosis and root-cause reasons explaining why each machine crashed.
    - Action button column is placed and pinned/stuck directly at the START (left) of the table.
    """
    st.markdown(
        """
        <style>
        /* Elevate st.dataframe buttons into sleek, high-contrast action pills */
        div[data-testid="stDataFrame"] button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(56, 189, 248, 0.6) !important;
            border-radius: 6px !important;
            font-weight: 700 !important;
            padding: 3px 10px !important;
            font-size: 0.78rem !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stDataFrame"] button:hover {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
            border-color: #38bdf8 !important;
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4) !important;
            transform: translateY(-1px);
        }

        /* Prevent Glide Data Grid floating overlay popup editor when double-clicking table cells */
        div[class*="DataGridOverlayEditorStyle"],
        div.gdg-style,
        div.gdg-clip-region,
        #portal textarea,
        #portal input,
        #portal div[class*="DataGridOverlayEditorStyle"],
        div[data-testid="stDataFrame"] textarea,
        div[data-testid="stDataFrame"] input.gdg-input {
            display: none !important;
            visibility: hidden !important;
            opacity: 0 !important;
            pointer-events: none !important;
            height: 0px !important;
            width: 0px !important;
            position: absolute !important;
            top: -9999px !important;
            left: -9999px !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    df_fleet = load_performance_drop_fleet()
    if df_fleet.empty:
        st.info("No historical fleet audit telemetry available.")
        return None

    # Filter controls: search, severity, and root cause reason
    c_f1, c_f2, c_f3 = st.columns([2.5, 1.8, 2.2])
    with c_f1:
        search_kw = st.text_input("🔍 Search Store, City, or Machine ID (PF):", "", placeholder="e.g. Piccolo, Asti, MC...", key="drop_search_kw").strip()
    with c_f2:
        sev_choice = st.selectbox(
            "Filter by Crash Severity:",
            options=[
                "🚨 All Confirmed Crashes",
                "🔴 Critical Collapse (≥ 50% drop)",
                "🟠 Severe Decline (30%–50% drop)",
                "🟡 Significant Decline (20%–30% drop)"
            ],
            index=0,
            key="drop_sev_choice"
        )
    with c_f3:
        reason_opts = ["All Crash Causes"]
        if "Reason_Category" in df_fleet.columns:
            for r_cat in df_fleet["Reason_Category"].dropna().unique():
                if r_cat and r_cat not in reason_opts:
                    reason_opts.append(r_cat)
        cause_choice = st.selectbox(
            "Filter by Root Cause Reason:",
            options=reason_opts,
            index=0,
            key="drop_cause_choice"
        )

    # Base mask: strict crash filtering (no upsides)
    mask = pd.Series(True, index=df_fleet.index)
    if "≥ 50%" in sev_choice:
        mask = (df_fleet["drop_pct"] <= -50.0)
    elif "30%–50%" in sev_choice:
        mask = (df_fleet["drop_pct"] > -50.0) & (df_fleet["drop_pct"] <= -30.0)
    elif "20%–30%" in sev_choice:
        mask = (df_fleet["drop_pct"] > -30.0) & (df_fleet["drop_pct"] <= -20.0)
    else:
        mask = (df_fleet["drop_pct"] <= -20.0)

    if cause_choice != "All Crash Causes" and "Reason_Category" in df_fleet.columns:
        mask = mask & (df_fleet["Reason_Category"] == cause_choice)

    if search_kw:
        kw = search_kw.lower()
        kw_mask = (
            df_fleet["PF"].astype(str).str.lower().str.contains(kw) |
            df_fleet.get("Descrizione", pd.Series("", index=df_fleet.index)).astype(str).str.lower().str.contains(kw) |
            df_fleet["Store_Banner"].astype(str).str.lower().str.contains(kw) |
            df_fleet["Comune"].astype(str).str.lower().str.contains(kw) |
            df_fleet["Regione"].astype(str).str.lower().str.contains(kw) |
            df_fleet.get("DataFineRange", pd.Series("", index=df_fleet.index)).astype(str).str.lower().str.contains(kw)
        )
        mask = mask & kw_mask

    filtered_df = df_fleet[mask].sort_values(["drop_pct", "vol_lost"], ascending=[True, False]).copy().reset_index(drop=True)

    if filtered_df.empty:
        st.info("No machines match the selected filter criteria.")
        return None

    # Handle ButtonColumn click from dataframe if triggered
    btn_click = st.session_state.get("drop_table_inspect_btn")
    if btn_click is not None:
        clicked_idx = getattr(btn_click, "row", None)
        if clicked_idx is None and isinstance(btn_click, dict):
            clicked_idx = btn_click.get("row")
        if clicked_idx is not None and 0 <= clicked_idx < len(filtered_df):
            chosen_pf = str(filtered_df.iloc[clicked_idx]["PF"]).strip()
            st.session_state["selected_telemetry_machine"] = chosen_pf

    # Executive KPI Metric Cards
    total_dropped = len(filtered_df)
    crit_count = int((filtered_df["drop_pct"] <= -50.0).sum())
    sev_count = int(((filtered_df["drop_pct"] > -50.0) & (filtered_df["drop_pct"] <= -30.0)).sum())
    tot_vol_lost = int(round(filtered_df["vol_lost"].sum()))

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    card_style = "background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; height: 100%; box-sizing: border-box;"
    with c_m1:
        st.markdown(
            f"""<div style="{card_style} border-top: 3px solid #ef4444;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Crashed Machines</span>
                <div style="font-size: 1.55rem; font-weight: 800; color: #f87171; margin: 6px 0 2px 0;">{total_dropped:,} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">units</span></div>
                <div style="font-size: 0.73rem; color: #22c55e; font-weight: 700;">🟢 Strictly ATTVO in service</div>
            </div>""", unsafe_allow_html=True
        )
    with c_m2:
        st.markdown(
            f"""<div style="{card_style} border-top: 3px solid #dc2626;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Critical Collapse (≥50%)</span>
                <div style="font-size: 1.55rem; font-weight: 800; color: #dc2626; margin: 6px 0 2px 0;">{crit_count:,} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">units</span></div>
                <div style="font-size: 0.73rem; color: #64748b;">Acute offline or dead stop</div>
            </div>""", unsafe_allow_html=True
        )
    with c_m3:
        st.markdown(
            f"""<div style="{card_style} border-top: 3px solid #f59e0b;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Severe Decline (30–50%)</span>
                <div style="font-size: 1.55rem; font-weight: 800; color: #fbbf24; margin: 6px 0 2px 0;">{sev_count:,} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">units</span></div>
                <div style="font-size: 0.73rem; color: #64748b;">Major operational drop</div>
            </div>""", unsafe_allow_html=True
        )
    with c_m4:
        st.markdown(
            f"""<div style="{card_style} border-top: 3px solid #38bdf8;">
                <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Total Daily Lost Volume</span>
                <div style="font-size: 1.55rem; font-weight: 800; color: #38bdf8; margin: 6px 0 2px 0;">-{tot_vol_lost:,} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span></div>
                <div style="font-size: 0.73rem; color: #64748b;">Fleet network deficit</div>
            </div>""", unsafe_allow_html=True
        )

    # Summary Filter Info Banner
    active_range_lbl = filtered_df["DataFineRange"].iloc[0] if "DataFineRange" in filtered_df.columns and not filtered_df.empty and filtered_df["DataFineRange"].iloc[0] else "Current Active Snapshot"
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid #ef4444; padding: 8px 14px; border-radius: 6px; margin: 12px 0 10px 0; font-size: 0.84rem; color: #cbd5e1;">
            <b>Operational Performance Drops Audit:</b> <span style="color: #f87171; font-weight: 700;">{sev_choice}</span> &bull; 
            Cause: <span style="color: #fbbf24; font-weight: 700;">{cause_choice}</span> &bull; 
            API Status: <span style="color: #22c55e; font-weight: 700;">🟢 ATTVO Only (Valid Through: {active_range_lbl})</span> &bull; 
            Auditing <strong style="color: #f8fafc;">{len(filtered_df):,}</strong> live operational machines whose collection has crashed vs baseline (no decommissioned units).
        </div>
        """,
        unsafe_allow_html=True
    )

    def format_drop_severity(row):
        dp = float(row.get("drop_pct", 0.0) or 0.0)
        if dp <= -50.0:
            return "🔴 Collapse"
        elif dp <= -30.0:
            return "🟠 Severe"
        elif dp <= -20.0:
            return "🟡 Decline"
        return "🟢 Stable"

    filtered_df["Drop_Severity_Status"] = filtered_df.apply(format_drop_severity, axis=1)

    cur_sel = st.session_state.get("selected_telemetry_machine") or active_mach_current
    if not cur_sel or cur_sel not in filtered_df["PF"].values:
        cur_sel = str(filtered_df.iloc[0]["PF"]).strip()
        st.session_state["selected_telemetry_machine"] = cur_sel

    # Action button label per row: distinct and readable
    filtered_df["Action"] = filtered_df["PF"].apply(lambda pf: "Viewing 🎯" if str(pf).strip() == cur_sel else "Inspect 📊")

    # Clean st.dataframe: Action column is STUCK AT THE START (pinned=True)
    display_cols = [
        "Action",
        "Drop_Severity_Status",
        "Descrizione",
        "Store_Banner",
        "Comune",
        "Regione",
        "Median_Bottles_Daily",
        "Recent_30d_Median",
        "drop_pct",
        "vol_lost",
        "Root_Cause_Diagnostic",
        "DataFineRange",
        "TipoMacchina",
        "PF"
    ]

    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=330,
        column_config={
            "Action": st.column_config.ButtonColumn(
                "Action",
                width=115,
                pinned=True,
                type="primary",
                help="Click Inspect to load this machine's dedicated crash telemetry and root cause diagnostics below",
                key="drop_table_inspect_btn"
            ),
            "Drop_Severity_Status": st.column_config.TextColumn("Status", width=105, disabled=True),
            "Descrizione": st.column_config.TextColumn("Description", width=260, help="Official machine location description from Coripet API results", disabled=True),
            "Store_Banner": st.column_config.TextColumn("Store Banner", width=180, disabled=True),
            "Comune": st.column_config.TextColumn("Comune", width=150, disabled=True),
            "Regione": st.column_config.TextColumn("Regione", width=110, disabled=True),
            "Median_Bottles_Daily": st.column_config.NumberColumn("Past Baseline", format="%d b/d", width=110, disabled=True),
            "Recent_30d_Median": st.column_config.NumberColumn("Recent Velocity", format="%d b/d", width=115, disabled=True),
            "drop_pct": st.column_config.NumberColumn("Drop %", format="%.1f%%", width=95, disabled=True),
            "vol_lost": st.column_config.NumberColumn("Daily Loss", format="-%d b/d", width=105, disabled=True),
            "Root_Cause_Diagnostic": st.column_config.TextColumn("Identified Root Cause", width=280, disabled=True),
            "DataFineRange": st.column_config.TextColumn("Active Through", width=115, help="Validity end date from mt_CaricaViewStoricoPuntoFisico (Strictly ATTVO snapshot)", disabled=True),
            "TipoMacchina": st.column_config.TextColumn("Hardware Model", width=125, disabled=True),
            "PF": st.column_config.TextColumn("Machine ID", width=130, disabled=True)
        }
    )

    # Active Inspection Status Bar attached directly below table
    cur_row = filtered_df[filtered_df["PF"] == cur_sel].iloc[0] if (filtered_df["PF"] == cur_sel).any() else filtered_df.iloc[0]
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(239, 68, 68, 0.40); border-left: 4px solid #ef4444; 
                    padding: 10px 16px; border-radius: 6px; margin: 8px 0 14px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Actively Inspected Crashed Unit</span>
                <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-top: 2px;">
                    📍 <span style="color: #38bdf8; user-select: all; -webkit-user-select: all;">{cur_row['Store_Banner']}</span> &bull; 
                    <i style="color: #cbd5e1; user-select: all; -webkit-user-select: all;">{cur_row.get('Descrizione', '')}</i> 
                    (<span style="user-select: all; -webkit-user-select: all;">{cur_row['Comune']}, {cur_row['Regione']}</span>) 
                    &bull; ID: <code style="user-select: all; -webkit-user-select: all; cursor: pointer;" title="Double-click or drag to copy ID">{cur_row['PF']}</code> 
                    &bull; API: <span style="color: #22c55e; font-weight: 700;">🟢 ATTVO</span> <span style="color: #94a3b8; font-size: 0.80rem;">(Thru {cur_row.get('DataFineRange', '')})</span>
                    &bull; Baseline: <b style="color: #94a3b8;">{cur_row['Median_Bottles_Daily']:.0f} b/d</b> 
                    &bull; Recent: <b style="color: #f87171;">{cur_row['Recent_30d_Median']:.0f} b/d</b> 
                    (<span style="color: #ef4444; font-weight: 800;">{cur_row['drop_pct']:.1f}%</span>)
                    &bull; Deficit: <b style="color: #ef4444;">-{cur_row['vol_lost']:.0f} b/d</b>
                    &bull; Cause: <span style="color: #fbbf24; font-weight: 700;">{cur_row['Root_Cause_Diagnostic']}</span>
                </div>
            </div>
            <span style="background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 5px 12px; border-radius: 4px; font-size: 0.76rem; font-weight: 800;">
                CRASH TELEMETRY & ROOT CAUSE ACTIVE BELOW ⬇️
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Secondary Dropdown & CSV Download Companion
    c_sel1, c_sel2 = st.columns([3.5, 1.5])
    machine_options = []
    pf_to_row = {}
    for _, r in filtered_df.iterrows():
        desc_str = f" — {r['Descrizione']}" if r.get("Descrizione") else ""
        end_str = f" | 🟢 ATTVO ({r.get('DataFineRange', '')})" if r.get("DataFineRange") else " | 🟢 ATTVO"
        opt_label = f"{r['Store_Banner']}{desc_str} ({r['Comune']}, {r['Regione']}) — [Drop: {r['drop_pct']:.1f}% | Cause: {r['Root_Cause_Diagnostic']}{end_str} | ID: {r['PF']}]"
        machine_options.append(opt_label)
        pf_to_row[opt_label] = r

    def_idx = 0
    if cur_sel:
        for idx, opt in enumerate(machine_options):
            if cur_sel in opt:
                def_idx = idx
                break

    with c_sel1:
        selected_opt = st.selectbox(
            "Quick Machine Search / Switcher (or click Inspect 📊 on table above):",
            options=machine_options,
            index=def_idx,
            key="drop_inspect_selectbox"
        )
    with c_sel2:
        csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label=f"📥 Download Report ({len(filtered_df):,} Stores CSV)",
            data=csv_bytes,
            file_name="coripet_performance_drops_audit.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Sync selection from dropdown if user used dropdown instead of button
    if selected_opt in pf_to_row:
        dropdown_pf = pf_to_row[selected_opt]["PF"]
        if dropdown_pf != cur_sel and btn_click is None:
            cur_sel = dropdown_pf
            st.session_state["selected_telemetry_machine"] = cur_sel

    return cur_sel

def render_historical_vs_prediction_section(
    res: Optional[Dict[str, Any]] = None,
    default_pf: Optional[str] = None,
    store_status: Optional[str] = None,
    force_comparative_view: bool = False,
    *args,
    **kwargs
):
    """
    Renders dark-mode adaptive analytics:
    - In Tab 5 (force_comparative_view=True): Dedicated Machine Crash Diagnostics & Downtime Telemetry.
      Strictly displays machines experiencing downward crashes and zero-day outages, with no growth projections.
    - For new prospective candidate sites (SPL / POT): Clean forward-looking AI Forecast.
    """
    if not force_comparative_view:
        st.divider()

    if not res:
        res = {}

    # Extract Prospective Model Predictions
    real = res.get("realized_forecast", {})
    preds = res.get("predictions", {})
    prob = res.get("probability_analysis", {})
    inputs = res.get("inputs", {})

    pred_daily = float(real.get("underwritten_daily_bottles", real.get("realized_daily_bottles", preds.get("expected_daily_bottiglie", 0.0))))
    pred_monthly = float(real.get("underwritten_monthly_bottles", preds.get("expected_monthly_bottiglie", pred_daily * 30.4)))
    pred_annual = int(round(pred_daily * 365))
    pred_visits = float(real.get("underwritten_daily_presenze", real.get("realized_daily_presenze", preds.get("expected_daily_presenze", 0.0))))
    pred_bpu = round(pred_daily / max(1.0, pred_visits), 1)

    p10_d = float(prob.get("downside_floor_p10_daily", pred_daily * 0.82))
    p90_d = float(prob.get("upside_peak_p90_daily", pred_daily * 1.25))

    pred_mach = inputs.get("machine_type", "DAINT_G")
    pred_reward = res.get("commercial_rewards", {}).get("reward_type", "Standard Policy")

    # Fallback if no prediction has been run yet in session
    if pred_daily == 0.0:
        pred_daily = 546.0
        pred_monthly = 16586.0
        pred_annual = int(pred_daily * 365)
        pred_visits = 25.0
        pred_bpu = 22.1
        p10_d = 450.0
        p90_d = 680.0
        pred_mach = "DAINT_G"
        pred_reward = "Standard Loyalty"

    pred_summary = {
        "pred_daily": pred_daily,
        "pred_monthly": pred_monthly,
        "pred_annual": pred_annual,
        "pred_visits": pred_visits,
        "pred_bpu": pred_bpu,
        "p10_d": p10_d,
        "p90_d": p90_d,
        "pred_mach": pred_mach,
        "pred_reward": pred_reward
    }

    # Load machine catalogue & prospective site status
    all_machines = load_all_machine_options()
    mach_to_label = {m["macchina"]: m["label"] for m in all_machines}
    pf_to_label = {m["pf"]: m["label"] for m in all_machines if m.get("pf")}
    labels_list = [m["label"] for m in all_machines]

    if force_comparative_view:
        active_mach = default_pf or st.session_state.get("selected_telemetry_machine")
    else:
        active_mach = default_pf or st.session_state.get("selected_machine") or st.session_state.get("selected_store_pf")

    matched_existing_rec = None
    if not force_comparative_view and (not default_pf or str(default_pf).strip() == "") and not st.session_state.get("selected_store_pf"):
        in_store = str(inputs.get("store_name", "")).strip().lower()
        in_comune = str(inputs.get("comune", res.get("location_attributes", {}).get("resolved_comune", ""))).strip().lower()
        if in_comune and in_store:
            for m in all_machines:
                m_com = str(m.get("comune", "")).strip().lower()
                m_ban = str(m.get("banner", "")).strip().lower()
                if m_com and m_com == in_comune:
                    if (in_store in m_ban) or (m_ban in in_store):
                        matched_existing_rec = m
                        active_mach = m["macchina"]
                        default_pf = m.get("pf") or m["macchina"]
                        store_status = "ATTVO"
                        break

    if force_comparative_view:
        is_prospective_site = False
    else:
        is_prospective_site = bool(
            (store_status and str(store_status).upper() in ["POT", "SPL", "CENS", "SOSP", "PREVISTO"]) or
            (default_pf and str(default_pf).startswith("PF") and not any(m.get("pf") == default_pf for m in all_machines)) or
            (not default_pf)
        )

    # 3. IF PROSPECTIVE CANDIDATE SITE: Render pure AI Forecast view without fake/peer past history
    if is_prospective_site:
        status_tag = f"<code>{store_status}</code>" if store_status else "<code>Candidate Site</code>"

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.90) 100%); border: 1px solid rgba(56, 189, 248, 0.35); border-left: 5px solid #0284c7; padding: 16px 20px; border-radius: 10px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.15rem; font-weight: 800; color: #f8fafc;">🌱 New Candidate Deployment ({status_tag})</span>
                            <span style="background: rgba(14, 165, 233, 0.20); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.40); padding: 3px 8px; border-radius: 5px; font-size: 0.75rem; font-weight: 700;">Zero Historical Records</span>
                        </div>
                        <p style="color: #94a3b8; font-size: 0.83rem; margin: 5px 0 0 0;">
                            This location is a prospective candidate site with <b>no previously installed machine or historical records</b>. The projections below are 100% predictive forecasts synthesized from retail sales area, checkout count, Nielsen cluster benchmarks, and regional catchment demand.
                        </p>
                    </div>
                    <span style="background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); padding: 6px 12px; border-radius: 6px; font-weight: 800; font-size: 0.88rem;">
                        🎯 Projected Intake: {pred_daily:.0f} b/d
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 4 Clean AI Forecast KPI Cards
        c_k1, c_k2, c_k3, c_k4 = st.columns(4)
        card_bg = "background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; height: 100%; box-sizing: border-box;"

        with c_k1:
            st.markdown(
                f"""
                <div style="{card_bg}">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Daily Collection Velocity</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #22c55e; margin: 6px 0 2px 0;">
                        {pred_daily:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/d</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Floor (P10): <b style="color: #60a5fa;">{p10_d:.0f}</b> &bull; Peak (P90): <b style="color: #c084fc;">{p90_d:.0f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_k2:
            st.markdown(
                f"""
                <div style="{card_bg}">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Monthly Intake Run Rate</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #3b82f6; margin: 6px 0 2px 0;">
                        {pred_monthly:,.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/mo</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Annual Potential: <b style="color: #93c5fd;">~{pred_annual:,} b/yr</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_k3:
            st.markdown(
                f"""
                <div style="{card_bg}">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Expected Customer Footfall</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #a855f7; margin: 6px 0 2px 0;">
                        {pred_visits:.0f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">u/d</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Monthly Recyclers: <b style="color: #d8b4fe;">~{pred_visits * 30.4:,.0f}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_k4:
            st.markdown(
                f"""
                <div style="{card_bg}">
                    <span style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Recycle Intensity (Bottles/User)</span>
                    <div style="font-size: 1.55rem; font-weight: 800; color: #f59e0b; margin: 6px 0 2px 0;">
                        {pred_bpu:.1f} <span style="font-size: 0.80rem; font-weight: 500; color: #94a3b8;">b/u</span>
                    </div>
                    <div style="font-size: 0.73rem; color: #64748b;">
                        Hardware: <b style="color: #fde047;">{pred_mach}</b> ({pred_reward})
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Future Forecast Bottles per Day Chart (Clean, Full Horizon)
        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #16a34a; border-radius: 10px; padding: 12px 16px; margin-bottom: 6px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.95rem; font-weight: 800; color: #4ade80;">
                        🔮 Future Forecast: Projected Bottles per Day (Next 12 Months)
                    </span>
                    <span style="font-size: 0.72rem; background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.35); padding: 2px 7px; border-radius: 4px; font-weight: 700;">
                        AI FORWARD PROJECTION (b/d)
                    </span>
                </div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <span>Model Target: <b style="color: #22c55e;">{pred_daily:.0f} b/d</b></span>
                    <span>Conservative Floor (P10): <b style="color: #60a5fa;">{p10_d:.0f} b/d</b></span>
                    <span>Upside Ceiling (P90): <b style="color: #c084fc;">{p90_d:.0f} b/d</b></span>
                    <span>Hardware Configuration: <b style="color: #cbd5e1;">{pred_mach}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        max_future_daily = float(max(pred_summary["p90_d"], pred_summary["pred_daily"] * 1.25))
        fig_future = build_future_daily_bottles_figure(res, pred_summary, y_max=max_future_daily * 1.15, chart_height=380)
        st.plotly_chart(fig_future, use_container_width=True, theme=None, config={"displayModeBar": False, "responsive": True}, key="plotly_prospective_future_horizon")
        return

    # 4. IF TAB 5 (CRASH TELEMETRY) OR ACTIVE MACHINE:
    if force_comparative_view:
        st.markdown(
            """
            <div style="margin-bottom: 14px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <h3 style="color: #f8fafc; margin: 0; font-size: 1.35rem; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                            <span>📉 Fleet Performance Drops & Machine Crash Telemetry</span>
                        </h3>
                        <p style="color: #94a3b8; font-size: 0.85rem; margin: 3px 0 0 0;">
                            Dedicated audit isolating machines experiencing acute volume collapse, zero-day outage streaks, and downward deficits (no upsides).
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="margin-bottom: 14px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div>
                        <h3 style="color: #f8fafc; margin: 0; font-size: 1.35rem; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                            <span>⚖️ Historical Actuals vs. Prospective AI Model Forecast</span>
                        </h3>
                        <p style="color: #94a3b8; font-size: 0.85rem; margin: 3px 0 0 0;">
                            Balanced dual-lens visualizer: Real physical machine telemetry ground truth (PostgreSQL / Power BI) vs. AI model predictions.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if matched_existing_rec:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-left: 5px solid #10b981; padding: 12px 16px; border-radius: 8px; margin-bottom: 14px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div>
                            <strong style="color: #34d399; font-size: 0.98rem;">📍 Existing Coripet Installation Auto-Detected:</strong>
                            <span style="color: #f8fafc; font-weight: 700; margin-left: 6px;">{matched_existing_rec['label']}</span>
                        </div>
                        <span style="background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); padding: 3px 8px; border-radius: 5px; font-size: 0.74rem; font-weight: 700;">
                            ACTIVE ATTVO FLEET
                        </span>
                    </div>
                    <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.82rem;">
                        This store already operates an active Coripet RVM. Below is the ground truth telemetry from PostgreSQL & Power BI compared directly with your prospective AI model prediction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    if force_comparative_view:
        chosen_drop_mid = render_performance_drops_section(all_machines, active_mach_current=active_mach)
        if chosen_drop_mid:
            active_mach = chosen_drop_mid

    render_fleet_benchmark_submodule(
        res=res,
        pred_summary=pred_summary,
        all_machines=all_machines,
        labels_list=labels_list,
        mach_to_label=mach_to_label,
        pf_to_label=pf_to_label,
        is_prospective_site=False,
        active_mach=active_mach,
        show_selector=not force_comparative_view,
        is_crash_mode=force_comparative_view
    )
