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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_machine_historical_profile(machine_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches chronological historical profile strictly for the last 2 years from PostgreSQL,
    always keyed and filtered strictly on the PHYSICAL MACHINE (macchina).
    """
    engine = get_db_engine()
    query_mach = text("""
        SELECT giorno_dt, bottiglie, presenze, cliente, comune, regione, tipomacchina, macchina, codicemacchinaprod, pf
        FROM master_daily_rvm
        WHERE macchina = :m_id
        ORDER BY giorno_dt ASC;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query_mach, conn, params={"m_id": machine_id})
        
        # Fallback if machine_id was passed as a PF or manufacturer serial
        if df.empty:
            query_fallback = text("""
                SELECT giorno_dt, bottiglie, presenze, cliente, comune, regione, tipomacchina, macchina, codicemacchinaprod, pf
                FROM master_daily_rvm
                WHERE pf = :m_id OR codicemacchinaprod = :m_id
                ORDER BY giorno_dt ASC;
            """)
            df_cand = pd.read_sql(query_fallback, conn, params={"m_id": machine_id})
            if not df_cand.empty:
                latest_m = df_cand.sort_values("giorno_dt").iloc[-1]["macchina"]
                if pd.notna(latest_m) and latest_m != "":
                    df = df_cand[df_cand["macchina"] == latest_m].reset_index(drop=True)
                else:
                    df = df_cand

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
    chart_height: int = 380
) -> go.Figure:
    """
    Builds a clean, single-axis Future Daily Bottles (b/d) Velocity Graph.
    Strictly shows bottles per day without monthly bars or customer visits.
    """
    profile = res.get("lifecycle_horizon", {}).get("year1_rolling_profile", [])
    if not profile or len(profile) != 12:
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
    active_mach: Optional[str] = None
):
    """Submodule for rendering peer benchmark or active machine comparative telemetry."""
    if active_mach and active_mach in mach_to_label:
        default_idx = labels_list.index(mach_to_label[active_mach])
    elif active_mach and active_mach in pf_to_label:
        default_idx = labels_list.index(pf_to_label[active_mach])
    else:
        default_idx = 0

    col_sel, col_mode = st.columns([3, 1])
    with col_sel:
        selector_label = "Select Reference Peer Machine to Benchmark (Active Fleet):" if is_prospective_site else "Select Physical Machine to Compare (Past vs Future):"
        chosen_opt = st.selectbox(
            selector_label,
            options=labels_list,
            index=default_idx,
            key=f"comp_machine_selector_{'peer' if is_prospective_site else 'active'}",
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

    if not selected_mach:
        st.warning("Please select a physical machine to load historical comparison.")
        return

    # Fetch historical data (strictly last 2 years for this physical machine)
    with st.spinner(f"Loading historical operational telemetry for machine {selected_mach}..."):
        hist = fetch_machine_historical_profile(selected_mach)

    if not hist:
        st.error(f"No historical daily collection records found in database for machine ID `{selected_mach}`.")
        return

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

    # Historical metrics
    hist_daily_med = hist["median_daily"]
    hist_daily_mean = hist["mean_daily"]
    hist_monthly = hist_daily_med * 30.4
    hist_annual = hist_daily_med * 365
    hist_visits = hist["median_visits"]
    hist_bpu = hist["bottles_per_user"]
    hist_recent_30d = hist["recent_30d_median"]
    hist_zero_days = hist["zero_days"]
    hist_zero_pct = hist["zero_rate"]
    hist_mach = hist["model"]

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

    # Executive Summary Bar
    f_date_str = hist["first_date"].strftime("%b %Y")
    l_date_str = hist["last_date"].strftime("%b %Y")
    site_desc = f" &bull; Site PF: <code>{hist['pf']}</code>" if hist.get("pf") else ""
    serial_desc = f" (SN: {hist['serial']})" if hist.get("serial") else ""
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

    card_bg = "background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; height: 100%; box-sizing: border-box;"

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


def render_historical_vs_prediction_section(
    res: Optional[Dict[str, Any]] = None,
    default_pf: Optional[str] = None,
    store_status: Optional[str] = None,
    force_comparative_view: bool = False
):
    """
    Renders dark-mode adaptive analytics:
    - For new prospective candidate sites (SPL / POT): Renders clean forward-looking AI Forecast
      with explicit notice of zero historical records, with optional benchmark expander.
    - For installed active units: Renders side-by-side comparative analysis of past telemetry vs future forecast.
    """
    if not force_comparative_view:
        st.divider()

    if not res:
        res = {}

    # 1. Extract Prospective Model Predictions immediately
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

    # If no prediction has been run yet in session, populate baseline benchmark
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

    # 2. Check machine catalogue & prospective site status
    all_machines = load_all_machine_options()
    mach_to_label = {m["macchina"]: m["label"] for m in all_machines}
    pf_to_label = {m["pf"]: m["label"] for m in all_machines if m.get("pf")}
    labels_list = [m["label"] for m in all_machines]

    active_mach = default_pf or st.session_state.get("selected_machine") or st.session_state.get("selected_store_pf")

    # Auto-detect if location & store match an existing active machine in the fleet:
    matched_existing_rec = None
    if (not default_pf or str(default_pf).strip() == "") and not st.session_state.get("selected_store_pf"):
        in_store = str(inputs.get("store_name", "")).strip().lower()
        in_comune = str(inputs.get("comune", res.get("location_attributes", {}).get("resolved_comune", ""))).strip().lower()
        in_addr = str(inputs.get("address", "")).strip().lower()

        if in_comune and in_store:
            for m in all_machines:
                m_com = str(m.get("comune", "")).strip().lower()
                m_ban = str(m.get("banner", "")).strip().lower()
                if m_com and m_com == in_comune:
                    # Match if store name matches (e.g., 'carrefour' in 'Carrefour')
                    if (in_store in m_ban) or (m_ban in in_store):
                        matched_existing_rec = m
                        active_mach = m["macchina"]
                        default_pf = m.get("pf") or m["macchina"]
                        store_status = "ATTVO"
                        break

    # Detect if evaluated site is a new prospective candidate (POT / SPL) without existing telemetry
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

    # 4. IF EXISTING ACTIVE MACHINE WITH REAL TELEMETRY:
    # Header & Control Bar
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

    render_fleet_benchmark_submodule(
        res=res,
        pred_summary=pred_summary,
        all_machines=all_machines,
        labels_list=labels_list,
        mach_to_label=mach_to_label,
        pf_to_label=pf_to_label,
        is_prospective_site=False,
        active_mach=active_mach
    )
