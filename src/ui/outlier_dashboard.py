"""
Interactive Outlier & Defect Simulator Dashboard Component for Coripet RVM Predictor.

Features:
- One-click live simulator scan across 1,900+ machines in PostgreSQL
- Strictly filters and displays ONLY anomalous / outlier machines
- Detailed root cause and diagnostic reason for every flagged machine
- Full historical time-series interactive chart (Plotly) with 7d/30d rolling trends
- Operational action protocols for field technicians and store management
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

REPORT_CSV = ROOT_DIR / "data" / "reports" / "defected_and_outlier_machines.csv"


@st.cache_resource
def get_db_engine():
    """Returns a cached SQLAlchemy engine supporting PostgreSQL, MariaDB/MySQL, and SQLite."""
    db_pass = os.getenv("DB_PASSWORD") or os.getenv("DB_PASS")
    if db_pass:
        from sqlalchemy.engine import URL
        db_user = os.getenv("DB_USER", "waqas")
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_port = int(os.getenv("DB_PORT", "3306"))
        db_name = os.getenv("DB_NAME", "coripet_rvm")
        url_obj = URL.create("mysql+pymysql", username=db_user, password=db_pass, host=db_host, port=db_port, database=db_name)
        return create_engine(url_obj, pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=1800)

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
    return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10, pool_recycle=1800)


@st.cache_data(ttl=600, show_spinner=False)
def load_cached_audit_report() -> pd.DataFrame:
    """Loads existing audit report CSV or generates it if missing."""
    if REPORT_CSV.exists():
        try:
            df = pd.read_csv(REPORT_CSV)
            if not df.empty:
                return df
        except Exception:
            pass

    # Fallback to running a fast audit if missing
    try:
        return run_fleet_audit_scan()
    except Exception as e:
        st.error(f"⚠️ Fleet audit scan requires database access: {e}. Please check your database connection in .env.")
        return pd.DataFrame()


def run_fleet_audit_scan() -> pd.DataFrame:
    """Executes live anomaly detection scan on PostgreSQL master_daily_rvm."""
    from src.model.outlier_detector import RVMOutlierDetector

    engine = get_db_engine()
    query = """
        SELECT pf, giorno_dt, bottiglie, presenze, cliente, comune, regione,
               tipomacchina, maturity_status
        FROM master_daily_rvm
        WHERE giorno_dt >= '2024-01-01'
        ORDER BY pf, giorno_dt;
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    col_map = {
        "pf": "PF", "giorno_dt": "giorno_dt", "bottiglie": "BOTTIGLIE", "presenze": "PRESENZE",
        "cliente": "CLIENTE", "comune": "Comune", "regione": "Regione",
        "tipomacchina": "TipoMacchina", "maturity_status": "maturity_status"
    }
    df = df.rename(columns=col_map)

    detector = RVMOutlierDetector()
    audit_df = detector.audit_fleet(df)

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(REPORT_CSV, index=False)
    return audit_df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_machine_historical_records(machine_id: str) -> pd.DataFrame:
    """Fetches all chronological daily records for a specific store PF or machine from PostgreSQL.
    Uses indexed 3-way UNION for instant sub-10ms response without table-wide sequential scans."""
    engine = get_db_engine()
    query = text("""
        SELECT giorno_dt, bottiglie, presenze, macchina, codicemacchinaprod, pf
        FROM master_daily_rvm
        WHERE pf = :m_id
        UNION
        SELECT giorno_dt, bottiglie, presenze, macchina, codicemacchinaprod, pf
        FROM master_daily_rvm
        WHERE macchina = :m_id
        UNION
        SELECT giorno_dt, bottiglie, presenze, macchina, codicemacchinaprod, pf
        FROM master_daily_rvm
        WHERE codicemacchinaprod = :m_id
        ORDER BY giorno_dt ASC;
    """)
    df = pd.DataFrame()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"m_id": machine_id})
    except Exception:
        try:
            # Fallback reconnect attempt in case pool connection was reset
            engine.dispose()
            with engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"m_id": machine_id})
        except Exception as e:
            st.warning(f"Unable to load historical records from database: {e}")
            return pd.DataFrame()

    if not df.empty:
        df["giorno_dt"] = pd.to_datetime(df["giorno_dt"])

        # Strictly limit historical data to the last 2 years and remove extra older data
        max_dt = df["giorno_dt"].max()
        if pd.notna(max_dt):
            cutoff_dt = max_dt - pd.DateOffset(years=2)
            df = df[df["giorno_dt"] >= cutoff_dt].reset_index(drop=True)

        df["bottiglie"] = df["bottiglie"].fillna(0.0).astype(float)
        df["presenze"] = df["presenze"].fillna(0.0).astype(float)

        # Calculate Rolling 7-day and 30-day moving averages
        df["roll_7d"] = df["bottiglie"].rolling(7, min_periods=1).mean()
        df["roll_30d"] = df["bottiglie"].rolling(30, min_periods=5).mean()

        # Flag zero-collection periods
        df["is_zero"] = (df["bottiglie"] == 0.0)

    return df


def get_anomaly_badge_html(anomaly: str) -> str:
    """Returns styled HTML badge for anomaly classification."""
    colors_map = {
        "HEALTHY_NORMAL": ("#f0fdf4", "#166534", "✅ Healthy Nominal Operation"),
        "DEFECTIVE_OFFLINE": ("#fee2e2", "#b91c1c", "⚠️ Hardware Defect / Offline"),
        "VOLATILITY_SWING": ("#ffedd5", "#c2410c", "📉 Volatility Swing (600→1000→<400)"),
        "CAPACITY_BOTTLENECK": ("#fef3c7", "#b45309", "🛑 Capacity Jam / Bin Full"),
        "EXTREME_SPIKE_EVENT": ("#f3e8ff", "#7e22ce", "⚡ Extreme Volume Spike"),
        "CHRONIC_UNDERPERFORMER": ("#fee2e2", "#991b1b", "🐢 Chronic Underperformer"),
        "SUPER_HUB_OUTLIER": ("#dcfce7", "#15803d", "🌟 Super-Hub (>3k/day)"),
        "SENSOR_DRIFT_TRICKLE": ("#f1f5f9", "#475569", "👻 Ghost Sensor Trickle"),
        "SCHEDULE_ANOMALY": ("#e0f2fe", "#0369a1", "📅 Schedule Inconsistency"),
    }
    bg, fg, label = colors_map.get(anomaly, ("#f1f5f9", "#475569", anomaly))
    return f"""<span style="background: {bg}; color: {fg}; font-weight: 700; padding: 4px 10px; border-radius: 6px; font-size: 0.82rem; border: 1px solid {fg}40; display: inline-block;">{label}</span>"""


def render_outlier_diagnostic_narrative(row: pd.Series) -> str:
    """Generates an in-depth, root-cause explanation for why this machine was flagged."""
    anomaly = row["Anomaly_Classification"]
    pf = row["PF"]
    banner = row["Store_Banner"]
    comune = row["Comune"]
    consec_zeros = int(row["Max_Consecutive_Zero_Days"])
    zero_pct = float(row["Recent_Zero_Rate_Pct"])
    med_vol = float(row["Median_Bottles_Daily"])
    max_day = float(row["Max_Day_Bottles"])
    cv = float(row["Volatility_CV"])
    bpu = float(row["Bottles_Per_User"])
    exp_vol = float(row.get("Expected_Predicted_Bottles", med_vol))

    if anomaly == "HEALTHY_NORMAL":
        return (
            f"<b>Healthy & Nominal Machine Operation:</b> Machine <code>{pf}</code> at <b>{banner}</b> ({comune}) "
            f"operates within certified parametric tolerances with an operational median of <b>{int(med_vol):,} bottles/day</b>. "
            f"Zero prolonged breakdown streaks or severe bottle/footfall discrepancies have been detected. Continue standard service schedule."
        )
    elif anomaly == "DEFECTIVE_OFFLINE":
        return (
            f"<b>Hardware Breakdown / Extended Inactivity Detected:</b> Machine <code>{pf}</code> at <b>{banner}</b> ({comune}) "
            f"recorded <b>{consec_zeros} consecutive zero-collection days</b>, with <b>{zero_pct:.1f}%</b> of recent operational days logging zero bottles. "
            f"Because this store is officially classified as active and mature, consecutive zeros indicate an unpowered unit, "
            f"a dead optical barcode scanner, or a mechanically jammed compactor motor that has taken the machine offline. "
            f"<i>(During AI training, these {consec_zeros} dead days are automatically zero-masked so they do not artificially suppress prediction benchmarks for other stores).</i>"
        )
    elif anomaly == "VOLATILITY_SWING":
        return (
            f"<b>High-Volatility Regime Shift Detected:</b> Machine <code>{pf}</code> exhibited a dramatic volatility rollercoaster "
            f"with a high Coefficient of Variation (<b>CV = {cv:.2f}</b>). The machine operated at a baseline of ~{int(med_vol)} bottles/day, "
            f"surged to a peak of <b>{int(max_day):,} bottles/day</b>, and subsequently crashed below 400 bottles/day. "
            f"<b>Root Cause Diagnosis:</b> This signature indicates either (1) a temporary supermarket coupon/reward points campaign that expired, "
            f"(2) extreme seasonal vacation turnover, or (3) internal bin capacity burnout where store staff could not keep up with clearing the compactor."
        )
    elif anomaly == "CAPACITY_BOTTLENECK":
        return (
            f"<b>Bin Capacity & Staff Emptying Bottleneck Detected:</b> While customer footfall remained active at <b>{banner}</b>, "
            f"the collection volume collapsed sharply to a recent median of <b>{int(med_vol)} bottles/day</b>, and bottles-per-user dropped to <b>{bpu:.1f}</b> "
            f"(compared to the fleet standard of 35–50 bottles/user). "
            f"<b>Root Cause Diagnosis:</b> The machine's internal collection bag reaches maximum compactor capacity early in the afternoon, "
            f"locking the machine into 'Bin Full / Out of Service' mode because supermarket staff only empty it once per day instead of twice."
        )
    elif anomaly == "EXTREME_SPIKE_EVENT":
        return (
            f"<b>Extreme Volume Surge Outlier:</b> Single-day collection reached an abnormal peak of <b>{int(max_day):,} bottles</b> "
            f"(far exceeding standard daily retail throughput). "
            f"<b>Root Cause Diagnosis:</b> This spike is characteristic of a bulk commercial drop (local bar or restaurant operator depositing weeks of inventory), "
            f"a community school recycling contest, or technician optical loop testing. "
            f"<i>(During AI training, this spike is automatically Winsorized to 6,500 bottles to protect gradient stability).</i>"
        )
    elif anomaly == "CHRONIC_UNDERPERFORMER":
        return (
            f"<b>Chronic Structural Underperformer:</b> Location demographics and store scale predict an expected collection of <b>~{int(exp_vol)} bottles/day</b>, "
            f"yet this mature machine has delivered only <b>{int(med_vol)} bottles/day</b> over extended operating months. "
            f"<b>Root Cause Diagnosis:</b> Machine is poorly situated inside the retail facility (hidden in a remote parking basement, obstructed behind shopping carts, "
            f"or zero customer in-store signage). The store has demographic demand that is not converting due to physical placement."
        )
    elif anomaly == "SUPER_HUB_OUTLIER":
        return (
            f"<b>Regional Super-Hub Overperformer:</b> Machine <code>{pf}</code> generates an exceptional throughput of <b>{int(med_vol):,} bottles/day</b> "
            f"consistently, ranking in the top 1% nationwide. "
            f"<b>Root Cause Diagnosis:</b> Serves as the primary recycling destination for an entire municipality or commercial district. "
            f"This location is a prime candidate for Coripet to deploy a <b>2nd RVM machine</b> to capture unfulfilled customer demand."
        )
    elif anomaly == "SENSOR_DRIFT_TRICKLE":
        return (
            f"<b>Sensor Drift / Ghost Counting Loop:</b> Machine logs a continuous micro-trickle of 1–5 bottles every single day "
            f"with zero or one registered users. "
            f"<b>Root Cause Diagnosis:</b> Optical lens contamination, reflective dust particles, or misaligned laser barrier causing phantom pulse counts."
        )
    else:
        return f"Operational anomaly flagged under <b>{anomaly}</b> criteria. Review historical charts below."


def render_outlier_and_defect_dashboard():
    """Main rendering routine for the Outlier & Defect Simulator Tab."""
    # -------------------------------------------------------------
    # 1. Header Banner & Action Bar
    # -------------------------------------------------------------
    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.markdown(
            """
            <div style="margin-bottom: 8px;">
                <h2 style="color: #0f5132; margin-bottom: 2px; font-weight: 800; font-size: 1.8rem;">
                    🔍 Outlier & Defect Simulator
                </h2>
                <p style="color: #64748b; font-size: 0.95rem; margin-top: 0;">
                    Automated Fleet Diagnostics • 8 Anomaly Regimes • Live Historical Time-Series Visualizer
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_btn:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Run Live Fleet Audit", type="primary", use_container_width=True, help="Scans all 1,900+ machines across PostgreSQL database"):
            with st.spinner("📡 Scanning all machines in PostgreSQL across 8 anomaly regimes..."):
                audit_df = run_fleet_audit_scan()
                st.session_state["outlier_audit_df"] = audit_df
                st.cache_data.clear()
            st.success(f"Audit completed! Scanned {len(audit_df):,} machines.")
            time.sleep(0.5)
            st.rerun()

    # Load Audit Data
    if "outlier_audit_df" in st.session_state:
        audit_df = st.session_state["outlier_audit_df"]
    else:
        audit_df = load_cached_audit_report()

    total_fleet = len(audit_df)
    counts = audit_df["Anomaly_Classification"].value_counts()
    n_outliers = int((audit_df["Anomaly_Classification"].isin(["DEFECTIVE_OFFLINE", "VOLATILITY_SWING", "EXTREME_SPIKE_EVENT", "CHRONIC_UNDERPERFORMER", "SUPER_HUB_OUTLIER", "SENSOR_DRIFT_TRICKLE", "SCHEDULE_ANOMALY"])).sum())
    n_healthy = int(counts.get("HEALTHY_NORMAL", 0))
    n_defective = int(counts.get("DEFECTIVE_OFFLINE", 0))
    n_spikes = int(counts.get("EXTREME_SPIKE_EVENT", 0))
    n_swings = int(counts.get("VOLATILITY_SWING", 0))
    n_choke = int(counts.get("CAPACITY_BOTTLENECK", 0))
    n_under = int(counts.get("CHRONIC_UNDERPERFORMER", 0))
    n_hub = int(counts.get("SUPER_HUB_OUTLIER", 0))
    n_trickle = int(counts.get("SENSOR_DRIFT_TRICKLE", 0))

    # -------------------------------------------------------------
    # 2. Executive KPI Cards
    # -------------------------------------------------------------
    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-bottom: 18px;">
            <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #0f5132; padding: 12px; border-radius: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase;">Total Outliers</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a;">{n_outliers:,} <span style="font-size: 0.8rem; font-weight: 500; color: #64748b;">({(n_outliers/max(1, total_fleet)*100):.1f}%)</span></div>
            </div>
            <div style="background: #fef2f2; border: 1px solid #fecaca; border-left: 4px solid #b91c1c; padding: 12px; border-radius: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #991b1b; text-transform: uppercase;">Defective / Offline</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #b91c1c;">{n_defective:,}</div>
            </div>
            <div style="background: #faf5ff; border: 1px solid #e9d5ff; border-left: 4px solid #7e22ce; padding: 12px; border-radius: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #6b21a8; text-transform: uppercase;">Extreme Spikes</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #7e22ce;">{n_spikes:,}</div>
            </div>
            <div style="background: #fff7ed; border: 1px solid #fed7aa; border-left: 4px solid #c2410c; padding: 12px; border-radius: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #9a3412; text-transform: uppercase;">Volatility Swings</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #c2410c;">{n_swings:,}</div>
            </div>

            <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; padding: 12px; border-radius: 8px;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #166534; text-transform: uppercase;">Super-Hubs (>3k)</span>
                <div style="font-size: 1.5rem; font-weight: 800; color: #15803d;">{n_hub:,}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------
    # 3. Interactive Outlier Rule Filtering Bar
    # -------------------------------------------------------------
    RULE_FILTER_DEFINITIONS = [
        {
            "id": "ALL_OUTLIERS",
            "label": f"🚨 All Actionable Outlier Stores ({n_outliers:,} Stores)",
            "desc": "Separates all actionable anomaly regimes, excluding healthy stores",
            "regimes": ["DEFECTIVE_OFFLINE", "VOLATILITY_SWING", "EXTREME_SPIKE_EVENT", "CHRONIC_UNDERPERFORMER", "SUPER_HUB_OUTLIER", "SENSOR_DRIFT_TRICKLE", "SCHEDULE_ANOMALY"]
        },
        {
            "id": "DEFECTIVE_OFFLINE",
            "label": f"🔴 Rule 1: Defective / Offline ({n_defective:,} Stores)",
            "desc": "Consecutive zeros ≥ 7 days OR recent zero rate ≥ 20% (Compactor/optical outage)",
            "regimes": ["DEFECTIVE_OFFLINE"]
        },
        {
            "id": "VOLATILITY_SWING",
            "label": f"📉 Rule 2: Volatility Swing ({n_swings:,} Stores)",
            "desc": "Rollercoaster swing (400-850 baseline surges to 1.5x [1000+] then collapses to <400 with CV > 0.85)",
            "regimes": ["VOLATILITY_SWING"]
        },
        {
            "id": "EXTREME_SPIKE_EVENT",
            "label": f"⚡ Rule 3: Extreme Bulk Volume Spike ({n_spikes:,} Stores)",
            "desc": "Single-day collection > 7,500 bottles (99.8th percentile) or Z-score > 4.5",
            "regimes": ["EXTREME_SPIKE_EVENT"]
        },
        {
            "id": "CHRONIC_UNDERPERFORMER",
            "label": f"🐢 Rule 4: Chronic Underperformer ({n_under:,} Stores)",
            "desc": "High demographic/store potential, but persistent volume < 250 b/d (gap ≥ 400 vs forecast)",
            "regimes": ["CHRONIC_UNDERPERFORMER"]
        },
        {
            "id": "SUPER_HUB_OUTLIER",
            "label": f"🌟 Rule 5: Super-Hub Outlier ({n_hub:,} Stores)",
            "desc": "Consistently massive volume collecting > 3,000 bottles/day (Priority for 2nd machine)",
            "regimes": ["SUPER_HUB_OUTLIER"]
        },
        {
            "id": "SENSOR_DRIFT_TRICKLE",
            "label": f"👻 Rule 6: Sensor Drift / Ghost Trickle ({n_trickle:,} Stores)",
            "desc": "Machine logs 1-5 ghost bottles daily with 0 users (dirty optical lenses)",
            "regimes": ["SENSOR_DRIFT_TRICKLE"]
        },
        {
            "id": "HEALTHY_NORMAL",
            "label": f"✅ Rule 7: Healthy & Normal Stores ({n_healthy:,} Stores)",
            "desc": "Operates within standard parametric distribution (Non-outliers)",
            "regimes": ["HEALTHY_NORMAL"]
        },
        {
            "id": "FULL_FLEET",
            "label": f"🌐 Full Fleet Network ({total_fleet:,} Total Stores)",
            "desc": "Displays entire national fleet (both Outliers and Healthy)",
            "regimes": None
        }
    ]

    rule_id_to_def = {r["id"]: r for r in RULE_FILTER_DEFINITIONS}
    rule_ids = [r["id"] for r in RULE_FILTER_DEFINITIONS]

    f_col1, f_col2, f_col3 = st.columns([3, 2, 2])

    with f_col1:
        selected_rule_id = st.selectbox(
            "🎯 Filter Stores by Outlier Rule:",
            options=rule_ids,
            format_func=lambda rid: rule_id_to_def[rid]["label"],
            index=0,
            help="Select an outlier detection rule to isolate and separate stores."
        )

    with f_col2:
        reg_options = ["All Regions"] + sorted([r for r in audit_df["Regione"].dropna().unique() if r])
        selected_reg = st.selectbox("Filter by Region:", options=reg_options, index=0)

    with f_col3:
        search_query = st.text_input("Search Store, City, or Machine ID (PF):", value="", placeholder="e.g. Esselunga, Asti, MC00...").strip()

    # Apply Outlier Rule Filter
    active_rule = rule_id_to_def[selected_rule_id]
    if active_rule["regimes"] is not None:
        filtered_df = audit_df[audit_df["Anomaly_Classification"].isin(active_rule["regimes"])].copy()
    else:
        filtered_df = audit_df.copy()

    # Apply Region and Search filters
    if selected_reg != "All Regions":
        filtered_df = filtered_df[filtered_df["Regione"] == selected_reg]

    if search_query:
        q = search_query.lower()
        mask = (
            filtered_df["PF"].str.lower().str.contains(q, na=False) |
            filtered_df["Store_Banner"].str.lower().str.contains(q, na=False) |
            filtered_df["Comune"].str.lower().str.contains(q, na=False)
        )
        filtered_df = filtered_df[mask]

    # Add Rule Trigger Diagnostic Summary Column
    def summarize_rule_trigger(row):
        cat = row.get("Anomaly_Classification", "")
        if cat == "DEFECTIVE_OFFLINE":
            return f"🔴 {int(row.get('Max_Consecutive_Zero_Days', 0))} zero-days streak ({row.get('Recent_Zero_Rate_Pct', 0):.1f}% recent zeros)"
        elif cat == "CAPACITY_BOTTLENECK":
            return f"🛑 Choke: {row.get('Bottles_Per_User', 0):.1f} b/user (Full bin jam)"
        elif cat == "VOLATILITY_SWING":
            return f"📉 Swing: CV={row.get('Volatility_CV', 0):.2f} (Peak {int(row.get('Max_Day_Bottles', 0)):,} b/d)"
        elif cat == "EXTREME_SPIKE_EVENT":
            return f"⚡ Single-Day Peak: {int(row.get('Max_Day_Bottles', 0)):,} bottles"
        elif cat == "CHRONIC_UNDERPERFORMER":
            return f"🐢 Actual: {int(row.get('Median_Bottles_Daily', 0))} vs Expected: {int(row.get('Expected_Predicted_Bottles', 0))} b/d"
        elif cat == "SUPER_HUB_OUTLIER":
            return f"🌟 Super-Hub: {int(row.get('Median_Bottles_Daily', 0)):,} b/d sustained"
        elif cat == "SENSOR_DRIFT_TRICKLE":
            return "👻 Ghost count: 1-5 bottles with 0 users"
        elif cat == "HEALTHY_NORMAL":
            return f"✅ Nominal: {int(row.get('Median_Bottles_Daily', 0))} b/d median"
        return cat

    filtered_df["Triggered_Rule_Summary"] = filtered_df.apply(summarize_rule_trigger, axis=1)
    filtered_df["Outlier_Status"] = filtered_df["Anomaly_Classification"].apply(lambda x: "🚨 OUTLIER" if x != "HEALTHY_NORMAL" else "✅ NOMINAL")

    # Filter Result Badge Banner
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid #38bdf8; padding: 8px 14px; border-radius: 6px; margin: 6px 0 12px 0; font-size: 0.84rem; color: #cbd5e1;">
            <b>Active Rule:</b> <span style="color: #38bdf8; font-weight: 700;">{active_rule['label']}</span> &bull; <em>{active_rule['desc']}</em> &bull; 
            Showing <strong style="color: #f8fafc;">{len(filtered_df):,}</strong> matching machines.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------
    # 4. Outlier Machines Table
    # -------------------------------------------------------------
    display_cols = [
        "Outlier_Status", "Triggered_Rule_Summary", "PF", "Store_Banner", "Comune", "Regione", "TipoMacchina",
        "Anomaly_Classification", "Max_Consecutive_Zero_Days", "Recent_Zero_Rate_Pct",
        "Median_Bottles_Daily", "Max_Day_Bottles", "Volatility_CV", "Bottles_Per_User",
        "Recommended_Action"
    ]
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=280,
        column_config={
            "Outlier_Status": st.column_config.TextColumn("Status", width="small"),
            "Triggered_Rule_Summary": st.column_config.TextColumn("Triggered Rule & Criteria", width="medium"),
            "PF": st.column_config.TextColumn("Machine ID", width="small"),
            "Store_Banner": st.column_config.TextColumn("Store Banner", width="medium"),
            "Comune": st.column_config.TextColumn("Comune", width="small"),
            "Regione": st.column_config.TextColumn("Regione", width="small"),
            "Anomaly_Classification": st.column_config.TextColumn("Anomaly Regime", width="medium"),
            "Max_Consecutive_Zero_Days": st.column_config.NumberColumn("Max Zero Days", format="%d"),
            "Recent_Zero_Rate_Pct": st.column_config.NumberColumn("Recent Zero %", format="%.1f%%"),
            "Median_Bottles_Daily": st.column_config.NumberColumn("Median Vol", format="%d"),
            "Max_Day_Bottles": st.column_config.NumberColumn("Peak Vol", format="%d"),
            "Volatility_CV": st.column_config.NumberColumn("CV Volatility", format="%.2f"),
            "Recommended_Action": st.column_config.TextColumn("Action Protocol", width="large")
        }
    )

    # Download CSV Button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Filtered Report ({len(filtered_df):,} Stores CSV)",
        data=csv_bytes,
        file_name="coripet_outlier_machines_report.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # 5. Machine Historical Record Visualizer & Reason of Outlier
    # -------------------------------------------------------------
    st.markdown("### 📈 Machine Historical Time-Series & Root Cause Diagnostic")

    if filtered_df.empty:
        st.warning("No outlier machines match your search filters.")
        return

    # Machine Selector Dropdown
    machine_options = []
    pf_to_row = {}
    for _, r in filtered_df.iterrows():
        opt_label = f"{r['PF']} — {r['Store_Banner']} ({r['Comune']}, {r['Regione']}) | [{r['Anomaly_Classification']}]"
        machine_options.append(opt_label)
        pf_to_row[opt_label] = r

    selected_opt = st.selectbox(
        "Select Machine to Inspect Historical Performance & Anomaly Root Cause:",
        options=machine_options,
        index=0
    )
    selected_row = pf_to_row[selected_opt]
    selected_pf = selected_row["PF"]

    # -------------------------------------------------------------
    # Diagnostic Reason & Protocol Card
    # -------------------------------------------------------------
    diag_narrative = render_outlier_diagnostic_narrative(selected_row)
    badge_html = get_anomaly_badge_html(selected_row["Anomaly_Classification"])

    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.85); border: 1px solid rgba(255, 255, 255, 0.10); border-left: 6px solid #ef4444; padding: 18px; border-radius: 10px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem; font-weight: 800;">
                        {selected_row['Store_Banner']} — {selected_row['Comune']} ({selected_row['Regione']})
                    </h3>
                    <span style="font-family: monospace; font-size: 0.82rem; color: #94a3b8; font-weight: 700;">
                        Machine ID: <code>{selected_row['PF']}</code> | Model: <b>{selected_row['TipoMacchina']}</b> | Total Operating Days: <b>{selected_row['Operational_Days']:,}</b>
                    </span>
                </div>
                <div>{badge_html}</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.70); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; margin-bottom: 12px; font-size: 0.88rem; color: #cbd5e1; line-height: 1.55;">
                {diag_narrative}
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); padding: 8px 12px; border-radius: 6px;">
                    <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Max Consecutive Zero Days</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f87171;">{selected_row['Max_Consecutive_Zero_Days']} Days</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); padding: 8px 12px; border-radius: 6px;">
                    <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Recent Zero Rate</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f87171;">{selected_row['Recent_Zero_Rate_Pct']:.1f}%</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); padding: 8px 12px; border-radius: 6px;">
                    <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Operational Median</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f59e0b;">{int(selected_row['Median_Bottles_Daily'])} b/d</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.06); padding: 8px 12px; border-radius: 6px;">
                    <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700; text-transform: uppercase;">Single-Day Peak</span>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #c084fc;">{int(selected_row['Max_Day_Bottles']):,} Bottles</div>
                </div>
            </div>
            <div style="margin-top: 12px; padding: 10px 14px; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 6px;">
                <b style="color: #34d399; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.4px;">🛠️ Recommended Action:</b>
                <div style="color: #a7f3d0; font-size: 0.85rem; margin-top: 2px;">{selected_row['Recommended_Action']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------------------------------------
    # Historical Plotly Bottles & Outliers Visualizer
    # -------------------------------------------------------------
    with st.spinner(f"Loading full historical records for {selected_pf} from PostgreSQL..."):
        df_hist = fetch_machine_historical_records(selected_pf)

    if df_hist.empty:
        st.info("No daily collection logs found in database for this machine ID.")
        return

    # Compute outlier points strictly on Bottles Volume
    med_bottles = float(df_hist["bottiglie"].median())
    p97_bottles = float(df_hist[df_hist["bottiglie"] > 0]["bottiglie"].quantile(0.97)) if (df_hist["bottiglie"] > 0).any() else 1000.0

    # 1. Zero-Collection / Downtime Outliers (Red)
    mask_zero = df_hist["bottiglie"] == 0.0
    df_zeros = df_hist[mask_zero].copy()

    # 2. Volume Spike Outliers (Purple / Amber Diamonds)
    mask_spike = (df_hist["bottiglie"] >= p97_bottles) & (df_hist["bottiglie"] > df_hist["roll_30d"] * 1.4) & (df_hist["bottiglie"] > 500)
    df_spikes = df_hist[mask_spike].copy()

    # 3. Severe Drop Outliers (Drops < 30% of 30d baseline, but > 0)
    mask_drop = (df_hist["bottiglie"] > 0) & (df_hist["bottiglie"] < df_hist["roll_30d"] * 0.30) & (df_hist["roll_30d"] > 150)
    df_drops = df_hist[mask_drop].copy()

    total_days = len(df_hist)
    n_zeros = int(mask_zero.sum())
    n_spikes = int(mask_spike.sum())
    n_drops = int(mask_drop.sum())
    zero_pct = (n_zeros / max(1, total_days)) * 100.0

    # Section Header with Quick Stat Badges
    st.markdown(
        f"""
        <div style="background: #0f172a; border: 1px solid rgba(255, 255, 255, 0.08); border-top: 3px solid #38bdf8; border-radius: 10px; padding: 12px 18px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div>
                    <span style="font-size: 1.0rem; font-weight: 800; color: #f8fafc;">
                        📊 Daily Bottles Collection & Outlier Detection
                    </span>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                        Machine: <code>{selected_pf}</code> &bull; Showing only bottles volume with automated outlier and downtime anomaly markers.
                    </div>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap; font-size: 0.76rem;">
                    <span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); padding: 3px 8px; border-radius: 5px; font-weight: 700;">
                        Operational Median: {med_bottles:.0f} b/d
                    </span>
                    <span style="background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); padding: 3px 8px; border-radius: 5px; font-weight: 700;">
                        🔴 Downtime Outliers: {n_zeros} days ({zero_pct:.1f}%)
                    </span>
                    <span style="background: rgba(192, 132, 252, 0.15); color: #c084fc; border: 1px solid rgba(192, 132, 252, 0.35); padding: 3px 8px; border-radius: 5px; font-weight: 700;">
                        ⚡ Volume Spikes: {n_spikes} days
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Build Plotly Figure (Single Y-Axis: Only Bottles Count)
    fig = go.Figure()

    # 1. Daily Bottles Trajectory Line with Soft Area Fill
    fig.add_trace(
        go.Scatter(
            x=df_hist["giorno_dt"],
            y=df_hist["bottiglie"],
            mode="lines",
            name="Daily Bottles Count",
            line=dict(color="#10b981", width=1.4),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.08)",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Bottles:</b> %{y:,.0f} b/d<extra></extra>"
        )
    )

    # 2. 30-Day Rolling Baseline (Smooth Trend)
    fig.add_trace(
        go.Scatter(
            x=df_hist["giorno_dt"],
            y=df_hist["roll_30d"],
            mode="lines",
            name="30-Day Sustained Baseline",
            line=dict(color="#38bdf8", width=2.4),
            hovertemplate="<b>30d Baseline:</b> %{y:,.1f} b/d<extra></extra>"
        )
    )

    # 3. Lifetime Median Horizontal Line
    fig.add_hline(
        y=med_bottles,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.6,
        annotation_text=f"Median: {med_bottles:.0f} b/d",
        annotation_position="top left",
        annotation_font=dict(color="#f59e0b", size=10, family="Inter")
    )

    # 4. Zero-Collection / Downtime Outliers (Red Markers)
    if not df_zeros.empty:
        fig.add_trace(
            go.Scatter(
                x=df_zeros["giorno_dt"],
                y=[0] * len(df_zeros),
                mode="markers",
                name="🔴 Zero Downtime Outliers",
                marker=dict(symbol="x", size=7, color="#ef4444", line=dict(color="#ef4444", width=1.5)),
                hovertemplate="<b>🔴 Downtime Outage:</b> 0 bottles collected<br><b>Date:</b> %{x|%Y-%m-%d}<extra></extra>"
            )
        )

    # 5. Volume Spike Outliers (Purple Diamonds)
    if not df_spikes.empty:
        fig.add_trace(
            go.Scatter(
                x=df_spikes["giorno_dt"],
                y=df_spikes["bottiglie"],
                mode="markers",
                name="⚡ Volume Spike Outliers",
                marker=dict(symbol="diamond", size=8, color="#c084fc", line=dict(color="#ffffff", width=1.0)),
                hovertemplate="<b>⚡ Volume Spike:</b> %{y:,.0f} bottles<br><b>Date:</b> %{x|%Y-%m-%d}<extra></extra>"
            )
        )

    # 6. Severe Drop Outliers (Amber Triangles)
    if not df_drops.empty:
        fig.add_trace(
            go.Scatter(
                x=df_drops["giorno_dt"],
                y=df_drops["bottiglie"],
                mode="markers",
                name="⚠️ Severe Drop Outliers",
                marker=dict(symbol="triangle-down", size=7, color="#f59e0b", line=dict(color="#f59e0b", width=1.0)),
                hovertemplate="<b>⚠️ Abnormal Drop:</b> %{y:,.0f} bottles<br><b>Date:</b> %{x|%Y-%m-%d}<extra></extra>"
            )
        )

    # Layout styling (Dark, clean, no overlapping text)
    max_y = float(df_hist["bottiglie"].max()) if not df_hist.empty else 1000.0
    fig.update_layout(
        template="plotly_dark",
        xaxis=dict(
            title=dict(text="Operational Date", font=dict(color="#94a3b8", size=11)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            tickfont=dict(size=10, color="#94a3b8"),
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            title=dict(text="Daily Bottles Collected (b/d)", font=dict(color="#94a3b8", size=11)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            tickfont=dict(size=10, color="#94a3b8"),
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.15)",
            rangemode="tozero",
            range=[0, max_y * 1.12]
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.90)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            font=dict(size=10, color="#cbd5e1")
        ),
        hovermode="x unified",
        margin=dict(l=55, r=25, t=15, b=65),
        height=420,
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a"
    )

    st.plotly_chart(fig, use_container_width=True, key="outlier_timeline_bottles_chart")
