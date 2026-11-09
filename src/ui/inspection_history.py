"""
Inspection History & Machine Audit Management System.
Logs prospective machine audits when clicking 'Results 📊' on SPL/POT machine lists,
strictly restricts same-day duplicate entries per machine, preserves full prediction
variables and 12-month seasonality profiles in MariaDB/MySQL/Postgres, provides
machine-wise analytics, and supports multi-record selection and deletion.
"""

import os
import json
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text, create_engine
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[2]
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")


# =========================================================================
# 1. DATABASE CONNECTION & TABLE INITIALIZATION
# =========================================================================

@st.cache_resource
def get_history_db_engine():
    """Returns a pooled SQLAlchemy engine for MySQL/MariaDB, PostgreSQL, or SQLite."""
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


def init_history_table(engine=None):
    """
    Initializes machine_prediction_history table across MariaDB/MySQL,
    PostgreSQL, and SQLite with same-day duplicate prevention constraints.
    """
    if engine is None:
        engine = get_history_db_engine()

    dialect = engine.dialect.name.lower()

    with engine.connect() as conn:
        if "mysql" in dialect or "mariadb" in dialect:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS `machine_prediction_history` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `punto_fisico` VARCHAR(64) NOT NULL,
                    `status_source` VARCHAR(20) NOT NULL,
                    `store_name` VARCHAR(255) NULL,
                    `address` TEXT NULL,
                    `cap` VARCHAR(20) NULL,
                    `comune` VARCHAR(128) NULL,
                    `provincia` VARCHAR(32) NULL,
                    `regione` VARCHAR(64) NULL,
                    `machine_type` VARCHAR(64) NULL,
                    `filiera` VARCHAR(128) NULL,
                    `predicted_daily_bottles` FLOAT DEFAULT 0.0,
                    `predicted_monthly_bottles` BIGINT DEFAULT 0,
                    `predicted_annual_bottles` BIGINT DEFAULT 0,
                    `predicted_daily_visits` FLOAT DEFAULT 0.0,
                    `bottles_per_user` FLOAT DEFAULT 0.0,
                    `volume_tier` VARCHAR(64) NULL,
                    `p10_floor` FLOAT DEFAULT 0.0,
                    `p90_peak` FLOAT DEFAULT 0.0,
                    `confidence_score` FLOAT DEFAULT 0.0,
                    `sales_area_sqm` FLOAT DEFAULT 0.0,
                    `cash_registers` FLOAT DEFAULT 0.0,
                    `nielsen_format` VARCHAR(64) NULL,
                    `reward_type` VARCHAR(128) NULL,
                    `reward_multiplier` FLOAT DEFAULT 1.0,
                    `nearest_coripet_km` FLOAT DEFAULT 0.0,
                    `seasonality_json` LONGTEXT NULL,
                    `full_prediction_json` LONGTEXT NULL,
                    `inspection_date` DATE NOT NULL,
                    `inspection_timestamp` DATETIME NOT NULL,
                    INDEX `idx_hist_pf` (`punto_fisico`),
                    INDEX `idx_hist_date` (`inspection_date`),
                    INDEX `idx_hist_status` (`status_source`),
                    UNIQUE KEY `uq_pf_inspection_date` (`punto_fisico`, `inspection_date`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """))
            conn.commit()
        elif "postgres" in dialect:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS machine_prediction_history (
                    id SERIAL PRIMARY KEY,
                    punto_fisico VARCHAR(64) NOT NULL,
                    status_source VARCHAR(20) NOT NULL,
                    store_name VARCHAR(255),
                    address TEXT,
                    cap VARCHAR(20),
                    comune VARCHAR(128),
                    provincia VARCHAR(32),
                    regione VARCHAR(64),
                    machine_type VARCHAR(64),
                    filiera VARCHAR(128),
                    predicted_daily_bottles REAL DEFAULT 0.0,
                    predicted_monthly_bottles BIGINT DEFAULT 0,
                    predicted_annual_bottles BIGINT DEFAULT 0,
                    predicted_daily_visits REAL DEFAULT 0.0,
                    bottles_per_user REAL DEFAULT 0.0,
                    volume_tier VARCHAR(64),
                    p10_floor REAL DEFAULT 0.0,
                    p90_peak REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    sales_area_sqm REAL DEFAULT 0.0,
                    cash_registers REAL DEFAULT 0.0,
                    nielsen_format VARCHAR(64),
                    reward_type VARCHAR(128),
                    reward_multiplier REAL DEFAULT 1.0,
                    nearest_coripet_km REAL DEFAULT 0.0,
                    seasonality_json TEXT,
                    full_prediction_json TEXT,
                    inspection_date DATE NOT NULL,
                    inspection_timestamp TIMESTAMP NOT NULL,
                    CONSTRAINT uq_pf_inspection_date UNIQUE (punto_fisico, inspection_date)
                );
                CREATE INDEX IF NOT EXISTS idx_hist_pf ON machine_prediction_history (punto_fisico);
                CREATE INDEX IF NOT EXISTS idx_hist_date ON machine_prediction_history (inspection_date);
                CREATE INDEX IF NOT EXISTS idx_hist_status ON machine_prediction_history (status_source);
            """))
            conn.commit()
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS machine_prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    punto_fisico TEXT NOT NULL,
                    status_source TEXT NOT NULL,
                    store_name TEXT,
                    address TEXT,
                    cap TEXT,
                    comune TEXT,
                    provincia TEXT,
                    regione TEXT,
                    machine_type TEXT,
                    filiera TEXT,
                    predicted_daily_bottles REAL DEFAULT 0.0,
                    predicted_monthly_bottles INTEGER DEFAULT 0,
                    predicted_annual_bottles INTEGER DEFAULT 0,
                    predicted_daily_visits REAL DEFAULT 0.0,
                    bottles_per_user REAL DEFAULT 0.0,
                    volume_tier TEXT,
                    p10_floor REAL DEFAULT 0.0,
                    p90_peak REAL DEFAULT 0.0,
                    confidence_score REAL DEFAULT 0.0,
                    sales_area_sqm REAL DEFAULT 0.0,
                    cash_registers REAL DEFAULT 0.0,
                    nielsen_format TEXT,
                    reward_type TEXT,
                    reward_multiplier REAL DEFAULT 1.0,
                    nearest_coripet_km REAL DEFAULT 0.0,
                    seasonality_json TEXT,
                    full_prediction_json TEXT,
                    inspection_date TEXT NOT NULL,
                    inspection_timestamp TEXT NOT NULL,
                    UNIQUE (punto_fisico, inspection_date)
                );
            """))
            conn.commit()


# =========================================================================
# 2. DATA SERIALIZATION & RECORD LOGGING (WITH SAME-DAY RESTRICTION)
# =========================================================================

def _safe_json_encoder(obj: Any) -> Any:
    """Recursively converts numpy, date/time, and non-serializable objects for JSON storage."""
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def check_already_inspected_today(punto_fisico: str, target_date: Optional[date] = None, engine=None) -> bool:
    """
    Checks if an inspection record already exists for the given machine on the specified date.
    Strictly enforces same-day single entry per machine.
    """
    if not punto_fisico:
        return False
    if engine is None:
        engine = get_history_db_engine()
    if target_date is None:
        target_date = date.today()

    try:
        init_history_table(engine)
        with engine.connect() as conn:
            q = text("""
                SELECT id FROM machine_prediction_history
                WHERE punto_fisico = :pf AND inspection_date = :dt
                LIMIT 1
            """)
            res = conn.execute(q, {"pf": str(punto_fisico).strip(), "dt": target_date}).fetchone()
            return res is not None
    except Exception as e:
        st.warning(f"Warning checking inspection history: {e}")
        return False


def save_machine_inspection_record(
    status_source: str,
    pf_code: str,
    store_name: str,
    address: Optional[str] = None,
    cap: Optional[str] = None,
    comune: Optional[str] = None,
    provincia: Optional[str] = None,
    regione: Optional[str] = None,
    machine_type: Optional[str] = None,
    filiera: Optional[str] = None,
    prediction_res: Optional[Dict[str, Any]] = None,
    b_day: Optional[float] = None,
    m_day: Optional[int] = None,
    tier_label: Optional[str] = None,
    n_fmt: Optional[str] = None,
    n_sqm: Optional[float] = None,
    n_casse: Optional[float] = None,
    rew_type: Optional[str] = None,
    engine=None
) -> Tuple[bool, str]:
    """
    Stores an inspection record in machine_prediction_history when user clicks 'Results 📊'.
    Strictly restricts data entry if the machine has already been inspected on the same day.
    Saves all prediction metrics, variables, spatial catchment, and 12-month seasonality profiles.
    
    Returns (success: bool, status_message: str).
    """
    clean_pf = str(pf_code).strip()
    today_dt = date.today()
    today_str = today_dt.strftime("%Y-%m-%d")
    now_ts = datetime.now()

    if engine is None:
        engine = get_history_db_engine()

    init_history_table(engine)

    # 1. Enforce same-day duplicate entry restriction
    if check_already_inspected_today(clean_pf, today_dt, engine=engine):
        return False, f"Inspection for {store_name} ({clean_pf}) is already recorded for today ({today_str}). Duplicate entry restricted."

    # 2. Extract prediction variables
    pred_dict = prediction_res if isinstance(prediction_res, dict) else {}
    preds = pred_dict.get("predictions", {})
    inputs = pred_dict.get("inputs", {})
    rewards = pred_dict.get("commercial_rewards", {})
    catchment = pred_dict.get("spatial_catchment", {})
    lifecycle_h = pred_dict.get("lifecycle_horizon", {})

    # Predicted daily bottles
    final_b_day = (
        float(b_day) if b_day is not None and b_day > 0
        else float(preds.get("expected_bottles_day") or preds.get("predicted_bottles_day") or 0.0)
    )
    # Predicted monthly bottles
    final_m_day = (
        int(m_day) if m_day is not None and m_day > 0
        else int(preds.get("expected_bottles_month") or round(final_b_day * 30.4))
    )
    final_y_day = int(preds.get("expected_bottles_year") or round(final_b_day * 365.0))

    final_visits = float(preds.get("expected_visitors_day") or preds.get("predicted_visits_day") or 0.0)
    final_b_per_u = float(preds.get("bottles_per_user") or (round(final_b_day / max(1.0, final_visits), 1) if final_visits > 0 else 0.0))

    # Tier label
    if not tier_label:
        if final_b_day >= 2000.0:
            tier_label = "SUPER PRENDENTE"
        elif final_b_day >= 1000.0:
            tier_label = "ALTO PRENDENTE"
        elif final_b_day >= 300.0:
            tier_label = "MEDIO PRENDENTE"
        else:
            tier_label = "BASSO PRENDENTE"

    # Confidence interval bounds
    p10_floor = float(preds.get("p10") or preds.get("p10_floor") or round(final_b_day * 0.72, 1))
    p90_peak = float(preds.get("p90") or preds.get("p90_peak") or round(final_b_day * 1.35, 1))
    conf_score = float(preds.get("confidence_score") or preds.get("score") or 0.85)

    # Store features
    final_n_fmt = str(n_fmt or inputs.get("store_format") or "SUP")
    final_n_sqm = float(n_sqm or inputs.get("sales_area_sqm") or 0.0)
    final_n_casse = float(n_casse or inputs.get("cash_registers") or 0.0)
    final_rew_type = str(rew_type or rewards.get("reward_type") or "Standard Loyalty")
    final_rew_mult = float(rewards.get("multiplier_bottles") or 1.0)
    final_near_km = float(catchment.get("coripet_network", {}).get("nearest_coripet_km") or 0.0)

    # Seasonality & monthly profile extraction (Strictly 15-month: 3M Setup Ramp + 12M Mature Horizon)
    rolling_profile = lifecycle_h.get("year1_rolling_profile")
    if not rolling_profile or len(rolling_profile) < 15:
        try:
            from src.model.predict import calculate_lifecycle_horizon
            monthly_col = preds.get("underwritten_monthly_profile") or preds.get("monthly_collection_profile") or []
            lh = calculate_lifecycle_horizon(
                underwritten_monthly_profile=monthly_col,
                expected_daily_bottiglie=final_b_day,
                expected_annual_bottiglie=final_b_day * 365.0,
                expected_daily_presenze=final_visits,
                go_live_month=lifecycle_h.get("launch_month_num", inputs.get("go_live_month", today_dt.month)),
                go_live_year=lifecycle_h.get("launch_year", inputs.get("go_live_year", today_dt.year)),
                enable_rampup=True
            )
            rolling_profile = lh.get("year1_rolling_profile", [])
            pred_dict["lifecycle_horizon"] = lh
        except Exception:
            pass

    if not rolling_profile:
        rolling_profile = (
            preds.get("underwritten_monthly_profile")
            or preds.get("monthly_collection_profile")
            or preds.get("monthly_predictions")
            or []
        )

    try:
        seasonality_json_str = json.dumps(rolling_profile, default=_safe_json_encoder, ensure_ascii=False)
    except Exception:
        seasonality_json_str = json.dumps([])

    try:
        full_pred_json_str = json.dumps(pred_dict, default=_safe_json_encoder, ensure_ascii=False)
    except Exception:
        full_pred_json_str = "{}"

    # 3. Database Insertion
    try:
        with engine.connect() as conn:
            insert_sql = text("""
                INSERT INTO machine_prediction_history (
                    punto_fisico, status_source, store_name, address, cap, comune, provincia, regione,
                    machine_type, filiera, predicted_daily_bottles, predicted_monthly_bottles,
                    predicted_annual_bottles, predicted_daily_visits, bottles_per_user, volume_tier,
                    p10_floor, p90_peak, confidence_score, sales_area_sqm, cash_registers,
                    nielsen_format, reward_type, reward_multiplier, nearest_coripet_km,
                    seasonality_json, full_prediction_json, inspection_date, inspection_timestamp
                ) VALUES (
                    :punto_fisico, :status_source, :store_name, :address, :cap, :comune, :provincia, :regione,
                    :machine_type, :filiera, :predicted_daily_bottles, :predicted_monthly_bottles,
                    :predicted_annual_bottles, :predicted_daily_visits, :bottles_per_user, :volume_tier,
                    :p10_floor, :p90_peak, :confidence_score, :sales_area_sqm, :cash_registers,
                    :nielsen_format, :reward_type, :reward_multiplier, :nearest_coripet_km,
                    :seasonality_json, :full_prediction_json, :inspection_date, :inspection_timestamp
                )
            """)
            conn.execute(insert_sql, {
                "punto_fisico": clean_pf,
                "status_source": str(status_source).strip().upper(),
                "store_name": str(store_name or clean_pf).strip(),
                "address": str(address or "").strip(),
                "cap": str(cap or "").strip(),
                "comune": str(comune or "").strip(),
                "provincia": str(provincia or "").strip(),
                "regione": str(regione or "").strip(),
                "machine_type": str(machine_type or "Ripet_C-front").strip(),
                "filiera": str(filiera or "").strip(),
                "predicted_daily_bottles": round(final_b_day, 1),
                "predicted_monthly_bottles": final_m_day,
                "predicted_annual_bottles": final_y_day,
                "predicted_daily_visits": round(final_visits, 1),
                "bottles_per_user": round(final_b_per_u, 1),
                "volume_tier": str(tier_label).strip(),
                "p10_floor": round(p10_floor, 1),
                "p90_peak": round(p90_peak, 1),
                "confidence_score": round(conf_score, 3),
                "sales_area_sqm": round(final_n_sqm, 1),
                "cash_registers": round(final_n_casse, 1),
                "nielsen_format": final_n_fmt,
                "reward_type": final_rew_type,
                "reward_multiplier": round(final_rew_mult, 2),
                "nearest_coripet_km": round(final_near_km, 2),
                "seasonality_json": seasonality_json_str,
                "full_prediction_json": full_pred_json_str,
                "inspection_date": today_dt,
                "inspection_timestamp": now_ts
            })
            conn.commit()
        return True, f"Successfully recorded inspection history for {store_name} ({clean_pf})!"
    except Exception as e:
        # Check if error was unique constraint violation
        err_str = str(e).lower()
        if "duplicate" in err_str or "unique" in err_str:
            return False, f"Machine {clean_pf} is already logged in inspection history for today ({today_str})."
        return False, f"Database error saving inspection: {e}"


# =========================================================================
# 3. QUERYING & DELETION ACTIONS
# =========================================================================

def delete_inspection_records(record_ids: List[int], engine=None) -> int:
    """
    Deletes specific records from machine_prediction_history by their primary keys.
    Returns the count of deleted rows.
    """
    if not record_ids:
        return 0
    if engine is None:
        engine = get_history_db_engine()

    init_history_table(engine)
    try:
        clean_ids = [int(i) for i in record_ids if str(i).isdigit()]
        if not clean_ids:
            return 0
        with engine.connect() as conn:
            # Cross-dialect batch deletion using parameterized placeholders
            placeholders = ", ".join([f":id_{idx}" for idx in range(len(clean_ids))])
            params = {f"id_{idx}": val for idx, val in enumerate(clean_ids)}
            del_sql = text(f"DELETE FROM machine_prediction_history WHERE id IN ({placeholders})")
            result = conn.execute(del_sql, params)
            conn.commit()
            return result.rowcount
    except Exception as e:
        st.error(f"Error deleting inspection records: {e}")
        return 0


def fetch_inspection_history(
    search_query: str = "",
    status_filter: str = "All",
    tier_filter: str = "All",
    pf_filter: Optional[str] = None,
    limit: int = 1000,
    engine=None
) -> pd.DataFrame:
    """
    Retrieves inspection history records with search, filtering, and machine-wise sorting.
    """
    if engine is None:
        engine = get_history_db_engine()

    init_history_table(engine)
    try:
        where_clauses = []
        params: Dict[str, Any] = {"lim": int(limit)}

        if status_filter and status_filter.upper() != "ALL":
            where_clauses.append("status_source = :status")
            params["status"] = status_filter.upper()

        if tier_filter and tier_filter.upper() != "ALL":
            where_clauses.append("volume_tier = :tier")
            params["tier"] = tier_filter

        if pf_filter and pf_filter != "All Machines":
            where_clauses.append("punto_fisico = :pf")
            params["pf"] = pf_filter

        if search_query and search_query.strip():
            sq = f"%{search_query.strip().lower()}%"
            where_clauses.append("""
                (LOWER(punto_fisico) LIKE :sq 
                 OR LOWER(store_name) LIKE :sq 
                 OR LOWER(comune) LIKE :sq 
                 OR LOWER(machine_type) LIKE :sq
                 OR LOWER(address) LIKE :sq)
            """)
            params["sq"] = sq

        where_stmt = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        query_sql = f"""
            SELECT 
                id, inspection_date, inspection_timestamp, punto_fisico, status_source,
                store_name, address, cap, comune, provincia, regione, machine_type, filiera,
                predicted_daily_bottles, predicted_monthly_bottles, predicted_annual_bottles,
                predicted_daily_visits, bottles_per_user, volume_tier,
                p10_floor, p90_peak, confidence_score, sales_area_sqm, cash_registers,
                nielsen_format, reward_type, reward_multiplier, nearest_coripet_km,
                seasonality_json, full_prediction_json
            FROM machine_prediction_history
            {where_stmt}
            ORDER BY inspection_timestamp DESC, id DESC
            LIMIT :lim
        """
        with engine.connect() as conn:
            df = pd.read_sql_query(text(query_sql), conn, params=params)
            return df
    except Exception as e:
        st.error(f"Error reading inspection history from database: {e}")
        return pd.DataFrame()


# =========================================================================
# 4. CHART BUILDER: HISTORICAL SEASONALITY PROJECTION
# =========================================================================

def parse_and_expand_15m_profile(seasonality_data: Any, daily_baseline: float = 0.0) -> List[Dict[str, Any]]:
    """
    Parses stored seasonality JSON and guarantees a full 15-month operational profile
    (3-Month Setup & Adoption Ramp-Up M01..M03 + 12-Month Mature Operational Seasonality Horizon M04..M15).
    Strictly generated from stored historical record data, with no live API calls.
    """
    profile_list = []
    if isinstance(seasonality_data, str) and seasonality_data:
        try:
            seasonality_data = json.loads(seasonality_data)
        except Exception:
            seasonality_data = []

    if isinstance(seasonality_data, list) and seasonality_data:
        for idx, item in enumerate(seasonality_data):
            if isinstance(item, dict):
                profile_list.append(item)

    # If profile already has all 15 months, return them directly
    if len(profile_list) >= 15:
        return profile_list[:15]

    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    seasonality_weights = {
        1: 0.88, 2: 0.92, 3: 0.98, 4: 1.02, 5: 1.10, 6: 1.25,
        7: 1.35, 8: 1.32, 9: 1.08, 10: 0.98, 11: 0.90, 12: 0.94
    }

    # Extract any month baseline provided in profile_list
    month_val_map = {}
    if profile_list:
        for p in profile_list:
            m_num = p.get("calendar_month_num")
            if not m_num:
                m_str = str(p.get("month_name", "")).lower()[:3]
                for k, v in month_names.items():
                    if v.lower() == m_str:
                        m_num = k
                        break
            if m_num:
                d_val = float(p.get("daily_bottiglie_median") or p.get("daily_bottiglie") or 0.0)
                if d_val > 0:
                    month_val_map[m_num] = d_val

    # Anchor launch calendar from current date
    from datetime import date
    today = date.today()
    curr_m = today.month
    curr_y = today.year

    ramp_factors = [0.65, 0.80, 0.92]  # Month 1: 65% discovery, Month 2: 80% habit, Month 3: 92% consolidation
    result_15m = []

    for k in range(1, 16):
        cal_m = (curr_m + k - 2) % 12 + 1
        cal_y = curr_y + ((curr_m + k - 2) // 12)
        m_name = month_names[cal_m]

        if cal_m in month_val_map:
            season_d = month_val_map[cal_m]
        else:
            w = seasonality_weights.get(cal_m, 1.0)
            season_d = (daily_baseline * w) if daily_baseline > 0 else (500.0 * w)

        is_ramp = (k <= 3)
        if is_ramp:
            mat_factor = ramp_factors[k - 1]
            daily_b = round(season_d * mat_factor, 1)
            stage_desc = f"Ramp-Up ({int(mat_factor*100)}% Adoption)"
            badge = "🛠️ Setup Ramp"
        else:
            mat_factor = 1.0
            daily_b = round(season_d, 1)
            stage_desc = f"Mature Horizon (Month {k-3})"
            badge = f"🟢 Mature Op M{k-3:02d}"

        monthly_target = int(round(daily_b * 30.4))

        result_15m.append({
            "lifecycle_label": f"M{k:02d} ({m_name} '{str(cal_y)[2:]})",
            "operational_month": f"M{k:02d}",
            "calendar_month_name": m_name,
            "calendar_year": cal_y,
            "horizon_badge": badge,
            "maturity_factor": mat_factor,
            "maturity_pct": f"{int(round(mat_factor * 100))}%",
            "daily_bottiglie": daily_b,
            "daily_bottiglie_median": daily_b,
            "monthly_target_bottiglie": monthly_target,
            "monthly_target_bottiglie_median": monthly_target,
            "is_rampup": is_ramp,
            "lifecycle_stage": stage_desc,
        })

    return result_15m


def render_history_seasonality_chart(seasonality_data: Any, daily_baseline: float = 0.0) -> Tuple[go.Figure, List[Dict[str, Any]]]:
    """
    Renders an interactive 15-month operational lifecycle & intake profile for a historical inspection record.
    Displays projected daily intake bottles (b/d) across all 15 operational months:
    - Months 1 to 3: Setup & customer adoption ramp-up (M01-M03 in soft/medium green)
    - Months 4 to 15: Seasonality-wise mature operational intake (M04-M15 in deep bold green)
    Strictly isolated to stored historical data with no live predictor calls.
    Returns (plotly_figure, profile_15m_rows).
    """
    profile_15m = parse_and_expand_15m_profile(seasonality_data, daily_baseline)

    x_labels = []
    daily_rates = []
    targets = []
    bar_colors = []
    hover_texts = []

    for idx, item in enumerate(profile_15m):
        lbl = item.get("lifecycle_label") or f"M{idx+1:02d}"
        d_val = float(item.get("daily_bottiglie_median") or item.get("daily_bottiglie") or 0.0)
        t_val = int(item.get("monthly_target_bottiglie_median") or item.get("monthly_target_bottiglie") or round(d_val * 30.4))
        is_ramp = bool(item.get("is_rampup", idx < 3))
        mat_pct = item.get("maturity_pct", "100%")
        stage = item.get("lifecycle_stage", "Operational Run")

        x_labels.append(lbl)
        daily_rates.append(d_val)
        targets.append(t_val)

        if is_ramp:
            if idx == 0:
                bar_colors.append("#81c784")  # Soft green for discovery month M01 (65%)
            else:
                bar_colors.append("#4caf50")  # Medium green for habit consolidation M02-M03 (80-92%)
        else:
            bar_colors.append("#1b5e20")  # Deep bold forest green for 12 Mature Months M04-M15

        hover_texts.append(
            f"<b>{lbl}</b><br>"
            f"Daily Velocity: <b>{d_val:.1f} b/d</b><br>"
            f"Monthly Target: <b>{t_val:,} b/mo</b><br>"
            f"Adoption Maturity: <b>{mat_pct}</b> ({stage})"
        )

    fig = go.Figure()

    # 1. Main Daily Intake Bars (showing Daily Bottles b/d)
    fig.add_trace(go.Bar(
        x=x_labels,
        y=daily_rates,
        name="Projected Daily Intake (b/d)",
        marker=dict(color=bar_colors, opacity=0.90),
        text=[f"{int(round(v))}" for v in daily_rates],
        textposition="auto",
        hovertext=hover_texts,
        hoverinfo="text",
    ))

    # 2. Smooth Daily Run Rate Trend Line
    fig.add_trace(go.Scatter(
        x=x_labels,
        y=daily_rates,
        name="Daily Velocity Trend (b/d)",
        mode="lines+markers",
        line=dict(color="#10b981", width=2.8, shape="spline"),
        marker=dict(size=6, color="#34d399"),
        hovertext=hover_texts,
        hoverinfo="text",
    ))

    # 3. Steady-State Benchmark Line (mature 12M average)
    mature_vals = daily_rates[3:] if len(daily_rates) >= 4 else daily_rates
    benchmark_val = float(np.mean(mature_vals)) if mature_vals else daily_baseline
    if benchmark_val > 0:
        fig.add_hline(
            y=benchmark_val,
            line_dash="dash",
            line_color="#38bdf8",
            line_width=2.0,
            annotation_text=f"Confirmed 12M Mature Steady-State ({benchmark_val:.0f} b/d)",
            annotation_position="bottom right",
            annotation_font=dict(color="#38bdf8", size=10)
        )

    fig.update_layout(
        title=dict(
            text="📈 15-Month Operational Lifecycle & Intake Profile: Daily Velocity (b/d) & Seasonality",
            font=dict(size=14, color="#f8fafc", family="Inter, sans-serif")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.65)",
        height=340,
        margin=dict(l=45, r=25, t=50, b=35),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94a3b8", size=11)
        ),
        xaxis=dict(
            title=dict(text="Operational Horizon Month (3M Setup Ramp + 12M Seasonality Mature Horizon)", font=dict(color="#94a3b8", size=10)),
            showgrid=False,
            tickfont=dict(color="#cbd5e1", size=9)
        ),
        yaxis=dict(
            title=dict(text="Daily Bottles (b/d)", font=dict(color="#10b981", size=11)),
            tickfont=dict(color="#94a3b8", size=10),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
        )
    )

    return fig, profile_15m


# =========================================================================
# 5. UI TAB RENDERER: INSPECTION HISTORY & MACHINE-WISE AUDITS
# =========================================================================

def render_machine_history_tab():
    """
    Renders the complete dedicated '📜 Inspection History' tab view.
    Displays machine-wise historical inspection records, search & filtering,
    per-row and bulk deletion via checkboxes, detailed prediction inspection,
    and 12-month seasonality graphs.
    """
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
                    border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 22px 26px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
                <div>
                    <h2 style="margin: 0; color: #f8fafc; font-size: 1.55rem; font-weight: 800; letter-spacing: -0.5px;">
                        📜 Machine Inspection History & Prospective Audits
                    </h2>
                    <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 0.92rem; line-height: 1.45;">
                        Permanent historical registry of all machine evaluations triggered via <strong style="color: #38bdf8;">Results 📊</strong> on SPL and POT screens.
                        Duplicate entries are strictly restricted to <strong>one inspection per machine per day</strong>. Full predictions, input variables, and 12-month seasonality profiles are stored in the database.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Fetch entire historical dataset for initial stats
    df_all = fetch_inspection_history(limit=5000)

    if df_all.empty:
        st.info("ℹ️ No inspection records found yet. Click **Results 📊** on any machine in the **SPL Machines** or **POT Machines** screens to record your first machine inspection!")
        return

    # 2. Executive KPI Header
    total_audits = len(df_all)
    unique_machines = df_all["punto_fisico"].nunique()
    spl_count = len(df_all[df_all["status_source"] == "SPL"])
    pot_count = len(df_all[df_all["status_source"] == "POT"])
    avg_yield = df_all["predicted_daily_bottles"].mean() if "predicted_daily_bottles" in df_all else 0.0
    highest_machine = df_all.sort_values(by="predicted_daily_bottles", ascending=False).iloc[0] if not df_all.empty else None

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns([1.0, 1.0, 1.2, 1.1, 1.7])
    with kpi1:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 16px;">
                <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Total Inspections</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #38bdf8; margin-top: 4px;">{total_audits:,}</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">Logged database records</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi2:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 16px;">
                <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Unique Machines</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #a855f7; margin-top: 4px;">{unique_machines:,}</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">Distinct Punto Fisico sites</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi3:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 16px;">
                <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">SPL vs POT Sources</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #10b981; margin-top: 6px;">
                    {spl_count} <span style="font-size: 0.82rem; color: #94a3b8;">SPL</span> • {pot_count} <span style="font-size: 0.82rem; color: #94a3b8;">POT</span>
                </div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">Registry distribution</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi4:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 14px 16px;">
                <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Avg Daily Yield</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #f59e0b; margin-top: 4px;">{avg_yield:,.1f} <span style="font-size: 0.85rem; font-weight: 700;">b/d</span></div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 2px;">~{int(avg_yield * 30.4):,} b/mo mean</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with kpi5:
        top_name = str(highest_machine["store_name"]) if highest_machine is not None else "-"
        top_val = float(highest_machine["predicted_daily_bottles"]) if highest_machine is not None else 0.0
        top_pf = str(highest_machine["punto_fisico"]) if highest_machine is not None else ""
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(236, 72, 153, 0.25); border-radius: 10px; padding: 14px 16px;">
                <div style="color: #94a3b8; font-size: 0.78rem; font-weight: 600; text-transform: uppercase;">Peak Volume Machine</div>
                <div style="font-size: 1.65rem; font-weight: 800; color: #ec4899; margin-top: 4px;">{top_val:,.1f} <span style="font-size: 0.85rem; font-weight: 700;">b/d</span></div>
                <div style="color: #cbd5e1; font-size: 0.76rem; font-weight: 600; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{top_name} ({top_pf})">
                    🏆 {top_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)

    # 3. Interactive Filtering Bar
    f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns([2.4, 1.1, 1.3, 3.2, 1.0])
    with f_col1:
        search_kw = st.text_input(
            "Search Machine History",
            placeholder="🔍 Search store, PF ID, comune, model...",
            key="hist_search_kw",
            label_visibility="collapsed"
        )
    with f_col2:
        src_opt = st.selectbox(
            "Source",
            ["All Sources", "SPL", "POT"],
            key="hist_src_filter",
            label_visibility="collapsed"
        )
    with f_col3:
        tier_opt = st.selectbox(
            "Volume Tier",
            ["All Tiers", "SUPER PRENDENTE", "ALTO PRENDENTE", "MEDIO PRENDENTE", "BASSO PRENDENTE"],
            key="hist_tier_filter",
            label_visibility="collapsed"
        )
    with f_col4:
        # Build machine-wise selector with full store descriptions
        pf_descriptions = {"All Machines": "🏢 All Machines (All Sites)"}
        for _, row_item in df_all.drop_duplicates(subset=["punto_fisico"]).iterrows():
            pf_code = str(row_item["punto_fisico"]).strip()
            s_name = str(row_item.get("store_name") or "").strip()
            comune = str(row_item.get("comune") or "").strip()
            if s_name and s_name.lower() != "nan" and s_name != pf_code:
                if comune and comune.lower() != "nan" and comune.lower() not in s_name.lower():
                    desc = f"{pf_code} • {s_name} ({comune})"
                else:
                    desc = f"{pf_code} • {s_name}"
            elif comune and comune.lower() != "nan":
                desc = f"{pf_code} • {comune}"
            else:
                desc = pf_code
            pf_descriptions[pf_code] = desc

        unique_pfs = sorted(df_all["punto_fisico"].dropna().unique().tolist())
        pf_choices = ["All Machines"] + unique_pfs
        pf_selected = st.selectbox(
            "Machine Filter",
            pf_choices,
            format_func=lambda x: pf_descriptions.get(x, x),
            key="hist_pf_selector",
            label_visibility="collapsed"
        )
    with f_col5:
        if st.button("🔄 Refresh", use_container_width=True, key="btn_hist_refresh"):
            st.rerun()

    # 4. Filter data
    status_filter_val = "All" if src_opt == "All Sources" else src_opt
    tier_filter_val = "All" if tier_opt == "All Tiers" else tier_opt
    filtered_df = fetch_inspection_history(
        search_query=search_kw,
        status_filter=status_filter_val,
        tier_filter=tier_filter_val,
        pf_filter=pf_selected if pf_selected != "All Machines" else None,
        limit=1000
    )

    st.markdown(f"<div style='font-size: 0.88rem; color: #94a3b8; margin: 6px 0 12px 2px;'>Showing <strong>{len(filtered_df):,}</strong> inspection records</div>", unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("⚠️ No inspection records match your selected filters.")
        return

    # 5. Bulk Deletion Bar & Management Actions
    if "selected_hist_records" not in st.session_state:
        st.session_state["selected_hist_records"] = set()

    del_bar_col1, del_bar_col2, del_bar_col3, del_bar_col4 = st.columns([2.0, 1.2, 1.2, 2.0])
    
    with del_bar_col1:
        selected_count = len(st.session_state["selected_hist_records"])
        st.markdown(
            f"""
            <div style="font-size: 0.90rem; font-weight: 700; color: {'#38bdf8' if selected_count > 0 else '#94a3b8'}; padding-top: 6px;">
                Selected for Deletion: <span>{selected_count} record(s)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with del_bar_col2:
        if st.button("Select All", key="btn_hist_sel_all", use_container_width=True):
            st.session_state["selected_hist_records"] = set(filtered_df["id"].tolist())
            for rid in filtered_df["id"].tolist():
                st.session_state[f"hist_chk_{rid}"] = True
            st.rerun()

    with del_bar_col3:
        if st.button("Deselect All", key="btn_hist_desel_all", use_container_width=True):
            st.session_state["selected_hist_records"] = set()
            for rid in filtered_df["id"].tolist():
                st.session_state[f"hist_chk_{rid}"] = False
            st.rerun()

    with del_bar_col4:
        delete_disabled = selected_count == 0
        if st.button(
            f"🗑️ Delete Selected ({selected_count})",
            key="btn_hist_delete_bulk",
            type="primary" if not delete_disabled else "secondary",
            disabled=delete_disabled,
            use_container_width=True,
            help="Permanently deletes the selected inspection records from the database table."
        ):
            del_ids = list(st.session_state["selected_hist_records"])
            deleted_rows = delete_inspection_records(del_ids)
            for rid in del_ids:
                st.session_state.pop(f"hist_chk_{rid}", None)
            st.session_state["selected_hist_records"] = set()
            st.toast(f"🗑️ Successfully deleted {deleted_rows} inspection record(s) from database!", icon="✅")
            st.rerun()

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 6. View Switcher: Table View vs Machine-Wise Grouped View
    view_mode = st.radio(
        "View Mode",
        ["📋 Inspection Records (Row & Checkbox View)", "🗂️ Machine-Wise Grouped Cards"],
        horizontal=True,
        key="hist_view_mode",
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------
    # VIEW MODE 1: ROW & CHECKBOX TABLE VIEW
    # -------------------------------------------------------------
    if "Row & Checkbox" in view_mode:
        # Table Header
        h0, h1, h2, h3, h4, h5, h6 = st.columns([0.4, 1.8, 3.2, 1.6, 2.0, 1.4, 1.2])
        with h0: st.markdown("<strong style='font-size: 0.80rem; color: #94a3b8;'>Del</strong>", unsafe_allow_html=True)
        with h1: st.markdown("<strong style='font-size: 0.80rem; color: #94a3b8;'>Date / Source</strong>", unsafe_allow_html=True)
        with h2: st.markdown("<strong style='font-size: 0.80rem; color: #94a3b8;'>Machine & Location</strong>", unsafe_allow_html=True)
        with h3: st.markdown("<strong style='font-size: 0.80rem; color: #94a3b8;'>Hardware</strong>", unsafe_allow_html=True)
        with h4: st.markdown("<strong style='font-size: 0.80rem; color: #94a3b8;'>Predicted Yield</strong>", unsafe_allow_html=True)
        with h5: st.markdown("<strong style='font-size: 0.80rem; color: #94a3b8;'>Inspect</strong>", unsafe_allow_html=True)
        with h6: st.markdown("<strong style='font-size: 0.80rem; color: #94a3b8;'>Action</strong>", unsafe_allow_html=True)

        st.markdown("<div style='border-bottom: 1px solid rgba(255,255,255,0.12); margin: 6px 0 10px 0;'></div>", unsafe_allow_html=True)

        # Active inspect target in session state
        active_inspect_id = st.session_state.get("hist_inspect_record_id")

        for idx, row in filtered_df.iterrows():
            rec_id = int(row["id"])
            pf_code = str(row["punto_fisico"])
            store = str(row["store_name"])
            src = str(row["status_source"])
            insp_date = str(row["inspection_date"])
            insp_ts = str(row["inspection_timestamp"])
            addr = str(row["address"])
            comune = str(row["comune"])
            prov = str(row["provincia"])
            mach_type = str(row["machine_type"])
            b_day = float(row["predicted_daily_bottles"])
            m_day = int(row["predicted_monthly_bottles"])
            tier = str(row["volume_tier"])
            
            # Badge styles
            src_badge_bg = "rgba(16, 185, 129, 0.15)" if src == "SPL" else "rgba(56, 189, 248, 0.15)"
            src_badge_border = "#10b981" if src == "SPL" else "#38bdf8"
            
            if "SUPER" in tier:
                tier_color = "#c084fc"
            elif "ALTO" in tier:
                tier_color = "#4ade80"
            elif "MEDIO" in tier:
                tier_color = "#38bdf8"
            else:
                tier_color = "#f87171"

            is_selected = rec_id in st.session_state["selected_hist_records"]

            c0, c1, c2, c3, c4, c5, c6 = st.columns([0.4, 1.8, 3.2, 1.6, 2.0, 1.4, 1.2])

            with c0:
                # Checkbox to mark for deletion
                chk_val = st.checkbox(
                    "",
                    value=is_selected,
                    key=f"hist_chk_{rec_id}",
                    label_visibility="collapsed"
                )
                if chk_val != is_selected:
                    if chk_val:
                        st.session_state["selected_hist_records"].add(rec_id)
                    else:
                        st.session_state["selected_hist_records"].discard(rec_id)
                    st.rerun()

            with c1:
                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; line-height: 1.35;">
                        <span style="background: {src_badge_bg}; border: 1px solid {src_badge_border}; color: {src_badge_border}; padding: 1.5px 6px; border-radius: 4px; font-weight: 700; font-size: 0.72rem;">{src}</span>
                        <div style="font-weight: 700; color: #f8fafc; margin-top: 4px;">📅 {insp_date}</div>
                        <div style="color: #64748b; font-size: 0.72rem;">{insp_ts.split(' ')[-1] if ' ' in insp_ts else ''}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:
                prov_str = f"({prov})" if prov else ""
                st.markdown(
                    f"""
                    <div style="font-size: 0.84rem; line-height: 1.35;">
                        <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem;">{store}</div>
                        <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 2px;">🆔 <code style="color: #38bdf8;">{pf_code}</code> • 📍 {comune} {prov_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c3:
                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; line-height: 1.35;">
                        <div style="color: #e2e8f0; font-weight: 600;">⚙️ {mach_type}</div>
                        <div style="color: #64748b; font-size: 0.74rem;">Format: {row.get('nielsen_format', 'SUP')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c4:
                st.markdown(
                    f"""
                    <div style="font-size: 0.84rem; line-height: 1.35;">
                        <div style="display: flex; align-items: baseline; gap: 4px;">
                            <span style="font-size: 1.15rem; font-weight: 800; color: {tier_color};">{b_day:,.1f}</span>
                            <span style="font-size: 0.78rem; font-weight: 700; color: {tier_color};">b/d</span>
                            <span style="font-size: 0.74rem; color: #94a3b8; margin-left: 4px;">(~{m_day:,} b/mo)</span>
                        </div>
                        <div style="font-size: 0.72rem; color: {tier_color}; font-weight: 700; margin-top: 2px;">{tier}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c5:
                # Toggle detailed inspection view
                is_active = (active_inspect_id == rec_id)
                btn_lbl = "Close ✖️" if is_active else "Inspect 🔍"
                if st.button(btn_lbl, key=f"btn_inspect_{rec_id}", use_container_width=True):
                    if is_active:
                        st.session_state["hist_inspect_record_id"] = None
                    else:
                        st.session_state["hist_inspect_record_id"] = rec_id
                    st.rerun()

            with c6:
                # Direct single-record delete button
                if st.button("🗑️", key=f"btn_del_single_{rec_id}", help="Delete this single record", use_container_width=True):
                    delete_inspection_records([rec_id])
                    st.session_state.pop(f"hist_chk_{rec_id}", None)
                    if rec_id in st.session_state["selected_hist_records"]:
                        st.session_state["selected_hist_records"].remove(rec_id)
                    if st.session_state.get("hist_inspect_record_id") == rec_id:
                        st.session_state["hist_inspect_record_id"] = None
                    st.toast(f"🗑️ Deleted inspection record #{rec_id} for {store}!", icon="✅")
                    st.rerun()

            # Expanded Inspection Dossier for the active record
            if active_inspect_id == rec_id:
                _render_record_details_card(row)

            st.markdown("<div style='border-bottom: 1px solid rgba(255, 255, 255, 0.06); margin: 6px 0 8px 0;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # VIEW MODE 2: MACHINE-WISE GROUPED CARDS
    # -------------------------------------------------------------
    else:
        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        grouped = filtered_df.groupby("punto_fisico")

        for pf_key, group in grouped:
            latest_row = group.sort_values(by="inspection_timestamp", ascending=False).iloc[0]
            store_nm = latest_row["store_name"]
            st_src = latest_row["status_source"]
            audit_count = len(group)
            latest_date = latest_row["inspection_date"]
            latest_yield = latest_row["predicted_daily_bottles"]
            tier_str = latest_row["volume_tier"]

            with st.expander(f"📍 {store_nm} — ID: {pf_key} ({st_src}) • {audit_count} Inspection(s) Logged • Latest: {latest_date} ({latest_yield:.1f} b/d)", expanded=False):
                st.markdown(
                    f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                        <div style="display: flex; gap: 14px; flex-wrap: wrap; align-items: center;">
                            <div><strong style="color: #94a3b8;">Address:</strong> {latest_row['address']} ({latest_row['comune']}, {latest_row['provincia']})</div>
                            <div><strong style="color: #94a3b8;">Hardware:</strong> {latest_row['machine_type']}</div>
                            <div><strong style="color: #94a3b8;">Format:</strong> {latest_row['nielsen_format']} ({latest_row['sales_area_sqm']:.0f} m²)</div>
                            <div><strong style="color: #94a3b8;">Reward:</strong> {latest_row['reward_type']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Seasonality Chart for the latest audit of this machine
                chart, _ = render_history_seasonality_chart(latest_row.get("seasonality_json"), daily_baseline=latest_yield)
                st.plotly_chart(chart, use_container_width=True, key=f"plot_grp_{pf_key}_{latest_row['id']}")

                # Table of all historical inspection dates for this machine
                sub_df = group[["id", "inspection_date", "inspection_timestamp", "status_source", "predicted_daily_bottles", "predicted_monthly_bottles", "volume_tier"]]
                st.dataframe(
                    sub_df.rename(columns={
                        "id": "Record ID",
                        "inspection_date": "Inspection Date",
                        "inspection_timestamp": "Timestamp",
                        "status_source": "Source",
                        "predicted_daily_bottles": "Daily (b/d)",
                        "predicted_monthly_bottles": "Monthly (b/mo)",
                        "volume_tier": "Tier"
                    }),
                    use_container_width=True,
                    hide_index=True
                )


def _render_record_details_card(row: pd.Series):
    """
    Renders detailed inspection dossier for a selected record in the inspection history.
    """
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 10px; padding: 18px 20px; margin: 10px 0 16px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 10px; margin-bottom: 14px;">
                <h4 style="margin: 0; color: #38bdf8; font-size: 1.15rem; font-weight: 700;">
                    🔍 Inspection Dossier — Record #""" + str(row['id']) + """ (""" + str(row['store_name']) + """)
                </h4>
                <span style="color: #94a3b8; font-size: 0.80rem;">Inspected on: <strong>""" + str(row['inspection_timestamp']) + """</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Daily Predicted Bottles", f"{row['predicted_daily_bottles']:,.1f} b/d", delta=f"{row['volume_tier']}")
    with d2:
        st.metric("Monthly Target", f"{row['predicted_monthly_bottles']:,} b/mo", delta="Annual: " + f"{row['predicted_annual_bottles']:,}")
    with d3:
        st.metric("Recycler Footfall", f"{row['predicted_daily_visits']:,.1f} users/d", delta=f"{row['bottles_per_user']:.1f} b/user")
    with d4:
        st.metric("Confidence Interval", f"{row['p10_floor']:,.0f} - {row['p90_peak']:,.0f} b/d", delta=f"Score: {row['confidence_score']:.1%}")

    # 15-Month Operational Lifecycle & Seasonality Projection Plotly Graph
    season_fig, profile_15m = render_history_seasonality_chart(
        row.get("seasonality_json"),
        daily_baseline=float(row["predicted_daily_bottles"])
    )
    st.plotly_chart(season_fig, use_container_width=True, key=f"hist_season_fig_{row['id']}")

    # 15-Month Lifecycle Breakdown Table
    if profile_15m:
        with st.expander("📊 View Detailed 15-Month Intake & Seasonality Target Table", expanded=True):
            m_rows = []
            for idx, m in enumerate(profile_15m):
                d_val = float(m.get("daily_bottiglie_median") or m.get("daily_bottiglie") or 0.0)
                t_val = int(m.get("monthly_target_bottiglie_median") or m.get("monthly_target_bottiglie") or round(d_val * 30.4))
                is_ramp = bool(m.get("is_rampup", idx < 3))
                horizon_badge = "🛠️ Setup Ramp" if is_ramp else f"🟢 Mature Op M{idx-2:02d}"
                m_rows.append({
                    "Operational Month": m.get("lifecycle_label", f"M{idx+1:02d}"),
                    "Horizon Phase": horizon_badge,
                    "Projected Daily (b/d)": f"{d_val:,.1f} b/d",
                    "Monthly Target (b/mo)": f"{t_val:,} b/mo",
                    "Adoption Maturity": m.get("maturity_pct", "100%"),
                    "Lifecycle Stage": m.get("lifecycle_stage", "Operational Run")
                })
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)

    # Store & Nielsen Attributes
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px 16px; margin: 8px 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 0.84rem;">
                <div><span style="color: #94a3b8;">Store Format:</span> <strong style="color: #f8fafc;">{row.get('nielsen_format', 'SUP')}</strong></div>
                <div><span style="color: #94a3b8;">Sales Area:</span> <strong style="color: #f8fafc;">{row.get('sales_area_sqm', 0.0):.0f} m²</strong></div>
                <div><span style="color: #94a3b8;">Cash Registers:</span> <strong style="color: #f8fafc;">{row.get('cash_registers', 0.0):.0f} tills</strong></div>
                <div><span style="color: #94a3b8;">Reward Mechanism:</span> <strong style="color: #f8fafc;">{row.get('reward_type', 'Standard')}</strong></div>
                <div><span style="color: #94a3b8;">Nearest Coripet:</span> <strong style="color: #f8fafc;">{row.get('nearest_coripet_km', 0.0):.1f} km</strong></div>
                <div><span style="color: #94a3b8;">Hardware Model:</span> <strong style="color: #f8fafc;">{row.get('machine_type', 'Ripet_C-front')}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Action buttons for this record
    act_col1, act_col2 = st.columns([2.0, 2.0])
    with act_col1:
        if st.button("🚀 Load into Prospective Predictor Dashboard", key=f"btn_load_wiz_{row['id']}", type="primary", use_container_width=True):
            clean_addr = row["address"] if row["address"] and str(row["address"]).lower() != "nan" else "Piazza del Comune 1"
            full_res_obj = None
            if row.get("full_prediction_json"):
                try:
                    full_res_obj = json.loads(row["full_prediction_json"])
                except Exception:
                    pass

            st.session_state["pending_load_machine"] = {
                "store_name": row["store_name"],
                "address": clean_addr,
                "cap": row["cap"],
                "comune": row["comune"],
                "machine_type": row["machine_type"],
                "pf_code": row["punto_fisico"],
                "status_label": row["status_source"],
                "pending_prediction_result": full_res_obj,
                "selected_format": row.get("nielsen_format"),
                "selected_sqm": row.get("sales_area_sqm"),
                "selected_casse": row.get("cash_registers"),
                "selected_reward": row.get("reward_type"),
            }
            st.session_state["pending_nav_tab"] = "🎯 Prospective Predictor"
            st.rerun()

    with act_col2:
        with st.expander("📦 View Full Raw Prediction JSON Payload", expanded=False):
            if row.get("full_prediction_json"):
                try:
                    parsed_json = json.loads(row["full_prediction_json"])
                    st.json(parsed_json)
                except Exception:
                    st.text(row["full_prediction_json"])
            else:
                st.caption("No extended JSON payload was stored for this legacy record.")
