"""
Database manager for hiding and restoring SPL and POT machines.
Maintains completely separate tables:
  - hidden_spl_machines
  - hidden_pot_machines
Supports SQLite (data/coripet_rvm.db) and PostgreSQL seamlessly.
"""

import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Set, Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

SQLITE_PATH = ROOT_DIR / "data" / "coripet_rvm.db"

# Cache reachability of Postgres to avoid connection timeout lags
_PG_AVAILABLE = None
_PG_LAST_CHECK = 0


def _is_pg_available() -> bool:
    global _PG_AVAILABLE, _PG_LAST_CHECK
    now = time.time()
    if _PG_AVAILABLE is not None and (now - _PG_LAST_CHECK) < 300:
        return _PG_AVAILABLE

    pg_url = os.getenv("DATABASE_URL", "").strip()
    if not pg_url or not pg_url.startswith("postgres"):
        _PG_AVAILABLE = False
        _PG_LAST_CHECK = now
        return False

    clean_url = pg_url.replace("postgresql+psycopg2://", "postgresql://")
    try:
        import psycopg2
        conn = psycopg2.connect(clean_url, connect_timeout=1)
        conn.close()
        _PG_AVAILABLE = True
    except Exception:
        _PG_AVAILABLE = False

    _PG_LAST_CHECK = now
    return _PG_AVAILABLE


def _get_connection():
    """Returns a SQLite connection or PostgreSQL connection if configured and reachable."""
    if _is_pg_available():
        pg_url = os.getenv("DATABASE_URL", "").strip().replace("postgresql+psycopg2://", "postgresql://")
        try:
            import psycopg2
            conn = psycopg2.connect(pg_url)
            return conn, "postgres"
        except Exception:
            pass

    # Default to SQLite
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_PATH))
    return conn, "sqlite"


def _get_table_name(status_label: str) -> str:
    """Returns table name strictly separated for SPL vs POT."""
    st_clean = str(status_label).upper().strip()
    if "POT" in st_clean:
        return "hidden_pot_machines"
    return "hidden_spl_machines"


def init_hidden_tables():
    """Initializes both hidden_spl_machines and hidden_pot_machines tables."""
    conn, engine_type = _get_connection()
    try:
        cur = conn.cursor()
        for tbl in ["hidden_spl_machines", "hidden_pot_machines"]:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {tbl} (
                    punto_fisico VARCHAR(64) PRIMARY KEY,
                    store_name TEXT,
                    address TEXT,
                    cap VARCHAR(16),
                    comune VARCHAR(128),
                    provincia VARCHAR(16),
                    regione VARCHAR(64),
                    machine_type VARCHAR(64),
                    predicted_bottles REAL DEFAULT 0.0,
                    predicted_month INTEGER DEFAULT 0,
                    volume_tier VARCHAR(64),
                    reward_type VARCHAR(128),
                    dt_fine_sop VARCHAR(32),
                    hidden_at VARCHAR(32)
                );
            """)
        conn.commit()
    finally:
        conn.close()


def get_hidden_pf_set(status_label: str) -> Set[str]:
    """Returns set of PuntoFisico strings currently hidden for SPL or POT."""
    table_name = _get_table_name(status_label)
    conn, engine_type = _get_connection()
    try:
        cur = conn.cursor()
        if engine_type == "sqlite":
            chk = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
            ).fetchone()
            if not chk:
                init_hidden_tables()
        cur.execute(f"SELECT punto_fisico FROM {table_name}")
        rows = cur.fetchall()
        return {str(r[0]).strip() for r in rows if r and r[0]}
    except Exception:
        init_hidden_tables()
        return set()
    finally:
        conn.close()


def get_hidden_machines_df(status_label: str) -> pd.DataFrame:
    """Fetches full DataFrame of hidden machines for the given status."""
    table_name = _get_table_name(status_label)
    conn, engine_type = _get_connection()
    try:
        init_hidden_tables()
        query = f"SELECT * FROM {table_name} ORDER BY hidden_at DESC"
        df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def hide_machine(status_label: str, row_dict: Dict[str, Any]) -> bool:
    """Hides a single machine into the respective table."""
    return hide_machines_batch(status_label, [row_dict])


def hide_machines_batch(status_label: str, rows_list: List[Dict[str, Any]]) -> bool:
    """Hides multiple machines in bulk into the respective table."""
    if not rows_list:
        return False

    table_name = _get_table_name(status_label)
    init_hidden_tables()
    conn, engine_type = _get_connection()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cur = conn.cursor()
        for r in rows_list:
            pf = str(r.get("PuntoFisico", r.get("punto_fisico", ""))).strip()
            if not pf:
                continue

            store_name = str(r.get("ResolvedStoreName", r.get("store_name", "")))
            addr = str(r.get("Indirizzo", r.get("address", "")))
            cap_val = str(r.get("Cap", r.get("cap", "")))
            comune = str(r.get("Comune", r.get("comune", "")))
            prov = str(r.get("Provincia", r.get("provincia", "")))
            reg = str(r.get("Regione", r.get("regione", "")))
            mach = str(r.get("ResolvedMachine", r.get("machine_type", "")))
            b_day = float(r.get("PredictedBottlesDay", r.get("predicted_bottles", 0.0)))
            b_mo = int(r.get("PredictedBottlesMonth", r.get("predicted_month", 0)))
            tier = str(r.get("VolumeTier", r.get("volume_tier", "MEDIO PRENDENTE")))
            reward = str(r.get("RewardType", r.get("reward_type", "Standard Loyalty")))
            dt_sop = str(r.get("DataFineSopralluogoFormatted", r.get("dt_fine_sop", "")))

            if engine_type == "sqlite":
                cur.execute(f"""
                    INSERT OR REPLACE INTO {table_name} (
                        punto_fisico, store_name, address, cap, comune, provincia, regione,
                        machine_type, predicted_bottles, predicted_month, volume_tier,
                        reward_type, dt_fine_sop, hidden_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pf, store_name, addr, cap_val, comune, prov, reg,
                    mach, b_day, b_mo, tier, reward, dt_sop, now_str
                ))
            else:
                cur.execute(f"""
                    INSERT INTO {table_name} (
                        punto_fisico, store_name, address, cap, comune, provincia, regione,
                        machine_type, predicted_bottles, predicted_month, volume_tier,
                        reward_type, dt_fine_sop, hidden_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (punto_fisico) DO UPDATE SET
                        store_name = EXCLUDED.store_name,
                        address = EXCLUDED.address,
                        cap = EXCLUDED.cap,
                        comune = EXCLUDED.comune,
                        provincia = EXCLUDED.provincia,
                        regione = EXCLUDED.regione,
                        machine_type = EXCLUDED.machine_type,
                        predicted_bottles = EXCLUDED.predicted_bottles,
                        predicted_month = EXCLUDED.predicted_month,
                        volume_tier = EXCLUDED.volume_tier,
                        reward_type = EXCLUDED.reward_type,
                        dt_fine_sop = EXCLUDED.dt_fine_sop,
                        hidden_at = EXCLUDED.hidden_at;
                """, (
                    pf, store_name, addr, cap_val, comune, prov, reg,
                    mach, b_day, b_mo, tier, reward, dt_sop, now_str
                ))
        conn.commit()
        return True
    except Exception as ex:
        print(f"Error hiding machines in {table_name}: {ex}")
        return False
    finally:
        conn.close()


def restore_machine(status_label: str, punto_fisico: str) -> bool:
    """Restores (unhides) a single machine from the respective table."""
    return restore_machines_batch(status_label, [punto_fisico])


def restore_machines_batch(status_label: str, pf_list: List[str]) -> bool:
    """Restores (unhides) multiple machines from the respective table."""
    if not pf_list:
        return False

    table_name = _get_table_name(status_label)
    conn, engine_type = _get_connection()
    try:
        cur = conn.cursor()
        clean_pfs = [str(pf).strip() for pf in pf_list if str(pf).strip()]
        if not clean_pfs:
            return False

        if engine_type == "sqlite":
            placeholders = ",".join(["?"] * len(clean_pfs))
            cur.execute(f"DELETE FROM {table_name} WHERE punto_fisico IN ({placeholders})", clean_pfs)
        else:
            placeholders = ",".join(["%s"] * len(clean_pfs))
            cur.execute(f"DELETE FROM {table_name} WHERE punto_fisico IN ({placeholders})", clean_pfs)
        conn.commit()
        return True
    except Exception as ex:
        print(f"Error restoring machines from {table_name}: {ex}")
        return False
    finally:
        conn.close()
