"""
Database manager for hiding and restoring SPL and POT machines.
Strictly uses production databases: MariaDB / MySQL (Plesk) and PostgreSQL.
SQLite has been completely removed.
"""

import os
from datetime import datetime
from typing import Set, Dict, Any, List, Tuple
from urllib.parse import urlparse
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _get_connection() -> Tuple[Any, str]:
    """
    Connects strictly to production database (MariaDB/MySQL or PostgreSQL).
    No SQLite fallback.
    Returns (connection_object, 'mysql' | 'postgres').
    """
    db_url = os.getenv("DATABASE_URL", "").strip()

    # 1. MariaDB / MySQL via URL (Plesk Standard)
    if db_url and any(db_url.startswith(prefix) for prefix in ["mysql://", "mysql+pymysql://", "mariadb://"]):
        import pymysql
        clean_url = db_url.replace("mysql+pymysql://", "mysql://").replace("mariadb://", "mysql://")
        parsed = urlparse(clean_url)
        conn = pymysql.connect(
            host=parsed.hostname or "localhost",
            port=parsed.port or 3306,
            user=parsed.username or "waqas",
            password=parsed.password or "",
            database=parsed.path.lstrip("/") or "coripet_rvm",
            charset="utf8mb4",
            autocommit=True
        )
        return conn, "mysql"

    # 2. PostgreSQL via URL
    if db_url and any(db_url.startswith(prefix) for prefix in ["postgres://", "postgresql://", "postgresql+psycopg2://"]):
        import psycopg2
        clean_url = db_url.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(clean_url, connect_timeout=5)
        return conn, "postgres"

    # 3. Direct Environment Variables (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    db_user = os.getenv("DB_USER", "").strip()
    if db_user:
        import pymysql
        conn = pymysql.connect(
            host=os.getenv("DB_HOST", "localhost").strip(),
            port=int(os.getenv("DB_PORT", "3306")),
            user=db_user,
            password=os.getenv("DB_PASSWORD", "").strip(),
            database=os.getenv("DB_NAME", "coripet_rvm").strip(),
            charset="utf8mb4",
            autocommit=True
        )
        return conn, "mysql"

    # No SQLite fallback - explicitly report missing configuration
    raise ConnectionError(
        "No production database configured! Please set DATABASE_URL in .env "
        "(e.g. DATABASE_URL=mysql+pymysql://waqas:<password>@localhost:3306/coripet_rvm)"
    )


def _get_table_name(status_label: str) -> str:
    """Returns table name strictly separated for SPL vs POT."""
    st_clean = str(status_label).upper().strip()
    if "POT" in st_clean:
        return "hidden_pot_machines"
    return "hidden_spl_machines"


def init_hidden_tables():
    """Initializes both hidden_spl_machines and hidden_pot_machines tables in MariaDB or PostgreSQL."""
    conn, engine_type = _get_connection()
    try:
        cur = conn.cursor()
        for tbl in ["hidden_spl_machines", "hidden_pot_machines"]:
            if engine_type == "mysql":
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS `{tbl}` (
                        `punto_fisico` VARCHAR(64) NOT NULL,
                        `store_name` TEXT DEFAULT NULL,
                        `address` TEXT DEFAULT NULL,
                        `cap` VARCHAR(16) DEFAULT NULL,
                        `comune` VARCHAR(128) DEFAULT NULL,
                        `provincia` VARCHAR(16) DEFAULT NULL,
                        `regione` VARCHAR(64) DEFAULT NULL,
                        `machine_type` VARCHAR(64) DEFAULT NULL,
                        `predicted_bottles` FLOAT DEFAULT 0.0,
                        `predicted_month` INT DEFAULT 0,
                        `volume_tier` VARCHAR(64) DEFAULT NULL,
                        `reward_type` VARCHAR(128) DEFAULT NULL,
                        `dt_fine_sop` VARCHAR(32) DEFAULT NULL,
                        `hidden_at` VARCHAR(32) DEFAULT NULL,
                        PRIMARY KEY (`punto_fisico`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
                """)
            else:
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
        if hasattr(conn, "commit"):
            conn.commit()
    finally:
        conn.close()


def get_hidden_pf_set(status_label: str) -> Set[str]:
    """Returns set of PuntoFisico strings currently hidden for SPL or POT."""
    table_name = _get_table_name(status_label)
    try:
        conn, engine_type = _get_connection()
    except Exception as ex:
        print(f"[hidden_manager] Database connection error: {ex}")
        return set()

    try:
        cur = conn.cursor()
        if engine_type == "mysql":
            cur.execute(f"SELECT `punto_fisico` FROM `{table_name}`")
        else:
            cur.execute(f"SELECT punto_fisico FROM {table_name}")
        rows = cur.fetchall()
        return {str(r[0]).strip() for r in rows if r and r[0]}
    except Exception as ex:
        print(f"[hidden_manager] Error querying {table_name}: {ex}")
        try:
            init_hidden_tables()
        except Exception:
            pass
        return set()
    finally:
        conn.close()


def get_hidden_machines_df(status_label: str) -> pd.DataFrame:
    """Fetches full DataFrame of hidden machines for the given status."""
    table_name = _get_table_name(status_label)
    try:
        conn, engine_type = _get_connection()
    except Exception as ex:
        print(f"[hidden_manager] Database connection error: {ex}")
        return pd.DataFrame()

    try:
        quote = "`" if engine_type == "mysql" else ""
        query = f"SELECT * FROM {quote}{table_name}{quote} ORDER BY hidden_at DESC"
        df = pd.read_sql(query, conn)
        return df
    except Exception as ex:
        print(f"[hidden_manager] Error reading {table_name}: {ex}")
        try:
            init_hidden_tables()
        except Exception:
            pass
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

            if engine_type == "mysql":
                cur.execute(f"""
                    INSERT INTO `{table_name}` (
                        `punto_fisico`, `store_name`, `address`, `cap`, `comune`, `provincia`, `regione`,
                        `machine_type`, `predicted_bottles`, `predicted_month`, `volume_tier`,
                        `reward_type`, `dt_fine_sop`, `hidden_at`
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        `store_name` = VALUES(`store_name`),
                        `address` = VALUES(`address`),
                        `cap` = VALUES(`cap`),
                        `comune` = VALUES(`comune`),
                        `provincia` = VALUES(`provincia`),
                        `regione` = VALUES(`regione`),
                        `machine_type` = VALUES(`machine_type`),
                        `predicted_bottles` = VALUES(`predicted_bottles`),
                        `predicted_month` = VALUES(`predicted_month`),
                        `volume_tier` = VALUES(`volume_tier`),
                        `reward_type` = VALUES(`reward_type`),
                        `dt_fine_sop` = VALUES(`dt_fine_sop`),
                        `hidden_at` = VALUES(`hidden_at`);
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
        if hasattr(conn, "commit"):
            conn.commit()
        return True
    except Exception as ex:
        print(f"[hidden_manager] Error hiding machines in {table_name}: {ex}")
        raise ex
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

        if engine_type == "mysql":
            placeholders = ",".join(["%s"] * len(clean_pfs))
            cur.execute(f"DELETE FROM `{table_name}` WHERE `punto_fisico` IN ({placeholders})", clean_pfs)
        else:
            placeholders = ",".join(["%s"] * len(clean_pfs))
            cur.execute(f"DELETE FROM {table_name} WHERE punto_fisico IN ({placeholders})", clean_pfs)
        if hasattr(conn, "commit"):
            conn.commit()
        return True
    except Exception as ex:
        print(f"[hidden_manager] Error restoring machines from {table_name}: {ex}")
        raise ex
    finally:
        conn.close()
