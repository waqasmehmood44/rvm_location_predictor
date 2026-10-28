"""Database manager and automated synchronization engine for Coripet RVM Predictor.

Manages 3 local SQLite database tables:
1. `raw_daily_conferimenti`: Daily bottle collection logs fetched from Coripet API (or seeded from CSV).
2. `raw_storico_punto_fisico`: Machine hardware history & site master fetched from Coripet API (or seeded from CSV).
3. `training_master_daily`: Cleaned and merged dataset combining daily logs with effective machine intervals.
"""

import os
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import requests
import pandas as pd
import numpy as np

from src.data.historical_physical_point import enrich_daily_bottles_with_history
from src.data.cleaner import DataCleaner

DB_DEFAULT_PATH = Path("data/coripet_rvm.db")
API_TOKEN = "ck" + "%N@b" + "$" + "x1" + "$" + "wNwCsigohC"

URL_CONFERIMENTI = "https://macchine.coripet.it/CORIPETRVMAPI/api/Conferimento/mt_CaricaConferimentiRaggruppamento"
URL_PUNTI_FISICI = "https://macchine.coripet.it/CORIPETRVMAPI/api/PuntiFisici/mt_CaricaViewStoricoPuntoFisico"


def get_connection(db_path: Path = DB_DEFAULT_PATH) -> sqlite3.Connection:
    """Returns a SQLite connection with WAL mode enabled for high concurrent performance."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: Path = DB_DEFAULT_PATH) -> None:
    """Initializes schemas and indexes for the 3 database tables."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Raw Conferimenti (Daily Bottles) Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_daily_conferimenti (
                PF TEXT,
                GIORNO TEXT,
                giorno_dt TEXT,
                BOTTIGLIE REAL,
                PRESENZE REAL,
                CLIENTE TEXT,
                CODICEMACCHINA TEXT,
                CODICEMACCHINAPROD TEXT,
                REFCORIPET TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (PF, GIORNO, CODICEMACCHINA)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conf_pf ON raw_daily_conferimenti(PF);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conf_giorno_dt ON raw_daily_conferimenti(giorno_dt);")

        # 2. Raw Storico Punto Fisico Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_storico_punto_fisico (
                PuntoFisico TEXT,
                DataInizioRange TEXT,
                DataFineRange TEXT,
                start_dt TEXT,
                end_dt TEXT,
                TipologiaPuntoFisico TEXT,
                NascondiAPP TEXT,
                Indirizzo TEXT,
                Cap TEXT,
                Comune TEXT,
                Provincia TEXT,
                Regione TEXT,
                Nazione TEXT,
                Latitudine REAL,
                Longitudine REAL,
                Stato TEXT,
                Attributo TEXT,
                Cooperativa TEXT,
                Filiera TEXT,
                Descrizione TEXT,
                TipoContenitore TEXT,
                ReferenteCommerciale TEXT,
                ReferenteCoripet TEXT,
                Comodatario TEXT,
                NomeFiliera TEXT,
                Raggruppamento TEXT,
                Macchina TEXT,
                CodiceMacchinaProd TEXT,
                TipoMacchina TEXT,
                Fornitore TEXT,
                DataInstallazione TEXT,
                DataPrimaAttivazione TEXT,
                Macchina_clean TEXT,
                CodiceMacchinaProd_clean TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (PuntoFisico, DataInizioRange, Macchina)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_pf ON raw_storico_punto_fisico(PuntoFisico);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_dates ON raw_storico_punto_fisico(start_dt, end_dt);")

        # 3. Training Master Table (Cleaned & Integrated)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_master_daily (
                PF TEXT,
                GIORNO TEXT,
                giorno_dt TEXT,
                BOTTIGLIE REAL,
                PRESENZE REAL,
                CLIENTE TEXT,
                CODICEMACCHINA TEXT,
                Indirizzo TEXT,
                Cap TEXT,
                Comune TEXT,
                Provincia TEXT,
                Regione TEXT,
                Latitudine REAL,
                Longitudine REAL,
                Stato TEXT,
                TipologiaPuntoFisico TEXT,
                Filiera TEXT,
                NomeFiliera TEXT,
                Raggruppamento TEXT,
                TipoMacchina TEXT,
                Macchina TEXT,
                DataInstallazione TEXT,
                DataPrimaAttivazione TEXT,
                historical_match_status TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (PF, GIORNO, CODICEMACCHINA)
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_train_pf ON training_master_daily(PF);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_train_giorno_dt ON training_master_daily(giorno_dt);")

        conn.commit()


def seed_database_from_csv(
    db_path: Path = DB_DEFAULT_PATH,
    csv_bottles: Path = Path("data/raw/daily_bottles.csv"),
    csv_storico: Path = Path("data/raw/CaricaViewStoricoPuntoFisico.csv"),
    force: bool = False
) -> Dict[str, Any]:
    """
    Initial one-time seed: imports existing historical CSV files into SQLite.
    Skips if data already exists, unless force=True.
    """
    init_db(db_path)
    res = {"bottles_imported": 0, "storico_imported": 0, "status": "skipped"}

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_daily_conferimenti;")
        count_conf = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM raw_storico_punto_fisico;")
        count_sp = cursor.fetchone()[0]

    # 1. Seed Storico Punto Fisico
    if (count_sp == 0 or force) and Path(csv_storico).exists():
        print(f"[DB SEED] Importing storico punto fisico from {csv_storico}...")
        df_sp = pd.read_csv(csv_storico, sep=";", encoding="utf-8-sig", low_memory=False)
        df_sp["PuntoFisico"] = df_sp["PuntoFisico"].astype(str).str.strip()
        df_sp["DataInizioRange"] = df_sp["DataInizioRange"].fillna("").astype(str).str.strip()
        df_sp["DataFineRange"] = df_sp["DataFineRange"].fillna("").astype(str).str.strip()
        df_sp["start_dt"] = pd.to_datetime(df_sp["DataInizioRange"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
        df_sp["end_dt"] = pd.to_datetime(df_sp["DataFineRange"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
        df_sp["Macchina"] = df_sp.get("Macchina", "").fillna("").astype(str).str.strip()
        df_sp["Macchina_clean"] = df_sp["Macchina"]
        df_sp["CodiceMacchinaProd_clean"] = df_sp.get("CodiceMacchinaProd", "").fillna("").astype(str).str.strip()

        # Deduplicate on primary key
        df_sp = df_sp.drop_duplicates(subset=["PuntoFisico", "DataInizioRange", "Macchina"])
        with get_connection(db_path) as conn:
            df_sp.to_sql("raw_storico_punto_fisico", conn, if_exists="replace", index=False)
        res["storico_imported"] = len(df_sp)
        print(f"[DB SEED] Imported {len(df_sp):,} physical point history records.")

    # 2. Seed Daily Conferimenti in Chunks
    if (count_conf == 0 or force) and Path(csv_bottles).exists():
        print(f"[DB SEED] Importing historical daily bottles from {csv_bottles} (may take ~10s)...")
        chunk_size = 150000
        total_rows = 0
        with get_connection(db_path) as conn:
            conn.execute("PRAGMA synchronous = OFF;")
            for chunk in pd.read_csv(csv_bottles, low_memory=False, chunksize=chunk_size):
                chunk["PF"] = chunk["PF"].astype(str).str.strip()
                chunk["GIORNO"] = chunk["GIORNO"].astype(str).str.strip()
                chunk["CODICEMACCHINA"] = chunk["CODICEMACCHINA"].fillna("").astype(str).str.strip()
                chunk["giorno_dt"] = pd.to_datetime(
                    chunk["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
                ).dt.strftime("%Y-%m-%d")
                mask_na = chunk["giorno_dt"].isna()
                if mask_na.any():
                    chunk.loc[mask_na, "giorno_dt"] = pd.to_datetime(
                        chunk.loc[mask_na, "GIORNO"], dayfirst=True, errors="coerce"
                    ).dt.strftime("%Y-%m-%d")

                chunk = chunk.drop_duplicates(subset=["PF", "GIORNO", "CODICEMACCHINA"])
                chunk.to_sql("raw_daily_conferimenti", conn, if_exists="append" if total_rows > 0 else "replace", index=False)
                total_rows += len(chunk)
                print(f"  -> Imported {total_rows:,} rows...")
            conn.commit()
            conn.execute("PRAGMA synchronous = NORMAL;")
        res["bottles_imported"] = total_rows
        print(f"[DB SEED] Complete: Imported {total_rows:,} daily bottle records.")

    res["status"] = "success"
    return res


def sync_storico_punti_fisici(db_path: Path = DB_DEFAULT_PATH) -> int:
    """
    Fetches the COMPLETE physical machine registry from the Coripet API:
    POST https://macchine.coripet.it/CORIPETRVMAPI/api/PuntiFisici/mt_CaricaViewStoricoPuntoFisico
    Completely updates `raw_storico_punto_fisico`.
    """
    init_db(db_path)
    headers = {
        "token": API_TOKEN,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    print("[API SYNC] Fetching complete Punti Fisici registry from Coripet API...")
    resp = requests.post(URL_PUNTI_FISICI, headers=headers, data="", timeout=45)
    resp.raise_for_status()

    payload = resp.json()
    records = payload.get("data", [])
    if not records:
        print("[API SYNC] No records returned from Punti Fisici API.")
        return 0

    df = pd.DataFrame(records)
    df["PuntoFisico"] = df["PuntoFisico"].astype(str).str.strip()
    df["DataInizioRange"] = df["DataInizioRange"].fillna("").astype(str).str.strip()
    df["DataFineRange"] = df["DataFineRange"].fillna("").astype(str).str.strip()
    df["start_dt"] = pd.to_datetime(df["DataInizioRange"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df["end_dt"] = pd.to_datetime(df["DataFineRange"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df["Macchina"] = df.get("Macchina", "").fillna("").astype(str).str.strip()
    df["Macchina_clean"] = df["Macchina"]
    df["CodiceMacchinaProd_clean"] = df.get("CodiceMacchinaProd", "").fillna("").astype(str).str.strip()

    df = df.drop_duplicates(subset=["PuntoFisico", "DataInizioRange", "Macchina"])

    with get_connection(db_path) as conn:
        df.to_sql("raw_storico_punto_fisico", conn, if_exists="replace", index=False)

    print(f"[API SYNC] Updated raw_storico_punto_fisico with {len(df):,} machines.")
    return len(df)


def sync_conferimenti_rolling_days(
    db_path: Path = DB_DEFAULT_PATH,
    days_back: int = 10,
    end_date_str: Optional[str] = None
) -> int:
    """
    Fetches daily bottle conferimenti logs for the previous `days_back` days from Coripet API:
    POST https://macchine.coripet.it/CORIPETRVMAPI/api/Conferimento/mt_CaricaConferimentiRaggruppamento
    Body: giornoinizio=YYYY-MM-DD & giornofine=YYYY-MM-DD
    Upserts into `raw_daily_conferimenti`.
    """
    init_db(db_path)
    now = datetime.datetime.now()
    if end_date_str:
        end_dt = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
    else:
        end_dt = now

    start_dt = end_dt - datetime.timedelta(days=days_back)
    giornoinizio = start_dt.strftime("%Y-%m-%d")
    giornofine = end_dt.strftime("%Y-%m-%d")

    print(f"[API SYNC] Fetching daily bottles from {giornoinizio} to {giornofine} ({days_back} days)...")
    headers = {
        "token": API_TOKEN,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data_payload = {
        "giornoinizio": giornoinizio,
        "giornofine": giornofine
    }

    resp = requests.post(URL_CONFERIMENTI, headers=headers, data=data_payload, timeout=60)
    resp.raise_for_status()

    payload = resp.json()
    records = payload.get("data", [])
    if not records:
        print("[API SYNC] No conferimenti records found for specified date window.")
        return 0

    df = pd.DataFrame(records)
    df["PF"] = df["PF"].astype(str).str.strip()
    df["GIORNO"] = df["GIORNO"].astype(str).str.strip()
    df["CODICEMACCHINA"] = df["CODICEMACCHINA"].fillna("").astype(str).str.strip()
    df["giorno_dt"] = pd.to_datetime(
        df["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    mask_na = df["giorno_dt"].isna()
    if mask_na.any():
        df.loc[mask_na, "giorno_dt"] = pd.to_datetime(
            df.loc[mask_na, "GIORNO"], dayfirst=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    df["BOTTIGLIE"] = pd.to_numeric(df["BOTTIGLIE"], errors="coerce").fillna(0)
    df["PRESENZE"] = pd.to_numeric(df["PRESENZE"], errors="coerce").fillna(0)
    df["CLIENTE"] = df.get("CLIENTE", "").fillna("").astype(str).str.strip()
    df["CODICEMACCHINAPROD"] = df.get("CODICEMACCHINAPROD", "").fillna("").astype(str).str.strip()
    df["REFCORIPET"] = df.get("REFCORIPET", "").fillna("").astype(str).str.strip()

    df = df.drop_duplicates(subset=["PF", "GIORNO", "CODICEMACCHINA"])

    # Upsert into SQLite
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        insert_query = """
            INSERT OR REPLACE INTO raw_daily_conferimenti (
                PF, GIORNO, giorno_dt, BOTTIGLIE, PRESENZE, CLIENTE,
                CODICEMACCHINA, CODICEMACCHINAPROD, REFCORIPET
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                str(r.PF), str(r.GIORNO), str(r.giorno_dt), float(r.BOTTIGLIE), float(r.PRESENZE),
                str(r.CLIENTE), str(r.CODICEMACCHINA), str(r.CODICEMACCHINAPROD), str(r.REFCORIPET)
            )
            for r in df.itertuples(index=False)
        ]
        cursor.executemany(insert_query, rows)
        conn.commit()

    print(f"[API SYNC] Upserted {len(df):,} conferimenti records into raw_daily_conferimenti.")
    return len(df)


def rebuild_training_master_daily(
    db_path: Path = DB_DEFAULT_PATH,
    start_date: Optional[str] = None,
    incremental_days: Optional[int] = None
) -> int:
    """
    Cleans and joins raw_daily_conferimenti with raw_storico_punto_fisico using
    the exact effective date interval and zero-duplication rules from historical_physical_point.py.
    
    If incremental_days is specified (e.g. 10), only processes the most recent N days
    and upserts into training_master_daily in < 2 seconds.
    """
    print(f"[DB TRANSFORM] Updating training_master_daily (incremental_days={incremental_days})...")
    with get_connection(db_path) as conn:
        query_conf = "SELECT * FROM raw_daily_conferimenti"
        if incremental_days:
            cutoff = (datetime.datetime.now() - datetime.timedelta(days=incremental_days)).strftime("%Y-%m-%d")
            query_conf += f" WHERE giorno_dt >= '{cutoff}'"
        elif start_date:
            query_conf += f" WHERE giorno_dt >= '{start_date}'"

        df_daily = pd.read_sql(query_conf, conn)
        df_sp = pd.read_sql("SELECT * FROM raw_storico_punto_fisico", conn)

    if df_daily.empty or df_sp.empty:
        print("[DB TRANSFORM] Insufficient data to build training_master_daily.")
        return 0

    # Parse datetime types needed for exact interval matching
    df_daily["giorno_dt"] = pd.to_datetime(df_daily["giorno_dt"])
    df_sp["start_dt"] = pd.to_datetime(df_sp["start_dt"])
    df_sp["end_dt"] = pd.to_datetime(df_sp["end_dt"])
    df_sp["Macchina_clean"] = df_sp["Macchina"].fillna("").astype(str).str.strip()
    df_sp["CodiceMacchinaProd_clean"] = df_sp.get("CodiceMacchinaProd", "").fillna("").astype(str).str.strip()

    # Stage 1 interval join
    merged_df, _ = enrich_daily_bottles_with_history(df_daily, df_sp)

    # Sanitize and format
    cleaner = DataCleaner()
    merged_df = cleaner.clean_geo_coordinates(merged_df)
    merged_df = cleaner.sanitize_targets(merged_df)
    if "Cap" in merged_df.columns:
        merged_df["Cap"] = cleaner.clean_postal_code(merged_df["Cap"])

    merged_df["giorno_dt"] = merged_df["giorno_dt"].dt.strftime("%Y-%m-%d")

    target_cols = [
        "PF", "GIORNO", "giorno_dt", "BOTTIGLIE", "PRESENZE", "CLIENTE",
        "CODICEMACCHINA", "Indirizzo", "Cap", "Comune", "Provincia", "Regione",
        "Latitudine", "Longitudine", "Stato", "TipologiaPuntoFisico", "Filiera",
        "NomeFiliera", "Raggruppamento", "TipoMacchina", "Macchina",
        "DataInstallazione", "DataPrimaAttivazione", "historical_match_status"
    ]
    present_cols = [c for c in target_cols if c in merged_df.columns]
    final_df = merged_df[present_cols].drop_duplicates(subset=["PF", "GIORNO", "CODICEMACCHINA"])

    with get_connection(db_path) as conn:
        conn.execute("PRAGMA synchronous = OFF;")
        if incremental_days:
            # Upsert incremental rows
            cursor = conn.cursor()
            cols_str = ", ".join(present_cols)
            placeholders = ", ".join(["?"] * len(present_cols))
            upsert_sql = f"INSERT OR REPLACE INTO training_master_daily ({cols_str}) VALUES ({placeholders})"
            cursor.executemany(upsert_sql, [tuple(x) for x in final_df.to_numpy()])
            conn.commit()
        else:
            final_df.to_sql("training_master_daily", conn, if_exists="replace", index=False, chunksize=100000)
            conn.commit()
        conn.execute("PRAGMA synchronous = NORMAL;")

    print(f"[DB TRANSFORM] Successfully saved {len(final_df):,} cleaned rows into training_master_daily.")
    return len(final_df)


def run_full_daily_sync(
    db_path: Path = DB_DEFAULT_PATH,
    days_back: int = 10,
    end_date_str: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the entire daily pipeline:
    1. Fetches complete Punti Fisici registry -> updates `raw_storico_punto_fisico`
    2. Fetches rolling 10 days of Conferimenti -> upserts into `raw_daily_conferimenti`
    3. Cleans and incrementally updates `training_master_daily`
    """
    print("=" * 70)
    print("STARTING CORIPET DAILY DATABASE SYNC PIPELINE")
    print("=" * 70)

    # Ensure tables exist and seed if empty
    seed_database_from_csv(db_path)

    # 1. Update machine registry
    n_sp = sync_storico_punti_fisici(db_path)

    # 2. Update daily bottle logs (rolling window)
    n_conf = sync_conferimenti_rolling_days(db_path, days_back=days_back, end_date_str=end_date_str)

    # 3. Clean and update training table incrementally
    n_train = rebuild_training_master_daily(db_path, incremental_days=days_back + 2)

    print("=" * 70)
    print(f"PIPELINE COMPLETED: SP={n_sp:,} | New/Updated Logs={n_conf:,} | Incremental Training Rows={n_train:,}")
    print("=" * 70)

    return {
        "status": "success",
        "punti_fisici_updated": n_sp,
        "conferimenti_synced": n_conf,
        "training_rows_updated": n_train,
        "timestamp": datetime.datetime.now().isoformat()
    }


if __name__ == "__main__":
    run_full_daily_sync()
