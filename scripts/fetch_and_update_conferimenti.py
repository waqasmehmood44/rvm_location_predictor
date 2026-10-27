"""
Fetch daily bottle collection (conferimenti) data from Coripet API
and update the PostgreSQL raw_daily_conferimenti table.

API endpoint:
  POST https://macchine.coripet.it/CORIPETRVMAPI/api/Conferimento/mt_CaricaConferimentiRaggruppamento
  Header: token: ck%N@b$x1$wNwCsigohC
  Form data: giornoinizio=YYYY-MM-DD, giornofine=YYYY-MM-DD
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

API_TOKEN = "ck" + "%N@b" + "$" + "x1" + "$" + "wNwCsigohC"
API_URL = "https://macchine.coripet.it/CORIPETRVMAPI/api/Conferimento/mt_CaricaConferimentiRaggruppamento"
SQLITE_PATH = Path("data/coripet_rvm.db")


def get_pg_engine():
    db_pass = os.getenv("DB_PASSWORD") or os.getenv("DB_PASS")
    if db_pass:
        from sqlalchemy.engine import URL
        db_user = os.getenv("DB_USER", "waqas")
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_port = int(os.getenv("DB_PORT", "3306"))
        db_name = os.getenv("DB_NAME", "coripet_rvm")
        url_obj = URL.create("mysql+pymysql", username=db_user, password=db_pass, host=db_host, port=db_port, database=db_name)
        return create_engine(url_obj)

    pg_url = os.getenv("DATABASE_URL", "").strip()
    if not pg_url:
        raise ValueError("DATABASE_URL or DB_PASSWORD environment variable is not set.")
    if pg_url.startswith("postgresql://") and "+" not in pg_url.split("://")[0]:
        pg_url = pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif pg_url.startswith("postgres://"):
        pg_url = pg_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif pg_url.startswith("mysql://") and "+" not in pg_url.split("://")[0]:
        pg_url = pg_url.replace("mysql://", "mysql+pymysql://", 1)
    return create_engine(pg_url)


def fetch_api_conferimenti_range(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches conferimenti from the API chunked in 7-day increments to prevent timeout or payload limits.
    """
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()

    if start_dt > end_dt:
        raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date}).")

    headers = {
        "token": API_TOKEN,
        "Content-Type": "application/x-www-form-urlencoded"
    }

    all_records = []
    curr_start = start_dt

    print(f"[API] Starting download from {start_date} to {end_date}...")
    while curr_start <= end_dt:
        curr_end = min(curr_start + datetime.timedelta(days=6), end_dt)
        s_str = curr_start.strftime("%Y-%m-%d")
        e_str = curr_end.strftime("%Y-%m-%d")

        print(f"  -> Fetching window: {s_str} to {e_str}...", end="", flush=True)
        payload = {"giornoinizio": s_str, "giornofine": e_str}

        try:
            resp = requests.post(API_URL, headers=headers, data=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            print(f" Retrieved {len(data):,} records.")
            all_records.extend(data)
        except Exception as ex:
            print(f" ERROR: {ex}")
            raise ex

        curr_start = curr_end + datetime.timedelta(days=1)

    print(f"[API] Total raw records fetched: {len(all_records):,}")
    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    return df


def clean_conferimenti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw API conferimenti dataframe, casts types, and normalizes columns for PostgreSQL.
    """
    if df.empty:
        return df

    out = pd.DataFrame()
    out["pf"] = df["PF"].astype(str).str.strip()
    out["giorno"] = df["GIORNO"].astype(str).str.strip()

    # Parse date
    out["giorno_dt"] = pd.to_datetime(
        out["giorno"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    ).dt.date
    mask_na = out["giorno_dt"].isna()
    if mask_na.any():
        out.loc[mask_na, "giorno_dt"] = pd.to_datetime(
            out.loc[mask_na, "giorno"], dayfirst=True, errors="coerce"
        ).dt.date

    out["bottiglie"] = pd.to_numeric(df.get("BOTTIGLIE"), errors="coerce").fillna(0.0)
    out["presenze"] = pd.to_numeric(df.get("PRESENZE"), errors="coerce").fillna(0.0)
    out["cliente"] = df.get("CLIENTE", "").fillna("").astype(str).str.strip()
    out["codicemacchina"] = df.get("CODICEMACCHINA", "").fillna("").astype(str).str.strip()
    out["codicemacchinaprod"] = df.get("CODICEMACCHINAPROD", "").fillna("").astype(str).str.strip()
    out["refcoripet"] = df.get("REFCORIPET", "").fillna("").astype(str).str.strip()

    # Deduplicate within batch
    before_len = len(out)
    out = out.drop_duplicates(subset=["pf", "giorno_dt", "codicemacchina"])
    after_len = len(out)
    if before_len != after_len:
        print(f"[CLEAN] Deduplicated {before_len - after_len:,} duplicate rows inside API batch.")

    return out


def update_postgres_conferimenti(df_clean: pd.DataFrame, start_date: str, end_date: str):
    """
    Updates PostgreSQL raw_daily_conferimenti table.
    Deletes existing rows in [start_date, end_date] and inserts new records atomically.
    """
    if df_clean.empty:
        print("[POSTGRES] No cleaned records to insert.")
        return

    engine = get_pg_engine()
    with engine.begin() as conn:
        # 1. Delete existing rows in date range
        del_res = conn.execute(
            text("DELETE FROM raw_daily_conferimenti WHERE giorno_dt >= :s AND giorno_dt <= :e;"),
            {"s": start_date, "e": end_date}
        )
        print(f"[POSTGRES] Deleted {del_res.rowcount:,} existing rows between {start_date} and {end_date}.")

        # 2. Bulk insert new cleaned rows
        print(f"[POSTGRES] Inserting {len(df_clean):,} records into raw_daily_conferimenti...")
        df_clean.to_sql(
            "raw_daily_conferimenti",
            con=conn,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000
        )

    print("[POSTGRES] Insertion completed successfully.")


def update_sqlite_conferimenti(df_clean: pd.DataFrame, start_date: str, end_date: str):
    """
    Optionally syncs SQLite local database if it exists.
    """
    if not SQLITE_PATH.exists() or df_clean.empty:
        return

    import sqlite3
    try:
        with sqlite3.connect(str(SQLITE_PATH)) as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM raw_daily_conferimenti WHERE giorno_dt >= ? AND giorno_dt <= ?;",
                (start_date, end_date)
            )
            print(f"[SQLITE] Deleted existing rows between {start_date} and {end_date} in {SQLITE_PATH}.")

            df_sqlite = df_clean.rename(columns={
                "pf": "PF",
                "giorno": "GIORNO",
                "bottiglie": "BOTTIGLIE",
                "presenze": "PRESENZE",
                "cliente": "CLIENTE",
                "codicemacchina": "CODICEMACCHINA",
                "codicemacchinaprod": "CODICEMACCHINAPROD",
                "refcoripet": "REFCORIPET"
            })
            df_sqlite["giorno_dt"] = df_sqlite["giorno_dt"].astype(str)
            df_sqlite.to_sql("raw_daily_conferimenti", conn, if_exists="append", index=False)
            conn.commit()
            print(f"[SQLITE] Synced {len(df_sqlite):,} rows into local SQLite.")
    except Exception as ex:
        print(f"[SQLITE] Warning: SQLite sync failed: {ex}")


def verify_database_state(start_date: str, end_date: str):
    """Prints verification summary of PostgreSQL raw_daily_conferimenti."""
    engine = get_pg_engine()
    with engine.connect() as conn:
        total_rows = conn.execute(text("SELECT COUNT(*) FROM raw_daily_conferimenti;")).scalar()
        min_dt = conn.execute(text("SELECT MIN(giorno_dt) FROM raw_daily_conferimenti;")).scalar()
        max_dt = conn.execute(text("SELECT MAX(giorno_dt) FROM raw_daily_conferimenti;")).scalar()
        range_count = conn.execute(
            text("SELECT COUNT(*) FROM raw_daily_conferimenti WHERE giorno_dt >= :s AND giorno_dt <= :e;"),
            {"s": start_date, "e": end_date}
        ).scalar()

        print("\n" + "=" * 80)
        print("VERIFICATION: POSTGRESQL raw_daily_conferimenti")
        print("=" * 80)
        print(f" * Total Table Rows:       {total_rows:,}")
        print(f" * Min Date:               {min_dt}")
        print(f" * Max Date:               {max_dt}")
        print(f" * Rows in Updated Window: {range_count:,} ({start_date} to {end_date})")

        # Daily breakdown sample for recent 7 days
        print("\n Recent Daily Breakdown:")
        res = conn.execute(text("""
            SELECT giorno_dt, COUNT(*) as rows, ROUND(SUM(bottiglie)::numeric, 0) as total_bottles, ROUND(SUM(presenze)::numeric, 0) as total_users
            FROM raw_daily_conferimenti
            WHERE giorno_dt >= :s
            GROUP BY giorno_dt
            ORDER BY giorno_dt DESC
            LIMIT 10;
        """), {"s": start_date}).fetchall()
        print(f" {'Date':<12} | {'Rows':<8} | {'Total Bottles':<16} | {'Total Users':<12}")
        print(" " + "-" * 55)
        for r in res:
            print(f" {str(r[0]):<12} | {r[1]:<8} | {int(r[2]):<16,} | {int(r[3]):<12,}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fetch live Coripet daily bottle API and update database.")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2026-09-01",
        help="Start date YYYY-MM-DD (default: 2026-09-01)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: yesterday)"
    )
    args = parser.parse_args()

    # Yesterday calculation
    if not args.end_date:
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).date()
        end_date = yesterday.strftime("%Y-%m-%d")
    else:
        end_date = args.end_date

    start_date = args.start_date
    print(f"\n[START] Fetching daily bottle conferimenti from {start_date} to {end_date}...")

    # 1. Fetch from API
    df_raw = fetch_api_conferimenti_range(start_date, end_date)
    if df_raw.empty:
        print("[ABORT] No data returned from API.")
        return

    # 2. Clean and format
    df_clean = clean_conferimenti(df_raw)

    # 3. Update PostgreSQL
    update_postgres_conferimenti(df_clean, start_date, end_date)

    # 4. Sync SQLite
    update_sqlite_conferimenti(df_clean, start_date, end_date)

    # 5. Verify
    verify_database_state(start_date, end_date)
    print("[DONE] Synchronization completed successfully.")


if __name__ == "__main__":
    main()
