"""
PostgreSQL Migration & Synchronization Utility for Coripet RVM Predictor
Migrates all tables and records from local SQLite (data/coripet_rvm.db) to PostgreSQL.

Usage:
  python migrate_to_postgres.py --force
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

SQLITE_PATH = Path("data/coripet_rvm.db")


def create_postgres_tables(pg_engine, force: bool = False):
    """Creates schemas and optimized indexes in PostgreSQL."""
    print("[POSTGRES] Setting up schemas and indexes...")
    with pg_engine.connect() as conn:
        if force:
            print("[POSTGRES] Recreating tables (--force)...")
            conn.execute(text("DROP TABLE IF EXISTS raw_storico_punto_fisico CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS raw_daily_conferimenti CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS training_master_daily CASCADE;"))
            conn.commit()

        # 1. raw_daily_conferimenti
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_daily_conferimenti (
                id BIGSERIAL PRIMARY KEY,
                pf VARCHAR(64),
                giorno VARCHAR(64),
                giorno_dt DATE,
                bottiglie DOUBLE PRECISION,
                presenze DOUBLE PRECISION,
                cliente VARCHAR(128),
                codicemacchina VARCHAR(64),
                codicemacchinaprod VARCHAR(64),
                refcoripet VARCHAR(128),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conf_pf ON raw_daily_conferimenti(pf);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conf_giorno_dt ON raw_daily_conferimenti(giorno_dt);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_conf_machine ON raw_daily_conferimenti(codicemacchina);"))

        # 2. raw_storico_punto_fisico
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_storico_punto_fisico (
                id SERIAL PRIMARY KEY,
                puntofisico VARCHAR(64),
                datainiziorange VARCHAR(64),
                datafinerange VARCHAR(64),
                start_dt DATE,
                end_dt DATE,
                tipologiapuntofisico VARCHAR(64),
                nascondiapp VARCHAR(16),
                indirizzo TEXT,
                cap VARCHAR(16),
                comune VARCHAR(128),
                provincia VARCHAR(16),
                regione VARCHAR(64),
                nazione VARCHAR(32),
                latitudine DOUBLE PRECISION,
                longitudine DOUBLE PRECISION,
                stato VARCHAR(32),
                attributo VARCHAR(64),
                cooperativa VARCHAR(128),
                filiera VARCHAR(128),
                descrizione TEXT,
                tipocontenitore VARCHAR(64),
                referentecommerciale VARCHAR(128),
                referentecoripet VARCHAR(128),
                comodatario VARCHAR(128),
                nomefiliera VARCHAR(128),
                raggruppamento VARCHAR(128),
                macchina VARCHAR(64),
                codicemacchinaprod VARCHAR(64),
                tipomacchina VARCHAR(128),
                fornitore VARCHAR(128),
                datainstallazione VARCHAR(64),
                dataprimaattivazione VARCHAR(64),
                macchina_clean VARCHAR(64),
                codicemacchinaprod_clean VARCHAR(64),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sp_pf ON raw_storico_punto_fisico(puntofisico);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sp_comune ON raw_storico_punto_fisico(comune);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_sp_dates ON raw_storico_punto_fisico(start_dt, end_dt);"))

        # 3. training_master_daily
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS training_master_daily (
                id BIGSERIAL PRIMARY KEY,
                pf VARCHAR(64),
                giorno VARCHAR(64),
                giorno_dt DATE,
                bottiglie DOUBLE PRECISION,
                presenze DOUBLE PRECISION,
                cliente VARCHAR(128),
                codicemacchina VARCHAR(64),
                indirizzo TEXT,
                cap VARCHAR(16),
                comune VARCHAR(128),
                provincia VARCHAR(16),
                regione VARCHAR(64),
                latitudine DOUBLE PRECISION,
                longitudine DOUBLE PRECISION,
                stato VARCHAR(32),
                tipologiapuntofisico VARCHAR(64),
                filiera VARCHAR(128),
                nomefiliera VARCHAR(128),
                raggruppamento VARCHAR(128),
                tipomacchina VARCHAR(128),
                macchina VARCHAR(64),
                datainstallazione VARCHAR(64),
                dataprimaattivazione VARCHAR(64),
                historical_match_status VARCHAR(64),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_train_pf ON training_master_daily(pf);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_train_giorno_dt ON training_master_daily(giorno_dt);"))
        conn.commit()
    print("[POSTGRES] Table schemas and indexes ready.")


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Cleans empty strings, normalizes names, and coerces numeric and date types for PostgreSQL."""
    df = chunk.copy()
    df.columns = [c.lower() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: None if pd.isna(x) or str(x).strip() in ("", "nan", "None", "NaN", "null") else str(x).strip()
            )

    numeric_cols = ["latitudine", "longitudine", "bottiglie", "presenze"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    date_cols = ["giorno_dt", "start_dt", "end_dt"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    return df


def migrate_table(sqlite_conn, pg_engine, table_name: str, chunk_size: int = 50000, force: bool = False):
    """Streams data from SQLite table to PostgreSQL table in batches."""
    print(f"\n[MIGRATE] Processing table: {table_name}...")
    cursor = sqlite_conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    total_records = cursor.fetchone()[0]
    print(f"  -> Total records in SQLite: {total_records:,}")

    if total_records == 0:
        print("  -> Table is empty, skipping.")
        return

    with pg_engine.connect() as conn:
        res = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
        pg_count = res.scalar()
        if pg_count > 0:
            print(f"  -> Found {pg_count:,} existing rows in PostgreSQL.")
            if force:
                conn.execute(text(f"TRUNCATE TABLE {table_name};"))
                conn.commit()
                print(f"  -> Table {table_name} truncated (--force).")
            else:
                answer = input(f"  Do you want to truncate and re-copy {table_name}? [y/N]: ").strip().lower()
                if answer == "y":
                    conn.execute(text(f"TRUNCATE TABLE {table_name};"))
                    conn.commit()
                    print(f"  -> Table {table_name} truncated.")
                else:
                    print(f"  -> Skipping migration for {table_name}.")
                    return

    copied = 0
    query = f"SELECT * FROM {table_name};"
    for chunk in pd.read_sql(query, sqlite_conn, chunksize=chunk_size):
        cleaned = clean_chunk(chunk)
        # Using chunksize=1000 ensures parameters stay safely below PostgreSQL's 65,535 limit
        cleaned.to_sql(
            table_name.lower(),
            pg_engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1000
        )
        copied += len(cleaned)
        pct = (copied / total_records) * 100
        print(f"  -> Copied {copied:,} / {total_records:,} ({pct:.1f}%)...")

    print(f"  -> Successfully migrated {table_name}: {copied:,} rows.")


def run_migration(pg_url: str, force: bool = False):
    if not SQLITE_PATH.exists():
        print(f"Error: SQLite source database not found at {SQLITE_PATH.resolve()}")
        sys.exit(1)

    # Normalize to postgresql+psycopg2 driver if plain postgresql:// is provided
    if pg_url.startswith("postgresql://") and "+" not in pg_url.split("://")[0]:
        pg_url = pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif pg_url.startswith("postgres://"):
        pg_url = pg_url.replace("postgres://", "postgresql+psycopg2://", 1)

    print("=" * 80)
    print("MIGRATING CORIPET RVM DATA TO POSTGRESQL")
    print(f"Source: {SQLITE_PATH.resolve()}")
    safe_url = pg_url.split("@")[-1] if "@" in pg_url else pg_url
    print(f"Target: postgresql://*****@{safe_url}")
    print("=" * 80)

    pg_engine = create_engine(pg_url, pool_pre_ping=True)
    create_postgres_tables(pg_engine, force=force)

    with sqlite3.connect(str(SQLITE_PATH)) as sqlite_conn:
        migrate_table(sqlite_conn, pg_engine, "raw_storico_punto_fisico", force=force)
        migrate_table(sqlite_conn, pg_engine, "training_master_daily", chunk_size=50000, force=force)
        migrate_table(sqlite_conn, pg_engine, "raw_daily_conferimenti", chunk_size=50000, force=force)

    print("\n" + "=" * 80)
    print("MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Coripet RVM SQLite DB to PostgreSQL")
    parser.add_argument("--url", type=str, default=None, help="PostgreSQL connection URL")
    parser.add_argument("--force", action="store_true", help="Automatically truncate/recreate tables without prompting")
    args = parser.parse_args()

    pg_url = args.url or os.getenv("DATABASE_URL")
    if not pg_url or not (pg_url.startswith("postgresql") or pg_url.startswith("postgres")):
        print("\nPlease provide a valid PostgreSQL connection URL!")
        print("Example:")
        print("  python migrate_to_postgres.py --url postgresql://postgres:password@localhost:5432/coripet_rvm")
        print("Or set DATABASE_URL in a .env file.")
        sys.exit(1)

    run_migration(pg_url, force=args.force)
