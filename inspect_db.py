"""
Database Inspector for Coripet RVM Database (Supports SQLite & PostgreSQL)
Run: python inspect_db.py
"""
import os
import sqlite3
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path("data/coripet_rvm.db")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def inspect_sqlite():
    if not DB_PATH.exists():
        print(f"SQLite database not found at {DB_PATH.resolve()}")
        return

    print("=" * 80)
    print("CORIPET RVM LOCAL SQLITE DATABASE")
    print(f"Location: {DB_PATH.resolve()} ({DB_PATH.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 80)

    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]

            cursor.execute(f"PRAGMA table_info({table});")
            cols_info = cursor.fetchall()
            col_names = [c[1] for c in cols_info]

            print(f"\n[TABLE]: {table}")
            print(f"   * Total Rows: {count:,}")
            print(f"   * Total Columns: {len(col_names)}")
            print(f"   * Columns: {', '.join(col_names[:8])}{' ...' if len(col_names) > 8 else ''}")

            df_preview = pd.read_sql(f"SELECT * FROM {table} LIMIT 3;", conn)
            print("\n   Sample Data (First 3 rows):")
            preview_cols = [c for c in ["PF", "GIORNO", "giorno_dt", "BOTTIGLIE", "PRESENZE", "CODICEMACCHINA", "PuntoFisico", "Comune", "Provincia", "Stato", "Macchina", "TipoMacchina"] if c in df_preview.columns]
            if not preview_cols:
                preview_cols = df_preview.columns[:6]
            print(df_preview[preview_cols].to_string(index=False))
            print("-" * 80)


def inspect_postgres(pg_url: str):
    if pg_url.startswith("postgresql://") and "+" not in pg_url.split("://")[0]:
        pg_url = pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif pg_url.startswith("postgres://"):
        pg_url = pg_url.replace("postgres://", "postgresql+psycopg2://", 1)

    safe_url = pg_url.split("@")[-1] if "@" in pg_url else pg_url
    print("=" * 80)
    print("CORIPET RVM POSTGRESQL DATABASE SERVER")
    print(f"Connection: postgresql://*****@{safe_url}")
    print("=" * 80)

    engine = create_engine(pg_url)

    with engine.connect() as conn:
        res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"))
        tables = [r[0] for r in res.fetchall()]
        for table in tables:
            try:
                res = conn.execute(text(f"SELECT COUNT(*) FROM {table};"))
                count = res.scalar()
                df_preview = pd.read_sql(text(f"SELECT * FROM {table} LIMIT 3;"), conn)

                print(f"\n[TABLE]: {table}")
                print(f"   * Total Rows: {count:,}")
                print(f"   * Total Columns: {len(df_preview.columns)}")
                print(f"   * Columns: {', '.join(list(df_preview.columns)[:8])}{' ...' if len(df_preview.columns) > 8 else ''}")
                print("\n   Sample Data (First 3 rows):")
                preview_cols = [c for c in ["pf", "giorno", "giorno_dt", "bottiglie", "presenze", "codicemacchina", "puntofisico", "comune", "provincia", "stato", "macchina", "tipomacchina"] if c in df_preview.columns]
                if not preview_cols:
                    preview_cols = df_preview.columns[:6]
                print(df_preview[preview_cols].to_string(index=False))
                print("-" * 80)
            except Exception as e:
                print(f"\n[TABLE]: {table} (Not found or error: {e})")


if __name__ == "__main__":
    if DATABASE_URL and (DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")):
        inspect_postgres(DATABASE_URL)
    else:
        inspect_sqlite()
