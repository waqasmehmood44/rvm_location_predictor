"""
Syncs PostgreSQL tables to data/coripet_rvm.db (SQLite).
Enables seamless deployment on hosting environments without PostgreSQL daemons.
"""
import os
import time
import sqlite3
import psycopg2
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

SQLITE_PATH = ROOT_DIR / "data" / "coripet_rvm.db"

def sync_sqlite():
    print("=" * 70)
    print("SYNCING POSTGRESQL TABLES TO SQLITE (data/coripet_rvm.db)")
    print("=" * 70)

    pg_url = os.getenv("DATABASE_URL", "postgresql://postgres:Coripet123*@localhost:5432/coripet_rvm")
    if "+" in pg_url:
        pg_url = pg_url.replace("postgresql+psycopg2://", "postgresql://")

    pg_conn = psycopg2.connect(pg_url)
    sq_conn = sqlite3.connect(str(SQLITE_PATH))

    # 1. Sync raw_storico_punto_fisico
    print("\n1. Syncing raw_storico_punto_fisico...")
    t0 = time.time()
    df_storico = pd.read_sql("SELECT * FROM raw_storico_punto_fisico;", pg_conn)
    df_storico.to_sql("raw_storico_punto_fisico", sq_conn, if_exists="replace", index=False)
    sq_conn.execute("CREATE INDEX IF NOT EXISTS idx_storico_pf ON raw_storico_punto_fisico(puntofisico);")
    sq_conn.execute("CREATE INDEX IF NOT EXISTS idx_storico_mach ON raw_storico_punto_fisico(macchina);")
    sq_conn.commit()
    print(f"--> Synced {len(df_storico):,} rows in {time.time() - t0:.2f}s")

    # 2. Sync master_daily_rvm
    print("\n2. Syncing master_daily_rvm...")
    t0 = time.time()
    
    # Create SQLite table schema
    sq_conn.execute("DROP TABLE IF EXISTS master_daily_rvm;")
    sq_conn.execute("""
        CREATE TABLE master_daily_rvm (
            id INTEGER PRIMARY KEY,
            cliente TEXT,
            giorno TEXT,
            giorno_dt TEXT,
            bottiglie REAL,
            presenze REAL,
            pf TEXT,
            macchina TEXT,
            codicemacchinaprod TEXT,
            codicemacchina TEXT,
            comune TEXT,
            regione TEXT,
            tipomacchina TEXT,
            maturity_status TEXT
        );
    """)

    cur_pg = pg_conn.cursor(name="cursor_stream")
    cur_pg.itersize = 50000
    cur_pg.execute("""
        SELECT id, cliente, giorno, CAST(giorno_dt AS TEXT), bottiglie, presenze,
               pf, macchina, codicemacchinaprod, codicemacchina, comune, regione,
               tipomacchina, maturity_status
        FROM master_daily_rvm;
    """)

    insert_sql = """
        INSERT INTO master_daily_rvm (
            id, cliente, giorno, giorno_dt, bottiglie, presenze,
            pf, macchina, codicemacchinaprod, codicemacchina, comune, regione,
            tipomacchina, maturity_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    total_inserted = 0
    while True:
        rows = cur_pg.fetchmany(50000)
        if not rows:
            break
        sq_conn.executemany(insert_sql, rows)
        sq_conn.commit()
        total_inserted += len(rows)
        print(f"--> Inserted {total_inserted:,} rows...")

    print(f"--> Finished master_daily_rvm in {time.time() - t0:.2f}s")

    # 3. Create B-Tree Indexes on SQLite
    print("\n3. Creating B-Tree indexes on SQLite...")
    t0 = time.time()
    sq_conn.execute("CREATE INDEX IF NOT EXISTS idx_sqlite_master_pf ON master_daily_rvm(pf);")
    sq_conn.execute("CREATE INDEX IF NOT EXISTS idx_sqlite_master_mach ON master_daily_rvm(macchina);")
    sq_conn.execute("CREATE INDEX IF NOT EXISTS idx_sqlite_master_codprod ON master_daily_rvm(codicemacchinaprod);")
    sq_conn.execute("CREATE INDEX IF NOT EXISTS idx_sqlite_master_date ON master_daily_rvm(giorno_dt);")
    sq_conn.commit()
    print(f"--> Indexes created in {time.time() - t0:.2f}s")

    sq_conn.close()
    pg_conn.close()
    print("\nSQLite database ready at:", SQLITE_PATH)

if __name__ == "__main__":
    sync_sqlite()
