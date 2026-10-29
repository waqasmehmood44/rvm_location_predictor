"""
High-Speed Loader: Ingest Master Daily RVM (1.68M rows) and Test Holdout into PostgreSQL.
Preserves the full 61-column certified schema from master_daily_rvm.parquet / .csv.
"""
import os
import io
import time
from pathlib import Path
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Coripet123*@localhost:5432/coripet_rvm")
PARQUET_PATH = Path("data/processed/master_daily_rvm.parquet")
HOLDOUT_PATH = Path("data/processed/test_holdout.csv")


def get_pg_connection(url: str):
    # Normalize URL for psycopg2
    clean_url = url.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(clean_url)


def create_master_table(cursor):
    print("\n[1/3] Creating table 'master_daily_rvm' with full 61-column schema...")
    cursor.execute("DROP TABLE IF EXISTS master_daily_rvm CASCADE;")
    cursor.execute("""
        CREATE TABLE master_daily_rvm (
            id BIGSERIAL PRIMARY KEY,
            cliente VARCHAR(128),
            giorno VARCHAR(64),
            bottiglie DOUBLE PRECISION,
            presenze DOUBLE PRECISION,
            pf VARCHAR(64),
            codicemacchinaprod VARCHAR(64),
            codicemacchina VARCHAR(64),
            refcoripet VARCHAR(128),
            giorno_dt DATE,
            puntofisico VARCHAR(64),
            indirizzo TEXT,
            cap VARCHAR(16),
            comune VARCHAR(128),
            provincia VARCHAR(16),
            regione VARCHAR(64),
            nazione VARCHAR(32),
            latitudine DOUBLE PRECISION,
            longitudine DOUBLE PRECISION,
            stato VARCHAR(32),
            tipologiapuntofisico VARCHAR(64),
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
            codice_macchina_prod VARCHAR(64),
            tipomacchina VARCHAR(64),
            fornitore VARCHAR(64),
            datainstallazione VARCHAR(64),
            dataprimaattivazione VARCHAR(64),
            historical_match_status VARCHAR(32),
            nielsen_match_status VARCHAR(32),
            nielsen_match_method VARCHAR(64),
            nielsen_match_confidence VARCHAR(32),
            nielsen_country_shop_id DOUBLE PRECISION,
            nielsen_tipo_negozio VARCHAR(64),
            nielsen_superficie_vendita DOUBLE PRECISION,
            nielsen_numero_casse DOUBLE PRECISION,
            nielsen_insegna VARCHAR(128),
            nielsen_gruppo VARCHAR(128),
            nielsen_area DOUBLE PRECISION,
            population DOUBLE PRECISION,
            surface DOUBLE PRECISION,
            pil DOUBLE PRECISION,
            population_density DOUBLE PRECISION,
            pil_per_capita DOUBLE PRECISION,
            municipality_match_status VARCHAR(32),
            puntovendita_eligible INTEGER,
            active_as_of_2026_08_31 INTEGER,
            daily_standard_eligible INTEGER,
            maturity_date VARCHAR(64),
            days_since_activation INTEGER,
            months_since_activation DOUBLE PRECISION,
            maturity_status VARCHAR(32),
            modeling_eligible INTEGER
        );
    """)


def load_parquet_to_table(conn, table_name: str, parquet_file: Path, chunk_size: int = 100000):
    print(f"\n[2/3] Reading {parquet_file.name} and bulk copying into '{table_name}'...")
    start_time = time.time()
    
    df = pd.read_parquet(parquet_file)
    total_rows = len(df)
    print(f"  -> Total records to insert: {total_rows:,}")
    
    # Standardize column names to lowercase
    col_mapping = {
        'CLIENTE': 'cliente', 'GIORNO': 'giorno', 'BOTTIGLIE': 'bottiglie', 'PRESENZE': 'presenze',
        'PF': 'pf', 'CODICEMACCHINAPROD': 'codicemacchinaprod', 'CODICEMACCHINA': 'codicemacchina',
        'REFCORIPET': 'refcoripet', 'giorno_dt': 'giorno_dt', 'PuntoFisico': 'puntofisico',
        'Indirizzo': 'indirizzo', 'Cap': 'cap', 'Comune': 'comune', 'Provincia': 'provincia',
        'Regione': 'regione', 'Nazione': 'nazione', 'Latitudine': 'latitudine', 'Longitudine': 'longitudine',
        'Stato': 'stato', 'TipologiaPuntoFisico': 'tipologiapuntofisico', 'Attributo': 'attributo',
        'Cooperativa': 'cooperativa', 'Filiera': 'filiera', 'Descrizione': 'descrizione',
        'TipoContenitore': 'tipocontenitore', 'ReferenteCommerciale': 'referentecommerciale',
        'ReferenteCoripet': 'referentecoripet', 'Comodatario': 'comodatario', 'NomeFiliera': 'nomefiliera',
        'Raggruppamento': 'raggruppamento', 'Macchina': 'macchina', 'CodiceMacchinaProd': 'codice_macchina_prod',
        'TipoMacchina': 'tipomacchina', 'Fornitore': 'fornitore', 'DataInstallazione': 'datainstallazione',
        'DataPrimaAttivazione': 'dataprimaattivazione', 'historical_match_status': 'historical_match_status',
        'nielsen_match_status': 'nielsen_match_status', 'nielsen_match_method': 'nielsen_match_method',
        'nielsen_match_confidence': 'nielsen_match_confidence', 'nielsen_country_shop_id': 'nielsen_country_shop_id',
        'nielsen_tipo_negozio': 'nielsen_tipo_negozio', 'nielsen_superficie_vendita': 'nielsen_superficie_vendita',
        'nielsen_numero_casse': 'nielsen_numero_casse', 'nielsen_insegna': 'nielsen_insegna',
        'nielsen_gruppo': 'nielsen_gruppo', 'nielsen_area': 'nielsen_area', 'population': 'population',
        'surface': 'surface', 'PIL': 'pil', 'population_density': 'population_density',
        'PIL_per_capita': 'pil_per_capita', 'municipality_match_status': 'municipality_match_status',
        'puntovendita_eligible': 'puntovendita_eligible', 'active_as_of_2026_08_31': 'active_as_of_2026_08_31',
        'daily_standard_eligible': 'daily_standard_eligible', 'maturity_date': 'maturity_date',
        'days_since_activation': 'days_since_activation', 'months_since_activation': 'months_since_activation',
        'maturity_status': 'maturity_status', 'modeling_eligible': 'modeling_eligible'
    }
    df = df.rename(columns=col_mapping)
    columns_order = list(col_mapping.values())
    df = df[columns_order]
    
    # Clean Lat/Long/Cap
    df['latitudine'] = pd.to_numeric(df['latitudine'], errors='coerce')
    df['longitudine'] = pd.to_numeric(df['longitudine'], errors='coerce')
    df['cap'] = df['cap'].fillna('').astype(str).str.split('.').str[0]
    df['giorno_dt'] = pd.to_datetime(df['giorno_dt'], errors='coerce').dt.date

    cursor = conn.cursor()
    cols_joined = ", ".join(columns_order)
    copy_sql = f"COPY {table_name} ({cols_joined}) FROM STDIN WITH CSV HEADER NULL 'NaN'"
    
    num_chunks = int(np.ceil(len(df) / chunk_size))
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        buf = io.StringIO()
        chunk.to_csv(buf, index=False, na_rep='NaN')
        buf.seek(0)
        cursor.copy_expert(copy_sql, buf)
        conn.commit()
        pct = min(100.0, ((i + 1) * chunk_size / total_rows) * 100)
        print(f"  -> Inserted {min((i + 1) * chunk_size, total_rows):,} / {total_rows:,} ({pct:.1f}%)...")

    duration = time.time() - start_time
    print(f"  -> Bulk COPY completed in {duration:.1f} seconds ({total_rows/duration:.0f} rows/sec)!")


def create_indexes(conn, table_name: str):
    print(f"\n[3/3] Creating optimized indexes on '{table_name}'...")
    cursor = conn.cursor()
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_pf ON {table_name}(pf);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_giorno_dt ON {table_name}(giorno_dt);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_machine ON {table_name}(codicemacchina);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_modeling ON {table_name}(modeling_eligible);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_comune ON {table_name}(comune);")
    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_provincia ON {table_name}(provincia);")
    conn.commit()
    print("  -> Indexes created successfully!")


def main():
    print("=" * 80)
    print("CORIPET RVM — POSTGRESQL MASTER DATASET BULK INGESTION")
    print(f"Source file: {PARQUET_PATH.resolve()}")
    print("=" * 80)
    
    conn = get_pg_connection(DB_URL)
    cursor = conn.cursor()
    
    # 1. Create table
    create_master_table(cursor)
    conn.commit()
    
    # 2. Bulk load master dataset (1.68M rows)
    load_parquet_to_table(conn, "master_daily_rvm", PARQUET_PATH)
    
    # 3. Create indexes
    create_indexes(conn, "master_daily_rvm")
    
    # 4. Also create test_holdout table
    if HOLDOUT_PATH.exists():
        print("\n[BONUS] Ingesting test_holdout table into PostgreSQL...")
        cursor.execute("DROP TABLE IF EXISTS test_holdout CASCADE;")
        cursor.execute("CREATE TABLE test_holdout (LIKE master_daily_rvm INCLUDING ALL);")
        conn.commit()
        
        # Load test holdout
        df_hold = pd.read_csv(HOLDOUT_PATH)
        # Convert to parquet in memory or use same logic
        temp_parquet = Path("data/cache/temp_test_holdout.parquet")
        temp_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_hold.to_parquet(temp_parquet, index=False)
        load_parquet_to_table(conn, "test_holdout", temp_parquet)
        if temp_parquet.exists():
            temp_parquet.unlink()
        print("  -> test_holdout successfully loaded into PostgreSQL!")
        
    # Check counts
    cursor.execute("SELECT COUNT(*) FROM master_daily_rvm;")
    master_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM test_holdout;")
    holdout_count = cursor.fetchone()[0]
    
    print("\n" + "=" * 80)
    print(f"VERIFICATION RESULTS:")
    print(f"  * master_daily_rvm: {master_count:,} rows")
    print(f"  * test_holdout:    {holdout_count:,} rows")
    print("=" * 80)
    conn.close()

if __name__ == "__main__":
    main()
