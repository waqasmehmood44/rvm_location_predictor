"""
Production Script: Prepare and Add New Daily Bottle Collection Records to master_daily_rvm

Workflow:
1. Queries new daily records from PostgreSQL `raw_daily_conferimenti` (>= 2026-09-01).
2. Enriches with CaricaViewStoricoPuntoFisico history interval joins.
3. Matches Nielsen retail store census and municipality demographic/macro features.
4. Calculates four-gate business eligibility and ramp-up maturity metrics.
5. Filters for PUNTOVENDITA (retail stores).
6. Atomically inserts the new certified records into PostgreSQL `master_daily_rvm`.
7. Updates local `data/processed/master_daily_rvm.parquet` for local training and pipeline consistency.
8. Verifies database state and displays summary statistics.
"""

import os
import io
import sys
import time
import argparse
from pathlib import Path
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Ensure root directory in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from src.data.load_data import load_filenielsen, load_macro_datasets
from src.data.historical_physical_point import enrich_daily_bottles_with_history
from src.data.nielsen_matching import match_physical_points_to_nielsen
from src.data.municipality_matching import build_municipality_feature_store, enrich_with_municipality_data

PARQUET_PATH = Path("data/processed/master_daily_rvm.parquet")


def get_db_engine():
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


def run_pipeline(start_date: str = "2026-09-01", end_date: str = "2026-10-21"):
    print("=" * 80)
    print("PREPARING NEW DATA FOR MASTER_DAILY_RVM")
    print(f"Date Window: {start_date} to {end_date}")
    print("=" * 80)

    engine = get_db_engine()

    # 1. Load sp_df from postgres raw_storico_punto_fisico
    print("\n[1/6] Loading Physical Point History from raw_storico_punto_fisico...")
    with engine.connect() as conn:
        sp_df = pd.read_sql("SELECT * FROM raw_storico_punto_fisico", conn)
        col_map = {c.lower(): c for c in [
            "PuntoFisico", "DataInizioRange", "DataFineRange", "TipologiaPuntoFisico",
            "NascondiAPP", "Indirizzo", "Cap", "Comune", "Provincia", "Regione", "Nazione",
            "Latitudine", "Longitudine", "Stato", "Attributo", "Cooperativa", "Filiera",
            "Descrizione", "TipoContenitore", "ReferenteCommerciale", "ReferenteCoripet",
            "Comodatario", "NomeFiliera", "Raggruppamento", "Macchina", "CodiceMacchinaProd",
            "TipoMacchina", "Fornitore", "DataInstallazione", "DataPrimaAttivazione",
            "start_dt", "end_dt", "Macchina_clean", "CodiceMacchinaProd_clean"
        ]}
        sp_df.rename(columns=col_map, inplace=True)
        sp_df["start_dt"] = pd.to_datetime(sp_df["start_dt"], errors="coerce")
        sp_df["end_dt"] = pd.to_datetime(sp_df["end_dt"], errors="coerce")
        sp_df["Macchina_clean"] = sp_df["Macchina"].fillna("").astype(str).str.strip()
        sp_df["CodiceMacchinaProd_clean"] = sp_df["CodiceMacchinaProd"].fillna("").astype(str).str.strip()

        # Query new raw daily conferimenti
        print(f"\n[2/6] Querying new daily conferimenti from PostgreSQL for {start_date} to {end_date}...")
        query = text("""
            SELECT pf, giorno, giorno_dt, bottiglie, presenze, cliente, codicemacchina, codicemacchinaprod, refcoripet
            FROM raw_daily_conferimenti
            WHERE giorno_dt >= :s AND giorno_dt <= :e
            ORDER BY giorno_dt, pf;
        """)
        daily_new = pd.read_sql(query, conn, params={"s": start_date, "e": end_date})
        daily_new.rename(columns={
            "pf": "PF", "giorno": "GIORNO", "bottiglie": "BOTTIGLIE", "presenze": "PRESENZE",
            "cliente": "CLIENTE", "codicemacchina": "CODICEMACCHINA",
            "codicemacchinaprod": "CODICEMACCHINAPROD", "refcoripet": "REFCORIPET"
        }, inplace=True)
        daily_new["giorno_dt"] = pd.to_datetime(daily_new["giorno_dt"], errors="coerce")

        # Get existing master_daily_rvm columns
        sample_master = pd.read_sql("SELECT * FROM master_daily_rvm LIMIT 1", conn)
        target_cols = [c for c in sample_master.columns if c != "id"]

    print(f"  -> Retrieved {len(daily_new):,} daily collection logs across {daily_new['PF'].nunique():,} machines.")
    if daily_new.empty:
        print("[ABORT] No new records found in raw_daily_conferimenti for the given date range.")
        return

    # 3. Merging with Physical Point History
    print("\n[3/6] Merging with Physical Point History (Stage 1)...")
    stage1_df, stage1_stats = enrich_daily_bottles_with_history(daily_new, sp_df)
    print(f"  -> Match breakdown: {stage1_df['historical_match_status'].value_counts().to_dict()}")

    # 4. Nielsen Store Census Matching
    print("\n[4/6] Matching with Nielsen Retail Store Census (Stage 3)...")
    fn_df = load_filenielsen()
    nielsen_lookup, _ = match_physical_points_to_nielsen(sp_df, fn_df)
    stage3_df = pd.merge(stage1_df, nielsen_lookup, on="PuntoFisico", how="left")

    # 5. Municipality Demographics & Economics
    print("\n[5/6] Enriching with Municipality Demographics & Macro Economics (Stage 4)...")
    pil_df, pop_df, surf_df = load_macro_datasets()
    macro_feature_store = build_municipality_feature_store(pil_df, pop_df, surf_df)
    stage4_df, _ = enrich_with_municipality_data(stage3_df, macro_feature_store, comune_col="Comune")

    # 6. Business Eligibility & Ramp-up Maturity Gates
    print("\n[6/6] Computing Business Eligibility Gates & Ramp-up Maturity...")
    stage4_df["puntovendita_eligible"] = (stage4_df["TipologiaPuntoFisico"] == "PUNTOVENDITA").astype(int)

    # Active network as of reference date
    ref_date = pd.Timestamp("2026-08-31")
    mask_ref = (sp_df["start_dt"] <= ref_date) & (sp_df["end_dt"].isna() | (sp_df["end_dt"] >= ref_date))
    sp_ref = sp_df[mask_ref].copy()
    active_pfs_all = set(sp_ref[sp_ref["Stato"] == "ATTVO"]["PuntoFisico"].unique())
    stage4_df["active_as_of_2026_08_31"] = stage4_df["PF"].isin(active_pfs_all).astype(int)

    stage4_df["daily_standard_eligible"] = (stage4_df["Attributo"] == "STANDARD").astype(int)

    activation_dt = pd.to_datetime(stage4_df["DataPrimaAttivazione"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    mask_act_na = activation_dt.isna()
    if mask_act_na.any():
        activation_dt.loc[mask_act_na] = pd.to_datetime(stage4_df.loc[mask_act_na, "DataPrimaAttivazione"], dayfirst=True, errors="coerce")

    giorno_dt = pd.to_datetime(stage4_df["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    mask_g_na = giorno_dt.isna()
    if mask_g_na.any():
        giorno_dt.loc[mask_g_na] = pd.to_datetime(stage4_df.loc[mask_g_na, "GIORNO"], dayfirst=True, errors="coerce")

    maturity_date = activation_dt + pd.DateOffset(months=3)
    days_since_act = (giorno_dt - activation_dt).dt.days
    months_since_act = (days_since_act / 30.4375).round(2)

    stage4_df["maturity_date"] = maturity_date.dt.strftime("%d/%m/%Y")
    stage4_df["days_since_activation"] = days_since_act.fillna(-1).astype(int)
    stage4_df["months_since_activation"] = months_since_act.fillna(-1.0)

    stage4_df["maturity_status"] = "UNKNOWN"
    valid_act = ~activation_dt.isna()
    stage4_df.loc[valid_act & (giorno_dt < maturity_date), "maturity_status"] = "IMMATURE"
    stage4_df.loc[valid_act & (giorno_dt >= maturity_date), "maturity_status"] = "MATURE"

    stage4_df["modeling_eligible"] = (
        (stage4_df["puntovendita_eligible"] == 1)
        & (stage4_df["active_as_of_2026_08_31"] == 1)
        & (stage4_df["daily_standard_eligible"] == 1)
        & (stage4_df["maturity_status"] == "MATURE")
    ).astype(int)

    # Filter for PUNTOVENDITA (Retail Supermarket Sites)
    master_new = stage4_df[stage4_df["puntovendita_eligible"] == 1].copy()

    # Use exact column mapping and ordering from load_master_to_postgres.py
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
        'surface': 'surface', 'pil': 'pil', 'population_density': 'population_density',
        'pil_per_capita': 'pil_per_capita', 'municipality_match_status': 'municipality_match_status',
        'puntovendita_eligible': 'puntovendita_eligible', 'active_as_of_2026_08_31': 'active_as_of_2026_08_31',
        'daily_standard_eligible': 'daily_standard_eligible', 'maturity_date': 'maturity_date',
        'days_since_activation': 'days_since_activation', 'months_since_activation': 'months_since_activation',
        'maturity_status': 'maturity_status', 'modeling_eligible': 'modeling_eligible'
    }

    columns_order = list(col_mapping.values())

    # Map case-insensitively from master_new to expected PostgreSQL columns
    col_map_lower = {c.lower(): c for c in master_new.columns}
    for tgt_col in columns_order:
        if tgt_col not in master_new.columns:
            if tgt_col in col_map_lower:
                master_new[tgt_col] = master_new[col_map_lower[tgt_col]]
            else:
                for src_col, target in col_mapping.items():
                    if target == tgt_col and src_col in master_new.columns:
                        master_new[tgt_col] = master_new[src_col]
                        break

    master_insert_df = master_new[columns_order].copy()

    # Clean data types
    master_insert_df['latitudine'] = pd.to_numeric(master_insert_df['latitudine'], errors='coerce')
    master_insert_df['longitudine'] = pd.to_numeric(master_insert_df['longitudine'], errors='coerce')
    master_insert_df['cap'] = master_insert_df['cap'].fillna('').astype(str).str.split('.').str[0]
    master_insert_df['giorno_dt'] = pd.to_datetime(master_insert_df['giorno_dt'], errors='coerce').dt.date

    print(f"  -> Prepared {len(master_insert_df):,} PUNTOVENDITA records across all {len(columns_order)} features.")

    # Ingest into Database using universal SQLAlchemy batch insertion
    print("\n[DB] Ingesting into 'master_daily_rvm' table...")
    with engine.begin() as conn:
        # Delete any existing records in window to prevent duplication
        del_res = conn.execute(
            text("DELETE FROM master_daily_rvm WHERE giorno_dt >= :s AND giorno_dt <= :e;"),
            {"s": start_date, "e": end_date}
        )
        if del_res.rowcount > 0:
            print(f"  -> Removed {del_res.rowcount:,} existing rows in {start_date} to {end_date}.")

        print(f"  -> Inserting {len(master_insert_df):,} records into master_daily_rvm...")
        master_insert_df.to_sql(
            "master_daily_rvm",
            con=conn,
            if_exists="append",
            index=False,
            chunksize=5000
        )
        print(f"  -> Successfully added {len(master_insert_df):,} records into 'master_daily_rvm'.")

    # Save local Parquet incremental partition
    inc_parquet_path = Path("data/processed/master_daily_rvm_new_2026.parquet")
    try:
        print(f"\n[PARQUET] Saving new partition to {inc_parquet_path}...")
        master_insert_df.to_parquet(inc_parquet_path, index=False, engine="pyarrow")
        print(f"  -> Saved {len(master_insert_df):,} rows to {inc_parquet_path} ({inc_parquet_path.stat().st_size / (1024*1024):.1f} MB).")
    except Exception as ex:
        print(f"  -> Warning saving parquet: {ex}")

    # Verification summary
    print("\n" + "=" * 80)
    print("VERIFICATION: master_daily_rvm POSTGRESQL STATE")
    print("=" * 80)
    with engine.connect() as conn:
        tot = conn.execute(text("SELECT COUNT(*) FROM master_daily_rvm;")).scalar()
        min_d = conn.execute(text("SELECT MIN(giorno_dt) FROM master_daily_rvm;")).scalar()
        max_d = conn.execute(text("SELECT MAX(giorno_dt) FROM master_daily_rvm;")).scalar()
        recent_cnt = conn.execute(
            text("SELECT COUNT(*) FROM master_daily_rvm WHERE giorno_dt >= :s;"),
            {"s": start_date}
        ).scalar()
        print(f" * Total Table Rows:       {tot:,}")
        print(f" * Min Date:               {min_d}")
        print(f" * Max Date:               {max_d}")
        print(f" * New Records Added:      {recent_cnt:,} ({start_date} to {end_date})")

        print("\n Recent 10-day Volume in master_daily_rvm:")
        res = conn.execute(text("""
            SELECT giorno_dt, COUNT(*) as rows, ROUND(SUM(bottiglie)::numeric, 0) as total_bottles, ROUND(SUM(presenze)::numeric, 0) as total_users
            FROM master_daily_rvm
            WHERE giorno_dt >= :s
            GROUP BY giorno_dt
            ORDER BY giorno_dt DESC
            LIMIT 10;
        """), {"s": start_date}).fetchall()
        print(f" {'Date':<12} | {'Rows':<8} | {'Total Bottles':<16} | {'Total Users':<12}")
        print(" " + "-" * 55)
        for r in res:
            print(f" {str(r[0]):<12} | {r[1]:<8} | {int(r[2]):<16,} | {int(r[3]):<12,}")
    print("=" * 80)
    print("[SUCCESS] master_daily_rvm table updated successfully.\n")


def main():
    parser = argparse.ArgumentParser(description="Enrich and add new daily bottle records to master_daily_rvm.")
    parser.add_argument("--start-date", type=str, default="2026-09-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="2026-10-21", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    run_pipeline(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
