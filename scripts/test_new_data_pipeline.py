import os
import sys
import dotenv
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

dotenv.load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

e = create_engine(os.getenv("DATABASE_URL"))

from src.data.load_data import load_filenielsen, load_macro_datasets
from src.data.historical_physical_point import enrich_daily_bottles_with_history
from src.data.nielsen_matching import match_physical_points_to_nielsen
from src.data.municipality_matching import build_municipality_feature_store, enrich_with_municipality_data

with e.connect() as conn:
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

    daily_new = pd.read_sql("SELECT * FROM raw_daily_conferimenti WHERE giorno_dt >= '2026-09-01' AND giorno_dt <= '2026-10-21'", conn)
    daily_new.rename(columns={
        "pf": "PF", "giorno": "GIORNO", "bottiglie": "BOTTIGLIE", "presenze": "PRESENZE",
        "cliente": "CLIENTE", "codicemacchina": "CODICEMACCHINA",
        "codicemacchinaprod": "CODICEMACCHINAPROD", "refcoripet": "REFCORIPET"
    }, inplace=True)
    daily_new["giorno_dt"] = pd.to_datetime(daily_new["giorno_dt"], errors="coerce")

    # Read schema of existing master_daily_rvm
    df_existing = pd.read_sql("SELECT * FROM master_daily_rvm LIMIT 1", conn)
    master_cols = [c for c in df_existing.columns if c != "id"]

print("1. Merging with Physical Point History...")
stage1_df, _ = enrich_daily_bottles_with_history(daily_new, sp_df)

print("2. Matching Nielsen Store Census...")
fn_df = load_filenielsen()
nielsen_lookup, _ = match_physical_points_to_nielsen(sp_df, fn_df)
stage3_df = pd.merge(stage1_df, nielsen_lookup, on="PuntoFisico", how="left")

print("3. Attaching Municipality Demographics & Economics...")
pil_df, pop_df, surf_df = load_macro_datasets()
macro_feature_store = build_municipality_feature_store(pil_df, pop_df, surf_df)
stage4_df, _ = enrich_with_municipality_data(stage3_df, macro_feature_store, comune_col="Comune")

print("4. Calculating Business Eligibility Gates & Ramp-up Maturity...")
stage4_df["puntovendita_eligible"] = (stage4_df["TipologiaPuntoFisico"] == "PUNTOVENDITA").astype(int)

# Reference date active network
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

# Filter for master_daily_rvm (Retail PUNTOVENDITA)
master_new = stage4_df[stage4_df["puntovendita_eligible"] == 1].copy()
# Rename CodiceMacchinaProd to match master_daily_rvm schema if needed
if "CodiceMacchinaProd" in master_new.columns and "codice_macchina_prod" not in master_new.columns:
    master_new["codice_macchina_prod"] = master_new["CodiceMacchinaProd"]

master_new.columns = [c.lower() for c in master_new.columns]
print(f"Master new rows: {len(master_new):,}")

# Check column alignment
missing_in_new = [c for c in master_cols if c not in master_new.columns]
extra_in_new = [c for c in master_new.columns if c not in master_cols]
print(f"Missing in new ({len(missing_in_new)}): {missing_in_new}")
print(f"Extra in new ({len(extra_in_new)}): {extra_in_new}")
