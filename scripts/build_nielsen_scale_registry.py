"""
Builds fast indexed store scale lookup registry and banner profiles from Nielsen 25,104 store census.
Outputs:
- data/processed/nielsen_store_scale_registry.parquet
- data/processed/nielsen_banner_scale_profiles.json
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import re
import unicodedata
import pandas as pd
import numpy as np

from src.model.preprocessing import clean_brand_name, clean_retail_text

def extract_street_core(addr: str) -> str:
    """Extracts core street name, stripping numbers, prefixes, and noise."""
    if not addr or pd.isna(addr):
        return ""
    norm = clean_retail_text(addr)
    prefixes = ["VIA", "CORSO", "PIAZZA", "VIALE", "VICOLO", "STRADA", "LOC", "LOCALITA", "LARGO", "CONTRADA", "FRAZIONE", "SS", "SP"]
    tokens = norm.split()
    filtered = [tok for tok in tokens if tok not in prefixes and not tok.isdigit() and len(tok) > 1]
    return " ".join(filtered) if filtered else norm

def build_registry():
    nielsen_path = "data/raw/filenielsen_1203.csv"
    if not os.path.exists(nielsen_path):
        print(f"Error: {nielsen_path} does not exist!")
        return

    print("Loading Nielsen universe (25,104 stores)...")
    df = pd.read_csv(nielsen_path, sep=";", encoding="latin1", low_memory=False)
    print(f"Loaded {len(df):,} raw records.")

    # 1. Clean columns
    df["cap_str"] = df["CAP"].fillna(0).astype(int).astype(str).str.zfill(5)
    df["comune_clean"] = df["COMUNE"].astype(str).str.strip().str.upper()
    df["provincia_clean"] = df["PROVINCIA"].astype(str).str.strip().str.upper()
    df["insegna_raw"] = df["INSEGNA"].fillna(df["RAGIONE SOCIALE"]).astype(str)
    df["clean_brand"] = df["insegna_raw"].apply(clean_brand_name)
    df["street_core"] = df["INDIRIZZO"].apply(extract_street_core)
    
    df["tipo_negozio"] = df["TIPO NEGOZIO"].astype(str).str.strip().str.upper()
    df["superficie_vendita"] = pd.to_numeric(df["METRI QUADRI SUPERFICIE VENDITA"], errors="coerce")
    df["numero_casse"] = pd.to_numeric(df["NUMERO CASSE TOTALI"], errors="coerce")

    # Filter valid rows
    valid_stores = df[
        (df["superficie_vendita"].notna()) & 
        (df["superficie_vendita"] > 50) & 
        (df["street_core"].str.len() >= 3)
    ].copy()
    
    print(f"Valid stores with clean address & surface: {len(valid_stores):,}")

    # 2. Build Address Lookup Parquet
    cols_to_save = [
        "cap_str", "street_core", "comune_clean", "provincia_clean", 
        "tipo_negozio", "superficie_vendita", "numero_casse", "clean_brand", "insegna_raw"
    ]
    # Drop duplicates by (cap_str, street_core), prioritizing larger store if multiple
    sorted_stores = valid_stores.sort_values(by=["superficie_vendita"], ascending=False)
    dedup_stores = sorted_stores.drop_duplicates(subset=["cap_str", "street_core"])
    
    os.makedirs("data/processed", exist_ok=True)
    out_parquet = "data/processed/nielsen_store_scale_registry.parquet"
    dedup_stores[cols_to_save].to_parquet(out_parquet, index=False)
    print(f"Saved {len(dedup_stores):,} unique address scale records to {out_parquet}")

    # 3. Build Banner Footprint Profiles
    print("\nComputing empirical banner footprint profiles...")
    banner_profiles = {}
    
    # Format-level medians as baseline
    format_medians = df.groupby("tipo_negozio")[["superficie_vendita", "numero_casse"]].median().to_dict(orient="index")
    
    # Specific banner profiles
    banner_groups = df[df["clean_brand"] != "INDIPENDENTE"].groupby("clean_brand")
    for brand, grp in banner_groups:
        if len(grp) >= 5:  # Sufficient sample size
            med_sqm = float(grp["superficie_vendita"].median())
            med_casse = float(grp["numero_casse"].median())
            top_tipo = grp["tipo_negozio"].mode()[0] if len(grp["tipo_negozio"].mode()) > 0 else "SUP"
            
            banner_profiles[brand] = {
                "tipo_negozio": top_tipo,
                "superficie_vendita": round(med_sqm, 0),
                "numero_casse": round(med_casse, 0),
                "stores_sampled": len(grp)
            }
            print(f"  {brand:<15}: {top_tipo} | {med_sqm:.0f} m2 | {med_casse:.0f} casse ({len(grp)} stores)")

    out_json = "data/processed/nielsen_banner_scale_profiles.json"
    with open(out_json, "w") as f:
        json.dump({
            "banner_profiles": banner_profiles,
            "format_defaults": {
                "IPR": {"superficie_vendita": 3750.0, "numero_casse": 21.0, "label": "Hypermarket / Mall Anchor"},
                "SUP": {"superficie_vendita": 750.0, "numero_casse": 4.0, "label": "Standard Supermarket"},
                "DIS": {"superficie_vendita": 740.0, "numero_casse": 4.0, "label": "Discount Store"},
                "LIS": {"superficie_vendita": 200.0, "numero_casse": 2.0, "label": "Small Neighborhood Market"}
            },
            "national_median": {
                "tipo_negozio": "SUP",
                "superficie_vendita": 750.0,
                "numero_casse": 4.0,
                "label": "National Median Supermarket"
            }
        }, f, indent=2)
    print(f"\nSaved banner profiles to {out_json}")

if __name__ == "__main__":
    build_registry()
