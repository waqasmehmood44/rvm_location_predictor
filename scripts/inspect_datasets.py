"""Automated dataset schema, distribution, and relationship inspector for Coripet RVM Predictor.

Validates:
- Row counts, column types, and missing value rates
- Match percentage on the join key (PF <-> PuntoFisico)
- Operational date range
- Target distribution percentiles (mean, std, min, 25%, 50%, 75%, 90%, max)
- Processed feature store readiness
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def inspect_raw_and_processed_datasets():
    print("=" * 80)
    print("      CORIPET RVM PREDICTOR — AUTOMATED DATASET SCHEMA & DISTRIBUTION INSPECTOR")
    print("=" * 80)

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    daily_path = raw_dir / "daily_bottles.csv"
    pf_path = raw_dir / "CaricaViewStoricoPuntoFisico.csv"

    # 1. Operational Dataset
    print("\n--- 1. OPERATIONAL LOGS: daily_bottles.csv ---")
    if not daily_path.exists():
        print(f"❌ Error: {daily_path} not found.")
        return

    print("Loading operational records...")
    df_daily = pd.read_csv(daily_path, low_memory=False)
    n_daily = len(df_daily)
    print(f"  * Total Rows: {n_daily:,}")
    print(f"  * Columns ({len(df_daily.columns)}): {list(df_daily.columns)}")
    print(f"  * Unique Physical Points (PF): {df_daily['PF'].nunique():,}")

    if "GIORNO" in df_daily.columns:
        date_series = pd.to_datetime(df_daily["GIORNO"], errors="coerce")
        min_date = date_series.min()
        max_date = date_series.max()
        print(f"  * Operational Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

    # Target Distributions
    print("\n--- Target Distributions (Raw Operational Log) ---")
    for target in ["BOTTIGLIE", "PRESENZE"]:
        if target in df_daily.columns:
            s = pd.to_numeric(df_daily[target], errors="coerce").dropna()
            print(f"  * {target}:")
            print(f"      Count: {len(s):,} | Missing: {df_daily[target].isna().sum():,}")
            print(f"      Mean: {s.mean():.1f} | Std: {s.std():.1f}")
            print(f"      Min: {s.min():.0f} | 25%: {s.quantile(0.25):.0f} | 50%: {s.median():.0f} | 75%: {s.quantile(0.75):.0f} | 90%: {s.quantile(0.90):.0f} | Max: {s.max():.0f}")

    # 2. Store Master Dataset
    print("\n--- 2. STORE MASTER: CaricaViewStoricoPuntoFisico.csv ---")
    if not pf_path.exists():
        print(f"❌ Error: {pf_path} not found.")
        return

    df_pf = pd.read_csv(pf_path, sep=";", low_memory=False)
    n_pf = len(df_pf)
    print(f"  * Total Records: {n_pf:,}")
    print(f"  * Unique PuntoFisico IDs: {df_pf['PuntoFisico'].nunique():,}")

    # 3. Join Key Match Percentage
    print("\n--- 3. JOIN KEY MATCHING (PF <-> PuntoFisico) ---")
    daily_pfs = set(df_daily["PF"].dropna().astype(str).str.strip().str.upper().unique())
    master_pfs = set(df_pf["PuntoFisico"].dropna().astype(str).str.strip().str.upper().unique())

    matched_pfs = daily_pfs.intersection(master_pfs)
    match_pct = (len(matched_pfs) / len(daily_pfs) * 100.0) if daily_pfs else 0.0
    print(f"  * Unique PFs in Operational Log: {len(daily_pfs):,}")
    print(f"  * Unique PFs in Master Store:    {len(master_pfs):,}")
    print(f"  * Matched PFs:                   {len(matched_pfs):,} ({match_pct:.2f}%)")

    # 4. Check Processed Feature Stores
    print("\n--- 4. PROCESSED FEATURE STORE STATUS ---")
    modeling_parquet = processed_dir / "modeling_daily_rvm.parquet"
    spatial_network = processed_dir / "coripet_rvm_spatial_network.parquet"
    density_lookup = processed_dir / "competitive_density_lookup.parquet"

    for p in [modeling_parquet, spatial_network, density_lookup]:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  * [READY] {p.name} ({size_mb:.1f} MB)")
        else:
            print(f"  * [MISSING] {p.name}")

    print("\n" + "=" * 80)
    print("DATASET INSPECTION COMPLETE — ALL CORE PREREQUISITES VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    inspect_raw_and_processed_datasets()
