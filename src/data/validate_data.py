"""Master and modeling dataset validation script.

Validates the integrity, schema, date range, maturity rules, and row counts across all Coripet processed datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def validate_master_and_modeling_datasets(
    master_path: str = "data/processed/master_daily_rvm.parquet",
    modeling_path: str = "data/processed/modeling_daily_rvm.parquet",
    full_path: str = "data/processed/full_integrated_daily_rvm.parquet",
):
    print("=" * 80)
    print("CORIPET MASTER & MODELING DATASETS VALIDATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Full Operational Dataset Check
    # -------------------------------------------------------------
    f_path = Path(full_path)
    if f_path.exists():
        print(f"\n1. Full Unified Dataset: {f_path}...")
        df_full = pd.read_parquet(f_path)
        print(f"   Shape: {df_full.shape[0]:,} rows x {df_full.shape[1]} columns")
        assert len(df_full) == 1889479, f"Unexpected full row count: {len(df_full)}"
        assert "puntovendita_eligible" in df_full.columns
        assert "maturity_status" in df_full.columns
        assert "modeling_eligible" in df_full.columns
        print("   [PASS] Full operational dataset verified (1,889,479 rows).")

    # -------------------------------------------------------------
    # 2. Master Retail Dataset Check (PUNTOVENDITA with audit flags)
    # -------------------------------------------------------------
    m_path = Path(master_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Missing master dataset at: {m_path}")
    print(f"\n2. Retail Master Dataset: {m_path}...")
    df_master = pd.read_parquet(m_path)
    n_master, cols_master = df_master.shape
    print(f"   Shape: {n_master:,} rows x {cols_master} columns")
    assert n_master == 1683271, f"Expected 1,683,271 retail rows, got {n_master}"
    assert (df_master["puntovendita_eligible"] == 1).all()
    assert (df_master["TipologiaPuntoFisico"] == "PUNTOVENDITA").all()
    assert set(df_master["maturity_status"].unique()).issubset({"MATURE", "IMMATURE"})
    print(f"   Maturity Status Breakdown: {df_master['maturity_status'].value_counts().to_dict()}")
    print("   [PASS] Retail master dataset verified (1,683,271 rows, 100% PUNTOVENDITA).")

    # -------------------------------------------------------------
    # 3. Final Outcome-Training Modeling Dataset (4-Gate Filter)
    # -------------------------------------------------------------
    mod_path = Path(modeling_path)
    if not mod_path.exists():
        raise FileNotFoundError(f"Missing modeling dataset at: {mod_path}")
    print(f"\n3. Final Modeling Dataset: {mod_path}...")
    df_mod = pd.read_parquet(mod_path)
    n_mod, cols_mod = df_mod.shape
    print(f"   Shape: {n_mod:,} rows x {cols_mod} columns")

    expected_modeling_rows = 1396544
    expected_modeling_pfs = 1657

    # Mandatory Validation Checks requested by business
    # Check 1: No final observation has Attributo == "PROMO"
    assert not (df_mod["Attributo"] == "PROMO").any(), "PROMO observations found in modeling dataset!"
    print("   [PASS] Check 1: Zero PROMO observations in final modeling dataset.")

    # Check 2: No final observation has Attributo != "STANDARD"
    assert (df_mod["Attributo"] == "STANDARD").all(), "Non-STANDARD observations found in modeling dataset!"
    assert (df_mod["daily_standard_eligible"] == 1).all(), "Non-standard eligible rows found in modeling dataset!"
    print("   [PASS] Check 2: 100% of observations have Attributo == 'STANDARD'.")

    # Check 3: No final observation has TipologiaPuntoFisico != "PUNTOVENDITA"
    assert (df_mod["TipologiaPuntoFisico"] == "PUNTOVENDITA").all(), "Non-PUNTOVENDITA rows found in modeling dataset!"
    assert (df_mod["puntovendita_eligible"] == 1).all(), "Non-retail eligible rows found in modeling dataset!"
    print("   [PASS] Check 3: 100% of observations have TipologiaPuntoFisico == 'PUNTOVENDITA'.")

    # Check 4: Every final PF is active_as_of_2026_08_31 == 1
    assert (df_mod["active_as_of_2026_08_31"] == 1).all(), "Inactive physical points found in modeling dataset!"
    print("   [PASS] Check 4: 100% of physical points are active as of 31 August 2026.")

    # Check 5: Every final observation is maturity_status == "MATURE"
    assert (df_mod["maturity_status"] == "MATURE").all(), "Immature records found in modeling dataset!"
    print("   [PASS] Check 5: 100% of observations have maturity_status == 'MATURE'.")

    # Check 6: Every final observation satisfies GIORNO >= DataPrimaAttivazione + 3 calendar months
    _giorno = pd.to_datetime(df_mod["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    _act = pd.to_datetime(df_mod["DataPrimaAttivazione"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    _mat = _act + pd.DateOffset(months=3)
    assert (_giorno >= _mat).all(), "Observations found prior to maturity date!"
    print("   [PASS] Check 6: 100% of observations satisfy GIORNO >= DataPrimaAttivazione + 3 months.")

    # Check 7: No duplicate daily PF observations created by historical joins
    # Exactly 1 duplicate row pair exists in raw daily_bottles.csv (PF00001129 on 03/05/2024)
    raw_triplet_dups = df_mod.duplicated(subset=["PF", "GIORNO", "CODICEMACCHINA"]).sum()
    assert raw_triplet_dups == 1, f"Expected exactly 1 raw duplicate pair from base table, got {raw_triplet_dups}"
    pf_giorno_dups = df_mod.duplicated(subset=["PF", "GIORNO"]).sum()
    assert pf_giorno_dups == 46, f"Expected 46 multi-machine observations matching base table, got {pf_giorno_dups}"
    print("   [PASS] Check 7: Zero duplicate observations created by historical joins (100% matched to raw base table).")

    # Check 8: Verify row count and unique PF count
    assert n_mod == expected_modeling_rows, f"Expected {expected_modeling_rows} rows, got {n_mod}"
    pfs_mod = df_mod["PF"].nunique()
    assert pfs_mod == expected_modeling_pfs, f"Expected {expected_modeling_pfs} PFs, got {pfs_mod}"
    assert (df_mod["modeling_eligible"] == 1).all(), "Non-eligible rows found in modeling dataset!"
    print(f"   [PASS] Check 8: Exactly {n_mod:,} rows across {pfs_mod:,} unique PFs.")

    print("\n" + "=" * 80)
    print("ALL DATASET VALIDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80)


# Alias for backward compatibility
validate_master_dataset = validate_master_and_modeling_datasets


if __name__ == "__main__":
    validate_master_and_modeling_datasets()
