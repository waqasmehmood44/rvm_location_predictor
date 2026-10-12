"""
Diagnostic and reconciliation script for Physical Point active status as of 2026-08-31.
Evaluates CaricaViewStoricoPuntoFisico against daily_bottles.
"""
import pandas as pd
import numpy as np

def run_reconciliation():
    ref_date = pd.Timestamp("2026-08-31")
    print(f"=== Reference Date: {ref_date.strftime('%Y-%m-%d')} ===")
    
    # Load CaricaViewStoricoPuntoFisico
    storico_path = "data/raw/CaricaViewStoricoPuntoFisico.csv"
    storico_df = pd.read_csv(storico_path, sep=";")
    print(f"Total rows in Storico: {len(storico_df):,}")
    print(f"Unique PuntoFisico in Storico: {storico_df['PuntoFisico'].nunique():,}")
    
    # Parse dates
    storico_df["start_dt"] = pd.to_datetime(storico_df["DataInizioRange"], format="%d/%m/%Y", errors="coerce")
    storico_df["end_dt"] = pd.to_datetime(storico_df["DataFineRange"], format="%d/%m/%Y", errors="coerce")
    storico_df["att_dt"] = pd.to_datetime(storico_df["DataPrimaAttivazione"], format="%d/%m/%Y", errors="coerce")
    storico_df["inst_dt"] = pd.to_datetime(storico_df["DataInstallazione"], format="%d/%m/%Y", errors="coerce")
    
    # Check distinct values of Stato in Storico
    print("\n--- Distinct Stato values in full Storico table ---")
    print(storico_df["Stato"].value_counts(dropna=False))
    
    print("\n--- Distinct TipologiaPuntoFisico values in full Storico table ---")
    print(storico_df["TipologiaPuntoFisico"].value_counts(dropna=False))
    
    # Filter valid on 2026-08-31
    # DataInizioRange <= 2026-08-31 AND (DataFineRange is null OR DataFineRange >= 2026-08-31)
    is_valid_on_ref = (storico_df["start_dt"] <= ref_date) & (storico_df["end_dt"].isna() | (storico_df["end_dt"] >= ref_date))
    valid_configs = storico_df[is_valid_on_ref].copy()
    print(f"\nTotal valid configuration records on {ref_date.strftime('%Y-%m-%d')}: {len(valid_configs):,}")
    print(f"Unique PuntoFisico with valid config on {ref_date.strftime('%Y-%m-%d')}: {valid_configs['PuntoFisico'].nunique():,}")
    
    # Check duplicates on ref date
    config_counts = valid_configs.groupby("PuntoFisico").size()
    multi_configs = config_counts[config_counts > 1]
    print(f"PFs with multiple valid configurations on {ref_date.strftime('%Y-%m-%d')}: {len(multi_configs)}")
    if len(multi_configs) > 0:
        print("Details of multiple configs:")
        for pf, count in multi_configs.items():
            sub = valid_configs[valid_configs["PuntoFisico"] == pf]
            print(f"  PF: {pf} (count={count}): Stato={sub['Stato'].tolist()}, Tipo={sub['TipologiaPuntoFisico'].tolist()}")
            
    # Check Stato values among valid configurations on 2026-08-31
    print(f"\n--- Distinct Stato values valid on {ref_date.strftime('%Y-%m-%d')} ---")
    print(valid_configs["Stato"].value_counts(dropna=False))
    
    print(f"\n--- Cross-tabulation: Stato vs TipologiaPuntoFisico on {ref_date.strftime('%Y-%m-%d')} ---")
    print(pd.crosstab(valid_configs["Stato"], valid_configs["TipologiaPuntoFisico"], margins=True))
    
    # Load daily_bottles
    daily_path = "data/raw/daily_bottles.csv"
    daily_df = pd.read_csv(daily_path)
    daily_df["giorno_dt"] = pd.to_datetime(daily_df["GIORNO"], format="%d/%m/%Y", errors="coerce")
    daily_pfs = set(daily_df["PF"].unique())
    print(f"\nTotal daily observations: {len(daily_df):,}")
    print(f"Total unique PFs in daily_bottles: {len(daily_pfs):,}")
    print(f"Date range in daily_bottles: {daily_df['giorno_dt'].min()} to {daily_df['giorno_dt'].max()}")
    
    # Check how daily_pfs relate to valid_configs
    valid_pfs = set(valid_configs["PuntoFisico"].unique())
    pfs_in_both = daily_pfs.intersection(valid_pfs)
    pfs_daily_no_valid = daily_pfs - valid_pfs
    pfs_valid_no_daily = valid_pfs - daily_pfs
    print(f"\nPFs in daily_bottles that have a valid config on {ref_date.strftime('%Y-%m-%d')}: {len(pfs_in_both)}")
    print(f"PFs in daily_bottles with NO valid config on {ref_date.strftime('%Y-%m-%d')}: {len(pfs_daily_no_valid)}")
    print(f"PFs with valid config on {ref_date.strftime('%Y-%m-%d')} but NO records in daily_bottles: {len(pfs_valid_no_daily)}")
    
    if len(pfs_daily_no_valid) > 0:
        print(f"Sample of daily PFs with no valid config on ref date: {list(pfs_daily_no_valid)[:10]}")
        # Why don't they have a valid config?
        sub_hist = storico_df[storico_df["PuntoFisico"].isin(pfs_daily_no_valid)]
        print(f"Their max end_dt: {sub_hist['end_dt'].max()}")
        print(f"Their distinct Stato values: {sub_hist['Stato'].value_counts().to_dict()}")

    # Active status on 2026-08-31
    # If a PF has Stato == 'ATTVO' on 2026-08-31
    active_configs = valid_configs[valid_configs["Stato"] == "ATTVO"]
    active_pfs = set(active_configs["PuntoFisico"].unique())
    print(f"\n=== Active PFs (Stato == 'ATTVO') on 2026-08-31: {len(active_pfs)} ===")
    print("Breakdown by TipologiaPuntoFisico among active PFs on 2026-08-31:")
    print(active_configs["TipologiaPuntoFisico"].value_counts())
    
    # Non-ATTVO on 2026-08-31
    inactive_configs = valid_configs[valid_configs["Stato"] != "ATTVO"]
    print(f"\nInactive/non-ATTVO valid configurations on 2026-08-31: {len(inactive_configs)}")
    print(inactive_configs["Stato"].value_counts())
    
    # Compare with 1,831!
    print(f"\nComparison: Active PFs ({len(active_pfs)}) vs Business Reference (1,831)")
    print(f"Difference: {len(active_pfs) - 1831:+d}")
    
    # Check date of export / latest dates in Storico
    print(f"\nLatest DataInizioRange in Storico: {storico_df['start_dt'].max()}")
    print(f"Latest DataFineRange in Storico (finite): {storico_df['end_dt'].max()}")
    
    # Check if there are transitions in early September 2026 in Storico
    sept_trans = storico_df[(storico_df["start_dt"] > ref_date) | ((storico_df["end_dt"] > ref_date) & (storico_df["end_dt"] <= pd.Timestamp("2026-09-30")))]
    print(f"Storico rows with transitions in September 2026: {len(sept_trans)}")
    
    # Count active as of 11 September 2026 (the export date)
    ref_sept11 = pd.Timestamp("2026-09-11")
    valid_sept11 = storico_df[(storico_df["start_dt"] <= ref_sept11) & (storico_df["end_dt"].isna() | (storico_df["end_dt"] >= ref_sept11))]
    active_sept11 = valid_sept11[valid_sept11["Stato"] == "ATTVO"]
    print(f"Active PFs as of 2026-09-11 (Snapshot date): {active_sept11['PuntoFisico'].nunique()}")
    print(active_sept11["TipologiaPuntoFisico"].value_counts())
    
    # Let's check the exact 4 PF difference
    pfs_attvo_aug31 = set(active_configs["PuntoFisico"].unique())
    pfs_attvo_sept11 = set(active_sept11["PuntoFisico"].unique())
    
    activated_in_sept = pfs_attvo_sept11 - pfs_attvo_aug31
    deactivated_in_sept = pfs_attvo_aug31 - pfs_attvo_sept11
    print(f"\nActivated between Aug 31 and Sept 11 (+{len(activated_in_sept)}): {sorted(list(activated_in_sept))}")
    for pf in sorted(list(activated_in_sept)):
        rows = storico_df[storico_df["PuntoFisico"] == pf][["DataInizioRange", "DataFineRange", "Stato", "TipologiaPuntoFisico"]]
        print(f"  PF {pf}:\n{rows.to_string(index=False)}")
        
    print(f"\nDeactivated between Aug 31 and Sept 11 (-{len(deactivated_in_sept)}): {sorted(list(deactivated_in_sept))}")
    for pf in sorted(list(deactivated_in_sept)):
        rows = storico_df[storico_df["PuntoFisico"] == pf][["DataInizioRange", "DataFineRange", "Stato", "TipologiaPuntoFisico"]]
        print(f"  PF {pf}:\n{rows.to_string(index=False)}")

    # Also check SOSP on August 31
    sosp_aug31 = valid_configs[valid_configs["Stato"] == "SOSP"]["PuntoFisico"].unique()
    print(f"\nPFs in SOSP on 2026-08-31: {len(sosp_aug31)} -> {sosp_aug31}")

if __name__ == "__main__":
    run_reconciliation()
