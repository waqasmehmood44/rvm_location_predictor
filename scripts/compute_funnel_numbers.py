"""
Calculate exact funnel numbers and exclusion breakdown for current_active_pf_report.md
and ramp_up_maturity_report.md
"""
import pandas as pd
import numpy as np

def compute_funnel():
    # Load full integrated daily rvm
    df = pd.read_parquet("data/processed/full_integrated_daily_rvm.parquet")
    print(f"Loaded full integrated dataset: {len(df):,} rows, {df['PF'].nunique():,} unique PFs")
    
    total_rows = len(df)
    total_pfs = df["PF"].nunique()
    
    # 1. TOTAL
    print(f"Step 1 - TOTAL: {total_pfs} PFs, {total_rows} rows (100.0%)")
    
    # 2. PUNTOVENDITA
    pv_df = df[df["puntovendita_eligible"] == 1]
    pv_pfs = pv_df["PF"].nunique()
    pv_rows = len(pv_df)
    print(f"Step 2 - PUNTOVENDITA: {pv_pfs} PFs, {pv_rows} rows ({pv_rows/total_rows*100:.2f}%)")
    
    # 3. ACTIVE AS OF 2026-08-31 (across all PFs)
    act_df = df[df["active_as_of_2026_08_31"] == 1]
    act_pfs = act_df["PF"].nunique()
    act_rows = len(act_df)
    print(f"Step 3 - ACTIVE AS OF 2026-08-31: {act_pfs} PFs, {act_rows} rows ({act_rows/total_rows*100:.2f}%)")
    
    # 4. ACTIVE + PUNTOVENDITA
    act_pv_df = df[(df["active_as_of_2026_08_31"] == 1) & (df["puntovendita_eligible"] == 1)]
    act_pv_pfs = act_pv_df["PF"].nunique()
    act_pv_rows = len(act_pv_df)
    print(f"Step 4 - ACTIVE + PUNTOVENDITA: {act_pv_pfs} PFs, {act_pv_rows} rows ({act_pv_rows/total_rows*100:.2f}%)")
    
    # 5. ACTIVE + PUNTOVENDITA + MATURE
    act_pv_mat_df = df[(df["active_as_of_2026_08_31"] == 1) & (df["puntovendita_eligible"] == 1) & (df["maturity_status"] == "MATURE")]
    act_pv_mat_pfs = act_pv_mat_df["PF"].nunique()
    act_pv_mat_rows = len(act_pv_mat_df)
    print(f"Step 5 - ACTIVE + PUNTOVENDITA + MATURE: {act_pv_mat_pfs} PFs, {act_pv_mat_rows} rows ({act_pv_mat_rows/total_rows*100:.2f}%)")
    
    # 6. FINAL MODELING PFs
    final_df = df[df["modeling_eligible"] == 1]
    final_pfs = final_df["PF"].nunique()
    final_rows = len(final_df)
    print(f"Step 6 - FINAL MODELING: {final_pfs} PFs, {final_rows} rows ({final_rows/total_rows*100:.2f}%)")
    
    # Assert Step 5 == Step 6
    assert act_pv_mat_rows == final_rows
    assert act_pv_mat_pfs == final_pfs
    
    # Exclusions Breakdown:
    # 1. Non-PUNTOVENDITA
    non_pv_df = df[df["puntovendita_eligible"] == 0]
    non_pv_pfs = non_pv_df["PF"].nunique()
    non_pv_rows = len(non_pv_df)
    
    # 2. Currently inactive as of 2026-08-31
    # Among PUNTOVENDITA:
    inact_pv_df = pv_df[pv_df["active_as_of_2026_08_31"] == 0]
    inact_pv_pfs = inact_pv_df["PF"].nunique()
    inact_pv_rows = len(inact_pv_df)
    
    # Across ALL PFs:
    inact_all_df = df[df["active_as_of_2026_08_31"] == 0]
    inact_all_pfs = inact_all_df["PF"].nunique()
    inact_all_rows = len(inact_all_df)
    
    # 3. No valid configuration on 2026-08-31
    no_valid_pfs = 0
    no_valid_rows = 0
    
    # 4. Ambiguous current status
    ambig_pfs = 0
    ambig_rows = 0
    
    # 5. Immature observations (within active PUNTOVENDITA)
    immature_df = act_pv_df[act_pv_df["maturity_status"] == "IMMATURE"]
    immature_rows = len(immature_df)
    immature_pfs = immature_df["PF"].nunique()
    
    # 6. PFs completely excluded because they never reached maturity
    # PFs in act_pv_pfs that have 0 MATURE observations
    pfs_with_mature = set(act_pv_mat_df["PF"].unique())
    pfs_never_mature = set(act_pv_df["PF"].unique()) - pfs_with_mature
    never_mature_pfs_count = len(pfs_never_mature)
    never_mature_rows = len(act_pv_df[act_pv_df["PF"].isin(pfs_never_mature)])
    
    print("\n--- EXCLUSION BREAKDOWN ---")
    print(f"1. Non-PUNTOVENDITA: {non_pv_pfs} PFs, {non_pv_rows:,} rows")
    print(f"2. Currently inactive as of 2026-08-31 (among PV): {inact_pv_pfs} PFs, {inact_pv_rows:,} rows")
    print(f"   (Currently inactive across all PFs: {inact_all_pfs} PFs, {inact_all_rows:,} rows)")
    print(f"3. No valid configuration on 2026-08-31: {no_valid_pfs} PFs, {no_valid_rows} rows")
    print(f"4. Ambiguous current status: {ambig_pfs} PFs, {ambig_rows} rows")
    print(f"5. Immature observations (among active PV): {immature_pfs} PFs, {immature_rows:,} rows")
    print(f"6. PFs completely excluded because they never reached maturity: {never_mature_pfs_count} PFs, {never_mature_rows:,} rows")
    print(f"7. Final eligible PFs: {final_pfs}")
    print(f"8. Final eligible daily observations: {final_rows:,}")
    
    print("\nList of 40 PFs that never reached maturity:")
    print(sorted(list(pfs_never_mature)))
    
    # Check their activation dates
    sub_never = act_pv_df[act_pv_df["PF"].isin(pfs_never_mature)]
    print("\nActivation dates of never mature PFs:")
    print(sub_never.groupby("PF")["DataPrimaAttivazione"].first().value_counts())

if __name__ == "__main__":
    compute_funnel()
