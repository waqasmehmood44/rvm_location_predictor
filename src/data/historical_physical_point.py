"""Stage 1 and Stage 2: Historical physical point integration and lifecycle analysis.

Associates each daily_bottles observation with the correct historical version
of CaricaViewStoricoPuntoFisico using effective date intervals without row duplication.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np


def enrich_daily_bottles_with_history(
    daily_df: pd.DataFrame,
    sp_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Merges daily_bottles with CaricaViewStoricoPuntoFisico based on effective date intervals.

    Guarantees:
    - Zero row multiplication: output row count strictly equals daily_df row count.
    - Resolves version ambiguity with CODICEMACCHINA == Macchina where applicable.
    - Sets historical_match_status: MATCHED, AMBIGUOUS, UNMATCHED.
    """
    initial_rows = len(daily_df)
    initial_pfs = daily_df["PF"].nunique()
    initial_pf_giorno = len(daily_df[["PF", "GIORNO"]].drop_duplicates())

    print("\n[STAGE 1] Merging daily_bottles with physical point history...")
    print(f"  Before Join -> Rows: {initial_rows:,}, Unique PFs: {initial_pfs:,}, Unique (PF, GIORNO): {initial_pf_giorno:,}")

    # Prepare historical records
    sp = sp_df.copy()
    
    # Store master attributes to attach
    sp_attr_cols = [
        "Indirizzo", "Cap", "Comune", "Provincia", "Regione", "Nazione",
        "Latitudine", "Longitudine", "Stato", "TipologiaPuntoFisico", "Attributo",
        "Cooperativa", "Filiera", "Descrizione", "TipoContenitore",
        "ReferenteCommerciale", "ReferenteCoripet", "Comodatario", "NomeFiliera",
        "Raggruppamento", "Macchina", "CodiceMacchinaProd", "TipoMacchina",
        "Fornitore", "DataInstallazione", "DataPrimaAttivazione"
    ]
    # Filter only available columns
    sp_attr_cols = [c for c in sp_attr_cols if c in sp.columns]

    # Pre-split SP into single-record PFs and multi-record PFs for performance
    pf_record_counts = sp["PuntoFisico"].value_counts()
    single_pfs = set(pf_record_counts[pf_record_counts == 1].index)
    multi_pfs = set(pf_record_counts[pf_record_counts > 1].index)

    # 1. Handle Single-record PFs (Fast direct join)
    sp_single = sp[sp["PuntoFisico"].isin(single_pfs)].copy()
    daily_single = daily_df[daily_df["PF"].isin(single_pfs)].copy()

    merged_single = pd.merge(
        daily_single,
        sp_single[["PuntoFisico", "start_dt", "end_dt"] + sp_attr_cols],
        left_on="PF",
        right_on="PuntoFisico",
        how="left"
    )

    # Check interval validity for single PFs
    in_range_single = (
        (merged_single["giorno_dt"] >= merged_single["start_dt"]) &
        (merged_single["giorno_dt"] <= merged_single["end_dt"] + pd.Timedelta(days=1))
    )
    merged_single["historical_match_status"] = np.where(in_range_single, "MATCHED", "UNMATCHED")

    # 2. Handle Multi-record PFs (Interval join with ambiguity resolution)
    daily_multi = daily_df[daily_df["PF"].isin(multi_pfs)].copy()
    sp_multi = sp[sp["PuntoFisico"].isin(multi_pfs)].copy()

    # Index daily_multi with an explicit original row identifier to prevent duplication
    daily_multi["_orig_idx"] = daily_multi.index

    # Candidate merge on PF
    merged_multi_cand = pd.merge(
        daily_multi[["_orig_idx", "PF", "GIORNO", "giorno_dt", "CODICEMACCHINA", "CODICEMACCHINAPROD"]],
        sp_multi[["PuntoFisico", "start_dt", "end_dt", "Macchina_clean", "CodiceMacchinaProd_clean"] + sp_attr_cols],
        left_on="PF",
        right_on="PuntoFisico",
        how="inner"
    )

    # Filter where giorno is within [start_dt, end_dt + 1 day]
    valid_cand = merged_multi_cand[
        (merged_multi_cand["giorno_dt"] >= merged_multi_cand["start_dt"]) &
        (merged_multi_cand["giorno_dt"] <= merged_multi_cand["end_dt"] + pd.Timedelta(days=1))
    ].copy()

    # Ambiguity check: Count how many candidate intervals matched each _orig_idx
    cand_counts = valid_cand.groupby("_orig_idx").size()
    single_match_indices = set(cand_counts[cand_counts == 1].index)
    multi_match_indices = set(cand_counts[cand_counts > 1].index)

    # A: Exactly 1 interval match
    cand_single = valid_cand[valid_cand["_orig_idx"].isin(single_match_indices)].copy()
    cand_single["historical_match_status"] = "MATCHED"

    # B: Multiple interval matches (boundary collision or dual machines)
    cand_multi = valid_cand[valid_cand["_orig_idx"].isin(multi_match_indices)].copy()
    
    # Try resolving via CODICEMACCHINA == Macchina
    mach_matches = cand_multi[cand_multi["CODICEMACCHINA"] == cand_multi["Macchina_clean"]]
    mach_counts = mach_matches.groupby("_orig_idx").size()
    
    resolved_by_mach = mach_matches[mach_matches["_orig_idx"].isin(mach_counts[mach_counts == 1].index)].copy()
    resolved_by_mach["historical_match_status"] = "MATCHED"

    # Any remaining unresolved: pick the latest start_dt, mark as AMBIGUOUS
    still_multi_indices = set(multi_match_indices) - set(resolved_by_mach["_orig_idx"])
    if still_multi_indices:
        remaining_multi = cand_multi[cand_multi["_orig_idx"].isin(still_multi_indices)].sort_values(
            by=["_orig_idx", "start_dt"], ascending=[True, False]
        ).drop_duplicates(subset=["_orig_idx"], keep="first").copy()
        remaining_multi["historical_match_status"] = "AMBIGUOUS"
    else:
        remaining_multi = pd.DataFrame()

    # Combine matched multi records
    resolved_multi = pd.concat([cand_single, resolved_by_mach, remaining_multi], ignore_index=True)

    # Join back to full daily_multi to account for any unmatched rows
    final_multi = pd.merge(
        daily_multi,
        resolved_multi.drop(columns=["PF", "GIORNO", "giorno_dt", "CODICEMACCHINA", "CODICEMACCHINAPROD", "start_dt", "end_dt", "Macchina_clean", "CodiceMacchinaProd_clean"]),
        on="_orig_idx",
        how="left"
    )
    final_multi["historical_match_status"] = final_multi["historical_match_status"].fillna("UNMATCHED")
    final_multi = final_multi.drop(columns=["_orig_idx"])

    # Drop temporary columns from merged_single
    merged_single = merged_single.drop(columns=["start_dt", "end_dt"])

    # Combine single and multi partitions
    master_enriched = pd.concat([merged_single, final_multi], ignore_index=True)

    # Sort to restore original chronological ordering
    master_enriched = master_enriched.sort_values(by=["giorno_dt", "PF"], ascending=[True, True]).reset_index(drop=True)

    final_rows = len(master_enriched)
    final_pfs = master_enriched["PF"].nunique()
    final_pf_giorno = len(master_enriched[["PF", "GIORNO"]].drop_duplicates())

    status_counts = master_enriched["historical_match_status"].value_counts().to_dict()

    print(f"  After Join  -> Rows: {final_rows:,}, Unique PFs: {final_pfs:,}, Unique (PF, GIORNO): {final_pf_giorno:,}")
    print(f"  Match Status Breakdown: {status_counts}")
    
    assert final_rows == initial_rows, f"ROW MULTIPLICATION DETECTED! Initial: {initial_rows}, Final: {final_rows}"

    stats = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "initial_pfs": initial_pfs,
        "final_pfs": final_pfs,
        "initial_pf_giorno": initial_pf_giorno,
        "final_pf_giorno": final_pf_giorno,
        "match_status_breakdown": status_counts,
    }

    return master_enriched, stats


def analyze_physical_point_history(sp_df: pd.DataFrame, master_df: pd.DataFrame) -> Dict[str, any]:
    """Computes detailed metrics on physical points for Stage 2 physical_point_history.md report."""
    print("\n[STAGE 2] Analyzing physical point historical lifecycles...")
    sp = sp_df.copy()

    # 1. Total unique physical points
    total_sp_pfs = sp["PuntoFisico"].nunique()
    total_active_pfs = master_df["PF"].nunique()

    # 2. Daily observations per physical point
    obs_per_pf = master_df.groupby("PF").size()
    obs_stats = {
        "min": int(obs_per_pf.min()),
        "max": int(obs_per_pf.max()),
        "mean": float(round(obs_per_pf.mean(), 1)),
        "median": float(round(obs_per_pf.median(), 1)),
    }

    # 3. PFs changing machine
    pfs_changing_machine = sp.groupby("PuntoFisico")["Macchina"].nunique()
    changed_machine_pfs = int((pfs_changing_machine > 1).sum())

    # 4. PFs changing machine type
    pfs_changing_type = sp.groupby("PuntoFisico")["TipoMacchina"].nunique()
    changed_type_pfs = int((pfs_changing_type > 1).sum())

    # 5. PFs changing address
    pfs_changing_addr = sp.groupby("PuntoFisico")["Indirizzo"].nunique()
    changed_addr_pfs = int((pfs_changing_addr > 1).sum())

    # 6. PFs changing store/client
    pfs_changing_client = sp.groupby("PuntoFisico")["Filiera"].nunique()
    changed_client_pfs = int((pfs_changing_client > 1).sum())

    # 7. PFs changing status
    pfs_changing_status = sp.groupby("PuntoFisico")["Stato"].nunique()
    changed_status_pfs = int((pfs_changing_status > 1).sum())

    # 8. PFs with multiple machine configurations (historical versions)
    pfs_multi_config = int((sp["PuntoFisico"].value_counts() > 1).sum())

    # 9. Gaps and Overlaps in historical configuration
    sp["_start"] = pd.to_datetime(sp["DataInizioRange"], dayfirst=True, errors="coerce")
    sp["_end"] = pd.to_datetime(sp["DataFineRange"], dayfirst=True, errors="coerce")

    overlapping_pfs = 0
    gap_pfs = 0
    multi_pfs = sp["PuntoFisico"].value_counts()[sp["PuntoFisico"].value_counts() > 1].index

    for pf in multi_pfs:
        sub = sp[sp["PuntoFisico"] == pf].sort_values("_start")
        prev_end = None
        has_overlap = False
        has_gap = False
        for _, r in sub.iterrows():
            st_dt = r["_start"]
            en_dt = r["_end"]
            if pd.isna(st_dt):
                continue
            if prev_end is not None and not pd.isna(prev_end):
                if st_dt < prev_end:
                    has_overlap = True
                elif (st_dt - prev_end).days > 1:
                    has_gap = True
            prev_end = en_dt
        if has_overlap:
            overlapping_pfs += 1
        if has_gap:
            gap_pfs += 1

    results = {
        "total_pfs_in_store_master": total_sp_pfs,
        "total_active_pfs_in_operations": total_active_pfs,
        "observations_per_pf": obs_stats,
        "pfs_changed_machine": changed_machine_pfs,
        "pfs_changed_machine_type": changed_type_pfs,
        "pfs_changed_address": changed_addr_pfs,
        "pfs_changed_client": changed_client_pfs,
        "pfs_changed_status": changed_status_pfs,
        "pfs_with_multiple_configurations": pfs_multi_config,
        "pfs_with_overlapping_periods": overlapping_pfs,
        "pfs_with_gaps": gap_pfs,
    }

    print(f"  Physical Point Lifecycle Summary:\n{results}")
    return results
