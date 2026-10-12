"""Comprehensive audit and relationship investigation across the 6 Coripet raw datasets."""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR = Path("data/raw")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(name: str):
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    
    df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    return df


def audit_single_dataset(name: str, df: pd.DataFrame):
    print(f"\n--- AUDITING {name} ---")
    n_rows, n_cols = df.shape
    columns = list(df.columns)
    dtypes = {col: str(df[col].dtype) for col in columns}
    missing_counts = {col: int(df[col].isna().sum()) for col in columns}
    missing_pcts = {col: round(float(df[col].isna().mean()) * 100, 2) for col in columns}
    
    full_dupes = int(df.duplicated().sum())
    
    candidate_keys = []
    for col in columns:
        if df[col].nunique(dropna=True) == n_rows:
            candidate_keys.append(col)
            
    date_cols = [c for c in columns if any(kw in c.lower() for kw in ["data", "giorno", "date", "anno", "year"])]
    date_ranges = {}
    for c in date_cols:
        parsed = pd.to_datetime(df[c], dayfirst=True, errors="coerce")
        valid = parsed.dropna()
        if not valid.empty:
            date_ranges[c] = {
                "min": str(valid.min()),
                "max": str(valid.max()),
                "valid_count": int(len(valid)),
                "valid_pct": round(float(len(valid) / n_rows) * 100, 2),
            }
            
    cat_summary = {}
    num_summary = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            s = df[col].dropna()
            if not s.empty:
                num_summary[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(round(s.mean(), 2)),
                    "median": float(round(s.median(), 2)),
                    "std": float(round(s.std(), 2)),
                    "p25": float(round(s.quantile(0.25), 2)),
                    "p75": float(round(s.quantile(0.75), 2)),
                }
            else:
                num_summary[col] = {"all_na": True}
        else:
            nunique = int(df[col].nunique(dropna=True))
            top_vals = df[col].value_counts(dropna=False).head(5).to_dict()
            cat_summary[col] = {
                "nunique": nunique,
                "top_5": {str(k): int(v) for k, v in top_vals.items()}
            }
            
    return {
        "name": name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "missing_pcts": missing_pcts,
        "full_duplicate_rows": full_dupes,
        "candidate_unique_keys": candidate_keys,
        "date_ranges": date_ranges,
        "categorical_variables": cat_summary,
        "numerical_variables": num_summary,
    }


def analyze_relationships(dfs):
    print("\n--- ANALYZING RELATIONSHIPS ---")
    rel_results = {}
    
    db = dfs["daily_bottles"]
    sp = dfs["CaricaViewStoricoPuntoFisico"]
    fn = dfs["filenielsen_1203"]
    pil = dfs["PIL"]
    pop = dfs["population"]
    surf = dfs["surface"]
    
    # 1. Keys connecting daily_bottles to CaricaViewStoricoPuntoFisico
    db_pfs = set(db["PF"].dropna().astype(str).str.strip().unique())
    sp_pfs = set(sp["PuntoFisico"].dropna().astype(str).str.strip().unique())
    fn_pfs = set(fn["PuntoFisico"].dropna().astype(str).str.strip().unique()) if "PuntoFisico" in fn.columns else set()
    
    rel_results["pf_overlap"] = {
        "db_unique_pfs": len(db_pfs),
        "sp_unique_pfs": len(sp_pfs),
        "fn_unique_pfs": len(fn_pfs),
        "db_in_sp": len(db_pfs.intersection(sp_pfs)),
        "db_in_sp_pct": round(len(db_pfs.intersection(sp_pfs)) / len(db_pfs) * 100, 2),
        "db_not_in_sp": len(db_pfs - sp_pfs),
        "sp_in_fn": len(sp_pfs.intersection(fn_pfs)),
        "sp_in_fn_pct": round(len(sp_pfs.intersection(fn_pfs)) / len(sp_pfs) * 100, 2) if sp_pfs else 0,
        "db_in_fn": len(db_pfs.intersection(fn_pfs)),
        "db_in_fn_pct": round(len(db_pfs.intersection(fn_pfs)) / len(db_pfs) * 100, 2),
    }
    
    # Machine keys
    db_machines = set(db["CODICEMACCHINA"].dropna().astype(str).str.strip().unique())
    sp_machines = set(sp["Macchina"].dropna().astype(str).str.strip().unique()) if "Macchina" in sp.columns else set()
    rel_results["machine_overlap"] = {
        "db_unique_machines": len(db_machines),
        "sp_unique_machines": len(sp_machines),
        "db_machines_in_sp": len(db_machines.intersection(sp_machines)),
        "db_machines_in_sp_pct": round(len(db_machines.intersection(sp_machines)) / len(db_machines) * 100, 2),
    }
    
    # MachineProd keys
    db_macprod = set(db["CODICEMACCHINAPROD"].dropna().astype(str).str.strip().unique())
    sp_macprod = set(sp["CodiceMacchinaProd"].dropna().astype(str).str.strip().unique()) if "CodiceMacchinaProd" in sp.columns else set()
    rel_results["macprod_overlap"] = {
        "db_unique_macprod": len(db_macprod),
        "sp_unique_macprod": len(sp_macprod),
        "db_macprod_in_sp": len(db_macprod.intersection(sp_macprod)),
        "db_macprod_in_sp_pct": round(len(db_macprod.intersection(sp_macprod)) / len(db_macprod) * 100, 2),
    }

    # Historical structure of CaricaViewStoricoPuntoFisico
    # Records per PuntoFisico
    pf_counts = sp["PuntoFisico"].value_counts()
    rel_results["sp_pf_distribution"] = {
        "total_records": len(sp),
        "unique_pfs": len(pf_counts),
        "pfs_with_single_record": int((pf_counts == 1).sum()),
        "pfs_with_multiple_records": int((pf_counts > 1).sum()),
        "max_records_for_single_pf": int(pf_counts.max()),
    }
    
    # Analyze multiple machines per PuntoFisico and machines moving between PFs
    pf_to_machines = sp.groupby("PuntoFisico")["Macchina"].nunique()
    mac_to_pfs = sp.groupby("Macchina")["PuntoFisico"].nunique() if "Macchina" in sp.columns else pd.Series()
    
    rel_results["cardinality"] = {
        "pfs_with_multiple_distinct_machines": int((pf_to_machines > 1).sum()),
        "machines_associated_with_multiple_pfs": int((mac_to_pfs > 1).sum()) if not mac_to_pfs.empty else 0,
        "max_pfs_per_machine": int(mac_to_pfs.max()) if not mac_to_pfs.empty else 0,
    }
    
    # Analyze effective date intervals in CaricaViewStoricoPuntoFisico
    sp["_start"] = pd.to_datetime(sp["DataInizioRange"], dayfirst=True, errors="coerce")
    sp["_end"] = pd.to_datetime(sp["DataFineRange"], dayfirst=True, errors="coerce")
    
    # Check open-ended end dates (NaT or far future)
    null_ends = int(sp["_end"].isna().sum())
    rel_results["date_intervals"] = {
        "null_start_dates": int(sp["_start"].isna().sum()),
        "null_end_dates": null_ends,
        "start_min": str(sp["_start"].min()),
        "start_max": str(sp["_start"].max()),
        "end_min": str(sp["_end"].min()),
        "end_max": str(sp["_end"].max()),
    }
    
    # Check overlaps and gaps among multi-record PFs
    overlapping_pfs = 0
    gap_pfs = 0
    multi_pfs = pf_counts[pf_counts > 1].index
    
    for pf in multi_pfs:
        sub = sp[sp["PuntoFisico"] == pf].sort_values("_start")
        has_overlap = False
        has_gap = False
        prev_end = None
        for _, row in sub.iterrows():
            st_dt = row["_start"]
            en_dt = row["_end"]
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
            
    rel_results["interval_consistency"] = {
        "total_multi_record_pfs": len(multi_pfs),
        "pfs_with_overlapping_intervals": overlapping_pfs,
        "pfs_with_gap_intervals": gap_pfs,
    }
    
    # Time-varying vs Stable fields in CaricaViewStoricoPuntoFisico
    stable_fields = []
    varying_fields = []
    check_cols = [c for c in sp.columns if c not in ["DataInizioRange", "DataFineRange", "_start", "_end"]]
    for col in check_cols:
        grouped_unique = sp.groupby("PuntoFisico")[col].nunique()
        if (grouped_unique > 1).sum() == 0:
            stable_fields.append(col)
        else:
            varying_fields.append({
                "column": col,
                "pfs_varying": int((grouped_unique > 1).sum()),
                "pct_varying": round(float((grouped_unique > 1).mean()) * 100, 2)
            })
            
    rel_results["field_stability"] = {
        "strictly_stable_across_all_pfs": stable_fields,
        "time_varying_fields": varying_fields,
    }
    
    # Macro datasets (PIL, population, surface) linkage investigation
    rel_results["macro_datasets"] = {
        "PIL_columns": list(pil.columns),
        "PIL_shape": list(pil.shape),
        "population_columns": list(pop.columns),
        "population_shape": list(pop.shape),
        "surface_columns": list(surf.columns),
        "surface_shape": list(surf.shape),
    }
    
    return rel_results


def main():
    dataset_names = [
        "daily_bottles",
        "CaricaViewStoricoPuntoFisico",
        "filenielsen_1203",
        "PIL",
        "population",
        "surface",
    ]
    
    audit_results = {}
    dfs = {}
    for name in dataset_names:
        df = load_dataset(name)
        dfs[name] = df
        audit_results[name] = audit_single_dataset(name, df)
        
    rel_results = analyze_relationships(dfs)
    
    output = {
        "datasets": audit_results,
        "relationships": rel_results,
    }
    
    with open(REPORTS_DIR / "full_audit_and_relationships.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        
    print("\n[SUCCESS] Completed comprehensive audit and saved reports/full_audit_and_relationships.json")


if __name__ == "__main__":
    main()
