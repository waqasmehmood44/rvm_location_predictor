"""Stage 3: Controlled hierarchical matching between Coripet Physical Points and Nielsen retail stores."""

from pathlib import Path
from typing import Dict, Tuple
import re
import pandas as pd
import numpy as np

from src.data.clean_data import clean_postal_code, clean_comune_name, clean_address_string


KNOWN_CHAINS = [
    "ESSELUNGA", "CONAD", "COOP", "IPERCOOP", "CARREFOUR", "PAM", "PANORAMA",
    "LIDL", "EUROSPIN", "BENNET", "IPER", "IPERAL", "TIGROS", "DESPAR",
    "INTERSPAR", "EUROSPAR", "CRAI", "SELEX", "FAMILA", "IL GIGANTE",
    "DECATHLON", "LEROY MERLIN", "BRICO", "MD", "ALDI", "PENNY", "DECO", "DODECA",
    "EMISFERO", "MIGROSS", "TIGRE", "MAXIDI"
]


def extract_brand_token(text: str) -> str:
    """Extracts standardized brand banner token from store text."""
    if not text or pd.isna(text):
        return ""
    clean = str(text).upper()
    for chain in KNOWN_CHAINS:
        if re.search(rf"\b{re.escape(chain)}\b", clean):
            return chain
    return ""


def match_physical_points_to_nielsen(
    sp_df: pd.DataFrame,
    nielsen_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Performs controlled hierarchical matching of unique physical points against Nielsen census.

    Guarantees:
    - Never duplicates physical point records.
    - Explicitly sets nielsen_match_status (MATCHED, AMBIGUOUS, UNMATCHED).
    - Explicitly sets nielsen_match_method and nielsen_match_confidence (HIGH, MEDIUM, LOW).
    - Preserves all physical point attributes untouched.
    """
    print("\n[STAGE 3] Performing controlled hierarchical Nielsen matching...")
    sp_unique = sp_df.drop_duplicates(subset=["PuntoFisico"]).copy()
    all_pfs = list(sp_unique["PuntoFisico"].unique())
    n_sp = len(all_pfs)

    # 1. Prepare and standardize Coripet attributes
    sp_unique["norm_cap"] = sp_unique["Cap"].apply(clean_postal_code)
    sp_unique["norm_comune"] = sp_unique["Comune"].apply(clean_comune_name)
    sp_unique["norm_addr"] = sp_unique["Indirizzo"].apply(clean_address_string)
    sp_unique["brand_token"] = (
        sp_unique["Filiera"].astype(str) + " " +
        sp_unique["Descrizione"].astype(str) + " " +
        sp_unique["NomeFiliera"].astype(str)
    ).apply(extract_brand_token)

    # 2. Prepare and standardize Nielsen attributes
    fn = nielsen_df.copy()
    fn["norm_cap"] = fn["CAP"].apply(clean_postal_code)
    fn["norm_comune"] = fn["COMUNE"].apply(clean_comune_name)
    fn["norm_addr"] = fn["INDIRIZZO"].apply(clean_address_string)
    fn["brand_token"] = (
        fn["INSEGNA"].astype(str) + " " +
        fn["GRUPPO"].astype(str) + " " +
        fn["RAGIONE SOCIALE"].astype(str)
    ).apply(extract_brand_token)

    match_records: Dict[str, dict] = {}
    method_stats = {}

    def record_match(pf: str, fn_row: pd.Series, method: str, confidence: str, is_ambiguous: bool = False):
        match_records[pf] = {
            "PuntoFisico": pf,
            "nielsen_match_status": "AMBIGUOUS" if is_ambiguous else "MATCHED",
            "nielsen_match_method": method,
            "nielsen_match_confidence": confidence,
            "nielsen_country_shop_id": float(fn_row["COUNTRY SHOP ID"]) if pd.notna(fn_row.get("COUNTRY SHOP ID")) else np.nan,
            "nielsen_tipo_negozio": str(fn_row["TIPO NEGOZIO"]) if pd.notna(fn_row.get("TIPO NEGOZIO")) else None,
            "nielsen_superficie_vendita": float(fn_row["METRI QUADRI SUPERFICIE VENDITA"]) if pd.notna(fn_row.get("METRI QUADRI SUPERFICIE VENDITA")) else np.nan,
            "nielsen_numero_casse": float(fn_row["NUMERO CASSE TOTALI"]) if pd.notna(fn_row.get("NUMERO CASSE TOTALI")) else np.nan,
            "nielsen_insegna": str(fn_row["INSEGNA"]) if pd.notna(fn_row.get("INSEGNA")) else None,
            "nielsen_gruppo": str(fn_row["GRUPPO"]) if pd.notna(fn_row.get("GRUPPO")) else None,
            "nielsen_area": float(fn_row["AREA NIELSEN"]) if pd.notna(fn_row.get("AREA NIELSEN")) else np.nan,
        }

    # Strategy 1: Exact Normalized Address + CAP + Comune
    print("  Testing Strategy 1: EXACT_ADDR_CAP_COMUNE...")
    unmatched_pfs = [p for p in all_pfs if p not in match_records]
    sp_rem = sp_unique[sp_unique["PuntoFisico"].isin(unmatched_pfs)]

    merged_s1 = pd.merge(
        sp_rem[sp_rem["norm_addr"] != ""][["PuntoFisico", "norm_addr", "norm_cap", "norm_comune", "brand_token"]],
        fn[["COUNTRY SHOP ID", "TIPO NEGOZIO", "METRI QUADRI SUPERFICIE VENDITA", "NUMERO CASSE TOTALI", "INSEGNA", "GRUPPO", "AREA NIELSEN", "norm_addr", "norm_cap", "norm_comune", "brand_token"]],
        on=["norm_addr", "norm_cap", "norm_comune"],
        how="inner",
        suffixes=("_sp", "_fn")
    )
    s1_matched = 0
    s1_ambiguous = 0

    for pf, grp in merged_s1.groupby("PuntoFisico"):
        if len(grp) == 1:
            record_match(pf, grp.iloc[0], "EXACT_ADDR_CAP_COMUNE", "HIGH")
            s1_matched += 1
        else:
            sp_b = grp.iloc[0]["brand_token_sp"]
            brand_filtered = grp[grp["brand_token_fn"] == sp_b] if sp_b else grp
            if len(brand_filtered) == 1:
                record_match(pf, brand_filtered.iloc[0], "EXACT_ADDR_CAP_COMUNE_BRAND", "HIGH")
                s1_matched += 1
            else:
                record_match(pf, grp.iloc[0], "EXACT_ADDR_CAP_COMUNE", "MEDIUM", is_ambiguous=True)
                s1_ambiguous += 1

    method_stats["EXACT_ADDR_CAP_COMUNE"] = {"matched": s1_matched, "ambiguous": s1_ambiguous}

    # Strategy 2: Exact Address + CAP
    print("  Testing Strategy 2: ADDR_CAP...")
    unmatched_pfs = [p for p in all_pfs if p not in match_records]
    sp_rem = sp_unique[sp_unique["PuntoFisico"].isin(unmatched_pfs)]

    merged_s2 = pd.merge(
        sp_rem[sp_rem["norm_addr"] != ""][["PuntoFisico", "norm_addr", "norm_cap", "brand_token"]],
        fn[["COUNTRY SHOP ID", "TIPO NEGOZIO", "METRI QUADRI SUPERFICIE VENDITA", "NUMERO CASSE TOTALI", "INSEGNA", "GRUPPO", "AREA NIELSEN", "norm_addr", "norm_cap", "brand_token"]],
        on=["norm_addr", "norm_cap"],
        how="inner",
        suffixes=("_sp", "_fn")
    )
    s2_matched = 0
    s2_ambiguous = 0
    for pf, grp in merged_s2.groupby("PuntoFisico"):
        if len(grp) == 1:
            record_match(pf, grp.iloc[0], "ADDR_CAP", "HIGH")
            s2_matched += 1
        else:
            sp_b = grp.iloc[0]["brand_token_sp"]
            brand_filtered = grp[grp["brand_token_fn"] == sp_b] if sp_b else grp
            if len(brand_filtered) == 1:
                record_match(pf, brand_filtered.iloc[0], "ADDR_CAP_BRAND", "HIGH")
                s2_matched += 1
            else:
                record_match(pf, grp.iloc[0], "ADDR_CAP", "MEDIUM", is_ambiguous=True)
                s2_ambiguous += 1
    method_stats["ADDR_CAP"] = {"matched": s2_matched, "ambiguous": s2_ambiguous}

    # Strategy 3: Brand + Exact Address + Comune
    print("  Testing Strategy 3: BRAND_ADDR_COMUNE...")
    unmatched_pfs = [p for p in all_pfs if p not in match_records]
    sp_rem = sp_unique[sp_unique["PuntoFisico"].isin(unmatched_pfs)]
    sp_rem_b = sp_rem[(sp_rem["brand_token"] != "") & (sp_rem["norm_addr"] != "")]

    merged_s3 = pd.merge(
        sp_rem_b[["PuntoFisico", "brand_token", "norm_addr", "norm_comune"]],
        fn[["COUNTRY SHOP ID", "TIPO NEGOZIO", "METRI QUADRI SUPERFICIE VENDITA", "NUMERO CASSE TOTALI", "INSEGNA", "GRUPPO", "AREA NIELSEN", "brand_token", "norm_addr", "norm_comune"]],
        on=["brand_token", "norm_addr", "norm_comune"],
        how="inner"
    )
    s3_matched = 0
    s3_ambiguous = 0
    for pf, grp in merged_s3.groupby("PuntoFisico"):
        if len(grp) == 1:
            record_match(pf, grp.iloc[0], "BRAND_ADDR_COMUNE", "MEDIUM")
            s3_matched += 1
        else:
            record_match(pf, grp.iloc[0], "BRAND_ADDR_COMUNE", "LOW", is_ambiguous=True)
            s3_ambiguous += 1
    method_stats["BRAND_ADDR_COMUNE"] = {"matched": s3_matched, "ambiguous": s3_ambiguous}

    # Strategy 4: Unique Brand in CAP + Comune
    print("  Testing Strategy 4: UNIQUE_BRAND_CAP_COMUNE...")
    unmatched_pfs = [p for p in all_pfs if p not in match_records]
    sp_rem = sp_unique[sp_unique["PuntoFisico"].isin(unmatched_pfs)]
    sp_rem_b = sp_rem[sp_rem["brand_token"] != ""]

    merged_s4 = pd.merge(
        sp_rem_b[["PuntoFisico", "brand_token", "norm_cap", "norm_comune"]],
        fn[["COUNTRY SHOP ID", "TIPO NEGOZIO", "METRI QUADRI SUPERFICIE VENDITA", "NUMERO CASSE TOTALI", "INSEGNA", "GRUPPO", "AREA NIELSEN", "brand_token", "norm_cap", "norm_comune"]],
        on=["brand_token", "norm_cap", "norm_comune"],
        how="inner"
    )
    s4_matched = 0
    s4_ambiguous = 0
    for pf, grp in merged_s4.groupby("PuntoFisico"):
        if len(grp) == 1:
            record_match(pf, grp.iloc[0], "UNIQUE_BRAND_CAP_COMUNE", "MEDIUM")
            s4_matched += 1
        else:
            record_match(pf, grp.iloc[0], "BRAND_CAP_COMUNE", "LOW", is_ambiguous=True)
            s4_ambiguous += 1
    method_stats["UNIQUE_BRAND_CAP_COMUNE"] = {"matched": s4_matched, "ambiguous": s4_ambiguous}

    # Fill all remaining PFs as UNMATCHED
    for pf in all_pfs:
        if pf not in match_records:
            match_records[pf] = {
                "PuntoFisico": pf,
                "nielsen_match_status": "UNMATCHED",
                "nielsen_match_method": "NONE",
                "nielsen_match_confidence": "NONE",
                "nielsen_country_shop_id": np.nan,
                "nielsen_tipo_negozio": None,
                "nielsen_superficie_vendita": np.nan,
                "nielsen_numero_casse": np.nan,
                "nielsen_insegna": None,
                "nielsen_gruppo": None,
                "nielsen_area": np.nan,
            }

    nielsen_lookup = pd.DataFrame(list(match_records.values()))

    status_breakdown = nielsen_lookup["nielsen_match_status"].value_counts().to_dict()
    confidence_breakdown = nielsen_lookup["nielsen_match_confidence"].value_counts().to_dict()
    total_matched = status_breakdown.get("MATCHED", 0)
    total_ambiguous = status_breakdown.get("AMBIGUOUS", 0)
    total_unmatched = status_breakdown.get("UNMATCHED", 0)

    summary = {
        "total_physical_points": n_sp,
        "matched": total_matched,
        "matched_pct": round(total_matched / n_sp * 100, 2),
        "ambiguous": total_ambiguous,
        "ambiguous_pct": round(total_ambiguous / n_sp * 100, 2),
        "unmatched": total_unmatched,
        "unmatched_pct": round(total_unmatched / n_sp * 100, 2),
        "status_breakdown": status_breakdown,
        "confidence_breakdown": confidence_breakdown,
        "method_stats": method_stats,
    }

    print(f"  Nielsen Matching Results Summary:\n{summary}")
    return nielsen_lookup, summary
