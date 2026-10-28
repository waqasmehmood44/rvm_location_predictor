"""Stage 4: Municipality data cleaning, joining, and macro feature derivation."""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np

from src.data.clean_data import clean_comune_name, clean_decimal_number


def build_municipality_feature_store(
    pil_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    surf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardizes municipality datasets and computes derived socio-economic indices."""
    print("\n[STAGE 4] Building municipality demographic and economic feature store...")

    # 1. Clean PIL
    pil = pil_df.copy()
    pil["norm_comune"] = pil["Comune"].apply(clean_comune_name)
    pil["PIL"] = pd.to_numeric(pil["PIL"], errors="coerce")
    pil = pil.dropna(subset=["norm_comune"]).drop_duplicates(subset=["norm_comune"])

    # 2. Clean Population
    pop = pop_df.copy()
    pop["norm_comune"] = pop["Comune"].apply(clean_comune_name)
    pop["Popolazione"] = pd.to_numeric(pop["Popolazione"], errors="coerce")
    pop = pop.dropna(subset=["norm_comune"]).drop_duplicates(subset=["norm_comune"])

    # 3. Clean Surface (handling comma decimal strings)
    surf = surf_df.copy()
    surf["norm_comune"] = surf["Comune"].apply(clean_comune_name)
    surf["Superficie"] = surf["Superficie"].apply(clean_decimal_number)
    surf = surf.dropna(subset=["norm_comune"]).drop_duplicates(subset=["norm_comune"])

    # Merge the three macro datasets
    macro = pd.merge(pop[["norm_comune", "Popolazione"]], surf[["norm_comune", "Superficie"]], on="norm_comune", how="outer")
    macro = pd.merge(macro, pil[["norm_comune", "PIL"]], on="norm_comune", how="outer")

    # Derive socio-economic ratios
    # Population density: residents per square kilometer
    macro["population_density"] = np.where(
        (macro["Superficie"] > 0) & (macro["Popolazione"].notna()),
        macro["Popolazione"] / macro["Superficie"],
        np.nan
    )

    # PIL per capita: aggregate income / population
    macro["PIL_per_capita"] = np.where(
        (macro["Popolazione"] > 0) & (macro["PIL"].notna()),
        macro["PIL"] / macro["Popolazione"],
        np.nan
    )

    print(f"  -> Built feature store for {len(macro):,} Italian municipalities")
    return macro


def enrich_with_municipality_data(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    comune_col: str = "Comune",
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """Joins municipality demographics onto physical points without row multiplication."""
    initial_rows = len(df)
    print("[STAGE 4] Enriching dataset with municipality macro features...")

    df_out = df.copy()
    df_out["norm_comune"] = df_out[comune_col].apply(clean_comune_name)

    enriched = pd.merge(
        df_out,
        macro_df[["norm_comune", "Popolazione", "Superficie", "PIL", "population_density", "PIL_per_capita"]],
        on="norm_comune",
        how="left"
    )

    # Determine municipality match status
    has_pop = enriched["Popolazione"].notna()
    has_pil = enriched["PIL"].notna()
    has_surf = enriched["Superficie"].notna()

    full_match = has_pop & has_pil & has_surf
    partial_match = (has_pop | has_pil | has_surf) & ~full_match

    enriched["municipality_match_status"] = np.select(
        [full_match, partial_match],
        ["MATCHED", "PARTIAL"],
        default="UNMATCHED"
    )

    # Rename to clean target column names
    enriched = enriched.rename(columns={
        "Popolazione": "population",
        "Superficie": "surface",
    })
    enriched = enriched.drop(columns=["norm_comune"])

    assert len(enriched) == initial_rows, f"ROW MULTIPLICATION! Initial: {initial_rows}, After: {len(enriched)}"

    stats = enriched["municipality_match_status"].value_counts().to_dict()
    print(f"  Municipality Match Status: {stats}")

    return enriched, stats
