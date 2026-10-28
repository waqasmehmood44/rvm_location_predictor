"""Modular data loader for Coripet datasets.

Handles reading raw datasets with exact date parsing and schema integrity.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


RAW_DIR = Path("data/raw")


def load_daily_bottles(path: str | Path = RAW_DIR / "daily_bottles.csv") -> pd.DataFrame:
    """Loads daily_bottles.csv and strictly parses GIORNO as DD/MM/YYYY HH:MM:SS."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    print(f"[LOAD] Loading daily_bottles from {path}...")
    df = pd.read_csv(path, low_memory=False)

    required_cols = [
        "CLIENTE",
        "GIORNO",
        "BOTTIGLIE",
        "PRESENZE",
        "PF",
        "CODICEMACCHINAPROD",
        "CODICEMACCHINA",
        "REFCORIPET",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"daily_bottles is missing columns: {missing}")

    # Explicit parsing of GIORNO: 01/01/2023 through 31/08/2026
    df["giorno_dt"] = pd.to_datetime(df["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if df["giorno_dt"].isna().sum() > 0:
        mask = df["giorno_dt"].isna()
        df.loc[mask, "giorno_dt"] = pd.to_datetime(df.loc[mask, "GIORNO"], dayfirst=True, errors="coerce")

    # Standardize string keys
    df["PF"] = df["PF"].astype(str).str.strip()
    df["CODICEMACCHINA"] = df["CODICEMACCHINA"].astype(str).str.strip()
    df["CODICEMACCHINAPROD"] = df["CODICEMACCHINAPROD"].astype(str).str.strip()

    print(f"  -> Loaded {len(df):,} records, {df['PF'].nunique():,} unique PFs, date span: {df['giorno_dt'].min()} to {df['giorno_dt'].max()}")
    return df


def load_storico_punto_fisico(path: str | Path = RAW_DIR / "CaricaViewStoricoPuntoFisico.csv") -> pd.DataFrame:
    """Loads CaricaViewStoricoPuntoFisico.csv preserving all historical version records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    print(f"[LOAD] Loading CaricaViewStoricoPuntoFisico from {path}...")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)

    if "PuntoFisico" not in df.columns:
        raise KeyError("PuntoFisico column not found in CaricaViewStoricoPuntoFisico")

    # Standardize key
    df["PuntoFisico"] = df["PuntoFisico"].astype(str).str.strip()

    # Parse effective interval dates
    df["start_dt"] = pd.to_datetime(df["DataInizioRange"], dayfirst=True, errors="coerce")
    df["end_dt"] = pd.to_datetime(df["DataFineRange"], dayfirst=True, errors="coerce")

    # Standardize machine columns if present
    if "Macchina" in df.columns:
        df["Macchina_clean"] = df["Macchina"].fillna("").astype(str).str.strip()
    if "CodiceMacchinaProd" in df.columns:
        df["CodiceMacchinaProd_clean"] = df["CodiceMacchinaProd"].fillna("").astype(str).str.strip()

    print(f"  -> Loaded {len(df):,} historical records across {df['PuntoFisico'].nunique():,} unique PFs")
    return df


def load_filenielsen(path: str | Path = RAW_DIR / "filenielsen_1203.csv") -> pd.DataFrame:
    """Loads filenielsen_1203.csv retail census."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")

    print(f"[LOAD] Loading filenielsen_1203 from {path}...")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    print(f"  -> Loaded {len(df):,} Nielsen stores across {df.shape[1]} columns")
    return df


def load_macro_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads PIL.csv, population.csv, and surface.csv."""
    print("[LOAD] Loading macroeconomic and demographic datasets (PIL, population, surface)...")
    pil = pd.read_csv(RAW_DIR / "PIL.csv", sep=";", encoding="utf-8-sig", low_memory=False)
    pop = pd.read_csv(RAW_DIR / "population.csv", sep=";", encoding="utf-8-sig", low_memory=False)
    surf = pd.read_csv(RAW_DIR / "surface.csv", sep=";", encoding="utf-8-sig", low_memory=False)

    print(f"  -> PIL: {len(pil):,} rows | Pop: {len(pop):,} rows | Surface: {len(surf):,} rows")
    return pil, pop, surf
