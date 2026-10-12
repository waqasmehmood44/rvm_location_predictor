"""Comprehensive audit and relationship analysis script for the 6 Coripet raw datasets."""

import os
from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path("data/raw")
FILES = {
    "daily_bottles": RAW_DIR / "daily_bottles.csv",
    "CaricaViewStoricoPuntoFisico": RAW_DIR / "CaricaViewStoricoPuntoFisico.csv",
    "filenielsen_1203": RAW_DIR / "filenielsen_1203.csv",
    "PIL": RAW_DIR / "PIL.csv",
    "population": RAW_DIR / "population.csv",
    "surface": RAW_DIR / "surface.csv",
}

def inspect_file_preview(name, path, nrows=5):
    print(f"\n{'='*80}\nPREVIEW: {name} ({path.name})\n{'='*80}")
    if not path.exists():
        print(f"FILE NOT FOUND: {path}")
        return
    # Check separator (could be comma or semicolon)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        second_line = f.readline()
    
    sep = ";" if first_line.count(";") > first_line.count(",") else ","
    print(f"Detected separator: '{sep}' | First line length: {len(first_line)}")
    print(f"Header snippet: {first_line[:200].strip()}...")
    
    try:
        df = pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False, encoding="utf-8", errors="replace")
    except Exception as e:
        df = pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False, encoding="latin1")
        
    print(f"Shape preview: {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(df.head(2).to_dict(orient="records"))

if __name__ == "__main__":
    for name, path in FILES.items():
        inspect_file_preview(name, path)
