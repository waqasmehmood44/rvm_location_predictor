"""Script to merge raw JSON operational bottle logs into a single sorted CSV file."""

import json
from pathlib import Path
import pandas as pd


def merge_raw_json_files(
    raw_dir: str = "data/raw",
    output_csv: str = "data/raw/operational_data.csv",
):
    raw_path = Path(raw_dir)
    out_path = Path(output_csv)

    # Discover all Bottle_* files
    all_files = [f for f in raw_path.iterdir() if f.is_file() and f.name.startswith("Bottle_")]

    if not all_files:
        print(f"[ERROR] No Bottle_* files found in {raw_path}")
        return

    # Natural chronological sorting based on filename start date
    all_files = sorted(all_files, key=lambda p: p.name)

    print("=" * 60)
    print(f"Found {len(all_files)} raw JSON files to merge:")
    for f in all_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print("=" * 60)

    records = []
    for f in all_files:
        print(f"Loading {f.name}...")
        with open(f, "r", encoding="utf-8") as fp:
            data_json = json.load(fp)
            file_records = data_json.get("data", [])
            print(f"  -> Extracted {len(file_records):,} records")
            records.extend(file_records)

    print(f"\nTotal raw records collected across all files: {len(records):,}")
    print("Converting to DataFrame...")
    df = pd.DataFrame(records)

    print(f"DataFrame shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Parse and sort chronologically by date
    print("Parsing GIORNO datetime and sorting chronologically...")
    # Convert 'DD/MM/YYYY HH:MM:SS'
    df["_date_parsed"] = pd.to_datetime(df["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

    # If any failed, try generic dayfirst
    if df["_date_parsed"].isna().sum() > 0:
        mask = df["_date_parsed"].isna()
        df.loc[mask, "_date_parsed"] = pd.to_datetime(df.loc[mask, "GIORNO"], dayfirst=True, errors="coerce")

    # Sort strictly by date ascending, then PF
    df = df.sort_values(by=["_date_parsed", "PF"], ascending=[True, True])
    df = df.drop(columns=["_date_parsed"])

    print(f"Saving merged dataset to: {out_path}...")
    df.to_csv(out_path, index=False)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"SUCCESS: Successfully saved complete operational dataset: {out_path} ({file_size_mb:.1f} MB)")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY REPORT OF MERGED DATASET:")
    print("=" * 60)
    print(f"Total Records: {len(df):,}")
    print(f"Date Range (first 5 dates): {df['GIORNO'].head(5).tolist()}")
    print(f"Date Range (last 5 dates):  {df['GIORNO'].tail(5).tolist()}")
    print(f"Unique PFs (Stores): {df['PF'].nunique():,}")
    print(f"Unique Clients: {df['CLIENTE'].nunique():,}")
    print(f"Total Bottles Collected: {df['BOTTIGLIE'].sum():,}")
    print(f"Total Presenze Logged: {df['PRESENZE'].sum():,}")
    print("=" * 60)


if __name__ == "__main__":
    merge_raw_json_files()
