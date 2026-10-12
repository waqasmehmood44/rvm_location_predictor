"""
Exports the untouched 20% Unseen-PF Test Holdout Set to CSV:
1. Full daily dataset (290,704 observations) -> data/processed/test_holdout.csv & models/preprocessing/test_holdout.csv
2. Store-level testing summary (331 unique physical locations) -> data/processed/test_holdout_stores_summary.csv & models/preprocessing/test_holdout_stores_summary.csv
"""
import os
import pandas as pd
import numpy as np

def export_test_holdout():
    parquet_path = "models/preprocessing/test_holdout.parquet"
    print(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows across {df['PF'].nunique():,} unique PFs.")

    # Paths
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/preprocessing", exist_ok=True)
    
    full_csv_processed = "data/processed/test_holdout.csv"
    full_csv_models = "models/preprocessing/test_holdout.csv"
    
    print(f"Exporting full daily test holdout to {full_csv_processed}...")
    df.to_csv(full_csv_processed, index=False)
    
    # Also copy to models/preprocessing/
    print(f"Exporting full daily test holdout to {full_csv_models}...")
    df.to_csv(full_csv_models, index=False)
    
    # Generate store-level testing summary for convenient testing of new locations
    print("Generating store-level testing summary for easy model evaluation...")
    
    # Helper to get first non-null
    def first_val(s):
        v = s.dropna()
        return v.iloc[0] if not v.empty else "N/A"
        
    store_summary = df.groupby("PF").agg(
        store_name=("CLIENTE", first_val),
        insegna=("nielsen_insegna", first_val),
        address=("Indirizzo", first_val),
        cap=("Cap", first_val),
        comune=("Comune", first_val),
        provincia=("Provincia", first_val),
        regione=("Regione", first_val),
        machine_type=("TipoMacchina", first_val),
        supplier=("Fornitore", first_val),
        latitudine=("Latitudine", first_val),
        longitudine=("Longitudine", first_val),
        actual_mean_bottiglie_day=("BOTTIGLIE", "mean"),
        actual_median_bottiglie_day=("BOTTIGLIE", "median"),
        actual_mean_presenze_day=("PRESENZE", "mean"),
        actual_median_presenze_day=("PRESENZE", "median"),
        total_days_observed=("GIORNO", "count"),
        min_date=("GIORNO", "min"),
        max_date=("GIORNO", "max")
    ).reset_index()
    
    # Clean numeric rounding
    store_summary["actual_mean_bottiglie_day"] = store_summary["actual_mean_bottiglie_day"].round(1)
    store_summary["actual_median_bottiglie_day"] = store_summary["actual_median_bottiglie_day"].round(1)
    store_summary["actual_mean_presenze_day"] = store_summary["actual_mean_presenze_day"].round(1)
    store_summary["actual_median_presenze_day"] = store_summary["actual_median_presenze_day"].round(1)
    
    # Sort by mean bottles descending
    store_summary = store_summary.sort_values(by="actual_mean_bottiglie_day", ascending=False)
    
    stores_csv_processed = "data/processed/test_holdout_stores_summary.csv"
    stores_csv_models = "models/preprocessing/test_holdout_stores_summary.csv"
    
    store_summary.to_csv(stores_csv_processed, index=False)
    store_summary.to_csv(stores_csv_models, index=False)
    
    print(f"Saved store summary to {stores_csv_processed} and {stores_csv_models}")
    print(f"Summary contains {len(store_summary)} stores.")
    
    # Display sample stores
    print("\n--- Sample 5 Unseen Test Stores for Model Testing ---")
    cols_preview = ["store_name", "address", "cap", "comune", "machine_type", "actual_mean_bottiglie_day", "actual_mean_presenze_day"]
    print(store_summary[cols_preview].head(5).to_string(index=False))

if __name__ == "__main__":
    export_test_holdout()
