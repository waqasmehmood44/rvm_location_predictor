"""
Utility to find historical peer stores and verify new prospective location predictions.
"""
import os
import sys
import pandas as pd
import numpy as np

def find_peer_stores(
    target_area_sqm: float = 1200.0,
    target_checkouts: float = 6.0,
    predicted_bottles: float = 347.9,
    predicted_presenze: float = 13.4,
    regione: str = "Friuli-Venezia Giulia",
    n_peers: int = 5
):
    csv_path = "data/processed/test_holdout_stores_summary.csv"
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate volume distance to prediction
    df["vol_diff"] = (df["actual_median_bottiglie_day"] - predicted_bottles).abs()
    df["pres_diff"] = (df["actual_median_presenze_day"] - predicted_presenze).abs()
    
    # Normalize score
    df["similarity_score"] = (df["vol_diff"] / predicted_bottles) + (df["pres_diff"] / predicted_presenze)
    
    peers = df.sort_values("similarity_score").head(n_peers)
    
    print("=" * 80)
    print(f"TOP {n_peers} HISTORICAL PEER STORES (Closest to {predicted_bottles:.1f} b/d & {predicted_presenze:.1f} u/d):")
    print("=" * 80)
    for idx, row in peers.iterrows():
        print(f"PF: {row['PF']} | Store: {row['store_name']} ({row['insegna']}) | {row['comune']} ({row['provincia']}, {row['regione']})")
        print(f"  Machine: {row['machine_type']} | Observed Days: {row['total_days_observed']}")
        print(f"  Actual Bottles/Day:  {row['actual_mean_bottiglie_day']:.1f} (Mean)  |  {row['actual_median_bottiglie_day']:.1f} (Median)")
        print(f"  Actual Presenze/Day: {row['actual_mean_presenze_day']:.1f} (Mean)  |  {row['actual_median_presenze_day']:.1f} (Median)")
        print(f"  Variance vs Prediction: {row['actual_median_bottiglie_day'] - predicted_bottles:+.1f} b/d ({((row['actual_median_bottiglie_day'] - predicted_bottles)/predicted_bottles)*100:+.1f}%)")
        print("-" * 80)

if __name__ == "__main__":
    find_peer_stores()
