"""
Precomputes spatial competition and network density features for all Coripet PFs:
1. Vectorized Haversine distances to find nearest operating Coripet RVM (km).
2. Count of other Coripet RVMs within 5 km radius.
3. Competitive grocery retail density (supermarkets, total food stores) from Nielsen lookup.
4. Local grocery monopoly indicator (supermarkets <= 1, hypermarkets == 0).
5. Isolated RVM indicator (0 other Coripet RVMs within 5 km).
Outputs: data/processed/pf_spatial_features.parquet
"""
import os
import time
import numpy as np
import pandas as pd

def build_pf_spatial_features():
    print("=" * 75)
    print("PRECOMPUTING STORE-LEVEL SPATIAL & MONOPOLY FEATURES")
    print("=" * 75)
    t0 = time.time()
    
    network_path = "data/processed/coripet_rvm_spatial_network.parquet"
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Missing {network_path}")
        
    df_net = pd.read_parquet(network_path)
    print(f"Loaded {len(df_net):,} physical points from {network_path}.")
    
    # 1. Pairwise Haversine Distances across all operational PFs
    lats = df_net["Latitudine"].to_numpy(dtype=float)
    lons = df_net["Longitudine"].to_numpy(dtype=float)
    
    r_lat = np.radians(lats)
    r_lon = np.radians(lons)
    
    dlat = r_lat[:, None] - r_lat[None, :]
    dlon = r_lon[:, None] - r_lon[None, :]
    
    a = np.sin(dlat / 2.0)**2 + np.cos(r_lat[:, None]) * np.cos(r_lat[None, :]) * np.sin(dlon / 2.0)**2
    dist_matrix = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    
    # Exclude self-distance by setting diagonal to infinity
    np.fill_diagonal(dist_matrix, np.inf)
    
    nearest_coripet_km = np.min(dist_matrix, axis=1)
    coripet_rvm_count_5km = np.sum(dist_matrix <= 5.0, axis=1)
    
    df_net["nearest_coripet_km"] = np.round(nearest_coripet_km, 2)
    df_net["coripet_rvm_count_5km"] = coripet_rvm_count_5km.astype(int)
    df_net["is_isolated_rvm"] = (coripet_rvm_count_5km == 0).astype(int)
    
    # 2. Competitive Grocery Density Lookup (CAP + Comune, fallback to Comune)
    density_cap_path = "data/processed/competitive_density_lookup.parquet"
    density_comune_path = "data/processed/competitive_density_comune.parquet"
    
    df_density_cap = pd.read_parquet(density_cap_path) if os.path.exists(density_cap_path) else pd.DataFrame()
    df_density_com = pd.read_parquet(density_comune_path) if os.path.exists(density_comune_path) else pd.DataFrame()
    
    df_net["CAP_str"] = df_net["Cap"].astype(str).str.zfill(5)
    df_net["COMUNE_norm"] = df_net["Comune"].str.strip().str.upper()
    
    # Merge primary: CAP + COMUNE
    if not df_density_cap.empty:
        cap_cols = ["CAP_str", "COMUNE_norm", "total_grocery_stores", "supermarkets", "hypermarkets", "discounts"]
        df_net = df_net.merge(df_density_cap[cap_cols], on=["CAP_str", "COMUNE_norm"], how="left")
    else:
        df_net["total_grocery_stores"] = np.nan
        df_net["supermarkets"] = np.nan
        df_net["hypermarkets"] = np.nan
        df_net["discounts"] = np.nan
        
    # Merge secondary fallback: COMUNE alone for any unmatched
    if not df_density_com.empty:
        com_cols = ["COMUNE_norm", "total_grocery_stores", "supermarkets", "hypermarkets", "discounts"]
        missing_idx = df_net[df_net["total_grocery_stores"].isna()].index
        if len(missing_idx) > 0:
            com_lookup = df_density_com[com_cols].set_index("COMUNE_norm")
            for idx in missing_idx:
                c_norm = df_net.loc[idx, "COMUNE_norm"]
                if c_norm in com_lookup.index:
                    matched_row = com_lookup.loc[c_norm]
                    if isinstance(matched_row, pd.DataFrame):
                        matched_row = matched_row.iloc[0]
                    df_net.loc[idx, "total_grocery_stores"] = matched_row["total_grocery_stores"]
                    df_net.loc[idx, "supermarkets"] = matched_row["supermarkets"]
                    df_net.loc[idx, "hypermarkets"] = matched_row["hypermarkets"]
                    df_net.loc[idx, "discounts"] = matched_row["discounts"]

    # Fill default national medians for any remaining unlisted rural communes
    df_net["total_grocery_stores"] = df_net["total_grocery_stores"].fillna(5.0)
    df_net["supermarkets"] = df_net["supermarkets"].fillna(2.0)
    df_net["hypermarkets"] = df_net["hypermarkets"].fillna(0.0)
    df_net["discounts"] = df_net["discounts"].fillna(1.0)
    
    # Rename & compute monopoly flags
    df_net["grocery_total_stores"] = df_net["total_grocery_stores"].astype(float)
    df_net["grocery_supermarket_count"] = df_net["supermarkets"].astype(float)
    df_net["is_local_grocery_monopoly"] = (
        (df_net["grocery_supermarket_count"] <= 1.0) & (df_net["hypermarkets"] == 0.0)
    ).astype(int)
    
    feature_cols = [
        "PF", "nearest_coripet_km", "coripet_rvm_count_5km", "is_isolated_rvm",
        "grocery_supermarket_count", "grocery_total_stores", "is_local_grocery_monopoly"
    ]
    
    out_df = df_net[feature_cols].copy()
    out_path = "data/processed/pf_spatial_features.parquet"
    out_df.to_parquet(out_path, index=False)
    
    print(f"\nSuccessfully generated {out_path} ({len(out_df)} PFs) in {time.time() - t0:.2f}s!")
    print("\n--- Summary of Computed Spatial Features ---")
    print(out_df.describe().round(2).to_string())
    print("\nSample Monopoly / Isolated stores:")
    sample_monopoly = out_df[out_df["is_local_grocery_monopoly"] == 1].head(5)
    print(sample_monopoly.to_string(index=False))

if __name__ == "__main__":
    build_pf_spatial_features()
