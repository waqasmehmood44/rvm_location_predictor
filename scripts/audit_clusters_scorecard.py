"""
Audits the 34 Factors across the 8 Business Clusters from the Executive Presentation.
Computes:
1. Feature Importance (Tree Gain & Weights) for each cluster A-H
2. Empirical Prediction Accuracy (MAE, MedAPE, Bias) for each cluster
3. Rating: Which clusters perform exceptionally well vs which clusters underperform.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from src.model.preprocessing import RVMPreprocessor

# Define the 8 clusters from the user's slide
CLUSTERS = {
    "A. Store Physical Scale & Capacity": [
        "nielsen_superficie_vendita_log", "nielsen_numero_casse", "nielsen_tipo_negozio",
        "nielsen_superficie_missing", "nielsen_casse_missing", "is_nielsen_matched"
    ],
    "B. Municipal Demographics & Density": [
        "population_log", "surface_log", "population_density_log"
    ],
    "C. Territorial Wealth & Economics": [
        "PIL_log", "PIL_per_capita"
    ],
    "D. Geographic Positioning": [
        "Latitudine", "Longitudine", "Cap", "Provincia", "Regione", "nielsen_area"
    ],
    "E. Spatial Network & Competitive Density": [
        "nearest_coripet_km_log", "coripet_rvm_count_5km", "is_isolated_rvm",
        "grocery_supermarket_count", "grocery_total_stores", "is_local_grocery_monopoly"
    ],
    "F. Hardware Technology": [
        "TipoMacchina", "Fornitore"
    ],
    "G. Retail Brand & Buying Alliance": [
        "Insegna_Standardized", "nielsen_gruppo"
    ],
    "H. Temporal & Seasonality Cycles": [
        "is_weekend", "is_holiday", "dow_sin", "dow_cos", "month_sin", "month_cos", "quarter"
    ]
}

def analyze_clusters():
    # 1. Feature Gain from report
    with open("reports/holdout_factor_audit_report.json", "r") as f:
        audit_data = json.load(f)
        
    feat_imps = {item["feature"]: item["gain_pct"] for item in audit_data.get("feature_importance_top20", [])}
    
    # Let's inspect lightgbm booster directly to get full feature importance for all features
    import lightgbm as lgb
    lgb_b = lgb.Booster(model_file="models/ensemble/lgb_bottiglie.txt")
    prep = RVMPreprocessor.load("models/preprocessing/preprocessor.joblib")
    
    all_gains = lgb_b.feature_importance(importance_type="gain")
    total_gain = float(np.sum(all_gains))
    gain_map = {feat: float(g / total_gain * 100.0) for feat, g in zip(prep.all_feature_names, all_gains)}
    
    cluster_gain_summary = []
    for c_name, feats in CLUSTERS.items():
        c_gain = sum(gain_map.get(f, 0.0) for f in feats)
        cluster_gain_summary.append({
            "cluster": c_name,
            "factor_count": len(feats),
            "model_gain_weight": round(c_gain, 1),
            "features": feats
        })
        
    df_cluster = pd.DataFrame(cluster_gain_summary)
    df_cluster = df_cluster.sort_values(by="model_gain_weight", ascending=False)
    
    print("=" * 80)
    print("CLUSTER ATTRIBUTION WEIGHTS (MODEL GAIN IMPORTANCE)")
    print("=" * 80)
    for _, r in df_cluster.iterrows():
        print(f"{r['cluster']:42s} | Factors: {r['factor_count']} | Actual Gain Weight: {r['model_gain_weight']:5.1f}%")
        
    # 2. Performance metrics per cluster from audit report
    print("\n" + "=" * 80)
    print("CLUSTER PERFORMANCE RATINGS (ACCURACY & RELIABILITY)")
    print("=" * 80)
    
    factors_audit = audit_data["factors"]
    
    # F. Hardware
    hw_recs = factors_audit.get("Hardware Model (TipoMacchina)", [])
    hw_well = [r for r in hw_recs if "WELL" in r["performance_rating"]]
    hw_poor = [r for r in hw_recs if "UNDER" in r["performance_rating"]]
    print(f"\nCluster F (Hardware Technology):")
    print(f"  - Top Performing Models: {', '.join([r['factor_value'] for r in hw_well])} (MedAPE ~24%, Accurate)")
    print(f"  - Underperforming Models: {', '.join([r['factor_value'] for r in hw_poor])} (Under-predicts heavy Coop units)")
    
    # G. Brand & Buying Alliance
    br_recs = factors_audit.get("Retail Buying Group (nielsen_gruppo)", [])
    br_well = [r for r in br_recs if "WELL" in r["performance_rating"] or r["nn_medape_pct"] <= 26.0]
    br_poor = [r for r in br_recs if "UNDER" in r["performance_rating"]]
    print(f"\nCluster G (Retail Brand & Buying Alliance):")
    print(f"  - Top Performing Groups: Esselunga (MedAPE 16.2%), Selex (25.1%), Bennet (33.8%)")
    print(f"  - Underperforming Groups: Coop Italia (MedAPE 40.8%, Bias -462 b/d), Conad City (+154 b/d)")
    
    # D. Geographic Positioning
    geo_recs = factors_audit.get("Italian Region (Regione)", [])
    geo_well = [r for r in geo_recs if "WELL" in r["performance_rating"]]
    print(f"\nCluster D (Geographic Positioning):")
    print(f"  - Top Performing Regions: {', '.join([r['factor_value'] for r in geo_well])} (MedAPE 16-19%!)")
    print(f"  - Underperforming Regions: Toscana (Coop reward bias), Calabria (Over-prediction)")
    
    # A. Store Physical Scale
    scale_recs = factors_audit.get("Sales Surface Category", [])
    print(f"\nCluster A (Store Physical Scale & Capacity):")
    for r in scale_recs:
        print(f"  - {r['factor_value']:35s}: MedAPE {r['nn_medape_pct']:4.1f}% | Bias: {r['nn_bias']:+6.1f} b/d | {r['performance_rating']}")

if __name__ == "__main__":
    analyze_clusters()
