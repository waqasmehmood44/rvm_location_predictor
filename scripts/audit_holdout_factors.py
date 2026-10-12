"""
Comprehensive Factor Audit & Store-Level Evaluation on 331 Test Holdout Stores.

Evaluates every store from the untouched 20% holdout set through:
1. Pure Multi-Task Neural Network (MultiTaskRVMNet)
2. Stacking Ensemble (LightGBM + NeuralNet)
3. Factor-Stratified Performance Breakdown:
   - Hardware Model (TipoMacchina)
   - Store Format / Scale (nielsen_tipo_negozio & surface sqm)
   - Retail Insegna / Brand & Buying Group (nielsen_gruppo)
   - Geographic Hierarchy (Nielsen Area & Regione)
   - Catchment Demographics & Competition Density
   - Volume Tiers (identifying compression or bias)
4. Exports store-by-store results and factor diagnostic scorecards.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

sys.path.insert(0, os.path.abspath("."))
from src.model.preprocessing import RVMPreprocessor
from src.model.network import MultiTaskRVMNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    medae = float(median_absolute_error(y_true, y_pred))
    
    # Avoid zero division in percentage error
    mask = y_true > 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        medape = float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape, medape = 0.0, 0.0
        
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    
    return {
        "MAE": round(mae, 1),
        "RMSE": round(rmse, 1),
        "MedAE": round(medae, 1),
        "MAPE_pct": round(mape, 1),
        "MedAPE_pct": round(medape, 1),
        "Bias": round(bias, 1),
        "R2": round(r2, 3),
        "N": int(len(y_true))
    }

def run_factor_audit(
    models_dir: str = "models",
    holdout_parquet: str = "models/preprocessing/test_holdout.parquet",
    output_dir: str = "reports"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    # 1. Load Preprocessor
    logger.info("Loading feature preprocessor...")
    preprocessor = RVMPreprocessor.load(f"{models_dir}/preprocessing/preprocessor.joblib")
    
    # 2. Load Neural Network
    logger.info("Loading production MultiTaskRVMNet...")
    nn_model = MultiTaskRVMNet(
        num_numeric_features=len(preprocessor.num_features),
        cat_cardinalities=preprocessor.get_cat_cardinalities(),
        cat_features=preprocessor.cat_features,
        hidden_dims=(256, 128, 64),
        head_dim=32,
        dropout=0.0
    ).to(device)
    nn_model.load_state_dict(torch.load(f"{models_dir}/best_model/model.pt", map_location=device, weights_only=True))
    nn_model.eval()
    
    # 3. Load Ensemble Models if available
    lgb_b, lgb_p = None, None
    w_b, w_p = 0.90, 0.65
    lgb_b_path = f"{models_dir}/ensemble/lgb_bottiglie.txt"
    lgb_p_path = f"{models_dir}/ensemble/lgb_presenze.txt"
    ens_cfg_path = f"{models_dir}/ensemble/ensemble_config.json"
    
    if os.path.exists(lgb_b_path) and os.path.exists(lgb_p_path):
        logger.info("Loading LightGBM ensemble boosters...")
        lgb_b = lgb.Booster(model_file=lgb_b_path)
        lgb_p = lgb.Booster(model_file=lgb_p_path)
        if os.path.exists(ens_cfg_path):
            with open(ens_cfg_path, "r") as f:
                cfg = json.load(f)
                w_b = float(cfg.get("blend_weight_bottiglie_lgb", 0.90))
                w_p = float(cfg.get("blend_weight_presenze_lgb", 0.65))
        logger.info(f"Ensemble blend weights: Bottiglie LGB={w_b:.2f} / NN={1-w_b:.2f}, Presenze LGB={w_p:.2f} / NN={1-w_p:.2f}")

    # 4. Load Holdout Dataset
    logger.info(f"Loading holdout test data from {holdout_parquet}...")
    df_test = pd.read_parquet(holdout_parquet)
    logger.info(f"Loaded {len(df_test):,} daily observations across {df_test['PF'].nunique()} unique physical locations.")
    
    # 5. Transform Features
    logger.info("Transforming holdout feature matrices...")
    X_num_te, X_cat_te, y_b_te, y_p_te = preprocessor.transform(df_test, nielsen_dropout=0.0)
    
    # 6. Run Neural Network Batch Inference
    logger.info("Running MultiTaskRVMNet forward inference...")
    batch_size = 8192
    preds_b_list, preds_p_list = [], []
    with torch.no_grad():
        for i in range(0, len(df_test), batch_size):
            bn = torch.tensor(X_num_te[i:i+batch_size], dtype=torch.float32, device=device)
            bc = torch.tensor(X_cat_te[i:i+batch_size], dtype=torch.long, device=device)
            pb, pp = nn_model(bn, bc)
            preds_b_list.append(pb.cpu().numpy())
            preds_p_list.append(pp.cpu().numpy())
            
    nn_pred_log_b = np.vstack(preds_b_list).squeeze()
    nn_pred_log_p = np.vstack(preds_p_list).squeeze()
    nn_pred_b = np.clip(np.expm1(nn_pred_log_b), 0.0, None)
    nn_pred_p = np.clip(np.expm1(nn_pred_log_p), 0.0, None)
    
    df_test["nn_pred_b"] = nn_pred_b
    df_test["nn_pred_p"] = nn_pred_p
    
    # 7. Run Ensemble Inference if available
    if lgb_b is not None and lgb_p is not None:
        logger.info("Running LightGBM ensemble inference...")
        X_all = np.hstack([X_num_te, X_cat_te])
        lgb_pred_log_b = lgb_b.predict(X_all)
        lgb_pred_log_p = lgb_p.predict(X_all)
        lgb_pred_b = np.clip(np.expm1(lgb_pred_log_b), 0.0, None)
        lgb_pred_p = np.clip(np.expm1(lgb_pred_log_p), 0.0, None)
        
        df_test["ens_pred_b"] = w_b * lgb_pred_b + (1.0 - w_b) * nn_pred_b
        df_test["ens_pred_p"] = w_p * lgb_pred_p + (1.0 - w_p) * nn_pred_p
    else:
        df_test["ens_pred_b"] = nn_pred_b
        df_test["ens_pred_p"] = nn_pred_p
        
    # 8. Aggregate Daily Observations to Store-Level (331 unique PFs)
    logger.info("Aggregating daily predictions to store-level (331 physical locations)...")
    
    def first_val(s):
        v = s.dropna()
        return v.iloc[0] if not v.empty else "UNKNOWN"
        
    stores = df_test.groupby("PF").agg(
        Descrizione=("Descrizione", first_val),
        store_name=("CLIENTE", first_val),
        insegna=("nielsen_insegna", first_val),
        gruppo=("nielsen_gruppo", first_val),
        tipo_negozio=("nielsen_tipo_negozio", first_val),
        machine_type=("TipoMacchina", first_val),
        fornitore=("Fornitore", first_val),
        regione=("Regione", first_val),
        provincia=("Provincia", first_val),
        comune=("Comune", first_val),
        cap=("Cap", first_val),
        address=("Indirizzo", first_val),
        nielsen_area=("nielsen_area", first_val),
        sales_area_sqm=("nielsen_superficie_vendita", "median"),
        checkouts=("nielsen_numero_casse", "median"),
        population=("population", "first"),
        pop_density=("population_density", "first"),
        total_days=("GIORNO", "count"),
        actual_mean_b=("BOTTIGLIE", "mean"),
        actual_median_b=("BOTTIGLIE", "median"),
        actual_mean_p=("PRESENZE", "mean"),
        actual_median_p=("PRESENZE", "median"),
        nn_pred_mean_b=("nn_pred_b", "mean"),
        nn_pred_median_b=("nn_pred_b", "median"),
        nn_pred_mean_p=("nn_pred_p", "mean"),
        nn_pred_median_p=("nn_pred_p", "median"),
        ens_pred_mean_b=("ens_pred_b", "mean"),
        ens_pred_median_b=("ens_pred_b", "median"),
        ens_pred_mean_p=("ens_pred_p", "mean"),
        ens_pred_median_p=("ens_pred_p", "median")
    ).reset_index()
    
    # Clean numeric rounding
    stores["actual_mean_b"] = stores["actual_mean_b"].round(1)
    stores["actual_median_b"] = stores["actual_median_b"].round(1)
    stores["actual_mean_p"] = stores["actual_mean_p"].round(1)
    stores["actual_median_p"] = stores["actual_median_p"].round(1)
    stores["nn_pred_mean_b"] = stores["nn_pred_mean_b"].round(1)
    stores["ens_pred_mean_b"] = stores["ens_pred_mean_b"].round(1)
    stores["nn_pred_mean_p"] = stores["nn_pred_mean_p"].round(1)
    stores["ens_pred_mean_p"] = stores["ens_pred_mean_p"].round(1)
    
    # Store-Level Error Computations
    stores["nn_abs_err_b"] = (stores["nn_pred_mean_b"] - stores["actual_mean_b"]).abs().round(1)
    stores["nn_pct_err_b"] = ((stores["nn_abs_err_b"] / stores["actual_mean_b"].replace(0, np.nan)) * 100.0).round(1)
    stores["nn_bias_b"] = (stores["nn_pred_mean_b"] - stores["actual_mean_b"]).round(1)
    
    stores["ens_abs_err_b"] = (stores["ens_pred_mean_b"] - stores["actual_mean_b"]).abs().round(1)
    stores["ens_pct_err_b"] = ((stores["ens_abs_err_b"] / stores["actual_mean_b"].replace(0, np.nan)) * 100.0).round(1)
    stores["ens_bias_b"] = (stores["ens_pred_mean_b"] - stores["actual_mean_b"]).round(1)
    
    stores["nn_accuracy_pct"] = np.clip(100.0 - stores["nn_pct_err_b"], 0.0, 100.0).round(1)
    stores["ens_accuracy_pct"] = np.clip(100.0 - stores["ens_pct_err_b"], 0.0, 100.0).round(1)

    # Binning Helper Columns for Granular Stratification
    # Surface bins
    def bin_surface(s):
        if pd.isna(s) or s <= 0:
            return "Unknown / Imputed"
        elif s < 450:
            return "Small / Proximity (< 450 m²)"
        elif s < 1200:
            return "Standard Supermarket (450-1,200 m²)"
        elif s < 2500:
            return "Superstore (1,200-2,500 m²)"
        else:
            return "Hypermarket (> 2,500 m²)"
    stores["surface_bin"] = stores["sales_area_sqm"].apply(bin_surface)

    # Population density bins
    def bin_density(d):
        if pd.isna(d) or d <= 0:
            return "Unknown"
        elif d < 200:
            return "Rural / Low Density (< 200 hab/km²)"
        elif d < 800:
            return "Suburban / Moderate (200-800 hab/km²)"
        elif d < 2500:
            return "Dense Urban (800-2,500 hab/km²)"
        else:
            return "Metropolitan (> 2,500 hab/km²)"
    stores["density_bin"] = stores["pop_density"].apply(bin_density)

    # Actual volume tiers
    def bin_volume(v):
        if v < 300:
            return "Tier 4: Low Volume (< 300 b/day)"
        elif v < 600:
            return "Tier 3: Moderate (300 - 600 b/day)"
        elif v < 1200:
            return "Tier 2: High (600 - 1,200 b/day)"
        else:
            return "Tier 1: Exceptional (> 1,200 b/day)"
    stores["volume_tier"] = stores["actual_mean_b"].apply(bin_volume)

    # 9. FACTOR-STRATIFIED PERFORMANCE BREAKDOWN
    logger.info("Computing factor-stratified scorecards...")
    
    factors_to_analyze = [
        ("Hardware Model (TipoMacchina)", "machine_type"),
        ("Hardware Supplier (Fornitore)", "fornitore"),
        ("Store Format (nielsen_tipo_negozio)", "tipo_negozio"),
        ("Sales Surface Category", "surface_bin"),
        ("Retail Buying Group (nielsen_gruppo)", "gruppo"),
        ("Standardized Insegna / Banner", "insegna"),
        ("Nielsen Macro Area", "nielsen_area"),
        ("Italian Region (Regione)", "regione"),
        ("Catchment Population Density", "density_bin"),
        ("Actual Store Volume Tier", "volume_tier")
    ]
    
    factor_results = {}
    
    for factor_label, factor_col in factors_to_analyze:
        grouped = stores.groupby(factor_col)
        factor_records = []
        for name, grp in grouped:
            if len(grp) < 1:
                continue
            nn_m = compute_metrics(grp["actual_mean_b"], grp["nn_pred_mean_b"])
            ens_m = compute_metrics(grp["actual_mean_b"], grp["ens_pred_mean_b"])
            
            # Diagnostic performance rating
            # Good: MedAPE <= 25% and abs(bias) <= 150
            # Moderate: MedAPE <= 40%
            # Poor: MedAPE > 40% or abs(bias) > 300
            if nn_m["MedAPE_pct"] <= 25.0 and abs(nn_m["Bias"]) <= 150.0:
                rating = "PERFORMS WELL (Accurate)"
            elif nn_m["MedAPE_pct"] <= 38.0:
                rating = "MODERATE (Acceptable)"
            else:
                rating = "UNDERPERFORMING (Action Needed)"
                
            rec = {
                "factor_value": str(name),
                "store_count": len(grp),
                "actual_mean_b": round(float(grp["actual_mean_b"].mean()), 1),
                "actual_median_b": round(float(grp["actual_mean_b"].median()), 1),
                "nn_pred_mean_b": round(float(grp["nn_pred_mean_b"].mean()), 1),
                "nn_mae": nn_m["MAE"],
                "nn_medae": nn_m["MedAE"],
                "nn_mape_pct": nn_m["MAPE_pct"],
                "nn_medape_pct": nn_m["MedAPE_pct"],
                "nn_bias": nn_m["Bias"],
                "ens_mae": ens_m["MAE"],
                "ens_medape_pct": ens_m["MedAPE_pct"],
                "ens_bias": ens_m["Bias"],
                "performance_rating": rating
            }
            factor_records.append(rec)
            
        # Sort by store count descending
        factor_records.sort(key=lambda x: x["store_count"], reverse=True)
        factor_results[factor_label] = factor_records

    # 10. LightGBM Feature Importance (Attribution)
    lgb_importance = []
    if lgb_b is not None:
        feat_names = preprocessor.all_feature_names
        gain_imp = lgb_b.feature_importance(importance_type="gain")
        split_imp = lgb_b.feature_importance(importance_type="split")
        total_gain = float(np.sum(gain_imp)) if np.sum(gain_imp) > 0 else 1.0
        
        for name, g, s in zip(feat_names, gain_imp, split_imp):
            lgb_importance.append({
                "feature": name,
                "gain": round(float(g), 1),
                "gain_pct": round(float(g) / total_gain * 100.0, 2),
                "splits": int(s)
            })
        lgb_importance.sort(key=lambda x: x["gain"], reverse=True)

    # 11. Overall Benchmark Summary
    overall_nn = compute_metrics(stores["actual_mean_b"], stores["nn_pred_mean_b"])
    overall_ens = compute_metrics(stores["actual_mean_b"], stores["ens_pred_mean_b"])
    
    summary_report = {
        "total_holdout_stores": len(stores),
        "total_holdout_daily_rows": len(df_test),
        "overall_nn_store_metrics": overall_nn,
        "overall_ensemble_store_metrics": overall_ens,
        "factors": factor_results,
        "feature_importance_top20": lgb_importance[:20] if lgb_importance else []
    }
    
    # Save full JSON report
    json_path = f"{output_dir}/holdout_factor_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved full JSON audit report to {json_path}")
    
    # Save store-by-store CSV (all 331 stores)
    csv_path = f"{output_dir}/holdout_store_predictions_331.csv"
    cols_to_save = [
        "Descrizione", "PF", "store_name", "insegna", "gruppo", "comune", "provincia", "regione",
        "machine_type", "total_days", "volume_tier",
        "actual_mean_b", "ens_pred_mean_b", "ens_abs_err_b", "ens_pct_err_b", "ens_accuracy_pct", "ens_bias_b",
        "actual_mean_p", "ens_pred_mean_p",
        "nn_pred_mean_b", "nn_pct_err_b", "nn_accuracy_pct"
    ]
    stores[cols_to_save].sort_values(by="actual_mean_b", ascending=False).to_csv(csv_path, index=False)
    logger.info(f"Saved store-by-store predictions (331 stores) to {csv_path}")
    
    # Generate structured Markdown Summary
    md_path = f"{output_dir}/factor_performance_audit.md"
    generate_markdown_audit_report(summary_report, md_path)
    logger.info(f"Saved markdown executive report to {md_path}")
    
    print("\n" + "=" * 80)
    print("HOLDOUT FACTOR AUDIT SUMMARY (331 UNSEEN STORES)")
    print("=" * 80)
    print(f"Total Stores Tested:    {len(stores)}")
    print(f"Overall Store NN MAE:   {overall_nn['MAE']} bottles/day (MedAPE: {overall_nn['MedAPE_pct']}%)")
    print(f"Overall Ensemble MAE:   {overall_ens['MAE']} bottles/day (MedAPE: {overall_ens['MedAPE_pct']}%)")
    print(f"Overall NN Bias:        {overall_nn['Bias']} b/day (Systematic Under/Over-prediction)")
    print("=" * 80)
    
    return summary_report

def generate_markdown_audit_report(report: dict, md_path: str):
    nn_m = report["overall_nn_store_metrics"]
    ens_m = report["overall_ensemble_store_metrics"]
    factors = report["factors"]
    
    lines = []
    lines.append("# Holdout Store Factor Performance Audit & Diagnostics")
    lines.append(f"**Total Stores Evaluated:** {report['total_holdout_stores']} Unseen Physical Locations (PFs)")
    lines.append(f"**Daily Records Processed:** {report['total_holdout_daily_rows']:,} Observations\n")
    
    lines.append("## 1. Overall Store-Level Predictive Accuracy")
    lines.append("| Metric | Pure Neural Network | Stacking Ensemble | Delta Improvement |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Mean Absolute Error (MAE)** | **{nn_m['MAE']} b/day** | **{ens_m['MAE']} b/day** | {ens_m['MAE'] - nn_m['MAE']:.1f} b/day |")
    lines.append(f"| **Median Absolute Error (MedAE)** | **{nn_m['MedAE']} b/day** | **{ens_m['MedAE']} b/day** | {ens_m['MedAE'] - nn_m['MedAE']:.1f} b/day |")
    lines.append(f"| **Median Absolute % Error (MedAPE)** | **{nn_m['MedAPE_pct']}%** | **{ens_m['MedAPE_pct']}%** | {ens_m['MedAPE_pct'] - nn_m['MedAPE_pct']:.1f}% |")
    lines.append(f"| **Mean Absolute % Error (MAPE)** | **{nn_m['MAPE_pct']}%** | **{ens_m['MAPE_pct']}%** | {ens_m['MAPE_pct'] - nn_m['MAPE_pct']:.1f}% |")
    lines.append(f"| **Systematic Bias (Pred - Act)** | **{nn_m['Bias']} b/day** | **{ens_m['Bias']} b/day** | - |")
    lines.append(f"| **R² Score** | **{nn_m['R2']}** | **{ens_m['R2']}** | +{ens_m['R2'] - nn_m['R2']:.3f} |\n")
    
    lines.append("## 2. Factor-by-Factor Diagnostic Scorecards\n")
    
    for f_label, f_records in factors.items():
        lines.append(f"### {f_label}")
        lines.append("| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
        for r in f_records:
            badge = "🟢 " if "WELL" in r["performance_rating"] else ("🟡 " if "MODERATE" in r["performance_rating"] else "🔴 ")
            lines.append(
                f"| **{r['factor_value']}** | {r['store_count']} | {r['actual_mean_b']} | {r['nn_pred_mean_b']} | "
                f"{r['nn_mae']} | {r['nn_medape_pct']}% | {r['nn_bias']:+.1f} | {badge}{r['performance_rating']} |"
            )
        lines.append("")
        
    lines.append("## 3. Top Predictive Factors (Feature Attribution - LightGBM Gain)")
    lines.append("| Rank | Factor / Feature | Gain Contribution % | Total Tree Splits |")
    lines.append("| :---: | :--- | :---: | :---: |")
    for idx, f in enumerate(report.get("feature_importance_top20", []), 1):
        lines.append(f"| {idx} | `{f['feature']}` | {f['gain_pct']}% | {f['splits']} |")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    run_factor_audit()
