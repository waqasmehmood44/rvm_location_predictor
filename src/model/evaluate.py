"""
Model evaluation, baseline comparison, and residual analysis for Coripet RVM Predictor.
Evaluates the best trained neural network on the completely untouched 20% unseen-PF test holdout.
Computes MAE, RMSE, R2, MedAE, relative errors, and breakdowns by machine type, region, banner, and volume.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from typing import Dict, Any, Tuple

from src.model.preprocessing import RVMPreprocessor
from src.model.dataset import RVMDataset
from src.model.network import MultiTaskRVMNet

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression metrics on original scale."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mean_val = float(np.mean(y_true))
    mae_pct = float((mae / mean_val * 100.0) if mean_val > 0 else 0.0)
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MedAE": round(medae, 2),
        "MAE_pct_mean": round(mae_pct, 2)
    }

def evaluate_model(
    models_dir: str = "models",
    test_parquet: str = "models/preprocessing/test_holdout.parquet",
    training_parquet: str = "data/processed/modeling_daily_rvm.parquet",
    batch_size: int = 4096
) -> Dict[str, Any]:
    print("=" * 80)
    print("CORIPET RVM PREDICTOR — MODEL EVALUATION & BASELINE BENCHMARK")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Hardware Device: {device}")
    
    # 1. Load preprocessor and test holdout
    preprocessor_path = f"{models_dir}/preprocessing/preprocessor.joblib"
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = RVMPreprocessor.load(preprocessor_path)
    
    print(f"Loading untouched test holdout from {test_parquet}...")
    test_df = pd.read_parquet(test_parquet)
    n_test_rows = len(test_df)
    n_test_pfs = test_df["PF"].nunique()
    print(f"Test Set: {n_test_rows:,} daily observations across {n_test_pfs} completely unseen PFs.")
    
    # 2. Transform test data
    X_num, X_cat, y_bottiglie_log, y_presenze_log = preprocessor.transform(test_df, nielsen_dropout=0.0)
    
    # Ground truth on original scale
    y_true_bottiglie = test_df["BOTTIGLIE"].values.astype(float)
    y_true_presenze = test_df["PRESENZE"].values.astype(float)
    
    # 3. Load trained neural network
    model_path = f"{models_dir}/best_model/model.pt"
    print(f"Loading trained model weights from {model_path}...")
    
    model = MultiTaskRVMNet(
        num_numeric_features=len(preprocessor.num_features),
        cat_cardinalities=preprocessor.get_cat_cardinalities(),
        cat_features=preprocessor.cat_features,
        hidden_dims=(256, 128, 64),
        head_dim=32,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 4. Run inference in batches
    preds_b_list = []
    preds_p_list = []
    
    with torch.no_grad():
        for start_idx in range(0, n_test_rows, batch_size):
            end_idx = min(start_idx + batch_size, n_test_rows)
            b_num = torch.tensor(X_num[start_idx:end_idx], dtype=torch.float32, device=device)
            b_cat = torch.tensor(X_cat[start_idx:end_idx], dtype=torch.long, device=device)
            
            pb, pp = model(b_num, b_cat)
            preds_b_list.append(pb.cpu().numpy())
            preds_p_list.append(pp.cpu().numpy())
            
    pred_log_bottiglie = np.vstack(preds_b_list).squeeze()
    pred_log_presenze = np.vstack(preds_p_list).squeeze()
    
    # 5. Reverse log1p transformation back to natural scale
    # prediction = expm1(predicted_log_target), strictly clipped at 0
    pred_bottiglie = np.clip(np.expm1(pred_log_bottiglie), a_min=0.0, a_max=None)
    pred_presenze = np.clip(np.expm1(pred_log_presenze), a_min=0.0, a_max=None)
    
    # Store predictions in dataframe for group breakdown
    test_eval = test_df.copy()
    test_eval["pred_bottiglie"] = pred_bottiglie
    test_eval["pred_presenze"] = pred_presenze
    test_eval["resid_bottiglie"] = y_true_bottiglie - pred_bottiglie
    test_eval["resid_presenze"] = y_true_presenze - pred_presenze
    
    # 6. Overall Neural Network Metrics
    nn_bottiglie_metrics = compute_metrics(y_true_bottiglie, pred_bottiglie)
    nn_presenze_metrics = compute_metrics(y_true_presenze, pred_presenze)
    
    print("\n" + "="*80)
    print("NEURAL NETWORK TEST PERFORMANCE (ON COMPLETELY UNSEEN PFs)")
    print("="*80)
    print("BOTTIGLIE:")
    for k, v in nn_bottiglie_metrics.items():
        print(f"  {k}: {v}")
    print("\nPRESENZE:")
    for k, v in nn_presenze_metrics.items():
        print(f"  {k}: {v}")
        
    # 7. Baseline Models (Fitted on Training Set Only)
    print("\n" + "="*80)
    print("COMPUTING BASELINE BENCHMARKS")
    print("="*80)
    # Load training metadata or compute train means
    split_info_path = f"{models_dir}/preprocessing/split_info.json"
    with open(split_info_path) as f:
        split_info = json.load(f)
        
    # We load training subset to compute clean baseline parameters
    full_df = pd.read_parquet(training_parquet, columns=["PF", "TipoMacchina", "Regione", "BOTTIGLIE", "PRESENZE"])
    train_mask = ~full_df["PF"].isin(set(test_df["PF"].unique()))
    train_data = full_df[train_mask]
    
    # Baseline 1: Global Train Mean
    b1_mean_bottiglie = float(train_data["BOTTIGLIE"].mean())
    b1_mean_presenze = float(train_data["PRESENZE"].mean())
    b1_pred_b = np.full(n_test_rows, b1_mean_bottiglie)
    b1_pred_p = np.full(n_test_rows, b1_mean_presenze)
    b1_b_metrics = compute_metrics(y_true_bottiglie, b1_pred_b)
    b1_p_metrics = compute_metrics(y_true_presenze, b1_pred_p)
    
    # Baseline 2: Machine-Type Train Mean
    tm_b_means = train_data.groupby("TipoMacchina", observed=False)["BOTTIGLIE"].mean().to_dict()
    tm_p_means = train_data.groupby("TipoMacchina", observed=False)["PRESENZE"].mean().to_dict()
    b2_pred_b = test_df["TipoMacchina"].map(lambda tm: tm_b_means.get(tm, b1_mean_bottiglie)).values
    b2_pred_p = test_df["TipoMacchina"].map(lambda tm: tm_p_means.get(tm, b1_mean_presenze)).values
    b2_b_metrics = compute_metrics(y_true_bottiglie, b2_pred_b)
    b2_p_metrics = compute_metrics(y_true_presenze, b2_pred_p)
    
    # Baseline 3: Machine-Type + Regional Train Mean
    tm_reg_b = train_data.groupby(["TipoMacchina", "Regione"], observed=False)["BOTTIGLIE"].mean().to_dict()
    tm_reg_p = train_data.groupby(["TipoMacchina", "Regione"], observed=False)["PRESENZE"].mean().to_dict()
    b3_pred_b = [tm_reg_b.get((r.TipoMacchina, r.Regione), tm_b_means.get(r.TipoMacchina, b1_mean_bottiglie)) for r in test_df.itertuples()]
    b3_pred_p = [tm_reg_p.get((r.TipoMacchina, r.Regione), tm_p_means.get(r.TipoMacchina, b1_mean_presenze)) for r in test_df.itertuples()]
    b3_b_metrics = compute_metrics(y_true_bottiglie, np.array(b3_pred_b))
    b3_p_metrics = compute_metrics(y_true_presenze, np.array(b3_pred_p))
    
    print("\n--- Baseline Comparison (BOTTIGLIE) ---")
    print(f"  1. Global Train Mean:       MAE = {b1_b_metrics['MAE']:.1f}, RMSE = {b1_b_metrics['RMSE']:.1f}, R2 = {b1_b_metrics['R2']:.4f}")
    print(f"  2. Machine-Type Mean:        MAE = {b2_b_metrics['MAE']:.1f}, RMSE = {b2_b_metrics['RMSE']:.1f}, R2 = {b2_b_metrics['R2']:.4f}")
    print(f"  3. Machine + Region Mean:    MAE = {b3_b_metrics['MAE']:.1f}, RMSE = {b3_b_metrics['RMSE']:.1f}, R2 = {b3_b_metrics['R2']:.4f}")
    print(f"  -> MULTI-TASK NEURAL NET:    MAE = {nn_bottiglie_metrics['MAE']:.1f}, RMSE = {nn_bottiglie_metrics['RMSE']:.1f}, R2 = {nn_bottiglie_metrics['R2']:.4f}")
    
    print("\n--- Baseline Comparison (PRESENZE) ---")
    print(f"  1. Global Train Mean:       MAE = {b1_p_metrics['MAE']:.1f}, RMSE = {b1_p_metrics['RMSE']:.1f}, R2 = {b1_p_metrics['R2']:.4f}")
    print(f"  2. Machine-Type Mean:        MAE = {b2_p_metrics['MAE']:.1f}, RMSE = {b2_p_metrics['RMSE']:.1f}, R2 = {b2_p_metrics['R2']:.4f}")
    print(f"  3. Machine + Region Mean:    MAE = {b3_p_metrics['MAE']:.1f}, RMSE = {b3_p_metrics['RMSE']:.1f}, R2 = {b3_p_metrics['R2']:.4f}")
    print(f"  -> MULTI-TASK NEURAL NET:    MAE = {nn_presenze_metrics['MAE']:.1f}, RMSE = {nn_presenze_metrics['RMSE']:.1f}, R2 = {nn_presenze_metrics['R2']:.4f}")
    
    # 8. Breakdown by Machine Type
    print("\n" + "="*80)
    print("PERFORMANCE BREAKDOWN BY MACHINE TYPE")
    print("="*80)
    by_machine = {}
    for tm, group in test_eval.groupby("TipoMacchina", observed=False):
        if len(group) >= 50:
            m_b = compute_metrics(group["BOTTIGLIE"].values, group["pred_bottiglie"].values)
            m_p = compute_metrics(group["PRESENZE"].values, group["pred_presenze"].values)
            by_machine[str(tm)] = {
                "observations": len(group),
                "unique_pfs": group["PF"].nunique(),
                "bottiglie_mae": m_b["MAE"],
                "bottiglie_r2": m_b["R2"],
                "presenze_mae": m_p["MAE"],
                "presenze_r2": m_p["R2"]
            }
            print(f"  {tm:<18} (N={len(group):,}, PFs={group['PF'].nunique()}): B_MAE={m_b['MAE']:.1f} (R2={m_b['R2']:.2f}) | P_MAE={m_p['MAE']:.1f} (R2={m_p['R2']:.2f})")
            
    # 9. Breakdown by Region (Top Regions)
    print("\n" + "="*80)
    print("PERFORMANCE BREAKDOWN BY REGION")
    print("="*80)
    by_region = {}
    for reg, group in test_eval.groupby("Regione", observed=False):
        if len(group) >= 1000:
            m_b = compute_metrics(group["BOTTIGLIE"].values, group["pred_bottiglie"].values)
            m_p = compute_metrics(group["PRESENZE"].values, group["pred_presenze"].values)
            by_region[str(reg)] = {
                "observations": len(group),
                "unique_pfs": group["PF"].nunique(),
                "bottiglie_mae": m_b["MAE"],
                "bottiglie_r2": m_b["R2"],
                "presenze_mae": m_p["MAE"],
                "presenze_r2": m_p["R2"]
            }
            print(f"  {reg:<20} (N={len(group):,}, PFs={group['PF'].nunique()}): B_MAE={m_b['MAE']:.1f} (R2={m_b['R2']:.2f}) | P_MAE={m_p['MAE']:.1f} (R2={m_p['R2']:.2f})")
            
    # 10. Breakdown by Volume Tiers (Low, Mid, High)
    print("\n" + "="*80)
    print("PERFORMANCE BREAKDOWN BY TARGET VOLUME TIERS")
    print("="*80)
    q25 = float(np.percentile(y_true_bottiglie, 25))
    q75 = float(np.percentile(y_true_bottiglie, 75))
    
    low_mask = y_true_bottiglie <= q25
    mid_mask = (y_true_bottiglie > q25) & (y_true_bottiglie <= q75)
    high_mask = y_true_bottiglie > q75
    
    tier_metrics = {
        "Low_Volume (<= 25th pct, <= 182)": {
            "N": int(low_mask.sum()),
            "BOTTIGLIE_MAE": round(float(mean_absolute_error(y_true_bottiglie[low_mask], pred_bottiglie[low_mask])), 1),
            "PRESENZE_MAE": round(float(mean_absolute_error(y_true_presenze[low_mask], pred_presenze[low_mask])), 1)
        },
        "Mid_Volume (25th-75th pct, 183-763)": {
            "N": int(mid_mask.sum()),
            "BOTTIGLIE_MAE": round(float(mean_absolute_error(y_true_bottiglie[mid_mask], pred_bottiglie[mid_mask])), 1),
            "PRESENZE_MAE": round(float(mean_absolute_error(y_true_presenze[mid_mask], pred_presenze[mid_mask])), 1)
        },
        "High_Volume (> 75th pct, > 763)": {
            "N": int(high_mask.sum()),
            "BOTTIGLIE_MAE": round(float(mean_absolute_error(y_true_bottiglie[high_mask], pred_bottiglie[high_mask])), 1),
            "PRESENZE_MAE": round(float(mean_absolute_error(y_true_presenze[high_mask], pred_presenze[high_mask])), 1)
        }
    }
    for t_name, t_vals in tier_metrics.items():
        print(f"  {t_name}: B_MAE = {t_vals['BOTTIGLIE_MAE']}, P_MAE = {t_vals['PRESENZE_MAE']} (N={t_vals['N']:,})")
        
    # 11. Temporal Benchmark: 2026 Test Data Only
    print("\n" + "="*80)
    print("TEMPORAL BENCHMARK: 2026 TEST DATA ONLY")
    print("="*80)
    test_eval["dt"] = pd.to_datetime(test_eval["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    mask_2026 = test_eval["dt"].dt.year == 2026
    if mask_2026.any():
        t26_b = compute_metrics(y_true_bottiglie[mask_2026], pred_bottiglie[mask_2026])
        t26_p = compute_metrics(y_true_presenze[mask_2026], pred_presenze[mask_2026])
        print(f"  2026 Observations: {mask_2026.sum():,} rows")
        print(f"  BOTTIGLIE: MAE = {t26_b['MAE']:.1f}, RMSE = {t26_b['RMSE']:.1f}, R2 = {t26_b['R2']:.4f}")
        print(f"  PRESENZE:  MAE = {t26_p['MAE']:.1f}, RMSE = {t26_p['RMSE']:.1f}, R2 = {t26_p['R2']:.4f}")
    else:
        t26_b, t26_p = {}, {}
        
    # 12. Save full evaluation results
    full_results = {
        "test_split_info": {
            "test_rows": n_test_rows,
            "test_unique_pfs": n_test_pfs
        },
        "neural_network": {
            "bottiglie": nn_bottiglie_metrics,
            "presenze": nn_presenze_metrics
        },
        "baselines": {
            "global_train_mean": {"bottiglie": b1_b_metrics, "presenze": b1_p_metrics},
            "machine_type_mean": {"bottiglie": b2_b_metrics, "presenze": b2_p_metrics},
            "machine_regional_mean": {"bottiglie": b3_b_metrics, "presenze": b3_p_metrics}
        },
        "breakdown_by_machine": by_machine,
        "breakdown_by_region": by_region,
        "volume_tiers": tier_metrics,
        "temporal_benchmark_2026": {"bottiglie": t26_b, "presenze": t26_p}
    }
    
    metrics_path = f"{models_dir}/best_model/evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved evaluation metrics to {metrics_path}")
    
    return full_results

if __name__ == "__main__":
    evaluate_model()
