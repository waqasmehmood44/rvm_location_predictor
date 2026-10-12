"""
Reproducible Stacking Ensemble benchmark pipeline:
Trains LightGBM regressor on training partition, tunes blend weights on validation partition,
and evaluates Stacking Ensemble (LightGBM + MultiTaskRVMNet) on the untouched 331-store test holdout.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

from src.model.preprocessing import RVMPreprocessor, split_pfs
from src.model.network import MultiTaskRVMNet

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MedAE": round(medae, 2),
    }

def run_ensemble_training_and_eval(
    data_path: str = "data/processed/modeling_daily_rvm.parquet",
    models_dir: str = "models",
    random_seed: int = 42
):
    print("=" * 80)
    print("LIGHTGBM + NEURAL NETWORK STACKING ENSEMBLE BENCHMARK")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    preprocessor = RVMPreprocessor.load(f"{models_dir}/preprocessing/preprocessor.joblib")
    
    full_df = pd.read_parquet(data_path)
    test_df = pd.read_parquet(f"{models_dir}/preprocessing/test_holdout.parquet")
    train_df, val_df, _, _ = split_pfs(full_df, seed=random_seed)
    
    print("Transforming feature matrices...")
    X_num_tr, X_cat_tr, y_b_tr, y_p_tr = preprocessor.transform(train_df, nielsen_dropout=0.1)
    X_num_val, X_cat_val, y_b_val, y_p_val = preprocessor.transform(val_df, nielsen_dropout=0.0)
    X_num_te, X_cat_te, y_b_te, y_p_te = preprocessor.transform(test_df, nielsen_dropout=0.0)
    
    X_train = np.hstack([X_num_tr, X_cat_tr])
    X_val = np.hstack([X_num_val, X_cat_val])
    X_test = np.hstack([X_num_te, X_cat_te])
    
    n_num = X_num_tr.shape[1]
    cat_indices = list(range(n_num, X_train.shape[1]))
    
    y_true_b_test = test_df["BOTTIGLIE"].values.astype(float)
    y_true_p_test = test_df["PRESENZE"].values.astype(float)
    y_true_b_val = val_df["BOTTIGLIE"].values.astype(float)
    y_true_p_val = val_df["PRESENZE"].values.astype(float)
    
    # 1. Production Neural Net Evaluation
    model = MultiTaskRVMNet(
        num_numeric_features=len(preprocessor.num_features),
        cat_cardinalities=preprocessor.get_cat_cardinalities(),
        cat_features=preprocessor.cat_features,
        hidden_dims=(256, 128, 64),
        head_dim=32,
        dropout=0.2
    ).to(device)
    model.load_state_dict(torch.load(f"{models_dir}/best_model/model.pt", map_location=device, weights_only=True))
    model.eval()
    
    batch_size = 4096
    nn_preds_b, nn_preds_p = [], []
    with torch.no_grad():
        for i in range(0, len(test_df), batch_size):
            bn = torch.tensor(X_num_te[i:i+batch_size], dtype=torch.float32, device=device)
            bc = torch.tensor(X_cat_te[i:i+batch_size], dtype=torch.long, device=device)
            pb, pp = model(bn, bc)
            nn_preds_b.append(pb.cpu().numpy())
            nn_preds_p.append(pp.cpu().numpy())
            
    nn_pred_log_b = np.vstack(nn_preds_b).squeeze()
    nn_pred_log_p = np.vstack(nn_preds_p).squeeze()
    nn_pred_b = np.clip(np.expm1(nn_pred_log_b), 0.0, None)
    nn_pred_p = np.clip(np.expm1(nn_pred_log_p), 0.0, None)
    
    nn_b_metrics = compute_metrics(y_true_b_test, nn_pred_b)
    nn_p_metrics = compute_metrics(y_true_p_test, nn_pred_p)
    
    # 2. Validation predictions for Neural Net
    nn_val_b, nn_val_p = [], []
    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            bn = torch.tensor(X_num_val[i:i+batch_size], dtype=torch.float32, device=device)
            bc = torch.tensor(X_cat_val[i:i+batch_size], dtype=torch.long, device=device)
            pb, pp = model(bn, bc)
            nn_val_b.append(pb.cpu().numpy())
            nn_val_p.append(pp.cpu().numpy())
    nn_val_pred_b = np.clip(np.expm1(np.vstack(nn_val_b).squeeze()), 0.0, None)
    nn_val_pred_p = np.clip(np.expm1(np.vstack(nn_val_p).squeeze()), 0.0, None)
    
    # 3. Train LightGBM Regressors
    lgb_params = {
        "objective": "huber",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": random_seed
    }
    
    train_data_b = lgb.Dataset(X_train, label=y_b_tr, categorical_feature=cat_indices, free_raw_data=False)
    val_data_b = lgb.Dataset(X_val, label=y_b_val, reference=train_data_b, categorical_feature=cat_indices, free_raw_data=False)
    lgb_b = lgb.train(
        lgb_params,
        train_data_b,
        num_boost_round=600,
        valid_sets=[val_data_b],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    train_data_p = lgb.Dataset(X_train, label=y_p_tr, categorical_feature=cat_indices, free_raw_data=False)
    val_data_p = lgb.Dataset(X_val, label=y_p_val, reference=train_data_p, categorical_feature=cat_indices, free_raw_data=False)
    lgb_p = lgb.train(
        lgb_params,
        train_data_p,
        num_boost_round=600,
        valid_sets=[val_data_p],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # 4. Blend Weight Optimization
    lgb_val_pred_b = np.clip(np.expm1(lgb_b.predict(X_val)), 0.0, None)
    lgb_val_pred_p = np.clip(np.expm1(lgb_p.predict(X_val)), 0.0, None)
    
    weights = np.linspace(0.0, 1.0, 21)
    best_w_b, best_val_mae_b = 0.0, 999999.0
    for w in weights:
        blend = w * lgb_val_pred_b + (1.0 - w) * nn_val_pred_b
        mae = mean_absolute_error(y_true_b_val, blend)
        if mae < best_val_mae_b:
            best_val_mae_b = mae
            best_w_b = w
            
    best_w_p, best_val_mae_p = 0.0, 999999.0
    for w in weights:
        blend = w * lgb_val_pred_p + (1.0 - w) * nn_val_pred_p
        mae = mean_absolute_error(y_true_p_val, blend)
        if mae < best_val_mae_p:
            best_val_mae_p = mae
            best_w_p = w
            
    # 5. Evaluate on Test Holdout
    lgb_pred_b = np.clip(np.expm1(lgb_b.predict(X_test)), 0.0, None)
    lgb_pred_p = np.clip(np.expm1(lgb_p.predict(X_test)), 0.0, None)
    
    ensemble_pred_b = best_w_b * lgb_pred_b + (1.0 - best_w_b) * nn_pred_b
    ensemble_pred_p = best_w_p * lgb_pred_p + (1.0 - best_w_p) * nn_pred_p
    
    ens_b_metrics = compute_metrics(y_true_b_test, ensemble_pred_b)
    ens_p_metrics = compute_metrics(y_true_p_test, ensemble_pred_p)
    
    print("\n--- Benchmark Summary ---")
    print(f"Production NN:      B_MAE = {nn_b_metrics['MAE']}, RMSE = {nn_b_metrics['RMSE']}, R2 = {nn_b_metrics['R2']}")
    print(f"Stacking Ensemble:  B_MAE = {ens_b_metrics['MAE']}, RMSE = {ens_b_metrics['RMSE']}, R2 = {ens_b_metrics['R2']}")
    print(f"BOTTIGLIE Delta:    {ens_b_metrics['MAE'] - nn_b_metrics['MAE']:.2f} b/day")
    print(f"PRESENZE Delta:     {ens_p_metrics['MAE'] - nn_p_metrics['MAE']:.2f} users/day")
    
    if ens_b_metrics['MAE'] < nn_b_metrics['MAE']:
        os.makedirs(f"{models_dir}/ensemble", exist_ok=True)
        lgb_b.save_model(f"{models_dir}/ensemble/lgb_bottiglie.txt")
        lgb_p.save_model(f"{models_dir}/ensemble/lgb_presenze.txt")
        config = {
            "blend_weight_bottiglie_lgb": float(best_w_b),
            "blend_weight_bottiglie_nn": float(1.0 - best_w_b),
            "blend_weight_presenze_lgb": float(best_w_p),
            "blend_weight_presenze_nn": float(1.0 - best_w_p),
            "metrics": {
                "bottiglie": ens_b_metrics,
                "presenze": ens_p_metrics
            }
        }
        with open(f"{models_dir}/ensemble/ensemble_config.json", "w") as f:
            json.dump(config, f, indent=2)
        print("Ensemble checkpoints updated in models/ensemble/")

if __name__ == "__main__":
    run_ensemble_training_and_eval()
