"""
Experiment Pipeline: Train Coripet RVM MultiTask Neural Network with Updated 2026 Data
and Compare Side-by-Side with Production Baseline.

Keeps `models/best_model` completely untouched and saves to `models/experiment_with_new_data`.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Ensure root directory in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

from src.model.preprocessing import RVMPreprocessor
from src.model.dataset import get_dataloaders
from src.model.network import MultiTaskRVMNet, MultiTaskHuberLoss
from src.model.evaluate import evaluate_model
from src.model.outlier_detector import RVMOutlierDetector
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

EXP_DIR = ROOT_DIR / "models" / "experiment_with_new_data"
PROD_DIR = ROOT_DIR / "models" / "best_model"
UPDATED_PARQUET = ROOT_DIR / "data" / "processed" / "modeling_daily_rvm_updated.parquet"


REQ_COLS = [
    "PF", "GIORNO", "BOTTIGLIE", "PRESENZE",
    "TipoMacchina", "Fornitore", "Provincia", "Regione", "CLIENTE",
    "nielsen_insegna", "nielsen_tipo_negozio", "nielsen_gruppo", "nielsen_area", "nielsen_match_status",
    "Latitudine", "Longitudine", "Cap",
    "population", "surface", "population_density", "PIL", "PIL_per_capita",
    "nielsen_superficie_vendita", "nielsen_numero_casse"
]


def export_updated_modeling_population() -> Path:
    """Combines certified baseline population with incremental 2026 PostgreSQL records with minimal memory overhead."""
    import gc
    print("=" * 80)
    print("STEP 1: Assembling Updated Modeling Population (Memory-Optimized)...")
    print("=" * 80)
    base_parquet = ROOT_DIR / "data" / "processed" / "modeling_daily_rvm.parquet"
    print(f"Loading certified baseline population from {base_parquet} (filtered columns)...")
    df_base = pd.read_parquet(base_parquet, columns=REQ_COLS)
    print(f"  -> Baseline rows: {len(df_base):,} (RAM: {df_base.memory_usage(deep=True).sum() / (1024**2):.1f} MB)")
    
    print("Querying new post-August 2026 records from PostgreSQL...")
    engine = create_engine(os.getenv("DATABASE_URL"))
    query = """
        SELECT pf, giorno, bottiglie, presenze,
               tipomacchina, fornitore, provincia, regione, cliente,
               nielsen_insegna, nielsen_tipo_negozio, nielsen_gruppo, nielsen_area, nielsen_match_status,
               latitudine, longitudine, cap,
               population, surface, population_density, pil, pil_per_capita,
               nielsen_superficie_vendita, nielsen_numero_casse
        FROM master_daily_rvm
        WHERE giorno_dt > '2026-08-31' AND modeling_eligible = 1;
    """
    with engine.connect() as conn:
        df_new = pd.read_sql(query, conn)
    print(f"  -> Incremental new rows: {len(df_new):,}")
    
    col_map = {
        "cliente": "CLIENTE", "giorno": "GIORNO", "bottiglie": "BOTTIGLIE", "presenze": "PRESENZE",
        "pf": "PF", "cap": "Cap", "provincia": "Provincia", "regione": "Regione",
        "latitudine": "Latitudine", "longitudine": "Longitudine",
        "tipomacchina": "TipoMacchina", "fornitore": "Fornitore",
        "pil": "PIL", "pil_per_capita": "PIL_per_capita"
    }
    df_new = df_new.rename(columns=col_map)
    
    # Harmonize dtypes
    for col in REQ_COLS:
        if col in df_new.columns:
            try:
                df_new[col] = df_new[col].astype(df_base[col].dtype)
            except Exception:
                pass
                
    df_updated = pd.concat([df_base, df_new[REQ_COLS]], ignore_index=True)
    del df_base, df_new
    gc.collect()
    
    UPDATED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_updated.to_parquet(UPDATED_PARQUET, index=False, engine="pyarrow")
    
    print(f"  -> Combined total: {len(df_updated):,} rows across {df_updated['PF'].nunique():,} unique PFs.")
    print(f"  -> Saved to {UPDATED_PARQUET} ({UPDATED_PARQUET.stat().st_size / (1024*1024):.1f} MB)")
    del df_updated
    gc.collect()
    return UPDATED_PARQUET


def split_with_fixed_test_set(df: pd.DataFrame, fixed_test_parquet: Path, seed: int = 42):
    """
    Keeps the test holdout PFs identical to baseline evaluation for 100% fair comparison.
    All new data for training PFs enters the training set.
    """
    test_ref = pd.read_parquet(fixed_test_parquet, columns=["PF"])
    test_pfs = set(test_ref["PF"].unique())
    
    # Candidate PFs from updated df
    all_pfs = set(df["PF"].unique())
    dev_pfs = sorted(list(all_pfs - test_pfs))
    
    # 20% of dev as validation
    rng = np.random.RandomState(seed)
    shuffled_dev = rng.permutation(dev_pfs)
    n_val = int(np.round(0.20 * len(dev_pfs)))
    val_pfs = set(shuffled_dev[:n_val])
    train_pfs = set(shuffled_dev[n_val:])
    
    train_df = df[df["PF"].isin(train_pfs)].copy()
    val_df = df[df["PF"].isin(val_pfs)].copy()
    test_df = df[df["PF"].isin(test_pfs)].copy()
    
    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_pfs": len(train_pfs),
        "val_pfs": len(val_pfs),
        "test_pfs": len(test_pfs),
        "total_rows": len(df),
        "total_pfs": len(all_pfs),
        "seed": seed
    }
    return train_df, val_df, test_df, split_info


def train_experiment():
    import gc
    parquet_path = export_updated_modeling_population()
    fixed_test_parquet = ROOT_DIR / "models" / "preprocessing" / "test_holdout.parquet"

    print("\n" + "=" * 80)
    print("STEP 2: Training Neural Network on Updated Population...")
    print("=" * 80)

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Hardware Device: {device}")

    df = pd.read_parquet(parquet_path)
    # Prepare directories
    prep_dir = EXP_DIR / "preprocessing"
    best_dir = EXP_DIR / "best_model"
    prep_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    test_ref = pd.read_parquet(fixed_test_parquet, columns=["PF"])
    test_pfs = set(test_ref["PF"].unique())
    all_pfs = set(df["PF"].unique())
    dev_pfs = sorted(list(all_pfs - test_pfs))

    rng = np.random.RandomState(random_seed)
    shuffled_dev = rng.permutation(dev_pfs)
    n_val = int(np.round(0.20 * len(dev_pfs)))
    val_pfs = set(shuffled_dev[:n_val])
    train_pfs = set(shuffled_dev[n_val:])

    split_info = {
        "train_rows": int(df["PF"].isin(train_pfs).sum()),
        "val_rows": int(df["PF"].isin(val_pfs).sum()),
        "test_rows": int(df["PF"].isin(test_pfs).sum()),
        "train_pfs": len(train_pfs),
        "val_pfs": len(val_pfs),
        "test_pfs": len(test_pfs),
        "total_rows": len(df),
        "total_pfs": len(all_pfs),
        "seed": random_seed
    }
    with open(prep_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"Train rows: {split_info['train_rows']:,} ({split_info['train_pfs']} PFs)")
    print(f"Val rows:   {split_info['val_rows']:,} ({split_info['val_pfs']} PFs)")
    print(f"Test rows:  {split_info['test_rows']:,} ({split_info['test_pfs']} PFs - untouched holdout)")

    # 0. Outlier and Defect Detection Audit
    print("\nExecuting Pre-Training Outlier & Defect Detection Audit...")
    detector = RVMOutlierDetector()
    fleet_audit = detector.audit_fleet(df)
    report_csv = ROOT_DIR / "data" / "reports" / "defected_and_outlier_machines.csv"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fleet_audit.to_csv(report_csv, index=False)
    fleet_audit.to_json(best_dir / "outlier_machines_report.json", orient="records", indent=2)
    print(f"  -> Outlier audit complete: {len(fleet_audit)} machines evaluated.")
    for cat, cnt in fleet_audit["Anomaly_Classification"].value_counts().items():
        print(f"     * {cat:<26}: {cnt:>4} machines")

    # 1. Fit & transform train_df & calculate sample weights
    print("\nFitting Preprocessor and transforming training set...")
    train_df = df[df["PF"].isin(train_pfs)].copy()
    train_weights, _ = detector.calculate_training_weights(train_df)
    n_masked = int((train_weights == 0.0).sum())
    print(f"  -> Loss Masking: {n_masked:,} breakdown zero-days masked from Huber loss.")

    preprocessor = RVMPreprocessor(random_seed=random_seed)
    preprocessor.fit(train_df)
    preprocessor.save(str(prep_dir / "preprocessor.joblib"))
    train_arrays = preprocessor.transform(train_df, nielsen_dropout=0.25)
    del train_df
    gc.collect()

    # 2. Transform val_df
    print("Transforming validation set...")
    val_df = df[df["PF"].isin(val_pfs)].copy()
    val_arrays = preprocessor.transform(val_df, nielsen_dropout=0.0)
    del val_df
    gc.collect()

    # 3. Transform test_df
    print("Transforming test set...")
    test_df = df[df["PF"].isin(test_pfs)].copy()
    del df
    gc.collect()
    test_df.to_parquet(prep_dir / "test_holdout.parquet", index=False)
    test_arrays = preprocessor.transform(test_df, nielsen_dropout=0.0)
    del test_df
    gc.collect()

    train_loader, val_loader, test_loader = get_dataloaders(
        train_arrays, val_arrays, test_arrays,
        train_weights=train_weights,
        batch_size=4096
    )

    # Initialize model
    model = MultiTaskRVMNet(
        num_numeric_features=len(preprocessor.num_features),
        cat_cardinalities=preprocessor.get_cat_cardinalities(),
        cat_features=preprocessor.cat_features,
        hidden_dims=(256, 128, 64),
        head_dim=32,
        dropout=0.2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = MultiTaskHuberLoss(delta=1.0, weight_bottiglie=1.0, weight_presenze=1.0)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    start_time = time.time()

    print("\nStarting Training Loop (Max 10 Epochs, Patience 4)...")
    for epoch in range(1, 11):
        ep_start = time.time()
        model.train()
        t_loss = t_loss_b = t_loss_p = 0.0
        n_b = 0

        for b_num, b_cat, yb, yp, w in train_loader:
            b_num, b_cat = b_num.to(device), b_cat.to(device)
            yb, yp = yb.to(device), yp.to(device)
            w = w.to(device)

            optimizer.zero_grad()
            pb, pp = model(b_num, b_cat)
            loss, lb, lp = criterion(pb, pp, yb, yp, sample_weights=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            t_loss += loss.item()
            t_loss_b += lb.item()
            t_loss_p += lp.item()
            n_b += 1

        t_loss /= n_b
        t_loss_b /= n_b
        t_loss_p /= n_b

        # Validation
        model.eval()
        v_loss = v_loss_b = v_loss_p = 0.0
        n_vb = 0
        with torch.no_grad():
            for b_num, b_cat, yb, yp, _ in val_loader:
                b_num, b_cat = b_num.to(device), b_cat.to(device)
                yb, yp = yb.to(device), yp.to(device)
                pb, pp = model(b_num, b_cat)
                loss, lb, lp = criterion(pb, pp, yb, yp)
                v_loss += loss.item()
                v_loss_b += lb.item()
                v_loss_p += lp.item()
                n_vb += 1

        v_loss /= n_vb
        v_loss_b /= n_vb
        v_loss_p /= n_vb
        scheduler.step(v_loss)

        ep_duration = time.time() - ep_start
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:2d}/10 | Train: {t_loss:.4f} (B:{t_loss_b:.4f}, P:{t_loss_p:.4f}) | Val: {v_loss:.4f} (B:{v_loss_b:.4f}, P:{v_loss_p:.4f}) | {ep_duration:.1f}s | lr={curr_lr:.5f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(t_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_loss_b": round(v_loss_b, 4),
            "val_loss_p": round(v_loss_p, 4),
            "duration": round(ep_duration, 1)
        })

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_dir / "model.pt")
            print(f"  --> Saved new best experiment checkpoint (val_loss={v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 4:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    train_time_min = round((time.time() - start_time) / 60.0, 2)
    print(f"\nTraining completed in {train_time_min} min. Best Val Loss: {best_val_loss:.4f}")

    with open(best_dir / "training_history.json", "w") as f:
        json.dump({"history": history, "best_val_loss": best_val_loss, "train_time_min": train_time_min}, f, indent=2)

    # -------------------------------------------------------------------------
    # STEP 3: Evaluation & Side-by-Side Comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 3: Evaluating Retrained Experiment vs Baseline Production Model...")
    print("=" * 80)

    # Evaluate new experiment
    print("\n[EVALUATING EXPERIMENT MODEL]...")
    exp_metrics = evaluate_model(
        models_dir=str(EXP_DIR),
        test_parquet=str(fixed_test_parquet),
        training_parquet=str(parquet_path)
    )

    # Load baseline metrics
    with open(PROD_DIR / "evaluation_metrics.json") as f:
        base_metrics = json.load(f)

    # Print Side-by-Side Comparison
    print("\n" + "=" * 80)
    print("               FINAL IMPACT COMPARISON: BASELINE vs RETRAINED MODEL")
    print("=" * 80)
    print(f" {'Metric':<25} | {'Baseline Model':<18} | {'Retrained (+New Data)':<24} | {'Difference'}")
    print(" " + "-" * 78)

    bb = base_metrics["neural_network"]["bottiglie"]
    eb = exp_metrics["neural_network"]["bottiglie"]
    print(f" {'BOTTIGLIE MAE':<25} | {bb['MAE']:<18.2f} | {eb['MAE']:<24.2f} | {eb['MAE'] - bb['MAE']:+.2f}")
    print(f" {'BOTTIGLIE RMSE':<25} | {bb['RMSE']:<18.2f} | {eb['RMSE']:<24.2f} | {eb['RMSE'] - bb['RMSE']:+.2f}")
    print(f" {'BOTTIGLIE R2':<25} | {bb['R2']:<18.4f} | {eb['R2']:<24.4f} | {eb['R2'] - bb['R2']:+.4f}")
    print(f" {'BOTTIGLIE MedAE':<25} | {bb['MedAE']:<18.2f} | {eb['MedAE']:<24.2f} | {eb['MedAE'] - bb['MedAE']:+.2f}")

    bp = base_metrics["neural_network"]["presenze"]
    ep = exp_metrics["neural_network"]["presenze"]
    print(" " + "-" * 78)
    print(f" {'PRESENZE MAE':<25} | {bp['MAE']:<18.2f} | {ep['MAE']:<24.2f} | {ep['MAE'] - bp['MAE']:+.2f}")
    print(f" {'PRESENZE RMSE':<25} | {bp['RMSE']:<18.2f} | {ep['RMSE']:<24.2f} | {ep['RMSE'] - bp['RMSE']:+.2f}")
    print(f" {'PRESENZE R2':<25} | {bp['R2']:<18.4f} | {ep['R2']:<24.4f} | {ep['R2'] - bp['R2']:+.4f}")
    print(f" {'PRESENZE MedAE':<25} | {bp['MedAE']:<18.2f} | {ep['MedAE']:<24.2f} | {ep['MedAE'] - bp['MedAE']:+.2f}")
    print("=" * 80)


if __name__ == "__main__":
    train_experiment()
