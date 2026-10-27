"""
Monthly Automated Retraining & Self-Validating Pipeline for Coripet RVM Predictor.
Runs on the 1st of every month at 05:00 AM (after the 04:00 AM daily ETL).

Features:
- Incremental data ingestion from PostgreSQL master_daily_rvm
- Memory-efficient float32 / int64 tensor transformations
- Multi-Task Neural Network training with early stopping
- Automated Quality Gate verification against untouched test holdout
- Safe zero-downtime production promotion with historical archives
"""

import os
import sys
import json
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.model.preprocessing import RVMPreprocessor
from src.model.dataset import get_dataloaders
from src.model.network import MultiTaskRVMNet, MultiTaskHuberLoss
from src.model.evaluate import evaluate_model
from src.model.outlier_detector import RVMOutlierDetector
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Paths
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "monthly_retrain.log"

PROD_DIR = ROOT_DIR / "models" / "best_model"
PREP_DIR = ROOT_DIR / "models" / "preprocessing"
EXP_DIR = ROOT_DIR / "models" / "experiment_with_new_data"
ARCHIVE_ROOT = ROOT_DIR / "models" / "archive"
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT_DIR / "data" / "processed"
BASE_PARQUET = DATA_DIR / "modeling_daily_rvm.parquet"
UPDATED_PARQUET = DATA_DIR / "modeling_daily_rvm_updated.parquet"
FIXED_TEST_PARQUET = PREP_DIR / "test_holdout.parquet"

# Setup Logging
logger = logging.getLogger("MonthlyRetrain")
logger.setLevel(logging.INFO)
# Clear old handlers
logger.handlers = []

fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)

formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

REQ_COLS = [
    "PF", "GIORNO", "BOTTIGLIE", "PRESENZE",
    "TipoMacchina", "Fornitore", "Provincia", "Regione", "CLIENTE",
    "nielsen_insegna", "nielsen_tipo_negozio", "nielsen_gruppo", "nielsen_area", "nielsen_match_status",
    "Latitudine", "Longitudine", "Cap",
    "population", "surface", "population_density", "PIL", "PIL_per_capita",
    "nielsen_superficie_vendita", "nielsen_numero_casse"
]


def get_db_engine():
    db_pass = os.getenv("DB_PASSWORD") or os.getenv("DB_PASS")
    if db_pass:
        from sqlalchemy.engine import URL
        db_user = os.getenv("DB_USER", "waqas")
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_port = int(os.getenv("DB_PORT", "3306"))
        db_name = os.getenv("DB_NAME", "coripet_rvm")
        url_obj = URL.create("mysql+pymysql", username=db_user, password=db_pass, host=db_host, port=db_port, database=db_name)
        return create_engine(url_obj)

    pg_url = os.getenv("DATABASE_URL", "").strip()
    if not pg_url:
        raise ValueError("DATABASE_URL or DB_PASSWORD environment variable is not set.")
    if pg_url.startswith("postgresql://") and "+" not in pg_url.split("://")[0]:
        pg_url = pg_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    elif pg_url.startswith("postgres://"):
        pg_url = pg_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif pg_url.startswith("mysql://") and "+" not in pg_url.split("://")[0]:
        pg_url = pg_url.replace("mysql://", "mysql+pymysql://", 1)
    return create_engine(pg_url)


def assemble_updated_dataset() -> Path:
    """Combines certified base population with incremental mature PostgreSQL records."""
    import gc
    logger.info("=" * 80)
    logger.info("STEP 1: Assembling Updated Modeling Dataset from PostgreSQL...")
    logger.info("=" * 80)

    logger.info(f"Loading base dataset from {BASE_PARQUET}...")
    df_base = pd.read_parquet(BASE_PARQUET, columns=REQ_COLS)
    logger.info(f"  -> Base rows: {len(df_base):,}")

    engine = get_db_engine()
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
    logger.info(f"  -> Incremental new records from PostgreSQL: {len(df_new):,}")

    col_map = {
        "cliente": "CLIENTE", "giorno": "GIORNO", "bottiglie": "BOTTIGLIE", "presenze": "PRESENZE",
        "pf": "PF", "cap": "Cap", "provincia": "Provincia", "regione": "Regione",
        "latitudine": "Latitudine", "longitudine": "Longitudine",
        "tipomacchina": "TipoMacchina", "fornitore": "Fornitore",
        "pil": "PIL", "pil_per_capita": "PIL_per_capita"
    }
    df_new = df_new.rename(columns=col_map)

    for col in REQ_COLS:
        if col in df_new.columns:
            try:
                df_new[col] = df_new[col].astype(df_base[col].dtype)
            except Exception:
                pass

    df_updated = pd.concat([df_base, df_new[REQ_COLS]], ignore_index=True)
    del df_base, df_new
    gc.collect()

    df_updated.to_parquet(UPDATED_PARQUET, index=False, engine="pyarrow")
    logger.info(f"  -> Successfully saved updated dataset: {len(df_updated):,} rows across {df_updated['PF'].nunique():,} PFs.")
    del df_updated
    gc.collect()
    return UPDATED_PARQUET


def train_candidate_model(dataset_path: Path) -> dict:
    """Trains a candidate neural network on updated population."""
    import gc
    logger.info("=" * 80)
    logger.info("STEP 2: Training Candidate Multi-Task Neural Network...")
    logger.info("=" * 80)

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Hardware Compute Device: {device}")

    df = pd.read_parquet(dataset_path)

    cand_prep_dir = EXP_DIR / "preprocessing"
    cand_best_dir = EXP_DIR / "best_model"
    cand_prep_dir.mkdir(parents=True, exist_ok=True)
    cand_best_dir.mkdir(parents=True, exist_ok=True)

    test_ref = pd.read_parquet(FIXED_TEST_PARQUET, columns=["PF"])
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
    with open(cand_prep_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    logger.info(f"Train set: {split_info['train_rows']:,} rows ({split_info['train_pfs']} PFs)")
    logger.info(f"Val set:   {split_info['val_rows']:,} rows ({split_info['val_pfs']} PFs)")
    logger.info(f"Test set:  {split_info['test_rows']:,} rows ({split_info['test_pfs']} PFs - untouched holdout)")

    # 0. Outlier and Defect Detection Audit across full dataset
    logger.info("Executing Pre-Training Outlier & Defect Detection Audit...")
    detector = RVMOutlierDetector()
    fleet_audit = detector.audit_fleet(df)
    report_csv = ROOT_DIR / "data" / "reports" / "defected_and_outlier_machines.csv"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fleet_audit.to_csv(report_csv, index=False)
    fleet_audit.to_json(cand_best_dir / "outlier_machines_report.json", orient="records", indent=2)
    logger.info(f"  -> Outlier audit completed for {len(fleet_audit)} machines.")
    for cat, cnt in fleet_audit["Anomaly_Classification"].value_counts().items():
        logger.info(f"     * {cat:<26}: {cnt:>4} machines")

    # 1. Fit Preprocessor on train_df & calculate sample weights
    logger.info("Fitting candidate preprocessor on training partition...")
    train_df = df[df["PF"].isin(train_pfs)].copy()
    train_weights, _ = detector.calculate_training_weights(train_df)
    n_masked = int((train_weights == 0.0).sum())
    logger.info(f"  -> Loss Masking: {n_masked:,} breakdown zero-days masked from Huber loss.")

    preprocessor = RVMPreprocessor(random_seed=random_seed)
    preprocessor.fit(train_df)
    preprocessor.save(str(cand_prep_dir / "preprocessor.joblib"))
    train_arrays = preprocessor.transform(train_df, nielsen_dropout=0.25)
    del train_df
    gc.collect()

    # 2. Transform val_df
    val_df = df[df["PF"].isin(val_pfs)].copy()
    val_arrays = preprocessor.transform(val_df, nielsen_dropout=0.0)
    del val_df
    gc.collect()

    # 3. Transform test_df
    test_df = df[df["PF"].isin(test_pfs)].copy()
    del df
    gc.collect()
    test_df.to_parquet(cand_prep_dir / "test_holdout.parquet", index=False)
    test_arrays = preprocessor.transform(test_df, nielsen_dropout=0.0)
    del test_df
    gc.collect()

    train_loader, val_loader, test_loader = get_dataloaders(
        train_arrays, val_arrays, test_arrays,
        train_weights=train_weights,
        batch_size=4096
    )

    # Initialize neural network
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
    start_time = time.time()
    history = []

    logger.info("Beginning training loop (Max 10 Epochs, Early Stopping Patience 3)...")
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

        logger.info(f"Epoch {epoch:2d}/10 | Train: {t_loss:.4f} (B:{t_loss_b:.4f}, P:{t_loss_p:.4f}) | Val: {v_loss:.4f} (B:{v_loss_b:.4f}, P:{v_loss_p:.4f}) | {ep_duration:.1f}s | lr={curr_lr:.5f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(t_loss, 4),
            "val_loss": round(v_loss, 4),
            "duration": round(ep_duration, 1)
        })

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_counter = 0
            torch.save(model.state_dict(), cand_best_dir / "model.pt")
            logger.info(f"  -> Checkpoint saved (val_loss={v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= 3:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    train_time_min = round((time.time() - start_time) / 60.0, 2)
    logger.info(f"Training completed in {train_time_min} min. Best Val Loss: {best_val_loss:.4f}")

    with open(cand_best_dir / "training_history.json", "w") as f:
        json.dump({"history": history, "best_val_loss": best_val_loss, "train_time_min": train_time_min}, f, indent=2)

    return {"best_val_loss": best_val_loss, "train_time_min": train_time_min}


def evaluate_and_verify_quality_gate(dataset_path: Path) -> bool:
    """Evaluates candidate model against current production and checks Quality Gate."""
    logger.info("=" * 80)
    logger.info("STEP 3: Evaluating Candidate Model Against Production Baseline...")
    logger.info("=" * 80)

    # Evaluate candidate
    cand_metrics = evaluate_model(
        models_dir=str(EXP_DIR),
        test_parquet=str(FIXED_TEST_PARQUET),
        training_parquet=str(dataset_path)
    )

    # Load current production metrics
    prod_metrics_file = PROD_DIR / "evaluation_metrics.json"
    with open(prod_metrics_file) as f:
        prod_metrics = json.load(f)

    cb = cand_metrics["neural_network"]["bottiglie"]
    pb = prod_metrics["neural_network"]["bottiglie"]

    cp = cand_metrics["neural_network"]["presenze"]
    pp = prod_metrics["neural_network"]["presenze"]

    logger.info("\n" + "=" * 80)
    logger.info("                  CANDIDATE vs PRODUCTION BENCHMARK")
    logger.info("=" * 80)
    logger.info(f" {'Metric':<25} | {'Production Model':<18} | {'Candidate Model':<18} | {'Difference'}")
    logger.info(" " + "-" * 78)
    logger.info(f" {'BOTTIGLIE MAE':<25} | {pb['MAE']:<18.2f} | {cb['MAE']:<18.2f} | {cb['MAE'] - pb['MAE']:+.2f}")
    logger.info(f" {'BOTTIGLIE RMSE':<25} | {pb['RMSE']:<18.2f} | {cb['RMSE']:<18.2f} | {cb['RMSE'] - pb['RMSE']:+.2f}")
    logger.info(f" {'BOTTIGLIE R2':<25} | {pb['R2']:<18.4f} | {cb['R2']:<18.4f} | {cb['R2'] - pb['R2']:+.4f}")
    logger.info(f" {'BOTTIGLIE MedAE':<25} | {pb['MedAE']:<18.2f} | {cb['MedAE']:<18.2f} | {cb['MedAE'] - pb['MedAE']:+.2f}")
    logger.info(" " + "-" * 78)
    logger.info(f" {'PRESENZE MAE':<25} | {pp['MAE']:<18.2f} | {cp['MAE']:<18.2f} | {cp['MAE'] - pp['MAE']:+.2f}")
    logger.info(f" {'PRESENZE R2':<25} | {pp['R2']:<18.4f} | {cp['R2']:<18.4f} | {cp['R2'] - pp['R2']:+.4f}")
    logger.info("=" * 80)

    # QUALITY GATE RULES:
    # 1. Candidate R2 for BOTTIGLIE must not regress by more than 0.03
    # 2. Candidate MAE for BOTTIGLIE must not increase by more than 5%
    # 3. Candidate MAE for PRESENZE must not increase by more than 1.5 visits
    r2_threshold = pb["R2"] - 0.03
    mae_threshold = pb["MAE"] * 1.05
    presenze_threshold = pp["MAE"] + 1.5

    gate_r2 = cb["R2"] >= r2_threshold
    gate_mae = cb["MAE"] <= mae_threshold
    gate_p = cp["MAE"] <= presenze_threshold

    passed = gate_r2 and gate_mae and gate_p

    logger.info(f"Quality Gate Checks:")
    logger.info(f"  [1] BOTTIGLIE R2 >= {r2_threshold:.4f}: {'PASS' if gate_r2 else 'FAIL'} (Actual: {cb['R2']:.4f})")
    logger.info(f"  [2] BOTTIGLIE MAE <= {mae_threshold:.2f}: {'PASS' if gate_mae else 'FAIL'} (Actual: {cb['MAE']:.2f})")
    logger.info(f"  [3] PRESENZE MAE <= {presenze_threshold:.2f}: {'PASS' if gate_p else 'FAIL'} (Actual: {cp['MAE']:.2f})")

    if passed:
        logger.info("--> QUALITY GATE PASSED: Candidate meets all accuracy standards.")
    else:
        logger.warning("--> QUALITY GATE FAILED: Candidate regressed compared to production. Promotion blocked!")

    return passed


def promote_candidate_to_production():
    """Archives previous production and promotes candidate model."""
    logger.info("=" * 80)
    logger.info("STEP 4: Promoting Candidate Model to Production...")
    logger.info("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dest = ARCHIVE_ROOT / f"model_{timestamp}"
    archive_dest.mkdir(parents=True, exist_ok=True)

    # 1. Archive current production
    shutil.copytree(PROD_DIR, archive_dest / "best_model", dirs_exist_ok=True)
    shutil.copytree(PREP_DIR, archive_dest / "preprocessing", dirs_exist_ok=True)
    logger.info(f"  -> Prior production model archived to: {archive_dest}")

    # 2. Promote candidate
    shutil.copy2(EXP_DIR / "best_model" / "model.pt", PROD_DIR / "model.pt")
    shutil.copy2(EXP_DIR / "best_model" / "evaluation_metrics.json", PROD_DIR / "evaluation_metrics.json")
    shutil.copy2(EXP_DIR / "best_model" / "training_history.json", PROD_DIR / "training_history.json")

    shutil.copy2(EXP_DIR / "preprocessing" / "preprocessor.joblib", PREP_DIR / "preprocessor.joblib")
    shutil.copy2(EXP_DIR / "preprocessing" / "split_info.json", PREP_DIR / "split_info.json")

    if (EXP_DIR / "best_model" / "outlier_machines_report.json").exists():
        shutil.copy2(EXP_DIR / "best_model" / "outlier_machines_report.json", PROD_DIR / "outlier_machines_report.json")
        logger.info("  -> Promoted outlier_machines_report.json to production models/best_model/")

    if UPDATED_PARQUET.exists():
        shutil.copy2(UPDATED_PARQUET, BASE_PARQUET)
        logger.info("  -> Updated modeling population promoted to data/processed/modeling_daily_rvm.parquet")

    logger.info("--> PROMOTION COMPLETE: Production model successfully updated with new monthly weights.")


def main():
    start_total = time.time()
    logger.info("=" * 80)
    logger.info(f"CORIPET RVM PREDICTOR — MONTHLY RETRAINING PIPELINE START [{datetime.now().isoformat()}]")
    logger.info("=" * 80)

    try:
        dataset_path = assemble_updated_dataset()
        train_candidate_model(dataset_path)
        gate_passed = evaluate_and_verify_quality_gate(dataset_path)

        if gate_passed:
            promote_candidate_to_production()
            logger.info("Monthly Retraining Cycle Completed: Model Updated Successfully.")
        else:
            logger.warning("Monthly Retraining Cycle Completed: Candidate failed Quality Gate; production kept intact.")

    except Exception as e:
        logger.exception(f"FATAL ERROR in monthly retraining pipeline: {e}")
        sys.exit(1)

    total_min = round((time.time() - start_total) / 60.0, 2)
    logger.info(f"Total Pipeline Runtime: {total_min} minutes.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
