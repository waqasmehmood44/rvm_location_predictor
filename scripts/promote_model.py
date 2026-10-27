import shutil
import json
from pathlib import Path
import torch
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]

PROD_DIR = ROOT_DIR / "models" / "best_model"
PREP_DIR = ROOT_DIR / "models" / "preprocessing"

EXP_DIR = ROOT_DIR / "models" / "experiment_with_new_data"
EXP_BEST = EXP_DIR / "best_model"
EXP_PREP = EXP_DIR / "preprocessing"

ARCHIVE_BEST = ROOT_DIR / "models" / "best_model_baseline_archive"
ARCHIVE_PREP = ROOT_DIR / "models" / "preprocessing_baseline_archive"

DATA_DIR = ROOT_DIR / "data" / "processed"
UPDATED_PARQUET = DATA_DIR / "modeling_daily_rvm_updated.parquet"
BASE_PARQUET = DATA_DIR / "modeling_daily_rvm.parquet"
ARCHIVE_PARQUET = DATA_DIR / "modeling_daily_rvm_baseline_archive.parquet"

print("=" * 80)
print("PROMOTING RETRAINED EXPERIMENT MODEL TO PRODUCTION")
print("=" * 80)

# 1. Archive current baseline production artifacts
print("\n1. Archiving current baseline artifacts...")
if PROD_DIR.exists():
    shutil.copytree(PROD_DIR, ARCHIVE_BEST, dirs_exist_ok=True)
    print(f"  -> Baseline model archived to: {ARCHIVE_BEST}")

if PREP_DIR.exists():
    shutil.copytree(PREP_DIR, ARCHIVE_PREP, dirs_exist_ok=True)
    print(f"  -> Baseline preprocessing archived to: {ARCHIVE_PREP}")

if BASE_PARQUET.exists() and not ARCHIVE_PARQUET.exists():
    shutil.copy2(BASE_PARQUET, ARCHIVE_PARQUET)
    print(f"  -> Baseline modeling dataset archived to: {ARCHIVE_PARQUET}")

# 2. Promote candidate model and preprocessing to production
print("\n2. Promoting new model weights and preprocessor to production...")
shutil.copy2(EXP_BEST / "model.pt", PROD_DIR / "model.pt")
print(f"  -> Promoted model.pt ({round((PROD_DIR / 'model.pt').stat().st_size / 1024, 1)} KB)")

if (EXP_BEST / "evaluation_metrics.json").exists():
    shutil.copy2(EXP_BEST / "evaluation_metrics.json", PROD_DIR / "evaluation_metrics.json")
    print("  -> Promoted evaluation_metrics.json")

if (EXP_BEST / "training_history.json").exists():
    shutil.copy2(EXP_BEST / "training_history.json", PROD_DIR / "training_history.json")
    print("  -> Promoted training_history.json")

shutil.copy2(EXP_PREP / "preprocessor.joblib", PREP_DIR / "preprocessor.joblib")
print(f"  -> Promoted preprocessor.joblib ({round((PREP_DIR / 'preprocessor.joblib').stat().st_size / 1024, 1)} KB)")

if (EXP_PREP / "split_info.json").exists():
    shutil.copy2(EXP_PREP / "split_info.json", PREP_DIR / "split_info.json")
    print("  -> Promoted split_info.json")

if (EXP_BEST / "outlier_machines_report.json").exists():
    shutil.copy2(EXP_BEST / "outlier_machines_report.json", PROD_DIR / "outlier_machines_report.json")
    print("  -> Promoted outlier_machines_report.json")

# 3. Promote updated modeling dataset
if UPDATED_PARQUET.exists():
    shutil.copy2(UPDATED_PARQUET, BASE_PARQUET)
    print(f"  -> Promoted modeling_daily_rvm.parquet ({round(BASE_PARQUET.stat().st_size / (1024*1024), 1)} MB)")

# 4. Smoke test inference
print("\n3. Validating production model with a live inference check...")
sys_path_root = str(ROOT_DIR)
import sys
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)

from src.model.preprocessing import RVMPreprocessor
from src.model.network import MultiTaskRVMNet

prep = RVMPreprocessor.load(str(PREP_DIR / "preprocessor.joblib"))
device = torch.device("cpu")
model = MultiTaskRVMNet(
    num_numeric_features=len(prep.num_features),
    cat_cardinalities=prep.get_cat_cardinalities(),
    cat_features=prep.cat_features,
    hidden_dims=(256, 128, 64),
    head_dim=32,
    dropout=0.2
).to(device)

model.load_state_dict(torch.load(str(PROD_DIR / "model.pt"), map_location=device, weights_only=True))
model.eval()

# Sample dummy forward pass
dummy_num = torch.zeros((1, len(prep.num_features)), dtype=torch.float32)
dummy_cat = torch.zeros((1, len(prep.cat_features)), dtype=torch.long)
with torch.no_grad():
    pb, pp = model(dummy_num, dummy_cat)
    pred_bottles = float(torch.expm1(pb).item())
    pred_presenze = float(torch.expm1(pp).item())

print(f"  -> Forward pass test successful!")
print(f"  -> Sample synthetic outputs: BOTTIGLIE={pred_bottles:.1f}, PRESENZE={pred_presenze:.1f}")

print("\n" + "=" * 80)
print("PROMOTION COMPLETED SUCCESSFULLY! PRODUCTION MODEL IS NOW LIVE.")
print("=" * 80)
