import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.model.evaluate import evaluate_model

print("Evaluating retrained experiment model...")
exp_metrics = evaluate_model(
    models_dir=str(ROOT_DIR / "models" / "experiment_with_new_data"),
    test_parquet=str(ROOT_DIR / "models" / "preprocessing" / "test_holdout.parquet"),
    training_parquet=str(ROOT_DIR / "data" / "processed" / "modeling_daily_rvm_updated.parquet")
)

with open(ROOT_DIR / "models" / "best_model" / "evaluation_metrics.json") as f:
    base_metrics = json.load(f)

print("\n" + "=" * 80)
print("         SIDE-BY-SIDE IMPACT COMPARISON: BASELINE vs RETRAINED MODEL")
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
