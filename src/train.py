import os
import sys
sys.path.insert(0, os.path.abspath("."))
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.splitter import TimeAndGroupSplitter
from src.features.geocoding import LocationEnricher
from src.features.store_features import StoreFeatureExtractor
from src.features.preprocessor import TabularPreprocessor
from src.models.neural_net import CoripetMultiTaskNN
from src.models.dataset import TabularDataset
from src.models.trainer import ModelTrainer, set_seed
from src.business.scoring import SuitabilityScorer


def run_training_pipeline(config_path: str = "configs/config.yaml"):
    """Executes full training sequence from raw data ingestion to model checkpointing."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Enforce strict reproducibility
    set_seed(cfg["project"]["random_seed"])

    print(">>> 1. Ingesting raw datasets...")
    loader = DataLoader(
        operational_key=cfg["data_schema"]["operational_key"],
        store_master_key=cfg["data_schema"]["store_master_key"],
        date_column=cfg["data_schema"]["date_column"],
    )
    op_df = loader.load_operational_data(cfg["paths"]["raw_operational"])
    sm_df = loader.load_store_master(cfg["paths"]["raw_store_master"])
    merged_df = loader.merge_datasets(op_df, sm_df)

    print(">>> 2. Cleaning and sanitizing records...")
    cleaner = DataCleaner(date_column=cfg["data_schema"]["date_column"])
    clean_df = cleaner.clean_dataset(merged_df)

    # Enrich store brand & administrative hierarchy
    extractor = StoreFeatureExtractor()
    clean_df["retail_chain"] = clean_df["CLIENTE"].apply(extractor.extract_retail_chain)
    clean_df["tipo_macchina"] = clean_df["TipoMacchina"].apply(extractor.normalize_machine_type)

    print(">>> 3. Calibrating business benchmarks empirically...")
    scorer = SuitabilityScorer(config_path=cfg["paths"]["business_rules"])
    scorer.calibrate_benchmarks(clean_df)

    print(">>> 4. Splitting data with time-aware and location-grouped validation...")
    splitter = TimeAndGroupSplitter(
        date_column=cfg["data_schema"]["date_column"],
        store_key=cfg["data_schema"]["operational_key"],
    )
    train_df, val_df, test_df = splitter.temporal_train_test_split(
        clean_df,
        test_days=cfg["validation"]["test_size_days"],
        val_days=cfg["validation"]["test_size_days"] // 2,
    )
    print(f"    Train records: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    print(">>> 5. Fitting tabular preprocessor (Train set only)...")
    preprocessor = TabularPreprocessor(
        categorical_features=cfg["data_schema"]["categorical_features"],
        numerical_features=cfg["data_schema"]["numerical_features"],
    )
    x_cat_train, x_cont_train = preprocessor.fit_transform(train_df)
    x_cat_val, x_cont_val = preprocessor.transform(val_df)
    x_cat_test, x_cont_test = preprocessor.transform(test_df)

    # Save preprocessor for inference
    prep_save_path = Path(cfg["paths"]["preprocessor_dir"]) / "preprocessor.joblib"
    preprocessor.save(prep_save_path)
    print(f"    Saved preprocessor to: {prep_save_path}")

    # Targets
    target_cols = cfg["data_schema"]["targets"]
    y_train = train_df[target_cols].values
    y_val = val_df[target_cols].values
    y_test = test_df[target_cols].values

    # DataLoaders
    batch_size = cfg["training"]["batch_size"]
    train_dataset = TabularDataset(x_cat_train, x_cont_train, y_train)
    val_dataset = TabularDataset(x_cat_val, x_cont_val, y_val)
    test_dataset = TabularDataset(x_cat_test, x_cont_test, y_test)

    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(">>> 6. Initializing Coripet Multi-Task Tabular Neural Network...")
    emb_dims = [
        (preprocessor.cat_dims[col], preprocessor.emb_dims[col])
        for col in preprocessor.categorical_features
        if col in preprocessor.cat_dims
    ]
    model = CoripetMultiTaskNN(
        emb_dims=emb_dims,
        num_continuous_features=len(preprocessor.numerical_features),
        shared_hidden_layers=cfg["model"]["shared_hidden_layers"],
        head_hidden_layers=cfg["model"]["presenze_head_layers"],
        dropout_rate=cfg["model"]["dropout_rate"],
    )

    print(">>> 7. Training model with early stopping...")
    trainer = ModelTrainer(
        model=model,
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        presenze_weight=cfg["training"]["loss_presenze_weight"],
        bottiglie_weight=cfg["training"]["loss_bottiglie_weight"],
        checkpoint_dir=cfg["paths"]["model_checkpoint_dir"],
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg["training"]["max_epochs"],
        patience=cfg["training"]["patience"],
        model_name="best_coripet_model.pt",
    )

    print(">>> 8. Evaluating on out-of-time test set...")
    test_metrics = trainer.evaluate(test_loader)
    print("\n--- TEST SET EVALUATION REPORT ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print("----------------------------------")
    print(f"Model successfully saved to {cfg['paths']['model_checkpoint_dir']}/best_coripet_model.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Coripet RVM Training Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["production", "modular"],
        default="production",
        help="Training mode: 'production' trains the multi-task neural network on certified modeling data; 'modular' trains the baseline scaffold.",
    )
    args = parser.parse_args()

    if args.mode == "production" and os.path.exists("data/processed/modeling_daily_rvm.parquet"):
        from src.model.train import train_model
        from src.model.evaluate import evaluate_model
        print(">>> Running Production Multi-Task Neural Network Pipeline...")
        train_model()
        print(">>> Running Untouched Test Holdout Evaluation...")
        evaluate_model()
    else:
        run_training_pipeline()
