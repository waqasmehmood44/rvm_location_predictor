"""
Model training pipeline for Coripet RVM Multi-Output Neural Network.
Executes leakage-free group-separated training and validation with early stopping.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any

from src.model.preprocessing import RVMPreprocessor, split_pfs
from src.model.dataset import get_dataloaders
from src.model.network import MultiTaskRVMNet, MultiTaskHuberLoss

def train_model(
    parquet_path: str = "data/processed/modeling_daily_rvm.parquet",
    models_dir: str = "models",
    batch_size: int = 4096,
    max_epochs: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    random_seed: int = 42,
    weight_bottiglie: float = 1.0,
    weight_presenze: float = 1.0
) -> Dict[str, Any]:
    print("=" * 80)
    print("CORIPET RVM PREDICTOR — NEURAL NETWORK TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Set reproducible random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}")
    
    # 2. Load certified dataset
    print(f"Loading certified population from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} observations across {df['PF'].nunique():,} unique PFs.")
    
    # 3. Group-separated split by Physical Point (PF)
    train_df, val_df, test_df, split_info = split_pfs(df, seed=random_seed)
    
    # 4. Save test set and split metadata for untouched final evaluation
    os.makedirs(f"{models_dir}/preprocessing", exist_ok=True)
    os.makedirs(f"{models_dir}/best_model", exist_ok=True)
    
    split_info_path = f"{models_dir}/preprocessing/split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split metadata to {split_info_path}")
    
    # Save test set partition to disk
    test_parquet_path = f"{models_dir}/preprocessing/test_holdout.parquet"
    test_df.to_parquet(test_parquet_path, index=False)
    print(f"Saved unseen-PF test partition ({len(test_df):,} rows) to {test_parquet_path}")
    
    # 5. Preprocessing: Fit strictly on train_df
    preprocessor = RVMPreprocessor(random_seed=random_seed)
    preprocessor.fit(train_df)
    preprocessor.save(f"{models_dir}/preprocessing/preprocessor.joblib")
    
    # 6. Transform partitions
    print("Transforming partitions into model-ready tensors...")
    train_arrays = preprocessor.transform(train_df, nielsen_dropout=0.25)
    val_arrays = preprocessor.transform(val_df, nielsen_dropout=0.0)
    test_arrays = preprocessor.transform(test_df, nielsen_dropout=0.0)
    
    # 7. Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        train_arrays, val_arrays, test_arrays, batch_size=batch_size
    )
    
    # 8. Initialize MultiTask Neural Network
    num_numeric_features = len(preprocessor.num_features)
    cat_cardinalities = preprocessor.get_cat_cardinalities()
    cat_features = preprocessor.cat_features
    
    model = MultiTaskRVMNet(
        num_numeric_features=num_numeric_features,
        cat_cardinalities=cat_cardinalities,
        cat_features=cat_features,
        hidden_dims=(256, 128, 64),
        head_dim=32,
        dropout=0.2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n--- Model Architecture ---")
    print(model)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # 9. Loss & Optimizer
    criterion = MultiTaskHuberLoss(
        weight_bottiglie=weight_bottiglie, 
        weight_presenze=weight_presenze, 
        delta=1.0
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    
    # 10. Training Loop with Early Stopping
    print(f"\nStarting training for up to {max_epochs} epochs (Batch size: {batch_size}, Early stopping patience: {patience})...")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []
    
    start_total_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss_total = 0.0
        train_loss_b = 0.0
        train_loss_p = 0.0
        num_train_batches = 0
        
        for batch_num, batch_cat, batch_yb, batch_yp in train_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_yb = batch_yb.to(device)
            batch_yp = batch_yp.to(device)
            
            optimizer.zero_grad()
            pred_b, pred_p = model(batch_num, batch_cat)
            
            total_loss, loss_b, loss_p = criterion(pred_b, pred_p, batch_yb, batch_yp)
            total_loss.backward()
            
            # Gradient clipping for numerical stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            train_loss_total += total_loss.item()
            train_loss_b += loss_b.item()
            train_loss_p += loss_p.item()
            num_train_batches += 1
            
        train_loss_total /= num_train_batches
        train_loss_b /= num_train_batches
        train_loss_p /= num_train_batches
        
        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_loss_b = 0.0
        val_loss_p = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_num, batch_cat, batch_yb, batch_yp in val_loader:
                batch_num = batch_num.to(device)
                batch_cat = batch_cat.to(device)
                batch_yb = batch_yb.to(device)
                batch_yp = batch_yp.to(device)
                
                pred_b, pred_p = model(batch_num, batch_cat)
                total_loss, loss_b, loss_p = criterion(pred_b, pred_p, batch_yb, batch_yp)
                
                val_loss_total += total_loss.item()
                val_loss_b += loss_b.item()
                val_loss_p += loss_p.item()
                num_val_batches += 1
                
        val_loss_total /= num_val_batches
        val_loss_b /= num_val_batches
        val_loss_p /= num_val_batches
        
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss_total)
        epoch_duration = time.time() - epoch_start
        
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss_total, 4),
            "train_loss_bottiglie": round(train_loss_b, 4),
            "train_loss_presenze": round(train_loss_p, 4),
            "val_loss": round(val_loss_total, 4),
            "val_loss_bottiglie": round(val_loss_b, 4),
            "val_loss_presenze": round(val_loss_p, 4),
            "lr": current_lr,
            "duration_sec": round(epoch_duration, 1)
        }
        history.append(epoch_metrics)
        
        # Checkpoint best model
        is_best = val_loss_total < best_val_loss
        if is_best:
            best_val_loss = val_loss_total
            best_epoch = epoch
            patience_counter = 0
            best_model_path = f"{models_dir}/best_model/model.pt"
            torch.save(model.state_dict(), best_model_path)
            mark = "(*) BEST MODEL SAVED"
        else:
            patience_counter += 1
            mark = f"(Patience: {patience_counter}/{patience})"
            
        print(f"Epoch {epoch:02d}/{max_epochs:02d} [{epoch_duration:.1f}s] - "
              f"Train: {train_loss_total:.4f} (B: {train_loss_b:.4f}, P: {train_loss_p:.4f}) | "
              f"Val: {val_loss_total:.4f} (B: {val_loss_b:.4f}, P: {val_loss_p:.4f}) | "
              f"lr: {current_lr:.1e} {mark}")
              
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val Loss = {best_val_loss:.4f}.")
            break
            
    total_train_time = time.time() - start_total_time
    print(f"\nTraining completed in {total_train_time/60:.1f} minutes.")
    
    # Save training history and summary
    training_summary = {
        "split_info": split_info,
        "total_params": total_params,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "total_train_time_min": round(total_train_time / 60, 2),
        "history": history,
        "weight_bottiglie": weight_bottiglie,
        "weight_presenze": weight_presenze
    }
    
    history_path = f"{models_dir}/best_model/training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_summary, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    return training_summary

if __name__ == "__main__":
    train_model()
