"""Reproducible training loop with early stopping, scheduling, and model checkpointing."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.neural_net import CoripetMultiTaskNN
from src.models.metrics import RegressionMetrics


def set_seed(seed: int = 42):
    """Enforces strict reproducibility across random number generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ModelTrainer:
    """Manages the training, validation, and serialization of the Coripet Multi-Task NN."""

    def __init__(
        self,
        model: CoripetMultiTaskNN,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        presenze_weight: float = 0.5,
        bottiglie_weight: float = 0.5,
        checkpoint_dir: str = "models/saved_models",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.presenze_weight = presenze_weight
        self.bottiglie_weight = bottiglie_weight

        # Huber Loss is resilient to volume outliers in RVM operational logs
        self.criterion_presenze = nn.HuberLoss(delta=1.5)
        self.criterion_bottiglie = nn.HuberLoss(delta=10.0)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Runs a single training epoch across all batches."""
        self.model.train()
        total_loss = 0.0

        for x_cat, x_cont, y in dataloader:
            x_cat = x_cat.to(self.device)
            x_cont = x_cont.to(self.device)
            y = y.to(self.device)

            y_presenze = y[:, 0:1]
            y_bottiglie = y[:, 1:2]

            self.optimizer.zero_grad()
            pred_p, pred_b = self.model(x_cat, x_cont)

            loss_p = self.criterion_presenze(pred_p, y_presenze)
            loss_b = self.criterion_bottiglie(pred_b, y_bottiglie)
            loss = (self.presenze_weight * loss_p) + (self.bottiglie_weight * loss_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)

        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates model on validation or test dataset."""
        self.model.eval()
        total_loss = 0.0
        p_preds, b_preds = [], []
        p_trues, b_trues = [], []

        with torch.no_grad():
            for x_cat, x_cont, y in dataloader:
                x_cat = x_cat.to(self.device)
                x_cont = x_cont.to(self.device)
                y = y.to(self.device)

                y_p = y[:, 0:1]
                y_b = y[:, 1:2]

                pred_p, pred_b = self.model(x_cat, x_cont)

                loss_p = self.criterion_presenze(pred_p, y_p)
                loss_b = self.criterion_bottiglie(pred_b, y_b)
                loss = (self.presenze_weight * loss_p) + (self.bottiglie_weight * loss_b)

                total_loss += loss.item() * len(y)

                p_preds.append(pred_p.cpu().numpy())
                b_preds.append(pred_b.cpu().numpy())
                p_trues.append(y_p.cpu().numpy())
                b_trues.append(y_b.cpu().numpy())

        p_preds = np.vstack(p_preds).flatten()
        b_preds = np.vstack(b_preds).flatten()
        p_trues = np.vstack(p_trues).flatten()
        b_trues = np.vstack(b_trues).flatten()

        metrics_p = RegressionMetrics.calculate_all(p_trues, p_preds, prefix="presenze_")
        metrics_b = RegressionMetrics.calculate_all(b_trues, b_preds, prefix="bottiglie_")

        results = {
            "val_loss": total_loss / len(dataloader.dataset),
            **metrics_p,
            **metrics_b,
        }
        return results

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        patience: int = 12,
        model_name: str = "best_coripet_model.pt",
    ) -> Dict[str, List[float]]:
        """Fits the model with early stopping and saves best checkpoint."""
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_checkpoint_path = self.checkpoint_dir / model_name

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            eval_metrics = self.evaluate(val_loader)
            val_loss = eval_metrics["val_loss"]

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "eval_metrics": eval_metrics,
                    },
                    best_checkpoint_path,
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        return history
