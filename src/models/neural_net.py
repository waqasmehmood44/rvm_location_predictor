"""PyTorch Multi-Task Tabular Neural Network for Coripet RVM Performance Prediction.

Architecture:
- Entity Embeddings for categorical features (chain, machine type, province, region, CAP).
- Continuous feature normalization and embedding concatenation.
- Deep shared representation trunk with residual connections, Layer/BatchNorm, and Dropout.
- Specialized regression heads for concurrent prediction of daily PRESENZE and BOTTIGLIE.
- Built-in Monte Carlo Dropout support for epistemic uncertainty estimation.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoripetMultiTaskNN(nn.Module):
    """Multi-task neural network predicting daily PRESENZE and BOTTIGLIE simultaneously."""

    def __init__(
        self,
        emb_dims: List[Tuple[int, int]],
        num_continuous_features: int,
        shared_hidden_layers: List[int] = [128, 64],
        head_hidden_layers: List[int] = [32, 16],
        dropout_rate: float = 0.2,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Categorical Embedding Layers
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in emb_dims]
        )
        self.emb_dropout = nn.Dropout(embedding_dropout)
        total_emb_dim = sum(emb_dim for _, emb_dim in emb_dims)

        # 2. Continuous Features Normalization
        self.num_continuous = num_continuous_features
        if num_continuous_features > 0:
            self.num_bn = nn.BatchNorm1d(num_continuous_features)
        else:
            self.num_bn = nn.Identity()

        # 3. Shared Representation Trunk
        input_dim = total_emb_dim + num_continuous_features
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in shared_hidden_layers:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        self.shared_trunk = nn.Sequential(*shared_layers)
        shared_output_dim = prev_dim

        # 4. Multi-Task Heads
        # Head 1: Expected daily PRESENZE
        presenze_layers = []
        p_dim = shared_output_dim
        for h_dim in head_hidden_layers:
            presenze_layers.extend([
                nn.Linear(p_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2),
            ])
            p_dim = h_dim
        presenze_layers.append(nn.Linear(p_dim, 1))
        # Softplus ensures non-negative predictions for physical counts
        presenze_layers.append(nn.Softplus())
        self.presenze_head = nn.Sequential(*presenze_layers)

        # Head 2: Expected daily BOTTIGLIE
        bottiglie_layers = []
        b_dim = shared_output_dim
        for h_dim in head_hidden_layers:
            bottiglie_layers.extend([
                nn.Linear(b_dim, h_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate / 2),
            ])
            b_dim = h_dim
        bottiglie_layers.append(nn.Linear(b_dim, 1))
        bottiglie_layers.append(nn.Softplus())
        self.bottiglie_head = nn.Sequential(*bottiglie_layers)

    def forward(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass generating predictions for PRESENZE and BOTTIGLIE.

        Args:
            x_cat: LongTensor of shape (batch_size, n_categorical_features)
            x_cont: FloatTensor of shape (batch_size, n_continuous_features)

        Returns:
            Tuple of (pred_presenze, pred_bottiglie), each shape (batch_size, 1)
        """
        # Embed categoricals
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        if embedded:
            x_emb = torch.cat(embedded, dim=1)
            x_emb = self.emb_dropout(x_emb)
        else:
            x_emb = torch.empty((x_cat.size(0), 0), device=x_cat.device)

        # Normalize continuous features
        if self.num_continuous > 0 and x_cont.size(1) > 0:
            x_num = self.num_bn(x_cont)
            x = torch.cat([x_emb, x_num], dim=1)
        else:
            x = x_emb

        # Shared representation
        shared = self.shared_trunk(x)

        # Target heads
        pred_presenze = self.presenze_head(shared)
        pred_bottiglie = self.bottiglie_head(shared)

        return pred_presenze, pred_bottiglie

    def predict_with_uncertainty(
        self, x_cat: torch.Tensor, x_cont: torch.Tensor, n_samples: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """Monte Carlo Dropout inference to estimate predictive mean and epistemic variance."""
        self.train()  # Enable dropout during inference
        presenze_samples = []
        bottiglie_samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                p, b = self.forward(x_cat, x_cont)
                presenze_samples.append(p.cpu())
                bottiglie_samples.append(b.cpu())

        self.eval()

        p_stack = torch.stack(presenze_samples)  # (samples, batch, 1)
        b_stack = torch.stack(bottiglie_samples)

        return {
            "presenze": {
                "mean": float(p_stack.mean().item()),
                "std": float(p_stack.std().item()),
            },
            "bottiglie": {
                "mean": float(b_stack.mean().item()),
                "std": float(b_stack.std().item()),
            },
        }
