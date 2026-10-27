"""
Multi-output, multi-task neural network architecture for Coripet RVM Predictor.
Implements shared representation trunk with dual heads for PRESENZE and BOTTIGLIE log1p targets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class MultiTaskRVMNet(nn.Module):
    """
    Multi-Task Neural Network with Shared Representation Trunk and Dual Output Heads.
    Input Features (Numerical + Categorical Embeddings)
                        ↓
            Shared Dense Trunk (256 -> 128 -> 64)
                        ↓
            ┌───────────────────────┐
            ↓                       ↓
    PRESENZE Head (32 -> 1)   BOTTIGLIE Head (32 -> 1)
            ↓                       ↓
    log1p(PRESENZE)           log1p(BOTTIGLIE)
    """
    def __init__(
        self,
        num_numeric_features: int,
        cat_cardinalities: Dict[str, int],
        cat_features: List[str],
        hidden_dims: Tuple[int, int, int] = (256, 128, 64),
        head_dim: int = 32,
        dropout: float = 0.2
    ):
        super().__init__()
        self.cat_features = cat_features
        self.embeddings = nn.ModuleDict()
        
        total_emb_dim = 0
        for feat in cat_features:
            vocab_size = cat_cardinalities[feat]
            # Sizing heuristic: between 4 and 32 dimensions based on cardinality
            emb_dim = int(min(32, max(4, round(vocab_size ** 0.5 * 2))))
            self.embeddings[feat] = nn.Embedding(vocab_size, emb_dim)
            total_emb_dim += emb_dim
            
        total_in_dim = num_numeric_features + total_emb_dim
        
        # Shared Representation Backbone
        h1, h2, h3 = hidden_dims
        self.trunk = nn.Sequential(
            nn.Linear(total_in_dim, h1),
            nn.GELU(),
            nn.LayerNorm(h1),
            nn.Dropout(dropout),
            
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.LayerNorm(h2),
            nn.Dropout(dropout),
            
            nn.Linear(h2, h3),
            nn.GELU(),
            nn.LayerNorm(h3),
            nn.Dropout(dropout * 0.5)
        )
        
        # Head 1: PRESENZE Head (predicts log1p(PRESENZE))
        self.head_presenze = nn.Sequential(
            nn.Linear(h3, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1)
        )
        
        # Head 2: BOTTIGLIE Head (predicts log1p(BOTTIGLIE))
        self.head_bottiglie = nn.Sequential(
            nn.Linear(h3, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1)
        )
        
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x_num: Float tensor [batch_size, num_numeric_features]
            x_cat: Long tensor [batch_size, num_cat_features]
        Returns:
            (pred_bottiglie, pred_presenze) in log1p space
        """
        # Embed each categorical feature
        embs = []
        for i, feat in enumerate(self.cat_features):
            embs.append(self.embeddings[feat](x_cat[:, i]))
            
        x_all = torch.cat([x_num] + embs, dim=1)
        shared_repr = self.trunk(x_all)
        
        pred_presenze = self.head_presenze(shared_repr)
        pred_bottiglie = self.head_bottiglie(shared_repr)
        
        return pred_bottiglie, pred_presenze

class MultiTaskHuberLoss(nn.Module):
    """Multi-task Huber Loss with task balancing weights and optional sample weights for outlier mitigation."""
    def __init__(self, weight_bottiglie: float = 1.0, weight_presenze: float = 1.0, delta: float = 1.0):
        super().__init__()
        self.weight_bottiglie = weight_bottiglie
        self.weight_presenze = weight_presenze
        self.delta = delta
        self.huber_none = nn.HuberLoss(reduction="none", delta=delta)
        self.huber_mean = nn.HuberLoss(reduction="mean", delta=delta)
        
    def forward(
        self, 
        pred_bottiglie: torch.Tensor, 
        pred_presenze: torch.Tensor, 
        target_bottiglie: torch.Tensor, 
        target_presenze: torch.Tensor,
        sample_weights: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sample_weights is None:
            loss_b = self.huber_mean(pred_bottiglie, target_bottiglie)
            loss_p = self.huber_mean(pred_presenze, target_presenze)
        else:
            loss_b_raw = self.huber_none(pred_bottiglie, target_bottiglie).squeeze()
            loss_p_raw = self.huber_none(pred_presenze, target_presenze).squeeze()
            sw = sample_weights.squeeze()
            w_sum = sw.sum().clamp(min=1e-5)
            loss_b = (loss_b_raw * sw).sum() / w_sum
            loss_p = (loss_p_raw * sw).sum() / w_sum

        total_loss = self.weight_bottiglie * loss_b + self.weight_presenze * loss_p
        return total_loss, loss_b, loss_p
