"""
PyTorch Dataset and DataLoader definitions for Coripet RVM Predictor.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple

class RVMDataset(Dataset):
    """PyTorch Dataset wrapping numerical features, categorical features, dual log1p targets, and sample weights."""
    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray, y_bottiglie: np.ndarray, y_presenze: np.ndarray, weights: np.ndarray = None):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y_bottiglie = torch.tensor(y_bottiglie, dtype=torch.float32).unsqueeze(1)
        self.y_presenze = torch.tensor(y_presenze, dtype=torch.float32).unsqueeze(1)
        if weights is not None:
            self.weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
        else:
            self.weights = torch.ones((len(X_num), 1), dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.X_num)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.y_bottiglie[idx],
            self.y_presenze[idx],
            self.weights[idx]
        )

def get_dataloaders(
    train_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    val_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    train_weights: np.ndarray = None,
    val_weights: np.ndarray = None,
    test_weights: np.ndarray = None,
    batch_size: int = 4096,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds efficient PyTorch DataLoaders for training, validation, and testing with outlier weighting."""
    train_dataset = RVMDataset(*train_arrays[:4], weights=train_weights)
    val_dataset = RVMDataset(*val_arrays[:4], weights=val_weights)
    test_dataset = RVMDataset(*test_arrays[:4], weights=test_weights)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader
