"""PyTorch Dataset wrapper for multi-task tabular inputs."""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Encapsulates categorical and continuous feature arrays along with regression targets."""

    def __init__(
        self,
        x_cat: np.ndarray,
        x_cont: np.ndarray,
        y: Optional[np.ndarray] = None,
    ):
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_cont = torch.tensor(x_cont, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.x_cat)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.x_cat[idx], self.x_cont[idx], self.y[idx]
        return self.x_cat[idx], self.x_cont[idx]
