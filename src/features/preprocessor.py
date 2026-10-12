"""Tabular feature preprocessing and serialization pipeline for PyTorch Neural Networks."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class TabularPreprocessor:
    """Preprocesses mixed categorical and continuous tabular features for Neural Networks.

    - Fits solely on training data to strictly prevent data leakage.
    - Encodes categorical levels as integers with dedicated index 0 for unseen/<UNK> tokens.
    - Computes PyTorch embedding dimension heuristics.
    - Scales numerical features with outlier-resistant RobustScaler.
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
    ):
        self.categorical_features = categorical_features or [
            "retail_chain",
            "tipo_macchina",
            "provincia",
            "regione",
            "cap",
        ]
        self.numerical_features = numerical_features or [
            "latitudine",
            "longitudine",
        ]
        self.cat_to_idx: Dict[str, Dict[str, int]] = {}
        self.cat_dims: Dict[str, int] = {}
        self.emb_dims: Dict[str, int] = {}
        self.scaler = RobustScaler()
        self.is_fitted = False

    def _calc_emb_dim(self, cardinality: int) -> int:
        """Heuristic for categorical embedding dimension: min(50, (cardinality + 1) // 2)."""
        return max(4, min(50, (cardinality + 1) // 2))

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fits vocabulary indices and numerical scaler on training set only."""
        # 1. Fit Categorical mappings
        for col in self.categorical_features:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().astype(str).unique())
                # Index 0 is reserved for UNKNOWN / unseen categories during inference
                mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
                self.cat_to_idx[col] = mapping
                cardinality = len(unique_vals) + 1  # include UNK
                self.cat_dims[col] = cardinality
                self.emb_dims[col] = self._calc_emb_dim(cardinality)

        # 2. Fit Numerical scaler
        num_cols = [c for c in self.numerical_features if c in df.columns]
        if num_cols:
            num_data = df[num_cols].fillna(df[num_cols].median())
            self.scaler.fit(num_data)

        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms DataFrame into categorical integer tensor array and continuous array."""
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transforming data.")

        # Transform Categoricals: Unseen values mapped to 0 (UNKNOWN)
        cat_encoded_cols = []
        for col in self.categorical_features:
            mapping = self.cat_to_idx.get(col, {})
            if col in df.columns:
                series = df[col].astype(str)
                indices = series.map(mapping).fillna(0).astype(np.int64).values
            else:
                indices = np.zeros(len(df), dtype=np.int64)
            cat_encoded_cols.append(indices)

        cat_array = (
            np.column_stack(cat_encoded_cols)
            if cat_encoded_cols
            else np.empty((len(df), 0), dtype=np.int64)
        )

        # Transform Numericals
        num_cols = [c for c in self.numerical_features if c in df.columns]
        if num_cols:
            num_data = df[num_cols].fillna(df[num_cols].median()).values
            num_scaled = self.scaler.transform(num_data).astype(np.float32)
        else:
            num_scaled = np.empty((len(df), 0), dtype=np.float32)

        return cat_array, num_scaled

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fits and transforms in a single call."""
        return self.fit(df).transform(df)

    def save(self, filepath: str | Path):
        """Serializes preprocessor state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, filepath: str | Path) -> "TabularPreprocessor":
        """Loads serialized preprocessor from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {path}")
        return joblib.load(path)
