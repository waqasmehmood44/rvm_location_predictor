"""Data ingestion module for Coripet operational and store master datasets.

Responsible for reading raw CSV/Excel/Parquet files, verifying schema keys,
and merging operational logs with store master profiles without data leakage.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd


class DataLoader:
    """Ingests and validates raw operational logs and Nielsen store master data."""

    def __init__(
        self,
        operational_key: str = "PF",
        store_master_key: str = "PuntoFisico",
        date_column: str = "GIORNO",
    ):
        self.operational_key = operational_key
        self.store_master_key = store_master_key
        self.date_column = date_column

    def load_operational_data(self, file_path: str | Path) -> pd.DataFrame:
        """Loads historical daily RVM operational logs."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Operational data file not found at: {path}")

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Basic validation
        expected_cols = [self.operational_key, self.date_column]
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise KeyError(
                f"Operational dataset is missing required columns: {missing}. "
                f"Present columns: {list(df.columns)}"
            )

        return df

    def load_store_master(self, file_path: str | Path) -> pd.DataFrame:
        """Loads Nielsen / store master profile data."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Store master data file not found at: {path}")

        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if self.store_master_key not in df.columns:
            raise KeyError(
                f"Store master dataset missing key column '{self.store_master_key}'. "
                f"Present columns: {list(df.columns)}"
            )

        return df

    def merge_datasets(
        self, operational_df: pd.DataFrame, store_master_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Joins daily operations with store master data on PF == PuntoFisico."""
        # Align key types
        op_df = operational_df.copy()
        sm_df = store_master_df.copy()

        op_df[self.operational_key] = op_df[self.operational_key].astype(str).str.strip()
        sm_df[self.store_master_key] = sm_df[self.store_master_key].astype(str).str.strip()

        # Remove duplicate store master keys if any, keeping first active record
        sm_deduped = sm_df.drop_duplicates(subset=[self.store_master_key])

        merged = pd.merge(
            op_df,
            sm_deduped,
            left_on=self.operational_key,
            right_on=self.store_master_key,
            how="inner",
        )

        return merged
