"""Time-aware and group-based data splitting to prevent temporal and store-level data leakage."""

from typing import Generator, List, Optional, Tuple
import pandas as pd
import numpy as np


class TimeAndGroupSplitter:
    """Splits operational data using temporal cutoffs and store-level grouping.

    Ensures zero data leakage:
    1. Temporal split respects causality (train on past, evaluate on future).
    2. Group split holds out complete store locations (PuntoFisico) to test
       cold-start generalization on brand-new locations.
    """

    def __init__(
        self,
        date_column: str = "GIORNO",
        store_key: str = "PF",
    ):
        self.date_column = date_column
        self.store_key = store_key

    def temporal_train_test_split(
        self,
        df: pd.DataFrame,
        test_days: int = 90,
        val_days: int = 45,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits data strictly by date into Train, Validation, and Test sets."""
        df_sorted = df.sort_values(self.date_column).copy()
        max_date = df_sorted[self.date_column].max()

        test_cutoff = max_date - pd.Timedelta(days=test_days)
        val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

        train_df = df_sorted[df_sorted[self.date_column] < val_cutoff].copy()
        val_df = df_sorted[
            (df_sorted[self.date_column] >= val_cutoff)
            & (df_sorted[self.date_column] < test_cutoff)
        ].copy()
        test_df = df_sorted[df_sorted[self.date_column] >= test_cutoff].copy()

        return train_df, val_df, test_df

    def group_store_train_test_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits data by holding out entire store locations (PuntoFisico).

        Critical for simulating real-world inference where the store is completely unseen.
        """
        unique_stores = df[self.store_key].unique()
        rng = np.random.RandomState(random_state)
        shuffled_stores = rng.permutation(unique_stores)

        n_stores = len(shuffled_stores)
        n_test = int(n_stores * test_ratio)
        n_val = int(n_stores * val_ratio)

        test_stores = set(shuffled_stores[:n_test])
        val_stores = set(shuffled_stores[n_test : n_test + n_val])
        train_stores = set(shuffled_stores[n_test + n_val :])

        train_df = df[df[self.store_key].isin(train_stores)].copy()
        val_df = df[df[self.store_key].isin(val_stores)].copy()
        test_df = df[df[self.store_key].isin(test_stores)].copy()

        return train_df, val_df, test_df

    def rolling_time_group_folds(
        self,
        df: pd.DataFrame,
        n_splits: int = 4,
        window_days: int = 60,
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Yields rolling temporal folds for robust backtesting."""
        df_sorted = df.sort_values(self.date_column).copy()
        max_date = df_sorted[self.date_column].max()
        min_date = df_sorted[self.date_column].min()

        total_days = (max_date - min_date).days
        step_days = total_days // (n_splits + 1)

        for i in range(1, n_splits + 1):
            train_cutoff = min_date + pd.Timedelta(days=step_days * i)
            test_cutoff = train_cutoff + pd.Timedelta(days=window_days)

            train_fold = df_sorted[df_sorted[self.date_column] < train_cutoff]
            test_fold = df_sorted[
                (df_sorted[self.date_column] >= train_cutoff)
                & (df_sorted[self.date_column] < test_cutoff)
            ]

            if not train_fold.empty and not test_fold.empty:
                yield train_fold, test_fold
