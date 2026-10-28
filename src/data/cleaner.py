"""Data cleaning and sanitization module for Coripet operational and store datasets."""

from typing import List, Optional
import numpy as np
import pandas as pd


class DataCleaner:
    """Cleans, formats, and validates merged raw operational and store master data."""

    def __init__(
        self,
        date_column: str = "GIORNO",
        targets: Optional[List[str]] = None,
    ):
        self.date_column = date_column
        self.targets = targets or ["PRESENZE", "BOTTIGLIE"]

    def clean_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses operational date column to proper datetime."""
        df = df.copy()
        if self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
            df = df.dropna(subset=[self.date_column]).sort_values(self.date_column)
        return df

    def clean_postal_code(self, series: pd.Series) -> pd.Series:
        """Standardizes Italian postal codes (CAP) to 5-digit strings with leading zeros."""
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(5)
        )
        # Mark invalid length/corrupt values as Unknown
        cleaned = cleaned.apply(lambda x: x if len(x) == 5 and x.isdigit() else "UNKNOWN")
        return cleaned

    def clean_geo_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes Latitude and Longitude columns to float, handles comma decimals."""
        df = df.copy()
        for col in ["Latitudine", "Longitudine", "latitudine", "longitudine"]:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", ".", regex=False)
                        .str.strip()
                    )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Standardize naming to lowercase
        if "Latitudine" in df.columns and "latitudine" not in df.columns:
            df["latitudine"] = df["Latitudine"]
        if "Longitudine" in df.columns and "longitudine" not in df.columns:
            df["longitudine"] = df["Longitudine"]

        return df

    def sanitize_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates numerical target values (PRESENZE, BOTTIGLIE)."""
        df = df.copy()
        for target in self.targets:
            if target in df.columns:
                df[target] = pd.to_numeric(df[target], errors="coerce")
                # Operational counts cannot be negative
                df = df[df[target].notna() & (df[target] >= 0)]
        return df

    def clean_text_categories(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Standardizes casing and whitespace for categorical features."""
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})
                )
        return df

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full cleaning sequence."""
        df = self.clean_date_column(df)
        df = self.sanitize_targets(df)
        df = self.clean_geo_coordinates(df)

        for cap_col in ["CAP", "cap"]:
            if cap_col in df.columns:
                df["cap"] = self.clean_postal_code(df[cap_col])

        cat_cols = ["CLIENTE", "TipoMacchina", "Comune", "Provincia", "Regione", "Stato"]
        df = self.clean_text_categories(df, cat_cols)

        return df
