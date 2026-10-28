"""Data cleaning and string normalization utilities for Coripet integration pipeline."""

import re
import unicodedata
from typing import Optional
import pandas as pd
import numpy as np


def clean_postal_code(val: any) -> str:
    """Standardizes Italian postal code (CAP) to 5-digit string with leading zeros."""
    if pd.isna(val):
        return "UNKNOWN"
    s = str(val).strip().replace(".0", "")
    s = re.sub(r"\D", "", s)
    if not s:
        return "UNKNOWN"
    return s.zfill(5)


def clean_comune_name(comune: any) -> str:
    """Normalizes municipality name: uppercase, trims whitespace, removes diacritics."""
    if pd.isna(comune):
        return "UNKNOWN"
    s = str(comune).strip()
    # Normalize unicode accents (e.g. Forlì -> FORLI)
    nfkd = unicodedata.normalize("NFKD", s)
    s = "".join([c for c in nfkd if not unicodedata.combining(c)])
    # Remove punctuation, extra spaces
    s = re.sub(r"['\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s if s else "UNKNOWN"


def clean_decimal_number(val: any) -> Optional[float]:
    """Converts Italian comma decimal strings to standard float."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def clean_address_string(addr: any) -> str:
    """Standardizes street address for exact and entity matching."""
    if pd.isna(addr):
        return ""
    s = str(addr).strip().upper()
    # Normalize common Italian street prefix abbreviations
    s = re.sub(r"\bV\.LE\b", "VIALE", s)
    s = re.sub(r"\bV\.\b", "VIA", s)
    s = re.sub(r"\bP\.ZZA\b", "PIAZZA", s)
    s = re.sub(r"\bC\.SO\b", "CORSO", s)
    s = re.sub(r"\bLOC\.\b", "LOCALITA", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
