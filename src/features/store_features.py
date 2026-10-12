"""Store feature engineering and brand extraction utilities."""

import re
from typing import Dict, List, Optional
import pandas as pd


# Common Italian supermarket and hypermarket retail brands in Coripet network
KNOWN_RETAIL_CHAINS = [
    "ESSELUNGA",
    "CONAD",
    "COOP",
    "IPERCOOP",
    "CARREFOUR",
    "PAM",
    "PANORAMA",
    "LIDL",
    "EUROSPIN",
    "BENNET",
    "IPER",
    "IPERAL",
    "TIGROS",
    "DESPAR",
    "INTERSPAR",
    "EUROSPAR",
    "CRAI",
    "SELEX",
    "FAMILA",
    "IL GIGANTE",
    "DECATHLON",
    "LEROY MERLIN",
    "BRICO",
    "MD",
    "ALDI",
]


class StoreFeatureExtractor:
    """Extracts standardized brand and operational features from store descriptions."""

    def __init__(self, known_chains: Optional[List[str]] = None):
        self.known_chains = known_chains or KNOWN_RETAIL_CHAINS

    def extract_retail_chain(self, store_name: str) -> str:
        """Parses store name to identify parent retail chain."""
        if not store_name or pd.isna(store_name):
            return "UNKNOWN"

        clean_name = str(store_name).strip().upper()

        for chain in self.known_chains:
            # Match whole word boundary to prevent false substring matches
            pattern = rf"\b{re.escape(chain)}\b"
            if re.search(pattern, clean_name):
                return chain

        # Fallback to first word as potential chain brand if not recognized
        first_token = re.split(r"[\s\-_,]+", clean_name)[0]
        return first_token if len(first_token) > 2 else "OTHER"

    def normalize_machine_type(self, machine_type: str) -> str:
        """Standardizes RVM machine model designation."""
        if not machine_type or pd.isna(machine_type):
            return "UNKNOWN"
        return str(machine_type).strip().upper()

    def transform_inference_input(
        self,
        store_name: str,
        machine_type: str,
    ) -> Dict[str, str]:
        """Maps user inference inputs to internal feature representations."""
        return {
            "retail_chain": self.extract_retail_chain(store_name),
            "tipo_macchina": self.normalize_machine_type(machine_type),
        }
