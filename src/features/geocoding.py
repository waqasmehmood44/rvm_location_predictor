"""Location enrichment and geocoding utilities for Italian RVM sites.

Maps user-provided store location (CAP, Address) to Italian administrative
hierarchies (Comune, Provincia, Regione) and estimated geo-coordinates.
"""

from typing import Dict, Optional, Tuple
import pandas as pd


# Reference mapping for Italian CAP prefixes (first 2 digits) to Province and Region
ITALIAN_CAP_PREFIX_MAP: Dict[str, Tuple[str, str]] = {
    "00": ("RM", "LAZIO"),
    "01": ("VT", "LAZIO"),
    "02": ("RI", "LAZIO"),
    "03": ("FR", "LAZIO"),
    "04": ("LT", "LAZIO"),
    "05": ("TR", "UMBRIA"),
    "06": ("PG", "UMBRIA"),
    "07": ("SS", "SARDEGNA"),
    "08": ("NU", "SARDEGNA"),
    "09": ("CA", "SARDEGNA"),
    "10": ("TO", "PIEMONTE"),
    "11": ("AO", "VALLE D'AOSTA"),
    "12": ("CN", "PIEMONTE"),
    "13": ("VC", "PIEMONTE"),
    "14": ("AT", "PIEMONTE"),
    "15": ("AL", "PIEMONTE"),
    "16": ("GE", "LIGURIA"),
    "17": ("SV", "LIGURIA"),
    "18": ("IM", "LIGURIA"),
    "19": ("SP", "LIGURIA"),
    "20": ("MI", "LOMBARDIA"),
    "21": ("VA", "LOMBARDIA"),
    "22": ("CO", "LOMBARDIA"),
    "23": ("SO", "LOMBARDIA"),
    "24": ("BG", "LOMBARDIA"),
    "25": ("BS", "LOMBARDIA"),
    "26": ("CR", "LOMBARDIA"),
    "27": ("PV", "LOMBARDIA"),
    "28": ("NO", "PIEMONTE"),
    "29": ("PC", "EMILIA-ROMAGNA"),
    "30": ("VE", "VENETO"),
    "31": ("TV", "VENETO"),
    "32": ("BL", "VENETO"),
    "33": ("UD", "FRIULI-VENEZIA GIULIA"),
    "34": ("TS", "FRIULI-VENEZIA GIULIA"),
    "35": ("PD", "VENETO"),
    "36": ("VI", "VENETO"),
    "37": ("VR", "VENETO"),
    "38": ("TN", "TRENTINO-ALTO ADIGE"),
    "39": ("BZ", "TRENTINO-ALTO ADIGE"),
    "40": ("BO", "EMILIA-ROMAGNA"),
    "41": ("MO", "EMILIA-ROMAGNA"),
    "42": ("RE", "EMILIA-ROMAGNA"),
    "43": ("PR", "EMILIA-ROMAGNA"),
    "44": ("FE", "EMILIA-ROMAGNA"),
    "45": ("RO", "VENETO"),
    "46": ("MN", "LOMBARDIA"),
    "47": ("FC", "EMILIA-ROMAGNA"),
    "48": ("RA", "EMILIA-ROMAGNA"),
    "50": ("FI", "TOSCANA"),
    "51": ("PT", "TOSCANA"),
    "52": ("AR", "TOSCANA"),
    "53": ("SI", "TOSCANA"),
    "54": ("MS", "TOSCANA"),
    "55": ("LU", "TOSCANA"),
    "56": ("PI", "TOSCANA"),
    "57": ("LI", "TOSCANA"),
    "58": ("GR", "TOSCANA"),
    "59": ("PO", "TOSCANA"),
    "60": ("AN", "MARCHE"),
    "61": ("PU", "MARCHE"),
    "62": ("MC", "MARCHE"),
    "63": ("AP", "MARCHE"),
    "64": ("TE", "ABRUZZO"),
    "65": ("PE", "ABRUZZO"),
    "66": ("CH", "ABRUZZO"),
    "67": ("AQ", "ABRUZZO"),
    "70": ("BA", "PUGLIA"),
    "71": ("FG", "PUGLIA"),
    "72": ("BR", "PUGLIA"),
    "73": ("LE", "PUGLIA"),
    "74": ("TA", "PUGLIA"),
    "75": ("MT", "BASILICATA"),
    "80": ("NA", "CAMPANIA"),
    "81": ("CE", "CAMPANIA"),
    "82": ("BN", "CAMPANIA"),
    "83": ("AV", "CAMPANIA"),
    "84": ("SA", "CAMPANIA"),
    "85": ("PZ", "BASILICATA"),
    "86": ("CB", "MOLISE"),
    "87": ("CS", "CALABRIA"),
    "88": ("CZ", "CALABRIA"),
    "89": ("RC", "CALABRIA"),
    "90": ("PA", "SICILIA"),
    "91": ("TP", "SICILIA"),
    "92": ("AG", "SICILIA"),
    "93": ("CL", "SICILIA"),
    "94": ("EN", "SICILIA"),
    "95": ("CT", "SICILIA"),
    "96": ("SR", "SICILIA"),
    "97": ("RG", "SICILIA"),
    "98": ("ME", "SICILIA"),
}


class LocationEnricher:
    """Enriches postal codes and addresses with regional context and coordinates."""

    def __init__(self, geo_lookup_df: Optional[pd.DataFrame] = None):
        self.geo_lookup = {}
        if geo_lookup_df is not None:
            self.build_lookup_from_master(geo_lookup_df)

    def build_lookup_from_master(self, store_master_df: pd.DataFrame):
        """Indexes average coordinates and administrative entities by postal code (CAP)."""
        valid_coords = store_master_df.dropna(subset=["CAP", "latitudine", "longitudine"])
        for cap, group in valid_coords.groupby("CAP"):
            cap_str = str(cap).strip().zfill(5)
            self.geo_lookup[cap_str] = {
                "latitudine": group["latitudine"].median(),
                "longitudine": group["longitudine"].median(),
                "provincia": group["Provincia"].mode()[0] if "Provincia" in group else "UNKNOWN",
                "regione": group["Regione"].mode()[0] if "Regione" in group else "UNKNOWN",
                "comune": group["Comune"].mode()[0] if "Comune" in group else "UNKNOWN",
            }

    def infer_provincia_regione(self, postal_code: str) -> Tuple[str, str]:
        """Infers Italian Provincia and Regione from 5-digit CAP prefix."""
        clean_cap = str(postal_code).strip().zfill(5)
        prefix = clean_cap[:2]
        return ITALIAN_CAP_PREFIX_MAP.get(prefix, ("UNKNOWN", "UNKNOWN"))

    def enrich(
        self,
        postal_code: str,
        address: Optional[str] = None,
        comune: Optional[str] = None,
    ) -> Dict[str, any]:
        """Returns normalized geo-demographic features for a given location."""
        clean_cap = str(postal_code).strip().zfill(5)

        # 1. Direct match in lookup indexed from store master
        if clean_cap in self.geo_lookup:
            entry = self.geo_lookup[clean_cap].copy()
            entry["cap"] = clean_cap
            if comune:
                entry["comune"] = str(comune).strip().upper()
            return entry

        # 2. Rule-based administrative prefix fallback
        provincia, regione = self.infer_provincia_regione(clean_cap)

        return {
            "cap": clean_cap,
            "comune": str(comune).strip().upper() if comune else "UNKNOWN",
            "provincia": provincia,
            "regione": regione,
            "latitudine": 42.5,  # Central Italy default if completely unknown
            "longitudine": 12.5,
        }
