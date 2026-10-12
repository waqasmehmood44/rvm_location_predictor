"""
Intelligent Store Scale Auto-Estimator (Nielsen-Powered).
Automatically resolves store format (IPR, SUP, LIS, DIS), sales area (m2),
and cash registers for prospective locations across Italy.
"""
import os
import json
import re
import unicodedata
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from src.model.preprocessing import clean_brand_name, clean_retail_text

def extract_street_core(addr: str) -> str:
    """Extracts core street name, stripping numbers, prefixes, and noise."""
    if not addr or pd.isna(addr):
        return ""
    norm = clean_retail_text(addr)
    prefixes = ["VIA", "CORSO", "PIAZZA", "VIALE", "VICOLO", "STRADA", "LOC", "LOCALITA", "LARGO", "CONTRADA", "FRAZIONE", "SS", "SP"]
    tokens = norm.split()
    filtered = [tok for tok in tokens if tok not in prefixes and not tok.isdigit() and len(tok) > 1]
    return " ".join(filtered) if filtered else norm

class StoreScaleAutoEstimator:
    """
    Cascading Auto-Estimator for Italian retail store format and physical scale.
    Combines exact Nielsen 25,104 store census with empirical banner medians and semantic keywords.
    """
    def __init__(
        self,
        registry_path: str = "data/processed/nielsen_store_scale_registry.parquet",
        profiles_path: str = "data/processed/nielsen_banner_scale_profiles.json"
    ):
        self.address_lookup = {}
        self.cap_stores_lookup = {}
        self.banner_profiles = {}
        self.format_defaults = {}
        self.national_median = {
            "tipo_negozio": "SUP",
            "superficie_vendita": 750.0,
            "numero_casse": 4.0,
            "label": "Standard Supermarket (National Median)"
        }
        
        self._load_data(registry_path, profiles_path)

    def _load_data(self, registry_path: str, profiles_path: str):
        # 1. Load Address Registry
        if os.path.exists(registry_path):
            df = pd.read_parquet(registry_path)
            for _, r in df.iterrows():
                cap = str(r["cap_str"]).strip().zfill(5)
                street = str(r["street_core"]).strip().upper()
                entry = {
                    "tipo_negozio": str(r["tipo_negozio"]),
                    "superficie_vendita": float(r["superficie_vendita"]),
                    "numero_casse": float(r["numero_casse"]),
                    "insegna_raw": str(r["insegna_raw"]),
                    "clean_brand": str(r["clean_brand"])
                }
                self.address_lookup[(cap, street)] = entry
                
                if cap not in self.cap_stores_lookup:
                    self.cap_stores_lookup[cap] = []
                self.cap_stores_lookup[cap].append((street, entry))
                
        # 2. Load Banner Profiles & Defaults
        if os.path.exists(profiles_path):
            with open(profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.banner_profiles = data.get("banner_profiles", {})
                self.format_defaults = data.get("format_defaults", {})
                if "national_median" in data:
                    self.national_median = data["national_median"]

    def estimate_scale(
        self,
        store_name: str,
        address: str,
        cap: str,
        comune: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Estimates store format, sales area (m2), and cash registers using a 4-layer cascade.
        """
        clean_cap = str(cap).strip().zfill(5)
        street_core = extract_street_core(address).upper()
        clean_brand = clean_brand_name(store_name)
        
        # -------------------------------------------------------------
        # LAYER 1: Direct Match in Nielsen 25,104 Store Census
        # -------------------------------------------------------------
        # 1A. Exact (CAP, street_core)
        if (clean_cap, street_core) in self.address_lookup:
            rec = self.address_lookup[(clean_cap, street_core)]
            return {
                "tipo_negozio": rec["tipo_negozio"],
                "superficie_vendita": rec["superficie_vendita"],
                "numero_casse": rec["numero_casse"],
                "scale_source": "EXACT_NIELSEN_STORE_MATCH",
                "confidence": "VERY_HIGH",
                "label": f"Exact census match for {rec['insegna_raw']} ({rec['superficie_vendita']:.0f} m2, {rec['numero_casse']:.0f} casse)",
                "details": f"Found exact store address match in Nielsen census on {address}"
            }
            
        # 1B. Fuzzy street match in same postal district
        if clean_cap in self.cap_stores_lookup and len(street_core) >= 4:
            for n_street, rec in self.cap_stores_lookup[clean_cap]:
                if len(n_street) >= 4 and (n_street in street_core or street_core in n_street):
                    return {
                        "tipo_negozio": rec["tipo_negozio"],
                        "superficie_vendita": rec["superficie_vendita"],
                        "numero_casse": rec["numero_casse"],
                        "scale_source": "LOCAL_NIELSEN_STREET_MATCH",
                        "confidence": "HIGH",
                        "label": f"Local street match for {rec['insegna_raw']} (~{rec['superficie_vendita']:.0f} m2, {rec['numero_casse']:.0f} casse)",
                        "details": f"Matched neighborhood grocery site on {n_street}"
                    }

        # -------------------------------------------------------------
        # LAYER 2: Sub-Banner / Specific Keyword Inferences
        # -------------------------------------------------------------
        full_text = clean_retail_text(f"{store_name} {address}")
        store_tokens = clean_retail_text(store_name).split()
        
        # 2A. Explicit Hypermarket Sub-Banners (e.g. Ipercoop, Spazio Conad, Bennet, Panorama)
        if any(k in full_text for k in ["IPERCOOP", "SPAZIO CONAD", "IPERMERCATO", "PANORAMA"]):
            d = self.format_defaults.get("IPR", {"superficie_vendita": 3750.0, "numero_casse": 21.0})
            return {
                "tipo_negozio": "IPR",
                "superficie_vendita": d["superficie_vendita"],
                "numero_casse": d["numero_casse"],
                "scale_source": "SUB_BANNER_HYPERMARKET",
                "confidence": "HIGH",
                "label": "Hypermarket / Mall Anchor (~3,750 m2, 21 checkouts)",
                "details": f"Identified dedicated hypermarket banner in '{store_name}'"
            }
            
        # 2B-1. Dedicated Urban Proximity Supermarket Sub-Banners (e.g. TuDay Conad)
        if "TUDAY" in full_text:
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 450.0,
                "numero_casse": 4.0,
                "scale_source": "SUB_BANNER_TUDAY_PROXIMITY_HUB",
                "confidence": "HIGH",
                "label": "Urban Proximity Supermarket / TuDay Conad (~450 m2, 4 checkouts)",
                "details": f"Identified high-turnover urban proximity format 'TuDay Conad' in '{store_name}'"
            }

        # 2B. Explicit Convenience / Local Sub-Banners (e.g. Conad City, Carrefour Express, InCoop)
        if any(k in full_text for k in ["CONAD CITY", "CARREFOUR EXPRESS", "PAM LOCAL", "INCOOP", "TODIS", "BOTTEGA", "MINIMARKET"]):
            d = self.format_defaults.get("LIS", {"superficie_vendita": 280.0, "numero_casse": 2.0})
            return {
                "tipo_negozio": "LIS",
                "superficie_vendita": d["superficie_vendita"],
                "numero_casse": d["numero_casse"],
                "scale_source": "SUB_BANNER_CONVENIENCE",
                "confidence": "HIGH",
                "label": "Neighborhood / Convenience Store (~280 m2, 2 checkouts)",
                "details": f"Identified convenience format sub-banner in '{store_name}'"
            }
            
        # 2C. Explicit Superstore Sub-Banners
        if "SUPERSTORE" in full_text:
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 1850.0,
                "numero_casse": 9.0,
                "scale_source": "SUB_BANNER_SUPERSTORE",
                "confidence": "HIGH",
                "label": "Superstore (~1,850 m2, 9 checkouts)",
                "details": f"Identified Superstore format in '{store_name}'"
            }
            
        # 2D. Explicit Big-Box DIY / Home Improvement Anchors (Leroy Merlin, Tecnomat, Brico, OBI, Ikea)
        if any(k in full_text for k in ["LEROY MERLIN", "TECNOMAT", "BRICOMAN", "BRICO", "BRICOCENTER", "OBI", "IKEA"]):
            is_mega = any(k in full_text for k in ["LEROY", "TECNOMAT", "BRICOMAN", "IKEA"])
            sqm = 8500.0 if is_mega else 2500.0
            casse = 24.0 if is_mega else 8.0
            return {
                "tipo_negozio": "IPR" if is_mega else "SUP",
                "superficie_vendita": sqm,
                "numero_casse": casse,
                "scale_source": "SUB_BANNER_DIY_BIGBOX",
                "confidence": "HIGH",
                "label": f"DIY / Home Improvement Anchor (~{sqm:.0f} m2, {casse:.0f} checkouts)",
                "details": f"Identified destination DIY/Big-Box anchor in '{store_name}'"
            }

        # 2E. Regional Supermarket Cooperatives (Unicoop Firenze / Coop.fi, Coop Lombardia, Coop Alleanza, Nova Coop, Unicoop Tirreno)
        # In central/northern Italy, regional cooperative supermarkets without "InCoop" or "Ipercoop" are standard full-size supermarkets (~1,450 m2, 7 casse)
        if any(k in full_text for k in ["UNICOOP.FI", "COOP.FI", "COOP FI", "UNICOOP FIRENZE", "COOP LOMBARDIA", "COOP ALLEANZA", "NOVA COOP", "UNICOOP TIRRENO", "COOP CENTRO"]):
            if not any(c in full_text for c in ["INCOOP", "MINI", "LOCAL", "IPER"]):
                return {
                    "tipo_negozio": "SUP",
                    "superficie_vendita": 1450.0,
                    "numero_casse": 7.0,
                    "scale_source": "SUB_BANNER_REGIONAL_COOP",
                    "confidence": "HIGH",
                    "label": "Regional Cooperative Supermarket (~1,450 m2, 7 checkouts)",
                    "details": f"Identified regional supermarket cooperative branch in '{store_name}'"
                }

        # 2F. Carrefour Market (Standard Supermarket vs Express Convenience)
        if any(k in full_text for k in ["CARREFOUR MARKET", "MARKET CARREFOUR"]):
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 1250.0,
                "numero_casse": 6.0,
                "scale_source": "SUB_BANNER_CARREFOUR_MARKET",
                "confidence": "HIGH",
                "label": "Carrefour Market Supermarket (~1,250 m2, 6 checkouts)",
                "details": f"Identified Carrefour Market supermarket format in '{store_name}'"
            }

        # 2G. Conad Superstore / Standard Conad Supermercato
        if "CONAD SUPERSTORE" in full_text:
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 1850.0,
                "numero_casse": 9.0,
                "scale_source": "SUB_BANNER_CONAD_SUPERSTORE",
                "confidence": "HIGH",
                "label": "Conad Superstore (~1,850 m2, 9 checkouts)",
                "details": f"Identified Conad Superstore format in '{store_name}'"
            }
        elif "CONAD" in full_text and not any(c in full_text for c in ["CITY", "MARGHERITA", "SPAZIO", "PETSTORE", "SAPORI"]):
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 950.0,
                "numero_casse": 5.0,
                "scale_source": "SUB_BANNER_CONAD_SUPERMARKET",
                "confidence": "HIGH",
                "label": "Standard Conad Supermarket (~950 m2, 5 checkouts)",
                "details": f"Identified standard Conad supermarket format in '{store_name}'"
            }

        # 2H. Despar Spar Hierarchy (Interspar / Eurospar / Despar)
        if any(k in full_text for k in ["INTERSPAR", "IPERSPAR"]):
            return {
                "tipo_negozio": "IPR",
                "superficie_vendita": 3500.0,
                "numero_casse": 18.0,
                "scale_source": "SUB_BANNER_INTERSPAR",
                "confidence": "HIGH",
                "label": "Interspar Hypermarket (~3,500 m2, 18 checkouts)",
                "details": f"Identified Interspar hypermarket format in '{store_name}'"
            }
        elif "EUROSPAR" in full_text:
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 1400.0,
                "numero_casse": 7.0,
                "scale_source": "SUB_BANNER_EUROSPAR",
                "confidence": "HIGH",
                "label": "Eurospar Supermarket (~1,400 m2, 7 checkouts)",
                "details": f"Identified Eurospar supermarket format in '{store_name}'"
            }

        # 2I. Pam Panorama Hierarchy
        if any(k in full_text for k in ["PAM PANORAMA", "PANORAMA"]):
            return {
                "tipo_negozio": "IPR",
                "superficie_vendita": 3500.0,
                "numero_casse": 18.0,
                "scale_source": "SUB_BANNER_PANORAMA",
                "confidence": "HIGH",
                "label": "Panorama Hypermarket (~3,500 m2, 18 checkouts)",
                "details": f"Identified Panorama hypermarket format in '{store_name}'"
            }
        elif "PAM" in full_text and not any(c in full_text for c in ["LOCAL", "CITY"]):
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 1100.0,
                "numero_casse": 6.0,
                "scale_source": "SUB_BANNER_PAM_SUPERMARKET",
                "confidence": "HIGH",
                "label": "Pam Supermarket (~1,100 m2, 6 checkouts)",
                "details": f"Identified standard Pam supermarket format in '{store_name}'"
            }

        # -------------------------------------------------------------
        # LAYER 3: Banner Empirical Profile (Learned from 25,104 Stores)
        # -------------------------------------------------------------
        if clean_brand != "INDIPENDENTE" and clean_brand in self.banner_profiles:
            prof = self.banner_profiles[clean_brand]
            tipo = prof["tipo_negozio"]
            sqm = prof["superficie_vendita"]
            casse = prof["numero_casse"]
            return {
                "tipo_negozio": tipo,
                "superficie_vendita": sqm,
                "numero_casse": casse,
                "scale_source": f"BANNER_PROFILE_{clean_brand}",
                "confidence": "HIGH",
                "label": f"{clean_brand} Chain Footprint (~{sqm:.0f} m2, {casse:.0f} casse)",
                "details": f"Empirical median across {prof['stores_sampled']} {clean_brand} stores in Italy"
            }
        
        # Hypermarkets / Shopping Centers
        if any(k in full_text for k in ["IPERMERCATO", "IPERCOOP", "SPAZIO CONAD", "IPER", "PANORAMA"]):
            d = self.format_defaults.get("IPR", {"superficie_vendita": 3750.0, "numero_casse": 21.0})
            return {
                "tipo_negozio": "IPR",
                "superficie_vendita": d["superficie_vendita"],
                "numero_casse": d["numero_casse"],
                "scale_source": "KEYWORD_HYPERMARKET",
                "confidence": "MEDIUM",
                "label": "Hypermarket / Mall Anchor (~3,750 m2, 21 checkouts)",
                "details": "Identified hypermarket keyword ('Iper' / 'Spazio') in store name"
            }
            
        if any(k in full_text for k in ["CENTRO COMMERCIALE", "SHOPPING CENTER", "MALL", "PARCO COMMERCIALE"]):
            return {
                "tipo_negozio": "IPR",
                "superficie_vendita": 3200.0,
                "numero_casse": 16.0,
                "scale_source": "KEYWORD_SHOPPING_CENTER",
                "confidence": "MEDIUM",
                "label": "Shopping Center Anchor (~3,200 m2, 16 checkouts)",
                "details": "Identified shopping mall or commercial center in address"
            }
            
        # Superstores
        if "SUPERSTORE" in full_text:
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": 1800.0,
                "numero_casse": 9.0,
                "scale_source": "KEYWORD_SUPERSTORE",
                "confidence": "MEDIUM",
                "label": "Superstore / Large Supermarket (~1,800 m2, 9 checkouts)",
                "details": "Identified 'Superstore' keyword in store name"
            }
            
        # Discounts (Word boundary check to prevent substring collisions like 'MERLIN STR' matching 'IN S')
        discount_kws = ["DISCOUNT", "D PIU", "IN S", "PRIX", "TUODI", "EKOM"]
        if any(re.search(r"\b" + re.escape(k) + r"\b", full_text) for k in discount_kws):
            d = self.format_defaults.get("DIS", {"superficie_vendita": 740.0, "numero_casse": 4.0})
            return {
                "tipo_negozio": "DIS",
                "superficie_vendita": d["superficie_vendita"],
                "numero_casse": d["numero_casse"],
                "scale_source": "KEYWORD_DISCOUNT",
                "confidence": "MEDIUM",
                "label": "Discount Store (~740 m2, 4 checkouts)",
                "details": "Identified discount grocery keyword in store name"
            }
            
        # Small neighborhood markets
        if any(k in full_text for k in ["CITY", "EXPRESS", "LOCAL", "BOTTEGA", "MINIMARKET", "PICCOLO", "PUNTO"]):
            d = self.format_defaults.get("LIS", {"superficie_vendita": 250.0, "numero_casse": 2.0})
            return {
                "tipo_negozio": "LIS",
                "superficie_vendita": d["superficie_vendita"],
                "numero_casse": d["numero_casse"],
                "scale_source": "KEYWORD_NEIGHBORHOOD_MARKET",
                "confidence": "MEDIUM",
                "label": "Neighborhood Market / Convenience (~250 m2, 2 checkouts)",
                "details": "Identified convenience / local market keyword in store name"
            }
            
        # Standard supermarkets
        if any(k in full_text for k in ["SUPERMERCATO", "SUPERMERCATI", "MARKET"]):
            d = self.format_defaults.get("SUP", {"superficie_vendita": 750.0, "numero_casse": 4.0})
            return {
                "tipo_negozio": "SUP",
                "superficie_vendita": d["superficie_vendita"],
                "numero_casse": d["numero_casse"],
                "scale_source": "KEYWORD_SUPERMARKET",
                "confidence": "MEDIUM",
                "label": "Standard Supermarket (~750 m2, 4 checkouts)",
                "details": "Identified supermarket keyword in store name"
            }

        # -------------------------------------------------------------
        # LAYER 4: National Median Supermarket Baseline
        # -------------------------------------------------------------
        return {
            "tipo_negozio": self.national_median.get("tipo_negozio", "SUP"),
            "superficie_vendita": float(self.national_median.get("superficie_vendita", 750.0)),
            "numero_casse": float(self.national_median.get("numero_casse", 4.0)),
            "scale_source": "NATIONAL_MEDIAN_DEFAULT",
            "confidence": "MODERATE",
            "label": "Standard Supermarket (National Median: ~750 m2, 4 checkouts)",
            "details": "Default national median applied for unlisted grocery location"
        }
