"""
Inference module for new prospective retail locations.
Takes user inputs:
  1. Store name
  2. Address
  3. CAP
  4. Machine type
  5. Comune (optional - auto-detected if omitted)
  6. Store Format / Scale (Improvement 2: Superette / Standard / Superstore / Hypermarket / Discount)

Features:
- Level 1: Multi-municipality CAP disambiguation & street-level geocoding resolution
- Level 2: Store scale and foot-traffic proxy (surface sqm, checkouts, store format)
- Level 3: Regional retail group and chain alliance expansion (banner & alliance embeddings)
- Level 4: Spatial catchment & competitive density enrichment (network distance, cannibalization, potential scoring & recommendations)
- Marginalization across a representative 7-day week for steady-state investment expectations.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import re
import unicodedata
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union

from src.model.preprocessing import (
    RVMPreprocessor, 
    clean_brand_name, 
    clean_retail_text,
    infer_retail_group, 
    infer_nielsen_area
)
from src.model.network import MultiTaskRVMNet
from src.spatial.catchment import SpatialCatchmentAnalyzer
from src.spatial.store_scale import StoreScaleAutoEstimator
from src.features.geocoding import ITALIAN_CAP_PREFIX_MAP
from src.model.scoring import (
    calculate_location_potential_score,
    determine_recommendation_tier,
    calculate_operational_bounds,
    calculate_probability_factors,
    evaluate_prediction_confidence,
    calculate_catchment_realization_factor,
    evaluate_location_suitability,
    generate_predictive_diagnostics
)
from src.business.rewards import (
    CommercialRewardsEngine, 
    REWARD_MENU_CHOICES, 
    prompt_reward_policy_selection,
    create_custom_reward_policy
)

def normalize_text(text: Any) -> str:
    """Standardizes string for case/space-insensitive matching."""
    if text is None or pd.isna(text):
        return ""
    t = str(text).strip().upper()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_street_core(address_str: str) -> str:
    """Extracts core street name (e.g. 'VIA FERROVIA 66' -> 'FERROVIA')."""
    norm = normalize_text(address_str)
    prefixes = ["VIA", "CORSO", "PIAZZA", "VIALE", "VICOLO", "STRADA", "LOC", "LOCALITA", "LARGO"]
    tokens = norm.split()
    filtered = [tok for tok in tokens if tok not in prefixes and not tok.isdigit()]
    return " ".join(filtered) if filtered else norm

# Preset configurations for store scales based on empirical Italian retail benchmarks
STORE_FORMAT_PRESETS = {
    "1": {
        "label": "Small Supermarket / Neighborhood Market (LIS/SUP, ~250-450 m2, 2-3 checkouts)",
        "tipo_negozio": "LIS",
        "superficie_vendita": 350.0,
        "numero_casse": 2.0
    },
    "2": {
        "label": "Standard Supermarket (SUP, ~750-1,100 m2, 4-5 checkouts)",
        "tipo_negozio": "SUP",
        "superficie_vendita": 750.0,
        "numero_casse": 4.0
    },
    "3": {
        "label": "Superstore / Large Supermarket (SUP, ~1,500-2,200 m2, 8-10 checkouts)",
        "tipo_negozio": "SUP",
        "superficie_vendita": 1850.0,
        "numero_casse": 9.0
    },
    "4": {
        "label": "Hypermarket / Mall Anchor (IPR, > 3,000 m2, 18+ checkouts)",
        "tipo_negozio": "IPR",
        "superficie_vendita": 3750.0,
        "numero_casse": 21.0
    },
    "5": {
        "label": "Discount Store (DIS, Eurospin, Lidl, MD, ~750-900 m2, 4-5 checkouts)",
        "tipo_negozio": "DIS",
        "superficie_vendita": 750.0,
        "numero_casse": 4.0
    }
}

# Recognized Coripet RVM hardware models & manufacturer families
VALID_MACHINE_TYPES = [
    "DAINT_G",
    "RIPET_C-FRONT",
    "DAINT_A-BIS",
    "RIPET_C-TER",
    "RIPET_V2",
    "RIPET_C-BIS",
    "RIPET_SMARPET",
    "DAINT_B-BIS",
    "RVM_MOD.E",
    "RVM_MOD.E_NUOVA",
    "DAINT_B",
    "RIPET_V1",
    "SUPERMARPET01S",
    "DAINT_Y_DRS",
]
VALID_MACHINE_FAMILIES = ["DAINT", "RIPET", "RVM", "EURVEN", "SUPERMARPET", "SMARPET"]

def normalize_comune_token(text: Any) -> str:
    """Standardizes Italian comune string for accent/case/punctuation-insensitive lookup."""
    if not text or pd.isna(text):
        return ""
    n = unicodedata.normalize("NFKD", str(text)).encode("ASCII", "ignore").decode("utf-8")
    return re.sub(r"[^A-Z0-9]", "", n.upper())

_KNOWN_ITALIAN_COMUNI = None

def get_known_italian_comuni() -> set:
    """Loads and caches the complete official set of Italian municipalities (ISTAT/census)."""
    global _KNOWN_ITALIAN_COMUNI
    if _KNOWN_ITALIAN_COMUNI is not None:
        return _KNOWN_ITALIAN_COMUNI
    
    comuni = set()
    pop_path = "data/raw/population.csv"
    if os.path.exists(pop_path):
        try:
            df = pd.read_csv(pop_path, sep=";", encoding="utf-8")
            for c in df["Comune"].dropna():
                tok = normalize_comune_token(c)
                if tok:
                    comuni.add(tok)
        except Exception:
            pass
            
    spat_path = "data/processed/coripet_rvm_spatial_network.parquet"
    if os.path.exists(spat_path):
        try:
            df = pd.read_parquet(spat_path, columns=["Comune"]).dropna()
            for c in df["Comune"].unique():
                tok = normalize_comune_token(c)
                if tok:
                    comuni.add(tok)
        except Exception:
            pass
            
    _KNOWN_ITALIAN_COMUNI = comuni
    return _KNOWN_ITALIAN_COMUNI

def validate_store_name(store_name: Any) -> str:
    """Validates store name."""
    if store_name is None:
        raise ValueError("Store name is required. Please enter a valid store name.")
    s_str = str(store_name).strip()
    if not s_str or len(s_str) < 2:
        raise ValueError("Store name is required. Please enter a valid store name.")
    return s_str

def validate_address(address: Any) -> str:
    """Validates street address input."""
    if address is None:
        raise ValueError("Address is required. Please provide a valid street address (e.g. 'Via Roma 10').")
    a_str = str(address).strip()
    if not a_str or len(a_str) < 3:
        raise ValueError("Address is required. Please provide a valid street address (e.g. 'Via Roma 10').")
    if not re.search(r"[A-Za-z]", a_str):
        raise ValueError("Invalid address: Address must contain street letters.")
    return a_str

def validate_cap(cap: Any) -> str:
    """
    Validates Italian CAP (Codice di Avviamento Postale).
    Supports standard 5-digit CAPs as well as postal codes with fewer digits (e.g. '7100', '185', '123')
    commonly stored without leading zeros, normalizing them via zero-padding.
    Raises ValueError if invalid.
    """
    if cap is None:
        raise ValueError("Location not found: CAP is required. Please enter a valid postal code.")
    cap_str = str(cap).strip()
    if not cap_str:
        raise ValueError("Location not found: CAP is required. Please enter a valid postal code.")
    if not cap_str.isdigit():
        raise ValueError(
            f"Location not found: Invalid CAP '{cap_str}'. Postal code (CAP) must contain numeric digits (e.g. '50019', '20121', '7100')."
        )
    if len(cap_str) > 5:
        raise ValueError(
            f"Location not found: Invalid CAP '{cap_str}'. Postal code cannot exceed 5 digits."
        )
    cap_padded = cap_str.zfill(5)
    prefix = cap_padded[:2]
    if prefix not in ITALIAN_CAP_PREFIX_MAP:
        raise ValueError(
            f"Location not found: Invalid CAP prefix '{prefix}' in '{cap_str}'. Prefix is not a valid Italian postal distribution area."
        )
    return cap_padded

def validate_comune(comune: Any, address: Optional[str] = None) -> Optional[str]:
    """
    Validates Italian Comune (municipality).
    Rejects street addresses passed as Comune or empty/bogus strings.
    """
    if comune is None or pd.isna(comune):
        return None
    c_str = str(comune).strip()
    if not c_str:
        return None
    
    c_norm = normalize_text(c_str)
    
    # Check if user passed identical street address string as Comune
    if address:
        addr_norm = normalize_text(address)
        if c_norm == addr_norm:
            raise ValueError(
                f"Location not found: Comune cannot be identical to the street address ('{c_str}'). Please provide the municipality name (e.g. Milano, Firenze)."
            )

    # Check if user entered a street address instead of a town name
    street_prefixes = [
        "VIA", "CORSO", "PIAZZA", "VIALE", "VICOLO", "STRADA", "LOC", 
        "LOCALITA", "LARGO", "CONTRADA", "PIAZZALE", "BORGO", "FRAZIONE"
    ]
    tokens = c_norm.split()
    first_token = tokens[0] if tokens else ""
    if first_token in street_prefixes:
        raise ValueError(
            f"Location not found: Invalid Comune '{c_str}'. '{c_str}' appears to be a street address, not an Italian municipality name."
        )
            
    # Must contain letters, not purely numbers or symbols
    if not re.search(r"[A-Za-z]", c_str):
        raise ValueError(
            f"Location not found: Invalid Comune '{c_str}'. Municipality name must contain valid letters."
        )
        
    # Check if recognized in known Italian municipalities
    known = get_known_italian_comuni()
    c_token = normalize_comune_token(c_str)
    if known and c_token not in known:
        raise ValueError(
            f"Location not found: Municipality '{c_str}' was not recognized as a valid Italian Comune. Please verify the municipality name."
        )
        
    return c_str

def validate_machine_type(machine_type: Any) -> str:
    """
    Validates machine hardware model against Coripet hardware families.
    Raises ValueError if invalid.
    """
    if machine_type is None:
        raise ValueError("Machine type is required. Please enter a valid Coripet hardware model.")
    m_str = str(machine_type).strip()
    if not m_str:
        raise ValueError("Machine type is required. Please enter a valid Coripet hardware model.")
    
    m_upper = m_str.upper().replace(" ", "_")
    
    # Check exact match in valid types (case-insensitive)
    for vm in VALID_MACHINE_TYPES:
        if m_upper == vm.upper():
            return m_str
            
    # Check recognized hardware family prefix
    for fam in VALID_MACHINE_FAMILIES:
        if fam in m_upper:
            return m_str
            
    valid_examples = ", ".join(VALID_MACHINE_TYPES[:6])
    raise ValueError(
        f"Invalid machine type '{m_str}'. Must be a recognized Coripet machine model (e.g. {valid_examples})."
    )

def calculate_lifecycle_horizon(
    underwritten_monthly_profile: List[Dict[str, Any]],
    expected_daily_bottiglie: float,
    expected_annual_bottiglie: float,
    expected_daily_presenze: float,
    go_live_month: Optional[int] = None,
    go_live_year: Optional[int] = None,
    enable_rampup: bool = True
) -> Dict[str, Any]:
    """
    Computes a full 12-month rolling lifecycle roadmap and 3-year commercial horizon.
    Anchored to the mature steady-state baseline without altering core predictions.
    Accounts for:
    1. Realistic customer discovery & onboarding ramp-up (Months 1 to 3)
    2. Exact launch calendar month timing and intra-year seasonality
    3. Empirical nationwide network awareness growth drift (+5.5% YoY)
    """
    now = pd.Timestamp.now()
    start_m = int(go_live_month) if (go_live_month and 1 <= int(go_live_month) <= 12) else int(now.month)
    start_y = int(go_live_year) if (go_live_year and int(go_live_year) >= 2024) else int(now.year)
    
    ANNUAL_NETWORK_GROWTH_RATE = 0.055  # 5.5% annual brand & app adoption tailwind
    delta_years = max(0, start_y - 2026)
    macro_multiplier = (1.0 + ANNUAL_NETWORK_GROWTH_RATE) ** delta_years
    
    # Ramp-up maturation curve for lifecycle months 1 to 12
    # Month 1: 65% (initial discovery & app download)
    # Month 2: 80% (habit formation & bag accumulation)
    # Month 3: 92% (adoption consolidation)
    # Month 4+: 100% (mature steady-state throughput)
    rampup_factors = {
        1: 0.65,
        2: 0.80,
        3: 0.92
    }
    
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    profile_lookup = {m["month_num"]: m for m in underwritten_monthly_profile}
    
    rolling_12m = []
    for k in range(1, 13):
        cal_m = (start_m - 1 + (k - 1)) % 12 + 1
        cal_y = start_y + ((start_m - 1 + (k - 1)) // 12)
        m_name = month_names.get(cal_m, f"Month {cal_m}")
        
        m_base = profile_lookup.get(cal_m, {})
        base_daily_b = float(m_base.get("daily_bottiglie", expected_daily_bottiglie))
        base_daily_p = float(m_base.get("daily_presenze", expected_daily_presenze))
        days_in_m = int(m_base.get("days_in_month", 30))
        season_ctx = m_base.get("context", "Standard Seasonal Rhythm")
        
        mat_factor = rampup_factors.get(k, 1.0) if enable_rampup else 1.0
        
        daily_b = round(base_daily_b * mat_factor * macro_multiplier, 1)
        daily_p = round(base_daily_p * np.sqrt(mat_factor) * np.sqrt(macro_multiplier), 1)
        monthly_b = int(round(daily_b * days_in_m))
        
        if k == 1:
            stage_desc = "Onboarding & Discovery (Ramp-up)"
        elif k == 2:
            stage_desc = "Habit Formation (Ramp-up)"
        elif k == 3:
            stage_desc = "Consolidation Phase"
        else:
            stage_desc = "Mature Steady-State"
            
        rolling_12m.append({
            "lifecycle_month": k,
            "lifecycle_label": f"M{k:02d} ({m_name[:3]} '{str(cal_y)[2:]})",
            "calendar_month_num": cal_m,
            "calendar_month_name": m_name,
            "calendar_year": cal_y,
            "days_in_month": days_in_m,
            "maturity_factor": mat_factor,
            "maturity_pct": f"{int(round(mat_factor * 100))}%",
            "daily_bottiglie": daily_b,
            "daily_presenze": daily_p,
            "monthly_target_bottiglie": monthly_b,
            "seasonal_context": season_ctx,
            "lifecycle_stage": stage_desc
        })
        
    year1_total_bottles = sum(m["monthly_target_bottiglie"] for m in rolling_12m)
    year1_total_days = sum(m["days_in_month"] for m in rolling_12m)
    year1_avg_daily_bottles = round(year1_total_bottles / max(1, year1_total_days), 1)
    
    # 3-Year Commercial Horizon
    y2_growth = (1.0 + ANNUAL_NETWORK_GROWTH_RATE) ** (delta_years + 1)
    y3_growth = (1.0 + ANNUAL_NETWORK_GROWTH_RATE) ** (delta_years + 2)
    
    year2_annual_bottles = int(round(expected_annual_bottiglie * y2_growth))
    year2_daily_bottles = round(expected_daily_bottiglie * y2_growth, 1)
    
    year3_annual_bottles = int(round(expected_annual_bottiglie * y3_growth))
    year3_daily_bottles = round(expected_daily_bottiglie * y3_growth, 1)
    
    multi_year = [
        {
            "horizon": "Year 1 (Adoption & Ramp-Up)",
            "annual_bottles": year1_total_bottles,
            "daily_average": year1_avg_daily_bottles,
            "capacity_utilization": f"{int(round((year1_total_bottles / max(1, expected_annual_bottiglie)) * 100))}%",
            "operational_focus": "Customer app onboarding, store lobby awareness, logistics route calibration"
        },
        {
            "horizon": "Year 2 (Mature Steady-State)",
            "annual_bottles": year2_annual_bottles,
            "daily_average": year2_daily_bottles,
            "capacity_utilization": "100%",
            "operational_focus": "Full commercial steady-state, optimized bin clearing frequency, retailer voucher redemption"
        },
        {
            "horizon": "Year 3+ (Optimized Network Anchor)",
            "annual_bottles": year3_annual_bottles,
            "daily_average": year3_daily_bottles,
            "capacity_utilization": f"{int(round((year3_annual_bottles / max(1, expected_annual_bottiglie)) * 100))}%",
            "operational_focus": "Solidified civic recycling routine, compounding Coripet national loyalty user base"
        }
    ]
    
    return {
        "launch_month_num": start_m,
        "launch_month_name": month_names.get(start_m, f"Month {start_m}"),
        "launch_year": start_y,
        "rampup_enabled": enable_rampup,
        "macro_annual_growth_rate": ANNUAL_NETWORK_GROWTH_RATE,
        "year1_rolling_profile": rolling_12m,
        "year1_total_bottles": year1_total_bottles,
        "year1_avg_daily_bottles": year1_avg_daily_bottles,
        "multi_year_forecast": multi_year
    }

class RVMNewLocationPredictor:
    def __init__(
        self, 
        models_dir: str = "models",
        data_path: str = "data/processed/coripet_rvm_spatial_network.parquet",
        fallback_data_path: str = "data/processed/modeling_daily_rvm.parquet"
    ):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Preprocessor
        preprocessor_path = os.path.join(models_dir, "preprocessing", "preprocessor.joblib")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Train model first!")
        self.preprocessor = RVMPreprocessor.load(preprocessor_path)
        
        # 2. Load Model
        model_path = os.path.join(models_dir, "best_model", "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Train model first!")
            
        self.model = MultiTaskRVMNet(
            num_numeric_features=len(self.preprocessor.num_features),
            cat_cardinalities=self.preprocessor.get_cat_cardinalities(),
            cat_features=self.preprocessor.cat_features,
            hidden_dims=(256, 128, 64),
            head_dim=32,
            dropout=0.0 # Inference mode
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # 2b. Load Stacking Ensemble (LightGBM) if available
        self.lgb_model_b = None
        self.lgb_model_p = None
        self.w_b = 0.90
        self.w_p = 0.65
        try:
            import lightgbm as lgb
            lgb_b_path = os.path.join(models_dir, "ensemble", "lgb_bottiglie.txt")
            lgb_p_path = os.path.join(models_dir, "ensemble", "lgb_presenze.txt")
            cfg_path = os.path.join(models_dir, "ensemble", "ensemble_config.json")
            if os.path.exists(lgb_b_path) and os.path.exists(lgb_p_path):
                self.lgb_model_b = lgb.Booster(model_file=lgb_b_path)
                self.lgb_model_p = lgb.Booster(model_file=lgb_p_path)
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r") as f:
                        ens_cfg = json.load(f)
                        self.w_b = float(ens_cfg.get("blend_weight_bottiglie_lgb", 0.90))
                        self.w_p = float(ens_cfg.get("blend_weight_presenze_lgb", 0.65))
        except Exception:
            self.lgb_model_b = None
            self.lgb_model_p = None

        # 3. Spatial & Catchment Engine (Level 4)
        self.catchment_analyzer = SpatialCatchmentAnalyzer()
        self.scale_estimator = StoreScaleAutoEstimator()
        self.rewards_engine = CommercialRewardsEngine()
        
        # 4. Build spatial & geocoding indices
        self.cap_comune_lookup = {}
        self.street_address_lookup = {}
        self.comune_lookup = {}
        self.cap_summary_lookup = {}
        self.provincia_summary_lookup = {}
        self.regione_summary_lookup = {}
        actual_path = data_path if os.path.exists(data_path) else fallback_data_path
        self._build_spatial_indices(actual_path)
        
    def _build_spatial_indices(self, data_path: str):
        """Builds granular spatial & demographic indices from verified registry."""
        if not os.path.exists(data_path):
            return
            
        cols = [
            "Cap", "Comune", "Provincia", "Regione", "Indirizzo",
            "Latitudine", "Longitudine", "population", "surface", 
            "population_density", "PIL", "PIL_per_capita"
        ]
        sample_cols = pd.read_parquet(data_path).columns
        read_cols = [c for c in cols if c in sample_cols]
        if "PF" in sample_cols:
            read_cols.insert(0, "PF")
            
        df = pd.read_parquet(data_path, columns=read_cols).drop_duplicates(subset=["PF"] if "PF" in read_cols else None)
        
        def clean_prov(row):
            p = row.get("Provincia")
            if pd.isna(p) or str(p).strip().upper() in ("NAN", "UNKNOWN", ""):
                reg = str(row.get("Regione", "")).strip().upper()
                com = normalize_text(row.get("Comune", ""))
                if reg == "CAMPANIA" and ("VOLLA" in com or "SAN GENNARO" in com or "NAPOLI" in com or "CERCOLA" in com or "POZZUOLI" in com):
                    return "NA"
            return str(p) if pd.notna(p) else "UNKNOWN"
            
        for _, row in df.iterrows():
            cap_val = row.get("Cap")
            cap_str = str(int(cap_val)).zfill(5) if pd.notna(cap_val) else ""
            comune_raw = str(row.get("Comune", "UNKNOWN"))
            comune_norm = normalize_text(comune_raw)
            prov_str = clean_prov(row)
            reg_str = str(row.get("Regione", "UNKNOWN"))
            
            entry = {
                "Comune": comune_raw,
                "Provincia": prov_str,
                "Regione": reg_str,
                "Latitudine": float(row.get("Latitudine", 41.9)),
                "Longitudine": float(row.get("Longitudine", 12.5)),
                "population": float(row.get("population", 35000.0)),
                "surface": float(row.get("surface", 45.0)),
                "population_density": float(row.get("population_density", 700.0)),
                "PIL": float(row.get("PIL", 7e8)),
                "PIL_per_capita": float(row.get("PIL_per_capita", 16000.0))
            }
            
            if cap_str and comune_norm:
                self.cap_comune_lookup[(cap_str, comune_norm)] = entry
                
            addr_raw = str(row.get("Indirizzo", ""))
            street_core = extract_street_core(addr_raw)
            if cap_str and street_core:
                self.street_address_lookup[(cap_str, street_core)] = entry
                
            if comune_norm and comune_norm not in self.comune_lookup:
                self.comune_lookup[comune_norm] = entry
                
        cap_groups = df.groupby("Cap")
        for cap_val, grp in cap_groups:
            cap_str = str(int(cap_val)).zfill(5)
            first_row = grp.iloc[0]
            self.cap_summary_lookup[cap_str] = {
                "Comune": str(first_row.get("Comune", "UNKNOWN")),
                "Provincia": clean_prov(first_row),
                "Regione": str(first_row.get("Regione", "UNKNOWN")),
                "Latitudine": float(grp["Latitudine"].astype(float).mean()),
                "Longitudine": float(grp["Longitudine"].astype(float).mean()),
                "population": float(grp["population"].astype(float).median()),
                "surface": float(grp["surface"].astype(float).median()),
                "population_density": float(grp["population_density"].astype(float).median()),
                "PIL": float(grp["PIL"].astype(float).median()),
                "PIL_per_capita": float(grp["PIL_per_capita"].astype(float).median())
            }

        prov_groups = df.groupby("Provincia")
        for prov_val, grp in prov_groups:
            prov_str = str(prov_val).strip().upper()
            if prov_str and prov_str not in ("UNKNOWN", "NAN", ""):
                first_row = grp.iloc[0]
                self.provincia_summary_lookup[prov_str] = {
                    "Comune": str(first_row.get("Comune", "UNKNOWN")),
                    "Provincia": prov_str,
                    "Regione": str(first_row.get("Regione", "UNKNOWN")),
                    "Latitudine": float(grp["Latitudine"].astype(float).mean()),
                    "Longitudine": float(grp["Longitudine"].astype(float).mean()),
                    "population": float(grp["population"].astype(float).median()),
                    "surface": float(grp["surface"].astype(float).median()),
                    "population_density": float(grp["population_density"].astype(float).median()),
                    "PIL": float(grp["PIL"].astype(float).median()),
                    "PIL_per_capita": float(grp["PIL_per_capita"].astype(float).median())
                }

        reg_groups = df.groupby("Regione")
        for reg_val, grp in reg_groups:
            reg_str = str(reg_val).strip().upper()
            if reg_str and reg_str not in ("UNKNOWN", "NAN", ""):
                first_row = grp.iloc[0]
                self.regione_summary_lookup[reg_str] = {
                    "Comune": str(first_row.get("Comune", "UNKNOWN")),
                    "Provincia": clean_prov(first_row),
                    "Regione": reg_str,
                    "Latitudine": float(grp["Latitudine"].astype(float).mean()),
                    "Longitudine": float(grp["Longitudine"].astype(float).mean()),
                    "population": float(grp["population"].astype(float).median()),
                    "surface": float(grp["surface"].astype(float).median()),
                    "population_density": float(grp["population_density"].astype(float).median()),
                    "PIL": float(grp["PIL"].astype(float).median()),
                    "PIL_per_capita": float(grp["PIL_per_capita"].astype(float).median())
                }

    def resolve_location(
        self, 
        address: str, 
        cap: str, 
        comune: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resolves spatial and demographic attributes using multi-level matching."""
        clean_cap = validate_cap(cap)
        clean_addr = normalize_text(address)
        street_core = extract_street_core(address)
        if comune:
            comune = validate_comune(comune, address=address)
            
        # Cross-validation: Check regional consistency between CAP and Comune if both provided
        prefix = clean_cap[:2]
        inferred_prov, inferred_reg = ITALIAN_CAP_PREFIX_MAP.get(prefix, ("UNKNOWN", "UNKNOWN"))
        if comune:
            comune_norm = normalize_text(comune)
            if comune_norm in self.comune_lookup:
                known_reg = str(self.comune_lookup[comune_norm].get("Regione", "")).strip().upper()
                if known_reg and inferred_reg != "UNKNOWN" and known_reg != inferred_reg:
                    # If user entered fewer than 5 digits (e.g. '2012' for Milano), check if Comune has a known CAP starting with those digits
                    raw_c_str = str(cap).strip()
                    resolved_alt = False
                    if len(raw_c_str) < 5 and raw_c_str.isdigit():
                        for (c_cap, c_com) in self.cap_comune_lookup.keys():
                            if c_com == comune_norm and c_cap.startswith(raw_c_str):
                                clean_cap = c_cap
                                prefix = clean_cap[:2]
                                inferred_prov, inferred_reg = ITALIAN_CAP_PREFIX_MAP.get(prefix, ("UNKNOWN", "UNKNOWN"))
                                resolved_alt = True
                                break
                    if not resolved_alt and known_reg != inferred_reg:
                        raise ValueError(
                            f"Location not found: Regional conflict between CAP '{clean_cap}' ({inferred_reg}) and Comune '{comune}' ({known_reg})."
                        )
        
        # 1. Street-level database match under this CAP (Highest Precision)
        if (clean_cap, street_core) in self.street_address_lookup:
            res = self.street_address_lookup[(clean_cap, street_core)].copy()
            res["geocoding_quality"] = "STREET_LEVEL_DATABASE_MATCH"
            if comune:
                res["Comune"] = str(comune).strip()
            return res
            
        for (c_cap, c_street), data in self.street_address_lookup.items():
            if c_cap == clean_cap and len(c_street) >= 4:
                if c_street in street_core or street_core in c_street:
                    res = data.copy()
                    res["geocoding_quality"] = "STREET_FUZZY_DATABASE_MATCH"
                    if comune:
                        res["Comune"] = str(comune).strip()
                    return res

        # 2. User provided explicit Comune (Centroid / Administrative fallback)
        if comune:
            comune_norm = normalize_text(comune)
            if (clean_cap, comune_norm) in self.cap_comune_lookup:
                res = self.cap_comune_lookup[(clean_cap, comune_norm)].copy()
                res["geocoding_quality"] = "EXACT_CAP_COMUNE_MATCH"
                return res
            if comune_norm in self.comune_lookup:
                res = self.comune_lookup[comune_norm].copy()
                res["geocoding_quality"] = "COMUNE_NATIONWIDE_MATCH"
                return res

        # 3. Check if address string contains any known Comune under this CAP
        for (c_cap, c_com), data in self.cap_comune_lookup.items():
            if c_cap == clean_cap and len(c_com) >= 4:
                if c_com in clean_addr:
                    res = data.copy()
                    res["geocoding_quality"] = "AUTO_COMUNE_EXTRACTED_FROM_ADDRESS"
                    return res

        # 4. CAP centroid / median fallback
        if clean_cap in self.cap_summary_lookup:
            res = self.cap_summary_lookup[clean_cap].copy()
            res["geocoding_quality"] = "CAP_CENTROID_APPROXIMATE"
            return res

        # 5. Province / Region prefix fallback from Italian CAP prefix
        if prefix in ITALIAN_CAP_PREFIX_MAP:
            if inferred_prov in self.provincia_summary_lookup:
                res = self.provincia_summary_lookup[inferred_prov].copy()
                if comune:
                    res["Comune"] = str(comune).strip()
                res["geocoding_quality"] = "PROVINCE_CENTROID_APPROXIMATE"
                return res
            if inferred_reg in self.regione_summary_lookup:
                res = self.regione_summary_lookup[inferred_reg].copy()
                if comune:
                    res["Comune"] = str(comune).strip()
                res["Provincia"] = inferred_prov
                res["geocoding_quality"] = "REGIONAL_CENTROID_APPROXIMATE"
                return res

        # 6. Unresolvable location -> THROW ERROR
        raise ValueError(
            f"Location not found: Unable to resolve location for address='{address}', CAP='{clean_cap}', Comune='{comune or 'N/A'}'. "
            f"Please verify that the street address, 5-digit Italian CAP, and municipality are correct."
        )

    def predict_new_location(
        self,
        store_name: str,
        address: str,
        cap: str,
        machine_type: str,
        comune: Optional[str] = None,
        store_format: Optional[str] = None,
        sales_area_sqm: Optional[float] = None,
        cash_registers: Optional[float] = None,
        optional_nielsen: Optional[Dict[str, Any]] = None,
        filiera: Optional[str] = None,
        explicit_reward: Optional[Union[str, Dict[str, Any]]] = None,
        interactive_policy: bool = False,
        go_live_month: Optional[int] = None,
        go_live_year: Optional[int] = None,
        enable_rampup: bool = True
    ) -> Dict[str, Any]:
        """
        Executes end-to-end prospective store prediction.
        Applies Improvement 1 (spatial disambiguation) and Improvement 2 (store format/scale).
        Marginalizes across a representative 7-day week to output steady-state expectations.
        """
        # 0. Strict input validation
        store_name = validate_store_name(store_name)
        address = validate_address(address)
        cap = validate_cap(cap)
        machine_type = validate_machine_type(machine_type)
        if comune:
            comune = validate_comune(comune, address=address)
        
        loc_info = self.resolve_location(address, cap, comune=comune)
        brand_clean = clean_brand_name(store_name)
        reg_clean = str(loc_info.get("Regione", "")).strip().upper()
        resolved_gruppo = infer_retail_group(brand_clean, store_name, regione=reg_clean)
        resolved_area = infer_nielsen_area(reg_clean)
        
        # Level 3.5: Commercial Rewards and Filiera Policy
        is_known_fil = self.rewards_engine.is_known_filiera(filiera=filiera, store_name=store_name)
        reward_info = self.rewards_engine.resolve_reward(
            filiera=filiera,
            store_name=store_name,
            retail_group=resolved_gruppo,
            regione=reg_clean,
            explicit_reward=explicit_reward
        )
        
        # If interactive policy selection/confirmation is requested
        if interactive_policy and not explicit_reward:
            chosen_policy = prompt_reward_policy_selection(
                store_name=store_name,
                comune=loc_info.get("Comune"),
                current_reward_info=reward_info,
                is_known=is_known_fil
            )
            reward_info = self.rewards_engine.resolve_reward(
                filiera=filiera,
                store_name=store_name,
                retail_group=resolved_gruppo,
                regione=reg_clean,
                explicit_reward=chosen_policy
            )
        
        # Improvement 2 & Level 3: Determine store scale and alliance features
        nielsen_tipo = "UNKNOWN"
        nielsen_sqm = np.nan
        nielsen_casse = np.nan
        matched_status = "MATCHED" if (brand_clean != "INDIPENDENTE" or resolved_gruppo != "UNKNOWN") else "UNMATCHED"
        scale_source = "DEFAULT_NATIONAL_MEDIAN"
        auto_scale_info = None
        
        # 1. Check if user provided explicit preset format
        if store_format and str(store_format) in STORE_FORMAT_PRESETS:
            preset = STORE_FORMAT_PRESETS[str(store_format)]
            nielsen_tipo = preset["tipo_negozio"]
            nielsen_sqm = float(preset["superficie_vendita"])
            nielsen_casse = float(preset["numero_casse"])
            matched_status = "MATCHED"
            scale_source = f"PRESET: {preset['label']}"
            
        # 2. Overwrite with explicit numeric inputs if provided
        if sales_area_sqm is not None and sales_area_sqm > 0:
            nielsen_sqm = float(sales_area_sqm)
            matched_status = "MATCHED"
            scale_source = "USER_CUSTOM_SCALE"
            # Auto-infer format if not set
            if nielsen_tipo == "UNKNOWN":
                if nielsen_sqm < 400:
                    nielsen_tipo = "LIS"
                elif nielsen_sqm < 2500:
                    nielsen_tipo = "SUP"
                else:
                    nielsen_tipo = "IPR"
                    
        if cash_registers is not None and cash_registers > 0:
            nielsen_casse = float(cash_registers)
            matched_status = "MATCHED"
            scale_source = "USER_CUSTOM_SCALE"
            
        # 3. Optional override dict
        if optional_nielsen:
            if "nielsen_tipo_negozio" in optional_nielsen:
                nielsen_tipo = optional_nielsen["nielsen_tipo_negozio"]
            if "nielsen_superficie_vendita" in optional_nielsen:
                nielsen_sqm = float(optional_nielsen["nielsen_superficie_vendita"])
            if "nielsen_numero_casse" in optional_nielsen:
                nielsen_casse = float(optional_nielsen["nielsen_numero_casse"])
            if "nielsen_gruppo" in optional_nielsen:
                resolved_gruppo = optional_nielsen["nielsen_gruppo"]
            if "nielsen_area" in optional_nielsen:
                resolved_area = optional_nielsen["nielsen_area"]
            matched_status = "MATCHED"
            scale_source = "EXPLICIT_NIELSEN_OVERRIDE"

        # 4. Intelligent Auto-Estimation if scale was not manually specified
        if pd.isna(nielsen_sqm) or pd.isna(nielsen_casse) or scale_source == "DEFAULT_NATIONAL_MEDIAN":
            auto_scale_info = self.scale_estimator.estimate_scale(
                store_name=store_name,
                address=address,
                cap=cap,
                comune=loc_info.get("Comune")
            )
            nielsen_tipo = auto_scale_info["tipo_negozio"]
            nielsen_sqm = float(auto_scale_info["superficie_vendita"])
            nielsen_casse = float(auto_scale_info["numero_casse"])
            scale_source = f"AUTO: {auto_scale_info['label']}"
            matched_status = "MATCHED"

        # Level 4: Spatial Catchment & Competitive Density Enrichment (Pre-Inference)
        clean_cap = str(cap).strip().zfill(5)
        catchment_info = self.catchment_analyzer.analyze_catchment(
            lat=float(loc_info["Latitudine"]),
            lon=float(loc_info["Longitudine"]),
            cap=clean_cap,
            comune=str(loc_info["Comune"]),
            address=address,
            surface=float(loc_info.get("surface", 25.0)),
            store_name=store_name,
            exclude_self_benchmark=True,
            enable_osm=False
        )
        
        # Extract spatial competition metrics for Neural Network conditioning
        net_info = catchment_info.get("coripet_network", {})
        comp_info = catchment_info.get("competitive_density", {})
        nearest_coripet_km = float(net_info.get("nearest_coripet_km", 2.0))
        coripet_rvm_count_5km = float(net_info.get("coripet_count_5km", 2.0))
        is_isolated_rvm = int(coripet_rvm_count_5km == 0)
        grocery_supermarkets = float(comp_info.get("supermarkets", 2.0))
        grocery_total = float(comp_info.get("total_grocery_stores", 5.0))
        hypermarkets = float(comp_info.get("hypermarkets", 0.0))
        is_local_monopoly = int(grocery_supermarkets <= 1.0 and hypermarkets == 0.0)

        # Build 12-month representative calendar (84 days: 12 Mon-Sun mid-month weeks across 2026)
        # Accurately models monthly seasonality across all 12 calendar months
        rep_rows = []
        days_meta = []
        second_mondays = [
            ("2026-01-12", 1, "January", 31, "Winter baseline / off-peak"),
            ("2026-02-09", 2, "February", 28, "Short winter cycle"),
            ("2026-03-09", 3, "March", 31, "Early spring resumption"),
            ("2026-04-13", 4, "April", 30, "Spring baseline"),
            ("2026-05-11", 5, "May", 31, "Late spring ramp-up"),
            ("2026-06-08", 6, "June", 30, "Early summer hydration ramp"),
            ("2026-07-13", 7, "July", 31, "Summer peak hydration"),
            ("2026-08-10", 8, "August", 31, "Mid-summer holiday rhythm"),
            ("2026-09-14", 9, "September", 30, "School / work return"),
            ("2026-10-12", 10, "October", 31, "Autumn steady-state"),
            ("2026-11-09", 11, "November", 30, "Late autumn transition"),
            ("2026-12-14", 12, "December", 31, "Holiday shopping period")
        ]
        
        for mon_str, m_num, m_name, m_days, m_desc in second_mondays:
            start_dt = pd.to_datetime(mon_str)
            for d in range(7):
                cur_dt = start_dt + pd.Timedelta(days=d)
                dow_name = cur_dt.day_name()
                days_meta.append({
                    "month_num": m_num,
                    "month_name": m_name,
                    "days_in_month": m_days,
                    "context": m_desc,
                    "dow": dow_name
                })
                row = {
                    "GIORNO": cur_dt.strftime("%Y-%m-%d"),
                    "CLIENTE": store_name,
                    "Denominazione": store_name,
                    "Cap": cap,
                    "Comune": loc_info["Comune"],
                    "Provincia": loc_info["Provincia"],
                    "Regione": loc_info["Regione"],
                    "Latitudine": loc_info["Latitudine"],
                    "Longitudine": loc_info["Longitudine"],
                    "population": loc_info["population"],
                    "surface": loc_info["surface"],
                    "population_density": loc_info["population_density"],
                    "PIL": loc_info["PIL"],
                    "PIL_per_capita": loc_info["PIL_per_capita"],
                    "TipoMacchina": machine_type,
                    "Fornitore": "DAINT" if "DAINT" in machine_type.upper() else ("RIPET" if "RIPET" in machine_type.upper() else "EURVEN"),
                    "Insegna_Standardized": brand_clean,
                    "nielsen_insegna": brand_clean,
                    "nielsen_match_status": matched_status,
                    "nielsen_tipo_negozio": nielsen_tipo,
                    "nielsen_gruppo": resolved_gruppo,
                    "nielsen_area": resolved_area,
                    "nielsen_superficie_vendita": nielsen_sqm,
                    "nielsen_numero_casse": nielsen_casse,
                    "nearest_coripet_km": nearest_coripet_km,
                    "coripet_rvm_count_5km": coripet_rvm_count_5km,
                    "is_isolated_rvm": is_isolated_rvm,
                    "grocery_supermarket_count": grocery_supermarkets,
                    "grocery_total_stores": grocery_total,
                    "is_local_grocery_monopoly": is_local_monopoly
                }
                rep_rows.append(row)
            
        rep_df = pd.DataFrame(rep_rows)
        rep_df["Insegna_Standardized"] = brand_clean
        X_num, X_cat, _, _ = self.preprocessor.transform(rep_df, nielsen_dropout=0.0)
        
        with torch.no_grad():
            t_num = torch.tensor(X_num, dtype=torch.float32, device=self.device)
            t_cat = torch.tensor(X_cat, dtype=torch.long, device=self.device)
            pred_log_b, pred_log_p = self.model(t_num, t_cat)
            
        pred_b_days_nn = np.clip(np.expm1(pred_log_b.cpu().numpy().squeeze()), a_min=0.0, a_max=None)
        pred_p_days_nn = np.clip(np.expm1(pred_log_p.cpu().numpy().squeeze()), a_min=0.0, a_max=None)

        # Stacking Ensemble blend (LightGBM + NeuralNet) when available
        if self.lgb_model_b is not None and self.lgb_model_p is not None:
            X_all = np.hstack([X_num, X_cat])
            lgb_pred_log_b = self.lgb_model_b.predict(X_all)
            lgb_pred_log_p = self.lgb_model_p.predict(X_all)
            lgb_pred_b = np.clip(np.expm1(lgb_pred_log_b), a_min=0.0, a_max=None)
            lgb_pred_p = np.clip(np.expm1(lgb_pred_log_p), a_min=0.0, a_max=None)
            pred_b_days = self.w_b * lgb_pred_b + (1.0 - self.w_b) * pred_b_days_nn
            pred_p_days = self.w_p * lgb_pred_p + (1.0 - self.w_p) * pred_p_days_nn
        else:
            lgb_pred_b = None
            lgb_pred_p = None
            pred_b_days = pred_b_days_nn
            pred_p_days = pred_p_days_nn
        
        # Aggregate 365-day annual steady-state metrics
        df_sim = pd.DataFrame({
            "month_num": [m["month_num"] for m in days_meta],
            "month_name": [m["month_name"] for m in days_meta],
            "days_in_month": [m["days_in_month"] for m in days_meta],
            "context": [m["context"] for m in days_meta],
            "dow": [m["dow"] for m in days_meta],
            "bottiglie": pred_b_days,
            "presenze": pred_p_days
        })
        
        # Aggregate 365-day annual steady-state metrics (Both Mean and Median)
        expected_daily_presenze = float(df_sim["presenze"].mean())
        expected_daily_presenze_median = float(df_sim["presenze"].median())
        expected_daily_bottiglie = float(df_sim["bottiglie"].mean())
        expected_daily_bottiglie_median = float(df_sim["bottiglie"].median())
        expected_monthly_bottiglie = expected_daily_bottiglie * 30.4
        expected_monthly_bottiglie_median = expected_daily_bottiglie_median * 30.4
        expected_annual_bottiglie = expected_daily_bottiglie * 365.0
        expected_annual_bottiglie_median = expected_daily_bottiglie_median * 365.0
        
        # Monthly Collection Profile & Targets across all 12 calendar months (Both Mean and Median)
        monthly_collection_profile = []
        for mon_str, m_num, m_name, m_days, m_desc in second_mondays:
            sub = df_sim[df_sim["month_num"] == m_num]
            m_daily_b = float(sub["bottiglie"].mean())
            m_daily_b_med = float(sub["bottiglie"].median())
            m_daily_p = float(sub["presenze"].mean())
            m_daily_p_med = float(sub["presenze"].median())
            m_target_b = round(m_daily_b * m_days)
            m_target_b_med = round(m_daily_b_med * m_days)
            monthly_collection_profile.append({
                "month_num": m_num,
                "month_name": m_name,
                "days_in_month": m_days,
                "context": m_desc,
                "daily_bottiglie": round(m_daily_b, 1),
                "daily_bottiglie_median": round(m_daily_b_med, 1),
                "monthly_target_bottiglie": int(m_target_b),
                "monthly_target_bottiglie_median": int(m_target_b_med),
                "daily_presenze": round(m_daily_p, 1),
                "daily_presenze_median": round(m_daily_p_med, 1)
            })
        
        # Deseasonalized Day-of-Week Rhythm across all 12 months - Both Mean and Median
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_means = df_sim.groupby("dow")["bottiglie"].mean()
        dow_medians = df_sim.groupby("dow")["bottiglie"].median()
        dow_dict = {day: round(float(dow_means.get(day, expected_daily_bottiglie)), 1) for day in dow_order}
        dow_median_dict = {day: round(float(dow_medians.get(day, expected_daily_bottiglie_median)), 1) for day in dow_order}
        
        # Catchment Dilution & Cannibalization Realization Layer
        realization = calculate_catchment_realization_factor(
            unconstrained_daily_bottles=expected_daily_bottiglie,
            unconstrained_daily_presenze=expected_daily_presenze,
            nearest_coripet_km=nearest_coripet_km,
            nearest_pf_name=net_info.get("nearest_pf_name", ""),
            store_name=store_name,
            coripet_count_1km=int(net_info.get("coripet_count_1km", 0)),
            coripet_count_3km=int(net_info.get("coripet_count_3km", 0)),
            coripet_count_5km=int(net_info.get("coripet_count_5km", 0)),
            grocery_total_stores=int(comp_info.get("total_grocery_stores", 0)),
            network_position=catchment_info["coripet_network"]["network_position"],
            sales_area_sqm=nielsen_sqm,
            numero_casse=nielsen_casse,
            is_independent_banner=(brand_clean == "INDIPENDENTE"),
            nielsen_area=f"Area {resolved_area}",
            is_proximity_market=(store_format == "1" or nielsen_tipo in ["LIS", "VIC"]),
            population_density=float(loc_info["population_density"]),
            population=float(loc_info["population"]),
            regione=str(loc_info.get("Regione", "")),
            machine_type=machine_type,
            reward_info=reward_info
        )

        # Location Suitability & Clearance Gate (Go / No-Go Decision Engine)
        suitability = evaluate_location_suitability(
            realized_daily_bottles=realization["realized_daily_bottles"],
            unconstrained_daily_bottles=expected_daily_bottiglie,
            nearest_coripet_km=nearest_coripet_km,
            nearest_pf_name=net_info.get("nearest_pf_name", ""),
            dilution_percentage=realization["dilution_percentage"],
            network_position=catchment_info["coripet_network"]["network_position"],
            sales_area_sqm=nielsen_sqm,
            numero_casse=nielsen_casse,
            population=float(loc_info["population"]),
            population_density=float(loc_info["population_density"]),
            total_grocery_stores=int(comp_info.get("total_grocery_stores", 0)),
            conservative_floor_bottles=realization["realized_bounds"]["floor_daily_b"],
            requested_machine=machine_type,
            brand_name=brand_clean,
            retail_group=resolved_gruppo,
            shopping_centers_count=int(comp_info.get("shopping_centers", 0))
        )

        # Level 4: Commercial Investment Scoring & Decision Engine
        potential_score = calculate_location_potential_score(expected_daily_bottiglie)
        cannibalization_alert = (catchment_info["coripet_network"]["network_position"] == "CANNIBALIZATION_ALERT")
        recommendation = determine_recommendation_tier(
            potential_score=potential_score,
            daily_bottles=realization["realized_daily_bottles"],
            cannibalization_alert=cannibalization_alert,
            density_tier=catchment_info["competitive_density"]["density_tier"],
            store_scale_applied=scale_source,
            suitability_status=suitability["suitability_status"]
        )
        operational_bounds = calculate_operational_bounds(
            daily_bottles=expected_daily_bottiglie,
            daily_presenze=expected_daily_presenze
        )
        confidence = evaluate_prediction_confidence(
            geocoding_quality=loc_info["geocoding_quality"],
            brand_embedding_active=(brand_clean != "INDIPENDENTE"),
            alliance_embedding_active=(resolved_gruppo != "UNKNOWN"),
            store_scale_applied=scale_source,
            nearest_coripet_km=catchment_info["coripet_network"]["nearest_coripet_km"]
        )

        # Statistical Probability Factor & Target Attainment
        probability_factors = calculate_probability_factors(
            daily_bottles=realization["realized_daily_bottles"],
            confidence_score=confidence.get("confidence_score", 100),
            requested_machine=machine_type
        )

        # Attach median metrics to realization
        realized_b_med = round(expected_daily_bottiglie_median * realization["realization_factor"], 1)
        realized_p_med = round(expected_daily_presenze_median * np.sqrt(realization["realization_factor"]), 1)
        realization["realized_daily_bottles_median"] = realized_b_med
        realization["realized_daily_presenze_median"] = realized_p_med
        realization["realized_monthly_bottles_median"] = int(round(realized_b_med * 30.4))
        realization["realized_annual_bottles_median"] = int(round(realized_b_med * 365.0))

        # -------------------------------------------------------------
        # Universal Risk-Gated Operational Underwriting Baseline
        # -------------------------------------------------------------
        is_demo_warn = (suitability.get("gate_results", {}).get("demographic_gate", {}).get("status") == "WARN")
        is_self_match = bool(net_info.get("is_benchmark_self_match"))
        hist_b = net_info.get("benchmark_pf_hist_bottles")
        
        realized_b = realization["realized_daily_bottles"]
        p10_floor = probability_factors.get("downside_floor_p10_daily", round(realized_b * 0.70, 1))
        p25_floor = operational_bounds.get("conservative_floor", {}).get("daily_bottiglie", round(realized_b * 0.82, 1))

        # Resolve natural baseline commercial reward for benchmark anchor matching
        baseline_reward = self.rewards_engine.resolve_reward(
            filiera=filiera,
            store_name=store_name,
            retail_group=resolved_gruppo,
            regione=reg_clean,
            explicit_reward=None
        )
        if explicit_reward is None:
            base_real_b = float(realized_b)
        else:
            base_real = calculate_catchment_realization_factor(
                unconstrained_daily_bottles=expected_daily_bottiglie,
                unconstrained_daily_presenze=expected_daily_presenze,
                nearest_coripet_km=nearest_coripet_km,
                nearest_pf_name=net_info.get("nearest_pf_name", ""),
                store_name=store_name,
                coripet_count_1km=int(net_info.get("coripet_count_1km", 0)),
                coripet_count_3km=int(net_info.get("coripet_count_3km", 0)),
                coripet_count_5km=int(net_info.get("coripet_count_5km", 0)),
                grocery_total_stores=int(comp_info.get("total_grocery_stores", 0)),
                network_position=catchment_info["coripet_network"]["network_position"],
                sales_area_sqm=nielsen_sqm,
                numero_casse=nielsen_casse,
                is_independent_banner=(brand_clean == "INDIPENDENTE"),
                nielsen_area=f"Area {resolved_area}",
                is_proximity_market=(store_format == "1" or nielsen_tipo in ["LIS", "VIC"]),
                population_density=float(loc_info["population_density"]),
                population=float(loc_info["population"]),
                regione=str(loc_info.get("Regione", "")),
                machine_type=machine_type,
                reward_info=baseline_reward
            )
            base_real_b = float(base_real["realized_daily_bottles"])

        if is_self_match and hist_b and hist_b > 0:
            eff_hist = round(float(hist_b), 1)
            # Relative commercial reward elasticity vs historical operational baseline policy
            rel_reward_scale_b = (float(realized_b) / base_real_b) if (base_real_b > 0) else 1.0

            # Hardware replacement / model upgrade relative scaling:
            # If the user changed the machine model vs the historical unit installed on site:
            hist_m = net_info.get("benchmark_pf_machine_type")
            rel_hw_scale_b = 1.0
            hw_adj_desc = ""
            if hist_m and str(hist_m).strip().upper() != str(machine_type).strip().upper() and str(hist_m).strip().upper() not in ["UNKNOWN", "NONE", ""]:
                rep_df_bm = rep_df.copy()
                rep_df_bm["TipoMacchina"] = str(hist_m).strip()
                rep_df_bm["Fornitore"] = "DAINT" if "DAINT" in str(hist_m).upper() else ("RIPET" if "RIPET" in str(hist_m).upper() else "EURVEN")
                X_num_bm, X_cat_bm, _, _ = self.preprocessor.transform(rep_df_bm, nielsen_dropout=0.0)
                with torch.no_grad():
                    t_num_bm = torch.tensor(X_num_bm, dtype=torch.float32, device=self.device)
                    t_cat_bm = torch.tensor(X_cat_bm, dtype=torch.long, device=self.device)
                    p_log_b_bm, _ = self.model(t_num_bm, t_cat_bm)
                pred_b_bm_nn = np.clip(np.expm1(p_log_b_bm.cpu().numpy().squeeze()), a_min=0.0, a_max=None)
                if self.lgb_model_b is not None:
                    X_all_bm = np.hstack([X_num_bm, X_cat_bm])
                    lgb_pred_log_b_bm = self.lgb_model_b.predict(X_all_bm)
                    lgb_pred_b_bm = np.clip(np.expm1(lgb_pred_log_b_bm), a_min=0.0, a_max=None)
                    pred_b_bm = self.w_b * lgb_pred_b_bm + (1.0 - self.w_b) * pred_b_bm_nn
                else:
                    pred_b_bm = pred_b_bm_nn
                expected_b_hist_m = float(pred_b_bm.mean())
                if expected_b_hist_m > 0:
                    rel_hw_scale_b = float(expected_daily_bottiglie / expected_b_hist_m)
                    hw_diff_pct = (rel_hw_scale_b - 1.0) * 100.0
                    hw_adj_desc = f", {'+' if hw_diff_pct > 0 else ''}{hw_diff_pct:.1f}% hardware model change vs {hist_m}"

            total_scale_b = rel_reward_scale_b * rel_hw_scale_b
            underwritten_daily_b = round(eff_hist * total_scale_b, 1)
            if abs(total_scale_b - 1.0) < 0.005:
                underwriting_mode = f"EMPIRICAL_TELEMETRY_ANCHORED ({net_info.get('benchmark_pf_id')}: {eff_hist} b/d verified operational baseline)"
            else:
                pct_adj = (total_scale_b - 1.0) * 100.0
                underwriting_mode = (
                    f"EMPIRICAL_TELEMETRY_ANCHORED ({net_info.get('benchmark_pf_id')}: {eff_hist} b/d baseline "
                    f"{'+' if pct_adj > 0 else ''}{pct_adj:.1f}% adjustment{hw_adj_desc})"
                )
        elif is_demo_warn:
            underwritten_daily_b = float(p10_floor)
            underwriting_mode = "DEMOGRAPHIC_RISK_GATED (P10 Downside Underwriting Bound)"
        else:
            underwritten_daily_b = float(realized_b)
            underwriting_mode = "STANDARD_STEADY_STATE"

        realization["underwritten_daily_bottles"] = underwritten_daily_b
        realization["underwritten_monthly_bottles"] = int(round(underwritten_daily_b * 30.4))
        realization["underwritten_annual_bottles"] = int(round(underwritten_daily_b * 365.0))
        realization["underwriting_mode"] = underwriting_mode
        realization["is_risk_adjusted"] = (underwriting_mode != "STANDARD_STEADY_STATE")

        # Dynamically scale monthly profile, day-of-week rhythm, and presenze to underwritten baseline
        k_scale = realization.get("realization_factor", 1.0)
        ann_med_base = round(expected_daily_bottiglie_median * k_scale, 1)
        underwrite_scale = (underwritten_daily_b / ann_med_base) if (realization["is_risk_adjusted"] and ann_med_base > 0) else 1.0

        # Scale realized footfall (presenze) proportionally with underwritten operational volume
        underwritten_daily_p = round(realization["realized_daily_presenze"] * underwrite_scale, 1)
        underwritten_daily_p_med = round(realization["realized_daily_presenze_median"] * underwrite_scale, 1)
        realization["underwritten_daily_presenze"] = underwritten_daily_p
        realization["underwritten_daily_presenze_median"] = underwritten_daily_p_med

        if realization["is_risk_adjusted"]:
            u_floor = round(underwritten_daily_b * 0.85, 1)
            u_peak = round(underwritten_daily_b * 1.15, 1)
            realization["underwritten_bounds"] = {
                "floor_daily_b": u_floor,
                "floor_monthly_b": int(round(u_floor * 30.4)),
                "base_daily_b": round(underwritten_daily_b, 1),
                "base_monthly_b": int(round(underwritten_daily_b * 30.4)),
                "peak_daily_b": u_peak,
                "peak_monthly_b": int(round(u_peak * 30.4))
            }
            if "economic_volume_gate" in suitability.get("gate_results", {}):
                suitability["gate_results"]["economic_volume_gate"]["metric"] = (
                    f"Robust collection yield ({underwritten_daily_b:.1f} b/day vs 120 b/d breakeven)"
                )
            if "hardware_sizing_gate" in suitability.get("gate_results", {}):
                suitability["gate_results"]["hardware_sizing_gate"]["metric"] = (
                    f"Hardware '{machine_type}' aligned with volume tier ({underwritten_daily_b:.1f} b/d)"
                )

        # Inland non-coastal seasonal vacation exodus shaping for risk-adjusted stores
        reg_upper = str(loc_info.get("Regione", "")).upper()
        comune_upper = str(loc_info.get("Comune", "")).upper()
        coastal_tokens = ["VIAREGGIO", "FORTE DEI MARMI", "LIVORNO", "GROSSETO", "FOLLONICA", "CECINA", "PIOMBINO", "ORBETELLO", "PORTOFERRAIO", "RIMINI", "RICCIONE", "CERVIA", "CESENATICO"]
        is_inland_town = ("TOSCANA" in reg_upper or "UMBRIA" in reg_upper or "EMILIA" in reg_upper or "PIEMONTE" in reg_upper or "LOMBARDIA" in reg_upper) and not any(c in comune_upper for c in coastal_tokens)

        underwritten_monthly_profile = []
        for m in monthly_collection_profile:
            m_num = m["month_num"]
            season_adj = 1.0
            if is_inland_town and realization["is_risk_adjusted"]:
                if m_num == 5:
                    season_adj = 0.96
                elif m_num == 6:
                    season_adj = 0.97
                elif m_num == 7:
                    season_adj = 1.05
                elif m_num == 8:
                    season_adj = 0.95

            d_b = round(m["daily_bottiglie"] * k_scale * underwrite_scale * season_adj, 1)
            d_b_med = round(m["daily_bottiglie_median"] * k_scale * underwrite_scale * season_adj, 1)
            underwritten_monthly_profile.append({
                **m,
                "daily_bottiglie": d_b,
                "daily_bottiglie_median": d_b_med,
                "monthly_target_bottiglie": int(round(d_b * m["days_in_month"])),
                "monthly_target_bottiglie_median": int(round(d_b_med * m["days_in_month"])),
                "daily_presenze": round(m.get("daily_presenze", expected_daily_presenze) * np.sqrt(k_scale) * underwrite_scale, 1),
                "daily_presenze_median": round(m.get("daily_presenze_median", expected_daily_presenze_median) * np.sqrt(k_scale) * underwrite_scale, 1)
            })

        underwritten_dow = {day: round(val * k_scale * underwrite_scale, 1) for day, val in dow_dict.items()}
        underwritten_dow_med = {day: round(val * k_scale * underwrite_scale, 1) for day, val in dow_median_dict.items()}

        realization["underwritten_scale_factor"] = round(underwrite_scale, 4)
        realization["underwritten_monthly_profile"] = underwritten_monthly_profile
        realization["underwritten_dow"] = underwritten_dow
        realization["underwritten_dow_median"] = underwritten_dow_med

        # Operational & Predictive Diagnostics (Root Causes of Rate & Accuracy Variance)
        diagnostics = generate_predictive_diagnostics(
            realized_daily_bottles=realization["realized_daily_bottles"],
            unconstrained_daily_bottles=expected_daily_bottiglie,
            realized_daily_presenze=realization["realized_daily_presenze"],
            sales_area_sqm=nielsen_sqm,
            numero_casse=nielsen_casse,
            population=float(loc_info["population"]),
            population_density=float(loc_info["population_density"]),
            total_grocery_stores=int(comp_info.get("total_grocery_stores", 0)),
            nearest_coripet_km=nearest_coripet_km,
            nearest_pf_name=net_info.get("nearest_pf_name", ""),
            coripet_count_3km=int(net_info.get("coripet_count_3km", 0)),
            coripet_count_5km=int(net_info.get("coripet_count_5km", 0)),
            brand_name=brand_clean,
            retail_group=resolved_gruppo,
            regione=str(loc_info.get("Regione", "")),
            geocoding_quality=str(loc_info.get("geocoding_quality", "")),
            dilution_percentage=realization.get("dilution_percentage", 0.0),
            dilution_reasons=realization.get("dilution_reasons", []),
            pil_per_capita=float(loc_info.get("PIL_per_capita", 0.0)),
            requested_machine=machine_type
        )

        # Level 3.5: All Reward Types Sensitivity & Comparative Impact Analysis
        reward_sensitivity = []
        is_custom_policy = bool(reward_info.get("is_custom", False))
        
        if is_custom_policy:
            c_label = reward_info.get("custom_label") or f"Custom ({reward_info.get('discount_euro', 3):.2f}€ su {reward_info.get('min_spend_euro', 25):.0f}€)"
            reward_sensitivity.append({
                "reward_type": c_label,
                "reward_label": f"Custom Promotion ({reward_info.get('contract_title', 'Custom')})",
                "incentive_strength": reward_info.get("incentive_strength", "CUSTOM"),
                "is_selected": True,
                "is_custom": True,
                "daily_bottles": underwritten_daily_b,
                "monthly_bottles": int(round(underwritten_daily_b * 30.4)),
                "daily_presenze": underwritten_daily_p,
                "diff_bottles": 0.0,
                "pct_variance": 0.0,
                "multiplier_presenze": reward_info.get("multiplier_presenze", 1.0),
                "multiplier_bottles": reward_info.get("multiplier_bottles", 1.0),
                "commercial_impact": reward_info.get("definition_en") or reward_info.get("definition_it") or reward_info.get("contract_rule") or "Custom configured promotion"
            })

        for r_key, r_name, r_label, r_desc in REWARD_MENU_CHOICES:
            test_rew = self.rewards_engine.resolve_reward(
                filiera=filiera,
                store_name=store_name,
                retail_group=resolved_gruppo,
                regione=reg_clean,
                explicit_reward=r_name
            )
            test_real = calculate_catchment_realization_factor(
                unconstrained_daily_bottles=expected_daily_bottiglie,
                unconstrained_daily_presenze=expected_daily_presenze,
                nearest_coripet_km=nearest_coripet_km,
                nearest_pf_name=net_info.get("nearest_pf_name", ""),
                store_name=store_name,
                coripet_count_1km=int(net_info.get("coripet_count_1km", 0)),
                coripet_count_3km=int(net_info.get("coripet_count_3km", 0)),
                coripet_count_5km=int(net_info.get("coripet_count_5km", 0)),
                grocery_total_stores=int(comp_info.get("total_grocery_stores", 0)),
                network_position=catchment_info["coripet_network"]["network_position"],
                sales_area_sqm=nielsen_sqm,
                numero_casse=nielsen_casse,
                is_independent_banner=(brand_clean == "INDIPENDENTE"),
                nielsen_area=f"Area {resolved_area}",
                is_proximity_market=(store_format == "1" or nielsen_tipo in ["LIS", "VIC"]),
                population_density=float(loc_info["population_density"]),
                population=float(loc_info["population"]),
                regione=str(loc_info.get("Regione", "")),
                machine_type=machine_type,
                reward_info=test_rew
            )
            t_real_b = test_real["realized_daily_bottles"]
            is_sel = (r_name == reward_info.get("reward_type")) and (reward_info.get("match_source") == "EXPLICIT_OVERRIDE") and not is_custom_policy
            if is_sel:
                t_underwritten_b = underwritten_daily_b
                t_underwritten_p = underwritten_daily_p
            else:
                rel_ratio = (float(t_real_b) / float(realized_b)) if (realized_b and realized_b > 0) else 1.0
                t_underwritten_b = round(underwritten_daily_b * rel_ratio, 1)
                rel_p_ratio = (float(test_real["realized_daily_presenze"]) / float(realization["realized_daily_presenze"])) if (realization.get("realized_daily_presenze", 0) > 0) else 1.0
                t_underwritten_p = round(underwritten_daily_p * rel_p_ratio, 1)
                
            t_diff_b = round(t_underwritten_b - underwritten_daily_b, 1)
            t_pct_b = round((t_underwritten_b - underwritten_daily_b) / underwritten_daily_b * 100.0, 1) if underwritten_daily_b > 0 else 0.0
            
            reward_sensitivity.append({
                "reward_type": r_name,
                "reward_label": r_label,
                "incentive_strength": test_rew.get("incentive_strength", "N/A"),
                "is_selected": is_sel,
                "is_custom": False,
                "daily_bottles": t_underwritten_b,
                "monthly_bottles": int(round(t_underwritten_b * 30.4)),
                "daily_presenze": t_underwritten_p,
                "diff_bottles": t_diff_b,
                "pct_variance": t_pct_b,
                "multiplier_presenze": test_rew.get("multiplier_presenze", 1.0),
                "multiplier_bottles": test_rew.get("multiplier_bottles", 1.0),
                "commercial_impact": r_desc
            })

        lifecycle_horizon = calculate_lifecycle_horizon(
            underwritten_monthly_profile=underwritten_monthly_profile,
            expected_daily_bottiglie=underwritten_daily_b,
            expected_annual_bottiglie=underwritten_daily_b * 365.0,
            expected_daily_presenze=underwritten_daily_p,
            go_live_month=go_live_month,
            go_live_year=go_live_year,
            enable_rampup=enable_rampup
        )

        return {
            "inputs": {
                "store_name": store_name,
                "address": address,
                "cap": cap,
                "comune_input": comune if comune else "AUTO_DETECTED",
                "machine_type": machine_type,
                "store_scale_applied": scale_source,
                "sales_area_sqm": nielsen_sqm,
                "cash_registers": nielsen_casse,
                "scale_auto_estimated": auto_scale_info is not None,
                "auto_scale_details": auto_scale_info["details"] if auto_scale_info else "MANUALLY_SPECIFIED",
                "go_live_month": lifecycle_horizon["launch_month_num"],
                "go_live_month_name": lifecycle_horizon["launch_month_name"],
                "go_live_year": lifecycle_horizon["launch_year"],
                "enable_rampup": enable_rampup
            },
            "location_attributes": {
                "resolved_comune": loc_info["Comune"],
                "resolved_provincia": loc_info["Provincia"],
                "resolved_regione": loc_info["Regione"],
                "latitudine": round(loc_info["Latitudine"], 5),
                "longitudine": round(loc_info["Longitudine"], 5),
                "population": round(loc_info["population"], 0),
                "density": round(loc_info["population_density"], 1),
                "geocoding_quality": loc_info["geocoding_quality"]
            },
            "retail_intelligence": {
                "standardized_brand": brand_clean,
                "retail_buying_group": resolved_gruppo,
                "nielsen_geographic_area": resolved_area,
                "brand_embedding_active": brand_clean != "INDIPENDENTE",
                "alliance_embedding_active": resolved_gruppo != "UNKNOWN",
                "nielsen_match_status": matched_status
            },
            "commercial_rewards": reward_info,
            "is_new_filiera": not is_known_fil,
            "reward_sensitivity_analysis": reward_sensitivity,
            "spatial_catchment": catchment_info,
            "location_suitability": suitability,
            "investment_assessment": {
                "location_potential_score": potential_score,
                "recommendation": recommendation,
                "suitability": suitability,
                "confidence": confidence,
                "operational_bounds": operational_bounds,
                "realized_bounds": realization["realized_bounds"],
                "probability_factors": probability_factors
            },
            "probability_analysis": probability_factors,
            "realized_forecast": realization,
            "diagnostics": diagnostics,
            "lifecycle_horizon": lifecycle_horizon,
            "predictions": {
                "model_engine": "STACKING ENSEMBLE (LightGBM + MultiTaskRVMNet)" if self.lgb_model_b is not None else "MultiTaskRVMNet (PyTorch Neural Net)",
                "lgb_daily_bottiglie": round(float(lgb_pred_b.mean()), 1) if (self.lgb_model_b is not None and lgb_pred_b is not None) else None,
                "lgb_daily_presenze": round(float(lgb_pred_p.mean()), 1) if (self.lgb_model_p is not None and lgb_pred_p is not None) else None,
                "nn_daily_bottiglie": round(float(pred_b_days_nn.mean()), 1),
                "nn_daily_presenze": round(float(pred_p_days_nn.mean()), 1),
                "ensemble_weights": {
                    "lgb_weight": self.w_b if self.lgb_model_b is not None else 0.0,
                    "nn_weight": round(1.0 - self.w_b, 2) if self.lgb_model_b is not None else 1.0,
                },
                "expected_daily_presenze": round(expected_daily_presenze, 1),
                "expected_daily_presenze_median": round(expected_daily_presenze_median, 1),
                "expected_daily_bottiglie": round(expected_daily_bottiglie, 1),
                "expected_daily_bottiglie_median": round(expected_daily_bottiglie_median, 1),
                "expected_monthly_bottiglie": round(expected_monthly_bottiglie, 0),
                "expected_monthly_bottiglie_median": round(expected_monthly_bottiglie_median, 0),
                "expected_annual_bottiglie": round(expected_annual_bottiglie, 0),
                "expected_annual_bottiglie_median": round(expected_annual_bottiglie_median, 0),
                "monthly_collection_profile": monthly_collection_profile,
                "daily_bottles_by_dayofweek": dow_dict,
                "daily_bottles_by_dayofweek_median": dow_median_dict,
                "underwritten_monthly_profile": underwritten_monthly_profile,
                "underwritten_dow": underwritten_dow,
                "underwritten_dow_median": underwritten_dow_med,
                "calculation_methodology": {
                    "daily_baseline_bottles": "Both Arithmetic Mean and Median across 84-day simulation grid (12 months x 7 days)",
                    "daily_baseline_presenze": "Both Arithmetic Mean and Median across 84-day simulation grid (12 months x 7 days)",
                    "monthly_bottles": "Daily mean and median multiplied by standardized month duration (30.4 days)",
                    "annual_bottles": "Daily mean and median multiplied by 365.0 days (cumulative annual sum)",
                    "monthly_collection_profile": "Both Arithmetic Mean and Median across 7 representative days for each of the 12 calendar months, reporting daily volume and monthly targets (Daily * Days in month)",
                    "days_of_the_week": "Both Arithmetic Mean and Median for each specific weekday across 12 monthly simulation cycles (12 Mondays, 12 Tuesdays, etc.)",
                    "operational_capacity_scenarios": {
                        "conservative_floor_p25": "25th Percentile lower operational bound (~82% of steady-state mean)",
                        "steady_state_base_p50": "50th Percentile / Median operational baseline expectation",
                        "optimistic_peak_p75": "75th Percentile upper operational ceiling (~125% of steady-state mean)"
                    }
                },
                "model_accuracy_benchmarks": {
                    "benchmark_sample_size": "28 validated retail store configurations",
                    "median_bottle_accuracy": "80.1%",
                    "median_error_margin": "±19.9%",
                    "median_footfall_accuracy": "86.8%",
                    "ground_truth_alignment": "97.7% vs multi-year database steady-state",
                    "p75_scenario_calibration": "99.9% exact corridor match on strategic anchors (e.g. Esselunga Verona: 774.2 vs 774.9 b/d)",
                    "dashboard_report": "reports/accuracy_dashboard.html"
                }
            }
        }

def print_prediction_summary(result: dict):
    """Prints the comprehensive Prediction & Investment Summary report with Mean & Median."""
    inputs = result.get("inputs", {})
    store_name = inputs.get("store_name", "")
    address = inputs.get("address", "")
    cap = inputs.get("cap", "")
    machine_type = inputs.get("machine_type", "")

    print("\n" + "=" * 65)
    print("                 PREDICTION & INVESTMENT SUMMARY")
    print("=" * 65)

    print(f"\nStore:       {store_name}")
    print(f"Address:     {address}")
    print(f"CAP:         {cap}")
    print(f"Machine:     {machine_type}")
    print(f"Engine:      {result['predictions'].get('model_engine', 'MultiTaskRVMNet')}")
    print(f"Scale:       {result['inputs']['store_scale_applied']}")
    print(f"Area:        {result['inputs']['sales_area_sqm']} m2 | Checkouts: {result['inputs']['cash_registers']}")

    print("\n--- RETAIL & ALLIANCE INTELLIGENCE (Level 3) ---")
    ret_intel = result["retail_intelligence"]
    brand_tag = f"{ret_intel['standardized_brand']} [EMBEDDING ACTIVE]" if ret_intel["brand_embedding_active"] else f"{ret_intel['standardized_brand']} [GENERIC INDEPENDENT]"
    group_tag = f"{ret_intel['retail_buying_group']} [ALLIANCE ACTIVE]" if ret_intel["alliance_embedding_active"] else f"{ret_intel['retail_buying_group']} [UNMATCHED]"
    print(f"Banner / Insegna: {brand_tag}")
    print(f"Retail Group:     {group_tag}")
    print(f"Nielsen Area:     Area {ret_intel['nielsen_geographic_area']}")
    print(f"Nielsen Status:   {ret_intel['nielsen_match_status']}")

    if "commercial_rewards" in result:
        rew = result["commercial_rewards"]
        print("\n--- COMMERCIAL REWARDS & INCENTIVE INTELLIGENCE (Level 3.5) ---")
        print(f"Commercial Filiera: {rew.get('filiera', 'N/A')} [{rew.get('match_source', 'N/A')}]")
        print(f"Reward Scheme:      {rew.get('reward_type', 'N/A')}")
        print(f"Reward Label:       {rew.get('reward_label', 'N/A')}")
        if rew.get("contract_title"):
            print(f"Contract Promo:     {rew.get('contract_title')}")
        if rew.get("contract_rule"):
            print(f"Contract Details:   {rew.get('contract_rule')}")
        pts = rew.get("points_required")
        disc = rew.get("discount_euro")
        spend = rew.get("min_spend_euro")
        rate = rew.get("valore_cent_per_bottiglia")
        disc_pct = rew.get("sconto_percentuale_carrello")
        if pts:
            print(f"Points Threshold:   {pts} pts (1 bottle = 1 point -> {pts} bottles per voucher)")
        if rate and disc is not None and spend is not None:
            print(f"Unit Reward Rate:   {rate:.2f} €-cent/bottle ({disc:.1f}€ discount on {spend:.0f}€ min spend)")
        if disc_pct:
            print(f"Cart Discount Rate: {disc_pct:.1f}% discount on grocery basket")
        mult_b = rew.get("multiplier_bottles", 1.0)
        mult_p = rew.get("multiplier_presenze", 1.0)
        elast_tag = f"Bottles: {mult_b:.3f}x | Footfall: {mult_p:.3f}x"
        if mult_b == 1.0 and mult_p == 1.0:
            elast_tag += " [NEUTRAL GDO BASELINE]"
        elif mult_p > 1.0:
            elast_tag += f" [HIGH INCENTIVE DRAW +{round((mult_p-1.0)*100, 1)}%]"
        else:
            elast_tag += f" [HIGH THRESHOLD FRICTION {round((mult_p-1.0)*100, 1)}%]"
        print(f"Behavioral Impact:  {elast_tag}")
        print(f"Incentive Strength: {rew.get('incentive_strength', 'N/A')} ({rew.get('incentive_tier', 'N/A')})")
        print(f"Policy Definition:  {rew.get('definition_en') or rew.get('definition_it', 'N/A')}")

    if "reward_sensitivity_analysis" in result:
        sens = result["reward_sensitivity_analysis"]
        is_custom_f = result.get("commercial_rewards", {}).get("is_custom", False)
        is_new_f = result.get("is_new_filiera", False)
        if is_custom_f:
            f_status = "CUSTOM COMMERCIAL REWARD CONFIGURED"
        elif is_new_f:
            f_status = "NEW / UNREGISTERED FILIERA"
        else:
            f_status = "KNOWN / REGISTERED FILIERA"

        print("\n--- ALL REWARD TYPES SENSITIVITY & IMPACT ANALYSIS ---")
        print(f"  [Filiera Status: {f_status}]")
        print("  [Comparative operational yield and footfall across selected policy and all standard Coripet schemes]")
        print("  " + "=" * 122)
        print(f"  {'#':<2} | {'Reward Policy Scheme':<38} | {'Daily Bottles':<14} | {'Monthly Vol':<13} | {'Active Users':<12} | {'Variance vs Sel':<18} | {'Incentive Tier'}")
        print("  " + "-" * 122)
        for idx, item in enumerate(sens, 1):
            r_name = item['reward_type']
            sel_tag = " [SELECTED]" if item.get('is_selected') else ""
            name_display = f"{r_name}{sel_tag}"
            b_str = f"{item['daily_bottles']:.1f} b/day"
            m_str = f"{item['monthly_bottles']:,} b/mo"
            u_str = f"{item['daily_presenze']:.1f} u/day"
            
            if item.get('is_selected'):
                var_str = "SELECTED POLICY"
            else:
                diff = item['diff_bottles']
                pct = item['pct_variance']
                sign = "+" if diff > 0 else ""
                var_str = f"{sign}{pct:.1f}% ({sign}{diff:.1f} b/d)"
                
            tier_str = f"{item['incentive_strength']}"
            num_display = "*" if item.get("is_custom") else str(idx)
            print(f"  {num_display:<2} | {name_display:<38} | {b_str:>14} | {m_str:>13} | {u_str:>12} | {var_str:>18} | {tier_str}")
        print("  " + "-" * 122)
        print("  Commercial Impact Breakdown:")
        for idx, item in enumerate(sens, 1):
            sel_bullet = "->" if item.get('is_selected') else " *"
            num_display = "*" if item.get("is_custom") else str(idx)
            print(f"   {sel_bullet} [{num_display}] {item['reward_type']}: {item['commercial_impact']}")
        print("  " + "=" * 122)

    print("\n--- SPATIAL CATCHMENT & COMPETITIVE DENSITY (Level 4) ---")
    net = result["spatial_catchment"]["coripet_network"]
    comp = result["spatial_catchment"]["competitive_density"]
    print(f"Network Position: {net['network_position']}")
    print(f"Status Note:      {net['network_status_description']}")
    if net.get("is_benchmark_self_match"):
        print(f"Self-Match Status: EXCLUDED FROM CANNIBALIZATION (Benchmarking active unit {net.get('benchmark_pf_id')}: {net.get('benchmark_pf_hist_bottles')} b/d baseline)")
    print(f"Nearest Coripet:  {net['nearest_coripet_km']} km -> {net['nearest_pf_name']} ({net['nearest_pf_comune']})")
    print(f"RVM Density:      {net['coripet_count_1km']} in 1km | {net['coripet_count_3km']} in 3km | {net['coripet_count_5km']} in 5km | {net['same_municipality_count']} in Comune")
    print(f"Trade Area Food:  {comp['total_grocery_stores']} stores ({comp['supermarkets']} Super, {comp['discounts']} Disc, {comp['hypermarkets']} Iper, {comp['shopping_centers']} Malls)")
    print(f"Top Competitors:  {comp['top_competing_banners']}")

    print("\n--- RESOLVED LOCATION ---")
    print(f"Comune:      {result['location_attributes']['resolved_comune']}")
    print(f"Provincia:   {result['location_attributes']['resolved_provincia']}")
    print(f"Regione:     {result['location_attributes']['resolved_regione']}")
    print(f"Latitude:    {result['location_attributes']['latitudine']}")
    print(f"Longitude:   {result['location_attributes']['longitudine']}")
    print(f"Population:  {result['location_attributes']['population']}")
    print(f"Density:     {result['location_attributes']['density']} inh/km2")
    print(f"Resolution:  {result['location_attributes']['geocoding_quality']}")

    print("\n--- OPERATIONAL PREDICTION (365-Day Annualized Steady-State) ---")
    print("  [Calculation: 84-Day Simulation Grid (12 months x 7 days) reporting both Arithmetic Mean & Median]")
    real = result.get("realized_forecast", {})
    k_scale = real.get("realization_factor", 1.0)
    dilution_pct = real.get("dilution_percentage", 0.0)
    uplift_pct = real.get("uplift_percentage", 0.0)
    real_b = real.get("realized_daily_bottles", result['predictions']['expected_daily_bottiglie'])
    real_b_med = real.get("realized_daily_bottles_median", result['predictions']['expected_daily_bottiglie_median'])
    real_p = real.get("realized_daily_presenze", result['predictions']['expected_daily_presenze'])
    real_p_med = real.get("realized_daily_presenze_median", result['predictions']['expected_daily_presenze_median'])
    real_mo = real.get("realized_monthly_bottles", int(round(real_b * 30.4)))
    real_mo_med = real.get("realized_monthly_bottles_median", int(round(real_b_med * 30.4)))
    real_yr = real.get("realized_annual_bottles", int(round(real_b * 365.0)))
    real_yr_med = real.get("realized_annual_bottles_median", int(round(real_b_med * 365.0)))
    has_realization_adjustment = abs(k_scale - 1.0) > 0.001 or len(real.get("dilution_reasons", [])) > 0
    is_risk_adjusted = real.get("is_risk_adjusted", False)

    if has_realization_adjustment or is_risk_adjusted:
        floor_b = real.get("realized_bounds", {}).get("floor_daily_b", round(real_b * 0.82, 1))
        floor_mo = real.get("realized_bounds", {}).get("floor_monthly_b", int(round(floor_b * 30.4)))
        floor_yr = int(round(floor_b * 365.0))

        if is_risk_adjusted:
            under_b = real.get("underwritten_daily_bottles", floor_b)
            under_mo = real.get("underwritten_monthly_bottles", int(round(under_b * 30.4)))
            under_yr = real.get("underwritten_annual_bottles", int(round(under_b * 365.0)))
            under_p = real.get("underwritten_daily_presenze", real_p)
            under_p_med = real.get("underwritten_daily_presenze_median", real_p_med)
            mode_lbl = real.get("underwriting_mode", "RISK_ADJUSTED")
            print(f"Unconstrained Physical Ceiling:    Mean: {result['predictions']['expected_daily_bottiglie']} b/d  |  Median: {result['predictions']['expected_daily_bottiglie_median']} b/d")
            print(f"Operational Underwriting Mode:     {mode_lbl}")
            print("-" * 65)
            print(f"OPERATIONAL YIELD (Underwritten Baseline): {under_b} b/day  |  {under_mo:,} b/mo  [~82% Serviced Uptime]")
            print(f"REALIZED PRESENZE/day:                     {under_p} u/day (Mean)    |  {under_p_med} u/day (Median)")
            print(f"ANNUALIZED PROJECTION:                     {under_yr:,} b/yr (Underwritten Baseline)")
            print(f"STEADY-STATE CEILING (100% Uptime):        {real_b} b/day (Mean)  |  {real_mo:,} b/mo  |  {real_b_med} b/day (Median)")
        else:
            print(f"Unconstrained Ceiling:    Mean: {result['predictions']['expected_daily_bottiglie']} b/d  |  Median: {result['predictions']['expected_daily_bottiglie_median']} b/d")
            if k_scale < 1.0:
                print(f"Catchment Realization:    {k_scale*100:.1f}% (Net Dilution: -{dilution_pct}%)")
            else:
                print(f"Catchment Realization:    {k_scale*100:.1f}% (Net Catchment Uplift: +{uplift_pct}%)")
            if real.get("dilution_reasons"):
                for r in real["dilution_reasons"]:
                    print(f"  * {r}")
            print("-" * 65)
            print(f"OPERATIONAL YIELD (Underwritten Baseline): {floor_b} b/day (Mean)  |  {floor_mo:,} b/mo  [~82% Serviced Uptime]")
            print(f"STEADY-STATE CEILING (100% Uptime):    {real_b} b/day (Mean)  |  {real_mo:,} b/mo  |  {real_b_med} b/day (Median)")
            print(f"ANNUALIZED PROJECTION:                 {real_yr:,} b/yr (Ceiling)  |  {floor_yr:,} b/yr (Serviced Baseline)")
            print(f"REALIZED PRESENZE/day:                 {real_p} u/day (Mean)    |  {real_p_med} u/day (Median)")
    else:
        print(f"PRESENZE/day:      {result['predictions']['expected_daily_presenze']} users/day (Mean)   |  {result['predictions']['expected_daily_presenze_median']} users/day (Median)")
        print(f"BOTTIGLIE/day:     {result['predictions']['expected_daily_bottiglie']} bottles/day (Mean) |  {result['predictions']['expected_daily_bottiglie_median']} bottles/day (Median)")
        exp_mo = int(round(result['predictions']['expected_monthly_bottiglie']))
        exp_mo_med = int(round(result['predictions'].get('expected_monthly_bottiglie_median', exp_mo)))
        exp_yr = int(round(result['predictions']['expected_annual_bottiglie']))
        exp_yr_med = int(round(result['predictions'].get('expected_annual_bottiglie_median', exp_yr)))
        print(f"BOTTIGLIE/month:   {exp_mo:,} b/mo (Mean)  |  {exp_mo_med:,} b/mo (Median)")
        print(f"BOTTIGLIE/year:    {exp_yr:,} b/yr (Mean)  |  {exp_yr_med:,} b/yr (Median)")

    # Determine effective underwritten scaling for monthly and weekly tables
    underwrite_scale = 1.0
    under_b_val = None
    if is_risk_adjusted:
        under_b_val = real.get("underwritten_daily_bottles")
        ann_med_base = round(result["predictions"]["expected_daily_bottiglie_median"] * k_scale, 1)
        if under_b_val and ann_med_base > 0:
            underwrite_scale = under_b_val / ann_med_base

    if is_risk_adjusted and under_b_val:
        print(f"\n--- MONTHLY COLLECTION PROFILE & TARGETS (Underwritten Baseline: {under_b_val} b/d) ---")
        print("  [Calculation: 12-Month Simulation Calendar - Seasonal Curve Scaled to Underwritten Baseline]")
    else:
        print("\n--- MONTHLY COLLECTION PROFILE & TARGETS (Median Baseline) ---")
        print("  [Calculation: 12-Month Simulation Calendar - 50th Percentile Median Daily Volume & Monthly Targets]")
    print(f"  {'Month':<16} | {'Daily Bottles (P50)':<21} | {'Monthly Target (P50)':<23} | Operational Context")
    print("  " + "-" * 88)
    monthly = result["predictions"].get("monthly_collection_profile", [])
    tot_tgt_med = 0
    for m in monthly:
        d_med = round(m["daily_bottiglie_median"] * k_scale * underwrite_scale, 1)
        m_days = m["days_in_month"]
        t_med = int(round(d_med * m_days))
        tot_tgt_med += t_med
        m_lbl = f"{m['month_name']} ({m_days}d)"
        ctx = m.get("context", "")
        print(f"  {m_lbl:<16} : {d_med:>7.1f} bottles/day   | {t_med:>8,} b/mo ({t_med/1000:>4.1f}k)    | {ctx}")
    print("  " + "-" * 88)
    ann_med = round(under_b_val, 1) if (is_risk_adjusted and under_b_val) else round(result["predictions"]["expected_daily_bottiglie_median"] * k_scale, 1)
    print(f"  {'Annual Total':<16} : {ann_med:>7.1f} bottles/day   | {tot_tgt_med:>8,} b/yr ({tot_tgt_med/1000:>4.1f}k)    | 365-Day Cumulative Annual Target (Median)")

    print("\n--- COMMERCIAL INVESTMENT ASSESSMENT (Level 4) ---")
    invest = result["investment_assessment"]
    rec = invest["recommendation"]
    conf = invest["confidence"]
    bounds = invest["operational_bounds"]
    suit = result.get("location_suitability", invest.get("suitability", {}))
    print(f"Potential Score:  {invest['location_potential_score']}/100 (Nationwide Percentile Rank)")
    print(f"Recommendation:   {rec['tier_name']}")
    print(f"Decision:         {rec['decision']}")
    print(f"Recommended Unit: {rec['recommended_machine']}")
    print(f"Logistics Rhythm: {rec['logistics_frequency']}")
    print(f"Rationale:        {rec['rationale']}")
    print(f"Confidence:       {conf['confidence_rating']} ({conf['confidence_score']}/100 pts)")
    print(f"Key Drivers:      {', '.join(conf['key_confidence_drivers'])}")

    if suit:
        print("\n--- LOCATION SUITABILITY & CLEARANCE (Go / No-Go Gate) ---")
        print(f"Suitability:       {suit['summary_title']}")
        print(f"Clearance Verdict: {suit['verdict']}")
        print(f"Suitability Score: {suit['suitability_score']} / 100 pts")
        print("Clearance Gates:")
        for gate_key, gate in suit.get("gate_results", {}).items():
            icon = "[PASS]" if gate["status"] == "PASS" else ("[WARN]" if gate["status"] == "WARN" else "[FAIL]")
            print(f"  {icon} {gate['label']:<25}: {gate['metric']}")
        if suit.get("disqualifying_reasons"):
            print("Disqualifying Blockers:")
            for reason in suit["disqualifying_reasons"]:
                print(f"  * {reason}")
        if suit.get("warning_flags") and not suit.get("disqualifying_reasons"):
            print("Underwriting Watch Items:")
            for warn in suit["warning_flags"]:
                print(f"  * {warn}")
        if suit.get("key_strengths") and not suit.get("disqualifying_reasons"):
            print("Operational Strengths:")
            for strength in suit["key_strengths"]:
                print(f"  * {strength}")
        print(f"Actionable Advice: {suit.get('recommended_action', '')}")

    # Statistical Probability Factor & Risk Profile
    prob = result.get("probability_analysis")
    if prob:
        print("\n--- STATISTICAL PROBABILITY & FUTURE PERFORMANCE PROFILE ---")
        print(f"Capital Safety Probability: {prob['success_probability']:.1f}% [{prob['risk_rating']}]")
        
        # Future Performance Tiers
        tiers = prob.get("performance_tiers", {})
        if tiers:
            print("\nFuture Performance Outlook (Probability by Volume Tier):")
            print(f"  * Low Yield / Review       (< 200 b/d)     : {tiers['p_low_under_200']:>5.1f}%")
            print(f"  * Moderate / Viable        (200 - 400 b/d) : {tiers['p_moderate_200_400']:>5.1f}%")
            print(f"  * High Performer Hub       (400 - 700 b/d) : {tiers['p_high_400_700']:>5.1f}%")
            print(f"  * Star / Strategic Anchor  (> 700 b/d)     : {tiers['p_star_over_700']:>5.1f}%")
            print(f"  -> Dominant Future Outlook : {tiers.get('most_likely_tier', '')}")

        # Cumulative Annual Milestones
        milestones = prob.get("annual_milestones", {})
        if milestones:
            print("\nCumulative Annual Collection Milestones (Year 1 Yield):")
            print(f"  * P(Year 1 >=  50,000 bottles) : {milestones['p_annual_50k']:>5.1f}%  [Base Survival Volume]")
            print(f"  * P(Year 1 >= 100,000 bottles) : {milestones['p_annual_100k']:>5.1f}%  [Core Operational Milestone]")
            print(f"  * P(Year 1 >= 150,000 bottles) : {milestones['p_annual_150k']:>5.1f}%  [Strong Commercial Target]")
            print(f"  * P(Year 1 >= 200,000 bottles) : {milestones['p_annual_200k']:>5.1f}%  [Top Tier Asset Output]")

        # Hardware Operational Fit
        hw = prob.get("hardware_operational_fit", {})
        if hw:
            print("\nHardware Operational Fit & Capacity Utilization:")
            print(f"  * Installed Unit Class         : {hw.get('hardware_class', '')}")
            print(f"  * Optimal Utilization Prob     : {hw.get('p_optimal_utilization', 0.0):>5.1f}%")
            print(f"  * Daily Saturation / Overflow  : {hw.get('p_daily_overflow_risk', 0.0):>5.1f}%  [Threshold: {hw.get('daily_saturation_threshold', 0):.0f} b/d]")
            print(f"  * Underutilization Risk        : {hw.get('p_underutilization_risk', 0.0):>5.1f}%")
            print(f"  * Recommended Logistics Cadence: {hw.get('recommended_service_rhythm', '')}")

        print("\nTarget Attainment Probabilities:")
        print(f"  * P(Volume >= 120 b/d - Economic Breakeven)  : {prob['p_breakeven_120']:>5.1f}%  [Capital Recovery Threshold]")
        print(f"  * P(Volume >= 300 b/d - Standard Baseline)   : {prob['p_standard_300']:>5.1f}%  [Commercial Viability]")
        print(f"  * P(Volume >= 500 b/d - High-Yield Anchor)   : {prob['p_high_yield_500']:>5.1f}%  [Top-Quartile Performance]")
        print(f"  * P(Volume >= 1,000 b/d - Strategic Anchor)  : {prob['p_anchor_1000']:>5.1f}%  [Hypermarket Capacity Demand]")
        p10_d = round(prob['downside_floor_p10_daily'] * underwrite_scale, 1) if (is_risk_adjusted and underwrite_scale < 1.0) else prob['downside_floor_p10_daily']
        p10_m = int(round(p10_d * 30.4))
        p90_d = round(prob['upside_peak_p90_daily'] * underwrite_scale, 1) if (is_risk_adjusted and underwrite_scale < 1.0) else prob['upside_peak_p90_daily']
        p90_m = int(round(p90_d * 30.4))
        print("Statistically Underwritten Volume (80% Confidence Band):")
        print(f"  * Downside Floor (P10 - 90% Statistical Certainty): {p10_d} b/day ({p10_m:,} b/month)")
        print(f"  * Upside Ceiling (P90 - Top 10% Peak Potential)   : {p90_d} b/day ({p90_m:,} b/month)")

    print("\n--- OPERATIONAL CAPACITY SCENARIOS ---")
    print("  [Calculation: Statistical Quantiles - Conservative Floor P25, Steady-State Base P50 Median, Optimistic Peak P75]")
    if is_risk_adjusted and under_b_val:
        ub = real.get("underwritten_bounds", {})
        u_p25_b = ub.get("floor_daily_b", round(under_b_val * 0.85, 1))
        u_p25_mo = ub.get("floor_monthly_b", int(round(u_p25_b * 30.4)))
        u_p50_b = ub.get("base_daily_b", round(under_b_val, 1))
        u_p50_mo = ub.get("base_monthly_b", int(round(u_p50_b * 30.4)))
        u_p75_b = ub.get("peak_daily_b", round(under_b_val * 1.15, 1))
        u_p75_mo = ub.get("peak_monthly_b", int(round(u_p75_b * 30.4)))
        print("  [Underwritten Operational Volumes - Risk & Cluster Calibrated]")
        print(f"  Conservative Floor (P25) : {u_p25_mo:,} b/month ({u_p25_b} b/day) - 25th Percentile Lower Bound")
        print(f"  Steady-State Base  (P50) : {u_p50_mo:,} b/month ({u_p50_b} b/day) - 50th Percentile Median Baseline")
        print(f"  Optimistic Peak    (P75) : {u_p75_mo:,} b/month ({u_p75_b} b/day) - 75th Percentile Upper Ceiling")
        print(f"  (Unconstrained Standalone Ceiling: {real_mo:,} b/month | {real_b} b/day)")
    elif has_realization_adjustment:
        rb = real["realized_bounds"]
        print("  [Realized Operational Volumes - Catchment Adjusted]")
        print(f"  Conservative Floor (P25) : {rb['floor_monthly_b']:,} b/month ({rb['floor_daily_b']} b/day) - 25th Percentile Lower Bound")
        print(f"  Steady-State Base  (P50) : {rb['base_monthly_b']:,} b/month ({rb['base_daily_b']} b/day) - 50th Percentile Median Baseline")
        print(f"  Optimistic Peak    (P75) : {rb['peak_monthly_b']:,} b/month ({rb['peak_daily_b']} b/day) - 75th Percentile Upper Ceiling")
        print(f"  (Unconstrained Standalone Ceiling: {bounds['steady_state_expected']['monthly_bottiglie']:,} b/month | {bounds['steady_state_expected']['daily_bottiglie']} b/day)")
    else:
        print(f"  Conservative Floor (P25) : {bounds['conservative_floor']['monthly_bottiglie']:,} b/month ({bounds['conservative_floor']['daily_bottiglie']} b/day) - 25th Percentile Lower Bound")
        print(f"  Steady-State Base  (P50) : {bounds['steady_state_expected']['monthly_bottiglie']:,} b/month ({bounds['steady_state_expected']['daily_bottiglie']} b/day) - 50th Percentile Median Baseline")
        print(f"  Optimistic Peak    (P75) : {bounds['optimistic_peak']['monthly_bottiglie']:,} b/month ({bounds['optimistic_peak']['daily_bottiglie']} b/day) - 75th Percentile Upper Ceiling")

    if is_risk_adjusted and under_b_val:
        print(f"\n--- EXPECTED DAILY BOTTLES (Deseasonalized Weekly Rhythm: {under_b_val} b/d Baseline) ---")
    else:
        print("\n--- EXPECTED DAILY BOTTLES (Deseasonalized Weekly Rhythm) ---")
    print("  [Calculation: 12 monthly simulation cycles averaged (Mean) and 50th percentile (Median) for each weekday]")
    print("  [Methodology: 12 Mondays aggregated for Monday, 12 Tuesdays for Tuesday, etc.; Average of 7 equals Daily Baseline]")
    dow_means = result["predictions"]["daily_bottles_by_dayofweek"]
    dow_meds = result["predictions"].get("daily_bottles_by_dayofweek_median", {})
    for day, mean_val in dow_means.items():
        m_val = round(mean_val * k_scale * underwrite_scale, 1)
        med_val = round(dow_meds.get(day, mean_val) * k_scale * underwrite_scale, 1)
        print(f"  {day:<12}: {m_val:>6.1f} b/day (Mean)  |  {med_val:>6.1f} b/day (Median)")
    avg_7_mean = round(sum(dow_means.values()) * k_scale * underwrite_scale / 7.0, 1)
    avg_7_med = round(sum(dow_meds.values()) * k_scale * underwrite_scale / 7.0, 1) if dow_meds else avg_7_mean
    print(f"  {'Average of 7':<12}: {avg_7_mean:>6.1f} b/day (Mean)  |  {avg_7_med:>6.1f} b/day (Median)")

    print("\n--- OPERATIONAL & PREDICTIVE DIAGNOSTICS ---")
    diag = result.get("diagnostics", {})
    drivers = diag.get("primary_performance_drivers", [])
    limiters = diag.get("limiting_factors", [])
    suggestions = diag.get("actionable_suggestions", [])
    accuracy_factors = diag.get("accuracy_variance_factors", [])

    print("1. Primary Performance Drivers (Why This Volume Was Predicted):")
    if drivers:
        for d in drivers:
            print(f"  * {d}")
    else:
        print("  * Standard commercial foot traffic and balanced network catchment.")

    if limiters:
        print("\n2. Limiting Factors & Bottlenecks (Why Results Are Not Higher):")
        for lim in limiters:
            print(f"  * {lim}")

    if suggestions:
        print("\n3. Actionable Model Suggestions (How to Increase Collection in Future):")
        for s in suggestions:
            print(f"  * {s}")

    print("\nKey Factors That May Cause Real-World Operational Telemetry Variance (Accuracy Limitations):")
    if accuracy_factors:
        for f in accuracy_factors:
            print(f"  * {f}")
    else:
        print("  * Minimal expected variance: Stable demographic profile with standard supermarket operational rhythms.")

    print("\n" + "=" * 65)


def save_result_to_dashboard(
    result: Dict[str, Any], 
    actual_bottles: Optional[float] = None, 
    actual_users: Optional[float] = None, 
    json_path: str = "reports/benchmark_tests.json"
) -> bool:
    """Appends prediction test result to benchmark_tests.json and refreshes accuracy dashboard."""
    try:
        inputs = result.get("inputs", {})
        loc = result.get("location_attributes", {})
        preds = result.get("predictions", {})
        realized = result.get("realized_forecast", {})
        suit = result.get("location_suitability", {})
        
        name = inputs.get("store_name", "Test Store")
        comune = loc.get("resolved_comune", "")
        prov = loc.get("resolved_provincia", "")
        pred_b = float(realized.get("underwritten_daily_bottles", realized.get("realized_daily_bottles", preds.get("expected_daily_bottiglie", 0.0))))
        pred_u = float(realized.get("underwritten_daily_presenze", realized.get("realized_daily_presenze", preds.get("expected_daily_presenze", 0.0))))
        cat = inputs.get("store_scale_applied", "Custom Store")
        clearance = suit.get("clearance_verdict", "GO [APPROVED]")
        
        store_entry = {
            "id": re.sub(r"[^a-z0-9_]", "", name.lower().replace(" ", "_")),
            "store_name": name,
            "comune": comune,
            "provincia": prov,
            "address": inputs.get("address", ""),
            "pred_bottles": round(pred_b, 1),
            "actual_bottles": round(float(actual_bottles), 1) if actual_bottles is not None else None,
            "pred_users": round(pred_u, 1),
            "actual_users": round(float(actual_users), 1) if actual_users is not None else None,
            "category": cat,
            "format": inputs.get("store_scale_applied", "Supermarket"),
            "clearance": clearance,
            "p75_bottles": round(pred_b * 1.25, 1),
            "p25_bottles": round(pred_b * 0.82, 1)
        }
        
        existing = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                
        # Replace if existing with same id, or insert at top
        existing = [t for t in existing if t.get("id") != store_entry["id"]]
        existing.insert(0, store_entry)
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            
        # Re-trigger dashboard builder
        try:
            from scripts.build_general_dashboard import generate_dashboard
            generate_dashboard()
        except Exception:
            pass

        return True
    except Exception as e:
        print(f"  [WARNING] Could not save to dashboard: {e}")
        return False


def run_interactive_prediction(predictor=None):
    """Runs interactive CLI prediction workflow."""
    print("\n" + "=" * 60)
    print("        CORIPET NEW LOCATION PREDICTOR (v4.1 — Auto-Scale)")
    print("=" * 60)

    if predictor is None:
        predictor = RVMNewLocationPredictor()

    # 1. Base inputs (All fields validated strictly)
    store_name = ""
    while not store_name:
        raw_name = input("\nEnter store name: ").strip()
        try:
            store_name = validate_store_name(raw_name)
        except ValueError as e:
            print(f"  [ERROR] {e}")

    address = ""
    while not address:
        raw_addr = input("Enter address: ").strip()
        try:
            address = validate_address(raw_addr)
        except ValueError as e:
            print(f"  [ERROR] {e}")

    cap = ""
    while not cap:
        raw_cap = input("Enter CAP: ").strip()
        try:
            cap = validate_cap(raw_cap)
        except ValueError as e:
            print(f"  [ERROR] {e}")

    comune = ""
    while not comune:
        raw_comune = input("Enter Comune: ").strip()
        try:
            comune = validate_comune(raw_comune, address=address)
        except ValueError as e:
            print(f"  [ERROR] {e}")

    machine_type = ""
    while not machine_type:
        raw_machine = input("Enter machine type (e.g. DAINT_G, Ripet_C-front, DAINT_A-bis): ").strip()
        try:
            machine_type = validate_machine_type(raw_machine)
        except ValueError as e:
            print(f"  [ERROR] {e}")

    # 2. Pre-verify Location Resolution before proceeding to scale and prediction
    try:
        loc_info = predictor.resolve_location(address, cap, comune=comune)
        print("\n" + "-" * 70)
        print(f"  [LOCATION VERIFIED] {loc_info.get('Comune')}, {loc_info.get('Provincia')} ({loc_info.get('Regione')})")
        print(f"  Geocoding Match Quality: {loc_info.get('geocoding_quality')}")
        print("-" * 70)
    except ValueError as e:
        print("\n" + "=" * 70)
        print(f"  [ERROR] {e}")
        print("=" * 70)
        return None

    # 3. Intelligent Store Format & Scale Auto-Detection
    auto_scale = predictor.scale_estimator.estimate_scale(store_name, address, cap, comune=comune)
    print("\n" + "-" * 70)
    print("  [AUTO-DETECTED STORE FORMAT & PHYSICAL SCALE]")
    print(f"  Format:       {auto_scale['tipo_negozio']} — {auto_scale['label']}")
    print(f"  Sales Area:   ~{auto_scale['superficie_vendita']:.0f} m2 | Cash Registers: {auto_scale['numero_casse']:.0f}")
    print(f"  Evidence:     {auto_scale['details']}")
    print("-" * 70)
    print("  Press [Enter] to ACCEPT Auto-Detection (RECOMMENDED), or select manual override:")
    print("    [1] Small Supermarket / Neighborhood Market (~250-450 m2, 2-3 checkouts)")
    print("    [2] Standard Supermarket (~750-1,100 m2, 4-5 checkouts)")
    print("    [3] Superstore / Large Supermarket (~1,500-2,200 m2, 8-10 checkouts)")
    print("    [4] Hypermarket / Mall Anchor (> 3,000 m2, 18+ checkouts)")
    print("    [5] Discount Store (Eurospin, Lidl, MD, ~750-900 m2, 4-5 checkouts)")
    print("    [6] Custom (enter exact square meters and checkouts)")
    
    scale_choice = input("\nSelect Option [Enter for Auto, or 1-6]: ").strip()
    
    store_format = None
    sales_area_sqm = None
    cash_registers = None
    
    if scale_choice in ("", "A", "a", "AUTO", "auto"):
        # Auto-detected scale applied
        store_format = auto_scale["tipo_negozio"]
        sales_area_sqm = float(auto_scale["superficie_vendita"])
        cash_registers = float(auto_scale["numero_casse"])
    elif scale_choice in ("1", "2", "3", "4", "5"):
        store_format = scale_choice
    elif scale_choice == "6":
        sqm_str = input("Enter sales area in square meters (e.g. 500): ").strip()
        casse_str = input("Enter number of cash registers (e.g. 4): ").strip()
        try:
            sales_area_sqm = float(sqm_str) if sqm_str else None
        except ValueError:
            sales_area_sqm = None
        try:
            cash_registers = float(casse_str) if casse_str else None
        except ValueError:
            cash_registers = None

    # 4. Interactive Promotional Reward Policy Selection
    is_known_fil = predictor.rewards_engine.is_known_filiera(store_name=store_name)
    current_reward = predictor.rewards_engine.resolve_reward(store_name=store_name, comune=comune)
    selected_reward = prompt_reward_policy_selection(
        store_name=store_name,
        comune=comune,
        current_reward_info=current_reward,
        is_known=is_known_fil
    )

    # Make prediction
    try:
        result = predictor.predict_new_location(
            store_name=store_name,
            address=address,
            cap=cap,
            comune=comune if comune else None,
            machine_type=machine_type,
            store_format=store_format,
            sales_area_sqm=sales_area_sqm,
            cash_registers=cash_registers,
            explicit_reward=selected_reward
        )
        print_prediction_summary(result)
        return result
    except ValueError as e:
        print("\n" + "=" * 70)
        print(f"  [ERROR] {e}")
        print("=" * 70)
        return None


if __name__ == "__main__":
    run_interactive_prediction()
