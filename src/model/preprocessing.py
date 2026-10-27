"""
Data preprocessing, feature encoding, and group-separated splitting for Coripet RVM Predictor.
Ensures zero data leakage: all scalers, encoders, imputers, and medians are fitted
STRICTLY on the training set (unseen PFs in validation/test are never touched).
"""
import os
import json
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any

# Canonical 4 Nielsen geographical areas of Italy
REGION_TO_NIELSEN_AREA = {
    "PIEMONTE": "1.0", "VALLE D AOSTA": "1.0", "VALLE D'AOSTA": "1.0", "VAL D AOSTA": "1.0", "LIGURIA": "1.0", "LOMBARDIA": "1.0",
    "VENETO": "2.0", "TRENTINO ALTO ADIGE": "2.0", "TRENTINO-ALTO ADIGE": "2.0", "FRIULI VENEZIA GIULIA": "2.0", "FRIULI-VENEZIA GIULIA": "2.0", "EMILIA ROMAGNA": "2.0", "EMILIA-ROMAGNA": "2.0",
    "TOSCANA": "3.0", "UMBRIA": "3.0", "MARCHE": "3.0", "LAZIO": "3.0", "SARDEGNA": "3.0",
    "ABRUZZO": "4.0", "MOLISE": "4.0", "PUGLIA": "4.0", "CAMPANIA": "4.0", "BASILICATA": "4.0", "CALABRIA": "4.0", "SICILIA": "4.0"
}

def clean_retail_text(text: Any) -> str:
    """Cleans and standardizes Italian store text, accents, and encoding artifacts."""
    if text is None or pd.isna(text):
        return ""
    s = str(text)
    s = s.replace("ò", "o").replace("Ò", "O")
    s = s.replace("à", "a").replace("À", "A")
    s = s.replace("è", "e").replace("È", "E")
    s = s.replace("é", "e").replace("É", "E")
    s = s.replace("ì", "i").replace("Ì", "I")
    s = s.replace("ù", "u").replace("Ù", "U")
    s = s.replace("\ufffd", "O")
    norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    norm = norm.upper()
    norm = re.sub(r"[^A-Z0-9\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()

def clean_brand_name(brand_raw: Any) -> str:
    """Standardizes store name or client to known retail banners in the model's 22-class vocabulary."""
    s = clean_retail_text(brand_raw)
    if not s:
        return "INDIPENDENTE"

    # Specific brand mappings matching the 22 Insegna_Standardized vocabulary
    if any(k in s for k in ["ESSELUNGA", "LAESSE"]):
        return "ESSELUNGA"
    if any(k in s for k in ["CONAD", "MARGHERITA", "SAPORI DINTORNI", "TUDAY", "TODIS"]):
        return "CONAD"
    if any(k in s for k in ["COOP", "UNICOOP", "INCOOP", "IPERCOOP", "SUPERCONTI"]):
        return "COOP"
    if any(k in s for k in ["CARREFOUR", "ETRURIA", "BEGOGO"]):
        return "CARREFOUR"
    if "PENNY" in s:
        return "PENNY"
    if "LEROY" in s:
        return "LEROY MERLIN"
    if any(k in s for k in ["BRICO", "TECNOMAT", "OBI"]):
        return "BRICO"
    if "BENNET" in s:
        return "BENNET"
    if "DODECA" in s:
        return "DODECA"
    if "DECO" in s:
        return "DECO"
    if any(k in s for k in ["CRAI", "PELLICANO", "TUTTIGIORNI"]):
        return "CRAI"
    if any(k in s for k in ["RETAILPRO", "RETAIL PRO"]):
        return "RETAILPRO"
    if "EKOM" in s:
        return "EKOM"
    if any(k in s for k in ["PAM", "PANORAMA"]):
        return "PAM"
    if any(k in s for k in ["DIMAR", "MERCATO"]):
        return "DIMAR"
    if any(k in s for k in ["DESPAR", "EUROSPAR", "INTERSPAR"]):
        return "DESPAR"
    if any(k in s for k in ["IPER LA GRANDE", "IPER MONTEBELLO", "IPERAL", "UNES", "U2"]):
        return "IPER"
    if "EUROSPIN" in s:
        return "EUROSPIN"
    if "MD" in s.split() or "MD DISCOUNT" in s or "MD SPA" in s:
        return "MD"
    if "GIGANTE" in s:
        return "IL GIGANTE"
    return "INDIPENDENTE"

def infer_retail_group(brand_clean: str, raw_name: Any, regione: str = "") -> str:
    """Infers commercial buying group alliance from clean banner, raw text, and region."""
    s = clean_retail_text(raw_name)
    reg = str(regione).strip().upper()

    # 1. Conad group
    if brand_clean == "CONAD" or any(k in s for k in ["CONAD", "TODIS", "TUDAY", "PAC 2000A"]):
        return "CONAD"

    # 2. Coop Italia
    if brand_clean == "COOP" or any(k in s for k in ["COOP", "UNICOOP", "SUPERCONTI"]):
        return "COOP ITALIA"

    # 3. Gruppo VéGé
    if brand_clean in ("DECO", "DODECA", "BENNET") or any(k in s for k in [
        "DECO", "DODECA", "BENNET", "PICCOLO", "VEGE", "MULTICEDI", "ARENA", "TOSANO"
    ]):
        return "GRUPPO VEGE"

    # 4. Esselunga
    if brand_clean == "ESSELUNGA" or "ESSELUNGA" in s:
        return "ESSELUNGA"

    # 5. Carrefour Italia
    if brand_clean == "CARREFOUR" or any(k in s for k in ["CARREFOUR", "ETRURIA"]):
        return "GRUPPO CARREFOUR ITALIA SPA"

    # 6. Rewe (Penny)
    if brand_clean == "PENNY" or any(k in s for k in ["PENNY", "REWE"]):
        return "REWE"

    # 7. Crai
    if brand_clean == "CRAI" or any(k in s for k in ["CRAI", "TUTTIGIORNI", "PELLICANO", "ABBI", "DETTORI"]):
        return "CRAI"

    # 8. Agorà Network (Ekom, Basko, Sogegross, Tigros, Poli, Iperal)
    if brand_clean == "EKOM" or any(k in s for k in ["EKOM", "BASKO", "SOGEGROSS", "TIGROS", "POLI", "IPERAL"]):
        return "AGORA' NETWORK SCARL"

    # 9. Gruppo PAM
    if brand_clean == "PAM" or any(k in s for k in ["PAM", "PANORAMA", "IN S DISCOUNT"]):
        return "GRUPPO PAM"

    # 10. MD Spa
    if brand_clean == "MD" or "MD" in s.split() or "MD SPA" in s:
        return "MD SPA"

    # 11. Eurospin
    if brand_clean == "EUROSPIN" or "EUROSPIN" in s:
        return "EUROSPIN"

    # 12. Despar / Aspiag
    if brand_clean == "DESPAR" or any(k in s for k in ["DESPAR", "EUROSPAR", "INTERSPAR"]):
        if reg in ("VENETO", "TRENTINO-ALTO ADIGE", "FRIULI-VENEZIA GIULIA", "EMILIA-ROMAGNA"):
            return "ASPIAG"
        return "DESPAR SERVIZI"

    # 13. Selex Commerciale (Largest regional retailer alliance in Italy)
    if brand_clean in ("RETAILPRO", "DIMAR") or any(k in s for k in [
        "SELEX", "FAMILA", "SI CON TE", "A O", "TIGRE", "OASI", "GABRIELLI", 
        "EMI", "EMISFERO", "UNICOMM", "GULLIVER", "DOK", "PAN", "ALI ", "ALI", "MAXI DI", "GIGANTE"
    ]):
        return "SELEX COMMERCIALE"

    # 14. D.IT Distribuzione Italiana (Sigma, Sisa, Contè)
    if any(k in s for k in ["SIGMA", "SISA", "CONTE", "D IT"]):
        return "D.IT. DISTRIBUZIONE ITALIANA"

    # 15. Prix Quality
    if "PRIX" in s:
        return "PRIX QUALITY"

    # 16. Gruppo Tuo
    if any(k in s for k in ["TUODI", "FRESCO MARKET"]):
        return "GRUPPO TUO"

    # 17. Consorzio C3
    if "C3" in s:
        return "C3"

    return "UNKNOWN"

def infer_nielsen_area(regione: Any) -> str:
    """Infers the canonical Italian Nielsen Geographic Area (1.0, 2.0, 3.0, 4.0) from Regione."""
    if regione is None or pd.isna(regione):
        return "UNKNOWN"
    norm_reg = clean_retail_text(regione)
    return REGION_TO_NIELSEN_AREA.get(norm_reg, "UNKNOWN")

def is_italian_holiday(dt: pd.Timestamp) -> int:
    """Identifies standard Italian national public holidays."""
    m, d = dt.month, dt.day
    # Fixed Italian holidays
    if (m == 1 and d in (1, 6)) or \
       (m == 4 and d == 25) or \
       (m == 5 and d == 1) or \
       (m == 6 and d == 2) or \
       (m == 8 and d == 15) or \
       (m == 11 and d == 1) or \
       (m == 12 and d in (8, 25, 26)):
        return 1
    return 0

class RVMPreprocessor:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.fitted = False
        
        # Numeric scaling & imputation states
        self.num_medians: Dict[str, float] = {}
        self.scaler = StandardScaler()
        
        # Categorical vocabularies (mapping value -> integer index; index 0 reserved for UNKNOWN)
        self.cat_vocabs: Dict[str, Dict[str, int]] = {}
        
        # Feature names list
        self.num_features: List[str] = []
        self.cat_features: List[str] = []
        self.all_feature_names: List[str] = []
        
    def prepare_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives clean continuous, temporal, and categorical attributes from dataframe."""
        data = df.copy()
        
        # 1. Parse date if needed
        if "dt" not in data.columns:
            data["dt"] = pd.to_datetime(data["GIORNO"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            if data["dt"].isna().all():
                data["dt"] = pd.to_datetime(data["GIORNO"], errors="coerce")
                
        # 2. Store brand standardization
        if "Insegna_Standardized" not in data.columns or data["Insegna_Standardized"].isna().all():
            banner_candidate = data["nielsen_insegna"].fillna(data["CLIENTE"]) if "nielsen_insegna" in data.columns else data["CLIENTE"]
            if "Denominazione" in data.columns:
                banner_candidate = banner_candidate.fillna(data["Denominazione"])
            data["Insegna_Standardized"] = banner_candidate.apply(clean_brand_name)
        else:
            data["Insegna_Standardized"] = data["Insegna_Standardized"].fillna("INDIPENDENTE")
        
        # 3. Temporal conditioning features
        data["day_of_week"] = data["dt"].dt.dayofweek
        data["month_of_year"] = data["dt"].dt.month
        data["quarter"] = data["dt"].dt.quarter
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
        data["is_holiday"] = data["dt"].apply(is_italian_holiday)
        
        # Cyclical calendar transformations
        data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
        data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
        data["month_sin"] = np.sin(2 * np.pi * (data["month_of_year"] - 1) / 12.0)
        data["month_cos"] = np.cos(2 * np.pi * (data["month_of_year"] - 1) / 12.0)
        
        # 4. Cast categorical candidates to string first (prevents LossySetitemError)
        cat_cols = [
            "TipoMacchina", "Fornitore", "Provincia", "Regione", 
            "Insegna_Standardized", "nielsen_tipo_negozio", 
            "nielsen_gruppo", "nielsen_area"
        ]
        for c in cat_cols:
            if c in data.columns:
                data[c] = data[c].astype(str).replace(["nan", "None", "<NA>", "NAN", "NONE", ""], "UNKNOWN").str.strip().str.upper()

        # Auto-infer missing Nielsen Area from Region
        if "nielsen_area" in data.columns and "Regione" in data.columns:
            missing_area = data["nielsen_area"].isin(["UNKNOWN", "NAN", "NONE", ""])
            if missing_area.any():
                reg_map = {r: infer_nielsen_area(r) for r in data["Regione"].unique()}
                data.loc[missing_area, "nielsen_area"] = data.loc[missing_area, "Regione"].map(reg_map).fillna("UNKNOWN")
                
        # Auto-infer missing Nielsen Gruppo from banner, client name, and region
        if "nielsen_gruppo" in data.columns:
            missing_grp = data["nielsen_gruppo"].isin(["UNKNOWN", "NAN", "NONE", ""])
            if missing_grp.any():
                cols_to_use = [c for c in ["Insegna_Standardized", "CLIENTE", "Regione"] if c in data.columns]
                combos = data.loc[missing_grp, cols_to_use].drop_duplicates()
                combo_map = {}
                for _, row in combos.iterrows():
                    b = str(row.get("Insegna_Standardized", "INDIPENDENTE"))
                    c = str(row.get("CLIENTE", "")) if pd.notna(row.get("CLIENTE")) else ""
                    reg = str(row.get("Regione", "")) if pd.notna(row.get("Regione")) else ""
                    combo_map[tuple(row[c] for c in cols_to_use)] = infer_retail_group(b, c, reg)
                idx_tuples = list(zip(*[data.loc[missing_grp, c] for c in cols_to_use]))
                data.loc[missing_grp, "nielsen_gruppo"] = [combo_map.get(t, "UNKNOWN") for t in idx_tuples]

        # 5. Nielsen missingness flags & cleanup
        is_matched = (data.get("nielsen_match_status") == "MATCHED")
        is_known_brand = (data.get("Insegna_Standardized", "INDIPENDENTE") != "INDIPENDENTE")
        is_known_group = (~data.get("nielsen_gruppo", "UNKNOWN").isin(["UNKNOWN", "NAN", "NONE", ""]))
        data["is_nielsen_matched"] = (is_matched | is_known_brand | is_known_group).astype(int)
        
        data["nielsen_superficie_missing"] = data["nielsen_superficie_vendita"].isna().astype(int) if "nielsen_superficie_vendita" in data.columns else 1
        data["nielsen_casse_missing"] = data["nielsen_numero_casse"].isna().astype(int) if "nielsen_numero_casse" in data.columns else 1
                
        # 5. Continuous coordinates & postal code
        data["Latitudine"] = pd.to_numeric(data["Latitudine"], errors="coerce")
        data["Longitudine"] = pd.to_numeric(data["Longitudine"], errors="coerce")
        data["Cap"] = pd.to_numeric(data["Cap"], errors="coerce")
        
        # 6. Macro demographics & economics
        macro_cols = ["population", "surface", "population_density", "PIL", "PIL_per_capita"]
        for mc in macro_cols:
            if mc in data.columns:
                data[mc] = pd.to_numeric(data[mc], errors="coerce")
                
        # 7. Nielsen numerics
        data["nielsen_superficie_vendita"] = pd.to_numeric(data["nielsen_superficie_vendita"], errors="coerce")
        data["nielsen_numero_casse"] = pd.to_numeric(data["nielsen_numero_casse"], errors="coerce")
        
        # 8. Spatial competition & network density features
        spatial_cols = [
            "nearest_coripet_km", "coripet_rvm_count_5km", "is_isolated_rvm",
            "grocery_supermarket_count", "grocery_total_stores", "is_local_grocery_monopoly"
        ]
        if any(sc not in data.columns for sc in spatial_cols):
            sp_path = "data/processed/pf_spatial_features.parquet"
            if os.path.exists(sp_path) and "PF" in data.columns:
                df_sp = pd.read_parquet(sp_path)
                missing_sp_cols = [sc for sc in spatial_cols if sc not in data.columns and sc in df_sp.columns]
                if missing_sp_cols:
                    data = data.merge(df_sp[["PF"] + missing_sp_cols], on="PF", how="left")

        # Numeric conversions and robust fallback imputation
        data["nearest_coripet_km"] = pd.to_numeric(data.get("nearest_coripet_km"), errors="coerce")
        data["coripet_rvm_count_5km"] = pd.to_numeric(data.get("coripet_rvm_count_5km"), errors="coerce")
        data["grocery_supermarket_count"] = pd.to_numeric(data.get("grocery_supermarket_count"), errors="coerce")
        data["grocery_total_stores"] = pd.to_numeric(data.get("grocery_total_stores"), errors="coerce")
        data["is_local_grocery_monopoly"] = pd.to_numeric(data.get("is_local_grocery_monopoly"), errors="coerce").fillna(0).astype(int)
        data["is_isolated_rvm"] = pd.to_numeric(data.get("is_isolated_rvm"), errors="coerce").fillna(0).astype(int)

        return data

    def fit(self, train_df: pd.DataFrame):
        """Fits scalers, encoders, and imputation statistics strictly on training data."""
        print(f"Fitting preprocessor on training data ({len(train_df):,} rows)...")
        df = self.prepare_base_features(train_df)
        
        # Numerical feature definitions
        # Skewed features that receive log1p:
        log_skewed = [
            "population", "surface", "population_density", "PIL", "nielsen_superficie_vendita",
            "nearest_coripet_km"
        ]
        regular_num = [
            "Latitudine", "Longitudine", "Cap", "PIL_per_capita", "nielsen_numero_casse",
            "coripet_rvm_count_5km", "grocery_supermarket_count", "grocery_total_stores"
        ]
        binary_num = [
            "is_nielsen_matched", "nielsen_superficie_missing", "nielsen_casse_missing",
            "is_weekend", "is_holiday", "dow_sin", "dow_cos", "month_sin", "month_cos", "quarter",
            "is_local_grocery_monopoly", "is_isolated_rvm"
        ]
        
        # Compute training medians for imputation
        for c in log_skewed + regular_num:
            val_median = float(df[c].dropna().median()) if df[c].dropna().shape[0] > 0 else 0.0
            self.num_medians[c] = val_median
            
        # Build categorical vocabularies
        self.cat_features = [
            "TipoMacchina", "Fornitore", "Provincia", "Regione", 
            "Insegna_Standardized", "nielsen_tipo_negozio", 
            "nielsen_gruppo", "nielsen_area"
        ]
        
        for c in self.cat_features:
            unique_vals = df[c].value_counts().index.tolist()
            # Index 0 is reserved for UNKNOWN
            vocab = {"<UNK>": 0}
            idx = 1
            for v in unique_vals:
                if v != "UNKNOWN" and v != "<UNK>":
                    vocab[v] = idx
                    idx += 1
            self.cat_vocabs[c] = vocab
            print(f"  Categorical '{c}': {len(vocab)} categories (including <UNK>)")
            
        # Prepare numeric matrix for StandardScaler fitting (fit on representative sample to save memory)
        sample_size = min(len(df), 200000)
        sample_df = df.iloc[:sample_size]
        X_num_sample = self._extract_num_matrix(sample_df, is_fit=True)
        self.scaler.fit(X_num_sample)
        del X_num_sample
        
        self.num_features = (
            [f"{c}_log" for c in log_skewed] + 
            regular_num + 
            binary_num
        )
        self.all_feature_names = self.num_features + self.cat_features
        self.fitted = True
        print(f"Fitted preprocessor: {len(self.num_features)} numeric features + {len(self.cat_features)} categorical features.")
        return self

    def _extract_num_matrix(self, df: pd.DataFrame, is_fit: bool = False) -> np.ndarray:
        """Extracts and transforms numerical features with strict median imputation directly into float32."""
        log_skewed = [
            "population", "surface", "population_density", "PIL", "nielsen_superficie_vendita",
            "nearest_coripet_km"
        ]
        regular_num = [
            "Latitudine", "Longitudine", "Cap", "PIL_per_capita", "nielsen_numero_casse",
            "coripet_rvm_count_5km", "grocery_supermarket_count", "grocery_total_stores"
        ]
        binary_num = [
            "is_nielsen_matched", "nielsen_superficie_missing", "nielsen_casse_missing",
            "is_weekend", "is_holiday", "dow_sin", "dow_cos", "month_sin", "month_cos", "quarter",
            "is_local_grocery_monopoly", "is_isolated_rvm"
        ]
        
        n_rows = len(df)
        total_num_cols = len(log_skewed) + len(regular_num) + len(binary_num)
        out = np.empty((n_rows, total_num_cols), dtype=np.float32)
        idx = 0
        
        # 1. Log-skewed features
        for c in log_skewed:
            med = self.num_medians.get(c, 2.0)
            s = df[c].fillna(med).clip(lower=0)
            out[:, idx] = np.log1p(s.values.astype(np.float32))
            idx += 1
            
        # 2. Regular numeric features
        for c in regular_num:
            med = self.num_medians.get(c, 0.0)
            s = df[c].fillna(med)
            out[:, idx] = s.values.astype(np.float32)
            idx += 1
            
        # 3. Binary & cyclical features
        for c in binary_num:
            out[:, idx] = df[c].fillna(0).values.astype(np.float32)
            idx += 1
            
        return out

    def transform(self, df: pd.DataFrame, nielsen_dropout: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforms input dataframe into model-ready numpy arrays.
        Returns: (X_num, X_cat, y_bottiglie, y_presenze)
        """
        if not self.fitted:
            raise ValueError("RVMPreprocessor must be fitted before calling transform!")
            
        data = self.prepare_base_features(df)
        
        # Optional training feature dropout for Nielsen features
        if nielsen_dropout > 0.0:
            mask = np.random.rand(len(data)) < nielsen_dropout
            if mask.any():
                data.loc[mask, "is_nielsen_matched"] = 0
                data.loc[mask, "nielsen_superficie_vendita"] = np.nan
                data.loc[mask, "nielsen_numero_casse"] = np.nan
                data.loc[mask, "nielsen_tipo_negozio"] = "UNKNOWN"
                data.loc[mask, "nielsen_gruppo"] = "UNKNOWN"
                data.loc[mask, "nielsen_area"] = "UNKNOWN"
                data.loc[mask, "nielsen_superficie_missing"] = 1
                data.loc[mask, "nielsen_casse_missing"] = 1
                
        # Transform numerical features
        X_num_raw = self._extract_num_matrix(data)
        X_num = self.scaler.transform(X_num_raw).astype(np.float32)
        del X_num_raw
        
        # Transform categorical features
        X_cat = np.empty((len(data), len(self.cat_features)), dtype=np.int64)
        for i, c in enumerate(self.cat_features):
            vocab = self.cat_vocabs[c]
            s = data[c].astype(str).str.strip().str.upper()
            X_cat[:, i] = s.map(lambda val: vocab.get(val, 0)).values
        
        # Extract targets if present
        y_bottiglie = np.zeros(len(data), dtype=np.float32)
        y_presenze = np.zeros(len(data), dtype=np.float32)
        if "BOTTIGLIE" in data.columns:
            y_bottiglie = np.log1p(data["BOTTIGLIE"].clip(lower=0).values).astype(np.float32)
        if "PRESENZE" in data.columns:
            y_presenze = np.log1p(data["PRESENZE"].clip(lower=0).values).astype(np.float32)
            
        return X_num, X_cat, y_bottiglie, y_presenze

    def get_cat_cardinalities(self) -> Dict[str, int]:
        """Returns vocabulary size for each categorical feature."""
        return {c: len(vocab) for c, vocab in self.cat_vocabs.items()}

    def save(self, filepath: str = "models/preprocessing/preprocessor.joblib"):
        """Saves fitted preprocessor and feature configuration."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        
        config = {
            "num_features": self.num_features,
            "cat_features": self.cat_features,
            "cat_cardinalities": self.get_cat_cardinalities(),
            "num_medians": self.num_medians,
            "random_seed": self.random_seed
        }
        config_path = os.path.join(os.path.dirname(filepath), "feature_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved preprocessor to {filepath} and configuration to {config_path}")

    @classmethod
    def load(cls, filepath: str = "models/preprocessing/preprocessor.joblib") -> "RVMPreprocessor":
        """Loads fitted preprocessor from file."""
        return joblib.load(filepath)

def split_pfs(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs group-separated split by Physical Point (PF):
    - 20% of unique PFs -> Test set (Completely unseen physical points)
    - From the remaining 80% PFs:
        - 80% -> Train set (~64% of total)
        - 20% -> Validation set (~16% of total)
    """
    unique_pfs = np.array(sorted(df["PF"].unique()))
    n_pfs = len(unique_pfs)
    print(f"Splitting {n_pfs:,} unique PFs into group-separated Train/Val/Test...")
    
    rng = np.random.RandomState(seed)
    shuffled_pfs = rng.permutation(unique_pfs)
    
    # 20% test
    n_test = int(np.round(0.20 * n_pfs))
    test_pfs = set(shuffled_pfs[:n_test])
    dev_pfs = shuffled_pfs[n_test:]
    
    # 20% of dev as validation
    n_val = int(np.round(0.20 * len(dev_pfs)))
    val_pfs = set(dev_pfs[:n_val])
    train_pfs = set(dev_pfs[n_val:])
    
    train_df = df[df["PF"].isin(train_pfs)].copy()
    val_df = df[df["PF"].isin(val_pfs)].copy()
    test_df = df[df["PF"].isin(test_pfs)].copy()
    
    # Verify zero overlap
    assert len(train_pfs.intersection(val_pfs)) == 0, "Leakage: Overlap between train and val PFs!"
    assert len(train_pfs.intersection(test_pfs)) == 0, "Leakage: Overlap between train and test PFs!"
    assert len(val_pfs.intersection(test_pfs)) == 0, "Leakage: Overlap between val and test PFs!"
    
    split_info = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_pfs": len(train_pfs),
        "val_pfs": len(val_pfs),
        "test_pfs": len(test_pfs),
        "total_rows": len(df),
        "total_pfs": n_pfs,
        "seed": seed
    }
    
    print(f"Split completed successfully:")
    print(f"  Train: {split_info['train_rows']:,} rows across {split_info['train_pfs']} PFs ({split_info['train_rows']/len(df)*100:.1f}%)")
    print(f"  Val:   {split_info['val_rows']:,} rows across {split_info['val_pfs']} PFs ({split_info['val_rows']/len(df)*100:.1f}%)")
    print(f"  Test:  {split_info['test_rows']:,} rows across {split_info['test_pfs']} PFs ({split_info['test_rows']/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df, split_info
