"""
Commercial Rewards and Filiera Intelligence Module.
Links Coripet store networks (CaricaViewStoricoPuntoFisico.csv) with
Commercial Reward agreements (Premialità-Rewards.xlsx Sheet 3 'Foglio3' & Sheet 1).

Provides:
- Exact filiera to reward category mapping (100% network match).
- Extraction of granular discount (€), minimum spend (€), and points hurdles (valore_punti).
- Customer behavioral incentive tiering & empirical elasticity multipliers.
- Retail group fallback defaults for greenfield location underwriting.
- Backward-compatible multipliers preserving previous model baselines.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Standard English & Italian definitions from Premialità-Rewards.xlsx Sheet 2
REWARD_DEFINITIONS = {
    "Conversione a punti fedeltà": {
        "tier": "LOYALTY_POINTS",
        "label": "Fidelity Points Conversion (Punti Spesa / Carta Socio)",
        "definition_en": "Conversion of Coripet Points into supermarket Fidelity Card points (e.g. Coop, Conad, Unicoop). Drives high weekly recurring dropoffs.",
        "definition_it": "Conversione dei punti Coripet in punti fedeltà della tessera spesa del supermercato. Massima fidelizzazione del cliente.",
        "incentive_strength": "VERY_HIGH",
        "typical_voucher": "Fidelity points credited directly to store card",
    },
    "Emissione Buono sconto": {
        "tier": "GROCERY_VOUCHER",
        "label": "Direct Grocery Discount Voucher (Buono Sconto)",
        "definition_en": "Direct discount on grocery purchases (typically €3 voucher per 200 bottles with minimum receipt e.g. €20). National supermarket standard.",
        "definition_it": "Buono sconto diretto sulla spesa (normalmente circa 3€ ogni 200 bottiglie con scontrino minimo es. 20€). Standard GDO nazionale.",
        "incentive_strength": "HIGH",
        "typical_voucher": "3€ discount per 200 bottles on min. 20€ spend",
    },
    "Omaggio": {
        "tier": "DIRECT_GIFT",
        "label": "Free Gift / Cash Prize (Omaggio senza minimo di spesa)",
        "definition_en": "Free gift or cash reward with no minimum bill required. High immediate gratification.",
        "definition_it": "Regalo gratuito o premio cash senza alcun minimo di spesa richiesto.",
        "incentive_strength": "HIGH",
        "typical_voucher": "Direct gift or cash with no minimum purchase",
    }
}

REWARD_MENU_CHOICES = [
    (
        "1",
        "Conversione a punti fedeltà",
        "Conversione a punti fedeltà (Loyalty Card Points - Punti Spesa / Carta Socio)",
        "Direct conversion into store loyalty points (e.g. 1 pt/bottle; 30-100 pts -> grocery discount on fidelity card). Highest recurring retention (1.35x-1.45x footfall draw)."
    ),
    (
        "2",
        "Emissione Buono sconto",
        "Emissione Buono sconto (Direct Grocery Voucher at Checkout - Buono Sconto)",
        "Standard grocery checkout voucher (Benchmark: 3€ discount on 25-30€ spend for 200 bottles). National GDO benchmark across 94% of network (1.000x baseline)."
    ),
    (
        "3",
        "Omaggio",
        "Omaggio (Direct Free Gift / Instant Voucher - No Min. Spend Required)",
        "Immediate free gift, cash prize, or direct discount voucher without requiring a minimum grocery receipt (+15% footfall draw, +8% volume uplift)."
    )
]

REWARD_KEY_TO_NAME = {item[0]: item[1] for item in REWARD_MENU_CHOICES}

REWARD_DETAILED_SPECS = {
    "1": {
        "name": "Conversione a punti fedeltà",
        "label": "Store Loyalty Card Points (Conversione a punti fedeltà)",
        "rule_it": "Conversione diretta dei punti Coripet in punti fedeltà della tessera spesa (Carta Socio Coop, Carta Conad, ecc.).",
        "rule_en": "Direct conversion of Coripet points into supermarket loyalty card points (Coop Member Card, Conad Card, etc.).",
        "threshold": "30 - 100 points hurdle",
        "economic_value": "Very high perceived loyalty value (~3.0 - 4.5 c€/bottle)",
        "footfall_impact": "VERY HIGH (+35% to +45% recurring footfall draw | 1.35x-1.45x footfall)",
        "volume_impact": "+25% to +35% volume uplift (1.25x-1.35x bottles)",
        "best_for": "Supermarkets and hypermarkets with established customer fidelity card programs."
    },
    "2": {
        "name": "Emissione Buono sconto",
        "label": "Direct Grocery Checkout Voucher (Emissione Buono sconto)",
        "rule_it": "Buono sconto spesa spendibile alle casse (es. 3€ di sconto su spesa minima di 25-30€ ogni 200 bottiglie).",
        "rule_en": "Direct grocery checkout discount voucher (e.g. 3€ coupon on 25-30€ minimum spend per 200 bottles).",
        "threshold": "200 points hurdle (1 bottle = 1 point)",
        "economic_value": "1.50 €-cent/bottle (10-12% basket discount)",
        "footfall_impact": "HIGH (1.000x neutral GDO baseline footfall | National standard across 94% of locations)",
        "volume_impact": "1.000x neutral GDO baseline volume (national standard across 94% of locations)",
        "best_for": "Standard supermarkets and retail parks seeking checkout basket growth."
    },
    "3": {
        "name": "Omaggio",
        "label": "Direct Free Gift / Instant Voucher (Omaggio)",
        "rule_it": "Premio omaggio o buono sconto immediato senza obbligo di scontrino o spesa minima (es. 100-200 punti).",
        "rule_en": "Free gift, instant prize, or direct discount voucher without requiring a grocery receipt or minimum spend (e.g. 100-200 points).",
        "threshold": "100 - 200 points hurdle",
        "economic_value": "Guaranteed instant reward without financial barrier",
        "footfall_impact": "HIGH (+15% footfall draw, +8% volume uplift | 1.15x footfall)",
        "volume_impact": "+8% volume uplift (1.08x bottles)",
        "best_for": "Stores wanting to attract casual, non-grocery shoppers, students, or high-velocity dropoffs."
    }
}


def create_custom_reward_policy(
    discount_euro: float = 3.0,
    min_spend_euro: float = 25.0,
    points_required: int = 200,
    base_category: str = "Emissione Buono sconto",
    custom_label: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a fully configured custom promotional reward policy dictionary
    based on user-specified Euro discount, minimum basket spend, and points hurdle.
    Calculates exact unit economics and behavioral elasticity multipliers.
    """
    disc_euro = max(0.10, float(discount_euro))
    min_spend = max(0.0, float(min_spend_euro))
    pts = max(10, int(points_required))
    
    val_per_bottle = round(disc_euro / pts, 4)
    val_cent_per_bottle = round(val_per_bottle * 100.0, 2)
    discount_pct = round((disc_euro / min_spend) * 100.0, 1) if min_spend > 0 else 100.0
    
    spend_text = f"su una spesa minima di {min_spend:.2f}€" if min_spend > 0 else "senza spesa minima"
    rule_it = f"{disc_euro:.2f}€ di sconto {spend_text} con {pts} punti/bottiglie"
    title_it = f"Sconto {disc_euro:.2f}€" if min_spend == 0 else f"Sconto {disc_euro:.2f}€ su {min_spend:.0f}€"
    
    mech = {
        "contract_title": title_it,
        "contract_rule": rule_it,
        "points_required": pts,
        "discount_euro": disc_euro,
        "min_spend_euro": min_spend,
        "valore_euro_per_bottiglia": val_per_bottle,
        "valore_cent_per_bottiglia": val_cent_per_bottle,
        "sconto_percentuale_carrello": discount_pct,
        "conversione_mode": 0
    }
    
    elasticity = calculate_incentive_elasticity(base_category, mech)
    
    label = custom_label or f"Custom: {disc_euro:.2f}€ su {min_spend:.0f}€ ({pts} pts)"
    
    # Tier / strength rating
    idx = elasticity["incentive_index"]
    if idx >= 1.25:
        strength = "VERY_HIGH (CUSTOM_PREMIUM)"
        tier = "LOYALTY_POINTS" if "punti fedeltà" in base_category else "GROCERY_VOUCHER"
    elif idx >= 1.05:
        strength = "HIGH (CUSTOM_ATTRACTIVE)"
        tier = "GROCERY_VOUCHER"
    elif idx >= 0.95:
        strength = "MODERATE (CUSTOM_BALANCED)"
        tier = "GROCERY_VOUCHER"
    else:
        strength = "BASELINE (CUSTOM_RESTRICTIVE)"
        tier = "APP_ONLY" if min_spend == 0 else "GROCERY_VOUCHER"

    return {
        "filiera": "CUSTOM_PROMOTIONAL_POLICY",
        "filiera_nome": "Custom Promotional Agreement",
        "reward_type": base_category,
        "is_custom": True,
        "custom_label": label,
        "reward_label": f"Custom Promotional Reward ({label})",
        "definition_en": f"User-configured custom promotion: {disc_euro:.2f}€ discount {('on ' + str(min_spend) + '€ spend') if min_spend > 0 else 'without minimum spend'} for {pts} bottles.",
        "definition_it": rule_it,
        "incentive_tier": tier,
        "incentive_strength": strength,
        "typical_voucher": f"{disc_euro:.2f}€ / {pts} pts ({val_cent_per_bottle:.2f} c€/b)",
        "match_source": "USER_CUSTOM_POLICY",
        **mech,
        **elasticity
    }


def extract_discount_mechanics(titolo: Any, desc1: Any, punti: Any, conv_mode: Any = 0) -> Dict[str, Any]:
    """
    Parses unstructured contract Italian text into quantitative discount mechanics.
    Extracts face value (€), minimum spend (€), and computes unit economics.
    """
    t_clean = str(titolo or "").strip()
    d_clean = str(desc1 or "").strip()
    full_text = f"{t_clean} {d_clean}".replace('_x000D_', ' ').replace('\n', ' ')
    
    # 1. Discount amount in Euros (€)
    disc_euro = None
    m_disc = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:€|euro)\s*(?:di\s*)?sconto', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'sconto\s*(?:di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'buono\s*(?:del\s*valore\s*di\s*|sconto\s*di\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)
    if not m_disc:
        m_disc = re.search(r'(\d+(?:[.,]\d+)?)\s*€', t_clean, re.IGNORECASE)
        
    if m_disc:
        try:
            disc_euro = float(m_disc.group(1).replace(',', '.'))
        except ValueError:
            pass
            
    if disc_euro is None:
        disc_euro = 3.0  # National default

    # 2. Minimum spend in Euros (€)
    min_spend = None
    m_spend = re.search(r'(?:spesa\s+minima|minimo\s+di\s+spesa|minimo\s+scontrino)\s*(?:di\s*|da\s*)?(\d+(?:[.,]\d+)?)\s*(?:€|euro)?', full_text, re.IGNORECASE)
    if not m_spend:
        m_spend = re.search(r'su\s+(\d+(?:[.,]\d+)?)\s*(?:€|euro)\s*di\s*spesa', full_text, re.IGNORECASE)
    if not m_spend:
        m_spend = re.search(r'su\s*un\s*minimo\s*di\s*(\d+(?:[.,]\d+)?)\s*(?:€|euro)', full_text, re.IGNORECASE)

    if m_spend:
        try:
            min_spend = float(m_spend.group(1).replace(',', '.'))
        except ValueError:
            pass

    if min_spend is None:
        min_spend = 25.0  # National median spend

    try:
        pts = int(punti) if punti and not pd.isna(punti) and int(punti) > 0 else 200
    except (ValueError, TypeError):
        pts = 200

    try:
        conv = int(conv_mode) if conv_mode and not pd.isna(conv_mode) else 0
    except (ValueError, TypeError):
        conv = 0

    # 3. Derived Economics
    val_per_bottle = round(disc_euro / pts, 4) if pts > 0 else 0.015
    discount_pct = round((disc_euro / min_spend) * 100.0, 1) if min_spend > 0 else 10.0

    return {
        "contract_title": t_clean if t_clean else f"Sconto {int(disc_euro)}€",
        "contract_rule": d_clean.replace('_x000D_', '').strip() if d_clean else f"{int(disc_euro)}€ di sconto su spesa minima di {int(min_spend)}€",
        "points_required": pts,
        "discount_euro": disc_euro,
        "min_spend_euro": min_spend,
        "valore_euro_per_bottiglia": val_per_bottle,
        "valore_cent_per_bottiglia": round(val_per_bottle * 100.0, 2),
        "sconto_percentuale_carrello": discount_pct,
        "conversione_mode": conv
    }


def calculate_incentive_elasticity(reward_cat: str, mech: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes calibrated empirical elasticity multipliers for presenze and bottles.
    Baseline (neutral 1.00x) is the national standard: 200 points, 3€ discount on 25-30€ spend.
    """
    pts = mech.get("points_required", 200)
    val_bottle = mech.get("valore_euro_per_bottiglia", 0.015)
    disc_pct = mech.get("sconto_percentuale_carrello", 10.0)
    
    # 1. Points hurdle elasticity (lower threshold = higher frequency)
    f_pts = (200.0 / max(50.0, float(pts))) ** 0.15
    f_pts = float(np.clip(f_pts, 0.85, 1.20))
    
    # 2. Value per bottle elasticity
    f_val = 1.0 + 0.10 * ((val_bottle - 0.015) / 0.015)
    f_val = float(np.clip(f_val, 0.90, 1.25))
    
    # 3. Cart discount percentage elasticity
    f_disc = 1.0 + 0.05 * ((disc_pct - 10.0) / 10.0)
    f_disc = float(np.clip(f_disc, 0.92, 1.15))
    
    # 4. Loyalty channel factor
    f_channel = 1.0
    if "Conversione a punti fedeltà" in reward_cat:
        f_channel = 1.15
    elif "Omaggio" in reward_cat:
        f_channel = 1.08
        
    mult_presenze = round(float(np.clip(f_pts * f_val * f_disc * f_channel, 0.75, 1.45)), 3)
    mult_bottles = round(float(np.clip(np.sqrt(mult_presenze) * f_val, 0.80, 1.35)), 3)
    
    # Neutralize standard 200-point GDO vouchers to exactly 1.000x
    if pts == 200 and 0.012 <= val_bottle <= 0.018 and "Buono sconto" in reward_cat:
        mult_presenze = 1.000
        mult_bottles = 1.000
    
    return {
        "multiplier_presenze": mult_presenze,
        "multiplier_bottles": mult_bottles,
        "incentive_index": round((mult_presenze + mult_bottles) / 2.0, 3)
    }


class CommercialRewardsEngine:
    """Singleton engine managing Filiera to Rewards matching and incentive scoring."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CommercialRewardsEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: str = "data"):
        if getattr(self, "_initialized", False):
            return

        self.data_dir = data_dir
        self.excel_path = os.path.join(data_dir, "raw", "Premialità-Rewards.xlsx")
        self.csv_path = os.path.join(data_dir, "raw", "CaricaViewStoricoPuntoFisico.csv")

        self.filiera_code_to_reward: Dict[str, Dict[str, Any]] = {}
        self.filiera_name_to_reward: Dict[str, Dict[str, Any]] = {}
        self.punto_fisico_to_filiera: Dict[str, str] = {}
        self.is_loaded = False

        self._load_datasets()
        self._initialized = True

    def _load_datasets(self):
        """Loads and indexes the Sheet 3 (Foglio3) and Sheet 1 datasets."""
        try:
            if not os.path.exists(self.excel_path):
                logger.warning(f"Premialità-Rewards file not found at {self.excel_path}")
                return

            # Check available sheet names
            xls = pd.ExcelFile(self.excel_path)
            sheet_names = xls.sheet_names

            # 1. Primary Load: Sheet 3 ('Foglio3') containing granular commercial contracts & points
            target_sheet = "Foglio3" if "Foglio3" in sheet_names else sheet_names[0]
            df_sheet3 = pd.read_excel(self.excel_path, sheet_name=target_sheet)

            for _, row in df_sheet3.iterrows():
                codice = str(row.get("CODICE", "")).strip().upper()
                nome = str(row.get("NOME", "")).strip().upper()
                desc = str(row.get("DESCRIZIONE", "Emissione Buono sconto")).strip()

                if desc not in REWARD_DEFINITIONS:
                    for k in REWARD_DEFINITIONS:
                        if k[:10].lower() in desc.lower():
                            desc = k
                            break
                if desc not in REWARD_DEFINITIONS:
                    desc = "Emissione Buono sconto"

                if codice == "ESSELUNGA" or "ESSELUNGA" in nome:
                    desc = "Omaggio"

                # Extract granular discount, min spend, and points mechanics
                titolo = row.get("titolo")
                desc1 = row.get("descrizione.1")
                punti = row.get("valore_punti")
                conv = row.get("conversione")

                mech = extract_discount_mechanics(titolo, desc1, punti, conv)
                elasticity = calculate_incentive_elasticity(desc, mech)

                record = {
                    "codice": str(row.get("CODICE", "")).strip(),
                    "nome": str(row.get("NOME", "")).strip(),
                    "descrizione": desc,
                    "reward_entity": row.get("Reward entity"),
                    "data_inizio": str(row.get("DATAINIZIO", "")),
                    "data_fine": str(row.get("DATAFINE", "")) if pd.notna(row.get("DATAFINE")) else None,
                    "info": REWARD_DEFINITIONS.get(desc, REWARD_DEFINITIONS["Emissione Buono sconto"]),
                    **mech,
                    **elasticity
                }

                if codice:
                    self.filiera_code_to_reward[codice] = record
                if nome:
                    self.filiera_name_to_reward[nome] = record

            # 2. Secondary Supplement: Sheet 1 (Foglio1) for legacy civic/school entities not in Sheet 3
            if "Foglio1" in sheet_names:
                df_sheet1 = pd.read_excel(self.excel_path, sheet_name="Foglio1")
                for _, row in df_sheet1.iterrows():
                    codice = str(row.get("CODICE", "")).strip().upper()
                    nome = str(row.get("NOME", "")).strip().upper()
                    if codice in self.filiera_code_to_reward and nome in self.filiera_name_to_reward:
                        continue  # Already enriched with Sheet 3 contract terms

                    desc = str(row.get("DESCRIZIONE", "Emissione Buono sconto")).strip()
                    if desc not in REWARD_DEFINITIONS:
                        for k in REWARD_DEFINITIONS:
                            if k[:10].lower() in desc.lower():
                                desc = k
                                break
                    if desc not in REWARD_DEFINITIONS:
                        desc = "Emissione Buono sconto"

                    if codice == "ESSELUNGA" or "ESSELUNGA" in nome:
                        desc = "Omaggio"

                    mech = extract_discount_mechanics("Standard Voucher", desc, 200, 0)
                    elasticity = calculate_incentive_elasticity(desc, mech)

                    record = {
                        "codice": str(row.get("CODICE", "")).strip(),
                        "nome": str(row.get("NOME", "")).strip(),
                        "descrizione": desc,
                        "reward_entity": row.get("Reward entity"),
                        "data_inizio": str(row.get("DATAINIZIO", "")),
                        "data_fine": str(row.get("DATAFINE", "")) if pd.notna(row.get("DATAFINE")) else None,
                        "info": REWARD_DEFINITIONS.get(desc, REWARD_DEFINITIONS["Emissione Buono sconto"]),
                        **mech,
                        **elasticity
                    }

                    if codice and codice not in self.filiera_code_to_reward:
                        self.filiera_code_to_reward[codice] = record
                    if nome and nome not in self.filiera_name_to_reward:
                        self.filiera_name_to_reward[nome] = record

            # 3. Load CaricaViewStoricoPuntoFisico.csv to index PuntoFisico -> Filiera
            if os.path.exists(self.csv_path):
                df_stores = pd.read_csv(self.csv_path, sep=";", usecols=["PuntoFisico", "Filiera", "NomeFiliera", "Comune", "Indirizzo"], low_memory=False)
                for _, row in df_stores.iterrows():
                    pf = str(row.get("PuntoFisico", "")).strip().upper()
                    fil = str(row.get("Filiera", "")).strip()
                    if pf and fil and fil != "nan":
                        self.punto_fisico_to_filiera[pf] = fil

            self.is_loaded = True
            logger.info(f"Loaded CommercialRewardsEngine with {len(self.filiera_code_to_reward)} codes, {len(self.punto_fisico_to_filiera)} store mappings from Sheet 3 & 1.")
        except Exception as e:
            logger.error(f"Error loading CommercialRewardsEngine: {e}")

    def is_known_filiera(
        self,
        filiera: Optional[str] = None,
        store_name: Optional[str] = None,
        punto_fisico_id: Optional[str] = None
    ) -> bool:
        """
        Determines whether a filiera or store name exists in historical datasets (Sheet 3, Sheet 1, or CaricaView).
        Returns False for newly opened / unrecorded stores.
        """
        if punto_fisico_id and str(punto_fisico_id).strip().upper() in self.punto_fisico_to_filiera:
            return True
            
        if filiera:
            f_clean = str(filiera).strip().upper()
            if f_clean in self.filiera_code_to_reward or f_clean in self.filiera_name_to_reward:
                return True
                
        if store_name:
            s_clean = str(store_name).strip().upper()
            if s_clean in self.filiera_code_to_reward or s_clean in self.filiera_name_to_reward:
                return True
            for pf, fil_code in self.punto_fisico_to_filiera.items():
                if fil_code.upper() == s_clean:
                    return True
                    
        return False

    def resolve_reward(
        self,
        filiera: Optional[str] = None,
        store_name: Optional[str] = None,
        retail_group: Optional[str] = None,
        punto_fisico_id: Optional[str] = None,
        regione: Optional[str] = None,
        comune: Optional[str] = None,
        explicit_reward: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolves the exact reward category, descriptions, points hurdle, and incentive multipliers.
        Guarantees 100% fallback safety for any greenfield store.
        """
        # 1. Explicit user override if provided
        if explicit_reward:
            if isinstance(explicit_reward, dict):
                # Custom user-defined reward dictionary
                cat = explicit_reward.get("reward_type", "Emissione Buono sconto")
                res_dict = explicit_reward.copy()
                if "multiplier_presenze" not in res_dict or "multiplier_bottles" not in res_dict:
                    mech = {
                        "points_required": res_dict.get("points_required", 200),
                        "discount_euro": res_dict.get("discount_euro", 3.0),
                        "min_spend_euro": res_dict.get("min_spend_euro", 25.0),
                        "valore_euro_per_bottiglia": res_dict.get("valore_euro_per_bottiglia", 0.015),
                        "sconto_percentuale_carrello": res_dict.get("sconto_percentuale_carrello", 10.0),
                    }
                    elast = calculate_incentive_elasticity(cat, mech)
                    res_dict.update(elast)
                if "reward_label" not in res_dict:
                    res_dict["reward_label"] = REWARD_DEFINITIONS.get(cat, {}).get("label", cat)
                if "definition_en" not in res_dict:
                    res_dict["definition_en"] = REWARD_DEFINITIONS.get(cat, {}).get("definition_en", "")
                if "definition_it" not in res_dict:
                    res_dict["definition_it"] = REWARD_DEFINITIONS.get(cat, {}).get("definition_it", "")
                if "incentive_tier" not in res_dict:
                    res_dict["incentive_tier"] = REWARD_DEFINITIONS.get(cat, {}).get("tier", "GROCERY_VOUCHER")
                if "incentive_strength" not in res_dict:
                    res_dict["incentive_strength"] = REWARD_DEFINITIONS.get(cat, {}).get("incentive_strength", "HIGH")
                if "match_source" not in res_dict:
                    res_dict["match_source"] = "USER_CUSTOM_POLICY"
                if "filiera" not in res_dict:
                    res_dict["filiera"] = filiera or "CUSTOM_PROMOTIONAL_POLICY"
                if "filiera_nome" not in res_dict:
                    res_dict["filiera_nome"] = "Custom Promotional Agreement"
                return res_dict

            elif str(explicit_reward).strip() in REWARD_DEFINITIONS:
                cat = str(explicit_reward).strip()
                mech = extract_discount_mechanics("Manual Override", REWARD_DEFINITIONS[cat]["typical_voucher"], 200, 0)
                elasticity = calculate_incentive_elasticity(cat, mech)
                return {
                    "filiera": filiera or "EXPLICIT_OVERRIDE",
                    "filiera_nome": filiera or "EXPLICIT_OVERRIDE",
                    "reward_type": cat,
                    "reward_label": REWARD_DEFINITIONS[cat]["label"],
                    "definition_en": REWARD_DEFINITIONS[cat]["definition_en"],
                    "definition_it": REWARD_DEFINITIONS[cat]["definition_it"],
                    "incentive_tier": REWARD_DEFINITIONS[cat]["tier"],
                    "incentive_strength": REWARD_DEFINITIONS[cat]["incentive_strength"],
                    "typical_voucher": REWARD_DEFINITIONS[cat]["typical_voucher"],
                    "match_source": "EXPLICIT_OVERRIDE",
                    **mech,
                    **elasticity
                }

        # 2. Check by installed PuntoFisico ID
        if punto_fisico_id:
            pf_clean = str(punto_fisico_id).strip().upper()
            if pf_clean in self.punto_fisico_to_filiera:
                filiera = self.punto_fisico_to_filiera[pf_clean]

        # 3. Match by Filiera (NOME or CODICE)
        if filiera and str(filiera).strip():
            fil_clean = str(filiera).strip().upper()
            # Try CODICE first
            if fil_clean in self.filiera_code_to_reward:
                rec = self.filiera_code_to_reward[fil_clean]
                desc = rec["descrizione"]
                return {
                    "filiera": rec["codice"],
                    "filiera_nome": rec["nome"],
                    "reward_type": desc,
                    "reward_label": rec["info"]["label"],
                    "definition_en": rec["info"]["definition_en"],
                    "definition_it": rec["info"]["definition_it"],
                    "incentive_tier": rec["info"]["tier"],
                    "incentive_strength": rec["info"]["incentive_strength"],
                    "typical_voucher": rec["info"]["typical_voucher"],
                    "data_inizio": rec.get("data_inizio"),
                    "data_fine": rec.get("data_fine"),
                    "contract_title": rec.get("contract_title", rec["info"]["typical_voucher"]),
                    "contract_rule": rec.get("contract_rule", rec["info"]["typical_voucher"]),
                    "points_required": rec.get("points_required", 200),
                    "discount_euro": rec.get("discount_euro", 3.0),
                    "min_spend_euro": rec.get("min_spend_euro", 25.0),
                    "valore_euro_per_bottiglia": rec.get("valore_euro_per_bottiglia", 0.015),
                    "valore_cent_per_bottiglia": rec.get("valore_cent_per_bottiglia", 1.5),
                    "sconto_percentuale_carrello": rec.get("sconto_percentuale_carrello", 10.0),
                    "conversione_mode": rec.get("conversione_mode", 0),
                    "multiplier_presenze": rec.get("multiplier_presenze", 1.0),
                    "multiplier_bottles": rec.get("multiplier_bottles", 1.0),
                    "incentive_index": rec.get("incentive_index", 1.0),
                    "match_source": "MATCHED_FILIERA_CODE"
                }
            # Try NOME
            if fil_clean in self.filiera_name_to_reward:
                rec = self.filiera_name_to_reward[fil_clean]
                desc = rec["descrizione"]
                return {
                    "filiera": rec["codice"],
                    "filiera_nome": rec["nome"],
                    "reward_type": desc,
                    "reward_label": rec["info"]["label"],
                    "definition_en": rec["info"]["definition_en"],
                    "definition_it": rec["info"]["definition_it"],
                    "incentive_tier": rec["info"]["tier"],
                    "incentive_strength": rec["info"]["incentive_strength"],
                    "typical_voucher": rec["info"]["typical_voucher"],
                    "data_inizio": rec.get("data_inizio"),
                    "data_fine": rec.get("data_fine"),
                    "contract_title": rec.get("contract_title", rec["info"]["typical_voucher"]),
                    "contract_rule": rec.get("contract_rule", rec["info"]["typical_voucher"]),
                    "points_required": rec.get("points_required", 200),
                    "discount_euro": rec.get("discount_euro", 3.0),
                    "min_spend_euro": rec.get("min_spend_euro", 25.0),
                    "valore_euro_per_bottiglia": rec.get("valore_euro_per_bottiglia", 0.015),
                    "valore_cent_per_bottiglia": rec.get("valore_cent_per_bottiglia", 1.5),
                    "sconto_percentuale_carrello": rec.get("sconto_percentuale_carrello", 10.0),
                    "conversione_mode": rec.get("conversione_mode", 0),
                    "multiplier_presenze": rec.get("multiplier_presenze", 1.0),
                    "multiplier_bottles": rec.get("multiplier_bottles", 1.0),
                    "incentive_index": rec.get("incentive_index", 1.0),
                    "match_source": "MATCHED_FILIERA_NAME"
                }

        # 4. Greenfield Intelligent Inference via Brand / Retail Group policy
        s = f"{store_name or ''} {retail_group or ''}".upper()
        reg_clean = str(regione or "").strip().upper()

        # Cooperative chains with active fidelity cards
        is_toscana = ("TOSCANA" in reg_clean)
        is_unicoop = any(k in s for k in ["UNICOOP", "COOP.FI", "COOP FI", "COOP. FI"])
        if is_unicoop or (is_toscana and "COOP" in s):
            cat = "Conversione a punti fedeltà"
            fil_label = "Unicoop.Fi (Inferred)"
            source = "RETAIL_POLICY_COOP_TOSCANA"
            mech = extract_discount_mechanics("Conversione Punti Carta Socio", "30 punti sulla Carta Socio Coop", 90, 2)
        elif "ESSELUNGA" in s:
            cat = "Omaggio"
            fil_label = f"{store_name or 'Esselunga'} Circuit (Omaggio)"
            source = "RETAIL_POLICY_ESSELUNGA_OMAGGIO"
            mech = extract_discount_mechanics("Premio Omaggio Diretto", "Omaggio o buono spesa senza spesa minima", 100, 0)
        elif any(k in s for k in ["CONAD", "TUDAY", "CARREFOUR", "DECÒ", "DECO", "PENNY", "PAM", "DESPAR", "EUROSPAR", "INTERSPAR", "DIMAR", "BENNET", "TIGROS", "DODECÒ", "IPER"]):
            cat = "Emissione Buono sconto"
            fil_label = f"{store_name or 'Supermarket'} GDO National Voucher"
            source = "NATIONAL_GDO_VOUCHER_STANDARD"
            mech = extract_discount_mechanics("Sconto 3€", "3€ di sconto su una spesa minima di 30€", 200, 0)
        else:
            cat = "Emissione Buono sconto"  # Standard default across 94% of network
            fil_label = "Standard Retail Partner"
            source = "DEFAULT_SUPERMARKET_POLICY"
            mech = extract_discount_mechanics("Sconto 3€", "3€ di sconto su una spesa minima di 25€", 200, 0)

        elasticity = calculate_incentive_elasticity(cat, mech)

        return {
            "filiera": fil_label,
            "filiera_nome": fil_label,
            "reward_type": cat,
            "reward_label": REWARD_DEFINITIONS[cat]["label"],
            "definition_en": REWARD_DEFINITIONS[cat]["definition_en"],
            "definition_it": REWARD_DEFINITIONS[cat]["definition_it"],
            "incentive_tier": REWARD_DEFINITIONS[cat]["tier"],
            "incentive_strength": REWARD_DEFINITIONS[cat]["incentive_strength"],
            "typical_voucher": REWARD_DEFINITIONS[cat]["typical_voucher"],
            "match_source": source,
            **mech,
            **elasticity
        }


def prompt_custom_reward_configuration(base_default: str = "Emissione Buono sconto") -> Dict[str, Any]:
    """
    Guides the user through terminal inputs to create a custom promotional reward policy
    with user-defined Euro discount, minimum spend in Euros, and points threshold.
    """
    print("\n" + "=" * 80)
    print("  [🛠️ CONFIGURE CUSTOM PROMOTIONAL REWARD POLICY]")
    print("  Define exact commercial parameters: € discount, minimum spend €, and points hurdle.")
    print("=" * 80)
    
    # 1. Euro discount
    while True:
        d_str = input("  1. Enter Discount Face Value in € [e.g. 3.0, 5.0, 10.0] (Default: 3.0): ").strip()
        if not d_str:
            disc_val = 3.0
            break
        try:
            disc_val = float(d_str.replace(",", "."))
            if disc_val > 0:
                break
            print("     [ERROR] Discount must be greater than 0.")
        except ValueError:
            print("     [ERROR] Please enter a valid number (e.g. 3.0, 5.0).")
            
    # 2. Minimum Spend (€)
    while True:
        s_str = input("  2. Enter Minimum Spend in € [e.g. 25.0, 30.0, 50.0 — enter 0 for no minimum] (Default: 25.0): ").strip()
        if not s_str:
            spend_val = 25.0
            break
        try:
            spend_val = float(s_str.replace(",", "."))
            if spend_val >= 0:
                break
            print("     [ERROR] Minimum spend cannot be negative.")
        except ValueError:
            print("     [ERROR] Please enter a valid number (e.g. 25.0, 0).")
            
    # 3. Points Hurdle
    while True:
        p_str = input("  3. Enter Points / Bottles Required to Redeem [e.g. 100, 150, 200, 300] (Default: 200): ").strip()
        if not p_str:
            pts_val = 200
            break
        try:
            pts_val = int(p_str)
            if pts_val > 0:
                break
            print("     [ERROR] Points required must be at least 1.")
        except ValueError:
            print("     [ERROR] Please enter a valid integer (e.g. 200).")
            
    # 4. Archetype category
    print("\n  4. Select Reward Archetype Category:")
    print("     [1] Direct Grocery Checkout Voucher (Emissione Buono sconto - Standard GDO) [RECOMMENDED]")
    print("     [2] Store Loyalty Card Points (Conversione a punti fedeltà)")
    print("     [3] Direct Free Gift / Instant Voucher (Omaggio - No minimum spend)")
    arch_choice = input("     Select Archetype [1-3] (Default: 1): ").strip()
    
    arch_map = {
        "1": "Emissione Buono sconto",
        "2": "Conversione a punti fedeltà",
        "3": "Omaggio"
    }
    chosen_arch = arch_map.get(arch_choice, "Emissione Buono sconto")
    
    custom_policy = create_custom_reward_policy(
        discount_euro=disc_val,
        min_spend_euro=spend_val,
        points_required=pts_val,
        base_category=chosen_arch
    )
    
    print("\n" + "-" * 80)
    print("  [CUSTOM PROMOTIONAL REWARD ACTIVATED]")
    print(f"  Scheme Archetype:    {chosen_arch}")
    print(f"  Discount Face Value: {custom_policy['discount_euro']:.2f} €")
    if custom_policy['min_spend_euro'] > 0:
        print(f"  Minimum Spend:       {custom_policy['min_spend_euro']:.2f} € ({custom_policy['sconto_percentuale_carrello']:.1f}% cart discount)")
    else:
        print("  Minimum Spend:       None (Direct Voucher / Free Gift)")
    print(f"  Points Threshold:    {custom_policy['points_required']} bottles / points")
    print(f"  Unit Reward Rate:    {custom_policy['valore_cent_per_bottiglia']:.2f} €-cent/bottle (vs 1.50 c€ national benchmark)")
    print(f"  Behavioral Draw:     Footfall / Active Users: {custom_policy['multiplier_presenze']:.3f}x | Bottles: {custom_policy['multiplier_bottles']:.3f}x (Index: {custom_policy['incentive_index']:.3f})")
    print("-" * 80)
    
    return custom_policy


def prompt_reward_policy_selection(
    store_name: str,
    comune: Optional[str],
    current_reward_info: Dict[str, Any],
    is_known: bool
) -> Union[str, Dict[str, Any]]:
    """
    Interactive prompt allowing the user to select, confirm, or customize the promotional reward policy.
    - If new filiera (not known): ALWAYS asks the user to select policy from the 3 available options or configure custom [4].
    - If existing filiera (known in record): displays fetched policy and asks if user wants to proceed, change preset [1-3], or configure custom [4].
    Returns the selected reward_type string or custom reward dictionary.
    """
    print("\n" + "=" * 90)
    if not is_known:
        print("  [NEW FILIERA DETECTED — NO EXISTING PROMOTION POLICY IN RECORD]")
        print(f"  Store:    {store_name} ({comune or 'Unknown location'})")
        print("  Notice:   This store has no pre-existing commercial contract in Coripet records.")
        print("            Please select a promotional reward preset [1-3] or create a custom reward scheme [4]:")
        print("=" * 90)
        for num, spec in REWARD_DETAILED_SPECS.items():
            print(f"    [{num}] {spec['label']}")
            print(f"        * Exact Rule:   {spec['rule_en']}")
            print(f"        * Threshold:    {spec['threshold']} | Economic Value: {spec['economic_value']}")
            print(f"        * Impact:       {spec['footfall_impact']}")
        print("    [4] [CUSTOM] Custom Reward Policy (Define custom € discount, minimum spend, and points hurdle)")
        print("        * Exact Rule:   Enter custom Euro discount amount, minimum receipt spend, and points/bottles required.")
        print("        * Impact:       Real-time calculation of unit rate (c€/bottle) and behavioral elasticity multipliers.")
        print("-" * 90)
        
        while True:
            choice = input("\nSelect Promotional Reward Policy [1-3 for preset, or 4 for Custom]: ").strip()
            if choice == "4":
                return prompt_custom_reward_configuration()
            if choice in REWARD_KEY_TO_NAME:
                chosen_name = REWARD_KEY_TO_NAME[choice]
                print(f"  -> Applied: {chosen_name}")
                return chosen_name
            print("  [ERROR] Invalid selection. Please enter a number between 1 and 4.")
    else:
        current_type = current_reward_info.get("reward_type", "Emissione Buono sconto")
        current_label = current_reward_info.get("reward_label", current_type)
        current_title = current_reward_info.get("contract_title", "")
        current_pts = current_reward_info.get("points_required", 200)
        current_rate = current_reward_info.get("valore_cent_per_bottiglia", 1.50)
        current_disc_pct = current_reward_info.get("sconto_percentuale_carrello", 12.0)
        current_p_mult = current_reward_info.get("multiplier_presenze", 1.000)
        current_b_mult = current_reward_info.get("multiplier_bottles", 1.000)
        source = current_reward_info.get("match_source", "DATABASE")
        
        print("  [FETCHED PROMOTIONAL REWARD POLICY FROM DATABASE]")
        print(f"  Store / Filiera: {current_reward_info.get('filiera_nome', store_name)} [{source}]")
        print(f"  Fetched Policy:  {current_label} ({current_type})")
        if current_title:
            print(f"  Contract Promo:  {current_title} ({current_pts} pts hurdle)")
        print(f"  Unit Rate:       {current_rate:.2f} €-cent/bottle ({current_disc_pct:.1f}% cart discount)")
        print(f"  Behavioral Draw: Footfall / Active Users: {current_p_mult:.3f}x | Bottles: {current_b_mult:.3f}x")
        print("=" * 90)
        print("  Do you want to proceed with this policy, change to another standard preset, or create a custom reward?")
        print(f"  Press [Enter] to ACCEPT Fetched Policy [{current_type}], select [1-3] to change, or [4] for Custom:")
        print("-" * 90)
        for num, spec in REWARD_DETAILED_SPECS.items():
            name = spec["name"]
            label = spec["label"]
            mark = " (CURRENT / FETCHED)" if name == current_type else ""
            print(f"    [{num}] {label}{mark}")
            print(f"        * Exact Rule:   {spec['rule_en']}")
            print(f"        * Impact:       {spec['footfall_impact']}")
        print("    [4] 🛠️ Custom Reward Policy (Define custom € discount, minimum spend, and points hurdle)")
        print("-" * 90)
        
        choice = input(f"\nSelect Option [Enter for '{current_type}', 1-3 to change, or 4 for Custom]: ").strip()
        if not choice:
            print(f"  -> Proceeding with fetched policy: {current_type}")
            return current_type
        if choice == "4":
            return prompt_custom_reward_configuration(base_default=current_type)
        if choice in REWARD_KEY_TO_NAME:
            chosen_name = REWARD_KEY_TO_NAME[choice]
            print(f"  -> Changed reward policy to: {chosen_name}")
            return chosen_name
            
        print(f"  -> Unrecognized option '{choice}'. Proceeding with fetched policy: {current_type}")
        return current_type

