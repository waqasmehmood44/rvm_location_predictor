"""Business recommendation engine for Coripet RVM location viability."""

from typing import Dict, Optional
try:
    import yaml
except ImportError:
    yaml = None


class PotentialRecommender:
    """Classifies prospective locations into business volume potential tiers.

    Tiers:
    - SUPER PRENDENTE: > 2,000 bottles/day (Strategic high-yield anchor)
    - ALTO PRENDENTE: 1,000 - 2,000 bottles/day (Top performing asset)
    - MEDIO PRENDENTE: 300 - 1,000 bottles/day (Standard network benchmark)
    - BASSO PRENDENTE: < 300 bottles/day (Modest yield; capital review required)
    """

    def __init__(self, config_path: Optional[str] = "configs/business_rules.yaml"):
        self.tiers = {
            "super_prendente": {
                "min_daily_bottles": 2000.0,
                "label": "SUPER PRENDENTE",
                "description": "Hub strategico di eccellenza (> 2.000 b/giorno) ad altissima rotazione.",
            },
            "alto_prendente": {
                "min_daily_bottles": 1000.0,
                "max_daily_bottles": 2000.0,
                "label": "ALTO PRENDENTE",
                "description": "Top performer commerciale (1.000 - 2.000 b/giorno) ad alto rendimento.",
            },
            "medio_prendente": {
                "min_daily_bottles": 300.0,
                "max_daily_bottles": 1000.0,
                "label": "MEDIO PRENDENTE",
                "description": "Benchmark di rete standard sostenibile (300 - 1.000 b/giorno).",
            },
            "basso_prendente": {
                "min_daily_bottles": 0.0,
                "max_daily_bottles": 300.0,
                "label": "BASSO PRENDENTE",
                "description": "Volume contenuto (< 300 b/giorno); richiede presidio o monitoraggio costi.",
            },
        }
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Loads tier thresholds from configuration."""
        if yaml is None:
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg and "recommendations" in cfg and "tiers" in cfg["recommendations"]:
                    self.tiers.update(cfg["recommendations"]["tiers"])
        except Exception:
            pass

    def recommend(self, daily_bottles: float, suitability_score_pct: Optional[float] = None) -> Dict[str, str]:
        """Maps daily volume to official Coripet performance classification tier."""
        vol = float(daily_bottles)

        if vol >= 2000.0:
            selected = self.tiers["super_prendente"]
        elif vol >= 1000.0:
            selected = self.tiers["alto_prendente"]
        elif vol >= 300.0:
            selected = self.tiers["medio_prendente"]
        else:
            selected = self.tiers["basso_prendente"]

        return {
            "recommendation": selected["label"],
            "rationale": selected["description"],
            "tier_label": selected["label"]
        }
