"""Business recommendation engine for Coripet RVM location viability."""

from typing import Dict, Optional
try:
    import yaml
except ImportError:
    yaml = None


class PotentialRecommender:
    """Classifies prospective locations into business potential tiers.

    Tiers:
    - HIGH POTENTIAL: Strong footfall and bottle throughput justifying priority deployment.
    - MEDIUM POTENTIAL: Viable location with standard commercial expectations.
    - LOW POTENTIAL: Low throughput risk; requires renegotiation or alternative site.
    """

    def __init__(self, config_path: Optional[str] = "configs/business_rules.yaml"):
        self.tiers = {
            "high_potential": {
                "min_score": 75.0,
                "label": "HIGH POTENTIAL",
                "description": "High volume location with strong footfall and efficient collection.",
            },
            "medium_potential": {
                "min_score": 50.0,
                "label": "MEDIUM POTENTIAL",
                "description": "Moderate operational volume; acceptable installation viability.",
            },
            "low_potential": {
                "min_score": 0.0,
                "label": "LOW POTENTIAL",
                "description": "Below operational viability threshold; requires scrutiny or alternative placement.",
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

    def recommend(self, suitability_score_pct: float) -> Dict[str, str]:
        """Maps suitability percentage to recommendation tier."""
        score = float(suitability_score_pct)

        if score >= self.tiers["high_potential"]["min_score"]:
            selected = self.tiers["high_potential"]
        elif score >= self.tiers["medium_potential"]["min_score"]:
            selected = self.tiers["medium_potential"]
        else:
            selected = self.tiers["low_potential"]

        return {
            "recommendation": selected["label"],
            "rationale": selected["description"],
        }
