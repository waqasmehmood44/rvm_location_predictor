"""Prediction confidence assessment based on data familiarity and epistemic uncertainty."""

from typing import Dict, List, Optional
try:
    import yaml
except ImportError:
    yaml = None


class ConfidenceAssessor:
    """Evaluates prediction reliability and uncertainty for prospective locations.

    Assigns HIGH, MEDIUM, or LOW confidence based on:
    - Postal code familiarity in historical training records.
    - Retail brand representation in training data.
    - Machine type familiarity.
    - Model predictive variance (from Monte Carlo Dropout if available).
    """

    def __init__(self, config_path: Optional[str] = "configs/business_rules.yaml"):
        self.factors = {
            "postal_code_exact_match_weight": 0.40,
            "retail_chain_known_weight": 0.35,
            "geo_proximity_weight": 0.25,
        }
        self.thresholds = {
            "high_confidence_min_score": 0.80,
            "medium_confidence_min_score": 0.55,
        }
        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Loads confidence weights and thresholds from YAML."""
        if yaml is None:
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg and "confidence" in cfg:
                    conf = cfg["confidence"]
                    if "factors" in conf:
                        self.factors.update(conf["factors"])
                    if "thresholds" in conf:
                        self.thresholds.update(conf["thresholds"])
        except Exception:
            pass

    def assess_confidence(
        self,
        is_postal_code_known: bool,
        is_retail_chain_known: bool,
        is_machine_type_known: bool,
        relative_variance: Optional[float] = None,
    ) -> Dict[str, any]:
        """Calculates composite confidence score and categorical label (HIGH, MEDIUM, LOW)."""
        score = 0.0

        # Knowledge domain coverage weights
        if is_postal_code_known:
            score += self.factors.get("postal_code_exact_match_weight", 0.40)
        else:
            score += self.factors.get("postal_code_exact_match_weight", 0.40) * 0.4  # partial if regional fallback

        if is_retail_chain_known:
            score += self.factors.get("retail_chain_known_weight", 0.35)
        else:
            score += self.factors.get("retail_chain_known_weight", 0.35) * 0.2

        if is_machine_type_known:
            score += self.factors.get("geo_proximity_weight", 0.25)
        else:
            score += self.factors.get("geo_proximity_weight", 0.25) * 0.3

        # Penalize if epistemic uncertainty (MC Dropout relative variance) is high
        if relative_variance is not None and relative_variance > 0.4:
            score *= 0.8

        score = min(1.0, max(0.0, score))

        if score >= self.thresholds.get("high_confidence_min_score", 0.80):
            label = "HIGH"
        elif score >= self.thresholds.get("medium_confidence_min_score", 0.55):
            label = "MEDIUM"
        else:
            label = "LOW"

        return {
            "confidence": label,
            "confidence_score": round(score, 2),
            "factors": {
                "known_postal_code": is_postal_code_known,
                "known_retail_chain": is_retail_chain_known,
                "known_machine_type": is_machine_type_known,
            },
        }
