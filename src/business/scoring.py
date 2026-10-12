"""Location suitability scoring and operational metric derivation."""

from typing import Dict, Optional
import numpy as np
import pandas as pd
try:
    import yaml
except ImportError:
    yaml = None


class SuitabilityScorer:
    """Calculates location suitability percentage (0 - 100%) and operational ratios.

    Uses configurable benchmarks calibrated from empirical dataset distributions.
    """

    def __init__(
        self,
        config_path: Optional[str] = "configs/business_rules.yaml",
        days_per_month: int = 30,
    ):
        self.days_per_month = days_per_month
        self.weights = {
            "bottles_daily_weight": 0.45,
            "presenze_daily_weight": 0.35,
            "bottles_per_person_weight": 0.20,
        }
        self.benchmarks = {
            "bottles_daily_target_p90": 1000.0,
            "presenze_daily_target_p90": 35.0,
            "bottles_per_person_target": 30.0,
        }

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Loads weights and benchmarks from YAML."""
        if yaml is None:
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if cfg:
                    self.days_per_month = cfg.get("operational_benchmarks", {}).get(
                        "days_per_month", self.days_per_month
                    )
                    scoring = cfg.get("scoring", {})
                    if "weights" in scoring:
                        self.weights.update(scoring["weights"])
                    if "benchmarks" in scoring:
                        self.benchmarks.update(scoring["benchmarks"])
        except Exception:
            # Fallback to default benchmarks if file is unreadable
            pass

    def calibrate_benchmarks(self, df: pd.DataFrame):
        """Empirically calibrates target benchmarks from training dataset quantiles.

        Guarantees that thresholds reflect real operational data rather than arbitrary guesses.
        """
        if "BOTTIGLIE" in df.columns and "PRESENZE" in df.columns:
            self.benchmarks["bottles_daily_target_p90"] = float(df["BOTTIGLIE"].quantile(0.90))
            self.benchmarks["presenze_daily_target_p90"] = float(df["PRESENZE"].quantile(0.90))

            valid = df[df["PRESENZE"] > 0]
            if not valid.empty:
                bpp = valid["BOTTIGLIE"] / valid["PRESENZE"]
                self.benchmarks["bottles_per_person_target"] = float(bpp.median())

    def calculate_metrics(
        self,
        pred_presenze: float,
        pred_bottiglie: float,
    ) -> Dict[str, float]:
        """Derives monthly projections, bottles-per-person ratio, and suitability percentage."""
        # Operational limits guard
        presenze_day = max(0.0, float(pred_presenze))
        bottiglie_day = max(0.0, float(pred_bottiglie))

        # Expected monthly volume
        bottiglie_month = bottiglie_day * self.days_per_month

        # Bottles per user ratio
        bottles_per_person = (
            bottiglie_day / max(presenze_day, 1.0) if presenze_day > 0 else 0.0
        )

        # Normalized component scores [0.0, 1.0] capped at 1.0
        b_score = min(1.0, bottiglie_day / max(self.benchmarks["bottles_daily_target_p90"], 1.0))
        p_score = min(1.0, presenze_day / max(self.benchmarks["presenze_daily_target_p90"], 1.0))
        r_score = min(
            1.0, bottles_per_person / max(self.benchmarks["bottles_per_person_target"], 1.0)
        )

        # Composite score
        w = self.weights
        composite = (
            (w["bottles_daily_weight"] * b_score)
            + (w["presenze_daily_weight"] * p_score)
            + (w["bottles_per_person_weight"] * r_score)
        )

        suitability_pct = float(np.clip(composite * 100.0, 0.0, 100.0))

        return {
            "expected_presenze_day": round(presenze_day, 1),
            "expected_bottiglie_day": round(bottiglie_day, 1),
            "expected_bottiglie_month": round(bottiglie_month, 0),
            "bottles_per_person": round(bottles_per_person, 1),
            "suitability_score_pct": round(suitability_pct, 1),
        }
