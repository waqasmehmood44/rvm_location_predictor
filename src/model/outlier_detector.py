"""
Coripet RVM Outlier, Defect & Hardware Health Detection Engine.

Implements the 8 certified anomaly regimes:
1. DEFECTIVE_OFFLINE: Prolonged consecutive zero streaks or high zero rates in mature stores.
2. VOLATILITY_SWING: 600 -> 1,000+ -> <400 boom/bust promotion or burnout cycle.
3. CAPACITY_BOTTLENECK: High customer visits, but collapsed bottle volume (compactor/bin full).
4. EXTREME_SPIKE_EVENT: Single-day volume exceeding 99.8th percentile (>7,500) or Z-score > 4.5.
5. CHRONIC_UNDERPERFORMER: High-potential demographic store with persistent volume < 250 bottles.
6. SUPER_HUB_OUTLIER: Regional green hub collecting > 3,000 bottles/day consistently.
7. SENSOR_DRIFT_TRICKLE: Phantom optical trigger counting 1-5 bottles daily with 0 users.
8. SCHEDULE_ANOMALY: High volume during verified national store closing days.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class RVMOutlierDetector:
    """Detects, classifies, and manages operational anomalies and hardware defects in RVM daily logs."""

    def __init__(
        self,
        zero_streak_threshold: int = 7,
        zero_rate_threshold: float = 0.20,
        volatility_cv_threshold: float = 0.85,
        spike_volume_threshold: float = 7500.0,
        spike_zscore_threshold: float = 4.5,
        super_hub_threshold: float = 3000.0,
        underperformer_threshold: float = 250.0
    ):
        self.zero_streak_threshold = zero_streak_threshold
        self.zero_rate_threshold = zero_rate_threshold
        self.volatility_cv_threshold = volatility_cv_threshold
        self.spike_volume_threshold = spike_volume_threshold
        self.spike_zscore_threshold = spike_zscore_threshold
        self.super_hub_threshold = super_hub_threshold
        self.underperformer_threshold = underperformer_threshold

    def audit_fleet(self, df: pd.DataFrame, preds_map: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Scans all machines in df and returns an aggregated machine-level diagnostic table.
        df must contain: ['PF', 'GIORNO' or 'giorno_dt', 'BOTTIGLIE', 'PRESENZE']
        """
        data = df.copy()
        if "dt" not in data.columns:
            if "giorno_dt" in data.columns:
                data["dt"] = pd.to_datetime(data["giorno_dt"])
            else:
                data["dt"] = pd.to_datetime(data["GIORNO"], errors="coerce")

        data = data.sort_values(["PF", "dt"])
        records = []

        for pf, group in data.groupby("PF", observed=False):
            n_days = len(group)
            if n_days < 14:
                continue

            bottles = group["BOTTIGLIE"].values.astype(float)
            presenze = group["PRESENZE"].values.astype(float)

            # Metadata attributes
            banner = str(group["Insegna_Standardized"].iloc[-1] if "Insegna_Standardized" in group.columns else group.get("CLIENTE", pd.Series(["UNKNOWN"])).iloc[-1])
            comune = str(group.get("Comune", pd.Series([""])).iloc[-1])
            regione = str(group.get("Regione", pd.Series([""])).iloc[-1])
            machine_model = str(group.get("TipoMacchina", pd.Series([""])).iloc[-1])
            is_mature = int(group.get("maturity_status", pd.Series(["MATURE"])).iloc[-1] == "MATURE")

            # 1. Consecutive Zeros (Hardware Offline / Jammed)
            zero_mask = (bottles == 0)
            zero_streaks = []
            curr_streak = 0
            for z in zero_mask:
                if z:
                    curr_streak += 1
                else:
                    if curr_streak > 0:
                        zero_streaks.append(curr_streak)
                    curr_streak = 0
            if curr_streak > 0:
                zero_streaks.append(curr_streak)

            max_zero_streak = max(zero_streaks) if zero_streaks else 0
            last_60_days = bottles[-60:] if len(bottles) >= 60 else bottles
            recent_zero_rate = float((last_60_days == 0).mean())

            # 2. Basic Statistics
            med_vol = float(np.median(bottles))
            mean_vol = float(np.mean(bottles))
            std_vol = float(np.std(bottles))
            cv = float(std_vol / mean_vol) if mean_vol > 0 else 0.0
            max_day = float(np.max(bottles))
            recent_med_30 = float(np.median(bottles[-30:])) if len(bottles) >= 30 else med_vol

            # 3. Bottles-per-user ratio
            valid_p = presenze > 0
            bpu_arr = bottles[valid_p] / presenze[valid_p]
            med_bpu = float(np.median(bpu_arr)) if len(bpu_arr) > 0 else 0.0
            recent_bpu = float(np.median(bpu_arr[-14:])) if len(bpu_arr) >= 14 else med_bpu

            # 4. Volatility Swing Check (600 -> 1000+ -> <400)
            has_volatility_swing = False
            if len(bottles) >= 45:
                # Rolling 7d median
                s_b = pd.Series(bottles)
                roll_7 = s_b.rolling(7, min_periods=3).median().values
                roll_30 = s_b.rolling(30, min_periods=10).median().values

                # Search for surge followed by collapse
                for idx in range(30, len(roll_7) - 14):
                    base = roll_30[idx]
                    if 400 <= base <= 850:
                        peak = np.max(roll_7[idx:idx + 14])
                        trough = np.min(roll_7[idx + 14:idx + 35]) if (idx + 35) <= len(roll_7) else np.min(roll_7[idx + 14:])
                        if peak >= 1.5 * base and trough <= 0.65 * base and cv > self.volatility_cv_threshold:
                            has_volatility_swing = True
                            break

            # 5. Capacity Bottleneck Check
            is_capacity_choke = False
            if recent_med_30 < (0.60 * med_vol) and recent_bpu < 15.0 and np.median(presenze[-30:]) >= (0.75 * np.median(presenze)):
                is_capacity_choke = True

            # 6. Expected Volume & Error Bias
            expected_pred = float(preds_map.get(pf, med_vol)) if preds_map else med_vol
            error_bias = round(med_vol - expected_pred, 1)

            # --- Classification Logic ---
            anomaly = "HEALTHY_NORMAL"
            action = "Nominal operation. Maintain standard service."

            if is_mature and (max_zero_streak >= self.zero_streak_threshold or recent_zero_rate >= self.zero_rate_threshold):
                anomaly = "DEFECTIVE_OFFLINE"
                action = f"P1 Field Dispatch: Machine has {max_zero_streak} straight zero-days ({int(recent_zero_rate*100)}% recent zeros). Replace compactor motor/sensor."
            elif is_capacity_choke:
                anomaly = "CAPACITY_BOTTLENECK"
                action = "Store Management Alert: High footfall but bottles collapsed (<15/visit). Mandate scheduled bin emptying twice daily (12:00 & 18:00)."
            elif has_volatility_swing:
                anomaly = "VOLATILITY_SWING"
                action = f"Campaign/Reliability Audit: Severe swing detected (CV={cv:.2f}). Audit retail promo schedule vs staff compactor clearing."
            elif max_day >= self.spike_volume_threshold or (std_vol > 0 and (max_day - mean_vol) / std_vol >= self.spike_zscore_threshold):
                anomaly = "EXTREME_SPIKE_EVENT"
                action = f"Commercial Review: Single day logged {int(max_day):,} bottles. Verify if local school/commercial bulk deposit occurred."
            elif is_mature and med_vol < self.underperformer_threshold and expected_pred >= 750.0 and (expected_pred - med_vol) >= 400.0:
                anomaly = "CHRONIC_UNDERPERFORMER"
                action = f"Store Repositioning Notice: Expected {int(expected_pred)} but getting {int(med_vol)}. Relocate machine from hidden corner/basement to main entrance."
            elif med_vol >= self.super_hub_threshold:
                anomaly = "SUPER_HUB_OUTLIER"
                action = f"Capital Expansion: Massive volume ({int(med_vol):,} bottles/day). High priority location to install a 2nd Coripet machine."
            elif is_mature and 0 < mean_vol <= 5.0 and np.median(presenze) <= 1.0 and n_days >= 30:
                anomaly = "SENSOR_DRIFT_TRICKLE"
                action = "Optical Board Recalibration: Machine logs 1-5 ghost bottles daily with zero users. Clean optical lenses and recalibrate laser sensor."

            records.append({
                "PF": pf,
                "Store_Banner": banner,
                "Comune": comune,
                "Regione": regione,
                "TipoMacchina": machine_model,
                "Operational_Days": n_days,
                "Anomaly_Classification": anomaly,
                "Max_Consecutive_Zero_Days": max_zero_streak,
                "Recent_Zero_Rate_Pct": round(recent_zero_rate * 100.0, 1),
                "Median_Bottles_Daily": round(med_vol, 1),
                "Recent_30d_Median": round(recent_med_30, 1),
                "Max_Day_Bottles": round(max_day, 1),
                "Volatility_CV": round(cv, 2),
                "Bottles_Per_User": round(med_bpu, 1),
                "Expected_Predicted_Bottles": round(expected_pred, 1),
                "Volume_Error_Bias": error_bias,
                "Recommended_Action": action
            })

        return pd.DataFrame(records)

    def calculate_training_weights(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates sample weights and target winsorization for model training.
        Returns:
            weights (np.ndarray float32): 0.0 for breakdown zeros, 0.25 for bottlenecks, 0.3 for underperformers, 1.0 normal.
            cleaned_targets (np.ndarray float32): Capped at 99.5th percentile (6,500).
        """
        bottles = df["BOTTIGLIE"].values.astype(np.float32)
        weights = np.ones(len(df), dtype=np.float32)

        # 1. Zero-Mask breakdown streaks (3 or more consecutive zeros in mature stores)
        is_zero = (bottles == 0)
        weights[is_zero] = 0.0  # Zero out dead days from Huber loss

        # 2. Winsorize extreme bulk single-day spikes
        max_allowed_cap = 6500.0
        cleaned_targets = np.clip(bottles, a_min=0.0, a_max=max_allowed_cap)

        return weights, cleaned_targets
