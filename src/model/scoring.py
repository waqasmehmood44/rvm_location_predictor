"""
Location Potential Scoring, Strategic Recommendation Tiers, and Operational Risk Bounds.
Calibrated on the verified empirical distribution of all 1,657 operating Coripet RVM locations in Italy.
"""
import math
import numpy as np
from typing import Dict, Any, Optional

# Empirical quantile knots of daily bottles across 1,657 operating Coripet PFs
# Source: modeling_daily_rvm.parquet (certified population)
BOTTLES_QUANTILES = [
    (0.0, 0.0),
    (100.0, 5.0),
    (167.7, 10.0),
    (220.0, 18.0),
    (270.0, 25.0),
    (350.0, 37.0),
    (447.6, 50.0),
    (580.0, 62.0),
    (751.1, 75.0),
    (950.0, 83.0),
    (1196.4, 90.0),
    (1601.8, 95.0),
    (2500.0, 99.0),
    (5850.0, 100.0)
]

def calculate_location_potential_score(daily_bottles: float) -> int:
    """
    Computes empirical percentile rank (0-100) of prospective store
    relative to all 1,657 operational Coripet recycling points in Italy.
    """
    if daily_bottles <= 0:
        return 1
    
    b_vals = [pt[0] for pt in BOTTLES_QUANTILES]
    p_vals = [pt[1] for pt in BOTTLES_QUANTILES]
    
    # Linear interpolation between quantile knots
    score = float(np.interp(daily_bottles, b_vals, p_vals))
    return int(round(np.clip(score, 1, 99)))

def determine_recommendation_tier(
    potential_score: int, 
    daily_bottles: float, 
    cannibalization_alert: bool = False,
    density_tier: str = "MODERATE_COMMERCIAL_CORRIDOR",
    store_scale_applied: str = "",
    suitability_status: str = "SUITABLE"
) -> Dict[str, Any]:
    """
    Classifies prospective store into strategic deployment tiers with machine selection and capex advice.
    """
    if suitability_status == "NOT_SUITABLE":
        return {
            "tier_name": "TIER 5 - NOT SUITABLE / UNVIABLE",
            "recommendation_code": "REJECT",
            "decision": "NOT RECOMMENDED (DO NOT INSTALL)",
            "recommended_machine": "None (Installation Not Recommended)",
            "logistics_frequency": "N/A (Installation Rejected)",
            "rationale": "Location fails critical operational/economic suitability gates. High risk of capital loss, negative cash flow, or local network cannibalization."
        }

    if cannibalization_alert:
        return {
            "tier_name": "TIER 4 - CANNIBALIZATION RISK",
            "recommendation_code": "CAUTION_RISK",
            "decision": "UNDERWRITING REVIEW (HIGH OVERLAP)",
            "recommended_machine": "Compact Single-Stream (Ripet_V2 / DAINT_B) or Defer Installation",
            "logistics_frequency": "Standard On-Demand",
            "rationale": "An operational Coripet machine exists < 500m away. High probability of volume dilution and customer sharing."
        }
        
    if potential_score >= 75 or daily_bottles >= 750.0:
        return {
            "tier_name": "TIER 1 - STRATEGIC HIGH-VOLUME ANCHOR",
            "recommendation_code": "STRONG_BUY",
            "decision": "HIGH-PRIORITY EXPANSION (PROCEED)",
            "recommended_machine": "Dual-Stream Heavy Duty (DAINT_G / DAINT_A-BIS / RIPET_C-FRONT)",
            "logistics_frequency": "Twice-Weekly Scheduled Servicing",
            "rationale": f"Ranks in the top {100 - potential_score}% of Coripet locations nationwide. High foot-traffic hub with exceptional recycling yield potential."
        }
        
    if potential_score >= 50 or daily_bottles >= 448.0:
        return {
            "tier_name": "TIER 2 - VIABLE STANDARD EXPANSION",
            "recommendation_code": "APPROVED",
            "decision": "COMMERCIALLY VIABLE (PROCEED)",
            "recommended_machine": "Standard Multi-Material RVM (DAINT_B-BIS / RIPET_C-FRONT)",
            "logistics_frequency": "Weekly Scheduled Servicing",
            "rationale": f"Ranks above the national median (Score: {potential_score}/100). Healthy payback profile with reliable collection volume."
        }
        
    if potential_score >= 25 or daily_bottles >= 270.0:
        return {
            "tier_name": "TIER 3 - NICHE / NEIGHBORHOOD MARKET",
            "recommendation_code": "CONDITIONAL",
            "decision": "SELECTIVE PLACEMENT (LOW CAPEX)",
            "recommended_machine": "Compact Single-Stream RVM (Ripet_V2 / DAINT_B)",
            "logistics_frequency": "Bi-Weekly Servicing",
            "rationale": f"Moderate localized foot-traffic (Score: {potential_score}/100). Viable with compact footprint and controlled machine depreciation."
        }
        
    return {
        "tier_name": "TIER 4 - MARGINAL / LOW POTENTIAL",
        "recommendation_code": "MARGINAL",
        "decision": "UNDERWRITING REVIEW (LOW VOLUME)",
        "recommended_machine": "Entry-Level Compact Unit / Mobile Pilot",
        "logistics_frequency": "Bi-Weekly / Call-in Basis",
        "rationale": f"Ranks in the bottom quartile nationwide (Score: {potential_score}/100). Recommend retailer co-financing or promotional foot-traffic guarantees."
    }

def calculate_operational_bounds(
    daily_bottles: float, 
    daily_presenze: float,
    dow_dict: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Computes 3-scenario operational bounds (Conservative Floor, Steady-State Expected, Optimistic Peak).
    """
    # Conservative Floor (P25 operational conditions, ~82% of steady state)
    floor_daily_b = round(daily_bottles * 0.82, 1)
    floor_daily_p = round(daily_presenze * 0.82, 1)
    floor_monthly_b = int(round(floor_daily_b * 30.4))
    
    # Expected Steady-State (P50 baseline)
    expected_daily_b = round(daily_bottles, 1)
    expected_daily_p = round(daily_presenze, 1)
    expected_monthly_b = int(round(expected_daily_b * 30.4))
    
    # Optimistic Peak (P75/P90 capacity conditions, ~125% of steady state)
    peak_daily_b = round(daily_bottles * 1.25, 1)
    peak_daily_p = round(daily_presenze * 1.25, 1)
    peak_monthly_b = int(round(peak_daily_b * 30.4))
    
    return {
        "conservative_floor": {
            "daily_bottiglie": floor_daily_b,
            "daily_presenze": floor_daily_p,
            "monthly_bottiglie": floor_monthly_b,
            "description": "Off-peak seasonal / conservative underwriting floor"
        },
        "steady_state_expected": {
            "daily_bottiglie": expected_daily_b,
            "daily_presenze": expected_daily_p,
            "monthly_bottiglie": expected_monthly_b,
            "description": "Annual steady-state baseline expectation"
        },
        "optimistic_peak": {
            "daily_bottiglie": peak_daily_b,
            "daily_presenze": peak_daily_p,
            "monthly_bottiglie": peak_monthly_b,
            "description": "Peak summer / high-engagement campaign capacity"
        }
    }

def _normal_cdf(z: float) -> float:
    """Computes standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def calculate_probability_factors(
    daily_bottles: float,
    confidence_score: Optional[int] = 100,
    breakeven_target: float = 120.0,
    standard_target: float = 300.0,
    high_yield_target: float = 500.0,
    anchor_target: float = 1000.0,
    requested_machine: Optional[str] = None
) -> Dict[str, Any]:
    """
    Computes rigorous statistical probabilities of achieving key commercial volume targets
    and evaluates comprehensive future performance for the machine (performance tiers, 
    cumulative annual milestones, and hardware capacity utilization / overflow risk).
    """
    req_m = (requested_machine or "").strip().upper()
    if "A-BIS" in req_m:
        hw_class = "Compact Single-Stream (Daint A-bis)"
        saturation_thresh = 300.0
        under_thresh = 80.0
    elif "B-BIS" in req_m:
        hw_class = "Compact Dual-Stream (Daint B-Bis)"
        saturation_thresh = 500.0
        under_thresh = 150.0
    elif "DAINT_G" in req_m or ("DAINT" in req_m and "G" in req_m):
        hw_class = "Heavy-Duty Dual-Stream (Daint G)"
        saturation_thresh = 4500.0
        under_thresh = 500.0
    else:
        hw_class = "Heavy-Duty Industrial Compactor (Ripet C-Front)"
        saturation_thresh = 4500.0
        under_thresh = 500.0

    if daily_bottles <= 0:
        return {
            "success_probability": 0.0,
            "risk_rating": "CRITICAL RISK (UNVIABLE)",
            "p_breakeven_120": 0.0,
            "p_standard_300": 0.0,
            "p_high_yield_500": 0.0,
            "p_anchor_1000": 0.0,
            "downside_floor_p10_daily": 0.0,
            "downside_floor_p10_monthly": 0,
            "upside_peak_p90_daily": 0.0,
            "upside_peak_p90_monthly": 0,
            "sigma_applied": 0.45,
            "performance_tiers": {
                "p_basso_under_300": 100.0,
                "p_medio_300_1000": 0.0,
                "p_alto_1000_2000": 0.0,
                "p_super_over_2000": 0.0,
                "p_low_under_200": 100.0,
                "p_moderate_200_400": 0.0,
                "p_high_400_700": 0.0,
                "p_star_over_700": 0.0,
                "most_likely_tier": "BASSO PRENDENTE (< 300 B/D)"
            },
            "annual_milestones": {
                "p_annual_50k": 0.0,
                "p_annual_100k": 0.0,
                "p_annual_150k": 0.0,
                "p_annual_200k": 0.0
            },
            "hardware_operational_fit": {
                "requested_machine": requested_machine or "Standard RVM",
                "hardware_class": hw_class,
                "daily_saturation_threshold": saturation_thresh,
                "p_daily_overflow_risk": 0.0,
                "p_optimal_utilization": 0.0,
                "p_underutilization_risk": 100.0,
                "recommended_service_rhythm": "Call-in / On-Demand Servicing (Low Volume)"
            }
        }

    conf = confidence_score if confidence_score is not None else 100
    if conf >= 85:
        sigma = 0.35
    elif conf >= 70:
        sigma = 0.39
    else:
        sigma = 0.45

    mu = math.log1p(max(0.0, float(daily_bottles)))

    def _get_prob(target: float) -> float:
        z = (math.log1p(target) - mu) / sigma
        return round(max(0.0, min(100.0, (1.0 - _normal_cdf(z)) * 100.0)), 1)

    p_120 = _get_prob(breakeven_target)
    p_300 = _get_prob(standard_target)
    p_500 = _get_prob(high_yield_target)
    p_1000 = _get_prob(anchor_target)

    # Risk Rating Assessment
    if p_120 >= 95.0:
        rating = "VERY LOW RISK (VIRTUALLY GUARANTEED)"
    elif p_120 >= 80.0:
        rating = "LOW RISK (HIGHLY LIKELY)"
    elif p_120 >= 60.0:
        rating = "MODERATE RISK (VIABLE)"
    else:
        rating = "HIGH RISK (SPECULATIVE / CAPITAL AT RISK)"

    # Value-at-Risk Quantiles (P10 90% certainty downside floor, P90 top 10% upside peak)
    p10_daily = round(max(0.0, math.expm1(mu - 1.28155 * sigma)), 1)
    p90_daily = round(max(0.0, math.expm1(mu + 1.28155 * sigma)), 1)

    # 1. Performance Tiers (Mutually exclusive intervals covering full Coripet volume spectrum)
    # Official Coripet Network Classification:
    # - Basso Prendente : < 300 b/d
    # - Medio Prendente : 300 - 1,000 b/d
    # - Alto Prendente  : 1,000 - 2,000 b/d
    # - Super Prendente : > 2,000 b/d
    cdf_300 = _normal_cdf((math.log1p(300.0) - mu) / sigma)
    cdf_1000 = _normal_cdf((math.log1p(1000.0) - mu) / sigma)
    cdf_2000 = _normal_cdf((math.log1p(2000.0) - mu) / sigma)

    p_basso = round(max(0.0, min(100.0, cdf_300 * 100.0)), 1)
    p_medio = round(max(0.0, min(100.0, (cdf_1000 - cdf_300) * 100.0)), 1)
    p_alto = round(max(0.0, min(100.0, (cdf_2000 - cdf_1000) * 100.0)), 1)
    p_super = round(max(0.0, min(100.0, (1.0 - cdf_2000) * 100.0)), 1)

    # Legacy tier fields preserved for backwards compatibility
    cdf_200 = _normal_cdf((math.log1p(200.0) - mu) / sigma)
    cdf_400 = _normal_cdf((math.log1p(400.0) - mu) / sigma)
    cdf_700 = _normal_cdf((math.log1p(700.0) - mu) / sigma)

    p_low = round(max(0.0, min(100.0, cdf_200 * 100.0)), 1)
    p_moderate = round(max(0.0, min(100.0, (cdf_400 - cdf_200) * 100.0)), 1)
    p_high = round(max(0.0, min(100.0, (cdf_700 - cdf_400) * 100.0)), 1)
    p_star = round(max(0.0, min(100.0, (1.0 - cdf_700) * 100.0)), 1)

    coripet_tier_map = {
        "BASSO PRENDENTE (< 300 B/D)": p_basso,
        "MEDIO PRENDENTE (300 - 1,000 B/D)": p_medio,
        "ALTO PRENDENTE (1,000 - 2,000 B/D)": p_alto,
        "SUPER PRENDENTE (> 2,000 B/D)": p_super
    }
    most_likely_tier = max(coripet_tier_map.items(), key=lambda x: x[1])[0]

    # 2. Cumulative Annual Collection Milestones (Operating 365 days)
    p_ann_50k = _get_prob(50000.0 / 365.0)
    p_ann_100k = _get_prob(100000.0 / 365.0)
    p_ann_150k = _get_prob(150000.0 / 365.0)
    p_ann_200k = _get_prob(200000.0 / 365.0)

    # 3. Hardware Operational Fit & Saturation Risk
    p_overflow = _get_prob(saturation_thresh)
    p_under = round(max(0.0, min(100.0, _normal_cdf((math.log1p(under_thresh) - mu) / sigma) * 100.0)), 1)
    p_optimal = round(max(0.0, min(100.0, 100.0 - p_overflow - p_under)), 1)

    if daily_bottles >= 650.0:
        service_rhythm = "2-3x per week (High Fill Rate)"
    elif daily_bottles >= 300.0:
        service_rhythm = "Weekly Scheduled Servicing (Optimal)"
    elif daily_bottles >= 150.0:
        service_rhythm = "Bi-Weekly Servicing (Standard Cadence)"
    else:
        service_rhythm = "Call-in / On-Demand Servicing (Low Volume)"

    return {
        # Preserved previous baseline fields
        "success_probability": p_120,
        "risk_rating": rating,
        "p_breakeven_120": p_120,
        "p_standard_300": p_300,
        "p_high_yield_500": p_500,
        "p_anchor_1000": p_1000,
        "downside_floor_p10_daily": p10_daily,
        "downside_floor_p10_monthly": int(round(p10_daily * 30.4)),
        "upside_peak_p90_daily": p90_daily,
        "upside_peak_p90_monthly": int(round(p90_daily * 30.4)),
        "sigma_applied": sigma,
        
        # Comprehensive Future Performance Dimensions
        "performance_tiers": {
            "p_basso_under_300": p_basso,
            "p_medio_300_1000": p_medio,
            "p_alto_1000_2000": p_alto,
            "p_super_over_2000": p_super,
            "p_low_under_200": p_low,
            "p_moderate_200_400": p_moderate,
            "p_high_400_700": p_high,
            "p_star_over_700": p_star,
            "most_likely_tier": most_likely_tier
        },
        "annual_milestones": {
            "p_annual_50k": p_ann_50k,
            "p_annual_100k": p_ann_100k,
            "p_annual_150k": p_ann_150k,
            "p_annual_200k": p_ann_200k
        },
        "hardware_operational_fit": {
            "requested_machine": requested_machine or "Standard RVM",
            "hardware_class": hw_class,
            "daily_saturation_threshold": saturation_thresh,
            "p_daily_overflow_risk": p_overflow,
            "p_optimal_utilization": p_optimal,
            "p_underutilization_risk": p_under,
            "recommended_service_rhythm": service_rhythm
        }
    }

def evaluate_prediction_confidence(
    geocoding_quality: str,
    brand_embedding_active: bool,
    alliance_embedding_active: bool,
    store_scale_applied: str,
    nearest_coripet_km: float
) -> Dict[str, Any]:
    """
    Evaluates prediction confidence score (0-100) and confidence tier.
    """
    score = 0
    factors = []
    
    # 1. Geocoding quality
    if "EXACT" in geocoding_quality or "STREET" in geocoding_quality:
        score += 35
        factors.append("Precise street/municipal address geocoded")
    elif "COMUNE" in geocoding_quality:
        score += 25
        factors.append("Municipal demographic centroid matched")
    else:
        score += 15
        factors.append("Postal code centroid fallback")
        
    # 2. Retail brand & alliance
    if brand_embedding_active:
        score += 25
        factors.append("Verified brand banner embedding active")
    elif alliance_embedding_active:
        score += 20
        factors.append("Commercial retail alliance embedding active")
    else:
        score += 10
        factors.append("Generic independent store profile")
        
    # 3. Store format & scale proxy
    if "CUSTOM" in store_scale_applied or "PRESET" in store_scale_applied:
        score += 25
        factors.append("Format scale and foot-traffic proxy conditioned")
    else:
        score += 15
        factors.append("National median store scale default")
        
    # 4. Spatial network proximity
    if nearest_coripet_km < 15.0:
        score += 15
        factors.append(f"Nearby Coripet network benchmark ({nearest_coripet_km:.1f} km)")
    else:
        score += 10
        factors.append("Regional benchmark territory")
        
    rating = "HIGH" if score >= 80 else ("MEDIUM" if score >= 60 else "LOW")
    
    return {
        "confidence_score": score,
        "confidence_rating": rating,
        "key_confidence_drivers": factors
    }

def calculate_catchment_realization_factor(
    unconstrained_daily_bottles: float,
    unconstrained_daily_presenze: float,
    nearest_coripet_km: float,
    nearest_pf_name: str,
    store_name: str,
    coripet_count_1km: int,
    coripet_count_3km: int,
    coripet_count_5km: int,
    grocery_total_stores: int,
    network_position: str,
    sales_area_sqm: float = 750.0,
    numero_casse: float = 4.0,
    is_independent_banner: bool = False,
    nielsen_area: str = "Area 4.0",
    is_proximity_market: bool = False,
    population_density: float = 700.0,
    population: float = 0.0,
    regione: str = "",
    machine_type: str = "",
    reward_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Translates the unconstrained standalone theoretical potential into a Realized Operational Forecast
    by applying empirical catchment dilution, network cluster saturation, rural bulk habits, and sole grocery monopoly gravity.
    """
    reasons = []
    
    # 1. Evaluate Direct Cannibalization / Co-Location (Immediate Proximity < 1.2 km)
    s1 = str(store_name).strip().upper()
    s2 = str(nearest_pf_name).strip().upper()
    is_self_match = (nearest_coripet_km <= 0.1) and (
        s1 in s2 or s2 in s1 or 
        ("DECO" in s1 and "DECO" in s2) or 
        ("MERKANTE" in s1 and ("MERKANTE" in s2 or "DECO" in s2)) or
        ("BRICO" in s1 and "BRICO" in s2) or
        ("CONAD" in s1 and "CONAD" in s2) or
        ("EUROESSE" in s1 and "EUROESSE" in s2)
    )
    
    cannibal_penalty = 0.0
    if network_position == "CANNIBALIZATION_ALERT" and not is_self_match:
        if nearest_coripet_km <= 0.1:
            cannibal_penalty = 0.45
            reasons.append("Direct Co-location / 0m Overlap with existing RVM (-45% customer sharing)")
        elif nearest_coripet_km <= 0.5:
            cannibal_penalty = 0.30
            reasons.append(f"Nearby operational RVM ({nearest_coripet_km*1000:.0f}m away) sharing local trade zone (-30%)")
        elif nearest_coripet_km <= 1.0:
            cannibal_penalty = 0.15
            reasons.append(f"Competing RVM within walking radius ({nearest_coripet_km*1000:.0f}m away, -15%)")
    elif is_self_match and coripet_count_1km >= 2:
        cannibal_penalty = 0.35
        reasons.append(f"High Immediate RVM Density ({coripet_count_1km} machines within 1 km, -35% volume dilution)")

    # 2. Regional Network Cluster Background Saturation (gentle macro dilution)
    cluster_penalty = 0.0
    if coripet_count_3km >= 8:
        cluster_penalty = 0.12
        reasons.append(f"Regional Urban Cluster Diffusion ({coripet_count_3km} RVMs within 3 km, -12% customer dispersion)")
    elif coripet_count_3km >= 5:
        cluster_penalty = 0.08
        reasons.append(f"Active Multi-RVM Urban Network ({coripet_count_3km} RVMs within 3 km, -8%)")
    elif coripet_count_3km >= 3:
        cluster_penalty = 0.04
        reasons.append(f"Moderate Network Presence ({coripet_count_3km} RVMs in 3 km, -4%)")

    # 3. Commercial Food Competitor Dispersion (Trade Area Grocery Competition)
    comp_penalty = 0.0
    if grocery_total_stores >= 20:
        if population_density >= 2000.0:
            # High-density metropolitan pedestrian corridor: high grocery count reflects dense pedestrian traffic, not dilution
            comp_penalty = 0.05
            reasons.append(f"Metropolitan High-Footfall Corridor ({grocery_total_stores} competitors in dense pedestrian zone, mild -5% dispersion)")
        elif population_density >= 600.0:
            comp_penalty = 0.15
            reasons.append(f"Congested Retail Corridor ({grocery_total_stores} competitors in commercial trade zone, -15% customer dispersion)")
        else:
            comp_penalty = 0.05
            reasons.append(f"Competitive Food Retail Corridor ({grocery_total_stores} competitors, -5%)")

    # Check if store operates as a sole municipal food anchor or rural destination (car-based bulk deposits)
    is_rural_or_monopoly = (grocery_total_stores <= 1) or (population_density < 150.0 and grocery_total_stores <= 2)
    is_rural_valley_hub = (population_density < 180.0 and nearest_coripet_km >= 3.5)

    # 4. Urban Proximity Format vs Regional Anchor
    format_penalty = 0.0
    is_urban_proximity_chain = any(k in s1 for k in ["TUDAY", "CONAD CITY", "TODIS"]) and population_density >= 500.0
    if any(h in s1 for h in ["BRICO", "LEROY", "OBI", "BRICOMAN", "TECNOMAT", "MONDOBRICO"]) or sales_area_sqm >= 2500.0 or is_urban_proximity_chain:
        format_penalty = 0.0
    elif is_rural_or_monopoly or is_rural_valley_hub:
        # Rural or regional valley hub supermarkets do not suffer walk-in pedestrian dilution; customers drive and bring bulk crates
        format_penalty = 0.0
    elif sales_area_sqm <= 400.0 or numero_casse <= 2.0:
        format_penalty = 0.20
        reasons.append("Compact Neighborhood Market footprint (walk-in traffic with smaller deposit batches, -20%)")
    elif sales_area_sqm <= 600.0 and numero_casse <= 3.0:
        format_penalty = 0.15
        reasons.append("Medium Proximity Store format calibration (-15%)")
    elif is_independent_banner and "Area 4" in nielsen_area and sales_area_sqm <= 800.0:
        format_penalty = 0.08
        reasons.append("Independent banner calibration against hypermarket priors (-8%)")

    # 5. Sole Municipal Grocery Monopoly Uplift (High Trade Area Concentration)
    monopoly_boost = 0.0
    if grocery_total_stores <= 1:
        if population >= 3500.0 and coripet_count_1km == 0:
            # Captive Municipal Anchor: Store is the exclusive supermarket for an entire town (e.g. Bolano, 7,451 inh on 1 store)
            # Shoppers have zero alternative food stores in the municipality, creating exceptional captive footfall gravity
            captive_scale = min(1.30, max(0.45, (population / 5700.0) * 1.0))
            monopoly_boost = round(captive_scale, 2)
            reasons.append(f"Captive Municipal Grocery Anchor (+{round(monopoly_boost*100):.0f}% captive municipal gravity, exclusive food anchor for {population:.0f} residents)")
        elif population_density < 150.0:
            monopoly_boost = 0.45
            reasons.append("Sole Municipal Grocery Monopoly in rural trade area (+45% regional catchment gravity)")
        elif population_density < 400.0:
            monopoly_boost = 0.20
            reasons.append("Sole Grocery Anchor in municipality (+20% trade area concentration)")

    # 6. Rural Bulk Deposit Multiplier (Area 4 / Southern Rural Bulk Habit)
    rural_bulk_boost = 0.0
    if "Area 4" in nielsen_area and population_density < 150.0:
        rural_bulk_boost = 0.20
        reasons.append("Rural Bulk Deposit Pattern (car-based bulk batch drops in Southern catchment, +20%)")

    # 7. Supermarket Loyalty Points Incentive Multiplier (Fidelity Card / Carta Socio)
    # Dynamically scales with physical store scale & checkout throughput to prevent overpredicting on mid-size formats
    unicoop_fi_boost = 0.0
    reward_type = reward_info.get("reward_type", "") if reward_info else ""
    reg_clean = str(regione).strip().upper()
    is_toscana = ("TOSCANA" in reg_clean) or ("Area 3" in nielsen_area and any(w in s1 for w in ["FIRENZE", "CASCINA", "PRATO", "EMPOLI", "AREZZO", "PISA", "TOSCANA"]))
    if reward_info is not None and "reward_type" in reward_info:
        is_loyalty = (reward_type == "Conversione a punti fedeltà")
    else:
        is_loyalty = (reward_type == "Conversione a punti fedeltà") or (is_toscana and any(k in s1 for k in ["UNICOOP", "COOP.FI", "COOP FI", "COOP. FI", "UNICOOP.FI"]))
    if is_loyalty:
        scale_factor = min(1.0, max(0.35, max(sales_area_sqm / 3500.0, numero_casse / 20.0)))
        unicoop_fi_boost = round(0.30 * scale_factor, 3)
        reasons.append(f"Supermarket Loyalty Point Incentive (+{round(unicoop_fi_boost*100, 1)}% deposit basket intensity via Fidelity Card / Carta Socio)")

    # 8. DIY / Home Improvement Big-Box Basket Multiplier (Car-Boot Bulk Drops)
    diy_bulk_boost = 0.0
    is_diy = any(h in s1 for h in ["BRICO", "LEROY", "OBI", "BRICOMAN", "TECNOMAT", "MONDOBRICO"])
    if is_diy:
        diy_bulk_boost = 0.25
        reasons.append("DIY / Retail Park Bulk Deposit Pattern (+25% car-boot multi-week bottle dropoff)")

    # 9. Isolated Rural Valley Monopoly Hub (e.g. Bisignano, unserved valley catchment draw)
    rural_valley_boost = 0.0
    if is_rural_valley_hub and grocery_total_stores > 1:
        if "Area 4" in nielsen_area:
            # Southern valley catchment (Area 4: Calabria, Campania, Sicilia, Puglia, Basilicata)
            # High per-capita bulk crate drops in unserved valley
            rural_valley_boost = 1.15
            reasons.append(f"Southern Rural Valley Hub Monopoly (+115% regional draw across unserved valley hamlets, nearest RVM {nearest_coripet_km:.1f}km)")
        elif "Area 3" in nielsen_area:
            # Central Italy rural valley hub (Area 3: Umbria, Marche, Lazio, Toscana)
            # Uplift is balanced against competing supermarkets (e.g. Gala Superstore, Conad in Gualdo Tadino)
            rural_valley_boost = 0.78
            reasons.append(f"Central Rural Valley Trade Hub (+78% regional draw across valley hamlets, nearest RVM {nearest_coripet_km:.1f}km)")

    # 10. High-Throughput Urban Proximity Chain Hub (e.g. TuDay Conad / Conad City in dense municipalities)
    # Dense pedestrian trade zones (>= 500 hab/km2) where high-frequency chain groceries with heavy-duty
    # hardware (Ripet_C-front) operate as primary neighborhood recycling destinations.
    urban_proximity_boost = 0.0
    is_urban_proximity_chain = any(k in s1 for k in ["TUDAY", "CONAD CITY", "TODIS"]) and population_density >= 500.0
    if is_urban_proximity_chain:
        format_penalty = 0.0  # Exempt from small-floorplan walk-in compression
        urban_proximity_boost = 0.35
        reasons.append("High-Throughput Urban Proximity Chain Hub (+35% recurring pedestrian frequency, exempt from format compression)")

    # Total Additive Dilution & Net Realization Factor (Bounded between 0.35 and 1.65)
    total_dilution = min(0.65, cannibal_penalty + cluster_penalty + comp_penalty + format_penalty)
    total_boost = monopoly_boost + rural_bulk_boost + unicoop_fi_boost + diy_bulk_boost + rural_valley_boost + urban_proximity_boost
    k_net = max(0.35, min(1.65, round(1.0 - total_dilution + total_boost, 3)))

    realized_b = round(unconstrained_daily_bottles * k_net, 1)
    
    # Mechanical Hardware Intake Saturation Ceiling
    # Single-head compact units (Ripet_V2, DAINT_B, CESTONE, RIPET1000) have mechanical intake limits
    # during peak queuing hours. Dual-head / heavy-duty units sustain higher sustained volumes.
    req_m_upper = str(machine_type).strip().upper() if machine_type else ""
    is_compact_single = any(c in req_m_upper for c in ["RIPET_V2", "DAINT_B", "CESTONE", "RIPET1000", "DAINT_A"])
    if is_compact_single and realized_b > 650.0:
        realized_b = round(650.0 + (realized_b - 650.0) * 0.25, 1)
        reasons.append("Hardware Mechanical Saturation: Compact single-feed unit intake limit applied")
    elif realized_b > 4800.0:
        realized_b = round(4800.0 + (realized_b - 4800.0) * 0.40, 1)
        reasons.append("High-Throughput Operational Saturation: Compaction queue & bin-full downtime compression applied above 4,800 b/d")
    
    realized_mo_b = int(round(realized_b * 30.4))
    realized_yr_b = int(round(realized_b * 365.0))
    
    # Presenze realization (footfall scales with local competition, cluster density, and catchment gravity)
    k_pres_cap = 2.50 if monopoly_boost >= 0.80 else 1.50
    k_pres_base = float(k_net ** 0.88) if monopoly_boost >= 0.80 else float(np.sqrt(k_net))
    k_pres = max(0.38, min(k_pres_cap, round(k_pres_base * (1.0 - cluster_penalty) * (1.0 - comp_penalty), 3)))
    realized_p = round(unconstrained_daily_presenze * k_pres, 1)
    
    # Level 3.5: Apply granular reward multipliers for non-loyalty policies (loyalty is handled via unicoop_fi_boost)
    if not is_loyalty and reward_info:
        reward_mult_b = float(reward_info.get("multiplier_bottles", 1.0))
        reward_mult_p = float(reward_info.get("multiplier_presenze", 1.0))
        if reward_mult_b != 1.0:
            realized_b = round(realized_b * reward_mult_b, 1)
            realized_mo_b = int(round(realized_b * 30.4))
            realized_yr_b = int(round(realized_b * 365.0))
        if reward_mult_p != 1.0:
            realized_p = round(realized_p * reward_mult_p, 1)
        
        rew_type = reward_info.get("reward_type", "")
        if rew_type == "Omaggio":
            reasons.append("Direct Free Gift / No Min. Spend Incentive (+8% volume, +15% footfall draw)")

    # Realized operational bounds (Conservative Floor P25, Steady-State P50, Optimistic Peak P75)
    floor_daily_b = round(realized_b * 0.82, 1)
    floor_mo_b = int(round(floor_daily_b * 30.4))
    peak_daily_b = round(realized_b * 1.25, 1)
    peak_mo_b = int(round(peak_daily_b * 30.4))

    dilution_pct = round((1.0 - k_net) * 100.0, 1) if k_net < 1.0 else 0.0
    uplift_pct = round((k_net - 1.0) * 100.0, 1) if k_net > 1.0 else 0.0

    return {
        "realization_factor": round(k_net, 3),
        "dilution_percentage": dilution_pct,
        "uplift_percentage": uplift_pct,
        "realized_daily_bottles": realized_b,
        "realized_monthly_bottles": realized_mo_b,
        "realized_annual_bottles": realized_yr_b,
        "realized_daily_presenze": realized_p,
        "realized_bounds": {
            "floor_daily_b": floor_daily_b,
            "floor_monthly_b": floor_mo_b,
            "base_daily_b": realized_b,
            "base_monthly_b": realized_mo_b,
            "peak_daily_b": peak_daily_b,
            "peak_monthly_b": peak_mo_b
        },
        "operational_baseline": {
            "serviced_daily_bottles": floor_daily_b,
            "serviced_monthly_bottles": floor_mo_b,
            "serviced_annual_bottles": int(round(floor_daily_b * 365.0)),
            "uptime_availability_factor": 0.82,
            "description": "Real-world serviced operational baseline (accounting for routine servicing, full-bin pauses, and holiday schedules)"
        },
        "dilution_reasons": reasons
    }


def evaluate_location_suitability(
    realized_daily_bottles: float,
    unconstrained_daily_bottles: float,
    nearest_coripet_km: float,
    nearest_pf_name: str,
    dilution_percentage: float = 0.0,
    network_position: str = "EXPANSION_NODE",
    sales_area_sqm: float = 750.0,
    numero_casse: float = 4.0,
    population: float = 50000.0,
    population_density: float = 1000.0,
    total_grocery_stores: int = 5,
    conservative_floor_bottles: Optional[float] = None,
    breakeven_floor: float = 120.0,
    critical_cannibalization_km: float = 0.25,
    min_sales_area_sqm: float = 150.0,
    min_checkouts: float = 2.0,
    requested_machine: str = "",
    brand_name: str = "",
    retail_group: str = "",
    shopping_centers_count: int = 0,
    reward_policy: Optional[str] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Evaluates prospective location suitability across 6 objective operational, physical, and financial clearance gates:
    1. Cannibalization Gate (network proximity < 0.25 km and dilution >= 35%)
    2. Economic Breakeven Gate (projected realized volume vs machine breakeven floor 120.0 b/d)
    3. Format & Footprint Gate (sales area >= 150 m2 & checkouts >= 2.0 for physical queues & service)
    4. Demographic Catchment Gate (population density and trade corridor)
    5. Hardware Sizing Match Gate (right-sizes machine capex against volume tier)
    6. Logistics Feasibility Gate (flags isolated outposts > 25km requiring high routing costs)
    
    Returns structured clearance verdict (GO / CONDITIONAL / NO-GO), suitability status,
    suitability score (0-100), diagnostic gate breakdown, blockers, and strategic advice.
    """
    if conservative_floor_bottles is None:
        conservative_floor_bottles = round(realized_daily_bottles * 0.82, 1)

    gate_results = {}
    disqualifying_reasons = []
    warning_flags = []
    key_strengths = []

    score = 100

    # -------------------------------------------------------------
    # GATE 1: SPATIAL CANNIBALIZATION GATE
    # -------------------------------------------------------------
    is_cannibal_alert = (network_position == "CANNIBALIZATION_ALERT")
    is_critical_dist = (nearest_coripet_km < critical_cannibalization_km)
    is_critical_dilution = (dilution_percentage >= 35.0)

    if is_critical_dist or is_critical_dilution or (is_cannibal_alert and nearest_coripet_km < 0.4):
        dist_str = f"{nearest_coripet_km * 1000:.0f}m" if nearest_coripet_km < 1.0 else f"{nearest_coripet_km:.1f}km"
        dilution_str = f" (-{dilution_percentage:.1f}% dilution)" if dilution_percentage > 0.0 else " (100% volume collision)"
        gate_results["cannibalization_gate"] = {
            "status": "FAIL",
            "label": "Cannibalization Gate",
            "metric": f"Existing RVM '{nearest_pf_name}' is only {dist_str} away{dilution_str}",
            "flag": "CRITICAL CANNIBALIZATION"
        }
        if dilution_percentage > 0.0:
            disqualifying_reasons.append(
                f"Severe catchment overlap: Existing Coripet machine '{nearest_pf_name}' is {dist_str} away. "
                f"High volume dilution (-{dilution_percentage:.1f}%) results in duplicate capital without generating incremental network bottles."
            )
        else:
            disqualifying_reasons.append(
                f"Severe catchment overlap: Existing Coripet machine '{nearest_pf_name}' is already operational {dist_str} away. "
                f"Deploying duplicate hardware results in near-zero incremental network volume."
            )
        score -= 45
    elif nearest_coripet_km < 0.75 or dilution_percentage >= 15.0 or is_cannibal_alert:
        dist_str = f"{nearest_coripet_km * 1000:.0f}m" if nearest_coripet_km < 1.0 else f"{nearest_coripet_km:.1f}km"
        dilution_str = f" (-{dilution_percentage:.1f}% dilution)" if dilution_percentage > 0.0 else ""
        gate_results["cannibalization_gate"] = {
            "status": "WARN",
            "label": "Cannibalization Gate",
            "metric": f"Nearby RVM '{nearest_pf_name}' is {dist_str} away{dilution_str}",
            "flag": "MODERATE OVERLAP"
        }
        if dilution_percentage > 0.0:
            warning_flags.append(f"Nearby machine within {dist_str} exerts moderate catchment pull (-{dilution_percentage:.1f}% dilution).")
        else:
            warning_flags.append(f"Nearby machine within {dist_str} shares local trade area.")
        score -= 20
    else:
        is_self_bench = bool(kwargs.get("is_benchmark_self_match", False))
        neighbor_muni = kwargs.get("nearest_pf_comune", "")
        if is_self_bench:
            loc_str = f" in {neighbor_muni}" if neighbor_muni else ""
            metric_cannibal = f"Existing site verified (nearest neighbor RVM is {nearest_coripet_km:.1f} km away{loc_str})"
            flag_cannibal = "AUTONOMOUS SITE"
            strength_cannibal = f"Existing site benchmark: Zero cannibalization from neighboring machines ({nearest_coripet_km:.1f} km to nearest sister RVM)."
        else:
            metric_cannibal = f"Clear network buffer (nearest RVM is {nearest_coripet_km:.1f} km away)"
            flag_cannibal = "AUTONOMOUS CATCHMENT"
            strength_cannibal = f"Zero local cannibalization: Nearest machine is {nearest_coripet_km:.1f} km away."

        gate_results["cannibalization_gate"] = {
            "status": "PASS",
            "label": "Cannibalization Gate",
            "metric": metric_cannibal,
            "flag": flag_cannibal
        }
        key_strengths.append(strength_cannibal)

    # -------------------------------------------------------------
    # GATE 2: ECONOMIC BREAKEVEN GATE
    # -------------------------------------------------------------
    if realized_daily_bottles < (breakeven_floor * 0.90) or conservative_floor_bottles < (breakeven_floor * 0.75):
        gate_results["economic_volume_gate"] = {
            "status": "FAIL",
            "label": "Economic Breakeven Gate",
            "metric": f"Realized {realized_daily_bottles:.1f} b/day (Floor: {conservative_floor_bottles:.1f} b/d) below breakeven ({breakeven_floor:.0f} b/d)",
            "flag": "UNECONOMIC VOLUME DEFICIT"
        }
        disqualifying_reasons.append(
            f"Projected volume ({realized_daily_bottles:.1f} bottles/day, P25 floor: {conservative_floor_bottles:.1f} b/day) "
            f"is below the minimum economic operating breakeven threshold ({breakeven_floor:.0f} b/day). Operational logistics and depreciation will yield negative cash flow."
        )
        score -= 40
    elif realized_daily_bottles < (breakeven_floor * 1.25):
        gate_results["economic_volume_gate"] = {
            "status": "WARN",
            "label": "Economic Breakeven Gate",
            "metric": f"Marginal volume ({realized_daily_bottles:.1f} b/day) tight against breakeven ({breakeven_floor:.0f} b/d)",
            "flag": "TIGHT MARGIN"
        }
        warning_flags.append(f"Volume ({realized_daily_bottles:.1f} b/day) is near breakeven floor ({breakeven_floor:.0f} b/day); requires lean operational costs.")
        score -= 15
    else:
        gate_results["economic_volume_gate"] = {
            "status": "PASS",
            "label": "Economic Breakeven Gate",
            "metric": f"Robust collection yield ({realized_daily_bottles:.1f} b/day vs {breakeven_floor:.0f} b/d breakeven)",
            "flag": "COMMERCIALLY PROFITABLE"
        }
        key_strengths.append(f"Strong economic headroom ({realized_daily_bottles:.1f} b/day, {int(round(realized_daily_bottles * 30.4)):,} b/month).")
        if realized_daily_bottles >= 600.0:
            score += 5  # High-volume bonus

    # -------------------------------------------------------------
    # GATE 3: FORMAT & PHYSICAL FOOTPRINT GATE
    # -------------------------------------------------------------
    # Dynamic checkout calibration: A store with >= 250 m2 sales area has physical space for >= 2 checkouts,
    # preventing administrative single-register census entries from falsely disqualifying real stores.
    effective_casse = max(numero_casse, round(sales_area_sqm / 180.0)) if sales_area_sqm >= 250.0 else numero_casse

    if (sales_area_sqm > 0 and sales_area_sqm < min_sales_area_sqm) or (effective_casse > 0 and effective_casse < min_checkouts):
        gate_results["store_format_gate"] = {
            "status": "FAIL",
            "label": "Store Format Gate",
            "metric": f"Constrained format ({sales_area_sqm:.0f} m2, {effective_casse:.0f} checkouts) below minimum threshold ({min_sales_area_sqm:.0f} m2, {min_checkouts:.0f} checkouts)",
            "flag": "INSUFFICIENT PHYSICAL SCALE"
        }
        disqualifying_reasons.append(
            f"Store footprint ({sales_area_sqm:.0f} m2, {effective_casse:.0f} checkouts) is below operational minimum ({min_sales_area_sqm:.0f} m2, {min_checkouts:.0f} checkouts) to comfortably accommodate RVM footprint, bag servicing, and user queuing."
        )
        score -= 30
    elif (sales_area_sqm > 0 and sales_area_sqm < (min_sales_area_sqm * 2.0)) or (effective_casse > 0 and effective_casse <= min_checkouts):
        gate_results["store_format_gate"] = {
            "status": "WARN",
            "label": "Store Format Gate",
            "metric": f"Small convenience market ({sales_area_sqm:.0f} m2, {effective_casse:.0f} checkouts)",
            "flag": "CONSTRAINED FOOTPRINT"
        }
        warning_flags.append(f"Small retail format ({sales_area_sqm:.0f} m2) requires compact machine footprint.")
        score -= 10
    else:
        gate_results["store_format_gate"] = {
            "status": "PASS",
            "label": "Store Format Gate",
            "metric": f"Adequate commercial floorplan ({sales_area_sqm:.0f} m2, {effective_casse:.0f} checkouts)",
            "flag": "OPTIMAL SCALE"
        }
        key_strengths.append(f"Commercial store format ({sales_area_sqm:.0f} m2, {effective_casse:.0f} checkouts) supports steady foot traffic.")

    # -------------------------------------------------------------
    # GATE 4: DEMOGRAPHIC CATCHMENT GATE
    # -------------------------------------------------------------
    if population > 0 and population < 2000.0 and population_density < 60.0 and total_grocery_stores <= 1:
        gate_results["demographic_gate"] = {
            "status": "FAIL",
            "label": "Demographic Catchment Gate",
            "metric": f"Isolated catchment ({population:.0f} inh, {population_density:.1f} inh/km2, {total_grocery_stores} stores)",
            "flag": "DEMOGRAPHIC DEFICIT"
        }
        disqualifying_reasons.append(
            f"Municipality population ({population:.0f} inhabitants) and density ({population_density:.1f} inh/km2) are too low to sustain a regular habit of returns."
        )
        score -= 25
    elif population > 0 and (population < 5000.0 or population_density < 120.0):
        gate_results["demographic_gate"] = {
            "status": "WARN",
            "label": "Demographic Catchment Gate",
            "metric": f"Modest municipal catchment ({population:.0f} inh, {population_density:.1f} inh/km2)",
            "flag": "MODEST POPULATION"
        }
        warning_flags.append(f"Modest local population ({population:.0f} inh); dependent on passing commuter traffic.")
        score -= 10
    else:
        gate_results["demographic_gate"] = {
            "status": "PASS",
            "label": "Demographic Catchment Gate",
            "metric": f"Urban/suburban population hub ({population:.0f} inh, {population_density:.1f} inh/km2)",
            "flag": "DENSE DEMOGRAPHICS"
        }
        key_strengths.append(f"Strong demographic base ({population:.0f} inhabitants in municipality).")

    # -------------------------------------------------------------
    # GATE 5: HARDWARE SIZING MATCH GATE
    # -------------------------------------------------------------
    req_m = requested_machine.strip().upper()
    if "A-BIS" in req_m:
        sat_limit = 300.0
        display_m = "Daint A-bis"
    elif "B-BIS" in req_m:
        sat_limit = 500.0
        display_m = "Daint B-Bis"
    elif "DAINT_G" in req_m or ("DAINT" in req_m and "G" in req_m):
        sat_limit = 4500.0
        display_m = "Daint G"
    else:
        sat_limit = 4500.0
        display_m = "Ripet C-Front"

    if realized_daily_bottles > sat_limit:
        overload = realized_daily_bottles - sat_limit
        gate_results["hardware_sizing_gate"] = {
            "status": "WARN",
            "label": "Hardware Sizing Gate",
            "metric": f"{display_m} capacity ({sat_limit:.0f} b/d) exceeded by +{overload:.0f} b/d ({realized_daily_bottles:.1f} b/d)",
            "flag": "CAPACITY BOTTLENECK"
        }
        warning_flags.append(
            f"Requested machine '{display_m}' has a rated threshold of {sat_limit:.0f} bottles/day. Projected volume ({realized_daily_bottles:.1f} b/d) exceeds capacity by +{overload:.0f} b/d (+{(overload/sat_limit)*100:.0f}%), risking overflow and service downtime; upgrade to high-capacity unit (Daint G / Ripet C-Front 4,500 b/d)."
        )
        score -= 10
    elif realized_daily_bottles < 150.0 and sat_limit >= 4500.0:
        gate_results["hardware_sizing_gate"] = {
            "status": "WARN",
            "label": "Hardware Sizing Gate",
            "metric": f"{display_m} (4,500 b/d) has high spare headroom for low volume ({realized_daily_bottles:.1f} b/d)",
            "flag": "OVERSIZED HARDWARE"
        }
        warning_flags.append(
            f"Requested unit '{display_m}' carries 4,500 b/d industrial capacity for low volume ({realized_daily_bottles:.1f} b/d); consider Daint A-bis (300 b/d) or Daint B-Bis (500 b/d) to optimize footprint."
        )
        score -= 5
    else:
        matched_str = f"Hardware '{display_m}' capacity ({sat_limit:,.0f} b/d) optimally matches volume ({realized_daily_bottles:.1f} b/d)"
        gate_results["hardware_sizing_gate"] = {
            "status": "PASS",
            "label": "Hardware Sizing Gate",
            "metric": matched_str,
            "flag": "OPTIMAL HARDWARE FIT"
        }

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # COMMERCIAL ANCHOR SITE FOOT-TRAFFIC (Advisory Context - Non-blocking)
    # -------------------------------------------------------------
    b_clean = brand_name.strip().upper()
    g_clean = retail_group.strip().upper()
    if b_clean not in ["INDIPENDENTE", "UNKNOWN", ""] or g_clean not in ["UNKNOWN", "INDIPENDENTE", ""]:
        alliance_label = g_clean if g_clean not in ["UNKNOWN", ""] else b_clean
        key_strengths.append(f"Commercial retail location ({alliance_label}) provides steady recurring customer foot-traffic.")

    # -------------------------------------------------------------
    # GATE 7: LOGISTICS SERVICING FEASIBILITY GATE
    # -------------------------------------------------------------
    if nearest_coripet_km > 25.0:
        if realized_daily_bottles < 250.0:
            gate_results["logistics_feasibility_gate"] = {
                "status": "WARN",
                "label": "Logistics Feasibility Gate",
                "metric": f"Isolated outpost ({nearest_coripet_km:.1f} km from nearest Coripet cluster)",
                "flag": "LOGISTICS DEADHEAD"
            }
            warning_flags.append(
                f"Geographically isolated location ({nearest_coripet_km:.1f} km away from nearest Coripet route); requires bi-weekly or on-demand logistics dispatch to contain transport costs."
            )
            score -= 10
        else:
            gate_results["logistics_feasibility_gate"] = {
                "status": "PASS",
                "label": "Logistics Feasibility Gate",
                "metric": f"Isolated location ({nearest_coripet_km:.1f} km away), but strong volume ({realized_daily_bottles:.1f} b/d) justifies logistics routing",
                "flag": "HIGH-YIELD OUTPOST"
            }
    else:
        is_self_bench = bool(kwargs.get("is_benchmark_self_match", False))
        if is_self_bench:
            logistics_metric = f"Active installed route node ({nearest_coripet_km:.1f} km to nearest sister hub)"
        else:
            logistics_metric = f"Integrated into active regional servicing route ({nearest_coripet_km:.1f} km to nearest node)"
        gate_results["logistics_feasibility_gate"] = {
            "status": "PASS",
            "label": "Logistics Feasibility Gate",
            "metric": logistics_metric,
            "flag": "EFFICIENT ROUTING"
        }

    # -------------------------------------------------------------
    # COMMERCIAL MALL / SHOPPING CENTER ANCHOR SYNERGY
    # -------------------------------------------------------------
    if shopping_centers_count >= 1 and sales_area_sqm >= 1200.0:
        key_strengths.append("Prime Commercial Park / Mall Anchor: Benefits from shared customer parking and multi-purpose weekend family foot traffic.")
        score += 5

    # -------------------------------------------------------------
    # OVERALL CLEARANCE & VERDICT
    # -------------------------------------------------------------
    final_score = int(round(np.clip(score, 5, 99)))
    has_fail = any(g["status"] == "FAIL" for g in gate_results.values())
    has_warn = any(g["status"] == "WARN" for g in gate_results.values())

    if has_fail:
        suitability_status = "NOT_SUITABLE"
        verdict = "NO-GO [REJECT]"
        summary_title = "NOT SUITABLE (REJECT / DO NOT INSTALL)"
        if is_critical_dist or (is_cannibal_alert and nearest_coripet_km < 0.4):
            dist_str = f"{nearest_coripet_km * 1000:.0f}m" if nearest_coripet_km < 1.0 else f"{nearest_coripet_km:.1f}km"
            recommended_action = (
                f"DO NOT install new hardware at this location. Direct store shoppers to the existing active RVM "
                f"at '{nearest_pf_name}' ({dist_str} away) or negotiate a dual-compartment/multi-stream upgrade on the existing unit."
            )
        elif realized_daily_bottles < (breakeven_floor * 0.90):
            recommended_action = (
                f"DO NOT install standard hardware. Projected collection volume ({realized_daily_bottles:.1f} b/day) "
                f"fails to recover machine lease and monthly servicing costs."
            )
        else:
            recommended_action = (
                f"DO NOT proceed with automated RVM deployment. Site fails operational physical/demographic prerequisites."
            )
    elif has_warn or final_score < 55:
        suitability_status = "CONDITIONAL_PILOT"
        verdict = "CONDITIONAL [REVIEW REQUIRED]"
        summary_title = "CONDITIONAL PILOT (SELECTIVE UNDERWRITING)"
        advice_parts = []
        if gate_results.get("hardware_sizing_gate", {}).get("status") == "WARN":
            if realized_daily_bottles < 250.0:
                advice_parts.append("Downsize hardware to a compact unit (Ripet_V2 / DAINT_B) to right-size capital depreciation.")
            else:
                advice_parts.append("Upgrade hardware to a heavy-duty unit (DAINT_G / Ripet_C-front) to prevent bin overflow downtime.")
        if gate_results.get("logistics_feasibility_gate", {}).get("status") == "WARN":
            advice_parts.append("Adopt bi-weekly or on-demand logistics dispatch to contain transport costs.")
        if not advice_parts:
            advice_parts.append(
                "Proceed only under selective conditional terms: Deploy an entry-level compact unit (Ripet_V2) "
                "with retailer co-financing, or establish a 6-month probationary review with footfall commitments."
            )
        recommended_action = " ".join(advice_parts)
    elif final_score >= 75 and realized_daily_bottles >= 500.0:
        suitability_status = "HIGHLY_SUITABLE"
        verdict = "GO [PRIORITY APPROVED]"
        summary_title = "HIGHLY SUITABLE (APPROVED STRATEGIC ANCHOR)"
        recommended_action = (
            "Approved for immediate high-priority deployment. Excellent volume and zero network friction justify "
            "a heavy-duty multi-material RVM (DAINT_G / RIPET_C-FRONT) with twice-weekly servicing."
        )
    else:
        suitability_status = "SUITABLE"
        verdict = "GO [STANDARD APPROVED]"
        summary_title = "SUITABLE (APPROVED STANDARD EXPANSION)"
        recommended_action = (
            "Approved for standard network expansion. All operational and economic clearance gates passed. "
            "Install standard multi-material RVM (DAINT_B-BIS / RIPET_C-FRONT) with weekly servicing rhythm."
        )

    return {
        "suitability_score": final_score,
        "suitability_status": suitability_status,
        "verdict": verdict,
        "summary_title": summary_title,
        "is_suitable": (suitability_status != "NOT_SUITABLE"),
        "gate_results": gate_results,
        "disqualifying_reasons": disqualifying_reasons,
        "warning_flags": warning_flags,
        "key_strengths": key_strengths,
        "recommended_action": recommended_action
    }


def generate_predictive_diagnostics(
    realized_daily_bottles: float,
    unconstrained_daily_bottles: float,
    realized_daily_presenze: float,
    sales_area_sqm: float,
    numero_casse: float,
    population: float,
    population_density: float,
    total_grocery_stores: int,
    nearest_coripet_km: float,
    nearest_pf_name: str = "",
    coripet_count_3km: int = 0,
    coripet_count_5km: int = 0,
    brand_name: str = "",
    retail_group: str = "",
    regione: str = "",
    geocoding_quality: str = "",
    dilution_percentage: float = 0.0,
    dilution_reasons: Optional[list] = None,
    pil_per_capita: float = 0.0,
    requested_machine: str = ""
) -> Dict[str, Any]:
    """
    Generates actionable operational diagnostics explaining:
    1. Primary reasons behind the results (performance drivers).
    2. Detailed bottlenecks and limiting factors (why results are not higher / not good).
    3. Actionable strategic model suggestions to increase future collection volume at this site.
    4. Factors that may cause real-world telemetry variance.
    """
    primary_drivers = []
    limiting_factors = []
    actionable_suggestions = []
    accuracy_variance_factors = []

    b_clean = brand_name.strip().upper()
    grp_clean = retail_group.strip().upper()

    # ==================================================================
    # 1. PRIMARY PERFORMANCE DRIVERS (Why This Volume Was Predicted)
    # ==================================================================
    if sales_area_sqm >= 800.0 or numero_casse >= 5.0:
        primary_drivers.append(
            f"Substantial Commercial Scale: Store footprint ({sales_area_sqm:.0f} m2 sales area, {numero_casse:.0f} checkouts) generates strong continuous shopper throughput capable of sustaining above-average recycling volumes."
        )
    elif sales_area_sqm >= 450.0:
        primary_drivers.append(
            f"Standard Supermarket Scale: Balanced format ({sales_area_sqm:.0f} m2, {numero_casse:.0f} checkouts) provides healthy, recurring neighborhood foot-traffic."
        )

    if population >= 50000.0 or population_density >= 1000.0:
        primary_drivers.append(
            f"High Demographic Density: Dense municipal catchment ({population:,.0f} inhabitants, {population_density:.1f} inh/km2) ensures a large, concentrated pool of frequent household repeat depositors."
        )
    elif population >= 15000.0:
        primary_drivers.append(
            f"Sufficient Population Base: Municipal population ({population:,.0f} inhabitants) provides adequate local customer volume for sustainable operation."
        )

    if pil_per_capita >= 20000.0:
        primary_drivers.append(
            f"Affluent Economic Catchment: Municipal purchasing power (EUR {pil_per_capita:,.0f} PIL/capita) correlates with higher consumption of packaged and bottled beverages."
        )

    if nearest_coripet_km >= 3.0 and coripet_count_3km == 0:
        primary_drivers.append(
            f"Undiluted Local Network Buffer: Zero competing Coripet machines within 3 km ({nearest_coripet_km:.1f} km to nearest node) grants this machine an exclusive, insulated local collection zone."
        )

    if population_density < 150.0 and realized_daily_presenze <= 12.0 and realized_daily_bottles >= 300.0:
        primary_drivers.append(
            "Car-Based Bulk Batch Dynamics: Southern/suburban transit pattern where customers arrive by car and deposit large bulk batches (40-80+ bottles per trip) rather than small walk-in drops."
        )

    if not primary_drivers:
        primary_drivers.append(
            f"Baseline Catchment Foot-Traffic: Location exhibits steady neighborhood convenience customer visits supporting {realized_daily_bottles:.1f} b/day."
        )

    # ==================================================================
    # 2. LIMITING FACTORS & BOTTLENECK DIAGNOSIS (Why Results Are Not Higher)
    # ==================================================================
    if (sales_area_sqm > 0 and sales_area_sqm < 600.0) or (numero_casse > 0 and numero_casse <= 3.0):
        limiting_factors.append(
            f"Constrained Physical Floorplan: Convenience scale ({sales_area_sqm:.0f} m2, {numero_casse:.0f} checkouts) limits daily customer throughput compared to full-scale hypermarkets or large supermarkets."
        )

    if nearest_coripet_km <= 0.05:
        limiting_factors.append(
            f"Direct 0m Self-Cannibalization: An active Coripet machine ('{nearest_pf_name}') is already operating at this exact address. Adding a duplicate unit divides existing volume without capturing new network bottles."
        )
    elif nearest_coripet_km < 1.5 or dilution_percentage > 15.0:
        dist_str = f"{nearest_coripet_km * 1000:.0f}m" if nearest_coripet_km < 1.0 else f"{nearest_coripet_km:.1f}km"
        limiting_factors.append(
            f"Network Cannibalization & Catchment Overlap: Existing machine at '{nearest_pf_name}' ({dist_str} away, {coripet_count_3km} in 3km) pulls approximately {dilution_percentage:.1f}% of local recyclers away."
        )

    if total_grocery_stores >= 10:
        limiting_factors.append(
            f"Commercial Trade Area Fragmentation: {total_grocery_stores} food stores operating in the local trade area distribute customer foot-traffic across multiple shopping locations."
        )

    if population > 0 and population < 15000 and population_density < 250:
        limiting_factors.append(
            f"Sparse Municipal Population Ceiling: Modest permanent resident base ({population:,.0f} inhabitants, {population_density:.1f} inh/km2) caps the total volume of recyclable bottles generated within walking/short-drive distance."
        )

    if pil_per_capita > 0 and pil_per_capita < 16000.0:
        limiting_factors.append(
            f"Restricted Disposable Income (EUR {pil_per_capita:,.0f} PIL/capita): Below-average municipal income dampens discretionary beverage spending and consumption volume."
        )

    limiting_factors.append(
        "Off-Peak Seasonal Dampening: Non-summer months (especially Winter and late Autumn) naturally reflect ~15-20% lower national beverage consumption and fewer recycling trips."
    )

    # ==================================================================
    # 3. ACTIONABLE MODEL SUGGESTIONS (How to Increase Collection in Future)
    # ==================================================================
    # Suggestion 1: Parking lot / entrance placement
    actionable_suggestions.append(
        "External Parking Plaza Placement: Position the RVM in the external customer parking lot near the main access road rather than in a rear store aisle. This allows car-based commuters and bulk depositors to drop 50-100 bottle bags directly from their car trunk without entering the store aisles."
    )

    # Suggestion 2: Dedicated RVM Directional Signage
    actionable_suggestions.append(
        "Dedicated RVM Signage & External Directional Wayfinding: Install prominent directional signage at the store entrance and customer parking lot directing shoppers directly to the Coripet recycling point. Clear visibility increases spontaneous bottle deposits by +15% to +25%."
    )

    # Suggestion 3: Community & School Recycling Drives
    actionable_suggestions.append(
        "Institutional Community Partnerships: Connect with local schools, municipal sports centers, and environmental clubs. Designating this machine as the official collection node for school recycling competitions drives consistent, high-volume youth and family participation."
    )

    # Suggestion 4: Operational servicing & bag-emptying uptime
    if realized_daily_bottles >= 450.0:
        actionable_suggestions.append(
            "Proactive Staff Emptying Protocol: Train store personnel to empty collection bags promptly when the 80% bin-full alert is triggered. In high-volume stores, machine-full downtime on busy Friday/Saturday afternoons permanently destroys up to 25% of potential daily volume."
        )

    # Suggestion 5: Hardware & Financing optimization
    req_m = requested_machine.upper()
    if realized_daily_bottles >= 650.0 and any(c in req_m for c in ["V2", "DAINT_B", "COMPACT"]):
        actionable_suggestions.append(
            "Hardware Compactor Upgrade: Upgrade hardware from compact to a heavy-duty compactor (DAINT_G / Ripet_C-front) with scheduled twice-weekly logistics collection to eliminate bin-full lockouts."
        )
    elif realized_daily_bottles < 250.0:
        actionable_suggestions.append(
            "Capital Cost Mitigation: Deploy an entry-level compact single-stream unit (Ripet_V2) or negotiate merchant co-financing / a 6-month probationary review with footfall commitments to protect return on capital."
        )

    # ==================================================================
    # 4. FACTORS CAUSING ACCURACY DISCREPANCIES WITH OPERATIONAL ACTUALS
    # ==================================================================
    is_tourist_region = any(reg in regione.lower() for reg in ["valle d'aosta", "trentino", "alto adige", "liguria", "marche", "costa", "mare"])
    if (population < 20000 or population_density < 300) and (sales_area_sqm >= 750 or is_tourist_region or total_grocery_stores <= 3):
        accuracy_variance_factors.append(
            f"Tourist / Commuter Corridor Effect (Potential Under-Prediction): "
            f"The model's baseline uses permanent municipal census data ({population:.0f} inh in {regione}). "
            f"If this store sits on a regional tourist gateway (e.g. alpine ski/hiking routes, coastal vacation corridors, or state highways), "
            f"seasonal floating foot traffic can swell actual collection volumes by +30% to +80% above permanent census projections."
        )

    if (total_grocery_stores <= 1) or (total_grocery_stores <= 2 and sales_area_sqm >= 450):
        accuracy_variance_factors.append(
            f"Local Grocery Monopoly Hub (Potential Under-Prediction): "
            f"With only {total_grocery_stores} food store(s) in the trade area, this location functions as the sole grocery destination for the entire municipality, "
            f"concentrating 100% of local grocery shoppers and recycling foot traffic beyond standard national benchmarks."
        )

    if population_density < 120.0 and realized_daily_presenze <= 8.0:
        accuracy_variance_factors.append(
            f"Rural Bulk Depositing Behavior: In rural/agricultural municipalities (density {population_density:.1f} inh/km2), "
            f"shoppers tend to accumulate large volumes at home and deposit in bulk batches (35-50+ bottles/user) by car rather than frequent walk-in drops, "
            f"yielding higher daily collection volumes than suburban foot-traffic models suggest."
        )

    if nearest_coripet_km <= 0.05:
        accuracy_variance_factors.append(
            f"Existing Hardware Conflict (0m Overlap): "
            f"An active Coripet machine is already operating at this exact location ('{nearest_pf_name}'). "
            f"If evaluating an additional unit, incremental net volume will be near zero (-50% to -100% self-cannibalization) unless replacing the existing machine."
        )

    if realized_daily_bottles >= 700.0 or unconstrained_daily_bottles >= 750.0:
        accuracy_variance_factors.append(
            "Mechanical Bin Capacity & Servicing Bottlenecks: "
            "For stores with demand exceeding ~750 bottles/day, bin capacity fills rapidly. "
            "If store personnel fail to empty collection bags promptly, machine full-bin downtime will artificially depress actual collected volume below theoretical demand."
        )

    if geocoding_quality not in ("EXACT_STREET_NUMBER", "EXACT_CAP_COMUNE_MATCH"):
        accuracy_variance_factors.append(
            f"Geocoding Centroid Fallback: "
            f"The location was resolved at municipal centroid ('{geocoding_quality}'). Exact street-level walking catchment may exhibit slight variance in local commercial foot traffic."
        )

    return {
        "primary_performance_drivers": primary_drivers,
        "limiting_factors": limiting_factors,
        "actionable_suggestions": actionable_suggestions,
        "accuracy_variance_factors": accuracy_variance_factors,
        "collection_rate_drivers": primary_drivers + limiting_factors
    }


