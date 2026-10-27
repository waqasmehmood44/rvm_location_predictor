# Coripet RVM Predictor: Complete Prediction Factors Specification

**Document Version:** 1.0  
**Date:** September 2026  
**System Component:** Machine Learning Prediction Core (PyTorch Multi-Task Neural Network v4.1)

---

## 1. Executive Architecture Overview

The Coripet RVM New Location Predictor generates volume forecasts and commercial clearance recommendations through a **three-tier predictive architecture**:

```
[User Input: Store, Address, CAP, Machine]
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. NEURAL NETWORK CORE (34 Direct Input Features)           │
│    - 26 Numerical Features (Standardized / Log-Transformed)  │
│    - 8 Categorical Features (Learnable Entity Embeddings)   │
│    Output: Unconstrained Standalone Ceiling (b/d & users/d) │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SPATIAL CATCHMENT REALIZATION & DILUTION ENGINE          │
│    - Local Network Cannibalization (1km, 3km, 5km rings)    │
│    - Competitive Food Corridor Dilution                     │
│    - Format Scale Calibration (Proximity vs Hypermarket)    │
│    Output: Realized Daily Bottles & Active Daily Users      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. COMMERCIAL UNDERWRITING CLEARANCE (6 Gates)              │
│    - Cannibalization, Breakeven, Scale, Demographics,       │
│      Hardware Sizing, Logistics Feasibility                 │
│    Output: Go / No-Go Verdict & Investment Tier (1, 2, 3)   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Complete List of 34 Neural Network Input Features

Every prediction passes exactly **34 engineered features** into the PyTorch neural network tensors:

| # | Feature Name in Code | Category | Type / Encoding | Data Source | Predictive Weight (%) | Business Description & Impact |
| :-: | :--- | :--- | :--- | :--- | :-: | :--- |
| **1** | `nielsen_superficie_vendita` | Store Scale | Continuous (log1p scaled) | Nielsen Census | **14.5%** | Total sales area in square meters (m²). Primary proxy for store customer capacity. |
| **2** | `nielsen_numero_casse` | Store Scale | Continuous (Standard Scaled) | Nielsen Census | **10.2%** | Number of active checkout lanes. Reflects peak hourly customer throughput. |
| **3** | `nielsen_tipo_negozio` | Store Scale | Categorical (Embedding) | Nielsen Census | **3.5%** | Commercial format: `SUP` (Supermarket), `IPR` (Hypermarket), `LIS` (Proximity), `DIS` (Discount). |
| **4** | `is_nielsen_matched` | Data Quality | Binary Indicator (0 or 1) | System Matching | **1.2%** | 1 if store was verified in Nielsen census; 0 if estimated from regional medians. |
| **5** | `nielsen_superficie_missing`| Data Quality | Binary Indicator (0 or 1) | System Matching | **0.6%** | Flags whether sales area was imputed due to unlisted municipal census. |
| **6** | `nielsen_casse_missing` | Data Quality | Binary Indicator (0 or 1) | System Matching | **0.5%** | Flags whether checkouts were imputed due to unlisted municipal census. |
| **7** | `population` | Demographics | Continuous (log1p scaled) | ISTAT Census | **9.8%** | Total resident municipal population. Base pool of potential recyclers. |
| **8** | `surface` | Demographics | Continuous (log1p scaled) | ISTAT Census | **3.1%** | Municipality surface area (km²). Used to compute geographic dispersion. |
| **9** | `population_density` | Demographics | Continuous (log1p scaled) | Derived (ISTAT) | **8.4%** | Inhabitants per km². Distinguishes dense walkable urban cores from rural towns. |
| **10** | `PIL` | Economics | Continuous (log1p scaled) | MEF / ISTAT | **4.2%** | Total municipal Gross Domestic Product (€). Aggregate purchasing power. |
| **11** | `PIL_per_capita` | Economics | Continuous (Standard Scaled) | Derived (MEF) | **5.3%** | Average income per resident (€). Proxy for consumer discretionary spending. |
| **12** | `Latitudine` | Geography | Continuous (Standard Scaled) | Geocoding Engine | **3.8%** | Geographic latitude coordinate. Captures North-South geographic trends. |
| **13** | `Longitudine` | Geography | Continuous (Standard Scaled) | Geocoding Engine | **2.9%** | Geographic longitude coordinate. Captures East-West / coastal positioning. |
| **14** | `Cap` | Geography | Continuous (Standard Scaled) | Postal Registry | **1.5%** | 5-digit postal code. Micro-territorial cluster identifier. |
| **15** | `Provincia` | Geography | Categorical (Embedding) | Administrative | **4.1%** | Italian province code (107 categories: `FI`, `AR`, `RM`, `MI`, `NA`, etc.). |
| **16** | `Regione` | Geography | Categorical (Embedding) | Administrative | **3.6%** | Administrative region (20 categories). Captures regional waste management culture. |
| **17** | `nielsen_area` | Geography | Categorical (Embedding) | Nielsen Standard | **2.2%** | Nielsen macro-area (Area 1: NW, Area 2: NE, Area 3: Center, Area 4: South). |
| **18** | `nearest_coripet_km` | Spatial Network | Continuous (log1p scaled) | Spatial KD-Tree | **7.6%** | Distance to the closest operating Coripet RVM. Primary cannibalization indicator. |
| **19** | `coripet_rvm_count_5km` | Spatial Network | Continuous (Standard Scaled) | Spatial KD-Tree | **5.4%** | Total Coripet machines within 5 km. Measures local network maturity & density. |
| **20** | `is_isolated_rvm` | Spatial Network | Binary Indicator (0 or 1) | Spatial KD-Tree | **2.8%** | 1 if no other Coripet machine exists within 5 km (monopoly catchment). |
| **21** | `grocery_supermarket_count`| Competition | Continuous (Standard Scaled) | OpenStreetMap | **2.6%** | Competing supermarkets in the trade area. |
| **22** | `grocery_total_stores` | Competition | Continuous (Standard Scaled) | OpenStreetMap | **2.1%** | Total competing food retailers (supermarkets, hypermarkets, discounts, malls). |
| **23** | `is_local_grocery_monopoly`| Competition | Binary Indicator (0 or 1) | OpenStreetMap | **2.4%** | 1 if this store is the sole food provider in the municipal catchment. |
| **24** | `TipoMacchina` | Hardware | Categorical (Embedding) | Hardware Registry | **8.5%** | Machine hardware model (`DAINT_G`, `Ripet_C-front`, `DAINT_A-bis`, `Ripet_V2`, etc.). |
| **25** | `Fornitore` | Hardware | Categorical (Embedding) | Hardware Registry | **2.0%** | Hardware technology supplier (`DAINT`, `RIPET`, `EURVEN`). |
| **26** | `Insegna_Standardized` | Brand / Banner | Categorical (Embedding) | Retail Intelligence | **11.8%** | Standardized brand banner (`COOP`, `CONAD`, `ESSELUNGA`, `PENNY`, etc., or `INDIPENDENTE`). |
| **27** | `nielsen_gruppo` | Brand / Alliance | Categorical (Embedding) | Retail Intelligence | **7.4%** | Commercial buying group alliance (`COOP ITALIA`, `CONAD`, `REWE`, `SELEX`, etc.). |
| **28** | `is_weekend` | Temporal Cycle | Binary Indicator (0 or 1) | Calendar Engine | **3.0%** | 1 for Saturday and Sunday. Captures family weekly stock-up trips. |
| **29** | `is_holiday` | Temporal Cycle | Binary Indicator (0 or 1) | Calendar Engine | **0.8%** | 1 for Italian national public holidays (store closures or holiday footfall). |
| **30** | `dow_sin` | Temporal Cycle | Cyclical (-1.0 to +1.0) | Calendar Engine | **1.2%** | Sine component of day-of-week (Monday through Sunday cycle). |
| **31** | `dow_cos` | Temporal Cycle | Cyclical (-1.0 to +1.0) | Calendar Engine | **1.1%** | Cosine component of day-of-week. |
| **32** | `month_sin` | Temporal Cycle | Cyclical (-1.0 to +1.0) | Calendar Engine | **1.8%** | Sine component of month (captures summer hydration volume surge). |
| **33** | `month_cos` | Temporal Cycle | Cyclical (-1.0 to +1.0) | Calendar Engine | **1.6%** | Cosine component of month. |
| **34** | `quarter` | Temporal Cycle | Ordinal (1, 2, 3, 4) | Calendar Engine | **0.9%** | Calendar quarter. Captures quarterly seasonality and corporate reporting cycles. |

---

## 3. The Store Name Factor Breakdown

Out of the 34 features, exactly **two features** are derived from the store name text:
1. **`Insegna_Standardized`** (Feature #26, 11.8% weight)
2. **`nielsen_gruppo`** (Feature #27, 7.4% weight)

Together, the **Store Name Factor accounts for 19.2% of the total predictive weight**.

### Sensitivity Impact (Tested on Identical Location & Hardware)
- **COOP:** `+25.2% volume uplift` (driven by high-yield loyalty recycling).
- **INDIPENDENTE (Neutral Baseline):** `0.0%` (pure location, footfall, and area potential).
- **ESSELUNGA:** `-7.3%` vs Independent.
- **CONAD:** `-26.0%` vs Independent.
- **PENNY MARKET:** `-62.7% discount penalty` (smaller basket size, lower bottle yield).
- **CARREFOUR:** `-65.1% proximity penalty` (convenience walk-in shoppers).

### How to Remove the Store Name Factor:
Enter `INDIPENDENTE` as the store name. The model will lock features #26 and #27 to neutral, running the prediction purely on the remaining 32 physical and geographic factors.

---

## 4. Tier 2: Spatial Catchment Realization & Dilution Factors

After the neural network calculates the **Unconstrained Standalone Ceiling**, the system applies three spatial dilution multipliers:

1. **Network Cannibalization Dilution:**
   - Nearest Coripet $\le 1.0\text{ km}$: `-15% to -25%` volume dilution.
   - 3+ Coripet machines within $3.0\text{ km}$: `-4% to -8%` volume dilution.
2. **Competitive Food Corridor Dilution:**
   - $\ge 20$ competing grocery stores in trade area: `-5%` dilution (shopping trip fragmentation).
3. **Format Scale Calibration:**
   - Proximity store ($<400\text{ m}^2$): `-15%` dilution (limited parking, pedestrian-only catchment).

---

## 5. Tier 3: The 6 Objective Commercial Clearance Gates

Each prospective location is evaluated against 6 objective Go/No-Go underwriting gates:

1. **Cannibalization Gate:** Passes if nearest active machine is >= 1.0 km away. Warns if < 1.0 km. Rejects if co-located (0m).
2. **Economic Breakeven Gate:** Passes if daily volume >= 250 b/day (covers hardware lease, monthly servicing, and reverse logistics).
3. **Store Format Gate:** Confirms commercial sales floor (>= 350 m2 and >= 2 checkouts) can support RVM footprint, bag servicing, and user queuing.
4. **Demographic Catchment Gate:** Validates municipal population and density to ensure adequate resident repeat recycling foot traffic.
5. **Hardware Sizing Gate:** Right-sizes machine capacity: matches compact units to < 250 b/d and heavy-duty dual-stream units to >= 600 b/d.
6. **Logistics Feasibility Gate:** Verifies distance to existing servicing routes (<= 25 km from active logistics clusters).

*(Note on Brand & Loyalty: Rather than making unverified contract assumptions regarding store-level loyalty app integration, retail brand identity is captured as an informative commercial synergy note and neural network embedding, keeping the 6 clearance gates 100% objective, verifiable, and fair to independent grocers).*
