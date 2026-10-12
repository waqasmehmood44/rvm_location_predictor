# Coripet RVM Prospective Location Predictor
## Complete System Architecture, Predictive Modeling & Business Rules Specification

---

## 1. Executive Summary & System Objectives

The **Coripet RVM Prospective Location Predictor** is a production-grade machine learning and spatial intelligence decision-support platform designed to evaluate prospective retail and commercial locations across Italy for the deployment of **Reverse Vending Machines (RVMs)**.

### Core Business Purpose
When an enterprise retailer, independent supermarket, or Coripet business development manager evaluates a candidate store for an RVM installation, the system answers five critical operational and investment questions:
1. **Expected Collection Volumes**: How many PET bottles will this location collect per day, month, and year under steady-state operational conditions?
2. **User Footfall & Habit**: How many unique recycling users (`PRESENZE`) will visit daily, and what is the interaction efficiency (`bottles/person`)?
3. **Catchment Friction & Cannibalization**: How much volume will be diluted by existing nearby Coripet machines or heavy supermarket competition?
4. **Site Clearance & Suitability (Go / No-Go)**: Does this site meet physical, demographic, logistical, and economic criteria to justify capital expenditure?
5. **Actionable Strategic Deployment**: What machine model should be deployed (compact vs. heavy-duty), what servicing frequency is required, and what commercial contract terms (e.g., retailer co-financing) are necessary?

### Minimal User Input Requirement
To evaluate any prospective store in Italy, only four inputs are required:
* **Store Name / Insegna**: e.g., `Esselunga Milano Certosa`, `Decò`, `Conad Superstore`, `Famila`
* **Address**: e.g., `Via delle Forze Armate 12`, `Via Circonvallazione 126`
* **Postal Code (CAP)**: 5-digit Italian CAP, e.g., `20147`, `11029`, `90121`
* **Machine Model**: e.g., `DAINT_A-bis`, `DAINT_G`, `Ripet_C-front`, `Ripet_V2`

*(Optional overrides: Store format preset 1–5, exact sales area in m², or checkout register count).*

---

## 2. Multi-Level Intelligence & Feature Architecture

The system processes candidate locations through a four-level hierarchical intelligence pipeline before invoking neural network inference:

```
Candidate Store Input: Store Name, Address, CAP, Machine Type
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Geographic & Spatial Disambiguation                           │
│  - 5-Digit CAP Validation & Italian Census (ISTAT) Matching            │
│  - Multi-tier Fallback (Exact CAP+Comune -> Street -> Municipality)    │
│  - Demographic Enrichment: Population, Density (inh/km²), Surface, PIL │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: Store Format & Scale Auto-Estimation                          │
│  - 25,104 Nielsen Store Universe Knowledge Base                       │
│  - Auto-infers Format (IPER, SUP, DIS, VIC) & Square Meters / Casse   │
│  - Accommodates User Custom Scale or Preset Format Overrides          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: Retail Intelligence & Commercial Alliances                    │
│  - Standardized Brand Extraction (e.g. "IPERCOOP" -> "COOP")           │
│  - National Buying Group Resolution (CONAD, COOP, SELEX, VEGE, etc.)   │
│  - Nielsen Geographic Area Mapping (Area 1 NW, Area 2 NE, 3 Center, 4) │
│  - Neural Entity Embedding Conditioning                                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 4: Spatial Catchment, Competitive Density & Network Analysis     │
│  - Coripet Network Analysis: Nearest RVM (km), Concentric Densities    │
│  - Network Position: CANNIBALIZATION_ALERT, DENSE_CLUSTER, EXPANSION   │
│  - Grocery Density: Supermarkets, Hypermarkets, Discounts, Malls       │
│  - Local Grocery Monopoly Detection (Trade Area Stores <= 1)           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ NEURAL NETWORK SIMULATION (MultiTaskRVMNet)                            │
│  - 84-Day Representative Grid (12 Mid-Month Weeks across 2026)         │
│  - Dual-Head Inference: log1p(Bottiglie) and log1p(Presenze)           │
│  - Balanced 4-Season Marginalization & Deseasonalized Day-of-Week     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ REALIZATION, DILUTION & MONOPOLY PHYSICS LAYER                         │
│  - Realization Factor (κ): Cannibalization, Cluster Saturation, Food   │
│  - Rural Monopoly Gravity (+45% / +20%) & Southern Bulk Drop (+20%)    │
│  - Dual-Metric Reporting: Arithmetic Mean & Median Side-by-Side        │
│  - Statistical Quantiles: P25 Floor, P50 Steady-State, P75 Peak        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LOCATION SUITABILITY & CLEARANCE GATE ENGINE                           │
│  - 7 Clearance Gates: Cannibalization, Breakeven, Format, Demographics,│
│    Hardware Sizing, Loyalty Alliance, Logistics Radius                 │
│  - Commercial Mall / Shopping Center Anchor Synergy                    │
│  - Suitability Score (0-100) & Verdict: GO / CONDITIONAL / NO-GO       │
│  - Investment Tier & Root-Cause Predictive Diagnostics                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Level 1: Geographic & Spatial Disambiguation
* **CAP Normalization**: Padded to 5 digits (e.g., `'11029'`, `'00185'`).
* **Multi-Tier Resolution**:
  1. *Exact Match*: Matches `(CAP, Comune)` in the official Italian municipal census.
  2. *Street Address Geocoding*: Geocodes street names to municipal centroids.
  3. *Comune Registry Lookup*: Resolves regional and provincial attributes.
  4. *National Median Fallback*: Used if postal records are unrecognized.
* **Census & Economic Attributes Extracted**:
  - `Comune`, `Provincia`, `Regione`, `Latitudine`, `Longitudine`
  - `population`: Total resident population in the municipality.
  - `surface`: Municipal surface area in km².
  - `population_density`: Inhabitants per km² ($population / surface$).
  - `PIL`: Municipal Gross Domestic Product.
  - `PIL_per_capita`: Per capita economic wealth indicator.

---

### Level 2: Store Format & Scale Auto-Estimation
Reverse vending machine throughput is strongly conditioned by store footprint and checkout capacity. When the user does not specify square meters or checkout counts, the system automatically infers them from the **Nielsen 25,104 Italian Store Universe**:

| Store Format | Code | Typical Sales Area (m²) | Typical Checkouts | Example Banners |
| :--- | :---: | :---: | :---: | :--- |
| **Hypermarket** | `IPER` | 2,500 – 10,000 m² | 12 – 40 checkouts | Ipercoop, Spazio Conad, Carrefour Iper |
| **Superstore / Large Super** | `SUP` | 1,200 – 2,500 m² | 6 – 15 checkouts | Esselunga, Conad Superstore, Famila Superstore |
| **Standard Supermarket** | `SUP` | 600 – 1,200 m² | 3 – 6 checkouts | Conad, Coop, Despar, Tigros |
| **Discount Store** | `DIS` | 750 – 1,100 m² | 4 – 6 checkouts | Eurospin, Lidl, MD, Penny Market |
| **Convenience / Proximity** | `LIS`/`VIC` | 150 – 500 m² | 1 – 3 checkouts | Carrefour Express, Conad City, Crai, In's |

*User overrides via `sales_area_sqm`, `cash_registers`, or presets 1–5 immediately bypass auto-estimation.*

---

### Level 3: Retail Intelligence & Commercial Alliances
* **Standardized Brand Extraction (`clean_brand_name`)**: Strips company suffixes, store numbers, and regional tags (e.g. `DECO' MAXISTORE PALERMO` -> `DECO`, `IPERCOOP BARI` -> `COOP`).
* **Retail Buying Group Resolution (`infer_retail_group`)**: Maps retail banners to their parent commercial alliances:
  - `CONAD`, `COOP ITALIA`, `ESSELUNGA`, `SELEX`, `GRUPPO VEGE`, `CRAI`, `CARREFOUR`, `AGORA NETWORK`, `DESPAR`, `EUROSPIN`, `LIDL`, `MD`.
  - Unaligned banners are mapped to `INDIPENDENTE` / `UNKNOWN`.
* **Nielsen Geographic Area Mapping**:
  - `Area 1 (Nord-Ovest)`: Piemonte, Valle d'Aosta, Liguria, Lombardia
  - `Area 2 (Nord-Est)`: Trentino-Alto Adige, Veneto, Friuli-Venezia Giulia, Emilia-Romagna
  - `Area 3 (Centro)`: Toscana, Umbria, Marche, Lazio, Sardegna
  - `Area 4 (Sud e Isole)`: Abruzzo, Molise, Campania, Puglia, Basilicata, Calabria, Sicilia
* **Neural Entity Embeddings**: Enables active brand and alliance embeddings during neural network forward passes.

---

### Level 4: Spatial Catchment, Competitive Density & Network Analysis
Analyzes spatial competition and network positioning using haversine geodetic distance:
* **Coripet Network Metrics**:
  - `nearest_coripet_km`: Distance to the nearest operational Coripet machine.
  - `coripet_count_1km`, `coripet_count_3km`, `coripet_count_5km`: Machine density within concentric radii.
  - `coripet_count_comune`: Total machines operating in the same municipality.
* **Network Position Classification**:
  - `CANNIBALIZATION_ALERT`: Existing machine within < 1.2 km or co-located at 0m.
  - `DENSE_CLUSTER`: >= 5 machines within 5 km (mature urban network).
  - `EXPANSION_NODE`: Healthy intermediate network spacing (1.5 – 10 km).
  - `ISOLATED_RVM`: Nearest machine > 10 km away (greenfield expansion).
* **Competitive Grocery Density (Nielsen 25,104 Store Base)**:
  - Counts of Supermarkets, Hypermarkets, Discounts, Neighborhood stores, and Commercial Shopping Centers (`shopping_centers`).
  - `is_local_monopoly`: True if trade area food stores <= 1 (or supermarkets <= 1 with 0 hypermarkets).

---

## 3. Deep Learning Architecture (`MultiTaskRVMNet`)

The predictive core is a custom PyTorch Multi-Task Tabular Neural Network trained on millions of historical daily Coripet operational records.

### Network Architecture Specification
1. **Categorical Entity Embeddings**:
   - Categorical inputs (`Insegna_Standardized`, `TipoMacchina`, `Provincia`, `Regione`, `nielsen_gruppo`, `nielsen_area`, `nielsen_tipo_negozio`) pass through individual trainable embedding layers:
     $$\text{Embedding Dimension} = \min\left(50, \left\lfloor \frac{\text{Cardinality}}{2} \right\rfloor + 1\right)$$
   - Out-of-vocabulary or unseen banners map to dedicated `<UNK>` index `0`.
2. **Continuous Feature Normalization**:
   - Numeric features (`Latitudine`, `Longitudine`, `population`, `surface`, `population_density`, `PIL`, `PIL_per_capita`, `sales_area`, `checkouts`, `nearest_coripet_km`, `rvm_count_5km`, `grocery_stores`, `is_monopoly`) are normalized using `RobustScaler` (median-centered, IQR-scaled) to resist demographic outliers.
3. **Shared Representation Trunk**:
   - Dense layers: `[256 -> 128 -> 64]` with `GELU` non-linear activations, `LayerNorm`, and `Dropout(0.2)`.
   - Residual skip-connections allow gradients to flow directly between early tabular representations and multi-task heads.
4. **Specialized Multi-Task Regression Heads**:
   - **`bottiglie_head`**: Predicts $\log(1 + \text{BOTTIGLIE})$.
   - **`presenze_head`**: Predicts $\log(1 + \text{PRESENZE})$.
   - Training loss: Joint Multi-Task Mean Squared Error:
     $$\mathcal{L} = 0.5 \cdot \text{MSE}(\log(1+\hat{b}), \log(1+b)) + 0.5 \cdot \text{MSE}(\log(1+\hat{p}), \log(1+p))$$
   - Physical count recovery at inference:
     $$\hat{y} = \max(0, \exp(\hat{z}) - 1)$$

---

### 84-Day Calendar Simulation Grid (Seasonal & Day-of-Week Marginalization)
Rather than evaluating a single static day, the model simulates an **84-day representative calendar** composed of 12 full Monday-to-Sunday weeks centered on the second Monday of each month across 2026:
* **Quarterly Seasons Evaluated**:
  - **Winter (Dec – Feb)**: Off-peak beverage consumption floor (~0.85x annual baseline).
  - **Spring (Mar – May)**: Moderate recovery baseline (~0.98x annual baseline).
  - **Summer (Jun – Aug)**: Peak hydration and recycling season (~1.15x annual baseline).
  - **Autumn (Sep – Nov)**: Steady-state return (~0.98x annual baseline).
* **Deseasonalized Day-of-Week Rhythm**:
  - Aggregates all 12 Mondays for Monday, 12 Tuesdays for Tuesday, through 12 Sundays.
  - Generates day-by-day staffing profiles for supermarket operations.

---

### Dual-Metric Reporting: Arithmetic Mean vs. Median Side-by-Side
Both **Arithmetic Mean** and **Median (50th percentile)** are reported across all operational, seasonal, and weekly tables:

1. **Arithmetic Mean**:
   - **Linear & Additive**: $\text{Mean} \times 365 = \text{Annual Cumulative Collection}$.
   - Essential for **financial budgeting, material weight forecasting, and EPR recycling fee revenue**.
2. **Median (50th Percentile)**:
   - **Outlier-Resistant**: Immune to temporary summer surges, holiday promotions, or abnormal machine downtime.
   - Represents the **typical day-to-day experience** for supermarket store managers and logistics teams.

---

## 4. Catchment Realization, Dilution & Monopoly Physics Layer

The neural network outputs an **Unconstrained Standalone Potential** ($V_{\text{unconstrained}}$) representing theoretical volume assuming a standalone monopoly. In reality, actual collections are governed by local competition, network cannibalization, or monopoly gravity.

The **Catchment Realization Layer** computes the net **Realized Operational Forecast** ($V_{\text{realized}}$):
$$V_{\text{realized}} = V_{\text{unconstrained}} \times \kappa$$
$$\text{Presenze}_{\text{realized}} = \text{Presenze}_{\text{unconstrained}} \times \sqrt{\kappa}$$

The realization factor $\kappa$ is computed dynamically:

### 1. Direct Cannibalization / Co-Location Penalty ($\kappa_{\text{cannibal}}$)
* **0m Overlap / Co-location**: If an existing machine is on the same premises:
  - Penalty: **-45% to -50%** customer sharing (unless verified as a self-match).
* **Nearby RVM <= 500m**: **-30%** dilution.
* **Nearby RVM <= 1.0 km**: **-15%** dilution.

### 2. Multi-RVM Cluster Saturation Penalty ($\kappa_{\text{density}}$)
* **>= 8 Coripet RVMs within 3 km**: **-32%** customer dilution (extreme urban saturation).
* **>= 5 Coripet RVMs within 3 km**: **-20%** customer dilution.
* **>= 3 Coripet RVMs within 3 km**: **-10%** customer dilution.

### 3. Commercial Food Retail Corridor Fragmentation ($\kappa_{\text{comp}}$)
* **>= 30 grocery competitors in trade area**: **-14%** customer fragmentation.
* **>= 20 grocery competitors in trade area**: **-8%** customer fragmentation.

### 4. Urban Proximity Format Calibration ($\kappa_{\text{format}}$)
* Small independent convenience stores (< 400 m² or <= 2 checkouts) in urban areas receive a **-28%** walk-in calibration reflecting pedestrian basket sizes.

---

### 5. Rural Monopoly & Southern Bulk Deposit Physics
For rural or isolated stores, pedestrian discounts are bypassed and empirical positive gravity is applied:
* **Walk-In Penalty Exemption**: Bypassed whenever `grocery_total_stores <= 1` or `population_density < 150 inh/km²`. In rural Italy, shoppers travel by car and carry bulk crates.
* **Sole Municipal Grocery Monopoly Uplift**:
  - If `grocery_total_stores <= 1` and `population_density < 150 inh/km²`: **+45% volume uplift**. The store acts as a regional destination hub for surrounding towns.
  - If `population_density < 400 inh/km²`: **+20% volume uplift**.
* **Southern Italy Rural Bulk Deposit Multiplier**:
  - In Nielsen Area 4 (Campania, Puglia, Calabria, Sicilia) with `population_density < 150 inh/km²`: **+20% bulk drop multiplier**. Rural Southern shoppers accumulate larger batches at home (empirical 43.5 b/user vs. 26.0 b/user national urban prior).

---

### Operational Capacity Scenarios (Statistical Quantiles)
Underwriting requires risk-managed operational boundaries:
* **Conservative Floor (P25)**: 25th percentile underwriting safety net (~82% of steady-state mean). Represents the baseline collection during adverse weather or local downtime.
* **Steady-State Base (P50)**: 50th percentile / **Median** operational expectation.
* **Optimistic Peak (P75)**: 75th percentile upper capacity ceiling (~125% of steady-state mean). Represents peak summer collection.

---

## 5. Location Suitability & Clearance Gate Engine (Go / No-Go Decision Gate)

The **Location Suitability & Clearance Gate Engine** serves as the authoritative operational and capital allocation gatekeeper. Every candidate location is evaluated across **7 operational, physical, and economic clearance gates**:

```
                       CANDIDATE STORE PROPOSAL
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GATE 1: Spatial Cannibalization Gate                                       │
│  [FAIL] Distance < 250m or Dilution >= 35% | [WARN] < 750m or Dilution >= 15%│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GATE 2: Economic Breakeven Gate                                            │
│  [FAIL] Realized < 108 b/d (Floor < 90 b/d) | [WARN] < 150 b/d              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GATE 3: Store Format & Footprint Gate                                      │
│  [FAIL] Sales Area < 150 m² or Checkouts < 2 | [WARN] Sales Area < 300 m²    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GATE 4: Demographic Catchment Gate                                         │
│  [FAIL] Pop < 2,000 & Density < 60 & Stores <= 1 | [WARN] Pop < 5,000      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GATE 5: Hardware-to-Volume Sizing Match Gate                               │
│  [WARN] Heavy-Duty Unit on < 250 b/d | [WARN] Compact Unit on > 600 b/d      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GATE 6: Commercial Loyalty Alliance Gate                                   │
│  [WARN] Independent Banner on < 180 b/d (No Loyalty App Vouchers)           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GATE 7: Logistics Servicing Radius Gate                                    │
│  [WARN] Remote Outpost > 25 km on < 250 b/d (High Deadhead Transport Cost)  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMMERCIAL MALL / SHOPPING CENTER ANCHOR SYNERGY                           │
│  Shopping Centers >= 1 & Area >= 1,200 m² -> +5 pts Bonus & Mall Anchor Tag │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
                     FINAL VERDICT & ACTION SYNTHESIS
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
   NO-GO [REJECT]         CONDITIONAL [REVIEW]             GO [APPROVED]
   (Any Gate FAIL)         (Any Gate WARN)           (All Gates PASS)
```

---

### Detailed Gate Specifications

#### Gate 1: Spatial Cannibalization Gate
* **Purpose**: Prevents deploying duplicate capital that merely splits existing Coripet customers without capturing incremental bottles.
* **Clearance Rules**:
  - `[FAIL]`: Distance to existing RVM < 0.25 km (250m), or net volume dilution $\ge 35\%$, or `CANNIBALIZATION_ALERT` with distance < 400m.
  - `[WARN]`: Distance < 0.75 km (750m), or dilution between 15% and 34.9%.
  - `[PASS]`: Autonomous catchment buffer (nearest machine $\ge 0.75\text{ km}$).
* **Score Impact**: `FAIL` = -45 points; `WARN` = -20 points.

#### Gate 2: Economic Breakeven Gate
* **Purpose**: Guarantees that projected collection volume recovers monthly machine lease depreciation, bin liner consumables, and logistics dispatch costs.
* **Operating Breakeven Floor**: **120.0 bottles/day** (~3,650 bottles/month).
* **Clearance Rules**:
  - `[FAIL]`: Realized volume < 108.0 b/d ($< 90\%$ of breakeven) OR Conservative P25 Floor < 90.0 b/d ($< 75\%$ of breakeven).
  - `[WARN]`: Realized volume between 108.0 b/d and 149.9 b/d ($< 125\%$ of breakeven). Tight operating margin.
  - `[PASS]`: Realized volume $\ge 150.0\text{ b/d}$. Commercially profitable.
  - *High-Volume Bonus*: Realized volume $\ge 600.0\text{ b/d}$ grants **+5 points**.
* **Score Impact**: `FAIL` = -40 points; `WARN` = -15 points.

#### Gate 3: Store Format & Physical Footprint Gate
* **Purpose**: Verifies that the store has sufficient physical area to house the RVM footprint, allow pallet jack access for 240L/1000L bin servicing, and accommodate queuing shoppers without blocking aisles.
* **Clearance Rules**:
  - `[FAIL]`: Sales area < 150.0 m² OR checkouts < 2.0. Insufficient physical scale.
  - `[WARN]`: Sales area between 150.0 m² and 299.9 m² OR checkouts == 2.0. Constrained convenience footprint.
  - `[PASS]`: Sales area $\ge 300.0\text{ m}^2$ AND checkouts $\ge 3.0$. Optimal scale.
* **Score Impact**: `FAIL` = -30 points; `WARN` = -10 points.

#### Gate 4: Demographic Catchment Gate
* **Purpose**: Validates that the municipality has sufficient permanent resident population and commercial density to establish a recurring weekly recycling habit.
* **Clearance Rules**:
  - `[FAIL]`: Municipal population < 2,000 AND density < 60.0 inh/km² AND grocery stores $\le 1$.
  - `[WARN]`: Population < 5,000 OR density < 120.0 inh/km². Modest catchment.
  - `[PASS]`: Population $\ge 5,000$ AND density $\ge 120.0\text{ inh/km}^2$. Dense demographics.
* **Score Impact**: `FAIL` = -25 points; `WARN` = -10 points.

#### Gate 5: Hardware-to-Volume Sizing Match Gate
* **Purpose**: Optimizes machine capex and compaction capacity against projected volume.
  - *Heavy-Duty Models*: `DAINT_G`, `Ripet_C-front`, `DAINT_A-bis`, `Ripet_Smarpet`, `Supermarpet`.
  - *Compact Models*: `Ripet_V2`, `DAINT_B`, `DAINT_B-bis`, `Ripet1000`, `Cestone`.
* **Clearance Rules**:
  - `[WARN] (Oversized Hardware)`: Heavy-duty machine requested on low volume (< 250.0 b/d). Advises downsizing to compact unit (`Ripet_V2` / `DAINT_B`) to protect ROI.
  - `[WARN] (Undercapacity Risk)`: Compact machine requested on high volume (> 600.0 b/d). Warns of bin overflow downtime; advises upgrade to heavy-duty unit (`DAINT_G` / `Ripet_C-front`).
  - `[PASS]`: Hardware model properly matched to volume tier.
* **Score Impact**: `WARN` = -10 points.

#### Gate 6: Commercial Loyalty Alliance Gate
* **Purpose**: Evaluates retail chain loyalty app integration. National supermarket chains offer automated loyalty voucher points in their apps, which drives higher return frequencies compared to independent grocers.
* **Clearance Rules**:
  - `[WARN]`: Independent banner (`brand == INDIPENDENTE` or `retail_group == UNKNOWN`) with volume < 180.0 b/d. Lacks national loyalty voucher incentives; recommends retailer co-financing or foot-traffic guarantees.
  - `[PASS]`: Active commercial alliance (`CONAD`, `COOP`, `ESSELUNGA`, `VEGE`, etc.) OR independent banner with self-sustaining volume $\ge 180.0\text{ b/d}$.
* **Score Impact**: `WARN` = -10 points.

#### Gate 7: Logistics Feasibility & Servicing Radius Gate
* **Purpose**: Evaluates route economics. Coripet collection trucks operate on scheduled regional milk-runs. Isolated sites incur high deadhead transport costs.
* **Clearance Rules**:
  - `[WARN]`: Distance to nearest Coripet node > 25.0 km AND realized volume < 250.0 b/d. Recommends bi-weekly or on-demand dispatch to contain logistics costs.
  - `[PASS]`: Integrated into active servicing route ($\le 25.0\text{ km}$) OR high-yield outpost ($\ge 250.0\text{ b/d}$) where high volume justifies dedicated routing.
* **Score Impact**: `WARN` = -10 points.

---

### Commercial Mall / Shopping Center Anchor Synergy
* **Detection Criteria**: `shopping_centers >= 1` AND `sales_area_sqm >= 1,200.0 m²`.
* **Synergy Impact**: Automatically awards **+5 points** to the suitability score and attaches the strategic strength:
  > *"Prime Commercial Park / Mall Anchor: Benefits from shared customer parking and multi-purpose weekend family foot traffic."*

---

### Suitability Score & Decision Verdicts

The composite **Suitability Score** starts at 100 points, applies gate deductions and bonuses, and is clipped to `[5, 99]`:

| Verdict | Suitability Status | Gate Conditions | Actionable Strategic Guidance |
| :--- | :--- | :--- | :--- |
| **`GO [PRIORITY APPROVED]`** | `HIGHLY_SUITABLE` | All 7 Gates `[PASS]`, Score $\ge 75$, Volume $\ge 500$ b/d | Approved for immediate high-priority deployment. Deploy heavy-duty multi-material unit (`DAINT_G` / `Ripet_C-front`) with twice-weekly servicing. |
| **`GO [STANDARD APPROVED]`** | `SUITABLE` | All 7 Gates `[PASS]`, Score $\ge 55$ | Approved for standard network expansion. Deploy standard multi-material unit (`DAINT_B-bis` / `Ripet_C-front`) with weekly servicing rhythm. |
| **`CONDITIONAL [REVIEW REQUIRED]`** | `CONDITIONAL_PILOT` | Any Gate is `[WARN]` OR Score < 55 | Proceed only under conditional terms: right-size hardware, require retailer co-financing, or adopt bi-weekly logistics dispatch. |
| **`NO-GO [REJECT]`** | `NOT_SUITABLE` | Any Gate is `[FAIL]` | **DO NOT install.** Direct shoppers to existing nearby machines or reject due to sub-breakeven volume or footprint deficits. |

---

## 6. Commercial Investment Tier Recommendations

In addition to site suitability, candidate stores receive an investment portfolio ranking based on the **Location Potential Score** (0–100 nationwide percentile):

* **Tier 1 - Flagship Anchor Hub (Score >= 80)**: Top nationwide volume generator. Decision: `HIGH PRIORITY (PROCEED)`. Logistics: Twice-weekly servicing.
* **Tier 2 - Viable Standard Expansion (Score 55–79)**: Healthy payback profile above the national median. Decision: `COMMERCIALLY VIABLE (PROCEED)`. Logistics: Weekly scheduled servicing.
* **Tier 3 - Secondary Candidate (Score 35–54)**: Modest volume. Decision: `CONDITIONAL (SELECTIVE APPROVAL)`. Logistics: Bi-weekly or co-financed servicing.
* **Tier 4 - Cannibalization / High Risk**: Triggered when `network_position == CANNIBALIZATION_ALERT`. Decision: `HIGH RISK (AVOID DUPLICATION)`.
* **Tier 5 - Not Suitable / Unviable**: Triggered when `suitability_status == NOT_SUITABLE`. Decision: `NOT RECOMMENDED (DO NOT INSTALL)`.

---

## 7. Operational & Predictive Diagnostics Engine

To explain **why collection rates are constrained** or **why Power BI operational data may diverge from model predictions**, an automated diagnostic engine evaluates every run:

### Drivers of Constrained Collection (< 450 b/day)
* **Store Footprint**: Small convenience format (< 600 m² or $\le 3$ checkouts) physically limits customer throughput.
* **Corridor Competition**: High food store fragmentation ($\ge 15$ competitors) divides shopper traffic.
* **Network Proximity**: Nearby machines within < 1.5 km split recycling customers.
* **Demographics**: Small municipality (< 15,000 inh) or sparse density (< 250 inh/km²).
* **Loyalty Gap**: Independent banner lacks automated chain loyalty voucher marketing.
* **Seasonal Dip**: Winter/Autumn nationwide beverage sales contraction (-15% to -25%).

### Predictive Accuracy Variance Factors (Why Actuals Diverge from Predictions)
* **Alpine / Coastal / Commuter Tourist Corridor (Under-Prediction Risk)**:
  Supermarkets in small resident municipalities that sit on state highways, ski routes, or beach resort corridors (e.g. Sarre, Aosta Valley, Adriatic coast). Seasonal tourist surges can exceed permanent resident predictions by **+30% to +80%**.
* **Regional Grocery Monopoly Hub (Under-Prediction Risk)**:
  When trade area grocery stores $\le 2$, the store serves as the sole regional food magnet for multiple surrounding villages, concentrating regional traffic far beyond municipal census figures.

---

## 8. Configuration Reference (`configs/business_rules.yaml`)

All scoring weights, thresholds, and operational benchmarks are centrally maintained in `configs/business_rules.yaml`:

```yaml
operational_benchmarks:
  days_per_month: 30.4

scoring:
  weights:
    bottles_daily_weight: 0.45
    presenze_daily_weight: 0.35
    bottles_per_person_weight: 0.20
  benchmarks:
    bottles_daily_target_p90: 1000.0
    presenze_daily_target_p90: 35.0
    bottles_per_person_target: 30.0

recommendations:
  tiers:
    high_potential:
      min_score: 75.0
    medium_potential:
      min_score: 50.0
    low_potential:
      min_score: 0.0

confidence:
  factors:
    postal_code_exact_match_weight: 0.40
    retail_chain_known_weight: 0.35
    geo_proximity_weight: 0.25
  thresholds:
    high_confidence_min_score: 0.80
    medium_confidence_min_score: 0.55

suitability_gates:
  economic_breakeven_daily_bottles: 120.0
  critical_cannibalization_distance_km: 0.25
  critical_dilution_percentage: 35.0
  min_sales_area_sqm: 150.0
  min_cash_registers: 2.0
  min_population_isolated: 2000.0
  min_population_density_isolated: 60.0
  heavy_duty_min_daily_bottles: 250.0
  compact_max_daily_bottles: 600.0
  isolated_outpost_distance_km: 25.0
  isolated_min_daily_bottles: 250.0
  independent_min_daily_bottles: 180.0
```

---

## 9. Python API & Execution Guide

### Running a Prediction Programmatically
```python
from src.model.predict import RVMNewLocationPredictor

predictor = RVMNewLocationPredictor()

result = predictor.predict_new_location(
    store_name="ESSELUNGA",
    address="Via Portanuova 1",
    cap="51100",
    machine_type="DAINT_A-bis",
    sales_area_sqm=2990.0,
    cash_registers=33.0
)

# Access predictions
print("Realized Daily Bottles:", result["realized_forecast"]["realized_daily_bottles"])
print("Realized Daily Users:  ", result["realized_forecast"]["realized_daily_presenze"])
print("Suitability Verdict:   ", result["location_suitability"]["verdict"])
print("Suitability Score:     ", result["location_suitability"]["suitability_score"])
print("Recommended Action:    ", result["location_suitability"]["recommended_action"])
```

### Running the Interactive CLI
```bash
python -m src.model.predict
```

### Running the Test Suite
```bash
python -m unittest tests/test_suitability_engine.py
```
*(All 11 unit tests verify the 7 clearance gates, cannibalization penalties, rural monopoly uplifts, and hardware sizing rules).*
