# Coripet RVM Predictor: Feature Engineering Audit & Model Input Design

**Author:** Antigravity AI Engineering  
**Date:** September 2026  
**Status:** Certified Audit Completed — Feature Inventory Frozen (Model Training NOT Started)  
**Reference Parquet:** `data/processed/modeling_daily_rvm.parquet`  
**Reference Companion:** [feature_dictionary.csv](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/reports/feature_dictionary.csv) | [target_stats_summary.json](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/reports/target_stats_summary.json)

---

## Executive Summary

The data eligibility and integration phase has successfully established the **certified modeling population**:
* **1,396,544 daily observations**
* **1,657 unique physical points (`PF`)**
* **Date Range:** 01/01/2023 – 31/08/2026
* **Eligibility Gates:** 100% `PUNTOVENDITA`, 100% active as of 31/08/2026, 100% `STANDARD` configuration on observation day, 100% mature ($\text{GIORNO} \ge \text{DataPrimaAttivazione} + 3\text{ calendar months}$), and **0 PROMO observations**.

This audit report details the transition from historical data integration to **predictive feature engineering** for the final production objective:
> **Target Business Application:**  
> A user enters a **NEW prospective retail store location**:
> * **Input:** Store name, Address, CAP, Machine type
> * **Output:** Expected PRESENZE/day, Expected BOTTIGLIE/day, Expected BOTTIGLIE/month, Location Potential Score (%), Recommendation tier, and Prediction Confidence.

### Crucial Principle: Historical Outcomes vs. New-Location Predictors
* **Historical BOTTIGLIE and PRESENZE are preserved as supervised learning TARGETS.** The model must train on 1.39M real-world daily outcomes to learn the relationship between store/location characteristics and operational volume.
* **Historical BOTTIGLIE and PRESENZE are STRICTLY FORBIDDEN as PREDICTORS.** For a brand-new prospective store, historical collections, lags, rolling averages, and past trends do not exist. Any inclusion of historical performance features would constitute catastrophic data leakage.

```
Prospective Store Features (Location + Demographics + Store + Machine)
                             ↓
              Shared Neural Network Trunk
                             ↓
                 ┌───────────────────────┐
                 ↓                       ↓
         Head 1: PRESENZE        Head 2: BOTTIGLIE
                 ↓                       ↓
             Daily Expected          Daily Expected
                PRESENZE                BOTTIGLIE
                 └───────────┬───────────┘
                             ↓
                 Monthly Expected BOTTIGLIE
                 Location Potential Score (%)
                 Recommendation Tier & Confidence
```

---

## Stage 1 — Complete Feature Inventory

A comprehensive audit was executed across `modeling_daily_rvm.parquet`, raw historical operational logs (`daily_bottles.csv`, `CaricaViewStoricoPuntoFisico.csv`), external demographic sources (`population.csv`, `surface.csv`, `PIL.csv`), retail census files (`filenielsen_1203.csv`), and user application inputs.

A total of **72 candidate variables** were evaluated. The complete machine-readable catalog is saved at [`reports/feature_dictionary.csv`](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/reports/feature_dictionary.csv).

### Summary Classification Breakdown

| Category | Total Columns | Examples | Description |
| :--- | :---: | :--- | :--- |
| **Targets (Outcomes)** | 2 | `BOTTIGLIE`, `PRESENZE` | Supervised labels only; strictly excluded as inputs |
| **Core Location / Spatial** | 6 | `Latitudine`, `Longitudine`, `Cap`, `Comune`, `Provincia`, `Regione` | Derived from user input (Address + CAP); geocoded & administrative hierarchy |
| **Core Macro-Demographics** | 5 | `population`, `surface`, `population_density`, `PIL`, `PIL_per_capita` | Municipality-level socioeconomic census indicators (>99% coverage) |
| **Core Hardware / Machine** | 2 | `TipoMacchina`, `Fornitore` | Hardware model family & OEM vendor selected by user |
| **Core Store Banner / Brand** | 2 | `Store_Name_Cleaned`, `Insegna_Standardized` | Cleaned retail chain banner derived from user input |
| **Optional Nielsen Enrichment** | 8 | `nielsen_superficie_vendita`, `nielsen_numero_casse`, `nielsen_tipo_negozio`, etc. | Detailed retail metrics; optional enrichment with missingness indicators |
| **Temporal Conditioning (Training only)** | 5 | `day_of_week`, `month_of_year`, `quarter`, `is_weekend`, `is_holiday` | Used to capture seasonality during training; marginalized out for Model A |
| **Excluded / Entity Leakage** | 8 | `PF`, `CODICEMACCHINA`, `Macchina`, `CodiceMacchinaProd`, `matricola`, etc. | High-cardinality serials/IDs that cause memorization |
| **Excluded / Post-Installation** | 6 | `DataPrimaAttivazione`, `DataInstallazione`, `days_since_activation`, etc. | Lifecycle timestamps not known pre-installation |
| **Excluded / Zero Variance** | 5 | `TipologiaPuntoFisico`, `Attributo`, `Stato`, `Installato`, `maturity_status` | Filter constants in certified population |
| **Excluded / Raw Text & Internal Keys** | 23 | `Indirizzo`, `Descrizione`, `Comodatario`, `StoricoPuntoFisico_Id`, etc. | Unstructured free text or pipeline internal surrogate keys |

---

## Stage 2 — Location Input Pipeline

### The 4 Application Inputs
The prospective location tool requires exactly four inputs from the business user:
1. **Store name** (e.g., `"Conad Superstore"`, `"Esselunga"`, `"Coop"`, `"Eurospin"`)
2. **Address** (e.g., `"Via Roma 145"`, `"Corso Buenos Aires 22"`)
3. **CAP** (e.g., `"20121"`, `"00185"`)
4. **Machine type** (e.g., `"DAINT_G"`, `"Ripet_C-front"`, `"DAINT_A-bis"`)

### Address & Geocoding Workflow
Raw address text (e.g., `"Via Roma 145"`) has high cardinality, typographic variation, and arbitrary formatting. It **must not** be passed directly into the neural network.

Instead, the production system routes the address through an automated geocoding and spatial enrichment pipeline:

```
[User Inputs: Store Name, Address, CAP, Machine Type]
                        │
                        ▼
            Geocoding Engine (OpenStreetMap / ISTAT Geocode)
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
[Latitude, Longitude]              [Validated CAP]
       │                                 │
       └────────────────┬────────────────┘
                        │
                        ▼
         Administrative Spatial Matching (ISTAT)
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
[Comune]           [Provincia]          [Regione]
    │                   │                   │
    └───────────────────┼───────────────────┘
                        ▼
           Socioeconomic Demographic Lookup
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
[Population, Surface]   [Pop. Density]    [PIL, PIL/Capita]
```

### Store Name Processing
The raw `Store name` string is parsed using fuzzy matching / entity extraction against a standardized retail brand dictionary:
* Normalizes strings (e.g., `"CONAD CITY"`, `"Conad Superstore"` $\rightarrow$ `Insegna = "CONAD"`, `Gruppo = "CONAD"`).
* If a known retail banner is detected, assign a categorical retail entity embedding.
* If unlisted or independent (e.g., `"Alimentari Rossi"`), categorize as `'INDIPENDENTE'`.

---

## Stage 3 — Location Features Evaluation

### 1. Administrative and Continuous Spatial Coordinates
* **Latitudine & Longitudine:** Present for 100% of observations in the certified population (1,649 unique coordinate pairs). Continuous WGS84 coordinates enable the neural network to capture spatial gradients across Italy.
* **Comune:** 962 unique municipalities in dataset (missing: 0.10%). Provides administrative grouping and demographic linkage.
* **Provincia:** 105 unique provinces (7.43% missing in raw, but 100% recoverable via CAP/Comune lookup).
* **Regione:** 20 Italian administrative regions (0.00% missing). Captures macro-regional recycling habits, packaging regulations, and beverage consumption patterns.
* **CAP:** 920 unique postal codes (0.00% missing). Provides localized urban clustering.

### 2. Demographic and Macroeconomic Features (ISTAT / MEF)
Demographic features were merged at the municipality level:
* **`population` (Resident Population):** Min = 467, Median = 34,147, Max = 2,749,031 (Roma). Missing = 0.43%.
* **`surface` (Land Area in $\text{km}^2$):** Min = 1.06 $\text{km}^2$, Median = 44.8 $\text{km}^2$, Max = 1,287.4 $\text{km}^2$. Missing = 1.06%.
* **`population_density` ($\text{residents}/\text{km}^2$):** Min = 10.1, Median = 673.4, Max = 11,949.0 (Portici/Napoli). Missing = 1.22%.
* **`PIL` (Gross Domestic Product in €):** Min = €7.38M, Median = €646M, Max = €49.2B (Milano). Missing = 0.38%.
* **`PIL_per_capita` (€/resident):** Min = €5,712, Median = €15,411, Max = €24,910. Missing = 0.54%.

> [!NOTE]
> **Macro Feature Scaling:** `population`, `surface`, `population_density`, and `PIL` span several orders of magnitude and follow heavy right-skewed distributions. They must be log-transformed ($\log_{10}(x)$ or $\log(1+x)$) and standardized before feeding to the neural network.

### 3. Nielsen Store Features & Missing-Data Strategy
The Nielsen dataset (`filenielsen_1203.csv`) contains high-value store operational characteristics:
* `nielsen_tipo_negozio`: Store format (`SUP` = Supermarket, `IPR` = Hypermarket, `LIS` = Superette, `DIS` = Discount).
* `nielsen_superficie_vendita`: Sales floor area in square meters.
* `nielsen_numero_casse`: Total checkout cash registers.
* `nielsen_insegna`: Standardized store banner.
* `nielsen_gruppo`: Retail alliance / buying group (e.g. Selex, Conad, Coop).
* `nielsen_parcheggio`: Dedicated customer parking lot indicator.
* `nielsen_centro_commerciale`: Enclosed shopping mall indicator.

#### Nielsen Match Audit in Certified Population
* **MATCHED:** 640,286 rows (45.85%)
* **UNMATCHED:** 584,571 rows (41.86%)
* **AMBIGUOUS:** 171,687 rows (12.29%)

#### Robust Missing-Data Strategy for Nielsen
The user mandate is explicit: **"Nielsen is OPTIONAL enrichment. DO NOT delete locations simply because Nielsen data is unavailable."**

When a prospective store is entered, Nielsen data may be missing. The system uses a **quad-part strategy**:
1. **Missingness Indicator:** Create a binary flag `is_nielsen_matched` (1 = matched, 0 = unmatched/missing).
2. **Numerical Imputation:** For `nielsen_superficie_vendita` and `nielsen_numero_casse`, impute missing values with the median of the corresponding `nielsen_tipo_negozio` (or overall population median: $\text{Superficie} = 1,150\text{ sqm}$, $\text{Casse} = 5$). Pair each with an explicit indicator `missing_superficie` and `missing_casse`.
3. **Categorical Imputation:** For `nielsen_tipo_negozio`, `nielsen_gruppo`, and `nielsen_insegna`, map missing values to an explicit `'UNKNOWN'` token, which receives its own learned embedding in the neural network.
4. **Feature Dropout During Training:** To ensure the neural network does not develop a brittle dependency on Nielsen features, randomly drop (mask to `'UNKNOWN'` / median) Nielsen features on 25% of training samples during training. The model will thus learn to make accurate predictions with OR without Nielsen data!

---

## Stage 4 — Machine Features Audit

### Machine Variable Inventory

| Column Name | Source | Unique Count | Missing | Nature | Evaluation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **`TipoMacchina`** | CaricaView | **14** | 23 | Functional Hardware Model | **RECOMMENDED APPLICATION FEATURE** |
| `Fornitore` | CaricaView | 4 | 23 | OEM Vendor (DAINT, RIPET, etc.) | Core hardware attribute; redundant with `TipoMacchina` |
| `Macchina` | CaricaView | 1,743 | 23 | Internal Asset Tag | **STRICT EXCLUSION (Entity Leakage)** |
| `CodiceMacchinaProd` | CaricaView | 1,743 | 23 | Manufacturer Serial | **STRICT EXCLUSION (Entity Leakage)** |
| `CODICEMACCHINA` | Daily Bottles | 1,743 | 0 | Asset Code | **STRICT EXCLUSION (Entity Leakage)** |
| `CODICEMACCHINAPROD`| Daily Bottles | 1,743 | 0 | Manufacturer Serial | **STRICT EXCLUSION (Entity Leakage)** |

### Cross-Tabulation: `TipoMacchina` vs. `Fornitore`

| TipoMacchina | DAINT | RIPET | EURVEN | DENTISTECH | Missing | Total Observations | Unique PFs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DAINT_G** | 564,365 | 0 | 0 | 0 | 0 | 564,365 | 924 |
| **Ripet_C-front** | 0 | 443,737 | 0 | 0 | 0 | 443,737 | 448 |
| **DAINT_A-bis** | 193,484 | 0 | 0 | 0 | 0 | 193,484 | 169 |
| **Ripet_C-Ter** | 0 | 47,564 | 0 | 0 | 0 | 47,564 | 38 |
| **Ripet_V2** | 0 | 47,322 | 0 | 0 | 0 | 47,322 | 36 |
| **Ripet_C-Bis** | 0 | 31,222 | 0 | 0 | 0 | 31,222 | 25 |
| **DAINT_B-bis** | 21,242 | 0 | 0 | 0 | 0 | 21,242 | 21 |
| **Ripet_Smarpet** | 0 | 0 | 0 | 15,907 | 0 | 15,907 | 17 |
| **RVM_Mod.E_Nuova** | 0 | 0 | 13,289 | 0 | 0 | 13,289 | 14 |
| **RVM_Mod.E** | 0 | 0 | 13,124 | 0 | 0 | 13,124 | 17 |
| **DAINT_B** | 2,678 | 0 | 0 | 0 | 0 | 2,678 | 2 |
| **Ripet_V1** | 0 | 1,502 | 0 | 0 | 0 | 1,502 | 2 |
| **SUPERMARPET01S** | 0 | 0 | 0 | 974 | 0 | 974 | 16 |
| **DAINT_Y_DRS** | 111 | 0 | 0 | 0 | 0 | 111 | 2 |
| **Missing** | 0 | 0 | 0 | 0 | 23 | 23 | 1 |
| **Total** | **781,880** | **571,347** | **26,413** | **16,881** | **23** | **1,396,544** | **1,657** |

### Operational Performance by `TipoMacchina`

| TipoMacchina | Observations | Unique PFs | Mean BOTTIGLIE | Median BOTTIGLIE | Mean PRESENZE | Median PRESENZE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ripet_Smarpet** | 15,907 | 17 | 700.4 | 554.0 | 18.8 | 18.0 |
| **Ripet_C-front** | 443,737 | 448 | 668.2 | 472.0 | 19.6 | 14.0 |
| **DAINT_G** | 564,365 | 924 | 631.1 | 432.0 | 18.8 | 14.0 |
| **Ripet_C-Ter** | 47,564 | 38 | 512.4 | 345.0 | 15.2 | 12.0 |
| **RVM_Mod.E** | 13,124 | 17 | 483.2 | 373.0 | 13.8 | 12.0 |
| **DAINT_B-bis** | 21,242 | 21 | 457.1 | 367.0 | 13.9 | 13.0 |
| **DAINT_A-bis** | 193,484 | 169 | 448.3 | 300.0 | 13.1 | 11.0 |
| **Ripet_C-Bis** | 31,222 | 25 | 395.2 | 307.0 | 10.7 | 11.0 |
| **Ripet_V2** | 47,322 | 36 | 389.4 | 265.0 | 10.5 | 8.0 |
| **RVM_Mod.E_Nuova** | 13,289 | 14 | 361.3 | 251.0 | 9.7 | 8.0 |

### Machine Type Recommendation
1. **Application Input Exposure:** The user interface must expose **`Machine type`** corresponding directly to the functional hardware category `TipoMacchina`.
2. **Serial ID Exclusion:** `Macchina`, `CodiceMacchinaProd`, `CODICEMACCHINA`, and `CODICEMACCHINAPROD` have 1,743 distinct values across 1,657 PFs. They are physical serial numbers that act as surrogate primary keys. **They must never be used as inputs.**
3. **Consolidation of Rare Types:** The top 3 types (`DAINT_G`, `Ripet_C-front`, `DAINT_A-bis`) account for **86.1%** of all installations. Rare types with $<20$ physical points can be grouped into an `'OTHER_COMPACT'` or `'OTHER_HIGH_CAPACITY'` category if necessary to stabilize training.

---

## Stage 5 — Targets Distribution Analysis

A complete distributional analysis of the certified population ($N = 1,396,544$) was conducted for both targets:

### Comprehensive Target Statistics Table

| Metric | BOTTIGLIE (Target 1) | PRESENZE (Target 2) | Notes |
| :--- | :---: | :---: | :--- |
| **Mean** | **594.01** | **17.43** | Arithmetic daily average |
| **Median (50th percentile)**| **400.00** | **13.00** | Robust daily center |
| **Standard Deviation** | **699.11** | **16.80** | Large variance |
| **Minimum** | 0.00 | 0.00 | True zero collection days |
| **1st Percentile** | 0.00 | 0.00 | Store closed or downtime |
| **5th Percentile** | 0.00 | 0.00 | Bottom 5% includes zeros |
| **25th Percentile (Q1)** | 182.00 | 7.00 | Lower quartile |
| **75th Percentile (Q3)** | 763.00 | 23.00 | Upper quartile |
| **90th Percentile** | 1,295.00 | 36.00 | Top 10% volume threshold |
| **95th Percentile** | 1,767.00 | 48.00 | Top 5% volume threshold |
| **99th Percentile** | 3,508.00 | 86.00 | Extreme high-traffic days |
| **99.9th Percentile** | 6,620.46 | 136.00 | Peak event threshold |
| **Maximum** | 11,496.00 | 234.00 | Maximum recorded daily volume |
| **Zero Count** | 77,456 | 76,241 | Exactly ~5.5% of records |
| **Zero Percentage** | **5.55%** | **5.46%** | Driven by store closures & downtime |
| **Raw Skewness** | **+3.72** | **+2.64** | Heavy positive skew |
| **Raw Kurtosis** | **22.84** | **11.38** | Extreme fat tails |
| **$\log(1+y)$ Mean** | 5.63 | 2.53 | Stabilized mean |
| **$\log(1+y)$ Std** | 1.72 | 0.97 | Stabilized variance |
| **$\log(1+y)$ Skewness** | -1.91 | -0.75 | Skew driven only by zero spike |
| **$\log(1+y)$ Skewness (when $y > 0$)** | **-0.94** | **-0.43** | **Near-normal, symmetric bell curve** |

### Target Relationships
* **Pearson Correlation ($r$):** **0.8923** (Strong positive linear correlation).
* **Spearman Rank Correlation ($\rho$):** **0.9149** (Very strong monotonic relationship).
* **Bottles per Presence Ratio ($\text{BOTTIGLIE} / \text{PRESENZE}$ when $>0$):**
  * Median: **30.5 bottles/visit**
  * Mean: **33.0 bottles/visit**
  * 90th percentile: **51.5 bottles/visit**

### Zero-Value Analysis
Audit of the 77,456 zero-bottle days reveals clear operational causes:
* **Sunday Store Closures:** 28.4% of all zero days occur on Sundays (10.98% zero rate on Sundays vs. 3.88% on Fridays and 4.27% on Saturdays).
* **National Holidays:** Severe zero spikes on Easter, Easter Monday, Christmas (25 Dec), Boxing Day (26 Dec), New Year's Day (1 Jan), and Ferragosto (15 Aug).
* **Temporary Maintenance:** Remaining zeros reflect temporary bin-full states, connectivity dropouts, or machine servicing.

### Target Transformation Recommendation: **B. $\log(1 + \text{Target})$**
> [!TIP]
> **Recommendation: Train the neural network on $\log(1 + y)$ for both targets.**
> 
> * **Justification:**
>   1. **Variance Stabilization:** Raw targets exhibit standard deviations exceeding their means ($\text{Std}=699$ vs $\text{Mean}=594$). Untransformed MSE loss would be overwhelmingly dominated by the top 1% outliers (e.g. days with 8,000+ bottles), leading to unstable gradients and poor predictions for typical stores.
>   2. **Distributional Symmetry:** For non-zero days, $\log(1+y)$ reduces `PRESENZE` skewness from $+2.64$ to **$-0.43$** and kurtosis from $11.38$ to **$0.39$**, producing an almost perfect Gaussian distribution.
>   3. **Non-Negativity Guarantee:** Natural exponentiation at inference time $\hat{y} = \exp(\hat{z}) - 1$ guarantees strictly non-negative predictions ($\ge 0$).
>   4. **Multiplicative Error Scale:** In retail foot traffic, predicting $\pm 20\%$ relative error is more appropriate than constant absolute error. Log loss naturally optimizes relative percentage error.

---

## Stage 6 — Time Features & Model Strategy

### Analysis of Temporal Dynamics in Training Data
Daily collection performance is not stationary; it varies strongly across the calendar:
1. **Day-of-Week Effect:**
   * Saturday is the peak volume day: Mean = **716.3 bottles**, **21.0 presenze**.
   * Friday: Mean = **599.5 bottles**, **18.2 presenze**.
   * Sunday: Mean = **519.5 bottles**, **14.1 presenze** (due to partial store closures).
2. **Seasonal / Month Effect:**
   * Summer Peak (July): Mean = **720.4 bottles**, **19.2 presenze** (hot weather drives beverage consumption).
   * Winter Trough (December): Mean = **502.1 bottles**, **15.4 presenze**.
3. **Annual Growth Trend:**
   * 2023 Mean = 554.3 bottles
   * 2024 Mean = 604.4 bottles
   * 2025 Mean = 577.3 bottles
   * 2026 Mean = 625.4 bottles

### Model A vs. Model B: Business Resolution
* **MODEL A (Steady-State Expected Daily Potential):** Predicts the canonical, typical daily and monthly performance of a new prospective retail location.
* **MODEL B (Calendar-Specific Date Prediction):** Predicts what a store will do on a specific historical or future date (e.g. Saturday, 18 July 2027).

### Coripet Application Recommendation: **Model A with Seasonal Marginalization**
For Coripet's business objective, capital allocation and machine deployment depend on **Model A** (steady-state potential score and expected monthly volumes). Decision makers do not evaluate an RVM investment based on a specific Tuesday in November.

However, naive Model A training (collapsing or ignoring daily temporal variables) causes severe confounding: a store observed mostly in July would appear inherently superior to a store observed mostly in February.

#### The Recommended Dual Approach:
1. **During Training (Conditioned Learning):** Include temporal features (`day_of_week`, `month_of_year`, `is_weekend`, `is_holiday`) as inputs so the neural network separates temporal variation from intrinsic location quality.
2. **During Inference for a New Location (Marginalized Expectation):**
   * Feed the prospective store features into the network across a representative 7-day week (Monday through Sunday) and average the outputs:
     $$\mathbb{E}[\text{Daily PRESENZE}] = \frac{1}{7} \sum_{d=0}^{6} \hat{y}_{\text{presenze}}(\mathbf{x}_{\text{store}}, d)$$
     $$\mathbb{E}[\text{Daily BOTTIGLIE}] = \frac{1}{7} \sum_{d=0}^{6} \hat{y}_{\text{bottiglie}}(\mathbf{x}_{\text{store}}, d)$$
   * Derive monthly volume:
     $$\mathbb{E}[\text{Monthly BOTTIGLIE}] = \mathbb{E}[\text{Daily BOTTIGLIE}] \times 30.4$$
   * This provides a pure, unconfounded **steady-state baseline** for the business user.

---

## Stage 7 — Leakage Audit

A strict leakage audit was conducted. The following features are **permanently excluded** from model inputs:

```
[STRICTLY EXCLUDED FROM NEW-LOCATION PREDICTORS]
├── DIRECT TARGETS & DERIVATIVES (Target Leakage)
│   ├── BOTTIGLIE, PRESENZE
│   ├── Historical averages (mean bottles/day, mean presenze/day)
│   ├── Previous-day lags (bottles_lag1, presenze_lag1)
│   ├── Rolling statistics (7-day, 30-day moving averages)
│   └── Cumulative historical totals
├── ENTITY IDENTIFIERS (Memorization Leakage)
│   ├── PF, PuntoFisico (Physical point ID)
│   ├── CODICEMACCHINA, CODICEMACCHINAPROD (Machine asset codes)
│   ├── Macchina, CodiceMacchinaProd, matricola (Hardware serials)
│   └── nielsen_country_shop_id (Nielsen store primary key)
├── POST-INSTALLATION LIFECYCLE TIMESTAMPS (Temporal Leakage)
│   ├── DataPrimaAttivazione (First turn-on date)
│   ├── DataInstallazione (Installation date)
│   ├── DataRimozione (Decommission date)
│   ├── days_since_activation, months_since_activation (Age of machine)
│   └── maturity_date (Ramp-up anchor date)
└── CONSTANT POPULATION GATES (Zero Variance)
    ├── TipologiaPuntoFisico (= PUNTOVENDITA)
    ├── Attributo (= STANDARD)
    ├── Stato (= Attivo), Installato (= True)
    └── maturity_status (= MATURE), modeling_eligible (= 1)
```

---

## Stage 8 — Validation Strategy for Repeated Daily Observations

### The Repeated-Observation Problem
The modeling population has **1,396,544 rows across 1,657 unique PFs** (an average of **842 daily observations per physical point**).
* **Random Split Failure:** If rows are randomly partitioned into train and test sets, observations from the *exact same store* will be present in both sets. The neural network would simply memorize store coordinates (`Latitudine`, `Longitudine`) and postal codes (`Cap`), producing deceptively stellar test metrics ($R^2 > 0.95$) that collapse completely when presented with a brand new, unseen physical store.

### Recommended Validation Architecture: **Group-Separated by `PF` + Temporal Holdout**

```
ALL 1,657 PHYSICAL POINTS (1,396,544 rows)
                      │
    ┌─────────────────┴─────────────────┐
    ▼                                   ▼
TRAIN + VALIDATION PFs (80%)      TEST PFs (20% Holdout)
~1,325 Unique PFs                 ~332 Unique PFs
~1,117,000 observations            ~279,000 observations
(NEVER seen by test)               (Completely unseen stores)
    │                                   │
    ▼                                   ▼
5-Fold GroupKFold                   Final Out-of-Store
Grouped strictly by PF              Generalization Benchmark
+ 2026 Temporal Validation          (Evaluates Model A & B)
```

1. **Unseen Physical Point Holdout (Spatial Test Set):**
   * Set aside **20% of unique physical points (~332 PFs)** as a pure holdout test set. No daily observation from these 332 PFs will ever enter training or hyperparameter tuning.
   * This directly evaluates: *"How well does the model predict an RVM installed at a brand new store?"*
2. **K-Fold Group Cross-Validation:**
   * Within the 80% development set, use **5-Fold GroupKFold** grouped strictly by `PF`.
3. **Temporal Holdout Benchmark:**
   * To evaluate temporal robustness and macro stability, evaluate performance across the final chronological segment (2026 data) for both seen and unseen physical points.

---

## Stage 9 — Proposed Final Feature Set

### 1. CORE FEATURES (Always Required & Derived from 4 User Inputs)

| Feature Name | Source | Meaning & Utility | Availability for New Location | Missing-Data Strategy |
| :--- | :--- | :--- | :---: | :--- |
| **`Latitudine`** | Geocoding API | Continuous spatial coordinate; latitude in decimal degrees | YES | Imputed via CAP/Comune centroid if address fails |
| **`Longitudine`** | Geocoding API | Continuous spatial coordinate; longitude in decimal degrees | YES | Imputed via CAP/Comune centroid if address fails |
| **`Cap`** | User Input | Italian 5-digit postal code; localized urban area | YES | Mandatory user input |
| **`Comune`** | Geocoding / ISTAT | Municipality name; administrative category | YES | High-frequency entity embedding; 'UNKNOWN' token |
| **`Provincia`** | Geocoding / ISTAT | Two-letter province code (105 classes); regional grouping | YES | Categorical embedding; resolved via CAP |
| **`Regione`** | Geocoding / ISTAT | 20 Administrative Regions; macro recycling behavior | YES | Categorical embedding; resolved via CAP |
| **`population`** | ISTAT Census | Total municipality resident population; market scale | YES | Log-transformed; median imputation by province |
| **`surface`** | ISTAT Census | Municipality land area in $\text{km}^2$; physical extent | YES | Log-transformed; median imputation by province |
| **`population_density`** | Derived | Inhabitants per $\text{km}^2$; urban vs rural typology | YES | Log-transformed; median imputation by province |
| **`PIL`** | MEF / ISTAT | Aggregate Gross Domestic Product; market wealth | YES | Log-transformed; median imputation by province |
| **`PIL_per_capita`** | Derived | GDP per resident; purchasing power proxy | YES | Continuous standardized; median imputation |
| **`TipoMacchina`** | User Input | Hardware model family (DAINT_G, Ripet_C-front, etc.) | YES | Mandatory user selection (categorical embedding) |
| **`Insegna_Cleaned`** | Store Name Input | Standardized retail banner (Conad, Coop, Esselunga, etc.) | YES | Fuzzy brand match; 'INDIPENDENTE' if unmatched |

### 2. OPTIONAL ENRICHMENT FEATURES (Nielsen Store Data)

| Feature Name | Source | Meaning & Utility | Availability for New Location | Missing-Data Strategy |
| :--- | :--- | :--- | :---: | :--- |
| **`is_nielsen_matched`**| Pipeline | Binary indicator (1=Matched, 0=Missing) | YES | Explicit indicator column |
| **`nielsen_tipo_negozio`**| Nielsen Census | Store format: `SUP`, `IPR`, `LIS`, `DIS` | YES (Optional) | Categorical: mapped to `'UNKNOWN'` token |
| **`nielsen_superficie_vendita`**| Nielsen Census | Sales area ($\text{m}^2$); physical capacity | YES (Optional) | Impute median ($1,150\text{ m}^2$) + missing flag |
| **`nielsen_numero_casse`**| Nielsen Census | Checkout cash register count; foot traffic proxy | YES (Optional) | Impute median ($5\text{ checkouts}$) + missing flag |
| **`nielsen_insegna`** | Nielsen Census | Verified retail chain banner | YES (Optional) | Categorical: mapped to `'UNKNOWN'` token |
| **`nielsen_gruppo`** | Nielsen Census | Retail alliance (Selex, Coop, Conad, etc.) | YES (Optional) | Categorical: mapped to `'UNKNOWN'` token |
| **`nielsen_area`** | Nielsen Census | Nielsen macro-geographic zone (1, 2, 3, 4) | YES (Optional) | Categorical: mapped to `'UNKNOWN'` token |
| **`nielsen_parcheggio`**| Nielsen Census | Dedicated parking availability flag | YES (Optional) | Binary: impute mode (0) + missing flag |
| **`nielsen_centro_commerciale`**| Nielsen Census | Shopping mall location flag | YES (Optional) | Binary: impute mode (0) + missing flag |

### 3. EXCLUDED FEATURES (Leakage & Invariant Audit)

| Feature Name | Source | Exclusion Category | Reason for Strict Exclusion |
| :--- | :--- | :--- | :--- |
| `BOTTIGLIE`, `PRESENZE` | Daily Bottles | Target Leakage | Outcome variables; used only as labels during training |
| `PF`, `PuntoFisico` | Base Tables | Entity Leakage | High-cardinality primary keys; induces memorization |
| `CODICEMACCHINA`, `Macchina` | Operational | Entity Leakage | Machine hardware asset tags; induces memorization |
| `CodiceMacchinaProd`, `matricola`| Operational | Entity Leakage | Hardware serial numbers; induces memorization |
| `DataPrimaAttivazione` | Historical | Lifecycle Leakage | Operational ramp-up date; unknown for prospective store |
| `DataInstallazione`, `DataRimozione`| Historical | Lifecycle Leakage | Post-installation operational dates |
| `days_since_activation`, `months_`| Historical | Lifecycle Leakage | Machine operational age; unknown for prospective store |
| `TipologiaPuntoFisico` | Historical | Zero Variance | 100% PUNTOVENDITA (constant across modeling population) |
| `Attributo` | Historical | Zero Variance | 100% STANDARD (constant across modeling population) |
| `Stato`, `Installato` | Historical | Zero Variance | 100% Attivo / True (constant across modeling population) |
| `Indirizzo` (raw string) | Historical | Unstructured Noise | Unstandardized free text; replaced by geocoded coordinates |

---

## Stage 10 — Final Dataset Design & Neural Network Architecture

### Recommended Multi-Output Neural Network Architecture

The prediction system is designed as a **Multi-Task Multi-Output Deep Neural Network** with a shared representation backbone and dual task heads:

```
                          ┌───────────────────────────┐
                          │    PREPROCESSED INPUTS    │
                          │ Numerical (Dense Scaled)  │
                          │ Categorical (Embeddings)  │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │   SHARED TRUNK (Dense)    │
                          │ Layer 1: Dense 128 + GELU │
                          │ LayerNorm + Dropout(0.2)  │
                          │ Layer 2: Dense 64 + GELU  │
                          │ LayerNorm + Dropout(0.2)  │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │    HEAD 1: PRESENZE       │                 │    HEAD 2: BOTTIGLIE      │
   │ Dense 32 + GELU           │                 │ Dense 32 + GELU           │
   │ Output: Linear            │                 │ Output: Linear            │
   │ (Predicts log1p(PRESENZE))│                 │ (Predicts log1p(BOTTIGLIE))│
   └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                 │                                             │
                 ▼                                             ▼
        $\hat{y}_1 = \log(1 + P)$                     $\hat{y}_2 = \log(1 + B)$
                 │                                             │
                 ▼                                             ▼
       $P = \exp(\hat{y}_1) - 1$                     $B = \exp(\hat{y}_2) - 1$
```

#### Shared Trunk Inductive Bias
* `PRESENZE` and `BOTTIGLIE` share an underlying latent cause: **store customer foot traffic and green consumer recycling affinity** (correlation $r = 0.892$).
* Learning a joint representation forces the network to discover features that govern store foot traffic generally, reducing overfitting on either target alone.
* Optionally, Head 2 (`BOTTIGLIE`) can accept the latent embedding from Head 1 (`PRESENZE`) as an auxiliary input, embedding the physical constraint that:
  $$\text{BOTTIGLIE} = \text{PRESENZE} \times \text{Bottles per Session}$$

#### Multi-Task Loss Function
The model trains using a combined Huber loss on the log-transformed targets:
$$\mathcal{L}_{\text{total}} = w_1 \cdot \text{Huber}_{\delta}(\hat{y}_{\text{presenze}}, y_{\text{presenze}}) + w_2 \cdot \text{Huber}_{\delta}(\hat{y}_{\text{bottiglie}}, y_{\text{bottiglie}})$$
*(Huber loss is robust against residual collection anomalies while maintaining quadratic penalization near the mean).*

### Location Potential Score & Recommendation Logic
Using the marginalized daily expectations $\hat{P}$ (Presenze) and $\hat{B}$ (Bottles):

1. **Location Potential Score (%):**
   * Normalized percentile of $\hat{B}$ relative to the empirical cumulative distribution of the certified active network:
     $$\text{Potential Score} = \hat{F}_{B}(\hat{B}) \times 100\%$$
   * A score of $85\%$ means the predicted volume exceeds $85\%$ of all active RVM installations in Italy.
2. **Recommendation Tiers:**
   * **HIGH POTENTIAL:** Potential Score $\ge 75\%$ ($\hat{B} \ge 763\text{ bottles/day}$) $\rightarrow$ Prioritize installation.
   * **MEDIUM POTENTIAL:** $40\% \le \text{Potential Score} < 75\%$ ($300 \le \hat{B} < 763\text{ bottles/day}$) $\rightarrow$ Viable installation.
   * **LOW POTENTIAL:** Potential Score $< 40\%$ ($\hat{B} < 300\text{ bottles/day}$) $\rightarrow$ Unfavorable ROI; explore alternative machine type or location.
3. **Prediction Confidence:**
   * Calculated from:
     * Geocoding accuracy (Exact address vs CAP centroid).
     * Nielsen enrichment status (1 if matched, 0.7 if missing).
     * Municipality demographic match confidence.
     * High / Medium / Low badge displayed to user.

---

## Data Problems Discovered & Decisions Requiring Approval

### 1. Data Problems Discovered
1. **Machine Identifiers Overfitting Risk:** `Macchina`, `CodiceMacchinaProd`, `CODICEMACCHINA`, and `CODICEMACCHINAPROD` are high-cardinality serial identifiers (1,743 distinct values). If included, any neural net would overfit by memorizing specific historical installations.
2. **Provincia Missingness (7.43%):** 103,776 rows lack `Provincia` in the historical table, but 100% have valid `Cap` and `Comune`. An automated CAP-to-Provincia lookup solves this completely.
3. **Nielsen Unmatched Stores (41.86%):** A significant share of operational stores are not present in the Nielsen census file. Dropping them would discard valuable training signal and break the app for non-Nielsen stores. The proposed missingness-indicator + median/unknown imputation + feature dropout strategy resolves this cleanly.
4. **Zero Inflation (5.55%):** Driven primarily by Sunday store closures (10.98% on Sundays) and national holidays. Modeling on $\log(1+y)$ handles zeros cleanly, but calendar conditioning during training is essential to avoid underestimating weekday volume.

### 2. Decisions Requiring Approval Before Modeling
1. **Model A vs. Model B Application Output:** Confirm that the production application should output **Model A (Steady-state typical daily expectation marginalized across the week)** for evaluating prospective store ROI.
2. **Machine Type Selection:** Confirm that the user UI will present the 14 standard hardware categories (`TipoMacchina`), with `DAINT_G`, `Ripet_C-front`, and `DAINT_A-bis` as primary options.
3. **Nielsen Enrichment Feature Dropout:** Confirm the feature-dropout strategy (25% masking during training) to ensure robust performance when Nielsen data is absent.
4. **Validation Split Protocol:** Confirm the 5-fold GroupKFold by `PF` with a 20% holdout test set of completely unseen physical points.

---
*The feature inventory and engineering design are complete. No models have been trained. Awaiting user approval to proceed.*
