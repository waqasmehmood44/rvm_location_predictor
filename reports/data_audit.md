# Comprehensive Data Audit Report: Coripet RVM Platform

**Execution Timestamp:** 2026-09-15  
**Data Directory:** `data/raw/`  
**Datasets Audited:** 6 files  
**Status:** Audit Completed | No ML Models Trained | Raw Files Preserved Verbatim

---

## Executive Summary

A comprehensive, non-destructive audit was performed across all six primary datasets in `data/raw/`. The operational log encompasses **1,889,479 daily records** spanning from **January 1, 2023 through August 31, 2026**, covering **2,142 unique physical installation sites (Punti Fisici)** and **2,186 unique RVM machines**. 

The master store repository `CaricaViewStoricoPuntoFisico` contains **5,632 historical version records** for **2,539 unique physical sites**, tracking machine replacements, hardware reallocations, and status evolutions via explicit effective date intervals (`DataInizioRange` to `DataFineRange`).

### Dataset Overview Matrix

| # | Dataset Name | File Format | Delimiter | Row Count | Column Count | Duplicate Rows | Candidate Primary Key |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `daily_bottles` | CSV | `,` | 1,889,479 | 8 | 0 | `(PF, GIORNO, CODICEMACCHINA)` |
| 2 | `CaricaViewStoricoPuntoFisico` | CSV | `;` | 5,632 | 30 | 0 | `(PuntoFisico, Macchina, DataInizioRange)` |
| 3 | `filenielsen_1203` | CSV | `;` | 25,104 | 60 | 0 | `COUNTRY SHOP ID` |
| 4 | `PIL` | CSV | `;` | 7,903 | 2 | 0 | `Comune` (nearly unique, 6 homonyms) |
| 5 | `population` | CSV | `;` | 7,901 | 2 | 0 | `Comune` (nearly unique, 5 homonyms) |
| 6 | `surface` | CSV | `;` | 8,094 | 2 | 0 | `Comune` (nearly unique, 9 homonyms) |

---

## 1. Dataset Audit: `daily_bottles.csv`

### 1.1 Basic Metrics
* **Total Rows:** 1,889,479
* **Total Columns:** 8
* **Full Duplicate Rows:** 0 (0.00%)
* **Candidate Primary Key:** Composite key `(PF, GIORNO, CODICEMACCHINA)` uniquely identifies 1,889,223 records (with 256 rare multi-machine timestamp collisions at boundary transitions).

### 1.2 Schema & Data Types
| Column Name | Inferred Dtype | Missing Count | Missing Pct | Description / Semantic Role |
| :--- | :---: | :---: | :---: | :--- |
| `CLIENTE` | `object` | 0 | 0.00% | Retail client banner or municipal partner (e.g. CONAD, Decò, Esselunga) |
| `GIORNO` | `object` (datetime) | 0 | 0.00% | Operational timestamp (`DD/MM/YYYY 00:00:00`) |
| `BOTTIGLIE` | `int64` | 0 | 0.00% | Daily target: Total PET bottles collected by the machine |
| `PRESENZE` | `int64` | 0 | 0.00% | Daily target: Number of unique individuals using the machine |
| `PF` | `object` | 0 | 0.00% | Physical installation site identifier (*Punto Fisico*) |
| `CODICEMACCHINAPROD` | `object` | 0 | 0.00% | Production machine hardware code (e.g. `Ripet0126`, `13154`) |
| `CODICEMACCHINA` | `object` | 0 | 0.00% | RVM logical asset code (e.g. `MC7O0Z2U4RQ7`, `MC00IBSMT523`) |
| `REFCORIPET` | `object` | 7 | 0.00% | Assigned Coripet account / regional manager |

### 1.3 Date Range
### 1.3 Verified Date Range & Temporal Breakdown
* **Date Column:** `GIORNO` (parsed format: `DD/MM/YYYY HH:MM:SS`)
* **First Date:** `01/01/2023 00:00:00` (January 1, 2023)
* **Last Date:** `31/08/2026 00:00:00` (August 31, 2026)
* **Total Unique Days:** 1,339 calendar days
* **Missing / Null Dates:** 0 (100.0% valid non-null timestamps)

> [!IMPORTANT]
> **Timeline Confirmation:** The operational dataset explicitly covers operations through **August 31, 2026** (it does **NOT** truncate at 31/12/2025). The year 2026 alone contains **437,655 records** across 8 full operational months.

#### Annual Record Distribution
| Year | Period | Record Count | Percentage | Unique PFs Active |
| :---: | :--- | :---: | :---: | :---: |
| **2023** | 01/01/2023 – 31/12/2023 | 343,633 | 18.19% | 1,220 |
| **2024** | 01/01/2024 – 31/12/2024 | 491,668 | 26.02% | 1,586 |
| **2025** | 01/01/2025 – 31/12/2025 | 616,523 | 32.63% | 1,935 |
| **2026** | 01/01/2026 – 31/08/2026 | 437,655 | 23.16% | 2,139 |
| **Total**| **01/01/2023 – 31/08/2026** | **1,889,479** | **100.00%** | **2,142** |

#### 2026 Monthly Breakdown (Jan – Aug 2026)
| Month (2026) | Period | Record Count | Unique Days |
| :--- | :--- | :---: | :---: |
| **January 2026** | 01/01/2026 – 31/01/2026 | 55,100 | 31 |
| **February 2026** | 01/02/2026 – 28/02/2026 | 49,950 | 28 |
| **March 2026** | 01/03/2026 – 31/03/2026 | 55,677 | 31 |
| **April 2026** | 01/04/2026 – 30/04/2026 | 54,165 | 30 |
| **May 2026** | 01/05/2026 – 31/05/2026 | 55,818 | 31 |
| **June 2026** | 01/06/2026 – 30/06/2026 | 54,096 | 30 |
| **July 2026** | 01/07/2026 – 31/07/2026 | 56,237 | 31 |
| **August 2026** | 01/08/2026 – 31/08/2026 | 56,612 | 31 |
| **Total 2026** | **01/01/2026 – 31/08/2026** | **437,655** | **243** |

### 1.4 Target Variables & Numerical Distribution
| Variable | Min | 25% | Median (50%) | Mean | 75% | 90% | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BOTTIGLIE`** | 0.0 | 97.0 | 319.0 | 506.27 | 711.0 | 1,241.0 | 12,867.0 | 586.67 |
| **`PRESENZE`** | 0.0 | 4.0 | 11.0 | 15.35 | 22.0 | 36.0 | 488.0 | 15.65 |
| **Derived: Bottles/Person** | 0.0 | 16.5 | 27.8 | 29.82 | 41.2 | 55.0 | >100.0 (anomalies) | 18.42 |

> [!NOTE]
> 12.8% of daily records show 0 bottles and 0 presenze, corresponding to store closure days (Sundays/holidays), maintenance downtime, or new machines awaiting operational activation.

### 1.5 Important Categorical Distributions
* **`PF`:** 2,142 unique physical locations.
* **`CODICEMACCHINA`:** 2,186 unique machines.
* **`CLIENTE`:** 499 unique client brands. Top 5:
  1. `CONAD`: 264,057 records (14.0%)
  2. `Decò`: 183,788 records (9.7%)
  3. `Penny Market`: 138,412 records (7.3%)
  4. `ESSELUNGA`: 133,083 records (7.0%)
  5. `Coop`: 98,340 records (5.2%)
* **`REFCORIPET`:** 23 account managers. Top 3: `Albano` (24.1%), `Graldi` (18.6%), `Ivaldi` (17.4%).

---

## 2. Dataset Audit: `CaricaViewStoricoPuntoFisico.csv`

### 2.1 Basic Metrics
* **Total Rows:** 5,632
* **Total Columns:** 30
* **Full Duplicate Rows:** 0 (0.00%)
* **Unique Physical Points (`PuntoFisico`):** 2,539
* **Candidate Composite Key:** `(PuntoFisico, Macchina, DataInizioRange)` uniquely identifies each historical configuration.

### 2.2 Effective Date Intervals
* `DataInizioRange`: Valid from `2019-05-01` to `2026-09-11` (100% complete)
* `DataFineRange`: Valid from `2019-07-21` to `2026-09-11` (100% complete)
* `DataInstallazione`: Valid from `2019-05-01` to `2026-09-11` (61.22% populated)
* `DataPrimaAttivazione`: Valid from `2019-05-01` to `2026-09-11` (90.68% populated)

### 2.3 Schema & Completeness
| Column Name | Inferred Dtype | Missing Count | Missing Pct | Semantic Role |
| :--- | :---: | :---: | :---: | :--- |
| `PuntoFisico` | `object` | 0 | 0.00% | Primary store location key (links to `daily_bottles.PF`) |
| `DataInizioRange` | `object` (date) | 0 | 0.00% | Effective interval start date |
| `DataFineRange` | `object` (date) | 0 | 0.00% | Effective interval end date |
| `TipologiaPuntoFisico` | `object` | 0 | 0.00% | Site classification: `PUNTOVENDITA` (92.1%), `COMUNE` (7.6%), `PRIVATO` (0.3%) |
| `NascondiAPP` | `bool` | 0 | 0.00% | Visibility flag for Coripet mobile app |
| `Indirizzo` | `object` | 1 | 0.02% | Physical street address |
| `Cap` | `int64` / `float64` | 0 | 0.00% | Italian Postal Code (needs 5-digit zero padding) |
| `Comune` | `object` | 4 | 0.07% | Municipality (1,265 unique) |
| `Provincia` | `object` | 328 | 5.82% | 2-letter Italian province code |
| `Regione` | `object` | 0 | 0.00% | Italian administrative region (20 regions) |
| `Nazione` | `object` | 0 | 0.00% | Country (`Italia`) |
| `Latitudine` | `float64` | 103 | 1.83% | Latitude coordinate |
| `Longitudine` | `float64` | 103 | 1.83% | Longitude coordinate |
| `Stato` | `object` | 8 | 0.14% | Operational status: `ATTVO`, `SOSPESO`, `DISMESSO`, `PROGRAMMATO` |
| `Attributo` | `object` | 8 | 0.14% | Installation attribute: `STANDARD`, `SPECIAL` |
| `Cooperativa` | `object` | 298 | 5.29% | Commercial cooperative cluster |
| `Filiera` | `object` | 298 | 5.29% | Retail banner / chain |
| `Descrizione` | `object` | 298 | 5.29% | Store name description |
| `TipoContenitore` | `object` | 2,202 | 39.10% | Collection container type (`Sacco`, `Cestone`) |
| `ReferenteCommerciale`| `object` | 5,072 | 90.06% | Commercial sales lead |
| `ReferenteCoripet` | `object` | 62 | 1.10% | Coripet account manager |
| `Comodatario` | `object` | 891 | 15.82% | Legal entity holding the bailment contract |
| `NomeFiliera` | `object` | 342 | 6.07% | Retail chain formal name |
| `Raggruppamento` | `object` | 636 | 11.29% | Retail holding group |
| `Macchina` | `object` | 2,184 | 38.78% | Hardware asset identifier |
| `CodiceMacchinaProd` | `object` | 2,184 | 38.78% | Production machine serial code |
| `TipoMacchina` | `object` | 2,184 | 38.78% | Machine model (e.g. `SUPERMARPET01S`, `Eurven Greeny`, `Aneco`) |
| `Fornitore` | `object` | 2,184 | 38.78% | Hardware manufacturer (`EURVEN`, `DENTISTECH`, etc.) |
| `DataInstallazione` | `object` (date) | 2,184 | 38.78% | Physical machine installation date |
| `DataPrimaAttivazione`| `object` (date) | 525 | 9.32% | Initial site activation date |

> [!IMPORTANT]
> Notice that `Macchina`, `CodiceMacchinaProd`, `TipoMacchina`, `Fornitore`, and `DataInstallazione` are missing in 2,184 rows (38.78%). These records represent physical points in `PROGRAMMATO` or `SOSPESO` status, or points prior to physical hardware assignment.

---

## 3. Dataset Audit: `filenielsen_1203.csv`

### 3.1 Basic Metrics
* **Total Rows:** 25,104
* **Total Columns:** 60
* **Full Duplicate Rows:** 0 (0.00%)
* **Candidate Primary Key:** `COUNTRY SHOP ID` (strictly unique across all 25,104 stores)
* **Nature of File:** Complete Nielsen retail universe of supermarkets, hypermarkets, and discount grocery stores across Italy.

### 3.2 Key Predictor Variables & Operational Dimensions
| Column Name | Type | Missing Pct | Summary Statistics / Values |
| :--- | :---: | :---: | :--- |
| `COUNTRY SHOP ID` | `int64` | 0.00% | Unique Nielsen store ID (`1` to `2,002,715`) |
| `TIPO NEGOZIO` | `object` | 0.00% | Store format: `LIS` (Free-service 100-399 sqm), `SUP` (Supermarket 400-2499 sqm), `DIS` (Discount), `IPP` (Hypermarket >2500 sqm) |
| `GRUPPO` | `object` | 0.00% | Retail group (25 groups: `CONAD`, `SELEX`, `COOP`, `ESSELUNGA`, `CARREFOUR`, `CRAI`, etc.) |
| `INSEGNA` | `object` | 0.00% | Commercial brand banner (343 banners: `EUROSPIN`, `CONAD`, `LIDL`, `MD`, etc.) |
| `COMUNE` | `object` | 0.00% | Municipality (5,086 unique across Italy) |
| `PROVINCIA` | `object` | 4.78% | Province code (107 provinces) |
| `CAP` | `int64` | 0.01% | Postal code (3,901 unique) |
| `METRI QUADRI SUPERFICIE VENDITA` | `int64` | 0.00% | Sales surface area in sqm: Min=100, Median=450, Mean=721.5, Max=13,500 |
| `NUMERO CASSE TOTALI` | `int64` | 0.00% | Total checkouts / cash registers: Min=1, Median=3, Mean=4.1, Max=57 |
| `AREA NIELSEN` | `int64` | 0.00% | Italian Nielsen macro-area: 1 (North-West), 2 (North-East), 3 (Center), 4 (South/Islands) |
| `PRESENZA POSTEGGIO` | `float64` | 43.41% | Binary flag (1 = customer parking lot available) |
| `PUNTO VENDITA IN CENTRO COMMERCIALE` | `float64` | 95.56% | Binary flag (1 = located inside shopping mall) |
| `BEVANDE` | `object` (float) | 0.00% | Turnover/volume index for Beverages category (comma decimal) |
| `TOT` | `object` (float) | 0.00% | Total store turnover/volume index (comma decimal) |

---

## 4. Dataset Audit: Macroeconomic & Demographic Datasets

### 4.1 `PIL.csv` (Municipal Aggregate Taxable Income / Gross Product)
* **Total Rows:** 7,903
* **Total Columns:** 2 (`Comune`, `PIL`)
* **Duplicate Rows:** 0
* **Unique Municipalities:** 7,896 (6 homonyms across different provinces)
* **Missing Values:** 1 row (0.01%)
* **Numerical Statistics (`PIL` in Euros):**
  - Minimum: €382,921 (tiny alpine hamlet)
  - 25th Percentile: €11,920,450
  - Median (50%): €29,985,877
  - Mean: €105,481,145
  - 75th Percentile: €78,540,120
  - Maximum: €49,195,511,468 (Rome metropolitan municipality)

### 4.2 `population.csv` (Municipal Resident Population)
* **Total Rows:** 7,901
* **Total Columns:** 2 (`Comune`, `Popolazione`)
* **Duplicate Rows:** 0
* **Unique Municipalities:** 7,896
* **Missing Values:** 0 (100% complete)
* **Numerical Statistics (`Popolazione`):**
  - Minimum: 31 residents
  - 25th Percentile: 1,020 residents
  - Median (50%): 2,406 residents
  - Mean: 7,471 residents
  - 75th Percentile: 6,180 residents
  - Maximum: 2,749,031 residents (Rome)

### 4.3 `surface.csv` (Municipal Land Surface Area)
* **Total Rows:** 8,094
* **Total Columns:** 2 (`Comune`, `Superficie`)
* **Duplicate Rows:** 0
* **Missing Values:** 2 in `Comune`, 1 in `Superficie`
* **Data Format:** `Superficie` is recorded as string with comma decimal (e.g. `11,77` representing $11.77\text{ km}^2$).
* **Numerical Statistics (parsed as float $km^2$):**
  - Minimum: $0.12\text{ km}^2$ (Atrani)
  - Median: $21.4\text{ km}^2$
  - Mean: $37.2\text{ km}^2$
  - Maximum: $1,287.4\text{ km}^2$ (Rome)

---

## 5. Summary Findings & Quality Checks

1. **No Data Corruption:** All six files are structurally intact with consistent headers and delimiters.
2. **Zero Nulls in Core Targets:** `daily_bottles` has 0 missing values in `GIORNO`, `PRESENZE`, `BOTTIGLIE`, `PF`, `CODICEMACCHINA`.
3. **100% Coverage on Operational Stores:** Every single physical point in `daily_bottles` is present in `CaricaViewStoricoPuntoFisico`.
4. **Clean Dates:** Date fields in both operational logs and historical master records parse consistently using Italian day-first format (`DD/MM/YYYY`).
