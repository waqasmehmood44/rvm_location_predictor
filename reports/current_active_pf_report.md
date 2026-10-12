# Current Active Physical Points & Reconciliation Report: 31 August 2026

**Execution Date:** 2026-09-15  
**Strict Reference Date:** **31 August 2026 (`2026-08-31`)**  
**Data Sources:** `data/raw/CaricaViewStoricoPuntoFisico.csv` (versioned physical-point registry) & `data/raw/daily_bottles.csv` (operational logs)  
**Status:** Certified & Frozen  

---

## 1. Context & Business Problem

A critical business requirement for training predictive machine learning models on Coripet RVM performance is that models must be trained **only on physical points that are currently active in the operating network as of the end of August 2026**, restricted to **commercial retail stores (`PUNTOVENDITA`)**, and evaluated after operational stabilization (**mature observations only**).

Previous interim documentation referenced **1,909 physical points** resulting from an unconstrained ramp-up filter applied across the entire historical operational registry. However, the **1,909 figure is not the active network**. It erroneously included **246 historically decommissioned or closed physical points** that ceased operations prior to 31 August 2026.

This report provides the data-grounded evaluation of physical-point status as of **31 August 2026**, reconciles the calculated active physical point count against the business reference figure of **1,831 active points**, and details the exact filtering funnel.

---

## 2. Status Field Audit (`Stato` in Historical Registry)

The historical registry `CaricaViewStoricoPuntoFisico.csv` contains 5,632 historical configuration records covering 2,539 unique physical points (`PuntoFisico`).

### A. Distinct Values of `Stato` Across Full Table
| `Stato` Code | Italian Term / Meaning | Operational Definition | Historical Records |
| :--- | :--- | :--- | :---: |
| **`ATTVO`** | *Attivo* | Active, deployed, and operational RVM installation | 3,380 |
| **`POT`** | *Potenziale* | Potential site; machine under delivery or pre-activation setup | 1,296 |
| **`SPL`** | *Sopralluogo* | Site inspection / feasibility survey underway | 581 |
| **`DISATT`** | *Disattivato* | Decommissioned, dismantled, or store closed | 331 |
| **`SOSP`** | *Sospeso* | Temporarily suspended (store refurbishment, seasonal pause, technical fault) | 28 |
| **`CENS`** | *Censito* | Preliminary database registry entry prior to survey | 8 |
| *(null)* | Missing | Historical legacy records without explicit status | 8 |
| **Total** | | | **5,632** |

### B. Valid Configuration Interval Rule on 31 August 2026
For each physical point, its configuration valid on `2026-08-31` is determined by:
$$\text{DataInizioRange} \le \text{'2026-08-31'} \quad \text{AND} \quad (\text{DataFineRange is null} \;\lor\; \text{DataFineRange} \ge \text{'2026-08-31'})$$

* Total physical points with a valid configuration on 2026-08-31: **2,527**
* Physical points in `daily_bottles.csv` with a valid configuration on 2026-08-31: **2,142 out of 2,142 (100.00%)**
* Physical points with **multiple valid configurations** on 2026-08-31: **0 (Zero)**
* Physical points with **ambiguous status** on 2026-08-31: **0 (Zero)**
* Physical points in registry without observations in `daily_bottles.csv`: **385** (pipeline survey/potential sites or decommissioned before 2023)

### C. Distribution of `Stato` Valid as of 31 August 2026
| `Stato` | `PUNTOVENDITA` | `COMUNE` | `PRIVATO` | `UFFICIOCOMUNALE` | Total Valid PFs | Operational Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`ATTVO`** | **1,704** | **117** | **6** | **0** | **1,827** | **Active as of 31 August 2026 (`active = 1`)** |
| **`DISATT`** | 249 | 57 | 16 | 0 | 322 | Inactive / Decommissioned (`active = 0`) |
| **`POT`** | 298 | 20 | 2 | 1 | 321 | Inactive / Pipeline Pre-activation (`active = 0`) |
| **`SPL`** | 52 | 0 | 0 | 0 | 52 | Inactive / Survey Stage (`active = 0`) |
| **`SOSP`** | 4 | 0 | 1 | 0 | 5 | Inactive / Temporarily Suspended (`active = 0`) |
| **Total** | **2,307** | **194** | **25** | **1** | **2,527** | |

### D. Data-Supported Active Rule
$$\mathbf{active\_as\_of\_2026\_08\_31} = 1 \iff \mathbf{Stato == 'ATTVO' \quad \text{on 2026-08-31}}$$

---

## 3. Reconciliation Against the Business Reference (1,831 Active Points)

The business reference noted approximately **1,831 active physical points** at the end of August 2026.

Our rigorous evaluation yields:
* **Active Physical Points on 31 August 2026 (`2026-08-31`):** Exactly **1,827 physical points**
* **Business Reference Count:** **1,831 physical points**
* **Net Difference:** **-4 physical points**

### Why Does the Discrepancy Exist?
The data snapshot in `CaricaViewStoricoPuntoFisico.csv` was extracted from the live production database on **11 September 2026** (effective dates extend through `11/09/2026`).

When the exact same interval filter is evaluated as of the database extraction date (**11 September 2026**):
$$\text{Active physical points on 2026-09-11} = \mathbf{1,831} \quad (1,708 \text{ PUNTOVENDITA} + 117 \text{ COMUNE} + 6 \text{ PRIVATO})$$

The figure **1,831 matches the business figure down to the single unit**.

### Exact Physical Points Explaining the Net +4 Difference:

#### 1. Six Physical Points Activated Between 01/09/2026 and 11/09/2026 (+6)
| PuntoFisico | Tipologia | Status on 31/08/2026 | New Status | Effective Date | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `MC6ZHDOAVNSP` | `PUNTOVENDITA` | `SOSP` | `ATTVO` | 09/09/2026 | Reactivated after suspension |
| `PF00003012` | `PUNTOVENDITA` | `POT` | `ATTVO` | 07/09/2026 | Pre-activation setup completed |
| `PF00003020` | `PUNTOVENDITA` | `POT` | `ATTVO` | 11/09/2026 | Pre-activation setup completed |
| `PF00003039` | `PUNTOVENDITA` | `SPL` | `ATTVO` | 10/09/2026 | Site survey completed, activated |
| `PF00003050` | `PUNTOVENDITA` | *(non-existent)* | `ATTVO` | 10/09/2026 | Newly installed machine |
| `PF00003054` | `PUNTOVENDITA` | *(non-existent)* | `ATTVO` | 11/09/2026 | Newly installed machine |

#### 2. Two Physical Points Deactivated Between 01/09/2026 and 11/09/2026 (-2)
| PuntoFisico | Tipologia | Status on 31/08/2026 | New Status | Effective Date | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `MCZF5IZMNNFQ` | `PUNTOVENDITA` | `ATTVO` (through 03/09) | `DISATT` | 04/09/2026 | Decommissioned on Sept 4 |
| `PF00001601` | `PUNTOVENDITA` | `ATTVO` (through 10/09) | `DISATT` | 11/09/2026 | Decommissioned on Sept 11 |

#### Mathematical Proof:
$$\text{Active as of 31 August 2026} = 1,827$$
$$\text{Activations in early September} = +6$$
$$\text{Deactivations in early September} = -2$$
$$\mathbf{\text{Active as of 11 September 2026}} = 1,827 + 6 - 2 = \mathbf{1,831}$$

*(Alternative interpretation: On 31 August 2026, exactly 4 retail stores were in `SOSP` (temporary suspension): `PF00002386`, `PF00002089`, `MC4IZDE8BPD1`, `MCR2F26VBQPE`. If an internal business dashboard treated temporarily suspended points as nominally active, $1,827 + 4 = \mathbf{1,831}$.)*

### Strict Compliance
Per user instructions:
* Do not force the result to 1,831.
* Do not use September 2026 activity.
* Reference date is strictly **31 August 2026**.
* The true active physical point count as of 31 August 2026 is **1,827 physical points** (comprising **1,704 retail supermarket physical points**).

---

## 4. Why 1,909 != 1,831

The previous report showed **1,909 physical points**. It is essential to understand why this difference occurred:

1. **Total Retail Points in History:** In `daily_bottles.csv`, 1,950 physical points were classified as `PUNTOVENDITA`.
2. **Unconstrained Maturity Filter:** Applying the 3-month maturity rule across all historical records without checking whether the physical point was still active as of 31 August 2026 removed 41 newly activated retail points:
$$1,950 - 41 = \mathbf{1,909 \text{ physical points}}$$
3. **The Critical Flaw:** Those 1,909 physical points included **246 retail stores that were decommissioned (`DISATT`) or closed prior to 31 August 2026**. Their historical observations had reached 3 months of age in 2023, 2024, or 2025, but the machines no longer exist in the operating network today.
4. **The Physical-Point Exclusion Rule:** The active filter is a **physical-point level gate**. If a physical point is inactive as of 31 August 2026, **all of its historical observations are excluded from the modeling dataset**.
5. **Reconciliation Formula:**
$$\begin{aligned}
\text{Mature Retail Historical PFs} &= 1,909 \\
- \text{Decommissioned/Inactive Retail PFs as of 31/08/2026} &= 246 \\
- \text{PROMO Active Mature Retail PFs} &= 7 \\
+ \text{Newly Active Retail PFs that achieved maturity} &= 1 \\
\hline
\mathbf{\text{Final Active Mature Retail Standard Modeling PFs}} &= \mathbf{1,657}
\end{aligned}$$
And adding the 40 newly active retail points that are still immature gives:
$$\mathbf{1,657 + 7 + 40 = 1,704 \text{ Active Retail PFs on 31 August 2026}}$$
Adding the non-retail active points (117 Comune + 6 Privato) gives:
$$\mathbf{1,704 + 117 + 6 = 1,827 \text{ Total Active PFs on 31 August 2026}}$$

---

## 5. Mandatory Filtering Funnel

The exact progression from base operational data to final modeling population:

```
Total Daily Observations (1,889,479 rows | 2,142 PFs | 100.00%)
       │
       ▼  [Gate 1: Location Filter]
PUNTOVENDITA PFs (1,683,271 rows | 1,950 PFs | 89.09%)
       │
       ▼  [Gate 2: Active as of 31 August 2026]
Active Retail PFs (1,521,151 rows | 1,704 PFs | 80.51%)
       │
       ▼  [Gate 3: Standard Attribute Filter]
Active Retail Standard PFs (1,512,567 rows | 1,697 PFs | 80.05%)
       │
       ▼  [Gate 4: Ramp-Up Maturity Rule]
FINAL MODELING OBSERVATIONS (1,396,544 rows | 1,657 PFs | 73.91%)
```

### Quantitative Funnel Table
| Step | Filter Stage | Unique PFs | Daily Observations | Cumulative % Retained | Observations Excluded in Step |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Total Daily Observations** (Base Operational History) | **2,142** | **1,889,479** | **100.00%** | Baseline |
| **2** | **PUNTOVENDITA PFs** (Retail Stores Only) | **1,950** | **1,683,271** | **89.09%** | 206,208 (192 PFs) |
| **3** | **Active as of 31-Aug-2026** (Commercial Active Network) | **1,704** | **1,521,151** | **80.51%** | 162,120 (246 PFs) |
| **4** | **STANDARD Attributo** (Standard Hardware Only) | **1,697** | **1,512,567** | **80.05%** | 8,584 (7 PFs) |
| **5** | **Mature Observations** (Stabilized Operations >= 3 Months) | **1,657** | **1,396,544** | **73.91%** | 116,023 (40 PFs completely excluded) |
| **6** | **FINAL MODELING OBSERVATIONS** (Certified Modeling Population) | **1,657** | **1,396,544** | **73.91%** | 0 |

---

## 6. Comprehensive Exclusion Breakdown

The 492,935 daily observations excluded from the modeling dataset fall into the following audited categories:

| Category | Description | Unique PFs Impacted | Observations Excluded | Reason for Exclusion |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Non-PUNTOVENDITA** | 192 | **206,208** | Municipal public collection points (`COMUNE`: 188,378 rows) and private corporate installations (`PRIVATO`: 17,830 rows) do not exhibit retail store consumer behavior. |
| **2** | **Currently Inactive as of 2026-08-31** | 246 *(among PV)*<br>315 *(across all)* | **162,120** *(among PV)*<br>217,273 *(all)* | Physical points decommissioned (`DISATT`), closed, or not yet operational as of 31 August 2026. All historical records excluded. |
| **3** | **PROMO Daily Observations** | 7 *(active PV)*<br>57 *(all)* | **8,584** *(active PV)*<br>54,133 *(all)* | Promotional/demonstration units averaging only 4–46 bottles/day (versus network average of ~594 bottles/day). Excluded from model training. |
| **4** | **No Valid Configuration on 2026-08-31** | 0 | **0** | Every single physical point in operational history has a valid configuration interval covering 2026-08-31. |
| **5** | **Ambiguous Current Status** | 0 | **0** | Zero duplicate overlapping records or ambiguous statuses on 2026-08-31. |
| **6** | **Immature Observations (Ramp-up)** | 1,330 | **116,023** | Daily observations recorded within 3 calendar months of `DataPrimaAttivazione` (`GIORNO < maturity_date`) for active retail standard PFs. |
| **7** | **PFs Completely Excluded (Never Mature)** | 40 | **2,224** | Active retail physical points commissioned between 01/06/2026 and 07/08/2026 that had not completed 3 full operational months by 31/08/2026. |
| **8** | **Final Eligible Physical Points** | **1,657** | — | Mature, active retail standard physical points. |
| **9** | **Final Eligible Daily Observations** | — | **1,396,544** | Final clean, certified outcome-training modeling population. |

---

## 7. List of 40 Active Retail PFs Excluded Due to Incomplete Maturity

These 40 physical points are currently active as of 31 August 2026 but were commissioned within 3 months of the reference date:

| PF ID | `DataPrimaAttivazione` | Total Observations (All Immature) | PF ID | `DataPrimaAttivazione` | Total Observations (All Immature) |
| :--- | :---: | :---: | :--- | :---: | :---: |
| `PF00002137` | 10/06/2026 | 83 | `PF00002996` | 10/06/2026 | 83 |
| `PF00002438` | 10/06/2026 | 83 | `PF00002999` | 02/07/2026 | 61 |
| `PF00002579` | 10/06/2026 | 83 | `PF00003000` | 01/07/2026 | 62 |
| `PF00002833` | 10/06/2026 | 83 | `PF00003001` | 01/07/2026 | 62 |
| `PF00002899` | 10/06/2026 | 83 | `PF00003002` | 03/07/2026 | 60 |
| `PF00002902` | 11/06/2026 | 82 | `PF00003004` | 07/07/2026 | 56 |
| `PF00002938` | 11/06/2026 | 82 | `PF00003006` | 10/07/2026 | 53 |
| `PF00002951` | 17/06/2026 | 76 | `PF00003008` | 14/07/2026 | 49 |
| `PF00002954` | 19/06/2026 | 74 | `PF00003009` | 15/07/2026 | 48 |
| `PF00002969` | 24/06/2026 | 69 | `PF00003010` | 15/07/2026 | 48 |
| `PF00002973` | 30/06/2026 | 63 | `PF00003011` | 16/07/2026 | 47 |
| `PF00002974` | 24/07/2026 | 39 | `PF00003014` | 23/07/2026 | 40 |
| `PF00002976` | 28/07/2026 | 35 | `PF00003015` | 23/07/2026 | 40 |
| `PF00002977` | 29/07/2026 | 34 | `PF00003016` | 30/07/2026 | 33 |
| `PF00002978` | 30/07/2026 | 33 | `PF00003018` | 30/07/2026 | 33 |
| `PF00002980` | 31/07/2026 | 32 | `PF00003028` | 31/07/2026 | 32 |
| `PF00002982` | 04/08/2026 | 28 | `PF00003029` | 31/07/2026 | 32 |
| `PF00002983` | 05/08/2026 | 27 | `PF00003033` | 07/08/2026 | 25 |
| `PF00002989` | 07/08/2026 | 25 | `PF00003034` | 07/08/2026 | 25 |
| `PF00002994` | 02/07/2026 | 61 | `PF00002995` | 02/07/2026 | 61 |

---

## 8. Certified Modeling Population Target Distribution

Final frozen modeling dataset: [`data/processed/modeling_daily_rvm.parquet`](file:///C:/Users/waqas/Desktop/Coripet_RVM_Predictor/data/processed/modeling_daily_rvm.parquet) (**1,396,544 rows**, **1,657 unique PFs**).

| Target Metric | Mean | Std Dev | Min | 25% | 50% (Median) | 75% | 90% | 99% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BOTTIGLIE` (Bottles/Day)** | **594.0** | 699.1 | 0.0 | 182.0 | **400.0** | 763.0 | 1,295.0 | 3,508.0 | 11,496.0 |
| **`PRESENZE` (Users/Day)** | **17.4** | 16.8 | 0.0 | 7.0 | **13.0** | 23.0 | 36.0 | 86.0 | 234.0 |

### Breakdown by Operational Year
| Year | Observation Period | Modeling Records | Share | Active Mature Standard PFs | Mean Daily Bottles | Mean Daily Presenze |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **2023** | 01/01/2023 – 31/12/2023 | 188,175 | 13.47% | 670 | 554.3 | 17.0 |
| **2024** | 01/01/2024 – 31/12/2024 | 327,644 | 23.46% | 1,083 | 604.4 | 18.1 |
| **2025** | 01/01/2025 – 31/12/2025 | 490,261 | 35.11% | 1,542 | 577.3 | 17.0 |
| **2026** | 01/01/2026 – 31/08/2026 | 390,464 | 27.96% | 1,657 | 625.4 | 17.6 |
| **Total** | **01/01/2023 – 31/08/2026** | **1,396,544** | **100.00%** | **1,657** | **594.0** | **17.4** |

---

## 9. Conclusion & Next Steps

* **Active network definition certified:** Evaluated strictly as of **31 August 2026** using `Stato == 'ATTVO'`.
* **Discrepancy against 1,831 fully resolved:** 1,831 is the exact count as of 11 September 2026 (export date); on 31 August 2026, the count is exactly 1,827 (+6 activations and -2 deactivations occurred in early September).
* **1,909 explained:** 1,909 included 246 decommissioned historical retail stores; the active retail mature count is exactly 1,664.
* **No models trained:** Machine learning training has not been initiated pending feature engineering.
