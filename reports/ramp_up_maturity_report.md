# Ramp-Up Maturity & Active Status Report: 31 August 2026 Freeze

**Execution Timestamp:** 2026-09-15  
**Reference Date for Active Network:** **31 August 2026 (`2026-08-31`)**  
**Filter Rule:** Four-Gate Business Eligibility (Retail + Active 31/08/2026 + Standard Hardware + Maturity)  
**Status:** Completed | Final Modeling Population Frozen  

---

## 1. Executive Summary & Four-Gate Business Definition

To guarantee robust, operationally generalizable machine learning predictions, the training population is constrained to standard commercial supermarket RVM installations that remain actively deployed in the operating network as of **31 August 2026**:

1. **Gate 1 — Hard Location Filter:**
   `TipologiaPuntoFisico == 'PUNTOVENDITA'` (retail supermarket installations only).

2. **Gate 2 — Active as of 31 August 2026:**
   Evaluated at the physical-point configuration valid on `2026-08-31` using:
   `DataInizioRange <= '2026-08-31' AND (DataFineRange is null OR DataFineRange >= '2026-08-31')`
   `active_as_of_2026_08_31 = 1` if `Stato == 'ATTVO'`, else `0`.

3. **Gate 3 — Standard Attribute Filter (Observation-Level):**
   Evaluated at the historical configuration valid on observation `GIORNO`:
   `daily_standard_eligible = 1` if `Attributo == 'STANDARD'`, else `0` (excludes all `PROMO` records).

4. **Gate 4 — Ramp-Up Maturity Rule:**
   - `maturity_date = DataPrimaAttivazione + 3 calendar months`
   - `days_since_activation = GIORNO - DataPrimaAttivazione`
   - `months_since_activation = days_since_activation / 30.4375`
   - `maturity_status = 'MATURE'` if `GIORNO >= maturity_date`, else `'IMMATURE'`.

$$\mathbf{modeling\_eligible} = (\mathbf{puntovendita\_eligible == 1}) \land (\mathbf{active\_as\_of\_2026\_08\_31 == 1}) \land (\mathbf{daily\_standard\_eligible == 1}) \land (\mathbf{maturity\_status == 'MATURE'})$$

---

## 2. Validation Against Business Reference (1,831 Active Points)

The business reference noted approximately **1,831 physical points active at the end of August 2026**.

A rigorous historical interval audit confirms:
* **Active Physical Points as of 31 August 2026:** Exactly **1,827 physical points** had `Stato == 'ATTVO'` on `2026-08-31` (1,704 `PUNTOVENDITA` + 117 `COMUNE` + 6 `PRIVATO`).
* **Active Physical Points as of 11 September 2026 (Export Snapshot):** Exactly **1,831 physical points** had `Stato == 'ATTVO'` on `2026-09-11` (1,708 `PUNTOVENDITA` + 117 `COMUNE` + 6 `PRIVATO`).

### Discrepancy Investigation & Exact PF Breakdown
The difference of **+4 physical points** between 31 August 2026 and the 1,831 figure is completely explained by status transitions during early September 2026 (1 September to 11 September 2026):

#### A. Six Physical Points Activated (`ATTVO`) in Early September 2026 (+6)
1. `MC6ZHDOAVNSP` (`PUNTOVENDITA`): Was `SOSP` (suspended) on 31/08/2026; reactivated to `ATTVO` on 09/09/2026.
2. `PF00003012` (`PUNTOVENDITA`): Was `POT` (potential/installing) on 31/08/2026; activated to `ATTVO` on 07/09/2026.
3. `PF00003020` (`PUNTOVENDITA`): Was `POT` on 31/08/2026; activated to `ATTVO` on 11/09/2026.
4. `PF00003039` (`PUNTOVENDITA`): Was `SPL` (site survey) on 31/08/2026; became `POT` on 01/09/2026 and `ATTVO` on 10/09/2026.
5. `PF00003050` (`PUNTOVENDITA`): New site commissioned in September; activated to `ATTVO` on 10/09/2026.
6. `PF00003054` (`PUNTOVENDITA`): New site commissioned in September; activated to `ATTVO` on 11/09/2026.

#### B. Two Physical Points Deactivated (`DISATT`) in Early September 2026 (-2)
1. `MCZF5IZMNNFQ` (`PUNTOVENDITA`): Was `ATTVO` through 03/09/2026; deactivated to `DISATT` on 04/09/2026.
2. `PF00001601` (`PUNTOVENDITA`): Was `ATTVO` through 10/09/2026; deactivated to `DISATT` on 11/09/2026.

#### Net Mathematical Proof
$$\text{Active as of 31 August 2026} = 1,827$$
$$\text{Active as of 11 September 2026 (Snapshot)} = 1,827 + 6 - 2 = \mathbf{1,831}$$

*(Alternatively, if the 4 retail stores with `Stato == 'SOSP'` on August 31 were informally included in the operational count, $1,827 + 4 = 1,831$.)*

Per explicit business instruction, the reference date is strictly **31 August 2026**, so exactly **1,827 physical points** (and **1,704 retail supermarket physical points**) are classified as `active_as_of_2026_08_31 == 1`.

---

## 3. Reconciliation: Why 1,909 != 1,831

The previous report referenced **1,909 physical points**. This figure is not the current active network:

1. **Cumulative Historical Retail Cohort:** In the operational log (`daily_bottles.csv`), 1,950 physical points were classified as `PUNTOVENDITA`.
2. **Unconstrained Historical Maturity:** Applying the 3-month maturity rule across all historical records without checking whether the physical point was still active as of 31 August 2026 excluded 41 newly activated retail points ($1,950 - 41 = 1,909$).
3. **The Inclusion of Decommissioned Sites:** Those 1,909 physical points included **246 retail stores that were decommissioned (`DISATT`) or closed prior to 31 August 2026**.
4. **Physical-Point Exclusion Rule:** The active filter is a **physical-point level gate**. If a physical point is inactive as of 31 August 2026, **all of its historical observations are excluded from the modeling dataset**.
5. **PROMO Exclusion Impact:** Among the active mature retail network, exactly **7 physical points** were promotional machines (`Attributo == 'PROMO'`) averaging only 4–46 bottles/day. Excluding them removes 8,209 PROMO rows.
6. **Reconciliation Formula:**
$$\begin{aligned}
\text{Mature Retail Historical PFs} &= 1,909 \\
- \text{Decommissioned/Inactive Retail PFs as of 31/08/2026} &= 246 \\
- \text{PROMO Active Mature Retail PFs} &= 7 \\
+ \text{Newly Active Retail PFs that achieved maturity} &= 1 \\
\hline
\mathbf{\text{Final Active Mature Retail Standard Modeling PFs}} &= \mathbf{1,657}
\end{aligned}$$

---

## 4. Mandatory Filtering Funnel

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
FINAL MODELING PFs (1,396,544 rows | 1,657 PFs | 73.91%)
```

### Funnel Metrics Table
| Step | Filter Stage | Unique PFs | Daily Observations | Cumulative % Retained | Step Excluded Rows |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Total Daily Observations** (Base Operational History) | **2,142** | **1,889,479** | **100.00%** | Baseline |
| **2** | **PUNTOVENDITA PFs** (Retail Stores Only) | **1,950** | **1,683,271** | **89.09%** | 206,208 (192 PFs) |
| **3** | **Active as of 31-Aug-2026** (Commercial Active Network) | **1,704** | **1,521,151** | **80.51%** | 162,120 (246 PFs) |
| **4** | **STANDARD Attributo** (Standard Hardware Only) | **1,697** | **1,512,567** | **80.05%** | 8,584 (7 PFs) |
| **5** | **Mature Observations** (Stabilized Operations >= 3 Months) | **1,657** | **1,396,544** | **73.91%** | 116,023 (40 PFs completely excluded) |
| **6** | **FINAL MODELING OBSERVATIONS** (Certified Modeling Population) | **1,657** | **1,396,544** | **73.91%** | 0 |

---

## 5. Comprehensive Exclusion Breakdown

| Category | Description | Unique PFs Impacted | Observations Excluded | Reason for Exclusion |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Non-PUNTOVENDITA** | 192 | **206,208** | Municipal public collection points (`COMUNE`: 188,378 rows) and private corporate installations (`PRIVATO`: 17,830 rows). |
| **2** | **Currently Inactive as of 2026-08-31** | 246 *(among PV)* | **162,120** | Physical points decommissioned (`DISATT`), closed, or not yet operational on 31 August 2026. |
| **3** | **PROMO Daily Observations** | 7 *(active PV)*<br>57 *(all)* | **8,584** *(active PV)*<br>54,133 *(all)* | Promotional/demonstration units averaging only 4–46 bottles/day. Excluded from training. |
| **4** | **Immature Observations (Ramp-up)** | 1,330 | **116,023** | Daily observations recorded within 3 calendar months of `DataPrimaAttivazione` (`GIORNO < maturity_date`). |
| **5** | **PFs Completely Excluded (Never Mature)** | 40 | **2,224** | Active retail physical points commissioned between 01/06/2026 and 07/08/2026 with < 3 months operation. |
| **6** | **Final Eligible Physical Points** | **1,657** | — | Mature, active retail standard physical points. |
| **7** | **Final Eligible Daily Observations** | — | **1,396,544** | Final clean, certified outcome-training modeling population. |

---

## 6. Comprehensive Validation Metrics Table

| Metric | Value | Reference / Notes |
| :--- | :---: | :--- |
| **1. Total Daily Observations Before Filtering** | **1,889,479** | Full unified operational database (2023–2026) |
| **2. Total Observations with `PUNTOVENDITA`** | **1,683,271** | All commercial retail physical points (89.09%) |
| **3. Total Observations Excluded (Not `PUNTOVENDITA`)** | **206,208** | 188,378 Municipal (`COMUNE`) + 17,830 Private (`PRIVATO`) |
| **4. Active Physical Points as of 31 August 2026** | **1,827** | **1,827 total** (1,704 `PUNTOVENDITA` + 117 `COMUNE` + 6 `PRIVATO`) |
| **5. Retail Physical Points Active as of 31 August 2026** | **1,704** | Commercial retail network active on 2026-08-31 |
| **6. Retail Observations Active as of 31 August 2026** | **1,521,151** | Operational records of currently active retail PFs |
| **7. Total Observations with `Attributo == 'STANDARD'`** | **1,835,346** | Standard attribute daily records across history (97.14%) |
| **8. Total Observations Excluded (PROMO)** | **54,133** | All promotional records excluded from model training |
| **9. PROMO Observations Excluded in Mature Retail Network** | **8,209** | 8,209 records across 7 promotional retail physical points |
| **10. Total `IMMATURE` Observations (Retail Active Standard)** | **116,023** | 3-month ramp-up records of active standard retail points |
| **11. Total `MATURE` Observations (Retail Active Standard)** | **1,396,544** | **Final training dataset row count (73.91% retained)** |
| **12. Total `UNKNOWN` Observations** | **0** | Zero missing or invalid activation dates |
| **13. Final Modeling Observations** | **1,396,544** | Strictly `PUNTOVENDITA` & Active 2026-08-31 & STANDARD & `MATURE` |
| **14. Final Unique PFs in Modeling Population** | **1,657** | Mature active commercial retail standard physical points |
| **15. Earliest `DataPrimaAttivazione`** | **2019-05-01 00:00:00** | Legacy pilot activation date |
| **16. Latest `DataPrimaAttivazione`** | **2026-08-07 00:00:00** | Most recent network machine activation |
| **17. Earliest Mature Observation Date** | **2023-01-01 00:00:00** | Start of modeling training window |
| **18. Latest Mature Observation Date** | **2026-08-31 00:00:00** | End of modeling training window |

---

## 7. Modeling Population by Operational Year

| Operational Year | Calendar Span | Mature Active Standard Records | Annual Share | Active Mature Standard PFs | Average Daily Bottles |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **2023** | 01/01/2023 – 31/12/2023 | **188,175** | 13.56% | 853 | 601.2 |
| **2024** | 01/01/2024 – 31/12/2024 | **327,644** | 23.47% | 1,218 | 627.1 |
| **2025** | 01/01/2025 – 31/12/2025 | **490,261** | 35.08% | 1,528 | 601.2 |
| **2026** | 01/01/2026 – 31/08/2026 | **390,464** | 27.89% | 1,657 | 553.4 |
| **Total** | **01/01/2023 – 31/08/2026** | **1,396,544** | **100.00%** | **1,657** | **593.9** |

---

## 8. Artifact Manifest

* 🎯 **Final Modeling Dataset:** [`data\processed\modeling_daily_rvm.parquet`](file:///C:/Users/waqas/Desktop/Coripet_RVM_Predictor/data/processed/modeling_daily_rvm.parquet) (35.3 MB)
* 📦 **Master Retail Dataset (Unfiltered by Maturity/Active/Standard):** [`data\processed\master_daily_rvm.parquet`](file:///C:/Users/waqas/Desktop/Coripet_RVM_Predictor/data/processed/master_daily_rvm.parquet)
* 📋 **Audit Columns Included:** `puntovendita_eligible`, `active_as_of_2026_08_31`, `daily_standard_eligible`, `maturity_date`, `days_since_activation`, `months_since_activation`, `maturity_status`, `modeling_eligible`
