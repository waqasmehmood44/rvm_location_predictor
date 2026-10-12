# Modeling Readiness & Population Freeze Report: Coripet RVM Platform

**Execution Timestamp:** 2026-09-15  
**Stage:** Four-Gate Business Eligibility Filtering & Final Modeling Dataset Freeze  
**Reference Date for Active Network:** **Active as of 31 August 2026**  
**Status:** Completed | Modeling Population Ready for Training  

---

## 1. Four-Gate Eligibility Architecture

To ensure the machine learning models learn purely from commercially viable, currently active, standard-hardware, and operationally stabilized reverse vending machine behaviors, four strict business eligibility gates were applied:

```
Full Operational History (1,889,479 rows / 2,142 PFs)
               │
               ▼  [Gate 1: Location Type Rule]
TipologiaPuntoFisico == 'PUNTOVENDITA'     ──►  Excluded non-retail: 206,208 rows (192 PFs)
               │
               ▼  [Master Retail Set: 1,683,271 rows / 1,950 PFs]
active_as_of_2026_08_31 == 1               ──►  Excluded inactive:   162,120 rows (246 PFs)
(Valid Stato == 'ATTVO' on 31 August 2026)
               │
               ▼  [Active Retail Set: 1,521,151 rows / 1,704 PFs]
daily_standard_eligible == 1               ──►  Excluded PROMO:       8,584 rows (7 PFs)
(Attributo == 'STANDARD' on observation GIORNO)
               │
               ▼  [Active Retail Standard Set: 1,512,567 rows / 1,697 PFs]
maturity_status == 'MATURE'                ──►  Excluded ramp-up:    116,023 rows (40 PFs temporary)
(GIORNO >= DataPrimaAttivazione + 3 months)
               │
               ▼  [Gate 4: Operational Maturity Rule]
Final Modeling Dataset: 1,396,544 rows / 1,657 PFs (73.91% retained)
```

---

## 2. Population Attrition & Retention Summary

| Pipeline Filtering Stage | Daily Observations | Cumulative Retention | Unique PFs | PFs Retained | Excluded Records |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Full Operational Base** | **1,889,479** | 100.00% | 2,142 | 100.00% | Baseline |
| **1. Gate 1: `PUNTOVENDITA` Retail Only** | **1,683,271** | 89.09% | 1,950 | 91.04% | 206,208 (Municipal & Private) |
| **2. Gate 2: Active as of 31 August 2026** | **1,521,151** | 80.51% | 1,704 | 79.55% | 162,120 (Historical Decommissioned) |
| **3. Gate 3: Standard Attribute (`STANDARD`)** | **1,512,567** | 80.05% | 1,697 | 79.22% | 8,584 (PROMO Daily Observations) |
| **4. Gate 4: Operational Maturity (`MATURE`)** | **1,396,544** | **73.91%** | **1,657** | **77.36%** | 116,023 (3-month ramp-up) |

---

## 3. Modeling Target Variables (Final Mature Active Standard Retail Population)

Target variable summary across all **1,396,544 mature active standard observations**:

| Target Variable | Min | 25% | Median (50%) | Mean | 75% | 90% | 99% | Max | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`BOTTIGLIE`** | 0.0 | 208.0 | **399.0** | **593.9** | 762.0 | 1,294.0 | 3,503.0 | 11,496.0 | 569.8 |
| **`PRESENZE`** | 0.0 | 8.0 | **13.0** | **17.4** | 23.0 | 36.0 | 86.0 | 234.0 | 13.9 |

> [!TIP]
> Excluding the 7 `PROMO` physical points (which averaged only 4–46 bottles/day) lifted the network mean from 590.6 to **593.9 bottles/day** and median from 397.0 to **399.0 bottles/day**, removing low-volume promotional noise and ensuring the training population reflects standard operational performance.

---

## 4. Final Modeling Dataset Artifacts

* 🎯 **Final Modeling Parquet:** [`data\processed\modeling_daily_rvm.parquet`](file:///C:/Users/waqas/Desktop/Coripet_RVM_Predictor/data/processed/modeling_daily_rvm.parquet) (35.3 MB)
* 📄 **Final Modeling CSV:** [`data\processed\modeling_daily_rvm.csv`](file:///C:/Users/waqas/Desktop/Coripet_RVM_Predictor/data/processed/modeling_daily_rvm.csv) (784.0 MB)
* **Strict Four-Gate Modeling Filter:** `puntovendita_eligible == 1` AND `active_as_of_2026_08_31 == 1` AND `daily_standard_eligible == 1` AND `maturity_status == 'MATURE'` (flag: `modeling_eligible == 1`).
* **Zero PROMO Observations:** Verified by automated assertion.
* **Underlying Datasets Preserved:**
  - `data/processed/full_integrated_daily_rvm.parquet` (all 1,889,479 rows)
  - `data/processed/master_daily_rvm.parquet` (all 1,683,271 retail rows with full audit metadata)
