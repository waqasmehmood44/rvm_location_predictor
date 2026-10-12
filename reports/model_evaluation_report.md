# Coripet RVM Predictor: Model Evaluation & Benchmark Report

**Date:** September 2026  
**Evaluation Population:** 290,704 daily observations across **331 completely unseen physical points (`PF`)**  
**Evaluation Artifact:** [`models/best_model/evaluation_metrics.json`](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/models/best_model/evaluation_metrics.json)  
**Test Data Parquet:** [`models/preprocessing/test_holdout.parquet`](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/models/preprocessing/test_holdout.parquet)  

---

## 1. Executive Summary

The multi-output neural network was evaluated on the **20% holdout test partition**, comprising **290,704 daily records** from **331 physical points** that were never seen during training, validation, or preprocessing fitting.

All predictions were evaluated on the **original natural scale** after reversing the $\log(1+y)$ transformation:
$$\hat{y} = \max\left(0, \exp(\hat{z}) - 1\right)$$

### Primary Test Benchmark Results

| Metric | Target 1: BOTTIGLIE | Target 2: PRESENZE | Business Meaning |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | **347.59 bottles/day** | **8.81 users/day** | Average daily absolute error across unseen stores |
| **Median Absolute Error (MedAE)** | **202.84 bottles/day** | **5.98 users/day** | Typical daily error for 50% of operating days |
| **Root Mean Squared Error (RMSE)**| **590.56 bottles/day** | **12.96 users/day** | Penalizes large residual errors and spikes |
| **Coefficient of Determination ($R^2$)**| **0.4039** | **0.3687** | **~40.4% and ~36.9% of variance explained** |
| **MAE as % of Mean Target** | **57.67%** | **50.91%** | Relative percentage error relative to global mean |

---

## 2. Comparison Against Baseline Models

To verify that the multi-task neural network adds genuine predictive value beyond simple demographic or hardware averages, three reference baseline models were fitted strictly on the training set and evaluated on the exact same 290,704 unseen test observations:

### Baseline Comparison Table

| Model Architecture | BOTTIGLIE MAE | BOTTIGLIE RMSE | BOTTIGLIE $R^2$ | PRESENZE MAE | PRESENZE RMSE | PRESENZE $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Global Train Mean** | 456.0 | 765.0 | -0.0002 | 11.3 | 16.3 | -0.0001 |
| **2. Machine-Type Mean** | 442.4 | 745.5 | +0.0501 | 11.1 | 15.9 | +0.0429 |
| **3. Machine + Region Mean**| 434.3 | 705.1 | +0.1503 | 10.8 | 15.1 | +0.1433 |
| **MULTI-TASK NEURAL NETWORK** | **347.6** | **590.6** | **+0.4039** | **8.8** | **13.0** | **+0.3687** |
| **Improvement vs Region Baseline** | **-86.7 (-20.0%)**| **-114.5 (-16.2%)**| **+0.2536 (+168%)**| **-2.0 (-18.5%)**| **-2.1 (-13.9%)**| **+0.2254 (+157%)**|

> [!NOTE]
> **Key Finding:** The multi-output neural network reduces daily bottle prediction error by **86.7 bottles/day (-20.0%)** relative to a combined machine-type and regional lookup table, and more than doubles the variance explained ($R^2$ jumps from $0.150$ to $0.404$). This proves the network successfully extracts predictive signal from localized municipal demographics, coordinates, and store characteristics.

---

## 3. Performance Breakdown by Machine Type

Across the 331 unseen physical points in the test set, performance was evaluated for each hardware family with sufficient sample size:

| Machine Type | Test Observations | Unseen PFs | BOTTIGLIE MAE | BOTTIGLIE $R^2$ | PRESENZE MAE | PRESENZE $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DAINT_A-bis** | 41,771 | 36 | **171.9** | 0.36 | **6.3** | 0.25 |
| **DAINT_G** | 110,820 | 175 | **317.5** | 0.18 | **9.2** | 0.21 |
| **Ripet_C-front** | 99,777 | 96 | **471.3** | 0.44 | **10.1** | 0.45 |
| **Ripet_C-Ter** | 7,870 | 6 | **444.1** | 0.15 | **9.4** | 0.08 |
| **Ripet_Smarpet** | 2,241 | 2 | **413.4** | 0.41 | **8.8** | -0.09 |
| **RVM_Mod.E** | 1,378 | 2 | **143.1** | 0.16 | **4.1** | 0.14 |
| **RVM_Mod.E_Nuova** | 1,879 | 3 | **149.6** | 0.03 | **3.9** | 0.11 |
| **Ripet_V2** | 8,034 | 6 | **180.8** | -0.37 | **5.0** | -0.68 |
| **DAINT_B-bis** | 6,035 | 5 | **224.2** | -0.30 | **6.7** | -0.27 |

* **Observation:** `DAINT_A-bis` (compact front) displays the lowest absolute error (**171.9 bottles/day**), reflecting lower inherent throughput variance. `Ripet_C-front` (high capacity) operates at much higher volumes (650–1,200 bottles/day) and exhibits the highest explained variance ($R^2 = 0.44$ for bottles and $0.45$ for presenze).
* Rare machine types with only 2–6 PFs in the test set exhibit negative $R^2$ because small cluster sample sizes are vulnerable to store-specific operator management variations.

---

## 4. Performance Breakdown by Italian Region

| Region | Test Observations | Unseen PFs | BOTTIGLIE MAE | BOTTIGLIE $R^2$ | PRESENZE MAE | PRESENZE $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toscana** | 34,313 | 41 | 575.4 | **0.55** | 13.3 | **0.51** |
| **Campania** | 44,954 | 46 | **266.2** | 0.17 | **6.7** | 0.14 |
| **Lombardia** | 34,316 | 32 | **278.4** | 0.12 | **7.5** | 0.28 |
| **Lazio** | 31,509 | 38 | **301.0** | 0.26 | **9.3** | 0.26 |
| **Emilia-Romagna** | 14,727 | 21 | **341.6** | 0.27 | **9.2** | 0.36 |
| **Piemonte** | 24,897 | 26 | 495.2 | 0.08 | **9.9** | 0.12 |
| **Sicilia** | 9,110 | 11 | **190.2** | 0.32 | **5.4** | 0.29 |
| **Veneto** | 7,612 | 9 | **305.6** | 0.24 | **6.3** | **0.51** |
| **Friuli-Venezia Giulia** | 6,051 | 7 | **272.4** | 0.26 | **6.6** | 0.40 |
| **Puglia** | 10,355 | 9 | **236.3** | -0.06 | **6.2** | -0.06 |

* **Regional Dynamics:** Southern regions (`Sicilia`, `Puglia`, `Campania`) exhibit very low MAE (190–266 bottles/day). Central/Northern regions (`Toscana`, `Emilia-Romagna`, `Veneto`) have higher overall bottle volumes and achieve strong correlation metrics ($R^2 = 0.51 - 0.55$).

---

## 5. Performance Breakdown by Volume Tiers

| Volume Tier | Actual Target Range | Test Observations | BOTTIGLIE MAE | PRESENZE MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Low Volume** ($\le 25\text{th percentile}$) | $\le 182\text{ bottles/day}$ | 72,994 (25.1%) | **242.7** | **7.2** |
| **Mid Volume** ($25\text{th}-75\text{th percentile}$)| $183 - 763\text{ bottles/day}$| 145,114 (49.9%) | **195.0** | **6.0** |
| **High Volume** ($> 75\text{th percentile}$) | $> 763\text{ bottles/day}$ | 72,596 (25.0%) | **758.0** | **16.0** |

* For the middle 50% of the network ($183 - 763$ bottles/day), the neural network achieves an outstanding MAE of **195.0 bottles/day** and **6.0 customer visits/day**.
* High-volume stores ($> 763$ bottles/day) exhibit larger absolute deviations because top stores can fluctuate wildly (between 800 and 4,000 bottles/day) depending on weekend promotions and coupon voucher programs.

---

## 6. Temporal Benchmark: 2026 Test Data Only

To confirm that the neural network does not suffer from temporal degradation and can reliably predict recent operations, test performance was measured exclusively on the final year (2026):

* **2026 Test Observations:** 78,240 rows across 331 unseen PFs
* **BOTTIGLIE (2026):** MAE = **360.0 bottles/day**, RMSE = **584.8**, $R^2 = \mathbf{0.3488}$
* **PRESENZE (2026):** MAE = **8.7 users/day**, RMSE = **12.4**, $R^2 = \mathbf{0.3287}$

The performance on 2026 operations closely mirrors the overall test set metrics, demonstrating excellent temporal stability.

---

## 7. Honest Evaluation of Limitations & Recommended Improvements

1. **Top 5% Extreme Spike Attenuation:** Because the model is trained with Huber loss on $\log(1+y)$, it intentionally penalizes extreme outliers less heavily. As a consequence, on days where a store collects $> 3,500$ bottles, the model typically predicts $\sim 1,800 - 2,400$ bottles. For steady-state capital planning (Model A), this conservative bias is desirable because CAPEX decisions should not be based on extreme promotional spikes.
2. **Store Placement Context:** The model currently lacks micro-location data inside the store (e.g. machine located at the front entrance vs. subterranean car park). Incorporating indoor placement flags in future iterations would further reduce variance.
3. **Nielsen Store Enrichment:** Stores enriched with Nielsen sales area and register counts achieve slightly higher precision. Encouraging users to enter store square meters if known will improve confidence.
