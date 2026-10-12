# Physical Point Lifecycle & History Report: Coripet RVM Network

**Execution Timestamp:** 2026-09-15  
**Stage:** Stage 2 — Physical Point Lifecycle Audit  
**Status:** Completed  

---

## 1. Executive Summary

This report analyzes the physical point (*Punto Fisico*) as the central operational unit of the Coripet platform. Physical points represent stationary installation sites, while machines, suppliers, retail clients, and commercial statuses evolve across distinct operational versions over time.

---

## 2. Key Physical Point Inventory & Observations

| Metric | Count / Statistic | Notes / Semantic Role |
| :--- | :---: | :--- |
| **Total Physical Points in Store Master** | **2,539** | All historical, active, and pipeline sites in `CaricaViewStoricoPuntoFisico` |
| **Active Physical Points in Operations** | **2,142** | Sites with actual logged recycling activity in `daily_bottles` (2023–2026) |
| **Non-Operational / Pipeline Sites** | **397** | Decommissioned prior to 2023, planned future sites, or non-active units |
| **Total Daily Observations Logged** | **1,889,479** | Chronological daily logs across all active sites |
| **Mean Daily Logs per Physical Point** | **882.1** days | Average operational tracking window per site |
| **Median Daily Logs per Physical Point** | **913.5** days | Median operational duration |
| **Min / Max Logs per Physical Point** | **20 / 1341** days | From single-day trials to continuous 1,339-day installations |

### Daily Observations per Physical Point (Quantiles)
| Percentile | 10% | 25% | 50% (Median) | 75% | 90% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Days Logged** | 363 | 545 | 913 | 1,319 | 1,339 | 1,341 |

---

## 3. Physical Point Dynamics & Hardware Evolution

Coripet physical points are not static; hardware units, machine types, and corporate entities change across validity intervals.

| Dimension | Physical Points Changing | Pct of Total PFs | Operational Explanation |
| :--- | :---: | :---: | :--- |
| **Changed Physical Machine (`Macchina`)** | **122** | **4.81%** | Hardware unit replaced (defective unit swap, capacity upgrade) |
| **Changed Machine Model (`TipoMacchina`)** | **94** | **3.70%** | Upgraded/downgraded model (e.g. from single-feed to dual-feed) |
| **Changed Physical Address (`Indirizzo`)** | **0** | **0.00%** | 0 changes: Physical locations are strictly stationary entities |
| **Changed Retail Banner / Client (`Filiera`)** | **128** | **5.04%** | Store rebranded following retail acquisition or franchise transfer |
| **Changed Operational Status (`Stato`)** | **1190** | **46.87%** | Transition between `ATTVO`, `SOSPESO`, `DISMESSO`, `PROGRAMMATO` |
| **Multiple Version Configurations** | **1767** | **69.59%** | Sites requiring interval-based historical version tracking |

---

## 4. Interval Consistency: Overlaps and Gaps

* **Gaps in Tracking:** **0 PFs (0.00%)**. Tracking across consecutive records is continuous.
* **Interval Overlaps:** **30 PFs (1.18%)**.
  - 26 PFs: Single-day boundary collisions (`DataFineRange` of old version matches `DataInizioRange` of new version). Resolved with left-closed right-open $[ 	ext{start}, 	ext{end} )$ intervals.
  - 4 PFs: Dual-machine concurrent installations (twin RVMs at high-volume hypermarkets, resolved with `CODICEMACCHINA == Macchina`).

---

## 5. Architectural Implications for Modeling

1. **Stationary Baseline:** Physical location attributes (`Indirizzo`, `Cap`, `Comune`, `Provincia`, `Regione`, `Latitudine`, `Longitudine`) are 100% time-invariant and serve as permanent site coordinates.
2. **Machine Type is Modifiable:** Because 94 sites underwent machine model upgrades, `TipoMacchina` is confirmed as a modifiable hardware configuration variable, matching the user scenario: *"Given a location and a selected machine type, predict daily volumes"*.
