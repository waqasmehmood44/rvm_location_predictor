# Controlled Nielsen Matching Report: Coripet RVM Network

**Execution Timestamp:** 2026-09-15  
**Stage:** Stage 3 — Nielsen Retail Census Linkage  
**Status:** Completed  

---

## 1. Executive Summary

This report details the controlled, non-destructive matching between Coripet Physical Points (`CaricaViewStoricoPuntoFisico`) and the Italian retail grocery census (`filenielsen_1203`, 25,104 stores).

Because `filenielsen_1203` does not share a native synthetic identifier with Coripet, matching was conducted using an explicit, prioritized hierarchy from strict exact geographic entity resolution down to brand-catchment resolution, with every link categorized by match method and confidence tier.

---

## 2. Match Results & Confidence Distribution

| Match Classification | Count of Physical Points | Percentage of Network | Notes / Handling Rule |
| :--- | :---: | :---: | :--- |
| **MATCHED (Single Unambiguous Store)** | **1,067** | **42.02%** | Exactly 1 Nielsen store confirmed; store attributes attached |
| **AMBIGUOUS (Multiple Candidates)** | **309** | **12.17%** | Multiple candidate stores in close proximity; flagged without data inflation |
| **UNMATCHED (No Candidate Found)** | **1,163** | **45.81%** | Municipalities / stand-alone sites outside grocery census; null enriched |
| **Total Physical Points Evaluated** | **2,539** | **100.00%** | All unique physical points |

### Confidence Breakdown
* **`HIGH` Confidence:** **620** sites (Exact Normalized Address + CAP + Comune, or Address + CAP confirmed by Brand).
* **`MEDIUM` Confidence:** **449** sites (Brand + Address + Comune, or Unique Brand presence in CAP).
* **`LOW` / None Confidence:** **1,470** sites.

---

## 3. Hierarchical Matching Performance by Strategy

| Hierarchy Level | Strategy Name | Criteria | Matched | Ambiguous |
| :---: | :--- | :--- | :---: | :---: |
| **Level 1** | `EXACT_ADDR_CAP_COMUNE` | Normalized Street Address + 5-digit CAP + Municipality (+ Brand) | 618 | 2 |
| **Level 2** | `ADDR_CAP` | Normalized Street Address + 5-digit CAP (+ Brand disambiguation) | 2 | 0 |
| **Level 3** | `BRAND_ADDR_COMUNE` | Retail Chain Brand Token + Street Address + Municipality | 93 | 0 |
| **Level 4** | `UNIQUE_BRAND_CAP_COMUNE`| Retail Chain Brand Token + CAP + Comune (only 1 store in postal code) | 354 | 307 |

---

## 4. Key Enriched Nielsen Attributes

For matched physical points, the following Nielsen store characteristics are attached to the master dataset:
* `nielsen_country_shop_id`: Nielsen unique shop identifier
* `nielsen_tipo_negozio`: Store format (`LIS` free-service, `SUP` supermarket, `DIS` discount, `IPP` hypermarket)
* `nielsen_superficie_vendita`: Sales floor surface area in square meters
* `nielsen_numero_casse`: Number of active cash registers / checkouts
* `nielsen_insegna`: Nielsen commercial banner
* `nielsen_gruppo`: Nielsen commercial holding group
* `nielsen_area`: Italian Nielsen macro geographic zone (1 to 4)

> [!IMPORTANT]
> **Data Integrity Guarantee:** Physical point attributes (`Indirizzo`, `Cap`, `Comune`, `Filiera`) were **never overwritten**. Nielsen fields are added strictly as supplementary explanatory features prefixed with `nielsen_`.
