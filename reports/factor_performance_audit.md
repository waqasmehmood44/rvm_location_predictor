# Holdout Store Factor Performance Audit & Diagnostics
**Total Stores Evaluated:** 331 Unseen Physical Locations (PFs)
**Daily Records Processed:** 290,704 Observations

## 1. Overall Store-Level Predictive Accuracy
| Metric | Pure Neural Network | Stacking Ensemble | Delta Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **273.7 b/day** | **251.6 b/day** | -22.1 b/day |
| **Median Absolute Error (MedAE)** | **147.0 b/day** | **147.6 b/day** | 0.6 b/day |
| **Median Absolute % Error (MedAPE)** | **34.8%** | **37.2%** | 2.4% |
| **Mean Absolute % Error (MAPE)** | **48.9%** | **44.3%** | -4.6% |
| **Systematic Bias (Pred - Act)** | **-120.9 b/day** | **-125.1 b/day** | - |
| **R² Score** | **0.271** | **0.429** | +0.158 |

## 2. Factor-by-Factor Diagnostic Scorecards

### Hardware Model (TipoMacchina)
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DAINT_G** | 169 | 567.0 | 494.6 | 243.1 | 33.8% | -72.4 | 🟡 MODERATE (Acceptable) |
| **Ripet_C-front** | 94 | 815.3 | 574.8 | 423.3 | 41.9% | -240.5 | 🔴 UNDERPERFORMING (Action Needed) |
| **DAINT_A-bis** | 35 | 314.1 | 285.0 | 94.0 | 27.8% | -29.1 | 🟡 MODERATE (Acceptable) |
| **Ripet_C-Bis** | 8 | 494.7 | 352.2 | 169.0 | 24.7% | -142.5 | 🟢 PERFORMS WELL (Accurate) |
| **Ripet_C-Ter** | 6 | 683.4 | 320.3 | 363.1 | 50.5% | -363.1 | 🔴 UNDERPERFORMING (Action Needed) |
| **Ripet_V2** | 6 | 276.8 | 372.6 | 132.7 | 23.5% | +95.8 | 🟢 PERFORMS WELL (Accurate) |
| **DAINT_B-bis** | 5 | 407.4 | 188.1 | 219.2 | 55.4% | -219.2 | 🔴 UNDERPERFORMING (Action Needed) |
| **RVM_Mod.E_Nuova** | 3 | 268.7 | 272.7 | 114.7 | 49.7% | +4.0 | 🔴 UNDERPERFORMING (Action Needed) |
| **RVM_Mod.E** | 2 | 596.4 | 453.8 | 142.5 | 27.5% | -142.5 | 🟡 MODERATE (Acceptable) |
| **Ripet_Smarpet** | 1 | 1311.8 | 1275.4 | 36.4 | 2.8% | -36.4 | 🟢 PERFORMS WELL (Accurate) |
| **Ripet_V1** | 1 | 207.4 | 31.6 | 175.8 | 84.8% | -175.8 | 🔴 UNDERPERFORMING (Action Needed) |
| **SUPERMARPET01S** | 1 | 279.9 | 454.0 | 174.1 | 62.2% | +174.1 | 🔴 UNDERPERFORMING (Action Needed) |

### Hardware Supplier (Fornitore)
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **DAINT** | 209 | 520.8 | 452.2 | 217.6 | 33.5% | -68.6 | 🟡 MODERATE (Acceptable) |
| **RIPET** | 115 | 752.7 | 530.7 | 385.1 | 41.6% | -222.0 | 🔴 UNDERPERFORMING (Action Needed) |
| **EURVEN** | 5 | 399.8 | 345.2 | 125.8 | 33.3% | -54.6 | 🟡 MODERATE (Acceptable) |
| **DENTISTECH** | 2 | 795.8 | 864.7 | 105.2 | 32.5% | +68.9 | 🟡 MODERATE (Acceptable) |

### Store Format (nielsen_tipo_negozio)
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **UNKNOWN** | 135 | 489.9 | 364.8 | 212.2 | 37.4% | -125.2 | 🟡 MODERATE (Acceptable) |
| **SUP** | 111 | 645.9 | 526.4 | 269.9 | 34.8% | -119.5 | 🟡 MODERATE (Acceptable) |
| **IPR** | 33 | 1027.1 | 871.9 | 385.9 | 33.0% | -155.2 | 🟡 MODERATE (Acceptable) |
| **LIS** | 33 | 657.3 | 471.6 | 483.0 | 49.0% | -185.7 | 🔴 UNDERPERFORMING (Action Needed) |
| **DIS** | 19 | 294.2 | 368.0 | 174.6 | 35.2% | +73.7 | 🟡 MODERATE (Acceptable) |

### Sales Surface Category
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Unknown / Imputed** | 135 | 489.9 | 364.8 | 212.2 | 37.4% | -125.2 | 🟡 MODERATE (Acceptable) |
| **Standard Supermarket (450-1,200 m²)** | 91 | 484.9 | 421.7 | 217.6 | 35.2% | -63.2 | 🟡 MODERATE (Acceptable) |
| **Small / Proximity (< 450 m²)** | 41 | 635.4 | 458.8 | 438.9 | 49.0% | -176.6 | 🔴 UNDERPERFORMING (Action Needed) |
| **Hypermarket (> 2,500 m²)** | 33 | 1027.1 | 871.9 | 385.9 | 33.0% | -155.2 | 🟡 MODERATE (Acceptable) |
| **Superstore (1,200-2,500 m²)** | 31 | 929.0 | 767.4 | 368.5 | 28.9% | -161.6 | 🟡 MODERATE (Acceptable) |

### Retail Buying Group (nielsen_gruppo)
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **UNKNOWN** | 135 | 489.9 | 364.8 | 212.2 | 37.4% | -125.2 | 🟡 MODERATE (Acceptable) |
| **CONAD** | 88 | 696.0 | 642.1 | 342.6 | 42.3% | -53.8 | 🔴 UNDERPERFORMING (Action Needed) |
| **COOP ITALIA** | 30 | 1242.2 | 779.3 | 666.4 | 40.8% | -462.9 | 🔴 UNDERPERFORMING (Action Needed) |
| **GRUPPO VEGE** | 16 | 530.6 | 445.6 | 217.2 | 33.3% | -85.0 | 🟡 MODERATE (Acceptable) |
| **REWE** | 14 | 228.1 | 228.0 | 96.0 | 34.8% | -0.1 | 🟡 MODERATE (Acceptable) |
| **GRUPPO CARREFOUR ITALIA SPA** | 12 | 235.5 | 189.4 | 95.7 | 34.4% | -46.1 | 🟡 MODERATE (Acceptable) |
| **SELEX COMMERCIALE** | 12 | 780.7 | 645.8 | 204.3 | 25.1% | -134.9 | 🟡 MODERATE (Acceptable) |
| **CRAI** | 8 | 312.0 | 199.2 | 121.6 | 27.9% | -112.8 | 🟡 MODERATE (Acceptable) |
| **ESSELUNGA** | 7 | 631.6 | 536.2 | 129.2 | 16.2% | -95.5 | 🟢 PERFORMS WELL (Accurate) |
| **AGORA' NETWORK SCARL** | 3 | 487.8 | 499.7 | 201.9 | 33.3% | +11.8 | 🟡 MODERATE (Acceptable) |
| **DESPAR SERVIZI** | 3 | 356.0 | 372.6 | 95.8 | 30.7% | +16.6 | 🟡 MODERATE (Acceptable) |
| **GRUPPO PAM** | 2 | 427.4 | 640.8 | 258.7 | 49.0% | +213.3 | 🔴 UNDERPERFORMING (Action Needed) |
| **ASPIAG** | 1 | 174.7 | 276.1 | 101.4 | 58.0% | +101.4 | 🔴 UNDERPERFORMING (Action Needed) |

### Standardized Insegna / Banner
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **UNKNOWN** | 135 | 489.9 | 364.8 | 212.2 | 37.4% | -125.2 | 🟡 MODERATE (Acceptable) |
| **CONAD** | 46 | 726.5 | 610.8 | 315.6 | 36.4% | -115.7 | 🟡 MODERATE (Acceptable) |
| **CONAD CITY** | 24 | 458.2 | 612.8 | 307.1 | 55.4% | +154.6 | 🔴 UNDERPERFORMING (Action Needed) |
| **PENNY** | 14 | 228.1 | 228.0 | 96.0 | 34.8% | -0.1 | 🟡 MODERATE (Acceptable) |
| **COOP** | 13 | 600.7 | 552.4 | 197.3 | 32.1% | -48.3 | 🟡 MODERATE (Acceptable) |
| **DECO' SUPERMERCATI** | 8 | 268.3 | 220.8 | 93.3 | 39.2% | -47.6 | 🔴 UNDERPERFORMING (Action Needed) |
| **CARREFOUR EXPRESS** | 7 | 205.3 | 116.8 | 96.9 | 43.4% | -88.5 | 🔴 UNDERPERFORMING (Action Needed) |
| **COOP.FI** | 6 | 3536.5 | 1373.2 | 2163.3 | 52.9% | -2163.3 | 🔴 UNDERPERFORMING (Action Needed) |
| **INCOOP** | 6 | 564.7 | 663.6 | 468.4 | 60.0% | +98.9 | 🔴 UNDERPERFORMING (Action Needed) |
| **SPAZIO CONAD** | 6 | 973.8 | 986.3 | 387.2 | 32.4% | +12.5 | 🟡 MODERATE (Acceptable) |
| **BENNET** | 5 | 767.0 | 701.8 | 415.2 | 33.8% | -65.2 | 🟡 MODERATE (Acceptable) |
| **CONAD SUPERSTORE** | 5 | 1222.8 | 773.9 | 448.8 | 18.8% | -448.8 | 🟡 MODERATE (Acceptable) |
| **ESSELUNGA SUPERSTORE** | 5 | 673.0 | 579.0 | 141.3 | 16.2% | -94.0 | 🟢 PERFORMS WELL (Accurate) |
| **CARREFOUR MARKET** | 4 | 209.6 | 180.7 | 72.2 | 24.4% | -28.9 | 🟢 PERFORMS WELL (Accurate) |
| **CRAI** | 4 | 236.9 | 194.0 | 51.2 | 24.1% | -42.9 | 🟢 PERFORMS WELL (Accurate) |
| **CRAI EXTRA** | 4 | 387.1 | 204.4 | 191.9 | 51.2% | -182.7 | 🔴 UNDERPERFORMING (Action Needed) |
| **EKOM DISCOUNT** | 3 | 487.8 | 499.7 | 201.9 | 33.3% | +11.8 | 🟡 MODERATE (Acceptable) |
| **IPERCOOP** | 3 | 999.9 | 912.0 | 290.6 | 33.8% | -87.9 | 🟡 MODERATE (Acceptable) |
| **DESPAR** | 2 | 264.8 | 370.0 | 105.2 | 44.4% | +105.2 | 🔴 UNDERPERFORMING (Action Needed) |
| **ESSELUNGA** | 2 | 528.3 | 429.2 | 99.1 | 18.3% | -99.1 | 🟢 PERFORMS WELL (Accurate) |
| **EUROSPAR** | 2 | 356.5 | 326.9 | 89.2 | 28.7% | -29.6 | 🟡 MODERATE (Acceptable) |
| **MERCATO'** | 2 | 1255.2 | 1193.9 | 61.3 | 5.0% | -61.3 | 🟢 PERFORMS WELL (Accurate) |
| **MERCATO' EXTRA** | 2 | 1615.7 | 1122.1 | 493.5 | 28.9% | -493.5 | 🟡 MODERATE (Acceptable) |
| **N. D.** | 2 | 567.4 | 313.1 | 254.3 | 48.9% | -254.3 | 🔴 UNDERPERFORMING (Action Needed) |
| **SPESA FACILE** | 2 | 1044.1 | 362.3 | 681.8 | 49.5% | -681.8 | 🔴 UNDERPERFORMING (Action Needed) |
| **TIGRE** | 2 | 377.6 | 308.0 | 69.6 | 20.8% | -69.6 | 🟢 PERFORMS WELL (Accurate) |
| **TODIS** | 2 | 466.8 | 1150.2 | 683.4 | 146.9% | +683.4 | 🔴 UNDERPERFORMING (Action Needed) |
| **CARREFOUR** | 1 | 550.1 | 731.9 | 181.8 | 33.0% | +181.8 | 🟡 MODERATE (Acceptable) |
| **COOP SUPERSTORE** | 1 | 788.4 | 866.8 | 78.4 | 9.9% | +78.4 | 🟢 PERFORMS WELL (Accurate) |
| **DODECA** | 1 | 543.8 | 394.5 | 149.3 | 27.5% | -149.3 | 🟡 MODERATE (Acceptable) |
| **EUROESSE** | 1 | 228.1 | 200.4 | 27.7 | 12.1% | -27.7 | 🟢 PERFORMS WELL (Accurate) |
| **FAMILA** | 1 | 933.4 | 693.0 | 240.4 | 25.8% | -240.4 | 🟡 MODERATE (Acceptable) |
| **FAMILA SUPERSTORE** | 1 | 231.9 | 175.3 | 56.6 | 24.4% | -56.6 | 🟢 PERFORMS WELL (Accurate) |
| **GRANDI MAGAZZINI SUPERCONTI** | 1 | 1061.2 | 373.4 | 687.8 | 64.8% | -687.8 | 🔴 UNDERPERFORMING (Action Needed) |
| **GULLIVER SUPERETTE** | 1 | 266.5 | 197.8 | 68.7 | 25.8% | -68.7 | 🟡 MODERATE (Acceptable) |
| **IPER DEM** | 1 | 421.1 | 797.1 | 376.0 | 89.3% | +376.0 | 🔴 UNDERPERFORMING (Action Needed) |
| **IPER FUTURA** | 1 | 1736.6 | 1259.8 | 476.8 | 27.5% | -476.8 | 🟡 MODERATE (Acceptable) |
| **PAM SUPERSTORE** | 1 | 275.4 | 230.0 | 45.4 | 16.5% | -45.4 | 🟢 PERFORMS WELL (Accurate) |
| **PANORAMA** | 1 | 579.5 | 1051.6 | 472.1 | 81.5% | +472.1 | 🔴 UNDERPERFORMING (Action Needed) |
| **SI' CON TE SUPERSTORE** | 1 | 821.6 | 401.1 | 420.5 | 51.2% | -420.5 | 🔴 UNDERPERFORMING (Action Needed) |
| **SUPER GULLIVER** | 1 | 196.6 | 237.5 | 40.9 | 20.8% | +40.9 | 🟢 PERFORMS WELL (Accurate) |
| **TUDAY** | 1 | 718.6 | 265.9 | 452.7 | 63.0% | -452.7 | 🔴 UNDERPERFORMING (Action Needed) |

### Nielsen Macro Area
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **UNKNOWN** | 135 | 489.9 | 364.8 | 212.2 | 37.4% | -125.2 | 🟡 MODERATE (Acceptable) |
| **3.0** | 89 | 830.2 | 591.7 | 451.0 | 49.0% | -238.5 | 🔴 UNDERPERFORMING (Action Needed) |
| **1.0** | 45 | 582.2 | 537.2 | 202.1 | 32.1% | -45.1 | 🟡 MODERATE (Acceptable) |
| **4.0** | 40 | 508.9 | 510.3 | 183.2 | 27.5% | +1.4 | 🟡 MODERATE (Acceptable) |
| **2.0** | 22 | 564.9 | 568.5 | 245.0 | 29.3% | +3.7 | 🟡 MODERATE (Acceptable) |

### Italian Region (Regione)
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Campania** | 46 | 420.8 | 296.4 | 165.9 | 26.9% | -124.4 | 🟡 MODERATE (Acceptable) |
| **Toscana** | 41 | 975.4 | 665.6 | 528.1 | 43.4% | -309.8 | 🔴 UNDERPERFORMING (Action Needed) |
| **Lazio** | 38 | 533.6 | 495.6 | 309.8 | 52.5% | -38.0 | 🔴 UNDERPERFORMING (Action Needed) |
| **Lombardia** | 32 | 517.9 | 456.3 | 173.9 | 30.5% | -61.6 | 🟡 MODERATE (Acceptable) |
| **Piemonte** | 26 | 710.4 | 423.0 | 345.9 | 39.4% | -287.3 | 🔴 UNDERPERFORMING (Action Needed) |
| **Emilia-Romagna** | 21 | 651.4 | 589.3 | 259.4 | 25.8% | -62.1 | 🟡 MODERATE (Acceptable) |
| **Calabria** | 17 | 618.5 | 728.3 | 306.7 | 58.0% | +109.9 | 🔴 UNDERPERFORMING (Action Needed) |
| **Sardegna** | 17 | 536.9 | 432.4 | 204.7 | 30.7% | -104.5 | 🟡 MODERATE (Acceptable) |
| **Marche** | 16 | 678.2 | 427.1 | 374.3 | 50.2% | -251.1 | 🔴 UNDERPERFORMING (Action Needed) |
| **Umbria** | 13 | 678.5 | 374.4 | 318.4 | 38.6% | -304.1 | 🔴 UNDERPERFORMING (Action Needed) |
| **Abruzzo** | 11 | 407.1 | 360.8 | 152.6 | 37.4% | -46.3 | 🟡 MODERATE (Acceptable) |
| **Sicilia** | 11 | 410.3 | 351.9 | 106.5 | 19.4% | -58.4 | 🟢 PERFORMS WELL (Accurate) |
| **Liguria** | 9 | 569.0 | 563.9 | 306.8 | 52.3% | -5.1 | 🔴 UNDERPERFORMING (Action Needed) |
| **Puglia** | 9 | 438.7 | 355.4 | 170.0 | 38.1% | -83.3 | 🔴 UNDERPERFORMING (Action Needed) |
| **Veneto** | 9 | 642.6 | 684.8 | 158.3 | 17.6% | +42.2 | 🟢 PERFORMS WELL (Accurate) |
| **Friuli-Venezia Giulia** | 7 | 491.8 | 510.8 | 98.7 | 18.0% | +19.0 | 🟢 PERFORMS WELL (Accurate) |
| **Basilicata** | 2 | 980.5 | 767.8 | 264.0 | 25.2% | -212.7 | 🟡 MODERATE (Acceptable) |
| **Molise** | 2 | 301.9 | 254.2 | 217.2 | 92.6% | -47.6 | 🔴 UNDERPERFORMING (Action Needed) |
| **Trentino-Alto Adige** | 2 | 254.0 | 262.2 | 39.5 | 16.7% | +8.1 | 🟢 PERFORMS WELL (Accurate) |
| **Valle d'Aosta** | 2 | 495.2 | 704.8 | 209.5 | 42.3% | +209.5 | 🔴 UNDERPERFORMING (Action Needed) |

### Catchment Population Density
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Suburban / Moderate (200-800 hab/km²)** | 122 | 691.7 | 521.4 | 322.9 | 38.3% | -170.2 | 🔴 UNDERPERFORMING (Action Needed) |
| **Dense Urban (800-2,500 hab/km²)** | 90 | 609.5 | 507.7 | 296.2 | 32.9% | -101.8 | 🟡 MODERATE (Acceptable) |
| **Rural / Low Density (< 200 hab/km²)** | 78 | 478.3 | 396.2 | 203.2 | 38.5% | -82.1 | 🔴 UNDERPERFORMING (Action Needed) |
| **Metropolitan (> 2,500 hab/km²)** | 40 | 553.4 | 465.4 | 213.1 | 33.5% | -88.0 | 🟡 MODERATE (Acceptable) |
| **Unknown** | 1 | 334.0 | 168.0 | 166.0 | 49.7% | -166.0 | 🔴 UNDERPERFORMING (Action Needed) |

### Actual Store Volume Tier
| Category / Factor Value | Stores | Actual Mean | NN Pred Mean | NN MAE | MedAPE | Bias | Diagnostic Rating |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 3: Moderate (300 - 600 b/day)** | 110 | 427.4 | 437.6 | 182.3 | 33.5% | +10.2 | 🟡 MODERATE (Acceptable) |
| **Tier 4: Low Volume (< 300 b/day)** | 104 | 202.4 | 277.5 | 121.6 | 33.4% | +75.1 | 🟡 MODERATE (Acceptable) |
| **Tier 2: High (600 - 1,200 b/day)** | 87 | 810.0 | 609.7 | 303.4 | 38.1% | -200.4 | 🔴 UNDERPERFORMING (Action Needed) |
| **Tier 1: Exceptional (> 1,200 b/day)** | 30 | 2015.8 | 965.5 | 1050.3 | 44.3% | -1050.3 | 🔴 UNDERPERFORMING (Action Needed) |

## 3. Top Predictive Factors (Feature Attribution - LightGBM Gain)
| Rank | Factor / Feature | Gain Contribution % | Total Tree Splits |
| :---: | :--- | :---: | :---: |
| 1 | `Provincia` | 30.33% | 2379 |
| 2 | `Insegna_Standardized` | 18.4% | 857 |
| 3 | `TipoMacchina` | 7.58% | 572 |
| 4 | `nielsen_numero_casse` | 4.29% | 141 |
| 5 | `population_density_log` | 3.27% | 435 |
| 6 | `nielsen_superficie_vendita_log` | 3.16% | 397 |
| 7 | `nielsen_gruppo` | 3.13% | 207 |
| 8 | `surface_log` | 2.85% | 426 |
| 9 | `nearest_coripet_km_log` | 2.68% | 486 |
| 10 | `population_log` | 2.62% | 387 |
| 11 | `month_cos` | 2.44% | 326 |
| 12 | `PIL_log` | 2.25% | 316 |
| 13 | `is_holiday` | 2.18% | 167 |
| 14 | `dow_sin` | 2.09% | 332 |
| 15 | `PIL_per_capita` | 2.06% | 388 |
| 16 | `Longitudine` | 1.54% | 266 |
| 17 | `coripet_rvm_count_5km` | 1.44% | 272 |
| 18 | `grocery_total_stores` | 1.42% | 192 |
| 19 | `Latitudine` | 1.16% | 233 |
| 20 | `grocery_supermarket_count` | 1.03% | 164 |
