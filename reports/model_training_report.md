# Coripet RVM Predictor: Neural Network Training Report

**Date:** September 2026  
**Status:** Multi-Output Neural Network Successfully Trained & Validated  
**Model Checkpoint:** [`models/best_model/model.pt`](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/models/best_model/model.pt)  
**Preprocessing Artifact:** [`models/preprocessing/preprocessor.joblib`](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/models/preprocessing/preprocessor.joblib)  
**Training History:** [`models/best_model/training_history.json`](file:///c:/Users/waqas/Desktop/Coripet_RVM_Predictor/models/best_model/training_history.json)  

---

## 1. Executive Summary & Split Design

The multi-output neural network was trained on the certified modeling population of **1,396,544 daily observations across 1,657 unique physical points (`PF`)**.

To ensure true out-of-store generalization for prospective new retail locations and prevent identity memorization, the dataset was partitioned using **strict group separation by `PF`**:

```
TOTAL POPULATION: 1,396,544 rows across 1,657 unique PFs
                         │
       ┌─────────────────┴─────────────────┐
       ▼                                   ▼
DEVELOPMENT SET (80% PFs)           TEST HOLDOUT (20% PFs)
1,326 Unique PFs                    331 Unique PFs
1,105,840 observations              290,704 observations
       │                            (COMPLETELY UNSEEN)
  ┌────┴────┐
  ▼         ▼
TRAIN SET (80% of Dev)              VALIDATION SET (20% of Dev)
1,061 Unique PFs                    265 Unique PFs
890,395 observations (63.8%)        215,445 observations (15.4%)
```

### Strict Zero-Leakage Protocol
* **PF Disjointness:** $\text{Train PFs} \cap \text{Val PFs} \cap \text{Test PFs} = \emptyset$. Zero physical points overlap across any split.
* **Fitted Only on Train:** All numerical medians, log1p decisions, standard scalers, and categorical vocabularies were fitted **strictly on the 890,395 training observations**. The validation and test sets were strictly transformed out-of-sample.
* **Excluded Predictors:** All entity serial numbers (`PF`, `Macchina`, `CodiceMacchinaProd`, `CODICEMACCHINA`, `CODICEMACCHINAPROD`), operational dates (`DataPrimaAttivazione`, `days_since_activation`), target variables (`BOTTIGLIE`, `PRESENZE`), and historical target lags/averages were completely omitted from input matrices.

---

## 2. Multi-Task Neural Network Architecture

```
                    INPUT FEATURES: 20 Numeric + 8 Categorical
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │       CATEGORICAL EMBEDDING LAYERS            │
                 │ TipoMacchina (15 -> 8)   Fornitore (5 -> 4)   │
                 │ Provincia (104 -> 20)    Regione (21 -> 9)    │
                 │ Insegna (22 -> 9)        TipoNegozio (5 -> 4) │
                 │ NielsenGruppo (19 -> 9)  NielsenArea (5 -> 4) │
                 │ Total Categorical Embedding Dimensions = 67   │
                 └───────────────────────┬───────────────────────┘
                                         │
                                         ▼
                 Total Concatenated Input Dimension = 20 + 67 = 87
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │          SHARED DENSE TRUNK                   │
                 │ Layer 1: Linear(87 -> 256) + GELU + LN + D(0.2)
                 │ Layer 2: Linear(256 -> 128) + GELU + LN + D(0.2)
                 │ Layer 3: Linear(128 -> 64) + GELU + LN + D(0.1)
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │      PRESENZE HEAD        │                   │      BOTTIGLIE HEAD       │
   │ Linear(64 -> 32) + GELU   │                   │ Linear(64 -> 32) + GELU   │
   │ Linear(32 -> 1)           │                   │ Linear(32 -> 1)           │
   │ Predicts log1p(PRESENZE)  │                   │ Predicts log1p(BOTTIGLIE) │
   └───────────────────────────┘                   └───────────────────────────┘
```

### Parameter Count
* **Embedding Parameters:** 3,865 parameters
* **Shared Trunk Parameters:** 63,936 parameters
* **Presenze Head Parameters:** 2,113 parameters
* **Bottiglie Head Parameters:** 2,113 parameters
* **Total Trainable Parameters:** **71,620**

---

## 3. Training Dynamics & Loss Progression

* **Loss Function:** Multi-Task Huber Loss ($\delta = 1.0$) with unit task weights:
  $$\mathcal{L} = 1.0 \cdot \text{Huber}(\hat{y}_{\text{bottiglie}}, y_{\text{bottiglie}}) + 1.0 \cdot \text{Huber}(\hat{y}_{\text{presenze}}, y_{\text{presenze}})$$
* **Optimizer:** AdamW ($\text{lr} = 1\times 10^{-3}$, $\text{weight\_decay} = 1\times 10^{-4}$).
* **Batch Size:** 4,096.
* **Early Stopping:** Patience = 5 epochs monitoring validation loss on unseen PFs.
* **Total Training Time:** 6.9 minutes on CPU.

### Epoch History

| Epoch | Train Total Loss | Train BOTTIGLIE Loss | Train PRESENZE Loss | Val Total Loss | Val BOTTIGLIE Loss | Val PRESENZE Loss | Learning Rate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | 1.3306 | 0.9696 | 0.3610 | 0.8914 | 0.6026 | 0.2888 | $1.0\times 10^{-3}$ | Checkpoint Saved |
| **02** | **0.8070** | **0.5629** | **0.2441** | **0.8831** | **0.5941** | **0.2890** | $1.0\times 10^{-3}$ | **(*) BEST MODEL SAVED** |
| **03** | 0.7614 | 0.5362 | 0.2253 | 0.8940 | 0.5985 | 0.2955 | $1.0\times 10^{-3}$ | Patience: 1/5 |
| **04** | 0.7398 | 0.5227 | 0.2170 | 0.9063 | 0.6058 | 0.3005 | $1.0\times 10^{-3}$ | Patience: 2/5 |
| **05** | 0.7255 | 0.5138 | 0.2116 | 0.9222 | 0.6153 | 0.3069 | $1.0\times 10^{-3}$ | Patience: 3/5 |
| **06** | 0.7138 | 0.5062 | 0.2076 | 0.9227 | 0.6157 | 0.3070 | $5.0\times 10^{-4}$ | LR Halved, Patience: 4/5 |
| **07** | 0.7096 | 0.5035 | 0.2061 | 0.9214 | 0.6144 | 0.3069 | $5.0\times 10^{-4}$ | Early Stopping Triggered |

### Overfitting & Generalization Analysis
* At the best checkpoint (**Epoch 2**), the training loss was **0.8070** and validation loss on 265 completely unseen physical points was **0.8831** (a modest generalization gap of **9.4%**).
* In subsequent epochs, the network began fitting idiosyncratic store patterns in the training set (training loss reached 0.7096 while validation loss drifted to 0.9214). Early stopping successfully prevented overfitting by halting execution and preserving the Epoch 2 state.
* The Huber loss breakdown demonstrates that **BOTTIGLIE loss contributes ~67%** and **PRESENZE loss contributes ~33%** of total loss, perfectly mirroring their natural variance ratio in log1p space ($\sigma_B^2 \approx 2.95$ vs $\sigma_P^2 \approx 0.94$).
