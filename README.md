# Coripet RVM Prospective Location Predictor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20Production-green.svg)](#system-architecture)
[![Model](https://img.shields.io/badge/Model-PyTorch%20Multi--Task%20NN-orange.svg)](#neural-network-architecture)

A production-oriented machine learning decision support system for **Coripet Reverse Vending Machines (RVMs)**.

Given prospective site inputs (Store Name, Address, Italian Postal Code / CAP, and RVM Machine Type), the application predicts expected daily/monthly operational volumes, computes location suitability benchmarks, assigns business potential tiers, and quantifies prediction confidence.

---

## 🎯 Business Objectives

When evaluating prospective retail locations for Coripet RVM installations, the system delivers:

1. **Expected PRESENZE / day**: Projected unique individuals using the RVM per day.
2. **Expected BOTTIGLIE / day**: Projected PET bottles collected per day.
3. **Expected BOTTIGLIE / month**: Standardized monthly collection volume ($30 \times \text{daily bottles}$).
4. **Bottles per person**: Operational interaction efficiency ratio ($\text{BOTTIGLIE} / \text{PRESENZE}$).
5. **Location Suitability Score (%)**: Composite index ($0 - 100\%$) bench-marked against operational network quantiles.
6. **Business Recommendation**: Categorical viability tier (`HIGH POTENTIAL`, `MEDIUM POTENTIAL`, `LOW POTENTIAL`).
7. **Prediction Confidence**: Reliability indicator (`HIGH`, `MEDIUM`, `LOW`) derived from domain familiarity and epistemic uncertainty.

### Realistic User Input Specification
At inference time for a prospective store, only four realistic variables are required:
* **Store name** (e.g., `Esselunga Milano Certosa`, `Conad Superstore`)
* **Address** (e.g., `Via delle Forze Armate 12`)
* **Postal code (CAP)** (e.g., `20147`, `00185`)
* **Machine type** (e.g., `Aneco Super`, `Eurven City`)

All intermediate features (Italian administrative hierarchy `Comune`, `Provincia`, `Regione`, approximate coordinates, and retail chain extraction) are derived automatically.

---

## 🏗️ System Architecture

```text
Coripet_RVM_Predictor/
├── configs/
│   ├── config.yaml                    # Master pipeline & model hyperparameters
│   └── business_rules.yaml            # Calibrated scoring weights & tier definitions
├── data/
│   ├── raw/                           # Destination for raw CSV/Excel datasets (ignored in git)
│   ├── processed/                     # Cleaned, joined feature stores
│   └── external/                      # Italian geographic & administrative lookup tables
├── models/
│   ├── saved_models/                  # PyTorch model checkpoints (.pt / .pth)
│   └── preprocessors/                 # Serialized feature transformers & scalers (.joblib)
├── scripts/
│   └── inspect_datasets.py            # Automated dataset schema & distribution validator
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Dataset ingestion & PF == PuntoFisico merging
│   │   ├── cleaner.py                 # Sanitization, CAP padding (5 digits), coordinate parsing
│   │   └── splitter.py                # Time-aware split & GroupKFold (PuntoFisico) splitters
│   ├── features/
│   │   ├── __init__.py
│   │   ├── geocoding.py               # Italian CAP prefix map & administrative hierarchy lookup
│   │   ├── store_features.py          # Brand/chain extraction & machine type normalizer
│   │   └── preprocessor.py            # Tabular preprocessor (Embeddings + RobustScaler)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_net.py              # PyTorch Multi-Task Tabular NN (Entity Embeddings + Heads)
│   │   ├── dataset.py                 # PyTorch TabularDataset wrapper
│   │   ├── trainer.py                 # Reproducible training loop with early stopping & scheduler
│   │   └── metrics.py                 # Operational regression metrics (MAE, RMSE, WAPE, MAPE, R2)
│   ├── business/
│   │   ├── __init__.py
│   │   ├── scoring.py                 # Suitability score calculation (0 - 100%)
│   │   ├── recommender.py             # Business potential tier classification
│   │   └── confidence.py              # Domain coverage & MC-Dropout uncertainty assessor
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py               # End-to-end inference orchestrator & reporter
│   ├── ui/
│   │   ├── __init__.py
│   │   └── app.py                     # Interactive Streamlit Web Application
│   └── train.py                       # CLI orchestrator for training & checkpointing
├── tests/
│   ├── __init__.py
│   └── test_architecture.py           # Unit tests for data, models, business rules, and inference
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning & Neural Network Design

### Multi-Task Tabular Neural Network (`CoripetMultiTaskNN`)
Rather than training disconnected single-target models, we utilize a **Multi-Task Tabular Neural Network**:
* **Categorical Feature Processing**: Discrete inputs (`retail_chain`, `tipo_macchina`, `provincia`, `regione`, `cap`) are passed through individual `nn.Embedding` layers (Entity Embeddings). Unseen categories at inference time are mapped to dedicated `<UNK>` index `0`.
* **Continuous Feature Normalization**: Numerical features (`latitudine`, `longitudine`) are normalized via `nn.BatchNorm1d`.
* **Shared Representation Trunk**: Dense residual layers with `GELU` activation, `BatchNorm`, and `Dropout` learn general location footfall and volume dynamics.
* **Specialized Multi-Task Regression Heads**:
  - `presenze_head`: Outputs expected daily footfall / active users.
  - `bottiglie_head`: Outputs expected daily PET bottle collection.
  - Both heads terminate with `nn.Softplus()` to mathematically enforce non-negative physical counts.

### Preventing Data Leakage
1. **Realistic Inference Inputs**: The model strictly avoids operational variables that are unavailable before installation (such as machine serial codes, historical meter counts, or dynamic operational flags).
2. **Time-Aware Validation**: Daily operational logs (`GIORNO`) are split chronologically (train on earlier periods, validate/test on future periods) to test temporal stability.
3. **Group Store Holdouts**: Store locations (`PF` / `PuntoFisico`) are grouped so that validation/test sets contain stores *completely unseen during training*, rigorously simulating cold-start predictions for new stores.
4. **Preprocessor Leakage Guard**: All encoders, vocabularies, and scalers are fitted **only on the training fold**.

### Empirical Calibration (No Invented Thresholds)
Business scoring benchmarks in `SuitabilityScorer` are calibrated dynamically from the training dataset's empirical percentiles (e.g. 90th percentile of volume and footfall across the operational network), ensuring that score distributions reflect real operational capabilities.

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Create and activate a virtual environment, then install dependencies:
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Add Datasets
Place your historical datasets into the `data/raw/` folder:
* Operational log: `data/raw/daily_bottles.csv` (contains `CLIENTE`, `GIORNO`, `BOTTIGLIE`, `PRESENZE`, `PF`, `CODICEMACCHINAPROD`, etc.)
* Store master data: `data/raw/CaricaViewStoricoPuntoFisico.csv` (contains `PuntoFisico`, `Indirizzo`, `CAP`, `Comune`, `Provincia`, `Regione`, `Latitudine`, `Longitudine`, `TipoMacchina`, `Filiera`, etc.)
* Nielsen Retail Census: `data/raw/filenielsen_1203.csv`
* Macroeconomic & Census: `data/raw/PIL.csv`, `data/raw/population.csv`, `data/raw/surface.csv`
* Commercial Rewards: `data/raw/Premialità-Rewards.xlsx`

*(Master file paths and settings are configurable in `configs/config.yaml`).*

### 3. Inspect Datasets & Data Relationships
Run the automated schema inspector before training:
```bash
python scripts/inspect_datasets.py
```
This inspects:
- Row counts, column types, and missing value rates
- Match percentage on the join key (`PF` <-> `PuntoFisico`)
- Operational date range
- Target distribution percentiles (mean, std, min, 25%, 50%, 75%, 90%, max)

### 4. Train the Neural Network
Execute the training pipeline:
```bash
python src/train.py
```
This will:
1. Clean and merge operational and store records.
2. Calibrate business scoring benchmarks from empirical quantiles.
3. Split data chronologically and by store group.
4. Fit and serialize preprocessors to `models/preprocessors/preprocessor.joblib`.
5. Train the PyTorch Multi-Task NN with early stopping and learning rate scheduling.
6. Save the best checkpoint to `models/saved_models/best_coripet_model.pt`.
7. Output comprehensive evaluation metrics (`MAE`, `RMSE`, `WAPE`, `MAPE`, `R2`).

### 5. Launch the Web Application
Launch the Streamlit interactive dashboard:
```bash
streamlit run src/ui/app.py
```
The interface allows business analysts to:
- Enter store name, address, 5-digit CAP, and machine type.
- View interactive gauge indicators, KPI summary cards, and the exact formatted **LOCATION ANALYSIS** summary.

### 6. Run Test Suite
Verify pipeline integrity and component contracts:
```bash
pytest tests/test_architecture.py -v
```

---

## 📊 Output Format Example

```text
LOCATION ANALYSIS

Expected PRESENZE/day: 25
Expected BOTTIGLIE/day: 735
Expected BOTTIGLIE/month: 22,050
Bottles per person: 29.4
Location suitability: 87%
Recommendation: HIGH POTENTIAL
Confidence: HIGH
```

---

## 📋 Current Status & Next Steps

* [x] **Project Architecture & Modular Scaffold**: Complete.
* [x] **Data Ingestion, Cleaning & Time-Aware Splitters**: Complete.
* [x] **Feature Engineering & Italian Administrative Geo-Enrichment**: Complete.
* [x] **PyTorch Multi-Task Tabular Neural Network**: Complete.
* [x] **Business Scoring, Recommendation & Confidence Engine**: Complete.
* [x] **Streamlit Web UI & CLI Orchestrators**: Complete.
* [x] **Full Dataset Ingestion & Historical Matching**: Complete (1,889,479 operational records across 2,142 PFs merged with store history and Nielsen census).
* [x] **Model Training & Benchmark Suite**: Complete (Trained Multi-Task NN + LightGBM ensemble evaluated on 331 untouched holdout stores and 43 nationwide benchmark locations).
* [x] **Commercial Rewards & Filiera Intelligence**: Complete (100% network match with Premialità-Rewards agreements, including Esselunga Omaggio resolution).
