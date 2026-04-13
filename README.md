# IDS 705 Final Project: Bank Marketing Subscription Prediction

This project builds and evaluates machine learning models to predict whether a bank client will subscribe to a term deposit using the Bank Marketing dataset.

The modeling strategy compares:
- a single **global model** trained on all clients, and
- a **segmented approach** that trains cluster-specific models after customer segmentation.

The end goal is not only predictive performance, but also business impact using a cost-benefit (expected profit) framework.

## Project Workflow

The analysis is organized as a notebook-first pipeline:

1. `00_initial_eda.ipynb`  
   Initial exploratory data analysis and baseline understanding of the raw dataset.
2. `01_eda.ipynb`  
   Deeper feature exploration and interpretation for downstream modeling choices.
3. `02_preprocessing.ipynb`  
   Feature engineering, data cleaning, train/validation/test construction, and clustering.
4. `03_training.ipynb`  
   Model training, calibration, segmentation-aware modeling, and threshold selection.
5. `04_evaluation.ipynb`  
   Final test-set evaluation comparing global vs segmented strategies with business metrics.

## Repository Structure

```text
.
|-- data/
|   |-- bank-full.csv
|   |-- *.parquet (processed splits and cluster-specific datasets)
|-- models/
|   |-- global_model.joblib
|   |-- global_calibrator.joblib
|   |-- global_metadata.joblib
|   |-- global_threshold.joblib
|   |-- c*_model.joblib
|   |-- c*_calibrator.joblib
|   |-- c*_metadata.joblib
|   |-- cluster_thresholds.joblib
|-- utils/
|   `-- utils.py
|-- 00_initial_eda.ipynb
|-- 01_eda.ipynb
|-- 02_preprocessing.ipynb
|-- 03_training.ipynb
|-- 04_evaluation.ipynb
`-- requirements.txt
```

## Environment Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

Start Jupyter and run notebooks in order:

```bash
jupyter notebook
```

Recommended execution order:
1. `00_initial_eda.ipynb`
2. `01_eda.ipynb`
3. `02_preprocessing.ipynb`
4. `03_training.ipynb`
5. `04_evaluation.ipynb`

Running in sequence ensures all intermediate datasets, trained models, calibrators, and thresholds are available for downstream notebooks.

## Data and Artifacts

- Raw source data: `data/bank-full.csv`
- Engineered datasets are written as `.parquet` files under `data/`
- Trained models and related metadata are saved under `models/` as `.joblib` files
- Utility functions for loading datasets and evaluating metrics are in `utils/utils.py`

## Evaluation Focus

The final evaluation includes:
- Classification metrics (AUC-PR, Precision, Recall, F1)
- Operational targeting rate (`% Contacted`)
- Expected profit using a business decision rule

This allows comparison of model quality and practical campaign value, both overall and per customer cluster.

## Key Results

- The project benchmarks a single global model against a segmented (cluster-specific) modeling strategy.
- Segmentation improves performance for some clusters, while the global model remains stronger in others.
- Business evaluation shows a clear precision-recall-contact-rate trade-off between the two strategies.
- Final model selection is based on both predictive quality and expected profit, not accuracy alone.