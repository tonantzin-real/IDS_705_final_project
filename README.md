# IDS 705 Final Project: Bank Marketing Subscription Prediction

This project builds and evaluates machine learning models to predict whether a bank client will subscribe to a term deposit using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

The modeling strategy compares:

- a single **global** LightGBM model trained on all clients, then calibrated with an optimal decision threshold, and  
- a **segmented** approach: K-Means customer segments (k = 4), then one LightGBM + calibrator + profit-optimal threshold per segment.

The end goal is not only predictive performance, but **business impact** using a cost–benefit framework (expected profit from contacting customers vs. call cost).

---

## Notebooks: what each one actually does

### `00_initial_eda.ipynb` — first-pass exploration

- Loads the raw CSV and confirms basic shape (**45,211 × 17**).
- **Target (`y`)**: class balance and intuition for imbalance-aware modeling later.
- **Categorical features**: quick frequency / relationship-to-`y` views.
- **Numeric features**: distributions, summaries, and simple comparisons by `y`.
- Purpose: orient the team to the raw table before the structured EDA in `01_eda.ipynb`.

### `01_eda.ipynb` — structured EDA that drives preprocessing

- Formal objectives: schema, **implicit missingness** (`"unknown"` categories, `pdays = -1` sentinels), outliers / skew, categorical vs. `y`, **class imbalance**, and **leakage** (especially `duration`, known to be post-call).
- Uses a consistent plotting style and documents decisions that feed `02_preprocessing.ipynb`.
- **Output artifact**: writes `data/01_raw_data.parquet` (cleaned / standardized raw table used as a convenient starting point for preprocessing).

### `02_preprocessing.ipynb` — feature engineering, splits, clustering

- Ingests the preprocessed raw table (`data/02_raw_data.parquet`; produced in this notebook from the engineered pipeline—run cells in order).
- **Feature engineering**: bucketed numerics (e.g. age, balance, campaign, pdays recency), binary indicators from campaign history and product flags, and review of engineered columns vs. subscription rate.
- Saves **`data/02_full_eng_data.parquet`**: fully engineered tabular dataset before the final modeling matrix.
- Builds **`X` / `y`**: drops the label from features; saves `data/02_X_raw.parquet` and `data/02_y_raw.parquet`.
- **Stratified split**: 80% train+validation vs. 20% test (`random_state` fixed); saves pre-clustering split features and labels as `data/02_X_train_val_raw.parquet`, `data/02_y_train_val_raw.parquet`, `data/02_X_test_raw.parquet`, `data/02_y_test_raw.parquet`.
- **Preprocessing** (scaling / encoding): fit **only** on the train+validation portion to avoid leakage; transform test accordingly.
- **Segmentation**: subset of demographic / socioeconomic columns → optional PCA for diagnostics → **K-Means with k = 4** chosen using inertia, silhouette, Davies–Bouldin, and Calinski–Harabasz; cluster IDs are appended as column `cluster`.
- Writes **cluster-augmented** modeling matrices: `data/02_X_train_val.parquet`, `data/02_y_train_val.parquet`, `data/02_X_test.parquet`, `data/02_y_test.parquet` (features include `cluster` where applicable).
- **Second stratified split (80/20)** inside train+validation: `data/02_X_train.parquet`, `data/02_y_train.parquet`, `data/02_X_validation.parquet`, `data/02_y_validation.parquet`.
- **Per-cluster slices** for segmented training: `data/02_X_{split}_c#.parquet` and `data/02_y_{split}_c#.parquet` for `split` ∈ {`train`, `validation`, `test`} and `#` ∈ {0,…,3} (see [Data files](#data-files) for the pattern).

### `03_training.ipynb` — model bake-off, calibration, thresholds

- **Models used (validation comparison)**: logistic regression, random forest, **XGBoost**, **LightGBM**, and **CatBoost** (with imbalance-aware class weighting).
- **Global model**: train LightGBM on all training data; **isotonic calibration** on validation probabilities; save `models/global_model.joblib`, `models/global_calibrator.joblib`, `models/global_metadata.joblib` (includes the feature list used at inference).
- **Segmented models**: repeat per cluster with **cluster-specific** `scale_pos_weight`; each segment gets its own `models/c#_*` artifacts.
- **Business-aligned thresholds**: on the validation set, search thresholds to **maximize expected profit** under fixed `C_call` and `B_sub` (where `C_call` is cost per call and `B_sub` is benefit per successful subscription; same constants as in `04_evaluation.ipynb`); save `models/global_threshold.joblib` and `models/cluster_thresholds.joblib`.
- **Figures**: per-cluster **precision–recall curves** comparing global vs. cluster-specific models (`plot_cluster_pr_curves` in `utils/utils.py`).

### `04_evaluation.ipynb` — held-out test, global vs. segmented

- Loads test features with `cluster`, all `models/*.joblib` artifacts, and applies **frozen** validation-chosen thresholds (no test tuning).
- Reports **AUC-PR**, precision, recall, F1, **% contacted**, and **expected profit** for global vs. segmented strategies (aggregate and per cluster).
- Interprets trade-offs: coverage (recall, % contacted) vs. efficiency (precision) and profit.

---

## Repository layout — every file

### Root

| File | Purpose |
|------|---------|
| `README.md` | Project overview, file map, and results summary. |
| `requirements.txt` | Pinned Python dependencies. |
| `00_initial_eda.ipynb` | First exploratory pass on the CSV. |
| `01_eda.ipynb` | Structured EDA; writes `data/01_raw_data.parquet`. |
| `02_preprocessing.ipynb` | Engineering, splits, K-Means, parquet exports. |
| `03_training.ipynb` | Model selection, calibration, threshold tuning, saves `models/`. |
| `04_evaluation.ipynb` | Final test evaluation and business metrics. |

### `data/`

| File | Purpose |
|------|---------|
| `bank-full.csv` | Original UCI Bank Marketing full dataset (source of truth for reproducibility). |
| `01_raw_data.parquet` | Table saved from `01_eda.ipynb` for a stable “raw” baseline in parquet form. |
| `02_full_eng_data.parquet` | All engineered columns before final `X`/`y` separation. |
| `02_raw_data.parquet` | Engineered / cleaned table at the stage written in preprocessing (feeds later steps in `02_preprocessing.ipynb`). |
| `02_X_raw.parquet` | Full feature matrix (all rows) after separating from `y`. |
| `02_y_raw.parquet` | Full target vector aligned with `02_X_raw.parquet`. |
| `02_X_train_val_raw.parquet` | Features for the 80% train+validation fold **before** clustering column is attached (preprocessed numerics/categories as produced in-notebook). |
| `02_y_train_val_raw.parquet` | Targets for that train+validation fold. |
| `02_X_test_raw.parquet` | Holdout 20% features, same stage as train+validation raw. |
| `02_y_test_raw.parquet` | Holdout 20% targets. |
| `02_X_train_val.parquet` | Train+validation features **with** `cluster` assignment. |
| `02_y_train_val.parquet` | Matching labels. |
| `02_X_test.parquet` | Test features **with** `cluster` (used in evaluation). |
| `02_y_test.parquet` | Test labels. |
| `02_X_train.parquet` | Training split (from train+validation) for fitting models. |
| `02_y_train.parquet` | Training labels. |
| `02_X_validation.parquet` | Validation split for tuning, calibration, and threshold search. |
| `02_y_validation.parquet` | Validation labels. |
| `02_X_{split}_c#.parquet` | Rows of `02_X_{split}.parquet` restricted to cluster `#`, with the same feature columns (no `cluster` column in the slice used for local model training—see notebook). `split` is `train`, `validation`, or `test`. |
| `02_y_{split}_c#.parquet` | Labels aligned with each cluster slice. |
| `02_train.parquet` | **Wide** matrix (one-hot / encoded design matrix, ~63 columns) covering the train portion; includes cluster indicator columns. Useful as an alternate modeling export; the main pipeline uses the `02_X_*` / `02_y_*` pairs above. |
| `02_test.parquet` | Same as `02_train.parquet` for the test portion. |

### `models/`

| File | Purpose |
|------|---------|
| `global_model.joblib` | Trained global **LightGBM** classifier. |
| `global_calibrator.joblib` | **Isotonic regression** mapping raw scores → calibrated probabilities (fit on validation). |
| `global_metadata.joblib` | Dict with at least the **ordered feature list** expected by `global_model`. |
| `global_threshold.joblib` | Scalar probability cutoff for “contact” that maximized validation expected profit. |
| `c0_model.joblib` … `c3_model.joblib` | Cluster-specific LightGBM models. |
| `c0_calibrator.joblib` … `c3_calibrator.joblib` | Per-cluster isotonic calibrators. |
| `c0_metadata.joblib` … `c3_metadata.joblib` | Per-cluster feature lists and ancillary training metadata. |
| `cluster_thresholds.joblib` | Mapping `cluster_id → threshold` from validation profit search. |

### `utils/`

| File | Purpose |
|------|---------|
| `utils.py` | Shared helpers: `load_dataset` (reads `02_*` parquet naming convention), `evaluate_model` / `evaluate_from_probs`, `evaluate_with_profit` (profit + metrics), `plot_cluster_pr_curves`. |
| `__pycache__/utils.cpython-*.pyc` | Bytecode cache (regenerated when you import `utils`; safe to delete). |

### `catboost_info/` (generated)

| File | Purpose |
|------|---------|
| `catboost_training.json` | CatBoost run summary (parameters, iterations). |
| `learn/events.out.tfevents` | TensorBoard-compatible event log. |
| `learn_error.tsv` | Training loss per iteration. |
| `time_left.tsv` | Time-remaining estimates during training. |

---

## Environment setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run notebooks

```bash
jupyter notebook
```

**Recommended order:** `00_initial_eda.ipynb` → `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_training.ipynb` → `04_evaluation.ipynb`.

Running in sequence regenerates parquet intermediates, `models/`, and keeps evaluation consistent with training.

---

## Evaluation focus

- **Classification**: AUC-PR (average precision), precision, recall, F1.  
- **Operations**: `% Contacted` (fraction predicted positive at the chosen threshold).  
- **Business**: **Expected profit** = sum over contacted rows of (calibrated probability × subscription benefit − call cost), with `C_call = 1.0` and `B_sub = 10.0` in `04_evaluation.ipynb`.

---

## Key results

Numbers below are **reproduced from the saved outputs in `04_evaluation.ipynb`** (held-out test, frozen validation thresholds). Slight differences can appear if you re-run training with a different seed or library version.

### Aggregate test performance (global vs. segmented)

| Strategy   | AUC-PR | Precision | Recall | F1    | % Contacted | Expected profit |
|-----------|--------|-----------|--------|-------|-------------|-----------------|
| Global    | 0.420  | 0.264     | 0.715  | 0.386 | 31.63       | 5159            |
| Segmented | 0.398  | 0.272     | 0.649  | 0.384 | 27.90       | 4833            |

**Takeaway:** The global model ranks slightly better overall (AUC-PR), reaches more true subscribers (recall), and yields **higher aggregate expected profit** under this cost model, at the cost of contacting a larger share of clients. Segmentation raises precision and lowers contact rate but does not beat global profit on the full test set here.

### Per-cluster test metrics (same decision rule family as in the notebook)

| Cluster | Model     | AUC-PR | Precision | Recall | F1    |
|--------|-----------|--------|-----------|--------|-------|
| 0      | Global    | 0.478  | 0.315     | 0.789  | 0.450 |
| 0      | Segmented | 0.444  | 0.286     | 0.771  | 0.418 |
| 1      | Global    | 0.358  | 0.267     | 0.583  | 0.367 |
| 1      | Segmented | 0.307  | 0.292     | 0.430  | 0.348 |
| 2      | Global    | 0.413  | 0.235     | 0.667  | 0.347 |
| 2      | Segmented | 0.383  | 0.272     | 0.618  | 0.378 |
| 3      | Global    | 0.429  | 0.255     | 0.786  | 0.385 |
| 3      | Segmented | 0.415  | 0.257     | 0.735  | 0.381 |

**Takeaway:** Globally, recall tends to favor the **global** model; **segmented** models sometimes win on precision or F1 in a given cluster (e.g. cluster 2 F1), but the pattern is **not uniform**.

### Per-cluster expected profit and contact intensity (test)

| Cluster | Global profit | Segmented profit | Global % contacted | Segmented % contacted |
|--------|---------------|------------------|--------------------|------------------------|
| 0      | 1313          | 1319             | 35.24              | 37.90                  |
| 1      | 805           | 558              | 18.09              | 12.24                  |
| 2      | 1125          | 844              | 31.71              | 25.35                  |
| 3      | 1916          | 2112             | 44.41              | 41.24                  |

**Takeaway:** Segmentation helps **expected profit in clusters 0 and 3** (with cluster 0 nearly a tie), while **clusters 1 and 2 favor the global model** under the same cost assumptions. That mixed pattern explains why aggregate profit can still favor the single global policy.

### Figures to consult in the notebooks

- **`03_training.ipynb`**: precision–recall curves **by cluster** (global vs. cluster-specific calibrated scores)—direct visual support for the AUC-PR / precision–recall trade-offs.  
- **`02_preprocessing.ipynb`**: clustering diagnostics (inertia / silhouette / DB / CH plots) and **subscription rate by cluster** (bar / stability views)—justify using k = 4 and show segment heterogeneity.  
- **`04_evaluation.ipynb`**: displays the same tables as above when executed; use it as the live source of truth if you change costs or retrain.

---

## Interpretation

Final model choice should combine **ranking quality (AUC-PR)**, **operational load (% contacted)**, and **expected profit**, not accuracy alone. This repository implements that end-to-end: calibrated probabilities, profit-optimal thresholds on validation, and a clean held-out test comparison.
