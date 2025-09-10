
# Rainfall Prediction — Model Comparison (Logistic Regression vs. Random Forest)

This repository contains a **clean, end‑to‑end comparison** of two supervised learning algorithms for **rainfall prediction** using the Australian weather dataset. The work is implemented in a single Jupyter notebook:

- `Building a Rainfall Prediction Classifier.ipynb`

The notebook trains and evaluates **Logistic Regression** and **Random Forest** using the same preprocessing pipeline and cross‑validation strategy, then compares their performance with standard classification metrics and visualizations.

---

## Project Overview

- **Goal**: Predict whether it will rain **tomorrow** (`RainTomorrow`) based on today’s weather measurements.
- **Models compared**: `LogisticRegression` and `RandomForestClassifier` (from scikit‑learn).
- **Primary optimization metric**: ROC‑AUC (via cross‑validated GridSearchCV).
- **Region subset**: The dataset is filtered to **Melbourne** stations (`Melbourne`, `MelbourneAirport`, `Watsonia`) to focus on a coherent climatic region.
- **Reproducibility**: Fixed `random_state=42` where applicable.

> 📓 All code, charts, and final metrics are in the notebook. Run it to reproduce the results.

---

## Data

- **Source**: IBM COS public copy of the **Weather in Australia** dataset  
  `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv`
- **Target**: `RainTomorrow` (Yes/No → 1/0)

### Feature Engineering & Cleaning (in the notebook)
- Convert **Yes/No** flags to binary: `RainToday`, `RainTomorrow` → {0,1}
- Extract **season** from `Date` (`Summer`, `Autumn`, `Winter`, `Spring`), then drop raw `Date`
- Keep only Melbourne‑area **Location** values: `Melbourne`, `MelbourneAirport`, `Watsonia`
- Drop rows with missing target; handle remaining missing values in the pipeline (see below)

---

## Modeling Pipeline

A consistent preprocessing pipeline is applied to both models:

- **Numeric features**: `SimpleImputer(strategy="median")` → `StandardScaler()`
- **Categorical features**: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`
- Combined with `ColumnTransformer` and wrapped in a `Pipeline` with each classifier.

### Hyperparameter Search

Cross‑validated grid search (`StratifiedKFold(n_splits=5, shuffle=True)`) optimized for **ROC‑AUC**.

- **Logistic Regression**
  - `solver`: `liblinear`
  - `penalty`: `["l1", "l2"]`
  - `class_weight`: `[None, "balanced"]`

- **Random Forest**
  - `n_estimators`: `[50, 100]`
  - `max_depth`: `[None, 10, 20]`
  - `min_samples_split`: `[2, 5]`
  - `class_weight`: `[None, "balanced"]`

---

## Evaluation

A stratified train/test split is used (`test_size=0.2`). For each best model found by GridSearchCV, the notebook reports:

- **Accuracy**
- **Precision**
- **Recall**
- **F1‑score**
- **ROC‑AUC** (on the test set)

### Visualizations
- Target distribution and **correlation heatmap**
- **Confusion matrix** for each model
- **Feature importance**:
  - Random Forest: `feature_importances_` bar chart (top features)
  - Logistic Regression: absolute coefficient magnitudes (top features)

> The notebook prints a consolidated results table at the end (one row per model).

---

## Results — Final Comparison (Test Set)

| Model               | Accuracy | Precision | Recall | F1   | ROC‑AUC |
|---------------------|----------|-----------|--------|------|---------|
| **Random Forest**   | **0.86** | **0.81**  | 0.46   | 0.58 | **0.90** |
| **Logistic Regression** | 0.80  | 0.53      | **0.75** | **0.62** | 0.86 |

**Takeaways**  
- **Primary metric (ROC‑AUC):** Random Forest wins (**0.90** vs **0.86**).  
- **Accuracy & Precision:** Random Forest is higher (0.86 acc, 0.81 precision).  
- **Recall & F1:** Logistic Regression recovers more positives (**0.75 recall**) and slightly higher **F1** (**0.62**).  
- **Interpretability:** Logistic Regression offers transparent coefficients; Random Forest provides feature importances but is less interpretable.

**When to choose which?**  
- Pick **Random Forest** if you care most about **ranking quality / ROC‑AUC** or overall accuracy.  
- Pick **Logistic Regression** if **recall** matters (catching more rainy days) or you need **interpretable** linear effects.

> Note: Best Logistic Regression params (via CV): `solver=liblinear, penalty=l1, class_weight=balanced`. Best CV score reported in the notebook: **0.85**.
