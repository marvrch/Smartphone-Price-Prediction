# Smartphone Price Prediction — End-to-End ML Project

Predict the retail price of a smartphone from its specifications.
This project focuses on **clean preprocessing**, **leak-free modeling**, and a **public Streamlit app**.

**Live app:** [https://group-dewaruci-smartphone-price-prediction.streamlit.app/](https://group-dewaruci-smartphone-price-prediction.streamlit.app/)

---

## Background & Goals

Real-world smartphone data is noisy: inconsistent variants (“pro max”, “promax”, …), mixed booleans, high-cardinality brand/model, and prices with a right-skewed distribution.
Our goals:

1. Turn raw specs into trustworthy, model-ready features.
2. Compare several regression algorithms fairly (no leakage, same folds).
3. Deploy the best non-overfitting model as an easy-to-use web app.

---

## Data & Target

* Source: compiled via **direct web extraction** (scraping) + rule-based cleaning.
* Target: `final_price` (continuous).

  * Distribution is **right-skewed** → we train on **`log1p(y)`** and convert predictions back with **`expm1`** at inference.

**Main features**

* Numeric: `variant_rank`, `ram`, `storage`
* Binary: `has_5g`, `has_bundle_accessories`, `free`
* Categorical (low-card): `color`
* Categorical (high-card): `brand`, `model`

---

## Tools

* Python, NumPy, Pandas
* scikit-learn (`Pipeline`, `ColumnTransformer`, `RobustScaler`, linear models, SVR, RF)
* XGBoost, LightGBM (benchmarks)
* (Optional during exploration) CatBoost
* Streamlit (deployment)

---

## Process & Key Learnings

1. **Data acquisition & cleaning**
   Not just median imputation. We **scraped** the source and applied rule-based fixes:

   * Normalize variant strings → **`variant_rank`** (e.g., “pro max” > “pro” > “lite”).
   * Standardize booleans (`has_5g`, `has_bundle_accessories`, `free`), de-duplicate, and fix obvious text inconsistencies.

2. **Leak-free setup**
   We **split train/test first**, then do **all** feature engineering and scaling **inside a `Pipeline` + `ColumnTransformer`**. This guarantees the exact same transforms for training, validation, and deployment.

3. **Target transformation**
   Because `final_price` is skewed, we model **`log1p(price)`**. The Streamlit app converts results back to the original unit with **`expm1`**.

4. **Categorical encoding**

   * **High-cardinality** (`brand`, `model`): **K-Fold Target Encoding** with smoothing and a safe **global-mean fallback** for unseen categories at inference.
   * **Low-cardinality** (`color`): **One-Hot Encoding**.
     This avoids the dimensionality blow-up of OHE on large vocabularies.

5. **Scaling**

   * Numeric features (`variant_rank`, `ram`, `storage`): **`RobustScaler`** (more resilient to outliers).
   * **No scaling** for tree-based models (not needed).

---

## Exploratory Insights

* Correlation consistently shows **`storage`** as the strongest driver of price, followed by **`ram`** and **`variant_rank`** (Pearson & Spearman agree).
* The heatmap confirms the ordering above and supports building simple, well-behaved numeric features.

---

## Modeling & Evaluation

**Models compared**

* Linear: **Linear Regression**, **Ridge**, **Lasso**, **ElasticNet**
* Kernel: **SVR**
* Trees/Boosting: **RandomForest**, **XGBoost**, **LightGBM**
* (Explored) **CatBoost** (similar to other boosting models; training time trade-off)

**Validation**

* `KFold` cross-validation with **RandomizedSearchCV** for hyperparameter tuning (CatBoost tuned manually with early stopping).
* Metrics: **R²** (primary, out-of-sample), **MAE**, **RMSE** (on original price scale).
* **Overfitting policy:** flag a model as overfitting if **`R²_train − R²_test > 0.05`**.

**Key result**

* Boosting / RandomForest achieved **high test R²** but **gap > 0.05** → **overfit** under our policy.
* Among **non-overfitting** models, **SVR (tuned)** delivered the **best test R² (~0.80)** with a gap ≈ **0.04** and competitive MAE/RMSE versus linear-regularized models.
* **Selected model:** **SVR (tuned)** — best balance of accuracy and generalization.

---

## Deployment

* The final **scikit-learn `Pipeline`** is saved as **`artifacts/model.pkl`**, with companion **`meta.json`**:

  * feature order and lists of choices,
  * `numeric_choices` for RAM/Storage dropdowns,
  * the `y_is_log1p` flag for correct inverse transform.
* **Streamlit app**:

  * **Variant** is a dropdown mapped to **`variant_rank`** internally.
  * **Brand/Model** go through the **target encoder** in the pipeline; unseen categories get a **global mean** fallback.
  * Predictions are converted back with **`expm1`** for user-friendly prices.
  * Clean UI: ProperCase labels, Yes/No for booleans.

**Live app:** [https://group-dewaruci-smartphone-price-prediction.streamlit.app/](https://group-dewaruci-smartphone-price-prediction.streamlit.app/)

---



## What’s Next (Suggestions)

**Features**

* Enrich with: release year, chipset, display size, camera/battery specs, ratings, brand families.
* Improve `model` and `variant_rank` extraction quality.
* Try safe interactions (e.g., `storage/ram`), or bucketed ranges. For non-linear models, allow the model to learn interactions.

**Modeling**

* If you want a stronger boosting model **without** overfitting:

  * reduce depth / `num_leaves`,
  * increase regularization (`reg_lambda`, `min_child_weight` / `min_data_in_leaf`),
  * use `subsample`/`colsample` < 1,
  * always early stop.
* Consider simple **stacking** (SVR + linear regularized) with strict CV.

**Monitoring**

* In production, log input distributions & errors; add **drift checks** and schedule **periodic retraining**.

**UX**

* Batch prediction (upload CSV),
* More informative error messages & tooltips in the app.

---



### Acknowledgements

Thanks to the data sources collected via web extraction ([iOS Ref](https://iosref.com/ram-processor), [GSM Arena](https://www.gsmarena.com/)) and the open-source community (scikit-learn, XGBoost, LightGBM, Streamlit).

---

