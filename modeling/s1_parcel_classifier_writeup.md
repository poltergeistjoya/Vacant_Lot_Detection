# Stage 1: Parcel-Level Classifier — Writeup

## Objective

Validate whether NAIP spectral features contain a supervised signal for distinguishing vacant from non-vacant parcels. This is a signal check before investing in pixel-level segmentation — if aggregated spectral statistics can't separate vacancy at the parcel level, raw pixel values are unlikely to work either.

---

## Step 1: Data Loading and Preparation

### What happens
1. Load the full MapPLUTO geodatabase (856,998 NYC tax lots) for land-use labels (`BldgClass` column).
2. Load pre-computed spectral statistics from GCS (`gs://thesis_parcels/eda/new_york_new_york_bldgclss/parcel_spectral_stats.csv`) — these were computed during EDA by reducing NAIP imagery over each parcel in Google Earth Engine.
3. Join the two on `BBL` (Borough-Block-Lot identifier), yielding 25,000 parcels with both spectral features and land-use labels.

### Justification
The 25,000-parcel sample was stratified during EDA to oversample vacant lots (LandUse code "11" / BldgClass codes V*, G7), ensuring the rare class has sufficient representation. Using pre-computed spectral stats avoids re-running the expensive GEE reduction.

### Output
A DataFrame of 25,000 parcels with 35 columns: spectral band statistics (mean, median, stdDev, count for each of R, G, B, NIR, NDVI, SAVI, Brightness, BareSoilProxy) plus BBL and BldgClass.

---

## Step 2: Label Construction

### What happens
`build_labels()` creates a binary target variable from BldgClass using the config's `vacant_codes` list (`["G7", "V0", "V1", ..., "V9"]`):
- 1 = vacant (BldgClass matches a vacant code)
- 0 = not vacant

### Output
- **500 vacant parcels (2.0%)** out of 25,000 total.

### Justification
Labels are derived from the config system (`CityConfig.parcel.vacant_codes`) rather than hardcoded, allowing the same pipeline to work for other cities with different land-use classification schemes. The 2% positive rate reflects the natural rarity of vacant land in NYC after stratified sampling.

---

## Step 3: Feature Selection

### What happens
Select 16 features — the **mean** and **standard deviation** of 8 spectral bands/indices per parcel:
- Raw bands: R, G, B, N (NIR)
- Derived indices: NDVI, SAVI, Brightness, BareSoilProxy

### What was excluded and why
- **Count columns** (pixel count per parcel): excluded because count is a proxy for parcel area, which is geometric information. The model must generalize to cities without parcel data at inference time, so geometric features are not allowed.
- **Median columns**: excluded to keep the feature set compact. Mean and stdDev capture central tendency and heterogeneity; median adds little beyond mean for this purpose.
- **No geometric features** (area, perimeter, shape): same rationale as count — not available at inference without parcel boundaries.

### NaN handling
32 NaN values were filled with column medians. These arise from parcels with very few pixels where stdDev computation can produce null values.

### Justification
- **Mean** captures the average spectral signature of the parcel (e.g., high NDVI mean = vegetation, high BareSoilProxy mean = bare/impervious surface).
- **StdDev** captures within-parcel heterogeneity — a proxy for spatial texture. A vacant lot with mixed bare soil and weeds has high spectral variance; a uniform building roof has low variance. This turned out to be the most important feature category (see Feature Importance below).

---

## Step 4: Train/Test Split

### What happens
80/20 stratified split with `random_state=42`, preserving the 2% vacant fraction in both sets.

### Output
- **Train:** 20,000 parcels (2.0% vacant = 400 vacant)
- **Test:** 5,000 parcels (2.0% vacant = 100 vacant)

### Justification
Stratified splitting ensures both sets have the same class distribution. The fixed random seed matches the visualization notebook (`01_visualize_sample.ipynb`) for consistency. 80/20 is standard and provides a test set of 5,000 parcels — enough for stable metric estimation even with only 100 positive examples.

---

## Step 5: Feature Scaling

### What happens
`StandardScaler` is fit on the training set (zero mean, unit variance) and applied to both train and test sets.

### Justification
Logistic Regression and SVM (RBF kernel) are sensitive to feature scale — features with larger absolute values would dominate the model. StandardScaler ensures all features contribute proportionally. The scaler is fit only on training data to prevent data leakage from the test set. Random Forest is scale-invariant but receives scaled features for consistency.

### Output
The fitted scaler is saved as `s1_scaler_001.joblib` for use in future inference.

---

## Step 6: Cross-Validation

### What happens
Stratified 5-fold cross-validation on the training set for each of three models:
- **Logistic Regression** (`class_weight="balanced"`)
- **SVM with RBF kernel** (`class_weight="balanced"`)
- **Random Forest** (200 trees, `class_weight="balanced"`)

Two metrics are computed per fold: **F1 score** and **Average Precision** (area under the precision-recall curve).

### Why these metrics
- **F1**: harmonic mean of precision and recall. With 2% positive rate, accuracy would be 98% by always predicting "not vacant" — F1 penalizes this degenerate behavior.
- **Average Precision (AP)**: summarizes the precision-recall curve across all classification thresholds. More informative than F1 at a single threshold because it shows how well the model *ranks* vacant parcels higher than non-vacant ones overall.
- **Accuracy is NOT reported** as a primary metric because it is misleading with severe class imbalance.

### Why `class_weight="balanced"`
With 2% positive rate, an unweighted model would learn to predict "not vacant" for everything (98% accuracy, 0% recall). `class_weight="balanced"` tells sklearn to inversely weight classes by frequency — vacant parcels get ~50× the loss weight of non-vacant parcels, forcing the model to pay attention to the rare class.

### Results

| Model | F1 (mean ± std) | AP (mean ± std) |
|-------|-----------------|-----------------|
| Logistic Regression | 0.149 ± 0.005 | 0.311 ± 0.056 |
| SVM (RBF) | 0.212 ± 0.013 | 0.254 ± 0.050 |
| Random Forest | 0.250 ± 0.051 | 0.306 ± 0.028 |

### Interpretation
- **Random baseline AP ≈ 0.02** (equal to the positive class fraction). All models achieve AP 10-15× above random, confirming the spectral signal exists.
- RF has the highest F1 (0.25) and comparable AP to LogReg (0.31 vs 0.31). SVM has lower AP (0.25), suggesting the RBF kernel may be overfitting or the data isn't well-suited to radial decision boundaries.
- The relatively high variance in RF F1 (±0.051) indicates sensitivity to which folds contain the harder-to-classify vacant parcels.

### Output
`modeling/outputs/data/nyc_2022/s1_cv_results.csv`

---

## Step 7: Final Model Training and Test Evaluation

### What happens
Each model is retrained on the full training set (20,000 parcels) and evaluated on the held-out test set (5,000 parcels).

### Results

| Model | F1 | AP | Precision | Recall |
|-------|----|----|-----------|--------|
| Logistic Regression | 0.159 | 0.320 | 0.09 | 0.74 |
| SVM (RBF) | 0.215 | 0.212 | 0.13 | 0.69 |
| Random Forest | 0.290 | 0.346 | 0.61 | 0.19 |

### Interpretation

The three models exhibit fundamentally different precision-recall tradeoffs:

**Logistic Regression** (precision=0.09, recall=0.74): Casts a wide net — catches 74% of vacant parcels but 91% of its "vacant" predictions are false positives. This is a linear model that draws a single hyperplane in 16-dimensional feature space. The high recall / low precision suggests the linear boundary captures the general vacant region but can't cleanly separate it from spectrally similar non-vacant parcels.

**SVM** (precision=0.13, recall=0.69): Similar behavior to LogReg but with a non-linear (RBF) kernel. Marginally better precision (0.13 vs 0.09) at slightly lower recall (0.69 vs 0.74). The non-linear boundary provides modest improvement but the fundamental challenge — spectral overlap between some vacant and non-vacant parcels — remains.

**Random Forest** (precision=0.61, recall=0.19): The opposite strategy — very conservative. When it predicts "vacant," it's right 61% of the time, but it only identifies 19% of actual vacant parcels. The RF learns strong decision rules for the most spectrally distinctive vacant parcels (high BareSoilProxy, high spectral variance) and stays silent on ambiguous cases. This precision-focused behavior is characteristic of tree ensembles on imbalanced data.

**Key takeaway:** The spectral signal exists but is partial. Some vacant parcels are spectrally distinctive (RF finds them with 61% precision), while many are spectrally ambiguous at the parcel-aggregate level (only 19% recall). This is expected — averaging spectral values over an entire parcel discards the within-parcel spatial patterns that distinguish vacancy. This motivates pixel-level segmentation, which preserves spatial information.

---

## Step 8: Precision-Recall Curves

### What happens
Plot precision vs. recall across all classification thresholds for each model.

### Justification
PR curves show the full tradeoff between precision and recall, not just performance at the default 0.5 threshold. This is important because the optimal threshold depends on the application — a city planner who wants to survey ALL potentially vacant lots would choose a low threshold (high recall), while one who wants to minimize wasted site visits would choose a high threshold (high precision).

### Output
`modeling/outputs/figures/nyc_2022/s1_pr_curves.png`

---

## Step 9: Feature Importance (Random Forest)

### What happens
Extract and plot the Gini importance of each feature from the trained Random Forest.

### Results
The **standard deviation features dominate** the importance rankings, with `Brightness_stdDev` being the most important feature. The stdDev features collectively outweigh the mean features.

### Interpretation
StdDev captures **within-parcel spectral heterogeneity** — a proxy for spatial texture:
- **Vacant lots** tend to have high spectral variance (mix of bare soil patches, weeds, debris, scattered materials).
- **Non-vacant parcels** with buildings tend to have more uniform spectral signatures (uniform rooftop, maintained lawn).

This finding strongly supports moving to pixel-level segmentation: the most important parcel-level features are *proxies* for spatial texture. A segmentation model (U-Net/DeepLabV3+) that operates on raw pixel patches gets the actual spatial texture directly — it doesn't need the proxy. The information that stdDev summarizes into a single number is available to the segmentation model as a full 256×256 spatial pattern.

### Output
`modeling/outputs/figures/nyc_2022/s1_feature_importance.png`

---

## Step 10: Model Serialization

### What happens
Save each trained model as a `.joblib` file with a JSON metadata sidecar containing: feature list, train/test sizes, vacant fractions, config name, model type, and test metrics.

### Naming convention
`s{stage}_{model_type}_{run_id}.joblib` — e.g., `s1_rf_001.joblib`

### Output artifacts

```
modeling/outputs/models/nyc_2022/
  s1_logreg_001.joblib    + s1_logreg_001.json
  s1_svm_001.joblib       + s1_svm_001.json
  s1_rf_001.joblib        + s1_rf_001.json
  s1_scaler_001.joblib    (fitted StandardScaler for inference)
```

### Justification
JSON sidecars ensure reproducibility — anyone loading a model knows exactly which features it expects, what data it was trained on, and how it performed. The scaler is saved separately because it must be applied to new data before prediction.

---

## Summary

| What was validated | Finding |
|---|---|
| Spectral signal exists | Yes — AP 10-15× above random baseline |
| Best model | Random Forest (F1=0.29, AP=0.35, precision=0.61) |
| Most important features | StdDev features (spectral heterogeneity), especially Brightness_stdDev |
| Limitation | Low recall (19%) — many vacant parcels are spectrally ambiguous when averaged |
| Implication for segmentation | StdDev dominance = spatial texture matters most → pixel-level models that see raw spatial patterns should improve on this |
