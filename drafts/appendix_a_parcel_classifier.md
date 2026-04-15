# Appendix A: Parcel-Level Spectral Signal Validation

## A.1 Motivation

Before investing in pixel-level segmentation, a preliminary analysis was conducted to determine whether NAIP spectral features contain a supervised signal for distinguishing vacant from non-vacant parcels. The reasoning is conservative: if aggregated spectral statistics cannot separate vacancy at the parcel level, raw pixel values are unlikely to succeed either. This appendix documents that signal check.

## A.2 Data Preparation

The full MapPLUTO geodatabase (856,998 NYC tax lots) was joined with pre-computed spectral statistics from Google Earth Engine. The spectral statistics were computed by reducing NAIP imagery over each parcel polygon, yielding per-parcel mean, median, and standard deviation for eight spectral bands and indices: R, G, B, NIR, NDVI, SAVI, Brightness, and BareSoilProxy.

The join on BBL (Borough-Block-Lot identifier) yielded 25,000 parcels with both spectral features and land-use labels. This sample was stratified during EDA to oversample vacant lots, ensuring the rare class had sufficient representation.

## A.3 Label Construction

Binary labels were derived from the BldgClass column using the same vacant codes as the segmentation task (V0--V9, G7):

- 1 = vacant (BldgClass matches a vacant code)
- 0 = not vacant

Of the 25,000 parcels, **500 (2.0%)** were labeled vacant. The 2% positive rate reflects the natural rarity of vacant land in NYC even after stratified sampling.

## A.4 Feature Selection

Sixteen features were selected -- the mean and standard deviation of eight spectral bands/indices per parcel:

- **Raw bands:** R, G, B, NIR
- **Derived indices:** NDVI, SAVI, Brightness, BareSoilProxy

The following were excluded:

- **Pixel count columns:** Count is a proxy for parcel area (geometric information). Since the segmentation model must generalize to cities without parcel data at inference time, geometric features were not permitted.
- **Median columns:** Mean and standard deviation capture central tendency and heterogeneity; median adds little beyond mean for this application.

Thirty-two NaN values (arising from parcels with very few pixels where standard deviation computation produces null values) were filled with column medians.

## A.5 Experimental Setup

The dataset was split 80/20 with stratified sampling (random_state=42), preserving the 2% vacant fraction in both sets:

- **Train:** 20,000 parcels (400 vacant, 2.0%)
- **Test:** 5,000 parcels (100 vacant, 2.0%)

Features were standardized using a StandardScaler fitted on the training set only (zero mean, unit variance) to prevent data leakage. While Random Forest is scale-invariant, Logistic Regression and SVM (RBF kernel) are sensitive to feature magnitude.

Three models were evaluated with `class_weight="balanced"` (inverse class frequency weighting, giving vacant parcels approximately 50x the loss weight of non-vacant parcels):

- Logistic Regression
- SVM with RBF kernel
- Random Forest (200 trees)

## A.6 Cross-Validation Results

Stratified 5-fold cross-validation on the training set:

**Table A.1: Cross-validation results (mean +/- std across 5 folds)**

| Model | F1 | Average Precision |
|-------|---:|------------------:|
| Logistic Regression | 0.149 +/- 0.005 | 0.311 +/- 0.056 |
| SVM (RBF) | 0.212 +/- 0.013 | 0.254 +/- 0.050 |
| Random Forest | 0.250 +/- 0.051 | 0.306 +/- 0.028 |

With a random baseline AP of approximately 0.02 (equal to the positive class fraction), all models achieve AP 10--15x above random, confirming that the spectral signal exists. The relatively high variance in RF F1 (+/-0.051) indicates sensitivity to which folds contain the harder-to-classify vacant parcels.

## A.7 Test Set Results

Each model was retrained on the full training set and evaluated on the held-out test set:

**Table A.2: Test set performance**

| Model | F1 | AP | Precision | Recall |
|-------|---:|---:|----------:|-------:|
| Logistic Regression | 0.159 | 0.320 | 0.09 | 0.74 |
| SVM (RBF) | 0.215 | 0.212 | 0.13 | 0.69 |
| Random Forest | 0.290 | 0.346 | 0.61 | 0.19 |

The three models exhibit fundamentally different precision-recall tradeoffs:

**Logistic Regression** (precision=0.09, recall=0.74) casts a wide net -- it catches 74% of vacant parcels but 91% of its "vacant" predictions are false positives. The linear decision boundary captures the general vacant region in feature space but cannot cleanly separate it from spectrally similar non-vacant parcels.

**SVM** (precision=0.13, recall=0.69) shows marginally better precision at slightly lower recall. The non-linear RBF kernel provides modest improvement but the fundamental spectral overlap between vacancy types remains.

**Random Forest** (precision=0.61, recall=0.19) adopts the opposite strategy -- very conservative predictions. When it predicts "vacant," it is correct 61% of the time, but it only identifies 19% of actual vacant parcels. The RF learns strong decision rules for the most spectrally distinctive vacant parcels (high BareSoilProxy, high spectral variance) and abstains on ambiguous cases. This precision-focused behavior is characteristic of tree ensembles on imbalanced data.

[INSERT FIGURE A.1: Precision-recall curves for all three models (s1_pr_curves.png)]

## A.8 Feature Importance

The Gini importance scores from the trained Random Forest reveal that **standard deviation features dominate** the importance rankings. `Brightness_stdDev` is the single most important feature, and the stdDev features collectively outweigh the mean features.

[INSERT FIGURE A.2: Feature importance bar chart (s1_feature_importance.png)]

Standard deviation captures within-parcel spectral heterogeneity -- a proxy for spatial texture:

- **Vacant lots** tend to have high spectral variance (mix of bare soil patches, weeds, debris, scattered materials).
- **Non-vacant parcels** with buildings tend to have more uniform spectral signatures (uniform rooftop, maintained lawn).

This finding directly motivates the pixel-level segmentation approach adopted in this thesis. The most important parcel-level features are *proxies* for spatial texture. A convolutional segmentation model operating on raw pixel patches accesses the actual spatial texture directly -- it does not need the proxy. The information that standard deviation summarizes into a single number is available to the segmentation model as a full spatial pattern across the input patch.

## A.9 Summary

| Finding | Detail |
|---------|--------|
| Spectral signal exists | AP 10--15x above random baseline |
| Best parcel-level model | Random Forest (F1=0.29, AP=0.35, precision=0.61) |
| Most important features | StdDev features (spectral heterogeneity), especially Brightness_stdDev |
| Key limitation | Low recall (19%) -- many vacant parcels are spectrally ambiguous when averaged |
| Implication for segmentation | StdDev dominance indicates spatial texture is the critical discriminator; pixel-level models that operate on raw spatial patterns should improve on aggregated parcel-level features |

## A.10 Artifacts

All parcel-level model artifacts are stored in `outputs/eda/parcel_classifier/`:

- `s1_logreg_001.joblib` + `.json` -- Logistic Regression model and metadata
- `s1_svm_001.joblib` + `.json` -- SVM model and metadata
- `s1_rf_001.joblib` + `.json` -- Random Forest model and metadata
- `s1_scaler_001.joblib` -- Fitted StandardScaler
- `s1_cv_results.csv` -- Cross-validation results
- `s1_pr_curves.png` -- Precision-recall curves
- `s1_feature_importance.png` -- Feature importance plot
