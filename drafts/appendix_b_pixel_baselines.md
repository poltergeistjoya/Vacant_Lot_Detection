# Appendix B: Pixel-Level Baseline Models (Random Forest and LightGBM)

## B.1 Motivation

Before training convolutional neural networks, two tree-based ensemble models were evaluated as pixel-level baselines. These models classify each pixel independently based on its spectral feature vector, without any spatial context from neighboring pixels. The purpose of this baseline is twofold: (1) to quantify how much of the vacancy signal is accessible from per-pixel spectral values alone, and (2) to establish a lower bound against which the CNN architectures (U-Net, DeepLabV3+) are compared.

## B.2 Feature Set

Each pixel was represented by a 10-dimensional feature vector:

- **4 NAIP bands:** R, G, B, NIR (normalized to [0, 1])
- **6 spectral indices:** NDVI, SAVI, Brightness, BareSoilProxy, EVI, GNDVI

The spectral indices were computed per-pixel from the NAIP bands. These indices capture vegetation vigor (NDVI, SAVI, EVI, GNDVI), overall reflectance (Brightness), and exposed soil presence (BareSoilProxy). Notably, these models receive no spatial information -- each pixel is classified in isolation, so spatial patterns such as the texture of a grassy lot versus a uniform rooftop are invisible.

## B.3 Sampling Strategy

The training split (Queens, 12,576 patches) contains approximately 586 million labeled pixels (19.7M vacant, 566.4M non-vacant). Loading all pixels into memory is infeasible. Reservoir sampling was used to draw a representative subset:

**LightGBM:** 79.7 million pixels sampled (19.7M vacant, 60M non-vacant). Since nearly all vacant pixels were sampled, the class ratio in the sample (~1:3) differs substantially from the true distribution (~1:29). To correct for this, `scale_pos_weight` was set to 9.43, computed from the true population distribution (3.4% vacant).

**Random Forest:** 40 million pixels sampled (10M vacant, 30M non-vacant). Class weights were computed to reflect the true population: `vacant_weight = 1.97`, `non-vacant_weight = 18.87`.

## B.4 Model Configuration

**LightGBM (run gbm/001):**
- `n_estimators`: 500
- `max_depth`: 12
- `learning_rate`: 0.05
- `num_leaves`: 63
- `min_child_samples`: 1,000
- `scale_pos_weight`: 9.43

**Random Forest (run rf/002):**
- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_leaf`: 50
- `n_jobs`: 4
- `class_weight`: computed from true population distribution

## B.5 Results

**Table B.1: Pixel-level baseline results**

| Model | Split | IoU | F1 | Precision | Recall | AP | Kappa | Pixels evaluated |
|-------|-------|----:|---:|----------:|-------:|---:|------:|-----------------:|
| LightGBM | Val (Bronx) | 0.0397 | 0.0763 | 0.0397 | 0.9659 | 0.0544 | 0.0049 | 224,451,453 |
| LightGBM | Test (Brooklyn) | 0.0347 | 0.0670 | 0.0347 | 0.9369 | 0.0452 | 0.0098 | 363,008,760 |
| Random Forest | Val (Bronx) | 0.0012 | 0.0024 | 0.1240 | 0.0012 | 0.0535 | 0.0017 | 224,451,453 |
| Random Forest | Test (Brooklyn) | 0.0005 | 0.0010 | 0.0431 | 0.0005 | 0.0445 | 0.0003 | 363,008,760 |

The two models exhibit diametrically opposed failure modes:

**LightGBM** achieves near-total recall (93--97%) at the cost of abysmal precision (3.5--4.0%). In effect, the model labels most pixels as vacant. With 3.4% true vacancy, this "predict everything as vacant" strategy yields an IoU of only 0.035. The high `scale_pos_weight` drives the model to avoid missing any vacant pixel, but without spatial context to discriminate spectrally similar surfaces, the model cannot distinguish a bare soil vacant lot from a bare soil road shoulder.

**Random Forest** adopts the opposite extreme: precision reaches 12.4% on validation but recall is essentially zero (0.12%). The model predicts almost nothing as vacant. Despite moderate precision when it does predict, the near-zero recall makes it useless in practice. This mirrors the behavior observed in the parcel-level Random Forest (Appendix A), where the model learned strong rules for the most distinctive vacant parcels but abstained on ambiguous cases.

Both models achieve Average Precision in the range of 0.04--0.05, only marginally above the random baseline of 0.034 (the true vacancy fraction). Cohen's kappa values near zero confirm that agreement with ground truth is barely above chance.

## B.6 Interpretation

The failure of both baselines confirms that **per-pixel spectral features alone are insufficient for vacancy detection**. This is consistent with the parcel-level analysis (Appendix A), where standard deviation features -- proxies for within-parcel spatial texture -- dominated the feature importance rankings. Without access to the spatial arrangement of pixels, the models cannot leverage texture, shape, or contextual cues that distinguish vacancy from spectrally similar land covers.

These results establish a clear lower bound and motivate the use of convolutional architectures (U-Net, DeepLabV3+) that operate on spatial patches and can learn texture, edge, and contextual features. The improvement from pixel-level baselines (IoU ~0.001--0.04) to the best CNN models (IoU ~0.14--0.19; see Section 5) quantifies the contribution of spatial context to the vacancy detection task.

## B.7 Training Details

Both models were trained on the school compute server (no GPU required for tree-based models). LightGBM training completed in approximately 6 minutes; Random Forest training required approximately 77 minutes. Evaluation on the validation and test sets (streaming over all patches) required 35 and 56 minutes respectively for LightGBM, and 42 and 70 minutes for Random Forest. Total wall-clock time for both runs was approximately 5 hours.
