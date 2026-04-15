# 5. Results

## 5.1 Baseline Results

The pixel-level tree-based baselines establish a lower bound for the task, confirming that per-pixel spectral features alone are insufficient for vacancy detection.

**Table 5.1: Pixel-level baseline results (default 0.5 threshold)**

| Model | Val IoU | Val F1 | Val Prec | Val Rec | Test IoU | Test F1 | Test Prec | Test Rec |
|-------|--------:|-------:|---------:|--------:|---------:|--------:|----------:|--------:|
| Random Forest | 0.0012 | 0.0024 | 0.1240 | 0.0012 | 0.0005 | 0.0010 | 0.0431 | 0.0005 |
| LightGBM | 0.0397 | 0.0763 | 0.0397 | 0.9659 | 0.0347 | 0.0670 | 0.0347 | 0.9369 |

The Random Forest achieves moderate precision (12.4% on validation) but near-zero recall, effectively refusing to classify any pixel as vacant. LightGBM takes the opposite approach: near-total recall (97%) at precision matching the base rate (~4%), essentially predicting everything as vacant. Both models achieve IoU below 0.04 and Cohen's kappa near zero. The failure of both baselines demonstrates that spatial context is essential for this task -- per-pixel spectral values cannot disambiguate vacant land from spectrally similar non-vacant surfaces. Full details are in Appendix B.

## 5.2 Experimental Progression

Over the course of 30 deep learning training runs (Table 5.2), the experimental program progressed through three phases of systematic improvement. Two additional runs with modified gradient norms for F2-priority optimization are currently in progress and will be reported when complete.

### Phase 1: Hyperparameter Exploration (Runs 001--013, v1 Mask, 256px Patches)

The initial runs explored backbone depth (ResNet-18, -34, -50), input channel count (4-band RGBN vs. 10-band with spectral indices), loss function composition (BCE + Dice, BCE + Lovasz), positive weight (10, 30, 50), and oversampling factor (4x, 8x).

Key findings from Phase 1:

- **4-band input outperformed 10-band.** Run 007 (DeepLabV3+, ResNet-34, 4 bands, val IoU 0.120) outperformed comparably configured 10-band runs (001--005, val IoU 0.089--0.105). The models can learn the derived spectral indices (NDVI, SAVI, etc.) directly from the raw bands, and the additional channels added noise without providing new information. 

- **Lovasz loss outperformed Dice loss.** Comparing UNet runs 006 (pure Dice, val IoU 0.089) and 007 (BCE + Dice, val IoU 0.103) against later Lovasz-based runs, the BCE + Lovasz combination showed more consistent convergence and better validation performance.

- **Oversampling matters.** Runs with 4x or 8x oversampling of vacant-containing patches (e.g., runs 003, 010) consistently outperformed non-oversampled runs (e.g., runs 001, 002).

- **Best Phase 1 result:** UNet run 013 (ResNet-34, 4ch, 8x oversample) achieved val IoU 0.129.

### Phase 2: Scaling Up (Runs 015--017, v1 Mask, 512/1024px Patches)

Increasing patch size from 256 to 512 and 1024 pixels provided the model with substantially more spatial context per prediction, closer to the ~410m extent used by Mao et al. (2022).

- DeepLabV3+ run 015 (ResNet-50, 1024px, 4ch): val IoU 0.125, test IoU 0.136
- UNet run 015 (ResNet-50, 512px, 4ch): val IoU 0.144, test IoU 0.128

Larger patches improved performance, particularly for DeepLabV3+ which benefits from its multi-scale ASPP module having more contextual information to aggregate. However, the v1 mask still contained systematic label errors that limited further gains.

### Phase 3: Mask Refinement and Training Stabilization (Runs 018--031, v2 Mask)

The v2 vacancy mask (Section 3.3) corrected 67 parcels and introduced planimetric roadbed masking. Concurrently, training was stabilized with linear LR warmup and gradient clipping after runs 019 and 020 collapsed at epoch 2.

- Runs 019 and 020 both collapsed immediately (best epoch = 2, early-stopped at epoch 42) despite using identical configuration to run 018, which converged normally (best epoch = 76). This demonstrated that the combination of pos_weight=30 and Lovasz loss creates a narrow convergence basin sensitive to initialization and batch ordering. The warmup + gradient clipping measures (Section 4.7) resolved this instability -- no subsequent run collapsed.

- The building probability channel was introduced as a 5th input in runs 018, 024, 029, 030, enabling comparison against 4-channel runs.

- **Best DeepLabV3+ by val F2:** Run 027 (ResNet-50, 5ch, 1024px, v2 mask, warmup): val F2 0.423, val F1 0.325 at threshold 0.298. Run 018 (identical config, no warmup) achieves the highest test F2 (0.385) despite a lower val F2 (0.405) -- discussed in Section 5.4. Run 024 (with warmup) achieves the highest validation IoU (0.193) but substantially lower F2 (val 0.295) because warmup shifts the model toward a precision-oriented decision boundary.
- **Best UNet by F2:** Runs 029 and 030 (ResNet-34, 5ch, 512px, v2 mask): val F2 0.385 and test F2 0.341 respectively. Run 031 (4ch) achieves the highest UNet IoU (test 0.141) but lower F2 (test 0.277) -- see Section 5.5.

**Table 5.2: Selected runs across all three phases** (metrics at F2-optimal val threshold)

| Run | Arch | Encoder | Ch | Patch | Mask | Val F2 | Val F1 | Test F2 | Test F1 | Note |
|-----|------|---------|---:|------:|------|--------:|-------:|---------:|--------:|------|
| 001 | DLV3+ | RN18 | 10 | 256 | v1 | 0.218 | 0.180 | 0.183 | 0.156 | Initial baseline |
| 003 | DLV3+ | RN18 | 10 | 256 | v1 | 0.238 | 0.187 | 0.258 | 0.209 | +4x oversample |
| 007 | DLV3+ | RN34 | 4 | 256 | v1 | 0.294 | 0.214 | 0.290 | 0.188 | 4ch > 10ch |
| 013 | UNet | RN34 | 4 | 256 | v1 | 0.293 | 0.229 | 0.330 | 0.231 | Best Phase 1 |
| 015 | DLV3+ | RN50 | 4 | 1024 | v1 | 0.325 | 0.223 | 0.356 | 0.239 | 1024px scale-up |
| 015 | UNet | RN50 | 4 | 512 | v1 | 0.331 | 0.252 | 0.313 | 0.227 | 512px scale-up |
| 018 | DLV3+ | RN50 | 5 | 1024 | v2 | 0.405 | 0.267 | **0.385** | **0.259** | Best test F2 |
| 019 | DLV3+ | RN50 | 5 | 1024 | v2 | 0.194 | 0.161 | 0.151 | 0.129 | Collapsed ep 2 |
| 024 | DLV3+ | RN50 | 5 | 1024 | v2 | 0.295 | **0.324** | 0.197 | 0.236 | Best val IoU; precision-shifted |
| 027 | DLV3+ | RN50 | 5 | 1024 | v2 | **0.423** | 0.325 | 0.278 | 0.278 | Best val F2; warmup |
| 029 | UNet | RN34 | 5 | 512 | v2 | 0.385 | 0.250 | 0.328 | 0.218 | Best UNet val F2 |
| 030 | UNet | RN34 | 5 | 512 | v2 | 0.367 | 0.237 | 0.341 | 0.222 | Best UNet test F2 |
| 031 | UNet | RN34 | 4 | 512 | v2 | 0.347 | 0.289 | 0.277 | 0.248 | Best UNet IoU (4ch) |

## 5.3 Ablation: Mask Quality (v1 vs. v2)

The v2 mask corrections produced consistent improvements across architectures. Comparing matched configurations:

- **DeepLabV3+** (ResNet-50, 5ch, 1024px): Run 015 (v1, 4ch) val F2 0.325 vs. Run 018 (v2, 5ch) val F2 0.405 (+25% relative). These runs also differ in channel count, but the v2 mask alone accounts for a substantial portion of the improvement by removing systematic label errors -- particularly highway parcels in Queens (training set) that were incorrectly labeled as vacant and were directly teaching the model to associate road surfaces with vacancy.

- **UNet** (ResNet-34, 5ch, 512px): No exactly matched v1/v2 pair exists for UNet at 512px. Run 028 (v1 mask equivalent, 4ch, 512px) achieves test F2 0.313; the v2-trained runs 029/030 (5ch) reach test F2 0.328--0.341, a consistent if modest improvement.

The primary mechanism of improvement is noise reduction in the training set. The planimetric roadbed mask alone removed 1.39M vacant pixels from Queens (Table 3.1) that were previously teaching the model to associate road surfaces with vacancy -- a systematic error that propagated to false positives across all boroughs at evaluation time.

## 5.4 Ablation: Training Stabilization

The necessity of warmup and gradient clipping is demonstrated by the collapse of runs 019 and 020:

| Run | Config | Best Epoch | Val F2 | Val IoU | Val Recall | Val Prec | Warmup | Grad Clip |
|-----|--------|:----------:|-------:|--------:|-----------:|---------:|:------:|:---------:|
| 018 | DLV3+ RN50 5ch 1024 v2 | 76 | 0.405 | 0.154 | 0.617 | 0.170 | No | No |
| 019 | identical to 018 | 2 | 0.194 | 0.088 | 0.226 | 0.125 | No | No |
| 020 | identical to 018 | 2 | 0.177 | 0.081 | 0.202 | 0.119 | No | No |
| 024 | DLV3+ RN50 5ch 1024 v2 | 58 | 0.295 | 0.193 | 0.278 | 0.388 | Yes (5 ep) | Yes (1.0) |

Run 018 converged normally while 019 and 020, with identical hyperparameters, collapsed at epoch 2. Introducing warmup and gradient clipping produced run 024, which converged reliably and achieved the highest validation IoU of any run (0.193). The stabilization measures allow the model to establish a feature representation during the warmup period before encountering full-magnitude gradients.

However, warmup also changed the precision-recall operating point. Run 018 (no warmup) reaches validation recall 0.617 at precision 0.170 (val F2 0.405). Run 024 (warmup) converges to recall 0.278 at precision 0.388 (val F2 0.295). The warmup appears to initialize the model more conservatively, resulting in a more precision-oriented decision boundary. By validation IoU, run 024 wins. By F2 -- the primary metric for a screening application -- run 018 wins. The tradeoff is architectural: warmup enables stable convergence to a higher-IoU local minimum, but that minimum is recall-penalized relative to the noisier but recall-favoring basin that run 018 found without warmup.

## 5.5 Ablation: Building Probability Channel (4ch vs. 5ch)

The building probability channel has a differential effect across architectures:

**DeepLabV3+:** Comparing the closest matched pair -- run 015 (4ch, v1, val F2 0.325) vs. run 018 (5ch, v2, val F2 0.405) -- suggests a benefit from the building channel, though the concurrent mask improvement (v1→v2) confounds the comparison.

**UNet:** The result depends on the metric. By IoU, run 031 (4ch, test IoU 0.141) outperforms runs 029 and 030 (5ch, test IoU 0.122 and 0.125). By F2, the result reverses: runs 029 and 030 (5ch) achieve test F2 0.328 and 0.341, well above run 031 (test F2 0.277). The 5ch UNet runs have substantially higher recall (test 0.497 and 0.530) than the 4ch run (test 0.301), at the cost of lower precision (test 0.139 and 0.140 vs. 0.210). The building probability channel provides the model with an explicit cue that a pixel is not a building, reducing false negatives on bare soil adjacent to structures. This helps recall and F2 while generating more false positives and thus lower IoU.

The building channel therefore helps by F2 for both architectures. The architectural claim that UNet "doesn't benefit" only holds if IoU is the evaluation criterion.

## 5.6 Ablation: Patch Size

Larger patches provide more spatial context per prediction:

**DeepLabV3+ (ResNet-50, 4ch, v1 mask baseline):**
- 256px: Runs 001-011 range, best val IoU ~0.120
- 1024px: Run 015, val IoU 0.125, test IoU 0.136

**UNet (ResNet-34, 4ch):**
- 256px: Run 013, val IoU 0.129
- 512px: Run 031, val IoU 0.169

The 512px and 1024px patch sizes provide spatial extents of ~307m and ~614m respectively, bracketing the ~410m extent used by Mao et al. (2022). Both architectures benefit from increased context, but the optimal size differs: DeepLabV3+ performs best at 1024px (where ASPP can leverage broad context), while UNet peaks at 512px. This is consistent with the architectural designs -- UNet's skip connections are most effective at moderate scales where encoder and decoder feature maps have comparable spatial detail, while ASPP explicitly pools information across larger receptive fields.

## 5.7 Best Model Comparison

Three models represent the Pareto frontier of this experiment program. Run 018 (DLV3+, no warmup) achieves the highest F2 across all runs. Runs 029 and 030 (UNet, 5ch) achieve the highest UNet F2, with run 030 having the better test result. Run 031 (UNet, 4ch) achieves the highest UNet IoU. For reference, run 024 (DLV3+, warmup) is included as the val-IoU-optimal model.

**Table 5.3: Head-to-head comparison of top models** (metrics at F2-optimal val threshold)

| Metric | DLV3+ 027 | DLV3+ 018 | UNet 030 | UNet 031 | DLV3+ 024 |
|--------|----------:|----------:|---------:|---------:|----------:|
| Architecture | DeepLabV3+ | DeepLabV3+ | UNet | UNet | DeepLabV3+ |
| Backbone | ResNet-50 | ResNet-50 | ResNet-34 | ResNet-34 | ResNet-50 |
| Channels | 5 | 5 | 5 | 4 | 5 |
| Patch size | 1024 | 1024 | 512 | 512 | 1024 |
| Warmup | Yes | No | No | No | Yes |
| **Val F2** | **0.423** | 0.405 | 0.367 | 0.347 | 0.295 |
| Val F1 | 0.325 | 0.267 | 0.237 | 0.289 | 0.324 |
| Val Precision | 0.289 | 0.170 | 0.149 | 0.227 | 0.388 |
| Val Recall | 0.372 | 0.617 | 0.580 | 0.400 | 0.278 |
| Val AP | 0.257 | 0.185 | 0.156 | 0.228 | 0.273 |
| **Test F2** | 0.278 | **0.385** | **0.341** | 0.277 | 0.197 |
| Test F1 | 0.278 | **0.259** | 0.222 | 0.248 | 0.236 |
| Test Precision | 0.278 | 0.168 | 0.140 | 0.210 | 0.357 |
| Test Recall | 0.278 | 0.570 | 0.530 | 0.301 | 0.177 |
| Test AP | 0.210 | 0.168 | 0.145 | 0.180 | 0.217 |
| Test IoU† | 0.161 | 0.149 | 0.125 | **0.141** | 0.134 |

†IoU is reported for completeness. Without polygon dilation, sub-pixel boundary errors and parcel-edge label noise inflate false positives and false negatives at every parcel boundary; this systematically deflates IoU for all models and makes cross-model comparisons via IoU unreliable for this task. F2 and F1 aggregate over the full vacant pixel set and are less sensitive to boundary artifacts.

**Note on run 027 split inversion.** Run 027 uses `patch_splits_1024_v2.json`, which assigns val=Brooklyn (BoroCode 3) and test=Bronx (BoroCode 2) -- the inverse of all other runs. This means run 027's "val" metrics correspond to Brooklyn and "test" metrics to the Bronx. The cross-borough comparison in Section 5.8 normalizes by borough to enable fair comparison.

**A note on AP.** Average Precision values are low across all runs (best test AP 0.217, run 024), but this does not straightforwardly indicate poor discriminative ability. AP is bounded below its theoretical maximum by irreducible label noise: the model's high-confidence vacant predictions include parcels that appear visually vacant but are administratively non-vacant -- parking lots with mismatched tax codes, recently cleared lots not yet reclassified, parcels where administrative and visual vacancy diverge by definition. These are counted as false positives at all thresholds, deflating precision throughout the PR curve and suppressing AP regardless of model quality. AP is best interpreted here as a relative measure between models rather than an absolute indicator of performance, and as a lower bound on true discriminative ability rather than an estimate of it.

**F2 as primary metric.** By raw val F2, run 027 achieves the highest score (0.423) of any run. However, run 018 achieves the highest test F2 (0.385). This apparent contradiction is explained by the split inversion: run 027's "val" is Brooklyn and "test" is Bronx, while run 018's "val" is Bronx and "test" is Brooklyn. When compared by borough rather than by split label (Section 5.8), run 027 achieves the highest F2 on Brooklyn (42.3%) while run 018 achieves the highest F2 on the Bronx (40.6%). Among UNet runs, run 030 achieves the highest test F2 (0.341). The high-F2 models share a common trait: high recall (0.37--0.62) at the cost of precision (0.14--0.29). For a screening application, this tradeoff is appropriate -- false positives can be discarded by a human reviewer, while missed vacant lots (false negatives) represent permanently lost opportunities.

**F2 vs. F1 operating points.** F2 and F1 reflect different use cases for the same model output. F2-optimal threshold selection (used here) pushes the decision boundary lower, accepting more false positives to gain recall. At the F1-optimal threshold, precision and recall are more balanced and the false positive burden on a human reviewer is lower, but more vacant lots are missed. A practitioner deploying the model should select the threshold based on the capacity of their review workflow: if one analyst is reviewing a shortlist of 500 flagged parcels, F1-optimal is appropriate; if systematic coverage of all vacant land is required regardless of reviewer burden, F2-optimal is correct.

**Val-test generalization.** Run 018 shows a moderate val-to-test drop in F2 (0.405 → 0.385, 5% relative), consistent with Bronx (val) and Queens (train) sharing more industrial and residential urban fabric than Brooklyn (test). Run 031 shows a larger relative F2 drop (0.347 → 0.277, 20%), likely because the 4ch model without a building probability signal is more sensitive to cross-borough distribution shift in bare-surface textures.

**Summary.** For the screening use case this model is designed for, run 027 (DLV3+, ResNet-50, 5ch, 1024px, v2 mask, warmup) is the recommended model based on the highest pixel-level F2 on Brooklyn (42.3%), the most architecturally diverse and challenging test borough. Run 018 (identical architecture, no warmup) remains the strongest model on the Bronx (40.6%) and achieves higher recall at the expense of precision. Run 030 (UNet, ResNet-34, 5ch, 512px) is the best alternative if a UNet architecture is preferred for deployment simplicity.

## 5.8 Cross-Borough Generalization

Because run 027 uses an inverted borough split (val=Brooklyn, test=Bronx) relative to all other runs (val=Bronx, test=Brooklyn), direct comparison of "val" and "test" metrics across runs is misleading. Table 5.4 normalizes by borough: each cell reports the F2 and precision achieved on a specific borough, regardless of whether it was labeled "val" or "test" for that run.

**Table 5.4: F2 by borough (at each run's F2-optimal threshold)**

| Arch | Run | Val Borough | Opt Thr | Brooklyn P | Brooklyn R | Brooklyn F2 | Bronx P | Bronx R | Bronx F2 |
|------|-----|-------------|--------:|-----------:|-----------:|------------:|--------:|--------:|---------:|
| DLV3+ | 018 | Bronx | 0.500 | 16.6% | 56.2% | 38.1% | 17.0% | 62.2% | 40.6% |
| DLV3+ | 027 | Brooklyn | 0.298 | 19.0% | 61.1% | **42.3%** | 18.5% | 53.9% | 39.0% |
| UNet | 029 | Bronx | 0.488 | 13.9% | 52.9% | 33.9% | 15.3% | 63.5% | 39.0% |
| UNet | 030 | Bronx | 0.500 | 14.0% | 57.5% | 35.4% | 14.3% | 62.4% | 37.3% |
| UNet | 032 | Bronx | 0.370 | 14.8% | 53.2% | 35.0% | 15.9% | 59.0% | 38.2% |

Several patterns emerge:

- **Run 027 achieves the highest Brooklyn F2 (42.3%)** of any run, likely because it was tuned on Brooklyn during validation. Run 018, tuned on the Bronx, achieves the highest Bronx F2 (40.6%). This suggests that threshold selection on a specific borough biases the operating point toward that borough's distribution.

- **The Bronx appears easier than Brooklyn across all models.** UNet runs trained with val=Bronx consistently score higher on the Bronx (37--39%) than on Brooklyn (34--35%). This may reflect the Bronx's more uniform urban fabric -- predominantly residential with regular block structure -- compared to Brooklyn's greater architectural diversity (brownstones, waterfront, industrial).

- **DeepLabV3+ outperforms UNet on both boroughs.** The best DLV3+ scores (40.6% Bronx, 42.3% Brooklyn) exceed the best UNet scores (39.0% Bronx, 35.4% Brooklyn), consistent with the 1024px patch providing more spatial context for the ASPP module.

- **Precision remains low (14--19%) across all models and boroughs.** This is the fundamental constraint of the task: many non-vacant surfaces are spectrally identical to vacant land, establishing a floor on false positive rate. At these precision levels, roughly 4--6 out of every 7 pixels predicted vacant are false positives at the pixel level. Section 6 analyzes which land-use categories drive these false positives.

## 5.9 Parcel-Level Evaluation

Pixel-level metrics capture model performance at the resolution of individual pixels, but the operational question is parcel-level: does the model flag enough pixels within a vacant parcel to detect it? To bridge this gap, per-parcel prediction fractions were computed for the best model (run 027, DLV3+, Brooklyn val split) at the F2-optimal threshold of 0.298.

For each of the 9,741 vacant parcels within the Brooklyn prediction TIF, the fraction of valid pixels predicted vacant was computed using rasterio polygon masking. Of these, 9,718 had at least 5 valid pixels and were considered evaluable.

**Table 5.5: Parcel-level recall at varying coverage thresholds (run 027, Brooklyn)**

| Coverage threshold | Detected | Total | Recall |
|-------------------:|---------:|------:|-------:|
| >= 10% pixels vacant | -- | 9,718 | 45.7% |
| >= 20% pixels vacant | -- | 9,718 | 40.7% |
| >= 30% pixels vacant | 3,210 | 9,718 | 33.0% |
| >= 50% pixels vacant | 1,338 | 9,718 | 13.8% |
| >= 70% pixels vacant | 234 | 9,718 | 2.4% |

The distribution of per-parcel prediction fractions is heavily skewed: the median prediction fraction across all evaluable vacant parcels is just 0.033 (3.3%), while the mean is 0.193 (19.3%). This means the model is partially detecting most parcels -- assigning above-zero probability to many pixels -- but rarely floods an entire parcel with high-confidence predictions. The 61% pixel-level recall from Section 5.7 is composed of partial detections spread across many parcels rather than complete detections of fewer parcels.

**Borough breakdown** within the TIF further illustrates the distribution shift:

| Borough | Detected (>= 30%) | Total | Recall |
|---------|-------------------:|------:|-------:|
| Brooklyn | 3,126 | 7,611 | 41.1% |
| Queens | 84 | 1,940 | 4.3% |
| Manhattan | 0 | 167 | 0.0% |

Brooklyn parcels, which this model was tuned on, achieve 41.1% parcel recall at the 30% coverage threshold. Queens parcels within the TIF extent (southern edge of the training borough) achieve only 4.3%, and the 167 Manhattan parcels within the TIF are entirely missed -- consistent with Manhattan's exclusion from training due to tall building occlusion.

**Interpretation for deployment.** At a 10% coverage threshold (any meaningful detection), the model identifies 46% of vacant parcels in Brooklyn -- nearly half of all vacant lots are flagged for human review. At 30%, the model still identifies a third of vacant parcels. For a city planner generating a shortlist, 10--20% coverage is likely the appropriate threshold: it maximizes the number of vacant parcels surfaced while keeping false positives manageable through manual review. The low median prediction fraction (3.3%) suggests that the model's pixel-level predictions should be aggregated at the parcel level rather than used as raw pixel masks for decision-making.

## 5.10 Qualitative Analysis

Qualitative inspection of model predictions was conducted primarily on UNet run 015 (ResNet-50, 4ch, v1 mask, 512px). This is not the best-performing run by F2 -- systematic inspection of its predictions on the Bronx and Brooklyn sets was the primary driver of v2 mask corrections, which subsequently improved all Phase 3 runs. The failure modes described here are structural to the task and imagery and remain applicable to the best Phase 3 models, as confirmed by spot-checking predictions from runs 018 and 030.

**Note:** The observations below were conducted at the default 0.5 threshold. At the F2-optimal threshold (which is lower), many of the reported false negatives become true positives, and some additional false positives appear. The F2-optimal threshold represents a more appropriate operating point for a screening application.

### Successes

The model reliably identifies several categories of vacant land:

- **Grassy vacant lots in Queens (training set):** Large, predominantly green vacant lots are detected with high confidence. The model assigns probabilities well above 0.5 to these parcels.
- **Large bare soil/dirt patches:** Clearly exposed soil areas are detected, e.g., BBL 3088762724 (Brooklyn, large dirt patch correctly identified).
- **Soccer fields as non-vacant:** The model consistently and correctly classifies soccer fields as non-vacant (e.g., BBLs 4023640023, 3030680001, 3027800001), indicating that it has learned to distinguish maintained recreational surfaces from vacant land.
- **Organized parking lots:** Parking lots with visible line markings are generally classified as non-vacant (e.g., BBL 2027320001).
- **Parks:** Well-maintained parks with clear structure (paths, equipment, maintained lawns) are generally not confused with vacancy.

### Systematic Failure Modes

The following failure modes are discussed in detail in Section 6 (Error Analysis):

- **Parking lots without lines** are frequently classified as vacant, particularly in industrial areas where lots are unpaved or disorganized.
- **Roads and highways within parcel boundaries** trigger false positives because the model has learned that asphalt within parcels (as opposed to roads masked to 255) is associated with vacancy. This was partially addressed by the v2 planimetric roadbed mask.
- **Under-construction parcels** are inconsistently classified due to label ambiguity.
- **Baseball diamonds and tennis courts** generate false positives, while soccer fields do not.
- **Dense tree cover in suburban areas** is sometimes classified as vacant.
- **Skinny or very small lots** are below the model's effective resolution and are consistently missed.

### Borough-Level Patterns

**Queens (training set):** Strongest performance on typical grassy vacant lots. The primary systematic issue is highway/road parcels -- many parcels labeled vacant by MapPLUTO are obstructed by elevated highways or are highway shoulders (e.g., cluster of BBLs around 4025170006--4025290071).

**Bronx (validation set):** Model probabilities on true vacant lots often sit below the 0.5 threshold but above 0. The industrial zones of Hunts Point are particularly challenging -- large warehouse and port areas with sparse textures are systematically misclassified. Some vacant lots in the Bronx have roads running through them that are not masked (e.g., around 2027310005).

**Brooklyn (test set):** Confirms the failure modes observed in the Bronx. Notable is the presence of visually indistinguishable adjacent parcels with opposite labels -- e.g., BBL 302608001 (vacant parking lot) and BBL 3025850001 (non-vacant parking lot) sit side by side and are spectrally identical. This represents irreducible label noise that sets a ceiling on achievable performance.

## 5.11 Future Results

**[PLACEHOLDER: Cross-City Inference]** An optional extension of this work would apply the trained models to NAIP imagery from another city (e.g., Philadelphia, Chicago) where parcel-level vacancy labels are available for evaluation. This would test whether the vacancy signal learned from NYC transfers to cities with different urban fabrics. However, this requires identifying and processing parcel data from the target city, which may not be feasible within the current timeline.
