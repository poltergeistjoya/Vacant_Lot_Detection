# Probability Calibration

## Why calibration matters here

The model is trained with `pos_weight=30` in the BCE loss to compensate for the 3.4%
vacancy rate. This reweighting changes the optimal decision boundary: the model learns to
output high probabilities for vacants more aggressively than a standard BCE model would.
As a result, the raw sigmoid outputs are **decision scores**, not estimates of
P(vacant | pixel). They're useful for thresholding but not for downstream tasks that
interpret the magnitude — risk maps, parcel scoring, or any application that aggregates
probabilities across pixels.

Calibration maps those decision scores to well-founded probabilities.

---

## Methods

Two post-hoc calibration methods are fit on the **val split** (held out from training)
and evaluated on both val and test.

### Temperature scaling

A single scalar T is applied to all logits before sigmoid:

```
p_calibrated = sigmoid(logit / T)
```

T > 1 softens predictions toward 0.5. T < 1 sharpens them. T = 1 is the identity.
T is found by minimizing the weighted negative log-likelihood on val logits.

### Platt scaling

A logistic regression `sigmoid(a · logit + b)` is fit on val logits. The two free
parameters allow both slope correction (a) and intercept correction (b).

---

## Critical implementation detail: weighting the NLL

Both methods are fit by minimizing NLL over a stratified reservoir sample (75k positive
+ 75k negative pixels). A naive `np.mean(NLL)` over this balanced sample implicitly
assumes a 50% vacancy rate. That is wrong — it finds the temperature that best separates
classes at parity, not the temperature that calibrates against the true 3.4% prior.

The correct objective weights each sample by its inverse sampling frequency:

```
w_pos = total_pos_pixels_seen / n_pos_sampled   # ≈ 20–30×
w_neg = total_neg_pixels_seen / n_neg_sampled   # ≈ 500–600×
```

This makes the optimizer care mostly about the low-probability regime (where 96.6% of
mass lives), which is the right behavior. Without this fix, temperature optimization
returned T = 5.48, which collapsed all predictions toward 0.5 and worsened ECE from
0.042 to 0.219. With correct weighting, T = 1.17.

---

## Results on kahan_027

| Method      | val ECE | test ECE |
|-------------|---------|----------|
| Raw         | 0.042   | 0.050    |
| Temperature | 0.044   | 0.052    |
| Platt       | 0.005   | 0.007    |

**Temperature (T = 1.17):** Nearly the identity — the model's logit scale is already
close to correct for the true prior. Marginal change in ECE either direction.

**Platt (a = 0.475, b = −1.64):** The slope `a < 1` compresses the logit range
(moderating overconfident high-probability predictions). The intercept `b = −1.64`
shifts predictions downward, adjusting for the pos_weight-inflated baseline. ECE drops
to ~0.5% on val and ~0.7% on test — well-calibrated by the reliability diagram.

Platt scaling is the recommended calibration for any application that uses raw probability
values rather than a fixed threshold.

---

## Why not isotonic regression?

Isotonic regression fits a free-form monotonic mapping with many more degrees of freedom
than Platt's two parameters. It sounds appealing for a large dataset, but the constraint
here is not total pixel count — it's the distribution of predicted probabilities.

~80% of weighted sample mass falls in the first probability bin (0–0.067). Above p ≈ 0.3,
the bins are extremely sparse regardless of total sample size, because the model rarely
predicts high vacancy probability. Isotonic regression would fit arbitrary step-function
jumps to handfuls of points in that upper region, producing an erratic and overfit
calibration curve. Platt's rigidity (slope + intercept) is a feature here: it can't chase
noise in sparse bins.

The miscalibration pattern — smooth, consistent overconfidence across the prediction
range — is well-captured by Platt's two parameters. Isotonic would be worth revisiting
if a future model produced a denser spread of high-probability predictions.

---

## Pixel-level vs parcel-level calibration

The current calibration operates at the **pixel level**: each pixel's logit is
independently corrected. This answers "when the model predicts 20% vacancy for a pixel,
are 20% of such pixels truly vacant?"

For downstream parcel-level decisions (flagging tax lots for review), a separate
**parcel-level** calibration may be more appropriate. The parcel-level question is
different: "when the model assigns a parcel a 30% vacancy score, are 30% of parcels at
that score actually vacant?" The answer depends on how pixel predictions are aggregated
per parcel (mean probability, fraction above threshold, max, etc.), and each aggregation
method would need its own calibration curve.

Pixel-level calibration is reported here because it characterizes the model's raw
behavior and is aggregation-agnostic. Parcel-level calibration is a deployment concern
that depends on the chosen aggregation strategy.

---

## What calibration does not fix

Calibration adjusts the probability scale but **does not change the ranking** of
predictions. AP, AUROC, and F-scores at an optimal threshold are identical before and
after calibration. A model with AP = 0.21 has AP = 0.21 after Platt scaling.

Ablation comparisons therefore use **uncalibrated outputs** at the F2-optimal threshold
(0.298). Calibration is a post-hoc correction applied to the final model for downstream
use — it is not an architectural choice and should not appear in the ablation table.

---

## ECE definition

Expected Calibration Error is computed over equal-width probability bins (15 bins, width
1/15 each). Bins with fewer than 50 samples are excluded.

Within each bin b:

- **confidence**: weighted mean predicted probability
- **accuracy**: weighted fraction of true positives
- **weight**: fraction of total sample weight in bin b

```
ECE = Σ_b  weight_b · |confidence_b − accuracy_b|
```

All weights reflect the true class distribution via the inverse-frequency correction
described above, so ECE is not biased by the balanced reservoir.

---

## Reading a reliability diagram

The x-axis is the model's predicted probability for a group of pixels (binned); the
y-axis is the fraction of those pixels that were truly vacant. A perfectly calibrated
model falls on the diagonal.

- **Curve below diagonal** → overconfident. The model predicts higher vacancy probability
  than reality. Example: predicts 20%, but only 6% are truly vacant.
- **Curve above diagonal** → underconfident. The model underestimates vacancy.

The histogram (bottom panel) shows the weighted fraction of predictions in each bin.
For a 3.4% vacancy rate, the vast majority of mass concentrates in the first bin — this
is correct behavior, not a bug.

### Discrimination vs calibration

A model can have reasonable F1/IoU but poor calibration if its probability outputs are
uninformative — i.e., it gets the count of flagged pixels roughly right at a threshold
but the probabilities themselves don't correlate with true vacancy likelihood. The
reliability curve exposes this: a model with good discrimination shows a steeply rising
curve; a model with poor discrimination shows a nearly flat curve regardless of predicted
probability. Aggregate metrics (F1, IoU) do not distinguish between these cases.
