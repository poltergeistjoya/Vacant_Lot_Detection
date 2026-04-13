# Mask v2 & Training Stabilization Changelog

Changes applied to the vacancy mask pipeline and training loop, in order.
`kahan_018` was the first run trained after change 1.

---

## 1. Vacancy mask v2 — BBL corrections + planimetric roadbed

**Mask file**: `outputs/labels/vacancy_mask_v2.tif`

### BBL overrides (from `docs/MASK_TUNING.md` review)

- **49 omit_bbls** (parcel -> 255 ignore):
  - Pre-existing shadow parcels + offshore islands (10)
  - Rikers Island (1)
  - Under-construction parcels in Bronx, Brooklyn, Queens (9)
  - Brooklyn water-edge / extends-into-water parcels (7)
  - Brooklyn too-small / obstructed (3)
  - Queens elevated-highway obstruction cluster (5)
  - Queens highway-obstructs-real-vacant (2)
  - Queens highway parcel (1)
  - Queens DOT-owned highway parcels not caught by road mask (9)
  - Queens under-construction (2)

- **11 force_nonvacant_bbls** (parcel -> 0):
  - Bronx/Brooklyn/Queens parcels with buildings mislabeled as vacant (5)
  - Queens road parcels mislabeled as vacant (6)

- **7 force_vacant_bbls** (parcel -> 1):
  - 1 vintage-confirmed (MapPLUTO 22v3 G1 -> 23v2 V1)
  - 6 user-confirmed from NAIP visual review

### Road mask: NYC Planimetric Roadbed

Replaced TIGER road centerlines (thin 1px lines) with NYC DoITT Planimetric Database
roadbed polygons (`data/parcels/nyc/NYC_Planimetrics_2022.gdb`, layer `ROADBED`).
104,959 MultiPolygon features representing actual paved road surfaces. Burned to 255
after erosion step.

TIGER roads disabled (`roads_mask.enabled: false`).

### Water mask: disabled

Water mask (`labels.water_mask.enabled: false`) removed. Model rarely confused water
with vacant (one Bronx case, handled by omit_bbls). Blanket water burn was masking
legitimate waterfront-adjacent vacant parcels.

### Pixel impact (v2 vs v1, per borough)

| Borough | Vacant delta | Ignore delta |
|---------|-------------|-------------|
| Manhattan | -296K | +4.7M |
| Bronx | -296K | +11.1M |
| Brooklyn | -385K | +9.5M |
| Queens | -1.39M | +41.4M |
| Staten Island | -2.15M | +9.8M |

### Cemetery extension (documentation only, no mask change)

St. Raymond's Cemetery extension parcels are labeled vacant per MapPLUTO (V-class).
City is the source of truth. These stay as-is in the mask but are noted as a known
source of label noise: `2055740001, 2055700001, 2055700156, 2055700152, 2055700112,
2055700144, 2055700128, 2055700272, 2055700100, 2055700105`.

---

## 2. LR scheduler: CosineAnnealingWarmRestarts option

Added `cosine_t_mult` config field. When `cosine_t_mult > 1`, the scheduler switches
from `CosineAnnealingLR` (symmetric oscillation) to `CosineAnnealingWarmRestarts`
(hard reset to max LR at each cycle boundary). Default `cosine_t_mult: 1` preserves
original `CosineAnnealingLR` behavior.

`eta_min=1e-6` was already set.

---

## 3. Training stabilization: warmup + gradient clipping

After `kahan_019` and `kahan_020` both collapsed at epoch 2 (early-stopped at 42)
with identical config to `kahan_018` (which converged at epoch 76), two stabilization
measures were added:

- **Linear LR warmup** (`warmup_epochs: 5`, default): LR ramps from 0.1% to 100% of
  target over the first 5 epochs. Prevents large gradient spikes from Lovasz + high
  pos_weight before the model has learned basic features.

- **Gradient clipping** (`grad_clip_norm: 1.0`, default): Caps gradient norm per
  optimizer step. Prevents occasional extreme gradients from destabilizing weights.

Both are controlled via `training` config and can be disabled (`warmup_epochs: 0`,
`grad_clip_norm: 0`).

Root cause of instability: `pos_weight=30` with 3.4% vacancy rate creates large
gradient magnitudes. Combined with Lovasz loss (which optimizes IoU directly and
produces non-standard probability distributions), the loss landscape has a narrow
convergence basin. CUDA non-determinism in batch ordering is enough to tip some runs
into immediate collapse.
