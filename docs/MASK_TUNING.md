# Tuning Log

## Status
Active evaluation. Priority: **mask cleanup** before further training. Building segmentation mask created — apply before next training run.

**Column key:**
- **Result**: FP = predicted vacant / label non-vacant | FN = predicted non-vacant / label vacant | TP = correctly predicted vacant | TN = correctly predicted non-vacant
- **Mask Action**: `keep` = no change | `→no-data` = set to 255/ignore | `→non-vacant` = set to 0 | `→vacant` = set to 1 | `investigate` = check newer MapPLUTO

---

## Bronx — Val Set

### General
- Model probabilities on true vacant lots are clearly non-zero but often sit below 0.5. Use **F2-optimal threshold** instead of 0.5 — many FNs may dissolve.

### Per-Parcel Observations

| BBL | Result | Mask Action | Notes |
|-----|--------|-------------|-------|
| `2048380031` | FP | →vacant | Label probably wrong — looks vacant in imagery |
| `2048490039` | FP | →no-data | Under construction |
| `2023720047` | FP | →no-data | Under construction |
| `2027320001` | FP | keep | Non-vacant parking lot with diagonal lines — model still calls it vacant |
| `2027700001` | FP | keep | Large sparse food warehouse, Hunts Point — industrial scale misclassified |
| `2027810500` | FP | keep | Same — sparse port/warehouse texture |
| `2027310005` | FP | keep | Railroad segment — sparse asphalt triggers vacant |
| `2026090039` | FP | keep | Backyard of single-family home (one parcel) |
| `2026090037` | FP | keep | Same |
| `2027750009` | FP | keep | Disorganized industrial parking lot |
| `2027800002` | FP | keep | Dirt/disorganized corner at Hunts Point water treatment plant |
| `2027330058` | FP | keep | Stacked/disorganized parking — texture triggers vacant |
| `2048190013` | FN | →non-vacant | Labeled vacant, clearly has a building |
| `2048250041` | FN | →non-vacant | Same |
| `2048370055` | FN | →non-vacant | Same |
| `2027310065` | FN | →no-data | Vacant lot under highway — obstructed |
| `2023720031` | FN | keep | Skinny lot — hard case |
| `2026790015` | FN | keep | Dense tree-covered vacant lots — may be acceptable (product decision on tree clearing) |
| `2026130034` | FN | keep | Grassy + dirt patch — model should get this |
| `2016130048` | FN | keep | Vacant parking lot — recovers at lower threshold |
| `2027750097` | FN | keep | Vacant lined parking lot — recovers at lower threshold |
| `2027810300` | FN (partial) | update with bldg seg | Building within vacant parcel — model correctly predicts building pixels as non-vacant but gets penalized; fix with building footprint mask |
| `2026050040` | — | →no-data | Rikers Island — anomaly, should be excluded |
| `2048040100` | partial | keep | Park edge — track/orderly parking correct, tree+dirt edge missed |
| `2026080040` area | TN | keep | Dense trees correctly not called vacant |
| `203600010` area | TP (questionable) | investigate | Labeled vacant but looks like private parking/tennis courts — check newer MapPLUTO |
| `2055700156` area | TP (questionable) | →non-vacant | St. Raymond's Cemetery extension coded vacant — doesn't look vacant |

### Systemic Issues (No Single BBL)
| Issue | Result | Mask Action | Notes |
|-------|--------|-------------|-------|
| Water bodies | FP | →no-data (global) | No water class in training; burn NHD/OSM water as 255 |
| Highway parcels | FP | →no-data (global) | Roads are no-data so model learned asphalt → vacant; highway-inside-parcel pixels not excluded |
| Dense suburban tree cover | FP | keep | Suburban Bronx leafy residential — not enough training examples |
| Partially grassy + dirt lots | FN | keep | Model inconsistent — sometimes gets them, sometimes not |

---

## Brooklyn — Test Set

### General
- Most Bronx failure modes confirmed: roads, parking, industrial, grassy+dirt inconsistency.
- Building segmentation expected to help significantly.
- `302608001` (vacant) and `3025850001` (non-vacant) are adjacent and visually indistinguishable — irreducible label noise floor.
- Model correctly handles soccer fields (`3030680001`, `3027800001`) — TN; inconsistent on baseball diamonds, tennis courts.

### Per-Parcel Observations

| BBL | Result | Mask Action | Notes |
|-----|--------|-------------|-------|
| `3085910075` | FP | keep | Floyd Bennett Field — trees and dirt paths, not vacant |
| `3085810100` | FN | keep | Swampy/wetland vacant lot — threshold-sensitive |
| `3085900700` | FP | keep | Dirt area with boat parking — not vacant |
| `3088665540` | →no-data | →no-data | Skinny lot at water edge — visually water |
| `3088661897` | →no-data | →no-data | Same |
| `3088661767` | →no-data | →no-data | Same |
| `3088661830` | →no-data | →no-data | Same |
| `3088762724` | TP | keep | Large dirt+grassy patch — model gets it correct |
| `3088661764` | FN | keep | Small dirt patch — model misses (gets large version but not small) |
| `3087900079` | FN | →no-data | Too small / obstructed |
| `3087900042` | FN | →no-data | Same |
| `3087870030` | FN | →no-data | Same |
| `3087740006` | FP | keep | Lined parking lot — model calls part vacant |
| `3086640411` area | FN | keep | Kinda-green vacant parcels — model misses entirely |
| `3022410039` | FN | →non-vacant | Labeled vacant, clearly has a building |
| `3022460001` | FP | keep | School city block — model correctly calls building non-vacant but calls tennis courts vacant |
| `3030760030` | FP | →no-data | Under construction, non-vacant — label in flux; no-data is fairest |
| `3030680001` | TN | keep | Soccer field — correctly non-vacant |
| `3027800001` | TN | keep | Soccer field — correctly non-vacant |
| `3023690019` | FN | →non-vacant | Labeled vacant, clearly has a building |
| `3023400012` | FN | →no-data | Parcel extends into water |
| `3023320040` | FN | →no-data | Same |
| `3023240060` | FN | →no-data | Same |
| `3027420009` | FP | →no-data | Under construction — label in flux |
| `3028590001` | FP | keep | Manicured lawn — model calls vacant |
| `3028370001` | FP | keep | Baseball field + green around industrial lot — looks decrepit but isn't vacant |
| `302608001` | FN | keep | Vacant parking lot — visually indistinguishable from adjacent non-vacant `3025850001` |
| `3025850001` | FP | keep | Non-vacant parking lot — visually indistinguishable from adjacent vacant `302608001` |
| `3047180049` | FN | keep | Dirt+grass vacant lot — track for future evaluation |
| `3036210049` | FP | investigate | MapPLUTO says non-vacant, model says vacant, no visible structure — may actually be vacant |
| `3057250006` | FP | keep | Trees around LIRR — not vacant |
| `3057490015` | FP | keep | Baseball diamond — model calls vacant |
| `3009030180` | FP | keep | MetroNorth train yard — industrial dirt + trucks texture |
| `3038490001` | FP | keep | Railroad area — not vacant |
| `3037640002` | FP | keep | Auto body shop — model calls vacant |

### Systemic Issues (No Single BBL)
| Issue | Result | Mask Action | Notes |
|-------|--------|-------------|-------|
| Roads not masked | FP | →no-data (global) | Several road pixels flagged as vacant |
| Darker green = vacant, green+dirt ≠ vacant | FN | keep | Inverted from expected; grassy+dirt harder than pure dark green |
| Under construction lots | FP | →no-data (global) | No-data is fairest — legal status in flux |

---

## Queens — Train Set

### General
- Grassy vacant lots are the model's strongest suit in Queens — multiple correct detections.
- Soccer fields correctly non-vacant; baseball diamond without traditional bases also handled.
- Highway/road masking is the most systemic issue: many parcels labeled vacant are visually roads; elevated highways obstruct clusters of real vacant lots.
- **Road mask proposal**: burn road geometry as non-vacant in training; mask to ignore when computing val/test IoU (otherwise these are too easy and inflate metrics).

### Per-Parcel Observations

| BBL | Result | Mask Action | Notes |
|-----|--------|-------------|-------|
| `4113430030` area | TP | keep | Baseball diamond (no bases) — model gets it; atypical |
| `4114870001` | FP | keep | House + asphalt patch on one large lot — model calls asphalt vacant; acceptable error |
| `4114880059` | FP | keep | Recreational area — model calls part vacant |
| `4115430002` | FP | keep | Green area around Aqueduct Racetrack — model calls vacant; may shift in non-peak-vegetation imagery |
| `4118470040` | FP | →no-data | Under construction — label in flux |
| `4118930016` | FP | →non-vacant | Mask says vacant but obviously a road |
| `4118940016` | FP | →non-vacant | Same |
| `4118970020` | FP | →non-vacant | Same |
| `4118480050` area | FP | →no-data | Highway — vacancy mask doesn't make sense here |
| `4116070033` | FP | keep | House with huge lot — model calls lot portion vacant |
| `4098260049` | TP | keep | Grassy vacant lot — model correctly identifies |
| `4098430003` | TP | keep | Same |
| `4098430002` | TP | keep | Same |
| `4098430001` | TP | keep | Same |
| `4097960063` | FP | →no-data | Under construction |
| `4098940048` | FP | investigate | Just grass, looks vacant — MapPLUTO says non-vacant |
| `4098940047` | FP | investigate | Same |
| `4098940051` | FP | investigate | Same |
| `4099380059` | FP | keep | Dense suburban trees — model calls vacant; same failure as Bronx |
| `4099500006` | TP | keep | Vacant green lot with lots of greenery — model correctly identifies |
| `4099500064` | FP | →no-data | Under construction |
| `4099310012` | FP | keep | Green patch within large lot — model calls vacant |
| `4104990092` | FN | →non-vacant | Labeled vacant, clearly has a building |
| `4023640023` | TN | keep | Soccer field — correctly non-vacant |
| `402510001` | TP | →non-vacant | MapPLUTO says vacant, model gets it, but it's a highway road — wrong training signal |
| `4025090001` | TP | →non-vacant | Same |
| `4025170006` | — | →no-data | Obstructed by elevated highway |
| `4025190150` | — | →no-data | Same |
| `4025200052` | — | →no-data | Same |
| `4025210040` | — | →no-data | Same |
| `4025290071` | — | →no-data | Same |
| `40001190020` | FP | →non-vacant | Road/highway labeled vacant |
| `4004270045` | — | →no-data | Highway obstructs real vacant parcel below |
| `4004650300` | — | →no-data | Same |
| `4004120025` | FP | investigate | Looks vacant (probably parking lot) — MapPLUTO says non-vacant |
| `4003970005` | FP | investigate | Looks vacant — MapPLUTO says non-vacant |
| `4003400136` | FN | keep | Very skinny vacant lot — model cannot identify |
| `4005790040` | FP | →no-data | Under construction |
| `4005790041` | FP | →no-data | Same |
| `4004900100` | TN | investigate | Model correctly says non-vacant but half looks vacant — check what this parcel is |
| `4004900110` | FP | keep | Waterfront green patch — not vacant; looks like biodiversity opportunity |
| `4005100017` | FN | keep | Green patch with trees, mainly grass — model misses |
| `4008870001` | FP | keep | Tiny cemetery — gravestones not visible; model calls vacant |
| `4009000020` | FN | keep | Vacant parking lot — model doesn't register as vacant |
| `4009140022` | FP | investigate | MapPLUTO says non-vacant, model says vacant, user agrees looks vacant |
| `4011840001` | partial | keep | Parking lot divided into 4 sub-lots; middle one marked non-vacant; confusing parcel split |
| `4010550054` | TP | keep | Model correctly identifies vacant lot but spills into adjacent parcels (unusually fragmented boundaries) |
| `4004880003` area | TP | keep | Model correctly calls vacant — looks like it could be a park; keep an eye on this |

### Systemic Issues (No Single BBL)
| Issue | Result | Mask Action | Notes |
|-------|--------|-------------|-------|
| Highway-labeled parcels | FP / TP (wrong) | →non-vacant or →no-data | Road parcels labeled vacant poison training signal |
| Elevated highway obstruction cluster | — | →no-data | Group around `4025170006` etc. |
| Industrial areas | FP | keep | Consistent with Bronx/Brooklyn — model broadly misclassifies industrial texture |
| Dense suburban trees | FP | keep | Consistent cross-borough failure |
| Under construction | FP | →no-data (global) | Best practice: no-data since label is in flux |

---

## Cross-Cutting Issues (All Boroughs)

| Issue | Boroughs | Notes |
|-------|----------|-------|
| Water bodies — no training signal | All | Burn NHD/OSM water as no-data globally |
| Roads inside parcels | All | "asphalt = vacant" learned from road no-data masking; root cause of highway/railroad/parking FPs |
| Under construction lots | All | No-data is fairest — NYC label changes when foundation is laid |
| Parking lots — disorganized/unlined | All | FP — looks like vacant bare land |
| Parking lots — lined/organized, actually vacant | All | FN — line cue works both ways |
| Industrial / port / yard areas | All | FP — large sparse industrial texture misread as vacant |
| Baseball diamonds | BK, QN | FP — model has no concept of sports fields |
| Tennis courts | BK | FP — model calls them vacant |
| Soccer fields | TP across BK, QN | TN — correctly not vacant; most consistent |
| Dense suburban tree cover | All | FP — low-density leafy residential not in training |
| Skinny lots | All | FN — model cannot identify very narrow parcels |
| Grassy + dirt mix | All | Inconsistent — sometimes FN, sometimes TP |
| Building within vacant parcel | All | FN (partial) — fix globally with building footprint mask |

---

## Prioritized Action Items

### P0 — Do immediately
1. **F2-optimal threshold** — re-report val/test at F2 optimum from `pr_curves.npz` instead of 0.5.
2. **Water → no-data** — burn NHD/OSM water polygons into vacancy mask as 255.
3. **Apply building footprint mask** — building pixels within vacant parcels → non-vacant or no-data. (Building seg mask already created — apply now.)
4. **Fix obvious mislabels** from tracker below: parcels with confirmed buildings → non-vacant; highway-obstructed lots → no-data.

### P1 — Next
5. **Road geometry mask** — burn OSM/TIGER road geometry into parcel pixels as no-data for training; mask same pixels when computing val/test IoU.
6. **Mask Rikers Island** → no-data across all splits.
7. **Under-construction policy** — decide: uniform →no-data or keep as-is and accept noise.

### P2 — Investigate
8. **MapPLUTO vintage check** — parcels marked `investigate` in tracker; newer vintage may resolve stale labels.
9. **Tree-covered vacant lots** — product decision: flag or not? (Real consequence: may push city to clear trees.)
10. **Industrial / Hunts Point scope** — in-scope → need industrial training samples; out-of-scope → mask M-zoned parcels.

### P3 — Larger changes
11. Multi-class heads for water / road / building.
12. Suburban tree coverage — add training patches or document limitation.
13. Parking lot disambiguation — defer until P1 done.

---

## Open Questions
- Sheds within vacant parcels: remain vacant or flip to non-vacant after building mask intersection?
- Is industrial Bronx (Hunts Point) in scope for the city-management use case?
- Tree-covered vacant lots: flag or not?
- Rikers Island: removing helps model — is that too much of a shortcut? Recommendation: remove from both train and eval consistently.
- RGB-only vs. RGBN: worth ablating given grassy+dirt confusion? Chinese papers use RGB but with inflated scores — skeptical without controlled test.

---

## Mask Correction Tracker

### → Non-Vacant (has building or permanent structure)

| BBL | Borough | Split | Notes |
|-----|---------|-------|-------|
| `2048190013` | Bronx | Val | Clearly has a building |
| `2048250041` | Bronx | Val | Same |
| `2048370055` | Bronx | Val | Same |
| `2055700156` area | Bronx | Val | Cemetery extension — doesn't look vacant |
| `3022410039` | Brooklyn | Test | Clearly has a building |
| `3023690019` | Brooklyn | Test | Same |
| `4104990092` | Queens | Train | Clearly has a building |
| `4118930016` | Queens | Train | Obviously a road |
| `4118940016` | Queens | Train | Same |
| `4118970020` | Queens | Train | Same |
| `40001190020` | Queens | Train | Road/highway mislabeled |
| `402510001` | Queens | Train | Highway road mislabeled as vacant |
| `4025090001` | Queens | Train | Same |

### → No-Data / Ignore (obstructed, ambiguous, or extends into water)

| BBL | Borough | Split | Reason |
|-----|---------|-------|--------|
| `2026050040` | Bronx | Val | Rikers Island — anomaly |
| `2027310065` | Bronx | Val | Under highway — obstructed |
| `2048490039` | Bronx | Val | Under construction |
| `2023720047` | Bronx | Val | Under construction |
| `3088665540` | Brooklyn | Test | Skinny water-edge lot |
| `3088661897` | Brooklyn | Test | Same |
| `3088661767` | Brooklyn | Test | Same |
| `3088661830` | Brooklyn | Test | Same |
| `3087900079` | Brooklyn | Test | Too small / obstructed |
| `3087900042` | Brooklyn | Test | Same |
| `3087870030` | Brooklyn | Test | Same |
| `3023400012` | Brooklyn | Test | Parcel extends into water |
| `3023320040` | Brooklyn | Test | Same |
| `3023240060` | Brooklyn | Test | Same |
| `3030760030` | Brooklyn | Test | Under construction — label in flux |
| `3027420009` | Brooklyn | Test | Under construction |
| `4025170006` | Queens | Train | Elevated highway obstruction |
| `4025190150` | Queens | Train | Same |
| `4025200052` | Queens | Train | Same |
| `4025210040` | Queens | Train | Same |
| `4025290071` | Queens | Train | Same |
| `4004270045` | Queens | Train | Highway obstructs real vacant parcel |
| `4004650300` | Queens | Train | Same |
| `4118480050` area | Queens | Train | Highway — mask doesn't make sense |
| `4118470040` | Queens | Train | Under construction |
| `4097960063` | Queens | Train | Under construction |
| `4099500064` | Queens | Train | Under construction |
| `4005790040` | Queens | Train | Under construction |
| `4005790041` | Queens | Train | Under construction |

### → Vacant (label says non-vacant but looks vacant — investigate newer MapPLUTO)

| BBL | Borough | Split | Notes |
|-----|---------|-------|-------|
| `2048380031` | Bronx | Val | Looks vacant; label probably wrong |
| `3036210049` | Brooklyn | Test | No visible structure — may actually be vacant |
| `4098940048` | Queens | Train | Just grass, looks vacant |
| `4098940047` | Queens | Train | Same |
| `4098940051` | Queens | Train | Same |
| `4004120025` | Queens | Train | Probably parking lot, looks vacant |
| `4003970005` | Queens | Train | Looks vacant |
| `4009140022` | Queens | Train | Looks vacant — user agrees |

---

## Train Set Evaluation Command

```bash
just train::visualize --run outputs/models/unet/kahan_015 --splits train --patch-size 512 --patch-splits outputs/labels/patch_splits_512.json --stride 256 --threshold 0.318 --suffix threshold_318_train
```
