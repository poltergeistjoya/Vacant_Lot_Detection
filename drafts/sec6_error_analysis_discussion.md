# 6. Error Analysis and Discussion

Qualitative inspection of model predictions across all three evaluation boroughs reveals systematic failure modes that illuminate both the strengths and fundamental limitations of recovering administrative vacancy from aerial imagery. This analysis draws on per-parcel inspection of UNet run 015 (v1 mask) and the best-performing model (DLV3+ run 027, v2 mask, Brooklyn), with over 200 individual parcel-level observations documented across the Bronx, Brooklyn, and Queens.

## 6.1 False Positive Taxonomy

False positives -- non-vacant parcels that the model predicts as vacant -- cluster into six recurring categories.

### 6.1.1 Parking Lots

Parking lots are the single most frequent source of false positives, and they exhibit an interesting asymmetry. **Unlined or disorganized parking lots** (unpaved, no painted markings, vehicles scattered irregularly) are routinely classified as vacant. This is particularly prevalent in industrial areas of the Bronx (e.g., Hunts Point: BBLs 2027750009, 2027800002, 2027330058) and Brooklyn, where dirt or gravel lots with haphazardly parked vehicles share the spectral and textural signatures of actual vacant lots.

Conversely, **lined parking lots with organized vehicles** (diagonal or parallel parking with painted markings) are generally classified correctly as non-vacant (e.g., BBL 2027320001 in the Bronx). The presence of painted lines creates a regular geometric pattern that the model has learned to associate with developed land.

This failure mode is partly irreducible: MapPLUTO classifies some parking lots as vacant (code G7, "outdoor parking, unattended") while visually identical lots with active commercial use are classified as non-vacant. BBLs 302608001 (vacant) and 3025850001 (non-vacant) in Brooklyn sit adjacent to each other and are visually indistinguishable in NAIP imagery -- a human annotator would not be able to distinguish them without the tax record. Run 027 further illustrates this on Brooklyn: BBL 3064950001 is a large multi-use lot where the model consistently labels the parking area as vacant, and BBL 3065130032 is technically a parking lot but functions more as a bare asphalt lot that could arguably be called vacant. The ambiguity is not merely a model weakness -- it reflects genuine definitional overlap between "unattended outdoor parking" and "vacant land with asphalt cover."

A related question arises from the false negative side: BBL 3065230069 (Brooklyn) is an asphalt-covered vacant lot that the model misses entirely. What distinguishes this surface from the parking lots the model aggressively labels as vacant? The most likely explanation is context: the parking lots flagged as FP tend to be adjacent to industrial or commercial structures, while this particular vacant asphalt lot sits within a residential block. The model may have learned that asphalt near commercial structures signals vacancy (from training examples of G7 parking lots in Queens) but does not generalize to isolated asphalt in residential contexts.

### 6.1.2 Industrial and Port Areas

Large industrial facilities with sparse ground textures -- warehouses, shipping yards, food distribution centers -- generate systematic false positives. The Hunts Point area of the Bronx (e.g., BBLs 2027700001, 2027810500) is particularly affected. These facilities feature large expanses of bare concrete, scattered equipment, and irregular vehicle staging areas that spectrally resemble vacant lots. The model has been trained primarily on dense urban residential and mixed-use fabric in Queens, where industrial textures are less prevalent.

### 6.1.3 Roads and Highways Within Parcels

Prior to the v2 mask, roads within parcel boundaries were a major source of false positives across all boroughs. The model learned that asphalt within a parcel context (as opposed to roads masked to 255) was associated with vacancy, because many highway parcels in Queens were indeed labeled vacant by MapPLUTO.

In Queens, clusters of parcels along elevated highways (BBLs around 4025170006--4025290071) and DOT-owned highway shoulders (e.g., 4000119020, 4004270045, 4004650300) were labeled vacant in MapPLUTO but are functionally road infrastructure. The v2 mask addressed this by (a) marking these specific parcels as ignore, and (b) introducing the planimetric roadbed mask, which removed 1.39M Queens road pixels from the training set. However, some road-within-parcel cases remain -- particularly railroad corridors (e.g., BBL 2027310005 in the Bronx, BBL 3057250006 in Brooklyn near the LIRR).

### 6.1.4 Sports Facilities

The model exhibits a striking inconsistency across sports facility types:

- **Soccer fields** are consistently and correctly classified as non-vacant (BBLs 4023640023, 3030680001, 3027800001). The regular grass surface with distinct boundary markings is apparently a learned feature.
- **Baseball diamonds** are frequently misclassified as vacant (e.g., BBLs 4113430030, 3057490015). The dirt infield, unusual shape, and mixed-surface composition (grass outfield, dirt infield, sand base paths) may resemble the heterogeneous surface texture of vacant lots.
- **Tennis courts** also generate false positives (e.g., at BBL 3022460001, a school campus where the building area is correctly classified but the courts are not).

This differential behavior likely reflects the training data distribution in Queens: soccer fields are common in urban parks and have distinctive visual signatures, while baseball diamonds and tennis courts are less frequent and share surface textures with vacant land.

Run 027 on Brooklyn reinforces this pattern. BBL 3020500001 is a large lot with apartment complexes that contains a basketball court -- the model predicts the court area as vacant even at a 0.5 threshold. Similarly, BBL 3075520054 (a school courtyard with a basketball court) triggers partial false positives. Conversely, BBL 3020250001 (baseball fields) is correctly classified as non-vacant -- though this success may reflect the adjacent park infrastructure rather than recognition of the diamond itself. The model also misclassifies the gravel shoulder and parking area of BBL 3075520120 (a park) as vacant while correctly identifying the vegetated core as non-vacant, demonstrating sensitivity to surface texture transitions within a single large parcel.

### 6.1.5 Dense Suburban Tree Cover

In lower-density residential areas, particularly in the Bronx and parts of Queens, dense tree canopy over residential lots triggers false positives (e.g., area around BBL 4099380059 in Queens, BBL 2026080040 in the Bronx). The model appears to have learned that dense vegetation is sometimes associated with vacancy (overgrown abandoned lots in Queens), but in suburban contexts, the same vegetation pattern represents maintained residential properties with large yards.

### 6.1.6 Manicured Lawns and Green Spaces

Well-maintained green areas that are not parks are occasionally misclassified. Examples include aqueduct racetrack green areas (BBL 4115430002 in Queens) and waterfront green patches (BBL 4004900110 in Queens). The model struggles to distinguish maintained institutional green space from grassy vacant lots without structural cues like buildings or paths.

## 6.2 False Negative Taxonomy

False negatives -- vacant parcels that the model misses -- fall into four main categories.

### 6.2.1 Tree-Covered Vacant Lots

Vacant lots with dense tree canopy are consistently missed. In the Bronx, BBL 2026790015 and surrounding parcels are labeled vacant but covered in dense trees -- the model does not detect these. This raises a product design question: tree-covered vacant lots may host mature urban canopy that provides ecological value; flagging them for development could lead to net environmental harm. On the other hand, some tree-covered vacant lots are simply overgrown and neglected. The model's inability to detect these may be an acceptable tradeoff in a screening application.

### 6.2.2 Skinny and Small Lots

Very narrow parcels (e.g., BBL 4003400136 in Queens, BBL 2023720031 in the Bronx) and tiny parcels near water boundaries (BBLs 3088665540, 3088661897 in Brooklyn) are below the model's effective spatial resolution. At 0.6 m/pixel, a lot only 3--4 meters wide occupies just 5--7 pixels, insufficient for the model to distinguish from adjacent land. These parcels were candidates for ignore masking in v2.

### 6.2.3 Inconsistent Grassy/Dirt Detection

The model's detection of mixed grass-and-dirt lots is inconsistent. A large dirt+grass patch (BBL 3088762724 in Brooklyn) is detected, but smaller patches with similar composition (BBL 3088661764) are not. Similarly, some clearly grassy vacant lots in Brooklyn (area around BBL 3086640411) are entirely missed. The inconsistency suggests the model has not learned a robust representation of this surface type -- it may be sensitive to context (surrounding land use) or to the specific ratio of grass to bare soil.

Run 027 confirms this pattern on Brooklyn: BBLs 3065270023 and 3065080031 are clearly grassy vacant lots that the model misses entirely. In a dense urban context like NYC, an isolated grass patch within a built-up block is a strong visual indicator of vacancy -- but the model, trained primarily on Queens (which has more suburban green space), has not learned this contextual signal. The model also struggles with partial detection: BBL 3054950870 is a large parcel that is clearly vacant on one side but the model marks only a few pixels as vacant, yielding a prediction fraction too low to trigger parcel-level detection at any reasonable threshold.

### 6.2.4 Buildings Within Vacant Parcels

Some parcels labeled vacant in MapPLUTO contain structures -- either buildings that were erected after the last assessor visit, or accessory structures (sheds, garages) that do not disqualify a parcel from vacant status under NYC Administrative Code. The model correctly identifies the building pixels as non-vacant, but is penalized in the IoU calculation because the entire parcel is labeled 1 in the ground truth. For example, BBL 2027810300 in the Bronx contains a building surrounded by clearly vacant land -- the model predicts the building pixels as non-vacant (correct) but this counts as false negatives.

The building probability channel (Section 3.4) partially addresses this by providing the model with explicit building footprint information, but the fundamental tension between parcel-level labels and pixel-level predictions remains.

### 6.2.5 Community-Used Vacant Lots

A distinct category of false negatives emerges from parcels that are administratively vacant but visually occupied by informal community use. BBL 3065230010 in Brooklyn appears to be a chalk mural or informal basketball court on a vacant lot -- clearly used by the community, but classified as vacant under NYC tax code. BBL 3065280050 functions as an informal play area. The model correctly identifies these as non-vacant from the visual signal (they show signs of human activity), but is penalized because the administrative label says vacant.

This category is significant for the screening use case. These lots are precisely the type of vacant land that prior studies have documented as informally tended by communities, particularly in lower-income neighborhoods. Flagging them for development without community input risks displacing existing informal uses. The model's false negatives on community-used vacant lots may paradoxically represent a feature rather than a bug: the model detects that the land is in active use, even if the tax code disagrees.

### 6.2.6 Skinny Vacant Lots

BBL 3020440008 in Brooklyn exemplifies a failure mode specific to parcel geometry: narrow lots that occupy only a few pixels across their short dimension. The model does not predict this parcel as vacant even at low thresholds. At 0.6 m/pixel, a 4-meter-wide lot is approximately 7 pixels across -- insufficient for the model to build a spatial representation distinct from adjacent built parcels. Parcel-level evaluation (Section 5.9) shows that the median prediction fraction is just 3.3%, indicating that partial detection is the norm and skinny lots with fewer total pixels have an even smaller margin for partial activation. Higher-resolution NAIP imagery (0.3 m, discussed in Section 6.7) would double the effective pixel count and could improve detection of these parcels.

## 6.3 Irreducible Label Noise

Several categories of prediction error reflect genuine noise in the ground truth labels rather than model deficiency:

**Visually identical parcels with opposite labels.** As noted in Section 6.1.1, adjacent parcels in Brooklyn (BBLs 302608001 and 3025850001) are spectrally indistinguishable yet carry opposite labels. This represents a hard floor on achievable performance -- no vision-based model can distinguish these cases without auxiliary information.

**Label errors.** Manual inspection identified parcels labeled vacant that clearly contain buildings: BBLs 2048190013, 2048250041, 2048370055 in the Bronx; BBLs 3022410039, 3023690019 in Brooklyn; BBL 4104990092 in Queens. In v2, these were either forced to non-vacant (0) or ignore (255). However, systematic survey of all 29,336 vacant parcels was not feasible, and additional mislabeled parcels certainly remain.

**Cemetery extensions.** The St. Raymond's Cemetery extension parcels in the Bronx (BBLs around 2055700156) are coded vacant per MapPLUTO (V-class), but their actual land use as cemetery extensions is visually distinct from typical vacant land. These were documented as known label noise but not corrected, as the City's designation is the authoritative source.

**Under-construction parcels.** Parcels under active construction (e.g., BBLs 2048490039, 2023720047 in the Bronx; BBL 3027420009 in Brooklyn; BBL 4097960063 in Queens) occupy a label gray zone. They may be technically vacant (no permanent structure) but visually resemble neither typical vacant land nor developed land. The v2 mask marked the most obvious cases as ignore, but the temporal mismatch between imagery and label updates means some construction-state parcels remain in the training data. Parcel-level inspection of run 027 on Brooklyn identified additional cases: BBL 3018910080 is labeled vacant but clearly under construction -- the building probability channel correctly identifies the structure, so the model classifies it as non-vacant (a correct prediction penalized by the label). BBLs 3065190055 and 3065090007 are also under construction and labeled vacant; the model correctly rejects these, but they inflate the false negative count. BBLs 3049940013, 3049950009, and 3050050054 clearly have buildings or are in mid-construction.

**Ambiguous or misclassified parcels.** BBL 3057410005 in Brooklyn is labeled vacant but does not appear vacant on visual inspection -- its actual use is unclear. BBL 3055160014 appears to have a building on it; the model correctly labels the built portion as non-vacant but captures the adjacent vacant land. BBL 3054951149 sits adjacent to a cemetery and near a baseball diamond -- MapPLUTO labels it vacant, but the imagery shows maintained recreational land. These cases reflect the fundamental tension between administrative definitions (based on tax records and assessor visits) and visual reality (based on aerial imagery at a single point in time).

## 6.4 Impact of Mask v2 Corrections

The v2 mask corrections (Section 3.3) targeted several of the failure modes identified above:

| Failure Mode | v2 Action | Addressed? |
|-------------|-----------|:----------:|
| Highway/road parcels | Planimetric roadbed mask + specific BBL ignore | Mostly |
| Building-on-vacant parcels | Force non-vacant for clear cases | Partially |
| Under-construction parcels | Ignore for obvious cases | Partially |
| Water-edge parcels | Ignore for Brooklyn waterfront | Yes |
| Rikers Island | Ignore | Yes |
| Parking lots | No systematic fix possible | No |
| Industrial/port | No systematic fix possible | No |
| Sports facilities | No systematic fix possible | No |
| Dense tree cover | No systematic fix possible | No |

The planimetric roadbed mask was the single most impactful correction, removing 4.5 million vacant pixels across all boroughs (Table 3.1). The BBL-level corrections addressed dozens of individual cases. However, the structural failure modes -- parking lot ambiguity, industrial texture confusion, sports facility misclassification, community-used lots -- are not addressable through mask corrections because they reflect genuine spectral/textural similarity between vacant and non-vacant land covers under the administrative definition. Parcel-level inspection of run 027 on Brooklyn identified at least 6 additional parcels (BBLs 3018910080, 3020500100, 3049940013, 3049950009, 3050050054, 3065190055) that should have been masked as ignore (under construction or clearly built) -- further evidence that the v2 mask, while a substantial improvement, does not capture all label noise in a dataset of ~30,000 vacant parcels.

## 6.5 Is Administrative Vacancy Visually Recoverable?

The central research question of this thesis is whether administrative vacancy can be recovered from aerial imagery. The evidence supports a nuanced answer: **partially, and with systematic blind spots**.

**What is recoverable:**
- Large bare soil or grassy vacant lots with clear boundaries
- Lots with heterogeneous surface textures (mixed soil, sparse vegetation, debris)
- Lots in dense urban contexts where vacancy creates a visible "gap" in the built fabric

**What is not recoverable:**
- Vacancy status that depends on non-visual attributes (ownership, tax status, zoning)
- Distinctions between spectrally identical surfaces with different administrative status (parking lots, cemetery extensions)
- Vacancy obscured by canopy, shadow, or active construction
- Very small or narrow lots below the effective spatial resolution

The parcel-level evaluation (Section 5.9) provides converging evidence. At a 10% coverage threshold, 46% of vacant parcels in Brooklyn are detected -- nearly half are surfaced for human review. At 30% coverage, a third of vacant parcels are flagged. However, the median prediction fraction is just 3.3%, indicating that the model partially activates on most parcels without reaching full confidence on any. This partial-detection pattern is consistent with the model having learned a real but noisy vacancy signal: it detects *something* on most parcels, but the signal is rarely strong enough to dominate the full parcel area.

The parcel-level confusion analysis (run pending via `eval_confusion_by_class.py`) will quantify which BldgClass and LandUse categories drive false positives and false negatives, providing a land-use-specific assessment of recoverability. Preliminary qualitative inspection suggests parking lots (LandUse 10), open space (LandUse 09), and industrial parcels (LandUse 06) are the dominant FP sources, while grassy vacant lots and skinny parcels dominate FN.

A direct numerical comparison with prior work such as Mao et al. (2022), who report IoU of 0.638 on Chinese urban vacant land, is not meaningful. The tasks differ in almost every dimension: Mao et al. used visually consistent labels from human auditors (not administrative records), operated at 1.6m resolution in Chinese cities with more regular urban grids and larger lot sizes, and computed IoU after polygon dilation. Most fundamentally, administrative vacancy (NYC) and visual vacancy (Mao et al.) represent different target concepts. The appropriate comparison is not cross-study IoU but rather whether the model is useful for its intended application -- screening.

## 6.6 Implications for Smaller Jurisdictions

The practical value of this model lies not in autonomous vacancy detection but in **shortlist generation**. A city planner using the model would:

1. Apply the model to NAIP imagery at the F2-optimal threshold (maximizing recall)
2. Receive a map of areas with high predicted vacancy probability
3. Manually review flagged areas against local records, site visits, or street-view imagery
4. Discard false positives and investigate true positives

In this workflow, false positives are acceptable (the human reviewer filters them) while false negatives represent missed opportunities. The model's value proposition is reducing the search space from an entire city to a manageable shortlist. The parcel-level evaluation (Section 5.9) quantifies this directly: at a 10% coverage threshold, the best model (run 027) identifies 46% of vacant parcels in Brooklyn. At a 30% threshold, it still identifies 33%. For a city planner starting without digitized cadastral records, surfacing a third to half of all vacant parcels from aerial imagery alone represents a substantial reduction in the search space -- even before considering that many false positives (parking lots, under-construction sites) are themselves of planning interest.

## 6.7 Sensor and Resolution Limitations

Several failure modes are directly attributable to sensor limitations of NAIP imagery:

**No SWIR bands.** Short-wave infrared (SWIR) bands are highly effective at distinguishing bare soil from impervious surfaces (asphalt, concrete) because these materials have distinct SWIR reflectance signatures that overlap in the visible and NIR ranges. Many of the parking lot and road false positives could potentially be resolved with SWIR data. However, NAIP does not include SWIR bands, and no freely available national-scale SWIR imagery exists at 0.6m resolution.

**No height information.** A digital surface model (DSM) or LiDAR point cloud would enable the model to detect buildings directly from their height signature, eliminating the building-within-vacant-parcel failure mode. LiDAR coverage of NYC exists but was not incorporated in this study due to temporal misalignment and processing complexity.

**0.6m resolution.** While 0.6m is high resolution by satellite standards, it limits detection of small lots (lots narrower than ~4m occupy fewer than 7 pixels). Higher-resolution NAIP imagery (0.3m, available for some states since 2019) would double the effective spatial resolution and could improve detection of narrow lots.

**Single-date imagery.** The 2022 NAIP coverage of NYC spans three acquisition dates (July, September, October), representing a snapshot during peak and late growing season. Temporal variability in vegetation state is not captured. A lot that appears green in July may be bare in November. Multi-temporal analysis could improve robustness but is limited by NAIP's 3-year revisit cycle.

## 6.8 Future Work

### Technical Extensions

- **Higher-resolution NAIP (0.3m):** Some states now receive 0.3m NAIP imagery. Training at 0.3m resolution would improve detection of small and narrow lots, which are among the most common false negatives.
- **SWIR integration:** Fusion of NAIP with Sentinel-2 SWIR bands (20m, resampled) or commercial SWIR imagery could reduce confusion between bare soil, asphalt, and rooftop surfaces.
- **Additional backbone architectures:** EfficientNet, ConvNeXt, or vision transformer (ViT) backbones may provide improved feature representations. Xception (used by DeepLabV3+ authors) was not tested.
- **Label refinement pipeline:** Active learning or human-in-the-loop mask refinement could iteratively improve label quality, breaking the ceiling imposed by administrative label noise.
- **Multi-class formulation:** Rather than binary vacant/non-vacant, a multi-class head predicting vacancy subtypes (bare soil, vegetated, construction, parking) could provide more actionable output for planners.
- **Parcel-level confusion by land use:** The parcel-level evaluation (Section 5.9) and confusion-by-class analysis establish which land-use categories the model reliably detects and which it confuses. Extending this to a multi-class formulation -- predicting vacancy subtypes (bare soil, vegetated, construction, parking) rather than binary vacant/non-vacant -- could provide more actionable output for planners and reduce the false positive burden on specific categories like parking lots.

### Cross-City Generalization

[PLACEHOLDER: Cross-city inference on Philadelphia or Chicago, if time permits. These cities have publicly available parcel-level vacancy data that could serve as evaluation labels. The key question is whether spatial and spectral patterns learned from NYC's urban fabric transfer to cities with different building stock, lot sizes, and vegetation patterns.]

### Application Extensions

- **Economic impact modeling:** Integrating model outputs with zoning and market data to estimate the housing unit potential, stormwater management capacity, or community garden suitability of identified vacant parcels.
- **Temporal vacancy tracking:** Applying the model to historical NAIP imagery (2016, 2019, 2022) to track vacancy persistence and turnover, following the spatio-temporal analysis framework of Zhang et al.
- **Policy implications:** Identification of parking lots consuming developable land in high-demand neighborhoods; assessment of bioswale potential along highways (Belt Parkway, BQE); analysis of vacancy concentration patterns in the context of affordable housing policy.
- **Ecological assessment:** Tree-covered vacant lots identified as false negatives may represent urban canopy that should be preserved rather than developed. Coupling vacancy predictions with canopy height data could distinguish "green vacancy" (preservation candidate) from "brown vacancy" (development candidate).
