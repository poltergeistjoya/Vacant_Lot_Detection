# 3. Data and Materials

## 3.1 NAIP Imagery

The National Agriculture Imagery Program (NAIP), administered by the USDA, has acquired high-resolution aerial orthophotos across the continental United States since 2003. Originally capturing imagery at 1 m resolution on a five-year revisit cycle per state, the program transitioned to a three-year cycle in 2009. Near-infrared was piloted in Arizona in 2007 and became a default four-band product (Red, Green, Blue, NIR) in 2010. In 2018, the native spatial resolution was upgraded to 0.6 m per pixel, with 0.3 m available for some coastal states. Imagery is orthorectified to a horizontal accuracy of 4 meters at 95% confidence, referenced to NAD83, and distributed as quarter-quadrangle tiles (3.75' longitude × 3.75' latitude with a 300-meter buffer). Cloud cover is limited to at most 10% per tile.

NAIP is freely available through providers such as Google Earth Engine, AWS STAC APIs, and the USGS data portal. For this study, 2022 NYC tiles were downloaded from the Microsoft Planetary Computer STAC API at no cost. NYC was imaged across three acquisition dates in 2022: July 19 (17 center tiles), September 15 (24 tiles covering the outer boroughs), and October 7 (6 tiles covering southwest Staten Island). Of the 85 quarter-quadrangle tiles downloaded, 38 covering New Jersey border areas were excluded. The remaining 47 tiles were mosaicked into a single virtual raster (VRT) using GDAL, with later acquisition dates placed on top to resolve any overlap. All bands were normalized to [0, 1] prior to model input. The 2022 acquisition year was chosen to align temporally with the MapPLUTO 22v3 parcel dataset.

[INSERT FIGURE 3.1: NYC NAIP tile coverage map, color-coded by acquisition date]
[INSERT FIGURE 3.2: NAIP composite for NYC illustrating 0.6 m resolution]

Despite its utility as a freely available, high-resolution national dataset, NAIP has notable limitations relevant to this study. Because imagery acquisition is partially state-funded, spatial resolution and spectral options (e.g., 0.3 m) vary by state, limiting portability of methods to other regions. The program does not guarantee annual coverage. Perhaps most practically, there is a substantial lag between imagery acquisition and availability on easy-to-use APIs: as of writing, 2024 imagery is not available on Planetary Computer or GEE and requires navigating USDA/USGS government portals directly. This lag constrains how current a deployed model's inputs can be relative to administrative records.

## 3.2 NYC MapPLUTO

MapPLUTO (Map Primary Land Use Tax Lot Output) is a publicly available geospatial dataset released by the NYC Department of City Planning that merges tax lot boundaries with land use, zoning, building, and ownership information for every parcel in the five boroughs. It is compiled from multiple source agencies -- primarily the Department of Finance (DOF) Property Tax System, the Department of Buildings (DOB), and the Department of City Planning itself -- and is released quarterly. The Building Class codes that encode vacancy status are updated from the DOF tax roll on a biannual schedule (Tentative Tax Roll in January, Final Tax Roll in June); other fields may be updated in intermediate quarterly releases.

MapPLUTO offers two approaches to identifying vacant land. The first is the `LandUse` field, where code "11" designates vacant land at a coarse level. The second, and the one used in this study, is the `BldgClass` (Building Class) field, derived directly from the DOF Property Tax System and providing finer granularity. Vacancy is encoded by Building Class codes V0 through V9 (vacant land subtypes) and G7 (unlicensed parking lot). The inclusion of G7 as a vacancy-equivalent designation is non-obvious and was confirmed through a practitioner interview: G7 parcels carry no permanent enclosed structure -- the operative criterion for vacancy -- and are therefore treated equivalently to V-class parcels in this study (L. McGeehan, NYC Department of Finance, personal communication, April 2026). The V-class codes are further subdivided by zoning: V0 (residentially zoned), V1 (commercially zoned), and V4--V8 (publicly owned land), though all are treated as a single binary vacant class here.

Of the 856,998 parcels in MapPLUTO 22v3, 30,792 (3.6%) carry a vacant designation under these codes. Parcel geometries are natively in EPSG:2263 (New York State Plane, Long Island) and were reprojected to EPSG:26918 (UTM Zone 18N) to align with the NAIP raster grid.

### Data Quality and Temporal Limitations

Despite being among the most comprehensive municipal parcel datasets in the United States, MapPLUTO carries several limitations relevant to its use as model ground truth.

**Classification lag.** Building Class codes are updated by City Tax Assessors through a largely manual process of field visits, permit checks, and photographic review. A parcel undergoing new construction remains classified as vacant until a foundation is in place -- at that point an assessor can reclassify it as an improved building class, but the exact timing of that update within a given year is not fixed. As a result, parcels with active construction visible in 2022 NAIP imagery may still carry a vacant label in 22v3, and conversely, recently demolished parcels may not yet appear as vacant if the assessor update has not propagated through the data release cycle (McGeehan, personal communication, April 2026).

**Multi-agency merge lag.** MapPLUTO aggregates data from DOF, DOB, and DCP on different schedules. DOF supplies updated assessment data to City Planning at each biannual tax roll release, but intermediate field updates may not be reflected until the subsequent public release. This chain introduces a potential lag of several months between a real-world change and its appearance in the public dataset.

**Overall temporal resolution is strong relative to other cities.** In spite of these lags, NYC's assessor infrastructure provides substantially better temporal coverage than most municipalities. Every lot receives drive-by photographic review twice per year, and the city targets an in-person visit to every parcel on a three-year cycle. Approximately 1,000--2,000 new vacant lots are created per year against a standing pool of roughly 30,000, meaning the dataset turns over meaningfully between annual snapshots (McGeehan, personal communication). The 2022 imagery and 22v3 label vintage were aligned to the same calendar year to minimize temporal mismatch.

## 3.3 NYC DoITT Planimetric Database

The NYC Department of Information Technology and Telecommunications (DoITT) Planimetric Database is a citywide vector dataset derived from aerial photogrammetry, capturing physical features of the urban environment as mapped polygons. For this study, the `ROADBED` layer was used -- 104,959 MultiPolygon features representing the actual paved surface of every road in the five boroughs, as opposed to road centerlines.

Planimetric roadbed polygons were used to mask road surfaces in the vacancy mask (Section 3.4). The motivation is that road surfaces within parcel boundaries are a significant source of label noise: many parcels classified as vacant by MapPLUTO are highway shoulders, elevated highway footprints, or DOT-owned road infrastructure where the visible surface is entirely asphalt. Burning these surfaces to ignore (255) prevents the model from learning an association between road asphalt and vacancy.

The planimetric database was used in favor of TIGER road centerline buffers, which approximate road surfaces as thin buffered lines and substantially underestimate road width, particularly for arterials and elevated highways. The DoITT polygons represent true pavement extents and are derived from the same 2022 aerial imagery program as the NAIP data used in this study.

## 3.4 Vacancy Mask Construction

Ground truth labels are encoded as a binary vacancy mask produced by rasterizing MapPLUTO parcel boundaries onto the NAIP pixel grid at 0.6 m resolution. The mask uses three values: 0 (non-vacant), 1 (vacant), and 255 (ignore). The 255 class follows the standard semantic segmentation convention [@everingham2010pascal]: pixels assigned 255 are excluded from both loss computation during training and metric computation during evaluation. Beyond handling label ambiguity, the ignore class also serves a class-imbalance purpose -- masking out roads, water, and other surfaces where vacancy is structurally impossible prevents hundreds of millions of additional non-vacant pixels from diluting an already sparse 3.4% vacancy signal.

[INSERT FIGURE 3.3: Vacancy mask over a representative Queens area, showing 0/1/255 encoding]

The mask was produced through two successive versions. An initial mask (v1) was used for early training runs; systematic inspection of model predictions identified label errors that were corrected in a refined mask (v2), which was used for all final reported results. The pipeline steps below note where v1 and v2 differ.

**Step 1 — Base burn.** All MapPLUTO parcel polygons are rasterized to 0 (non-vacant). Pixels outside any parcel boundary -- road rights-of-way, water bodies, and other unregistered surfaces -- default to 255 (ignore). This is the key design choice that keeps class imbalance manageable: roads and water are never presented to the model as non-vacant training signal.

**Step 2 — Vacant burn.** Parcels with vacant Building Class codes (V0--V9, G7) are overwritten with 1 (vacant).

**Step 3 — Manual corrections.** A small set of parcels were reassigned based on visual inspection of NAIP imagery and, for v2, cross-referencing a newer MapPLUTO vintage (23v2):

| | v1 | v2 |
|--|:--:|:--:|
| Set to ignore (255) | 11 (shadow-occluded + South Brother Island) | ~49 total [verify against `config/data.yaml`] |
| Forced non-vacant (0) | — | 11 (buildings present; highway parcels) |
| Forced vacant (1) | — | 7 (visual confirmation or vintage cross-check) |

The v2 additions include parcels under active construction, waterfront parcels extending into water, Queens parcels obstructed by elevated highways, and DOT-owned highway shoulders mislabeled as vacant. Rikers Island was set to ignore given its anomalous land use as a detention facility.

**Step 4 — Boundary erosion.** A 2-pixel morphological erosion is applied at transitions between vacant and non-vacant parcels. The transition zone is identified by dilating both classes and computing their overlap; pixels in this overlap zone on the vacant side are set to 255. This reduces noise from imprecise parcel boundary rasterization and mixed pixels at parcel edges. Boundaries between two parcels of the same class are left untouched.

**Step 5 — Road masking.**
- *v1:* No explicit road masking step. Roads outside any parcel boundary already default to 255 from Step 1 (base burn), so major road rights-of-way were naturally excluded. The gap was roads *within* parcels -- elevated highway footprints and DOT-owned shoulders that MapPLUTO labels as vacant but are functionally road infrastructure. These were not masked in v1, causing systematic false positives in the training set.
- *v2:* NYC DoITT Planimetric Database roadbed polygons (104,959 MultiPolygon features) explicitly burned to 255, addressing within-parcel road surfaces at true pavement width.

After the v1 pipeline, the mask contained 29,336 vacant parcels (3.4% of 855,322 eligible parcels). The v2 corrections reduced vacant pixel counts in every borough, with Queens seeing the largest change due to the planimetric roadbed mask eliminating 1.39M previously mislabeled road pixels:

**Table 3.1: Pixel changes from vacancy mask v1 to v2**

| Borough | Vacant pixel delta | Ignore pixel delta |
|---------|-------------------:|-------------------:|
| Manhattan | −296,000 | +4,700,000 |
| Bronx | −296,000 | +11,100,000 |
| Brooklyn | −385,000 | +9,500,000 |
| Queens | −1,390,000 | +41,400,000 |
| Staten Island | −2,150,000 | +9,800,000 |

### 3.4 Label Noise and Iterative Refinement

The vacancy mask inherits several sources of noise from its administrative origin.

**Temporal lag.** As described in Section 3.2, Building Class codes are updated manually on an irregular schedule. Parcels under active construction may retain a vacant label until a foundation is in place; recently demolished lots may not yet be reclassified as vacant. Both directions of lag are visible in the 2022 imagery -- construction sites with bare soil labeled vacant, and recently cleared lots still labeled non-vacant.

**Administrative vs. visual mismatch.** The most fundamental noise source is that administrative vacancy is not a visual property. Two parcels can be spectrally identical yet carry opposite labels (see Figure 2.1). This represents an irreducible noise floor that any vision-based model must contend with.

**Boundary noise.** At 0.6 m resolution, a 2-pixel parcel boundary uncertainty corresponds to ~1.2 m of positional error. The boundary erosion in Step 4 converts the most ambiguous edge pixels to ignore, but mixed pixels at sharp lot boundaries remain.

An iterative approach was taken to address the tractable sources of noise. The v1 mask was used to train initial runs; inspection of predictions on the Bronx (validation) and Brooklyn (test) sets revealed systematic patterns -- highway parcels teaching the model that road surfaces are vacant, parcels with visible buildings labeled as vacant -- that informed the targeted v2 corrections. All final reported models are trained on the v2 mask; runs using v1 are included in Section 5.3 for ablation purposes only.

## 3.5 Borough-Based Spatial Split

To evaluate geographic generalization and prevent spatial autocorrelation from inflating performance metrics, the dataset was partitioned by NYC borough rather than by random sampling. Each borough serves as a spatially contiguous holdout region:

- **Train: Queens (BoroCode 4).** Queens contains the largest number of vacant pixels and exhibits a diverse distribution of vacancy types, including residential, industrial, and mixed-use areas.

- **Validation: Bronx (BoroCode 2).** The Bronx provides a distinct urban fabric from Queens, with denser residential areas and prominent industrial zones (e.g., Hunts Point).

- **Test: Brooklyn (BoroCode 3).** Brooklyn offers a broad mix of vacancy types -- waterfront, residential, industrial, and commercial -- providing a rigorous test of generalization.

- **Excluded: Manhattan (BoroCode 1).** Manhattan's extreme building density and prevalence of tall structures create pervasive shadow occlusion in aerial imagery, making vacancy labels unreliable.

- **Excluded: Staten Island (BoroCode 5).** Staten Island is overwhelmingly characterized by suburban tree cover, which is spectrally dissimilar to the urban vacancy patterns present in the other boroughs.

This spatial split ensures that the model is evaluated on boroughs it has never seen during training, providing a conservative estimate of real-world performance.

## 3.6 Patch Extraction

The mosaicked NAIP raster and corresponding vacancy mask were divided into non-overlapping square patches for training. Two patch sizes were evaluated:

- **512 x 512 pixels** (~307 m spatial extent at 0.6 m/px)
- **1024 x 1024 pixels** (~614 m spatial extent at 0.6 m/px)

The choice of patch size represents a tradeoff between spatial context and computational cost. Mao et al. (2022), who performed vacant land detection at 1.6 m resolution with 256 x 256 patches (~410 m extent), demonstrated that sufficient spatial context is critical for distinguishing vacancy from spectrally similar land covers. At 0.6 m resolution, 512 x 512 patches provide comparable spatial coverage (~75% of Mao et al.), while 1024 x 1024 patches provide ~150%.

A patch was retained if at least 50% of its pixels carried a valid label (0 or 1, not 255). However, any patch containing at least one vacant pixel was also retained regardless of its valid-pixel fraction, ensuring that small or boundary-adjacent vacant areas were not discarded. During training, patches containing vacant pixels were oversampled by a factor of 4 to partially compensate for class imbalance.

At the 512 x 512 patch size, the split statistics are:

**Table 3.2: Patch and pixel statistics per split (512 x 512)**

| Split | Borough | Patches | Vacant px | Non-vacant px | Ignore px | Vacant % |
|-------|---------|--------:|----------:|--------------:|----------:|---------:|
| Train | Queens | 12,576 | 19,729,417 | 566,361,850 | 238,089,469 | 3.37% |
| Val | Bronx | 4,943 | 8,390,527 | 216,060,926 | 99,492,995 | 3.74% |
| Test | Brooklyn | 8,137 | 10,877,486 | 352,131,274 | 170,257,672 | 3.00% |

The median number of vacant pixels per patch is 0 across all splits, reflecting the extreme sparsity of vacancy -- the majority of patches contain no vacant pixels at all. The mean vacant pixels per patch ranges from 1,337 (Brooklyn) to 1,698 (Bronx).

During inference, overlapping strides (256 pixels for 512-patch models, 512 pixels for 1024-patch models) were used to produce smoother predictions and reduce boundary artifacts.
