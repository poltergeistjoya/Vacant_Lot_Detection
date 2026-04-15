# 2.6 Why Land Cover ≠ Administrative Vacancy

The fundamental challenge of this task is that administrative vacancy is not a visual class. Land cover classification -- the dominant paradigm in remote sensing -- assigns labels based on the physical appearance of the earth's surface: vegetation, impervious surface, bare soil, water. Administrative vacancy, by contrast, is a legal and fiscal designation assigned by a municipal assessor based on the presence or absence of a permanent enclosed structure on a tax lot. Two parcels that are spectrally identical may carry opposite labels, and a single vacant lot may look nothing like another.

[INSERT FIGURE 2.1: Manual inspection of administratively vacant lots in Williamsburg, Brooklyn. All highlighted parcels are labeled vacant (BldgClass V* or G7) in MapPLUTO 22v3.]

Figure 2.1 illustrates this heterogeneity for a cluster of administratively vacant lots in Williamsburg, Brooklyn. The lots carry the same label but bear almost no visual resemblance to one another:

- **Lot 1** is fully paved -- a bare asphalt surface with no vegetation. Spectrally, it resembles a road or parking lot more than any conventionally "vacant" land cover.
- **Lot 2** is a basketball court, visually identical to the non-vacant basketball court immediately adjacent to it. No spectral or textural feature distinguishes these two parcels.
- **Lot 3** is a collection of sub-lots ranging from sparse vegetation to unpaved and paved parking surfaces -- heterogeneous within a single parcel.
- **Lots 4 and 6** both appear to have active construction. Only lot 4 is labeled vacant; lot 6 is not. Under NYC Administrative Code, a parcel remains vacant until a permanent enclosed structure is present, so lot 4 retains its vacant label while lot 6 -- which has progressed further in construction -- has already been reclassified. The bare soil visible in both lots is indistinguishable from aerial imagery.
- **Lot 5** appears to be a pool or water feature, again carrying a vacant designation despite its clearly developed appearance.

This single block illustrates every core challenge of the task: surface-type diversity within the vacant class, visual indistinguishability between adjacent vacant and non-vacant parcels, construction-state label lag, and the presence of clearly developed features within technically vacant tax lots.

The implication is direct: a model trained to predict administrative vacancy cannot rely on learning a consistent visual class the way a land cover model learns "forest" or "water." Instead, it must learn a statistical association between visual appearance and a label that is partially determined by non-visual factors -- ownership history, permit status, tax classification. The achievable performance ceiling is therefore lower than for conventional land cover segmentation tasks, and the failure modes are qualitatively different (Section 6).
