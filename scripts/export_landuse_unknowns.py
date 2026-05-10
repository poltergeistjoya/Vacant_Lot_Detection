"""
Export MapPLUTO 22v3 parcels with unknown (NULL) LandUse to GeoPackage,
joining in LandUse and BldgClass from 23v2 to track what they became.

Output: outputs/figures/landuse_unknowns_22v3_vs_23v2.gpkg

Columns:
  BBL          - Parcel ID (integer)
  BoroCode     - Borough (1=Manhattan … 5=SI)
  Address      - Street address from 22v3
  BldgClass    - Building class in 22v3
  LandUse      - Always NULL (the filter criterion)
  BldgClass23  - Building class in 23v2 (None if parcel dropped)
  LandUse23    - Land use in 23v2 (None if parcel dropped / still unknown)
  changed      - True if LandUse23 is non-null and differs from LandUse (22v3)
  geometry     - Parcel polygon (EPSG:2263, native MapPLUTO CRS)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

# Locate shared root the same way the rest of the project does
try:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
    from vacant_lot.config import _get_shared_root
    SHARED_ROOT = _get_shared_root()
except Exception:
    p = SCRIPT_DIR.parent
    while p != p.parent:
        if (p / "data" / "parcels").exists():
            SHARED_ROOT = p
            break
        p = p.parent
    else:
        SHARED_ROOT = SCRIPT_DIR.parent

GDB_22 = SHARED_ROOT / "data/parcels/nyc/mappluto_22v3/MapPLUTO22v3.gdb"
GDB_23 = SHARED_ROOT / "data/parcels/nyc/mappluto_23v2/MapPLUTO23v2.gdb"
LAYER_22 = "MapPLUTO_22v3_clipped"
LAYER_23 = "MapPLUTO_23v2_clipped"
OUT_PATH = SHARED_ROOT / "outputs/figures/landuse_unknowns_22v3_vs_23v2.gpkg"

LANDUSE_LABELS = {
    "01": "1-2 Family Residential",
    "02": "Multi-Family Walk-Up",
    "03": "Multi-Family Elevator",
    "04": "Mixed Res./Commercial",
    "05": "Commercial & Office",
    "06": "Industrial & Manufacturing",
    "07": "Transportation & Utility",
    "08": "Public Facilities",
    "09": "Open Space & Recreation",
    "10": "Parking Facilities",
    "11": "Vacant Land",
    None: "Unknown / Not Mapped",
}


def bbl_to_int(series: pd.Series) -> pd.Series:
    """Convert float BBL (e.g. 1.000010e+09) to Int64 for reliable joins."""
    return series.round().astype("Int64")


def main() -> None:
    print(f"Shared root : {SHARED_ROOT}")
    print(f"22v3 GDB    : {GDB_22}")
    print(f"23v2 GDB    : {GDB_23}")

    # ------------------------------------------------------------------
    # Load 22v3 unknowns (LandUse IS NULL)
    # ------------------------------------------------------------------
    print("\nLoading 22v3 LandUse=NULL parcels …")
    gdf22 = gpd.read_file(
        GDB_22,
        layer=LAYER_22,
        where="LandUse IS NULL",
        columns=["BBL", "BoroCode", "Address", "BldgClass", "LandUse", "geometry"],
    )
    gdf22["BBL"] = bbl_to_int(gdf22["BBL"])
    print(f"  {len(gdf22):,} parcels with unknown LandUse in 22v3")
    print(f"  By borough:\n{gdf22['BoroCode'].value_counts().sort_index().to_string()}")

    # ------------------------------------------------------------------
    # Load 23v2 — only BBL + LandUse + BldgClass (no geometry needed)
    # ------------------------------------------------------------------
    print("\nLoading 23v2 LandUse + BldgClass for join …")
    gdf23 = gpd.read_file(
        GDB_23,
        layer=LAYER_23,
        columns=["BBL", "LandUse", "BldgClass"],
    )
    gdf23["BBL"] = bbl_to_int(gdf23["BBL"])

    # Drop geometry from 23v2 — we only need the attribute table
    lookup23 = pd.DataFrame(gdf23.drop(columns="geometry")).rename(
        columns={"LandUse": "LandUse23", "BldgClass": "BldgClass23"}
    )
    # Deduplicate (sub-lots share a BBL in condo buildings)
    lookup23 = lookup23.drop_duplicates(subset="BBL", keep="first")
    print(f"  {len(lookup23):,} unique BBLs in 23v2")

    # ------------------------------------------------------------------
    # Join
    # ------------------------------------------------------------------
    merged = gdf22.merge(lookup23, on="BBL", how="left")

    # Flag parcels that gained a classification
    merged["changed"] = merged["LandUse23"].notna()

    print(f"\n22v3 unknowns matched to 23v2: {merged['changed'].sum():,} / {len(merged):,}")
    print("\nLandUse23 distribution (what they became):")
    counts = merged["LandUse23"].value_counts(dropna=False)
    for code, n in counts.items():
        label = LANDUSE_LABELS.get(code, code)
        print(f"  {str(code):>4}  {label:<35}  {n:>5}")

    # ------------------------------------------------------------------
    # Write GeoPackage
    # ------------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Keep native CRS (EPSG:2263) — reproject if you need WGS84 in QGIS
    out = merged[
        ["BBL", "BoroCode", "Address", "BldgClass", "LandUse",
         "BldgClass23", "LandUse23", "changed", "geometry"]
    ].copy()

    print(f"\nWriting {len(out):,} parcels → {OUT_PATH}")
    out.to_file(OUT_PATH, driver="GPKG")
    print("Done.")


if __name__ == "__main__":
    main()
