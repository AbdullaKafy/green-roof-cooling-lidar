"""Step 2: turn the raw layers into the model inputs used in the paper.

Run from this folder after download.py:  python prepare.py
Output (small, stored in git):
  data/features_30m.csv   one row per 30 m cell: 8 urban form inputs and DLST
  data/roofs_10m.npz      10 m roof map, NDVI and NDBI for green roof scenarios
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from greenroof import indices, lidar, morphology

raw = np.load("raw/layers.npz")
meta = json.loads(Path("raw/meta.json").read_text())
lat, lon = meta["center"]
DAY = 223  # 11 August 2023

# 2 m lidar layers
dsm = raw["dsm"].astype("float64")
dsm = np.where(np.isfinite(dsm), dsm, np.nanpercentile(dsm, 1))  # water has no returns
hag = np.nan_to_num(raw["hag"].astype("float64"), nan=0.0).clip(0, None)
bmask = lidar.building_mask(hag, classification=raw["cls"], min_pixels=10)  # 40 m² or more

# 10 m Sentinel-2 layers
green, red, nir, swir1 = (raw[b].astype("float64") for b in ("green", "red", "nir", "swir1"))
ndvi10 = indices.ndvi(nir, red)
ndbi10 = indices.ndbi(swir1, nir)
water10 = (indices.mndwi(green, swir1) > 0) & (ndvi10 < 0.1)

# Inputs on the 30 m grid
F2, F10 = 15, 3  # 2 m pixels and 10 m pixels per 30 m cell
bh_sum = lidar.aggregate(np.where(bmask, hag, 0.0), F2, "sum")
b_n = lidar.aggregate(bmask.astype(float), F2, "sum")
feat = {
    "BH_m": np.divide(bh_sum, b_n, out=np.zeros_like(bh_sum), where=b_n > 0),
    "BRI_m2": lidar.roof_area_index(bmask, F2, 4.0),
    "BVD": lidar.volume_density(hag, bmask, F2, 4.0),
    "SVF": lidar.aggregate(morphology.sky_view_factor(dsm, 2.0, radius_m=90, directions=16), F2),
    "SR_Wh_m2": lidar.aggregate(
        morphology.daily_direct_insolation(dsm, 2.0, lat, lon, DAY), F2),
    "NDVI": lidar.aggregate(ndvi10, F10),
    "NDBI": lidar.aggregate(ndbi10, F10),
    "WBD_m": lidar.aggregate(morphology.distance_to_water(water10, 10.0), F10),
}
dlst = raw["dlst_f"].astype("float64")
rows, cols = np.indices(dlst.shape)
df = pd.DataFrame({"row": rows.ravel(), "col": cols.ravel(),
                   **{k: v.ravel() for k, v in feat.items()}, "DLST_F": dlst.ravel()})
water30 = lidar.aggregate(water10.astype(float), F10) > 0.5
df["water"] = water30.ravel()
Path("data").mkdir(exist_ok=True)
df.round(4).to_csv("data/features_30m.csv", index=False)

# 10 m roof layer for scenarios: a 10 m pixel is roof if most of it is building
roof10 = lidar.aggregate(bmask.astype(float), 5) >= 0.5
height10 = lidar.aggregate(np.where(bmask, hag, np.nan), 5)
np.savez_compressed("data/roofs_10m.npz", roof_ids=lidar.label_buildings(roof10).astype("int32"),
                    height=np.nan_to_num(height10).astype("float32"),
                    ndvi=ndvi10.astype("float32"), ndbi=ndbi10.astype("float32"))
(Path("data") / "meta.json").write_text(json.dumps(meta, indent=2))
print(df.describe().T.round(2))
