"""Step 1: download lidar, Landsat 8 and Sentinel-2 layers for downtown Austin.

Run from this folder:  python download.py
Output: raw/layers.npz and raw/meta.json (about 60 MB, not stored in git).
Then run prepare.py to make the small model tables in data/.
"""

import json
from pathlib import Path

import numpy as np

from greenroof.data import grids, load_landsat_st, load_lidar, load_sentinel2

# Downtown Austin, inside City Council District 9 (the paper's study area).
CENTER = (30.2700, -97.7420)
SIZE_M = 3600  # 3.6 km square: 120 x 120 cells of 30 m
LANDSAT_DATE = "2023-08-11"  # the scene used in the paper
SENTINEL_DATE = "2023-08-12"

out = Path("raw")
out.mkdir(exist_ok=True)
g2, g10, g30 = grids(*CENTER, SIZE_M)
lid = load_lidar(g2)
dlst, lmeta = load_landsat_st(g30, LANDSAT_DATE)
s2, s2_id = load_sentinel2(g10, SENTINEL_DATE)

np.savez_compressed(out / "layers.npz", dsm=lid["dsm"].astype("float32"),
                    hag=lid["hag"].astype("float32"), cls=lid["cls"].astype("uint8"),
                    dlst_f=dlst, **s2)
meta = {"center": CENTER, "size_m": SIZE_M, "crs": "EPSG:32614",
        "bounds": list(g30.extent.boundingbox), "landsat": lmeta, "sentinel2": s2_id,
        "lidar_items": {k: lid[k] for k in lid if k.endswith("_items")}}
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
