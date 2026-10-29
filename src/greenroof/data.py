"""Download lidar, Landsat 8 and Sentinel-2 layers for a small study box.

All data come from the Microsoft Planetary Computer STAC catalog (no account
needed):

* USGS 3DEP lidar products at 2 m: surface model (DSM), height above ground
  (HAG, the same as an nDSM) and point classification (6 = building, 9 = water).
* Landsat 8 Collection 2 Level-2 surface temperature (band 10), 30 m.
* Sentinel-2 Level-2A surface reflectance, 10 m bands resampled to the fine grid.

Three grids share the same origin: 2 m for lidar, 10 m for Sentinel-2 and green
roof edits (as in the paper), and 30 m for Landsat and the model. Every 30 m cell
holds exactly 15 x 15 lidar pixels and 3 x 3 Sentinel-2 pixels.
"""

from __future__ import annotations

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def grids(lat: float, lon: float, size_m: float, crs: str = "EPSG:32614",
          resolutions=(2.0, 10.0, 30.0)):
    """GeoBoxes (one per resolution, finest first) for a square box centered on (lat, lon)."""
    from odc.geo.geobox import GeoBox
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    coarse_m = max(resolutions)
    n = int(size_m // coarse_m)
    x0 = np.floor((x - n * coarse_m / 2) / coarse_m) * coarse_m
    y0 = np.floor((y - n * coarse_m / 2) / coarse_m) * coarse_m
    bbox = (x0, y0, x0 + n * coarse_m, y0 + n * coarse_m)
    return [GeoBox.from_bbox(bbox, crs=crs, resolution=r) for r in resolutions]


def _search(collection, geobox, **kw):
    bbox = geobox.geographic_extent.boundingbox
    return list(_catalog().search(collections=[collection],
                                  bbox=(bbox.left, bbox.bottom, bbox.right, bbox.top),
                                  **kw).items())


def load_lidar(fine):
    """DSM, height above ground and classification on the fine grid."""
    from odc.stac import load

    out = {}
    for name, coll, resampling in (("dsm", "3dep-lidar-dsm", "bilinear"),
                                   ("hag", "3dep-lidar-hag", "bilinear"),
                                   ("cls", "3dep-lidar-classification", "nearest")):
        items = _search(coll, fine)
        da = load(items, bands=["data"], geobox=fine, resampling=resampling,
                  groupby="solar_day", chunks={})["data"]
        out[name] = da.max("time").compute().values
        out[name + "_items"] = [i.id for i in items]
    return out


def load_landsat_st(coarse, date: str):
    """Landsat 8 band 10 surface temperature (°F) on the coarse grid for one date."""
    from odc.stac import load

    items = [i for i in _search("landsat-c2-l2", coarse, datetime=date)
             if i.properties["platform"] == "landsat-8"]
    ds = load(items, bands=["lwir11", "qa_pixel"], geobox=coarse, resampling="bilinear",
              chunks={}).isel(time=0).compute()
    k = ds["lwir11"].where(ds["lwir11"] > 0) * 0.00341802 + 149.0
    f = (k - 273.15) * 9 / 5 + 32
    meta = {"scene": items[0].id, "sun_elevation": items[0].properties["view:sun_elevation"],
            "sun_azimuth": items[0].properties["view:sun_azimuth"]}
    return f.values.astype("float32"), meta


def load_sentinel2(fine, date: str):
    """Sentinel-2 L2A reflectance (0 to 1) for green, red, NIR and SWIR1 (use the 10 m grid)."""
    from odc.stac import load

    items = sorted(_search("sentinel-2-l2a", fine, datetime=date),
                   key=lambda i: i.properties["eo:cloud_cover"])[:1]
    ds = load(items, bands=["B03", "B04", "B08", "B11"], geobox=fine,
              resampling="bilinear", chunks={}).isel(time=0).compute()
    # Processing baseline 04.00 and later adds an offset of 1000 to reflectance.
    base = float(items[0].properties.get("s2:processing_baseline", "0"))
    off = 1000 if base >= 4.0 else 0
    names = {"B03": "green", "B04": "red", "B08": "nir", "B11": "swir1"}
    out = {v: ((ds[k].values.astype("float32") - off) / 10000).clip(0, 1)
           for k, v in names.items()}
    return out, items[0].id
