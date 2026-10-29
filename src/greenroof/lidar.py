"""Building layers from airborne lidar rasters.

Inputs are gridded lidar products on one grid: a digital surface model (DSM,
top of roofs and trees) and a digital terrain model (DTM, bare ground). Their
difference is the normalized DSM (nDSM), the height of things above the ground.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def ndsm(dsm, dtm, min_height: float = 0.0):
    """Height above ground (m) = DSM - DTM, with small negatives set to 0."""
    h = np.asarray(dsm, dtype="float64") - np.asarray(dtm, dtype="float64")
    return np.where(h < min_height, 0.0, h)


def building_mask(height, classification=None, min_height: float = 3.0,
                  building_code: int = 6, min_pixels: int = 10):
    """Boolean building mask.

    With a lidar classification raster (ASPRS code 6 = building) the mask is
    exact. Without it, everything taller than `min_height` is kept, which also
    keeps tall trees; pass an NDVI-based tree mask to remove them if you have one.
    Small blobs under `min_pixels` are dropped.
    """
    if classification is not None:
        m = np.asarray(classification) == building_code
    else:
        m = np.asarray(height) >= min_height
    lab, n = ndimage.label(m)
    sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
    keep = np.isin(lab, np.flatnonzero(sizes >= min_pixels) + 1)
    return keep


def label_buildings(mask):
    """Give each connected building footprint its own integer id (0 = none)."""
    lab, _ = ndimage.label(mask)
    return lab


def building_heights(height, labels, stat: str = "mean"):
    """Per-pixel building height: the mean (or max, median) nDSM of its footprint."""
    ids = np.arange(1, labels.max() + 1)
    fn = {"mean": ndimage.mean, "max": ndimage.maximum, "median": ndimage.median}[stat]
    vals = np.asarray(fn(height, labels, index=ids)) if ids.size else np.array([])
    out = np.zeros(labels.shape)
    out[labels > 0] = vals[labels[labels > 0] - 1]
    return out


def building_volume(height, mask, pixel_area_m2: float, window: int = 1):
    """Building volume (m³) per pixel, or summed over a window x window block."""
    v = np.where(mask, height, 0.0) * pixel_area_m2
    if window > 1:
        v = ndimage.uniform_filter(v, size=window, mode="constant") * window * window
    return v


def building_density(mask, window: int):
    """Share of a window x window neighborhood covered by buildings (0 to 1)."""
    return ndimage.uniform_filter(np.asarray(mask, dtype="float64"), size=window,
                                  mode="nearest")


def aggregate(values, factor: int, how: str = "mean"):
    """Block-aggregate a fine raster (e.g. 1 m lidar) to a coarse grid (e.g. 30 m)."""
    a = np.asarray(values, dtype="float64")
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    b = a[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return {"mean": np.nanmean, "max": np.nanmax, "sum": np.nansum}[how](b, axis=(1, 3))


def roof_area_index(mask, factor: int, pixel_area_m2: float):
    """BRI: total roof area (m²) inside each coarse cell (factor x factor fine pixels)."""
    return aggregate(np.asarray(mask, dtype="float64") * pixel_area_m2, factor, "sum")


def volume_density(height, mask, factor: int, pixel_area_m2: float):
    """BVD: sum of roof area x height, divided by the coarse cell area (m³ per m²)."""
    vol = aggregate(np.where(mask, height, 0.0) * pixel_area_m2, factor, "sum")
    return vol / (pixel_area_m2 * factor * factor)
