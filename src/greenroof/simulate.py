"""Green roof scenarios: choose roofs, change their surface, predict DLST again.

The paper picked residential and commercial roofs on DLST hot spots by eye,
covering about 4.2% of all roof area, and gave those roofs the NDVI (0.55) and
NDBI (-0.178) of a city park lawn. Here the choice is written as a rule so it can
be repeated: take buildings on the hottest cells first until the target share of
roof area is reached, with optional height and size limits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAWN_NDVI = 0.55
LAWN_NDBI = -0.178


def select_rooftops(labels, dlst_on_fine_grid, target_share: float = 0.042,
                    min_height: float | None = None, max_height: float | None = None,
                    heights=None, min_area_px: int = 0):
    """Boolean mask of roofs to turn green.

    labels: building ids on the fine grid (0 = not a building).
    dlst_on_fine_grid: DLST resampled to the fine grid (hotter roofs go first).
    target_share: share of all roof pixels to convert (0.042 = 4.2%).
    """
    ids = np.unique(labels[labels > 0])
    df = pd.DataFrame({"id": ids})
    flat_l = labels.ravel()
    m = flat_l > 0
    s = pd.DataFrame({"id": flat_l[m], "t": np.asarray(dlst_on_fine_grid).ravel()[m]})
    if heights is not None:
        s["h"] = np.asarray(heights).ravel()[m]
    g = s.groupby("id")
    df = df.join(g["t"].mean().rename("dlst"), on="id").join(g.size().rename("px"), on="id")
    if heights is not None:
        df = df.join(g["h"].mean().rename("height"), on="id")
        if min_height is not None:
            df = df[df.height >= min_height]
        if max_height is not None:
            df = df[df.height <= max_height]
    df = df[df.px >= min_area_px].sort_values("dlst", ascending=False)
    goal = target_share * m.sum()
    chosen, total = [], 0
    for bid, px in zip(df.id, df.px, strict=True):  # hottest first, skip roofs that overshoot
        if total + px <= goal:
            chosen.append(bid)
            total += px
    return np.isin(labels, chosen)


def green_roof_scenario(ndvi_fine, ndbi_fine, roof_mask, ndvi_value: float = LAWN_NDVI,
                        ndbi_value: float = LAWN_NDBI):
    """NDVI and NDBI after the chosen roofs become green (fine grid)."""
    ndvi = np.where(roof_mask, ndvi_value, ndvi_fine)
    ndbi = np.where(roof_mask, ndbi_value, ndbi_fine)
    return ndvi, ndbi


def cooling_summary(before, after, no_change: float = 0.02) -> dict:
    """DLST change (after - before) on cells that changed, in the input unit."""
    d = np.asarray(after, dtype="float64") - np.asarray(before, dtype="float64")
    d = d[np.isfinite(d)]
    changed = d[np.abs(d) > no_change]
    return {
        "cells_changed": int(changed.size),
        "mean_change": float(changed.mean()) if changed.size else 0.0,
        "min_change": float(d.min()),
        "max_change": float(d.max()),
        "cells_cooler": int(np.sum(d < -no_change)),
        "cells_warmer": int(np.sum(d > no_change)),
    }
