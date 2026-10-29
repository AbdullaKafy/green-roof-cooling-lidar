"""Spectral indices (reflectance 0 to 1)."""

from __future__ import annotations

import numpy as np


def _nd(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def ndvi(nir, red):
    """Vegetation: (NIR - Red) / (NIR + Red)."""
    return _nd(nir, red)


def ndbi(swir1, nir):
    """Built-up: (SWIR1 - NIR) / (SWIR1 + NIR)."""
    return _nd(swir1, nir)


def mndwi(green, swir1):
    """Water: (Green - SWIR1) / (Green + SWIR1)."""
    return _nd(green, swir1)
