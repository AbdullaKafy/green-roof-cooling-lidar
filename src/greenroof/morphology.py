"""Urban form layers: sky view factor, solar radiation, distance to water."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def sky_view_factor(surface, pixel_m: float, radius_m: float = 100.0, directions: int = 16):
    """Sky view factor (0 = closed, 1 = open sky) from a surface model.

    For each direction, find the highest horizon angle within `radius_m`, then
    SVF = 1 - mean(sin(horizon angle)). This is the common horizon method for
    raster SVF. It is slow for very large rasters; tile them first.
    """
    z = np.asarray(surface, dtype="float64")
    steps = max(1, int(radius_m / pixel_m))
    sin_sum = np.zeros_like(z)
    pad = steps + 1
    zp = np.pad(z, pad, mode="edge")
    h, w = z.shape
    for k in range(directions):
        ang = 2 * np.pi * k / directions
        dr, dc = -np.cos(ang), np.sin(ang)
        best = np.zeros_like(z)
        for s in range(1, steps + 1):
            r = int(round(dr * s))
            c = int(round(dc * s))
            if r == 0 and c == 0:
                continue
            dist = np.hypot(r, c) * pixel_m
            other = zp[pad + r: pad + r + h, pad + c: pad + c + w]
            best = np.maximum(best, np.arctan2(other - z, dist))
        sin_sum += np.sin(best)
    return 1 - sin_sum / directions


def solar_radiation_proxy(surface, pixel_m: float, sun_elevation_deg: float,
                          sun_azimuth_deg: float):
    """Relative direct sun on each pixel (0 to 1): hillshade times not-in-shadow.

    A light stand-in for GIS solar radiation tools. It uses one sun position
    (for example the Landsat overpass time) instead of a full-day integral.
    """
    z = np.asarray(surface, dtype="float64")
    gy, gx = np.gradient(z, pixel_m)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    el = np.radians(sun_elevation_deg)
    az = np.radians(sun_azimuth_deg)
    shade = np.sin(el) * np.cos(slope) + np.cos(el) * np.sin(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    return shade * (1 - cast_shadow(z, pixel_m, sun_elevation_deg, sun_azimuth_deg))


def cast_shadow(surface, pixel_m: float, sun_elevation_deg: float, sun_azimuth_deg: float,
                max_m: float = 300.0):
    """1 where a pixel is in the shadow of a taller object, else 0."""
    z = np.asarray(surface, dtype="float64")
    az = np.radians(sun_azimuth_deg)
    dr, dc = -np.cos(az), np.sin(az)  # step toward the sun
    tan_el = np.tan(np.radians(sun_elevation_deg))
    steps = int(max_m / pixel_m)
    pad = steps + 1
    zp = np.pad(z, pad, mode="edge")
    h, w = z.shape
    shadow = np.zeros(z.shape, dtype=bool)
    for s in range(1, steps + 1):
        r, c = int(round(dr * s)), int(round(dc * s))
        other = zp[pad + r: pad + r + h, pad + c: pad + c + w]
        shadow |= other - z > np.hypot(r, c) * pixel_m * tan_el
    return shadow.astype(float)


def distance_to_water(water_mask, pixel_m: float):
    """Distance (m) from every pixel to the nearest water pixel."""
    return ndimage.distance_transform_edt(~np.asarray(water_mask, dtype=bool)) * pixel_m


def sun_position(lat: float, lon: float, day_of_year: int, utc_hour: float):
    """Approximate sun elevation and azimuth (degrees) from the NOAA equations.

    Accurate to well under a degree, which is enough for shading on a 2 m grid.
    """
    g = 2 * np.pi / 365 * (day_of_year - 1 + (utc_hour - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = utc_hour * 60 + eqt + 4 * lon
    ha = np.radians(tst / 4 - 180)
    la = np.radians(lat)
    cos_z = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cos_z, -1, 1))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(la) - np.tan(decl) * np.cos(la)))
    return 90 - np.degrees(zen), (az + 180) % 360


def daily_direct_insolation(surface, pixel_m: float, lat: float, lon: float, day_of_year: int,
                            dni_w_m2: float = 850.0, step_h: float = 1.0, max_m: float = 200.0):
    """Clear-sky direct-beam insolation for one day (Wh/m²), with terrain and building shade.

    Sums hillshade x (not in shadow) x DNI over the hours the sun is up. Diffuse
    and reflected light are left out, so values are lower than full GIS solar
    radiation tools; use it to compare places, not as an absolute energy figure.
    """
    total = np.zeros(np.shape(surface))
    for h in np.arange(0, 24, step_h):
        el, az = sun_position(lat, lon, day_of_year, h + step_h / 2)
        if el <= 2:
            continue
        lit = solar_radiation_proxy(surface, pixel_m, el, az)
        total += lit * dni_w_m2 * step_h
    return total
