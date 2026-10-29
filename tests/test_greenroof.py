"""Offline tests on small synthetic rasters."""

import numpy as np
import pandas as pd

from greenroof import lidar, model, morphology, simulate


def city():
    dtm = np.zeros((60, 60))
    dsm = dtm.copy()
    dsm[10:20, 10:20] = 30  # tall block
    dsm[40:44, 40:44] = 6   # low block
    return dsm, dtm


def test_ndsm_and_buildings():
    dsm, dtm = city()
    h = lidar.ndsm(dsm, dtm)
    mask = lidar.building_mask(h, min_height=3, min_pixels=5)
    lab = lidar.label_buildings(mask)
    assert lab.max() == 2
    bh = lidar.building_heights(h, lab)
    assert bh[15, 15] == 30 and bh[41, 41] == 6
    assert lidar.roof_area_index(mask, 30, 1.0).sum() == 116
    assert np.isclose(lidar.volume_density(h, mask, 60, 1.0)[0, 0], (100 * 30 + 16 * 6) / 3600)


def test_svf_open_and_closed():
    dsm, _ = city()
    svf = morphology.sky_view_factor(dsm, pixel_m=1.0, radius_m=15, directions=8)
    assert np.isclose(svf[55, 5], 1.0)       # open ground far from buildings
    assert svf[21, 15] < 0.8                 # street next to the tall block
    assert np.isclose(svf[15, 15], 1.0)      # roof top sees open sky


def test_shadow_and_insolation():
    dsm, _ = city()
    sh = morphology.cast_shadow(dsm, 1.0, sun_elevation_deg=45, sun_azimuth_deg=180)
    assert sh[5, 15] == 1 and sh[21, 15] == 0  # north side (small rows) shaded, sun in south
    ins = morphology.daily_direct_insolation(dsm, 1.0, 30.27, -97.74, 223, max_m=40)
    assert ins[15, 15] > ins[8, 15]


def test_sun_position_noon():
    el, az = morphology.sun_position(30.27, -97.74, 172, 18.5)  # near solar noon, June
    assert el > 80 and 100 < az < 260


def test_distance_to_water():
    w = np.zeros((5, 5), bool)
    w[0, 0] = True
    assert morphology.distance_to_water(w, 10.0)[0, 3] == 30


def test_select_and_scenario():
    labels = np.zeros((10, 10), int)
    labels[0:2, 0:2] = 1
    labels[5:7, 5:7] = 2
    heat = np.zeros((10, 10))
    heat[5:7, 5:7] = 10
    sel = simulate.select_rooftops(labels, heat, target_share=0.5)
    assert sel[5, 5] and not sel[0, 0]
    ndvi, ndbi = simulate.green_roof_scenario(np.zeros((10, 10)), np.ones((10, 10)), sel)
    assert ndvi[5, 5] == simulate.LAWN_NDVI and ndbi[0, 0] == 1
    s = simulate.cooling_summary(np.array([10.0, 10.0, 10.0]), np.array([8.0, 10.0, 10.01]))
    assert s["cells_cooler"] == 1 and s["mean_change"] == -2


def test_compare_networks_runs():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list("abc"))
    y = 2 * X.a - X.b + rng.normal(scale=0.1, size=300)
    groups = np.arange(300) // 30
    nets = {"lbfgs": dict(solver="lbfgs", activation="tanh")}
    res = model.compare_networks(X, y, groups, nets, n_splits=3)
    assert res.r2.iloc[0] > 0.9
    imp, base = model.drop_column_importance(nets["lbfgs"], X, y, groups, n_splits=3)
    assert imp.left_out.iloc[0] == "a"
