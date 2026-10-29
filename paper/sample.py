"""Green roof cooling with a neural network, as in Fig. 8 of the paper.

What it shows: a 95-10 neural network learns daytime LST (DLST, deg F) from eight
urban form inputs (BH, BRI, BVD, SVF, SR, NDVI, NDBI, WBD), scored with spatial
block cross-validation (Table 2). The hottest roofs, about 4.2% of roof cells, then
get lawn NDVI (0.55) and NDBI (-0.178), and the network predicts DLST again.
The difference is the simulated green roof cooling (Fig. 8c).

Synthetic data: numbers will not match the paper.
Run: python paper/sample.py [--plot]
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greenroof import model, simulate

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="save sample_output.png")
args = parser.parse_args()
warnings.filterwarnings("ignore", message="lbfgs failed")  # short demo fits

rng = np.random.default_rng(11)
shape = (40, 40)  # 30 m cells
rows, cols = np.indices(shape)
roof = rng.random(shape) < 0.35  # cells that hold a building roof
bh = np.where(roof, rng.gamma(2.0, 8.0, shape), 0.0)  # building height (m)
X = pd.DataFrame({
    "BH": bh.ravel(), "BRI": (roof * rng.uniform(200, 900, shape)).ravel(),
    "BVD": (bh * roof * rng.uniform(0.2, 1.0, shape)).ravel(),
    "SVF": np.clip(1 - bh / 80 + rng.normal(0, 0.05, shape), 0.2, 1).ravel(),
    "SR": rng.uniform(400, 2000, shape).ravel(),
    "NDVI": np.where(roof, rng.uniform(-0.1, 0.2, shape), rng.uniform(0.1, 0.6, shape)).ravel(),
    "NDBI": np.where(roof, rng.uniform(0.0, 0.4, shape), rng.uniform(-0.4, 0.1, shape)).ravel(),
    "WBD": (30 * np.hypot(rows - 39, cols - 5)).ravel(),
})
dlst = (95 - 5 * X.NDVI + 3 * X.NDBI + 0.002 * X.WBD - 0.03 * X.BH + 0.001 * X.SR
        + rng.normal(0, 0.8, len(X)))

groups = model.spatial_groups(rows.ravel(), cols.ravel(), block=10)
net = {"95-10 lbfgs tanh (like trainbfg)": model.NETWORKS["95-10 lbfgs tanh (like trainbfg)"]}
print(model.compare_networks(X, dlst, groups, networks=net, n_splits=3).round(3).to_string())

fit = model.make_network(next(iter(net.values()))).fit(X, dlst)
before = fit.predict(X).reshape(shape)
labels = np.where(roof, np.arange(roof.size).reshape(shape) + 1, 0)  # one building per cell
chosen = simulate.select_rooftops(labels, before, target_share=0.042)
ndvi, ndbi = simulate.green_roof_scenario(X.NDVI.values.reshape(shape),
                                          X.NDBI.values.reshape(shape), chosen)
after = fit.predict(X.assign(NDVI=ndvi.ravel(), NDBI=ndbi.ravel())).reshape(shape)
print(f"\nGreen roof cells: {chosen.sum()} of {roof.sum()} roof cells")
print(pd.Series(simulate.cooling_summary(before[chosen], after[chosen])).round(2).to_string())

if args.plot:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(before, cmap="RdYlGn_r")
    axes[0].set_title("Predicted DLST before (deg F)")
    im1 = axes[1].imshow(np.where(chosen, after - before, np.nan), cmap="Blues_r")
    axes[1].set_title("DLST change on green roofs (deg F)")
    for ax, im in zip(axes, (im0, im1), strict=True):
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_axis_off()
    fig.savefig("sample_output.png", dpi=120, bbox_inches="tight")
    print("Saved sample_output.png")
