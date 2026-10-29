"""Neural network models that predict daytime LST from urban form.

`compare_networks` fits several neural network set-ups on the same data and
scores each one with spatial block cross-validation, so the best model is picked
on how well it predicts places it has not seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The paper used one network shape, two hidden layers of 95 and 10 neurons, and
# compared MATLAB training algorithms (trainbfg was best). scikit-learn has
# different optimizers, so this set keeps the 95-10 shape and varies the
# optimizer and activation. The closest match to trainbfg (BFGS quasi-Newton)
# is "lbfgs".
NETWORKS = {
    "95-10 lbfgs tanh (like trainbfg)": dict(solver="lbfgs", activation="tanh"),
    "95-10 lbfgs relu": dict(solver="lbfgs", activation="relu"),
    "95-10 adam relu": dict(solver="adam", activation="relu"),
    "95-10 sgd momentum (like traingdx)": dict(solver="sgd", activation="relu",
                                               momentum=0.9, learning_rate="adaptive"),
}
HIDDEN = (95, 10)


def make_network(params: dict, seed: int = 0):
    p = {"hidden_layer_sizes": HIDDEN, "max_iter": 300, "alpha": 1e-3, "random_state": seed,
         **params}
    if p.get("solver", "adam") == "adam":
        p.setdefault("early_stopping", True)
    return make_pipeline(StandardScaler(), MLPRegressor(**p))


def spatial_groups(rows, cols, block: int) -> np.ndarray:
    """Block id for each sample from its row and column on the grid."""
    return (np.asarray(rows) // block) * 100_000 + (np.asarray(cols) // block)


def compare_networks(X: pd.DataFrame, y, groups, networks: dict = NETWORKS,
                     n_splits: int = 5, seed: int = 0) -> pd.DataFrame:
    """R² and RMSE of each network under spatial block cross-validation."""
    y = np.asarray(y, dtype="float64")
    rows = []
    for name, params in networks.items():
        r2s, rmses = [], []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = make_network(params, seed).fit(X.iloc[tr], y[tr])
            p = m.predict(X.iloc[te])
            r2s.append(r2_score(y[te], p))
            rmses.append(float(np.sqrt(mean_squared_error(y[te], p))))
        rows.append({"model": name, "r2": float(np.mean(r2s)), "rmse": float(np.mean(rmses))})
    return pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)


def sensitivity(model, X: pd.DataFrame, y, n_repeats: int = 10, seed: int = 0) -> pd.Series:
    """Importance of each input: drop in R² when that input is shuffled."""
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed,
                               scoring="r2")
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def drop_column_importance(params: dict, X: pd.DataFrame, y, groups, n_splits: int = 5,
                           seed: int = 0) -> pd.DataFrame:
    """Leave each input out, refit, and report the rise in cross-validated MSE.

    This follows the paper's sensitivity test (Table 4): a big rise means the
    model needs that input.
    """
    y = np.asarray(y, dtype="float64")

    def cv_mse(cols):
        errs = []
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            m = make_network(params, seed).fit(X.iloc[tr][cols], y[tr])
            errs.append(mean_squared_error(y[te], m.predict(X.iloc[te][cols])))
        return float(np.mean(errs))

    base = cv_mse(list(X.columns))
    rows = [{"left_out": c, "mse": cv_mse([k for k in X.columns if k != c])}
            for c in X.columns]
    out = pd.DataFrame(rows)
    out["mse_increase"] = out["mse"] - base
    return out.sort_values("mse_increase", ascending=False).reset_index(drop=True), base
