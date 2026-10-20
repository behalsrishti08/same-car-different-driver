"""Models for the rating effect (Experiment 3) and the checks (Experiment 5).

The core trick is the *within-car* comparison: every variable is measured
relative to its own car's average in that race ("demeaning"). That removes
everything the teammates share (car, team, setup, track, weather on the day),
so the rating effect is learned only from teammates in the same car.

Models, added in blocks (ablation):
  M0  raw          rating only, no controls (the confounded baseline)
  M1  same race    compare within each race
  M2  same car     compare within each car in each race
  M3  + tyres      M2 + tyre age + hours into the race
  M4  + driver     M3 + a random intercept for each driver (mixed-effects model)

M4 is the main answer. M0-M3 use ordinary least squares with standard errors
clustered by race; M4 uses statsmodels MixedLM.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.regression.mixed_linear_model import MixedLM

REFERENCE = "Platinum"
CONTROLS = ["est_tire_age", "race_hour"]


def rated(laps: pd.DataFrame) -> pd.DataFrame:
    return laps.dropna(subset=["rating"]).reset_index(drop=True)


def tier_columns(laps: pd.DataFrame) -> list[str]:
    present = [str(r) for r in laps["rating"].cat.categories if (laps["rating"] == r).any()]
    return [r for r in present if r != REFERENCE]


def design(laps: pd.DataFrame, y: str, group: str | None, controls: list[str]):
    """Build (y, X) with rating dummies and controls, demeaned within `group`."""
    tiers = tier_columns(laps)
    X = pd.DataFrame({t: (laps["rating"] == t).astype(float) for t in tiers})
    for c in controls:
        X[c] = laps[c].astype(float)
    Y = laps[y].astype(float)
    if group is None:
        X = add_constant(X)
    else:
        g = laps[group]
        X = X - X.groupby(g).transform("mean")
        Y = Y - Y.groupby(g).transform("mean")
    return Y, X, tiers


def fit_ols(laps, y, group, controls):
    Y, X, tiers = design(laps, y, group, controls)
    clusters = laps["race"].astype("category").cat.codes
    res = OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    return res, tiers


def fit_mixed(laps, y="lap_time", controls=CONTROLS):
    """M4: within-car demeaned outcome, driver random intercept."""
    Y, X, tiers = design(laps, y, "car_race", controls)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = MixedLM(Y, X, groups=laps["driver"]).fit(reml=True, method="lbfgs")
    return res, tiers


MODELS = [
    ("M0 raw", None, []),
    ("M1 same race", "race", []),
    ("M2 same car", "car_race", []),
    ("M3 + tyres & race hour", "car_race", CONTROLS),
]


def ablation(laps: pd.DataFrame, y: str = "lap_time") -> pd.DataFrame:
    """Rating coefficients as controls are added. Shows how the gap shrinks."""
    d = rated(laps)
    rows = []
    for name, group, controls in MODELS:
        res, tiers = fit_ols(d, y, group, controls)
        ci = res.conf_int()
        for t in tiers:
            rows.append((name, t, res.params[t], res.bse[t], ci.loc[t, 0], ci.loc[t, 1], res.pvalues[t]))
    res, tiers = fit_mixed(d, y)
    ci = res.conf_int()
    for t in tiers:
        rows.append(("M4 + driver (mixed)", t, res.params[t], res.bse[t], ci.loc[t, 0], ci.loc[t, 1], res.pvalues[t]))
    out = pd.DataFrame(rows, columns=["model", "rating", "effect", "se", "ci_low", "ci_high", "p_value"])
    return out


def identification(laps: pd.DataFrame) -> dict:
    """How much within-car rating variation exists (what the comparison relies on)."""
    d = rated(laps)
    tiers_per_car = d.groupby("car_race")["rating"].nunique()
    unrated = laps[laps["rating"].isna()]
    return {
        "rated_laps": int(len(d)),
        "unrated_laps_dropped": int(len(unrated)),
        "unrated_drivers": sorted(unrated["driver"].unique().tolist()),
        "car_races_with_2plus_tiers": int((tiers_per_car >= 2).sum()),
        "car_races_total": int(len(tiers_per_car)),
        "laps_by_rating": d["rating"].value_counts().reindex(d["rating"].cat.categories).fillna(0).astype(int).to_dict(),
        "drivers_by_rating": d.groupby("rating", observed=False)["driver"].nunique().to_dict(),
    }
