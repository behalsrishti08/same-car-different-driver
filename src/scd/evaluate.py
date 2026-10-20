"""Checks that the rating effect is trustworthy (evaluation plan).

- bootstrap:  resample whole races ~N times, refit, take the spread -> error bars
- loro:       leave-one-race-out: fit on 32 races, test on the 33rd, repeat
- placebo:    shuffle ratings between drivers; the effect should drop to ~0

Everything is done by race, never by lap, because laps of one race share
weather, track and traffic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import CONTROLS, design, fit_mixed, fit_ols, rated


def bootstrap(laps: pd.DataFrame, n: int = 200, y: str = "lap_time", seed: int = 0,
              mixed: bool = True) -> pd.DataFrame:
    d = rated(laps)
    races = d["race"].unique()
    by_race = {r: d[d["race"] == r] for r in races}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n):
        pick = rng.choice(races, size=len(races), replace=True)
        parts = []
        for k, r in enumerate(pick):
            part = by_race[r].copy()
            part["car_race"] = part["car_race"] + f"|{k}"   # a repeated race counts as a new one
            part["race"] = part["race"] + f"|{k}"
            parts.append(part)
        s = pd.concat(parts, ignore_index=True)
        try:
            res, tiers = fit_mixed(s, y) if mixed else fit_ols(s, y, "car_race", CONTROLS)
        except Exception:      # a resample can miss a whole tier; skip it
            continue
        rows.append({t: res.params.get(t, np.nan) for t in tiers} | {"b": b})
    return pd.DataFrame(rows)


def bootstrap_ci(boot: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    a = (1 - level) / 2
    cols = [c for c in boot.columns if c != "b"]
    return pd.DataFrame({
        "rating": cols,
        "boot_mean": [boot[c].mean() for c in cols],
        "ci_low": [boot[c].quantile(a) for c in cols],
        "ci_high": [boot[c].quantile(1 - a) for c in cols],
        "n_resamples": [int(boot[c].notna().sum()) for c in cols],
    }).round(4)


def _metrics(y_true, y_pred):
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return mae, rmse, r2


def leave_one_race_out(laps: pd.DataFrame, y: str = "lap_time") -> pd.DataFrame:
    """For each race: fit on the others, predict each lap's gap to its own car's
    average in the held-out race. Baseline predicts 0 (every driver equal).

    Drivers may be unseen in the held-out race, so predictions use only the
    fixed part (rating + tyres + race hour), which is the honest test."""
    d = rated(laps)
    rows = []
    for r in d["race"].unique():
        train, test = d[d["race"] != r], d[d["race"] == r]
        if test["rating"].nunique() < 1 or len(test) < 50:
            continue
        res, tiers = fit_mixed(train, y)
        Yt, Xt, _ = design(test, y, "car_race", CONTROLS)
        for t in tiers:
            if t not in Xt:
                Xt[t] = 0.0
        pred = Xt[res.fe_params.index].to_numpy() @ res.fe_params.to_numpy()
        mae, rmse, r2 = _metrics(Yt.to_numpy(), pred)
        bmae, brmse, _ = _metrics(Yt.to_numpy(), np.zeros(len(Yt)))
        rows.append({"held_out_race": r, "test_laps": len(test),
                     "mae_model": mae, "mae_baseline": bmae,
                     "rmse_model": rmse, "rmse_baseline": brmse, "r2_model": r2}
                    | {f"effect_{t}": res.fe_params[t] for t in tiers})
    return pd.DataFrame(rows)


def placebo(laps: pd.DataFrame, n: int = 300, y: str = "lap_time", seed: int = 1) -> pd.DataFrame:
    """Shuffle which driver has which rating (keeping how many of each),
    refit the within-car model, and record the fake 'effects'."""
    d = rated(laps)
    drivers = d.drop_duplicates("driver")[["driver", "rating"]].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        fake = dict(zip(drivers["driver"], rng.permutation(drivers["rating"].to_numpy())))
        s = d.copy()
        s["rating"] = pd.Categorical(s["driver"].map(fake), categories=d["rating"].cat.categories)
        res, tiers = fit_ols(s, y, "car_race", CONTROLS)
        rows.append({t: res.params[t] for t in tiers} | {"i": i})
    return pd.DataFrame(rows)


def placebo_pvalues(real: pd.Series, fake: pd.DataFrame) -> pd.DataFrame:
    """Share of shuffles with an effect at least as large as the real one."""
    out = []
    for t, v in real.items():
        if t not in fake:
            continue
        f = fake[t].dropna()
        p = (np.sum(np.abs(f) >= abs(v)) + 1) / (len(f) + 1)
        out.append({"rating": t, "real_effect": v, "placebo_mean": f.mean(),
                    "placebo_sd": f.std(), "permutation_p": p})
    return pd.DataFrame(out).round(4)
