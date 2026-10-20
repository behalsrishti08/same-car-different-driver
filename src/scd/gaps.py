"""Descriptive teammate comparisons (Experiments 1 and 2).

A "gap" is how many seconds per lap a driver's median clean lap is slower
than the fastest teammate's median in the same car and the same race.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def driver_medians(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per car per race: median clean lap time and lap count."""
    g = laps.groupby(["race", "car_race", "car", "driver"], observed=True)
    out = g.agg(
        median_lap=("lap_time", "median"),
        median_pct=("pct", "median"),
        laps=("lap_time", "size"),
        rating=("rating", "first"),
    ).reset_index()
    return out


def teammate_gaps(medians: pd.DataFrame) -> pd.DataFrame:
    """Gap of every driver to the fastest teammate in the same car-race.

    Car-races with only one driver are dropped (there is no teammate).
    The fastest driver of each car has gap 0 and is flagged with is_fastest.
    """
    m = medians.copy()
    m["n_teammates"] = m.groupby("car_race")["driver"].transform("size")
    m = m[m["n_teammates"] >= 2].copy()
    m["gap_s"] = m["median_lap"] - m.groupby("car_race")["median_lap"].transform("min")
    m["is_fastest"] = m["gap_s"] == 0
    return m


def gap_summary(gaps: pd.DataFrame) -> dict:
    g = gaps.loc[~gaps["is_fastest"], "gap_s"]
    return {
        "n_gaps": int(len(g)),
        "median_s": float(round(g.median(), 3)),
        "iqr_low_s": float(round(g.quantile(0.25), 3)),
        "iqr_high_s": float(round(g.quantile(0.75), 3)),
        "share_under_0_5s": float(round((g < 0.5).mean(), 3)),
    }


def gaps_by_race(gaps: pd.DataFrame) -> pd.DataFrame:
    g = gaps[~gaps["is_fastest"]]
    return (g.groupby("race")["gap_s"].median().rename("median_gap_s")
             .sort_values(ascending=False).reset_index())


def car_vs_driver_split(medians: pd.DataFrame) -> dict:
    """Within each race, split the spread of driver pace into
    between-car (cars differ) and within-car (teammates differ) parts.
    Sum of squares of driver medians around the race mean = between + within."""
    m = medians.copy()
    m["race_mean"] = m.groupby("race")["median_lap"].transform("mean")
    m["car_mean"] = m.groupby("car_race")["median_lap"].transform("mean")
    between = ((m["car_mean"] - m["race_mean"]) ** 2).sum()
    within = ((m["median_lap"] - m["car_mean"]) ** 2).sum()
    total = between + within
    return {"between_car_share": float(round(between / total, 3)),
            "within_car_share": float(round(within / total, 3))}


def pairwise_vs_platinum(medians: pd.DataFrame) -> pd.DataFrame:
    """Experiment 2: in cars that had a Platinum driver AND a lower-rated driver,
    how much slower was the lower-rated driver than the car's Platinum driver(s)?
    One row per lower-rated driver per car-race."""
    m = medians.dropna(subset=["rating"]).copy()
    plat = (m[m["rating"] == "Platinum"].groupby("car_race")["median_lap"].mean()
             .rename("platinum_median"))
    low = m[m["rating"] != "Platinum"].join(plat, on="car_race", how="inner")
    low["gap_to_platinum_s"] = low["median_lap"] - low["platinum_median"]
    return low


def pairwise_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    return (pairs.groupby("rating", observed=True)["gap_to_platinum_s"]
                 .agg(n="size", median="median", mean="mean",
                      share_slower=lambda s: (s > 0).mean())
                 .round(3).reset_index())


def raw_by_rating(laps: pd.DataFrame) -> pd.DataFrame:
    """Experiment 1: the unadjusted comparison (confounded baseline)."""
    d = laps.dropna(subset=["rating"])
    return (d.groupby("rating", observed=True)
             .agg(laps=("pct", "size"), drivers=("driver", "nunique"),
                  median_pct=("pct", "median"), mean_pct=("pct", "mean"),
                  mean_lap_s=("lap_time", "mean"))
             .round(3).reset_index())


def tyre_slope(laps: pd.DataFrame) -> float:
    """Seconds added per extra lap on the tyres, within each car-race."""
    d = laps[["car_race", "lap_time", "est_tire_age"]].copy()
    for c in ["lap_time", "est_tire_age"]:
        d[c] = d[c] - d.groupby("car_race")[c].transform("mean")
    x, y = d["est_tire_age"].to_numpy(), d["lap_time"].to_numpy()
    return float(np.dot(x, y) / np.dot(x, x))
