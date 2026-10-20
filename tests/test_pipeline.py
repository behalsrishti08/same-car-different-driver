"""Tests on made-up laps where we KNOW the true answer.

We simulate races where Gold drivers are truly 0.5 s/lap slower than Platinum,
but Gold drivers are also put in slower cars (the confounder). A fair method
must recover ~0.5 s; a naive one will not.

Run with:  python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scd import data, evaluate, gaps, model  # noqa: E402

TRUE_GOLD = 0.5


def fake_laps(seed: int = 0, n_races: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drivers = {f"P{i}": "Platinum" for i in range(20)} | {f"G{i}": "Gold" for i in range(10)}
    skill = {d: rng.normal(0, 0.15) for d in drivers}            # each driver's own bit
    rows = []
    for r in range(n_races):
        base = 90 + 10 * rng.random()                             # track length
        plats = rng.permutation([d for d in drivers if d.startswith("P")])
        golds = rng.permutation([d for d in drivers if d.startswith("G")])
        for c in range(8):
            has_gold = c < 4
            car_pace = rng.normal(0, 0.6) + (0.8 if has_gold else 0.0)   # confounder: gold in slower cars
            crew = [plats[2 * c % 20], golds[c % 10] if has_gold else plats[(2 * c + 1) % 20]]
            lap = 0
            for stint, drv in enumerate(crew * 2, start=1):
                for k in range(30):
                    lap += 1
                    t = (base + car_pace + skill[drv] + (TRUE_GOLD if drivers[drv] == "Gold" else 0)
                         + 0.004 * k + rng.normal(0, 0.8))
                    rows.append(dict(year=2025, event=f"Track{r}", car=c, driver_name=drv, lap=lap,
                                     stint_number=stint, stint_lap=k, lap_time=t, flags="GF",
                                     pit_time=np.nan, est_tire_age=k, air_temp_f=np.nan))
    df = pd.DataFrame(rows)
    df["race"] = df["year"].astype(str) + " " + df["event"]
    df["car_race"] = df["race"] + " #" + df["car"].astype(str)
    df["driver"] = df["driver_name"]
    df["rating"] = pd.Categorical(df["driver"].map(drivers), categories=data.RATING_ORDER, ordered=True)
    df = data.add_race_progress(df)
    laps, _ = data.clean_laps(df)
    return laps


LAPS = fake_laps()


def test_cleaning_keeps_green_racing_laps():
    assert (LAPS["flags"] == "GF").all()
    assert (LAPS["stint_lap"] > 0).all()
    assert LAPS["pct"].abs().max() < 10


def test_naive_comparison_is_biased_by_the_car():
    abl = model.ablation(LAPS).set_index(["model", "rating"])["effect"]
    # Raw/same-race comparisons absorb the slower cars (true 0.5 + 0.8 car penalty).
    assert abl[("M1 same race", "Gold")] > TRUE_GOLD + 0.4


def test_same_car_model_recovers_true_effect():
    res, tiers = model.fit_mixed(model.rated(LAPS))
    assert tiers == ["Gold"]
    assert abs(res.fe_params["Gold"] - TRUE_GOLD) < 0.12


def test_teammate_gaps_are_non_negative_and_fastest_is_zero():
    g = gaps.teammate_gaps(gaps.driver_medians(LAPS))
    assert (g["gap_s"] >= 0).all()
    assert g.groupby("car_race")["is_fastest"].any().all()


def test_placebo_effects_center_on_zero():
    fake = evaluate.placebo(LAPS, n=30)
    assert abs(fake["Gold"].mean()) < 0.25


def test_bootstrap_interval_contains_truth():
    boot = evaluate.bootstrap(LAPS, n=20, mixed=False)
    ci = evaluate.bootstrap_ci(boot).set_index("rating").loc["Gold"]
    assert ci["ci_low"] < TRUE_GOLD < ci["ci_high"]


def test_leave_one_race_out_beats_baseline():
    loro = evaluate.leave_one_race_out(LAPS)
    assert len(loro) == LAPS["race"].nunique()
    assert loro["mae_model"].mean() < loro["mae_baseline"].mean()
