"""Loading, driver-name cleanup, ratings join and lap cleaning.

Every cleaning step is recorded in an audit table so the report can show
how many laps each rule removed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RATING_ORDER = ["Platinum", "Gold", "Silver", "Bronze"]


@dataclass
class CleaningRules:
    keep_flags: tuple = ("GF",)          # green-flag racing only
    drop_pit_in: bool = True             # lap with a pit stop recorded
    drop_out_lap: bool = True            # first lap of each stint (leaving the pits)
    slow_factor: float = 1.07            # drop laps slower than 107% of the race median
    min_laps_per_driver: int = 10        # driver needs this many clean laps in a car-race


def load_raw(raw_dir: str | Path) -> pd.DataFrame:
    """Read every laps_imsa_*.csv file in raw_dir into one table."""
    files = sorted(Path(raw_dir).glob("laps_imsa_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No laps_imsa_*.csv files in {raw_dir}. Download the 'imsa' folder of the "
            "Kaggle dataset 'motorsport-strategy-lab-derived' and put the CSVs there."
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["race"] = df["year"].astype(str) + " " + df["event"]
    df["car_race"] = df["race"] + " #" + df["car"].astype(str)
    return df


def fix_driver_names(df: pd.DataFrame, aliases_csv: str | Path) -> pd.DataFrame:
    """Merge spellings of the same person (e.g. 'Alexander Palou' -> 'Alex Palou')."""
    aliases = dict(pd.read_csv(aliases_csv)[["alias", "canonical"]].values)
    out = df.copy()
    out["driver"] = out["driver_name"].replace(aliases)
    return out


def add_ratings(df: pd.DataFrame, ratings_csv: str | Path) -> pd.DataFrame:
    """Attach the FIA rating to every lap. Drivers without a rating get NaN."""
    ratings = pd.read_csv(ratings_csv, keep_default_na=False)
    ratings = ratings[ratings["rating"].isin(RATING_ORDER)][["driver_name", "rating"]]
    ratings = ratings.rename(columns={"driver_name": "driver"})
    out = df.merge(ratings, on="driver", how="left")
    out["rating"] = pd.Categorical(out["rating"], categories=RATING_ORDER, ordered=True)
    return out


def add_race_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Hours since the start of the race for each lap (cumulative lap time per car)."""
    out = df.sort_values(["car_race", "lap"]).copy()
    out["race_hour"] = out.groupby("car_race")["lap_time"].cumsum() / 3600.0
    return out


def clean_laps(df: pd.DataFrame, rules: CleaningRules = CleaningRules()):
    """Apply the cleaning rules. Returns (clean_laps, audit_table)."""
    audit = [("All laps (top class, named drivers)", len(df))]
    out = df[df["flags"].isin(rules.keep_flags)]
    audit.append(("Green-flag laps only (no cautions)", len(out)))
    if rules.drop_pit_in:
        out = out[out["pit_time"].isna()]
        audit.append(("Remove pit-in laps", len(out)))
    if rules.drop_out_lap:
        out = out[out["stint_lap"] > 0]
        audit.append(("Remove out-laps (first lap of each stint)", len(out)))
    race_median = out.groupby("race")["lap_time"].transform("median")
    out = out[out["lap_time"] <= rules.slow_factor * race_median]
    audit.append((f"Remove laps slower than {rules.slow_factor:.0%} of race median", len(out)))
    n_laps = out.groupby(["car_race", "driver"])["lap_time"].transform("size")
    out = out[n_laps >= rules.min_laps_per_driver]
    audit.append((f"Drivers with at least {rules.min_laps_per_driver} clean laps in a car", len(out)))

    out = out.copy()
    # Lap time as % slower than the race's median clean lap: makes tracks comparable.
    out["pct"] = 100.0 * (out["lap_time"] / out.groupby("race")["lap_time"].transform("median") - 1.0)
    audit = pd.DataFrame(audit, columns=["step", "laps"])
    audit["share_of_all"] = (audit["laps"] / audit["laps"].iloc[0]).round(3)
    return out.reset_index(drop=True), audit


def describe(df: pd.DataFrame) -> dict:
    """Headline counts for the report."""
    per_car = df.groupby("car_race")["driver"].nunique()
    return {
        "laps": int(len(df)),
        "races": int(df["race"].nunique()),
        "car_races": int(df["car_race"].nunique()),
        "drivers": int(df["driver"].nunique()),
        "drivers_with_rating": int(df.loc[df["rating"].notna(), "driver"].nunique()),
        "most_common_drivers_per_car": int(per_car.mode().iloc[0]),
        "weather_missing_share": float(np.round(df["air_temp_f"].isna().mean(), 3)),
    }


def prepare(raw_dir, aliases_csv, ratings_csv, rules: CleaningRules = CleaningRules()):
    """Full data pipeline: load -> fix names -> ratings -> race progress -> clean."""
    df = load_raw(raw_dir)
    df = fix_driver_names(df, aliases_csv)
    df = add_ratings(df, ratings_csv)
    df = add_race_progress(df)
    return clean_laps(df, rules)
