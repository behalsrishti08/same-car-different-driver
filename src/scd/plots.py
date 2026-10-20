"""Figures for the report. Each function saves one PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ACCENT = "#b8393d"   # the red from the slides
GREY = "#8a8a94"
TIER_COLORS = {"Platinum": "#b8393d", "Gold": "#c9a227", "Silver": "#8a8a94", "Bronze": "#a0663a"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
})


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def cleaning_funnel(audit: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(audit))[::-1]
    ax.barh(y, audit["laps"], color=[GREY] * (len(audit) - 1) + [ACCENT])
    ax.set_yticks(y, audit["step"])
    for yi, v in zip(y, audit["laps"]):
        ax.text(v, yi, f" {v:,}", va="center", fontsize=9)
    ax.set_xlabel("Laps")
    ax.set_title("From raw laps to clean racing laps")
    ax.set_xlim(0, audit["laps"].max() * 1.15)
    return _save(fig, out)


def gaps_by_race(by_race: pd.DataFrame, out: Path) -> Path:
    d = by_race.sort_values("median_gap_s")
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [ACCENT if v > 1 else GREY for v in d["median_gap_s"]]
    ax.barh(d["race"], d["median_gap_s"], color=colors)
    ax.axvline(d["median_gap_s"].median(), color="black", lw=0.8, ls="--")
    ax.set_xlabel("Median gap to fastest teammate (s per lap)")
    ax.set_title("Teammate gaps by race")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, out)


def raw_vs_within(laps: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> Path:
    """Left: raw lap times by rating (confounded). Right: gap to a Platinum teammate
    in the same car (fair comparison)."""
    d = laps.dropna(subset=["rating"])
    tiers = [t for t in ["Platinum", "Gold", "Silver", "Bronze"] if (d["rating"] == t).any()]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    data = [d.loc[d["rating"] == t, "pct"] for t in tiers]
    bp = axes[0].boxplot(data, tick_labels=tiers, showfliers=False, patch_artist=True)
    for patch, t in zip(bp["boxes"], tiers):
        patch.set_facecolor(TIER_COLORS[t]); patch.set_alpha(0.6)
    axes[0].set_ylabel("% slower than race median lap")
    axes[0].set_title("Raw: all laps by rating")

    low = [t for t in tiers if t != "Platinum"]
    data2 = [pairs.loc[pairs["rating"] == t, "gap_to_platinum_s"] for t in low]
    for i, (t, v) in enumerate(zip(low, data2), start=1):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(v))
        axes[1].scatter(np.full(len(v), i) + jitter, v, s=14, alpha=0.6, color=TIER_COLORS[t])
        if len(v):
            axes[1].text(i, v.max() + 0.1, f"n={len(v)}", ha="center", fontsize=8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(range(1, len(low) + 1), low)
    axes[1].set_ylabel("s per lap slower than Platinum teammate")
    axes[1].set_title("Same car: vs Platinum teammate")
    return _save(fig, out)


def ablation_plot(abl: pd.DataFrame, out: Path) -> Path:
    tiers = list(dict.fromkeys(abl["rating"]))
    models = list(dict.fromkeys(abl["model"]))
    fig, axes = plt.subplots(1, len(tiers), figsize=(3.6 * len(tiers) + 1.6, 3.4), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], tiers):
        d = abl[abl["rating"] == t].set_index("model").reindex(models)
        y = np.arange(len(models))[::-1]
        ax.errorbar(d["effect"], y, xerr=[d["effect"] - d["ci_low"], d["ci_high"] - d["effect"]],
                    fmt="o", color=TIER_COLORS.get(t, ACCENT), capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y, models)
        ax.set_xlabel("Seconds per lap vs Platinum")
        ax.set_title(f"{t}: effect as controls are added")
    return _save(fig, out)


def placebo_plot(fake: pd.DataFrame, real: pd.Series, out: Path) -> Path:
    tiers = [t for t in real.index if t in fake]
    fig, axes = plt.subplots(1, len(tiers), figsize=(4.6 * len(tiers), 3.2), squeeze=False)
    for ax, t in zip(axes[0], tiers):
        ax.hist(fake[t].dropna(), bins=30, color=GREY, alpha=0.8)
        ax.axvline(real[t], color=ACCENT, lw=2, label=f"real = {real[t]:.3f} s")
        ax.set_xlabel("Effect with shuffled ratings (s per lap)")
        ax.set_title(f"Placebo test: {t}")
        ax.legend(frameon=False)
    return _save(fig, out)


def loro_plot(loro: pd.DataFrame, out: Path) -> Path:
    cols = [c for c in loro.columns if c.startswith("effect_")]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for c in cols:
        t = c.removeprefix("effect_")
        ax.plot(range(len(loro)), loro[c], "o-", ms=3, label=t, color=TIER_COLORS.get(t, ACCENT))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(loro)), loro["held_out_race"], rotation=90, fontsize=7)
    ax.set_ylabel("Effect when this race is left out (s)")
    ax.set_title("Leave-one-race-out: is the effect stable?")
    ax.legend(frameon=False)
    return _save(fig, out)
