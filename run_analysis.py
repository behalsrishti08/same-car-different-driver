"""Run the whole project from raw laps to figures, tables and a results summary.

    python run_analysis.py            # full run (1,000 bootstrap resamples)
    python run_analysis.py --quick    # fast check (50 resamples, 50 placebo shuffles)

Outputs go to outputs/: tables/*.csv, figures/*.png and results.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scd import data, evaluate, gaps, model, plots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=ROOT / "data" / "raw", type=Path, help="folder with laps_imsa_*.csv")
    ap.add_argument("--out", default=ROOT / "outputs", type=Path)
    ap.add_argument("--boot", default=1000, type=int, help="bootstrap resamples of races")
    ap.add_argument("--placebo", default=500, type=int, help="placebo shuffles")
    ap.add_argument("--quick", action="store_true", help="50 resamples / 50 shuffles, for testing")
    args = ap.parse_args()
    if args.quick:
        args.boot, args.placebo = 50, 50

    tables, figs = args.out / "tables", args.out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("Loading and cleaning laps")
    laps, audit = data.prepare(args.raw, ROOT / "data" / "driver_aliases.csv", ROOT / "data" / "driver_ratings.csv")
    audit.to_csv(tables / "01_cleaning_audit.csv", index=False)
    info = data.describe(laps)
    ident = model.identification(laps)

    step("Experiment 1 & 2: raw and teammate comparisons")
    raw = gaps.raw_by_rating(laps)
    raw.to_csv(tables / "02_raw_by_rating.csv", index=False)
    med = gaps.driver_medians(laps)
    tg = gaps.teammate_gaps(med)
    tg.to_csv(tables / "03_teammate_gaps.csv", index=False)
    gsum = gaps.gap_summary(tg)
    by_race = gaps.gaps_by_race(tg)
    by_race.to_csv(tables / "04_gaps_by_race.csv", index=False)
    split = gaps.car_vs_driver_split(med)
    pairs = gaps.pairwise_vs_platinum(med)
    pairs.to_csv(tables / "05_pairs_vs_platinum.csv", index=False)
    psum = gaps.pairwise_summary(pairs)
    psum.to_csv(tables / "06_pairs_summary.csv", index=False)
    tyre = gaps.tyre_slope(laps)

    step("Experiment 3: models with controls added in blocks")
    abl = model.ablation(laps, y="lap_time")
    abl.to_csv(tables / "07_ablation_seconds.csv", index=False)
    abl_pct = model.ablation(laps, y="pct")
    abl_pct.to_csv(tables / "08_ablation_percent.csv", index=False)
    main_fit, tiers = model.fit_mixed(model.rated(laps))
    real = main_fit.fe_params[tiers]

    step(f"Bootstrap: {args.boot} race resamples")
    boot = evaluate.bootstrap(laps, n=args.boot)
    boot.to_csv(tables / "09_bootstrap_draws.csv", index=False)
    bci = evaluate.bootstrap_ci(boot)
    bci.to_csv(tables / "10_bootstrap_ci.csv", index=False)

    step("Leave-one-race-out")
    loro = evaluate.leave_one_race_out(laps)
    loro.to_csv(tables / "11_leave_one_race_out.csv", index=False)

    step(f"Placebo: {args.placebo} shuffles")
    fake = evaluate.placebo(laps, n=args.placebo)
    m3, _ = model.fit_ols(model.rated(laps), "lap_time", "car_race", model.CONTROLS)
    pv = evaluate.placebo_pvalues(m3.params[tiers], fake)
    pv.to_csv(tables / "12_placebo.csv", index=False)

    step("Figures")
    plots.cleaning_funnel(audit, figs / "fig1_cleaning_funnel.png")
    plots.gaps_by_race(by_race, figs / "fig2_gaps_by_race.png")
    plots.raw_vs_within(laps, pairs, figs / "fig3_raw_vs_same_car.png")
    plots.ablation_plot(abl, figs / "fig4_ablation.png")
    plots.placebo_plot(fake, m3.params[tiers], figs / "fig5_placebo.png")
    plots.loro_plot(loro, figs / "fig6_leave_one_race_out.png")
    plots.hero(abl, bci, figs / "fig0_headline.png")

    step("Writing results.md")
    write_results(args, info, ident, audit, raw, gsum, split, tyre, psum, abl, abl_pct, bci, loro, pv, real)
    (args.out / "summary.json").write_text(json.dumps(
        {"data": info, "identification": ident, "teammate_gaps": gsum, "car_vs_driver": split,
         "tyre_s_per_lap": tyre, "main_effects_s": real.round(4).to_dict()}, indent=2, default=str))
    step(f"Done. See {args.out / 'results.md'}")


def md_table(df) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_results(args, info, ident, audit, raw, gsum, split, tyre, psum, abl, abl_pct, bci, loro, pv, real):
    n_silver_drivers = ident["drivers_by_rating"].get("Silver", 0)
    lines = [
        "# Results: Same Car, Different Driver",
        "",
        "_Generated by `run_analysis.py`. Numbers update every time you re-run it._",
        f"_Bootstrap resamples: {args.boot} · placebo shuffles: {args.placebo}_",
        "",
        "## Data",
        f"- {info['laps']:,} clean laps · {info['races']} races · {info['car_races']} car-races · "
        f"{info['drivers']} drivers ({info['drivers_with_rating']} with a rating)",
        f"- Car-races with two or more rating tiers (what the same-car comparison uses): "
        f"{ident['car_races_with_2plus_tiers']} of {ident['car_races_total']}",
        f"- Drivers by rating: {ident['drivers_by_rating']}",
        f"- Laps by rating: {ident['laps_by_rating']}",
        f"- Drivers without a rating (left out of the models): {', '.join(ident['unrated_drivers']) or 'none'}",
        "",
        md_table(audit),
        "",
        "## Experiment 1: raw comparison (confounded baseline)",
        md_table(raw),
        "",
        "## Experiment 2: teammates in the same car",
        f"- Gap to fastest teammate: {gsum['n_gaps']} gaps, median {gsum['median_s']:.3f} s per lap, "
        f"middle half {gsum['iqr_low_s']:.2f}-{gsum['iqr_high_s']:.2f} s, "
        f"{gsum['share_under_0_5s']:.0%} under 0.5 s",
        f"- Within each race, {split['between_car_share']:.0%} of the pace spread is between cars and "
        f"{split['within_car_share']:.0%} between teammates",
        f"- Tyres: each extra lap on a set adds about {tyre:.4f} s",
        "",
        "Lower-rated drivers vs their Platinum teammate in the same car:",
        "",
        md_table(psum),
        "",
        "## Experiment 3: effect as controls are added (seconds per lap vs Platinum)",
        md_table(abl),
        "",
        "Same models with lap time as % of the race median (robustness):",
        "",
        md_table(abl_pct),
        "",
        "## Evaluation",
        "Bootstrap (resampling whole races) for the main mixed model:",
        "",
        md_table(bci),
        "",
        "Leave-one-race-out (predicting each lap's gap to its own car's average in an unseen race):",
        "",
        f"- Model MAE {loro['mae_model'].mean():.3f} s vs baseline (everyone equal) {loro['mae_baseline'].mean():.3f} s",
        f"- Model RMSE {loro['rmse_model'].mean():.3f} s vs baseline {loro['rmse_baseline'].mean():.3f} s",
        f"- Mean held-out R²: {loro['r2_model'].mean():.4f}",
    ]
    for c in [c for c in loro.columns if c.startswith("effect_")]:
        lines.append(f"- {c.removeprefix('effect_')} effect across folds: "
                     f"{loro[c].min():.3f} to {loro[c].max():.3f} s")
    lines += ["", "Placebo test (ratings shuffled between drivers):", "", md_table(pv), "",
              "## Main answer (mixed model, seconds per lap slower than Platinum)", ""]
    for t, v in real.items():
        row = bci.set_index("rating").loc[t] if t in set(bci["rating"]) else None
        ci = f" (95% bootstrap CI {row['ci_low']:.3f} to {row['ci_high']:.3f})" if row is not None else ""
        lines.append(f"- **{t}: {v:+.3f} s per lap**{ci}")
    lines += ["", "## Read this before quoting the numbers", ""]
    if n_silver_drivers <= 2:
        lines.append(f"- Silver is based on only {n_silver_drivers} driver(s). That is one person's pace, "
                     "not a property of the Silver tier. Report it as a case study, not a tier effect.")
    if ident["drivers_by_rating"].get("Bronze", 0) == 0:
        lines.append("- No Bronze drivers have laps in this data (driver names exist only for the top class).")
    lines.append("- Ratings come from current FIA lists. A driver's category can change between seasons, "
                 "so check each driver's rating for each year before the final report.")
    lines.append("- Observational data: teams choose who drives when, so this is an association, not proof of cause.")
    (args.out / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
