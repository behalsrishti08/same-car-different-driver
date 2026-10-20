# Same Car, Different Driver

**Do official FIA driver ratings (Platinum, Gold, Silver, Bronze) show up in lap times when the car is the same?**

In endurance racing, two to four drivers share one car in the same race. Comparing teammates holds the car, team, setup, track and weather constant, so whatever gap remains is the driver. This project measures that gap for each rating, lap by lap, using IMSA WeatherTech GTP races from 2023–2026.

**Srishti Behal**

## Main results

All numbers come from `outputs/results.md`. Re-running the analysis regenerates them.

| | |
|---|---|
| Clean laps | 64,001 from 33 races, 345 car-races, 84 drivers |
| Teammate gap (all drivers) | median **0.36 s/lap** (middle half 0.16–0.64 s) |
| Pace spread, within a race | 53% between cars · 47% between teammates (F1 studies: ~88% car) |
| Tyre wear | ~0.006 s per extra lap on a set |
| **Gold vs Platinum, same car** | **+0.03 s/lap** (95% bootstrap CI −0.07 to +0.13; placebo p = 0.39) |
| Silver vs Platinum, same car | +1.17 s/lap, but **one driver, one race** (case study) |

**Interpretation.** Compared within the same race but not the same car, Gold drivers look 0.17 s/lap slower than Platinum. That gap almost disappears once the comparison is between teammates in the **same car**. In the top class, the rating difference between Gold and Platinum is mostly explained by the car the driver is in, not by lap pace.

![Effect as controls are added](outputs/figures/fig4_ablation.png)

## How it works

```
373k laps (Kaggle, all classes)
  → 82,348 top-class laps with driver names
  → 64,001 clean laps (green flag, no pit in/out laps, within 107% of race median)
  → teammate gaps and car-vs-driver split          (Experiments 1–2)
  → models with controls added in blocks           (Experiment 3)
  → bootstrap, leave-one-race-out, placebo         (evaluation)
```

**Model.** Every lap time is measured relative to its own car's average in that race ("demeaning"). This removes everything teammates share. A linear mixed-effects model (statsmodels `MixedLM`) then estimates:

```
lap time vs car average = rating effect + tyre age + race hour + driver's own bit + noise
                          (fixed)         (fixed)    (fixed)     (random)
```

Controls are added one block at a time (M0 raw → M1 same race → M2 same car → M3 + tyres → M4 + driver). This shows how much of the raw gap each factor explains.

**Checks.**

| Check | Method | What it shows |
|---|---|---|
| How big, how sure? | Bootstrap: resample whole races 1,000× | 95% error bars |
| Is it fair? | Baseline vs model (ablation) | The gap shrinks as the car is controlled |
| Does it hold? | Leave-one-race-out (33 folds) | The effect is stable across races |
| Does it fit? | MAE / RMSE / R² on held-out races | vs predicting "every teammate equal" |
| Is it real? | Placebo: shuffle ratings 500× | Fake effects center on zero |

Testing is always done **by race, not by lap**, because laps from one race share weather, track and traffic.

## Run it yourself (Mac)

```bash
# 1. Get the code
git clone https://github.com/behalsrishti08/same-car-different-driver.git
cd same-car-different-driver

# 2. Install Python packages (once)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Put the Kaggle IMSA CSVs in data/raw/  (see data/raw/README.md)

# 4. Run
python run_analysis.py --quick   # ~1 minute, for checking
python run_analysis.py           # full run, ~12 minutes
python -m pytest -q              # tests on made-up data with a known answer
```

## Project layout

```
run_analysis.py          one command: raw laps → tables, figures, results.md
src/scd/
  data.py                load, fix driver names, add ratings, clean laps
  gaps.py                teammate gaps, car-vs-driver split, raw comparison
  model.py               the five models (ablation) and the mixed model
  evaluate.py            bootstrap, leave-one-race-out, placebo
  plots.py               figures
data/
  driver_aliases.csv     same person spelled two ways (e.g. Alex / Alexander Palou)
  driver_ratings.csv     FIA rating for each driver, with source
  raw/                   Kaggle CSVs (not committed)
tests/test_pipeline.py   checks the method recovers a known effect on simulated laps
outputs/                 results.md, tables/, figures/
```

## Limitations

- **Ratings.** The data has no rating column. Ratings were added from current FIA categorisation lists, and 5 drivers are still unrated (see `data/driver_ratings.csv`). Categories can change between seasons.
- **Silver and Bronze.** Driver names exist only for the top class, which is almost all Platinum and Gold. Silver is one driver and Bronze has none, so those tiers can't be estimated here.
- **Observational data.** Teams choose who drives when (e.g. putting the slower driver in easier conditions), which can bias the effect toward zero. Traffic is not observed.
- **Weather** is missing for 46% of laps, so it isn't used as a control.

## Data and references

- Data: Motorsport Strategy Lab, *Derived data*, Kaggle (IMSA files). FIA Driver Categorisation.
- Bell, Smith, Sabel & Jones (2016). Formula for success: multilevel modelling of F1 driver and constructor performance. *J. Quant. Analysis in Sports.*
- van Kesteren & Bergkamp (2023). Bayesian analysis of F1 race results. *J. Quant. Analysis in Sports.*
- Heilmeier, Graf & Lienkamp (2018). A race simulation for strategy decisions in circuit motorsports. *IEEE ITSC.*
- Hernán & Robins (2020). *Causal Inference: What If.*
- Motorsport Strategy Lab, Kaggle (IMSA lap data) · FIA Driver Categorisation.
