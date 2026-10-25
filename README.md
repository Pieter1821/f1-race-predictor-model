# F1 Race Predictor

Predicts the finishing order of a Formula 1 Grand Prix, plus win, podium, points and retirement
probabilities for every driver, using only information available before lights out. Built on
[FastF1](https://github.com/theOehrly/Fast-F1)'s Jolpica (Ergast) client and validated with
walk-forward backtests on 2022-2026 races.

First live prediction: **2026 United States GP**, pushed before the race
(tag [`austin-2026-prediction`](../../tree/austin-2026-prediction)).

## Honest performance

| Walk-forward test, 2025-26 (41 races, never used for choices) | MAE (places) | Spearman | Winner | Podium |
|---|---|---|---|---|
| "Finish = grid" baseline | 3.31 | 0.658 | 68% | 70% |
| This model | **3.21** | **0.681** | 61% | 70% |

About 0.1 places per driver better than the grid: likely real (better in 21 races, worse in 12,
equal in 8) but not proven (95% CI of the gain [-0.21, +0.004] places). The model is best where the grid is most
misleading (penalised drivers: +4.6 places; back-of-grid starters: +0.5) and no better than the grid
at the very front. Probabilities are well calibrated for podium and points; win probabilities needed
a calibration step. Practice-session pace from OpenF1 was tested and added nothing (a tie, below), so a
small edge over the grid is the realistic ceiling with the data available.

**What it's good for.** Not for picking winners: "pole wins" does better. For odds-style questions
("Colapinto's chance of points from P21") its probabilities are calibrated and honest. Against bookmakers
it is untested and probably behind, since odds include practice pace and paddock news. One race settles
nothing either way; the scorecard over many races does.

## Quick start

Python 3.12 (on this machine: WSL2 Ubuntu, venv outside the repo).

```bash
python3 -m venv ~/venvs/f1 && ~/venvs/f1/bin/pip install -r requirements.lock && ~/venvs/f1/bin/pip install -e .
python -m pytest -q                                              # 94 tests, offline (tests/ only)
python scripts/predict_race.py --season 2026 --round 18          # live: refuses after lights out
python scripts/predict_race.py --season 2025 --round 19 --backtest   # replay a past race
python scripts/score_race.py --season 2026 --round 18            # after the race -> predictions/scorecard.csv
python scripts/score_strategy.py --season 2026 --round 18        # the strategy estimate -> strategy_scorecard.csv
python scripts/check_outliers.py --season 2026 --round 19        # data checks before a race
```

Every pipeline script takes `--season/--round`; nothing defaults to a particular race.

Before a live prediction, put the final grid (after penalties) and qualifying in
`data/manual/{season}_R{round}_grid.csv` and `_qualifying.csv`. Jolpica publishes no pre-race
grid, so penalties must be entered by hand; the files are strictly validated.

## How it works

**1. Data, as of lights out.** Results, qualifying, calendar and driver debuts from Jolpica, frozen
in timestamped snapshots under `data/snapshots/` (never edited). Every load is cut off at the race's
start time (`as_of`): later races don't exist as far as the model knows.

**2. Features** (`src/f1pred/features.py`): grid, qualifying gap, grid penalty, team form, driver level,
recent form, track history, standings, team DNF rate. Every one uses earlier races only, tested by
rewriting future results and checking nothing moves. Small samples are **shrunk** toward a sensible
default:

> estimate = (sum + k × prior) / (count + k)

"Pretend you've already seen k races at the default value." Example: Lindblad's level after 16 races
= (16 × 0.464 + 5 × 0.476) / 21 = 0.467, pulled slightly toward his team's form.

**3. Order model** (`src/f1pred/models.py`): the average of two rankings.
- **Ridge regression**: a weighted sum of features with a penalty on large weights (anti-overfitting).
- **Gradient-boosted trees** (LightGBM): many small if/else trees, each fixing the previous ones'
  errors. They predict only the **correction to the grid** (final = grid + correction), because trees
  are poor at drawing the smooth "start further back, finish further back" line.

**4. Walk-forward validation** (`src/f1pred/backtest.py`): to score race k, train on races before k
only. Choices were made on 2024 (and 2023-24 for the simulation), the final score reported on
2025-26 once. Like a CI pipeline that only tests against code the model has never seen.

**5. Monte Carlo simulation** (`src/f1pred/simulate.py`): replay the race 20,000 times. Each run:
pace = model score + noise (σ = 0.12); each car rolls its DNF dice (team rate shrunk hard toward the
season rate); retirements go to the back; sort. P(win) = wins / 20,000. Win probabilities are then
calibrated: p^1.75, renormalised (favourites were under-rated: "27%" drivers won 41%).

**6. Metrics** (`src/f1pred/validate.py`, worked example: 4 drivers, B wins, model predicts B-A-C-D,
grid A-B-C-D, actual B-A-D-C):

| Metric | Formula | Example | Good in F1 |
|---|---|---|---|
| MAE | mean \|predicted − actual position\| | model 0.5, grid 1.0 | random 6.65, grid ≈ 3.3 |
| Spearman | 1 − 6Σd² / (n(n²−1)) | model 0.8, grid 0.6 | random 0, grid ≈ 0.64 |
| Winner / podium hit | P1 correct; top-3 overlap / 3 | 1 vs 0; 2/3 each | pole wins ≈ 60-70% |
| Brier | mean (probability − outcome)² | P(win) 40/45/10/5%, B wins: 0.119 | win ≈ 0.025-0.03 |
| Log loss | −ln P(actual winner) | 45% → 0.80; 5% → 3.0 | uniform 3.09, ours ≈ 1.2 |
| Bootstrap CI | resample races 10,000×, middle 95% of the mean gain | −0.10 [−0.21, +0.004] | excludes 0 = convincing |

## Model versions

| Version | Used for | Change | Clean test, 2025-26 (41 races) |
|---|---|---|---|
| v1 | Austin 2026 (official) | — | MAE 3.21 vs grid 3.31: gain 0.10 places, CI [−0.004, +0.21] |
| **v2** | default from 2026 R19 | drop `driver_vs_teammate` | a tie with v1 (nested gain −0.002, CI [−0.024, +0.020]); kept because simpler |

Over all 87 races of 2023-26, v1 gains 0.19 places (CI [0.09, 0.29]), but that figure is **optimistic**:
2023-24 are the seasons the features, tuning and simulation were chosen on, and keeping whichever option
did best there bakes their luck in, so their gain (about 0.27 places) is roughly three times the clean one.

**Over-/underfitting check** (`scripts/experiments/diagnose.py`): training and walk-forward errors are close (about 3.1
places), more flexible models don't help, and 400 trees overfit. The model is short of *information*, not
capacity.

**Experiments, 25 Oct** (all logged in `reports/experiments.csv`, failures included):
- Rejected:
  - weighted ensemble, and LambdaRank,
  - sprint features,
  - in-season win re-calibration,
  - four recency weightings,
  - a front-row start bonus.
- A nested **pole start bonus** was promising (win Brier −0.0003, log loss −0.010, both CIs just touching 0)
  but rejected for now. Re-test it after the 2026 season.

**Practice data** now comes from [OpenF1](https://openf1.org): free laps, stints and pit stops from 2023 on,
no login, in an immutable snapshot. Cleaned long-run pace and tyre-wear features did *not* improve the
order model: +0.001 when trained on 2023+ data, a tie. They mostly repeat qualifying, and practice fuel
loads are hidden.

## Tyre strategy (experimental)

`scripts/experiments/part5_strategy.py` picks the fastest 1- or 2-stop plan:

> race time = Σ laps (compound offset + wear × tyre age) + stops × pit loss

- Tyre wear comes from this weekend's practice long runs, fuel-corrected, and scaled by a practice→race
  calibration of about 0.36: practice overstates race wear about 2.8×.
- Offsets and the fuel effect come from earlier race laps, and pit loss from earlier races at the same circuit.

**Walk-forward over 56 dry races:**
- The stop count is right 66% of the time, vs 55% for "always one stop" (sign test p = 0.11: suggestive, not proven).
- The starting compound is right only 46% of the time, so trust stop counts and pit windows, not compound order.

Austin 2026 estimate: `predictions/2026_R18_strategy_*.csv`, labelled experimental.

## Rules (fixed 25 Oct 2026, before the Austin result was known)

**Official predictions.**
- The first prediction saved and pushed before lights out is the official one.
- If the official starting grid changes before lights out (pit-lane start, late penalty), re-run
  `predict_race.py` with corrected inputs and the same model version (for Austin 2026: `--model v1`).
  That input-only refresh becomes official.
- A model change never replaces the official prediction for a race. A new model version becomes the
  default from the next race. For Austin 2026, `2026_R18_20261025T083234Z.csv` stays official.
- Variants are scored alongside it, but marked as not official.

**Improving accuracy.**
- **Primary metric:** walk-forward MAE of the predicted order.
- **Guardrails** (must not get clearly worse): Spearman, winner and podium hit rate, Brier total, and
  winner log loss.
- **Nested walk-forward.** 2025-26 has already been looked at, so no untouched past test is left. For
  each test race, every choice (features, settings, tuning, calibration) is made using only earlier
  races, and the choosing is redone once per half-season.
- **A change is a candidate only if:**
  - the bootstrap CI of the gain excludes zero,
  - it helps in at least 3 of 4 seasons (2023-26),
  - and no guardrail gets clearly worse.
- **A simplification** (dropping a feature) is a candidate only if its worst case, the CI's upper end, stays
  within the noise floor (+0.022 places). A wide CI means "not measured precisely", not "not worse".
- **Candidate is not adopted.** A candidate becomes the model only through a separate check that is
  logged as `adopted` (one so far: v2). Ties go to the simpler model.
- **Logging:** every experiment, failures included, is logged in `reports/experiments.csv`.
- **The real test** is the scorecard: the rest of 2026 and beyond.

## Tuning guide

Every setting lives in `src/f1pred/config.py`, each with a comment saying what it does and what goes wrong
if it's too low or too high. The values there *are* the locked model: a test checks they match the
recorded decisions in `reports/`, and another rebuilds the official Austin prediction from its snapshot
and fails if a single probability changes.

Try one change on dev seasons only:

```bash
python scripts/try_setting.py FEATURES=-track_level --season 2026 --round 19   # drop a feature
python scripts/try_setting.py SHRINK_TRACK=10 --season 2026 --round 19         # change a shrinkage weight
python scripts/try_setting.py SIM_POLE_BONUS=0.02 --season 2026 --round 19     # also scores the simulation
```

It prints before vs after vs the grid, race by race, with a bootstrap 95% CI, and writes nothing.

Rules for accepting a change:
1. Judge on dev seasons only, never on test seasons or scorecard races.
2. One change at a time.
3. Keep it only if the CI of the per-race gain excludes zero (a simplification: its worst case within noise).
4. On a tie, keep the simpler version.
5. Re-tune at most once a season, and update `config.py` and the record in `reports/` together.

First experiments (25 Oct 2026):
- **Dropping `track_level`:** within noise (+0.016 MAE, CI [−0.04, +0.09]), so it stays.
- **A pole bonus:** within noise on dev, and dev polesitters were already *over*-rated (60% predicted
  vs 54% actual) while 2025-26 ones were under-rated (~50% vs 67-71%). Pole conversion changes by
  season, so a fixed bonus can't fix it.

The idea to test later is a pole-conversion rate learned as the season goes on.

## Repository layout

```
src/f1pred/          data, features, models, backtest, simulate, pipeline, validate, practice, strategy, openf1
scripts/             the pipeline: predict_race, score_race, score_strategy, export_web_data, check_outliers,
                     build_snapshot, fetch_openf1, try_setting, sensitivity, smoke_test
scripts/experiments/ one-off records of how the model was built and tested (each file's docstring says what it ran)
tests/               94 tests incl. leakage, as-of, row-order, golden-master and negative tests
web/                 Next.js app: race page, results checker, model scorecard (reads web/public/data/*.json)
data/snapshots/      immutable API snapshots          data/manual/  hand-entered pre-race grids
predictions/         timestamped pre-race predictions + scorecards
reports/             locked model and simulation settings, figures, the experiment log
```

## Web app

`web/` shows the pipeline's own numbers; it never recomputes a metric. After any prediction or result:

```bash
python scripts/export_web_data.py --season 2026 --round 18   # writes web/public/data/*.json (+ the result, once published)
cd web && npm install && npm run build && npm run start      # http://localhost:3000
```

Three screens: **Race** (the call, grid-to-flag chart, every driver's chances, pit strategy), **Results**
(predicted against finished for the live race and all 87 backtests, model vs grid per race) and **Model**
(the clean scorecard, calibration, where it wins, every experiment).
