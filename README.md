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
a calibration step. Without practice-session data (F1's timing archive needs an F1 TV login since
Oct 2026) a small edge over the grid is the realistic ceiling.

## Quick start

Python 3.12 (on this machine: WSL2 Ubuntu, venv outside the repo).

```bash
python3 -m venv ~/venvs/f1 && ~/venvs/f1/bin/pip install -r requirements.lock && ~/venvs/f1/bin/pip install -e .
python -m pytest -q                                              # 82 tests, offline
python scripts/predict_race.py --season 2026 --round 18          # live: refuses after lights out
python scripts/predict_race.py --season 2025 --round 19 --backtest   # replay a past race
python scripts/score_race.py --season 2026 --round 18            # after the race -> predictions/scorecard.csv
```

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

## Rules (fixed 25 Oct 2026, before the Austin result was known)

**Official predictions.**
- The first prediction saved and pushed before lights out is the official one.
- If the official starting grid changes before lights out (pit-lane start, late penalty), re-run
  `predict_race.py` with corrected inputs. That input-only refresh becomes official.
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
- **Accept a change only if:**
  - the bootstrap CI of the gain excludes zero,
  - it helps in at least 3 of 4 seasons (2023-26),
  - and no guardrail gets clearly worse.
- **Ties** go to the simpler model.
- **Logging:** every experiment, failures included, is logged in `reports/experiments.csv`.
- **The real test** is the scorecard: the rest of 2026 and beyond.

## Tuning guide

Every setting lives in `src/f1pred/config.py`, each with a comment saying what it does and what goes wrong
if it's too low or too high. The values there *are* the locked model: a test checks they match the
recorded decisions in `reports/`, and another rebuilds the official Austin prediction from its snapshot
and fails if a single probability changes.

Try one change on dev seasons only:

```bash
python scripts/try_setting.py FEATURES=-track_level      # drop a feature
python scripts/try_setting.py SHRINK_TRACK=10            # change a shrinkage weight
python scripts/try_setting.py SIM_POLE_BONUS=0.02        # simulation settings also score the simulation
```

It prints before vs after vs the grid, race by race, with a bootstrap 95% CI, and writes nothing.

Rules for accepting a change:
1. Judge on dev seasons only, never on test seasons or scorecard races.
2. One change at a time.
3. Keep it only if the CI of the per-race gain excludes zero.
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
src/f1pred/      data, features, models, backtest, simulate, pipeline, validate, weather, weights
scripts/         stage01..07 (the learning journey), predict_race, score_race, check_outliers, build_snapshot
tests/           82 tests incl. leakage, as-of, row-order and negative tests
data/snapshots/  immutable API snapshots      data/manual/  hand-entered pre-race grids
predictions/     timestamped pre-race predictions + scorecard.csv
reports/         locked model and simulation settings, figures, recorded test results
```
