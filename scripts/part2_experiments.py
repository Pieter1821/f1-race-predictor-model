"""Part 2, experiments 3-6 on walk-forward 2023-26, each one change vs v1, logged to reports/experiments.csv.

    python scripts/part2_experiments.py
  3. ensemble: nested ridge/lightgbm weight; adding LambdaRank
  4. sprint results as features
  5. simulation: win calibration fixed vs re-fitted on earlier races (expanding / last 12 races)
  6. recency weighting candidates
"""
import json

import numpy as np
import pandas as pd

from f1pred import config, experiments
from f1pred.backtest import race_key, walk_forward
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.evaluate import SEASONS, compare, line, order_scores
from f1pred.features import SPRINT_FEATURES, add_sprint_features, build_features
from f1pred.models import (Ensemble, LGBMRankerModel, LGBMRegressorModel, RidgeModel, add_finisher_target,
                           final_sim_model)
from f1pred.simulate import add_dnf_probability, settings_from_config, simulate_many, temper_win
from f1pred.validate import score_many
from f1pred.weights import CANDIDATES

SPRINTS = sorted((config.DATA_DIR / "snapshots").glob("jolpica_sprints_*"))[-1]
data = read_snapshot(config.SNAPSHOT)
history = load_results(as_of=race_start(data.schedule, config.TARGET_SEASON, config.TARGET_ROUND), data=data)
table = add_dnf_probability(add_finisher_target(build_features(history.assign(is_target=False))))
table = add_sprint_features(table, pd.read_csv(SPRINTS / "sprints.csv"))
F = list(config.FEATURES)
params = json.loads(config.LGBM_PARAMS_FILE.read_text())["params"]
base = order_scores(table, F)
print(f"v1 walk-forward 2023-26: MAE {base['mae'].mean():.3f}\n", flush=True)

# ---------------------------------------------------------------- 3. ensemble
print("3. Ensemble")
ridge_p = walk_forward(table, RidgeModel, F, SEASONS)
lgbm_p = walk_forward(table, lambda: LGBMRegressorModel(residual=True, **params), F, SEASONS)
rk = lambda p: p["pred"].groupby([p["season"], p["round"]]).rank()
both = ridge_p.assign(r_ridge=rk(ridge_p).values, r_lgbm=rk(lgbm_p).values)
keys = race_key(both)


def mae_with(w, rows):
    return score_many(rows.assign(pred=w * rows["r_ridge"] + (1 - w) * rows["r_lgbm"]), "pred")["mae"].mean()


# Nested: before each half-season, pick the ridge weight that worked best on all earlier test races
WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
chosen, parts = {}, []
for season in SEASONS:
    for half, rounds in (("H1", range(1, 12)), ("H2", range(12, 30))):
        block = both[(both["season"] == season) & both["round"].isin(rounds)]
        if block.empty:
            continue
        earlier = both[keys < race_key(block).min()]
        w = 0.5 if earlier.empty else min(WEIGHTS, key=lambda w: mae_with(w, earlier))
        chosen[f"{season}{half}"] = w
        parts.append(block.assign(pred=w * block["r_ridge"] + (1 - w) * block["r_lgbm"]))
nested = score_many(pd.concat(parts), "pred").set_index(["season", "round"])
print(f"  ridge weight chosen per half-season (earlier races only): {chosen}")
print("  " + line(compare(base, nested, "nested weighted ensemble (ridge vs lightgbm)")), flush=True)
three = order_scores(table, F, make=lambda: Ensemble([RidgeModel(), LGBMRegressorModel(residual=True, **params),
                                                      LGBMRankerModel()]))
print("  " + line(compare(base, three, "ensemble + LambdaRank (3-way rank average)")), flush=True)

# ---------------------------------------------------------------- 4. sprint features
print("\n4. Sprint results (2023+ sprint weekends)")
sprint = order_scores(table, F + SPRINT_FEATURES)
r = compare(base, sprint, "add sprint features")
print("  " + line(r))
sw = pd.concat([base.assign(v="v1"), sprint.assign(v="sprint")]).reset_index()
sw = sw.merge(table[["season", "round", "has_sprint"]].drop_duplicates(["season", "round", "has_sprint"])
              .groupby(["season", "round"])["has_sprint"].max().reset_index(), on=["season", "round"])
print("  on sprint weekends only (MAE):", sw[sw["has_sprint"] == 1].groupby("v")["mae"].mean().round(3).to_dict(),
      "| other weekends:", sw[sw["has_sprint"] == 0].groupby("v")["mae"].mean().round(3).to_dict(), flush=True)

# ---------------------------------------------------------------- 5. simulation: win calibration
print("\n5. Simulation win calibration (walk-forward 2023-26 pace scores, 3000 sims per race)")
pace = walk_forward(table, final_sim_model, F, SEASONS, target=config.SIM_TARGET)
raw = simulate_many(pace, settings_from_config(n_sims=3000))
rkeys = race_key(raw)
A = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]


def win_brier(sim):
    return float(((sim["p_win"] - (sim["position"] == 1)) ** 2).mean())


def per_race(sim):
    won = sim["position"] == 1
    return pd.DataFrame({
        "win_brier": ((sim["p_win"] - won) ** 2).groupby([sim["season"], sim["round"]]).mean(),
        "log_loss": (-np.log(sim.loc[won, "p_win"].clip(1e-4))).groupby([sim.loc[won, "season"], sim.loc[won, "round"]]).mean(),
    })


def adaptive(window):
    out = []
    for k, race in raw.groupby(rkeys):
        earlier = raw[rkeys < k]
        if window:
            earlier = earlier[race_key(earlier).isin(sorted(race_key(earlier).unique())[-window:])]
        a = config.SIM_WIN_TEMPERATURE if race_key(earlier).nunique() < 10 else min(A, key=lambda a: win_brier(temper_win(earlier, a)))
        out.append(temper_win(race, a).assign(a_used=a))
    return pd.concat(out)


fixed = per_race(temper_win(raw, config.SIM_WIN_TEMPERATURE))
for name, sim in (("re-fitted on all earlier races", adaptive(None)), ("re-fitted on the last 12 races", adaptive(12))):
    pr = per_race(sim)
    d = pr["win_brier"] - fixed["win_brier"]
    lo, hi = experiments.bootstrap_ci(d)
    seasons_better = int((d.groupby(level=0).mean() < 0).sum())
    dl = pr["log_loss"] - fixed["log_loss"]
    pole = sim[(sim["grid"] == 1) & (sim["season"] == 2026)]
    ok = hi < 0 and seasons_better >= 3 and experiments.bootstrap_ci(dl)[0] <= 0
    decision = "ACCEPT" if ok else "reject"
    experiments.log(f"win calibration {name}", SEASONS, {"win_brier_change": d.mean(), "log_loss_change": dl.mean(),
                                                        "better_seasons": seasons_better}, (lo, hi), decision)
    print(f"  {name:<34} win Brier {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}] better in {seasons_better}/4 | "
          f"log loss {dl.mean():+.3f} | 2026 pole P(win) {pole['p_win'].mean():.1%} vs actual "
          f"{(pole['position'] == 1).mean():.1%} | a used {sim['a_used'].min()}-{sim['a_used'].max()} -> {decision}", flush=True)

# ---------------------------------------------------------------- 6. recency weighting
print("\n6. Recency weighting (training rows weighted by race age)")
for name, w in CANDIDATES.items():
    if name == "uniform":
        continue
    print("  " + line(compare(base, order_scores(table, F, weighting=w), f"recency weights: {name}")), flush=True)
