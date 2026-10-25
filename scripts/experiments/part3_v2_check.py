"""Part 3: v2 (v1 minus driver_vs_teammate) vs v1 vs grid, walk-forward 2023-26, with simulation guardrails.

    python scripts/experiments/part3_v2_check.py
Adopt v2 from the next race only if it ties or wins: order MAE not measurably worse and no guardrail
(Spearman, winner, podium, Brier total, winner log loss) clearly worse.
"""
import numpy as np
import pandas as pd

from f1pred import config, experiments
from f1pred.backtest import walk_forward
from f1pred.evaluate import SEASONS, compare, line, order_scores
from f1pred.models import GridBaseline, final_sim_model
from f1pred.pipeline import history_table
from f1pred.simulate import settings_from_config, simulate_many, temper_win

V1 = list(config.FEATURES)
V2 = [f for f in V1 if f != "driver_vs_teammate"]
_, _, _, table = history_table(config.TARGET_SEASON, config.TARGET_ROUND)  # recorded for Austin 2026

o1, o2 = order_scores(table, V1), order_scores(table, V2)
grid = order_scores(table, V1, make=GridBaseline)
print("ORDER MODEL, walk-forward 2023-26 (87 races)      MAE    Spearman  winner  podium")
for name, s in (("grid", grid), ("v1", o1), ("v2", o2)):
    print(f"  {name:<46} {s['mae'].mean():.3f}  {s['spearman'].mean():.3f}     {s['winner_hit'].mean():.0%}     "
          f"{s['podium_hit'].mean():.0%}")
r = compare(o1, o2, "Part 3: v2 (v1 minus driver_vs_teammate) order model", simplification=True)
print("  " + line(r))
for s, name in ((o1, "v1"), (o2, "v2")):
    d = s["mae"] - grid["mae"]
    lo, hi = experiments.bootstrap_ci(d)
    print(f"  {name} vs grid: {d.mean():+.3f} places per driver [{lo:+.3f}, {hi:+.3f}]")


def sim_per_race(features):
    pace = walk_forward(table, final_sim_model, features, SEASONS, target=config.SIM_TARGET)
    sim = temper_win(simulate_many(pace, settings_from_config(n_sims=3000)), config.SIM_WIN_TEMPERATURE)
    won = sim["position"] == 1
    err = sum((sim[c] - (sim["position"] <= k)) ** 2 for c, k in (("p_win", 1), ("p_podium", 3), ("p_points", 10)))
    return pd.DataFrame({
        "brier_total": err.groupby([sim["season"], sim["round"]]).mean(),
        "log_loss": (-np.log(sim.loc[won, "p_win"].clip(1e-4))).groupby([sim.loc[won, "season"], sim.loc[won, "round"]]).mean(),
    })


print("\nSIMULATION GUARDRAILS, walk-forward 2023-26 (lower is better)")
s1, s2 = sim_per_race(V1), sim_per_race(V2)
worse = []
for m in ("brier_total", "log_loss"):
    d = s2[m] - s1[m]
    lo, hi = experiments.bootstrap_ci(d)
    if lo > 0:
        worse.append(m)
    print(f"  {m:<12} v1 {s1[m].mean():.4f}  v2 {s2[m].mean():.4f}  change {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]"
          f"{'  CLEARLY WORSE' if lo > 0 else ''}")

adopt = r["accepted"] and not worse
decision = "adopted: v2 from the next race" if adopt else "reject: keep v1"
experiments.log("Part 3: v2 full check incl. simulation guardrails", SEASONS,
                {"mae_change": r["mae_change"], "brier_total_change": (s2["brier_total"] - s1["brier_total"]).mean(),
                 "log_loss_change": (s2["log_loss"] - s1["log_loss"]).mean()}, r["ci"], decision)
print(f"\nDECISION: {decision}")
