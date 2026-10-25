"""Try ONE setting change on dev seasons only: before vs after vs the grid, with a bootstrap CI.

    python scripts/try_setting.py SHRINK_TRACK=10 --season 2026 --round 19
    python scripts/try_setting.py FEATURES=-track_level          # drop a feature (+name adds one)
    python scripts/try_setting.py SIM_POLE_BONUS=0.03            # simulation settings score the simulation

Rules for accepting a change (the script enforces the first three):
  1. judge on dev seasons only (config.DEV_SEASONS / SIM_DEV_SEASONS), never on test or scorecard races
  2. one change at a time
  3. keep it only if the 95% CI of the per-race gain excludes zero
  4. on a tie, keep the simpler version
  5. re-tune at most once a season, then update config.py AND the record in reports/ together
Nothing is written: config.py is changed in memory for this run only.
"""
import argparse
import ast

import pandas as pd

from f1pred import config
from f1pred.backtest import walk_forward
from f1pred.experiments import bootstrap_ci
from f1pred.features import build_features
from f1pred.models import add_finisher_target, final_model, final_sim_model
from f1pred.pipeline import history_table
from f1pred.simulate import add_dnf_probability, settings_from_config, simulate_many, temper_win
from f1pred.validate import score_many

LOCKED = {"DEV_SEASONS", "SIM_DEV_SEASONS", "TEST_SEASONS"}
SIM_SETTINGS = {"SIM_SPACING", "SIM_SIGMA", "SIM_GRID_BLEND", "SIM_POLE_BONUS", "SIM_WIN_TEMPERATURE",
                "DNF_PRIOR_WEIGHT", "N_SIMS", "SIM_TARGET"}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("change", help="NAME=VALUE (exactly one)")
parser.add_argument("--season", type=int, required=True, help="history up to this race's start")
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--sim", action="store_true", help="also score the simulation (automatic for SIM_* settings)")
args = parser.parse_args()

name, _, raw = args.change.partition("=")
if not hasattr(config, name) or name in LOCKED or not raw:
    raise SystemExit(f"{name!r}: unknown, locked, or no value. Validation periods can't be changed here.")
old = getattr(config, name)
if name == "FEATURES" and raw[0] in "+-":
    new = [f for f in old if f != raw[1:]] if raw[0] == "-" else old + [raw[1:]]
else:
    new = ast.literal_eval(raw) if raw not in ("None",) else None
    new = raw if isinstance(old, str) and not isinstance(new, str) else new
score_sim = args.sim or name in SIM_SETTINGS

_, _, history, _ = history_table(args.season, args.round)  # features are rebuilt per run below: feature settings change them


def evaluate() -> dict:
    table = add_dnf_probability(add_finisher_target(build_features(history.assign(is_target=False))))
    preds = walk_forward(table, final_model, config.FEATURES, config.DEV_SEASONS, weighting=config.RECENCY_WEIGHTING)
    out = {"order": score_many(preds, "pred").set_index(["season", "round"]),
           "grid": score_many(preds.assign(pred=preds["grid"]), "pred").set_index(["season", "round"])}
    if score_sim:
        p = walk_forward(table, final_sim_model, config.FEATURES, config.SIM_DEV_SEASONS, target=config.SIM_TARGET)
        sim = temper_win(simulate_many(p, settings_from_config(n_sims=3000)), config.SIM_WIN_TEMPERATURE)
        err = sum((sim[c] - (sim["position"] <= k)) ** 2 for c, k in (("p_win", 1), ("p_podium", 3), ("p_points", 10)))
        out["sim"] = err.groupby([sim["season"], sim["round"]]).mean().rename("brier_total").to_frame()
        pole = sim[sim["grid"] == 1]
        out["pole"] = (pole["p_win"].mean(), (pole["position"] == 1).mean())
    return out


def verdict(diff: pd.Series, lower_is_better: bool = True) -> str:
    lo, hi = bootstrap_ci(diff)
    better = (diff < 0).sum() if lower_is_better else (diff > 0).sum()
    worse = (diff > 0).sum() if lower_is_better else (diff < 0).sum()
    beats = hi < 0 if lower_is_better else lo > 0
    hurts = lo > 0 if lower_is_better else hi < 0
    call = "BEATS THE NOISE: candidate to keep" if beats else ("WORSE beyond noise: reject" if hurts else
                                                              "within noise: keep the current setting")
    return f"{diff.mean():+.4f} per race, 95% CI [{lo:+.4f}, {hi:+.4f}], better {better} / worse {worse} -> {call}"


print(f"{name}: {old!r} -> {new!r} | order on {config.DEV_SEASONS}" +
      (f", simulation on {config.SIM_DEV_SEASONS}" if score_sim else ""), flush=True)
before = evaluate()
setattr(config, name, new)
after = evaluate()

o_b, o_a, g = before["order"], after["order"], before["grid"]
print(f"\nORDER MODEL ({len(o_b)} dev races)        MAE     Spearman  winner  podium")
for label, s in (("grid baseline", g), ("before", o_b), ("after", o_a)):
    print(f"  {label:<30} {s['mae'].mean():.3f}   {s['spearman'].mean():.3f}     "
          f"{s['winner_hit'].mean():.0%}     {s['podium_hit'].mean():.0%}")
print("  MAE change: " + verdict(o_a["mae"] - o_b["mae"]))
if score_sim:
    b, a = before["sim"]["brier_total"], after["sim"]["brier_total"]
    print(f"\nSIMULATION ({len(b)} dev races)   brier_total before {b.mean():.4f}  after {a.mean():.4f}")
    print(f"  polesitter P(win) predicted vs actual: before {before['pole'][0]:.1%} vs {before['pole'][1]:.1%} | "
          f"after {after['pole'][0]:.1%}")
    print("  brier change: " + verdict(a - b))
print("\nNothing was written. To adopt: change config.py AND the record in reports/, then re-run the tests.")
