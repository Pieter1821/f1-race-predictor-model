"""Part 2, experiment 1: feature ablation on walk-forward 2023-26.

    python scripts/part2_ablation.py     # logs every run to reports/experiments.csv
"""
import json

import pandas as pd

from f1pred import config
from f1pred.backtest import race_key
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.evaluate import compare, line, order_scores
from f1pred.features import build_features
from f1pred.models import add_finisher_target

data = read_snapshot(config.SNAPSHOT)
history = load_results(as_of=race_start(data.schedule, config.TARGET_SEASON, config.TARGET_ROUND), data=data)
table = add_finisher_target(build_features(history.assign(is_target=False)))
FULL = list(config.FEATURES)
base = order_scores(table, FULL)
print(f"v1 walk-forward 2023-26: MAE {base['mae'].mean():.3f} over {len(base)} races\n", flush=True)

print("1a. Drop one feature at a time (simplification test)")
singles = []
for f in FULL:
    r = compare(base, order_scores(table, [x for x in FULL if x != f]), f"drop {f}", simplification=True)
    singles.append(r)
    print("  " + line(r), flush=True)

print("\n1b. Drop groups")
GROUPS = {"quali extras (teammate delta, no-time flag, sprint flag)": ["teammate_quali_delta", "no_quali_time", "sprint_sets_grid"],
          "late additions (track, dnf rate, vs teammate)": ["track_level", "team_dnf_rate", "driver_vs_teammate"],
          "standings (driver + constructor)": ["driver_standing", "constructor_standing"],
          "duplicate grid (grid_pct)": ["grid_pct"]}
for name, fs in GROUPS.items():
    r = compare(base, order_scores(table, [x for x in FULL if x not in fs]), f"drop group: {name}", simplification=True)
    print("  " + line(r), flush=True)

print("\n1c. Backward elimination: drop the least useful feature while still not measurably worse than the FULL model")
current = list(FULL)
while len(current) > 4:
    trials = {f: order_scores(table, [x for x in current if x != f]) for f in current}
    f_best = min(trials, key=lambda f: trials[f]["mae"].mean())
    r = compare(base, trials[f_best], f"elimination: drop {f_best} ({len(current) - 1} left)", simplification=True)
    print("  " + line(r), flush=True)
    if not r["accepted"]:
        break
    current.remove(f_best)

removed = [f for f in FULL if f not in current]
print(f"\nkept {len(current)} features, removed: {removed}")
(config.PROJECT_ROOT / "reports" / "part2_ablation.json").write_text(json.dumps({"kept": current, "removed": removed}, indent=2))
