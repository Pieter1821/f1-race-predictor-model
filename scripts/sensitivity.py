"""How much does a race's prediction move under reasonable alternative settings? Prints only; saves nothing.

    python scripts/sensitivity.py --season 2026 --round 18 --snapshot data/snapshots/<folder>
"""
import argparse
from pathlib import Path

import pandas as pd

from f1pred import config
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.pipeline import predict, race_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, required=True)
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--snapshot", type=Path, required=True)
args = parser.parse_args()

data = read_snapshot(args.snapshot)
as_of = race_start(data.schedule, args.season, args.round)
history = load_results(as_of=as_of, data=data)
inputs, _ = race_inputs(args.season, args.round, data, history, as_of, backtest=False)
circuit = data.schedule.set_index(["season", "round"]).loc[(args.season, args.round), "circuit_id"]

SCENARIOS = {
    "locked model": {},
    "sigma 0.10": {"SIM_SIGMA": 0.10},
    "sigma 0.16": {"SIM_SIGMA": 0.16},
    "DNF prior 40": {"DNF_PRIOR_WEIGHT": 40},
    "no track_level": {"FEATURES": [f for f in config.FEATURES if f != "track_level"]},
    "win temperature 1.0 (raw)": {"SIM_WIN_TEMPERATURE": 1.0},
}
rows, tops = [], {}
for name, changes in SCENARIOS.items():
    saved = {k: getattr(config, k) for k in changes}
    for k, v in changes.items():
        setattr(config, k, v)
    sim = predict(history, inputs, circuit, as_of).sort_values("predicted_position")
    for k, v in saved.items():
        setattr(config, k, v)
    p = sim.set_index("code")["p_win"]
    top3 = p.sort_values(ascending=False).head(3)
    rows.append({"scenario": name, **{c: f"{p[c]:.1%}" for c in top3.index.union(["ANT", "VER", "LEC"])}})
    tops[name] = " ".join(sim["code"].head(5))
    print(f"done: {name}", flush=True)

pd.set_option("display.width", 160)
table = pd.DataFrame(rows).set_index("scenario").fillna("")
table["top 5 predicted order"] = pd.Series(tops)
print("\n" + table.to_string())
