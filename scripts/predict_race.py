"""Predict any race before it starts.

    python scripts/predict_race.py --season 2026 --round 18             # live: refreshes data, refuses after lights out
    python scripts/predict_race.py --season 2025 --round 19 --backtest  # replay a past race from the snapshot

Writes predictions/{season}_R{round}_{UTC timestamp}.csv (or predictions/backtests/ with --backtest).
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from f1pred.config import FIRST_SEASON, PREDICTIONS_DIR, SNAPSHOT
from f1pred.data import fetch_raw, load_results, race_start, read_snapshot, save_snapshot
from f1pred.pipeline import prediction_table, predict, race_inputs

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--season", type=int, required=True)
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--backtest", action="store_true", help="allow a race that has already started")
parser.add_argument("--snapshot", type=Path, help="use this snapshot instead of fetching fresh data")
parser.add_argument("--n-sims", type=int, default=20_000)
args = parser.parse_args()

seasons = range(FIRST_SEASON, args.season + 1)
if args.snapshot or args.backtest:
    data = read_snapshot(args.snapshot or SNAPSHOT)
    data_source = data.source
else:
    print("Fetching fresh data from Jolpica...", flush=True)
    data = fetch_raw(seasons)
    data_source = f"snapshot {save_snapshot(data, seasons).name}"

as_of = race_start(data.schedule, args.season, args.round)
now = pd.Timestamp(datetime.now(timezone.utc))
if now >= as_of and not args.backtest:
    raise SystemExit(f"{args.season} R{args.round} started at {as_of} (now {now:%Y-%m-%d %H:%M} UTC). "
                     f"A prediction made after lights out isn't a prediction; use --backtest to replay it.")

race = data.schedule.set_index(["season", "round"]).loc[(args.season, args.round)]
history = load_results(as_of=as_of, data=data)
inputs, inputs_source = race_inputs(args.season, args.round, data, history, as_of, args.backtest)
print(f"{race['race_name']} | as of {as_of} | {data_source}\ninputs: {inputs_source}", flush=True)

sim = predict(history, inputs, race["circuit_id"], as_of, n_sims=args.n_sims)
out = prediction_table(sim, race["race_name"], as_of, inputs_source, data_source)

folder = PREDICTIONS_DIR / "backtests" if args.backtest else PREDICTIONS_DIR
folder.mkdir(parents=True, exist_ok=True)
path = folder / f"{args.season}_R{args.round}_{now:%Y%m%dT%H%M%SZ}.csv"
out.to_csv(path, index=False)

print(f"\ntrained on {out['train_races'].iloc[0]} earlier races | saved {path.relative_to(PREDICTIONS_DIR.parent)}\n")
show = out[["predicted_position", "code", "team", "grid", "grid_penalty", "expected_position",
            "p_win", "p_podium", "p_points", "p_dnf"]].copy()
for c in ("p_win", "p_podium", "p_points", "p_dnf"):
    show[c] = (show[c] * 100).round(1).astype(str) + "%"
print(show.to_string(index=False))
