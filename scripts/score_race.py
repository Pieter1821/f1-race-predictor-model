"""Score a saved prediction against the real result and append it to predictions/scorecard.csv.

    python scripts/score_race.py --season 2026 --round 18              # latest live prediction, result from Jolpica
    python scripts/score_race.py --season 2025 --round 19 --backtest   # latest backtest, result from the snapshot
    python scripts/score_race.py --season 2026 --round 18 --file predictions/2026_R18_variant_....csv  # a variant
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from f1pred.config import PREDICTIONS_DIR, SNAPSHOT
from f1pred.data import NON_STARTERS, fetch_season_results, read_snapshot
from f1pred.validate import score_race

SCORECARD = PREDICTIONS_DIR / "scorecard.csv"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--season", type=int, required=True)
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--backtest", action="store_true")
parser.add_argument("--file", type=Path, help="a specific prediction CSV (default: the latest for this race)")
args = parser.parse_args()

folder = PREDICTIONS_DIR / "backtests" if args.backtest else PREDICTIONS_DIR
# Default = the latest OFFICIAL prediction; labelled variants are only scored when passed with --file
files = sorted(f for f in folder.glob(f"{args.season}_R{args.round}_*.csv") if "variant" not in f.name)
path = args.file or (files[-1] if files else None)
if path is None:
    raise SystemExit(f"No saved prediction for {args.season} R{args.round} in {folder}")
pred = pd.read_csv(path)

if args.backtest:
    res = read_snapshot(SNAPSHOT).results
    res = res[(res["season"] == args.season) & (res["round"] == args.round)]
else:
    res = fetch_season_results(args.season)
    res = res[res["round"] == args.round]
if res.empty:
    raise SystemExit(f"No result for {args.season} R{args.round} yet; try again once Jolpica has published it.")

res = res[~res["status"].isin(NON_STARTERS)].copy()
res["position"] = res["position"].rank(method="first").astype(int)
df = pred.merge(res[["driver_id", "position", "status"]], on="driver_id", how="inner", validate="one_to_one")
missing = set(pred["driver_id"]) - set(df["driver_id"])
if missing:
    print(f"note: predicted but did not start: {sorted(missing)} (excluded from scoring)")

pre_race = pd.Timestamp(pred["generated_at_utc"].iloc[0]) < pd.Timestamp(pred["as_of_utc"].iloc[0])
model = score_race(df, "predicted_position")
grid = score_race(df, "grid")
winner = df[df["position"] == 1].iloc[0]
brier = float(np.mean([((df[c] - (df["position"] <= k)) ** 2).mean()
                       for c, k in (("p_win", 1), ("p_podium", 3), ("p_points", 10))]) * 3)
row = {
    "season": args.season, "round": args.round, "race_name": pred["race_name"].iloc[0],
    "prediction_file": path.name, "official": "variant" not in path.name,
    "made_before_lights_out": pre_race, "backtest": args.backtest,
    "scored_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    **{f"model_{k}": round(v, 4) for k, v in model.items()},
    **{f"grid_{k}": round(v, 4) for k, v in grid.items()},
    "actual_winner": winner["code"], "p_actual_winner": winner["p_win"],
    "winner_log_loss": round(float(-np.log(max(winner["p_win"], 1e-4))), 4), "brier_total": round(brier, 4),
}

card = pd.read_csv(SCORECARD) if SCORECARD.exists() else pd.DataFrame()
card = pd.concat([card[card.get("prediction_file", pd.Series(dtype=str)) != path.name], pd.DataFrame([row])],
                 ignore_index=True)
card.to_csv(SCORECARD, index=False)

print(f"{row['race_name']} {args.season} | {path.name} | made before lights out: {pre_race}")
print(f"  model: MAE {model['mae']:.2f}  Spearman {model['spearman']:.3f}  winner {model['winner_hit']:.0f}  podium {model['podium_hit']:.2f}")
print(f"  grid:  MAE {grid['mae']:.2f}  Spearman {grid['spearman']:.3f}  winner {grid['winner_hit']:.0f}  podium {grid['podium_hit']:.2f}")
print(f"  winner {row['actual_winner']} was given {row['p_actual_winner']:.1%} | brier_total {brier:.4f}")
print(f"  appended to {SCORECARD.name} ({len(card)} rows)")
