"""Stage 1: how good is 'finishing position = grid position'?

    python scripts/stage01_baseline.py                      # from the committed snapshot
    python scripts/stage01_baseline.py --api                # fresh from the API
    python scripts/stage01_baseline.py --api --snapshot     # fresh, and save a new snapshot
"""
import argparse

import pandas as pd

from f1pred.config import PROCESSED_DIR, SNAPSHOT, TARGET_ROUND, TARGET_SEASON
from f1pred.data import load_raw, load_results, race_start
from f1pred.validate import score_many

parser = argparse.ArgumentParser()
parser.add_argument("--api", action="store_true", help="fetch from Jolpica instead of the snapshot")
parser.add_argument("--snapshot", action="store_true", help="with --api: save what was fetched")
args = parser.parse_args()

pd.set_option("display.width", 140)

data = load_raw(from_snapshot=None if args.api else SNAPSHOT, save_new_snapshot=args.snapshot)
# Everything before the target race's lights out, and nothing after
results = load_results(as_of=race_start(data.schedule, TARGET_SEASON, TARGET_ROUND), data=data)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
results.to_parquet(PROCESSED_DIR / "results.parquet", index=False)
print(f"source: {results.attrs['source']} | as of: {results.attrs['as_of']}")
print(f"{len(results)} driver-results across {results.groupby(['season', 'round']).ngroups} races\n")

print("Status buckets per season:")
print(pd.crosstab(results["status_bucket"], results["season"], margins=True).to_string(), "\n")

print("Disqualified drivers, official vs on-track position:")
dsq = results[results["status_bucket"] == "dsq"]
print(dsq[["season", "round", "race_name", "driver_code", "laps", "position", "on_track_position"]].to_string(index=False), "\n")

scores = score_many(results, pred_col="grid")
metrics = ["mae", "spearman", "winner_hit", "podium_hit"]
by_season = scores.groupby("season")[metrics].mean()
by_season.loc["all"] = scores[metrics].mean()
print("Grid baseline per season:")
print(by_season.round(3).to_string())
