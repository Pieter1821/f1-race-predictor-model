"""Stage 1: how good is 'finishing position = grid position'?"""
import pandas as pd

from f1pred.config import PROCESSED_DIR
from f1pred.data import load_results
from f1pred.validate import score_many

pd.set_option("display.width", 140)

results = load_results(snapshot=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
results.to_parquet(PROCESSED_DIR / "results.parquet", index=False)
print(f"{len(results)} driver-results across {results.groupby(['season', 'round']).ngroups} races\n")

print("Status buckets per season:")
print(pd.crosstab(results["status_bucket"], results["season"], margins=True).to_string(), "\n")

print("Disqualified drivers, official vs on-track position:")
dsq = results[results["status_bucket"] == "dsq"]
print(dsq[["season", "round", "race_name", "driver_code", "position", "on_track_position"]].to_string(index=False), "\n")

scores = score_many(results, pred_col="grid")
metrics = ["mae", "spearman", "winner_hit", "podium_hit"]
by_season = scores.groupby("season")[metrics].mean()
by_season.loc["all"] = scores[metrics].mean()
print("Grid baseline per season:")
print(by_season.round(3).to_string())
