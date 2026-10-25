"""Stage 1: how good is 'finishing position = grid position'?"""
from f1pred.config import PROCESSED_DIR
from f1pred.data import load_results
from f1pred.validate import score_many

results = load_results()
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
results.to_parquet(PROCESSED_DIR / "results.parquet", index=False)
print(f"{len(results)} driver-results across {results.groupby(['season', 'round']).ngroups} races\n")

scores = score_many(results, pred_col="grid")
by_season = scores.groupby("season")[["mae", "spearman", "winner_hit", "podium_hit"]].mean()
by_season.loc["all"] = scores[["mae", "spearman", "winner_hit", "podium_hit"]].mean()
print("Grid baseline per season:")
print(by_season.round(3).to_string())
