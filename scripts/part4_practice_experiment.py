"""Part 4: do OpenF1 practice long-run features improve walk-forward accuracy over v2?

    python scripts/part4_practice_experiment.py
Prints the cleaning report and data coverage, then v2 vs v2 + practice features (same acceptance rules).
"""
import pandas as pd

from f1pred import config
from f1pred.evaluate import compare, line, order_scores
from f1pred.pipeline import history_table
from f1pred.practice import PRACTICE_FEATURES, add_practice_features, clean_laps, longrun_features, map_drivers

OPENF1 = sorted((config.DATA_DIR / "snapshots").glob("openf1_*"))[-1]
pd.set_option("display.width", 160)
_, _, history, table = history_table()

laps = pd.read_parquet(OPENF1 / "practice_laps.parquet")
stints = pd.read_parquet(OPENF1 / "practice_stints.parquet")
drivers = pd.read_parquet(OPENF1 / "drivers.parquet")
clean, report = clean_laps(laps, stints)
print(f"OpenF1 snapshot {OPENF1.name}\ncleaning report:")
for k, v in report.items():
    print(f"  {k:<40} {v:>8}")
feats = longrun_features(clean)
entries = history[["season", "round", "driver_id", "driver_code"]].rename(columns={"driver_code": "code"})
mapping = map_drivers(drivers, entries)
unmapped = feats.merge(mapping, on=["season", "round", "driver_number"], how="left")["driver_id"].isna().sum()
print(f"\nlong-run feature rows: {len(feats)} | not mapped to a race entrant (reserves): {unmapped}")

t = add_practice_features(table, feats, mapping)
cov = t[t["season"] >= 2023].groupby("season")["fp_has_longrun"].mean()
print("share of race starters with a practice long run:", cov.round(3).to_dict())
fin = t[(t["fp_has_longrun"] == 1) & t["finished"].astype(bool)]
print("rank correlation, long-run gap vs finishing position (finishers):",
      round(fin["fp_longrun_gap_pct"].corr(fin["position"], method="spearman"), 3),
      "| vs grid:", round(fin["fp_longrun_gap_pct"].corr(fin["grid"], method="spearman"), 3))

base = order_scores(t, config.FEATURES)
cand = order_scores(t, config.FEATURES + PRACTICE_FEATURES)
r = compare(base, cand, "Part 4: v2 + practice long-run features (OpenF1)", notes=f"snapshot {OPENF1.name}")
print("\n" + line(r))
print(f"v2 MAE {base['mae'].mean():.3f} -> {cand['mae'].mean():.3f}")
