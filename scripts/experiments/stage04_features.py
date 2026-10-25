"""Stage 4: build the feature table for every past race plus the target race.

    python scripts/experiments/stage04_features.py      # writes data/processed/features.parquet
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from f1pred.config import PROCESSED_DIR, SNAPSHOT, TARGET_ROUND, TARGET_SEASON
from f1pred.data import load_raw, load_results, race_start
from f1pred.features import FEATURE_NAMES, FEATURES, append_target_race, build_features
from f1pred.race_inputs import load_race_inputs

pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 70)

data = load_raw(from_snapshot=SNAPSHOT)
as_of = race_start(data.schedule, TARGET_SEASON, TARGET_ROUND)
history = load_results(as_of=as_of, data=data)
race = data.schedule.set_index(["season", "round"]).loc[(TARGET_SEASON, TARGET_ROUND)]
inputs = load_race_inputs(TARGET_SEASON, TARGET_ROUND, history, race["qualifying_start_utc"], as_of)

table = build_features(append_target_race(history, inputs, race["circuit_id"], as_of.normalize().tz_localize(None)))
table.attrs.update(history.attrs)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
table.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
train, target = table[~table["is_target"]], table[table["is_target"]]
print(f"source: {history.attrs['source']} | as of {as_of}")
print(f"{len(train)} training rows ({train.groupby(['season', 'round']).ngroups} races) + {len(target)} target rows\n")

print("Feature registry:")
print(pd.DataFrame([vars(f) for f in FEATURES]).to_string(index=False), "\n")

missing = table[FEATURE_NAMES].isna().sum()
print("Missing values (expected only in the very first race of the data, 2022 R1):")
print(missing[missing > 0].to_string() if missing.any() else "none", "\n")
assert not target[FEATURE_NAMES].isna().any().any(), "target race has missing features"


def partial_spearman(x, y, z) -> float:
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    resid = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)
    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


ok = train.dropna(subset=FEATURE_NAMES)
strength = pd.DataFrame({
    "vs_finish": [ok[f].corr(ok["position"], method="spearman") for f in FEATURE_NAMES],
    "beyond_grid": [partial_spearman(ok[f], ok["position"], ok["grid"]) if f != "grid" else np.nan for f in FEATURE_NAMES],
}, index=FEATURE_NAMES)
print("Signal check on the final features (Spearman):")
print(strength.round(3).to_string(), "\n")

cols = ["driver_code", "grid", "quali_position", "grid_penalty", "quali_best_gap_pct", "teammate_quali_delta",
        "team_form", "driver_level", "driver_vs_teammate", "driver_recent_form", "track_level", "team_dnf_rate", "driver_standing"]
target = target.assign(driver_code=target["code"])
print(f"{TARGET_SEASON} R{TARGET_ROUND} feature rows (form columns: 0 = always at the front, 1 = always at the back):")
print(target.sort_values("grid")[cols].round(3).to_string(index=False))
