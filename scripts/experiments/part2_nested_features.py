"""Part 2, experiment 1 done properly: NESTED feature selection.

Before each half-season from 2024 on, run backward elimination using only walk-forward results of
EARLIER races (2023 onward), then score that half-season with the selected features. 2023 uses all
features (no earlier walk-forward races to choose on). Compared with v1 on 2023-26.

    python scripts/experiments/part2_nested_features.py
"""
import json

import pandas as pd

from f1pred import config
from f1pred.evaluate import NOISE_FLOOR, compare, line, order_scores
from f1pred.pipeline import history_table

data, as_of, history, table = history_table(config.TARGET_SEASON, config.TARGET_ROUND)  # recorded for Austin 2026
FULL = list(config.FEATURES)
cache: dict[tuple, pd.DataFrame] = {}


def scores(features) -> pd.DataFrame:
    """Per-race walk-forward scores for a feature set, 2023-26 (valid for any cutoff: race k only uses races < k)."""
    key = tuple(sorted(features))
    if key not in cache:
        cache[key] = order_scores(table, list(features))
    return cache[key]


def eliminate(cutoff: int) -> list[str]:
    """Backward elimination judged only on races before `cutoff`; stops when dropping makes it worse than the
    full set by more than the noise floor, or makes nothing better."""
    def mae(fs):
        s = scores(fs)
        return s[(s.index.get_level_values(0) * 100 + s.index.get_level_values(1)) < cutoff]["mae"].mean()

    current, full = list(FULL), mae(FULL)
    while len(current) > 4:
        trial = {f: mae([x for x in current if x != f]) for f in current}
        f_best = min(trial, key=trial.get)
        if trial[f_best] > mae(current) or trial[f_best] - full > NOISE_FLOOR:
            break
        current.remove(f_best)
    return current


base = scores(FULL)
keys = base.index.get_level_values(0) * 100 + base.index.get_level_values(1)
parts, chosen = [], {}
for season in (2023, 2024, 2025, 2026):
    for half, lo, hi in (("H1", 1, 11), ("H2", 12, 30)):
        mask = (base.index.get_level_values(0) == season) & (base.index.get_level_values(1) >= lo) & \
               (base.index.get_level_values(1) <= hi)
        if not mask.any():
            continue
        cutoff = keys[mask].min()
        feats = FULL if season == 2023 else eliminate(cutoff)
        chosen[f"{season}{half}"] = sorted(set(FULL) - set(feats))
        parts.append(scores(feats)[mask])
        print(f"  {season}{half}: dropped {chosen[f'{season}{half}']} ({len(cache)} feature sets evaluated)", flush=True)

nested = pd.concat(parts)
r = compare(base, nested, "NESTED backward elimination (choices on earlier races only)", simplification=True,
            notes="dropped per half-season: " + json.dumps(chosen))
strict = compare(base, nested, "NESTED backward elimination, strict improvement test", log=False)
print("\n" + line(r) + "\n" + line(strict))
(config.PROJECT_ROOT / "reports" / "part2_nested_features.json").write_text(json.dumps(
    {"dropped_per_half_season": chosen, "mae_change": r["mae_change"], "ci": r["ci"], "decision": r["decision"]}, indent=2))
