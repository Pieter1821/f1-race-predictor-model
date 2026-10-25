"""Labelled VARIANT of the Austin 2026 prediction (not the official model).

Motivation (scripts/investigate_austin_pole.py): polesitters were under-rated in 2025-26 walk-forward
predictions (2026: 47.7% predicted vs 70.6% actual), and for Austin the gap to Verstappen comes from
history features built mostly on 2022-25 (Red Bull era, his Austin wins). Two changes:
  1. driver_level and track_level weight each earlier season half as much as the next (season decay)
  2. win calibration refitted on 2023-26 walk-forward simulations, i.e. including 2025-26
It can't be validated cleanly: the only evidence for the bias is the 2025-26 races it is fitted on.
"""
import numpy as np
import pandas as pd

from f1pred.features import race_order

SEASON_DECAY = 0.5  # one season back counts half; two seasons back a quarter


def decayed_prior_mean(df: pd.DataFrame, by: pd.Series, col: str, prior_weight: float,
                       prior_value: pd.Series, decay: float = SEASON_DECAY) -> pd.Series:
    """Like features.prior_mean, but each earlier race is weighted decay ** (seasons before this one).

    Strictly earlier races only (per-race totals, so a teammate's same-race result can't leak in).
    """
    idx = race_order(df)
    out = pd.Series(np.nan, index=df.index)
    frame = pd.DataFrame({"key": by.values, "race": idx.values, "season": df["season"].values,
                          "x": df[col].values}, index=df.index)
    per_race = frame.groupby(["key", "race"]).agg(season=("season", "first"), s=("x", "sum"), n=("x", "count"))
    for key, rows in frame.groupby("key"):
        hist = per_race.loc[key]
        races, seasons = hist.index.to_numpy(), hist["season"].to_numpy()
        sums, counts = hist["s"].to_numpy(), hist["n"].to_numpy()
        for i, (race, season) in zip(rows.index, zip(rows["race"], rows["season"])):
            earlier = races < race
            w = decay ** (season - seasons[earlier])
            weight = (w * counts[earlier]).sum() + prior_weight
            # no history and no prior weight -> unknown (NaN), as in features.prior_mean
            out[i] = ((w * sums[earlier]).sum() + prior_weight * prior_value[i]) / weight if weight > 0 else np.nan
    return out


def apply_variant_features(table: pd.DataFrame) -> pd.DataFrame:
    """Recompute driver_level and track_level with season decay, for every row (history and target)."""
    t = table.copy()
    t["driver_level"] = decayed_prior_mean(t, t["driver_id"], "on_track_pct", 5, t["team_form"])
    t["track_level"] = decayed_prior_mean(t, t["driver_id"] + "@" + t["circuit_id"], "on_track_pct", 2,
                                          t["driver_level"])
    return t
