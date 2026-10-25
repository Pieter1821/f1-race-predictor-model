"""Training-row weights by race age. A modelling choice to test in walk-forward validation, not a cleaning rule.

Weights depend only on when a race happened relative to the training cutoff, never on its result.
"""
import numpy as np
import pandas as pd

# Candidates compared in Stage 5; the winner is chosen on validation folds only and recorded with the model
CANDIDATES = {
    "uniform": {"half_life_races": None, "current_regs_boost": 1.0},
    "half_life_12": {"half_life_races": 12, "current_regs_boost": 1.0},   # ~half a season
    "half_life_24": {"half_life_races": 24, "current_regs_boost": 1.0},   # ~one season
    "half_life_48": {"half_life_races": 48, "current_regs_boost": 1.0},   # ~two seasons
    "regs_2026_x3": {"half_life_races": None, "current_regs_boost": 3.0},  # new 2026 rules: favour this season
}
REGULATION_ERAS = {2022: "ground_effect", 2023: "ground_effect", 2024: "ground_effect",
                   2025: "ground_effect", 2026: "2026_rules"}


def recency_weights(train: pd.DataFrame, half_life_races: float | None = None,
                    current_regs_boost: float = 1.0) -> pd.Series:
    """Weight per training row. The most recent training race gets 1; a race `h` races older gets 0.5.

    current_regs_boost multiplies rows from the same regulation era as the latest training race.
    """
    race_idx = train.groupby(["season", "round"]).ngroup()
    age = race_idx.max() - race_idx  # 0 = latest race in this training set
    w = pd.Series(1.0, index=train.index) if half_life_races is None else 0.5 ** (age / half_life_races)
    eras = train["season"].map(REGULATION_ERAS)
    latest_era = eras[race_idx == race_idx.max()].iloc[0]
    w = w * np.where(eras == latest_era, current_regs_boost, 1.0)
    return w.astype(float)
