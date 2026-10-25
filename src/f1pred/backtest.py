"""Walk-forward backtest: for each test race, train on every earlier race only, then predict it.

This mirrors race day exactly. Features in the table are already as-of each race (proven by
tests/test_features.py), so one feature table can serve every fold; the training rows are what
moves forward.
"""
from typing import Callable

import pandas as pd

from f1pred.models import TARGET
from f1pred.validate import score_many
from f1pred.weights import recency_weights

DEV_SEASONS = [2024]          # make choices here
TEST_SEASONS = [2025, 2026]   # report here, once


def race_key(df: pd.DataFrame) -> pd.Series:
    return df["season"] * 100 + df["round"]


def walk_forward(table: pd.DataFrame, make_model: Callable, features: list[str], test_seasons: list[int],
                 weighting: dict | None = None, min_train_races: int = 10) -> pd.DataFrame:
    """Predictions for every race in test_seasons, each from a model trained on strictly earlier races."""
    history = table[~table["is_target"] & table[TARGET].notna()]
    keys = race_key(history)
    preds = []
    for key in sorted(keys[history["season"].isin(test_seasons)].unique()):
        train = history[keys < key]
        test = history[keys == key]
        assert train[["season", "round"]].drop_duplicates().shape[0] >= min_train_races
        assert (race_key(train) < key).all()  # never train on the race being predicted, or later
        weights = None if weighting is None else recency_weights(train, **weighting).to_numpy()
        model = make_model().fit(train, features, weights)
        preds.append(test.assign(pred=model.predict(test, features)))
    return pd.concat(preds)


def summarise(preds: pd.DataFrame) -> pd.Series:
    scores = score_many(preds, "pred")
    return scores[["mae", "spearman", "winner_hit", "podium_hit"]].mean()
