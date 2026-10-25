"""Scoring: how close is a predicted finishing order to the real one?"""
import pandas as pd
from scipy.stats import spearmanr


def predicted_rank(race: pd.DataFrame, pred_col: str) -> pd.Series:
    """Turn a prediction score into places 1..N.

    Ties are broken by grid, then driver_id, never by row order: rows used to be stored in
    finishing order, where a row-order tie-break would quietly use the result.
    """
    order = race.sort_values([pred_col, "grid", "driver_id"], kind="stable").index
    return pd.Series(range(1, len(race) + 1), index=order).reindex(race.index)


def score_race(race: pd.DataFrame, pred_col: str, actual_col: str = "position") -> dict:
    actual = race[actual_col]
    pred_rank = predicted_rank(race, pred_col)
    return {
        "mae": (pred_rank - actual).abs().mean(),
        "spearman": spearmanr(actual, pred_rank).statistic,
        "winner_hit": float(actual[pred_rank == 1].iloc[0] == 1),
        "podium_hit": ((pred_rank <= 3) & (actual <= 3)).sum() / 3,
    }


def score_many(df: pd.DataFrame, pred_col: str, actual_col: str = "position") -> pd.DataFrame:
    """Score every race separately, one row per race."""
    rows = []
    for (season, rnd), race in df.groupby(["season", "round"]):
        rows.append({"season": season, "round": rnd, **score_race(race, pred_col, actual_col)})
    return pd.DataFrame(rows)
