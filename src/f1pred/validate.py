"""Scoring: how close is a predicted finishing order to the real one?"""
import pandas as pd
from scipy.stats import spearmanr


def score_race(actual: pd.Series, predicted: pd.Series) -> dict:
    """Both series hold finishing positions (1 = winner) for the same drivers, in the same order."""
    pred_rank = predicted.rank(method="first")
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
        rows.append({"season": season, "round": rnd, **score_race(race[actual_col], race[pred_col])})
    return pd.DataFrame(rows)
