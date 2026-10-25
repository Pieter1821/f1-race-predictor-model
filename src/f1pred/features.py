"""Features. Every feature for a race uses only races that finished before that race started."""
import pandas as pd


def race_order(df: pd.DataFrame) -> pd.Series:
    """0, 1, 2, ... in calendar order: one number per race."""
    return df.groupby(["season", "round"]).ngroup()


def prior_mean(df: pd.DataFrame, by: str, col: str, prior_weight: float = 0.0,
               prior_value: float | None = None) -> pd.Series:
    """Average of `col` for the same `by` (driver, team, ...) over strictly earlier races.

    Totals are taken per race first and then shifted by whole races, so a teammate's
    result from the same race can never leak in. With prior_weight > 0 the average is
    shrunk toward prior_value: (sum + w * prior) / (count + w).
    """
    idx = race_order(df)
    per_race = (df.assign(_race=idx).groupby([by, "_race"])[col]
                .agg(["sum", "count"]).sort_index())
    before = per_race.groupby(level=by).cumsum() - per_race  # everything up to, not including, this race
    prior_value = df[col].mean() if prior_value is None else prior_value
    smoothed = (before["sum"] + prior_weight * prior_value) / (before["count"] + prior_weight)
    keys = pd.MultiIndex.from_arrays([df[by], idx])
    return pd.Series(smoothed.reindex(keys).values, index=df.index)


def add_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Team and driver DNF rates from earlier races, shrunk toward the overall rate (few races = noisy)."""
    df = df.copy()
    df["dnf"] = (~df["finished"] & (df["status_bucket"] != "dsq")).astype(float)
    df["team_dnf_rate"] = prior_mean(df, "team", "dnf", prior_weight=10)
    df["driver_dnf_rate"] = prior_mean(df, "driver_id", "dnf", prior_weight=10)
    return df
