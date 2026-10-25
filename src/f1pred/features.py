"""Features. Every feature for a race uses only races that finished before that race started.

The same code builds training rows (past races, results known) and prediction rows (the target
race, results unknown): prediction rows are appended with empty outcome columns, so nothing in
here can depend on a race's own result.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1pred import config

# Columns that are only known after the race. Features must never read these for the same race.
OUTCOME_COLUMNS = ["position", "on_track_position", "position_pct", "status", "status_bucket",
                   "finished", "laps", "points"]



# ---------------------------------------------------------------- leak-free building blocks

def race_order(df: pd.DataFrame) -> pd.Series:
    """0, 1, 2, ... in calendar order: one number per race."""
    return df.groupby(["season", "round"]).ngroup()


def league_prior(df: pd.DataFrame, col: str) -> pd.Series:
    """League-wide average of `col` over all strictly earlier races (NaN before the first race)."""
    idx = race_order(df)
    per_race = df.assign(_race=idx).groupby("_race")[col].agg(["sum", "count"]).sort_index()
    before = per_race.cumsum() - per_race
    avg = before["sum"] / before["count"].replace(0, np.nan)
    return pd.Series(avg.reindex(idx).values, index=df.index)


def prior_mean(df: pd.DataFrame, by: str, col: str, prior_weight: float = 0.0,
               prior_value: float | pd.Series | None = None) -> pd.Series:
    """Average of `col` for the same `by` (driver, team, ...) over strictly earlier races.

    Totals are taken per race first and then shifted by whole races, so a teammate's
    result from the same race can never leak in. With prior_weight > 0 the average is
    shrunk toward prior_value: (sum + w * prior) / (count + w). prior_value may be a
    per-row Series (hierarchical shrinkage); by default it's the league average so far.
    """
    idx = race_order(df)
    per_race = (df.assign(_race=idx).groupby([by, "_race"])[col]
                .agg(["sum", "count"]).sort_index())
    before = per_race.groupby(level=by).cumsum() - per_race  # everything up to, not including, this race
    keys = pd.MultiIndex.from_arrays([df[by], idx])
    s = pd.Series(before["sum"].reindex(keys).values, index=df.index)
    n = pd.Series(before["count"].reindex(keys).values, index=df.index)
    if prior_value is None:
        prior_value = league_prior(df, col)
    return (s + prior_weight * prior_value) / (n + prior_weight)


def recent_mean(df: pd.DataFrame, by: str, col: str, window: int) -> pd.Series:
    """Average of `col` over the last `window` earlier races of the same `by` (one row per race per `by`)."""
    ordered = df.assign(_race=race_order(df)).sort_values("_race")
    rolled = ordered.groupby(by)[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return rolled.reindex(df.index)


# ---------------------------------------------------------------- the feature set

@dataclass(frozen=True)
class Feature:
    name: str
    definition: str
    missing: str
    direction: str  # expected effect on finishing position (higher number = further back)


FEATURES = [
    Feature("grid", "Final starting position after penalties; pit-lane start = back of grid",
            "never missing", "+ (further back -> worse finish)"),
    Feature("grid_pct", "grid / number of starters (20 vs 22-car fields comparable)",
            "never missing", "+"),
    Feature("quali_position", "Qualifying classification",
            "no qualifying time -> back of the field, with no_quali_time=1", "+"),
    Feature("no_quali_time", "1 if the driver set no qualifying time", "never missing", "+"),
    Feature("grid_penalty", "grid - quali_position (+ = starts further back than he qualified)",
            "0 when unknown (2022 sprint-set grids, no quali time); flags say why", "- (recovers places)"),
    Feature("sprint_sets_grid", "2022 sprint weekends, where the sprint set the race grid", "never missing", "?"),
    Feature("quali_best_gap_pct", "Smallest % gap to the fastest lap in any session the driver ran, capped at 5%",
            "no time -> 5% (the cap)", "+"),
    Feature("teammate_quali_delta", "quali_best_gap_pct minus the teammate's (negative = beat teammate)",
            "0 if either car has no time", "+"),
    Feature("team_form", "Team's average on-track finish (0=front, 1=back) this season, shrunk toward the team's all-time form",
            "first race of the dataset only", "+"),
    Feature("driver_level", "Driver's average on-track finish over all earlier races, shrunk toward team_form",
            "first race of the dataset only", "+"),
    Feature("driver_vs_teammate", "Driver's average on-track finish minus his teammate's (same car), earlier races, shrunk toward 0",
            "debut -> 0 (level with teammate)", "+ (negative = beats teammates)"),
    Feature("driver_recent_form", "Driver's average on-track finish over his last 5 races",
            "debut -> driver_level", "+"),
    Feature("track_level", "Driver's average on-track finish at this circuit, shrunk toward driver_level",
            "first race of the dataset only", "+"),
    Feature("team_dnf_rate", "Team's share of starts not finished this season, shrunk toward this season's league rate so far",
            "first race of the dataset only", "+"),
    Feature("driver_standing", "Driver's championship position before this race (from our own results)",
            "round 1 -> everyone tied", "+"),
    Feature("constructor_standing", "Team's championship position before this race (from our own results)",
            "round 1 -> everyone tied", "+"),
]
FEATURE_NAMES = [f.name for f in FEATURES]


def _quali_features(df: pd.DataFrame) -> pd.DataFrame:
    gaps = [(df[q] / df.groupby(["season", "round"])[q].transform("min") - 1) * 100 for q in ("q1", "q2", "q3")]
    best_gap = pd.concat(gaps, axis=1).min(axis=1)
    df["no_quali_time"] = best_gap.isna().astype(int)
    df["quali_best_gap_pct"] = best_gap.clip(upper=config.QUALI_GAP_CAP).fillna(config.QUALI_GAP_CAP)
    df["quali_position"] = df["quali_position"].fillna(df["field_size"])
    known = ~df["sprint_sets_grid"].astype(bool) & (df["no_quali_time"] == 0)
    df["grid_penalty"] = (df["grid"] - df["quali_position"]).where(known, 0)

    team_sum = df.groupby(["season", "round", "team"])["quali_best_gap_pct"].transform("sum")
    team_n = df.groupby(["season", "round", "team"])["quali_best_gap_pct"].transform("count")
    teammate_gap = (team_sum - df["quali_best_gap_pct"]) / (team_n - 1).replace(0, np.nan)
    has_both = (df.groupby(["season", "round", "team"])["no_quali_time"].transform("sum") == 0) & (team_n == 2)
    df["teammate_quali_delta"] = (df["quali_best_gap_pct"] - teammate_gap).where(has_both, 0.0)
    return df


def _standings(df: pd.DataFrame, by: str) -> pd.Series:
    """Championship position before this race, from points in earlier races of the same season."""
    pts = df.groupby(["season", "round", by])["points"].transform("sum").fillna(0)
    per_race = df.assign(_pts=pts).drop_duplicates(["season", "round", by])[["season", "round", by, "_pts"]]
    per_race = per_race.sort_values(["season", "round"])
    per_race["_before"] = per_race.groupby(["season", by])["_pts"].cumsum() - per_race["_pts"]
    before = df.merge(per_race[["season", "round", by, "_before"]], on=["season", "round", by], how="left")["_before"]
    before.index = df.index
    return before.groupby([df["season"], df["round"]]).rank(ascending=False, method="min")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every feature in FEATURES. `df` holds past races (outcomes known) and, optionally,
    the target race (outcomes NaN). Rows must be one per driver per race."""
    df = df.copy()
    # Target-race rows have empty outcomes; after a concat those columns can arrive as generic objects
    for col in ("position", "on_track_position", "points", "laps"):
        if col in df:
            df[col] = pd.to_numeric(df[col])
    df["grid_pct"] = df["grid"] / df["field_size"]
    df["sprint_sets_grid"] = df["sprint_sets_grid"].fillna(False).astype(int)
    df = _quali_features(df)

    df["on_track_pct"] = df["on_track_position"] / df["field_size"]
    df["dnf"] = np.where(df["finished"].isna(), np.nan,
                         (~df["finished"].astype(bool) & (df["status_bucket"] != "dsq")).astype(float))

    team_all = prior_mean(df, "team", "on_track_pct", prior_weight=config.SHRINK_TEAM_ALL)
    df["_team_season"] = df["team"] + "_" + df["season"].astype(str)
    df["team_form"] = prior_mean(df, "_team_season", "on_track_pct", prior_weight=config.SHRINK_TEAM_SEASON,
                                 prior_value=team_all)
    df["driver_level"] = prior_mean(df, "driver_id", "on_track_pct", prior_weight=config.SHRINK_DRIVER,
                                    prior_value=df["team_form"])
    # Driver vs teammate (same car): the driver's edge, separated from how good the car is
    mate_sum = df.groupby(["season", "round", "team"])["on_track_pct"].transform("sum")
    mate_n = df.groupby(["season", "round", "team"])["on_track_pct"].transform("count")
    df["_vs_mate"] = (df["on_track_pct"] - (mate_sum - df["on_track_pct"]) / (mate_n - 1)).where(mate_n == 2)
    df["driver_vs_teammate"] = prior_mean(df, "driver_id", "_vs_mate", prior_weight=config.SHRINK_TEAMMATE, prior_value=0.0)
    df["driver_recent_form"] = recent_mean(df, "driver_id", "on_track_pct", window=config.FORM_WINDOW).fillna(df["driver_level"])
    df["_driver_circuit"] = df["driver_id"] + "@" + df["circuit_id"]
    df["track_level"] = prior_mean(df, "_driver_circuit", "on_track_pct", prior_weight=config.SHRINK_TRACK,
                                   prior_value=df["driver_level"])

    # DNF base rates differ a lot by season (2026 ~19% vs ~11% in 2024-25), so shrink a team toward
    # this season's league rate so far, which is itself shrunk toward the all-time league rate
    df["_season"] = df["season"].astype(str)
    season_dnf = prior_mean(df, "_season", "dnf", prior_weight=config.SHRINK_DNF_SEASON,
                            prior_value=league_prior(df, "dnf"))
    df["team_dnf_rate"] = prior_mean(df, "_team_season", "dnf", prior_weight=config.SHRINK_DNF_TEAM,
                                     prior_value=season_dnf)

    df["driver_standing"] = _standings(df, "driver_id")
    df["constructor_standing"] = _standings(df, "team")
    return df.drop(columns=["_team_season", "_driver_circuit", "_season", "_vs_mate"])


def append_target_race(history: pd.DataFrame, inputs: pd.DataFrame, circuit_id: str,
                       race_date: pd.Timestamp) -> pd.DataFrame:
    """History plus the target race's rows (pre-race inputs only; every outcome column left empty)."""
    target = inputs.assign(
        circuit_id=circuit_id, date=race_date, field_size=len(inputs),
        pit_lane_start=False, sprint_sets_grid=False, is_target=True,
    )
    for col in OUTCOME_COLUMNS:
        target[col] = np.nan
    return pd.concat([history.assign(is_target=False), target], ignore_index=True)


def add_dnf_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Team and driver DNF rates from earlier races (used by the Stage 3 EDA)."""
    df = df.copy()
    df["dnf"] = (~df["finished"] & (df["status_bucket"] != "dsq")).astype(float)
    df["team_dnf_rate"] = prior_mean(df, "team", "dnf", prior_weight=10)
    df["driver_dnf_rate"] = prior_mean(df, "driver_id", "dnf", prior_weight=10)
    return df
