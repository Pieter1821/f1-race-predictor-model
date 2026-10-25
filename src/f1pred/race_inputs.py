"""Pre-race inputs for the race being predicted: qualifying and the final starting grid.

These come from hand-entered CSVs (data/manual/) when Jolpica hasn't published them yet,
so they get stricter checks than API data: a typo here goes straight into the prediction.
"""
from pathlib import Path

import pandas as pd

from f1pred.config import MANUAL_DIR
from f1pred.data import EXPECTED_ENTRIES, TEAM_LINEAGE


class RaceInputError(ValueError):
    pass


def read_manual_csv(path: Path) -> pd.DataFrame:
    """The hand-entered files use Jolpica's camelCase ids; switch to this project's names."""
    return pd.read_csv(path).rename(columns={"driverId": "driver_id", "constructorId": "constructor_id"})


def _lap_seconds(text) -> float:
    """'1:35.390' -> 95.39. Blank or 'No Time Set' -> NaN."""
    if pd.isna(text) or not str(text).strip():
        return float("nan")
    minutes, seconds = str(text).strip().split(":")
    return int(minutes) * 60 + float(seconds)


def _check_positions(df: pd.DataFrame, col: str, n: int, label: str, problems: list[str]) -> None:
    if len(df) != n:
        problems.append(f"{label}: {len(df)} rows, expected {n}")
    dupes = df.loc[df["driver_id"].duplicated(keep=False), "driver_id"].tolist()
    if dupes:
        problems.append(f"{label}: driver listed twice: {sorted(set(dupes))}")
    if sorted(df[col]) != list(range(1, len(df) + 1)):
        problems.append(f"{label}: {col} is not 1..{len(df)} (got {sorted(df[col].tolist())})")


def _q2_cutoff(n: int) -> int:
    """How many cars reach Q2: 6 knocked out in Q1 with 22 cars, 5 with 20."""
    return n - 6 if n == 22 else n - 5


def validate_race_inputs(quali: pd.DataFrame, grid: pd.DataFrame, history: pd.DataFrame,
                         season: int, qualifying_start: pd.Timestamp, as_of: pd.Timestamp) -> list[str]:
    """Every problem found, as a readable message. Empty list = inputs are consistent."""
    problems: list[str] = []
    n = EXPECTED_ENTRIES[season]
    _check_positions(quali, "position", n, "qualifying", problems)
    _check_positions(grid, "grid", n, "grid", problems)

    q_drivers, g_drivers = set(quali["driver_id"]), set(grid["driver_id"])
    if q_drivers != g_drivers:
        problems.append(f"drivers differ: only in qualifying {sorted(q_drivers - g_drivers)}, "
                        f"only on grid {sorted(g_drivers - q_drivers)}")

    # Each driver must exist in this season's history, in the team they last raced for
    season_hist = history[history["season"] == season].sort_values(["round"])
    latest_team = season_hist.groupby("driver_id")["constructor_id"].last()
    for frame, label in ((quali, "qualifying"), (grid, "grid")):
        for _, row in frame.iterrows():
            driver, team = row["driver_id"], row["constructor_id"]
            if driver not in latest_team.index:
                problems.append(f"{label}: {driver} has no {season} race before this one (typo, or a debut to confirm)")
            elif latest_team[driver] != team:
                problems.append(f"{label}: {driver} constructor_id={team!r}, but last raced for {latest_team[driver]!r}")

    # Session logic: only the top cars can have Q2/Q3 times
    q2_cut = _q2_cutoff(n)
    times = quali.assign(q1s=quali["q1"].map(_lap_seconds), q2s=quali["q2"].map(_lap_seconds),
                         q3s=quali["q3"].map(_lap_seconds))
    for _, row in times.iterrows():
        pos, driver = row["position"], row["driver_id"]
        if pos > q2_cut and pd.notna(row["q2s"]):
            problems.append(f"qualifying: {driver} is P{pos} (out in Q1) but has a Q2 time {row['q2']}")
        if pos > 10 and pd.notna(row["q3s"]):
            problems.append(f"qualifying: {driver} is P{pos} (out in Q2) but has a Q3 time {row['q3']}")
        if pos <= q2_cut and pd.isna(row["q2s"]) and pd.isna(row["q3s"]):
            problems.append(f"qualifying: {driver} is P{pos} (reached Q2) but has no Q2 time")
    for col in ("q1s", "q2s", "q3s"):
        bad = times[(times[col] < 60) | (times[col] > 200)]
        for _, row in bad.iterrows():
            problems.append(f"qualifying: {row['driver_id']} {col[:2].upper()} = {row[col[:2]]} is not a plausible lap time")

    if not qualifying_start < as_of:
        problems.append(f"qualifying started {qualifying_start}, not before the prediction cutoff {as_of}")
    return problems


def load_race_inputs(season: int, rnd: int, history: pd.DataFrame, qualifying_start: pd.Timestamp,
                     as_of: pd.Timestamp, manual_dir: Path = MANUAL_DIR) -> pd.DataFrame:
    """One row per starter: qualifying position and times, final grid, grid penalty. Raises on any problem."""
    quali = read_manual_csv(manual_dir / f"{season}_R{rnd}_qualifying.csv")
    grid = read_manual_csv(manual_dir / f"{season}_R{rnd}_grid.csv")
    problems = validate_race_inputs(quali, grid, history, season, qualifying_start, as_of)
    if problems:
        raise RaceInputError(f"{season} R{rnd} inputs failed {len(problems)} check(s):\n  " + "\n  ".join(problems))

    df = grid.merge(
        quali.rename(columns={"position": "quali_position"}).drop(columns=["code", "constructor_id"]),
        on="driver_id", how="inner", validate="one_to_one",
    )
    for q in ("q1", "q2", "q3"):
        df[q] = df[q].map(_lap_seconds)
    df["team"] = df["constructor_id"].replace(TEAM_LINEAGE)
    df["grid_penalty"] = df["grid"] - df["quali_position"]
    df.insert(0, "season", season)
    df.insert(1, "round", rnd)
    return df.sort_values("grid").reset_index(drop=True)
