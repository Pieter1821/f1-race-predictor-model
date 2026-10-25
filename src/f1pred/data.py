"""Load race results from Jolpica (through FastF1's Ergast client) into one tidy, validated table.

Everything is loaded "as of" a moment in time, normally the start of the race we're predicting:
races at or after that moment don't exist yet as far as the model is concerned.
"""
import re
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
from fastf1.ergast import Ergast

from f1pred.config import DATA_DIR, SEASONS, TARGET_ROUND, TARGET_SEASON, enable_cache

SNAPSHOT_DIR = DATA_DIR / "snapshots"

RESULT_COLUMNS = {
    "driverId": "driver_id",
    "driverCode": "driver_code",
    "constructorId": "constructor_id",
    "grid": "grid",
    "position": "position",
    "status": "status",
    "laps": "laps",
    "points": "points",
}

NON_STARTERS = {"Did not start", "Withdrew"}

# Cars entered per race, by season
EXPECTED_ENTRIES = {2022: 20, 2023: 20, 2024: 20, 2025: 20, 2026: 22}
# Races with one car fewer than usual, with the reason
KNOWN_SHORT_FIELDS = {
    (2024, 3): (19, "Sargeant withdrawn, his car given to Albon"),
    (2025, 9): (19, "Stroll withdrew injured"),
}

# Jolpica's status text -> what happened. Since 2023 most retirements are just "Retired" (cause unknown).
STATUS_BUCKETS = {
    "finished": {"Finished"},
    "dsq": {"Disqualified"},
    # Front wing / undertray retirements are almost always contact damage
    "dnf_accident": {"Accident", "Collision", "Collision damage", "Spun off", "Front wing", "Undertray"},
    "dnf_mechanical": {
        "Engine", "Power Unit", "Hydraulics", "Gearbox", "Water leak", "Fuel pressure", "Water pressure",
        "Water pump", "Suspension", "Fuel leak", "Cooling system", "Fuel pump", "Vibrations", "Turbo",
        "Mechanical", "Power loss", "Oil leak", "Differential", "Brakes", "Electrical", "Electronics",
        "Exhaust", "Overheating", "Wheel", "Wheel nut", "Puncture", "Battery", "Clutch", "Steering",
        "Transmission", "Driveshaft", "Throttle", "Radiator",
    },
    "dnf_unknown": {"Retired"},
}
_STATUS_TO_BUCKET = {status: bucket for bucket, statuses in STATUS_BUCKETS.items() for status in statuses}
_LAPPED = re.compile(r"^(\+\d+ Laps?|Lapped)$")


def status_bucket(status: str) -> str:
    if _LAPPED.match(status):
        return "lapped"
    if status in NON_STARTERS:
        return "did_not_start"
    try:
        return _STATUS_TO_BUCKET[status]
    except KeyError:
        raise ValueError(f"Unmapped status {status!r}: add it to STATUS_BUCKETS") from None


# ---------------------------------------------------------------- fetching from the API

def _ergast() -> Ergast:
    enable_cache()
    return Ergast(result_type="pandas", auto_cast=True, limit=100)


def _flatten(response, races: list[pd.DataFrame]) -> None:
    """Each page holds a description table (one row per race) and a results table per race."""
    for (_, race), results in zip(response.description.iterrows(), response.content):
        df = results[list(RESULT_COLUMNS)].rename(columns=RESULT_COLUMNS)
        df.insert(0, "season", race["season"])
        df.insert(1, "round", race["round"])
        df.insert(2, "race_name", race["raceName"])
        df.insert(3, "circuit_id", race["circuitId"])
        df.insert(4, "date", race["raceDate"])
        races.append(df)


def fetch_season_results(season: int) -> pd.DataFrame:
    """All race results for one season, exactly as Jolpica returns them. Pages of 100 rows."""
    response = _ergast().get_race_results(season=season)
    races: list[pd.DataFrame] = []
    _flatten(response, races)
    while True:
        try:
            response = response.get_next_result_page()
        except ValueError:  # raised when there is no next page
            break
        _flatten(response, races)
    df = pd.concat(races, ignore_index=True)
    # A race split across two pages arrives in two parts; the drivers don't overlap, but be safe
    return df.drop_duplicates(subset=["season", "round", "driver_id"]).reset_index(drop=True)


def _combine(day: pd.Timestamp, clock: time | None) -> pd.Timestamp:
    """Race date + start time (UTC). A missing start time counts as midnight."""
    if clock is None or pd.isna(clock):
        clock = time(0, 0)
    return pd.Timestamp(datetime.combine(day.date(), clock.replace(tzinfo=None)), tz="UTC")


def fetch_schedule(season: int) -> pd.DataFrame:
    s = _ergast().get_race_schedule(season=season)
    return pd.DataFrame({
        "season": season,
        "round": s["round"],
        "race_name": s["raceName"],
        "start_utc": [_combine(d, t) for d, t in zip(s["raceDate"], s["raceTime"])],
    })


def fetch_dsq_last_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """For each disqualified driver: their last lap and the position they held on it."""
    rows = []
    for _, r in raw[raw["status"] == "Disqualified"].iterrows():
        laps = _ergast().get_lap_times(season=r["season"], round=r["round"], driver=r["driver_id"]).content
        if not laps or laps[0].empty:
            continue
        last = laps[0].iloc[-1]
        rows.append({"season": r["season"], "round": r["round"], "driver_id": r["driver_id"],
                     "last_lap": int(last["number"]), "last_lap_position": int(last["position"])})
    return pd.DataFrame(rows, columns=["season", "round", "driver_id", "last_lap", "last_lap_position"])


# ---------------------------------------------------------------- snapshots

def save_snapshot(raw: pd.DataFrame, schedule: pd.DataFrame, dsq: pd.DataFrame, seasons=SEASONS) -> Path:
    """Dated copy of exactly what the API returned, so every result can be reproduced later."""
    path = SNAPSHOT_DIR / f"jolpica_{min(seasons)}-{max(seasons)}_{date.today().isoformat()}"
    path.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path / "results.csv", index=False)
    schedule.to_csv(path / "schedule.csv", index=False)
    dsq.to_csv(path / "dsq_last_laps.csv", index=False)
    return path


def read_snapshot(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = Path(path)
    raw = pd.read_csv(path / "results.csv", parse_dates=["date"])
    schedule = pd.read_csv(path / "schedule.csv")
    schedule["start_utc"] = pd.to_datetime(schedule["start_utc"], utc=True)
    dsq = pd.read_csv(path / "dsq_last_laps.csv")
    return raw, schedule, dsq


# ---------------------------------------------------------------- as-of filtering and validation

def target_race_start(schedule: pd.DataFrame, season=TARGET_SEASON, rnd=TARGET_ROUND) -> pd.Timestamp:
    row = schedule[(schedule["season"] == season) & (schedule["round"] == rnd)]
    assert len(row) == 1, f"{season} R{rnd} is not on the calendar"
    return row["start_utc"].iloc[0]


def races_before(schedule: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Calendar rows for races that started strictly before `as_of`."""
    return schedule[schedule["start_utc"] < as_of]


def restrict_to_as_of(raw: pd.DataFrame, schedule: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Drop every result from a race at or after `as_of`: that race hasn't happened yet."""
    known = races_before(schedule, as_of)[["season", "round"]]
    return raw.merge(known, on=["season", "round"], how="inner")


def validate_raw(raw: pd.DataFrame, schedule: pd.DataFrame, as_of: pd.Timestamp) -> None:
    expected = races_before(schedule, as_of).groupby("season").size()
    loaded = raw.groupby("season")["round"].nunique()
    for season, n in expected.items():
        assert loaded.get(season, 0) == n, f"{season}: loaded {loaded.get(season, 0)} races, calendar says {n} ran before {as_of}"
    for (season, rnd), race in raw.groupby(["season", "round"]):
        entries = len(race)
        if (season, rnd) in KNOWN_SHORT_FIELDS:
            want, _reason = KNOWN_SHORT_FIELDS[(season, rnd)]
        else:
            want = EXPECTED_ENTRIES[season]
        assert entries == want, f"{season} R{rnd}: {entries} entries, expected {want}"
        _assert_positions_complete(race["position"], f"{season} R{rnd} (raw)")


def _assert_positions_complete(positions: pd.Series, label: str) -> None:
    assert sorted(positions) == list(range(1, len(positions) + 1)), f"{label}: positions are not 1..N"


# ---------------------------------------------------------------- cleaning

def clean_results(raw: pd.DataFrame, dsq_last_laps: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["status_bucket"] = df["status"].map(status_bucket)
    # Drivers who never took the start have no race result to learn from
    df = df[df["status_bucket"] != "did_not_start"].copy()
    df["position"] = df.groupby(["season", "round"])["position"].rank(method="first").astype(int)

    field_size = df.groupby(["season", "round"])["driver_id"].transform("count")
    # grid 0 means a pit-lane start: that's behind the whole field, not ahead of pole
    df["pit_lane_start"] = df["grid"] == 0
    df.loc[df["pit_lane_start"], "grid"] = field_size[df["pit_lane_start"]]
    df["field_size"] = field_size
    df["finished"] = df["status_bucket"].isin(["finished", "lapped"])
    df["position_pct"] = df["position"] / df["field_size"]
    df["on_track_position"] = _on_track_positions(df, dsq_last_laps)

    for (season, rnd), race in df.groupby(["season", "round"]):
        _assert_positions_complete(race["position"], f"{season} R{rnd}")
        _assert_positions_complete(race["on_track_position"], f"{season} R{rnd} on-track")
    return df.sort_values(["season", "round", "position"]).reset_index(drop=True)


def _on_track_positions(df: pd.DataFrame, dsq_last_laps: pd.DataFrame) -> pd.Series:
    """Official order, except disqualified drivers go back to where they actually ran.

    - DSQ after completing the full distance: the position they crossed the line in.
    - DSQ partway through (black flag): treated like a retirement at that lap, i.e.
      behind everyone who completed more laps.

    The official target stays `position`; this is only for history features.
    """
    out = df["position"].copy()
    dsq = dsq_last_laps.set_index(["season", "round", "driver_id"])
    for (season, rnd), race in df.groupby(["season", "round"]):
        dsq_here = [d for d in race["driver_id"] if (season, rnd, d) in dsq.index]
        if not dsq_here:
            continue
        race_distance = race["laps"].max()
        order = race[~race["driver_id"].isin(dsq_here)].sort_values("position")
        order = list(zip(order["driver_id"], order["laps"]))
        for driver in sorted(dsq_here, key=lambda d: dsq.loc[(season, rnd, d), "last_lap_position"]):
            last_lap, last_pos = dsq.loc[(season, rnd, driver), ["last_lap", "last_lap_position"]]
            if last_lap >= race_distance:
                slot = min(last_pos, len(order) + 1) - 1
            else:
                # behind everyone who completed at least as many laps
                slot = sum(1 for _, laps in order if laps >= last_lap)
            order.insert(slot, (driver, last_lap))
        new_pos = {d: i + 1 for i, (d, _) in enumerate(order)}
        out.loc[race.index] = race["driver_id"].map(new_pos).values
    return out.astype(int)


# ---------------------------------------------------------------- entry point

def load_results(
    seasons=SEASONS,
    as_of: pd.Timestamp | None = None,
    from_snapshot: Path | None = None,
    save_new_snapshot: bool = False,
) -> pd.DataFrame:
    """Clean results for every race that started before `as_of` (default: the target race's start).

    from_snapshot: read a committed snapshot folder instead of calling the API.
    The returned table's .attrs record which source and cutoff were used.
    """
    if from_snapshot is not None:
        raw, schedule, dsq = read_snapshot(from_snapshot)
        raw = raw[raw["season"].isin(seasons)]
        schedule = schedule[schedule["season"].isin(seasons)]
        source = f"snapshot {Path(from_snapshot).name}"
    else:
        raw = pd.concat([fetch_season_results(s) for s in seasons], ignore_index=True)
        schedule = pd.concat([fetch_schedule(s) for s in seasons], ignore_index=True)
        dsq = fetch_dsq_last_laps(raw)
        source = "api"
        if save_new_snapshot:
            source = f"api (saved as {save_snapshot(raw, schedule, dsq, seasons).name})"

    as_of = as_of if as_of is not None else target_race_start(schedule)
    raw = restrict_to_as_of(raw, schedule, as_of)
    validate_raw(raw, schedule, as_of)
    df = clean_results(raw, dsq)
    df.attrs.update({"source": source, "as_of": as_of.isoformat()})
    return df
