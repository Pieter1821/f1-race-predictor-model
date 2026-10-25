"""Load race data from Jolpica (through FastF1's Ergast client) into tidy, validated tables.

Everything is loaded "as of" a moment in time, normally the start of the race being predicted:
races at or after that moment don't exist yet as far as the model is concerned. There is
deliberately no default; every caller must say which moment it means.
"""
import re
import time as clock
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd
from fastf1.ergast import Ergast

from f1pred.config import DATA_DIR, SEASONS, enable_cache

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

# Renamed teams keep their history: same factory, same people
TEAM_LINEAGE = {"alfa": "sauber", "audi": "sauber", "alphatauri": "rb"}

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


def _all_pages(response) -> list:
    """Jolpica returns 100 rows per page; follow the pages until there are none left."""
    pages = [response]
    while True:
        try:
            pages.append(pages[-1].get_next_result_page())
        except ValueError:  # raised when there is no next page
            return pages


def _race_tables(pages, columns: dict) -> pd.DataFrame:
    """Each page holds a description table (one row per race) and a data table per race."""
    races = []
    for page in pages:
        for (_, race), table in zip(page.description.iterrows(), page.content):
            # reindex: a page holding only Q1 eliminations has no Q2/Q3 columns at all
            df = table.reindex(columns=list(columns)).rename(columns=columns)
            df.insert(0, "season", race["season"])
            df.insert(1, "round", race["round"])
            df.insert(2, "race_name", race["raceName"])
            df.insert(3, "circuit_id", race["circuitId"])
            df.insert(4, "date", race["raceDate"])
            # Lap times as seconds before concat: concatenating timedeltas hits a NumPy 2.5 deprecation in pandas
            for col in df.select_dtypes("timedelta").columns:
                df[col] = df[col].dt.total_seconds()
            races.append(df.dropna(axis=1, how="all"))  # all-blank pieces would muddle concat's dtypes
    ordered = ["season", "round", "race_name", "circuit_id", "date", *columns.values()]
    # A race split across two pages arrives in two parts. Only drop exact duplicate rows,
    # so a driver wrongly listed twice (e.g. under two teams) is still caught by the checks.
    return pd.concat(races, ignore_index=True).reindex(columns=ordered).drop_duplicates().reset_index(drop=True)


def fetch_season_results(season: int) -> pd.DataFrame:
    """All race results for one season, exactly as Jolpica returns them."""
    return _race_tables(_all_pages(_ergast().get_race_results(season=season)), RESULT_COLUMNS)


def fetch_season_qualifying(season: int) -> pd.DataFrame:
    columns = {"driverId": "driver_id", "constructorId": "constructor_id", "position": "quali_position",
               "Q1": "q1", "Q2": "q2", "Q3": "q3"}
    df = _race_tables(_all_pages(_ergast().get_qualifying_results(season=season)), columns)  # times in seconds
    return df[["season", "round", "driver_id", "constructor_id", "quali_position", "q1", "q2", "q3"]]


def _combine(day, clock: time | None) -> pd.Timestamp:
    """Date + start time (UTC). A missing start time counts as midnight; a missing date stays missing."""
    if pd.isna(day):
        return pd.NaT
    if clock is None or pd.isna(clock):
        clock = time(0, 0)
    return pd.Timestamp(datetime.combine(pd.Timestamp(day).date(), clock.replace(tzinfo=None)), tz="UTC")


def fetch_schedule(season: int) -> pd.DataFrame:
    s = _ergast().get_race_schedule(season=season)
    return pd.DataFrame({
        "season": season,
        "round": s["round"],
        "race_name": s["raceName"],
        "circuit_id": s["circuitId"],
        "lat": s["lat"],
        "lon": s["long"],
        "start_utc": [_combine(d, t) for d, t in zip(s["raceDate"], s["raceTime"])],
        "has_sprint": s["sprintDate"].notna(),
        "qualifying_start_utc": [_combine(d, t) for d, t in zip(s["qualifyingDate"], s["qualifyingTime"])],
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


def fetch_driver_debuts(driver_ids) -> pd.DataFrame:
    """Dates of each driver's first three career races (from all of F1 history, not just our seasons)."""
    rows = []
    for driver in sorted(set(driver_ids)):
        clock.sleep(0.5)  # ~70 lookups in a row trips Jolpica's rate limit (HTTP 429) otherwise
        first = _ergast().get_race_results(driver=driver, limit=3).description
        dates = sorted(first["raceDate"])
        rows.append({"driver_id": driver, **{f"race_{i + 1}_date": d for i, d in enumerate(dates)}})
    return pd.DataFrame(rows, columns=["driver_id", "race_1_date", "race_2_date", "race_3_date"])


# ---------------------------------------------------------------- the raw bundle and snapshots

def snapshot_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")


@dataclass
class RawData:
    results: pd.DataFrame
    schedule: pd.DataFrame
    qualifying: pd.DataFrame
    dsq_last_laps: pd.DataFrame
    driver_debuts: pd.DataFrame
    source: str

    _FILES = ("results", "schedule", "qualifying", "dsq_last_laps", "driver_debuts")


def fetch_raw(seasons=SEASONS) -> RawData:
    results = pd.concat([fetch_season_results(s) for s in seasons], ignore_index=True)
    return RawData(
        results=results,
        schedule=pd.concat([fetch_schedule(s) for s in seasons], ignore_index=True),
        qualifying=pd.concat([fetch_season_qualifying(s) for s in seasons], ignore_index=True),
        dsq_last_laps=fetch_dsq_last_laps(results),
        driver_debuts=fetch_driver_debuts(results["driver_id"]),
        source="api",
    )


def save_snapshot(data: RawData, seasons=SEASONS) -> Path:
    """Copy of exactly what the API returned, named by fetch time (UTC). Snapshots are never edited:
    new or extra data always goes into a new folder."""
    path = SNAPSHOT_DIR / f"jolpica_{min(seasons)}-{max(seasons)}_{snapshot_stamp()}"
    path.mkdir(parents=True, exist_ok=False)  # refuses to overwrite an existing snapshot
    for name in RawData._FILES:
        getattr(data, name).to_csv(path / f"{name}.csv", index=False)
    return path


def read_snapshot(path: Path) -> RawData:
    path = Path(path)
    tables = {name: pd.read_csv(path / f"{name}.csv") for name in RawData._FILES}
    tables["results"]["date"] = pd.to_datetime(tables["results"]["date"])
    for col in ("start_utc", "qualifying_start_utc"):
        tables["schedule"][col] = pd.to_datetime(tables["schedule"][col], utc=True)
    for col in ("race_1_date", "race_2_date", "race_3_date"):
        tables["driver_debuts"][col] = pd.to_datetime(tables["driver_debuts"][col])
    return RawData(**tables, source=f"snapshot {path.name}")


def load_raw(from_snapshot: Path | None = None, save_new_snapshot: bool = False) -> RawData:
    if from_snapshot is not None:
        return read_snapshot(from_snapshot)
    data = fetch_raw()
    if save_new_snapshot:
        data.source = f"api (saved as {save_snapshot(data).name})"
    return data


# ---------------------------------------------------------------- as-of filtering and validation

def race_start(schedule: pd.DataFrame, season: int, rnd: int) -> pd.Timestamp:
    row = schedule[(schedule["season"] == season) & (schedule["round"] == rnd)]
    assert len(row) == 1, f"{season} R{rnd} is not on the calendar"
    return row["start_utc"].iloc[0]


def races_before(schedule: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Calendar rows for races that started strictly before `as_of`."""
    return schedule[schedule["start_utc"] < as_of]


def restrict_to_as_of(table: pd.DataFrame, schedule: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Drop every row from a race at or after `as_of`: that race hasn't happened yet."""
    known = races_before(schedule, as_of)[["season", "round"]]
    return table.merge(known, on=["season", "round"], how="inner")


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
        assert race["driver_id"].is_unique, f"{season} R{rnd}: a driver appears twice"
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


def add_context(df: pd.DataFrame, data: RawData) -> pd.DataFrame:
    """Qualifying position, grid penalty, and flags for unusual-but-real situations."""
    quali = data.qualifying[["season", "round", "driver_id", "quali_position", "q1", "q2", "q3"]]
    # one_to_one: a duplicated qualifying row would otherwise silently duplicate a driver's result
    df = df.merge(quali, on=["season", "round", "driver_id"], how="left", validate="one_to_one")
    df = df.merge(data.schedule[["season", "round", "has_sprint"]], on=["season", "round"],
                  how="left", validate="many_to_one")

    # 2022 only: the sprint result, not qualifying, set the race grid
    df["sprint_sets_grid"] = (df["season"] == 2022) & df["has_sprint"]
    df["grid_penalty"] = (df["grid"] - df["quali_position"]).where(~df["sprint_sets_grid"] & ~df["pit_lane_start"])

    debuts = data.driver_debuts.set_index("driver_id")
    third_race = df["driver_id"].map(debuts["race_3_date"])
    # Fewer than 3 career starts before this race = this is one of the driver's first three races
    df["rookie"] = third_race.isna() | (df["date"] <= third_race)

    df["team"] = df["constructor_id"].replace(TEAM_LINEAGE)
    first_race = df.groupby(["season", "round"]).ngroup()
    df["_race_idx"] = first_race
    established = set(df.loc[df["_race_idx"] == 0, "team"])  # teams already racing before our data starts
    prior = df.groupby("team")["_race_idx"].transform(lambda s: s.rank(method="dense") - 1)
    df["new_team"] = ~df["team"].isin(established) & (prior < 3)
    return df.drop(columns=["_race_idx"])


# ---------------------------------------------------------------- entry point

def load_results(as_of: pd.Timestamp, from_snapshot: Path | None = None,
                 save_new_snapshot: bool = False, data: RawData | None = None) -> pd.DataFrame:
    """Clean results for every race that started strictly before `as_of`.

    as_of is required: the start of the race being predicted (or, for scoring, a moment after it).
    The returned table's .attrs record which source and cutoff were used.
    """
    data = data or load_raw(from_snapshot, save_new_snapshot)
    raw = restrict_to_as_of(data.results, data.schedule, as_of)
    validate_raw(raw, data.schedule, as_of)
    df = add_context(clean_results(raw, data.dsq_last_laps), data)
    df.attrs.update({"source": data.source, "as_of": as_of.isoformat()})
    return df
