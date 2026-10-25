"""Load race results from Jolpica (through FastF1's Ergast client) into one tidy, validated table."""
import re
from datetime import date, datetime, timezone

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

# Races where only 19 cars were entered, with the reason
KNOWN_SHORT_FIELDS = {
    (2024, 3): "Sargeant withdrawn, his car given to Albon",
    (2025, 9): "Stroll withdrew injured",
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


def load_season_results(season: int) -> pd.DataFrame:
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


def completed_races_on_calendar(season: int, now: datetime | None = None) -> int:
    """How many of the season's races had started by `now`, according to the official calendar."""
    now = now or datetime.now(timezone.utc)
    schedule = _ergast().get_race_schedule(season=season)
    starts = pd.to_datetime(
        schedule["raceDate"].astype(str) + " " + schedule["raceTime"].astype(str).fillna("00:00:00"), utc=True
    )
    return int((starts < now).sum())


def load_dsq_last_laps(raw: pd.DataFrame) -> pd.DataFrame:
    """For each disqualified driver: the position they held on their last lap (where they crossed the line)."""
    rows = []
    for _, r in raw[raw["status"] == "Disqualified"].iterrows():
        laps = _ergast().get_lap_times(season=r["season"], round=r["round"], driver=r["driver_id"]).content
        if not laps or laps[0].empty:
            continue
        last = laps[0].iloc[-1]
        rows.append({"season": r["season"], "round": r["round"], "driver_id": r["driver_id"],
                     "last_lap": int(last["number"]), "last_lap_position": int(last["position"])})
    return pd.DataFrame(rows, columns=["season", "round", "driver_id", "last_lap", "last_lap_position"])


def save_snapshot(raw: pd.DataFrame, dsq: pd.DataFrame, seasons=SEASONS) -> None:
    """Dated copy of exactly what the API returned, so results can be reproduced if the API changes."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{min(seasons)}-{max(seasons)}_{date.today().isoformat()}"
    raw.to_csv(SNAPSHOT_DIR / f"jolpica_results_{tag}.csv", index=False)
    dsq.to_csv(SNAPSHOT_DIR / f"jolpica_dsq_last_laps_{tag}.csv", index=False)


def validate_raw(raw: pd.DataFrame, seasons=SEASONS) -> None:
    for season in seasons:
        loaded = raw.loc[raw["season"] == season, "round"].nunique()
        expected = completed_races_on_calendar(season)
        assert loaded == expected, f"{season}: loaded {loaded} races, calendar says {expected} have run"
    for (season, rnd), race in raw.groupby(["season", "round"]):
        entries = len(race)
        if (season, rnd) in KNOWN_SHORT_FIELDS:
            assert entries == 19, f"{season} R{rnd}: expected the documented 19-car field, got {entries}"
        else:
            assert entries in (20, 22), f"{season} R{rnd}: {entries} entries, expected 20 or 22"
        _assert_positions_complete(race["position"], f"{season} R{rnd} (raw)")


def _assert_positions_complete(positions: pd.Series, label: str) -> None:
    assert sorted(positions) == list(range(1, len(positions) + 1)), f"{label}: positions are not 1..N"


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
    """Official order, except disqualified drivers go back to where they crossed the line.

    The official target stays `position`; this is only for history features
    (a DSQ'd P2 still shows the car's real pace).
    """
    out = df["position"].copy()
    dsq = dsq_last_laps.set_index(["season", "round", "driver_id"])["last_lap_position"]
    for (season, rnd), race in df.groupby(["season", "round"]):
        dsq_here = {d: dsq[(season, rnd, d)] for d in race["driver_id"] if (season, rnd, d) in dsq.index}
        if not dsq_here:
            continue
        others = race[~race["driver_id"].isin(dsq_here)].sort_values("position")["driver_id"].tolist()
        order = others[:]
        for driver, pos in sorted(dsq_here.items(), key=lambda kv: kv[1]):
            order.insert(min(pos, len(order) + 1) - 1, driver)
        new_pos = {d: i + 1 for i, d in enumerate(order)}
        out.loc[race.index] = race["driver_id"].map(new_pos).values
    return out.astype(int)


def load_results(seasons=SEASONS, snapshot: bool = False) -> pd.DataFrame:
    raw = pd.concat([load_season_results(s) for s in seasons], ignore_index=True)
    validate_raw(raw, seasons)
    dsq = load_dsq_last_laps(raw)
    if snapshot:
        save_snapshot(raw, dsq, seasons)
    return clean_results(raw, dsq)
