import pandas as pd
import pytest

from f1pred.data import clean_results, races_before, restrict_to_as_of, status_bucket, validate_raw

AUSTIN_2026_START = pd.Timestamp("2026-10-25 20:00", tz="UTC")


@pytest.mark.parametrize("status, bucket", [
    ("Finished", "finished"), ("+1 Lap", "lapped"), ("+6 Laps", "lapped"), ("Lapped", "lapped"),
    ("Disqualified", "dsq"), ("Collision damage", "dnf_accident"), ("Gearbox", "dnf_mechanical"),
    ("Retired", "dnf_unknown"), ("Did not start", "did_not_start"),
])
def test_status_bucket(status, bucket):
    assert status_bucket(status) == bucket


def test_unknown_status_fails_loudly():
    with pytest.raises(ValueError):
        status_bucket("Alien abduction")


def _race(season, rnd, drivers, laps, status, n_laps=56):
    return pd.DataFrame({
        "season": season, "round": rnd, "driver_id": drivers,
        "grid": range(1, len(drivers) + 1), "position": range(1, len(drivers) + 1),
        "laps": laps, "status": status,
    })


def test_full_distance_dsq_returns_to_crossing_position():
    # Austin 2023: Hamilton crossed the line 2nd, Leclerc 6th; both disqualified afterwards
    order = ["max_verstappen", "norris", "sainz", "perez", "russell", "gasly", "hamilton", "leclerc"]
    raw = _race(2023, 18, order, 56, ["Finished"] * 6 + ["Disqualified"] * 2)
    raw.loc[raw["driver_id"].isin(["hamilton", "leclerc"]), "laps"] = 0  # Jolpica zeroes DSQ laps
    dsq = pd.DataFrame({"season": 2023, "round": 18, "driver_id": ["hamilton", "leclerc"],
                        "last_lap": 56, "last_lap_position": [2, 6]})
    on_track = clean_results(raw, dsq).set_index("driver_id")["on_track_position"]
    assert on_track["hamilton"] == 2 and on_track["leclerc"] == 6
    assert on_track["norris"] == 3 and on_track["perez"] == 5  # pushed back by the DSQ'd cars
    assert sorted(on_track) == list(range(1, 9))


def test_mid_race_dsq_is_ordered_like_a_retirement():
    # Black-flagged on lap 30 while running P5: must land behind everyone who did more laps
    drivers = ["a", "b", "c", "d", "e", "hul"]
    raw = _race(2024, 21, drivers, [69, 69, 69, 45, 12, 0],
                ["Finished", "Finished", "Finished", "Retired", "Retired", "Disqualified"])
    dsq = pd.DataFrame({"season": 2024, "round": 21, "driver_id": ["hul"],
                        "last_lap": 30, "last_lap_position": [5]})
    on_track = clean_results(raw, dsq).set_index("driver_id")["on_track_position"]
    assert on_track["hul"] == 5  # behind d (45 laps), ahead of e (12 laps)
    assert on_track["e"] == 6


def _schedule():
    return pd.DataFrame({
        "season": 2026, "round": [17, 18, 19], "race_name": ["Singapore", "United States", "Mexico City"],
        "start_utc": pd.to_datetime(["2026-10-11 12:00", "2026-10-25 20:00", "2026-11-01 20:00"], utc=True),
    })


def test_as_of_austin_start_never_includes_austin():
    drivers = [f"d{i}" for i in range(22)]
    raw = pd.concat([_race(2026, r, drivers, 50, "Finished") for r in (17, 18)], ignore_index=True)
    kept = restrict_to_as_of(raw, _schedule(), AUSTIN_2026_START)
    assert set(kept["round"]) == {17}


def test_calendar_check_uses_as_of_not_the_clock():
    # Even after Austin has run (result posted or not), a load as of Austin's start expects 17 races
    assert len(races_before(_schedule(), AUSTIN_2026_START)) == 1  # only R17 in this mini-calendar


def test_entry_count_is_season_aware():
    sched = _schedule()
    short = _race(2026, 17, [f"d{i}" for i in range(20)], 50, "Finished")  # 2026 needs 22
    with pytest.raises(AssertionError, match="expected 22"):
        validate_raw(short, sched[sched["round"] == 17], AUSTIN_2026_START)
