import pandas as pd
import pytest

from f1pred.data import clean_results, status_bucket


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


def test_dsq_driver_returns_to_on_track_position():
    # Austin 2023: Hamilton crossed the line 2nd, Leclerc 6th; both disqualified
    order = ["max_verstappen", "norris", "sainz", "perez", "russell", "gasly", "hamilton", "leclerc"]
    raw = pd.DataFrame({
        "season": 2023, "round": 18, "driver_id": order,
        "grid": range(1, 9), "position": range(1, 9), "laps": 56,
        "status": ["Finished"] * 6 + ["Disqualified"] * 2,
    })
    dsq = pd.DataFrame({"season": 2023, "round": 18, "driver_id": ["hamilton", "leclerc"],
                        "last_lap": 56, "last_lap_position": [2, 6]})
    on_track = clean_results(raw, dsq).set_index("driver_id")["on_track_position"]
    assert on_track["hamilton"] == 2 and on_track["leclerc"] == 6
    assert on_track["norris"] == 3 and on_track["perez"] == 5  # pushed back by the DSQ'd cars
    assert sorted(on_track) == list(range(1, 9))
