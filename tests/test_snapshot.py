"""Regression tests on the committed snapshots: offline, and they fail if data or pipeline drift."""
import numpy as np
import pandas as pd
import pytest

from f1pred.config import SNAPSHOT, TARGET_ROUND, TARGET_SEASON, WEATHER_SNAPSHOT
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.race_inputs import load_race_inputs
from f1pred.validate import score_many
from f1pred.weather import forecasts_as_of, read_forecasts


@pytest.fixture(scope="module")
def data():
    return read_snapshot(SNAPSHOT)


@pytest.fixture(scope="module")
def as_of(data):
    return race_start(data.schedule, TARGET_SEASON, TARGET_ROUND)


@pytest.fixture(scope="module")
def results(data, as_of):
    return load_results(as_of=as_of, data=data)


def test_snapshot_reproduces_dataset_and_baseline(results):
    assert results.groupby(["season", "round"]).ngroups == 109
    assert len(results) == 2193
    scores = score_many(results, "grid")[["mae", "spearman", "winner_hit", "podium_hit"]].mean()
    assert scores["mae"] == pytest.approx(3.411, abs=5e-4)
    assert scores["spearman"] == pytest.approx(0.635, abs=5e-4)
    assert scores["winner_hit"] == pytest.approx(63 / 109)
    assert scores["podium_hit"] == pytest.approx(0.670, abs=5e-4)


def test_loading_is_deterministic(data, as_of, results):
    pd.testing.assert_frame_equal(results, load_results(as_of=as_of, data=data))


def test_target_race_is_not_in_training_data(results):
    assert not ((results["season"] == TARGET_SEASON) & (results["round"] == TARGET_ROUND)).any()


def test_unusual_but_real_rows_are_kept(results):
    assert (results["status_bucket"] == "dsq").sum() == 10
    hamilton_austin_2023 = results[(results["season"] == 2023) & (results["round"] == 18)
                                   & (results["driver_id"] == "hamilton")].iloc[0]
    assert hamilton_austin_2023["position"] == 19 and hamilton_austin_2023["on_track_position"] == 2
    assert results["rookie"].sum() > 0 and results["new_team"].sum() > 0


def test_austin_inputs_match_real_2026_history(data, as_of, results):
    quali_start = data.schedule.set_index(["season", "round"]).loc[(TARGET_SEASON, TARGET_ROUND), "qualifying_start_utc"]
    inputs = load_race_inputs(TARGET_SEASON, TARGET_ROUND, results, quali_start, as_of)
    assert len(inputs) == 22
    assert inputs.set_index("driver_id").loc["lawson", "constructor_id"] == "rb"


def test_weather_snapshot_vintages(data, as_of):
    usable = forecasts_as_of(read_forecasts(WEATHER_SNAPSHOT), data.schedule, as_of)
    assert len(usable) == 110  # 109 past races + Austin's own pre-race forecast
    assert (usable["lead_hours"] == 24).all()
    austin = usable[(usable["season"] == TARGET_SEASON) & (usable["round"] == TARGET_ROUND)].iloc[0]
    assert austin["issued_at_or_before_utc"] == as_of - pd.DateOffset(hours=24)
    old = usable[usable["season"] <= 2023]
    assert old["forecast_rain_mm"].isna().all() and not old["forecast_has_rain_fields"].any()
    assert old["forecast_temp_c"].notna().all()
    assert np.isfinite(usable.loc[usable["season"] >= 2024, "forecast_rain_mm"]).all()


def test_rows_are_not_stored_in_finishing_order(results):
    # The cause of two leaks: rows in finishing order let any row-order tie-break see the result
    for _, race in results.groupby(["season", "round"]):
        assert list(race["driver_id"]) == sorted(race["driver_id"])
