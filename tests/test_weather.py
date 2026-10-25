import numpy as np
import pandas as pd
import pytest

from f1pred.weather import forecasts_as_of, summarise_window

AUSTIN = pd.Timestamp("2026-10-25 20:00", tz="UTC")


def _window(rain):
    hours = pd.date_range(AUSTIN, periods=3, freq="h")
    return pd.DataFrame({"temperature_2m": [27.0, 28.0, 29.0], "precipitation": rain,
                         "precipitation_probability": [0, 10, 5], "wind_speed_10m": [10.0, 12.0, 14.0]},
                        index=hours)


def test_summary_records_vintage_and_lead_time():
    row = summarise_window(_window([0.0, 0.2, 0.0]), AUSTIN)
    assert row["lead_hours"] == 24
    assert row["issued_at_or_before_utc"] == AUSTIN - pd.DateOffset(hours=24)
    assert row["forecast_temp_c"] == pytest.approx(28.0)
    assert row["forecast_rain_mm"] == pytest.approx(0.2)
    assert row["forecast_has_rain_fields"]


def test_missing_rain_forecast_stays_missing_not_zero():
    # 2022-23 races: the archive has no day-ahead rain forecast. That must not become "0 mm, dry".
    row = summarise_window(_window([np.nan, np.nan, np.nan]), AUSTIN)
    assert np.isnan(row["forecast_rain_mm"]) and np.isnan(row["forecast_rain_prob_max"])
    assert not row["forecast_has_rain_fields"]


def test_partially_missing_window_is_treated_as_missing():
    row = summarise_window(_window([0.0, np.nan, 0.0]), AUSTIN)
    assert np.isnan(row["forecast_rain_mm"])


def _schedule():
    return pd.DataFrame({"season": 2026, "round": [17, 18, 19],
                         "start_utc": pd.to_datetime(["2026-10-11 12:00", "2026-10-25 20:00", "2026-11-01 20:00"], utc=True)})


def _forecasts(issued_r18):
    return pd.DataFrame({"season": 2026, "round": [17, 18, 19],
                         "issued_at_or_before_utc": pd.to_datetime(["2026-10-10 12:00", issued_r18, "2026-10-31 20:00"], utc=True)})


def test_as_of_keeps_target_race_forecast_but_not_later_races():
    usable = forecasts_as_of(_forecasts("2026-10-24 20:00"), _schedule(), AUSTIN)
    assert set(usable["round"]) == {17, 18}


def test_forecast_issued_after_cutoff_is_rejected():
    # e.g. someone swaps in a same-day (lead 0) or observed value for Austin
    with pytest.raises(AssertionError, match="issued at/after the cutoff"):
        forecasts_as_of(_forecasts("2026-10-25 20:00"), _schedule(), AUSTIN)
