"""Race-day weather *forecasts*, as they stood a fixed time before each race.

Source: Open-Meteo Previous Runs API. A variable with suffix `_previous_day1` is the value
that model run issued 24 h before the valid time predicted. Using the same lead time for
every historical race and for the live prediction means the model never trains on fresher
information than it gets on race day.

Coverage (checked 25 Oct 2026, GFS): temperature from 2021, rain and wind from 2024 only.
Missing values stay missing (with a flag); observed weather is never substituted.
"""
import time as clock
from pathlib import Path

import pandas as pd
import requests

from f1pred.config import DATA_DIR

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_seamless"
LEAD_DAYS = 1
VARIABLES = ["temperature_2m", "precipitation", "precipitation_probability", "wind_speed_10m"]
RACE_WINDOW_HOURS = 3  # start hour and the two after it: roughly one race distance

WEATHER_SNAPSHOT_DIR = DATA_DIR / "snapshots"
COLUMNS = ["season", "round", "valid_from_utc", "issued_at_or_before_utc", "lead_hours", "model", "source",
           "forecast_temp_c", "forecast_rain_mm", "forecast_rain_prob_max", "forecast_wind_kmh",
           "forecast_has_rain_fields"]


def fetch_race_forecast(lat: float, lon: float, start_utc: pd.Timestamp) -> dict:
    """Day-ahead forecast for the race window at one circuit."""
    first_hour = start_utc.floor("h")
    hours = pd.date_range(first_hour, periods=RACE_WINDOW_HOURS, freq="h")
    response = requests.get(PREVIOUS_RUNS_URL, timeout=30, params={
        "latitude": lat, "longitude": lon, "timezone": "UTC", "models": MODEL,
        "hourly": ",".join(f"{v}_previous_day{LEAD_DAYS}" for v in VARIABLES),
        "start_date": hours[0].date().isoformat(), "end_date": hours[-1].date().isoformat(),
    })
    response.raise_for_status()
    hourly = pd.DataFrame(response.json()["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
    window = hourly.set_index("time").reindex(hours)
    window.columns = [c.replace(f"_previous_day{LEAD_DAYS}", "") for c in window.columns]
    return summarise_window(window, first_hour)


def summarise_window(window: pd.DataFrame, first_hour: pd.Timestamp) -> dict:
    """Collapse hourly forecasts into one row. NaN stays NaN: min_count stops sum() turning 'no data' into 0 mm."""
    has_rain = bool(window["precipitation"].notna().all())
    return {
        "valid_from_utc": first_hour,
        # A day-1 value was issued by a run at least 24 h before the hour it describes
        "issued_at_or_before_utc": first_hour - pd.DateOffset(hours=24 * LEAD_DAYS),
        "lead_hours": 24 * LEAD_DAYS,
        "model": MODEL,
        "source": "open-meteo previous-runs",
        "forecast_temp_c": window["temperature_2m"].mean(),
        "forecast_rain_mm": window["precipitation"].sum(min_count=len(window)) if has_rain else float("nan"),
        "forecast_rain_prob_max": window["precipitation_probability"].max() if has_rain else float("nan"),
        "forecast_wind_kmh": window["wind_speed_10m"].mean(),
        "forecast_has_rain_fields": has_rain,
    }


def fetch_forecasts(schedule: pd.DataFrame, pause_s: float = 0.2) -> pd.DataFrame:
    rows = []
    for _, race in schedule.iterrows():
        rows.append({"season": race["season"], "round": race["round"],
                     **fetch_race_forecast(race["lat"], race["lon"], race["start_utc"])})
        clock.sleep(pause_s)  # be polite to a free API
    return pd.DataFrame(rows, columns=COLUMNS)


def save_forecasts(forecasts: pd.DataFrame, name: str) -> Path:
    path = WEATHER_SNAPSHOT_DIR / name
    path.mkdir(parents=True, exist_ok=False)  # snapshots are never edited in place
    forecasts.to_csv(path / "forecasts.csv", index=False)
    return path


def read_forecasts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path) / "forecasts.csv")
    for col in ("valid_from_utc", "issued_at_or_before_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df


def forecasts_as_of(forecasts: pd.DataFrame, schedule: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Forecasts usable at `as_of`: for races up to and including the one starting at as_of,
    and only vintages issued before as_of. Fails loudly if a later vintage sneaks in."""
    races = schedule.loc[schedule["start_utc"] <= as_of, ["season", "round"]]
    usable = forecasts.merge(races, on=["season", "round"], how="inner", validate="one_to_one")
    late = usable[usable["issued_at_or_before_utc"] >= as_of]
    assert late.empty, f"forecasts issued at/after the cutoff {as_of}: {late[['season', 'round']].values.tolist()}"
    return usable
