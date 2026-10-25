"""Fetch fresh data from the APIs and save dated snapshots. The only script that writes snapshots.

    python scripts/build_snapshot.py
Then point config.SNAPSHOT / config.WEATHER_SNAPSHOT at the new folders.
"""
from f1pred.config import SEASONS, TARGET_ROUND, TARGET_SEASON
from f1pred.data import fetch_raw, race_start, save_snapshot, snapshot_stamp
from f1pred.weather import fetch_forecasts, save_forecasts

data = fetch_raw()
print(f"jolpica snapshot: {save_snapshot(data)}")

# Forecasts for every race up to and including the target: the target's own day-ahead forecast is a pre-race input
target_start = race_start(data.schedule, TARGET_SEASON, TARGET_ROUND)
races = data.schedule[data.schedule["start_utc"] <= target_start]
forecasts = fetch_forecasts(races)
name = f"openmeteo_previous_runs_{min(SEASONS)}-{max(SEASONS)}_{snapshot_stamp()}"
print(f"weather snapshot: {save_forecasts(forecasts, name)} ({len(forecasts)} races)")
print(forecasts.groupby("season")[["forecast_temp_c", "forecast_rain_mm", "forecast_wind_kmh"]].count()
      .rename(columns=lambda c: c + " (non-missing)").to_string())
