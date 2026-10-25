"""Paths and constants shared by every stage."""
from pathlib import Path

import fastf1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MANUAL_DIR = DATA_DIR / "manual"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# Cache lives on the Linux filesystem: SQLite on /mnt/c is slow and its file locking is unreliable
CACHE_DIR = Path.home() / ".cache" / "f1-race-predictor-model"

FIRST_SEASON = 2022  # start of the ground-effect era; history before this is a different kind of car
SEASONS = range(FIRST_SEASON, 2027)  # ground-effect era (2022-25) + the 2026 regulations
TARGET_SEASON, TARGET_ROUND = 2026, 18  # United States GP, Austin

JOLPICA_URL = "https://api.jolpi.ca/ergast/f1"

# The committed data snapshot the pipeline trains on (every prediction records this)
SNAPSHOT = DATA_DIR / "snapshots" / "jolpica_2022-2026_2026-10-25T0711Z"
WEATHER_SNAPSHOT = DATA_DIR / "snapshots" / "openmeteo_previous_runs_2022-2026_2026-10-25T0713Z"


def enable_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(CACHE_DIR))
