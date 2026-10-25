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

SEASONS = range(2022, 2027)  # ground-effect era (2022-25) + the 2026 regulations
TARGET_SEASON, TARGET_ROUND = 2026, 18  # United States GP, Austin

JOLPICA_URL = "https://api.jolpi.ca/ergast/f1"


def enable_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(CACHE_DIR))
