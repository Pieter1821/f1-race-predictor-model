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


# =====================================================================================================
# Tunable settings. Values here ARE the locked model (they match reports/*.json, enforced by a test).
# Change one at a time with scripts/try_setting.py (dev seasons only) and keep a change only if the
# gain beats the noise. Each line: what it does | too low | too high.
# =====================================================================================================

# --- Validation periods
DEV_SEASONS = [2024]               # races used to make choices | too few: choices fit luck | too many: no test left
SIM_DEV_SEASONS = [2023, 2024]     # simulation choices (one season was too narrow for "how dominant is pole")
TEST_SEASONS = [2025, 2026]        # reported once, never used for choices

# --- Feature shrinkage: estimate = (sum + k * prior) / (count + k); k = "how many races the prior is worth"
SHRINK_TEAM_ALL = 3        # team's all-time form -> league average | low: noisy | high: every team looks average
SHRINK_TEAM_SEASON = 4     # team's season form -> its all-time form | low: overreacts early season | high: slow to see change
SHRINK_DRIVER = 5          # driver level -> team form | low: rookies judged on 2 races | high: drivers look like their car
SHRINK_TEAMMATE = 5        # driver vs teammate -> level | low: one bad weekend sticks | high: signal vanishes
SHRINK_TRACK = 2           # track history -> driver level | low: one race at a track dominates | high: track history ignored
SHRINK_DNF_TEAM = 10       # team DNF feature -> season rate | low: luck looks like unreliability | high: all teams equal
SHRINK_DNF_SEASON = 44     # season DNF rate -> all-time rate (~2 races x 22 cars) | low: early-season noise | high: slow to see a new era
FORM_WINDOW = 5            # races in driver_recent_form | low: noisy | high: stale, same as driver_level
QUALI_GAP_CAP = 5.0        # % cap on the qualifying gap | low: hides real gaps | high: failed sessions look like pace

FEATURES = ["grid", "grid_pct", "quali_position", "grid_penalty", "quali_best_gap_pct", "no_quali_time",
            "teammate_quali_delta", "sprint_sets_grid", "team_form", "driver_level", "driver_recent_form",
            "driver_standing", "constructor_standing", "track_level", "team_dnf_rate", "driver_vs_teammate"]

# --- Order model
RECENCY_WEIGHTING = None   # None or a weights.CANDIDATES entry | strong decay: forgets useful history
RIDGE_ALPHA = 1.0          # ridge's penalty on big weights | low: overfits overlapping features | high: ignores features
LGBM_PARAMS_FILE = PROJECT_ROOT / "reports" / "stage05_lgbm_params.json"  # tuned on DEV_SEASONS by Optuna
ENSEMBLE_MEMBERS = ["ridge", "lightgbm"]  # averaged models | one: no error-cancelling | many: slower, little gain

# --- Simulation (reports/sim_settings.json records why)
SIM_TARGET = "finisher_pct"   # pace model's target: order among finishers (DNFs only via the dice)
SIM_SPACING = "score"         # "score" = model's own gaps | "linear"/"log" = rank-based shapes
SIM_SIGMA = 0.12              # race-day randomness | low: favourites too certain | high: everyone equal
SIM_GRID_BLEND = 0.0          # share of pace taken from the grid | high: model ignored
SIM_POLE_BONUS = 0.0          # extra pace for the polesitter (start/track-position effect) | 0 = off (locked)
SIM_WIN_TEMPERATURE = 1.75    # p_win ** a, renormalised | low: favourites under-rated | high: upsets get ~0%
DNF_PRIOR_WEIGHT = 80         # team DNF dice -> season rate (car-starts) | low: luck = unreliable | high: all equal
N_SIMS = 20_000               # simulated races | low: noisy probabilities | high: slower, no extra accuracy
SEED = 0                      # random seed for LightGBM's row/column sampling (vary it to measure the noise floor)


def enable_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.configure(cache_dir=str(CACHE_DIR))
