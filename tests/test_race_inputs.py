import pandas as pd
import pytest

from f1pred.config import MANUAL_DIR
from f1pred.race_inputs import RaceInputError, load_race_inputs, read_manual_csv, validate_race_inputs

QUALI_START = pd.Timestamp("2026-10-24 21:00", tz="UTC")
AS_OF = pd.Timestamp("2026-10-25 20:00", tz="UTC")


def _real():
    return (read_manual_csv(MANUAL_DIR / "2026_R18_qualifying.csv"), read_manual_csv(MANUAL_DIR / "2026_R18_grid.csv"))


def _history_from(quali: pd.DataFrame) -> pd.DataFrame:
    """A 2026 history in which every driver last raced for the team the inputs say."""
    return quali[["driver_id", "constructor_id"]].assign(season=2026, round=17)


def _problems(quali, grid, history=None, quali_start=QUALI_START):
    history = _history_from(_real()[0]) if history is None else history
    return validate_race_inputs(quali, grid, history, 2026, quali_start, AS_OF)


def test_real_austin_inputs_are_consistent():
    quali, grid = _real()
    assert _problems(quali, grid) == []


def test_real_austin_inputs_load_with_penalties():
    quali, _ = _real()
    df = load_race_inputs(2026, 18, _history_from(quali), QUALI_START, AS_OF)
    assert len(df) == 22
    pen = df.set_index("driver_id")["grid_penalty"]
    assert pen["bortoleto"] == 9 and pen["colapinto"] == 7 and pen["arvid_lindblad"] == 5
    assert df.loc[df["driver_id"] == "antonelli", "q3"].iloc[0] == pytest.approx(94.035)


def test_missing_driver_is_caught():
    quali, grid = _real()
    problems = _problems(quali, grid.iloc[:-1])
    assert any("21 rows, expected 22" in p for p in problems)
    assert any("drivers differ" in p and "bortoleto" in p for p in problems)


def test_duplicate_driver_is_caught():
    quali, grid = _real()
    grid.loc[grid["driver_id"] == "bottas", "driver_id"] = "perez"
    assert any("driver listed twice" in p and "perez" in p for p in _problems(quali, grid))


def test_typo_in_driver_id_is_caught():
    quali, grid = _real()
    for df in (quali, grid):
        df.loc[df["driver_id"] == "hulkenberg", "driver_id"] = "hulkenburg"
    assert any("hulkenburg has no 2026 race" in p for p in _problems(quali, grid))


def test_wrong_team_is_caught_with_expected_and_observed():
    quali, grid = _real()
    grid.loc[grid["driver_id"] == "lawson", "constructor_id"] = "red_bull"
    assert any("lawson constructor_id='red_bull', but last raced for 'rb'" in p for p in _problems(quali, grid))


def test_q2_time_for_q1_eliminee_is_caught():
    quali, grid = _real()
    quali.loc[quali["driver_id"] == "bottas", "q2"] = "1:36.000"
    assert any("bottas is P22 (out in Q1) but has a Q2 time" in p for p in _problems(quali, grid))


def test_implausible_lap_time_is_caught():
    quali, grid = _real()
    quali.loc[quali["driver_id"] == "albon", "q1"] = "0:09.000"
    assert any("albon Q1" in p for p in _problems(quali, grid))


def test_qualifying_after_cutoff_is_caught():
    quali, grid = _real()
    assert any("not before the prediction cutoff" in p for p in _problems(quali, grid, quali_start=AS_OF))


def test_load_raises_with_every_problem_listed(tmp_path):
    quali, grid = _real()
    grid.iloc[:-1].to_csv(tmp_path / "2026_R18_grid.csv", index=False)
    quali.to_csv(tmp_path / "2026_R18_qualifying.csv", index=False)
    with pytest.raises(RaceInputError, match="failed 2 check"):
        load_race_inputs(2026, 18, _history_from(quali), QUALI_START, AS_OF, manual_dir=tmp_path)
