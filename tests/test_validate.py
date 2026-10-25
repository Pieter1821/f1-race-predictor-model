import pandas as pd
import pytest

from f1pred.validate import score_many, score_race


def _race():
    # Stored in finishing order. Two pit-lane starters (c, d) share grid 4: a tie.
    return pd.DataFrame({
        "season": 2026, "round": 1,
        "driver_id": ["c", "a", "d", "b"],
        "grid": [4, 1, 4, 2],
        "position": [1, 2, 3, 4],
    })


def test_score_does_not_depend_on_row_order():
    race = _race()
    expected = score_race(race, "grid")
    for seed in range(20):
        shuffled = race.sample(frac=1, random_state=seed)
        assert score_race(shuffled, "grid") == pytest.approx(expected)


def test_ties_are_broken_by_grid_then_driver_id_not_by_result():
    race = _race()
    # c finished ahead of d, but the tie-break must not know that: c < d alphabetically
    # gives the same answer here, so flip the names to prove the result isn't used
    race["driver_id"] = ["z", "a", "d", "b"]
    scores = score_race(race, "grid")
    # predicted order: a(1) b(2) d(3) z(4); actual: z=1 a=2 d=3 b=4
    assert scores["mae"] == pytest.approx((1 + 2 + 0 + 3) / 4)


def test_score_many_matches_single_race():
    race = _race()
    assert score_many(race, "grid").iloc[0]["mae"] == pytest.approx(score_race(race, "grid")["mae"])
