import numpy as np
import pandas as pd
import pytest

from f1pred.simulate import SimSettings, pace_score, race_probabilities, simulate_positions


def _race(n=20, p_dnf=0.1):
    return pd.DataFrame({"season": 2026, "round": 1, "driver_id": [f"d{i}" for i in range(n)],
                         "pred": np.arange(n, dtype=float), "grid": np.arange(1, n + 1), "sim_dnf_prob": p_dnf})


def test_probabilities_add_up():
    sim = race_probabilities(_race(), SimSettings(n_sims=4000))
    assert sim["p_win"].sum() == pytest.approx(1.0)
    assert sim["p_podium"].sum() == pytest.approx(3.0)
    assert sim["p_points"].sum() == pytest.approx(10.0)
    assert sorted(sim["predicted_position"]) == list(range(1, 21))


def test_no_noise_no_dnfs_reproduces_the_order():
    sim = race_probabilities(_race(p_dnf=0.0), SimSettings(sigma=1e-9, n_sims=200))
    assert sim["p_win"].iloc[0] == 1.0 and (sim["p_win"].iloc[1:] == 0).all()
    assert list(sim["predicted_position"]) == list(range(1, 21))


def test_faster_driver_wins_more_often():
    sim = race_probabilities(_race(), SimSettings(n_sims=4000))
    assert sim["p_win"].is_monotonic_decreasing


def test_dnf_probability_is_respected():
    rng = np.random.default_rng(0)
    pos = simulate_positions(np.zeros(4), np.array([0.0, 0.5, 0.0, 0.0]), 0.1, 20000, rng)
    # driver 1 retires half the time, and a retirement is always classified behind every finisher
    assert (pos[:, 1] == 4).mean() == pytest.approx(0.5 + 0.5 * 0.25, abs=0.02)


def test_unreliable_car_is_penalised_only_through_dnfs():
    # Same pace, different reliability: the gap in P(win) comes from the DNF dice alone
    race = _race(n=2).assign(pred=[0.0, 0.0], grid=[1, 1], sim_dnf_prob=[0.0, 0.5])
    sim = race_probabilities(race, SimSettings(sigma=0.3, n_sims=40000))
    assert sim["p_win"].iloc[0] == pytest.approx(0.5 * 0.5 + 0.5, abs=0.02)


def test_grid_blend_moves_score_toward_grid():
    model_says_p2_wins = np.array([1.0, 0.0, 2.0])
    grid = np.array([1, 2, 3])
    assert pace_score(model_says_p2_wins, grid, 0.0)[0] > pace_score(model_says_p2_wins, grid, 0.0)[1]
    assert pace_score(model_says_p2_wins, grid, 1.0)[0] < pace_score(model_says_p2_wins, grid, 1.0)[1]


def test_same_seed_same_answer():
    a = race_probabilities(_race(), SimSettings(n_sims=1000, seed=7))
    b = race_probabilities(_race(), SimSettings(n_sims=1000, seed=7))
    pd.testing.assert_frame_equal(a, b)


def test_row_order_never_changes_pace():
    # Backtest rows arrive in finishing order; tied scores must not be broken by that order
    race = _race().assign(pred=np.repeat(np.arange(10.0), 2), grid=np.repeat(np.arange(1, 11), 2))
    pace = pd.Series(pace_score(race["pred"].to_numpy(), race["grid"].to_numpy(), 0.3), index=race["driver_id"])
    shuffled = race.sample(frac=1, random_state=3)
    pace_shuffled = pd.Series(pace_score(shuffled["pred"].to_numpy(), shuffled["grid"].to_numpy(), 0.3),
                              index=shuffled["driver_id"])
    pd.testing.assert_series_equal(pace, pace_shuffled.loc[pace.index])
    assert pace["d0"] == pace["d1"]  # tied drivers get equal pace, whichever row comes first


def test_log_spacing_widens_front_gaps():
    pace = pace_score(np.arange(20.0), np.arange(1, 21), 0.0, spacing="log")
    assert pace[1] - pace[0] > 5 * (pace[11] - pace[10])
    assert pace[0] == 0.0 and pace[-1] == pytest.approx(1.0)


def test_log_spacing_makes_pole_more_dominant():
    lin = race_probabilities(_race(p_dnf=0.0), SimSettings(sigma=0.1, n_sims=5000))
    log = race_probabilities(_race(p_dnf=0.0), SimSettings(sigma=0.1, n_sims=5000, spacing="log"))
    assert log["p_win"].iloc[0] > lin["p_win"].iloc[0]
