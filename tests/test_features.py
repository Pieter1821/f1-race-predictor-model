import numpy as np
import pandas as pd

from f1pred.features import prior_mean


def _history(n_races=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for rnd in range(1, n_races + 1):
        for team in ("ferrari", "haas"):
            for car in (1, 2):
                rows.append({"season": 2026, "round": rnd, "team": team,
                             "driver_id": f"{team}{car}", "dnf": float(rng.random() < 0.3)})
    return pd.DataFrame(rows)


def test_feature_never_uses_its_own_race_or_later():
    df = _history()
    feature = prior_mean(df, "team", "dnf")
    k = 4
    # Rewrite history from race k onward (as if those results were different): race k's feature must not move
    changed = df.copy()
    later = changed["round"] >= k
    changed.loc[later, "dnf"] = 1 - changed.loc[later, "dnf"]
    changed_feature = prior_mean(changed, "team", "dnf")
    at_k_and_before = df["round"] <= k
    pd.testing.assert_series_equal(feature[at_k_and_before], changed_feature[at_k_and_before])


def test_teammate_in_same_race_does_not_leak():
    df = pd.DataFrame({"season": 2026, "round": [1, 1, 2, 2], "team": "haas",
                       "driver_id": ["a", "b", "a", "b"], "dnf": [0.0, 1.0, 0.0, 0.0]})
    f = prior_mean(df, "team", "dnf", prior_value=0.5)
    # Race 1: no earlier races, so both cars get NaN (no history), not each other's result
    assert f.iloc[:2].isna().all()
    # Race 2: both cars see race 1's team average (0 + 1) / 2
    assert (f.iloc[2:] == 0.5).all()


def test_shrinkage_pulls_small_samples_toward_prior():
    df = pd.DataFrame({"season": 2026, "round": [1, 2], "team": "x", "driver_id": "a", "dnf": [1.0, 0.0]})
    f = prior_mean(df, "team", "dnf", prior_weight=10, prior_value=0.2)
    assert abs(f.iloc[1] - (1 + 10 * 0.2) / (1 + 10)) < 1e-12
