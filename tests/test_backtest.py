import numpy as np
import pandas as pd
import pytest

from f1pred.backtest import race_key, walk_forward
from f1pred.models import Ensemble, GridBaseline, LGBMRankerModel, LGBMRegressorModel, RidgeModel


def _table(seasons=(2024, 2025), rounds=8, cars=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for r in range(1, rounds + 1):
            grid = rng.permutation(cars) + 1
            finish = np.argsort(np.argsort(grid + rng.normal(0, 1.5, cars))) + 1
            for car, (g, f) in enumerate(zip(grid, finish)):
                rows.append({"season": s, "round": r, "driver_id": f"d{car}", "grid": int(g), "grid_pct": g / cars,
                             "x": rng.normal(), "position": int(f), "position_pct": f / cars, "is_target": False})
    return pd.DataFrame(rows)


class Spy:
    """Records what it was trained on for every prediction it makes."""
    seen: list = []

    def fit(self, train, features, weights=None):
        self.trained_on = set(race_key(train))
        return self

    def predict(self, rows, features):
        Spy.seen.append((set(race_key(rows)), self.trained_on))
        return rows["grid"].to_numpy(dtype=float)


def test_each_race_is_predicted_by_a_model_trained_only_on_earlier_races():
    Spy.seen = []
    walk_forward(_table(), Spy, ["grid"], test_seasons=[2025], min_train_races=1)
    assert len(Spy.seen) == 8
    for predicted, trained in Spy.seen:
        (key,) = predicted
        assert trained and max(trained) < key


def test_target_race_rows_are_never_trained_on_or_scored():
    t = _table()
    t.loc[(t["season"] == 2025) & (t["round"] == 8), ["is_target", "position_pct"]] = [True, np.nan]
    Spy.seen = []
    walk_forward(t, Spy, ["grid"], test_seasons=[2025], min_train_races=1)
    assert all(202508 not in trained and 202508 not in predicted for predicted, trained in Spy.seen)


def test_imputer_is_fitted_on_training_rows_only():
    t = _table()
    t.loc[t.index[:5], "x"] = np.nan
    train, test = t[t["season"] == 2024], t[t["season"] == 2025].assign(x=1000.0)  # extreme test values
    model = RidgeModel().fit(train, ["grid_pct", "x"])
    assert model.pipe.named_steps["simpleimputer"].statistics_[1] == pytest.approx(train["x"].median())


@pytest.mark.parametrize("make", [GridBaseline, RidgeModel, lambda: LGBMRegressorModel(residual=True, n_estimators=20),
                                  lambda: LGBMRankerModel(n_estimators=20),
                                  lambda: Ensemble([RidgeModel(), LGBMRegressorModel(n_estimators=20)])])
def test_every_model_gives_one_score_per_row(make):
    t = _table()
    train, test = t[t["season"] == 2024], t[t["season"] == 2025]
    pred = make().fit(train, ["grid", "grid_pct", "x"]).predict(test, ["grid", "grid_pct", "x"])
    assert len(pred) == len(test) and np.isfinite(pred).all()
