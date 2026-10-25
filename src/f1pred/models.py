"""Models that turn a race's feature rows into a finishing-order score (lower = further forward).

Every model has the same two methods, so the backtest can treat them alike:
    fit(train, features, weights) and predict(rows, features) -> score per row.
Learned preprocessing (imputation, scaling) lives inside the model and is fitted on its
training rows only.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "position_pct"  # official finishing position / starters: 0 = front, 1 = back
# Order among cars that finished: 0 = first finisher, 1 = last. "Pace if he finishes", no reliability
# baked in, for the simulation (which adds DNFs itself; using TARGET there would count DNF risk twice)
FINISHER_TARGET = "finisher_pct"


def add_finisher_target(table: pd.DataFrame) -> pd.DataFrame:
    fin = table["finished"].fillna(False).astype(bool) & table["position"].notna()
    rank = table[fin].groupby(["season", "round"])["position"].rank()
    n = table[fin].groupby(["season", "round"])["position"].transform("count")
    return table.assign(**{FINISHER_TARGET: (rank / n).reindex(table.index)})


class GridBaseline:
    name = "grid baseline"

    def fit(self, train, features, weights=None):
        return self

    def predict(self, rows, features):
        return rows["grid"].to_numpy(dtype=float)


class RidgeModel:
    def __init__(self, alpha: float = 1.0, name: str = "ridge", target: str = TARGET):
        self.alpha, self.name, self.target = alpha, name, target

    def fit(self, train, features, weights=None):
        self.pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=self.alpha))
        self.pipe.fit(train[features], train[self.target], ridge__sample_weight=weights)
        return self

    def predict(self, rows, features):
        return self.pipe.predict(rows[features])


LGBM_DEFAULTS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


class LGBMRegressorModel:
    """With residual=True the trees predict only the correction to the grid (places gained/lost as a
    share of the field), so they don't have to rebuild the smooth grid->finish line out of steps."""

    def __init__(self, name: str = "lightgbm", residual: bool = False, target: str = TARGET, **params):
        self.name, self.residual, self.target = name, residual, target
        self.params = {**LGBM_DEFAULTS, **params}

    def _offset(self, rows):
        return rows["grid_pct"].to_numpy() if self.residual else 0.0

    def fit(self, train, features, weights=None):
        self.model = lgb.LGBMRegressor(random_state=0, **self.params)
        self.model.fit(train[features], train[self.target] - self._offset(train), sample_weight=weights)
        return self

    def predict(self, rows, features):
        return self._offset(rows) + self.model.predict(rows[features])


class LGBMRankerModel:
    """LambdaRank: learns to order drivers within a race rather than to hit a number."""

    def __init__(self, name: str = "lambdarank", **params):
        self.name, self.params = name, {**LGBM_DEFAULTS, **params}

    def fit(self, train, features, weights=None):
        train = train.assign(_w=1.0 if weights is None else weights).sort_values(["season", "round"])
        groups = train.groupby(["season", "round"], sort=False).size().to_numpy()
        # Relevance: higher = better. Finishing order turned into 0..30 (lambdarank's label range)
        relevance = np.round((1 - train[TARGET]) * 30).astype(int)
        self.model = lgb.LGBMRanker(objective="lambdarank", random_state=0, **self.params)
        # every car in a race carries that race's recency weight
        self.model.fit(train[features], relevance, group=groups, sample_weight=train["_w"].to_numpy())
        return self

    def predict(self, rows, features):
        return -self.model.predict(rows[features])  # ranker: high = good; we want low = forward


class Ensemble:
    """Average of the members' within-race ranks (ranks make different models' scores comparable)."""

    def __init__(self, members, name: str = "ensemble"):
        self.members, self.name = members, name

    def fit(self, train, features, weights=None):
        for m in self.members:
            m.fit(train, features, weights)
        return self

    def predict(self, rows, features):
        ranks = [pd.Series(m.predict(rows, features), index=rows.index)
                 .groupby([rows["season"], rows["round"]]).rank() for m in self.members]
        return np.mean(ranks, axis=0)


# ---------------------------------------------------------------- the model chosen in Stage 5
# Chosen on 2024 walk-forward dev races only (rule: lowest dev MAE; within 0.02 -> simpler model),
# before the 2025-26 test races were scored. Recency weighting: uniform (weights didn't help on dev).
FINAL_FEATURES = ["grid", "grid_pct", "quali_position", "grid_penalty", "quali_best_gap_pct", "no_quali_time",
                  "teammate_quali_delta", "sprint_sets_grid", "team_form", "driver_level", "driver_recent_form",
                  "driver_standing", "constructor_standing", "track_level", "team_dnf_rate", "driver_vs_teammate"]
FINAL_WEIGHTING = None


def final_model(params_path=None, target: str = TARGET) -> Ensemble:
    import json

    from f1pred.config import PROJECT_ROOT
    path = params_path or PROJECT_ROOT / "reports" / "stage05_lgbm_params.json"
    params = json.loads(path.read_text())["params"]
    return Ensemble([RidgeModel(target=target), LGBMRegressorModel(residual=True, target=target, **params)],
                    name="ridge + tuned lightgbm")


class MeanScore(Ensemble):
    """Average of members' raw predictions. Unlike Ensemble (which averages ranks), this keeps the
    gaps: a clear favourite stands apart, a three-way fight stays close. Used by the simulation."""

    def predict(self, rows, features):
        return np.mean([m.predict(rows, features) for m in self.members], axis=0)


def final_sim_model(params_path=None, target: str = FINISHER_TARGET) -> MeanScore:
    """Pace model for the simulation (v3, chosen on 2023-24): finishers-only target, so DNF risk
    enters only through the simulation's DNF draws."""
    import json

    from f1pred.config import PROJECT_ROOT
    path = params_path or PROJECT_ROOT / "reports" / "stage05_lgbm_params.json"
    params = json.loads(path.read_text())["params"]
    return MeanScore([RidgeModel(target=target), LGBMRegressorModel(residual=True, target=target, **params)],
                     name="ridge + tuned lightgbm (mean score)")
