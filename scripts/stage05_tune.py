"""Stage 5: tune the residual LightGBM with Optuna, scored on walk-forward 2024 (dev) races only.

    python scripts/stage05_tune.py        # writes reports/stage05_lgbm_params.json
"""
import json

import optuna
import pandas as pd

from f1pred.backtest import DEV_SEASONS, summarise, walk_forward
from f1pred.config import PROCESSED_DIR, PROJECT_ROOT
from f1pred.models import LGBMRegressorModel

FEATURES = ["grid", "grid_pct", "quali_position", "grid_penalty", "quali_best_gap_pct", "no_quali_time",
            "teammate_quali_delta", "sprint_sets_grid", "team_form", "driver_level", "driver_recent_form",
            "driver_standing", "constructor_standing", "track_level", "team_dnf_rate", "driver_vs_teammate"]
N_TRIALS = 40
table = pd.read_parquet(PROCESSED_DIR / "features.parquet")


def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 20, 300, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 40, 300, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
    }
    preds = walk_forward(table, lambda: LGBMRegressorModel(residual=True, **params), FEATURES, DEV_SEASONS)
    return float(summarise(preds)["mae"])


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=N_TRIALS, callbacks=[
    lambda s, t: print(f"trial {t.number:>2}: MAE {t.value:.3f}  (best so far {s.best_value:.3f})", flush=True)])

out = PROJECT_ROOT / "reports" / "stage05_lgbm_params.json"
out.write_text(json.dumps({"tuned_on": DEV_SEASONS, "trials": N_TRIALS, "dev_mae": study.best_value,
                           "params": study.best_params}, indent=2))
print(f"\nbest dev MAE {study.best_value:.3f} with {study.best_params}\nsaved to {out}")
