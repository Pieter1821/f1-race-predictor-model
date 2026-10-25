"""Stage 5: walk-forward model comparison.

    python scripts/stage05_train.py dev     # choices: feature groups, models, weights (2024 races)
    python scripts/stage05_train.py test    # the locked choice vs references on 2025-26, run once
"""
import sys
import time

import pandas as pd

from f1pred.backtest import DEV_SEASONS, TEST_SEASONS, summarise, walk_forward
from f1pred.config import PROCESSED_DIR
from f1pred.models import (FINAL_FEATURES, FINAL_WEIGHTING, Ensemble, GridBaseline, LGBMRankerModel,
                           LGBMRegressorModel, RidgeModel, final_model)
from f1pred.validate import score_many
from f1pred.weights import CANDIDATES

pd.set_option("display.width", 160)

GRID = ["grid", "grid_pct"]
QUALI = ["quali_position", "grid_penalty", "quali_best_gap_pct", "no_quali_time", "teammate_quali_delta",
         "sprint_sets_grid"]
FORM = ["team_form", "driver_level", "driver_recent_form", "driver_standing", "constructor_standing"]
EXTRA = ["track_level", "team_dnf_rate", "driver_vs_teammate"]
ALL = GRID + QUALI + FORM + EXTRA

table = pd.read_parquet(PROCESSED_DIR / "features.parquet")
phase = sys.argv[1] if sys.argv[1:] in (["dev"], ["test"]) else sys.exit("usage: stage05_train.py dev|test")
seasons = DEV_SEASONS if phase == "dev" else TEST_SEASONS


def run(label, make_model, features, weighting=None):
    t = time.time()
    s = summarise(walk_forward(table, make_model, features, seasons, weighting))
    rows.append({"step": label, **s.round(3).to_dict(), "secs": round(time.time() - t, 1)})
    print(f"  {label:<44} MAE {s['mae']:.3f}  Spearman {s['spearman']:.3f}  "
          f"winner {s['winner_hit']:.0%}  podium {s['podium_hit']:.0%}", flush=True)


rows = []
if phase == "test":
    print(f"TEST: walk-forward on {seasons}, choice locked on {DEV_SEASONS}\n")
    run("grid baseline (finish = grid)", GridBaseline, GRID)
    run("reference: ridge, grid + quali + form", RidgeModel, GRID + QUALI + FORM)
    final = walk_forward(table, final_model, FINAL_FEATURES, seasons, FINAL_WEIGHTING)
    final.to_parquet(PROCESSED_DIR / "stage05_test_predictions.parquet", index=False)
    s = summarise(final)
    rows.append({"step": "CHOSEN: ridge + tuned lightgbm", **s.round(3).to_dict()})
    print(f"  {'CHOSEN: ridge + tuned lightgbm':<44} MAE {s['mae']:.3f}  Spearman {s['spearman']:.3f}  "
          f"winner {s['winner_hit']:.0%}  podium {s['podium_hit']:.0%}")
    per_season = score_many(final, "pred").groupby("season")[["mae", "spearman", "winner_hit", "podium_hit"]].mean()
    base = score_many(final.assign(pred=final["grid"]), "pred").groupby("season")[["mae"]].mean()
    print("\nper season (chosen model vs grid MAE):")
    print(per_season.join(base.rename(columns={"mae": "grid_mae"})).round(3).to_string())
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "stage05_test.csv", index=False)
    sys.exit()

print(f"Walk-forward on {seasons} ({table[table['season'].isin(seasons)].groupby(['season', 'round']).ngroups} races)\n")
print("1. Linear model, adding feature groups:")
run("grid baseline (finish = grid)", GridBaseline, GRID)
run("ridge: grid", RidgeModel, GRID)
run("ridge: grid + quali", RidgeModel, GRID + QUALI)
run("ridge: grid + quali + form", RidgeModel, GRID + QUALI + FORM)
run("ridge: all features", RidgeModel, ALL)

print("\n2. Stronger models (all features):")
run("lightgbm regressor", LGBMRegressorModel, ALL)
run("lightgbm lambdarank", LGBMRankerModel, ALL)
run("ensemble: ridge + lightgbm + lambdarank", lambda: Ensemble([RidgeModel(), LGBMRegressorModel(), LGBMRankerModel()]), ALL)

print("\n3. Recency weights (ridge, all features):")
for name, params in CANDIDATES.items():
    run(f"ridge, weights={name}", RidgeModel, ALL, params)

pd.DataFrame(rows).to_csv(PROCESSED_DIR / "stage05_dev.csv", index=False)
