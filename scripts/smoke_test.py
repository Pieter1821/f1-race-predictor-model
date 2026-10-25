"""Check the environment: every library imports and FastF1 can fetch real data."""
from pathlib import Path

import fastf1
import lightgbm
import optuna
import pandas as pd
import shap
import sklearn
import streamlit
import xgboost
from fastf1.ergast import Ergast

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.configure(cache_dir=str(CACHE_DIR))

for lib in (fastf1, pd, sklearn, lightgbm, xgboost, optuna, shap, streamlit):
    print(f"{lib.__name__:<12} {lib.__version__}")

# Same data as the Postman request "Race result (one round)"
ergast = Ergast(result_type="pandas", auto_cast=True)
singapore = ergast.get_race_results(season=2026, round=17)
results = singapore.content[0]
print(f"\n{singapore.description.loc[0, 'raceName']}: {len(results)} drivers")
print(results[["position", "grid", "driverCode", "constructorName", "status"]].head(6).to_string(index=False))
