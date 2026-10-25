"""Why does the 2026 Austin polesitter (Antonelli) get 48.6% to win, and is that a 2026-specific bias?

    python scripts/investigate_austin_pole.py
Reads the same snapshot and inputs as the official prediction; changes nothing.
"""
import numpy as np
import pandas as pd
import shap

from f1pred.config import PROCESSED_DIR
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.features import append_target_race, build_features
from f1pred.models import FINAL_FEATURES, FINISHER_TARGET, add_finisher_target, final_sim_model
from f1pred.pipeline import race_inputs
from f1pred.simulate import add_dnf_probability

pd.set_option("display.width", 170)
SNAP = "data/snapshots/jolpica_2022-2026_2026-10-25T0831Z"
data = read_snapshot(SNAP)
as_of = race_start(data.schedule, 2026, 18)
history = load_results(as_of=as_of, data=data)
inputs, _ = race_inputs(2026, 18, data, history, as_of, backtest=False)
table = add_dnf_probability(add_finisher_target(build_features(
    append_target_race(history, inputs, "americas", as_of.normalize().tz_localize(None)))))
train = table[~table["is_target"] & table[FINISHER_TARGET].notna()].sort_values(["season", "round", "driver_id"])
pair = table[table["is_target"] & table["driver_id"].isin(["antonelli", "max_verstappen"])].set_index("code")

# ---------------------------------------------------------------- 1. why Verstappen gets 33%
pace = final_sim_model().fit(train, FINAL_FEATURES)
ridge, lgbm = pace.members
X = pair[FINAL_FEATURES]
print("1. The simulation's pace score (lower = faster; 0 = typical winner territory)")
print(f"   ANT {pace.predict(X.assign(season=2026, round=18), FINAL_FEATURES)[0]:.4f} | "
      f"VER {pace.predict(X.assign(season=2026, round=18), FINAL_FEATURES)[1]:.4f}\n")

imp, scaler, rr = (ridge.pipe.named_steps[k] for k in ("simpleimputer", "standardscaler", "ridge"))
ridge_contrib = pd.DataFrame(rr.coef_ * (scaler.transform(imp.transform(X)) - 0), index=X.index, columns=FINAL_FEATURES).T
explainer = shap.TreeExplainer(lgbm.model)
tree_contrib = pd.DataFrame(explainer.shap_values(X), index=X.index, columns=FINAL_FEATURES).T

both = pd.DataFrame({
    "ANT value": X.loc["ANT"], "VER value": X.loc["VER"],
    # the pace score averages ridge and grid+trees, so each part counts half
    "ridge: ANT-VER": (ridge_contrib["ANT"] - ridge_contrib["VER"]) / 2,
    "trees: ANT-VER": (tree_contrib["ANT"] - tree_contrib["VER"]) / 2,
})
both["total: ANT-VER"] = both["ridge: ANT-VER"] + both["trees: ANT-VER"]
both.loc["grid_pct (trees' starting point)", "total: ANT-VER"] = (X.loc["ANT", "grid_pct"] - X.loc["VER", "grid_pct"]) / 2
print("Feature by feature: negative = helps ANT relative to VER, positive = helps VER")
print(both.sort_values("total: ANT-VER", key=abs, ascending=False).round(4).to_string(), "\n")

old = ["driver_level", "track_level"]
from_history = both.loc[old, "total: ANT-VER"].sum()
gap = both["total: ANT-VER"].sum()
print(f"   pace gap ANT-VER {gap:+.4f}; of which driver_level + track_level (2022-25 history) {from_history:+.4f}\n")

for code, drv in (("ANT", "antonelli"), ("VER", "max_verstappen")):
    h = history[history["driver_id"] == drv]
    a = h[h["circuit_id"] == "americas"]
    print(f"   {code}: {len(h)} races since 2022, {int((h['position'] == 1).sum())} wins "
          f"({int((h[h['season'] == 2026]['position'] == 1).sum())} in 2026); at Austin: "
          + ", ".join(f"{s}: P{int(p)}" for s, p in zip(a["season"], a["position"])))

# ---------------------------------------------------------------- 2. were 2026 polesitters under-rated?
sim = pd.read_parquet(PROCESSED_DIR / "stage07_sim_v3_2025_26.parquet")
pole = sim[(sim["season"] == 2026) & (sim["grid"] == 1)].copy()
pole["won"] = pole["position"] == 1
pole["dnf"] = ~pole["finished"].astype(bool)
fin = pole[~pole["dnf"]]
print("\n2. 2026 polesitters, walk-forward predictions (each made before its race)")
print(f"   P(win): predicted {pole['p_win'].mean():.1%} (before win calibration {pole['p_win_raw'].mean():.1%}) "
      f"vs actual {pole['won'].mean():.1%} ({int(pole['won'].sum())}/{len(pole)})")
print(f"   P(win | finished): predicted {(fin['p_win'] / (1 - fin['p_dnf'])).mean():.1%} "
      f"vs actual {fin['won'].mean():.1%} ({int(fin['won'].sum())}/{len(fin)})")
others = sim[(sim["season"] == 2026) & (sim["grid"] == 2)]
print(f"   P2 starters: predicted {others['p_win'].mean():.1%} vs actual {(others['position'] == 1).mean():.1%}")
for yr in (2025,):
    p = sim[(sim["season"] == yr) & (sim["grid"] == 1)]
    print(f"   {yr} polesitters for comparison: predicted {p['p_win'].mean():.1%} vs actual {(p['position'] == 1).mean():.1%}")
rng = np.random.default_rng(0)
gaps = [rng.choice((pole["won"] - pole["p_win"]).values, len(pole)).mean() for _ in range(10000)]
print(f"   actual - predicted = {(pole['won'] - pole['p_win']).mean():+.1%}, "
      f"95% CI [{np.percentile(gaps, 2.5):+.1%}, {np.percentile(gaps, 97.5):+.1%}] over the 17 races")
print(pole[["round", "race_name", "code", "p_win", "p_dnf", "position"]].assign(
    p_win=lambda d: (d["p_win"] * 100).round(1)).to_string(index=False))
