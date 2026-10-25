"""VARIANT prediction for 2026 Austin (not official): recent-weighted history + win calibration incl. 2025-26.

    python scripts/experiments/predict_austin_variant.py
See src/f1pred/variant.py for why, and why it can't be validated cleanly.
"""
import json
from datetime import datetime, timezone

import pandas as pd

from f1pred.backtest import walk_forward
from f1pred.config import PREDICTIONS_DIR, PROJECT_ROOT
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.features import append_target_race, build_features
from f1pred.models import FINAL_FEATURES, FINISHER_TARGET, add_finisher_target, final_model, final_sim_model
from f1pred.pipeline import prediction_table, race_inputs
from f1pred.simulate import SimSettings, add_dnf_probability, race_probabilities, simulate_many, temper_win
from f1pred.validate import predicted_rank
from f1pred.variant import SEASON_DECAY, apply_variant_features

SNAP = "data/snapshots/jolpica_2022-2026_2026-10-25T0831Z"
s = json.loads((PROJECT_ROOT / "reports" / "sim_settings.json").read_text())
settings = SimSettings(grid_blend=s["grid_blend"], sigma=s["sigma"], spacing=s["spacing"])

data = read_snapshot(SNAP)
as_of = race_start(data.schedule, 2026, 18)
history = load_results(as_of=as_of, data=data)
inputs, inputs_source = race_inputs(2026, 18, data, history, as_of, backtest=False)
table = build_features(append_target_race(history, inputs, "americas", as_of.normalize().tz_localize(None)))
table = add_dnf_probability(add_finisher_target(apply_variant_features(table)))

# 1. Win calibration refitted on 2023-26 walk-forward simulations of the variant
preds = walk_forward(table, final_sim_model, FINAL_FEATURES, [2023, 2024, 2025, 2026], target=FINISHER_TARGET)
sims = simulate_many(preds, SimSettings(settings.grid_blend, settings.sigma, settings.spacing, n_sims=4000))


def win_brier(a):
    t = temper_win(sims, a)
    return float(((t["p_win"] - (t["position"] == 1)) ** 2).mean())


grid_a = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
briers = {a: win_brier(a) for a in grid_a}
a_var = min(briers, key=briers.get)
print("win calibration, variant, fitted on 2023-26:", {a: round(b, 5) for a, b in briers.items()}, "-> a =", a_var)
cal = temper_win(sims, a_var)
for yr in (2025, 2026):
    p = cal[(cal["season"] == yr) & (cal["grid"] == 1)]
    print(f"  {yr} polesitters (in-sample, not a clean test): predicted {p['p_win'].mean():.1%} "
          f"vs actual {(p['position'] == 1).mean():.1%}")

# 2. Predict Austin with the variant
train = table[~table["is_target"] & table["position"].notna()].sort_values(["season", "round", "driver_id"])
target = table[table["is_target"]].sort_values("driver_id").copy()
order = final_model().fit(train, FINAL_FEATURES)
target["order_score"] = order.predict(target, FINAL_FEATURES)
target["predicted_position"] = predicted_rank(target, "order_score")
pace = final_sim_model().fit(train[train[FINISHER_TARGET].notna()], FINAL_FEATURES)
target["pred"] = pace.predict(target, FINAL_FEATURES)
sim = temper_win(race_probabilities(target, SimSettings(settings.grid_blend, settings.sigma, settings.spacing,
                                                        n_sims=20_000)), a_var)
sim.attrs.update({"sim_version": f"3.1-VARIANT (season decay {SEASON_DECAY}, win a={a_var} fitted 2023-26)",
                  "train_races": train.groupby(["season", "round"]).ngroups})
out = prediction_table(sim, "United States Grand Prix", as_of, inputs_source, f"snapshot {SNAP.split('/')[-1]}")
out["model"] = "VARIANT (not official): " + out["model"]

now = datetime.now(timezone.utc)
path = PREDICTIONS_DIR / f"2026_R18_variant_{now:%Y%m%dT%H%M%SZ}.csv"
out.to_csv(path, index=False)
official = pd.read_csv(sorted(PREDICTIONS_DIR.glob("2026_R18_2026*.csv"))[0]).set_index("code")
cmp = out.set_index("code")[["predicted_position", "p_win", "p_podium", "p_points"]].join(
    official[["predicted_position", "p_win", "p_podium"]], rsuffix="_official")
print(f"\nsaved {path.name} | git {out['git_commit'].iloc[0]}\n")
print(cmp.sort_values("predicted_position").head(10).round(3).to_string())
