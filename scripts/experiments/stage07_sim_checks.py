"""Stage 7 checks on the simulation, on 2023-24 dev races only.

    python scripts/experiments/stage07_sim_checks.py
  1. A vs B margin compared with Monte Carlo noise (seeds) and race-sampling noise (bootstrap)
  2. Does feeding the models' actual score gaps ("score" spacing) beat rank-based log spacing?
"""
import json

import numpy as np
import pandas as pd

from f1pred.backtest import race_key, walk_forward
from f1pred.config import PROCESSED_DIR, PROJECT_ROOT
from f1pred.models import (FINAL_FEATURES, FINISHER_TARGET, TARGET, LGBMRegressorModel, MeanScore, RidgeModel,
                           add_finisher_target, final_model)
from f1pred.simulate import SimSettings, add_dnf_probability, probability_scores, simulate_many

DEV = [2023, 2024]
params = json.loads((PROJECT_ROOT / "reports" / "stage05_lgbm_params.json").read_text())["params"]
table = add_dnf_probability(add_finisher_target(pd.read_parquet(PROCESSED_DIR / "features.parquet")))


def preds(make, target):
    return walk_forward(table, make, FINAL_FEATURES, DEV, target=target)


def per_race_brier(sim):
    e = ((sim["p_win"] - (sim["position"] == 1)) ** 2 + (sim["p_podium"] - (sim["position"] <= 3)) ** 2
         + (sim["p_points"] - (sim["position"] <= 10)) ** 2)
    return e.groupby(race_key(sim)).mean()


# ---------------------------------------------------------------- 1. A vs B, with noise
A = (preds(lambda: final_model(target=TARGET), TARGET), SimSettings(sigma=0.16, spacing="log"))
B = (preds(lambda: final_model(target=FINISHER_TARGET), FINISHER_TARGET), SimSettings(sigma=0.16, spacing="linear"))
print("1. A (official-position score) vs B (finishers-only score), each at its best dev settings")
runs = {}
for name, (p, st) in {"A": A, "B": B}.items():
    sims = [simulate_many(p, SimSettings(st.grid_blend, st.sigma, st.spacing, n_sims=4000, seed=s)) for s in range(5)]
    totals = [probability_scores(s)["brier_total"] for s in sims]
    runs[name] = sims[0]
    print(f"  {name}: brier_total {np.mean(totals):.4f}  (sd across 5 seeds {np.std(totals):.4f})")
diff = per_race_brier(runs["A"]) - per_race_brier(runs["B"])
rng = np.random.default_rng(0)
boot = [rng.choice(diff.values, len(diff)).mean() for _ in range(10000)]
print(f"  A - B per race: {diff.mean():+.4f}, 95% CI over races [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")

# ---------------------------------------------------------------- 2. score spacing
print("\n2. Score spacing (the models' own predicted gaps) vs rank-based log spacing")
cands = {
    "A ridge raw score": preds(lambda: RidgeModel(target=TARGET), TARGET),
    "A ridge+lightgbm mean score": preds(lambda: MeanScore([RidgeModel(target=TARGET),
                                                            LGBMRegressorModel(residual=True, target=TARGET, **params)]), TARGET),
    "B ridge+lightgbm mean score": preds(lambda: MeanScore([RidgeModel(target=FINISHER_TARGET),
                                                            LGBMRegressorModel(residual=True, target=FINISHER_TARGET, **params)]), FINISHER_TARGET),
}
print(f"  reference v2 (A, log ranks, sigma 0.16): brier_total {probability_scores(runs['A'])['brier_total']:.4f}")
for name, p in cands.items():
    best = min(((probability_scores(simulate_many(p, SimSettings(sigma=sg, spacing="score", n_sims=4000)))["brier_total"], sg)
                for sg in (0.03, 0.05, 0.07, 0.09, 0.12, 0.16)))
    print(f"  {name:<30} best brier_total {best[0]:.4f} at sigma {best[1]}")

# ---------------------------------------------------------------- 3. is the score-spacing gain beyond race noise?
print("\n3. B with score gaps vs v2, paired over races; plus grid blend")
b_score = cands["B ridge+lightgbm mean score"]
for blend in (0.0, 0.25):
    sim = simulate_many(b_score, SimSettings(grid_blend=blend, sigma=0.12, spacing="score", n_sims=4000))
    d = per_race_brier(sim) - per_race_brier(runs["A"])
    boot = [rng.choice(d.values, len(d)).mean() for _ in range(10000)]
    s = probability_scores(sim)
    print(f"  blend {blend}: brier_total {s['brier_total']:.4f}  winner log loss {s['winner_log_loss']:.3f}  "
          f"vs v2 per race {d.mean():+.4f} [{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]  "
          f"better in {(d < 0).sum()}/{len(d)} races")
