"""Stage 7: where is the model right, where is it wrong, and are its probabilities calibrated?

    python scripts/stage07_evaluate.py
Order model: the clean 2025-26 walk-forward test predictions from Stage 5.
Simulation v3: 2025-26 (designed after the v1 test, so contaminated, but not tuned there).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1pred.backtest import TEST_SEASONS, walk_forward
from f1pred.config import PROCESSED_DIR, PROJECT_ROOT
from f1pred.models import FINAL_FEATURES, add_finisher_target, final_sim_model
from f1pred.simulate import SimSettings, add_dnf_probability, probability_scores, simulate_many, temper_win
from f1pred.validate import predicted_rank, score_many

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
BLUE, GREY, INK, MUTED, GRIDLINE = "#2a78d6", "#b9b8b1", "#0b0b0b", "#898781", "#e1e0d9"
pd.set_option("display.width", 170)
plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDLINE, "grid.linewidth": 0.6,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "font.size": 9})

# ---------------------------------------------------------------- 1. order model: error by slice
p = pd.read_parquet(PROCESSED_DIR / "stage05_test_predictions.parquet")
p["pred_rank"] = pd.concat([predicted_rank(r, "pred") for _, r in p.groupby(["season", "round"])])
p["grid_rank"] = pd.concat([predicted_rank(r.assign(g=r["grid"]), "g") for _, r in p.groupby(["season", "round"])])
p["err_model"] = (p["pred_rank"] - p["position"]).abs()
p["err_grid"] = (p["grid_rank"] - p["position"]).abs()
p["dnf"] = ~p["finished"].astype(bool) & (p["status_bucket"] != "dsq")

slices = {
    "all drivers": p.index == p.index,
    "started P1-3": p["grid"] <= 3, "started P4-10": p["grid"].between(4, 10),
    "started P11-16": p["grid"].between(11, 16), "started P17+": p["grid"] >= 17,
    "finished": ~p["dnf"], "DNF": p["dnf"],
    "grid penalty 5+": p["grid_penalty"] >= 5, "no penalty": p["grid_penalty"] == 0,
    "rookie (first 3 races)": p["rookie"].astype(bool),
}
rows = [{"slice": k, "rows": int(m.sum()), "model_mae": p.loc[m, "err_model"].mean(),
         "grid_mae": p.loc[m, "err_grid"].mean()} for k, m in slices.items()]
sl = pd.DataFrame(rows).assign(gain=lambda d: d["grid_mae"] - d["model_mae"])
print("1. Order model, clean 2025-26 test: error by slice (gain > 0 = model beats the grid)")
print(sl.round(2).to_string(index=False), "\n")

by_driver = p.groupby("driver_code").agg(races=("err_model", "size"), model_mae=("err_model", "mean"),
                                         grid_mae=("err_grid", "mean"), dnfs=("dnf", "sum"))
by_driver = by_driver[by_driver["races"] >= 10].assign(gain=lambda d: d["grid_mae"] - d["model_mae"])
print("Drivers the model gets most wrong (10+ test races):")
print(by_driver.nlargest(5, "model_mae").round(2).to_string())
print("Drivers where the model helps most vs the grid:")
print(by_driver.nlargest(5, "gain").round(2).to_string(), "\n")

races = score_many(p, "pred").merge(
    p.groupby(["season", "round"]).agg(race=("race_name", "first"), dnfs=("dnf", "sum"),
                                       grid_mae=("err_grid", "mean")).reset_index(), on=["season", "round"])
surprise = p.assign(miss=p["position"] - p["pred_rank"]).loc[lambda d: d.groupby(["season", "round"])["miss"]
                                                             .transform(lambda s: s.abs().idxmax()) == d.index]
races = races.merge(surprise[["season", "round", "driver_code", "pred_rank", "position"]], on=["season", "round"])
races["biggest_miss"] = races["driver_code"] + " pred P" + races["pred_rank"].astype(int).astype(str) + \
    " -> P" + races["position"].astype(int).astype(str)
cols = ["season", "round", "race", "mae", "grid_mae", "dnfs", "biggest_miss"]
print("Worst 6 races for the model:")
print(races.nlargest(6, "mae")[cols].round(2).to_string(index=False))
print("Best 4 races:")
print(races.nsmallest(4, "mae")[cols].round(2).to_string(index=False))
print(f"\nDNFs per race vs model MAE, correlation {races['dnfs'].corr(races['mae']):.2f}\n")

# ---------------------------------------------------------------- 2. simulation v3: calibration
settings = json.loads((PROJECT_ROOT / "reports" / "sim_settings.json").read_text())
table = add_dnf_probability(add_finisher_target(pd.read_parquet(PROCESSED_DIR / "features.parquet")))
sim_preds = walk_forward(table, final_sim_model, FINAL_FEATURES, TEST_SEASONS, target=settings["target"])
sim = simulate_many(sim_preds, SimSettings(grid_blend=settings["grid_blend"], sigma=settings["sigma"],
                                           spacing=settings["spacing"]))
raw_scores = probability_scores(sim)
sim = temper_win(sim, settings["win_temperature"])
sim.to_parquet(PROCESSED_DIR / "stage07_sim_v3_2025_26.parquet", index=False)
print("2. Simulation v3 on 2025-26 (contaminated, not tuned there)")
print("  before win calibration:", {k: round(v, 4) for k, v in raw_scores.items()})
print(f"  after  (a={settings['win_temperature']}):  ", {k: round(v, 4) for k, v in probability_scores(sim).items()})

BINS = [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
print("calibration (predicted vs actual, by probability bin):")
for ax, (col, places, title) in zip(axes, [("p_win", 1, "Win"), ("p_podium", 3, "Podium"), ("p_points", 10, "Points")]):
    hit = sim["position"] <= places
    b = pd.cut(sim[col], BINS, include_lowest=True)
    cal = sim.assign(hit=hit).groupby(b, observed=True).agg(pred=(col, "mean"), actual=("hit", "mean"), n=("hit", "size"))
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1)
    ax.plot(cal["pred"], cal["actual"], color=BLUE, linewidth=2, marker="o", markersize=6)
    for _, r in cal[cal["pred"] >= 0.1].iterrows():  # label only where points don't pile up
        ax.annotate(f"n={int(r['n'])}", (r["pred"], r["actual"]), textcoords="offset points", xytext=(5, -10),
                    fontsize=7, color=MUTED)
    ax.set(title=f"{title}: predicted vs actual", xlabel="Predicted probability", ylabel="Actual frequency",
           xlim=(0, 1), ylim=(0, 1))
    print(f"  {title}: " + ", ".join(f"{r.pred:.2f}->{r.actual:.2f} (n={int(r.n)})" for r in cal.itertuples()))
fig.suptitle(f"Simulation v3 calibration, 2025-26, win probabilities tempered (a={settings['win_temperature']}); "
             "on the dashed line = well calibrated", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "stage07_calibration.png", dpi=130)

fig, ax = plt.subplots(figsize=(8, 4.5))
show = sl[sl["slice"] != "all drivers"].iloc[::-1]
ax.barh(show["slice"], show["gain"], color=[BLUE if g > 0 else GREY for g in show["gain"]], height=0.6)
ax.axvline(0, color=MUTED, linewidth=1)
ax.set(title="Where the model beats the grid (clean 2025-26 test)",
       xlabel="Places of error saved per driver vs 'finish = grid' (+ = model better)")
fig.tight_layout()
fig.savefig(FIG_DIR / "stage07_error_slices.png", dpi=130)
print(f"\nfigures written to {FIG_DIR}")
