"""Part 1: is the order model under- or overfitting? Learning curves, complexity curves, Optuna optimism,
noise floor, leak check, and a breakdown by season and situation.

    python scripts/diagnose.py      # prints results, writes reports/figures/diagnose_*.png, logs to reports/experiments.csv
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1pred import config, experiments
from f1pred.backtest import race_key, walk_forward
from f1pred.data import load_results, race_start, read_snapshot
from f1pred.features import build_features
from f1pred.models import GridBaseline, LGBMRegressorModel, RidgeModel, add_finisher_target, final_model
from f1pred.validate import score_many

pd.set_option("display.width", 170)
FIG = config.PROJECT_ROOT / "reports" / "figures"
BLUE, ORANGE, GREY, MUTED, GRIDLINE = "#2a78d6", "#d9822b", "#b9b8b1", "#898781", "#e1e0d9"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRIDLINE,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
                     "axes.titleweight": "bold", "axes.titlesize": 10})
ALL = [2023, 2024, 2025, 2026]
TUNED = json.loads(config.LGBM_PARAMS_FILE.read_text())["params"]

data = read_snapshot(config.SNAPSHOT)
history = load_results(as_of=race_start(data.schedule, config.TARGET_SEASON, config.TARGET_ROUND), data=data)
table = add_finisher_target(build_features(history.assign(is_target=False)))
table = table.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)
keys = race_key(table)
race_list = sorted(keys.unique())
F = config.FEATURES


def mae_on(model, rows) -> float:
    return float(score_many(rows.assign(pred=model.predict(rows, F)), "pred")["mae"].mean())


def wf(make, seasons=ALL, tbl=table, features=F) -> pd.DataFrame:
    return walk_forward(tbl, make, features, seasons)


def per_season(preds) -> pd.Series:
    return score_many(preds, "pred").groupby("season")["mae"].mean()


grid_wf = wf(GridBaseline)
grid_season = per_season(grid_wf)

# ---------------------------------------------------------------- 1. learning curves
print("1. LEARNING CURVES: train on the first N races, test on the next 9 (MAE, places)")
lc = []
for n in (20, 40, 60, 80, 100):
    tr = table[keys.isin(race_list[:n])]
    va = table[keys.isin(race_list[n:n + 9])]
    row = {"races": n, "grid (next 9)": float(score_many(va.assign(pred=va["grid"]), "pred")["mae"].mean())}
    for name, make in (("ridge", RidgeModel), ("lightgbm", lambda: LGBMRegressorModel(residual=True, **TUNED)),
                       ("ensemble", final_model)):
        m = make().fit(tr, F)
        row[f"{name} train"], row[f"{name} next 9"] = mae_on(m, tr), mae_on(m, va)
    lc.append(row)
lc = pd.DataFrame(lc).set_index("races")
print(lc.round(2).to_string(), "\n")

# ---------------------------------------------------------------- 2. complexity curves (train: races before 2024; test: walk-forward 2024)
print("2. COMPLEXITY CURVES: training error (fit on 2022-23) vs walk-forward error on 2024")
train_rows = table[table["season"] < 2024]
curves = {}


def complexity(label, values, make):
    out = []
    for v in values:
        m = make(v)
        out.append({"value": v, "train": mae_on(m().fit(train_rows, F), train_rows),
                    "walk-forward 2024": float(score_many(wf(m, [2024]), "pred")["mae"].mean())})
    curves[label] = pd.DataFrame(out).set_index("value")
    print(f"  {label}:\n" + curves[label].round(3).to_string().replace("\n", "\n    "))


complexity("ridge alpha (penalty)", [0.01, 1, 10, 100, 1000], lambda a: (lambda: RidgeModel(alpha=a)))
complexity("lightgbm num_leaves", [3, 9, 15, 31], lambda v: (lambda: LGBMRegressorModel(residual=True, **{**TUNED, "num_leaves": v})))
complexity("lightgbm n_estimators", [20, 52, 150, 400], lambda v: (lambda: LGBMRegressorModel(residual=True, **{**TUNED, "n_estimators": v})))
complexity("lightgbm min_child_samples", [20, 53, 150, 300], lambda v: (lambda: LGBMRegressorModel(residual=True, **{**TUNED, "min_child_samples": v})))
groups = {2: F[:2], 8: F[:8], 13: F[:13], 16: F}
out = []
for n, feats in groups.items():
    tr_m = RidgeModel().fit(train_rows, feats)
    va = walk_forward(table, RidgeModel, feats, [2024])
    out.append({"value": n, "train": float(score_many(train_rows.assign(pred=tr_m.predict(train_rows, feats)), "pred")["mae"].mean()),
                "walk-forward 2024": float(score_many(va, "pred")["mae"].mean())})
curves["number of features (ridge)"] = pd.DataFrame(out).set_index("value")
print("  number of features (ridge):\n" + curves["number of features (ridge)"].round(3).to_string().replace("\n", "\n    "), "\n")

# ---------------------------------------------------------------- 3. Optuna optimism
print("3. OPTUNA OPTIMISM: tuned LightGBM (tuned on 2024) vs untuned defaults, by season (walk-forward MAE)")
tuned = per_season(wf(lambda: LGBMRegressorModel(residual=True, **TUNED)))
default = per_season(wf(lambda: LGBMRegressorModel(residual=True)))
opt = pd.DataFrame({"defaults": default, "tuned": tuned, "gain": default - tuned, "grid": grid_season})
print(opt.round(3).to_string())
print(f"  gain on 2024 (tuned there): {opt.loc[2024, 'gain']:.3f} | average gain on other seasons: "
      f"{opt.drop(2024)['gain'].mean():.3f}\n")
experiments.log("diagnostic: Optuna optimism (tuned vs default lightgbm)", ALL,
                {"gain_2024": opt.loc[2024, "gain"], "gain_other": opt.drop(2024)["gain"].mean()}, None, "diagnostic")

# ---------------------------------------------------------------- 4. noise floor
print("4. NOISE FLOOR: the chosen ensemble with 5 different random seeds (walk-forward MAE)")
runs = {}
for seed in range(5):
    config.SEED = seed
    runs[seed] = wf(final_model)
config.SEED = 0
nf = pd.DataFrame({s: per_season(p) for s, p in runs.items()})
nf["all"] = nf.mean(axis=1)
overall = pd.Series({s: score_many(p, "pred")["mae"].mean() for s, p in runs.items()})
print(nf.round(3).to_string())
print(f"  overall MAE across seeds: {overall.mean():.3f} +/- {overall.std():.3f} (range {overall.min():.3f}-{overall.max():.3f})\n")
experiments.log("diagnostic: noise floor (5 seeds)", ALL, {"mae_mean": overall.mean(), "mae_sd": overall.std(),
                                                          "mae_range": overall.max() - overall.min()}, None, "diagnostic")

# ---------------------------------------------------------------- 5. leak check
print("5. LEAK CHECK: train on finishing positions shuffled within each race (nonsense labels)")
shuffled = table.copy()
rng = np.random.default_rng(0)
shuffled["position_pct"] = shuffled.groupby(["season", "round"])["position_pct"].transform(lambda s: rng.permutation(s.to_numpy()))
leak = score_many(wf(final_model, tbl=shuffled), "pred")["mae"].mean()
real = score_many(runs[0], "pred")["mae"].mean()
grid_all = score_many(grid_wf, "pred")["mae"].mean()
print(f"  shuffled labels {leak:.3f} | grid {grid_all:.3f} | real model {real:.3f} -> "
      + ("OK: no better than the grid" if leak >= grid_all else "WARNING: beats the grid on nonsense labels"), "\n")
experiments.log("diagnostic: leak check (shuffled labels)", ALL, {"mae_shuffled": leak, "mae_grid": grid_all, "mae_real": real},
                None, "pass" if leak >= grid_all else "FAIL")

# ---------------------------------------------------------------- 6. breakdown by season and situation
print("6. BREAKDOWN, walk-forward 2023-26 (seed 0): model vs grid MAE per driver")
p = runs[0].reset_index(drop=True)
p["pred_rank"] = p.groupby(["season", "round"])["pred"].rank(method="first")
p["err"] = (p["pred_rank"] - p["position"]).abs()
p["grid_err"] = (p["grid"].groupby([p["season"], p["round"]]).rank(method="first") - p["position"]).abs()
dnfs = p.assign(d=~p["finished"].astype(bool)).groupby(["season", "round"])["d"].transform("sum")
slices = {"all": pd.Series(True, index=p.index), "front (P1-3)": p["grid"] <= 3, "midfield (P4-12)": p["grid"].between(4, 12),
          "back (P13+)": p["grid"] >= 13, "penalised 5+": p["grid_penalty"] >= 5, "rookies": p["rookie"].astype(bool),
          "races with 4+ DNFs": dnfs >= 4}
rows = []
for season in ALL + ["all"]:
    q = p if season == "all" else p[p["season"] == season]
    for name, m in slices.items():
        m = m.loc[q.index]
        if m.sum():
            rows.append({"season": season, "slice": name, "rows": int(m.sum()),
                         "gain vs grid": q.loc[m, "grid_err"].mean() - q.loc[m, "err"].mean()})
bd = pd.DataFrame(rows).pivot(index="slice", columns="season", values="gain vs grid").reindex(list(slices))
print("  places saved per driver vs the grid (+ = model better):")
print(bd.round(2).to_string(), "\n")

# ---------------------------------------------------------------- figures
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name in zip(axes, ("ridge", "lightgbm", "ensemble")):
    ax.plot(lc.index, lc[f"{name} train"], color=BLUE, marker="o", label="training error")
    ax.plot(lc.index, lc[f"{name} next 9"], color=ORANGE, marker="o", label="error on the next 9 races")
    ax.plot(lc.index, lc["grid (next 9)"], color=GREY, linestyle="--", label="grid baseline (next 9)")
    ax.set(title=f"Learning curve: {name}", xlabel="training races", ylabel="MAE (places)")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "diagnose_learning_curves.png", dpi=130)

fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.6))
for ax, (label, c) in zip(axes, curves.items()):
    x = range(len(c))
    ax.plot(x, c["train"], color=BLUE, marker="o", label="training error")
    ax.plot(x, c["walk-forward 2024"], color=ORANGE, marker="o", label="walk-forward 2024")
    ax.set_xticks(list(x), [str(v) for v in c.index])
    ax.set(title=label, ylabel="MAE (places)")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "diagnose_complexity.png", dpi=130)
print(f"figures: {FIG / 'diagnose_learning_curves.png'}, {FIG / 'diagnose_complexity.png'}")
